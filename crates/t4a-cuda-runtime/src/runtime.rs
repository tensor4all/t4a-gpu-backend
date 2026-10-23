//! The execution owner: one CUDA stream, one explicit pool, one allocation
//! domain, private caches, and the admission rules for numerical work.
//!
//! Submission takes `&mut Runtime` on the calling thread. There is no worker
//! thread, no ambient current-stream dispatch and no process-global registry.

use std::sync::Arc;

use cudarc::driver::result as drv;
use cudarc::driver::sys;

use crate::compile::{Artifact, ArtifactCache, CompileRequest};
use crate::context::ContextHolder;
use crate::error::{Error, Result};
use crate::kernel::{self, Arg, BoundFunction, LaunchConfig, ModuleCache};
use crate::memory::{Alloc, Buffer, Config, Domain, Stats, UninitBuffer};
use crate::raw::{self, RawResource, RawScope};
use crate::transfer::{self, Completion, CopyRoute, PendingDownload, PinnedHostBuffer};

/// A CUDA execution owner.
///
/// One runtime owns one real, non-default, non-blocking stream, one explicitly
/// created memory pool and one allocation-domain identity. Opening the same
/// ordinal twice yields two independent runtimes, even when they share the
/// device's primary context.
pub struct Runtime {
    ordinal: usize,
    config: Config,
    domain: Option<Arc<Domain>>,
    artifacts: ArtifactCache,
    modules: ModuleCache,
    // `Runtime` is `Send + !Sync`: numerical work is admitted through `&mut self`
    // on one thread at a time.
    _not_sync: std::marker::PhantomData<std::cell::Cell<()>>,
}

impl Runtime {
    /// Open `ordinal` and create its stream, pool and caches.
    ///
    /// # Errors
    ///
    /// [`Error::DriverUnavailable`] when the driver cannot be loaded,
    /// [`Error::InvalidDevice`] for an unknown ordinal, [`Error::Unsupported`]
    /// when the device lacks memory pools or a required driver symbol, and
    /// [`Error::Driver`] for a failing driver call.
    pub fn open(ordinal: usize, config: Config) -> Result<Self> {
        config.validate()?;
        let holder = ContextHolder::open(ordinal)?;
        let domain = Domain::create(holder, &config)?;
        Ok(Self {
            ordinal,
            artifacts: ArtifactCache::new(
                config.artifact_cache_entries,
                config.artifact_cache_bytes,
            ),
            modules: ModuleCache::new(config.module_cache_entries, config.module_cache_bytes),
            config,
            domain: Some(domain),
            _not_sync: std::marker::PhantomData,
        })
    }

    /// Ordinal this runtime was opened for.
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    /// Allocation-domain identity, distinct for every opened runtime.
    ///
    /// # Errors
    ///
    /// [`Error::Closed`] after [`Runtime::close`].
    pub fn domain_id(&self) -> Result<u64> {
        Ok(self.domain()?.id())
    }

    /// Opaque identity of this runtime's real CUDA stream.
    ///
    /// Two runtimes opened on the same ordinal report different values. The value
    /// is a raw handle and is only meaningful to unsafe consumers.
    ///
    /// # Errors
    ///
    /// [`Error::Closed`] after [`Runtime::close`].
    pub fn stream_id(&self) -> Result<u64> {
        Ok(self.domain()?.stream() as usize as u64)
    }

    /// The configuration this runtime was opened with.
    pub fn config(&self) -> &Config {
        &self.config
    }

    fn domain(&self) -> Result<&Arc<Domain>> {
        self.domain.as_ref().ok_or(Error::Closed)
    }

    /// Allocate device memory whose contents are not initialised.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] for a non-power-of-two alignment,
    /// [`Error::OutOfMemory`] when the pool cannot satisfy the request, and
    /// [`Error::Broken`] after an earlier unrecoverable failure.
    pub fn alloc_uninit(&mut self, bytes: usize, alignment: usize) -> Result<UninitBuffer> {
        let domain = self.domain()?.clone();
        validate_alignment(alignment)?;
        let ptr = domain.allocate(bytes, alignment)?;
        Ok(UninitBuffer::from_parts(
            Alloc::new(ptr, bytes, domain),
            bytes,
            alignment,
        ))
    }

    /// Allocate device memory and zero it in stream order.
    ///
    /// # Errors
    ///
    /// As [`Runtime::alloc_uninit`], plus [`Error::Driver`] when the zero fill
    /// cannot be enqueued.
    pub fn alloc_zeroed(&mut self, bytes: usize, alignment: usize) -> Result<Buffer> {
        let domain = self.domain()?.clone();
        validate_alignment(alignment)?;
        let ptr = domain.allocate(bytes, alignment)?;
        let alloc = Alloc::new(ptr, bytes, domain);
        let buffer = Buffer::from_parts(alloc, bytes, alignment);
        if bytes > 0 {
            let _scope = buffer.alloc().domain().scope()?;
            // SAFETY: `ptr` addresses `bytes` writable bytes in this domain.
            unsafe {
                drv::memset_d8_async(
                    buffer.device_ptr(),
                    0,
                    bytes,
                    buffer.alloc().domain().stream(),
                )
            }
            .map_err(Error::driver("cuMemsetD8Async"))?;
            buffer.alloc().domain().retirement().note_zero_fill();
        }
        Ok(buffer)
    }

    /// Allocate a caller-owned pinned host buffer.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] when pinned memory cannot be allocated, and
    /// [`Error::Driver`] for a failing driver call.
    pub fn alloc_pinned(&mut self, bytes: usize) -> Result<PinnedHostBuffer> {
        let domain = self.domain()?.clone();
        PinnedHostBuffer::new(domain.holder(), bytes)
    }

    /// Upload `data` into a new device buffer.
    ///
    /// The bytes are copied into runtime-owned staging before the DMA is queued,
    /// so no queued transfer reads the caller's slice after this returns.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`], [`Error::Driver`], or [`Error::Broken`].
    pub fn upload(&mut self, data: &[u8]) -> Result<Buffer> {
        let domain = self.domain()?.clone();
        transfer::upload(&domain, data)
    }

    /// Upload a consumed pinned allocation.
    ///
    /// # Errors
    ///
    /// As [`Runtime::upload`].
    pub fn upload_owned(&mut self, pinned: PinnedHostBuffer) -> Result<Buffer> {
        let domain = self.domain()?.clone();
        transfer::upload_owned(&domain, pinned)
    }

    /// Queue a device-to-host download.
    ///
    /// # Errors
    ///
    /// [`Error::ForeignDomain`] when the buffer belongs to another runtime,
    /// [`Error::Driver`], [`Error::OutOfMemory`] or [`Error::Broken`].
    pub fn download(&mut self, buffer: &Buffer) -> Result<PendingDownload> {
        let domain = self.domain()?.clone();
        transfer::download(&domain, buffer)
    }

    /// Copy `source` into this runtime from `source_owner`.
    ///
    /// Both runtimes participate in admission. `source` may be dropped as soon as
    /// this returns; its pending read is ordered before any later free.
    ///
    /// # Errors
    ///
    /// [`Error::ForeignDomain`] when `source` belongs to neither endpoint,
    /// [`Error::InvalidArgument`] for identical endpoints, and transfer failures.
    pub fn copy_between(
        &mut self,
        source_owner: &mut Runtime,
        source: &Buffer,
    ) -> Result<(Buffer, Completion, CopyRoute)> {
        let dst = self.domain()?.clone();
        let src = source_owner.domain()?.clone();
        transfer::copy_between(&src, &dst, source, self.config.force_staged_route)
    }

    /// Retire completed work and report observed asynchronous failures.
    ///
    /// # Errors
    ///
    /// [`Error::Async`] or [`Error::Broken`] when submitted work failed.
    pub fn observe(&mut self) -> Result<()> {
        self.domain()?.observe()
    }

    /// Wait for every queued operation on this runtime's stream.
    ///
    /// # Errors
    ///
    /// [`Error::Async`], [`Error::Broken`] or [`Error::Driver`].
    pub fn synchronize(&mut self) -> Result<()> {
        self.domain()?.synchronize()
    }

    /// Drop this runtime's caches and release its hold on the domain.
    ///
    /// Buffers, bound functions and pending records keep the CUDA stream, pool and
    /// context alive until the last of them is dropped; `close` never destroys
    /// resources that are still referenced.
    ///
    /// # Errors
    ///
    /// [`Error::Async`] or [`Error::Broken`] when queued work failed. The runtime
    /// is closed even when an error is reported.
    pub fn close(&mut self) -> Result<()> {
        let Some(domain) = self.domain.clone() else {
            return Ok(());
        };
        let result = domain.synchronize();
        self.artifacts.clear();
        self.modules.clear();
        self.domain = None;
        result
    }

    /// Resource counters for this runtime.
    pub fn stats(&self) -> Stats {
        let mut stats = match &self.domain {
            Some(domain) => domain.stats(),
            None => Stats {
                ordinal: self.ordinal,
                ..Stats::default()
            },
        };
        stats.artifact_entries = self.artifacts.len();
        stats.artifact_bytes = self.artifacts.bytes();
        stats.module_entries = self.modules.entries();
        stats.module_bytes = self.modules.bytes();
        stats
    }

    /// Artifact-cache hit and miss counters.
    pub fn artifact_cache_hits(&self) -> u64 {
        self.artifacts.hits()
    }

    /// Artifact-cache miss counters; each miss attempted a compile.
    pub fn artifact_cache_misses(&self) -> u64 {
        self.artifacts.misses()
    }

    /// Actual NVRTC compilation attempts made for this runtime.
    pub fn compile_attempts(&self) -> u64 {
        self.artifacts.compile_attempts()
    }

    /// Actual NVRTC compilations that succeeded for this runtime.
    pub fn compile_successes(&self) -> u64 {
        self.artifacts.compile_successes()
    }

    /// Number of module loads performed by this runtime.
    pub fn module_loads(&self) -> u64 {
        self.modules.loads()
    }

    /// Release pool memory above `keep_bytes` back to the driver.
    ///
    /// # Errors
    ///
    /// [`Error::Driver`] or [`Error::Closed`].
    pub fn trim_pool(&mut self, keep_bytes: usize) -> Result<()> {
        self.domain()?.trim_pool(keep_bytes)
    }

    /// Drop every cached module lookup. Bound functions keep their module alive.
    pub fn clear_modules(&mut self) {
        self.modules.clear();
    }

    /// Compile `request` unless an identical artifact is already cached.
    ///
    /// # Errors
    ///
    /// [`Error::NvrtcUnavailable`], [`Error::Compile`], or
    /// [`Error::InvalidArgument`].
    pub fn compile(&mut self, request: &CompileRequest) -> Result<Arc<Artifact>> {
        self.artifacts.get_or_compile(request)
    }

    /// Bind `entry` from `artifact` for this runtime.
    ///
    /// Each runtime loads its own module, even when two runtimes share an ordinal.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownEntryPoint`], [`Error::Driver`], [`Error::Closed`] or
    /// [`Error::Broken`].
    pub fn bind(&mut self, artifact: &Arc<Artifact>, entry: &str) -> Result<BoundFunction> {
        let domain = self.domain()?.clone();
        match kernel::bind(&domain, &mut self.modules, artifact, entry) {
            Ok(function) => Ok(function),
            Err(Error::Driver { source, .. }) => {
                // cuModuleGetFunction reports a missing entry point through the
                // driver; keep the original cause but expose the name.
                if source.0 == sys::CUresult::CUDA_ERROR_NOT_FOUND {
                    Err(Error::UnknownEntryPoint {
                        name: entry.to_string(),
                    })
                } else {
                    Err(Error::Driver {
                        context: "cuModuleGetFunction",
                        source,
                    })
                }
            }
            Err(other) => Err(other),
        }
    }

    /// Kernel parameter bytes this device accepts in one launch.
    ///
    /// The driver exposes no attribute for the limit, so it follows the device's
    /// major compute capability: 4 KiB below 7.0 and 32,764 bytes from 7.0 on.
    /// Large parameters also need a compiler and driver that support them, which
    /// this value does not promise; exceeding them fails in `cuLaunchKernel`.
    ///
    /// # Errors
    ///
    /// [`Error::Closed`] after [`Runtime::close`].
    pub fn max_kernel_parameter_bytes(&self) -> Result<usize> {
        Ok(self.domain()?.max_kernel_parameter_bytes())
    }

    /// Enqueue a checked kernel launch.
    ///
    /// # Safety
    ///
    /// The kernel ABI must match `args`, every device pointer must be valid for
    /// the accesses the kernel performs, and outputs the caller later treats as
    /// initialised must be fully overwritten.
    ///
    /// # Errors
    ///
    /// [`Error::ForeignDomain`] for a function or buffer from another runtime,
    /// [`Error::InvalidArgument`] for zero launch dimensions,
    /// [`Error::Driver`] for a failed launch, or [`Error::Broken`].
    pub unsafe fn launch(
        &mut self,
        function: &BoundFunction,
        config: LaunchConfig,
        args: &[Arg<'_>],
    ) -> Result<()> {
        let domain = self.domain()?.clone();
        // SAFETY: forwarded to the caller's contract.
        unsafe { kernel::launch(&domain, function, config, args) }
    }

    /// Run `closure` with the runtime's real stream.
    ///
    /// # Safety
    ///
    /// The closure must use this stream, obey each kernel's ABI, read and write
    /// only within registered buffers or pointers it owns, and keep foreign and
    /// vendor resources alive for as long as their queued work runs.
    ///
    /// # Errors
    ///
    /// [`Error::ForeignDomain`] for a buffer from another runtime,
    /// [`Error::Backpressure`] when the declared keepalives cannot be reserved,
    /// or [`Error::Broken`].
    pub unsafe fn with_raw<R>(
        &mut self,
        resources: Vec<RawResource<'_>>,
        closure: impl FnOnce(RawScope<'_>) -> R,
    ) -> Result<R> {
        let domain = self.domain()?.clone();
        // SAFETY: forwarded to the caller's contract.
        unsafe { raw::with_raw(&domain, resources, closure) }
    }
}

fn validate_alignment(alignment: usize) -> Result<()> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(Error::invalid("alignment must be a non-zero power of two"));
    }
    if alignment > 1 << 16 {
        return Err(Error::unsupported(
            "alignment above 64 KiB cannot be promised for pool allocations",
        ));
    }
    Ok(())
}

impl Drop for Runtime {
    fn drop(&mut self) {
        // Best effort: retire what completed, then release this runtime's hold.
        // Unfinished work keeps the domain alive through buffers and modules.
        if let Some(domain) = &self.domain {
            let _ = domain.synchronize();
        }
    }
}

#[cfg(test)]
mod tests;
