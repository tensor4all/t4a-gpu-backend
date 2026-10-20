//! Runtime-local modules, bound functions and checked kernel launches.
//!
//! A bound function belongs to exactly one runtime/domain. Launching arbitrary
//! consumer kernels is `unsafe`: descriptor checks cannot prove the kernel's ABI,
//! its memory accesses or that it fully overwrites its outputs.

use std::ffi::CString;
use std::ptr;
use std::sync::Arc;

use cudarc::driver::result::DriverError;
use cudarc::driver::{result as drv, sys};

use crate::compile::{Artifact, CompileKey};
use crate::context::ContextHolder;
use crate::error::{Error, Result};
use crate::memory::{Buffer, Domain, UninitBuffer};

/// One loaded module, retained until every queued use has completed.
pub(crate) struct ModuleInner {
    holder: Arc<ContextHolder>,
    module: sys::CUmodule,
    bytes: usize,
    domain: std::sync::Weak<Domain>,
}

// SAFETY: a module handle is process-wide and context-scoped. Every use runs
// inside the owning domain's context scope on the using thread.
unsafe impl Send for ModuleInner {}
unsafe impl Sync for ModuleInner {}

impl ModuleInner {
    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }
}

impl Drop for ModuleInner {
    fn drop(&mut self) {
        if let Some(domain) = self.domain.upgrade() {
            domain.retirement().release_module_bytes(self.bytes);
        }
        // SAFETY: this is the last reference and every queued use has completed.
        // If the context cannot be activated, leak the module instead of acting
        // on a handle from an inactive context.
        if let Ok(_scope) = self.holder.activate() {
            unsafe {
                let _ = drv::module::unload(self.module);
            }
        }
    }
}

/// A function bound to one runtime.
pub struct BoundFunction {
    domain: Arc<Domain>,
    module: Arc<ModuleInner>,
    function: sys::CUfunction,
    entry: String,
    artifact_key: CompileKey,
}

// SAFETY: the handle carries no submission authority; launching needs
// `&mut Runtime` and is `unsafe`.
unsafe impl Send for BoundFunction {}
unsafe impl Sync for BoundFunction {}

impl BoundFunction {
    /// Entry point name.
    pub fn entry(&self) -> &str {
        &self.entry
    }

    /// Allocation-domain identity this function belongs to.
    pub fn domain_id(&self) -> u64 {
        self.domain.id()
    }

    /// Artifact this function was bound from.
    pub fn artifact_key(&self) -> CompileKey {
        self.artifact_key
    }
}

impl std::fmt::Debug for BoundFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BoundFunction")
            .field("entry", &self.entry)
            .field("domain_id", &self.domain.id())
            .finish()
    }
}

/// Checked launch dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaunchConfig {
    /// Grid dimensions; each must be non-zero.
    pub grid: (u32, u32, u32),
    /// Block dimensions; each must be non-zero.
    pub block: (u32, u32, u32),
    /// Dynamic shared memory in bytes.
    pub shared_bytes: u32,
}

impl LaunchConfig {
    /// Build a one-dimensional launch.
    pub fn new(grid: u32, block: u32) -> Self {
        Self {
            grid: (grid, 1, 1),
            block: (block, 1, 1),
            shared_bytes: 0,
        }
    }

    fn validate(&self) -> Result<()> {
        if self.grid.0 == 0 || self.grid.1 == 0 || self.grid.2 == 0 {
            return Err(Error::invalid("grid dimensions must be non-zero"));
        }
        if self.block.0 == 0 || self.block.1 == 0 || self.block.2 == 0 {
            return Err(Error::invalid("block dimensions must be non-zero"));
        }
        Ok(())
    }
}

/// A scalar or device-pointer kernel argument.
#[derive(Debug, Clone, Copy)]
pub enum Arg<'a> {
    /// 32-bit unsigned scalar.
    U32(u32),
    /// 64-bit unsigned scalar.
    U64(u64),
    /// 32-bit signed scalar.
    I32(i32),
    /// 32-bit float scalar.
    F32(f32),
    /// 64-bit float scalar.
    F64(f64),
    /// Device pointer of a buffer in the launching runtime's domain.
    DevicePtr(&'a Buffer),
    /// Device pointer of a buffer whose contents are not yet initialised.
    ///
    /// The kernel must fully overwrite it before it is converted with
    /// [`UninitBuffer::assume_init`].
    DevicePtrUninit(&'a UninitBuffer),
}

/// A bounded, runtime-owned module cache.
#[derive(Default)]
pub(crate) struct ModuleCache {
    entries: Vec<ModuleEntry>,
    bytes: usize,
    max_entries: usize,
    max_bytes: usize,
    loads: u64,
}

struct ModuleEntry {
    artifact: Arc<Artifact>,
    module: Arc<ModuleInner>,
    bytes: usize,
}

impl ModuleCache {
    pub(crate) fn new(max_entries: usize, max_bytes: usize) -> Self {
        Self {
            entries: Vec::new(),
            bytes: 0,
            max_entries,
            max_bytes,
            loads: 0,
        }
    }

    pub(crate) fn lookup(&mut self, artifact: &Artifact) -> Option<Arc<ModuleInner>> {
        self.entries
            .iter()
            .find(|candidate| *candidate.artifact == *artifact)
            .map(|candidate| Arc::clone(&candidate.module))
    }

    /// Retain a loaded module, evicting by the configured entry/byte bounds.
    ///
    /// The cache owns the module, so a bound function can be re-bound without
    /// reloading; eviction does not unload a module still held by a
    /// `BoundFunction` or a queued launch.
    pub(crate) fn insert(&mut self, artifact: &Arc<Artifact>, module: &Arc<ModuleInner>) {
        // Count the real load even when the entry is not retained.
        self.loads += 1;
        if self.max_entries == 0 || module.bytes() > self.max_bytes {
            return;
        }
        while !self.entries.is_empty()
            && (self.entries.len() >= self.max_entries
                || self.bytes + module.bytes() > self.max_bytes)
        {
            let evicted = self.entries.remove(0);
            self.bytes = self.bytes.saturating_sub(evicted.bytes);
        }
        self.bytes += module.bytes();
        self.entries.push(ModuleEntry {
            artifact: Arc::clone(artifact),
            module: Arc::clone(module),
            bytes: module.bytes(),
        });
    }

    pub(crate) fn entries(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }

    pub(crate) fn loads(&self) -> u64 {
        self.loads
    }

    /// Drop cached module references. Bound functions and queued launches keep
    /// their own references, so this never unloads code still in use.
    pub(crate) fn clear(&mut self) {
        self.entries.clear();
        self.bytes = 0;
    }
}

pub(crate) fn bind(
    domain: &Arc<Domain>,
    cache: &mut ModuleCache,
    artifact: &Arc<Artifact>,
    entry: &str,
) -> Result<BoundFunction> {
    if entry.is_empty() {
        return Err(Error::invalid("entry point name must not be empty"));
    }
    let module = match cache.lookup(artifact) {
        Some(module) => module,
        None => {
            let image_bytes = artifact.image().len();
            let retirement = domain.retirement();
            if retirement.module_bytes() + image_bytes > domain.pending_budget() {
                return Err(Error::Backpressure { bytes: image_bytes });
            }
            let _scope = domain.scope()?;
            let image = artifact.image();
            if image.is_empty() {
                return Err(Error::invalid("artifact image is empty"));
            }
            let mut terminated = Vec::with_capacity(image.len() + 1);
            terminated.extend_from_slice(image);
            terminated.push(0);
            // SAFETY: the image is a NUL-terminated PTX/CUBIN buffer that outlives
            // the call; the domain context is active.
            let module = unsafe { drv::module::load_data(terminated.as_ptr().cast()) }
                .map_err(Error::driver("cuModuleLoadData"))?;
            let inner = Arc::new(ModuleInner {
                holder: Arc::clone(domain.holder()),
                module,
                bytes: image_bytes,
                domain: Arc::downgrade(domain),
            });
            domain.retirement().retain_module_bytes(image_bytes);
            cache.insert(artifact, &inner);
            inner
        }
    };
    let name = CString::new(entry).map_err(|_| Error::invalid("entry point contains NUL"))?;
    let _scope = domain.scope()?;
    // SAFETY: `module` is alive (owned by `module`) and the context is active.
    let function = match unsafe { drv::module::get_function(module.module, name) } {
        Ok(function) => function,
        Err(error) => {
            // SAFETY: the module is still owned here; unload at most once through
            // the Arc drop path by dropping our reference.
            drop(module);
            return Err(Error::from_driver("cuModuleGetFunction", error));
        }
    };
    Ok(BoundFunction {
        domain: Arc::clone(domain),
        module,
        function,
        entry: entry.to_string(),
        artifact_key: artifact.key(),
    })
}

/// Enqueue a checked kernel launch.
///
/// # Safety
///
/// The bound function's kernel ABI must match `args`, every device pointer must
/// address `arg` bytes actually used by the kernel, and any output the caller
/// treats as initialised must be fully overwritten by this launch.
pub(crate) unsafe fn launch(
    domain: &Arc<Domain>,
    function: &BoundFunction,
    config: LaunchConfig,
    args: &[Arg<'_>],
) -> Result<()> {
    if !Arc::ptr_eq(&function.domain, domain) {
        return Err(Error::ForeignDomain);
    }
    config.validate()?;
    if let Some(detail) = domain.broken() {
        return Err(Error::Broken { detail });
    }
    // Small launches marshal arguments without heap allocation.
    const MAX_ARGS: usize = 32;
    if args.len() > MAX_ARGS {
        return Err(Error::invalid("a launch accepts at most 32 arguments"));
    }
    let mut slots = [[0u64; 1]; MAX_ARGS];
    for (index, arg) in args.iter().enumerate() {
        let value = match arg {
            Arg::U32(value) => *value as u64,
            Arg::U64(value) => *value,
            Arg::I32(value) => *value as u32 as u64,
            Arg::F32(value) => value.to_bits() as u64,
            Arg::F64(value) => value.to_bits(),
            Arg::DevicePtr(buffer) => {
                if !Arc::ptr_eq(buffer.alloc().domain(), domain) {
                    return Err(Error::ForeignDomain);
                }
                if buffer.is_empty() {
                    return Err(Error::invalid("an empty buffer has no device address"));
                }
                buffer.device_ptr()
            }
            Arg::DevicePtrUninit(buffer) => {
                if !Arc::ptr_eq(buffer.alloc().domain(), domain) {
                    return Err(Error::ForeignDomain);
                }
                if buffer.is_empty() {
                    return Err(Error::invalid("an empty buffer has no device address"));
                }
                buffer.device_ptr()
            }
        };
        slots[index] = [value];
    }
    let _scope = domain.scope()?;
    domain.retirement().during_admission(|admission| {
        // Pin the module before the launch is enqueued so a dropped
        // `BoundFunction` cannot unload code that is still queued, and reserve
        // its record before the submission.
        admission.add_keepalive(
            domain.stream(),
            Box::new(Arc::clone(&function.module)),
            0,
            true,
        )?;
        let mut params: Vec<*mut std::ffi::c_void> = slots
            .iter_mut()
            .map(|slot| slot.as_mut_ptr().cast())
            .collect();
        // SAFETY: the caller guarantees the ABI and pointer validity; the domain
        // context is active and every checked descriptor was validated above.
        let status = unsafe {
            sys::cuLaunchKernel(
                function.function,
                config.grid.0,
                config.grid.1,
                config.grid.2,
                config.block.0,
                config.block.1,
                config.block.2,
                config.shared_bytes,
                domain.stream(),
                params.as_mut_ptr(),
                ptr::null_mut(),
            )
        };
        if status != sys::CUresult::CUDA_SUCCESS {
            return Err(Error::from_driver("cuLaunchKernel", DriverError(status)));
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests;
