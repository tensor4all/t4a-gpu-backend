//! Explicit memory pools, allocation domains, device buffers and statistics.
//!
//! Each [`Domain`] owns one non-default non-blocking stream, one explicitly
//! created `CUmemPool` and one allocation-domain identity. Buffers are
//! allocated from that pool with `cuMemAllocFromPoolAsync` and freed with
//! `cuMemFreeAsync` on the same stream, which is already ordered after every
//! admitted use. No host-side lease or completion event is needed for them.

use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use cudarc::driver::result as drv;
use cudarc::driver::sys;

use crate::context::{ContextHolder, CtxScope};
use crate::error::{Error, Result};
use crate::retirement::{Limits, Retirement, RetirementStats};

static NEXT_DOMAIN_ID: AtomicU64 = AtomicU64::new(1);

/// Runtime configuration.
///
/// The defaults are reviewable starting points, not measured optima. Every
/// limit has a configure/trim/clear/stats path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    /// Maximum idle pinned staging bytes retained per runtime.
    pub max_idle_pinned_bytes: usize,
    /// Maximum size of one pinned staging chunk.
    pub max_chunk_bytes: usize,
    /// Budget for host/module resources that are held by unfinished work.
    pub pending_host_budget: usize,
    /// Maximum outstanding retirement records and witnesses.
    pub max_pending_records: usize,
    /// Maximum entries in the runtime's artifact cache.
    pub artifact_cache_entries: usize,
    /// Maximum estimated payload bytes in the artifact cache.
    pub artifact_cache_bytes: usize,
    /// Maximum entries in the runtime's module cache.
    pub module_cache_entries: usize,
    /// Maximum estimated payload bytes in the module cache.
    pub module_cache_bytes: usize,
    /// Pool memory retained above this size is released at the next
    /// synchronisation. This is a hint, not a hard cap.
    pub pool_release_threshold: usize,
    /// Test-only: force the staged cross-device route even when both runtimes
    /// share one ordinal, so the chunked staging protocol can be exercised on a
    /// single-GPU host. Never enable this in production.
    #[doc(hidden)]
    pub force_staged_route: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            max_idle_pinned_bytes: 64 << 20,
            max_chunk_bytes: 8 << 20,
            pending_host_budget: 256 << 20,
            max_pending_records: 1024,
            artifact_cache_entries: 64,
            artifact_cache_bytes: 256 << 20,
            module_cache_entries: 64,
            module_cache_bytes: 256 << 20,
            pool_release_threshold: 256 << 20,
            force_staged_route: false,
        }
    }
}

impl Config {
    /// Validate the configuration without touching CUDA.
    pub fn validate(&self) -> Result<()> {
        if self.max_chunk_bytes == 0 {
            return Err(Error::invalid("max_chunk_bytes must be non-zero"));
        }
        if self.max_idle_pinned_bytes < self.max_chunk_bytes {
            return Err(Error::invalid(
                "max_idle_pinned_bytes must be at least max_chunk_bytes",
            ));
        }
        if self.pending_host_budget < self.max_chunk_bytes {
            return Err(Error::invalid(
                "pending_host_budget must be at least max_chunk_bytes",
            ));
        }
        if self.max_pending_records == 0 {
            return Err(Error::invalid("max_pending_records must be non-zero"));
        }
        Ok(())
    }

    pub(crate) fn limits(&self) -> Limits {
        Limits {
            pending_bytes: self.pending_host_budget,
            pending_records: self.max_pending_records,
            idle_pinned_bytes: self.max_idle_pinned_bytes,
            max_chunk_bytes: self.max_chunk_bytes,
        }
    }
}

/// A snapshot of runtime-owned resources.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Stats {
    /// Device ordinal.
    pub ordinal: usize,
    /// Allocation-domain identity; distinct for every opened runtime.
    pub domain_id: u64,
    /// Identity of this runtime's explicit memory pool.
    pub pool_id: u64,
    /// Live device allocations.
    pub live_allocations: usize,
    /// Live device bytes.
    pub live_bytes: usize,
    /// Device memory currently used by the runtime pool.
    pub pool_used_bytes: u64,
    /// Device memory currently reserved by the runtime pool from the driver.
    pub pool_reserved_bytes: u64,
    /// Bytes held for unfinished work (staging, keepalives, witness records).
    pub pending_bytes: usize,
    /// Outstanding retirement records.
    pub pending_records: usize,
    /// Idle pinned staging bytes.
    pub idle_pinned_bytes: usize,
    /// Number of pending retirement detail records.
    pub retirement: RetirementStats,
    /// Artifact-cache entries.
    pub artifact_entries: usize,
    /// Estimated artifact-cache payload bytes.
    pub artifact_bytes: usize,
    /// Module-cache entries.
    pub module_entries: usize,
    /// Estimated module-cache payload bytes.
    pub module_bytes: usize,
}

struct DomainState {
    live_allocations: usize,
    live_bytes: usize,
    quarantined: Vec<sys::CUdeviceptr>,
    broken: Option<String>,
}

/// One execution domain: stream, pool and allocation identity.
///
/// `Domain` is shared by buffers and modules that may outlive the `Runtime` and
/// be dropped on another thread. Every CUDA call activates the owning context
/// first, so the handles are not thread-affine.
pub(crate) struct Domain {
    id: u64,
    holder: Arc<ContextHolder>,
    stream: sys::CUstream,
    pool: sys::CUmemoryPool,
    retirement: Option<Retirement>,
    state: Mutex<DomainState>,
}

// SAFETY: handles are context-scoped, never thread-affine, and every CUDA call
// activates the owning context on the calling thread. Mutable state is behind a
// `Mutex`.
unsafe impl Send for Domain {}
unsafe impl Sync for Domain {}

impl Domain {
    pub(crate) fn create(holder: Arc<ContextHolder>, config: &Config) -> Result<Arc<Self>> {
        let scope = holder.activate()?;
        let stream = drv::stream::create(drv::stream::StreamKind::NonBlocking)
            .map_err(Error::driver("cuStreamCreate"))?;
        let pool = match unsafe { create_pool(holder.ordinal()) } {
            Ok(pool) => pool,
            Err(error) => {
                // SAFETY: the stream is owned here and nothing was enqueued.
                unsafe {
                    let _ = drv::stream::destroy(stream);
                }
                return Err(error);
            }
        };
        let mut threshold = config.pool_release_threshold as u64;
        // SAFETY: `pool` was just created and the attribute expects a u64.
        if let Err(error) = unsafe {
            drv::mem_pool::set_attribute(
                pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                (&mut threshold) as *mut u64 as *mut _,
            )
        } {
            // SAFETY: tearing down a pool that has no allocations.
            unsafe {
                let _ = drv::mem_pool::destroy(pool);
                let _ = drv::stream::destroy(stream);
            }
            return Err(Error::from_driver("cuMemPoolSetAttribute", error));
        }
        drop(scope);
        Ok(Arc::new(Self {
            id: NEXT_DOMAIN_ID.fetch_add(1, Ordering::Relaxed),
            retirement: Some(Retirement::new(config.limits())),
            holder,
            stream,
            pool,
            state: Mutex::new(DomainState {
                live_allocations: 0,
                live_bytes: 0,
                quarantined: Vec::new(),
                broken: None,
            }),
        }))
    }

    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    pub(crate) fn ordinal(&self) -> usize {
        self.holder.ordinal()
    }

    pub(crate) fn holder(&self) -> &Arc<ContextHolder> {
        &self.holder
    }

    pub(crate) fn stream(&self) -> sys::CUstream {
        self.stream
    }

    pub(crate) fn retirement(&self) -> &Retirement {
        self.retirement
            .as_ref()
            .expect("retirement state is only taken during final teardown")
    }

    pub(crate) fn pending_budget(&self) -> usize {
        self.retirement().pending_budget()
    }

    pub(crate) fn scope(&self) -> Result<CtxScope> {
        self.holder.activate()
    }

    pub(crate) fn broken(&self) -> Option<String> {
        let state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        state.broken.clone().or_else(|| self.retirement().broken())
    }

    pub(crate) fn mark_broken(&self, detail: impl Into<String>) {
        let detail = detail.into();
        self.retirement().mark_broken(detail.clone());
        self.state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .broken
            .get_or_insert(detail);
    }

    /// Keep an allocation alive even though its foreign reader is unconfirmed.
    pub(crate) fn quarantine(&self, ptr: sys::CUdeviceptr) {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if !state.quarantined.contains(&ptr) {
            state.quarantined.push(ptr);
        }
    }

    /// Allocate `bytes` from the runtime pool in stream order.
    pub(crate) fn allocate(&self, bytes: usize, alignment: usize) -> Result<sys::CUdeviceptr> {
        if bytes == 0 {
            return Ok(0);
        }
        if let Some(detail) = self.broken() {
            return Err(Error::Broken { detail });
        }
        let _scope = self.scope()?;
        // SAFETY: pool and stream belong to this domain and are alive.
        let ptr = unsafe { drv::mem_pool::alloc_async(self.pool, bytes, self.stream) }
            .map_err(|e| Error::from_alloc(bytes, "cuMemAllocFromPoolAsync", e))?;
        if alignment > 1 && (ptr as usize) % alignment != 0 {
            // Never expose a misaligned buffer; return the memory to the pool.
            // SAFETY: the allocation was just created on this stream.
            unsafe {
                let _ = drv::free_async(ptr, self.stream);
            }
            return Err(Error::unsupported(format!(
                "pool allocation is {} but {alignment} alignment was requested",
                ptr % alignment as u64
            )));
        }
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        state.live_allocations += 1;
        state.live_bytes += bytes;
        Ok(ptr)
    }

    /// Enqueue the stream-ordered free for a buffer that is being dropped.
    pub(crate) fn release(&self, ptr: sys::CUdeviceptr, bytes: usize) {
        if ptr == 0 {
            return;
        }
        {
            let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
            if state.quarantined.contains(&ptr) {
                // An earlier failure could not prove that a foreign read ended.
                return;
            }
            state.live_allocations = state.live_allocations.saturating_sub(1);
            state.live_bytes = state.live_bytes.saturating_sub(bytes);
        }
        if self.broken().is_some() {
            self.quarantine(ptr);
            return;
        }
        let queued = self.scope().and_then(|_scope| {
            // SAFETY: this domain owns the allocation and the stream; the free is
            // ordered after every admitted use on this stream.
            unsafe { drv::free_async(ptr, self.stream) }
                .map_err(|e| Error::from_driver("cuMemFreeAsync", e))
        });
        if let Err(error) = queued {
            self.quarantine(ptr);
            self.mark_broken(error.to_string());
        }
    }

    /// Retire completed work and report observed asynchronous failures.
    pub(crate) fn observe(&self) -> Result<()> {
        let _scope = self.scope()?;
        self.retirement().observe(self.stream)
    }

    /// Observation that only reports readiness once upstream witnesses resolve.
    pub(crate) fn observe_gated(&self) -> Result<bool> {
        let _scope = self.scope()?;
        self.retirement().observe_gated(self.stream)
    }

    pub(crate) fn synchronize(&self) -> Result<()> {
        let _scope = self.scope()?;
        while !self.retirement().observe_gated(self.stream)? {
            std::thread::yield_now();
        }
        // SAFETY: the stream belongs to this domain.
        unsafe { drv::stream::synchronize(self.stream) }
            .map_err(Error::driver("cuStreamSynchronize"))
    }

    pub(crate) fn trim_pool(&self, keep_bytes: usize) -> Result<()> {
        let _scope = self.scope()?;
        // SAFETY: the pool belongs to this domain.
        unsafe { drv::mem_pool::trim_to(self.pool, keep_bytes) }
            .map_err(Error::driver("cuMemPoolTrimTo"))
    }

    pub(crate) fn stats(&self) -> Stats {
        let state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        let (used, reserved) = self.pool_attribute();
        Stats {
            ordinal: self.ordinal(),
            domain_id: self.id,
            pool_id: self.pool as usize as u64,
            live_allocations: state.live_allocations,
            live_bytes: state.live_bytes,
            pool_used_bytes: used,
            pool_reserved_bytes: reserved,
            pending_bytes: self.retirement().pending_bytes_hint(),
            pending_records: self.retirement().stats().pending_records,
            idle_pinned_bytes: self.retirement().stats().idle_pinned_bytes,
            retirement: self.retirement().stats(),
            artifact_entries: 0,
            artifact_bytes: 0,
            module_entries: 0,
            module_bytes: 0,
        }
    }

    fn pool_attribute(&self) -> (u64, u64) {
        let mut used = 0u64;
        let mut reserved = 0u64;
        let _scope = self.scope();
        if _scope.is_ok() {
            // SAFETY: the pool is alive and each attribute expects a cuuint64_t.
            unsafe {
                let _ = drv::mem_pool::get_attribute(
                    self.pool,
                    sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
                    (&mut used) as *mut u64 as *mut _,
                );
                let _ = drv::mem_pool::get_attribute(
                    self.pool,
                    sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
                    (&mut reserved) as *mut u64 as *mut _,
                );
            }
        }
        (used, reserved)
    }
}

impl Drop for Domain {
    fn drop(&mut self) {
        // SAFETY: this is the last reference to the domain. All buffers, modules
        // and witnesses are already gone, so a single stream synchronisation
        // completes every queued free.
        let drained = match self.scope() {
            Ok(_scope) => {
                let sync_ok = unsafe { drv::stream::synchronize(self.stream) }
                    .map_err(Error::driver("cuStreamSynchronize"))
                    .is_ok();
                let broken = self
                    .state
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .broken
                    .is_some();
                // A local synchronisation alone does not prove that foreign
                // witnesses resolved or that retirement is empty.
                let drained_retirement = self.retirement().observe(self.stream).is_ok()
                    && self.retirement().is_drained();
                if sync_ok && !broken && drained_retirement {
                    let quarantined = {
                        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
                        std::mem::take(&mut state.quarantined)
                    };
                    for ptr in quarantined {
                        // SAFETY: the stream completed, so nothing can still read it.
                        unsafe {
                            let _ = drv::free_sync(ptr);
                        }
                    }
                    true
                } else {
                    false
                }
            }
            Err(_) => false,
        };
        let _scope = self.scope();
        if !drained {
            // Completion could not be proved. Retain (leak) the whole resource
            // bundle rather than releasing pinned staging, modules or the
            // context underneath work that may still be running.
            std::mem::forget(Arc::clone(&self.holder));
            if let Some(retirement) = self.retirement.take() {
                std::mem::forget(retirement);
            }
            return;
        }
        // SAFETY: every allocation was freed (or deliberately quarantined), so
        // the pool and stream can be destroyed exactly once.
        unsafe {
            let _ = drv::mem_pool::destroy(self.pool);
            let _ = drv::stream::destroy(self.stream);
        }
    }
}

unsafe fn create_pool(ordinal: usize) -> Result<sys::CUmemoryPool> {
    let props = sys::CUmemPoolProps {
        allocType: sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED,
        handleTypes: sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_NONE,
        location: sys::CUmemLocation {
            type_: sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
            id: ordinal as i32,
        },
        win32SecurityAttributes: ptr::null_mut(),
        maxSize: 0,
        usage: 0,
        reserved: [0; 54],
    };
    // SAFETY: `props` is fully initialised for the bound CUDA version.
    // SAFETY: `props` is fully initialised for the bound CUDA version.
    unsafe { drv::mem_pool::create(&props) }.map_err(Error::driver("cuMemPoolCreate"))
}

/// The allocation behind a buffer, freed in stream order when dropped.
pub(crate) struct Alloc {
    ptr: sys::CUdeviceptr,
    bytes: usize,
    domain: Arc<Domain>,
}

impl Alloc {
    pub(crate) fn new(ptr: sys::CUdeviceptr, bytes: usize, domain: Arc<Domain>) -> Self {
        Self { ptr, bytes, domain }
    }

    pub(crate) fn ptr(&self) -> sys::CUdeviceptr {
        self.ptr
    }

    pub(crate) fn domain(&self) -> &Arc<Domain> {
        &self.domain
    }
}

impl Drop for Alloc {
    fn drop(&mut self) {
        self.domain.release(self.ptr, self.bytes);
    }
}

/// Check `offset + required <= len` with overflow detection.
pub(crate) fn check_span(len: usize, offset: usize, required: usize) -> Result<()> {
    match offset.checked_add(required) {
        Some(end) if end <= len => Ok(()),
        _ => Err(Error::OutOfRange {
            len,
            offset,
            required,
        }),
    }
}

/// A device buffer with initialised contents.
///
/// `Buffer` is `Send + Sync` but not `Clone`: share one allocation through
/// `Arc<Buffer>` instead of duplicating allocation authority.
pub struct Buffer {
    alloc: Alloc,
    len: usize,
    align: usize,
}

// SAFETY: `Buffer` carries no submission authority; every CUDA call on it needs
// `&mut Runtime` or an `unsafe` raw scope. A cross-thread last drop takes only
// the domain's short holder lock to enqueue `cuMemFreeAsync`.
unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Buffer")
            .field("len", &self.len)
            .field("alignment", &self.align)
            .field("domain_id", &self.domain_id())
            .finish()
    }
}

impl Buffer {
    pub(crate) fn from_parts(alloc: Alloc, len: usize, align: usize) -> Self {
        Self { alloc, len, align }
    }

    /// Length in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Requested alignment in bytes.
    pub fn alignment(&self) -> usize {
        self.align
    }

    /// Allocation-domain identity of the runtime that owns this buffer.
    pub fn domain_id(&self) -> u64 {
        self.alloc.domain().id()
    }

    pub(crate) fn device_ptr(&self) -> sys::CUdeviceptr {
        self.alloc.ptr()
    }

    pub(crate) fn alloc(&self) -> &Alloc {
        &self.alloc
    }
}

/// A device buffer whose contents are not initialised.
///
/// Safe code cannot read it. Convert it with [`UninitBuffer::assume_init`] only
/// after a full overwrite has been admitted on the owning stream.
pub struct UninitBuffer {
    alloc: Alloc,
    len: usize,
    align: usize,
}

// SAFETY: see `Buffer`.
unsafe impl Send for UninitBuffer {}
unsafe impl Sync for UninitBuffer {}

impl std::fmt::Debug for UninitBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UninitBuffer")
            .field("len", &self.len)
            .field("domain_id", &self.domain_id())
            .finish()
    }
}

impl UninitBuffer {
    pub(crate) fn from_parts(alloc: Alloc, len: usize, align: usize) -> Self {
        Self { alloc, len, align }
    }

    /// Length in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Allocation-domain identity of the runtime that owns this buffer.
    pub fn domain_id(&self) -> u64 {
        self.alloc.domain().id()
    }

    pub(crate) fn device_ptr(&self) -> sys::CUdeviceptr {
        self.alloc.ptr()
    }

    pub(crate) fn alloc(&self) -> &Alloc {
        &self.alloc
    }

    /// Declare the contents initialised.
    ///
    /// # Safety
    ///
    /// Every byte must have been completely written by work that was admitted on
    /// this buffer's stream, and that work must be ordered before any read.
    pub unsafe fn assume_init(self) -> Buffer {
        let this = std::mem::ManuallyDrop::new(self);
        // SAFETY: `this` is not dropped, so ownership of `alloc` moves exactly once.
        let alloc = unsafe { ptr::read(&this.alloc) };
        Buffer::from_parts(alloc, this.len, this.align)
    }
}

#[cfg(test)]
mod tests;
