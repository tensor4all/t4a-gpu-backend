//! The allocation domain (one stream, one explicit pool) and its buffers.

use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use t4a_hip_sys::{
    HipMemPool, HipStream, HipTable, MemPoolProps, HIP_MEM_POOL_ATTR_RESERVED_MEM_CURRENT,
    HIP_MEM_POOL_ATTR_USED_MEM_CURRENT, HIP_STREAM_NON_BLOCKING,
};

use crate::device::check;
use crate::error::Result;

static NEXT_DOMAIN: AtomicU64 = AtomicU64::new(1);

/// One stream, one explicitly created pool and the identity they share.
///
/// Every buffer holds the domain alive, so the stream and pool outlive every
/// allocation and every queued free. The domain is destroyed only after its last
/// buffer is gone: the stream is synchronised, the pool destroyed, then the stream.
pub(crate) struct Domain {
    hip: &'static HipTable,
    ordinal: i32,
    stream: HipStream,
    pool: HipMemPool,
    id: u64,
}

// SAFETY: the HIP runtime API is thread-safe. The stream and pool handles are plain
// values whose operations are ordered by the stream; every entry point activates the
// owning device first because HIP's current device is per thread.
unsafe impl Send for Domain {}
unsafe impl Sync for Domain {}

impl Domain {
    pub(crate) fn create(hip: &'static HipTable, ordinal: i32) -> Result<Arc<Domain>> {
        // SAFETY: `ordinal` was validated by the caller against the device count.
        check(hip, "hipSetDevice", unsafe { (hip.set_device)(ordinal) })?;
        let mut stream: HipStream = std::ptr::null_mut();
        // SAFETY: `stream` is writable; the flag is a plain integer.
        check(hip, "hipStreamCreateWithFlags", unsafe {
            (hip.stream_create_with_flags)(&mut stream, HIP_STREAM_NON_BLOCKING)
        })?;
        let props = MemPoolProps::device(ordinal);
        let mut pool: HipMemPool = std::ptr::null_mut();
        // SAFETY: `pool` is writable and `props` is a fully initialised structure.
        let created = check(hip, "hipMemPoolCreate", unsafe {
            (hip.mem_pool_create)(&mut pool, &props)
        });
        if let Err(error) = created {
            // SAFETY: the stream was created above and has no work.
            unsafe { (hip.stream_destroy)(stream) };
            return Err(error);
        }
        Ok(Arc::new(Domain {
            hip,
            ordinal,
            stream,
            pool,
            id: NEXT_DOMAIN.fetch_add(1, Ordering::Relaxed),
        }))
    }

    pub(crate) fn hip(&self) -> &'static HipTable {
        self.hip
    }

    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    pub(crate) fn ordinal(&self) -> i32 {
        self.ordinal
    }

    pub(crate) fn stream(&self) -> HipStream {
        self.stream
    }

    /// Make this domain's device current on the calling thread.
    pub(crate) fn activate(&self) -> Result<()> {
        // SAFETY: the ordinal was valid when the domain was created.
        check(self.hip, "hipSetDevice", unsafe {
            (self.hip.set_device)(self.ordinal)
        })
    }

    /// Stream-ordered allocation from this domain's own pool.
    pub(crate) fn allocate(&self, bytes: usize) -> Result<*mut c_void> {
        if bytes == 0 {
            return Ok(std::ptr::null_mut());
        }
        self.activate()?;
        let mut pointer: *mut c_void = std::ptr::null_mut();
        // SAFETY: `pointer` is writable, and the pool and stream belong to this domain.
        check(self.hip, "hipMallocFromPoolAsync", unsafe {
            (self.hip.malloc_from_pool_async)(&mut pointer, bytes, self.pool, self.stream)
        })?;
        Ok(pointer)
    }

    /// Stream-ordered free, queued after every use already on the stream.
    fn free(&self, pointer: *mut c_void) {
        if pointer.is_null() || self.activate().is_err() {
            return;
        }
        // SAFETY: `pointer` came from this domain's pool and is freed exactly once, on
        // the stream every use of it was ordered on.
        unsafe { (self.hip.free_async)(pointer, self.stream) };
    }

    pub(crate) fn synchronize(&self) -> Result<()> {
        self.activate()?;
        // SAFETY: the stream belongs to this domain and is alive.
        check(self.hip, "hipStreamSynchronize", unsafe {
            (self.hip.stream_synchronize)(self.stream)
        })
    }

    pub(crate) fn pool_attribute(&self, attribute: i32) -> Result<u64> {
        self.activate()?;
        let mut value = 0u64;
        // SAFETY: both attributes this crate reads are `uint64_t`.
        check(self.hip, "hipMemPoolGetAttribute", unsafe {
            (self.hip.mem_pool_get_attribute)(self.pool, attribute, (&mut value as *mut u64).cast())
        })?;
        Ok(value)
    }

    pub(crate) fn pool_stats(&self) -> Result<PoolStats> {
        Ok(PoolStats {
            used_bytes: self.pool_attribute(HIP_MEM_POOL_ATTR_USED_MEM_CURRENT)?,
            reserved_bytes: self.pool_attribute(HIP_MEM_POOL_ATTR_RESERVED_MEM_CURRENT)?,
        })
    }

    pub(crate) fn trim(&self, keep_bytes: usize) -> Result<()> {
        self.activate()?;
        // SAFETY: the pool belongs to this domain.
        check(self.hip, "hipMemPoolTrimTo", unsafe {
            (self.hip.mem_pool_trim_to)(self.pool, keep_bytes)
        })
    }
}

impl Drop for Domain {
    fn drop(&mut self) {
        // Best effort: no buffer is left (each holds the domain), so after the stream
        // drains every queued free has run and the pool is empty.
        if self.activate().is_err() {
            return;
        }
        // SAFETY: the handles belong to this domain and nothing else refers to them.
        unsafe {
            (self.hip.stream_synchronize)(self.stream);
            (self.hip.mem_pool_destroy)(self.pool);
            (self.hip.stream_destroy)(self.stream);
        }
    }
}

/// Memory held by one runtime's pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PoolStats {
    /// Bytes currently allocated to live buffers (`hipMemPoolAttrUsedMemCurrent`).
    pub used_bytes: u64,
    /// Bytes the pool holds from the device, used or cached
    /// (`hipMemPoolAttrReservedMemCurrent`).
    pub reserved_bytes: u64,
}

struct Allocation {
    pointer: *mut c_void,
    bytes: usize,
    domain: Arc<Domain>,
}

// SAFETY: the allocation is a device address plus its owning domain; the only
// operation on it is the stream-ordered free in `Drop`, which is thread-safe.
unsafe impl Send for Allocation {}
unsafe impl Sync for Allocation {}

impl Drop for Allocation {
    fn drop(&mut self) {
        self.domain.free(self.pointer);
    }
}

/// A device allocation from one runtime's pool.
///
/// Handles carry no submission authority. Clones share the allocation; the last drop
/// queues `hipFreeAsync` on the owning stream, which is already ordered after every
/// use submitted through that runtime, including vendor calls made in a raw scope.
#[derive(Clone)]
pub struct Buffer {
    allocation: Arc<Allocation>,
}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Buffer")
            .field("bytes", &self.allocation.bytes)
            .field("domain", &self.allocation.domain.id)
            .finish()
    }
}

impl Buffer {
    pub(crate) fn new(pointer: *mut c_void, bytes: usize, domain: Arc<Domain>) -> Buffer {
        Buffer {
            allocation: Arc::new(Allocation {
                pointer,
                bytes,
                domain,
            }),
        }
    }

    /// Size in bytes.
    pub fn len(&self) -> usize {
        self.allocation.bytes
    }

    /// Whether the buffer holds no bytes.
    pub fn is_empty(&self) -> bool {
        self.allocation.bytes == 0
    }

    /// Identity of the runtime this buffer belongs to.
    pub fn domain_id(&self) -> u64 {
        self.allocation.domain.id
    }

    pub(crate) fn pointer(&self) -> *mut c_void {
        self.allocation.pointer
    }

    pub(crate) fn domain(&self) -> &Arc<Domain> {
        &self.allocation.domain
    }
}
