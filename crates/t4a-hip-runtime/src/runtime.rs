//! The execution owner: one HIP stream, one explicit pool, one allocation domain.

use std::cell::RefCell;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::sync::Arc;

use t4a_hip_sys::{
    HipTable, PointerAttributes, HIP_MEMCPY_DEVICE_TO_HOST, HIP_MEMCPY_HOST_TO_DEVICE,
    HIP_MEMORY_TYPE_DEVICE, HIP_MEMORY_TYPE_HOST, HIP_MEMORY_TYPE_MANAGED,
    HIP_MEMORY_TYPE_UNREGISTERED,
};

use crate::device::{check, device_info, DeviceInfo};
use crate::error::{Error, Result};
use crate::memory::{Buffer, Domain, PoolStats};

/// What kind of memory an address is, as HIP classifies it.
///
/// Vendor entry points differ in the pointer class they accept, and a wrong class does
/// not always fail cleanly: rocSOLVER's LAPACK-style `info` must be device memory, and a
/// host address there makes the writing kernel page-fault and the process wait forever.
/// A seam checks the class with [`Runtime::pointer_class`] before such a call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointerClass {
    /// Device memory on the given ordinal, including stream-ordered pool allocations.
    Device {
        /// Device the memory belongs to.
        ordinal: i32,
    },
    /// Registered or pinned host memory.
    Host,
    /// Managed memory.
    Managed,
    /// An address HIP knows nothing about: a stack or ordinary heap address.
    Unregistered,
}

/// A HIP execution owner.
///
/// One runtime owns one real non-blocking stream, one explicitly created memory pool
/// and one allocation-domain identity. Opening the same ordinal twice yields two
/// independent runtimes with different streams and pools. Submission takes
/// `&mut Runtime`, so a runtime is `Send` but not `Sync`.
pub struct Runtime {
    domain: Arc<Domain>,
    device: DeviceInfo,
    _not_sync: PhantomData<std::cell::Cell<()>>,
}

impl std::fmt::Debug for Runtime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Runtime")
            .field("device", &self.device)
            .field("domain", &self.domain.id())
            .finish()
    }
}

/// Keeps a runtime's stream and pool alive without any submission authority.
///
/// A vendor handle bound to the runtime's stream (a rocBLAS handle after
/// `rocblas_set_stream`) holds one, so the stream cannot be destroyed while the handle
/// still refers to it.
#[derive(Clone)]
pub struct DomainAnchor {
    domain: Arc<Domain>,
}

impl std::fmt::Debug for DomainAnchor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DomainAnchor")
            .field("domain", &self.domain.id())
            .finish()
    }
}

impl DomainAnchor {
    /// Identity of the anchored runtime.
    pub fn domain_id(&self) -> u64 {
        self.domain.id()
    }

    /// Make the anchored device current on this thread, for a vendor call made outside
    /// a raw scope (a handle's destruction).
    ///
    /// # Errors
    ///
    /// [`Error::Hip`] when the device cannot be activated.
    pub fn activate(&self) -> Result<()> {
        self.domain.activate()
    }

    /// Wait for every operation queued on the anchored stream, for example before a
    /// vendor handle bound to it is destroyed.
    ///
    /// # Errors
    ///
    /// [`Error::Hip`], including an asynchronous failure of earlier work.
    pub fn synchronize(&self) -> Result<()> {
        self.domain.synchronize()
    }
}

impl Runtime {
    /// Open `ordinal` and create its stream and pool.
    ///
    /// # Errors
    ///
    /// [`Error::Unavailable`] when HIP is absent, [`Error::LibraryUnusable`] or
    /// [`Error::MissingSymbol`] for a broken or partial install, [`Error::DuplicateRuntime`] when another `libamdhip64` is
    /// already mapped, [`Error::Configuration`] for an empty
    /// `HSA_OVERRIDE_GFX_VERSION`, [`Error::InvalidDevice`] for an unknown ordinal, [`Error::Unsupported`] when the device has no memory pools, and
    /// [`Error::Hip`] for a failing HIP call.
    pub fn open(ordinal: usize) -> Result<Runtime> {
        let hip = HipTable::load()?;
        let device = device_info(ordinal)?;
        if !device.memory_pools {
            return Err(Error::Unsupported {
                detail: format!(
                    "device {ordinal} ({}) has no stream-ordered memory pools",
                    device.gcn_arch_name
                ),
            });
        }
        let ordinal =
            i32::try_from(ordinal).map_err(|_| Error::InvalidDevice { ordinal, count: 0 })?;
        let domain = Domain::create(hip, ordinal)?;
        Ok(Runtime {
            domain,
            device,
            _not_sync: PhantomData,
        })
    }

    /// What HIP reports for this runtime's device.
    pub fn device(&self) -> &DeviceInfo {
        &self.device
    }

    /// Allocation-domain identity, distinct for every opened runtime.
    pub fn domain_id(&self) -> u64 {
        self.domain.id()
    }

    /// Opaque identity of this runtime's stream. Two runtimes on one ordinal differ.
    pub fn stream_id(&self) -> u64 {
        self.domain.stream() as usize as u64
    }

    /// An anchor that keeps this runtime's stream and pool alive.
    pub fn anchor(&self) -> DomainAnchor {
        DomainAnchor {
            domain: self.domain.clone(),
        }
    }

    /// Allocate `bytes` from this runtime's pool and zero them in stream order.
    ///
    /// # Errors
    ///
    /// [`Error::Hip`] when the allocation or the fill fails.
    pub fn alloc_zeroed(&mut self, bytes: usize) -> Result<Buffer> {
        alloc_zeroed(&self.domain, bytes)
    }

    /// Copy `data` into a new device buffer.
    ///
    /// The call returns once the copy has completed, so the caller's slice is never
    /// read afterwards. Pinned staging is not part of this runtime yet.
    ///
    /// # Errors
    ///
    /// [`Error::Hip`] when the allocation, the copy or the synchronisation fails.
    pub fn upload(&mut self, data: &[u8]) -> Result<Buffer> {
        let pointer = self.domain.allocate(data.len())?;
        let buffer = Buffer::new(pointer, data.len(), self.domain.clone());
        if !data.is_empty() {
            let hip = self.domain.hip();
            // SAFETY: `pointer` addresses `data.len()` bytes of this domain and `data` is
            // live until the stream is synchronised below.
            check(hip, "hipMemcpyAsync", unsafe {
                (hip.memcpy_async)(
                    pointer,
                    data.as_ptr().cast(),
                    data.len(),
                    HIP_MEMCPY_HOST_TO_DEVICE,
                    self.domain.stream(),
                )
            })?;
            self.domain.synchronize()?;
        }
        Ok(buffer)
    }

    /// Copy a buffer back to the host, after every operation queued before it.
    ///
    /// # Errors
    ///
    /// [`Error::ForeignDomain`] for a buffer of another runtime, and [`Error::Hip`]
    /// when the copy or the synchronisation fails (including an earlier asynchronous
    /// failure on the stream).
    pub fn download(&mut self, buffer: &Buffer) -> Result<Vec<u8>> {
        self.own(buffer)?;
        let mut host = vec![0u8; buffer.len()];
        if !host.is_empty() {
            self.domain.activate()?;
            let hip = self.domain.hip();
            // SAFETY: `host` has `buffer.len()` writable bytes and stays alive until the
            // stream is synchronised below; the source belongs to this domain.
            check(hip, "hipMemcpyAsync", unsafe {
                (hip.memcpy_async)(
                    host.as_mut_ptr().cast(),
                    buffer.pointer(),
                    host.len(),
                    HIP_MEMCPY_DEVICE_TO_HOST,
                    self.domain.stream(),
                )
            })?;
        }
        self.domain.synchronize()?;
        Ok(host)
    }

    /// Wait for every operation queued on this runtime's stream.
    ///
    /// # Errors
    ///
    /// [`Error::Hip`], including an asynchronous failure of earlier work.
    pub fn synchronize(&mut self) -> Result<()> {
        self.domain.synchronize()
    }

    /// Bytes this runtime's pool holds.
    ///
    /// Memory a vendor library allocates on its own is not here; that is why vendor
    /// workspaces are handed pool buffers.
    ///
    /// # Errors
    ///
    /// [`Error::Hip`] when the pool cannot be queried.
    pub fn pool_stats(&self) -> Result<PoolStats> {
        self.domain.pool_stats()
    }

    /// Return cached pool memory above `keep_bytes` to the device.
    ///
    /// # Errors
    ///
    /// [`Error::Hip`].
    pub fn trim_pool(&mut self, keep_bytes: usize) -> Result<()> {
        self.domain.synchronize()?;
        self.domain.trim(keep_bytes)
    }

    /// Classify `address` with `hipPointerGetAttributes`.
    ///
    /// # Errors
    ///
    /// [`Error::Hip`] when HIP cannot classify the address, and [`Error::Unsupported`]
    /// for a memory type this crate does not know.
    pub fn pointer_class(&self, address: *const c_void) -> Result<PointerClass> {
        pointer_class(&self.domain, address)
    }

    /// Run `closure` with this runtime's stream and the device pointers of `buffers`.
    ///
    /// The runtime checks that every buffer belongs to it and makes its device current
    /// on this thread before the closure runs. Work the closure queues on
    /// [`RawScope::stream`] is ordered with everything else this runtime submits, so a
    /// buffer dropped after the scope is freed only after that work.
    ///
    /// # Safety
    ///
    /// The closure must queue its work on [`RawScope::stream`], must read and write only
    /// within buffers it registered or allocated through the scope, and must keep any
    /// other resource its queued work reads alive until that work completes.
    ///
    /// # Errors
    ///
    /// [`Error::ForeignDomain`] for a buffer of another runtime, and [`Error::Hip`]
    /// when the device cannot be activated.
    pub unsafe fn with_raw<R>(
        &mut self,
        buffers: &[&Buffer],
        closure: impl FnOnce(&RawScope<'_>) -> R,
    ) -> Result<R> {
        for buffer in buffers {
            self.own(buffer)?;
        }
        self.domain.activate()?;
        let scope = RawScope {
            domain: &self.domain,
            registered: RefCell::new(buffers.iter().map(|b| b.pointer() as usize).collect()),
            _not_send: PhantomData,
        };
        Ok(closure(&scope))
    }

    fn own(&self, buffer: &Buffer) -> Result<()> {
        if !Arc::ptr_eq(buffer.domain(), &self.domain) {
            return Err(Error::ForeignDomain);
        }
        Ok(())
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        // Best effort; buffers still alive keep the stream and pool alive.
        let _ = self.domain.synchronize();
    }
}

fn alloc_zeroed(domain: &Arc<Domain>, bytes: usize) -> Result<Buffer> {
    let pointer = domain.allocate(bytes)?;
    let buffer = Buffer::new(pointer, bytes, domain.clone());
    if bytes > 0 {
        let hip = domain.hip();
        // SAFETY: `pointer` addresses `bytes` writable bytes of this domain.
        check(hip, "hipMemsetAsync", unsafe {
            (hip.memset_async)(pointer, 0, bytes, domain.stream())
        })?;
    }
    Ok(buffer)
}

fn pointer_class(domain: &Domain, address: *const c_void) -> Result<PointerClass> {
    // The classification names a device ordinal, so ask on the runtime's device.
    domain.activate()?;
    let hip = domain.hip();
    let mut attributes = PointerAttributes::zeroed();
    // SAFETY: `attributes` is writable; HIP only inspects `address`, never dereferences
    // it.
    let code = unsafe { (hip.pointer_get_attributes)(&mut attributes, address) };
    if code != t4a_hip_sys::HIP_SUCCESS {
        // Clear the per-thread last error the query set.
        // SAFETY: reads and resets thread-local runtime state.
        unsafe { (hip.get_last_error)() };
        return Err(Error::Hip {
            call: "hipPointerGetAttributes",
            code,
            name: hip.error_name(code),
        });
    }
    match attributes.memory_type {
        HIP_MEMORY_TYPE_DEVICE => Ok(PointerClass::Device {
            ordinal: attributes.device,
        }),
        HIP_MEMORY_TYPE_HOST => Ok(PointerClass::Host),
        HIP_MEMORY_TYPE_MANAGED => Ok(PointerClass::Managed),
        HIP_MEMORY_TYPE_UNREGISTERED => Ok(PointerClass::Unregistered),
        other => Err(Error::Unsupported {
            detail: format!("hipPointerGetAttributes reported memory type {other}"),
        }),
    }
}

/// A raw view of a runtime for vendor calls, valid only on the calling thread.
///
/// Deliberately `!Send + !Sync`: HIP's current device is per thread, and the scope made
/// it current on this one.
pub struct RawScope<'a> {
    domain: &'a Arc<Domain>,
    registered: RefCell<Vec<usize>>,
    _not_send: PhantomData<*const ()>,
}

impl RawScope<'_> {
    /// The runtime's stream, as a raw `hipStream_t`.
    pub fn stream(&self) -> *mut c_void {
        self.domain.stream()
    }

    /// The runtime's device ordinal.
    pub fn ordinal(&self) -> i32 {
        self.domain.ordinal()
    }

    /// Identity of the runtime this scope belongs to.
    pub fn domain_id(&self) -> u64 {
        self.domain.id()
    }

    /// Device pointer of a buffer registered with this scope.
    ///
    /// # Errors
    ///
    /// [`Error::ForeignDomain`] for a buffer of another runtime, and
    /// [`Error::InvalidArgument`] for a buffer of this runtime that was not registered.
    pub fn device_ptr(&self, buffer: &Buffer) -> Result<*mut c_void> {
        if !Arc::ptr_eq(buffer.domain(), self.domain) {
            return Err(Error::ForeignDomain);
        }
        let pointer = buffer.pointer();
        if !self.registered.borrow().contains(&(pointer as usize)) {
            return Err(Error::invalid(
                "buffer was not registered with this raw scope",
            ));
        }
        Ok(pointer)
    }

    /// Register a buffer the caller only learned about inside the scope.
    ///
    /// # Errors
    ///
    /// [`Error::ForeignDomain`] for a buffer of another runtime.
    pub fn register(&self, buffer: &Buffer) -> Result<*mut c_void> {
        if !Arc::ptr_eq(buffer.domain(), self.domain) {
            return Err(Error::ForeignDomain);
        }
        let pointer = buffer.pointer();
        let mut registered = self.registered.borrow_mut();
        if !registered.contains(&(pointer as usize)) {
            registered.push(pointer as usize);
        }
        Ok(pointer)
    }

    /// Allocate `bytes` zeroed bytes from the runtime's pool and register them.
    ///
    /// This is how a vendor workspace is taken from the pool, so the pool's accounting
    /// sees it.
    ///
    /// # Errors
    ///
    /// [`Error::Hip`] when the allocation or the fill fails.
    pub fn alloc_zeroed(&self, bytes: usize) -> Result<Buffer> {
        let buffer = alloc_zeroed(self.domain, bytes)?;
        self.register(&buffer)?;
        Ok(buffer)
    }

    /// Classify `address`; see [`Runtime::pointer_class`].
    ///
    /// # Errors
    ///
    /// As [`Runtime::pointer_class`].
    pub fn pointer_class(&self, address: *const c_void) -> Result<PointerClass> {
        pointer_class(self.domain, address)
    }
}
