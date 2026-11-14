//! Shared-mode `MTLBuffer` allocations.
//!
//! Every buffer this runtime hands out lives in the device's shared storage mode,
//! so the CPU engine and the GPU read and write the same address. The runtime
//! makes no copy and tracks no transfer. Ordering host access against submitted
//! work is the caller's responsibility, which is why [`Buffer::host_ptr`] returns a
//! raw pointer instead of a slice.

use std::sync::Arc;

use t4a_metal_sys::objc::Id;

use crate::context::ContextHolder;
use crate::error::{Error, Result};

/// `MTLResourceStorageModeShared`.
const STORAGE_MODE_SHARED: u64 = 0;

/// One Metal buffer and its owning context.
struct Allocation {
    context: Arc<ContextHolder>,
    buffer: Id,
    ptr: *mut u8,
    len: usize,
    id: u64,
}

// SAFETY: the holder is thread-safe and `ptr` is only handed out through unsafe
// accessors that require the caller to order it against submitted work.
unsafe impl Send for Allocation {}
unsafe impl Sync for Allocation {}

impl Drop for Allocation {
    fn drop(&mut self) {
        self.context.note_free(self.len);
        // SAFETY: this is the `+1` reference from `newBufferWithLength:options:`,
        // released exactly once, after the last handle that could use it is gone.
        unsafe { self.context.table.objc.release(self.buffer) };
    }
}

/// A buffer whose contents are initialised.
///
/// The handle carries no authority to submit work; launching needs
/// `&mut Runtime`. It is `Send + Sync` so a caller can share one allocation through
/// [`Arc<Buffer>`] instead of cloning allocation authority (there is no `Clone`).
pub struct Buffer {
    allocation: Arc<Allocation>,
}

/// A buffer whose contents are not yet initialised.
///
/// A kernel that fully overwrites it converts it with [`UninitBuffer::assume_init`].
pub struct UninitBuffer {
    allocation: Arc<Allocation>,
}

impl Buffer {
    pub(crate) fn allocate(context: &Arc<ContextHolder>, len: usize, zeroed: bool) -> Result<Self> {
        Ok(Self {
            allocation: allocate(context, len, zeroed)?,
        })
    }

    /// Length in bytes.
    pub fn len(&self) -> usize {
        self.allocation.len
    }

    /// Whether this buffer has no bytes.
    pub fn is_empty(&self) -> bool {
        self.allocation.len == 0
    }

    /// The allocation identity, stable for this allocation's lifetime.
    pub fn allocation_id(&self) -> u64 {
        self.allocation.id
    }

    /// The allocation-domain identity of the runtime that owns this buffer.
    pub fn domain_id(&self) -> u64 {
        self.allocation.context.domain_id()
    }

    /// The host address of the buffer's first byte.
    ///
    /// Metal's shared storage mode makes this the same address the GPU reads and
    /// writes. The caller must not touch these bytes while a submitted command
    /// buffer that reads or writes them may still be running.
    pub fn host_ptr(&self) -> *mut u8 {
        self.allocation.ptr
    }

    pub(crate) fn handle(&self) -> Id {
        self.allocation.buffer
    }

    pub(crate) fn context(&self) -> &Arc<ContextHolder> {
        &self.allocation.context
    }
}

impl UninitBuffer {
    pub(crate) fn allocate(context: &Arc<ContextHolder>, len: usize) -> Result<Self> {
        Ok(Self {
            allocation: allocate(context, len, false)?,
        })
    }

    /// Length in bytes.
    pub fn len(&self) -> usize {
        self.allocation.len
    }

    /// Whether this buffer has no bytes.
    pub fn is_empty(&self) -> bool {
        self.allocation.len == 0
    }

    /// The allocation identity, stable for this allocation's lifetime.
    pub fn allocation_id(&self) -> u64 {
        self.allocation.id
    }

    /// The allocation-domain identity of the runtime that owns this buffer.
    pub fn domain_id(&self) -> u64 {
        self.allocation.context.domain_id()
    }

    /// The host address of the buffer's first byte.
    pub fn host_ptr(&self) -> *mut u8 {
        self.allocation.ptr
    }

    pub(crate) fn handle(&self) -> Id {
        self.allocation.buffer
    }

    pub(crate) fn context(&self) -> &Arc<ContextHolder> {
        &self.allocation.context
    }

    /// Declare the contents initialised after a kernel fully overwrote them.
    ///
    /// # Safety
    ///
    /// Every byte must have been written by submitted work that has completed, and
    /// no later submission may leave part of it unwritten.
    pub unsafe fn assume_init(self) -> Buffer {
        Buffer {
            allocation: self.allocation,
        }
    }
}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Buffer")
            .field("len", &self.allocation.len)
            .field("allocation_id", &self.allocation.id)
            .field("domain_id", &self.allocation.context.domain_id())
            .finish()
    }
}

impl std::fmt::Debug for UninitBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UninitBuffer")
            .field("len", &self.allocation.len)
            .field("allocation_id", &self.allocation.id)
            .finish()
    }
}

fn allocate(context: &Arc<ContextHolder>, len: usize, zeroed: bool) -> Result<Arc<Allocation>> {
    context.note_alloc(len);
    if len == 0 {
        // Metal may return nil for a zero-length request, and an empty buffer has
        // no address to touch. Represent it without asking the device.
        return Ok(Arc::new(Allocation {
            context: Arc::clone(context),
            buffer: std::ptr::null_mut(),
            ptr: std::ptr::null_mut(),
            len: 0,
            id: context.next_allocation_id(),
        }));
    }
    if len as u64 > context.info.max_buffer_length {
        context.note_free(len);
        return Err(Error::OutOfMemory { bytes: len });
    }
    let table = context.table;
    let selectors = table.selectors;
    // SAFETY: the device is live and the length is inside `maxBufferLength`, which
    // makes `newBufferWithLength:options:` the documented way to allocate shared
    // storage. It returns a `+1` buffer or nil.
    let buffer = unsafe {
        selectors
            .device_new_buffer
            .send(context.device(), len as u64, STORAGE_MODE_SHARED)
    };
    if buffer.is_null() {
        context.note_free(len);
        return Err(Error::OutOfMemory { bytes: len });
    }
    // SAFETY: `buffer` is a live `MTLBuffer`; `contents` is valid for `length`
    // bytes because the storage mode is shared.
    let ptr = unsafe { selectors.buffer_contents.send(buffer) }.cast::<u8>();
    if zeroed {
        // SAFETY: the buffer was just allocated, nothing can reference it yet, and
        // `ptr` addresses `len` writable bytes in shared storage.
        unsafe { std::ptr::write_bytes(ptr, 0, len) };
    }
    Ok(Arc::new(Allocation {
        context: Arc::clone(context),
        buffer,
        ptr,
        len,
        id: context.next_allocation_id(),
    }))
}
