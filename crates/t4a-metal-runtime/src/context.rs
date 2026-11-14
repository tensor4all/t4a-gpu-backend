//! The device, its command queue and the runtime's allocation-domain identity.

use std::sync::atomic::{AtomicU64, Ordering};

use t4a_metal_sys::objc::{bool_value, Id};
use t4a_metal_sys::Table;

use crate::error::{Error, Result};

/// The runtime's domain identity for the next context.
static NEXT_DOMAIN: AtomicU64 = AtomicU64::new(1);

/// Facts about the Metal device this runtime owns.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// The device's own name.
    pub name: String,
    /// Whether CPU and GPU memory are one address space.
    pub unified_memory: bool,
    /// Largest single buffer the device accepts, in bytes.
    pub max_buffer_length: u64,
    /// The device's recommended working set, in bytes.
    pub recommended_max_working_set: u64,
}

/// One device, one command queue and one allocation-domain identity.
///
/// Shared by every handle that can outlive the [`crate::Runtime`], which is why it
/// is reference counted: releasing the runtime while a submission is running must
/// not release the queue under it.
pub(crate) struct ContextHolder {
    pub(crate) table: Table,
    device: Id,
    queue: Id,
    pub(crate) info: DeviceInfo,
    domain_id: u64,
    next_allocation: AtomicU64,
    live_allocations: AtomicU64,
    live_bytes: AtomicU64,
}

// SAFETY: Metal device and queue handles are documented as thread-safe, and every
// mutation here is atomic. Submission still requires `&mut Runtime`.
unsafe impl Send for ContextHolder {}
unsafe impl Sync for ContextHolder {}

impl ContextHolder {
    /// Take the system default device and one command queue from it.
    pub(crate) fn open(table: Table) -> Result<Self> {
        let Some(device) = table.system_default_device() else {
            return Err(Error::unavailable(
                "no Metal device",
                "MTLCreateSystemDefaultDevice returned nil, so this host has no \
                 usable Metal device",
            ));
        };
        let selectors = table.selectors;
        // SAFETY: `device` is the live device taken above.
        let name = unsafe { table.device_name(device) }.unwrap_or_else(|| "unknown".to_string());
        // SAFETY: `device` is the live `+1` device taken above.
        let unified_memory =
            bool_value(unsafe { selectors.device_has_unified_memory.send(device) });
        // SAFETY: as above; these selectors return scalars.
        let max_buffer_length = unsafe { selectors.device_max_buffer_length.send(device) };
        // SAFETY: as above.
        let recommended_max_working_set =
            unsafe { selectors.device_recommended_max_working_set.send(device) };
        // SAFETY: as above; `newCommandQueue` returns a `+1` queue or nil.
        let queue = unsafe { selectors.device_new_command_queue.send(device) };
        if queue.is_null() {
            // SAFETY: release the device reference taken above; nothing else holds it.
            unsafe { table.objc.release(device) };
            return Err(Error::unavailable(
                "no Metal command queue",
                "the device returned nil for a new command queue",
            ));
        }
        Ok(Self {
            table,
            device,
            queue,
            info: DeviceInfo {
                name,
                unified_memory,
                max_buffer_length,
                recommended_max_working_set,
            },
            domain_id: NEXT_DOMAIN.fetch_add(1, Ordering::Relaxed),
            next_allocation: AtomicU64::new(1),
            live_allocations: AtomicU64::new(0),
            live_bytes: AtomicU64::new(0),
        })
    }

    /// The device handle. Valid while this holder is alive.
    pub(crate) fn device(&self) -> Id {
        self.device
    }

    /// The command queue handle. Valid while this holder is alive.
    pub(crate) fn queue(&self) -> Id {
        self.queue
    }

    /// This runtime's allocation-domain identity.
    pub(crate) fn domain_id(&self) -> u64 {
        self.domain_id
    }

    /// A fresh allocation identity, stable for the allocation's lifetime.
    pub(crate) fn next_allocation_id(&self) -> u64 {
        self.next_allocation.fetch_add(1, Ordering::Relaxed)
    }

    pub(crate) fn note_alloc(&self, bytes: usize) {
        self.live_allocations.fetch_add(1, Ordering::Relaxed);
        self.live_bytes.fetch_add(bytes as u64, Ordering::Relaxed);
    }

    pub(crate) fn note_free(&self, bytes: usize) {
        self.live_allocations.fetch_sub(1, Ordering::Relaxed);
        self.live_bytes.fetch_sub(bytes as u64, Ordering::Relaxed);
    }

    pub(crate) fn live_allocations(&self) -> u64 {
        self.live_allocations.load(Ordering::Relaxed)
    }

    pub(crate) fn live_bytes(&self) -> u64 {
        self.live_bytes.load(Ordering::Relaxed)
    }
}

impl Drop for ContextHolder {
    fn drop(&mut self) {
        // Every allocation and submission that could still use these handles holds
        // its own `Arc<ContextHolder>`, so reaching here means nothing is in flight.
        // SAFETY: these are the `+1` handles taken in `open`, released once.
        unsafe {
            self.table.objc.release(self.queue);
            self.table.objc.release(self.device);
        }
    }
}

impl std::fmt::Debug for ContextHolder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContextHolder")
            .field("domain_id", &self.domain_id)
            .field("device", &self.info.name)
            .finish_non_exhaustive()
    }
}
