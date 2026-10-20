//! CUDA driver initialisation, primary-context ownership and scoped context
//! activation.
//!
//! Every CUDA call in this crate runs with the owning context current. A
//! [`CtxScope`] saves the caller's current context and restores it on success,
//! returned error and unwind, so this crate never leaves a thread bound to a
//! context it did not own.

use std::ptr;
use std::sync::Arc;

use cudarc::driver::result as drv;
use cudarc::driver::sys;

use crate::error::{Error, Result};

/// Driver entry points this runtime requires at run time.
const REQUIRED_SYMBOLS: &[&[u8]] = &[
    b"cuInit\0",
    b"cuDeviceGet\0",
    b"cuDeviceGetCount\0",
    b"cuDeviceGetName\0",
    b"cuDeviceGetAttribute\0",
    b"cuDevicePrimaryCtxRetain\0",
    b"cuDevicePrimaryCtxRelease_v2\0",
    b"cuCtxSetCurrent\0",
    b"cuCtxGetCurrent\0",
    b"cuStreamCreate\0",
    b"cuStreamDestroy_v2\0",
    b"cuStreamSynchronize\0",
    b"cuStreamWaitEvent\0",
    b"cuMemPoolCreate\0",
    b"cuMemPoolDestroy\0",
    b"cuMemPoolTrimTo\0",
    b"cuMemPoolGetAttribute\0",
    b"cuMemPoolSetAttribute\0",
    b"cuMemAllocFromPoolAsync\0",
    b"cuMemFreeAsync\0",
    b"cuMemcpyHtoDAsync_v2\0",
    b"cuMemcpyDtoHAsync_v2\0",
    b"cuMemcpyDtoDAsync_v2\0",
    b"cuMemsetD8Async\0",
    b"cuMemHostAlloc\0",
    b"cuMemFreeHost\0",
    b"cuEventCreate\0",
    b"cuEventRecord\0",
    b"cuEventQuery\0",
    b"cuEventDestroy_v2\0",
    b"cuModuleLoadData\0",
    b"cuModuleUnload\0",
    b"cuModuleGetFunction\0",
    b"cuLaunchKernel\0",
];

/// Initialise the driver, reporting an unavailable driver as a typed error.
pub(crate) fn init() -> Result<()> {
    // SAFETY: probing the dynamic loader only; it does not initialise CUDA.
    if !unsafe { sys::is_culib_present() } {
        return Err(Error::DriverUnavailable {
            detail: "libcuda could not be loaded".to_string(),
        });
    }
    drv::init().map_err(Error::driver("cuInit"))?;
    probe_driver_symbols()
}

/// Resolve the required driver symbols once, converting a missing symbol into a
/// typed error instead of letting a later wrapper panic.
fn probe_driver_symbols() -> Result<()> {
    // SAFETY: `culib()` returns the library that was successfully loaded above.
    let lib = unsafe { sys::culib() };
    for symbol in REQUIRED_SYMBOLS {
        // SAFETY: only resolving the symbol; the value is dropped immediately.
        let resolved = unsafe { lib.get::<*const ()>(*symbol) };
        if resolved.is_err() {
            return Err(Error::Unsupported {
                detail: format!(
                    "driver symbol {} is missing",
                    String::from_utf8_lossy(&symbol[..symbol.len() - 1])
                ),
            });
        }
    }
    Ok(())
}

/// Number of CUDA devices visible to the driver.
pub fn device_count() -> Result<usize> {
    init()?;
    let count = drv::device::get_count().map_err(Error::driver("cuDeviceGetCount"))?;
    Ok(count as usize)
}

/// Name reported by the driver for `ordinal`, if it exists.
pub fn device_name(ordinal: usize) -> Result<String> {
    let device = device(ordinal)?;
    drv::device::get_name(device).map_err(Error::driver("cuDeviceGetName"))
}

fn device(ordinal: usize) -> Result<sys::CUdevice> {
    init()?;
    let count = drv::device::get_count().map_err(Error::driver("cuDeviceGetCount"))? as usize;
    if ordinal >= count {
        return Err(Error::InvalidDevice { ordinal, count });
    }
    drv::device::get(ordinal as i32).map_err(Error::driver("cuDeviceGet"))
}

/// Whether `ordinal` supports stream-ordered memory pools.
pub fn supports_memory_pool(ordinal: usize) -> Result<bool> {
    let device = device(ordinal)?;
    // SAFETY: `device` came from cuDeviceGet.
    let supported = unsafe {
        drv::device::get_attribute(
            device,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED,
        )
    }
    .map_err(Error::driver("cuDeviceGetAttribute"))?;
    Ok(supported != 0)
}

/// A retained CUDA primary context for one device.
///
/// The handle keeps the primary context alive for as long as any buffer,
/// module, event or pending record refers to it. Releasing it releases only
/// this holder's retain; other holders of the same primary context are
/// unaffected.
pub(crate) struct ContextHolder {
    device: sys::CUdevice,
    context: sys::CUcontext,
    ordinal: usize,
}

// SAFETY: a retained context handle is process-wide and not thread-affine; every
// use activates it on the calling thread through `activate`.
unsafe impl Send for ContextHolder {}
unsafe impl Sync for ContextHolder {}

impl ContextHolder {
    pub(crate) fn open(ordinal: usize) -> Result<Arc<Self>> {
        let device = device(ordinal)?;
        if !supports_memory_pool(ordinal)? {
            return Err(Error::unsupported(
                "device does not support CUDA memory pools",
            ));
        }
        // SAFETY: `device` is a valid ordinal from cuDeviceGet.
        let context = unsafe { drv::primary_ctx::retain(device) }
            .map_err(Error::driver("cuDevicePrimaryCtxRetain"))?;
        Ok(Arc::new(Self {
            device,
            context,
            ordinal,
        }))
    }

    pub(crate) fn ordinal(&self) -> usize {
        self.ordinal
    }

    #[allow(dead_code)]
    pub(crate) fn raw(&self) -> sys::CUcontext {
        self.context
    }

    /// Make this context current and return a scope that restores the previous
    /// one when dropped.
    pub(crate) fn activate(self: &Arc<Self>) -> Result<CtxScope> {
        let previous = drv::ctx::get_current().map_err(Error::driver("cuCtxGetCurrent"))?;
        if previous != Some(self.context) {
            // SAFETY: this holder retains the context.
            unsafe { drv::ctx::set_current(self.context) }
                .map_err(Error::driver("cuCtxSetCurrent"))?;
        }
        Ok(CtxScope {
            holder: Arc::clone(self),
            previous,
        })
    }
}

impl Drop for ContextHolder {
    fn drop(&mut self) {
        // SAFETY: this is the last strong reference; re-activate, release the
        // retain, and never restore a handle that was just released.
        let previous = drv::ctx::get_current().ok().flatten();
        unsafe {
            let _ = drv::ctx::set_current(self.context);
            let _ = drv::primary_ctx::release(self.device);
            match previous {
                // Another holder may still retain the same primary context, so the
                // handle stays valid and must remain current for its user.
                Some(previous) if previous == self.context => {
                    if drv::ctx::set_current(previous).is_err() {
                        let _ = drv::ctx::set_current(ptr::null_mut());
                    }
                }
                Some(previous) => {
                    let _ = drv::ctx::set_current(previous);
                }
                None => {
                    let _ = drv::ctx::set_current(ptr::null_mut());
                }
            }
        }
    }
}

/// An active context binding that restores the previous binding when dropped.
pub(crate) struct CtxScope {
    #[allow(dead_code)]
    holder: Arc<ContextHolder>,
    previous: Option<sys::CUcontext>,
}

impl Drop for CtxScope {
    fn drop(&mut self) {
        // SAFETY: restoring the value saved by `activate`; a failure here cannot
        // be reported from Drop and leaves the thread on the owning context.
        unsafe {
            match self.previous {
                Some(previous) => {
                    let _ = drv::ctx::set_current(previous);
                }
                None => {
                    let _ = drv::ctx::set_current(ptr::null_mut());
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
