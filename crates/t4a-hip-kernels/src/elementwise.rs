//! The flat element-wise entries of the shared ABI, safely.

use std::ffi::c_void;

use t4a_cuda_kernels::{plan_elementwise, DType, Entry};
use t4a_hip_runtime::{Buffer, Runtime};

use crate::error::{Error, Result};
use crate::program::Program;

/// The three buffers of a flat binary entry.
#[derive(Clone, Copy)]
pub struct Binary<'a> {
    /// Destination, `len` elements.
    pub out: &'a Buffer,
    /// Left operand, `len` elements.
    pub lhs: &'a Buffer,
    /// Right operand, `len` elements.
    pub rhs: &'a Buffer,
}

/// Run a flat binary entry (`t4a_cuda_add`, `_sub`, `_mul`, `_div`) over `len`
/// contiguous elements: `out[i] = lhs[i] op rhs[i]`.
///
/// The kernel signature is the ABI's `(void* out, const void* lhs, const void* rhs,
/// int dtype, long long len)`. The launch is queued on the runtime's stream; nothing is
/// waited for, so `runtime.synchronize()` or a download orders after it.
///
/// # Errors
///
/// [`Error::Plan`] for an entry or dtype the ABI does not allow,
/// [`Error::BufferTooSmall`], and [`Error::Runtime`] (including a buffer of another
/// runtime) from the launch.
pub fn elementwise(
    runtime: &mut Runtime,
    program: &Program,
    entry: Entry,
    dtype: DType,
    buffers: Binary<'_>,
    len: i64,
) -> Result<()> {
    let Binary { out, lhs, rhs } = buffers;
    let launch = plan_elementwise(entry, dtype, len)?;
    let need = len as usize * dtype.size();
    for (which, buffer) in [("out", out), ("lhs", lhs), ("rhs", rhs)] {
        if buffer.len() < need {
            return Err(Error::BufferTooSmall {
                which,
                have: buffer.len(),
                need,
            });
        }
    }
    // SAFETY: the closure queues one launch on the scope's stream; every device
    // pointer comes from a buffer registered with the scope and checked to hold `len`
    // elements, and the argument list is the ABI declaration of the flat entries.
    unsafe {
        runtime.with_raw(&[out, lhs, rhs], |scope| {
            let mut out_ptr = scope.device_ptr(out)?;
            let mut lhs_ptr = scope.device_ptr(lhs)?;
            let mut rhs_ptr = scope.device_ptr(rhs)?;
            let mut tag: i32 = dtype.tag();
            let mut count: i64 = len;
            let mut args: [*mut c_void; 5] = [
                (&mut out_ptr as *mut *mut c_void).cast(),
                (&mut lhs_ptr as *mut *mut c_void).cast(),
                (&mut rhs_ptr as *mut *mut c_void).cast(),
                (&mut tag as *mut i32).cast(),
                (&mut count as *mut i64).cast(),
            ];
            program.launch(scope, entry.name(), &launch, &mut args)
        })?
    }
}
