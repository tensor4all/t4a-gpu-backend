//! The unsafe raw-CUDA consumer seam.
//!
//! A raw scope lets a consumer use its own vendor handle (cuBLAS, cuFFT, ...) on
//! this runtime's exact stream, or launch a kernel directly. The runtime
//! pre-validates registered resources and reserves their retained sizes before
//! the stream is exposed, and keeps cleanup outside the caller's control.

use std::any::Any;
use std::marker::PhantomData;
use std::sync::Arc;

use cudarc::driver::sys;

use crate::error::{Error, Result};
use crate::memory::{check_span, Buffer, Domain};
use crate::retirement::Admission;

/// A resource registered for the duration of a raw scope.
pub enum RawResource<'a> {
    /// A device buffer owned by the same runtime.
    Buffer(&'a Buffer),
    /// A caller-owned value kept alive until the scope's completion.
    Keepalive {
        /// The value to retain.
        value: Box<dyn Any + Send>,
        /// Bytes this value is estimated to retain; counted against the budget.
        bytes: usize,
    },
}

/// A raw view of the runtime that is only valid on the calling thread.
///
/// This type is deliberately `!Send + !Sync`: the CUDA stream and context are
/// bound to the thread that opened the scope.
pub struct RawScope<'a> {
    domain: &'a Arc<Domain>,
    stream: sys::CUstream,
    _not_send: PhantomData<*const ()>,
}

impl RawScope<'_> {
    /// The runtime's real stream, as a raw `CUstream` value.
    pub fn stream(&self) -> u64 {
        self.stream as usize as u64
    }

    /// Allocation-domain identity of this runtime.
    pub fn domain_id(&self) -> u64 {
        self.domain.id()
    }

    /// Device pointer of a buffer registered with this scope.
    pub fn device_ptr(&self, buffer: &Buffer) -> Result<u64> {
        if !Arc::ptr_eq(buffer.alloc().domain(), self.domain) {
            return Err(Error::ForeignDomain);
        }
        Ok(buffer.device_ptr())
    }

    /// Device pointer of a checked `offset..offset + len` span of a registered
    /// buffer.
    pub fn device_ptr_range(&self, buffer: &Buffer, offset: usize, len: usize) -> Result<u64> {
        self.device_ptr(buffer)?;
        check_span(buffer.len(), offset, len)?;
        Ok(buffer.device_ptr() + offset as u64)
    }
}

struct PanicGuard<'a, 'b> {
    domain: &'a Arc<Domain>,
    admission: &'a Admission<'b>,
    armed: bool,
}

impl Drop for PanicGuard<'_, '_> {
    fn drop(&mut self) {
        if self.armed {
            // A panicking closure may have left queued or partially queued work
            // whose resources we cannot account for. Poison new admissions and
            // seal whatever was registered so it can still be retired.
            self.domain
                .mark_broken("raw CUDA closure panicked during submission");
            let _ = self.admission.seal(self.domain.stream());
        }
    }
}

pub(crate) unsafe fn with_raw<R>(
    domain: &Arc<Domain>,
    resources: Vec<RawResource<'_>>,
    closure: impl FnOnce(RawScope<'_>) -> R,
) -> Result<R> {
    if let Some(detail) = domain.broken() {
        return Err(Error::Broken { detail });
    }
    let mut keepalives: Vec<(Box<dyn Any + Send>, usize)> = Vec::new();
    for resource in resources {
        match resource {
            RawResource::Buffer(buffer) => {
                if !Arc::ptr_eq(buffer.alloc().domain(), domain) {
                    return Err(Error::ForeignDomain);
                }
            }
            RawResource::Keepalive { value, bytes } => keepalives.push((value, bytes)),
        }
    }
    let _scope = domain.scope()?;
    let stream = domain.stream();
    domain.retirement().during_admission(|admission| {
        // Reserve the whole scope before the first enqueue; the operation lock is
        // held for the entire closure, so observation cannot seal or reclaim any
        // of these resources before their work is admitted.
        let bytes: usize = keepalives.iter().map(|(_, bytes)| *bytes).sum();
        admission.ensure_capacity(stream, bytes, keepalives.len())?;
        for (value, bytes) in keepalives {
            admission.add_keepalive(stream, value, bytes)?;
        }
        let mut guard = PanicGuard {
            domain,
            admission,
            armed: true,
        };
        let raw = RawScope {
            domain,
            stream,
            _not_send: PhantomData,
        };
        // SAFETY: the caller upholds the kernel ABI, pointer bounds and foreign
        // stream contract documented on `with_raw`.
        let value = closure(raw);
        guard.armed = false;
        drop(guard);
        admission.seal(stream)?;
        Ok(value)
    })
}

#[cfg(test)]
mod tests;
