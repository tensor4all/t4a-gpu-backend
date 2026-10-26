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
use crate::kernel::{self, Arg, BoundFunction, LaunchConfig};
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
    registered: &'a std::cell::RefCell<Vec<usize>>,
    admission: &'a Admission<'a>,
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
    ///
    /// # Errors
    ///
    /// [`Error::ForeignDomain`] for a buffer of another runtime and
    /// [`Error::InvalidArgument`] for a same-runtime buffer that was not listed
    /// in `resources`.
    pub fn device_ptr(&self, buffer: &Buffer) -> Result<u64> {
        if !Arc::ptr_eq(buffer.alloc().domain(), self.domain) {
            return Err(Error::ForeignDomain);
        }
        let pointer = buffer.device_ptr() as usize;
        if !self.registered.borrow().contains(&pointer) {
            return Err(Error::invalid(
                "buffer was not registered with this raw scope",
            ));
        }
        Ok(pointer as u64)
    }

    /// Register `buffer` with this scope and return its device pointer.
    ///
    /// A caller that only learns which allocation it will address *inside* the scope — a vendor
    /// plan that names its own workspace, an extension that picks its operand at execution time —
    /// registers it here instead of listing it up front. The registration is idempotent and is
    /// checked exactly like a declared one: the buffer must belong to this scope's domain.
    ///
    /// # Errors
    ///
    /// [`Error::ForeignDomain`] for a buffer of another runtime.
    pub fn register(&self, buffer: &Buffer) -> Result<u64> {
        if !Arc::ptr_eq(buffer.alloc().domain(), self.domain) {
            return Err(Error::ForeignDomain);
        }
        let pointer = buffer.device_ptr() as usize;
        let mut registered = self.registered.borrow_mut();
        if !registered.contains(&pointer) {
            registered.push(pointer);
        }
        Ok(pointer as u64)
    }

    /// Device pointer of a checked `offset..offset + len` span of a registered
    /// buffer.
    pub fn device_ptr_range(&self, buffer: &Buffer, offset: usize, len: usize) -> Result<u64> {
        self.device_ptr(buffer)?;
        check_span(buffer.len(), offset, len)?;
        Ok(buffer.device_ptr() + offset as u64)
    }

    /// Enqueue a checked launch through this runtime's own module cache.
    ///
    /// The launch is validated, retained and counted exactly like
    /// [`crate::Runtime::launch`]; use it when the consumer already holds the stream for
    /// other work in this scope.
    ///
    /// # Safety
    ///
    /// As [`crate::Runtime::launch`]: the bound function's kernel ABI must match `args`,
    /// every device pointer must address the bytes the kernel uses, and any output
    /// the caller later treats as initialised must be fully overwritten.
    ///
    /// # Errors
    ///
    /// [`Error::ForeignDomain`] for a function or buffer from another runtime,
    /// [`Error::KernelParameterLimit`] when the arguments exceed the device's
    /// parameter limit, and [`Error::Driver`] for a failed launch.
    pub unsafe fn launch(
        &self,
        function: &BoundFunction,
        config: LaunchConfig,
        args: &[Arg<'_>],
    ) -> Result<()> {
        let mut parameters = kernel::prepare(self.domain, function, config, args)?;
        // The scope seals once, after the closure, so its declared resources stay
        // pinned while the caller is still enqueueing external work.
        kernel::enqueue(
            self.domain,
            function,
            &mut parameters,
            self.admission,
            false,
        )
    }

    /// Keep `value` alive until this scope's work completes.
    ///
    /// A vendor call or a foreign function handle never passes through [`Arg`],
    /// so the runtime cannot see what such an enqueue uses. Register every one of
    /// those resources here before enqueueing and declare the bytes it retains;
    /// `bytes` is charged against the pending budget. Arbitrary device pointers,
    /// foreign streams and the ordering of external work stay the caller's
    /// responsibility.
    ///
    /// # Errors
    ///
    /// [`Error::Backpressure`] when the pending budget cannot cover `bytes`. A
    /// raw scope never seals its batch early to make room, because that event
    /// would cover only the work already queued and the resource could be
    /// released while a later external enqueue still uses it.
    pub fn retain(&self, value: Box<dyn Any + Send>, bytes: usize) -> Result<()> {
        self.admission
            .add_keepalive(self.stream, value, bytes, false)
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
    let mut registered: Vec<usize> = Vec::new();
    for resource in resources {
        match resource {
            RawResource::Buffer(buffer) => {
                if !Arc::ptr_eq(buffer.alloc().domain(), domain) {
                    return Err(Error::ForeignDomain);
                }
                registered.push(buffer.device_ptr() as usize);
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
            // Never seal this batch before the closure: these keepalives protect
            // work that has not been enqueued yet.
            admission.add_keepalive(stream, value, bytes, false)?;
        }
        let mut guard = PanicGuard {
            domain,
            admission,
            armed: true,
        };
        let registered = std::cell::RefCell::new(registered);
        let raw = RawScope {
            domain,
            stream,
            registered: &registered,
            admission,
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
