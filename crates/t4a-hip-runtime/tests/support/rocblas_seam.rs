//! The rocBLAS vendor seam: the ROCm mirror of tenferro-gpu's `Cublas` + `GemmRequest`
//! (`crates/tenferro-gpu/src/cuda/vendor.rs` in tenferro-rs), bound to a
//! `t4a-hip-runtime` stream and pool.
//!
//! It lives in the test target on purpose (tenferro-rs#2489 stage 1: "a test crate
//! holds the seam"): like cuBLAS, which sits in tenferro-gpu rather than in
//! `t4a-cuda-runtime`, the vendor adapter belongs to the consumer, and stage 2 moves it
//! into `tenferro-gpu/src/rocm/`. Stage 1 proves that the runtime's ownership contract
//! carries a vendor library:
//!
//! - the handle is created inside a raw scope, which makes the runtime's device current,
//!   and bound to the runtime's stream; the stream binding and the pointer mode are read
//!   back (`rocblas_get_stream`, `rocblas_get_pointer_mode`);
//! - **pointer classes are explicit**: GEMM coefficients are host scalars, so the handle's
//!   pointer mode is set to host and re-read before every call; rocSOLVER's `info` and
//!   `ipiv` are device pointers, so the typed entry only takes runtime buffers and the
//!   raw entry checks each pointer's class with HIP before rocSOLVER sees it;
//! - **workspace is pool-owned**: the handle gets a pool buffer at creation (which makes
//!   rocBLAS free the 32 MiB it allocates for itself), every call's workspace is
//!   measured with rocBLAS's size query first, and a larger need is met by a new pool
//!   buffer. `rocblas_is_managing_device_memory` is checked before and after every call:
//!   a handle that ever reports managing its own memory again is refused, so no call runs
//!   with memory the pool cannot account for.

#![allow(dead_code)]

use std::ffi::c_void;

use t4a_hip_runtime::sys::{
    GfxOverride, RocblasHandle, RocblasTable, RocsolverTable, ROCBLAS_OPERATION_NONE,
    ROCBLAS_OPERATION_TRANSPOSE, ROCBLAS_POINTER_MODE_HOST, ROCBLAS_STATUS_SIZE_INCREASED,
    ROCBLAS_STATUS_SIZE_UNCHANGED, ROCBLAS_STATUS_SUCCESS,
};
use t4a_hip_runtime::{Buffer, DomainAnchor, Error, PointerClass, RawScope, Runtime};

/// Element type of a GEMM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    /// `f32`.
    F32,
    /// `f64`.
    F64,
    /// Complex `f32` (`rocblas_float_complex`: two `f32`, 4-byte aligned).
    C32,
    /// Complex `f64` (`rocblas_double_complex`: two `f64`, 8-byte aligned).
    C64,
}

impl DType {
    /// Every dtype the seam serves.
    pub const ALL: [DType; 4] = [DType::F32, DType::F64, DType::C32, DType::C64];

    /// Bytes per element.
    pub fn size(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F64 | DType::C32 => 8,
            DType::C64 => 16,
        }
    }

    /// Whether the dtype is complex.
    pub fn is_complex(self) -> bool {
        matches!(self, DType::C32 | DType::C64)
    }

    /// Machine epsilon of the dtype's real component.
    pub fn epsilon(self) -> f64 {
        match self {
            DType::F32 | DType::C32 => f64::from(f32::EPSILON),
            DType::F64 | DType::C64 => f64::EPSILON,
        }
    }

    /// Floating-point operations of one `(m, n, k)` product.
    pub fn flops(self, m: f64, n: f64, k: f64) -> f64 {
        let per_fma = if self.is_complex() { 8.0 } else { 2.0 };
        per_fma * m * n * k
    }
}

/// A seam failure, shaped like tenferro-gpu's `GpuError` for the cases the CUDA seam
/// tests observe.
#[derive(Debug)]
pub enum SeamError {
    /// The library or handle cannot be used at all.
    Unavailable {
        /// Why.
        detail: String,
    },
    /// The request cannot be served as asked.
    Unsupported {
        /// Operation.
        op: &'static str,
        /// Why.
        detail: String,
    },
    /// A pointer argument is not of the class the vendor entry requires.
    ///
    /// Reported before the vendor call, because rocSOLVER does not report a wrong class:
    /// its kernel faults on the address and the process waits forever.
    PointerClass {
        /// Operation.
        op: &'static str,
        /// Argument name in the vendor signature.
        argument: &'static str,
        /// What the vendor entry requires.
        expected: &'static str,
        /// What HIP says the address is.
        found: PointerClass,
    },
    /// The runtime refused the call.
    Runtime(Error),
}

impl From<Error> for SeamError {
    fn from(error: Error) -> SeamError {
        SeamError::Runtime(error)
    }
}

impl SeamError {
    fn unsupported(op: &'static str, detail: impl Into<String>) -> SeamError {
        SeamError::Unsupported {
            op,
            detail: detail.into(),
        }
    }
}

/// One GEMM request over runtime allocations, in column-major order.
///
/// Field for field the CUDA seam's `GemmRequest`, with the seam's own [`DType`].
#[derive(Debug)]
pub struct GemmRequest<'a> {
    /// Dtype of all three operands.
    pub dtype: DType,
    /// Left operand.
    pub a: &'a Buffer,
    /// Whether `a` is transposed.
    pub trans_a: bool,
    /// Right operand.
    pub b: &'a Buffer,
    /// Whether `b` is transposed.
    pub trans_b: bool,
    /// Destination.
    pub c: &'a Buffer,
    /// `(m, n, k)` of the product.
    pub shape: (i64, i64, i64),
    /// Leading dimensions `(lda, ldb, ldc)` in elements.
    pub leading: (i64, i64, i64),
    /// `(batches, stride_a, stride_b, stride_c)` in elements; `batches = 1` is a plain
    /// GEMM.
    pub batches: (i64, i64, i64, i64),
}

/// A rocSOLVER `info` slot: one `rocblas_int` of **device** memory.
///
/// The type exists so the pointer class is a property of the argument, not of the
/// caller's care: it can only be made from the runtime's pool.
pub struct InfoSlot {
    buffer: Buffer,
}

impl InfoSlot {
    /// Allocate a zeroed slot from `runtime`'s pool.
    pub fn new(runtime: &mut Runtime) -> Result<InfoSlot, SeamError> {
        Ok(InfoSlot {
            buffer: runtime.alloc_zeroed(std::mem::size_of::<i32>())?,
        })
    }

    /// Read the slot back, after every queued operation.
    pub fn read(&self, runtime: &mut Runtime) -> Result<i32, SeamError> {
        let bytes = runtime.download(&self.buffer)?;
        let mut word = [0u8; 4];
        word.copy_from_slice(&bytes);
        Ok(i32::from_ne_bytes(word))
    }

    /// The underlying buffer.
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }
}

/// A rocBLAS handle bound to one runtime's stream, with a pool-owned workspace.
pub struct Rocblas {
    blas: &'static RocblasTable,
    handle: RocblasHandle,
    anchor: DomainAnchor,
    workspace: Buffer,
    /// Set when the handle's pointer mode could not be proven host; such a handle is
    /// refused rather than allowed to read stack coefficients as device addresses.
    mode_unproven: bool,
    /// What rocBLAS held of its own when the handle was created, before the seam handed
    /// it a pool buffer.
    library_memory_at_creation: LibraryMemory,
    /// The largest workspace any call through this handle asked for, in bytes.
    largest_query: usize,
}

/// Device memory a rocBLAS handle manages itself, as `rocblas_is_managing_device_memory`
/// and `rocblas_get_device_memory_size` report it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LibraryMemory {
    /// Whether rocBLAS manages the handle's device memory.
    pub managed: bool,
    /// The handle's device-memory size in bytes.
    pub bytes: usize,
}

impl Rocblas {
    /// Workspace handed to the handle at creation. Any non-empty pool buffer switches
    /// rocBLAS to user-owned memory; calls that need more get a larger pool buffer.
    pub const INITIAL_WORKSPACE: usize = 4096;

    /// Create the handle on `runtime`'s device, set host pointer mode, bind the
    /// runtime's stream and hand it a pool-owned workspace. Each setting is read back.
    pub fn new(runtime: &mut Runtime) -> Result<Rocblas, SeamError> {
        let blas = RocblasTable::load().map_err(|error| SeamError::Unavailable {
            detail: error.to_string(),
        })?;
        let anchor = runtime.anchor();
        // SAFETY: the closure only creates and configures a handle; the workspace it
        // hands over is a pool buffer kept alive by the returned `Rocblas`.
        let created = unsafe {
            runtime.with_raw(
                &[],
                |raw| -> Result<(RocblasHandle, Buffer, LibraryMemory), SeamError> {
                    let mut handle: RocblasHandle = std::ptr::null_mut();
                    let status = (blas.create_handle)(&mut handle);
                    if status != ROCBLAS_STATUS_SUCCESS {
                        return Err(SeamError::Unavailable {
                            detail: format!(
                                "rocblas_create_handle failed: {} (HSA_OVERRIDE_GFX_VERSION: {:?})",
                                blas.status_name(status),
                                GfxOverride::current()
                            ),
                        });
                    }
                    match configure(blas, raw, handle) {
                        Ok((workspace, before)) => Ok((handle, workspace, before)),
                        Err(error) => {
                            (blas.destroy_handle)(handle);
                            Err(error)
                        }
                    }
                },
            )?
        };
        let (handle, workspace, library_memory_at_creation) = created?;
        Ok(Rocblas {
            blas,
            handle,
            anchor,
            workspace,
            mode_unproven: false,
            library_memory_at_creation,
            largest_query: 0,
        })
    }

    /// What rocBLAS managed itself when the handle was created, before the seam handed
    /// over a pool buffer.
    pub fn library_memory_at_creation(&self) -> LibraryMemory {
        self.library_memory_at_creation
    }

    /// The handle's device-memory size as rocBLAS reports it now: the pool buffer's
    /// size once the seam owns the workspace.
    pub fn library_memory_now(&self) -> LibraryMemory {
        library_memory(self.blas, self.handle)
    }

    /// Hand device-memory management back to rocBLAS (`rocblas_set_workspace(null, 0)`),
    /// for the test that checks the seam then refuses the handle.
    pub fn return_memory_to_rocblas_for_test(&mut self) {
        // SAFETY: the handle is live; null and zero switch it back to rocBLAS-managed
        // memory, which is the state under test.
        unsafe { (self.blas.set_workspace)(self.handle, std::ptr::null_mut(), 0) };
    }

    /// The largest workspace, in bytes, that rocBLAS's size query reported for any call
    /// made through this handle.
    pub fn largest_query_bytes(&self) -> usize {
        self.largest_query
    }

    /// Bytes of the pool buffer the handle currently uses as workspace.
    pub fn workspace_bytes(&self) -> usize {
        self.workspace.len()
    }

    /// Whether rocBLAS manages device memory of its own for this handle. Always
    /// `false` for a handle this seam created.
    pub fn manages_own_memory(&self) -> bool {
        // SAFETY: the handle is alive.
        unsafe { (self.blas.is_managing_device_memory)(self.handle) }
    }

    /// The handle's stream, as rocBLAS reports it.
    pub fn bound_stream(&self) -> *mut c_void {
        let mut stream: *mut c_void = std::ptr::null_mut();
        // SAFETY: the handle is alive and `stream` is writable.
        unsafe { (self.blas.get_stream)(self.handle, &mut stream) };
        stream
    }

    /// Refuse a handle whose pointer mode is not provably host.
    ///
    /// The GEMM entries pass their coefficients as host stack addresses; a handle in
    /// device mode would read them as device addresses.
    fn ensure_host_pointer_mode(&mut self) -> Result<(), SeamError> {
        let mut mode = -1;
        // SAFETY: the handle is alive and `mode` is writable.
        let status = unsafe { (self.blas.get_pointer_mode)(self.handle, &mut mode) };
        if self.mode_unproven
            || status != ROCBLAS_STATUS_SUCCESS
            || mode != ROCBLAS_POINTER_MODE_HOST
        {
            self.mode_unproven = true;
            return Err(SeamError::Unavailable {
                detail: format!(
                    "the rocBLAS handle's pointer mode is not provably host \
                     (status {}, mode {mode}); the handle must be recreated",
                    self.blas.status_name(status)
                ),
            });
        }
        Ok(())
    }

    fn ensure_own_runtime(&self, runtime: &Runtime) -> Result<(), SeamError> {
        if runtime.domain_id() != self.anchor.domain_id() {
            return Err(SeamError::Runtime(Error::ForeignDomain));
        }
        Ok(())
    }

    /// One dense or strided-batched GEMM in column-major order, `c = a * b`.
    ///
    /// # Errors
    ///
    /// [`SeamError::Unsupported`] for a layout the seam does not take (a non-positive
    /// extent, a leading dimension below the operand's row count, an operand buffer
    /// too small for the extents, leading dimension and strides, or an extent, leading
    /// dimension or batch count above `i32::MAX`), and the library's status otherwise.
    pub fn gemm(&mut self, runtime: &mut Runtime, call: &GemmRequest<'_>) -> Result<(), SeamError> {
        self.ensure_own_runtime(runtime)?;
        self.ensure_host_pointer_mode()?;
        let (m, n, k) = call.shape;
        let (lda, ldb, ldc) = call.leading;
        let (batches, stride_a, stride_b, stride_c) = call.batches;
        let a_rows = if call.trans_a { k } else { m };
        let a_cols = if call.trans_a { m } else { k };
        let b_rows = if call.trans_b { n } else { k };
        let b_cols = if call.trans_b { k } else { n };
        if m <= 0 || n <= 0 || k <= 0 || batches <= 0 {
            return Err(SeamError::unsupported("gemm", "extents must be positive"));
        }
        if lda < a_rows || ldb < b_rows || ldc < m {
            return Err(SeamError::unsupported(
                "gemm",
                "a leading dimension is below the operand's row count",
            ));
        }
        let m32 = narrow_i32("gemm", "m", m)?;
        let n32 = narrow_i32("gemm", "n", n)?;
        let k32 = narrow_i32("gemm", "k", k)?;
        let lda32 = narrow_i32("gemm", "lda", lda)?;
        let ldb32 = narrow_i32("gemm", "ldb", ldb)?;
        let ldc32 = narrow_i32("gemm", "ldc", ldc)?;
        let batches32 = narrow_i32("gemm", "batch count", batches)?;
        // The CUDA seam trusts the caller's buffers here; this one checks that every
        // element the library will touch lies inside its buffer.
        let element = call.dtype.size();
        fits(
            "gemm", "a", call.a, element, a_rows, a_cols, lda, batches, stride_a,
        )?;
        fits(
            "gemm", "b", call.b, element, b_rows, b_cols, ldb, batches, stride_b,
        )?;
        fits("gemm", "c", call.c, element, m, n, ldc, batches, stride_c)?;
        let trans_a = operation(call.trans_a);
        let trans_b = operation(call.trans_b);
        let blas = self.blas;
        let handle = self.handle;
        let dtype = call.dtype;
        let workspace = &mut self.workspace;
        let largest_query = &mut self.largest_query;
        // SAFETY: the scope registered all three operands, every element the call reads
        // or writes lies inside them (checked above), the workspace is a pool buffer kept
        // alive by `self`, and the call is queued on the runtime's stream.
        unsafe {
            runtime.with_raw(&[call.a, call.b, call.c], |raw| -> Result<(), SeamError> {
                let a = raw.device_ptr(call.a)?;
                let b = raw.device_ptr(call.b)?;
                let c = raw.device_ptr(call.c)?;
                let invoke = || {
                    let coefficients = Coefficients::new(dtype);
                    if batches32 == 1 {
                        let entry = match dtype {
                            DType::F32 => blas.sgemm,
                            DType::F64 => blas.dgemm,
                            DType::C32 => blas.cgemm,
                            DType::C64 => blas.zgemm,
                        };
                        entry(
                            handle,
                            trans_a,
                            trans_b,
                            m32,
                            n32,
                            k32,
                            coefficients.alpha(),
                            a,
                            lda32,
                            b,
                            ldb32,
                            coefficients.beta(),
                            c,
                            ldc32,
                        )
                    } else {
                        let entry = match dtype {
                            DType::F32 => blas.sgemm_strided_batched,
                            DType::F64 => blas.dgemm_strided_batched,
                            DType::C32 => blas.cgemm_strided_batched,
                            DType::C64 => blas.zgemm_strided_batched,
                        };
                        entry(
                            handle,
                            trans_a,
                            trans_b,
                            m32,
                            n32,
                            k32,
                            coefficients.alpha(),
                            a,
                            lda32,
                            stride_a,
                            b,
                            ldb32,
                            stride_b,
                            coefficients.beta(),
                            c,
                            ldc32,
                            stride_c,
                            batches32,
                        )
                    }
                };
                let needed = ensure_workspace(blas, handle, raw, workspace, &invoke)?;
                *largest_query = (*largest_query).max(needed);
                let status = invoke();
                ensure_pool_owned(blas, handle)?;
                if status != ROCBLAS_STATUS_SUCCESS {
                    return Err(SeamError::unsupported(
                        "gemm",
                        format!("rocBLAS reported {}", blas.status_name(status)),
                    ));
                }
                Ok(())
            })?
        }
    }

    /// LU factorisation through rocSOLVER, with `info` and `ipiv` typed as device memory.
    ///
    /// `info` receives rocSOLVER's status: 0, or the 1-based index of the first zero
    /// pivot.
    #[allow(clippy::too_many_arguments)]
    pub fn getrf(
        &mut self,
        runtime: &mut Runtime,
        dtype: DType,
        n: i64,
        a: &Buffer,
        lda: i64,
        ipiv: &Buffer,
        info: &InfoSlot,
    ) -> Result<(), SeamError> {
        self.ensure_own_runtime(runtime)?;
        if n <= 0 || lda < n {
            return Err(SeamError::unsupported(
                "getrf",
                "n must be positive and lda >= n",
            ));
        }
        let element = dtype.size();
        fits("getrf", "a", a, element, n, n, lda, 1, 0)?;
        if ipiv.len() < n as usize * std::mem::size_of::<i32>() {
            return Err(SeamError::unsupported(
                "getrf",
                "ipiv holds fewer than n entries",
            ));
        }
        let info_buffer = info.buffer();
        // SAFETY: the three buffers are registered runtime allocations of the sizes
        // checked above; `getrf_raw` queues on the runtime's stream.
        unsafe {
            runtime.with_raw(&[a, ipiv, info_buffer], |raw| -> Result<(), SeamError> {
                let a = raw.device_ptr(a)?;
                let ipiv = raw.device_ptr(ipiv)?.cast::<i32>();
                let info = raw.device_ptr(info_buffer)?.cast::<i32>();
                self.getrf_raw(raw, dtype, n, a, lda, ipiv, info)
            })?
        }
    }

    /// The raw rocSOLVER entry: pointers instead of buffers.
    ///
    /// Every pointer's class is checked with HIP before rocSOLVER sees it. A host
    /// address for `info` (the mistake that cost a day on #2489: `&mut info` on the
    /// stack) is refused with [`SeamError::PointerClass`] instead of reaching
    /// `rocsolver::reset_info`, which would page-fault on it and leave the process
    /// waiting forever.
    ///
    /// # Safety
    ///
    /// `a` must address an `n x n` matrix with leading dimension `lda` of `dtype`,
    /// `ipiv` `n` and `info` one `rocblas_int`, all of them allocations of the scope's
    /// runtime that outlive the queued work.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn getrf_raw(
        &mut self,
        raw: &RawScope<'_>,
        dtype: DType,
        n: i64,
        a: *mut c_void,
        lda: i64,
        ipiv: *mut i32,
        info: *mut i32,
    ) -> Result<(), SeamError> {
        if raw.domain_id() != self.anchor.domain_id() {
            return Err(SeamError::Runtime(Error::ForeignDomain));
        }
        require_device(raw, "getrf", "info", info.cast_const().cast())?;
        require_device(raw, "getrf", "ipiv", ipiv.cast_const().cast())?;
        require_device(raw, "getrf", "A", a.cast_const())?;
        let solver = RocsolverTable::load().map_err(|error| SeamError::Unavailable {
            detail: error.to_string(),
        })?;
        let entry = match dtype {
            DType::F32 => solver.sgetrf,
            DType::F64 => solver.dgetrf,
            other => {
                return Err(SeamError::unsupported(
                    "getrf",
                    format!("{other:?} is not bound in this seam"),
                ))
            }
        };
        let n32 = narrow_i32("getrf", "n", n)?;
        let lda32 = narrow_i32("getrf", "lda", lda)?;
        let handle = self.handle;
        // SAFETY: pointers were classified as device memory above and the caller
        // guarantees their extents.
        let invoke = || unsafe { entry(handle, n32, n32, a, lda32, ipiv, info) };
        let needed = ensure_workspace(self.blas, handle, raw, &mut self.workspace, &invoke)?;
        self.largest_query = self.largest_query.max(needed);
        let status = invoke();
        ensure_pool_owned(self.blas, handle)?;
        if status != ROCBLAS_STATUS_SUCCESS {
            return Err(SeamError::unsupported(
                "getrf",
                format!("rocSOLVER reported {}", self.blas.status_name(status)),
            ));
        }
        Ok(())
    }
}

impl Drop for Rocblas {
    fn drop(&mut self) {
        // Queued calls may still use the handle's internal state; drain the stream on
        // the handle's device first. The workspace buffer drops after this, and its
        // free is ordered on the same stream anyway.
        if self.anchor.activate().is_ok() {
            let _ = self.anchor.synchronize();
            // SAFETY: the handle is alive and no queued work uses it after the drain.
            unsafe { (self.blas.destroy_handle)(self.handle) };
        }
    }
}

/// Configure a fresh handle: host pointer mode, the runtime's stream, a pool-owned
/// workspace. Each setting is read back.
///
/// # Safety
///
/// `handle` must be a live handle created on the scope's device.
unsafe fn configure(
    blas: &'static RocblasTable,
    raw: &RawScope<'_>,
    handle: RocblasHandle,
) -> Result<(Buffer, LibraryMemory), SeamError> {
    let failed = |what: &str, status| SeamError::Unavailable {
        detail: format!("{what} failed: {}", blas.status_name(status)),
    };
    // SAFETY (all calls below): `handle` is live per the contract; out-parameters are
    // writable locals.
    let status = unsafe { (blas.set_pointer_mode)(handle, ROCBLAS_POINTER_MODE_HOST) };
    if status != ROCBLAS_STATUS_SUCCESS {
        return Err(failed("rocblas_set_pointer_mode", status));
    }
    let mut mode = -1;
    let status = unsafe { (blas.get_pointer_mode)(handle, &mut mode) };
    if status != ROCBLAS_STATUS_SUCCESS || mode != ROCBLAS_POINTER_MODE_HOST {
        return Err(SeamError::Unavailable {
            detail: format!("the handle did not keep host pointer mode (mode {mode})"),
        });
    }
    let status = unsafe { (blas.set_stream)(handle, raw.stream()) };
    if status != ROCBLAS_STATUS_SUCCESS {
        return Err(failed("rocblas_set_stream", status));
    }
    let mut bound: *mut c_void = std::ptr::null_mut();
    let status = unsafe { (blas.get_stream)(handle, &mut bound) };
    if status != ROCBLAS_STATUS_SUCCESS || bound != raw.stream() {
        return Err(SeamError::Unavailable {
            detail: "the handle is not bound to the runtime's stream".to_string(),
        });
    }
    let before = library_memory(blas, handle);
    let workspace = raw.alloc_zeroed(Rocblas::INITIAL_WORKSPACE)?;
    let pointer = raw.device_ptr(&workspace)?;
    let status = unsafe { (blas.set_workspace)(handle, pointer, workspace.len()) };
    if status != ROCBLAS_STATUS_SUCCESS {
        return Err(failed("rocblas_set_workspace", status));
    }
    if unsafe { (blas.is_managing_device_memory)(handle) } {
        return Err(SeamError::Unavailable {
            detail: "rocBLAS still manages device memory of its own after \
                     rocblas_set_workspace; the pool cannot account for it"
                .to_string(),
        });
    }
    Ok((workspace, before))
}

/// Read a handle's own device-memory state.
fn library_memory(blas: &'static RocblasTable, handle: RocblasHandle) -> LibraryMemory {
    let mut bytes = 0usize;
    // SAFETY: the handle is live and `bytes` is writable.
    let status = unsafe { (blas.get_device_memory_size)(handle, &mut bytes) };
    LibraryMemory {
        // SAFETY: the handle is live.
        managed: unsafe { (blas.is_managing_device_memory)(handle) },
        bytes: if status == ROCBLAS_STATUS_SUCCESS {
            bytes
        } else {
            0
        },
    }
}

/// Refuse a handle on which rocBLAS manages device memory of its own: such memory is
/// outside the runtime's pool and its accounting.
fn ensure_pool_owned(blas: &'static RocblasTable, handle: RocblasHandle) -> Result<(), SeamError> {
    // SAFETY: the handle is live.
    if unsafe { (blas.is_managing_device_memory)(handle) } {
        return Err(SeamError::Unavailable {
            detail: "rocBLAS manages device memory of its own on this handle; the pool \
                     cannot account for it, so the handle is refused"
                .to_string(),
        });
    }
    Ok(())
}

/// Measure what `invoke` needs with rocBLAS's size query, and give the handle a larger
/// pool buffer when the current one is too small.
///
/// The replaced buffer is dropped after the handle has switched; its free is queued on
/// the runtime's stream, after every call that used it. Returns the queried size.
fn ensure_workspace(
    blas: &'static RocblasTable,
    handle: RocblasHandle,
    raw: &RawScope<'_>,
    workspace: &mut Buffer,
    invoke: &dyn Fn() -> i32,
) -> Result<usize, SeamError> {
    ensure_pool_owned(blas, handle)?;
    // SAFETY: the handle is live; while the query is on, rocBLAS and rocSOLVER entries
    // only record their workspace size and launch nothing.
    let status = unsafe { (blas.start_device_memory_size_query)(handle) };
    if status != ROCBLAS_STATUS_SUCCESS {
        return Err(SeamError::Unavailable {
            detail: format!("size query start: {}", blas.status_name(status)),
        });
    }
    let queried = invoke();
    let mut needed = 0usize;
    // SAFETY: as above; `needed` is writable.
    let stopped = unsafe { (blas.stop_device_memory_size_query)(handle, &mut needed) };
    if queried != ROCBLAS_STATUS_SIZE_UNCHANGED
        && queried != ROCBLAS_STATUS_SIZE_INCREASED
        && queried != ROCBLAS_STATUS_SUCCESS
    {
        return Err(SeamError::Unsupported {
            op: "workspace query",
            detail: format!("the size query reported {}", blas.status_name(queried)),
        });
    }
    if stopped != ROCBLAS_STATUS_SUCCESS {
        return Err(SeamError::Unavailable {
            detail: format!("size query stop: {}", blas.status_name(stopped)),
        });
    }
    if needed > workspace.len() {
        let larger = raw.alloc_zeroed(needed)?;
        let pointer = raw.device_ptr(&larger)?;
        // SAFETY: `larger` is a live pool buffer of `needed` bytes on this stream.
        let status = unsafe { (blas.set_workspace)(handle, pointer, larger.len()) };
        if status != ROCBLAS_STATUS_SUCCESS {
            return Err(SeamError::Unavailable {
                detail: format!("rocblas_set_workspace: {}", blas.status_name(status)),
            });
        }
        *workspace = larger;
        ensure_pool_owned(blas, handle)?;
    }
    Ok(needed)
}

/// Refuse a pointer that HIP does not classify as device memory of the scope's device.
pub fn require_device(
    raw: &RawScope<'_>,
    op: &'static str,
    argument: &'static str,
    address: *const c_void,
) -> Result<(), SeamError> {
    match raw.pointer_class(address)? {
        PointerClass::Device { ordinal } if ordinal == raw.ordinal() => Ok(()),
        found => Err(SeamError::PointerClass {
            op,
            argument,
            expected: "device memory of the runtime's device",
            found,
        }),
    }
}

/// Host-pointer-mode coefficients `alpha = 1`, `beta = 0` in the dtype's layout.
///
/// Complex values are `[re, im]` arrays, which have the 4-byte (`f32`) and 8-byte
/// (`f64`) alignment of `rocblas_float_complex` and `rocblas_double_complex`.
struct Coefficients {
    alpha: [f64; 2],
    beta: [f64; 2],
    alpha32: [f32; 2],
    beta32: [f32; 2],
    dtype: DType,
}

impl Coefficients {
    fn new(dtype: DType) -> Coefficients {
        Coefficients {
            alpha: [1.0, 0.0],
            beta: [0.0, 0.0],
            alpha32: [1.0, 0.0],
            beta32: [0.0, 0.0],
            dtype,
        }
    }

    fn alpha(&self) -> *const c_void {
        match self.dtype {
            DType::F32 | DType::C32 => self.alpha32.as_ptr().cast(),
            DType::F64 | DType::C64 => self.alpha.as_ptr().cast(),
        }
    }

    fn beta(&self) -> *const c_void {
        match self.dtype {
            DType::F32 | DType::C32 => self.beta32.as_ptr().cast(),
            DType::F64 | DType::C64 => self.beta.as_ptr().cast(),
        }
    }
}

fn operation(transposed: bool) -> i32 {
    if transposed {
        ROCBLAS_OPERATION_TRANSPOSE
    } else {
        ROCBLAS_OPERATION_NONE
    }
}

/// Narrow one rocBLAS size argument to `rocblas_int`.
///
/// An `as` cast above `i32::MAX` wraps silently and would run a GEMM of a different size
/// (tenferro-rs#2424 on the CUDA side), so the value is refused with its field and value.
pub fn narrow_i32(op: &'static str, field: &'static str, value: i64) -> Result<i32, SeamError> {
    i32::try_from(value).map_err(|_| {
        SeamError::unsupported(
            op,
            format!("{field} = {value} does not fit the i32 range of the rocBLAS entry points"),
        )
    })
}

/// Check that a column-major `rows x cols` operand with leading dimension `ld`, repeated
/// `batches` times at `stride`, lies inside `buffer`.
#[allow(clippy::too_many_arguments)]
fn fits(
    op: &'static str,
    name: &'static str,
    buffer: &Buffer,
    element: usize,
    rows: i64,
    cols: i64,
    ld: i64,
    batches: i64,
    stride: i64,
) -> Result<(), SeamError> {
    let last = (|| {
        let one = (cols - 1).checked_mul(ld)?.checked_add(rows)?;
        let span = if batches > 1 {
            if stride < 0 {
                return None;
            }
            (batches - 1).checked_mul(stride)?.checked_add(one)?
        } else {
            one
        };
        usize::try_from(span).ok()?.checked_mul(element)
    })();
    match last {
        Some(bytes) if bytes <= buffer.len() => Ok(()),
        Some(bytes) => Err(SeamError::unsupported(
            op,
            format!(
                "operand {name} needs {bytes} bytes for its extents, leading dimension \
                 and strides; the buffer holds {}",
                buffer.len()
            ),
        )),
        None => Err(SeamError::unsupported(
            op,
            format!("operand {name}'s extent overflows or its batch stride is negative"),
        )),
    }
}
