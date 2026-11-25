//! The rocBLAS and rocSOLVER function tables.
//!
//! Both libraries are opened from the directory the HIP runtime was loaded from, never
//! by a bare soname, and the process is checked for a second `libamdhip64` right after
//! each one is opened: both libraries name `libamdhip64.so.7` as `NEEDED`, and a soname
//! that resolved to another file would put two HIP runtimes in one process.

use std::ffi::{c_char, c_void, CStr};
use std::path::Path;
use std::sync::OnceLock;

use crate::error::Result;
use crate::hip::{HipStream, HipTable};
use crate::loader::{
    ensure_single_hip_runtime, sibling_candidates, Library, ROCBLAS_SONAMES, ROCSOLVER_SONAMES,
};

/// `rocblas_handle`.
pub type RocblasHandle = *mut c_void;
/// `rocblas_status`.
pub type RocblasStatus = i32;
/// `rocblas_int` in the LP64 build ROCm ships.
pub type RocblasInt = i32;
/// `rocblas_stride`.
pub type RocblasStride = i64;

/// `rocblas_status_success`.
pub const ROCBLAS_STATUS_SUCCESS: RocblasStatus = 0;
/// `rocblas_status_invalid_handle`.
pub const ROCBLAS_STATUS_INVALID_HANDLE: RocblasStatus = 1;
/// `rocblas_status_memory_error`: in user-owned workspace mode, the workspace is too
/// small for the call.
pub const ROCBLAS_STATUS_MEMORY_ERROR: RocblasStatus = 5;
/// `rocblas_status_size_increased`: a size query recorded a larger workspace.
pub const ROCBLAS_STATUS_SIZE_INCREASED: RocblasStatus = 9;
/// `rocblas_status_size_unchanged`: a size query recorded no larger workspace.
pub const ROCBLAS_STATUS_SIZE_UNCHANGED: RocblasStatus = 10;
/// `rocblas_operation_none`.
pub const ROCBLAS_OPERATION_NONE: i32 = 111;
/// `rocblas_operation_transpose`.
pub const ROCBLAS_OPERATION_TRANSPOSE: i32 = 112;
/// `rocblas_pointer_mode_host`: scalar arguments are host addresses.
pub const ROCBLAS_POINTER_MODE_HOST: i32 = 0;
/// `rocblas_pointer_mode_device`: scalar arguments are device addresses.
pub const ROCBLAS_POINTER_MODE_DEVICE: i32 = 1;

/// The C signature shared by `rocblas_{s,d,c,z}gemm`.
///
/// The four entries differ only in the pointee type of their scalar and matrix
/// pointers, which this table erases to `c_void`; the caller passes pointers of the
/// dtype the entry expects.
pub type GemmFn = unsafe extern "C" fn(
    RocblasHandle,
    i32,
    i32,
    RocblasInt,
    RocblasInt,
    RocblasInt,
    *const c_void,
    *const c_void,
    RocblasInt,
    *const c_void,
    RocblasInt,
    *const c_void,
    *mut c_void,
    RocblasInt,
) -> RocblasStatus;

/// The C signature shared by `rocblas_{s,d,c,z}gemm_strided_batched`.
pub type GemmStridedBatchedFn = unsafe extern "C" fn(
    RocblasHandle,
    i32,
    i32,
    RocblasInt,
    RocblasInt,
    RocblasInt,
    *const c_void,
    *const c_void,
    RocblasInt,
    RocblasStride,
    *const c_void,
    RocblasInt,
    RocblasStride,
    *const c_void,
    *mut c_void,
    RocblasInt,
    RocblasStride,
    RocblasInt,
) -> RocblasStatus;

/// The C signature shared by `rocsolver_{s,d}getrf`.
///
/// `ipiv` and `info` are **device** pointers: a rocSOLVER kernel writes them. A host
/// address there makes the kernel page-fault and the process wait forever instead of
/// returning an error.
pub type GetrfFn = unsafe extern "C" fn(
    RocblasHandle,
    RocblasInt,
    RocblasInt,
    *mut c_void,
    RocblasInt,
    *mut RocblasInt,
    *mut RocblasInt,
) -> RocblasStatus;

/// The rocBLAS entry points this repository uses.
#[allow(missing_docs)]
#[derive(Debug)]
pub struct RocblasTable {
    library: Library,
    pub create_handle: unsafe extern "C" fn(*mut RocblasHandle) -> RocblasStatus,
    pub destroy_handle: unsafe extern "C" fn(RocblasHandle) -> RocblasStatus,
    pub set_stream: unsafe extern "C" fn(RocblasHandle, HipStream) -> RocblasStatus,
    pub get_stream: unsafe extern "C" fn(RocblasHandle, *mut HipStream) -> RocblasStatus,
    pub set_pointer_mode: unsafe extern "C" fn(RocblasHandle, i32) -> RocblasStatus,
    pub get_pointer_mode: unsafe extern "C" fn(RocblasHandle, *mut i32) -> RocblasStatus,
    pub set_workspace: unsafe extern "C" fn(RocblasHandle, *mut c_void, usize) -> RocblasStatus,
    pub is_managing_device_memory: unsafe extern "C" fn(RocblasHandle) -> bool,
    pub get_device_memory_size: unsafe extern "C" fn(RocblasHandle, *mut usize) -> RocblasStatus,
    pub start_device_memory_size_query: unsafe extern "C" fn(RocblasHandle) -> RocblasStatus,
    pub stop_device_memory_size_query:
        unsafe extern "C" fn(RocblasHandle, *mut usize) -> RocblasStatus,
    pub get_version_string_size: unsafe extern "C" fn(*mut usize) -> RocblasStatus,
    pub get_version_string: unsafe extern "C" fn(*mut c_char, usize) -> RocblasStatus,
    pub status_to_string: unsafe extern "C" fn(RocblasStatus) -> *const c_char,
    pub sgemm: GemmFn,
    pub dgemm: GemmFn,
    pub cgemm: GemmFn,
    pub zgemm: GemmFn,
    pub sgemm_strided_batched: GemmStridedBatchedFn,
    pub dgemm_strided_batched: GemmStridedBatchedFn,
    pub cgemm_strided_batched: GemmStridedBatchedFn,
    pub zgemm_strided_batched: GemmStridedBatchedFn,
}

static ROCBLAS: OnceLock<Result<RocblasTable>> = OnceLock::new();

impl RocblasTable {
    /// Load rocBLAS from the HIP runtime's directory, once per process.
    ///
    /// # Errors
    ///
    /// Whatever [`HipTable::load`] reports, [`crate::Error::LibraryUnavailable`],
    /// [`crate::Error::MissingSymbol`], or [`crate::Error::DuplicateRuntime`] when
    /// opening rocBLAS mapped a second `libamdhip64`. The first outcome is cached.
    pub fn load() -> Result<&'static RocblasTable> {
        ROCBLAS
            .get_or_init(|| {
                let hip = HipTable::load()?;
                let candidates = sibling_candidates(hip.library_path(), ROCBLAS_SONAMES);
                let library = Library::open("rocBLAS", &candidates, b"rocblas_create_handle\0")?;
                ensure_single_hip_runtime()?;
                // SAFETY: each type below is the C signature of the rocBLAS function
                // named; the four GEMM widths share one erased signature (`GemmFn`).
                unsafe {
                    Ok(RocblasTable {
                        create_handle: library.function(b"rocblas_create_handle\0")?,
                        destroy_handle: library.function(b"rocblas_destroy_handle\0")?,
                        set_stream: library.function(b"rocblas_set_stream\0")?,
                        get_stream: library.function(b"rocblas_get_stream\0")?,
                        set_pointer_mode: library.function(b"rocblas_set_pointer_mode\0")?,
                        get_pointer_mode: library.function(b"rocblas_get_pointer_mode\0")?,
                        set_workspace: library.function(b"rocblas_set_workspace\0")?,
                        is_managing_device_memory: library
                            .function(b"rocblas_is_managing_device_memory\0")?,
                        get_device_memory_size: library
                            .function(b"rocblas_get_device_memory_size\0")?,
                        start_device_memory_size_query: library
                            .function(b"rocblas_start_device_memory_size_query\0")?,
                        stop_device_memory_size_query: library
                            .function(b"rocblas_stop_device_memory_size_query\0")?,
                        get_version_string_size: library
                            .function(b"rocblas_get_version_string_size\0")?,
                        get_version_string: library.function(b"rocblas_get_version_string\0")?,
                        status_to_string: library.function(b"rocblas_status_to_string\0")?,
                        sgemm: library.function(b"rocblas_sgemm\0")?,
                        dgemm: library.function(b"rocblas_dgemm\0")?,
                        cgemm: library.function(b"rocblas_cgemm\0")?,
                        zgemm: library.function(b"rocblas_zgemm\0")?,
                        sgemm_strided_batched: library
                            .function(b"rocblas_sgemm_strided_batched\0")?,
                        dgemm_strided_batched: library
                            .function(b"rocblas_dgemm_strided_batched\0")?,
                        cgemm_strided_batched: library
                            .function(b"rocblas_cgemm_strided_batched\0")?,
                        zgemm_strided_batched: library
                            .function(b"rocblas_zgemm_strided_batched\0")?,
                        library,
                    })
                }
            })
            .as_ref()
            .map_err(Clone::clone)
    }

    /// The `librocblas` file the dynamic loader mapped.
    pub fn library_path(&self) -> &Path {
        self.library.path()
    }

    /// `rocblas_get_version_string`, for example `5.2.0.70204-93`.
    pub fn version(&self) -> String {
        let mut size = 0usize;
        // SAFETY: `size` is a writable `size_t`.
        let status = unsafe { (self.get_version_string_size)(&mut size) };
        if status != ROCBLAS_STATUS_SUCCESS || size == 0 {
            return String::new();
        }
        let mut buffer = vec![0u8; size];
        // SAFETY: `buffer` has the `size` writable bytes the library asked for.
        let status = unsafe { (self.get_version_string)(buffer.as_mut_ptr().cast(), size) };
        if status != ROCBLAS_STATUS_SUCCESS {
            return String::new();
        }
        let end = buffer.iter().position(|&b| b == 0).unwrap_or(buffer.len());
        String::from_utf8_lossy(&buffer[..end]).into_owned()
    }

    /// `rocblas_status_to_string` for `status`.
    pub fn status_name(&self, status: RocblasStatus) -> String {
        // SAFETY: the function accepts any value and returns a static string or null.
        let name = unsafe { (self.status_to_string)(status) };
        if name.is_null() {
            return format!("rocblas_status {status}");
        }
        // SAFETY: non-null is a static NUL-terminated string.
        unsafe { CStr::from_ptr(name) }
            .to_string_lossy()
            .into_owned()
    }
}

/// The rocSOLVER entry points this repository uses.
///
/// rocSOLVER runs on a rocBLAS handle; it has no handle of its own.
#[allow(missing_docs)]
#[derive(Debug)]
pub struct RocsolverTable {
    library: Library,
    pub sgetrf: GetrfFn,
    pub dgetrf: GetrfFn,
}

static ROCSOLVER: OnceLock<Result<RocsolverTable>> = OnceLock::new();

impl RocsolverTable {
    /// Load rocSOLVER from the HIP runtime's directory, once per process.
    ///
    /// # Errors
    ///
    /// As [`RocblasTable::load`].
    pub fn load() -> Result<&'static RocsolverTable> {
        ROCSOLVER
            .get_or_init(|| {
                RocblasTable::load()?;
                let hip = HipTable::load()?;
                let candidates = sibling_candidates(hip.library_path(), ROCSOLVER_SONAMES);
                let library = Library::open("rocSOLVER", &candidates, b"rocsolver_dgetrf\0")?;
                ensure_single_hip_runtime()?;
                // SAFETY: both entries have the `GetrfFn` signature.
                unsafe {
                    Ok(RocsolverTable {
                        sgetrf: library.function(b"rocsolver_sgetrf\0")?,
                        dgetrf: library.function(b"rocsolver_dgetrf\0")?,
                        library,
                    })
                }
            })
            .as_ref()
            .map_err(Clone::clone)
    }

    /// The `librocsolver` file the dynamic loader mapped.
    pub fn library_path(&self) -> &Path {
        self.library.path()
    }
}
