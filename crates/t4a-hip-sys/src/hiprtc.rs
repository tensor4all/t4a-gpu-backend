//! The hiprtc function table: run-time compilation of HIP C++ to a code object.
//!
//! `libhiprtc` is opened from the directory the HIP runtime was loaded from with local
//! symbol scope (`RTLD_NOW | RTLD_LOCAL`), exactly like rocBLAS, so that it can live in
//! one process next to NVRTC without either library's symbols resolving into the other.

use std::ffi::{c_char, c_int, c_void, CStr};
use std::path::Path;
use std::sync::OnceLock;

use crate::error::Result;
use crate::hip::HipTable;
use crate::loader::{ensure_single_hip_runtime, sibling_candidates, Library, HIPRTC_SONAMES};

/// `hiprtcProgram`.
pub type HiprtcProgram = *mut c_void;
/// `hiprtcResult`; zero is success.
pub type HiprtcResult = i32;
/// `HIPRTC_SUCCESS`.
pub const HIPRTC_SUCCESS: HiprtcResult = 0;

/// `hiprtcCreateProgram(prog, src, name, numHeaders, headers, includeNames)`.
pub type HiprtcCreateProgramFn = unsafe extern "C" fn(
    *mut HiprtcProgram,
    *const c_char,
    *const c_char,
    c_int,
    *const *const c_char,
    *const *const c_char,
) -> HiprtcResult;

/// The hiprtc entry points this repository uses.
#[allow(missing_docs)]
#[derive(Debug)]
pub struct HiprtcTable {
    library: Library,
    pub create_program: HiprtcCreateProgramFn,
    pub compile_program:
        unsafe extern "C" fn(HiprtcProgram, c_int, *const *const c_char) -> HiprtcResult,
    pub get_code_size: unsafe extern "C" fn(HiprtcProgram, *mut usize) -> HiprtcResult,
    pub get_code: unsafe extern "C" fn(HiprtcProgram, *mut c_char) -> HiprtcResult,
    pub get_program_log_size: unsafe extern "C" fn(HiprtcProgram, *mut usize) -> HiprtcResult,
    pub get_program_log: unsafe extern "C" fn(HiprtcProgram, *mut c_char) -> HiprtcResult,
    pub destroy_program: unsafe extern "C" fn(*mut HiprtcProgram) -> HiprtcResult,
    pub get_error_string: unsafe extern "C" fn(HiprtcResult) -> *const c_char,
}

static HIPRTC: OnceLock<Result<HiprtcTable>> = OnceLock::new();

impl HiprtcTable {
    /// Load hiprtc from the HIP runtime's directory, once per process.
    ///
    /// # Errors
    ///
    /// Whatever [`HipTable::load`] reports, [`crate::Error::LibraryUnavailable`],
    /// [`crate::Error::MissingSymbol`], or [`crate::Error::DuplicateRuntime`]. The first
    /// outcome is cached.
    pub fn load() -> Result<&'static HiprtcTable> {
        HIPRTC
            .get_or_init(|| {
                let hip = HipTable::load()?;
                let candidates = sibling_candidates(hip.library_path(), HIPRTC_SONAMES);
                let library = Library::open("hiprtc", &candidates, b"hiprtcCreateProgram\0")?;
                ensure_single_hip_runtime()?;
                // SAFETY: each type below is the C signature of the hiprtc function named.
                unsafe {
                    Ok(HiprtcTable {
                        create_program: library.function(b"hiprtcCreateProgram\0")?,
                        compile_program: library.function(b"hiprtcCompileProgram\0")?,
                        get_code_size: library.function(b"hiprtcGetCodeSize\0")?,
                        get_code: library.function(b"hiprtcGetCode\0")?,
                        get_program_log_size: library.function(b"hiprtcGetProgramLogSize\0")?,
                        get_program_log: library.function(b"hiprtcGetProgramLog\0")?,
                        destroy_program: library.function(b"hiprtcDestroyProgram\0")?,
                        get_error_string: library.function(b"hiprtcGetErrorString\0")?,
                        library,
                    })
                }
            })
            .as_ref()
            .map_err(Clone::clone)
    }

    /// The `libhiprtc` file the dynamic loader mapped.
    pub fn library_path(&self) -> &Path {
        self.library.path()
    }

    /// `hiprtcGetErrorString` for a result code.
    pub fn error_string(&self, code: HiprtcResult) -> String {
        // SAFETY: hiprtc returns a static NUL-terminated string.
        let name = unsafe { (self.get_error_string)(code) };
        if name.is_null() {
            return format!("hiprtc result {code}");
        }
        // SAFETY: non-null, NUL-terminated, static.
        unsafe { CStr::from_ptr(name) }
            .to_string_lossy()
            .into_owned()
    }
}
