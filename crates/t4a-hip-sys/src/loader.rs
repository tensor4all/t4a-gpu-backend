//! Dynamic loading of the ROCm shared libraries.
//!
//! Every library is opened with `RTLD_NOW | RTLD_LOCAL`, so no ROCm symbol enters the
//! process's global namespace (tenferro-rs#2489, M0 point 6: LLVM inside comgr must not
//! clash with LLVM inside NVRTC), and nothing is linked at build time. Opened libraries
//! are never closed: unloading a GPU runtime while any of its threads or allocations
//! are live is unsound, and one leaked image per library is the cheap correct choice.

use std::ffi::{c_void, OsStr, OsString};
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

/// HIP runtime sonames this crate accepts, newest first.
///
/// The unversioned `libamdhip64.so` is not a candidate: it is a development symlink that
/// may point at any major version, including the distro's HIP 5 that some hosts carry
/// next to a ROCm install.
pub const HIP_SONAMES: &[&str] = &["libamdhip64.so.7", "libamdhip64.so.6"];

/// rocBLAS sonames, newest first. They are only looked up next to the HIP runtime.
pub const ROCBLAS_SONAMES: &[&str] = &["librocblas.so.5", "librocblas.so.4"];

/// rocSOLVER sonames. They are only looked up next to the HIP runtime.
pub const ROCSOLVER_SONAMES: &[&str] = &["librocsolver.so.0"];

/// Where to look for the HIP runtime.
///
/// With `ROCM_PATH` set and non-empty, only `$ROCM_PATH/lib/<soname>` is tried, so a
/// caller that names a prefix never silently gets another one. Without it the bare
/// sonames are handed to the dynamic loader, which honours `LD_LIBRARY_PATH` and the
/// system's loader cache.
pub fn hip_candidates(rocm_path: Option<&OsStr>) -> Vec<PathBuf> {
    match rocm_path {
        Some(prefix) if !prefix.is_empty() => HIP_SONAMES
            .iter()
            .map(|soname| Path::new(prefix).join("lib").join(soname))
            .collect(),
        _ => HIP_SONAMES.iter().map(PathBuf::from).collect(),
    }
}

/// Candidates for a math library: only the directory the HIP runtime came from.
///
/// The math libraries must come from the same ROCm prefix as the runtime they link
/// against; a rocBLAS from another prefix would pull a second `libamdhip64` by soname.
pub fn sibling_candidates(hip_library: &Path, sonames: &[&str]) -> Vec<PathBuf> {
    let directory = hip_library.parent().unwrap_or_else(|| Path::new(""));
    sonames
        .iter()
        .map(|soname| directory.join(soname))
        .collect()
}

/// Distinct `libamdhip64` files named in a `/proc/<pid>/maps` listing.
///
/// Only the path column is read. Paths are returned as written; the caller decides
/// whether to canonicalise them.
pub fn hip_runtimes_in_maps(maps: &str) -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = Vec::new();
    for line in maps.lines() {
        // address perms offset dev inode [path]; the path is the sixth column and may
        // contain spaces, so split off the first five columns only.
        let mut columns = line.splitn(6, char::is_whitespace);
        let path = match (0..5).try_fold((), |_, _| columns.next().map(|_| ())) {
            Some(()) => columns.next().unwrap_or("").trim_start(),
            None => continue,
        };
        let is_hip = Path::new(path)
            .file_name()
            .and_then(OsStr::to_str)
            .is_some_and(|name| name.starts_with("libamdhip64.so"));
        if is_hip {
            let path = PathBuf::from(path);
            if !found.contains(&path) {
                found.push(path);
            }
        }
    }
    found
}

/// Every distinct HIP runtime file currently mapped into this process, canonicalised.
///
/// Empty when `/proc/self/maps` cannot be read (a non-Linux target).
pub fn mapped_hip_runtimes() -> Vec<PathBuf> {
    let Ok(maps) = std::fs::read_to_string("/proc/self/maps") else {
        return Vec::new();
    };
    let mut canonical: Vec<PathBuf> = Vec::new();
    for path in hip_runtimes_in_maps(&maps) {
        let path = std::fs::canonicalize(&path).unwrap_or(path);
        if !canonical.contains(&path) {
            canonical.push(path);
        }
    }
    canonical
}

/// Refuse a process that has more than one HIP runtime mapped.
///
/// # Errors
///
/// [`Error::DuplicateRuntime`] naming every distinct file.
pub fn ensure_single_hip_runtime() -> Result<()> {
    let paths = mapped_hip_runtimes();
    if paths.len() > 1 {
        return Err(Error::DuplicateRuntime { paths });
    }
    Ok(())
}

/// The state of `HSA_OVERRIDE_GFX_VERSION` as this process sees it.
///
/// The variable is read and reported, never set, unset or rewritten: the HSA runtime
/// reads it once at initialisation, and changing the environment of a multi-threaded
/// process is unsound.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GfxOverride {
    /// Not in the environment. The device runs as the ISA it reports.
    Unset,
    /// Exported with an empty value. On gfx1151 with rocBLAS 5.2.0 this makes
    /// `rocblas_create_handle` return `rocblas_status_invalid_handle`, which reads like a
    /// missing dtype or device and is neither.
    Empty,
    /// Exported with a value; the device runs under that override.
    Value(String),
}

impl GfxOverride {
    /// The environment variable this type describes.
    pub const VARIABLE: &'static str = "HSA_OVERRIDE_GFX_VERSION";

    /// Classify a raw environment value.
    pub fn from_value(value: Option<&OsStr>) -> GfxOverride {
        match value {
            None => GfxOverride::Unset,
            Some(value) if value.is_empty() => GfxOverride::Empty,
            Some(value) => GfxOverride::Value(value.to_string_lossy().into_owned()),
        }
    }

    /// Read the variable from this process's environment.
    pub fn current() -> GfxOverride {
        let value: Option<OsString> = std::env::var_os(Self::VARIABLE);
        GfxOverride::from_value(value.as_deref())
    }

    /// Refuse the one state that is known to break rocBLAS.
    ///
    /// # Errors
    ///
    /// [`Error::Configuration`] naming the variable when it is set but empty.
    pub fn check(&self) -> Result<()> {
        match self {
            GfxOverride::Empty => Err(Error::Configuration {
                variable: Self::VARIABLE,
                detail: "is set but empty; rocblas_create_handle then fails with \
                         rocblas_status_invalid_handle. Unset it (do not export an empty \
                         value)."
                    .to_string(),
            }),
            GfxOverride::Unset | GfxOverride::Value(_) => Ok(()),
        }
    }
}

/// One opened library, leaked for the life of the process.
#[derive(Debug)]
pub(crate) struct Library {
    library: &'static str,
    handle: *mut c_void,
    path: PathBuf,
}

// SAFETY: the handle is the address of a loaded image that is never closed; it carries
// no state of its own and is immutable for the life of the process.
unsafe impl Send for Library {}
unsafe impl Sync for Library {}

impl Library {
    /// The file the dynamic loader actually mapped.
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    /// Open the first loadable candidate.
    ///
    /// `anchor` is a symbol the library must export; its address is used to learn which
    /// file the loader mapped, because a bare soname says nothing about the prefix.
    ///
    /// # Errors
    ///
    /// [`Error::LibraryUnavailable`] when no candidate exists, listing every candidate
    /// and why it failed; [`Error::LibraryUnusable`] when one exists but cannot be
    /// loaded; and [`Error::MissingSymbol`] when the opened image lacks `anchor`.
    pub(crate) fn open(
        library: &'static str,
        candidates: &[PathBuf],
        anchor: &'static [u8],
    ) -> Result<Library> {
        let mut failures = Vec::new();
        let mut present = false;
        for candidate in candidates {
            match imp::open(candidate) {
                Ok(handle) => {
                    let mut opened = Library {
                        library,
                        handle,
                        path: candidate.clone(),
                    };
                    let address = opened.raw_symbol(anchor)?;
                    if let Some(path) = imp::object_path(address) {
                        opened.path = path;
                    }
                    return Ok(opened);
                }
                Err(detail) => {
                    present |= candidate_present(candidate, &detail);
                    failures.push(format!("{}: {detail}", candidate.display()));
                }
            }
        }
        if failures.is_empty() {
            failures.push("no candidate paths".to_string());
        }
        let detail = failures.join("; ");
        if present {
            return Err(Error::LibraryUnusable { library, detail });
        }
        Err(Error::unavailable(library, detail))
    }

    fn raw_symbol(&self, name: &'static [u8]) -> Result<*mut c_void> {
        imp::symbol(self.handle, name).ok_or(Error::MissingSymbol {
            library: self.library,
            symbol: symbol_name(name),
        })
    }

    /// Resolve `name` as a value of the function-pointer type `F`.
    ///
    /// # Errors
    ///
    /// [`Error::MissingSymbol`] when the library does not export `name`.
    ///
    /// # Safety
    ///
    /// `F` must be an `unsafe extern "C" fn` type matching the symbol's real C
    /// signature.
    pub(crate) unsafe fn function<F: Copy>(&self, name: &'static [u8]) -> Result<F> {
        assert_eq!(
            std::mem::size_of::<F>(),
            std::mem::size_of::<*mut c_void>(),
            "a symbol is resolved only as a function pointer"
        );
        let address = self.raw_symbol(name)?;
        // SAFETY: `F` is a function pointer type of pointer size (asserted above) and
        // the caller guarantees it matches the symbol's signature.
        Ok(unsafe { std::mem::transmute_copy::<*mut c_void, F>(&address) })
    }
}

/// Whether a candidate that failed to load is nevertheless there.
///
/// A path candidate is there when the directory entry exists, a dangling symlink
/// included. A bare soname is resolved by the dynamic loader, so its own message decides:
/// anything other than "No such file or directory" means it found a file it could not
/// load.
pub(crate) fn candidate_present(candidate: &Path, loader_message: &str) -> bool {
    let is_path = candidate
        .parent()
        .is_some_and(|parent| !parent.as_os_str().is_empty());
    if is_path {
        candidate.symlink_metadata().is_ok()
    } else {
        !loader_message.contains("No such file or directory")
    }
}

fn symbol_name(name: &'static [u8]) -> &'static str {
    let name = name.strip_suffix(b"\0").unwrap_or(name);
    std::str::from_utf8(name).unwrap_or("<non-UTF-8 symbol>")
}

#[cfg(target_os = "linux")]
mod imp {
    use std::ffi::{c_void, CStr, CString};
    use std::os::unix::ffi::OsStrExt;
    use std::path::{Path, PathBuf};

    pub(super) fn open(path: &Path) -> Result<*mut c_void, String> {
        let name = CString::new(path.as_os_str().as_bytes())
            .map_err(|_| "the path contains a NUL byte".to_string())?;
        // Clear any stale message so the one read below belongs to this call.
        // SAFETY: `dlerror` only reads and resets thread-local loader state.
        unsafe { libc::dlerror() };
        // SAFETY: `name` is NUL-terminated and the flags are plain integers. The image is
        // intentionally never closed.
        let handle = unsafe { libc::dlopen(name.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
        if handle.is_null() {
            return Err(last_dl_error());
        }
        Ok(handle)
    }

    pub(super) fn symbol(handle: *mut c_void, name: &'static [u8]) -> Option<*mut c_void> {
        let name = CStr::from_bytes_with_nul(name).ok()?;
        // SAFETY: `handle` was returned by `dlopen` and never closed; `name` is
        // NUL-terminated.
        let address = unsafe { libc::dlsym(handle, name.as_ptr()) };
        // Every symbol this crate resolves is a function, so null is a miss.
        (!address.is_null()).then_some(address)
    }

    pub(super) fn object_path(address: *mut c_void) -> Option<PathBuf> {
        let mut info = std::mem::MaybeUninit::<libc::Dl_info>::zeroed();
        // SAFETY: `address` is a resolved symbol address, and `info` is a writable
        // `Dl_info` that `dladdr` fills on success.
        let found = unsafe { libc::dladdr(address.cast_const(), info.as_mut_ptr()) };
        if found == 0 {
            return None;
        }
        // SAFETY: `dladdr` succeeded, so the structure is initialised.
        let info = unsafe { info.assume_init() };
        if info.dli_fname.is_null() {
            return None;
        }
        // SAFETY: non-null `dli_fname` is a NUL-terminated path owned by the loader for
        // as long as the object stays mapped, which is the life of the process.
        let name = unsafe { CStr::from_ptr(info.dli_fname) };
        Some(PathBuf::from(std::ffi::OsStr::from_bytes(name.to_bytes())))
    }

    fn last_dl_error() -> String {
        // SAFETY: `dlerror` returns a thread-local NUL-terminated string or null; it is
        // copied immediately.
        let message = unsafe { libc::dlerror() };
        if message.is_null() {
            return "the dynamic loader reported no detail".to_string();
        }
        // SAFETY: non-null means a NUL-terminated string owned by the loader.
        unsafe { CStr::from_ptr(message) }
            .to_string_lossy()
            .into_owned()
    }
}

#[cfg(not(target_os = "linux"))]
mod imp {
    use std::ffi::c_void;
    use std::path::{Path, PathBuf};

    pub(super) fn open(_path: &Path) -> Result<*mut c_void, String> {
        Err("HIP is only loaded on Linux".to_string())
    }

    pub(super) fn symbol(_handle: *mut c_void, _name: &'static [u8]) -> Option<*mut c_void> {
        None
    }

    pub(super) fn object_path(_address: *mut c_void) -> Option<PathBuf> {
        None
    }
}

#[cfg(test)]
mod tests;
