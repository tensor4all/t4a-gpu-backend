//! Dynamic loading of framework images.
//!
//! The tables are opened on first use with `dlopen`/`dlsym`, so no framework is
//! linked at build time. Images are never closed (see [`Image`]).

use std::ffi::{c_void, CStr};

use crate::error::{Error, Result};

/// An opened framework image.
///
/// # Invariant
///
/// The image is leaked for the life of the process and never handed to
/// `dlclose`: a system framework whose Objective-C classes are registered cannot
/// be safely unloaded while any of those classes, or instances of them, are
/// reachable. One leaked image per process is the cheap correct alternative.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Image(*mut c_void);

impl Image {
    /// The process-wide address of the opened image.
    pub fn handle(self) -> *mut c_void {
        self.0
    }
}

// SAFETY: the handle is the address of a loaded image; it carries no state and is
// immutable for the life of the process.
unsafe impl Send for Image {}
unsafe impl Sync for Image {}

/// Where framework images and symbols come from.
///
/// # Safety
///
/// An implementation must return handles and symbol addresses that are valid for
/// the life of the process, because the tables built from them are used
/// unsynchronised from any thread.
pub trait Loader {
    /// Open `path` and report the failure against `framework`.
    ///
    /// # Errors
    ///
    /// [`Error::FrameworkUnavailable`] when the image cannot be opened, carrying the
    /// dynamic loader's own message.
    ///
    /// # Safety
    ///
    /// The returned image must stay loaded for the life of the process.
    unsafe fn open(&self, framework: &'static str, path: &CStr) -> Result<Image>;

    /// Resolve `name` in `image`.
    ///
    /// # Errors
    ///
    /// [`Error::MissingSymbol`] when the image does not export `name`.
    ///
    /// # Safety
    ///
    /// The returned address must be the symbol's real entry point, called with
    /// the C signature the caller casts it to.
    unsafe fn symbol(
        &self,
        framework: &'static str,
        image: Image,
        name: &CStr,
    ) -> Result<*mut c_void>;
}

/// The process's dynamic loader (`dlopen`/`dlsym`).
#[derive(Debug, Clone, Copy, Default)]
pub struct System;

#[cfg(target_os = "macos")]
impl Loader for System {
    unsafe fn open(&self, framework: &'static str, path: &CStr) -> Result<Image> {
        // SAFETY: `path` is a NUL-terminated name, and the flags are plain
        // integers. The image is intentionally never closed.
        let handle = unsafe { libc::dlopen(path.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
        if handle.is_null() {
            return Err(Error::framework(framework, last_dl_error()));
        }
        Ok(Image(handle))
    }

    unsafe fn symbol(
        &self,
        framework: &'static str,
        image: Image,
        name: &CStr,
    ) -> Result<*mut c_void> {
        // SAFETY: `image` is a handle this module opened and never closed, and
        // `name` is NUL-terminated.
        let symbol = unsafe { libc::dlsym(image.0, name.as_ptr()) };
        if symbol.is_null() {
            // `dlsym` may legitimately return null for a symbol whose value is 0;
            // every symbol this crate resolves is a function, so null is a miss.
            return Err(Error::MissingSymbol {
                framework,
                symbol: leak_name(name),
            });
        }
        Ok(symbol)
    }
}

#[cfg(target_os = "macos")]
fn last_dl_error() -> String {
    // SAFETY: `dlerror` returns a thread-local NUL-terminated string or null. It
    // is read immediately and not retained.
    let message = unsafe { libc::dlerror() };
    if message.is_null() {
        return "the dynamic loader reported no detail".to_string();
    }
    // SAFETY: non-null means a NUL-terminated string owned by the loader.
    unsafe { CStr::from_ptr(message) }
        .to_string_lossy()
        .into_owned()
}

#[cfg(target_os = "macos")]
fn leak_name(name: &CStr) -> &'static str {
    // Cheap: this runs only on a failed load, once per name at most.
    Box::leak(name.to_string_lossy().into_owned().into_boxed_str())
}

#[cfg(not(target_os = "macos"))]
impl Loader for System {
    unsafe fn open(&self, framework: &'static str, _path: &CStr) -> Result<Image> {
        Err(Error::framework(
            framework,
            "this target has no Apple frameworks",
        ))
    }

    unsafe fn symbol(
        &self,
        framework: &'static str,
        _image: Image,
        _name: &CStr,
    ) -> Result<*mut c_void> {
        Err(Error::framework(
            framework,
            "this target has no Apple frameworks",
        ))
    }
}

#[cfg(test)]
mod tests;
