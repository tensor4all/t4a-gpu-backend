//! A loader that can refuse to open a framework or resolve a symbol.
//!
//! Everything not forced to fail is resolved by the real dynamic loader, so a
//! table built through it has genuine `libobjc` entry points and a caller may
//! actually use it.

use std::ffi::{c_void, CStr};

use crate::error::{Error, Result};
use crate::loader::{Image, Loader, System};

/// A loader that injects the failures the design's absent-framework lane needs.
#[derive(Debug, Clone, Copy, Default)]
pub struct Faulty {
    /// Framework names whose `open` must fail.
    pub fail_open: &'static [&'static str],
    /// `(framework, symbol)` pairs whose resolution must fail.
    pub miss_symbol: &'static [(&'static str, &'static str)],
}

impl Loader for Faulty {
    unsafe fn open(&self, framework: &'static str, path: &CStr) -> Result<Image> {
        if self.fail_open.contains(&framework) {
            return Err(Error::framework(
                framework,
                "forced to fail by the test loader",
            ));
        }
        // SAFETY: forwarded; the real loader keeps the image loaded for the process.
        unsafe { System.open(framework, path) }
    }

    unsafe fn symbol(
        &self,
        framework: &'static str,
        image: Image,
        name: &CStr,
    ) -> Result<*mut c_void> {
        if let Some((_, symbol)) = self
            .miss_symbol
            .iter()
            .find(|(owner, symbol)| *owner == framework && name.to_bytes() == symbol.as_bytes())
        {
            return Err(Error::MissingSymbol { framework, symbol });
        }
        // SAFETY: forwarded; the real loader's addresses are valid for the process.
        unsafe { System.symbol(framework, image, name) }
    }
}
