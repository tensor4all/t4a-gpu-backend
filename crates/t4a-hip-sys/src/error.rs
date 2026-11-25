//! Errors reported while loading the HIP runtime and the ROCm math libraries.
//!
//! Nothing here is a capability decision: the loader reports what it could not open,
//! resolve or accept, and stops. Deciding what a missing library means belongs to the
//! consumer.

use std::path::PathBuf;

/// Result alias used throughout this crate.
pub type Result<T> = std::result::Result<T, Error>;

/// A library, symbol or environment setting prevented a table from loading.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// No candidate path for a library could be opened.
    #[error("{library} is unavailable: {detail}")]
    LibraryUnavailable {
        /// Library that was requested.
        library: &'static str,
        /// Every candidate that was tried, with the dynamic loader's message for each.
        detail: String,
    },
    /// A library was opened but does not export a required symbol.
    #[error("{library} is missing the required symbol `{symbol}`")]
    MissingSymbol {
        /// Library the symbol was looked up in.
        library: &'static str,
        /// Symbol name.
        symbol: &'static str,
    },
    /// The environment carries a setting that is known to break the stack.
    ///
    /// Reported before any library is opened, because the HSA runtime reads these
    /// variables once at initialisation.
    #[error("{variable} {detail}")]
    Configuration {
        /// The environment variable at fault.
        variable: &'static str,
        /// What is wrong with it and what to do.
        detail: String,
    },
    /// More than one HIP runtime image is mapped into this process.
    ///
    /// rocBLAS and rocSOLVER resolve `libamdhip64` by soname. When that soname resolves to
    /// a different file than the one this crate opened, two HIP runtimes with separate
    /// device state live in one process, and streams or allocations from one are invalid
    /// in the other.
    #[error("more than one HIP runtime is mapped into this process: {paths:?}")]
    DuplicateRuntime {
        /// Every distinct `libamdhip64` file found in the process's mappings.
        paths: Vec<PathBuf>,
    },
}

impl Error {
    pub(crate) fn unavailable(library: &'static str, detail: impl Into<String>) -> Error {
        Error::LibraryUnavailable {
            library,
            detail: detail.into(),
        }
    }
}
