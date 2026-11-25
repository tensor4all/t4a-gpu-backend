//! Typed failures of the HIP runtime.

use t4a_hip_sys::Error as SysError;

/// Result alias used throughout this crate.
pub type Result<T> = std::result::Result<T, Error>;

/// A HIP runtime failure.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// A library this runtime needs is absent from the host.
    ///
    /// This is the only "not here" outcome: a library that loads but is unusable, or an
    /// environment that breaks it, has its own variant, so a caller (or a test) that
    /// treats absence as a skip never skips a broken setup.
    #[error("{what} is unavailable: {detail}")]
    Unavailable {
        /// What is missing.
        what: &'static str,
        /// The loader's or the runtime's own explanation.
        detail: String,
    },
    /// The environment carries a setting that is known to break the stack.
    #[error("{variable} {detail}")]
    Configuration {
        /// The environment variable at fault.
        variable: &'static str,
        /// What is wrong with it and what to do.
        detail: String,
    },
    /// A library is present but cannot be loaded (not an image, wrong architecture,
    /// dangling symlink, failing dependency): a broken install, not an absent one.
    #[error("{library} is present but unusable: {detail}")]
    LibraryUnusable {
        /// Library that was requested.
        library: &'static str,
        /// Every candidate that was tried, with the loader's message for each.
        detail: String,
    },
    /// A library loaded but lacks a symbol this runtime needs: a wrong or partial
    /// install, not an absent one.
    #[error("{library} is missing the required symbol `{symbol}`")]
    MissingSymbol {
        /// Library the symbol was looked up in.
        library: &'static str,
        /// Symbol name.
        symbol: &'static str,
    },
    /// More than one HIP runtime file is mapped into this process (for example a
    /// preloaded distro `libamdhip64.so.5` next to the ROCm one). Two runtimes have
    /// separate device state, so nothing is created in either.
    #[error("more than one HIP runtime is mapped into this process: {paths:?}")]
    DuplicateRuntime {
        /// Every distinct `libamdhip64` file found in the process's mappings.
        paths: Vec<std::path::PathBuf>,
    },
    /// The ordinal does not name a visible device.
    #[error("HIP device {ordinal} does not exist ({count} visible)")]
    InvalidDevice {
        /// Requested ordinal.
        ordinal: usize,
        /// Devices HIP reports.
        count: usize,
    },
    /// The device lacks something this runtime's contract depends on.
    #[error("unsupported: {detail}")]
    Unsupported {
        /// What is missing.
        detail: String,
    },
    /// A HIP call returned an error.
    #[error("{call} failed: {name} ({code})")]
    Hip {
        /// The HIP function.
        call: &'static str,
        /// Its `hipError_t`.
        code: i32,
        /// `hipGetErrorName` for the code.
        name: String,
    },
    /// A resource of another runtime was passed to this one.
    #[error("the resource belongs to another runtime")]
    ForeignDomain,
    /// An argument the runtime cannot accept.
    #[error("invalid argument: {detail}")]
    InvalidArgument {
        /// What is wrong.
        detail: String,
    },
}

impl Error {
    pub(crate) fn invalid(detail: impl Into<String>) -> Error {
        Error::InvalidArgument {
            detail: detail.into(),
        }
    }
}

impl From<SysError> for Error {
    fn from(error: SysError) -> Error {
        match &error {
            SysError::Configuration { variable, detail } => Error::Configuration {
                variable,
                detail: detail.clone(),
            },
            SysError::LibraryUnavailable { library, detail } => Error::Unavailable {
                what: library,
                detail: detail.clone(),
            },
            SysError::LibraryUnusable { library, detail } => Error::LibraryUnusable {
                library,
                detail: detail.clone(),
            },
            SysError::MissingSymbol { library, symbol } => Error::MissingSymbol { library, symbol },
            SysError::DuplicateRuntime { paths } => Error::DuplicateRuntime {
                paths: paths.clone(),
            },
            _ => Error::Unavailable {
                what: "HIP runtime",
                detail: error.to_string(),
            },
        }
    }
}
