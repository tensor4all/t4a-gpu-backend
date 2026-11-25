//! Typed failures of the HIP runtime.

use t4a_hip_sys::Error as SysError;

/// Result alias used throughout this crate.
pub type Result<T> = std::result::Result<T, Error>;

/// A HIP runtime failure.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// A library this runtime needs cannot be loaded, or the host has no device.
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
            SysError::MissingSymbol { library, .. } => Error::Unavailable {
                what: library,
                detail: error.to_string(),
            },
            _ => Error::Unavailable {
                what: "HIP runtime",
                detail: error.to_string(),
            },
        }
    }
}
