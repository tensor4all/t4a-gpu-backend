//! Errors returned by the Metal runtime.
//!
//! Every failure keeps its typed cause: an absent device or symbol, a compiler
//! diagnostic, a pipeline that could not be built, a caller mistake, a failing
//! command buffer, or an observation that never arrived. Nothing is collapsed into
//! a single internal variant, and nothing panics because Metal is missing.

use t4a_metal_sys::Error as SysError;

/// Result alias used throughout this crate.
pub type Result<T> = std::result::Result<T, Error>;

/// A typed runtime failure.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The Metal backend cannot be used on this host.
    #[error("Metal is unavailable: {what}")]
    Unavailable {
        /// What was probed and found missing.
        what: String,
        /// The loader's or the OS's own detail.
        detail: String,
    },
    /// Metal's MSL compiler rejected the source.
    #[error("kernel compilation failed: {detail}")]
    Compile {
        /// Compiler diagnostics.
        detail: String,
    },
    /// The entry point compiled but no pipeline state could be built from it.
    #[error("no pipeline state for `{entry}`: {detail}")]
    Pipeline {
        /// Requested entry point.
        entry: String,
        /// Metal's diagnostic.
        detail: String,
    },
    /// The library does not export the requested entry point.
    #[error("unknown entry point `{name}` in this library")]
    UnknownEntryPoint {
        /// Requested entry point.
        name: String,
    },
    /// A caller-supplied value violates the documented contract.
    #[error("invalid argument: {detail}")]
    InvalidArgument {
        /// Which argument and why.
        detail: String,
    },
    /// A span does not fit inside the buffer it addresses.
    #[error("span of {required} bytes at offset {offset} exceeds buffer length {len}")]
    OutOfRange {
        /// Buffer length in bytes.
        len: usize,
        /// Requested offset in bytes.
        offset: usize,
        /// Requested span in bytes.
        required: usize,
    },
    /// The device could not satisfy an allocation.
    #[error("out of device memory while requesting {bytes} bytes")]
    OutOfMemory {
        /// Requested size in bytes.
        bytes: usize,
    },
    /// One by-value parameter exceeds what `setBytes:length:atIndex:` accepts.
    #[error("kernel parameter of {bytes} bytes, above the {limit}-byte limit")]
    KernelParameterLimit {
        /// Requested parameter size in bytes.
        bytes: usize,
        /// The limit this runtime enforces.
        limit: usize,
    },
    /// The buffer or function belongs to another runtime's allocation domain.
    #[error("buffer or function belongs to a different allocation domain")]
    ForeignDomain,
    /// The runtime has already been closed.
    #[error("the runtime has been closed")]
    Closed,
    /// Work that was already submitted failed.
    #[error("asynchronous Metal failure: {detail}")]
    Async {
        /// What the command buffer reported.
        detail: String,
    },
    /// A submission did not complete within the configured timeout.
    #[error("a submission did not complete within {millis} ms")]
    Timeout {
        /// The timeout that elapsed, in milliseconds.
        millis: u64,
    },
}

impl Error {
    pub(crate) fn invalid(detail: impl Into<String>) -> Error {
        Error::InvalidArgument {
            detail: detail.into(),
        }
    }

    pub(crate) fn unavailable(what: impl Into<String>, detail: impl Into<String>) -> Error {
        Error::Unavailable {
            what: what.into(),
            detail: detail.into(),
        }
    }

    pub(crate) fn async_failure(detail: impl Into<String>) -> Error {
        Error::Async {
            detail: detail.into(),
        }
    }
}

#[cfg(test)]
mod tests;

impl From<SysError> for Error {
    /// A table that cannot be loaded means the backend cannot start.
    fn from(error: SysError) -> Self {
        match error {
            SysError::FrameworkUnavailable { framework, detail } => Error::Unavailable {
                what: format!("{framework} could not be opened"),
                detail,
            },
            SysError::MissingSymbol { framework, symbol } => Error::Unavailable {
                what: format!("{framework} does not export `{symbol}`"),
                detail: format!("the required symbol `{symbol}` was not resolved"),
            },
            SysError::InvalidName { name } => Error::Unavailable {
                what: format!("`{name}` is not a valid framework or selector name"),
                detail: "the loader rejected a name".to_string(),
            },
            other => Error::Unavailable {
                what: "the Metal table could not be loaded".to_string(),
                detail: other.to_string(),
            },
        }
    }
}
