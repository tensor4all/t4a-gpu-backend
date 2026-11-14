//! Errors reported while loading the Apple GPU frameworks.
//!
//! Nothing here is a capability decision: the loader reports what it could not
//! open or resolve and stops. Deciding that a missing framework means "this
//! backend is unavailable" belongs to the consumer.

use std::ffi::CStr;

/// Result alias used throughout this crate.
pub type Result<T> = std::result::Result<T, Error>;

/// A framework, image or symbol could not be obtained.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// A framework image could not be opened.
    #[error("{framework} is unavailable: {detail}")]
    FrameworkUnavailable {
        /// Framework that was requested.
        framework: &'static str,
        /// What the dynamic loader reported.
        detail: String,
    },
    /// An image was opened but does not export a required symbol.
    #[error("{framework} is missing the required symbol `{symbol}`")]
    MissingSymbol {
        /// Framework the symbol was looked up in.
        framework: &'static str,
        /// Symbol name.
        symbol: &'static str,
    },
    /// A name was not a valid NUL-terminated byte string.
    ///
    /// Every name this crate looks up is a literal, so this reports a defect in
    /// the table rather than a property of the host.
    #[error("`{name:?}` is not a valid NUL-terminated name")]
    InvalidName {
        /// The offending name, lossily decoded for the message.
        name: String,
    },
}

impl Error {
    pub(crate) fn invalid_name(bytes: &[u8]) -> Error {
        Error::InvalidName {
            name: String::from_utf8_lossy(bytes).into_owned(),
        }
    }

    pub(crate) fn framework(framework: &'static str, detail: impl Into<String>) -> Error {
        Error::FrameworkUnavailable {
            framework,
            detail: detail.into(),
        }
    }
}

/// Convert a byte literal ending in one NUL into a `CStr`.
pub(crate) fn cstr(bytes: &'static [u8]) -> Result<&'static CStr> {
    CStr::from_bytes_with_nul(bytes).map_err(|_| Error::invalid_name(bytes))
}
