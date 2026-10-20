//! Errors returned by the CUDA runtime.
//!
//! Every failure keeps its typed cause: an unavailable driver, an unsupported
//! device capability, an invalid argument, a compilation diagnostic, a CUDA
//! driver error with its original source, or an asynchronous failure observed
//! later. Errors are not collapsed into a single internal variant.

use cudarc::driver::result::DriverError;
use cudarc::driver::sys;

/// Result alias used throughout this crate.
pub type Result<T> = std::result::Result<T, Error>;

/// A typed runtime failure.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The CUDA driver library could not be loaded or initialised.
    #[error("CUDA driver is unavailable: {detail}")]
    DriverUnavailable {
        /// What was probed and why it failed.
        detail: String,
    },
    /// The NVRTC library could not be loaded.
    #[error("NVRTC is unavailable: {detail}")]
    NvrtcUnavailable {
        /// What was probed and why it failed.
        detail: String,
    },
    /// The requested ordinal does not name an existing device.
    #[error("invalid CUDA device ordinal {ordinal} (device count {count})")]
    InvalidDevice {
        /// Requested ordinal.
        ordinal: usize,
        /// Devices visible to the driver.
        count: usize,
    },
    /// The device or driver lacks a capability this runtime requires.
    #[error("unsupported capability: {detail}")]
    Unsupported {
        /// The capability and the observation that it is missing.
        detail: String,
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
    /// The device or pool could not satisfy an allocation.
    #[error("out of device memory while requesting {bytes} bytes")]
    OutOfMemory {
        /// Requested size in bytes.
        bytes: usize,
    },
    /// NVRTC rejected the program.
    #[error("kernel compilation failed: {detail}")]
    Compile {
        /// Compiler diagnostics.
        detail: String,
    },
    /// A CUDA driver call failed synchronously.
    #[error("{context} failed: {source}")]
    Driver {
        /// The driver entry point that failed.
        context: &'static str,
        /// Original driver error.
        #[source]
        source: DriverError,
    },
    /// Work that was already submitted failed.
    #[error("asynchronous CUDA failure: {detail}")]
    Async {
        /// The observation that reported the failure.
        detail: String,
    },
    /// A previous failure made this execution owner unusable.
    #[error("runtime is unusable after an earlier failure: {detail}")]
    Broken {
        /// The original failure.
        detail: String,
    },
    /// Admission cannot proceed within the configured pending budget.
    #[error("insufficient pending-resource budget for {bytes} bytes")]
    Backpressure {
        /// Bytes the new admission needs.
        bytes: usize,
    },
    /// The buffer belongs to another allocation domain.
    #[error("buffer belongs to a different allocation domain")]
    ForeignDomain,
    /// The runtime has already been closed.
    #[error("the runtime has been closed")]
    Closed,
    /// The module does not export the requested entry point.
    #[error("unknown entry point `{name}`")]
    UnknownEntryPoint {
        /// Requested entry point.
        name: String,
    },
}

impl Error {
    /// Wrap a driver error with the entry point that produced it.
    pub(crate) fn driver(context: &'static str) -> impl FnOnce(DriverError) -> Error {
        move |source| Error::from_driver(context, source)
    }

    /// Wrap a driver error, classifying out-of-memory separately.
    pub(crate) fn from_driver(context: &'static str, source: DriverError) -> Error {
        if source.0 == sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY {
            Error::OutOfMemory { bytes: 0 }
        } else {
            Error::Driver { context, source }
        }
    }

    /// Wrap a driver error that was caused by an allocation of `bytes`.
    pub(crate) fn from_alloc(bytes: usize, context: &'static str, source: DriverError) -> Error {
        match Error::from_driver(context, source) {
            Error::OutOfMemory { .. } => Error::OutOfMemory { bytes },
            other => other,
        }
    }

    pub(crate) fn invalid(detail: impl Into<String>) -> Error {
        Error::InvalidArgument {
            detail: detail.into(),
        }
    }

    pub(crate) fn unsupported(detail: impl Into<String>) -> Error {
        Error::Unsupported {
            detail: detail.into(),
        }
    }
}
