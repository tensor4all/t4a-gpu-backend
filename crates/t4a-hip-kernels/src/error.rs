//! Typed failures of the HIP kernel layer.

use t4a_hip_runtime::Error as RuntimeError;

/// Result alias used throughout this crate.
pub type Result<T> = std::result::Result<T, Error>;

/// A failure to compile, load, plan or launch a kernel.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The HIP runtime, hiprtc, or a HIP call failed. `Unavailable` inside means the
    /// host has no ROCm, which a caller may treat as a skip; nothing else may be.
    #[error(transparent)]
    Runtime(#[from] RuntimeError),
    /// hiprtc refused the source for this target. Carries the target, the exact options
    /// and hiprtc's own log: a target hiprtc cannot build for is a finding, not a retry.
    #[error("hiprtc could not compile `{module}` for `{target}` ({result}); options {options:?}; log:\n{log}")]
    Compile {
        /// Module name.
        module: String,
        /// The `--offload-arch` value that was passed.
        target: String,
        /// The options hiprtc was given.
        options: Vec<String>,
        /// `hiprtcGetErrorString` of the failing call.
        result: String,
        /// `hiprtcGetProgramLog`.
        log: String,
    },
    /// The shared CUDA crate's host planning rejected the request.
    #[error("launch planning rejected the request: {0}")]
    Plan(t4a_cuda_kernels::Error),
    /// A buffer is too small for the planned element count.
    #[error("buffer `{which}` holds {have} bytes but the launch needs {need}")]
    BufferTooSmall {
        /// Argument name.
        which: &'static str,
        /// Bytes in the buffer.
        have: usize,
        /// Bytes the launch reads or writes.
        need: usize,
    },
    /// The program does not export the symbol.
    #[error("the program has no entry `{symbol}`: {detail}")]
    NoSuchEntry {
        /// Kernel symbol.
        symbol: String,
        /// HIP's explanation.
        detail: String,
    },
    /// The planned launch does not fit what this layer can launch.
    #[error("unsupported launch: {detail}")]
    Unsupported {
        /// What is unsupported.
        detail: String,
    },
}

impl From<t4a_hip_sys::Error> for Error {
    fn from(error: t4a_hip_sys::Error) -> Error {
        Error::Runtime(error.into())
    }
}

impl From<t4a_cuda_kernels::Error> for Error {
    fn from(error: t4a_cuda_kernels::Error) -> Error {
        Error::Plan(error)
    }
}
