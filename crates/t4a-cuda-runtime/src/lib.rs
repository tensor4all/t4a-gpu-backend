//! A standalone CUDA execution runtime.
//!
//! `t4a-cuda-runtime` owns one real CUDA stream, one explicitly created memory
//! pool, one allocation domain, ordered host transfers and the retirement of
//! resources that CUDA's stream-ordered allocator does not cover. It depends on
//! no tensor, graph, AD, CPU-provider, CubeCL/CubeK or vendor-library crate.
//!
//! # Design
//!
//! - One [`Runtime`] is one execution owner: submission takes `&mut Runtime` on
//!   the calling thread. There is no worker thread and no ambient current stream.
//! - [`Buffer`] handles carry no submission authority. Their last `Drop` enqueues
//!   `cuMemFreeAsync` on the owning stream, which is already ordered after every
//!   admitted use.
//! - Host and device never exchange data implicitly; every transfer is explicit
//!   and ordered.
//! - Opening the same ordinal twice yields two independent runtimes with
//!   different streams, pools and allocation-domain identities. They may share
//!   the device's primary context and its sticky errors.
//!
//! # Example
//!
//! ```no_run
//! use t4a_cuda_runtime::{Config, Runtime};
//!
//! # fn main() -> Result<(), t4a_cuda_runtime::Error> {
//! let mut runtime = Runtime::open(0, Config::default())?;
//! let buffer = runtime.upload(&[1u8, 2, 3, 4])?;
//! assert_eq!(buffer.len(), 4);
//! let mut pending = runtime.download(&buffer)?;
//! assert_eq!(pending.wait()?, vec![1, 2, 3, 4]);
//! runtime.close()?;
//! # Ok(())
//! # }
//! ```
//!
//! The example is marked `no_run` because it requires a CUDA device. The
//! equivalent assertions run in the `gpu-tests` integration suite.

#![deny(unsafe_op_in_unsafe_fn)]

mod compile;
mod context;
mod error;
mod kernel;
mod memory;
mod raw;
mod retirement;
mod runtime;
mod transfer;

pub use compile::{
    compile, nvrtc_version, Artifact, ArtifactCache, CompileKey, CompileRequest, OutputKind,
};
pub use context::{device_count, device_name, supports_memory_pool};
pub use error::{Error, Result};
pub use kernel::{Arg, BoundFunction, LaunchConfig};
pub use memory::{Buffer, Config, Stats, UninitBuffer};
pub use raw::{RawResource, RawScope};
pub use retirement::RetirementStats;
pub use runtime::Runtime;
pub use transfer::{Completion, CopyRoute, PendingDownload, PinnedHostBuffer};
