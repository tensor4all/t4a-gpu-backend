//! A standalone Metal execution runtime.
//!
//! `t4a-metal-runtime` owns one Metal device, one command queue and one
//! allocation-domain identity. Every buffer it hands out is a shared-storage
//! `MTLBuffer`, so the CPU and the GPU read and write the same address and the
//! runtime never copies. It compiles MSL at run time, keeps one bounded artifact
//! and pipeline cache per runtime, and submits checked, explicitly committed
//! command buffers.
//!
//! # Boundaries
//!
//! - Nothing is linked at build time. The frameworks are opened through
//!   `t4a-metal-sys` on first use, and `otool -L` on a binary that depends on this
//!   crate shows no `Metal`, `MetalPerformanceShaders` or `Accelerate` entry.
//! - Nothing here knows about tensors, graphs, AD, CPU providers or kernel
//!   catalogues. It has no `upload`/`download` and no transfer accounting: in a
//!   shared allocation domain there is nothing to transfer.
//! - No implicit fallback. A dtype the caller does not serve through this runtime is
//!   the caller's decision; [`probe`] reports what this host's compiler accepts,
//!   including that it rejects `double`.
//! - A submission is explicit. Launches encode into one command buffer, and only
//!   [`Runtime::commit`], [`Runtime::synchronize`] or [`Runtime::close`] submits it;
//!   encoded work that is never submitted is discarded when the runtime is dropped.
//!
//! # Example
//!
//! ```
//! # fn run() -> Result<(), t4a_metal_runtime::Error> {
//! use t4a_metal_runtime::{Arg, CompileRequest, LaunchConfig, Runtime};
//!
//! const ADD: &str = r#"#include <metal_stdlib>
//! using namespace metal;
//! kernel void add(device float* a [[buffer(0)]],
//!                 device const float* b [[buffer(1)]],
//!                 uint index [[thread_position_in_grid]]) {
//!     a[index] = a[index] + b[index];
//! }
//! "#;
//!
//! let mut runtime = Runtime::open(Default::default())?;
//! let bytes = 2 * std::mem::size_of::<f32>();
//! let mut a = runtime.alloc_zeroed(bytes)?;
//! let mut b = runtime.alloc_zeroed(bytes)?;
//! // SAFETY: both buffers are fresh, `bytes` long, and unreferenced by any
//! // submission, so the host may write them.
//! unsafe {
//!     std::slice::from_raw_parts_mut(a.host_ptr().cast::<f32>(), 2).copy_from_slice(&[1.0, 2.0]);
//!     std::slice::from_raw_parts_mut(b.host_ptr().cast::<f32>(), 2).copy_from_slice(&[10.0, 20.0]);
//! }
//! let artifact = runtime.compile(&CompileRequest::new("add", ADD))?;
//! let add = runtime.bind(&artifact, "add")?;
//! // SAFETY: the kernel reads and writes exactly two f32 values at index 0 and
//! // reads two at index 1, and the dispatch covers exactly those two elements.
//! unsafe {
//!     runtime.launch(
//!         &add,
//!         LaunchConfig::threads(2),
//!         &[Arg::Buffer { buffer: &a, offset: 0 }, Arg::Buffer { buffer: &b, offset: 0 }],
//!     )?;
//! }
//! runtime.commit()?.wait()?;
//! // SAFETY: every submission that could touch `a` has completed.
//! let values = unsafe { std::slice::from_raw_parts(a.host_ptr().cast::<f32>(), 2) };
//! assert_eq!(values, [11.0, 22.0]);
//! # Ok(()) }
//! # match run() {
//! #     Ok(()) => {}
//! #     Err(t4a_metal_runtime::Error::Unavailable { .. }) => {}
//! #     Err(other) => panic!("unexpected runtime failure: {other}"),
//! # }
//! ```
//!
//! The example runs as a doctest wherever a Metal device exists and skips the
//! dispatch where it does not, because that path is a typed error rather than a
//! panic.

#![deny(unsafe_op_in_unsafe_fn)]
#![deny(missing_docs)]

mod compile;
mod context;
mod error;
mod kernel;
mod memory;
mod probe;
mod raw;
mod runtime;
mod submission;

pub use compile::{source_hash, Artifact, CompileRequest};
pub use context::DeviceInfo;
pub use error::{Error, Result};
pub use kernel::{Arg, BoundFunction, LaunchConfig, Pod, MAX_PARAMETER_BYTES};
pub use memory::{Buffer, UninitBuffer};
pub use probe::{probe, probe_sources, Capabilities, ProbeSource};
pub use raw::RawScope;
pub use runtime::{Config, Runtime, Stats};
pub use submission::Completion;
