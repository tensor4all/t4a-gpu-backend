//! HIP kernels compiled at run time with hiprtc, reusing the `t4a-cuda-kernels` C ABI.
//!
//! # One source, two vendors
//!
//! This crate carries **no kernel source of its own**. `t4a-cuda-kernels` is the single
//! source of truth for the kernel text, the ABI header (`t4a/abi.cuh`, revision
//! [`t4a_cuda_kernels::ABI_REVISION`]), the dtype tags and the launch planning
//! ([`t4a_cuda_kernels::plan_elementwise`] and friends). This crate hands the very
//! same composed text ([`t4a_cuda_kernels::source::compose`]) to hiprtc instead of NVRTC,
//! passes the planned [`t4a_cuda_kernels::Launch`] to `hipModuleLaunchKernel` instead of
//! the CUDA driver, and passes the kernel arguments by the same `kernelParams`
//! convention. A kernel fixed on the CUDA side is fixed here, and the other way round.
//!
//! # Interface (the contract M3 consumes)
//!
//! - **load**: [`Program::compile`] / [`Program::compile_entry`] take a
//!   [`t4a_hip_runtime::Runtime`] and a CUDA-crate module, compile for the device's own
//!   `gfx` target, load into the device, and report ([`Program::report`]) the target,
//!   the options, the compile and load times and `HSA_OVERRIDE_GFX_VERSION`.
//! - **launch**: [`Program::launch`] is the raw, `unsafe` seam inside
//!   [`t4a_hip_runtime::Runtime::with_raw`]: a symbol, a planned `Launch`, and
//!   `kernelParams`. [`elementwise`] is the safe wrapper for the flat entries.
//! - **error**: [`Error`], with [`Error::Compile`] carrying target, options and the
//!   hiprtc log, and [`Error::Runtime`] wrapping the runtime's typed errors (so
//!   `Unavailable` still means "no ROCm here").

#![deny(unsafe_op_in_unsafe_fn)]
#![deny(missing_docs)]

mod elementwise;
mod error;
mod program;

pub use elementwise::{elementwise, Binary};
pub use error::{Error, Result};
pub use program::{compose, CompileReport, Program, HIP_PRELUDE};
pub use t4a_cuda_kernels as abi;
pub use t4a_hip_runtime as runtime;
