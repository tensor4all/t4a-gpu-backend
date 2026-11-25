//! Run-time loaded HIP, rocBLAS and rocSOLVER tables for tensor4all.
//!
//! This crate opens `libamdhip64`, `librocblas` and `librocsolver` with
//! `dlopen(RTLD_NOW | RTLD_LOCAL)` on first use, so no ROCm library is linked at build
//! time and `readelf -d` on a binary that depends on it lists none of them. A host
//! without ROCm gets a typed [`Error::LibraryUnavailable`], never a failure to start.
//!
//! # What it is
//!
//! Function-pointer tables ([`HipTable`], [`RocblasTable`], [`RocsolverTable`]) plus the
//! C types they pass, and three load-time rules:
//!
//! - **One prefix.** The math libraries are opened only from the directory the HIP
//!   runtime came from, and the process is checked for a second `libamdhip64` right
//!   after the HIP runtime and after each math library is opened
//!   ([`Error::DuplicateRuntime`]).
//! - **No empty `HSA_OVERRIDE_GFX_VERSION`.** It is checked before anything is opened
//!   and refused with [`Error::Configuration`]; see [`GfxOverride`]. The variable is
//!   never set or changed by this crate.
//! - **Explicit pointer classes.** The rocSOLVER signatures document which arguments
//!   are device pointers, and [`PointerAttributes`] lets a consumer check a pointer's
//!   class before a call that would fault on the wrong one.
//!
//! # What it is not
//!
//! No safe wrapper and no policy: streams, pools, handles and their ownership belong to
//! `t4a-hip-runtime` and its consumers.
//!
//! # Example
//!
//! ```
//! use t4a_hip_sys::{Error, HipTable};
//!
//! match HipTable::load() {
//!     Ok(hip) => {
//!         let mut count = 0;
//!         // SAFETY: `count` is a writable `int`.
//!         let status = unsafe { (hip.get_device_count)(&mut count) };
//!         assert!(status == t4a_hip_sys::HIP_SUCCESS || count == 0);
//!     }
//!     // No ROCm on this host; the load is fallible by design.
//!     Err(Error::LibraryUnavailable { .. }) => {}
//!     Err(other) => panic!("unexpected load failure: {other}"),
//! }
//! ```

#![deny(unsafe_op_in_unsafe_fn)]
#![deny(missing_docs)]

mod error;
mod hip;
mod loader;
mod rocblas;

pub use error::{Error, Result};
pub use hip::*;
pub use loader::{
    ensure_single_hip_runtime, hip_candidates, hip_runtimes_in_maps, mapped_hip_runtimes,
    sibling_candidates, GfxOverride, HIP_SONAMES, ROCBLAS_SONAMES, ROCSOLVER_SONAMES,
};
pub use rocblas::*;
