//! A standalone HIP execution runtime.
//!
//! `t4a-hip-runtime` owns one real HIP stream, one explicitly created memory pool and
//! one allocation domain per [`Runtime`], ordered host copies, and a raw seam
//! ([`Runtime::with_raw`]) through which a vendor library runs on that exact stream with
//! pool-owned memory. It depends on no tensor, graph, AD or vendor-library crate; the
//! ROCm libraries are loaded at run time by `t4a-hip-sys`.
//!
//! This is stage 1 of tenferro-rs#2489: the parts of the `t4a-cuda-runtime` contract a
//! vendor GEMM needs, on HIP. Kernel compilation (hiprtc), pinned staging, completion
//! handles and cross-runtime copies are later stages.
//!
//! # Design
//!
//! - One [`Runtime`] is one execution owner: submission takes `&mut Runtime`.
//! - [`Buffer`] handles carry no submission authority. Their last drop queues
//!   `hipFreeAsync` on the owning stream, after every use already queued there.
//! - Memory comes from the runtime's own `hipMemPool`, so [`Runtime::pool_stats`]
//!   accounts for it. A vendor workspace is taken from the same pool through
//!   [`RawScope::alloc_zeroed`].
//! - Pointer classes are explicit: [`Runtime::pointer_class`] tells device memory from
//!   host memory before a vendor call that would fault on the wrong one.
//! - `HSA_OVERRIDE_GFX_VERSION` set but empty is refused with
//!   [`Error::Configuration`] before HIP is loaded; the runtime never changes it.
//!
//! # Example
//!
//! ```
//! use t4a_hip_runtime::{Error, Runtime};
//!
//! match Runtime::open(0) {
//!     Ok(mut runtime) => {
//!         let buffer = runtime.upload(&[1u8, 2, 3, 4])?;
//!         assert_eq!(runtime.download(&buffer)?, vec![1, 2, 3, 4]);
//!     }
//!     // No ROCm or no device on this host; opening is fallible by design.
//!     Err(Error::Unavailable { .. } | Error::InvalidDevice { .. }) => {}
//!     Err(other) => return Err(other),
//! }
//! # Ok::<(), Error>(())
//! ```

#![deny(unsafe_op_in_unsafe_fn)]
#![deny(missing_docs)]

mod device;
mod error;
mod memory;
mod runtime;

pub use device::{device_count, device_info, probe, DeviceInfo};
pub use error::{Error, Result};
pub use memory::{Buffer, PoolStats};
pub use runtime::{DomainAnchor, PointerClass, RawScope, Runtime};
pub use t4a_hip_sys as sys;
