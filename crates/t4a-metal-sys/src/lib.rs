//! Run-time loaded Metal tables for tensor4all.
//!
//! This crate resolves the Apple GPU frameworks with `dlopen`/`dlsym` and the
//! Objective-C runtime with `objc_getClass`/`sel_registerName`/`objc_msgSend`, so
//! no framework is linked at build time and `otool -L` on any binary that depends
//! on it shows no `Metal`, `MetalPerformanceShaders` or `Accelerate` entry.
//!
//! # What it is
//!
//! A table of function pointers and selectors: [`Table`] plus the [`objc`] module
//! that owns every `objc_msgSend` cast this repository uses. Nothing here decides
//! policy. It does not ask whether a device is usable, which dtypes work, or which
//! kernels to compile; a consumer does that with its own typed errors.
//!
//! # What it is not
//!
//! - No safe wrapper around Metal objects. A [`objc::Id`] is a raw pointer whose
//!   lifetime, ownership and selector validity the caller keeps.
//! - No build-time linking. `Metal.framework` is opened from
//!   `/System/Library/Frameworks` on first use.
//! - No `MetalPerformanceShaders` call. Those frameworks are opened so class
//!   presence can be reported; the selectors arrive with the consumer that needs
//!   them.
//!
//! # Example
//!
//! ```
//! use t4a_metal_sys::{Error, Framework, Table};
//!
//! match Table::load() {
//!     Ok(table) => {
//!         assert!(table.has_framework(Framework::Metal));
//!         if let Some(device) = table.system_default_device() {
//!             // SAFETY: the device came from this table.
//!             let name = unsafe { table.device_name(device) };
//!             assert!(!name.unwrap_or_default().is_empty());
//!         }
//!     }
//!     // No Apple frameworks on this host; the load is fallible by design.
//!     Err(Error::FrameworkUnavailable { .. }) => {}
//!     Err(other) => panic!("unexpected load failure: {other}"),
//! }

#![deny(unsafe_op_in_unsafe_fn)]
#![deny(missing_docs)]

mod error;
mod loader;
mod metal;
pub mod objc;

pub use error::{Error, Result};
pub use loader::{Image, Loader, System};
pub use metal::{CreateSystemDefaultDevice, Framework, Table};

// Only the macOS-only table tests use the faulty loader, so the helper is compiled
// only where it is used.
#[cfg(all(test, target_os = "macos"))]
mod test_support;
