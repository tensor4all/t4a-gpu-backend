//! The framework table: what was opened and what was resolved.
//!
//! Loading is explicit and fallible; nothing is probed lazily behind a `OnceLock`
//! so that a consumer can decide when to pay for it and can report the failure.

use std::ffi::{c_char, c_void};

use crate::error::{cstr, Result};
use crate::loader::{Image, Loader, System};
use crate::objc::{Id, Objc, Pool, Selectors};

/// A framework this crate knows how to open.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Framework {
    /// `Metal.framework`, required.
    Metal,
    /// `MetalPerformanceShaders.framework`, optional.
    MetalPerformanceShaders,
    /// `MetalPerformanceShadersGraph.framework`, optional.
    MetalPerformanceShadersGraph,
}

impl Framework {
    /// The framework's name, for error messages.
    pub fn name(self) -> &'static str {
        match self {
            Framework::Metal => "Metal.framework",
            Framework::MetalPerformanceShaders => "MetalPerformanceShaders.framework",
            Framework::MetalPerformanceShadersGraph => "MetalPerformanceShadersGraph.framework",
        }
    }

    /// The framework's install path.
    pub fn path(self) -> &'static [u8] {
        match self {
            Framework::Metal => b"/System/Library/Frameworks/Metal.framework/Metal\0".as_slice(),
            Framework::MetalPerformanceShaders => {
                b"/System/Library/Frameworks/MetalPerformanceShaders.framework/MetalPerformanceShaders\0"
                    .as_slice()
            }
            Framework::MetalPerformanceShadersGraph => {
                b"/System/Library/Frameworks/MetalPerformanceShadersGraph.framework/MetalPerformanceShadersGraph\0"
                    .as_slice()
            }
        }
    }
}

const LIBOBJC: &str = "libobjc";
const LIBOBJC_PATH: &[u8] = b"/usr/lib/libobjc.A.dylib\0";

/// The Metal function `MTLCreateSystemDefaultDevice`.
pub type CreateSystemDefaultDevice = unsafe extern "C" fn() -> Id;

/// Everything this crate resolved: the Objective-C runtime, the selectors and the
/// framework images.
///
/// Holding a `Table` means the frameworks in it are loaded. It makes no claim that
/// a device exists or is usable; that is the consumer's probe.
#[derive(Debug, Clone, Copy)]
pub struct Table {
    /// The Objective-C runtime entry points.
    pub objc: Objc,
    /// The selectors this repository sends.
    pub selectors: Selectors,
    /// `MTLCreateSystemDefaultDevice`.
    pub create_system_default_device: CreateSystemDefaultDevice,
    /// The opened `Metal.framework` image.
    pub metal: Image,
    /// The opened `MetalPerformanceShaders.framework` image, when present.
    pub mps: Option<Image>,
    /// The opened `MetalPerformanceShadersGraph.framework` image, when present.
    pub mpsgraph: Option<Image>,
}

impl Table {
    /// Open the frameworks and resolve the table with the system dynamic loader.
    ///
    /// # Errors
    ///
    /// [`crate::Error::FrameworkUnavailable`] when `libobjc` or `Metal.framework`
    /// cannot be opened, and [`crate::Error::MissingSymbol`] when a required symbol
    /// is absent. The two MetalPerformanceShaders frameworks are optional: their
    /// absence is recorded and is not an error.
    pub fn load() -> Result<Self> {
        Self::load_with(&System)
    }

    /// Open the frameworks through `loader`.
    ///
    /// This is the seam the absent-framework tests use: a loader that refuses to
    /// open a framework must produce the same typed error a host without that
    /// framework produces.
    ///
    /// # Errors
    ///
    /// As [`Table::load`].
    pub fn load_with(loader: &dyn Loader) -> Result<Self> {
        // SAFETY: each name below is a NUL-terminated literal, and each symbol is
        // cast to its documented C signature. Images stay loaded for the process,
        // which is the `Loader` contract.
        unsafe {
            let libobjc_path = cstr(LIBOBJC_PATH)?;
            let objc_image = loader.open(LIBOBJC, libobjc_path)?;

            let get_class: unsafe extern "C" fn(*const c_char) -> Id = std::mem::transmute(
                loader.symbol(LIBOBJC, objc_image, cstr(b"objc_getClass\0")?)?,
            );
            let sel_register_name: unsafe extern "C" fn(*const c_char) -> *const c_void =
                std::mem::transmute(loader.symbol(
                    LIBOBJC,
                    objc_image,
                    cstr(b"sel_registerName\0")?,
                )?);
            let msg_send = loader.symbol(LIBOBJC, objc_image, cstr(b"objc_msgSend\0")?)?;
            let retain: unsafe extern "C" fn(Id) -> Id =
                std::mem::transmute(loader.symbol(LIBOBJC, objc_image, cstr(b"objc_retain\0")?)?);
            let release: unsafe extern "C" fn(Id) = std::mem::transmute(loader.symbol(
                LIBOBJC,
                objc_image,
                cstr(b"objc_release\0")?,
            )?);
            let autorelease_pool_push: unsafe extern "C" fn() -> *mut c_void = std::mem::transmute(
                loader.symbol(LIBOBJC, objc_image, cstr(b"objc_autoreleasePoolPush\0")?)?,
            );
            let autorelease_pool_pop: unsafe extern "C" fn(*mut c_void) = std::mem::transmute(
                loader.symbol(LIBOBJC, objc_image, cstr(b"objc_autoreleasePoolPop\0")?)?,
            );

            let objc = Objc {
                get_class,
                sel_register_name,
                msg_send,
                retain,
                release,
                autorelease_pool_push,
                autorelease_pool_pop,
            };
            let selectors = Selectors::register(&objc)?;

            let metal = loader.open(Framework::Metal.name(), cstr(Framework::Metal.path())?)?;
            let create_system_default_device: CreateSystemDefaultDevice =
                std::mem::transmute(loader.symbol(
                    Framework::Metal.name(),
                    metal,
                    cstr(b"MTLCreateSystemDefaultDevice\0")?,
                )?);

            // Both are optional: a host whose OS ships an older Metal simply has no
            // image, and `Table::class` then reports the classes as absent.
            let mps = loader
                .open(
                    Framework::MetalPerformanceShaders.name(),
                    cstr(Framework::MetalPerformanceShaders.path())?,
                )
                .ok();
            let mpsgraph = loader
                .open(
                    Framework::MetalPerformanceShadersGraph.name(),
                    cstr(Framework::MetalPerformanceShadersGraph.path())?,
                )
                .ok();

            Ok(Self {
                objc,
                selectors,
                create_system_default_device,
                metal,
                mps,
                mpsgraph,
            })
        }
    }

    /// The default Metal device, or `None` when the host has none.
    pub fn system_default_device(&self) -> Option<Id> {
        // SAFETY: `MTLCreateSystemDefaultDevice` has no preconditions and returns a
        // `+1` device or nil. The caller owns that reference.
        let device = unsafe { (self.create_system_default_device)() };
        (!device.is_null()).then_some(device)
    }

    /// Look up an Objective-C class by name.
    ///
    /// The Objective-C runtime registers classes when their image loads, and images
    /// are process-wide: this answers "is the class loaded in this process", not
    /// "did this table open the framework". A table that failed to open an optional
    /// framework still sees its classes when another table in the process loaded it.
    pub fn class(&self, name: &str) -> Option<Id> {
        self.objc.class(name)
    }

    /// Whether `which` was opened.
    pub fn has_framework(&self, which: Framework) -> bool {
        match which {
            Framework::Metal => true,
            Framework::MetalPerformanceShaders => self.mps.is_some(),
            Framework::MetalPerformanceShadersGraph => self.mpsgraph.is_some(),
        }
    }

    /// Create an `NSString` holding `text`, owned by the caller (`+1`).
    ///
    /// A `None` result means the text could not become an `NSString`, which for MSL
    /// source can only mean an interior NUL byte.
    pub fn new_string(&self, text: &str) -> Option<Id> {
        let class = self.class("NSString")?;
        let bytes = std::ffi::CString::new(text).ok()?;
        let pool = Pool::push(self.objc);
        // SAFETY: `class` is the live `NSString` class and `bytes` is NUL-terminated;
        // the result is an autoreleased string inside this pool.
        let string = unsafe {
            self.selectors
                .class_string_with_utf8
                .send(class, bytes.as_ptr())
        };
        if string.is_null() {
            drop(pool);
            return None;
        }
        // SAFETY: `string` is live in the open pool; the extra reference outlives it.
        let retained = unsafe { self.objc.retain(string) };
        drop(pool);
        Some(retained)
    }

    /// The device's name, read from the device itself.
    ///
    /// # Safety
    ///
    /// `device` must be a live `MTLDevice` obtained from this table.
    pub unsafe fn device_name(&self, device: Id) -> Option<String> {
        let pool = Pool::push(self.objc);
        // SAFETY: the caller guarantees a live device, and `name` returns an
        // autoreleased string inside this pool.
        let name = unsafe { self.selectors.device_name.send(device) };
        if name.is_null() {
            drop(pool);
            return None;
        }
        // SAFETY: `name` is a live string owned by this pool.
        let retained = unsafe { self.objc.retain(name) };
        drop(pool);
        // SAFETY: `retained` is a `+1` string, live until the release below.
        let value = unsafe { crate::objc::utf8_owned(&self.selectors, retained) };
        // SAFETY: this releases exactly the reference taken above.
        unsafe { self.objc.release(retained) };
        value
    }
}

#[cfg(test)]
mod tests;
