//! The absent-device path, reached by replacing the device function.
//!
//! The frameworks are real, so this is the same code path a Mac without a usable GPU
//! takes: `MTLCreateSystemDefaultDevice` answers nil and none of it panics.

#![cfg(target_os = "macos")]

use t4a_metal_sys::objc::Id;
use t4a_metal_sys::Table;

use crate::error::Error;
use crate::probe::probe_with;
use crate::runtime::{Config, Runtime};

unsafe extern "C" fn no_device() -> Id {
    std::ptr::null_mut()
}

fn table_without_a_device() -> Table {
    let mut table = Table::load().expect("Metal.framework loads on this host");
    table.create_system_default_device = no_device;
    table
}

#[test]
fn a_nil_device_is_a_typed_error() {
    match Runtime::open_with(Config::default(), table_without_a_device()) {
        Err(Error::Unavailable { what, detail }) => {
            assert_eq!(what, "no Metal device");
            assert!(detail.contains("MTLCreateSystemDefaultDevice"), "{detail}");
        }
        Err(other) => panic!("unexpected error: {other}"),
        Ok(_) => panic!("a nil device must not produce a runtime"),
    }
}

#[test]
fn the_probe_passes_the_absent_device_through() {
    match probe_with(&table_without_a_device()) {
        Err(Error::Unavailable { what, .. }) => assert_eq!(what, "no Metal device"),
        Err(other) => panic!("unexpected error: {other}"),
        Ok(_) => panic!("the probe must not report capabilities without a device"),
    }
}
