use std::ffi::CStr;

use crate::error::Error;
use crate::loader::{Loader, System};

fn cstr(bytes: &'static [u8]) -> &'static CStr {
    CStr::from_bytes_with_nul(bytes).expect("test literal is NUL-terminated")
}

#[test]
fn a_missing_image_names_the_framework() {
    // SAFETY: the path is NUL-terminated and points at nothing loadable.
    let error = unsafe { System.open("Nowhere.framework", cstr(b"/nonexistent/Nowhere\0")) }
        .expect_err("opening a nonexistent image must fail");
    match error {
        Error::FrameworkUnavailable { framework, detail } => {
            assert_eq!(framework, "Nowhere.framework");
            assert!(!detail.is_empty(), "the loader's own detail is kept");
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn a_missing_symbol_names_the_symbol() {
    // SAFETY: `libobjc` is loaded for the process by any Objective-C program.
    let image = unsafe {
        System
            .open("libobjc", cstr(b"/usr/lib/libobjc.A.dylib\0"))
            .expect("libobjc loads")
    };
    // SAFETY: the image is loaded and the name is NUL-terminated.
    let error = unsafe { System.symbol("libobjc", image, cstr(b"objc_not_a_symbol\0")) }
        .expect_err("an absent symbol must not resolve");
    match error {
        Error::MissingSymbol { framework, symbol } => {
            assert_eq!(framework, "libobjc");
            assert_eq!(symbol, "objc_not_a_symbol");
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn a_present_symbol_resolves() {
    // SAFETY: as above.
    let image = unsafe {
        System
            .open("libobjc", cstr(b"/usr/lib/libobjc.A.dylib\0"))
            .expect("libobjc loads")
    };
    // SAFETY: as above.
    let symbol = unsafe { System.symbol("libobjc", image, cstr(b"objc_msgSend\0")) };
    assert!(symbol.is_ok(), "objc_msgSend must resolve: {symbol:?}");
}

#[test]
fn images_are_not_closed() {
    // SAFETY: as above.
    let first = unsafe {
        System
            .open("libobjc", cstr(b"/usr/lib/libobjc.A.dylib\0"))
            .expect("libobjc loads")
    };
    // SAFETY: as above.
    let second = unsafe {
        System
            .open("libobjc", cstr(b"/usr/lib/libobjc.A.dylib\0"))
            .expect("libobjc loads")
    };
    assert_eq!(
        first.handle(),
        second.handle(),
        "the loader returns the same process-wide image twice"
    );
}
