//! Tests for the loader-error mapping.

use t4a_metal_sys::Error as SysError;

use crate::error::Error;

#[test]
fn a_missing_framework_names_it_in_the_remedy() {
    let error = Error::from(SysError::FrameworkUnavailable {
        framework: "Metal.framework",
        detail: "image not found".to_string(),
    });
    match error {
        Error::Unavailable { what, detail } => {
            assert_eq!(what, "Metal.framework could not be opened");
            assert_eq!(detail, "image not found");
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn a_missing_symbol_names_both_the_framework_and_the_symbol() {
    let error = Error::from(SysError::MissingSymbol {
        framework: "libobjc",
        symbol: "objc_msgSend",
    });
    match error {
        Error::Unavailable { what, detail } => {
            assert_eq!(what, "libobjc does not export `objc_msgSend`");
            assert!(detail.contains("objc_msgSend"), "detail: {detail}");
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn every_unavailable_message_names_what_failed() {
    for error in [
        Error::from(SysError::FrameworkUnavailable {
            framework: "Metal.framework",
            detail: String::new(),
        }),
        Error::from(SysError::InvalidName {
            name: "bad".to_string(),
        }),
    ] {
        let text = error.to_string();
        assert!(
            text.contains("Metal") || text.contains("bad"),
            "the message names its cause: {text}"
        );
    }
}
