use crate::error::Error;
use crate::metal::{Framework, Table};
use crate::test_support::Faulty;

#[test]
fn the_table_loads_the_required_framework() {
    let table = Table::load().expect("Metal.framework must load on this host");
    assert!(table.has_framework(Framework::Metal));
    // Exercise the table end to end: a real device, its selector, the autoreleased
    // name, the retain/release pair and `-[NSString UTF8String]`.
    if let Some(device) = table.system_default_device() {
        // SAFETY: the device came from this table.
        let name = unsafe { table.device_name(device) }.expect("a live device reports a name");
        assert!(!name.is_empty(), "the device name is a non-empty string");
    }
}

#[test]
fn opening_a_framework_makes_its_classes_visible() {
    // The measurement in tenferro-rs#2402 reported `objc_getClass("MPSGraph")` as
    // absent. MPSGraph lives in its own framework, so the class appears only after
    // that image is loaded; this asserts the re-check the design asks for.
    let table = Table::load().expect("Metal.framework must load on this host");
    assert!(
        table.has_framework(Framework::MetalPerformanceShadersGraph),
        "MetalPerformanceShadersGraph.framework opens on this host"
    );
    assert!(
        table.class("MPSGraph").is_some(),
        "MPSGraph is visible once its framework is loaded"
    );
    assert!(
        table.class("MPSMatrixMultiplication").is_some(),
        "MetalPerformanceShaders classes are visible once that framework is loaded"
    );
    assert!(
        table.class("T4AClassThatDoesNotExist").is_none(),
        "an unknown class is absent, so the lookup is not fabricating handles"
    );
}

#[test]
fn a_missing_required_framework_names_it() {
    let loader = Faulty {
        fail_open: &["Metal.framework"],
        ..Faulty::default()
    };
    let error = Table::load_with(&loader).expect_err("Metal.framework was forced to fail");
    match error {
        Error::FrameworkUnavailable { framework, .. } => assert_eq!(framework, "Metal.framework"),
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn a_missing_required_symbol_names_it() {
    let loader = Faulty {
        miss_symbol: &[("libobjc", "objc_retain")],
        ..Faulty::default()
    };
    let error = Table::load_with(&loader).expect_err("objc_retain was forced to fail");
    match error {
        Error::MissingSymbol { framework, symbol } => {
            assert_eq!(framework, "libobjc");
            assert_eq!(symbol, "objc_retain");
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn the_optional_frameworks_may_be_absent() {
    let loader = Faulty {
        fail_open: &[
            "MetalPerformanceShaders.framework",
            "MetalPerformanceShadersGraph.framework",
        ],
        ..Faulty::default()
    };
    let table =
        Table::load_with(&loader).expect("only the optional frameworks were forced to fail");
    assert!(table.has_framework(Framework::Metal));
    assert!(!table.has_framework(Framework::MetalPerformanceShaders));
    assert!(!table.has_framework(Framework::MetalPerformanceShadersGraph));
    // Class visibility is process-wide, not per table: another table in this test
    // binary may already have loaded the framework, so the absence of the image in
    // *this* table is what is asserted, not the absence of the class.
}
