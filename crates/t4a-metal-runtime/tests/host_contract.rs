//! Public contract tests that need no device.
//!
//! On a host with Metal these paths construct a real runtime; on a host without it
//! they observe the typed absent error. Neither case may panic, and neither is
//! reported as a hardware test success.

use t4a_metal_runtime::{probe, probe_sources, source_hash, Config, Error, Runtime};

#[test]
fn invalid_configuration_is_rejected_without_touching_metal() {
    let config = Config {
        max_artifact_entries: 0,
        ..Config::default()
    };
    assert!(matches!(
        Runtime::open(config),
        Err(Error::InvalidArgument { .. })
    ));
}

#[test]
fn the_probe_reports_a_typed_result() {
    match probe() {
        Ok(capabilities) => {
            assert!(!capabilities.device.name.is_empty());
            assert!(
                !capabilities.f64_supported,
                "no Apple GPU compiles a double kernel"
            );
        }
        Err(Error::Unavailable { what, detail }) => {
            assert!(!what.is_empty());
            assert!(!detail.is_empty());
        }
        Err(other) => panic!("unexpected probe error: {other}"),
    }
}

#[test]
fn opening_a_runtime_is_fallible_and_never_panics() {
    match Runtime::open(Config::default()) {
        Ok(runtime) => {
            assert!(!runtime.device().name.is_empty());
            assert!(runtime.domain_id() > 0);
        }
        Err(Error::Unavailable { .. }) => {}
        Err(other) => panic!("unexpected open error: {other}"),
    }
}

#[test]
fn source_hashes_are_sha256_of_the_source() {
    // SHA-256 of the empty string, the published test vector.
    assert_eq!(
        source_hash(""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_ne!(source_hash("a"), source_hash("b"));
    for source in probe_sources() {
        let hash = source_hash(source.source);
        assert_eq!(
            hash.len(),
            64,
            "{} is a 64-character hex digest",
            source.label
        );
        assert!(
            hash.chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
            "{} is lowercase hex: {hash}",
            source.label
        );
    }
}

#[test]
fn probe_labels_are_distinct_so_a_report_can_name_them() {
    let mut labels: Vec<&str> = probe_sources().iter().map(|source| source.label).collect();
    let count = labels.len();
    labels.sort_unstable();
    labels.dedup();
    assert_eq!(labels.len(), count, "probe labels are unique");
    assert!(labels.contains(&"f64"), "the rejected dtype is probed too");
}
