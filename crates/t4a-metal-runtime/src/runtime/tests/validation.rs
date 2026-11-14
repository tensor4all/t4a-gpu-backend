use crate::error::Error;
use crate::runtime::Config;

#[test]
fn a_zero_in_flight_bound_is_rejected() {
    let config = Config {
        max_submissions_in_flight: 0,
        ..Config::default()
    };
    match config.validate() {
        Err(Error::InvalidArgument { detail }) => {
            assert!(detail.contains("max_submissions_in_flight"), "{detail}");
        }
        other => panic!("expected an invalid-argument error, got {other:?}"),
    }
}

#[test]
fn a_zero_timeout_is_rejected() {
    let config = Config {
        completion_timeout_ms: 0,
        ..Config::default()
    };
    match config.validate() {
        Err(Error::InvalidArgument { detail }) => assert!(detail.contains("completion_timeout_ms")),
        other => panic!("expected an invalid-argument error, got {other:?}"),
    }
}

#[test]
fn every_zero_bound_is_rejected_by_name() {
    let cases: [(&str, Config); 3] = [
        (
            "max_artifact_entries",
            Config {
                max_artifact_entries: 0,
                ..Config::default()
            },
        ),
        (
            "max_artifact_bytes",
            Config {
                max_artifact_bytes: 0,
                ..Config::default()
            },
        ),
        (
            "max_pipeline_entries",
            Config {
                max_pipeline_entries: 0,
                ..Config::default()
            },
        ),
    ];
    for (name, config) in cases {
        match config.validate() {
            Err(Error::InvalidArgument { detail }) => {
                assert!(detail.contains(name), "{name} must be named, got {detail}");
            }
            other => panic!("{name}: expected an invalid-argument error, got {other:?}"),
        }
    }
}

#[test]
fn the_default_config_is_valid_and_bounded() {
    let config = Config::default();
    config.validate().expect("the defaults are valid");
    assert!(config.max_submissions_in_flight > 0);
    assert!(config.completion_timeout_ms > 0);
    assert!(config.max_artifact_entries > 0);
    assert!(config.max_artifact_bytes > 0);
    assert!(config.max_pipeline_entries > 0);
}
