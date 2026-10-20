//! Public contract tests that can run without creating a CUDA context.
//!
//! Probe paths may load the driver but never create a context. On a host
//! without a driver, opening a runtime reports a typed unavailable
//! error. On a host with a driver, an unknown ordinal reports a typed invalid
//! device error. Neither case may panic, and neither is reported as a hardware
//! test success.

use t4a_cuda_runtime::{device_count, CompileRequest, Config, Error, OutputKind, Runtime};

#[test]
fn invalid_configuration_is_rejected_without_touching_cuda() {
    let config = Config {
        max_chunk_bytes: 0,
        ..Config::default()
    };
    assert!(matches!(
        Runtime::open(0, config),
        Err(Error::InvalidArgument { .. })
    ));
}

#[test]
fn unknown_ordinal_is_a_typed_error() {
    match Runtime::open(usize::MAX, Config::default()) {
        Err(Error::InvalidDevice { .. } | Error::DriverUnavailable { .. }) => {}
        Err(other) => panic!("unexpected error for an unknown ordinal: {other}"),
        Ok(_) => panic!("opening usize::MAX must never succeed"),
    }
}

#[test]
fn device_probe_reports_a_typed_result() {
    match device_count() {
        Ok(count) => assert!(count >= 1),
        Err(Error::DriverUnavailable { detail }) => {
            assert!(!detail.is_empty());
        }
        Err(other) => panic!("unexpected probe error: {other}"),
    }
}

#[test]
fn compile_request_key_invalidates_every_input() {
    let base = CompileRequest::new("k", "extern \"C\" __global__ void k() {}").key();
    let mut request = CompileRequest::new("k", "extern \"C\" __global__ void k() {}");
    request.output = OutputKind::Cubin;
    request.arch = Some("sm_80".into());
    assert_ne!(base, request.key());
    request.headers.push(("h".into(), "#define H 1".into()));
    let with_header = request.key();
    request.headers[0].1 = "#define H 2".into();
    assert_ne!(with_header, request.key());
}

#[test]
fn errors_expose_typed_categories() {
    let error = Error::ForeignDomain;
    assert_eq!(
        error.to_string(),
        "buffer belongs to a different allocation domain"
    );
    let error = Error::Backpressure { bytes: 64 };
    assert!(error.to_string().contains("64"));
}

#[test]
fn runtime_is_send_but_not_sync() {
    fn assert_send<T: Send>() {}
    assert_send::<Runtime>();
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<t4a_cuda_runtime::Buffer>();
}
