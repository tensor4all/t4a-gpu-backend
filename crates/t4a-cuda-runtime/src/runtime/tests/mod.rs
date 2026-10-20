use super::*;

#[test]
fn alignment_must_be_a_power_of_two() {
    assert!(validate_alignment(1).is_ok());
    assert!(validate_alignment(256).is_ok());
    assert!(validate_alignment(0).is_err());
    assert!(validate_alignment(3).is_err());
    assert!(validate_alignment(1 << 20).is_err());
}

#[cfg(feature = "gpu-tests")]
#[test]
fn closed_runtime_reports_closed() {
    let mut runtime = Runtime::open(0, Config::default()).expect("device 0 must be usable");
    let domain_id = runtime.domain_id().unwrap();
    runtime.close().unwrap();
    assert!(matches!(runtime.domain_id(), Err(Error::Closed)));
    assert!(matches!(runtime.synchronize(), Err(Error::Closed)));
    // Closing twice is idempotent.
    runtime.close().unwrap();
    assert_eq!(runtime.stats().domain_id, 0);
    assert_ne!(domain_id, 0);
}
