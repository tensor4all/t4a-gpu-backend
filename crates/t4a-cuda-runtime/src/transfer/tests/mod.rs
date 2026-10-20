#[allow(unused_imports)]
use super::*;

#[cfg(feature = "gpu-tests")]
#[test]
fn upload_roundtrip_is_byte_exact() {
    let mut runtime = crate::Runtime::open(0, crate::Config::default()).unwrap();
    let payload: Vec<u8> = (0..2048u32).map(|value| value as u8).collect();
    let buffer = runtime.upload(&payload).unwrap();
    let mut pending = runtime.download(&buffer).unwrap();
    assert_eq!(pending.wait().unwrap(), payload);
}

#[cfg(feature = "gpu-tests")]
#[test]
fn small_pending_budget_still_moves_a_large_transfer() {
    let config = crate::Config {
        max_chunk_bytes: 4096,
        max_idle_pinned_bytes: 4096,
        pending_host_budget: 8192,
        ..crate::Config::default()
    };
    let mut runtime = crate::Runtime::open(0, config).unwrap();
    let payload: Vec<u8> = (0..40_960u32).map(|value| value as u8).collect();
    let buffer = runtime.upload(&payload).unwrap();
    assert_eq!(buffer.len(), payload.len());
    let mut pending = runtime.download(&buffer).unwrap();
    assert_eq!(pending.wait().unwrap(), payload);
}

#[cfg(feature = "gpu-tests")]
#[test]
fn dropped_and_forgotten_downloads_do_not_free_staging_early() {
    let mut runtime = crate::Runtime::open(0, crate::Config::default()).unwrap();
    let payload: Vec<u8> = (0..8192u32).map(|value| value as u8).collect();
    let buffer = runtime.upload(&payload).unwrap();

    drop(runtime.download(&buffer).unwrap());
    // Deliberately forget an observation handle: the runtime must still own the
    // in-flight staging and never write into freed host memory.
    std::mem::forget(runtime.download(&buffer).unwrap());

    // Later admissions still succeed and are ordered after the dropped downloads.
    let again = runtime.upload(&payload).unwrap();
    let mut pending = runtime.download(&again).unwrap();
    assert_eq!(pending.wait().unwrap(), payload);
}

#[cfg(feature = "gpu-tests")]
#[test]
fn source_may_be_dropped_right_after_same_device_copy_admission() {
    let config = crate::Config::default();
    let mut source_runtime = crate::Runtime::open(0, config.clone()).unwrap();
    let mut destination = crate::Runtime::open(0, config).unwrap();
    let payload: Vec<u8> = (0..4096u32).map(|value| value as u8).collect();
    let source = source_runtime.upload(&payload).unwrap();
    let (copy, _completion, route) = destination
        .copy_between(&mut source_runtime, &source)
        .unwrap();
    assert_eq!(route, CopyRoute::SameDevice);
    drop(source);
    let mut pending = destination.download(&copy).unwrap();
    assert_eq!(pending.wait().unwrap(), payload);
}

#[cfg(feature = "gpu-tests")]
#[test]
fn injected_post_enqueue_failure_quarantines_the_source() {
    use crate::retirement::{arm_fault, disarm_fault, FaultPoint};

    let config = crate::Config::default();
    let mut source_runtime = crate::Runtime::open(0, config.clone()).unwrap();
    let mut destination = crate::Runtime::open(0, config).unwrap();
    let payload: Vec<u8> = (0..2048u32).map(|value| value as u8).collect();
    let source = source_runtime.upload(&payload).unwrap();

    arm_fault(FaultPoint::BeforeEventRecord);
    let outcome = destination.copy_between(&mut source_runtime, &source);
    disarm_fault();
    assert!(outcome.is_err(), "the injected fault must be reported");

    // The source allocation must not be released while a foreign read may still
    // be queued, and further admission on the source is refused.
    drop(source);
    assert!(matches!(
        source_runtime.upload(&payload),
        Err(crate::Error::Broken { .. })
    ));
    // The destination owns a copy it never published; observation reports the
    // failure instead of claiming readiness.
    assert!(destination.observe().is_err() || destination.stats().pending_records == 0);
}
