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
    // The destination must not report the unpublished copy as ready.
    match destination.observe() {
        Err(_) => {}
        Ok(()) => assert_eq!(destination.stats().pending_records, 0),
    }
}

#[cfg(feature = "gpu-tests")]
#[test]
fn injected_chunk_commit_failure_poisons_the_owner() {
    use crate::retirement::{arm_fault, disarm_fault, FaultPoint};

    let mut runtime = crate::Runtime::open(0, crate::Config::default()).unwrap();
    arm_fault(FaultPoint::AfterChunkEnqueue);
    let upload = runtime.upload(&[1u8; 64]);
    disarm_fault();
    assert!(upload.is_err(), "the injected fault must be reported");
    assert!(matches!(
        runtime.upload(&[1u8; 64]),
        Err(crate::Error::Broken { .. })
    ));
}

#[cfg(feature = "gpu-tests")]
#[test]
fn injected_source_back_wait_failure_quarantines_the_source() {
    use crate::retirement::{arm_fault, disarm_fault, FaultPoint};

    let config = crate::Config::default();
    let mut source_runtime = crate::Runtime::open(0, config.clone()).unwrap();
    let mut destination = crate::Runtime::open(0, config).unwrap();
    let payload: Vec<u8> = (0..1024u32).map(|value| value as u8).collect();
    let source = source_runtime.upload(&payload).unwrap();

    arm_fault(FaultPoint::BeforeSourceBackWait);
    let outcome = destination.copy_between(&mut source_runtime, &source);
    disarm_fault();
    assert!(
        outcome.is_err(),
        "the injected back-wait fault must be reported"
    );
    drop(source);
    assert!(matches!(
        source_runtime.upload(&payload),
        Err(crate::Error::Broken { .. })
    ));
}

#[cfg(feature = "gpu-tests")]
#[test]
fn source_mutation_after_upload_does_not_reach_the_device() {
    let mut runtime = crate::Runtime::open(0, crate::Config::default()).unwrap();
    let mut caller = vec![7u8; 4096];
    let buffer = runtime.upload(&caller).unwrap();
    // The runtime copied into its own staging before returning.
    caller.fill(9);
    let mut pending = runtime.download(&buffer).unwrap();
    assert_eq!(pending.wait().unwrap(), vec![7u8; 4096]);
}

#[cfg(feature = "gpu-tests")]
#[test]
fn owned_pinned_upload_uses_direct_and_staged_routes() {
    let config = crate::Config {
        max_chunk_bytes: 1024,
        max_idle_pinned_bytes: 2048,
        pending_host_budget: 2048,
        ..crate::Config::default()
    };
    let mut runtime = crate::Runtime::open(0, config).unwrap();

    // Fits the budget: direct DMA from the consumed pinned allocation.
    let mut small = runtime.alloc_pinned(512).unwrap();
    small.as_mut_slice().fill(3);
    let direct = runtime.upload_owned(small).unwrap();
    let mut pending = runtime.download(&direct).unwrap();
    assert_eq!(pending.wait().unwrap(), vec![3u8; 512]);

    // Larger than the budget: staged through bounded chunks.
    let mut large = runtime.alloc_pinned(8192).unwrap();
    large.as_mut_slice().fill(5);
    let staged = runtime.upload_owned(large).unwrap();
    let mut pending = runtime.download(&staged).unwrap();
    assert_eq!(pending.wait().unwrap(), vec![5u8; 8192]);
}
