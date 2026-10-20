//! Cross-device staging, source-context witnesses and opposing transfers.
//!
//! This target requires two physical GPUs. Run it only on such a host; on a
//! one-GPU host it fails with an explicit message instead of passing silently.
//!
//! Not covered here yet: opposing transfers, forced chunk-ring recycling under
//! pressure, and cross-context failure propagation. Those acceptance rows remain
//! open even when this target passes.

#![cfg(feature = "gpu-tests")]

use t4a_cuda_runtime::{device_count, device_name, Config, CopyRoute, Runtime};

#[test]
fn staged_copy_between_two_physical_gpus() {
    let count = device_count().expect("driver must be available");
    assert!(
        count >= 2,
        "NOT RUN: this target requires two physical GPUs, host reports {count}"
    );
    let names = (
        device_name(0).unwrap_or_default(),
        device_name(1).unwrap_or_default(),
    );
    println!("devices: 0={} 1={}", names.0, names.1);

    let mut source_runtime = Runtime::open(0, Config::default()).expect("device 0");
    let mut destination = Runtime::open(1, Config::default()).expect("device 1");
    let payload: Vec<u8> = (0..50_000u32).map(|value| (value % 241) as u8).collect();

    let source = source_runtime.upload(&payload).expect("upload");
    let (copy, mut completion, route) = destination
        .copy_between(&mut source_runtime, &source)
        .expect("staged copy must be admitted");
    assert_eq!(route, CopyRoute::Staged);

    // The source allocation must be releasable right after admission; its D2H is
    // ordered on the source stream.
    drop(source);

    completion.wait().expect("staged copy must complete");
    let mut pending = destination.download(&copy).expect("download");
    assert_eq!(pending.wait().unwrap(), payload);
}

#[test]
fn three_runtimes_keep_dependencies_across_devices() {
    let count = device_count().expect("driver must be available");
    assert!(
        count >= 2,
        "NOT RUN: this target requires two physical GPUs, host reports {count}"
    );
    let mut a = Runtime::open(0, Config::default()).expect("device 0");
    let mut b = Runtime::open(1, Config::default()).expect("device 1");
    let mut c = Runtime::open(0, Config::default()).expect("device 0");
    let payload: Vec<u8> = (0..12_288u32).map(|value| (value % 233) as u8).collect();

    let first = a.upload(&payload).unwrap();
    let (second, ab, _) = b.copy_between(&mut a, &first).unwrap();
    // Admit the next hop without waiting for the first, so the chain exercises
    // ordering rather than a caller-side barrier.
    let (third, mut bc, _) = c.copy_between(&mut b, &second).unwrap();
    drop(ab);
    drop(second);
    bc.wait().unwrap();
    let mut pending = c.download(&third).unwrap();
    assert_eq!(pending.wait().unwrap(), payload);
}
