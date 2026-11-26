//! The pool accounting a vendor workspace relies on: `pool_stats` reports the bytes
//! live buffers hold now, not a high-water mark, and works from whichever thread owns
//! the runtime.
//!
//! Without a HIP device each test prints the typed reason and returns, unless
//! `T4A_REQUIRE_HIP=1` makes that a failure.

mod support;

use support::runtime;

/// Room the pool may add on top of a request for its own granularity. ROCm 7.2.4 on
/// gfx1151 reports exactly the requested bytes.
const GRANULARITY_SLACK: u64 = 2 << 20;

#[test]
fn pool_stats_follow_an_allocation_up_and_back_down() {
    let Some(mut runtime) = runtime() else {
        return;
    };
    runtime.synchronize().expect("drain");
    let before = runtime.pool_stats().expect("stats").used_bytes;
    for bytes in [1usize, 4096, 790_784, 3 << 20] {
        let buffer = runtime.alloc_zeroed(bytes).expect("allocate");
        runtime.synchronize().expect("drain");
        let held = runtime.pool_stats().expect("stats").used_bytes;
        let grown = held - before;
        assert!(
            grown >= bytes as u64 && grown <= bytes as u64 + GRANULARITY_SLACK,
            "{bytes} bytes allocated but the pool's used memory grew by {grown}"
        );
        drop(buffer);
        runtime.synchronize().expect("the queued free has run");
        let after = runtime.pool_stats().expect("stats").used_bytes;
        assert_eq!(
            after, before,
            "after the free the pool must report current use, not the {held}-byte peak"
        );
    }
}

#[test]
fn a_runtime_moved_to_another_thread_keeps_its_device_and_pool() {
    // `Runtime` is `Send`; HIP's current device is per thread, so every entry point,
    // the pool queries included, makes the runtime's device current first. On a
    // one-device host the device is current anyway, so this pins the usage rather than
    // the activation itself.
    let Some(mut runtime) = runtime() else {
        return;
    };
    let buffer = runtime.upload(&[7u8; 64]).expect("upload");
    let (runtime, bytes) = std::thread::spawn(move || {
        let mut runtime = runtime;
        let stats = runtime.pool_stats().expect("stats on another thread");
        assert!(stats.used_bytes >= 64);
        let bytes = runtime
            .download(&buffer)
            .expect("download on another thread");
        (runtime, bytes)
    })
    .join()
    .expect("worker");
    assert_eq!(bytes, vec![7u8; 64]);
    drop(runtime);
}
