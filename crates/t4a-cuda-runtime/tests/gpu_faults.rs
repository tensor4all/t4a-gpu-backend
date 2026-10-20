//! Real asynchronous failures, isolated in a child process.
//!
//! A sticky CUDA context error poisons the whole process, so the test re-invokes
//! its own binary and performs the faulting work there. The child has an external
//! deadline from the supervising test runner.

#![cfg(feature = "gpu-tests")]

use t4a_cuda_runtime::{Config, RawResource, Runtime};

#[test]
fn illegal_address_is_reported_and_contained() {
    if std::env::var("T4A_FAULT_CHILD").is_ok() {
        child_fault();
        return;
    }
    let executable = std::env::current_exe().expect("test binary path");
    let mut child = std::process::Command::new(executable)
        .args([
            "--exact",
            "illegal_address_is_reported_and_contained",
            "--nocapture",
        ])
        .env("T4A_FAULT_CHILD", "1")
        .spawn()
        .expect("spawn child");
    // A poisoned context can wedge the child; keep an explicit deadline so the
    // parent reports a failure instead of waiting forever.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
    loop {
        if let Some(status) = child.try_wait().expect("wait for child") {
            assert!(status.success(), "the isolated fault run failed: {status}");
            return;
        }
        if std::time::Instant::now() > deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!("the isolated fault run exceeded its 120 s deadline");
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
}

fn child_fault() {
    let mut runtime = Runtime::open(0, Config::default()).expect("device 0");
    let buffer = runtime.alloc_zeroed(64, 256).expect("allocation");
    // SAFETY: this deliberately writes to an unmapped device address through the
    // runtime's own stream to provoke a real asynchronous fault. The damage is
    // confined to this child process's context.
    let outcome = unsafe {
        runtime.with_raw(vec![RawResource::Buffer(&buffer)], |raw| {
            let stream = raw.stream();
            // cuMemsetD8Async on a bogus pointer.
            cudarc::driver::sys::cuMemsetD8Async(
                0xdead_0000u64,
                0,
                64,
                stream as usize as *mut cudarc::driver::sys::CUstream_st,
            )
        })
    };
    assert!(outcome.is_ok(), "admission itself is asynchronous");
    let observed = runtime.synchronize();
    assert!(
        observed.is_err(),
        "synchronize must report the asynchronous fault"
    );
    // A poisoned runtime refuses further admissions.
    assert!(runtime.upload(&[1u8, 2, 3]).is_err());
}
