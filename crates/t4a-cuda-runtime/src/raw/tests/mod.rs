#[allow(unused_imports)]
use super::*;

#[cfg(feature = "gpu-tests")]
#[test]
fn raw_scope_exposes_the_runtime_stream_and_registered_buffers() {
    let mut runtime = crate::Runtime::open(0, crate::Config::default()).unwrap();
    let buffer = runtime.alloc_zeroed(64, 256).unwrap();
    let resource = RawResource::Buffer(&buffer);
    let keepalive = RawResource::Keepalive {
        value: Box::new(7u32),
        bytes: 4,
    };
    // SAFETY: only a memset within the registered buffer is issued, on this
    // runtime's own stream.
    let value = unsafe {
        runtime
            .with_raw(vec![resource, keepalive], |raw| {
                let ptr = raw.device_ptr(&buffer).unwrap();
                assert_eq!(raw.stream(), raw.stream());
                assert_ne!(ptr, 0);
                raw.domain_id()
            })
            .unwrap()
    };
    assert_eq!(value, runtime.domain_id().unwrap());
    runtime.synchronize().unwrap();
}

#[cfg(feature = "gpu-tests")]
#[test]
fn panicking_raw_closure_poisons_later_admission() {
    let mut runtime = crate::Runtime::open(0, crate::Config::default()).unwrap();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: the closure panics before issuing any CUDA call.
        unsafe {
            let _: () = runtime.with_raw(Vec::new(), |_raw| panic!("boom")).unwrap();
        }
    }));
    assert!(result.is_err());
    assert!(matches!(runtime.observe(), Err(Error::Broken { .. })));
}

#[cfg(feature = "gpu-tests")]
#[test]
fn unregistered_buffers_are_refused_inside_a_raw_scope() {
    let mut runtime = crate::Runtime::open(0, crate::Config::default()).unwrap();
    let outside = runtime.alloc_zeroed(16, 256).unwrap();
    // SAFETY: the closure only inspects pointers and issues no CUDA call.
    let outcome = unsafe { runtime.with_raw(Vec::new(), |raw| raw.device_ptr(&outside)) };
    match outcome {
        Ok(Err(Error::InvalidArgument { .. })) => {}
        other => panic!("expected an invalid-argument refusal, got {other:?}"),
    }
}

#[cfg(feature = "gpu-tests")]
#[test]
fn a_buffer_can_be_registered_after_the_scope_opens() {
    let mut runtime = crate::Runtime::open(0, crate::Config::default()).unwrap();
    let declared = runtime.alloc_zeroed(64, 256).unwrap();
    let late = runtime.alloc_zeroed(128, 256).unwrap();
    // SAFETY: the closure only reads addresses; the late buffer is registered before its address
    // is used, and both allocations outlive the scope.
    let (late_ptr, declared_ptr) = unsafe {
        runtime
            .with_raw(vec![RawResource::Buffer(&declared)], |raw| {
                let late_ptr = raw.register(&late).unwrap();
                // Registration is idempotent and makes the buffer readable through the scope.
                assert_eq!(raw.register(&late).unwrap(), late_ptr);
                assert_eq!(raw.device_ptr(&late).unwrap(), late_ptr);
                (late_ptr, raw.device_ptr(&declared).unwrap())
            })
            .unwrap()
    };
    assert_ne!(late_ptr, 0);
    assert_ne!(late_ptr, declared_ptr);
    runtime.synchronize().unwrap();
}

#[cfg(feature = "gpu-tests")]
#[test]
fn registering_another_runtimes_buffer_is_refused() {
    let mut first = crate::Runtime::open(0, crate::Config::default()).unwrap();
    let mut second = crate::Runtime::open(0, crate::Config::default()).unwrap();
    let foreign = second.alloc_zeroed(64, 256).unwrap();
    // SAFETY: the closure issues no CUDA call; it only asks to register another runtime's buffer.
    let outcome = unsafe { first.with_raw(Vec::new(), |raw| raw.register(&foreign)) };
    assert!(
        matches!(outcome, Ok(Err(crate::Error::ForeignDomain))),
        "expected a foreign-domain refusal, got {outcome:?}"
    );
    first.synchronize().unwrap();
    second.synchronize().unwrap();
}

#[cfg(feature = "gpu-tests")]
#[test]
fn a_scope_can_allocate_without_taking_the_runtime_back() {
    let mut runtime = crate::Runtime::open(0, crate::Config::default()).unwrap();
    // SAFETY: the closure allocates on this scope's stream and reads the pointer back.
    let allocated = unsafe {
        runtime
            .with_raw(Vec::new(), |raw| {
                let allocated = raw.alloc_zeroed(256, 256).unwrap();
                // A fresh allocation becomes addressable by registering it, exactly like one the
                // caller declared up front.
                assert_eq!(
                    raw.register(&allocated).unwrap(),
                    raw.device_ptr(&allocated).unwrap()
                );
                raw.device_ptr(&allocated).unwrap()
            })
            .unwrap()
    };
    assert_ne!(allocated, 0);
    runtime.synchronize().unwrap();
}
