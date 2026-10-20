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
