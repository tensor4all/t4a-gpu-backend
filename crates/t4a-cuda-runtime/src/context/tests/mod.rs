use super::*;
#[cfg(feature = "gpu-tests")]
use cudarc::driver::result as drv;

#[test]
fn missing_stack_preconditions_are_typed() {
    // Opening a device that does not exist must never panic and must produce a
    // typed error, whether the driver is present or not.
    match ContextHolder::open(usize::MAX) {
        Err(Error::InvalidDevice { .. } | Error::DriverUnavailable { .. }) => {}
        Err(other) => panic!("expected a typed device error, got {other}"),
        Ok(_) => panic!("opening usize::MAX must never succeed"),
    }
}

#[cfg(feature = "gpu-tests")]
#[test]
fn activate_restores_the_previous_context() {
    let holder = ContextHolder::open(0).expect("device 0 must be usable");
    let before = drv::ctx::get_current().expect("cuCtxGetCurrent must succeed");
    {
        let _scope = holder.activate().expect("activation must succeed");
        assert_eq!(
            drv::ctx::get_current().expect("cuCtxGetCurrent must succeed"),
            Some(holder.raw())
        );
    }
    assert_eq!(
        drv::ctx::get_current().expect("cuCtxGetCurrent must succeed"),
        before
    );
}

#[cfg(feature = "gpu-tests")]
#[test]
fn activate_restores_the_previous_context_on_unwind() {
    let holder = ContextHolder::open(0).expect("device 0 must be usable");
    let before = drv::ctx::get_current().expect("cuCtxGetCurrent must succeed");
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _scope = holder.activate().expect("activation must succeed");
        panic!("unwind through the context scope");
    }));
    assert!(result.is_err());
    assert_eq!(
        drv::ctx::get_current().expect("cuCtxGetCurrent must succeed"),
        before
    );
}

#[cfg(feature = "gpu-tests")]
#[test]
fn pool_capability_is_probed() {
    assert!(supports_memory_pool(0).expect("probe must succeed"));
    assert!(device_count().expect("count must succeed") >= 1);
}
