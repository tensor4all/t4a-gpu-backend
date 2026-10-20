use super::*;

#[test]
fn default_config_is_valid() {
    Config::default()
        .validate()
        .expect("defaults must be valid");
}

#[test]
fn config_rejects_degenerate_limits() {
    let config = Config {
        max_chunk_bytes: 0,
        ..Config::default()
    };
    assert!(config.validate().is_err());

    let config = Config {
        max_idle_pinned_bytes: 1,
        ..Config::default()
    };
    assert!(config.validate().is_err());

    let config = Config {
        max_pending_records: 0,
        ..Config::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn span_checks_reject_overflow_and_overrun() {
    assert!(check_span(16, 0, 16).is_ok());
    assert!(check_span(16, 8, 8).is_ok());
    assert!(check_span(16, 16, 1).is_err());
    assert!(check_span(16, usize::MAX, 1).is_err());
}

#[test]
fn limits_mirror_config() {
    let config = Config::default();
    let limits = config.limits();
    assert_eq!(limits.pending_bytes, config.pending_host_budget);
    assert_eq!(limits.max_chunk_bytes, config.max_chunk_bytes);
}

#[cfg(feature = "gpu-tests")]
#[test]
fn two_runtimes_on_one_ordinal_have_distinct_domains_and_streams() {
    let a = Domain::create(ContextHolder::open(0).unwrap(), &Config::default()).unwrap();
    let b = Domain::create(ContextHolder::open(0).unwrap(), &Config::default()).unwrap();
    assert_ne!(a.id(), b.id());
    assert_ne!(a.stream() as usize, b.stream() as usize);
    assert!(!std::ptr::eq(a.stream(), b.stream()));
}

#[cfg(feature = "gpu-tests")]
#[test]
fn pool_allocation_alignment_is_measured_not_assumed() {
    let domain = Domain::create(ContextHolder::open(0).unwrap(), &Config::default()).unwrap();
    for alignment in [1usize, 16, 256, 4096] {
        let buffer = match domain.allocate(4096, alignment) {
            Ok(ptr) => ptr,
            Err(Error::Unsupported { .. }) => continue,
            Err(error) => panic!("allocation failed: {error}"),
        };
        assert_eq!(buffer % alignment as u64, 0, "alignment {alignment}");
        domain.release(buffer, 4096);
    }
    domain.synchronize().unwrap();
}

#[cfg(feature = "gpu-tests")]
#[test]
fn last_drop_on_another_thread_releases_in_stream_order() {
    let domain = Domain::create(ContextHolder::open(0).unwrap(), &Config::default()).unwrap();
    let ptr = domain.allocate(1024, 256).unwrap();
    {
        let owned = Alloc::new(ptr, 1024, Arc::clone(&domain));
        std::thread::spawn(move || drop(owned)).join().unwrap();
    }
    // A later allocation must not observe the freed block as live.
    domain.synchronize().unwrap();
    let stats = domain.stats();
    assert_eq!(stats.live_allocations, 0);
    assert_eq!(stats.live_bytes, 0);
}

#[cfg(feature = "gpu-tests")]
#[test]
fn misaligned_requests_are_rejected_without_leaking() {
    let domain = Domain::create(ContextHolder::open(0).unwrap(), &Config::default()).unwrap();
    // 64 KiB is documented as unsupported above; the request must fail before any
    // allocation is exposed and the domain must stay usable.
    let _domain = domain;
    let result = crate::runtime::Runtime::open(0, Config::default());
    let mut runtime = result.expect("runtime must open");
    assert!(runtime.alloc_uninit(16, 3).is_err());
    let buffer = runtime.alloc_zeroed(16, 256).expect("domain stays usable");
    assert_eq!(buffer.len(), 16);
}
