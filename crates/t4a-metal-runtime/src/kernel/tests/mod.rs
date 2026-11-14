use crate::kernel::LaunchConfig;

#[test]
fn threads_covers_every_index_exactly_once() {
    let config = LaunchConfig::threads(7);
    assert_eq!(config.threadgroups, (7, 1, 1));
    assert_eq!(config.threads_per_threadgroup, (1, 1, 1));
    let grid = u64::from(config.threadgroups.0) * u64::from(config.threads_per_threadgroup.0);
    assert_eq!(grid, 7, "the grid is the element count, not its square");
    assert!(config.validate(1024).is_ok());
}

#[test]
fn threads_is_legal_above_the_per_threadgroup_limit() {
    // The limit bounds one threadgroup, so a one-thread group never reaches it.
    assert!(LaunchConfig::threads(64 * 1024).validate(1024).is_ok());
}

#[test]
fn a_wide_threadgroup_above_the_limit_names_the_limit() {
    match LaunchConfig::tiled(1, 1025).validate(1024) {
        Err(crate::error::Error::InvalidArgument { detail }) => {
            assert!(detail.contains("1024"), "{detail}");
            assert!(detail.contains("1025"), "{detail}");
        }
        other => panic!("expected an invalid-argument error, got {other:?}"),
    }
}

#[test]
fn zero_dimensions_are_rejected() {
    assert!(LaunchConfig::tiled(0, 4).validate(1024).is_err());
    assert!(LaunchConfig::tiled(4, 0).validate(1024).is_err());
    assert!(LaunchConfig::threads(0).validate(1024).is_err());
}
