use super::{Arg, LaunchConfig};

#[test]
fn launch_dimensions_must_be_non_zero() {
    assert!(LaunchConfig::new(1, 1).validate().is_ok());
    let mut config = LaunchConfig::new(1, 1);
    config.grid.0 = 0;
    assert!(config.validate().is_err());
    let mut config = LaunchConfig::new(1, 1);
    config.block.2 = 0;
    assert!(config.validate().is_err());
}

#[test]
fn default_launch_is_one_dimensional() {
    let config = LaunchConfig::new(8, 32);
    assert_eq!(config.grid, (8, 1, 1));
    assert_eq!(config.block, (32, 1, 1));
    assert_eq!(config.shared_bytes, 0);
}

#[test]
fn scalar_arguments_keep_their_values() {
    let args = [Arg::U32(7), Arg::F64(0.5), Arg::I32(-1)];
    assert!(matches!(args[0], Arg::U32(7)));
    assert!(matches!(args[1], Arg::F64(value) if value == 0.5));
    assert!(matches!(args[2], Arg::I32(-1)));
}
