use super::{Arg, LaunchConfig, Parameters, Resolved, Slot};

/// The device limit used by the encoder tests; the encoder itself is pure.
const LIMIT: usize = 32_764;

fn word(value: u64, size: usize) -> Resolved<'static> {
    Resolved::Word { value, size }
}

fn block(bytes: &[u8], align: usize) -> Resolved<'_> {
    Resolved::Bytes { bytes, align }
}

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

#[test]
fn parameters_keep_declaration_order_and_size() {
    let layout = [0u8; 144];
    let args = [word(7, 4), block(&layout, 8), word(9, 8)];
    let parameters = Parameters::encode(&args, LaunchConfig::new(1, 1), LIMIT).unwrap();
    assert_eq!(parameters.count, 3);
    assert_eq!(parameters.slots[0], [7]);
    assert_eq!(parameters.slots[2], [9]);
    assert!(matches!(parameters.kinds[0], Slot::Word));
    assert!(matches!(parameters.kinds[1], Slot::Block { offset: 0 }));
    assert!(matches!(parameters.kinds[2], Slot::Word));
    // 4-byte scalar + 144-byte block + 8-byte scalar.
    assert_eq!(parameters.bytes, 156);
}

#[test]
fn by_value_blocks_are_copied_at_eight_byte_offsets() {
    let first = [1u8, 2, 3];
    let second = [4u8, 5, 6, 7, 8];
    let args = [block(&first, 1), block(&second, 8)];
    let mut parameters = Parameters::encode(&args, LaunchConfig::new(1, 1), LIMIT).unwrap();
    let (
        Slot::Block {
            offset: first_offset,
        },
        Slot::Block {
            offset: second_offset,
        },
    ) = (parameters.kinds[0], parameters.kinds[1])
    else {
        panic!("both arguments must be blocks");
    };
    assert_eq!(first_offset, 0);
    assert_eq!(second_offset, 8);
    assert_eq!(parameters.bytes, 8);
    // The blocks were copied, so the arguments may change after encoding.
    let first = [9u8, 9, 9];
    let second = [9u8; 5];
    assert_eq!(first, [9u8, 9, 9]);
    assert_eq!(second, [9u8; 5]);
    let params = parameters.param_ptrs();
    // SAFETY: each pointer addresses the copied block, whose first byte came from
    // the argument as it was at encode time.
    unsafe {
        assert_eq!(*params[0].cast::<u8>(), 1);
        assert_eq!(*params[1].cast::<u8>(), 4);
    }
}

#[test]
fn the_limit_counts_the_checked_abi_extent() {
    // A 4-byte scalar followed by an 8-byte-aligned block: the ABI pads the scalar
    // to 8 bytes, so a 4 + 144-byte launch needs 152 bytes, not 148.
    let layout = [0u8; 144];
    let args = [word(7, 4), block(&layout, 8)];
    match Parameters::encode(&args, LaunchConfig::new(1, 1), 148) {
        Err(super::Error::KernelParameterLimit {
            bytes: 152,
            limit: 148,
        }) => {}
        _ => panic!("the padded extent must be what the limit applies to"),
    }
    assert!(Parameters::encode(&args, LaunchConfig::new(1, 1), 152).is_ok());
}

#[test]
fn empty_by_value_blocks_are_rejected() {
    let args = [block(&[], 1)];
    assert!(matches!(
        Parameters::encode(&args, LaunchConfig::new(1, 1), LIMIT),
        Err(super::Error::InvalidArgument { .. })
    ));
}

#[test]
fn over_aligned_blocks_are_rejected() {
    let payload = [0u8; 8];
    for align in [0usize, 3, 16, 64] {
        let args = [block(&payload, align)];
        match Parameters::encode(&args, LaunchConfig::new(1, 1), LIMIT) {
            Err(super::Error::InvalidArgument { detail }) => {
                assert!(detail.contains("alignment"), "{detail}");
            }
            other => panic!("align {align} must be refused, got {:?}", other.is_ok()),
        }
    }
    for align in [1usize, 2, 4, 8] {
        let args = [block(&payload, align)];
        assert!(
            Parameters::encode(&args, LaunchConfig::new(1, 1), LIMIT).is_ok(),
            "align {align} must be accepted"
        );
    }
}

#[test]
fn parameters_above_the_device_limit_are_rejected() {
    let block4k = [0u8; 4096];
    let args = [block(&block4k, 8), block(&block4k, 8)];
    match Parameters::encode(&args, LaunchConfig::new(1, 1), 4096) {
        Err(super::Error::KernelParameterLimit { bytes, limit }) => {
            assert_eq!(bytes, 8192);
            assert_eq!(limit, 4096);
        }
        _ => panic!("a launch above the parameter limit must be refused"),
    }
}

#[test]
fn the_fused_block_fits_the_default_limit() {
    // The largest ABI parameter block the adoption needs is one 4 KiB block
    // beside a 144-byte layout and a handful of scalars.
    let fused = [0u8; 4096];
    let layout = [0u8; 144];
    let args = [
        block(&layout, 8),
        block(&fused, 8),
        word(1, 8),
        word(2, 8),
        word(3, 8),
    ];
    assert!(Parameters::encode(&args, LaunchConfig::new(1, 1), LIMIT).is_ok());
    assert!(Parameters::encode(&args, LaunchConfig::new(1, 1), 4096).is_err());
}
