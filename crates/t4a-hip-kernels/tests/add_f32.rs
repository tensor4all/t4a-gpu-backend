//! M2: `t4a_cuda_add` (the CUDA crate's own source, unchanged) compiled by hiprtc for the
//! device's gfx target and run on it, compared bit for bit with the CPU.
//!
//! Without a HIP device the test prints the typed reason and returns, unless
//! `T4A_REQUIRE_HIP=1` makes that a failure.

use t4a_hip_kernels::abi::{DType, Entry};
use t4a_hip_kernels::runtime::{Error as RuntimeError, Runtime};
use t4a_hip_kernels::{elementwise, Binary, Error, Program};

fn runtime() -> Option<Runtime> {
    match Runtime::open(0) {
        Ok(runtime) => Some(runtime),
        Err(error @ (RuntimeError::Unavailable { .. } | RuntimeError::InvalidDevice { .. })) => {
            assert!(
                std::env::var("T4A_REQUIRE_HIP").map_or(true, |v| v != "1"),
                "T4A_REQUIRE_HIP=1 but no HIP device is usable: {error}"
            );
            eprintln!("skipped: {error}");
            None
        }
        Err(other) => panic!("opening HIP device 0 failed: {other}"),
    }
}

fn bytes<T: Copy>(values: &[T]) -> Vec<u8> {
    let size = std::mem::size_of_val(values);
    // SAFETY: plain-old-data values viewed as bytes.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), size) }.to_vec()
}

fn words<const N: usize>(raw: &[u8]) -> Vec<[u8; N]> {
    raw.chunks_exact(N).map(|c| c.try_into().unwrap()).collect()
}

/// Run `add` over `lhs + rhs` and return the output bytes.
fn run(
    runtime: &mut Runtime,
    program: &Program,
    dtype: DType,
    lhs: &[u8],
    rhs: &[u8],
    len: i64,
) -> Vec<u8> {
    let out = runtime.alloc_zeroed(lhs.len()).unwrap();
    let a = runtime.upload(lhs).unwrap();
    let b = runtime.upload(rhs).unwrap();
    elementwise(
        runtime,
        program,
        Entry::Add,
        dtype,
        Binary {
            out: &out,
            lhs: &a,
            rhs: &b,
        },
        len,
    )
    .unwrap();
    runtime.download(&out).unwrap()
}

#[test]
fn add_matches_the_cpu_bit_for_bit() {
    let Some(mut runtime) = runtime() else { return };
    let program = Program::compile_entry(&runtime, Entry::Add).expect("hiprtc compile and load");
    let report = program.report();
    assert!(report.target.starts_with("gfx"), "{}", report.target);
    eprintln!(
        "target {} options {:?} compile {:?} load {:?} code {} bytes override {:?}",
        report.target,
        report.options,
        report.compile_time,
        report.load_time,
        report.code_bytes,
        report.gfx_override
    );

    // The shape checked by hand: [1, 2, 3, 4, 5] + [10, 20, 30, 40, 50] = [11, 22, 33, 44, 55].
    let lhs = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let rhs = [10.0f32, 20.0, 30.0, 40.0, 50.0];
    let got = run(
        &mut runtime,
        &program,
        DType::F32,
        &bytes(&lhs),
        &bytes(&rhs),
        5,
    );
    let got: Vec<f32> = words::<4>(&got)
        .into_iter()
        .map(f32::from_ne_bytes)
        .collect();
    assert_eq!(got, [11.0, 22.0, 33.0, 44.0, 55.0]);

    // Full mantissas across a block boundary (256) and a ragged tail: 700 elements.
    let n = 700usize;
    let lhs: Vec<f32> = (0..n).map(|i| (i as f32 * 0.123_456_79).sin()).collect();
    let rhs: Vec<f32> = (0..n)
        .map(|i| (i as f32 * 0.987_654_3).cos() / 3.0)
        .collect();
    let got = run(
        &mut runtime,
        &program,
        DType::F32,
        &bytes(&lhs),
        &bytes(&rhs),
        n as i64,
    );
    for (i, word) in words::<4>(&got).into_iter().enumerate() {
        assert_eq!(word, (lhs[i] + rhs[i]).to_ne_bytes(), "f32 element {i}");
    }

    // The same entry at f64 and (wrapping) i32 through the dtype tag.
    let lhs: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).exp().recip()).collect();
    let rhs: Vec<f64> = (0..n).map(|i| (i as f64).sqrt() / 7.0).collect();
    let got = run(
        &mut runtime,
        &program,
        DType::F64,
        &bytes(&lhs),
        &bytes(&rhs),
        n as i64,
    );
    for (i, word) in words::<8>(&got).into_iter().enumerate() {
        assert_eq!(word, (lhs[i] + rhs[i]).to_ne_bytes(), "f64 element {i}");
    }
    let lhs: Vec<i32> = (0..n as i32).map(|i| i32::MAX - i).collect();
    let rhs: Vec<i32> = (0..n as i32).map(|i| 2 * i).collect();
    let got = run(
        &mut runtime,
        &program,
        DType::I32,
        &bytes(&lhs),
        &bytes(&rhs),
        n as i64,
    );
    for (i, word) in words::<4>(&got).into_iter().enumerate() {
        assert_eq!(
            word,
            lhs[i].wrapping_add(rhs[i]).to_ne_bytes(),
            "i32 element {i}"
        );
    }
}

#[test]
fn the_planner_and_buffer_checks_fire_before_any_launch() {
    let Some(mut runtime) = runtime() else { return };
    let program = Program::compile_entry(&runtime, Entry::Add).expect("compile");
    let small = runtime.upload(&[0u8; 4]).unwrap();
    let ops = Binary {
        out: &small,
        lhs: &small,
        rhs: &small,
    };
    // Bool add is not an ABI entry: the CUDA crate's planner refuses it.
    let error = elementwise(&mut runtime, &program, Entry::Add, DType::Bool, ops, 1).unwrap_err();
    assert!(matches!(error, Error::Plan(_)), "{error}");
    let error = elementwise(&mut runtime, &program, Entry::Add, DType::F32, ops, 2).unwrap_err();
    assert!(matches!(error, Error::BufferTooSmall { .. }), "{error}");
    // An empty request is a validated no-op.
    elementwise(&mut runtime, &program, Entry::Add, DType::F32, ops, 0).unwrap();
}

/// How far the reuse goes today: which of the CUDA crate's translation units hiprtc
/// builds for this device. Recorded, not gated: a module that fails is M3's finding to
/// fix, and the list below is what the PR body quotes. About two minutes of hiprtc,
/// so opt in with `--ignored`.
#[test]
#[ignore = "two minutes of hiprtc; run with --ignored"]
fn survey_every_cuda_module_under_hiprtc() {
    let Some(runtime) = runtime() else { return };
    let mut failed = Vec::new();
    for module in t4a_hip_kernels::abi::source::MODULES {
        match Program::compile(&runtime, module) {
            Ok(program) => eprintln!(
                "SURVEY {:<16} ok   {:>8.1?} {:>7} bytes",
                module.name,
                program.report().compile_time,
                program.report().code_bytes
            ),
            Err(Error::Compile { log, .. }) => {
                eprintln!(
                    "SURVEY {:<16} FAIL {}",
                    module.name,
                    log.lines().find(|l| l.contains("error")).unwrap_or("")
                );
                failed.push(module.name);
            }
            Err(other) => panic!("{}: {other}", module.name),
        }
    }
    assert!(failed.is_empty(), "modules hiprtc rejects: {failed:?}");
}
