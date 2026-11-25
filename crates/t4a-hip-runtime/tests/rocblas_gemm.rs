//! tenferro-rs#2489 stage 1: rocBLAS GEMM and strided-batched GEMM through the `Rocblas`
//! seam, on a `t4a-hip-runtime` stream and pool, matched against a CPU reference.
//!
//! The request cases are those of tenferro-gpu's `tests/seam_vendor.rs` (plain, one
//! transposed operand, strided-batched, an impossible leading dimension, sizes above
//! `i32::MAX`), run for f32, f64, c32 and c64, plus the ownership checks the CUDA seam
//! gets from its runtime: the stream binding, the pool-owned workspace and a foreign
//! runtime's buffer.
//!
//! Results are compared element by element against `|c - c_ref| <= f * k * eps * |a||b|`
//! with `eps` of the dtype under test (f64 epsilon for f64 and c64), not bit for bit:
//! the library's summation order differs from the reference's.
//!
//! Without a HIP device each test prints the typed reason and returns, unless
//! `T4A_REQUIRE_HIP=1` makes that a failure.

mod support;

use support::rocblas_seam::{DType, GemmRequest, Rocblas, SeamError};
use support::{decode, encode, reference, round, runtime, values, worst_ratio, Operand};
use t4a_hip_runtime::sys::{mapped_hip_runtimes, RocblasTable, RocsolverTable};
use t4a_hip_runtime::{Error, Runtime};

/// One GEMM layout: extents, transpose flags, batch count and padding of the leading
/// dimensions and batch strides beyond the dense minimum.
#[derive(Clone, Copy, Debug)]
struct Case {
    m: usize,
    n: usize,
    k: usize,
    trans_a: bool,
    trans_b: bool,
    batches: usize,
    ld_pad: usize,
    stride_pad: usize,
}

impl Case {
    fn plain(m: usize, n: usize, k: usize) -> Case {
        Case {
            m,
            n,
            k,
            trans_a: false,
            trans_b: false,
            batches: 1,
            ld_pad: 0,
            stride_pad: 0,
        }
    }
}

/// Run `case` for `dtype` through the seam and return the worst error ratio against
/// the bound (at most 1 passes).
fn run(runtime: &mut Runtime, blas: &mut Rocblas, dtype: DType, case: Case) -> f64 {
    let Case {
        m,
        n,
        k,
        trans_a,
        trans_b,
        batches,
        ld_pad,
        stride_pad,
    } = case;
    // Stored shapes: `a` is (m, k), or (k, m) when transposed; `b` is (k, n) or (n, k).
    let (a_rows, a_cols) = if trans_a { (k, m) } else { (m, k) };
    let (b_rows, b_cols) = if trans_b { (n, k) } else { (k, n) };
    let (lda, ldb, ldc) = (a_rows + ld_pad, b_rows + ld_pad, m + ld_pad);
    let stride_a = lda * a_cols + stride_pad;
    let stride_b = ldb * b_cols + stride_pad;
    let stride_c = ldc * n + stride_pad;
    let a = values(dtype, stride_a * batches, 1);
    let b = values(dtype, stride_b * batches, 2);
    let a_buf = runtime.upload(&encode(dtype, &a)).expect("upload a");
    let b_buf = runtime.upload(&encode(dtype, &b)).expect("upload b");
    let c_buf = runtime
        .alloc_zeroed(stride_c * batches * dtype.size())
        .expect("allocate c");
    blas.gemm(
        runtime,
        &GemmRequest {
            dtype,
            a: &a_buf,
            trans_a,
            b: &b_buf,
            trans_b,
            c: &c_buf,
            shape: (m as i64, n as i64, k as i64),
            leading: (lda as i64, ldb as i64, ldc as i64),
            batches: (
                batches as i64,
                stride_a as i64,
                stride_b as i64,
                stride_c as i64,
            ),
        },
    )
    .unwrap_or_else(|error| panic!("{dtype:?} {case:?}: gemm failed: {error:?}"));
    let observed = decode(dtype, &runtime.download(&c_buf).expect("download c"));
    let mut worst = 0.0f64;
    for batch in 0..batches {
        let (expected, magnitude) = reference(
            &Operand {
                values: &a[batch * stride_a..],
                ld: lda,
                transposed: trans_a,
            },
            &Operand {
                values: &b[batch * stride_b..],
                ld: ldb,
                transposed: trans_b,
            },
            m,
            n,
            k,
        );
        let got: Vec<_> = (0..n)
            .flat_map(|col| {
                let start = batch * stride_c + col * ldc;
                observed[start..start + m].iter().copied()
            })
            .collect();
        let ratio = worst_ratio(dtype, &got, &expected, &magnitude, k);
        worst = worst.max(ratio);
        // The padding between columns and batches is never written.
        for col in 0..n {
            let start = batch * stride_c + col * ldc;
            assert!(
                observed[start + m..start + ldc]
                    .iter()
                    .all(|&v| v == (0.0, 0.0)),
                "{dtype:?} {case:?}: gemm wrote into the leading-dimension padding"
            );
        }
    }
    worst
}

fn check(runtime: &mut Runtime, blas: &mut Rocblas, case: Case) {
    for dtype in DType::ALL {
        let ratio = run(runtime, blas, dtype, case);
        eprintln!("{dtype:?} {case:?}: worst |c - c_ref| / bound = {ratio:.3e}");
        assert!(
            ratio <= 1.0,
            "{dtype:?} {case:?}: error exceeds f*k*eps*|a||b| by {ratio:.3e}x"
        );
    }
}

fn setup() -> Option<(Runtime, Rocblas)> {
    let mut runtime = runtime()?;
    let blas = Rocblas::new(&mut runtime).expect("rocBLAS handle on the runtime's stream");
    Some((runtime, blas))
}

#[test]
fn the_device_and_libraries_are_recorded() {
    let Some((runtime, blas)) = setup() else {
        return;
    };
    let device = runtime.device();
    let rocblas = RocblasTable::load().expect("rocBLAS loaded");
    let rocsolver = RocsolverTable::load().expect("rocSOLVER loads from the same prefix");
    eprintln!(
        "device:            {} ({})",
        device.name, device.gcn_arch_name
    );
    eprintln!("warpSize:          {}", device.warp_size);
    eprintln!("integrated:        {}", device.integrated);
    eprintln!("totalGlobalMem:    {} bytes", device.total_global_mem);
    eprintln!("memory pools:      {}", device.memory_pools);
    eprintln!("hipRuntimeVersion: {}", device.runtime_version);
    eprintln!("hipDriverVersion:  {}", device.driver_version);
    eprintln!("HSA override:      {:?}", device.gfx_override);
    eprintln!("libamdhip64:       {}", device.library.display());
    eprintln!(
        "rocBLAS:           {} ({})",
        rocblas.version(),
        rocblas.library_path().display()
    );
    eprintln!("rocSOLVER:         {}", rocsolver.library_path().display());
    eprintln!(
        "ROCBLAS_USE_HIPBLASLT={:?} ROCBLAS_LAYER={:?}",
        std::env::var("ROCBLAS_USE_HIPBLASLT").ok(),
        std::env::var("ROCBLAS_LAYER").ok()
    );
    eprintln!(
        "workspace:         {} bytes from the pool",
        blas.workspace_bytes()
    );
    assert!(!device.gcn_arch_name.is_empty());
    assert!(device.warp_size == 32 || device.warp_size == 64);
    assert!(device.memory_pools);
    assert_eq!(
        rocblas.library_path().parent(),
        device.library.parent(),
        "rocBLAS must come from the HIP runtime's prefix"
    );
    // Stage-1 stop condition: rocBLAS and rocSOLVER pull `libamdhip64` by soname. If that
    // mapped a second file, two HIP runtimes would share the process.
    let mapped = mapped_hip_runtimes();
    assert_eq!(mapped.len(), 1, "HIP runtimes mapped: {mapped:?}");
}

#[test]
fn the_handle_runs_on_the_runtime_stream_with_a_pool_owned_workspace() {
    let Some(mut runtime) = runtime() else {
        return;
    };
    let before = runtime.pool_stats().expect("pool stats").used_bytes;
    let blas = Rocblas::new(&mut runtime).expect("handle");
    runtime.synchronize().expect("drain");
    let after = runtime.pool_stats().expect("pool stats").used_bytes;
    assert_eq!(blas.bound_stream() as usize as u64, runtime.stream_id());
    assert!(
        !blas.manages_own_memory(),
        "rocBLAS must not keep device memory outside the runtime's pool"
    );
    assert!(blas.workspace_bytes() >= Rocblas::INITIAL_WORKSPACE);
    assert!(
        after >= before + blas.workspace_bytes() as u64,
        "the pool must account for the workspace: {before} -> {after} bytes, workspace {}",
        blas.workspace_bytes()
    );
}

#[test]
fn a_plain_gemm_matches_the_host() {
    let Some((mut runtime, mut blas)) = setup() else {
        return;
    };
    check(&mut runtime, &mut blas, Case::plain(37, 29, 41));
}

#[test]
fn a_transposed_operand_matches_the_host() {
    let Some((mut runtime, mut blas)) = setup() else {
        return;
    };
    for (trans_a, trans_b) in [(true, false), (false, true), (true, true)] {
        check(
            &mut runtime,
            &mut blas,
            Case {
                trans_a,
                trans_b,
                ..Case::plain(16, 12, 20)
            },
        );
    }
}

#[test]
fn a_strided_batched_gemm_matches_the_host() {
    let Some((mut runtime, mut blas)) = setup() else {
        return;
    };
    check(
        &mut runtime,
        &mut blas,
        Case {
            batches: 3,
            ..Case::plain(8, 6, 5)
        },
    );
    // Padded leading dimensions and batch strides, with a transposed operand.
    check(
        &mut runtime,
        &mut blas,
        Case {
            trans_b: true,
            batches: 4,
            ld_pad: 3,
            stride_pad: 7,
            ..Case::plain(9, 7, 11)
        },
    );
}

#[test]
fn a_gemm_past_one_tile_matches_the_host() {
    let Some((mut runtime, mut blas)) = setup() else {
        return;
    };
    // Extents that are not multiples of any tile size, so edge tiles run too.
    check(&mut runtime, &mut blas, Case::plain(129, 257, 513));
}

#[test]
fn the_f64_bound_rejects_a_result_computed_at_f32_precision() {
    // The f64 check must not be satisfiable by a library that computed in f32: round
    // a correct dgemm result through f32 and the same bound has to fail.
    let Some((mut runtime, mut blas)) = setup() else {
        return;
    };
    let (m, n, k) = (24usize, 24usize, 64usize);
    let a = values(DType::F64, m * k, 11);
    let b = values(DType::F64, k * n, 12);
    let a_buf = runtime.upload(&encode(DType::F64, &a)).expect("a");
    let b_buf = runtime.upload(&encode(DType::F64, &b)).expect("b");
    let c_buf = runtime.alloc_zeroed(m * n * 8).expect("c");
    blas.gemm(
        &mut runtime,
        &GemmRequest {
            dtype: DType::F64,
            a: &a_buf,
            trans_a: false,
            b: &b_buf,
            trans_b: false,
            c: &c_buf,
            shape: (m as i64, n as i64, k as i64),
            leading: (m as i64, k as i64, m as i64),
            batches: (1, 0, 0, 0),
        },
    )
    .expect("dgemm");
    let observed = decode(DType::F64, &runtime.download(&c_buf).expect("c"));
    let a_op = Operand {
        values: &a,
        ld: m,
        transposed: false,
    };
    let b_op = Operand {
        values: &b,
        ld: k,
        transposed: false,
    };
    let (expected, magnitude) = reference(&a_op, &b_op, m, n, k);
    let exact = worst_ratio(DType::F64, &observed, &expected, &magnitude, k);
    let through_f32: Vec<_> = observed
        .iter()
        .map(|&(re, im)| (round(DType::F32, re), im))
        .collect();
    let degraded = worst_ratio(DType::F64, &through_f32, &expected, &magnitude, k);
    eprintln!("dgemm: {exact:.3e} of the f64 bound; rounded through f32: {degraded:.3e}");
    assert!(exact <= 1.0, "dgemm misses the f64 bound: {exact:.3e}");
    assert!(
        degraded > 1.0,
        "an f32-precision result must fail the f64 bound, got {degraded:.3e}"
    );
}

#[test]
fn an_impossible_leading_dimension_is_refused() {
    let Some((mut runtime, mut blas)) = setup() else {
        return;
    };
    let a_buf = runtime
        .upload(&encode(DType::F64, &[(1.0, 0.0); 4]))
        .expect("a");
    let b_buf = runtime
        .upload(&encode(DType::F64, &[(1.0, 0.0); 4]))
        .expect("b");
    let c_buf = runtime.alloc_zeroed(4 * 8).expect("c");
    let refused = blas.gemm(
        &mut runtime,
        &GemmRequest {
            dtype: DType::F64,
            a: &a_buf,
            trans_a: false,
            b: &b_buf,
            trans_b: false,
            c: &c_buf,
            shape: (4, 4, 4),
            leading: (2, 2, 2),
            batches: (1, 0, 0, 0),
        },
    );
    assert!(matches!(
        refused,
        Err(SeamError::Unsupported { op: "gemm", .. })
    ));
}

#[test]
fn an_operand_buffer_smaller_than_its_layout_is_refused() {
    // The CUDA seam does not check this; here a 4x4 product over 2x2 buffers is refused
    // before rocBLAS reads past them.
    let Some((mut runtime, mut blas)) = setup() else {
        return;
    };
    let small = runtime.alloc_zeroed(4 * 8).expect("2x2");
    let refused = blas.gemm(
        &mut runtime,
        &GemmRequest {
            dtype: DType::F64,
            a: &small,
            trans_a: false,
            b: &small,
            trans_b: false,
            c: &small,
            shape: (4, 4, 4),
            leading: (4, 4, 4),
            batches: (1, 0, 0, 0),
        },
    );
    match refused {
        Err(SeamError::Unsupported { op, detail }) => {
            assert_eq!(op, "gemm");
            assert!(detail.contains("operand a needs 128 bytes"), "{detail}");
        }
        other => panic!("expected a typed refusal, got {other:?}"),
    }
}

#[test]
fn sizes_that_do_not_fit_the_rocblas_entry_points_are_refused_before_launch() {
    let Some((mut runtime, mut blas)) = setup() else {
        return;
    };
    let a_buf = runtime.alloc_zeroed(4 * 8).expect("a");
    let b_buf = runtime.alloc_zeroed(4 * 8).expect("b");
    let c_buf = runtime.alloc_zeroed(4 * 8).expect("c");
    let too_large = i64::from(i32::MAX) + 1;
    let request = |shape: (i64, i64, i64),
                   leading: (i64, i64, i64),
                   batches: (i64, i64, i64, i64)| GemmRequest {
        dtype: DType::F64,
        a: &a_buf,
        trans_a: false,
        b: &b_buf,
        trans_b: false,
        c: &c_buf,
        shape,
        leading,
        batches,
    };
    // Every leading dimension still covers its operand's row count, so the request
    // reaches the narrowing step instead of being refused by the layout checks.
    let cases = [
        (
            "m",
            request((too_large, 1, 1), (too_large, 1, too_large), (1, 0, 0, 0)),
        ),
        ("n", request((1, too_large, 1), (1, 1, 1), (1, 0, 0, 0))),
        (
            "k",
            request((1, 1, too_large), (1, too_large, 1), (1, 0, 0, 0)),
        ),
        ("lda", request((1, 1, 1), (too_large, 1, 1), (1, 0, 0, 0))),
        ("ldb", request((1, 1, 1), (1, too_large, 1), (1, 0, 0, 0))),
        ("ldc", request((1, 1, 1), (1, 1, too_large), (1, 0, 0, 0))),
        (
            "batch count",
            request((1, 1, 1), (1, 1, 1), (too_large, 0, 0, 0)),
        ),
    ];
    for (field, call) in cases {
        match blas.gemm(&mut runtime, &call) {
            Err(SeamError::Unsupported { op, detail }) => {
                assert_eq!(op, "gemm");
                assert!(
                    detail.contains(&format!("{field} = {too_large}")),
                    "{field}: expected the error to name this field and value, got {detail}"
                );
            }
            other => panic!("{field}: expected a typed Unsupported error, got {other:?}"),
        }
    }
}

#[test]
fn a_foreign_runtime_or_its_buffers_are_refused() {
    let Some((mut runtime, mut blas)) = setup() else {
        return;
    };
    let mut other = Runtime::open(0).expect("a second runtime on the same device");
    assert_ne!(other.domain_id(), runtime.domain_id());
    assert_ne!(other.stream_id(), runtime.stream_id());
    let local = runtime.alloc_zeroed(8).expect("local");
    let foreign = other.alloc_zeroed(8).expect("foreign");
    let request = |a| GemmRequest {
        dtype: DType::F64,
        a,
        trans_a: false,
        b: &local,
        trans_b: false,
        c: &local,
        shape: (1, 1, 1),
        leading: (1, 1, 1),
        batches: (1, 0, 0, 0),
    };
    assert!(matches!(
        blas.gemm(&mut runtime, &request(&foreign)),
        Err(SeamError::Runtime(Error::ForeignDomain))
    ));
    // The handle is bound to `runtime`'s stream; submitting through another runtime
    // would order the call on the wrong stream.
    assert!(matches!(
        blas.gemm(&mut other, &request(&local)),
        Err(SeamError::Runtime(Error::ForeignDomain))
    ));
}

#[test]
fn a_nan_result_fails_the_bound_instead_of_vanishing() {
    // Needs no device: the checker itself. `f64::max` drops a NaN, so a NaN element would
    // otherwise pass as "no error".
    let expected = [(1.0, 0.0), (2.0, 0.0)];
    let magnitude = [1.0, 2.0];
    for observed in [[(f64::NAN, 0.0), (2.0, 0.0)], [(1.0, 0.0), (f64::NAN, 0.0)]] {
        let ratio = worst_ratio(DType::F64, &observed, &expected, &magnitude, 4);
        assert!(
            ratio > 1.0,
            "a NaN element must fail the bound, got {ratio}"
        );
    }
}
