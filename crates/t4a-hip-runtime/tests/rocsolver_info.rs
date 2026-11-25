//! The device-`info` trap (tenferro-rs#2489, retraction comment): rocSOLVER's
//! LAPACK-style entries write `info` (and `ipiv`) from a kernel, so both must be device
//! memory. A host address there does not return an error: `rocsolver::reset_info`
//! page-faults on it and the HIP runtime waits forever. That cost a day of blaming the
//! library.
//!
//! These tests pin the seam's answer: the typed entry only takes an [`InfoSlot`] (pool
//! memory by construction), the raw entry classifies every pointer with HIP and refuses a
//! host one before rocSOLVER runs, and every GPU call is under a watchdog, so a
//! regression shows up as a failure naming the trap rather than as a hung CI job.

mod support;

use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::Duration;

use support::rocblas_seam::{require_device, DType, InfoSlot, Rocblas, SeamError};
use support::{decode, encode, runtime, values, Value};
use t4a_hip_runtime::{PointerClass, Runtime};

/// Generous for an 8x8 factorisation; the trap never returns at all.
const WATCHDOG: Duration = Duration::from_secs(120);

/// Run `body` on its own thread and fail loudly if it does not return.
///
/// On a timeout the process is aborted after the message: a thread stuck in a faulted
/// HIP wait can also block the runtime's exit handlers, so ending the process is the
/// only way to turn the hang into a reported failure.
fn within_watchdog<T: Send + 'static>(
    what: &'static str,
    body: impl FnOnce() -> T + Send + 'static,
) -> T {
    let (sender, receiver) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        let _ = sender.send(body());
    });
    match receiver.recv_timeout(WATCHDOG) {
        Ok(value) => value,
        Err(RecvTimeoutError::Disconnected) => match worker.join() {
            Err(panic) => std::panic::resume_unwind(panic),
            Ok(()) => panic!("{what}: the worker ended without a result"),
        },
        Err(RecvTimeoutError::Timeout) => {
            eprintln!(
                "{what} did not return within {} s. A rocSOLVER kernel is waiting on a \
                 faulted address: this is the host-`info` trap (an `info` or `ipiv` \
                 pointer that is not device memory). Aborting so the failure is reported.",
                WATCHDOG.as_secs()
            );
            std::process::abort();
        }
    }
}

/// Factor `matrix` (column-major `n x n`) and return the factors, pivots and info.
fn factor(
    runtime: &mut Runtime,
    blas: &mut Rocblas,
    dtype: DType,
    n: usize,
    matrix: &[Value],
) -> (Vec<Value>, Vec<i32>, i32) {
    let a = runtime.upload(&encode(dtype, matrix)).expect("upload a");
    let ipiv = runtime.alloc_zeroed(n * 4).expect("ipiv");
    let info = InfoSlot::new(runtime).expect("info slot");
    blas.getrf(runtime, dtype, n as i64, &a, n as i64, &ipiv, &info)
        .unwrap_or_else(|error| panic!("{dtype:?} getrf: {error:?}"));
    let status = info.read(runtime).expect("read info");
    let factors = decode(dtype, &runtime.download(&a).expect("download a"));
    let pivots = runtime
        .download(&ipiv)
        .expect("download ipiv")
        .chunks_exact(4)
        .map(|word| i32::from_ne_bytes([word[0], word[1], word[2], word[3]]))
        .collect();
    (factors, pivots, status)
}

/// Check `P*A = L*U` for rocSOLVER's packed real factors against `4 n eps |L||U|`.
///
/// `P` is rebuilt from the sequential 1-based row interchanges in `pivots`.
fn assert_lu(matrix: &[Value], n: usize, lu: &[Value], pivots: &[i32]) {
    let mut permuted: Vec<f64> = matrix.iter().map(|&(re, _)| re).collect();
    for (row, &pivot) in pivots.iter().enumerate() {
        let other = usize::try_from(pivot - 1).expect("1-based pivot");
        assert!(other < n);
        for col in 0..n {
            permuted.swap(row + col * n, other + col * n);
        }
    }
    let at = |row: usize, col: usize| lu[row + col * n].0;
    for col in 0..n {
        for row in 0..n {
            let (mut sum, mut magnitude) = (0.0f64, 0.0f64);
            for inner in 0..=row.min(col) {
                let l = if inner == row { 1.0 } else { at(row, inner) };
                let u = at(inner, col);
                sum += l * u;
                magnitude += (l * u).abs();
            }
            let error = (sum - permuted[row + col * n]).abs();
            assert!(
                error <= 4.0 * n as f64 * f64::EPSILON * magnitude.max(f64::MIN_POSITIVE),
                "P*A != L*U at ({row}, {col}): error {error:e}, scale {magnitude:e}"
            );
        }
    }
}

#[test]
fn a_zero_pivot_is_reported_through_device_info() {
    within_watchdog("getrf with a device info slot", || {
        let Some(mut runtime) = runtime() else {
            return;
        };
        let mut blas = Rocblas::new(&mut runtime).expect("handle");
        let n = 5;
        for dtype in [DType::F32, DType::F64] {
            // Column 3 (1-based) is zero, so elimination meets an exactly zero pivot
            // there whatever rows are swapped.
            let mut matrix = values(dtype, n * n, 21);
            for row in 0..n {
                matrix[row + 2 * n] = (0.0, 0.0);
            }
            let (_, _, status) = factor(&mut runtime, &mut blas, dtype, n, &matrix);
            assert_eq!(status, 3, "{dtype:?}: info must name the first zero pivot");
        }
        assert!(!blas.manages_own_memory());
    });
}

#[test]
fn a_nonsingular_matrix_factors_with_info_zero() {
    within_watchdog("getrf on a nonsingular matrix", || {
        let Some(mut runtime) = runtime() else {
            return;
        };
        let mut blas = Rocblas::new(&mut runtime).expect("handle");
        let n = 8;
        let dtype = DType::F64;
        let matrix = values(dtype, n * n, 22);
        let (lu, pivots, status) = factor(&mut runtime, &mut blas, dtype, n, &matrix);
        assert_eq!(status, 0);
        assert_lu(&matrix, n, &lu, &pivots);
        // rocSOLVER's workspace, if it needed more than the initial buffer, came from the
        // runtime's pool as well.
        assert!(!blas.manages_own_memory());
        let used = runtime.pool_stats().expect("pool").used_bytes;
        assert!(used >= blas.workspace_bytes() as u64);
        eprintln!(
            "rocSOLVER getrf n={n}: workspace {} bytes from the pool",
            blas.workspace_bytes()
        );
    });
}

#[test]
fn a_call_that_needs_more_workspace_gets_a_larger_pool_buffer() {
    // rocSOLVER's getrf queries 0 bytes up to n = 64 and 790,784 bytes at n = 256 on
    // gfx1151 (rocSOLVER from ROCm 7.2.4), so this call cannot run inside the handle's
    // initial 4 KiB: in user-owned mode rocSOLVER would report a memory error instead of
    // allocating. The seam must size the call first and hand over a larger pool buffer.
    within_watchdog("getrf that outgrows the initial workspace", || {
        let Some(mut runtime) = runtime() else {
            return;
        };
        let mut blas = Rocblas::new(&mut runtime).expect("handle");
        assert_eq!(blas.workspace_bytes(), Rocblas::INITIAL_WORKSPACE);
        let n = 256;
        let matrix = values(DType::F64, n * n, 24);
        let (lu, pivots, status) = factor(&mut runtime, &mut blas, DType::F64, n, &matrix);
        assert_eq!(status, 0);
        assert_lu(&matrix, n, &lu, &pivots);
        let workspace = blas.workspace_bytes();
        assert!(
            workspace > Rocblas::INITIAL_WORKSPACE,
            "the workspace must have grown, it is {workspace} bytes"
        );
        assert!(
            !blas.manages_own_memory(),
            "growing must not hand memory management back to rocBLAS"
        );
        let used = runtime.pool_stats().expect("pool").used_bytes;
        assert!(
            used >= workspace as u64,
            "the pool must account for the grown workspace: {used} < {workspace}"
        );
        eprintln!("rocSOLVER getrf n={n}: workspace grew to {workspace} bytes from the pool");
    });
}

#[test]
fn a_host_info_pointer_is_refused_before_rocsolver_runs() {
    within_watchdog("getrf_raw with a host info pointer", || {
        let Some(mut runtime) = runtime() else {
            return;
        };
        let mut blas = Rocblas::new(&mut runtime).expect("handle");
        let n = 4usize;
        let a = runtime
            .upload(&encode(DType::F64, &values(DType::F64, n * n, 23)))
            .expect("a");
        let ipiv = runtime.alloc_zeroed(n * 4).expect("ipiv");
        // Exactly the reproducer's mistake: `&mut info` on the host stack, and then the
        // same through the heap.
        let mut stack_info = 0i32;
        let mut heap_info = Box::new(0i32);
        for (label, host_info) in [
            ("stack", &mut stack_info as *mut i32),
            ("heap", &mut *heap_info as *mut i32),
        ] {
            // SAFETY: `a` and `ipiv` are registered runtime buffers of the stated sizes;
            // `host_info` is exactly what the guard has to refuse before rocSOLVER runs.
            let outcome = unsafe {
                runtime
                    .with_raw(&[&a, &ipiv], |raw| {
                        let a_ptr = raw.device_ptr(&a).expect("a");
                        let ipiv_ptr = raw.device_ptr(&ipiv).expect("ipiv").cast::<i32>();
                        blas.getrf_raw(
                            raw,
                            DType::F64,
                            n as i64,
                            a_ptr,
                            n as i64,
                            ipiv_ptr,
                            host_info,
                        )
                    })
                    .expect("scope")
            };
            match outcome {
                Err(SeamError::PointerClass {
                    op: "getrf",
                    argument: "info",
                    found: PointerClass::Unregistered,
                    ..
                }) => {}
                other => panic!("{label} info: expected a pointer-class refusal, got {other:?}"),
            }
        }
        assert_eq!(stack_info, 0, "nothing may have written the host info");
        assert_eq!(*heap_info, 0, "nothing may have written the host info");
        // The handle and the stream are still healthy after the refusals.
        runtime.synchronize().expect("the stream is healthy");
    });
}

#[test]
fn the_pointer_class_guard_tells_device_memory_from_host_memory() {
    let Some(mut runtime) = runtime() else {
        return;
    };
    let device = runtime.alloc_zeroed(64).expect("device");
    let host = [0u8; 64];
    let heap = Box::new([0u8; 64]);
    // SAFETY: the closure only classifies addresses; nothing is queued.
    unsafe {
        runtime
            .with_raw(&[&device], |raw| {
                let pointer = raw.device_ptr(&device).expect("device");
                assert!(require_device(raw, "probe", "device", pointer.cast_const()).is_ok());
                // An interior address of a pool allocation is device memory too.
                let interior = pointer.cast::<u8>().wrapping_add(17).cast_const().cast();
                assert!(require_device(raw, "probe", "interior", interior).is_ok());
                for (label, address) in [
                    ("stack", host.as_ptr().cast()),
                    ("heap", heap.as_ptr().cast()),
                ] {
                    match require_device(raw, "probe", "info", address) {
                        Err(SeamError::PointerClass { found, .. }) => {
                            assert_eq!(found, PointerClass::Unregistered, "{label}")
                        }
                        other => panic!("{label}: expected a refusal, got {other:?}"),
                    }
                }
            })
            .expect("scope");
    }
}
