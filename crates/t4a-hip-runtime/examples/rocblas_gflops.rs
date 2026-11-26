//! GEMM throughput through the `Rocblas` seam, per dtype and size.
//!
//! ```text
//! cargo run --release -p t4a-hip-runtime --example rocblas_gflops -- [out.csv] [sizes...]
//! ```
//!
//! Square column-major `n x n x n` products, `c = a * b`, timed on the host around a
//! runtime synchronisation: two warm-up calls, then repetitions until at least one
//! second has passed (at least three). The figure is recorded data for tenferro-rs#2489,
//! not a gate. On an integrated GPU the operands share LPDDR5X with the CPU, so these
//! numbers say nothing about discrete MI-series cards.

#[path = "../tests/support/mod.rs"]
mod support;

use std::io::Write;
use std::time::{Duration, Instant};

use support::rocblas_seam::{DType, GemmRequest, Rocblas};
use t4a_hip_runtime::sys::RocblasTable;
use t4a_hip_runtime::Runtime;

fn main() {
    let mut args = std::env::args().skip(1);
    let out = args.next();
    let sizes: Vec<usize> = {
        let given: Vec<usize> = args.map(|s| s.parse().expect("size")).collect();
        if given.is_empty() {
            vec![256, 1024, 2048, 4096, 8192]
        } else {
            given
        }
    };
    let mut runtime = Runtime::open(0).expect("HIP device 0");
    let mut blas = Rocblas::new(&mut runtime).expect("rocBLAS handle");
    let device = runtime.device().clone();
    let rocblas = RocblasTable::load().expect("rocBLAS");
    let backend = std::env::var("ROCBLAS_USE_HIPBLASLT").unwrap_or_else(|_| "unset".into());
    eprintln!(
        "# {} {} warp {} integrated {} hipRuntime {} rocBLAS {} ROCBLAS_USE_HIPBLASLT={backend}",
        device.name,
        device.gcn_arch_name,
        device.warp_size,
        device.integrated,
        device.runtime_version,
        rocblas.version()
    );
    let mut rows = vec![
        "dtype,n,reps,seconds_per_call,gflops,arch,hip_runtime,rocblas,rocblas_use_hipblaslt"
            .to_string(),
    ];
    for dtype in DType::ALL {
        for &n in &sizes {
            let elements = n * n;
            let a = runtime
                .upload(&support::encode(
                    dtype,
                    &support::values(dtype, elements, 1),
                ))
                .expect("a");
            let b = runtime
                .upload(&support::encode(
                    dtype,
                    &support::values(dtype, elements, 2),
                ))
                .expect("b");
            let c = runtime.alloc_zeroed(elements * dtype.size()).expect("c");
            let request = GemmRequest {
                dtype,
                a: &a,
                trans_a: false,
                b: &b,
                trans_b: false,
                c: &c,
                shape: (n as i64, n as i64, n as i64),
                leading: (n as i64, n as i64, n as i64),
                batches: (1, 0, 0, 0),
            };
            for _ in 0..2 {
                blas.gemm(&mut runtime, &request).expect("warm-up gemm");
            }
            runtime.synchronize().expect("sync");
            let start = Instant::now();
            let mut reps = 0u32;
            while reps < 3 || start.elapsed() < Duration::from_secs(1) {
                blas.gemm(&mut runtime, &request).expect("gemm");
                reps += 1;
                if reps % 4 == 0 || reps < 3 {
                    runtime.synchronize().expect("sync");
                }
            }
            runtime.synchronize().expect("sync");
            let per_call = start.elapsed().as_secs_f64() / f64::from(reps);
            let gflops = dtype.flops(n as f64, n as f64, n as f64) / per_call / 1e9;
            eprintln!("{dtype:?} n={n:5} reps={reps:4} {per_call:.6} s/call {gflops:9.1} GFLOP/s");
            rows.push(format!(
                "{dtype:?},{n},{reps},{per_call:.6e},{gflops:.1},{},{},{},{backend}",
                device.gcn_arch_name,
                device.runtime_version,
                rocblas.version()
            ));
        }
    }
    let csv = rows.join("\n") + "\n";
    match out {
        Some(path) => std::fs::File::create(&path)
            .and_then(|mut file| file.write_all(csv.as_bytes()))
            .expect("write the CSV"),
        None => print!("{csv}"),
    }
}
