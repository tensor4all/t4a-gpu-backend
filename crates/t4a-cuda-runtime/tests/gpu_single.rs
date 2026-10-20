//! Public-contract tests on one real physical GPU.
//!
//! These run only with `--features gpu-tests`. A missing or unusable device fails
//! the test instead of being reported as a pass.

#![cfg(feature = "gpu-tests")]

use t4a_cuda_runtime::{
    Arg, Buffer, CompileRequest, Config, CopyRoute, LaunchConfig, Runtime, UninitBuffer,
};

const SCALE2: &str = r#"
extern "C" __global__ void scale2(const unsigned char* input, unsigned char* output, unsigned long long n) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = (unsigned char)(input[i] * 2u);
    }
}
"#;

fn open(ordinal: usize) -> Runtime {
    Runtime::open(ordinal, Config::default()).expect("a usable CUDA device is required")
}

fn bind_scale2(runtime: &mut Runtime) -> t4a_cuda_runtime::BoundFunction {
    let request = CompileRequest::new("scale2", SCALE2);
    let artifact = runtime.compile(&request).expect("kernel must compile");
    runtime
        .bind(&artifact, "scale2")
        .expect("entry point must bind")
}

fn launch_scale2(
    runtime: &mut Runtime,
    function: &t4a_cuda_runtime::BoundFunction,
    input: &Buffer,
) -> Buffer {
    let out: UninitBuffer = runtime.alloc_uninit(input.len(), 256).expect("allocation");
    let threads = 128u32;
    let blocks = (input.len() as u32).div_ceil(threads);
    // SAFETY: `scale2` reads `input.len()` bytes and fully overwrites the same
    // number of bytes in `out`; dimensions cover the whole domain.
    unsafe {
        runtime
            .launch(
                function,
                LaunchConfig::new(blocks, threads),
                &[
                    Arg::DevicePtr(input),
                    Arg::DevicePtrUninit(&out),
                    Arg::U64(input.len() as u64),
                ],
            )
            .expect("launch must be admitted");
    }
    // SAFETY: the launch fully overwrites every byte of `out`.
    unsafe { out.assume_init() }
}

#[test]
fn two_runtimes_on_one_ordinal_are_independent() {
    let first = open(0);
    let second = open(0);
    assert_ne!(first.domain_id().unwrap(), second.domain_id().unwrap());
    assert_ne!(first.stream_id().unwrap(), second.stream_id().unwrap());
    let mut first = first;
    let mut second = second;
    let left = first.upload(b"left").unwrap();
    // A buffer from another domain is refused before any work is admitted.
    assert!(matches!(
        second.download(&left),
        Err(t4a_cuda_runtime::Error::ForeignDomain)
    ));
}

#[test]
fn kernel_copy_chain_matches_the_cpu_reference() {
    let mut source_runtime = open(0);
    let mut destination = open(0);
    let payload: Vec<u8> = (0..10_000u32).map(|value| (value % 251) as u8).collect();
    // A and B both apply `scale2`, so the destination result is payload * 4.
    let expected: Vec<u8> = payload.iter().map(|value| value.wrapping_mul(4)).collect();

    let function = bind_scale2(&mut source_runtime);
    let input = source_runtime.upload(&payload).unwrap();
    let scaled = launch_scale2(&mut source_runtime, &function, &input);

    let (copy, completion, route) = destination
        .copy_between(&mut source_runtime, &scaled)
        .expect("copy must be admitted");
    assert_eq!(route, CopyRoute::SameDevice);
    // The completion handle is an observation handle; dropping it must not remove
    // the ordering the destination depends on.
    drop(completion);

    // The source may be dropped immediately after a successful admission.
    drop(scaled);
    drop(input);

    // No caller wait: the destination kernel and readback rely on stream order
    // alone, which is what proves the copy dependency is real.
    let destination_function = bind_scale2(&mut destination);
    let consumed = launch_scale2(&mut destination, &destination_function, &copy);
    let mut pending = destination.download(&consumed).unwrap();
    assert_eq!(pending.wait().unwrap(), expected);
}

#[test]
fn warm_module_is_reused_and_eviction_is_safe() {
    let mut runtime = open(0);
    let function = bind_scale2(&mut runtime);
    let input = runtime.upload(&[1u8, 2, 3, 4]).unwrap();
    for _ in 0..3 {
        let out = launch_scale2(&mut runtime, &function, &input);
        let mut pending = runtime.download(&out).unwrap();
        assert_eq!(pending.wait().unwrap(), vec![2u8, 4, 6, 8]);
    }
    let loads_after_warmup = runtime.module_loads();
    assert_eq!(loads_after_warmup, 1, "a warm module must not reload");

    // Clearing lookups must not unload a function that is still bound.
    runtime.clear_modules();
    let out = launch_scale2(&mut runtime, &function, &input);
    let mut pending = runtime.download(&out).unwrap();
    assert_eq!(pending.wait().unwrap(), vec![2u8, 4, 6, 8]);
    assert_eq!(runtime.stats().module_entries, 0);
}

#[test]
fn bounded_staging_moves_more_than_one_chunk() {
    let config = Config {
        max_chunk_bytes: 4096,
        max_idle_pinned_bytes: 8192,
        pending_host_budget: 16_384,
        ..Config::default()
    };
    let mut runtime = Runtime::open(0, config).unwrap();
    let payload: Vec<u8> = (0..70_000u32).map(|value| (value % 253) as u8).collect();
    let buffer = runtime.upload(&payload).unwrap();
    let mut pending = runtime.download(&buffer).unwrap();
    assert_eq!(pending.wait().unwrap(), payload);
    assert!(runtime.stats().retirement.chunk_allocs >= 2);
}

#[test]
fn downloads_survive_dropped_and_forgotten_handles() {
    let mut runtime = open(0);
    let payload: Vec<u8> = (0..16_384u32).map(|value| (value % 249) as u8).collect();
    let buffer = runtime.upload(&payload).unwrap();
    drop(runtime.download(&buffer).unwrap());
    std::mem::forget(runtime.download(&buffer).unwrap());
    let mut again = runtime.download(&buffer).unwrap();
    assert_eq!(again.wait().unwrap(), payload);
}

#[test]
fn teardown_with_outstanding_work_is_safe() {
    let mut runtime = open(0);
    let payload: Vec<u8> = (0..8192u32).map(|value| value as u8).collect();
    let buffer = runtime.upload(&payload).unwrap();
    let pending = runtime.download(&buffer).unwrap();
    drop(pending);
    runtime.close().expect("close must drain queued work");
    // The buffer outlives the closed runtime and still releases safely.
    drop(buffer);
}
