//! Public-contract tests on one real physical GPU.
//!
//! These run only with `--features gpu-tests`. A missing or unusable device fails
//! the test instead of being reported as a pass.

#![cfg(feature = "gpu-tests")]

use t4a_cuda_runtime::{
    Arg, Buffer, CompileRequest, Config, CopyRoute, LaunchConfig, Pod, RawResource, Runtime,
    UninitBuffer,
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

/// The ABI's largest by-value parameter blocks: a 144-byte layout beside a
/// 4 KiB fused-elementwise block.
const BY_VALUE: &str = r#"
typedef struct {
    int rank;
    int reserved;
    long long offset;
    long long extents[8];
    long long strides[8];
} layout144;

typedef struct { unsigned char bytes[4096]; } block4k;

extern "C" __global__ void byvalue(const layout144 layout, const block4k block, unsigned long long* out) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out[0] = (unsigned long long)layout.rank;
        out[1] = (unsigned long long)layout.offset;
        out[2] = (unsigned long long)layout.extents[7];
        out[3] = (unsigned long long)layout.strides[3];
        out[4] = (unsigned long long)block.bytes[4095];
        out[5] = (unsigned long long)block.bytes[0];
    }
}
"#;

#[repr(C)]
#[derive(Clone, Copy)]
struct Layout144 {
    rank: i32,
    reserved: i32,
    offset: i64,
    extents: [i64; 8],
    strides: [i64; 8],
}

// SAFETY: repr(C) plain data: integers only, no interior pointer and no
// invariant that reading its padding could break.
unsafe impl Pod for Layout144 {}

#[repr(C)]
#[derive(Clone, Copy)]
struct Block4k {
    bytes: [u8; 4096],
}

// SAFETY: repr(C) plain data: one byte array, no interior pointer.
unsafe impl Pod for Block4k {}

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

#[test]
fn foreign_functions_and_arguments_are_rejected() {
    let mut a = open(0);
    let mut b = open(0);
    let function = bind_scale2(&mut a);
    let input_a = a.upload(&[1u8, 2, 3, 4]).unwrap();
    let input_b = b.upload(&[1u8, 2, 3, 4]).unwrap();

    // A function bound by another runtime is refused before any launch.
    let out = a.alloc_uninit(4, 256).unwrap();
    // SAFETY: rejected before the driver is called; the ABI is irrelevant.
    let outcome = unsafe {
        b.launch(
            &function,
            LaunchConfig::new(1, 4),
            &[
                Arg::DevicePtr(&input_a),
                Arg::DevicePtrUninit(&out),
                Arg::U64(4),
            ],
        )
    };
    assert!(matches!(
        outcome,
        Err(t4a_cuda_runtime::Error::ForeignDomain)
    ));

    // A buffer from another domain is refused even with a matching function.
    let function_b = bind_scale2(&mut b);
    let out_b = b.alloc_uninit(4, 256).unwrap();
    let _ = &input_b;
    // SAFETY: rejected before the driver is called.
    let outcome = unsafe {
        b.launch(
            &function_b,
            LaunchConfig::new(1, 4),
            &[
                Arg::DevicePtr(&input_a),
                Arg::DevicePtrUninit(&out_b),
                Arg::U64(4),
            ],
        )
    };
    assert!(matches!(
        outcome,
        Err(t4a_cuda_runtime::Error::ForeignDomain)
    ));
}

#[test]
fn unknown_entry_point_is_typed() {
    let mut runtime = open(0);
    let request = CompileRequest::new("scale2", SCALE2);
    let artifact = runtime.compile(&request).unwrap();
    assert!(matches!(
        runtime.bind(&artifact, "no_such_kernel"),
        Err(t4a_cuda_runtime::Error::UnknownEntryPoint { .. })
    ));
}

#[test]
fn synchronous_launch_failure_is_reported() {
    let mut runtime = open(0);
    let function = bind_scale2(&mut runtime);
    let input = runtime.upload(&[1u8, 2, 3, 4]).unwrap();
    let out = runtime.alloc_uninit(4, 256).unwrap();
    // 2048 threads per block exceeds the device limit; the driver rejects it.
    // SAFETY: the launch is rejected by the driver before any kernel runs.
    let outcome = unsafe {
        runtime.launch(
            &function,
            LaunchConfig::new(1, 4096),
            &[
                Arg::DevicePtr(&input),
                Arg::DevicePtrUninit(&out),
                Arg::U64(4),
            ],
        )
    };
    assert!(matches!(
        outcome,
        Err(t4a_cuda_runtime::Error::Driver { .. })
    ));
}

#[test]
fn warm_launches_do_not_record_or_wait_per_submission() {
    let mut runtime = open(0);
    let function = bind_scale2(&mut runtime);
    let input = runtime.upload(&[1u8, 2, 3, 4]).unwrap();
    // The upload itself seals one staging epoch; only the launches are measured.
    let before = runtime.stats().retirement.events_recorded;
    for _ in 0..20 {
        let out = launch_scale2(&mut runtime, &function, &input);
        std::mem::forget(out);
    }
    let after = runtime.stats().retirement.events_recorded;
    assert_eq!(
        after, before,
        "plain launches must not record a completion event"
    );
}

#[test]
fn cubin_artifact_binds_and_runs() {
    let mut runtime = open(0);
    let mut request = CompileRequest::new("scale2-cubin", SCALE2);
    request.output = t4a_cuda_runtime::OutputKind::Cubin;
    // Local-architecture CUBIN; the validated host is an A100 (sm_80).
    request.arch = Some("sm_80".to_string());
    let artifact = runtime.compile(&request).expect("cubin must compile");
    assert!(artifact.image().len() > 4);
    let function = runtime.bind(&artifact, "scale2").expect("cubin must bind");
    let input = runtime.upload(&[1u8, 2, 3, 4]).unwrap();
    let out = launch_scale2(&mut runtime, &function, &input);
    let mut pending = runtime.download(&out).unwrap();
    assert_eq!(pending.wait().unwrap(), vec![2u8, 4, 6, 8]);
}

#[test]
fn raw_scope_runs_a_real_driver_call() {
    use t4a_cuda_runtime::RawResource;
    let mut runtime = open(0);
    let buffer = runtime.upload(&[7u8; 32]).unwrap();
    // SAFETY: a memset over exactly the registered buffer, on this runtime's own
    // stream, followed by an explicit observation before the readback.
    unsafe {
        runtime
            .with_raw(vec![RawResource::Buffer(&buffer)], |raw| {
                let ptr = raw.device_ptr(&buffer).unwrap();
                cudarc::driver::sys::cuMemsetD8Async(
                    ptr,
                    0,
                    32,
                    raw.stream() as usize as *mut cudarc::driver::sys::CUstream_st,
                )
            })
            .expect("admission must succeed");
    }
    let mut pending = runtime.download(&buffer).unwrap();
    assert_eq!(pending.wait().unwrap(), vec![0u8; 32]);
}

#[test]
fn forced_staged_route_moves_bytes_through_the_chunk_ring() {
    let config = Config {
        max_chunk_bytes: 4096,
        max_idle_pinned_bytes: 8192,
        pending_host_budget: 16_384,
        force_staged_route: true,
        ..Config::default()
    };
    let mut source_runtime = Runtime::open(0, config.clone()).unwrap();
    let mut destination = Runtime::open(0, config).unwrap();
    let payload: Vec<u8> = (0..30_000u32).map(|value| (value % 251) as u8).collect();
    let source = source_runtime.upload(&payload).unwrap();
    let (copy, mut completion, route) = destination
        .copy_between(&mut source_runtime, &source)
        .expect("staged admission");
    assert_eq!(route, CopyRoute::Staged);
    drop(source);
    completion.wait().expect("staged copy must complete");
    let mut pending = destination.download(&copy).unwrap();
    assert_eq!(pending.wait().unwrap(), payload);
}

#[test]
fn dropping_a_runtime_buffer_inside_another_runtimes_raw_scope_keeps_its_context() {
    use t4a_cuda_runtime::RawResource;
    let mut a = open(0);
    let mut b = open(0);
    let doomed = a.upload(&[1u8; 16]).unwrap();
    let kept = b.upload(&[2u8; 16]).unwrap();

    // SAFETY: dropping a buffer only enqueues its stream-ordered free; no driver
    // call is made from this closure.
    unsafe {
        b.with_raw(vec![RawResource::Buffer(&kept)], |_raw| {
            drop(doomed);
        })
        .expect("raw admission must succeed");
    }
    let mut pending = b.download(&kept).unwrap();
    assert_eq!(pending.wait().unwrap(), vec![2u8; 16]);
}

#[test]
fn runtimes_expose_distinct_pool_identities() {
    let a = open(0);
    let b = open(0);
    assert_ne!(a.stats().pool_id, b.stats().pool_id);
    assert_ne!(a.stats().domain_id, b.stats().domain_id);
}

#[test]
fn module_cache_evicts_by_capacity_and_clears() {
    let config = Config {
        module_cache_entries: 1,
        ..Config::default()
    };
    let mut runtime = Runtime::open(0, config).unwrap();
    let first = {
        let request = CompileRequest::new("scale2-a", SCALE2);
        let artifact = runtime.compile(&request).unwrap();
        runtime.bind(&artifact, "scale2").unwrap()
    };
    // A second artifact must evict the first cached entry.
    let second_source = format!("{SCALE2}\nextern \"C\" __global__ void unused() {{}}\n");
    let second = {
        let request = CompileRequest::new("scale2-b", second_source);
        let artifact = runtime.compile(&request).unwrap();
        runtime.bind(&artifact, "scale2").unwrap()
    };
    assert_eq!(runtime.stats().module_entries, 1);
    assert_eq!(runtime.module_loads(), 2);
    runtime.clear_modules();
    assert_eq!(runtime.stats().module_entries, 0);
    // Bound functions keep their module alive and remain usable after eviction.
    let input = runtime.upload(&[1u8, 2]).unwrap();
    for function in [&first, &second] {
        let out = launch_scale2(&mut runtime, function, &input);
        let mut pending = runtime.download(&out).unwrap();
        assert_eq!(pending.wait().unwrap(), vec![2u8, 4]);
    }
}

#[test]
fn entry_points_of_one_artifact_share_a_single_module_load() {
    let source = format!(
        "{SCALE2}\nextern \"C\" __global__ void scale2_b(const unsigned char* i, unsigned char* o, unsigned long long n) {{ unsigned long long k = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; if (k < n) {{ o[k] = (unsigned char)(i[k] * 2u); }} }}\n"
    );
    let mut runtime = open(0);
    let artifact = runtime
        .compile(&CompileRequest::new("two-entries", source))
        .unwrap();
    let first = runtime.bind(&artifact, "scale2").unwrap();
    let second = runtime.bind(&artifact, "scale2_b").unwrap();
    assert_eq!(runtime.module_loads(), 1);
    let input = runtime.upload(&[3u8, 4]).unwrap();
    for function in [&first, &second] {
        let out = launch_scale2(&mut runtime, function, &input);
        let mut pending = runtime.download(&out).unwrap();
        assert_eq!(pending.wait().unwrap(), vec![6u8, 8]);
    }
}

#[test]
fn empty_buffers_are_rejected_as_kernel_arguments() {
    let mut runtime = open(0);
    let function = bind_scale2(&mut runtime);
    let empty = runtime.upload(&[]).unwrap();
    let out = runtime.alloc_uninit(4, 256).unwrap();
    // SAFETY: rejected before the driver is called.
    let outcome = unsafe {
        runtime.launch(
            &function,
            LaunchConfig::new(1, 1),
            &[
                Arg::DevicePtr(&empty),
                Arg::DevicePtrUninit(&out),
                Arg::U64(0),
            ],
        )
    };
    assert!(matches!(
        outcome,
        Err(t4a_cuda_runtime::Error::InvalidArgument { .. })
    ));
}

#[test]
fn counters_track_launches_allocations_zero_fills_and_events() {
    let mut runtime = open(0);
    let function = bind_scale2(&mut runtime);
    let before = runtime.stats().retirement;
    let input = runtime.upload(&[1u8, 2, 3, 4]).unwrap();
    let out = runtime.alloc_zeroed(4, 256).unwrap();
    let after_io = runtime.stats().retirement;
    assert!(
        after_io.allocations >= before.allocations + 2,
        "upload and alloc"
    );
    assert_eq!(after_io.zero_fills, before.zero_fills + 1);
    assert!(
        after_io.events_recorded > before.events_recorded,
        "upload chunk"
    );

    // SAFETY: scale2 reads `input.len()` bytes and overwrites the same span.
    unsafe {
        runtime
            .launch(
                &function,
                LaunchConfig::new(1, 4),
                &[Arg::DevicePtr(&input), Arg::DevicePtr(&out), Arg::U64(4)],
            )
            .unwrap();
    }
    let after_launch = runtime.stats().retirement;
    assert_eq!(after_launch.launches, before.launches + 1);
    let _ = out;
}

#[test]
fn source_overwrite_after_copy_admission_keeps_the_copied_bytes() {
    use t4a_cuda_runtime::RawResource;
    let config = Config::default();
    let mut source_runtime = Runtime::open(0, config.clone()).unwrap();
    let mut destination = Runtime::open(0, config).unwrap();
    let payload: Vec<u8> = (0..4096u32).map(|value| (value % 251) as u8).collect();
    let source = source_runtime.upload(&payload).unwrap();
    let (copy, _completion, _route) = destination
        .copy_between(&mut source_runtime, &source)
        .unwrap();
    // Overwrite the source immediately after admission; the copy must still hold
    // the bytes it was admitted against.
    // SAFETY: memset covers exactly the registered source buffer on its stream.
    unsafe {
        source_runtime
            .with_raw(vec![RawResource::Buffer(&source)], |raw| {
                let ptr = raw.device_ptr(&source).unwrap();
                cudarc::driver::sys::cuMemsetD8Async(
                    ptr,
                    0,
                    4096,
                    raw.stream() as usize as *mut cudarc::driver::sys::CUstream_st,
                )
            })
            .unwrap();
    }
    let mut pending = destination.download(&copy).unwrap();
    assert_eq!(pending.wait().unwrap(), payload);
}

#[test]
fn by_value_parameters_reach_the_kernel() {
    let mut runtime = open(0);
    // A 4 KiB block beside the 144-byte layout needs a Volta-or-later target,
    // so this compiles a local-architecture CUBIN; the validated host is an A100.
    let mut request = CompileRequest::new("byvalue", BY_VALUE);
    request.output = t4a_cuda_runtime::OutputKind::Cubin;
    request.arch = Some("sm_80".to_string());
    let artifact = runtime.compile(&request).expect("kernel must compile");
    let function = runtime.bind(&artifact, "byvalue").expect("bind");
    let limit = runtime.max_kernel_parameter_bytes().unwrap();
    assert!(
        limit >= 4096 + 144,
        "the device must fit the largest ABI parameter block, got {limit}"
    );

    let out: UninitBuffer = runtime.alloc_uninit(6 * 8, 256).expect("allocation");
    let layout = Layout144 {
        rank: 3,
        reserved: 0,
        offset: -5,
        extents: [0, 1, 2, 3, 4, 5, 6, 7],
        strides: [10, 11, 12, 13, 14, 15, 16, 17],
    };
    let mut block = Block4k { bytes: [0; 4096] };
    block.bytes[0] = 200;
    block.bytes[4095] = 111;
    let before = runtime.stats().retirement;
    // SAFETY: the kernel reads both blocks by value and writes the six words of
    // `out`; both blocks stay alive across the call.
    unsafe {
        runtime
            .launch(
                &function,
                LaunchConfig::new(1, 1),
                &[
                    Arg::pod(&layout),
                    Arg::pod(&block),
                    Arg::DevicePtrUninit(&out),
                ],
            )
            .expect("launch must be admitted");
    }
    let after = runtime.stats().retirement;
    assert_eq!(after.launches, before.launches + 1);
    assert_eq!(
        after.launch_parameter_bytes,
        before.launch_parameter_bytes + 144 + 4096 + 8
    );

    // SAFETY: the launch fully overwrote `out`.
    let out = unsafe { out.assume_init() };
    let mut pending = runtime.download(&out).expect("download");
    let bytes = pending.wait().expect("readback");
    let words: Vec<u64> = bytes
        .chunks_exact(8)
        .map(|word| u64::from_le_bytes(word.try_into().unwrap()))
        .collect();
    assert_eq!(
        words,
        vec![3, (-5i64) as u64, 7, 13, 111, 200],
        "the kernel must see the by-value layout and block unchanged"
    );
}

#[test]
fn raw_scope_launch_is_admitted_and_retains_external_resources() {
    let mut runtime = open(0);
    let function = bind_scale2(&mut runtime);
    let input = runtime.upload(&[1u8, 2, 3, 4]).expect("upload");
    let out: UninitBuffer = runtime.alloc_uninit(input.len(), 256).expect("allocation");
    let before = runtime.stats().retirement;
    let external = std::sync::Arc::new(());
    // SAFETY: scale2 reads `input.len()` bytes and fully overwrites the same span
    // of `out`; the closure enqueues only that launch on this runtime's stream.
    unsafe {
        runtime
            .with_raw(vec![RawResource::Buffer(&input)], |raw| {
                // A vendor-style consumer addresses its operands itself, so the
                // runtime can only learn about them from `retain`.
                let input = raw.device_ptr(&input)?;
                raw.retain(Box::new(std::sync::Arc::clone(&external)), 8)?;
                raw.launch(
                    &function,
                    LaunchConfig::new(1, 4),
                    &[Arg::U64(input), Arg::DevicePtrUninit(&out), Arg::U64(4)],
                )
            })
            .expect("the raw scope must be admitted")
            .expect("the raw launch must be admitted");
    }
    let after = runtime.stats().retirement;
    assert_eq!(after.launches, before.launches + 1);
    assert_eq!(
        after.launch_parameter_bytes,
        before.launch_parameter_bytes + 24
    );
    assert_eq!(
        std::sync::Arc::strong_count(&external),
        2,
        "the scope must keep the registered resource alive past closure exit"
    );
    assert!(
        after.pending_bytes >= before.pending_bytes + 8,
        "the retained bytes stay reserved until observation"
    );

    // SAFETY: the launch fully overwrote `out`.
    let out = unsafe { out.assume_init() };
    let mut pending = runtime.download(&out).expect("download");
    assert_eq!(pending.wait().expect("readback"), vec![2, 4, 6, 8]);

    runtime.synchronize().expect("synchronize");
    for _ in 0..16 {
        runtime.observe().expect("observe");
        if std::sync::Arc::strong_count(&external) == 1 {
            break;
        }
        std::thread::yield_now();
    }
    assert_eq!(
        std::sync::Arc::strong_count(&external),
        1,
        "observation must release the retained resource"
    );
}
