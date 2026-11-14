//! Real-device tests.
//!
//! These fail when no Metal device is usable: an invoked hardware test on a host
//! without a device must not pass silently.

use t4a_metal_runtime::{
    source_hash, Arg, Buffer, CompileRequest, Config, Error, LaunchConfig, Runtime, UninitBuffer,
};

const ADD: &str = r#"#include <metal_stdlib>
using namespace metal;

kernel void add(device float* a [[buffer(0)]],
                device const float* b [[buffer(1)]],
                uint index [[thread_position_in_grid]]) {
    a[index] = a[index] + b[index];
}

kernel void scale(device float* a [[buffer(0)]],
                  constant float& factor [[buffer(1)]],
                  uint index [[thread_position_in_grid]]) {
    a[index] = a[index] * factor;
}

kernel void fill(device float* a [[buffer(0)]],
                 constant float& value [[buffer(1)]],
                 uint index [[thread_position_in_grid]]) {
    a[index] = value;
}

kernel void mark(device uint* marks [[buffer(0)]],
                 uint index [[thread_position_in_grid]]) {
    marks[index] = marks[index] + 1;
}

kernel void spin(device float* out [[buffer(0)]],
                 constant uint& iterations [[buffer(1)]],
                 uint index [[thread_position_in_grid]]) {
    float acc = 0.0f;
    for (uint i = 0; i < iterations; ++i) {
        acc = fma(acc, 1.0000001f, 1.0f);
    }
    out[index] = acc;
}
"#;

fn runtime() -> Runtime {
    Runtime::open(Config::default())
        .unwrap_or_else(|error| panic!("this suite needs a usable Metal device: {error}"))
}

fn f32_bytes(elements: usize) -> usize {
    elements * std::mem::size_of::<f32>()
}

fn write(runtime: &mut Runtime, values: &[f32]) -> Buffer {
    let buffer = runtime
        .alloc_zeroed(f32_bytes(values.len()))
        .expect("allocation succeeds");
    // SAFETY: the buffer was just allocated, is exactly `values.len()` floats long, and
    // no submission references it yet.
    unsafe {
        std::slice::from_raw_parts_mut(buffer.host_ptr().cast::<f32>(), values.len())
            .copy_from_slice(values);
    }
    buffer
}

fn read(buffer: &Buffer) -> Vec<f32> {
    let len = buffer.len() / std::mem::size_of::<f32>();
    // SAFETY: callers synchronise the runtime before reading, and no submission can
    // still write the bytes.
    unsafe { std::slice::from_raw_parts(buffer.host_ptr().cast::<f32>(), len) }.to_vec()
}

fn read_u32(buffer: &Buffer) -> Vec<u32> {
    let len = buffer.len() / std::mem::size_of::<u32>();
    // SAFETY: as `read`, after the runtime was synchronised.
    unsafe { std::slice::from_raw_parts(buffer.host_ptr().cast::<u32>(), len) }.to_vec()
}

fn add_kernel(runtime: &mut Runtime) -> t4a_metal_runtime::BoundFunction {
    let artifact = runtime
        .compile(&CompileRequest::new("add", ADD))
        .expect("the add kernel compiles");
    runtime.bind(&artifact, "add").expect("add binds")
}

#[test]
fn the_probe_reports_the_measured_capabilities() {
    let capabilities = t4a_metal_runtime::probe().expect("this host has a Metal device");
    assert!(capabilities.fp16, "fp16 compiles");
    assert!(capabilities.bf16, "bf16 compiles");
    assert!(capabilities.c32, "a float2 kernel compiles");
    assert!(
        !capabilities.f64_supported,
        "no Apple GPU compiles a double kernel"
    );
    let diagnostic = capabilities
        .f64_diagnostic
        .expect("the rejected dtype keeps Metal's diagnostic");
    assert!(
        diagnostic.contains("double"),
        "the diagnostic names the type: {diagnostic}"
    );
    assert!(
        capabilities.device.unified_memory,
        "Apple GPUs share memory"
    );
    assert!(!capabilities.device.name.is_empty());
    assert!(capabilities.device.max_buffer_length > 0);
    assert!(capabilities.mps_framework, "MetalPerformanceShaders opens");
    assert!(
        capabilities.mps_matrix_multiplication,
        "MPSMatrixMultiplication is visible once its framework is loaded"
    );
    assert!(
        capabilities.mpsgraph_framework,
        "MetalPerformanceShadersGraph opens"
    );
    assert!(
        capabilities.mpsgraph_class,
        "MPSGraph is visible once its framework is loaded (tenferro-rs#2402 re-check)"
    );
}

#[test]
fn a_dispatch_round_trips_exactly() {
    let mut runtime = runtime();
    let bytes = f32_bytes(4);
    let a = runtime.alloc_zeroed(bytes).expect("allocation");
    let b = runtime.alloc_zeroed(bytes).expect("allocation");
    // SAFETY: both buffers are fresh, `bytes` long and unreferenced.
    unsafe {
        std::slice::from_raw_parts_mut(a.host_ptr().cast::<f32>(), 4)
            .copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        std::slice::from_raw_parts_mut(b.host_ptr().cast::<f32>(), 4).copy_from_slice(&[10.0; 4]);
    }
    let add = add_kernel(&mut runtime);
    // SAFETY: the kernel reads and writes four f32 values at index 0 and reads four at
    // index 1; the dispatch covers exactly those elements.
    unsafe {
        runtime
            .launch(
                &add,
                LaunchConfig::threads(4),
                &[
                    Arg::Buffer {
                        buffer: &a,
                        offset: 0,
                    },
                    Arg::Buffer {
                        buffer: &b,
                        offset: 0,
                    },
                ],
            )
            .expect("launch")
    };
    runtime
        .commit()
        .expect("commit")
        .wait()
        .expect("completion");
    assert_eq!(read(&a), [11.0, 12.0, 13.0, 14.0]);
}

#[test]
fn a_binding_offset_selects_the_tail() {
    // The binding-alignment constant is a measurement of what Metal accepts, so the
    // smallest offset it permits is exercised rather than assumed: binding `a` at byte 4
    // puts element 0 of the kernel at byte 4 of the buffer, so the four sums land in the
    // middle of the output and the first element stays untouched.
    let mut runtime = runtime();
    let a = write(&mut runtime, &[0.0; 8]);
    let b = write(&mut runtime, &[10.0, 20.0, 30.0, 40.0]);
    let add = add_kernel(&mut runtime);
    // SAFETY: the kernel reads and writes four f32 from index 0 starting at byte 4, which
    // is inside the 32-byte `a`, and reads exactly four from index 1, which is `b`.
    unsafe {
        runtime
            .launch(
                &add,
                LaunchConfig::threads(4),
                &[
                    Arg::Buffer {
                        buffer: &a,
                        offset: 4,
                    },
                    Arg::Buffer {
                        buffer: &b,
                        offset: 0,
                    },
                ],
            )
            .expect("an offset of 4 is accepted")
    };
    runtime
        .commit()
        .expect("commit")
        .wait()
        .expect("completion");
    assert_eq!(
        read(&a),
        [0.0, 10.0, 20.0, 30.0, 40.0, 0.0, 0.0, 0.0],
        "the dispatch starts at byte 4, leaving element 0 alone"
    );
}

#[test]
fn the_grid_covers_exactly_the_requested_threads() {
    // The marks buffer is deliberately wider than the dispatch, and it starts zeroed, so
    // "no thread ran here" is observable *inside* the allocation. An over-large grid then
    // fails here instead of writing out of bounds where nothing reads the damage.
    let mut runtime = runtime();
    let width = 16usize;
    let marks = runtime
        .alloc_zeroed(width * std::mem::size_of::<u32>())
        .expect("allocation");
    let artifact = runtime
        .compile(&CompileRequest::new("mark", ADD))
        .expect("the mark kernel compiles");
    let mark = runtime.bind(&artifact, "mark").expect("mark binds");
    // SAFETY: `mark` writes exactly one u32 at the index it is given; the grid below
    // runs four threads, and the buffer is sixteen elements long.
    unsafe {
        runtime
            .launch(
                &mark,
                LaunchConfig::threads(4),
                &[Arg::Buffer {
                    buffer: &marks,
                    offset: 0,
                }],
            )
            .expect("launch")
    };
    runtime
        .commit()
        .expect("commit")
        .wait()
        .expect("completion");
    let counts = read_u32(&marks);
    assert_eq!(
        &counts[..4],
        &[1, 1, 1, 1],
        "each of the four requested threads ran exactly once"
    );
    assert!(
        counts[4..].iter().all(|count| *count == 0),
        "no thread beyond the fourth ran: {counts:?}"
    );
}

#[test]
fn two_dispatches_in_one_command_buffer_are_ordered() {
    let mut runtime = runtime();
    let a = write(&mut runtime, &[1.0, 2.0, 3.0, 4.0]);
    let b = write(&mut runtime, &[10.0; 4]);
    let add = add_kernel(&mut runtime);
    let config = LaunchConfig::threads(4);
    // SAFETY: as `a_dispatch_round_trips_exactly`; both dispatches use the same
    // argument shapes.
    unsafe {
        for _ in 0..2 {
            runtime
                .launch(
                    &add,
                    config,
                    &[
                        Arg::Buffer {
                            buffer: &a,
                            offset: 0,
                        },
                        Arg::Buffer {
                            buffer: &b,
                            offset: 0,
                        },
                    ],
                )
                .expect("launch");
        }
    }
    let stats_before = runtime.stats();
    runtime
        .commit()
        .expect("commit")
        .wait()
        .expect("completion");
    assert_eq!(
        runtime.stats().submissions,
        stats_before.submissions + 1,
        "both dispatches share one command buffer"
    );
    assert_eq!(
        read(&a),
        [21.0, 22.0, 23.0, 24.0],
        "the second dispatch read the first dispatch's output"
    );
}

#[test]
fn separate_submissions_are_waited_by_their_own_completion() {
    let mut runtime = runtime();
    let a = write(&mut runtime, &[1.0, 2.0]);
    let b = write(&mut runtime, &[10.0, 20.0]);
    let add = add_kernel(&mut runtime);
    let config = LaunchConfig::threads(2);
    let mut completions = Vec::new();
    for _ in 0..3 {
        // SAFETY: two-element buffers, two-element dispatch.
        unsafe {
            runtime
                .launch(
                    &add,
                    config,
                    &[
                        Arg::Buffer {
                            buffer: &a,
                            offset: 0,
                        },
                        Arg::Buffer {
                            buffer: &b,
                            offset: 0,
                        },
                    ],
                )
                .expect("launch");
        }
        completions.push(runtime.commit().expect("commit"));
    }
    for completion in &completions {
        completion.wait().expect("completion");
    }
    assert_eq!(read(&a), [31.0, 62.0]);
}

#[test]
fn a_by_value_parameter_reaches_the_kernel() {
    let mut runtime = runtime();
    let a = write(&mut runtime, &[1.0, 2.0, 3.0]);
    let artifact = runtime
        .compile(&CompileRequest::new("scale", ADD))
        .expect("compile");
    let scale = runtime.bind(&artifact, "scale").expect("bind");
    let factor = 2.5f32;
    // SAFETY: the kernel scales three f32 values; `factor` is one `float` passed by
    // value at index 1, which is what the MSL declares.
    unsafe {
        runtime
            .launch(
                &scale,
                LaunchConfig::threads(3),
                &[
                    Arg::Buffer {
                        buffer: &a,
                        offset: 0,
                    },
                    Arg::pod(&factor),
                ],
            )
            .expect("launch");
    }
    runtime
        .commit()
        .expect("commit")
        .wait()
        .expect("completion");
    assert_eq!(read(&a), [2.5, 5.0, 7.5]);
}

#[test]
fn an_uninitialised_buffer_becomes_initialised_after_a_full_overwrite() {
    let mut runtime = runtime();
    let uninit: UninitBuffer = runtime.alloc_uninit(f32_bytes(4)).expect("allocation");
    let artifact = runtime
        .compile(&CompileRequest::new("fill", ADD))
        .expect("compile");
    let fill = runtime.bind(&artifact, "fill").expect("bind");
    let value = 7.0f32;
    // SAFETY: the kernel writes all four elements of index 0 and reads the factor at
    // index 1, so the whole buffer is overwritten.
    unsafe {
        runtime
            .launch(
                &fill,
                LaunchConfig::threads(4),
                &[
                    Arg::BufferUninit {
                        buffer: &uninit,
                        offset: 0,
                    },
                    Arg::pod(&value),
                ],
            )
            .expect("launch");
    }
    runtime
        .commit()
        .expect("commit")
        .wait()
        .expect("completion");
    // SAFETY: every byte was written by a completed submission.
    let filled = unsafe { uninit.assume_init() };
    assert_eq!(read(&filled), [7.0; 4]);
}

#[test]
fn the_artifact_cache_repays_a_repeat_compile() {
    let mut runtime = runtime();
    let request = CompileRequest::new("add", ADD);
    let first = runtime.compile(&request).expect("compile");
    let second = runtime.compile(&request).expect("compile");
    assert_eq!(first.source_hash(), second.source_hash());
    let stats = runtime.stats();
    assert_eq!(stats.artifact_entries, 1);
    assert_eq!(stats.artifact_misses, 1);
    assert_eq!(stats.artifact_hits, 1);
}

#[test]
fn a_rejected_dtype_is_a_typed_compile_error() {
    let mut runtime = runtime();
    let f64_source = t4a_metal_runtime::probe_sources()
        .iter()
        .find(|source| source.label == "f64")
        .expect("the probe carries the f64 source");
    match runtime.compile(&CompileRequest::new("f64", f64_source.source)) {
        Err(Error::Compile { detail }) => {
            assert!(detail.contains("double"), "Metal names the type: {detail}");
        }
        Err(other) => panic!("unexpected error: {other}"),
        Ok(_) => panic!("an f64 kernel must not compile"),
    }
}

#[test]
fn an_unknown_entry_point_is_typed() {
    let mut runtime = runtime();
    let artifact = runtime
        .compile(&CompileRequest::new("add", ADD))
        .expect("compile");
    match runtime.bind(&artifact, "not_a_kernel") {
        Err(Error::UnknownEntryPoint { name }) => assert_eq!(name, "not_a_kernel"),
        Err(other) => panic!("unexpected error: {other}"),
        Ok(_) => panic!("an absent entry point must not bind"),
    }
}

#[test]
fn a_threadgroup_above_the_pipeline_limit_is_rejected() {
    let mut runtime = runtime();
    let a = write(&mut runtime, &[0.0; 4]);
    let b = write(&mut runtime, &[0.0; 4]);
    let add = add_kernel(&mut runtime);
    let limit = add.max_total_threads_per_threadgroup();
    // SAFETY: the launch is rejected before anything is encoded.
    let error = unsafe {
        runtime.launch(
            &add,
            LaunchConfig::tiled(1, limit as u32 + 1),
            &[
                Arg::Buffer {
                    buffer: &a,
                    offset: 0,
                },
                Arg::Buffer {
                    buffer: &b,
                    offset: 0,
                },
            ],
        )
    }
    .expect_err("a threadgroup above the pipeline limit must be refused");
    match error {
        Error::InvalidArgument { detail } => {
            assert!(detail.contains(&limit.to_string()), "{detail}");
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn malformed_arguments_are_rejected_with_specific_kinds() {
    let mut runtime = runtime();
    let a = write(&mut runtime, &[0.0; 4]);
    let b = write(&mut runtime, &[0.0; 4]);
    let empty = runtime
        .alloc_zeroed(0)
        .expect("a zero-length buffer exists");
    let add = add_kernel(&mut runtime);
    let config = LaunchConfig::threads(4);

    // SAFETY: every call below is rejected before Metal sees it.
    unsafe {
        assert!(matches!(
            runtime.launch(
                &add,
                config,
                &[
                    Arg::Buffer {
                        buffer: &a,
                        offset: 2
                    },
                    Arg::Buffer {
                        buffer: &b,
                        offset: 0
                    },
                ],
            ),
            Err(Error::InvalidArgument { .. })
        ));
        assert!(matches!(
            runtime.launch(
                &add,
                config,
                &[
                    Arg::Buffer {
                        buffer: &a,
                        offset: a.len(),
                    },
                    Arg::Buffer {
                        buffer: &b,
                        offset: 0
                    },
                ],
            ),
            Err(Error::OutOfRange { .. })
        ));
        assert!(matches!(
            runtime.launch(
                &add,
                config,
                &[
                    Arg::Buffer {
                        buffer: &empty,
                        offset: 0,
                    },
                    Arg::Buffer {
                        buffer: &b,
                        offset: 0
                    },
                ],
            ),
            Err(Error::InvalidArgument { .. })
        ));
        let oversized = vec![0u8; t4a_metal_runtime::MAX_PARAMETER_BYTES + 1];
        assert!(matches!(
            runtime.launch(
                &add,
                config,
                &[
                    Arg::Buffer {
                        buffer: &a,
                        offset: 0
                    },
                    Arg::Bytes { bytes: &oversized },
                ],
            ),
            Err(Error::KernelParameterLimit { .. })
        ));
        assert!(matches!(
            runtime.launch(
                &add,
                config,
                &[
                    Arg::Buffer {
                        buffer: &a,
                        offset: 0
                    },
                    Arg::Bytes { bytes: &[] },
                ],
            ),
            Err(Error::InvalidArgument { .. })
        ));
        assert!(matches!(
            runtime.launch(&add, LaunchConfig::tiled(0, 4), &[]),
            Err(Error::InvalidArgument { .. })
        ));
    }
}

#[test]
fn a_buffer_from_another_runtime_is_foreign() {
    let mut first = runtime();
    let mut second = runtime();
    assert_ne!(first.domain_id(), second.domain_id());
    let local = first.alloc_zeroed(f32_bytes(4)).expect("allocation");
    let foreign = second.alloc_zeroed(f32_bytes(4)).expect("allocation");
    let add = add_kernel(&mut first);
    // SAFETY: the argument is rejected before anything is encoded.
    let error = unsafe {
        first.launch(
            &add,
            LaunchConfig::threads(4),
            &[
                Arg::Buffer {
                    buffer: &local,
                    offset: 0,
                },
                Arg::Buffer {
                    buffer: &foreign,
                    offset: 0,
                },
            ],
        )
    }
    .expect_err("a buffer from another runtime must be refused");
    assert!(matches!(error, Error::ForeignDomain));
}

#[test]
fn dropping_buffers_returns_the_live_counters() {
    let mut runtime = runtime();
    assert_eq!(runtime.stats().live_allocations, 0);
    for _ in 0..1000 {
        let buffer = runtime.alloc_zeroed(64).expect("allocation");
        drop(buffer);
    }
    let stats = runtime.stats();
    assert_eq!(
        stats.live_allocations, 0,
        "no allocation outlives its handle"
    );
    assert_eq!(stats.live_bytes, 0);
    assert_eq!(stats.allocations, 1000);
}

#[test]
fn a_wait_that_exceeds_the_completion_timeout_is_reported_and_drop_abandons_it() {
    // One second of GPU work observed with a one-millisecond bound: the wait must report
    // `Timeout`, and dropping the runtime must abandon the submission rather than wait out
    // the kernel, because a drop cannot report the timeout.
    let mut runtime = Runtime::open(Config {
        completion_timeout_ms: 1,
        ..Config::default()
    })
    .unwrap_or_else(|error| panic!("this suite needs a usable Metal device: {error}"));
    let out = runtime.alloc_zeroed(f32_bytes(1)).expect("allocation");
    let artifact = runtime
        .compile(&CompileRequest::new("spin", ADD))
        .expect("the spin kernel compiles");
    let spin = runtime.bind(&artifact, "spin").expect("spin binds");
    // 300M dependent fma iterations: about 3.9 s on this host, measured at 1.3 s per 100M.
    let iterations = 300_000_000u32;
    // SAFETY: one thread writes one f32 into a four-byte buffer and reads one by-value
    // parameter.
    unsafe {
        runtime
            .launch(
                &spin,
                LaunchConfig::threads(1),
                &[
                    Arg::Buffer {
                        buffer: &out,
                        offset: 0,
                    },
                    Arg::pod(&iterations),
                ],
            )
            .expect("launch")
    };
    let completion = runtime.commit().expect("commit");
    match completion.wait() {
        Err(Error::Timeout { millis }) => assert_eq!(millis, 1, "the configured bound is reported"),
        other => panic!("expected a timeout from a one-millisecond bound, got {other:?}"),
    }
    let start = std::time::Instant::now();
    drop(runtime);
    let elapsed = start.elapsed();
    assert!(
        elapsed < std::time::Duration::from_secs(1),
        "drop abandons a submission past the timeout instead of waiting out the kernel: \
         {elapsed:?}"
    );
}

#[test]
fn dropping_a_runtime_discards_work_that_was_never_committed() {
    let mut runtime = runtime();
    let a = write(&mut runtime, &[1.0, 2.0, 3.0, 4.0]);
    let b = write(&mut runtime, &[10.0; 4]);
    let add = add_kernel(&mut runtime);
    // SAFETY: the same shape as the round-trip test, and nothing is committed below.
    unsafe {
        runtime
            .launch(
                &add,
                LaunchConfig::threads(4),
                &[
                    Arg::Buffer {
                        buffer: &a,
                        offset: 0,
                    },
                    Arg::Buffer {
                        buffer: &b,
                        offset: 0,
                    },
                ],
            )
            .expect("launch")
    };
    drop(runtime);
    assert_eq!(
        read(&a),
        [1.0, 2.0, 3.0, 4.0],
        "a launch that was never committed does not run when the runtime is dropped"
    );
}

#[test]
fn dropping_a_runtime_completes_work_that_was_committed() {
    let mut runtime = runtime();
    let a = write(&mut runtime, &[1.0, 2.0, 3.0, 4.0]);
    let b = write(&mut runtime, &[10.0; 4]);
    let add = add_kernel(&mut runtime);
    // SAFETY: as the round-trip test.
    unsafe {
        runtime
            .launch(
                &add,
                LaunchConfig::threads(4),
                &[
                    Arg::Buffer {
                        buffer: &a,
                        offset: 0,
                    },
                    Arg::Buffer {
                        buffer: &b,
                        offset: 0,
                    },
                ],
            )
            .expect("launch")
    };
    runtime.commit().expect("commit");
    drop(runtime);
    assert_eq!(
        read(&a),
        [11.0, 12.0, 13.0, 14.0],
        "committed work is finished before the runtime goes away"
    );
}

#[test]
fn the_in_flight_bound_is_respected() {
    let mut runtime = Runtime::open(Config {
        max_submissions_in_flight: 2,
        ..Config::default()
    })
    .expect("a device is present");
    for _ in 0..5 {
        runtime.commit().expect("commit");
        assert!(
            runtime.stats().submissions_in_flight <= 2,
            "the bound holds: {}",
            runtime.stats().submissions_in_flight
        );
    }
    runtime.synchronize().expect("synchronise");
    assert_eq!(runtime.stats().submissions_in_flight, 0);
    assert_eq!(runtime.stats().submissions, 5);
}

#[test]
fn the_raw_scope_keeps_the_submission_consistent() {
    let mut runtime = runtime();
    let a = write(&mut runtime, &[1.0, 1.0]);
    let b = write(&mut runtime, &[1.0, 1.0]);
    let add = add_kernel(&mut runtime);
    let config = LaunchConfig::threads(2);
    let domain_id = runtime.domain_id();
    // SAFETY: the launches are the same shape as the round-trip test; the closure only
    // inspects the handles it is given and opens no encoder.
    unsafe {
        runtime
            .launch(
                &add,
                config,
                &[
                    Arg::Buffer {
                        buffer: &a,
                        offset: 0,
                    },
                    Arg::Buffer {
                        buffer: &b,
                        offset: 0,
                    },
                ],
            )
            .expect("launch");
        runtime
            .with_raw(Vec::new(), |scope| {
                assert!(!scope.command_buffer().is_null());
                assert!(!scope.device().is_null());
                assert_eq!(scope.domain_id(), domain_id);
                assert!(!scope
                    .buffer(&a)
                    .expect("the buffer is in this domain")
                    .is_null());
            })
            .expect("raw scope");
        runtime
            .launch(
                &add,
                config,
                &[
                    Arg::Buffer {
                        buffer: &a,
                        offset: 0,
                    },
                    Arg::Buffer {
                        buffer: &b,
                        offset: 0,
                    },
                ],
            )
            .expect("launch after the raw scope");
    }
    runtime
        .commit()
        .expect("commit")
        .wait()
        .expect("completion");
    assert_eq!(read(&a), [3.0, 3.0], "both dispatches reached the device");
}

#[test]
fn a_panicking_raw_scope_discards_its_session() {
    let mut runtime = runtime();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: the closure encodes nothing and asserts no ABI; it only panics.
        unsafe { runtime.with_raw(Vec::new(), |_scope| panic!("closure panics")) }
    }));
    assert!(result.is_err(), "the panic reaches the caller");
    assert_eq!(
        runtime.stats().submissions,
        0,
        "the discarded session committed nothing"
    );
    // The runtime is still usable, and the commit below submits an empty buffer rather
    // than whatever the panicking closure had left behind.
    runtime
        .commit()
        .expect("commit after the panic")
        .wait()
        .expect("completion");
    assert_eq!(runtime.stats().submissions, 1);
}

#[test]
fn a_closed_runtime_refuses_new_work() {
    let mut runtime = runtime();
    runtime.close().expect("close");
    assert!(matches!(runtime.alloc_zeroed(16), Err(Error::Closed)));
    assert!(matches!(runtime.commit(), Err(Error::Closed)));
    assert!(matches!(runtime.synchronize(), Err(Error::Closed)));
    runtime.close().expect("closing twice is not an error");
}

#[test]
fn stats_count_what_the_caller_did() {
    let mut runtime = runtime();
    let a = write(&mut runtime, &[0.0; 4]);
    let b = write(&mut runtime, &[0.0; 4]);
    let add = add_kernel(&mut runtime);
    // SAFETY: four-element buffers and a four-element dispatch.
    unsafe {
        runtime
            .launch(
                &add,
                LaunchConfig::threads(4),
                &[
                    Arg::Buffer {
                        buffer: &a,
                        offset: 0,
                    },
                    Arg::Buffer {
                        buffer: &b,
                        offset: 0,
                    },
                ],
            )
            .expect("launch");
    }
    runtime
        .commit()
        .expect("commit")
        .wait()
        .expect("completion");
    let stats = runtime.stats();
    assert_eq!(stats.launches, 1);
    assert_eq!(stats.submissions, 1);
    assert!(stats.live_allocations >= 2);
    assert_eq!(source_hash(ADD).len(), 64);
}
