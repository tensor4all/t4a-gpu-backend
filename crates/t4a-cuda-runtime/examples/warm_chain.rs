//! Warm-chain counters and a small direct-cudarc comparison.
//!
//! Run on a quiet host with an explicit single thread:
//!
//! ```text
//! cargo run -j 16 --release -p t4a-cuda-runtime --example warm_chain
//! ```
//!
//! The reported latency numbers are indicative on a shared development host; they
//! are not a performance gate. Counters (submissions, launches, events, host
//! waits) are exact for the runtime-managed path.

use std::time::Instant;

use t4a_cuda_runtime::{Arg, CompileRequest, Config, LaunchConfig, Runtime, UninitBuffer};

const KERNEL: &str = r#"
extern "C" __global__ void add1(const unsigned int* input, unsigned int* output, unsigned long long n) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = input[i] + 1u;
}
"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let iterations = 200usize;
    let elements = 1024usize;
    let mut runtime = Runtime::open(0, Config::default())?;

    let request = CompileRequest::new("add1", KERNEL);
    let cold = Instant::now();
    let artifact = runtime.compile(&request)?;
    let cold_compile = cold.elapsed();
    let warm = runtime.compile(&request)?;
    assert_eq!(artifact.key(), warm.key());
    let function = runtime.bind(&artifact, "add1")?;

    let input: Vec<u32> = (0..elements as u32).collect();
    let bytes: Vec<u8> = input.iter().flat_map(|value| value.to_ne_bytes()).collect();
    let device_in = runtime.upload(&bytes)?;
    let device_out: UninitBuffer = runtime.alloc_uninit(bytes.len(), 256)?;
    let config = LaunchConfig::new(4, 256);

    // Warm up so the driver's lazy module paths are not measured.
    for _ in 0..10 {
        // SAFETY: add1 reads `elements` u32s and fully overwrites the same span.
        unsafe {
            runtime.launch(
                &function,
                config,
                &[
                    Arg::DevicePtr(&device_in),
                    Arg::DevicePtrUninit(&device_out),
                    Arg::U64(elements as u64),
                ],
            )?;
        }
    }
    let before = runtime.stats();
    let start = Instant::now();
    for _ in 0..iterations {
        // SAFETY: as above.
        unsafe {
            runtime.launch(
                &function,
                config,
                &[
                    Arg::DevicePtr(&device_in),
                    Arg::DevicePtrUninit(&device_out),
                    Arg::U64(elements as u64),
                ],
            )?;
        }
    }
    let launch_elapsed = start.elapsed();
    // SAFETY: every launch fully overwrote the output.
    let device_out = unsafe { device_out.assume_init() };
    let start = Instant::now();
    let mut pending = runtime.download(&device_out)?;
    let host = pending.wait()?;
    let download_elapsed = start.elapsed();

    let expected: Vec<u8> = input
        .iter()
        .flat_map(|value| (value + 1).to_ne_bytes())
        .collect();
    assert_eq!(
        host, expected,
        "numerical result must match the CPU reference"
    );

    let after = runtime.stats();
    println!("runtime warm chain");
    println!("  cold compile: {:?}", cold_compile);
    println!(
        "  {} requested launches; window deltas: {} events, {} epochs reclaimed, {} host waits, {} pending records",
        iterations,
        after.retirement.events_recorded - before.retirement.events_recorded,
        after.retirement.reclaimed_epochs - before.retirement.reclaimed_epochs,
        after.retirement.host_waits - before.retirement.host_waits,
        after.pending_records
    );
    println!(
        "  per-launch: {:.3} us",
        launch_elapsed.as_secs_f64() * 1e6 / iterations as f64
    );
    println!("  download+sync: {:?}", download_elapsed);
    println!(
        "  pending records after: {} (before {})",
        after.pending_records, before.pending_records
    );
    println!(
        "  artifact cache: {} hit(s) / {} miss(es); NVRTC {} attempt(s) / {} success(es); module loads {}",
        runtime.artifact_cache_hits(),
        runtime.artifact_cache_misses(),
        runtime.compile_attempts(),
        runtime.compile_successes(),
        runtime.module_loads()
    );

    // A direct-cudarc baseline for the same three operations, using the same
    // explicit pool/stream configuration the runtime uses.
    let direct = direct_baseline(&bytes, iterations)?;
    println!(
        "direct cudarc baseline: {:.3} us/iteration (same process, unpaired; indicative only)",
        direct
    );
    runtime.close()?;
    Ok(())
}

fn direct_baseline(bytes: &[u8], iterations: usize) -> Result<f64, Box<dyn std::error::Error>> {
    use cudarc::driver::{result as drv, sys};
    use std::time::Instant;

    unsafe {
        if !sys::is_culib_present() {
            return Ok(f64::NAN);
        }
        drv::init()?;
        let device = drv::device::get(0)?;
        let context = drv::primary_ctx::retain(device)?;
        drv::ctx::set_current(context)?;
        let stream = drv::stream::create(drv::stream::StreamKind::NonBlocking)?;
        let pool = drv::mem_pool::create(&sys::CUmemPoolProps {
            allocType: sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED,
            handleTypes: sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_NONE,
            location: sys::CUmemLocation {
                type_: sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
                id: 0,
            },
            win32SecurityAttributes: std::ptr::null_mut(),
            maxSize: 0,
            usage: 0,
            reserved: [0; 54],
        })?;
        let src = drv::mem_pool::alloc_async(pool, bytes.len(), stream)?;
        let dst = drv::mem_pool::alloc_async(pool, bytes.len(), stream)?;
        drv::memcpy_htod_async(src, bytes, stream)?;
        // Compile through the same context-free artifact path, then execute with
        // raw driver calls only.
        let artifact = t4a_cuda_runtime::compile(&t4a_cuda_runtime::CompileRequest::new(
            "add1-direct",
            "extern \"C\" __global__ void add1(const unsigned int* input, unsigned int* output, unsigned long long n) { unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; if (i < n) output[i] = input[i] + 1u; }",
        ))?;
        let mut image = artifact.image().to_vec();
        image.push(0);
        let module = drv::module::load_data(image.as_ptr().cast())?;
        let function = drv::module::get_function(module, std::ffi::CString::new("add1")?)?;
        let elements = (bytes.len() / 4) as u64;
        let mut n = elements;
        let mut params = [
            (&src) as *const u64 as *mut std::ffi::c_void,
            (&dst) as *const u64 as *mut std::ffi::c_void,
            (&mut n) as *mut u64 as *mut std::ffi::c_void,
        ];
        let mut launch = || {
            sys::cuLaunchKernel(
                function,
                4,
                1,
                1,
                256,
                1,
                1,
                0,
                stream,
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        for _ in 0..10 {
            assert_eq!(launch(), sys::CUresult::CUDA_SUCCESS);
        }
        let start = Instant::now();
        for _ in 0..iterations {
            assert_eq!(launch(), sys::CUresult::CUDA_SUCCESS);
        }
        // Stop the submission window before synchronising, so this number and the
        // runtime's submission number cover the same work.
        let elapsed = start.elapsed().as_secs_f64() * 1e6 / iterations as f64;
        drv::stream::synchronize(stream)?;
        let _ = drv::mem_pool::destroy(pool);
        let _ = drv::stream::destroy(stream);
        let _ = drv::primary_ctx::release(device);
        let _ = drv::ctx::set_current(std::ptr::null_mut());
        Ok(elapsed)
    }
}
