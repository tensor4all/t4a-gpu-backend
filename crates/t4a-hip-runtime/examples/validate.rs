//! Standalone validation of the HIP runtime (tenferro-rs#2489, M1).
//!
//! ```text
//! cargo run --release -p t4a-hip-runtime --example validate -- [--cuda-first]
//! ```
//!
//! The counterpart of what the CUDA side validates, on one AMD device:
//!
//! 1. **Enumeration** — everything HIP reports for every visible device.
//! 2. **GEMM** — rocBLAS `f32` and `f64` through the runtime's own stream and pool,
//!    checked element by element against the compensated CPU reference, then timed; the
//!    `f64`/`f32` throughput ratio is printed next to the issue's recorded ~1/7.
//! 3. **Dual-vendor symbol test** — the HIP libraries (`libamdhip64`, rocBLAS, and
//!    `libhiprtc` with the LLVM inside `libamd_comgr`) and the CUDA libraries (NVRTC,
//!    which bundles its own LLVM, and the driver library) are loaded into one process,
//!    every one with local symbol scope, and then both are asked to do real work: hiprtc
//!    and NVRTC each compile a kernel, and a rocBLAS GEMM runs again afterwards. Neither
//!    vendor's symbols may be visible through the global namespace.
//!
//! Exit status 0 means every step that could run passed. A step that cannot run on this
//! host (no NVIDIA driver library, no NVRTC) is printed as `NOT RUN` with the reason and
//! never as a pass. Throughput is recorded, never gated; a ratio far from 1/7 is printed
//! as a disagreement because M2 builds on it.

#[path = "../tests/support/mod.rs"]
mod support;

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::Path;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use support::rocblas_seam::{DType, GemmRequest, Rocblas};
use support::{decode, encode, reference, values, worst_ratio, Operand};
use t4a_hip_runtime::{probe, Runtime};

/// Edge of the CPU-checked product.
const CHECK_N: usize = 192;
/// Edge of the timed product.
const TIMED_N: usize = 2048;

fn main() -> ExitCode {
    let cuda_first = std::env::args().any(|arg| arg == "--cuda-first");
    let mut failures: Vec<String> = Vec::new();
    let mut not_run: Vec<String> = Vec::new();

    println!("== 1. device enumeration ==");
    match probe() {
        Ok(devices) if devices.is_empty() => {
            println!("HIP loaded, no device visible");
            failures.push("no HIP device".into());
        }
        Ok(devices) => {
            for device in &devices {
                println!("device {}: {}", device.ordinal, device.name);
                println!("  gcnArchName        {}", device.gcn_arch_name);
                println!("  warpSize           {}", device.warp_size);
                println!("  integrated         {}", device.integrated);
                println!(
                    "  totalGlobalMem     {} bytes ({:.2} GiB)",
                    device.total_global_mem,
                    device.total_global_mem as f64 / (1u64 << 30) as f64
                );
                println!("  memory pools       {}", device.memory_pools);
                println!("  hipRuntimeVersion  {}", device.runtime_version);
                println!("  hipDriverVersion   {}", device.driver_version);
                println!("  libamdhip64        {}", device.library.display());
                println!("  HSA_OVERRIDE_GFX   {:?}", device.gfx_override);
            }
        }
        Err(error) => {
            println!("HIP is not usable on this host: {error}");
            failures.push(format!("enumeration: {error}"));
            return finish(&failures, &not_run);
        }
    }

    let mut runtime = match Runtime::open(0) {
        Ok(runtime) => runtime,
        Err(error) => {
            println!("opening device 0 failed: {error}");
            failures.push(format!("open: {error}"));
            return finish(&failures, &not_run);
        }
    };
    let mut blas = match Rocblas::new(&mut runtime) {
        Ok(blas) => blas,
        Err(error) => {
            println!("rocBLAS handle failed: {error:?}");
            failures.push(format!("rocBLAS handle: {error:?}"));
            return finish(&failures, &not_run);
        }
    };

    println!("\n== 2. rocBLAS GEMM, f32 and f64 ==");
    let mut gflops = [0.0f64; 2];
    for (slot, dtype) in [DType::F32, DType::F64].into_iter().enumerate() {
        match check_against_cpu(&mut runtime, &mut blas, dtype) {
            Ok(worst) => println!(
                "{dtype:?} {CHECK_N}x{CHECK_N}x{CHECK_N} vs CPU reference: worst error/bound \
                 {worst:.3e} (pass <= 1) {}",
                if worst <= 1.0 { "PASS" } else { "FAIL" }
            ),
            Err(error) => {
                println!("{dtype:?} check failed: {error}");
                failures.push(format!("{dtype:?} check: {error}"));
                continue;
            }
        }
        match time_gemm(&mut runtime, &mut blas, dtype, TIMED_N) {
            Ok(rate) => {
                println!("{dtype:?} n={TIMED_N}: {rate:9.1} GFLOP/s");
                gflops[slot] = rate;
            }
            Err(error) => failures.push(format!("{dtype:?} timing: {error}")),
        }
    }
    if gflops[0] > 0.0 && gflops[1] > 0.0 {
        let ratio = gflops[1] / gflops[0];
        println!(
            "fp64/fp32 throughput ratio at n={TIMED_N}: {ratio:.4} (= 1/{:.2}); the issue \
             records about 1/7 = 0.143",
            1.0 / ratio
        );
        if (0.08..=0.25).contains(&ratio) {
            println!("  agrees with the issue's ~1/7 (accepted band 0.08..0.25)");
        } else {
            println!("  DISAGREES with the issue's ~1/7: report this before M2 builds on it");
        }
    }

    println!(
        "\n== 3. dual-vendor symbol test (order: {}) ==",
        if cuda_first {
            "CUDA, then HIP"
        } else {
            "HIP, then CUDA"
        }
    );
    let hip_dir = runtime
        .device()
        .library
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_default();
    dual_vendor(
        &mut runtime,
        &mut blas,
        &hip_dir,
        cuda_first,
        &mut failures,
        &mut not_run,
    );

    finish(&failures, &not_run)
}

fn finish(failures: &[String], not_run: &[String]) -> ExitCode {
    println!();
    for step in not_run {
        println!("NOT RUN: {step}");
    }
    if failures.is_empty() {
        println!(
            "RESULT: every step that ran passed ({} step(s) NOT RUN on this host)",
            not_run.len()
        );
        ExitCode::SUCCESS
    } else {
        println!("RESULT: FAILED");
        for failure in failures {
            println!("  - {failure}");
        }
        ExitCode::FAILURE
    }
}

fn gemm_request<'a>(
    dtype: DType,
    a: &'a t4a_hip_runtime::Buffer,
    b: &'a t4a_hip_runtime::Buffer,
    c: &'a t4a_hip_runtime::Buffer,
    n: usize,
) -> GemmRequest<'a> {
    let n = n as i64;
    GemmRequest {
        dtype,
        a,
        trans_a: false,
        b,
        trans_b: false,
        c,
        shape: (n, n, n),
        leading: (n, n, n),
        batches: (1, 0, 0, 0),
    }
}

fn check_against_cpu(
    runtime: &mut Runtime,
    blas: &mut Rocblas,
    dtype: DType,
) -> Result<f64, String> {
    let n = CHECK_N;
    let a_host = values(dtype, n * n, 11);
    let b_host = values(dtype, n * n, 12);
    let a = runtime
        .upload(&encode(dtype, &a_host))
        .map_err(|e| e.to_string())?;
    let b = runtime
        .upload(&encode(dtype, &b_host))
        .map_err(|e| e.to_string())?;
    let c = runtime
        .alloc_zeroed(n * n * dtype.size())
        .map_err(|e| e.to_string())?;
    blas.gemm(runtime, &gemm_request(dtype, &a, &b, &c, n))
        .map_err(|e| format!("{e:?}"))?;
    runtime.synchronize().map_err(|e| e.to_string())?;
    let observed = decode(dtype, &runtime.download(&c).map_err(|e| e.to_string())?);
    let (expected, magnitude) = reference(
        &Operand {
            values: &a_host,
            ld: n,
            transposed: false,
        },
        &Operand {
            values: &b_host,
            ld: n,
            transposed: false,
        },
        n,
        n,
        n,
    );
    Ok(worst_ratio(dtype, &observed, &expected, &magnitude, n))
}

fn time_gemm(
    runtime: &mut Runtime,
    blas: &mut Rocblas,
    dtype: DType,
    n: usize,
) -> Result<f64, String> {
    let a = runtime
        .upload(&encode(dtype, &values(dtype, n * n, 1)))
        .map_err(|e| e.to_string())?;
    let b = runtime
        .upload(&encode(dtype, &values(dtype, n * n, 2)))
        .map_err(|e| e.to_string())?;
    let c = runtime
        .alloc_zeroed(n * n * dtype.size())
        .map_err(|e| e.to_string())?;
    let request = gemm_request(dtype, &a, &b, &c, n);
    for _ in 0..2 {
        blas.gemm(runtime, &request).map_err(|e| format!("{e:?}"))?;
    }
    runtime.synchronize().map_err(|e| e.to_string())?;
    let start = Instant::now();
    let mut reps = 0u32;
    while reps < 3 || start.elapsed() < Duration::from_secs(1) {
        blas.gemm(runtime, &request).map_err(|e| format!("{e:?}"))?;
        reps += 1;
        if reps % 4 == 0 {
            runtime.synchronize().map_err(|e| e.to_string())?;
        }
    }
    runtime.synchronize().map_err(|e| e.to_string())?;
    let per_call = start.elapsed().as_secs_f64() / f64::from(reps);
    Ok(dtype.flops(n as f64, n as f64, n as f64) / per_call / 1e9)
}

// ---------------------------------------------------------------------------------
// Dual-vendor symbol test
// ---------------------------------------------------------------------------------

/// Whether `symbol` resolves through the process's *global* namespace.
fn globally_visible(symbol: &str) -> bool {
    let name = CString::new(symbol).expect("symbol");
    // SAFETY: `RTLD_DEFAULT` with a valid NUL-terminated name.
    !unsafe { libc::dlsym(libc::RTLD_DEFAULT, name.as_ptr()) }.is_null()
}

/// A `libhiprtc` opened `RTLD_NOW | RTLD_LOCAL` from the HIP runtime's own directory.
struct Hiprtc {
    handle: *mut c_void,
}

impl Hiprtc {
    fn open(directory: &Path) -> Result<Hiprtc, String> {
        let mut tried = Vec::new();
        for soname in ["libhiprtc.so.7", "libhiprtc.so.6"] {
            let path = directory.join(soname);
            let c = CString::new(path.to_string_lossy().as_bytes()).map_err(|e| e.to_string())?;
            // SAFETY: a valid path string; the library is never closed.
            let handle = unsafe { libc::dlopen(c.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
            if !handle.is_null() {
                return Ok(Hiprtc { handle });
            }
            // SAFETY: `dlerror` returns a NUL-terminated message or null.
            let message = unsafe {
                let e = libc::dlerror();
                if e.is_null() {
                    String::new()
                } else {
                    CStr::from_ptr(e).to_string_lossy().into_owned()
                }
            };
            tried.push(format!("{}: {message}", path.display()));
        }
        Err(tried.join("; "))
    }

    fn symbol(&self, name: &str) -> Result<*mut c_void, String> {
        let c = CString::new(name).expect("symbol");
        // SAFETY: a live handle and a valid name.
        let ptr = unsafe { libc::dlsym(self.handle, c.as_ptr()) };
        if ptr.is_null() {
            Err(format!("libhiprtc lacks `{name}`"))
        } else {
            Ok(ptr)
        }
    }

    /// Compile `source` for `arch` and return the code object's size in bytes.
    fn compile(&self, source: &str, arch: &str) -> Result<usize, String> {
        type Create = unsafe extern "C" fn(
            *mut *mut c_void,
            *const c_char,
            *const c_char,
            c_int,
            *const *const c_char,
            *const *const c_char,
        ) -> c_int;
        type Compile = unsafe extern "C" fn(*mut c_void, c_int, *const *const c_char) -> c_int;
        type Size = unsafe extern "C" fn(*mut c_void, *mut usize) -> c_int;
        type Get = unsafe extern "C" fn(*mut c_void, *mut c_char) -> c_int;
        type Destroy = unsafe extern "C" fn(*mut *mut c_void) -> c_int;
        // SAFETY: each symbol is cast to the documented hiprtc prototype.
        let (create, compile, size, get, log_size, log, destroy) = unsafe {
            (
                std::mem::transmute::<*mut c_void, Create>(self.symbol("hiprtcCreateProgram")?),
                std::mem::transmute::<*mut c_void, Compile>(self.symbol("hiprtcCompileProgram")?),
                std::mem::transmute::<*mut c_void, Size>(self.symbol("hiprtcGetCodeSize")?),
                std::mem::transmute::<*mut c_void, Get>(self.symbol("hiprtcGetCode")?),
                std::mem::transmute::<*mut c_void, Size>(self.symbol("hiprtcGetProgramLogSize")?),
                std::mem::transmute::<*mut c_void, Get>(self.symbol("hiprtcGetProgramLog")?),
                std::mem::transmute::<*mut c_void, Destroy>(self.symbol("hiprtcDestroyProgram")?),
            )
        };
        let source = CString::new(source).map_err(|e| e.to_string())?;
        let name = CString::new("validate.hip").map_err(|e| e.to_string())?;
        let option = CString::new(format!("--offload-arch={arch}")).map_err(|e| e.to_string())?;
        let options = [option.as_ptr()];
        let mut program: *mut c_void = std::ptr::null_mut();
        // SAFETY: every pointer is valid for the call; the program is destroyed below.
        unsafe {
            let status = create(
                &mut program,
                source.as_ptr(),
                name.as_ptr(),
                0,
                std::ptr::null(),
                std::ptr::null(),
            );
            if status != 0 {
                return Err(format!("hiprtcCreateProgram returned {status}"));
            }
            let status = compile(program, 1, options.as_ptr());
            let result = if status == 0 {
                let mut bytes = 0usize;
                if size(program, &mut bytes) == 0 && bytes > 0 {
                    let mut code = vec![0 as c_char; bytes];
                    if get(program, code.as_mut_ptr()) == 0 {
                        Ok(bytes)
                    } else {
                        Err("hiprtcGetCode failed".to_string())
                    }
                } else {
                    Err("hiprtcGetCodeSize returned no code".to_string())
                }
            } else {
                let mut bytes = 0usize;
                let _ = log_size(program, &mut bytes);
                let mut text = vec![0 as c_char; bytes.max(1)];
                let _ = log(program, text.as_mut_ptr());
                Err(format!(
                    "hiprtcCompileProgram returned {status}: {}",
                    CStr::from_ptr(text.as_ptr()).to_string_lossy()
                ))
            };
            destroy(&mut program);
            result
        }
    }
}

const HIP_KERNEL: &str =
    "extern \"C\" __global__ void t4a_validate(float* x, float a) { x[threadIdx.x] += a; }\n";
const CUDA_KERNEL: &str =
    "extern \"C\" __global__ void t4a_validate(float* x, float a) { x[threadIdx.x] += a; }\n";

/// Compile one kernel with NVRTC through cudarc (the loader `t4a-cuda-runtime` uses).
/// Needs no driver and no device.
fn nvrtc_compile() -> Result<(usize, String), String> {
    // SAFETY: cudarc's own probe; it only attempts to open the library.
    if !unsafe { cudarc::nvrtc::sys::is_culib_present() } {
        return Err("no libnvrtc is loadable".into());
    }
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(
        CUDA_KERNEL,
        cudarc::nvrtc::CompileOptions {
            arch: Some("compute_80"),
            ..Default::default()
        },
    )
    .map_err(|e| format!("{e:?}"))?;
    let text = ptx.to_src();
    let target = text
        .lines()
        .find(|line| line.starts_with(".target"))
        .unwrap_or("")
        .to_string();
    Ok((text.len(), target))
}

fn nvrtc_version() -> Option<(c_int, c_int)> {
    let (mut major, mut minor) = (0, 0);
    // SAFETY: two writable ints; only reached after `is_culib_present` held.
    let status = unsafe { cudarc::nvrtc::sys::nvrtcVersion(&mut major, &mut minor) };
    (status == cudarc::nvrtc::sys::nvrtcResult::NVRTC_SUCCESS).then_some((major, minor))
}

fn hip_side(
    label: &str,
    runtime: &mut Runtime,
    blas: &mut Rocblas,
    hiprtc: &Hiprtc,
    arch: &str,
    failures: &mut Vec<String>,
) {
    match hiprtc.compile(HIP_KERNEL, arch) {
        Ok(bytes) => println!("[{label}] hiprtc compiled a kernel for {arch}: {bytes} bytes"),
        Err(error) => {
            println!("[{label}] hiprtc FAILED: {error}");
            failures.push(format!("{label}: hiprtc: {error}"));
        }
    }
    match check_against_cpu(runtime, blas, DType::F32) {
        Ok(worst) if worst <= 1.0 => {
            println!("[{label}] rocBLAS sgemm vs CPU: worst error/bound {worst:.3e} PASS")
        }
        Ok(worst) => {
            println!("[{label}] rocBLAS sgemm FAIL: {worst:.3e}");
            failures.push(format!("{label}: rocBLAS result wrong"));
        }
        Err(error) => {
            println!("[{label}] rocBLAS sgemm FAILED: {error}");
            failures.push(format!("{label}: rocBLAS: {error}"));
        }
    }
}

fn cuda_side(label: &str, failures: &mut Vec<String>) -> bool {
    match nvrtc_compile() {
        Ok((bytes, target)) => {
            let version = nvrtc_version()
                .map(|(major, minor)| format!("NVRTC {major}.{minor}"))
                .unwrap_or_else(|| "NVRTC (version query failed)".into());
            println!("[{label}] {version} compiled a kernel: {bytes} bytes of PTX ({target})");
            true
        }
        Err(error) if error.contains("no libnvrtc") => {
            println!("[{label}] NOT RUN: {error}; set LD_LIBRARY_PATH to a CUDA lib directory");
            false
        }
        Err(error) => {
            println!("[{label}] NVRTC FAILED: {error}");
            failures.push(format!("{label}: nvrtc: {error}"));
            false
        }
    }
}

fn dual_vendor(
    runtime: &mut Runtime,
    blas: &mut Rocblas,
    hip_dir: &Path,
    cuda_first: bool,
    failures: &mut Vec<String>,
    not_run: &mut Vec<String>,
) {
    let arch = runtime.device().gcn_arch_name.clone();
    let arch = arch.split(':').next().unwrap_or("").to_string();
    let hiprtc = match Hiprtc::open(hip_dir) {
        Ok(hiprtc) => hiprtc,
        Err(error) => {
            println!("libhiprtc could not be loaded: {error}");
            failures.push(format!("hiprtc load: {error}"));
            return;
        }
    };
    println!("libhiprtc loaded RTLD_LOCAL from {}", hip_dir.display());

    let cuda_ran;
    if cuda_first {
        cuda_ran = cuda_side("cuda", failures);
        hip_side("hip ", runtime, blas, &hiprtc, &arch, failures);
    } else {
        hip_side("hip ", runtime, blas, &hiprtc, &arch, failures);
        cuda_ran = cuda_side("cuda", failures);
    }
    // Both stacks are now resident. Ask each again: a symbol resolved against the wrong
    // LLVM would show up here, not at load time.
    if cuda_ran {
        hip_side("hip2", runtime, blas, &hiprtc, &arch, failures);
        cuda_side("cuda2", failures);
    }

    // CUDA driver library (needs an NVIDIA kernel driver to exist at all).
    // SAFETY: cudarc's own probe; it only attempts to open the library.
    if unsafe { cudarc::driver::sys::is_culib_present() } {
        println!("[cuda] libcuda is loadable");
    } else {
        println!(
            "[cuda] NOT RUN: libcuda is not loadable (no NVIDIA driver on this host); device \
             execution and cudaMalloc/launch are untested here"
        );
        not_run.push("libcuda + CUDA device execution beside HIP (no NVIDIA driver)".into());
    }

    println!("-- local scope: none of these may resolve through the global namespace --");
    let mut leaked = Vec::new();
    for symbol in [
        "hipGetDeviceCount",
        "hiprtcCreateProgram",
        "rocblas_create_handle",
        "amd_comgr_create_data",
        "LLVMInitializeAMDGPUTarget",
        "nvrtcCreateProgram",
        "cuInit",
    ] {
        let visible = globally_visible(symbol);
        println!(
            "  dlsym(RTLD_DEFAULT, {symbol:<28}) = {}",
            if visible { "VISIBLE" } else { "null (local)" }
        );
        if visible {
            leaked.push(symbol);
        }
    }
    if !leaked.is_empty() {
        failures.push(format!("symbols leaked into the global scope: {leaked:?}"));
    }
    if !cuda_ran {
        not_run.push("NVRTC beside HIP (no libnvrtc loadable)".into());
        println!(
            "dual-vendor result: HIP half PASSED; CUDA half NOT RUN on this host, so the \
             combined assertion is NOT established here"
        );
    }
}
