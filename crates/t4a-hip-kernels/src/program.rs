//! hiprtc compilation, module loading and the raw launch.

use std::ffi::{c_char, c_int, c_void, CString};
use std::ptr;
use std::time::{Duration, Instant};

use t4a_cuda_kernels::source;
use t4a_cuda_kernels::Launch;
use t4a_hip_runtime::{RawScope, Runtime};
use t4a_hip_sys::{
    GfxOverride, HipFunction, HipModule, HipTable, HiprtcProgram, HiprtcTable, HIPRTC_SUCCESS,
    HIP_SUCCESS,
};

use crate::error::{Error, Result};

/// Text placed **before** the unmodified CUDA text; it is the whole HIP-specific surface.
///
/// The ABI header's warp helpers call `__shfl_xor_sync(unsigned mask, ...)` and
/// `rrqr.cu` calls `__shfl_sync(kWarpMask, ...)`. HIP's `_sync` shuffles are templates
/// that static-assert a 64-bit mask, so the text as written does not compile under
/// hiprtc (`sizeof(unsigned int) == 8` fails). The ABI itself - struct layouts, kernel signatures, dtype tags, launch
/// geometry - is untouched; only this device-helper spelling differs. The shim maps it
/// to HIP's maskless `__shfl_xor`/`__shfl` with an explicit width of 32, the plane width the CUDA
/// sources assume (`PLANE_BLOCK`), so the same lane groups exchange on a wave64 device.
pub const HIP_PRELUDE: &str = "\
#define __shfl_xor_sync(mask, value, offset) __shfl_xor((value), (offset), 32)\n\
#define __shfl_sync(mask, value, lane) __shfl((value), (lane), 32)\n";

/// The text hiprtc compiles: [`HIP_PRELUDE`], then the CUDA crate's composition unchanged.
pub fn compose(module: &source::Module) -> String {
    let mut text = String::from(HIP_PRELUDE);
    text.push_str(&source::compose(module));
    text
}

/// What one compilation did, for a reader to see which architecture the result is about.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompileReport {
    /// Module name (`cuda/src/<name>.cu` in `t4a-cuda-kernels`).
    pub module: String,
    /// The `--offload-arch` value passed to hiprtc, for example `gfx1150`.
    pub target: String,
    /// Every option passed to hiprtc, in order.
    pub options: Vec<String>,
    /// Wall time of `hiprtcCompileProgram` plus fetching the code object.
    pub compile_time: Duration,
    /// Wall time of `hipModuleLoadData`.
    pub load_time: Duration,
    /// Size of the code object.
    pub code_bytes: usize,
    /// hiprtc's log, usually empty on success.
    pub log: String,
    /// `HSA_OVERRIDE_GFX_VERSION` as the process runs under it.
    pub gfx_override: GfxOverride,
}

/// One kernel module compiled for the runtime's device and loaded into it.
///
/// Not `Send`: a module belongs to the device context of the thread that loaded it, as
/// [`Runtime`]'s raw seam does.
#[derive(Debug)]
pub struct Program {
    hip: &'static HipTable,
    module: HipModule,
    ordinal: i32,
    report: CompileReport,
}

impl Program {
    /// Compile one `t4a-cuda-kernels` module for the device of `runtime` and load it.
    ///
    /// The text is exactly [`source::compose`] of the module: the ABI header followed
    /// by the translation unit, unchanged. The target is the device's own
    /// `gcnArchName` without feature suffixes.
    ///
    /// # Errors
    ///
    /// [`Error::Compile`] with the target, options and hiprtc log when hiprtc refuses
    /// the source; [`Error::Runtime`] for a missing library or a failing HIP call.
    pub fn compile(runtime: &Runtime, module: &source::Module) -> Result<Program> {
        let device = runtime.device();
        let target = device
            .gcn_arch_name
            .split(':')
            .next()
            .unwrap_or_default()
            .to_string();
        let options = vec![format!("--offload-arch={target}"), "-std=c++17".to_string()];
        let hip = HipTable::load()?;
        let rtc = HiprtcTable::load()?;
        let text = compose(module);
        let started = Instant::now();
        let (image, log) = compile_image(rtc, module.name, &text, &target, &options)?;
        let compile_time = started.elapsed();

        // The module is loaded into the current context of this thread, so make the
        // runtime's device current first.
        // SAFETY: plain device selection by ordinal the runtime itself opened.
        let code = unsafe { (hip.set_device)(device.ordinal as i32) };
        check(hip, "hipSetDevice", code)?;
        let mut handle: HipModule = ptr::null_mut();
        let started = Instant::now();
        // SAFETY: `image` is a complete code object produced by hiprtc.
        let code = unsafe { (hip.module_load_data)(&mut handle, image.as_ptr().cast()) };
        check(hip, "hipModuleLoadData", code)?;
        let load_time = started.elapsed();

        Ok(Program {
            hip,
            module: handle,
            ordinal: device.ordinal as i32,
            report: CompileReport {
                module: module.name.to_string(),
                target,
                options,
                compile_time,
                load_time,
                code_bytes: image.len(),
                log,
                gfx_override: device.gfx_override.clone(),
            },
        })
    }

    /// The module of `t4a-cuda-kernels` that exports `entry`, compiled and loaded.
    ///
    /// # Errors
    ///
    /// [`Error::NoSuchEntry`] when no module exports the entry; otherwise as
    /// [`Program::compile`].
    pub fn compile_entry(runtime: &Runtime, entry: t4a_cuda_kernels::Entry) -> Result<Program> {
        let module = source::module_of(entry).ok_or_else(|| Error::NoSuchEntry {
            symbol: entry.name().to_string(),
            detail: "no translation unit exports it".to_string(),
        })?;
        Program::compile(runtime, module)
    }

    /// What the compilation did.
    pub fn report(&self) -> &CompileReport {
        &self.report
    }

    /// Queue one kernel on the scope's stream.
    ///
    /// `args` holds one pointer per kernel parameter, each pointing at the parameter's
    /// value, in the order of the `extern "C"` declaration (the `hipModuleLaunchKernel`
    /// `kernelParams` convention, which is also the CUDA driver's). The launch geometry
    /// is a `t4a-cuda-kernels` [`Launch`], used unchanged. An empty plan queues nothing.
    ///
    /// # Safety
    ///
    /// `symbol` must be an entry of this program, `args` must match its declaration in
    /// count, type and size, and every device pointer among them must address memory of
    /// this scope's runtime valid for what the kernel reads and writes until the work
    /// completes.
    ///
    /// # Errors
    ///
    /// [`Error::NoSuchEntry`] and [`Error::Runtime`].
    pub unsafe fn launch(
        &self,
        scope: &RawScope<'_>,
        symbol: &str,
        launch: &Launch,
        args: &mut [*mut c_void],
    ) -> Result<()> {
        if launch.is_empty() {
            return Ok(());
        }
        if scope.ordinal() != self.ordinal {
            return Err(Error::Unsupported {
                detail: format!(
                    "program loaded on device {} but the scope runs on device {}",
                    self.ordinal,
                    scope.ordinal()
                ),
            });
        }
        let name = CString::new(symbol).map_err(|_| Error::NoSuchEntry {
            symbol: symbol.to_string(),
            detail: "contains NUL".to_string(),
        })?;
        let mut function: HipFunction = ptr::null_mut();
        // SAFETY: `name` is NUL-terminated; the module is loaded.
        let code =
            unsafe { (self.hip.module_get_function)(&mut function, self.module, name.as_ptr()) };
        if code != HIP_SUCCESS {
            return Err(Error::NoSuchEntry {
                symbol: symbol.to_string(),
                detail: self.hip.error_name(code),
            });
        }
        // SAFETY: the caller guarantees `args` matches the kernel declaration.
        let code = unsafe {
            (self.hip.module_launch_kernel)(
                function,
                launch.grid[0],
                launch.grid[1],
                launch.grid[2],
                launch.block[0],
                launch.block[1],
                launch.block[2],
                launch.shared_bytes,
                scope.stream(),
                args.as_mut_ptr(),
                ptr::null_mut(),
            )
        };
        check(self.hip, "hipModuleLaunchKernel", code)
    }
}

impl Drop for Program {
    fn drop(&mut self) {
        // Best effort: unloading needs the owning device current.
        // SAFETY: both calls take plain handles this value owns.
        unsafe {
            (self.hip.set_device)(self.ordinal);
            (self.hip.module_unload)(self.module);
        }
    }
}

fn check(hip: &HipTable, call: &'static str, code: i32) -> Result<()> {
    if code == HIP_SUCCESS {
        return Ok(());
    }
    Err(Error::Runtime(t4a_hip_runtime::Error::Hip {
        call,
        code,
        name: hip.error_name(code),
    }))
}

fn program_log(rtc: &HiprtcTable, program: HiprtcProgram) -> String {
    let mut size = 0usize;
    // SAFETY: `program` is live and `size` writable.
    if unsafe { (rtc.get_program_log_size)(program, &mut size) } != HIPRTC_SUCCESS || size == 0 {
        return String::new();
    }
    let mut buffer = vec![0u8; size];
    // SAFETY: `buffer` holds `size` bytes as reported.
    if unsafe { (rtc.get_program_log)(program, buffer.as_mut_ptr().cast::<c_char>()) }
        != HIPRTC_SUCCESS
    {
        return String::new();
    }
    String::from_utf8_lossy(&buffer)
        .trim_end_matches('\0')
        .trim()
        .to_string()
}

fn compile_image(
    rtc: &HiprtcTable,
    module: &str,
    text: &str,
    target: &str,
    options: &[String],
) -> Result<(Vec<u8>, String)> {
    let fail = |result: String, log: String| Error::Compile {
        module: module.to_string(),
        target: target.to_string(),
        options: options.to_vec(),
        result,
        log,
    };
    let source =
        CString::new(text).map_err(|_| fail("source contains NUL".into(), String::new()))?;
    let name = CString::new(format!("{module}.cu")).expect("module names carry no NUL");
    let mut program: HiprtcProgram = ptr::null_mut();
    // SAFETY: the strings are NUL-terminated and outlive the call; no headers.
    let status = unsafe {
        (rtc.create_program)(
            &mut program,
            source.as_ptr(),
            name.as_ptr(),
            0,
            ptr::null(),
            ptr::null(),
        )
    };
    if status != HIPRTC_SUCCESS {
        return Err(fail(rtc.error_string(status), String::new()));
    }
    let owned: Vec<CString> = options
        .iter()
        .map(|o| CString::new(o.as_str()).expect("options carry no NUL"))
        .collect();
    let pointers: Vec<*const c_char> = owned.iter().map(|o| o.as_ptr()).collect();
    // SAFETY: `pointers` holds `len` NUL-terminated option strings.
    let status =
        unsafe { (rtc.compile_program)(program, pointers.len() as c_int, pointers.as_ptr()) };
    let log = program_log(rtc, program);
    let result = (|| {
        if status != HIPRTC_SUCCESS {
            return Err(fail(rtc.error_string(status), log.clone()));
        }
        let mut size = 0usize;
        // SAFETY: `program` compiled; `size` writable.
        let status = unsafe { (rtc.get_code_size)(program, &mut size) };
        if status != HIPRTC_SUCCESS || size == 0 {
            return Err(fail(rtc.error_string(status), log.clone()));
        }
        let mut image = vec![0u8; size];
        // SAFETY: `image` holds `size` bytes as reported.
        let status = unsafe { (rtc.get_code)(program, image.as_mut_ptr().cast::<c_char>()) };
        if status != HIPRTC_SUCCESS {
            return Err(fail(rtc.error_string(status), log.clone()));
        }
        Ok(image)
    })();
    // SAFETY: `program` was created above and is destroyed once.
    unsafe { (rtc.destroy_program)(&mut program) };
    result.map(|image| (image, log))
}
