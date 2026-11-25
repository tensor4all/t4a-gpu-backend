//! Public contract tests that need no device.
//!
//! On a host with ROCm these paths reach real HIP; on a host without it they observe the
//! typed absent error. Neither case may panic or fail to start.

use std::process::Command;

use t4a_hip_runtime::sys::{GfxOverride, RocblasTable};
use t4a_hip_runtime::{device_count, probe, Error, Runtime};

/// Set in the child process that runs [`child_reports_the_empty_override`].
const CHILD_MARKER: &str = "T4A_HIP_EMPTY_OVERRIDE_CHILD";

#[test]
fn opening_a_runtime_is_fallible_and_never_panics() {
    match Runtime::open(0) {
        Ok(runtime) => assert!(!runtime.device().gcn_arch_name.is_empty()),
        Err(Error::Unavailable { what, detail }) => {
            assert!(!what.is_empty());
            assert!(!detail.is_empty());
        }
        Err(Error::InvalidDevice {
            ordinal: 0,
            count: 0,
        }) => {}
        Err(other) => panic!("unexpected open error: {other}"),
    }
}

#[test]
fn the_probe_and_device_count_report_typed_results() {
    match (device_count(), probe()) {
        (Ok(count), Ok(devices)) => assert_eq!(count, devices.len()),
        (Err(Error::Unavailable { .. }), Err(Error::Unavailable { .. })) => {}
        other => panic!("inconsistent discovery: {other:?}"),
    }
}

#[test]
fn an_ordinal_past_the_visible_devices_is_typed() {
    match Runtime::open(10_000) {
        Err(Error::InvalidDevice { ordinal, .. }) => assert_eq!(ordinal, 10_000),
        Err(Error::Unavailable { .. }) => {}
        other => panic!("expected a typed refusal, got {other:?}"),
    }
}

/// Runs only inside the child process the next test starts with
/// `HSA_OVERRIDE_GFX_VERSION=` (exported, empty).
#[test]
fn child_reports_the_empty_override() {
    if std::env::var_os(CHILD_MARKER).is_none() {
        return;
    }
    assert_eq!(GfxOverride::current(), GfxOverride::Empty);
    match Runtime::open(0) {
        Err(Error::Configuration { variable, detail }) => {
            assert_eq!(variable, "HSA_OVERRIDE_GFX_VERSION");
            println!("runtime refused: {variable} {detail}");
        }
        other => panic!("an empty HSA_OVERRIDE_GFX_VERSION must be refused by name, got {other:?}"),
    }
    // rocBLAS is refused for the same reason, before any handle could fail with a
    // misleading `rocblas_status_invalid_handle`.
    match RocblasTable::load() {
        Err(t4a_hip_runtime::sys::Error::Configuration { variable, .. }) => {
            assert_eq!(variable, "HSA_OVERRIDE_GFX_VERSION");
        }
        other => panic!("rocBLAS must be refused by name, got {other:?}"),
    }
}

#[test]
fn an_empty_gfx_override_is_refused_by_name_before_hip_loads() {
    // The variable cannot be changed inside this multi-threaded test process, so a child
    // process gets it from the start, as a user's shell would export it.
    let output = Command::new(std::env::current_exe().expect("test binary"))
        .args(["--exact", "child_reports_the_empty_override", "--nocapture"])
        .env(CHILD_MARKER, "1")
        .env("HSA_OVERRIDE_GFX_VERSION", "")
        .output()
        .expect("run the child test");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "child failed:\n{stdout}\n{stderr}");
    assert!(
        stdout.contains("runtime refused: HSA_OVERRIDE_GFX_VERSION is set but empty"),
        "the child must name the variable:\n{stdout}"
    );
}

/// Set in the child process that runs [`child_refuses_a_second_mapped_hip_runtime`].
const DUPLICATE_MARKER: &str = "T4A_HIP_DUPLICATE_RUNTIME_CHILD";

/// Runs only inside the child process the next test starts with a second copy of
/// `libamdhip64` preloaded.
#[test]
fn child_refuses_a_second_mapped_hip_runtime() {
    if std::env::var_os(DUPLICATE_MARKER).is_none() {
        return;
    }
    match Runtime::open(0) {
        Err(Error::Unavailable { detail, .. }) if detail.contains("more than one HIP runtime") => {
            println!("runtime refused: {detail}");
        }
        other => panic!(
            "a second mapped HIP runtime must be refused before any stream exists, got {other:?}"
        ),
    }
}

#[test]
fn a_second_mapped_hip_runtime_is_refused_before_a_stream_is_created() {
    // Review of t4a-gpu-backend#19: with another `libamdhip64` already in the process (the
    // reviewer preloaded the distro's `libamdhip64.so.5` on evo2), `Runtime::open` went on
    // to create a stream in one of two runtimes and the process died. The portable way to
    // get two runtimes is to preload a copy of the real library under another path and
    // pin `ROCM_PATH` to the real prefix, so the loader maps both files.
    let library = match t4a_hip_runtime::sys::HipTable::load() {
        Ok(hip) => std::fs::canonicalize(hip.library_path()).expect("canonical HIP path"),
        Err(error) => {
            assert!(
                !matches!(std::env::var("T4A_REQUIRE_HIP").as_deref(), Ok("1")),
                "T4A_REQUIRE_HIP=1 but HIP is unavailable: {error}"
            );
            eprintln!("skipped: {error}");
            return;
        }
    };
    let lib_dir = library.parent().expect("library directory");
    if lib_dir.file_name().and_then(|name| name.to_str()) != Some("lib") {
        eprintln!(
            "skipped: {} is not in a `<prefix>/lib` layout",
            library.display()
        );
        return;
    }
    let prefix = lib_dir.parent().expect("ROCm prefix");
    let copy_dir = std::env::temp_dir().join(format!("t4a-hip-duplicate-{}", std::process::id()));
    std::fs::create_dir_all(&copy_dir).expect("copy directory");
    let copy = copy_dir.join("libamdhip64.so.7");
    std::fs::copy(&library, &copy).expect("copy libamdhip64");
    let output = Command::new(std::env::current_exe().expect("test binary"))
        .args([
            "--exact",
            "child_refuses_a_second_mapped_hip_runtime",
            "--nocapture",
        ])
        .env(DUPLICATE_MARKER, "1")
        .env("LD_PRELOAD", &copy)
        .env("ROCM_PATH", prefix)
        .output()
        .expect("run the child test");
    let _ = std::fs::remove_dir_all(&copy_dir);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "child failed ({:?}):\n{stdout}\n{stderr}",
        output.status
    );
    assert!(
        stdout.contains("runtime refused: more than one HIP runtime is mapped"),
        "the child must report the duplicate runtime:\n{stdout}"
    );
}
