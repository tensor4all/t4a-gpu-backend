use std::ffi::OsStr;
use std::path::{Path, PathBuf};

use super::*;

#[test]
fn rocm_path_pins_the_prefix_and_excludes_the_unversioned_name() {
    let candidates = hip_candidates(Some(OsStr::new("/opt/rocm-7.2.4")));
    assert_eq!(
        candidates,
        vec![
            PathBuf::from("/opt/rocm-7.2.4/lib/libamdhip64.so.7"),
            PathBuf::from("/opt/rocm-7.2.4/lib/libamdhip64.so.6"),
        ]
    );
}

#[test]
fn without_rocm_path_the_bare_sonames_go_to_the_dynamic_loader() {
    for unset in [None, Some(OsStr::new(""))] {
        assert_eq!(
            hip_candidates(unset),
            vec![
                PathBuf::from("libamdhip64.so.7"),
                PathBuf::from("libamdhip64.so.6")
            ]
        );
    }
}

#[test]
fn math_libraries_are_only_looked_up_next_to_the_runtime() {
    let runtime = Path::new("/opt/rocm-7.2.4/lib/libamdhip64.so.7.2.70204");
    assert_eq!(
        sibling_candidates(runtime, ROCBLAS_SONAMES),
        vec![
            PathBuf::from("/opt/rocm-7.2.4/lib/librocblas.so.5"),
            PathBuf::from("/opt/rocm-7.2.4/lib/librocblas.so.4"),
        ]
    );
}

#[test]
fn the_maps_scan_finds_each_distinct_hip_runtime_once() {
    // Two segments of one file plus a distro HIP 5 runtime: two distinct files.
    let maps = "\
7f0000000000-7f0000001000 r--p 00000000 103:02 1 /opt/rocm-7.2.4/lib/libamdhip64.so.7.2.70204
7f0000001000-7f0000002000 r-xp 00001000 103:02 1 /opt/rocm-7.2.4/lib/libamdhip64.so.7.2.70204
7f0000003000-7f0000004000 r--p 00000000 103:02 2 /lib/x86_64-linux-gnu/libamdhip64.so.5
7f0000005000-7f0000006000 r--p 00000000 103:02 3 /opt/rocm-7.2.4/lib/librocblas.so.5.2.70204
7f0000007000-7f0000008000 rw-p 00000000 00:00 0
7f0000009000-7f000000a000 r--p 00000000 103:02 4 /home/a user/libamdhip64.so.7
";
    assert_eq!(
        hip_runtimes_in_maps(maps),
        vec![
            PathBuf::from("/opt/rocm-7.2.4/lib/libamdhip64.so.7.2.70204"),
            PathBuf::from("/lib/x86_64-linux-gnu/libamdhip64.so.5"),
            PathBuf::from("/home/a user/libamdhip64.so.7"),
        ]
    );
}

#[test]
fn the_gfx_override_distinguishes_unset_empty_and_a_value() {
    assert_eq!(GfxOverride::from_value(None), GfxOverride::Unset);
    assert_eq!(
        GfxOverride::from_value(Some(OsStr::new(""))),
        GfxOverride::Empty
    );
    assert_eq!(
        GfxOverride::from_value(Some(OsStr::new("11.5.1"))),
        GfxOverride::Value("11.5.1".to_string())
    );
    assert!(GfxOverride::Unset.check().is_ok());
    assert!(GfxOverride::Value("11.5.1".to_string()).check().is_ok());
    match GfxOverride::Empty.check() {
        Err(Error::Configuration { variable, detail }) => {
            assert_eq!(variable, "HSA_OVERRIDE_GFX_VERSION");
            assert!(detail.contains("set but empty"), "{detail}");
        }
        other => panic!("an empty override must be a configuration error, got {other:?}"),
    }
}

#[test]
fn a_library_that_is_absent_reports_every_candidate() {
    let candidates = vec![
        PathBuf::from("/nonexistent/libamdhip64.so.7"),
        PathBuf::from("/nonexistent/libamdhip64.so.6"),
    ];
    match Library::open("HIP runtime", &candidates, b"hipGetDeviceCount\0") {
        Err(Error::LibraryUnavailable { library, detail }) => {
            assert_eq!(library, "HIP runtime");
            assert!(detail.contains("/nonexistent/libamdhip64.so.7"), "{detail}");
            assert!(detail.contains("/nonexistent/libamdhip64.so.6"), "{detail}");
        }
        other => panic!("expected a typed unavailable error, got {other:?}"),
    }
}
