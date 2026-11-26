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

#[test]
fn a_present_but_unloadable_candidate_is_unusable_not_absent() {
    // Review round 3 of t4a-gpu-backend#19: a broken install must not read as "no ROCm".
    let dir = std::env::temp_dir().join(format!("t4a-hip-sys-broken-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let not_elf = dir.join("libamdhip64.so.7");
    std::fs::write(&not_elf, b"not an elf\n").expect("fake image");
    let dangling = dir.join("libamdhip64.so.6");
    #[cfg(unix)]
    std::os::unix::fs::symlink(dir.join("missing-target"), &dangling).expect("symlink");
    let absent = dir.join("libnothing.so.1");
    let outcome = |candidates: Vec<PathBuf>| Library::open("HIP runtime", &candidates, b"x\0");
    #[cfg(target_os = "linux")]
    {
        assert!(matches!(
            outcome(vec![not_elf.clone()]),
            Err(Error::LibraryUnusable { .. })
        ));
        assert!(matches!(
            outcome(vec![absent.clone(), dangling.clone()]),
            Err(Error::LibraryUnusable { .. })
        ));
    }
    assert!(matches!(
        outcome(vec![absent.clone()]),
        Err(Error::LibraryUnavailable { .. })
    ));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_bare_soname_is_absent_only_when_the_loader_found_nothing() {
    let bare = Path::new("libamdhip64.so.7");
    assert!(!candidate_present(
        bare,
        "libamdhip64.so.7: cannot open shared object file: No such file or directory"
    ));
    assert!(candidate_present(
        bare,
        "/opt/rocm/lib/libamdhip64.so.7: file too short"
    ));
}

#[test]
fn a_bare_soname_with_a_missing_dependency_is_present() {
    // Review round 5 of t4a-gpu-backend#19: glibc reports a missing *dependency* with the
    // same "No such file or directory" phrase, but naming the dependency. The candidate
    // was found, so this is a broken install, not an absent one.
    let bare = Path::new("libamdhip64.so.7");
    assert!(candidate_present(
        bare,
        "libstub.so: cannot open shared object file: No such file or directory"
    ));
    assert!(candidate_present(
        bare,
        "libamdhip64.so.7.2: cannot open shared object file: No such file or directory"
    ));
}
