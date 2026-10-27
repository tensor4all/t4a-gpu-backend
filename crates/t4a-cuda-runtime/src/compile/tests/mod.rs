use super::*;

fn request() -> CompileRequest {
    CompileRequest::new("test-kernel", "extern \"C\" __global__ void k() {}")
}

#[test]
fn key_changes_with_every_compilation_input() {
    let base = request().key();

    let mut changed = request();
    changed.source.push(' ');
    assert_ne!(base, changed.key());

    let mut changed = request();
    changed.headers.push(("a.h".into(), "#define A 1".into()));
    assert_ne!(base, changed.key());

    let mut changed = request();
    changed.options.push("--use_fast_math".into());
    assert_ne!(base, changed.key());

    let mut changed = request();
    changed.arch = Some("sm_80".into());
    assert_ne!(base, changed.key());

    let mut changed = request();
    changed.output = OutputKind::Cubin;
    assert_ne!(base, changed.key());

    let mut changed = request();
    changed.source_id = "other".into();
    assert_ne!(base, changed.key());
}

#[test]
fn cubin_requires_a_target_architecture() {
    let mut request = request();
    request.output = OutputKind::Cubin;
    assert!(request.validate().is_err());
    request.arch = Some("sm_80".into());
    assert!(request.validate().is_ok());
}

#[test]
fn empty_source_is_rejected() {
    let request = CompileRequest::new("empty", "   \n");
    assert!(matches!(
        request.validate(),
        Err(Error::InvalidArgument { .. })
    ));
}

#[test]
fn headers_precede_the_source() {
    let mut request = request();
    request
        .headers
        .push(("h.h".into(), "#define VALUE 3".into()));
    let program = request.program_source();
    let header = program.find("#define VALUE 3").unwrap();
    let source = program.find("__global__").unwrap();
    assert!(header < source, "headers must be emitted before the source");
}

#[cfg(feature = "gpu-tests")]
#[test]
fn compiles_ptx_and_reports_diagnostics() {
    let artifact = compile(&request()).expect("compile must succeed");
    assert_eq!(artifact.output(), OutputKind::Ptx);
    assert!(!artifact.image().is_empty());
    assert!(artifact.image().starts_with(b"//\n") || artifact.image().starts_with(b".version"));

    let mut broken = request();
    broken.source = "this is not cuda".into();
    match compile(&broken) {
        Err(Error::Compile { detail }) => assert!(!detail.trim().is_empty()),
        other => panic!("expected a compile diagnostic, got {other:?}"),
    }
}

#[cfg(feature = "gpu-tests")]
#[test]
fn cache_counts_real_compile_attempts() {
    let mut cache = ArtifactCache::new(4, 1 << 20);
    let request = request();
    cache.get_or_compile(&request).unwrap();
    cache.get_or_compile(&request).unwrap();
    assert_eq!(cache.misses(), 1);
    assert_eq!(cache.hits(), 1);
    assert_eq!(cache.len(), 1);
    // Warm use performs no NVRTC work; the counters must show exactly one
    // attempt and one success so far.
    assert_eq!(cache.compile_attempts(), 1);
    assert_eq!(cache.compile_successes(), 1);

    let mut changed = request.clone();
    changed.options.push("--use_fast_math".into());
    cache.get_or_compile(&changed).unwrap();
    assert_eq!(cache.misses(), 2);
    assert_eq!(cache.len(), 2);
    assert_eq!(cache.compile_attempts(), 2);
    assert_eq!(cache.compile_successes(), 2);
}

/// A directory that only this test uses, removed on drop.
struct ScratchDir(std::path::PathBuf);

impl ScratchDir {
    fn new(label: &str) -> Self {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let unique = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "t4a-artifacts-{}-{label}-{unique}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        Self(path)
    }

    fn path(&self) -> &std::path::Path {
        &self.0
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn the_persistent_digest_covers_every_compilation_input() {
    let base = persistent_digest(&request(), "12.6");

    let mut changed = request();
    changed.source.push(' ');
    assert_ne!(base, persistent_digest(&changed, "12.6"));

    let mut changed = request();
    changed.source_id = "other".into();
    assert_ne!(base, persistent_digest(&changed, "12.6"));

    let mut changed = request();
    changed.headers.push(("h.h".into(), "#define X 1".into()));
    assert_ne!(base, persistent_digest(&changed, "12.6"));

    let mut changed = request();
    changed.options.push("--use_fast_math".into());
    assert_ne!(base, persistent_digest(&changed, "12.6"));

    let mut changed = request();
    changed.arch = Some("sm_90".into());
    assert_ne!(base, persistent_digest(&changed, "12.6"));

    let mut changed = request();
    changed.output = OutputKind::Cubin;
    assert_ne!(base, persistent_digest(&changed, "12.6"));

    // A different compiler version must not serve an image it did not produce.
    assert_ne!(base, persistent_digest(&request(), "12.8"));
}

#[cfg(feature = "gpu-tests")]
#[test]
fn a_persistent_image_serves_a_second_cache_without_compiling() {
    let scratch = ScratchDir::new("second-cache");
    let request = request();

    let mut first = ArtifactCache::new(4, 1 << 20).with_dir(scratch.path());
    let compiled = first.get_or_compile(&request).expect("cold compile");
    assert_eq!(first.compile_attempts(), 1);
    assert_eq!(first.disk_writes(), 1);
    drop(first);

    let names: Vec<_> = std::fs::read_dir(scratch.path())
        .expect("directory")
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.file_name())
        .collect();
    assert_eq!(names.len(), 1, "one file per translation unit: {names:?}");
    assert!(names[0].to_string_lossy().starts_with("v1-test_kernel-"));

    let mut second = ArtifactCache::new(4, 1 << 20).with_dir(scratch.path());
    let served = second.get_or_compile(&request).expect("warm read");
    assert_eq!(second.compile_attempts(), 0, "no NVRTC on a persistent hit");
    assert_eq!(second.disk_hits(), 1);
    assert_eq!(served.image(), compiled.image());

    // A different option is a different image, so it compiles rather than reusing this one.
    let mut changed = request.clone();
    changed.options.push("--use_fast_math".into());
    let mut third = ArtifactCache::new(4, 1 << 20).with_dir(scratch.path());
    third.get_or_compile(&changed).expect("recompile");
    assert_eq!(
        third.disk_hits(),
        0,
        "a different option is a different key"
    );
    assert_eq!(
        third.compile_attempts(),
        1,
        "a key that is not on disk compiles rather than reusing another image"
    );
}

#[cfg(feature = "gpu-tests")]
#[test]
fn an_empty_persistent_entry_is_recompiled() {
    let scratch = ScratchDir::new("empty-entry");
    let request = request();
    std::fs::create_dir_all(scratch.path()).expect("directory");
    let name = persistent_name(&request).expect("name");
    std::fs::write(scratch.path().join(&name), b"").expect("truncate");

    let mut cache = ArtifactCache::new(4, 1 << 20).with_dir(scratch.path());
    let artifact = cache.get_or_compile(&request).expect("compile");
    assert_eq!(cache.compile_attempts(), 1, "an empty file is a miss");
    let repaired = std::fs::read(scratch.path().join(&name)).expect("repaired");
    assert_eq!(repaired, artifact.image(), "the miss repairs the entry");
}

#[cfg(feature = "gpu-tests")]
#[test]
fn concurrent_writers_publish_one_complete_image() {
    let scratch = ScratchDir::new("concurrent");
    let request = request();
    let writers: Vec<_> = (0..2)
        .map(|_| {
            let dir = scratch.path().to_path_buf();
            let request = request.clone();
            std::thread::spawn(move || {
                let mut cache = ArtifactCache::new(4, 1 << 20).with_dir(dir);
                let artifact = cache.get_or_compile(&request).expect("compile");
                artifact.image().to_vec()
            })
        })
        .collect();
    let images: Vec<Vec<u8>> = writers
        .into_iter()
        .map(|writer| writer.join().expect("writer"))
        .collect();
    assert_eq!(images[0], images[1], "both writers produce the same image");

    let name = persistent_name(&request).expect("name");
    let stored = std::fs::read(scratch.path().join(&name)).expect("published");
    assert_eq!(
        stored, images[0],
        "the published file is one complete image"
    );
    let leftovers: Vec<_> = std::fs::read_dir(scratch.path())
        .expect("directory")
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.file_name())
        .filter(|name| name.to_string_lossy().contains(".tmp"))
        .collect();
    assert!(
        leftovers.is_empty(),
        "no staging files remain: {leftovers:?}"
    );
}

#[test]
fn a_cache_without_a_directory_never_touches_the_filesystem() {
    let cache = ArtifactCache::new(4, 1 << 20);
    assert!(cache.dir().is_none());
    assert_eq!(cache.disk_hits(), 0);
    assert_eq!(cache.disk_writes(), 0);
}
