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

// Bounded persistent directory (tensor4all/tenferro-rs#2475). These tests write fake cache
// files directly, so they need neither NVRTC nor a device.

/// Publish a fake image named like a real entry, last used `age_secs` ago.
fn plant(dir: &std::path::Path, label: &str, bytes: usize, age_secs: u64) -> std::path::PathBuf {
    std::fs::create_dir_all(dir).expect("directory");
    let path = dir.join(format!("v1-{label}-{:032x}.ptx", label.len()));
    std::fs::write(&path, vec![b'x'; bytes]).expect("image");
    let used = std::time::SystemTime::now() - std::time::Duration::from_secs(age_secs);
    std::fs::File::options()
        .write(true)
        .open(&path)
        .and_then(|file| file.set_modified(used))
        .expect("mtime");
    path
}

#[test]
fn a_persistent_directory_has_a_finite_default_bound() {
    let cache = ArtifactCache::new(4, 1 << 20).with_dir("/nonexistent/t4a-artifacts");
    assert_eq!(cache.dir_max_bytes(), DEFAULT_ARTIFACT_DIR_BYTES);
    assert_eq!(DEFAULT_ARTIFACT_DIR_BYTES, 256 << 20);
    let cache = cache.with_dir_max_bytes(1 << 10);
    assert_eq!(
        cache.dir_max_bytes(),
        1 << 10,
        "the caller overrides the bound"
    );
}

#[test]
fn introspection_reports_entries_and_bytes_of_cache_files_only() {
    let scratch = ScratchDir::new("stats");
    plant(scratch.path(), "a", 100, 30);
    plant(scratch.path(), "bb", 250, 20);
    // Not ours: a file the cache did not name is neither counted nor touched.
    std::fs::write(scratch.path().join("notes.txt"), b"keep me").expect("foreign file");

    let stats = artifact_dir_stats(scratch.path()).expect("stats");
    assert_eq!((stats.entries, stats.bytes), (2, 350));

    let cache = ArtifactCache::new(4, 1 << 20).with_dir(scratch.path());
    let stats = cache.dir_stats().expect("stats");
    assert_eq!((stats.entries, stats.bytes), (2, 350));

    // A directory that does not exist yet holds nothing; it is not an error.
    let missing = artifact_dir_stats(&scratch.path().join("absent")).expect("missing dir");
    assert_eq!((missing.entries, missing.bytes), (0, 0));
}

#[test]
fn crossing_the_bound_drops_the_least_recently_used_entries_first() {
    let scratch = ScratchDir::new("trim");
    let oldest = plant(scratch.path(), "a", 100, 300);
    let middle = plant(scratch.path(), "bb", 100, 200);
    let newest = plant(scratch.path(), "ccc", 100, 100);

    let evicted = trim_artifact_dir(scratch.path(), 250).expect("trim");
    assert_eq!(
        (evicted.entries, evicted.bytes),
        (1, 100),
        "one entry over the bound"
    );
    assert!(!oldest.exists(), "the least recently used entry goes first");
    assert!(middle.exists() && newest.exists());

    let stats = artifact_dir_stats(scratch.path()).expect("stats");
    assert_eq!((stats.entries, stats.bytes), (2, 200));
    assert!(stats.bytes <= 250);
}

#[test]
fn lowering_the_bound_trims_now_and_counts_what_it_dropped() {
    let scratch = ScratchDir::new("set-bound");
    plant(scratch.path(), "a", 100, 300);
    plant(scratch.path(), "bb", 100, 200);
    let kept = plant(scratch.path(), "ccc", 100, 100);

    let mut cache = ArtifactCache::new(4, 1 << 20).with_dir(scratch.path());
    assert_eq!(cache.dir_evictions(), 0);
    cache.set_dir_max_bytes(150);
    assert_eq!(
        cache.dir_evictions(),
        2,
        "evictions are counted, not silent"
    );
    assert_eq!(cache.dir_evicted_bytes(), 200);
    assert!(kept.exists(), "the most recently used entry survives");
    let stats = cache.dir_stats().expect("stats");
    assert_eq!((stats.entries, stats.bytes), (1, 100));
}

#[test]
fn a_trim_that_fails_part_way_still_counts_what_it_removed() {
    // The shape of a sticky shared directory: the older image is ours and goes, the newer
    // one belongs to another user and its removal is refused.
    let scratch = ScratchDir::new("partial-trim");
    let ours = plant(scratch.path(), "a", 100, 200);
    let theirs = plant(scratch.path(), "bb", 100, 100);

    let mut cache = ArtifactCache::new(4, 1 << 20).with_dir(scratch.path());
    cache.dir_max_bytes = 0;
    cache.trim_dir_with(None, |path| {
        if path == theirs.as_path() {
            Err(std::io::Error::from(std::io::ErrorKind::PermissionDenied))
        } else {
            dir::remove(path)
        }
    });

    assert!(!ours.exists(), "the removable image went");
    assert!(theirs.exists(), "the refused image stayed");
    assert_eq!(
        (cache.dir_evictions(), cache.dir_evicted_bytes()),
        (1, 100),
        "the image removed before the failure is counted"
    );
}

#[test]
fn clear_empties_the_directory_of_cache_files_and_leaves_the_rest() {
    let scratch = ScratchDir::new("clear");
    plant(scratch.path(), "a", 100, 30);
    plant(scratch.path(), "bb", 250, 20);
    // A staging file a crashed writer left behind is ours too.
    std::fs::write(scratch.path().join(".v1-a-0.ptx.4242.tmp"), b"partial").expect("staging");
    std::fs::write(scratch.path().join("notes.txt"), b"keep me").expect("foreign file");

    let mut cache = ArtifactCache::new(4, 1 << 20).with_dir(scratch.path());
    let removed = cache.clear_dir().expect("clear");
    assert_eq!((removed.entries, removed.bytes), (2, 350));

    let stats = cache.dir_stats().expect("stats");
    assert_eq!((stats.entries, stats.bytes), (0, 0));
    assert!(!scratch.path().join(".v1-a-0.ptx.4242.tmp").exists());
    assert!(
        scratch.path().join("notes.txt").exists(),
        "a foreign file is never removed"
    );

    // The free function is the same operation for a caller without a cache.
    plant(scratch.path(), "c", 10, 1);
    let removed = clear_artifact_dir(scratch.path()).expect("clear");
    assert_eq!((removed.entries, removed.bytes), (1, 10));
    assert_eq!(
        artifact_dir_stats(scratch.path()).expect("stats").entries,
        0
    );
}

#[test]
fn a_cache_without_a_directory_reports_an_empty_directory() {
    let mut cache = ArtifactCache::new(4, 1 << 20);
    let stats = cache.dir_stats().expect("stats");
    assert_eq!((stats.entries, stats.bytes), (0, 0));
    let removed = cache.clear_dir().expect("clear");
    assert_eq!((removed.entries, removed.bytes), (0, 0));
}

#[cfg(feature = "gpu-tests")]
#[test]
fn a_publish_over_the_bound_evicts_the_least_recently_used_image() {
    let scratch = ScratchDir::new("publish-trim");
    let stale = plant(scratch.path(), "stale", 4096, 3600);
    let request = request();
    let image_bytes = compile(&request).expect("compile").image().len() as u64;

    // Room for the new image but not for both.
    let mut cache = ArtifactCache::new(4, 1 << 20)
        .with_dir(scratch.path())
        .with_dir_max_bytes(image_bytes + 1);
    cache.get_or_compile(&request).expect("compile");
    assert_eq!(cache.disk_writes(), 1);
    assert!(!stale.exists(), "the older image made room");
    assert_eq!(cache.dir_evictions(), 1);
    assert_eq!(cache.dir_evicted_bytes(), 4096);
    let stats = cache.dir_stats().expect("stats");
    assert_eq!((stats.entries, stats.bytes), (1, image_bytes));

    // An image larger than the whole bound is served but not published.
    let mut tiny = ArtifactCache::new(4, 1 << 20)
        .with_dir(scratch.path().join("tiny"))
        .with_dir_max_bytes(1);
    tiny.get_or_compile(&request).expect("compile");
    assert_eq!(tiny.disk_writes(), 0);
}
