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

    let mut changed = request.clone();
    changed.options.push("--use_fast_math".into());
    cache.get_or_compile(&changed).unwrap();
    assert_eq!(cache.misses(), 2);
    assert_eq!(cache.len(), 2);
}
