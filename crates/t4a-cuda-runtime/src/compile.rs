//! NVRTC compilation into immutable, context-free artifacts.
//!
//! Compilation is independent of any runtime: an artifact can be prepared before
//! an execution owner exists and bound to every runtime that needs it. Binding is
//! the runtime-specific step.

use std::collections::hash_map::DefaultHasher;
use std::ffi::CString;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use cudarc::nvrtc::result as nvrtc;
use cudarc::nvrtc::sys as nvrtc_sys;

use crate::error::{Error, Result};

/// Output kind requested from NVRTC.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OutputKind {
    /// PTX; the driver JITs it when the module is loaded.
    Ptx,
    /// CUBIN for the exact target architecture; no driver JIT on load.
    Cubin,
}

/// Everything that determines the compiled image.
///
/// Two requests are interchangeable only when all fields are equal. The compiler
/// version observed at compile time is added to the [`Artifact`] and is therefore
/// part of the cache identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompileRequest {
    /// Stable identity of the generator or source revision (for diagnostics).
    pub source_id: String,
    /// CUDA C++ source.
    pub source: String,
    /// Header name and contents, prepended in order before `source`.
    pub headers: Vec<(String, String)>,
    /// Extra NVRTC options.
    pub options: Vec<String>,
    /// Target architecture such as `sm_80`; required for [`OutputKind::Cubin`].
    pub arch: Option<String>,
    /// Requested output kind.
    pub output: OutputKind,
}

impl CompileRequest {
    /// Build a request with no headers or extra options.
    pub fn new(source_id: impl Into<String>, source: impl Into<String>) -> Self {
        Self {
            source_id: source_id.into(),
            source: source.into(),
            headers: Vec::new(),
            options: Vec::new(),
            arch: None,
            output: OutputKind::Ptx,
        }
    }

    /// Cache key derived from the actual inputs, not from a caller-supplied label.
    pub fn key(&self) -> CompileKey {
        let mut hasher = DefaultHasher::new();
        self.source_id.hash(&mut hasher);
        self.source.hash(&mut hasher);
        for (name, body) in &self.headers {
            name.hash(&mut hasher);
            body.hash(&mut hasher);
        }
        for option in &self.options {
            option.hash(&mut hasher);
        }
        self.arch.hash(&mut hasher);
        (self.output as u8).hash(&mut hasher);
        CompileKey {
            hash: hasher.finish(),
        }
    }

    fn program_source(&self) -> String {
        let mut program = String::new();
        for (name, body) in &self.headers {
            program.push_str("// header: ");
            program.push_str(name);
            program.push('\n');
            program.push_str(body);
            program.push('\n');
        }
        program.push_str("#line 1 \"kernel\"\n");
        program.push_str(&self.source);
        program
    }

    fn validate(&self) -> Result<()> {
        if self.source.trim().is_empty() {
            return Err(Error::invalid("compile source must not be empty"));
        }
        if self.source.contains('\0') || self.source_id.contains('\0') {
            return Err(Error::invalid("compile source contains a NUL byte"));
        }
        for (name, body) in &self.headers {
            if name.contains('\0') || body.contains('\0') {
                return Err(Error::invalid("header contains a NUL byte"));
            }
            if name.contains('\n') || name.contains('\r') {
                return Err(Error::invalid("header name must be a single line"));
            }
        }
        for option in &self.options {
            if option.contains('\0') {
                return Err(Error::invalid("compile option contains a NUL byte"));
            }
        }
        if self.arch.as_deref().is_some_and(|arch| arch.contains('\0')) {
            return Err(Error::invalid("target architecture contains a NUL byte"));
        }
        if self.output == OutputKind::Cubin && self.arch.is_none() {
            return Err(Error::invalid(
                "cubin output requires an explicit target architecture",
            ));
        }
        Ok(())
    }
}

/// Hash-based cache key with exact post-lookup verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CompileKey {
    hash: u64,
}

/// A compiled, context-free kernel image.
pub struct Artifact {
    key: CompileKey,
    request: CompileRequest,
    image: Vec<u8>,
    compiler_version: String,
}

impl Artifact {
    /// Cache key.
    pub fn key(&self) -> CompileKey {
        self.key
    }

    /// The request that produced this image.
    pub fn request(&self) -> &CompileRequest {
        &self.request
    }

    /// Bytes this artifact retains: the image plus the request text and headers.
    pub fn retained_bytes(&self) -> usize {
        let headers: usize = self
            .request
            .headers
            .iter()
            .map(|(name, body)| name.len() + body.len())
            .sum();
        let options: usize = self.request.options.iter().map(String::len).sum();
        self.image.len()
            + self.request.source.len()
            + self.request.source_id.len()
            + self.request.arch.as_ref().map(String::len).unwrap_or(0)
            + headers
            + options
    }

    /// Raw PTX or CUBIN bytes.
    pub fn image(&self) -> &[u8] {
        &self.image
    }

    /// Output kind of [`Artifact::image`].
    pub fn output(&self) -> OutputKind {
        self.request.output
    }

    /// NVRTC version observed when the image was produced.
    pub fn compiler_version(&self) -> &str {
        &self.compiler_version
    }
}

impl std::fmt::Debug for Artifact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Artifact")
            .field("key", &self.key)
            .field("source_id", &self.request.source_id)
            .field("output", &self.request.output)
            .field("bytes", &self.image.len())
            .field("compiler_version", &self.compiler_version)
            .finish()
    }
}

impl PartialEq for Artifact {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
            && self.request == other.request
            && self.compiler_version == other.compiler_version
            && self.image == other.image
    }
}

const REQUIRED_NVRTC_SYMBOLS: &[&[u8]] = &[
    b"nvrtcVersion\0",
    b"nvrtcCreateProgram\0",
    b"nvrtcCompileProgram\0",
    b"nvrtcGetProgramLogSize\0",
    b"nvrtcGetProgramLog\0",
    b"nvrtcGetPTXSize\0",
    b"nvrtcGetPTX\0",
    b"nvrtcGetCUBINSize\0",
    b"nvrtcGetCUBIN\0",
    b"nvrtcDestroyProgram\0",
];

fn probe_nvrtc() -> Result<()> {
    // SAFETY: probing the dynamic loader only; it does not initialise NVRTC.
    if !unsafe { nvrtc_sys::is_culib_present() } {
        return Err(Error::NvrtcUnavailable {
            detail: "libnvrtc could not be loaded".to_string(),
        });
    }
    // SAFETY: the library was just loaded successfully.
    let lib = unsafe { nvrtc_sys::culib() };
    for symbol in REQUIRED_NVRTC_SYMBOLS {
        // SAFETY: only resolving the symbol; the value is dropped immediately.
        if unsafe { lib.get::<*const ()>(*symbol) }.is_err() {
            return Err(Error::NvrtcUnavailable {
                detail: format!(
                    "nvrtc symbol {} is missing",
                    String::from_utf8_lossy(&symbol[..symbol.len() - 1])
                ),
            });
        }
    }
    Ok(())
}

/// Observed NVRTC version, used in cache identity.
pub fn nvrtc_version() -> Result<String> {
    probe_nvrtc()?;
    let mut major = 0i32;
    let mut minor = 0i32;
    // SAFETY: both out-pointers are valid for the duration of the call.
    let status = unsafe { nvrtc_sys::nvrtcVersion(&mut major, &mut minor) };
    if !matches!(status, nvrtc_sys::nvrtcResult::NVRTC_SUCCESS) {
        return Err(Error::NvrtcUnavailable {
            detail: format!("nvrtcVersion returned {status:?}"),
        });
    }
    Ok(format!("{major}.{minor}"))
}

/// Compile a request into an immutable artifact.
///
/// # Errors
///
/// [`Error::NvrtcUnavailable`] when NVRTC cannot be loaded, [`Error::Compile`]
/// with the compiler diagnostics when the program does not compile, and
/// [`Error::InvalidArgument`] for an invalid request.
pub fn compile(request: &CompileRequest) -> Result<Artifact> {
    let mut attempts = 0;
    let mut successes = 0;
    compile_counted(request, &mut attempts, &mut successes)
}

/// Compile, reporting the actual NVRTC call boundary to the caller's counters.
pub(crate) fn compile_counted(
    request: &CompileRequest,
    attempts: &mut u64,
    successes: &mut u64,
) -> Result<Artifact> {
    request.validate()?;
    let compiler_version = nvrtc_version()?;
    let program_source = request.program_source();
    let source = CString::new(program_source)
        .map_err(|_| Error::invalid("kernel source contains an interior NUL byte"))?;
    let name = CString::new(request.source_id.as_str())
        .map_err(|_| Error::invalid("source_id contains an interior NUL byte"))?;
    let program = nvrtc::create_program(&source, Some(&name)).map_err(nvrtc_error)?;
    // One cleanup guard for every exit path, including extraction failures.
    let program = ProgramGuard(program);
    let program = program.0;
    let mut options: Vec<String> = request.options.clone();
    if let Some(arch) = &request.arch {
        options.push(format!("--gpu-architecture={arch}"));
    }
    *attempts += 1;
    // SAFETY: `program` was created above and is destroyed by `ProgramGuard`.
    let compiled = unsafe { nvrtc::compile_program(program, &options) };
    if let Err(error) = compiled {
        // SAFETY: the program was created above and not destroyed yet.
        let log = unsafe { nvrtc::get_program_log(program) }.unwrap_or_default();
        let log: Vec<u8> = log.into_iter().map(|byte| byte as u8).collect();
        let detail = String::from_utf8_lossy(&log).into_owned();
        let detail = if detail.trim().is_empty() {
            format!("NVRTC returned {:?} without diagnostics", error.0)
        } else {
            detail
        };
        return Err(Error::Compile { detail });
    }
    // The compiler call itself succeeded; extraction failures below must not
    // turn that into "no successful compilation".
    *successes += 1;
    let image = match request.output {
        OutputKind::Ptx => {
            // SAFETY: the program compiled successfully.
            unsafe { nvrtc::get_ptx(program) }
                .map_err(nvrtc_error)?
                .into_iter()
                .map(|byte| byte as u8)
                .collect::<Vec<u8>>()
        }
        OutputKind::Cubin => cubin_bytes(program)?,
    };
    Ok(Artifact {
        key: request.key(),
        request: request.clone(),
        image,
        compiler_version,
    })
}

/// Destroys the NVRTC program on every exit path.
struct ProgramGuard(nvrtc_sys::nvrtcProgram);

impl Drop for ProgramGuard {
    fn drop(&mut self) {
        // SAFETY: the guard owns the program handle and destroys it once.
        unsafe {
            let _ = nvrtc::destroy_program(self.0);
        }
    }
}

fn cubin_bytes(program: nvrtc_sys::nvrtcProgram) -> Result<Vec<u8>> {
    let mut size = 0usize;
    // SAFETY: the program compiled successfully and `size` is a valid out-pointer.
    let status = unsafe { nvrtc_sys::nvrtcGetCUBINSize(program, &mut size) };
    if !matches!(status, nvrtc_sys::nvrtcResult::NVRTC_SUCCESS) {
        return Err(Error::Compile {
            detail: format!("nvrtcGetCUBINSize returned {status:?}"),
        });
    }
    if size == 0 {
        return Err(Error::Compile {
            detail: "NVRTC produced an empty CUBIN (is the target a virtual architecture?)"
                .to_string(),
        });
    }
    let mut bytes = vec![0u8; size];
    // SAFETY: `bytes` holds `size` writable bytes.
    let status = unsafe { nvrtc_sys::nvrtcGetCUBIN(program, bytes.as_mut_ptr().cast()) };
    if !matches!(status, nvrtc_sys::nvrtcResult::NVRTC_SUCCESS) {
        return Err(Error::Compile {
            detail: format!("nvrtcGetCUBIN returned {status:?}"),
        });
    }
    Ok(bytes)
}

fn nvrtc_error(error: cudarc::nvrtc::result::NvrtcError) -> Error {
    Error::Compile {
        detail: error.to_string(),
    }
}

/// A bounded, caller-owned artifact cache.
///
/// The runtime uses the same type and the same [`compile`] operation, so a
/// preparation stage that runs before any runtime exists does not need a second
/// compilation path.
#[derive(Debug)]
pub struct ArtifactCache {
    entries: Vec<std::sync::Arc<Artifact>>,
    bytes: usize,
    max_entries: usize,
    max_bytes: usize,
    hits: u64,
    misses: u64,
    attempts: u64,
    successes: u64,
    dir: Option<PathBuf>,
    disk_hits: u64,
    disk_writes: u64,
}

impl ArtifactCache {
    /// Create a cache with the given bounds.
    pub fn new(max_entries: usize, max_bytes: usize) -> Self {
        Self {
            entries: Vec::new(),
            bytes: 0,
            max_entries,
            max_bytes,
            hits: 0,
            misses: 0,
            attempts: 0,
            successes: 0,
            dir: None,
            disk_hits: 0,
            disk_writes: 0,
        }
    }

    /// Also persist compiled images in `dir`, so a later process reads them instead of
    /// recompiling.
    ///
    /// The directory holds one file per translation unit, named
    /// `v1-<source_id>-<128-bit digest>.<ptx|cubin>`. The digest covers everything that
    /// determines the image (`source_id`, source, headers, options, architecture, output
    /// kind and the NVRTC version), so editing one module's source invalidates that module
    /// only; nothing global is in the key. Files are published by writing a temporary file
    /// and renaming it, so a concurrent reader never sees a partial image, and a directory
    /// that cannot be read or written is simply a cache miss. Removing stale files is the
    /// caller's to do: the directory is content-addressed, so an old entry is dead weight,
    /// not a hazard.
    pub fn with_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.dir = Some(dir.into());
        self
    }

    /// The persistent directory this cache reads and writes, if any.
    pub fn dir(&self) -> Option<&Path> {
        self.dir.as_deref()
    }

    /// Images served from the persistent directory instead of NVRTC.
    pub fn disk_hits(&self) -> u64 {
        self.disk_hits
    }

    /// Images published to the persistent directory.
    pub fn disk_writes(&self) -> u64 {
        self.disk_writes
    }

    /// Look up `request`, compiling and inserting it on a miss.
    pub fn get_or_compile(&mut self, request: &CompileRequest) -> Result<std::sync::Arc<Artifact>> {
        let key = request.key();
        if let Some(index) = self
            .entries
            .iter()
            .position(|entry| entry.key == key && &entry.request == request)
        {
            self.hits += 1;
            return Ok(std::sync::Arc::clone(&self.entries[index]));
        }
        self.misses += 1;
        // A persistent image needs no NVRTC: this is the whole point of the directory.
        if let Some(artifact) = self.read_dir(request, key) {
            self.disk_hits += 1;
            let artifact = std::sync::Arc::new(artifact);
            self.insert(std::sync::Arc::clone(&artifact));
            return Ok(artifact);
        }
        let mut attempts = 0;
        let mut successes = 0;
        let compiled = compile_counted(request, &mut attempts, &mut successes);
        self.attempts += attempts;
        self.successes += successes;
        let compiled = compiled?;
        self.write_dir(request, &compiled);
        let artifact = std::sync::Arc::new(compiled);
        self.insert(std::sync::Arc::clone(&artifact));
        Ok(artifact)
    }

    /// The image `request` produced earlier, read back from the persistent directory.
    ///
    /// A missing, unreadable or empty entry is a miss; a cache never fails a build.
    fn read_dir(&mut self, request: &CompileRequest, key: CompileKey) -> Option<Artifact> {
        let dir = self.dir.clone()?;
        let path = dir.join(persistent_name(request)?);
        let image = std::fs::read(&path).ok()?;
        if image.is_empty() {
            return None;
        }
        Some(Artifact {
            key,
            request: request.clone(),
            image,
            compiler_version: nvrtc_version().unwrap_or_default(),
        })
    }

    /// Publish `artifact` into the persistent directory, best effort.
    fn write_dir(&mut self, request: &CompileRequest, artifact: &Artifact) {
        let Some(dir) = self.dir.clone() else {
            return;
        };
        let Some(name) = persistent_name(request) else {
            return;
        };
        let staged = dir.join(format!(".{name}.{}.tmp", std::process::id()));
        let published = std::fs::create_dir_all(&dir)
            .and_then(|()| std::fs::write(&staged, artifact.image()))
            // Atomic within one filesystem: a reader sees the old entry or the whole image.
            .and_then(|()| std::fs::rename(&staged, dir.join(&name)));
        if published.is_ok() {
            self.disk_writes += 1;
        } else {
            let _ = std::fs::remove_file(&staged);
        }
    }

    fn insert(&mut self, artifact: std::sync::Arc<Artifact>) {
        // A zero-capacity cache keeps nothing, and an image larger than the whole
        // byte budget is not retained, so eviction never removes an absent entry.
        let retained = artifact.retained_bytes();
        if self.max_entries == 0 || retained > self.max_bytes {
            return;
        }
        while self.entries.len() >= self.max_entries
            || (self.bytes + retained > self.max_bytes && !self.entries.is_empty())
        {
            if self.entries.is_empty() {
                return;
            }
            let evicted = self.entries.remove(0);
            self.bytes = self.bytes.saturating_sub(evicted.retained_bytes());
        }
        self.bytes += retained;
        self.entries.push(artifact);
    }

    /// Number of cached artifacts.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Estimated payload bytes retained.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Actual NVRTC compilation attempts made by this cache.
    pub fn compile_attempts(&self) -> u64 {
        self.attempts
    }

    /// Actual NVRTC compilations that succeeded.
    pub fn compile_successes(&self) -> u64 {
        self.successes
    }

    /// Cache hits since creation.
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// Cache misses since creation (each one triggered a compile attempt).
    pub fn misses(&self) -> u64 {
        self.misses
    }

    /// Drop every artifact this process holds; a persistent directory is left alone.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.bytes = 0;
    }
}

/// The persistent file name for `request`, or `None` when NVRTC is unavailable.
///
/// The name is `v1-<source-id>-<128-bit digest>.<ptx|cubin>`: the digest is the identity, the
/// rest is for a human reading the directory. `v1-` changes only if the *stored form* changes,
/// never because a kernel changed.
fn persistent_name(request: &CompileRequest) -> Option<String> {
    let compiler_version = nvrtc_version().ok()?;
    let digest = persistent_digest(request, &compiler_version);
    let label: String = request
        .source_id
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '_' {
                character
            } else {
                '_'
            }
        })
        .take(48)
        .collect();
    let extension = match request.output {
        OutputKind::Ptx => "ptx",
        OutputKind::Cubin => "cubin",
    };
    Some(format!("v1-{label}-{digest}.{extension}"))
}

/// 128-bit digest of everything that determines the image.
///
/// `DefaultHasher` is deliberately not reused: its algorithm is unspecified and may change
/// between Rust releases, and a persistent key has to mean the same thing to a later process.
/// Two FNV-1a lanes over a length-prefixed serialization, finished with a splitmix64 avalanche,
/// keep the key stable and collision-free for practical purposes without a crypto dependency.
fn persistent_digest(request: &CompileRequest, compiler_version: &str) -> String {
    const OFFSETS: [u64; 2] = [0xcbf2_9ce4_8422_2325, 0x9e37_79b9_7f4a_7c15];
    const PRIMES: [u64; 2] = [0x0000_0100_0000_01b3, 0x0000_0100_0000_01b5];
    let mut state = OFFSETS;
    {
        let mut feed = |bytes: &[u8]| {
            for (index, lane) in state.iter_mut().enumerate() {
                // The length prefix is fed by the caller, so no two field lists can
                // concatenate into the same stream.
                let mut mix = |byte: u8| {
                    *lane = (*lane ^ byte as u64).wrapping_mul(PRIMES[index]);
                };
                mix(index as u8);
                for &byte in bytes {
                    mix(byte);
                }
            }
        };
        let mut field = |bytes: &[u8]| {
            feed(&(bytes.len() as u64).to_le_bytes());
            feed(bytes);
        };
        field(request.source_id.as_bytes());
        field(request.source.as_bytes());
        field(&(request.headers.len() as u64).to_le_bytes());
        for (name, body) in &request.headers {
            field(name.as_bytes());
            field(body.as_bytes());
        }
        field(&(request.options.len() as u64).to_le_bytes());
        for option in &request.options {
            field(option.as_bytes());
        }
        match &request.arch {
            Some(arch) => {
                field(b"arch");
                field(arch.as_bytes());
            }
            None => field(b"no-arch"),
        }
        field(&[request.output as u8]);
        field(compiler_version.as_bytes());
    }
    let avalache = |mut value: u64| {
        value ^= value >> 30;
        value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value ^= value >> 27;
        value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    };
    format!("{:016x}{:016x}", avalache(state[0]), avalache(state[1]))
}

#[cfg(test)]
mod tests;
