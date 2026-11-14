//! Run-time MSL compilation and the per-runtime artifact cache.
//!
//! Metal compiles MSL source to a library at run time, so an artifact is a live
//! `MTLLibrary` rather than a portable image. Its identity is the SHA-256 of the
//! source, which is also what a report records next to any number measured with
//! that kernel.

use std::sync::Arc;

use sha2::{Digest, Sha256};
use t4a_metal_sys::objc::{error_detail, Id, Pool};

use crate::context::ContextHolder;
use crate::error::{Error, Result};

/// The MSL source hash of a kernel: lowercase hex SHA-256.
///
/// The hash is the cache key and the provenance record. It is not a cryptographic
/// claim about the kernel's behaviour.
///
/// # Examples
///
/// ```
/// let hash = t4a_metal_runtime::source_hash("kernel void k() {}");
/// assert_eq!(hash.len(), 64);
/// assert_ne!(hash, t4a_metal_runtime::source_hash("kernel void k() { }"));
/// ```
pub fn source_hash(source: &str) -> String {
    let digest = Sha256::digest(source.as_bytes());
    let mut hex = String::with_capacity(64);
    use std::fmt::Write as _;
    for byte in digest {
        let _ = write!(hex, "{byte:02x}");
    }
    hex
}

/// One MSL compilation request.
#[derive(Debug, Clone)]
pub struct CompileRequest {
    name: String,
    source: String,
    hash: String,
}

impl CompileRequest {
    /// A request for `source`, labelled `name` for diagnostics and statistics.
    ///
    /// The label is not part of the artifact's identity: two requests with the same
    /// source share one artifact even when their labels differ.
    pub fn new(name: impl Into<String>, source: impl Into<String>) -> Self {
        let source = source.into();
        let hash = source_hash(&source);
        Self {
            name: name.into(),
            source,
            hash,
        }
    }

    /// The label.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The MSL source.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// The source's SHA-256, lowercase hex.
    pub fn source_hash(&self) -> &str {
        &self.hash
    }
}

/// A compiled MSL library.
///
/// Immutable and shareable; it belongs to one runtime's device but carries no
/// submission authority.
pub struct Artifact {
    context: Arc<ContextHolder>,
    request: CompileRequest,
    library: Id,
}

// SAFETY: an `MTLLibrary` is an immutable, thread-safe Metal object, and the handle
// carries no submission authority.
unsafe impl Send for Artifact {}
unsafe impl Sync for Artifact {}

impl Artifact {
    /// The source's SHA-256, lowercase hex.
    pub fn source_hash(&self) -> &str {
        self.request.source_hash()
    }

    /// Retained size estimate: the source text.
    pub fn retained_bytes(&self) -> usize {
        self.request.source.len()
    }

    pub(crate) fn library(&self) -> Id {
        self.library
    }

    pub(crate) fn context(&self) -> &Arc<ContextHolder> {
        &self.context
    }
}

impl Drop for Artifact {
    fn drop(&mut self) {
        // SAFETY: this is the `+1` library reference taken at compile time, released
        // once, when the last artifact handle is gone.
        unsafe { self.context.table.objc.release(self.library) };
    }
}

impl std::fmt::Debug for Artifact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Artifact")
            .field("name", &self.request.name)
            .field("source_hash", &self.request.hash)
            .finish()
    }
}

/// A bounded, runtime-owned artifact cache.
pub(crate) struct ArtifactCache {
    entries: Vec<Arc<Artifact>>,
    bytes: usize,
    max_entries: usize,
    max_bytes: usize,
    hits: u64,
    misses: u64,
}

impl ArtifactCache {
    pub(crate) fn new(max_entries: usize, max_bytes: usize) -> Self {
        Self {
            entries: Vec::new(),
            bytes: 0,
            max_entries,
            max_bytes,
            hits: 0,
            misses: 0,
        }
    }

    pub(crate) fn get_or_compile(
        &mut self,
        context: &Arc<ContextHolder>,
        request: &CompileRequest,
    ) -> Result<Arc<Artifact>> {
        if let Some(cached) = self
            .entries
            .iter()
            .find(|entry| entry.source_hash() == request.source_hash())
        {
            self.hits += 1;
            return Ok(Arc::clone(cached));
        }
        self.misses += 1;
        let library = compile(context, request)?;
        let artifact = Arc::new(Artifact {
            context: Arc::clone(context),
            request: request.clone(),
            library,
        });
        self.bytes += artifact.retained_bytes();
        self.entries.push(Arc::clone(&artifact));
        self.evict();
        Ok(artifact)
    }

    fn evict(&mut self) {
        while self.entries.len() > self.max_entries || self.bytes > self.max_bytes {
            let Some(evicted) = self.entries.first() else {
                break;
            };
            // A caller may still hold a bound function or artifact from this entry;
            // dropping the cache's reference is what eviction means.
            self.bytes = self.bytes.saturating_sub(evicted.retained_bytes());
            self.entries.remove(0);
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }

    pub(crate) fn hits(&self) -> u64 {
        self.hits
    }

    pub(crate) fn misses(&self) -> u64 {
        self.misses
    }

    pub(crate) fn clear(&mut self) {
        self.entries.clear();
        self.bytes = 0;
    }
}

/// Compile `request` into a `+1` `MTLLibrary`.
fn compile(context: &Arc<ContextHolder>, request: &CompileRequest) -> Result<Id> {
    let table = context.table;
    let selectors = table.selectors;
    let pool = Pool::push(table.objc);
    let Some(source) = table.new_string(request.source()) else {
        drop(pool);
        return Err(Error::Compile {
            detail: "the source text could not be turned into an NSString".to_string(),
        });
    };
    let mut error: Id = std::ptr::null_mut();
    // SAFETY: the device, the source string and the error slot are live; nil options
    // selects Metal's defaults. The call returns a `+1` library or nil with `error`
    // set. `error` is read before the pool is popped.
    let library = unsafe {
        selectors.device_new_library.send(
            context.device(),
            source,
            std::ptr::null_mut(),
            &mut error as *mut Id,
        )
    };
    let detail = if library.is_null() {
        // SAFETY: `error` is either null or a live `NSError` inside the open pool.
        unsafe { error_detail(&selectors, error) }
    } else {
        None
    };
    // SAFETY: releases the `+1` string taken above.
    unsafe { table.objc.release(source) };
    drop(pool);
    if !library.is_null() {
        return Ok(library);
    }
    Err(Error::Compile {
        detail: detail.unwrap_or_else(|| "Metal returned no library and no diagnostic".to_string()),
    })
}
