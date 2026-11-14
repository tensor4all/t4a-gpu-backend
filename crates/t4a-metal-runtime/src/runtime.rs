//! The runtime: one device, one queue, shared buffers, checked launches.

use std::sync::Arc;

use t4a_metal_sys::Table;

use crate::compile::{Artifact, ArtifactCache, CompileRequest};
use crate::context::{ContextHolder, DeviceInfo};
use crate::error::{Error, Result};
use crate::kernel::{self, Arg, BoundFunction, LaunchConfig, PipelineState};
use crate::memory::{Buffer, UninitBuffer};
use crate::raw::RawScope;
use crate::submission::{Completion, Session};

/// Default number of submissions that may be in flight before `commit` waits.
pub const DEFAULT_MAX_SUBMISSIONS_IN_FLIGHT: usize = 64;

/// Default bound on the per-runtime MSL artifact cache.
pub const DEFAULT_MAX_ARTIFACT_ENTRIES: usize = 64;

/// Default byte bound on the per-runtime MSL artifact cache, measured in source
/// bytes.
pub const DEFAULT_MAX_ARTIFACT_BYTES: usize = 64 << 20;

/// Default bound on the per-runtime pipeline-state cache.
pub const DEFAULT_MAX_PIPELINE_ENTRIES: usize = 128;

/// Default wait for one submission to complete.
pub const DEFAULT_COMPLETION_TIMEOUT_MS: u64 = 60_000;

/// Runtime configuration.
///
/// The defaults are reviewable policy, not measurements. Every bound exists so that
/// a loop that never waits cannot grow without limit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    /// Submissions that may be in flight before [`Runtime::commit`] waits for the
    /// oldest. Must be non-zero.
    pub max_submissions_in_flight: usize,
    /// How long one wait for a submission lasts before it reports [`Error::Timeout`].
    ///
    /// It bounds every wait this crate makes: [`Completion::wait`], the waits behind
    /// [`Runtime::synchronize`] and [`Runtime::close`], and the ones [`Runtime::drop`]
    /// makes on the way out. Must be non-zero.
    pub completion_timeout_ms: u64,
    /// Artifact-cache entry bound. Must be non-zero.
    pub max_artifact_entries: usize,
    /// Artifact-cache byte bound. Must be non-zero.
    pub max_artifact_bytes: usize,
    /// Pipeline-state cache entry bound. Must be non-zero.
    pub max_pipeline_entries: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            max_submissions_in_flight: DEFAULT_MAX_SUBMISSIONS_IN_FLIGHT,
            completion_timeout_ms: DEFAULT_COMPLETION_TIMEOUT_MS,
            max_artifact_entries: DEFAULT_MAX_ARTIFACT_ENTRIES,
            max_artifact_bytes: DEFAULT_MAX_ARTIFACT_BYTES,
            max_pipeline_entries: DEFAULT_MAX_PIPELINE_ENTRIES,
        }
    }
}

impl Config {
    /// Check the bounds.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] naming the field that is zero.
    pub fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("max_submissions_in_flight", self.max_submissions_in_flight),
            ("max_artifact_entries", self.max_artifact_entries),
            ("max_artifact_bytes", self.max_artifact_bytes),
            ("max_pipeline_entries", self.max_pipeline_entries),
        ] {
            if value == 0 {
                return Err(Error::invalid(format!("`{name}` must be non-zero")));
            }
        }
        if self.completion_timeout_ms == 0 {
            return Err(Error::invalid("`completion_timeout_ms` must be non-zero"));
        }
        Ok(())
    }
}

/// Resource counters for one runtime.
#[derive(Debug, Clone, Copy, Default)]
pub struct Stats {
    /// This runtime's allocation-domain identity.
    pub domain_id: u64,
    /// Allocations handed out, including the zero-length representation.
    pub allocations: u64,
    /// Bytes handed out across those allocations.
    pub allocated_bytes: u64,
    /// Allocations still alive.
    pub live_allocations: u64,
    /// Bytes still alive.
    pub live_bytes: u64,
    /// Cached MSL artifacts.
    pub artifact_entries: usize,
    /// Cached source bytes.
    pub artifact_bytes: usize,
    /// Artifact-cache hits.
    pub artifact_hits: u64,
    /// Artifact-cache misses, each of which attempted a compile.
    pub artifact_misses: u64,
    /// Cached pipeline states.
    pub pipeline_entries: usize,
    /// Dispatches encoded.
    pub launches: u64,
    /// Command buffers committed.
    pub submissions: u64,
    /// Submissions not yet observed as complete.
    pub submissions_in_flight: usize,
}

/// One platform runtime: a device, a command queue and one allocation domain.
pub struct Runtime {
    holder: Arc<ContextHolder>,
    config: Config,
    session: Option<Session>,
    in_flight: std::collections::VecDeque<Completion>,
    artifacts: ArtifactCache,
    pipelines: Vec<CacheEntry>,
    counters: Counters,
    closed: bool,
}

#[derive(Default)]
struct Counters {
    allocations: u64,
    allocated_bytes: u64,
    launches: u64,
    submissions: u64,
}

struct CacheEntry {
    source_hash: String,
    entry: String,
    state: Arc<PipelineState>,
}

impl Runtime {
    /// Open the default Metal device, or report that this host has none.
    ///
    /// Nothing is constructed before the frameworks load and a device answers; a
    /// multi-backend binary calls [`crate::probe`] first and only opens a runtime
    /// that answered.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] for an invalid configuration, and
    /// [`Error::Unavailable`] when the frameworks, a required symbol or a device is
    /// missing.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn run() -> Result<(), t4a_metal_runtime::Error> {
    /// let mut runtime = t4a_metal_runtime::Runtime::open(Default::default())?;
    /// let buffer = runtime.alloc_zeroed(16)?;
    /// assert_eq!(buffer.len(), 16);
    /// # Ok(()) }
    /// # match run() {
    /// #     Ok(()) => {}
    /// #     Err(t4a_metal_runtime::Error::Unavailable { .. }) => {}
    /// #     Err(other) => panic!("unexpected open failure: {other}"),
    /// # }
    /// ```
    pub fn open(config: Config) -> Result<Self> {
        // Before the frameworks: an invalid configuration is reported as such on every
        // host, including one without Metal, so the error a caller sees does not depend
        // on whether a device happens to be there.
        config.validate()?;
        let table = Table::load().map_err(Error::from)?;
        Self::open_with(config, table)
    }

    /// Open a runtime over an already loaded table.
    ///
    /// The caller has already validated the configuration; [`Runtime::open`] does it
    /// before loading the table, and the probe passes [`Config::default`].
    pub(crate) fn open_with(config: Config, table: Table) -> Result<Self> {
        let holder = Arc::new(ContextHolder::open(table)?);
        Ok(Self {
            holder,
            artifacts: ArtifactCache::new(config.max_artifact_entries, config.max_artifact_bytes),
            config,
            session: None,
            in_flight: std::collections::VecDeque::new(),
            pipelines: Vec::new(),
            counters: Counters::default(),
            closed: false,
        })
    }

    /// The configuration this runtime was opened with.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Facts about the device.
    pub fn device(&self) -> &DeviceInfo {
        &self.holder.info
    }

    /// This runtime's allocation-domain identity.
    pub fn domain_id(&self) -> u64 {
        self.holder.domain_id()
    }

    /// Allocate a shared buffer whose contents are not initialised.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] when the size exceeds the device's maximum buffer
    /// length or the device refuses the allocation, and [`Error::Closed`] after
    /// [`Runtime::close`].
    pub fn alloc_uninit(&mut self, bytes: usize) -> Result<UninitBuffer> {
        self.check_open()?;
        let buffer = UninitBuffer::allocate(&self.holder, bytes)?;
        self.counters.allocations += 1;
        self.counters.allocated_bytes += bytes as u64;
        Ok(buffer)
    }

    /// Allocate a shared buffer and zero it.
    ///
    /// The zero fill is a host write before the buffer is reachable by any
    /// submission, so it needs no command buffer and no ordering.
    ///
    /// # Errors
    ///
    /// As [`Runtime::alloc_uninit`].
    pub fn alloc_zeroed(&mut self, bytes: usize) -> Result<Buffer> {
        self.check_open()?;
        let buffer = Buffer::allocate(&self.holder, bytes, true)?;
        self.counters.allocations += 1;
        self.counters.allocated_bytes += bytes as u64;
        Ok(buffer)
    }

    /// Compile MSL source, or return the cached artifact for the same source.
    ///
    /// # Errors
    ///
    /// [`Error::Compile`] with Metal's diagnostic when the compiler rejects the
    /// source, and [`Error::Closed`] after [`Runtime::close`].
    pub fn compile(&mut self, request: &CompileRequest) -> Result<Arc<Artifact>> {
        self.check_open()?;
        self.artifacts.get_or_compile(&self.holder, request)
    }

    /// Bind `entry` from `artifact` into a pipeline state for this runtime.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownEntryPoint`] when the library does not export `entry`,
    /// [`Error::Pipeline`] when Metal cannot build a pipeline for it,
    /// [`Error::ForeignDomain`] for an artifact from another runtime, and
    /// [`Error::Closed`] after [`Runtime::close`].
    pub fn bind(&mut self, artifact: &Arc<Artifact>, entry: &str) -> Result<BoundFunction> {
        self.check_open()?;
        if !Arc::ptr_eq(artifact.context(), &self.holder) {
            return Err(Error::ForeignDomain);
        }
        let hash = artifact.source_hash().to_string();
        let state = match self
            .pipelines
            .iter()
            .find(|candidate| candidate.source_hash == hash && candidate.entry == entry)
        {
            Some(cached) => Arc::clone(&cached.state),
            None => {
                let state = Arc::new(kernel::create_pipeline(&self.holder, artifact, entry)?);
                self.pipelines.push(CacheEntry {
                    source_hash: hash.clone(),
                    entry: entry.to_string(),
                    state: Arc::clone(&state),
                });
                while self.pipelines.len() > self.config.max_pipeline_entries {
                    self.pipelines.remove(0);
                }
                state
            }
        };
        Ok(BoundFunction::new(state, entry.to_string(), hash))
    }

    /// Encode one checked dispatch into the runtime's open command buffer.
    ///
    /// The `i`-th argument binds to buffer index `i`. Nothing is submitted until
    /// [`Runtime::commit`], [`Runtime::synchronize`] or [`Runtime::close`]; encoded work
    /// that none of them submits is discarded when the runtime is dropped.
    ///
    /// # Safety
    ///
    /// The kernel's MSL signature must match `args`: one parameter per index, each
    /// with the type the argument supplies. Every bound buffer must be valid for the
    /// accesses the kernel performs, and outputs the caller later treats as
    /// initialised must be fully overwritten. The grid must also match: a kernel that
    /// indexes an `n`-element buffer must be launched with a grid of exactly `n`
    /// threads ([`LaunchConfig::threads`]) or guard its indices, because every thread the
    /// grid contains runs and Metal does not bound-check an index ([`LaunchConfig::tiled`]).
    ///
    /// # Errors
    ///
    /// [`Error::ForeignDomain`] for a function or buffer from another runtime,
    /// [`Error::InvalidArgument`] for zero dimensions, a threadgroup larger than the
    /// pipeline accepts, a misaligned binding offset or a malformed by-value
    /// parameter, [`Error::OutOfRange`] for an offset outside its buffer,
    /// [`Error::KernelParameterLimit`] for a by-value parameter above the limit, and
    /// [`Error::Closed`] after [`Runtime::close`].
    pub unsafe fn launch(
        &mut self,
        function: &BoundFunction,
        config: LaunchConfig,
        args: &[Arg<'_>],
    ) -> Result<()> {
        self.check_open()?;
        // SAFETY: the caller owns the kernel ABI contract; `encode` performs every
        // check this layer can make.
        let result = kernel::encode(&self.holder, &mut self.session, function, config, args);
        if result.is_ok() {
            self.counters.launches += 1;
        }
        result
    }

    /// Close the encoder, submit the command buffer and return its completion.
    ///
    /// Committing with nothing encoded submits an empty command buffer, which is a
    /// cheap way to obtain a handle that resolves with the queue.
    ///
    /// # Errors
    ///
    /// [`Error::Unavailable`] when Metal cannot create the event or the buffer, and
    /// [`Error::Async`] or [`Error::Timeout`] when the bound on in-flight submissions
    /// forces a wait that fails. [`Error::Closed`] after [`Runtime::close`].
    pub fn commit(&mut self) -> Result<Completion> {
        self.check_open()?;
        let session = match self.session.take() {
            Some(session) => session,
            None => Session::new(Arc::clone(&self.holder))?,
        };
        let completion = self.submit(session)?;
        while self.in_flight.len() > self.config.max_submissions_in_flight {
            let Some(oldest) = self.in_flight.pop_front() else {
                break;
            };
            oldest.wait()?;
        }
        Ok(completion)
    }

    /// Submit what is pending, then wait for every submission.
    ///
    /// Each wait is bounded by [`Config::completion_timeout_ms`], so this reports
    /// [`Error::Timeout`] instead of blocking forever on a stalled submission.
    ///
    /// # Errors
    ///
    /// The first [`Error::Async`] or [`Error::Timeout`] seen, after every other
    /// submission has been waited for. [`Error::Closed`] after [`Runtime::close`].
    pub fn synchronize(&mut self) -> Result<()> {
        self.check_open()?;
        self.drain()
    }

    /// Run `closure` with the submission the runtime is building.
    ///
    /// The open compute encoder is closed first, so the closure may open any encoder
    /// type. Anything it encodes is submitted by the next [`Runtime::commit`].
    ///
    /// # Safety
    ///
    /// The closure owns the encoder lifetime and every ABI it asserts, must not read
    /// or write outside the buffers it was given, and keeps every foreign resource
    /// alive through `keepalives`. A panic unwinds after discarding the partly encoded
    /// command buffer, so nothing half written can be submitted later.
    ///
    /// # Errors
    ///
    /// [`Error::Unavailable`] when Metal cannot create the command buffer, and
    /// [`Error::Closed`] after [`Runtime::close`].
    pub unsafe fn with_raw<R>(
        &mut self,
        keepalives: Vec<Box<dyn std::any::Any + Send>>,
        closure: impl FnOnce(RawScope<'_>) -> R,
    ) -> Result<R> {
        self.check_open()?;
        if self.session.is_none() {
            self.session = Some(Session::new(Arc::clone(&self.holder))?);
        }
        let Some(session) = self.session.as_mut() else {
            return Err(Error::Closed);
        };
        session.end_encoder();
        let command_buffer = session.command_buffer();
        let scope = RawScope::new(&self.holder, command_buffer);
        // Dropping an uncommitted command buffer discards its work, which is what a
        // panicking closure must leave behind rather than a buffer a later commit could
        // submit half encoded.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| closure(scope)));
        let value = match result {
            Ok(value) => value,
            Err(payload) => {
                self.session = None;
                std::panic::resume_unwind(payload);
            }
        };
        if let Some(session) = self.session.as_mut() {
            for keepalive in keepalives {
                session.retain_keepalive(keepalive);
            }
        }
        Ok(value)
    }

    /// Resource counters.
    pub fn stats(&self) -> Stats {
        Stats {
            domain_id: self.holder.domain_id(),
            allocations: self.counters.allocations,
            allocated_bytes: self.counters.allocated_bytes,
            live_allocations: self.holder.live_allocations(),
            live_bytes: self.holder.live_bytes(),
            artifact_entries: self.artifacts.len(),
            artifact_bytes: self.artifacts.bytes(),
            artifact_hits: self.artifacts.hits(),
            artifact_misses: self.artifacts.misses(),
            pipeline_entries: self.pipelines.len(),
            launches: self.counters.launches,
            submissions: self.counters.submissions,
            submissions_in_flight: self.in_flight.len(),
        }
    }

    /// Submit what is pending, wait for everything, and drop the caches.
    ///
    /// Each wait is bounded by [`Config::completion_timeout_ms`], so a stalled submission
    /// is reported as [`Error::Timeout`] rather than blocking this call forever.
    ///
    /// Buffers, bound functions and completions keep the device and queue alive
    /// until the last of them is dropped; `close` never releases anything they use.
    ///
    /// # Errors
    ///
    /// [`Error::Async`] or [`Error::Timeout`] from the final synchronisation. The
    /// runtime is closed even when an error is reported.
    pub fn close(&mut self) -> Result<()> {
        if self.closed {
            return Ok(());
        }
        let result = self.drain();
        self.artifacts.clear();
        self.pipelines.clear();
        self.session = None;
        self.closed = true;
        result
    }

    /// Submit the open session, then wait for every submission.
    fn drain(&mut self) -> Result<()> {
        if let Some(session) = self.session.take() {
            self.submit(session)?;
        }
        let mut first_error = None;
        while let Some(completion) = self.in_flight.pop_front() {
            if let Err(error) = completion.wait() {
                first_error.get_or_insert(error);
            }
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    fn submit(&mut self, session: Session) -> Result<Completion> {
        let submission = session.finish(self.config.completion_timeout_ms)?;
        let completion = Completion::new(submission);
        self.counters.submissions += 1;
        self.in_flight.push_back(completion.clone());
        Ok(completion)
    }

    fn check_open(&self) -> Result<()> {
        if self.closed {
            return Err(Error::Closed);
        }
        Ok(())
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        // An uncommitted session is discarded, never submitted: `launch` and `with_raw`
        // promise that nothing runs until `commit`, `synchronize` or `close`, and a drop
        // cannot report an error, so submitting here would break that promise and hide
        // the failure.
        self.session = None;
        // Committed work is waited for with the same bound as every other observation,
        // `completion_timeout_ms`. A drop cannot report the `Error::Timeout`, so the first
        // one ends the wait and the remaining submissions are abandoned. Metal retains the
        // buffers a command buffer references, so releasing them here never frees memory
        // the GPU is still reading; a caller that wants the work finished and the errors
        // reported calls `close` instead.
        while let Some(completion) = self.in_flight.pop_front() {
            if completion.wait().is_err() {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests;

impl std::fmt::Debug for Runtime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Runtime")
            .field("domain_id", &self.holder.domain_id())
            .field("device", &self.holder.info.name)
            .field("closed", &self.closed)
            .finish_non_exhaustive()
    }
}
