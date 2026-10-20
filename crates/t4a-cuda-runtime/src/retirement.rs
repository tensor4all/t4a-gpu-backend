//! Retirement of resources whose lifetime CUDA's stream-ordered free does not
//! cover: pinned staging, loaded modules, consumer keepalives and cross-context
//! completion witnesses.
//!
//! Device buffers are **not** tracked here: their last `Drop` enqueues
//! `cuMemFreeAsync` on the owning stream, which is already ordered after every
//! admitted use on that stream.
//!
//! Invariant: every method that issues a CUDA call is called with the domain's
//! context active ([`crate::context::CtxScope`]); a [`Witness`] activates *its
//! own* context for each query and restores the previous binding.

use std::any::Any;
use std::collections::VecDeque;
use std::mem::MaybeUninit;
use std::ptr;
use std::sync::{Arc, Mutex, Weak};

use cudarc::driver::result as drv;
use cudarc::driver::sys;

use crate::context::ContextHolder;
use crate::error::{Error, Result};

/// Limits applied to retirement resources.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Limits {
    pub pending_bytes: usize,
    pub pending_records: usize,
    pub idle_pinned_bytes: usize,
    pub max_chunk_bytes: usize,
}

/// Counters describing retirement activity.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RetirementStats {
    /// Bytes held for work that is still in flight.
    pub pending_bytes: usize,
    /// Bytes retained by live loaded modules.
    pub module_bytes: usize,
    /// Retirement records (epochs, open leases, unresolved witnesses).
    pub pending_records: usize,
    /// Pinned bytes currently idle in the staging ring.
    pub idle_pinned_bytes: usize,
    /// Idle staging chunks.
    pub idle_chunks: usize,
    /// Sealed epochs that have not been reclaimed.
    pub epochs: usize,
    /// Unresolved producer witnesses exported from other contexts.
    pub frontiers: usize,
    /// Epochs reclaimed after completion.
    pub reclaimed_epochs: u64,
    /// Poll iterations spent waiting for the pending budget.
    pub pressure_polls: u64,
    /// Kernel launches admitted by the runtime.
    pub launches: u64,
    /// Device allocations made by the runtime.
    pub allocations: u64,
    /// Zero fills enqueued by the runtime.
    pub zero_fills: u64,
    /// Completion events recorded on the stream, including cross-runtime ones.
    pub events_recorded: u64,
    /// Blocking observation loops entered by the caller.
    pub host_waits: u64,
    /// Pinned staging chunks allocated from CUDA.
    pub chunk_allocs: u64,
    /// Staging chunks leaked because completion could not be proved.
    pub abandoned_chunks: u64,
}

/// Narrow post-enqueue injection point for deterministic failure tests.
///
/// Compiled only under `cfg(test)`; production builds always return `Ok`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FaultPoint {
    /// After work was enqueued but before its completion event is recorded.
    BeforeEventRecord,
    /// Before the source stream is ordered behind the copy.
    BeforeSourceBackWait,
    /// After a host-transfer chunk was enqueued but before its epoch is committed.
    AfterChunkEnqueue,
}

#[cfg(test)]
mod fault {
    use super::FaultPoint;
    use std::cell::Cell;

    thread_local! {
        static ARMED: Cell<Option<FaultPoint>> = const { Cell::new(None) };
    }

    pub(crate) fn arm(point: FaultPoint) {
        ARMED.with(|armed| armed.set(Some(point)));
    }

    pub(crate) fn disarm() {
        ARMED.with(|armed| armed.set(None));
    }

    pub(crate) fn trigger(point: FaultPoint) -> bool {
        ARMED.with(|armed| armed.get() == Some(point))
    }
}

#[cfg(not(test))]
mod fault {
    use super::FaultPoint;

    #[allow(dead_code)]
    pub(crate) fn arm(_point: FaultPoint) {}
    #[allow(dead_code)]
    pub(crate) fn disarm() {}
    pub(crate) fn trigger(_point: FaultPoint) -> bool {
        false
    }
}

/// Fail when the named boundary is armed by a test.
pub(crate) fn fault_point(point: FaultPoint) -> Result<()> {
    if fault::trigger(point) {
        return Err(Error::Async {
            detail: format!("injected fault at {point:?}"),
        });
    }
    Ok(())
}

#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn arm_fault(point: FaultPoint) {
    fault::arm(point);
}

#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn disarm_fault() {
    fault::disarm();
}

/// A pinned host staging chunk.
pub(crate) struct PinnedChunk {
    ptr: *mut u8,
    len: usize,
    holder: Arc<ContextHolder>,
}

// SAFETY: the chunk is pinned host memory with no thread affinity. Every CUDA
// call on it runs inside a `CtxScope` created for `holder` on the using thread.
unsafe impl Send for PinnedChunk {}

impl PinnedChunk {
    pub(crate) fn alloc(holder: &Arc<ContextHolder>, len: usize) -> Result<Self> {
        let _scope = holder.activate()?;
        // SAFETY: portable page-locked memory, so the staged cross-device route
        // may DMA into it from the source context as well.
        let ptr = unsafe { drv::malloc_host(len.max(1), sys::CU_MEMHOSTALLOC_PORTABLE) }
            .map_err(|e| Error::from_alloc(len, "cuMemHostAlloc", e))? as *mut u8;
        Ok(Self {
            ptr,
            len,
            holder: Arc::clone(holder),
        })
    }

    pub(crate) fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }
}

impl Drop for PinnedChunk {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        // SAFETY: this chunk owns the allocation and is freed once. If the owning
        // context cannot be activated, leak the memory rather than freeing it
        // without an active context.
        match self.holder.activate() {
            Ok(_scope) => unsafe {
                let _ = drv::free_host(self.ptr.cast());
            },
            // Drop the pointer without freeing: the allocation is deliberately
            // leaked because its completion cannot be proved.
            Err(_) => self.ptr = ptr::null_mut(),
        }
    }
}

/// Shared, one-shot observation of a pending host download.
pub(crate) struct DownloadSlot {
    pub(crate) state: Mutex<SlotState>,
}

pub(crate) struct SlotState {
    pub(crate) data: Vec<MaybeUninit<u8>>,
    pub(crate) done: usize,
    pub(crate) failed: Option<String>,
    pub(crate) taken: bool,
}

impl DownloadSlot {
    pub(crate) fn new(len: usize) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(SlotState {
                data: vec![MaybeUninit::uninit(); len],
                done: 0,
                failed: None,
                taken: false,
            }),
        })
    }
}

/// Where a completed staging chunk is copied before it is recycled.
pub(crate) struct DownloadTarget {
    slot: Weak<DownloadSlot>,
    offset: usize,
    len: usize,
}

/// A resource retained until the epoch it belongs to completes.
pub(crate) enum Lease {
    Chunk {
        chunk: PinnedChunk,
        download: Option<DownloadTarget>,
    },
    Keepalive(#[allow(dead_code)] Box<dyn Any + Send>),
}

/// A sealed completion boundary on one stream.
struct Epoch {
    event: sys::CUevent,
    leases: Vec<Lease>,
    witnesses: Vec<Arc<Witness>>,
    bytes: usize,
}

#[derive(Default)]
struct OpenEpoch {
    leases: Vec<Lease>,
    witnesses: Vec<Arc<Witness>>,
    bytes: usize,
}

struct State {
    pending_bytes: usize,
    module_bytes: usize,
    open: OpenEpoch,
    epochs: VecDeque<Epoch>,
    idle: Vec<PinnedChunk>,
    idle_bytes: usize,
    frontiers: Vec<Arc<Witness>>,
    broken: Option<String>,
    stats: RetirementStats,
}

/// A completion witness exported from another context.
///
/// The witness owns its event and keeps the source context alive. It carries the
/// unresolved upstream witnesses transitively, so observing `C` after `A -> B ->
/// C` also observes `A`.
pub(crate) struct Witness {
    ctx: Arc<ContextHolder>,
    event: sys::CUevent,
    label: String,
    upstream: Mutex<Vec<Arc<Witness>>>,
    /// Terminal observation: `Some(None)` = confirmed, `Some(Some(msg))` = failed.
    terminal: Mutex<Option<Option<String>>>,
    /// Serialises the whole check/query/update sequence.
    query_lock: Mutex<()>,
}

// SAFETY: the event belongs to `ctx`; every query activates that context on the
// querying thread and restores the previous binding before returning.
unsafe impl Send for Witness {}
unsafe impl Sync for Witness {}

pub(crate) enum WitnessStatus {
    Pending,
    Done,
}

impl Witness {
    pub(crate) fn new(
        ctx: &Arc<ContextHolder>,
        event: sys::CUevent,
        label: impl Into<String>,
        upstream: Vec<Arc<Witness>>,
    ) -> Arc<Self> {
        Arc::new(Self {
            ctx: Arc::clone(ctx),
            event,
            label: label.into(),
            upstream: Mutex::new(upstream),
            terminal: Mutex::new(None),
            query_lock: Mutex::new(()),
        })
    }

    /// Non-blocking observation under the witness's own context.
    ///
    /// The first terminal answer is remembered, so a later unrelated failure in
    /// the source context cannot retroactively fail an already confirmed
    /// transfer, and a shared upstream is not re-queried.
    pub(crate) fn query(&self) -> Result<WitnessStatus> {
        let _serialised = self
            .query_lock
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        {
            let terminal = self
                .terminal
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            match &*terminal {
                Some(None) => return Ok(WitnessStatus::Done),
                Some(Some(detail)) => {
                    return Err(Error::Async {
                        detail: detail.clone(),
                    })
                }
                None => {}
            }
        }
        // Check upstream dependencies first: a pending intermediate event must not
        // hide a failure that already happened further upstream.
        let upstream = self
            .upstream
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .clone();
        let mut upstream_pending = false;
        for dependency in &upstream {
            match dependency.query() {
                Ok(WitnessStatus::Pending) => upstream_pending = true,
                Ok(WitnessStatus::Done) => {}
                Err(error) => return Err(error),
            }
        }
        let _scope = self.ctx.activate()?;
        // SAFETY: `event` was created for this context and is owned by `self`.
        match unsafe { drv::event::query(self.event) } {
            Ok(()) => {}
            Err(error) if error.0 == sys::CUresult::CUDA_ERROR_NOT_READY => {
                return Ok(WitnessStatus::Pending)
            }
            Err(error) => {
                let error = Error::Async {
                    detail: format!("source-context witness `{}`: {error}", self.label),
                };
                *self
                    .terminal
                    .lock()
                    .unwrap_or_else(|poison| poison.into_inner()) = Some(Some(error.to_string()));
                return Err(error);
            }
        }
        if upstream_pending {
            return Ok(WitnessStatus::Pending);
        }
        // Terminal success: drop the upstream history so repeated copies do not
        // retain completed witness chains.
        self.upstream
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .clear();
        *self
            .terminal
            .lock()
            .unwrap_or_else(|poison| poison.into_inner()) = Some(None);
        Ok(WitnessStatus::Done)
    }
}

impl Drop for Witness {
    fn drop(&mut self) {
        // SAFETY: the event is owned here and was created for this context. A
        // failure to activate leaks the handle rather than acting on a stale one.
        if let Ok(_scope) = self.ctx.activate() {
            unsafe {
                let _ = drv::event::destroy(self.event);
            }
        }
    }
}

/// Retirement state for one allocation domain.
///
/// One operation lock serialises admission against observation: a concurrent
/// observer must never seal an epoch in the middle of an admission, and a
/// reclaimer must never recycle the storage of an epoch it did not check.
pub(crate) struct Retirement {
    state: Mutex<State>,
    operation: Mutex<()>,
    owner: Mutex<Option<std::thread::ThreadId>>,
    limits: Limits,
}

impl Retirement {
    pub(crate) fn new(limits: Limits) -> Self {
        Self {
            state: Mutex::new(State {
                pending_bytes: 0,
                module_bytes: 0,
                open: OpenEpoch::default(),
                epochs: VecDeque::new(),
                idle: Vec::new(),
                idle_bytes: 0,
                frontiers: Vec::new(),
                broken: None,
                stats: RetirementStats::default(),
            }),
            operation: Mutex::new(()),
            owner: Mutex::new(None),
            limits,
        }
    }

    fn lock_operation(&self) -> std::sync::MutexGuard<'_, ()> {
        self.operation
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    /// Run `closure` with admission serialised against observation.
    pub(crate) fn during_admission<R>(
        &self,
        closure: impl FnOnce(&Admission<'_>) -> Result<R>,
    ) -> Result<R> {
        let _operation = self.lock_operation();
        {
            let mut owner = self.owner.lock().unwrap_or_else(|p| p.into_inner());
            *owner = Some(std::thread::current().id());
        }
        // Reset the owner on unwind too, so a panicking closure cannot make the
        // runtime look permanently re-entered.
        let _owner = OwnerGuard { retirement: self };
        closure(&Admission { retirement: self })
    }

    /// Reject observation from inside an admission on the same thread instead
    /// of deadlocking on the operation lock.
    fn check_reentry(&self) -> Result<()> {
        let owner = *self.owner.lock().unwrap_or_else(|p| p.into_inner());
        if owner == Some(std::thread::current().id()) {
            return Err(Error::invalid(
                "observation is not allowed from inside an admission on the same thread",
            ));
        }
        Ok(())
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    /// Charge a live module's image size against the pending budget.
    pub(crate) fn retain_module_bytes(&self, bytes: usize) {
        self.lock().module_bytes += bytes;
    }

    /// Release a live module's charge.
    pub(crate) fn release_module_bytes(&self, bytes: usize) {
        let mut state = self.lock();
        state.module_bytes = state.module_bytes.saturating_sub(bytes);
    }

    /// Count one admitted kernel launch.
    pub(crate) fn note_launch(&self) {
        self.lock().stats.launches += 1;
    }

    /// Count one device allocation.
    pub(crate) fn note_allocation(&self) {
        self.lock().stats.allocations += 1;
    }

    /// Count one zero fill.
    pub(crate) fn note_zero_fill(&self) {
        self.lock().stats.zero_fills += 1;
    }

    /// Record a completion event and count it, including for cross-runtime work.
    pub(crate) fn record_completion(&self, stream: sys::CUstream) -> Result<sys::CUevent> {
        self.record(stream)
    }

    /// Count one caller-driven blocking observation loop.
    pub(crate) fn note_host_wait(&self) {
        self.lock().stats.host_waits += 1;
    }

    /// Record a failure that makes further admission unsafe.
    pub(crate) fn mark_broken(&self, detail: impl Into<String>) {
        let mut state = self.lock();
        state.broken = Some(detail.into());
    }

    /// Record an already-submitted failure that must be reported by observation.
    pub(crate) fn break_on_error(&self, error: &Error) {
        if !matches!(error, Error::Backpressure { .. }) {
            self.mark_broken(error.to_string());
        }
    }

    fn capacity_fits(state: &State, limits: &Limits, bytes: usize, records: usize) -> bool {
        state.pending_bytes + state.module_bytes + bytes <= limits.pending_bytes
            && state.pending_records() + records <= limits.pending_records
    }

    /// Reserve capacity before an enqueue. Requires the operation lock.
    fn reserve_locked(&self, stream: sys::CUstream, bytes: usize, records: usize) -> Result<()> {
        {
            let state = self.lock();
            // Live modules cannot be reclaimed by draining epochs, so an
            // admission that cannot fit together with them must fail instead of
            // spinning forever.
            if state.module_bytes + bytes > self.limits.pending_bytes {
                return Err(Error::Backpressure { bytes });
            }
        }
        if bytes > self.limits.pending_bytes || records > self.limits.pending_records {
            return Err(Error::Backpressure { bytes });
        }
        loop {
            {
                let state = self.lock();
                if let Some(detail) = &state.broken {
                    return Err(Error::Broken {
                        detail: detail.clone(),
                    });
                }
                if Self::capacity_fits(&state, &self.limits, bytes, records) {
                    return Ok(());
                }
            }
            let settled = self.settle_one_locked()?;
            let pruned = self.prune_frontiers_locked()?;
            if !settled && !pruned {
                // Nothing completed. Seal the open epoch so its leases become
                // reclaimable, then wait for progress instead of blocking on a
                // foreign-dependent event.
                let open = { !self.lock().open.leases.is_empty() };
                if open {
                    self.seal_locked(stream)?;
                    continue;
                }
                let mut state = self.lock();
                state.stats.pressure_polls += 1;
                drop(state);
                std::thread::yield_now();
            }
        }
    }

    /// Reclaim at most one completed epoch. Requires the operation lock.
    fn settle_one_locked(&self) -> Result<bool> {
        let (event, witnesses) = {
            let state = self.lock();
            match state.epochs.front() {
                Some(epoch) => (epoch.event, epoch.witnesses.clone()),
                None => return Ok(false),
            }
        };
        for witness in &witnesses {
            match witness.query() {
                Ok(WitnessStatus::Done) => {}
                Ok(WitnessStatus::Pending) => return Ok(false),
                Err(error) => {
                    self.break_on_error(&error);
                    return Err(error);
                }
            }
        }
        // SAFETY: called with the domain context active.
        match unsafe { drv::event::query(event) } {
            Ok(()) => {}
            Err(error) if error.0 == sys::CUresult::CUDA_ERROR_NOT_READY => return Ok(false),
            Err(error) => {
                let error = Error::Async {
                    detail: format!("retirement epoch: {error}"),
                };
                self.break_on_error(&error);
                return Err(error);
            }
        }
        let epoch = {
            let mut state = self.lock();
            // The operation lock is held, so the front epoch is still the one
            // whose completion was just observed.
            let epoch = state.epochs.pop_front();
            if let Some(epoch) = &epoch {
                state.pending_bytes = state.pending_bytes.saturating_sub(epoch.bytes);
                state.stats.reclaimed_epochs += 1;
            }
            state.stats.pending_records = state.pending_records();
            epoch
        };
        if let Some(epoch) = epoch {
            // SAFETY: the event captured all leases; completion was observed.
            unsafe {
                let _ = drv::event::destroy(epoch.event);
            }
            for lease in epoch.leases {
                self.settle_lease(lease);
            }
        }
        Ok(true)
    }

    /// Prune producer frontiers whose dependencies completed. Requires the
    /// operation lock.
    fn prune_frontiers_locked(&self) -> Result<bool> {
        let frontiers = { self.lock().frontiers.clone() };
        let mut progress = false;
        for witness in frontiers {
            match witness.query() {
                Ok(WitnessStatus::Pending) => {}
                Ok(WitnessStatus::Done) => {
                    self.prune_frontier(&witness);
                    progress = true;
                }
                Err(error) => {
                    self.break_on_error(&error);
                    return Err(error);
                }
            }
        }
        Ok(progress)
    }

    fn settle_lease(&self, lease: Lease) {
        match lease {
            Lease::Chunk { chunk, download } => {
                if let Some(target) = download {
                    if let Some(slot) = target.slot.upgrade() {
                        let mut state = slot.state.lock().unwrap_or_else(|p| p.into_inner());
                        if state.failed.is_none() {
                            let remaining = state.data.len().saturating_sub(target.offset);
                            let copied = remaining.min(target.len).min(chunk.len());
                            // SAFETY: `offset + copied <= data.len()` and the D2H
                            // completed, so these bytes are initialised.
                            unsafe {
                                ptr::copy_nonoverlapping(
                                    chunk.as_ptr(),
                                    state.data.as_mut_ptr().cast::<u8>().add(target.offset),
                                    copied,
                                );
                            }
                            state.done += copied;
                        }
                    }
                }
                self.recycle_chunk(chunk);
            }
            Lease::Keepalive(_) => {}
        }
    }

    fn recycle_chunk(&self, chunk: PinnedChunk) {
        let mut state = self.lock();
        if state.idle_bytes + chunk.len() <= self.limits.idle_pinned_bytes {
            state.idle_bytes += chunk.len();
            state.idle.push(chunk);
        }
        state.stats.idle_pinned_bytes = state.idle_bytes;
        state.stats.idle_chunks = state.idle.len();
    }

    /// Take an idle chunk or allocate a new pinned staging chunk.
    pub(crate) fn take_chunk(
        &self,
        holder: &Arc<ContextHolder>,
        want: usize,
    ) -> Result<PinnedChunk> {
        // Never pick a chunk that cannot fit beside live modules.
        let available = self.available_host_bytes().max(1);
        let want = want.clamp(1, self.limits.max_chunk_bytes.min(available));
        let idle = {
            let mut state = self.lock();
            let index = state
                .idle
                .iter()
                .position(|chunk| chunk.len() >= want && chunk.len() <= available);
            let chunk = index.map(|index| state.idle.remove(index));
            if let Some(chunk) = &chunk {
                state.idle_bytes = state.idle_bytes.saturating_sub(chunk.len());
                state.stats.idle_pinned_bytes = state.idle_bytes;
                state.stats.idle_chunks = state.idle.len();
            }
            chunk
        };
        match idle {
            Some(chunk) => Ok(chunk),
            None => {
                let chunk = PinnedChunk::alloc(holder, want)?;
                self.lock().stats.chunk_allocs += 1;
                Ok(chunk)
            }
        }
    }

    /// Commit a staging chunk whose capacity was reserved before its enqueue.
    ///
    /// Only event creation can still fail; the caller then abandons the chunk
    /// instead of dropping storage a queued DMA may still write.
    fn commit_chunk_locked(
        &self,
        stream: sys::CUstream,
        chunk: PinnedChunk,
        download: Option<DownloadTarget>,
        witnesses: Vec<Arc<Witness>>,
    ) -> std::result::Result<(), (Error, PinnedChunk)> {
        let bytes = chunk.len();
        let event = match self.record(stream) {
            Ok(event) => event,
            // The DMA is already enqueued: hand the storage back so the caller can
            // abandon it instead of freeing memory a queued transfer may write.
            Err(error) => return Err((error, chunk)),
        };
        let mut state = self.lock();
        state.pending_bytes += bytes;
        state.epochs.push_back(Epoch {
            event,
            leases: vec![Lease::Chunk { chunk, download }],
            witnesses,
            bytes,
        });
        state.stats.pending_records = state.pending_records();
        Ok(())
    }

    /// Retain a keepalive in the open epoch. Capacity must already be reserved.
    fn add_keepalive_locked(&self, keepalive: Box<dyn Any + Send>, bytes: usize) {
        let mut state = self.lock();
        state.pending_bytes += bytes;
        state.open.bytes += bytes;
        state.open.leases.push(Lease::Keepalive(keepalive));
        state.stats.pending_records = state.pending_records();
    }

    fn seal_if_half_locked(&self, stream: sys::CUstream) -> Result<()> {
        let should_seal = {
            let state = self.lock();
            state.open.leases.len() >= (self.limits.pending_records / 2).max(1)
        };
        if should_seal {
            self.seal_locked(stream)?;
        }
        Ok(())
    }

    fn record(&self, stream: sys::CUstream) -> Result<sys::CUevent> {
        let event = record_event(stream)?;
        self.lock().stats.events_recorded += 1;
        Ok(event)
    }

    /// Seal the open epoch so its keepalives can be retired.
    ///
    /// The event is created and recorded *before* the leases leave the open
    /// epoch, so a failure keeps them registered instead of dropping resources
    /// whose completion is unproved.
    fn seal_locked(&self, stream: sys::CUstream) -> Result<()> {
        {
            let state = self.lock();
            if state.open.leases.is_empty() {
                return Ok(());
            }
        }
        let event = self.record(stream)?;
        let mut state = self.lock();
        let open = std::mem::take(&mut state.open);
        state.epochs.push_back(Epoch {
            event,
            leases: open.leases,
            witnesses: open.witnesses,
            bytes: open.bytes,
        });
        state.stats.pending_records = state.pending_records();
        Ok(())
    }

    /// Retire everything that has completed and adopt observed failures.
    ///
    /// Returns whether every upstream producer witness is confirmed; a pending
    /// witness means the derived result is not yet trustworthy.
    pub(crate) fn observe_gated(&self, stream: sys::CUstream) -> Result<bool> {
        self.observe_inner(stream)
    }

    pub(crate) fn observe(&self, stream: sys::CUstream) -> Result<()> {
        self.observe_inner(stream).map(|_| ())
    }

    fn observe_inner(&self, stream: sys::CUstream) -> Result<bool> {
        self.check_reentry()?;
        let _operation = self.lock_operation();
        self.seal_locked(stream)?;
        while self.settle_one_locked()? {}
        self.prune_frontiers_locked()?;
        let state = self.lock();
        if let Some(detail) = &state.broken {
            // A failure that was already recorded (for example by a failed Drop)
            // is reported once observation reaches this boundary.
            return Err(Error::Broken {
                detail: detail.clone(),
            });
        }
        Ok(state.frontiers.is_empty())
    }

    /// Snapshot the unresolved producer frontiers for export to a downstream
    /// runtime, so observation stays transitive.
    pub(crate) fn frontier_snapshot(&self) -> Vec<Arc<Witness>> {
        self.lock().frontiers.clone()
    }

    fn prune_frontier(&self, witness: &Arc<Witness>) {
        let mut state = self.lock();
        state.frontiers.retain(|entry| !Arc::ptr_eq(entry, witness));
        state.stats.pending_records = state.pending_records();
    }

    /// Abandon a staging chunk whose last DMA could not be proved complete.
    ///
    /// The pinned memory is deliberately leaked: freeing host memory that a
    /// queued transfer may still read is worse than leaking it.
    fn abandon_chunk_locked(&self, chunk: PinnedChunk) {
        let mut state = self.lock();
        state.stats.abandoned_chunks += 1;
        std::mem::forget(chunk);
    }

    pub(crate) fn stats(&self) -> RetirementStats {
        let state = self.lock();
        let mut stats = state.stats;
        stats.pending_bytes = state.pending_bytes;
        stats.module_bytes = state.module_bytes;
        stats.pending_records = state.pending_records();
        stats.epochs = state.epochs.len();
        stats.frontiers = state.frontiers.len();
        stats.idle_chunks = state.idle.len();
        stats.idle_pinned_bytes = state.idle_bytes;
        stats
    }

    pub(crate) fn broken(&self) -> Option<String> {
        self.lock().broken.clone()
    }

    /// Whether every epoch and frontier has been resolved.
    pub(crate) fn is_drained(&self) -> bool {
        let state = self.lock();
        state.epochs.is_empty() && state.open.leases.is_empty() && state.frontiers.is_empty()
    }

    /// Host bytes still available beside live loaded modules.
    pub(crate) fn available_host_bytes(&self) -> usize {
        let state = self.lock();
        self.limits.pending_bytes.saturating_sub(state.module_bytes)
    }

    pub(crate) fn pending_bytes_hint(&self) -> usize {
        self.lock().pending_bytes
    }
}

impl State {
    fn pending_records(&self) -> usize {
        self.epochs.len() + self.open.leases.len() + self.frontiers.len()
    }
}

/// Construct a download target for `offset` bytes into `slot`.
struct OwnerGuard<'a> {
    retirement: &'a Retirement,
}

impl Drop for OwnerGuard<'_> {
    fn drop(&mut self) {
        *self
            .retirement
            .owner
            .lock()
            .unwrap_or_else(|poison| poison.into_inner()) = None;
    }
}

/// Admission view that holds the operation lock for a whole enqueue sequence.
///
/// Resources are reserved here *before* the corresponding CUDA call, so a failed
/// enqueue never drops storage whose last use is still queued.
pub(crate) struct Admission<'a> {
    retirement: &'a Retirement,
}

impl Admission<'_> {
    /// Reserve pending capacity, retiring or pruning earlier work first.
    pub(crate) fn ensure_capacity(
        &self,
        stream: sys::CUstream,
        bytes: usize,
        records: usize,
    ) -> Result<()> {
        self.retirement.reserve_locked(stream, bytes, records)
    }

    /// Commit a chunk after its DMA has been enqueued.
    ///
    /// On failure the chunk is handed back so the caller can abandon it rather
    /// than free storage whose last DMA may still be queued.
    pub(crate) fn commit_chunk(
        &self,
        stream: sys::CUstream,
        chunk: PinnedChunk,
        download: Option<DownloadTarget>,
        witnesses: Vec<Arc<Witness>>,
    ) -> std::result::Result<(), (Error, PinnedChunk)> {
        self.retirement
            .commit_chunk_locked(stream, chunk, download, witnesses)
    }

    /// Retain a keepalive until the open epoch completes.
    pub(crate) fn add_keepalive(
        &self,
        stream: sys::CUstream,
        keepalive: Box<dyn Any + Send>,
        bytes: usize,
        allow_seal: bool,
    ) -> Result<()> {
        if allow_seal {
            self.retirement.seal_if_half_locked(stream)?;
        }
        self.retirement.reserve_locked(stream, bytes, 1)?;
        self.retirement.add_keepalive_locked(keepalive, bytes);
        Ok(())
    }

    /// Retain a producer witness for later observation.
    pub(crate) fn add_witness(&self, witness: Arc<Witness>) -> Result<()> {
        loop {
            {
                let state = self.retirement.lock();
                if let Some(detail) = &state.broken {
                    return Err(Error::Broken {
                        detail: detail.clone(),
                    });
                }
                if state.pending_records() < self.retirement.limits.pending_records {
                    break;
                }
            }
            if !self.retirement.prune_frontiers_locked()? {
                return Err(Error::Backpressure { bytes: 0 });
            }
        }
        let mut state = self.retirement.lock();
        state.frontiers.push(witness);
        state.stats.pending_records = state.pending_records();
        Ok(())
    }

    /// Seal the open epoch.
    pub(crate) fn seal(&self, stream: sys::CUstream) -> Result<()> {
        self.retirement.seal_locked(stream)
    }

    /// Abandon a chunk whose last DMA could not be proved complete.
    pub(crate) fn abandon_chunk(&self, chunk: PinnedChunk) {
        self.retirement.abandon_chunk_locked(chunk);
    }

    pub(crate) fn take_chunk(
        &self,
        holder: &Arc<ContextHolder>,
        want: usize,
    ) -> Result<PinnedChunk> {
        self.retirement.take_chunk(holder, want)
    }

    pub(crate) fn chunk_limit(&self) -> usize {
        self.retirement.limits.max_chunk_bytes
    }

    /// Largest chunk that still fits beside live modules.
    pub(crate) fn available_host_bytes(&self) -> usize {
        let state = self.retirement.lock();
        self.retirement
            .limits
            .pending_bytes
            .saturating_sub(state.module_bytes)
    }
}

pub(crate) fn download_target(
    slot: &Arc<DownloadSlot>,
    offset: usize,
    len: usize,
) -> DownloadTarget {
    DownloadTarget {
        slot: Arc::downgrade(slot),
        offset,
        len,
    }
}

/// Create and record a completion event on `stream`.
pub(crate) fn record_event(stream: sys::CUstream) -> Result<sys::CUevent> {
    let event = drv::event::create(sys::CUevent_flags::CU_EVENT_DEFAULT)
        .map_err(Error::driver("cuEventCreate"))?;
    // SAFETY: `event` was just created for the active context.
    if let Err(error) = unsafe { drv::event::record(event, stream) } {
        // SAFETY: the event never captured work.
        unsafe {
            let _ = drv::event::destroy(event);
        }
        return Err(Error::from_driver("cuEventRecord", error));
    }
    Ok(event)
}

#[cfg(test)]
mod tests;
