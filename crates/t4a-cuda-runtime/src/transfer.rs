//! Ordered host transfers and cross-runtime copies.
//!
//! Host and device never exchange data implicitly. Every transfer is admitted on
//! the owning stream and reported through [`PendingDownload`] or [`Completion`].

use std::sync::Arc;

use cudarc::driver::result as drv;
use cudarc::driver::sys;

use crate::context::ContextHolder;
use crate::error::{Error, Result};
use crate::memory::{Alloc, Buffer, Domain};
use crate::retirement::{
    download_target, fault_point, record_event, DownloadSlot, FaultPoint, PinnedChunk, Witness,
};

/// A pinned host buffer the caller fills before handing it to the runtime.
pub struct PinnedHostBuffer {
    chunk: Option<PinnedChunk>,
    len: usize,
}

// SAFETY: pinned host memory has no thread affinity. Every CUDA call on it runs
// inside the owning domain's context scope.
unsafe impl Send for PinnedHostBuffer {}

impl PinnedHostBuffer {
    pub(crate) fn new(holder: &Arc<ContextHolder>, len: usize) -> Result<Self> {
        let chunk = PinnedChunk::alloc(holder, len)?;
        // SAFETY: the chunk owns `len` initialised-able bytes of pinned memory.
        unsafe {
            std::ptr::write_bytes(chunk.as_ptr(), 0, len);
        }
        Ok(Self {
            chunk: Some(chunk),
            len,
        })
    }

    /// Length in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Mutable view of the pinned bytes.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        let chunk = self
            .chunk
            .as_ref()
            .expect("pinned buffer is not consumed yet");
        // SAFETY: the allocation is at least `len` bytes and initialised.
        unsafe { std::slice::from_raw_parts_mut(chunk.as_ptr(), self.len) }
    }

    /// Immutable view of the pinned bytes.
    pub fn as_slice(&self) -> &[u8] {
        let chunk = self
            .chunk
            .as_ref()
            .expect("pinned buffer is not consumed yet");
        // SAFETY: the allocation is at least `len` bytes and initialised.
        unsafe { std::slice::from_raw_parts(chunk.as_ptr(), self.len) }
    }
}

/// How a cross-runtime copy moved its bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CopyRoute {
    /// Both endpoints share one physical device; no host staging was used.
    SameDevice,
    /// Distinct physical devices; bytes were staged through pinned host memory.
    Staged,
}

fn check_domain(domain: &Arc<Domain>, buffer_domain: &Arc<Domain>) -> Result<()> {
    if !Arc::ptr_eq(domain, buffer_domain) {
        return Err(Error::ForeignDomain);
    }
    Ok(())
}

/// Allocate a device buffer and upload `data`, chunked through pinned staging.
pub(crate) fn upload(domain: &Arc<Domain>, data: &[u8]) -> Result<Buffer> {
    if let Some(detail) = domain.broken() {
        return Err(Error::Broken { detail });
    }
    let alignment = 256;
    let alloc = allocate_buffer(domain, data.len(), alignment)?;
    let buffer = Buffer::from_parts(alloc, data.len(), alignment);
    stage_upload(domain, &buffer, data)?;
    Ok(buffer)
}

fn allocate_buffer(domain: &Arc<Domain>, len: usize, alignment: usize) -> Result<Alloc> {
    let ptr = domain.allocate(len, alignment)?;
    Ok(Alloc::new(ptr, len, Arc::clone(domain)))
}

/// Copy `data` into bounded staging chunks and enqueue the H2D of each chunk.
fn stage_upload(domain: &Arc<Domain>, buffer: &Buffer, data: &[u8]) -> Result<()> {
    if data.is_empty() {
        return Ok(());
    }
    let _scope = domain.scope()?;
    let stream = domain.stream();
    domain.retirement().during_admission(|admission| {
        let mut offset = 0usize;
        while offset < data.len() {
            let want = (data.len() - offset)
                .min(admission.chunk_limit())
                .min(admission.available_host_bytes().max(1));
            // Capacity is reserved before the stored allocation is taken, and the
            // chunk is committed only after its DMA is enqueued.
            let chunk = admission.take_chunk(domain.holder(), want)?;
            let len = want.min(chunk.len());
            admission.ensure_capacity(stream, chunk.len(), 1)?;
            // SAFETY: `chunk` has at least `len` bytes; the caller slice has `len`
            // bytes at `offset`. The copy completes before the DMA is enqueued, so
            // the DMA never reads caller-mutable memory.
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr().add(offset), chunk.as_ptr(), len);
            }
            // SAFETY: destination range was checked against the buffer length.
            let copied = unsafe {
                drv::memcpy_htod_async(
                    buffer.device_ptr() + offset as u64,
                    std::slice::from_raw_parts(chunk.as_ptr(), len),
                    stream,
                )
            };
            if let Err(error) = copied {
                admission.abandon_chunk(chunk);
                return Err(Error::from_driver("cuMemcpyHtoDAsync", error));
            }
            if let Err(error) = fault_point(FaultPoint::AfterChunkEnqueue) {
                admission.abandon_chunk(chunk);
                domain.mark_broken(error.to_string());
                return Err(error);
            }
            if let Err((error, chunk)) = admission.commit_chunk(stream, chunk, None, Vec::new()) {
                // The DMA may still run, so the storage is abandoned (leaked)
                // rather than freed.
                admission.abandon_chunk(chunk);
                domain.mark_broken(error.to_string());
                return Err(error);
            }
            offset += len;
        }
        Ok(())
    })
}

/// Upload a caller-owned pinned allocation.
///
/// The direct DMA path is used only when the whole allocation fits the pending
/// budget. Otherwise the bytes are staged through bounded chunks and the
/// original allocation is released before returning, so no queued DMA reads it.
pub(crate) fn upload_owned(domain: &Arc<Domain>, mut pinned: PinnedHostBuffer) -> Result<Buffer> {
    if let Some(detail) = domain.broken() {
        return Err(Error::Broken { detail });
    }
    let alignment = 256;
    let alloc = allocate_buffer(domain, pinned.len(), alignment)?;
    let buffer = Buffer::from_parts(alloc, pinned.len(), alignment);
    if pinned.is_empty() {
        return Ok(buffer);
    }
    // Live modules already consume part of the budget; only the headroom counts.
    let fits = pinned.len() <= domain.retirement().available_host_bytes();
    if !fits {
        stage_upload(domain, &buffer, pinned.as_slice())?;
        drop(pinned);
        return Ok(buffer);
    }
    let chunk = pinned
        .chunk
        .take()
        .expect("pinned buffer is not consumed yet");
    let len = chunk.len();
    let _scope = domain.scope()?;
    let stream = domain.stream();
    domain.retirement().during_admission(|admission| {
        admission.ensure_capacity(stream, len, 1)?;
        // SAFETY: destination was allocated for `len` bytes; the pinned chunk is
        // owned here and stays alive through the retirement epoch.
        let copied = unsafe {
            drv::memcpy_htod_async(
                buffer.device_ptr(),
                std::slice::from_raw_parts(chunk.as_ptr(), len),
                stream,
            )
        };
        if let Err(error) = copied {
            admission.abandon_chunk(chunk);
            return Err(Error::from_driver("cuMemcpyHtoDAsync", error));
        }
        if let Err(error) = fault_point(FaultPoint::AfterChunkEnqueue) {
            admission.abandon_chunk(chunk);
            domain.mark_broken(error.to_string());
            return Err(error);
        }
        if let Err((error, chunk)) = admission.commit_chunk(stream, chunk, None, Vec::new()) {
            admission.abandon_chunk(chunk);
            domain.mark_broken(error.to_string());
            return Err(error);
        }
        Ok(())
    })?;
    Ok(buffer)
}

/// Queue a device-to-host download into runtime-owned staging.
pub(crate) fn download(domain: &Arc<Domain>, buffer: &Buffer) -> Result<PendingDownload> {
    check_domain(domain, buffer.alloc().domain())?;
    if let Some(detail) = domain.broken() {
        return Err(Error::Broken { detail });
    }
    let slot = DownloadSlot::new(buffer.len());
    if buffer.is_empty() {
        return Ok(PendingDownload {
            slot,
            domain: Arc::clone(domain),
            _not_sync: std::marker::PhantomData,
        });
    }
    let _scope = domain.scope()?;
    let stream = domain.stream();
    domain.retirement().during_admission(|admission| {
        let mut offset = 0usize;
        while offset < buffer.len() {
            let want = (buffer.len() - offset)
                .min(admission.chunk_limit())
                .min(admission.available_host_bytes().max(1));
            let chunk = admission.take_chunk(domain.holder(), want)?;
            let len = want.min(chunk.len());
            admission.ensure_capacity(stream, chunk.len(), 1)?;
            // SAFETY: the source range was checked against the buffer length and
            // the chunk can hold `len` bytes; the chunk outlives the DMA through
            // the retirement epoch.
            // SAFETY: raw driver call; no Rust reference is created over the
            // uninitialised pinned staging.
            let copied = unsafe {
                sys::cuMemcpyDtoHAsync_v2(
                    chunk.as_ptr().cast(),
                    buffer.device_ptr() + offset as u64,
                    len,
                    stream,
                )
            }
            .result();
            if let Err(error) = copied {
                admission.abandon_chunk(chunk);
                return Err(Error::from_driver("cuMemcpyDtoHAsync", error));
            }
            if let Err(error) = fault_point(FaultPoint::AfterChunkEnqueue) {
                admission.abandon_chunk(chunk);
                domain.mark_broken(error.to_string());
                return Err(error);
            }
            let target = download_target(&slot, offset, len);
            if let Err((error, chunk)) =
                admission.commit_chunk(stream, chunk, Some(target), Vec::new())
            {
                admission.abandon_chunk(chunk);
                domain.mark_broken(error.to_string());
                return Err(error);
            }
            offset += len;
        }
        Ok(())
    })?;
    Ok(PendingDownload {
        slot,
        domain: Arc::clone(domain),
        _not_sync: std::marker::PhantomData,
    })
}

/// A pending host download.
///
/// The handle is an observation handle only. Dropping or forgetting it cannot
/// free staging that CUDA still writes to, and cannot let CUDA write into freed
/// host memory.
pub struct PendingDownload {
    slot: Arc<DownloadSlot>,
    domain: Arc<Domain>,
    _not_sync: std::marker::PhantomData<std::cell::Cell<()>>,
}

// SAFETY: the handle owns no submission authority; all CUDA work was queued on
// the owning stream before it was created.
unsafe impl Send for PendingDownload {}

impl PendingDownload {
    /// Non-blocking observation.
    ///
    /// Returns `Some(bytes)` once every byte is initialised, `None` while work is
    /// still pending, and a typed error if the transfer or an upstream dependency
    /// failed.
    pub fn poll(&mut self) -> Result<Option<Vec<u8>>> {
        // A derived result is only ready once its upstream sources are confirmed.
        if !self.domain.observe_gated()? {
            return Ok(None);
        }
        let mut state = self.slot.state.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(detail) = &state.failed {
            return Err(Error::Async {
                detail: detail.clone(),
            });
        }
        if state.done < state.data.len() {
            return Ok(None);
        }
        if state.taken {
            return Err(Error::invalid("download result was already observed"));
        }
        state.taken = true;
        let data = std::mem::take(&mut state.data);
        // SAFETY: every byte was reported initialised by a completed DMA.
        Ok(Some(unsafe { assume_init_vec(data) }))
    }

    /// Observe until the download completes.
    pub fn wait(&mut self) -> Result<Vec<u8>> {
        self.domain.retirement().note_host_wait();
        loop {
            if let Some(bytes) = self.poll()? {
                return Ok(bytes);
            }
            std::thread::yield_now();
        }
    }
}

unsafe fn assume_init_vec(data: Vec<std::mem::MaybeUninit<u8>>) -> Vec<u8> {
    let mut data = std::mem::ManuallyDrop::new(data);
    // SAFETY: `Vec<MaybeUninit<u8>>` and `Vec<u8>` have identical layout, and the
    // caller proved every element initialised.
    unsafe { Vec::from_raw_parts(data.as_mut_ptr().cast::<u8>(), data.len(), data.capacity()) }
}

/// An observation handle for an admitted copy.
///
/// Consumers never need to keep it: dependencies are also recorded in the
/// destination runtime's retirement state.
pub struct Completion {
    domain: Arc<Domain>,
    event: sys::CUevent,
    done: bool,
    _not_sync: std::marker::PhantomData<std::cell::Cell<()>>,
}

// SAFETY: the handle owns no submission authority.
unsafe impl Send for Completion {}

impl Completion {
    fn new(domain: Arc<Domain>, event: sys::CUevent) -> Self {
        Self {
            domain,
            event,
            done: false,
            _not_sync: std::marker::PhantomData,
        }
    }

    /// Non-blocking observation of the destination completion.
    pub fn poll(&mut self) -> Result<Option<()>> {
        if !self.domain.observe_gated()? {
            return Ok(None);
        }
        if self.done {
            return Ok(Some(()));
        }
        let _scope = self.domain.scope()?;
        // SAFETY: the event was created for this domain's context.
        match unsafe { drv::event::query(self.event) } {
            Ok(()) => {
                self.done = true;
                Ok(Some(()))
            }
            Err(error) if error.0 == sys::CUresult::CUDA_ERROR_NOT_READY => Ok(None),
            Err(error) => Err(Error::Async {
                detail: format!("copy completion: {error}"),
            }),
        }
    }

    /// Observe until the copy completes.
    pub fn wait(&mut self) -> Result<()> {
        self.domain.retirement().note_host_wait();
        loop {
            if self.poll()?.is_some() {
                return Ok(());
            }
            std::thread::yield_now();
        }
    }
}

impl Drop for Completion {
    fn drop(&mut self) {
        // SAFETY: the event is owned here. A failure to activate the context
        // leaks the handle rather than acting on a stale one.
        if let Ok(_scope) = self.domain.scope() {
            unsafe {
                let _ = drv::event::destroy(self.event);
            }
        }
    }
}

/// Copy `source` into a new buffer owned by `dst`.
///
/// Both endpoints participate in admission: the source gets the dependency it
/// needs before any later write or free, and the destination gets one before its
/// consumers run. No implicit transfer happens anywhere else.
pub(crate) fn copy_between(
    src: &Arc<Domain>,
    dst: &Arc<Domain>,
    source: &Buffer,
    force_staged: bool,
) -> Result<(Buffer, Completion, CopyRoute)> {
    check_domain(src, source.alloc().domain())?;
    if Arc::ptr_eq(src, dst) {
        return Err(Error::invalid(
            "copy_between requires two distinct allocation domains",
        ));
    }
    for domain in [src, dst] {
        if let Some(detail) = domain.broken() {
            return Err(Error::Broken { detail });
        }
    }
    let len = source.len();
    let alignment = 256;
    let alloc = allocate_buffer(dst, len, alignment)?;
    let out = Buffer::from_parts(alloc, len, alignment);
    // Upstream dependencies of the source are inherited by every witness this
    // copy exports, so an A -> B -> C chain cannot report C ready without A.
    let upstream = src.retirement().frontier_snapshot();
    let _scope = dst.scope()?;
    let (event, route) = dst.retirement().during_admission(|admission| {
        // Same-device copies retain two witnesses; staged copies retain one
        // witness plus one chunk epoch per chunk. Reserve both before enqueue.
        if len > 0 {
            admission.ensure_capacity(dst.stream(), 0, 2)?;
        }
        let route = if len == 0 {
            CopyRoute::SameDevice
        } else if src.ordinal() == dst.ordinal() && !force_staged {
            copy_same_device(src, dst, source, &out, admission, &upstream)?
        } else {
            copy_staged(src, dst, source, &out, admission, &upstream)?
        };
        let event = record_event(dst.stream())?;
        Ok((event, route))
    })?;
    Ok((out, Completion::new(Arc::clone(dst), event), route))
}

fn copy_same_device(
    src: &Arc<Domain>,
    dst: &Arc<Domain>,
    source: &Buffer,
    out: &Buffer,
    admission: &crate::retirement::Admission<'_>,
    upstream: &[Arc<Witness>],
) -> Result<CopyRoute> {
    let ready = {
        let _src_scope = src.scope()?;
        record_event(src.stream())?
    };
    // SAFETY: the event belongs to the shared primary context and both endpoints
    // participate in this admission.
    if let Err(error) = unsafe {
        drv::stream::wait_event(
            dst.stream(),
            ready,
            sys::CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT,
        )
    } {
        src.quarantine(source.device_ptr());
        src.mark_broken("cuStreamWaitEvent failed during copy admission");
        // SAFETY: nothing else references this event.
        unsafe {
            let _ = drv::event::destroy(ready);
        }
        return Err(Error::from_driver("cuStreamWaitEvent", error));
    }
    // SAFETY: both ranges were checked; both pointers live in the same device and
    // the shared primary context is current.
    if let Err(error) = unsafe {
        drv::memcpy_dtod_async(
            out.device_ptr(),
            source.device_ptr(),
            source.len(),
            dst.stream(),
        )
    } {
        src.quarantine(source.device_ptr());
        src.mark_broken("cuMemcpyDtoDAsync failed during copy admission");
        return Err(Error::from_driver("cuMemcpyDtoDAsync", error));
    }
    // Every post-enqueue failure must still order the source behind the copy.
    if let Err(error) = fault_point(FaultPoint::BeforeEventRecord) {
        src.quarantine(source.device_ptr());
        src.mark_broken(error.to_string());
        return Err(error);
    }
    let done = match record_event(dst.stream()) {
        Ok(event) => event,
        Err(error) => {
            src.quarantine(source.device_ptr());
            src.mark_broken(error.to_string());
            return Err(error);
        }
    };
    if let Err(error) = fault_point(FaultPoint::BeforeSourceBackWait) {
        src.quarantine(source.device_ptr());
        src.mark_broken(error.to_string());
        return Err(error);
    }
    let source_scope = src.scope();
    let back_wait = match source_scope {
        Err(error) => Err(error),
        Ok(_src_scope) => {
            // SAFETY: CUDA permits waiting on an event from another context on the
            // same device.
            unsafe {
                drv::stream::wait_event(
                    src.stream(),
                    done,
                    sys::CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT,
                )
            }
            .map_err(Error::driver("cuStreamWaitEvent"))
        }
    };
    if let Err(error) = back_wait {
        src.quarantine(source.device_ptr());
        src.mark_broken("source back-wait failed after copy admission");
        return Err(error);
    }
    // Retain both events through the destination frontier: `ready` is the
    // source-ordering witness (same primary context) and `done` must outlive the
    // source's back-wait.
    for (holder, event, label) in [
        (src.holder(), ready, "same-device source frontier"),
        (dst.holder(), done, "same-device copy completion"),
    ] {
        let witness = Witness::new(holder, event, label, upstream.to_vec());
        admission.add_witness(witness)?;
    }
    Ok(CopyRoute::SameDevice)
}

fn copy_staged(
    src: &Arc<Domain>,
    dst: &Arc<Domain>,
    source: &Buffer,
    out: &Buffer,
    admission: &crate::retirement::Admission<'_>,
    upstream: &[Arc<Witness>],
) -> Result<CopyRoute> {
    let mut offset = 0usize;
    let chunk_limit = admission.chunk_limit();
    while offset < source.len() {
        let want = (source.len() - offset)
            .min(chunk_limit)
            .min(admission.available_host_bytes().max(1));
        let chunk = admission.take_chunk(dst.holder(), want)?;
        let len = want.min(chunk.len());
        admission.ensure_capacity(dst.stream(), chunk.len(), 2)?;
        let staged = {
            let _src_scope = src.scope()?;
            // SAFETY: the source range was checked; the chunk can hold `len`
            // bytes and outlives the DMA through the destination epoch.
            // SAFETY: raw driver call — the chunk is uninitialised pinned storage
            // and only the driver writes it.
            let copied = unsafe {
                sys::cuMemcpyDtoHAsync_v2(
                    chunk.as_ptr().cast(),
                    source.device_ptr() + offset as u64,
                    len,
                    src.stream(),
                )
            }
            .result();
            if let Err(error) = copied {
                admission.abandon_chunk(chunk);
                return Err(Error::from_driver("cuMemcpyDtoHAsync", error));
            }
            match record_event(src.stream()) {
                Ok(event) => event,
                Err(error) => {
                    src.quarantine(source.device_ptr());
                    src.mark_broken(error.to_string());
                    admission.abandon_chunk(chunk);
                    return Err(error);
                }
            }
        };
        let witness = Witness::new(
            src.holder(),
            staged,
            "staged source chunk",
            upstream.to_vec(),
        );
        // SAFETY: `staged` belongs to the source context and CUDA permits a
        // cross-device stream wait.
        if let Err(error) = unsafe {
            drv::stream::wait_event(
                dst.stream(),
                staged,
                sys::CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT,
            )
        } {
            src.quarantine(source.device_ptr());
            src.mark_broken("cross-device cuStreamWaitEvent failed");
            admission.abandon_chunk(chunk);
            return Err(Error::from_driver("cuStreamWaitEvent", error));
        }
        // SAFETY: raw driver call; the destination range was checked and the
        // chunk holds `len` bytes written by the source D2H.
        let copied = unsafe {
            sys::cuMemcpyHtoDAsync_v2(
                out.device_ptr() + offset as u64,
                chunk.as_ptr().cast(),
                len,
                dst.stream(),
            )
        }
        .result();
        if let Err(error) = copied {
            src.quarantine(source.device_ptr());
            src.mark_broken("staged cuMemcpyHtoDAsync failed");
            admission.abandon_chunk(chunk);
            return Err(Error::from_driver("cuMemcpyHtoDAsync", error));
        }
        let witnesses = vec![Arc::clone(&witness)];
        if let Err((error, chunk)) = admission.commit_chunk(dst.stream(), chunk, None, witnesses) {
            // The destination H2D may still read this chunk; leak it and keep the
            // source witness alive so its event is not destroyed under a wait.
            admission.abandon_chunk(chunk);
            std::mem::forget(witness);
            src.quarantine(source.device_ptr());
            dst.mark_broken(error.to_string());
            return Err(error);
        }
        admission.add_witness(witness)?;
        offset += len;
    }
    Ok(CopyRoute::Staged)
}

#[cfg(test)]
mod tests;
