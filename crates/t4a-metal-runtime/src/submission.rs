//! Command-buffer submission and completion.
//!
//! A launch encodes into one compute encoder over one command buffer. Committing
//! closes the encoder, encodes a signal on a fresh [`MTLSharedEvent`] and submits.
//! Each submission signals its own event, so waiting for one never infers anything
//! about another: a command queue orders execution, and the runtime does not read
//! that as a guarantee about which submission finished first.

use std::any::Any;
use std::sync::Arc;
use std::time::{Duration, Instant};

use t4a_metal_sys::objc::{error_detail, Id, Pool};

use crate::context::ContextHolder;
use crate::error::{Error, Result};

/// The value each submission writes to its own event.
const SIGNAL_VALUE: u64 = 1;

/// How long to wait before the first re-check of a pending submission.
const FIRST_BACKOFF: Duration = Duration::from_micros(50);

/// The largest backoff between re-checks.
const MAX_BACKOFF: Duration = Duration::from_millis(2);

/// One open command buffer and its encoder.
pub(crate) struct Session {
    context: Arc<ContextHolder>,
    command_buffer: Id,
    encoder: Option<Id>,
    keepalives: Vec<Box<dyn Any + Send>>,
}

impl Session {
    /// Open a command buffer on the runtime's queue.
    pub(crate) fn new(context: Arc<ContextHolder>) -> Result<Self> {
        let table = context.table;
        let pool = Pool::push(table.objc);
        // SAFETY: the queue is live; `commandBuffer` returns an autoreleased buffer.
        let command_buffer = unsafe { table.selectors.queue_command_buffer.send(context.queue()) };
        if command_buffer.is_null() {
            drop(pool);
            return Err(Error::unavailable(
                "no Metal command buffer",
                "the command queue returned nil for `commandBuffer`",
            ));
        }
        // SAFETY: the buffer is live inside the open pool; the extra reference
        // outlives the pool.
        let retained = unsafe { table.objc.retain(command_buffer) };
        drop(pool);
        Ok(Self {
            context,
            command_buffer: retained,
            encoder: None,
            keepalives: Vec::new(),
        })
    }

    /// The command buffer handle.
    pub(crate) fn command_buffer(&self) -> Id {
        self.command_buffer
    }

    /// The open compute encoder, opening one if needed.
    pub(crate) fn ensure_encoder(&mut self) -> Result<Id> {
        if let Some(encoder) = self.encoder {
            return Ok(encoder);
        }
        let table = self.context.table;
        let pool = Pool::push(table.objc);
        // SAFETY: the command buffer is live; the encoder is autoreleased.
        let encoder = unsafe {
            table
                .selectors
                .command_buffer_compute_encoder
                .send(self.command_buffer)
        };
        if encoder.is_null() {
            drop(pool);
            return Err(Error::unavailable(
                "no compute encoder",
                "the command buffer returned nil for `computeCommandEncoder`",
            ));
        }
        // SAFETY: the encoder is live inside the open pool.
        let retained = unsafe { table.objc.retain(encoder) };
        drop(pool);
        self.encoder = Some(retained);
        Ok(retained)
    }

    /// Close the open encoder, if any.
    ///
    /// Anything encoded after this goes into a new encoder, which is also what lets
    /// the raw seam open an encoder of another type.
    pub(crate) fn end_encoder(&mut self) {
        if let Some(encoder) = self.encoder.take() {
            // SAFETY: the encoder was opened by this session and is released once.
            unsafe {
                self.context
                    .table
                    .selectors
                    .encoder_end_encoding
                    .send(encoder);
                self.context.table.objc.release(encoder);
            }
        }
    }

    /// Pin a foreign resource until this submission completes.
    pub(crate) fn retain_keepalive(&mut self, keepalive: Box<dyn Any + Send>) {
        self.keepalives.push(keepalive);
    }

    /// End the encoder, signal a fresh event, submit, and hand off the buffer.
    pub(crate) fn finish(mut self, timeout_ms: u64) -> Result<Submission> {
        self.end_encoder();
        let table = self.context.table;
        let selectors = table.selectors;
        let pool = Pool::push(table.objc);
        // SAFETY: the device is live; `newSharedEvent` returns a `+1` event or nil.
        let event = unsafe {
            selectors
                .device_new_shared_event
                .send(self.context.device())
        };
        if event.is_null() {
            drop(pool);
            return Err(Error::unavailable(
                "no Metal shared event",
                "the device returned nil for `newSharedEvent`",
            ));
        }
        // SAFETY: the command buffer and the event are live; the signal is encoded
        // before the commit and the buffer retains the event.
        unsafe {
            selectors
                .command_buffer_encode_signal
                .send(self.command_buffer, event, SIGNAL_VALUE);
            selectors.command_buffer_commit.send(self.command_buffer);
        }
        drop(pool);
        let command_buffer = std::mem::replace(&mut self.command_buffer, std::ptr::null_mut());
        let keepalives = std::mem::take(&mut self.keepalives);
        Ok(Submission {
            context: Arc::clone(&self.context),
            command_buffer,
            event,
            keepalives,
            timeout_ms,
        })
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.end_encoder();
        if !self.command_buffer.is_null() {
            // SAFETY: this is the `+1` command buffer reference taken in `new` or
            // moved out by `finish`, released once.
            unsafe { self.context.table.objc.release(self.command_buffer) };
        }
    }
}

/// One submitted command buffer.
pub(crate) struct Submission {
    context: Arc<ContextHolder>,
    command_buffer: Id,
    event: Id,
    /// Foreign resources that must outlive this submission.
    #[allow(dead_code)]
    keepalives: Vec<Box<dyn Any + Send>>,
    timeout_ms: u64,
}

// SAFETY: `MTLCommandBuffer` and `MTLSharedEvent` are thread-safe Metal objects, and
// every mutating use of a submission happens before it is wrapped in `Completion`;
// observation only reads two scalars and one error object.
unsafe impl Send for Submission {}
unsafe impl Sync for Submission {}

impl Submission {
    fn poll(&self) -> Result<bool> {
        let table = self.context.table;
        let selectors = table.selectors;
        // SAFETY: `event` is a live `MTLSharedEvent` and the selector returns a
        // scalar.
        let signalled = unsafe { selectors.shared_event_signaled_value.send(self.event) };
        if signalled < SIGNAL_VALUE {
            return Ok(false);
        }
        let pool = Pool::push(table.objc);
        // SAFETY: the command buffer is live and its `error` is valid until it is
        // released, which this handle prevents.
        let error = unsafe { selectors.command_buffer_error.send(self.command_buffer) };
        // SAFETY: `error` is null or a live `NSError` inside the open pool.
        let detail = unsafe { error_detail(&selectors, error) };
        drop(pool);
        match detail {
            Some(detail) => Err(Error::async_failure(detail)),
            None => Ok(true),
        }
    }

    fn wait(&self) -> Result<()> {
        let start = Instant::now();
        let timeout = Duration::from_millis(self.timeout_ms);
        let mut backoff = FIRST_BACKOFF;
        loop {
            if self.poll()? {
                return Ok(());
            }
            if start.elapsed() >= timeout {
                return Err(Error::Timeout {
                    millis: self.timeout_ms,
                });
            }
            std::thread::sleep(backoff);
            backoff = (backoff * 2).min(MAX_BACKOFF);
        }
    }
}

impl Drop for Submission {
    fn drop(&mut self) {
        // SAFETY: both are the `+1` references taken for this submission, released
        // once. Metal retains the resources the buffer encoded, so dropping the
        // buffer handle here never releases anything still in use.
        unsafe {
            self.context.table.objc.release(self.event);
            self.context.table.objc.release(self.command_buffer);
        }
    }
}

/// An observation handle for one submission.
///
/// It does not keep the work alive in any other sense: dropping it cannot release a
/// resource the GPU still reads, because Metal retains what a command buffer
/// encoded.
#[derive(Clone)]
pub struct Completion {
    submission: Arc<Submission>,
}

impl Completion {
    pub(crate) fn new(submission: Submission) -> Self {
        Self {
            submission: Arc::new(submission),
        }
    }

    /// Whether the submission has completed, without blocking.
    ///
    /// # Errors
    ///
    /// [`Error::Async`] when the command buffer completed with an error.
    pub fn poll(&self) -> Result<bool> {
        self.submission.poll()
    }

    /// Wait for the submission, re-checking until it completes or the runtime's
    /// timeout elapses.
    ///
    /// # Errors
    ///
    /// [`Error::Async`] when the command buffer completed with an error, and
    /// [`Error::Timeout`] when it did not complete in time.
    pub fn wait(&self) -> Result<()> {
        self.submission.wait()
    }
}

impl std::fmt::Debug for Completion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Completion")
            .field("domain_id", &self.submission.context.domain_id())
            .finish_non_exhaustive()
    }
}
