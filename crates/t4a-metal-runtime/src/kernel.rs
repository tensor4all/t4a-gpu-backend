//! Bound functions, launch configuration and checked argument marshalling.
//!
//! Launching a consumer kernel is `unsafe`: MSL has no reflection, so no descriptor
//! check can prove a kernel's argument count, the type behind each buffer index, or
//! that it fully overwrites its outputs.

use std::sync::Arc;

use t4a_metal_sys::objc::{Id, Pool};

use crate::context::ContextHolder;
use crate::error::{Error, Result};
use crate::memory::{Buffer, UninitBuffer};
use crate::submission::Session;

/// Largest by-value parameter `setBytes:length:atIndex:` accepts.
///
/// Metal documents a 4 KB limit for the bytes a single index can carry. A larger
/// structure must be uploaded into a buffer and passed by index.
pub const MAX_PARAMETER_BYTES: usize = 4096;

/// A buffer binding offset must be a multiple of this.
///
/// `a_binding_offset_selects_the_tail` binds at offset 4 on real hardware, because the
/// value is a measurement of what Metal accepts and not a documented guarantee.
const BINDING_ALIGN: usize = 4;

/// Checked dispatch dimensions.
///
/// A dispatch runs `threadgroups × threads_per_threadgroup` threads and Metal does not
/// bound-check an index, so the grid decides which elements are touched: pick
/// [`LaunchConfig::threads`] or [`LaunchConfig::tiled`] by whether the kernel guards.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaunchConfig {
    /// Threadgroup counts, each non-zero.
    pub threadgroups: (u32, u32, u32),
    /// Threads per threadgroup, each non-zero.
    pub threads_per_threadgroup: (u32, u32, u32),
}

impl LaunchConfig {
    /// A one-dimensional dispatch of exactly `threads` threads.
    ///
    /// One thread per threadgroup, so the grid covers every index exactly once and a
    /// kernel that indexes a buffer of `threads` elements needs no bounds guard. Any
    /// thread count is legal, including one above the pipeline's per-threadgroup limit,
    /// which this form never reaches.
    pub fn threads(threads: u32) -> Self {
        Self {
            threadgroups: (threads, 1, 1),
            threads_per_threadgroup: (1, 1, 1),
        }
    }

    /// A one-dimensional tiled dispatch of `threadgroups × threads_per_threadgroup`
    /// threads.
    ///
    /// This is the throughput form: a group width of 32 or more fills the SIMD lanes a
    /// one-thread group would leave idle. The grid is the product, so when the element
    /// count is not a multiple of the group width the kernel must guard its last group
    /// (`if (index >= count) { return; }`); otherwise the extra threads write past the
    /// buffer, which Metal does not check.
    pub fn tiled(threadgroups: u32, threads_per_threadgroup: u32) -> Self {
        Self {
            threadgroups: (threadgroups, 1, 1),
            threads_per_threadgroup: (threads_per_threadgroup, 1, 1),
        }
    }

    fn validate(&self, max_threads_per_threadgroup: u64) -> Result<()> {
        let (gx, gy, gz) = self.threadgroups;
        if gx == 0 || gy == 0 || gz == 0 {
            return Err(Error::invalid("threadgroup counts must be non-zero"));
        }
        let (tx, ty, tz) = self.threads_per_threadgroup;
        if tx == 0 || ty == 0 || tz == 0 {
            return Err(Error::invalid("threads per threadgroup must be non-zero"));
        }
        let total = u64::from(tx) * u64::from(ty) * u64::from(tz);
        if total > max_threads_per_threadgroup {
            return Err(Error::invalid(format!(
                "this pipeline accepts at most {max_threads_per_threadgroup} threads per \
                 threadgroup, but the launch asks for {total}"
            )));
        }
        Ok(())
    }
}

/// One argument of a kernel launch.
///
/// The `i`-th argument binds to buffer index `i`, so MSL declares its parameters as
/// `[[buffer(0)]]`, `[[buffer(1)]]`, and so on, in the same order.
#[derive(Debug, Clone, Copy)]
pub enum Arg<'a> {
    /// 32-bit unsigned scalar.
    U32(u32),
    /// 64-bit unsigned scalar.
    U64(u64),
    /// 32-bit signed scalar.
    I32(i32),
    /// 32-bit float scalar.
    F32(f32),
    /// A device buffer in the launching runtime's domain, bound at `offset`.
    Buffer {
        /// The buffer to bind.
        buffer: &'a Buffer,
        /// Byte offset into the buffer; a multiple of 4 and inside it.
        offset: usize,
    },
    /// A buffer whose contents are not yet initialised.
    ///
    /// The kernel must fully overwrite it before it is converted with
    /// [`UninitBuffer::assume_init`].
    BufferUninit {
        /// The buffer to bind.
        buffer: &'a UninitBuffer,
        /// Byte offset into the buffer; a multiple of 4 and inside it.
        offset: usize,
    },
    /// A by-value C value, passed in the kernel's constant address space.
    ///
    /// `setBytes:length:atIndex:` copies the bytes into the command stream, so
    /// `bytes` may be reused or dropped as soon as the launch returns.
    Bytes {
        /// The value's bytes, in the kernel's C layout.
        bytes: &'a [u8],
    },
}

impl<'a> Arg<'a> {
    /// Wrap one POD value as a by-value parameter.
    pub fn pod<T: Pod>(value: &'a T) -> Self {
        let size = std::mem::size_of::<T>();
        // SAFETY: `Pod` guarantees plain data, so these `size` bytes are readable
        // and the kernel reads them as one C value.
        let bytes = unsafe { std::slice::from_raw_parts((value as *const T).cast::<u8>(), size) };
        Self::Bytes { bytes }
    }
}

/// A by-value kernel parameter: plain data the kernel reads as one C value.
///
/// # Safety
///
/// `Self` must be `#[repr(C)]` plain data with no padding and only initialised
/// bytes (declare explicit padding fields), and no references or interior pointers
/// the kernel would dereference: [`Arg::pod`] reads every byte of the value.
pub unsafe trait Pod: Copy {}

// SAFETY: each type below is plain data with no padding and no references, so reading
// its bytes as the kernel's C value is well defined. Dtype policy is not this layer's:
// a kernel that declares a type MSL rejects fails to compile, which is the measurement
// the probe records.
macro_rules! pod_scalars {
    ($($ty:ty),* $(,)?) => {
        $(
            // SAFETY: plain data, as above.
            unsafe impl Pod for $ty {}
        )*
    };
}

pod_scalars!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);

/// Where one argument's bytes come from.
enum Resolved<'a> {
    Word { bytes: [u8; 8], len: usize },
    Buffer { handle: Id, offset: u64 },
    Bytes(&'a [u8]),
}

/// One compute pipeline state, shared by the runtime's cache and every bound
/// function made from it.
pub(crate) struct PipelineState {
    context: Arc<ContextHolder>,
    pipeline: Id,
    max_threads: u64,
}

// SAFETY: `MTLComputePipelineState` is an immutable, thread-safe Metal object.
unsafe impl Send for PipelineState {}
unsafe impl Sync for PipelineState {}

impl PipelineState {
    pub(crate) fn max_threads(&self) -> u64 {
        self.max_threads
    }
}

impl Drop for PipelineState {
    fn drop(&mut self) {
        // SAFETY: the `+1` pipeline-state reference from the bind, released once after
        // the last bound function and cache entry let go. Encoded command buffers
        // retain the state themselves.
        unsafe { self.context.table.objc.release(self.pipeline) };
    }
}

impl std::fmt::Debug for PipelineState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PipelineState")
            .field("domain_id", &self.context.domain_id())
            .field("max_threads", &self.max_threads)
            .finish()
    }
}

/// A function bound to one runtime and one pipeline state.
///
/// It carries no submission authority: launching needs `&mut Runtime`.
pub struct BoundFunction {
    state: Arc<PipelineState>,
    entry: String,
    source_hash: String,
}

// SAFETY: `MTLComputePipelineState` is an immutable, thread-safe Metal object, and
// the handle carries no submission authority.
unsafe impl Send for BoundFunction {}
unsafe impl Sync for BoundFunction {}

impl BoundFunction {
    pub(crate) fn new(state: Arc<PipelineState>, entry: String, source_hash: String) -> Self {
        Self {
            state,
            entry,
            source_hash,
        }
    }

    /// The MSL entry point name.
    pub fn entry(&self) -> &str {
        &self.entry
    }

    /// The allocation-domain identity this function belongs to.
    pub fn domain_id(&self) -> u64 {
        self.state.context.domain_id()
    }

    /// The source hash of the artifact this function was bound from.
    pub fn source_hash(&self) -> &str {
        &self.source_hash
    }

    /// The largest thread count this pipeline accepts per threadgroup.
    pub fn max_total_threads_per_threadgroup(&self) -> u64 {
        self.state.max_threads()
    }

    pub(crate) fn context(&self) -> &Arc<ContextHolder> {
        &self.state.context
    }

    pub(crate) fn pipeline(&self) -> Id {
        self.state.pipeline
    }
}

impl std::fmt::Debug for BoundFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BoundFunction")
            .field("entry", &self.entry)
            .field("domain_id", &self.state.context.domain_id())
            .field("max_threads", &self.state.max_threads())
            .finish()
    }
}

/// Build a pipeline state for `entry` in `artifact`.
pub(crate) fn create_pipeline(
    context: &Arc<ContextHolder>,
    artifact: &crate::compile::Artifact,
    entry: &str,
) -> Result<PipelineState> {
    let table = context.table;
    let selectors = table.selectors;
    let pool = Pool::push(table.objc);
    let Some(name) = table.new_string(entry) else {
        drop(pool);
        return Err(Error::invalid(format!(
            "the entry point name `{entry}` could not become an NSString"
        )));
    };
    // SAFETY: the library, the name string and the runtime are live; the call
    // returns a `+1` function or nil.
    let function = unsafe {
        selectors
            .library_new_function
            .send(artifact.library(), name)
    };
    if function.is_null() {
        // SAFETY: releases the `+1` name taken above.
        unsafe { table.objc.release(name) };
        drop(pool);
        return Err(Error::UnknownEntryPoint {
            name: entry.to_string(),
        });
    }
    let mut error: Id = std::ptr::null_mut();
    // SAFETY: the function and the error slot are live; the call returns a `+1`
    // pipeline state or nil with `error` set.
    let pipeline = unsafe {
        selectors
            .device_new_pipeline
            .send(context.device(), function, &mut error as *mut Id)
    };
    let detail = if pipeline.is_null() {
        // SAFETY: `error` is null or a live `NSError` inside the open pool.
        unsafe { t4a_metal_sys::objc::error_detail(&selectors, error) }
    } else {
        None
    };
    // SAFETY: releases the `+1` objects taken above; the pipeline state retains its
    // function, and the library retains its functions.
    unsafe {
        table.objc.release(function);
        table.objc.release(name);
    }
    drop(pool);
    if pipeline.is_null() {
        return Err(Error::Pipeline {
            entry: entry.to_string(),
            detail: detail.unwrap_or_else(|| "Metal reported no diagnostic".to_string()),
        });
    }
    // SAFETY: `pipeline` is a live `MTLComputePipelineState` and the selector returns
    // a scalar.
    let max_threads = unsafe { selectors.pipeline_max_threads.send(pipeline) };
    Ok(PipelineState {
        context: Arc::clone(context),
        pipeline,
        max_threads,
    })
}

/// Validate the arguments and encode one dispatch into the runtime's session.
pub(crate) fn encode(
    context: &Arc<ContextHolder>,
    session: &mut Option<Session>,
    function: &BoundFunction,
    config: LaunchConfig,
    args: &[Arg<'_>],
) -> Result<()> {
    if !Arc::ptr_eq(function.context(), context) {
        return Err(Error::ForeignDomain);
    }
    config.validate(function.max_total_threads_per_threadgroup())?;
    let mut resolved = Vec::with_capacity(args.len());
    for arg in args {
        resolved.push(resolve(context, arg)?);
    }

    if session.is_none() {
        *session = Some(Session::new(Arc::clone(context))?);
    }
    let Some(session) = session.as_mut() else {
        return Err(Error::Closed);
    };
    let encoder = session.ensure_encoder()?;
    let selectors = context.table.selectors;
    // SAFETY: the encoder is open, the pipeline and every buffer handle are live
    // Metal objects, and each argument's index is its position in `args`, which is
    // the binding the caller's MSL declares.
    unsafe {
        selectors
            .encoder_set_pipeline
            .send(encoder, function.pipeline());
        for (index, arg) in resolved.iter().enumerate() {
            let index = index as u64;
            match arg {
                Resolved::Word { bytes, len } => selectors.encoder_set_bytes.send(
                    encoder,
                    bytes.as_ptr().cast(),
                    *len as u64,
                    index,
                ),
                Resolved::Buffer { handle, offset } => selectors
                    .encoder_set_buffer
                    .send(encoder, *handle, *offset, index),
                Resolved::Bytes(bytes) => selectors.encoder_set_bytes.send(
                    encoder,
                    bytes.as_ptr().cast(),
                    bytes.len() as u64,
                    index,
                ),
            }
        }
        let (gx, gy, gz) = config.threadgroups;
        let (tx, ty, tz) = config.threads_per_threadgroup;
        selectors.encoder_dispatch_threadgroups.send(
            encoder,
            t4a_metal_sys::objc::Size::new(gx.into(), gy.into(), gz.into()),
            t4a_metal_sys::objc::Size::new(tx.into(), ty.into(), tz.into()),
        );
    }
    Ok(())
}

fn resolve<'a>(context: &Arc<ContextHolder>, arg: &Arg<'a>) -> Result<Resolved<'a>> {
    Ok(match *arg {
        Arg::U32(value) => word(value.to_ne_bytes().as_slice(), 4),
        Arg::U64(value) => word(value.to_ne_bytes().as_slice(), 8),
        Arg::I32(value) => word(value.to_ne_bytes().as_slice(), 4),
        Arg::F32(value) => word(value.to_bits().to_ne_bytes().as_slice(), 4),
        Arg::Buffer { buffer, offset } => checked_binding(
            context,
            buffer.context(),
            buffer.handle(),
            buffer.len(),
            buffer.is_empty(),
            offset,
        )?,
        Arg::BufferUninit { buffer, offset } => checked_binding(
            context,
            buffer.context(),
            buffer.handle(),
            buffer.len(),
            buffer.is_empty(),
            offset,
        )?,
        Arg::Bytes { bytes } => {
            if bytes.is_empty() {
                return Err(Error::invalid("a by-value parameter must have a size"));
            }
            if bytes.len() > MAX_PARAMETER_BYTES {
                return Err(Error::KernelParameterLimit {
                    bytes: bytes.len(),
                    limit: MAX_PARAMETER_BYTES,
                });
            }
            Resolved::Bytes(bytes)
        }
    })
}

fn word(bytes: &[u8], len: usize) -> Resolved<'static> {
    let mut value = [0u8; 8];
    value[..len].copy_from_slice(&bytes[..len]);
    Resolved::Word { bytes: value, len }
}

#[allow(clippy::too_many_arguments)]
fn checked_binding(
    context: &Arc<ContextHolder>,
    owner: &Arc<ContextHolder>,
    handle: Id,
    len: usize,
    empty: bool,
    offset: usize,
) -> Result<Resolved<'static>> {
    if !Arc::ptr_eq(owner, context) {
        return Err(Error::ForeignDomain);
    }
    if empty {
        return Err(Error::invalid(
            "an empty buffer has no address to bind a kernel to",
        ));
    }
    if offset >= len {
        return Err(Error::OutOfRange {
            len,
            offset,
            required: 1,
        });
    }
    if offset % BINDING_ALIGN != 0 {
        return Err(Error::invalid(format!(
            "a buffer binding offset must be a multiple of {BINDING_ALIGN}, got {offset}"
        )));
    }
    Ok(Resolved::Buffer {
        handle,
        offset: offset as u64,
    })
}

#[cfg(test)]
mod tests;
