//! Runtime-local modules, bound functions and checked kernel launches.
//!
//! A bound function belongs to exactly one runtime/domain. Launching arbitrary
//! consumer kernels is `unsafe`: descriptor checks cannot prove the kernel's ABI,
//! its memory accesses or that it fully overwrites its outputs.

use std::ffi::CString;
use std::ptr;
use std::sync::Arc;

use cudarc::driver::result::DriverError;
use cudarc::driver::{result as drv, sys};

use crate::compile::{Artifact, CompileKey};
use crate::context::ContextHolder;
use crate::error::{Error, Result};
use crate::memory::{Buffer, Domain, UninitBuffer};
use crate::retirement::Admission;

/// One loaded module, retained until every queued use has completed.
pub(crate) struct ModuleInner {
    holder: Arc<ContextHolder>,
    module: sys::CUmodule,
    bytes: usize,
    domain: std::sync::Weak<Domain>,
}

// SAFETY: a module handle is process-wide and context-scoped. Every use runs
// inside the owning domain's context scope on the using thread.
unsafe impl Send for ModuleInner {}
unsafe impl Sync for ModuleInner {}

impl ModuleInner {
    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }
}

impl Drop for ModuleInner {
    fn drop(&mut self) {
        if let Some(domain) = self.domain.upgrade() {
            domain.retirement().release_module_bytes(self.bytes);
        }
        // SAFETY: this is the last reference and every queued use has completed.
        // If the context cannot be activated, leak the module instead of acting
        // on a handle from an inactive context.
        if let Ok(_scope) = self.holder.activate() {
            unsafe {
                let _ = drv::module::unload(self.module);
            }
        }
    }
}

/// A function bound to one runtime.
pub struct BoundFunction {
    domain: Arc<Domain>,
    module: Arc<ModuleInner>,
    function: sys::CUfunction,
    entry: String,
    artifact_key: CompileKey,
}

// SAFETY: the handle carries no submission authority; launching needs
// `&mut Runtime` and is `unsafe`.
unsafe impl Send for BoundFunction {}
unsafe impl Sync for BoundFunction {}

impl BoundFunction {
    /// Entry point name.
    pub fn entry(&self) -> &str {
        &self.entry
    }

    /// Allocation-domain identity this function belongs to.
    pub fn domain_id(&self) -> u64 {
        self.domain.id()
    }

    /// Artifact this function was bound from.
    pub fn artifact_key(&self) -> CompileKey {
        self.artifact_key
    }
}

impl std::fmt::Debug for BoundFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BoundFunction")
            .field("entry", &self.entry)
            .field("domain_id", &self.domain.id())
            .finish()
    }
}

/// Checked launch dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaunchConfig {
    /// Grid dimensions; each must be non-zero.
    pub grid: (u32, u32, u32),
    /// Block dimensions; each must be non-zero.
    pub block: (u32, u32, u32),
    /// Dynamic shared memory in bytes.
    pub shared_bytes: u32,
}

impl LaunchConfig {
    /// Build a one-dimensional launch.
    pub fn new(grid: u32, block: u32) -> Self {
        Self {
            grid: (grid, 1, 1),
            block: (block, 1, 1),
            shared_bytes: 0,
        }
    }

    fn validate(&self) -> Result<()> {
        if self.grid.0 == 0 || self.grid.1 == 0 || self.grid.2 == 0 {
            return Err(Error::invalid("grid dimensions must be non-zero"));
        }
        if self.block.0 == 0 || self.block.1 == 0 || self.block.2 == 0 {
            return Err(Error::invalid("block dimensions must be non-zero"));
        }
        Ok(())
    }
}

/// A scalar, device-pointer or by-value parameter of one kernel launch.
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
    /// 64-bit float scalar.
    F64(f64),
    /// Device pointer of a buffer in the launching runtime's domain.
    DevicePtr(&'a Buffer),
    /// Device pointer of a buffer whose contents are not yet initialised.
    ///
    /// The kernel must fully overwrite it before it is converted with
    /// [`UninitBuffer::assume_init`].
    DevicePtrUninit(&'a UninitBuffer),
    /// A by-value C struct or array, passed in the kernel's const bank.
    ///
    /// The block is copied into the launch's parameter buffer at admission, so
    /// `bytes` may be reused or dropped as soon as the launch returns. `align`
    /// is the alignment the kernel's C type requires: a power of two of at most
    /// [`MAX_PARAMETER_ALIGN`].
    Bytes {
        /// The parameter's bytes, in the kernel's C layout.
        bytes: &'a [u8],
        /// Alignment the C type requires, in bytes.
        align: usize,
    },
}

impl<'a> Arg<'a> {
    /// Wrap one POD value as a by-value parameter, taking its size and alignment
    /// from `T`.
    pub fn pod<T: Pod>(value: &'a T) -> Self {
        let size = std::mem::size_of::<T>();
        // SAFETY: `Pod` guarantees plain data, so these `size` bytes are readable
        // and the kernel reads them as one C value.
        let bytes = unsafe { std::slice::from_raw_parts((value as *const T).cast::<u8>(), size) };
        Self::Bytes {
            bytes,
            align: std::mem::align_of::<T>(),
        }
    }

    /// Validate this argument against `domain` and resolve the bytes to marshal.
    fn resolve(&self, domain: &Arc<Domain>) -> Result<Resolved<'a>> {
        Ok(match *self {
            Arg::U32(value) => Resolved::Word {
                value: value as u64,
                size: 4,
            },
            Arg::U64(value) => Resolved::Word { value, size: 8 },
            Arg::I32(value) => Resolved::Word {
                value: value as u32 as u64,
                size: 4,
            },
            Arg::F32(value) => Resolved::Word {
                value: value.to_bits() as u64,
                size: 4,
            },
            Arg::F64(value) => Resolved::Word {
                value: value.to_bits(),
                size: 8,
            },
            Arg::DevicePtr(buffer) => Resolved::Word {
                value: checked_device_ptr(
                    domain,
                    buffer.alloc().domain(),
                    buffer.device_ptr(),
                    buffer.is_empty(),
                )?,
                size: 8,
            },
            Arg::DevicePtrUninit(buffer) => Resolved::Word {
                value: checked_device_ptr(
                    domain,
                    buffer.alloc().domain(),
                    buffer.device_ptr(),
                    buffer.is_empty(),
                )?,
                size: 8,
            },
            Arg::Bytes { bytes, align } => Resolved::Bytes { bytes, align },
        })
    }
}

/// A by-value kernel parameter: plain data the kernel reads as one C value.
///
/// # Safety
///
/// `Self` must be `#[repr(C)]` plain data with no padding and only initialised
/// bytes (declare explicit padding fields), no references and no interior pointer
/// the kernel would dereference: [`Arg::pod`] reads every byte of the value.
pub unsafe trait Pod: Copy {}

/// Largest alignment a by-value parameter may require.
///
/// Blocks are copied into the launch parameter buffer on 8-byte boundaries, so a
/// C type with a larger alignment cannot be passed through [`Arg::Bytes`].
pub const MAX_PARAMETER_ALIGN: usize = 8;

/// Largest number of arguments one launch accepts.
const MAX_ARGS: usize = 32;

/// One argument after domain validation.
#[derive(Clone, Copy)]
enum Resolved<'a> {
    /// A scalar or device pointer, held in one 8-byte parameter slot.
    Word { value: u64, size: usize },
    /// A by-value block.
    Bytes { bytes: &'a [u8], align: usize },
}

impl Resolved<'_> {
    /// Alignment and declared size of this argument in the parameter ABI, which
    /// is what the device's parameter limit applies to.
    fn layout(&self) -> Result<(usize, usize)> {
        match *self {
            Resolved::Word { size, .. } => Ok((size, size)),
            Resolved::Bytes { bytes, align } => {
                if bytes.is_empty() {
                    return Err(Error::invalid("a by-value parameter must have a size"));
                }
                if align == 0 || !align.is_power_of_two() || align > MAX_PARAMETER_ALIGN {
                    return Err(Error::invalid(format!(
                        "a by-value parameter requires {align}-byte alignment; the \
                         parameter buffer guarantees {MAX_PARAMETER_ALIGN}"
                    )));
                }
                Ok((align, bytes.len()))
            }
        }
    }
}

fn align_up(offset: usize, align: usize) -> usize {
    (offset + align - 1) & !(align - 1)
}

fn checked_device_ptr(
    domain: &Arc<Domain>,
    owner: &Arc<Domain>,
    pointer: u64,
    empty: bool,
) -> Result<u64> {
    if !Arc::ptr_eq(owner, domain) {
        return Err(Error::ForeignDomain);
    }
    if empty {
        return Err(Error::invalid("an empty buffer has no device address"));
    }
    Ok(pointer)
}

/// Where [`Parameters`] keeps one argument's bytes.
#[derive(Clone, Copy)]
enum Slot {
    Word,
    Block { offset: usize },
}

/// Marshalled launch parameters in declaration order.
///
/// Word arguments live in fixed slots; by-value blocks are copied into one
/// buffer of 8-byte words, so a launch without blocks allocates nothing.
pub(crate) struct Parameters {
    config: LaunchConfig,
    slots: [[u64; 1]; MAX_ARGS],
    kinds: [Slot; MAX_ARGS],
    blocks: Vec<u64>,
    count: usize,
    bytes: usize,
}

impl Parameters {
    fn encode(args: &[Resolved<'_>], config: LaunchConfig, limit: usize) -> Result<Self> {
        // Validate every argument and the whole checked parameter extent before
        // copying any block, so an oversized launch cannot duplicate a large slice
        // or abort the process on host memory first.
        let mut extent = 0;
        for arg in args {
            let (align, size) = arg.layout()?;
            extent = align_up(extent, align) + size;
        }
        if extent > limit {
            return Err(Error::KernelParameterLimit {
                bytes: extent,
                limit,
            });
        }
        let mut parameters = Self {
            config,
            slots: [[0; 1]; MAX_ARGS],
            kinds: [Slot::Word; MAX_ARGS],
            blocks: Vec::new(),
            count: args.len(),
            bytes: 0,
        };
        for (index, arg) in args.iter().enumerate() {
            match *arg {
                Resolved::Word { value, size } => {
                    parameters.slots[index] = [value];
                    parameters.kinds[index] = Slot::Word;
                    parameters.bytes += size;
                }
                Resolved::Bytes { bytes, align } => parameters.push_block(index, bytes, align)?,
            }
        }
        Ok(parameters)
    }

    fn push_block(&mut self, index: usize, bytes: &[u8], align: usize) -> Result<()> {
        // Both rules were already checked by `encode`; keep them here because
        // `push_block` is the only place that writes a block.
        if bytes.is_empty() || align == 0 || !align.is_power_of_two() || align > MAX_PARAMETER_ALIGN
        {
            return Err(Error::invalid("invalid by-value parameter block"));
        }
        let offset = self.blocks.len() * std::mem::size_of::<u64>();
        let words = bytes.len().div_ceil(std::mem::size_of::<u64>());
        self.blocks
            .resize(offset / std::mem::size_of::<u64>() + words, 0);
        let target = &mut self.blocks[offset / std::mem::size_of::<u64>()..];
        // SAFETY: `target` was just resized to `words` words, so its byte view
        // covers `words * 8 >= bytes.len()` writable bytes.
        let target =
            unsafe { std::slice::from_raw_parts_mut(target.as_mut_ptr().cast::<u8>(), words * 8) };
        target[..bytes.len()].copy_from_slice(bytes);
        self.kinds[index] = Slot::Block { offset };
        self.bytes += bytes.len();
        Ok(())
    }

    /// One pointer per argument, in declaration order, valid until the next
    /// mutation of these parameters.
    fn param_ptrs(&mut self) -> [*mut std::ffi::c_void; MAX_ARGS] {
        let blocks = self.blocks.as_mut_ptr().cast::<u8>();
        let slots = self.slots.as_mut_ptr();
        let mut params = [std::ptr::null_mut(); MAX_ARGS];
        for (index, param) in params.iter_mut().enumerate().take(self.count) {
            *param = match self.kinds[index] {
                // SAFETY: `index` is below `MAX_ARGS` and `slots` keeps the array.
                Slot::Word => unsafe { (*slots.add(index)).as_mut_ptr().cast() },
                // SAFETY: the offset was recorded while `blocks` covered it.
                Slot::Block { offset } => unsafe { blocks.add(offset).cast() },
            };
        }
        params
    }
}

/// A bounded, runtime-owned module cache.
#[derive(Default)]
pub(crate) struct ModuleCache {
    entries: Vec<ModuleEntry>,
    bytes: usize,
    max_entries: usize,
    max_bytes: usize,
    loads: u64,
}

struct ModuleEntry {
    artifact: Arc<Artifact>,
    module: Arc<ModuleInner>,
    bytes: usize,
}

impl ModuleCache {
    pub(crate) fn new(max_entries: usize, max_bytes: usize) -> Self {
        Self {
            entries: Vec::new(),
            bytes: 0,
            max_entries,
            max_bytes,
            loads: 0,
        }
    }

    pub(crate) fn lookup(&mut self, artifact: &Artifact) -> Option<Arc<ModuleInner>> {
        self.entries
            .iter()
            .find(|candidate| *candidate.artifact == *artifact)
            .map(|candidate| Arc::clone(&candidate.module))
    }

    /// Retain a loaded module, evicting by the configured entry/byte bounds.
    ///
    /// The cache owns the module, so a bound function can be re-bound without
    /// reloading; eviction does not unload a module still held by a
    /// `BoundFunction` or a queued launch.
    pub(crate) fn insert(&mut self, artifact: &Arc<Artifact>, module: &Arc<ModuleInner>) {
        // Count the real load even when the entry is not retained.
        self.loads += 1;
        // The entry retains the whole artifact (source, headers, options), not
        // only the loaded image.
        let retained = artifact.retained_bytes().max(module.bytes());
        if self.max_entries == 0 || retained > self.max_bytes {
            return;
        }
        while !self.entries.is_empty()
            && (self.entries.len() >= self.max_entries || self.bytes + retained > self.max_bytes)
        {
            let evicted = self.entries.remove(0);
            self.bytes = self.bytes.saturating_sub(evicted.bytes);
        }
        self.bytes += retained;
        self.entries.push(ModuleEntry {
            artifact: Arc::clone(artifact),
            module: Arc::clone(module),
            bytes: retained,
        });
    }

    pub(crate) fn entries(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }

    pub(crate) fn loads(&self) -> u64 {
        self.loads
    }

    /// Drop cached module references. Bound functions and queued launches keep
    /// their own references, so this never unloads code still in use.
    pub(crate) fn clear(&mut self) {
        self.entries.clear();
        self.bytes = 0;
    }
}

pub(crate) fn bind(
    domain: &Arc<Domain>,
    cache: &mut ModuleCache,
    artifact: &Arc<Artifact>,
    entry: &str,
) -> Result<BoundFunction> {
    if entry.is_empty() {
        return Err(Error::invalid("entry point name must not be empty"));
    }
    let module = match cache.lookup(artifact) {
        Some(module) => module,
        None => {
            let image_bytes = artifact.image().len();
            let _scope = domain.scope()?;
            // Reserve the module against the whole pending budget (staging and
            // keepalives included) before loading, under the admission lock.
            domain.retirement().during_admission(|admission| {
                admission.ensure_capacity(domain.stream(), image_bytes, 1)
            })?;
            let image = artifact.image();
            if image.is_empty() {
                return Err(Error::invalid("artifact image is empty"));
            }
            let mut terminated = Vec::with_capacity(image.len() + 1);
            terminated.extend_from_slice(image);
            terminated.push(0);
            // SAFETY: the image is a NUL-terminated PTX/CUBIN buffer that outlives
            // the call; the domain context is active.
            let module = unsafe { drv::module::load_data(terminated.as_ptr().cast()) }
                .map_err(Error::driver("cuModuleLoadData"))?;
            let inner = Arc::new(ModuleInner {
                holder: Arc::clone(domain.holder()),
                module,
                bytes: image_bytes,
                domain: Arc::downgrade(domain),
            });
            domain.retirement().retain_module_bytes(image_bytes);
            let _ = &inner;
            cache.insert(artifact, &inner);
            inner
        }
    };
    let name = CString::new(entry).map_err(|_| Error::invalid("entry point contains NUL"))?;
    let _scope = domain.scope()?;
    // SAFETY: `module` is alive (owned by `module`) and the context is active.
    let function = match unsafe { drv::module::get_function(module.module, name) } {
        Ok(function) => function,
        Err(error) => {
            // SAFETY: the module is still owned here; unload at most once through
            // the Arc drop path by dropping our reference.
            drop(module);
            return Err(Error::from_driver("cuModuleGetFunction", error));
        }
    };
    Ok(BoundFunction {
        domain: Arc::clone(domain),
        module,
        function,
        entry: entry.to_string(),
        artifact_key: artifact.key(),
    })
}

/// Validate `args` and marshal them for one launch of `function`.
pub(crate) fn prepare(
    domain: &Arc<Domain>,
    function: &BoundFunction,
    config: LaunchConfig,
    args: &[Arg<'_>],
) -> Result<Parameters> {
    if !Arc::ptr_eq(&function.domain, domain) {
        return Err(Error::ForeignDomain);
    }
    config.validate()?;
    if let Some(detail) = domain.broken() {
        return Err(Error::Broken { detail });
    }
    if args.len() > MAX_ARGS {
        return Err(Error::invalid(format!(
            "a launch accepts at most {MAX_ARGS} arguments"
        )));
    }
    let mut resolved = [Resolved::Word { value: 0, size: 0 }; MAX_ARGS];
    for (index, arg) in args.iter().enumerate() {
        resolved[index] = arg.resolve(domain)?;
    }
    Parameters::encode(
        &resolved[..args.len()],
        config,
        domain.max_kernel_parameter_bytes(),
    )
}

/// Enqueue `parameters` inside an open admission on the runtime's stream.
///
/// `allow_seal` follows the enclosing admission: a plain launch may seal its
/// batch at half the record budget, while a raw scope must keep every resource it
/// declared pinned until the scope seals once, because its later external
/// enqueues are not covered by an event recorded now.
pub(crate) fn enqueue(
    domain: &Arc<Domain>,
    function: &BoundFunction,
    parameters: &mut Parameters,
    admission: &Admission<'_>,
    allow_seal: bool,
) -> Result<()> {
    let _scope = domain.scope()?;
    // Pin the module before the launch is enqueued so a dropped `BoundFunction`
    // cannot unload code that is still queued, and reserve its record before the
    // submission.
    admission.add_keepalive(
        domain.stream(),
        Box::new(Arc::clone(&function.module)),
        0,
        allow_seal,
    )?;
    let mut params = parameters.param_ptrs();
    let config = parameters.config;
    // SAFETY: the caller guarantees the kernel ABI and pointer validity; the
    // domain context is active and every argument was validated by `prepare`.
    let status = unsafe {
        sys::cuLaunchKernel(
            function.function,
            config.grid.0,
            config.grid.1,
            config.grid.2,
            config.block.0,
            config.block.1,
            config.block.2,
            config.shared_bytes,
            domain.stream(),
            params.as_mut_ptr(),
            ptr::null_mut(),
        )
    };
    if status != sys::CUresult::CUDA_SUCCESS {
        return Err(Error::from_driver("cuLaunchKernel", DriverError(status)));
    }
    domain.retirement().note_launch(parameters.bytes);
    Ok(())
}

/// Enqueue a checked kernel launch.
///
/// # Safety
///
/// The bound function's kernel ABI must match `args`, every device pointer must
/// address `arg` bytes actually used by the kernel, and any output the caller
/// treats as initialised must be fully overwritten by this launch.
pub(crate) unsafe fn launch(
    domain: &Arc<Domain>,
    function: &BoundFunction,
    config: LaunchConfig,
    args: &[Arg<'_>],
) -> Result<()> {
    let mut parameters = prepare(domain, function, config, args)?;
    domain
        .retirement()
        .during_admission(|admission| enqueue(domain, function, &mut parameters, admission, true))
}

#[cfg(test)]
mod tests;
