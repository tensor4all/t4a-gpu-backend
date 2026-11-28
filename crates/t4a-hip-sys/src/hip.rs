//! The HIP runtime function table and the C types it passes.
//!
//! Layouts and enum values are those of the ROCm 7.2 headers (`hip_runtime_api.h`),
//! checked against `offsetof`/`sizeof` from `hipcc` on gfx1151. The compile-time asserts
//! below pin every size this crate depends on; [`DevicePropR0600`] is a versioned ABI
//! (the `R0600` suffix), so its layout does not move with later ROCm releases.

use std::ffi::{c_char, c_void, CStr, OsString};
use std::path::Path;
use std::sync::OnceLock;

use crate::error::Result;
use crate::loader::{ensure_single_hip_runtime, hip_candidates, GfxOverride, Library};

/// `hipModule_t`: a code object loaded into the current context.
pub type HipModule = *mut c_void;
/// `hipFunction_t`: one kernel of a [`HipModule`].
pub type HipFunction = *mut c_void;

/// `hipError_t`.
pub type HipError = i32;
/// `hipStream_t`.
pub type HipStream = *mut c_void;
/// `hipMemPool_t`.
pub type HipMemPool = *mut c_void;

/// `hipSuccess`.
pub const HIP_SUCCESS: HipError = 0;
/// `hipStreamNonBlocking`.
pub const HIP_STREAM_NON_BLOCKING: u32 = 1;
/// `hipMemcpyHostToDevice`.
pub const HIP_MEMCPY_HOST_TO_DEVICE: i32 = 1;
/// `hipMemcpyDeviceToHost`.
pub const HIP_MEMCPY_DEVICE_TO_HOST: i32 = 2;
/// `hipMemAllocationTypePinned`.
pub const HIP_MEM_ALLOCATION_TYPE_PINNED: i32 = 1;
/// `hipMemHandleTypeNone`.
pub const HIP_MEM_HANDLE_TYPE_NONE: i32 = 0;
/// `hipMemLocationTypeDevice`.
pub const HIP_MEM_LOCATION_TYPE_DEVICE: i32 = 1;
/// `hipMemPoolAttrReservedMemCurrent` (a `u64`).
pub const HIP_MEM_POOL_ATTR_RESERVED_MEM_CURRENT: i32 = 5;
/// `hipMemPoolAttrUsedMemCurrent` (a `u64`).
pub const HIP_MEM_POOL_ATTR_USED_MEM_CURRENT: i32 = 7;
/// `hipDeviceAttributeIntegrated`.
pub const HIP_DEVICE_ATTRIBUTE_INTEGRATED: i32 = 16;
/// `hipDeviceAttributeWarpSize`.
pub const HIP_DEVICE_ATTRIBUTE_WARP_SIZE: i32 = 87;
/// `hipDeviceAttributeMemoryPoolsSupported`.
pub const HIP_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED: i32 = 88;
/// `hipMemoryTypeUnregistered`: an address HIP knows nothing about (a stack or plain
/// heap address).
pub const HIP_MEMORY_TYPE_UNREGISTERED: i32 = 0;
/// `hipMemoryTypeHost`: registered or pinned host memory.
pub const HIP_MEMORY_TYPE_HOST: i32 = 1;
/// `hipMemoryTypeDevice`: device memory, including stream-ordered pool allocations.
pub const HIP_MEMORY_TYPE_DEVICE: i32 = 2;
/// `hipMemoryTypeManaged`.
pub const HIP_MEMORY_TYPE_MANAGED: i32 = 3;

/// `hipMemLocation`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MemLocation {
    /// `hipMemLocationType`.
    pub kind: i32,
    /// Device ordinal for a device location.
    pub id: i32,
}

/// `hipMemPoolProps`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MemPoolProps {
    /// `hipMemAllocationType`.
    pub alloc_type: i32,
    /// `hipMemAllocationHandleType`.
    pub handle_types: i32,
    /// Where the pool's memory lives.
    pub location: MemLocation,
    /// Windows only; null here.
    pub win32_security_attributes: *mut c_void,
    /// Upper bound on the pool's size in bytes; 0 means no bound.
    pub max_size: usize,
    /// Reserved; must be zero.
    pub reserved: [u8; 56],
}

impl MemPoolProps {
    /// Properties of a pinned device-memory pool on `ordinal`.
    pub fn device(ordinal: i32) -> MemPoolProps {
        MemPoolProps {
            alloc_type: HIP_MEM_ALLOCATION_TYPE_PINNED,
            handle_types: HIP_MEM_HANDLE_TYPE_NONE,
            location: MemLocation {
                kind: HIP_MEM_LOCATION_TYPE_DEVICE,
                id: ordinal,
            },
            win32_security_attributes: std::ptr::null_mut(),
            max_size: 0,
            reserved: [0; 56],
        }
    }
}

/// `hipPointerAttribute_t`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PointerAttributes {
    /// `hipMemoryType`; see the `HIP_MEMORY_TYPE_*` constants.
    pub memory_type: i32,
    /// Device the allocation belongs to.
    pub device: i32,
    /// Device-side address.
    pub device_pointer: *mut c_void,
    /// Host-side address, if any.
    pub host_pointer: *mut c_void,
    /// Non-zero for managed memory.
    pub is_managed: i32,
    /// Allocation flags.
    pub allocation_flags: u32,
}

impl PointerAttributes {
    /// An all-zero value for `hipPointerGetAttributes` to fill.
    pub fn zeroed() -> PointerAttributes {
        PointerAttributes {
            memory_type: HIP_MEMORY_TYPE_UNREGISTERED,
            device: -1,
            device_pointer: std::ptr::null_mut(),
            host_pointer: std::ptr::null_mut(),
            is_managed: 0,
            allocation_flags: 0,
        }
    }
}

/// `hipDeviceProp_tR0600`, kept as bytes with accessors for the fields this crate reads.
///
/// The structure has well over a hundred fields; transcribing all of them would add
/// layout risk for nothing. The offsets below come from `offsetof` in `hipcc` (ROCm
/// 7.2.4) and the size is asserted at compile time.
#[repr(C, align(8))]
pub struct DevicePropR0600 {
    bytes: [u8; DevicePropR0600::SIZE],
}

impl DevicePropR0600 {
    /// `sizeof(hipDeviceProp_tR0600)`.
    pub const SIZE: usize = 1472;
    const NAME: usize = 0;
    const TOTAL_GLOBAL_MEM: usize = 288;
    const WARP_SIZE: usize = 308;
    const INTEGRATED: usize = 396;
    const GCN_ARCH_NAME: usize = 1160;
    const STRING: usize = 256;

    /// An all-zero value for `hipGetDevicePropertiesR0600` to fill.
    pub fn zeroed() -> DevicePropR0600 {
        DevicePropR0600 {
            bytes: [0; Self::SIZE],
        }
    }

    fn string(&self, offset: usize) -> String {
        let field = &self.bytes[offset..offset + Self::STRING];
        let end = field.iter().position(|&b| b == 0).unwrap_or(field.len());
        String::from_utf8_lossy(&field[..end]).into_owned()
    }

    fn i32_at(&self, offset: usize) -> i32 {
        let mut word = [0u8; 4];
        word.copy_from_slice(&self.bytes[offset..offset + 4]);
        i32::from_ne_bytes(word)
    }

    /// `name`: the marketing name, for example `AMD Radeon 8060S`.
    pub fn name(&self) -> String {
        self.string(Self::NAME)
    }

    /// `gcnArchName`: the ISA with its target features, for example `gfx1151`.
    pub fn gcn_arch_name(&self) -> String {
        self.string(Self::GCN_ARCH_NAME)
    }

    /// `totalGlobalMem` in bytes.
    pub fn total_global_mem(&self) -> u64 {
        let mut word = [0u8; 8];
        word.copy_from_slice(&self.bytes[Self::TOTAL_GLOBAL_MEM..Self::TOTAL_GLOBAL_MEM + 8]);
        u64::from_ne_bytes(word)
    }

    /// `warpSize`.
    pub fn warp_size(&self) -> i32 {
        self.i32_at(Self::WARP_SIZE)
    }

    /// `integrated`: non-zero for a GPU that shares memory with the host.
    pub fn integrated(&self) -> i32 {
        self.i32_at(Self::INTEGRATED)
    }
}

const _: () = assert!(std::mem::size_of::<MemLocation>() == 8);
const _: () = assert!(std::mem::size_of::<MemPoolProps>() == 88);
const _: () = assert!(std::mem::size_of::<PointerAttributes>() == 32);
const _: () = assert!(std::mem::size_of::<DevicePropR0600>() == DevicePropR0600::SIZE);

/// The HIP runtime entry points this repository uses.
///
/// Every field is the real entry point of the loaded `libamdhip64`; calling one is
/// `unsafe` with the C contract of the HIP function of the same name.
#[allow(missing_docs)]
#[derive(Debug)]
pub struct HipTable {
    library: Library,
    gfx_override: GfxOverride,
    pub get_error_name: unsafe extern "C" fn(HipError) -> *const c_char,
    pub runtime_get_version: unsafe extern "C" fn(*mut i32) -> HipError,
    pub driver_get_version: unsafe extern "C" fn(*mut i32) -> HipError,
    pub get_device_count: unsafe extern "C" fn(*mut i32) -> HipError,
    pub set_device: unsafe extern "C" fn(i32) -> HipError,
    pub get_device_properties: unsafe extern "C" fn(*mut DevicePropR0600, i32) -> HipError,
    pub device_get_attribute: unsafe extern "C" fn(*mut i32, i32, i32) -> HipError,
    pub stream_create_with_flags: unsafe extern "C" fn(*mut HipStream, u32) -> HipError,
    pub stream_destroy: unsafe extern "C" fn(HipStream) -> HipError,
    pub stream_synchronize: unsafe extern "C" fn(HipStream) -> HipError,
    pub mem_pool_create: unsafe extern "C" fn(*mut HipMemPool, *const MemPoolProps) -> HipError,
    pub mem_pool_destroy: unsafe extern "C" fn(HipMemPool) -> HipError,
    pub mem_pool_get_attribute: unsafe extern "C" fn(HipMemPool, i32, *mut c_void) -> HipError,
    pub mem_pool_trim_to: unsafe extern "C" fn(HipMemPool, usize) -> HipError,
    pub malloc_from_pool_async:
        unsafe extern "C" fn(*mut *mut c_void, usize, HipMemPool, HipStream) -> HipError,
    pub free_async: unsafe extern "C" fn(*mut c_void, HipStream) -> HipError,
    pub memcpy_async:
        unsafe extern "C" fn(*mut c_void, *const c_void, usize, i32, HipStream) -> HipError,
    pub memset_async: unsafe extern "C" fn(*mut c_void, i32, usize, HipStream) -> HipError,
    pub pointer_get_attributes:
        unsafe extern "C" fn(*mut PointerAttributes, *const c_void) -> HipError,
    pub get_last_error: unsafe extern "C" fn() -> HipError,
    pub module_load_data: unsafe extern "C" fn(*mut HipModule, *const c_void) -> HipError,
    pub module_unload: unsafe extern "C" fn(HipModule) -> HipError,
    pub module_get_function:
        unsafe extern "C" fn(*mut HipFunction, HipModule, *const c_char) -> HipError,
    pub module_launch_kernel: ModuleLaunchKernelFn,
}

/// `hipModuleLaunchKernel(f, gx, gy, gz, bx, by, bz, sharedBytes, stream, params, extra)`.
pub type ModuleLaunchKernelFn = unsafe extern "C" fn(
    HipFunction,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    HipStream,
    *mut *mut c_void,
    *mut *mut c_void,
) -> HipError;

static HIP: OnceLock<Result<HipTable>> = OnceLock::new();

impl HipTable {
    /// Load the HIP runtime once per process and return its table.
    ///
    /// `HSA_OVERRIDE_GFX_VERSION` is checked before anything is opened: when it is set
    /// but empty the load is refused with a configuration error naming it (see
    /// [`GfxOverride`]).
    ///
    /// # Errors
    ///
    /// [`crate::Error::Configuration`], [`crate::Error::LibraryUnavailable`] listing
    /// every candidate, [`crate::Error::MissingSymbol`], or
    /// [`crate::Error::DuplicateRuntime`] when another `libamdhip64` file is already
    /// mapped into the process. The first outcome is cached for the life of the
    /// process.
    pub fn load() -> Result<&'static HipTable> {
        HIP.get_or_init(|| {
            let gfx_override = GfxOverride::current();
            gfx_override.check()?;
            let rocm_path: Option<OsString> = std::env::var_os("ROCM_PATH");
            let candidates = hip_candidates(rocm_path.as_deref());
            let library = Library::open("HIP runtime", &candidates, b"hipGetDeviceCount\0")?;
            // Another `libamdhip64` may already be in the process (preloaded, or pulled in
            // by another library under a different soname). Two runtimes have separate
            // device state, and creating a stream in one of them can crash the process, so
            // the check runs here, before any table entry can be called.
            ensure_single_hip_runtime()?;
            // SAFETY: each type below is the C signature of the HIP function named.
            unsafe {
                Ok(HipTable {
                    get_error_name: library.function(b"hipGetErrorName\0")?,
                    runtime_get_version: library.function(b"hipRuntimeGetVersion\0")?,
                    driver_get_version: library.function(b"hipDriverGetVersion\0")?,
                    get_device_count: library.function(b"hipGetDeviceCount\0")?,
                    set_device: library.function(b"hipSetDevice\0")?,
                    get_device_properties: library.function(b"hipGetDevicePropertiesR0600\0")?,
                    device_get_attribute: library.function(b"hipDeviceGetAttribute\0")?,
                    stream_create_with_flags: library.function(b"hipStreamCreateWithFlags\0")?,
                    stream_destroy: library.function(b"hipStreamDestroy\0")?,
                    stream_synchronize: library.function(b"hipStreamSynchronize\0")?,
                    mem_pool_create: library.function(b"hipMemPoolCreate\0")?,
                    mem_pool_destroy: library.function(b"hipMemPoolDestroy\0")?,
                    mem_pool_get_attribute: library.function(b"hipMemPoolGetAttribute\0")?,
                    mem_pool_trim_to: library.function(b"hipMemPoolTrimTo\0")?,
                    malloc_from_pool_async: library.function(b"hipMallocFromPoolAsync\0")?,
                    free_async: library.function(b"hipFreeAsync\0")?,
                    memcpy_async: library.function(b"hipMemcpyAsync\0")?,
                    memset_async: library.function(b"hipMemsetAsync\0")?,
                    pointer_get_attributes: library.function(b"hipPointerGetAttributes\0")?,
                    get_last_error: library.function(b"hipGetLastError\0")?,
                    module_load_data: library.function(b"hipModuleLoadData\0")?,
                    module_unload: library.function(b"hipModuleUnload\0")?,
                    module_get_function: library.function(b"hipModuleGetFunction\0")?,
                    module_launch_kernel: library.function(b"hipModuleLaunchKernel\0")?,
                    library,
                    gfx_override,
                })
            }
        })
        .as_ref()
        .map_err(Clone::clone)
    }

    /// The `libamdhip64` file the dynamic loader mapped.
    pub fn library_path(&self) -> &Path {
        self.library.path()
    }

    /// The `HSA_OVERRIDE_GFX_VERSION` state this process was loaded under.
    pub fn gfx_override(&self) -> &GfxOverride {
        &self.gfx_override
    }

    /// `hipGetErrorName` for `code`, or the number when HIP has no name for it.
    pub fn error_name(&self, code: HipError) -> String {
        // SAFETY: `hipGetErrorName` accepts any value and returns a static string or
        // null.
        let name = unsafe { (self.get_error_name)(code) };
        if name.is_null() {
            return format!("hipError {code}");
        }
        // SAFETY: non-null is a static NUL-terminated string owned by the runtime.
        unsafe { CStr::from_ptr(name) }
            .to_string_lossy()
            .into_owned()
    }
}
