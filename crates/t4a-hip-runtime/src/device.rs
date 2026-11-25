//! Device discovery: what HIP itself reports for each visible device.

use std::path::PathBuf;

use t4a_hip_sys::{
    DevicePropR0600, GfxOverride, HipTable, HIP_DEVICE_ATTRIBUTE_INTEGRATED,
    HIP_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED, HIP_DEVICE_ATTRIBUTE_WARP_SIZE, HIP_SUCCESS,
};

use crate::error::{Error, Result};

/// `hipErrorNoDevice`: the library loaded and no device is visible.
const HIP_ERROR_NO_DEVICE: i32 = 100;

/// What the HIP runtime reports for one device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    /// HIP ordinal, after `HIP_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES`.
    pub ordinal: usize,
    /// Marketing name, for example `AMD Radeon 8060S`.
    pub name: String,
    /// ISA with target features, for example `gfx1151`.
    pub gcn_arch_name: String,
    /// Wavefront width the device executes.
    pub warp_size: u32,
    /// Whether the GPU shares memory with the host (an APU such as gfx1151).
    pub integrated: bool,
    /// `totalGlobalMem` in bytes. On an integrated GPU this depends on BIOS and kernel
    /// settings, so it is reported, never assumed.
    pub total_global_mem: u64,
    /// Whether the device supports stream-ordered memory pools, which this runtime
    /// requires.
    pub memory_pools: bool,
    /// `hipRuntimeGetVersion`.
    pub runtime_version: i32,
    /// `hipDriverGetVersion`.
    pub driver_version: i32,
    /// The `libamdhip64` file that was loaded.
    pub library: PathBuf,
    /// The `HSA_OVERRIDE_GFX_VERSION` state the process runs under.
    pub gfx_override: GfxOverride,
}

pub(crate) fn check(hip: &HipTable, call: &'static str, code: i32) -> Result<()> {
    if code == HIP_SUCCESS {
        return Ok(());
    }
    Err(Error::Hip {
        call,
        code,
        name: hip.error_name(code),
    })
}

/// Number of visible HIP devices.
///
/// # Errors
///
/// [`Error::Unavailable`] when the HIP runtime cannot be loaded,
/// [`Error::Configuration`] for an empty `HSA_OVERRIDE_GFX_VERSION`, and
/// [`Error::Hip`] for any other failure of `hipGetDeviceCount`. A host whose runtime
/// loads but has no device reports `Ok(0)`.
pub fn device_count() -> Result<usize> {
    let hip = HipTable::load()?;
    let mut count = 0i32;
    // SAFETY: `count` is a writable `int`.
    let code = unsafe { (hip.get_device_count)(&mut count) };
    if code == HIP_ERROR_NO_DEVICE {
        return Ok(0);
    }
    check(hip, "hipGetDeviceCount", code)?;
    Ok(usize::try_from(count).unwrap_or(0))
}

/// Describe device `ordinal`.
///
/// # Errors
///
/// As [`device_count`], plus [`Error::InvalidDevice`] for an ordinal past the visible
/// devices, and [`Error::Unsupported`] when the warp size read from the properties
/// structure disagrees with the attribute query, which would mean the structure's
/// layout is not the one this crate was built against.
pub fn device_info(ordinal: usize) -> Result<DeviceInfo> {
    let hip = HipTable::load()?;
    let count = device_count()?;
    if ordinal >= count {
        return Err(Error::InvalidDevice { ordinal, count });
    }
    let device = i32::try_from(ordinal).map_err(|_| Error::InvalidDevice { ordinal, count })?;
    let mut props = DevicePropR0600::zeroed();
    // SAFETY: `props` is a writable `hipDeviceProp_tR0600` of the asserted size.
    check(hip, "hipGetDevicePropertiesR0600", unsafe {
        (hip.get_device_properties)(&mut props, device)
    })?;
    let attribute = |attribute: i32, call: &'static str| -> Result<i32> {
        let mut value = 0i32;
        // SAFETY: `value` is a writable `int`; `device` is a visible ordinal.
        check(hip, call, unsafe {
            (hip.device_get_attribute)(&mut value, attribute, device)
        })?;
        Ok(value)
    };
    let warp_size = attribute(HIP_DEVICE_ATTRIBUTE_WARP_SIZE, "hipDeviceGetAttribute")?;
    let integrated = attribute(HIP_DEVICE_ATTRIBUTE_INTEGRATED, "hipDeviceGetAttribute")?;
    let memory_pools = attribute(
        HIP_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED,
        "hipDeviceGetAttribute",
    )?;
    if props.warp_size() != warp_size || props.integrated() != integrated {
        return Err(Error::Unsupported {
            detail: format!(
                "hipDeviceProp_tR0600 layout mismatch: warpSize {} vs attribute {warp_size}, \
                 integrated {} vs attribute {integrated}",
                props.warp_size(),
                props.integrated()
            ),
        });
    }
    let mut runtime_version = 0i32;
    // SAFETY: a writable `int`.
    check(hip, "hipRuntimeGetVersion", unsafe {
        (hip.runtime_get_version)(&mut runtime_version)
    })?;
    let mut driver_version = 0i32;
    // SAFETY: a writable `int`.
    check(hip, "hipDriverGetVersion", unsafe {
        (hip.driver_get_version)(&mut driver_version)
    })?;
    Ok(DeviceInfo {
        ordinal,
        name: props.name(),
        gcn_arch_name: props.gcn_arch_name(),
        warp_size: u32::try_from(warp_size).unwrap_or(0),
        integrated: integrated != 0,
        total_global_mem: props.total_global_mem(),
        memory_pools: memory_pools != 0,
        runtime_version,
        driver_version,
        library: hip.library_path().to_path_buf(),
        gfx_override: hip.gfx_override().clone(),
    })
}

/// Describe every visible device.
///
/// # Errors
///
/// As [`device_info`].
pub fn probe() -> Result<Vec<DeviceInfo>> {
    (0..device_count()?).map(device_info).collect()
}
