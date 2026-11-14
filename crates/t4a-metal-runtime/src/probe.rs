//! The capability probe.
//!
//! The probe answers, in this order: can the frameworks load, does a device exist,
//! are the classes the backend needs present, does the MSL compiler accept each
//! dtype, and does one small real dispatch round-trip. A failed probe is a typed
//! error naming what failed; nothing panics, and nothing is constructed before the
//! device answers.

use t4a_metal_sys::{Framework, Table};

use crate::compile::CompileRequest;
use crate::context::DeviceInfo;
use crate::error::{Error, Result};
use crate::kernel::{Arg, LaunchConfig};
use crate::runtime::{Config, Runtime};

/// One probe kernel source, kept so a report can record its hash.
#[derive(Debug, Clone, Copy)]
pub struct ProbeSource {
    /// The dtype or capability this source tests.
    pub label: &'static str,
    /// The MSL source.
    pub source: &'static str,
}

const F32_SOURCE: &str = r#"#include <metal_stdlib>
using namespace metal;

kernel void add(device float* a [[buffer(0)]],
                device const float* b [[buffer(1)]],
                uint index [[thread_position_in_grid]]) {
    a[index] = a[index] + b[index];
}
"#;

const F64_SOURCE: &str = r#"#include <metal_stdlib>
using namespace metal;

kernel void scale(device double* a [[buffer(0)]],
                  uint index [[thread_position_in_grid]]) {
    a[index] = a[index] * 2.0;
}
"#;

const FP16_SOURCE: &str = r#"#include <metal_stdlib>
using namespace metal;

kernel void scale(device half* a [[buffer(0)]],
                  uint index [[thread_position_in_grid]]) {
    a[index] = a[index] * half(2.0);
}
"#;

const BF16_SOURCE: &str = r#"#include <metal_stdlib>
using namespace metal;

kernel void scale(device bfloat* a [[buffer(0)]],
                  uint index [[thread_position_in_grid]]) {
    a[index] = a[index] * bfloat(2.0);
}
"#;

const C32_SOURCE: &str = r#"#include <metal_stdlib>
using namespace metal;

kernel void scale(device float2* a [[buffer(0)]],
                  uint index [[thread_position_in_grid]]) {
    float2 value = a[index];
    a[index] = float2(value.x * value.x - value.y * value.y,
                      value.x * value.y + value.y * value.x);
}
"#;

/// The probe kernels, in probe order.
///
/// This is not the kernel catalogue; the catalogue lives in the kernel repository.
pub const PROBE_SOURCES: [ProbeSource; 5] = [
    ProbeSource {
        label: "f32",
        source: F32_SOURCE,
    },
    ProbeSource {
        label: "f64",
        source: F64_SOURCE,
    },
    ProbeSource {
        label: "fp16",
        source: FP16_SOURCE,
    },
    ProbeSource {
        label: "bf16",
        source: BF16_SOURCE,
    },
    ProbeSource {
        label: "c32",
        source: C32_SOURCE,
    },
];

/// What this host's Metal backend can serve.
///
/// `f64_supported` is measured, not declared: the MSL compiler's own answer decides
/// it, and `f64_diagnostic` keeps what it said.
#[derive(Debug, Clone)]
pub struct Capabilities {
    /// Facts about the device.
    pub device: DeviceInfo,
    /// Whether an fp16 kernel compiles.
    pub fp16: bool,
    /// Whether a bf16 kernel compiles.
    pub bf16: bool,
    /// Whether a `float2` (Complex32 as a pair) kernel compiles.
    pub c32: bool,
    /// Whether a `double` kernel compiles. This is what the installed compiler
    /// answered, not a declaration: on every Apple GPU measured so far it rejects it.
    pub f64_supported: bool,
    /// Metal's diagnostic for the `double` kernel, when it rejected it.
    pub f64_diagnostic: Option<String>,
    /// Whether `MetalPerformanceShaders.framework` opened.
    pub mps_framework: bool,
    /// Whether `MetalPerformanceShadersGraph.framework` opened.
    pub mpsgraph_framework: bool,
    /// Whether `MPSMatrixMultiplication` is visible.
    pub mps_matrix_multiplication: bool,
    /// Whether `MPSGraph` is visible once its framework is open.
    ///
    /// The tenferro-rs#2402 measurement reported it absent because that framework
    /// had not been loaded; this field is the re-check.
    pub mpsgraph_class: bool,
}

/// Probe the backend through the system dynamic loader.
///
/// # Errors
///
/// [`Error::Unavailable`] when the frameworks, a symbol or a device is missing, and
/// [`Error::Compile`] when the installed MSL compiler rejects a probe kernel.
///
/// # Examples
///
/// ```
/// match t4a_metal_runtime::probe() {
///     Ok(capabilities) => {
///         println!("{} unified={}", capabilities.device.name, capabilities.device.unified_memory);
///         assert!(!capabilities.f64_supported, "no Apple GPU compiles a double kernel");
///     }
///     Err(t4a_metal_runtime::Error::Unavailable { .. }) => {}
///     Err(other) => panic!("unexpected probe failure: {other}"),
/// }
/// ```
pub fn probe() -> Result<Capabilities> {
    let table = Table::load().map_err(Error::from)?;
    probe_with(&table)
}

/// The probe kernels without a device.
///
/// # Examples
///
/// ```
/// for source in t4a_metal_runtime::probe_sources() {
///     let hash = t4a_metal_runtime::source_hash(source.source);
///     assert_eq!(hash.len(), 64, "{} has a source hash", source.label);
/// }
/// ```
pub fn probe_sources() -> &'static [ProbeSource] {
    &PROBE_SOURCES
}

pub(crate) fn probe_with(table: &Table) -> Result<Capabilities> {
    let mut runtime = Runtime::open_with(Config::default(), *table)?;
    let device = runtime.device().clone();

    // f32 is required: a backend that cannot compile it serves nothing, so a rejection
    // is this host's error rather than a capability the caller has to interpret.
    runtime.compile(&CompileRequest::new("probe-f32", F32_SOURCE))?;
    let (f64_supported, f64_diagnostic) =
        match runtime.compile(&CompileRequest::new("probe-f64", F64_SOURCE)) {
            Ok(_) => (true, None),
            Err(Error::Compile { detail }) => (false, Some(detail)),
            Err(other) => return Err(other),
        };
    let fp16 = compiles(&mut runtime, "probe-fp16", FP16_SOURCE)?;
    let bf16 = compiles(&mut runtime, "probe-bf16", BF16_SOURCE)?;
    let c32 = compiles(&mut runtime, "probe-c32", C32_SOURCE)?;

    roundtrip(&mut runtime)?;

    Ok(Capabilities {
        device,
        fp16,
        bf16,
        c32,
        f64_supported,
        f64_diagnostic,
        mps_framework: table.has_framework(Framework::MetalPerformanceShaders),
        mpsgraph_framework: table.has_framework(Framework::MetalPerformanceShadersGraph),
        mps_matrix_multiplication: table.class("MPSMatrixMultiplication").is_some(),
        mpsgraph_class: table.class("MPSGraph").is_some(),
    })
}

fn compiles(runtime: &mut Runtime, label: &str, source: &str) -> Result<bool> {
    match runtime.compile(&CompileRequest::new(label, source)) {
        Ok(_) => Ok(true),
        Err(Error::Compile { .. }) => Ok(false),
        Err(other) => Err(other),
    }
}

/// Dispatch `[1,2,3,4] + [10,10,10,10]` and require exactly `[11,12,13,14]`.
///
/// Integer-valued f32 arithmetic is exact, so this is an equality check and not a
/// tolerance check.
fn roundtrip(runtime: &mut Runtime) -> Result<()> {
    let elements = 4usize;
    let bytes = elements * std::mem::size_of::<f32>();
    let a = runtime.alloc_zeroed(bytes)?;
    let b = runtime.alloc_zeroed(bytes)?;
    // SAFETY: both buffers were just allocated, are 16 bytes long and no submission
    // references them yet, so the host may write them.
    unsafe {
        let a_values = std::slice::from_raw_parts_mut(a.host_ptr().cast::<f32>(), elements);
        let b_values = std::slice::from_raw_parts_mut(b.host_ptr().cast::<f32>(), elements);
        for index in 0..elements {
            a_values[index] = index as f32 + 1.0;
            b_values[index] = 10.0;
        }
    }
    let artifact = runtime.compile(&CompileRequest::new("probe-add", F32_SOURCE))?;
    let function = runtime.bind(&artifact, "add")?;
    // SAFETY: the kernel reads index 1 and reads and writes index 0, both `elements`
    // f32 values long; the dispatch covers exactly those elements.
    unsafe {
        runtime.launch(
            &function,
            LaunchConfig::threads(elements as u32),
            &[
                Arg::Buffer {
                    buffer: &a,
                    offset: 0,
                },
                Arg::Buffer {
                    buffer: &b,
                    offset: 0,
                },
            ],
        )
    }?;
    runtime.commit()?.wait()?;
    // SAFETY: every submission that could touch `a` has completed and the runtime was
    // synchronised, so the host may read it.
    let values = unsafe { std::slice::from_raw_parts(a.host_ptr().cast::<f32>(), elements) };
    let expected = [11.0f32, 12.0, 13.0, 14.0];
    if values != expected {
        return Err(Error::unavailable(
            "Metal compute roundtrip",
            format!("the probe kernel produced {values:?}, expected {expected:?}"),
        ));
    }
    Ok(())
}
