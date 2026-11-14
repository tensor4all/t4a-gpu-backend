//! Print this host's Metal capability dump and the probe sources' hashes.
//!
//! The dump is what a report about this backend records next to any number measured
//! with it: which device, which dtypes the MSL compiler accepted, and the SHA-256 of
//! every kernel source. On a host without a Metal device it prints the typed error
//! and exits non-zero, because a capability report about an absent backend is not a
//! successful report.

use std::process::ExitCode;

fn main() -> ExitCode {
    let capabilities = match t4a_metal_runtime::probe() {
        Ok(capabilities) => capabilities,
        Err(error) => {
            println!("metal: unavailable");
            println!("reason: {error}");
            return ExitCode::FAILURE;
        }
    };
    let device = &capabilities.device;
    println!("device: {}", device.name);
    println!("unified_memory: {}", device.unified_memory);
    println!("max_buffer_length: {}", device.max_buffer_length);
    println!(
        "recommended_max_working_set: {}",
        device.recommended_max_working_set
    );
    println!("fp16: {}", capabilities.fp16);
    println!("bf16: {}", capabilities.bf16);
    println!("c32: {}", capabilities.c32);
    println!("f64: {}", capabilities.f64_supported);
    if let Some(diagnostic) = &capabilities.f64_diagnostic {
        // One line, so the dump stays greppable.
        let first_line = diagnostic.lines().next().unwrap_or_default();
        println!("f64_diagnostic: {first_line}");
    }
    println!("mps_framework: {}", capabilities.mps_framework);
    println!("mpsgraph_framework: {}", capabilities.mpsgraph_framework);
    println!(
        "mps_matrix_multiplication: {}",
        capabilities.mps_matrix_multiplication
    );
    println!("mpsgraph_class: {}", capabilities.mpsgraph_class);
    for source in t4a_metal_runtime::probe_sources() {
        println!(
            "source {}: {}",
            source.label,
            t4a_metal_runtime::source_hash(source.source)
        );
    }
    ExitCode::SUCCESS
}
