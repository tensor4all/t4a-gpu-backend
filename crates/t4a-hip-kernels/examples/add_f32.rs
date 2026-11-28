//! Compile and run `t4a_cuda_add` through hiprtc on device 0 and print the evidence.

use t4a_hip_kernels::abi::{DType, Entry, ABI_REVISION};
use t4a_hip_kernels::runtime::{probe, Runtime};
use t4a_hip_kernels::{elementwise, Binary, Program};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let devices = probe()?;
    let mut runtime = Runtime::open(0)?;
    let device = runtime.device().clone();
    println!(
        "device           {} ({})",
        device.name, device.gcn_arch_name
    );
    println!("warp size        {}", device.warp_size);
    println!(
        "HIP runtime      {} (driver {})",
        device.runtime_version, device.driver_version
    );
    println!("libamdhip64      {}", device.library.display());
    println!("HSA_OVERRIDE_GFX_VERSION={:?}", device.gfx_override);
    println!("visible devices  {}", devices.len());
    println!("t4a ABI revision {ABI_REVISION}");

    let program = Program::compile_entry(&runtime, Entry::Add)?;
    let r = program.report();
    println!("module           {}", r.module);
    println!("target string    {}", r.target);
    println!("hiprtc options   {:?}", r.options);
    println!("compile time     {:?}", r.compile_time);
    println!("load time        {:?}", r.load_time);
    println!("code object      {} bytes", r.code_bytes);
    if !r.log.is_empty() {
        println!("hiprtc log       {}", r.log);
    }

    let lhs = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let rhs = [10.0f32, 20.0, 30.0, 40.0, 50.0];
    let raw = |v: &[f32]| v.iter().flat_map(|x| x.to_ne_bytes()).collect::<Vec<u8>>();
    let out = runtime.alloc_zeroed(20)?;
    let a = runtime.upload(&raw(&lhs))?;
    let b = runtime.upload(&raw(&rhs))?;
    elementwise(
        &mut runtime,
        &program,
        Entry::Add,
        DType::F32,
        Binary {
            out: &out,
            lhs: &a,
            rhs: &b,
        },
        5,
    )?;
    let got: Vec<f32> = runtime
        .download(&out)?
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
        .collect();
    let want: Vec<f32> = lhs.iter().zip(&rhs).map(|(x, y)| x + y).collect();
    println!("lhs {lhs:?}\nrhs {rhs:?}\ngpu {got:?}\ncpu {want:?}");
    let exact = got
        .iter()
        .zip(&want)
        .all(|(g, w)| g.to_bits() == w.to_bits());
    println!("bit-identical    {exact}");
    println!("CUDA device      none on this host: CUDA path NOT RUN");
    if exact {
        Ok(())
    } else {
        Err("mismatch".into())
    }
}
