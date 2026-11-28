# t4a-hip-kernels

HIP kernels compiled at run time with hiprtc, reusing the `t4a-cuda-kernels` C ABI
(tenferro-rs#2489, M2).

No kernel source lives here. `t4a-cuda-kernels` owns the text, the ABI header, the dtype
tags and the launch planning; this crate compiles the same composed text with hiprtc for
the device's own `gfx` target, loads it, and launches it with the planned `Launch`.
The one HIP-specific piece is `HIP_PRELUDE`, placed before the unmodified CUDA text: HIP's
`__shfl_sync`/`__shfl_xor_sync` static-assert a 64-bit mask, the CUDA sources pass a 32-bit
one.

Interface: `Program::compile` / `compile_entry` (load, with a `CompileReport`),
`Program::launch` (the raw seam inside `Runtime::with_raw`), `elementwise` (safe flat
entries), `Error`.

```sh
cargo run --release -p t4a-hip-kernels --example add_f32
cargo test --release -p t4a-hip-kernels            # skips without a device; T4A_REQUIRE_HIP=1 forbids that
cargo test --release -p t4a-hip-kernels -- --ignored   # hiprtc survey of all 11 CUDA modules (~2 min)
```
