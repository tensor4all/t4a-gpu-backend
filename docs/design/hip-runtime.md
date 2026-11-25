# HIP runtime and rocBLAS seam — stage 1 record

**Status:** stage 1 of [tenferro-rs#2489](https://github.com/tensor4all/tenferro-rs/issues/2489)
(the revised staged plan and the state-of-play comment at its end). Standalone
validation on one AMD APU; no CI job runs the hardware tests, and nothing in tenferro
changes.

Measured on evo2: AMD Ryzen AI MAX+ 395 with Radeon 8060S (`gfx1151`, wave32,
integrated, `totalGlobalMem` 34,359,738,368 bytes), ROCm 7.2.4 packages
(`hipRuntimeGetVersion` 70253211, rocBLAS `5.2.0.dabb6df2b9`), `rustc 1.100.0`,
`HSA_OVERRIDE_GFX_VERSION` unset.

## 1. Boundary

| Crate | Role | Depends on |
| --- | --- | --- |
| `t4a-hip-sys` | run-time loaded `libamdhip64`, `librocblas`, `librocsolver` function tables and the C types they pass; the load-time rules in §3 | `libc` (Linux), `thiserror` |
| `t4a-hip-runtime` | one stream, one explicit `hipMemPool`, stream-ordered buffers, blocking copies, pointer classification, a raw vendor scope | `t4a-hip-sys`, `thiserror` |
| test target of `t4a-hip-runtime` | the `Rocblas` seam (`tests/support/rocblas_seam.rs`), the mirror of tenferro-gpu's `Cublas` + `GemmRequest` | the two crates |

The seam is test code on purpose. On CUDA the vendor adapter lives in the consumer
(`tenferro-gpu/src/cuda/vendor.rs`), not in `t4a-cuda-runtime`, and stage 2 of the plan
moves the ROCm one into `tenferro-gpu/src/rocm/`. Stage 1 proves that the runtime's
ownership contract carries a vendor library; it does not publish a vendor API.

No ROCm library is linked. `readelf -d` on all three test binaries lists only
`libgcc_s.so.1`, `libc.so.6` and `ld-linux-x86-64.so.2`, and the same binaries run in
a plain `ubuntu:24.04` container (no ROCm) with every test reporting the typed
*unavailable* reason and passing; with `T4A_REQUIRE_HIP=1` they fail and name each
soname that was tried.

No `rocm` feature exists in this repository or in tenferro. The hardware tests select
themselves at run time instead of through a Cargo feature, so the binary that runs on
evo2 is the one that runs on a hosted runner.

## 2. Ownership

- **One stream, one pool per `Runtime`.** `Runtime::open` checks
  `hipDeviceAttributeMemoryPoolsSupported` (1 on gfx1151) and creates a non-blocking
  stream and a `hipMemPoolCreate` pool for the device. Two runtimes on one ordinal have
  different streams, pools and domain ids (`a_foreign_runtime_or_its_buffers_are_refused`).
- **Buffers** come from `hipMallocFromPoolAsync` on the runtime's stream. The last drop
  of a `Buffer` queues `hipFreeAsync` on the same stream, so it is ordered after every
  use submitted through that runtime, vendor calls included. Each buffer holds the
  domain alive; the domain destroys pool and stream only after the last buffer.
- **Copies** are blocking in stage 1 (`hipMemcpyAsync` + `hipStreamSynchronize`), so a
  caller's host slice is never read after `upload` returns. Pinned staging, completion
  handles and cross-runtime copies are not here.
- **Raw scope.** `Runtime::with_raw(&[&Buffer], |raw| ..)` checks every buffer belongs to
  the runtime, makes its device current on the calling thread (HIP's current device is
  per thread) and exposes the stream, registered device pointers, pool allocation and
  pointer classification.

## 3. Loading rules

- Every library is opened with `RTLD_NOW | RTLD_LOCAL` and never closed.
- HIP candidates are `libamdhip64.so.7` then `.so.6`: `$ROCM_PATH/lib/<soname>` when
  `ROCM_PATH` is set, otherwise the bare soname for the dynamic loader. The unversioned
  `libamdhip64.so` is not a candidate; evo2 also carries a distro
  `/lib/x86_64-linux-gnu/libamdhip64.so.5`.
- rocBLAS and rocSOLVER are opened **only** from the directory of the `libamdhip64` that
  was mapped (found with `dladdr`). Both name `libamdhip64.so.7` as `NEEDED`.
- **One HIP runtime per process.** Right after the HIP runtime is opened, and again after
  each math library, `/proc/self/maps` is scanned and a second distinct `libamdhip64` file
  is refused (`DuplicateRuntime`) before any stream exists. Two runtimes have separate
  device state: with the distro `libamdhip64.so.5` preloaded, opening a runtime used to
  dump core, and now fails with an error naming both files. The regression test preloads
  a copy of the real library in a child process
  (`a_second_mapped_hip_runtime_is_refused_before_a_stream_is_created`). On evo2, without
  a preload, exactly one is mapped (`the_device_and_libraries_are_recorded`).
- `HSA_OVERRIDE_GFX_VERSION` is read before anything is opened. Set but empty, it is
  refused with a configuration error that names it; with rocBLAS 5.2.0 that state makes
  `rocblas_create_handle` return `rocblas_status_invalid_handle`, which reads like a
  missing dtype. The variable is never set or changed by this code. The regression test
  runs a child process with the variable exported empty
  (`an_empty_gfx_override_is_refused_by_name_before_hip_loads`).

## 4. The rocBLAS seam

- **Handle setup** mirrors `Cublas::new`: `rocblas_create_handle` inside a raw scope →
  `rocblas_set_pointer_mode(host)` → `rocblas_set_stream(runtime stream)`, and each is
  read back (`rocblas_get_pointer_mode`, `rocblas_get_stream`). A handle that fails
  creation reports the status together with the `HSA_OVERRIDE_GFX_VERSION` state it saw.
- **Pointer classes are explicit.**
  - GEMM coefficients are host stack values, so the handle's pointer mode is re-read
    before every call; a handle not provably in host mode is refused (`mode_unproven`).
  - rocSOLVER's `info` and `ipiv` are written by a kernel and must be device memory. A
    host stack address there makes `rocsolver::reset_info` page-fault and the process
    wait forever (the #2489 retraction). The typed `getrf` takes `info` only as an
    `InfoSlot` (a pool allocation by construction). The raw `getrf_raw` classifies `info`,
    `ipiv` and `A` with `hipPointerGetAttributes` first and refuses anything that is not
    device memory of the runtime's device. On gfx1151, a stack or heap address is
    reported as `hipMemoryTypeUnregistered`, and a pool address (also an interior one)
    as `hipMemoryTypeDevice`.
- **Workspace is pool-owned.** A fresh handle reports
  `rocblas_is_managing_device_memory = true` and a device-memory size of its own
  (33,554,432 bytes, recorded by the seam before it hands over the pool buffer and
  asserted `managed` in `the_handle_runs_on_the_runtime_stream_with_a_pool_owned_workspace`).
  The seam hands it a 4 KiB pool buffer with `rocblas_set_workspace`, after which rocBLAS
  frees its own memory and reports `false` with exactly the pool buffer's size. The pool
  accounting is checked two-sided: `used` grows by the allocation and returns to its
  starting value after the free (`tests/runtime_pool.rs`), so a high-water mark would
  fail. Before every call the seam runs rocBLAS's
  size query (`rocblas_{start,stop}_device_memory_size_query`) on the same arguments and
  hands over a larger pool buffer if needed. The replaced buffer's free is ordered
  after the calls that used it. On gfx1151 every GEMM in the test suite queried 0 bytes
  (each case logs `largest workspace query so far`, and asserts the pool buffer covers
  it). rocSOLVER's `dgetrf` queries 0 bytes up to n = 64 and 790,784 bytes at
  n = 256, and at n = 256 the seam grows the pool buffer to exactly that
  (`a_call_that_needs_more_workspace_gets_a_larger_pool_buffer`). With the growth step
  removed, the same call fails with `rocblas_status_memory_error`: in user-owned mode
  the library refuses rather than allocating behind the pool's back.
- **Request checks**: the CUDA seam's checks (positive extents, leading dimension ≥ row
  count, `i32` narrowing that names the field and value), plus one the CUDA seam lacks:
  every element the call touches must lie inside its buffer.

## 5. Correctness

Each case runs for f32, f64, c32 and c64 and is compared element by element with a
compensated (Dot2) CPU reference against `|c − c_ref| ≤ f·k·ε·(|A||B|)`, with ε the
machine epsilon of the dtype under test and `f = 2` (real) or `4` (complex). Inputs are
pseudo-random with full mantissas.

| case (m, n, k) | worst error / bound over the four dtypes |
| --- | --- |
| plain (37, 29, 41) | 1.36e-2 |
| `trans_a`, `trans_b`, both (16, 12, 20) | ≤ 3.06e-2 |
| strided-batched (8, 6, 5) × 3 | 8.12e-2 |
| strided-batched, `trans_b`, padded `ld` and strides (9, 7, 11) × 4 | 6.25e-2 |
| past one tile (129, 257, 513) | 2.28e-3 |

The f64 check cannot be met at f32 precision: a correct `dgemm` result is at 6.3e-3
of the f64 bound, and the same result rounded through f32 is at 5.3e5 of it
(`the_f64_bound_rejects_a_result_computed_at_f32_precision`).

`ROCBLAS_LAYER=8` logs which GEMM backend rocBLAS used. With
`ROCBLAS_USE_HIPBLASLT=0` (and unset) every call goes to `rocblas_gemm_tensile_backend`.
With `=1`, f32 calls go to `rocblas_gemm_hipblaslt_backend` and f64, c32 and c64 fall
back to Tensile. The whole suite passes under both settings.

## 6. Throughput (recorded, not gated)

`cargo run --release -p t4a-hip-runtime --example rocblas_gflops`: square column-major
`n × n × n` products through the seam, two warm-up calls, then repetitions for at least
one second (at least three), timed on the host around runtime synchronisation. Complex
products count 8 flops per multiply-add. GFLOP/s, `ROCBLAS_USE_HIPBLASLT=0` (Tensile);
the second column per dtype is a repeat run:

| n | f32 | f32 rep | f64 | f64 rep | c32 | c32 rep | c64 | c64 rep |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 1408 | 1409 | 309 | 308 | 3633 | 3637 | 349 | 348 |
| 1024 | 3061 | 3056 | 434 | 432 | 5868 | 5755 | 440 | 441 |
| 2048 | 3238 | — | 445 | — | 6052 | — | 450 | — |
| 4096 | 3213 | 3205 | 447 | 447 | 6007 | 5896 | 454 | 453 |
| 8192 | 3164 | 3154 | 448 | 448 | 5948 | 5844 | 454 | 454 |

- **fp64 ratio:** f64/f32 is 0.142 at n=1024, 0.139 at 4096 and 0.142 at 8192, about
  **1/7 at every size from 1024 up**; it does not shrink at large n. c64 runs at the
  same ~450 GFLOP/s as f64, which is 0.076 of c32 (about 1/13) at n=8192.
- **hipBLASLt is not a better f32 path here.** With `ROCBLAS_USE_HIPBLASLT=1`, f32 runs
  at 68, 1564, 1386, 691 and 377 GFLOP/s for n = 256 … 8192; f64 and complex, which
  fall back to Tensile, are unchanged.
- **What these numbers are not.** gfx1151 is an APU: operands live in the LPDDR5X the
  CPU shares, and these are single-process host-timed figures. They say nothing about
  discrete MI-series cards, must not feed a cost model, and must not be read as fp64
  parity with the CUDA lane.

Raw data: [`hip-runtime-data/`](hip-runtime-data/) (`gfx1151-gflops-tensile.csv`,
`gfx1151-gflops-tensile-rep2.csv`, `gfx1151-gflops-hipblaslt.csv`).

## 7. Running it

```sh
cargo test -p t4a-hip-runtime                      # skips with a reason without a device
T4A_REQUIRE_HIP=1 cargo test -p t4a-hip-runtime    # a missing device fails
ROCBLAS_LAYER=8 ROCBLAS_USE_HIPBLASLT=1 T4A_REQUIRE_HIP=1 \
  cargo test -p t4a-hip-runtime -- --nocapture     # logs the backend per call
```

Never export `HSA_OVERRIDE_GFX_VERSION` empty; the runtime refuses that state by name.

## 8. Not in this unit

hiprtc and the kernel catalogue on HIP, pinned staging and asynchronous transfers,
completion handles, cross-runtime copies, `dot` (BLAS-1), rocSOLVER beyond `getrf`,
rocFFT, the tenferro `RocmEngine` and `rocm` feature (stage 2), and anything wave64.

## 9. Limits of this record

- One device (gfx1151, wave32, integrated); nothing here validates CDNA or wave64.
- No CI job runs the hardware tests; the evidence is the local run on evo2.
- The watchdog test for the host-`info` trap has been observed to pass (the guard
  refuses the pointer); the trap itself was not deliberately triggered on this shared
  host, because doing so page-faults the GPU. The guard's mutation was checked with the
  classification-only test instead.
