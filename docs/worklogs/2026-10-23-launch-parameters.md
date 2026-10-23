# Launch parameters and raw-scope retention — local evidence

Date: 2026-10-23. Issue:
[t4a-gpu-backend#5](https://github.com/tensor4all/t4a-gpu-backend/issues/5) — the
two launch-side capabilities tenferro's adoption
([tenferro-rs#2208](https://github.com/tensor4all/tenferro-rs/issues/2208)) waits
on.

## What landed

- `Arg::Bytes { bytes, align }` and `Arg::pod(&value)` (with the `unsafe Pod`
  marker) pass a C struct or array by value in the const bank. The encoder copies
  each block into one buffer of 8-byte words in declaration order, so a launch
  without blocks still allocates nothing. A block whose alignment requirement is
  zero, not a power of two, or above `MAX_PARAMETER_ALIGN` (8) is refused with
  `InvalidArgument`.
- Every argument's alignment and the whole checked ABI extent — each argument
  padded to its own alignment, 4 bytes per `U32`/`I32`/`F32`, 8 per
  `U64`/`F64`/pointer, `bytes.len()` per block — are validated *before* any block
  is copied, so an oversized launch cannot duplicate a large caller slice or
  abort on host memory first. The extent, not the payload sum, is what the
  device's parameter limit applies to; exceeding it is
  `Error::KernelParameterLimit { bytes, limit }`.
- The limit is derived from the device's major compute capability, because the
  driver exposes no attribute for it: 4 KiB below 7.0, 32,764 bytes from 7.0 on.
  `max_kernel_parameter_bytes(ordinal)` probes it and
  `Runtime::max_kernel_parameter_bytes()` reports it for an open runtime. NVRTC
  still enforces the limit of its own target, so a 4 KiB block plus the 144-byte
  layout needs a Volta-or-later `arch` at compile time.
- `RawScope::launch` submits through the runtime's own validated launcher
  (module pinning, counters, exact stream) and `RawScope::retain` registers an
  external keepalive against the pending budget, for work the runtime cannot see.
  Arbitrary pointers, foreign streams and the order of external work stay the
  caller's responsibility.
- **A raw scope never seals its batch early.** `Retirement::reserve_locked` used
  to seal the open batch under pending-budget pressure, which for a raw scope
  records an event covering only the work already queued: a resource declared for
  a *later* external enqueue could be released while that work still ran. The
  admission now carries `may_seal`; a raw scope passes `false` and gets a typed
  `Backpressure` when a new resource cannot be reserved, instead of losing
  lifetime protection. Plain launches and transfers keep sealing at half the
  record budget and under pressure, as before.
- `RetirementStats::launch_parameter_bytes` counts the parameter bytes admitted
  launches marshalled.

## Local gates

Host: one physical **NVIDIA A100 80GB PCIe**, driver **580.126.09**, CUDA toolkit
**12.6** (`/usr/local/cuda-12.6`, NVRTC from it).

| Command | Result |
| --- | --- |
| `cargo fmt --all --check` | clean |
| `cargo clippy -j 16 -p t4a-cuda-runtime --all-targets --features gpu-tests -- -D warnings` | clean |
| `cargo test -j 16 -p t4a-cuda-runtime` (CPU only) | 25 lib + 7 `host_contract` + 1 doctest passed |
| `cargo test -j 16 -p t4a-cuda-runtime --features gpu-tests --lib` | 48 passed |
| `cargo test -j 16 -p t4a-cuda-runtime --features gpu-tests --test gpu_single` | 22 passed |
| `cargo test -j 16 -p t4a-cuda-runtime --features gpu-tests --test gpu_faults` | 1 passed (two partitions) |
| `cargo doc -j 16 -p t4a-cuda-runtime --no-deps` | no warnings |
| `cargo test -j 16 -p t4a-cuda-runtime --features gpu-tests --test gpu_multi` | 2 failed: `NOT RUN: this target requires two physical GPUs, host reports 1` (expected on this host) |

The two new hardware tests:

- `by_value_parameters_reach_the_kernel` compiles a kernel taking a 144-byte
  layout and a 4 KiB block by value, launches it into an `UninitBuffer`, and
  reads the six words it wrote back: the kernel sees
  `rank=3, offset=-5, extents[7]=7, strides[3]=13, bytes[4095]=111, bytes[0]=200`.
  The counter must grow by `144 + 4096 + 8` parameter bytes.
- `raw_scope_launch_is_admitted_and_retains_external_resources` enqueues the same
  `scale2` kernel through `RawScope::launch` with the input address read from the
  registered buffer, retains an `Arc` for the duration of the scope (strong count
  2 after the closure, 1 once observed), and reads the doubled bytes back.

CPU unit tests (`src/kernel/tests/mod.rs`) cover declaration order and the byte
accounting, the 8-byte placement of consecutive blocks with the sources changed
after encoding, alignment refusals for 0/3/16/64 and acceptance for 1/2/4/8, empty
blocks, the padded ABI extent that the limit applies to, the limit refusal with
its device limit, and that the largest ABI block combination fits the 32,764-byte
limit but not the legacy 4 KiB one. `src/retirement/tests/mod.rs` covers the
non-sealing admission: a full batch that may not seal reports `Backpressure`
instead of releasing its declared resources.

## Review round

An independent review (`openai/gpt-6.1-sol`) of the finished diff raised seven
findings; all were verified against the source and fixed, except where noted:

- Pressure-driven sealing of a raw batch (verified in
  `Retirement::reserve_locked`) — fixed with `may_seal`, covered by the new
  retirement unit test.
- `Pod` allowed uninitialised padding — the contract now requires padding-free,
  fully initialised bytes.
- The parameter limit ignored ABI padding — the encoder now validates the padded
  extent, covered by `the_limit_counts_the_checked_abi_extent`.
- Blocks were copied before the limit check — validation is now a separate pass.
- The 32,764-byte value was presented as unconditional — the rustdoc of
  `max_kernel_parameter_bytes` and `Runtime::max_kernel_parameter_bytes` now says
  large parameters also need a compiler and driver that support them and that
  exceeding them fails in `cuLaunchKernel`. No driver-version gate was added: the
  limit cannot be verified offline, and the failure stays a typed driver error.
- The retention test did not prove reservation — it now asserts the reserved
  pending bytes. What it cannot prove is a use-after-free of a *buffer* handle:
  buffers are protected by stream ordering, which is why only non-stream-ordered
  resources are retained.
- The encoder copy test left the sources unchanged — it now changes them after
  encoding and asserts the stored bytes.
- Delete list: the test-only `Parameters::bytes()` getter is gone. The duplicated
  sections 8/9 in `docs/design/cuda-runtime.md` are pre-existing drift, unrelated
  to this change, and were left alone.

## Not done here

- No tenferro-side change: the consumer port stays in tenferro-rs#2208.
- The 32,764-byte limit is not exercised on a pre-Volta device (none on this
  host); only its refusal path is unit-tested.
