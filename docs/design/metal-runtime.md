# Metal runtime — detailed design

**Status:** implemented on `feat/metal-runtime` for U2 of
[tenferro-rs#2402](https://github.com/tensor4all/tenferro-rs/issues/2402), following
[tensor4all-meta `docs/design/metal-backend.md` @ `8befe98`](https://github.com/tensor4all/tensor4all-meta/blob/8befe98/docs/design/metal-backend.md)
§6 and §10. The consumer (`MetalContext`/`MetalEngine` in tenferro-rs) is U4 and is not
part of this record. Standalone validation only: no macOS CI job exists in this
repository yet.

Measured on macOS 26.7.2, Apple M5 Max (40-core GPU, Metal 4, unified memory), `rustc
1.100.0`, commit on `feat/metal-runtime`.

## 1. Boundary

Two crates, split the way the design assigns them:

| Crate | Role | Depends on |
| --- | --- | --- |
| `t4a-metal-sys` | run-time loaded `libobjc`, `Metal.framework`, and (optionally) `MetalPerformanceShaders[Graph]`; every `objc_msgSend` cast in the repository | `libc`, `thiserror` |
| `t4a-metal-runtime` | one device, one command queue, shared-mode buffers, MSL compilation, checked launches, submission and completion, raw seam | `t4a-metal-sys`, `sha2`, `thiserror` |

Neither crate depends on tenferro, on a kernel catalogue, on CubeCL/CubeK, or on a
vendor library. There is no build script, no `links =`, no `-framework` flag and no
`objc2-metal`: frameworks are opened with `dlopen`/`dlsym` and classes with
`objc_getClass`. `otool -L` on the `metal_single` test binary lists only
`libiconv.2.dylib` and `libSystem.B.dylib`:

```sh
BIN=$(find target -name 'metal_single-*' -type f -perm +111 | head -1)
otool -L "$BIN" | tail -n +2 | grep -Ei 'Metal|MetalPerformanceShaders|Accelerate'   # prints nothing
```

(`tail -n +2` drops the header line, which contains the binary's own path.)

The runtime has **no transfers**. Every buffer is `MTLResourceStorageModeShared`, so
the CPU engine and the GPU read and write the same address. `upload`, `download`,
pinned staging, memory pools and cross-runtime copies are absent rather than stubbed:
they are tenferro's shared-allocation-domain concern (design §4), not this layer's.

## 2. Ownership

| Object | Owner and permitted use |
| --- | --- |
| `Runtime` | Non-`Clone` execution owner. One device, one command queue, one allocation domain. Submission and allocation take `&mut Runtime`. |
| `Buffer`, `UninitBuffer` | Immutable domain/length/address metadata plus a reference-counted allocation. No submission authority, no `Clone`; share one allocation with `Arc<Buffer>`. `host_ptr()` is a raw pointer, so host access is the caller's ordering responsibility. |
| `BoundFunction` | One `MTLComputePipelineState`, shared with the runtime's cache through an `Arc`. No submission authority. |
| `Artifact` | One `MTLLibrary`, immutable and shareable, no submission authority. |
| `Completion` | Observation handle for one submission. `poll` never blocks; `wait` re-checks until completion or `completion_timeout_ms`. |
| `RawScope` | Borrowed for one closure: the `MTLCommandBuffer`, the device and the domain check. The unsafe caller owns the encoders it opens and the ABIs it asserts. |

Device, queue and allocation-domain identity live in one `Arc<ContextHolder>` that
every buffer, bound function and completion also holds. Dropping the `Runtime` while a
submission is still running therefore cannot release the queue under it, and Metal
retains the resources a command buffer encoded.

Submission stays a thing the caller does. `close` submits the open session and waits for
everything, because it is an explicit call that returns the error. `drop` does not: an
uncommitted session is discarded, so a launch nobody committed never runs, and work that
*was* committed is waited for.

Every wait is bounded by `Config::completion_timeout_ms` (60 s by default), which covers
`Completion::wait`, the waits inside `synchronize` and `close`, and the ones `drop` makes.
A stalled submission therefore becomes `Error::Timeout` in the first two; `drop` cannot
report anything, so the first timeout ends its wait and the remaining submissions are
abandoned. Metal retains the buffers a command buffer references, so abandoning never frees
memory the GPU is still reading.

What abandoning does not promise is that a caller can drop *every* handle — runtime, buffers
and completions — while the GPU is still running the work; at that point this crate releases
the queue and the device with a command buffer outstanding. A process exiting with work in
flight does the same thing, so this is a limit to know about rather than a hazard to design
around: `close` is the way to finish the work and see the error.

The first version of this file said drop "makes the same best-effort call" as close, and
the code did: `Drop` called `drain`, which submitted the pending session and threw away
whatever it returned. That contradicted what `launch` documents and what a caller would
assume from a method named `drop`, and a review of the PR reproduced it — a launch with no
commit still landed its result. The rule now is the one the docs already promised: only
`commit`, `synchronize` and `close` submit, and `dropping_a_runtime_discards_work_that_was_never_committed`
holds it.

## 3. Probe before constructing

`probe()` performs, in order: `dlopen` of the frameworks and `libobjc`, symbol
resolution, `MTLCreateSystemDefaultDevice` (nil ⇒ `Error::Unavailable`, never a panic),
class presence, the per-dtype MSL compiles, and one real dispatch
(`[1,2,3,4] + [10,10,10,10]`, exact equality because integer-valued f32 arithmetic is
exact). `Runtime::open` repeats only the cheap half — frameworks plus a device plus a
command queue — and returns the same typed error, so a multi-backend binary can probe
once and only open runtimes that answered.

An f32 rejection is not a capability the caller interprets: `probe` propagates the
compiler's own `Error::Compile`, because a backend that cannot compile f32 serves
nothing.

The dtype answer comes from the compiler, not from a declaration:

| Compiles | On this host |
| --- | --- |
| f32 | yes |
| fp16 (`half`) | yes |
| bf16 (`bfloat`) | yes |
| c32 (`float2`) | yes |
| f64 (`double`) | **no** — `program_source:4:26: error: 'double' is not supported in Metal` |

`capability_report` prints that dump with the SHA-256 of every probe source, which is
what a report about this backend records next to any measured number:

```sh
cargo run -q -p t4a-metal-runtime --example capability_report
# device: Apple M5 Max
# unified_memory: true
# fp16: true ... f64: false
# f64_diagnostic: program_source:4:26: error: 'double' is not supported in Metal
# mpsgraph_class: true
# source f32: a97afa627a0e5e799685d53c99b5a2e95e89fc577dfaa4850dc1adeca3da874d
```

End to end the report takes 0.19–0.24 s wall clock on this host (three runs of the
command above after a warm build; it includes process start, five MSL compiles and one
dispatch). That is a smoke number on a shared machine, not a benchmark.

### MPSGraph re-check

`objc_getClass("MPSGraph")` is **present** once
`MetalPerformanceShadersGraph.framework` is opened. The tenferro-rs#2402 measurement
that reported it absent had not loaded that framework. Class registration is
process-wide, so `Table::class` answers "is this class loaded in this process", not
"did this table open its framework".

No MetalPerformanceShaders selector is registered or called here; MPS GEMM is U5. The
two frameworks are opened so their presence can be reported.

## 4. Submission and completion

A launch encodes into one `MTLComputeCommandEncoder` over one `MTLCommandBuffer`, which
the runtime opens lazily on the first encode. `commit` closes the encoder, creates a
fresh `MTLSharedEvent`, encodes `signal(1)` on it, commits, and returns a `Completion`
for that submission.

- **One event per submission.** A queue orders execution, but the runtime does not read
  that as a guarantee about which submission finished first, so no observation is
  inferred from another submission's value. `synchronize` waits every in-flight
  submission individually.
- **Within one command buffer**, encoder order is deterministic, which is the ordering
  a fused region (U6) relies on.
- **Completion handlers are not used.** `addCompletedHandler:` would need a
  hand-built Objective-C block ABI; `MTLSharedEvent` needs one selector.
- **Bounded.** At most `max_submissions_in_flight` (default 64) submissions are in
  flight; exceeding it waits for the oldest, so a loop that never waits cannot grow.
- **Failure surfaces on observation.** `Completion::poll`/`wait` read the command
  buffer's `error` after its own event signalled and return `Error::Async` with Metal's
  own text. A submission that never completes becomes `Error::Timeout`.

## 5. Allocation and identity

`newBufferWithLength:options:` with `MTLResourceStorageModeShared`. Each `Buffer`
carries an `allocation_id` (unique per runtime, stable for the allocation's life) and a
`domain_id` (unique per opened runtime). A zero-length request is represented without
asking the device, because Metal may return nil for it and an empty buffer has no
address; binding one as a kernel argument is `Error::InvalidArgument`. A request above
the device's `maxBufferLength`, or a nil return, is `Error::OutOfMemory`.

`alloc_zeroed` zeroes in place before the buffer is reachable by any submission, so it
needs no command buffer and no ordering. There is no device-private mode and no
`newBufferWithBytesNoCopy` (design §4 rejects wrapping host memory).

## 6. Compilation and caches

An artifact is a live `MTLLibrary`; its identity is the SHA-256 of the MSL source.
`bind` is Metal's second compilation step (`newFunctionWithName:` then
`newComputePipelineStateWithFunction:error:`) and is cached per runtime, keyed by
(source hash, entry point). Both caches are bounded by entries and bytes
(`max_artifact_entries`/`_bytes`, `max_pipeline_entries`), and eviction only drops the
cache's reference: a bound function and a compiled artifact keep their object alive.

There is no on-disk artifact cache. Metal's portable form is `MTLBinaryArchive`, which
is a separate mechanism with its own policy; the design asks for a per-runtime module
cache, which is what this is.

The probe kernels are three-line sources compiled from `&'static str` inside the
runtime. They are probe sources, not the kernel catalogue: the catalogue and its
`Target` are U3 in the kernel repository.

## 7. Checked launches

The `i`-th argument binds to buffer index `i`, which is the natural MSL declaration
(`[[buffer(0)]]`, `[[buffer(1)]]`, …). Before anything is encoded the runtime checks:

| Check | Error |
| --- | --- |
| function belongs to this runtime | `ForeignDomain` |
| threadgroup counts non-zero | `InvalidArgument` |
| threadgroup total ≤ pipeline's `maxTotalThreadsPerThreadgroup` (1024 here), with the limit in the message | `InvalidArgument` |
| buffer belongs to this runtime, is non-empty | `ForeignDomain`, `InvalidArgument` |
| binding offset `< len` | `OutOfRange` |
| binding offset a multiple of 4 | `InvalidArgument` |
| by-value parameter non-empty and ≤ 4096 bytes (`setBytes` limit) | `InvalidArgument`, `KernelParameterLimit` |

Coverage is made a constructor rather than a comment, because Metal does not bound-check
an index: a dispatch runs `threadgroups × threads_per_threadgroup` threads, whatever the
buffer lengths are. `LaunchConfig::threads(n)` is one thread per threadgroup, so the grid
touches exactly `n` positions and a kernel needs no guard; `LaunchConfig::tiled(groups,
width)` is the throughput form, and there the kernel has to guard its tail. The first
version of this crate had one constructor, `(threadgroups, threads_per_threadgroup)`, and
every call site passed `(n, n)` — `n²` threads for an `n`-element buffer. Nothing failed:
the extra writes landed past the allocation where nothing read them, and the first `n`
lanes produced the asserted values. `the_grid_covers_exactly_the_requested_threads` is the
regression that makes an overshoot observable by giving the kernel a buffer wider than the
dispatch, so extra threads mark elements the test checks are untouched.

The 4-byte binding floor is a measurement, not a quoted rule: `a_binding_offset_selects_the_tail`
binds at byte 4 on real hardware and checks that the four results land at the second
element of an eight-element buffer. If a future OS rejects it, that test is where it
shows.

Launch stays `unsafe`: MSL has no reflection, so no descriptor check can prove a
kernel's parameter count, the type behind each index, or that it fully overwrites its
outputs. `Arg::pod` takes `Pod` values (plain data, no padding); the runtime is
byte-oriented and does not enforce dtypes — a kernel that declares a type MSL rejects
fails to compile, which is the same measurement the probe reports.

## 8. Errors

`t4a-metal-sys` reports only what it could not open or resolve
(`FrameworkUnavailable`, `MissingSymbol`, `InvalidName`). The runtime maps every load
failure to `Error::Unavailable { what, detail }` and keeps the vocabulary the CUDA
runtime uses where the semantics match: `ForeignDomain`, `Closed`, `InvalidArgument`,
`OutOfRange`, `OutOfMemory`, `Async`, plus Metal-specific `Compile`, `Pipeline`,
`UnknownEntryPoint`, `KernelParameterLimit` and `Timeout`. No entry point panics
because Metal is missing; the non-Apple build compiles the same surface with the loader
path returning `Unavailable`.

## 9. Name mapping to `t4a-cuda-runtime`

| CUDA runtime | Metal runtime | Note |
| --- | --- | --- |
| `Runtime::open(ordinal, config)` | `Runtime::open(config)` | Metal has one system default device, no ordinal |
| `alloc_uninit` / `alloc_zeroed` | same | shared storage, no pool |
| `upload` / `download` / `copy_between` | — | nothing to transfer in a shared domain |
| `compile(&CompileRequest)` | same | MSL source instead of NVRTC |
| `bind(&Artifact, entry)` | same | also builds the pipeline state |
| `unsafe launch(...)` | same | buffer indices instead of a packed parameter buffer |
| `checkpoint()` | `commit()` | one `Completion` per submission |
| `synchronize()` | same | waits every in-flight submission |
| `close()` / `stats()` / `with_raw()` | same | raw seam hands out the command buffer |
| `Completion::poll` / `wait` | same | bounded `wait` returns `Timeout` |
| `domain_id()` | same | unique per opened runtime |

## 10. Not in this unit

MPS GEMM and einsum (U5), the MSL kernel catalogue and `Target` (U3), the MSL fusion
emitter (U6), FFT and linalg (later), the macOS CI lane with `TENFERRO_REQUIRE_METAL`
(U4), disk artifact persistence, device-private storage, multi-GPU.

## 11. Limits of this record

- arm64 macOS only in practice. No selector that returns a struct by value is used, so
  the `objc_msgSend_stret` variant that x86_64 would need is never required — but no
  x86_64 macOS host was available to confirm the crate on that target.
- No CI job runs any of this; the evidence is the local run recorded in the worklog.
- The declared workspace MSRV (`rust-version = "1.75"`) does not build: `thiserror
  2.0.21` needs 1.77 and, in the pre-existing `t4a-cuda-runtime`, `libloading 0.9.0`
  needs 1.88. The new crates build at 1.85; the workspace claim is a pre-existing
  defect recorded in `CONTRIBUTING.md`.
