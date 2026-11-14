# Metal runtime (U2): decisions and verification

Tracking issue: [tenferro-rs#2402](https://github.com/tensor4all/tenferro-rs/issues/2402)
(kept open; U1 merged as `83fe982d` in tenferro-rs and untouched here). Design:
tensor4all-meta `docs/design/metal-backend.md` @ `8befe98` §6, §10 (U2). The mechanism
record is [`docs/design/metal-runtime.md`](../design/metal-runtime.md).

Host for every number below: macOS 26.7.2, Apple M5 Max (40-core GPU, Metal 4, unified
memory), `rustc 1.100.0`, branch `feat/metal-runtime`.

## What was built and why

`crates/t4a-metal-sys` (run-time loaded `libobjc`/Metal/MPS tables, every
`objc_msgSend` cast in one macro-generated module) and `crates/t4a-metal-runtime` (one
device, one queue, shared-mode buffers, MSL compilation, checked launches, submission
and completion, raw seam). Names mirror `t4a-cuda-runtime` except where Metal's
semantics differ; the mapping table is in the design record, §9.

Decisions worth keeping:

- **`MTLSharedEvent`, one per submission, instead of a completion-handler block.** A
  block ABI built by hand is more unsafe code than the event it would replace. One
  event per submission also removes any need to reason about whether a later signal
  implies an earlier command buffer finished.
- **No transfer surface at all.** Shared-mode buffers make `upload`/`download`
  meaningless here; they belong to tenferro's shared allocation domain (design §4.4).
- **No selector returning a struct by value.** `maxThreadsPerThreadgroup` would have
  needed `objc_msgSend_stret` on x86_64; the pipeline's scalar
  `maxTotalThreadsPerThreadgroup` is the limit the launch check actually needs.
- **Absence is injectable in the layer that owns loading.** `t4a-metal-sys` takes a
  `Loader`, so "the framework is not there" is a unit test with no environment
  variable, no test-only hook in production code, and no fake Metal.
- **Probe kernels stay in the runtime.** Three-line sources compiled from `&'static
  str`; the catalogue and its `Target` are U3 in the kernel repository.
- **`MPSGraph` re-check answered.** `objc_getClass("MPSGraph")` is present once
  `MetalPerformanceShadersGraph.framework` is opened; the issue's 2026-11-02
  observation predates loading that framework. Class visibility is process-wide, so
  the sys crate documents `Table::class` as "loaded in this process", not "opened by
  this table".

## Rejected alternatives

| Alternative | Why not |
| --- | --- |
| One crate that loads its own frameworks | The design splits sys from policy, and the injection seam for absent-framework tests belongs with loading |
| `addCompletedHandler:` blocks | Hand-built block ABI beats one selector in unsafe code |
| `waitUntilCompleted` on every commit | Turns every launch into a host round trip and forbids one command buffer per fused region (U6) |
| `newBufferWithBytesNoCopy` over caller memory (the issue body) | Superseded by the shared allocation domain (design §4) |
| Routing f64 to the CPU inside the runtime | Requirement 1/2: the reported capability decides, and the caller routes |
| Disk-persisted compiled libraries | Metal's portable form is `MTLBinaryArchive`, a separate mechanism; the design asks for a per-runtime memory cache |

## Verification

Local, on the host above:

- `cargo fmt --all --check`, `cargo clippy --workspace --all-targets --all-features --
  -D warnings`, `cargo test --workspace`, `RUSTDOCFLAGS=-D warnings cargo doc
  --workspace --no-deps`, `cargo test --workspace --doc`: pass.
- `cargo test -p t4a-metal-runtime --features metal-tests`: 23 hardware tests pass,
  including the exact round trip (`[1,2,3,4] + [10,10,10,10] = [11,12,13,14]`), two
  dispatches ordered inside one command buffer, separate submissions waited by their
  own completion, a by-value `float` parameter, `UninitBuffer::assume_init` after a full
  overwrite, foreign-domain and malformed-argument rejection, the in-flight bound, 1000
  allocations returning the live counters to zero, a binding offset of 4 landing in the
  middle of a buffer, a grid that covers exactly the requested threads and nothing more, a
  panicking raw scope discarding its session, a drop that discards uncommitted work, one
  that completes committed work, one that abandons a submission past its timeout while the
  wait that made it reports `Timeout`, and the raw seam.
- Absence: with `create_system_default_device` replaced by a function returning nil,
  `Runtime::open` and `probe()` both return `Error::Unavailable { what: "no Metal
  device" }` and nothing panics; a `Loader` that refuses `Metal.framework` or
  `objc_retain` produces the typed error naming the framework or symbol.
- No build-time link: `otool -L` on the `metal_single` binary lists only
  `libiconv.2.dylib` and `libSystem.B.dylib`; no Metal, MPS or Accelerate entry.
- Off-Apple path: `cargo check --workspace --all-targets --all-features --target
  x86_64-unknown-linux-gnu` passes, so the `#[cfg(not(target_os = "macos"))]` stubs
  compile and the doctests are written to pass on a host without Metal.
- `capability_report` (this host): f32/fp16/bf16/c32 compile, `double` is rejected with
  `error: 'double' is not supported in Metal`, `mpsgraph_class: true`, and the source
  hashes are printed. End to end 0.19–0.24 s wall clock over three warm runs — a smoke
  number on a shared machine, not a benchmark.

Not verified, and why:

- **MSRV.** `cargo +1.75.0 check --workspace --all-targets` fails, but not on this
  change: `thiserror 2.0.21` requires rustc 1.77, and the pre-existing
  `t4a-cuda-runtime` pulls `libloading 0.9.0`, which requires 1.88. The two new crates
  check clean at 1.85.0. The workspace `rust-version = "1.75"` claim is a pre-existing
  defect that `CONTRIBUTING.md` already records as unenforced; fixing it is a
  repository-wide decision and is not bundled here.
- **Missing repository gate.** The shared rules require
  `scripts/check-public-error-docs.py` for the public `# Errors` audit; this repository
  has no such script and no step running it. The `# Errors` sections are written by
  hand, and the missing gate is a finding for the maintainer rather than something to
  add inside this unit.
- **x86_64 macOS.** No such host was available. No struct-returning selector is used, so
  the code should be portable, but that is reasoning, not a run.
- **Hosted CI.** This repository's only job is `ubuntu-latest`; there is no macOS lane.
  Design §9 puts that lane in U4.

## Plan review

Plan reviewed before implementation by `ollama-cloud/glm-5.3-flash` (GLM-5.3-flash),
read-only, in a separate pi process: round 1 `VERDICT: CHANGES` (no blocker, five
important findings — cross-command-buffer completion inference, undefined
`Runtime::drop` lifetime, doctests that would fail off Apple, the missing error-docs
gate, and the unverified MSRV claim), all accepted and folded into
`IMPLEMENTATION-PLAN.md`; round 2 `VERDICT: APPROVE`, no blocker, no important finding.

## Finished-diff review

Reviewed after implementation by `openai/gpt-6.1-sol` (read-only, separate pi process,
both rounds in this session; same model for the plan and the diff), against the committed
diff of `e13666d`. Round 1 `VERDICT: CHANGES`, one blocker and two important findings;
six of the seven findings were fixed and one was dismissed:

- Blocker: three public `Result`-returning items had no `# Errors` (`Objc::sel`,
  `Loader::open`, `Loader::symbol`), which is the A10 bar — fixed. `cargo clippy` does
  not catch this because the matching lint is `clippy::pedantic`, which is not enabled
  here, so the missing checker script is the reason it had to be found by reading.
- Important: `probe`'s `# Errors` promised `Error::Compile` where the code returned
  `Unavailable`, and the always-true `Capabilities.fp32` field was the reason for the
  confusion — fixed by propagating the real `Compile` error and deleting the field.
- Important: `with_raw` documented that a panicking closure leaves a session a caller
  must not commit, but no API could discard it — fixed: the closure runs under
  `catch_unwind`, a panic drops the uncommitted command buffer, and the panic is
  resumed; `a_panicking_raw_scope_discards_its_session` covers it.
- Minor: the binding-offset rule was uncited and never exercised — fixed with the offset-4
  hardware test above, and the constant now says it is a measurement.
- Minor: unused `objc` parameter in `utf8_owned` — deleted, with the caller.
- Minor: `libc` was a target-independent dependency while only the macOS loader uses it —
  moved under `[target.'cfg(target_os = "macos")'.dependencies]`.
- Minor, dismissed: dropping `Capabilities.fp32` fixed the dead-state finding already.

Round 2 `VERDICT: APPROVE`: all eight items verified in the tree, with three new minor
dead-API findings, all deleted rather than documented — `objc::null_id` (no caller),
`Artifact::request` (no caller) and `BoundFunction::max_threads` (a copy of
`PipelineState::max_threads`).

## Post-PR review finding: the grid was the square of the element count

The first commit's only constructor was `LaunchConfig::new(threadgroups,
threads_per_threadgroup)`, and all twelve call sites passed `(n, n)`: `n` threadgroups of
`n` threads, so `n²` threads indexing an `n`-element buffer. The `probe` roundtrip, the
crate-level doctest and ten hardware tests all dispatched that way and all passed, because
Metal does not bound-check `thread_position_in_grid`: the surplus threads wrote past the
allocation, where nothing read them, and the first `n` lanes produced exactly the asserted
values. A review of the PR caught it.

Fixed by making coverage part of the API instead of a comment:

- `LaunchConfig::threads(n)` — one thread per threadgroup, so the grid is exactly `n`
  positions. Always legal, including above the pipeline's per-threadgroup limit, which a
  one-thread group never reaches.
- `LaunchConfig::tiled(groups, width)` — the throughput form, named so that a caller who
  wants many threads per group has to read that its kernel must guard the tail.
- `new` is gone: no name is left that takes two counts without saying what the product is.

Every call site was audited and moved: `probe::roundtrip` and the `lib.rs` doctest now
dispatch `threads(n)`, and eleven hardware tests do — the ten that were moved plus the
regression below. The two that
deliberately need a wide or empty group (the per-threadgroup limit rejection and the
zero-threadgroup rejection) use `tiled(1, limit + 1)` and `tiled(0, 4)` and are rejected
before anything is encoded. Device-free `LaunchConfig` tests in `src/kernel/tests/mod.rs`
pin the grid product, the above-limit legality and the naming of the limit in the error.

The regression is `the_grid_covers_exactly_the_requested_threads`: a `mark` kernel
increments the element at its index in a **sixteen**-element buffer while the dispatch asks
for four threads, then the test requires `[1,1,1,1,0,…]`. Making the overshoot land inside
the allocation is the point, because that is what the old tests could not see. Verified by
putting `tiled(4, 4)` back for one run: it fails with `[1; 16]`.

## Post-PR review finding: drop submitted work nobody committed

`Drop for Runtime` called `drain()`, which submits the open session — so a `launch` that was
never committed still ran when the runtime was dropped, and the error the submit produced was
thrown away, because a drop cannot return one. That directly contradicted two lines this
crate already documented: `Runtime::launch` ("nothing is submitted until `commit`,
`synchronize` or `close`") and the crate-level bullet that a submission is explicit. A second
review of the PR reproduced it: launch, no commit, drop, and the buffer held the sums.

The contract chosen is the one the docs already promised, in the direction that cannot hide
an error: **drop discards the uncommitted session and waits only for work that was already
committed.** Submitting inside `drop` would mean that whether a kernel ran depended on where
a `drop` happened, with no way to observe the submission's result.

- `Drop` now drops `self.session` and then waits the in-flight completions with the runtime's
  own bound (see the next finding). `close` still submits and waits, because it is an
  explicit call that returns the error; `synchronize` is unchanged.
- `Runtime::launch`'s doc and the crate-level bullet now say what happens to encoded work
  nobody submits.
- Two hardware regressions: `dropping_a_runtime_discards_work_that_was_never_committed`
  (launch, no commit, drop, buffer unchanged — verified to fail on the old `drain` with
  `[11,12,13,14]`) and `dropping_a_runtime_completes_work_that_was_committed` (commit, drop,
  results present), so the fix cannot silently turn into "drop loses work".

## Post-PR review finding: "unbounded wait" was not true

The previous fix's comment claimed drop's wait was unbounded and that only `synchronize`
had a bound. Both halves were wrong: `Submission::wait` always stops at the runtime's
`completion_timeout_ms` and returns `Error::Timeout`, so drop's `let _ = completion.wait()`
was a bounded wait whose error was silently dropped. The design record and this worklog had
copied the same false claim.

The contract kept is the bounded one, because it is the one the code can honour and the one
that cannot wedge a process behind a dead GPU:

- `Config::completion_timeout_ms` bounds **every** wait: `Completion::wait`, the waits
  behind `synchronize` and `close`, and drop's.
- `synchronize` and `close` report `Error::Timeout`; `drop` cannot report anything, so the
  first timeout ends its wait and the remaining submissions are abandoned. That is safe
  because Metal retains the resources a committed command buffer references.
- The `Drop` comment, `close`/`synchronize` docs, `Config::completion_timeout_ms`'s doc and
  design §2 now state one thing. The "unbounded" claim is gone from all of them.
- Regression `a_wait_that_exceeds_the_completion_timeout_is_reported_and_drop_abandons_it`:
  300 M fma iterations (about 3.9 s on this host, measured as 1.3 s per 100 M) observed with
  a 1 ms bound, so `wait()` must report `Timeout { millis: 1 }` and `drop` must return in
  under a second instead of waiting out the kernel — a margin wide enough for a loaded host
  and nowhere near the kernel's own duration, which is what makes the test non-vacuous.

## Remaining limits

- `Runtime::drop` discards an uncommitted session and gives up on a committed submission
  after `completion_timeout_ms`; `close` is the way to flush on the way out and to see the
  error. Dropping every handle while abandoned work still runs releases the queue and device
  with a command buffer outstanding — the same situation as a process exiting with work in
  flight, recorded in design §2.
- No on-disk artifact cache (`MTLBinaryArchive`), no device-private storage, no
  multi-GPU.
- MPS and MPSGraph are opened and probed only; no MPS call exists yet (U5).
