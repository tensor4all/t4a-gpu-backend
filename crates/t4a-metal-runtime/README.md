# t4a-metal-runtime

A standalone Metal execution runtime: one device, one command queue and one
allocation-domain identity per runtime, shared-mode `MTLBuffer` allocations, MSL
compiled at run time, checked kernel launches, and explicitly committed command
buffers.

It depends on no tensor, graph, AD, CPU-provider, CubeCL/CubeK or vendor-library crate,
and it does not link a framework: the Metal tables come from `t4a-metal-sys`, which
opens them with `dlopen` on first use.

There are no transfers. Every buffer is in Metal's shared storage mode, so the CPU and
the GPU read and write the same address; `upload`, `download` and pinned staging do not
exist here, because a shared allocation domain has nothing to transfer. The consumer
that pairs this runtime with a CPU engine owns that story.

No implicit fallback either. `probe()` reports what this host's MSL compiler accepts —
on Apple silicon that is f32, fp16, bf16 and `float2`, and it rejects `double` — and the
caller decides which dtype goes to which engine. A backend that is not there is a typed
error, never a panic.

Validation: `cargo test --workspace` covers the paths that need no device;
`cargo test -p t4a-metal-runtime --features metal-tests` runs the hardware suite and
fails rather than skipping when no device is usable. `cargo run -p t4a-metal-runtime
--example capability_report` prints the capability dump with the SHA-256 of every probe
source.

The mechanism record, the name mapping to `t4a-cuda-runtime` and the measured numbers
are in [`docs/design/metal-runtime.md`](../../docs/design/metal-runtime.md).
