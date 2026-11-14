# t4a-gpu-backend

Follow the [Tensor4all agentic rules](https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/index.md). Read the relevant common and Rust rules; do not copy them into this repository.

For the CUDA runtime, [issue #1](https://github.com/tensor4all/t4a-gpu-backend/issues/1) defines scope and acceptance. [`docs/design/cuda-runtime.md`](docs/design/cuda-runtime.md) is a draft design, not an implementation or API approval.

For the Apple GPU runtime (`crates/t4a-metal-sys`, `crates/t4a-metal-runtime`), [tenferro-rs#2402](https://github.com/tensor4all/tenferro-rs/issues/2402) is the tracking issue and [tensor4all-meta `docs/design/metal-backend.md`](https://github.com/tensor4all/tensor4all-meta/blob/main/docs/design/metal-backend.md) is the approved design; [`docs/design/metal-runtime.md`](docs/design/metal-runtime.md) records what U2 implemented.
