# t4a-gpu-backend

Independent GPU runtimes for tensor4all.

This repository is being developed separately from tenferro so that execution, memory ownership, transfers and kernel interfaces can be validated before a consumer changes its backend.

## Status

Design and initial repository setup. No runtime or numerical kernel implementation is present yet.

The first milestone is **[t4a-cuda-runtime](https://github.com/tensor4all/t4a-gpu-backend/issues/1)**: a standalone Rust CUDA execution foundation built on cudarc and CUDA APIs. Its design issue is the source of truth for scope, ownership contracts and acceptance criteria.

## Boundaries

The CUDA runtime provides:

- one explicitly owned real CUDA stream and allocation domain per runtime;
- independent memory pools and caches, including when two runtimes open the same GPU;
- owned device/pinned memory and explicit ordered transfers;
- asynchronous lifetime, completion and retirement management;
- CUDA module loading, compilation and kernel launch interfaces.

The foundational runtime does not depend on tenferro Tensor, graph, CPU-provider or AD crates, or on CubeCL/CubeK. Tensor semantics, graph placement and differentiation stay in consumers. Reusable native CUDA kernels live in the separate [t4a-cuda-kernels repository](https://github.com/tensor4all/t4a-cuda-kernels), with its [migration design](https://github.com/tensor4all/t4a-cuda-kernels/blob/main/docs/design/cubecl-kernel-port.md). That kernel crate does not depend on t4a-cuda-runtime; consumers can combine either project with their own CUDA infrastructure. Vendor adapters can be developed separately without moving Tensor semantics into the runtime.

HIP/ROCm, Metal and Intel GPU implementations are future directions. Each can use its native execution model. Shared numerical specifications and tests should grow from concrete implementations; a universal runtime abstraction or kernel language is not a prerequisite.

## Initial layout

The planned first workspace member is `crates/t4a-cuda-runtime/`. Only the CUDA milestone is in scope initially; placeholder crates for every future platform are unnecessary. Registry publication is deferred.

## Validation and adoption

Standalone correctness tests will cover two independent runtimes on one physical GPU, real multi-GPU copies, allocation reuse, raw kernels, errors and teardown. Performance measurements distinguish submission, synchronization and transfer costs.

The [tenferro adoption tracker](https://github.com/tensor4all/tenferro-rs/issues/2208) records evaluation and a later maintainer decision. Creating or implementing the standalone runtime does not itself switch tenferro's existing backend.

Development follows the applicable [tensor4all agent rules](https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/index.md). Implementation should add the focused design, API documentation and tests described in the milestone issue.
