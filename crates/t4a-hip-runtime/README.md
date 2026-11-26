# t4a-hip-runtime

A standalone HIP execution runtime: one real non-blocking stream, one explicitly created
`hipMemPool` and one allocation-domain identity per runtime, stream-ordered buffers,
blocking host copies, pointer classification, and a raw scope through which a vendor
library (rocBLAS, rocSOLVER) runs on that exact stream with pool-owned workspace.

It depends on no tensor, graph, AD or vendor-library crate, and links no ROCm library:
the tables come from `t4a-hip-sys` at run time.

This is stage 1 of [tenferro-rs#2489](https://github.com/tensor4all/tenferro-rs/issues/2489).
The `Rocblas` seam that proves the contract — the ROCm mirror of tenferro-gpu's `Cublas`
and `GemmRequest` — lives in this crate's test target (`tests/support/rocblas_seam.rs`),
as vendor adapters live in the consumer on the CUDA side.

Validation: `cargo test -p t4a-hip-runtime` runs everywhere and reports a typed reason
for each hardware test it cannot run; `T4A_REQUIRE_HIP=1` turns a missing device into a
failure. `cargo run --release -p t4a-hip-runtime --example rocblas_gflops` records GEMM
throughput per dtype and size. Measurements, mechanisms and limits are in
[`docs/design/hip-runtime.md`](../../docs/design/hip-runtime.md).
