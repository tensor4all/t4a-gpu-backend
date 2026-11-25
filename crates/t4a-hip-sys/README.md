# t4a-hip-sys

Run-time loaded HIP, rocBLAS and rocSOLVER tables for tensor4all: `libamdhip64`,
`librocblas` and `librocsolver`, opened with `dlopen(RTLD_NOW | RTLD_LOCAL)` on first use.

Nothing is linked at build time, so `readelf -d` on a binary that depends on this crate
lists no ROCm library, and a host without ROCm gets a typed `LibraryUnavailable` error
instead of failing to start.

Three load-time rules live here because getting them wrong produces misleading failures
rather than clean errors:

- **One prefix, one HIP runtime.** rocBLAS and rocSOLVER are opened only from the
  directory the HIP runtime was mapped from, and the process is checked for a second
  `libamdhip64` after each one (`DuplicateRuntime`).
- **No empty `HSA_OVERRIDE_GFX_VERSION`.** Exported but empty, it makes
  `rocblas_create_handle` fail with `rocblas_status_invalid_handle`, which reads like a
  missing dtype. It is refused by name before anything is opened, and never modified.
- **Explicit pointer classes.** The rocSOLVER signatures mark which arguments must be
  device memory, and `PointerAttributes` lets a consumer check a pointer's class before a
  call that would page-fault and hang on the wrong one.

This crate has no safe API and no policy; `t4a-hip-runtime` owns streams, pools and
their lifetimes. See [`docs/design/hip-runtime.md`](../../docs/design/hip-runtime.md).
