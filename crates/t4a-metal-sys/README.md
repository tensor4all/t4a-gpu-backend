# t4a-metal-sys

Run-time loaded Metal tables for tensor4all: the Objective-C runtime, `Metal.framework`,
and (optionally) `MetalPerformanceShaders` and `MetalPerformanceShadersGraph`, resolved
with `dlopen`/`dlsym` and `objc_getClass` on first use.

Nothing is linked at build time, so `otool -L` on a binary that depends on this crate
shows no Metal, MetalPerformanceShaders or Accelerate entry. There is no build script,
no `links =` key and no `-framework` flag, and the `objc2-metal` bindings are not used
for exactly that reason.

This crate has no safe API and no policy. It does not decide whether a device is
usable, which dtypes work, or which kernels to compile: it reports which framework,
image or symbol it could not obtain, and a consumer turns that into its own typed
error. Every `objc_msgSend` cast in the repository lives in `src/objc.rs`, one
newtype per selector, so the signatures can be audited in one place.

`Table::load_with(&dyn Loader)` is the seam the absent-framework tests use: a loader
that refuses a framework produces the same typed error a host without that framework
produces, with no environment variable and no test-only branch in the loader.

The mechanism record and the measured capability dump are in
[`docs/design/metal-runtime.md`](../../docs/design/metal-runtime.md).
