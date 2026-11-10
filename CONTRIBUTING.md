<!-- unified contribution policy: ../../docs/contributing/rust-repositories.md in tensor4all/tensor4all-meta -->
# Contributing to t4a-gpu-backend

The rules that apply to every Rust repository in `tensor4all` are in one place:
**[tensor4all-meta · docs/contributing/rust-repositories.md](https://github.com/tensor4all/tensor4all-meta/blob/main/docs/contributing/rust-repositories.md)**.
Read that first — it covers the issue-first workflow, what a pull request body has to contain, red-first tests, the
gates every repository runs, the review requirement, and the version policy.

This file only records what is specific to `t4a-gpu-backend`:

## Local facts

- **MSRV**: `rust-version = "1.75"` in the workspace manifest, inherited by `t4a-cuda-runtime`. CI pins `stable` and
  has no `msrv` job, so the claim is not checked by CI.
- **Feature combinations CI builds**: `default` (no features) for the tests, and `--all-features` for clippy, which
  compiles the `gpu-tests` targets that need a real CUDA device. The `gpu-tests` suite is executed on a self-hosted GPU
  host, never in the hosted job; an invoked hardware test on a host without a device fails rather than skipping.
- **Platforms claimed**: `ubuntu-latest` for the single `host` job (fmt, clippy, tests, rustdoc). No macOS or Windows
  job exists. That job does not run the CUDA device tests: `.github/workflows/ci.yml:27-28` only records that they need a real device and run on a self-hosted GPU host, and no workflow here configures such a runner.
- **Local gate**: `cargo test --workspace`.

## Anything here that departs from the shared rules

One does: the `msrv` gate is not run — `rust-version = "1.75"` is declared but nothing builds on 1.75, so the claim
can drift with a dependency bump. That is a gap to close, not an exemption from the other gates.

The GPU test suite runs outside hosted CI, on the self-hosted GPU host; the pull request records the GPU model, the
driver and toolkit versions, the exact command and the observed result.
