<!-- unified contribution policy: ../../docs/contributing/rust-repositories.md in tensor4all/tensor4all-meta -->
# Contributing to strided-rs-benchmark-suite

The rules that apply to every Rust repository in `tensor4all` are in one place:
**[tensor4all-meta · docs/contributing/rust-repositories.md](https://github.com/tensor4all/tensor4all-meta/blob/main/docs/contributing/rust-repositories.md)**.

This repository follows [`strided-rs`](https://github.com/tensor4all/strided-rs/blob/main/CONTRIBUTING.md) for its local
facts — MSRV, feature combinations, platforms and the local gate — because the suite benchmarks that workspace and
carries no CI of its own. The `parallel` and `hptt` features here mirror `strided-rs`'s parallel lane; the README
records the build commands for each.
