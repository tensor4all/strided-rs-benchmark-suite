# strided-rs-benchmark-suite

Kernel benchmarks for [strided-rs](https://github.com/tensor4all/strided-rs):
permutation, transpose-scale, fused elementwise and dense kernels, each
compared against a credible naive baseline and, where directly comparable,
HPTT, Julia Base and Strided.jl.

The einsum benchmark half (the einsum-benchmark dataset, the
`strided-opteinsum` and OMEinsum.jl runners, and their results) was removed
together with the einsum crates of strided-rs after 0.4.4; it remains in this
repository's git history.

## Benchmark Guides

- [Benchmark index](benchmarks/README.md): entry point for all result pages.
- [Strided benchmarks](benchmarks/strided_benchmarks/README.md)
- [Dense CPU kernels](benchmarks/strided_benchmarks/dense_kernels/README.md):
  migrated AXPBY/structural kernels and shared multiply SIMD, with isolated 1T
  instruction windows and nonzero correctness checks.

## Setup

The suite builds against the published strided 0.7.0 crates from crates.io
(`strided-kernel`, `strided-view`, `strided-perm`); no sibling checkout is
needed. To test against a local strided-rs checkout, add a temporary
`[patch.crates-io]` section to `Cargo.toml` pointing those crates (and any
other strided crate they pull in) at `path = "..."` entries in that checkout.

```bash
cargo build --release                      # serial kernels
cargo build --release --features parallel  # Rayon-backed kernels
cargo build --release --features hptt      # HPTT comparison rows
julia --project=. -e 'using Pkg; Pkg.instantiate()'   # Julia comparisons
```

Each benchmark page documents its command lines, thread counts and CPU
pinning. Record the exact strided-rs git hash beside every published table.
