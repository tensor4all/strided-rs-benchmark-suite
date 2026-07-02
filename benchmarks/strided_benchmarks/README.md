# Strided Benchmarks

This directory contains kernel-level benchmarks for `strided-rs` components.
Each benchmark page keeps the measurement code and the published results
together.

## Pages

| Page | Compares | HPTT coverage |
|---|---|---|
| [Permutation kernels](permute/README.md) | JSON-defined permutation patterns comparing naive, `strided_perm`, HPTT, Julia Base, and Strided.jl where applicable | Included in the unified Rust runner for directly matching contiguous-source transpose cases |
| [Transpose-scale kernels](transpose_scale/README.md) | raw pointer naive loops versus `copy_transpose_scale_into`, `map_into`, and `strided_perm::copy_into` where applicable | Not included yet; add only if the HPTT runner covers the same scale/update semantics |

## Result Policy

- Keep setup/allocation outside timed regions unless the page says otherwise.
- Record the `strided-rs` git hash beside measured results.
- Do not publish HPTT rows unless the case is directly comparable and checked
  for correctness.
- Run thread-count variants sequentially.
