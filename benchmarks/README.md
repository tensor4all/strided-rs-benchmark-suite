# Benchmarks

Benchmark results are grouped by what is being measured.

## Strided Benchmarks

Kernel-level benchmarks. Each page compares a credible naive baseline against
the relevant `strided-rs` path, and includes HPTT only for directly matching
cases:

- [Permutation kernels](strided_benchmarks/permute/README.md): JSON-defined
  patterns comparing naive, `strided_perm`, HPTT-compatible transpose cases,
  Julia Base, and Strided.jl.
- [Transpose-scale kernels](strided_benchmarks/transpose_scale/README.md): raw
  pointer naive loops, `copy_transpose_scale_into`, `map_into`, and
  `strided_perm::copy_into` where applicable.
- [Fused elementwise kernels](strided_benchmarks/fused_elementwise/README.md):
  per-op reused buffers versus `fused_elementwise_into` static DAG
  specializations and interpreter fallback.

## Result Policy

- Keep benchmark setup and allocation outside timed regions unless the page says
  otherwise.
- Record the `strided-rs` git hash beside measured results.
- Do not publish HPTT rows unless the case is directly comparable and checked
  for correctness.
- Run thread-count variants sequentially.
