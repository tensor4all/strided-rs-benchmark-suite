# Benchmarks

Benchmark results are grouped by what is being measured.

## Einsum Benchmarks

Full contraction benchmarks and focused einsum case studies:

- [Full einsum benchmark suite](einsum_benchmarks/README.md): repository-level
  `strided-opteinsum` versus OMEinsum.jl results.
- [Step 408 bottleneck case study](einsum_benchmarks/step408/README.md):
  focused late-step contraction from `tensornetwork_permutation_light_415`.

## Strided Benchmarks

Kernel-level benchmarks. Each page compares a credible naive baseline against
the relevant `strided-rs` path, and includes HPTT only for directly matching
cases:

- [Permutation kernels](strided_benchmarks/permute/README.md): naive odometer,
  `strided_perm`, and HPTT-compatible transpose cases.
- [Transpose-scale kernels](strided_benchmarks/transpose_scale/README.md): raw
  pointer naive loops, `copy_transpose_scale_into`, `map_into`, and
  `strided_perm::copy_into` where applicable.

## Result Policy

- Keep benchmark setup and allocation outside timed regions unless the page says
  otherwise.
- Record the `strided-rs` git hash beside measured results.
- Do not publish HPTT rows unless the case is directly comparable and checked
  for correctness.
- Run thread-count variants sequentially.
