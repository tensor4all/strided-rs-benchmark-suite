# Einsum Step 408 Benchmark

Reproduces the bottleneck step of `tensornetwork_permutation_light_415`.
This page is a focused einsum case study for one late contraction step. It is
not intended to summarize the full benchmark suite or make a broad
Rust-vs-Julia claim.

Binary einsum with all dims = 2: A(13 dims, 8K) × B(24 dims, 16M) → C(18 dims, 262K).
m=4, k=256, n=8192, batch=8.

## Run

```bash
# Rust (blas backend)
cargo run --release --no-default-features --features blas --bin step408_bench

# Julia
julia --project=. benchmarks/einsum_benchmarks/step408/step408_bench.jl

# Fair comparison (scrambled vs natural labels)
julia --project=. benchmarks/einsum_benchmarks/step408/step408_fair.jl
```

## Results (Apple M5 MacBook Pro)

Measured on macOS, 2026-07-02. macOS runs are not CPU-pinned.

- Rust compiler flags: `RUSTFLAGS="-C target-cpu=native"`
- strided-rs: `91a0aca`
- benchmark suite: this commit
- Julia environment: Manifest-instantiated `OMEinsum v0.9.3`

### Rust BLAS

Median milliseconds. Run with
`cargo run --release --no-default-features --features blas --bin step408_bench`.

| Threads | full scattered | copy B | copy A | contiguous GEMM |
|---:|---:|---:|---:|---:|
| 1 | 16.805 | 12.302 | 0.006 | 5.193 |
| 4 | 16.759 | 12.504 | 0.005 | 5.183 |

### Julia Decomposition

Median milliseconds. Run with `julia --project=. benchmarks/einsum_benchmarks/step408/step408_bench.jl`.
The manual `permutedims! + BLAS gemm` path in this script currently reports a
large max error, so treat this table as component timing, not correctness
validation.

| Threads | OMEinsum DynamicEinCode | permutedims! + BLAS | permutedims! B | permutedims! A | BLAS GEMM |
|---:|---:|---:|---:|---:|---:|
| 1 | 5.926 | 27.941 | 22.110 | 0.008 | 5.770 |
| 4 | 2.197 | 24.692 | 22.370 | 0.008 | 2.043 |

### Julia Fair Comparison

Median milliseconds. Run with `julia --project=. benchmarks/einsum_benchmarks/step408/step408_fair.jl`.

| Threads | OMEinsum scrambled | OMEinsum natural | permutedims! B Rust perm | permutedims! B batch-front | BLAS GEMM |
|---:|---:|---:|---:|---:|---:|
| 1 | 27.695 | 5.848 | 21.773 | 16.779 | 5.765 |
| 4 | 24.549 | 2.132 | 21.832 | 16.909 | 2.072 |
