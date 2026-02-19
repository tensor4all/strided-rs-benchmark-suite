# Step 408 Micro-benchmark

Reproduces the bottleneck step of `tensornetwork_permutation_light_415`.

Binary einsum with all dims = 2: A(13 dims, 8K) × B(24 dims, 16M) → C(18 dims, 262K).
m=4, k=256, n=8192, batch=8.

## Run

```bash
# Rust (blas backend)
cargo run --release --no-default-features --features blas --bin step408_bench

# Julia
julia --project=. micro_bench/step408_bench.jl

# Fair comparison (scrambled vs natural labels)
julia --project=. micro_bench/step408_fair.jl
```

## Results (AMD EPYC 7713P)

### 1T

| Operation | Rust (blas) | Julia |
|---|---|---|
| Full einsum (scrambled labels) | 300 ms | 108 ms |
| Full einsum (natural labels) | — | 19 ms |
| copy_into/permutedims B (scattered perm) | 236 ms | 58 ms |
| GEMM only | 63 ms | 18 ms |

### 4T

| Operation | Rust (blas) | Julia |
|---|---|---|
| Full einsum (scrambled labels) | 313 ms | 102 ms |
| Full einsum (natural labels) | — | 5 ms |
| copy_into/permutedims B (scattered perm) | 241 ms | 63 ms |
| GEMM only | 62 ms | 5 ms |

## Root Causes

1. **`strided_perm::copy_into` is 4x slower than Julia's `permutedims!`** for 24-dim binary tensors with scattered permutation. Julia's `permutedims!` uses a simpler recursive approach that is more cache-friendly for high-rank small-dim tensors.

2. **Batched GEMM overhead**: Rust's einsum2 dispatches 8 separate GEMM calls with m=4, k=256, n=8192. The per-call overhead (plan building, validation, prepare_input_owned checks) adds up. Julia's `mul!` goes directly to BLAS with minimal overhead.

3. **Label ordering**: In the full benchmark, OMEinsum.jl stores intermediates contiguously with sorted labels, so subsequent steps need minimal permutation. strided-rs uses lazy permutation, causing accumulated scrambled strides that require expensive copy-to-contiguous.
