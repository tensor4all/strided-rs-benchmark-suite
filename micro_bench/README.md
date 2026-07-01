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

Median milliseconds. Run with `julia --project=. micro_bench/step408_bench.jl`.
The manual `permutedims! + BLAS gemm` path in this script currently reports a
large max error, so treat this table as component timing, not correctness
validation.

| Threads | OMEinsum DynamicEinCode | permutedims! + BLAS | permutedims! B | permutedims! A | BLAS GEMM |
|---:|---:|---:|---:|---:|---:|
| 1 | 5.926 | 27.941 | 22.110 | 0.008 | 5.770 |
| 4 | 2.197 | 24.692 | 22.370 | 0.008 | 2.043 |

### Julia Fair Comparison

Median milliseconds. Run with `julia --project=. micro_bench/step408_fair.jl`.

| Threads | OMEinsum scrambled | OMEinsum natural | permutedims! B Rust perm | permutedims! B batch-front | BLAS GEMM |
|---:|---:|---:|---:|---:|---:|
| 1 | 27.695 | 5.848 | 21.773 | 16.779 | 5.765 |
| 4 | 24.549 | 2.132 | 21.832 | 16.909 | 2.072 |

# Permutation Micro-benchmark

Migrated from `strided-perm/benches/permute.rs`. Covers the high-rank binary
permutation, contiguous-source permutation, memcpy baseline, small tensors, and
large 3D transposes.

## Run

Run these sequentially. Do not run thread-count variants at the same time.

```bash
RUSTFLAGS="-C target-cpu=native" \
  RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  cargo run --release --bin permute

RUSTFLAGS="-C target-cpu=native" \
  RAYON_NUM_THREADS=4 OMP_NUM_THREADS=1 \
  cargo run --release --features parallel --bin permute
```

## Results (Apple M5 MacBook Pro)

Measured on macOS, 2026-07-02. macOS runs are not CPU-pinned.

- compiler flags: `RUSTFLAGS="-C target-cpu=native"`
- strided-rs: `91a0aca`
- benchmark suite: this commit

Median milliseconds.

### 1T

| Scenario | Path | ms |
|---|---|---:|
| memcpy baseline | `std::ptr::copy_nonoverlapping` | 1.911 |
| memcpy baseline | `copy_into` contiguous fast path | 1.913 |
| 24D scattered -> col-major | naive odometer | 20.212 |
| 24D scattered -> col-major | `copy_into` | 5.124 |
| 24D scattered -> col-major | `copy_into_col_major` | 5.153 |
| 24D contig source, same perm | `copy_into` | 3.186 |
| 24D contig -> contig perm | `copy_into` | 3.165 |
| 24D contig -> contig perm | `copy_into_col_major` | 3.206 |
| 24D contig -> contig perm | naive odometer | 18.167 |
| 13D small reverse | `copy_into` | 0.004 |
| 13D small cyclic | `copy_into` | 0.003 |
| 13D small reverse | naive odometer | 0.009 |
| 3D 256^3 [2,0,1] | `copy_into` | 14.416 |
| 3D 256^3 [2,0,1] | `copy_into_col_major` | 14.454 |
| 3D 256^3 [2,0,1] | naive odometer | 61.731 |
| 3D 256^3 [1,0,2] | `copy_into` | 11.078 |

### 4T

| Scenario | Path | ms |
|---|---|---:|
| memcpy baseline | `std::ptr::copy_nonoverlapping` | 1.916 |
| memcpy baseline | `copy_into` contiguous fast path | 1.916 |
| 24D scattered -> col-major | naive odometer | 20.140 |
| 24D scattered -> col-major | `copy_into` | 5.116 |
| 24D scattered -> col-major | `copy_into_col_major` | 5.157 |
| 24D scattered -> col-major | `copy_into_par` | 3.048 |
| 24D scattered -> col-major | `copy_into_col_major_par` | 3.253 |
| 24D contig source, same perm | `copy_into` | 3.280 |
| 24D contig -> contig perm | `copy_into` | 3.214 |
| 24D contig -> contig perm | `copy_into_col_major` | 3.222 |
| 24D contig -> contig perm | naive odometer | 18.945 |
| 24D contig -> contig perm | `copy_into_par` | 1.152 |
| 13D small reverse | `copy_into` | 0.004 |
| 13D small cyclic | `copy_into` | 0.002 |
| 13D small reverse | naive odometer | 0.009 |
| 13D small reverse | `copy_into_par` | 0.004 |
| 3D 256^3 [2,0,1] | `copy_into` | 13.315 |
| 3D 256^3 [2,0,1] | `copy_into_col_major` | 13.294 |
| 3D 256^3 [2,0,1] | naive odometer | 45.754 |
| 3D 256^3 [2,0,1] | `copy_into_par` | 4.792 |
| 3D 256^3 [1,0,2] | `copy_into` | 11.056 |
| 3D 256^3 [1,0,2] | `copy_into_par` | 3.244 |

# Scale Transpose Micro-benchmark

Tracks 2D transpose-scale paths where a raw pointer naive loop previously
matched or beat the strided implementation.

Cases:

- dtypes: `f32`, `f64`, `c32`, `c64`, `u64`
- sizes: `1000`, `1024`, `2048` by default
- scales: `0`, `1`, `3`
- paths: raw pointer naive, `copy_transpose_scale_into`, `map_into` on a
  transposed view, and `strided_perm::copy_into` for `scale == 1`

## Run

Run these sequentially. Do not run thread-count variants at the same time.

```bash
# Exploratory local run
RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  cargo run --release --bin scale_transpose

RAYON_NUM_THREADS=4 OMP_NUM_THREADS=1 \
  cargo run --release --features parallel --bin scale_transpose

# Faster smoke run
SIZES=1000 WARMUP=1 NRUNS=3 \
  RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  cargo run --release --bin scale_transpose
```

For README result updates on Linux, pin cores as required by `AGENTS.md`, for
example `taskset -c 0` for 1T and `taskset -c 0-3` for 4T.

## Results (Apple M5 MacBook Pro)

Measured on macOS, 2026-07-02. macOS runs are not CPU-pinned.

- benchmark command: `SIZES=1000,1024,2048 WARMUP=3 NRUNS=11`
- compiler flags: `RUSTFLAGS="-C target-cpu=native"`
- strided-rs: `91a0aca`
- benchmark suite: this commit

### 1T

Median milliseconds. `copy_into` is reported only for `scale=1`.

| dtype | n | scale | naive | strided | map | copy_into |
|---|---:|---:|---:|---:|---:|---:|
| f32 | 1000 | 0 | 0.015 | 0.015 | 0.354 | - |
| f32 | 1000 | 1 | 0.185 | 0.142 | 0.357 | 0.188 |
| f32 | 1000 | 3 | 0.185 | 0.146 | 0.376 | - |
| f32 | 1024 | 0 | 0.016 | 0.016 | 0.751 | - |
| f32 | 1024 | 1 | 1.116 | 0.361 | 0.757 | 0.262 |
| f32 | 1024 | 3 | 1.106 | 0.346 | 0.752 | - |
| f32 | 2048 | 0 | 0.064 | 0.064 | 3.609 | - |
| f32 | 2048 | 1 | 9.124 | 2.266 | 3.549 | 2.595 |
| f32 | 2048 | 3 | 9.019 | 2.303 | 3.481 | - |
| f64 | 1000 | 0 | 0.030 | 0.030 | 0.488 | - |
| f64 | 1000 | 1 | 0.237 | 0.189 | 0.487 | 0.268 |
| f64 | 1000 | 3 | 0.238 | 0.197 | 0.486 | - |
| f64 | 1024 | 0 | 0.032 | 0.032 | 0.998 | - |
| f64 | 1024 | 1 | 1.214 | 0.432 | 0.910 | 0.563 |
| f64 | 1024 | 3 | 1.273 | 0.434 | 0.930 | - |
| f64 | 2048 | 0 | 0.127 | 0.129 | 6.329 | - |
| f64 | 2048 | 1 | 11.709 | 3.367 | 6.271 | 5.031 |
| f64 | 2048 | 3 | 11.754 | 3.610 | 6.370 | - |
| c32 | 1000 | 0 | 0.030 | 0.030 | 0.583 | - |
| c32 | 1000 | 1 | 0.380 | 0.280 | 0.538 | 0.302 |
| c32 | 1000 | 3 | 0.377 | 0.299 | 0.542 | - |
| c32 | 1024 | 0 | 0.031 | 0.032 | 1.048 | - |
| c32 | 1024 | 1 | 1.171 | 0.454 | 0.937 | 0.541 |
| c32 | 1024 | 3 | 1.226 | 0.456 | 0.940 | - |
| c32 | 2048 | 0 | 0.129 | 0.128 | 6.426 | - |
| c32 | 2048 | 1 | 11.633 | 4.064 | 6.446 | 4.994 |
| c32 | 2048 | 3 | 11.625 | 3.881 | 6.468 | - |
| c64 | 1000 | 0 | 0.061 | 0.061 | 1.401 | - |
| c64 | 1000 | 1 | 0.566 | 0.544 | 1.279 | 0.624 |
| c64 | 1000 | 3 | 0.569 | 0.556 | 1.288 | - |
| c64 | 1024 | 0 | 0.063 | 0.063 | 1.760 | - |
| c64 | 1024 | 1 | 2.015 | 0.904 | 1.731 | 3.699 |
| c64 | 1024 | 3 | 1.912 | 0.919 | 1.693 | - |
| c64 | 2048 | 0 | 0.256 | 0.256 | 8.999 | - |
| c64 | 2048 | 1 | 13.671 | 6.846 | 8.947 | 14.839 |
| c64 | 2048 | 3 | 14.519 | 6.885 | 8.940 | - |
| u64 | 1000 | 0 | 0.031 | 0.031 | 0.505 | - |
| u64 | 1000 | 1 | 0.239 | 0.197 | 0.505 | 0.275 |
| u64 | 1000 | 3 | 0.235 | 0.197 | 0.490 | - |
| u64 | 1024 | 0 | 0.032 | 0.032 | 0.993 | - |
| u64 | 1024 | 1 | 1.324 | 0.442 | 0.901 | 0.540 |
| u64 | 1024 | 3 | 1.284 | 0.449 | 0.899 | - |
| u64 | 2048 | 0 | 0.125 | 0.126 | 6.151 | - |
| u64 | 2048 | 1 | 11.841 | 3.500 | 6.096 | 5.008 |
| u64 | 2048 | 3 | 11.910 | 3.512 | 6.152 | - |

### 4T

Median milliseconds. `copy_into` is reported only for `scale=1`.

| dtype | n | scale | naive | strided | map | copy_into |
|---|---:|---:|---:|---:|---:|---:|
| f32 | 1000 | 0 | 0.023 | 0.019 | 0.205 | - |
| f32 | 1000 | 1 | 0.254 | 0.093 | 0.304 | 0.232 |
| f32 | 1000 | 3 | 0.232 | 0.085 | 0.281 | - |
| f32 | 1024 | 0 | 0.018 | 0.018 | 0.413 | - |
| f32 | 1024 | 1 | 1.175 | 0.133 | 0.383 | 0.248 |
| f32 | 1024 | 3 | 1.095 | 0.122 | 0.340 | - |
| f32 | 2048 | 0 | 0.065 | 0.064 | 1.656 | - |
| f32 | 2048 | 1 | 6.786 | 1.366 | 1.605 | 2.608 |
| f32 | 2048 | 3 | 6.964 | 1.174 | 1.448 | - |
| f64 | 1000 | 0 | 0.033 | 0.030 | 0.158 | - |
| f64 | 1000 | 1 | 0.240 | 0.091 | 0.155 | 0.274 |
| f64 | 1000 | 3 | 0.245 | 0.087 | 0.158 | - |
| f64 | 1024 | 0 | 0.031 | 0.031 | 0.280 | - |
| f64 | 1024 | 1 | 1.180 | 0.218 | 0.319 | 0.605 |
| f64 | 1024 | 3 | 1.187 | 0.179 | 0.278 | - |
| f64 | 2048 | 0 | 0.123 | 0.126 | 2.043 | - |
| f64 | 2048 | 1 | 11.260 | 1.809 | 1.983 | 4.988 |
| f64 | 2048 | 3 | 11.560 | 1.728 | 2.044 | - |
| c32 | 1000 | 0 | 0.031 | 0.034 | 0.221 | - |
| c32 | 1000 | 1 | 0.399 | 0.128 | 0.201 | 0.312 |
| c32 | 1000 | 3 | 0.395 | 0.114 | 0.179 | - |
| c32 | 1024 | 0 | 0.032 | 0.033 | 0.326 | - |
| c32 | 1024 | 1 | 1.210 | 0.232 | 0.346 | 0.524 |
| c32 | 1024 | 3 | 1.154 | 0.203 | 0.294 | - |
| c32 | 2048 | 0 | 0.127 | 0.126 | 2.045 | - |
| c32 | 2048 | 1 | 11.488 | 1.814 | 2.019 | 4.996 |
| c32 | 2048 | 3 | 10.816 | 1.788 | 1.937 | - |
| c64 | 1000 | 0 | 0.061 | 0.061 | 0.631 | - |
| c64 | 1000 | 1 | 0.547 | 0.181 | 0.583 | 0.494 |
| c64 | 1000 | 3 | 0.550 | 0.184 | 0.647 | - |
| c64 | 1024 | 0 | 0.064 | 0.065 | 0.802 | - |
| c64 | 1024 | 1 | 1.775 | 0.371 | 0.807 | 3.662 |
| c64 | 1024 | 3 | 1.769 | 0.354 | 0.792 | - |
| c64 | 2048 | 0 | 0.258 | 0.257 | 2.678 | - |
| c64 | 2048 | 1 | 14.510 | 2.241 | 2.598 | 14.580 |
| c64 | 2048 | 3 | 13.556 | 2.304 | 2.650 | - |
| u64 | 1000 | 0 | 0.030 | 0.031 | 0.177 | - |
| u64 | 1000 | 1 | 0.237 | 0.095 | 0.170 | 0.284 |
| u64 | 1000 | 3 | 0.237 | 0.094 | 0.173 | - |
| u64 | 1024 | 0 | 0.031 | 0.032 | 0.296 | - |
| u64 | 1024 | 1 | 1.191 | 0.184 | 0.293 | 0.640 |
| u64 | 1024 | 3 | 1.194 | 0.178 | 0.278 | - |
| u64 | 2048 | 0 | 0.121 | 0.123 | 1.922 | - |
| u64 | 2048 | 1 | 10.893 | 1.800 | 1.955 | 5.141 |
| u64 | 2048 | 3 | 12.024 | 1.764 | 1.897 | - |

# HPTT Comparison Micro-benchmark

Compares HPTT C++ against directly matching contiguous-source tensor transpose
cases. HPTT supports contiguous source tensor transpositions with an optional
output update; cases that require arbitrary source strides or only have
timing-only validation are intentionally omitted.

## Run

Requires an OpenMP-capable C++ compiler. On macOS, install Homebrew GCC and run
with `g++-15`:

```bash
HPTT_DIR=../hptt CXX=/opt/homebrew/bin/g++-15 \
  THREADS=1 WARMUP=3 NRUNS=11 \
  micro_bench/run_hptt_compare.sh

HPTT_DIR=../hptt CXX=/opt/homebrew/bin/g++-15 \
  THREADS=4 WARMUP=3 NRUNS=11 \
  micro_bench/run_hptt_compare.sh
```

The benchmark reports both reusable-plan execution time (`hptt execute`) and
plan construction plus execution (`hptt create+execute`). The runner performs
sample correctness checks for every reported case.

## Results (Apple M5 MacBook Pro)

Measured on macOS, 2026-07-02. macOS runs are not CPU-pinned.

- compiler: Homebrew GCC `g++-15` with `-O3 -DNDEBUG -fopenmp -mcpu=native`
- HPTT source: local clone at `../hptt`

Median milliseconds.

### 1T

| Scenario | HPTT execute | HPTT create+execute |
|---|---:|---:|
| 2D 1024^2 transpose [1,0] | 0.655 | 0.648 |
| 3D 256^3 transpose [2,0,1] | 24.473 | 24.477 |
| 3D 256^3 transpose [1,0,2] | 14.329 | 14.310 |

### 4T

| Scenario | HPTT execute | HPTT create+execute |
|---|---:|---:|
| 2D 1024^2 transpose [1,0] | 0.201 | 0.224 |
| 3D 256^3 transpose [2,0,1] | 7.929 | 7.956 |
| 3D 256^3 transpose [1,0,2] | 4.384 | 4.614 |
