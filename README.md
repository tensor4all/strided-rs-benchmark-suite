# strided-rs-benchmark-suite

Benchmark suite for [strided-rs](https://github.com/tensor4all/strided-rs) based on the [einsum benchmark](https://benchmark.einsum.org/) (168 standardized einsum problems across 7 categories).

## Overview

This repository provides:

- A Python pipeline to extract einsum benchmark metadata (shapes, dtypes, contraction paths) into portable JSON
- A **Rust** benchmark runner using [strided-opteinsum](https://github.com/tensor4all/strided-rs)
- A **Julia** benchmark runner using [OMEinsum.jl](https://github.com/under-Peter/OMEinsum.jl)

Only metadata is stored — tensors are generated at benchmark time (zero-filled), keeping the repo lightweight.

See [tensor4all/strided-rs#63](https://github.com/tensor4all/strided-rs/issues/63) for the full design discussion.

## Project Structure

```
strided-rs-benchmark-suite/
  src/
    main.rs                 # Rust benchmark runner (strided-opteinsum)
    main.jl                 # Julia benchmark runner (OMEinsum.jl)
  scripts/
    run_all.sh              # Run all benchmarks (configurable thread count)
    generate_dataset.py     # Filter & export benchmark instances as JSON
    convert_tensornetwork.py # Convert TensorNetworkBenchmarks format to strided-rs JSON
    create_lightweight_instance.py # Extract BFS-connected subgraph for lighter tensor network
    format_results.py       # Parse logs and output markdown tables
  data/
    instances/              # Exported JSON metadata (one file per instance)
    results/                # Benchmark logs and markdown results
  Cargo.toml                # Rust project
  Project.toml              # Julia project
  pyproject.toml            # Python project
```

## Setup

### Python (dataset export)

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

### Rust

Requires a local clone of [strided-rs](https://github.com/tensor4all/strided-rs) at `../strided-rs`.

Two GEMM backends are supported (mutually exclusive):

- **faer** (default) — pure Rust GEMM via [faer](https://github.com/sarah-quinones/faer-rs)
- **blas** — links to OpenBLAS (`brew install openblas` on macOS / `apt install libopenblas-dev` on Ubuntu)

```bash
cargo build --release                                  # faer (default)
cargo build --release --no-default-features --features blas   # OpenBLAS
```

### Julia

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Usage

### 1. Export benchmark metadata

```bash
uv run python scripts/generate_dataset.py
```

This selects instances by category with laptop-scale criteria and saves JSON metadata to `data/instances/`. `rnd_mixed_` instances are excluded (not yet supported by strided-rs).

**Optional: Convert TensorNetworkBenchmarks format**

To add the TensorNetworkBenchmarks tensor network (550 tensors, 2^33.2 complexity) as a strided-rs instance:

```bash
python scripts/convert_tensornetwork.py
```

Requires `TensorNetworkBenchmarks/data/tensornetwork_permutation_optimized.json` at `../TensorNetworkBenchmarks/` relative to the benchmark suite.

**Optional: Create a lightweight tensor network instance**

The full `tensornetwork_permutation_optimized` instance (550 tensors) takes ~40 seconds. To create a lighter version (~5 seconds), extract a connected subgraph with an optimized contraction path:

```bash
uv run python scripts/create_lightweight_instance.py \
  data/instances/tensornetwork_permutation_optimized.json \
  data/instances/tensornetwork_permutation_light_415.json \
  415
```

The script finds a BFS-connected subset of the given size with minimal free indices, sets the output to scalar, and computes an optimized contraction path via `opt_einsum`.

**Optional: Focus on the dominant late contraction steps (step408/step409)**

For kernel-level profiling of the most expensive region in `tensornetwork_permutation_light_415`, this repo also includes:

- `data/instances/tensornetwork_permutation_focus_step409_316.json`

This instance is extracted from the original 415-tensor contraction tree by taking the subtree that ends at original **step 409**. It preserves the expensive late-stage structure (original step 408 and 409), while reducing total tree size from 415 to 316 tensors for faster iteration. Unlike the lightweight scalar instance, this focused instance has a non-scalar output (rank-18), because it represents an internal contraction state.

Use it when you want to benchmark or profile the bottleneck contraction kernels directly:

```bash
BENCH_INSTANCE=tensornetwork_permutation_focus_step409_316 ./scripts/run_all.sh 1
```

**Selection criteria (per category):**

| Category | Prefix | log10[FLOPS] | log2[SIZE] | num_tensors | dtype |
|----------|--------|--------------|------------|-------------|-------|
| Language model | `lm_` | < 10 | < 25 | ≤ 100 | float64 or complex128 |
| Graphical model | `gm_` | < 10 | < 27 | ≤ 200 | float64 or complex128 |
| Structured | `str_` | < 11 | < 26 | ≤ 200 | float64 or complex128 |

### 2. Run all benchmarks

```bash
./scripts/run_all.sh          # 1 thread (default), canonical ids off
./scripts/run_all.sh 4        # 4 threads, canonical ids off
./scripts/run_all.sh 1 1      # 1 thread, canonical [lo, ro, batch] ids on
```

Runs Rust (faer + blas) and Julia benchmarks.
- First argument sets `OMP_NUM_THREADS`, `RAYON_NUM_THREADS`, and `JULIA_NUM_THREADS`.
- Second argument sets `STRIDED_OPTEINSUM_CANONICAL_BINARY_IDS` (`0` or `1`).
If OpenBLAS is not installed, the blas benchmark is skipped with a warning. Results are saved to `data/results/`. To run only one instance, see [Run a single instance](#3-run-a-single-instance) below.

Instance JSON files that fail to read or parse are skipped with a warning; the suite continues with the rest. Instances that trigger a backend error (e.g. duplicate axis labels in strided-opteinsum) are reported as **SKIP** in the table with the reason on stderr.

### 3. Run a single instance

To run the benchmark for **one instance only**, set the environment variable `BENCH_INSTANCE` to the instance name. Useful for heavy instances (e.g. `gm_queen5_5_3.wcsp`, `str_nw_mera_closed_120`, `tensornetwork_permutation_optimized`), lighter tensor network tests (e.g. `tensornetwork_permutation_light_415`), or focused bottleneck testing (e.g. `tensornetwork_permutation_focus_step409_316`).

**With the full script (Rust + Julia):**

```bash
BENCH_INSTANCE=str_nw_mera_closed_120 ./scripts/run_all.sh 1
BENCH_INSTANCE=gm_queen5_5_3.wcsp ./scripts/run_all.sh 4
BENCH_INSTANCE=tensornetwork_permutation_light_415 ./scripts/run_all.sh 1 1
BENCH_INSTANCE=tensornetwork_permutation_focus_step409_316 ./scripts/run_all.sh 1
```

**Rust or Julia alone:**

```bash
BENCH_INSTANCE=str_nw_mera_closed_120 cargo run --release
BENCH_INSTANCE=str_nw_mera_closed_120 julia --project=. src/main.jl
```

- **Instance name** must match the `name` field in the JSON (i.e. the filename without `.json`). To list available names: `ls data/instances/` → e.g. `str_nw_mera_closed_120.json` → use `str_nw_mera_closed_120`.
- Both the Rust and Julia runners respect `BENCH_INSTANCE`; if set, only that instance is loaded and run for every strategy and mode.

### 4. Run individually

**Rust (strided-opteinsum, faer backend):**

```bash
RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 cargo run --release
```

**Rust (strided-opteinsum, blas backend):**

```bash
RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 cargo run --release --no-default-features --features blas
```

**Julia (OMEinsum.jl):**

```bash
OMP_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --project=. src/main.jl
```

Julia benchmark modes:

- **omeinsum_path** — follows the same pre-computed contraction path as Rust (fair kernel-level comparison)
- **omeinsum_opt** — OMEinsum.jl with `optimize_code` and `TreeSA()` (optimizer-chosen path)

### 5. Profiling

**CPU flamegraph** (requires `cargo install flamegraph`):

```bash
BENCH_INSTANCE=tensornetwork_permutation_light_415 RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  cargo flamegraph --profile release-with-debug -o flamegraph.svg --no-default-features --features faer
```

Opens an interactive SVG showing where CPU time is spent. On macOS, uses `xctrace`; on Linux, uses `perf`.

**Internal profiler** (plan/bgemm/buffer stats):

```bash
BENCH_INSTANCE=tensornetwork_permutation_light_415 STRIDED_EINSUM2_PROFILE=1 \
  RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 cargo run --release --no-default-features --features faer
```

Prints strided-einsum2 dispatch stats to stderr for each strategy:

- `total_calls`, `direct_calls`, `packed_calls`
- `m_packed_hist`, `n_packed_hist`, `k_packed_hist` (bucketed packed-path dimensions)

### Row-major to Column-major Conversion

NumPy arrays are row-major (C order). strided-rs uses column-major (Fortran order). The conversion is metadata-only:

- Reverse each tensor's shape: `[M, K]` -> `[K, M]`
- Reverse each operand's index labels: `"ij,jk->ik"` -> `"ji,kj->ki"`

Both the original (`format_string`, `shapes`) and converted (`format_string_colmajor`, `shapes_colmajor`) metadata are stored in each JSON file.

## Reproducing Benchmarks

Run all benchmarks (Rust faer + Rust blas + Julia):

```bash
./scripts/run_all.sh        # 1 thread, canonical ids off
./scripts/run_all.sh 4      # 4 threads, canonical ids off
./scripts/run_all.sh 1 1    # 1 thread, canonical ids on
```

This script:

1. Sets `OMP_NUM_THREADS`, `RAYON_NUM_THREADS`, `JULIA_NUM_THREADS` to the given thread count (default: 1) and `STRIDED_OPTEINSUM_CANONICAL_BINARY_IDS` (default: 0)
2. Builds and runs the Rust benchmark with the **faer** backend
3. Builds and runs the Rust benchmark with the **blas** (OpenBLAS) backend (skipped if OpenBLAS is not installed)
4. Runs the Julia benchmark (`julia --project=. src/main.jl`)
5. Formats results as a markdown table via `scripts/format_results.py`
6. Saves all outputs to `data/results/` with timestamps

To format existing log files into a markdown table:

```bash
uv run python scripts/format_results.py data/results/rust_*.log data/results/julia_*.log
```

## Benchmark Instances

Instances are from the [einsum benchmark](https://benchmark.einsum.org/) suite. Selection is per-category (see [Export benchmark metadata](#1-export-benchmark-metadata)); dtype is float64 or complex128; tensors are zero-filled at runtime.

| Instance | Category | Tensors | Dims | Typical shapes | Steps | log10(FLOPS) | log2(SIZE) |
|----------|----------|--------:|------|----------------|------:|-------------:|------------:|
| `gm_queen5_5_3.wcsp` | Graphical model | 160 | 2D | 3×3 | 159 | 9.75 | 26.94 |
| `lm_batch_likelihood_brackets_4_4d` | Language model | 84 | 2–4D | 4×4, 4×4×4, 7×1996 | 83 | 8.37 | 18.96 |
| `lm_batch_likelihood_sentence_3_12d` | Language model | 38 | 2–4D | 11×11, 11×11×11, 100×1100 | 37 | 9.20 | 20.86 |
| `lm_batch_likelihood_sentence_4_4d` | Language model | 84 | 2–4D | 4×4, 4×4×4, 7×1900 | 83 | 8.46 | 18.89 |
| `str_matrix_chain_multiplication_100` | Structured | 100 | 2D | 21×478 to 511×507 | 99 | 8.48 | 17.26 |
| `str_mps_varying_inner_product_200` | Structured (MPS) | 200 | 2D | varying | 199 | 8.31 | 15.48 |
| `str_nw_mera_closed_120` | Structured (MERA) | 120 | 2D | 3×3, etc. | 119 | 10.66 | 25.02 |
| `str_nw_mera_open_26` | Structured (MERA) | 26 | 2D | 3×3, etc. | 25 | 10.49 | 25.36 |
| `tensornetwork_permutation_light_415` | Tensor network | 415 | 2D | 2×2 (uniform) | 414 | 9.65 | 24.0 |
| `tensornetwork_permutation_focus_step409_316` | Tensor network (focused) | 316 | 2D | 2×2 (uniform) | 315 | 9.65 | 24.0 |

- **Graphical model (gm_*)**: e.g. WCSP / constraint networks; many small 2D factors (e.g. 3×3), full contraction to scalar.
- **Language model (lm_*)**: many small multi-dimensional tensors (3D/4D) with large batch dimensions; many steps with small GEMM kernels.
- **Structured — matrix chain (str_matrix_chain_*)**: large 2D matrices; each step is one large GEMM.
- **Structured — MPS (str_mps_*)**: matrix product state–style networks; varying inner dimensions, many 2D contractions.
- **Structured — MERA (str_nw_mera_*)**: tensor networks from multi-scale entanglement renormalization; many small 3×3-like tensors, heavy contraction.
- **Tensor network (tensornetwork_permutation_light_415)**: lightweight variant (~5 s vs ~40 s); 415 tensors extracted from the full instance via BFS-connected subgraph. Create via `scripts/create_lightweight_instance.py` (see [Optional: Create a lightweight tensor network instance](#optional-create-a-lightweight-tensor-network-instance)).
- **Tensor network focused (tensornetwork_permutation_focus_step409_316)**: focused subtree instance for profiling original late bottleneck steps (408/409). Contains 316 tensors and keeps the high-cost intermediate (`log2_size = 24`) while reducing total contractions.

## Benchmark Results

Environment: Apple Silicon M2. Median ± IQR (ms) of 15 runs (3 warmup). Julia BLAS: OpenBLAS (lbt). Run date: 2026-02-19.

### 1 thread (`OMP_NUM_THREADS=1`, `RAYON_NUM_THREADS=1`, `JULIA_NUM_THREADS=1`)

#### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | strided-rs faer (ms) | strided-rs OpenBLAS (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|
| gm_queen5_5_3.wcsp | **2138.802 ± 9.502** | 2699.958 ± 341.080 | - |
| lm_batch_likelihood_brackets_4_4d | **12.756 ± 0.649** | 13.834 ± 0.538 | 17.715 ± 1.198 |
| lm_batch_likelihood_sentence_3_12d | **41.627 ± 0.573** | 43.898 ± 0.616 | 53.521 ± 10.400 |
| lm_batch_likelihood_sentence_4_4d | **14.229 ± 0.548** | 14.971 ± 0.627 | 18.001 ± 1.146 |
| str_matrix_chain_multiplication_100 | 11.760 ± 0.408 | **9.846 ± 0.175** | 13.065 ± 0.298 |
| str_mps_varying_inner_product_200 | **10.886 ± 0.147** | 11.566 ± 0.300 | 15.921 ± 1.285 |
| str_nw_mera_closed_120 | 1050.475 ± 2.384 | **1027.667 ± 12.146** | 1114.944 ± 2.929 |
| str_nw_mera_open_26 | 686.154 ± 1.929 | **671.864 ± 2.927** | 840.617 ± 26.332 |
| tensornetwork_permutation_focus_step409_316 | 213.163 ± 1.148 | 215.888 ± 1.753 | **175.591 ± 83.700** |
| tensornetwork_permutation_light_415 | 208.457 ± 1.227 | 212.985 ± 2.494 | **165.571 ± 83.322** |

#### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | strided-rs faer (ms) | strided-rs OpenBLAS (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|
| gm_queen5_5_3.wcsp | **872.782 ± 8.945** | 949.736 ± 31.140 | - |
| lm_batch_likelihood_brackets_4_4d | **13.338 ± 0.375** | 15.223 ± 0.233 | 18.159 ± 1.011 |
| lm_batch_likelihood_sentence_3_12d | 43.317 ± 0.822 | **43.288 ± 0.774** | 49.566 ± 8.968 |
| lm_batch_likelihood_sentence_4_4d | **15.036 ± 0.462** | 16.112 ± 0.403 | 18.412 ± 1.326 |
| str_matrix_chain_multiplication_100 | 11.744 ± 0.391 | **9.847 ± 0.211** | 13.259 ± 0.699 |
| str_mps_varying_inner_product_200 | **10.671 ± 0.186** | 11.783 ± 0.297 | 14.897 ± 0.763 |
| str_nw_mera_closed_120 | 1082.755 ± 58.010 | **1071.333 ± 24.872** | 1085.299 ± 24.702 |
| str_nw_mera_open_26 | **704.060 ± 51.919** | 708.362 ± 36.283 | 841.979 ± 2.174 |
| tensornetwork_permutation_focus_step409_316 | **213.782 ± 1.293** | 227.500 ± 21.644 | 245.958 ± 83.558 |
| tensornetwork_permutation_light_415 | 208.217 ± 0.891 | 219.293 ± 11.226 | **180.142 ± 84.338** |

### 4 threads (`OMP_NUM_THREADS=4`, `RAYON_NUM_THREADS=4`, `JULIA_NUM_THREADS=4`)

#### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | strided-rs faer (ms) | strided-rs OpenBLAS (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|
| gm_queen5_5_3.wcsp | 2410.765 ± 28.722 | **2137.887 ± 6.843** | - |
| lm_batch_likelihood_brackets_4_4d | **10.731 ± 0.714** | 11.628 ± 1.376 | 16.370 ± 2.247 |
| lm_batch_likelihood_sentence_3_12d | **21.725 ± 2.314** | 23.682 ± 1.328 | 31.721 ± 4.973 |
| lm_batch_likelihood_sentence_4_4d | **11.177 ± 0.609** | 11.329 ± 0.527 | 17.369 ± 3.254 |
| str_matrix_chain_multiplication_100 | 9.131 ± 0.800 | **7.415 ± 2.705** | 10.231 ± 9.850 |
| str_mps_varying_inner_product_200 | **11.914 ± 0.538** | 12.866 ± 0.363 | 18.351 ± 18.897 |
| str_nw_mera_closed_120 | 380.443 ± 9.651 | **358.055 ± 10.831** | 578.227 ± 120.271 |
| str_nw_mera_open_26 | 233.263 ± 4.257 | **231.085 ± 4.213** | 431.691 ± 41.821 |
| tensornetwork_permutation_focus_step409_316 | **146.746 ± 2.502** | 152.224 ± 2.210 | 151.123 ± 37.191 |
| tensornetwork_permutation_light_415 | **142.118 ± 3.828** | 149.364 ± 16.407 | 172.460 ± 39.705 |

#### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | strided-rs faer (ms) | strided-rs OpenBLAS (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|
| gm_queen5_5_3.wcsp | 656.406 ± 7.247 | **584.510 ± 28.120** | - |
| lm_batch_likelihood_brackets_4_4d | **11.584 ± 0.438** | 13.319 ± 0.627 | 16.954 ± 1.688 |
| lm_batch_likelihood_sentence_3_12d | **21.677 ± 0.929** | 24.041 ± 0.998 | 36.536 ± 17.143 |
| lm_batch_likelihood_sentence_4_4d | **11.784 ± 0.551** | 12.951 ± 0.352 | 16.073 ± 4.154 |
| str_matrix_chain_multiplication_100 | 9.270 ± 0.883 | **7.926 ± 2.786** | 12.439 ± 10.444 |
| str_mps_varying_inner_product_200 | **11.733 ± 0.447** | 12.867 ± 0.380 | 15.806 ± 3.575 |
| str_nw_mera_closed_120 | 375.974 ± 8.865 | **359.085 ± 9.187** | 445.431 ± 61.282 |
| str_nw_mera_open_26 | **234.930 ± 4.626** | 321.400 ± 109.029 | 448.764 ± 43.730 |
| tensornetwork_permutation_focus_step409_316 | 147.409 ± 2.348 | 161.007 ± 6.599 | **113.116 ± 43.287** |
| tensornetwork_permutation_light_415 | 142.301 ± 2.870 | 173.806 ± 29.273 | **118.710 ± 40.308** |

**Notes:**
- `-` in tables indicates the instance was skipped (e.g. strided-opteinsum skips operands with duplicate axis labels). Skipped instances are printed as **SKIP** with the reason on stderr.
- **strided-rs faer** uses [faer](https://github.com/sarah-quinones/faer-rs) (pure Rust GEMM). **strided-rs OpenBLAS** uses OpenBLAS via `cblas-sys`.
- **strided-rs** and **OMEinsum.jl** (omeinsum_path mode) use the same pre-computed contraction path for fair comparison.
- OMEinsum.jl shows high IQR on tensor network instances (e.g. 83ms IQR at 165ms median), suggesting GC or allocation variance. strided-rs results are highly stable (IQR < 4ms).

---

Environment: AMD EPYC 7713P (64-Core, Zen 3). Median ± IQR (ms) of 15 runs (3 warmup). Julia 1.11.2, BLAS: lbt. Run date: 2026-02-19.

### 1 thread (`OMP_NUM_THREADS=1`, `RAYON_NUM_THREADS=1`, `JULIA_NUM_THREADS=1`)

#### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | strided-rs faer (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|
| gm_queen5_5_3.wcsp | **5425.432 ± 30.905** | - |
| lm_batch_likelihood_brackets_4_4d | **17.616 ± 2.519** | 28.692 ± 41.708 |
| lm_batch_likelihood_sentence_3_12d | **65.800 ± 4.669** | 84.372 ± 74.713 |
| lm_batch_likelihood_sentence_4_4d | **17.113 ± 1.231** | 36.441 ± 4.748 |
| str_matrix_chain_multiplication_100 | **10.936 ± 1.814** | 16.916 ± 3.780 |
| str_mps_varying_inner_product_200 | **13.829 ± 1.969** | 29.727 ± 13.144 |
| str_nw_mera_closed_120 | **1379.119 ± 24.361** | 1442.946 ± 29.444 |
| str_nw_mera_open_26 | **904.426 ± 23.936** | 1283.975 ± 76.196 |
| tensornetwork_permutation_focus_step409_316 | **462.691 ± 8.581** | 472.680 ± 75.893 |
| tensornetwork_permutation_light_415 | 460.808 ± 9.420 | **457.909 ± 39.131** |

#### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | strided-rs faer (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|
| gm_queen5_5_3.wcsp | **1641.988 ± 62.481** | - |
| lm_batch_likelihood_brackets_4_4d | **15.334 ± 2.493** | 34.520 ± 3.294 |
| lm_batch_likelihood_sentence_3_12d | **50.472 ± 5.506** | 100.981 ± 70.747 |
| lm_batch_likelihood_sentence_4_4d | **19.962 ± 0.671** | 36.470 ± 5.345 |
| str_matrix_chain_multiplication_100 | **10.963 ± 0.640** | 17.418 ± 1.579 |
| str_mps_varying_inner_product_200 | **12.530 ± 0.821** | 24.484 ± 7.706 |
| str_nw_mera_closed_120 | **1248.746 ± 26.460** | 1255.150 ± 56.252 |
| str_nw_mera_open_26 | **938.363 ± 41.570** | 1286.824 ± 34.062 |
| tensornetwork_permutation_focus_step409_316 | **473.367 ± 4.872** | 476.555 ± 80.028 |
| tensornetwork_permutation_light_415 | 465.709 ± 14.434 | **454.003 ± 70.465** |

### 4 threads (`OMP_NUM_THREADS=4`, `RAYON_NUM_THREADS=4`, `JULIA_NUM_THREADS=4`)

#### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | strided-rs faer (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|
| gm_queen5_5_3.wcsp | **6668.064 ± 128.566** | - |
| lm_batch_likelihood_brackets_4_4d | 35.454 ± 0.864 | **34.250 ± 29.449** |
| lm_batch_likelihood_sentence_3_12d | 79.085 ± 6.006 | **64.289 ± 50.246** |
| lm_batch_likelihood_sentence_4_4d | **30.035 ± 2.846** | 40.612 ± 28.321 |
| str_matrix_chain_multiplication_100 | **10.699 ± 7.630** | 16.358 ± 2.706 |
| str_mps_varying_inner_product_200 | 30.944 ± 6.748 | **30.917 ± 7.867** |
| str_nw_mera_closed_120 | **830.419 ± 16.015** | 884.792 ± 16.590 |
| str_nw_mera_open_26 | **534.922 ± 10.166** | 781.395 ± 32.119 |
| tensornetwork_permutation_focus_step409_316 | 495.643 ± 17.080 | **394.722 ± 20.700** |
| tensornetwork_permutation_light_415 | 494.529 ± 18.628 | **394.431 ± 39.800** |

#### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | strided-rs faer (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|
| gm_queen5_5_3.wcsp | **1673.836 ± 46.261** | - |
| lm_batch_likelihood_brackets_4_4d | **30.993 ± 2.207** | 39.117 ± 3.149 |
| lm_batch_likelihood_sentence_3_12d | **49.556 ± 6.771** | 81.003 ± 45.662 |
| lm_batch_likelihood_sentence_4_4d | **36.452 ± 4.090** | 41.886 ± 27.692 |
| str_matrix_chain_multiplication_100 | **10.103 ± 0.752** | 14.062 ± 1.830 |
| str_mps_varying_inner_product_200 | 29.891 ± 3.109 | **25.688 ± 4.314** |
| str_nw_mera_closed_120 | **635.751 ± 23.938** | 699.768 ± 11.451 |
| str_nw_mera_open_26 | **543.366 ± 14.096** | 753.542 ± 48.265 |
| tensornetwork_permutation_focus_step409_316 | 480.023 ± 17.737 | **394.444 ± 34.031** |
| tensornetwork_permutation_light_415 | 491.230 ± 10.455 | **403.218 ± 25.625** |

**Notes:**
- OpenBLAS is not available on this AMD system; only faer backend was benchmarked for strided-rs.
- `gm_queen5_5_3.wcsp` skipped by Julia 1.11.2 due to a `MethodError` (3D tensor not supported by Matrix constructor in this version).
- strided-rs 4T shows regression on some small-tensor instances (lm_*, str_mps_*) due to threading overhead on tasks too small for parallelism to help.

## References

- [Einsum Benchmark](https://benchmark.einsum.org/) — standardized einsum benchmark suite
- [ti2-group/einsum_benchmark](https://github.com/ti2-group/einsum_benchmark) — Python package
- [tensor4all/strided-rs](https://github.com/tensor4all/strided-rs) — Rust tensor library
- [OMEinsum.jl](https://github.com/under-Peter/OMEinsum.jl) — Julia einsum library
- [TensorNetworkBenchmarks](https://github.com/TensorBFS/TensorNetworkBenchmarks) — tensor network contraction benchmarks (PyTorch, OMEinsum.jl)
