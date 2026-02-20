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
- **blas** — links to OpenBLAS (>= 0.3.29 recommended)

**OpenBLAS setup:**

- macOS: `brew install openblas`
- Ubuntu: build from source (system `libopenblas-dev` on Ubuntu 20.04 is 0.3.8 — too old for fair benchmarks)

```bash
# Build OpenBLAS 0.3.29 from source (one-time)
curl -L -o /tmp/OpenBLAS-0.3.29.tar.gz \
  https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.29/OpenBLAS-0.3.29.tar.gz
cd /tmp && tar xf OpenBLAS-0.3.29.tar.gz && cd OpenBLAS-0.3.29
make -j$(nproc) USE_OPENMP=0 NO_LAPACK=1
make PREFIX=$HOME/opt/openblas-0.3.29 install

# Set environment for building/running
export OPENBLAS_LIB_DIR=$HOME/opt/openblas-0.3.29/lib
export LD_LIBRARY_PATH=$HOME/opt/openblas-0.3.29/lib:$LD_LIBRARY_PATH
```

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

- [Apple Silicon M2](#apple-silicon-m2)
- [AMD EPYC 7713P](#amd-epyc-7713p)

### Apple Silicon M2

Environment: Apple Silicon M2. Median ± IQR (ms) of 15 runs (3 warmup). Julia BLAS: OpenBLAS (lbt). Run date: 2026-02-20.

#### 1 thread (`OMP_NUM_THREADS=1`, `RAYON_NUM_THREADS=1`, `JULIA_NUM_THREADS=1`)

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | strided-rs faer (ms) | strided-rs OpenBLAS (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|
| gm_queen5_5_3.wcsp | **2250.885 ± 50.871** | 2558.532 ± 37.899 | - |
| lm_batch_likelihood_brackets_4_4d | **13.708 ± 0.538** | 15.609 ± 0.217 | 17.640 ± 1.774 |
| lm_batch_likelihood_sentence_3_12d | **43.202 ± 0.697** | 45.763 ± 0.398 | 51.767 ± 2.982 |
| lm_batch_likelihood_sentence_4_4d | **14.807 ± 0.738** | 15.711 ± 0.446 | 18.634 ± 0.698 |
| str_matrix_chain_multiplication_100 | 11.658 ± 0.192 | **9.747 ± 0.364** | 13.160 ± 0.287 |
| str_mps_varying_inner_product_200 | **12.681 ± 0.226** | 13.392 ± 0.275 | 16.417 ± 1.914 |
| str_nw_mera_closed_120 | 1136.073 ± 3.496 | **1108.740 ± 4.346** | 1129.092 ± 6.852 |
| str_nw_mera_open_26 | 705.168 ± 1.201 | **692.574 ± 2.937** | 847.499 ± 3.184 |
| tensornetwork_permutation_focus_step409_316 | **165.356 ± 1.548** | 172.696 ± 1.162 | 165.546 ± 85.734 |
| tensornetwork_permutation_light_415 | **166.450 ± 1.231** | 174.129 ± 2.272 | 248.907 ± 89.249 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | strided-rs faer (ms) | strided-rs OpenBLAS (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|
| gm_queen5_5_3.wcsp | **841.226 ± 18.452** | 897.783 ± 3.498 | - |
| lm_batch_likelihood_brackets_4_4d | **14.760 ± 0.579** | 18.549 ± 0.294 | 18.209 ± 0.845 |
| lm_batch_likelihood_sentence_3_12d | 48.802 ± 1.312 | **46.992 ± 0.744** | 49.174 ± 13.261 |
| lm_batch_likelihood_sentence_4_4d | **16.276 ± 0.711** | 17.534 ± 0.300 | 18.218 ± 0.937 |
| str_matrix_chain_multiplication_100 | 12.071 ± 0.601 | **9.704 ± 0.158** | 13.336 ± 0.898 |
| str_mps_varying_inner_product_200 | **12.225 ± 0.291** | 13.280 ± 0.242 | 15.718 ± 1.029 |
| str_nw_mera_closed_120 | 1190.591 ± 5.063 | 1154.118 ± 22.288 | **1098.401 ± 49.397** |
| str_nw_mera_open_26 | 716.191 ± 1.987 | **704.318 ± 2.481** | 855.141 ± 5.573 |
| tensornetwork_permutation_focus_step409_316 | **167.244 ± 1.371** | 173.720 ± 2.152 | 248.619 ± 85.563 |
| tensornetwork_permutation_light_415 | **168.680 ± 1.415** | 175.384 ± 1.009 | 251.087 ± 85.844 |

#### 4 threads (`OMP_NUM_THREADS=4`, `RAYON_NUM_THREADS=4`, `JULIA_NUM_THREADS=4`)

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | strided-rs faer (ms) | strided-rs OpenBLAS (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|
| gm_queen5_5_3.wcsp | 2226.390 ± 262.773 | **1795.207 ± 277.982** | - |
| lm_batch_likelihood_brackets_4_4d | 13.742 ± 0.751 | **13.391 ± 0.745** | 16.089 ± 2.112 |
| lm_batch_likelihood_sentence_3_12d | 28.813 ± 4.249 | **24.992 ± 1.696** | 32.007 ± 4.012 |
| lm_batch_likelihood_sentence_4_4d | 13.970 ± 1.789 | **12.617 ± 0.503** | 15.539 ± 2.651 |
| str_matrix_chain_multiplication_100 | 12.969 ± 2.270 | 9.885 ± 4.190 | **8.482 ± 0.797** |
| str_mps_varying_inner_product_200 | 14.872 ± 0.956 | **14.643 ± 0.199** | 15.611 ± 5.016 |
| str_nw_mera_closed_120 | 527.808 ± 80.691 | **421.520 ± 8.264** | 428.417 ± 7.367 |
| str_nw_mera_open_26 | 283.615 ± 40.364 | **268.889 ± 87.239** | 411.360 ± 101.623 |
| tensornetwork_permutation_focus_step409_316 | **69.439 ± 10.994** | 73.977 ± 5.958 | 142.740 ± 43.651 |
| tensornetwork_permutation_light_415 | **69.437 ± 8.207** | 73.813 ± 1.085 | 111.945 ± 41.524 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | strided-rs faer (ms) | strided-rs OpenBLAS (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|
| gm_queen5_5_3.wcsp | 500.195 ± 16.561 | **444.300 ± 47.283** | - |
| lm_batch_likelihood_brackets_4_4d | **13.509 ± 1.127** | 17.356 ± 1.896 | 16.298 ± 1.074 |
| lm_batch_likelihood_sentence_3_12d | **24.599 ± 2.393** | 27.128 ± 2.045 | 38.989 ± 19.676 |
| lm_batch_likelihood_sentence_4_4d | 16.040 ± 1.558 | **15.145 ± 0.726** | 16.340 ± 1.505 |
| str_matrix_chain_multiplication_100 | 10.947 ± 1.928 | **6.966 ± 2.002** | 9.269 ± 3.527 |
| str_mps_varying_inner_product_200 | 15.724 ± 1.752 | 14.345 ± 0.368 | **14.137 ± 0.712** |
| str_nw_mera_closed_120 | 422.126 ± 36.977 | 409.026 ± 31.399 | **398.989 ± 13.093** |
| str_nw_mera_open_26 | 299.660 ± 53.889 | **248.069 ± 3.115** | 380.936 ± 5.200 |
| tensornetwork_permutation_focus_step409_316 | 87.663 ± 19.809 | **73.989 ± 1.122** | 144.334 ± 31.868 |
| tensornetwork_permutation_light_415 | 76.011 ± 17.900 | **75.467 ± 1.945** | 145.619 ± 35.652 |

**Notes:**
- `-` in tables indicates the instance was skipped (e.g. strided-opteinsum skips operands with duplicate axis labels). Skipped instances are printed as **SKIP** with the reason on stderr.
- **strided-rs faer** uses [faer](https://github.com/sarah-quinones/faer-rs) (pure Rust GEMM). **strided-rs OpenBLAS** uses OpenBLAS via `cblas-sys`. Julia uses OpenBLAS via libblastrampoline (lbt).
- **strided-rs** and **OMEinsum.jl** (omeinsum_path mode) use the same pre-computed contraction path for fair comparison.
- At 4T, **OpenBLAS outperforms faer** on most BLAS-heavy instances (gm, lm, str_nw_mera, str_matrix_chain), while **faer leads on tensor network instances** (`tensornetwork_permutation_light_415`: 69ms faer vs 74ms blas vs 112ms Julia).
- Tensor network instances show significant improvement with HPTT-based permutation and source-order copy optimizations in strided-rs (e.g. `tensornetwork_permutation_light_415`: 166ms vs 249ms Julia at 1T, 69ms vs 112ms at 4T).

### AMD EPYC 7713P

Environment: AMD EPYC 7713P (64-Core, Zen 3). Median ± IQR (ms) of 15 runs (3 warmup). Rust OpenBLAS: 0.3.29 (local build). Julia 1.12.5, BLAS: lbt (OpenBLAS 0.3.29). Run date: 2026-02-20.

#### 1 thread (`OMP_NUM_THREADS=1`, `RAYON_NUM_THREADS=1`, `JULIA_NUM_THREADS=1`)

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | strided-rs faer (ms) | strided-rs OpenBLAS (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|
| gm_queen5_5_3.wcsp | 5542.321 ± 26.615 | **5395.628 ± 57.270** | - |
| lm_batch_likelihood_brackets_4_4d | **17.427 ± 0.033** | 23.323 ± 2.287 | 29.871 ± 42.715 |
| lm_batch_likelihood_sentence_3_12d | **71.128 ± 2.650** | 80.187 ± 3.902 | 82.160 ± 77.071 |
| lm_batch_likelihood_sentence_4_4d | **19.763 ± 1.309** | 25.754 ± 3.391 | 35.540 ± 4.936 |
| str_matrix_chain_multiplication_100 | **12.680 ± 1.176** | 13.638 ± 2.429 | 16.436 ± 1.527 |
| str_mps_varying_inner_product_200 | **15.699 ± 1.585** | 17.178 ± 2.093 | 27.053 ± 5.870 |
| str_nw_mera_closed_120 | **1475.915 ± 22.714** | 1507.508 ± 73.751 | 1488.099 ± 40.482 |
| str_nw_mera_open_26 | 917.542 ± 40.738 | **898.958 ± 14.881** | 1284.592 ± 75.551 |
| tensornetwork_permutation_focus_step409_316 | **278.747 ± 7.136** | 307.484 ± 9.203 | 462.286 ± 73.367 |
| tensornetwork_permutation_light_415 | **283.434 ± 6.328** | 305.516 ± 11.052 | 476.160 ± 79.239 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | strided-rs faer (ms) | strided-rs OpenBLAS (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|
| gm_queen5_5_3.wcsp | **1672.395 ± 34.857** | 1707.855 ± 29.879 | - |
| lm_batch_likelihood_brackets_4_4d | **19.388 ± 2.974** | 25.041 ± 1.540 | 31.792 ± 4.406 |
| lm_batch_likelihood_sentence_3_12d | **52.899 ± 4.354** | 61.813 ± 4.960 | 95.622 ± 86.318 |
| lm_batch_likelihood_sentence_4_4d | **21.059 ± 2.453** | 26.218 ± 2.871 | 36.105 ± 4.797 |
| str_matrix_chain_multiplication_100 | **11.602 ± 0.121** | 12.013 ± 1.192 | 16.972 ± 1.076 |
| str_mps_varying_inner_product_200 | **15.023 ± 2.102** | 19.113 ± 2.282 | 24.292 ± 4.368 |
| str_nw_mera_closed_120 | 1519.474 ± 48.670 | 1363.183 ± 32.105 | **1296.149 ± 33.795** |
| str_nw_mera_open_26 | 917.680 ± 26.883 | **902.614 ± 13.524** | 1265.585 ± 57.818 |
| tensornetwork_permutation_focus_step409_316 | **282.867 ± 4.850** | 305.947 ± 8.968 | 471.158 ± 76.847 |
| tensornetwork_permutation_light_415 | **287.021 ± 7.655** | 317.272 ± 15.311 | 473.007 ± 83.423 |

#### 4 threads (`OMP_NUM_THREADS=4`, `RAYON_NUM_THREADS=4`, `JULIA_NUM_THREADS=4`)

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | strided-rs faer (ms) | strided-rs OpenBLAS (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|
| gm_queen5_5_3.wcsp | 5631.734 ± 88.782 | **3909.427 ± 42.022** | - |
| lm_batch_likelihood_brackets_4_4d | **36.329 ± 2.958** | 38.391 ± 3.338 | 36.342 ± 26.072 |
| lm_batch_likelihood_sentence_3_12d | 77.878 ± 6.026 | 83.117 ± 2.895 | **72.126 ± 55.662** |
| lm_batch_likelihood_sentence_4_4d | **32.956 ± 3.651** | 40.358 ± 2.673 | 39.889 ± 4.459 |
| str_matrix_chain_multiplication_100 | **10.325 ± 0.963** | 14.094 ± 1.803 | 15.849 ± 5.181 |
| str_mps_varying_inner_product_200 | 41.822 ± 3.892 | **28.154 ± 1.577** | 30.827 ± 9.599 |
| str_nw_mera_closed_120 | **786.369 ± 36.456** | 1155.440 ± 26.093 | 907.198 ± 11.287 |
| str_nw_mera_open_26 | **509.079 ± 56.843** | 671.856 ± 37.436 | 786.319 ± 54.526 |
| tensornetwork_permutation_focus_step409_316 | **222.743 ± 11.824** | 286.620 ± 6.626 | 388.525 ± 22.756 |
| tensornetwork_permutation_light_415 | **222.260 ± 12.892** | 285.224 ± 7.933 | 388.819 ± 23.382 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | strided-rs faer (ms) | strided-rs OpenBLAS (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|
| gm_queen5_5_3.wcsp | 1350.054 ± 23.290 | **1179.117 ± 31.789** | - |
| lm_batch_likelihood_brackets_4_4d | 37.350 ± 4.178 | **34.962 ± 1.430** | 37.704 ± 2.242 |
| lm_batch_likelihood_sentence_3_12d | **64.967 ± 5.152** | 80.514 ± 1.665 | 77.740 ± 54.195 |
| lm_batch_likelihood_sentence_4_4d | **41.229 ± 8.446** | 41.573 ± 2.322 | 43.323 ± 2.809 |
| str_matrix_chain_multiplication_100 | 10.455 ± 0.236 | **8.874 ± 0.608** | 15.406 ± 2.052 |
| str_mps_varying_inner_product_200 | 46.643 ± 2.623 | 28.784 ± 1.952 | **26.125 ± 1.967** |
| str_nw_mera_closed_120 | 697.184 ± 16.293 | 1073.124 ± 7.464 | **691.290 ± 19.449** |
| str_nw_mera_open_26 | **502.281 ± 33.969** | 687.454 ± 35.558 | 775.115 ± 40.166 |
| tensornetwork_permutation_focus_step409_316 | **220.152 ± 9.612** | 287.258 ± 5.717 | 384.707 ± 35.756 |
| tensornetwork_permutation_light_415 | **226.255 ± 6.634** | 290.641 ± 9.655 | 392.802 ± 32.123 |

**Notes:**
- **strided-rs OpenBLAS** uses OpenBLAS 0.3.29 (locally built, see [Setup](#rust)). **strided-rs faer** uses [faer](https://github.com/sarah-quinones/faer-rs) (pure Rust GEMM).
- **OMEinsum.jl** uses OpenBLAS 0.3.29 bundled with Julia 1.12.5 via libblastrampoline (lbt).
- `gm_queen5_5_3.wcsp` skipped by Julia due to a `MethodError` (3D tensor not supported by Matrix constructor).
- At 4T, **faer outperforms blas** on most instances. strided-kernel parallel (rayon) provides large speedups for copy/permutation operations, especially on tensor network instances (e.g. `tensornetwork_permutation_light_415`: faer 222ms, blas 285ms vs Julia 388ms).
- strided-rs 4T shows regression on some small-tensor instances (lm_*, str_mps_*) due to threading overhead on tasks too small for parallelism to help.

## References

- [Einsum Benchmark](https://benchmark.einsum.org/) — standardized einsum benchmark suite
- [ti2-group/einsum_benchmark](https://github.com/ti2-group/einsum_benchmark) — Python package
- [tensor4all/strided-rs](https://github.com/tensor4all/strided-rs) — Rust tensor library
- [OMEinsum.jl](https://github.com/under-Peter/OMEinsum.jl) — Julia einsum library
- [TensorNetworkBenchmarks](https://github.com/TensorBFS/TensorNetworkBenchmarks) — tensor network contraction benchmarks (PyTorch, OMEinsum.jl)
