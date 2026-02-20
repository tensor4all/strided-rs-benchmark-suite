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

This requires the [TensorNetworkBenchmarks](https://github.com/TensorBFS/TensorNetworkBenchmarks) repository cloned as a sibling directory:

```bash
cd .. && git clone https://github.com/TensorBFS/TensorNetworkBenchmarks.git && cd -
```

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

#### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | strided-rs faer (ms) | strided-rs OpenBLAS (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|
| gm_queen5_5_3.wcsp | **2054.330 ± 58.939** | 2429.877 ± 66.449 | - |
| lm_batch_likelihood_brackets_4_4d | **11.597 ± 0.096** | 13.082 ± 0.012 | 16.325 ± 40.165 |
| lm_batch_likelihood_sentence_3_12d | **40.257 ± 0.995** | 41.400 ± 1.965 | 55.301 ± 69.093 |
| lm_batch_likelihood_sentence_4_4d | **12.936 ± 0.112** | 14.012 ± 0.028 | 17.239 ± 5.973 |
| str_matrix_chain_multiplication_100 | 10.183 ± 0.165 | **9.338 ± 0.180** | 12.425 ± 0.369 |
| str_mps_varying_inner_product_200 | **10.580 ± 0.059** | 12.005 ± 0.336 | 15.732 ± 11.607 |
| str_nw_mera_closed_120 | 1029.458 ± 13.583 | **994.700 ± 11.886** | 1169.895 ± 68.600 |
| str_nw_mera_open_26 | 668.449 ± 7.164 | **658.564 ± 2.241** | 905.097 ± 40.212 |
| tensornetwork_permutation_focus_step409_316 | **204.914 ± 1.585** | 208.848 ± 1.175 | 242.716 ± 10.592 |
| tensornetwork_permutation_light_415 | **203.023 ± 1.199** | 208.206 ± 3.764 | 242.415 ± 3.871 |

#### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | strided-rs faer (ms) | strided-rs OpenBLAS (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|
| gm_queen5_5_3.wcsp | **802.483 ± 3.670** | 888.207 ± 9.465 | - |
| lm_batch_likelihood_brackets_4_4d | **12.671 ± 0.064** | 14.821 ± 0.163 | 16.430 ± 10.745 |
| lm_batch_likelihood_sentence_3_12d | 41.237 ± 0.392 | **40.456 ± 0.982** | 55.733 ± 74.042 |
| lm_batch_likelihood_sentence_4_4d | **13.916 ± 0.052** | 15.943 ± 0.199 | 16.873 ± 5.622 |
| str_matrix_chain_multiplication_100 | 10.376 ± 0.568 | **9.394 ± 0.161** | 12.317 ± 0.562 |
| str_mps_varying_inner_product_200 | **10.177 ± 0.166** | 11.824 ± 0.370 | 14.082 ± 0.875 |
| str_nw_mera_closed_120 | 1037.025 ± 12.973 | **1007.038 ± 15.957** | 1107.183 ± 69.989 |
| str_nw_mera_open_26 | 668.463 ± 3.169 | **656.337 ± 13.777** | 913.331 ± 26.657 |
| tensornetwork_permutation_focus_step409_316 | **205.090 ± 1.225** | 209.114 ± 1.121 | 246.776 ± 8.711 |
| tensornetwork_permutation_light_415 | **202.232 ± 2.898** | 209.465 ± 4.182 | 245.172 ± 1.571 |

#### 4 threads (`OMP_NUM_THREADS=4`, `RAYON_NUM_THREADS=4`, `JULIA_NUM_THREADS=4`)

#### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | strided-rs faer (ms) | strided-rs OpenBLAS (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|
| gm_queen5_5_3.wcsp | 2142.267 ± 21.018 | **1937.117 ± 7.069** | - |
| lm_batch_likelihood_brackets_4_4d | **10.287 ± 0.456** | 11.791 ± 0.187 | 14.973 ± 23.651 |
| lm_batch_likelihood_sentence_3_12d | **18.962 ± 0.489** | 21.107 ± 0.430 | 53.829 ± 28.348 |
| lm_batch_likelihood_sentence_4_4d | **10.554 ± 1.242** | 11.852 ± 0.165 | 35.352 ± 29.476 |
| str_matrix_chain_multiplication_100 | 8.021 ± 0.228 | **7.796 ± 0.119** | 26.935 ± 18.299 |
| str_mps_varying_inner_product_200 | **11.756 ± 0.273** | 15.154 ± 0.231 | 16.429 ± 12.281 |
| str_nw_mera_closed_120 | 310.146 ± 4.654 | **302.529 ± 3.577** | 402.899 ± 27.089 |
| str_nw_mera_open_26 | 193.329 ± 2.267 | **192.600 ± 1.033** | 352.312 ± 22.305 |
| tensornetwork_permutation_focus_step409_316 | 137.439 ± 2.193 | 141.690 ± 1.012 | **131.448 ± 19.086** |
| tensornetwork_permutation_light_415 | 135.437 ± 1.367 | 140.215 ± 1.627 | **119.559 ± 44.121** |

#### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | strided-rs faer (ms) | strided-rs OpenBLAS (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|
| gm_queen5_5_3.wcsp | 581.936 ± 10.790 | **524.492 ± 29.697** | - |
| lm_batch_likelihood_brackets_4_4d | **11.302 ± 0.267** | 12.708 ± 0.394 | 14.751 ± 13.813 |
| lm_batch_likelihood_sentence_3_12d | **19.485 ± 0.335** | 20.652 ± 0.362 | 46.534 ± 24.080 |
| lm_batch_likelihood_sentence_4_4d | **11.836 ± 0.308** | 13.375 ± 0.263 | 13.368 ± 10.760 |
| str_matrix_chain_multiplication_100 | **7.353 ± 0.370** | 7.805 ± 0.131 | 8.419 ± 0.239 |
| str_mps_varying_inner_product_200 | **11.328 ± 0.092** | 15.592 ± 0.299 | 13.365 ± 11.631 |
| str_nw_mera_closed_120 | 320.486 ± 2.468 | **315.732 ± 6.818** | 344.773 ± 23.264 |
| str_nw_mera_open_26 | **193.917 ± 1.573** | 194.489 ± 2.752 | 367.635 ± 9.190 |
| tensornetwork_permutation_focus_step409_316 | 136.560 ± 1.751 | 142.500 ± 2.283 | **117.932 ± 2.795** |
| tensornetwork_permutation_light_415 | 135.696 ± 1.032 | 141.111 ± 0.816 | **120.311 ± 2.997** |

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
| gm_queen5_5_3.wcsp | 5537.697 ± 45.135 | **5324.688 ± 135.417** | - |
| lm_batch_likelihood_brackets_4_4d | **17.368 ± 2.215** | 18.763 ± 1.707 | 29.871 ± 42.715 |
| lm_batch_likelihood_sentence_3_12d | **65.352 ± 4.420** | 74.242 ± 5.115 | 82.160 ± 77.071 |
| lm_batch_likelihood_sentence_4_4d | **16.324 ± 1.464** | 22.961 ± 2.740 | 35.540 ± 4.936 |
| str_matrix_chain_multiplication_100 | **11.516 ± 1.161** | 12.807 ± 1.214 | 16.436 ± 1.527 |
| str_mps_varying_inner_product_200 | **11.975 ± 1.813** | 14.818 ± 1.250 | 27.053 ± 5.870 |
| str_nw_mera_closed_120 | 1395.985 ± 30.376 | **1370.740 ± 12.609** | 1488.099 ± 40.482 |
| str_nw_mera_open_26 | 909.086 ± 14.858 | **891.900 ± 12.933** | 1284.592 ± 75.551 |
| tensornetwork_permutation_focus_step409_316 | **433.995 ± 7.424** | 456.124 ± 8.158 | 462.286 ± 73.367 |
| tensornetwork_permutation_light_415 | **434.751 ± 4.536** | 457.964 ± 4.190 | 476.160 ± 79.239 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | strided-rs faer (ms) | strided-rs OpenBLAS (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|
| gm_queen5_5_3.wcsp | **1681.508 ± 14.145** | 1671.100 ± 10.409 | - |
| lm_batch_likelihood_brackets_4_4d | **15.323 ± 1.682** | 19.528 ± 0.927 | 31.792 ± 4.406 |
| lm_batch_likelihood_sentence_3_12d | **47.618 ± 3.064** | 58.805 ± 4.733 | 95.622 ± 86.318 |
| lm_batch_likelihood_sentence_4_4d | **19.376 ± 2.779** | 25.622 ± 3.387 | 36.105 ± 4.797 |
| str_matrix_chain_multiplication_100 | **11.890 ± 0.867** | 12.155 ± 1.397 | 16.972 ± 1.076 |
| str_mps_varying_inner_product_200 | **11.761 ± 0.693** | 15.552 ± 0.201 | 24.292 ± 4.368 |
| str_nw_mera_closed_120 | 1237.077 ± 12.789 | **1172.212 ± 22.763** | 1296.149 ± 33.795 |
| str_nw_mera_open_26 | 916.477 ± 11.933 | **893.765 ± 13.125** | 1265.585 ± 57.818 |
| tensornetwork_permutation_focus_step409_316 | **449.992 ± 8.882** | 460.593 ± 8.330 | 471.158 ± 76.847 |
| tensornetwork_permutation_light_415 | **451.441 ± 13.424** | 452.876 ± 9.170 | 473.007 ± 83.423 |

#### 4 threads (`OMP_NUM_THREADS=4`, `RAYON_NUM_THREADS=4`, `JULIA_NUM_THREADS=4`)

##### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | strided-rs faer (ms) | strided-rs OpenBLAS (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|
| gm_queen5_5_3.wcsp | 6491.460 ± 111.296 | **4114.639 ± 57.824** | - |
| lm_batch_likelihood_brackets_4_4d | 35.003 ± 5.908 | **21.033 ± 1.596** | 36.342 ± 26.072 |
| lm_batch_likelihood_sentence_3_12d | 79.813 ± 6.737 | **52.578 ± 2.579** | 72.126 ± 55.662 |
| lm_batch_likelihood_sentence_4_4d | 35.670 ± 1.568 | **23.602 ± 2.009** | 39.889 ± 4.459 |
| str_matrix_chain_multiplication_100 | 18.607 ± 1.961 | **14.386 ± 0.736** | 15.849 ± 5.181 |
| str_mps_varying_inner_product_200 | 43.818 ± 1.874 | **26.867 ± 2.413** | 30.827 ± 9.599 |
| str_nw_mera_closed_120 | 856.928 ± 6.909 | **620.274 ± 19.162** | 907.198 ± 11.287 |
| str_nw_mera_open_26 | 554.782 ± 2.985 | **357.109 ± 10.224** | 786.319 ± 54.526 |
| tensornetwork_permutation_focus_step409_316 | 483.477 ± 21.691 | **355.615 ± 14.221** | 388.525 ± 22.756 |
| tensornetwork_permutation_light_415 | 477.138 ± 21.357 | **355.189 ± 11.617** | 388.819 ± 23.382 |

##### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | strided-rs faer (ms) | strided-rs OpenBLAS (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|
| gm_queen5_5_3.wcsp | 1797.177 ± 75.937 | **1237.864 ± 20.780** | - |
| lm_batch_likelihood_brackets_4_4d | 33.226 ± 2.685 | **22.060 ± 2.718** | 37.704 ± 2.242 |
| lm_batch_likelihood_sentence_3_12d | 55.566 ± 3.212 | **37.245 ± 1.808** | 77.740 ± 54.195 |
| lm_batch_likelihood_sentence_4_4d | 40.370 ± 3.605 | **29.752 ± 1.678** | 43.323 ± 2.809 |
| str_matrix_chain_multiplication_100 | 18.155 ± 1.929 | **13.009 ± 1.498** | 15.406 ± 2.052 |
| str_mps_varying_inner_product_200 | 28.319 ± 3.117 | **20.078 ± 2.219** | 26.125 ± 1.967 |
| str_nw_mera_closed_120 | 653.923 ± 11.513 | **464.277 ± 13.885** | 691.290 ± 19.449 |
| str_nw_mera_open_26 | 556.268 ± 9.222 | **360.748 ± 8.509** | 775.115 ± 40.166 |
| tensornetwork_permutation_focus_step409_316 | 469.104 ± 20.256 | **352.028 ± 10.472** | 384.707 ± 35.756 |
| tensornetwork_permutation_light_415 | 473.424 ± 12.774 | **353.143 ± 11.077** | 392.802 ± 32.123 |

**Notes:**
- **strided-rs OpenBLAS** uses OpenBLAS 0.3.29 (locally built, see [Setup](#rust)). **strided-rs faer** uses [faer](https://github.com/sarah-quinones/faer-rs) (pure Rust GEMM).
- **OMEinsum.jl** uses OpenBLAS 0.3.29 bundled with Julia 1.12.5 via libblastrampoline (lbt).
- `gm_queen5_5_3.wcsp` skipped by Julia due to a `MethodError` (3D tensor not supported by Matrix constructor).
- At 1T, **faer outperforms blas** on most instances due to lower per-call overhead for small GEMMs, while **blas leads on large MERA instances** where GEMM size is larger.
- At 4T, **blas (OpenBLAS 0.3.29) outperforms both faer and OMEinsum.jl** on all instances. OpenBLAS internal threading (OMP_NUM_THREADS=4) provides significant speedups, especially on MERA instances (e.g. `str_nw_mera_open_26`: blas 357ms vs faer 555ms vs Julia 786ms).
- faer 4T shows regression on `gm_queen5_5_3.wcsp` (6491ms vs 5538ms at 1T) due to threading overhead on many small GEMMs.

## References

- [Einsum Benchmark](https://benchmark.einsum.org/) — standardized einsum benchmark suite
- [ti2-group/einsum_benchmark](https://github.com/ti2-group/einsum_benchmark) — Python package
- [tensor4all/strided-rs](https://github.com/tensor4all/strided-rs) — Rust tensor library
- [OMEinsum.jl](https://github.com/under-Peter/OMEinsum.jl) — Julia einsum library
- [TensorNetworkBenchmarks](https://github.com/TensorBFS/TensorNetworkBenchmarks) — tensor network contraction benchmarks (PyTorch, OMEinsum.jl)
