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

Environment: Apple Silicon M2. Median ± IQR (ms) of 15 runs (3 warmup). Julia BLAS: OpenBLAS (lbt). Run date: 2026-02-19.

### 1 thread (`OMP_NUM_THREADS=1`, `RAYON_NUM_THREADS=1`, `JULIA_NUM_THREADS=1`)

#### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | strided-rs faer (ms) | strided-rs OpenBLAS (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|
| gm_queen5_5_3.wcsp | **2293.947 ± 118.922** | 2710.654 ± 249.186 | - |
| lm_batch_likelihood_brackets_4_4d | **14.297 ± 0.475** | 15.316 ± 0.304 | 18.187 ± 2.978 |
| lm_batch_likelihood_sentence_3_12d | **44.128 ± 0.375** | 45.104 ± 0.433 | 53.252 ± 30.763 |
| lm_batch_likelihood_sentence_4_4d | **15.191 ± 1.104** | 15.424 ± 0.130 | 18.570 ± 0.911 |
| str_matrix_chain_multiplication_100 | 11.695 ± 0.454 | **9.654 ± 0.104** | 13.155 ± 0.303 |
| str_mps_varying_inner_product_200 | **12.740 ± 0.156** | 13.134 ± 0.087 | 16.244 ± 1.084 |
| str_nw_mera_closed_120 | 1165.061 ± 51.209 | 1189.823 ± 100.642 | **1134.559 ± 37.823** |
| str_nw_mera_open_26 | **715.934 ± 20.386** | 713.277 ± 34.098 | 839.955 ± 21.235 |
| tensornetwork_permutation_focus_step409_316 | **169.680 ± 3.082** | 189.348 ± 34.369 | 246.726 ± 81.376 |
| tensornetwork_permutation_light_415 | **172.845 ± 3.484** | 179.838 ± 5.402 | 248.408 ± 84.455 |

#### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | strided-rs faer (ms) | strided-rs OpenBLAS (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|
| gm_queen5_5_3.wcsp | **875.227 ± 56.962** | 947.915 ± 95.993 | - |
| lm_batch_likelihood_brackets_4_4d | **15.017 ± 0.636** | 19.166 ± 4.556 | 19.099 ± 3.499 |
| lm_batch_likelihood_sentence_3_12d | 50.582 ± 3.272 | **48.951 ± 1.142** | 49.299 ± 1.835 |
| lm_batch_likelihood_sentence_4_4d | **17.221 ± 0.677** | 17.706 ± 0.345 | 18.815 ± 1.489 |
| str_matrix_chain_multiplication_100 | 11.981 ± 1.096 | **10.209 ± 0.793** | 13.384 ± 0.440 |
| str_mps_varying_inner_product_200 | **12.200 ± 0.148** | 13.693 ± 0.265 | 14.819 ± 1.054 |
| str_nw_mera_closed_120 | 1192.865 ± 30.828 | 1175.004 ± 47.950 | **1088.357 ± 23.789** |
| str_nw_mera_open_26 | 726.016 ± 24.137 | **703.123 ± 4.997** | 861.167 ± 59.008 |
| tensornetwork_permutation_focus_step409_316 | **172.317 ± 3.458** | 179.844 ± 15.151 | 247.999 ± 88.427 |
| tensornetwork_permutation_light_415 | **171.573 ± 3.874** | 174.626 ± 1.372 | 248.293 ± 84.110 |

### 4 threads (`OMP_NUM_THREADS=4`, `RAYON_NUM_THREADS=4`, `JULIA_NUM_THREADS=4`)

#### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | strided-rs faer (ms) | strided-rs OpenBLAS (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|
| gm_queen5_5_3.wcsp | 2547.236 ± 70.968 | **2245.540 ± 59.151** | - |
| lm_batch_likelihood_brackets_4_4d | **13.311 ± 0.966** | 13.427 ± 0.501 | 16.711 ± 3.942 |
| lm_batch_likelihood_sentence_3_12d | **23.310 ± 1.309** | 26.498 ± 2.440 | 32.211 ± 3.479 |
| lm_batch_likelihood_sentence_4_4d | **11.781 ± 0.632** | 12.825 ± 0.713 | 15.726 ± 1.178 |
| str_matrix_chain_multiplication_100 | 8.934 ± 1.012 | 11.169 ± 7.579 | **8.554 ± 2.127** |
| str_mps_varying_inner_product_200 | **13.984 ± 0.865** | 14.680 ± 0.857 | 16.431 ± 5.513 |
| str_nw_mera_closed_120 | 476.490 ± 16.763 | **446.637 ± 10.185** | 482.418 ± 59.053 |
| str_nw_mera_open_26 | 260.214 ± 10.304 | **253.835 ± 2.248** | 423.446 ± 59.976 |
| tensornetwork_permutation_focus_step409_316 | **104.277 ± 2.075** | 110.928 ± 6.319 | 140.147 ± 43.168 |
| tensornetwork_permutation_light_415 | **107.595 ± 4.729** | 110.015 ± 1.184 | 142.755 ± 48.206 |

#### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | strided-rs faer (ms) | strided-rs OpenBLAS (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|
| gm_queen5_5_3.wcsp | 622.023 ± 65.657 | **539.577 ± 6.852** | - |
| lm_batch_likelihood_brackets_4_4d | **12.893 ± 0.535** | 16.440 ± 0.330 | 21.712 ± 5.793 |
| lm_batch_likelihood_sentence_3_12d | **26.398 ± 1.545** | 28.628 ± 1.055 | 35.932 ± 28.054 |
| lm_batch_likelihood_sentence_4_4d | **13.522 ± 0.299** | 14.852 ± 3.186 | 31.672 ± 20.763 |
| str_matrix_chain_multiplication_100 | 8.730 ± 0.514 | **8.288 ± 1.853** | 18.472 ± 7.088 |
| str_mps_varying_inner_product_200 | **13.160 ± 0.456** | 14.354 ± 0.231 | 19.165 ± 6.920 |
| str_nw_mera_closed_120 | 492.347 ± 4.677 | 484.296 ± 4.207 | **416.292 ± 62.340** |
| str_nw_mera_open_26 | 263.213 ± 3.290 | **259.488 ± 5.107** | 420.373 ± 42.759 |
| tensornetwork_permutation_focus_step409_316 | **103.292 ± 3.148** | 110.588 ± 2.507 | 129.011 ± 47.470 |
| tensornetwork_permutation_light_415 | **105.283 ± 2.466** | 111.308 ± 0.964 | 130.417 ± 42.376 |

**Notes:**
- `-` in tables indicates the instance was skipped (e.g. strided-opteinsum skips operands with duplicate axis labels). Skipped instances are printed as **SKIP** with the reason on stderr.
- **strided-rs faer** uses [faer](https://github.com/sarah-quinones/faer-rs) (pure Rust GEMM). **strided-rs OpenBLAS** uses OpenBLAS via `cblas-sys`. Julia uses OpenBLAS via libblastrampoline (lbt).
- **strided-rs** and **OMEinsum.jl** (omeinsum_path mode) use the same pre-computed contraction path for fair comparison.
- Tensor network instances show significant improvement with HPTT-based permutation and source-order copy optimizations in strided-rs (e.g. `tensornetwork_permutation_light_415`: 172ms vs 248ms Julia at 1T, 107ms vs 142ms at 4T).

---

Environment: AMD EPYC 7713P (64-Core, Zen 3). Median ± IQR (ms) of 15 runs (3 warmup). Rust OpenBLAS: 0.3.29 (local build). Julia 1.12.5, BLAS: lbt (OpenBLAS 0.3.29). Run date: 2026-02-20.

### 1 thread (`OMP_NUM_THREADS=1`, `RAYON_NUM_THREADS=1`, `JULIA_NUM_THREADS=1`)

#### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | strided-rs faer (ms) | strided-rs OpenBLAS (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|
| gm_queen5_5_3.wcsp | **5503.558 ± 20.638** | 5280.021 ± 42.924 | - |
| lm_batch_likelihood_brackets_4_4d | **17.824 ± 2.794** | 20.688 ± 2.891 | 29.871 ± 42.715 |
| lm_batch_likelihood_sentence_3_12d | **64.648 ± 3.554** | 78.422 ± 5.866 | 82.160 ± 77.071 |
| lm_batch_likelihood_sentence_4_4d | **18.268 ± 2.847** | 23.003 ± 2.610 | 35.540 ± 4.936 |
| str_matrix_chain_multiplication_100 | **10.972 ± 1.333** | 13.635 ± 1.236 | 16.436 ± 1.527 |
| str_mps_varying_inner_product_200 | **16.141 ± 0.458** | 20.704 ± 2.840 | 27.053 ± 5.870 |
| str_nw_mera_closed_120 | **1484.530 ± 7.763** | 1595.281 ± 124.767 | 1488.099 ± 40.482 |
| str_nw_mera_open_26 | **919.722 ± 7.499** | 929.284 ± 26.252 | 1284.592 ± 75.551 |
| tensornetwork_permutation_focus_step409_316 | **283.846 ± 4.546** | 313.720 ± 11.795 | 462.286 ± 73.367 |
| tensornetwork_permutation_light_415 | **287.091 ± 5.881** | 308.327 ± 6.466 | 476.160 ± 79.239 |

#### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | strided-rs faer (ms) | strided-rs OpenBLAS (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|
| gm_queen5_5_3.wcsp | **1680.353 ± 42.578** | 1719.710 ± 26.138 | - |
| lm_batch_likelihood_brackets_4_4d | **19.653 ± 1.254** | 26.240 ± 3.085 | 31.792 ± 4.406 |
| lm_batch_likelihood_sentence_3_12d | **54.186 ± 4.939** | 63.836 ± 6.380 | 95.622 ± 86.318 |
| lm_batch_likelihood_sentence_4_4d | **19.803 ± 2.432** | 26.410 ± 1.647 | 36.105 ± 4.797 |
| str_matrix_chain_multiplication_100 | **11.698 ± 1.634** | 15.136 ± 2.505 | 16.972 ± 1.076 |
| str_mps_varying_inner_product_200 | **14.423 ± 1.211** | 17.211 ± 1.840 | 24.292 ± 4.368 |
| str_nw_mera_closed_120 | 1434.332 ± 9.871 | 1376.634 ± 7.266 | **1296.149 ± 33.795** |
| str_nw_mera_open_26 | 951.040 ± 43.429 | **933.301 ± 22.067** | 1265.585 ± 57.818 |
| tensornetwork_permutation_focus_step409_316 | **285.206 ± 10.155** | 307.148 ± 10.904 | 471.158 ± 76.847 |
| tensornetwork_permutation_light_415 | **287.297 ± 15.436** | 308.467 ± 4.736 | 473.007 ± 83.423 |

### 4 threads (`OMP_NUM_THREADS=4`, `RAYON_NUM_THREADS=4`, `JULIA_NUM_THREADS=4`)

#### Strategy: opt_flops

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | strided-rs faer (ms) | strided-rs OpenBLAS (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|
| gm_queen5_5_3.wcsp | 6916.885 ± 76.474 | **4486.677 ± 44.589** | - |
| lm_batch_likelihood_brackets_4_4d | 38.474 ± 2.334 | **25.065 ± 1.084** | 36.342 ± 26.072 |
| lm_batch_likelihood_sentence_3_12d | 82.203 ± 4.340 | **54.598 ± 2.833** | 72.126 ± 55.662 |
| lm_batch_likelihood_sentence_4_4d | 36.483 ± 2.086 | **23.534 ± 2.051** | 39.889 ± 4.459 |
| str_matrix_chain_multiplication_100 | 17.840 ± 8.264 | **12.587 ± 4.720** | 15.849 ± 5.181 |
| str_mps_varying_inner_product_200 | 48.043 ± 4.244 | **25.300 ± 2.266** | 30.827 ± 9.599 |
| str_nw_mera_closed_120 | 970.535 ± 6.928 | **718.695 ± 17.932** | 907.198 ± 11.287 |
| str_nw_mera_open_26 | 582.737 ± 8.017 | **383.457 ± 11.890** | 786.319 ± 54.526 |
| tensornetwork_permutation_focus_step409_316 | 315.117 ± 8.064 | **208.600 ± 9.091** | 388.525 ± 22.756 |
| tensornetwork_permutation_light_415 | 319.071 ± 8.845 | **210.193 ± 14.301** | 388.819 ± 23.382 |

#### Strategy: opt_size

Median ± IQR (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | strided-rs faer (ms) | strided-rs OpenBLAS (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|
| gm_queen5_5_3.wcsp | 1758.417 ± 17.337 | **1277.747 ± 13.272** | - |
| lm_batch_likelihood_brackets_4_4d | 35.371 ± 4.727 | **27.585 ± 2.707** | 37.704 ± 2.242 |
| lm_batch_likelihood_sentence_3_12d | 63.174 ± 4.467 | **48.172 ± 2.130** | 77.740 ± 54.195 |
| lm_batch_likelihood_sentence_4_4d | 39.862 ± 5.123 | **34.395 ± 2.653** | 43.323 ± 2.809 |
| str_matrix_chain_multiplication_100 | 10.460 ± 0.928 | **8.797 ± 0.103** | 15.406 ± 2.052 |
| str_mps_varying_inner_product_200 | 48.528 ± 5.710 | 27.963 ± 2.054 | **26.125 ± 1.967** |
| str_nw_mera_closed_120 | 856.386 ± 12.757 | **661.369 ± 14.310** | 691.290 ± 19.449 |
| str_nw_mera_open_26 | 590.162 ± 12.765 | **389.905 ± 12.423** | 775.115 ± 40.166 |
| tensornetwork_permutation_focus_step409_316 | 306.405 ± 7.328 | **198.984 ± 5.297** | 384.707 ± 35.756 |
| tensornetwork_permutation_light_415 | 308.294 ± 8.652 | **203.769 ± 4.935** | 392.802 ± 32.123 |

**Notes:**
- **strided-rs OpenBLAS** uses OpenBLAS 0.3.29 (locally built, see [Setup](#rust)). At 4T, OpenBLAS 0.3.29 outperforms both faer and OMEinsum.jl on most instances.
- **OMEinsum.jl** uses OpenBLAS 0.3.29 bundled with Julia 1.12.5 via libblastrampoline (lbt).
- `gm_queen5_5_3.wcsp` skipped by Julia due to a `MethodError` (3D tensor not supported by Matrix constructor).
- strided-rs faer 4T shows regression on some small-tensor instances (lm_*, str_mps_*) due to threading overhead on tasks too small for parallelism to help.

## References

- [Einsum Benchmark](https://benchmark.einsum.org/) — standardized einsum benchmark suite
- [ti2-group/einsum_benchmark](https://github.com/ti2-group/einsum_benchmark) — Python package
- [tensor4all/strided-rs](https://github.com/tensor4all/strided-rs) — Rust tensor library
- [OMEinsum.jl](https://github.com/under-Peter/OMEinsum.jl) — Julia einsum library
- [TensorNetworkBenchmarks](https://github.com/TensorBFS/TensorNetworkBenchmarks) — tensor network contraction benchmarks (PyTorch, OMEinsum.jl)
