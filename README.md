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

This selects instances matching laptop-scale criteria and saves JSON metadata to `data/instances/`.

**Selection criteria:**

| Filter | Threshold |
|--------|-----------|
| log10[FLOPS] | < 10 |
| log2[SIZE] | < 25 |
| dtype | float64 or complex128 |
| num_tensors | <= 100 |

### 2. Run all benchmarks

```bash
./scripts/run_all.sh        # 1 thread (default)
./scripts/run_all.sh 4      # 4 threads
```

Runs Rust (faer + blas) and Julia benchmarks. The argument sets `OMP_NUM_THREADS`, `RAYON_NUM_THREADS`, and `JULIA_NUM_THREADS`. If OpenBLAS is not installed, the blas benchmark is skipped with a warning. Results are saved to `data/results/`. To run only one instance, see [Run a single instance](#3-run-a-single-instance) below.

Instance JSON files that fail to read or parse are skipped with a warning; the suite continues with the rest. Instances that trigger a backend error (e.g. duplicate axis labels in strided-opteinsum) are reported as **SKIP** in the table with the reason on stderr.

### 3. Run a single instance

To run the benchmark for **one instance only**, set the environment variable `BENCH_INSTANCE` to the instance name. Useful for heavy instances (e.g. `gm_queen5_5_3.wcsp`, `str_nw_mera_closed_120`) or to avoid timeouts.

**With the full script (Rust + Julia):**

```bash
BENCH_INSTANCE=str_nw_mera_closed_120 ./scripts/run_all.sh 1
BENCH_INSTANCE=gm_queen5_5_3.wcsp ./scripts/run_all.sh 4
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

### Row-major to Column-major Conversion

NumPy arrays are row-major (C order). strided-rs uses column-major (Fortran order). The conversion is metadata-only:

- Reverse each tensor's shape: `[M, K]` -> `[K, M]`
- Reverse each operand's index labels: `"ij,jk->ik"` -> `"ji,kj->ki"`

Both the original (`format_string`, `shapes`) and converted (`format_string_colmajor`, `shapes_colmajor`) metadata are stored in each JSON file.

## Reproducing Benchmarks

Run all benchmarks (Rust faer + Rust blas + Julia):

```bash
./scripts/run_all.sh        # 1 thread
./scripts/run_all.sh 4      # 4 threads
```

This script:

1. Sets `OMP_NUM_THREADS`, `RAYON_NUM_THREADS`, `JULIA_NUM_THREADS` to the given thread count (default: 1)
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

All instances are from the [einsum benchmark](https://benchmark.einsum.org/) suite (float64, zero-filled tensors).

| Instance | Category | Tensors | Dims | Typical shapes | Steps | log10(FLOPS) | log2(SIZE) |
|---|---|---:|---|---|---:|---:|---:|
| `lm_batch_likelihood_brackets_4_4d` | Language model | 84 | 2-4D | 4×4, 4×4×4, 7×1996 | 83 | 8.37 | 18.96 |
| `lm_batch_likelihood_sentence_3_12d` | Language model | 38 | 2-4D | 11×11, 11×11×11, 100×1100 | 37 | 9.20 | 20.86 |
| `lm_batch_likelihood_sentence_4_4d` | Language model | 84 | 2-4D | 4×4, 4×4×4, 7×1900 | 83 | 8.46 | 18.89 |
| `str_matrix_chain_multiplication_100` | Structured | 100 | 2D | 21×478 to 511×507 | 99 | 8.48 | 17.26 |
| `str_nw_mera_closed_120` | Structured (MERA) | 120 | 2D | 3×3, etc. | 119 | 10.66 | 25.02 |
| `str_nw_mera_open_26` | Structured (MERA) | 26 | 2D | 3×3, etc. | 25 | 10.49 | 25.36 |

- **Language model** instances: many small multi-dimensional tensors (3D/4D) with a few large batch tensors. Many contraction steps with tiny GEMM kernels.
- **Matrix chain** instance: 100 large 2D matrices. Each contraction step is a single large GEMM.
- **MERA (str_nw_mera_*)** instances: tensor networks from multi-scale entanglement renormalization; many small 3×3 (or similar) tensors, heavy contraction.

## Benchmark Results

Environment: Apple Silicon M2. Median time (ms) of 5 runs (2 warmup). Julia BLAS: OpenBLAS (lbt).

### 1 thread (`OMP_NUM_THREADS=1`, `RAYON_NUM_THREADS=1`, `JULIA_NUM_THREADS=1`)

#### Strategy: opt_flops

Median time (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | strided-rs faer (ms) | strided-rs OpenBLAS (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|
| gm_queen5_5_3.wcsp | 3774.411 | 4316.193 | - |
| lm_batch_likelihood_brackets_4_4d | 17.420 | 18.351 | 19.711 |
| lm_batch_likelihood_sentence_3_12d | 43.569 | 45.240 | 53.387 |
| lm_batch_likelihood_sentence_4_4d | 18.185 | 20.146 | 28.886 |
| str_matrix_chain_multiplication_100 | 11.602 | 10.452 | 17.005 |
| str_mps_varying_inner_product_200 | 16.958 | 18.597 | 17.210 |
| str_nw_mera_closed_120 | 1107.324 | 1069.042 | 1116.350 |
| str_nw_mera_open_26 | 704.454 | 686.719 | 793.797 |

#### Strategy: opt_size

Median time (ms). JULIA_NUM_THREADS=1, OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | strided-rs faer (ms) | strided-rs OpenBLAS (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|
| gm_queen5_5_3.wcsp | 1636.198 | 1708.649 | - |
| lm_batch_likelihood_brackets_4_4d | 19.331 | 21.666 | 18.644 |
| lm_batch_likelihood_sentence_3_12d | 47.218 | 44.857 | 52.214 |
| lm_batch_likelihood_sentence_4_4d | 25.352 | 27.664 | 20.785 |
| str_matrix_chain_multiplication_100 | 12.835 | 10.858 | 15.107 |
| str_mps_varying_inner_product_200 | 17.901 | 19.126 | 15.932 |
| str_nw_mera_closed_120 | 1082.278 | 1041.386 | 1085.563 |
| str_nw_mera_open_26 | 720.243 | 701.569 | 792.436 |

### 4 threads (`OMP_NUM_THREADS=4`, `RAYON_NUM_THREADS=4`, `JULIA_NUM_THREADS=4`)

#### Strategy: opt_flops

Median time (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | strided-rs faer (ms) | strided-rs OpenBLAS (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|
| gm_queen5_5_3.wcsp | 3595.949 | 4199.687 | - |
| lm_batch_likelihood_brackets_4_4d | 14.764 | 41.028 | 28.081 |
| lm_batch_likelihood_sentence_3_12d | 23.178 | 62.389 | 30.902 |
| lm_batch_likelihood_sentence_4_4d | 15.998 | 35.051 | 16.169 |
| str_matrix_chain_multiplication_100 | 10.000 | 21.842 | 13.867 |
| str_mps_varying_inner_product_200 | 18.205 | 37.144 | 15.491 |
| str_nw_mera_closed_120 | 385.426 | 649.763 | 366.599 |
| str_nw_mera_open_26 | 234.414 | 247.319 | 268.863 |

#### Strategy: opt_size

Median time (ms). JULIA_NUM_THREADS=4, OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | strided-rs faer (ms) | strided-rs OpenBLAS (ms) | OMEinsum.jl OpenBLAS (ms) |
|---|---:|---:|---:|
| gm_queen5_5_3.wcsp | 1434.909 | 1423.930 | - |
| lm_batch_likelihood_brackets_4_4d | 17.600 | 20.225 | 16.262 |
| lm_batch_likelihood_sentence_3_12d | 22.618 | 24.571 | 29.823 |
| lm_batch_likelihood_sentence_4_4d | 22.274 | 23.634 | 16.916 |
| str_matrix_chain_multiplication_100 | 8.192 | 8.494 | 12.227 |
| str_mps_varying_inner_product_200 | 18.095 | 21.052 | 17.636 |
| str_nw_mera_closed_120 | 361.526 | 369.733 | 338.994 |
| str_nw_mera_open_26 | 236.600 | 273.848 | 279.396 |


**Notes:**
- `-` in tables indicates the instance was skipped (e.g. strided-opteinsum skips operands with duplicate axis labels). Skipped instances are printed as **SKIP** with the reason on stderr.
- **strided-rs faer** uses [faer](https://github.com/sarah-quinones/faer-rs) (pure Rust GEMM). **strided-rs OpenBLAS** uses OpenBLAS via `cblas-sys`.
- **strided-rs** and **OMEinsum.jl** (omeinsum_path mode) use the same pre-computed contraction path for fair comparison.

## References

- [Einsum Benchmark](https://benchmark.einsum.org/) — standardized einsum benchmark suite
- [ti2-group/einsum_benchmark](https://github.com/ti2-group/einsum_benchmark) — Python package
- [tensor4all/strided-rs](https://github.com/tensor4all/strided-rs) — Rust tensor library
- [OMEinsum.jl](https://github.com/under-Peter/OMEinsum.jl) — Julia einsum library