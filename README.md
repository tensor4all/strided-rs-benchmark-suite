# strided-rs-benchmark-suite

Benchmark suite for [strided-rs](https://github.com/tensor4all/strided-rs) based on the [einsum benchmark](https://benchmark.einsum.org/) (168 standardized einsum problems across 7 categories).

## Overview

This repository provides:

- A Python pipeline to extract einsum benchmark metadata (shapes, dtypes, contraction paths) into portable JSON
- A **Rust** benchmark runner using [strided-opteinsum](https://github.com/tensor4all/strided-rs)
- A **Julia** benchmark runner using [OMEinsum.jl](https://github.com/under-Peter/OMEinsum.jl) and [TensorOperations.jl](https://github.com/Jutho/TensorOperations.jl)

Only metadata is stored — tensors are generated at benchmark time (zero-filled), keeping the repo lightweight.

See [tensor4all/strided-rs#63](https://github.com/tensor4all/strided-rs/issues/63) for the full design discussion.

## Project Structure

```
strided-rs-benchmark-suite/
  src/
    main.rs                 # Rust benchmark runner (strided-opteinsum)
    main.jl                 # Julia benchmark runner (OMEinsum.jl + TensorOperations.jl)
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

Runs Rust (faer + blas) and Julia benchmarks. The argument sets `OMP_NUM_THREADS`, `RAYON_NUM_THREADS`, and `JULIA_NUM_THREADS`. If OpenBLAS is not installed, the blas benchmark is skipped with a warning. Results are saved to `data/results/`.

### 3. Run individually

**Rust (strided-opteinsum, faer backend):**

```bash
RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 cargo run --release
```

**Rust (strided-opteinsum, blas backend):**

```bash
RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 cargo run --release --no-default-features --features blas
```

**Julia (OMEinsum.jl + TensorOperations.jl):**

```bash
OMP_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --project=. src/main.jl
```

Julia benchmark modes:

- **omeinsum_path** — follows the same pre-computed contraction path as Rust (fair kernel-level comparison)
- **tensorops** — uses TensorOperations.jl's `ncon` for full network contraction

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

- **Language model** instances: many small multi-dimensional tensors (3D/4D) with a few large batch tensors. Many contraction steps with tiny GEMM kernels.
- **Matrix chain** instance: 100 large 2D matrices. Each contraction step is a single large GEMM.

## Benchmark Results

Environment: Apple Silicon M4. Median time (ms) of 5 runs (2 warmup). Julia BLAS: OpenBLAS (lbt).

### 1 thread (`OMP_NUM_THREADS=1`, `RAYON_NUM_THREADS=1`, `JULIA_NUM_THREADS=1`)

#### Strategy: opt_flops

| Instance | strided-rs faer (ms) | strided-rs OpenBLAS (ms) | OMEinsum.jl OpenBLAS (ms) | TensorOperations.jl (ms) |
|---|---:|---:|---:|---:|
| lm_batch_likelihood_brackets_4_4d | 17.095 | 20.838 | 22.229 | - |
| lm_batch_likelihood_sentence_3_12d | 46.373 | 54.721 | 60.480 | - |
| lm_batch_likelihood_sentence_4_4d | 19.161 | 21.338 | 23.772 | - |
| str_matrix_chain_multiplication_100 | 12.342 | 11.034 | 22.643 | 69.159 |

#### Strategy: opt_size

| Instance | strided-rs faer (ms) | strided-rs OpenBLAS (ms) | OMEinsum.jl OpenBLAS (ms) | TensorOperations.jl (ms) |
|---|---:|---:|---:|---:|
| lm_batch_likelihood_brackets_4_4d | 18.555 | 24.583 | 20.815 | - |
| lm_batch_likelihood_sentence_3_12d | 49.950 | 50.263 | 57.162 | - |
| lm_batch_likelihood_sentence_4_4d | 28.089 | 30.366 | 23.673 | - |
| str_matrix_chain_multiplication_100 | 13.504 | 11.686 | 17.961 | 78.007 |

### 4 threads (`OMP_NUM_THREADS=4`, `RAYON_NUM_THREADS=4`, `JULIA_NUM_THREADS=4`)

#### Strategy: opt_flops

| Instance | strided-rs faer (ms) | strided-rs OpenBLAS (ms) | OMEinsum.jl OpenBLAS (ms) | TensorOperations.jl (ms) |
|---|---:|---:|---:|---:|
| lm_batch_likelihood_brackets_4_4d | 18.376 | 18.238 | 29.350 | - |
| lm_batch_likelihood_sentence_3_12d | 32.548 | 29.650 | 51.554 | - |
| lm_batch_likelihood_sentence_4_4d | 20.122 | 17.354 | 27.398 | - |
| str_matrix_chain_multiplication_100 | 13.066 | 8.344 | 13.286 | 54.623 |

#### Strategy: opt_size

| Instance | strided-rs faer (ms) | strided-rs OpenBLAS (ms) | OMEinsum.jl OpenBLAS (ms) | TensorOperations.jl (ms) |
|---|---:|---:|---:|---:|
| lm_batch_likelihood_brackets_4_4d | 19.082 | 19.607 | 22.566 | - |
| lm_batch_likelihood_sentence_3_12d | 32.019 | 31.343 | 50.647 | - |
| lm_batch_likelihood_sentence_4_4d | 26.239 | 25.453 | 27.738 | - |
| str_matrix_chain_multiplication_100 | 14.530 | 13.461 | 12.797 | 41.531 |

**Notes:**
- `-` indicates TensorOperations.jl could not handle the instance (output index appears in multiple input tensors, which `ncon` does not support).
- **strided-rs faer** uses [faer](https://github.com/sarah-quinones/faer-rs) (pure Rust GEMM). **strided-rs OpenBLAS** uses OpenBLAS via `cblas-sys`.
- **strided-rs** and **OMEinsum.jl** use the same pre-computed contraction path for fair comparison.
- **TensorOperations.jl** uses `TensorOperations.ncon` for full network contraction with its own contraction ordering.

## References

- [Einsum Benchmark](https://benchmark.einsum.org/) — standardized einsum benchmark suite
- [ti2-group/einsum_benchmark](https://github.com/ti2-group/einsum_benchmark) — Python package
- [tensor4all/strided-rs](https://github.com/tensor4all/strided-rs) — Rust tensor library
- [OMEinsum.jl](https://github.com/under-Peter/OMEinsum.jl) — Julia einsum library
- [TensorOperations.jl](https://github.com/Jutho/TensorOperations.jl) — Julia tensor contraction library