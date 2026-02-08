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
    run_all.sh              # Run all benchmarks (single-threaded)
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

```bash
cargo build --release
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
./scripts/run_all.sh
```

Runs Rust and Julia benchmarks with single-threaded execution enforced (`OMP_NUM_THREADS=1`, `RAYON_NUM_THREADS=1`, `JULIA_NUM_THREADS=1`). Results are saved to `data/results/`.

### 3. Run individually

**Rust (strided-opteinsum):**

```bash
RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 cargo run --release
```

**Julia (OMEinsum.jl + TensorOperations.jl):**

```bash
OMP_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --project=. src/main.jl
```

Julia benchmark modes:

- **omeinsum_path** — follows the same pre-computed contraction path as Rust (fair comparison)
- **omeinsum_opt** — uses OMEinsum's built-in `TreeSA` optimizer to find its own contraction order
- **tensorops** — uses TensorOperations.jl's `ncon` for full network contraction

### Row-major to Column-major Conversion

NumPy arrays are row-major (C order). strided-rs uses column-major (Fortran order). The conversion is metadata-only:

- Reverse each tensor's shape: `[M, K]` -> `[K, M]`
- Reverse each operand's index labels: `"ij,jk->ik"` -> `"ji,kj->ki"`

Both the original (`format_string`, `shapes`) and converted (`format_string_colmajor`, `shapes_colmajor`) metadata are stored in each JSON file.

## Reproducing Benchmarks

Run all benchmarks (Rust + Julia) with single-threaded execution:

```bash
./scripts/run_all.sh
```

This script:

1. Sets `OMP_NUM_THREADS=1`, `RAYON_NUM_THREADS=1`, `JULIA_NUM_THREADS=1`
2. Builds and runs the Rust benchmark (`cargo run --release`)
3. Runs the Julia benchmark (`julia --project=. src/main.jl`)
4. Formats results as a markdown table via `scripts/format_results.py`
5. Saves all outputs to `data/results/` with timestamps

To format existing log files into a markdown table:

```bash
uv run python scripts/format_results.py data/results/rust_*.log data/results/julia_*.log
```

## Benchmark Results

Environment: Apple Silicon M2, single-threaded. Median time (ms) of 5 runs (2 warmup).

All benchmarks run with `OMP_NUM_THREADS=1`, `RAYON_NUM_THREADS=1`, `JULIA_NUM_THREADS=1`. BLAS: OpenBLAS (lbt).

### Strategy: opt_flops

| Instance | Rust strided-opteinsum (ms) | Julia OMEinsum path (ms) | Julia OMEinsum opt (ms) | Julia TensorOps (ms) |
|---|---:|---:|---:|---:|
| lm_batch_likelihood_brackets_4_4d | 16.726 | 18.263 | 15.659 | - |
| lm_batch_likelihood_sentence_3_12d | 45.747 | 56.953 | 40.618 | - |
| lm_batch_likelihood_sentence_4_4d | 18.302 | 19.237 | 17.567 | - |
| str_matrix_chain_multiplication_100 | 10.139 | 15.786 | 14.521 | 66.578 |

### Strategy: opt_size

| Instance | Rust strided-opteinsum (ms) | Julia OMEinsum path (ms) | Julia OMEinsum opt (ms) | Julia TensorOps (ms) |
|---|---:|---:|---:|---:|
| lm_batch_likelihood_brackets_4_4d | 17.404 | 18.482 | 14.636 | - |
| lm_batch_likelihood_sentence_3_12d | 50.000 | 47.817 | 41.130 | - |
| lm_batch_likelihood_sentence_4_4d | 24.120 | 19.103 | 17.800 | - |
| str_matrix_chain_multiplication_100 | 10.593 | 14.444 | 15.221 | 65.774 |

**Notes:**
- `-` indicates TensorOperations.jl could not handle the instance (output index appears in multiple input tensors, which `ncon` does not support).
- **Rust strided-opteinsum** and **Julia OMEinsum path** use the same pre-computed contraction path for fair comparison.
- **Julia OMEinsum opt** uses OMEinsum's `TreeSA` optimizer (optimization time excluded from measurement).
- **Julia TensorOps** uses `TensorOperations.ncon` for full network contraction with its own contraction ordering.

## References

- [Einsum Benchmark](https://benchmark.einsum.org/) — standardized einsum benchmark suite
- [ti2-group/einsum_benchmark](https://github.com/ti2-group/einsum_benchmark) — Python package
- [tensor4all/strided-rs](https://github.com/tensor4all/strided-rs) — Rust tensor library
- [OMEinsum.jl](https://github.com/under-Peter/OMEinsum.jl) — Julia einsum library
- [TensorOperations.jl](https://github.com/Jutho/TensorOperations.jl) — Julia tensor contraction library