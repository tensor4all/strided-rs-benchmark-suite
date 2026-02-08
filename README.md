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
    generate_dataset.py     # Filter & export benchmark instances as JSON
  data/
    instances/              # Exported JSON metadata (one file per instance)
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

### 2. Run Rust benchmark (strided-opteinsum)

```bash
cargo run --release
```

Executes all instances using pre-computed contraction paths with two strategies: `opt_flops` (minimize FLOPs) and `opt_size` (minimize memory).

### 3. Run Julia benchmark (OMEinsum.jl)

```bash
julia --project=. src/main.jl
```

Executes all instances in two modes:

- **path** — follows the same pre-computed contraction path as Rust (fair comparison)
- **optimizer** — uses OMEinsum's built-in `TreeSA` optimizer to find its own contraction order

### Row-major to Column-major Conversion

NumPy arrays are row-major (C order). strided-rs uses column-major (Fortran order). The conversion is metadata-only:

- Reverse each tensor's shape: `[M, K]` -> `[K, M]`
- Reverse each operand's index labels: `"ij,jk->ik"` -> `"ji,kj->ki"`

Both the original (`format_string`, `shapes`) and converted (`format_string_colmajor`, `shapes_colmajor`) metadata are stored in each JSON file.

## Benchmark Results

Environment: Apple M2 (macOS), 2 warmup + 5 timed runs, average reported.

### Rust (strided-opteinsum) — pre-computed path

| Instance | Tensors | log10[FLOPS] | log2[SIZE] | opt_flops (ms) | opt_size (ms) |
|----------|--------:|-------------:|-----------:|---------------:|--------------:|
| lm_batch_likelihood_brackets_4_4d | 84 | 8.37 | 18.96 | 17.765 | 19.047 |
| lm_batch_likelihood_sentence_3_12d | 38 | 9.20 | 20.86 | 49.853 | 52.237 |
| lm_batch_likelihood_sentence_4_4d | 84 | 8.46 | 18.89 | 19.798 | 25.537 |
| str_matrix_chain_multiplication_100 | 100 | 8.48 | 17.26 | 11.379 | 11.412 |

### Julia (OMEinsum.jl) — pre-computed path (same contraction order as Rust)

| Instance | Tensors | log10[FLOPS] | log2[SIZE] | opt_flops (ms) | opt_size (ms) |
|----------|--------:|-------------:|-----------:|---------------:|--------------:|
| lm_batch_likelihood_brackets_4_4d | 84 | 8.37 | 18.96 | 50.069 | 34.437 |
| lm_batch_likelihood_sentence_3_12d | 38 | 9.20 | 20.86 | 49.223 | 48.554 |
| lm_batch_likelihood_sentence_4_4d | 84 | 8.46 | 18.89 | 17.909 | 34.686 |
| str_matrix_chain_multiplication_100 | 100 | 8.48 | 17.26 | 33.875 | 27.686 |

### Julia (OMEinsum.jl) — OMEinsum optimizer (TreeSA)

| Instance | Tensors | log10[FLOPS] | log2[SIZE] | opt_flops (ms) | opt_size (ms) |
|----------|--------:|-------------:|-----------:|---------------:|--------------:|
| lm_batch_likelihood_brackets_4_4d | 84 | 8.37 | 18.96 | 35.289 | 13.912 |
| lm_batch_likelihood_sentence_3_12d | 38 | 9.20 | 20.86 | 41.950 | 40.941 |
| lm_batch_likelihood_sentence_4_4d | 84 | 8.46 | 18.89 | 41.759 | 52.631 |
| str_matrix_chain_multiplication_100 | 100 | 8.48 | 17.26 | 30.149 | 11.238 |

## References

- [Einsum Benchmark](https://benchmark.einsum.org/) — standardized einsum benchmark suite
- [ti2-group/einsum_benchmark](https://github.com/ti2-group/einsum_benchmark) — Python package
- [tensor4all/strided-rs](https://github.com/tensor4all/strided-rs) — Rust tensor library
- [OMEinsum.jl](https://github.com/under-Peter/OMEinsum.jl) — Julia einsum library