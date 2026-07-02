# Strided Permutation Benchmark

This benchmark compares materializing strided/permuted `f64` tensors into a
col-major destination. The benchmark patterns live in
[`patterns.json`](patterns.json), and the Rust and Julia runners consume that
same file.

The unified pattern set incorporates the third-party transpose comparison from
[ultimatile/297160e33005b42595c3e2a416e0549b](https://gist.github.com/ultimatile/297160e33005b42595c3e2a416e0549b):
naive, HPTT, and `strided-perm` are compared on the same col-major `f64`
transpose semantics, with correctness checked before timing.

## Pattern Schema

Each pattern records the operation, not the result:

- `shape`: source tensor shape.
- `perm`: 0-indexed permutation with semantics
  `out[i0,...,ik] = src[i_perm0,...,i_permk]`.
- `src_layout` and `dst_layout`: currently `col_major` or explicit source
  strides.
- `participants`: implementations that should be reported for the pattern.
- `data`: fixed to `deterministic_index_value`; no random seed is recorded.

HPTT rows are emitted only for patterns with contiguous source and destination
layouts. Explicit source-stride cases such as `tn_light_415_24d_scattered_to_colmajor`
are intentionally excluded from HPTT because the public HPTT API does not
represent arbitrary source strides.

## Run

Run thread-count variants sequentially. Do not run benchmark processes in
parallel.

Rust, serial:

```bash
RUSTFLAGS="-C target-cpu=native" \
  RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  cargo run --release --bin permute
```

Rust, parallel `strided-perm`:

```bash
RUSTFLAGS="-C target-cpu=native" \
  RAYON_NUM_THREADS=4 OMP_NUM_THREADS=1 \
  cargo run --release --features parallel --bin permute
```

Rust with HPTT rows:

```bash
RUSTFLAGS="-C target-cpu=native" \
  RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  cargo run --release --features hptt --bin permute
```

Rust with parallel `strided-perm` and HPTT using the same Rayon thread count:

```bash
RUSTFLAGS="-C target-cpu=native" \
  RAYON_NUM_THREADS=4 OMP_NUM_THREADS=1 \
  cargo run --release --features parallel,hptt --bin permute
```

Julia Base and Strided.jl:

```bash
JULIA_NUM_THREADS=1 julia --project=. \
  benchmarks/strided_benchmarks/permute/permute.jl

JULIA_NUM_THREADS=4 julia --project=. \
  benchmarks/strided_benchmarks/permute/permute.jl
```

To run one pattern while investigating:

```bash
PATTERN_ID=transpose_2d_1024 cargo run --release --features hptt --bin permute

PATTERN_ID=transpose_2d_1024 julia --project=. \
  benchmarks/strided_benchmarks/permute/permute.jl
```

## Implementations

| Runner | Implementations |
|---|---|
| Rust | naive odometer, `std::ptr::copy_nonoverlapping`, `strided_perm::copy_into`, `strided_perm::copy_into_col_major`, parallel variants, optional `hptt` crate |
| Julia | Base `permutedims!` or generic `copyto!`, Strided.jl `@strided` materialization |

## Results

Full results should be regenerated from the unified runners after benchmark
changes. Record the `strided-rs` git hash beside the result table when updating
this page.
