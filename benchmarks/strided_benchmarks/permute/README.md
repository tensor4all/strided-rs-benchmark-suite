# Strided Permutation Benchmark

Compares credible naive baselines with `strided_perm` copy/permutation paths.
For cases that HPTT can represent directly, this page also reports HPTT C++
transpose timings from the companion `hptt_compare.cpp` runner.

Cases that require arbitrary source strides are intentionally not compared
against HPTT because HPTT's public API accepts contiguous source tensors.

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

For HPTT-compatible contiguous-source transpose cases:

```bash
HPTT_DIR=../hptt CXX=/opt/homebrew/bin/g++-15 \
  THREADS=1 WARMUP=3 NRUNS=11 \
  benchmarks/strided_benchmarks/permute/run_hptt_compare.sh

HPTT_DIR=../hptt CXX=/opt/homebrew/bin/g++-15 \
  THREADS=4 WARMUP=3 NRUNS=11 \
  benchmarks/strided_benchmarks/permute/run_hptt_compare.sh
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

### HPTT-Compatible Transpose Cases

These rows come from `hptt_compare.cpp`, which checks every reported case for
correctness. They are listed here only for contiguous-source transpose cases
that correspond to the permutation benchmark's large 3D transpose scenarios.

#### 1T

| Scenario | HPTT execute | HPTT create+execute |
|---|---:|---:|
| 2D 1024^2 transpose [1,0] | 0.655 | 0.648 |
| 3D 256^3 transpose [2,0,1] | 24.473 | 24.477 |
| 3D 256^3 transpose [1,0,2] | 14.329 | 14.310 |

#### 4T

| Scenario | HPTT execute | HPTT create+execute |
|---|---:|---:|
| 2D 1024^2 transpose [1,0] | 0.201 | 0.224 |
| 3D 256^3 transpose [2,0,1] | 7.929 | 7.956 |
| 3D 256^3 transpose [1,0,2] | 4.384 | 4.614 |

### HPTT Exclusions

- `24D scattered -> col-major` uses arbitrary source strides, so it is not a
  direct HPTT API comparison.
- The high-rank binary contiguous-source permutation is covered by
  `strided_perm`, but HPTT high-rank timings are omitted unless the suite also
  carries an equivalent correctness checker.
