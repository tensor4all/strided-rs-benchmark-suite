# Raw copy/scale/axpy rank boundary

Measures the borrowed-metadata raw elementwise family of `strided-basic`
(`copy_scale_raw`, `copy_scale_conj_raw`, `axpy_raw`, `axpy_conj_raw`, and
`CopyPlan::execute*`) across ranks that straddle the fused-rank limit, to check
that a caller never has to prepare a plan or keep a cache just to avoid a
rank-shaped performance cliff (tensor4all/strided-rs#300).

## What is compared

| Side | Source |
|---|---|
| baseline | `strided-rs` `7bb8022` (origin/main when the change was prepared) |
| candidate | the change that fuses any original rank before choosing the execution path (diff sha256 `4ee724ab8292b84a51072981c8538bfb23bfa40132c2e567a824a21f03a26d03` on `7bb8022`) |

`harness.rs` is the runner: one process prints one CSV row per
`(op, rank, kind, total)` case with the median-of-iterations ns/element and the
allocations per call, measured with a counting global allocator. `run_paired.sh`
alternates the two binaries; the protocol is in `PROTOCOL.md` and the results
tables are in `results/`.

Cases (the issue's predeclared list): ranks {2, 4, 6, 8, 9, 12, 16} × layouts
{contiguous, padded with the short inner axis, padded with the long inner axis,
reversed source} × element counts {4096, 1048576} × ops
{`axpy_raw`, `copy_scale_raw`, `CopyPlan::execute`} = 168 cases.

Host: AMD EPYC 7713P, 64C/128T, Ubuntu 24.04, `schedutil` governor, a co-tenant
at roughly 50% load. No isolated CPUs. One worker (`taskset -c 5`,
`RAYON_NUM_THREADS=1`) for the main table, four workers for the secondary table.

## Result summary

15 interleaved repetitions, per-case median. A null A/A control (the baseline
binary against itself, 7 repetitions, `results/summary-null-aa-7.csv`) puts the
protocol's resolution at roughly ±1%: p05 0.984, p50 1.000, p95 1.010, max 1.052.

### The cliff (original rank > 8 and fused rank > 8): 36 cases

These are the layouts for which the baseline left the fused replay and built a
view plan on every call.

| | baseline | candidate |
|---|---|---|
| allocations per call | 46 … 112 | 1 … 2 |
| ns/element relative to baseline | 1.00 | **0.32 … 0.96** |

34 of 36 cases improve by 1.04x-3.13x. The two exceptions are `axpy_raw` on
padded layouts with a long inner axis at 1M elements (ranks 9 and 12), where the
baseline was already at the memory-bandwidth floor (0.307-0.321 ns/element)
despite building a plan, and the fused replay is 3.4-3.9% slower. Nothing was
left to win there.

### The common path (original rank <= 8): 96 cases

The change does not touch the fused replay these layouts already used. 82 of the
96 cases have a baseline spread `(max-min)/median` above 5% and cannot resolve a
5% difference at all. Of the 14 clean cases the worst is `axpy_raw` rank 6
`padded-short-inner` at +5.5% (1.055, baseline spread 0.04); the next are +2.3%
and +1.8%.

### Fused-small high-rank layouts (original rank > 8, fused rank <= 8): 36 cases

Here the baseline never used the fused replay either: it reached whichever view
kernel fast path applied without planning. The candidate is generally faster,
often much faster (`CopyPlan::execute` on a contiguous rank-9/12/16 tensor:
2.93 → 0.185 ns/element, 15x; `copy_scale_raw` on the same layouts:
0.69-0.80 → 0.21 ns/element, 3.7x). The exception is `axpy_raw` on *fully
contiguous* layouts at 4096 elements, where the baseline's fallback hit the view
kernels' AVX-512-dispatched contiguous axpy: 0.21 → 0.27-0.30 ns/element
(1.28-1.44x). At 1M elements the same cases are at parity (0.281 vs 0.283
ns/element) because both are DRAM/L3-bandwidth bound. The raw fused replay is a
closure-based loop and cannot match a runtime-dispatched SIMD leaf for that one
shape; a naive `pulp` dispatch inside the replay was measured and rejected
(0.70 ns/element, 3.6x worse).

### Four threads, 1M elements (`results/summary-4t-7.csv`)

7 interleaved repetitions. The cliff improves as at one thread (rank 9
`padded-short-inner`: 3.59 → 2.74; ranks 12 and 16: 7.0 → 2.8 and 2.8, and
`CopyPlan::execute` 6.7-7.0 → 1.6-1.9). High-rank contiguous `axpy_raw` is
1.07-1.30x slower than the baseline, which reached the view kernels' parallel
branch there; the raw path itself remains serial by contract, for every rank.

## Reproducing

```bash
# two cargo projects, each depending on one side's strided-basic
# base-serial/, cand-serial/ (default features), base-parallel/, cand-parallel/
#  (features = ["parallel"]), each with src/main.rs = harness.rs
(cd base-serial && cargo build --release)
(cd cand-serial && cargo build --release)
RAYON_NUM_THREADS=1 taskset -c 5 base-serial/target/release/rawrank300 <totals>
./run_paired.sh <base-bin> <cand-bin> results 15 4096 1048576
```

`results/summary-*.csv` holds the aggregated per-case table
(`base_median_ns_per_elem`, `cand_median_ns_per_elem`, ratio, min/max, baseline
relative spread, allocations per call); the per-repetition CSVs are in the
`per-repetition-*.tar.gz` archives.
