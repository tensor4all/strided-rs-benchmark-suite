# Raw copy/scale/axpy rank boundary — and the view-path comparison

Measures the borrowed-metadata raw elementwise family of `strided-basic`
(`copy_scale_raw`, `copy_scale_conj_raw`, `axpy_raw`, `axpy_conj_raw`, and
`CopyPlan::execute*`) and the equivalent view entry points (`axpy`,
`copy_scale`, `copy_into`) across ranks that straddle the fused-rank limit, to
check that a caller never has to prepare a plan or keep a cache just to avoid a
rank-shaped performance cliff (tensor4all/strided-rs#300).

## What is compared

| Side | Source |
|---|---|
| baseline | `strided-rs` `7bb8022` (origin/main when the change was prepared) |
| candidate | the change that fuses any original rank before choosing the execution path, plus the by-value fix for the view path's dispatched contiguous leaf (staged-diff sha256 `0401e30756e237b39f33b8afcdfa2a4a4c7769723982b9640c8a04c3061b2830` on `7bb8022`) |

`harness.rs` is the raw-API runner and `harness_view.rs` the view-API runner; one
process prints one CSV row per `(op, rank, kind, total)` case with the
median-of-iterations ns/element and the allocations per call, measured with a
counting global allocator. `run_paired.sh` alternates the two binaries; the
protocol is in `PROTOCOL.md`, the aggregated tables in `results/`.

Cases (the issue's predeclared list): ranks {2, 4, 6, 8, 9, 12, 16} × layouts
{contiguous, padded with the short inner axis, padded with the long inner axis,
reversed source} × element counts {4096, 1048576} × three operations per API.

Host: AMD EPYC 7713P, 64C/128T, Ubuntu 24.04, `schedutil` governor, a co-tenant
at roughly 50% load. No isolated CPUs. One worker (`taskset -c 5`,
`RAYON_NUM_THREADS=1`) for the main tables, four workers for the secondary one.

## Result summary

15 interleaved repetitions, per-case median. A null A/A control (the baseline
binary against itself, `results/summary-null-aa-7.csv`) puts the protocol's
resolution *within* one experiment at about ±1%: p05 0.984, p50 1.000, p95 1.010,
max 1.052. Between complete experiments of the same source the per-case median
moves by up to ~5%, and — see the last section — any change anywhere in the crate
can move the short-run layouts by 8-24%, so per-case deltas below that are not
attributable to a specific edit.

### Raw API (`results/summary-raw-1t-15.csv`, `results/summary-raw-4t-7.csv`)

| case set | result |
|---|---|
| fused rank > 8 (36 cases, the issue's target) | allocations per call **46-112 → 1-2**; median ns/element **0.98x-3.12x** of baseline |
| contiguous layouts, all ranks | `CopyPlan::execute` up to 15x, `copy_scale_raw` up to 3.8x, `axpy_raw` at ranks 2-8 1.2-1.4x |
| fully contiguous `axpy_raw`, original rank 9-16, 4096 elements | **1.27-1.33x slower** (0.207-0.213 → 0.265-0.279 ns/element) |
| `padded-short-inner`, ranks 2-8 | within 1.05x |

The two rank-9/12 padded long-inner cases at 1M elements were already at the
memory-bandwidth floor in the baseline, so they stay there.

The fully contiguous high-rank `axpy_raw` case is the one real regression: the
baseline left the fused replay for the view kernel, whose contiguous leaf is
runtime-dispatched (AVX2) while the fused replay is compiled for the x86-64
baseline ISA. At 1M elements both are bandwidth-bound and equal. The measured fix
is in the last section — it works, but costs more than it gains in this crate.

### View API (`results/summary-view-1t-15.csv`)

- `copy_scale` on contiguous layouts: **3.1-3.4x faster** (0.68-0.74 → 0.19-0.24
  ns/element). The dispatched leaf in `map_parts_into_validated` captured its
  operation closure by reference, so it never vectorized; taking it by value out
  of line fixes it. `axpy`'s own leaf is unchanged and already at that speed.
- `padded-short-inner`, rank 6-8: **+13-27%** in this build. The view kernels'
  source for those layouts is unchanged, and the same swing appears in a control
  that contains only the raw-side change (see below), so it is a whole-crate
  codegen effect rather than a semantic change.
- The view path still has the planner cliff for its own callers: on a rank-9
  padded 4096-element layout `axpy`/`copy_scale` cost 7.0-7.4 ns/element and 64
  allocations per call where the raw replay now costs 2.5-2.6 and 1-2. Routing the
  two-operand view ops through the fused traversal is the fix; it changes the view
  API's blocking/parallel strategy and needs its own protocol.

### Four threads, 1M elements (`results/summary-raw-4t-7.csv`)

7 interleaved repetitions. The cliff improves as at one thread (rank 9
`padded-short-inner` 3.59 → 2.74; ranks 12 and 16 7.0 → 2.8, `CopyPlan::execute`
6.7-7.0 → 1.6-1.9). High-rank contiguous `axpy_raw` is 1.07-1.30x slower than the
baseline, which reached the view kernels' parallel branch there; the raw path
itself remains serial by contract, at every rank.

## Follow-up measurements

### Instruction-set detection is not the cost

| call | ns/call |
|---|---|
| `pulp::Arch::new()` | 0.55 |
| `pulp::Arch::new().dispatch(\|\| ())` | 0.45 |
| `fearless_simd::Level::new()` | 0.71 |
| `fearless_simd::dispatch!(Level::new(), …)` | 0.70 |

Both cache their detection in a static (`__static_available::AVAILABLE` for
pulp), i.e. about 0.1% of a 4096-element call. A per-call dispatch is affordable;
the cost is in whether the dispatched closure vectorizes.

### Moving the operation into the dispatched closure is what vectorizes it

A dispatched closure that only *borrows* the operation closure (which is what
`map_parts_into_validated` and the raw replay both did) is left scalar, with a
reload of the captured payload per element. Moving the operation in by value
(the pattern `elem-simd-math-rs` documents for `fearless_simd::dispatch!`) fixes
it: a contiguous rank-2 4096-element `axpy_raw` goes 0.31 → **0.19 ns/element**,
matching `-C target-cpu=native` (0.188) and beating the view leaf (0.21).

### Why the raw side did not get it

Adding that dispatch to the raw family (at the entry points, in
`apply_fused_pair`, and with and without `#[inline(never)]` on either replay
function) each time cost the *short-run* layouts 10-24%:

| placement | contiguous `axpy_raw` | `padded-short-inner` rank 2 / rank 8 |
|---|---|---|
| none (this change) | 0.25 (unchanged) | 1.33 / 1.79 (unchanged) |
| entry-point dispatch | **0.19** | 1.50 / 2.10 |
| entry + `#[inline(never)]` on `apply_fused_pair` | **0.19** | 1.50 / 1.76-1.95 |
| entry + `#[inline(never)]` on `apply_fused_range` | **0.19** | 1.48-1.54 / 1.98-2.00 |

+0.17-0.31 ns/element on the short-run layouts against -0.06 on the contiguous
one is the wrong trade in absolute time, so it is left out. Landing it needs the
SIMD leaf monomorphized on its own (an op-typed kernel for the raw/view
elementwise family) rather than another branch placement.

### Whole-crate codegen sensitivity

A control binary containing only the raw-side change (view sources untouched)
still moves the *view* path's padded layouts by +8-13% on this host. The carry
loop that runs once per short run is that sensitive to the inlining and register
allocation of whatever surrounds it, which is why the "same loop as before"
argument in the work log is backed by measurements rather than by a claim about
generated code.

## Reproducing

```bash
# one cargo project per side and harness:
#   base-serial/, cand-serial/, base-parallel/, cand-parallel/  (src/main.rs = harness.rs)
#   viewbase/, viewcand/                                       (src/main.rs = harness_view.rs)
(cd base-serial && cargo build --release)
(cd cand-serial && cargo build --release)
RAYON_NUM_THREADS=1 taskset -c 5 base-serial/target/release/rawrank300 <totals>
./run_paired.sh <base-bin> <cand-bin> results 15 4096 1048576
```

`results/summary-*.csv` holds the aggregated per-case table
(`base_median_ns_per_elem`, `cand_median_ns_per_elem`, ratio, min/max, baseline
relative spread, allocations per call); the per-repetition CSVs are in the
`per-repetition-*.tar.gz` archives.
