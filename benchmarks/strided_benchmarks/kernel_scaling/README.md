# Kernel scaling

Coverage for [tensor4all/strided-rs#269](https://github.com/tensor4all/strided-rs/issues/269):
elementwise, reduction, and structural copy kernels measured through the erased
entries, the typed entries, and a raw pointer baseline, at 1 and 4 threads,
next to Julia Base and Strided.jl. A gate script turns the four defect classes
of the issue into threshold checks:

| Defect in #269 | Cases that expose it | Gate |
|---|---|---|
| Erased elementwise entries dispatch the op per element | `ew_*` erased and erased_uninit versus typed and raw | (b), (c) |
| Parallel reduce paths use a scalar single accumulator leaf | `red_*_all_*`, `red_*_axis0_*` versus raw 8 lane accumulators | (a), (c) |
| Contiguous axis reductions with a strided output take the scalar general path | `red_*_axis1_*` (and axis0 with its compact output) | (b), (c), (d) |
| CopyPlan structural ops have no parallel branch | `copy_*` at 4T versus 1T | (a) |

## Files

- `kernel_scaling.rs`: Rust binary (`[[bin]] kernel_scaling`, feature `parallel`).
- `kernel_scaling.jl`: Julia counterpart with the same case names and shapes.
- `run.sh`: builds against any strided-rs tree and writes the CSVs.
- `gate.py`: reads the CSVs, prints a table, and exits nonzero on flags (Python stdlib only).

## Building against a strided-rs tree

The suite `Cargo.toml` points at the sibling `../strided-rs`. To measure any
other checkout without editing tracked files, `run.sh` writes a standalone
manifest under `target/kernel-scaling/<rev>/` (gitignored) whose only path
dependency is `$STRIDED_RS_DIR/strided-kernel` with features `parallel,simd`.
The strided-rs revision is taken from `git rev-parse` in that tree, suffixed
with `-dirty` when it has uncommitted tracked changes, compiled into the binary,
printed in its header, and used in the output directory name
`data/results/kernel-scaling-<rev>/`. `RUSTFLAGS` defaults to
`-C target-cpu=native`.

```sh
page=benchmarks/strided_benchmarks/kernel_scaling
export STRIDED_RS_DIR=/path/to/strided-rs

$page/run.sh build            # compile only
$page/run.sh check 4          # correctness of every case at 4T, no timing
$page/run.sh rust 1           # data/results/kernel-scaling-<rev>/rust_1t.csv
$page/run.sh rust 4           # data/results/kernel-scaling-<rev>/rust_4t.csv
$page/run.sh julia 1          # data/results/kernel-scaling-<rev>/julia_1t.csv
$page/run.sh julia 4          # data/results/kernel-scaling-<rev>/julia_4t.csv
python3 $page/gate.py data/results/kernel-scaling-<rev>
```

Run the four timing commands one after another, never concurrently, on a quiet
host. On Linux the runs are pinned with `taskset -c ${CPUS:-0-(T-1)}`. On macOS
`taskset` does not exist and the script says that pinning was not applied.
`OUTPUT_DIR` overrides the result directory. Julia uses `julia --project=.`
at the suite root (Strided 2.6.1).

With the sibling checkout the binary can also be built in tree:
`cargo build --release --features parallel --bin kernel_scaling`, then
`target/release/kernel_scaling --threads 4 [--time]`.

### Binary options

| Setting | Default | Meaning |
|---|---|---|
| `--threads N` | required | thread budget, enforced (see below) |
| `--time` | off | without it every case runs once and is checked; with it the CSV is printed |
| `BENCH_RUNS` | 11 | timed samples per row; the median is reported |
| `BENCH_WARMUP` | 2 | untimed warmup calls per row |
| `KERNEL_SCALING_SHRINK` | 1 | divides every 2D extent by S and every 1D length by S squared (smoke runs, for example 16) |
| `BENCH_FILTER` | empty | only cases whose name contains this substring |

The Julia script reads the same environment variables and `--time`.

### Thread enforcement

The Rust binary builds the global Rayon pool with exactly N threads and runs
every case inside `rayon::scope`, so all work, including the raw baselines,
runs on that pool. Strided entries receive `ExecContext::serial()` at 1T and
`ExecContext::max_threads(N)` otherwise; typed entries run inside
`ExecContext::run`. Before any case the binary asserts that
`rayon::current_num_threads() == N`, that the body runs on a pool worker, that a
4M element typed `map_into` probe touched at most N distinct workers and none
outside the pool, and that the process has at most N + 1 OS threads
(`/proc/self/status` on Linux, `ps -M` on macOS). The header line on stderr
prints all of these, for example:

```
# threads: requested=4 rayon_pool=4 exec_ctx=ExecContext { kind: MaxThreads(4) } probe_workers=4 os_threads=5
```

The Julia script asserts `Threads.nthreads() == JULIA_NUM_THREADS` and sets
`Strided.set_num_threads(N)`.

## Protocol

Inputs, outputs, erased descriptors, compiled plans, and views are created
before timing. Each row is `BENCH_WARMUP` untimed calls followed by
`BENCH_RUNS` timed calls; the median is reported. Inputs and outputs pass
through `black_box`. After each variant is timed its output is compared on
every element with a plain sequential reference (exact for elementwise f64,
copies and max/min, relative 1e-12 for complex elementwise, relative 1e-9 for
sum and prod). Check mode runs the same comparisons with a single call.
Exception: typed `reduce_axis` returns a freshly allocated `StridedArray`, so
that allocation is inside its timed region. It is the only typed per axis API.

CSV columns: `case,variant,threads,median_ns,samples`.

## Cases

Sizes are nominal (`KERNEL_SCALING_SHRINK=1`). All arrays are column major f64
unless stated.

| Family | Case | Shape |
|---|---|---|
| Elementwise | `ew_{add,sub,mul,div,max,min,neg,abs}_f64_contig` | 1D, 33554432 elements |
| Elementwise | `ew_{add,sub,mul,div,max,min,neg,abs}_f64_trans` | 8192 x 4096; lhs is row major (strides `[4096, 1]`), rhs and destination column major |
| Elementwise | `ew_{add,mul,div,neg,conj,abs}_c64_contig` | 1D, 33554432 Complex64 (abs writes f64) |
| Elementwise | `ew_conj_c64_trans` | 8192 x 4096 Complex64, transposed source |
| Reduction | `red_{sum,prod,max,min}_{all,axis0,axis1}_{8192x4096,2048x2048}` | column major source; axis0 reduces the contiguous axis, axis1 the strided axis |
| Structural | `copy_slice_step2` | 8192 x 4096 to 4096 x 4096, step `[2, 1]` |
| Structural | `copy_reverse_axis0`, `copy_reverse_axis1` | 4096 x 4096 |
| Structural | `copy_concat_axis0`, `copy_concat_axis1` | two halves into 4096 x 4096 |
| Structural | `copy_dynslice_rank1` | length 33554432 operand, 16777216 window, start 5592405 |
| Structural | `copy_dynslice_rank2` | 4608 x 4608 operand, 4096 x 4096 window, starts `[100, 200]` (i64) |
| Structural | `copy_pad` | 4000 x 4000 padded by 48 on every side to 4096 x 4096, fill 0 |

Binary max and min are NaN propagating, matching `ErasedZipOp::Maximum` and
`Minimum`; the typed and raw closures use the same function.

## Variants

| Variant | Rust entry | Julia API |
|---|---|---|
| `raw` | slice loops the compiler vectorizes, split into contiguous chunks over the same Rayon pool; 64 x 64 tiles for transposed reads; 8 lane accumulators (with a NaN flag for max/min) for reductions; `copy_nonoverlapping` or per column loops for copies | |
| `typed` | `zip_map2_into`, `map_into`, `reduce`, `reduce_axis`, `SlicePlan`, `ReversePlan`, `ConcatenatePlan`, `DynamicSlicePlan`, `PadPlan` inside `ExecContext::run` | |
| `erased` | `erased_zip_into`, `erased_map_into`, `ErasedReducePlan::compile` or `compile_axes`, `Erased{Slice,Reverse,Concatenate,DynamicSlice,Pad}Plan::execute` | |
| `erased_uninit` | `erased_zip_into_uninit`, `erased_map_into_uninit` (elementwise only) | |
| `julia_base` | | elementwise: `out .= f.(a, b)` into a preallocated array; reductions: `sum(A)` etc. for all, `sum!(out, A)` etc. for dims |
| `julia_strided` | | elementwise and copies: `@strided so .= ...` on `StridedView`s; reductions: `sum(StridedView(A))` for all, `sum!(StridedView(out), StridedView(A))` for dims |
| `julia_alloc` | | allocating Base call: `sum(A; dims=d)`, `A[1:2:end, :]`, `reverse(A; dims=d)`, `vcat` or `hcat`, `A[s+1:s+W, ...]` |
| `julia_copyto` | | copies: `copyto!(out, view(A, ...))`; concat: two `copyto!` into views of `out`; pad: zero the four border strips, then `copyto!` the interior |

The Julia reductions use `sum`, `prod`, `maximum`, `minimum` and their
in place forms. Julia has no pad or dynamic slice primitive, so those rows use
the preallocated copies listed above.

## Gate

```sh
python3 benchmarks/strided_benchmarks/kernel_scaling/gate.py DIR_OR_CSVS... [--report-only]
```

Strided variants are `typed`, `erased`, and `erased_uninit`. Flags:

| Flag | Condition | Default |
|---|---|---|
| (a) | a strided variant is not at least 1.3x faster at 4T than at 1T, for cases whose 1T median is at least 1 ms (tensor sized); the raw scaling of the same case is printed for context | `--scaling 1.3 --threads 4 --min-scaling-ns 1e6` |
| (b) | an erased variant is more than 1.25x slower than typed at the same thread count | `--erased 1.25` |
| (c) | a strided variant is more than 1.5x slower than `raw` at the same thread count | `--raw 1.5` |
| (d) | a strided variant is more than 2x slower than the fastest Julia variant at the same thread count | `--julia 2.0` |

`--min-ns` skips (b), (c) and (d) when the strided side is below that many ns
(default 0). `--filter` restricts to matching cases. The script exits 1 when
anything is flagged unless `--report-only` is given.

## Not covered

- Complex elementwise is limited to add, mul, div, neg, conj, abs on the
  contiguous layout plus conj on the transposed layout. Complex sub is
  omitted, and erased max and min do not accept complex inputs.
- Uninitialized destination entries of the structural plans are not measured.
- Typed reductions have no preallocated per axis entry, so `typed` axis rows
  include the output allocation.
- Only f64 and Complex64 are exercised; f32 and integer dtypes are not.

## Results: pending coordinator run

No timings have been recorded yet. Only correctness checks and a
`KERNEL_SCALING_SHRINK=16` smoke run at 1T and 4T were executed, against
strided-rs `c1c14a6` (v0.4.2) on macOS, where CPU pinning is unavailable.
Record the strided-rs revision, host, CPU list, and `RUSTFLAGS` beside the
tables when the measured results are added.
