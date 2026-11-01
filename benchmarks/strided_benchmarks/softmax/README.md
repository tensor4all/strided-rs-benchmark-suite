# Fused softmax (`strided-nn`)

One `softmax_axis0_f32` / `softmax_axis0_f64` call over contiguous column-major
`[axis_len, columns]` data, with caller scratch and an explicit `Exec`
placement token. The timing covers the kernel call only: the host runtime's
dtype dispatch, pooled output and plan cache are outside this crate's contract.

```bash
cargo build --release --features parallel
THREADS=1 ./target/release/softmax
THREADS=4 ./target/release/softmax
THREADS=8 ./target/release/softmax
```

Run the thread counts one at a time, never two benchmark processes in parallel.
`THREADS` sets the budget of a token minted from a pool of that width
(`Exec::new`; `THREADS=1` uses `Exec::serial()`), and `AXIS_LENS`, `COLUMNS`,
`RUNS` and `MAX_ELEMENTS` narrow the matrix. The pool is built and one token is
minted before the first timed region, so the allocation column measures the
steady state.

The table reports median and inter-quartile range of one call, GB/s over
read+write traffic, and the allocations observed during the timed loop. The
steady-state allocation column is the point of the caller-scratch contract: a
warm kernel must show zero. A single run that walks many cases in one process
can charge one one-time worker initialisation to one row; measuring that case
on its own reports zero.

Measured against the published `strided-nn` 0.7.0 on 2026-11-02 (macOS arm64,
18 cores, unpinned — no `taskset` available), one call, `RUNS=10`:

| dtype | axis_len | columns | threads | median_ms | GB/s | allocations |
| --- | --- | --- | --- | --- | --- | --- |
| f32 | 1024 | 1024 | 1 | 1.921 | 4.37 | 0 |
| f32 | 1024 | 1024 | 4 | 0.299 | 28.05 | 0 |
| f32 | 1024 | 1024 | 8 | 0.314 | 26.75 | 0 |
| f64 | 1024 | 1024 | 1 | 2.799 | 6.00 | 0 |
| f64 | 1024 | 1024 | 4 | 0.581 | 28.87 | 0 |
| f64 | 1024 | 1024 | 8 | 0.426 | 39.38 | 0 |
| f32 | 16384 | 1 | 1 | 0.019 | 7.02 | 0 |
| f32 | 16384 | 1 | 4 | 0.017 | 7.84 | 0 |

The `16384 x 1` row is one long column: below the per-task work threshold the
kernel keeps one task, so extra workers neither help nor add launch cost, and
the 4T and 8T rows stay at 7.84 GB/s. Pinning is required before these numbers
are used for a thread-scaling claim.

Comparison against the tenferro-rs host kernel that this crate replaces needs a
tenferro-rs build that depends on the released `strided-nn`; record both git
hashes and the tenferro bench command in the same table when it lands.
