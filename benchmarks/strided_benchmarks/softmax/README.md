# Fused softmax (`strided-nn`)

One `softmax_axis0_f32` / `softmax_axis0_f64` call over contiguous column-major
`[axis_len, columns]` data, with caller scratch and an explicit `ExecContext`.
The timing covers the kernel call only: the host runtime's dtype dispatch,
pooled output and plan cache are outside this crate's contract.

```bash
cargo build --release --features parallel
THREADS=1 ./target/release/softmax
THREADS=4 ./target/release/softmax
THREADS=8 ./target/release/softmax
```

Run the thread counts one at a time, never two benchmark processes in parallel.
`THREADS` sets the budget handed to `ExecContext::run_on` on a pool of that
width; `AXIS_LENS`, `COLUMNS`, `RUNS` and `MAX_ELEMENTS` narrow the matrix.

The table reports median and inter-quartile range of one call, GB/s over
read+write traffic, and the allocations observed during the timed loop. The
steady-state allocation column is the point of the caller-scratch contract: a
warm kernel must show zero.

Measured on 2026-10-24 (macOS arm64, 18 cores, unpinned — no `taskset`
available), one call, `RUNS=10`:

| dtype | axis_len | columns | threads | median_ms | GB/s | allocations |
| --- | --- | --- | --- | --- | --- | --- |
| f32 | 1024 | 1024 | 1 | 1.435 | 5.84 | 0 |
| f32 | 1024 | 1024 | 4 | 0.297 | 28.20 | 0 |
| f32 | 1024 | 1024 | 8 | 0.315 | 26.67 | 0 |
| f64 | 1024 | 1024 | 1 | 2.423 | 6.93 | 0 |
| f64 | 1024 | 1024 | 4 | 0.572 | 29.36 | 0 |
| f64 | 1024 | 1024 | 8 | 0.458 | 36.67 | 0 |
| f32 | 16384 | 1 | 1 | 0.018 | 7.38 | 0 |
| f32 | 16384 | 1 | 4 | 0.026 | 5.13 | 0 |

The `16384 x 1` row is one long column: below the per-task work threshold the
kernel keeps one task, so extra workers add launch cost. Pinning is required
before these numbers are used for a thread-scaling claim.

Comparison against the tenferro-rs host kernel that this crate replaces needs a
tenferro-rs build that depends on the released `strided-nn`; record both git
hashes and the tenferro bench command in the same table when it lands.
