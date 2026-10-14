# Predeclared paired protocol v3 — strided-rs issue #300 (raw rank-8 cliff)

Registered before the v3 run. v1 and v2 outcomes are reported unchanged; v3 only
re-scopes the case sets to the issue's own framing. No threshold or statistic
changes from v2 beyond the case sets below, and no v3 measurement existed when
this was written.

v1 primary gate: "every case with fused_rank > 8 must improve >= 1.3x" -> FAILED on
three fully contiguous layouts (fused_rank 1) where the baseline had no cliff and
instead reached the view kernels' contiguous leaf; that gate was unsatisfiable
there by construction.

v2 non-regression set: "fused_rank <= 8" -> FAILED on the same three contiguous
layouts (original rank 9/12/16), where `axpy_raw` is 1.28-1.31x slower than the
baseline's AVX-512-dispatched contiguous view leaf. That set was keyed on the
*fused* rank, which is 1 for a contiguous high-rank layout.

v3 case sets (the issue's own framing: original rank vs the rank-8 boundary):

- **Common path (non-regression), original rank <= 8.** These are the layouts
  that used the fused replay both before and after the change.
  Gate: median ratio <= 1.05 for every case whose baseline spread
  `(max-min)/median` is <= 0.05; noisier cases are reported as low-confidence and
  judged against the baseline's own observed range.
- **Cliff cases (primary), original rank > 8 and fused_rank > 8.** These are
  exactly the layouts for which the baseline left the fused replay and built a
  view plan. Gate: median improvement >= 1.3x on every such case.
- **Fused-small high-rank cases (reported, not gated): original rank > 8 with
  fused_rank <= 8.** The baseline reached a view-kernel leaf directly here (SIMD
  contiguous axpy, `copy_nonoverlapping`, or a planning path) rather than the
  fused replay, so both improvements and small regressions are expected and are
  reported case by case.

Everything else is as in v2: baseline `7bb8022`, candidate = branch under test,
harness `/tmp/raw300/harness.rs`, rustc 1.97.1 release, AMD EPYC 7713P,
`taskset -c 5`, `RAYON_NUM_THREADS=1`, 15 interleaved repetitions, median as the
primary statistic, allocation gates (0 per call for original rank <= 8, <= 2 for
original rank > 8 raw calls, <= 1 per `CopyPlan::execute`), and a secondary 4T
check (7 interleaved repetitions at 1M elements, `taskset -c 5-8`,
`RAYON_NUM_THREADS=4`, both binaries built with `--features parallel`).
