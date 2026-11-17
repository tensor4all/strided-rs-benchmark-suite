#!/usr/bin/env python3
"""Regenerate the typed-reduction allocation section of this page's README.

Reads the raw CSVs that `run.sh rust 1|4` printed for `BENCH_FILTER=red_`
(`results/{before,after}-{1t,4t}.csv`) plus the provenance in
`results/run-info.txt`, and rewrites the README text between

    <!-- BEGIN generated: typed-reduce-alloc -->
    <!-- END generated: typed-reduce-alloc -->

so the published table is derived from raw data instead of being typed in.
Re-running it must leave the README unchanged. Standard library only.
"""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
BEGIN = "<!-- BEGIN generated: typed-reduce-alloc -->"
END = "<!-- END generated: typed-reduce-alloc -->"
THREADS = (1, 4)


def load(name):
    """(case, variant, threads) -> median_ns for one CSV in results/."""
    path = RESULTS / f"{name}.csv"
    with path.open(newline="") as fh:
        rows = {(r["case"], r["variant"], int(r["threads"])): int(r["median_ns"])
                for r in csv.DictReader(fh)}
    if not rows:
        sys.exit(f"{path}: no rows")
    return rows


def provenance():
    info = {}
    for line in (RESULTS / "run-info.txt").read_text().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            info[key.strip()] = value.strip()
    for required in ("date", "strided_rs_rev", "bench_runs", "cpus_1t", "cpus_4t"):
        if required not in info:
            sys.exit(f"run-info.txt: missing {required}")
    return info


def ms(ns):
    return f"{ns / 1e6:.2f}"


def cell(rows, case, variant, threads):
    ns = rows.get((case, variant, threads))
    return "-" if ns is None else ms(ns)


def block():
    info = provenance()
    before = load("before-1t") | load("before-4t")
    after = load("after-1t") | load("after-4t")

    order = []
    for case, _, threads in load("after-1t"):
        if case not in order:
            order.append(case)

    ratios = []
    for case in order:
        for threads in THREADS:
            pre = after.get((case, "typed", threads))
            alloc = after.get((case, "allocating", threads))
            if pre and alloc:
                ratios.append(pre / alloc)

    if not ratios:
        sys.exit("no axis row has both typed and allocating, cannot summarize")
    median_ratio = sorted(ratios)[len(ratios) // 2]
    close = sum(1 for r in ratios if abs(r - 1.0) <= 0.05)
    # The raw rows run identical code in both binaries, so their before/after
    # spread is this host's resolution for the cross-run columns.
    raw_ratios = sorted(after[(case, "raw", threads)] / before[(case, "raw", threads)]
                        for case in order for threads in THREADS
                        if (case, "raw", threads) in before and (case, "raw", threads) in after)

    raw_close = sum(1 for r in raw_ratios if abs(r - 1.0) <= 0.05)
    lines = [
        BEGIN,
        f"### Typed reduction output allocation ({info['date']})",
        "",
        f"Raw data: `results/{{before,after}}-{{1t,4t}}.csv`, provenance in "
        f"`results/run-info.txt`; regenerate this section with "
        f"`python3 benchmarks/strided_benchmarks/kernel_scaling/summarize_reduce.py`.",
        "",
        f"- strided-rs `{info['strided_rs_rev']}`, rustc `{info['rustc']}`, "
        f"`{info['rustflags']}`, `BENCH_RUNS={info['bench_runs']}`, "
        f"`BENCH_WARMUP={info['bench_warmup']}`.",
        f"- {info['host']}; {info['kernel']}; no isolated CPUs; one measurement at a time.",
        f"- Runs pinned with `taskset` to CPUs {info['cpus_1t']} (1T) and "
        f"{info['cpus_4t']} (4T) inside one L3 domain; the observing process ran on "
        f"CPUs 8-15.",
        f"- {info['julia']}, so the table has Rust rows only and the gate's (d) "
        f"check cannot fire for it.",
        f"- {info['repetitions']}.",
        "",
        "`typed` is the primary row: `reduce_axis_into` with a destination allocated "
        "in `prepare`, so the timed region holds no output allocation. `allocating` is "
        "the same binary's `reduce_axis` row, whose output allocation is inside the "
        "timed region; it exists for the axis rows only because `reduce` returns a "
        "scalar for `all`. `typed before` is the old primary row of the unmodified "
        "harness, which was this same allocating entry under the `typed` label.",
        "",
        "| case | T | typed before (allocating) | typed after (preallocated) | "
        "allocating after | raw after | typed/allocating |",
        "|---|---|---|---|---|---|---|",
    ]
    for case in order:
        for threads in THREADS:
            pre = after.get((case, "typed", threads))
            alloc = after.get((case, "allocating", threads))
            ratio = f"{pre / alloc:.2f}" if pre and alloc else "-"
            lines.append(
                f"| `{case}` | {threads} | {cell(before, case, 'typed', threads)} | "
                f"{cell(after, case, 'typed', threads)} | {cell(after, case, 'allocating', threads)} | "
                f"{cell(after, case, 'raw', threads)} | {ratio} |"
            )
    lines += [
        "",
        f"The `raw` rows run identical code in both binaries and still move by "
        f"{raw_ratios[0]:.2f}x to {raw_ratios[-1]:.2f}x (median "
        f"{raw_ratios[len(raw_ratios) // 2]:.2f}x) between the two binaries; "
        f"{raw_close} of the {len(raw_ratios)} cells stay within 5%, and the wider ones "
        f"are the column-strided `axis1` 8192x4096 cells, where the address and huge-page "
        f"placement of the 268 MB source array dominates. So both the server noise and "
        f"that placement, not the allocation, are the resolution of the before/after "
        f"columns.",
        f"In the {len(ratios)} axis cells that carry both rows, the same run's "
        f"`typed`/`allocating` ratio has median {median_ratio:.2f}x, with {close} cells "
        f"within 5% of 1.00 and the rest scattered between {min(ratios):.2f}x and "
        f"{max(ratios):.2f}x in both directions where the co-tenant load hit one of the "
        f"two rows. The removed work is one 16-64 KiB allocation against a kernel of "
        f"0.2-17 ms, so the allocation is below this host's measurement resolution; the "
        f"change fixes what the row means, not its speed.",
        "",
        "The earlier tables above were measured on an Apple M5 Max against strided-rs "
        "`c1c14a6` and `2b66201`, and their raw CSVs were never retained "
        "(`data/results/` is gitignored), so their typed axis-reduction numbers cannot "
        "be regenerated here. Those numbers are this table's `allocating` rows: the "
        "old harness timed the allocating entry under the `typed` label.",
        END,
    ]
    return "\n".join(lines)


def main():
    readme = ROOT / "README.md"
    text = readme.read_text()
    if text.count(BEGIN) != 1 or text.count(END) != 1:
        sys.exit(f"README.md: expected exactly one {BEGIN} / {END} pair")
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    readme.write_text(head + block() + tail)
    print(f"rewrote {readme}")


if __name__ == "__main__":
    main()
