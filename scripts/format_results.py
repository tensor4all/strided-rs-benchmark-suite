"""Parse benchmark log files and format results as a markdown table.

Usage:
    python scripts/format_results.py data/results/rust_*.log data/results/julia_*.log

Outputs a markdown table in the style of strided-rs/strided-opteinsum README.
"""

import re
import sys


def parse_log(filepath: str) -> dict[str, dict[str, float]]:
    """Parse a benchmark log and return {(instance, strategy, mode): median_ms}."""
    results: dict[tuple[str, str, str], float] = {}
    current_mode = None
    current_strategy = None

    # Detect engine from header
    engine = None

    with open(filepath) as f:
        for line in f:
            line = line.rstrip()

            # Detect engine
            if "strided-opteinsum" in line.lower() and engine is None:
                engine = "strided-opteinsum"
            elif "julia einsum" in line.lower() and engine is None:
                engine = "julia"

            # Parse strategy header
            # Rust: "Strategy: opt_flops"
            m = re.match(r"^Strategy:\s+(\w+)", line)
            if m:
                current_strategy = m.group(1)
                current_mode = "strided-opteinsum"
                continue

            # Julia: "Mode: omeinsum_path / Strategy: opt_flops"
            m = re.match(r"^Mode:\s+(\w+)\s*/\s*Strategy:\s+(\w+)", line)
            if m:
                current_mode = m.group(1)
                current_strategy = m.group(2)
                continue

            # Skip headers and separators
            if line.startswith("Instance") or line.startswith("-"):
                continue

            # Parse data line: name, tensors, log10flops, log2size, median_ms
            parts = line.split()
            if len(parts) >= 5 and current_mode and current_strategy:
                try:
                    name = parts[0]
                    median_ms = float(parts[-1])
                    key = (name, current_strategy, current_mode)
                    results[key] = median_ms
                except (ValueError, IndexError):
                    continue

    return results


def format_markdown_table(all_results: dict) -> str:
    """Format results as a markdown table grouped by strategy."""
    # Collect all instances and modes
    instances = sorted(set(name for name, _, _ in all_results.keys()))
    strategies = sorted(set(strat for _, strat, _ in all_results.keys()))
    modes = sorted(set(mode for _, _, mode in all_results.keys()))

    # Determine column order
    mode_order = []
    for m in [
        "strided-opteinsum",
        "omeinsum_path",
        "omeinsum_opt",
        "tensorops",
    ]:
        if m in modes:
            mode_order.append(m)
    for m in modes:
        if m not in mode_order:
            mode_order.append(m)

    mode_labels = {
        "strided-opteinsum": "Rust strided-opteinsum (ms)",
        "omeinsum_path": "Julia OMEinsum path (ms)",
        "omeinsum_opt": "Julia OMEinsum opt (ms)",
        "tensorops": "Julia TensorOps (ms)",
    }

    lines = []

    for strategy in strategies:
        lines.append(f"### Strategy: {strategy}")
        lines.append("")
        lines.append(
            "Median time (ms), single-threaded (OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1, JULIA_NUM_THREADS=1)."
        )
        lines.append("")

        # Header
        cols = [mode_labels.get(m, m) for m in mode_order]
        header = "| Instance | " + " | ".join(cols) + " |"
        separator = "|---|" + "|".join("---:" for _ in cols) + "|"
        lines.append(header)
        lines.append(separator)

        # Data rows
        for name in instances:
            row = [name]
            for mode in mode_order:
                key = (name, strategy, mode)
                if key in all_results:
                    row.append(f"{all_results[key]:.3f}")
                else:
                    row.append("-")
            lines.append("| " + " | ".join(row) + " |")

        lines.append("")

    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <logfile1> [logfile2] ...")
        sys.exit(1)

    all_results = {}
    for filepath in sys.argv[1:]:
        results = parse_log(filepath)
        all_results.update(results)

    print(format_markdown_table(all_results))


if __name__ == "__main__":
    main()
