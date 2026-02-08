#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Benchmark runner for strided-rs-benchmark-suite
#
# Enforces single-threaded execution for fair comparison:
#   - OMP_NUM_THREADS=1   (OpenBLAS internal threading)
#   - RAYON_NUM_THREADS=1  (Rust rayon parallelism)
#   - JULIA_NUM_THREADS=1  (Julia task parallelism)
# ---------------------------------------------------------------------------

export OMP_NUM_THREADS=1
export RAYON_NUM_THREADS=1
export JULIA_NUM_THREADS=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$PROJECT_DIR/data/results"

mkdir -p "$RESULTS_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================"
echo " strided-rs benchmark suite"
echo "============================================"
echo "Project dir:  $PROJECT_DIR"
echo "Results dir:  $RESULTS_DIR"
echo "Timestamp:    $TIMESTAMP"
echo ""
echo "Threading policy (single-threaded):"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "  RAYON_NUM_THREADS=$RAYON_NUM_THREADS"
echo "  JULIA_NUM_THREADS=$JULIA_NUM_THREADS"
echo ""

# ---------------------------------------------------------------------------
# Rust benchmark (strided-opteinsum)
# ---------------------------------------------------------------------------
echo "============================================"
echo " [1/2] Rust: strided-opteinsum"
echo "============================================"

RUST_LOG="$RESULTS_DIR/rust_${TIMESTAMP}.log"

echo "Building Rust (release)..."
cargo build --release --manifest-path="$PROJECT_DIR/Cargo.toml" 2>&1

echo "Running Rust benchmark..."
cargo run --release --manifest-path="$PROJECT_DIR/Cargo.toml" 2>&1 | tee "$RUST_LOG"

echo ""
echo "Rust results saved to: $RUST_LOG"
echo ""

# ---------------------------------------------------------------------------
# Julia benchmark (OMEinsum.jl + TensorOperations.jl)
# ---------------------------------------------------------------------------
echo "============================================"
echo " [2/2] Julia: OMEinsum.jl + TensorOperations.jl"
echo "============================================"

JULIA_LOG="$RESULTS_DIR/julia_${TIMESTAMP}.log"

echo "Running Julia benchmark..."
julia --project="$PROJECT_DIR" "$PROJECT_DIR/src/main.jl" 2>&1 | tee "$JULIA_LOG"

echo ""
echo "Julia results saved to: $JULIA_LOG"
echo ""

# ---------------------------------------------------------------------------
# Format results as markdown table
# ---------------------------------------------------------------------------
MARKDOWN_OUT="$RESULTS_DIR/results_${TIMESTAMP}.md"

echo "Formatting results as markdown..."
uv run python "$PROJECT_DIR/scripts/format_results.py" "$RUST_LOG" "$JULIA_LOG" | tee "$MARKDOWN_OUT"

echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "============================================"
echo " Benchmark complete"
echo "============================================"
echo "Results:"
echo "  Rust log:     $RUST_LOG"
echo "  Julia log:    $JULIA_LOG"
echo "  Markdown:     $MARKDOWN_OUT"
