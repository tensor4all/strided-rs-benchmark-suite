#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Benchmark runner for strided-rs-benchmark-suite
#
# Usage: ./scripts/run_all.sh [NUM_THREADS]
#
# NUM_THREADS (default: 1) controls:
#   - OMP_NUM_THREADS   (OpenBLAS internal threading)
#   - RAYON_NUM_THREADS  (Rust rayon parallelism)
#   - JULIA_NUM_THREADS  (Julia task parallelism)
#
# Rust benchmarks are run twice: with "faer" and "blas" backends.
# The blas backend requires OpenBLAS (brew install openblas on macOS).
#
# IMPORTANT: On Linux, set OPENBLAS_LIB_DIR and LD_LIBRARY_PATH to a
# recent OpenBLAS (>= 0.3.29). The system libopenblas-dev on Ubuntu 20.04
# is 0.3.8, which is ~2x slower than Julia's bundled 0.3.29 on AMD EPYC.
# Example:
#   export OPENBLAS_LIB_DIR=$HOME/opt/openblas-0.3.29/lib
#   export LD_LIBRARY_PATH=$HOME/opt/openblas-0.3.29/lib:$LD_LIBRARY_PATH
# ---------------------------------------------------------------------------

NUM_THREADS="${1:-1}"

export OMP_NUM_THREADS="$NUM_THREADS"
export RAYON_NUM_THREADS="$NUM_THREADS"
export JULIA_NUM_THREADS="$NUM_THREADS"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$PROJECT_DIR/data/results"

# ---------------------------------------------------------------------------
# Auto-detect custom OpenBLAS at $HOME/opt/openblas-0.3.29
# ---------------------------------------------------------------------------
CUSTOM_OPENBLAS="$HOME/opt/openblas-0.3.29/lib"
if [ -z "${OPENBLAS_LIB_DIR:-}" ] && [ -d "$CUSTOM_OPENBLAS" ]; then
    export OPENBLAS_LIB_DIR="$CUSTOM_OPENBLAS"
    export LD_LIBRARY_PATH="${CUSTOM_OPENBLAS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "Auto-detected OpenBLAS 0.3.29 at $CUSTOM_OPENBLAS"
fi

mkdir -p "$RESULTS_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================"
echo " strided-rs benchmark suite"
echo "============================================"
echo "Project dir:  $PROJECT_DIR"
echo "Results dir:  $RESULTS_DIR"
echo "Timestamp:    $TIMESTAMP"
echo ""
echo "Threading policy (threads=${NUM_THREADS}):"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "  RAYON_NUM_THREADS=$RAYON_NUM_THREADS"
echo "  JULIA_NUM_THREADS=$JULIA_NUM_THREADS"
echo ""

RUST_LOGS=()

# ---------------------------------------------------------------------------
# Rust benchmark (strided-opteinsum) — faer backend
# ---------------------------------------------------------------------------
echo "============================================"
echo " [1/3] Rust: strided-opteinsum (faer)"
echo "============================================"

RUST_FAER_LOG="$RESULTS_DIR/rust_faer_t${NUM_THREADS}_${TIMESTAMP}.log"

echo "Building Rust (release, faer)..."
cargo build --release --no-default-features --features faer --manifest-path="$PROJECT_DIR/Cargo.toml" 2>&1

echo "Running Rust benchmark (faer)..."
cargo run --release --no-default-features --features faer --bin strided-rs-benchmark-suite --manifest-path="$PROJECT_DIR/Cargo.toml" 2>&1 | tee "$RUST_FAER_LOG"

echo ""
echo "Rust (faer) results saved to: $RUST_FAER_LOG"
echo ""
RUST_LOGS+=("$RUST_FAER_LOG")

# ---------------------------------------------------------------------------
# Rust benchmark (strided-opteinsum) — blas (OpenBLAS) backend
# ---------------------------------------------------------------------------
echo "============================================"
echo " [2/3] Rust: strided-opteinsum (blas)"
echo "============================================"

RUST_BLAS_LOG="$RESULTS_DIR/rust_blas_t${NUM_THREADS}_${TIMESTAMP}.log"

echo "Building Rust (release, blas)..."
if cargo build --release --no-default-features --features blas --manifest-path="$PROJECT_DIR/Cargo.toml" 2>&1; then
    echo "Running Rust benchmark (blas)..."
    cargo run --release --no-default-features --features blas --bin strided-rs-benchmark-suite --manifest-path="$PROJECT_DIR/Cargo.toml" 2>&1 | tee "$RUST_BLAS_LOG"
    echo ""
    echo "Rust (blas) results saved to: $RUST_BLAS_LOG"
    RUST_LOGS+=("$RUST_BLAS_LOG")
else
    echo "WARNING: blas build failed (OpenBLAS not found?). Skipping blas benchmark."
    echo "  Install OpenBLAS: brew install openblas (macOS) / apt install libopenblas-dev (Ubuntu)"
fi
echo ""

# ---------------------------------------------------------------------------
# Julia benchmark (OMEinsum.jl + TensorOperations.jl)
# ---------------------------------------------------------------------------
echo "============================================"
echo " [3/3] Julia: OMEinsum.jl + TensorOperations.jl"
echo "============================================"

JULIA_LOG="$RESULTS_DIR/julia_t${NUM_THREADS}_${TIMESTAMP}.log"

echo "Running Julia benchmark..."
julia --startup-file=no --project="$PROJECT_DIR" "$PROJECT_DIR/src/main.jl" 2>&1 | tee "$JULIA_LOG"

echo ""
echo "Julia results saved to: $JULIA_LOG"
echo ""

# ---------------------------------------------------------------------------
# Format results as markdown table
# ---------------------------------------------------------------------------
MARKDOWN_OUT="$RESULTS_DIR/results_t${NUM_THREADS}_${TIMESTAMP}.md"

echo "Formatting results as markdown..."
uv run python "$PROJECT_DIR/scripts/format_results.py" "${RUST_LOGS[@]}" "$JULIA_LOG" | tee "$MARKDOWN_OUT"

echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "============================================"
echo " Benchmark complete"
echo "============================================"
echo "Results:"
for log in "${RUST_LOGS[@]}"; do
    echo "  Rust log:     $log"
done
echo "  Julia log:    $JULIA_LOG"
echo "  Markdown:     $MARKDOWN_OUT"
