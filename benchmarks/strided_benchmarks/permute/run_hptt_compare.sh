#!/usr/bin/env bash
set -euo pipefail

HPTT_DIR="${HPTT_DIR:-../hptt}"
CXX="${CXX:-/opt/homebrew/bin/g++-15}"
OUT="${OUT:-target/hptt_compare}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$(dirname "$OUT")"

"$CXX" \
  -O3 \
  -DNDEBUG \
  -std=gnu++11 \
  -fopenmp \
  -mcpu=native \
  -I"$HPTT_DIR/include" \
  "$SCRIPT_DIR/hptt_compare.cpp" \
  "$HPTT_DIR/src/hptt.cpp" \
  "$HPTT_DIR/src/plan.cpp" \
  "$HPTT_DIR/src/transpose.cpp" \
  "$HPTT_DIR/src/utils.cpp" \
  -o "$OUT"

"$OUT"
