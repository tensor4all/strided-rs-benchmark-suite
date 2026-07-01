#!/usr/bin/env bash
set -euo pipefail

HPTT_DIR="${HPTT_DIR:-../hptt}"
CXX="${CXX:-/opt/homebrew/bin/g++-15}"
OUT="${OUT:-target/hptt_compare}"

mkdir -p "$(dirname "$OUT")"

"$CXX" \
  -O3 \
  -DNDEBUG \
  -std=gnu++11 \
  -fopenmp \
  -mcpu=native \
  -I"$HPTT_DIR/include" \
  micro_bench/hptt_compare.cpp \
  "$HPTT_DIR/src/hptt.cpp" \
  "$HPTT_DIR/src/plan.cpp" \
  "$HPTT_DIR/src/transpose.cpp" \
  "$HPTT_DIR/src/utils.cpp" \
  -o "$OUT"

"$OUT"

