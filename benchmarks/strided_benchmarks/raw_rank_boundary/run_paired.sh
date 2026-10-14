#!/usr/bin/env bash
# Paired interleaved run: base,cand,base,cand,... for REPS repetitions.
# Usage: run_paired.sh <base-bin> <cand-bin> <outdir> <reps> <totals...>
set -euo pipefail
BASE=$1; CAND=$2; OUT=$3; REPS=$4; shift 4
mkdir -p "$OUT"
for r in $(seq 1 "$REPS"); do
  "$BASE" "$@" > "$OUT/base_$r.csv"
  "$CAND" "$@" > "$OUT/cand_$r.csv"
done
