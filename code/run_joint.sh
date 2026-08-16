#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
OUT="${1:-$ROOT/results/joint_reproduced}"
CASES="$OUT/raw_cases"
CHECKS="$OUT/checks"
SUMMARY="$OUT/summary"
FIGURES="$OUT/figures"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-$OUT/.mplconfig}"
rm -rf "$OUT"
mkdir -p "$CASES" "$CHECKS" "$SUMMARY" "$FIGURES" "$MPLCONFIGDIR"
PY="${PYTHON:-python3}"
BENCH="$HERE/joint.py"
AGG="$HERE/aggregate.py"
for EF in cosine smoothstep; do
  for PF in gauss midpoint; do
    echo "[run] production family $EF/$PF"
    "$PY" "$BENCH" --worker-family-dir "$CASES" --electron-family "$EF" --photon-family "$PF"
  done
done
for L in 4 12; do
  for M in 1 4; do
    echo "[run] cutoff Nmax=5 check L=$L M=$M"
    "$PY" "$BENCH" --single-case-out "$CHECKS/cutoff_L${L}_M${M}_N5.npz" \
      --L "$L" --M "$M" --electron-family cosine --photon-family gauss \
      --cutoff 5 --case-rtol 2e-9 --case-atol 2e-11
    echo "[run] tightened-tolerance check L=$L M=$M"
    "$PY" "$BENCH" --single-case-out "$CHECKS/tight_L${L}_M${M}.npz" \
      --L "$L" --M "$M" --electron-family cosine --photon-family gauss \
      --cutoff 4 --case-rtol 5e-10 --case-atol 5e-12
  done
done
"$PY" "$AGG" --cases-dir "$CASES" --checks-dir "$CHECKS" --output-dir "$SUMMARY" --figure-dir "$FIGURES"
echo "[done] outputs: $OUT"
