#!/usr/bin/env bash
# Builds and runs the four aldFoam comparison cases, then compares them with the steady-exposure solution.
#   bash run_aldfoam.sh            (cases in ~/ald_cases/A..D, each a copy of ~/aldFoam/examples/aldFoam)
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"; CASES="${CASES:-$HOME/ald_cases}"; OUT="$ROOT/results"; mkdir -p "$OUT/logs"
for c in A B C D; do [ -d "$CASES/$c" ] || cp -r "$HOME/aldFoam/examples/aldFoam" "$CASES/$c"; done
python "$ROOT/code/make_aldfoam_cases.py" "$CASES" || exit 1
for c in A B C D; do
  echo "=== case $c"; ( cd "$CASES/$c" && blockMesh > log.blockMesh 2>&1 && aldFoam > log.aldFoam 2>&1 \
     && postProcess -func writeCellCentres -latestTime > log.cellCentres 2>&1 ) \
     && echo "    done: $(ls -d "$CASES/$c"/[1-9]* 2>/dev/null | tail -1)" \
     || { echo "    FAILED: see $CASES/$c/log.*"; tail -n 20 "$CASES/$c"/log.* ; }
for f in "$CASES/$c"/log.*; do cp "$f" "$OUT/logs/aldfoam_${c}_$(basename "$f")"; done
done
cd "$ROOT/code"
python compare_aldfoam.py "$CASES/A" "$CASES/B" "$CASES/C" --h 1.0 --patch substrate --labels A,B,C --out "$OUT/aldfoam_gap1.0" 2>&1 | tee "$OUT/logs/aldfoam_compare_gap1.0.log"
python compare_aldfoam.py "$CASES/D" --h 0.5 --patch substrate --labels D --out "$OUT/aldfoam_gap0.5" 2>&1 | tee "$OUT/logs/aldfoam_compare_gap0.5.log"
