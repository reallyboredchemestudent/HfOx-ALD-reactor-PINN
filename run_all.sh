#!/usr/bin/env bash
# Runs every step that needs no hand-editing, in order; each step logs to results/logs/<step>.log.
#   bash run_all.sh                      all steps
#   bash run_all.sh load collapse        only the named steps
# Steps: check selfcheck synthpinn load collapse fit howo extrap pinn compare julia collect
#        recal pinnm comparem   (FV and PINN with the error budget measured on the saturated wafers: budget.py)
#        window                 (process window from the calibrated FV fit: run after recal)
# Settings (environment): CSV=path/to.csv  REGIME=100-200  JOBS=<parallel fits>  SEEDS=0-9
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"; CODE="$ROOT/code"; RES="$ROOT/results"; LOGS="$RES/logs"; mkdir -p "$LOGS"
CSV="${CSV:-$ROOT/data/lee2026_hfox_ald_dataset.csv}"
REGIME="${REGIME:-100-200}"     # deposition temperatures used for fitting; revisit after reading results/collapse_summary.txt
JOBS="${JOBS:-$(( $(nproc) > 1 ? $(nproc) / 2 : 1 ))}"
SEEDS="${SEEDS:-0-9}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
STEPS="${*:-check selfcheck synthpinn load collapse fit howo extrap pinn compare julia collect}"

run() {   # run <log name> <command...>   (inside code/, output to screen and log)
  local name=$1; shift
  echo; echo "=== $name: $*"; local t0=$SECONDS
  ( cd "$CODE" && "$@" ) 2>&1 | tee "$LOGS/$name.log"; local rc=${PIPESTATUS[0]}
  echo "=== $name finished in $(( SECONDS - t0 )) s (exit $rc)" | tee -a "$LOGS/$name.log"; return $rc
}

for s in $STEPS; do
  case $s in
    check)     run check python check_install.py ;;
    selfcheck) run selfcheck python pinn_jax.py selfcheck ;;
    synthpinn) # JAX PINN on synthetic set A: must reproduce (or beat) the NumPy PINN numbers already in the doc
               run synthpinn python pinn_jax.py run --data synthA_rep0.npz --tag JA --seeds 0-4 --holds=-1,0,15,30 --jobs "$JOBS" --out "$CODE"
               run synthcompare python compare.py A 0,1,2,3,4 0,15,30 JA ;;
    load)      [ -f "$CSV" ] || { echo "dataset not found at $CSV (set CSV=...)"; exit 1; }
               run load python data_real.py "$CSV"
               echo ">>> read results/load_report.txt before going on (columns, units, flow direction, site map, cycles)" ;;
    collapse)  run collapse python run_real.py collapse --regime "$REGIME" ;;
    fit)       run fit_all python run_real.py fit --jobs "$JOBS"
               run fit_regime python run_real.py fit --td "$REGIME" --jobs "$JOBS" ;;
    howo)      run howo python run_real.py howo --td "$REGIME" --holds 55 --jobs "$JOBS" ;;
    extrap)    run extrap_td python run_real.py extrap --train-td "$REGIME" --jobs "$JOBS"
               run extrap_dose python run_real.py extrap --train-td "$REGIME" --train-tp 60-75 --jobs "$JOBS" ;;
    pinn)      run pinn python pinn_jax.py run --data "$RES/real_Y.npz" --tag REAL --seeds "$SEEDS" --holds=-1,55 --td "$REGIME" --jobs "$JOBS" --out "$RES" ;;
    compare)   run compare python run_real.py compare --tag REAL --seeds "$SEEDS" ;;
    recal)     run recal_fit python recalibrated.py fit --td "$REGIME"
               run recal_howo python recalibrated.py howo --td "$REGIME" --jobs "$JOBS"
               run recal_extrap python recalibrated.py extrap --td "$REGIME" ;;
    pinnm)     run pinnm python pinn_jax.py run --data "$RES/real_Y.npz" --tag REALM --pre-tag REAL --budget measured --seeds "$SEEDS" --holds=-1,55 --td "$REGIME" --jobs "$JOBS" --out "$RES" ;;
    comparem)  run comparem python recalibrated.py compare --td "$REGIME" --tag REALM --seeds "$SEEDS" ;;
    window)    run window python process_window.py --jobs "$JOBS" ;;
    julia)     if command -v julia > /dev/null; then run julia julia identifiability.jl && cp "$CODE/si_results.txt" "$RES/"
               else echo "julia not on PATH: skipping (see SETUP.md section 3)"; fi ;;
    collect)   cp "$CODE"/pinn_JA_*.npz "$CODE"/compare_A_JA.npy "$RES/" 2>/dev/null
               ( cd "$ROOT" && python -m zipfile -c results_to_send.zip results ) && echo "send back: $ROOT/results_to_send.zip" ;;
    *)         echo "unknown step $s"; exit 1 ;;
  esac
done
