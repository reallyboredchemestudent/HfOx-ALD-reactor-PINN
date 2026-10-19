#!/usr/bin/env bash
# Runs every step that needs no hand-editing, in order; each step logs to results/logs/<step>.log.
#   bash run_all.sh                      all steps
#   bash run_all.sh load collapse        only the named steps
# Steps: check selfcheck synthpinn load collapse fit howo extrap pinn compare julia collect
#        recal pinnm comparem   (FV and PINN with the error budget measured on the saturated wafers: budget.py)
#        window                 (process window from the calibrated FV fit: run after recal)
#        fitregime              (only the regime fit of 'fit': what the figures need)
#        si                     (profile likelihood of U and the broken-assumption fits, into results/si/)
#        frozen                 (copy the workstation-only outputs in frozen/ into results/, never overwriting)
#        figdata                (export the paper figures' data: paper/figures/data/, with checks)
# Settings (environment): CSV=path/to.csv  REGIME=100-200  JOBS=<parallel fits>  SEEDS=0-9
set -euo pipefail      # stop at the first failed step: a later step must never run on a missing or failed input
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
  set +e; ( cd "$CODE" && "$@" ) 2>&1 | tee "$LOGS/$name.log"; local rc=${PIPESTATUS[0]}; set -e
  echo "=== $name finished in $(( SECONDS - t0 )) s (exit $rc)" | tee -a "$LOGS/$name.log"
  if [ "$rc" -ne 0 ]; then echo "=== STOPPED: step '$name' failed (exit $rc); log: $LOGS/$name.log" >&2; exit "$rc"; fi
}

for s in $STEPS; do
  case $s in
    check)     run check python check_install.py ;;
    selfcheck) run selfcheck python pinn_jax.py selfcheck ;;
    synthpinn) # JAX PINN on synthetic set A: must reproduce (or beat) the NumPy PINN numbers already in the doc
               mkdir -p "$RES/synth"
               run synthpinn python pinn_jax.py run --data synthA_rep0.npz --tag JA --seeds 0-4 --holds=-1,0,15,30 --jobs "$JOBS" --out "$RES/synth"
               run synthcompare python compare.py A 0,1,2,3,4 0,15,30 JA "$RES/synth" ;;
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
    fitregime) run fit_regime python run_real.py fit --td "$REGIME" --jobs "$JOBS" ;;
    si)        mkdir -p "$RES/si"
               run si_profile bash -c "cd '$RES/si' && python '$CODE/profile_lik.py' '$CODE/synthA_rep0.npz' U 2,4,6,8,10,12,14,16,18,20,24"
               for c in nominal second_order flow_spike pool_unpurged mediated surface_cvd; do
                 run "si_break_$c" bash -c "cd '$RES/si' && python '$CODE/run_breaks.py' $c"
               done ;;
    frozen)    # outputs that need JAX, OpenFOAM or Julia, produced on a workstation (see frozen/README.md)
               ( cd "$ROOT/frozen" && find . -type f ! -name README.md -print0 | while IFS= read -r -d '' f; do
                   mkdir -p "$RES/$(dirname "$f")"; [ -e "$RES/$f" ] || cp "$f" "$RES/$f"; done ) ;;
    figdata)   run figdata python export_figdata.py --results "$RES" ;;
    julia)     if command -v julia > /dev/null; then run julia julia identifiability.jl && cp "$CODE/si_results.txt" "$RES/"
               else echo "julia not on PATH: skipping (see SETUP.md section 3)"; fi ;;
    collect)   ( cd "$ROOT" && python -m zipfile -c results.zip results ) && echo "archived: $ROOT/results.zip" ;;
    *)         echo "unknown step $s"; exit 1 ;;
  esac
done
echo; echo "all steps finished without error: $STEPS"
