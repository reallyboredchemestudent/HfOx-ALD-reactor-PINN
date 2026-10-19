# Frozen workstation outputs

Raw outputs of the three stages that need tools a CI runner does not have. They are inputs to the paper figures
(`code/export_figdata.py`), committed so that a clean clone can rebuild every figure's data
(`bash run_all.sh frozen figdata`). Everything else the figures use is recomputed from `data/` by the pipeline.

| Files | Stage | Needs | Command that writes them |
|---|---|---|---|
| `pinn_REALM_s<seed>_h<hold>.npz`, `pinn_REALM_s<seed>.log` | PINN ensembles, measured budget, seeds 0–9, holds none and each 55 °C wafer | JAX | `bash run_all.sh pinnm` |
| `aldfoam_gap1.0/`, `aldfoam_gap0.5/`, `aldfoam_dtcheck/` (`aldfoam_compare.csv`, `.txt`) | aldFoam (OpenFOAM-10) equal-dose pulses vs the steady solution | OpenFOAM-10 + aldFoam | `bash run_aldfoam.sh` |
| `si_results.txt` | StructuralIdentifiability.jl verdicts | Julia | `bash run_all.sh julia` |

Provenance: produced on the author's workstation (Linux, Python 3.10), 16 Oct 2026, before the
reproducibility review's code changes (see `REVIEW.md`). The PINN comparison built from these files reproduces the
workstation's `recal_compare_REALM.csv` exactly under the pinned environment (Python 3.13, NumPy 2.5.3), which is
checked every time the figure data are exported (`checks.csv`, figure 5). SHA-256 of every file used for the figures
is in `paper/figures/data/provenance.json`.

`aldfoam_gap1.0` case B used the default time step and is not converged in time (0.13 nm from case A);
`aldfoam_dtcheck` repeats it at a quarter of the step (0.027 nm). The figures use the converged run.

Regenerate these files only deliberately, on a machine with the tools above, then re-export the figure data and
read the diff of `paper/figures/data/checks.csv` before committing.
