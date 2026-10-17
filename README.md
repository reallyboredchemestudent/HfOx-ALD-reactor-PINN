# Time-free modelling of a pulsed ALD reactor: HfO₂ from TDMA-Hf / H₂O

A pulsed atomic layer deposition (ALD) reactor can be written as a **steady** problem: integrate the precursor's
gas transport over one pulse and purge, and the final surface coverage depends only on the **exposure** each point
receives. This repository derives that reduction, tests where it holds, and uses it to fit, cross-check and
extrapolate 315 real thickness measurements from Lee et al. (2026). Two inverse solvers are compared on equal
terms, a finite-volume adjoint solver and a physics-informed neural network (PINN), and the calibrated model gives a
process window.

**Main results**

- **The reduction is exact.** It holds whenever the gas reacts first order at the wall, the flow is steady and the
  purge clears the gas. An independent transient solver (aldFoam, OpenFOAM-10) agrees: equal-dose pulses from 0.05 to
  0.4 s give the same coverage to 0.04 nm of film.
- **Identifiability is settled exactly.** StructuralIdentifiability.jl proves one dose fixes only the front
  position; a second dose at the same temperature separates reach from the upstream reactive length.
- **The real wafers obey mass balance.** On Lee et al.'s 55 °C-source wafers, reach × growth per cycle is constant
  to 2.6% from 100 to 200 °C.
- **The textbook error budget fails on real data; the measured one works.** With noise measured on the saturated
  wafers, the FV inverse predicts held-out dose-limited wafers to 1.0–2.6 nm with 96% interval coverage, and
  predicts 225–250 °C wafers it never saw to ≤ 0.63 nm (94%).
- **The PINN fits better and predicts worse.** Under the same error budget it fits the training wafers better than
  the transport equation can (χ² ≈ 220 vs 448), then predicts held-out wafers worse (1.6–3.9 nm, 80%). Its soft
  physics term absorbs run-to-run structure.
- **Process window.** Covering the whole 100 mm wafer takes 1.2–1.7× the 55 °C-source dose; with margin, a 65 °C
  source at 100 °C falling to 60 °C at 250 °C. The requirement scales with growth per cycle.

## Every result and the command that reproduces it

Run from `code/` unless noted. Times are for an 8-core desktop.

| Result | Number | Command | Output |
| --- | --- | --- | --- |
| Reduction exact on the chamber plane (1D/2D, transient vs steady) | 0.018 nm | `cd code/reduction && python run1d_base.py && python run2d.py` | stdout |
| Reduction in a gap-resolved channel, and pressure-spike sensitivity | 10⁻⁸; ≤ 0.34 nm | `python gapresolved.py` | stdout |
| Same in aldFoam: equal-dose pulses 0.05/0.2/0.4 s | 0.027–0.036 nm | `bash run_aldfoam.sh` (repo root; OpenFOAM-10 + aldFoam) | `results/aldfoam_gap1.0/` |
| Structural identifiability, 4 models | one dose: front position only | `julia identifiability.jl` | `si_results.txt` |
| Fisher information and rank tests at Lee's design | sd per parameter | `python identifiability.py`, `python rank1d.py` | stdout |
| Profile likelihoods | upstream length 7.5–16 cm | `python profile_lik.py synthA_rep0.npz U 4,6,8,10,12,14,16,18` | `profile_U.npy` |
| Parameter recovery at the real design | RMS z 0.88–1.36 | `python recovery.py 0 20` | `recovery_0.npy` |
| Broken assumptions flagged by residuals | 4 of 5 flagged | `python run_breaks.py second_order` (and `flow_spike`, `pool_unpurged`, `mediated`, `surface_cvd`) | `breakfit_*.npz` |
| FV vs PINN on synthetic wafers | coverage FV 0.98 | `bash run_all.sh synthpinn` (root) | `results/logs/synthcompare.log` |
| Real-wafer loading and site map | 315/315 sites | `bash run_all.sh load` | `results/load_report.txt` |
| Mass-balance collapse on real wafers | 2.6%, χ² 1.2 / 3 | `bash run_all.sh collapse` | `results/collapse_real.png` |
| Ideal error budget on real wafers | χ² 21,124 / 225 | `bash run_all.sh fit howo extrap` | `results/fit_td100-200.json` |
| Measured error budget: fit, hold-out, extrapolation | χ² 448; 96%; 94% | `bash run_all.sh recal` | `results/recal_*` |
| PINN vs FV under the measured budget | 1.6–3.9 vs 1.0–2.6 nm | `bash run_all.sh pinnm comparem` | `results/recal_compare_REALM.csv` |
| Process window | ×1.2–1.7; 65→60 °C source | `bash run_all.sh window` | `results/process_window.csv` |

## Quick start

Linux or WSL2 (Ubuntu 22.04), Python ≥ 3.10:

```bash
python3 -m venv ~/ald-env && source ~/ald-env/bin/activate
pip install -r requirements.txt
bash run_all.sh check selfcheck        # FV fit reproduces the reference numbers; JAX gradients match NumPy
bash run_all.sh load collapse recal window
```

`check` must print `INSTALL CHECK PASSED` and `selfcheck` `SELF-CHECK PASSED`. Full setup, including Julia and
OpenFOAM-10 + aldFoam: [docs/SETUP.md](docs/SETUP.md).

## What is in here

| Path | What |
| --- | --- |
| `code/fvmodel.py`, `fit.py` | Steady exposure equation on the chamber plane (finite volume, Newton, forward and adjoint sensitivities) and the inverse with run-level dose offsets |
| `code/instrument.py`, `ellips.py`, `make_lookup.py` | Ellipsometry measurement model: what a single-layer Cauchy fit reports for a true film |
| `code/budget.py`, `recalibrated.py` | Error budget measured on the saturated wafers; FV fit, hold-out, extrapolation and FV-vs-PINN under it |
| `code/pinn.py`, `pinn_jax.py` | The same inverse PINN in NumPy (hand-derived gradients) and JAX (autodiff), self-checked against each other |
| `code/data_real.py`, `run_real.py` | Real-data loader with site-map and flow-direction checks; collapse, fits, hold-out, extrapolation |
| `code/process_window.py` | Dose for full coverage against deposition temperature, with uncertainty |
| `code/gapresolved.py`, `breaks.py`, `reduction/` | Checks of the reduction: gap-resolved channel, broken assumptions, transient vs steady |
| `code/identifiability.py`, `rank1d.py`, `profile_lik.py`, `identifiability.jl` | Identifiability: Fisher information, rank tests, profile likelihoods, StructuralIdentifiability.jl |
| `code/aldfoam_params.py`, `make_aldfoam_cases.py`, `compare_aldfoam.py`, `aldfoam/` | aldFoam comparison cases and their analysis |
| `code/*.npz`, `*.npy` | Synthetic datasets A and B and the reference FV hold-out results used by `compare.py` |
| `data/` | The 315-site dataset and its provenance |

## Data

`data/lee2026_hfox_ald_dataset.csv` holds thickness, refractive index and wet etch rate at 9 sites for 7
deposition temperatures × 5 precursor temperatures (100 cycles, 0.2/20 s pulse/purge). It was transcribed from
Supplementary Fig. 1 of Lee et al. (2026): each value read by eye and again by OCR, all 945 in agreement. See
[data/README.md](data/README.md). The original measurements belong to the authors; cite their paper when using the data.

## Limits

- The reduction assumes first-order wall reaction, steady flow and a complete purge. Second-order and
  precursor-mediated chemistry, unpurged pools and surface CVD break it, and the residuals flag them; a pressure
  spike during the pulse does not.
- The whole-wafer process window extrapolates 2 cm beyond the 9 measured sites, and the inlet profile is constrained
  only for |x| ≤ 3 cm. A 49-point map at 100 °C with a 60 °C and a 65 °C source would test it.
- Not modelled: the fixed thicker band on the two outlet-side rows of every saturated wafer (cause open), soft
  saturation at 100–150 °C, and run-to-run left/right asymmetry of dose-limited wafers.
- The pulse-time column of the process window assumes dose ∝ pulse time, which this dataset does not test.

## Citation

If you use this code, cite this repository (see `CITATION.cff`) and the data source:

M. Lee, D. S. Kim, …, S. J. Kim and J. Kim, "Machine learning-guided optimization of atomic layer deposition
process", *Communications Materials* (2026). https://www.nature.com/articles/s43246-026-01206-w

Related tools: aldFoam (Yanguas-Gil and Elam, https://github.com/aldsim/aldFoam);
StructuralIdentifiability.jl.

## License

Code: MIT (see `LICENSE`). Data: see `data/README.md`.
