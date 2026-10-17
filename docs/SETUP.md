# Running the blocked parts of the steady-exposure stage on your PC

The cloud workspace could not install JAX, Julia or OpenFOAM, and it does not have Lee et al.'s CSV. This package
runs those four pieces on your machine and collects everything into one zip to send back:

| Piece | What it settles | Script | Rough time |
|---|---|---|---|
| Install check | your machine reproduces the cloud FV fit exactly | `code/check_install.py` | 2–5 min |
| JAX PINN | a real autodiff PINN, checked against the NumPy one, then ensembles on real wafers | `code/pinn_jax.py` | self-check 1 min; ensembles 1–3 h |
| Real wafers | collapse plot, FV fit, per-wafer flags, hold-one-wafer-out, extrapolation, FV vs PINN | `code/data_real.py`, `code/run_real.py` | 30–60 min |
| StructuralIdentifiability.jl | exact answer on which groups one or two doses identify | `code/identifiability.jl` | 30 min install, minutes to run |
| aldFoam (OpenFOAM-10) | the time-free reduction inside an independent solver | `code/aldfoam_params.py`, `code/compare_aldfoam.py` | 1–2 h including setup |

Everything below runs inside WSL2 (Ubuntu 22.04) on Windows; on native Linux skip section 1.

---

## 1. WSL2 with Ubuntu 22.04

In PowerShell (as administrator):

```powershell
wsl --install -d Ubuntu-22.04
```

Restart, open "Ubuntu 22.04" from the Start menu and create a user. Then put the package in your Linux home (not under
`/mnt/c`: OpenFOAM and Python are several times slower on the Windows filesystem):

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip build-essential unzip git curl
cp /mnt/c/Users/<your Windows user>/Downloads/hfox-steady-exposure ~ && cd ~ && unzip hfox-steady-exposure && cd hfox_local
```

Ubuntu 22.04 is needed because aldFoam builds only against OpenFOAM-10, and openfoam.org packages OpenFOAM-10 for 20.04/22.04.

## 2. Python environment

```bash
python3 -m venv ~/ald-env && source ~/ald-env/bin/activate      # put the `source` line in ~/.bashrc too
pip install --upgrade pip
pip install numpy scipy pandas matplotlib "jax[cpu]"
python -c "import jax; jax.config.update('jax_enable_x64', True); print(jax.__version__, jax.devices())"
```

pip picks the newest JAX that supports Ubuntu 22.04's Python 3.10; any JAX from 0.4.20 on works with this code.
JAX runs on the CPU here on purpose: its AMD GPU (ROCm) support does not cover WSL reliably, and this problem
(a few thousand collocation points, a 100-unit network) is fastest as several single-threaded CPU processes in parallel.

Check the install, then the PINN:

```bash
cd ~/hfox-steady-exposure
bash run_all.sh check selfcheck
```

`check` must print `INSTALL CHECK PASSED` (your FV fit of synthetic set A equals the cloud one), and `selfcheck` must
print `SELF-CHECK PASSED` (the JAX loss and its gradient equal the NumPy implementation's, whose gradient was checked by
finite differences). The self-check also prints milliseconds per loss-and-gradient evaluation; one ensemble member
takes about 7,000 evaluations plus compilation.

## 3. Julia and StructuralIdentifiability.jl

```bash
curl -fsSL https://install.julialang.org | sh        # juliaup; accept the defaults, then open a new terminal
julia -e 'using Pkg; Pkg.add("StructuralIdentifiability")'   # 10-30 min, mostly precompilation
bash run_all.sh julia                                 # writes results/si_results.txt
```

Four models are assessed: plug flow; with axial dispersion; the exact plug-flow profile with reach and reactive length
written out (one dose); and the same with a second wafer at twice the dose (55 vs 65 °C source). The expectation from
the Fisher-information work is that one dose identifies only the front position, and two doses separate reach from the
reactive length. If Julia says otherwise, that overturns a result in the doc.

## 4. The dataset

Put the CSV at `data/lee2026_hfox_ald_dataset.csv` (or set `CSV=/path/to/file.csv`), then:

```bash
bash run_all.sh load
```

Read `results/load_report.txt` before going on. It lists which column it took for each role, the units it assumed,
the cycle count, the implied GPC per temperature, and how it placed the 9 sites in the model frame (y along the flow,
wafer centre at 7.5 cm). Three things to check against the paper:

* **Flow direction.** If the CSV gives coordinates, the loader matches them to the site map used in the synthetic study
  and picks the rotation closest to the downhill thickness direction of the 55 °C wafers. Confirm against the flow
  arrow in the paper's reactor or site figure; override with `python code/data_real.py <csv> --flow-angle <deg>`.
* **Site labels only (no coordinates).** Labels are assigned in order to the synthetic site map; if the paper's map
  differs, write `sites.json` (`{"1": [x_cm, y_cm], ...}` in the model frame) and pass `--sites sites.json`.
* **Column names.** If a role is missed, write `data_map.json` (format in the docstring of `data_real.py`) and pass `--map`.

If the report says the thickest film exceeds the instrument table, run `cd code && python make_lookup.py <nm>` once.
Rows outside the 7 × 5 design (for example a pulse-time series) go to `results/real_extra.csv`; send that too, since a
pulse-time series at fixed temperatures is a direct test of the "dose only" claim.

## 5. Real-wafer runs and PINN ensembles

```bash
bash run_all.sh collapse
```

Look at `results/collapse_real.png` and `results/collapse_summary.txt` first. The regime used for all fitting is
`REGIME=100-200` (where reach × GPC was constant in the earlier T7 numbers). If the real collapse holds over a
different range, rerun everything below with, for example, `REGIME=100-225 bash run_all.sh collapse fit ...`.

```bash
bash run_all.sh fit howo extrap pinn compare
```

* `fit` – FV adjoint inverse on all 35 wafers and on the regime, multistart, Laplace uncertainty, per-wafer χ² flags
  (`per_wafer_*.csv`, flag above 27.9 on 9 sites), group table per wafer (`groups_*.csv`).
* `howo` – refits without each 55 °C wafer in the regime and predicts it with a 90 % interval.
* `extrap` – fits on the regime and predicts the hotter wafers; fits on the 60–75 °C sources and predicts the 55 °C ones.
  This is the "what the model predicts beyond the wafers it saw" tier.
* `pinn` – JAX PINN, one member per seed (default 10), on all regime data and on each held-out 55 °C wafer.
* `compare` – FV vs PINN ensemble on the same held-out wafers (RMSE, 90 % coverage, interval width).

Optional but useful: `bash run_all.sh synthpinn` reruns the PINN on synthetic set A with JAX and compares it with the
FV results already in the doc (`results/logs/synthcompare.log`), so the JAX PINN is calibrated against the NumPy one
before anyone reads its real-wafer numbers.

Parallelism: `JOBS` defaults to half your logical cores; each fit or PINN member is single-threaded.

**Measured error budget (after the first real-wafer run).** The ideal noise model is far too tight on Lee's wafers. These
steps measure it on the saturated wafers (`code/budget.py`) and rerun FV and the PINN under it:

```bash
bash run_all.sh selfcheck recal pinnm comparem collect
```

`recal` = FV refit, hold-out and extrapolation (`recalibrated.py`); `pinnm` = PINN members under the same budget
(tag `REALM`, reusing the `REAL` pretraining); `comparem` = FV vs PINN on the held-out 55 °C wafers.

## 6. aldFoam (OpenFOAM-10)

Install OpenFOAM-10 from openfoam.org and build aldFoam (commands from the aldFoam README):

```bash
sudo sh -c "wget -O - https://dl.openfoam.org/gpg.key > /etc/apt/trusted.gpg.d/openfoam.asc"
sudo add-apt-repository http://dl.openfoam.org/ubuntu
sudo apt update && sudo apt install -y openfoam10
echo "source /opt/openfoam10/etc/bashrc" >> ~/.bashrc && source ~/.bashrc

git clone https://github.com/aldsim/aldFoam.git ~/aldFoam
cd ~/aldFoam/firstorder && wmake && cd ../dose && wmake && cd ../aldFoam && wmake
```

The README states the master branch works with OpenFOAM-10 and not with newer versions, so do not substitute 11 or 12.

**Cases.** `python code/aldfoam_params.py` prints and saves (`aldfoam/aldfoam_cases.json`) the inputs for four cases
that reproduce the gap-resolved check in the doc: TDMA-Hf at 175 °C between parallel plates 24 cm long, Poiseuille
flow, both plates reactive with aldFoam's ideal first-order model, reach 16 cm, front width 0.9 cm:

| Case | Gap | Mean speed | Pulse | Inlet precursor | Same molecules as A? |
|---|---|---|---|---|---|
| A | 1.0 cm | 1.79 m/s | 0.20 s | 1.99 Pa (14.9 mTorr) | reference |
| B | 1.0 cm | 1.79 m/s | 0.05 s | 7.97 Pa | yes |
| C | 1.0 cm | 1.79 m/s | 0.40 s | 1.00 Pa | yes |
| D | 0.5 cm | 3.58 m/s | 0.20 s | 1.99 Pa | yes, at half the gap |

β0 = 0.0243, D = 0.0214 m²/s, site area 0.278 nm² (one Hf per site per cycle at 1.30 Å/cycle), purge 1.5 s after the pulse.

**What I could not verify from the cloud:** GitHub's file browser was not reachable, so I have the README's build
steps and the paper's model, but not the names of the example case's files and keywords. For each case:

1. `cp -r ~/aldFoam/examples/aldFoam ~/ald_cases/A` (B, C, D likewise) and read the example's README/Allrun.
2. Replace `system/blockMeshDict` with `aldfoam/system/blockMeshDict_gap1.0` (D: `_gap0.5`). Patch names are
   `inlet`, `outlet`, `bottom`, `top`, `frontAndBack`; make the files in `0/` use the same names, give `bottom` and
   `top` the example's reactive-wall condition, and `frontAndBack` type `empty`.
3. Put the numbers from `aldfoam_cases.json` where the example defines them: diffusivity, β0, site area, molar mass
   and temperature (or thermal velocity), inlet partial pressure or concentration, dose time and ramp, inlet velocity.
4. In `system/controlDict`: `endTime` = pulse + 1.5 s, `writeFormat ascii;`, and make sure the final time is written.
5. Run in the example's order (typically `blockMesh`, the flow solution, then `aldFoam`), then
   `postProcess -func writeCellCentres -latestTime` (writes the face centres the comparison reads).

Then:

```bash
cd ~/hfox-steady-exposure/code
python compare_aldfoam.py ~/ald_cases/A ~/ald_cases/B ~/ald_cases/C --h 1.0 --out ../results/aldfoam_gap1.0
python compare_aldfoam.py ~/ald_cases/D --h 0.5 --out ../results/aldfoam_gap0.5
```

The script finds the coverage on the `bottom` patch by itself (it prints what it used and the alternatives; force it
with `--field`, `--key`, `--patch`). A pass looks like this:

* case vs case (B and C against A) below about 1e-3 in coverage: pulse length and strength do not matter at equal
  dose. This is the reduction tested inside someone else's solver; a larger difference is the most important thing to report.
* aldFoam vs this project's steady solution within a few 1e-3 at matched mass (aldFoam fixes the inlet concentration,
  the stand-in fixes the inlet flux, so the nominal-dose comparison can differ slightly more).
* the depth-averaged fit (the PINN's equation) within about 0.01 nm of film, as in the stand-in.

## 7. What to send back

```bash
bash run_all.sh collect
```

and send `results_to_send.zip`. It contains the load report, collapse plot and tables, FV fits and flags, hold-out
and extrapolation tables, PINN members and the comparison, `si_results.txt`, the aldFoam comparisons and every log.
If a step failed, its log in `results/logs/` is what I need.
