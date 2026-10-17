# Reproducibility review

Standard: (a) a stranger reproduces the published numbers from a clean clone; (b) no edit can change a number
silently. Physics and validation claims are out of scope. Line numbers refer to commit `053e817`, the state reviewed.

## What was run

Fresh clone of `053e817`, Python 3.13.16, NumPy 2.5.3, SciPy 1.18.1, pandas 3.0.5, one thread per process.
PyPI was not reachable from the review machine, so JAX and Julia did not run.

| Run | Result |
|---|---|
| `bash run_all.sh check selfcheck synthpinn` | `check` PASSED (parameters equal the recorded reference to 2e-11). `selfcheck` and `synthpinn` failed (no JAX). `synthcompare` then ran on no PINN files, printed "0 of 0 PINN members", and the script exited 0. |
| All scripts that need only shipped files | `instrument`, `identifiability`, `rank1d`, `reduction/run1d_base`, `run1d_pool`, `run1d_pool2`, `run2d`, `gapresolved`, `aldfoam_params`, `data_real`, `run_real collapse`, `run_real fit --td 100-200` (with and without a prior `collapse`), `recalibrated fit` (two starts), `fit.py`, `make_B.py`, `compare.py` (FV only), `howo_sampled.py`: all ran. Outputs are frozen in `tests/golden/`. |
| README numbers reproduced | 315/315 sites, collapse 2.6 % and χ² 1.15 on 3 dof, gap-resolved 1e-8 and ≤ 0.34 nm, measured-budget χ² 448, install-check values. |
| README numbers not reproduced | χ² 21,124 (got 21,168.8 or 21,045.8, depending on whether `collapse` ran first; finding 1); FV coverage 0.98 (got 1.00 with the README command, 0.97 over all seven holds); 0.018 nm (no printed value matches). Each is a strict xfail in `tests/test_published_numbers.py`. |

## Findings, ranked by risk to a published result

Each finding gives: **where**; **failure**; **fix** (state); **proof** (the golden test showing the fix changed no
number); **enforced by** (what keeps it fixed).

### 1. Earlier outputs silently become inputs
- **Where:** `code/run_real.py:191-201` (`init_params` starts the fit from `results/collapse_summary.json` *if the file happens to exist*); `code/recalibrated.py:85-87` (same with `results/fit_<tag>.json`); `run_all.sh:30` (`--out "$CODE"`) with `code/compare.py:37` (globs PINN files in the working directory); `code/fit.py:69` and `code/make_B.py:11` (overwrite the shipped `synthA_rep0.npz` / `synthB_rep0.npz` that `check_install.py` reads); more outputs in `code/` (`fim.npz`, `reduction/run2d.npz`, `reduction/base_N800.npz`, `pinn_pre_*.npy`).
- **Failure:** this is why the published ideal-budget χ² does not reproduce. The same code, data and versions give χ² **21,168.8** when `collapse` ran earlier in the same results folder (the `run_all.sh` order), **21,045.8** when it did not (e.g. `bash run_all.sh load fit`), and the README says **21,124**. The 40-evaluation multistart ends somewhere different from each start, and nothing records which start was used. The collapse summary itself depends on `REGIME`, so that setting reaches the fit through a file too. The measured-budget fit is insensitive here (447.7 started from the ideal fit, 448.3 from the prior; README 448), but it has the same silent dependency. PINN members from an earlier run with other settings stay in `code/` and are counted in the next ensemble. `python fit.py` rewrites the reference data set in place: byte-identical with NumPy 2.5.3, but NumPy does not promise distribution streams across releases.
- **Fix (done):** `run_real.py` takes `--start collapse|prior` (default `collapse`, the `run_all.sh` order), exits with an explanation if the collapse file is missing, and writes the start used into `fit_<tag>.json`. `synthpinn`/`synthcompare` use `results/synth/`. `fit.py` and `make_B.py` write regenerated sets to `results/`. **Remaining:** the same explicit-start treatment for `recalibrated.py --start`; an `--out` defaulting to `results/` for the other scripts that write into `code/`; reconcile the README's 21,124 (re-run in the documented order and update the number, or document the start that gave it).
- **Proof:** `test_real_fit_ideal_budget_golden` (slow; load → collapse → fit, χ² 21,168.8 ± 0.5), `test_fit_refuses_a_hidden_start`, `test_shipped_files_unchanged` (SHA-256 of every shipped input), `test_synthetic_sets_regenerate`; strict xfail `test_readme_real_fit_ideal_chi2`.
- **Enforced by:** those tests in CI; `.gitignore` keeps `results/` out of commits.

### 2. A failed check does not fail
- **Where:** `code/check_install.py:21`, `run_all.sh:9` and `:21`, `code/compare.py:37-40`.
- **Failure:** `check_install.py` printed `DIFFERS` and exited 0. `run_all.sh` (`set -u` only) went on past a failed step and exited 0. From a clean clone, `selfcheck` and `synthpinn` failed, and `synthcompare` then reported FV numbers next to "0 of 0 PINN members" as if the comparison had run.
- **Fix (done):** `check_install.py` exits 1 on a mismatch. `run_all.sh` uses `set -euo pipefail` and stops at the first failed step, naming its log. `compare.py` exits 1 when a PINN tag finds no files.
- **Proof:** full fast suite unchanged; `compare_synthA_fv` stdout golden unchanged.
- **Enforced by:** CI job `golden` runs `bash run_all.sh check selfcheck` and fails on a non-zero exit; `shellcheck` in CI job `lint`.

### 3. Configuration read from the environment at import time
- **Where:** `code/fvmodel.py:16-17` (`ALD_NCYC` → `NCYC`, `THK`); `code/pinn.py:14` copies `THK` at import; `code/pinn.py:18-19` (`PHYS_SCALE`, `PDE_W`); `code/run_real.py:32` and `code/pinn_jax.py:43` must set `ALD_NCYC` before the first import; `code/run_real.py:36` and `pinn_jax.py:46` overwrite `FV.SITES` in place; `code/pinn_run.py:10,25,30` default to N55 260 / NSAT 50 / STAGES 10 / PDE_W 1 / PHYS_SCALE 1e-3, not the study configuration in `pinn_jax.py:17`.
- **Failure:** if anything imports `fvmodel` before the entry script sets `ALD_NCYC` (a notebook, a test, a reordered import), every thickness is scaled by 100/ncyc with no error. Lee et al. is 100 cycles, the default, so today's numbers are right by coincidence. A `PHYS_SCALE` or `PDE_W` left in the shell changes the PINN. `python pinn_run.py` does not reproduce the NumPy study unless five variables are set by hand.
- **Fix (done, partial):** `fvmodel.require_ncyc(ncyc, *modules)`, called by `run_real`, `recalibrated`, `process_window` and `pinn_jax` after import, raises on a stale cycle count or a stale `THK` copy. `pinn_run.py` prints its effective settings and documents the study values. The tests pin the environment in `tests/conftest.py` and check it took effect. **Remaining:** pass `ncyc`, sites and PINN scales as arguments instead of environment and module state. That is the full fix. It touches every model call site, so make it as its own change with the goldens as the check.
- **Proof:** `test_import_time_config_pinned`, `test_require_ncyc_catches_a_stale_import`; all goldens unchanged.
- **Enforced by:** ruff `TID251` bans `os.environ` and `sys.path` everywhere except an explicit allowlist in `pyproject.toml` (the remaining debt, which can only shrink).

### 4. Two README numbers come from commands that do not print them
- **Where:** `README.md:102` ("coverage FV 0.98", command `run_all.sh synthpinn`) and `README.md:94` ("0.018 nm", command `run1d_base.py && run2d.py`); `code/compare.py:15-16, 45`.
- **Failure:** `synthpinn` holds out wafers 0, 15 and 30 and prints FV coverage 1.00. All seven 55 °C hold-outs give 61/63 = 0.97. Both differ from 0.98. The cause is that `compare.py` draws the FV and PINN dose samples from one generator (`default_rng(11)`), so the FV interval for a wafer depends on which other wafers are listed and on how many PINN files exist. The 250 °C interval width is 0.87 nm in one command and 0.67 nm in the other. For 0.018 nm, the commands print ≤ 3.7e-5 (1D) and 1.4e-3 × 15 nm ≈ 0.021 nm (2D).
- **Fix (not applied; it changes a number by design):** give each method and hold-out its own stream, e.g. `np.random.default_rng([11, h, 0])` for FV and `[11, h, 1]` for the PINN, and raise `n` above 120 so the 5th/95th percentiles are stable. Then re-record `compare_synthA_fv` deliberately, and correct both README rows to the value and command that produce them.
- **Proof:** strict xfails `test_readme_fv_coverage_synthetic` and `test_readme_reduction_exact`, which turn red once reconciled.
- **Enforced by:** `tests/test_published_numbers.py` ties README rows to golden outputs.

### 5. Loose dependency ranges
- **Where:** `requirements.txt:1-5` (`>=` only); `docs/SETUP.md:41` installed packages by name, bypassing even those ranges.
- **Failure:** nothing fixed the versions a published number was made with. NumPy 2 already changed printed output (`np.float64(0.71)` where NumPy 1 prints `0.71`), which broke text comparison of logs. The install check passed on NumPy 2.5.3 / SciPy 1.18.1 (agreement 2e-11), but it guards only its own code path, and the README's commands had no other check. The README quick start installed whatever was newest.
- **Fix (done, partial):** `requirements-lock.txt` pins the verified versions; SETUP and the README install from it; the goldens cover far more than the install check. **Remaining:** pin JAX from the first machine where `selfcheck` passes; regenerate the lock with hashes (`uv pip compile --generate-hashes`).
- **Proof:** the whole golden suite, recorded with the versions in each golden's `_provenance` block.
- **Enforced by:** lockfile in CI job `golden`; CI job `latest` (allowed to fail, plus a weekly schedule) reports when a new upstream release moves a number.

### 6. The PINN gradient check the docstring cites did not exist
- **Where:** `code/pinn.py:3` ("checked against finite differences in test_pinn.py"); no such file.
- **Failure:** the JAX self-check validates JAX against the NumPy gradients. With nothing validating the NumPy gradients, the chain had no anchor.
- **Fix (done):** `tests/test_pinn.py::test_gradient_matches_finite_differences` checks all physics parameters and samples of every weight block. It passes: worst relative error 2e-5, on a gradient of 4e-3. It runs after a short training because at the initial weights the PDE term is about 7e23 and no difference quotient resolves it. The docstring now points to the test.
- **Proof:** the test itself, plus `test_loss_and_gradient_golden` and `test_short_training_golden`.
- **Enforced by:** CI job `golden`.

### 7. Star imports and `sys.path` edits
- **Where:** 16 `from M import *` (e.g. `code/compare.py:7`, `code/fit.py:4`, `code/howo_fv.py:4`, `code/run_breaks.py:4`, `code/reduction/run1d_pool.py:3`) supplying 104 names; `sys.path.insert` in 7 files (e.g. `code/run_real.py:34`, `code/pinn_jax.py:41`).
- **Failure:** names came through two levels of star import (`howo_fv.py` and `run_breaks.py` got `INS` from `fit`'s namespace). A new global in `fvmodel` or `fit` could silently shadow a script's name. With star imports, ruff cannot detect undefined names at all.
- **Fix (done):** explicit imports generated from ruff's F405 list; F821 confirms none is undefined. The `sys.path` edits are removed: they were redundant for scripts run from `code/`, and tests use `conftest.py`. 17 unused imports were removed.
- **Proof:** every stdout golden and the fast suite unchanged. Scripts without a stdout golden (`howo_sampled`, `run_breaks`, `recovery`, `pinn_run`) were run before and after: output identical, except the settings line `pinn_run.py` now prints first.
- **Enforced by:** ruff `F403`, `F405`, `F821`, `F401`, `TID251` (`sys.path`), committed in `pyproject.toml`.

### 8. Setup docs were a private hand-off, not instructions for a stranger
- **Where:** `docs/SETUP.md:1-8` ("the cloud workspace could not install…", "one zip to send back"), `:30-31` (unzip a download instead of clone), `:39-45` (Python 3.10, unpinned install), `:164`, `:194-202`; "send this output back" in `check_install.py:21`, `pinn_jax.py:268`, `run_real.py:10`; `run_all.sh:52` (`results_to_send.zip`).
- **Failure:** a new user cannot follow the setup from the repository. It assumes a zip they do not have, a CSV the repo now ships, and someone to send results to.
- **Fix (done):** SETUP clones the repo, uses Python 3.12 via uv on Ubuntu 22.04, installs the lockfile, and has an "If a check fails" section that the code's failure messages point to. The README quick start includes the clone and `pytest`.
- **Proof:** wording only; the fast suite is unchanged.
- **Enforced by:** CI job `golden` runs the README quick-start commands on a clean runner.

### 9. The data licence is unresolved while the data are redistributed
- **Where:** `data/README.md:26` ("check the article's licence … before redistributing"), with the CSV already in the repository.
- **Failure:** a reviewer cannot tell whether the dataset may be redistributed. JOSS and pyOpenSci both check this.
- **Fix (not applied; needs you to confirm):** *Communications Materials* publishes its articles under CC BY 4.0 (journal policy listing on Sherpa). The article page could not be fetched from the review machine to confirm this article's licence line. Once confirmed, replace the sentence with: "Transcribed from Supplementary Fig. 1 of Lee et al. (2026), published under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Changes: values read from the figure and tabulated; no other change." State in `LICENSE` or the README that the MIT licence covers code only.
- **Proof:** none needed; no code changes.
- **Enforced by:** `reuse lint` (REUSE specification) if you add SPDX headers / a `LICENSES/` folder; otherwise review.

### 10. Duplicated set-up in the 1D pool scripts
- **Where:** `code/reduction/run1d_pool.py:4-8` and `run1d_pool2.py:4-8`: pulse length set as `BASE['tau_p']` in one, the literal `2.0` in the other, and `2.0` again in the pulse shape and solver breaks.
- **Failure:** the two agree only while `BASE['tau_p'] == 2.0`; a change to `BASE` would silently desynchronise the two published tables.
- **Fix (done):** `reduce1d.pool_setup()` is the single definition; both scripts call it.
- **Proof:** `test_script_stdout[run1d_pool]`, `test_script_stdout[run1d_pool2]` unchanged.
- **Enforced by:** the stdout goldens; `pylint --disable=all --enable=duplicate-code code` if you want a duplication gate.

## Tolerances

Stated in each test file's docstring. In summary: shipped inputs exact (SHA-256); instrument rtol 1e-12; FV forward
rtol 1e-9; FV fit abs 1e-4 (observed 2e-11 against the recorded reference); PINN loss and gradient rtol 1e-9; short
PINN training abs 1e-4; script stdout to one unit in the last printed digit (integers exact); real FV fit chi2 abs 0.5.

## Status against "done"

| Criterion | State |
|---|---|
| Golden tests pass on a clean runner | On a fresh clone of the final commit: 30 passed, 3 strict xfails (the README numbers in findings 1 and 4), 1 skipped (JAX), 13.5 min on two cores. Not yet run on GitHub Actions; `.github/workflows/ci.yml` is in place. |
| Linters report zero findings under a committed config | `ruff check .` clean. `shellcheck` was not available here; CI runs it. |
| README alone reaches `INSTALL CHECK PASSED` | Quick start followed on a fresh clone with the lockfile versions: `INSTALL CHECK PASSED`. `run_all.sh` then stopped at `selfcheck` with a non-zero exit, because JAX was not installable here. |
| JAX golden | Not recorded. Run `python tests/record_golden.py --jax` once on a machine where `selfcheck` passes, commit `tests/golden/pinn_jax.json`, and pin JAX in the lockfile. |
