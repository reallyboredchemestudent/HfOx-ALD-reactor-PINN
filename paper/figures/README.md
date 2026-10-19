# Paper figures

One standalone `.tex` per figure (TikZ + pgfplots). Every number a figure draws is read from `data/`, which
`code/export_figdata.py` writes from the pipeline's raw outputs. No figure contains a typed-in value.

```bash
make -C paper/figures            # build every figure from the committed data/  ->  build/*.pdf
make -C paper/figures check      # fail if any figure's check failed, or a figure is wider than the text (6.5 in)
make -C paper/figures preview    # build/preview.pdf: every figure unscaled on the paper's page, draft captions
make -C paper/figures gray       # greyscale renders: every figure must read without colour
bash run_all.sh load collapse fitregime recal frozen comparem window si figdata   # regenerate data/ from data/ + frozen/
```

Needs TeX Live with pgfplots ≥ 1.18, siunitx, booktabs, latexmk and poppler-utils (`pdfinfo`, `pdftoppm`).

## Figures, claims and checks

Each figure supports one sentence of the paper (README "Main results"). The export recomputes that sentence's numbers
from the raw outputs and compares them with the text **at the precision the text prints them**
(`data/checks.csv`). A figure with a failed check still builds, with a red stamp naming the check, so a figure
can never silently disagree with the text. INFO rows are reported numbers with no stated criterion.

| File | Plan | Claim | Raw inputs (`results/`) | Checks |
|---|---|---|---|---|
| `fig1_reduction` | 1 | The pulsed reactor reduces exactly to a steady exposure equation | `aldfoam_{gap1.0,dtcheck}/aldfoam_compare.csv` (frozen); steady solutions recomputed with `gapresolved.py` | equal-dose pulses agree to the README's 0.027–0.036 nm; matched-mass agreement |
| `figS2_identifiability` | 2 (SI) | One dose fixes only the front position; a second dose separates reach and upstream length | `si/profile_U.npy`, `si_results.txt` (frozen); rank test computed in the export | SI.jl verdicts; rank deficiency with one dose, full rank with two |
| `fig3_massbalance` | 3 | The real wafers obey mass balance | `collapse_points.csv`, `collapse_summary.json` | line χ² and reach × GPC CV as printed |
| `fig4_errorbudget` | 4 | The textbook error budget fails; the measured one works | `per_wafer_td100-200.csv`, `fit_td100-200.json`, `recal_per_wafer_td100-200.csv`, `recal_fit_td100-200.json`, `real_Y.npz` | both budgets' χ²; per-wafer χ² sum to the fit's |
| `fig5_fv_vs_pinn` | 5 | The PINN fits better and predicts worse | `recal_compare_sites_REALM.csv`, `recal_compare_REALM.csv`, `pinn_REALM_s*_h*.npz` (frozen) | held-out wafer excluded from every member; RMSE ranges and interval coverage as printed; FV better on every wafer; PINN rows reproduce the workstation run |
| `fig6_extrapolation` | 6 | The fit predicts 225–250 °C wafers it never saw | `recal_extrap_td100-200.csv` | worst RMSE and coverage as printed |
| `fig7_window` | 7 | Domain of applicability and the process window | `process_window.csv`, `recal_fit_td100-200.json` (fields solved in the export) | dose multipliers and setpoint source temperatures as printed; setpoint covers the wafer |
| `fig8_failures` | 8 | Where the model fails | `recal_per_wafer_td100-200.csv`, `si/breakfit_*.npz` | control not flagged; 4 of 5 broken assumptions flagged |
| `figS9_pinn` | 9 (SI) | PINN diagnostics | `pinn_REALM_s*.log`, `pinn_REALM_s*_h-1.npz` (frozen), `recal_fit_td100-200.json` | NumPy PDE residual consistent with the saved one |

"Frozen" inputs need JAX, OpenFOAM or Julia and come from `frozen/` (see `frozen/README.md`).
`data/provenance.json` holds the SHA-256 of every input file the current `data/` was exported from.

## Conventions

- **Size.** Drawn at final size for an arXiv single-column article (10 pt, 1 in margins, `\textwidth` = 6.5 in) and
  included without scaling, so figure text is 8 pt (`\footnotesize`) in the body font. Change `\figW` and the axis
  widths if the geometry changes; `make check` enforces the width.
- **Notation.** `notation.tex` is shared with the paper (`\input{figures/notation}`); symbols follow `code/fvmodel.py`.
- **Colour.** FV blue, PINN vermillion, measured data black (Okabe–Ito; validated for colour-vision deficiency).
  Every series also has its own marker and dash pattern. Deposition temperature uses one hue, light to dark;
  residuals use a diverging map with a neutral midpoint. `make gray` renders every figure in greyscale.
- **Error bars** always read a file holding only that series (`*_fit.csv`, `fig5_sites_fv_100.csv`, …), never a
  filtered table.
- **Rasters** (model fields only, never interpolated measurements) are PNGs written by the export and placed with
  `\addplot graphics`; axes, labels and colour bars stay vector.
