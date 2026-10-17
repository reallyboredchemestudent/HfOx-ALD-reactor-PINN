"""Process window: the smallest TDMA-Hf dose that gives full coverage, against deposition temperature.

From the FV model calibrated with the measured error budget (recalibrated.py fit). Dose is a multiplier on the
55 C source at the 0.2 s pulse; it converts to a source temperature at 0.2 s (supplier vapour-pressure curve) and to a
pulse length at the 55 C source (if dose = vapour pressure x pulse time, which this dataset does not test).

    python process_window.py [--draws 60] [--jobs 2] [--threshold 0.98]
        -> results/process_window.csv, process_window.json

Two answers per temperature, because they rest on different evidence:
  sites   worst of the 9 measured sites reaches the threshold (the model inside its data)
  wafer   worst point of the whole 100 mm wafer reaches it (the model beyond its sites: the edges and the far rim
          are 2 cm outside the measured pattern, and the inlet profile is only constrained for |x| <= 3 cm)
Uncertainty: parameter draws from the Laplace covariance of the fit (5-95 %); the setpoint adds a margin for the
run-to-run dose scatter (one-sided 95 %, SIG_O).
"""
import os, sys, json, argparse, csv
import numpy as np, multiprocessing
ap = argparse.ArgumentParser()
ap.add_argument('--npz', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'real_Y.npz'))
ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results'))
ap.add_argument('--fit', default=None, help='default: <out>/recal_fit_td100-200.json')
ap.add_argument('--draws', type=int, default=60); ap.add_argument('--jobs', type=int, default=2)
ap.add_argument('--threshold', type=float, default=0.98); ap.add_argument('--td', default='100:250:15')
A = ap.parse_args()
Z = np.load(A.npz, allow_pickle=True); os.environ['ALD_NCYC'] = str(float(Z['ncyc']))
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(k, '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fvmodel as FV
FV.SITES[:] = Z['sites']
from fit import SIG_O, LB, UB
fit = json.load(open(A.fit or os.path.join(A.out, 'recal_fit_td100-200.json')))
P0 = np.array([fit['params'][n] for n in FV.NAMES]); SD = np.array([fit['sd'][n] for n in FV.NAMES])
COV = np.array(fit['corr'])*np.outer(SD, SD)
a, b, s = map(float, A.td.split(':')); TDS = np.arange(a, b + 0.1, s)
WAFER_R, WAFER_C = 5.0, 7.5                      # 100 mm wafer centred at y = 7.5 cm

def worst(p, td, logm, grid, g, wafer):
    lam, zb, l, gpc = FV.groups(p, td, 55.0, logm)
    e, *_ = FV.solve(grid, lam, zb, l, g)
    th = 1 - np.exp(-e)
    return (1 - np.exp(-(grid.P@e))).min(), th[wafer].min()

def required(p):
    """log dose multiplier at which the sites and the whole wafer reach the threshold, for each temperature."""
    grid = FV.Grid(p[4]); g = FV.inlet_profile(grid.xi, p)
    X, Yc = np.meshgrid(grid.xc, grid.yc); wafer = (X**2 + (Yc - WAFER_C)**2 <= WAFER_R**2).ravel()
    out = []
    for td in TDS:
        res = []
        for k in (0, 1):                             # 0 = sites, 1 = whole wafer
            lo, hi = np.log(0.2), np.log(20.0)
            if worst(p, td, hi, grid, g, wafer)[k] < A.threshold: res.append(np.inf); continue
            for _ in range(14):
                mid = 0.5*(lo + hi)
                if worst(p, td, mid, grid, g, wafer)[k] >= A.threshold: hi = mid
                else: lo = mid
            res.append(hi)
        out.append(res)
    return np.array(out)

def draw(seed):
    rng = np.random.default_rng(seed)
    p = np.clip(rng.multivariate_normal(P0, COV), LB + 1e-6, UB - 1e-6)
    return required(p)

def tp_equiv(m):
    """Source temperature (C) whose vapour pressure is m times that at 55 C (supplier curve log10 p = 9.99 - 3377/T)."""
    return 1/(1/(55 + 273.15) - np.log10(m)/3377) - 273.15

if __name__ == '__main__':
    best = required(P0)
    with multiprocessing.get_context('fork').Pool(A.jobs) as pool: D = np.array(pool.map(draw, range(A.draws)))
    margin = 1.645*SIG_O
    rows = []
    for i, td in enumerate(TDS):
        r = dict(Td=td)
        for k, name in ((0, 'sites'), (1, 'wafer')):
            lm = D[:, i, k]; ok = np.isfinite(lm)
            q5, q50, q95 = np.percentile(lm[ok], [5, 50, 95]) if ok.any() else (np.nan,)*3
            r.update({f'{name}_best': float(np.exp(best[i, k])), f'{name}_p5': float(np.exp(q5)), f'{name}_p50': float(np.exp(q50)),
                      f'{name}_p95': float(np.exp(q95)), f'{name}_setpoint': float(np.exp(q95 + margin)),
                      f'{name}_setpoint_Tp_C': float(tp_equiv(np.exp(q95 + margin))), f'{name}_draws_unreached': int((~ok).sum())})
        r['tier'] = 'fitted' if td <= 200 else 'extrapolated (225 and 250 C held out and predicted)'
        rows.append(r)
        print(f"{td:5.0f} C  sites: x{r['sites_best']:.2f} (5-95% x{r['sites_p5']:.2f}-{r['sites_p95']:.2f}), setpoint x{r['sites_setpoint']:.2f} = "
              f"{r['sites_setpoint_Tp_C']:.1f} C source | wafer: x{r['wafer_best']:.2f} (x{r['wafer_p5']:.2f}-{r['wafer_p95']:.2f}), "
              f"setpoint x{r['wafer_setpoint']:.2f} = {r['wafer_setpoint_Tp_C']:.1f} C source", flush=True)
    with open(os.path.join(A.out, 'process_window.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    json.dump(dict(threshold=A.threshold, draws=A.draws, run_dose_margin=float(np.exp(margin)), rows=rows,
                   source_55C_multipliers={str(tp): float(10**(-3377/(tp + 273.15) + 3377/(55 + 273.15))) for tp in (55, 60, 65, 70, 75)}),
              open(os.path.join(A.out, 'process_window.json'), 'w'), indent=1)
