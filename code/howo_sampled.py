"""Re-score hold-one-wafer-out intervals: propagate the unknown run dose offset by sampling through the
nonlinear model (instead of linearising), plus parameter uncertainty (linearised) and instrument noise."""
import numpy as np, sys, warnings
warnings.filterwarnings('ignore')
from fvmodel import CONDS, forward
import instrument as INS
from fit import SIG_O
tag = sys.argv[1]
R = np.load(f'howo_fv_{tag}.npy', allow_pickle=True)
rng = np.random.default_rng(7); out = []
for r in R:
    h = r['hold']; c = CONDS[h]; p = r['p']
    # parameter variance: total linearised variance minus its dose and instrument parts (recomputed at the fit)
    T_, S_, So_ = forward(p, np.zeros(35), sens=True, conds=[c])
    m0, s0, dm0 = INS.marginal(T_[0])
    var_dose_lin = (dm0*So_[0])**2*SIG_O**2
    var_par = np.maximum(r['sd']**2 - var_dose_lin - s0**2, 0)
    sims = []
    for k in range(200):
        o = np.zeros(35); o[h] = rng.normal(0, SIG_O)
        t = forward(p, o, conds=[c])[0]
        m, s, dm = INS.marginal(t)
        sims.append(m + rng.normal(0, 1, 9)*np.sqrt(s**2 + var_par))
    sims = np.array(sims)
    lo, hi = np.percentile(sims, 5, 0), np.percentile(sims, 95, 0); med = np.median(sims, 0)
    inside = (r['obs'] >= lo) & (r['obs'] <= hi)
    err = r['obs'] - med
    out.append(dict(hold=h, rmse=np.sqrt(np.mean(err**2)), cover=inside.mean(), width=np.mean(hi-lo)))
    print(f"{c}: RMSE {np.sqrt(np.mean(err**2)):.2f} nm, inside 90% {inside.mean():.2f}, mean width {np.mean(hi-lo):.2f} nm", flush=True)
allc = np.mean([o['cover'] for o in out]); print(f"overall coverage {allc:.2f}")
np.save(f'howo_fv_{tag}_sampled.npy', out, allow_pickle=True)
