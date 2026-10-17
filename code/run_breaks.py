"""Generate break-case wafers (same noise draw as truth A replicate 0) and fit each with the steady FV inverse."""
import numpy as np, sys, warnings
warnings.filterwarnings('ignore')
from fit import *
import breaks
case = sys.argv[1]
rng = np.random.default_rng(0)
off = rng.normal(0, SIG_O, len(CONDS))                 # same draws as make_data(seed 0)
T = breaks.truth_maps(TRUTH_A, case, off) if case != 'nominal_transient' else breaks.truth_maps(TRUTH_A, 'nominal', off)
Y = np.array([INS.generate(T[r], INS.sample_state(rng), rng) for r in range(len(CONDS))])
np.savez(f'break_{case}.npz', Y=Y, T=T, off=off)
p0 = PRIOR_MU.copy(); p0[0] = np.log(22); p0[1] = np.log(1.2)
r = fit(Y, p0)
T_fit = forward(r['p'], r['o'])
m, s, dm = INS.marginal(T_fit)
res = (Y - m - r['b'][:, None])/np.sqrt(np.maximum(s**2 - SIG_B**2, 0.05**2))
per_wafer = (res**2).sum(1)
np.savez(f'breakfit_{case}.npz', p=r['p'], chi2=r['chi2_data'], per_wafer=per_wafer, res=res)
print(f"{case}: data chi2 {r['chi2_data']:.1f}; worst wafer {CONDS[int(np.argmax(per_wafer))]} chi2 {per_wafer.max():.1f}; params {np.round(r['p'],3)}", flush=True)
