"""Profile likelihood on one synthetic dataset: fix one parameter on a grid, refit everything else."""
import numpy as np, sys, warnings
warnings.filterwarnings('ignore')
from fit import NAMES, PRIOR_MU, fit
data = sys.argv[1]; k = NAMES.index(sys.argv[2]); grid = [float(v) for v in sys.argv[3].split(',')]
Y = np.load(data)['Y']
p0 = PRIOR_MU.copy(); p0[0] = np.log(22); p0[1] = np.log(1.2)
best = fit(Y, p0); print(f"free fit {NAMES[k]} = {best['p'][k]:.3f}, cost {best['cost']:.2f}", flush=True)
out = [(best['p'][k], best['cost'])]
for v in grid:
    r = fit(Y, best['p'], fixed={k: v}, max_nfev=25)
    out.append((v, r['cost'])); print(f"  {NAMES[k]} = {v:.3f}: delta chi2 {r['cost'] - best['cost']:.2f}", flush=True)
np.save(f'profile_{NAMES[k]}.npy', np.array(out))
