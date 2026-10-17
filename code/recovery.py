"""Recovery at the real design: noisy synthetic wafers from truth A, refit, check bias and interval coverage."""
import numpy as np, sys, warnings
warnings.filterwarnings('ignore')
from fit import *
seed0, n = int(sys.argv[1]), int(sys.argv[2])
rows = []
for rep in range(seed0, seed0+n):
    rng = np.random.default_rng(100+rep)
    Y, T, off = make_data(TRUTH_A, rng)
    p0 = PRIOR_MU.copy(); p0[0] = np.log(22); p0[1] = np.log(1.2)
    r = fit(Y, p0)
    sd = np.sqrt(np.diag(r['cov'])[:9])
    rows.append(np.concatenate([r['p'], sd, [r['chi2_data']]]))
    np.save(f'recovery_{seed0}.npy', np.array(rows))
    print(rep, np.round(r['p'], 3), round(r['chi2_data'], 1), flush=True)
