"""Hold-one-wafer-out with the finite-volume inverse: refit without one 55 C wafer, predict it with a 90 % interval."""
import numpy as np, sys, warnings
warnings.filterwarnings('ignore')
import instrument as INS
from fit import CONDS, PRIOR_MU, SIG_O, fit, forward
data, tag = sys.argv[1], sys.argv[2]
holds = [int(h) for h in sys.argv[3].split(',')] if len(sys.argv) > 3 else [i for i, c in enumerate(CONDS) if c[1] == 55.0]
z = np.load(data); Y = z['Y']
p0 = PRIOR_MU.copy(); p0[0] = np.log(22); p0[1] = np.log(1.2)
rows = []
for h in holds:
    mask = np.ones(35, bool); mask[h] = False
    r = fit(Y, p0, mask=mask)
    p = r['p']; o = r['o'].copy(); o[h] = 0.0
    T, S, So = forward(p, o, sens=True)
    m, s, dm = INS.marginal(T[h])
    Cp = r['cov'][:9, :9]
    Jt = S[h][:, r['free']]
    var_par = np.einsum('ij,jk,ik->i', Jt, Cp, Jt)*dm**2
    var_dose = (dm*So[h])**2*SIG_O**2
    sd_pred = np.sqrt(var_par + var_dose + s**2)          # s already includes the run-level instrument spread
    rows.append(dict(hold=h, pred=m, sd=sd_pred, obs=Y[h], true=z['T'][h], p=p))
    err = Y[h] - m
    print(f"hold {CONDS[h]}: RMSE {np.sqrt(np.mean(err**2)):.2f} nm, inside 90% {np.mean(np.abs(err) <= 1.645*sd_pred):.2f}, mean width {np.mean(2*1.645*sd_pred):.2f} nm", flush=True)
np.save(f'howo_fv_{tag}.npy', rows, allow_pickle=True)
