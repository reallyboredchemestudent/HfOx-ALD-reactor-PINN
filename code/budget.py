"""Measured error budget, from the saturated wafers (60-75 C source: coverage 1 everywhere, so nothing on them is transport).

  sig    fixed relative site signature of the saturated film (9 values, mean 0 over the sites)
  sig_r  run-to-run relative scatter of the saturated film at fixed deposition temperature (log scale)
  sig_d  site scatter (nm) left after the signature and each wafer's level are removed

Used identically by recalibrated.py (FV), pinn.py (NumPy PINN) and pinn_jax.py (JAX PINN), so the solvers are judged on
the same error model:  reported = instrument( film * (1 + sig_site) * exp(r_run) ) + site noise (instrument and sig_d),
with r_run ~ N(0, sig_r) a per-run film offset."""
import numpy as np

def calibrate(Y, run_mask, conds):
    conds = [tuple(map(float, c)) for c in conds]
    sat = [i for i in range(len(conds)) if run_mask[i] and conds[i][1] >= 60 and np.isfinite(Y[i]).all()]
    if len(sat) < 4: raise ValueError(f"only {len(sat)} complete saturated wafers in the training set; cannot calibrate")
    M = Y[sat]; wm = M.mean(1, keepdims=True); P = M/wm - 1; sig = P.mean(0)
    resid = (P - sig)*wm
    sig_d = float(np.sqrt(np.sum(resid**2)/(resid.size - len(sat) - (M.shape[1] - 1))))
    tds = sorted({conds[i][0] for i in sat}); dev = []
    for td in tds:
        m = np.array([wm[k, 0] for k, i in enumerate(sat) if conds[i][0] == td])
        if m.size > 1: dev += list(np.log(m/np.exp(np.mean(np.log(m)))))
    ngroups = sum(1 for td in tds if sum(1 for i in sat if conds[i][0] == td) > 1)
    sig_r = float(np.std(dev, ddof=1)*np.sqrt(len(dev)/(len(dev) - ngroups)))
    return dict(sig=sig, sig_r=sig_r, sig_d=sig_d, n=len(sat))

def describe(b):
    return (f"measured budget from {b['n']} saturated wafers: site signature (%) {np.round(100*b['sig'], 2).tolist()}, "
            f"run scatter {100*b['sig_r']:.2f} %, site scatter {b['sig_d']:.3f} nm")
