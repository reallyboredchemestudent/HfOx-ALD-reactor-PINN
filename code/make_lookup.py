"""Instrument lookup from the V1 ellipsometry model: reported single-layer Cauchy thickness
for a true stack Si / SiOx (1.0 nm) / HfO2 (t_true), with a run-level angle-of-incidence offset a (deg)."""
import numpy as np, ellips, sys
from ellips import psidelta, cauchy, n_sio2, WL, fit_single
t_true = np.array([0.0,0.25,0.5,0.75,1,1.5,2,2.5,3,3.5,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
# usage: python make_lookup.py [tmax_nm]  -> extends the table in 1 nm steps (needed if real films exceed 17 nm)
if len(sys.argv) > 1 and float(sys.argv[1]) > 17: t_true = np.concatenate([t_true, np.arange(18, float(sys.argv[1]) + 1)])
a_off = np.linspace(-0.05, 0.05, 11)
T_OX = 1.0
out = np.zeros((t_true.size, a_off.size)); ri = np.zeros_like(out)
for j, a in enumerate(a_off):
    a0 = ellips.ANG.copy(); ellips.ANG = a0 + np.deg2rad(a)
    for i, t in enumerate(t_true):
        L = []
        if t > 0: L.append((cauchy(WL, 2.07, 0.012), t))
        L.append((n_sio2(WL), T_OX))
        P, D = psidelta(L)
        ellips.ANG = a0
        prev = out[i-1, j] if i > 0 else 1.0
        d, n, _ = fit_single(P, D, starts=[(max(prev,0.3), 2.0), (t+0.8, 2.1), (t+2, 1.8), (max(t-0.5,0.3), 2.6)])
        out[i, j] = d; ri[i, j] = n
        ellips.ANG = a0 + np.deg2rad(a)
    ellips.ANG = a0
np.savez('v1_lookup.npz', t_true=t_true, a_off=a_off, d_rep=out, ri_rep=ri, t_ox=T_OX)
for i in (0, 2, 4, 6, 8, 12, 16, 20):
    print(f"true {t_true[i]:5.2f} nm -> reported {out[i,5]:5.2f} (a=0) | {out[i,3]:5.2f} (a=-0.02) | {out[i,7]:5.2f} (a=+0.02)")
