import numpy as np, warnings
warnings.filterwarnings('ignore')
from fvmodel import *
import instrument as INS
np.set_printoptions(precision=3, suppress=True, linewidth=160)
p = TRUTH_A.copy()

print("=== (B) Rank test, continuous output along the flow (structural, no noise) ===")
# one 55 C wafer, uniform inlet: output = theta at every grid cell on the centreline
def cont_out(q, conds, Ny=96):
    grid = Grid(q[4], 30, Ny); g = np.ones(grid.Nx); res = []
    for (td, tp) in conds:
        lam, zb, l, gpc = groups(q, td, tp)
        e, *_ = solve(grid, lam, zb, l, g)
        res.append(1 - np.exp(-e.reshape(grid.Ny, grid.Nx)[:, grid.Nx//2]))
    return np.concatenate(res)
def rank_report(idx, conds, label):
    J = []
    for k in idx:
        h = 1e-4 if k != 4 else 1e-3
        qp = p.copy(); qp[k] += h; qm = p.copy(); qm[k] -= h
        J.append((cont_out(qp, conds) - cont_out(qm, conds))/(2*h))
    J = np.array(J).T; J = J/np.linalg.norm(J, axis=0)
    U_, s, Vt = np.linalg.svd(J, full_matrices=False)
    print(f"{label}: params {[NAMES[k] for k in idx]}\n  singular values (normalized) {s/s[0]}")
    print(f"  weakest direction: {dict(zip([NAMES[k] for k in idx], np.round(Vt[-1],2)))}")
one = [(100.0, 55.0)]
rank_report([0,1,3,4], one, "one wafer, 100 C / 55 C (log_lam, log_zb, log_l, U)")
rank_report([0,1,3], one, "one wafer, U known")
seven = [(td, 55.0) for td in TD]
rank_report([0,1,2,3,4], seven, "seven 55 C wafers with temperature laws")

print("\n=== (C) Fisher information at Lee's design: 35 wafers x 9 sites, V1 noise, run dose offsets (prior sd 0.05) ===")
T, S, So = forward(p, sens=True)
m, s, dm = INS.marginal(T)
# weighted Jacobian of reported thickness: d m / d t * d t / d p / s
Jp = (dm/s)[:, :, None]*S                       # 35 x 9 x 9
Jo = (dm/s)*So                                  # 35 x 9  (each run's own offset)
nP = 9; nR = 35
F = np.zeros((nP+nR, nP+nR))
for r in range(nR):
    Jr = np.zeros((9, nP+nR)); Jr[:, :nP] = Jp[r]; Jr[:, nP+r] = Jo[r]
    F += Jr.T@Jr
F[nP:, nP:] += np.eye(nR)/0.05**2
C = np.linalg.inv(F)[:nP, :nP]
sd = np.sqrt(np.diag(C)); corr = C/np.outer(sd, sd)
for k in range(nP): print(f"  {NAMES[k]:8s} truth {p[k]:7.3f}   Cramer-Rao sd {sd[k]:.3g}")
print("  correlations above 0.8:")
for i in range(nP):
    for j in range(i+1, nP):
        if abs(corr[i,j]) > 0.8: print(f"    {NAMES[i]} ~ {NAMES[j]}: {corr[i,j]:+.2f}")
w, V = np.linalg.eigh(np.linalg.inv(C))
print("  eigenvalues of the physics block (stiffest first):", np.sort(w)[::-1])
v = V[:, np.argmin(w)]; print("  sloppiest direction:", dict(zip(NAMES, np.round(v, 2))))
np.savez('fim.npz', C=C, sd=sd, corr=corr)
# which wafers carry the information: drop-one-wafer trace of physics covariance
print("\n  information by wafer (increase in total physics variance if the wafer is removed), top 6:")
gain = []
for r in range(nR):
    Jr = np.zeros((9, nP+nR)); Jr[:, :nP] = Jp[r]; Jr[:, nP+r] = Jo[r]
    Fm = F - Jr.T@Jr; Cm = np.linalg.inv(Fm)[:nP, :nP]
    gain.append(np.sum(np.diag(Cm)/np.diag(C)) - nP)
for r in np.argsort(gain)[::-1][:6]: print(f"    Td {CONDS[r][0]:.0f} / Tp {CONDS[r][1]:.0f}: {gain[r]:.2f}")
