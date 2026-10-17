import numpy as np, warnings
warnings.filterwarnings('ignore')
from fvmodel import *
np.set_printoptions(precision=4, suppress=True, linewidth=160)
p = TRUTH_A.copy()
YOBS = np.linspace(2.5, 12.5, 81)   # continuous output across the wafer, fixed physical positions
def cont_out(q, conds, Ny=160):
    grid = Grid(q[4], 30, Ny); g = np.ones(grid.Nx); res = []
    for (td, tp) in conds:
        lam, zb, l, gpc = groups(q, td, tp)
        e, *_ = solve(grid, lam, zb, l, g)
        res.append(np.interp(YOBS, grid.yc, 1 - np.exp(-e.reshape(grid.Ny, grid.Nx)[:, grid.Nx//2])))
    return np.concatenate(res)
def rank_report(idx, conds, label):
    J = []
    for k in idx:
        h = 1e-4 if k != 4 else 1e-3
        qp = p.copy(); qp[k] += h; qm = p.copy(); qm[k] -= h
        J.append((cont_out(qp, conds) - cont_out(qm, conds))/(2*h))
    J = np.array(J).T; J = J/np.linalg.norm(J, axis=0)
    _, s, Vt = np.linalg.svd(J, full_matrices=False)
    print(f"{label}\n  singular values (normalized) {s/s[0]}")
    print(f"  weakest direction: {dict(zip([NAMES[k] for k in idx], np.round(Vt[-1],3)))}")
rank_report([0,1,3,4], [(100.0, 55.0)], "One 55 C wafer (100 C), continuous profile: log_lam, log_zb, log_l, U")
rank_report([0,1,3], [(100.0, 55.0)], "Same, U fixed: log_lam, log_zb, log_l")
rank_report([0,1,2,3,4], [(td, 55.0) for td in TD], "Seven 55 C wafers + temperature laws: log_lam, log_zb, Ea, log_l, U")
rank_report([0,1,2,3,4], [(td, tp) for td in TD for tp in (55.0, 60.0)], "Add the seven 60 C wafers")
rank_report([0,1,2,3,4], [(td, tp) for td in TD for tp in (50.0, 55.0, 60.0)], "Hypothetical: add 50 C wafers (dose-equivalence / lower dose)")
