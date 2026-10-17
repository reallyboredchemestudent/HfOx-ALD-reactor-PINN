import numpy as np, warnings, time
warnings.filterwarnings('ignore')
from scipy.optimize import least_squares
from fvmodel import *
import instrument as INS
SIG_O = 0.05
SIG_B = 0.06   # run-level instrument shift (nm), shared by a wafer's 9 sites
PRIOR_MU = np.array([np.log(19.0), np.log(1.0), 0.0, 0.0, 9.0, 0.0, 0.0, 1.3, -0.2])
PRIOR_SD = np.array([2.0, 2.0, 0.3, 2.0, 20.0, 1.0, 1.0, 1.0, 1.0])   # weak
LB = np.array([np.log(3), np.log(0.05), -1.0, np.log(0.02), 0.0, -0.9, -0.6, 0.5, -1.0])
UB = np.array([np.log(80), np.log(10), 1.0, np.log(20), 30.0, 0.9, 0.6, 2.5, 1.0])

def make_data(p, rng, gfun=None, sig_o=SIG_O, Nx=30, Ny=96, T=None):
    off = rng.normal(0, sig_o, len(CONDS))
    if T is None: T = forward(p, off, gfun, Nx, Ny)
    Y = np.array([INS.generate(T[r], INS.sample_state(rng), rng) for r in range(len(CONDS))])
    return Y, T, off

def fit(Y, p0, mask=None, fixed=None, Nx=30, Ny=96, verbose=0, max_nfev=40):
    """mask: boolean (35,) wafers used. fixed: dict {param index: value} for profile likelihood."""
    mask = np.ones(len(CONDS), bool) if mask is None else np.asarray(mask, bool)
    mask = np.repeat(mask[:, None], 9, 1) if mask.ndim == 1 else mask       # site-level mask (35, 9)
    mask = mask & np.isfinite(Y); Y = np.nan_to_num(Y)
    fixed = {} if fixed is None else fixed
    free = [k for k in range(9) if k not in fixed]
    nf = len(free); nR = len(CONDS)
    def unpack(z):
        p = p0.copy()
        for k, v in fixed.items(): p[k] = v
        p[free] = z[:nf]; return p, z[nf:nf+nR], z[nf+nR:]
    cache = {}
    def model(z):
        key = z.tobytes()
        if key not in cache:
            p, o, b = unpack(z)
            T, S, So = forward(p, o, None, Nx, Ny, sens=True)
            cache.clear(); cache[key] = (T, S, So)
        return cache[key]
    def sres(s): return np.sqrt(np.maximum(s**2 - SIG_B**2, 0.05**2))
    def res(z):
        p, o, b = unpack(z); T, S, So = model(z)
        m, s, dm = INS.marginal(T)
        r = ((Y - m - b[:, None])/sres(s))[mask].ravel()
        return np.concatenate([r, o/SIG_O, b/SIG_B, ((p - PRIOR_MU)/PRIOR_SD)[free]])
    def jac(z):
        p, o, b = unpack(z); T, S, So = model(z)
        m, s, dm = INS.marginal(T)
        ss = sres(s); w = (-dm/ss)
        Jp = (w[:, :, None]*S)[:, :, free]                     # 35 x 9 x nf
        Jo = np.zeros((nR, 9, nR)); Jo[np.arange(nR), :, np.arange(nR)] = w*So
        Jb = np.zeros((nR, 9, nR)); Jb[np.arange(nR), :, np.arange(nR)] = -1/ss
        Jd = np.concatenate([Jp, Jo, Jb], 2)[mask].reshape(-1, nf+2*nR)
        Jpri = np.zeros((2*nR, nf+2*nR)); Jpri[:nR, nf:nf+nR] = np.eye(nR)/SIG_O; Jpri[nR:, nf+nR:] = np.eye(nR)/SIG_B
        Jpp = np.zeros((nf, nf+2*nR)); Jpp[:, :nf] = np.diag(1/PRIOR_SD[free])
        return np.vstack([Jd, Jpri, Jpp])
    z0 = np.concatenate([p0[free], np.zeros(2*nR)])
    lb = np.concatenate([LB[free], -0.5*np.ones(nR), -1*np.ones(nR)]); ub = np.concatenate([UB[free], 0.5*np.ones(nR), np.ones(nR)])
    z0 = np.clip(z0, lb + 1e-9, ub - 1e-9)
    sol = least_squares(res, z0, jac=jac, bounds=(lb, ub), method='trf', x_scale='jac', max_nfev=max_nfev, verbose=verbose)
    p, o, b = unpack(sol.x)
    J = sol.jac; 
    try: Cov = np.linalg.inv(J.T@J)
    except np.linalg.LinAlgError: Cov = np.linalg.pinv(J.T@J)
    return dict(p=p, o=o, b=b, cost=2*sol.cost, chi2_data=np.sum(res(sol.x)[:mask.sum()]**2), ndata=int(mask.sum()), cov=Cov, free=free, nfev=sol.nfev)

if __name__ == '__main__':
    rng = np.random.default_rng(0)
    Y, T, off = make_data(TRUTH_A, rng)
    np.savez('synthA_rep0.npz', Y=Y, T=T, off=off)
    p0 = PRIOR_MU.copy(); p0[0] = np.log(22); p0[1] = np.log(1.2); p0[5] = 0.0; p0[6] = 0.0
    t = time.time(); r = fit(Y, p0, verbose=0)
    sd = np.sqrt(np.diag(r['cov'])[:9])
    print(f"fit time {time.time()-t:.0f}s, nfev {r['nfev']}, data chi2 {r['chi2_data']:.1f} for {35*9} points")
    for k in range(9): print(f"  {NAMES[k]:8s} truth {TRUTH_A[k]:7.3f}  fit {r['p'][k]:7.3f} +- {sd[k]:.3f}")
