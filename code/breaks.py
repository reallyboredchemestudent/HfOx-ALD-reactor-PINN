"""Synthetic truths that break the steady-exposure assumptions, solved in time on the chamber plane.
Time is measured as tau = u*t (cm of gas flow). In these units the steady model reads
    d/dy(e - l e_y) - l e_xx + (1/zb)(1 - exp(-e)) = 0,  inlet flux delta*g,
and the matching transient model (kappa = 1, site capacity S = 1/zb) is
    c_tau + d/dy(c - l c_y) - l c_xx = -S * theta_tau,  theta_tau = c (1 - theta),  inlet flux c_in(tau) g.
Break cases:
  nominal       : the transient model itself (must reproduce the steady solve)
  second_order  : theta_tau = c^2/c_ref (1 - theta)                     -> breaks 'first order in gas'
  flow_spike    : gas speed +30 %, pressure +20 % during the pulse, same molecules  -> breaks 'steady transport'
  pool_unpurged : a physisorbed pool (30 % of site capacity at 100 C, shrinking with T) that the purge
                  does not empty; held precursor becomes film when water arrives  -> breaks 'nothing held over'
  mediated      : chemisorption only from the physisorbed pool             -> breaks 'first order in gas'
  surface_cvd   : extra non-saturating growth ~ exposure, rising with T (steady, exact) -> missing mechanism"""
import numpy as np
from scipy.sparse import diags, bmat, identity, csr_matrix
from scipy.integrate import solve_ivp
from fvmodel import Grid, groups, inlet_profile, solve as steady_solve, CONDS, SITES, THK

U_175 = 179.0     # cm/s at 175 C for a 1 cm gap (V2 bracket)
def pulse_len(td): return U_175*(td + 273.15)/448.15*0.2      # cm of flow during the 0.2 s pulse

def transient_wafer(p, td, tp, o, case, Nx=30, Ny=96, tau_end=None):
    grid = Grid(p[4], Nx, Ny); g = inlet_profile(grid.xi, p); gt = np.tile(g, grid.Ny)
    lam, zb, l, gpc = groups(p, td, tp, o)
    delta = lam/zb; S = 1/zb; n = grid.n
    tp_len = pulse_len(td); cref = delta/tp_len
    Adv, Dif = grid.Ay, (grid.Dyy + grid.Dxx)
    pool = case in ('pool_unpurged', 'mediated')
    if case == 'pool_unpurged': Sp = 0.3*S*np.exp(-(td - 100)/50); ka, kd = 2.0, 1e-4
    if case == 'mediated': Sp = 0.3*S; ka, kd, km = 2.0, 0.5, 5.0
    fu_p, fD_p = (1.3, 1/1.2) if case == 'flow_spike' else (1.0, 1.0)
    if tau_end is None: tau_end = tp_len + (3600.0 if pool else 240.0)
    def parts(t):
        on = t < tp_len
        fu = fu_p if on else 1.0; fD = fD_p if on else 1.0
        Op = fu*Adv + fD*l*Dif
        src = grid.binlet*gt*(cref/fu if on else 0.0)*fu      # same molecules: conc/fu, flux scaled by fu
        return Op, src
    def rhs(t, y):
        c = y[:n]; th = y[n:2*n]; Op, src = parts(t); cp = np.maximum(c, 0)
        if case == 'second_order': r = cp**2/cref*(1 - th)
        elif case == 'mediated': r = None
        else: r = cp*(1 - th)
        if pool:
            pp = y[2*n:]
            ads = ka*cp*(1 - pp) - kd*pp
            if case == 'mediated':
                r = km*pp*(1 - th); dp = ads - (S/Sp)*r
            else: dp = ads
            dc = -Op@c + src - Sp*ads - (0.0 if case == 'mediated' else S*r)   # mediated: chem draws on the pool, not the gas
            return np.concatenate([dc, r, dp])
        return np.concatenate([-Op@c + src - S*r, r])
    def jac(t, y):
        c = y[:n]; th = y[n:2*n]; Op, _ = parts(t); cp = np.maximum(c, 0)
        if case == 'second_order': rc = 2*cp/cref*(1 - th); rt = -cp**2/cref
        else: rc = cp*0 + (1 - th); rt = -cp
        if not pool:
            return bmat([[-Op - diags(S*rc), diags(-S*rt)], [diags(rc), diags(rt)]]).tocsc()
        pp = y[2*n:]
        ac = ka*(1 - pp); ap = -ka*cp - kd
        if case == 'mediated':
            r_p = km*(1 - th); r_t = -km*pp
            return bmat([[-Op - diags(Sp*ac), None, diags(-Sp*ap)],
                         [None, diags(r_t), diags(r_p)],
                         [diags(ac), diags(-(S/Sp)*r_t), diags(ap - (S/Sp)*r_p)]]).tocsc()
        return bmat([[-Op - diags(S*rc + Sp*ac), diags(-S*rt), diags(-Sp*ap)],
                     [diags(rc), diags(rt), None], [diags(ac), None, diags(ap)]]).tocsc()
    y = np.zeros(3*n if pool else 2*n)
    for (a, b) in ((0, tp_len), (tp_len, tau_end)):
        sol = solve_ivp(rhs, (a, b), y, method='BDF', jac=jac, rtol=1e-7, atol=1e-10)
        y = sol.y[:, -1]
    th = grid.P@y[n:2*n]
    extra = (Sp/S)*(grid.P@y[2*n:]) if case == 'pool_unpurged' else 0.0
    return THK*gpc*(th + extra)

def steady_cvd(p, td, tp, o, kc175=0.00005, Ec=1.0, Nx=30, Ny=96):
    """Extra non-saturating surface growth with rate ~ local exposure: film += 10*gpc*kc*e (exact steady)."""
    from scipy.sparse.linalg import splu
    grid = Grid(p[4], Nx, Ny); g = inlet_profile(grid.xi, p); lam, zb, l, gpc = groups(p, td, tp, o)
    kc = kc175*np.exp(-Ec/8.617e-5*(1/(td + 273.15) - 1/448.15))
    L = grid.lin(l); src = grid.binlet*(lam/zb)*np.tile(g, grid.Ny); e = np.ones(grid.n)
    for it in range(60):     # extra sink kc*e/zb consumes precursor too
        R = L@e + (1/zb)*(1 - np.exp(-e)) + (kc/zb)*e - src
        J = (L + diags((1/zb)*np.exp(-e) + kc/zb)).tocsc(); de = splu(J).solve(-R); e += de
        if np.max(np.abs(de)) < 1e-11: break
    es = grid.P@e
    return THK*gpc*(1 - np.exp(-es) + kc*es)

def truth_maps(p, case, offsets, conds_transient=None):
    from fvmodel import forward
    T = forward(p, offsets)
    if case == 'surface_cvd':
        return np.array([steady_cvd(p, td, tp, offsets[i]) for i, (td, tp) in enumerate(CONDS)])
    conds_transient = conds_transient if conds_transient is not None else [i for i, c in enumerate(CONDS) if c[1] <= 60 or case == 'pool_unpurged']
    for i in conds_transient:
        td, tp = CONDS[i]; T[i] = transient_wafer(p, td, tp, offsets[i], case)
    return T
