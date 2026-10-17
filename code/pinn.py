"""Inverse PINN for the steady exposure equation, NumPy only (no autodiff libraries are installable here).
Network: one hidden tanh layer, all weights trained, analytic input derivatives up to second order and
analytic parameter gradients of every loss term (checked against finite differences in test_pinn.py).

  e(x, y; Td, Tp) = delta_c * N(phi_x(xi), phi_y(eta), sd, sp)
  phi_x = cos(pi (xi+1)/2)  -> de/dx = 0 at both side walls by construction
  phi_y = (1 - eta)^2       -> de/dy = 0 at the outlet by construction
  inlet Robin condition (e - l e_y = delta g) enforced softly; g(xi) is a small learned network, mean 1.
Loss = w_pde * mean(r_pde^2) + w_in * mean(r_in^2) + sum(data residuals^2)/n_data_scale + priors,
with r_pde scaled by the front width so all terms are O(1). Physics parameters and run offsets are learned jointly."""
import numpy as np
from scipy.optimize import minimize
import instrument as INS
from fvmodel import TD, TP, SITES, CONDS, pv, TREF, KB, NAMES, THK

XH = 7.5; YEND = 15.0
import os
PHYS_SCALE = float(os.environ.get('PHYS_SCALE', 1e-3))
PDE_W = float(os.environ.get('PDE_W', 1.0))
PHYS = ['log_lam', 'log_zb', 'Ea', 'log_l', 'U', 'G0', 'G1']     # same physics as FV (inlet replaced by a network)
SIG_O, SIG_B = 0.05, 0.06
PRIOR_MU = np.array([np.log(19.0), np.log(1.0), 0.0, 0.0, 9.0, 1.3, -0.2])
PRIOR_SD = np.array([2.0, 2.0, 0.3, 2.0, 20.0, 1.0, 1.0])

class Layout:
    def __init__(self, H=100, HG=6):
        self.H, self.HG = H, HG
        sizes = [('a', H), ('W', H*4), ('b', H), ('a0', 1), ('gc', HG), ('gv', HG), ('gd', HG),
                 ('phys', 7), ('o', 35), ('bo', 35)]
        self.sl = {}; i = 0
        for k, n in sizes: self.sl[k] = slice(i, i+n); i += n
        self.n = i
    def get(self, th, k):
        v = th[self.sl[k]]
        return v.reshape(self.H, 4) if k == 'W' else v

def init_params(lay, rng, phys0=None):
    th = np.zeros(lay.n)
    th[lay.sl['a']] = rng.normal(0, 1/np.sqrt(lay.H), lay.H)
    W = rng.normal(0, 1.0, (lay.H, 4)); W[:, 1] *= 4.0       # sharper features along the flow
    th[lay.sl['W']] = W.ravel(); th[lay.sl['b']] = rng.normal(0, 1.0, lay.H); th[lay.sl['a0']] = 0.3
    th[lay.sl['gc']] = rng.normal(0, 0.05, lay.HG); th[lay.sl['gv']] = rng.normal(0, 2.0, lay.HG); th[lay.sl['gd']] = rng.normal(0, 1.0, lay.HG)
    th[lay.sl['phys']] = PRIOR_MU if phys0 is None else phys0
    return th

def cond_coeffs(phys, o):
    """Per-condition log lam_c, log zb_c, log l_c, gpc, and their derivatives w.r.t. the physics vector."""
    lam_ref, zb_ref, Ea, l_ref, U, G0, G1 = phys
    out = []
    for ci, (td, tp) in enumerate(CONDS):
        T = td + 273.15; sd = (td - 175)/75; gpc = G0 + G1*sd
        llam = lam_ref + np.log(pv(tp)/pv(55)) + np.log(G0/gpc) + o[ci]
        lzb = zb_ref + 0.5*np.log(T/TREF) + Ea/KB*(1/T - 1/TREF)
        ll = l_ref + 0.75*np.log(T/TREF)
        d = dict(llam=np.array([1,0,0,0,0, 1/G0 - 1/gpc, -sd/gpc]),
                 lzb=np.array([0,1,(1/KB)*(1/T - 1/TREF),0,0,0,0]), ll=np.array([0,0,0,1,0,0,0]))
        out.append((llam, lzb, ll, gpc, sd, (tp - 65)/10, d))
    return out

def features(xi, eta):
    fx = np.cos(np.pi*(xi+1)/2); fx1 = -(np.pi/2)*np.sin(np.pi*(xi+1)/2); fx2 = -(np.pi/2)**2*fx
    fy = (1-eta)**2; fy1 = -2*(1-eta); fy2 = 2*np.ones_like(eta)
    return fx, fx1, fx2, fy, fy1, fy2

def net(th, lay, Z, fx1, fx2, fy1, fy2, need2=True):
    """Z: (P,4) inputs. Returns N, N_xi, N_xixi, N_eta, N_etaeta and cached arrays for gradients."""
    a = lay.get(th, 'a'); W = lay.get(th, 'W'); b = lay.get(th, 'b'); a0 = th[lay.sl['a0']][0]
    h = Z@W.T + b; s = np.tanh(h); s1 = 1 - s*s; s2 = -2*s*s1; s3 = -2*s1*s1 - 2*s*s2
    wx = W[:, 0]; wy = W[:, 1]
    N = s@a + a0
    Nx = (s1*wx)@a * fx1; Ny = (s1*wy)@a * fy1
    out = dict(N=N, Nx=Nx, Ny=Ny, s=s, s1=s1, s2=s2, s3=s3, Z=Z, fx1=fx1, fx2=fx2, fy1=fy1, fy2=fy2)
    if need2:
        out['Nxx'] = (s2*wx**2)@a*fx1**2 + (s1*wx)@a*fx2
        out['Nyy'] = (s2*wy**2)@a*fy1**2 + (s1*wy)@a*fy2
    return out

def net_grad(th, lay, c, cN=None, cNx=None, cNy=None, cNxx=None, cNyy=None):
    """Gradient w.r.t. network weights of sum_p [cN N + cNx N_xi + cNy N_eta + cNxx N_xixi + cNyy N_etaeta]."""
    a = lay.get(th, 'a'); W = lay.get(th, 'W'); wx = W[:, 0]; wy = W[:, 1]
    s, s1, s2, s3, Z = c['s'], c['s1'], c['s2'], c['s3'], c['Z']
    P = Z.shape[0]; z0 = np.zeros(P)
    cN = z0 if cN is None else cN; cNx = z0 if cNx is None else cNx; cNy = z0 if cNy is None else cNy
    cNxx = z0 if cNxx is None else cNxx; cNyy = z0 if cNyy is None else cNyy
    f1x, f2x, f1y, f2y = c['fx1'], c['fx2'], c['fy1'], c['fy2']
    # d/da_j
    ga = cN@s + (cNx*f1x)@(s1)*wx + (cNy*f1y)@(s1)*wy \
         + (cNxx*f1x**2)@s2*wx**2 + (cNxx*f2x)@s1*wx + (cNyy*f1y**2)@s2*wy**2 + (cNyy*f2y)@s1*wy
    # coefficient multiplying dh_j (derivative of each term w.r.t. the pre-activation h_j), per point:
    # grouped as rank-one outer products so each P x H array is touched only a few times
    u2x = cNx*f1x + cNxx*f2x; u2y = cNy*f1y + cNyy*f2y
    B2 = np.outer(u2x, wx) + np.outer(u2y, wy)
    B3 = np.outer(cNxx*f1x**2, wx**2) + np.outer(cNyy*f1y**2, wy**2)
    A = (cN[:, None]*s1 + s2*B2 + s3*B3)*a
    gb = A.sum(0)
    gW = A.T@Z
    # explicit dependence on w_x and w_y (not through h)
    gW[:, 0] += a*((cNx*f1x)@s1 + 2*wx*((cNxx*f1x**2)@s2) + (cNxx*f2x)@s1)
    gW[:, 1] += a*((cNy*f1y)@s1 + 2*wy*((cNyy*f1y**2)@s2) + (cNyy*f2y)@s1)
    ga0 = cN.sum()
    return ga, gW, gb, ga0

def inlet_g(th, lay, xi):
    c = th[lay.sl['gc']]; v = th[lay.sl['gv']]; d = th[lay.sl['gd']]
    t = np.tanh(np.outer(xi, v) + d); logG = t@c; G = np.exp(logG); g = G/G.mean()
    # d log G / d params
    dl_c = t; dl_v = (1-t*t)*c*xi[:, None]; dl_d = (1-t*t)*c
    def grad(w):       # gradient of sum_i w_i g_i
        out = []
        for dl in (dl_c, dl_v, dl_d):
            dg = g[:, None]*(dl - (g[:, None]*dl).mean(0))
            out.append(w@dg)
        return np.concatenate(out)
    return g, grad

class Problem:
    def __init__(self, Y, mask, lay, rng, n55=260, nsat=50, nin=24, Lref=24.0, budget=None):
        # budget = None: ideal error model (additive run shift 'bo', prior SIG_B). budget = dict from budget.calibrate:
        # measured error model, film * (1 + sig_site) * exp(bo_run), bo ~ N(0, sig_r), site variance + sig_d^2
        self.budget = budget
        mask = np.asarray(mask, bool)
        self.mask2 = np.repeat(mask[:, None], 9, 1) if mask.ndim == 1 else mask
        self.mask2 = self.mask2 & np.isfinite(Y)
        self.Y, self.mask, self.lay = np.nan_to_num(Y), mask, lay
        # collocation: per condition, eta skewed toward the wafer region; more points where fronts can sit
        pts = []
        for ci, (td, tp) in enumerate(CONDS):
            m = n55 if tp <= 60 else nsat
            xi = rng.uniform(-1, 1, m); eta = rng.beta(1.3, 1.1, m)
            pts.append(np.stack([np.full(m, ci), xi, eta], 1))
        self.col = np.concatenate(pts)
        self.inl = np.concatenate([np.stack([np.full(nin, ci), np.linspace(-1, 1, nin), np.zeros(nin)], 1) for ci in range(35)])
        self.w = dict(pde=PDE_W, inl=1.0, data=1.0)
        self.ndata = int(self.mask2.sum())

    def loss(self, th, want_grad=True, parts=False):
        lay = self.lay; phys = th[lay.sl['phys']]; o = th[lay.sl['o']]; bo = th[lay.sl['bo']]
        U = phys[4]; L = YEND + U
        cc = cond_coeffs(phys, o)
        G = np.zeros(lay.n); total = 0.0; comp = {}
        # ---------- PDE residual ----------
        ci = self.col[:, 0].astype(int); xi = self.col[:, 1]; eta = self.col[:, 2]
        llam = np.array([c[0] for c in cc])[ci]; lzb = np.array([c[1] for c in cc])[ci]; ll = np.array([c[2] for c in cc])[ci]
        sd = np.array([c[4] for c in cc])[ci]; sp = np.array([c[5] for c in cc])[ci]
        fx, fx1, fx2, fy, fy1, fy2 = features(xi, eta)
        Z = np.stack([fx, fy, sd, sp], 1)
        c = net(th, lay, Z, fx1, fx2, fy1, fy2)
        lam = np.exp(llam); zb = np.exp(lzb); l = np.exp(ll)
        e = lam*c['N']/zb; ee = np.exp(-np.clip(e, -30, 60)); eed = ee*((e > -30) & (e < 60))
        lap = c['Nyy']/L**2 + c['Nxx']/XH**2
        r = lam*c['Ny']/L - l*lam*lap + 1 - ee
        P = r.size; wp = self.w['pde']/P
        total += wp*np.sum(r*r); comp['pde'] = np.mean(r*r)
        if want_grad:
            k = 2*wp*r
            ga, gW, gb, ga0 = net_grad(th, lay, c, cN=k*lam/zb*eed, cNy=k*lam/L, cNyy=-k*l*lam/L**2, cNxx=-k*l*lam/XH**2)
            G[lay.sl['a']] += ga; G[lay.sl['W']] += gW.ravel(); G[lay.sl['b']] += gb; G[lay.sl['a0']] += ga0
            dr_dllam = lam*c['Ny']/L - l*lam*lap + e*eed
            dr_dlzb = -e*eed
            dr_dll = -l*lam*lap
            dr_dL = -lam*c['Ny']/L**2 + 2*l*lam*c['Nyy']/L**3
            for q, (a_, b_, d_) in enumerate(zip([cc[j][6]['llam'] for j in range(35)], [cc[j][6]['lzb'] for j in range(35)], [cc[j][6]['ll'] for j in range(35)])):
                sel = ci == q
                if not sel.any(): continue
                kk = k[sel]
                G[lay.sl['phys']] += a_*np.sum(kk*dr_dllam[sel]) + b_*np.sum(kk*dr_dlzb[sel]) + d_*np.sum(kk*dr_dll[sel])
                G[lay.sl['o'].start + q] += np.sum(kk*dr_dllam[sel])
            G[lay.sl['phys'].start + 4] += np.sum(k*dr_dL)
        # ---------- inlet Robin residual: N - l N_eta / L - g(xi) = 0 at eta = 0 ----------
        ci2 = self.inl[:, 0].astype(int); xi2 = self.inl[:, 1]; eta2 = self.inl[:, 2]
        fx, fx1, fx2, fy, fy1, fy2 = features(xi2, eta2)
        sd2 = np.array([cc[j][4] for j in range(35)])[ci2]; sp2 = np.array([cc[j][5] for j in range(35)])[ci2]
        c2 = net(th, lay, np.stack([fx, fy, sd2, sp2], 1), fx1, fx2, fy1, fy2, need2=False)
        l2 = np.exp(np.array([cc[j][2] for j in range(35)])[ci2])
        xig = np.linspace(-1, 1, 24); g, ggrad = inlet_g(th, lay, xig)
        gi = np.tile(g, 35)
        r2 = c2['N'] - l2*c2['Ny']/L - gi
        wi = self.w['inl']/r2.size
        total += wi*np.sum(r2*r2); comp['inl'] = np.mean(r2*r2)
        if want_grad:
            k2 = 2*wi*r2
            ga, gW, gb, ga0 = net_grad(th, lay, c2, cN=k2, cNy=-k2*l2/L)
            G[lay.sl['a']] += ga; G[lay.sl['W']] += gW.ravel(); G[lay.sl['b']] += gb; G[lay.sl['a0']] += ga0
            G[lay.sl['phys'].start + 3] += np.sum(k2*(-l2*c2['Ny']/L))
            G[lay.sl['phys'].start + 4] += np.sum(k2*(l2*c2['Ny']/L**2))
            wsum = -k2.reshape(35, 24).sum(0)
            gg = ggrad(wsum); HG = lay.HG
            G[lay.sl['gc']] += gg[:HG]; G[lay.sl['gv']] += gg[HG:2*HG]; G[lay.sl['gd']] += gg[2*HG:]
        # ---------- data through the instrument model ----------
        mask = self.mask
        cidx = np.repeat(np.arange(35), 9); xs = np.tile(SITES[:, 0], 35)/XH; ys = np.tile(SITES[:, 1], 35)
        eta3 = (ys + U)/L
        fx, fx1, fx2, fy, fy1, fy2 = features(xs, eta3)
        sd3 = np.array([cc[j][4] for j in range(35)])[cidx]; sp3 = np.array([cc[j][5] for j in range(35)])[cidx]
        c3 = net(th, lay, np.stack([fx, fy, sd3, sp3], 1), fx1, fx2, fy1, fy2, need2=False)
        lam3 = np.exp(np.array([cc[j][0] for j in range(35)])[cidx]); zb3 = np.exp(np.array([cc[j][1] for j in range(35)])[cidx])
        gpc3 = np.array([cc[j][3] for j in range(35)])[cidx]
        e3 = lam3*c3['N']/zb3; ee3 = np.exp(-np.clip(e3, -30, 60)); ee3d = ee3*((e3 > -30) & (e3 < 60))
        B = self.budget
        if B is None:
            fac = 1.0; t3 = THK*gpc3*(1 - ee3)
        else:
            fac = np.tile(1 + B['sig'], 35)*np.exp(bo[cidx]); t3 = THK*gpc3*(1 - ee3)*fac
        m, s, dm, ds = INS.marginal_full(t3)
        big = s**2 - SIG_B**2 > 0.05**2
        v = np.maximum(s**2 - SIG_B**2, 0.05**2) + (0.0 if B is None else B['sig_d']**2)
        ss = np.sqrt(v); dss = np.where(big, s*ds/ss, 0.0)
        rd = (self.Y.ravel() - m - (bo[cidx] if B is None else 0.0))/ss
        md = self.mask2.ravel()
        wd = self.w['data']/self.ndata
        total += wd*np.sum(rd[md]**2); comp['data_chi2'] = np.sum(rd[md]**2)
        if want_grad:
            k3 = np.where(md, 2*wd*rd, 0.0)
            dt = (-dm/ss - rd*dss/ss)*k3                     # d loss / d t3 (noise level depends on t3)
            de = dt*THK*gpc3*fac*ee3d                         # d loss / d e3
            ga, gW, gb, ga0 = net_grad(th, lay, c3, cN=de*lam3/zb3)
            G[lay.sl['a']] += ga; G[lay.sl['W']] += gW.ravel(); G[lay.sl['b']] += gb; G[lay.sl['a0']] += ga0
            for q in range(35):
                sel = cidx == q
                dll = np.sum(de[sel]*e3[sel]); dlz = -dll
                dgpc = np.sum(dt[sel]*THK*(1 - ee3[sel])*(fac[sel] if B is not None else 1.0))
                G[lay.sl['phys']] += cc[q][6]['llam']*dll + cc[q][6]['lzb']*dlz
                G[lay.sl['phys'].start + 5] += dgpc; G[lay.sl['phys'].start + 6] += dgpc*cc[q][4]
                G[lay.sl['o'].start + q] += dll
                G[lay.sl['bo'].start + q] += np.sum(-k3[sel]/ss[sel]) if B is None else np.sum(dt[sel]*t3[sel])
            deta_dU = (YEND - ys)/L**2
            G[lay.sl['phys'].start + 4] += np.sum(de*lam3/zb3*c3['Ny']*deta_dU)
        # ---------- priors ----------
        rp = (phys - PRIOR_MU)/PRIOR_SD; total += np.sum(rp**2)/self.ndata
        sbo = SIG_B if self.budget is None else self.budget['sig_r']
        total += (np.sum((o/SIG_O)**2) + np.sum((bo/sbo)**2))/self.ndata
        if want_grad:
            G[lay.sl['phys']] += 2*rp/PRIOR_SD/self.ndata
            G[lay.sl['o']] += 2*o/SIG_O**2/self.ndata; G[lay.sl['bo']] += 2*bo/sbo**2/self.ndata
        if parts: return total, G, comp
        return (total, G) if want_grad else total

    def predict(self, th, ci_list):
        """True thickness at the 9 sites for given conditions (run offsets set to their fitted values or 0).
        With a measured budget the fixed site signature is included (the run's own film offset is not)."""
        lay = self.lay; phys = th[lay.sl['phys']]; o = th[lay.sl['o']]
        U = phys[4]; L = YEND + U; cc = cond_coeffs(phys, o); out = []
        for q in ci_list:
            xs = SITES[:, 0]/XH; eta = (SITES[:, 1] + U)/L
            fx, fx1, fx2, fy, fy1, fy2 = features(xs, eta)
            Z = np.stack([fx, fy, np.full(9, cc[q][4]), np.full(9, cc[q][5])], 1)
            c = net(th, lay, Z, fx1, fx2, fy1, fy2, need2=False)
            e = np.exp(cc[q][0])*c['N']/np.exp(cc[q][1])
            out.append(THK*cc[q][3]*(1 - np.exp(-e))*(1.0 if self.budget is None else 1 + self.budget['sig']))
        return np.array(out)

def train(prob, th0, stages=8, iters=500, adapt=True, physics_only=False, verbose=True, bounds=None):
    th = th0.copy(); lay = prob.lay
    if physics_only: prob.w['data'] = 0.0
    for st in range(stages):
        if adapt and not physics_only and st > 0:
            # gradient-norm balancing on the network weights (Wang, Teng & Perdikaris 2021), moving average
            net_sl = slice(0, lay.sl['a0'].stop)
            grads = {}
            for key in ('pde', 'inl', 'data'):
                w0 = dict(prob.w); prob.w = {k: (1.0 if k == key else 0.0) for k in w0}
                _, g = prob.loss(th); grads[key] = np.abs(g[net_sl]); prob.w = w0
            ref = grads['pde'].max()
            for key in ('inl', 'data'):
                lam_hat = ref/(grads[key].mean() + 1e-12)
                prob.w[key] = min(0.9*prob.w[key] + 0.1*lam_hat, 50.0)   # capped: an uncapped weight starves the PDE term
        # L-BFGS-B's first trial step has unit length in the variables; this loss is far stiffer than that,
        # so optimise scaled variables u = th / SC (first step 1e-3 in th), with bounds scaled to match.
        SC = np.full(lay.n, 1e-3); SC[lay.sl['phys']] = PHYS_SCALE    # physics parameters move on their own scale
        fs = lambda u: (lambda fg: (fg[0], fg[1]*SC))(prob.loss(u*SC))
        bs = None if bounds is None else [(None if lo is None else lo/SC[i], None if hi is None else hi/SC[i]) for i, (lo, hi) in enumerate(bounds)]
        res = minimize(fs, th/SC, jac=True, method='L-BFGS-B', bounds=bs, options=dict(maxiter=iters, maxcor=30, ftol=1e-13))
        th = res.x*SC
        if verbose:
            _, _, comp = prob.loss(th, parts=True)
            print(f"  stage {st}: loss {res.fun:.4g} pde {comp['pde']:.2e} inlet {comp['inl']:.2e} data chi2 {comp['data_chi2']:.1f} "
                  f"w {prob.w['inl']:.2g}/{prob.w['data']:.2g} phys {np.round(th[lay.sl['phys']],3)}", flush=True)
    return th
