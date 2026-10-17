"""Inverse PINN for the steady exposure equation in JAX: the same network, loss, priors, adaptive weights and
optimiser as pinn.py (NumPy, hand-derived gradients), but every derivative is automatic: d/dxi, d2/dxi2, d/deta,
d2/deta2 of the network (forward mode) and the gradient of the loss with respect to all weights, physics parameters
and run offsets (reverse mode). x64 throughout.

  python pinn_jax.py selfcheck
        value and gradient against the NumPy implementation at random weights (must agree to ~1e-8 relative)
  python pinn_jax.py run --data synthA_rep0.npz --tag JA --seeds 0-4 --holds=-1,0,15,30 --jobs 5
        synthetic replication of the NumPy study (compare with pinn_A_* results / compare.py)
  python pinn_jax.py run --data ../results/real_Y.npz --tag REAL --seeds 0-9 --holds 55 --td 100-200 --jobs 8
        real wafers: one ensemble member per seed for every held-out 55 C wafer (matches run_real.py howo)
  python pinn_jax.py run ... --budget measured --tag REALM --pre-tag REAL
        the same with the error budget measured on the saturated wafers (budget.py; matches recalibrated.py)

Outputs pinn_<tag>_s<seed>_h<hold>.npz in --out with the keys pinn_run.py writes (th, pred, phys, comp, g, time)
plus the final adaptive weights. Settings default to the configuration that worked in the NumPy study:
N55=200, NSAT=40, STAGES=8, PDE_W=20, PHYS_SCALE=1e-2, physics-only pretraining, start at the FV fitter's start.
"""
import os, sys, time, argparse, subprocess, json
import numpy as np

# '--holds -1,55' would be read as an option because the value starts with '-': join it as '--holds=-1,55'
for _i, _a in enumerate(sys.argv[:-1]):
    if _a == '--holds': sys.argv[_i:_i + 2] = [f'--holds={sys.argv[_i + 1]}']; break
ap = argparse.ArgumentParser()
ap.add_argument('cmd', choices=['selfcheck', 'run'])
ap.add_argument('--data', default='synthA_rep0.npz'); ap.add_argument('--tag', default='JA')
ap.add_argument('--out', default='.'); ap.add_argument('--seeds', default='0')
ap.add_argument('--holds', default='-1', help="comma list: -1 = all data, 55 = each 55 C wafer in --td, all, or run indices 0,15,30")
ap.add_argument('--td', default=None, help='deposition temperatures used for training, e.g. 100-200 (default all)')
ap.add_argument('--n55', type=int, default=200); ap.add_argument('--nsat', type=int, default=40)
ap.add_argument('--stages', type=int, default=8); ap.add_argument('--iters', type=int, default=400)
ap.add_argument('--pre-stages', type=int, default=6); ap.add_argument('--pre-iters', type=int, default=600)
ap.add_argument('--pde-w', type=float, default=20.0); ap.add_argument('--phys-scale', type=float, default=1e-2)
ap.add_argument('--hidden', type=int, default=100); ap.add_argument('--jobs', type=int, default=1)
ap.add_argument('--budget', choices=['ideal', 'measured'], default='ideal', help='error model: ideal (instrument only) or measured on the saturated wafers (budget.py)')
ap.add_argument('--pre-tag', default=None, help='reuse physics-only pretraining saved under this tag (it does not depend on the data or the budget)')
A = ap.parse_args()

# ---- data first: the cycle count and the site positions must be set before the model modules are imported ----
HERE = os.path.dirname(os.path.abspath(__file__))
Z = np.load(A.data if os.path.exists(A.data) else os.path.join(HERE, A.data), allow_pickle=True)
if 'ncyc' in Z: os.environ['ALD_NCYC'] = str(float(Z['ncyc']))
os.environ['PDE_W'] = str(A.pde_w); os.environ['PHYS_SCALE'] = str(A.phys_scale)
import fvmodel as FV
if 'sites' in Z: FV.SITES[:] = Z['sites']
import instrument as INS
import pinn as PN                                     # NumPy reference: layout, collocation, priors, reporting
import budget as BU
if 'ncyc' in Z: FV.require_ncyc(Z['ncyc'], PN)

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from scipy.optimize import minimize

CONDS = FV.CONDS; XH, YEND = PN.XH, PN.YEND
TDc = np.array([c[0] for c in CONDS]); TPc = np.array([c[1] for c in CONDS])
# constants as JAX arrays: they are indexed with traced index arrays inside the loss
LPV = jnp.asarray(np.log(FV.pv(TPc)/FV.pv(55))); SDc = jnp.asarray((TDc - 175)/75); SPc = jnp.asarray((TPc - 65)/10)
TKc = jnp.asarray(TDc + 273.15); XIG = jnp.asarray(np.linspace(-1, 1, 24))

def pp_eval(xk, ck, t):
    """scipy PPoly (PchipInterpolator) cubic piece evaluation, differentiable in t."""
    i = jnp.clip(jnp.searchsorted(xk, t, side='right') - 1, 0, xk.shape[0] - 2)
    dx = t - xk[i]
    return ((ck[0, i]*dx + ck[1, i])*dx + ck[2, i])*dx + ck[3, i]
MX, MC = jnp.asarray(INS._m.x), jnp.asarray(INS._m.c); SX, SCc = jnp.asarray(INS._s.x), jnp.asarray(INS._s.c)

def marginal(t):
    """Mean and total sd of the reported thickness (instrument.marginal_full, values; derivatives by autodiff)."""
    tc = jnp.clip(t, 0.0, INS.TMAX)
    sv = pp_eval(SX, SCc, tc)
    return pp_eval(MX, MC, tc), jnp.sqrt(sv**2 + INS.SIG0**2)

def make_loss(lay, budget=None):
    """budget None = ideal error model (additive run shift bo, prior SIG_B); a budget.calibrate dict = measured model
    (film x (1 + sig_site) x exp(bo_run), bo prior sig_r, site variance + sig_d^2), exactly as pinn.Problem(budget=...)."""
    H = lay.H; sl = lay.sl
    FAC = None if budget is None else jnp.asarray(np.tile(1 + budget['sig'], 35))
    def unpack(th):
        return (th[sl['a']], th[sl['W']].reshape(H, 4), th[sl['b']], th[sl['a0']][0],
                th[sl['gc']], th[sl['gv']], th[sl['gd']], th[sl['phys']], th[sl['o']], th[sl['bo']])
    def coeffs(phys, o):
        lam_ref, zb_ref, Ea, l_ref, U, G0, G1 = [phys[k] for k in range(7)]
        gpc = G0 + G1*SDc
        llam = lam_ref + LPV + jnp.log(G0/gpc) + o
        lzb = zb_ref + 0.5*jnp.log(TKc/FV.TREF) + Ea/FV.KB*(1/TKc - 1/FV.TREF)
        ll = l_ref + 0.75*jnp.log(TKc/FV.TREF)
        return llam, lzb, ll, gpc
    def net(a, W, b, a0, xi, eta, sdv, spv):
        fx = jnp.cos(jnp.pi*(xi + 1)/2); fy = (1 - eta)**2          # side-wall and outlet conditions by construction
        h = fx[:, None]*W[:, 0] + fy[:, None]*W[:, 1] + sdv[:, None]*W[:, 2] + spv[:, None]*W[:, 3] + b
        return jnp.tanh(h)@a + a0
    def d1(f, x):            # elementwise function of a point array: N and dN/dx by one forward-mode pass
        return jax.jvp(f, (x,), (jnp.ones_like(x),))
    def d2(f, x):            # dN/dx and d2N/dx2
        g = lambda x_: d1(f, x_)[1]
        return d1(g, x)
    def inlet_g(gc, gv, gd):
        t = jnp.tanh(jnp.outer(XIG, gv) + gd); G = jnp.exp(t@gc); return G/jnp.mean(G)
    def terms(th, col_ci, col_xi, col_eta, inl_ci, inl_xi, Yv, md, ndata):
        a, W, b, a0, gc, gv, gd, phys, o, bo = unpack(th)
        U = phys[4]; L = YEND + U
        llam, lzb, ll, gpc = coeffs(phys, o)
        # PDE residual at the collocation points (per-u form, scaled by the front width)
        sdv, spv = SDc[col_ci], SPc[col_ci]
        fxi = lambda x_: net(a, W, b, a0, x_, col_eta, sdv, spv)
        feta = lambda e_: net(a, W, b, a0, col_xi, e_, sdv, spv)
        N = fxi(col_xi); Nx, Nxx = d2(fxi, col_xi); Ny, Nyy = d2(feta, col_eta)
        lam = jnp.exp(llam[col_ci]); zb = jnp.exp(lzb[col_ci]); l = jnp.exp(ll[col_ci])
        e = lam*N/zb; ee = jnp.exp(-jnp.clip(e, -30, 60))
        lap = Nyy/L**2 + Nxx/XH**2
        r = lam*Ny/L - l*lam*lap + 1 - ee
        pde = jnp.mean(r*r)
        # inlet Robin residual N - l N_eta / L - g(xi) = 0 at eta = 0
        sd2, sp2 = SDc[inl_ci], SPc[inl_ci]
        f2 = lambda e_: net(a, W, b, a0, inl_xi, e_, sd2, sp2)
        N2, Ny2 = d1(f2, jnp.zeros_like(inl_xi))
        l2 = jnp.exp(ll[inl_ci]); gi = jnp.tile(inlet_g(gc, gv, gd), 35)
        r2 = N2 - l2*Ny2/L - gi
        inl = jnp.mean(r2*r2)
        # data through the instrument model
        cidx = jnp.asarray(np.repeat(np.arange(35), 9)); xs = jnp.asarray(np.tile(FV.SITES[:, 0], 35)/XH); ys = jnp.asarray(np.tile(FV.SITES[:, 1], 35))
        eta3 = (ys + U)/L
        N3 = net(a, W, b, a0, xs, eta3, SDc[cidx], SPc[cidx])
        e3 = jnp.exp(llam[cidx])*N3/jnp.exp(lzb[cidx]); ee3 = jnp.exp(-jnp.clip(e3, -30, 60))
        if budget is None:
            t3 = FV.THK*gpc[cidx]*(1 - ee3)
            m, s = marginal(t3)
            ss = jnp.sqrt(jnp.maximum(s**2 - PN.SIG_B**2, 0.05**2))
            rd = (Yv - m - bo[cidx])/ss; sbo = PN.SIG_B
        else:
            t3 = FV.THK*gpc[cidx]*(1 - ee3)*FAC*jnp.exp(bo[cidx])
            m, s = marginal(t3)
            ss = jnp.sqrt(jnp.maximum(s**2 - PN.SIG_B**2, 0.05**2) + budget['sig_d']**2)
            rd = (Yv - m)/ss; sbo = budget['sig_r']
        chi2 = jnp.sum(md*rd*rd)
        rp = (phys - jnp.asarray(PN.PRIOR_MU))/jnp.asarray(PN.PRIOR_SD)
        prior = (jnp.sum(rp**2) + jnp.sum((o/PN.SIG_O)**2) + jnp.sum((bo/sbo)**2))/ndata
        return pde, inl, chi2/ndata, prior, chi2
    def loss(th, w, *args):
        pde, inl, dat, prior, _ = terms(th, *args)
        return w[0]*pde + w[1]*inl + w[2]*dat + prior
    def term(th, k, *args): return terms(th, *args)[k]
    return loss, terms, term

def problem_args(prob):
    c = prob.col; i = prob.inl
    return (c[:, 0].astype(int), c[:, 1], c[:, 2], i[:, 0].astype(int), i[:, 1],
            np.nan_to_num(prob.Y).ravel(), prob.mask2.ravel().astype(float), float(prob.ndata))

class Trainer:
    def __init__(self, lay, budget=None):
        self.lay = lay
        loss, terms, term = make_loss(lay, budget)
        self.terms = jax.jit(terms)
        self.vg = jax.jit(jax.value_and_grad(loss))
        self.tg = jax.jit(jax.grad(term), static_argnums=1)
        self.net_sl = slice(0, lay.sl['a0'].stop)
    def train(self, th0, args, w, stages, iters, adapt, bounds=None, verbose=True):
        th = th0.copy(); lay = self.lay; w = np.array(w, float)
        jargs = tuple(jnp.asarray(v) for v in args)
        for st in range(stages):
            if adapt and st > 0:
                # gradient-norm balancing on the network weights (Wang, Teng & Perdikaris 2021), moving average, capped
                gp = np.abs(np.asarray(self.tg(jnp.asarray(th), 0, *jargs))[self.net_sl]); ref = gp.max()
                for k in (1, 2):
                    gk = np.abs(np.asarray(self.tg(jnp.asarray(th), k, *jargs))[self.net_sl])
                    w[k] = min(0.9*w[k] + 0.1*ref/(gk.mean() + 1e-12), 50.0)
            SC = np.full(lay.n, 1e-3); SC[lay.sl['phys']] = A.phys_scale
            wj = jnp.asarray(w)
            def fs(u):
                v, g = self.vg(jnp.asarray(u*SC), wj, *jargs)
                return float(v), np.asarray(g)*SC
            bs = None if bounds is None else [(None if lo is None else lo/SC[i], None if hi is None else hi/SC[i]) for i, (lo, hi) in enumerate(bounds)]
            res = minimize(fs, th/SC, jac=True, method='L-BFGS-B', bounds=bs, options=dict(maxiter=iters, maxcor=30, ftol=1e-13))
            th = res.x*SC
            if verbose:
                pde, inl, dat, prior, chi2 = [float(v) for v in self.terms(jnp.asarray(th), *jargs)]
                print(f"  stage {st}: loss {res.fun:.4g} pde {pde:.2e} inlet {inl:.2e} data chi2 {chi2:.1f} w {w[1]:.2g}/{w[2]:.2g} "
                      f"phys {np.round(th[lay.sl['phys']], 3)}", flush=True)
        return th, w

def fair_start(th, lay):
    """Same start as the FV fitter: synthetic -> log reach 22 cm, log width 1.2 cm; real -> the collapse-informed start."""
    i0 = lay.sl['phys'].start
    th[i0] = np.log(22.0); th[i0 + 1] = np.log(1.2)
    cj = os.path.join(A.out, 'collapse_summary.json')
    if 'sites' in Z and os.path.exists(cj):
        c = json.load(open(cj)); gp = np.array(c['gpc']); g175 = np.interp(175, FV.TD, gp)
        th[i0] = np.log(np.clip(c['C']/g175, 4, 70)); th[i0 + 4] = float(np.clip(c['U'], 0.5, 29))
        sl_ = np.polyfit((FV.TD - 175)/75, gp, 1); th[i0 + 5], th[i0 + 6] = sl_[1], sl_[0]
    return th

def td_mask():
    if A.td is None: return np.ones(35, bool)
    lo, hi = map(float, A.td.split('-')) if '-' in A.td else (None, None)
    tds = FV.TD[(FV.TD >= lo) & (FV.TD <= hi)] if lo is not None else np.array([float(v) for v in A.td.split(',')])
    return np.isin(TDc, tds)

def run_seed(seed, holds):
    Y = Z['Y']; base = td_mask()
    B = BU.calibrate(Y, base, CONDS) if A.budget == 'measured' else None
    if B is not None: print(f"[seed {seed}] " + BU.describe(B), flush=True)
    lay = PN.Layout(H=A.hidden, HG=6); tr = Trainer(lay, B)
    pre_f = os.path.join(A.out, f'pinnjax_pre_{A.pre_tag or A.tag}_{seed}.npy')
    rng = np.random.default_rng(seed)
    prob = PN.Problem(Y, base, lay, rng, n55=A.n55, nsat=A.nsat, nin=24, budget=B)
    t0 = time.time()
    if os.path.exists(pre_f):
        th_pre = np.load(pre_f); print(f"[seed {seed}] loaded {pre_f}", flush=True)
    else:
        th = fair_start(PN.init_params(lay, rng), lay)
        print(f"[seed {seed}] physics-only pretraining at the start parameters (no data)", flush=True)
        b = [(None, None)]*lay.n
        for i in range(lay.sl['phys'].start, lay.sl['phys'].stop): b[i] = (th[i], th[i])
        for i in range(lay.sl['o'].start, lay.sl['bo'].stop): b[i] = (0.0, 0.0)
        th_pre, _ = tr.train(th, problem_args(prob), [A.pde_w, 1.0, 0.0], A.pre_stages, A.pre_iters, adapt=False, bounds=b)
        np.save(pre_f, th_pre); print(f"[seed {seed}] pretraining {time.time()-t0:.0f} s", flush=True)
    lo = [np.log(3), np.log(0.05), -1.0, np.log(0.02), 0.0, 0.5, -1.0]; hi = [np.log(80), np.log(10), 1.0, np.log(20), 30.0, 2.5, 1.0]
    bnd = [(None, None)]*lay.n
    for k, i in enumerate(range(lay.sl['phys'].start, lay.sl['phys'].stop)): bnd[i] = (lo[k], hi[k])
    for h in holds:
        out_f = os.path.join(A.out, f'pinn_{A.tag}_s{seed}_h{h}.npz')
        if os.path.exists(out_f): print(f"[seed {seed}] {out_f} exists, skipping", flush=True); continue
        mask = base.copy()
        if h >= 0: mask[h] = False
        p = PN.Problem(Y, mask, lay, np.random.default_rng(seed), n55=A.n55, nsat=A.nsat, nin=24, budget=B)   # same points as pretraining
        t1 = time.time(); print(f"[seed {seed}] hold {h} " + (f"({CONDS[h][0]:.0f} C dep / {CONDS[h][1]:.0f} C source)" if h >= 0 else "(none: all data)"), flush=True)
        th, w = tr.train(th_pre, problem_args(p), [A.pde_w, 1.0, 1.0], A.stages, A.iters, adapt=True, bounds=bnd)
        _, _, comp = p.loss(th, parts=True)                      # report through the NumPy implementation (independent check)
        np.savez(out_f, th=th, pred=p.predict(th, list(range(35))), phys=th[lay.sl['phys']],
                 comp=np.array([comp['pde'], comp['inl'], comp['data_chi2']]), g=PN.inlet_g(th, lay, np.linspace(-1, 1, 41))[0],
                 time=time.time() - t1, w=w, seed=seed, hold=h, ndata=p.ndata, budget=A.budget,
                 sig=(B['sig'] if B else np.zeros(9)), sig_r=(B['sig_r'] if B else 0.0), sig_d=(B['sig_d'] if B else 0.0))
        print(f"[seed {seed}] hold {h} done {time.time()-t1:.0f} s: data chi2 {comp['data_chi2']:.1f} on {p.ndata}, "
              f"pde {comp['pde']:.2e}, phys {np.round(th[lay.sl['phys']], 3)}", flush=True)

def selfcheck():
    """JAX value and gradient against the NumPy implementation (whose gradients were checked by finite differences),
    for the ideal error model and for the measured one."""
    Y = Z['Y']; ok_all = True
    if 'sites' in Z: Bm = BU.calibrate(Y, np.ones(35, bool), CONDS)
    else:
        r0 = np.random.default_rng(1); sg = r0.normal(0, 0.03, 9); Bm = dict(sig=sg - sg.mean(), sig_r=0.033, sig_d=0.19, n=0)
    for label, B in (('ideal error model', None), ('measured error model', Bm)):
        lay = PN.Layout(H=A.hidden, HG=6); rng = np.random.default_rng(7)
        mask = np.ones((35, 9), bool); mask[3, 4] = False; mask[20] = False
        prob = PN.Problem(Y, mask, lay, rng, n55=60, nsat=12, nin=24, budget=B)
        th = PN.init_params(lay, rng); th[lay.sl['phys']] += rng.normal(0, 0.05, 7)
        th[lay.sl['a']] *= 0.05                                   # realistic regime: exposure e = lam N / zb of order 5
        th[lay.sl['o']] = rng.normal(0, 0.05, 35); th[lay.sl['bo']] = rng.normal(0, 0.03, 35)
        w = np.array([A.pde_w, 2.3, 0.7]); prob.w = dict(pde=w[0], inl=w[1], data=w[2])
        f_np, g_np = prob.loss(th)
        tr = Trainer(lay, B); args = tuple(jnp.asarray(v) for v in problem_args(prob))
        t0 = time.time(); f_j, g_j = tr.vg(jnp.asarray(th), jnp.asarray(w), *args); g_j = np.asarray(g_j); tc = time.time() - t0
        t0 = time.time()
        for _ in range(20): tr.vg(jnp.asarray(th), jnp.asarray(w), *args)[0].block_until_ready()
        te = (time.time() - t0)/20
        rel_f = abs(float(f_j) - f_np)/abs(f_np); rel_g = np.max(np.abs(g_j - g_np))/np.max(np.abs(g_np))
        print(f"[{label}] loss NumPy {f_np:.12g}  JAX {float(f_j):.12g}  rel diff {rel_f:.1e}")
        print(f"[{label}] grad max |JAX - NumPy| / max |grad| = {rel_g:.1e}")
        for k in ('a', 'W', 'b', 'gc', 'phys', 'o', 'bo'):
            s_ = lay.sl[k]; d = np.max(np.abs(g_j[s_] - g_np[s_]))/(np.max(np.abs(g_np[s_])) + 1e-300)
            print(f"  block {k:5s} rel diff {d:.1e}")
        print(f"  compile + first call {tc:.1f} s, then {1e3*te:.1f} ms per value-and-gradient ({prob.col.shape[0]} collocation points)")
        ok_all &= rel_f < 1e-8 and rel_g < 1e-6
    print("SELF-CHECK", "PASSED" if ok_all else "FAILED (see docs/SETUP.md, 'If a check fails')")
    return ok_all

if __name__ == '__main__':
    if A.cmd == 'selfcheck':
        sys.exit(0 if selfcheck() else 1)
    a_, b_ = (A.seeds.split('-') + [None])[:2]
    seeds = list(range(int(a_), int(b_) + 1)) if b_ is not None else [int(s) for s in A.seeds.split(',')]
    holds = []
    for tok in A.holds.split(','):           # '-1' = all data, '55' = every 55 C wafer in --td, 'all' = every wafer in --td
        if tok == '55': holds += [i for i in range(35) if td_mask()[i] and CONDS[i][1] == 55.0]
        elif tok == 'all': holds += [i for i in range(35) if td_mask()[i]]
        else: holds.append(int(tok))
    os.makedirs(A.out, exist_ok=True)
    if A.jobs > 1 and len(seeds) > 1:
        # one process per seed (JAX and fork do not mix); each process is single-threaded
        env = dict(os.environ, XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1', OMP_NUM_THREADS='1')
        argv = [a for a in sys.argv[1:]]
        def with_seed(s):
            out = []; skip = False
            for i, a in enumerate(argv):
                if skip: skip = False; continue
                if a in ('--seeds', '--jobs'): skip = True; continue
                out.append(a)
            return [sys.executable, os.path.abspath(__file__)] + out + ['--seeds', str(s), '--jobs', '1']
        running = []; queue = list(seeds)
        while queue or running:
            while queue and len(running) < A.jobs:
                s = queue.pop(0); log = open(os.path.join(A.out, f'pinn_{A.tag}_s{s}.log'), 'a')
                running.append((s, subprocess.Popen(with_seed(s), env=env, stdout=log, stderr=subprocess.STDOUT), log))
                print(f"started seed {s} -> pinn_{A.tag}_s{s}.log", flush=True)
            time.sleep(5)
            for item in list(running):
                if item[1].poll() is not None:
                    running.remove(item); item[2].close(); print(f"seed {item[0]} finished (exit {item[1].returncode})", flush=True)
    else:
        for s in seeds: run_seed(s, holds)
