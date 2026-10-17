"""FV inverse with an error budget calibrated on the saturated wafers (no new transport mechanism).

Why: on Lee et al.'s wafers the ideal error model (instrument noise ~0.05-0.1 nm, run shift 0.06 nm) is far too tight.
The saturated wafers (60-75 C source, coverage 1 everywhere) carry no transport information, so they measure the
non-transport part of the error directly:
  s_i      fixed site signature of the saturated film (relative), the same on every saturated wafer
  SIG_R    run-to-run relative scatter of the saturated film (GPC from run to run)
  SIG_D    site-level scatter left after the signature and the run level are removed
Model:  reported = instrument( THK * GPC(T) * (1 + s_i) * exp(r_run) * theta_i ) + site noise (instrument + SIG_D)
with theta from the same steady-exposure FV solver, run dose offsets o (SIG_O) and run film offsets r (SIG_R).

    python recalibrated.py fit    --td 100-200            -> recal_fit_<tag>.json, recal_per_wafer_<tag>.csv
    python recalibrated.py howo   --td 100-200 --jobs 2   -> recal_howo_<tag>.csv  (each 55 C wafer held out)
    python recalibrated.py extrap --td 100-200            -> recal_extrap_<tag>.csv (predict the other temperatures)
"""
import os, sys, json, argparse, time, warnings, csv
import numpy as np, multiprocessing
warnings.filterwarnings('ignore')
ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=['fit', 'howo', 'extrap', 'compare'])
ap.add_argument('--npz', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'real_Y.npz'))
ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results'))
ap.add_argument('--td', default='100-200'); ap.add_argument('--jobs', type=int, default=2); ap.add_argument('--start')
ap.add_argument('--tag', default='REALM', help='compare: PINN files pinn_<tag>_s<seed>_h<hold>.npz'); ap.add_argument('--seeds', default='0-9')
A = ap.parse_args()
Z = np.load(A.npz, allow_pickle=True); os.environ['ALD_NCYC'] = str(float(Z['ncyc']))
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(k, '1')
import fvmodel as FV
FV.SITES[:] = Z['sites']
FV.require_ncyc(Z['ncyc'])
import instrument as INS, fit as F
from scipy.optimize import least_squares
Y = Z['Y']; CONDS = FV.CONDS; TD = FV.TD
lo, hi = map(float, A.td.split('-')); TDS = TD[(TD >= lo) & (TD <= hi)]
BASE = np.array([c[0] in TDS for c in CONDS]); TAG = f"td{int(lo)}-{int(hi)}"
MP = multiprocessing.get_context('fork')

def calibrate(mask):
    """Signature, run scatter and site scatter from the saturated wafers inside the training mask (budget.py)."""
    import budget as BU
    b = BU.calibrate(Y, mask, CONDS)
    sat = [i for i in range(35) if mask[i] and CONDS[i][1] >= 60 and np.isfinite(Y[i]).all()]
    return b['sig'], b['sig_r'], b['sig_d'], sat

def site_sd(t, sig_d):
    m, s, dm = INS.marginal(t)
    return np.sqrt(np.maximum(s**2 - F.SIG_B**2, 0.05**2) + sig_d**2)

def fit(mask, p0, cal, max_nfev=40):
    sig, sig_r, sig_d, _ = cal; fac = 1 + sig
    mask2 = np.repeat(mask[:, None], 9, 1) & np.isfinite(Y); Yn = np.nan_to_num(Y)
    nR = 35; cache = {}
    def unpack(z): return z[:9], z[9:9+nR], z[9+nR:]
    def model(z):
        k = z.tobytes()
        if k not in cache:
            p, o, r = unpack(z); cache.clear(); cache[k] = FV.forward(p, o, sens=True)
        return cache[k]
    def res(z):
        p, o, r = unpack(z); T, S, So = model(z); t = T*fac*np.exp(r)[:, None]
        m, s, dm = INS.marginal(t); sd = site_sd(t, sig_d)
        return np.concatenate([((Yn - m)/sd)[mask2], o/F.SIG_O, r/sig_r, (p - F.PRIOR_MU)/F.PRIOR_SD])
    def jac(z):
        p, o, r = unpack(z); T, S, So = model(z); g = fac*np.exp(r)[:, None]; t = T*g
        m, s, dm = INS.marginal(t); sd = site_sd(t, sig_d); w = -dm/sd
        Jp = (w*g)[:, :, None]*S
        Jo = np.zeros((nR, 9, nR)); Jo[np.arange(nR), :, np.arange(nR)] = w*g*So
        Jr = np.zeros((nR, 9, nR)); Jr[np.arange(nR), :, np.arange(nR)] = w*t
        Jd = np.concatenate([Jp, Jo, Jr], 2)[mask2]
        Jpri = np.zeros((2*nR, 9 + 2*nR)); Jpri[:nR, 9:9+nR] = np.eye(nR)/F.SIG_O; Jpri[nR:, 9+nR:] = np.eye(nR)/sig_r
        Jpp = np.zeros((9, 9 + 2*nR)); Jpp[:, :9] = np.diag(1/F.PRIOR_SD)
        return np.vstack([Jd, Jpri, Jpp])
    z0 = np.concatenate([np.clip(p0, F.LB + 1e-6, F.UB - 1e-6), np.zeros(2*nR)])
    lb = np.concatenate([F.LB, -0.5*np.ones(nR), -0.3*np.ones(nR)]); ub = np.concatenate([F.UB, 0.5*np.ones(nR), 0.3*np.ones(nR)])
    sol = least_squares(res, z0, jac=jac, bounds=(lb, ub), method='trf', x_scale='jac', max_nfev=max_nfev)
    p, o, r = unpack(sol.x); J = sol.jac; cov = np.linalg.pinv(J.T@J)
    rr = res(sol.x); nd = int(mask2.sum())
    return dict(p=p, o=o, r=r, cov=cov, chi2=float(np.sum(rr[:nd]**2)), ndata=nd, nfev=sol.nfev, cost=float(2*sol.cost))

def per_wafer(fr, cal):
    sig, sig_r, sig_d, _ = cal
    T = FV.forward(fr['p'], fr['o']); t = T*(1 + sig)*np.exp(fr['r'])[:, None]
    m, s, dm = INS.marginal(t); return (Y - m)/site_sd(t, sig_d), m

def start():
    f = A.start or os.path.join(A.out, f'fit_{TAG}.json')
    if os.path.exists(f): d = json.load(open(f)); return np.array([d['params'][n] for n in FV.NAMES])
    p0 = F.PRIOR_MU.copy(); p0[0] = np.log(22); p0[1] = np.log(1.2); return p0

def predict(fr, cal, h, rng, n=200):
    """90 % interval for wafer h: parameters (linearised), its own dose offset and film offset (sampled), site scatter."""
    sig, sig_r, sig_d, _ = cal; c = CONDS[h]
    T_, S_, So_ = FV.forward(fr['p'], np.zeros(35), sens=True, conds=[c])
    t0 = T_[0]*(1 + sig); m0, s0, dm0 = INS.marginal(t0)
    Jt = S_[0]*(1 + sig)[:, None]; var_par = np.einsum('ij,jk,ik->i', Jt, fr['cov'][:9, :9], Jt)*dm0**2
    sims = []
    for k in range(n):
        o = np.zeros(35); o[h] = rng.normal(0, F.SIG_O)
        t = FV.forward(fr['p'], o, conds=[c])[0]*(1 + sig)*np.exp(rng.normal(0, sig_r))
        m, s, _ = INS.marginal(t); sims.append(m + rng.normal(0, 1, 9)*np.sqrt(site_sd(t, sig_d)**2 + var_par))
    sims = np.array(sims); return np.median(sims, 0), np.percentile(sims, 5, 0), np.percentile(sims, 95, 0)

def _howo(h):
    mask = BASE.copy(); mask[h] = False; cal = calibrate(mask)
    fr = fit(mask, start(), cal)
    med, lo_, hi_ = predict(fr, cal, h, np.random.default_rng(100 + h))
    return dict(h=h, med=med, lo=lo_, hi=hi_, chi2=fr['chi2'], nd=fr['ndata'])

def table(rows, name):
    with open(os.path.join(A.out, name), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['Td', 'Tp', 'rmse_nm', 'inside90', 'mean_width_nm'] + [f'{k}{i}' for k in ('obs', 'pred', 'lo', 'hi') for i in range(9)])
        allin = []
        for r in rows:
            h = r['h']; obs = Y[h]; ok = np.isfinite(obs); ins = ((obs >= r['lo']) & (obs <= r['hi']))[ok]; allin += list(ins)
            rm = np.sqrt(np.mean((obs - r['med'])[ok]**2)); wd = np.mean((r['hi'] - r['lo'])[ok])
            w.writerow([CONDS[h][0], CONDS[h][1], round(rm, 3), round(ins.mean(), 3), round(wd, 3)] + [round(v, 3) for k in ('obs', 'med', 'lo', 'hi') for v in (obs if k == 'obs' else r[k])])
            print(f"  {CONDS[h][0]:.0f} C dep / {CONDS[h][1]:.0f} C source: RMSE {rm:.2f} nm, inside 90 % {ins.mean():.2f}, width {wd:.2f} nm", flush=True)
    print(f"  overall: {np.mean(allin):.2f} of {len(allin)} sites inside their 90 % interval")

if __name__ == '__main__':
    t0 = time.time()
    if A.cmd == 'fit':
        cal = calibrate(BASE); sig, sig_r, sig_d, sat = cal
        print(f"calibration from {len(sat)} saturated wafers in {TDS.tolist()} C: signature (%) {np.round(100*sig, 2).tolist()}, "
              f"run scatter {100*sig_r:.2f} %, site scatter {sig_d:.3f} nm", flush=True)
        fr = fit(BASE, start(), cal); sd = np.sqrt(np.diag(fr['cov'])[:9])
        rz, mfit = per_wafer(fr, cal); per = np.nansum(rz**2, 1)
        print(f"[{TAG}] data chi2 {fr['chi2']:.1f} on {fr['ndata']} sites ({fr['nfev']} evaluations, {time.time()-t0:.0f} s)")
        for k, nm in enumerate(FV.NAMES): print(f"  {nm:8s} {fr['p'][k]:8.3f} +- {sd[k]:.3f}")
        fl = [(f"{CONDS[i][0]:.0f}/{CONDS[i][1]:.0f}", round(float(per[i]), 1)) for i in range(35) if BASE[i] and per[i] > 27.88]
        print(f"  wafers flagged (chi2 > 27.9 on 9 sites): {fl if fl else 'none'}")
        print("  run film offsets r (%), rows Td, columns Tp 55..75:")
        for td in TDS: print(f"    {td:.0f} C: " + "  ".join(f"{100*fr['r'][CONDS.index((td, tp))]:+5.1f}" for tp in FV.TP))
        json.dump(dict(params=dict(zip(FV.NAMES, fr['p'].tolist())), sd=dict(zip(FV.NAMES, sd.tolist())), chi2=fr['chi2'], ndata=fr['ndata'],
                       signature=sig.tolist(), sig_r=sig_r, sig_d=sig_d, r=fr['r'].tolist(), o=fr['o'].tolist(),
                       corr=(fr['cov'][:9, :9]/np.outer(sd, sd)).tolist()), open(os.path.join(A.out, f'recal_fit_{TAG}.json'), 'w'), indent=1)
        with open(os.path.join(A.out, f'recal_per_wafer_{TAG}.csv'), 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['Td', 'Tp', 'used', 'chi2_9sites', 'flag'] + [f'res{i}' for i in range(9)])
            for i, c in enumerate(CONDS): w.writerow([c[0], c[1], int(BASE[i]), round(float(per[i]), 2), int(per[i] > 27.88)] + list(np.round(rz[i], 3)))
    elif A.cmd == 'howo':
        holds = [i for i in range(35) if BASE[i] and CONDS[i][1] == 55.0]
        with MP.Pool(A.jobs) as pool: rows = pool.map(_howo, holds)
        table(rows, f'recal_howo_{TAG}.csv')
    elif A.cmd == 'extrap':
        cal = calibrate(BASE); fr = fit(BASE, start(), cal); rng = np.random.default_rng(5)
        rows = []
        for h in [i for i in range(35) if not BASE[i]]:
            med, lo_, hi_ = predict(fr, cal, h, rng); rows.append(dict(h=h, med=med, lo=lo_, hi=hi_))
        print(f"[extrap from {TAG}] data chi2 {fr['chi2']:.1f} on {fr['ndata']}")
        table(rows, f'recal_extrap_{TAG}.csv')
    if A.cmd != 'compare': print(f"done in {time.time()-t0:.0f} s")

def compare():
    """FV vs PINN on the same held-out 55 C wafers, both under the measured budget. PINN interval: ensemble spread + the
    held-out run's dose offset (through the FV solver at the member's own physics and learned inlet) + film offset + site noise."""
    import glob
    from scipy.interpolate import interp1d
    sig, sig_r, sig_d, _ = calibrate(BASE)
    fvf = os.path.join(A.out, f'recal_howo_{TAG}.csv')
    if not os.path.exists(fvf): sys.exit(f"run `python recalibrated.py howo --td {A.td}` first")
    FVR = {}
    for r in csv.DictReader(open(fvf)):
        h = CONDS.index((float(r['Td']), float(r['Tp'])))
        FVR[h] = {k: np.array([float(r[f'{k}{i}']) for i in range(9)]) for k in ('pred', 'lo', 'hi')}
    a, b = (A.seeds.split('-') + [None])[:2]; seeds = list(range(int(a), int(b) + 1)) if b else [int(v) for v in A.seeds.split(',')]
    rng = np.random.default_rng(11); out = []
    for h, r in sorted(FVR.items()):
        obs = Y[h]; ok = np.isfinite(obs)
        out.append(('FV', h, np.sqrt(np.mean((obs - r['pred'])[ok]**2)), np.mean(((obs >= r['lo']) & (obs <= r['hi']))[ok]), np.mean((r['hi'] - r['lo'])[ok]), 1, np.nan))
        zs = [np.load(f) for s_ in seeds for f in glob.glob(os.path.join(A.out, f'pinn_{A.tag}_s{s_}_h{h}.npz'))]
        if not zs: continue
        best = min(float(z['comp'][2]) for z in zs); good = [z for z in zs if float(z['comp'][2]) <= 1.25*best]
        preds, sims = [], []
        for z in good:
            ph = z['phys']; pfv = np.array([ph[0], ph[1], ph[2], ph[3], ph[4], 0, 0, ph[5], ph[6]])
            gf = interp1d(np.linspace(-1, 1, len(z['g'])), z['g'], fill_value='extrapolate')
            D = []
            for k in range(40):
                o = np.zeros(35); o[h] = rng.normal(0, F.SIG_O)
                D.append(FV.forward(pfv, o, gfun=lambda x, gf=gf: gf(x), conds=[CONDS[h]])[0]*(1 + sig))
            D = np.array(D); D = D - np.median(D, 0) + z['pred'][h]; D = D*np.exp(rng.normal(0, sig_r, (len(D), 1)))
            preds.append(z['pred'][h]); sims.append(D)
        preds = np.array(preds); D = np.concatenate(sims); ens = preds.var(0, ddof=1) if len(preds) > 1 else np.zeros(9)
        m, s, _ = INS.marginal(D); sm = m + rng.normal(0, 1, m.shape)*np.sqrt(site_sd(D, sig_d)**2 + ens)
        lo_, hi_, med = np.percentile(sm, 5, 0), np.percentile(sm, 95, 0), np.median(sm, 0)
        pde = np.median([float(z['comp'][0]) for z in good])
        out.append(('PINN', h, np.sqrt(np.mean((obs - med)[ok]**2)), np.mean(((obs >= lo_) & (obs <= hi_))[ok]), np.mean((hi_ - lo_)[ok]), len(good), pde))
    with open(os.path.join(A.out, f'recal_compare_{A.tag}.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['method', 'Td', 'Tp', 'rmse_nm', 'inside90', 'mean_width_nm', 'members', 'pinn_pde_residual'])
        for row in out:
            w.writerow([row[0], CONDS[row[1]][0], CONDS[row[1]][1], round(row[2], 3), round(row[3], 3), round(row[4], 3), row[5], row[6]])
            print(f"{row[0]:5s} {CONDS[row[1]][0]:.0f} C dep / {CONDS[row[1]][1]:.0f} C source: RMSE {row[2]:.2f} nm, inside 90 % {row[3]:.2f}, "
                  f"width {row[4]:.2f} nm" + (f" ({row[5]} members, PDE residual {row[6]:.3f})" if row[0] == 'PINN' else ''), flush=True)

if __name__ == '__main__' and A.cmd == 'compare':
    compare()
