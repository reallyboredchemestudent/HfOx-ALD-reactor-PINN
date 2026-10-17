"""Real-wafer stage: everything the steady-exposure model claims, tested on Lee et al.'s 315 measured sites.

  python run_real.py collapse                  mass-balance collapse of the 55 C fronts (model-light; no PDE)
  python run_real.py fit      [--td 100-250]   FV adjoint inverse with Laplace uncertainty, per-wafer misfit flags
  python run_real.py howo     [--td 100-200]   hold-one-wafer-out: refit without each 55 C wafer, predict it
  python run_real.py extrap   --train-td 100-200 [--train-tp 55-75]   fit on part of the design, predict the rest
  python run_real.py compare  [--tag REAL]     FV vs JAX-PINN ensemble on the held-out wafers (after pinn_jax.py)

Common options: --npz ../results/real_Y.npz (from data_real.py), --out ../results, --jobs N (parallel fits).
Every command writes CSV/JSON into --out and prints a short summary; send the whole results folder back.
"""
import os, sys, json, argparse, time, warnings
import numpy as np, multiprocessing
warnings.filterwarnings('ignore')
MP = multiprocessing.get_context('fork')   # workers inherit the parsed options and the site positions

ap = argparse.ArgumentParser()
ap.add_argument('cmd', choices=['collapse', 'fit', 'howo', 'extrap', 'compare'])
ap.add_argument('--npz', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'real_Y.npz'))
ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results'))
ap.add_argument('--td', default=None, help='deposition temperatures used, e.g. 100-200 or 100,150,200')
ap.add_argument('--tp', default=None, help='precursor temperatures used, e.g. 55-75')
ap.add_argument('--regime', default='100-200', help='collapse: temperatures that define the mass-balance fit')
ap.add_argument('--train-td', default='100-200'); ap.add_argument('--train-tp', default='55-75')
ap.add_argument('--holds', default='55', help="howo: '55' = the 55 C wafers in --td, 'all' = every wafer in --td, or run indices 0,5,...")
ap.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2)//2))
ap.add_argument('--tag', default='REAL'); ap.add_argument('--seeds', default='0-9')
ap.add_argument('--quick', action='store_true', help='coarse grid and few iterations (smoke test only)')
A = ap.parse_args()

Z = np.load(A.npz, allow_pickle=True)
os.environ['ALD_NCYC'] = str(float(Z['ncyc']))           # must be set before the model modules are imported
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(k, '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fvmodel as FV
FV.SITES[:] = Z['sites']                                  # real site positions (in place: every module sees them)
import instrument as INS
import fit as F
from scipy.optimize import least_squares
Y = Z['Y']; CONDS = FV.CONDS; TD, TP = FV.TD, FV.TP
os.makedirs(A.out, exist_ok=True)
NX, NY = (20, 64) if A.quick else (30, 96)
NFEV = 6 if A.quick else 40
CHI2_FLAG = 27.88          # 9 sites, p = 0.001

def cname(c): return f"{float(c[0]):.0f} C dep / {float(c[1]):.0f} C source"

def temps(spec, grid):
    if spec is None: return grid.copy()
    if '-' in spec: lo, hi = map(float, spec.split('-')); return grid[(grid >= lo) & (grid <= hi)]
    return np.array([float(v) for v in spec.split(',')])
def run_mask(tds, tps): return np.array([(c[0] in tds) and (c[1] in tps) for c in CONDS])
def save_json(name, obj):
    with open(os.path.join(A.out, name), 'w') as f: json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))

# ---------------------------------------------------------------------------------------------------------------
def t_true_from_reported(r):
    """Invert the instrument's mean curve (reported -> true thickness), for saturated films."""
    tt = np.linspace(3, INS.TMAX, 400); return np.interp(r, INS.marginal(tt)[0], tt)   # monotone above 3 nm

def collapse():
    """Mass balance alone predicts: at fixed dose (55 C source), reach x GPC is the same at every deposition temperature.
    Per temperature: saturated film from the 75 C wafer; front position and width from the 55 C wafer through the
    instrument model (thin films are biased), with a cross-flow front shape shared by all temperatures.
    Then y_front = C / GPC - U on the regime temperatures; the others are predictions."""
    S = FV.SITES; reg = temps(A.regime, TD)
    rows55 = [CONDS.index((t, 55.0)) for t in TD]; rows75 = [CONDS.index((t, 75.0)) for t in TD]
    n = len(TD)
    def unpack(q):     # front positions, front widths, shared cross-flow front shape, saturated films (per temperature)
        return q[:n], np.exp(q[n:2*n]), q[2*n], q[2*n+1], np.exp(q[2*n+2:3*n+2])
    def theta(yf, w, c1, c2, k):
        yloc = yf[k] + c1*S[:, 0] + c2*(S[:, 0]**2 - np.mean(S[:, 0]**2)); return 1/(1 + np.exp((S[:, 1] - yloc)/w[k]))
    def res(q):
        yf, w, c1, c2, ts = unpack(q); out = []
        for k in range(n):
            for row, th in ((rows55[k], theta(yf, w, c1, c2, k)), (rows75[k], np.ones(9))):
                m, s, _ = INS.marginal(ts[k]*th); ok = np.isfinite(Y[row])
                out.append(((Y[row] - m)/s)[ok])
        return np.concatenate(out)
    ts0 = np.array([t_true_from_reported(np.nanmedian(Y[r])) for r in rows75])
    q0 = np.r_[np.full(n, 7.5), np.log(np.full(n, 0.8)), 0.0, 0.0, np.log(ts0)]
    best = None
    for y0 in (5.0, 7.5, 10.0):                               # multi-start on the front position
        q0[:n] = y0; r = least_squares(res, q0, method='trf')
        if best is None or r.cost < best.cost: best = r
    J = best.jac; cov = np.linalg.pinv(J.T@J)*max(1.0, 2*best.cost/max(1, J.shape[0] - J.shape[1]))
    yf, w, c1, c2, ts = unpack(best.x); sd = np.sqrt(np.diag(cov))
    yf_sd = sd[:n]; w_sd = w*sd[n:2*n]
    # a front is resolved only if it sits inside the site window and its width is not collapsing onto one site row
    thmin = np.array([theta(yf, w, c1, c2, k).min() for k in range(n)]); thmax = np.array([theta(yf, w, c1, c2, k).max() for k in range(n)])
    resolved = (thmin < 0.8) & (thmax > 0.2) & (w > 0.25*np.median(w)) & (yf_sd < 0.5)
    gpc = ts*10/float(Z['ncyc'])                               # A/cycle from the saturated wafer
    sat_cv = np.array([np.nanstd(Y[r])/np.nanmean(Y[r]) for r in rows75])
    # mass-balance line on the regime temperatures. Each wafer also carries its own unknown dose offset (the model's
    # run-level log-dose scatter SIG_O), which moves its front by reach*SIG_O: that is part of the expected scatter.
    ir = np.isin(TD, reg) & resolved; x = 1/gpc
    if ir.sum() < 3: sys.exit(f"collapse: only {ir.sum()} resolved fronts in the regime {reg.tolist()}; widen --regime")
    C, U = 0.0, 0.0
    for it in range(20):
        sd_tot = np.sqrt(yf_sd**2 + (np.maximum(yf + U, 1.0)*F.SIG_O)**2)
        Wt = 1/sd_tot[ir]**2; Amat = np.c_[x[ir], -np.ones(ir.sum())]
        coef, *_ = np.linalg.lstsq(Amat*np.sqrt(Wt)[:, None], yf[ir]*np.sqrt(Wt), rcond=None)
        if np.allclose(coef, (C, U), atol=1e-6): break
        C, U = coef
    covc = np.linalg.inv((Amat*Wt[:, None]).T@Amat)
    pred = C*x - U; chi2 = np.sum(((yf - pred)/sd_tot)[ir]**2); dof = ir.sum() - 2
    reach = yf + U; prod = reach*gpc
    excess = np.mean((yf - pred)[ir]**2 - yf_sd[ir]**2)
    unres = [f"{t:.0f}" for t, r_ in zip(TD, resolved) if not r_]
    sig_dose_implied = np.sqrt(max(excess, 0)*ir.sum()/max(dof, 1))/np.mean(reach[ir])
    cv = np.std(prod[ir])/np.mean(prod[ir]); maxdev = np.max(np.abs(prod[ir]/np.mean(prod[ir]) - 1))
    lines = [f"front fit: data chi2 {2*best.cost:.1f} for {J.shape[0]} sites ({J.shape[0]-J.shape[1]} dof)",
             f"cross-flow front shape: tilt {c1:+.3f} cm/cm, curvature {c2:+.4f} cm/cm^2 (shared by all temperatures)",
             f"mass-balance line on {reg.tolist()} C: y_front = C/GPC - U with C = {C:.2f} +- {np.sqrt(covc[0,0]):.2f} cm*A/cycle, "
             f"U = {U:.2f} +- {np.sqrt(covc[1,1]):.2f} cm (corr {covc[0,1]/np.sqrt(covc[0,0]*covc[1,1]):+.2f}); chi2 {chi2:.2f} on {dof} dof "
             f"(front sd from the fit plus run dose scatter {F.SIG_O:.2f} in log dose)",
             f"run-to-run log-dose scatter implied by the regime residuals: {sig_dose_implied:.3f} (model assumes {F.SIG_O:.2f})",
             f"reach x GPC over the regime: CV {100*cv:.1f} %, max deviation {100*maxdev:.1f} % ({np.max(np.abs(reach - C/gpc)[ir]):.2f} cm)",
             "fronts not resolved on the wafer (transition outside the sites, or width/position degenerate; excluded from the line): " + (", ".join(unres) if unres else "none"),
             "U here is an effective offset (logistic front vs mass-balance reach), not the PDE model's U",
             "Td   GPC(A/cyc) sat-wafer CV  front y (cm)    width (cm)    reach (cm)  reach*GPC  predicted front  dev (total sd)"]
    for k, t in enumerate(TD):
        sdp = np.sqrt(sd_tot[k]**2 + np.array([x[k], -1])@covc@np.array([x[k], -1]))
        lines.append(f"{t:4.0f} {gpc[k]:9.3f} {100*sat_cv[k]:8.1f} %  {yf[k]:6.2f} +- {yf_sd[k]:4.2f}  {w[k]:5.2f} +- {w_sd[k]:4.2f}  "
                     f"{reach[k]:7.2f}   {prod[k]:7.2f}   {pred[k]:8.2f}      {(yf[k]-pred[k])/sdp:+5.1f}"
                     + ('   (not resolved)' if not resolved[k] else '' if ir[k] else '   (prediction)'))
    rep = "\n".join(lines); print(rep); open(os.path.join(A.out, 'collapse_summary.txt'), 'w').write(rep + "\n")
    save_json('collapse_summary.json', dict(Td=TD, gpc=gpc, front=yf, front_sd=yf_sd, front_sd_total=sd_tot, sig_dose_implied=sig_dose_implied,
                                            width=w, width_sd=w_sd, C=C, U=U, cov_CU=covc,
                                            chi2=chi2, dof=dof, regime=reg, resolved=resolved, cv=cv, maxdev=maxdev, tilt=c1, curv=c2, t_sat=ts, sat_cv=sat_cv))
    # per-site table for the plot / later re-plotting
    import csv
    with open(os.path.join(A.out, 'collapse_points.csv'), 'w', newline='') as f:
        wr = csv.writer(f); wr.writerow(['Td', 'site', 'x_cm', 'y_cm', 'reported_nm', 'coverage', 's_norm', 'thin_lt5nm', 'in_regime'])
        for k, t in enumerate(TD):
            yeff = S[:, 1] - c1*S[:, 0] - c2*(S[:, 0]**2 - np.mean(S[:, 0]**2))
            s = (yeff + U)/(C/gpc[k]); cov_ = Y[rows55[k]]/INS.marginal(ts[k])[0]
            for i in range(9): wr.writerow([t, i, S[i, 0], S[i, 1], Y[rows55[k], i], cov_[i], s[i], int(Y[rows55[k], i] < 5), int(t in reg)])
    plot_collapse(yf, sd_tot, w, gpc, C, U, c1, c2, ts, reg, rows55, resolved)

def plot_collapse(yf, sd_tot, w, gpc, C, U, c1, c2, ts, reg, rows55, resolved):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    # validated ordinal ramps (dataviz validator, light surface): regime = blue, outside = orange
    BLUE = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#104281']; ORNG = ['#eb6834', '#a8401a']
    SURF, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
    ir = [t in reg for t in TD]; nr, no = sum(ir), len(TD) - sum(ir)
    pick = lambda pal, k: [pal[int(round(i*(len(pal)-1)/max(1, k-1)))] for i in range(k)] if k <= len(pal) else [pal[-1]]*k
    cols = iter(pick(BLUE, nr)); colo = iter(pick(ORNG, no)); color = [next(cols) if r else next(colo) for r in ir]
    marks = ['o' if r else m for r, m in zip(ir, ['o']*len(TD))]
    if no > 2:      # more outside temperatures than validated orange steps: shape carries the order too
        it = iter(['s', '^', 'D', 'v', 'P']); marks = ['o' if r else next(it) for r in ir]
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2,
                         'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.facecolor': SURF, 'figure.facecolor': SURF})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw=dict(width_ratios=[1, 1.35]))
    S = FV.SITES
    # (a) front position against 1/GPC: mass balance is a straight line
    xg = np.linspace(min(1/gpc)*0.97, max(1/gpc)*1.03, 50)
    a1.plot(xg, C*xg - U, color=MUTED, lw=2, zorder=1, label='mass balance: reach × GPC = const')
    for k, t in enumerate(TD):
        if not resolved[k]: continue
        a1.errorbar(1/gpc[k], yf[k], yerr=sd_tot[k], fmt=marks[k], ms=8, color=color[k], mec=SURF, mew=1.5, ecolor=color[k], elinewidth=2, zorder=3)
        a1.annotate(f"{t:.0f} °C", (1/gpc[k], yf[k]), textcoords='offset points', xytext=(8, 5) if k % 2 == 0 else (8, -14),
                    ha='left', fontsize=9, color=INK2)
    a1.set_xlabel('1 / GPC  (cycle / Å)'); a1.set_ylabel('front position on the wafer, y  (cm)')
    a1.set_title('(a) Front position vs 1/GPC, 55 °C source', loc='left', fontsize=10.5, color=INK)
    a1.grid(color=GRID, lw=0.8); a1.set_axisbelow(True); [a1.spines[s].set_visible(False) for s in ('top', 'right')]
    a1.legend(frameon=False, fontsize=9, loc='upper left', labelcolor=INK2)
    if not all(resolved):
        a1.text(0.99, 0.02, 'front not on the wafer: ' + ', '.join(f"{t:.0f} °C" for t, r_ in zip(TD, resolved) if not r_),
                transform=a1.transAxes, ha='right', va='bottom', fontsize=8.5, color=INK2)
    # (b) coverage against distance / predicted reach: one master curve if the collapse holds
    wr = np.mean([w[k]/(C/gpc[k]) for k in range(len(TD)) if ir[k] and resolved[k]])
    yeff = S[:, 1] - c1*S[:, 0] - c2*(S[:, 0]**2 - np.mean(S[:, 0]**2))
    sall = np.concatenate([(yeff + U)/(C/gpc[k]) for k in range(len(TD))]); s_lo, s_hi = sall.min() - 0.08, sall.max() + 0.08
    sg = np.linspace(s_lo, s_hi, 200); a2.plot(sg, 1/(1 + np.exp((sg - 1)/wr)), color=MUTED, lw=2, zorder=1, label='master curve (regime mean width)')
    for k, t in enumerate(TD):
        s = (yeff + U)/(C/gpc[k]); cv = Y[rows55[k]]/INS.marginal(ts[k])[0]; thin = Y[rows55[k]] < 5   # reported / reported saturated
        a2.scatter(s[~thin], cv[~thin], s=46, marker=marks[k], color=color[k], edgecolors=SURF, linewidths=1.2, zorder=3,
                   label=f"{t:.0f} °C" + ('' if ir[k] else ' (prediction)'))
        a2.scatter(s[thin], cv[thin], s=46, marker=marks[k], facecolors='none', edgecolors=color[k], linewidths=1.6, zorder=3)
    a2.scatter([], [], s=46, facecolors='none', edgecolors=MUTED, linewidths=1.6, label='hollow: film < 5 nm (ellipsometry-biased)')
    a2.set_xlabel('distance from the reactive edge / predicted reach,  (y + U) · GPC / C'); a2.set_ylabel('coverage = thickness / saturated film')
    a2.set_title('(b) Collapsed coverage profiles', loc='left', fontsize=10.5, color=INK)
    a2.set_ylim(-0.05, max(1.15, np.nanmax(Y[rows55]/INS.marginal(ts)[0][:, None]) + 0.05)); a2.set_xlim(s_lo, s_hi)
    a2.grid(color=GRID, lw=0.8); a2.set_axisbelow(True); [a2.spines[s].set_visible(False) for s in ('top', 'right')]
    a2.legend(frameon=False, fontsize=8.5, loc='center left', bbox_to_anchor=(1.0, 0.5), labelcolor=INK2)
    fig.tight_layout(); fig.savefig(os.path.join(A.out, 'collapse_real.png'), dpi=180); plt.close(fig)
    print(f"wrote {os.path.join(A.out, 'collapse_real.png')}")

# ---------------------------------------------------------------------------------------------------------------
def init_params():
    """Start the FV fit from the model-light collapse result when available (reach at 175 C, U, GPC law)."""
    p0 = F.PRIOR_MU.copy(); p0[0] = np.log(22); p0[1] = np.log(1.2)
    f = os.path.join(A.out, 'collapse_summary.json')
    sat = np.array([np.nanmedian(Y[CONDS.index((t, 75.0))]) for t in TD])
    g = np.array([t_true_from_reported(s) for s in sat])*10/float(Z['ncyc'])
    sl = np.polyfit((TD - 175)/75, g, 1); p0[7], p0[8] = sl[1], sl[0]
    if os.path.exists(f):
        c = json.load(open(f)); gp175 = np.interp(175, TD, np.array(c['gpc']))
        p0[0] = np.log(np.clip(c['C']/gp175, 4, 70)); p0[4] = float(np.clip(c['U'], 0.5, 29))
    return np.clip(p0, F.LB + 1e-6, F.UB - 1e-6)

def _fit_job(args):
    Ysel, p0, mask = args
    return F.fit(Ysel, p0, mask=mask, Nx=NX, Ny=NY, max_nfev=NFEV)

def multistart_fit(mask, label, pool=None):
    p0 = init_params(); starts = [p0]
    for d in ((0.3, 0.0, 0.0), (-0.3, 0.4, 3.0)):
        q = p0.copy(); q[0] += d[0]; q[1] += d[1]; q[4] = np.clip(q[4] + d[2], 0.5, 29); starts.append(q)
    jobs = [(Y, s, mask) for s in starts]
    rs = pool.map(_fit_job, jobs) if pool is not None else [_fit_job(j) for j in jobs]
    best = min(rs, key=lambda r: r['cost'])
    print(f"[{label}] multistart costs {[round(float(r['cost']), 1) for r in rs]} -> best {best['cost']:.1f}", flush=True)
    return best, rs

def residuals(r):
    T = FV.forward(r['p'], r['o'], Nx=NX, Ny=NY); m, s, dm = INS.marginal(T)
    return (Y - m - r['b'][:, None])/np.sqrt(np.maximum(s**2 - F.SIG_B**2, 0.05**2)), m + r['b'][:, None]

def fit_cmd():
    Pool = MP.Pool
    tds, tps = temps(A.td, TD), temps(A.tp, TP); mask = run_mask(tds, tps)
    tag = f"td{int(tds.min())}-{int(tds.max())}" + ("" if A.tp is None else f"_tp{int(tps.min())}-{int(tps.max())}")
    t0 = time.time()
    with Pool(min(3, A.jobs)) as pool: r, allr = multistart_fit(mask, tag, pool)
    sd = np.sqrt(np.diag(r['cov'])[:9]); C9 = r['cov'][:9, :9]; corr = C9/np.outer(sd, sd)
    res, mfit = residuals(r)
    per = np.nansum(np.where(np.isfinite(res), res, 0)**2, 1)
    out = dict(params=dict(zip(FV.NAMES, r['p'])), sd=dict(zip(FV.NAMES, sd)), corr=corr, names=FV.NAMES, chi2_data=r['chi2_data'],
               ndata=r['ndata'], dose_offsets=r['o'], instrument_offsets=r['b'], starts_cost=[a['cost'] for a in allr], seconds=time.time() - t0,
               used_runs=[CONDS[i] for i in range(35) if mask[i]], grid=[NX, NY], ncyc=float(Z['ncyc']))
    save_json(f'fit_{tag}.json', out)
    import csv
    with open(os.path.join(A.out, f'per_wafer_{tag}.csv'), 'w', newline='') as f:
        wr = csv.writer(f); wr.writerow(['Td', 'Tp', 'used_in_fit', 'chi2_9sites', 'flag_p001', 'worst_site', 'worst_resid_sd', 'dose_offset', 'instr_offset_nm']
                                        + [f'res{i}' for i in range(9)] + [f'obs{i}' for i in range(9)] + [f'fit{i}' for i in range(9)])
        for i, c in enumerate(CONDS):
            k = int(np.nanargmax(np.abs(res[i]))) if np.isfinite(res[i]).any() else -1
            wr.writerow([c[0], c[1], int(mask[i]), round(per[i], 2), int(per[i] > CHI2_FLAG), k, round(res[i, k], 2) if k >= 0 else '',
                         round(r['o'][i], 4), round(r['b'][i], 4)] + list(np.round(res[i], 3)) + list(np.round(Y[i], 3)) + list(np.round(mfit[i], 3)))
    with open(os.path.join(A.out, f'groups_{tag}.csv'), 'w', newline='') as f:
        wr = csv.writer(f); wr.writerow(['Td', 'Tp', 'reach_lam_cm', 'front_width_zb_cm', 'dispersion_l_cm', 'gpc_A', 'delta', 'Da', 'Pe', 'R_reach_frac'])
        L = 15 + r['p'][4]
        for i, (td, tp) in enumerate(CONDS):
            lam, zb, l, gpc = FV.groups(r['p'], td, tp, r['o'][i])
            wr.writerow([td, tp] + list(np.round([lam, zb, l, gpc, lam/zb, L/zb, L/l, lam/L], 4)))
    print(f"[{tag}] data chi2 {r['chi2_data']:.1f} on {r['ndata']} sites, {time.time()-t0:.0f} s")
    for k, nm in enumerate(FV.NAMES): print(f"  {nm:8s} {r['p'][k]:8.3f} +- {sd[k]:.3f}")
    fl = [(cname(CONDS[i]), round(float(per[i]), 1)) for i in range(35) if mask[i] and per[i] > CHI2_FLAG]
    print(f"  wafers flagged (chi2 > {CHI2_FLAG} on 9 sites): {fl if fl else 'none'}")
    big = [(FV.NAMES[a], FV.NAMES[b], round(float(corr[a, b]), 2)) for a in range(9) for b in range(a+1, 9) if abs(corr[a, b]) > 0.9]
    print(f"  |correlations| > 0.9: {big}")

# ---------------------------------------------------------------------------------------------------------------
SIG_O = F.SIG_O
def predict_interval(p, cov, free, h, o_fit=None, rng=None, n=200):
    """90 % interval for wafer h: parameter uncertainty (linearised), its own unknown dose offset (sampled through the
    nonlinear model) and the instrument (marginal over angle offsets, plus run-level shift)."""
    c = CONDS[h]
    T_, S_, So_ = FV.forward(p, np.zeros(35), sens=True, conds=[c], Nx=NX, Ny=NY)
    m0, s0, dm0 = INS.marginal(T_[0])
    Jt = S_[0][:, free]; var_par = np.einsum('ij,jk,ik->i', Jt, cov[:len(free), :len(free)], Jt)*dm0**2
    sims, tsims = [], []
    for k in range(n):
        o = np.zeros(35); o[h] = rng.normal(0, SIG_O)
        t = FV.forward(p, o, conds=[c], Nx=NX, Ny=NY)[0]; m, s, _ = INS.marginal(t)
        sims.append(m + rng.normal(0, 1, 9)*np.sqrt(s**2 + var_par)); tsims.append(t)
    sims = np.array(sims)
    return np.median(sims, 0), np.percentile(sims, 5, 0), np.percentile(sims, 95, 0), np.median(tsims, 0)

def _howo_job(h):
    mask = run_mask(temps(A.td, TD), temps(A.tp, TP)); mask[h] = False
    r = F.fit(Y, init_params(), mask=mask, Nx=NX, Ny=NY, max_nfev=NFEV)
    med, lo, hi, tmed = predict_interval(r['p'], r['cov'], r['free'], h, rng=np.random.default_rng(100 + h), n=60 if A.quick else 200)
    return dict(hold=h, cond=CONDS[h], p=r['p'], pred=med, lo=lo, hi=hi, true_pred=tmed, obs=Y[h], cov=r['cov'][:9, :9], chi2=r['chi2_data'])

def summarize_pred(rows, name):
    import csv
    with open(os.path.join(A.out, name + '.csv'), 'w', newline='') as f:
        wr = csv.writer(f); wr.writerow(['Td', 'Tp', 'rmse_nm', 'inside90', 'mean_width_nm', 'n_sites'] + [f'{k}{i}' for k in ('obs', 'pred', 'lo', 'hi') for i in range(9)])
        for r in rows:
            ok = np.isfinite(r['obs']); err = (r['obs'] - r['pred'])[ok]; ins = ((r['obs'] >= r['lo']) & (r['obs'] <= r['hi']))[ok]
            r['rmse'], r['cover'], r['width'] = np.sqrt(np.mean(err**2)), ins.mean(), np.mean((r['hi'] - r['lo'])[ok])
            wr.writerow([r['cond'][0], r['cond'][1], round(r['rmse'], 3), round(r['cover'], 3), round(r['width'], 3), int(ok.sum())]
                        + [round(v, 3) for k in ('obs', 'pred', 'lo', 'hi') for v in r[k]])
            print(f"  {cname(r['cond'])}: RMSE {r['rmse']:.2f} nm, inside 90 % {r['cover']:.2f}, mean width {r['width']:.2f} nm", flush=True)
    allin = np.concatenate([((r['obs'] >= r['lo']) & (r['obs'] <= r['hi']))[np.isfinite(r['obs'])] for r in rows])
    print(f"  overall: {allin.mean():.2f} of {allin.size} held-out sites inside their 90 % interval")
    np.save(os.path.join(A.out, name + '.npy'), rows, allow_pickle=True)

def howo_cmd():
    Pool = MP.Pool
    tds, tps = temps(A.td, TD), temps(A.tp, TP); mask = run_mask(tds, tps)
    if A.holds == '55': holds = [i for i in range(35) if mask[i] and CONDS[i][1] == 55.0]
    elif A.holds == 'all': holds = [i for i in range(35) if mask[i]]
    else: holds = [int(v) for v in A.holds.split(',')]
    print(f"hold-one-wafer-out over {len(holds)} wafers with {A.jobs} parallel fits", flush=True)
    with Pool(A.jobs) as pool: rows = pool.map(_howo_job, holds)
    tag = f"td{int(tds.min())}-{int(tds.max())}"
    summarize_pred(rows, f'howo_{tag}')

def extrap_cmd():
    """Fit on a sub-design and predict every other wafer: the 'beyond the training wafers' tier."""
    trd, trp = temps(A.train_td, TD), temps(A.train_tp, TP); mask = run_mask(trd, trp)
    Pool = MP.Pool
    with Pool(min(3, A.jobs)) as pool: r, _ = multistart_fit(mask, 'extrap', pool)
    tests = [i for i in range(35) if not mask[i]]
    rng = np.random.default_rng(5); rows = []
    for h in tests:
        med, lo, hi, tmed = predict_interval(r['p'], r['cov'], r['free'], h, rng=rng, n=60 if A.quick else 200)
        rows.append(dict(hold=h, cond=CONDS[h], pred=med, lo=lo, hi=hi, true_pred=tmed, obs=Y[h]))
    tag = f"train_td{int(trd.min())}-{int(trd.max())}_tp{int(trp.min())}-{int(trp.max())}"
    print(f"[{tag}] trained on {mask.sum()} wafers (data chi2 {r['chi2_data']:.1f} on {r['ndata']}); predicting {len(tests)} wafers")
    summarize_pred(rows, f'extrap_{tag}')
    save_json(f'extrap_{tag}_fit.json', dict(params=dict(zip(FV.NAMES, r['p'])), sd=dict(zip(FV.NAMES, np.sqrt(np.diag(r['cov'])[:9]))), chi2=r['chi2_data']))

# ---------------------------------------------------------------------------------------------------------------
def compare_cmd():
    """FV vs PINN on the same held-out real wafers. PINN interval: deep-ensemble spread + the held-out run's dose offset
    (sampled through the FV solver at the PINN's own physics and learned inlet profile) + instrument."""
    import glob
    from scipy.interpolate import interp1d
    fvf = sorted(glob.glob(os.path.join(A.out, 'howo_td*.npy')))
    if not fvf: sys.exit("run `python run_real.py howo` first")
    FVR = {r['hold']: r for r in np.load(fvf[-1], allow_pickle=True)}
    a, b = (A.seeds.split('-') + [None])[:2]; seeds = list(range(int(a), int(b) + 1)) if b else [int(s) for s in A.seeds.split(',')]
    rng = np.random.default_rng(11); rows = []
    for h, r in sorted(FVR.items()):
        obs = Y[h]; ok = np.isfinite(obs)
        rows.append(('FV', h, np.sqrt(np.mean((obs - r['pred'])[ok]**2)), np.mean(((obs >= r['lo']) & (obs <= r['hi']))[ok]), np.mean((r['hi'] - r['lo'])[ok]), 1))
        files = [f for s in seeds for f in glob.glob(os.path.join(A.out, f'pinn_{A.tag}_s{s}_h{h}.npz'))]
        zs = [np.load(f) for f in files]
        if not zs: continue
        # screen failed members: data chi2 within 25 % of the best member's (the synthetic study used chi2 < 330 of 315)
        best = min(float(z['comp'][2]) for z in zs); good = [z for z in zs if float(z['comp'][2]) <= 1.25*best]
        print(f"hold {cname(CONDS[h])}: {len(good)} of {len(zs)} PINN members kept (data chi2 {sorted(round(float(z['comp'][2]), 1) for z in zs)})")
        preds, sims_all = [], []
        for z in good:
            phys = z['phys']; pfv = np.array([phys[0], phys[1], phys[2], phys[3], phys[4], 0, 0, phys[5], phys[6]])
            gf = interp1d(np.linspace(-1, 1, len(z['g'])), z['g'], fill_value='extrapolate')
            D = []
            for k in range(40):
                o = np.zeros(35); o[h] = rng.normal(0, SIG_O); D.append(FV.forward(pfv, o, gfun=lambda x, gf=gf: gf(x), conds=[CONDS[h]], Nx=NX, Ny=NY)[0])
            D = np.array(D); D = D - np.median(D, 0) + z['pred'][h]; preds.append(z['pred'][h]); sims_all.append(D)
        preds = np.array(preds); D = np.concatenate(sims_all); ens_var = preds.var(0, ddof=1) if len(preds) > 1 else np.zeros(9)
        m, s, _ = INS.marginal(D); sims = m + rng.normal(0, 1, m.shape)*np.sqrt(s**2 + ens_var)
        lo, hi, med = np.percentile(sims, 5, 0), np.percentile(sims, 95, 0), np.median(sims, 0)
        rows.append((f'PINN', h, np.sqrt(np.mean((obs - med)[ok]**2)), np.mean(((obs >= lo) & (obs <= hi))[ok]), np.mean((hi - lo)[ok]), len(good)))
    import csv
    with open(os.path.join(A.out, f'compare_{A.tag}.csv'), 'w', newline='') as f:
        wr = csv.writer(f); wr.writerow(['method', 'Td', 'Tp', 'rmse_nm', 'inside90', 'mean_width_nm', 'n_members'])
        for row in rows:
            wr.writerow([row[0], CONDS[row[1]][0], CONDS[row[1]][1], round(row[2], 3), round(row[3], 3), round(row[4], 3), row[5]])
            print(f"{row[0]:5s} {cname(CONDS[row[1]])}: RMSE {row[2]:.2f} nm, inside 90 % {row[3]:.2f}, width {row[4]:.2f} nm (members {row[5]})")

if __name__ == '__main__':
    {'collapse': collapse, 'fit': fit_cmd, 'howo': howo_cmd, 'extrap': extrap_cmd, 'compare': compare_cmd}[A.cmd]()
