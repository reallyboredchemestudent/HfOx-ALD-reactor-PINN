"""Turn raw pipeline outputs into the tidy CSVs the paper figures read, and check each figure's claim.

    python export_figdata.py [--results ../results] [--out ../paper/figures/data] [--only fig3,fig5]

Every number a figure draws comes from a CSV written here; no figure contains a typed-in value. Every figure also
gets checks: the sentence of the paper (README "Main results" and "Every result" table) that the figure supports,
recomputed from the raw outputs and compared at the precision the text prints it. Results go to
<out>/checks.csv (one row per check) and <out>/checks.tex (one PASS/FAIL per figure, read by figstyle.tex: a figure
with a failed check is built with a visible stamp, never quietly). INFO rows are reported numbers with no criterion.

Inputs (all under --results): real_Y.npz, collapse_*.{csv,json}, fit_td100-200.json, per_wafer_td100-200.csv,
recal_*.{csv,json}, recal_compare_sites_REALM.csv, pinn_REALM_s*_h*.npz, pinn_REALM_s*.log, process_window.csv,
aldfoam_{gap1.0,gap0.5,dtcheck}/aldfoam_compare.{csv,txt}, si_results.txt, si/profile_U.npy, si/breakfit_*.npz.
Inputs that need tools CI does not have (JAX, OpenFOAM, Julia) are produced on a workstation; their SHA-256 are in
<out>/provenance.json so a figure can always be traced to the exact file it was drawn from.
"""
import os, sys, re, json, csv, glob, hashlib, argparse, platform
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--results', default=os.path.join(HERE, '..', 'results'))
ap.add_argument('--out', default=os.path.join(HERE, '..', 'paper', 'figures', 'data'))
ap.add_argument('--only', default=None, help='comma list of figure ids (default: all)')
A = ap.parse_args()
R = os.path.abspath(A.results); OUT = os.path.abspath(A.out); os.makedirs(OUT, exist_ok=True)

Z = np.load(os.path.join(R, 'real_Y.npz'), allow_pickle=True)
import fvmodel as FV                                     # noqa: E402
FV.SITES[:] = Z['sites']
FV.require_ncyc(Z['ncyc'])                               # raises if the data are not the default 100 cycles
Y = Z['Y']; CONDS = FV.CONDS; SITES = FV.SITES

INPUTS = {}
def src(path):
    """Register an input file (for provenance) and return its absolute path; a missing input is an error."""
    p = path if os.path.isabs(path) else os.path.join(R, path)
    if not os.path.exists(p): raise FileNotFoundError(f"missing input {p}: run the pipeline step that writes it")
    if os.path.isfile(p):
        INPUTS[os.path.relpath(p, R)] = hashlib.sha256(open(p, 'rb').read()).hexdigest()
    return p

def write(name, header, rows, fmt=None):
    with open(os.path.join(OUT, name), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(header)
        for r in rows: w.writerow([(f"{v:.6g}" if isinstance(v, (float, np.floating)) else v) for v in r])

CHECKS = []
def check(fig, name, value, claim, ok, info=False):
    CHECKS.append(dict(figure=fig, check=name, value=value, claim=claim, status='INFO' if info else ('PASS' if ok else 'FAIL')))

def same_as_printed(x, printed):
    """True if x, rounded to the decimals of the printed string, equals the printed number."""
    d = len(printed.split('.')[1]) if '.' in printed else 0
    return abs(float(x) - float(printed)) <= 0.5*10**(-d) + 1e-12

def order_along_flow():
    """Site order along the flow (y), then across (x): the x position used by the per-site panels."""
    return sorted(range(9), key=lambda i: (SITES[i, 1], SITES[i, 0]))

# ------------------------------------------------------------------------------------------------------------------
def fig1():
    """The reduction: aldFoam equal-dose pulses against the steady exposure solution."""
    import pandas as pd
    a = pd.read_csv(src('aldfoam_gap1.0/aldfoam_compare.csv')); b = pd.read_csv(src('aldfoam_dtcheck/aldfoam_compare.csv'))
    d = pd.read_csv(src('aldfoam_gap0.5/aldfoam_compare.csv'))
    txt = open(src('aldfoam_gap1.0/aldfoam_compare.txt')).read()
    get = lambda df, c: df[df.case == c].sort_values('y_cm')
    A_, B4, C_, B1, D_ = get(b, 'A'), get(b, 'B_dt4'), get(a, 'C'), get(a, 'B'), get(d, 'D')
    y = A_.y_cm.values
    for o in (B4, C_, B1, D_): assert np.allclose(o.y_cm.values, y), "aldFoam cases on different face centres"
    assert np.allclose(get(a, 'A').theta_aldfoam.values, A_.theta_aldfoam.values), "case A differs between runs"
    # steady exposure at matched mass (same molecules consumed as aldFoam case A): the reach compare_aldfoam found
    import gapresolved as GR, aldfoam_params as AP
    from scipy.optimize import brentq
    lam_txt = float(re.search(r'\[A\].*?matched mass \(reach ([\d.]+) cm', txt, re.S).group(1))
    mass = np.trapezoid(A_.theta_aldfoam.values, y)                 # same construction as compare_aldfoam.py
    def mass_ours(lam):
        s2 = GR.setup(1.0, 179.0, vbar=AP.vth*100, lam=lam); return np.trapezoid(np.interp(y, s2['yc'], GR.steady_exposure(s2)[0]), y) - mass
    lam_m = brentq(mass_ours, 4.0, 23.0, xtol=1e-3)
    assert abs(lam_m - lam_txt) < 0.006, f"matched reach {lam_m} differs from compare_aldfoam's {lam_txt}"
    s_nom = GR.setup(1.0, 179.0, vbar=AP.vth*100); s_m = GR.setup(1.0, 179.0, vbar=AP.vth*100, lam=lam_m)
    th_nom = np.interp(y, s_nom['yc'], GR.steady_exposure(s_nom)[0]); th_m = np.interp(y, s_m['yc'], GR.steady_exposure(s_m)[0])
    assert np.allclose(th_nom, A_.theta_steady_nominal.values, atol=1e-6), "steady solution does not reproduce the shipped CSV"
    write('fig1_profiles.csv', ['y', 'A', 'B', 'C', 'D', 'steady_nominal', 'steady_matched'],
          zip(y, A_.theta_aldfoam, B4.theta_aldfoam, C_.theta_aldfoam, D_.theta_aldfoam, th_nom, th_m))
    NM = 15.0                                            # nm of film per unit coverage (saturated film, synthetic study)
    dB, dC, dD, dB1 = [(o.theta_aldfoam.values - A_.theta_aldfoam.values)*NM for o in (B4, C_, D_, B1)]
    dM = (A_.theta_aldfoam.values - th_m)*NM
    write('fig1_diff.csv', ['y', 'dB', 'dC', 'dD', 'dBcoarse', 'dMatched'], zip(y, dB, dC, dD, dB1, dM))
    worst = max(np.abs(dB).max(), np.abs(dC).max())
    check('fig1', 'equal-dose pulses 0.05/0.2/0.4 s agree (aldFoam)', f"{np.abs(dB).max():.3f}, {np.abs(dC).max():.3f} nm",
          'README: 0.027-0.036 nm', same_as_printed(np.abs(dB).max(), '0.027') and same_as_printed(np.abs(dC).max(), '0.036'))
    check('fig1', 'reduction exact to', f"{worst:.3f} nm", 'README: to 0.04 nm', worst <= 0.04)
    check('fig1', '0.05 s pulse at the default time step', f"{np.abs(dB1).max():.3f} nm", 'not converged in time: shown, not used', True, info=True)
    check('fig1', 'aldFoam vs steady at matched mass', f"{np.abs(dM).max():.3f} nm (reach {lam_m:.2f} vs nominal 16 cm)", 'compare_aldfoam: 0.024 nm',
          same_as_printed(np.abs(dM).max(), '0.024'))
    check('fig1', 'gap 0.5 cm at twice the speed vs gap 1.0 cm (case D, not drawn)', f"{np.abs(dD).max():.3f} nm",
          'aldfoam_params.py says the depth-averaged equation predicts the same front; the README makes no claim', True, info=True)

# ------------------------------------------------------------------------------------------------------------------
def fig2():
    """Identifiability: profile likelihood of U at Lee's design, rank test one dose vs two, SI.jl verdicts."""
    P = np.load(src('si/profile_U.npy'))
    free_U, free_cost = P[0]; prof = P[1:][np.argsort(P[1:, 0])]
    # a refit capped at 25 evaluations (profile_lik.py) that ends 10^3+ above the optimum did not converge: it says
    # nothing about U, so it is reported, not drawn
    bad = prof[prof[:, 1] - free_cost > 1e3]; prof = prof[prof[:, 1] - free_cost <= 1e3]
    write('fig2_profile.csv', ['U', 'dchi2'], [(u, c - free_cost) for u, c in prof])
    check('fig2', 'profile refits that did not converge (not drawn)', ', '.join(f"U = {u:g} (dchi2 {c - free_cost:.3g})" for u, c in bad) or 'none',
          'profile_lik.py caps each refit at 25 evaluations', True, info=True)
    write('fig2_free.csv', ['U', 'dchi2'], [(free_U, 0.0)])
    inside = prof[prof[:, 1] - free_cost <= 3.84, 0]
    check('fig2', 'U within the 95 % profile interval', f"{inside.min():.1f}-{inside.max():.1f} cm (grid)", 'README: upstream length 7.5-16 cm',
          True, info=True)
    # numerical rank test along the centreline, one dose vs two doses at the same deposition temperature
    from fvmodel import Grid, groups, solve, TRUTH_A
    p = TRUTH_A.copy(); YOBS = np.linspace(0.0, 14.5, 59)
    def out(q, conds):
        grid = Grid(q[4], 30, 96); res = []
        for td, tp in conds:
            lam, zb, l, gpc = groups(q, td, tp); e, *_ = solve(grid, lam, zb, l, np.ones(grid.Nx))
            th = 1 - np.exp(-e.reshape(grid.Ny, grid.Nx)[:, grid.Nx//2])
            res.append(np.interp(YOBS, grid.yc, th))          # fixed physical positions: the grid moves with U
        return np.concatenate(res)
    rows = []
    for label, conds in (('one dose', [(100.0, 55.0)]), ('two doses', [(100.0, 55.0), (100.0, 57.5)])):
        J = []
        for k in (0, 1, 3, 4):
            h = 1e-4 if k != 4 else 1e-3; qp = p.copy(); qp[k] += h; qm = p.copy(); qm[k] -= h
            J.append((out(qp, conds) - out(qm, conds))/(2*h))
        J = np.array(J).T; J = J/np.linalg.norm(J, axis=0); s = np.linalg.svd(J, compute_uv=False); s = s/s[0]
        rows += [(label, k + 1, max(v, 1e-16)) for k, v in enumerate(s)]
    write('fig2_rank.csv', ['design', 'k', 'sv'], rows)
    write('fig2_rank_one.csv', ['k', 'sv'], [r[1:] for r in rows if r[0] == 'one dose'])
    write('fig2_rank_two.csv', ['k', 'sv'], [r[1:] for r in rows if r[0] == 'two doses'])
    sv1 = [r[2] for r in rows if r[0] == 'one dose'][-1]; sv2 = [r[2] for r in rows if r[0] == 'two doses'][-1]
    check('fig2', 'smallest singular value (log lam, log zb, log l, U)', f"one dose {sv1:.1e}, two doses {sv2:.1e}",
          'one dose: numerically rank deficient (< 1e-3; exact only for plug flow, SI.jl); two doses: > 1e-2', sv1 < 1e-3 and sv2 > 1e-2)
    # StructuralIdentifiability.jl verdicts, transcribed from its own output
    t = open(src('si_results.txt')).read(); rows = []
    for blk in re.split(r'=== model ', t)[1:]:
        title = blk.split('===')[0].strip(); num, desc = title.split(None, 1)
        verdicts = re.findall(r'^\s+(\S+?)(?:\(t\))?\s+=>\s+(\w+)', blk, re.M)
        rows += [(int(num), desc.strip(), v, s_) for v, s_ in verdicts]
    write('fig2_si.csv', ['model', 'description', 'parameter', 'verdict'], rows)
    # the two models that answer "one dose or two": a small table, typeset from this file (semicolon separated)
    funcs = {int(b.split(None, 1)[0]): re.search(r'identifiable functions \(with states\): (.*)', b).group(1)
             for b in re.split(r'=== model ', t)[1:]}
    word = {'globally': 'yes', 'locally': 'locally', 'nonidentifiable': 'no'}
    with open(os.path.join(OUT, 'fig2_si_table.csv'), 'w') as f:
        f.write('design;reach;offset;combos\n')
        for m, lab in ((3, 'one dose'), (4, 'two doses')):
            v = {p_: s_ for mm, _, p_, s_ in rows if mm == m}
            combos = funcs[m].replace('(t)', '').replace('//', '/').replace(' ', '')
            combos = combos.replace(',', ',\\;')
            f.write(f"{lab};{word[v['E']]};{word[v['q']]};${combos}$\n")
    m3 = {v: s_ for m, _, v, s_ in rows if m == 3}; m4 = {v: s_ for m, _, v, s_ in rows if m == 4}
    check('fig2', 'SI.jl: one dose', str(m3), 'README: one dose fixes only the front position', m3.get('E') == 'nonidentifiable' and m3.get('q') == 'nonidentifiable')
    check('fig2', 'SI.jl: two doses', str(m4), 'README: a second dose separates reach from upstream length', set(m4.values()) == {'globally'})

# ------------------------------------------------------------------------------------------------------------------
def fig3():
    """Mass balance on the real wafers: raw profiles, collapse, front vs 1/GPC."""
    import pandas as pd
    pts = pd.read_csv(src('collapse_points.csv')); S = json.load(open(src('collapse_summary.json')))
    order = order_along_flow()
    rows = []
    for _, r in pts.iterrows():
        rows.append((r.Td, int(r.site), order.index(int(r.site)) + 1, r.y_cm, r.x_cm, r.reported_nm, r.coverage, r.s_norm, int(r.thin_lt5nm), int(r.in_regime)))
    write('fig3_points.csv', ['Td', 'site', 'pos', 'y', 'x', 'reported', 'cov', 's', 'thin', 'regime'], rows)
    td = np.array(S['Td']); gpc = np.array(S['gpc']); fr = np.array(S['front']); sdt = np.array(S['front_sd_total'])
    res = np.array([bool(v) for v in S['resolved']])
    fr_rows = [(t, 1/g, f, (s if np.isfinite(s) else 0.0), int(ok), int(t in S['regime'])) for t, g, f, s, ok in zip(td, gpc, fr, sdt, res)]
    hdr = ['Td', 'invgpc', 'front', 'sd', 'resolved', 'regime']
    write('fig3_fronts.csv', hdr, fr_rows)
    # plots with error bars read one file per series (pgfplots misplaces explicit error bars when rows are filtered)
    write('fig3_fronts_fit.csv', hdr, [r for r in fr_rows if r[5] == 1])
    write('fig3_fronts_excl.csv', hdr, [r for r in fr_rows if r[5] == 0 and r[4] == 1 or (r[5] == 0 and r[3] > 0)])
    C, U = S['C'], S['U']; cv = np.array(S['cov_CU'])
    xg = np.linspace(0.6, 1.05, 60); rows = []
    for x in xg:                                         # y = C x - U, 90 % band from the (C, U) covariance
        v = x*x*cv[0, 0] + cv[1, 1] - 2*x*cv[0, 1]; m = C*x - U; s = np.sqrt(max(v, 0)); rows.append((x, m, m - 1.645*s, m + 1.645*s))
    write('fig3_line.csv', ['invgpc', 'y', 'lo', 'hi'], rows)
    write('fig3_stats.csv', ['C', 'Csd', 'U', 'Usd', 'chi2', 'dof', 'cv'], [(C, np.sqrt(cv[0, 0]), U, np.sqrt(cv[1, 1]), S['chi2'], S['dof'], 100*S['cv'])])
    from scipy.stats import chi2 as CH
    p = CH.sf(S['chi2'], S['dof'])
    check('fig3', 'mass-balance line chi2', f"{S['chi2']:.2f} on {S['dof']} (p = {p:.2f})", 'README: chi2 1.2 / 3', same_as_printed(S['chi2'], '1.2') and S['dof'] == 3)
    check('fig3', 'reach x GPC constant over 100-200 C', f"CV {100*S['cv']:.2f} %", 'README: to 2.6 %', same_as_printed(100*S['cv'], '2.6'))

# ------------------------------------------------------------------------------------------------------------------
def fig4():
    """Error budget: per-wafer chi2 under the ideal and the measured budget; the site signature from raw data."""
    import pandas as pd
    ide = pd.read_csv(src('per_wafer_td100-200.csv')); mea = pd.read_csv(src('recal_per_wafer_td100-200.csv'))
    fi = json.load(open(src('fit_td100-200.json'))); fm = json.load(open(src('recal_fit_td100-200.json')))
    rows = []
    for (_, a), (_, b) in zip(ide.iterrows(), mea.iterrows()):
        assert (a.Td, a.Tp) == (b.Td, b.Tp)
        rows.append((a.Td, a.Tp, int(a.used_in_fit), a.chi2_9sites, np.log10(a.chi2_9sites), b.chi2_9sites, np.log10(b.chi2_9sites)))
    write('fig4_chi2.csv', ['Td', 'Tp', 'used', 'ideal', 'lideal', 'measured', 'lmeasured'], rows)
    used = ide.used_in_fit.values == 1
    # signature: the calibrated one, and the raw relative deviations of every saturated wafer it was measured on
    sig = np.array(fm['signature']); order = order_along_flow(); rows = []; raw = []
    for i, c in enumerate(CONDS):
        if c[1] >= 60 and 100 <= c[0] <= 200 and np.isfinite(Y[i]).all():
            dev = Y[i]/np.mean(Y[i]) - 1; raw += [(order.index(k) + 1, 100*dev[k], c[0], c[1]) for k in range(9)]
    write('fig4_sigraw.csv', ['pos', 'dev', 'Td', 'Tp'], raw)
    write('fig4_sig.csv', ['pos', 'site', 'y', 'x', 'sig'], [(order.index(k) + 1, k, SITES[k, 1], SITES[k, 0], 100*sig[k]) for k in range(9)])
    write('fig4_stats.csv', ['ideal', 'measured', 'ndata', 'sigr', 'sigd', 'nflag_ideal', 'nflag_measured'],
          [(fi['chi2'] if 'chi2' in fi else fi['chi2_data'], fm['chi2'], fm['ndata'], 100*fm['sig_r'], fm['sig_d'],
            int((ide.chi2_9sites[used] > 27.88).sum()), int((mea.chi2_9sites[used] > 27.88).sum()))])
    ci = fi['chi2'] if 'chi2' in fi else fi['chi2_data']
    check('fig4', 'ideal budget, data chi2', f"{ci:.1f} on {fm['ndata']}", 'README: chi2 21,124 / 225', same_as_printed(ci, '21124'))
    check('fig4', 'measured budget, data chi2', f"{fm['chi2']:.1f} on {fm['ndata']}", 'README: chi2 448', same_as_printed(fm['chi2'], '448'))
    check('fig4', 'per-wafer chi2 sums to the fit chi2 (measured)', f"{mea.chi2_9sites[used].sum():.1f} vs {fm['chi2']:.1f}",
          'integrity: within 0.5 %', abs(mea.chi2_9sites[used].sum() - fm['chi2']) < 0.005*fm['chi2'])
    check('fig4', 'wafers flagged at p < 0.001 (chi2 > 27.9 on 9 sites)', f"ideal {(ide.chi2_9sites[used] > 27.88).sum()}, measured {(mea.chi2_9sites[used] > 27.88).sum()} of {used.sum()}",
          'reported', True, info=True)

# ------------------------------------------------------------------------------------------------------------------
def fig5():
    """FV vs PINN on held-out wafers under the measured budget."""
    import pandas as pd
    st = pd.read_csv(src('recal_compare_sites_REALM.csv')); sm = pd.read_csv(src('recal_compare_REALM.csv'))
    order = order_along_flow(); rows = []
    for _, r in st.iterrows():
        rows.append((r.method, r.Td, int(r.site), order.index(int(r.site)) + 1, r.y_cm, r.obs_nm, r.pred_nm, r.lo90_nm, r.hi90_nm,
                     r.pred_nm - r.lo90_nm, r.hi90_nm - r.pred_nm, int(r.lo90_nm <= r.obs_nm <= r.hi90_nm)))
    hdr = ['method', 'Td', 'site', 'pos', 'y', 'obs', 'pred', 'lo', 'hi', 'elo', 'ehi', 'inside']
    write('fig5_sites.csv', hdr, rows)
    for m_ in ('FV', 'PINN'):
        write(f'fig5_sites_{m_.lower()}.csv', hdr[1:], [r[1:] for r in rows if r[0] == m_])
        for t in sorted({r[1] for r in rows}):
            write(f'fig5_sites_{m_.lower()}_{t:.0f}.csv', hdr[1:], [r[1:] for r in rows if r[0] == m_ and r[1] == t])
    write('fig5_summary.csv', ['method', 'Td', 'rmse', 'inside', 'width', 'members'],
          [(r.method, r.Td, r.rmse_nm, r.inside90, r.mean_width_nm, r.members) for _, r in sm.iterrows()])
    # every ensemble member: its own fit to the training wafers (all data) against its own held-out error
    holds = sorted({CONDS.index((r.Td, r.Tp)) for _, r in st.iterrows()})
    mem = []
    for f in sorted(glob.glob(os.path.join(R, 'pinn_REALM_s*_h-1.npz'))):
        z = np.load(src(f)); s = int(z['seed']); errs = []; nd_ok = True
        for h in holds:
            zh = np.load(src(os.path.join(R, f'pinn_REALM_s{s}_h{h}.npz')))
            nd_ok &= int(zh['ndata']) == int(z['ndata']) - int(np.isfinite(Y[h]).sum())
            errs += list(zh['pred'][h] - Y[h])
        mem.append((s, float(z['comp'][2]), float(np.sqrt(np.mean(np.square(errs)))), int(nd_ok)))
    fm = json.load(open(src('recal_fit_td100-200.json')))
    fv = st[st.method == 'FV']; fv_rmse = float(np.sqrt(np.mean((fv.pred_nm - fv.obs_nm)**2)))
    write('fig5_members.csv', ['seed', 'chi2', 'rmse', 'holdout_ok'], mem)
    pn = st[st.method == 'PINN']; pn_rmse = float(np.sqrt(np.mean((pn.pred_nm - pn.obs_nm)**2)))
    write('fig5_fv.csv', ['chi2', 'rmse'], [(fm['chi2'], fv_rmse)])
    write('fig5_ens.csv', ['chi2', 'rmse'], [(float(np.median([m[1] for m in mem])), pn_rmse)])
    # per-wafer summary, wide: one row per held-out wafer (FV and PINN side by side)
    F_, P_ = sm[sm.method == 'FV'].reset_index(), sm[sm.method == 'PINN'].reset_index()
    write('fig5_wafers.csv', ['Td', 'k', 'fv_rmse', 'pinn_rmse', 'fv_in', 'pinn_in', 'members'],
          [(F_.Td[i], i + 1, F_.rmse_nm[i], P_.rmse_nm[i], 100*F_.inside90[i], 100*P_.inside90[i], P_.members[i]) for i in range(len(F_))])
    check('fig5', 'held-out wafer excluded from every PINN member', f"{sum(m[3] for m in mem)}/{len(mem)} seeds x {len(holds)} holds",
          'n_data(hold) = n_data(all) - 9', all(m[3] for m in mem))
    F = sm[sm.method == 'FV']; P = sm[sm.method == 'PINN']
    cov = lambda d: float(st[st.method == d].apply(lambda r: r.lo90_nm <= r.obs_nm <= r.hi90_nm, axis=1).mean())
    check('fig5', 'FV held-out RMSE range', f"{F.rmse_nm.min():.2f}-{F.rmse_nm.max():.2f} nm", 'README: 1.0-2.6 nm',
          same_as_printed(F.rmse_nm.min(), '1.0') and same_as_printed(F.rmse_nm.max(), '2.6'))
    check('fig5', 'PINN held-out RMSE range', f"{P.rmse_nm.min():.2f}-{P.rmse_nm.max():.2f} nm", 'README: 1.6-3.9 nm',
          same_as_printed(P.rmse_nm.min(), '1.6') and same_as_printed(P.rmse_nm.max(), '3.9'))
    check('fig5', 'FV 90 % interval coverage', f"{100*cov('FV'):.1f} % of {len(fv)} sites ({len(holds)} wafers)", 'README: 96 %', same_as_printed(100*cov('FV'), '96'))
    check('fig5', 'PINN 90 % interval coverage', f"{100*cov('PINN'):.1f} %", 'README: 80 %', same_as_printed(100*cov('PINN'), '80'))
    check('fig5', 'FV beats PINN on every held-out wafer', f"{int((F.rmse_nm.values < P.rmse_nm.values).sum())}/{len(F)}",
          'README: the PINN predicts held-out wafers worse', bool(np.all(F.rmse_nm.values < P.rmse_nm.values)))
    pc = [m[1] for m in mem]
    check('fig5', 'PINN fits the training wafers better', f"chi2 {min(pc):.0f}-{max(pc):.0f} vs FV {fm['chi2']:.0f} on {fm['ndata']}",
          'README: chi2 ~220 vs 448', max(pc) < fm['chi2'])
    up = os.path.join(R, '..', 'results_upload', 'recal_compare_REALM.csv')      # optional cross-machine check
    if os.path.exists(up):
        U_ = pd.read_csv(up); a_ = U_[U_.method == 'PINN'][['rmse_nm', 'inside90', 'mean_width_nm']].values
        b_ = P[['rmse_nm', 'inside90', 'mean_width_nm']].values
        check('fig5', 'PINN rows reproduce the workstation run', f"max |diff| {np.abs(a_ - b_).max():.3g}", 'same files, same RNG', np.abs(a_ - b_).max() < 1e-3)

# ------------------------------------------------------------------------------------------------------------------
def fig6():
    """Extrapolation to 225-250 C from the 100-200 C fit."""
    import pandas as pd
    ex = pd.read_csv(src('recal_extrap_td100-200.csv')); order = order_along_flow(); rows = []; ins = []
    for _, r in ex.iterrows():
        for i in range(9):
            o, p, l, h = r[f'obs{i}'], r[f'pred{i}'], r[f'lo{i}'], r[f'hi{i}']
            zi = (o - p)/(h - p) if o >= p else (o - p)/(p - l)        # +-1 = edge of the 90 % interval
            jit = order.index(i) + 1 + 0.06*(list(FV.TP).index(r.Tp) - 2) + (0.33 if r.Td == 250 else 0.0) - 0.165
            rows.append((r.Td, r.Tp, i, order.index(i) + 1, jit, SITES[i, 1], o, p, l, h, p - l, h - p, o - p, zi, int(l <= o <= h)))
            ins.append(l <= o <= h)
    hdr = ['Td', 'Tp', 'site', 'pos', 'jit', 'y', 'obs', 'pred', 'lo', 'hi', 'elo', 'ehi', 'res', 'z', 'inside']
    write('fig6_sites.csv', hdr, rows)
    for t in (225.0, 250.0): write(f'fig6_sites_{t:.0f}.csv', hdr, [r for r in rows if r[0] == t])
    write('fig6_summary.csv', ['Td', 'Tp', 'rmse', 'inside', 'width'], [(r.Td, r.Tp, r.rmse_nm, r.inside90, r.mean_width_nm) for _, r in ex.iterrows()])
    write('fig6_stats.csv', ['worst', 'coverage', 'n'], [(ex.rmse_nm.max(), 100*np.mean(ins), len(ins))])
    check('fig6', 'worst extrapolated-wafer RMSE', f"{ex.rmse_nm.max():.3f} nm", 'README: <= 0.63 nm', same_as_printed(ex.rmse_nm.max(), '0.63'))
    check('fig6', '90 % interval coverage, 225-250 C', f"{100*np.mean(ins):.1f} % of {len(ins)} sites", 'README: 94 %', same_as_printed(100*np.mean(ins), '94'))

# ------------------------------------------------------------------------------------------------------------------
def fig7():
    """Domain of applicability and process window, with the model field on the whole wafer."""
    import pandas as pd
    pw = pd.read_csv(src('process_window.csv'))
    write('fig7_window.csv', list(pw.columns) + ['extrap'], [list(r) + [int(r['Td'] > 200)] for _, r in pw.iterrows()])
    rows = []
    for td, tp in CONDS:
        role = 'extrapolated' if td > 200 else ('heldout' if tp == 55 else 'fitted')
        rows.append((td, tp, role, int(role == 'fitted'), int(role == 'heldout'), int(role == 'extrapolated')))
    write('fig7_design.csv', ['Td', 'Tp', 'role', 'fitted', 'heldout', 'extrap'], rows)
    check('fig7', 'whole-wafer dose multiplier, 100-250 C', f"x{pw.wafer_p50.min():.2f}-{pw.wafer_p50.max():.2f}", 'README: 1.2-1.7x the 55 C dose',
          same_as_printed(pw.wafer_p50.min(), '1.2') and same_as_printed(pw.wafer_p50.max(), '1.7'))
    a, b = pw.wafer_setpoint_Tp_C.iloc[0], pw.wafer_setpoint_Tp_C.iloc[-1]
    check('fig7', 'setpoint source temperature 100 -> 250 C', f"{a:.1f} -> {b:.1f} C", 'README: 65 C falling to 60 C',
          same_as_printed(a, '65') and same_as_printed(b, '60'))
    # model coverage on the whole wafer at 100 C: the measured 55 C dose and the setpoint dose
    fm = json.load(open(src('recal_fit_td100-200.json'))); p = np.array([fm['params'][n] for n in FV.NAMES])
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib import colormaps
    cmap = colormaps['Blues']
    grid = FV.Grid(p[4], 120, 384); g = FV.inlet_profile(grid.xi, p)
    XX, YY = np.meshgrid(grid.xc, grid.yc); wafer = (XX**2 + (YY - 7.5)**2) <= 5.0**2
    mult = {'measured': 1.0, 'setpoint': float(pw.wafer_setpoint.iloc[0])}
    stats = []
    for name, m in mult.items():
        lam, zb, l, gpc = FV.groups(p, 100.0, 55.0, np.log(m)); e, *_ = FV.solve(grid, lam, zb, l, g)
        th = (1 - np.exp(-e)).reshape(grid.Ny, grid.Nx)
        rgba = cmap(th); rgba[..., 3] = np.where(wafer, 1.0, 0.0)
        sel = (YY >= 2.5) & (YY <= 12.5) & (XX >= -5) & (XX <= 5)
        ny, nx = sel.any(1).sum(), sel.any(0).sum(); img = rgba[sel].reshape(ny, nx, 4)[::-1]
        plt.imsave(os.path.join(OUT, f'fig7_field_{name}.png'), img)
        es = grid.P@e; stats.append((name, m, float(th[wafer].min()), float((1 - np.exp(-es)).min()), grid.xc[sel.any(0)][0] - grid.dx/2,
                                     grid.xc[sel.any(0)][-1] + grid.dx/2, grid.yc[sel.any(1)][0] - grid.dy/2, grid.yc[sel.any(1)][-1] + grid.dy/2))
    write('fig7_fields.csv', ['case', 'mult', 'min_wafer', 'min_sites', 'xmin', 'xmax', 'ymin', 'ymax'], stats)
    # the pgfplots colour bar uses the same colormap, sampled
    with open(os.path.join(OUT, 'cmap_blues.tex'), 'w') as f:
        f.write('\\pgfplotsset{colormap={blues}{' + ' '.join('rgb=({:.4f},{:.4f},{:.4f})'.format(*cmap(t)[:3]) for t in np.linspace(0, 1, 9)) + '}}\n')
    write('fig7_sites.csv', ['x', 'y'], [(SITES[i, 0], SITES[i, 1]) for i in range(9)])
    check('fig7', 'setpoint dose covers the whole wafer (model)', f"min coverage {stats[1][2]:.3f}", 'threshold 0.98 (process_window.py)', stats[1][2] >= 0.98 - 1e-3)

# ------------------------------------------------------------------------------------------------------------------
def fig8():
    """Where the model fails: flagged wafers under the measured budget, and synthetic broken assumptions."""
    import pandas as pd
    mea = pd.read_csv(src('recal_per_wafer_td100-200.csv'))
    rows = []
    for _, r in mea.iterrows():
        for i in range(9): rows.append((r.Td, r.Tp, int(r.used), int(r.flag), r.chi2_9sites, i, SITES[i, 0], SITES[i, 1], r[f'res{i}']))
    write('fig8_resid.csv', ['Td', 'Tp', 'used', 'flag', 'chi2', 'site', 'x', 'y', 'z'], rows)
    order = order_along_flow(); heat = []; rws = []
    for k, (_, r) in enumerate(mea.iterrows()):                 # rows: wafers in (Td, Tp) order; columns: sites along the flow
        row = k + 1
        rws.append((row, r.Td, r.Tp, int(r.used), int(r.flag), r.chi2_9sites, row if r.flag else np.nan))
        for pos in range(1, 10):
            z = float(r[f'res{order[pos - 1]}']); heat.append((pos, row, z, float(np.clip(z, -4, 4))))
    write('fig8_heat.csv', ['pos', 'row', 'z', 'zc'], heat)
    write('fig8_rows.csv', ['row', 'Td', 'Tp', 'used', 'flag', 'chi2', 'flagrow'], rws)
    write('fig8_groups.csv', ['row', 'Td'], [(5*i + 3, t) for i, t in enumerate(FV.TD)])
    fl = mea[(mea.used == 1) & (mea.flag == 1)]
    write('fig8_flagged.csv', ['Td', 'Tp', 'chi2'], [(r.Td, r.Tp, r.chi2_9sites) for _, r in fl.iterrows()])
    cases = [('nominal', 'none (control)'), ('second_order', 'second order in gas'), ('flow_spike', 'flow spike in pulse'),
             ('pool_unpurged', 'unpurged pool'), ('mediated', 'pool-mediated'), ('surface_cvd', 'surface CVD')]; rows = []
    for k, (c, lab) in enumerate(cases):
        z = np.load(src(f'si/breakfit_{c}.npz')); pw_ = z['per_wafer']
        rows.append((k + 1, c, float(z['chi2']), float(pw_.max()), int((pw_ > 27.88).sum()), int((pw_ > 27.88).any()), lab))
    write('fig8_breaks.csv', ['k', 'case', 'chi2', 'worst', 'nflag', 'flagged', 'label'], rows)
    nb = sum(r[5] for r in rows[1:])
    check('fig8', 'nominal (control) is not flagged', f"{rows[0][4]} wafers > 27.9", 'no false alarm', rows[0][5] == 0)
    check('fig8', 'broken assumptions flagged by residuals', f"{nb} of 5 ({', '.join(r[1] for r in rows[1:] if not r[5])} not flagged)",
          'README: 4 of 5 flagged', nb == 4)
    check('fig8', 'flagged real wafers (measured budget, p < 0.001)', ', '.join(f"{a:.0f}/{b:.0f}" for a, b in zip(fl.Td, fl.Tp)) or 'none', 'reported', True, info=True)

# ------------------------------------------------------------------------------------------------------------------
def fig9():
    """PINN diagnostics: training stages, physics parameters per member against FV, PDE residual field."""
    rows = []
    for f in sorted(glob.glob(os.path.join(R, 'pinn_REALM_s*.log'))):
        seed = int(re.search(r'_s(\d+)\.log', f).group(1)); hold = None
        for line in open(src(f)):
            m = re.search(r'\] hold (-?\d+) \(', line)
            if m: hold = int(m.group(1)); continue
            m = re.search(r'stage (\d+): loss ([\d.e+-]+) pde ([\d.e+-]+) inlet ([\d.e+-]+) data chi2 ([\d.e+-]+)', line)
            if m and hold == -1: rows.append((seed, int(m.group(1)), *[float(m.group(k)) for k in range(2, 6)]))
    write('fig9_stages.csv', ['seed', 'stage', 'loss', 'pde', 'inlet', 'chi2'], rows)
    fm = json.load(open(src('recal_fit_td100-200.json')))
    names = ['log_lam', 'log_zb', 'Ea', 'log_l', 'U', 'G0', 'G1']; rows = []
    for f in sorted(glob.glob(os.path.join(R, 'pinn_REALM_s*_h-1.npz'))):
        z = np.load(src(f))
        for k, n in enumerate(names): rows.append((int(z['seed']), k + 1, n, (z['phys'][k] - fm['params'][n])/fm['sd'][n]))
    write('fig9_phys.csv', ['seed', 'k', 'name', 'z'], rows)
    write('fig9_fv.csv', ['chi2', 'ndata'], [(fm['chi2'], fm['ndata'])])
    zz = np.array([r[3] for r in rows]).reshape(-1, 7)
    check('fig9', 'PINN physics within 3 sd of FV (all members, all-data fit)', f"{int((np.abs(zz) <= 3).all(0).sum())} of 7 parameters; worst |z| {np.abs(zz).max():.1f}",
          'reported', True, info=True)
    # PDE residual of one member (seed 0, all data) on a grid at 100 C / 55 C source, NumPy forward pass
    import pinn as PN
    lay = PN.Layout(H=100, HG=6); z = np.load(src(os.path.join(R, 'pinn_REALM_s0_h-1.npz'))); th = z['th']
    phys = th[lay.sl['phys']]; o = th[lay.sl['o']]; cc = PN.cond_coeffs(phys, o); ci = CONDS.index((100.0, 55.0))
    llam, lzb, ll, gpc, sd, sp, _ = cc[ci]; U = phys[4]; L = PN.YEND + U
    xi, eta = np.meshgrid(np.linspace(-1, 1, 121), np.linspace(0, 1, 241)); xi, eta = xi.ravel(), eta.ravel()
    fx, fx1, fx2, fy, fy1, fy2 = PN.features(xi, eta)
    c = PN.net(th, lay, np.stack([fx, fy, np.full_like(xi, sd), np.full_like(xi, sp)], 1), fx1, fx2, fy1, fy2)
    lam, zb, l = np.exp(llam), np.exp(lzb), np.exp(ll)
    e = lam*c['N']/zb; ee = np.exp(-np.clip(e, -30, 60)); lap = c['Nyy']/L**2 + c['Nxx']/PN.XH**2
    r = (lam*c['Ny']/L - l*lam*lap + 1 - ee).reshape(241, 121)
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib import colormaps
    lr = np.log10(np.abs(r) + 1e-12); vmin, vmax = -3.0, 0.0
    plt.imsave(os.path.join(OUT, 'fig9_resid.png'), np.clip((lr - vmin)/(vmax - vmin), 0, 1)[::-1], cmap=colormaps['Greys'], vmin=0, vmax=1)
    cov_ = (1 - ee).reshape(241, 121)
    write('fig9_resid_meta.csv', ['xmin', 'xmax', 'ymin', 'ymax', 'vmin', 'vmax', 'rms', 'rms_reported', 'L'],
          [(-PN.XH, PN.XH, -U, PN.YEND, vmin, vmax, float(np.sqrt(np.mean(r**2))), float(z['comp'][0]), L)])
    # coverage contours of the same member, to show where the residual sits relative to the front
    ys = -U + np.linspace(0, 1, 241)*L; mid = cov_[:, 60]
    write('fig9_front.csv', ['y', 'cov'], [(y_, c_) for y_, c_ in zip(ys[::4], mid[::4])])
    k = int(np.argmax(mid < 0.5)); yf = ys[k - 1] + (0.5 - mid[k - 1])*(ys[k] - ys[k - 1])/(mid[k] - mid[k - 1])
    write('fig9_front50.csv', ['x', 'y'], [(-PN.XH, yf), (PN.XH, yf)])
    with open(os.path.join(OUT, 'cmap_greys.tex'), 'w') as f:
        cg = colormaps['Greys']; f.write('\\pgfplotsset{colormap={greys}{' + ' '.join('rgb=({:.4f},{:.4f},{:.4f})'.format(*cg(t)[:3]) for t in np.linspace(0, 1, 9)) + '}}\n')
    # collocation points of the 100 C / 55 C condition for that member (same construction as pinn_jax run_seed)
    base = np.isin(np.array([c_[0] for c_ in CONDS]), [100.0, 125.0, 150.0, 175.0, 200.0])
    import budget as BU
    B = BU.calibrate(Y, base, CONDS)
    prob = PN.Problem(Y, base, lay, np.random.default_rng(int(z['seed'])), n55=200, nsat=40, nin=24, budget=B)
    col = prob.col[prob.col[:, 0].astype(int) == ci]
    write('fig9_colloc.csv', ['x', 'y'], [(a_*PN.XH, -U + b_*L) for a_, b_ in col[:, 1:3]])
    check('fig9', 'NumPy PDE residual reproduces the saved one', f"grid rms^2 {np.mean(r**2):.3g} (this condition) vs saved mean {float(z['comp'][0]):.3g} (all collocation)",
          'same order of magnitude', 0.1 < np.mean(r**2)/float(z['comp'][0]) < 10, info=False)

FIGS = dict(fig1=fig1, fig2=fig2, fig3=fig3, fig4=fig4, fig5=fig5, fig6=fig6, fig7=fig7, fig8=fig8, fig9=fig9)

if __name__ == '__main__':
    todo = A.only.split(',') if A.only else list(FIGS)
    prev = []
    cf = os.path.join(OUT, 'checks.csv')
    if A.only and os.path.exists(cf): prev = [r for r in csv.DictReader(open(cf)) if r['figure'] not in todo]
    errors = {}
    for k in todo:
        try: FIGS[k](); print(f"{k}: exported")
        except FileNotFoundError as e:
            errors[k] = str(e); CHECKS.append(dict(figure=k, check='inputs present', value='missing', claim=str(e), status='FAIL'))
            print(f"{k}: MISSING INPUT: {e}", file=sys.stderr)
    allc = prev + CHECKS
    with open(cf, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['figure', 'check', 'value', 'claim', 'status']); w.writeheader(); w.writerows(allc)
    status = {}
    for c in allc: status[c['figure']] = 'FAIL' if c['status'] == 'FAIL' or status.get(c['figure']) == 'FAIL' else 'PASS'
    with open(os.path.join(OUT, 'checks.tex'), 'w') as f:
        for k in FIGS: f.write(f"\\figcheckstatus{{{k}}}{{{status.get(k, 'MISSING')}}}\n")
    pv = os.path.join(OUT, 'provenance.json'); old = json.load(open(pv))['inputs'] if os.path.exists(pv) and A.only else {}
    old.update(INPUTS)
    json.dump(dict(note='SHA-256 of every results/ file the figure data were exported from', python=platform.python_version(),
                   numpy=np.__version__, inputs=dict(sorted(old.items()))), open(pv, 'w'), indent=1)
    w = max(len(c['check']) for c in allc)
    for c in allc: print(f"  {c['status']:4s} {c['figure']}  {c['check']:<{w}}  {c['value']}   [{c['claim']}]")
    nf = sum(c['status'] == 'FAIL' for c in allc)
    print(f"{nf} failed check(s)" if nf else "all checks passed")
