"""Compare aldFoam's final wall coverage with this project's steady gap-resolved exposure solution.

    python compare_aldfoam.py <caseA> [<caseB> <caseC> ...] [--h 1.0] [--patch bottom] [--time latest] [--field F --key K]

Each case is an OpenFOAM case directory run to the end of its purge with `writeFormat ascii;`. The script finds the
coverage on the wall patch (any keyed `nonuniform List<scalar>` with values in [0, 1] on that patch: a coverage field's
`value`, or a coverage entry stored by the reactive boundary condition) and the face centres (`C`/`Cx` from
`postProcess -func writeCellCentres -latestTime`, otherwise a uniform 480-face patch along 24 cm). It reports:
  1. aldFoam vs the steady exposure solution at the nominal dose (same inputs as gapresolved.py)
  2. the same at matched mass (reach adjusted so both consume the same number of sites): front shape only
  3. the best depth-averaged steady fit (reach, width, dispersion: the PINN's equation) and its worst misfit
  4. case vs case: the exposure reduction says pulse length and strength do not matter at equal dose
All differences are also given in nm of film (x 15 nm, the saturated film of the synthetic study).
Writes aldfoam_compare.txt / .csv / .png next to the first case (or --out).
"""
import os, re, argparse
import numpy as np

def tokenize(text):
    text = re.sub(r'/\*.*?\*/', ' ', text, flags=re.S); text = re.sub(r'//[^\n]*', ' ', text)
    return re.findall(r'[{}();]|[^\s{}();]+', text)

def parse_list(tk, i):
    out = []
    while tk[i] != ')':
        if tk[i] == '(': sub, i = parse_list(tk, i + 1); out.append(sub)
        else: out.append(tk[i]); i += 1
    return out, i + 1

def parse_dict(tk, i, top=False):
    d = {}
    while i < len(tk):
        t = tk[i]
        if t == '}': return d, i + 1
        if t.startswith('#'): i += 2; continue            # #include "file" etc.
        key = t; i += 1
        if i < len(tk) and tk[i] == '{':
            sub, i = parse_dict(tk, i + 1); d[key] = sub
        else:
            vals = []
            while i < len(tk) and tk[i] != ';':
                if tk[i] == '(': lst, i = parse_list(tk, i + 1); vals.append(lst)
                else: vals.append(tk[i]); i += 1
            d[key] = vals; i += 1
    return d, i

def read_foam(path):
    with open(path, 'r', errors='replace') as f: txt = f.read()
    if re.search(r'format\s+binary', txt): raise ValueError(f"{path} is binary: set `writeFormat ascii;` in system/controlDict and rerun")
    return parse_dict(tokenize(txt), 0, top=True)[0]

def as_array(vals):
    """OpenFOAM value entry -> numpy array (scalar list, vector list as (N,3), or a uniform scalar)."""
    if not vals: return None
    if vals[0] == 'uniform':
        v = vals[1]
        return np.array([float(x) for x in v]) if isinstance(v, list) else np.array([float(v)])
    lst = [v for v in vals if isinstance(v, list)]
    if not lst: return None
    L = lst[-1]
    try:
        return np.array([[float(x) for x in e] for e in L]) if L and isinstance(L[0], list) else np.array([float(x) for x in L])
    except ValueError:
        return None

def time_dir(case, t):
    ts = []
    for d in os.listdir(case):
        try: ts.append((float(d), d))
        except ValueError: pass
    if not ts: raise SystemExit(f"{case}: no time directories")
    return os.path.join(case, max(ts)[1] if t == 'latest' else min(ts, key=lambda p: abs(p[0] - float(t)))[1])

def find_coverage(case, patch, t, field=None, key=None):
    td = time_dir(case, t); cands = []
    for fn in sorted(os.listdir(td)):
        p = os.path.join(td, fn)
        if not os.path.isfile(p) or (field and fn != field): continue
        try: F = read_foam(p)
        except ValueError as e: raise SystemExit(str(e))
        except Exception: continue
        bf = F.get('boundaryField', {})
        for pn, pd in bf.items():
            if patch and pn != patch: continue
            if not isinstance(pd, dict): continue
            for k, v in pd.items():
                if key and k != key: continue
                a = as_array(v) if isinstance(v, list) else None
                if a is None or a.ndim != 1 or a.size < 10: continue
                if a.min() >= -1e-6 and a.max() <= 1 + 1e-6:
                    score = 2*bool(re.search(r'cov|theta', k, re.I)) + bool(re.search(r'cov|theta', fn, re.I))
                    cands.append((score, fn, pn, k, a))
    if not cands: raise SystemExit(f"{case}: no [0,1] list on patch '{patch}' in {td}; check --patch/--field/--key and ascii output")
    cands.sort(key=lambda c: -c[0])
    return td, cands

def face_x(td, patch, n, Ly=0.24):
    for fn in ('Cx', 'C'):
        p = os.path.join(td, fn)
        if os.path.exists(p):
            F = read_foam(p); a = as_array(F['boundaryField'][patch]['value'])
            if a is not None: return (a if a.ndim == 1 else a[:, 0]), f"face centres from {fn}"
    return (np.arange(n) + 0.5)*Ly/n, f"uniform faces along {Ly*100:.0f} cm (no C/Cx field found)"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('cases', nargs='+'); ap.add_argument('--h', type=float, default=1.0, help='gap (cm)')
    ap.add_argument('--u', type=float, default=None, help='mean speed (cm/s); default from the gap (u h = 179 cm^2/s)')
    ap.add_argument('--patch', default='bottom'); ap.add_argument('--time', default='latest'); ap.add_argument('--field'); ap.add_argument('--key')
    ap.add_argument('--out'); ap.add_argument('--labels')
    a = ap.parse_args()
    import gapresolved as GR, aldfoam_params as AP
    from scipy.optimize import least_squares, brentq
    u = a.u if a.u else 179.0/a.h
    s = GR.setup(a.h, u, vbar=AP.vth*100)
    out_dir = a.out or os.path.dirname(os.path.abspath(a.cases[0])); os.makedirs(out_dir, exist_ok=True); log = []
    say = lambda m: (print(m), log.append(m))
    say(f"gap {a.h} cm, mean speed {u:.0f} cm/s, sticking {s['beta']:.4g}, D {s['D']} cm^2/s, reach 16 cm, width 0.9 cm")
    th_s = GR.steady_exposure(s); y = s['yc']
    labels = a.labels.split(',') if a.labels else [os.path.basename(os.path.normpath(c)) for c in a.cases]
    curves = []
    for c, lab in zip(a.cases, labels):
        td, cands = find_coverage(c, a.patch, a.time, a.field, a.key)
        sc, fn, pn, k, th = cands[0]
        x, src = face_x(td, pn, th.size); x = x*100                      # cm
        o = np.argsort(x); x, th = x[o], th[o]
        say(f"\n[{lab}] {td}: coverage = '{k}' on patch '{pn}' of field '{fn}' ({th.size} faces; {src}); "
            + (f"other candidates: {[(cc[1], cc[3]) for cc in cands[1:4]]}" if len(cands) > 1 else "only candidate"))
        ours = np.interp(x, y, th_s[0])
        d1 = np.max(np.abs(th - ours)); say(f"  1. vs steady exposure, nominal dose: max |dtheta| {d1:.4f} ({15*d1:.3f} nm)")
        mass = np.trapezoid(th, x) if hasattr(np, 'trapezoid') else np.trapz(th, x)
        def mass_ours(lam):
            s2 = GR.setup(a.h, u, vbar=AP.vth*100, lam=lam); t2 = GR.steady_exposure(s2)[0]
            return (np.trapezoid(np.interp(x, y, t2), x) if hasattr(np, 'trapezoid') else np.trapz(np.interp(x, y, t2), x)) - mass
        try:
            lam_m = brentq(mass_ours, 4.0, 23.0, xtol=1e-3)
            t2 = np.interp(x, y, GR.steady_exposure(GR.setup(a.h, u, vbar=AP.vth*100, lam=lam_m))[0])
            d2 = np.max(np.abs(th - t2)); say(f"  2. matched mass (reach {lam_m:.2f} cm vs nominal 16): max |dtheta| {d2:.4f} ({15*d2:.3f} nm)")
        except ValueError:
            lam_m, d2 = np.nan, np.nan; say("  2. matched mass: no reach in 4-23 cm matches (front off the channel?)")
        from fvmodel import Grid, solve as solve_da
        def model(q):
            lam, zb, l = np.exp(q); g = Grid(0.0, 1, 480, W=1.0, Lend=24.0)
            e, *_ = solve_da(g, lam, zb, l, np.ones(1)); return np.interp(x, g.yc, 1 - np.exp(-e))
        r = least_squares(lambda q: model(q) - th, np.log([16.0, 0.9, s['D']/u]))
        (lf, zf, lf_), err = np.exp(r.x), np.max(np.abs(r.fun))
        say(f"  3. depth-averaged steady fit: reach {lf:.2f} cm, width {zf:.3f} cm, dispersion {lf_:.3f} cm (D/u {s['D']/u:.3f}); worst misfit {err:.4f} ({15*err:.3f} nm)")
        curves.append(dict(label=lab, x=x, th=th, d_nominal=d1, d_mass=d2, lam_mass=lam_m, fit=[lf, zf, lf_], fit_err=err))
    if len(curves) > 1:
        say("\n4. case vs case (equal dose): the exposure reduction predicts no difference")
        ref = curves[0]
        for cv in curves[1:]:
            d = np.max(np.abs(np.interp(ref['x'], cv['x'], cv['th']) - ref['th'])); cv['d_vs_first'] = d
            say(f"  {cv['label']} vs {ref['label']}: max |dtheta| {d:.4f} ({15*d:.3f} nm)")
    open(os.path.join(out_dir, 'aldfoam_compare.txt'), 'w').write("\n".join(log) + "\n")
    import csv
    with open(os.path.join(out_dir, 'aldfoam_compare.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['case', 'y_cm', 'theta_aldfoam', 'theta_steady_nominal'])
        for cv in curves:
            for xi, ti in zip(cv['x'], cv['th']): w.writerow([cv['label'], round(xi, 4), ti, np.interp(xi, y, th_s[0])])
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    COLS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
    SURF, _INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
    plt.rcParams.update({'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED,
                         'axes.facecolor': SURF, 'figure.facecolor': SURF, 'font.size': 10})
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(y, th_s[0], color=MUTED, lw=4, alpha=0.6, label='steady exposure (this project)', zorder=1)
    for k, cv in enumerate(curves): ax.plot(cv['x'], cv['th'], color=COLS[k % 8], lw=2, label=f"aldFoam {cv['label']}", zorder=2)
    ax.set_xlabel('distance along the flow (cm)'); ax.set_ylabel('final wall coverage')
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True); [ax.spines[k].set_visible(False) for k in ('top', 'right')]
    ax.legend(frameon=False, labelcolor=INK2, fontsize=9); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'aldfoam_compare.png'), dpi=170)
    print(f"\nwrote aldfoam_compare.txt/.csv/.png in {out_dir}")

if __name__ == '__main__':
    main()
