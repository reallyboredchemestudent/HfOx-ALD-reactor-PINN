"""Load the Lee et al. 2026 HfOx ALD dataset (CSV) into the model's layout and say exactly what was done.

    python data_real.py ../data/lee2026_hfox_ald_dataset.csv            -> ../results/real_Y.npz + load_report.txt
    python data_real.py <csv> --map data_map.json                       (override column detection)
    python data_real.py <csv> --sites sites.json                        (override site positions, model frame, cm)
    python data_real.py <csv> --ncyc 100                                (ALD cycles per wafer if not in the CSV)
    python data_real.py <csv> --flow-angle -90                          (flow direction, degrees from the CSV +y axis)

Output arrays follow the model's order: 35 runs = CONDS (7 deposition x 5 precursor temperatures), 9 sites each.
Y = reported thickness (nm, NaN where missing), RI and WER where present, sites = (9, 2) site positions in the
model frame (x across the flow, y along the flow, cm; wafer centre at y = 7.5), flow direction inferred from data.

Column detection is by name; anything it cannot decide is printed with the fix. To force a mapping, write
data_map.json like {"td": "Deposition temperature (C)", "tp": "TDMAHf temperature (C)", "thickness": "Thickness (nm)",
"site": "Point", "x": "x (mm)", "y": "y (mm)", "ri": "RI", "wer": "WER (nm/min)", "cycles": "Cycles"}.
"""
import numpy as np, pandas as pd, json, re, sys, os, argparse

TD = np.array([100, 125, 150, 175, 200, 225, 250.0]); TP = np.array([55, 60, 65, 70, 75.0])
CONDS = [(td, tp) for td in TD for tp in TP]
SITES_SYNTH = np.array([(0, 4.5), (-1.5, 6), (1.5, 6), (-3, 7.5), (0, 7.5), (3, 7.5), (-1.5, 9), (1.5, 9), (0, 10.5)])

def norm(s): return re.sub(r'[^a-z0-9]', '', str(s).lower())

RULES = {   # role -> list of predicates on the normalised column name (first matching column wins)
    'td': [lambda n: 'dep' in n and 'temp' in n, lambda n: n.startswith('td') and len(n) <= 6, lambda n: 'substrate' in n and 'temp' in n,
           lambda n: 'process' in n and 'temp' in n, lambda n: 'growth' in n and 'temp' in n, lambda n: n in ('tsub', 'tdep', 'temperature', 'tempc', 'tc')],
    'tp': [lambda n: 'prec' in n and 'temp' in n, lambda n: n.startswith('tp') and len(n) <= 6, lambda n: ('source' in n or 'bubbler' in n or 'canister' in n or 'ampoule' in n) and 'temp' in n,
           lambda n: 'tdma' in n and 'temp' in n, lambda n: n in ('tprec', 'tsource')],
    'thickness': [lambda n: 'thick' in n, lambda n: n.startswith('thk'), lambda n: n in ('d', 'dnm', 'tnm', 'film')],
    'ri': [lambda n: n.startswith('ri') and len(n) <= 8, lambda n: 'refract' in n, lambda n: n.startswith('n633') or n.startswith('n632'), lambda n: n in ('n', 'index')],
    'wer': [lambda n: 'wer' in n, lambda n: 'etch' in n],
    'site': [lambda n: n.startswith('site'), lambda n: n.startswith('point') or n in ('pt', 'pos', 'position', 'loc', 'location', 'spot')],
    'x': [lambda n: n in ('x', 'xmm', 'xcm', 'xpos', 'xposition', 'xcoord', 'xcoordinate', 'xmmwafer')],
    'y': [lambda n: n in ('y', 'ymm', 'ycm', 'ypos', 'yposition', 'ycoord', 'ycoordinate', 'ymmwafer')],
    'cycles': [lambda n: 'cycle' in n and 'time' not in n],
    'wafer': [lambda n: n.startswith('wafer') or n.startswith('run') or n.startswith('sample')],
    'pulse': [lambda n: 'pulse' in n],
    'purge': [lambda n: 'purge' in n],
}

def detect(cols, override):
    out, used = {}, set()
    for role, preds in RULES.items():
        if role in override: out[role] = override[role]; used.add(override[role]); continue
        for pr in preds:
            hit = [c for c in cols if c not in used and pr(norm(c))]
            if hit: out[role] = hit[0]; used.add(hit[0]); break
    return out

def wide_to_long(df, log):
    """Sites as columns (e.g. 'Site 1' ... 'Site 9', 'P1'...'P9', 'thickness_1'...) -> one row per site."""
    pat = re.compile(r'^(site|point|p|pt|pos|thickness|thk|t|d)[^0-9]*([1-9])$')
    sitecols = [c for c in df.columns if pat.match(norm(c))]
    if len(sitecols) < 9: return None
    idc = [c for c in df.columns if c not in sitecols]
    long = df.melt(id_vars=idc, value_vars=sitecols, var_name='site', value_name='thickness')
    long['site'] = long['site'].map(lambda c: int(pat.match(norm(c)).group(2)))
    log.append(f"wide layout: melted {len(sitecols)} site columns {sitecols} into rows (site = trailing digit)")
    return long

def snap(v, grid, tol=2.0):
    k = np.argmin(np.abs(grid - v)); return grid[k] if abs(grid[k] - v) <= tol else np.nan

def rotate_to_flow(xy, ang):
    """Rotate centred site positions so the flow direction (angle ang from CSV +y, degrees) points along +y."""
    f = np.array([np.sin(np.radians(ang)), np.cos(np.radians(ang))]); c, s = f[1], f[0]
    return xy@np.array([[c, -s], [s, c]]).T

def pattern_matches(xy, ref, tol=0.25):
    """Flow angles (deg from CSV +y) and mirror flags that map the centred CSV pattern onto ref (centred):
    one best angle per contiguous window of matching angles."""
    out = []
    for mir in (False, True):
        angs = np.arange(-180, 180, 0.25); hits = []
        for ang in angs:
            r = rotate_to_flow(xy, ang)*np.array([-1 if mir else 1, 1])
            dist = max(np.min(np.hypot(*(ref - p).T)) for p in r)
            if dist < tol: hits.append((ang, dist))
        cl = []
        for a_, d_ in hits:
            if cl and a_ - cl[-1][-1][0] <= 0.5: cl[-1].append((a_, d_))
            else: cl.append([(a_, d_)])
        if len(cl) > 1 and cl[0][0][0] <= -179.5 and cl[-1][-1][0] >= 179.5: cl[0] = cl.pop() + cl[0]   # wrap-around
        for c in cl:
            a_, d_ = min(c, key=lambda t: t[1]); out.append((float((a_ + 180) % 360 - 180), mir, d_))
    return out

def infer_frame(xy, Y55, log, flow_angle=None):
    """Centre the site pattern, rotate the flow onto +y, put the wafer centre at y = 7.5 cm.
    Flow axis: if the CSV pattern is the synthetic site map up to a rotation/mirror, the candidates are those
    rotations and the one closest to the downhill thickness direction of the 55 C wafers is taken. Thickness alone is
    not trusted for the angle itself: an inlet tilt rotates the downhill direction by tens of degrees (seen on
    synthetic data: 25 deg), and a free front model fits rotated layouts as well as the true one."""
    xy = xy - xy.mean(0)
    A = np.c_[np.ones(len(xy)), xy]; grads = []
    for t in Y55:
        ok = np.isfinite(t)
        if ok.sum() >= 5: grads.append(np.linalg.lstsq(A[ok], t[ok], rcond=None)[0][1:])
    d = -np.mean(grads, 0); a0 = np.degrees(np.arctan2(d[0], d[1]))
    log.append(f"downhill thickness direction of the 55 C wafers: {a0:+.1f} deg from CSV +y (|grad| {np.linalg.norm(d):.2f} nm/cm)")
    mirror = False
    if flow_angle is not None:
        ang = float(flow_angle); log.append(f"flow direction given on the command line: {ang:+.1f} deg from CSV +y")
    else:
        cands = pattern_matches(xy, SITES_SYNTH - np.array([0, 7.5]))
        if cands:
            gap = lambda c: abs((c[0] - a0 + 180) % 360 - 180)
            best = min(cands, key=lambda c: (gap(c), c[1]))
            ang, mirror = best[0], best[1]
            others = sorted({round(c[0]) for c in cands if not c[1]})
            log.append(f"the CSV sites are the synthetic site map rotated (candidate flow angles {others} deg); took {ang:+.1f} deg, "
                       f"{gap(best):.0f} deg from the downhill direction" + (" (with a left/right mirror)" if mirror else "")
                       + ". CHECK this against the flow arrow in the paper's reactor/site figure; override with --flow-angle.")
            if gap(best) > 35: log.append("WARNING: no candidate is within 35 deg of the downhill direction; the flow axis is uncertain")
        else:
            ang = a0
            log.append("WARNING: the CSV site pattern is not the synthetic site map; using the downhill direction as the flow axis. "
                       "An inlet tilt biases this by up to tens of degrees: read the flow direction off the paper and pass --flow-angle.")
    out = rotate_to_flow(xy, ang)*np.array([-1 if mirror else 1, 1]); out[:, 1] += 7.5
    log.append("  (a left/right mirror is not identifiable from thickness; it only flips the sign of the inlet tilt g1)")
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('csv'); ap.add_argument('--map'); ap.add_argument('--sites'); ap.add_argument('--ncyc', type=float); ap.add_argument('--flow-angle', type=float)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'real_Y.npz'))
    a = ap.parse_args(); log = []
    df = pd.read_csv(a.csv, encoding_errors='replace', sep=None, engine='python')
    log.append(f"read {a.csv}: {df.shape[0]} rows x {df.shape[1]} columns")
    log.append("columns: " + ", ".join(map(str, df.columns)))
    override = json.load(open(a.map)) if a.map else {}
    m = detect(list(df.columns), override)
    if 'thickness' not in m:
        long = wide_to_long(df, log)
        if long is None: sys.exit("\n".join(log) + "\nERROR: no thickness column found; write data_map.json (see the docstring) and rerun with --map")
        df = long; m = detect(list(df.columns), override); m['thickness'] = 'thickness'; m['site'] = 'site'
    for role in ('td', 'tp', 'thickness'):
        if role not in m: sys.exit("\n".join(log) + f"\nERROR: could not find the '{role}' column; give it in data_map.json")
    log.append("column roles: " + json.dumps(m))
    num = lambda c: pd.to_numeric(df[c], errors='coerce')
    td, tp, th = num(m['td']).values, num(m['tp']).values, num(m['thickness']).values
    if np.nanmedian(th) > 40:
        th = th/10; log.append("thickness median > 40: read as angstrom, converted to nm")
    # cycles
    ncyc = a.ncyc
    if ncyc is None and 'cycles' in m:
        cyc = num(m['cycles']).values; u = np.unique(cyc[np.isfinite(cyc)])
        if len(u) == 1: ncyc = float(u[0]); log.append(f"cycles per wafer from the CSV: {ncyc:g}")
        else: log.append(f"WARNING: several cycle counts in the CSV {u}; rows with other counts are moved to real_extra.csv")
    # extra experiments (pulse, purge, cycle series) are kept aside, not mixed into the 35-run design
    keep = np.isfinite(td) & np.isfinite(tp) & np.isfinite(th)
    for role in ('pulse', 'purge', 'cycles'):
        if role in m:
            v = num(m[role]).values; vv = v[keep & np.isfinite(v)]
            if vv.size and len(np.unique(vv)) > 1:
                mode = pd.Series(vv).mode().iloc[0]
                log.append(f"'{m[role]}' varies ({np.unique(vv)}): keeping the base recipe value {mode:g}; other rows -> real_extra.csv "
                           f"(useful later: a pulse-time series at fixed temperatures is a direct test of 'dose only')")
                keep &= (v == mode)
                if role == 'cycles' and ncyc is None: ncyc = float(mode)
    if ncyc is None:
        ncyc = 100.0; log.append("WARNING: cycles per wafer not in the CSV; assuming 100 (pass --ncyc if different)")
    tds = np.array([snap(v, TD) for v in td]); tps = np.array([snap(v, TP) for v in tp])
    extra = ~keep | ~np.isfinite(tds) | ~np.isfinite(tps)
    outdir = os.path.dirname(os.path.abspath(a.out)); os.makedirs(outdir, exist_ok=True)
    if extra.any():
        df[extra].to_csv(os.path.join(outdir, 'real_extra.csv'), index=False)
        log.append(f"{extra.sum()} rows outside the 7 x 5 design (or missing values) written to real_extra.csv")
    d = df[~extra].copy(); d['_td'] = tds[~extra]; d['_tp'] = tps[~extra]; d['_th'] = th[~extra]
    for role in ('ri', 'wer'):
        d['_' + role] = pd.to_numeric(d[m[role]], errors='coerce') if role in m else np.nan
    # ---- sites ----
    if a.sites:
        S = json.load(open(a.sites)); labels = list(S.keys()); sites = np.array([S[k] for k in labels], float)
        key = d[m['site']].astype(str).str.strip().values; log.append(f"site positions from {a.sites} (model frame, cm)")
    elif 'x' in m and 'y' in m:
        x, y = pd.to_numeric(d[m['x']], errors='coerce').values, pd.to_numeric(d[m['y']], errors='coerce').values
        unit = 'mm' if np.nanmax(np.abs(np.r_[x, y])) > 20 or 'mm' in norm(m['x']) else 'cm'
        if unit == 'mm': x, y = x/10, y/10
        key = np.array([f"{u:.2f},{v:.2f}" for u, v in zip(x, y)])
        labels = sorted(set(key), key=lambda s: (float(s.split(',')[1]), float(s.split(',')[0])))
        xy = np.array([[float(v) for v in s.split(',')] for s in labels])
        log.append(f"site positions from columns {m['x']}, {m['y']} (read as {unit})")
        sites = None
    elif 'site' in m:
        key = d[m['site']].astype(str).str.strip().values
        labels = sorted(set(key), key=lambda s: (len(s), s))
        sites = SITES_SYNTH[:len(labels)].copy()
        log.append("WARNING: only site labels, no coordinates. Labels " + str(labels) + " were assigned IN ORDER to the synthetic "
                   "pattern " + str(SITES_SYNTH.tolist()) + ". Check this against the paper's site map and pass --sites sites.json "
                   "({label: [x_cm, y_cm]} in the model frame: y along the flow, wafer centre at 7.5) if it differs.")
    else:
        sys.exit("\n".join(log) + "\nERROR: no site label or x/y columns; give them in data_map.json")
    if len(labels) != 9:
        sys.exit("\n".join(log) + f"\nERROR: found {len(labels)} distinct sites {labels}; the model expects 9 per wafer. Fix with --map/--sites.")
    lab_idx = {k: i for i, k in enumerate(labels)}
    Y = np.full((35, 9), np.nan); RI = np.full((35, 9), np.nan); WER = np.full((35, 9), np.nan); REP = np.full((35, 9), np.nan)
    nrep = np.zeros((35, 9), int)
    for (ctd, ctp, k), grp in d.groupby(['_td', '_tp', key]):
        r = CONDS.index((ctd, ctp)); s = lab_idx[k]
        Y[r, s] = grp['_th'].mean(); RI[r, s] = grp['_ri'].mean(); WER[r, s] = grp['_wer'].mean(); nrep[r, s] = len(grp)
        if len(grp) > 1: REP[r, s] = grp['_th'].std(ddof=1)
    if nrep.max() > 1:
        log.append(f"repeat measurements at {np.sum(nrep > 1)} wafer-sites (up to {nrep.max()}) averaged; their sd is stored as rep_sd "
                   f"(median {np.nanmedian(REP):.3f} nm: compare with the instrument model's site noise 0.05 nm)")
    if sites is None:
        dl = [r for r, c in enumerate(CONDS) if c[1] == 55.0]
        sites = infer_frame(xy, Y[dl], log, a.flow_angle)
    if a.sites is None and 'x' in m:
        # compare with the synthetic pattern (nearest-site distance), for the record only
        dmin = np.array([np.min(np.hypot(*(SITES_SYNTH - s).T)) for s in sites])
        log.append(f"distance of each real site to the nearest synthetic site (cm): {np.round(dmin, 2).tolist()}")
    order = np.lexsort((sites[:, 0], sites[:, 1]))         # sort sites upstream -> downstream, then left -> right
    sites, Y, RI, WER, REP = sites[order], Y[:, order], RI[:, order], WER[:, order], REP[:, order]
    labels = [labels[i] for i in order]
    if np.any(np.abs(sites[:, 0]) > 7.4) or np.any(sites[:, 1] > 14.9) or np.any(sites[:, 1] < 0):
        log.append("WARNING: a site falls outside the model domain (x in +-7.5, y in 0..15 cm); check units / --sites")
    miss = [f"{CONDS[r][0]:.0f}/{CONDS[r][1]:.0f} C site {labels[s]}" for r in range(35) for s in range(9) if not np.isfinite(Y[r, s])]
    log.append(f"filled {np.isfinite(Y).sum()} of 315 wafer-sites" + (f"; missing: {miss[:12]}{' ...' if len(miss) > 12 else ''}" if miss else ""))
    # plausibility: saturated film and implied GPC
    sat = np.array([np.nanmedian(Y[CONDS.index((t, 75.0))]) for t in TD])
    log.append("median thickness of the 75 C (saturated) wafer per deposition temperature: " + ", ".join(f"{t:.0f} C {s:.2f} nm" for t, s in zip(TD, sat)))
    gpc = sat*10/ncyc
    log.append(f"implied GPC at {ncyc:g} cycles (A/cycle): " + ", ".join(f"{g:.3f}" for g in gpc))
    if not (0.6 < np.nanmedian(gpc) < 2.2):
        log.append("WARNING: implied GPC is outside 0.6-2.2 A/cycle; the cycle count is probably wrong (pass --ncyc)")
    import instrument as INS
    top = float(INS.marginal(np.array(INS.TMAX - 0.5))[0])      # reported thickness near the top of the lookup table
    if np.nanmax(Y) > top:
        log.append(f"WARNING: thickest reported film {np.nanmax(Y):.2f} nm is beyond the instrument table (reported {top:.2f} nm at "
                   f"{INS.TMAX - 0.5:.1f} nm true): run `cd code && python make_lookup.py {int(np.ceil(np.nanmax(Y) + 3))}` once")
    else:
        log.append(f"thickest reported film {np.nanmax(Y):.2f} nm: inside the instrument table (covers reported films up to {top:.2f} nm)")
    np.savez(a.out, Y=Y, RI=RI, WER=WER, rep_sd=REP, nrep=nrep, sites=sites, site_labels=np.array(labels), ncyc=ncyc,
             conds=np.array(CONDS), columns=json.dumps(m))
    rep = "\n".join(log); print(rep)
    open(os.path.join(outdir, 'load_report.txt'), 'w').write(rep + "\n")
    print(f"\nwrote {os.path.abspath(a.out)}")

if __name__ == '__main__':
    main()
