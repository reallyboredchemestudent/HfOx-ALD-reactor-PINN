"""Compare two exports of the figure data (e.g. the committed paper/figures/data/ and a fresh one from a clean clone).

    python compare_figdata.py <reference dir> <new dir> [--rtol 1e-4] [--atol 1e-6]

Numeric CSV cells must agree to rtol/atol, text cells exactly; PNG rasters (model fields) to 2/255 per channel;
checks.csv must give every check the same status. provenance.json is not compared (hashes of inputs differ by design
when results are regenerated). Exits 1 on any difference, listing the first few per file.
"""
import os, sys, csv, argparse
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument('ref'); ap.add_argument('new')
ap.add_argument('--rtol', type=float, default=1e-4); ap.add_argument('--atol', type=float, default=1e-6)
A = ap.parse_args()

def num(v):
    try: return float(v)
    except ValueError: return None

def cmp_csv(a, b, sep):
    ra = list(csv.reader(open(a), delimiter=sep)); rb = list(csv.reader(open(b), delimiter=sep)); bad = []
    if len(ra) != len(rb): return [f"{len(ra)} rows vs {len(rb)}"]
    for i, (x, y) in enumerate(zip(ra, rb)):
        if len(x) != len(y): bad.append(f"row {i}: {len(x)} vs {len(y)} cells"); continue
        for j, (u, v) in enumerate(zip(x, y)):
            fu, fv = num(u), num(v)
            if fu is not None and fv is not None:
                if not (np.isclose(fu, fv, rtol=A.rtol, atol=A.atol) or (np.isnan(fu) and np.isnan(fv))):
                    bad.append(f"row {i} col {ra[0][j] if j < len(ra[0]) else j}: {u} vs {v}")
            elif u != v and os.path.basename(a) != 'checks.csv':
                bad.append(f"row {i} col {j}: '{u}' vs '{v}'")
    return bad

def cmp_checks(a, b):
    sa = {(r['figure'], r['check']): r['status'] for r in csv.DictReader(open(a))}
    sb = {(r['figure'], r['check']): r['status'] for r in csv.DictReader(open(b))}
    return [f"{k}: {sa.get(k)} vs {sb.get(k)}" for k in sorted(set(sa) | set(sb)) if sa.get(k) != sb.get(k)]

def cmp_png(a, b):
    import matplotlib.image as mpimg
    x, y = mpimg.imread(a), mpimg.imread(b)
    if x.shape != y.shape: return [f"shape {x.shape} vs {y.shape}"]
    d = np.abs(x.astype(float) - y.astype(float)).max()
    return [f"max pixel difference {d*255:.1f}/255"] if d > 2/255 + 1e-9 else []

fa = {f for f in os.listdir(A.ref) if not f.startswith('.')} - {'provenance.json'}
fb = {f for f in os.listdir(A.new) if not f.startswith('.')} - {'provenance.json'}
problems = {}
for f in sorted(fa ^ fb): problems[f] = ['only in ' + ('reference' if f in fa else 'new')]
for f in sorted(fa & fb):
    a, b = os.path.join(A.ref, f), os.path.join(A.new, f)
    if f == 'checks.csv': p = cmp_checks(a, b)
    elif f.endswith('.csv'): p = cmp_csv(a, b, ';' if f == 'fig2_si_table.csv' else ',')
    elif f.endswith('.png'): p = cmp_png(a, b)
    else: p = [] if open(a, 'rb').read() == open(b, 'rb').read() else ['differs']
    if p: problems[f] = p
for f, p in problems.items():
    print(f"DIFFERS {f}: " + "; ".join(p[:4]) + (f" (+{len(p) - 4} more)" if len(p) > 4 else ''))
print(f"{len(fa & fb) - len([f for f in problems if f in fa & fb])} of {len(fa | fb)} files agree"
      + (f" (rtol {A.rtol}, atol {A.atol})" if not problems else ''))
sys.exit(1 if problems else 0)
