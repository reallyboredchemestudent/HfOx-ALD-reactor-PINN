"""Usage: python compare.py <A|B> <seeds 0,1,..> <holds 0,15,30> [pinn tag, e.g. JA, or NONE for FV only] [pinn dir]
Finite-volume adjoint inverse vs PINN on hold-one-wafer-out (55 C wafers at 100, 175, 250 C), same data.
Intervals for both: epistemic part (FV: Laplace; PINN: deep-ensemble spread) + the held-out run's unknown dose,
sampled through the FV solver at each method's own fitted parameters (PINN: its learned inlet profile) + instrument.
PINN files are read from [pinn dir] (default: the current directory) and the comparison is written there too.
Exits 1 if a PINN tag other than NONE is given and no PINN file is found for any hold-out."""
import numpy as np, sys, glob, warnings
warnings.filterwarnings('ignore')
from fvmodel import CONDS, forward
import instrument as INS
from fit import SIG_O
from scipy.interpolate import interp1d
tag = sys.argv[1]; seeds = [int(s) for s in sys.argv[2].split(',')]; holds = [int(h) for h in sys.argv[3].split(',')]
ptag = sys.argv[4] if len(sys.argv) > 4 else tag     # PINN file tag: e.g. JA for the JAX PINN on synthetic set A
pdir = sys.argv[5] if len(sys.argv) > 5 else '.'      # where the PINN files are (and where the comparison is written)
n_found = 0
Z = np.load(f'synth{tag}_rep0.npz'); Y = Z['Y']; TT = Z['T']
FV = {r['hold']: r for r in np.load(f'howo_fv_{tag}.npy', allow_pickle=True)}
rng = np.random.default_rng(11)
def dose_samples(p, c, h, gfun=None, n=120):
    out = []
    for k in range(n):
        o = np.zeros(35); o[h] = rng.normal(0, SIG_O)
        out.append(forward(p, o, gfun, conds=[c])[0])
    return np.array(out)
def phys_to_fv(phys):   # PINN physics vector -> FV parameter vector (inlet coefficients unused when gfun given)
    return np.array([phys[0], phys[1], phys[2], phys[3], phys[4], 0.0, 0.0, phys[5], phys[6]])
rows = []
for h in holds:
    c = CONDS[h]; obs = Y[h]
    # ---- FV ----
    r = FV[h]; p = r['p']
    T_, S_, So_ = forward(p, np.zeros(35), sens=True, conds=[c]); m0, s0, dm0 = INS.marginal(T_[0])
    var_par = np.maximum(r['sd']**2 - (dm0*So_[0])**2*SIG_O**2 - s0**2, 0)
    D = dose_samples(p, c, h)
    m, s, _ = INS.marginal(D)
    sims = m + rng.normal(0, 1, m.shape)*np.sqrt(s**2 + var_par)
    lo, hi, med = np.percentile(sims, 5, 0), np.percentile(sims, 95, 0), np.median(sims, 0)
    rows.append(('FV', h, np.sqrt(np.mean((obs-med)**2)), np.mean((obs>=lo)&(obs<=hi)), np.mean(hi-lo), np.sqrt(np.mean((np.median(D,0)-TT[h])**2))))
    # ---- PINN ensemble ----
    files = [f'{pdir}/pinn_{ptag}_s{s}_h{h}.npz' for s in seeds if glob.glob(f'{pdir}/pinn_{ptag}_s{s}_h{h}.npz')]
    n_found += len(files)
    CHI2_MAX = float(__import__('os').environ.get('CHI2_MAX', 330))      # screen failed members (data chi2 of ~306 points)
    kept = [f for f in files if float(np.load(f)['comp'][2]) < CHI2_MAX]
    print(f"hold {CONDS[h]}: {len(kept)} of {len(files)} PINN members with data chi2 < {CHI2_MAX:g}"); files = kept
    if not files: continue
    preds = []; sims_all = []
    for f in files:
        z = np.load(f); pred_t = z['pred'][h]; phys = z['phys']
        xi = np.linspace(-1, 1, 41); gf = interp1d(xi, z['g'], fill_value='extrapolate')
        Dp = dose_samples(phys_to_fv(phys), c, h, gfun=lambda x, gf=gf: gf(x), n=60)
        # shift the FV dose samples so their median sits on the PINN's own prediction
        Dp = Dp - np.median(Dp, 0) + pred_t
        preds.append(pred_t); sims_all.append(Dp)
    preds = np.array(preds); Dp = np.concatenate(sims_all)
    ens_var = preds.var(0, ddof=1) if len(preds) > 1 else np.zeros(9)
    m, s, _ = INS.marginal(Dp)
    sims = m + rng.normal(0, 1, m.shape)*np.sqrt(s**2 + ens_var)
    lo, hi, med = np.percentile(sims, 5, 0), np.percentile(sims, 95, 0), np.median(sims, 0)
    rows.append((f'PINN x{len(files)}', h, np.sqrt(np.mean((obs-med)**2)), np.mean((obs>=lo)&(obs<=hi)), np.mean(hi-lo), np.sqrt(np.mean((preds.mean(0)-TT[h])**2))))
for row in rows: print(f"{row[0]:8s} hold {CONDS[row[1]]}: RMSE vs reported {row[2]:.2f} nm, vs true film {row[5]:.2f} nm, inside 90% {row[3]:.2f}, mean width {row[4]:.2f} nm")
np.save(f'{pdir}/compare_{tag}{"" if ptag == tag else "_" + ptag}.npy', np.array(rows, dtype=object), allow_pickle=True)
if ptag.upper() != 'NONE' and n_found == 0:
    sys.exit(f"compare.py: no PINN files pinn_{ptag}_s*_h*.npz in {pdir}: the PINN step did not run or wrote elsewhere")
