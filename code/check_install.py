"""Install check: the FV inverse on the shipped synthetic data set must reproduce the reference numbers in
check_install_expected.json (same code, same data, deterministic). Also checks the instrument lookup and the package
versions. Exits 1 if anything differs.

    python check_install.py          (about 1-4 minutes)
"""
import numpy as np, scipy, time, json, os, sys, warnings
warnings.filterwarnings('ignore')
import fit as F, fvmodel as FV, instrument as INS
EXPECTED = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'check_install_expected.json')))
print(f"numpy {np.__version__}, scipy {scipy.__version__}, python {sys.version.split()[0]}")
m, s, dm = INS.marginal(np.array([2.0, 10.0]))
print(f"instrument: true 2 nm -> reported {m[0]:.4f} (expected {EXPECTED['ins2']:.4f}); 10 nm -> {m[1]:.4f} (expected {EXPECTED['ins10']:.4f})")
Y = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'synthA_rep0.npz'))['Y']
p0 = F.PRIOR_MU.copy(); p0[0] = np.log(22); p0[1] = np.log(1.2)
t = time.time(); r = F.fit(Y, p0); dt = time.time() - t
d = np.max(np.abs(r['p'] - np.array(EXPECTED['p'])))
print(f"FV fit of synthetic set A: data chi2 {r['chi2_data']:.3f} (expected {EXPECTED['chi2']:.3f}), {dt:.0f} s, {r['nfev']} evaluations")
for k, nm in enumerate(FV.NAMES): print(f"  {nm:8s} {r['p'][k]:9.5f}  expected {EXPECTED['p'][k]:9.5f}  truth {FV.TRUTH_A[k]:8.4f}")
ok = d < 1e-3 and abs(r['chi2_data'] - EXPECTED['chi2']) < 0.05 and abs(m[0] - EXPECTED['ins2']) < 1e-6
print("INSTALL CHECK", "PASSED" if ok else f"DIFFERS (max parameter difference {d:.2e}); see docs/SETUP.md, 'If a check fails'")
sys.exit(0 if ok else 1)
