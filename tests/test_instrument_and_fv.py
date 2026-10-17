"""Instrument model, FV forward model and FV inverse on synthetic set A.

Tolerances:
  instrument   rtol 1e-12  pure interpolation of a shipped table; nothing iterative.
  forward      rtol 1e-9   one sparse LU and a Newton solve per condition; differences across BLAS builds are at
                           round-off (~1e-13), far below 1e-9, while any real change to the physics moves T by > 1e-6.
  fit          abs 1e-4 on parameters, 0.01 on chi2. The endpoint of a bounded trust-region least-squares fit; the
               cloud-recorded check_install_expected.json and this machine (NumPy 2.5.3, SciPy 1.18.1) agree to 2e-11,
               so 1e-4 leaves room for other BLAS/LAPACK builds while staying 10x tighter than check_install's 1e-3. The fit is also held to the published
               check_install numbers at check_install's own tolerance.
"""
import numpy as np
import pytest

from conftest import CODE
from golden_io import load_json


def test_instrument_golden():
    import instrument as INS
    g = load_json("instrument.json")
    t = np.array(g["t"])
    m, s, dm = INS.marginal(t)
    _, _, _, ds = INS.marginal_full(t)
    np.testing.assert_allclose(m, g["mean"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(s, g["sd"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(dm, g["dmean"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(ds, g["dsd"], rtol=1e-12, atol=1e-12)
    cols = np.array([INS.generate(t, k) for k in range(INS.A.size)])
    np.testing.assert_allclose(cols, g["generate"], rtol=1e-12, atol=1e-12)


def test_instrument_published_example():
    """README fact 5 and check_install: a real 2 nm film is reported as about 3.68 nm on average."""
    import json
    import instrument as INS
    exp = json.loads((CODE / "check_install_expected.json").read_text())
    m, _, _ = INS.marginal(np.array([2.0, 10.0]))
    assert abs(m[0] - exp["ins2"]) < 1e-6 and abs(m[1] - exp["ins10"]) < 1e-6


def test_fv_forward_golden():
    import fvmodel as FV
    g = load_json("fv_forward.json")
    T, S, So = FV.forward(FV.TRUTH_A, sens=True)
    np.testing.assert_allclose(T, g["T"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(np.asarray(S), g["S"], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(np.asarray(So), g["So"], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(FV.forward(FV.TRUTH_A, np.array(g["offsets"])), g["T_offsets"], rtol=1e-9, atol=1e-12)


@pytest.mark.slow
def test_fv_fit_synthA_golden():
    """The FV inverse on synthetic set A (what check_install.py runs): about 150 s on one core."""
    import json
    import fit as F
    g = load_json("fv_fit_synthA.json")
    Y = np.load(CODE / "synthA_rep0.npz")["Y"]
    p0 = F.PRIOR_MU.copy(); p0[0] = np.log(22); p0[1] = np.log(1.2)
    r = F.fit(Y, p0)
    np.testing.assert_allclose(r["p"], g["p"], rtol=0, atol=1e-4)
    np.testing.assert_allclose(np.sqrt(np.diag(r["cov"])[:9]), g["sd"], rtol=1e-3)
    assert abs(r["chi2_data"] - g["chi2_data"]) < 0.01
    assert r["nfev"] == g["nfev"]
    exp = json.loads((CODE / "check_install_expected.json").read_text())   # the cloud-recorded reference
    assert np.max(np.abs(r["p"] - np.array(exp["p"]))) < 1e-3 and abs(r["chi2_data"] - exp["chi2"]) < 0.05
