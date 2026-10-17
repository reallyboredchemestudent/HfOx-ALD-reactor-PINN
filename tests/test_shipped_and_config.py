"""The inputs every published number depends on, and the import-time configuration.

Tolerance: exact. These are bytes and integers; any change is a change to the inputs.
"""
import hashlib

import numpy as np
import pytest

from conftest import CODE, PINNED_ENV, REPO
from golden_io import load_json


def test_shipped_files_unchanged():
    """synthA/B, the reference FV hold-out fits, the instrument lookup and the dataset are inputs, not outputs.
    fit.py and make_B.py rewrite synthA/synthB in place when run from code/; this catches a changed rewrite."""
    want = load_json("shipped_files.json")
    changed = [p for p, h in want.items() if not p.startswith("_")
               and hashlib.sha256((REPO / p).read_bytes()).hexdigest() != h]
    assert not changed, f"shipped input files changed: {changed}"


def test_import_time_config_pinned():
    """fvmodel and pinn read ALD_NCYC / PDE_W / PHYS_SCALE once at import; check the pinned values took effect."""
    import fvmodel as FV
    import pinn as PN
    assert FV.NCYC == float(PINNED_ENV["ALD_NCYC"]) and FV.THK == FV.NCYC / 10
    assert PN.PDE_W == float(PINNED_ENV["PDE_W"]) and PN.PHYS_SCALE == float(PINNED_ENV["PHYS_SCALE"])
    assert PN.THK == FV.THK, "pinn.THK is a copy taken at import; it has drifted from fvmodel.THK"


def test_dataset_shape():
    import pandas as pd
    df = pd.read_csv(REPO / "data/lee2026_hfox_ald_dataset.csv")
    assert df.shape == (315, 9)
    assert set(df["Cycles"]) == {100}
    assert df.groupby(["Deposition temperature (C)", "Precursor temperature (C)"]).size().eq(9).all()


@pytest.mark.parametrize("tag", ["A", "B"])
def test_synthetic_sets_regenerate(tag):
    """The synthetic data sets are regenerated bit for bit from their seeds. NumPy keeps the bit generator stable
    across releases but not every distribution method, so this is the test that says when an upgrade changed them."""
    import fit as F
    import fvmodel as FV
    rng = np.random.default_rng(0)
    if tag == "A":
        Y, T, off = F.make_data(FV.TRUTH_A, rng)
    else:
        import make_B
        Y, T, off = F.make_data(FV.TRUTH_A, rng, gfun=make_B.gB)
    z = np.load(CODE / f"synth{tag}_rep0.npz")
    np.testing.assert_allclose(off, z["off"], rtol=0, atol=1e-15)
    np.testing.assert_allclose(T, z["T"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(Y, z["Y"], rtol=1e-12, atol=1e-12)


def test_require_ncyc_catches_a_stale_import():
    """Importing fvmodel before ALD_NCYC is set fixes the cycle count at the default; the entry scripts' guard
    must turn that into an error instead of a silently rescaled thickness."""
    import fvmodel as FV
    import pinn as PN
    FV.require_ncyc(100, PN)                      # the pinned configuration passes
    with pytest.raises(RuntimeError, match="configured for 100 cycles but the data say 200"):
        FV.require_ncyc(200)
    class Stale:                                  # a module that copied THK under another cycle count
        __name__ = "stale"; THK = 20.0
    with pytest.raises(RuntimeError, match="stale THK"):
        FV.require_ncyc(100, Stale)
