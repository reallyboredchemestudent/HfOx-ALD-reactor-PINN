"""The JAX PINN (pinn_jax.py) on synthetic set A.

1. The self-check: JAX value and gradient against the NumPy PINN (itself checked by finite differences in
   test_pinn.py). pinn_jax.py prints FAILED but exits 0 on failure, so the verdict line is what is tested.
2. A short deterministic run frozen in tests/golden/pinn_jax.json. That golden is recorded on the first machine
   with JAX (`python tests/record_golden.py --jax`) and committed; until then the test is skipped, not passed.
   Tolerance: abs 1e-4 on the physics parameters, for the same reason as the NumPy short training.
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from conftest import CODE, run_script
from golden_io import GOLDEN

jax = pytest.importorskip("jax")
pytestmark = pytest.mark.jax


def test_selfcheck_passes():
    out = run_script(CODE, "pinn_jax.py", "selfcheck")
    assert "SELF-CHECK PASSED" in out, out


@pytest.mark.slow
def test_short_run_golden():
    f = GOLDEN / "pinn_jax.json"
    if not f.exists():
        pytest.skip("no JAX golden recorded yet: run `python tests/record_golden.py --jax` once and commit it")
    gold = json.loads(f.read_text())
    with tempfile.TemporaryDirectory() as td:
        run_script(CODE, "pinn_jax.py", "run", "--data", "synthA_rep0.npz", "--tag", "GOLD", "--seeds", "0",
                   "--holds=-1", "--stages", "2", "--iters", "40", "--pre-stages", "1", "--pre-iters", "40", "--out", td)
        z = np.load(Path(td) / "pinn_GOLD_s0_h-1.npz")
        np.testing.assert_allclose(z["phys"], gold["phys"], rtol=0, atol=1e-4)
        np.testing.assert_allclose(z["pred"], gold["pred"], rtol=0, atol=1e-3)
