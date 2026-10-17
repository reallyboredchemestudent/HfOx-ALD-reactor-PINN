"""Shared setup for the golden-master tests.

The model modules read their configuration from the environment once, at import time (fvmodel.NCYC from
ALD_NCYC, pinn.PDE_W / pinn.PHYS_SCALE, instrument's lookup from ALD_LOOKUP). A value left over in the shell
would silently change every number, so the configuration the goldens were recorded under is pinned here,
before any model module is imported, and test_config.py checks that it took effect.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CODE = REPO / "code"
GOLDEN = Path(__file__).resolve().parent / "golden"

PINNED_ENV = {
    "ALD_NCYC": "100",        # Lee et al.: 100 cycles per wafer (the CSV's Cycles column)
    "PDE_W": "20.0",          # PINN study configuration, as documented in pinn_jax.py
    "PHYS_SCALE": "0.01",
    "OMP_NUM_THREADS": "1",   # single-threaded BLAS: reductions in a fixed order
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}
os.environ.update(PINNED_ENV)
os.environ.pop("ALD_LOOKUP", None)
for _k in ("N55", "NSAT", "STAGES", "FAIR_START", "CHI2_MAX"):
    os.environ.pop(_k, None)

if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: takes more than about 30 s (deselect with -m 'not slow')")
    config.addinivalue_line("markers", "jax: needs JAX installed")


@pytest.fixture
def repo_copy(tmp_path):
    """A throwaway copy of the repository, so scripts that write next to themselves cannot touch the checkout."""
    dst = tmp_path / "repo"
    shutil.copytree(REPO, dst, ignore=shutil.ignore_patterns(".git", "results", "__pycache__", ".pytest_cache", "tests"))
    (dst / "results").mkdir(exist_ok=True)
    return dst


def run_script(cwd, *args, timeout=1800):
    """Run `python <args>` in cwd with the pinned environment; return stdout (fails the test on a non-zero exit)."""
    env = dict(os.environ, **PINNED_ENV, PYTHONHASHSEED="0")
    proc = subprocess.run([sys.executable, *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        pytest.fail(f"{' '.join(args)} exited {proc.returncode}\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")
    return proc.stdout
