"""Reading, writing and comparing golden values.

Two kinds of golden:
  * JSON files of arrays/scalars, compared with an explicit tolerance per quantity (stated in each test);
  * the full stdout of a script, compared token by token: the text around the numbers must match exactly, and each
    number must agree to within one unit in its last printed digit (so a value sitting on a rounding boundary
    may flip, but nothing larger can change unnoticed). Two exceptions, both seen on CI across BLAS/LAPACK builds:
      - NOISE_FLOOR: two values that are both below 1e-7 in magnitude agree. The scripts print such values only as
        round-off or solver-tolerance residuals (coverage differences, residual gas, plate symmetry), whose digits
        depend on the floating-point library; the claims made about them are of the form "below 1e-7". Real
        discretisation errors in these outputs are 1e-6 and above and are still compared digit by digit.
      - SIGN_FREE lines: a singular vector / eigenvector is defined only up to sign, so a printed direction agrees
        if every component matches either as printed or with all signs flipped.
"""
import json
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

GOLDEN = Path(__file__).resolve().parent / "golden"
REPO = GOLDEN.parents[1]

_NUM = re.compile(r"(?<![\w.])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?(?![\w.])|\bnan\b|\binf\b")
_NP_REPR = re.compile(r"np\.float(?:64|32)\(([^)]*)\)")   # NumPy >= 2 prints np.float64(0.2); NumPy 1 prints 0.2


def provenance():
    import numpy, scipy
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = "unknown"
    return dict(recorded_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), commit=commit,
                python=platform.python_version(), numpy=numpy.__version__, scipy=scipy.__version__,
                machine=platform.machine(), system=platform.system())


def save_json(name, values):
    out = {"_provenance": provenance()}
    for k, v in values.items():
        out[k] = np.asarray(v).tolist() if isinstance(v, (np.ndarray, np.generic, list, tuple)) else v
    (GOLDEN / name).write_text(json.dumps(out, indent=1) + "\n")


def load_json(name):
    return json.loads((GOLDEN / name).read_text())


def normalize(text, drop=()):
    """Strip NumPy-version repr noise and any lines matching the regexes in `drop` (timings, absolute paths)."""
    text = _NP_REPR.sub(r"\1", text)
    lines = [ln.rstrip() for ln in text.splitlines()]
    return [ln for ln in lines if ln and not any(re.search(p, ln) for p in drop)]


def _ulp(tok):
    if tok in ("nan", "inf"):
        return 0.0
    mant, _, exp = tok.lower().partition("e")
    if "." not in mant and not exp:
        return 0.0                      # integers (counts, grid sizes, temperatures) must match exactly
    dec = len(mant.split(".")[1]) if "." in mant else 0
    return 10.0 ** (int(exp or 0) - dec)


def _value(tok):
    return float(tok)


NOISE_FLOOR = 1e-7
SIGN_FREE = re.compile(r"(weakest|sloppiest) direction:")


def _numbers_agree(tg, tw, sign=1.0):
    vg, vw = sign * _value(tg), _value(tw)
    if (np.isnan(vw) and np.isnan(vg)) or vg == vw:
        return True
    if abs(vg) < NOISE_FLOOR and abs(vw) < NOISE_FLOOR:
        return True
    return abs(vg - vw) <= 1.01 * max(_ulp(tg), _ulp(tw))


def compare_stdout(got_lines, want_lines):
    """Return a list of human-readable differences (empty when the outputs agree)."""
    diffs = []
    if len(got_lines) != len(want_lines):
        diffs.append(f"line count {len(got_lines)} != golden {len(want_lines)}")
    for i, (g, w) in enumerate(zip(got_lines, want_lines)):
        if _NUM.sub("#", g) != _NUM.sub("#", w):
            diffs.append(f"line {i + 1} text differs:\n   got    {g}\n   golden {w}")
            continue
        pairs = list(zip(_NUM.findall(g), _NUM.findall(w)))
        if SIGN_FREE.search(w) and all(_numbers_agree(tg, tw, -1.0) for tg, tw in pairs):
            continue
        for tg, tw in pairs:
            if not _numbers_agree(tg, tw):
                diffs.append(f"line {i + 1}: {tg} != golden {tw}\n   {w}")
    return diffs
