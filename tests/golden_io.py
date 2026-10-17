"""Reading, writing and comparing golden values.

Two kinds of golden:
  * JSON files of arrays/scalars, compared with an explicit tolerance per quantity (stated in each test);
  * the full stdout of a script, compared token by token: the text around the numbers must match exactly, and each
    number must agree to within one unit in its last printed digit (so a value sitting on a rounding boundary
    may flip, but nothing larger can change unnoticed).
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

_NUM = re.compile(r"(?<![\w.])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?(?![\w.])|nan|inf")
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


def compare_stdout(got_lines, want_lines):
    """Return a list of human-readable differences (empty when the outputs agree)."""
    diffs = []
    if len(got_lines) != len(want_lines):
        diffs.append(f"line count {len(got_lines)} != golden {len(want_lines)}")
    for i, (g, w) in enumerate(zip(got_lines, want_lines)):
        if _NUM.sub("#", g) != _NUM.sub("#", w):
            diffs.append(f"line {i + 1} text differs:\n   got    {g}\n   golden {w}")
            continue
        for tg, tw in zip(_NUM.findall(g), _NUM.findall(w)):
            vg, vw = _value(tg), _value(tw)
            if np.isnan(vw) and np.isnan(vg):
                continue
            if abs(vg - vw) > 1.01 * max(_ulp(tg), _ulp(tw)):
                diffs.append(f"line {i + 1}: {tg} != golden {tw}\n   {w}")
    return diffs
