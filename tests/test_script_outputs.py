"""Every number printed by the scripts that need only shipped files, frozen as text (tests/golden/stdout/).

Each script runs in a throwaway copy of the repository. Text around numbers must match exactly; each number
must match to one unit in its last printed digit (golden_io.compare_stdout). Wall-clock timings are masked and
absolute paths dropped (script_specs.py). The published README numbers these outputs carry are also asserted
directly in test_published_numbers.py.
"""
import re

import pytest

from conftest import run_script
from golden_io import GOLDEN, compare_stdout, normalize
from script_specs import SPECS


def _params():
    for name, spec in SPECS.items():
        marks = [pytest.mark.slow] if spec.get("slow") else []
        yield pytest.param(name, spec, id=name, marks=marks)


def run_spec(spec, root):
    out = ""
    for cmd in spec["cmds"]:
        out = run_script(root / spec["cwd"], *cmd)
    for pat in spec.get("mask", []):
        out = "\n".join(re.sub(pat, "<t>", ln) for ln in out.splitlines())
    return normalize(out, spec.get("drop", ()))


@pytest.mark.parametrize("name,spec", list(_params()))
def test_script_stdout(name, spec, repo_copy):
    got = run_spec(spec, repo_copy)
    want = (GOLDEN / "stdout" / f"{name}.txt").read_text().splitlines()
    diffs = compare_stdout(got, want)
    assert not diffs, f"{name}: printed numbers changed\n" + "\n".join(diffs[:20])
