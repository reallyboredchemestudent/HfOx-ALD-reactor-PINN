"""The README's published numbers, checked against what the code produces from a clean clone.

The stdout goldens freeze what the code prints today; this file ties those printed values to the numbers in the
README's results table, at the precision the README states them. Two README numbers are NOT reproduced by the
command the README gives; they are marked xfail(strict=True), so the test turns red the day they are reconciled
(fix the README or the command, then remove the mark).
"""
import json
import re

import pytest

from conftest import REPO, run_script
from golden_io import GOLDEN

README = (REPO / "README.md").read_text()


def golden(name):
    return (GOLDEN / "stdout" / f"{name}.txt").read_text()


def test_readme_collapse():
    """README: reach x GPC constant to 2.6 %; collapse chi2 1.2 / 3."""
    out = golden("collapse")
    assert re.search(r"CV 2\.6 %", out)
    chi2, dof = re.search(r"chi2 ([\d.]+) on (\d+) dof", out).groups()
    assert abs(float(chi2) - 1.2) <= 0.051 and int(dof) == 3       # 1.15 printed; README rounds half up
    assert "2.6%" in README and "χ² 1.2 / 3" in README


def test_readme_load():
    """README: 315/315 sites."""
    assert "filled 315 of 315 wafer-sites" in golden("data_real") and "315/315" in README


def test_readme_gapresolved():
    """README: reduction in a gap-resolved channel 1e-8; pressure spike <= 0.34 nm."""
    out = golden("gapresolved")
    d = [float(x) for x in re.findall(r"transient vs steady gap-resolved exposure: max\|dtheta\| ([\d.e+-]+)", out)]
    assert d and max(d) < 1e-7
    spikes = [float(x) for x in re.findall(r"during pulse: max\|dtheta\| [\d.]+ \(([\d.]+) nm\)", out)]
    assert max(spikes) == pytest.approx(0.34, abs=0.005)


@pytest.mark.xfail(strict=True, reason="README 'coverage FV 0.98': the README command (run_all.sh synthpinn) uses holds "
                   "0,15,30 and prints 1.00; all seven 55 C holds give 61/63 = 0.97. compare.py's FV intervals share one "
                   "RNG with the PINN branch, so the FV number also depends on which PINN files exist (REVIEW.md, finding 4)")
def test_readme_fv_coverage_synthetic():
    rows = re.findall(r"^FV .*inside 90% ([\d.]+)", golden("compare_synthA_fv"), flags=re.M)
    cov = sum(float(c) * 9 for c in rows) / (9 * len(rows))
    assert round(cov, 2) == 0.98


@pytest.mark.slow
def test_real_fit_ideal_budget_golden(repo_copy):
    """FV fit of the 100-200 C real wafers under the ideal error budget, in the run_all.sh order (load, collapse, fit),
    so the fit starts from the collapse result. About 8-12 min on one or two cores.
    Tolerance: abs 0.5 on chi2 (2e-5 relative) and 1e-3 on parameters: a 40-evaluation multistart fit."""
    gold = json.loads((GOLDEN / "real_fit_td100-200.json").read_text())
    code = repo_copy / "code"
    run_script(code, "data_real.py", "../data/lee2026_hfox_ald_dataset.csv")
    run_script(code, "run_real.py", "collapse", "--regime", "100-200")
    run_script(code, "run_real.py", "fit", "--td", "100-200", "--jobs", "2", timeout=3600)
    got = json.loads((repo_copy / "results" / "fit_td100-200.json").read_text())
    assert got["ndata"] == gold["ndata"] == 225
    assert got["chi2_data"] == pytest.approx(gold["chi2_data"], abs=0.5)
    for k, v in gold["params"].items():
        assert got["params"][k] == pytest.approx(v, abs=1e-3), k


@pytest.mark.xfail(strict=True, reason="README 'chi2 21,124 / 225' for the ideal budget; from a clean clone the fit gives "
                   "21,168.8 when collapse ran first (run_all.sh order) and 21,045.8 when it did not; the fit silently "
                   "started from results/collapse_summary.json if present (REVIEW.md, finding 1)")
def test_readme_real_fit_ideal_chi2():
    gold = json.loads((GOLDEN / "real_fit_td100-200.json").read_text())
    assert round(gold["chi2_data"]) == 21124


@pytest.mark.xfail(strict=True, reason="README '0.018 nm' (reduction exact on the chamber plane): the README command "
                   "(run1d_base.py && run2d.py) prints 1D time-vs-steady max|dtheta| <= 3.7e-5 and 2D max|dtheta| 1.4e-3 "
                   "(x 15 nm = 0.020-0.022 nm) and 3.6e-4; none is 0.018 nm (REVIEW.md, finding 4)")
def test_readme_reduction_exact():
    out = golden("run1d_base") + golden("run2d")
    nm = [float(x) for x in re.findall(r"\(([\d.]+) nm\)", out)]
    nm += [15 * float(x) for x in re.findall(r"time vs steady max\|dtheta\| ([\d.e+-]+)", out)]
    assert any(abs(v - 0.018) < 0.0005 for v in nm)


def test_fit_refuses_a_hidden_start(repo_copy):
    """run_real.py fit used results/collapse_summary.json as its start if the file happened to exist, and the prior
    otherwise; the two end 123 apart in chi2. It must now refuse to guess."""
    import subprocess, sys
    code = repo_copy / "code"
    run_script(code, "data_real.py", "../data/lee2026_hfox_ald_dataset.csv")
    proc = subprocess.run([sys.executable, "run_real.py", "fit", "--td", "100-200", "--jobs", "1"], cwd=code,
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode != 0 and "collapse_summary.json" in proc.stderr and "--start prior" in proc.stderr
