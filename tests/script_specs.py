"""Scripts whose printed numbers are frozen as stdout goldens (tests/golden/stdout/<name>.txt).

Each entry: working directory (relative to the repo root), the commands to run there in order (only the last
command's stdout is compared), regexes for lines to drop (absolute paths) and for tokens to mask (wall-clock
timings), and whether the script is slow (> ~30 s on one core).
Every script here needs only files shipped in the repository.
"""
TIMING = r"\b\d+ ?s\b(?=\s*$)"      # a trailing "12s" / "12 s" timing at the end of a line

SPECS = {
    "instrument": dict(cwd="code", cmds=[["instrument.py"]]),
    "identifiability": dict(cwd="code", cmds=[["identifiability.py"]]),
    "rank1d": dict(cwd="code", cmds=[["rank1d.py"]], slow=True),
    "run1d_base": dict(cwd="code/reduction", cmds=[["run1d_base.py"]], mask=[TIMING]),
    "run1d_pool": dict(cwd="code/reduction", cmds=[["run1d_pool.py"]]),
    "run1d_pool2": dict(cwd="code/reduction", cmds=[["run1d_pool2.py"]]),
    "run2d": dict(cwd="code/reduction", cmds=[["run2d.py"]], mask=[TIMING]),
    "gapresolved": dict(cwd="code", cmds=[["gapresolved.py"]]),
    "aldfoam_params": dict(cwd="code", cmds=[["aldfoam_params.py"]], drop=[r"^wrote "]),
    "data_real": dict(cwd="code", cmds=[["data_real.py", "../data/lee2026_hfox_ald_dataset.csv"]], drop=[r"^wrote "]),
    "collapse": dict(cwd="code", cmds=[["data_real.py", "../data/lee2026_hfox_ald_dataset.csv"],
                                       ["run_real.py", "collapse", "--regime", "100-200"]],
                     drop=[r"^wrote "], slow=True),
    # FV hold-one-wafer-out intervals on synthetic set A from the shipped reference fits (README: "coverage FV 0.98")
    "compare_synthA_fv": dict(cwd="code", cmds=[["compare.py", "A", "0", "0,5,10,15,20,25,30", "NONE"]], slow=True),
}
