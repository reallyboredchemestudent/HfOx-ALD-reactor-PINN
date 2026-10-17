"""Record the golden values the tests compare against. Run this ONLY on purpose, never to make a failing test pass
without first understanding why the number moved:

    python tests/record_golden.py              everything except JAX (about 6 minutes on one core)
    python tests/record_golden.py --only fit   one group: shipped, instrument, forward, fit, pinn, stdout, jax
    python tests/record_golden.py --jax        the JAX PINN golden (needs JAX; records on the machine that runs it)

The goldens committed in tests/golden/ were recorded from commit 053e817 (before any review change), with the
versions written into each file's _provenance block. A change that moves a golden is a change to a result: say so
in the commit message and update the README numbers it feeds.
"""
import argparse
import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import conftest  # noqa: E402  (pins the import-time environment and puts code/ on sys.path)
import golden_io as G  # noqa: E402
import numpy as np  # noqa: E402

from script_specs import SPECS  # noqa: E402

REPO, CODE = conftest.REPO, conftest.CODE

SHIPPED = ["code/synthA_rep0.npz", "code/synthB_rep0.npz", "code/howo_fv_A.npy", "code/howo_fv_B.npy",
           "code/v1_lookup.npz", "code/check_install_expected.json", "data/lee2026_hfox_ald_dataset.csv"]
T_GRID = np.linspace(0.0, 17.0, 69)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rec_shipped():
    G.save_json("shipped_files.json", {p: sha256(REPO / p) for p in SHIPPED})


def rec_instrument():
    import instrument as INS
    m, s, dm = INS.marginal(T_GRID)
    _, _, _, ds = INS.marginal_full(T_GRID)
    cols = np.array([INS.generate(T_GRID, k) for k in range(INS.A.size)])
    G.save_json("instrument.json", dict(t=T_GRID, mean=m, sd=s, dmean=dm, dsd=ds, generate=cols, w=INS.w, TMAX=INS.TMAX))


def rec_forward():
    import fvmodel as FV
    T, S, So = FV.forward(FV.TRUTH_A, sens=True)
    off = np.linspace(-0.1, 0.1, 35)
    G.save_json("fv_forward.json", dict(T=T, S=np.asarray(S), So=np.asarray(So), T_offsets=FV.forward(FV.TRUTH_A, off), offsets=off))


def rec_fit():
    import fit as F
    Y = np.load(CODE / "synthA_rep0.npz")["Y"]
    p0 = F.PRIOR_MU.copy(); p0[0] = np.log(22); p0[1] = np.log(1.2)
    r = F.fit(Y, p0)
    G.save_json("fv_fit_synthA.json", dict(p=r["p"], o=r["o"], b=r["b"], chi2_data=r["chi2_data"], cost=r["cost"],
                                           nfev=r["nfev"], sd=np.sqrt(np.diag(r["cov"])[:9])))


def pinn_problem():
    import pinn as PN
    Y = np.load(CODE / "synthA_rep0.npz")["Y"]
    rng = np.random.default_rng(0)
    lay = PN.Layout(H=100, HG=6)
    prob = PN.Problem(Y, np.ones(35, bool), lay, rng, n55=200, nsat=40, nin=24)
    th = PN.init_params(lay, rng)
    return PN, prob, lay, th


def rec_pinn():
    PN, prob, lay, th = pinn_problem()
    prob.w = dict(pde=20.0, inl=1.0, data=1.0)
    v, g = prob.loss(th)
    _, _, comp = prob.loss(th, parts=True)
    th2 = PN.train(prob, th.copy(), stages=2, iters=40, adapt=True, verbose=False)
    v2, _ = prob.loss(th2)
    G.save_json("pinn_numpy.json", dict(loss=v, grad=g, comp=[comp["pde"], comp["inl"], comp["data_chi2"]],
                                        train_phys=th2[lay.sl["phys"]], train_loss=v2, w_after=[prob.w["inl"], prob.w["data"]]))


def run_spec(name, spec, root):
    out = ""
    for cmd in spec["cmds"]:
        out = conftest.run_script(root / spec["cwd"], *cmd)
    return out


def rec_stdout(names=None):
    (G.GOLDEN / "stdout").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "repo"
        shutil.copytree(REPO, root, ignore=shutil.ignore_patterns(".git", "results", "__pycache__", "tests"))
        (root / "results").mkdir()
        for name, spec in SPECS.items():
            if names and name not in names:
                continue
            print(f"  stdout: {name}", flush=True)
            text = run_spec(name, spec, root)
            for pat in spec.get("mask", []):
                import re
                text = "\n".join(re.sub(pat, "<t>", ln) for ln in text.splitlines())
            (G.GOLDEN / "stdout" / f"{name}.txt").write_text("\n".join(G.normalize(text, spec.get("drop", ()))) + "\n")


def rec_jax():
    """JAX PINN: self-check must pass, then a short deterministic run is frozen."""
    with tempfile.TemporaryDirectory() as td:
        out = conftest.run_script(CODE, "pinn_jax.py", "selfcheck")
        if "SELF-CHECK PASSED" not in out:
            sys.exit("pinn_jax.py selfcheck did not pass; not recording a JAX golden\n" + out)
        conftest.run_script(CODE, "pinn_jax.py", "run", "--data", "synthA_rep0.npz", "--tag", "GOLD", "--seeds", "0",
                            "--holds=-1", "--stages", "2", "--iters", "40", "--pre-stages", "1", "--pre-iters", "40",
                            "--out", td)
        z = np.load(Path(td) / "pinn_GOLD_s0_h-1.npz")
        G.save_json("pinn_jax.json", dict(phys=z["phys"], comp=z["comp"], pred=z["pred"]))


GROUPS = dict(shipped=rec_shipped, instrument=rec_instrument, forward=rec_forward, fit=rec_fit, pinn=rec_pinn,
              stdout=rec_stdout, jax=rec_jax)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=list(GROUPS))
    ap.add_argument("--jax", action="store_true")
    a = ap.parse_args()
    todo = ["jax"] if a.jax else (a.only or [k for k in GROUPS if k != "jax"])
    for k in todo:
        print(f"recording {k}", flush=True)
        GROUPS[k]()
