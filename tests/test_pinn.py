"""The NumPy inverse PINN (pinn.py), on synthetic set A, in the study configuration (PDE_W 20, PHYS_SCALE 1e-2,
N55 200, NSAT 40; see pinn_jax.py's docstring).

pinn.py's docstring says its analytic gradients are "checked against finite differences in test_pinn.py"; that file
was not in the repository. test_gradient_matches_finite_differences is that check.

Tolerances:
  finite differences  |analytic - fd| <= 1e-5 x max(|analytic|, |fd|) + 2e-6, central differences with step
                      1e-6 x scale, at the end of the short training (loss ~2e3). The absolute floor is the round-off of
                      the difference quotient (loss x eps / step ~ 5e-7); an error in a hand-derived term is O(1).
                      Not at the initial weights: there the PDE term is ~7e23 and no difference quotient resolves it.
  loss / gradient     rtol 1e-9: a deterministic sum over fixed collocation points.
  short training      abs 1e-4 on the 7 physics parameters after 2 stages x 40 L-BFGS-B iterations. Iterative, so
                      round-off can grow; the published PINN numbers come from far longer ensembles and are checked
                      only to two significant figures, so 1e-4 still catches any change that could reach them.
"""
import numpy as np
import pytest

from conftest import CODE
from golden_io import load_json


@pytest.fixture(scope="module")
def problem():
    import pinn as PN
    Y = np.load(CODE / "synthA_rep0.npz")["Y"]
    rng = np.random.default_rng(0)
    lay = PN.Layout(H=100, HG=6)
    prob = PN.Problem(Y, np.ones(35, bool), lay, rng, n55=200, nsat=40, nin=24)
    th = PN.init_params(lay, rng)
    prob.w = dict(pde=20.0, inl=1.0, data=1.0)
    return PN, prob, lay, th


@pytest.fixture(scope="module")
def trained(problem):
    PN, prob, lay, th = problem
    prob.w = dict(pde=20.0, inl=1.0, data=1.0)
    th2 = PN.train(prob, th.copy(), stages=2, iters=40, adapt=True, verbose=False)
    w_after = [prob.w["inl"], prob.w["data"]]
    prob.w = dict(pde=20.0, inl=1.0, data=1.0)
    return th2, w_after


def test_gradient_matches_finite_differences(problem, trained):
    PN, prob, lay, _ = problem
    th = trained[0]
    prob.w = dict(pde=20.0, inl=1.0, data=1.0)
    _, g = prob.loss(th)
    rng = np.random.default_rng(42)
    idx = list(range(lay.sl["phys"].start, lay.sl["phys"].stop))                      # every physics parameter
    for k in ("a", "W", "b", "a0", "gc", "gv", "gd", "o", "bo"):                       # and a sample of each block
        s = lay.sl[k]; idx += list(rng.choice(np.arange(s.start, s.stop), size=min(3, s.stop - s.start), replace=False))
    bad = []
    for i in idx:
        h = 1e-6 * max(1.0, abs(th[i]))
        tp, tm = th.copy(), th.copy(); tp[i] += h; tm[i] -= h
        fd = (prob.loss(tp, want_grad=False) - prob.loss(tm, want_grad=False)) / (2 * h)
        if abs(fd - g[i]) > 1e-5 * max(abs(fd), abs(g[i])) + 2e-6:
            bad.append((i, g[i], fd))
    assert not bad, f"analytic vs finite-difference gradient (index, analytic, fd): {bad}"


def test_loss_and_gradient_golden(problem):
    PN, prob, lay, th = problem
    gold = load_json("pinn_numpy.json")
    prob.w = dict(pde=20.0, inl=1.0, data=1.0)
    v, g = prob.loss(th)
    _, _, comp = prob.loss(th, parts=True)
    assert v == pytest.approx(gold["loss"], rel=1e-9)
    np.testing.assert_allclose(g, gold["grad"], rtol=1e-9, atol=1e-9 * np.max(np.abs(gold["grad"])))
    np.testing.assert_allclose([comp["pde"], comp["inl"], comp["data_chi2"]], gold["comp"], rtol=1e-9)


def test_short_training_golden(problem, trained):
    PN, prob, lay, _ = problem
    gold = load_json("pinn_numpy.json")
    th2, w_after = trained
    np.testing.assert_allclose(th2[lay.sl["phys"]], gold["train_phys"], rtol=0, atol=1e-4)
    np.testing.assert_allclose(w_after, gold["w_after"], rtol=1e-4)
    prob.w = dict(pde=20.0, inl=w_after[0], data=w_after[1])         # the loss as training left it
    assert prob.loss(th2, want_grad=False) == pytest.approx(gold["train_loss"], rel=1e-6)
    prob.w = dict(pde=20.0, inl=1.0, data=1.0)
