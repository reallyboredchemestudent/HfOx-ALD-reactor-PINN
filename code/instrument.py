"""Measurement model from the V1 lookup.
generate(): reported thickness for a true film, given a run's discrete angle-offset state.
marginal(): smooth mean and sd of the reported thickness, averaging over angle offsets ~ N(0, 0.02 deg),
            used in the likelihood (within-run correlation of thin-film errors is ignored)."""
import numpy as np, os
from scipy.interpolate import PchipInterpolator
_z = np.load(os.environ.get('ALD_LOOKUP', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v1_lookup.npz')))
T, A, D = _z['t_true'], _z['a_off'], _z['d_rep']
SIG_A = 0.02
w = np.exp(-0.5*(A/SIG_A)**2); w /= w.sum()
_mean = D @ w
_sd = np.sqrt(((D - _mean[:, None])**2) @ w)
_m = PchipInterpolator(T, _mean); _s = PchipInterpolator(T, _sd)
_cols = [PchipInterpolator(T, D[:, j]) for j in range(A.size)]
TMAX = float(T.max())   # top of the tabulated true-thickness range (nm)
SIG0 = 0.05   # residual site noise (nm): repeatability + transcription rounding
def marginal(t):
    inside = (t > 0) & (t < TMAX)
    tc = np.clip(t, 0, TMAX); return _m(tc), np.sqrt(_s(tc)**2 + SIG0**2), _m.derivative()(tc)*inside
def sample_state(rng): return rng.choice(A.size, p=w)
def generate(t, state, rng=None):
    t = np.clip(t, 0, TMAX); out = _cols[state](t)
    if rng is not None: out = out + rng.normal(0, SIG0, np.shape(t))
    return out
if __name__ == '__main__':
    for t in (0, 1, 2, 3, 4, 5, 6, 8, 10, 15):
        m, s, dm = marginal(np.array(t, float)); print(f"true {t:4.1f} -> mean reported {m:5.2f}, sd {s:4.2f}, slope {dm:4.2f}")

def marginal_full(t):
    """mean, total sd, d mean/dt, d sd/dt (zero outside the tabulated range)."""
    inside = (t > 0) & (t < TMAX); tc = np.clip(t, 0, TMAX)
    sv = _s(tc); sd = np.sqrt(sv**2 + SIG0**2)
    return _m(tc), sd, _m.derivative()(tc)*inside, (sv*_s.derivative()(tc)/sd)*inside
