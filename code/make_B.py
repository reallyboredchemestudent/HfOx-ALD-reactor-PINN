"""Truth B: same physics as truth A, but the inlet profile is a lopsided jet outside the FV solver's quadratic family."""
import numpy as np
from fvmodel import TRUTH_A
from fit import make_data
def gB(xi):
    raw = 1 + 0.8/(1 + np.exp(-(xi - 0.35)/0.12))
    xs = np.linspace(-1, 1, 2001); return raw/np.mean(1 + 0.8/(1 + np.exp(-(xs - 0.35)/0.12)))
if __name__ == '__main__':
    rng = np.random.default_rng(0)
    Y, T, off = make_data(TRUTH_A, rng, gfun=gB)
    import os   # code/synthB_rep0.npz is a shipped input: write the regenerated set beside the results
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'synthB_rep0.npz'); os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, Y=Y, T=T, off=off); print(f"wrote {os.path.normpath(out)} (compare with code/synthB_rep0.npz)")
    print("truth B 175C/55C sites:", np.round(T[15], 2))
