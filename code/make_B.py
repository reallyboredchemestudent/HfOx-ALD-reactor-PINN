"""Truth B: same physics as truth A, but the inlet profile is a lopsided jet outside the FV solver's quadratic family."""
import numpy as np
from fvmodel import *
from fit import make_data
def gB(xi):
    raw = 1 + 0.8/(1 + np.exp(-(xi - 0.35)/0.12))
    xs = np.linspace(-1, 1, 2001); return raw/np.mean(1 + 0.8/(1 + np.exp(-(xs - 0.35)/0.12)))
if __name__ == '__main__':
    rng = np.random.default_rng(0)
    Y, T, off = make_data(TRUTH_A, rng, gfun=gB)
    np.savez('synthB_rep0.npz', Y=Y, T=T, off=off)
    print("truth B 175C/55C sites:", np.round(T[15], 2))
