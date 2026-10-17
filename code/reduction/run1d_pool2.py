import numpy as np, warnings
warnings.filterwarnings('ignore')
from reduce1d import *
P = BASE; N = 400
x = (np.arange(N)+0.5)/N
_, ths = steady_solve(x, Pe=P['Pe'], A=P['A'], K=P['K'], Phi=2.0)
sq = lambda t: 1.0 if t < 2.0 else 0.0
kw = dict(N=N, Pe=P['Pe'], A=P['A'], K=P['K'], breaks=[2.0])
print("Pool with chemisorption from the gas: error vs purge length (1 time unit = one residence time)")
for kd in (1.0, 0.3):
    for tend in (20, 60, 200):
        x_, c, th, p, E = time_solve_pool(cin=sq, Ap=4.0, kd=kd, t_end=float(tend), **kw)
        d = np.max(np.abs(th-ths))
        print(f"kd={kd}: purge to t={tend:3d}  max|dtheta| {d:.1e} ({d*15:.3f} nm) | pool left {p.max():.0e}")
x_, c, th, p, E = time_solve_pool(cin=sq, Ap=4.0, kd=1.0, mediated=True, Km=3.7, t_end=200.0, **kw)
d = np.max(np.abs(th-ths)); print(f"mediated, purge to t=200: max|dtheta| {d:.1e} ({d*15:.2f} nm) | pool left {p.max():.0e}")
