import numpy as np, warnings
warnings.filterwarnings('ignore')
from reduce1d import pool_setup, time_solve_pool
P, x, ths, sq, kw = pool_setup(N=400)
print("Pool with chemisorption from the gas: error vs purge length (1 time unit = one residence time)")
for kd in (1.0, 0.3):
    for tend in (20, 60, 200):
        x_, c, th, p, E = time_solve_pool(cin=sq, Ap=4.0, kd=kd, t_end=float(tend), **kw)
        d = np.max(np.abs(th-ths))
        print(f"kd={kd}: purge to t={tend:3d}  max|dtheta| {d:.1e} ({d*15:.3f} nm) | pool left {p.max():.0e}")
x_, c, th, p, E = time_solve_pool(cin=sq, Ap=4.0, kd=1.0, mediated=True, Km=3.7, t_end=200.0, **kw)
d = np.max(np.abs(th-ths)); print(f"mediated, purge to t=200: max|dtheta| {d:.1e} ({d*15:.2f} nm) | pool left {p.max():.0e}")
