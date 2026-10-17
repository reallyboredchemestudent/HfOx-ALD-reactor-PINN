import numpy as np, warnings
warnings.filterwarnings('ignore')
from reduce1d import pool_setup, time_solve_pool
P, x, ths, sq, kw = pool_setup(N=400); TP = P['tau_p']
def rep(label, th, c, p):
    d = np.max(np.abs(th-ths))
    print(f"{label:58s} max|dtheta| {d:.1e} ({d*15:.2f} nm) | gas left {c.max():.0e} | pool left {p.max():.0e}")
print("--- Physisorbed pool (holdup) ---")
x_, c, th, p, E = time_solve_pool(cin=sq, Ap=0.0, **kw);                rep("no pool (reference)", th, c, p)
for kd in (1.0, 0.3):
    x_, c, th, p, E = time_solve_pool(cin=sq, Ap=4.0, kd=kd, **kw);     rep(f"pool, chemisorption from gas, kd={kd}, purged by t=20", th, c, p)
x_, c, th, p, E = time_solve_pool(cin=sq, Ap=4.0, kd=0.3, t_end=4.0, **kw); rep("same pool, purge cut to t=4 (pool not emptied)", th, c, p)
x_, c, th, p, E = time_solve_pool(cin=sq, Ap=4.0, kd=1.0, mediated=True, Km=3.7, **kw); rep("chemisorption THROUGH the pool (precursor-mediated)", th, c, p)
print("--- Flow faster during the pulse, same molecules delivered ---")
for a in (0.2, 0.5):
    uf = (lambda t, a=a: 1+a if t < TP else 1.0)
    x_, c, th, p, E = time_solve_pool(cin=sq, Ap=0.0, u_pulse=uf, **kw); rep(f"flow {int(a*100)}% faster during pulse", th, c, p)
