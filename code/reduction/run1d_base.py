import numpy as np, time
from reduce1d import BASE, plugflow_closed_form, square, steady_solve, time_solve
P = BASE; Phi = P['tau_p']
print("Base case: Pe=%.1f A=%.1f K=%.1f dose=%.1f (front near x=0.5)" % (P['Pe'],P['A'],P['K'],Phi))
for N in (200, 400, 800):
    t=time.time()
    cin, br = square(P['tau_p'])
    x, c, th, E = time_solve(N=N, Pe=P['Pe'], A=P['A'], K=P['K'], cin=cin, breaks=br, t_end=20.0)
    Es, ths = steady_solve(x, Pe=P['Pe'], A=P['A'], K=P['K'], Phi=Phi)
    kin_id = np.max(np.abs(th - (1-np.exp(-P['K']*E))))
    print(f"N={N:4d}: residual gas max {c.max():.1e} | kinetics identity max|dtheta| {kin_id:.1e} | "
          f"time vs steady max|dtheta| {np.max(np.abs(th-ths)):.1e} | max|dE|/max E {np.max(np.abs(E-Es))/E.max():.1e} | {time.time()-t:.0f}s")
np.savez('base_N800.npz', x=x, th=th, ths=ths, E=E, Es=Es)
pf = plugflow_closed_form(x, P['A'], P['K'], Phi)
print(f"Plug-flow closed form vs Pe={P['Pe']} steady: max|dtheta| {np.max(np.abs(pf-ths)):.3f}")
x50 = lambda t: x[np.argmin(np.abs(t-0.5))]
print(f"front (theta=0.5) position: steady {x50(ths):.3f}, plug-flow {x50(pf):.3f}")
# 10-90 width
def width(t):
    i90 = np.argmin(np.abs(t-0.9)); i10 = np.argmin(np.abs(t-0.1)); return x[i10]-x[i90]
print(f"front 90-10 width: with diffusion {width(ths):.3f}, plug-flow {width(pf):.3f} (in units of L=20 cm)")
