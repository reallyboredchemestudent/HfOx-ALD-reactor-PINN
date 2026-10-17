import numpy as np, warnings
warnings.filterwarnings('ignore')
from reduce1d import BASE, gamma_pulse, square, steady_solve, time_solve
P = BASE; Phi = P['tau_p']; N = 400; GPC_NM = 15.0   # ~15 nm saturated film over 100 cycles
x = (np.arange(N)+0.5)/N
_, ths = steady_solve(x, Pe=P['Pe'], A=P['A'], K=P['K'], Phi=Phi)
def err(th, ref=ths):
    d = np.max(np.abs(th-ref)); return f"max|dtheta| {d:.1e}  ({d*GPC_NM:.2f} nm on a 15 nm film)"
kw = dict(N=N, Pe=P['Pe'], A=P['A'], K=P['K'])
rows = []
def run(label, **extra):
    cin, br = extra.pop('pulse', square(P['tau_p']))
    ref = extra.pop('ref', ths)
    x_, c, th, E = time_solve(cin=cin, breaks=br, **kw, **extra)
    line = f"{label:48s} {err(th, ref)} | gas left {c.max():.1e}"
    print(line); rows.append(line)
    return th
print("--- Should still hold ---")
run("square pulse (reference)")
run("valve-shaped pulse, same total dose", pulse=gamma_pulse(Phi, 0.5), t_end=30.0)
cin, br = square(4.0)
run("half concentration for twice as long", pulse=((lambda t: 0.5 if t<4.0 else 0.0),[4.0]))
_, ths_sq = steady_solve(x, Pe=P['Pe'], A=P['A'], K=P['K'], Phi=Phi, kin='sq')
run("coverage kinetics (1-theta)^2 [steady uses own theta(E)]", kin='sq', ref=ths_sq)
print("--- Built to break it ---")
for Kd in (0.01, 0.1, 1.0):
    run(f"reversible adsorption, desorption Kd={Kd}", Kd=Kd)
run("second order in gas concentration (m=2)", m=2)
for a in (0.2, 0.5):
    run(f"flow {int(a*100)}% faster during the pulse", u_pulse=(lambda t, a=a: 1+a if t<P['tau_p'] else 1.0))
for tend in (2.5, 3.0, 4.0):
    run(f"purge stopped at t={tend} (pulse ends at 2.0)", t_end=tend)
open('break_results.txt','w').write("\n".join(rows))
