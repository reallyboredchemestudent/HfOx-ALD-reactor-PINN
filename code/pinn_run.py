"""PINN driver. Usage: python pinn_run.py <dataset.npz> <tag> <seed> <holdout index or -1> [pretrained.npy]

Settings come from the environment and default to N55=260 NSAT=50 STAGES=10 PDE_W=1 PHYS_SCALE=1e-3. These are NOT the
study configuration documented in pinn_jax.py (N55=200 NSAT=40 STAGES=8 PDE_W=20 PHYS_SCALE=1e-2 FAIR_START=1); set
those explicitly to reproduce it. The effective settings are printed first so every log records them."""
import numpy as np, sys, time
from pinn import Layout, PDE_W, PHYS_SCALE, Problem, init_params, inlet_g, train
data, tag, seed, hold = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
pre = sys.argv[5] if len(sys.argv) > 5 else None
Y = np.load(data)['Y']; mask = np.ones(35, bool)
if hold >= 0: mask[hold] = False
rng = np.random.default_rng(seed)
lay = Layout(H=100, HG=6)
prob = Problem(Y, mask, lay, rng, n55=int(__import__("os").environ.get("N55", 260)), nsat=int(__import__("os").environ.get("NSAT", 50)), nin=24)
_env = __import__('os').environ
print(f"settings: N55={_env.get('N55', 260)} NSAT={_env.get('NSAT', 50)} STAGES={_env.get('STAGES', 10)} PDE_W={PDE_W} "
      f"PHYS_SCALE={PHYS_SCALE} FAIR_START={bool(_env.get('FAIR_START'))}", flush=True)
t0 = time.time()
if pre is None:
    th = init_params(lay, rng)
    if __import__('os').environ.get('FAIR_START'): th[lay.sl['phys'].start] = np.log(22.0); th[lay.sl['phys'].start+1] = np.log(1.2)
    print("physics-only pretraining at prior-mean parameters (no data)", flush=True)
    phys_fixed = th[lay.sl['phys']].copy()
    bounds = [(None, None)]*lay.n
    for i in range(lay.sl['phys'].start, lay.sl['phys'].stop): bounds[i] = (th[i], th[i])
    for i in range(lay.sl['o'].start, lay.sl['bo'].stop): bounds[i] = (0.0, 0.0)
    th = train(prob, th, stages=6, iters=600, adapt=False, physics_only=True, bounds=bounds)
    np.save(f'pinn_pre_{seed}.npy', th)
    print(f"pretrain done {time.time()-t0:.0f}s", flush=True)
else:
    th = np.load(pre)
prob.w = dict(pde=PDE_W, inl=1.0, data=1.0)
# physical bounds as in the FV fitter
b = [(None, None)]*lay.n
lo = [np.log(3), np.log(0.05), -1.0, np.log(0.02), 0.0, 0.5, -1.0]; hi = [np.log(80), np.log(10), 1.0, np.log(20), 30.0, 2.5, 1.0]
for k, i in enumerate(range(lay.sl['phys'].start, lay.sl['phys'].stop)): b[i] = (lo[k], hi[k])
th = train(prob, th, stages=int(__import__("os").environ.get("STAGES", 10)), iters=400, adapt=True, bounds=b)
_, _, comp = prob.loss(th, parts=True)
pred = prob.predict(th, list(range(35)))
np.savez(f'pinn_{tag}_s{seed}_h{hold}.npz', th=th, pred=pred, phys=th[lay.sl['phys']], comp=np.array([comp['pde'], comp['inl'], comp['data_chi2']]),
         g=inlet_g(th, lay, np.linspace(-1, 1, 41))[0], time=time.time()-t0)
print(f"done {time.time()-t0:.0f}s phys {np.round(th[lay.sl['phys']],3)}", flush=True)
