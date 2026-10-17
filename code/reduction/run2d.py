import numpy as np, time, warnings
warnings.filterwarnings('ignore')
from reduce2d import build, steady_solve, time_solve
from scipy.interpolate import RegularGridInterpolator
W = 0.75
gfun = lambda x: 1 + 0.4*x/(W/2)          # inlet 40% richer on one side
ufun = lambda x: 1 + 0.25*x/(W/2)         # flow 25% faster on the same side
res = {}
for (Nx, Ny) in ((24, 32), (48, 64), (96, 128)):
    t = time.time()
    M, b0, xc, yc, idx = build(Nx, Ny, W=W, uprof=ufun)
    g = np.zeros(Nx*Ny); g[idx(np.arange(Nx),0)] = gfun(xc)
    Es, ths = steady_solve(M, b0, g)
    out = dict(xc=xc, yc=yc, ths=ths.reshape(Ny, Nx))
    if Nx <= 48:
        c, th, E = time_solve(M, b0, g)
        out['th'] = th.reshape(Ny, Nx)
        print(f"grid {Nx}x{Ny}: same-grid time vs steady max|dtheta| {np.max(np.abs(th-ths)):.1e} | gas left {c.max():.0e} | {time.time()-t:.0f}s")
    res[(Nx,Ny)] = out
# convergence: compare each time solve with the finest steady solution
fine = res[(96,128)]
F = RegularGridInterpolator((fine['yc'], fine['xc']), fine['ths'], bounds_error=False, fill_value=None)
for key in ((24,32),(48,64)):
    r = res[key]; Y, X = np.meshgrid(r['yc'], r['xc'], indexing='ij')
    d = np.max(np.abs(r['th'] - F(np.stack([Y, X], -1))))
    print(f"time solve {key[0]}x{key[1]} vs finest steady: max|dtheta| {d:.1e} ({d*15:.2f} nm)")
th = res[(48,64)]['th']; xc = res[(48,64)]['xc']; yc = res[(48,64)]['yc']
j = np.argmin(np.abs(yc-0.5)); print(f"lateral asymmetry at y=0.5: theta {th[j,0]:.2f} (lean side) vs {th[j,-1]:.2f} (rich side)")
np.savez('run2d.npz', th=th, ths=res[(48,64)]['ths'], xc=xc, yc=yc)
