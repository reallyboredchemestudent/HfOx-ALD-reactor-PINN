"""Gap-resolved check (stand-in for aldFoam): flow direction y x gap z, parabolic (Poiseuille) flow,
precursor consumed only at the two plates through a kinetic wall flux k_s c_wall (1 - theta).
Lee-like numbers at 175 C (cm, s): gas speed u, diffusivity D, sticking chosen so the plug-flow
front width is ~0.9 cm and the front sits mid-chamber.
Compares: (1) transient vs steady gap-resolved exposure (exact if flow is steady),
          (2) transient with a pressure spike during the pulse, same molecules delivered,
          (3) best-fit depth-averaged steady model (the PINN's equation) vs the gap-resolved truth."""
import numpy as np, sys
from scipy.sparse import diags, kron, identity, csr_matrix, bmat
from scipy.sparse.linalg import splu, spsolve
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from fvmodel import Grid, solve as solve_da

def setup(h, ubar, D=214.0, Ly=24.0, Ny=160, Nz=12, vbar=16400.0, zb=0.9, lam=16.0, tp=0.2):
    beta = 2*h*ubar/(vbar*zb); ks = beta*vbar/4          # wall reaction velocity (cm/s), q = 1
    Phi = 2*lam/(ubar*h)                                 # exposure at inlet so that reach = lam
    dy, dz = Ly/Ny, h/Nz
    zc = (np.arange(Nz)+0.5)*dz; uz = 6*ubar*(zc/h)*(1-zc/h)
    yc = (np.arange(Ny)+0.5)*dy
    # operators on c[k*Ny + j] (k: z index, j: y index)
    ma = np.zeros((Ny, Ny)); md = np.zeros((Ny, Ny))
    for j in range(Ny):
        if j > 0: ma[j,j-1] -= .5; ma[j,j] -= .5; md[j,j] += 1; md[j,j-1] -= 1
        if j < Ny-1: ma[j,j] += .5; ma[j,j+1] += .5; md[j,j] += 1; md[j,j+1] -= 1
        else: ma[j,j] += 1
    mz = np.zeros((Nz, Nz))
    for k in range(Nz):
        if k > 0: mz[k,k] += 1; mz[k,k-1] -= 1
        if k < Nz-1: mz[k,k] += 1; mz[k,k+1] -= 1
    Adv = kron(diags(uz), csr_matrix(ma)/dy).tocsr()
    Dyy = kron(identity(Nz), csr_matrix(md)/dy**2).tocsr()
    Dzz = kron(csr_matrix(mz)/dz**2, identity(Ny)).tocsr()
    inlet = np.zeros(Nz*Ny); inlet[np.arange(Nz)*Ny] = uz/dy        # times c_in(t)
    wall_rows = np.concatenate([np.arange(Ny), (Nz-1)*Ny + np.arange(Ny)])   # bottom then top plate cells
    return dict(h=h, ubar=ubar, D=D, ks=ks, Phi=Phi, dz=dz, Ny=Ny, Nz=Nz, yc=yc, uz=uz, Adv=Adv, Dyy=Dyy, Dzz=Dzz,
                inlet=inlet, wall=wall_rows, tp=tp, beta=beta)

def transient(s, spike_u=0.0, spike_p=0.0, t_end=2.0):
    n = s['Nz']*s['Ny']; nw = s['wall'].size; ks, dz = s['ks'], s['dz']
    c0 = s['Phi']/s['tp']
    def ops(t):
        on = t < s['tp']
        fu = (1+spike_u) if on else 1.0; fD = 1/(1+spike_p) if on else 1.0
        L = fu*s['Adv'] + s['D']*fD*(s['Dyy'] + s['Dzz'])   # Dyy, Dzz hold -d2 (positive definite)
        src = s['inlet']*fu*(c0/(1+spike_u) if on else 0.0)        # same molecules delivered
        return L, src
    W = csr_matrix((np.ones(nw), (s['wall'], np.arange(nw))), shape=(n, nw))
    def rhs(t, y):
        c = y[:n]; th = y[n:]
        L, src = ops(t)
        flux = ks*np.maximum(c[s['wall']], 0)*(1-th)            # per area of plate
        dc = -L@c + src - (W@flux)/dz
        return np.concatenate([dc, flux])                          # q = 1
    def jac(t, y):
        c = y[:n]; th = y[n:]; L, _ = ops(t)
        dfdc = ks*(1-th); dfdth = -ks*np.maximum(c[s['wall']], 0)
        Jcc = -L - (W@diags(dfdc)@W.T)/dz
        Jct = -(W@diags(dfdth))/dz
        Jtc = diags(dfdc)@W.T
        return bmat([[Jcc, Jct], [Jtc, diags(dfdth)]]).tocsc()
    y = np.zeros(n+nw)
    for (a, b) in ((0, s['tp']), (s['tp'], t_end)):
        sol = solve_ivp(rhs, (a, b), y, method='BDF', jac=jac, rtol=1e-7, atol=1e-10)
        y = sol.y[:, -1]
    return y[n:].reshape(2, s['Ny']), y[:n].max()

def steady_exposure(s):
    n = s['Nz']*s['Ny']; nw = s['wall'].size; ks, dz = s['ks'], s['dz']
    L = s['Adv'] + s['D']*(s['Dyy'] + s['Dzz']); src = s['inlet']*s['Phi']
    W = csr_matrix((np.ones(nw), (s['wall'], np.arange(nw))), shape=(n, nw))
    def resid(E):
        Ew = np.maximum(E[s['wall']], 0)
        return L@E - src + (W@(1 - np.exp(-ks*Ew)))/dz
    E = np.full(n, s['Phi']*0.5)
    R = resid(E)
    for it in range(200):     # damped Newton; wall sink per area = q*(1 - exp(-ks*E_wall)), q = 1
        Ew = np.maximum(E[s['wall']], 0)
        J = (L + (W@diags(ks*np.exp(-ks*Ew))@W.T)/dz).tocsc()
        dE = spsolve(J, -R); a = 1.0; n0 = np.linalg.norm(R)
        while True:
            En = E + a*dE; Rn = resid(En)
            if np.linalg.norm(Rn) < n0 or a < 1e-4: break
            a *= 0.5
        E, R = En, Rn
        if np.max(np.abs(a*dE))*ks < 1e-10: break
    return (1 - np.exp(-ks*E[s['wall']])).reshape(2, s['Ny'])

def depth_averaged_fit(s, th_wall):
    """Fit the depth-averaged steady model (reach, width, dispersion length) to the gap-resolved coverage."""
    y = s['yc']; target = th_wall.mean(0)
    def model(q):
        lam, zb, l = np.exp(q)
        g = Grid(0.0, 1, 480, W=1.0, Lend=24.0)
        e, *_ = solve_da(g, lam, zb, l, np.ones(1))
        return np.interp(y, g.yc, 1 - np.exp(-e))
    r = least_squares(lambda q: model(q) - target, np.log([16.0, 0.9, s['D']/s['ubar']]))
    return np.exp(r.x), np.max(np.abs(r.fun)), model(r.x)

if __name__ == '__main__':
    for h, ubar in ((1.0, 179.0), (0.5, 358.0)):
        s = setup(h, ubar)
        ths = steady_exposure(s)
        th0, cl = transient(s)
        print(f"\ngap {h} cm, mean speed {ubar} cm/s: sticking {s['beta']:.3f}, wall Damkohler ks*h/D = {s['ks']*h/s['D']:.2f}, gap Peclet {ubar*h/s['D']:.2f}")
        print(f"  (1) transient vs steady gap-resolved exposure: max|dtheta| {np.max(np.abs(th0-ths)):.1e} ({np.max(np.abs(th0-ths))*15:.4f} nm); gas left {cl:.0e}")
        for su, sp in ((0.1, 0.1), (0.3, 0.2), (0.5, 0.4)):
            th1, _ = transient(s, su, sp)
            d = np.max(np.abs(th1-ths)); print(f"  (2) spike: flow +{int(su*100)}%, pressure +{int(sp*100)}% during pulse: max|dtheta| {d:.3f} ({d*15:.2f} nm)")
        (lam, zb, l), err, _ = depth_averaged_fit(s, ths)
        print(f"  (3) depth-averaged steady fit: reach {lam:.2f} cm, width {zb:.3f} cm (nominal 0.9), dispersion {l:.3f} cm (D/u {s['D']/ubar:.3f}); worst misfit {err:.4f} ({err*15:.3f} nm)")
        print(f"      top vs bottom plate max difference {np.max(np.abs(ths[0]-ths[1])):.1e}")
