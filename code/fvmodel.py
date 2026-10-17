"""Steady exposure model on the chamber plane (depth-averaged, both plates reactive).
Dimensional form divided by gas speed u (lengths in cm):
    d/dy (e - l de/dy) - l d2e/dx2 + (1/zb) (1 - exp(-e)) = 0
    inlet  (y = -U):  e - l de/dy = delta * g(x),  delta = lam / zb
    outlet (y = 15):  de/dy = 0;   side walls (x = +-7.5): de/dx = 0
    e = k E (dimensionless exposure), theta = 1 - exp(-e)
Groups: lam = reach (cm, mass balance), zb = plug-flow front width (cm), l = D/u dispersion length (cm),
        U = reactive length upstream of the site origin (cm), g(x) = inlet flux profile (mean 1)."""
import numpy as np
from scipy.sparse import diags, kron, identity, csr_matrix, csc_matrix
from scipy.sparse.linalg import splu
import instrument as INS

import os
KB = 8.617333e-5
NCYC = float(os.environ.get('ALD_NCYC', 100))   # ALD cycles per wafer
THK = NCYC/10                                     # nm of film per (A/cycle) of GPC
TD = np.array([100,125,150,175,200,225,250.0]); TP = np.array([55,60,65,70,75.0])
SITES = np.array([(0,4.5),(-1.5,6),(1.5,6),(-3,7.5),(0,7.5),(3,7.5),(-1.5,9),(1.5,9),(0,10.5)])
CONDS = [(td, tp) for td in TD for tp in TP]          # 35 runs, one wafer each
pv = lambda tp: 10**(9.99 - 3377/(tp + 273.15))      # Torr, supplier curve
TREF = 175 + 273.15
NAMES = ['log_lam','log_zb','Ea','log_l','U','g1','g2','G0','G1']
TRUTH_A = np.array([np.log(19.0), np.log(0.9), 0.05, np.log(1.0), 9.0, 0.30, -0.15, 1.30, -0.20])

def leg2(xi): return 1.5*xi**2 - 0.5
def inlet_profile(xi, p, gfun=None):
    return gfun(xi) if gfun is not None else 1 + p[5]*xi + p[6]*leg2(xi)

def groups(p, td, tp, o=0.0):
    T = td + 273.15
    gpc = p[7] + p[8]*(td - 175)/75
    gpc_ref = p[7]
    lam = np.exp(p[0] + o) * (pv(tp)/pv(55)) * (gpc_ref/gpc)
    zb = np.exp(p[1]) * (T/TREF)**0.5 * np.exp(p[2]/KB*(1/T - 1/TREF))
    l = np.exp(p[3]) * (T/TREF)**0.75
    return lam, zb, l, gpc

class Grid:
    def __init__(self, U, Nx=30, Ny=96, W=15.0, Lend=15.0):
        self.Nx, self.Ny = Nx, Ny
        self.dx = W/Nx; self.dy = (Lend + U)/Ny; self.U = U
        self.xc = -W/2 + (np.arange(Nx)+0.5)*self.dx
        self.yc = -U + (np.arange(Ny)+0.5)*self.dy
        self.xi = self.xc/(W/2)
        n = Nx*Ny; self.n = n
        # y-direction operator for d/dy(e - l e_y): advective part Ay, diffusive Dy (times l); index k = j*Nx + i
        Iy = identity(Ny); Ix = identity(Nx)
        # advection (central) with inlet flux handled by source; outlet flux = e_N
        ma = np.zeros((Ny, Ny))
        for j in range(Ny):
            if j > 0: ma[j, j-1] -= 0.5; ma[j, j] -= 0.5      # minus incoming face value avg
            if j < Ny-1: ma[j, j] += 0.5; ma[j, j+1] += 0.5   # outgoing face avg
            else: ma[j, j] += 1.0                              # outlet face value = cell value
        md = np.zeros((Ny, Ny))   # -(d2/dy2) with no-flux at inlet face for the diffusive part (Robin folded into source)
        for j in range(Ny):
            if j > 0: md[j, j] += 1; md[j, j-1] -= 1
            if j < Ny-1: md[j, j] += 1; md[j, j+1] -= 1
        mx = np.zeros((Nx, Nx))
        for i in range(Nx):
            if i > 0: mx[i, i] += 1; mx[i, i-1] -= 1
            if i < Nx-1: mx[i, i] += 1; mx[i, i+1] -= 1
        self.Ay = kron(csr_matrix(ma)/self.dy, Ix).tocsr()
        self.Dyy = kron(csr_matrix(md)/self.dy**2, Ix).tocsr()
        self.Dxx = kron(Iy, csr_matrix(mx)/self.dx**2).tocsr()
        self.binlet = np.zeros(n); self.binlet[:Nx] = 1.0/self.dy   # inlet face flux enters first row
        # bilinear observation weights at sites
        rows = []
        for (xs, ys) in SITES:
            fx = (xs - self.xc[0])/self.dx; fy = (ys - self.yc[0])/self.dy
            i0 = int(np.floor(fx)); j0 = int(np.floor(fy)); tx = fx - i0; ty = fy - j0
            wrow = np.zeros(n)
            for (di, dj, ww) in ((0,0,(1-tx)*(1-ty)), (1,0,tx*(1-ty)), (0,1,(1-tx)*ty), (1,1,tx*ty)):
                wrow[(j0+dj)*Nx + (i0+di)] += ww
            rows.append(wrow)
        self.P = csr_matrix(np.array(rows))
    def lin(self, l): return self.Ay + l*(self.Dyy + self.Dxx)

def solve(grid, lam, zb, l, g, e0=None, tol=1e-11):
    """Newton solve; returns e and LU of the Jacobian at the solution."""
    L = grid.lin(l); delta = lam/zb
    src = grid.binlet*delta*np.tile(g, grid.Ny)
    e = np.full(grid.n, 1.0) if e0 is None else e0.copy()
    for it in range(60):
        R = L@e + (1/zb)*(1 - np.exp(-e)) - src
        J = (L + diags((1/zb)*np.exp(-e))).tocsc()
        lu = splu(J); de = lu.solve(-R); e += de
        if np.max(np.abs(de)) < tol: break
    J = (L + diags((1/zb)*np.exp(-e))).tocsc(); lu = splu(J)
    return e, lu, L, src

def forward(p, offsets=None, gfun=None, Nx=30, Ny=96, sens=False, conds=None):
    """Predicted true film thickness (nm) at the 9 sites for each condition, optional sensitivities d t / d p.
    Sensitivities for all physics params except U come from one LU per condition (forward/tangent-linear)."""
    conds = CONDS if conds is None else conds
    off = np.zeros(len(CONDS)) if offsets is None else offsets
    grid = Grid(p[4], Nx, Ny)
    g = inlet_profile(grid.xi, p, gfun); gt = np.tile(g, grid.Ny)
    out = []; S = []; Soff = []
    for c in conds:
        ci = CONDS.index(c); td, tp = c
        lam, zb, l, gpc = groups(p, td, tp, off[ci])
        e, lu, L, src = solve(grid, lam, zb, l, g)
        es = grid.P@e; th = 1 - np.exp(-es)
        t = THK*gpc*th; out.append(t)
        if sens:
            T = td + 273.15; delta = lam/zb
            dRdp = np.zeros((grid.n, 9))
            dRdp[:, 0] = -grid.binlet*delta*gt                                   # log_lam
            dlogzb = np.array([1.0, (1/KB)*(1/T - 1/TREF)])
            dR_dlogzb = -(1/zb)*(1 - np.exp(-e)) + grid.binlet*delta*gt          # sink and delta = lam/zb
            dRdp[:, 1] = dR_dlogzb*dlogzb[0]; dRdp[:, 2] = dR_dlogzb*dlogzb[1]
            dRdp[:, 3] = l*((grid.Dyy + grid.Dxx)@e)                               # log_l
            dRdp[:, 5] = -grid.binlet*delta*np.tile(grid.xi, grid.Ny) if gfun is None else 0
            dRdp[:, 6] = -grid.binlet*delta*np.tile(leg2(grid.xi), grid.Ny) if gfun is None else 0
            # GPC enters lam through gpc_ref/gpc: d log lam / dG0 = 1/G0 - 1/gpc ; dG1 = -(s)/gpc
            s_d = (td - 175)/75
            dloglam_dG0 = 1/p[7] - 1/gpc; dloglam_dG1 = -s_d/gpc
            dRdp[:, 7] = dRdp[:, 0]*dloglam_dG0; dRdp[:, 8] = dRdp[:, 0]*dloglam_dG1
            de = -lu.solve(dRdp)                                                     # n x 9
            dth = (np.exp(-es))[:, None]*(grid.P@de)
            dt = THK*gpc*dth
            dt[:, 7] += THK*th; dt[:, 8] += THK*th*s_d                                 # direct GPC scaling
            S.append(dt); Soff.append(THK*gpc*np.exp(-es)*(grid.P@(-lu.solve(dRdp[:, 0]))))
    out = np.array(out)
    if not sens: return out
    S = np.array(S)
    # U by central difference (it changes the grid)
    h = 0.05
    pp = p.copy(); pp[4] += h; tp_ = forward(pp, offsets, gfun, Nx, Ny, conds=conds)
    pm = p.copy(); pm[4] -= h; tm_ = forward(pm, offsets, gfun, Nx, Ny, conds=conds)
    S[:, :, 4] = (tp_ - tm_)/(2*h)
    return out, S, np.array(Soff)

def adjoint_gradient(p, w_site, cond, gfun=None, Nx=30, Ny=96):
    """Gradient of phi = sum_i w_i * theta_i at one condition w.r.t. log_lam, log_zb, log_l via one adjoint solve."""
    grid = Grid(p[4], Nx, Ny); g = inlet_profile(grid.xi, p, gfun); gt = np.tile(g, grid.Ny)
    td, tp = cond; lam, zb, l, gpc = groups(p, td, tp)
    e, lu, L, src = solve(grid, lam, zb, l, g)
    es = grid.P@e
    dphi_de = grid.P.T@(w_site*np.exp(-es))
    mu = lu.solve(dphi_de, trans='T')         # adjoint: J^T mu = dphi/de
    delta = lam/zb
    dR = np.stack([-grid.binlet*delta*gt, -(1/zb)*(1-np.exp(-e)) + grid.binlet*delta*gt, l*((grid.Dyy+grid.Dxx)@e)], 1)
    return -(mu@dR)
