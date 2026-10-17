"""2D check: wafer plane, flow along y (inlet y=0), lateral x with no-flux side walls.
Time-dependent: c_t = M c + b(x,t) - A r,  theta_t = r = K c (1-theta)
Steady exposure: 0 = M E + B(x) - A (1 - exp(-K E)),  B = int b dt
M = finite-volume advection (u_y(x), any steady divergence-free profile) + isotropic diffusion."""
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, diags, bmat, identity
from scipy.sparse.linalg import spsolve
from scipy.integrate import solve_ivp

def build(Nx, Ny, W=0.75, Pe=26.7, uprof=lambda x: np.ones_like(x)):
    dx, dy = W/Nx, 1.0/Ny; D = 1.0/Pe
    xc = -W/2 + (np.arange(Nx)+0.5)*dx; yc = (np.arange(Ny)+0.5)*dy
    u = uprof(xc)                       # u_y as a function of x only: divergence-free
    n = Nx*Ny; idx = lambda i, j: j*Nx + i   # i lateral, j along flow
    M = lil_matrix((n, n))
    for j in range(Ny):
        for i in range(Nx):
            k = idx(i, j)
            # y faces
            if j > 0:   # face j-1/2: flux in = u*avg + D*(c[j-1]-c[j])/dy
                M[k, idx(i,j-1)] += (u[i]*0.5 + D/dy)/dy; M[k, k] += (u[i]*0.5 - D/dy)/dy
            # inlet face handled by source b (Danckwerts flux = u*c_in)
            if j < Ny-1:  # face j+1/2: flux out = u*avg - D*(c[j+1]-c[j])/dy
                M[k, k] -= (u[i]*0.5 + D/dy)/dy; M[k, idx(i,j+1)] -= (u[i]*0.5 - D/dy)/dy
            else:         # outlet: flux out = u*c
                M[k, k] -= u[i]/dy
            # x faces: diffusion only, walls no-flux
            if i > 0:   M[k, idx(i-1,j)] += D/dx**2; M[k, k] -= D/dx**2
            if i < Nx-1: M[k, idx(i+1,j)] += D/dx**2; M[k, k] -= D/dx**2
    b0 = np.zeros(n); b0[idx(np.arange(Nx), 0)] = u/dy   # times c_in(x,t)
    return csr_matrix(M), b0, xc, yc, idx

def time_solve(M, b0, g, A=4.0, K=3.7, tau_p=2.0, t_end=40.0):
    n = M.shape[0]; bg = b0*g
    def rhs(t, z):
        c = z[:n]; th = z[n:2*n]
        r = K*np.maximum(c,0)*(1-th)
        return np.concatenate([M@c + (bg if t < tau_p else 0) - A*r, r, np.maximum(c,0)])
    def jac(t, z):
        c = z[:n]; th = z[n:2*n]
        dr_dc = diags(K*(1-th)); dr_dth = diags(-K*c); Z = csr_matrix((n,n))
        return bmat([[M - A*dr_dc, -A*dr_dth, Z],[dr_dc, dr_dth, Z],[identity(n), Z, Z]]).tocsc()
    z = np.zeros(3*n)
    for (t0, t1) in ((0.0, tau_p), (tau_p, t_end)):
        sol = solve_ivp(rhs, (t0, t1), z, method='BDF', jac=jac, rtol=1e-8, atol=1e-11)
        z = sol.y[:, -1]
    return z[:n], z[n:2*n], z[2*n:]

def steady_solve(M, b0, g, A=4.0, K=3.7, Phi=2.0, tol=1e-12):
    B = b0*g*Phi; E = np.zeros(M.shape[0])
    for it in range(100):
        Rres = M@E + B - A*(1-np.exp(-K*E))
        J = M - A*diags(K*np.exp(-K*E))
        dE = spsolve(J.tocsc(), -Rres); E += dE
        if np.max(np.abs(dE)) < tol: break
    return E, 1-np.exp(-K*E)
