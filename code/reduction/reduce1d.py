"""Check the exposure reduction for pulsed cross-flow ALD in 1D.

Time-dependent model (dimensionless: x by reactor length L, t by L/u, c by inlet conc c0):
  c_t + u(t) c_x - (1/Pe) c_xx = -A * r            (A = 2q/(h c0): site capacity / inlet conc)
  theta_t = r = K c^m f(theta) - Kd theta           (K = k c0 L/u)
  inlet (Danckwerts): u c - (1/Pe) c_x = u c_in(t);  outlet: c_x = 0
Reduction (valid if m=1, Kd=0, u constant, purge complete):
  E_x - (1/Pe) E_xx + A * theta(E) = 0,  theta(E) = G^{-1}(K E),  G(th) = int_0^th ds/f(s)
  E(0) - (1/Pe) E_x(0) = Phi = int c_in dt;  E_x(1) = 0
"""
import numpy as np
from scipy.integrate import solve_ivp, solve_bvp
from scipy.sparse import diags, bmat, identity, csr_matrix, hstack

BASE = dict(Pe=26.7, A=4.0, K=3.7, tau_p=2.0)   # Lee-like: L=20 cm, u=200 cm/s, D=150 cm2/s, front ~10 cm

def f_lin(th): return 1-th
def f_sq(th):  return (1-th)**2
TH_OF_E = {'lin': lambda KE: 1-np.exp(-KE), 'sq': lambda KE: 1-1/(1+KE)}
F_OF_TH = {'lin': f_lin, 'sq': f_sq}

def square(tau_p): return (lambda t: 1.0 if t < tau_p else 0.0), [tau_p]
def gamma_pulse(Phi, t1):   # valve-like shape with the same integral
    return (lambda t: Phi*t*np.exp(-t/t1)/t1**2), []

def time_solve(N=400, Pe=26.7, A=4.0, K=3.7, cin=None, breaks=(), t_end=20.0,
               m=1, Kd=0.0, kin='lin', u_pulse=None):
    dx = 1.0/N
    f = F_OF_TH[kin]
    uf = (lambda t: 1.0) if u_pulse is None else u_pulse
    def rhs(t, y):
        c = y[:N]; th = y[N:]
        u = uf(t)
        F = np.empty(N+1)
        F[0] = u*cin(t)
        F[1:N] = u*0.5*(c[:-1]+c[1:]) - (c[1:]-c[:-1])/(Pe*dx)
        F[N] = u*c[-1]
        r = K*np.maximum(c,0)**m*f(th) - Kd*th
        dc = -(F[1:]-F[:-1])/dx - A*r
        return np.concatenate([dc, r])
    # sparsity: tridiagonal in c, diagonal coupling
    T = diags([1.,1.,1.],[-1,0,1],shape=(N,N)); I = identity(N)
    S = bmat([[T, I],[I, I]]).tocsr()
    y = np.zeros(2*N); E = np.zeros(N); t0 = 0.0
    segs = sorted(set([*breaks, t_end]))
    for t1 in segs:
        if t1 <= t0: continue
        # integrate exposure alongside by augmenting state
        def rhs_aug(t, z): return np.concatenate([rhs(t, z[:2*N]), np.maximum(z[:N],0)])
        Saug = bmat([[S, csr_matrix((2*N,N))],[hstack([identity(N), csr_matrix((N,N))]), csr_matrix((N,N))]]).tocsr()
        sol = solve_ivp(rhs_aug, (t0, t1), np.concatenate([y, E]), method='BDF',
                        jac_sparsity=Saug, rtol=1e-9, atol=1e-12)
        z = sol.y[:, -1]; y = z[:2*N]; E = z[2*N:]; t0 = t1
    x = (np.arange(N)+0.5)*dx
    return x, y[:N], y[N:], E

def steady_solve(x_out, Pe=26.7, A=4.0, K=3.7, Phi=2.0, kin='lin'):
    thE = TH_OF_E[kin]
    def ode(x, Y):  # Y = [E, E']
        E, Ep = Y
        return np.vstack([Ep, Pe*(Ep + A*thE(K*E))])
    def bc(Ya, Yb): return np.array([Ya[0]-Ya[1]/Pe - Phi, Yb[1]])
    xm = np.linspace(0,1,2001)
    E0 = np.maximum(Phi - A*xm, 0.01)   # plug-flow-ish guess
    sol = solve_bvp(ode, bc, xm, np.vstack([E0, np.gradient(E0, xm)]), tol=1e-8, max_nodes=200000)
    assert sol.success, sol.message
    E = sol.sol(x_out)[0]
    return E, thE(K*E)

def plugflow_closed_form(x, A=4.0, K=3.7, Phi=2.0):
    d = K*Phi
    return (np.exp(d)-1)/(np.exp(A*K*x)+np.exp(d)-1)

def time_solve_pool(N=400, Pe=26.7, A=4.0, K=3.7, cin=None, breaks=(), t_end=20.0,
                    Ap=4.0, ka=3.7, kd=1.0, mediated=False, Km=3.7, u_pulse=None, cin_scale=None):
    """Two surface pools: chemisorbed theta (irreversible) and a physisorbed pool p (reversible).
    mediated=False: chemisorption from the gas (theta_t = K c (1-theta)).
    mediated=True : chemisorption from the physisorbed pool (theta_t = Km p (1-theta))."""
    dx = 1.0/N
    uf = (lambda t: 1.0) if u_pulse is None else u_pulse
    def rhs(t, y):
        c = y[:N]; th = y[N:2*N]; p = y[2*N:3*N]
        u = uf(t)
        F = np.empty(N+1)
        F[0] = cin(t)  # inlet molar flux (already includes velocity)
        F[1:N] = u*0.5*(c[:-1]+c[1:]) - (c[1:]-c[:-1])/(Pe*dx)
        F[N] = u*c[-1]
        cp = np.maximum(c,0)
        rp = ka*cp*(1-p) - kd*p if Ap > 0 else np.zeros(N)
        rth = (Km*p*(1-th)) if mediated else (K*cp*(1-th))
        dc = -(F[1:]-F[:-1])/dx - A*rth - Ap*rp
        if mediated: rp = rp - (A/max(Ap,1e-12))*rth   # chemisorbing molecules leave the pool
        return np.concatenate([dc, rth, rp, cp])
    T = diags([1.,1.,1.],[-1,0,1],shape=(N,N)); I = identity(N); Z = csr_matrix((N,N))
    S = bmat([[T, I, I, Z],[I, I, I, Z],[I, I, I, Z],[I, Z, Z, Z]]).tocsr()
    z = np.zeros(4*N); t0 = 0.0
    for t1 in sorted(set([*breaks, t_end])):
        if t1 <= t0: continue
        sol = solve_ivp(rhs, (t0, t1), z, method='BDF', jac_sparsity=S, rtol=1e-9, atol=1e-12)
        z = sol.y[:, -1]; t0 = t1
    x = (np.arange(N)+0.5)*dx
    return x, z[:N], z[N:2*N], z[2*N:3*N], z[3*N:]


def pool_setup(N=400):
    """Shared set-up of run1d_pool.py and run1d_pool2.py: the base case, its steady reference coverage, a square pulse
    of the base pulse length, and the time-solver keywords. One definition, so the two scripts cannot drift apart."""
    P = BASE; tp = P['tau_p']
    x = (np.arange(N)+0.5)/N
    _, ths = steady_solve(x, Pe=P['Pe'], A=P['A'], K=P['K'], Phi=tp)
    sq = lambda t: 1.0 if t < tp else 0.0
    kw = dict(N=N, Pe=P['Pe'], A=P['A'], K=P['K'], breaks=[tp])
    return P, x, ths, sq, kw
