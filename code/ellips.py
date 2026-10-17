"""E1: ellipsometry forward model for Si / SiOx / HfO2 stacks and what a
single-layer Cauchy fit (the authors' stated model) returns.
Transfer-matrix (Abeles) for isotropic layers; Psi/Delta at 55/65/75 deg.
Si optical constants: approximate anchor table (Green 2008-like), interpolated.
"""
import numpy as np
from scipy.optimize import least_squares

WL = np.linspace(450, 1000, 56)            # nm
ANG = np.deg2rad([55.0, 65.0, 75.0])
_si = np.array([[450,4.67,0.130],[500,4.30,0.073],[550,4.08,0.040],[600,3.94,0.025],
                [633,3.88,0.018],[700,3.78,0.010],[800,3.68,0.005],[900,3.62,0.002],[1000,3.57,0.0005]])
def n_si(wl, scale=1.0):
    return scale*np.interp(wl,_si[:,0],_si[:,1]) - 1j*np.interp(wl,_si[:,0],_si[:,2])
def n_sio2(wl):
    l=wl/1000; l2=l*l
    return np.sqrt(1+0.6961663*l2/(l2-0.0684043**2)+0.4079426*l2/(l2-0.1162414**2)+0.8974794*l2/(l2-9.896161**2))
def cauchy(wl,A,B):  # B in um^2
    return A + B/(wl/1000)**2
def rpr_s(layers, nsub, wl, th0):
    """layers: list of (n_array, d_nm) top->bottom; ambient n=1. returns rp, rs."""
    n0=1.0; s0=np.sin(th0)
    out=[]
    for pol in ('p','s'):
        def adm(n):
            c=np.sqrt(1-(n0*s0/n)**2+0j)
            return (n/c if pol=='p' else n*c), c
        Y0,_=adm(n0+0j)
        M11=np.ones_like(wl,dtype=complex);M12=np.zeros_like(M11);M21=np.zeros_like(M11);M22=np.ones_like(M11)
        for n,d in layers:
            Y,c=adm(n+0j)
            delta=2*np.pi*n*d*c/wl
            a=np.cos(delta); b=1j*np.sin(delta)/Y; cc=1j*Y*np.sin(delta)
            M11,M12,M21,M22=M11*a+M12*cc, M11*b+M12*a, M21*a+M22*cc, M21*b+M22*a
        Ys,_=adm(nsub)
        B=M11+M12*Ys; C=M21+M22*Ys
        r=(Y0*B-C)/(Y0*B+C)
        out.append(r)
    return out
def psidelta(layers, nsub_scale=1.0):
    P=[];D=[]
    for th in ANG:
        rp,rs=rpr_s(layers,n_si(WL,nsub_scale),WL,th)
        rho=rp/rs
        P.append(np.degrees(np.arctan(np.abs(rho)))); D.append(np.degrees(np.angle(rho)))
    return np.concatenate(P),np.concatenate(D)
def truth(t_ox,t_hf,A=2.07,B=0.012,sub=1.0):
    L=[]
    if t_hf>0: L.append((cauchy(WL,A,B),t_hf))
    if t_ox>0: L.append((n_sio2(WL),t_ox))
    return psidelta(L,sub)
def fit_single(P,D,fixB=None,sP=0.02,sD=0.05,starts=None):
    def res(x):
        d,A=x[0],x[1]; B=fixB if fixB is not None else x[2]
        p,q=psidelta([(cauchy(WL,A,B),d)])
        dq=(q-D+180)%360-180
        return np.concatenate([(p-P)/sP, dq/sD])
    best=None
    starts = starts or [(d0,A0) for d0 in (1,3,6,10,15) for A0 in (1.6,2.0,2.4)]
    for d0,A0 in starts:
        x0=[d0,A0] if fixB is not None else [d0,A0,0.012]
        lb=[0.05,1.2] if fixB is not None else [0.05,1.2,-0.05]
        ub=[40,3.5] if fixB is not None else [40,3.5,0.2]
        r=least_squares(res,x0,bounds=(lb,ub))
        if best is None or r.cost<best.cost: best=r
    d,A=best.x[0],best.x[1]; B=fixB if fixB is not None else best.x[2]
    return d, cauchy(633,A,B), best
