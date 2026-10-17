"""Numbers for the aldFoam cases that reproduce the gap-resolved check (gapresolved.py) in an independent solver.

    python aldfoam_params.py            -> prints SI inputs for the cases and writes aldfoam_cases.json

Physics shared with gapresolved.setup: parallel plates of gap h, Poiseuille flow of mean speed u, precursor diffusivity
D, both plates reactive with first-order Langmuir kinetics, dTheta/dt = s0 beta0 J (1 - Theta), J = v_th n / 4
(aldFoam's ideal self-limited model, Yanguas-Gil & Elam, JVST A 39, 062404). The site area s0 only rescales the
inlet pressure; the reach lam fixes the delivered molecules: lam = u h Phi / 2 with Phi = s0 * integral(n dt).

Cases (all with the same molecules delivered, so the exposure reduction predicts identical final coverage):
  A   0.20 s pulse at p0          (the reference, as in gapresolved.py)
  B   0.05 s pulse at 4 p0        (short and strong)
  C   0.40 s pulse at p0 / 2      (long and weak)
  D   gap 0.5 cm at twice the speed (same u h, same sticking): the depth-averaged equation predicts the same front
"""
import numpy as np, json, os

KB = 1.380649e-23; NA = 6.02214076e23
T = 175 + 273.15                      # K
M = 354.8e-3                          # kg/mol, Hf[N(CH3)2]4
m = M/NA
vth = np.sqrt(8*KB*T/(np.pi*m))       # m/s, mean thermal speed
# site area from the film itself: one Hf per site per cycle at GPC 1.30 A/cycle, HfO2 9.68 g/cm3, 210.49 g/mol
rho, Mf, gpc = 9.68e3, 210.49e-3, 1.30e-10
s0 = 1/(rho/Mf*NA*gpc)                # m^2 per site
D = 214e-4                            # m^2/s (gapresolved default 214 cm^2/s)
zb = 0.9e-2                           # m, plug-flow front width (gapresolved default)
lam = 16e-2                           # m, reach (gapresolved default)
Ly = 0.24                             # m, channel length

def case(name, h, u, tp, pscale):
    beta0 = 2*h*u/(vth*zb)                    # sticking that gives front width zb (gapresolved: beta = 2 h u / (v zb))
    Phi = 2*lam/(u*h)                         # s0 * integral n dt (s/m)
    n0 = Phi/(s0*0.20)*pscale                 # molecules / m^3 during the pulse (case A: 0.20 s)
    p0 = n0*KB*T
    return dict(case=name, gap_m=h, mean_speed_m_s=u, inlet_speed_if_uniform_m_s=u, pulse_s=tp, purge_s=1.5, end_time_s=tp + 1.5,
                ramp_s=min(0.01, tp/10), D_m2_s=D, T_K=T, molar_mass_kg_mol=M, v_thermal_m_s=vth, site_area_m2=s0, beta0=beta0,
                inlet_number_density_m3=n0, inlet_partial_pressure_Pa=p0, inlet_partial_pressure_mTorr=p0/0.133322,
                dose_Pa_s=p0*tp, channel_length_m=Ly, cells=dict(along=480, across=20), reach_m=lam, front_width_m=zb,
                residence_time_s=Ly/u)

cases = [case('A', 0.010, 1.79, 0.20, 1.0), case('B', 0.010, 1.79, 0.05, 4.0), case('C', 0.010, 1.79, 0.40, 0.5),
         case('D', 0.005, 3.58, 0.20, 1.0)]
if __name__ == '__main__':
    print(f"TDMA-Hf at {T:.0f} K: thermal speed {vth:.1f} m/s, site area {s0*1e18:.3f} nm^2, D {D:.4f} m^2/s")
    for c in cases:
        print(f"case {c['case']}: gap {c['gap_m']*100:.1f} cm, mean speed {c['mean_speed_m_s']:.2f} m/s, pulse {c['pulse_s']:.2f} s "
              f"(ramp {c['ramp_s']:.3f} s), purge {c['purge_s']} s; beta0 {c['beta0']:.4g}; inlet precursor {c['inlet_partial_pressure_Pa']:.3f} Pa "
              f"({c['inlet_partial_pressure_mTorr']:.2f} mTorr), dose {c['dose_Pa_s']:.4f} Pa s")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'aldfoam', 'aldfoam_cases.json')
    os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(cases, open(out, 'w'), indent=1)
    print(f"wrote {os.path.abspath(out)}")
