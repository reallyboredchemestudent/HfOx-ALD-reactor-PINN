"""Turn copies of aldFoam's example case into the four comparison cases (A-D of aldfoam_params.py).

    python make_aldfoam_cases.py ~/ald_cases          (expects ~/ald_cases/A ... D, each a copy of examples/aldFoam)

Per case it writes: the parallel-plate mesh (24 cm long, gap 1.0 or 0.5 cm, 480 x 20 cells, same patch names as the
example: inlet, outlet, substrate = bottom plate, Wall = top plate, frontAndBack empty), a fully developed Poiseuille
flow field U (aldFoam transports the precursor in a given flow; the example ships a precomputed one), p and growth reset,
the inlet pulse table in 0/N1 (area = pulse length, so dose = pressure x pulse), D1 and vth, beta0 and site area on both
plates, the precursor pressure and temperature, and an endTime of pulse + 1.5 s with one write at the end.
Assumption to check on case A: 'pressure' in constant/processData is the precursor partial pressure (Pa) when N1 = 1.
If case A's front is far from ~16 cm, that unit is different; the case-vs-case test (B, C vs A) does not depend on it.
"""
import os, re, sys
import numpy as np
import aldfoam_params as AP

HDR = """/*--------------------------------*- C++ -*----------------------------------*\\
  written by make_aldfoam_cases.py for the HfOx steady-exposure check
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    location    "{loc}";
    object      {obj};
}}
"""
NX, NY, LY_CM = 480, 20, 24.0

def write(path, text):
    with open(path, 'w') as f: f.write(text)

def block_mesh(h_cm):
    return HDR.format(cls='dictionary', loc='system', obj='blockMeshDict') + f"""
convertToMeters 0.01;

vertices
(
    (0 0 0) ({LY_CM:g} 0 0) ({LY_CM:g} {h_cm:g} 0) (0 {h_cm:g} 0)
    (0 0 0.1) ({LY_CM:g} 0 0.1) ({LY_CM:g} {h_cm:g} 0.1) (0 {h_cm:g} 0.1)
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({NX} {NY} 1) simpleGrading (1 1 1)
);

edges
(
);

boundary
(
    frontAndBack {{ type empty; faces ((0 3 2 1) (4 5 6 7)); }}
    inlet        {{ type wall;  faces ((0 4 7 3)); }}
    outlet       {{ type wall;  faces ((2 6 5 1)); }}
    substrate    {{ type wall;  faces ((1 5 4 0)); }}
    Wall         {{ type wall;  faces ((3 7 6 2)); }}
);

mergePatchPairs
(
);
"""

def poiseuille(h_m, u_mean):
    yc = (np.arange(NY) + 0.5)/NY                     # cell centres across the gap, as fractions of h
    return 6*u_mean*yc*(1 - yc)

def u_field(h_m, u_mean):
    uy = poiseuille(h_m, u_mean)
    cells = "\n".join(f"({uy[j]:.8g} 0 0)" for j in range(NY) for i in range(NX))     # blockMesh order: x fastest
    inlet = "\n".join(f"({uy[j]:.8g} 0 0)" for j in range(NY))
    return HDR.format(cls='volVectorField', loc='0', obj='U') + f"""
dimensions      [0 1 -1 0 0 0 0];

internalField   nonuniform List<vector>
{NX*NY}
(
{cells}
)
;

boundaryField
{{
    inlet        {{ type fixedValue; value nonuniform List<vector> {NY} ( {inlet.replace(chr(10), ' ')} ); }}
    outlet       {{ type zeroGradient; }}
    substrate    {{ type noSlip; }}
    Wall         {{ type noSlip; }}
    frontAndBack {{ type empty; }}
}}
"""

def scalar_field(obj, dims, inlet='calculated', outlet='calculated'):
    def bc(t): return f"{{ type {t}; value uniform 0; }}" if t in ('calculated', 'fixedValue') else f"{{ type {t}; }}"
    return HDR.format(cls='volScalarField', loc='0', obj=obj) + f"""
dimensions      {dims};

internalField   uniform 0;

boundaryField
{{
    inlet        {bc(inlet)}
    outlet       {bc(outlet)}
    substrate    {bc('calculated' if obj != 'p' else 'zeroGradient')}
    Wall         {bc('calculated' if obj != 'p' else 'zeroGradient')}
    frontAndBack {{ type empty; }}
}}
"""

def n1_field(tp, tr):
    # trapezoid of area tp: ramp up over tr, hold, ramp down over tr  ->  dose = pressure x tp, for any tr
    return HDR.format(cls='volScalarField', loc='0', obj='N1') + f"""
dimensions      [0 0 0 0 0 0 0];

internalField   uniform 0;

boundaryField
{{
    inlet
    {{
        type            uniformFixedValue;
        uniformValue    table ( (0 0) ({tr:g} 1) ({tp:g} 1) ({tp + tr:g} 0) );
    }}
    Wall
    {{
        type            firstorder;
        betaField       beta;
        diffCoeff       D1;
        vth             vth;
        value           uniform 0.0;
    }}
    substrate
    {{
        type            firstorder;
        betaField       beta;
        diffCoeff       D1;
        vth             vth;
        value           uniform 0.0;
    }}
    outlet
    {{
        type            zeroGradient;
    }}
    frontAndBack {{ type empty; }}
}}
"""

def transport(c):
    return HDR.format(cls='dictionary', loc='constant', obj='transportProperties') + f"""
D1              D1 [ 0 2 -1 0 0 0 0 ] {c['D_m2_s']:.6g};
vth             vth [ 0 1 -1 0 0 0 0 ] {c['v_thermal_m_s']:.6g};
"""

def surface(c):
    one = f"""        sitearea    {c['site_area_m2']:.6g};
        beta0       {c['beta0']:.6g};
        beta1       0.0;
        betarec     0.0;"""
    return HDR.format(cls='dictionary', loc='constant', obj='surfaceProcesses') + f"""
numPatches      2;

surfacePatches{{

    Wall{{
{one}
    }}

    substrate{{
{one}
    }}

}}
"""

def process(c):
    return HDR.format(cls='dictionary', loc='constant', obj='processData') + f"""
pressure        {c['inlet_partial_pressure_Pa']:.6g};
temperature     {c['T_K']:.6g};
"""

def control(path, c):
    s = open(path).read()
    end = c['end_time_s']
    umax = 1.5*c['mean_speed_m_s']; dx = LY_CM/100/NX
    dt = min(1e-4, 0.4*dx/umax)                   # implicit, but keep the Courant number near 0.4 for accuracy
    for key, val in (('endTime', f"{end:g}"), ('deltaT', f"{dt:.3g}"), ('writeControl', 'runTime'), ('writeInterval', f"{end:g}"),
                     ('writeFormat', 'ascii'), ('startTime', '0')):
        s, n = re.subn(rf'^(\s*{key}\s+)[^;]*;', rf'\g<1>{val};', s, flags=re.M)
        if n == 0: s = s.replace('// ****', f'{key} {val};\n\n// ****', 1)
    write(path, s)

def main(root):
    cases = {c['case']: c for c in AP.cases}
    for name, c in cases.items():
        d = os.path.join(os.path.expanduser(root), name)
        if not os.path.isdir(d): print(f"skip {name}: {d} not found (copy examples/aldFoam there first)"); continue
        h = c['gap_m']; u = c['mean_speed_m_s']
        write(os.path.join(d, 'system', 'blockMeshDict'), block_mesh(h*100))
        write(os.path.join(d, '0', 'U'), u_field(h, u))
        write(os.path.join(d, '0', 'p'), scalar_field('p', '[0 2 -2 0 0 0 0]', inlet='zeroGradient', outlet='fixedValue'))
        write(os.path.join(d, '0', 'growth'), scalar_field('growth', '[0 2 -2 0 0 0 0]'))
        for f in ('Exp1', 'beta', 'cov'):
            dims = '[0 0 0 0 0 0 0]'
            write(os.path.join(d, '0', f), scalar_field(f, dims))
        write(os.path.join(d, '0', 'N1'), n1_field(c['pulse_s'], c['ramp_s']))
        write(os.path.join(d, 'constant', 'transportProperties'), transport(c))
        write(os.path.join(d, 'constant', 'surfaceProcesses'), surface(c))
        write(os.path.join(d, 'constant', 'processData'), process(c))
        control(os.path.join(d, 'system', 'controlDict'), c)
        for old in os.listdir(d):                    # stale results and mesh from the example
            try: float(old); is_time = old != '0'
            except ValueError: is_time = False
            if is_time: __import__('shutil').rmtree(os.path.join(d, old))
        pm = os.path.join(d, 'constant', 'polyMesh')
        if os.path.isdir(pm): __import__('shutil').rmtree(pm)
        print(f"case {name}: gap {h*100:g} cm, mean speed {u:g} m/s, pulse {c['pulse_s']:g} s (ramp {c['ramp_s']:g} s), "
              f"precursor {c['inlet_partial_pressure_Pa']:.4g} Pa, beta0 {c['beta0']:.4g}, end {c['end_time_s']:g} s -> {d}")

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '~/ald_cases')
