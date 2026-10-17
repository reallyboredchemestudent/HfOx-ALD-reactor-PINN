# Structural identifiability of the steady exposure equation along the flow, with StructuralIdentifiability.jl
# (differential algebra, exact; no data, no noise). The flow coordinate y plays the role of "time" t.
#
#   julia identifiability.jl            -> prints the results and writes si_results.txt next to this file
#
# Per-u form on the centreline:   d/dy (e - l de/dy) + (1/zb) (1 - exp(-e)) = 0,   coverage theta = 1 - exp(-e)
# Rational form with w = exp(-e) (fraction of sites still free) and v = de/dy:
#     w' = -w v,     v' = (v + a (1 - w)) / l,      a = 1/zb  (front width),  b = 1/l  (dispersion)
# Plug flow (l -> 0) gives v = -a (1 - w), i.e. w' = a w (1 - w): a logistic front.
#
# Reach lam and the reactive length U enter only through the inlet condition (e - l e' = lam/zb at y = -U), i.e. through
# the state at the inlet, and an autonomous profile is invariant to shifts in y. Models 3 and 4 make this explicit with the
# exact plug-flow profile theta = (E - 1)/(q + E - 1), E = exp(lam/zb), q = exp((y + U)/zb), q' = a q, q(0) = exp(a U):
#   model 3: one dose                      -> expected: only q(0)/(E - 1), i.e. the front position, not lam and U apart
#   model 4: a second wafer at the same deposition temperature with twice the dose (55 vs 65 C source: vapor-pressure
#            ratio 2.02 on the supplier curve), E2 = E^2  -> expected: a, E and q(0) all identifiable, so lam and U apart.
# This is the structural (continuous-output) upper bound; with 9 sites per wafer the practical answer is the Fisher
# information and profile likelihoods already computed (identifiability.py, profile_lik.py).

using StructuralIdentifiability

models = [
    ("1  plug flow, one profile",
     @ODEmodel(w'(t) = a * w(t) * (1 - w(t)),
               y1(t) = 1 - w(t))),
    ("2  with axial dispersion, one profile",
     @ODEmodel(w'(t) = -w(t) * v(t),
               v'(t) = b * (v(t) + a * (1 - w(t))),
               y1(t) = 1 - w(t))),
    ("3  plug flow, explicit reach E = exp(lam/zb) and offset q = exp((y+U)/zb), one dose",
     @ODEmodel(q'(t) = a * q(t),
               y1(t) = (E - 1) / (q(t) + E - 1))),
    ("4  plug flow, two doses at one deposition temperature (dose ratio 2: E2 = E^2)",
     @ODEmodel(q'(t) = a * q(t),
               y1(t) = (E - 1) / (q(t) + E - 1),
               y2(t) = (E^2 - 1) / (q(t) + E^2 - 1))),
]

report = String[]
say(parts...) = (line = string(parts...); println(line); push!(report, line))
for (name, ode) in models
    say("\n=== model ", name, " ===")
    t0 = time()
    res = assess_identifiability(ode)
    say("assess_identifiability (parameters and initial states):")
    for (k, v) in res
        say("   ", rpad(string(k), 10), " => ", v)
    end
    fns = find_identifiable_functions(ode, with_states = true)
    say("identifiable functions (with states): ", join(string.(fns), ", "))
    say("time ", round(time() - t0, digits = 1), " s")
end
write(joinpath(@__DIR__, "si_results.txt"), join(report, "\n") * "\n")
println("\nwrote ", joinpath(@__DIR__, "si_results.txt"))
