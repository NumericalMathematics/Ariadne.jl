# This elixir demonstrates how an implicit-explicit (IMEX) time integration scheme can be applied to the stiff and non-stiff parts of a right hand side, respectively. 
# We define separate solvers, boundary conditions, and source terms, and create a `SemidiscretizationHyperbolicSplit`, which will return a `SplitODEProblem` compatible with `OrdinaryDiffEqBDF`, cf. https://docs.sciml.ai/OrdinaryDiffEq/stable/implicit/SDIRK/#IMEX-DIRK .
# Note: This is currently more of a proof of concept and not particularly useful in practice, as fully explicit methods are still faster at the moment.

using Trixi
using Theseus
using CairoMakie

# Notes:
# You must disable both Polyester and LoopVectorization for Enzyme to be able to differentiate Trixi.jl.
#
# LocalPreferences.jl
# ```toml
# [Trixi]
# loop_vectorization = false
# backend = "static"
# ```

@assert Trixi._PREFERENCE_THREADING !== :polyester
@assert !Trixi._PREFERENCE_LOOPVECTORIZATION

function initial_condition_warm_bubble(x, t, equations::CompressibleEulerEquations2D)
    g = 9.81
    c_p = 1004.0
    c_v = 717.0

    # center of perturbation
    center_x = 10000.0
    center_z = 2000.0
    # radius of perturbation
    radius = 2000.0
    # distance of current x to center of perturbation
    r = sqrt((x[1] - center_x)^2 + (x[2] - center_z)^2)

    # perturbation in potential temperature
    potential_temperature_ref = 300.0
    potential_temperature_perturbation = 0.0
    if r <= radius
        potential_temperature_perturbation = 2 * cospi(0.5 * r / radius)^2
    end
    potential_temperature = potential_temperature_ref + potential_temperature_perturbation

    # Exner pressure, solves hydrostatic equation for x[2]
    exner = 1 - g / (c_p * potential_temperature) * x[2]

    # pressure
    p_0 = 100_000.0  # reference pressure
    R = c_p - c_v    # gas constant (dry air)
    p = p_0 * exner^(c_p / R)

    # temperature
    T = potential_temperature * exner

    # density
    rho = p / (R * T)

    v1 = 20.0
    v2 = 0.0
    return prim2cons(SVector(rho, v1, v2, p), equations)
end

@inline function source_terms_gravity(u, x, t, equations::CompressibleEulerEquations2D)
    g = 9.81
    rho, _, rho_v2, _ = u
    return SVector(zero(eltype(u)), zero(eltype(u)), -g * rho, -g * rho_v2)
end

gamma = 1004 / 717
equations = CompressibleEulerEquations2D(gamma)

polydeg = 2
basis = LobattoLegendreBasis(polydeg)

volume_integral = VolumeIntegralFluxDifferencing(flux_kennedy_gruber)
solver = DGSEM(basis, FluxLMARS(340.0), volume_integral)


coordinates_min = (0.0, 0.0)
coordinates_max = (20_000.0, 10_000.0)
trees_per_dimension = (16, 8)
mesh = P4estMesh(trees_per_dimension; polydeg = polydeg,
                 coordinates_min = coordinates_min, coordinates_max = coordinates_max,
                 periodicity = (true, false), initial_refinement_level = 0)

boundary_conditions = Dict(:y_neg => boundary_condition_slip_wall,
                           :y_pos => boundary_condition_slip_wall)

initial_condition = initial_condition_warm_bubble

semi = SemidiscretizationHyperbolic(mesh,
                                         equations,
                                         initial_condition,
                                         solver;
                                         boundary_conditions = boundary_conditions,
                                         source_terms = source_terms_gravity)
tspan = (0.0, 1000.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_callback = AnalysisCallback(semi, interval = 1000)

alive_callback = AliveCallback(analysis_interval = 1000)

save_solution = SaveSolutionCallback(interval = 1000, solution_variables = cons2prim)

callbacks = CallbackSet(summary_callback, analysis_callback, save_solution, alive_callback)

###############################################################################
# run the simulation

sol = solve(
    ode,
    Theseus.ROS2(); # ARS111, ARS222, ARS443
    dt = 0.01, # solve needs some value here but it will be overwritten by the stepsize_callback
    ode_default_options()..., callback = callbacks,
    krylov_algo = :gmres,
);

