# # Using an implicit-explicit (IMEX) Runge-Kutta solver based on Ariadne with Trixi.jl

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

# First call to load callbacks
trixi_include(joinpath(examples_dir(), "p4est_2d_dgsem", "elixir_navierstokes_vortex_street.jl"), sol = nothing, analysis_interval = 10);

stepsize_callback = StepsizeCallback(cfl = 1.0)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, stepsize_callback)
tspan = (0.0, 0.5)
trixi_include(joinpath(examples_dir(), "p4est_2d_dgsem", "elixir_navierstokes_vortex_street.jl"), sol = nothing, callbacks = callbacks, tspan = tspan);
###############################################################################
# run the simulation

sol = solve(
    ode,
    Theseus.ARS443(); # ARS111, ARS222, ARS443
    dt = 0.001, # solve needs some value here but it will be overwritten by the stepsize_callback
    ode_default_options()..., callback = callbacks,
    krylov_algo = :gmres,
);
