# # Using a diagonally implicit Runge-Kutta (DIRK) solver based on Ariadne with Trixi.jl

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

polydeg = 3
initial_refinement_level = 4
trixi_include(@__MODULE__, joinpath(examples_dir(), "tree_2d_dgsem", "elixir_euler_warm_bubble.jl"), cfl = 1.0, sol = nothing, polydeg = 3, initial_refinement_level = initial_refinement_level);

###############################################################################
# run the simulation

sol = solve(
    ode, Theseus.Crouzeix23();
    dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
    ode_default_options()..., callback = callbacks,
    # verbose=1,
    krylov_algo = :gmres,
);
