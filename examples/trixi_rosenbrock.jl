# # Using a linearly implicit Rosenbrock solver based on Ariadne with Trixi.jl

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

trixi_include(@__MODULE__, joinpath(examples_dir(), "tree_2d_dgsem", "elixir_advection_basic.jl"), cfl = 10.0, sol = nothing);

###############################################################################
# run the simulation

sol = solve(
    ode, Theseus.SSPKnoth();
    dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
    ode_default_options()..., callback = callbacks,
    # verbose=1,
    krylov_algo = :gmres,
);
