# # Using a linearly implicit Rosenbrock solver with adaptive mesh refinement (AMR) in Trixi.jl

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

# We advect a Gaussian pulse on a `TreeMesh`. The `AMRCallback` refines the mesh around the pulse
# and coarsens it again behind it, so the number of degrees of freedom changes during the simulation.
# Theseus resizes its internal buffers (stages, residual, Krylov workspace) whenever this happens.

trixi_include(
    @__MODULE__, joinpath(examples_dir(), "tree_2d_dgsem", "elixir_advection_amr.jl"),
    tspan = (0.0, 2.0), cfl = 5.0, sol = nothing
);

# Drop the `SaveSolutionCallback` of the elixir, we only keep the final state in memory.

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, amr_callback, stepsize_callback);

###############################################################################
# run the simulation

sol = solve(
    ode, Theseus.SSPKnoth();
    dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
    ode_default_options()..., callback = callbacks,
    krylov_algo = :gmres,
    # Trixi.jl stores intermediate values that depend on `u` in the cache of the semidiscretization `p`,
    # so the Jacobian has to differentiate through `p` as well.
    assume_p_const = false,
);

# ### Plot the solution
#
# `sol.prob.p` is the semidiscretization, which holds the refined mesh at the final time.

plot(Trixi.PlotData2DTriangulated(sol.u[end], sol.prob.p))
