# # Using the an implicit solver based on Ariadne with Trixi.jl

using Trixi
using Implicit
using CairoMakie


# Notes:
# Must disable both Polyester and LoopVectorization for Enzyme to be able to differentiate Trixi.jl
# Using https://github.com/trixi-framework/Trixi.jl/pull/2295
#
# LocalPreferences.jl
# ```toml
# [Trixi]
# loop_vectorization = false
# polyester = false
# ```

@assert !Trixi._PREFERENCE_POLYESTER
@assert !Trixi._PREFERENCE_LOOPVECTORIZATION

trixi_include(joinpath(examples_dir(), "tree_2d_dgsem", "elixir_advection_basic.jl"), CFL = 10.0, sol = nothing);

###############################################################################
# run the simulation

sol = solve(
	ode, Implicit.Rosenbrock();
	dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
	ode_default_options()..., callback = callbacks,
	# verbose=1,
	krylov_algo = :gmres,
);
