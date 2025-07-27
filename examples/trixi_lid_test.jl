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

#@assert !Trixi._PREFERENCE_POLYESTER
#@assert !Trixi._PREFERENCE_LOOPVECTORIZATION

trixi_include(joinpath(examples_dir(), "tree_2d_dgsem", "elixir_navierstokes_lid_driven_cavity.jl"), sol = nothing, mu = 0.1);
dt = 0.01/8
ode = semidiscretize(semi, (0.0, 10.0))
###############################################################################
# run the simulation

sol = solve(
	ode, 
     Implicit.RKLSSPIMEX332Z();
	#Implicit.KS22();
    dt = dt, # solve needs some value here but it will be overwritten by the stepsize_callback
	ode_default_options()..., callback = callbacks,
	# verbose=1,
	krylov_algo = :gmres,
);

## dt_vec = [0.001, 0.001/2, 0.0001]
## l2_vec = [4.95311953e-04, 2.48996489e-04   ,5.40458032e-05, ]
