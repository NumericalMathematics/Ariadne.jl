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

trixi_include(joinpath(examples_dir(), "tree_2d_dgsem", "elixir_advection_basic.jl"), cfl = 1.25/2, sol = nothing, save_solution = nothing, 
                                                                 initial_refinement_level = 6, advection_velocity = (1.0, 1.0));

ode = semidiscretize(semi, (0.0, 2.0))

###############################################################################
# run the simulation

sol = solve(
	ode, 
    # Implicit.RKImplicitEuler();
    # Implicit.KS22();
    # Implicit.C23();
    # Implicit.L33();
      Implicit.RKTRBDF2();
    dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
	ode_default_options()..., callback = callbacks,
	# verbose=1,
	krylov_algo = :gmres,
);

cfl_vec = [10.0, 5.0, 2.5, 1.25, 1.25/2]
l2_vec = [1.07287321e-02, 2.69376612e-03, 6.75387935e-04, 1.68974528e-04, 4.22696680e-05]
linf_vec = [1.51727157e-02, 3.80957386e-03, 9.55254494e-04, 2.38966657e-04, 5.97799432e-05]
f = Figure()
ax = Axis(f[1, 1], xlabel = "CFL", ylabel = "Error", 
    xscale = log10, yscale = log10, title = "Convergence Plot")

lines!(ax, cfl_vec, l2_vec, label = "L2 norm")
lines!(ax, cfl_vec, linf_vec, label = "L∞ norm")
lines!(ax, cfl_vec, 1e-4.*cfl_vec.^2, label = "2nd order")

axislegend(ax; position = :rb)

f