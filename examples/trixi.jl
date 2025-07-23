# # Using the an implicit solver based on Ariadne with Trixi.jl

using Trixi
using Theseus
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

# ## Load Trixi Example
# trixi_include(joinpath(examples_dir(), "tree_2d_dgsem", "elixir_advection_basic.jl"), sol = nothing);
trixi_include(joinpath(examples_dir(), "tree_2d_dgsem", "elixir_advection_basic.jl"));

ref = copy(sol)

u = copy(ode.u0)
du = zero(ode.u0)
res = zero(ode.u0)

F! = Theseus.nonlinear_problem(Theseus.ImplicitEuler(), ode.f)
J = Theseus.Ariadne.JacobianOperator(F!, res, u, (ode.u0, 1.0, du, ode.p, 0.0, (), 1))

using LinearAlgebra
out = zero(u)
v = zero(u)
@time mul!(u, J, v)
@time F!(res, u, (ode.u0, 1.0, du, ode.p, 0.0, (), 1))

# Cost of time(Jvp) ≈ 2 * time(rhs)

# ### Jacobian (of the implicit function given the ode)
# J = Theseus.jacobian(Theseus.ImplicitEuler(), ode, 1.0)

# ### Solve using ODE interface

sol_euler = solve(
    ode, Theseus.ImplicitEuler();
    dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
    ode_default_options()..., callback = callbacks,
    # verbose=1,
    krylov_algo = :gmres,
    # krylov_kwargs=(;verbose=1)
);

sol_midpoint = solve(
    ode, Theseus.ImplicitMidpoint();
    dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
    ode_default_options()..., callback = callbacks,
    # verbose=1,
    krylov_algo = :gmres,
    # krylov_kwargs=(;verbose=1)
);

sol_trapezoid = solve(
    ode, Theseus.ImplicitTrapezoid();
    dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
    ode_default_options()..., callback = callbacks,
    # verbose=1,
    krylov_algo = :gmres,
    # krylov_kwargs=(;verbose=1)
);


sol_trbdf2 = solve(
    ode, Theseus.TRBDF2();
    dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
    ode_default_options()..., callback = callbacks,
    # verbose=1,
    krylov_algo = :gmres,
    # krylov_kwargs=(;verbose=1)
);

import OrdinaryDiffEqSDIRK
import DifferentiationInterface: AutoFiniteDiff
sol_sdrik = solve(
    ode, OrdinaryDiffEqSDIRK.TRBDF2(autodiff = AutoFiniteDiff());
    dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
    ode_default_options()..., callback = callbacks,
    adaptive = false
);

# ### Plot the (reference) solution

# We have to manually convert the sol since Theseus has it's own leightweight solution type.
# Create an extension.
## pd = PlotData2D(sol.u[end], sol.prob.p)

plot(Trixi.PlotData2DTriangulated(ref.u[1], ref.prob.p))

# ### Plot the solution

plot(Trixi.PlotData2DTriangulated(sol_trbdf2.u[end], sol.prob.p))

# ## Increase CFL numbers

trixi_include(joinpath(examples_dir(), "tree_2d_dgsem", "elixir_advection_basic.jl"), cfl = 10, sol = nothing);

sol = solve(
    ode, Theseus.ImplicitEuler();
    dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
    ode_default_options()..., callback = callbacks,
    # verbose=1,
    krylov_algo = :gmres,
    # krylov_kwargs=(;verbose=1)
);

@show callbacks.discrete_callbacks[4]

# ### Plot the solution

plot(Trixi.PlotData2DTriangulated(sol.u[end], sol.prob.p))
