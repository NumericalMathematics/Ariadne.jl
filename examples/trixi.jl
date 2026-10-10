# # Using the an implicit solver based on Ariadne with Trixi.jl

using Trixi
using Theseus
using CairoMakie
using LinearAlgebra
import Ariadne: JacobianOperator


# Notes:
# Must disable both Polyester and LoopVectorization for Enzyme to be able to differentiate Trixi.jl
#
# LocalPreferences.jl
# ```toml
# [Trixi]
# loop_vectorization = false
# backend = "static"
# ```

@assert Trixi._PREFERENCE_THREADING !== :polyester
@assert !Trixi._PREFERENCE_LOOPVECTORIZATION

# ## Load Trixi Example
trixi_include(@__MODULE__, joinpath(examples_dir(), "tree_2d_dgsem", "elixir_advection_basic.jl"), sol = nothing);

u = copy(ode.u0)
du = zero(ode.u0)
res = zero(ode.u0)

F! = Theseus.nonlinear_problem(Theseus.ImplicitEuler(), ode.f)
J = JacobianOperator(F!, res, u, (ode.u0, 1.0, du, ode.p, 0.0, (), 1))

out = zero(u)
v = zero(u)

## precompile
mul!(u, J, v)
F!(res, u, (ode.u0, 1.0, du, ode.p, 0.0, (), 1))

@time mul!(u, J, v)
@time F!(res, u, (ode.u0, 1.0, du, ode.p, 0.0, (), 1))

# Cost of time(mul!) ≈ 2 * time(F!)

# ### Solve using ODE interface

sol_trbdf2 = solve(
    ode, Theseus.TRBDF2();
    dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
    ode_default_options()..., callback = callbacks,
    ## verbose=1,
    krylov_algo = :gmres,
    ## krylov_kwargs=(;verbose=1)
);

# #### Plot the solution

# We have to manually convert the sol since Theseus has it's own lightweight solution type.

plot(Trixi.PlotData2DTriangulated(sol_trbdf2.u[end], sol_trbdf2.prob.p))

# ### Solve using OrdinaryDiffEqSDIRK

import OrdinaryDiffEqSDIRK
import DifferentiationInterface: AutoFiniteDiff
sol_sdrik = solve(
    ode, OrdinaryDiffEqSDIRK.TRBDF2(autodiff = AutoFiniteDiff());
    dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
    ode_default_options()..., callback = callbacks,
    adaptive = false
);

# #### Plot the solution

plot(Trixi.PlotData2DTriangulated(sol_sdrik.u[end], sol_sdrik.prob.p))

# ## Increase CFL numbers

trixi_include(@__MODULE__, joinpath(examples_dir(), "tree_2d_dgsem", "elixir_advection_basic.jl"), cfl = 10, sol = nothing);

sol = solve(
    ode, Theseus.ImplicitEuler();
    dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
    ode_default_options()..., callback = callbacks,
    ## verbose=1,
    krylov_algo = :gmres,
    ## krylov_kwargs=(;verbose=1)
);

@show callbacks.discrete_callbacks[4]

# ### Plot the solution

plot(Trixi.PlotData2DTriangulated(sol.u[end], sol.prob.p))
