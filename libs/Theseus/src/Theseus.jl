module Theseus

using UnPack
using LinearAlgebra
import Ariadne: JacobianOperator, newton_krylov!
using Krylov

abstract type RKTableau end

# Wrapper type for solutions from Theseus.jl's own time integrators, partially mimicking
# SciMLBase.ODESolution
struct TimeIntegratorSolution{tType, uType, P}
    t::tType
    u::uType
    prob::P
end

# Abstract supertype of Theseus.jl's own time integrators for dispatch
abstract type AbstractTimeIntegrator end

using DiffEqBase: DiffEqBase

import DiffEqBase: solve, CallbackSet, ODEProblem, SplitODEProblem
export solve, ODEProblem, SplitODEProblem

# Interface required by DiffEqCallbacks.jl
function DiffEqBase.get_tstops(integrator::AbstractTimeIntegrator)
    return integrator.opts.tstops
end
function DiffEqBase.get_tstops_array(integrator::AbstractTimeIntegrator)
    return get_tstops(integrator).valtree
end
function DiffEqBase.get_tstops_max(integrator::AbstractTimeIntegrator)
    return maximum(get_tstops_array(integrator))
end

function finalize_callbacks(integrator::AbstractTimeIntegrator)
    callbacks = integrator.opts.callback

    return if callbacks isa CallbackSet
        foreach(callbacks.discrete_callbacks) do cb
            cb.finalize(cb, integrator.u, integrator.t, integrator)
        end
        foreach(callbacks.continuous_callbacks) do cb
            cb.finalize(cb, integrator.u, integrator.t, integrator)
        end
    end
end

using SciMLBase: SciMLBase

import SciMLBase: get_du, get_tmp_cache, u_modified!,
    init, step!, check_error,
    get_proposed_dt, set_proposed_dt!,
    terminate!, remake, add_tstop!, has_tstop, first_tstop

# To keep backwards compatibility with SciMLBase v2, see
# https://github.com/trixi-framework/Trixi.jl/pull/2918#issuecomment-4233720339
@static if isdefined(SciMLBase, :derivative_discontinuity!)
    import SciMLBase: derivative_discontinuity!
else
    const derivative_discontinuity! = SciMLBase.u_modified!
end


# Abstract base type for time integration schemes
abstract type SimpleImplicitAlgorithm{N} end
abstract type NonLinearImplicitAlgorithm{N} <: SimpleImplicitAlgorithm{N} end
stages(::NonLinearImplicitAlgorithm{N}) where {N} = N

"""
    ImplicitEuler()

The backward (implicit) Euler method: a first-order, single-stage, A-stable,
and L-stable nonlinear implicit Runge-Kutta method.

The stage equation is
```math
u^{n+1} = u^n + \\Delta t \\, f(u^{n+1},\\, t^{n+1}).
```

Each time step requires solving one nonlinear system via Newton-Krylov.
"""
struct ImplicitEuler <: NonLinearImplicitAlgorithm{1} end
function (::ImplicitEuler)(res, uₙ, Δt, f!, du, u, p, t, stages, stage)
    f!(du, u, p, t + Δt) # t = t0 + c_1 * Δt

    res .= uₙ .+ Δt .* du .- u # Δt * a_11
    return nothing
end

"""
    ImplicitMidpoint()

The implicit midpoint method: a second-order, single-stage, A-stable
nonlinear implicit Runge-Kutta method.

The stage equation evaluates ``f`` at the midpoint ``(u^n + u^{n+1})/2``:
```math
u^{n+1} = u^n + \\Delta t \\, f\\!\\left(\\frac{u^n + u^{n+1}}{2},\\; t + \\frac{\\Delta t}{2}\\right).
```

Each time step requires solving one nonlinear system via Newton-Krylov.
"""
struct ImplicitMidpoint <: NonLinearImplicitAlgorithm{1} end
function (::ImplicitMidpoint)(res, uₙ, Δt, f!, du, u, p, t, stages, stage)
    # Evaluate f at midpoint: f((uₙ + u)/2, t + Δt/2)
    # Use res for a temporary allocation (uₙ .+ u) ./ 2
    uuₙ = res
    uuₙ .= 0.5 .* (uₙ .+ u)
    f!(du, uuₙ, p, t + 0.5 * Δt)

    res .= uₙ .+ Δt .* du .- u
    return nothing
end

"""
    ImplicitTrapezoid()

The implicit trapezoidal rule (Crank–Nicolson): a second-order, single-stage,
A-stable (but not L-stable) nonlinear implicit method.

The update averages the RHS at both endpoints:
```math
u^{n+1} = u^n + \\frac{\\Delta t}{2}\\left[f(u^n, t) + f(u^{n+1}, t + \\Delta t)\\right].
```

Each time step requires solving one nonlinear system via Newton-Krylov.
"""
struct ImplicitTrapezoid <: NonLinearImplicitAlgorithm{1} end
function (::ImplicitTrapezoid)(res, uₙ, Δt, f!, du, u, p, t, stages, stage)
    # Need to evaluate f at both endpoints
    # f(uₙ, t) and f(u, t + Δt)
    # Use res as the temporary for duₙ = f(uₙ, t)
    duₙ = res
    f!(duₙ, uₙ, p, t)
    f!(du, u, p, t + Δt)

    res .= uₙ .+ (Δt / 2) .* (duₙ .+ du) .- u
    return nothing
end

"""
	TRBDF2

TR-BDF2 based solver after [Bank1985-gh](@cite).
Using the formula given in [Bonaventura2021-za](@cite) eq (1).
See [Hosea1996-xv](@cite) for how it relates to implicit RK methods
"""
struct TRBDF2 <: NonLinearImplicitAlgorithm{2} end
function (::TRBDF2)(res, uₙ, Δt, f!, du, u, p, t, stages, stage)
    γ = 2 - √2
    return if stage == 1
        # Stage 1: Trapezoidal rule to t + γΔt
        # u here is u₁ candidate
        duₙ = res
        f!(duₙ, uₙ, p, t)
        f!(du, u, p, t + γ * Δt)

        res .= uₙ .+ ((γ / 2) * Δt) .* (duₙ .+ du) .- u
    else
        # Stage 2: BDF2 from t + γΔt to t + Δt
        # Note these are unequal timestep
        f!(du, u, p, t + Δt)

        u₁ = stages[1]

        # Bank1985 defines in eq 32
        # (2-γ)u + (1-γ)Δt * f(u, t+Δt) = 1/γ * u₁ - 1/γ * (1-γ)^2 * uₙ
        # Manual derivation (division by (2-γ) and then move everything to one side.)
        # a₁ = -((1 - γ)^2) / (γ * (2 - γ))
        # a₂ = 1 / (γ * (2 - γ))
        # a₃ = - (1 - γ) / (2 - γ)
        # res .= a₁ .* uₙ .+ a₂ .* u₁ .+  a₃ .* Δt .* du .- u

        # after Bonaventura2021
        # They define the second stage as:
        # u - γ₂ * Δt * f(u, t+Δt) = (1-γ₃)uₙ + γ₃u₁
        # Which differs from Bank1985
        # (2-γ)u + (1-γ)Δt * f(u, t+Δt) = 1/γ * u₁ - 1/γ * (1-γ)^2 * uₙ
        # In the sign of u - γ₂ * Δt
        # a₁ == (1-γ₃)
        # a₂ == γ₃
        # a₃ == -γ₂
        γ₂ = (1 - γ) / (2 - γ)
        γ₃ = 1 / (γ * (2 - γ))

        res .= (1 - γ₃) .* uₙ .+ γ₃ .* u₁ + (γ₂ * Δt) .* du .- u
    end
end

function nonlinear_problem(alg::NonLinearImplicitAlgorithm, f::F) where {F}
    return (res, u, (uₙ, Δt, du, p, t, stages, stage)) -> alg(res, uₙ, Δt, f, du, u, p, t, stages, stage)
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L1
mutable struct NonLinearImplicitOptions{Callback}
    callback::Callback # callbacks; used in Trixi.jl
    adaptive::Bool # whether the algorithm is adaptive; ignored
    dtmax::Float64 # ignored
    maxiters::Int # maximal number of time steps
    tstops::Vector{Float64} # tstops from https://diffeq.sciml.ai/v6.8/basics/common_solver_opts/#Output-Control-1; ignored
    verbose::Int
    algo::Symbol
    krylov_kwargs::Any
end


function NonLinearImplicitOptions(callback, tspan; maxiters = typemax(Int), verbose = 0, krylov_algo = :gmres, krylov_kwargs = (;), kwargs...)
    return NonLinearImplicitOptions{typeof(callback)}(
        callback, false, Inf, maxiters,
        [last(tspan)],
        verbose,
        krylov_algo,
        krylov_kwargs,
    )
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.jl.
mutable struct NonLinearImplicit{
        RealT <: Real, uType, Params, Sol, F, M, Alg <: NonLinearImplicitAlgorithm,
        NonLinearImplicitOptions,
    } <: AbstractTimeIntegrator
    u::uType
    du::uType
    u_tmp::uType
    stages::NTuple{M, uType}
    res::uType
    t::RealT
    dt::RealT # current time step
    dtcache::RealT # ignored
    iter::Int # current number of time steps (iteration)
    p::Params # will be the semidiscretization from Trixi.jl
    sol::Sol # faked
    f::F # `rhs!` of the semidiscretization
    alg::Alg # NonLinearImplicitAlgorithm
    opts::NonLinearImplicitOptions
    finalstep::Bool # added for convenience
end

# Forward integrator.stats.naccept to integrator.iter (see GitHub PR#771)
function Base.getproperty(integrator::NonLinearImplicit, field::Symbol)
    if field === :stats
        return (naccept = getfield(integrator, :iter),)
    end
    # general fallback
    return getfield(integrator, field)
end

function init(
        ode::ODEProblem, alg::NonLinearImplicitAlgorithm{N};
        dt, callback::Union{CallbackSet, Nothing} = nothing, kwargs...,
    ) where {N}
    u = copy(ode.u0)
    du = zero(u)
    res = zero(u)
    u_tmp = similar(u)
    stages = ntuple(_ -> similar(u), Val(N))
    t = first(ode.tspan)
    iter = 0
    integrator = NonLinearImplicit(
        u, du, u_tmp, stages, res, t, dt, zero(dt), iter, ode.p,
        (prob = ode,), ode.f, alg,
        NonLinearImplicitOptions(
            callback, ode.tspan;
            kwargs...,
        ), false
    )

    # initialize callbacks
    if callback isa CallbackSet
        foreach(callback.continuous_callbacks) do cb
            throw(ArgumentError("Continuous callbacks are unsupported with the implicit time integration methods."))
        end
        foreach(callback.discrete_callbacks) do cb
            cb.initialize(cb, integrator.u, integrator.t, integrator)
        end
    end

    return integrator
end

# Fakes `solve`: https://diffeq.sciml.ai/v6.8/basics/overview/#Solving-the-Problems-1
function solve(
        ode::ODEProblem, alg::NonLinearImplicitAlgorithm;
        dt, callback = nothing, kwargs...,
    )
    integrator = init(ode, alg, dt = dt, callback = callback; kwargs...)

    # Start actual solve
    return solve!(integrator)
end

function solve!(integrator::NonLinearImplicit)
    @unpack prob = integrator.sol

    integrator.finalstep = false

    while !integrator.finalstep
        step!(integrator)
    end # "main loop" timer

    finalize_callbacks(integrator)

    return TimeIntegratorSolution(
        (first(prob.tspan), integrator.t),
        (prob.u0, integrator.u),
        integrator.sol.prob,
    )
end

function stage!(integrator::NonLinearImplicit, alg)
    for stage in 1:stages(alg)
        F! = nonlinear_problem(alg, integrator.f)
        # TODO: Pass in `stages[1:(stage-1)]` or full tuple?
        _, stats = newton_krylov!(
            F!, integrator.u_tmp, (integrator.u, integrator.dt, integrator.du, integrator.p, integrator.t, integrator.stages, stage), integrator.res;
            verbose = integrator.opts.verbose, krylov_kwargs = integrator.opts.krylov_kwargs,
            algo = integrator.opts.algo, tol_abs = 6.0e-6,
        )
        @assert stats.solved
        if stage < stages(alg)
            # Store the solution for each stage in stages
            integrator.stages[stage] .= integrator.u_tmp
        end
    end
    return
end

function step!(integrator::NonLinearImplicit)
    @unpack prob = integrator.sol
    @unpack alg = integrator
    t_end = last(prob.tspan)
    callbacks = integrator.opts.callback

    @assert !integrator.finalstep
    if isnan(integrator.dt)
        error("time step size `dt` is NaN")
    end

    # if the next iteration would push the simulation beyond the end time, set dt accordingly
    if integrator.t + integrator.dt > t_end ||
            isapprox(integrator.t + integrator.dt, t_end)
        integrator.dt = t_end - integrator.t
        terminate!(integrator)
    end

    # one time step
    integrator.u_tmp .= integrator.u

    stage!(integrator, alg)

    integrator.u .= integrator.u_tmp

    integrator.iter += 1
    integrator.t += integrator.dt

    begin
        # handle callbacks
        if callbacks isa CallbackSet
            foreach(callbacks.discrete_callbacks) do cb
                if cb.condition(integrator.u, integrator.t, integrator)
                    cb.affect!(integrator)
                end
                return nothing
            end
        end
    end

    # respect maximum number of iterations
    return if integrator.iter >= integrator.opts.maxiters && !integrator.finalstep
        @warn "Interrupted. Larger maxiters is needed."
        terminate!(integrator)
    end
end

# get a cache where the RHS can be stored
get_du(integrator::NonLinearImplicit) = integrator.du
get_tmp_cache(integrator::NonLinearImplicit) = (integrator.u_tmp,)

# some algorithms from DiffEq like FSAL-ones need to be informed when a callback has modified u
derivative_discontinuity!(integrator::NonLinearImplicit, ::Bool) = false

# used by adaptive timestepping algorithms in DiffEq
function set_proposed_dt!(integrator::NonLinearImplicit, dt)
    return integrator.dt = dt
end

# Required e.g. for `glm_speed_callback`
function get_proposed_dt(integrator::NonLinearImplicit)
    return integrator.dt
end

# stop the time integration
function terminate!(integrator::NonLinearImplicit)
    integrator.finalstep = true
    return empty!(integrator.opts.tstops)
end

# used for AMR
function Base.resize!(integrator::NonLinearImplicit, new_size)
    resize!(integrator.u, new_size)
    resize!(integrator.du, new_size)
    return resize!(integrator.u_tmp, new_size)
end

### Helper
jacobian(G!, ode::ODEProblem, Δt) = jacobian(G!, ode.f, ode.u0, ode.p, Δt, first(ode.tspan))

function jacobian(G!, f!, uₙ, p, Δt, t)
    u = copy(uₙ)
    du = zero(uₙ)
    res = zero(uₙ)

    F! = nonlinear_problem(G!, f!)

    J = Ariadne.JacobianOperator(F!, res, u, (uₙ, Δt, du, p, t))
    return collect(J)
end

include("rosenbrock/rosenbrock.jl")
include("imex/imex.jl")
include("dirk/dirk.jl")
end # module Theseus
