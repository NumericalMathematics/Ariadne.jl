##
# SciMLBase.jl interface: `solve`, `init`, `solve!`, `step!` for `SteadyStateProblem`s and
# `NonlinearProblem`s
##

const SteadyOrNonlinearProblem = Union{SteadyStateProblem, NonlinearProblem}

# In-place steady residual `f!(res, u, p)` of the problems
struct SteadyStateResidual{F, T}
    f::F
    t::T
end
(r::SteadyStateResidual)(res, u, p) = (r.f(res, u, p, r.t); nothing)

struct SteadyStateResidualOOP{F, T}
    f::F
    t::T
end
(r::SteadyStateResidualOOP)(res, u, p) = (res .= r.f(u, p, r.t); nothing)

struct NonlinearResidual{F}
    f::F
end
(r::NonlinearResidual)(res, u, p) = (r.f(res, u, p); nothing)

struct NonlinearResidualOOP{F}
    f::F
end
(r::NonlinearResidualOOP)(res, u, p) = (res .= r.f(u, p); nothing)

# The function of the user without the SciMLFunction wrapper
unwrap(f::Union{SciMLBase.ODEFunction, SciMLBase.NonlinearFunction}) = f.f
unwrap(f) = f

function steady_residual(prob::SteadyStateProblem, t)
    f = unwrap(prob.f)
    return SciMLBase.isinplace(prob) ? SteadyStateResidual(f, t) : SteadyStateResidualOOP(f, t)
end
function steady_residual(prob::NonlinearProblem, _)
    f = unwrap(prob.f)
    return SciMLBase.isinplace(prob) ? NonlinearResidual(f) : NonlinearResidualOOP(f)
end

# du/dτ = f for steady-state problems and du/dτ = -f for nonlinear problems
pseudo_time_sign(::SteadyStateProblem) = 1
pseudo_time_sign(::NonlinearProblem) = -1

"""
    PseudoTransientCache

Cache of `init(prob, alg::PseudoTransientNewtonKrylov)` for a `SteadyStateProblem` or
`NonlinearProblem`. `solve!(cache)` runs the pseudo-transient continuation and returns
the solution; `step!(cache)` takes one accepted pseudo-time step. The
[`PseudoTransientWorkspace`](@ref) is `cache.ws`.
"""
mutable struct PseudoTransientCache{P, A, W, K}
    prob::P
    alg::A
    ws::W
    solve_kwargs::K
end

function retcode(status::Symbol)
    status === :converged && return ReturnCode.Success
    status === :max_iterations && return ReturnCode.MaxIters
    status === :terminated && return ReturnCode.Terminated
    status === :nonfinite && return ReturnCode.Unstable
    status === :cfl_too_small && return ReturnCode.ConvergenceFailure
    return ReturnCode.Failure
end

function should_alias_u0(alias, alias_u0)
    alias_u0 !== nothing && return alias_u0
    if alias isa Bool
        return alias
    elseif alias !== nothing && hasproperty(alias, :alias_u0) && alias.alias_u0 !== nothing
        return alias.alias_u0
    end
    return false
end

function CommonSolve.init(prob::SteadyOrNonlinearProblem, alg::PseudoTransientNewtonKrylov, args...; kwargs...)
    return SciMLBase.__init(prob, alg, args...; kwargs...)
end

function SciMLBase.__init(
        prob::SteadyOrNonlinearProblem, alg::PseudoTransientNewtonKrylov, args...;
        abstol = nothing, reltol = nothing, maxiters = 1000, verbose = false,
        callback = nothing, alias = nothing, alias_u0 = nothing, t = Inf,
        termination_condition = nothing, kwargs...
    )
    kwargs = (; prob.kwargs..., kwargs...)
    if !isempty(kwargs)
        @warn "PseudoTransientNewtonKrylov: ignoring unsupported keyword arguments $(keys(kwargs))"
    end
    termination_condition === nothing || throw(
        ArgumentError(
            "PseudoTransientNewtonKrylov: `termination_condition` is not supported; use `abstol`, `reltol`, and `callback`"
        )
    )
    if callback !== nothing && !(callback isa Function)
        throw(ArgumentError("PseudoTransientNewtonKrylov: `callback` must be a function `(ws, info) -> Bool`, not a $(typeof(callback))"))
    end
    u = should_alias_u0(alias, alias_u0) ? prob.u0 : copy(prob.u0)
    f! = steady_residual(prob, t)
    ws = PseudoTransientWorkspace(f!, u, prob.p, alg; σ = pseudo_time_sign(prob))
    solve_kwargs = (;
        abstol = something(abstol, 0.0), reltol = something(reltol, 1.0e-8), maxiters,
        verbose = verbose isa Bool ? Int(verbose) : verbose, callback,
    )
    return PseudoTransientCache(prob, alg, ws, solve_kwargs)
end

function CommonSolve.solve!(cache::PseudoTransientCache)
    (; prob, alg, ws) = cache
    pseudo_transient!(ws; cache.solve_kwargs...)
    stats = ws.stats
    nlstats = NLStats(
        stats.residual_evaluations, stats.preconditioner_builds,
        stats.preconditioner_builds, stats.newton_iterations, stats.steps
    )
    resid = copy(ws.res)
    return SciMLBase.build_solution(
        prob, alg, ws.u, resid; retcode = retcode(ws.status), stats = nlstats,
        original = ws
    )
end

"""
    step!(cache::PseudoTransientCache) -> status

Take one accepted pseudo-time step ([`ptc_step!`](@ref)) of the pseudo-transient
continuation, starting it ([`ptc_start!`](@ref)) at the first call. Returns `:accepted` or
`:cfl_too_small`. The convergence test is left to the caller, e.g.,
`cache.ws.norm_res <= reltol * cache.ws.norm_res_initial`.
"""
function SciMLBase.step!(cache::PseudoTransientCache)
    ws = cache.ws
    verbose = cache.solve_kwargs.verbose
    if ws.status === :initialized
        ptc_start!(ws; verbose)
        isfinite(ws.norm_res) || return :nonfinite
    end
    return ptc_step!(ws; verbose)
end

function CommonSolve.solve(prob::SteadyOrNonlinearProblem, alg::PseudoTransientNewtonKrylov, args...; kwargs...)
    return SciMLBase.__solve(prob, alg, args...; kwargs...)
end

function SciMLBase.__solve(prob::SteadyOrNonlinearProblem, alg::PseudoTransientNewtonKrylov, args...; kwargs...)
    return CommonSolve.solve!(SciMLBase.__init(prob, alg, args...; kwargs...))
end
