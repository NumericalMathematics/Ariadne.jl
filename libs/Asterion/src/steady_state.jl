##
# Pseudo-transient continuation (PTC) with Jacobian-free Newton-Krylov
##

"""
    PseudoTransientResidual(f!, σ)

Residual of one implicit Euler step of the pseudo-time ODE `du/dτ = σ f(u, p)`,
`G(u) = (u - uₙ) ./ Δτ - σ f(u, p)`, called as `G(res, u, P::PseudoTransientParameters)`.
"""
struct PseudoTransientResidual{F}
    f::F
    σ::Int
end

"""
    PseudoTransientParameters

Parameters of the [`PseudoTransientResidual`](@ref), mutated in place in each
pseudo-time step:

- `u_n`: the state at the beginning of the pseudo-time step
- `inv_dtau`: the inverse pseudo-time steps `1 ./ Δτ`
- `p`: the parameters of the user
- `cfl`: `Ref` to the CFL number of the pseudo-time step

Preconditioner builders for [`PseudoTransientNewtonKrylov`](@ref) receive the
[`JacobianOperator`](@ref Ariadne.JacobianOperator) `J` of the pseudo-transient residual and can access
`J.u` (state), `J.p.p` (parameters of the user), `J.p.inv_dtau`, `J.p.cfl[]`, and
`J.f.f` (the steady residual `f!`), `J.f.σ`.
"""
struct PseudoTransientParameters{A, P, C}
    u_n::A
    inv_dtau::A
    p::P
    cfl::Base.RefValue{C}
end

Ariadne.user_parameters(P::PseudoTransientParameters) = P.p

function (G::PseudoTransientResidual)(res, u, P::PseudoTransientParameters)
    G.f(res, u, P.p)
    σ = G.σ
    (; u_n, inv_dtau) = P
    @. res = inv_dtau * (u - u_n) - σ * res
    return nothing
end

##
# CFL strategies
##

"""
    AbstractCFLStrategy

Evolution of the CFL number in [`PseudoTransientNewtonKrylov`](@ref).
See [`SER`](@ref).
"""
abstract type AbstractCFLStrategy end

"""
    SER(; initial = 1.0, min = 1.0e-6, max = 1.0e8, growth_min = 0.1, growth_max = 2.0,
          exponent = 1.0, reference = :previous, n_variables = 0,
          rejection_factor = 0.1)

Switched evolution relaxation (SER, Mulder & van Leer 1985) of the CFL number.
After each accepted pseudo-time step, the CFL number is updated with the residual ratio
`f = R_new / R_ref`:

- `reference = :previous` (local SER): `CFL ← CFL ⋅ clamp(f^(-exponent), growth_min, growth_max)`
  with `R_ref` the residual of the previous step,
- `reference = :initial` (global SER, Lodares et al. 2022, Eq. 122):
  `CFL ← clamp(initial / f^exponent, growth_min ⋅ CFL, growth_max ⋅ CFL)` with `R_ref`
  the initial residual,

and the result is clamped to `[min, max]`.

The residual ratio `f` is computed with the norm of the algorithm if `n_variables == 0`,
and otherwise with [`variable_residual_ratio`](@ref Ariadne.variable_residual_ratio) for `n_variables` variables stored
interleaved (median over the variables of the maximum of the ratios of the 2-norm and
max-norm, Lodares et al. 2022, Eq. 123).

When a pseudo-time step is rejected, the CFL number is multiplied by `rejection_factor`.
If it drops below `min`, the solver stops with status `:cfl_too_small`.


`initial`, `min`, and `max` can be `NamedTuple`s for several CFL numbers that are
evolved with the same factor, e.g., `initial = (advective = 1.0, diffusive = 0.1)`, which
are passed to the `dtau!` hook of [`PseudoTransientNewtonKrylov`](@ref) as a `NamedTuple`.
"""
Base.@kwdef struct SER{C, L, H} <: AbstractCFLStrategy
    initial::C = 1.0
    min::L = 1.0e-6
    max::H = 1.0e8
    growth_min::Float64 = 0.1
    growth_max::Float64 = 2.0
    exponent::Float64 = 1.0
    reference::Symbol = :previous
    n_variables::Int = 0
    rejection_factor::Float64 = 0.1
    function SER(
            initial::C, min::L, max::H, growth_min, growth_max, exponent, reference, n_variables,
            rejection_factor
        ) where {C, L, H}
        @assert reference in (:previous, :initial) "reference must be :previous or :initial"
        @assert 0 < rejection_factor < 1 "rejection_factor must be in (0, 1)"
        @assert 0 <= growth_min <= growth_max "need 0 <= growth_min <= growth_max"
        return new{C, L, H}(
            initial, min, max, growth_min, growth_max, exponent, reference, n_variables,
            rejection_factor
        )
    end
end

initial_cfl(s::SER) = s.initial

_get(x::NamedTuple, k) = x[k]
_get(x, _) = x
_map_cfl(f, cfl::Number, args...) = f(cfl, args...)
function _map_cfl(f, cfl::NamedTuple{K}, args...) where {K}
    return NamedTuple{K}(map(k -> f(cfl[k], map(a -> _get(a, k), args)...), K))
end

"""
    update_cfl(strategy, cfl, info) -> cfl

New CFL number after an accepted pseudo-time step. `info` has the fields
`norm_res`, `norm_res_prior`, `norm_res_initial`, `res`, `res_prior`, `res_initial`
(norms and vectors of the steady residual).
"""
function update_cfl(s::SER, cfl, info)
    if s.n_variables > 0
        res_ref = s.reference === :previous ? info.res_prior : info.res_initial
        f = variable_residual_ratio(info.res, res_ref, s.n_variables)
    else
        norm_ref = s.reference === :previous ? info.norm_res_prior : info.norm_res_initial
        f = info.norm_res / norm_ref
    end
    (; growth_min, growth_max, exponent) = s
    return _map_cfl(cfl, s.initial, s.min, s.max) do c, c₀, lo, hi
        if s.reference === :previous
            c_new = c * clamp(f^(-exponent), growth_min, growth_max)
        else
            c_new = clamp(c₀ / f^exponent, growth_min * c, growth_max * c)
        end
        clamp(c_new, lo, hi)
    end
end

"""
    reject_cfl(strategy, cfl) -> cfl

New CFL number after a rejected pseudo-time step.
"""
reject_cfl(s::SER, cfl) = _map_cfl(c -> c * s.rejection_factor, cfl)

"""
    cfl_too_small(strategy, cfl) -> Bool
"""
cfl_too_small(s::SER, cfl) = any(_flatten(_map_cfl(<, cfl, s.min)))
_flatten(c::Union{Number, Bool}) = (c,)
_flatten(c::NamedTuple) = values(c)

# Default pseudo-time step: a global Δτ = CFL
function default_dtau!(dtau, u, p, cfl)
    cfl isa Number || throw(ArgumentError("multiple CFL numbers need a `dtau!` hook"))
    fill!(dtau, cfl)
    return dtau
end

##
# Algorithm
##

"""
    PseudoTransientNewtonKrylov(; kwargs...)

Pseudo-transient continuation (PTC) for steady states `f(u, p) = 0` with
Jacobian-free Newton-Krylov (Enzyme.jl Jacobian-vector products, Krylov.jl) for the
implicit pseudo-time steps. Each pseudo-time step solves
```math
G(u) = \\frac{u - u_n}{Δτ} - σ f(u, p) = 0
```
by `newton_iterations` inexact Newton steps, i.e., implicit Euler for the pseudo-time ODE
`du/dτ = σ f(u, p)`.

Use it with [`pseudo_transient!`](@ref).

## Keyword arguments
- `cfl = SER()`: CFL evolution strategy, see [`SER`](@ref).
- `dtau! = nothing`: hook `dtau!(dtau, u, p, cfl)` that fills the (local) pseudo-time
  steps for the CFL number `cfl`, e.g., from the convective CFL condition. The default is
  the global pseudo-time step `Δτ = cfl`.
- `isadmissible = nothing`: hook `isadmissible(u, p) -> Bool`, e.g., positivity of
  density and pressure. Pseudo-time steps leading to inadmissible states are rejected
  (and the CFL number is reduced). Use it also in an
  [`AdmissibleLineSearch`](@ref Ariadne.AdmissibleLineSearch) to shorten Newton steps instead.
- `newton_iterations = 1`: maximal number of Newton iterations per pseudo-time step.
- `newton_reltol = 0.0`: relative tolerance of the Newton iterations per pseudo-time step
  (with `0`, exactly `newton_iterations` iterations are taken).
- `forcing = Fixed(1.0e-2)`: forcing term of the inexact Newton method.
- `linesearch = NoLineSearch()`: line search of the Newton iterations.
- `preconditioner = nothing`: right preconditioner of the Krylov solver: `nothing`,
  an [`AbstractPreconditioner`](@ref Ariadne.AbstractPreconditioner) such as [`LaggedPreconditioner`](@ref Ariadne.LaggedPreconditioner) (reused
  across pseudo-time steps and solves; its builder gets the [`JacobianOperator`](@ref Ariadne.JacobianOperator)
  of the pseudo-transient residual, see [`PseudoTransientParameters`](@ref)), or a function
  `J -> operator` called before each Krylov solve. Preconditioners are applied with
  `ldiv!` (`krylov_kwargs = (; ldiv = true)`).
- `krylov = :gmres`: Krylov method.
- `krylov_kwargs = (; ldiv = true, itmax = 100)`: keyword arguments of the Krylov solver.
- `reject_krylov_failure = true`: reject pseudo-time steps in which the Krylov solver
  does not reach its tolerance.
- `max_residual_growth = Inf`: reject pseudo-time steps that increase the steady residual
  norm by more than this factor.
- `norm = LinearAlgebra.norm`: norm of the steady residual for the termination criteria
  and the SER, and of the residual of the Newton iterations, e.g., [`ScaledNorm`](@ref Ariadne.ScaledNorm).
- `assume_p_const = false`: passed to [`JacobianOperator`](@ref Ariadne.JacobianOperator).

The preconditioner is refreshed (see [`refresh!`](@ref Ariadne.refresh!)) after rejected steps.
"""
Base.@kwdef struct PseudoTransientNewtonKrylov{CS, D, A, F, LS, P, K, N}
    cfl::CS = SER()
    dtau!::D = nothing
    isadmissible::A = nothing
    newton_iterations::Int = 1
    newton_reltol::Float64 = 0.0
    forcing::F = Ariadne.Fixed(1.0e-2)
    linesearch::LS = NoLineSearch()
    preconditioner::P = nothing
    krylov::Symbol = :gmres
    krylov_kwargs::K = (; ldiv = true, itmax = 100)
    reject_krylov_failure::Bool = true
    max_residual_growth::Float64 = Inf
    norm::N = LinearAlgebra.norm
    assume_p_const::Bool = false
end

instantiate_preconditioner(P, f!, u, p) = P

"""
    PseudoTransientStats

Statistics of [`pseudo_transient!`](@ref):
- `steps`: accepted pseudo-time steps
- `rejected_steps`: rejected pseudo-time steps
- `newton_iterations`: Newton iterations (including those of rejected steps)
- `krylov_iterations`: Krylov iterations (including those of rejected steps)
- `krylov_failures`: Krylov solves that did not reach their tolerance
- `residual_evaluations`: evaluations of `f!` (without line search trials and
  Jacobian-vector products)
- `preconditioner_builds`: builds of a [`LaggedPreconditioner`](@ref Ariadne.LaggedPreconditioner)
- `norm_res_initial`, `norm_res`: initial and final norm of the steady residual
- `timings`: `Dict` of wall times in seconds (`:total`, `:newton` (including the
  preconditioner), `:residual`)
"""
Base.@kwdef mutable struct PseudoTransientStats
    steps::Int = 0
    rejected_steps::Int = 0
    newton_iterations::Int = 0
    krylov_iterations::Int = 0
    krylov_failures::Int = 0
    residual_evaluations::Int = 0
    preconditioner_builds::Int = 0
    norm_res_initial::Float64 = NaN
    norm_res::Float64 = NaN
    timings::Dict{Symbol, Float64} = Dict{Symbol, Float64}()
end

"""
    PseudoTransientWorkspace(f!, u, p, alg::PseudoTransientNewtonKrylov; σ = 1)

Workspace of [`pseudo_transient!`](@ref) for the steady residual `f!(res, u, p)` and the
pseudo-time ODE `du/dτ = σ f(u, p)`. `u` is the initial guess and is updated in place.

## Fields of interest
- `u`: the state, `p`: the parameters
- `res`: the steady residual `f(u, p)` of the last accepted state
- `dtau`: the pseudo-time steps of the last step
- `cfl`: the current CFL number
- `newton`: the [`NewtonKrylovWorkspace`](@ref Ariadne.NewtonKrylovWorkspace) of the pseudo-transient residual with its
  [`JacobianOperator`](@ref Ariadne.JacobianOperator) `newton.J`
- `preconditioner`: the preconditioner (e.g., a [`LaggedPreconditioner`](@ref Ariadne.LaggedPreconditioner))
- `stats`: [`PseudoTransientStats`](@ref)
- `history`: vector of `NamedTuple`s `(; step, cfl, norm_res, newton_iterations,
  krylov_iterations, rejections, time)` per accepted step (`step = 0`: initial state)
- `status`: `:converged`, `:max_iterations`, `:cfl_too_small`, `:nonfinite` (non-finite
  initial residual), `:terminated` (by the callback), or `:initialized`

For adjoint and sensitivity computations at the converged state, see
[`steady_jacobian`](@ref).
"""
mutable struct PseudoTransientWorkspace{ALG, F, A, P, PP, NW, PC, C}
    const alg::ALG
    const f::F
    const σ::Int
    const u::A
    const p::P
    const res::A
    const res_initial::A
    const res_prior::A
    const dtau::A
    const params::PP
    const newton::NW
    const preconditioner::PC
    cfl::C
    const stats::PseudoTransientStats
    const history::Vector{Any}
    status::Symbol
    # state of the iteration (see `ptc_start!` and `ptc_step!`)
    norm_res::Float64
    norm_res_initial::Float64
    norm_res_prior::Float64
    nsteps::Int
    t_start::UInt64
end

function PseudoTransientWorkspace(
        f!, u::AbstractArray, p, alg::PseudoTransientNewtonKrylov; σ::Integer = 1
    )
    @assert σ == 1 || σ == -1 "σ must be ±1"
    t₀ = time_ns()
    alloc() = (x = similar(u); Enzyme.make_zero!(x); x)
    res, res_initial, res_prior, dtau = alloc(), alloc(), alloc(), alloc()
    cfl = initial_cfl(alg.cfl)
    params = PseudoTransientParameters(alloc(), alloc(), p, Ref(cfl))
    G = PseudoTransientResidual(f!, Int(σ))
    newton = NewtonKrylovWorkspace(
        G, u, params, alloc(), Val(alg.krylov);
        alg.assume_p_const, alg.norm
    )
    preconditioner = instantiate_preconditioner(alg.preconditioner, f!, u, p)
    stats = PseudoTransientStats()
    stats.timings[:setup] = (time_ns() - t₀) / 1.0e9
    return PseudoTransientWorkspace(
        alg, f!, Int(σ), u, p, res, res_initial, res_prior, dtau, params, newton,
        preconditioner, cfl, stats, Any[], :initialized, NaN, NaN, NaN, 0, time_ns()
    )
end

"""
    steady_jacobian(ws::PseudoTransientWorkspace; assume_p_const = false)

[`JacobianOperator`](@ref Ariadne.JacobianOperator) of the steady residual `f!(res, u, p)` at the current state
`ws.u` (e.g., the converged steady state), with Jacobian-vector products by forward mode
and transposed products `mul!(y, transpose(J), x)` by reverse mode, e.g., for an adjoint
solve `(∂f/∂u)ᵀ λ = g`. Note that it aliases `ws.u`.
"""
function steady_jacobian(ws::PseudoTransientWorkspace; assume_p_const::Bool = false)
    res = similar(ws.res)
    Enzyme.make_zero!(res)
    return JacobianOperator(ws.f, res, ws.u, ws.p; assume_p_const)
end

function evaluate_steady!(ws::PseudoTransientWorkspace)
    t = @elapsed ws.f(ws.res, ws.u, ws.p)
    ws.stats.timings[:residual] = get(ws.stats.timings, :residual, 0.0) + t
    ws.stats.residual_evaluations += 1
    # The norm prepared by the Newton workspace (e.g., a `ScaledNorm` with its scales
    # expanded once), which applies to the steady residual as well
    return ws.newton.norm(ws.res)
end

function set_dtau!(ws::PseudoTransientWorkspace)
    (; alg, dtau, u, p, cfl, params) = ws
    if alg.dtau! === nothing
        default_dtau!(dtau, u, p, cfl)
    else
        alg.dtau!(dtau, u, p, cfl)
    end
    @. params.inv_dtau = inv(dtau)
    params.cfl[] = cfl
    return nothing
end

preconditioner_builds(P::LaggedPreconditioner) = P.n_builds
preconditioner_builds(P) = 0

refresh_preconditioner!(P::LaggedPreconditioner) = refresh!(P)
refresh_preconditioner!(P) = nothing

const PTC_KWARGS_DOCS = """
## Keyword arguments
- `abstol = 0.0`, `reltol = 1.0e-8`: the iteration stops when the norm of the steady
  residual satisfies `‖f(u)‖ <= abstol` or `‖f(u)‖ <= reltol ‖f(u₀)‖`.
- `maxiters = 1000`: maximal number of accepted pseudo-time steps.
- `verbose = 0`: `1` prints a line per pseudo-time step, `2` also the rejections,
  `3` also the Newton iterations.
- `callback = nothing`: `callback(ws, info) -> Bool` called after each accepted step
  with `info = (; step, cfl, norm_res, norm_res_initial, newton_iterations,
  krylov_iterations, rejections)`; return `true` to stop (status `:terminated`).
"""

"""
    pseudo_transient!(f!, u, p, alg = PseudoTransientNewtonKrylov(); σ = 1, kwargs...)
    pseudo_transient!(ws::PseudoTransientWorkspace; kwargs...)

Compute a steady state `f(u, p) = 0` by pseudo-transient continuation of
`du/dτ = σ f(u, p)` with the [`PseudoTransientNewtonKrylov`](@ref) algorithm `alg`.
`u` is the initial guess and is updated in place.
Returns `(u, ws)` with the [`PseudoTransientWorkspace`](@ref) `ws` (fields `status`,
`stats`, `history`, …).

Calling `pseudo_transient!(ws)` again continues from the current state and CFL number
(e.g., after changing parameters in `ws.p` in place); `reltol` then refers to the residual
at the restart, and `stats` and `history` are accumulated.

$(PTC_KWARGS_DOCS)
"""
function pseudo_transient!(
        f!, u::AbstractArray, p, alg::PseudoTransientNewtonKrylov = PseudoTransientNewtonKrylov();
        σ::Integer = 1, kwargs...
    )
    ws = PseudoTransientWorkspace(f!, u, p, alg; σ)
    return pseudo_transient!(ws; kwargs...)
end

function pseudo_transient!(
        ws::PseudoTransientWorkspace;
        abstol = 0.0, reltol = 1.0e-8, maxiters::Integer = 1000, verbose = 0,
        callback = nothing
    )
    t₀ = time_ns()
    verbose = Int(verbose)
    builds₀ = preconditioner_builds(ws.preconditioner)
    ptc_start!(ws; verbose)
    tol = max(abstol, reltol * ws.norm_res_initial)
    status = :max_iterations
    if !isfinite(ws.norm_res)
        status = :nonfinite
    elseif ws.norm_res <= tol
        status = :converged
    end
    step = 0
    while status === :max_iterations && step < maxiters
        step += 1
        if ptc_step!(ws; verbose) === :cfl_too_small
            status = :cfl_too_small
            break
        end
        info = ws.history[end]
        if ws.norm_res <= tol
            status = :converged
        elseif callback !== nothing && callback(
                ws, (;
                    info.step, info.cfl, ws.norm_res, ws.norm_res_initial,
                    info.newton_iterations, info.krylov_iterations, info.rejections,
                )
            ) === true
            status = :terminated
        end
    end
    ws.status = status
    ptc_finish!(ws, builds₀, t₀; verbose)
    return ws.u, ws
end

"""
    ptc_start!(ws::PseudoTransientWorkspace; verbose = 0) -> norm_res

Start (or restart) the pseudo-transient continuation at the current state `ws.u`:
evaluate the steady residual, which becomes the reference residual of the SER, and
record it in the history. Followed by calls of [`ptc_step!`](@ref).
"""
function ptc_start!(ws::PseudoTransientWorkspace; verbose = 0)
    ws.t_start = time_ns()
    timings = ws.stats.timings
    for k in (:newton, :residual)
        timings[k] = get(timings, k, 0.0)
    end
    norm_res = evaluate_steady!(ws)
    ws.norm_res = norm_res
    ws.norm_res_initial = norm_res
    ws.norm_res_prior = norm_res
    ws.stats.norm_res_initial = norm_res
    ws.stats.norm_res = norm_res
    copyto!(ws.res_initial, ws.res)
    copyto!(ws.res_prior, ws.res)
    push!(
        ws.history, (;
            step = ws.nsteps, cfl = ws.cfl, norm_res, newton_iterations = 0,
            krylov_iterations = 0, rejections = 0, time = 0.0,
        )
    )
    ws.status = isfinite(norm_res) ? :running : :nonfinite
    Int(verbose) > 0 && @printf("PTC %5d  residual %.4e\n", ws.nsteps, norm_res)
    return norm_res
end

"""
    ptc_step!(ws::PseudoTransientWorkspace; verbose = 0) -> status

One accepted pseudo-time step of the pseudo-transient continuation (see
[`ptc_start!`](@ref)): inexact Newton iterations of the implicit Euler step, rejected and
repeated with a reduced CFL number until the step is accepted, followed by the CFL update.
Returns `:accepted`, with the steady residual `ws.res` and its norm `ws.norm_res` at
the new state, or `:cfl_too_small`, with the state and residual of the last accepted step.
The convergence test is left to the caller.
"""
function ptc_step!(ws::PseudoTransientWorkspace; verbose = 0)
    verbose = Int(verbose)
    (; alg, u, params, newton, preconditioner, stats, history) = ws
    timings = stats.timings
    N = preconditioner
    on_krylov_failure = alg.reject_krylov_failure ? :stop : :continue
    norm_res = ws.norm_res
    rejections = 0
    newton_its = 0
    krylov_its = 0
    set_dtau!(ws)
    while true
        params.u_n .= u
        t_newton = @elapsed begin
            _, result = newton_krylov!(
                newton; max_niter = alg.newton_iterations,
                tol_rel = alg.newton_reltol, tol_abs = 0.0, alg.forcing,
                linesearch! = alg.linesearch, N, alg.krylov_kwargs,
                on_krylov_failure, verbose = max(verbose - 2, 0)
            )
        end
        timings[:newton] = get(timings, :newton, 0.0) + t_newton
        newton_its += result.stats.outer_iterations
        krylov_its += result.stats.inner_iterations
        stats.newton_iterations += result.stats.outer_iterations
        stats.krylov_iterations += result.stats.inner_iterations
        stats.krylov_failures += result.stats.krylov_failures
        stats.residual_evaluations += result.stats.outer_iterations + 1

        reason = result.status
        norm_res_new = NaN
        if result.status in (:converged, :max_iterations)
            if alg.isadmissible !== nothing && !alg.isadmissible(u, ws.p)
                reason = :inadmissible
            else
                norm_res_new = evaluate_steady!(ws)
                if !isfinite(norm_res_new)
                    reason = :nonfinite
                elseif norm_res_new > alg.max_residual_growth * norm_res
                    reason = :residual_growth
                else
                    reason = :accepted
                end
            end
        end
        reason === :accepted && break

        u .= params.u_n
        # Reject the step, reduce the CFL number, and refresh the preconditioner
        rejections += 1
        stats.rejected_steps += 1
        ws.cfl = reject_cfl(alg.cfl, ws.cfl)
        verbose > 1 && @printf("    step rejected (%s), reducing CFL to %s\n", reason, ws.cfl)
        if cfl_too_small(alg.cfl, ws.cfl)
            # restore the steady residual of the last accepted state
            evaluate_steady!(ws)
            ws.status = :cfl_too_small
            return :cfl_too_small
        end
        set_dtau!(ws)
        refresh_preconditioner!(preconditioner)
    end
    norm_res_prior = norm_res
    norm_res = evaluate_steady_norm(ws)
    cfl_used = ws.cfl
    info = (;
        norm_res, norm_res_prior, ws.norm_res_initial,
        res = ws.res, res_prior = ws.res_prior, res_initial = ws.res_initial,
    )
    ws.cfl = update_cfl(alg.cfl, ws.cfl, info)
    copyto!(ws.res_prior, ws.res)
    ws.norm_res_prior = norm_res_prior
    ws.norm_res = norm_res
    ws.nsteps += 1
    stats.steps += 1
    stats.norm_res = norm_res
    push!(
        history, (;
            step = ws.nsteps, cfl = cfl_used, norm_res, newton_iterations = newton_its,
            krylov_iterations = krylov_its, rejections,
            time = (time_ns() - ws.t_start) / 1.0e9,
        )
    )
    verbose > 0 && @printf(
        "PTC %5d  residual %.4e  (rel %.3e)  CFL %s  Newton %d  Krylov %4d\n",
        ws.nsteps, norm_res, norm_res / ws.norm_res_initial, _show_cfl(cfl_used), newton_its, krylov_its
    )
    return :accepted
end

"""
    ptc_reset_reference!(ws::PseudoTransientWorkspace)

Make the current steady residual the reference residual of the SER (e.g., after the state
was changed outside of [`ptc_step!`](@ref)), without resetting the CFL number.
"""
function ptc_reset_reference!(ws::PseudoTransientWorkspace)
    norm_res = evaluate_steady!(ws)
    ws.norm_res = norm_res
    ws.norm_res_prior = norm_res
    copyto!(ws.res_prior, ws.res)
    return norm_res
end

function ptc_finish!(ws::PseudoTransientWorkspace, builds₀, t₀; verbose = 0)
    (; stats, preconditioner) = ws
    timings = stats.timings
    stats.preconditioner_builds += preconditioner_builds(preconditioner) - builds₀
    timings[:total] = get(timings, :total, 0.0) + (time_ns() - t₀) / 1.0e9
    Int(verbose) > 0 && @printf(
        "PTC %s: %d steps (%d rejected), %d Newton, %d Krylov iterations, %d preconditioner builds, %.2f s\n",
        ws.status, stats.steps, stats.rejected_steps, stats.newton_iterations,
        stats.krylov_iterations, stats.preconditioner_builds, timings[:total]
    )
    return nothing
end

# The steady residual was evaluated for the acceptance test
evaluate_steady_norm(ws::PseudoTransientWorkspace) = ws.newton.norm(ws.res)

_show_cfl(c::Number) = @sprintf("%.3e", c)
_show_cfl(c::NamedTuple) = join((@sprintf("%s=%.3e", k, v) for (k, v) in pairs(c)), ",")

"""
    Asterion.successful(ws::PseudoTransientWorkspace)

Whether the pseudo-transient continuation converged (or was terminated by the callback).
"""
successful(ws::PseudoTransientWorkspace) = ws.status === :converged || ws.status === :terminated
