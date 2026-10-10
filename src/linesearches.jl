module LineSearches

using LinearAlgebra
import ..evaluate!, ..user_parameters

"""
    AbstractLineSearch

Line search updates `ws.u` in-place along the Newton direction `d` and calls
`evaluate!(ws)` to refresh `ws.res` and obtain the new residual norm.
It returns `(norm_res, status)`: the residual norm of the new state and
`status = :success`, or `:failed` if it did not find a step with sufficient decrease
(and took its last trial step), which [`newton_krylov!`](@ref Ariadne.newton_krylov!) counts in
`stats.linesearch_failures`.

## Implemented variants
- [`NoLineSearch`](@ref)
- [`BacktrackingLineSearch`](@ref)
- [`AdmissibleLineSearch`](@ref), which restricts another line search to admissible states

## Custom line searches
```julia
struct CustomLineSearch <: AbstractLineSearch
    # parameters for the line search
end

function (ls::CustomLineSearch)(ws, norm_res_prior, d; verbose = 0)
    # update ws.u
    ws.u .+= d # for example, take the full Newton step
    return evaluate!(ws), :success
end
```

A line search is called as `ls(ws, norm_res_prior, d; verbose)` and must accept the keyword
argument `verbose`, the verbosity level of [`newton_krylov!`](@ref Ariadne.newton_krylov!).

A line search should report the step length `λ` of the state it returns, `u + λ d`, with
`Ariadne.LineSearches.record_step_length!(ws, λ)` (a no-op for a `NewtonKrylovWorkspace`);
[`AdmissibleLineSearch`](@ref) uses it to undo failed steps without a copy of `u`.
"""
abstract type AbstractLineSearch end

# Whether the exception `err` is of one of the `types`, also if it was thrown in tasks:
# a `TaskFailedException` (e.g., from `fetch`) matches if the exception of its task matches,
# a `CompositeException` (e.g., from a `Threads.@threads` loop in the residual) matches if
# all of its exceptions match, and a `CapturedException` matches if its exception matches.
function matches_exception(err, types::Tuple)
    any(T -> err isa T, types) && return true
    if err isa CompositeException
        return !isempty(err.exceptions) && all(e -> matches_exception(e, types), err.exceptions)
    elseif err isa TaskFailedException
        return matches_exception(err.task.result, types)
    elseif err isa CapturedException
        return matches_exception(err.ex, types)
    end
    return false
end

# `evaluate!(ws)`, but `Inf` if an exception of one of the `types` is thrown, e.g., a
# `DomainError` from `sqrt` or `log` of a negative quantity in a trial state. The exception
# is logged with `@info` if `verbose > 0`, and with `@debug` otherwise.
function evaluate_or_inf!(ws, types::Tuple; verbose = 0)
    isempty(types) && return evaluate!(ws)
    try
        return evaluate!(ws)
    catch err
        matches_exception(err, types) || rethrow()
        msg = "Line search: the residual of the trial state threw an exception, treating it as an infinite residual"
        if verbose > 0
            @info msg exception = (err, catch_backtrace())
        else
            @debug msg exception = (err, catch_backtrace())
        end
        return Inf
    end
end

"""
    parabolic_step(λc, λm, ff0, ffc, ffm; σ₀ = 0.1, σ₁ = 0.5)

Safeguarded three-point parabolic model for the step length of a line search, as in
[Kelley2022](@cite) (`parab3p` of SIAMFANLEquations.jl): minimize the parabola through the
squared residual norms `ff0` at `λ = 0`, `ffc` at the current step length `λc`, and `ffm` at
the previous step length `λm`, and clamp the result to `[σ₀ λc, σ₁ λc]`.

The model is `p(λ) = ff0 + (c₁ λ + c₂ λ²) / d₁` with `d₁ = (λc - λm) λc λm < 0`, so it is
convex if `c₂ < 0`, and its minimum is at `λ = -c₁ / (2 c₂)`. If the parabola has
negative curvature, the model is not helpful and the smallest step length `σ₀ λc` is taken
(the corrected behavior of `parab3p`, see the comments there). If a residual norm is not
finite, e.g., because the trial state threw an exception, `λc` is halved (`σ₁ λc`).

Adapted from <https://github.com/ctkelley/SIAMFANLEquations.jl/blob/e5603e177dd007b065265641fb232d54020c4282/src/Tools/armijo.jl#L57>
(MIT license).
"""
function parabolic_step(λc, λm, ff0, ffc, ffm; σ₀ = 0.1, σ₁ = 0.5)
    (isfinite(ffc) && isfinite(ffm)) || return σ₁ * λc
    c2 = λm * (ffc - ff0) - λc * (ffm - ff0)
    # Negative curvature
    c2 >= 0 && return σ₀ * λc
    c1 = λc^2 * (ffm - ff0) - λm^2 * (ffc - ff0)
    λp = -c1 / (2 * c2)
    return clamp(λp, σ₀ * λc, σ₁ * λc)
end

# Report the step length `λ` of the state returned by a line search, `u + λ d`
record_step_length!(ws, λ) = nothing

"""
    NoLineSearch()

A line search that does not perform any line search: it simply takes the full Newton step.
"""
struct NoLineSearch <: AbstractLineSearch end

function (::NoLineSearch)(ws, norm_res_prior, d; verbose = 0)
    ws.u .+= d
    record_step_length!(ws, 1.0)
    return evaluate!(ws), :success
end

"""
    BacktrackingLineSearch(; n_iter_max = 10, alpha = 1.0e-4,
                           reject_exceptions = (DomainError,), parabolic = false)

Armijo backtracking: the step length `λ`, starting from `1`, is reduced until
`‖F(u + λ d)‖ <= (1 - alpha λ) ‖F(u)‖`, for at most `n_iter_max` trials. If no trial
satisfies this condition, the last trial step is taken and the line search counts as
failed (`stats.linesearch_failures` of [`newton_krylov!`](@ref Ariadne.newton_krylov!)).

By default, `λ` is halved in each reduction. With `parabolic = true`, the first reduction
halves `λ` and later reductions use the safeguarded three-point parabolic model of
[Kelley2022](@cite) ([`Ariadne.LineSearches.parabolic_step`](@ref)), which reduces `λ` by a
factor in `[0.1, 0.5]`. The parabolic model fails the Armijo condition less often, but
does not reduce the number of Newton iterations on, e.g., the generalized Rosenbrock
problem.

Trial states whose residual evaluation throws an exception of one of the types
`reject_exceptions` (e.g., a `DomainError` from `sqrt` or `log` of a quantity that became
negative in a too long step) count as trial states with infinite residual norm, so the
step length is reduced further. This also holds for such exceptions thrown in tasks, e.g.,
in a `Threads.@threads` loop of the residual. The caught exceptions are logged with
`@info` if [`newton_krylov!`](@ref Ariadne.newton_krylov!) is called with `verbose > 0`, and with `@debug`
otherwise. Other exceptions are rethrown. If all trials
throw, the line search returns `Inf` and [`newton_krylov!`](@ref Ariadne.newton_krylov!) stops with status
`:nonfinite`. Use `reject_exceptions = ()` to rethrow all exceptions.

## References

- Kelley, C. T. (2022).
  Solving nonlinear equations with iterative methods:
  Solvers and examples in Julia.
  Society for Industrial and Applied Mathematics.
- <https://github.com/ctkelley/SIAMFANLEquations.jl>
"""
Base.@kwdef struct BacktrackingLineSearch <: AbstractLineSearch
    n_iter_max::Int = 10
    alpha::Float64 = 1.0e-4
    reject_exceptions::Tuple = (DomainError,)
    parabolic::Bool = false
end

function (ls::BacktrackingLineSearch)(ws, norm_res_prior, d; verbose = 0)
    alpha = ls.alpha
    lambda = 1.0

    @assert ls.n_iter_max > 0 "n_iter_max must be positive and larger than 0"
    @assert alpha > 0 "alpha must be positive"

    # Take the full Newton step (lambda = 1.0)
    ws.u .= muladd.(lambda, d, ws.u) # u = u + lambda * d
    norm_res = evaluate_or_inf!(ws, ls.reject_exceptions; verbose)

    # Squared residual norms at λ = 0, at the current and at the previous step length
    ff0 = norm_res_prior^2
    ffc = norm_res^2
    lambda_m = lambda
    ffm = ffc

    for iter in 2:ls.n_iter_max
        # Armijo condition
        if norm_res <= (1 - alpha * lambda) * norm_res_prior
            record_step_length!(ws, lambda)
            return norm_res, :success
        end

        if iter == 2 || !ls.parabolic
            new_lambda = lambda * 0.5
        else
            new_lambda = parabolic_step(lambda, lambda_m, ff0, ffc, ffm)
        end
        # Retract the excess step incrementally:
        # u goes from u + old_lambda*d to u + new_lambda*d,
        # so the adjustment is (new_lambda - old_lambda)*d (negative).
        s = new_lambda - lambda
        ws.u .= muladd.(s, d, ws.u) # u = u + (new_lambda - old_lambda) * d
        lambda_m, ffm = lambda, ffc
        lambda = new_lambda
        norm_res = evaluate_or_inf!(ws, ls.reject_exceptions; verbose)
        ffc = norm_res^2
    end
    status = norm_res <= (1 - alpha * lambda) * norm_res_prior ? :success : :failed
    record_step_length!(ws, lambda)
    return norm_res, status
end

"""
    AdmissibleLineSearch(isadmissible, linesearch = BacktrackingLineSearch(); max_step = nothing)

Restrict the line search `linesearch` to admissible states, e.g., states with positive
density and pressure: a trial state `u` with `isadmissible(u, p) == false` counts as a
trial state with infinite residual norm (without evaluating the residual), in the same
way as a trial state whose residual throws a `DomainError` in
[`BacktrackingLineSearch`](@ref). A backtracking line search then reduces the step length
until the state is admissible.

`p` are the parameters of the problem, `Ariadne.user_parameters(ws.p)` (solvers that wrap
the parameters of the user, e.g., pseudo-transient continuation, extend `user_parameters`).

The optional hook `max_step(u, d, p)` returns the largest step length allowed by the state
`u` and the Newton direction `d`, e.g., to limit the relative change of the density and
pressure per Newton step (solution update limiting); `d` is scaled by it (in place) before
`linesearch` is called.

If `linesearch` does not find an admissible state with finite residual, `u` is reset to the
state before the step (by undoing the step `λ d` that `linesearch` reports with
`Ariadne.LineSearches.record_step_length!`, so no copy of `u` is needed) and the line
search returns `(Inf, :failed)`, so that
[`newton_krylov!`](@ref Ariadne.newton_krylov!) stops with status `:nonfinite`.

## Examples

```julia
positive(u, p) = all(>(0), u)
AdmissibleLineSearch(positive) # backtracking until u > 0 and the Armijo condition hold
AdmissibleLineSearch(positive, NoLineSearch(); max_step) # limited full steps only
```
"""
struct AdmissibleLineSearch{A, L <: AbstractLineSearch, S} <: AbstractLineSearch
    isadmissible::A
    linesearch::L
    max_step::S
    step_length::Base.RefValue{Float64} # reported by `linesearch`
end

function AdmissibleLineSearch(isadmissible, linesearch::AbstractLineSearch = BacktrackingLineSearch(); max_step = nothing)
    return AdmissibleLineSearch(isadmissible, linesearch, max_step, Ref(NaN))
end

# The workspace seen by the inner line search: `evaluate!` returns `Inf` for inadmissible
# states and evaluates the residual otherwise, and it records the step length
struct AdmissibleWorkspace{W, A, P}
    ws::W
    isadmissible::A
    user_p::P # `user_parameters(ws.p)`, passed to `isadmissible`
    step_length::Base.RefValue{Float64}
end

# The properties of the workspace that line searches use
function Base.getproperty(w::AdmissibleWorkspace, s::Symbol)
    if s === :u || s === :res || s === :p
        return getproperty(getfield(w, :ws), s)
    end
    return getfield(w, s)
end

record_step_length!(w::AdmissibleWorkspace, λ) = (w.step_length[] = λ; nothing)

function evaluate!(w::AdmissibleWorkspace)
    if w.isadmissible(w.u, w.user_p)
        return evaluate!(w.ws)
    else
        return float(real(eltype(w.u)))(Inf)
    end
end

function (ls::AdmissibleLineSearch)(ws, norm_res_prior, d; verbose = 0)
    aws = AdmissibleWorkspace(ws, ls.isadmissible, user_parameters(ws.p), ls.step_length)
    if ls.max_step !== nothing
        λ = clamp(ls.max_step(ws.u, d, aws.user_p), 0, 1)
        λ < 1 && (d .*= λ)
    end
    ls.step_length[] = NaN
    norm_res, status = ls.linesearch(aws, norm_res_prior, d; verbose)
    λ = ls.step_length[]
    record_step_length!(ws, λ)
    if !isfinite(norm_res)
        # No admissible state with finite residual: undo the step `u + λ d` (if the inner
        # line search reported `λ`)
        if !isnan(λ)
            ws.u .= muladd.(-λ, d, ws.u)
            evaluate!(ws)
        end
        return oftype(norm_res_prior, Inf), :failed
    end
    return norm_res, status
end

end # module LineSearches
