module LineSearches

using LinearAlgebra
import ..evaluate!

"""
    AbstractLineSearch

Line search updates `ws.u` in-place along the Newton direction `d` and calls
`evaluate!(ws)` to refresh `ws.res` and obtain the new residual norm.

## Implemented variants
- [`NoLineSearch`](@ref)
- [`BacktrackingLineSearch`](@ref)

## Custom line searches
```julia
struct CustomLineSearch <: AbstractLineSearch
    # parameters for the line search
end

function (ls::CustomLineSearch)(ws, norm_res_prior, d; verbose = 0)
    # update ws.u
    ws.u .+= d # for example, take the full Newton step
    return evaluate!(ws)
end
```

A line search is called as `ls(ws, norm_res_prior, d; verbose)` and must accept the keyword
argument `verbose`, the verbosity level of [`newton_krylov!`](@ref Ariadne.newton_krylov!).
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
    NoLineSearch()

A line search that does not perform any line search: it simply takes the full Newton step.
"""
struct NoLineSearch <: AbstractLineSearch end

function (::NoLineSearch)(ws, norm_res_prior, d; verbose = 0)
    ws.u .+= d
    return evaluate!(ws)
end

"""
    BacktrackingLineSearch(; n_iter_max = 10, alpha = 1.0e-4,
                           reject_exceptions = (DomainError,))

Armijo backtracking: the step length is halved until
`‖F(u + λ d)‖ <= (1 - alpha λ) ‖F(u)‖`, for at most `n_iter_max` trials, after which the
last trial step is taken.

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
end

function (ls::BacktrackingLineSearch)(ws, norm_res_prior, d; verbose = 0)
    alpha = ls.alpha
    lambda = 1.0

    @assert ls.n_iter_max > 0 "n_iter_max must be positive and larger than 0"
    @assert alpha > 0 "alpha must be positive"

    # Take the full Newton step (lambda = 1.0)
    ws.u .= muladd.(lambda, d, ws.u) # u = u + lambda * d
    norm_res = evaluate_or_inf!(ws, ls.reject_exceptions; verbose)

    for _ in 2:ls.n_iter_max
        # Armijo condition
        if norm_res <= (1 - alpha * lambda) * norm_res_prior
            return norm_res
        end

        # Halve lambda and retract the excess step incrementally:
        # u goes from u + old_lambda*d to u + new_lambda*d,
        # so the adjustment is (new_lambda - old_lambda)*d (negative).
        new_lambda = lambda * 0.5
        s = new_lambda - lambda
        ws.u .= muladd.(s, d, ws.u) # u = u + (new_lambda - old_lambda) * d
        lambda = new_lambda
        norm_res = evaluate_or_inf!(ws, ls.reject_exceptions; verbose)
    end
    return norm_res
end

end # module LineSearches
