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

function (ls::CustomLineSearch)(ws, norm_res_prior, d)
    # update ws.u
    ws.u .+= d # for example, take the full Newton step
    return evaluate!(ws)
end
```
"""
abstract type AbstractLineSearch end

"""
    NoLineSearch()

A line search that does not perform any line search: it simply takes the full Newton step.
"""
struct NoLineSearch <: AbstractLineSearch end

function (::NoLineSearch)(ws, _, d)
    ws.u .+= d
    return evaluate!(ws)
end

"""
    BacktrackingLineSearch(; n_iter_max = 10)

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
end

function (ls::BacktrackingLineSearch)(ws, norm_res_prior, d)
    alpha = ls.alpha
    lambda = 1.0

    @assert ls.n_iter_max > 0 "n_iter_max must be positive and larger than 0"
    @assert alpha > 0 "alpha must be positive"

    # Take the full Newton step (lambda = 1.0)
    ws.u .= muladd.(lambda, d, ws.u) # u = u + lambda * d
    norm_res = evaluate!(ws)

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
        norm_res = evaluate!(ws)
    end
    return norm_res
end

struct LineSearches_JL{T<:Any} <: AbstractLineSearch
    linesearch::T
end

end # module LineSearches
