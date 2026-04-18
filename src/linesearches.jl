module LineSearches

using LinearAlgebra

"""
    AbstractLineSearch

Line search may update the solution `u` and the residual `res` in-place,
given the function `F!`, parameters `p`, and the Newton direction `d`.

They must call `F!(res, u, p)` to update the residual after updating `u`.
"""
abstract type AbstractLineSearch end

"""
    NoLineSearch()

A line search that does not perform any line search: it simply takes the full Newton step.
"""
struct NoLineSearch <: AbstractLineSearch end

function (::NoLineSearch)(F!, res, n_res_prior, u, p, d)
    # No line search: take the full Newton step
    u .+= d
    F!(res, u, p)
    return norm(res)
end

"""
    BacktrackingLineSearch(; n_iter_max = 10, parabolic = false)

## References

- Kelley, C. T. (2022).
  Solving nonlinear equations with iterative methods:
  Solvers and examples in Julia.
  Society for Industrial and Applied Mathematics.
- <https://github.com/ctkelley/SIAMFANLEquations.jl>
"""
Base.@kwdef struct BacktrackingLineSearch <: AbstractLineSearch
    n_iter_max::Int = 10
    parabolic::Bool = false
end

function (ls::BacktrackingLineSearch)(F!, res, n_res_prior, u, p, d)
    alpha = 1.0e-4
    lambda = 1.0

    @assert ls.n_iter_max > 0 "n_iter_max must be positive"
    @assert alpha > 0 "alpha must be positive"

    n_res = Inf
    u_trial = similar(u)

    for _ in 1:ls.n_iter_max
        # Take a step of size s
        u_trial .= muladd.(lambda, d, u) # u = u + lambda * d
        F!(res, u_trial, p)
        n_res = norm(res)

        # Armijo condition
        if n_res <= (1 - alpha * lambda) * n_res_prior
            u .= u_trial
            return n_res
        end

        lambda *= 0.5
    end
    u .= u_trial
    return n_res
end

# SIAMFANL is using a parabolic line search
# https://github.com/ctkelley/SIAMFANLEquations.jl/blob/e5603e177dd007b065265641fb232d54020c4282/src/Tools/armijo.jl#L57

end # module LineSearches
