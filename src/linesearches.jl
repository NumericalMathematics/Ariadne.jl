module LineSearches

using LinearAlgebra

"""
    AbstractLineSearch

Line search may update the solution `u` and the residual `res` in-place,
given the function `F!`, parameters `p`, and the Newton direction `d`.

They must call `F!(res, u, p)` to update the residual after updating `u`.

```julia
struct NewLineSearch <: AbstractLineSearch
    # parameters for the line search
end

function (ls::NewLineSearch)(J::AbstractJacobianOperator, F!, res, norm_res_prior, u, p, d)
    # perform line search to find an appropriate step size
    # ...
    # update u and res in-place
    F!(res, u, p)
    return norm(res)
end
```
"""
abstract type AbstractLineSearch end

"""
    NoLineSearch()

A line search that does not perform any line search: it simply takes the full Newton step.
"""
struct NoLineSearch <: AbstractLineSearch end

function (::NoLineSearch)(J, F!, res, norm_res_prior, u, p, d)
    # No line search: take the full Newton step
    u .+= d
    F!(res, u, p)
    return norm(res)
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
end

function (ls::BacktrackingLineSearch)(J, F!, res, norm_res_prior, u, p, d)
    alpha = 1.0e-4
    lambda = 1.0

    @assert ls.n_iter_max > 0 "n_iter_max must be positive"
    @assert alpha > 0 "alpha must be positive"

    # Take a step of size s in the direction d
    u .= muladd.(lambda, d, u) # u = u + lambda * d
    F!(res, u, p)
    norm_res = norm(res)

    previous_lambda = lambda
    for _ in 2:ls.n_iter_max
        # Armijo condition
        if norm_res <= (1 - alpha * lambda) * norm_res_prior
            return norm_res
        end

        s = previous_lambda - lambda
        # Take a step of size s in the direction d
        u .= muladd.(s, d, u) # u = u + s * d
        F!(res, u, p)
        norm_res = norm(res)

        previous_lambda = lambda
        lambda *= 0.5
    end
    return norm_res
end

end # module LineSearches
