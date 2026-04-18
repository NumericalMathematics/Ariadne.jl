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

function (::NoLineSearch)(F!, res, _, u, p, d)
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

    ff0 = n_res_prior^2
    ffc = ff0
    ffm = ffc
    lamc = lambda

    for i in 1:ls.n_iter_max
        # Take a step of size s
        u_trial .= muladd.(lambda, d, u) # u = u + lambda * d
        F!(res, u_trial, p)
        n_res = norm(res)
        @info "Line search iteration $i: lambda = $lambda, residual norm = $n_res, previous residual norm = $n_res_prior"

        # Armijo condition
        if n_res <= (1 - alpha * lambda) * n_res_prior
            u .= u_trial
            return n_res
        end

        ffm = ffc
        ffc = n_res^2
        lambda = update_lambda(i, ls.parabolic, lambda, lamc, ff0, ffc, ffm)
    end
    u .= u_trial
    return n_res
end

function update_lambda(i, parabolic, lambda, lamc, ff0, ffc, ffm)
    if !parabolic
        return lambda * 0.5
    end
    if i == 1
        return lambda * 0.5
    else
        return parab3p(lambda, lamc, ff0, ffc, ffm)
    end
end

# From https://github.com/ctkelley/SIAMFANLEquations.jl/blob/e5603e177dd007b065265641fb232d54020c4282/src/Tools/armijo.jl#L57
"""
parab3p(lambdac, lambdam, ff0, ffc, ffm)

Three point parabolic line search.

input:\n
       lambdac = current steplength
       lambdam = previous steplength
       ff0 = value of || F(x_c) ||^2
       ffc = value of || F(x_c + lambdac d) ||^2
       ffm = value of || F(x_c + lambdam d) ||^2

output:\n
       lambdap = new value of lambda

internal parameters:\n
       sigma0 = .1, sigma1=.5, safeguarding bounds for the linesearch

You get here if cutting the steplength in half doesn't get you
sufficient decrease. Now you have three points and can build a parabolic
model. I do not like cubic models because they either need four points
or a derivative. 

So let's think about how this works. I cheat a bit and check the model
for negative curvature, which I don't want to see.

 The polynomial is

 p(lambda) = ff0 + (c1 lambda + c2 lambda^2)/d1

 d1 = (lambdac - lambdam)*lambdac*lambdam < 0
 So if c2 > 0 we have negative curvature and default to
      lambdap = sigma0 * lambda
 The logic is that negative curvature is telling us that
 the polynomial model is not helping much, so it looks better
 to take the smallest possible step. This is not what I did in the
 matlab code because I did it wrong. I have sinced fixed it.

 So (Students, listen up!) if c2 < 0 then all we gotta do is minimize
 (c1 lambda + c2 lambda^2)/d1 over [.1* lambdac, .5*lambdac]
 This means to MAXIMIZE c1 lambda + c2 lambda^2 becase d1 < 0.
 So I find the zero of the derivative and check the endpoints.

"""
function parab3p(lambdac, lambdam, ff0, ffc, ffm)
    #
    # internal parameters
    #
    sigma0 = 0.1
    sigma1 = 0.5
    #
    c2 = lambdam * (ffc - ff0) - lambdac * (ffm - ff0)
    if c2 >= 0
        #
        # Sanity check for negative curvature
        #
        lambdap = sigma0 * lambdac
    else
        #
        # It's a convex parabola, so use calculus!
        #
        c1 = lambdac * lambdac * (ffm - ff0) - lambdam * lambdam * (ffc - ff0)
        lambdap = -c1 * 0.5 / c2
        #
        lambdaup = sigma1 * lambdac
        lambdadown = sigma0 * lambdac
        lambdap = max(lambdadown, min(lambdaup, lambdap))
    end
end

end # module LineSearches
