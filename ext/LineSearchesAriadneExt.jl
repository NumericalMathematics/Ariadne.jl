module LineSearchesAriadneExt

import Ariadne.LineSearches: LineSearches_JL
using LineSearches
using LinearAlgebra

using Enzyme: autodiff, Forward, ForwardWithPrimal, Duplicated, make_zero


function (ls::LineSearches_JL)(F!, res, n_res_prior, u, p, d)
    function ϕ(α)
        u_trial = muladd.(α, d, u) # u_trial = u + α * d
        F!(res, u_trial, p)
        return norm(res)
    end

    function dϕ(α)
        dr, = autodiff(Forward, Duplicated(ϕ, make_zero(ϕ)), Duplicated(α, one(α)))
        return dr
    end

    function ϕdϕ(α)
        dr, r = autodiff(ForwardWithPrimal, Duplicated(ϕ, make_zero(ϕ)), Duplicated(α, one(α)))
        return r, dr
    end

    # ϕ0 = ϕ(0.0) # ϕ(0) is the norm of the residual at the current point, i.e., n_res_prior
    dϕ0 = dϕ(0.0) # can we get this for free?

    α, fx = ls.linesearch(ϕ, dϕ, ϕdϕ, 1.0, n_res_prior, dϕ0)
    u .= muladd.(α, d, u)
    return fx
end

end # module LineSearchesAriadneExt
