module LineSearchesAriadneExt

import Ariadne.LineSearches: LineSearches_JL
using LineSearches
using LinearAlgebra

import Ariadne: evaluate!

function (ls::LineSearches_JL)(ws, norm_res_prior, d)
    # TODO: avoid allocations in the line by providing a workspace for the line search
    u₀ = copy(ws.u)
    jvp = similar(ws.res)

    # LineSearches.jl operates on a scalar objective ϕ(α) and its derivative dϕ(α).
    # We use ϕ(α) = ‖F(u₀ + α·d)‖²/2 so that dϕ = dot(F, J·d) without having to calculate
    # norm(ws.res). Inspired by the implementation in LineSearch.jl
    function ϕ(α)
        ws.u .= muladd.(α, d, u₀) # u = u₀ + α * d
        fx = evaluate!(ws)
        return fx^2 / 2
    end

    function dϕ(α)
        ws.u .= muladd.(α, d, u₀) # u = u₀ + α * d
        mul!(jvp, ws.J, d)         # sets ws.res = F(ws.u) as primal, jvp = J(ws.u)*d
        return dot(ws.res, jvp)
    end

    function ϕdϕ(α)
        ws.u .= muladd.(α, d, u₀) # u = u₀ + α * d
        mul!(jvp, ws.J, d)        # sets ws.res = F(ws.u) as primal, jvp = J(ws.u)*d
        fx = norm(ws.res)
        return fx^2 / 2, dot(ws.res, jvp)
    end

    ϕ₀ = norm_res_prior^2 / 2
    dϕ₀ = dϕ(0.0)

    dϕ₀ ≥ 0 && return oftype(norm_res_prior, Inf) # d is not a descent direction; take no step

    α, fx = ls.linesearch(ϕ, dϕ, ϕdϕ, 1.0, ϕ₀, dϕ₀)
    ws.u .= muladd.(α, d, u₀) # ensure ws.u is at the final step
    return sqrt(2 * fx)
end

end # module LineSearchesAriadneExt
