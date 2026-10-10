# # [Implicit schemes](@id implicit_schemes)
using Ariadne

# ## Implicit Euler

function G_Euler!(res, uₙ, Δt, f!, du, u, p, t)
    f!(du, u, p, t)

    res .= uₙ .+ Δt .* du .- u
    return nothing
end

# ## Implicit Midpoint

function G_Midpoint!(res, uₙ, Δt, f!, du, u, p, t; α = 0.5)
    ## Use res for a temporary allocation (uₙ .+ u) ./ 2
    uuₙ = res
    uuₙ .= (α .* uₙ .+ (1 - α) .* u)
    f!(du, uuₙ, p, t + α * Δt)

    res .= uₙ .+ Δt .* du .- u
    return nothing
end

# ## Implicit Trapezoid

function G_Trapezoid!(res, uₙ, Δt, f!, du, u, p, t)
    ## Use res as the temporary
    duₙ = res
    f!(duₙ, uₙ, p, t)
    f!(du, u, p, t + Δt)

    res .= uₙ .+ (Δt / 2) .* (duₙ .+ du) .- u
    return nothing
end

# ## Jacobian of various G

function jacobian(G!, f!, uₙ, p, Δt, t)
    u = copy(uₙ)
    du = zero(uₙ)
    res = zero(uₙ)

    F!(res, u, (uₙ, Δt, du, p, t)) = G!(res, uₙ, Δt, f!, du, u, p, t)

    J = Ariadne.JacobianOperator(F!, res, u, (uₙ, Δt, du, p, t))
    return collect(J)
end

# ## Non-adaptive time stepping

function solve(
        G!, f!, uₙ, p, Δt, ts; callback = _ -> nothing,
        verbose = 0, algo = :gmres, krylov_kwargs = (;)
    )
    u = copy(uₙ)
    du = zero(uₙ)
    res = zero(uₙ)
    F!(res, u, (uₙ, Δt, du, p, t)) = G!(res, uₙ, Δt, f!, du, u, p, t)

    for t in ts
        if t == first(ts)
            continue
        end
        _, stats = newton_krylov!(
            F!, u, (uₙ, Δt, du, p, t), res;
            verbose, algo, tol_abs = 6.0e-6, krylov_kwargs
        )
        if !stats.solved
            @warn "non linear solve failed marching on" t stats
        end
        callback(u)
        uₙ .= u
    end
    return uₙ
end
