##
# Line search with admissibility of the states
##

# Parameters of the user: for the pseudo-transient residual, the parameters of the user
# inside its parameters (see `PseudoTransientParameters`)
user_parameters(p) = p

"""
    AdmissibleBacktrackingLineSearch(isadmissible = (u, p) -> true; max_step = nothing,
                                     armijo = true, alpha = 1.0e-4, factor = 0.5,
                                     n_iter_max = 10)

Backtracking line search that only accepts physically admissible states, e.g., states
with positive density and pressure. Starting from the step length `λ = 1` (or
`λ = min(1, max_step(u, d, p))`), the step length is multiplied by `factor` until the
new state `u + λ d`
1. is admissible, i.e., `isadmissible(u + λ d, p)` returns `true`,
2. has a finite residual norm, and
3. if `armijo == true`, satisfies the Armijo condition
   `‖F(u + λ d)‖ <= (1 - alpha λ) ‖F(u)‖`.

`p` are the parameters of the problem (of the user, also inside the parameters of a
pseudo-transient residual, see `Asterion.user_parameters`).

The optional hook `max_step(u, d, p)` returns the largest step length allowed by the
state `u` and the direction `d`, e.g., to limit the relative change of the density and
pressure per Newton step (solution update limiting).

If no admissible state with finite residual is found within `n_iter_max` trials, `u` is
reset to the state before the step and the line search returns `Inf`, so that
[`newton_krylov!`](@ref Ariadne.newton_krylov!) stops with status `:nonfinite`. If an admissible state with finite
residual is found but the Armijo condition is not satisfied within `n_iter_max` trials,
the last trial step is taken (as in [`BacktrackingLineSearch`](@ref Ariadne.BacktrackingLineSearch)).
"""
struct AdmissibleBacktrackingLineSearch{A, S} <: Ariadne.LineSearches.AbstractLineSearch
    isadmissible::A
    max_step::S
    armijo::Bool
    alpha::Float64
    factor::Float64
    n_iter_max::Int
end

function AdmissibleBacktrackingLineSearch(
        isadmissible = (u, p) -> true; max_step = nothing, armijo::Bool = true,
        alpha = 1.0e-4, factor = 0.5, n_iter_max::Integer = 10
    )
    @assert n_iter_max > 0 "n_iter_max must be positive"
    @assert 0 < factor < 1 "factor must be in (0, 1)"
    return AdmissibleBacktrackingLineSearch(
        isadmissible, max_step, armijo, Float64(alpha), Float64(factor), Int(n_iter_max)
    )
end

function (ls::AdmissibleBacktrackingLineSearch)(ws, norm_res_prior, d)
    p = user_parameters(ws.p)
    λ = 1.0
    if ls.max_step !== nothing
        λ = clamp(Float64(ls.max_step(ws.u, d, p)), 0.0, 1.0)
    end
    ws.u .= muladd.(λ, d, ws.u)
    norm_res = oftype(norm_res_prior, Inf)
    last_ok = false # the current trial state is admissible with finite residual
    for k in 1:ls.n_iter_max
        last_ok = false
        if ls.isadmissible(ws.u, p)
            norm_res = Ariadne.evaluate!(ws)
            if isfinite(norm_res)
                last_ok = true
                if !ls.armijo || norm_res <= (1 - ls.alpha * λ) * norm_res_prior
                    return norm_res
                end
            end
        end
        k == ls.n_iter_max && break
        new_λ = ls.factor * λ
        ws.u .= muladd.(new_λ - λ, d, ws.u)
        λ = new_λ
    end
    if last_ok
        return norm_res
    end
    # Reset to the state before the step
    ws.u .= muladd.(-λ, d, ws.u)
    Ariadne.evaluate!(ws)
    return oftype(norm_res_prior, Inf)
end
