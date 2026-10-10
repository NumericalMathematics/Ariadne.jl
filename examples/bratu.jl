# # 1D bratu equation from (Kan2022-ko)[@cite]

# ## Necessary packages
using Ariadne, Krylov
using KrylovPreconditioners
using SparseArrays, LinearAlgebra
using CairoMakie

# ## 1D Bratu equations
# $y′′ + λ * exp(y) = 0$

# F(u) = 0

function bratu!(res, y, (Δx, λ))
    N = length(y)
    for i in 1:N
        y_l = i == 1 ? zero(eltype(y)) : y[i - 1]
        y_r = i == N ? zero(eltype(y)) : y[i + 1]
        y′′ = (y_r - 2y[i] + y_l) / Δx^2

        res[i] = y′′ + λ * exp(y[i]) # = 0
    end
    return nothing
end

function bratu(y, p)
    res = similar(y)
    bratu!(res, y, p)
    return res
end

# ## Reference solution
function true_sol_bratu(x)
    ## for λ = 3.51382, 2nd sol θ = 4.8057
    θ = 4.79173
    return -2 * log(cosh(θ * (x - 0.5) / 2) / (cosh(θ / 4)))
end

# ## Choice of parameters
const N = 10_000
const λ = 3.51382
const dx = 1 / (N + 1) # Grid-spacing

# ### Domain and Initial condition
x = LinRange(0.0 + dx, 1.0 - dx, N)
u₀ = sin.(x .* π)

lines(x, u₀, label = "Initial guess")

# ## Reference solution evaluated over domain
reference = true_sol_bratu.(x)

fig, ax = lines(x, u₀, label = "Initial guess")
lines!(ax, x, reference, label = "Reference solution")
axislegend(ax, position = :cb)
fig

# ## Solving using inplace variant and CG
uₖ, _ = newton_krylov!(
    bratu!,
    copy(u₀), (dx, λ), similar(u₀);
    algo = :cg,
)

ϵ = abs2.(uₖ .- reference)

let
    fig = Figure(size = (800, 800))
    ax = Axis(fig[1, 1], title = "", ylabel = "", xlabel = "")

    lines!(ax, x, reference, label = "True solution")
    lines!(ax, x, u₀, label = "Initial guess")
    lines!(ax, x, uₖ, label = "Newton-Krylov solution")

    axislegend(ax, position = :cb)

    ax = Axis(fig[1, 2], title = "Error", ylabel = "abs2 err", xlabel = "")
    lines!(ax, abs2.(uₖ .- reference))

    fig
end

# ## Solving using the out of place variant

_, stats = newton_krylov(
    bratu,
    copy(u₀), (dx, λ);
    algo = :cg
)
stats

# ## Solving with a fixed forcing
_, stats = newton_krylov!(
    bratu!,
    copy(u₀), (dx, λ), similar(u₀);
    algo = :cg,
    forcing = Ariadne.Fixed(0.1)
)
stats

# ## Solving with no forcing
_, stats = newton_krylov!(
    bratu!,
    copy(u₀), (dx, λ), similar(u₀);
    algo = :cg,
    forcing = nothing
)
stats

# ## Solve using GMRES -- doesn't converge
# ```julia
# _, stats = newton_krylov!(
#     bratu!,
#     copy(u₀), (dx, λ), similar(u₀);
#     algo = :gmres,
# )
# stats
# ```

# ## Solve using GMRES + ILU Preconditioner
_, stats = newton_krylov!(
    bratu!,
    copy(u₀), (dx, λ), similar(u₀);
    algo = :gmres,
    N = (J) -> ilu(collect(J)), # Assembles the full Jacobian
    krylov_kwargs = (; ldiv = true)
)
stats

# ## Solve using FGMRES + ILU Preconditioner
_, stats = newton_krylov!(
    bratu!,
    copy(u₀), (dx, λ), similar(u₀);
    algo = :fgmres,
    N = (J) -> ilu(collect(J)), # Assembles the full Jacobian
    krylov_kwargs = (; ldiv = true)
)
stats

# ## Solve using FGMRES + GMRES Preconditioner
struct GmresPreconditioner{JOp}
    J::JOp
    itmax::Int
end

function LinearAlgebra.mul!(y, P::GmresPreconditioner, x)
    sol, _ = gmres(P.J, x; P.itmax)
    return copyto!(y, sol)
end

_, stats = newton_krylov!(
    bratu!,
    copy(u₀), (dx, λ), similar(u₀);
    algo = :fgmres,
    N = (J) -> GmresPreconditioner(J, 5),
)
stats

# ## Explodes..
# ```julia
# newton_krylov!(
# 	bratu!,
# 	copy(u₀), (dx, λ), similar(u₀);
# 	algo = :cgls, # Cgne
#   krylov_kwargs = (; verbose=1)
# )
# ```
#
# ```julia
# newton_krylov!(
# 	bratu!,
# 	copy(u₀), (dx, λ), similar(u₀);
# 	verbose = 1,
# 	algo = :bicgstab, # L=2
# 	η_max = nothing
# )
# ```
