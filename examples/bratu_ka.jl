# # 1D bratu equation from (Kan2022-ko)[@cite]

# ## Necessary packages
using Ariadne, Krylov
using KrylovPreconditioners
using SparseArrays, LinearAlgebra
using CairoMakie
using KernelAbstractions

# ## 1D Bratu equations
# $y′′ + λ * exp(y) = 0$

@kernel function bratu_kernel!(res, y, (Δx, λ))
    i = @index(Global, Linear)
    N = length(res)
    y_l = i == 1 ? zero(eltype(y)) : y[i - 1]
    y_r = i == N ? zero(eltype(y)) : y[i + 1]
    y′′ = (y_r - 2y[i] + y_l) / Δx^2

    res[i] = y′′ + λ * exp(y[i]) # = 0
end

function bratu!(res, y, p)
    device = KernelAbstractions.get_backend(res)
    kernel = bratu_kernel!(device)
    kernel(res, y, p, ndrange = length(res))
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
