### A Pluto.jl notebook ###
# v0.20.8

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 5b93d37c-4d03-49e5-9289-a2e9eed81229
begin
    import Pkg
    # careful: this is _not_ a reproducible environment
    # activate the local environment
    Pkg.activate(".")
    Pkg.instantiate()
    using PlutoUI, PlutoLinks
    using CairoMakie
end

# ╔═╡ 0d2ecac4-e987-408f-8ffe-50f8c935b93d
@revise using Ariadne

# ╔═╡ 46f41bde-5d70-47a3-b010-8d1cfa02728e
using Krylov

# ╔═╡ 2bd8b52b-ecbb-4a0f-b6eb-6c457e9d27d8
using OffsetArrays

# ╔═╡ ed9acf9a-c1fd-47c2-856f-4f0e8dd8f11a
function bc_periodic!(u)
    N, M = size(u)
    N = N - 2
    M = M - 2

    # Enforce boundary conditions
    # (wrap around)
    u[0, :] .= u[N, :]
    u[N + 1, :] .= u[1, :]
    u[:, 0] .= u[:, N]
    u[:, N + 1] .= u[:, 1]
    return nothing
end

# ╔═╡ 36bab2a0-ab68-4c39-9053-5a94187d079a
function bc_zero!(u)
    N, M = size(u)
    N = N - 2
    M = M - 2

    u[0, :] .= 0
    u[N + 1, :] .= 0
    u[:, 0] .= 0
    u[:, N + 1] .= 0
    return nothing
end

# ╔═╡ 91d47a8c-387b-436b-b554-6e72496cd963
function diffusion!(du, u, (a, Δx, Δy, bc!), _)
    N, M = size(u)
    N = N - 2
    M = M - 2

    # Impose boundary conditions
    bc!(u)

    for i in 1:N
        for j in 1:M
            du[i, j] = a * (
                (u[i + 1, j] - 2 * u[i, j] + u[i - 1, j]) / Δx^2 +
                    (u[i, j + 1] - 2 * u[i, j] + u[i, j - 1]) / Δy^2
            )
        end
    end
    return
end

# ╔═╡ 91d3c59d-a144-4dd2-9e1f-d532c010f442
HaloVectors = @ingredients(joinpath(dirname(pathof(Ariadne)), "../examples/halovector.jl"))

# ╔═╡ 2031ff0e-3b8a-4b21-a49f-e661d810994f
import .HaloVectors: HaloVector

# ╔═╡ 8aebcc1b-ac2b-40a5-9ca2-cbf3e24fdbb8
Implicit = @ingredients(joinpath(dirname(pathof(Ariadne)), "../examples/implicit.jl"));

# ╔═╡ 56354eaa-a634-45e1-9001-84c1d69f845f
import .Implicit: jacobian, solve, G_Euler!

# ╔═╡ 26f4d66e-dddf-4583-b501-9b232f6f4960
function diffusion!(du::HaloVector, u::HaloVector, p, t)
    return diffusion!(du.data, u.data, p, t)
end

# ╔═╡ 550407b6-c287-459b-99de-3243c9c108eb
begin
    a = 0.01
    N = M = 40
end

# ╔═╡ a176451a-4491-454c-a9b5-5f8c3ba2c6c7
begin
    Δx = 1 / (N + 1)  # x-grid spacing
    Δy = 1 / (M + 1)  # y-grid spacing

    # TODO: This is a time-step from explicit
    Δt = Δx^2 * Δy^2 / (2.0 * a * (Δx^2 + Δy^2)) # Largest stable time step
end

# ╔═╡ e18f6f2b-8b7a-4825-bf6d-fa8a23d695e6
let
    N = M = 12
    u₀ = HaloVector(OffsetArray(zeros(N + 2, M + 2), 0:(N + 1), 0:(M + 1)))
    jacobian(G_Euler!, diffusion!, u₀, (a, Δx, Δy, bc_zero!), Δt, 0.0)
end

# ╔═╡ 66d89242-1c67-432c-a24e-c4553630b340
begin
    xs = 0.0:Δx:1.0
    ys = 0.0:Δy:1.0
end

# ╔═╡ ab710836-e597-4601-b9cf-866cf546edb0
length(xs)

# ╔═╡ 8a7b3e9f-074b-4969-b76c-848b62085656
function f(x, y)
    return sin(π * x) .* sin(π * y)
end

# ╔═╡ 1ed331aa-69ba-4f4b-8046-4740afa821aa
u₀ = let
    _u₀ = zeros(N + 2, M + 2)
    _u₀ .= f.(xs, ys')
    HaloVector(OffsetArray(_u₀, 0:(N + 1), 0:(M + 1)))
end

# ╔═╡ bac89f75-4eb2-40c9-8c60-c4d89564f075
function plot_state(u)
    data = u.data.parent
    fig = Figure()
    ax = Axis(fig[1, 1])
    heatmap!(ax, xs, ys, data)

    ax1 = Axis(fig[2, 1])
    lines!(ax1, xs, data[:, M ÷ 2])

    ax2 = Axis(fig[2, 2])
    lines!(ax2, ys, data[N ÷ 2, :])

    return fig
end

# ╔═╡ a53af5f3-ef65-4998-b346-be41b4c71616
plot_state(u₀)

# ╔═╡ 3156b62c-bdc5-4fb7-8cb8-42761061a723
begin
    bc! = bc_zero!
    t_end = 10.0

    # TODO: With periodic boundaries -- newton_krylov solve struggles to reach target accuracy.

    # bc! = bc_periodic!
    # t_end = 3*Δt

    ts = 0:Δt:t_end
end

# ╔═╡ 001ee05a-86e7-4296-a136-4defde8afab0
hist = let
    hist = [copy(u₀)]
    solve(G_Euler!, diffusion!, copy(u₀), (a, Δx, Δy, bc!), Δt, ts; callback = (u) -> push!(hist, copy(u)), verbose = 1)
    hist
end

# ╔═╡ 1d2ebc80-4a25-495a-b8c2-3473beaef5b5
@bind i PlutoUI.Slider(1:length(ts))

# ╔═╡ b9a5717c-8160-4238-a590-84a4cb138d60
plot_state(hist[i])

# ╔═╡ 5d66f81d-7673-4fdd-9abe-76b3c33f3af8
plot_state(hist[1])

# ╔═╡ 3ed18462-c57a-47d0-9e52-d085979c6df1
any(isnan, hist[i].data)

# ╔═╡ Cell order:
# ╠═5b93d37c-4d03-49e5-9289-a2e9eed81229
# ╠═0d2ecac4-e987-408f-8ffe-50f8c935b93d
# ╠═46f41bde-5d70-47a3-b010-8d1cfa02728e
# ╠═ed9acf9a-c1fd-47c2-856f-4f0e8dd8f11a
# ╠═36bab2a0-ab68-4c39-9053-5a94187d079a
# ╠═91d47a8c-387b-436b-b554-6e72496cd963
# ╠═91d3c59d-a144-4dd2-9e1f-d532c010f442
# ╠═2031ff0e-3b8a-4b21-a49f-e661d810994f
# ╠═8aebcc1b-ac2b-40a5-9ca2-cbf3e24fdbb8
# ╠═56354eaa-a634-45e1-9001-84c1d69f845f
# ╠═2bd8b52b-ecbb-4a0f-b6eb-6c457e9d27d8
# ╠═26f4d66e-dddf-4583-b501-9b232f6f4960
# ╠═550407b6-c287-459b-99de-3243c9c108eb
# ╠═a176451a-4491-454c-a9b5-5f8c3ba2c6c7
# ╠═e18f6f2b-8b7a-4825-bf6d-fa8a23d695e6
# ╠═66d89242-1c67-432c-a24e-c4553630b340
# ╠═ab710836-e597-4601-b9cf-866cf546edb0
# ╠═8a7b3e9f-074b-4969-b76c-848b62085656
# ╠═1ed331aa-69ba-4f4b-8046-4740afa821aa
# ╠═a53af5f3-ef65-4998-b346-be41b4c71616
# ╠═bac89f75-4eb2-40c9-8c60-c4d89564f075
# ╠═3156b62c-bdc5-4fb7-8cb8-42761061a723
# ╠═001ee05a-86e7-4296-a136-4defde8afab0
# ╠═b9a5717c-8160-4238-a590-84a4cb138d60
# ╠═1d2ebc80-4a25-495a-b8c2-3473beaef5b5
# ╠═5d66f81d-7673-4fdd-9abe-76b3c33f3af8
# ╠═3ed18462-c57a-47d0-9e52-d085979c6df1
