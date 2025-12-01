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

# ╔═╡ 3036ce97-baaf-4b2b-819a-89a274af161a
begin
    import Pkg
    # careful: this is _not_ a reproducible environment
    # activate the local environment
    Pkg.activate(".")
    Pkg.instantiate()
    using PlutoUI, PlutoLinks
    using CairoMakie
end

# ╔═╡ 32e51b56-268b-11f0-2d27-f3a6d736affe
@revise using NewtonKrylov

# ╔═╡ 42460e8e-91ea-488f-9833-c660d26bad75
using SummationByPartsOperators

# ╔═╡ da797592-5bdc-4330-ae8b-c29c86488d40
using LinearAlgebra

# ╔═╡ 5bffc041-0019-4f1f-9711-170b82f62926
Implicit = @ingredients(joinpath(dirname(pathof(NewtonKrylov)), "../examples/implicit.jl"));

# ╔═╡ 493850fa-87db-452d-961a-26f92c88d18f
md"""
- Caching for `du1`.
"""

# ╔═╡ 99fbb969-8282-40c9-a24a-661989760e58
function heat_1D!(du, u, (D1m, D1p), t)
    du1 = D1p * u
    mul!(du, D1m, du1)
    return
end

# ╔═╡ fb2bbeda-ba56-4f29-9805-00fe032cea8c
# inital conditions
f(x) = sin(x * π)

# ╔═╡ 19736a4a-e350-47d0-9b59-c40013cb8861
begin
    xmin = 0.0
    xmax = 1.0

    polydeg = 3
    elements = 40
end

# ╔═╡ f68be7a4-df33-4424-8ae2-79d2dd05f6de
begin
    D_local = legendre_derivative_operator(xmin = -1.0, xmax = 1.0, N = polydeg + 1)
    mesh = UniformPeriodicMesh1D(; xmin, xmax, Nx = elements)
end

# ╔═╡ 6b36a1f1-f34a-48de-b76e-49805921513c
begin
    D1m = couple_discontinuously(D_local, mesh, Val(:minus))
    D1p = couple_discontinuously(D_local, mesh, Val(:plus))
end

# ╔═╡ 1dd4180a-e220-4cfb-9bb6-bcd9e4eb7840
x = grid(D1m);

# ╔═╡ 4ffcb456-911e-4e18-bb51-81ffd8456346
u₀ = f.(x);

# ╔═╡ 066f6e6f-5435-4871-9abd-9bd04cd6d0ae
lines(x, u₀)

# ╔═╡ 0b2a4a51-96d5-484e-9b4f-2e16b1ee4462
function solve_heat_1D(G!, x, Δt, t_final, initial_condition, p)
    ts = 0.0:Δt:t_final

    u₀ = initial_condition.(x)

    hist = [copy(u₀)]
    callback = (u) -> push!(hist, copy(u))
    Implicit.solve(G!, heat_1D!, u₀, p, Δt, ts; callback)

    return ts, hist
end

# ╔═╡ d48af076-8ba6-4b18-a0ae-25c5e4dc8017
begin
    Δt = 0.01
    t_final = 50.0
end

# ╔═╡ 41a80a1d-07ca-454f-86f7-058a4e617079
ts, hist_E = solve_heat_1D(Implicit.G_Euler!, x, Δt, t_final, f, (D1m, D1p));

# ╔═╡ 70af8096-673a-4525-a6f8-b2d5c72fc28f
_, hist_M = solve_heat_1D(Implicit.G_Midpoint!, x, Δt, t_final, f, (D1m, D1p));

# ╔═╡ 74c5f50e-4343-4c0f-82ed-283c01d9433b
_, hist_T = solve_heat_1D(Implicit.G_Trapezoid!, x, Δt, t_final, f, (D1m, D1p));

# ╔═╡ bdc258fb-56fa-447b-aa56-275aa72b8d81
@time solve_heat_1D(Implicit.G_Midpoint!, x, Δt, t_final, f, (D1m, D1p));

# ╔═╡ 09c0062e-4c1b-4118-8ccc-6eaa6c25df3b
function plot_timesteps(x, hist, ts, points; title = "")
    fig = Figure()
    ax = Axis(fig[1, 1]; title)
    for p in points
        lines!(ax, x, hist[p], label = "t = $(ts[p])")
    end
    axislegend(ax, position = :cb)
    return fig
end

# ╔═╡ c65d5dd2-cc1e-4093-bf48-8506398688d3
plot_timesteps(x, hist_E, ts, [1, 2, 3, 4, 5, 10, length(ts)]; title = "Euler Δt=$(Δt)")

# ╔═╡ d8bed1d3-5e98-4e75-8f4e-2b9424333159
plot_timesteps(x, hist_M, ts, [1, 2, 3, 4, 5, 10, length(ts)]; title = "Midpoint Δt=$(Δt)")

# ╔═╡ df77e663-6faa-4aac-9ac3-57fcf87927de
plot_timesteps(x, hist_T, ts, [1, 2, 3, 4, 5, 10, length(ts)]; title = "Trapezoid Δt=$(Δt)")

# ╔═╡ 604011c7-3145-4260-8c6f-5033303e2912
md"""
## Jacobian
"""

# ╔═╡ c36c3087-949c-45e9-abbb-25a8b5f23256
J_E = Implicit.jacobian(Implicit.G_Euler!, heat_1D!, zero(x), (D1m, D1p), 0.1, 0.0)

# ╔═╡ e86c8881-626b-463d-ae97-2d8300938d5c
J_M = Implicit.jacobian(Implicit.G_Midpoint!, heat_1D!, zero(x), (D1m, D1p), 0.1, 0.0)

# ╔═╡ 277d1f52-13d4-4dea-9124-6399950410b4
J_T = Implicit.jacobian(Implicit.G_Trapezoid!, heat_1D!, zero(x), (D1m, D1p), 0.1, 0.0)

# ╔═╡ b8d2c8ec-9631-4891-901a-d2bf0f0f54f2
J_M .* 2 + I == J_E

# ╔═╡ 5e0b80bc-385e-4b68-b929-f8dd40e0db6e
J_M == J_T

# ╔═╡ 113c61ff-beaa-4552-9978-55c5bc66f6cb
md"""
# Upwind
"""

# ╔═╡ 7284520b-5115-43ac-a1e9-0e1bdd2c0d12
begin
    nnodes = 120
    accuracy_order = 3
    D = upwind_operators(
        periodic_derivative_operator;
        accuracy_order, xmin, xmax, N = nnodes
    )
    x_U = grid(D.minus)
end

# ╔═╡ bbbf1ea6-fa8b-45b9-af33-24d470ff2ec5
_, hist_EU = solve_heat_1D(Implicit.G_Euler!, x_U, Δt, t_final, f, (D.minus, D.plus));

# ╔═╡ 4483475a-d3ef-40f3-872b-a85aa6728d70
_, hist_MU = solve_heat_1D(Implicit.G_Midpoint!, x_U, Δt, t_final, f, (D.minus, D.plus));

# ╔═╡ dae91988-9c15-43ba-9948-97a5f01650e7
plot_timesteps(x_U, hist_EU, ts, [1, 2, 3, 4, 5, 10, length(ts)]; title = "Euler-Upwind Δt=$(Δt)")

# ╔═╡ 3a5c3f81-a818-4d35-a4bc-d507ed89c2fa
md"""
## Oscilations at the boundary
"""

# ╔═╡ 901dacf2-fb00-4f5f-9d33-e8cae05064fc
@bind i PlutoUI.Slider(1:length(ts))

# ╔═╡ 76454c29-96de-41e7-b5f5-7fd4ce14784f
md"""
t=$(ts[i])
"""

# ╔═╡ cc0f35c4-3d4b-4577-82aa-9ed3037598c9
let
    fig = Figure(title = "t = $(ts[i])")
    ax = Axis(fig[1, 1])
    lines!(ax, x, hist_E[i], label = "Euler")
    lines!(ax, x, hist_M[i], label = "Midpoint")
    # lines!(ax, x, hist_T[i], label="Trapezoid")
    axislegend(ax)
    fig
end

# ╔═╡ 1c3937a1-ae36-406f-92a7-f7bbf599cf4d
lines(x, hist_E[10])

# ╔═╡ f6bf25bd-4990-4758-8a18-84a185a28b5d
lines(x, hist_M[10])

# ╔═╡ Cell order:
# ╠═3036ce97-baaf-4b2b-819a-89a274af161a
# ╠═32e51b56-268b-11f0-2d27-f3a6d736affe
# ╠═5bffc041-0019-4f1f-9711-170b82f62926
# ╠═42460e8e-91ea-488f-9833-c660d26bad75
# ╟─493850fa-87db-452d-961a-26f92c88d18f
# ╠═99fbb969-8282-40c9-a24a-661989760e58
# ╠═fb2bbeda-ba56-4f29-9805-00fe032cea8c
# ╠═19736a4a-e350-47d0-9b59-c40013cb8861
# ╠═f68be7a4-df33-4424-8ae2-79d2dd05f6de
# ╠═6b36a1f1-f34a-48de-b76e-49805921513c
# ╠═1dd4180a-e220-4cfb-9bb6-bcd9e4eb7840
# ╠═4ffcb456-911e-4e18-bb51-81ffd8456346
# ╠═066f6e6f-5435-4871-9abd-9bd04cd6d0ae
# ╠═0b2a4a51-96d5-484e-9b4f-2e16b1ee4462
# ╠═d48af076-8ba6-4b18-a0ae-25c5e4dc8017
# ╠═41a80a1d-07ca-454f-86f7-058a4e617079
# ╠═70af8096-673a-4525-a6f8-b2d5c72fc28f
# ╠═74c5f50e-4343-4c0f-82ed-283c01d9433b
# ╠═bdc258fb-56fa-447b-aa56-275aa72b8d81
# ╠═09c0062e-4c1b-4118-8ccc-6eaa6c25df3b
# ╠═c65d5dd2-cc1e-4093-bf48-8506398688d3
# ╠═d8bed1d3-5e98-4e75-8f4e-2b9424333159
# ╠═df77e663-6faa-4aac-9ac3-57fcf87927de
# ╟─604011c7-3145-4260-8c6f-5033303e2912
# ╠═da797592-5bdc-4330-ae8b-c29c86488d40
# ╠═c36c3087-949c-45e9-abbb-25a8b5f23256
# ╠═e86c8881-626b-463d-ae97-2d8300938d5c
# ╠═277d1f52-13d4-4dea-9124-6399950410b4
# ╠═b8d2c8ec-9631-4891-901a-d2bf0f0f54f2
# ╠═5e0b80bc-385e-4b68-b929-f8dd40e0db6e
# ╟─113c61ff-beaa-4552-9978-55c5bc66f6cb
# ╠═7284520b-5115-43ac-a1e9-0e1bdd2c0d12
# ╠═bbbf1ea6-fa8b-45b9-af33-24d470ff2ec5
# ╠═4483475a-d3ef-40f3-872b-a85aa6728d70
# ╠═dae91988-9c15-43ba-9948-97a5f01650e7
# ╟─3a5c3f81-a818-4d35-a4bc-d507ed89c2fa
# ╠═901dacf2-fb00-4f5f-9d33-e8cae05064fc
# ╟─76454c29-96de-41e7-b5f5-7fd4ce14784f
# ╠═cc0f35c4-3d4b-4577-82aa-9ed3037598c9
# ╠═1c3937a1-ae36-406f-92a7-f7bbf599cf4d
# ╠═f6bf25bd-4990-4758-8a18-84a185a28b5d
