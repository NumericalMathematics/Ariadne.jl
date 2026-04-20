### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# ╔═╡ eb792fe6-3cbe-11f1-2eca-0bd287486337
begin
    import Pkg
    # careful: this is _not_ a reproducible environment
    # activate the local environment
    Pkg.activate(".")
    Pkg.instantiate()
    using PlutoUI, PlutoLinks
    using CairoMakie
end

# ╔═╡ 869833fd-ee32-4469-ad9f-ef6489f68811
@revise using Ariadne

# ╔═╡ 0aa45e3b-433b-4385-aa8c-e71e8acf725a
import NonlinearSolve as NLS

# ╔═╡ fa13b4ca-a809-43c4-979d-e0864874d8f6
md"""
## Generalized Rosenbrock

This example is taken from Fig. 1 of:
> A. Pal et al., "NonlinearSolve.jl: High-performance and robust solvers for systems
> of nonlinear equations in Julia," arXiv [math.NA], 24-Mar-2024.
> https://arxiv.org/abs/2403.16341

### Problem definition

The generalized Rosenbrock function in $N$ dimensions:
```math
F(x)_1 = 1 - x_1, \quad F(x)_i = 10(x_i - x_{i-1}^2), \quad i = 2, \ldots, N
```
"""

# ╔═╡ edf5d7a1-4f17-484a-aebf-729ea149b580
function generalized_rosenbrock(x, _)
    return vcat(
        1 - x[1],
        10 .* (x[2:end] .- x[1:(end - 1)] .* x[1:(end - 1)])
    )
end

# ╔═╡ 078b6618-f037-44b1-9c17-4a1785170bfc
N = 5

# ╔═╡ 80cde332-fbd1-49a0-bd61-944720f885fe
x_start = vcat(-1.2, ones(N - 1))

# ╔═╡ 4bb5bf4b-a37b-4d02-97a6-00bece7dc9f8
md"""
## using Ariadne
"""

# ╔═╡ 772047e4-debd-4871-a659-f91d15c3ea45
let
    _, stats = newton_krylov(
        generalized_rosenbrock,
        copy(x_start);
        algo = :gmres,
        linesearch! = NoLineSearch(),
        max_niter = 100_000
    )
    stats
end

# ╔═╡ 42a30e51-8cf7-49fa-8348-159803d4a1c3
let
    _, stats = newton_krylov(
        generalized_rosenbrock,
        copy(x_start);
        algo = :gmres,
        linesearch! = BacktrackingLineSearch(),
        max_niter = 100_000
    )
    stats
end

# ╔═╡ 4ec97f25-7af2-4138-9203-f933d8e593d9
md"""
## using NonlinearSolve
"""

# ╔═╡ eafb63ff-857d-4cc3-84eb-19ad4dd5755a
prob = NLS.NonlinearProblem(generalized_rosenbrock, x_start)

# ╔═╡ 69a3e7b5-bf52-408b-bd33-80e0618b539b
alg = NLS.NewtonRaphson(
    linesearch = missing,
    forcing = NLS.EisenstatWalkerForcing2(),
    linsolve = NLS.KrylovJL()
)

# ╔═╡ aa53ea6f-9d39-43ea-b130-f0c98f187958
alg2 = NLS.NewtonRaphson(
    linesearch = NLS.BackTracking(),
    forcing = NLS.EisenstatWalkerForcing2(),
    linsolve = NLS.KrylovJL()
)

# ╔═╡ 08473c35-a4f6-4115-8db5-03b13f4a2cce
let
    sol = NLS.solve(prob, alg, reltol = 1.0e-6, abstol = 1.0e-12, verbose = false)
    sol.stats
end

# ╔═╡ 9cea0413-f915-4784-a2c0-9b041c71b649
let
    sol = NLS.solve(prob, alg2, reltol = 1.0e-6, abstol = 1.0e-12, verbose = false)
    sol.stats
end

# ╔═╡ Cell order:
# ╠═eb792fe6-3cbe-11f1-2eca-0bd287486337
# ╠═869833fd-ee32-4469-ad9f-ef6489f68811
# ╠═0aa45e3b-433b-4385-aa8c-e71e8acf725a
# ╟─fa13b4ca-a809-43c4-979d-e0864874d8f6
# ╠═edf5d7a1-4f17-484a-aebf-729ea149b580
# ╠═078b6618-f037-44b1-9c17-4a1785170bfc
# ╠═80cde332-fbd1-49a0-bd61-944720f885fe
# ╟─4bb5bf4b-a37b-4d02-97a6-00bece7dc9f8
# ╠═772047e4-debd-4871-a659-f91d15c3ea45
# ╠═42a30e51-8cf7-49fa-8348-159803d4a1c3
# ╟─4ec97f25-7af2-4138-9203-f933d8e593d9
# ╠═eafb63ff-857d-4cc3-84eb-19ad4dd5755a
# ╠═69a3e7b5-bf52-408b-bd33-80e0618b539b
# ╠═aa53ea6f-9d39-43ea-b130-f0c98f187958
# ╠═08473c35-a4f6-4115-8db5-03b13f4a2cce
# ╠═9cea0413-f915-4784-a2c0-9b041c71b649
