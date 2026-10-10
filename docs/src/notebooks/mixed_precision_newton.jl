### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# ╔═╡ a1ee3fc2-049b-4046-8a7b-b0dc9838fa26
begin
    import Pkg
    # careful: this is _not_ a reproducible environment
    # activate the local environment
    Pkg.activate(".")
    Pkg.instantiate()
    using CairoMakie
end

# ╔═╡ 046a38d2-cda9-4b8f-870c-ec3bd075942e
using Ariadne

# ╔═╡ 18ba5ed9-a5a7-4f38-8efb-e46a3d2b7e0b
using Ariadne: MixedPrecisionLU, IterativeRefinementWorkspace, LaggedPreconditioner

# ╔═╡ e1a4188c-6a1d-425b-b4eb-dc4bf1c463f2
using LinearAlgebra

# ╔═╡ 2f9fdb12-9b8e-4c3d-887d-3671b6ea1e7c
using StochasticRounding

# ╔═╡ afeeef67-9452-4ece-8f65-5def6bfd8f37
md"""
# Stochastic rounding in three-precision Newton

Kelley's *Newton's method in three precisions* computes the residual in double precision,
stores the Jacobian in single precision, factors it in half precision, and solves for the
Newton step with iterative refinement (IR) on the single-precision Jacobian, preconditioned
by the half-precision factors ("IR 32-16"). As long as the refinement converges, the Newton
iteration is the same as with a double-precision Jacobian.

For the nearly singular Chandrasekhar H-equation (``c = 0.9999``, Table 3 of the paper) IR
32-16 with ``N = 4096`` stops converging after a few Newton steps, and Newton stalls near
``5 \cdot 10^{-4}``. This notebook shows that the same factorization computed with
**stochastic rounding** (`Float16sr` of
[StochasticRounding.jl](https://github.com/milankl/StochasticRounding.jl)) keeps the
refinement converging: the number of refinement sweeps stays nearly constant as ``N``
grows, while with round-to-nearest it grows until IR fails.

The only difference between the two runs is the rounding mode of the arithmetic in the
half-precision LU factorization. The Jacobian is rounded to half precision the same way
(to nearest) in both.

- C. T. Kelley, *Newton's method in three precisions*, Pacific J. Optim. 20 (2024), arXiv:2307.16051.
- M. Croci, M. Fasi, N. J. Higham, T. Mary, M. Mikaitis, *Stochastic rounding: implementation, error analysis and applications*, R. Soc. Open Sci. 9 (2022).
"""

# ╔═╡ 92da1b0d-30c9-4805-b78d-aaa03fec4889
md"""
## The H-equation

The composite midpoint rule discretization of the Chandrasekhar H-equation with nodes
``\mu_i = (i - 1/2)/N`` is

```math
F(x)_i = x_i - \left(1 - \frac{c}{2N} \sum_{j=1}^N \frac{\mu_i x_j}{\mu_i + \mu_j}\right)^{-1} = 0,
```

with ``\mu_i / (\mu_i + \mu_j) = (i - 1/2)/(i + j - 1)``. Its Jacobian is
``F'(x) = I - \operatorname{diag}(G(x)^2 \, c (i - 1/2)) H`` with
``H_{ij} = 1/(2N (i + j - 1))`` and ``G`` the inverse term above. The initial iterate is
``x_0 = 1``.
"""

# ╔═╡ be57ab18-74d4-4cb4-b95c-1fabd70158cd
function heq!(F, x, c)
    N = length(x)
    for i in 1:N
        s = zero(eltype(F))
        for j in 1:N
            s += x[j] / (2N * (i + j - 1))
        end
        F[i] = x[i] - inv(1 - c * (i - 1 / 2) * s)
    end
    return nothing
end

# ╔═╡ bc1b9f7f-367c-4f44-8eba-a6eeae93ba4c
function heq_jacobian(x, c)
    N = length(x)
    G = similar(x)
    for i in 1:N
        s = zero(eltype(x))
        for j in 1:N
            s += x[j] / (2N * (i + j - 1))
        end
        G[i] = inv(1 - c * (i - 1 / 2) * s)
    end
    return [(i == j) - G[i]^2 * c * (i - 1 / 2) / (2N * (i + j - 1)) for i in 1:N, j in 1:N]
end

# ╔═╡ afa81835-98fe-4784-83cc-fb5c8a127d5f
md"""
## IR 32-16

`MixedPrecisionLU` stores the Jacobian in single precision and factors it in the
`factor_precision` `TF`; the triangular solves use the factors converted to single
precision. `IterativeRefinementWorkspace` replaces the Krylov solver of `newton_krylov!`
with iterative refinement on the stored single-precision Jacobian, using Kelley's stopping
rule (relative residual ``10 \varepsilon_{32}``, stop when the residual decreases by less
than a factor ``0.9``). Every Newton step builds a new factorization.

StochasticRounding.jl uses one global random number generator, so the stochastically
rounded factorization runs on one thread.
"""

# ╔═╡ 1fdea504-a09b-43eb-9b98-7a022c559ad7
function solve_heq(N, c, TF; seed = 1, max_niter = 20)
    StochasticRounding.seed(seed)
    u = ones(N)
    res = zeros(N)
    threaded = !(TF <: StochasticRounding.AbstractStochasticFloat)
    P = LaggedPreconditioner(
        J -> MixedPrecisionLU(
            Matrix{Float32}(heq_jacobian(J.u, c));
            factor_precision = TF, solve_precision = Float32, threaded
        )
    )
    residuals = Float64[]
    sweeps = Int[]
    ws = NewtonKrylovWorkspace(heq!, u, c, res, IterativeRefinementWorkspace(res))
    _, result = newton_krylov!(
        ws; tol_rel = 1.0e-8, tol_abs = 1.0e-8, max_niter, forcing = nothing, N = P,
        # As in Kelley's code, take the step of a refinement that stopped converging
        on_krylov_failure = :continue,
        iteration_callback = function (ws, info)
            isempty(residuals) && push!(residuals, info.norm_res_prior)
            push!(residuals, info.norm_res)
            push!(sweeps, info.krylov_iterations)
            return nothing
        end
    )
    return (; N, TF, result.status, residuals = residuals ./ first(residuals), sweeps)
end

# ╔═╡ 562678f3-8698-4c76-a47b-610448d2af16
c = 0.9999

# ╔═╡ 2d7f1482-73b0-4aae-8110-e1aef7289510
Ns = [512, 1024, 2048]

# ╔═╡ 7e861bdb-055a-4020-9816-e280289fd0bd
runs = Dict((N, TF) => solve_heq(N, c, TF) for N in Ns, TF in (Float16, Float16sr))

# ╔═╡ 6b0829ce-576e-43f3-ad3c-3775a9cb9523
md"""
## Results

Newton steps to ``\|F(x_n)\| \le 10^{-8} \|F(x_0)\| + 10^{-8}`` and the total number of
refinement sweeps over all Newton steps:
"""

# ╔═╡ c5448d32-7754-4abb-ac77-f1efe5aea877
let
    rows = map(Ns) do N
        rn = runs[(N, Float16)]
        sr = runs[(N, Float16sr)]
        "| $N | $(length(rn.sweeps)) | $(sum(rn.sweeps)) | $(length(sr.sweeps)) | $(sum(sr.sweeps)) |"
    end
    Markdown.parse(
        """
        | N | Newton steps, Float16 | sweeps, Float16 | Newton steps, Float16sr | sweeps, Float16sr |
        |---|---|---|---|---|
        """ * join(rows, "\n")
    )
end

# ╔═╡ c9f2d069-1be6-4b15-ba72-09055bd4b2fe
let
    fig = Figure(size = (900, 360))
    ax1 = Axis(fig[1, 1]; yscale = log10, xlabel = "Newton step", ylabel = "‖F(xₙ)‖ / ‖F(x₀)‖", title = "Residual")
    ax2 = Axis(fig[1, 2]; xlabel = "Newton step", ylabel = "refinement sweeps", title = "Sweeps per Newton step")
    colors = Makie.wong_colors()
    for (k, N) in enumerate(Ns)
        for (TF, style) in ((Float16, :solid), (Float16sr, :dash))
            r = runs[(N, TF)]
            label = "N = $N, $(TF)"
            scatterlines!(ax1, 0:(length(r.residuals) - 1), r.residuals; color = colors[k], linestyle = style, label)
            scatterlines!(ax2, 1:length(r.sweeps), r.sweeps; color = colors[k], linestyle = style, label)
        end
    end
    Legend(fig[1, 3], ax2)
    fig
end

# ╔═╡ fd68722f-a7af-4924-994b-b451418660a2
md"""
With round-to-nearest factors the number of sweeps grows with ``N``; the sweeps of the last
Newton steps, where the Jacobian is closest to singular, grow fastest. With stochastically
rounded factors it stays nearly constant.

At ``N = 4096``, the size of Kelley's Table 3, round-to-nearest IR 32-16 fails: after five
Newton steps the refinement stops making progress, and Newton converges only linearly. These runs take a few minutes for the
stochastically rounded factorization on one thread, so the numbers below were computed
once with the same setup (three seeds for `Float16sr`) and are not recomputed here:

| N = 4096, c = 0.9999 | Newton steps | sweeps | ‖F‖/‖F₀‖ at the end |
|---|---|---|---|
| Float16 (Kelley: IR 32-16) | 10 (not converged) | 146 | 4.5e-4 |
| Float16sr | 8 | 38–43 | 3.9e-10 |
| Float64 Jacobian, LU in Float64 | 8 | – | 3.9e-10 |

For ``c = 0.99`` (Table 2) both converge in 5 Newton steps, with 86 sweeps for Float16 and
17–18 for Float16sr.

A likely explanation: with round-to-nearest the rounding errors in the ``O(N)`` updates of
each entry of the LU factors accumulate with a bias, roughly like ``N u``, while stochastic
rounding makes them unbiased, so they grow like ``\sqrt{N} u`` (Connolly, Higham and Mary,
*Stochastic rounding and its probabilistic backward error analysis*, SISC 2021). This is
an inference from the convergence rates; the backward errors of the factors were not
measured here.
"""

# ╔═╡ Cell order:
# ╠═a1ee3fc2-049b-4046-8a7b-b0dc9838fa26
# ╠═046a38d2-cda9-4b8f-870c-ec3bd075942e
# ╠═18ba5ed9-a5a7-4f38-8efb-e46a3d2b7e0b
# ╠═e1a4188c-6a1d-425b-b4eb-dc4bf1c463f2
# ╠═2f9fdb12-9b8e-4c3d-887d-3671b6ea1e7c
# ╟─afeeef67-9452-4ece-8f65-5def6bfd8f37
# ╟─92da1b0d-30c9-4805-b78d-aaa03fec4889
# ╠═be57ab18-74d4-4cb4-b95c-1fabd70158cd
# ╠═bc1b9f7f-367c-4f44-8eba-a6eeae93ba4c
# ╟─afa81835-98fe-4784-83cc-fb5c8a127d5f
# ╠═1fdea504-a09b-43eb-9b98-7a022c559ad7
# ╠═562678f3-8698-4c76-a47b-610448d2af16
# ╠═2d7f1482-73b0-4aae-8110-e1aef7289510
# ╠═7e861bdb-055a-4020-9816-e280289fd0bd
# ╟─6b0829ce-576e-43f3-ad3c-3775a9cb9523
# ╠═c5448d32-7754-4abb-ac77-f1efe5aea877
# ╠═c9f2d069-1be6-4b15-ba72-09055bd4b2fe
# ╟─fd68722f-a7af-4924-994b-b451418660a2
