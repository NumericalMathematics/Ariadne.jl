using Test
using Ariadne
using Asterion
using Asterion: update_cfl, reject_cfl, cfl_too_small, successful
using LinearAlgebra

# Steady state of the 1D Bratu problem du/dt = u'' + λ exp(u), u(0) = u(1) = 0
function bratu!(du, u, p)
    (; λ, h) = p
    n = length(u)
    for i in 1:n
        ul = i > 1 ? u[i - 1] : zero(eltype(u))
        ur = i < n ? u[i + 1] : zero(eltype(u))
        du[i] = (ul - 2 * u[i] + ur) / h^2 + λ * exp(u[i])
    end
    return nothing
end

# F(u) = atan(u - c) - ε Δu: Newton diverges from a bad initial guess
function atan_residual!(res, u, p)
    (; c, ε) = p
    n = length(u)
    for i in 1:n
        ul = i > 1 ? u[i - 1] : zero(eltype(u))
        ur = i < n ? u[i + 1] : zero(eltype(u))
        res[i] = atan(u[i] - c[i]) - ε * (ul - 2 * u[i] + ur)
    end
    return nothing
end

# F(u) = log|u| - log a has the physical root u = a > 0 and the unphysical root u = -a
log_residual!(res, u, p) = (@. res = log(abs(u)) - log(p.a); nothing)
positive(u, p) = all(>(0), u)

@testset "LodaresSER" begin
    info(r, r_prior, r0) = (;
        norm_res = r, norm_res_prior = r_prior, norm_res_initial = r0,
        res = nothing, res_prior = nothing, res_initial = nothing,
    )
    # Median-of-max per-variable residual ratio (Eq. 123), variables stored interleaved
    l = LodaresSER(2; initial = 1.0, growth_max = 100.0)
    res0 = [1.0, 1.0, 1.0, 1.0]
    res = [0.1, 0.5, 0.1, 0.5] # f₁ = 0.1, f₂ = 0.5, median = 0.3
    i = (; norm_res = NaN, norm_res_prior = NaN, norm_res_initial = NaN, res, res_prior = res0, res_initial = res0)
    @test update_cfl(l, 1.0, i) ≈ 1 / 0.3
    @test l.reference === :initial
    @test l.growth_min == 0

    n = 64
    p = (; λ = 1.0, h = 1 / (n + 1))
    alg = PseudoTransientNewtonKrylov(; cfl = LodaresSER(1; initial = 1.0e-3, growth_max = 4.0))
    u, ws = pseudo_transient!(bratu!, zeros(n), p, alg; reltol = 1.0e-10)
    @test ws.status === :converged
    @test maximum(u) ≈ 0.14 rtol = 0.01
end

@testset "SER: tolerant growth during slow transients" begin
    info(r, r_prior) = (;
        norm_res = r, norm_res_prior = r_prior, norm_res_initial = 1.0,
        res = nothing, res_prior = nothing, res_initial = nothing,
    )
    s = SER(; growth_min = 0.1, growth_max = 2.0, increase_tolerance = 0.2, tolerant_growth = 1.2)
    @test update_cfl(s, 1.0, info(1.1, 1.0)) ≈ 1.2 # small increase: grow anyway
    @test update_cfl(s, 1.0, info(1.5, 1.0)) ≈ 1 / 1.5 # large increase: plain SER
    @test update_cfl(s, 1.0, info(0.5, 1.0)) ≈ 2.0
    @test update_cfl(s, 1.0, info(0.9, 1.0)) ≈ 1.2 # decrease, but slower than tolerant_growth
    # The defaults give plain SER
    d = SER(; growth_min = 0.1, growth_max = 2.0)
    @test update_cfl(d, 1.0, info(1.1, 1.0)) ≈ 1 / 1.1
    @test_throws AssertionError SER(; tolerant_growth = 0.5)
end
