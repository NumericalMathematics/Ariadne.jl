using Test
using Ariadne
using Asterion
using Asterion: update_cfl, reject_cfl, cfl_too_small, successful
using LinearAlgebra
using SparseArrays

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

@testset "CFL strategies" begin
    s = SER(; growth_min = 0.1, growth_max = 2.0, max = 100.0)
    info(r, r_prior, r0) = (;
        norm_res = r, norm_res_prior = r_prior, norm_res_initial = r0,
        res = nothing, res_prior = nothing, res_initial = nothing,
    )
    @test update_cfl(s, 1.0, info(0.5, 1.0, 1.0)) ≈ 2.0
    @test update_cfl(s, 1.0, info(0.8, 1.0, 1.0)) ≈ 1.25
    @test update_cfl(s, 1.0, info(0.1, 1.0, 1.0)) ≈ 2.0 # growth_max
    @test update_cfl(s, 1.0, info(100.0, 1.0, 1.0)) ≈ 0.1 # growth_min
    @test update_cfl(s, 80.0, info(0.5, 1.0, 1.0)) ≈ 100.0 # max
    @test reject_cfl(s, 1.0) ≈ 0.1
    @test !cfl_too_small(s, 1.0e-6)
    @test cfl_too_small(s, 1.0e-7)

    # Global SER with respect to the initial residual (Lodares et al. 2022, Eq. 122)
    g = SER(; initial = 2.0, reference = :initial, growth_min = 0.0, growth_max = 10.0)
    @test update_cfl(g, 2.0, info(0.1, 0.5, 1.0)) ≈ 20.0
    @test update_cfl(g, 2.0, info(0.01, 0.5, 1.0)) ≈ 20.0 # k CFLⁿ
    @test update_cfl(g, 2.0, info(4.0, 0.5, 1.0)) ≈ 0.5

    # Multiple CFL numbers evolve with the same factor
    m = SER(; initial = (advective = 1.0, diffusive = 0.1), max = (advective = 1.0e8, diffusive = 0.15))
    c = update_cfl(m, m.initial, info(0.5, 1.0, 1.0))
    @test c.advective ≈ 2.0
    @test c.diffusive ≈ 0.15
    @test reject_cfl(m, c).advective ≈ 0.2
    @test !cfl_too_small(m, c)

    # Median-of-max per-variable residual ratio (Eq. 123), variables stored interleaved
    l = SER(; n_variables = 2, reference = :initial, initial = 1.0, growth_min = 0.0, growth_max = 100.0)
    res0 = [1.0, 1.0, 1.0, 1.0]
    res = [0.1, 0.5, 0.1, 0.5] # f₁ = 0.1, f₂ = 0.5, median = 0.3
    @test Ariadne.variable_residual_ratio(res, res0, 2) ≈ 0.3
    i = (; norm_res = NaN, norm_res_prior = NaN, norm_res_initial = NaN, res, res_prior = res0, res_initial = res0)
    @test update_cfl(l, 1.0, i) ≈ 1 / 0.3
end

@testset "PTC: Bratu" begin
    n = 64
    p = (; λ = 1.0, h = 1 / (n + 1))
    for cfl in (SER(; initial = 1.0e-3), SER(; initial = 1.0e-3, n_variables = 1, reference = :initial, growth_min = 0.0, growth_max = 4.0))
        u = zeros(n)
        u, ws = pseudo_transient!(bratu!, u, p, PseudoTransientNewtonKrylov(; cfl); reltol = 1.0e-10)
        @test ws.status === :converged
        @test successful(ws)
        res = similar(u)
        bratu!(res, u, p)
        @test norm(res) <= 1.0e-10 * ws.stats.norm_res_initial
        @test ws.stats.steps == length(ws.history) - 1
        @test ws.stats.krylov_iterations == sum(h -> h.krylov_iterations, ws.history)
        @test ws.stats.timings[:total] > 0
        @test maximum(u) ≈ 0.14 rtol = 0.01 # lower branch, max ≈ 0.1405
    end

    # max_iterations status and callback termination
    u, ws = pseudo_transient!(bratu!, zeros(n), p, PseudoTransientNewtonKrylov(; cfl = SER(; initial = 1.0e-3)); maxiters = 3)
    @test ws.status === :max_iterations
    @test ws.stats.steps == 3
    calls = Ref(0)
    cb = (ws, info) -> (calls[] += 1; info.step == 2)
    u, ws = pseudo_transient!(bratu!, zeros(n), p, PseudoTransientNewtonKrylov(; cfl = SER(; initial = 1.0e-3)); callback = cb)
    @test ws.status === :terminated
    @test calls[] == 2

    # Restart from a workspace after changing parameters in place
    q = (; λ = Ref(0.5), h = p.h)
    bratu_ref!(du, u, q) = bratu!(du, u, (; λ = q.λ[], h = q.h))
    alg = PseudoTransientNewtonKrylov(; cfl = SER(; initial = 1.0e-3))
    ws = PseudoTransientWorkspace(bratu_ref!, zeros(n), q, alg)
    pseudo_transient!(ws; reltol = 1.0e-10)
    @test ws.status === :converged
    steps = ws.stats.steps
    u_half = copy(ws.u)
    q.λ[] = 1.0
    pseudo_transient!(ws; reltol = 1.0e-10)
    @test ws.status === :converged
    @test ws.stats.steps > steps
    @test maximum(ws.u) > maximum(u_half)
    @test maximum(ws.u) ≈ 0.14 rtol = 0.01

    # Local pseudo-time steps from a hook and several Newton iterations per step
    dtau!(dtau, u, p, cfl) = (dtau .= cfl * p.h^2 ./ (2 .+ p.h^2 * p.λ .* exp.(u)); dtau)
    alg = PseudoTransientNewtonKrylov(;
        dtau!, cfl = SER(; initial = 1.0, growth_max = 10.0),
        newton_iterations = 3, newton_reltol = 1.0e-3, forcing = Ariadne.EisenstatWalker()
    )
    u, ws = pseudo_transient!(bratu!, zeros(n), p, alg; reltol = 1.0e-10)
    @test ws.status === :converged
    @test any(h -> h.newton_iterations > 1, ws.history)

    # A scaled norm is prepared once by the Newton workspace and also measures the steady residual
    alg = PseudoTransientNewtonKrylov(; cfl = SER(; initial = 1.0e-3), norm = ScaledNorm((10.0,)))
    u, ws = pseudo_transient!(bratu!, zeros(n), p, alg; reltol = 1.0e-10)
    @test ws.status === :converged
    res = similar(u)
    bratu!(res, u, p)
    @test ws.stats.norm_res ≈ norm(res) / 10
    steady_norm(ws) = @allocated Asterion.evaluate_steady_norm(ws)
    steady_norm(ws)
    @test steady_norm(ws) == 0
end

@testset "PTC from a bad initial guess" begin
    n = 20
    p = (; c = collect(range(-1, 1; length = n)), ε = 0.01)
    u₀ = fill(10.0, n)
    # Newton diverges
    _, result = newton_krylov!(atan_residual!, copy(u₀), p; max_niter = 30)
    @test !result.solved
    # PTC for du/dτ = -F(u) converges
    alg = PseudoTransientNewtonKrylov(; cfl = SER(; initial = 1.0, growth_max = 10.0))
    u, ws = pseudo_transient!(atan_residual!, copy(u₀), p, alg; σ = -1, reltol = 1.0e-12)
    @test ws.status === :converged
    res = similar(u)
    atan_residual!(res, u, p)
    @test norm(res) < 1.0e-10
    # With σ = +1 (the wrong direction), the pseudo-time ODE is unstable
    u, ws = pseudo_transient!(atan_residual!, copy(u₀), p, alg; σ = 1, maxiters = 50)
    @test ws.status !== :converged
end

@testset "Admissibility in PTC" begin
    n = 4
    p = (; a = ones(n))
    u₀ = fill(4.0, n) # the full Newton step jumps to u = 4 - 4 log(4) < 0

    # PTC: inadmissible steps are rejected and the CFL number is reduced
    alg = PseudoTransientNewtonKrylov(; cfl = SER(; initial = 1.0e6), isadmissible = positive)
    u, ws = pseudo_transient!(log_residual!, copy(u₀), p, alg; σ = -1, reltol = 1.0e-12)
    @test ws.status === :converged
    @test ws.stats.rejected_steps >= 1
    @test u ≈ ones(n)

    # PTC with the admissible line search inside the Newton iterations: the user
    # parameters are passed to the hook
    seen_p = Ref{Any}(nothing)
    hook = (u, p) -> (seen_p[] = p; positive(u, p))
    alg = PseudoTransientNewtonKrylov(;
        cfl = SER(; initial = 1.0e6), isadmissible = positive,
        linesearch = AdmissibleLineSearch(hook)
    )
    u, ws = pseudo_transient!(log_residual!, copy(u₀), p, alg; σ = -1, reltol = 1.0e-12)
    @test ws.status === :converged
    @test ws.stats.rejected_steps == 0
    @test seen_p[] === p
    @test u ≈ ones(n)

    # Without any admissible state, the CFL number drops below its minimum
    alg = PseudoTransientNewtonKrylov(; cfl = SER(; initial = 1.0, min = 1.0e-3), isadmissible = (u, p) -> false)
    u, ws = pseudo_transient!(log_residual!, copy(u₀), p, alg; σ = -1)
    @test ws.status === :cfl_too_small
    @test u == u₀
    @test !successful(ws)
end

@testset "Non-finite initial residual" begin
    u, ws = pseudo_transient!(log_residual!, [0.0, 1.0], (; a = ones(2)); σ = -1)
    @test ws.status === :nonfinite
    @test ws.stats.steps == 0
end
