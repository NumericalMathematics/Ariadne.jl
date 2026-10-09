using Test
using Ariadne
using Asterion

# atan(u - c) = ε Δu: Newton steps from far away overshoot
function atan_residual!(res, u, p)
    (; c, ε) = p
    n = length(u)
    for i in 1:n
        ul = i > 1 ? u[i - 1] : 0.0
        ur = i < n ? u[i + 1] : 0.0
        res[i] = atan(u[i] - c[i]) - ε * (ul - 2 * u[i] + ur)
    end
    return nothing
end

@testset "SER limit cycle" begin
    n = 8
    p = (; c = zeros(n), ε = 0.01)
    u₀ = fill(10.0, n)
    # CFL₀ = 1e2: the residual alternates between increase and decrease;
    # CFL₀ = 10: the pseudo-time step maps u to about -u, the residual reverses its
    # direction at a constant norm
    for cfl₀ in (10.0, 1.0e2)
        alg = PseudoTransientNewtonKrylov(; cfl = SER(; initial = cfl₀, cycle_window = 0))
        _, ws = pseudo_transient!(atan_residual!, copy(u₀), p, alg; σ = -1, reltol = 1.0e-10, maxiters = 100)
        @test ws.status === :max_iterations
        @test ws.stats.rejected_steps == 0

        alg = PseudoTransientNewtonKrylov(; cfl = SER(; initial = cfl₀))
        u, ws = pseudo_transient!(atan_residual!, copy(u₀), p, alg; σ = -1, reltol = 1.0e-10, maxiters = 100)
        @test ws.status === :converged
        @test ws.stats.cfl_ceilings >= 1
        @test maximum(abs, u) < 1.0e-8
    end
    # No cap without a cycle
    alg = PseudoTransientNewtonKrylov(; cfl = SER(; initial = 1.0))
    _, ws = pseudo_transient!(atan_residual!, copy(u₀), p, alg; σ = -1, reltol = 1.0e-10)
    @test ws.status === :converged
    @test ws.stats.cfl_ceilings == 0
    # A cap with several CFL numbers
    alg = PseudoTransientNewtonKrylov(;
        cfl = SER(; initial = (a = 1.0e2, b = 1.0e1)),
        dtau! = (dtau, u, p, cfl) -> fill!(dtau, cfl.a)
    )
    _, ws = pseudo_transient!(atan_residual!, copy(u₀), p, alg; σ = -1, reltol = 1.0e-10, maxiters = 100)
    @test ws.status === :converged
    @test ws.stats.cfl_ceilings >= 1
end
