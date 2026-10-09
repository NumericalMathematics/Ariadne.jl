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

@testset "PTC: retry Krylov failures with a new preconditioner" begin
    n = 64
    p = (; λ = 1.0, h = 1 / (n + 1))
    # An exact preconditioner needs a single GMRES iteration, a lagged one more
    build = J -> lu(collect(J))
    function run(retry_krylov_failure)
        alg = PseudoTransientNewtonKrylov(;
            cfl = SER(; initial = 1.0e-3, growth_max = 4.0),
            preconditioner = LaggedPreconditioner(build; refresh_interval = 1000),
            krylov_kwargs = (; ldiv = true, itmax = 1), retry_krylov_failure
        )
        return pseudo_transient!(bratu!, zeros(n), p, alg; reltol = 1.0e-10)
    end
    _, ws_retry = run(true)
    @test ws_retry.status === :converged
    @test ws_retry.stats.krylov_retries > 0
    @test ws_retry.stats.rejected_steps == 0
    _, ws_reject = run(false)
    @test ws_reject.stats.krylov_retries == 0
    @test ws_reject.stats.rejected_steps > 0
end
