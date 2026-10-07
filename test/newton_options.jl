using Test
using Ariadne
import Ariadne: evaluate!
using LinearAlgebra

function F!(res, x, _)
    res[1] = x[1]^2 + x[2]^2 - 2
    res[2] = exp(x[1] - 1) + x[2]^2 - 2
    return nothing
end

@testset "max_niter" begin
    # `max_niter` is the maximal number of Newton iterations
    for max_niter in (0, 1, 3)
        x₀ = [2.0, 0.5]
        _, result = newton_krylov!(F!, x₀; max_niter, tol_rel = 0.0, tol_abs = 0.0)
        @test result.stats.outer_iterations == max_niter
        @test result.status === :max_iterations
        @test !result.solved
    end
    # The default still converges
    _, result = newton_krylov!(F!, [2.0, 0.5])
    @test result.status === :converged
    @test result.solved
end

@testset "status and failure handling" begin
    # Krylov solver failure: with only one GMRES iteration and an exact Newton
    # method requested, the Krylov solver cannot converge.
    kw = (; forcing = nothing, krylov_kwargs = (; itmax = 1, rtol = 1.0e-14, atol = 0.0))
    x₀ = [2.0, 0.5]
    x, result = newton_krylov!(F!, copy(x₀); kw..., on_krylov_failure = :stop)
    @test result.status === :krylov_failed
    @test !result.solved
    @test result.stats.krylov_failures == 1
    @test x == x₀ # no update after the failed Krylov solve
    @test result.stats.norm_res ≈ norm([2.0^2 + 0.5^2 - 2, exp(2.0 - 1) + 0.5^2 - 2])

    # With `:continue` (default), the Newton steps are taken and the failures are counted
    _, result = newton_krylov!(F!, copy(x₀); kw..., max_niter = 3)
    @test result.stats.krylov_failures >= 1
    @test result.status in (:converged, :max_iterations)

    @test_throws AssertionError newton_krylov!(F!, copy(x₀); on_krylov_failure = :ignore)

    # Non-finite residuals stop the iteration with a clear status:
    # the Newton step from x₁ = 1 jumps to x₁ = 2, where the residual is NaN
    H!(res, x, _) = (res[1] = x[1] > 1.5 ? NaN : x[1] - 2; res[2] = x[2]; nothing)
    _, result = newton_krylov!(H!, [1.0, 1.0]; max_niter = 20)
    @test result.status === :nonfinite
    @test !result.solved
    @test result.stats.outer_iterations == 1

    # Non-finite initial residual
    _, result = newton_krylov!(H!, [2.0, 1.0])
    @test result.status === :nonfinite
    @test result.stats.outer_iterations == 0
end
