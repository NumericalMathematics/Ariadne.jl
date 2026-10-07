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
    x, result = newton_krylov!(F!, copy(x₀); kw...) # `on_krylov_failure = :stop` (default)
    @test result.status === :krylov_failed
    @test !result.solved
    @test result.stats.krylov_failures == 1
    @test x == x₀ # no update after the failed Krylov solve
    @test result.stats.norm_res ≈ norm([2.0^2 + 0.5^2 - 2, exp(2.0 - 1) + 0.5^2 - 2])

    # With `:continue`, the Newton steps are taken and the failures are counted
    _, result = newton_krylov!(F!, copy(x₀); kw..., on_krylov_failure = :continue, max_niter = 3)
    @test result.stats.krylov_failures >= 1
    @test result.status in (:converged, :max_iterations)

    @test_throws ArgumentError newton_krylov!(F!, copy(x₀); on_krylov_failure = :ignore)

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

@testset "scaled norms" begin
    # Variables of very different magnitude
    S!(res, x, _) = (res[1] = x[1] - 1.0e6; res[2] = x[2] - 1.0e-6; nothing)
    scale = (1.0e6, 1.0e-6)

    n = ScaledNorm(scale)
    @test n([1.0e6, 1.0e-6]) ≈ sqrt(2)
    @test n([2.0e6, 0.0, 1.0e6, 1.0e-6]) ≈ sqrt(4 + 0 + 1 + 1)
    @test ScaledNorm([2.0, 4.0])([2.0, 4.0]) ≈ sqrt(2)

    # The workspace uses the given norm for the residual
    ws = NewtonKrylovWorkspace(S!, [0.0, 0.0], nothing, zeros(2); norm = n)
    @test evaluate!(ws) ≈ sqrt(2)

    # With the unscaled norm, the second variable is ignored by the termination criterion
    x, result = newton_krylov!(
        S!, [0.0, 0.0]; forcing = Ariadne.Fixed(0.5),
        tol_rel = 1.0e-3, tol_abs = 0.0
    )
    @test result.solved
    # With the scaled norm, both variables are solved to the relative tolerance
    x_scaled, result = newton_krylov!(
        S!, [0.0, 0.0]; forcing = Ariadne.Fixed(0.5),
        tol_rel = 1.0e-3, tol_abs = 0.0, norm = n
    )
    @test result.solved
    @test abs(x_scaled[2] - 1.0e-6) <= 1.0e-3 * 1.0e-6 * sqrt(2)
    @test abs(x_scaled[1] - 1.0e6) <= 1.0e-3 * 1.0e6 * sqrt(2)

    # Per-variable residual reduction (median over variables, Lodares et al. 2022)
    res₀ = [1.0, 10.0, 100.0, 1.0, 10.0, 100.0]
    res = [0.5, 1.0, 100.0, 0.5, 1.0, 100.0]
    @test Ariadne.variable_residual_ratio(res, res₀, 3) ≈ 0.5

    # Variables with zero initial residual are treated as converged
    res₀ = [2.0, 0.0, 2.0, 0.0, 2.0, 0.0]
    res = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    @test Ariadne.variable_residual_ratio(res, res₀, 2) ≈ 0.5
    @test Ariadne.variable_residual_ratio(res, zeros(6), 2) == 0
end
