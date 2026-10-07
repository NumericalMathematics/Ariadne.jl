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
        @test !result.solved
    end
    # The default still converges
    _, result = newton_krylov!(F!, [2.0, 0.5])
    @test result.solved
end
