using Test
using Ariadne

# Generalized Rosenbrock from:
# A. Pal et al., "NonlinearSolve.jl: High-performance and robust solvers for
# systems of nonlinear equations in Julia," arXiv [math.NA], 24-Mar-2024.
# https://arxiv.org/abs/2403.16341 (Fig. 1)
function generalized_rosenbrock(x, _)
    return vcat(
        1 - x[1],
        10 .* (x[2:end] .- x[1:(end - 1)] .* x[1:(end - 1)])
    )
end

@testset "Generalized Rosenbrock" begin
    @testset "NoLineSearch" begin
        # NoLineSearch converges for small N
        for N in (2, 4, 6, 8)
            x_start = vcat(-1.2, ones(N - 1))
            _, stats = newton_krylov(
                generalized_rosenbrock, x_start;
                max_niter = 100_000,
                linesearch! = NoLineSearch(),
            )
            @test stats.solved
        end

        for N in (9, 10, 12)
            x_start = vcat(-1.2, ones(N - 1))
            _, stats = newton_krylov(
                generalized_rosenbrock, x_start;
                linesearch! = NoLineSearch(),
                max_niter = 100_000,
            )
            @test !stats.solved
        end
    end

    @testset "BacktrackingLineSearch" begin
        # BacktrackingLineSearch converges for larger N where NoLineSearch fails
        for N in (9, 10, 12)
            x_start = vcat(-1.2, ones(N - 1))
            _, stats = newton_krylov(
                generalized_rosenbrock, x_start;
                linesearch! = BacktrackingLineSearch(),
                max_niter = 100_000,
            )
            @test stats.solved
        end
    end
end
