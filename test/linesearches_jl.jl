using Test
using Ariadne
using LineSearches

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

@testset "LineSearches.jl extension" begin
    @testset "$method" for method in (BackTracking(), StrongWolfe(), MoreThuente())
        for N in (9, 10, 12)
            x_start = vcat(-1.2, ones(N - 1))
            _, stats = newton_krylov(
                generalized_rosenbrock, copy(x_start);
                linesearch! = Ariadne.LineSearches.LineSearches_JL(method),
                max_niter = 100_000,
            )
            @test stats.solved
        end
    end
end
