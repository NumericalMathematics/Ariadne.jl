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

@testset "BacktrackingLineSearch with exceptions in trial states" begin
    # The full Newton step from x = 3 for log(x) = 0 is x = 3 - 3 log(3) < 0,
    # where `log` throws a `DomainError`
    L!(res, x, _) = (res[1] = log(x[1]); nothing)
    x, result = newton_krylov!(L!, [3.0]; linesearch! = BacktrackingLineSearch())
    @test result.solved
    @test x[1] ≈ 1
    @test_throws DomainError newton_krylov!(L!, [3.0]; linesearch! = NoLineSearch())
    @test_throws DomainError newton_krylov!(
        L!, [3.0]; linesearch! = BacktrackingLineSearch(; reject_exceptions = ())
    )

    # Exceptions thrown in a `Threads.@threads` loop of the residual are wrapped in a
    # `CompositeException` of `TaskFailedException`s
    function L_threaded!(res, x, _)
        Threads.@threads for i in eachindex(res, x)
            res[i] = log(x[i])
        end
        return nothing
    end
    x, result = newton_krylov!(L_threaded!, [3.0, 4.0]; linesearch! = BacktrackingLineSearch())
    @test result.solved
    @test x ≈ [1, 1]

    # The caught exceptions are logged: with `@info` if verbose, with `@debug` otherwise
    @test_logs (:info, r"threw an exception") match_mode = :any newton_krylov!(
        L!, [3.0]; linesearch! = BacktrackingLineSearch(), verbose = 1
    )
    @test_logs (:debug, r"threw an exception") min_level = Base.CoreLogging.Debug match_mode = :any newton_krylov!(
        L!, [3.0]; linesearch! = BacktrackingLineSearch()
    )
    @test_logs min_level = Base.CoreLogging.Info newton_krylov!(L!, [3.0]; linesearch! = BacktrackingLineSearch())
    # Custom line searches without the `verbose` keyword still work
    struct FullStep <: Ariadne.LineSearches.AbstractLineSearch end
    (::FullStep)(ws, _, d) = (ws.u .+= d; Ariadne.evaluate!(ws))
    _, result = newton_krylov!((res, x, _) -> (res .= x .- 1; nothing), [3.0]; linesearch! = FullStep(), verbose = 1)
    @test result.solved

    # Other exceptions are rethrown
    E!(res, x, _) = (x[1] < 0 && throw(ArgumentError("negative")); res[1] = log(abs(x[1])); nothing)
    @test_throws ArgumentError newton_krylov!(E!, [3.0]; linesearch! = BacktrackingLineSearch())

    # If all trial states throw, the status is `:nonfinite`
    T!(res, x, _) = (x[1] < 2.9 && throw(DomainError(x[1])); res[1] = x[1] - 1; nothing)
    _, result = newton_krylov!(T!, [3.0]; linesearch! = BacktrackingLineSearch(; n_iter_max = 3))
    @test result.status === :nonfinite

    @test Ariadne.LineSearches.matches_exception(DomainError(1), (DomainError,))
    @test !Ariadne.LineSearches.matches_exception(ArgumentError(""), (DomainError,))
    @test !Ariadne.LineSearches.matches_exception(CompositeException(), (DomainError,))
end
