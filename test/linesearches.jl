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
    # Custom line searches get the verbosity level
    seen_verbose = Ref(-1)
    struct FullStep <: Ariadne.LineSearches.AbstractLineSearch end
    (::FullStep)(ws, norm_res_prior, d; verbose = 0) = (seen_verbose[] = verbose; ws.u .+= d; (Ariadne.evaluate!(ws), :success, 1.0))
    _, result = newton_krylov!((res, x, _) -> (res .= x .- 1; nothing), [3.0]; linesearch! = FullStep(), verbose = 1)
    @test result.solved
    @test seen_verbose[] == 1

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

@testset "BacktrackingLineSearch failures and parabolic step" begin
    parabolic_step = Ariadne.LineSearches.parabolic_step
    # Exact for a parabola: ff(λ) = (λ - 0.3)^2 + 0.91 has its minimum at λ = 0.3
    ff(λ) = (λ - 0.3)^2 + 0.91
    @test parabolic_step(1.0, 2.0, ff(0.0), ff(1.0), ff(2.0)) ≈ 0.3
    # The step is clamped to [0.1 λc, 0.5 λc]
    @test parabolic_step(0.5, 1.0, ff(0.0), ff(0.5), ff(1.0)) ≈ 0.25
    ff_low(λ) = (λ - 0.01)^2
    @test parabolic_step(1.0, 2.0, ff_low(0.0), ff_low(1.0), ff_low(2.0)) ≈ 0.1
    # Negative curvature (no minimum): the smallest step length 0.1 λc
    @test parabolic_step(1.0, 2.0, 1.0, 2.0, 1.0) ≈ 0.1
    @test parabolic_step(0.5, 1.0, 1.0, 2.0, 1.0) ≈ 0.05
    # Non-finite residuals, e.g., from a trial state that threw: halve
    @test parabolic_step(1.0, 2.0, 1.0, Inf, 4.0) == 0.5
    @test parabolic_step(1.0, 2.0, 1.0, 4.0, Inf) == 0.5

    # The full Newton step for atan(x) = 0 from x = 3 overshoots to |x| > 3, where
    # |atan(x)| is larger, so a single trial cannot satisfy the Armijo condition
    A!(res, x, _) = (res[1] = atan(x[1]); nothing)
    _, result = newton_krylov!(
        A!, [3.0]; linesearch! = BacktrackingLineSearch(; n_iter_max = 1), max_niter = 1
    )
    @test result.stats.linesearch_failures == 1
    @test result.status === :max_iterations

    for parabolic in (true, false)
        x, result = newton_krylov!(A!, [3.0]; linesearch! = BacktrackingLineSearch(; parabolic))
        @test result.solved
        @test abs(x[1]) < 1.0e-6
        @test result.stats.linesearch_failures == 0
    end

    # Line searches return the residual norm, a status and the step length
    ws = NewtonKrylovWorkspace(A!, [3.0], nothing, zeros(1))
    norm_res_prior = Ariadne.evaluate!(ws)
    d = [-atan(3.0) * (1 + 3.0^2)] # Newton direction at x = 3
    norm_res, status, λ = BacktrackingLineSearch(; n_iter_max = 1)(ws, norm_res_prior, copy(d))
    @test status === :failed
    @test λ == 1
    @test norm_res > norm_res_prior
    ws.u .= 3.0
    norm_res, status, λ = BacktrackingLineSearch()(ws, norm_res_prior, copy(d))
    @test status === :success
    @test ws.u ≈ [3.0] + λ * d
    @test 0 < λ < 1
    @test norm_res < norm_res_prior
    ws.u .= 3.0
    @test NoLineSearch()(ws, norm_res_prior, copy(d))[2:3] == (:success, 1.0)
    # The step length of `AdmissibleLineSearch` refers to the direction before `max_step`
    ws.u .= 3.0
    norm_res, status, λ = AdmissibleLineSearch((u, p) -> true, NoLineSearch(); max_step = (u, d, p) -> 0.25)(ws, norm_res_prior, copy(d))
    @test λ == 0.25
    @test ws.u ≈ [3.0] + λ * d
    # A failed admissible line search undoes the step and returns the step length 0
    ws.u .= 3.0
    @test AdmissibleLineSearch((u, p) -> false, NoLineSearch())(ws, norm_res_prior, copy(d)) == (Inf, :failed, 0.0)
    @test ws.u == [3.0]

    # Without a line search, no failures are reported
    _, result = newton_krylov!(A!, [3.0]; linesearch! = NoLineSearch(), max_niter = 3)
    @test result.stats.linesearch_failures == 0
end

# F(u) = log|u| - log a has the physical root u = a > 0 and the unphysical root u = -a
log_residual!(res, u, p) = (@. res = log(abs(u)) - log(p.a); nothing)
positive(u, p) = all(>(0), u)

@testset "AdmissibleLineSearch" begin
    n = 4
    p = (; a = ones(n))
    u₀ = fill(4.0, n) # the full Newton step jumps to u = 4 - 4 log(4) < 0

    # Plain Newton converges to the unphysical root
    u, result = newton_krylov!(log_residual!, copy(u₀), p; forcing = nothing)
    @test result.solved
    @test u ≈ -ones(n) rtol = 1.0e-5

    # The admissible line search keeps u > 0 and finds the physical root
    iterates = Vector{Float64}[]
    u, result = newton_krylov!(
        log_residual!, copy(u₀), p; forcing = nothing,
        linesearch! = AdmissibleLineSearch(positive),
        callback = (u, res, n) -> push!(iterates, copy(u))
    )
    @test result.solved
    @test u ≈ ones(n) rtol = 1.0e-5
    @test all(x -> all(>(0), x), iterates)

    # A step limiter (at most a relative change of 50% per Newton step) with full steps
    max_step(u, d, p) = minimum(i -> d[i] < 0 ? 0.5 * u[i] / -d[i] : Inf, eachindex(u, d))
    iterates = Vector{Float64}[]
    u, result = newton_krylov!(
        log_residual!, copy(u₀), p; forcing = nothing,
        linesearch! = AdmissibleLineSearch(positive, NoLineSearch(); max_step),
        callback = (u, res, n) -> push!(iterates, copy(u))
    )
    @test result.solved
    @test u ≈ ones(n) rtol = 1.0e-5
    @test all(i -> all(iterates[i + 1] .>= 0.5 .* iterates[i] .- 1.0e-12), 1:(length(iterates) - 1))

    # No admissible state along the direction: the line search fails with Inf and
    # resets the state
    u, result = newton_krylov!(
        log_residual!, copy(u₀), p; forcing = nothing,
        linesearch! = AdmissibleLineSearch((u, p) -> false, BacktrackingLineSearch(; n_iter_max = 3))
    )
    @test result.status === :nonfinite
    @test u ≈ u₀

    # In admissible states, residuals that throw are still rejected by the backtracking:
    # the full Newton step from x = 3 for log(x) = 0 goes to x ≈ -0.3, which this hook
    # admits, but where `log` throws a `DomainError`
    L!(res, x, _) = (res[1] = log(x[1]); nothing)
    x, result = newton_krylov!(L!, [3.0]; linesearch! = AdmissibleLineSearch((u, p) -> u[1] > -1))
    @test result.solved
    @test x[1] ≈ 1
    # and logged with the verbosity of `newton_krylov!`
    @test_logs (:info, r"threw an exception") match_mode = :any newton_krylov!(
        L!, [3.0]; linesearch! = AdmissibleLineSearch((u, p) -> u[1] > -1), verbose = 1
    )

    # The line search can be reused across solves
    ls = AdmissibleLineSearch(positive)
    for _ in 1:2
        u, result = newton_krylov!(log_residual!, copy(u₀), p; forcing = nothing, linesearch! = ls)
        @test result.solved
        @test u ≈ ones(n) rtol = 1.0e-5
    end

    # The hooks get `Ariadne.user_parameters(p)`
    struct Wrapped{P}
        p::P
    end
    Ariadne.user_parameters(w::Wrapped) = w.p
    W!(res, u, w::Wrapped) = log_residual!(res, u, w.p)
    seen = Ref{Any}(nothing)
    hook = (u, q) -> (seen[] = q; positive(u, q))
    u, result = newton_krylov!(W!, copy(u₀), Wrapped(p); forcing = nothing, linesearch! = AdmissibleLineSearch(hook))
    @test result.solved
    @test seen[] === p
end
