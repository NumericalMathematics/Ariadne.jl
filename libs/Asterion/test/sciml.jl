using Test
using Ariadne
using Asterion
using SciMLBase
using SciMLBase: ReturnCode, SteadyStateProblem, NonlinearProblem, successful_retcode
using LinearAlgebra
using SparseArrays

function bratu!(du, u, p, t)
    (; λ, h) = p
    n = length(u)
    for i in 1:n
        ul = i > 1 ? u[i - 1] : zero(eltype(u))
        ur = i < n ? u[i + 1] : zero(eltype(u))
        du[i] = (ul - 2 * u[i] + ur) / h^2 + λ * exp(u[i])
    end
    return nothing
end
bratu(u, p, t) = (du = similar(u); bratu!(du, u, p, t); du)

function atan_residual!(res, u, p)
    n = length(u)
    for i in 1:n
        ul = i > 1 ? u[i - 1] : zero(eltype(u))
        ur = i < n ? u[i + 1] : zero(eltype(u))
        res[i] = atan(u[i] - p.c[i]) - p.ε * (ul - 2 * u[i] + ur)
    end
    return nothing
end

@testset "SteadyStateProblem" begin
    n = 32
    p = (; λ = 1.0, h = 1 / (n + 1))
    u0 = zeros(n)
    alg = PseudoTransientNewtonKrylov(; cfl = SER(; initial = 1.0e-3))
    prob = SteadyStateProblem(bratu!, u0, p)
    sol = solve(prob, alg; reltol = 1.0e-10)
    @test sol.retcode == ReturnCode.Success
    @test successful_retcode(sol)
    @test sol.u !== u0 # alias_u0 = false by default
    @test all(iszero, u0)
    du = similar(u0)
    bratu!(du, sol.u, p, Inf)
    @test sol.resid ≈ du
    @test norm(du) <= 1.0e-10 * sol.original.stats.norm_res_initial
    @test sol.stats.nsteps == sol.original.stats.steps
    @test sol.stats.nsolve == sol.original.stats.newton_iterations
    @test sol.original isa PseudoTransientWorkspace
    @test sol.original.newton.J isa Ariadne.JacobianOperator
    @test sol.prob === prob
    @test sol.alg === alg

    # out-of-place
    sol_oop = solve(SteadyStateProblem(bratu, u0, p), alg; reltol = 1.0e-10)
    @test sol_oop.retcode == ReturnCode.Success
    @test sol_oop.u ≈ sol.u

    # aliasing
    u = zeros(n)
    sol = solve(SteadyStateProblem(bratu!, u, p), alg; alias_u0 = true, reltol = 1.0e-10)
    @test sol.u === u
    u = zeros(n)
    sol = solve(SteadyStateProblem(bratu!, u, p), alg; alias = SciMLBase.NonlinearAliasSpecifier(; alias_u0 = true))
    @test sol.u === u

    # maxiters, abstol, callbacks
    sol = solve(prob, alg; maxiters = 2)
    @test sol.retcode == ReturnCode.MaxIters
    @test sol.stats.nsteps == 2
    sol = solve(prob, alg; abstol = 1.0e-3, reltol = 0.0)
    @test sol.retcode == ReturnCode.Success
    @test norm(sol.resid) <= 1.0e-3
    sol = solve(prob, alg; callback = (ws, info) -> info.step >= 3)
    @test sol.retcode == ReturnCode.Terminated
    @test sol.stats.nsteps == 3

    # init / solve!
    cache = init(prob, alg; reltol = 1.0e-10)
    sol = solve!(cache)
    @test sol.retcode == ReturnCode.Success

    # with the assembled preconditioner
    pattern = spdiagm(-1 => ones(n - 1), 0 => ones(n), 1 => ones(n - 1))
    alg_prec = PseudoTransientNewtonKrylov(;
        cfl = SER(; initial = 1.0e-3),
        preconditioner = AssembledJacobianPreconditioner(; sparsity = pattern, refresh_interval = 10)
    )
    sol_prec = solve(prob, alg_prec; reltol = 1.0e-10)
    @test sol_prec.retcode == ReturnCode.Success
    @test sol_prec.u ≈ sol_oop.u rtol = 1.0e-6
    @test sol_prec.stats.njacs >= 1

    @test_throws ArgumentError solve(prob, alg; termination_condition = :foo)
    @test_logs (:warn, r"unsupported") solve(prob, alg; maxiters = 1, foo = 1)
end

@testset "NonlinearProblem" begin
    n = 16
    p = (; c = collect(range(-1, 1; length = n)), ε = 0.01)
    u0 = fill(10.0, n)
    alg = PseudoTransientNewtonKrylov(; cfl = SER(; initial = 1.0, growth_max = 10.0))
    sol = solve(NonlinearProblem(atan_residual!, u0, p), alg; reltol = 1.0e-12)
    @test sol.retcode == ReturnCode.Success
    res = similar(u0)
    atan_residual!(res, sol.u, p)
    @test norm(res) < 1.0e-10
    @test sol.resid ≈ res

    # out-of-place
    F(u, p) = (r = similar(u); atan_residual!(r, u, p); r)
    sol2 = solve(NonlinearProblem(F, u0, p), alg; reltol = 1.0e-12)
    @test sol2.u ≈ sol.u

    # failure: no admissible state
    alg_fail = PseudoTransientNewtonKrylov(; cfl = SER(; min = 1.0e-2), isadmissible = (u, p) -> false)
    sol = solve(NonlinearProblem(atan_residual!, u0, p), alg_fail)
    @test sol.retcode == ReturnCode.ConvergenceFailure
    @test !successful_retcode(sol)
    @test sol.u == u0
end

@testset "init and step!" begin
    n = 32
    p = (; λ = 1.0, h = 1 / (n + 1))
    alg = PseudoTransientNewtonKrylov(; cfl = SER(; initial = 1.0e-3))
    prob = SteadyStateProblem(bratu!, zeros(n), p)
    sol = solve(prob, alg; reltol = 1.0e-10)

    cache = init(prob, alg; reltol = 1.0e-10)
    @test cache isa Asterion.PseudoTransientCache
    ws = cache.ws
    @test ws.status === :initialized
    nsteps = 0
    while true
        @test step!(cache) === :accepted
        nsteps += 1
        ws.norm_res <= 1.0e-10 * ws.norm_res_initial && break
        @test nsteps < 1000
    end
    @test nsteps == ws.stats.steps == sol.original.stats.steps
    @test ws.u ≈ sol.u
    @test length(ws.history) == nsteps + 1
end
