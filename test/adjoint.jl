using Test
using Ariadne
import Ariadne: JacobianOperator, BatchedJacobianOperator, Krylov
using Enzyme, LinearAlgebra, SparseArrays

# f(u, p) = A u + u.^3 / 3 - θ₁ b - θ₂ c = 0 with parameters θ in a mutable struct
mutable struct Params
    θ1::Float64
    θ2::Float64
end

const n = 10
const A = sparse(SymTridiagonal(fill(4.0, n), fill(-1.0, n - 1))) + sparse(0.5I, n, n)[:, end:-1:1]
const b = collect(range(1.0, 2.0; length = n))
const c = sin.(1:n)

function f!(res, u, p)
    θ = p.params
    mul!(res, A, u)
    for i in eachindex(res, u)
        res[i] += u[i]^3 / 3 - θ.θ1 * b[i] - θ.θ2 * c[i]
    end
    return nothing
end

jacobian_u(u) = Matrix(A) + Diagonal(u .^ 2)

function newton_solve!(u, p)
    res = similar(u)
    for _ in 1:50
        f!(res, u, p)
        norm(res) < 1.0e-14 && break
        u .-= jacobian_u(u) \ res
    end
    return u
end

objective(u) = sum(abs2, u) / 2 + u[1]

function solution(θ1, θ2)
    u = zeros(n)
    newton_solve!(u, (; params = Params(θ1, θ2)))
    return u
end

@testset "adjoint" begin
    p = (; params = Params(1.0, 0.3))
    u = newton_solve!(zeros(n), p)
    Ju = jacobian_u(u)
    g = u .+ (1:n .== 1)

    @testset "adjoint_solve" begin
        J = JacobianOperator(f!, zeros(n), copy(u), p)
        λ, stats = adjoint_solve(J, g)
        @test stats.solved
        @test λ ≈ Ju' \ g
        # with the transposed LU of the Jacobian as preconditioner
        λ, stats = adjoint_solve(J, g; preconditioner = lu(sparse(Ju)))
        @test stats.solved
        @test stats.niter <= 2
        @test λ ≈ Ju' \ g
        # Krylov.jl measures `rtol` relative to the initial residual, so a warm start pays
        # off with an absolute tolerance
        λ, stats_cold = adjoint_solve(J, g; atol = 1.0e-10, rtol = 0.0)
        λ, stats_warm = adjoint_solve(J, g; λ0 = λ .+ 1.0e-6, atol = 1.0e-10, rtol = 0.0)
        @test stats_warm.solved
        @test stats_warm.niter < stats_cold.niter
        @test λ ≈ Ju' \ g
    end

    @testset "adjoint_solve (workspace)" begin
        J = JacobianOperator(f!, zeros(n), copy(u), p)
        ws = ImplicitFunctionWorkspace(u)
        λ, stats = adjoint_solve(J, g; workspace = ws.adjoint)
        @test stats.solved
        @test λ === Krylov.solution(ws.adjoint)
        @test λ ≈ Ju' \ g
        # The Krylov.jl storage is reused
        @test (@allocated adjoint_solve(J, g; workspace = ws.adjoint)) <
            (@allocated adjoint_solve(J, g))
        @inferred ImplicitFunctionWorkspace(u)
        F = ImplicitFunction(f!, newton_solve!, u)
        @test @inferred(Ariadne.workspace(F, Val(1), u)) === F.workspace
        # another batch width gets a temporary workspace
        @test @inferred(Ariadne.workspace(F, Val(2), u)) isa ImplicitFunctionWorkspace{2}
        @test @inferred(Ariadne.workspace(F, Val(1), zeros(n + 1))) !== F.workspace
    end

    @testset "parameter_vjp" begin
        λ = randn(n)
        p̄ = parameter_vjp(f!, zeros(n), u, p, λ)
        @test p̄.params.θ1 ≈ -dot(λ, b)
        @test p̄.params.θ2 ≈ -dot(λ, c)
    end

    # reference: dJ/dθ = λᵀ (b, c) with Jᵤᵀ λ = g
    λ = Ju' \ g
    dθ_ref = (dot(λ, b), dot(λ, c))
    h = 1.0e-6
    dθ1_fd = (objective(solution(1.0 + h, 0.3)) - objective(solution(1.0 - h, 0.3))) / 2h
    @test dθ1_fd ≈ dθ_ref[1] rtol = 1.0e-6

    @testset "adjoint_gradient" begin
        obj(u, p) = objective(u)
        r = adjoint_gradient(obj, f!, u, p; preconditioner = lu(sparse(Ju)))
        @test r.value ≈ objective(u)
        @test r.stats.solved
        @test r.dp.params.θ1 ≈ dθ_ref[1]
        @test r.dp.params.θ2 ≈ dθ_ref[2]
        # with a Jacobian operator and an already transposed preconditioner
        J = JacobianOperator(f!, zeros(n), copy(u), p)
        r = adjoint_gradient(obj, J, u, p; transposed_preconditioner = transpose(lu(sparse(Ju))))
        @test r.dp.params.θ1 ≈ dθ_ref[1]
        @test r.stats.niter <= 2
        # non-convergence is reported
        r = @test_logs (:warn, r"did not converge") adjoint_gradient(obj, f!, u, p; itmax = 2)
        @test !r.solved
        @test !r.stats.solved
        # warm start
        r = adjoint_gradient(obj, f!, u, p; λ0 = Ju' \ g, atol = 1.0e-10, rtol = 0.0)
        @test r.solved
        @test r.stats.niter == 0
        @test r.dp.params.θ1 ≈ dθ_ref[1]
    end

    @testset "implicit_solve! (reverse)" begin
        F = ImplicitFunction(
            f!, newton_solve!, zeros(n);
            preconditioner = (u, p) -> lu(sparse(jacobian_u(u)))
        )
        function obj(u, p)
            implicit_solve!(F, u, p)
            return objective(u)
        end
        u0 = zeros(n)
        p̄ = Enzyme.make_zero(p)
        autodiff(Reverse, Const(obj), Active, Duplicated(u0, zeros(n)), Duplicated(p, p̄))
        @test p̄.params.θ1 ≈ dθ_ref[1]
        @test p̄.params.θ2 ≈ dθ_ref[2]
        @test Krylov.statistics(F.workspace.adjoint).solved
    end

    @testset "implicit_solve! (reverse, overwritten solution)" begin
        F = ImplicitFunction(f!, newton_solve!, zeros(n))
        function obj(u, p)
            implicit_solve!(F, u, p)
            val = objective(u)
            # The rule must have saved the solution
            fill!(u, 0)
            return val
        end
        p̄ = Enzyme.make_zero(p)
        autodiff(Reverse, Const(obj), Active, Duplicated(zeros(n), zeros(n)), Duplicated(p, p̄))
        @test p̄.params.θ1 ≈ dθ_ref[1]
        @test p̄.params.θ2 ≈ dθ_ref[2]
    end

    @testset "implicit_solve! (reverse, warm start)" begin
        F = ImplicitFunction(
            f!, newton_solve!, zeros(n); warm_start = true,
            adjoint_kwargs = (; atol = 1.0e-10, rtol = 0.0)
        )
        function obj(u, p)
            implicit_solve!(F, u, p)
            return objective(u)
        end
        p̄ = Enzyme.make_zero(p)
        autodiff(Reverse, Const(obj), Active, Duplicated(zeros(n), zeros(n)), Duplicated(p, p̄))
        niter_cold = Krylov.statistics(F.workspace.adjoint).niter
        @test Krylov.solution(F.workspace.adjoint) ≈ λ
        p̄ = Enzyme.make_zero(p)
        autodiff(Reverse, Const(obj), Active, Duplicated(zeros(n), zeros(n)), Duplicated(p, p̄))
        @test Krylov.statistics(F.workspace.adjoint).niter < niter_cold
        @test p̄.params.θ1 ≈ dθ_ref[1]
        @test p̄.params.θ2 ≈ dθ_ref[2]
    end

    @testset "implicit_solve! (reverse, dynamic dispatch)" begin
        # A non-constant global makes the call type unstable, so that Enzyme.jl passes the
        # parameters to the rule as `MixedDuplicated`
        global F_dynamic = ImplicitFunction(f!, newton_solve!, zeros(n))
        function obj_dynamic(u, p)
            implicit_solve!(F_dynamic, u, p)
            return objective(u)
        end
        p̄ = Enzyme.make_zero(p)
        autodiff(Reverse, obj_dynamic, Active, Duplicated(zeros(n), zeros(n)), Duplicated(p, p̄))
        @test p̄.params.θ1 ≈ dθ_ref[1]
        @test p̄.params.θ2 ≈ dθ_ref[2]
    end

    @testset "implicit_solve! (forward)" begin
        F = ImplicitFunction(f!, newton_solve!, zeros(n))
        u0 = zeros(n)
        du = zeros(n)
        autodiff(
            Forward, implicit_solve!, Const(F), Duplicated(u0, du),
            Duplicated(p, (; params = Params(1.0, 0.0)))
        )
        @test u0 ≈ u
        @test du ≈ Ju \ b
        @test dot(g, du) ≈ dθ_ref[1]
    end

    @testset "implicit_solve! (forward, warm start)" begin
        F = ImplicitFunction(
            f!, newton_solve!, zeros(n); warm_start = true,
            adjoint_kwargs = (; atol = 1.0e-10, rtol = 0.0)
        )
        ṗ = (; params = Params(1.0, 0.0))
        du = zeros(n)
        autodiff(Forward, implicit_solve!, Const(F), Duplicated(zeros(n), du), Duplicated(p, ṗ))
        niter_cold = Krylov.statistics(F.workspace.tangent).niter
        @test Krylov.solution(F.workspace.tangent) ≈ Ju \ b
        du = zeros(n)
        autodiff(Forward, implicit_solve!, Const(F), Duplicated(zeros(n), du), Duplicated(p, ṗ))
        @test Krylov.statistics(F.workspace.tangent).niter < niter_cold
        @test du ≈ Ju \ b
    end

    if VERSION >= v"1.11.0"
        # A second functional that depends on the parameters explicitly
        functional2(u, p) = p.params.θ1 * sum(u)
        function functionals!(out, u, p)
            out[1] = objective(u)
            out[2] = functional2(u, p)
            return nothing
        end
        g2 = fill(p.params.θ1, n)
        λ2 = Ju' \ g2
        dθ2_ref = (sum(u) + dot(λ2, b), dot(λ2, c))

        @testset "adjoint_solve (batched)" begin
            J = BatchedJacobianOperator{2}(f!, zeros(n), copy(u), p)
            G = [g g2]
            Λ, stats = adjoint_solve(J, G)
            @test stats.solved
            @test Λ ≈ Ju' \ G
            Λ, stats = adjoint_solve(J, G; preconditioner = lu(sparse(Ju)))
            @test stats.solved
            @test stats.niter <= 2
            @test Λ ≈ Ju' \ G
            Λ, stats = adjoint_solve(J, G; λ0 = Ju' \ G, atol = 1.0e-10, rtol = 0.0)
            @test stats.solved
            @test stats.niter == 0
            @test Λ ≈ Ju' \ G
            @test_throws DimensionMismatch adjoint_solve(J, g[:, :])
        end

        @testset "parameter_vjp (batched)" begin
            λs = (randn(n), randn(n))
            p̄s = parameter_vjp!((Enzyme.make_zero(p), Enzyme.make_zero(p)), f!, zeros(n), u, p, λs)
            for (p̄, λ) in zip(p̄s, λs)
                @test p̄.params.θ1 ≈ -dot(λ, b)
                @test p̄.params.θ2 ≈ -dot(λ, c)
            end
        end

        @testset "adjoint_gradient (multiple functionals)" begin
            r = adjoint_gradient(functionals!, f!, u, p, Val(2); preconditioner = lu(sparse(Ju)))
            @test r.solved
            @test r.value ≈ [objective(u), functional2(u, p)]
            @test r.dJdu ≈ [g g2]
            @test r.λ ≈ [λ λ2]
            @test r.dp[1].params.θ1 ≈ dθ_ref[1]
            @test r.dp[1].params.θ2 ≈ dθ_ref[2]
            @test r.dp[2].params.θ1 ≈ dθ2_ref[1]
            @test r.dp[2].params.θ2 ≈ dθ2_ref[2]
            # same as one call per functional
            r1 = adjoint_gradient((u, p) -> objective(u), f!, u, p)
            r2 = adjoint_gradient(functional2, f!, u, p)
            @test r.dp[1].params.θ1 ≈ r1.dp.params.θ1
            @test r.dp[2].params.θ1 ≈ r2.dp.params.θ1
            @test r.dp[2].params.θ2 ≈ r2.dp.params.θ2
            # non-convergence is reported
            r = @test_logs (:warn, r"did not converge") adjoint_gradient(functionals!, f!, u, p, Val(2); itmax = 1)
            @test !r.solved
        end

        @testset "implicit_solve! (batched reverse)" begin
            F = ImplicitFunction(
                f!, newton_solve!, zeros(n), Val(2);
                preconditioner = (u, p) -> lu(sparse(jacobian_u(u)))
            )
            function obj!(out, u, p)
                implicit_solve!(F, u, p)
                functionals!(out, u, p)
                return nothing
            end
            p̄s = (Enzyme.make_zero(p), Enzyme.make_zero(p))
            autodiff(
                Reverse, Const(obj!), Const, BatchDuplicated(zeros(2), ([1.0, 0.0], [0.0, 1.0])),
                BatchDuplicated(zeros(n), (zeros(n), zeros(n))), BatchDuplicated(p, p̄s)
            )
            @test Krylov.statistics(F.workspace.adjoint).solved
            @test p̄s[1].params.θ1 ≈ dθ_ref[1]
            @test p̄s[1].params.θ2 ≈ dθ_ref[2]
            @test p̄s[2].params.θ1 ≈ dθ2_ref[1]
            @test p̄s[2].params.θ2 ≈ dθ2_ref[2]
        end

        @testset "implicit_solve! (batched reverse, dynamic dispatch)" begin
            global F_dynamic2 = ImplicitFunction(f!, newton_solve!, zeros(n), Val(2))
            function obj_dynamic!(out, u, p)
                implicit_solve!(F_dynamic2, u, p)
                functionals!(out, u, p)
                return nothing
            end
            p̄s = (Enzyme.make_zero(p), Enzyme.make_zero(p))
            autodiff(
                Reverse, obj_dynamic!, Const, BatchDuplicated(zeros(2), ([1.0, 0.0], [0.0, 1.0])),
                BatchDuplicated(zeros(n), (zeros(n), zeros(n))), BatchDuplicated(p, p̄s)
            )
            @test p̄s[1].params.θ1 ≈ dθ_ref[1]
            @test p̄s[2].params.θ2 ≈ dθ2_ref[2]
        end

        @testset "implicit_solve! (batched forward)" begin
            F = ImplicitFunction(f!, newton_solve!, zeros(n), Val(2))
            u0 = zeros(n)
            du = (zeros(n), zeros(n))
            autodiff(
                Forward, implicit_solve!, Const(F), BatchDuplicated(u0, du),
                BatchDuplicated(p, ((; params = Params(1.0, 0.0)), (; params = Params(0.0, 1.0))))
            )
            @test Krylov.statistics(F.workspace.tangent).solved
            @test u0 ≈ u
            @test du[1] ≈ Ju \ b
            @test du[2] ≈ Ju \ c
        end

        @testset "implicit_solve! (batched, other width than the workspace)" begin
            F = ImplicitFunction(f!, newton_solve!, zeros(n))
            du = (zeros(n), zeros(n))
            autodiff(
                Forward, implicit_solve!, Const(F), BatchDuplicated(zeros(n), du),
                BatchDuplicated(p, ((; params = Params(1.0, 0.0)), (; params = Params(0.0, 1.0))))
            )
            @test du[1] ≈ Ju \ b
            @test du[2] ≈ Ju \ c
            @test !F.workspace.tangent_solved
        end

        @testset "implicit_solve! (batched, warm start)" begin
            F = ImplicitFunction(
                f!, newton_solve!, zeros(n), Val(2); warm_start = true,
                adjoint_kwargs = (; atol = 1.0e-10, rtol = 0.0)
            )
            ṗs = ((; params = Params(1.0, 0.0)), (; params = Params(0.0, 1.0)))
            for _ in 1:2
                autodiff(
                    Forward, implicit_solve!, Const(F),
                    BatchDuplicated(zeros(n), (zeros(n), zeros(n))), BatchDuplicated(p, ṗs)
                )
            end
            # The second solve starts from the solution of the first one
            @test Krylov.statistics(F.workspace.tangent).niter == 0
            @test Krylov.solution(F.workspace.tangent) ≈ Ju \ [b c]
            function obj!(out, u, p)
                implicit_solve!(F, u, p)
                functionals!(out, u, p)
                return nothing
            end
            for _ in 1:2
                autodiff(
                    Reverse, Const(obj!), Const, BatchDuplicated(zeros(2), ([1.0, 0.0], [0.0, 1.0])),
                    BatchDuplicated(zeros(n), (zeros(n), zeros(n))),
                    BatchDuplicated(p, (Enzyme.make_zero(p), Enzyme.make_zero(p)))
                )
            end
            @test Krylov.statistics(F.workspace.adjoint).niter == 0
            @test Krylov.solution(F.workspace.adjoint) ≈ [λ λ2]
        end
    end
end
