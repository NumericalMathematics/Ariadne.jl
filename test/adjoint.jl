using Test
using Ariadne
import Ariadne: JacobianOperator
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
    end

    @testset "implicit_solve! (reverse)" begin
        F = ImplicitFunction(
            f!, newton_solve!;
            preconditioner = (u, p) -> lu(sparse(jacobian_u(u)))
        )
        function obj(u, p)
            implicit_solve!(F, u, p)
            return objective(u)
        end
        u0 = zeros(n)
        p̄ = Enzyme.make_zero(p)
        autodiff(Reverse, obj, Active, Duplicated(u0, zeros(n)), Duplicated(p, p̄))
        @test p̄.params.θ1 ≈ dθ_ref[1]
        @test p̄.params.θ2 ≈ dθ_ref[2]
        @test F.last_stats[].solved
    end

    @testset "implicit_solve! (reverse, dynamic dispatch)" begin
        # A non-constant global makes the call type unstable, so that Enzyme.jl passes the
        # parameters to the rule as `MixedDuplicated`
        global F_dynamic = ImplicitFunction(f!, newton_solve!)
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
        F = ImplicitFunction(f!, newton_solve!)
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
end
