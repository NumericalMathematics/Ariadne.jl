using Test
using Ariadne

function F!(res, x, _)
    res[1] = x[1]^2 + x[2]^2 - 2
    return res[2] = exp(x[1] - 1) + x[2]^2 - 2
end

function F(x, p)
    res = similar(x)
    F!(res, x, p)
    return res
end

let x₀ = [2.0, 0.5]
    x, stats = newton_krylov!(F!, x₀)
    @test stats.solved
end

let x₀ = [3.0, 5.0]
    x, stats = newton_krylov(F, x₀)
    @test stats.solved
end

@testset "Float32" begin
    x, result = newton_krylov!(F!, Float32[2, 0.5])
    @test result.solved
    @test eltype(x) === Float32
    @test result.stats.norm_res isa Float32
    @test x ≈ [1, 1] atol = 1.0f-3

    x, result = newton_krylov!(F!, Float32[2, 0.5]; forcing = nothing)
    @test result.solved
    @test eltype(x) === Float32
end

import Ariadne: JacobianOperator, BatchedJacobianOperator
using Enzyme, LinearAlgebra, SparseArrays

@testset "Jacobian" begin
    J_Enz = jacobian(Forward, x -> F(x, nothing), [3.0, 5.0]) |> only
    J = JacobianOperator(F!, zeros(2), [3.0, 5.0], nothing)

    @test size(J) == (2, 2)
    @test length(J) == 4
    @test eltype(J) == Float64

    out = [NaN, NaN]
    mul!(out, J, [1.0, 0.0])
    @test out == [6.0, 7.38905609893065]

    out = [NaN, NaN]
    mul!(out, transpose(J), [1.0, 0.0])
    @test out == [6.0, 10.0]

    J_NK = collect(J)

    @test J_NK == J_Enz

    v = rand(2)
    out = similar(v)
    mul!(out, J, v)

    @test out ≈ J_Enz * v

    @test collect(transpose(J)) == transpose(collect(J))

    # Batched
    if VERSION >= v"1.11.0"
        J = BatchedJacobianOperator{2}(F!, zeros(2), [3.0, 5.0], nothing)

        V = [1.0 0.0; 0.0 1.0]
        Out = similar(V)
        mul!(Out, J, V)

        @test Out == J_Enz

        mul!(Out, transpose(J), V)
        @test Out == J_Enz'
        @test collect(J) == J_Enz
        @test collect(transpose(J)) == J_Enz'
    end
end

@testset "collect" begin
    # Non-square Jacobian whose number of columns is not a multiple of the batch size
    G!(res, x, _) = (res[1] = x[1] * x[2]; res[2] = x[2] + x[3]^2; nothing)
    x = [1.0, 2.0, 3.0]
    J_ref = [2.0 1.0 0.0; 0.0 1.0 6.0]
    J = JacobianOperator(G!, zeros(2), x, nothing)
    @test size(J) == (2, 3)
    @test collect(J) == J_ref
    @test collect(transpose(J)) == J_ref'
    @test collect(J) isa SparseMatrixCSC{Float64, Int}

    J = JacobianOperator(G!, zeros(Float32, 2), Float32.(x), nothing)
    @test collect(J) isa SparseMatrixCSC{Float32, Int}
    @test collect(J) == J_ref

    if VERSION >= v"1.11.0"
        J = BatchedJacobianOperator{2}(G!, zeros(2), x, nothing)
        @test collect(J) == J_ref
        @test collect(transpose(J)) == J_ref'
    end
end

@testset "Jacobian of accumulating residual" begin
    # The output buffer of the JVP is the shadow of `res`. A residual that
    # accumulates into `res` must not pick up stale values from it.
    A!(res, x, _) = (res .+= x .^ 2; nothing)
    J = JacobianOperator(A!, zeros(2), [1.0, 2.0], nothing)
    out = [100.0, 100.0]
    mul!(out, J, [1.0, 0.0])
    @test out == [2.0, 0.0]

    if VERSION >= v"1.11.0"
        J = BatchedJacobianOperator{2}(A!, zeros(2), [1.0, 2.0], nothing)
        Out = fill(100.0, 2, 2)
        mul!(Out, J, [1.0 0.0; 0.0 1.0])
        @test Out == [2.0 0.0; 0.0 4.0]
    end
end
