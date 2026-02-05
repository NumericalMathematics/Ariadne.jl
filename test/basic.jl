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

import Ariadne: JacobianOperator
using Enzyme, LinearAlgebra
using ADTypes

@testset "Enzyme: JacobianOperator" begin
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
end

@testset "DifferentiationInterface: JacobianOperator" begin
    backend = ADTypes.AutoEnzyme()
    J = Ariadne.DIJacobianOperator(backend, F!, zeros(2), [3.0, 5.0], nothing)

    @test size(J) == (2, 2)
    @test length(J) == 4
    @test eltype(J) == Float64

    out = [NaN, NaN]
    mul!(out, J, [1.0, 0.0])
    @test out == [6.0, 7.38905609893065]
end
