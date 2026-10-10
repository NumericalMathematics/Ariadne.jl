# MixedPrecisionLU and the refinement workspaces on arrays without scalar indexing
using Test
using Ariadne
using Ariadne: MixedPrecisionLU, lu_in_precision!, IterativeRefinementWorkspace,
    GMRESIRWorkspace
using Ariadne: krylov_solve!
using LinearAlgebra
using JLArrays

JLArrays.allowscalar(false)

@testset "array_lu! on JLArray" begin
    A = rand(48, 48) + 8I
    for T in (Float64, Float32, Float16)
        F = lu_in_precision!(JLArray(Matrix{T}(A)))
        G = LinearAlgebra.generic_lufact!(Matrix{T}(A))
        # Same operations per entry as the generic LU
        @test Array(F.factors) == G.factors
        @test F.ipiv == G.ipiv
    end
end

function check_solves(method)
    n = 48
    A = rand(n, n) + 8I
    b = rand(n)
    x = A \ b
    P = MixedPrecisionLU(JLArray(Matrix{Float32}(A)); factor_precision = Float16, triangular_solves = method)
    @test norm(Array(P \ JLArray(b)) - x) / norm(x) < 1.0e-2
    Q = MixedPrecisionLU(
        JLArray(Matrix{Float32}(A)); factor_precision = Float16, solve_precision = Float32,
        triangular_solves = method
    )
    # Three-precision refinement recovers single precision
    for ws in (IterativeRefinementWorkspace(JLArray(b)), GMRESIRWorkspace(JLArray(b)))
        krylov_solve!(ws, nothing, JLArray(b); N = Q)
        @test ws.stats.solved
        @test norm(Array(ws.x) - x) / norm(x) < 1.0e-5
    end
    # GMRES-IR with the inner GMRES in Float32 and a Float64 working precision
    R = MixedPrecisionLU(JLArray(A); factor_precision = Float16, solve_precision = Float32, triangular_solves = method)
    ws = GMRESIRWorkspace(JLArray(b); gmres_precision = Float32)
    krylov_solve!(ws, nothing, JLArray(b); N = R)
    @test ws.stats.solved
    @test norm(Array(ws.x) - x) / norm(x) < 1.0e-12
    return nothing
end

@testset "MixedPrecisionLU on JLArray, array solves" begin
    check_solves(:array)
end

using NextLA

@testset "MixedPrecisionLU on JLArray, NextLA solves" begin
    @test Ariadne.default_triangular_solves(JLArray(zeros(2, 2))) === :nextla
    @test Ariadne.default_triangular_solves(zeros(2, 2)) === :stdlib
    check_solves(:nextla)
end
