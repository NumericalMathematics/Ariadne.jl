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

@testset "verbose log" begin
    logs, (x, result) = Test.collect_test_logs() do
        newton_krylov!(F!, [2.0, 0.5]; verbose = 1)
    end
    @test result.solved
    newton_logs = filter(l -> l.message == "Newton", logs)
    @test length(newton_logs) == result.stats.outer_iterations
    for (i, l) in enumerate(newton_logs)
        @test l.kwargs[:iter] == i
        @test l.kwargs[:norm_res] == l.kwargs[:stats].norm_res
    end
    @test last(newton_logs).kwargs[:norm_res] == result.stats.norm_res
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

@testset "Jacobian with mutable parameters" begin
    C!(res, u, p) = (res .= p.c .* u .^ 2; nothing)
    # A scratch buffer in `p` that is written before it is read
    S!(res, u, p) = (p.tmp .= u .^ 2; res .= p.c .* p.tmp; nothing)
    u = [1.0, 2.0]
    v = [3.0, 5.0]
    w = [7.0, 11.0]
    V = [v 2v]
    W = [w 2w]
    p = (c = [2.0, 3.0], tmp = zeros(2))
    jvp = 2 .* p.c .* u .* v
    vjp = 2 .* p.c .* u .* w

    for lazy_zero_shadows in (false, true)
        # `p.c` is read by the residual, so after a reverse-mode product its shadow holds
        # adjoints. These must not leak into the tangent of a following forward-mode product.
        J = JacobianOperator(C!, zeros(2), copy(u), p; lazy_zero_shadows)
        out = zeros(2)
        mul!(out, J, v)
        @test out ≈ jvp
        mul!(out, transpose(J), w)
        @test out ≈ vjp
        @test J.p′.c ≈ u .^ 2 .* w # adjoint of `p.c`
        mul!(out, J, v)
        @test out ≈ jvp

        J = JacobianOperator(S!, zeros(2), copy(u), p; lazy_zero_shadows)
        for k in 1:3
            mul!(out, J, k .* v)
            @test out ≈ k .* jvp
            mul!(out, transpose(J), w)
            @test out ≈ vjp
        end
        for k in 1:3
            mul!(out, J, k .* v)
            @test out ≈ k .* jvp
        end

        if VERSION >= v"1.11.0"
            for F! in (C!, S!)
                J = BatchedJacobianOperator{2}(F!, zeros(2), copy(u), p; lazy_zero_shadows)
                Out = zeros(2, 2)
                for k in 1:3
                    mul!(Out, J, k .* V)
                    @test Out ≈ k .* [jvp 2jvp]
                    mul!(Out, transpose(J), W)
                    @test Out ≈ 2 .* p.c .* u .* W
                end
                mul!(Out, J, V)
                @test Out ≈ [jvp 2jvp]
            end
        end
    end

    # A scratch buffer in `p` that is accumulated into without being reset, so the
    # residual depends on its history. The tangent of `p.tmp` from the previous product
    # must not leak into the next one, so the shadows are zeroed before every product
    # by default.
    A!(res, u, p) = (p.tmp .+= u .^ 2; res .= p.tmp; nothing)
    p = (tmp = zeros(2),)
    J = JacobianOperator(A!, zeros(2), copy(u), p)
    out = zeros(2)
    mul!(out, J, v)
    @test out ≈ 2 .* u .* v
    mul!(out, transpose(J), w)
    @test out ≈ 2 .* u .* w
    for k in 1:3
        mul!(out, J, k .* v)
        @test out ≈ 2 .* u .* k .* v
    end
    if VERSION >= v"1.11.0"
        J = BatchedJacobianOperator{2}(A!, zeros(2), copy(u), p)
        Out = zeros(2, 2)
        mul!(Out, J, V)
        @test Out ≈ 2 .* u .* V
        mul!(Out, transpose(J), W)
        @test Out ≈ 2 .* u .* W
        for k in 1:3
            mul!(Out, J, k .* V)
            @test Out ≈ 2 .* u .* k .* V
        end
    end
end

@testset "Jacobian shadows are zeroed lazily" begin
    C!(res, u, p) = (res .= p.c .* u .^ 2; nothing)
    u = [1.0, 2.0]
    v = [3.0, 5.0]
    p = (c = [2.0, 3.0], unused = [0.0])
    out = zeros(2)

    # The shadow of the unused `p.unused` is neither read nor written by Enzyme,
    # so a sentinel there shows whether the shadows were zeroed.
    J = JacobianOperator(C!, zeros(2), copy(u), p)
    @test !J.lazy_zero_shadows
    J.p′.unused[1] = 42
    mul!(out, J, v)
    @test J.p′.unused[1] == 0

    J = JacobianOperator(C!, zeros(2), copy(u), p; lazy_zero_shadows = true)
    @test !J.dirty[]
    J.p′.unused[1] = 42
    mul!(out, J, v)
    @test !J.dirty[]
    @test J.p′.unused[1] == 42
    mul!(out, transpose(J), v)
    @test J.dirty[]
    @test J.p′.unused[1] == 0
    J.p′.unused[1] = 42
    mul!(out, J, v)
    @test !J.dirty[]
    @test J.p′.unused[1] == 0

    if VERSION >= v"1.11.0"
        V = [v v]
        Out = zeros(2, 2)
        J = BatchedJacobianOperator{2}(C!, zeros(2), copy(u), p)
        J.p′[1].unused[1] = 42
        mul!(Out, J, V)
        @test J.p′[1].unused[1] == 0

        J = BatchedJacobianOperator{2}(C!, zeros(2), copy(u), p; lazy_zero_shadows = true)
        @test !J.dirty[]
        J.p′[1].unused[1] = 42
        J.p′[2].unused[1] = 42
        mul!(Out, J, V)
        @test !J.dirty[]
        @test J.p′[1].unused[1] == 42 && J.p′[2].unused[1] == 42
        mul!(Out, transpose(J), V)
        @test J.dirty[]
        @test J.p′[1].unused[1] == 0 && J.p′[2].unused[1] == 0
        J.p′[1].unused[1] = 42
        mul!(Out, J, V)
        @test !J.dirty[]
        @test J.p′[1].unused[1] == 0
    end

    # Passed through by `NewtonKrylovWorkspace`
    F!(res, u, p) = (res .= u .^ 2 .- 4; nothing)
    ws = NewtonKrylovWorkspace(F!, [1.0, 1.0], nothing, zeros(2); lazy_zero_shadows = true)
    @test ws.J.lazy_zero_shadows
end
