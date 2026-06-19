using Test
using Ariadne
import Ariadne: SplitJacobianOperator, prepare!
using Enzyme, LinearAlgebra

# Skip tests if ForwardSplitNoPrimal is not available
const FORWARD_SPLIT_AVAILABLE = isdefined(Enzyme, :ForwardSplitNoPrimal)

function F!(res, x, _)
    res[1] = x[1]^2 + x[2]^2 - 2
    res[2] = exp(x[1] - 1) + x[2]^2 - 2
    return nothing
end

function F(x, p)
    res = similar(x)
    F!(res, x, p)
    return res
end

@testset "SplitJacobianOperator" begin
    if !FORWARD_SPLIT_AVAILABLE
        @test_skip "ForwardSplitNoPrimal not available, skipping SplitJacobianOperator tests"
        return
    end

    @testset "constructor" begin
        x = [3.0, 5.0]
        res = zeros(2)
        p = nothing

        J_split = SplitJacobianOperator(F!, res, x, p)

        @test size(J_split) == (2, 2)
        @test length(J_split) == 4
        @test eltype(J_split) == Float64
        @test J_split.tape[] === nothing  # not prepared yet
    end

    @testset "prepare! and mul!" begin
        x = [3.0, 5.0]
        res = zeros(2)
        p = nothing

        J_split = SplitJacobianOperator(F!, res, x, p)

        # Before prepare!, res should be zeros
        @test res == [0.0, 0.0]

        # After prepare!, res should contain F!(res, x, p) and tape should be stored
        prepare!(J_split)
        expected_res = F(x, p)
        @test res ≈ expected_res
        @test J_split.tape[] !== nothing

        # Test JVP computation
        v = [1.0, 0.0]
        out = zeros(2)
        mul!(out, J_split, v)

        # Compare with regular JacobianOperator result
        J_regular = Ariadne.JacobianOperator(F!, copy(res), x, p)
        out_regular = zeros(2)
        mul!(out_regular, J_regular, v)

        @test out ≈ out_regular
    end

    @testset "consistency with JacobianOperator" begin
        x = [3.0, 5.0]
        res = zeros(2)
        p = nothing

        # Create both operators
        J_split = SplitJacobianOperator(F!, res, x, p)
        prepare!(J_split)

        res_regular = copy(res)
        J_regular = Ariadne.JacobianOperator(F!, res_regular, x, p)

        # Test on multiple directions
        for v in ([1.0, 0.0], [0.0, 1.0], [1.0, 1.0], rand(2))
            out_split = zeros(2)
            out_regular = zeros(2)

            mul!(out_split, J_split, v)
            mul!(out_regular, J_regular, v)

            @test out_split ≈ out_regular rtol = 1.0e-12
        end
    end

    @testset "collect" begin
        x = [3.0, 5.0]
        res = zeros(2)
        p = nothing

        J_split = SplitJacobianOperator(F!, res, x, p)
        prepare!(J_split)

        # Compare collected matrix with Enzyme jacobian
        J_matrix_split = collect(J_split)
        J_matrix_enz = jacobian(Forward, x -> F(x, nothing), x) |> only

        @test J_matrix_split ≈ J_matrix_enz rtol = 1.0e-12
    end

    @testset "adjoint/transpose error" begin
        x = [3.0, 5.0]
        res = zeros(2)
        p = nothing

        J_split = SplitJacobianOperator(F!, res, x, p)
        prepare!(J_split)

        v = [1.0, 0.0]
        out = zeros(2)

        # Adjoint and transpose should throw errors
        @test_throws ErrorException mul!(out, adjoint(J_split), v)
        @test_throws ErrorException mul!(out, transpose(J_split), v)
    end

    @testset "parameter handling" begin
        # Test with actual parameters
        function F_param!(res, x, p)
            res[1] = x[1]^2 + x[2]^2 - p[1]
            res[2] = exp(x[1] - 1) + x[2]^2 - p[2]
            return nothing
        end

        x = [3.0, 5.0]
        res = zeros(2)
        p = [2.0, 2.0]

        # Test with assume_p_const=true
        J_split = SplitJacobianOperator(F_param!, res, x, p; assume_p_const = true)
        @test J_split.p′ === nothing

        prepare!(J_split)
        @test res ≈ [3.0^2 + 5.0^2 - 2.0, exp(3.0 - 1) + 5.0^2 - 2.0]

        # Test with assume_p_const=false (if p is not guaranteed const by Enzyme)
        if !Enzyme.Compiler.guaranteed_const(typeof(p))
            J_split2 = SplitJacobianOperator(F_param!, copy(res), x, p; assume_p_const = false)
            @test J_split2.p′ !== nothing
        end
    end
end
