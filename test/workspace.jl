using Test
using Ariadne

function F!(res, x, _)
    res[1] = x[1]^2 + x[2]^2 - 2
    res[2] = exp(x[1] - 1) + x[2]^2 - 2
    return nothing
end

@testset "NewtonKrylovWorkspace" begin

    @testset "correctness" begin
        # Workspace API must give the same solution as the convenience API
        x_ref = [2.0, 0.5]
        _, stats_ref = newton_krylov!(F!, x_ref)
        @test stats_ref.solved

        # When the same array used to construct the workspace is passed in,
        # the returned solution aliases it directly (no hidden copies).
        x₀ = [2.0, 0.5]
        ws = NewtonKrylovWorkspace(F!, x₀)
        x_sol, stats_ws = newton_krylov!(ws, x₀, nothing)
        @test stats_ws.solved
        @test x_sol ≈ x_ref
        @test x_sol === x₀
    end

    @testset "reuse" begin
        # Workspace can be reused across multiple solves.
        # A different initial-guess array is copied into ws.u automatically.
        ws = NewtonKrylovWorkspace(F!, [2.0, 0.5])

        for x₀ in ([2.0, 0.5], [0.5, 2.0], [1.5, 0.8])
            _, stats = newton_krylov!(ws, x₀, nothing)
            @test stats.solved
        end
    end

    @testset "reduced allocations" begin
        kw = (; history = false)
        cb = (u, res, n) -> nothing

        # Non-workspace API: creates JacobianOperator, Krylov workspace, and
        # neg_res buffer on every call.
        x₀ = [2.0, 0.5]
        newton_krylov!(F!, x₀; krylov_kwargs = kw, callback = cb)  # warmup
        x₀ .= [2.0, 0.5]
        allocs_noworkspace = @allocated newton_krylov!(F!, x₀; krylov_kwargs = kw, callback = cb)

        # Workspace API: all buffers pre-allocated; only Krylov.jl's internal
        # SimpleStats.status String update allocates (~80 bytes × outer iters).
        ws = NewtonKrylovWorkspace(F!, [2.0, 0.5])
        x_init = [2.0, 0.5]
        newton_krylov!(ws, x_init, nothing; krylov_kwargs = kw, callback = cb)  # warmup
        allocs_workspace = @allocated newton_krylov!(ws, x_init, nothing; krylov_kwargs = kw, callback = cb)

        @test allocs_workspace < allocs_noworkspace
        # The only remaining allocation comes from Krylov.jl's SimpleStats.status
        # String update, which is proportional to the number of outer iterations.
        @test allocs_workspace < 1500
    end

end
