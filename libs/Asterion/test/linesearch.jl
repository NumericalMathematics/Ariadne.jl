using Test
using Ariadne
using Asterion

# F(u) = log|u| - log a has the physical root u = a > 0 and the unphysical root u = -a
log_residual!(res, u, p) = (@. res = log(abs(u)) - log(p.a); nothing)
positive(u, p) = all(>(0), u)

@testset "AdmissibleBacktrackingLineSearch" begin
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
        linesearch! = AdmissibleBacktrackingLineSearch(positive),
        callback = (u, res, n) -> push!(iterates, copy(u))
    )
    @test result.solved
    @test u ≈ ones(n) rtol = 1.0e-5
    @test all(x -> all(>(0), x), iterates)

    # A step limiter: at most a relative change of 50% per Newton step
    max_step(u, d, p) = minimum(i -> d[i] < 0 ? 0.5 * u[i] / -d[i] : Inf, eachindex(u, d))
    iterates = Vector{Float64}[]
    u, result = newton_krylov!(
        log_residual!, copy(u₀), p; forcing = nothing,
        linesearch! = AdmissibleBacktrackingLineSearch(positive; max_step, armijo = false),
        callback = (u, res, n) -> push!(iterates, copy(u))
    )
    @test result.solved
    @test u ≈ ones(n) rtol = 1.0e-5
    @test all(i -> all(iterates[i + 1] .>= 0.5 .* iterates[i] .- 1.0e-12), 1:(length(iterates) - 1))

    # No admissible state along the direction: the line search fails with Inf
    u, result = newton_krylov!(
        log_residual!, copy(u₀), p; forcing = nothing,
        linesearch! = AdmissibleBacktrackingLineSearch((u, p) -> false; n_iter_max = 3)
    )
    @test result.status === :nonfinite
    @test u ≈ u₀ # reset to the state before the step
end
