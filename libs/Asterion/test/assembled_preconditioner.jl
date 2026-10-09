using Test
using Ariadne
using Asterion
using LinearAlgebra
using SparseArrays
using SparseConnectivityTracer

# Steady state of the 2D Bratu problem du/dt = Δu + λ exp(u) on an m × m grid
function bratu2d!(du, u, p)
    (; λ, m) = p
    h = 1 / (m + 1)
    U = reshape(u, m, m)
    dU = reshape(du, m, m)
    z = zero(eltype(u))
    for j in 1:m, i in 1:m
        ul = i > 1 ? U[i - 1, j] : z
        ur = i < m ? U[i + 1, j] : z
        ud = j > 1 ? U[i, j - 1] : z
        uu = j < m ? U[i, j + 1] : z
        dU[i, j] = (ul + ur + ud + uu - 4 * U[i, j]) / h^2 + λ * exp(U[i, j])
    end
    return nothing
end

function laplace_pattern(m)
    T = spdiagm(-1 => ones(m - 1), 0 => ones(m), 1 => ones(m - 1))
    return kron(sparse(I, m, m), T) + kron(T, sparse(I, m, m)) .!= 0
end

# Bratu with a branch: the coupling to the right neighbor only exists for u[i] > 0
function bratu_branch!(du, u, p)
    bratu2d!(du, u, p)
    m = p.m
    for i in 1:(m^2 - 1)
        if u[i] > 0
            du[i] += u[i + 1]^2
        end
    end
    return nothing
end

@testset "Pattern updates" begin
    m = 8
    n = m^2
    p = (; λ = 2.0, m)
    u_neg = fill(-0.1, n)
    u_pos = fill(0.1, n)
    local_pattern(f!, u, p) = Ariadne.jacobian_sparsity(f!, u, p; detector = TracerLocalSparsityDetector())
    # The assembled preconditioner detects the pattern again and recolors
    P = assembled_preconditioner(bratu_branch!, u_neg, p, AssembledJacobianPreconditioner(; sparsity = local_pattern, batchsize = 4))
    b = P.build
    J = Ariadne.JacobianOperator(bratu_branch!, zeros(n), u_neg, p)
    b(J)
    @test b.n_pattern_updates == 0
    J = Ariadne.JacobianOperator(bratu_branch!, zeros(n), u_pos, p)
    @test_logs (:warn,) match_mode = :any b(J)
    @test b.n_pattern_updates == 1
    @test b.jacobian.missed_entries == 0
    @test b.jacobian.J ≈ collect(J)
    @test b.jacobian.n_assemblies == 3 # including the one with the old pattern
    @test b.assembly_time > 0
    @test b.matrix ≈ collect(J)
    # no update for missed entries below the relative tolerance
    P = assembled_preconditioner(bratu_branch!, u_neg, p, AssembledJacobianPreconditioner(; sparsity = local_pattern, pattern_update_rtol = 1.0))
    @test_logs (:warn,) match_mode = :any P.build(J)
    @test P.build.n_pattern_updates == 0
    @test P.build.jacobian.missed_entries > 0
end

@testset "Assembled preconditioner" begin
    m = 16
    n = m^2
    p = (; λ = 2.0, m)
    pattern = laplace_pattern(m)

    # Newton-Krylov with the assembled preconditioner
    _, plain = newton_krylov!(bratu2d!, zeros(n), p; forcing = Ariadne.Fixed(1.0e-6))
    P = assembled_preconditioner(pattern; refresh_interval = 100)
    _, prec = newton_krylov!(
        bratu2d!, zeros(n), p; forcing = Ariadne.Fixed(1.0e-6), N = P,
        krylov_kwargs = (; ldiv = true)
    )
    @test plain.solved && prec.solved
    @test prec.stats.inner_iterations < plain.stats.inner_iterations / 5
    @test P.n_builds == 1

    # PTC with the assembled preconditioner of D⁻¹ - ∂f/∂u
    u₀ = zeros(n)
    alg = PseudoTransientNewtonKrylov(; cfl = SER(; initial = 1.0e-3, growth_max = 4.0))
    u_ref, ws_ref = pseudo_transient!(bratu2d!, copy(u₀), p, alg; reltol = 1.0e-10)
    @test ws_ref.status === :converged
    for factorize in (lu, RowScaled(lu))
        spec = AssembledJacobianPreconditioner(;
            sparsity = pattern, factorize, refresh_interval = 5, batchsize = 4
        )
        alg = PseudoTransientNewtonKrylov(;
            cfl = SER(; initial = 1.0e-3, growth_max = 4.0), preconditioner = spec
        )
        u, ws = pseudo_transient!(bratu2d!, copy(u₀), p, alg; reltol = 1.0e-10)
        @test ws.status === :converged
        @test u ≈ u_ref rtol = 1.0e-6
        @test ws.stats.krylov_iterations < ws_ref.stats.krylov_iterations / 2
        @test 1 < ws.stats.preconditioner_builds < ws.stats.steps
        @test ws.stats.timings[:assembly] > 0
        @test ws.stats.timings[:factorization] > 0
    end

    # Parallel assembly over batches of colors with per-task parameter copies
    spec = AssembledJacobianPreconditioner(;
        sparsity = pattern, refresh_interval = 5, batchsize = 4,
        task_parameters = p -> [deepcopy(p) for _ in 1:2]
    )
    alg = PseudoTransientNewtonKrylov(; cfl = SER(; initial = 1.0e-3, growth_max = 4.0), preconditioner = spec)
    u, ws = pseudo_transient!(bratu2d!, copy(u₀), p, alg; reltol = 1.0e-10)
    @test ws.status === :converged
    @test u ≈ u_ref rtol = 1.0e-6
    @test ws.preconditioner.build.task_parameters_cache !== nothing

    # Sparsity detection by a function at initialization
    spec = AssembledJacobianPreconditioner(; sparsity = Ariadne.jacobian_sparsity, refresh_interval = 5)
    alg = PseudoTransientNewtonKrylov(; cfl = SER(; initial = 1.0e-3, growth_max = 4.0), preconditioner = spec)
    u, ws = pseudo_transient!(bratu2d!, copy(u₀), p, alg; reltol = 1.0e-10)
    @test ws.status === :converged

    # Converged Jacobian for an adjoint solve
    A = jacobian_assembler(ws)
    @test A isa SparseJacobian
    J_sparse = copy(assemble!(A, ws.f, ws.u, ws.p))
    Jop = steady_jacobian(ws)
    x = rand(n)
    y = similar(x)
    mul!(y, transpose(Jop), x)
    @test y ≈ transpose(J_sparse) * x
    mul!(y, Jop, x)
    @test y ≈ J_sparse * x

    # A user-provided LaggedPreconditioner sees the PTC parameters
    seen = Ref{Any}(nothing)
    P = LaggedPreconditioner(J -> (seen[] = (J.p.p, J.p.cfl[], copy(J.p.inv_dtau)); I); refresh_interval = 1)
    alg = PseudoTransientNewtonKrylov(; cfl = SER(; initial = 1.0e-3), preconditioner = P)
    u, ws = pseudo_transient!(bratu2d!, copy(u₀), p, alg; maxiters = 2)
    @test seen[][1] === p
    @test seen[][3] ≈ fill(1 / seen[][2], n)
    @test ws.stats.preconditioner_builds == 2
end
