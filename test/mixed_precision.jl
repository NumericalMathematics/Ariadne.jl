using Test
using Ariadne
using Ariadne: MixedPrecisionLU, ConvertedPreconditioner, lu_in_precision!, DirectSolveWorkspace,
    IterativeRefinementWorkspace, GMRESIRWorkspace, LaggedPreconditioner
using LinearAlgebra

# Chandrasekhar H-equation (Kelley 2022, Eq. (3.2)), midpoint rule
function heq!(F, x, c)
    N = length(x)
    for i in 1:N
        s = zero(eltype(F))
        for j in 1:N
            s += x[j] / (2N * (i + j - 1))
        end
        F[i] = x[i] - inv(1 - c * (i - 1 / 2) * s)
    end
    return nothing
end

function heq_jacobian(x, c)
    N = length(x)
    G = [inv(1 - c * (i - 1 / 2) * sum(x[j] / (2N * (i + j - 1)) for j in 1:N)) for i in 1:N]
    return [(i == j) - G[i]^2 * c * (i - 1 / 2) / (2N * (i + j - 1)) for i in 1:N, j in 1:N]
end

@testset "lu_in_precision!" begin
    A = rand(64, 64) + 8I
    # Non-BLAS types: same operations as LinearAlgebra.generic_lufact!
    for T in (Float16, BigFloat)
        F = lu_in_precision!(Matrix{T}(A))
        G = LinearAlgebra.generic_lufact!(Matrix{T}(A))
        @test F.factors == G.factors
        @test F.ipiv == G.ipiv
        @test lu_in_precision!(Matrix{T}(A); threaded = false).factors == F.factors
    end
    # BLAS types use LAPACK
    @test lu_in_precision!(Matrix{Float32}(A)).factors == lu(Matrix{Float32}(A)).factors
end

@testset "MixedPrecisionLU" begin
    A = rand(32, 32) + 8I
    b = rand(32)
    x = A \ b
    P = MixedPrecisionLU(Matrix{Float32}(A); factor_precision = Float16)
    @test eltype(P) == Float32
    @test Ariadne.factor_precision(P) == Float16
    @test Ariadne.solve_precision(P) == Float16
    @test norm(P \ b - x) / norm(x) < 1.0e-2
    # Scaling keeps tiny right-hand sides from underflowing in half precision
    @test norm(P \ (1.0e-12 * b) - 1.0e-12 * x) / norm(1.0e-12 * x) < 1.0e-2
    # Factors promoted to single precision for the solves
    Q = MixedPrecisionLU(Matrix{Float32}(A); factor_precision = Float16, solve_precision = Float32)
    @test Ariadne.solve_precision(Q) == Float32
    @test Q.solver.factors == Float32.(P.factors.factors)
    @test norm(Q \ Float32.(b) - x) / norm(x) < 1.0e-2
end

@testset "ConvertedPreconditioner" begin
    A = rand(32, 32) + 8I
    b = rand(32)
    x = A \ b
    C = ConvertedPreconditioner{Float32}(lu(Matrix{Float32}(A)))
    y = C \ b
    @test eltype(y) == Float64
    @test norm(y - x) / norm(x) < 1.0e-5
    # Scaling keeps tiny right-hand sides from underflowing in half precision
    H = ConvertedPreconditioner{Float16}(Diagonal(fill(Float16(2), 32)))
    @test H \ fill(1.0e-12, 32) ≈ fill(0.5e-12, 32)
    @test ConvertedPreconditioner{Float16}(Diagonal(fill(Float16(2), 32)); scale = false) \
        fill(1.0e-12, 32) == zeros(32)
    @test C \ zeros(32) == zeros(32)
    # mul! is applied in the low precision as well
    D = ConvertedPreconditioner{Float32}(Diagonal(fill(2.0f0, 32)))
    @test mul!(similar(b), D, b) ≈ 2b rtol = 1.0e-6
end

@testset "Newton in mixed precision (H-equation)" begin
    N = 128
    c = 0.99
    kw = (; tol_rel = 1.0e-8, tol_abs = 1.0e-8, max_niter = 10, forcing = nothing)
    function solve(TA, TF, TS, step)
        u = ones(N)
        res = zeros(N)
        P = LaggedPreconditioner(
            J -> MixedPrecisionLU(
                Matrix{TA}(heq_jacobian(J.u, c));
                factor_precision = TF, solve_precision = TS
            )
        )
        hist = Float64[]
        ws = NewtonKrylovWorkspace(heq!, u, c, res, step isa Symbol ? Val(step) : step(res))
        _, result = newton_krylov!(
            ws; kw..., N = P,
            iteration_callback = (ws, info) -> push!(hist, info.norm_res)
        )
        return result, hist
    end
    r64, h64 = solve(Float64, Float64, Float64, :direct)
    @test r64.solved
    @test r64.stats.outer_iterations == 5
    r32, h32 = solve(Float32, Float32, Float32, DirectSolveWorkspace)
    @test r32.stats.outer_iterations == 5
    @test h32[1:4] ≈ h64[1:4] rtol = 1.0e-3
    # Half precision: q-linear convergence (Kelley 2022)
    r16, h16 = solve(Float16, Float16, Float16, :direct)
    @test r16.stats.outer_iterations > 5
    # Three precisions: IR and GMRES-IR recover the Newton iteration (Kelley 2024)
    rir, hir = solve(Float32, Float16, Float32, :ir)
    @test rir.stats.outer_iterations == 5
    @test hir[1:4] ≈ h64[1:4] rtol = 1.0e-3
    @test rir.stats.inner_iterations > 5
    rgm, hgm = solve(Float32, Float16, Float32, :gmresir)
    @test rgm.stats.outer_iterations == 5
    @test hgm[1:4] ≈ h64[1:4] rtol = 1.0e-3
    # Five precisions: GMRES-IR with the inner GMRES (basis and arithmetic) in Float32 for a
    # Float64 working precision, Float16 factors applied in Float32
    r5, h5 = solve(Float64, Float16, Float32, res -> GMRESIRWorkspace(res; gmres_precision = Float32))
    @test r5.stats.outer_iterations == 5
    @test h5[1:4] ≈ h64[1:4] rtol = 1.0e-3
    ws5 = GMRESIRWorkspace(zeros(N); gmres_precision = Float32)
    _, _ = solve(Float64, Float16, Float32, res -> ws5)
    @test eltype(ws5.gmres.x) == Float32
    # The same with the matrix-free Enzyme JVP as operator
    rj5, hj5 = solve(Float64, Float16, Float32, res -> GMRESIRWorkspace(res; operator = :jacobian, gmres_precision = Float32))
    @test rj5.solved
    @test hj5[1:4] ≈ h64[1:4] rtol = 1.0e-3
    # IR with the matrix-free Enzyme JVP as operator
    rj, hj = solve(Float64, Float16, Float32, res -> IterativeRefinementWorkspace(res; operator = :jacobian))
    @test rj.solved
    @test hj[1:4] ≈ h64[1:4] rtol = 1.0e-3
    # The factorization as preconditioner of the Krylov solver
    u = ones(N)
    P = LaggedPreconditioner(
        J -> MixedPrecisionLU(heq_jacobian(J.u, c); factor_precision = Float16, solve_precision = Float32)
    )
    _, result = newton_krylov!(heq!, u, c; tol_rel = 1.0e-8, tol_abs = 1.0e-8, N = P, krylov_kwargs = (; ldiv = true))
    @test result.solved
    # A Float32 preconditioner inside FGMRES, built by a LaggedPreconditioner
    P32 = ConvertedPreconditioner{Float32}(
        LaggedPreconditioner(J -> lu(Matrix{Float32}(heq_jacobian(J.u, c))))
    )
    _, result = newton_krylov!(
        heq!, ones(N), c; algo = :fgmres, tol_rel = 1.0e-8, tol_abs = 1.0e-8, N = P32,
        krylov_kwargs = (; ldiv = true)
    )
    @test result.solved
    @test P32.P.n_builds == result.stats.outer_iterations
    # Step workspaces through the `algo` keyword
    _, result = newton_krylov!(heq!, ones(N), c; algo = :ir, kw..., N = P)
    @test result.solved
    # A factorization is required
    @test_throws ArgumentError newton_krylov!(heq!, ones(N), c; algo = :direct, kw...)
end
