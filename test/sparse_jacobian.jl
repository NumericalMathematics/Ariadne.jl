using Test
using Ariadne
using LinearAlgebra
using SparseArrays
using SparseConnectivityTracer
using SparseMatrixColorings

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

dense_jacobian(f!, u, p) = collect(Ariadne.JacobianOperator(f!, zeros(length(u)), copy(u), p))

@testset "Coloring and assembly" begin
    m = 8
    n = m^2
    p = (; λ = 2.0, m)
    pattern = laplace_pattern(m)

    # Sparsity detection with SparseConnectivityTracer.jl
    detected = Ariadne.jacobian_sparsity(bratu2d!, rand(n), p)
    @test (detected .!= 0) == pattern

    u = 0.1 * rand(n)
    J_dense = dense_jacobian(bratu2d!, u, p)
    for batchsize in (1, 4, 8)
        A = SparseJacobian(bratu2d!, zeros(n), u, p, pattern; batchsize)
        @test A.operator isa Ariadne.AbstractJacobianOperator
        @test 5 <= ncolors(A.coloring) <= 8 # optimal: 5
        J = assemble!(A)
        @test J ≈ J_dense
        @test J isa SparseMatrixCSC{Float64, Int}
        @test nnz(J) == nnz(pattern)
        @test A.n_assemblies == 1
        @test A.missed_entries == 0
        # the Jacobian operator aliases `u`: assemble at another state
        u .= 0.2 .* rand.()
        @test assemble!(A) ≈ dense_jacobian(bratu2d!, u, p)
        u .= 0.1 .* rand.()
        J_dense = dense_jacobian(bratu2d!, u, p)
    end
    # Coloring algorithms of SparseMatrixColorings.jl and given colors
    A = SparseJacobian(bratu2d!, zeros(n), u, p, pattern; coloring = GreedyColoringAlgorithm(LargestFirst()))
    @test assemble!(A) ≈ J_dense
    colors = column_colors(A.coloring)
    A = SparseJacobian(bratu2d!, zeros(n), u, p, pattern; coloring = colors)
    @test column_colors(A.coloring) == colors
    @test assemble!(A) ≈ J_dense
    # Invalid colors: two neighbors with the same color
    @test_throws Exception SparseJacobian(bratu2d!, zeros(n), u, p, pattern; coloring = ones(Int, n))
    # The diagonal is always part of the pattern
    A = SparseJacobian(bratu2d!, zeros(n), u, p, spzeros(Bool, n, n))
    @test nnz(A.J) == n
    # Row coloring (vector-Jacobian products) and symmetric (star) coloring
    for batchsize in (1, 4)
        A = SparseJacobian(bratu2d!, zeros(n), u, p, pattern; partition = :row, batchsize)
        @test A.coloring isa SparseMatrixColorings.AbstractColoringResult{:nonsymmetric, :row}
        @test assemble!(A) ≈ J_dense
        @test A.missed_entries == 0
        A = SparseJacobian(bratu2d!, zeros(n), u, p, pattern; structure = :symmetric, batchsize)
        @test A.coloring isa SparseMatrixColorings.AbstractColoringResult{:symmetric, :column}
        @test ncolors(A.coloring) <= 5
        @test assemble!(A) ≈ J_dense
        @test A.missed_entries == 0
    end
    A = SparseJacobian(bratu2d!, zeros(n), u, p, pattern; partition = :row, coloring = colors)
    @test row_colors(A.coloring) == colors
    @test assemble!(A) ≈ J_dense
    @test_throws ArgumentError SparseJacobian(bratu2d!, zeros(n), u, p, pattern; partition = :bidirectional)
    @test_throws ArgumentError SparseJacobian(
        bratu2d!, zeros(n), u, p, pattern; structure = :symmetric,
        coloring = GreedyColoringAlgorithm(; decompression = :substitution)
    )

    # The assembled Jacobian as a preconditioner for Newton-Krylov
    u = zeros(n)
    A = SparseJacobian(bratu2d!, zeros(n), u, p, pattern)
    P = lu(assemble!(A))
    _, plain = newton_krylov!(bratu2d!, zeros(n), p; forcing = Ariadne.Fixed(1.0e-6))
    _, prec = newton_krylov!(
        bratu2d!, zeros(n), p; forcing = Ariadne.Fixed(1.0e-6), N = J -> P,
        krylov_kwargs = (; ldiv = true)
    )
    @test plain.solved && prec.solved
    @test prec.stats.inner_iterations < plain.stats.inner_iterations
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

@testset "Missing nonzeros of the pattern" begin
    m = 8
    n = m^2
    p = (; λ = 2.0, m)
    pattern = laplace_pattern(m)
    # The local pattern at u < 0 misses the u[i + 1]^2 terms at u > 0
    # (except where i + 1 is already a neighbor)
    P_neg = Ariadne.jacobian_sparsity(bratu_branch!, fill(-0.1, n), p; detector = TracerLocalSparsityDetector())
    @test (P_neg .!= 0) == pattern
    for batchsize in (1, 4)
        u = fill(-0.1, n)
        A = SparseJacobian(bratu_branch!, zeros(n), u, p, P_neg; batchsize)
        @test assemble!(A) ≈ dense_jacobian(bratu_branch!, u, p)
        @test A.missed_entries == 0
        u .= 0.1
        J = @test_logs (:warn,) match_mode = :any assemble!(A)
        @test A.missed_entries > 0
        @test A.missed_max ≈ 0.2
        @test !(J ≈ dense_jacobian(bratu_branch!, u, p))
        # without the check
        A = SparseJacobian(bratu_branch!, zeros(n), u, p, P_neg; batchsize, check_pattern = false)
        assemble!(A)
        @test A.missed_entries == 0
    end
end

@testset "Error hints without the package extensions" begin
    # A fresh process in which only Ariadne is loaded
    code = """
    using Ariadne
    for f in (() -> SparseJacobian(identity, [0.0], [0.0], nothing, [true;;]; batchsize = 1),
              () -> assemble!(nothing),
              () -> Ariadne.jacobian_sparsity(identity, [0.0], nothing))
        try
            f()
        catch e
            showerror(stdout, e)
            println(stdout)
        end
    end
    """
    out = read(`$(Base.julia_cmd()) --project=$(Base.active_project()) --startup-file=no -e $code`, String)
    @test count("requires SparseMatrixColorings.jl to be loaded", out) == 2
    @test occursin("requires SparseConnectivityTracer.jl to be loaded", out)
end
