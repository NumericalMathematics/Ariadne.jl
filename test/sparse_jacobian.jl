using Test
using Ariadne
using LinearAlgebra
using SparseArrays
using SparseConnectivityTracer
using Random

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

@testset "Coloring and assembly" begin
    m = 8
    n = m^2
    p = (; λ = 2.0, m)
    pattern = laplace_pattern(m)

    # Sparsity detection with SparseConnectivityTracer.jl
    detected = Ariadne.jacobian_sparsity(bratu2d!, rand(n), p)
    @test (detected .!= 0) == pattern

    colors = greedy_column_coloring(pattern)
    @test maximum(colors) <= 8 # optimal: 5
    # Columns of the same color are structurally orthogonal
    for c in 1:maximum(colors)
        cols = findall(==(c), colors)
        @test all(<=(1), sum(pattern[:, cols]; dims = 2))
    end

    u = 0.1 * rand(n)
    J_dense = collect(Ariadne.JacobianOperator(bratu2d!, zeros(n), u, p))
    for batchsize in (1, 4, 8)
        A = SparseJacobian(pattern; batchsize)
        J = assemble!(A, bratu2d!, u, p)
        @test J ≈ J_dense
        @test J isa SparseMatrixCSC{Float64, Int}
        @test nnz(J) == nnz(pattern)
        @test A.n_assemblies == 1
        @test A.missed_entries == 0
        # with given colors
        A = SparseJacobian(pattern; batchsize, colors)
        @test assemble!(A, bratu2d!, u, p) ≈ J_dense
        # repeated assembly reuses the cached operator
        u2 = 0.2 * rand(n)
        @test assemble!(A, bratu2d!, u2, p) ≈ collect(Ariadne.JacobianOperator(bratu2d!, zeros(n), u2, p))
    end
    # Parallel assembly over batches of colors with one parameter copy per task
    for batchsize in (1, 4)
        A = SparseJacobian(pattern; batchsize)
        ps = PerTaskParameters([deepcopy(p) for _ in 1:3])
        @test assemble!(A, bratu2d!, u, ps) ≈ J_dense
        @test assemble!(A, bratu2d!, u, ps) ≈ J_dense # cached operators
        @test A.n_assemblies == 2
        @test A.missed_entries == 0
    end
    # Colors from a function of the pattern
    A = SparseJacobian(pattern; colors = P -> greedy_column_coloring(P; order = :largest_first))
    @test assemble!(A, bratu2d!, u, p) ≈ J_dense
    A = SparseJacobian(pattern; colors = column_coloring)
    @test assemble!(A, bratu2d!, u, p) ≈ J_dense
    A = SparseJacobian(pattern; colors = quotient_coloring(pattern, [1 + mod(i, 3) + 3 * mod(j, 3) for j in 0:(m - 1) for i in 0:(m - 1)]))
    @test assemble!(A, bratu2d!, u, p) ≈ J_dense
    # The diagonal is always part of the pattern
    A = SparseJacobian(spzeros(Bool, n, n))
    @test nnz(A.J) == n

    # Assembly from a JacobianOperator
    A = SparseJacobian(pattern)
    @test assemble!(A, Ariadne.JacobianOperator(bratu2d!, zeros(n), u, p)) ≈ J_dense

    # The assembled Jacobian as a preconditioner for Newton-Krylov
    P = LinearAlgebra.lu(assemble!(SparseJacobian(pattern), bratu2d!, zeros(n), p))
    _, plain = newton_krylov!(bratu2d!, zeros(n), p; forcing = Ariadne.Fixed(1.0e-6))
    _, prec = newton_krylov!(
        bratu2d!, zeros(n), p; forcing = Ariadne.Fixed(1.0e-6), N = J -> P,
        krylov_kwargs = (; ldiv = true)
    )
    @test plain.solved && prec.solved
    @test prec.stats.inner_iterations < plain.stats.inner_iterations
end

@testset "Column coloring" begin
    for pattern in (laplace_pattern(8), sprand(Random.Xoshiro(1), Bool, 200, 150, 0.03), spzeros(Bool, 5, 5))
        n = size(pattern, 2)
        lb = Ariadne.column_coloring_lower_bound(pattern)
        for order in (:natural, :reverse, :largest_first, randperm(Random.Xoshiro(2), n))
            colors = greedy_column_coloring(pattern; order)
            @test is_column_coloring(pattern, colors)
            @test maximum(colors; init = 0) >= lb
        end
        natural = maximum(greedy_column_coloring(pattern); init = 0)
        colors = column_coloring(pattern; recolor_iterations = 10)
        @test is_column_coloring(pattern, colors)
        @test lb <= maximum(colors; init = 0) <= natural
        ptr, cols = Ariadne.color_groups(colors)
        @test sort(cols) == 1:n
        @test all(colors[cols[ptr[c]:(ptr[c + 1] - 1)]] == fill(c, ptr[c + 1] - ptr[c]) for c in 1:(length(ptr) - 1))
    end
    pattern = laplace_pattern(8)
    @test is_column_coloring(pattern, ones(Int, 64)) == false
    @test is_column_coloring(pattern, collect(1:63)) == false
    @test_throws ArgumentError greedy_column_coloring(pattern; order = :unknown)

    # Quotient coloring with a periodic cell of a × a grid points
    m = 12
    pattern = laplace_pattern(m)
    cell(a) = [1 + mod(i, a) + a * mod(j, a) for j in 0:(m - 1) for i in 0:(m - 1)]
    for a in (3, 4)
        colors = quotient_coloring(pattern, cell(a))
        @test is_column_coloring(pattern, colors)
        @test 5 <= maximum(colors) <= a^2
        @test all(colors[j] == colors[k] for j in 1:(m^2), k in 1:(m^2) if cell(a)[j] == cell(a)[k])
    end
    @test_throws ArgumentError quotient_coloring(pattern, cell(2))
    @test_throws ArgumentError quotient_coloring(pattern, cell(3)[1:10])
    @test maximum(quotient_coloring(pattern, 1:(m^2))) <= maximum(greedy_column_coloring(pattern))
    # Block classes: blocks of 2 columns with labels 7, 3, 7
    @test block_classes(2, [7, 3, 7]) == [1, 2, 3, 4, 1, 2]
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
    u_neg = fill(-0.1, n)
    u_pos = fill(0.1, n)
    # The local pattern at u_neg misses the u[i + 1]^2 terms of u_pos
    # (except where i + 1 is already a neighbor)
    P_neg = Ariadne.jacobian_sparsity(bratu_branch!, u_neg, p; detector = TracerLocalSparsityDetector())
    @test (P_neg .!= 0) == pattern
    for parallel in (false, true), batchsize in (1, 4)
        ps = parallel ? PerTaskParameters([deepcopy(p) for _ in 1:2]) : p
        A = SparseJacobian(P_neg; batchsize)
        J_dense = collect(Ariadne.JacobianOperator(bratu_branch!, zeros(n), u_neg, p))
        @test assemble!(A, bratu_branch!, u_neg, ps) ≈ J_dense
        @test A.missed_entries == 0
        J_dense = collect(Ariadne.JacobianOperator(bratu_branch!, zeros(n), u_pos, p))
        J = @test_logs (:warn,) match_mode = :any assemble!(A, bratu_branch!, u_pos, ps)
        @test A.missed_entries > 0
        @test A.missed_max ≈ 0.2
        @test !(J ≈ J_dense)
        # without the check
        A = SparseJacobian(P_neg; batchsize, check_pattern = false)
        assemble!(A, bratu_branch!, u_pos, ps)
        @test A.missed_entries == 0
    end
end
