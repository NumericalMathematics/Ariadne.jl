module AriadneSparseMatrixColoringsExt

using Ariadne
using Ariadne: AbstractJacobianOperator, JacobianOperator, BatchedJacobianOperator, SparseJacobian,
    finish_assembly!
using LinearAlgebra
using SparseArrays
using SparseMatrixColorings: SparseMatrixColorings, AbstractColoringResult, ColoringProblem,
    GreedyColoringAlgorithm, ConstantColoringAlgorithm, column_groups, row_groups, ncolors,
    decompress_single_color!

Ariadne.num_colors(result::AbstractColoringResult) = ncolors(result)

# Groups of columns (rows) of each color of a column (row) coloring
color_groups(result::AbstractColoringResult{S, :column}) where {S} = column_groups(result)
color_groups(result::AbstractColoringResult{S, :row}) where {S} = row_groups(result)

function Ariadne.SparseJacobian(
        f!, res, u, p, pattern::AbstractMatrix; coloring = GreedyColoringAlgorithm(),
        structure::Symbol = :nonsymmetric, partition::Symbol = :column,
        batchsize::Integer = 8, check_pattern::Bool = true
    )
    partition in (:column, :row) ||
        throw(ArgumentError("SparseJacobian: `partition` must be `:column` or `:row`, got `$(repr(partition))`"))
    T = eltype(u)
    P = SparseMatrixCSC{Bool, Int}(sparse(pattern) .!= 0)
    dropzeros!(P)
    m, n = size(P)
    @assert (m, n) == (length(res), length(u)) "pattern must have size (length(res), length(u))"
    if m == n
        P = P .| sparse(I, n, n)
    end
    problem = ColoringProblem(; structure, partition)
    algorithm = coloring isa AbstractVector ?
        ConstantColoringAlgorithm(P, coloring; structure, partition) : coloring
    result = SparseMatrixColorings.coloring(P, problem, algorithm)
    J = SparseMatrixCSC{T, Int}(P)
    fill!(nonzeros(J), zero(T))
    hasmethod(decompress_single_color!, Tuple{typeof(J), Vector{T}, Int, typeof(result)}) ||
        throw(ArgumentError("SparseJacobian: the coloring $(typeof(result)) does not support decompression by color, e.g., use star coloring (`GreedyColoringAlgorithm(; decompression = :direct)`) for `structure = :symmetric`"))
    BS = VERSION < v"1.11.0" ? 1 : Int(batchsize)
    operator = BS == 1 ? JacobianOperator(f!, res, u, p) : BatchedJacobianOperator{BS}(f!, res, u, p)
    # Seeds are columns (`partition = :column`, Jacobian-vector products) or rows
    # (`partition = :row`, vector-Jacobian products) of the colors; `pattern` has the seeded
    # dimension as columns
    pattern_by_seed = partition === :column ? P : sparse(transpose(P))
    n_seed, n_out = size(pattern_by_seed, 2), size(pattern_by_seed, 1)
    return SparseJacobian{T, BS, typeof(operator), typeof(result)}(
        J, operator, result, pattern_by_seed, zeros(T, n_seed, BS), zeros(T, n_out, BS),
        zeros(Int, n_out), check_pattern, 0, 0.0, 0, zero(T)
    )
end

function Ariadne.assemble!(A::SparseJacobian{T, BS}) where {T, BS}
    t₀ = time_ns()
    missed = 0
    missed_max = zero(T)
    for offset in 0:BS:(ncolors(A.coloring) - 1)
        n, mx = assemble_batch!(A, offset)
        missed += n
        missed_max = max(missed_max, mx)
    end
    finish_assembly!(A, missed, missed_max, t₀)
    return A.J
end

# Jacobian-vector (column coloring) or vector-Jacobian (row coloring) products
product!(out, operator, seeds, ::AbstractColoringResult{S, :column}) where {S} =
    mul!(out, operator, seeds)
product!(out, operator, seeds, ::AbstractColoringResult{S, :row}) where {S} =
    mul!(out, transpose(operator), seeds)

# Columns (rows) of the colors `offset + 1:offset + BS` of `A.J` by one (batched) product.
# Returns the number and largest magnitude of the nonzeros of the products in entries that
# are not covered by the pattern of the columns (rows) of their color (if `A.check_pattern`).
function assemble_batch!(A::SparseJacobian{T, BS}, offset) where {T, BS}
    (; J, operator, coloring, pattern, seeds, compressed, covered) = A
    groups = color_groups(coloring)
    nb = min(BS, ncolors(coloring) - offset)
    rows = rowvals(pattern)
    fill!(seeds, zero(T))
    for k in 1:nb, j in groups[offset + k]
        seeds[j, k] = one(T)
    end
    if BS == 1
        product!(vec(compressed), operator, vec(seeds), coloring)
    else
        product!(compressed, operator, seeds, coloring)
    end
    missed = 0
    missed_max = zero(T)
    for k in 1:nb
        c = offset + k
        decompress_single_color!(J, view(compressed, :, k), c, coloring)
        A.check_pattern || continue
        for j in groups[c], idx in nzrange(pattern, j)
            covered[rows[idx]] = c
        end
        for i in axes(compressed, 1)
            x = compressed[i, k]
            if covered[i] != c && !iszero(x)
                missed += 1
                missed_max = max(missed_max, abs(x))
            end
        end
    end
    return missed, missed_max
end

end # module
