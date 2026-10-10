module AriadneSparseMatrixColoringsExt

using Ariadne
using Ariadne: AbstractJacobianOperator, JacobianOperator, BatchedJacobianOperator, SparseJacobian,
    finish_assembly!
using LinearAlgebra
using SparseArrays
using SparseMatrixColorings: SparseMatrixColorings, ColoringProblem, GreedyColoringAlgorithm,
    ConstantColoringAlgorithm, column_groups, ncolors, decompress_single_color!

Ariadne.num_colors(result::SparseMatrixColorings.AbstractColoringResult) = ncolors(result)

function Ariadne.SparseJacobian(
        f!, res, u, p, pattern::AbstractMatrix; coloring = GreedyColoringAlgorithm(),
        batchsize::Integer = 8, check_pattern::Bool = true
    )
    T = eltype(u)
    P = SparseMatrixCSC{Bool, Int}(sparse(pattern) .!= 0)
    dropzeros!(P)
    m, n = size(P)
    @assert (m, n) == (length(res), length(u)) "pattern must have size (length(res), length(u))"
    if m == n
        P = P .| sparse(I, n, n)
    end
    problem = ColoringProblem(; structure = :nonsymmetric, partition = :column)
    algorithm = coloring isa AbstractVector ?
        ConstantColoringAlgorithm(P, coloring; partition = :column) : coloring
    result = SparseMatrixColorings.coloring(P, problem, algorithm)
    J = SparseMatrixCSC{T, Int}(P)
    fill!(nonzeros(J), zero(T))
    BS = VERSION < v"1.11.0" ? 1 : Int(batchsize)
    operator = BS == 1 ? JacobianOperator(f!, res, u, p) : BatchedJacobianOperator{BS}(f!, res, u, p)
    return SparseJacobian{T, BS, typeof(operator), typeof(result)}(
        J, operator, result, zeros(T, n, BS), zeros(T, m, BS), zeros(Int, m), check_pattern,
        0, 0.0, 0, zero(T)
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

# Columns of the colors `offset + 1:offset + BS` of `A.J` by one (batched) product. Returns
# the number and largest magnitude of the nonzeros of the products in rows that are not
# covered by the pattern of the columns of their color (if `A.check_pattern`).
function assemble_batch!(A::SparseJacobian{T, BS}, offset) where {T, BS}
    (; J, operator, coloring, seeds, compressed, covered) = A
    groups = column_groups(coloring)
    nb = min(BS, ncolors(coloring) - offset)
    rows = rowvals(J)
    fill!(seeds, zero(T))
    for k in 1:nb, j in groups[offset + k]
        seeds[j, k] = one(T)
    end
    if BS == 1
        mul!(vec(compressed), operator, vec(seeds))
    else
        mul!(compressed, operator, seeds)
    end
    missed = 0
    missed_max = zero(T)
    for k in 1:nb
        c = offset + k
        decompress_single_color!(J, view(compressed, :, k), c, coloring)
        A.check_pattern || continue
        for j in groups[c], idx in nzrange(J, j)
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
