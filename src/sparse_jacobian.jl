##
# Assembled sparse Jacobians by colored (batched) forward-mode AD with Enzyme
##

"""
    jacobian_sparsity(f!, u, p; kwargs...)

Detect the sparsity pattern of the Jacobian of `f!(res, u, p)` with respect to `u`.
Requires SparseConnectivityTracer.jl to be loaded and `f!` to be generic in the element
type of `u` and `res` (buffers in `p` must not have a fixed element type).
The keyword argument `detector` defaults to `TracerSparsityDetector()` (global patterns);
use `TracerLocalSparsityDetector()` for local patterns at `u`.

For discretizations with caches of fixed element type, compute the pattern yourself and
pass it to [`SparseJacobian`](@ref).
"""
function jacobian_sparsity end

"""
    SparseJacobian(f!, res, u, p, pattern; coloring = GreedyColoringAlgorithm(),
                   batchsize = 8, check_pattern = true)

Workspace for the assembly of the sparse Jacobian `∂f/∂u` of the in-place function
`f!(res, u, p)` at the state `u` with the sparsity `pattern` (any matrix whose structural
nonzeros are the possible nonzeros of the Jacobian), by colored forward-mode AD. `res` is a
residual buffer. The Jacobian operator aliases `u` and `p`: [`assemble!`](@ref) assembles
the Jacobian at their current values, so mutate them in place between assemblies.

The columns are grouped by a distance-2 column coloring of SparseMatrixColorings.jl, and
`batchsize` colors are computed together by one batched forward-mode Enzyme.jl pass
([`BatchedJacobianOperator`](@ref), Julia ≥ 1.11; on older versions or for `batchsize = 1`
one [`JacobianOperator`](@ref) product per color is used). `coloring` is a coloring
algorithm of SparseMatrixColorings.jl (e.g., `GreedyColoringAlgorithm(LargestFirst())`) or a
vector with the color of each column. The coloring is computed once and reused for all
assemblies.

The diagonal is always included in the pattern of square Jacobians, so that
`Diagonal(d) - J` has the same pattern.

With `check_pattern = true`, each assembly also checks whether the products have nonzeros in
rows that no column of the color covers, i.e., whether `pattern` misses nonzeros of the
Jacobian (e.g., a pattern detected at a different state for a function with branches).
These entries are dropped from `A.J`; their number and largest magnitude in the last
assembly are `A.missed_entries` and `A.missed_max`.

The Jacobian is stored in `A.J`.
"""
mutable struct SparseJacobian{T, BS, Op <: AbstractJacobianOperator, R}
    const J::SparseMatrixCSC{T, Int}
    const operator::Op # (Batched)JacobianOperator of `f!` at `u` and `p`
    const coloring::R # column coloring of SparseMatrixColorings.jl
    const seeds::Matrix{T}
    const compressed::Matrix{T}
    const covered::Vector{Int} # covered[i] == c: row `i` is covered by a column of color `c`
    const check_pattern::Bool
    n_assemblies::Int
    time::Float64
    missed_entries::Int
    missed_max::T
end

function SparseJacobian(
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

Base.size(A::SparseJacobian) = size(A.J)
batch_size(::SparseJacobian{T, BS}) where {T, BS} = BS
Base.show(io::IO, A::SparseJacobian{T, BS}) where {T, BS} =
    print(io, "SparseJacobian{$T, $BS}(", size(A, 1), "×", size(A, 2), ", nnz = ", nnz(A.J), ", ", ncolors(A.coloring), " colors)")

"""
    assemble!(A::SparseJacobian) -> A.J

Assemble the sparse Jacobian `∂f/∂u` of `f!(res, u, p)` at the current state `u` (and
parameters `p`) of `A` into `A.J` by colored forward-mode AD.
"""
function assemble!(A::SparseJacobian{T, BS}) where {T, BS}
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

function finish_assembly!(A::SparseJacobian, missed, missed_max, t₀)
    A.missed_entries = missed
    A.missed_max = missed_max
    # Products can have roundoff-level nonzeros outside the pattern, so only warn about
    # entries that are not small compared with the Jacobian
    if missed > 0 && (relative = missed_max / maximum(abs, nonzeros(A.J); init = zero(missed_max))) > sqrt(eps(one(relative)))
        @warn "SparseJacobian: the sparsity pattern misses $missed nonzeros of the Jacobian (largest magnitude $missed_max, $relative relative to the largest entry); they are dropped. Detect the pattern at this state or use a conservative pattern." maxlog = 1
    end
    A.n_assemblies += 1
    A.time += (time_ns() - t₀) / 1.0e9
    return nothing
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
