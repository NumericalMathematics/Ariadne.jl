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
    SparseJacobian(pattern; colors = nothing, batchsize = 8, check_pattern = true)

Workspace for the assembly of the sparse Jacobian `∂f/∂u` of an in-place function
`f!(res, u, p)` with the sparsity `pattern` (any matrix whose structural nonzeros are the
possible nonzeros of the Jacobian). The columns are grouped by the column `colors` and
`batchsize` colors are computed together by one batched forward-mode Enzyme.jl pass
([`BatchedJacobianOperator`](@ref), Julia ≥ 1.11; on older versions or for
`batchsize = 1` one [`JacobianOperator`](@ref) product per color is used).

`colors` is a vector with the color of each column, a function `pattern -> colors`, or
`nothing` for [`greedy_column_coloring`](@ref). See SparseMatrixColorings.jl for
colorings with fewer colors. The coloring is computed once and reused for all assemblies.

The diagonal is always included in the pattern of square Jacobians, so that
`Diagonal(d) - J` has the same pattern.

With `check_pattern = true`, each assembly also checks whether the products have nonzeros in
rows that no column of the color covers, i.e., whether `pattern` misses nonzeros of the
Jacobian (e.g., a pattern detected at a different state for a function with branches).
These entries are dropped from `A.J`; their number and largest magnitude in the last
assembly are `A.missed_entries` and `A.missed_max`.

Use [`assemble!`](@ref) to compute the Jacobian, which is stored in `A.J`.
"""
mutable struct SparseJacobian{T, BS}
    const J::SparseMatrixCSC{T, Int}
    const colors::Vector{Int}
    const ncolors::Int
    # columns of color `c`: `color_cols[color_ptr[c]:(color_ptr[c + 1] - 1)]`
    const color_ptr::Vector{Int}
    const color_cols::Vector{Int}
    const seeds::Matrix{T}
    const compressed::Matrix{T}
    const covered::Vector{Int} # covered[i] == c: row `i` is covered by a column of color `c`
    const check_pattern::Bool
    operator::Any # cached (Batched)JacobianOperator for the last (f!, u, p)
    parallel::Any # cached operators and buffers for the last (f!, u, ps), see `assemble!`
    n_assemblies::Int
    time::Float64
    missed_entries::Int
    missed_max::T
end

function SparseJacobian(
        pattern::AbstractMatrix; colors = nothing, batchsize::Integer = 8,
        eltype::Type{T} = Float64, check_pattern::Bool = true
    ) where {T}
    pattern = bool_pattern(pattern)
    m, n = size(pattern)
    if m == n
        pattern = pattern .| sparse(I, n, n)
    end
    if colors === nothing
        colors = greedy_column_coloring(pattern)
    elseif !(colors isa AbstractVector)
        colors = colors(pattern)
    end
    @assert length(colors) == n "colors must have one entry per column"
    J = SparseMatrixCSC{T, Int}(pattern)
    fill!(nonzeros(J), zero(T))
    ncolors = maximum(colors; init = 0)
    color_ptr, color_cols = color_groups(colors, ncolors)
    N = Int(batchsize)
    if VERSION < v"1.11.0"
        N = 1
    end
    return SparseJacobian{T, N}(
        J, Vector{Int}(colors), ncolors, color_ptr, color_cols, zeros(T, n, N), zeros(T, m, N),
        zeros(Int, m), check_pattern, nothing, nothing, 0, 0.0, 0, zero(T)
    )
end

Base.size(A::SparseJacobian) = size(A.J)
batch_size(::SparseJacobian{T, BS}) where {T, BS} = BS
Base.show(io::IO, A::SparseJacobian{T, BS}) where {T, BS} =
    print(io, "SparseJacobian{$T, $BS}(", size(A, 1), "×", size(A, 2), ", nnz = ", nnz(A.J), ", ", A.ncolors, " colors)")

function jacobian_operator(A::SparseJacobian{T, BS}, f!, u, p) where {T, BS}
    op = A.operator
    if op === nothing || op.f !== f! || op.u !== u || op.p !== p
        res = similar(u)
        Enzyme.make_zero!(res)
        if BS == 1
            op = JacobianOperator(f!, res, u, p)
        else
            op = BatchedJacobianOperator{BS}(f!, res, u, p)
        end
        A.operator = op
    end
    return op
end

"""
    assemble!(A::SparseJacobian, f!, u, p) -> A.J
    assemble!(A::SparseJacobian, J::JacobianOperator) -> A.J

Assemble the sparse Jacobian `∂f/∂u` of `f!(res, u, p)` at `u` into `A.J` by colored
forward-mode AD. The second form uses the function, state and parameters of the
Jacobian operator `J` of [`newton_krylov!`](@ref).
"""
function assemble!(A::SparseJacobian{T, BS}, f!::F, u, p) where {T, BS, F}
    t₀ = time_ns()
    op = jacobian_operator(A, f!, u, p)
    (; seeds, compressed, covered) = A
    missed = 0
    missed_max = zero(T)
    for offset in 0:BS:(A.ncolors - 1)
        n, mx = assemble_batch!(A, op, seeds, compressed, covered, offset)
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
function assemble_batch!(A::SparseJacobian, op, seeds::AbstractMatrix{T}, compressed, covered, offset) where {T}
    (; J, color_ptr, color_cols, ncolors) = A
    BS = size(seeds, 2)
    nb = min(BS, ncolors - offset)
    rows = rowvals(J)
    vals = nonzeros(J)
    fill!(seeds, zero(T))
    for k in 1:nb, idx in color_ptr[offset + k]:(color_ptr[offset + k + 1] - 1)
        seeds[color_cols[idx], k] = one(T)
    end
    if BS == 1
        mul!(vec(compressed), op, vec(seeds))
    else
        mul!(compressed, op, seeds)
    end
    missed = 0
    missed_max = zero(T)
    for k in 1:nb
        c = offset + k
        for idx in color_ptr[c]:(color_ptr[c + 1] - 1)
            j = color_cols[idx]
            for idx2 in nzrange(J, j)
                vals[idx2] = compressed[rows[idx2], k]
                covered[rows[idx2]] = c
            end
        end
        if A.check_pattern
            for i in axes(compressed, 1)
                x = compressed[i, k]
                if covered[i] != c && !iszero(x)
                    missed += 1
                    missed_max = max(missed_max, abs(x))
                end
            end
        end
    end
    return missed, missed_max
end

"""
    PerTaskParameters(ps::AbstractVector)

Independent copies `ps` of the parameters `p` of `f!(res, u, p)`, one per task, for the
parallel [`assemble!`](@ref) over batches of colors.
"""
struct PerTaskParameters{V <: AbstractVector}
    ps::V
end
Base.length(p::PerTaskParameters) = length(p.ps)

function parallel_operators(A::SparseJacobian{T, BS}, f!, u, ps) where {T, BS}
    cache = A.parallel
    if cache === nothing || cache.f !== f! || cache.u !== u || cache.ps !== ps
        ops = map(ps) do p
            res = similar(u)
            Enzyme.make_zero!(res)
            BS == 1 ? JacobianOperator(f!, res, u, p) : BatchedJacobianOperator{BS}(f!, res, u, p)
        end
        m, n = size(A.J)
        cache = (;
            f = f!, u, ps, ops,
            seeds = [zeros(T, n, BS) for _ in ps], compressed = [zeros(T, m, BS) for _ in ps],
            covered = [zeros(Int, m) for _ in ps],
        )
        A.parallel = cache
    end
    return cache
end

"""
    assemble!(A::SparseJacobian, f!, u, ps::PerTaskParameters)

Parallel assembly: the batches of colors are distributed over tasks (`Threads.@threads`),
and each task uses its own parameters `ps.ps[k]` (and its own Enzyme shadows). The elements
of `ps` must not share mutable state that `f!` writes, e.g., independent copies of the caches
of a discretization, and `f!` must be safe to call concurrently (for example, not itself use
`Threads.@threads :static`). Use `length(ps) == Threads.nthreads()`.

Compared with threading inside `f!`, this avoids the synchronization of every threaded loop
of `f!` in every product, and Enzyme.jl does not need to differentiate threaded loops.
"""
function assemble!(A::SparseJacobian{T, BS}, f!::F, u, ps::PerTaskParameters) where {T, BS, F}
    t₀ = time_ns()
    (; ops, seeds, compressed, covered) = parallel_operators(A, f!, u, ps.ps)
    pool = Channel{Int}(length(ops))
    foreach(k -> put!(pool, k), eachindex(ops))
    missed = Threads.Atomic{Int}(0)
    missed_max = Ref(zero(T))
    lk = ReentrantLock()
    # Batches write to disjoint columns of `J`
    Threads.@threads :dynamic for offset in 0:BS:(A.ncolors - 1)
        k = take!(pool)
        try
            n, mx = assemble_batch!(A, ops[k], seeds[k], compressed[k], covered[k], offset)
            if n > 0
                Threads.atomic_add!(missed, n)
                @lock lk missed_max[] = max(missed_max[], mx)
            end
        finally
            put!(pool, k)
        end
    end
    finish_assembly!(A, missed[], missed_max[], t₀)
    return A.J
end

assemble!(A::SparseJacobian, J::JacobianOperator) = assemble!(A, J.f, J.u, J.p)
