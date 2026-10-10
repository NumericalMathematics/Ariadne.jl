##
# Assembled sparse Jacobians by colored (batched) AD with Enzyme
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
                   structure = :nonsymmetric, partition = :column, batchsize = 8,
                   check_pattern = true)
    SparseJacobian(f!, res, u, ps::PerTaskParameters, pattern; kwargs...)

Workspace for the assembly of the sparse Jacobian `∂f/∂u` of the in-place function
`f!(res, u, p)` at the state `u` with the sparsity `pattern` (any matrix whose structural
nonzeros are the possible nonzeros of the Jacobian), by colored AD. `res` is a
residual buffer. The Jacobian operator aliases `u` and `p`: [`assemble!`](@ref) assembles
the Jacobian at their current values, so mutate them in place between assemblies.

The columns are grouped by a coloring of SparseMatrixColorings.jl for the
`ColoringProblem(; structure, partition)`, and `batchsize` colors are computed together by
one batched Enzyme.jl pass ([`BatchedJacobianOperator`](@ref), Julia ≥ 1.11; on older
versions or for `batchsize = 1` one [`JacobianOperator`](@ref) product per color is used):
- `partition = :column`: a distance-2 column coloring, one forward-mode Jacobian-vector
  product per color;
- `partition = :row`: a distance-2 row coloring, one reverse-mode vector-Jacobian product
  per color (fewer colors for Jacobians with dense rows, e.g., `length(res) < length(u)`);
- `structure = :symmetric` (with `partition = :column`): a star coloring for Jacobians that
  are symmetric (the `pattern` must be symmetric), which needs fewer colors.
`coloring` is a coloring algorithm of SparseMatrixColorings.jl (e.g.,
`GreedyColoringAlgorithm(LargestFirst())`) or a vector with the color of each column (row).
With a tuple of orders, e.g., `GreedyColoringAlgorithm((NaturalOrder(), LargestFirst(),
SmallestLast(), IncidenceDegree(), DynamicLargestFirst()))`, the coloring with the fewest
colors is kept (for the 5-point Laplacian on a 60 × 60 grid, 5 instead of the 7 colors of
the default natural order). The coloring is computed once and reused for all assemblies.

Given colors fit discretizations with a known block structure. For example, for a DG-like
discretization with `N` nodes per cell and `K` cells in a periodic 1D mesh, where the
residual of a cell depends on the cell and its two neighbors (`K` divisible by 3), the
columns with the same element-local index `i` in every third cell can share a color:
```julia
pattern = kron(spdiagm(-1 => ones(K - 1), 0 => ones(K), 1 => ones(K - 1),
                       K - 1 => ones(1), 1 - K => ones(1)), ones(N, N))
colors = [i + N * mod(k - 1, 3) for i in 1:N, k in 1:K]  # 3N colors
A = SparseJacobian(f!, res, u, p, pattern; coloring = vec(colors))
```

The diagonal is always included in the pattern of square Jacobians, so that
`Diagonal(d) - J` has the same pattern.

With `check_pattern = true`, each assembly also checks whether the products have nonzeros in
rows (columns for `partition = :row`) that no column (row) of the color covers, i.e., whether `pattern` misses nonzeros of the
Jacobian (e.g., a pattern detected at a different state for a function with branches).
These entries are dropped from `A.J`; their number and largest magnitude in the last
assembly are `A.missed_entries` and `A.missed_max`.

With [`PerTaskParameters`](@ref) `ps` instead of `p`, [`assemble!`](@ref) distributes the
batches of colors over tasks, each with its own copy of the parameters.

The Jacobian is stored in `A.J`.

!!! note
    `SparseJacobian` requires SparseMatrixColorings.jl to be loaded (package extension).
"""
mutable struct SparseJacobian{T, BS, Op <: AbstractJacobianOperator, R}
    const J::SparseMatrixCSC{T, Int}
    # (Batched)JacobianOperators of `f!` at `u`, one per task (for the parameters of each
    # task, see `PerTaskParameters`), with their buffers
    const operators::Vector{Op}
    const coloring::R # column or row coloring of SparseMatrixColorings.jl
    const pattern::SparseMatrixCSC{Bool, Int} # pattern with the seeded dimension as columns
    const seeds::Vector{Matrix{T}}
    const compressed::Vector{Matrix{T}}
    const covered::Vector{Vector{Int}} # covered[k][i] == c: entry `i` of a product of task `k` is covered by color `c`
    const check_pattern::Bool
    n_assemblies::Int
    time::Float64
    missed_entries::Int
    missed_max::T
end

"""
    PerTaskParameters(ps::AbstractVector)

Independent copies `ps` of the parameters `p` of `f!(res, u, p)`, one per task, for a
[`SparseJacobian`](@ref) whose [`assemble!`](@ref) distributes the batches of colors over
tasks (`Threads.@threads`). Each task uses its own parameters (and its own Enzyme shadows).
The elements of `ps` must not share mutable state that `f!` writes, e.g., independent copies
of the caches of a discretization, and `f!` must be safe to call concurrently (for example,
not itself use `Threads.@threads :static`). Use `length(ps) == Threads.nthreads()`.

Compared with threading inside `f!`, this avoids the synchronization of every threaded loop
of `f!` in every product, and Enzyme.jl does not need to differentiate threaded loops.
"""
struct PerTaskParameters{V <: AbstractVector}
    ps::V
end
Base.length(p::PerTaskParameters) = length(p.ps)

Base.size(A::SparseJacobian) = size(A.J)
batch_size(::SparseJacobian{T, BS}) where {T, BS} = BS
Base.show(io::IO, A::SparseJacobian{T, BS}) where {T, BS} =
    print(io, "SparseJacobian{$T, $BS}(", size(A, 1), "×", size(A, 2), ", nnz = ", nnz(A.J), ", ", num_colors(A.coloring), " colors)")

# Number of colors of a coloring result, defined by the SparseMatrixColorings.jl extension
function num_colors end

"""
    assemble!(A::SparseJacobian) -> A.J

Assemble the sparse Jacobian `∂f/∂u` of `f!(res, u, p)` at the current state `u` (and
parameters `p`) of `A` into `A.J` by colored AD. With [`PerTaskParameters`](@ref), the
batches of colors are distributed over tasks.
"""
function assemble! end

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

# Point to the package that must be loaded for the methods defined in package extensions
function sparse_jacobian_error_hint(io, exc, argtypes, kwargs)
    f = exc.f
    # Calls with keyword arguments throw a MethodError of `Core.kwcall(kwargs, f, args...)`
    if f === Core.kwcall && length(exc.args) >= 2
        f = exc.args[2]
    end
    if f === SparseJacobian || f === assemble!
        pkg, ext = "SparseMatrixColorings", :AriadneSparseMatrixColoringsExt
    elseif f === jacobian_sparsity
        pkg, ext = "SparseConnectivityTracer", :AriadneSparseConnectivityTracerExt
    else
        return
    end
    if Base.get_extension(@__MODULE__, ext) === nothing
        print(io, "\n`$f` requires $pkg.jl to be loaded: run `using $pkg`.")
    end
    return
end

function __init__()
    Base.Experimental.register_error_hint(sparse_jacobian_error_hint, MethodError)
    return
end
