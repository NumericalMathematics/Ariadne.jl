##
# Factorizations
##

"""
    RowScaled(factorize)

Factorization `A -> RowScaled(factorize)(A)` of the row-scaled matrix `D A` with
`D = Diagonal(1 ./ abs.(diag(A)))`, applied as `y = (D A)⁻¹ D x`. This makes absolute drop
tolerances of incomplete factorizations relative to the diagonal, e.g.,
`RowScaled(A -> IncompleteLU.ilu(A; τ = 1.0e-3))`.
"""
struct RowScaled{F}
    factorize::F
end

struct RowScaledFactorization{F, V}
    factorization::F
    scaling::V
    tmp::V
end

function (r::RowScaled)(A)
    scaling = inv.(abs.(diag(A)))
    return RowScaledFactorization(r.factorize(Diagonal(scaling) * A), scaling, similar(scaling))
end

function LinearAlgebra.ldiv!(y, P::RowScaledFactorization, x)
    @. P.tmp = P.scaling * x
    return ldiv!(y, P.factorization, P.tmp)
end

##
# Preconditioner from the assembled Jacobian
##

"""
    AssembledJacobianPreconditioner(; sparsity, coloring = GreedyColoringAlgorithm(),
                                      batchsize = 8, factorize = lu, refresh_interval = 1,
                                      refresh_iterations = typemax(Int),
                                      check_pattern = true, pattern_update_rtol = sqrt(eps()),
                                      task_parameters = nothing)

Specification of a preconditioner for [`PseudoTransientNewtonKrylov`](@ref) (and for
[`newton_krylov!`](@ref Ariadne.newton_krylov!) via [`assembled_preconditioner`](@ref)) that factorizes the
sparse Jacobian assembled by colored forward-mode AD with Enzyme.jl
(see [`SparseJacobian`](@ref Ariadne.SparseJacobian)).

- `sparsity`: sparsity pattern of `∂f/∂u` (a matrix), or a function `(f!, u, p) -> pattern`
  called at initialization, e.g., [`Ariadne.jacobian_sparsity`](@ref) (requires
  SparseConnectivityTracer.jl). If it is a function and an assembly finds nonzeros outside
  the pattern (`check_pattern = true`) larger than `pattern_update_rtol` times the largest
  entry of the Jacobian, the pattern is detected again at the current state, merged with
  the previous one, the coloring is recomputed, and the Jacobian is assembled again (see
  `n_pattern_updates` of the builder). This matters for local sparsity patterns of
  functions with branches, which can change along the solution path.
- `coloring`: column coloring algorithm of SparseMatrixColorings.jl, or a vector of column
  colors, see [`SparseJacobian`](@ref Ariadne.SparseJacobian). After a pattern update, the
  merged pattern is colored with the same algorithm, or greedily for a vector of colors or
  if the algorithm gives a coloring that is invalid for the new nonzeros.
- `batchsize`: number of colors computed by one batched forward-mode pass.
- `factorize`: function `A -> F` with `ldiv!(y, F, x)`, e.g., `lu`,
  `A -> IncompleteLU.ilu(A; τ = 1e-3)`, or [`RowScaled`](@ref)`(…)`.
- `refresh_interval`, `refresh_iterations`: refresh policy of the resulting
  [`LaggedPreconditioner`](@ref Ariadne.LaggedPreconditioner).
- `check_pattern`: check each assembly for nonzeros outside the pattern.
- `task_parameters`: `nothing`, or a function `p -> ps` returning independent copies of the
  parameters, one per task, for the parallel assembly over batches of colors
  (see [`PerTaskParameters`](@ref Ariadne.PerTaskParameters)), called once at initialization.

In pseudo-transient continuation, the factorized matrix is `Diagonal(1 ./ Δτ) - σ ∂f/∂u`
with the pseudo-time steps `Δτ` of the PTC step in which the preconditioner is rebuilt
(see [`PseudoTransientNewtonKrylov`](@ref) for the sign `σ`). For a plain Newton-Krylov
solve, it is `∂f/∂u`.
"""
Base.@kwdef struct AssembledJacobianPreconditioner{S, C, F, TP}
    sparsity::S
    coloring::C = GreedyColoringAlgorithm()
    batchsize::Int = 8
    factorize::F = lu
    refresh_interval::Int = 1
    refresh_iterations::Int = typemax(Int)
    check_pattern::Bool = true
    pattern_update_rtol::Float64 = sqrt(eps())
    task_parameters::TP = nothing
end

"""
    AssembledJacobianBuilder(jacobian::SparseJacobian, factorize = lu;
                             sparsity = nothing, coloring = GreedyColoringAlgorithm(),
                             pattern_update_rtol = sqrt(eps()))

Callable `J -> factorization` used as `build` of a [`LaggedPreconditioner`](@ref Ariadne.LaggedPreconditioner): assembles
the Jacobian with `jacobian` (whose operators alias the state `u` and the parameters `p`)
and factorizes it. `J` is the [`JacobianOperator`](@ref Ariadne.JacobianOperator) of the Newton-Krylov solve, at
the same state `u`. For the pseudo-transient residual of [`PseudoTransientNewtonKrylov`](@ref),
it assembles `∂f/∂u` of the steady residual `f!` and factorizes
`Diagonal(1 ./ Δτ) - σ ∂f/∂u`. Timings are accumulated in the fields
`assembly_time` and `factorization_time`. See [`AssembledJacobianPreconditioner`](@ref) for
the keyword arguments.
"""
mutable struct AssembledJacobianBuilder{SJ <: SparseJacobian, F, M, S, C}
    jacobian::SJ
    const factorize::F
    matrix::M # matrix that is factorized
    diagonal_indices::Vector{Int}
    factorization_time::Float64
    const sparsity::S # `nothing` or `(f!, u, p) -> pattern` to update the pattern
    const coloring::C
    const pattern_update_rtol::Float64
    n_pattern_updates::Int
    previous_assembly_time::Float64 # of the replaced `SparseJacobian`s
end

function AssembledJacobianBuilder(
        jacobian::SparseJacobian, factorize = lu; sparsity = nothing,
        coloring = GreedyColoringAlgorithm(), pattern_update_rtol = sqrt(eps())
    )
    matrix, diagonal_indices = factorization_matrix(jacobian)
    return AssembledJacobianBuilder(
        jacobian, factorize, matrix, diagonal_indices, 0.0, sparsity, coloring,
        Float64(pattern_update_rtol), 0, 0.0
    )
end

function factorization_matrix(jacobian::SparseJacobian)
    matrix = copy(jacobian.J)
    m, n = size(matrix)
    diagonal_indices = Int[]
    if m == n
        rows = rowvals(matrix)
        diagonal_indices = map(1:n) do j
            r = nzrange(matrix, j)
            r[searchsortedfirst(view(rows, r), j)]
        end
    end
    return matrix, diagonal_indices
end

function Base.getproperty(b::AssembledJacobianBuilder, s::Symbol)
    if s === :assembly_time
        return getfield(b, :previous_assembly_time) + getfield(b, :jacobian).time
    else
        return getfield(b, s)
    end
end

# The function, state, and parameters (or `PerTaskParameters`) of a `SparseJacobian`
function jacobian_arguments(A::SparseJacobian)
    ops = A.operators
    op = first(ops)
    params = length(ops) == 1 ? op.p : PerTaskParameters(map(o -> o.p, ops))
    return op.f, op.res, op.u, params
end

# Assemble ∂f/∂u at `u` and, if the pattern misses nonzeros larger than
# `pattern_update_rtol` times the largest entry and can be detected again, merge the
# pattern at `u` into the pattern, recolor, and assemble again
function assemble_and_update!(b::AssembledJacobianBuilder, u)
    A = b.jacobian
    f!, res, u_A, params = jacobian_arguments(A)
    u === u_A || throw(ArgumentError("the assembled preconditioner was created for another state array `u`; create it with the state of the solve"))
    Jf = assemble!(A)
    if A.missed_entries > 0 && b.sparsity !== nothing &&
            A.missed_max > b.pattern_update_rtol * maximum(abs, nonzeros(Jf); init = zero(A.missed_max))
        J = A.J
        old = SparseMatrixCSC(size(J)..., copy(SparseArrays.getcolptr(J)), copy(rowvals(J)), fill(true, nnz(J)))
        p = params isa PerTaskParameters ? first(params.ps) : params
        pattern = old .| (sparse(b.sparsity(f!, u, p)) .!= 0)
        b.previous_assembly_time += A.time
        b.jacobian = recolored_jacobian(b, A, f!, res, u, params, pattern)
        b.jacobian.n_assemblies = A.n_assemblies
        b.matrix, b.diagonal_indices = factorization_matrix(b.jacobian)
        b.n_pattern_updates += 1
        Jf = assemble!(b.jacobian)
    end
    return Jf
end

# The `SparseJacobian` for the merged `pattern`, colored with the coloring of the builder.
# Given colors (a vector, or an algorithm such as `ConstantColoringAlgorithm` that returns
# fixed colors) can be invalid for the new nonzeros, so fall back to a greedy coloring.
function recolored_jacobian(b::AssembledJacobianBuilder, A, f!, res, u, params, pattern)
    batchsize = Ariadne.batch_size(A)
    if !(b.coloring isa AbstractVector)
        try
            return SparseJacobian(f!, res, u, params, pattern; b.coloring, batchsize, A.check_pattern)
        catch err
            err isa SparseMatrixColorings.InvalidColoringError || rethrow()
            @warn "Assembled preconditioner: the coloring is invalid for the updated sparsity pattern; using a greedy coloring" maxlog = 1
        end
    end
    return SparseJacobian(f!, res, u, params, pattern; coloring = GreedyColoringAlgorithm(), batchsize, A.check_pattern)
end

function (b::AssembledJacobianBuilder)(J::JacobianOperator)
    Jf = assemble_and_update!(b, J.u)
    A = b.matrix
    if J.f isa PseudoTransientResidual
        σ = J.f.σ
        vals = nonzeros(A)
        @. vals = -σ * $nonzeros(Jf)
        inv_dtau = J.p.inv_dtau
        for (j, idx) in enumerate(b.diagonal_indices)
            vals[idx] += inv_dtau[j]
        end
    else
        copyto!(nonzeros(A), nonzeros(Jf))
    end
    t₀ = time_ns()
    F = b.factorize(A)
    b.factorization_time += (time_ns() - t₀) / 1.0e9
    return F
end

"""
    assembled_preconditioner(f!, u, p, spec::AssembledJacobianPreconditioner)
    assembled_preconditioner(f!, u, p, pattern::AbstractMatrix; kwargs...)

Create the [`LaggedPreconditioner`](@ref Ariadne.LaggedPreconditioner) described by `spec` for the residual
`f!(res, u, p)` at the state `u` (which the Jacobian operators alias, so pass the state of
the solve), for use with [`newton_krylov!`](@ref Ariadne.newton_krylov!) (`N = P`,
`krylov_kwargs = (; ldiv = true)`). The second form takes the sparsity pattern and the
keyword arguments of [`AssembledJacobianPreconditioner`](@ref).
"""
function assembled_preconditioner(f!, u, p, spec::AssembledJacobianPreconditioner)
    if spec.sparsity isa AbstractMatrix
        pattern = spec.sparsity
        sparsity = nothing
    else
        pattern = spec.sparsity(f!, u, p)
        sparsity = spec.sparsity
    end
    return assembled_preconditioner(
        f!, u, p, pattern; spec.coloring, spec.batchsize, spec.factorize,
        spec.refresh_interval, spec.refresh_iterations, spec.check_pattern,
        spec.task_parameters, sparsity, spec.pattern_update_rtol
    )
end

function assembled_preconditioner(
        f!, u, p, pattern::AbstractMatrix; coloring = GreedyColoringAlgorithm(), batchsize = 8,
        factorize = lu, refresh_interval = 1, refresh_iterations = typemax(Int),
        check_pattern = true, task_parameters = nothing, sparsity = nothing,
        pattern_update_rtol = sqrt(eps())
    )
    params = task_parameters === nothing ? p : PerTaskParameters(task_parameters(p))
    res = similar(u)
    Enzyme.make_zero!(res)
    jacobian = SparseJacobian(f!, res, u, params, pattern; coloring, batchsize, check_pattern)
    builder = AssembledJacobianBuilder(jacobian, factorize; sparsity, coloring, pattern_update_rtol)
    return LaggedPreconditioner(builder; refresh_interval, refresh_iterations)
end
