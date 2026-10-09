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
    AssembledJacobianPreconditioner(; sparsity, colors = nothing, batchsize = 8,
                                      factorize = lu, refresh_interval = 1,
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
- `colors`: column colors, a function `pattern -> colors`, or `nothing` for
  [`greedy_column_coloring`](@ref Ariadne.greedy_column_coloring).
- `batchsize`: number of colors computed by one batched forward-mode pass.
- `factorize`: function `A -> F` with `ldiv!(y, F, x)`, e.g., `lu`,
  `A -> IncompleteLU.ilu(A; τ = 1e-3)`, or [`RowScaled`](@ref)`(…)`.
- `refresh_interval`, `refresh_iterations`: refresh policy of the resulting
  [`LaggedPreconditioner`](@ref Ariadne.LaggedPreconditioner).
- `check_pattern`: check each assembly for nonzeros outside the pattern.
- `task_parameters`: `nothing`, or a function `p -> ps` returning independent copies of the
  parameters, one per task, for the parallel [`assemble!`](@ref Ariadne.assemble!) over batches of colors
  (see [`PerTaskParameters`](@ref Ariadne.PerTaskParameters)). It is called again when `p` is a different object.

In pseudo-transient continuation, the factorized matrix is `Diagonal(1 ./ Δτ) - σ ∂f/∂u`
with the pseudo-time steps `Δτ` of the PTC step in which the preconditioner is rebuilt
(see [`PseudoTransientNewtonKrylov`](@ref) for the sign `σ`). For a plain Newton-Krylov
solve, it is `∂f/∂u`.
"""
Base.@kwdef struct AssembledJacobianPreconditioner{S, C, F, TP}
    sparsity::S
    colors::C = nothing
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
                             sparsity = nothing, colors = nothing, task_parameters = nothing,
                             pattern_update_rtol = sqrt(eps()))

Callable `J -> factorization` used as `build` of a [`LaggedPreconditioner`](@ref Ariadne.LaggedPreconditioner): assembles
the Jacobian of the residual of the [`JacobianOperator`](@ref Ariadne.JacobianOperator) `J` with `jacobian` and
factorizes it. For the pseudo-transient residual of [`PseudoTransientNewtonKrylov`](@ref),
it assembles `∂f/∂u` of the steady residual `f!` and factorizes
`Diagonal(1 ./ Δτ) - σ ∂f/∂u`. Timings are accumulated in the fields
`assembly_time` and `factorization_time`. See [`AssembledJacobianPreconditioner`](@ref) for
the keyword arguments.
"""
mutable struct AssembledJacobianBuilder{SJ <: SparseJacobian, F, M, S, C, TP}
    jacobian::SJ
    const factorize::F
    matrix::M # matrix that is factorized
    diagonal_indices::Vector{Int}
    factorization_time::Float64
    const sparsity::S # `nothing` or `(f!, u, p) -> pattern` to update the pattern
    const colors::C
    const task_parameters::TP
    const pattern_update_rtol::Float64
    task_parameters_cache::Any # (p, PerTaskParameters)
    n_pattern_updates::Int
    previous_assembly_time::Float64 # of the replaced `SparseJacobian`s
end

function AssembledJacobianBuilder(
        jacobian::SparseJacobian, factorize = lu; sparsity = nothing, colors = nothing,
        task_parameters = nothing, pattern_update_rtol = sqrt(eps())
    )
    matrix, diagonal_indices = factorization_matrix(jacobian)
    return AssembledJacobianBuilder(
        jacobian, factorize, matrix, diagonal_indices, 0.0, sparsity, colors, task_parameters,
        Float64(pattern_update_rtol), nothing, 0, 0.0
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

function assemble_jacobian!(b::AssembledJacobianBuilder, f!, u, p)
    if b.task_parameters === nothing
        return assemble!(b.jacobian, f!, u, p)
    end
    cache = b.task_parameters_cache
    if cache === nothing || cache[1] !== p
        cache = (p, PerTaskParameters(b.task_parameters(p)))
        b.task_parameters_cache = cache
    end
    return assemble!(b.jacobian, f!, u, cache[2])
end

# Assemble ∂f/∂u and, if the pattern misses nonzeros larger than `pattern_update_rtol`
# times the largest entry and can be detected again, merge the pattern at `u` into the
# pattern, recolor, and assemble again
function assemble_and_update!(b::AssembledJacobianBuilder, f!, u, p)
    Jf = assemble_jacobian!(b, f!, u, p)
    A = b.jacobian
    if A.missed_entries > 0 && b.sparsity !== nothing &&
            A.missed_max > b.pattern_update_rtol * maximum(abs, nonzeros(Jf); init = zero(A.missed_max))
        J = A.J
        old = SparseMatrixCSC(size(J)..., copy(SparseArrays.getcolptr(J)), copy(rowvals(J)), fill(true, nnz(J)))
        pattern = old .| Ariadne.bool_pattern(b.sparsity(f!, u, p))
        b.previous_assembly_time += A.time
        b.jacobian = SparseJacobian(
            pattern; colors = b.colors isa AbstractVector ? nothing : b.colors, batchsize = Ariadne.batch_size(A), eltype = eltype(J),
            A.check_pattern
        )
        b.jacobian.n_assemblies = A.n_assemblies
        b.matrix, b.diagonal_indices = factorization_matrix(b.jacobian)
        b.n_pattern_updates += 1
        Jf = assemble_jacobian!(b, f!, u, p)
    end
    return Jf
end

function (b::AssembledJacobianBuilder)(J::JacobianOperator)
    if J.f isa PseudoTransientResidual
        Jf = assemble_and_update!(b, J.f.f, J.u, J.p.p)
        A = b.matrix
        σ = J.f.σ
        vals = nonzeros(A)
        @. vals = -σ * $nonzeros(Jf)
        inv_dtau = J.p.inv_dtau
        for (j, idx) in enumerate(b.diagonal_indices)
            vals[idx] += inv_dtau[j]
        end
    else
        Jf = assemble_and_update!(b, J.f, J.u, J.p)
        A = b.matrix
        copyto!(nonzeros(A), nonzeros(Jf))
    end
    t₀ = time_ns()
    F = b.factorize(A)
    b.factorization_time += (time_ns() - t₀) / 1.0e9
    return F
end

"""
    assembled_preconditioner(f!, u, p, spec::AssembledJacobianPreconditioner)
    assembled_preconditioner(pattern; kwargs...)

Create the [`LaggedPreconditioner`](@ref Ariadne.LaggedPreconditioner) described by `spec` for the residual
`f!(res, u, p)`, for use with [`newton_krylov!`](@ref Ariadne.newton_krylov!) (`N = P`,
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
        pattern; spec.colors, spec.batchsize, spec.factorize, spec.refresh_interval,
        spec.refresh_iterations, spec.check_pattern, spec.task_parameters, sparsity,
        spec.pattern_update_rtol
    )
end

function assembled_preconditioner(
        pattern::AbstractMatrix; colors = nothing, batchsize = 8, factorize = lu,
        refresh_interval = 1, refresh_iterations = typemax(Int), check_pattern = true,
        task_parameters = nothing, sparsity = nothing, pattern_update_rtol = sqrt(eps())
    )
    jacobian = SparseJacobian(pattern; colors, batchsize, check_pattern)
    builder = AssembledJacobianBuilder(jacobian, factorize; sparsity, colors, task_parameters, pattern_update_rtol)
    return LaggedPreconditioner(builder; refresh_interval, refresh_iterations)
end
