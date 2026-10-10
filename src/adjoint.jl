##
# Discrete adjoints of steady states (implicit function theorem)
##

using Enzyme: EnzymeRules

# Krylov.jl solves with `transpose(J)` of a Jacobian operator directly: the products
# `Jᵀ v` are computed by Enzyme.jl reverse mode (see `mul!` above). A preconditioner `P ≈ J`
# of the primal system is applied transposed as `ldiv!(y, transpose(P), x)`, which works
# for the factorizations of LinearAlgebra.jl and SparseArrays.jl (e.g., UMFPACK `lu`).

# Solve `A x = b` with `solver` (`Krylov.gmres` or `Krylov.block_gmres`), right-preconditioned
# with `ldiv!(y, Pr, x)` and warm-started from `x0` if they are not `nothing`.
function krylov_solve(solver::S, A, b, x0, Pr; kwargs...) where {S}
    args = x0 === nothing ? (A, b) : (A, b, x0)
    if Pr === nothing
        return solver(args...; kwargs...)
    else
        return solver(args...; N = Pr, ldiv = true, kwargs...)
    end
end

maybe_transpose(::Nothing) = nothing
maybe_transpose(P) = transpose(P)

function warn_unsolved(name, stats)
    stats.solved || @warn "$name: the Krylov solve did not converge" stats
    return stats.solved
end

"""
    adjoint_solve(J::JacobianOperator, g; preconditioner = nothing, λ0 = nothing, kwargs...) -> λ, stats
    adjoint_solve(J::BatchedJacobianOperator{N}, G; preconditioner = nothing, λ0 = nothing, kwargs...) -> Λ, stats

Solve the adjoint system `Jᵀ λ = g` with GMRES, where `J` is the Jacobian of `f!(res, u, p)`
with respect to `u` at `J.u`. The products with `Jᵀ` are computed by Enzyme.jl reverse mode.

If a `preconditioner` `P ≈ J` is given, the system is right-preconditioned with `P⁻ᵀ`, i.e.,
`ldiv!(y, transpose(P), x)`. This works for the factorizations of LinearAlgebra.jl and
SparseArrays.jl (e.g., the `lu` of an assembled Jacobian). For other preconditioners, add a
method `ldiv!(y, ::Transpose{<:Any, <:MyPreconditioner}, x)` or pass an already transposed
preconditioner as `transposed_preconditioner`. `λ0` is an initial guess (warm start).

For a [`BatchedJacobianOperator`](@ref) with `N` columns of right-hand sides `G` (a matrix of
size `length(u) × N`), the `N` adjoint systems are solved together with block GMRES
(`Krylov.block_gmres`), whose products with `Jᵀ` are one batched Enzyme.jl reverse sweep.
`λ0` is then a matrix. This requires Julia 1.11 or later.

The remaining keyword arguments are passed to `Krylov.gmres` or `Krylov.block_gmres`, e.g.,
`rtol`, `atol`, `itmax`, `memory`, and `history`.
"""
function adjoint_solve(
        J::JacobianOperator, g; preconditioner = nothing,
        transposed_preconditioner = maybe_transpose(preconditioner), λ0 = nothing,
        memory = 50, rtol = 1.0e-10, atol = 0.0, kwargs...
    )
    return krylov_solve(
        Krylov.gmres, transpose(J), g, λ0, transposed_preconditioner;
        memory, rtol, atol, kwargs...
    )
end

# Solve the tangent system `J u̇ = rhs`, see `adjoint_solve`
function tangent_solve(
        J::JacobianOperator, rhs; preconditioner = nothing, u̇0 = nothing,
        memory = 50, rtol = 1.0e-10, atol = 0.0, kwargs...
    )
    return krylov_solve(Krylov.gmres, J, rhs, u̇0, preconditioner; memory, rtol, atol, kwargs...)
end

"""
    adjoint_gradient(functional, f!, u, p; preconditioner = nothing, λ0 = nothing, kwargs...)

Discrete adjoint gradient of `functional(u, p)` (a scalar) at a solution `u` of
`f!(res, u, p) = 0` with respect to the parameters `p`, i.e.,
`dJ/dp = ∂J/∂p - λᵀ ∂f/∂p` with `(∂f/∂u)ᵀ λ = (∂J/∂u)ᵀ`:
1. `∂J/∂u` and `∂J/∂p` by one Enzyme.jl reverse-mode sweep through `functional`,
2. `λ` by [`adjoint_solve`](@ref) (GMRES with Enzyme.jl vector-Jacobian products of `f!`),
3. `λᵀ ∂f/∂p` by [`parameter_vjp!`](@ref).

Returns `(; value, dp, λ, stats, solved, dJdu, timings)`, where `dp` is a shadow of `p`
(see `Enzyme.make_zero`) that holds `dJ/dp`. If the adjoint solve did not converge, `solved`
is `false` and a warning is printed. `preconditioner`, `transposed_preconditioner`, the
initial guess `λ0` (e.g., the `λ` of a previous call), and the remaining keyword arguments
are passed to [`adjoint_solve`](@ref). Instead of `f!`, a [`JacobianOperator`](@ref) of `f!`
at `u` can be passed (e.g., the one of a steady-state solver), which is then reused for the
adjoint solve.

    adjoint_gradient(functional!, f!, u, p, Val(N); kwargs...)

Gradients of `N` functionals at once, where `functional!(out, u, p)` writes the `N` values
into the vector `out` (it must overwrite `out`, not accumulate into it). This takes one
batched Enzyme.jl reverse sweep through `functional!`, one block GMRES solve of the `N`
adjoint systems (see [`adjoint_solve`](@ref)), and one batched reverse sweep through `f!`.
Returns `(; value, dp, λ, stats, solved, dJdu, timings)` as above, where `value` is the
vector of the `N` values, `dp` is a tuple of `N` shadows of `p` (the gradient of the `i`-th
functional is `dp[i]`), and `λ` and `dJdu` are matrices with `N` columns. Instead of `f!`, a
`BatchedJacobianOperator{N}` can be passed. This requires Julia 1.11 or later.
"""
function adjoint_gradient(functional::F, f!, u, p; kwargs...) where {F}
    J = JacobianOperator(f!, similar(u), copy(u), p)
    return adjoint_gradient(functional, J, u, p; kwargs...)
end

function adjoint_gradient(functional::F, J::JacobianOperator, u, p; kwargs...) where {F}
    t_functional = @elapsed begin
        dJdu = zero(u)
        dp = Enzyme.make_zero(p)
        _, value = autodiff(
            ReverseWithPrimal, Const(functional), Active,
            Duplicated(copy(u), dJdu), Duplicated(p, dp)
        )
    end
    t_adjoint = @elapsed λ, stats = adjoint_solve(J, vec(dJdu); kwargs...)
    solved = warn_unsolved("adjoint_gradient", stats)
    # dp -= (∂f/∂p)ᵀ λ
    t_parameter_vjp = @elapsed parameter_vjp!(dp, J.f, similar(J.res), u, p, -λ)
    timings = (; functional = t_functional, adjoint = t_adjoint, parameter_vjp = t_parameter_vjp)
    return (; value, dp, λ, stats, solved, dJdu, timings)
end

"""
    parameter_vjp!(p̄, f!, res, u, p, λ)
    parameter_vjp!(p̄s::NTuple{N}, f!, res, u, p, λs::NTuple{N})

Accumulate the vector-Jacobian product `(∂f/∂p)ᵀ λ` of `f!(res, u, p)` with respect to the
parameters into the Enzyme.jl shadow `p̄` of `p`, e.g., `p̄ = Enzyme.make_zero(p)`.
Uses one reverse-mode sweep through `f!` (at the state `u`, which is not modified).
`res` is used as scratch space.

With tuples of `N` shadows `p̄s` and `N` arrays `λs`, the `N` products are computed by one
batched reverse-mode sweep (Julia 1.11 or later).
"""
function parameter_vjp!(p̄, f!::F, res, u, p, λ) where {F}
    f′ = init_cache(f!)
    autodiff(
        Reverse,
        maybe_duplicated(f!, f′), Const,
        Duplicated(res, copy(reshape(λ, size(res)))),
        Duplicated(copy(u), zero(u)),
        Duplicated(p, p̄)
    )
    return p̄
end

"""
    parameter_vjp(f!, res, u, p, λ) -> p̄

Like [`parameter_vjp!`](@ref), but returns a new shadow `p̄ = (∂f/∂p)ᵀ λ` of `p`.
"""
parameter_vjp(f!::F, res, u, p, λ) where {F} = parameter_vjp!(Enzyme.make_zero(p), f!, res, u, p, λ)

if VERSION >= v"1.11.0"

    function adjoint_solve(
            J::BatchedJacobianOperator{N}, G::AbstractMatrix; preconditioner = nothing,
            transposed_preconditioner = maybe_transpose(preconditioner), λ0 = nothing,
            memory = 50, rtol = 1.0e-10, atol = 0.0, kwargs...
        ) where {N}
        size(G, 2) == N ||
            throw(DimensionMismatch("expected $N right-hand sides, got $(size(G, 2))"))
        # The batched products need the columns of dense matrices
        G = convert(Matrix{eltype(J)}, G)
        return krylov_solve(
            Krylov.block_gmres, transpose(J), G, λ0, transposed_preconditioner;
            memory, rtol, atol, kwargs...
        )
    end

    function tangent_solve(
            J::BatchedJacobianOperator{N}, RHS::AbstractMatrix; preconditioner = nothing,
            u̇0 = nothing, memory = 50, rtol = 1.0e-10, atol = 0.0, kwargs...
        ) where {N}
        RHS = convert(Matrix{eltype(J)}, RHS)
        return krylov_solve(
            Krylov.block_gmres, J, RHS, u̇0, preconditioner; memory, rtol, atol, kwargs...
        )
    end

    function adjoint_gradient(functional!::F, f!, u, p, ::Val{N}; kwargs...) where {F, N}
        J = BatchedJacobianOperator{N}(f!, similar(u), copy(u), p)
        return adjoint_gradient(functional!, J, u, p; kwargs...)
    end

    function adjoint_gradient(
            functional!::F, J::BatchedJacobianOperator{N}, u, p; kwargs...
        ) where {F, N}
        T = eltype(u)
        t_functional = @elapsed begin
            value = zeros(T, N)
            # The unit seeds pick the rows of ∂J/∂u and ∂J/∂p
            seeds = ntuple(i -> T.((1:N) .== i), Val(N))
            dJdu = zeros(T, length(u), N)
            dp = ntuple(_ -> Enzyme.make_zero(p), Val(N))
            autodiff(
                Reverse, Const(functional!), Const, BatchDuplicated(value, seeds),
                BatchDuplicated(copy(u), tuple_of_vectors(dJdu, size(u))),
                BatchDuplicated(p, dp)
            )
        end
        t_adjoint = @elapsed λ, stats = adjoint_solve(J, dJdu; kwargs...)
        solved = warn_unsolved("adjoint_gradient", stats)
        # dp[i] -= (∂f/∂p)ᵀ λ[:, i]
        t_parameter_vjp = @elapsed parameter_vjp!(
            dp, J.f, similar(J.res), u, p, tuple_of_vectors(-λ, size(J.res))
        )
        timings = (; functional = t_functional, adjoint = t_adjoint, parameter_vjp = t_parameter_vjp)
        return (; value, dp, λ, stats, solved, dJdu, timings)
    end

    function parameter_vjp!(
            p̄s::NTuple{N, Any}, f!::F, res, u, p, λs::NTuple{N, AbstractArray}
        ) where {N, F}
        f′ = init_cache(f!, Val(N))
        autodiff(
            Reverse,
            maybe_duplicated(f!, f′, Val(N)), Const,
            BatchDuplicated(res, map(λ -> copy(reshape(λ, size(res))), λs)),
            BatchDuplicated(copy(u), ntuple(_ -> zero(u), Val(N))),
            BatchDuplicated(p, p̄s)
        )
        return p̄s
    end

end # VERSION >= v"1.11.0"

"""
    ImplicitFunction(f!, solve!; preconditioner = (u, p) -> nothing, adjoint_kwargs = (;), warm_start = false)

The solution `u(p)` of `f!(res, u, p) = 0`, computed by `solve!(u, p)`, which overwrites
`u` (that contains the initial guess) with the solution. Use it with
[`implicit_solve!`](@ref).

For the derivatives with Enzyme.jl, the solver is not differentiated. Instead, the implicit
function theorem `∂u/∂p = -(∂f/∂u)⁻¹ ∂f/∂p` is applied at the solution:
- Reverse mode solves one adjoint system `(∂f/∂u)ᵀ λ = ū` with [`adjoint_solve`](@ref) and
  adds `-(∂f/∂p)ᵀ λ` to the shadow of `p` (see [`parameter_vjp!`](@ref)).
- Forward mode solves `(∂f/∂u) u̇ = -(∂f/∂p) ṗ` with GMRES.

In batched mode (`BatchDuplicated` with width `N > 1`, Julia 1.11 or later), the `N` adjoint
(or tangent) systems are solved together with block GMRES on a
[`BatchedJacobianOperator`](@ref), and the products with `∂f/∂p` are batched Enzyme.jl sweeps.

`preconditioner(u, p)` returns a preconditioner `P ≈ ∂f/∂u` at the solution (anything that
supports `ldiv!`, e.g., the LU factorization of an assembled Jacobian) or `nothing`.
In reverse mode, it is applied transposed (see [`adjoint_solve`](@ref)). `adjoint_kwargs` are
passed to `Krylov.gmres` (or `Krylov.block_gmres`). If `warm_start` is `true`, an adjoint
(tangent) solve starts from the solution `λ` (`u̇`) of the previous adjoint (tangent) solve if
it has the same size, which saves iterations, e.g., in an optimization loop. Krylov.jl
measures `rtol` relative to the initial residual, so this pays off with an absolute
tolerance `atol` in `adjoint_kwargs`.

The statistics of the last adjoint (or tangent) solve are stored in `last_stats[]`, the
last adjoint solution in `last_λ[]`, and the last tangent solution in `last_u̇[]`.
"""
struct ImplicitFunction{F, S, P, K}
    f!::F
    solve!::S
    preconditioner::P
    adjoint_kwargs::K
    warm_start::Bool
    last_stats::Base.RefValue{Any}
    last_λ::Base.RefValue{Any}
    last_u̇::Base.RefValue{Any}
end

function ImplicitFunction(
        f!, solve!; preconditioner = (u, p) -> nothing,
        adjoint_kwargs = (;), warm_start::Bool = false
    )
    return ImplicitFunction(
        f!, solve!, preconditioner, adjoint_kwargs, warm_start,
        Ref{Any}(nothing), Ref{Any}(nothing), Ref{Any}(nothing)
    )
end

# The previous solution in `last[]` as an initial guess for a solve with right-hand side `b`
function initial_guess(F::ImplicitFunction, last, b)
    x0 = last[]
    return F.warm_start && x0 isa typeof(b) && size(x0) == size(b) ? x0 : nothing
end

"""
    implicit_solve!(F::ImplicitFunction, u, p)

Overwrite `u` with the solution of `F.f!(res, u, p) = 0` by calling `F.solve!(u, p)`.
Differentiating a function that calls `implicit_solve!` with Enzyme.jl does not
differentiate the solver, but uses the implicit function theorem at the solution, i.e.,
one adjoint solve per reverse pass (see [`ImplicitFunction`](@ref)).

The derivative of the solution with respect to the initial guess in `u` is zero.
The shadows of `p` must only carry derivatives of parameters, i.e., temporary
storage in `p` that is overwritten by `f!` must not carry derivatives into
`implicit_solve!` from later uses.

## Example

```julia
F = ImplicitFunction(f!, (u, p) -> newton_krylov!(f!, u, p))
function objective(u, p)
    implicit_solve!(F, u, p)
    return J(u, p)
end
autodiff(Reverse, Const(objective), Active, Duplicated(u, zero(u)), Duplicated(p, make_zero(p)))
```

Pass a closure that captures an `ImplicitFunction` as `Const(objective)`, as above. Otherwise,
Enzyme.jl may fail to prove that the closure is read-only and throw an
`EnzymeMutabilityException`.
"""
function implicit_solve!(F::ImplicitFunction, u, p)
    F.solve!(u, p)
    return nothing
end

# Tangents of `u` and shadows of the parameters as tuples, whatever the batch width.
# Immutable parameters with mutable fields (e.g., a `NamedTuple` of arrays) can be passed to
# rules as `MixedDuplicated`, whose shadow is in a `Ref`. The derivatives are accumulated in
# the shadows of the mutable fields.
tangents(x::Duplicated) = (x.dval,)
tangents(x::BatchDuplicated) = x.dval
shadows(p::Union{Duplicated, BatchDuplicated}) = tangents(p)
shadows(p::MixedDuplicated) = (p.dval[],)
shadows(p::BatchMixedDuplicated) = map(getindex, p.dval)

function check_width(N)
    return N == 1 || VERSION >= v"1.11.0" ||
        error("implicit_solve!: batched mode requires Julia 1.11 or later")
end

function EnzymeRules.augmented_primal(
        config, func::Const{typeof(implicit_solve!)},
        ::Type{RT}, F::Const{<:ImplicitFunction}, u::Annotation,
        p::Annotation
    ) where {RT}
    check_width(EnzymeRules.width(config))
    F.val.solve!(u.val, p.val)
    # The solution is needed in the reverse pass. Copy it if `u` is overwritten later.
    tape = EnzymeRules.overwritten(config)[3] ? copy(u.val) : u.val
    return EnzymeRules.AugmentedReturn(nothing, nothing, tape)
end

function EnzymeRules.reverse(
        config, func::Const{typeof(implicit_solve!)},
        ::Type{RT}, tape, F::Const{<:ImplicitFunction}, u::Annotation,
        p::Annotation
    ) where {RT}
    F = F.val
    N = EnzymeRules.width(config)
    if !(u isa Const) && !(p isa Const)
        ū = tangents(u)
        u_star = tape
        res = similar(u_star)
        P = F.preconditioner(u_star, p.val)
        if N == 1
            J = JacobianOperator(F.f!, res, u_star, p.val)
            g = vec(only(ū))
            λ0 = initial_guess(F, F.last_λ, g)
            λ, stats = adjoint_solve(J, g; preconditioner = P, λ0, F.adjoint_kwargs...)
            # p̄ -= (∂f/∂p)ᵀ λ
            parameter_vjp!(only(shadows(p)), F.f!, res, u_star, p.val, -λ)
        else
            J = BatchedJacobianOperator{N}(F.f!, res, u_star, p.val)
            G = stack(vec, ū)
            λ0 = initial_guess(F, F.last_λ, G)
            λ, stats = adjoint_solve(J, G; preconditioner = P, λ0, F.adjoint_kwargs...)
            # p̄[i] -= (∂f/∂p)ᵀ λ[:, i]
            parameter_vjp!(shadows(p), F.f!, res, u_star, p.val, tuple_of_vectors(-λ, size(res)))
        end
        F.last_stats[] = stats
        F.last_λ[] = λ
        warn_unsolved("implicit_solve!", stats)
    end
    if !(u isa Const)
        # `u` is overwritten by the solution, which does not depend on the initial guess
        foreach(ū -> fill!(ū, 0), tangents(u))
    end
    return (nothing, nothing, nothing)
end

function EnzymeRules.forward(
        config, func::Const{typeof(implicit_solve!)},
        ::Type{RT}, F::Const{<:ImplicitFunction}, u::Annotation,
        p::Annotation
    ) where {RT}
    N = EnzymeRules.width(config)
    check_width(N)
    F = F.val
    F.solve!(u.val, p.val)
    u isa Const && return nothing
    if p isa Const
        foreach(u̇ -> fill!(u̇, 0), tangents(u))
        return nothing
    end
    res = similar(u.val)
    P = F.preconditioner(u.val, p.val)
    # rhs = -(∂f/∂p) ṗ at the solution
    if N == 1
        rhs = zero(u.val)
        autodiff(
            Forward, maybe_duplicated(F.f!, init_cache(F.f!)), Const,
            Duplicated(res, rhs), Duplicated(copy(u.val), zero(u.val)),
            Duplicated(p.val, only(shadows(p)))
        )
        rhs .*= -1
        J = JacobianOperator(F.f!, res, copy(u.val), p.val)
        rhs = vec(rhs)
        u̇0 = initial_guess(F, F.last_u̇, rhs)
        u̇, stats = tangent_solve(J, rhs; preconditioner = P, u̇0, F.adjoint_kwargs...)
        copyto!(only(tangents(u)), u̇)
    else
        RHS = zeros(eltype(u.val), length(u.val), N)
        autodiff(
            Forward, maybe_duplicated(F.f!, init_cache(F.f!, Val(N)), Val(N)), Const,
            BatchDuplicated(res, tuple_of_vectors(RHS, size(res))),
            BatchDuplicated(copy(u.val), ntuple(_ -> zero(u.val), Val(N))),
            BatchDuplicated(p.val, shadows(p))
        )
        RHS .*= -1
        J = BatchedJacobianOperator{N}(F.f!, res, copy(u.val), p.val)
        u̇0 = initial_guess(F, F.last_u̇, RHS)
        u̇, stats = tangent_solve(J, RHS; preconditioner = P, u̇0, F.adjoint_kwargs...)
        foreach((ẋ, i) -> copyto!(ẋ, view(u̇, :, i)), tangents(u), 1:N)
    end
    F.last_stats[] = stats
    F.last_u̇[] = u̇
    warn_unsolved("implicit_solve!", stats)
    return nothing
end
