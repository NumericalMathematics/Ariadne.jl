##
# Discrete adjoints of steady states (implicit function theorem)
##

using Enzyme: EnzymeRules

"""
    TransposedOperator(J)

The transpose `Jᵀ` of a [`JacobianOperator`](@ref) as an operator that Krylov.jl can use,
i.e., with `size`, `eltype`, and `mul!`. The products `Jᵀ v` are computed by Enzyme.jl
reverse mode (vector-Jacobian products).
"""
struct TransposedOperator{JOp <: JacobianOperator}
    J::JOp
end

Base.size(T::TransposedOperator) = reverse(size(T.J))
Base.size(T::TransposedOperator, d::Integer) = size(T)[d]
Base.eltype(T::TransposedOperator) = eltype(T.J)
Base.length(T::TransposedOperator) = prod(size(T))
LinearAlgebra.mul!(out, T::TransposedOperator, v) = mul!(out, transpose(T.J), v)

"""
    TransposedPreconditioner(P)

The transpose of the preconditioner `P`, applied by `ldiv!(y, TransposedPreconditioner(P), x)`
as `y = P⁻ᵀ x`. It is used to right-precondition the adjoint system with the transpose
of a preconditioner of the primal (Newton) system, e.g., the LU factorization of an
assembled Jacobian.

The transposed solve is done by [`transpose_ldiv!(y, P, x)`](@ref transpose_ldiv!),
which uses `ldiv!(y, transpose(P), x)` by default. This works for the factorizations
of LinearAlgebra.jl and SparseArrays.jl (e.g., UMFPACK `lu`). For other preconditioners,
add a method to [`transpose_ldiv!`](@ref).
"""
struct TransposedPreconditioner{P}
    P::P
end

"""
    transpose_ldiv!(y, P, x)

Compute `y = P⁻ᵀ x` for a preconditioner `P` that is applied via `ldiv!(y, P, x)`.
"""
transpose_ldiv!(y, P, x) = ldiv!(y, transpose(P), x)
transpose_ldiv!(y, P::TransposedPreconditioner, x) = ldiv!(y, P.P, x)

LinearAlgebra.ldiv!(y, P::TransposedPreconditioner, x) = transpose_ldiv!(y, P.P, x)

"""
    adjoint_solve(J::JacobianOperator, g; preconditioner = nothing, kwargs...) -> λ, stats

Solve the adjoint system `Jᵀ λ = g` with GMRES, where `J` is the Jacobian of `f!(res, u, p)`
with respect to `u` at `J.u`. The products with `Jᵀ` are computed by Enzyme.jl reverse mode.
If a `preconditioner` `P ≈ J` is given (anything that supports `ldiv!`, see
[`TransposedPreconditioner`](@ref)), the system is right-preconditioned with `P⁻ᵀ`.

The remaining keyword arguments are passed to `Krylov.gmres`, e.g., `rtol`, `atol`,
`itmax`, `memory`, and `history`.
"""
function adjoint_solve(
        J::JacobianOperator, g; preconditioner = nothing,
        transposed_preconditioner = preconditioner === nothing ? nothing :
            TransposedPreconditioner(preconditioner),
        memory = 50, rtol = 1.0e-10, atol = 0.0, kwargs...
    )
    T = TransposedOperator(J)
    if transposed_preconditioner === nothing
        λ, stats = Krylov.gmres(T, g; memory, rtol, atol, kwargs...)
    else
        λ, stats = Krylov.gmres(
            T, g; N = transposed_preconditioner, ldiv = true, memory, rtol, atol,
            kwargs...
        )
    end
    return λ, stats
end

"""
    adjoint_gradient(functional, f!, u, p; preconditioner = nothing, kwargs...)

Discrete adjoint gradient of `functional(u, p)` (a scalar) at a solution `u` of
`f!(res, u, p) = 0` with respect to the parameters `p`, i.e.,
`dJ/dp = ∂J/∂p - λᵀ ∂f/∂p` with `(∂f/∂u)ᵀ λ = (∂J/∂u)ᵀ`:
1. `∂J/∂u` and `∂J/∂p` by one Enzyme.jl reverse-mode sweep through `functional`,
2. `λ` by [`adjoint_solve`](@ref) (GMRES with Enzyme.jl vector-Jacobian products of `f!`),
3. `λᵀ ∂f/∂p` by [`parameter_vjp!`](@ref).

Returns `(; value, dp, λ, stats, dJdu, timings)`, where `dp` is a shadow of `p`
(see `Enzyme.make_zero`) that holds `dJ/dp`. `preconditioner`, `transposed_preconditioner`,
and the remaining keyword arguments are passed to [`adjoint_solve`](@ref). Instead of
`f!`, a [`JacobianOperator`](@ref) of `f!` at `u` can be passed (e.g., the one of a steady-state
solver), which is then reused for the adjoint solve.
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
    # dp -= (∂f/∂p)ᵀ λ
    t_parameter_vjp = @elapsed parameter_vjp!(dp, J.f, similar(J.res), u, p, -λ)
    timings = (; functional = t_functional, adjoint = t_adjoint, parameter_vjp = t_parameter_vjp)
    return (; value, dp, λ, stats, dJdu, timings)
end

"""
    parameter_vjp!(p̄, f!, res, u, p, λ)

Accumulate the vector-Jacobian product `(∂f/∂p)ᵀ λ` of `f!(res, u, p)` with respect to the
parameters into the Enzyme.jl shadow `p̄` of `p`, e.g., `p̄ = Enzyme.make_zero(p)`.
Uses one reverse-mode sweep through `f!` (at the state `u`, which is not modified).
`res` is used as scratch space.
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

"""
    ImplicitFunction(f!, solve!; preconditioner = (u, p) -> nothing, adjoint_kwargs = (;))

The solution `u(p)` of `f!(res, u, p) = 0`, computed by `solve!(u, p)`, which overwrites
`u` (that contains the initial guess) with the solution. Use it with
[`implicit_solve!`](@ref).

For the derivatives with Enzyme.jl, the solver is not differentiated. Instead, the implicit
function theorem `∂u/∂p = -(∂f/∂u)⁻¹ ∂f/∂p` is applied at the solution:
- Reverse mode solves one adjoint system `(∂f/∂u)ᵀ λ = ū` with [`adjoint_solve`](@ref) and
  adds `-(∂f/∂p)ᵀ λ` to the shadow of `p` (see [`parameter_vjp!`](@ref)).
- Forward mode solves `(∂f/∂u) u̇ = -(∂f/∂p) ṗ` with GMRES.

`preconditioner(u, p)` returns a preconditioner `P ≈ ∂f/∂u` at the solution (anything that
supports `ldiv!`, e.g., the LU factorization of an assembled Jacobian) or `nothing`.
In reverse mode, it is
applied transposed (see [`TransposedPreconditioner`](@ref)). `adjoint_kwargs` are passed to
`Krylov.gmres`.

The statistics of the last adjoint (or tangent) solve are stored in `last_stats[]`.
"""
struct ImplicitFunction{F, S, P, K}
    f!::F
    solve!::S
    preconditioner::P
    adjoint_kwargs::K
    last_stats::Base.RefValue{Any}
end

function ImplicitFunction(
        f!, solve!; preconditioner = (u, p) -> nothing,
        adjoint_kwargs = (;)
    )
    return ImplicitFunction(f!, solve!, preconditioner, adjoint_kwargs, Ref{Any}(nothing))
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
autodiff(Reverse, objective, Active, Duplicated(u, zero(u)), Duplicated(p, make_zero(p)))
```
"""
function implicit_solve!(F::ImplicitFunction, u, p)
    F.solve!(u, p)
    return nothing
end

# Shadow of the parameters. Immutable parameters with mutable fields (e.g., a `NamedTuple`
# of arrays) can be passed to rules as `MixedDuplicated`, whose shadow is in a `Ref`.
# The derivatives are accumulated in the shadows of the mutable fields.
shadow(p::Duplicated) = p.dval
shadow(p::MixedDuplicated) = p.dval[]

function EnzymeRules.augmented_primal(
        config, func::Const{typeof(implicit_solve!)},
        ::Type{RT}, F::Const{<:ImplicitFunction}, u::Annotation,
        p::Annotation
    ) where {RT}
    EnzymeRules.width(config) == 1 ||
        error("implicit_solve!: batched reverse mode is not supported")
    F.val.solve!(u.val, p.val)
    # The solution is needed in the reverse pass, but `u` may be overwritten later
    tape = copy(u.val)
    return EnzymeRules.AugmentedReturn(nothing, nothing, tape)
end

function EnzymeRules.reverse(
        config, func::Const{typeof(implicit_solve!)},
        ::Type{RT}, tape, F::Const{<:ImplicitFunction}, u::Annotation,
        p::Annotation
    ) where {RT}
    F = F.val
    if u isa Duplicated && !(p isa Const)
        ū = u.dval
        u_star = tape
        res = similar(u_star)
        J = JacobianOperator(F.f!, res, u_star, p.val)
        P = F.preconditioner(u_star, p.val)
        λ, stats = adjoint_solve(J, vec(ū); preconditioner = P, F.adjoint_kwargs...)
        F.last_stats[] = stats
        stats.solved || @warn "implicit_solve!: adjoint GMRES did not converge" stats
        # p̄ -= (∂f/∂p)ᵀ λ
        λ .*= -1
        parameter_vjp!(shadow(p), F.f!, res, u_star, p.val, λ)
    end
    if u isa Duplicated
        # `u` is overwritten by the solution, which does not depend on the initial guess
        fill!(u.dval, 0)
    end
    return (nothing, nothing, nothing)
end

function EnzymeRules.forward(
        config, func::Const{typeof(implicit_solve!)},
        ::Type{RT}, F::Const{<:ImplicitFunction}, u::Annotation,
        p::Annotation
    ) where {RT}
    EnzymeRules.width(config) == 1 ||
        error("implicit_solve!: batched forward mode is not supported")
    F = F.val
    F.solve!(u.val, p.val)
    if u isa Duplicated
        if p isa Const
            fill!(u.dval, 0)
        else
            # rhs = (∂f/∂p) ṗ at the solution
            res = similar(u.val)
            rhs = similar(u.val)
            f′ = init_cache(F.f!)
            autodiff(
                Forward, maybe_duplicated(F.f!, f′), Const,
                Duplicated(res, rhs), Duplicated(copy(u.val), zero(u.val)),
                Duplicated(p.val, p.dval)
            )
            rhs .*= -1
            J = JacobianOperator(F.f!, res, copy(u.val), p.val)
            P = F.preconditioner(u.val, p.val)
            kwargs = (; memory = 50, rtol = 1.0e-10, atol = 0.0, F.adjoint_kwargs...)
            if P === nothing
                du, stats = Krylov.gmres(J, vec(rhs); kwargs...)
            else
                du, stats = Krylov.gmres(J, vec(rhs); N = P, ldiv = true, kwargs...)
            end
            F.last_stats[] = stats
            copyto!(u.dval, du)
        end
    end
    return nothing
end
