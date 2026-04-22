module Ariadne

export newton_krylov, newton_krylov!, NewtonKrylovWorkspace

using Krylov
using LinearAlgebra, SparseArrays

##
# JacobianOperator
##
import LinearAlgebra: mul!

abstract type AbstractJacobianOperator end

# Interface:
# Base.size(J::AbstractJacobianOperator)
# Base.eltype(J::AbstractJacobianOperator)
# Base.length(J::AbstractJacobianOperator)
# mul!(out, J::AbstractJacobianOperator, v)
# LinearAlgebra.adjoint(J::AbstractJacobianOperator)
# LinearAlgebra.transpose(J::AbstractJacobianOperator)
# mul!(out, J′::Union{Adjoint{<:Any, <:AbstractJacobianOperator}, Transpose{<:Any, <:AbstractJacobianOperator}}, v)


include("operators/enzyme.jl")
include("operators/di.jl")

const JacobianOperator = EnzymeJacobianOperator

function Base.collect(JOp::Union{Adjoint{<:Any, <:AbstractJacobianOperator}, Transpose{<:Any, <:AbstractJacobianOperator}, AbstractJacobianOperator})
    N, M = size(JOp)
    if JOp isa JacobianOperator
        v = zero(JOp.u)
        out = zero(JOp.res)
    else
        v = zero(parent(JOp).res)
        out = zero(parent(JOp).u)
    end
    J = SparseMatrixCSC{eltype(v), Int}(undef, size(JOp)...)
    for j in 1:M
        out .= 0.0
        v .= 0.0
        v[j] = 1.0
        mul!(out, JOp, v)
        for i in 1:N
            if out[i] != 0
                J[i, j] = out[i]
            end
        end
    end
    return J
end

##
# Newton-Krylov
##
import Base: @kwdef

"""
    Forcing

Implements forcing for inexact Newton-Krylov.
The equation ``‖F′(u)d + F(u)‖ <= η * ‖F(u)‖`` gives
the inexact Newton termination criterion.

## Implemented variants
- [`Fixed`](@ref)
- [`EisenstatWalker`](@ref)
"""
abstract type Forcing end

"""
    Fixed(η = 0.1)
"""
@kwdef struct Fixed <: Forcing
    η::Float64 = 0.1
end

function (F::Fixed)(args...)
    return F.η
end
initial(F::Fixed) = F.η

"""
    EisenstatWalker(η_max = 0.999, γ = 0.9)
"""
@kwdef struct EisenstatWalker <: Forcing
    η_max::Float64 = 0.999
    γ::Float64 = 0.9
end

"""
Compute the Eisenstat-Walker forcing term for n > 0
"""
function (F::EisenstatWalker)(η, tol, norm_res, norm_res_prior)
    η_res = F.γ * norm_res^2 / norm_res_prior^2
    # Eq 3.6
    if F.γ * η^2 <= 1 // 10
        η_safe = min(F.η_max, η_res)
    else
        η_safe = min(F.η_max, max(η_res, F.γ * η^2))
    end
    return min(F.η_max, max(η_safe, 1 // 2 * tol / norm_res)) # Eq 3.5
end
initial(F::EisenstatWalker) = F.η_max

struct Stats
    outer_iterations::Int
    inner_iterations::Int
    norm_res::Float64
end
function update(stats::Stats, inner_iterations, norm_res::Float64)
    return Stats(
        stats.outer_iterations + 1,
        stats.inner_iterations + inner_iterations,
        norm_res
    )
end

"""
    NewtonKrylovWorkspace

Pre-allocated workspace for [`newton_krylov!`](@ref).
Holds the residual buffer, negated-residual buffer, Jacobian operator (with its
Enzyme caches), and the Krylov solver workspace so that no intermediate arrays
are allocated during the Newton iteration.

!!! note
    To change the parameters `p` you have to create a new workspace.
    To change the initial guess `u`, you can pass it to [`newton_krylov!(ws, u)`](@ref).

## Constructor

    NewtonKrylovWorkspace(F!, u, p, res, alg=Val(:gmres); assume_p_const = false)

- `F!`: in-place residual function `F!(res, u, p)`
- `u`: initial-guess array (used as template; the workspace holds a reference to it)
- `p`: parameters
- `res`: pre-allocated residual buffer
- `algo`: Krylov algorithm symbol (e.g. `:gmres`, `:fgmres`) passed as a `Val`.
- `assume_p_const`: passed through to [`JacobianOperator`](@ref)

## Example

```julia
    ws = NewtonKrylovWorkspace(F!, u, p, res, Val(:gmres))
    newton_krylov!(ws)

    # To change the initial guess, pass it to newton_krylov!:
    newton_krylov!(ws, u_new)    
```

"""
struct NewtonKrylovWorkspace{F, A, P, JOp <: AbstractJacobianOperator, KW}
    f::F
    u::A
    res::A
    neg_res::A
    p::P
    J::JOp
    krylov::KW
end

function NewtonKrylovWorkspace(
        F!, u::AbstractArray, p, res::AbstractArray, ::Val{Algo}=Val(:gmres);
        assume_p_const::Bool = false
    ) where {Algo}
    # res .= 0 might ignore ghost cells
    # memory allocated with similar might contain NaN/Inf
    Enzyme.make_zero!(res)
    neg_res = similar(res)
    Enzyme.make_zero!(neg_res)
    J = JacobianOperator(F!, res, u, p; assume_p_const)
    kc = KrylovConstructor(res)
    krylov = krylov_workspace(Val(Algo), kc)
    return NewtonKrylovWorkspace(F!, u, res, neg_res, p, J, krylov)
end

"""
    Ariadne.evaluate!(ws::NewtonKrylovWorkspace) -> norm_res

Evaluate `F!(ws.res, ws.u, ws.p)` in-place and return `norm(ws.res)`.
"""
function evaluate!(ws::NewtonKrylovWorkspace)
    ws.f(ws.res, ws.u, ws.p)
    return norm(ws.res)
end

##
# LineSearches
##

include("linesearches.jl")
import .LineSearches: AbstractLineSearch, NoLineSearch, BacktrackingLineSearch
export NoLineSearch, BacktrackingLineSearch


const KWARGS_DOCS = """
## Keyword Arguments
  - `tol_rel`: Relative tolerance
  - `tol_abs`: Absolute tolerance
  - `max_niter`: Maximum number of iterations
  - `forcing`: Maximum forcing term for inexact Newton.
             If `nothing` an exact Newton method is used.
  - `linesearch!`: Line search strategy. Must be a subtype of `AbstractLineSearch`.
  - `verbose::Int`: Verbosity level
  - `M::Union{Nothing, Function}`: If provided, `M(ws.J)` is passed as a keyword argument to the Krylov solver.
  - `N::Union{Nothing, Function}`: If provided, `N(ws.J)` is passed as a keyword argument to the Krylov solver.
  - `krylov_kwargs`: Keyword arguments passed to the Krylov solver.
  - `callback`: A function called after each Newton iteration with signature `callback(u, res, norm_res)`.
"""

"""
    newton_krylov(F, u₀::AbstractArray, p = nothing, M::Int = length(u₀); kwargs...)

Takes a out-of-place residual function `F(u, p)`.

## Arguments
  - `F`: `res = F(u₀, p)` solves `res = F(u₀) = 0`
  - `u₀`: Initial guess
  - `p`: Parameters
  - `M`: Length of the output of `F`. Defaults to `length(u₀)`.

$(KWARGS_DOCS)
"""
function newton_krylov(F, u₀::AbstractArray, p = nothing, M::Int = length(u₀); kwargs...)
    F!(res, u, p) = (res .= F(u, p); nothing)
    return newton_krylov!(F!, u₀, p, M; kwargs...)
end

"""
    newton_krylov!(F!, u₀::AbstractArray, p = nothing, M::Int = length(u₀); kwargs...)

Takes an in-place residual function `F!(res, u, p)`.

## Arguments
  - `F!`: `F!(res, u, p)` solves `res = F(u) = 0`
  - `u₀`: Initial guess
  - `p`: Parameters
  - `M`: Length of  the output of `F!`. Defaults to `length(u₀)`

$(KWARGS_DOCS)
"""
function newton_krylov!(F!, u₀::AbstractArray, p = nothing, M::Int = length(u₀); algo::Symbol = :gmres, assume_p_const::Bool = false, kwargs...)
    res = similar(u₀, M)
    ws = NewtonKrylovWorkspace(F!, u₀, p, res, Val(algo); assume_p_const)
    return newton_krylov!(ws; kwargs...)
end

"""
    newton_krylov!(F!, u, p, res; kwargs...)

Takes an in-place residual function `F!(res, u, p)`.

## Arguments
  - `F!`: `F!(res, u, p)` solves `res = F(u) = 0`
  - `u`: Initial guess (modified in-place)
  - `p`: Parameters
  - `res`: Temporary for residual

$(KWARGS_DOCS)
"""
function newton_krylov!(
        F!, u::AbstractArray, p, res::AbstractArray;
        algo::Symbol = :gmres,
        assume_p_const::Bool = false,
        kwargs...,
    )
    ws = NewtonKrylovWorkspace(F!, u, p, res, Val(algo); assume_p_const)
    return newton_krylov!(ws; kwargs...)
end

"""
    newton_krylov!(ws::NewtonKrylovWorkspace, u; kwargs...)

Updates `ws.u` with the initial guess `u` and then calls `newton_krylov!(ws; kwargs...)`.

## Arguments
  - `F!`: `F!(res, u, p)` solves `res = F(u) = 0`
  - `u`: Initial guess (must have the same shape as `ws.u`)

$(KWARGS_DOCS)
"""
function newton_krylov!(ws::NewtonKrylovWorkspace, u::AbstractArray; kwargs...)
    # If a different initial-guess array is provided, copy it into ws.u so that
    # J.u (which aliases ws.u) always reflects the current iterate.
    if ws.u !== u
        ws.u .= u
    end
    return newton_krylov!(ws; kwargs...)
end

"""
    newton_krylov!(ws; kwargs...)

## Arguments
  - `ws`: Pre-allocated [`NewtonKrylovWorkspace`](@ref)

$(KWARGS_DOCS)
"""
function newton_krylov!(
        ws::NewtonKrylovWorkspace;
        tol_rel = 1.0e-6,
        tol_abs = 1.0e-12, # Scipy uses 6e-6
        max_niter = 50,
        forcing::Union{Forcing, Nothing} = EisenstatWalker(),
        linesearch!::AbstractLineSearch = NoLineSearch(),
        verbose = 0,
        M = nothing,
        N = nothing,
        krylov_kwargs = (;),
        callback = (args...) -> nothing,
    )
    t₀ = time_ns()
    norm_res = evaluate!(ws)
    callback(ws.u, ws.res, norm_res)

    tol = tol_rel * norm_res + tol_abs

    if forcing !== nothing
        η = initial(forcing)
    end

    verbose > 0 && @info "Jacobian-Free Newton-Krylov" res₀ = norm_res tol tol_rel tol_abs η

    stats = Stats(0, 0, norm_res)
    while norm_res > tol && stats.outer_iterations <= max_niter
        # Handle kwargs for Preconditioners
        kwargs = krylov_kwargs
        if N !== nothing
            kwargs = (; N = N(ws.J), kwargs...)
        end
        if M !== nothing
            kwargs = (; M = M(ws.J), kwargs...)
        end
        if forcing !== nothing
            # The termination criterion of the inner Krylov solver is
            #   ‖F′(u) d + F(u)‖ <= η ‖F(u)‖
            # i.e., we use an inexact Newton method. Since the initial
            # guess of the Krylov solver is zero, we set an absolute
            # tolerance of `atol = 0` and a relative tolerance of
            # `rtol = η`.
            # Since the user-provided `kwargs` are appended at the end,
            # the user can override these settings.
            kwargs = (; atol = zero(η), rtol = η, kwargs...)
        end

        # Solve: J d = -res = -F(u)
        # The Newton method is formulated as J d = -F(u).
        # `res` is modified by J (Enzyme forward pass writes into it),
        # so we negate into a pre-allocated buffer instead of allocating.
        (; neg_res, res) = ws
        @. neg_res = -res
        krylov_solve!(ws.krylov, ws.J, neg_res; kwargs...)

        d = ws.krylov.x # Newton direction

        # Perform line search to find an appropriate step size and update `u` and `res` in-place
        norm_res_prior = norm_res
        norm_res = linesearch!(ws, norm_res_prior, d)

        callback(ws.u, ws.res, norm_res)

        if isinf(norm_res) || isnan(norm_res)
            @error "Inner solver blew up" stats
            break
        end

        if forcing !== nothing
            η = forcing(η, tol, norm_res, norm_res_prior)
        end

        # This is almost to be expected for implicit time-stepping
        if verbose > 0 && ws.krylov.stats.niter == 0 && forcing !== nothing
            @info "Inexact Newton thinks our step is good enough " η stats
        end

        stats = update(stats, ws.krylov.stats.niter, norm_res)
        verbose > 0 && @info "Newton" iter = norm_res η stats
    end
    t = (time_ns() - t₀) / 1.0e9
    return ws.u, (; solved = norm_res <= tol, stats, t)
end

end # module Ariadne
