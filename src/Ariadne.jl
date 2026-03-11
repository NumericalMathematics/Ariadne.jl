module Ariadne

export newton_krylov, newton_krylov!

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

# @assert η_max === nothing || 0.0 < η_max < 1.0

"""
Compute the Eisenstat-Walker forcing term for n > 0
"""
function (F::EisenstatWalker)(η, tol, n_res, n_res_prior)
    η_res = F.γ * n_res^2 / n_res_prior^2
    # Eq 3.6
    if F.γ * η^2 <= 1 // 10
        η_safe = min(F.η_max, η_res)
    else
        η_safe = min(F.η_max, max(η_res, F.γ * η^2))
    end
    return min(F.η_max, max(η_safe, 1 // 2 * tol / n_res)) # Eq 3.5
end
initial(F::EisenstatWalker) = F.η_max

const KWARGS_DOCS = """
## Keyword Arguments
  - `tol_rel`: Relative tolerance
  - `tol_abs`: Absolute tolerance
  - `max_niter`: Maximum number of iterations
  - `forcing`: Maximum forcing term for inexact Newton.
             If `nothing` an exact Newton method is used.
  - `verbose`:
  - `Workspace`:
  - `M`:
  - `N`:
  - `krylov_kwarg`
  - `callback`:
"""

"""
    newton_krylov(F, u₀::AbstractArray, M::Int = length(u₀); kwargs...)

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
## Arguments
  - `F!`: `F!(res, u, p)` solves `res = F(u) = 0`
  - `u₀`: Initial guess
  - `p`: Parameters
  - `M`: Length of  the output of `F!`. Defaults to `length(u₀)`

$(KWARGS_DOCS)
"""
function newton_krylov!(F!, u₀::AbstractArray, p = nothing, M::Int = length(u₀); kwargs...)
    res = similar(u₀, M)
    Enzyme.make_zero!(res) # u₀ .= 0 might ignore ghost cells
    return newton_krylov!(F!, u₀, p, res; kwargs...)
end

struct Stats
    outer_iterations::Int
    inner_iterations::Int
    n_res::Float64
end
function update(stats::Stats, inner_iterations, n_res::Float64)
    return Stats(
        stats.outer_iterations + 1,
        stats.inner_iterations + inner_iterations,
        n_res
    )
end

"""

## Arguments
  - `F!`: `F!(res, u, p)` solves `res = F(u) = 0`
  - `u`: Initial guess
  - `p`:
  - `res`: Temporary for residual

$(KWARGS_DOCS)
"""
function newton_krylov!(
        F!, u::AbstractArray, p, res::AbstractArray;
        tol_rel = 1.0e-6,
        tol_abs = 1.0e-12, # Scipy uses 6e-6
        max_niter = 50,
        forcing::Union{Forcing, Nothing} = EisenstatWalker(),
        verbose = 0,
        algo = :gmres,
        M = nothing,
        N = nothing,
        krylov_kwargs = (;),
        callback = (args...) -> nothing,
    )
    t₀ = time_ns()
    F!(res, u, p) # res = F(u)
    n_res = norm(res)
    callback(u, res, n_res)

    tol = tol_rel * n_res + tol_abs

    if forcing !== nothing
        η = initial(forcing)
    end

    verbose > 0 && @info "Jacobian-Free Newton-Krylov" algo res₀ = n_res tol tol_rel tol_abs η

    J = JacobianOperator(F!, res, u, p)

    # TODO: Refactor to provide method that re-uses the cache here.
    kc = KrylovConstructor(res)
    workspace = krylov_workspace(algo, kc)

    stats = Stats(0, 0, n_res)
    while n_res > tol && stats.outer_iterations <= max_niter
        # Handle kwargs for Preconditioners
        kwargs = krylov_kwargs
        if N !== nothing
            kwargs = (; N = N(J), kwargs...)
        end
        if M !== nothing
            kwargs = (; M = M(J), kwargs...)
        end
        if forcing !== nothing
            # The termination cirterion of the inner Krylov solver is
            #   ‖F′(u) d + F(u)‖ <= η ‖F(u)‖
            # i.e., we use an inexact Newton method. Since the initial
            # guess of the Krylov solver is zero, we set an absolute
            # tolerance of `atol = 0` and a relative tolerance of
            # `rtol = η`.
            # Since the user-provided `kwargs` are appended at the end,
            # the user can override these settings.
            kwargs = (; atol = zero(η), rtol = η, kwargs...)
        end

        # Solve: J d = res = F(u)
        # Typically, the Newton method is formulated as J d = -F(u)
        # with update u = u + d.
        # To simplify the implementation, we solve J d = F(u)
        # and update u = u - d instead.
        # `res` is modified by J, so we create a copy `res`
        # TODO: provide a temporary storage for `res`
        krylov_solve!(workspace, J, copy(res); kwargs...)

        d = workspace.x # (negative) Newton direction
        s = 1           # Scaling of the Newton step TODO: LineSearch

        # Update u
        u .= muladd.(-s, d, u) # u = u - s * d

        # Update residual and norm
        n_res_prior = n_res

        F!(res, u, p) # res = F(u)
        n_res = norm(res)
        callback(u, res, n_res)

        if isinf(n_res) || isnan(n_res)
            @error "Inner solver blew up" stats
            break
        end

        if forcing !== nothing
            η = forcing(η, tol, n_res, n_res_prior)
        end

        # This is almost to be expected for implicit time-stepping
        if verbose > 0 && workspace.stats.niter == 0 && forcing !== nothing
            @info "Inexact Newton thinks our step is good enough " η stats
        end

        stats = update(stats, workspace.stats.niter, n_res)
        verbose > 0 && @info "Newton" iter = n_res η stats
    end
    t = (time_ns() - t₀) / 1.0e9
    return u, (; solved = n_res <= tol, stats, t)
end

end # module Ariadne
