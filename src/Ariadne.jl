module Ariadne

export newton_krylov, newton_krylov!, NewtonKrylovWorkspace

using Krylov
using LinearAlgebra, SparseArrays
using Enzyme

##
# JacobianOperator
##
import LinearAlgebra: mul!

function init_cache(x)
    if !Enzyme.Compiler.guaranteed_const(typeof(x))
        Enzyme.make_zero(x)
    else
        return nothing
    end
end

function maybe_duplicated(x::T, x′::Union{Nothing, T}) where {T}
    if x′ === nothing
        return Const(x)
    else
        Enzyme.remake_zero!(x′)
        return Duplicated(x, x′)
    end
end

abstract type AbstractJacobianOperator end


"""
    JacobianOperator

Efficient implementation of `J(f,x,p) * v` and `v * J(f, x,p)'`
"""
struct JacobianOperator{F, F′, A, P, P′} <: AbstractJacobianOperator
    f::F # F!(res, u, p)
    f′::F′ # cache
    res::A
    u::A
    p::P
    p′::P′ # cache
end

"""
    JacobianOperator(f::F, res, u, p; assume_p_const::Bool = false)

Creates a Jacobian operator for `f!(res, u, p)` where `res` is the residual,
`u` is the state variable, and `p` are the parameters.

If `assume_p_const` is `true`, the parameters `p` are assumed to be constant
during the Jacobian computation, which can improve performance by not requiring the
shadow for `p`.
"""
function JacobianOperator(f::F, res, u, p; assume_p_const::Bool = false) where {F}
    f′ = init_cache(f)
    if assume_p_const
        p′ = nothing
    else
        p′ = init_cache(p)
    end
    return JacobianOperator(f, f′, res, u, p, p′)
end

batch_size(::JacobianOperator) = 1

Base.size(J::JacobianOperator) = (length(J.res), length(J.u))
Base.eltype(J::JacobianOperator) = eltype(J.u)
Base.length(J::JacobianOperator) = prod(size(J))

function mul!(out, J::JacobianOperator, v)
    # `out` is the shadow of `res`. If `F!` accumulates into `res`
    # instead of overwriting it, stale values in `out` would leak into the result.
    fill!(out, 0)
    autodiff(
        Forward,
        maybe_duplicated(J.f, J.f′), Const,
        Duplicated(J.res, reshape(out, size(J.res))),
        Duplicated(J.u, reshape(v, size(J.u))),
        maybe_duplicated(J.p, J.p′)
    )
    return nothing
end

LinearAlgebra.adjoint(J::JacobianOperator) = Adjoint(J)
LinearAlgebra.transpose(J::JacobianOperator) = Transpose(J)

# Jᵀ(y, u) = ForwardDiff.gradient!(y, x -> dot(F(x), u), xk)
# or just reverse mode

function mul!(out, J′::Union{Adjoint{<:Any, <:JacobianOperator}, Transpose{<:Any, <:JacobianOperator}}, v)
    J = parent(J′)
    # TODO: provide cache for `copy(v)`
    # Enzyme zeros input derivatives and that confuses the solvers.
    # If `out` is non-zero we might get spurious gradients
    fill!(out, 0)
    autodiff(
        Reverse,
        maybe_duplicated(J.f, J.f′), Const,
        Duplicated(J.res, reshape(copy(v), size(J.res))),
        Duplicated(J.u, reshape(out, size(J.u))),
        maybe_duplicated(J.p, J.p′)
    )
    return nothing
end


function init_cache(x, ::Val{N}) where {N}
    if !Enzyme.Compiler.guaranteed_const(typeof(x))
        return ntuple(_ -> Enzyme.make_zero(x), Val(N))
    else
        return nothing
    end
end

function maybe_duplicated(x::T, x′::Union{Nothing, NTuple{N, T}}, ::Val{N}) where {T, N}
    if x′ === nothing
        return Const(x)
    else
        Enzyme.remake_zero!(x′)
        return BatchDuplicated(x, x′)
    end
end

"""
    BatchedJacobianOperator{N}


"""
struct BatchedJacobianOperator{N, F, A, P} <: AbstractJacobianOperator
    f::F # F!(res, u, p)
    f′::Union{Nothing, NTuple{N, F}} # cache
    res::A
    u::A
    p::P
    p′::Union{Nothing, NTuple{N, P}} # cache
    function BatchedJacobianOperator{N}(f::F, res, u, p) where {F, N}
        f′ = init_cache(f, Val(N))
        p′ = init_cache(p, Val(N))
        return new{N, F, typeof(u), typeof(p)}(f, f′, res, u, p, p′)
    end
end

batch_size(::BatchedJacobianOperator{N}) where {N} = N

Base.size(J::BatchedJacobianOperator) = (length(J.res), length(J.u))
Base.eltype(J::BatchedJacobianOperator) = eltype(J.u)
Base.length(J::BatchedJacobianOperator) = prod(size(J))

LinearAlgebra.adjoint(J::BatchedJacobianOperator) = Adjoint(J)
LinearAlgebra.transpose(J::BatchedJacobianOperator) = Transpose(J)

if VERSION >= v"1.11.0"

    function tuple_of_vectors(M::Matrix{T}, shape) where {T}
        n, m = size(M)
        return ntuple(m) do i
            vec = Base.wrap(Array, memoryref(M.ref, (i - 1) * n + 1), (n,))
            reshape(vec, shape)
        end
    end

    function mul!(Out, J::BatchedJacobianOperator{N}, V) where {N}
        @assert size(Out, 2) == size(V, 2)
        out = tuple_of_vectors(Out, size(J.res))
        v = tuple_of_vectors(V, size(J.u))

        @assert N == length(out)
        # See the non-batched forward `mul!`
        fill!(Out, 0)
        autodiff(
            Forward,
            maybe_duplicated(J.f, J.f′, Val(N)), Const,
            BatchDuplicated(J.res, out),
            BatchDuplicated(J.u, v),
            maybe_duplicated(J.p, J.p′, Val(N))
        )
        return nothing
    end

    function mul!(Out, J′::Union{Adjoint{<:Any, <:BatchedJacobianOperator{N}}, Transpose{<:Any, <:BatchedJacobianOperator{N}}}, V) where {N}
        J = parent(J′)
        @assert size(Out, 2) == size(V, 2)

        # If `out` is non-zero we might get spurious gradients
        fill!(Out, 0)

        # TODO: provide cache for `copy(v)`
        # Enzyme zeros input derivatives and that confuses the solvers.
        V = copy(V)

        out = tuple_of_vectors(Out, size(J.u))
        v = tuple_of_vectors(V, size(J.res))

        @assert N == length(out)

        autodiff(
            Reverse,
            maybe_duplicated(J.f, J.f′, Val(N)), Const,
            BatchDuplicated(J.res, v),
            BatchDuplicated(J.u, out),
            maybe_duplicated(J.p, J.p′, Val(N))
        )
        return nothing
    end
end # VERSION >= v"1.11.0"

function Base.collect(JOp::Union{Adjoint{<:Any, <:AbstractJacobianOperator}, Transpose{<:Any, <:AbstractJacobianOperator}, AbstractJacobianOperator})
    op = JOp isa AbstractJacobianOperator ? JOp : parent(JOp)
    T = eltype(op)
    N, M = size(JOp)
    # A batched operator is applied to `B` unit vectors at once
    B = batch_size(op)
    if op isa BatchedJacobianOperator
        v = zeros(T, M, B)
        out = zeros(T, N, B)
    else
        # Keep the array types of `u` and `res`, which the Enzyme shadows must match
        is_adjoint = !(JOp isa AbstractJacobianOperator)
        v = zero(is_adjoint ? op.res : op.u)
        out = zero(is_adjoint ? op.u : op.res)
    end
    Is = Int[]
    Js = Int[]
    Vs = T[]
    for j₀ in 1:B:M
        nb = min(B, M - j₀ + 1)
        fill!(v, 0)
        fill!(out, 0)
        for k in 1:nb
            # Linear indices for the non-batched case: `v` may not support `v[j, 1]`
            v[(k - 1) * M + j₀ + k - 1] = 1
        end
        mul!(out, JOp, v)
        for k in 1:nb, i in 1:N
            val = out[(k - 1) * N + i]
            if val != 0
                push!(Is, i)
                push!(Js, j₀ + k - 1)
                push!(Vs, val)
            end
        end
    end
    return sparse(Is, Js, Vs, N, M)
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

Constant forcing term: every inner Krylov solve uses the relative tolerance `η`.
"""
@kwdef struct Fixed <: Forcing
    η::Float64 = 0.1
end

function (F::Fixed)(args...)
    return F.η
end
initial(F::Fixed) = F.η

"""
    EisenstatWalker(η₀ = 0.999, η_max = 0.999, γ = 0.9, α = 2.0)

Adaptive forcing term for inexact Newton-Krylov methods. The forcing term η
is updated each iteration as in [Kelley2022](@cite), Eq. (3.6).

## Parameters
- `η₀`:    Initial forcing term. Default: `0.999`.
- `η_max`: Maximum forcing term; must satisfy `η_max ∈ (0, 1)`. Default: `0.999`.
- `γ`:     Contraction factor for the convergence rate estimate. Default: `0.9`.
- `α`:     Exponent for the convergence rate estimate; should be in `(1, 2]`. Default: `2.0`.

## References
- Kelley, C. T. (2022).
  Solving nonlinear equations with iterative methods:
  Solvers and examples in Julia.
  Society for Industrial and Applied Mathematics.
- Eisenstat, S. C., & Walker, H. F. (1996).
  Choosing the forcing terms in an inexact Newton method.
  SIAM Journal on Scientific Computing, 17(1), 16-32.
"""
@kwdef struct EisenstatWalker <: Forcing
    η₀::Float64 = 0.999
    η_max::Float64 = 0.999
    γ::Float64 = 0.9
    α::Float64 = 2.0
end

"""
Compute the Eisenstat-Walker forcing term for n > 0
"""
function (F::EisenstatWalker)(η, tol, norm_res, norm_res_prior)
    (; γ, α, η_max) = F
    η_res = γ * (norm_res / norm_res_prior)^α
    # Eq 3.6 from Kelley2022
    if γ * η^α <= 1 // 10
        η_safe = min(η_max, η_res)
    else
        η_safe = min(η_max, max(η_res, γ * η^α))
    end
    return min(η_max, max(η_safe, 1 // 2 * tol / norm_res)) # Eq 3.5
end
initial(F::EisenstatWalker) = F.η₀

struct Stats{T <: Real}
    outer_iterations::Int
    inner_iterations::Int
    norm_res::T
end
function update(stats::Stats, inner_iterations, norm_res)
    return Stats(
        stats.outer_iterations + 1,
        stats.inner_iterations + inner_iterations,
        norm_res
    )
end

# Keyword arguments of `Krylov.krylov_workspace` (as of Krylov.jl v0.10). `krylov_kwargs`
# holds both these and the keyword arguments of `krylov_solve!`.
const KRYLOV_WORKSPACE_KEYS = (:memory, :window)
krylov_solve_kwargs(kwargs::NamedTuple) = Base.structdiff(kwargs, NamedTuple{KRYLOV_WORKSPACE_KEYS})
krylov_workspace_kwargs(kwargs::NamedTuple) = Base.structdiff(kwargs, krylov_solve_kwargs(kwargs))

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

    NewtonKrylovWorkspace(F!, u, p, res, alg=Val(:gmres); assume_p_const = false,
                          krylov_kwargs = (;))

- `F!`: in-place residual function `F!(res, u, p)`
- `u`: initial-guess array (used as template; the workspace holds a reference to it)
- `p`: parameters
- `res`: pre-allocated residual buffer
- `algo`: Krylov algorithm symbol (e.g. `:gmres`, `:fgmres`) passed as a `Val`.
- `assume_p_const`: passed through to [`JacobianOperator`](@ref)
- `krylov_kwargs`: the keyword arguments of the Krylov solver. The keys that configure the
  Krylov workspace (`memory` and `window`, e.g., `memory = 50` for the size of the Krylov
  basis of GMRES and FGMRES) are passed to `Krylov.krylov_workspace`, all other keys are
  ignored. The same `krylov_kwargs` can therefore also be passed to
  [`newton_krylov!(ws)`](@ref), which ignores the workspace keys.

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
        F!, u::AbstractArray, p, res::AbstractArray, ::Val{Algo} = Val(:gmres);
        assume_p_const::Bool = false, krylov_kwargs = (;)
    ) where {Algo}
    # res .= 0 might ignore ghost cells
    # memory allocated with similar might contain NaN/Inf
    Enzyme.make_zero!(res)
    neg_res = similar(res)
    Enzyme.make_zero!(neg_res)
    J = JacobianOperator(F!, res, u, p; assume_p_const)
    kc = KrylovConstructor(res)
    krylov = krylov_workspace(Val(Algo), kc; krylov_workspace_kwargs(krylov_kwargs)...)
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
  - `tol_rel`: Relative tolerance. Newton stops when `‖F(u)‖ <= tol_rel * ‖F(u₀)‖ + tol_abs`.
  - `tol_abs`: Absolute tolerance
  - `max_niter`: Maximum number of Newton iterations
  - `forcing`: Forcing-term strategy for the inexact Newton method, a [`Forcing`](@ref Ariadne.Forcing)
             such as [`EisenstatWalker()`](@ref Ariadne.EisenstatWalker) (default) or [`Fixed(η)`](@ref Ariadne.Fixed).
             It sets `atol = 0` and `rtol = η` for each Krylov solve; `krylov_kwargs` can override both.
             If `nothing`, Ariadne does not set `atol` or `rtol`, so the Krylov solver uses
             its own default tolerances, or those given in `krylov_kwargs`.
  - `linesearch!`: Line search strategy, an instance of a subtype of `AbstractLineSearch`.
  - `verbose::Int`: Verbosity level. If `verbose > 0`, progress is logged with `@info`.
  - `M::Union{Nothing, Function}`: If provided, `M(ws.J)` is passed as a keyword argument to the Krylov solver.
  - `N::Union{Nothing, Function}`: If provided, `N(ws.J)` is passed as a keyword argument to the Krylov solver.\
  - `krylov_kwargs`: Keyword arguments passed to the Krylov solver, e.g.,
    `(; restart = true, itmax = 100)`. The keys that configure the Krylov workspace
    (`memory` and `window`, e.g., `memory = 50` for the restart length of GMRES(k))
    only take effect when a [`NewtonKrylovWorkspace`](@ref) is created, i.e., in the
    methods that take a residual function, and are ignored by `newton_krylov!(ws)`.
  - `algo`, `assume_p_const`: Only for the methods that take a residual function;
    see [`NewtonKrylovWorkspace`](@ref).
  - `callback`: A function called once for the initial guess and then after each Newton iteration,
               with signature `callback(u, res, norm_res)`.
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
  - `M`: Length of the output of `F!`. Defaults to `length(u₀)`

$(KWARGS_DOCS)
"""
function newton_krylov!(
        F!, u₀::AbstractArray, p = nothing, M::Int = length(u₀);
        algo::Symbol = :gmres, assume_p_const::Bool = false,
        krylov_kwargs = (;), kwargs...
    )
    res = similar(u₀, M)
    ws = NewtonKrylovWorkspace(F!, u₀, p, res, Val(algo); assume_p_const, krylov_kwargs)
    return newton_krylov!(ws; krylov_kwargs, kwargs...)
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
        krylov_kwargs = (;),
        kwargs...,
    )
    ws = NewtonKrylovWorkspace(F!, u, p, res, Val(algo); assume_p_const, krylov_kwargs)
    return newton_krylov!(ws; krylov_kwargs, kwargs...)
end

"""
    newton_krylov!(ws::NewtonKrylovWorkspace, u; kwargs...)

Updates `ws.u` with the initial guess `u` and then calls `newton_krylov!(ws; kwargs...)`.

## Arguments
  - `ws`: Pre-allocated [`NewtonKrylovWorkspace`](@ref)
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
    else
        η = missing
    end

    verbose > 0 && @info "Jacobian-Free Newton-Krylov" res₀ = norm_res tol tol_rel tol_abs η

    stats = Stats(0, 0, norm_res)
    while norm_res > tol && stats.outer_iterations < max_niter
        # Handle kwargs for Preconditioners
        kwargs = krylov_solve_kwargs(krylov_kwargs)
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
            # The forcing terms are `Float64`, but Krylov.jl requires
            # tolerances of the real type of the state.
            rtol = convert(real(eltype(ws.u)), η)
            kwargs = (; atol = zero(rtol), rtol, kwargs...)
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
        verbose > 0 && @info "Newton" iter = stats.outer_iterations norm_res η stats
    end
    t = (time_ns() - t₀) / 1.0e9
    return ws.u, (; solved = norm_res <= tol, stats, t)
end

end # module Ariadne
