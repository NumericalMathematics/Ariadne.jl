##
# Mixed-precision Newton steps
#
# - C. T. Kelley (2022). Newton's method in mixed precision. SIAM Review 64(1), 191-211.
# - C. T. Kelley (2024). Newton's method in three precisions.
#   Pacific Journal of Optimization 20, 461-474. arXiv:2307.16051.
##

"""
    lu_in_precision!(A::AbstractMatrix{T}; threaded = true) -> LU{T}

LU factorization with partial pivoting of `A` in place, where every operation is
rounded to `T`. Uses LAPACK for BLAS floats. Otherwise the factorization performs
exactly the operations of `LinearAlgebra.generic_lufact!` in the same order per
entry, so the result does not depend on `threaded`; the trailing update is split
over columns with `Threads.@threads` if `threaded` is `true`.

Types whose arithmetic uses global state (for example the random number generator
of StochasticRounding.jl) must be factored with `threaded = false`.
"""
function lu_in_precision!(A::AbstractMatrix{T}; threaded::Bool = true) where {T}
    if T <: LinearAlgebra.BlasFloat
        return lu!(A; check = false)
    end
    m, n = size(A)
    minmn = min(m, n)
    info = 0
    ipiv = Vector{LinearAlgebra.BlasInt}(undef, minmn)
    @inbounds for k in 1:minmn
        kp = k
        if k < m
            amax = abs(A[k, k])
            for i in (k + 1):m
                absi = abs(A[i, k])
                if absi > amax
                    kp = i
                    amax = absi
                end
            end
        end
        ipiv[k] = kp
        if !iszero(A[kp, k])
            if k != kp
                for i in 1:n
                    A[k, i], A[kp, i] = A[kp, i], A[k, i]
                end
            end
            Akkinv = inv(A[k, k])
            for i in (k + 1):m
                A[i, k] *= Akkinv
            end
        elseif info == 0
            info = k
        end
        if threaded && n - k > 64
            Threads.@threads :static for j in (k + 1):n
                _lu_column_update!(A, k, j, m)
            end
        else
            for j in (k + 1):n
                _lu_column_update!(A, k, j, m)
            end
        end
    end
    return LU{T, typeof(A), typeof(ipiv)}(A, ipiv, LinearAlgebra.BlasInt(info))
end

@inline function _lu_column_update!(A, k, j, m)
    Akj = A[k, j]
    @inbounds @simd for i in (k + 1):m
        A[i, j] -= A[i, k] * Akj
    end
    return nothing
end

"""
    Ariadne.round_to(T, x)

Rounds `x` to the precision `T`. Defaults to `T(x)`; add methods for number types
without a direct conversion.
"""
round_to(::Type{T}, x) where {T} = T(x)

"""
    MixedPrecisionLU(A; factor_precision = eltype(A), solve_precision = factor_precision,
                     threaded = true)

The Jacobian `A` stored in the precision `eltype(A)` together with an LU
factorization of `A` rounded to `factor_precision`.

`ldiv!(y, P, x)` solves `A y ≈ x` with the triangular solves in `solve_precision`:

- If `eltype(x) == solve_precision`, the solve uses `x` directly ("on the fly"
  interprecision transfer in Kelley's terminology when the factors are of lower
  precision).
- Otherwise `x` is scaled by `‖x‖∞`, rounded to `solve_precision`, solved and
  scaled back (Kelley 2024, Eqs. (2.17)-(2.18)). Scaling avoids underflow in
  half precision.

If `solve_precision != factor_precision`, the factors are converted once to
`solve_precision` after the factorization (Kelley's "heavy" `MPHArray`), so the
triangular solves see the values of the low-precision factors but compute in
`solve_precision`. GMRES-IR needs this.

`A` is kept as the operator for iterative refinement (see [`IterativeRefinement`](@ref)
and [`GMRESIR`](@ref)).
"""
struct MixedPrecisionLU{TA, TF, TS, MA <: AbstractMatrix{TA}, LF, LS, V}
    A::MA
    factors::LF # LU in TF
    solver::LS # LU in TS
    buffer::V # Vector{TS}
end

function MixedPrecisionLU(
        A::AbstractMatrix{TA}; factor_precision::Type = TA,
        solve_precision::Type = factor_precision, threaded::Bool = true,
    ) where {TA}
    TF = factor_precision
    TS = solve_precision
    F = lu_in_precision!(Matrix{TF}(A); threaded)
    if TS === TF
        S = F
    else
        S = LU{TS, Matrix{TS}, typeof(F.ipiv)}(Matrix{TS}(F.factors), F.ipiv, F.info)
    end
    buffer = Vector{TS}(undef, size(A, 1))
    return MixedPrecisionLU{TA, TF, TS, typeof(A), typeof(F), typeof(S), typeof(buffer)}(A, F, S, buffer)
end

Base.size(P::MixedPrecisionLU, args...) = size(P.A, args...)
Base.eltype(::MixedPrecisionLU{TA}) where {TA} = TA
factor_precision(::MixedPrecisionLU{TA, TF}) where {TA, TF} = TF
solve_precision(::MixedPrecisionLU{TA, TF, TS}) where {TA, TF, TS} = TS
LinearAlgebra.issuccess(P::MixedPrecisionLU) = issuccess(P.factors)

function LinearAlgebra.ldiv!(y::AbstractVector, P::MixedPrecisionLU, x::AbstractVector)
    TS = solve_precision(P)
    if eltype(x) === TS
        y === x || copyto!(y, x)
        ldiv!(P.solver, y)
        return y
    end
    s = norm(x, Inf)
    if iszero(s) || !isfinite(s)
        y .= s .* x
        return y
    end
    b = P.buffer
    b .= round_to.(TS, x ./ s)
    ldiv!(P.solver, b)
    y .= s .* b
    return y
end
LinearAlgebra.ldiv!(P::MixedPrecisionLU, x::AbstractVector) = ldiv!(x, P, x)
Base.:\(P::MixedPrecisionLU, x::AbstractVector) = ldiv!(similar(x), P, x)
LinearAlgebra.mul!(y, P::MixedPrecisionLU, x) = mul!(y, P.A, x)

"""
    ConvertedPreconditioner{T}(P; scale = true)

Applies the preconditioner `P` in the precision `T`, e.g., a preconditioner whose
operators, smoothers and factors are all stored in `Float32` inside a solver in `Float64`.
`ldiv!(y, C, x)` (and `mul!`) rounds `x` to `T`, applies `P` in `T` and converts the result
back to the precision of `y`. With `scale = true`, `x` is scaled by `‖x‖∞` before rounding
and the result is scaled back (Kelley's interprecision transfer), which keeps small
residuals from underflowing in low precision.

Rounding makes the preconditioner slightly nonlinear, so use it with a flexible Krylov
method (e.g. `algo = :fgmres`) unless the rounding error is negligible.

If `P` is an [`AbstractPreconditioner`](@ref), e.g., a [`LaggedPreconditioner`](@ref) that
builds the low-precision operator, [`prepare!`](@ref) and [`record!`](@ref) are forwarded
to it.

## Example

```julia
# A multigrid hierarchy built in Float32, applied inside FGMRES in Float64
P = ConvertedPreconditioner{Float32}(LaggedPreconditioner(J -> build_multigrid_f32(J.u, J.p)))
newton_krylov!(ws; N = P, krylov_kwargs = (; ldiv = true))
```
"""
mutable struct ConvertedPreconditioner{T, OP} <: AbstractPreconditioner
    const P::OP
    const scale::Bool
    x::Any # buffers in precision T, allocated on first use with `similar`
    y::Any
end
ConvertedPreconditioner{T}(P; scale::Bool = true) where {T} =
    ConvertedPreconditioner{T, typeof(P)}(P, scale, nothing, nothing)

function converted_buffers(C::ConvertedPreconditioner{T}, x) where {T}
    if C.x === nothing || axes(C.x) != axes(x)
        C.x = similar(x, T)
        C.y = similar(x, T)
    end
    return C.x, C.y
end

function apply_converted!(op!, y, C::ConvertedPreconditioner{T}, x) where {T}
    xT, yT = converted_buffers(C, x)
    s = C.scale ? norm(x, Inf) : one(real(eltype(x)))
    if iszero(s) || !isfinite(s)
        # Nothing to scale: the result of P on a zero (or non-finite) vector
        y .= s .* x
        return y
    end
    xT .= round_to.(T, x ./ s)
    op!(yT, C.P, xT)
    y .= s .* yT
    return y
end

LinearAlgebra.ldiv!(y, C::ConvertedPreconditioner, x) = apply_converted!(ldiv!, y, C, x)
LinearAlgebra.mul!(y, C::ConvertedPreconditioner, x) = apply_converted!(mul!, y, C, x)
Base.:\(C::ConvertedPreconditioner, x::AbstractVector) = ldiv!(similar(x), C, x)

function prepare!(C::ConvertedPreconditioner, J)
    C.P isa AbstractPreconditioner && prepare!(C.P, J)
    return C
end
record!(C::ConvertedPreconditioner, stats) =
    C.P isa AbstractPreconditioner ? record!(C.P, stats) : nothing
refresh!(C::ConvertedPreconditioner) = (applicable(refresh!, C.P) && refresh!(C.P); C)

# The stored Jacobian of a preconditioner, used as operator of iterative refinement
stored_jacobian(P::MixedPrecisionLU) = P.A
stored_jacobian(P::LaggedPreconditioner) = stored_jacobian(P.operator)
stored_jacobian(C::ConvertedPreconditioner) = stored_jacobian(C.P)
stored_jacobian(P) = throw(ArgumentError("$(typeof(P)) does not store a Jacobian; use `operator = :jacobian`"))

##
# Workspaces for the Newton step other than Krylov methods.
# They behave like Krylov.jl workspaces (fields `x` and `stats`, `krylov_solve!`), so they
# can be the `krylov` workspace of a `NewtonKrylovWorkspace`. They use the preconditioner
# `N` (or `M`) passed to `krylov_solve!` as factorization, usually a `LaggedPreconditioner`
# that builds a `MixedPrecisionLU`.
##

"""
    AbstractStepWorkspace

Workspace for the Newton step `J d = -F(u)` that replaces the Krylov workspace of a
[`NewtonKrylovWorkspace`](@ref). Like a Krylov.jl workspace it has the fields `x` (the
solution) and `stats` (with `solved`, `niter`, `status`) and implements
`Krylov.krylov_solve!(ws, A, b; N, M, kwargs...)`. The factorization is the preconditioner
`N` (or `M`) of [`newton_krylov!`](@ref). The tolerances `atol` and `rtol` that
`newton_krylov!` derives from the forcing term are ignored; each workspace has its own
stopping rule.

Implemented: [`DirectSolveWorkspace`](@ref), [`IterativeRefinementWorkspace`](@ref),
[`GMRESIRWorkspace`](@ref).
"""
abstract type AbstractStepWorkspace end

"""
    StepStats

Statistics of an [`AbstractStepWorkspace`](@ref) solve: `solved`, `niter` (refinement
steps for IR, total GMRES iterations for GMRES-IR), `status`, and the `history` of the
refinement residual norms.
"""
struct StepStats
    solved::Bool
    niter::Int
    status::String
    history::Vector{Float64}
end
StepStats() = StepStats(false, 0, "unknown", Float64[])

"""
    DirectSolveWorkspace(b)

Newton step `d = P⁻¹(-F(u))` with the (possibly low-precision) factorization `P`.
With a [`MixedPrecisionLU`](@ref) this is Newton's method in two precisions
(Kelley 2022). `b` is a template vector of the residual type.

    ws = NewtonKrylovWorkspace(F!, u, p, res, DirectSolveWorkspace(res))
    newton_krylov!(ws; N = LaggedPreconditioner(J -> MixedPrecisionLU(A32(J))), forcing = nothing)
"""
mutable struct DirectSolveWorkspace{V} <: AbstractStepWorkspace
    const x::V
    stats::StepStats
end
DirectSolveWorkspace(b::AbstractVector) = DirectSolveWorkspace(zero(b), StepStats())

"""
    IterativeRefinementWorkspace(b; operator = :stored, rtol = nothing, maxiter = 50,
                                 decrease = 0.9, p = Inf)

Solve for the Newton step with iterative refinement preconditioned by the
factorization `P` (Kelley 2024, Algorithm 2.1 IR):

    r = b - A x;  x ← x + P⁻¹ r

- `operator = :stored`: `A` is the Jacobian stored in `P` (see [`MixedPrecisionLU`](@ref))
  and the iteration runs in `eltype(A)`, e.g., single precision.
- `operator = :jacobian`: `A` is the operator passed to `krylov_solve!`, i.e., the
  matrix-free [`JacobianOperator`](@ref) (Enzyme JVPs), and the iteration runs in the
  precision of `b`.

Stops when `‖r‖ₚ ≤ rtol ‖b‖ₚ` (default `rtol = 10 eps(T)`, `p = Inf` as in Kelley's
`mpgeslir`), when `‖r‖ₚ` decreases by less than the factor `decrease`, or after `maxiter`
iterations. As in Kelley's code, the last iterate is returned.
"""
mutable struct IterativeRefinementWorkspace{V} <: AbstractStepWorkspace
    const x::V
    stats::StepStats
    const operator::Symbol
    const rtol::Union{Nothing, Float64}
    const maxiter::Int
    const decrease::Float64
    const p::Float64
end
function IterativeRefinementWorkspace(
        b::AbstractVector; operator::Symbol = :stored, rtol = nothing,
        maxiter::Integer = 50, decrease::Real = 0.9, p::Real = Inf
    )
    check_operator(operator)
    return IterativeRefinementWorkspace(zero(b), StepStats(), operator, rtol, Int(maxiter), Float64(decrease), Float64(p))
end

"""
    GMRESIRWorkspace(b; operator = :stored, gmres_precision = nothing, rtol = nothing,
                     memory = 10, maxiter = 50, decrease = 0.99, p = 2)

GMRES-IR (Carson & Higham 2017, 2018; Kelley 2024 §2.2): iterative refinement where
the correction solves `P⁻¹ A d = P⁻¹ r` with left-preconditioned GMRES (no restarts,
at most `memory` iterations, relative tolerance `rtol`, default `10 eps(T)`).
See [`IterativeRefinementWorkspace`](@ref) for `operator` and the stopping rule
(`p = 2` as in Kelley's `mpgmir`).

`gmres_precision` is the precision of the inner GMRES iteration: its Krylov basis, its
arithmetic, and the vectors passed to `A` and `P` (default: the working precision `T`).
The products with `A` are computed in `T` and rounded to `gmres_precision`. With the
precisions of the factorization and of the application of `P` (`factor_precision` and
`solve_precision` of a [`MixedPrecisionLU`](@ref)), of the working precision and of the
residual (`operator`), these are the five precisions of Amestoy, Buttari, Higham,
L'Excellent, Mary and Vieublé, *Five-precision GMRES-based iterative refinement*,
SIAM J. Matrix Anal. Appl. (2024). A lower `gmres_precision` halves the memory of the
Krylov basis for `Float32`.

For the best accuracy, `P` solves in `gmres_precision` (the "on the fly" interprecision
transfer, `solve_precision` of a `MixedPrecisionLU` equal to `gmres_precision`).
"""
mutable struct GMRESIRWorkspace{V} <: AbstractStepWorkspace
    const x::V
    stats::StepStats
    const operator::Symbol
    const gmres_precision::Union{Nothing, Type}
    const rtol::Union{Nothing, Float64}
    const memory::Int
    const maxiter::Int
    const decrease::Float64
    const p::Float64
    gmres::Any # GMRES workspace of the inner iteration, kept between solves
end
function GMRESIRWorkspace(
        b::AbstractVector; operator::Symbol = :stored, gmres_precision = nothing,
        rtol = nothing, memory::Integer = 10, maxiter::Integer = 50, decrease::Real = 0.99,
        p::Real = 2
    )
    check_operator(operator)
    return GMRESIRWorkspace(
        zero(b), StepStats(), operator, gmres_precision, rtol, Int(memory), Int(maxiter),
        Float64(decrease), Float64(p), nothing
    )
end

# The operator `A` applied to vectors of precision `TG`: the product is computed in the
# precision `T` of `A` and rounded to `TG`.
struct PrecisionConvertedOperator{TG, OP, V}
    A::OP
    x::V
    y::V
end
PrecisionConvertedOperator{TG}(A, b) where {TG} =
    PrecisionConvertedOperator{TG, typeof(A), typeof(b)}(A, similar(b), similar(b))
Base.size(op::PrecisionConvertedOperator, args...) = size(op.A, args...)
Base.eltype(::PrecisionConvertedOperator{TG}) where {TG} = TG
function LinearAlgebra.mul!(y, op::PrecisionConvertedOperator{TG}, x) where {TG}
    op.x .= x
    mul!(op.y, op.A, op.x)
    y .= round_to.(TG, op.y)
    return y
end

function inner_gmres_workspace(ws::GMRESIRWorkspace, bG)
    G = ws.gmres
    if G === nothing || length(G.x) != length(bG) || eltype(G.x) != eltype(bG)
        G = ws.gmres = GmresWorkspace(KrylovConstructor(bG); memory = ws.memory)
    end
    return G
end

check_operator(op) = op in (:stored, :jacobian) || throw(ArgumentError("operator must be :stored or :jacobian"))

function step_factorization(M, N)
    P = N === nothing ? M : N
    P === nothing && throw(ArgumentError("$(@__MODULE__) step workspaces need a factorization as preconditioner `N`"))
    return P
end

function Krylov.krylov_solve!(ws::DirectSolveWorkspace, A, b; M = nothing, N = nothing, kwargs...)
    ldiv!(ws.x, step_factorization(M, N), b)
    ws.stats = StepStats(all(isfinite, ws.x), 1, "direct", Float64[])
    return ws
end

function refinement_operator(ws, A, b, P)
    if ws.operator === :stored
        As = stored_jacobian(P)
        return As, eltype(As)
    else
        return A, eltype(b)
    end
end

function Krylov.krylov_solve!(
        ws::Union{IterativeRefinementWorkspace, GMRESIRWorkspace}, A, rhs;
        M = nothing, N = nothing, kwargs...
    )
    P = step_factorization(M, N)
    Aop, T = refinement_operator(ws, A, rhs, P)
    rtol = ws.rtol === nothing ? 10 * eps(T) : T(ws.rtol)
    d = ws.x
    # Kelley scales the right-hand side by its norm before rounding to the working precision
    s = norm(rhs, Inf)
    if iszero(s)
        fill!(d, 0)
        ws.stats = StepStats(true, 0, "zero rhs", Float64[])
        return ws
    end
    b = round_to.(T, rhs ./ s)
    x = zero(b)
    r = copy(b)
    c = similar(b)
    tol = rtol * norm(b, ws.p)
    rnrm = norm(r, ws.p)
    rprev = 2 * rnrm
    history = Float64[Float64(rnrm)]
    niter = 0
    k = 0
    if ws isa GMRESIRWorkspace
        TG = ws.gmres_precision === nothing ? T : ws.gmres_precision
        rG = similar(b, TG)
        gws = inner_gmres_workspace(ws, rG)
        AG = TG === T ? Aop : PrecisionConvertedOperator{TG}(Aop, b)
        # GMRES in TG cannot reduce the residual below its roundoff
        rtolG = TG(max(rtol, 10 * eps(TG)))
    end
    while rnrm > tol && rnrm <= ws.decrease * rprev && k < ws.maxiter
        if ws isa IterativeRefinementWorkspace
            ldiv!(c, P, r)
            niter += 1
        else
            rG .= round_to.(TG, r ./ rnrm)
            krylov_solve!(
                gws, AG, rG; M = P, ldiv = true, restart = false,
                itmax = ws.memory, atol = zero(rtolG), rtol = rtolG
            )
            c .= rnrm .* gws.x
            niter += gws.stats.niter
        end
        x .+= c
        mul!(r, Aop, x)
        r .= b .- r
        rprev = rnrm
        rnrm = norm(r, ws.p)
        push!(history, Float64(rnrm))
        k += 1
    end
    d .= s .* x
    solved = rnrm <= tol
    status = solved ? "converged" : (k >= ws.maxiter ? "maxiter" : "stagnated")
    ws.stats = StepStats(solved, niter, status, history)
    return ws
end

# Workspace for the Newton step from the `algo` symbol of `newton_krylov!`
newton_step_workspace(::Val{Algo}, res, krylov_kwargs) where {Algo} =
    krylov_workspace(Val(Algo), KrylovConstructor(res); krylov_workspace_kwargs(krylov_kwargs)...)
newton_step_workspace(::Val{:direct}, res, krylov_kwargs) = DirectSolveWorkspace(res)
newton_step_workspace(::Val{:ir}, res, krylov_kwargs) = IterativeRefinementWorkspace(res)
newton_step_workspace(::Val{:gmresir}, res, krylov_kwargs) = GMRESIRWorkspace(res)
