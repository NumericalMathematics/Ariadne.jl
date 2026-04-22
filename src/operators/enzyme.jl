using Enzyme

function init_cache(x)
    if !Enzyme.Compiler.guaranteed_const(typeof(x))
        create_shadow(x)
    else
        return nothing
    end
end

function maybe_duplicated(x::T, x′::Union{Nothing, T}) where {T}
    if x′ === nothing
        return Const(x)
    else
        zero_shadow!(x′)
        return Duplicated(x, x′)
    end
end

"""
    EnzymeJacobianOperator

Efficient implementation of `J(f,x,p) * v` and `v * J(f, x,p)'`
"""
struct EnzymeJacobianOperator{F, F′, A, P, P′} <: AbstractJacobianOperator
    f::F # F!(res, u, p)
    f′::F′ # cache
    res::A
    u::A
    p::P
    p′::P′ # cache
end

"""
    EnzymeJacobianOperator(f::F, res, u, p; assume_p_const::Bool = false)

Creates a Jacobian operator for `f!(res, u, p)` where `res` is the residual,
`u` is the state variable, and `p` are the parameters.

If `assume_p_const` is `true`, the parameters `p` are assumed to be constant
during the Jacobian computation, which can improve performance by not requiring the
shadow for `p`.
"""
function EnzymeJacobianOperator(f::F, res, u, p; assume_p_const::Bool = false) where {F}
    f′ = init_cache(f)
    if assume_p_const
        p′ = nothing
    else
        p′ = init_cache(p)
    end
    return EnzymeJacobianOperator(f, f′, res, u, p, p′)
end

Base.size(J::EnzymeJacobianOperator) = (length(J.res), length(J.u))
Base.eltype(J::EnzymeJacobianOperator) = eltype(J.u)
Base.length(J::EnzymeJacobianOperator) = prod(size(J))

function mul!(out, J::EnzymeJacobianOperator, v)
    autodiff(
        Forward,
        maybe_duplicated(J.f, J.f′), Const,
        Duplicated(J.res, reshape(out, size(J.res))),
        Duplicated(J.u, reshape(v, size(J.u))),
        maybe_duplicated(J.p, J.p′)
    )
    return nothing
end

LinearAlgebra.adjoint(J::EnzymeJacobianOperator) = Adjoint(J)
LinearAlgebra.transpose(J::EnzymeJacobianOperator) = Transpose(J)

# Jᵀ(y, u) = ForwardDiff.gradient!(y, x -> dot(F(x), u), xk)
# or just reverse mode

function mul!(out, J′::Union{Adjoint{<:Any, <:EnzymeJacobianOperator}, Transpose{<:Any, <:EnzymeJacobianOperator}}, v)
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
