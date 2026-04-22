import DifferentiationInterface as DI

"""
    DIJacobianOperator
"""
struct DIJacobianOperator{F, A, P} <: AbstractJacobianOperator
    f::F # F!(res, u, p)
    res::A
    u::A
    p::P
    prep
    backend
end

"""
    DIJacobianOperator(f::F, res, u, p)

Creates a Jacobian operator for `f!(res, u, p)` where `res` is the residual,
`u` is the state variable, and `p` are the parameters.
"""
function DIJacobianOperator(backend, f::F, res, u, p) where {F}
    tu = zero(u) # dummy tangent
    prep = DI.prepare_pushforward(f, res, backend, u, (tu,), DI.ConstantOrCache(p))

    return DIJacobianOperator(f, res, u, p, prep, backend)
end

Base.size(J::DIJacobianOperator) = (length(J.res), length(J.u))
Base.eltype(J::DIJacobianOperator) = eltype(J.u)
Base.length(J::DIJacobianOperator) = prod(size(J))

function mul!(out, J::DIJacobianOperator, v)
    DI.pushforward!(
        J.f,
        J.res,
        (out,),
        J.prep,
        J.backend,
        J.u,
        (v,), # TODO: Must we zero this?
        DI.ConstantOrCache(J.p)
    )
    return nothing
end
