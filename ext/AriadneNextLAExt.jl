module AriadneNextLAExt

# Triangular solves of `MixedPrecisionLU` with the recursive TRSM of NextLA.jl for matrices
# that are not `Array`s, e.g. GPU arrays, for any element type.

using Ariadne
using LinearAlgebra: diagind
using NextLA: unified_rectrxm!

Ariadne.default_triangular_solves(::AbstractMatrix) = :nextla

# `unified_rectrxm!` of NextLA 0.2 reads the diagonal, so keep a copy of the unit lower factor
function Ariadne.solver_data(::Val{:nextla}, F)
    L = copy(F.factors)
    view(L, diagind(L)) .= one(eltype(L))
    return L
end

function Ariadne.triangular_solves!(::Val{:nextla}, S, x)
    T = eltype(x)
    B = reshape(x, :, 1)
    unified_rectrxm!('L', 'L', 'N', one(T), 'S', S.L, B)
    unified_rectrxm!('L', 'U', 'N', one(T), 'S', S.F.factors, B)
    return x
end

end
