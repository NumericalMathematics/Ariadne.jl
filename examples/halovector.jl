using OffsetArrays

struct HaloVector{FC, D} <: AbstractVector{FC}
    data::D

    function HaloVector(data::D) where {D}
        FC = eltype(data)
        return new{FC, D}(data)
    end
end

function Base.similar(v::HaloVector)
    data = similar(v.data)
    return HaloVector(data)
end

function Base.length(v::HaloVector)
    m, n = size(v.data)
    l = (m - 2) * (n - 2)
    return l
end

function Base.size(v::HaloVector)
    l = length(v)
    return (l,)
end

function Base.getindex(v::HaloVector, idx)
    m, n = size(v.data)
    row = div(idx - 1, n - 2) + 1
    col = mod(idx - 1, n - 2) + 1
    return v.data[row, col]
end

function Base.setindex!(v::HaloVector, val, idx)
    m, n = size(v.data)
    row = div(idx - 1, n - 2) + 1
    col = mod(idx - 1, n - 2) + 1
    return v.data[row, col] = val
end

Base.zero(v::HaloVector) = HaloVector(zero(v.data))

# Otherwise we get Uninitialized data in the halo...
Base.copy(v::HaloVector) = HaloVector(copy(v.data))


using Krylov
import Krylov.FloatOrComplex

function Krylov.kdot(n::Integer, x::HaloVector{T}, y::HaloVector{T}) where {T <: FloatOrComplex}
    mx, nx = size(x.data)
    _x = x.data
    _y = y.data
    res = zero(T)
    for i in 1:(mx - 1)
        for j in 1:(nx - 1)
            res += _x[i, j] * _y[i, j]
        end
    end
    return res
end

function Krylov.knorm(n::Integer, x::HaloVector{T}) where {T <: FloatOrComplex}
    mx, nx = size(x.data)
    _x = x.data
    res = zero(T)
    for i in 1:(mx - 1)
        for j in 1:(nx - 1)
            res += _x[i, j]^2
        end
    end
    return sqrt(res)
end

function Krylov.kscal!(n::Integer, s::T, x::HaloVector{T}) where {T <: FloatOrComplex}
    mx, nx = size(x.data)
    _x = x.data
    for i in 1:(mx - 1)
        for j in 1:(nx - 1)
            _x[i, j] = s * _x[i, j]
        end
    end
    return x
end

function Krylov.kaxpy!(n::Integer, s::T, x::HaloVector{T}, y::HaloVector{T}) where {T <: FloatOrComplex}
    mx, nx = size(x.data)
    _x = x.data
    _y = y.data
    for i in 1:(mx - 1)
        for j in 1:(nx - 1)
            _y[i, j] += s * _x[i, j]
        end
    end
    return y
end

function Krylov.kaxpby!(n::Integer, s::T, x::HaloVector{T}, t::T, y::HaloVector{T}) where {T <: FloatOrComplex}
    mx, nx = size(x.data)
    _x = x.data
    _y = y.data
    for i in 1:(mx - 1)
        for j in 1:(nx - 1)
            _y[i, j] = s * _x[i, j] + t * _y[i, j]
        end
    end
    return y
end

function Krylov.kcopy!(n::Integer, y::HaloVector{T}, x::HaloVector{T}) where {T <: FloatOrComplex}
    mx, nx = size(x.data)
    _x = x.data
    _y = y.data
    for i in 1:(mx - 1)
        for j in 1:(nx - 1)
            _y[i, j] = _x[i, j]
        end
    end
    return y
end

function Krylov.kfill!(x::HaloVector{T}, val::T) where {T <: FloatOrComplex}
    mx, nx = size(x.data)
    _x = x.data
    for i in 1:(mx - 1)
        for j in 1:(nx - 1)
            _x[i, j] = val
        end
    end
    return x
end

function Krylov.kref!(n::Integer, x::HaloVector{T}, y::HaloVector{T}, c::T, s::T) where {T <: FloatOrComplex}
    mx, nx = size(x.data)
    _x = x.data
    _y = y.data
    for i in 1:(mx - 1)
        for j in 1:(nx - 1)
            x_ij = _x[i, j]
            y_ij = _y[i, j]
            _x[i, j] = c * x_ij + s * y_ij
            _y[i, j] = conj(s) * x_ij - c * y_ij
        end
    end
    return x, y
end
