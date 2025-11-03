struct DIRKButcher{T1 <: AbstractArray, T2 <: AbstractArray} <: RKTableau
    a::T1
    b::T2
    c::T2
end


"""
    Crouzeix23()

A third-order A and L-stable DIRK method developed by Nørsett (1974) and Crouzeix (1975).

## References
- Ernst Hairer, Syvert P. Nørsett, and Gerhard Wanner (1993) 
  *Solving Ordinary Differential Equations I: Nonstiff Problems.* 
  *Springer Series in Computational Mathematics,* 2nd edition. 
  [DOI: 10.1007/978-3-540-78862-1](https://doi.org/10.1007/978-3-540-78862-1)
"""
struct Crouzeix23 <: DIRK{2} end
function RKTableau(alg::Crouzeix23, RealT)
    nstage = 2
    sqrt3 = sqrt(convert(RealT, 3))
    a = zeros(RealT, nstage, nstage)
    a[1, 1] = 1 // 2 + sqrt3 / 6
    a[2, 1] = -sqrt3 / 3
    a[2, 2] = 1 // 2 + sqrt3 / 6
    b = zeros(RealT, nstage)
    b[1] = 1 // 2
    b[2] = 1 // 2

    c = zeros(Float64, nstage)
    c[1] = 1 // 2 + sqrt(3) / 6
    c[2] = 1 // 2 - sqrt(3) / 6
    return DIRKButcher(a, b, c)
end


"""
    LobattoIIIA()

A second order and A-stable DIRK method.

## References
- Ernst Hairer, Syvert P. Nørsett, and Gerhard Wanner (1993) 
  *Solving Ordinary Differential Equations I: Nonstiff Problems.* 
  *Springer Series in Computational Mathematics,* 2nd edition. 
  [DOI: 10.1007/978-3-540-78862-1](https://doi.org/10.1007/978-3-540-78862-1)
"""
struct LobattoIIIA <: DIRK{1} end
function RKTableau(alg::Crouzeix23, RealT)
    nstage = 2
    a = zeros(RealT, nstage, nstage)
    a[2, 1] = 1 // 2
    a[2, 2] = 1 // 2
    b = zeros(RealT, nstage)
    b[1] = 1 // 2
    b[2] = 1 // 2

    c = zeros(Float64, nstage)
    c[2] = 1
    return DIRKButcher(a, b, c)
end
