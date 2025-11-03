struct DIRKButcher{T1 <: AbstractArray, T2 <: AbstractArray} <: RKTableau
    a::T1
    b::T2
    c::T2
end


"""
    Crouzeix32()

A third-order A and L-stable DIRK method developed by Nørsett (1974) and Crouzeix (1975).

## References
- Ernst Hairer, Syvert P. Nørsett, and Gerhard Wanner (1993) 
  *Solving Ordinary Differential Equations I: Nonstiff Problems.* 
  *Springer Series in Computational Mathematics,* 2nd edition. 
  [DOI: 10.1007/978-3-540-78862-1](https://doi.org/10.1007/978-3-540-78862-1)
"""
struct Crouzeix32 <: DIRK{2} end
function RKTableau(alg::Crouzeix32, RealT)
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
struct LobattoIIIA <: DIRK{2} end
function RKTableau(alg::LobattoIIIA, RealT)
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

"""
    DIRK43()

A fourth order and A-stable DIRK method.

## References
- D. Fränken and Karlheinz Ochs (2003) 
  *Passive Runge–Kutta Methods—Properties, Parametric Representation, and Order Conditions.* 
  *BIT Numerical Mathematics* 43(2):339–361. 
  [DOI: 10.1023/A:1026039820006](https://doi.org/10.1023/A:1026039820006)
"""
struct DIRK43 <: DIRK{3} end
function RKTableau(alg::DIRK43, RealT)
    nstage = 3
    a = zeros(RealT, nstage, nstage)
    a[1, 1] = 1 // 2 + 1 / (2 * sqrt(convert(RealT, 2)))
    a[2, 1] = -1 - sqrt(convert(RealT, 2))
    a[2, 2] = 3 // 2 + sqrt(convert(RealT, 2))
    a[3, 1] = 1 + 1 / sqrt(convert(RealT, 2))
    a[3, 2] = -1 - sqrt(convert(RealT, 2))
    a[3, 3] = 1 // 2 + 1 / (2 * sqrt(convert(RealT, 2)))

    b = zeros(RealT, nstage)
    b[1] = 1 // 3
    b[2] = 1 // 3
    b[3] = 1 // 3

    c = zeros(Float64, nstage)
    c[1] = a[1, 1]
    c[2] = 1 // 2
    c[3] = 1 // 2 - 1 / (2 * sqrt(convert(RealT, 2)))
    return DIRKButcher(a, b, c)
end
