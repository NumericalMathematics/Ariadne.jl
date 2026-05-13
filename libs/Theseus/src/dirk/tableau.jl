struct DIRKButcher{T1 <: AbstractArray, T2 <: AbstractArray} <: RKTableau
    a::T1
    b::T2
    c::T2
end


"""
    Crouzeix32()

A third-order, two-stage, A- and L-stable diagonally implicit Runge-Kutta (DIRK) method
developed by Nørsett (1974) and Crouzeix (1975). The nodes and weights are the ones of
the two-point Gauss–Legendre quadrature. Thus, this method has order four when applied
to quadrature problems.

## References
See Table 7.2 on p. 207 for the Butcher tableau.
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

    c = zeros(RealT, nstage)
    c[1] = 1 // 2 + sqrt(3) / 6
    c[2] = 1 // 2 - sqrt(3) / 6
    return DIRKButcher(a, b, c)
end


"""
    LobattoIIIA2()

A second-order, two-stage, A-stable DIRK method from the general
class of Lobatto IIIA methods.

## References
See Table (213) on p. 69 for the Butcher tableau.
- Christopher A. Kennedy and Mark H. Carpenter (2016)
  *Diagonally Implicit Runge–Kutta Methods for Ordinary Differential Equations: A Review.*
  *NASA Technical Memorandum NASA/TM-2016-219173, Langley Research Center, Hampton, VA, United States.*
"""
struct LobattoIIIA2 <: DIRK{2} end
function RKTableau(alg::LobattoIIIA2, RealT)
    nstage = 2
    a = zeros(RealT, nstage, nstage)
    a[2, 1] = 1 // 2
    a[2, 2] = 1 // 2
    b = zeros(RealT, nstage)
    b[1] = 1 // 2
    b[2] = 1 // 2

    c = zeros(RealT, nstage)
    c[2] = 1
    return DIRKButcher(a, b, c)
end

"""
    DIRK43()

A fourth-order, three-stage, A-stable DIRK method.

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

    c = zeros(RealT, nstage)
    c[1] = a[1, 1]
    c[2] = 1 // 2
    c[3] = 1 // 2 - 1 / (2 * sqrt(convert(RealT, 2)))
    return DIRKButcher(a, b, c)
end

"""
    Theseus.ESDIRK43SA2()

A fourth-order, six-stage, stiffly accurate ESDIRK method
with an embedded third-order method for error estimation.

## References
See Table 9 on p. 242 for the Butcher tableau.
- Christopher A. Kennedy and Mark H. Carpenter (2019)
  *Diagonally Implicit Runge–Kutta Methods for stiff ODEs*
  *Applied Numerical Mathematics* 146:221–244.
  [DOI: 10.1016/j.apnum.2019.07.008] (https://doi.org/10.1016/j.apnum.2019.07.008)
"""
struct ESDIRK43SA2 <: DIRK{6} end
function RKTableau(alg::ESDIRK43SA2, RealT)
    nstage = 6
    a = zeros(RealT, nstage, nstage)
    a[2, 1] = 31 // 125
    a[2, 2] = a[2, 1]
    a[3, 1] = -360286518617 // 7014585480527
    a[3, 2] = a[3, 1]
    a[3, 3] = a[2, 2]
    a[4, 1] = -506388693497 // 5937754990171
    a[4, 2] = a[4, 1]
    a[4, 3] = 7149918333491 // 13390931526268
    a[4, 4] = a[2, 2]
    a[5, 1] = -7628305438933 // 11061539393788
    a[5, 2] = a[5, 1]
    a[5, 3] = 21592626537567 // 14352247503901
    a[5, 4] = 11630056083252 // 17263101053231
    a[5, 5] = a[2, 2]
    a[6, 1] = -12917657251 // 5222094901039
    a[6, 2] = a[6, 1]
    a[6, 3] = 5602338284630 // 15643096342197
    a[6, 4] = 9002339615474 // 18125249312447
    a[6, 5] = -2420307481369 // 24731958684496
    a[6, 6] = a[2, 2]

    b = a[6, :]
    #weights for the embedded method:
    #b_e = zeros(RealT, nstage)
    #b_e[1] = -1007911106287 // 12117826057527
    #b_e[2] = b_e[1]
    #b_e[3] = 17694008993113 // 35931961998873
    #b_e[4] = 5816803040497 // 11256217655929
    #b_e[5] = -538664890905 // 7490061179786
    #b_e[6] = 2032560730450 // 8872919773257

    c = zeros(RealT, nstage)
    c[2] = 62 // 125
    c[3] = 486119545908 // 3346201505189
    c[4] = 1043 // 1706
    c[5] = 1361 // 1300
    c[6] = 1
    return DIRKButcher(a, b, c)
end

"""
    Theseus.CooperSayfy5()

A fifth-order, five-stage, A-stable DIRK method.

## References
- E. Hairer, G. Wanner. Solving ordinary differential equations II: Stiff and Differential-Algebraic Problems.
  Springer, 1996.
  page.101
- Cooper, G. J., and A. Sayfy. Semiexplicit Runge-Kutta methods for stiff differential equations.
  Mathematics of Computation 33,
  no. 146 (1979): 541-556.
  doi:10.1090/S0025-5718-1979-0521275-1.
"""
struct CooperSayfy5 <: DIRK{5} end
function RKTableau(alg::CooperSayfy5, RealT)
    nstage = 5
    sqrt6 = sqrt(convert(RealT, 6))

    γ = (6 - sqrt6) / 10

    a = zeros(RealT, nstage, nstage)

    a[1, 1] = γ

    a[2, 1] = (6 + 5 * sqrt6) / 14
    a[2, 2] = γ

    a[3, 1] = (888 + 607 * sqrt6) / 2850
    a[3, 2] = (126 - 161 * sqrt6) / 1425
    a[3, 3] = γ

    a[4, 1] = (3153 - 3082 * sqrt6) / 14250
    a[4, 2] = (3213 + 1148 * sqrt6) / 28500
    a[4, 3] = (-267 + 88 * sqrt6) / 500
    a[4, 4] = γ

    a[5, 1] = (-32583 + 14638 * sqrt6) / 71250
    a[5, 2] = (-17199 + 364 * sqrt6) / 142500
    a[5, 3] = (1329 - 544 * sqrt6) / 2500
    a[5, 4] = (-96 + 131 * sqrt6) / 625
    a[5, 5] = γ

    b = zeros(RealT, nstage)
    b[1] = 0
    b[2] = 0
    b[3] = 1 // 9
    b[4] = (16 - sqrt6) / 36
    b[5] = (16 + sqrt6) / 36

    c = zeros(RealT, nstage)
    c[1] = γ
    c[2] = (6 + 9 * sqrt6) / 35
    c[3] = 1
    c[4] = (4 - sqrt6) / 10
    c[5] = (4 + sqrt6) / 10

    return DIRKButcher(a, b, c)
end


"""
    Theseus.HairerWannerSDIRK4()

A fourth-order, five-stage, L-stable SDIRK method.

## References
- E. Hairer, G. Wanner. Solving ordinary differential equations II: Stiff and Differential-Algebraic Problems.
  Springer, 1996.
  page. 100
"""
struct HairerWannerSDIRK4 <: DIRK{5} end
function RKTableau(alg::HairerWannerSDIRK4, RealT)
    nstage = 5
    a = zeros(RealT, nstage, nstage)

    γ = 1 // 4

    a[1, 1] = γ

    a[2, 1] = 1 // 2
    a[2, 2] = γ

    a[3, 1] = 17 // 50
    a[3, 2] = -1 // 25
    a[3, 3] = γ

    a[4, 1] = 371 // 1360
    a[4, 2] = -137 // 2720
    a[4, 3] = 15 // 544
    a[4, 4] = γ

    a[5, 1] = 25 // 24
    a[5, 2] = -49 // 48
    a[5, 3] = 125 // 16
    a[5, 4] = -85 // 12
    a[5, 5] = γ

    b = zeros(RealT, nstage)
    b[1] = 25 // 24
    b[2] = -49 // 48
    b[3] = 125 // 16
    b[4] = -85 // 12
    b[5] = 1 // 4

    c = zeros(RealT, nstage)
    c[1] = 1 // 4
    c[2] = 3 // 4
    c[3] = 11 // 20
    c[4] = 1 // 2
    c[5] = 1

    return DIRKButcher(a, b, c)
end


"""
    Theseus.CrouzeixRaviart34()

A fourth-order, three-stage, L-stable SDIRK method.

## References
- E. Hairer, G. Wanner. Solving ordinary differential equations II: Stiff and Differential-Algebraic Problems.
  Springer, 1996.
  page.100
- M. Crouzeix. Sur l’approximation des équations différentielles opérationnelles linéaires par des méthodes de Runge-Kutta.
  Thèse d'état, Univ. Paris 6 192pp, 1975.
"""
struct CrouzeixRaviart34 <: DIRK{3} end
function RKTableau(alg::CrouzeixRaviart34, RealT)
    nstage = 3

    sqrt3 = sqrt(convert(RealT, 3))
    γ = (1 / sqrt3) * cospi(one(RealT) / 18) + 1 // 2
    δ = 1 / (6 * (2 * γ - 1)^2)

    a = zeros(RealT, nstage, nstage)

    a[1, 1] = γ

    a[2, 1] = 1 // 2 - γ
    a[2, 2] = γ

    a[3, 1] = 2 * γ
    a[3, 2] = 1 - 4 * γ
    a[3, 3] = γ

    b = zeros(RealT, nstage)
    b[1] = δ
    b[2] = 1 - 2 * δ
    b[3] = δ

    c = zeros(RealT, nstage)
    c[1] = γ
    c[2] = 1 // 2
    c[3] = 1 - γ

    return DIRKButcher(a, b, c)
end
