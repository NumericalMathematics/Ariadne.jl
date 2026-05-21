struct IMEXButcher{T1 <: AbstractArray, T2 <: AbstractArray} <: RKTableau
    a_ex::T1
    b_ex::T2
    c_ex::T2
    a_im::T1
    b_im::T2
    c_im::T2
end

"""
    Theseus.SP111()

The symplectic Euler method, a first-order, one-stage type I IMEX method
combining an explicit and an implicit Euler method.

## References
- Sebastiano Boscarino and Giovanni Russo (2024)
  *Asymptotic preserving methods for quasilinear hyperbolic systems with stiff relaxation: a review.*
  [DOI: 10.1007/s40324-024-00351-x](https://doi.org/10.1007/s40324-024-00351-x)
"""
struct SP111 <: RKIMEX{1} end
function RKTableau(alg::SP111, RealT)
    nstage = 1
    a = zeros(RealT, nstage, nstage)
    b = zeros(RealT, nstage)
    b[1] = 1
    c = zeros(RealT, nstage)
    a_im = zeros(RealT, nstage, nstage)
    a_im[1, 1] = 1
    b_im = zeros(RealT, nstage)
    b_im[1] = 1
    c_im = zeros(RealT, nstage)
    c_im[1] = 1
    return IMEXButcher(a, b, c, a_im, b_im, c_im)
end

"""
    Theseus.H222()

A second-order, two-stage type I IMEX method.
The explicit part is strong stability preserving (SSP), the implicit part
is A-stable but not L-stable.

## References
- Sebastiano  Boscarino, Lorenzo Pareschi, and Giovanni Russo (2025)
  Implicit-explicit methods for evolutionary partial differential equations.
  [DOI: 10.1137/1.9781611978209](https://doi.org/10.1137/1.9781611978209)
"""
struct H222 <: RKIMEX{2} end
function RKTableau(alg::H222, RealT)
    nstage = 2
    a = zeros(RealT, nstage, nstage)
    a[2, 1] = 1
    b = zeros(RealT, nstage)
    b[1] = 1 // 2
    b[2] = 1 // 2
    c = zeros(RealT, nstage)
    c[2] = 1
    a_im = zeros(RealT, nstage, nstage)
    a_im[1, 1] = 1 // 2
    a_im[2, 2] = 1 // 2
    b_im = zeros(RealT, nstage)
    b_im[1] = 1 // 2
    b_im[2] = 1 // 2
    c_im = zeros(RealT, nstage)
    c_im[1] = 1 // 2
    c_im[2] = 1 // 2
    return IMEXButcher(a, b, c, a_im, b_im, c_im)
end

"""
    Theseus.SSP2222()

A second-order, two-stage type I IMEX method developed by Pareschi and Russo (2005).
The explicit part is strong stability preserving (SSP), the implicit part
is L-stable.

## References
- Lorenzo Pareschi and Giovanni Russo (2005)
  *Implicit–Explicit Runge–Kutta schemes and applications to hyperbolic systems with relaxation.*
  *Journal of Computational Physics* 203(2):469–491.
  [DOI: 10.1007/s10915-004-4636-4](https://doi.org/10.1007/s10915-004-4636-4)
- Sebastiano Boscarino and Giovanni Russo (2024)
  *Asymptotic preserving methods for quasilinear hyperbolic systems with stiff relaxation: a review.*
  [DOI: 10.1007/s40324-024-00351-x](https://doi.org/10.1007/s40324-024-00351-x)
"""
struct SSP2222 <: RKIMEX{2} end
function RKTableau(alg::SSP2222, RealT)
    # IMEX-SSP2(2,2,2) L-Stable Scheme
    nstage = 2
    gamma = 1 - 1 / sqrt(convert(RealT, 2))
    a = zeros(RealT, nstage, nstage)
    a[2, 1] = 1
    b = zeros(RealT, nstage)
    b[1] = 1 // 2
    b[2] = 1 // 2
    c = zeros(RealT, nstage)
    c[2] = 1
    a_im = zeros(RealT, nstage, nstage)
    a_im[1, 1] = gamma
    a_im[2, 1] = 1 - 2 * gamma
    a_im[2, 2] = gamma
    b_im = zeros(RealT, nstage)
    b_im[1] = 1 // 2
    b_im[2] = 1 // 2
    c_im = zeros(RealT, nstage)
    c_im[1] = gamma
    c_im[2] = 1 - gamma
    return IMEXButcher(a, b, c, a_im, b_im, c_im)
end

"""
    Theseus.SSP2322()

A second-order, three-stage type I IMEX method developed by Pareschi and Russo (2005).
The explicit part is strong stability preserving (SSP), the implicit part
is stiffly accurate (SA) and thus L-stable.

## References
- Lorenzo Pareschi and Giovanni Russo (2005)
  *Implicit–Explicit Runge–Kutta schemes and applications to hyperbolic systems with relaxation.*
  *Journal of Computational Physics* 203(2):469–491.
  [DOI: 10.1007/s10915-004-4636-4](https://doi.org/10.1007/s10915-004-4636-4)
"""
struct SSP2322 <: RKIMEX{3} end
function RKTableau(alg::SSP2322, RealT)
    # IMEX-SSP2(3,2,2) Stiffly Accurate Scheme
    nstage = 3
    a = zeros(RealT, nstage, nstage)
    a[3, 2] = 1
    b = zeros(RealT, nstage)
    b[2] = 1 // 2
    b[3] = 1 // 2
    c = zeros(RealT, nstage)
    c[3] = 1
    a_im = zeros(RealT, nstage, nstage)
    a_im[1, 1] = 1 // 2
    a_im[2, 1] = -1 // 2
    a_im[2, 2] = 1 // 2
    a_im[3, 2] = 1 // 2
    a_im[3, 3] = 1 // 2
    b_im = zeros(RealT, nstage)
    b_im[2] = 1 // 2
    b_im[3] = 1 // 2
    c_im = zeros(RealT, nstage)
    c_im[1] = 1 // 2
    c_im[3] = 1
    return IMEXButcher(a, b, c, a_im, b_im, c_im)
end

"""
    Theseus.SSP2332()

A second-order, three-stage type I IMEX method developed by Pareschi and Russo (2005).
The explicit part is strong stability preserving (SSP), the implicit part
is stiffly accurate (SA) and thus L-stable.

## References
- Lorenzo Pareschi and Giovanni Russo (2005)
  *Implicit–Explicit Runge–Kutta schemes and applications to hyperbolic systems with relaxation.*
  *Journal of Computational Physics* 203(2):469–491.
  [DOI: 10.1007/s10915-004-4636-4](https://doi.org/10.1007/s10915-004-4636-4)
- Sebastiano Boscarino and Giovanni Russo (2024)
  *Asymptotic preserving methods for quasilinear hyperbolic systems with stiff relaxation: a review.*
  [DOI: 10.1007/s40324-024-00351-x](https://doi.org/10.1007/s40324-024-00351-x)
"""
struct SSP2332 <: RKIMEX{3} end
function RKTableau(alg::SSP2332, RealT)
    # IMEX-SSP2(3,3,2) Stiffly Accurate Scheme
    nstage = 3
    a = zeros(RealT, nstage, nstage)
    a[2, 1] = 1 // 2
    a[3, 1] = 1 // 2
    a[3, 2] = 1 // 2
    b = zeros(RealT, nstage)
    b[1] = 1 // 3
    b[2] = 1 // 3
    b[3] = 1 // 3
    c = zeros(RealT, nstage)
    c[2] = 1 // 2
    c[3] = 1
    a_im = zeros(RealT, nstage, nstage)
    a_im[1, 1] = 1 // 4
    a_im[2, 2] = 1 // 4
    a_im[3, 1] = 1 // 3
    a_im[3, 2] = 1 // 3
    a_im[3, 3] = 1 // 3
    b_im = zeros(RealT, nstage)
    b_im[1] = 1 // 3
    b_im[2] = 1 // 3
    b_im[3] = 1 // 3
    c_im = zeros(RealT, nstage)
    c_im[1] = 1 // 4
    c_im[2] = 1 // 4
    c_im[3] = 1
    return IMEXButcher(a, b, c, a_im, b_im, c_im)
end

"""
    Theseus.SSP3332()

A second-order, three-stage type I IMEX method developed by Pareschi and Russo (2005).
The explicit part is strong stability preserving (SSP) and third-order accurate,
the implicit part is L-stable.

## References
- Lorenzo Pareschi and Giovanni Russo (2005)
  *Implicit–Explicit Runge–Kutta schemes and applications to hyperbolic systems with relaxation.*
  *Journal of Computational Physics* 203(2):469–491.
  [DOI: 10.1007/s10915-004-4636-4](https://doi.org/10.1007/s10915-004-4636-4)
"""
struct SSP3332 <: RKIMEX{3} end
function RKTableau(alg::SSP3332, RealT)
    # IMEX-SSP3(3,3,2) L-Stable Scheme
    nstage = 3
    gamma = 1 - 1 / sqrt(convert(RealT, 2))
    a = zeros(RealT, nstage, nstage)
    a[2, 1] = 1
    a[3, 1] = 1 // 4
    a[3, 2] = 1 // 4
    b = zeros(RealT, nstage)
    b[1] = 1 // 6
    b[2] = 1 // 6
    b[3] = 2 // 3
    c = zeros(RealT, nstage)
    c[2] = 1
    c[3] = 1 // 2
    a_im = zeros(RealT, nstage, nstage)
    a_im[1, 1] = gamma
    a_im[2, 1] = 1 - 2 * gamma
    a_im[2, 2] = gamma
    a_im[3, 1] = 1 // 2 - gamma
    a_im[3, 3] = gamma
    b_im = zeros(RealT, nstage)
    b_im[1] = 1 // 6
    b_im[2] = 1 // 6
    b_im[3] = 2 // 3
    c_im = zeros(RealT, nstage)
    c_im[1] = gamma
    c_im[2] = 1 - gamma
    c_im[3] = 1 // 2
    return IMEXButcher(a, b, c, a_im, b_im, c_im)
end

"""
    Theseus.SSP3433()

A third-order, four-stage type I IMEX method developed by Pareschi and Russo (2005).
The explicit part is strong stability preserving (SSP) and third-order accurate,
the implicit part is L-stable.

## References
- Lorenzo Pareschi and Giovanni Russo (2005)
  *Implicit–Explicit Runge–Kutta schemes and applications to hyperbolic systems with relaxation.*
  *Journal of Computational Physics* 203(2):469–491.
  [DOI: 10.1007/s10915-004-4636-4](https://doi.org/10.1007/s10915-004-4636-4)
- Sebastiano Boscarino and Giovanni Russo (2024)
  *Asymptotic preserving methods for quasilinear hyperbolic systems with stiff relaxation: a review.*
  [DOI: 10.1007/s40324-024-00351-x](https://doi.org/10.1007/s40324-024-00351-x)
"""
struct SSP3433 <: RKIMEX{4} end
function RKTableau(alg::SSP3433, RealT)
    # IMEX-SSP3(4,3,3) L-Stable Scheme
    nstage = 4
    alpha = RealT(0.24169426078821)
    beta = RealT(0.06042356519705)
    eta = RealT(0.1291528696059)
    a = zeros(RealT, nstage, nstage)
    a[3, 2] = 1
    a[4, 2] = 1 // 4
    a[4, 3] = 1 // 4
    b = zeros(RealT, nstage)
    b[2] = 1 // 6
    b[3] = 1 // 6
    b[4] = 2 // 3
    c = zeros(RealT, nstage)
    c[3] = 1
    c[4] = 1 // 2
    a_im = zeros(RealT, nstage, nstage)
    a_im[1, 1] = alpha
    a_im[2, 1] = -alpha
    a_im[2, 2] = alpha
    a_im[3, 2] = 1 - alpha
    a_im[3, 3] = alpha
    a_im[4, 1] = beta
    a_im[4, 2] = eta
    a_im[4, 3] = 1 // 2 - beta - eta - alpha
    a_im[4, 4] = alpha
    b_im = zeros(RealT, nstage)
    b_im[2] = 1 // 6
    b_im[3] = 1 // 6
    b_im[4] = 2 // 3
    c_im = zeros(RealT, nstage)
    c_im[1] = alpha
    c_im[3] = 1
    c_im[4] = 1 // 2
    return IMEXButcher(a, b, c, a_im, b_im, c_im)
end

"""
    Theseus.AGSA342()

A second-order, four-stage type I IMEX method listed by Biswas, Ketcheson,
Ranocha, and Schütz (2025), Table 12. The explicit part is FSAL and the
implicit part is stiffly accurate, hence the method is globally stiffly accurate
(GSA).

## References
- Abhijit Biswas, David I. Ketcheson, Hendrik Ranocha, and Jochen Schütz (2025)
  *Traveling-Wave Solutions and Structure-Preserving Numerical Methods for a
  Hyperbolic Approximation of the Korteweg-de Vries Equation.*
  [DOI: 10.1007/s10915-025-02898-x](https://doi.org/10.1007/s10915-025-02898-x)
"""
struct AGSA342 <: RKIMEX{4} end
function RKTableau(alg::AGSA342, RealT)
    nstage = 4

    a = zeros(RealT, nstage, nstage)
    a[2, 1] = RealT(-139833537) / RealT(38613965)
    a[3, 1] = RealT(85870407) / RealT(49798258)
    a[3, 2] = RealT(-121251843) / RealT(1756367063)

    b = zeros(RealT, nstage)
    b[1] = RealT(1) / RealT(6)
    b[2] = RealT(1) / RealT(6)
    b[3] = RealT(2) / RealT(3)

    a[4, :] .= b
    c = vec(sum(a, dims = 2))
    c[4] = one(RealT)

    gamma = RealT(202439144) / RealT(118586105)

    a_im = zeros(RealT, nstage, nstage)
    a_im[1, 1] = RealT(168999711) / RealT(74248304)
    a_im[2, 1] = RealT(44004295) / RealT(24775207)
    a_im[2, 2] = gamma
    a_im[3, 1] = RealT(-6418119) / RealT(169001713)
    a_im[3, 2] = RealT(-748951821) / RealT(1043823139)
    a_im[3, 3] = RealT(12015439) / RealT(183058594)

    b_im = zeros(RealT, nstage)
    b_im[1] = one(RealT) - gamma - RealT(1) / RealT(3)
    b_im[2] = RealT(1) / RealT(3)
    b_im[4] = gamma

    a_im[4, :] .= b_im
    c_im = vec(sum(a_im, dims = 2))

    @assert c ≈ vec(sum(a, dims = 2))
    @assert c_im ≈ vec(sum(a_im, dims = 2))
    @assert b ≈ a[end, :]
    @assert b_im ≈ a_im[end, :]

    return IMEXButcher(a, b, c, a_im, b_im, c_im)
end

"""
    Theseus.HT222()

A second-order, two-stage type II IMEX method.
The explicit part is strong stability preserving (SSP), the implicit part
is A-stable but not L-stable.

## References
- Sebastiano  Boscarino, Lorenzo Pareschi, and Giovanni Russo (2025)
  Implicit-explicit methods for evolutionary partial differential equations.
  [DOI: 10.1137/1.9781611978209](https://doi.org/10.1137/1.9781611978209)
"""
struct HT222 <: RKIMEX{2} end
function RKTableau(alg::HT222, RealT)
    nstage = 2
    a = zeros(RealT, nstage, nstage)
    a[2, 1] = 1
    b = zeros(RealT, nstage)
    b[1] = 1 // 2
    b[2] = 1 // 2
    c = zeros(RealT, nstage)
    c[2] = 1
    a_im = zeros(RealT, nstage, nstage)
    a_im[2, 1] = 1 // 2
    a_im[2, 2] = 1 // 2
    b_im = zeros(RealT, nstage)
    b_im[1] = 1 // 2
    b_im[2] = 1 // 2
    c_im = zeros(RealT, nstage)
    c_im[2] = 1
    return IMEXButcher(a, b, c, a_im, b_im, c_im)
end

"""
    Theseus.ARS111()

A first-order, effectively one-stage, globally stiffly accurate (GSA) type II IMEX method
developed by Ascher, Ruuth, and Spiteri (1997).

## References

- Uri M. Ascher, Steven J. Ruuth, and Raymond J Spiteri (1997)
  Implicit-explicit Runge-Kutta methods for time-dependent
  partial differential equations.
  [DOI: 10.1016/S0168-9274(97)00056-1](https://doi.org/10.1016/S0168-9274(97)00056-1)
- Sebastiano Boscarino and Giovanni Russo (2024)
  Asymptotic preserving methods for quasilinear hyperbolic systems with
  stiff relaxation: a review.
  [DOI: 10.1007/s40324-024-00351-x](https://doi.org/10.1007/s40324-024-00351-x)
- Sebastiano  Boscarino, Lorenzo Pareschi, and Giovanni Russo (2025)
  Implicit-explicit methods for evolutionary partial differential equations.
  [DOI: 10.1137/1.9781611978209](https://doi.org/10.1137/1.9781611978209)
"""
struct ARS111 <: RKIMEX{2} end
function RKTableau(alg::ARS111, RealT)
    # ARS(1,1,1) IMEX Runge-Kutta - First order stiffly accurate
    nstage = 2
    a = zeros(RealT, nstage, nstage)
    a[2, 1] = 1
    b = zeros(RealT, nstage)
    b[1] = 1
    c = zeros(RealT, nstage)
    c[2] = 1
    a_im = zeros(RealT, nstage, nstage)
    a_im[2, 2] = 1
    b_im = zeros(RealT, nstage)
    b_im[2] = 1
    c_im = zeros(RealT, nstage)
    c_im[2] = 1
    return IMEXButcher(a, b, c, a_im, b_im, c_im)
end

"""
    Theseus.ARS222()

A second-order, effectively two-stage, globally stiffly accurate (GSA) type II IMEX method
developed by Ascher, Ruuth, and Spiteri (1997).

## References

- Uri M. Ascher, Steven J. Ruuth, and Raymond J Spiteri (1997)
  Implicit-explicit Runge-Kutta methods for time-dependent
  partial differential equations.
  [DOI: 10.1016/S0168-9274(97)00056-1](https://doi.org/10.1016/S0168-9274(97)00056-1)
- Sebastiano Boscarino and Giovanni Russo (2024)
  Asymptotic preserving methods for quasilinear hyperbolic systems with
  stiff relaxation: a review.
  [DOI: 10.1007/s40324-024-00351-x](https://doi.org/10.1007/s40324-024-00351-x)
- Sebastiano  Boscarino, Lorenzo Pareschi, and Giovanni Russo (2025)
  Implicit-explicit methods for evolutionary partial differential equations.
  [DOI: 10.1137/1.9781611978209](https://doi.org/10.1137/1.9781611978209)
"""
struct ARS222 <: RKIMEX{3} end
function RKTableau(alg::ARS222, RealT)
    # ARS(2,2,2) IMEX Runge-Kutta - Second order
    nstage = 3
    gamma = 1 - sqrt(convert(RealT, 2)) / 2
    delta = 1 - 1 / (2 * gamma)
    a = zeros(RealT, nstage, nstage)
    a[2, 1] = gamma
    a[3, 1] = delta
    a[3, 2] = 1 - delta
    b = zeros(RealT, nstage)
    b[1] = delta
    b[2] = 1 - delta
    c = zeros(RealT, nstage)
    c[2] = gamma
    c[3] = 1
    a_im = zeros(RealT, nstage, nstage)
    a_im[2, 2] = gamma
    a_im[3, 2] = 1 - gamma
    a_im[3, 3] = gamma
    b_im = zeros(RealT, nstage)
    b_im[2] = 1 - gamma
    b_im[3] = gamma
    c_im = zeros(RealT, nstage)
    c_im[2] = gamma
    c_im[3] = 1
    return IMEXButcher(a, b, c, a_im, b_im, c_im)
end

"""
    Theseus.ARS233()

A third-order, effectively three-stage type II IMEX method
developed by Ascher, Ruuth, and Spiteri (1997).
The implicit part is A-stable but not L-stable.

## References

- Uri M. Ascher, Steven J. Ruuth, and Raymond J Spiteri (1997)
  Implicit-explicit Runge-Kutta methods for time-dependent
  partial differential equations.
  [DOI: 10.1016/S0168-9274(97)00056-1](https://doi.org/10.1016/S0168-9274(97)00056-1)
- Sebastiano  Boscarino, Lorenzo Pareschi, and Giovanni Russo (2025)
  Implicit-explicit methods for evolutionary partial differential equations.
  [DOI: 10.1137/1.9781611978209](https://doi.org/10.1137/1.9781611978209)
"""
struct ARS233 <: RKIMEX{3} end
function RKTableau(alg::ARS233, RealT)
    nstage = 3
    gamma = (3 + sqrt(convert(RealT, 3))) / 6
    a = zeros(RealT, nstage, nstage)
    a[2, 1] = gamma
    a[3, 1] = gamma - 1
    a[3, 2] = 2 * (1 - gamma)
    b = zeros(RealT, nstage)
    b[2] = 1 // 2
    b[3] = 1 // 2
    c = zeros(RealT, nstage)
    c[2] = gamma
    c[3] = 1 - gamma
    a_im = zeros(RealT, nstage, nstage)
    a_im[2, 2] = gamma
    a_im[3, 2] = 1 - 2 * gamma
    a_im[3, 3] = gamma
    b_im = zeros(RealT, nstage)
    b_im[2] = 1 // 2
    b_im[3] = 1 // 2
    c_im = zeros(RealT, nstage)
    c_im[2] = gamma
    c_im[3] = 1 - gamma
    return IMEXButcher(a, b, c, a_im, b_im, c_im)
end

"""
    Theseus.ARS443()

A third-order, effectively four-stage, globally stiffly accurate (GSA) type II IMEX method
developed by Ascher, Ruuth, and Spiteri (1997).

## References

- Uri M. Ascher, Steven J. Ruuth, and Raymond J Spiteri (1997)
  Implicit-explicit Runge-Kutta methods for time-dependent
  partial differential equations.
  [DOI: 10.1016/S0168-9274(97)00056-1](https://doi.org/10.1016/S0168-9274(97)00056-1)
- Sebastiano Boscarino and Giovanni Russo (2024)
  Asymptotic preserving methods for quasilinear hyperbolic systems with
  stiff relaxation: a review.
  [DOI: 10.1007/s40324-024-00351-x](https://doi.org/10.1007/s40324-024-00351-x)
- Sebastiano  Boscarino, Lorenzo Pareschi, and Giovanni Russo (2025)
  Implicit-explicit methods for evolutionary partial differential equations.
  [DOI: 10.1137/1.9781611978209](https://doi.org/10.1137/1.9781611978209)
"""
struct ARS443 <: RKIMEX{5} end
function RKTableau(alg::ARS443, RealT)
    # ARS(4,4,3) IMEX Runge-Kutta - Third order
    nstage = 5
    a = zeros(RealT, nstage, nstage)
    a[2, 1] = 1 // 2
    a[3, 1] = 11 // 18
    a[3, 2] = 1 // 18
    a[4, 1] = 5 // 6
    a[4, 2] = -5 // 6
    a[4, 3] = 1 // 2
    a[5, 1] = 1 // 4
    a[5, 2] = 7 // 4
    a[5, 3] = 3 // 4
    a[5, 4] = -7 // 4
    b = zeros(RealT, nstage)
    b[1] = 1 // 4
    b[2] = 7 // 4
    b[3] = 3 // 4
    b[4] = -7 // 4
    c = zeros(RealT, nstage)
    c[2] = 1 // 2
    c[3] = 2 // 3
    c[4] = 1 // 2
    c[5] = 1
    a_im = zeros(RealT, nstage, nstage)
    a_im[2, 2] = 1 // 2
    a_im[3, 2] = 1 // 6
    a_im[3, 3] = 1 // 2
    a_im[4, 2] = -1 // 2
    a_im[4, 3] = 1 // 2
    a_im[4, 4] = 1 // 2
    a_im[5, 2] = 3 // 2
    a_im[5, 3] = -3 // 2
    a_im[5, 4] = 1 // 2
    a_im[5, 5] = 1 // 2
    b_im = zeros(RealT, nstage)
    b_im[2] = 3 // 2
    b_im[3] = -3 // 2
    b_im[4] = 1 // 2
    b_im[5] = 1 // 2
    c_im = zeros(RealT, nstage)
    c_im[2] = 1 // 2
    c_im[3] = 2 // 3
    c_im[4] = 1 // 2
    c_im[5] = 1
    return IMEXButcher(a, b, c, a_im, b_im, c_im)
end


"""
    Theseus.KenCarpARK324L2SA()

A third-order, four-stage type II IMEX method of Kennedy and Carpenter (2003),
also denoted ARK3(2)4 L[2]SA. The implicit method is stiffly accurate.

This method is listed in Biswas, Ketcheson, Ranocha, and Schütz (2025), Table 16.

## References
- Christopher A. Kennedy and Mark H. Carpenter (2003)
  *Additive Runge–Kutta schemes for convection-diffusion-reaction equations.*
  *Applied Numerical Mathematics* 44(1-2):139-181.
  [DOI: 10.1016/S0168-9274(02)00138-1](https://doi.org/10.1016/S0168-9274(02)00138-1)
- Abhijit Biswas, David I. Ketcheson, Hendrik Ranocha, and Jochen Schütz (2025)
  *Traveling-Wave Solutions and Structure-Preserving Numerical Methods for a
  Hyperbolic Approximation of the Korteweg-de Vries Equation.*
  [DOI: 10.1007/s10915-025-02898-x](https://doi.org/10.1007/s10915-025-02898-x)
"""
struct KenCarpARK324L2SA <: RKIMEX{4} end
function RKTableau(alg::KenCarpARK324L2SA, RealT)
    nstage = 4
    gamma = RealT(1767732205903) / RealT(4055673282236)

    c_ex = zeros(RealT, nstage)
    c_ex[2] = RealT(1767732205903) / RealT(2027836641118)
    c_ex[3] = RealT(3) / RealT(5)
    c_ex[4] = RealT(1)

    a_ex = zeros(RealT, nstage, nstage)
    a_ex[2, 1] = c_ex[2]
    a_ex[3, 1] = RealT(5535828885825) / RealT(10492691773637)
    a_ex[3, 2] = RealT(788022342437) / RealT(10882634858940)
    a_ex[4, 1] = RealT(6485989280629) / RealT(16251701735622)
    a_ex[4, 2] = RealT(-4246266847089) / RealT(9704473918619)
    a_ex[4, 3] = RealT(10755448449292) / RealT(10357097424841)

    a_im = zeros(RealT, nstage, nstage)
    a_im[2, 1] = gamma
    a_im[2, 2] = gamma
    a_im[3, 1] = RealT(2746238789719) / RealT(10658868560708)
    a_im[3, 2] = RealT(-640167445237) / RealT(6845629431997)
    a_im[3, 3] = gamma
    a_im[4, 1] = RealT(1471266399579) / RealT(7840856788654)
    a_im[4, 2] = RealT(-4482444167858) / RealT(7529755066697)
    a_im[4, 3] = RealT(11266239266428) / RealT(11593286722821)
    a_im[4, 4] = gamma

    b_ex = copy(a_im[end, :])
    b_im = copy(b_ex)
    c_im = copy(c_ex)

    @assert c_ex ≈ vec(sum(a_ex, dims = 2))
    @assert c_im ≈ vec(sum(a_im, dims = 2))

    return IMEXButcher(a_ex, b_ex, c_ex, a_im, b_im, c_im)
end

"""
    Theseus.KenCarpARK436L2SA()

A fourth-order, six-stage type II IMEX method of Kennedy and Carpenter (2003),
also denoted ARK4(3)6 L[2]SA. The implicit method is stiffly accurate.

This method is listed in Biswas, Ketcheson, Ranocha, and Schütz (2025), Table 17.

## References
- Christopher A. Kennedy and Mark H. Carpenter (2003)
  *Additive Runge–Kutta schemes for convection-diffusion-reaction equations.*
  *Applied Numerical Mathematics* 44(1-2):139-181.
  [DOI: 10.1016/S0168-9274(02)00138-1](https://doi.org/10.1016/S0168-9274(02)00138-1)
- Abhijit Biswas, David I. Ketcheson, Hendrik Ranocha, and Jochen Schütz (2025)
  *Traveling-Wave Solutions and Structure-Preserving Numerical Methods for a
  Hyperbolic Approximation of the Korteweg-de Vries Equation.*
  [DOI: 10.1007/s10915-025-02898-x](https://doi.org/10.1007/s10915-025-02898-x)
"""
struct KenCarpARK436L2SA <: RKIMEX{6} end
function RKTableau(alg::KenCarpARK436L2SA, RealT)
    nstage = 6
    gamma = RealT(1) / RealT(4)

    c_ex = zeros(RealT, nstage)
    c_ex[2] = RealT(1) / RealT(2)
    c_ex[3] = RealT(83) / RealT(250)
    c_ex[4] = RealT(31) / RealT(50)
    c_ex[5] = RealT(17) / RealT(20)
    c_ex[6] = RealT(1)

    a_ex = zeros(RealT, nstage, nstage)
    a_ex[2, 1] = RealT(1) / RealT(2)
    a_ex[3, 1] = RealT(13861) / RealT(62500)
    a_ex[3, 2] = RealT(6889) / RealT(62500)
    a_ex[4, 1] = RealT(-116923316275) / RealT(2393684061468)
    a_ex[4, 2] = RealT(-2731218467317) / RealT(15368042101831)
    a_ex[4, 3] = RealT(9408046702089) / RealT(11113171139209)
    a_ex[5, 1] = RealT(-451086348788) / RealT(2902428689909)
    a_ex[5, 2] = RealT(-2682348792572) / RealT(7519795681897)
    a_ex[5, 3] = RealT(12662868775082) / RealT(11960479115383)
    a_ex[5, 4] = RealT(3355817975965) / RealT(11060851509271)
    a_ex[6, 1] = RealT(647845179188) / RealT(3216320057751)
    a_ex[6, 2] = RealT(73281519250) / RealT(8382639484533)
    a_ex[6, 3] = RealT(552539513391) / RealT(3454668386233)
    a_ex[6, 4] = RealT(3354512671639) / RealT(8306763924573)
    a_ex[6, 5] = RealT(4040) / RealT(17871)

    a_im = zeros(RealT, nstage, nstage)
    a_im[2, 1] = gamma
    a_im[2, 2] = gamma
    a_im[3, 1] = RealT(8611) / RealT(62500)
    a_im[3, 2] = RealT(-1743) / RealT(31250)
    a_im[3, 3] = gamma
    a_im[4, 1] = RealT(5012029) / RealT(34652500)
    a_im[4, 2] = RealT(-654441) / RealT(2922500)
    a_im[4, 3] = RealT(174375) / RealT(388108)
    a_im[4, 4] = gamma
    a_im[5, 1] = RealT(15267082809) / RealT(155376265600)
    a_im[5, 2] = RealT(-71443401) / RealT(120774400)
    a_im[5, 3] = RealT(730878875) / RealT(902184768)
    a_im[5, 4] = RealT(2285395) / RealT(8070912)
    a_im[5, 5] = gamma
    a_im[6, 1] = RealT(82889) / RealT(524892)
    a_im[6, 3] = RealT(15625) / RealT(83664)
    a_im[6, 4] = RealT(69875) / RealT(102672)
    a_im[6, 5] = RealT(-2260) / RealT(8211)
    a_im[6, 6] = gamma

    b_ex = copy(a_im[end, :])
    b_im = copy(b_ex)
    c_im = copy(c_ex)

    @assert c_ex ≈ vec(sum(a_ex, dims = 2))
    @assert c_im ≈ vec(sum(a_im, dims = 2))

    return IMEXButcher(a_ex, b_ex, c_ex, a_im, b_im, c_im)
end

"""
    Theseus.BHR553G1()
   
A third order, stiffly accurate, L-stable type II IMEX method developed by
Boscarino and Russo (2009).

## References
- Sebastiano Boscarino and Giovanni Russo (2009)
  On a class of uniformly accurate IMEX Runge-Kutta schemes and 
  applications to hyperbolic systems with relaxation
  [DOI: 10.1137/080713562], (https://doi.org/10.1137/080713562)
"""
struct BHR553G1 <: RKIMEX{5} end
function RKTableau(alg::BHR553G1, RealT)
    # BHR(5,5,3)_g1 IMEX Runge-Kutta - Third order
    nstage = 5
    gamma = 424782 // 974569
    a = zeros(RealT, nstage, nstage)
    a[2, 1] = 2*gamma
    a[3, 1] = gamma
    a[3, 2] = gamma
    a[4, 1] = -475883375220285986033264 // 594112726933437845704163
    a[4, 3] = 1866233449822026827708736//594112726933437845704163
    a[5, 1] = 62828845818073169585635881686091391737610308247 //176112910684412105319781630311686343715753056000
    a[5, 2] = -302987763081184622639300143137943089 //1535359944203293318639180129368156500
    a[5, 3] = 262315887293043739337088563996093207 // 297427554730376353252081786906492000
    a[5, 4] = -987618231894176581438124717087 // 23877337660202969319526901856000
    b = zeros(RealT, nstage)
    b[1] = 487698502336740678603511//1181159636928185920260208
    b[3] = 302987763081184622639300143137943089//1535359944203293318639180129368156500
    b[4] = -105235928335100616072938218863//2282554452064661756575727198000
    #
    b[5] = gamma
    c = zeros(RealT, nstage)
    c[2] = 2* gamma
    c[3] = 902905985686//1035759735069
    c[4] = 2684624//1147171
    c[5] = 1
    a_im = zeros(RealT, nstage, nstage)
    a_im[2, 1] = gamma
    a_im[2, 2] = gamma
    a_im[3, 1] = gamma
    a_im[4, 1] = -3012378541084922027361996761794919360516301377809610//45123394056585269977907753045030512597955897345819349
    a_im[5, 1] = b[1]   
    a_im[3, 2] = -31733082319927313//455705377221960889379854647102
    a_im[3, 3] = gamma
    a_im[4, 2] = -62865589297807153294268//102559673441610672305587327019095047
    a_im[4, 3] = 418769796920855299603146267001414900945214277000//212454360385257708555954598099874818603217167139
    a_im[4, 4] = gamma
    a_im[5, 3] = b[3]
    a_im[5, 4] = b[4]
    a_im[5, 5] = gamma
    b_im = zeros(RealT, nstage)
    b_im[1] = b[1]
    b_im[3] = b[3]
    b_im[4] = b[4]
    b_im[5] = b[5]
    c_im = zeros(RealT, nstage)
    c_im[2] = c[2]
    c_im[3] = c[3]
    c_im[4] = c[4]
    c_im[5] = c[5]
    return IMEXButcher(a, b, c, a_im, b_im, c_im)
end

"""
    Theseus.BHR553G2()
   
A third order, stiffly accurate, L-stable type II IMEX method developed by
Boscarino and Russo (2009).

## References
- Sebastiano Boscarino and Giovanni Russo (2009)
  On a class of uniformly accurate IMEX Runge-Kutta schemes and 
  applications to hyperbolic systems with relaxation
  [DOI: 10.1137/080713562], (https://doi.org/10.1137/080713562)
"""
struct BHR553G2 <: RKIMEX{5} end
function RKTableau(alg::BHR553G2, RealT)
    # BHR(5,5,3)_g2 IMEX Runge-Kutta - Third order
    nstage = 5
    gamma2= 2051948 // 3582211
    a = zeros(RealT, nstage, nstage)
    a[2, 1] = 2*gamma2
    a[3, 1] = 473447115440655855452482357894373//1226306256343706154920072735579148
    a[4, 1] = 37498105210828143724516848//172642583546398006173766007
    a[5, 1] =  -3409975860212064612303539855622639333030782744869519//5886704102363745137792385361113084313351870216475136
    a[3, 2] = 129298766034131882323069978722019//1226306256343706154920072735579148
    a[5, 2] =  -237416352433826978856941795734073//554681702576878342891447163499456
    a[4, 3] = 76283359742561480140804416//172642583546398006173766007
    a[5, 3] = 4298159710546228783638212411650783228275//2165398513352098924587211488610407046208
    a[5, 4] = 6101865615855760853571922289749//272863973025878249803640374568448
    b = zeros(RealT, nstage)
    b[1] =  -2032971420760927701493589//38017147656515384190997416
    b[3] = 2197602776651676983265261109643897073447//945067123279139583549933947379097184164
    b[4] = -128147215194260398070666826235339//69468482710687503388562952626424
    b[5] = gamma2
    c = zeros(RealT, nstage)
    c[2] = 2* gamma2
    c[3] = 12015769930846//24446477850549
    c[4] = 3532944//5360597
    c[5] = 1
    a_im = zeros(RealT, nstage, nstage)
    a_im[2, 1] = gamma2
    a_im[2, 2] = gamma2
    a_im[3, 1] = 259252258169672523902708425780469319755//4392887760843243968922388674191715336228
    a_im[4, 1] = 1103202061574553405285863729195740268785131739395559693754//9879457735937277070641522414590493459028264677925767305837
    a_im[5, 1] = b[1]   
    a_im[3, 2] = -172074174703261986564706189586177//1226306256343706154920072735579148
    a_im[3, 3] = gamma2
    a_im[4, 2] =  -103754520567058969566542556296087324094//459050363888246734833121482275319954529
    a_im[4, 3] = 3863207083069979654596872190377240608602701071947128//19258690251287609765240683320611425745736762681950551
    a_im[4, 4] = gamma2
    a_im[5, 3] = b[3]
    a_im[5, 4] = b[4]
    a_im[5, 5] = gamma2
    b_im = zeros(RealT, nstage)
    b_im[1] = b[1]
    b_im[3] = b[3]
    b_im[4] = b[4]
    b_im[5] = b[5]
    c_im = zeros(RealT, nstage)
    c_im[2] = c[2]
    c_im[3] = c[3]
    c_im[4] = c[4]
    c_im[5] = c[5]
    return IMEXButcher(a, b, c, a_im, b_im, c_im)
end

"""
    Theseus.KenCarpARK437()

A fourth-order, seven-stage type II IMEX method developed by Kennedy and Carpenter (2019).
The implicit method is A-stable, L-stable, and stiffly accurate.

## References
- Christopher A. Kennedy and Mark H. Carpenter (2019)
  *Higher-order additive Runge–Kutta schemes for ordinary differential equations.*
  *Applied Numerical Mathematics* 136:183-205.
  [DOI: 10.1016/j.apnum.2018.10.007](https://doi.org/10.1016/j.apnum.2018.10.007)
"""
struct KenCarpARK437 <: RKIMEX{7} end
function RKTableau(alg::KenCarpARK437, RealT)
    nstage = 7
    gamma = RealT(1235) / RealT(10_000)

    c_ex = zeros(RealT, nstage)
    c_ex[2] = RealT(247) / RealT(1000)
    c_ex[3] = RealT(4276536705230) / RealT(10142255878289)
    c_ex[4] = RealT(67) / RealT(200)
    c_ex[5] = RealT(3) / RealT(40)
    c_ex[6] = RealT(7) / RealT(10)
    c_ex[7] = RealT(1)

    b_ex = zeros(RealT, nstage)
    b_ex[3] = RealT(9164257142617) / RealT(17756377923965)
    b_ex[4] = RealT(-10812980402763) / RealT(74029279521829)
    b_ex[5] = RealT(1335994250573) / RealT(5691609445217)
    b_ex[6] = RealT(2273837961795) / RealT(8368240463276)
    b_ex[7] = RealT(247) / RealT(2000)

    a_ex = zeros(RealT, nstage, nstage)
    a_ex[2, 1] = c_ex[2]
    a_ex[3, 1] = RealT(247) / RealT(4000)
    a_ex[3, 2] = RealT(2694949928731) / RealT(7487940209513)
    a_ex[4, 1] = RealT(464650059369) / RealT(8764239774964)
    a_ex[4, 2] = RealT(878889893998) / RealT(2444806327765)
    a_ex[4, 3] = RealT(-952945855348) / RealT(12294611323341)
    a_ex[5, 1] = RealT(476636172619) / RealT(8159180917465)
    a_ex[5, 2] = RealT(-1271469283451) / RealT(7793814740893)
    a_ex[5, 3] = RealT(-859560642026) / RealT(4356155882851)
    a_ex[5, 4] = RealT(1723805262919) / RealT(4571918432560)
    a_ex[6, 1] = RealT(6338158500785) / RealT(11769362343261)
    a_ex[6, 2] = RealT(-4970555480458) / RealT(10924838743837)
    a_ex[6, 3] = RealT(3326578051521) / RealT(2647936831840)
    a_ex[6, 4] = RealT(-880713585975) / RealT(1841400956686)
    a_ex[6, 5] = RealT(-1428733748635) / RealT(8843423958496)
    a_ex[7, 1] = RealT(760814592956) / RealT(3276306540349)
    a_ex[7, 2] = a_ex[7, 1]
    a_ex[7, 3] = RealT(-47223648122716) / RealT(6934462133451)
    a_ex[7, 4] = RealT(71187472546993) / RealT(9669769126921)
    a_ex[7, 5] = RealT(-13330509492149) / RealT(9695768672337)
    a_ex[7, 6] = RealT(11565764226357) / RealT(8513123442827)
    @assert c_ex ≈ sum(a_ex, dims = 2)

    a_im = zeros(RealT, nstage, nstage)
    a_im[2, 1] = gamma
    a_im[2, 2] = gamma
    a_im[3, 1] = RealT(624185399699) / RealT(4186980696204)
    a_im[3, 2] = a_im[3, 1]
    a_im[3, 3] = gamma
    a_im[4, 1] = RealT(1258591069120) / RealT(10082082980243)
    a_im[4, 2] = a_im[4, 1]
    a_im[4, 3] = RealT(-322722984531) / RealT(8455138723562)
    a_im[4, 4] = gamma
    a_im[5, 1] = RealT(-436103496990) / RealT(5971407786587)
    a_im[5, 2] = a_im[5, 1]
    a_im[5, 3] = RealT(-2689175662187) / RealT(11046760208243)
    a_im[5, 4] = RealT(4431412449334) / RealT(12995360898505)
    a_im[5, 5] = gamma
    a_im[6, 1] = RealT(-2207373168298) / RealT(14430576638973)
    a_im[6, 2] = a_im[6, 1]
    a_im[6, 3] = RealT(242511121179) / RealT(3358618340039)
    a_im[6, 4] = RealT(3145666661981) / RealT(7780404714551)
    a_im[6, 5] = RealT(5882073923981) / RealT(14490790706663)
    a_im[6, 6] = gamma
    a_im[7, 1] = RealT(0)
    a_im[7, 2] = a_im[7, 1]
    a_im[7, 3] = RealT(9164257142617) / RealT(17756377923965)
    a_im[7, 4] = RealT(-10812980402763) / RealT(74029279521829)
    a_im[7, 5] = RealT(1335994250573) / RealT(5691609445217)
    a_im[7, 6] = RealT(2273837961795) / RealT(8368240463276)
    a_im[7, 7] = gamma
    b_im = copy(b_ex)
    c_im = copy(c_ex)
    @assert c_im ≈ sum(a_im, dims = 2)

    # TODO: Implement embedded methods

    return IMEXButcher(a_ex, b_ex, c_ex, a_im, b_im, c_im)
end


"""
    Theseus.KenCarpARK548()

A fifth-order, eight-stage type II IMEX method developed by Kennedy and Carpenter (2019).
The implicit method is A-stable, L-stable, and stiffly accurate.

## References
- Christopher A. Kennedy and Mark H. Carpenter (2019)
  *Higher-order additive Runge–Kutta schemes for ordinary differential equations.*
  *Applied Numerical Mathematics* 136:183-205.
  [DOI: 10.1016/j.apnum.2018.10.007](https://doi.org/10.1016/j.apnum.2018.10.007)
"""
struct KenCarpARK548 <: RKIMEX{8} end
function RKTableau(alg::KenCarpARK548, RealT)
    nstage = 8
    gamma = RealT(2) / RealT(9)

    c_ex = zeros(RealT, nstage)
    c_ex[2] = RealT(4) / RealT(9)
    c_ex[3] = RealT(6456083330201) / RealT(8509243623797)
    c_ex[4] = RealT(1632083962415) / RealT(14158861528103)
    c_ex[5] = RealT(6365430648612) / RealT(17842476412687)
    c_ex[6] = RealT(18) / RealT(25)
    c_ex[7] = RealT(191) / RealT(200)
    c_ex[8] = RealT(1)

    b_ex = zeros(RealT, nstage)
    b_ex[3] = RealT(3517720773327) / RealT(20256071687669)
    b_ex[4] = RealT(4569610470461) / RealT(17934693873752)
    b_ex[5] = RealT(2819471173109) / RealT(11655438449929)
    b_ex[6] = RealT(3296210113763) / RealT(10722700128969)
    b_ex[7] = RealT(-1142099968913) / RealT(5710983926999)
    b_ex[8] = gamma

    a_ex = zeros(RealT, nstage, nstage)
    a_ex[2, 1] = c_ex[2]
    a_ex[3, 1] = RealT(1) / RealT(9)
    a_ex[3, 2] = RealT(1183333538310) / RealT(1827251437969)
    a_ex[4, 1] = RealT(895379019517) / RealT(9750411845327)
    a_ex[4, 2] = RealT(477606656805) / RealT(13473228687314)
    a_ex[4, 3] = RealT(-112564739183) / RealT(9373365219272)
    a_ex[5, 1] = RealT(-4458043123994) / RealT(13015289567637)
    a_ex[5, 2] = RealT(-2500665203865) / RealT(9342069639922)
    a_ex[5, 3] = RealT(983347055801) / RealT(8893519644487)
    a_ex[5, 4] = RealT(2185051477207) / RealT(2551468980502)
    a_ex[6, 1] = RealT(-167316361917) / RealT(17121522574472)
    a_ex[6, 2] = RealT(1605541814917) / RealT(7619724128744)
    a_ex[6, 3] = RealT(991021770328) / RealT(13052792161721)
    a_ex[6, 4] = RealT(2342280609577) / RealT(11279663441611)
    a_ex[6, 5] = RealT(3012424348531) / RealT(12792462456678)
    a_ex[7, 1] = RealT(6680998715867) / RealT(14310383562358)
    a_ex[7, 2] = RealT(5029118570809) / RealT(3897454228471)
    a_ex[7, 3] = RealT(2415062538259) / RealT(6382199904604)
    a_ex[7, 4] = RealT(-3924368632305) / RealT(6964820224454)
    a_ex[7, 5] = RealT(-4331110370267) / RealT(15021686902756)
    a_ex[7, 6] = RealT(-3944303808049) / RealT(11994238218192)
    a_ex[8, 1] = RealT(2193717860234) / RealT(3570523412979)
    a_ex[8, 2] = a_ex[8, 1]
    a_ex[8, 3] = RealT(5952760925747) / RealT(18750164281544)
    a_ex[8, 4] = RealT(-4412967128996) / RealT(6196664114337)
    a_ex[8, 5] = RealT(4151782504231) / RealT(36106512998704)
    a_ex[8, 6] = RealT(572599549169) / RealT(6265429158920)
    a_ex[8, 7] = RealT(-457874356192) / RealT(11306498036315)
    @assert c_ex ≈ sum(a_ex, dims = 2)

    a_im = zeros(RealT, nstage, nstage)
    a_im[2, 1] = gamma
    a_im[2, 2] = gamma
    a_im[3, 1] = RealT(2366667076620) / RealT(8822750406821)
    a_im[3, 2] = a_im[3, 1]
    a_im[3, 3] = gamma
    a_im[4, 1] = RealT(-257962897183) / RealT(4451812247028)
    a_im[4, 2] = a_im[4, 1]
    a_im[4, 3] = RealT(128530224461) / RealT(14379561246022)
    a_im[4, 4] = gamma
    a_im[5, 1] = RealT(-486229321650) / RealT(11227943450093)
    a_im[5, 2] = a_im[5, 1]
    a_im[5, 3] = RealT(-225633144460) / RealT(6633558740617)
    a_im[5, 4] = RealT(1741320951451) / RealT(6824444397158)
    a_im[5, 5] = gamma
    a_im[6, 1] = RealT(621307788657) / RealT(4714163060173)
    a_im[6, 2] = a_im[6, 1]
    a_im[6, 3] = RealT(-125196015625) / RealT(3866852212004)
    a_im[6, 4] = RealT(940440206406) / RealT(7593089888465)
    a_im[6, 5] = RealT(961109811699) / RealT(6734810228204)
    a_im[6, 6] = gamma
    a_im[7, 1] = RealT(2036305566805) / RealT(6583108094622)
    a_im[7, 2] = a_im[7, 1]
    a_im[7, 3] = RealT(-3039402635899) / RealT(4450598839912)
    a_im[7, 4] = RealT(-1829510709469) / RealT(31102090912115)
    a_im[7, 5] = RealT(-286320471013) / RealT(6931253422520)
    a_im[7, 6] = RealT(8651533662697) / RealT(9642993110008)
    a_im[7, 7] = gamma
    a_im[8, 1] = b_ex[1]
    a_im[8, 2] = b_ex[2]
    a_im[8, 3] = b_ex[3]
    a_im[8, 4] = b_ex[4]
    a_im[8, 5] = b_ex[5]
    a_im[8, 6] = b_ex[6]
    a_im[8, 7] = b_ex[7]
    a_im[8, 8] = gamma
    b_im = copy(b_ex)
    c_im = copy(c_ex)
    @assert c_im ≈ sum(a_im, dims = 2)

    # TODO: Implement embedded methods

    return IMEXButcher(a_ex, b_ex, c_ex, a_im, b_im, c_im)
end
