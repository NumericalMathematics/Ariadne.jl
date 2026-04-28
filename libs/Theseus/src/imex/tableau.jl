struct IMEXButcher{T1 <: AbstractArray, T2 <: AbstractArray} <: RKTableau
    a_ex::T1
    b_ex::T2
    c_ex::T2
    a_im::T1
    b_im::T2
    c_im::T2
end

"""
    SP111()

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
    H222()

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
    SSP2222()

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
    SSP2322()

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
    SSP2332()

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
    SSP3332()

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
    SSP3433()

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
    HT222()

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
    ARS111()

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
    ARS222()

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
    ARS233()

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
    ARS443()

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
   TODO 
"""
struct BHR553G1 <: RKIMEX{5} end
function RKTableau(alg::BHR553G1, RealT)
    # BHR(5,5,3)_g1 IMEX Runge-Kutta - Third order
    nstage = 5
    gamma=424782 // 974569
    a = zeros(RealT, nstage, nstage)
    a[2, 1] = 2*gamma
    a[3, 1] = gamma
    a[3, 2] = gamma
    a[4, 1] = −475883375220285986033264 // 594112726933437845704163
    a[4, 3] = 1866233449822026827708736/594112726933437845704163
    a[5, 1] = 62828845818073169585635881686091391737610308247 //176112910684412105319781630311686343715753056000
    a[5, 2] = -302987763081184622639300143137943089 //1535359944203293318639180129368156500
    a[5, 3] = 262315887293043739337088563996093207 // 297427554730376353252081786906492000
    a[5, 4] = −987618231894176581438124717087 // 23877337660202969319526901856000
    b = zeros(RealT, nstage)
    b[1] = 487698502336740678603511//1181159636928185920260208
    b[3] = - a[5, 2]
    b[4] = −105235928335100616072938218863//2282554452064661756575727198000
    c = zeros(RealT, nstage)
    c[2] = 2* gamma
    c[3] = 902905985686//1035759735069
    c[4] = 2684624//1147171
    c[5] = 1
    a_im = zeros(RealT, nstage, nstage)
    a_im[2, 1] = gamma
    a_im[2, 2] = gamma
    a_im[3, 1] = gamma
    a_im[4, 1] = −3012378541084922027361996761794919360516301377809610//45123394056585269977907753045030512597955897345819349
    a_im[5, 1] = b[1]   
    a_im[3, 2] = −31733082319927313//455705377221960889379854647102
    a_im[3, 3] = gamma
    a_im[4, 1] = −3012378541084922027361996761794919360516301377809610//45123394056585269977907753045030512597955897345819349
    a_im[4, 2] = −62865589297807153294268//102559673441610672305587327019095047
    a_im[4, 3] = 418769796920855299603146267001414900945214277000//212454360385257708555954598099874818603217167139
    a_im[4, 4] = gamma
    a_im[5, 1] = 487698502336740678603511//1181159636928185920260208
    a_im[5, 2] = 0
    a_im[5, 3] = 0
    a_im[5, 4] = 02987763081184622639300143137943089//1535359944203293318639180129368156500
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