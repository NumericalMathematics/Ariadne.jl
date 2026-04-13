struct RosenbrockButcher{T1 <: AbstractArray, T2 <: AbstractArray} <: RKTableau
    a::T1
    c::T1
    m::T2
    gamma::T2
end

"""
    SSPKnoth

A three-stage, second-order strong-stability preserving (SSP) Rosenbrock-W method
by Knoth and Wolke.

## References
- O. Knoth and R. Wolke (1998)
  *Implicit-explicit coupled multirate methods for reactive flow with stiff chemistry.*
  Atmospheric Environment, 32(3):507–519.
  [DOI: 10.1016/S1352-2310(97)00135-3](https://doi.org/10.1016/S1352-2310(97)00135-3)
"""
struct SSPKnoth <: RosenbrockAlgorithm{3} end

function RKTableau(alg::SSPKnoth, RealT)
    # SSP - Knoth
    nstage = 3
    alpha = zeros(RealT, nstage, nstage)
    alpha[2, 1] = 1
    alpha[3, 1] = 1 // 4
    alpha[3, 2] = 1 // 4

    b = zeros(RealT, nstage)
    b[1] = 1 // 6
    b[2] = 1 // 6
    b[3] = 2 // 3

    gamma = zeros(RealT, nstage, nstage)
    gamma[1, 1] = 1
    gamma[2, 2] = 1
    gamma[3, 1] = -3 // 4
    gamma[3, 2] = -3 // 4
    gamma[3, 3] = 1

    inv_gamma = inv(gamma)
    a = alpha * inv_gamma
    m = transpose(b) * inv_gamma
    c = diagm(inv.(diag(gamma))) - inv_gamma
    diag_gamma = zeros(RealT, nstage)
    diag_gamma[1] = gamma[1, 1]
    diag_gamma[2] = gamma[2, 2]
    diag_gamma[3] = gamma[3, 3]
    return RosenbrockButcher(a, c, vec(m), diag_gamma)
end

"""
    ROS2

A two-stage, second-order Rosenbrock-W method with diagonal parameter
``\\gamma = (1 + 1/\\sqrt{3})/2``.

## References
- Ernst Hairer and Gerhard Wanner (1996)
  *Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems.*
  Springer Series in Computational Mathematics, 2nd edition.
  [DOI: 10.1007/978-3-642-05221-7](https://doi.org/10.1007/978-3-642-05221-7)
"""
struct ROS2 <: RosenbrockAlgorithm{2} end

function RKTableau(alg::ROS2, RealT)

    nstage = 2
    alpha = zeros(RealT, nstage, nstage)
    alpha[2, 1] = 2 // 3

    b = zeros(RealT, nstage)
    b[1] = 1 // 4
    b[2] = 3 // 4

    gamma = zeros(RealT, nstage, nstage)
    gam = (1 + 1 / sqrt(convert(RealT, 3))) / 2
    gamma[1, 1] = gam
    gamma[2, 1] = -4 * gam / 3
    gamma[2, 2] = gam

    inv_gamma = inv(gamma)
    a = alpha * inv_gamma
    m = transpose(b) * inv_gamma
    c = diagm(inv.(diag(gamma))) - inv_gamma
    diag_gamma = zeros(RealT, nstage)
    diag_gamma[1] = gamma[1, 1]
    diag_gamma[2] = gamma[2, 2]

    return RosenbrockButcher(a, c, vec(m), diag_gamma)
end
