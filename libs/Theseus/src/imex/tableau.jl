struct IMEXButcher{T1 <: AbstractArray, T2 <: AbstractArray} <: RKTableau
    a_ex::T1
    b_ex::T2
    c_ex::T2
    a_im::T1
    b_im::T2
    c_im::T2
end

struct SSP2222 <: RKIMEX{2} end
function RKTableau(alg::SSP2222, RealT)
    # IMEX-SSP2(2,2,2) L-Stable Scheme
    nstage = 2
    gamma = 1 - 1 / sqrt(2)
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

struct SSP3332 <: RKIMEX{3} end
function RKTableau(alg::SSP3332, RealT)
    # IMEX-SSP3(3,3,2) L-Stable Scheme
    nstage = 3
    gamma = 1 - 1 / sqrt(2)
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

struct ARS222 <: RKIMEX{3} end
function RKTableau(alg::ARS222, RealT)
    # ARS(2,2,2) IMEX Runge-Kutta - Second order
    nstage = 3
    gamma = 1 - sqrt(2) / 2
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
