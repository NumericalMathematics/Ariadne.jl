struct MISRK{T1<:AbstractArray, T2<:AbstractArray} <: RKTableau
        beta::T1
	alfa::T1
	gamma::T1
	d::T2
end

struct MISRK3 <: MISSlowAlgorithm{4} end

function RKTableau(alg::MISRK3, RealT)
    rkstages = 4
    beta = zeros(RealT, rkstages, rkstages)
    alfa = zeros(RealT, rkstages, rkstages)
    gamma = zeros(RealT, rkstages, rkstages)
    d = zeros(RealT, rkstages, 1)
    beta[2, 1] = 1/3
    beta[3, 2] = 1/2
    beta[4, 3] = 1
    d[2] = 1/3
    d[3] = 1/2
    d[4] = 1

    return MISRK(beta, alfa, gamma, d)
end
