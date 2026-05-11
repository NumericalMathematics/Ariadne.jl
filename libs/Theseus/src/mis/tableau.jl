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
    d = SVector(0.0, d[2], d[3], d[4])

    return MISRK(beta, alfa, gamma, d)
end


struct MISRK4 <: MISSlowAlgorithm{5} end

function RKTableau(alg::MISRK4, RealT)
            rkstages = 5
            beta = zeros(RealT, rkstages, rkstages)
            alfa = zeros(RealT, rkstages, rkstages)
            gamma = zeros(RealT, rkstages, rkstages)
            d = zeros(RealT, rkstages, 1)
            beta[2, 1] = 0.38758444641450318
            beta[3, 1] = -2.5318448354142823E-002
            beta[3, 2] = 0.38668943087310403
            beta[4, 1] = 0.20899983523553325
            beta[4, 2] = -0.45856648476371231
            beta[4, 3] = 0.43423187573425748
            beta[5, 1] = -0.10048822195663100
            beta[5, 2] = -0.46186171956333327
            beta[5, 3] = 0.83045062122462809
            beta[5, 4] = 0.27014914900250392
    
            alfa[3, 2] = 0.52349249922385610
            alfa[4, 2] = 1.1683374366893629
            alfa[4, 3] = -0.75762080241712637
            alfa[5, 2] = -3.6477233846797109E-002
            alfa[5, 3] = 0.56936148730740477
            alfa[5, 4] = 0.47746263002599681
    
            gamma[3, 2] = 0.13145089796226542
            gamma[4, 2] = -0.36855857648747881
            gamma[4, 3] = 0.33159232636600550
            gamma[5, 2] = -6.5767130537473045E-002
            gamma[5, 3] = 4.0591093109036858E-002
            gamma[5, 4] = 6.4902111640806712E-002
    
            d2 = beta[2, 1]
            d3 = beta[3, 1] + beta[3, 2]
            d4 = beta[4, 1] + beta[4, 2] + beta[4, 3]
            d5 = beta[5, 1] + beta[5, 2] + beta[5, 3] + beta[5, 4]
	    d = SVector(0, d2, d3, d4, d5)

	return MISRK(beta, alfa, gamma, d)
end

struct MIS2 <: MISSlowAlgorithm{3} end

function RKTableau(alg::MIS2, RealT)
    rkstages = 4
    beta  = zeros(RealT, rkstages, rkstages)
    alfa  = zeros(RealT, rkstages, rkstages)
    gamma = zeros(RealT, rkstages, rkstages)
    d     = zeros(RealT, rkstages, 1)

    # β coefficients
    beta[2, 1] =  0.126848494553
    beta[3, 1] = -0.784838278826
    beta[3, 2] =  1.37442675268
    beta[4, 1] = -0.0456727081749
    beta[4, 2] = -0.00875082271190
    beta[4, 3] =  0.524775788629

    # α coefficients
    alfa[3, 2] =  0.536946566710
    alfa[4, 2] =  0.480892968551
    alfa[4, 3] =  0.500561163566

    # γ coefficients
    gamma[3, 2] =  0.652465126004
    gamma[4, 2] = -0.0732769849457
    gamma[4, 3] =  0.144902430420

    d2 = beta[2, 1]
    d3 = beta[3, 1] + beta[3, 2]
    d4 = beta[4, 1] + beta[4, 2] + beta[4, 3]
    d  = SVector(0, d2, d3, d4)

    return MISRK(beta, alfa, gamma, d)
end

struct JEB3 <: MISSlowAlgorithm{4} end

function RKTableau(alg::JEB3, RealT)
rkstages = 4
    beta = zeros(RealT, rkstages, rkstages)
    alpha = zeros(RealT, rkstages, rkstages)
    gamma = zeros(RealT, rkstages, rkstages)
    d = zeros(RealT, rkstages)

    beta[2,1] = 2.0492941060709863e-001
    beta[3,1] = -4.5477553356788974e-001
    beta[3,2] = 9.5613538239378981e-001
    beta[4,1] = -3.5970281266252929e-002
    beta[4,2] = -1.5363649484946584e-001
    beta[4,3] = 7.0259062712330234e-001

    gamma[3,2] = -8.2176071248067006e-001
    gamma[4,2] = -3.8080670922635063e-001
    gamma[4,3] = 4.5653105107801978e-001

    alpha[3,2] = 7.0302371060435331e-001
    alpha[4,2] = 4.2492220536139252e-001
    alpha[4,3] = 5.4545718243573982e-001

    d[2] = beta[2, 1]
    d[3] = beta[3, 1] + beta[3, 2]
    d[4] = beta[4, 1] + beta[4, 2] + beta[4, 3]
    d = SVector(0, d[2], d[3], d[4])
	return MISRK(beta, alpha, gamma, d)
end 
