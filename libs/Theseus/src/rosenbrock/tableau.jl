struct RosenbrockButcher{T1 <: AbstractArray, T2 <: AbstractArray} <: RKTableau
	a::T1
	c::T1
	m::T2
	gamma::T2
end

struct SSPKnoth <: RosenbrockAlgorithm{3} end

function RKTableau(alg::SSPKnoth)
		# SSP - Knoth
	nstage = 3
	alpha = zeros(Float64, nstage, nstage)
	alpha[2, 1] = 1
	alpha[3, 1] = 1 / 4
	alpha[3, 2] = 1 / 4

	b = zeros(Float64, nstage)
	b[1] = 1 / 6
	b[2] = 1 / 6
	b[3] = 2 / 3

	gamma = zeros(Float64, nstage, nstage)
	gamma[1, 1] = 1
	gamma[2, 2] = 1
	gamma[3, 1] = -3 / 4
	gamma[3, 2] = -3 / 4
	gamma[3, 3] = 1

	a = alpha * inv(gamma)
	m = transpose(b) * inv(gamma)
	c = diagm(inv.(diag(gamma))) - inv(gamma)
	diag_gamma =  zeros(Float64, nstage)
	diag_gamma[1] = gamma[1,1]
	diag_gamma[2] = gamma[2,2]
	diag_gamma[3] = gamma[3,3]
	return RosenbrockButcher(a, c, vec(m), diag_gamma)
end

struct ROS2 <: RosenbrockAlgorithm{2} end

function RKTableau(alg::ROS2)

	nstage = 2
	alpha = zeros(Float64, nstage, nstage)
	alpha[2, 1] = 2 / 3

	b = zeros(Float64, nstage)
	b[1] = 1 / 4
	b[2] = 3 / 4

	gamma = zeros(Float64, nstage, nstage)
	gam = (1 + 1/sqrt(3))/2
	gamma[1, 1] = gam
	gamma[2, 1] = -4/3 * gam
	gamma[2, 2] = gam

	a = alpha * inv(gamma)
	m = transpose(b) * inv(gamma)
	c = diagm(inv.(diag(gamma))) - inv(gamma)
	diag_gamma =  zeros(Float64, nstage)
	diag_gamma[1] = gamma[1,1]
	diag_gamma[2] = gamma[2,2]
	
	return RosenbrockButcher(a, c, vec(m), diag_gamma)
end
