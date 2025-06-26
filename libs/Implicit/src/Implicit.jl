module Implicit

using UnPack
using Ariadne: Ariadne
using LinearAlgebra
import Ariadne: JacobianOperator, MOperator
using Krylov

struct MOperator{JOp}
	J::JOp
	dt::Float64
end

Base.size(M::MOperator) = size(M.J)
Base.eltype(M::MOperator) = eltype(M.J)
Base.length(M::MOperator) = length(M.J)

import LinearAlgebra: mul!
function mul!(out::AbstractVector, M::MOperator, v::AbstractVector)
	# out = (I/dt - J(f,x,p)) * v
	mul!(out, M.J, v)
	@. out = v / M.dt - out
	return nothing
end

# Wrapper type for solutions from Implicit.jl's own time integrators, partially mimicking
# SciMLBase.ODESolution
struct TimeIntegratorSolution{tType, uType, P}
	t::tType
	u::uType
	prob::P
end

# Abstract supertype of Implict.jl's own time integrators for dispatch
abstract type AbstractTimeIntegrator end

using DiffEqBase: DiffEqBase

import DiffEqBase: solve, CallbackSet, ODEProblem
export solve, ODEProblem

# Interface required by DiffEqCallbacks.jl
function DiffEqBase.get_tstops(integrator::AbstractTimeIntegrator)
	return integrator.opts.tstops
end
function DiffEqBase.get_tstops_array(integrator::AbstractTimeIntegrator)
	return get_tstops(integrator).valtree
end
function DiffEqBase.get_tstops_max(integrator::AbstractTimeIntegrator)
	return maximum(get_tstops_array(integrator))
end

function finalize_callbacks(integrator::AbstractTimeIntegrator)
	callbacks = integrator.opts.callback

	return if callbacks isa CallbackSet
		foreach(callbacks.discrete_callbacks) do cb
			cb.finalize(cb, integrator.u, integrator.t, integrator)
		end
		foreach(callbacks.continuous_callbacks) do cb
			cb.finalize(cb, integrator.u, integrator.t, integrator)
		end
	end
end

import SciMLBase: get_du, get_tmp_cache, u_modified!,
	init, step!, check_error,
	get_proposed_dt, set_proposed_dt!,
	terminate!, remake, add_tstop!, has_tstop, first_tstop


# Abstract base type for time integration schemes
abstract type SimpleImplicitAlgorithm{N} end
abstract type NonDirect{N} <: SimpleImplicitAlgorithm{N} end
abstract type Direct{N} <: SimpleImplicitAlgorithm{N} end

stages(::SimpleImplicitAlgorithm{N}) where {N} = N

struct ImplicitEuler <: NonDirect{1} end
function (::ImplicitEuler)(res, uₙ, Δt, f!, du, u, p, t, stages, stage)
	f!(du, u, p, t + Δt) # t = t0 + c_1 * Δt

	res .= uₙ .+ Δt .* du .- u # Δt * a_11
	return nothing
end

struct ImplicitMidpoint <: SimpleImplicitAlgorithm{1} end
function (::ImplicitMidpoint)(res, uₙ, Δt, f!, du, u, p, t, stages, stage)
	# Evaluate f at midpoint: f((uₙ + u)/2, t + Δt/2)
	# Use res for a temporary allocation (uₙ .+ u) ./ 2
	uuₙ = res
	uuₙ .= 0.5 .* (uₙ .+ u)
	f!(du, uuₙ, p, t + 0.5 * Δt)

	res .= uₙ .+ Δt .* du .- u
	return nothing
end

struct ImplicitTrapezoid <: SimpleImplicitAlgorithm{1} end
function (::ImplicitTrapezoid)(res, uₙ, Δt, f!, du, u, p, t, stages, stage)
	# Need to evaluate f at both endpoints
	# f(uₙ, t) and f(u, t + Δt)
	# Use res as the temporary for duₙ = f(uₙ, t)
	duₙ = res
	f!(duₙ, uₙ, p, t)
	f!(du, u, p, t + Δt)

	res .= uₙ .+ (Δt / 2) .* (duₙ .+ du) .- u
	return nothing
end

"""
	TRBDF2

TR-BDF2 based solver after [Bank1985-gh](@cite).
Using the formula given in [Bonaventura2021-za](@cite) eq (1).
See [Hosea1996-xv](@cite) for how it relates to implicit RK methods
"""
struct TRBDF2 <: SimpleImplicitAlgorithm{2} end
function (::TRBDF2)(res, uₙ, Δt, f!, du, u, p, t, stages, stage)
	γ = 2 - √2
	return if stage == 1
		# Stage 1: Trapezoidal rule to t + γΔt
		# u here is u₁ candidate
		duₙ = res
		f!(duₙ, uₙ, p, t)
		f!(du, u, p, t + γ * Δt)

		res .= uₙ .+ ((γ / 2) * Δt) .* (duₙ .+ du) .- u
	else
		# Stage 2: BDF2 from t + γΔt to t + Δt
		# Note these are unequal timestep
		f!(du, u, p, t + Δt)

		u₁ = stages[1]

		# Bank1985 defines in eq 32
		# (2-γ)u + (1-γ)Δt * f(u, t+Δt) = 1/γ * u₁ - 1/γ * (1-γ)^2 * uₙ
		# Manual derivation (devision by (2-γ) and then move everything to one side.)
		# a₁ = -((1 - γ)^2) / (γ * (2 - γ))
		# a₂ = 1 / (γ * (2 - γ))
		# a₃ = - (1 - γ) / (2 - γ)
		# res .= a₁ .* uₙ .+ a₂ .* u₁ .+  a₃ .* Δt .* du .- u

		# after Bonaventura2021
		# They define the second stage as:
		# u - γ₂ * Δt * f(u, t+Δt) = (1-γ₃)uₙ + γ₃u₁
		# Which differs from Bank1985)
		# (2-γ)u + (1-γ)Δt * f(u, t+Δt) = 1/γ * u₁ - 1/γ * (1-γ)^2 * uₙ
		# In the sign of u - γ₂ * Δt
		# a₁ == (1-γ₃)
		# a₂ == γ₃
		# a₃ == -γ₂
		γ₂ = (1 - γ) / (2 - γ)
		γ₃ = 1 / (γ * (2 - γ))

		res .= (1 - γ₃) .* uₙ .+ γ₃ .* u₁ + (γ₂ * Δt) .* du .- u
	end
end

abstract type DIRK{N} <: SimpleImplicitAlgorithm{N} end

struct RKImplicitEuler <: DIRK{1} end

function (::RKImplicitEuler)(res, uₙ, Δt, f!, du, u, p, t, stages, stage, RK)
	if stage == 1
		# Stage 1: 
		f!(du, u, p, t + RK.c[stage] * Δt)
	 	return	res .= u .- uₙ .- RK.a[stage, stage] * Δt .* du
	else
		@. u = uₙ + RK.b[1] * Δt * stages[1]
	end

end

struct KS2 <: DIRK{2} end
struct QZ2 <: DIRK{2} end
struct Crouzeix <: DIRK{2} end

function (::DIRK{2})(res, uₙ, Δt, f!, du, u, p, t, stages, stage, RK)
	if stage == 1
		f!(du, u, p, t + RK.c[stage] * Δt)
	 	return	res .= u .- uₙ .- RK.a[stage, stage] * Δt .* du
	elseif stage == 2
		f!(du, u, p, t + RK.c[stage] * Δt)
	 	return	res .= u .- uₙ .- RK.a[stage, 1] * Δt .* stages[1] - RK.a[stage,2] * Δt .* du
		else
		@. u = uₙ +  Δt * (RK.b[1]  * stages[1] + RK.b[2] * stages[2])
	end

end

struct Rosenbrock <: Direct{3} end

function (::Rosenbrock)(res, uₙ, Δt, f!, du, u, p, t, stages, stage, workspace, M, RK)
	invdt = inv(Δt)
	@. u = uₙ
	@. res = 0
	for j in 1:(stage-1)
		@. u = u + RK.a[stage, j] * stages[j]
		@. res = res + RK.c[stage, j] * stages[j] * invdt
	end

	## It does not work for non-autonomous systems.
	f!(du, u, p, t + Δt)

	krylov_solve!(workspace, M, copy(du .+ res))
	stages[stage] .= workspace.x

	if stage == 3
		@. u = uₙ
		for j in 1:stage
			@. u = u + RK.m[j] * stages[j]
		end
	end

end
abstract type RKTableau end

struct RosenbrockButcher{T1 <: AbstractArray, T2 <: AbstractArray} <: RKTableau
	a::T1
	c::T1
	m::T2
end

struct DIRKButcher{T1 <: AbstractArray, T2 <: AbstractArray} <: RKTableau
	a::T1
	b::T2
	c::T2
end

function RosenbrockTableau()

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
	return RosenbrockButcher(a, c, vec(m))

end

function ImplicitEulerTableau()

	nstage = 1
	a = zeros(Float64, nstage, nstage)
	a[1,1] = 1 

	b = zeros(Float64, nstage)
	b[1] = 1

	c = zeros(Float64, nstage)
	c[1] = 1
	return DIRKButcher(a,b,c)
end

# Kraaijevanger and Spijker's two-stage Diagonally Implicit Runge–Kutta method: 
function KS2Tableau()
		nstage = 2
	a = zeros(Float64, nstage, nstage)
	a[1,1] = 1/2 
	a[2,1] = -1/2
	a[2,2] = 2
	b = zeros(Float64, nstage)
	b[1] = -1/2
	b[2] = 3/2

	c = zeros(Float64, nstage)
	c[1] = 1/2
	c[2] = 3/2
	return DIRKButcher(a,b,c)

end
# Qin and Zhang's two-stage, 2nd order, symplectic Diagonally Implicit Runge–Kutta method: 
function QZ2Tableau()
	nstage = 2
	a = zeros(Float64, nstage, nstage)
	a[1,1] = 1/4
	a[2,1] = 1/2
	a[2,2] = 1/4
	b = zeros(Float64, nstage)
	b[1] = 1/2
	b[2] = 1/2

	c = zeros(Float64, nstage)
	c[1] = 1/4
	c[2] = 3/4
	return DIRKButcher(a,b,c)
end

# Crouzeix's two-stage, 3rd order Diagonally Implicit Runge–Kutta method
function CrouzeixTableau()
	nstage = 2
	a = zeros(Float64, nstage, nstage)
	a[1,1] = 1/2 + sqrt(3)/6
	a[2,1] = -sqrt(3)/3
	a[2,2] = 1/2 + sqrt(3)/6
	b = zeros(Float64, nstage)
	b[1] = 1/2
	b[2] = 1/2

	c = zeros(Float64, nstage)
	c[1] = 1/2 + sqrt(3)/6
	c[2] = 1/2 - sqrt(3)/6
	return DIRKButcher(a,b,c)
end


function RKTableau(alg::Direct)
	return RosenbrockTableau()
end

function RKTableau(alg::NonDirect)
	return RosenbrockTableau()
end

function RKTableau(alg::RKImplicitEuler)
	return ImplicitEulerTableau()
end

function RKTableau(alg::KS2)
	return KS2Tableau()
end

function RKTableau(alg::QZ2)
	return QZ2Tableau()
end

function RKTableau(alg::Crouzeix)
	return CrouzeixTableau()
end

function nonlinear_problem(alg::SimpleImplicitAlgorithm, f::F) where {F}
	return (res, u, (uₙ, Δt, du, p, t, stages, stage)) -> alg(res, uₙ, Δt, f, du, u, p, t, stages, stage)
end

function nonlinear_problem(alg::DIRK, f::F) where {F}
	return (res, u, (uₙ, Δt, du, p, t, stages, stage, RK)) -> alg(res, uₙ, Δt, f, du, u, p, t, stages, stage, RK)
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L1
mutable struct SimpleImplicitOptions{Callback}
	callback::Callback # callbacks; used in Trixi.jl
	adaptive::Bool # whether the algorithm is adaptive; ignored
	dtmax::Float64 # ignored
	maxiters::Int # maximal number of time steps
	tstops::Vector{Float64} # tstops from https://diffeq.sciml.ai/v6.8/basics/common_solver_opts/#Output-Control-1; ignored
	verbose::Int
	algo::Symbol
	krylov_kwargs::Any
end


function SimpleImplicitOptions(callback, tspan; maxiters = typemax(Int), verbose = 0, krylov_algo = :gmres, krylov_kwargs = (;), kwargs...)
	return SimpleImplicitOptions{typeof(callback)}(
		callback, false, Inf, maxiters,
		[last(tspan)],
		verbose,
		krylov_algo,
		krylov_kwargs,
	)
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.jl.
mutable struct SimpleImplicit{
	RealT <: Real, uType, Params, Sol, F, M, Alg <: SimpleImplicitAlgorithm,
	SimpleImplicitOptions, RKTableau,
} <: AbstractTimeIntegrator
	u::uType
	du::uType
	u_tmp::uType
	stages::NTuple{M, uType}
	res::uType
	t::RealT
	dt::RealT # current time step
	dtcache::RealT # ignored
	iter::Int # current number of time steps (iteration)
	p::Params # will be the semidiscretization from Trixi.jl
	sol::Sol # faked
	f::F # `rhs!` of the semidiscretization
	alg::Alg # SimpleImplicitAlgorithm
	opts::SimpleImplicitOptions
	finalstep::Bool # added for convenience
	RK::RKTableau
end

# Forward integrator.stats.naccept to integrator.iter (see GitHub PR#771)
function Base.getproperty(integrator::SimpleImplicit, field::Symbol)
	if field === :stats
		return (naccept = getfield(integrator, :iter),)
	end
	# general fallback
	return getfield(integrator, field)
end

function init(
	ode::ODEProblem, alg::SimpleImplicitAlgorithm{N};
	dt, callback::Union{CallbackSet, Nothing} = nothing, kwargs...,
) where {N}
	u = copy(ode.u0)
	du = zero(u)
	res = zero(u)
	u_tmp = similar(u)
	stages = ntuple(_ -> similar(u), Val(N))
	t = first(ode.tspan)
	iter = 0
	integrator = SimpleImplicit(
		u, du, u_tmp, stages, res, t, dt, zero(dt), iter, ode.p,
		(prob = ode,), ode.f, alg,
		SimpleImplicitOptions(
			callback, ode.tspan;
			kwargs...,
		), false, RKTableau(alg))

	# initialize callbacks
	if callback isa CallbackSet
		foreach(callback.continuous_callbacks) do cb
			throw(ArgumentError("Continuous callbacks are unsupported with the implicit time integration methods."))
		end
		foreach(callback.discrete_callbacks) do cb
			cb.initialize(cb, integrator.u, integrator.t, integrator)
		end
	end

	return integrator
end

# Fakes `solve`: https://diffeq.sciml.ai/v6.8/basics/overview/#Solving-the-Problems-1
function solve(
	ode::ODEProblem, alg::SimpleImplicitAlgorithm;
	dt, callback = nothing, kwargs...,
)
	integrator = init(ode, alg, dt = dt, callback = callback; kwargs...)

	# Start actual solve
	return solve!(integrator)
end

function solve!(integrator::SimpleImplicit)
	@unpack prob = integrator.sol

	integrator.finalstep = false

	while !integrator.finalstep
		step!(integrator)
	end # "main loop" timer

	finalize_callbacks(integrator)

	return TimeIntegratorSolution(
		(first(prob.tspan), integrator.t),
		(prob.u0, integrator.u),
		integrator.sol.prob,
	)
end

function stage!(integrator, alg::NonDirect)
	for stage in 1:stages(alg)
		F! = nonlinear_problem(alg, integrator.f)
		# TODO: Pass in `stages[1:(stage-1)]` or full tuple?
		_, stats = Ariadne.newton_krylov!(
			F!, integrator.u_tmp, (integrator.u, integrator.dt, integrator.du, integrator.p, integrator.t, integrator.stages, stage), integrator.res;
			verbose = integrator.opts.verbose, krylov_kwargs = integrator.opts.krylov_kwargs,
			algo = integrator.opts.algo, tol_abs = 6.0e-6,
		)
		@assert stats.solved
		if stage < stages(alg)
			# Store the solution for each stage in stages
			integrator.stages[stage] .= integrator.u_tmp
		end
	end
end

function stage!(integrator, alg::DIRK)
	for stage in 1:stages(alg)
		F! = nonlinear_problem(alg, integrator.f)
		# TODO: Pass in `stages[1:(stage-1)]` or full tuple?
		_, stats = Ariadne.newton_krylov!(
			F!, integrator.u_tmp, (integrator.u, integrator.dt, integrator.du, integrator.p, integrator.t, integrator.stages, stage, integrator.RK), integrator.res;
			verbose = integrator.opts.verbose, krylov_kwargs = integrator.opts.krylov_kwargs,
			algo = integrator.opts.algo, tol_abs = 6.0e-6,
		)
		@assert stats.solved

			# Store the solution for each stage in stages
			integrator.f(integrator.du, integrator.u_tmp, integrator.p, integrator.t + integrator.RK.c[stage] * integrator.dt)
			integrator.stages[stage] .= integrator.du
		if stage == stages(alg)	
			alg(integrator.res, integrator.u, integrator.dt, integrator.f, integrator.du, integrator.u_tmp, integrator.p, integrator.t, integrator.stages, stage+1, integrator.RK)
		end
		
	end
end

function stage!(integrator, alg::Direct)

	F!(du, u, p) = integrator.f(du, u, p, integrator.t)
	J = JacobianOperator(F!, integrator.du, integrator.u, integrator.p)
	M = MOperator(J, integrator.dt)
	kc = KrylovConstructor(integrator.res)
	workspace = krylov_workspace(:gmres, kc)

	for stage in 1:stages(alg)
		alg(integrator.res, integrator.u, integrator.dt, integrator.f, integrator.du, integrator.u_tmp, integrator.p, integrator.t, integrator.stages, stage, workspace, M, integrator.RK)
	end

end

function step!(integrator::SimpleImplicit)
	@unpack prob = integrator.sol
	@unpack alg = integrator
	t_end = last(prob.tspan)
	callbacks = integrator.opts.callback

	@assert !integrator.finalstep
	if isnan(integrator.dt)
		error("time step size `dt` is NaN")
	end

	# if the next iteration would push the simulation beyond the end time, set dt accordingly
	if integrator.t + integrator.dt > t_end ||
	   isapprox(integrator.t + integrator.dt, t_end)
		integrator.dt = t_end - integrator.t
		terminate!(integrator)
	end

	# one time step
	integrator.u_tmp .= integrator.u

	stage!(integrator, alg)

	integrator.u .= integrator.u_tmp

	integrator.iter += 1
	integrator.t += integrator.dt

	begin
		# handle callbacks
		if callbacks isa CallbackSet
			foreach(callbacks.discrete_callbacks) do cb
				if cb.condition(integrator.u, integrator.t, integrator)
					cb.affect!(integrator)
				end
				return nothing
			end
		end
	end

	# respect maximum number of iterations
	return if integrator.iter >= integrator.opts.maxiters && !integrator.finalstep
		@warn "Interrupted. Larger maxiters is needed."
		terminate!(integrator)
	end
end

# get a cache where the RHS can be stored
get_du(integrator::SimpleImplicit) = integrator.du
get_tmp_cache(integrator::SimpleImplicit) = (integrator.u_tmp,)

# some algorithms from DiffEq like FSAL-ones need to be informed when a callback has modified u
u_modified!(integrator::SimpleImplicit, ::Bool) = false

# used by adaptive timestepping algorithms in DiffEq
function set_proposed_dt!(integrator::SimpleImplicit, dt)
	return integrator.dt = dt
end

# Required e.g. for `glm_speed_callback`
function get_proposed_dt(integrator::SimpleImplicit)
	return integrator.dt
end

# stop the time integration
function terminate!(integrator::SimpleImplicit)
	integrator.finalstep = true
	return empty!(integrator.opts.tstops)
end

# used for AMR
function Base.resize!(integrator::SimpleImplicit, new_size)
	resize!(integrator.u, new_size)
	resize!(integrator.du, new_size)
	return resize!(integrator.u_tmp, new_size)
end

### Helper
jacobian(G!, ode::ODEProblem, Δt) = jacobian(G!, ode.f, ode.u0, ode.p, Δt, first(ode.tspan))

function jacobian(G!, f!, uₙ, p, Δt, t)
	u = copy(uₙ)
	du = zero(uₙ)
	res = zero(uₙ)

	F! = nonlinear_problem(G!, f!)

	J = Ariadne.JacobianOperator(F!, res, u, (uₙ, Δt, du, p, t))
	return collect(J)
end

end # module Implicit
