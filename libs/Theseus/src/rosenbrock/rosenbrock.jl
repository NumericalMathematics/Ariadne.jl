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

abstract type RosenbrockAlgorithm{N} <: SimpleImplicitAlgorithm{N} end

stages(::RosenbrockAlgorithm{N}) where {N} = N

"""
	(::RosenbrockAlgorithm{N})(res, uₙ, Δt, f!, du, u, p, t, stages, stage, workspace, RK, assume_p_const) where N

JVPs and matrix free implementation of Rosenbrock methods.
Note this is Rosenbrock-W method, as the jacobian is "inexact".
References:
- E. Hairer and G. Wanner (1996)
Solving Ordinary Differential Equations II, pag. 111 (Implementation of Rosenbrock-Type Methods)
"""

function (::RosenbrockAlgorithm{N})(res, uₙ, Δt, f!, du, u, p, t, stages, stage, workspace, RK, assume_p_const, atol, rtol) where N
	invdt = inv(Δt)
	F!(du, u, p) = f!(du, u, p, t)
	@. u = uₙ
	@. res = 0
	J = JacobianOperator(F!, du, uₙ, p, assume_p_const = assume_p_const)
	M = MOperator(J, RK.gamma[stage] * Δt)

	for j in 1:(stage-1)
		@. u = u + RK.a[stage, j] * stages[j]
		@. res = res + RK.c[stage, j] * stages[j] * invdt
	end

	## It does not work for non-autonomous systems.
	f!(du, u, p, t + Δt)
	@. res = res + du
	krylov_solve!(workspace, M, res, atol = atol, rtol = rtol)
	stages[stage] .= workspace.x

	if stage == N
		@. u = uₙ
		for j in 1:stage
			@. u = u + RK.m[j] * stages[j]
		end
	end

end

mutable struct RosenbrockOptions{Callback}
	callback::Callback # callbacks; used in Trixi.jl
	adaptive::Bool # whether the algorithm is adaptive; ignored
	dtmax::Float64 # ignored
	maxiters::Int # maximal number of time steps
	tstops::Vector{Float64} # tstops from https://diffeq.sciml.ai/v6.8/basics/common_solver_opts/#Output-Control-1; ignored
	verbose::Int
	algo::Symbol
	assume_p_const::Bool
	krylov_atol::Float64
	krylov_rtol::Float64
	krylov_kwargs::Any
end


function RosenbrockOptions(callback, tspan; maxiters = typemax(Int), verbose = 0, krylov_algo = :gmres, assume_p_const = true, krylov_atol = 1e-6, krylov_rtol = 1e-6, krylov_kwargs = (;), kwargs...)
	return RosenbrockOptions{typeof(callback)}(
		callback, false, Inf, maxiters,
		[last(tspan)],
		verbose,
		krylov_algo,
		assume_p_const,
		krylov_atol,
		krylov_rtol,
		krylov_kwargs,
	)
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.jl.
mutable struct Rosenbrock{
	RealT <: Real, uType, Params, Sol, F, M, Alg <: RosenbrockAlgorithm,
	RosenbrockOptions, RKTableau,
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
	alg::Alg # RosenbrockAlgorithm
	opts::RosenbrockOptions
	finalstep::Bool # added for convenience
	RK::RKTableau
end

# Forward integrator.stats.naccept to integrator.iter (see GitHub PR#771)
function Base.getproperty(integrator::Rosenbrock, field::Symbol)
	if field === :stats
		return (naccept = getfield(integrator, :iter),)
	end
	# general fallback
	return getfield(integrator, field)
end

function init(
	ode::ODEProblem, alg::RosenbrockAlgorithm{N};
	dt, callback::Union{CallbackSet, Nothing} = nothing, kwargs...,
) where {N}
	u = copy(ode.u0)
	du = zero(u)
	res = zero(u)
	u_tmp = similar(u)
	stages = ntuple(_ -> similar(u), Val(N))
	t = first(ode.tspan)
	iter = 0
	integrator = Rosenbrock(
		u, du, u_tmp, stages, res, t, dt, zero(dt), iter, ode.p,
		(prob = ode,), ode.f, alg,
		RosenbrockOptions(
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
	ode::ODEProblem, alg::RosenbrockAlgorithm;
	dt, callback = nothing, kwargs...,
)
	integrator = init(ode, alg, dt = dt, callback = callback; kwargs...)

	# Start actual solve
	return solve!(integrator)
end

function solve!(integrator::Rosenbrock)
	@unpack prob = integrator.sol

	integrator.finalstep = false

	kc = KrylovConstructor(integrator.res)
	workspace = krylov_workspace(integrator.opts.algo, kc)

	while !integrator.finalstep
		step!(integrator, workspace)
	end # "main loop" timer

	finalize_callbacks(integrator)

	return TimeIntegratorSolution(
		(first(prob.tspan), integrator.t),
		(prob.u0, integrator.u),
		integrator.sol.prob,
	)
end

function stage!(integrator, alg::RosenbrockAlgorithm, workspace)

	for stage in 1:stages(alg)
		alg(integrator.res, integrator.u, integrator.dt, integrator.f, integrator.du, integrator.u_tmp, integrator.p, integrator.t,
			integrator.stages, stage, workspace, integrator.RK, integrator.opts.assume_p_const, integrator.opts.krylov_atol, integrator.opts.krylov_rtol)
	end

end

function step!(integrator::Rosenbrock, workspace)
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

	stage!(integrator, alg, workspace)

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
get_du(integrator::Rosenbrock) = integrator.du
get_tmp_cache(integrator::Rosenbrock) = (integrator.u_tmp,)

# some algorithms from DiffEq like FSAL-ones need to be informed when a callback has modified u
u_modified!(integrator::Rosenbrock, ::Bool) = false

# used by adaptive timestepping algorithms in DiffEq
function set_proposed_dt!(integrator::Rosenbrock, dt)
	return integrator.dt = dt
end

# Required e.g. for `glm_speed_callback`
function get_proposed_dt(integrator::Rosenbrock)
	return integrator.dt
end

# stop the time integration
function terminate!(integrator::Rosenbrock)
	integrator.finalstep = true
	return empty!(integrator.opts.tstops)
end

# used for AMR
function Base.resize!(integrator::Rosenbrock, new_size)
	resize!(integrator.u, new_size)
	resize!(integrator.du, new_size)
	return resize!(integrator.u_tmp, new_size)
end

include("tableau.jl")
