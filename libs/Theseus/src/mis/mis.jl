abstract type MISAlgorithm{N} end
abstract type MISSlowAlgorithm{N} <: MISAlgorithm{N} end
abstract type MISFastAlgorithm{N} <: MISAlgorithm{N} end

stages(::MISAlgorithm{N}) where {N} = N

"""
	(::MISAlgorithm{N})(res, uₙ, Δt, f!, du, u, p, t, stages, stage, workspace, RK, assume_p_const) where N

"""
function (::MISSlowAlgorithm{N})(un, Δt, fslow!, Zn0, dZn, Yn, du, p, t, stages, stage, RK, integrator_fast, dt_fast) where {N}
    invdt = inv(Δt)
	
    @. Zn0 = un

    @trixi_timeit timer() "init MIS" compute_initial_condition!(Zn0, Yn, un, RK.alfa, stage)
	
    	o = zero(eltype(un))
	@. dZn = o
	@trixi_timeit timer() "setting ODE MIS"	compute_slow_tendencies!(dZn, Yn, stages, un, RK.d, RK.gamma, RK.beta, invdt, stage)
	
	if stage == 1
		@. Yn[stage] = un
	else
	     @trixi_timeit timer() "solve fast ODE" solve_fastode!(integrator_fast.alg, integrator_fast, Zn0, dZn, Yn, stage, RK.d[stage] * Δt, dt_fast)
	end
		
	@trixi_timeit timer() "slow tendencies" fslow!(du, Yn[stage], p, t + Δt)
	stages[stage] .= du
end

function solve_fastode!(alg, integrator_fast, Zn0, dZn, Yn, stage, T, dt)
    reinit!(integrator_fast, Zn0; t0 = zero(eltype(Zn0)), tf = T)
    integrator_fast.dt = dt
    while integrator_fast.t < T
    step!(integrator_fast)
    end

    @. Yn[stage] = integrator_fast.u
end

 function compute_initial_condition!(Zn0, Yn, u, alfa, stage)
        for j in 1:(stage - 1)
		alfa_stage = alfa[stage,j]
     @. Zn0 = Zn0 + alfa_stage * (Yn[j] - u)
        end
    end
    
    function compute_slow_tendencies!(dZn, Yn, du, u, d, gamma, beta, invdt, stage)
        for j in 1:(stage - 1)
		gamma_j = gamma[stage,j]
		beta_j = beta[stage,j]
		inv_d_j = inv(d[stage])
      @. dZn = dZn + inv_d_j * (invdt * gamma_j * (Yn[j] - u) + beta_j * du[j])
        end
    end

mutable struct MISOptions{Callback, TStops}
    callback::Callback # callbacks; used in Trixi.jl
    adaptive::Bool # whether the algorithm is adaptive; ignored
    dtmax::Float64 # ignored
    maxiters::Int # maximal number of time steps
    verbose::Int
    tstops::TStops # tstops from https://diffeq.sciml.ai/v6.8/basics/common_solver_opts/#Output-Control-1; ignored
end

function MISOptions(callback, tspan; maxiters = typemax(Int), verbose = 0, kwargs...)
    tstops_internal = BinaryHeap{eltype(tspan)}(FasterForward())
    # We add last(tspan) to make sure that the time integration stops at the end time
    push!(tstops_internal, last(tspan))
    # We add 2 * last(tspan) because add_tstop!(integrator, t) is only called by DiffEqCallbacks.jl if tstops contains a time that is larger than t
    # (https://github.com/SciML/DiffEqCallbacks.jl/blob/025dfe99029bd0f30a2e027582744528eb92cd24/src/iterative_and_periodic.jl#L92)
    push!(tstops_internal, 2 * last(tspan))
    return MISOptions{typeof(callback), typeof(tstops_internal)}(callback, false,
                                                                        Inf, maxiters,
                                                                        verbose,
                                                                        tstops_internal)
end


# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.jl.
mutable struct MISSlow{RealT <: Real, uType, Params, Sol, F, M,
                          Alg <: MISSlowAlgorithm,
                          MISOptions, RKTableau, Thread, IntegratorType} <:
               AbstractTimeIntegrator
    u::uType
    du::uType
    u_tmp::uType
    stages::NTuple{M, uType}
    Zn0::uType
    dZn::uType
    Yn::NTuple{M, uType}
    t::RealT
    tdir::RealT # DIRection of time integration, i.e., if one marches forward or backward in time
    dt::RealT # current time step
    dtcache::RealT # ignored
    iter::Int # current number of time steps (iteration)
    p::Params # will be the semidiscretization from Trixi.jl
    sol::Sol # faked
    f::F # `rhs!` of the semidiscretization
    alg::Alg # MISAlgorithm
    opts::MISOptions
    finalstep::Bool # added for convenience
    RK::RKTableau
    const dtchangeable::Bool
    const force_stepfail::Bool
    thread::Thread
    integrator_fast::IntegratorType
    dt_fast::RealT
end

function init(ode::SplitODEProblem, alg::Tuple{MISSlowAlgorithm{N},FastAlg};
              dt, callback::Union{CallbackSet, Nothing} = nothing,
	      dt_fast, kwargs...,) where {N, FastAlg}
    u = copy(ode.u0)
    du = zero(u)
    u_tmp = similar(u)
    Zn0 = similar(u)
    dZn = similar(u)
    u_tmp = similar(u)
    stages = ntuple(_ -> similar(u), Val(N))
    Yn = ntuple(_ -> similar(u), Val(N))
    t = first(ode.tspan)
    iter = 0
    tdir = sign(ode.tspan[end] - ode.tspan[1])

alg_slow = alg[1]
alg_fast = alg[2]
integrator = (; dZn) 

integrator_fast = let
    corrected_ffast! = (du, u, p, t) -> begin
        ode.f.f1!(du, u, p, t)
        du .+= integrator.dZn
    end

    prob_fast = ODEProblem(corrected_ffast!, copy(u), (0.0, 1.0), ode.p)
    init(prob_fast, alg_fast; dt = dt_fast, kwargs...)
end
    integrator = MISSlow(u, du, u_tmp, stages, Zn0, dZn, Yn, t, tdir, dt, zero(dt), iter, ode.p,
                            (prob = ode,), ode.f.f2, alg_slow,
                            MISOptions(callback, ode.tspan;
                                              kwargs...,), false, RKTableau(alg_slow, eltype(u)),
                            false, false, thread, integrator_fast, dt_fast)
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
function solve(ode::SplitODEProblem,alg::Tuple{<:MISSlowAlgorithm,Any};
               dt, callback = nothing, 
               kwargs...,)
    integrator = init(ode, alg, dt = dt, callback = callback; kwargs...)

    # Start actual solve
    return solve!(integrator)
end

function solve!(integrator::MISSlow)
    @unpack prob = integrator.sol

    integrator.finalstep = false

    while !integrator.finalstep
        step!(integrator)
    end # "main loop" timer

    finalize_callbacks(integrator)

    return TimeIntegratorSolution((first(prob.tspan), integrator.t),
                                  (prob.u0, integrator.u),
                                  integrator.sol.prob)
end

function stage!(integrator, alg::MISSlowAlgorithm)

    for stage in 1:stages(alg)
        @trixi_timeit timer() "slow time integration" alg(integrator.u, integrator.dt,
                                               integrator.f, integrator.Zn0, integrator.dZn, integrator.Yn, integrator.du,
                                               integrator.p, integrator.t,
                                               integrator.stages, stage, integrator.RK,
                                               integrator.integrator_fast, integrator.dt_fast)
    end

 @. integrator.u_tmp = integrator.Yn[end]

    return
end

function step!(integrator::MISSlow)
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

# Forward integrator.stats.naccept to integrator.iter (see GitHub PR#771)
function Base.getproperty(integrator::MISSlow, field::Symbol)
    if field === :stats
        return (naccept = getfield(integrator, :iter),)
    end
    # general fallback
    return getfield(integrator, field)
end

# get a cache where the RHS can be stored
get_du(integrator::MISSlow) = integrator.du
get_tmp_cache(integrator::MISSlow) = (integrator.u_tmp,)

# some algorithms from DiffEq like FSAL-ones need to be informed when a callback has modified u
u_modified!(integrator::MISSlow, ::Bool) = false

# used by adaptive timestepping algorithms in DiffEq
function set_proposed_dt!(integrator::MISSlow, dt)
    return integrator.dt = dt
end

# Required e.g. for `glm_speed_callback`
function get_proposed_dt(integrator::MISSlow)
    return integrator.dt
end

# stop the time integration
function terminate!(integrator::MISSlow)
    integrator.finalstep = true
    return empty!(integrator.opts.tstops)
end

# used for AMR
function Base.resize!(integrator::MISSlow, new_size)
    resize!(integrator.u, new_size)
    resize!(integrator.du, new_size)
    return resize!(integrator.u_tmp, new_size)
end

include("tableau.jl")
