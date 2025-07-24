abstract type SimpleLinearImplicitExplicitAlgorithm{N} end

abstract type RKLIMEX{N} <: SimpleLinearImplicitExplicitAlgorithm{N} end


struct IMEXRKButcher{T1<:AbstractArray,T2<:AbstractArray} <: RKTableau
    a::T1
    b::T2
    c::T2
    ah::T1
    bh::T2
    ch::T2	
end

struct RKLinearImplicitExplicitEuler <: RKLIMEX{1} end

struct RKLSSPIMEX332 <: RKLIMEX{3} end

function mul!(out::AbstractVector, M::LMOperator, v::AbstractVector)
    # out = (I/dt - J(f,x,p)) * v
    mul!(out, M.J, v)
    @. out = v - out * M.dt
    return nothing
end

function (::RKLinearImplicitExplicitEuler)(res, uₙ, Δt, f1!, f2!, du, du_tmp, u, p, t, stages, ustages, jstages, stage, RK, M, lin_du_tmp, lin_du_tmp1, workspace)
    if stage == 1
        # Stage 1:
	## f2 is the conservative part
	## f1 is the parabolic part
	mul!(lin_du_tmp, M.J, uₙ)
#	mul!(lin_du_tmp1, J, u)	
	f2!(du, uₙ, p, t + RK.c[stage] * Δt)
        f1!(du_tmp, uₙ, p, t + RK.c[stage] * Δt)
	
	res .= uₙ .+ RK.a[stage, stage] * Δt .* (du .+ du_tmp .- lin_du_tmp)
	krylov_solve!(workspace, M, res, atol = 1e-6, rtol = 1e-6)
	@. u = workspace.x
#	f2!(du, workspace.x, p, t + RK.c[stage] * Δt)
#       f1!(du_tmp, workspace.x, p, t + RK.c[stage] * Δt)
#    	stages[stage] .= du .+ du_tmp  
#        @. u = uₙ + RK.b[1] * Δt * stages[1]
    end
end


function (::RKLIMEX{3})(res, uₙ, Δt, f1!, f2!, du, du_tmp, u, p, t, stages, ustages, jstages, stage, RK, M, lin_du_tmp, lin_du_tmp1, workspace)
	F!(du, u, p) = f1!(du, u, p, t) ## parabolic
    if stage == 1
        # Stage 1:
	## f2 is the conservative part
	## f1 is the parabolic part
   	J = JacobianOperator(F!, du, uₙ, p)
   	M = LMOperator(J, RK.ah[stage,stage] * Δt)
	krylov_solve!(workspace, M, uₙ, atol = 1e-6, rtol = 1e-6)
	@. u = workspace.x
   	J = JacobianOperator(F!, du, u, p)
	mul!(jstages[stage], J, u)
	
	f2!(du, workspace.x, p, t + RK.c[stage] * Δt)
    f1!(du_tmp, workspace.x, p, t + RK.c[stage] * Δt)
	@. stages[stage] = du + du_tmp - jstages[stage]
	@. ustages[stage] = u	
#        @. u = uₙ + RK.b[1] * Δt * stages[1]
	elseif stage == 2
	J = JacobianOperator(F!, du, ustages[1], p)
	M = LMOperator(J, RK.ah[stage,stage] * Δt)
	mul!(lin_du_tmp, M.J, uₙ)
#	mul!(lin_du_tmp1, J, u)	
	@. res = uₙ + RK.a[stage,1] * Δt * stages[1] - Δt * RK.ah[stage,1] * jstages[1]  

	krylov_solve!(workspace, M, res, atol = 1e-6, rtol = 1e-6)
	@. u = workspace.x
   	J = JacobianOperator(F!, du, u, p)
	mul!(jstages[stage], J, u)	
	f2!(du, workspace.x, p, t + RK.c[stage] * Δt)
	f1!(du_tmp, workspace.x, p, t + RK.c[stage] * Δt)
	@. stages[stage] = du + du_tmp - jstages[stage]
	@. ustages[stage] = u	

	elseif stage == 3
	J = JacobianOperator(F!, du, ustages[2], p)
	M = LMOperator(J, RK.ah[stage,stage] * Δt)
	
	mul!(lin_du_tmp, M.J, uₙ)
	@. res = uₙ + RK.a[stage,1] * Δt * stages[1] + RK.a[stage,2] * Δt * stages[2] - Δt * RK.ah[stage,1] * jstages[1] - Δt * RK.ah[stage,2] * jstages[2]  

	krylov_solve!(workspace, M, res, atol = 1e-6, rtol = 1e-6)
	@. u = workspace.x
   	J = JacobianOperator(F!, du, u, p)

	mul!(jstages[stage], J, u)	
	f2!(du, workspace.x, p, t + RK.c[stage] * Δt)
    f1!(du_tmp, workspace.x, p, t + RK.c[stage] * Δt)
@.	stages[stage] = du .+ du_tmp .- jstages[stage]
@.	ustages[stage] = u	

@. u = uₙ + RK.b[1] * Δt * stages[1] + RK.b[2] * Δt * stages[2] + RK.b[3] * Δt * stages[3] -  RK.bh[1] * Δt * jstages[1]  -  RK.bh[2] * Δt * jstages[2]  -  RK.bh[3] * Δt * jstages[3]  
	end
end

stages(::SimpleLinearImplicitExplicitAlgorithm{N}) where {N} = N

function nonlinear_problem(alg::SimpleLinearImplicitExplicitAlgorithm, f2::F2) where {F2}
    return (res, u, (uₙ, Δt, f1, du, du_tmp, p, t, stages, stage, RK, J, lin_du_tmp, lin_du_tmp1)) -> alg(res, uₙ, Δt, f1, f2, du, du_tmp, u, p, t, stages, stage, RK, J, lin_du_tmp, lin_du_tmp1)
end

mutable struct SimpleLinearImplicitExplicitOptions{Callback}
    callback::Callback # callbacks; used in Trixi.jl
    adaptive::Bool # whether the algorithm is adaptive; ignored
    dtmax::Float64 # ignored
    maxiters::Int # maximal number of time steps
    tstops::Vector{Float64} # tstops from https://diffeq.sciml.ai/v6.8/basics/common_solver_opts/#Output-Control-1; ignored
    verbose::Int
    algo::Symbol
    krylov_kwargs::Any
end

function RKTableau(alg::RKLSSPIMEX332)
return RKLSSPIMEX332Tableau()
end

function RKLSSPIMEX332Tableau()

    nstage = 3
    a = zeros(Float64, nstage, nstage)
    a[2, 1] = 0.5
    a[3, 1] = 0.5
    a[3, 2] = 0.5
    
    b = zeros(Float64, nstage)
    b[1] = 1/3
    b[2] = 1/3
    b[3] = 1/3	 

    c = zeros(Float64, nstage)
    c[2] = 0.5
    c[3] = 1.0
    ah = zeros(Float64, nstage, nstage)
    ah[1, 1] = 1/4
    ah[2, 2] = 1/4
    ah[3, 1] = 1/3
    ah[3, 2] = 1/3
    ah[3, 3] = 1/3
    
    bh = zeros(Float64, nstage)
    bh[1] = 1/3
    bh[2] = 1/3
    bh[3] = 1/3	 

    ch = zeros(Float64, nstage)
	ch[1] = 1/4
    ch[2] = 1/4
    ch[3] = 1.0
    return IMEXRKButcher(a, b, c, ah, bh, ch)
end

function RKTableau(alg::RKLinearImplicitExplicitEuler)
    return LinearImplicitExplicitEulerTableau()
end

function LinearImplicitExplicitEulerTableau()

    nstage = 1
    a = zeros(Float64, nstage, nstage)
    a[1, 1] = 1

    b = zeros(Float64, nstage)
    b[1] = 1

    c = zeros(Float64, nstage)
    c[1] = 1
    return DIRKButcher(a, b, c)
end


function SimpleLinearImplicitExplicitOptions(callback, tspan; maxiters=typemax(Int), verbose=0, krylov_algo=:gmres, krylov_kwargs=(;), kwargs...)
    return SimpleLinearImplicitExplicitOptions{typeof(callback)}(
        callback, false, Inf, maxiters,
        [last(tspan)],
        verbose,
        krylov_algo,
        krylov_kwargs,
    )
end

mutable struct SimpleLinearImplicitExplicit{
    RealT<:Real,uType,Params,Sol,F,F1,F2,M,Alg<:SimpleLinearImplicitExplicitAlgorithm,
    SimpleLinearImplicitExplicitOptions,RKTableau,
} <: AbstractTimeIntegrator
    u::uType
    du::uType
    du_tmp::uType
    lin_du_tmp::uType
    lin_du_tmp1::uType
    u_tmp::uType
    stages::NTuple{M,uType}
    ustages::NTuple{M,uType}
    jstages::NTuple{M,uType}
    res::uType
    t::RealT
    dt::RealT # current time step
    dtcache::RealT # ignored
    iter::Int # current number of time steps (iteration)
    p::Params # will be the semidiscretization from Trixi.jl
    sol::Sol # faked
    f::F #TODO: that should be sum of f1 and f2
    f1::F1 # `rhs!` parabolic
    f2::F2 # rhs! conservative
    alg::Alg # SimpleImplicitAlgorithm
    opts::SimpleLinearImplicitExplicitOptions
    finalstep::Bool # added for convenience
    RK::RKTableau
end


function Base.getproperty(integrator::SimpleLinearImplicitExplicit, field::Symbol)
    if field === :stats
        return (naccept=getfield(integrator, :iter),)
    end
    # general fallback
    return getfield(integrator, field)
end

function init(
    ode::ODEProblem, alg::SimpleLinearImplicitExplicitAlgorithm{N};
    dt, callback::Union{CallbackSet,Nothing}=nothing, kwargs...,
) where {N}
    u = copy(ode.u0)
    du = zero(u)
    res = zero(u)
    u_tmp = similar(u)
    stages = ntuple(_ -> similar(u), Val(N))
    ustages = ntuple(_ -> similar(u), Val(N))
    jstages = ntuple(_ -> similar(u), Val(N))
    t = first(ode.tspan)
    iter = 0
    integrator = SimpleLinearImplicitExplicit(
		u, du, copy(du),copy(du), copy(du), u_tmp,stages, ustages, jstages, res, t, dt, zero(dt), iter, ode.p,
        (prob=ode,), ode.f.f1, ode.f.f1, ode.f.f2, alg,
        SimpleLinearImplicitExplicitOptions(
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
    ode::ODEProblem, alg::SimpleLinearImplicitExplicitAlgorithm;
    dt, callback=nothing, kwargs...,
)
    integrator = init(ode, alg, dt=dt, callback=callback; kwargs...)

    # Start actual solve
    return solve!(integrator)
end

function solve!(integrator::SimpleLinearImplicitExplicit)
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


function step!(integrator::SimpleLinearImplicitExplicit)
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

function stage!(integrator, alg::RKLIMEX)
   F!(du, u, p) = integrator.f1(du, u, p, integrator.t) ## parabolic
   J = JacobianOperator(F!, integrator.du, integrator.u, integrator.p)
   	M = LMOperator(J, integrator.dt)
    kc = KrylovConstructor(integrator.res)
    workspace = krylov_workspace(:gmres, kc)
	for stage in 1:stages(alg)
        # Store the solution for each stage in stages
	## For a split Problem we need to compute rhs_conservative and rhs_parabolic
		alg(integrator.res, integrator.u, integrator.dt, integrator.f1, integrator.f2, integrator.du, integrator.du_tmp, integrator.u_tmp, integrator.p, integrator.t, integrator.stages, integrator.ustages, integrator.jstages, stage, integrator.RK, M, integrator.lin_du_tmp, integrator.lin_du_tmp1, workspace)

    end
end

# get a cache where the RHS can be stored
get_du(integrator::SimpleLinearImplicitExplicit) = integrator.du
get_tmp_cache(integrator::SimpleLinearImplicitExplicit) = (integrator.u_tmp,)

# some algorithms from DiffEq like FSAL-ones need to be informed when a callback has modified u
u_modified!(integrator::SimpleLinearImplicitExplicit, ::Bool) = false

# used by adaptive timestepping algorithms in DiffEq
function set_proposed_dt!(integrator::SimpleLinearImplicitExplicit, dt)
    return integrator.dt = dt
end

# Required e.g. for `glm_speed_callback`
function get_proposed_dt(integrator::SimpleLinearImplicitExplicit)
    return integrator.dt
end

# stop the time integration
function terminate!(integrator::SimpleLinearImplicitExplicit)
    integrator.finalstep = true
    return empty!(integrator.opts.tstops)
end

# used for AMR
function Base.resize!(integrator::SimpleLinearImplicitExplicit, new_size)
    resize!(integrator.u, new_size)
    resize!(integrator.du, new_size)
    return resize!(integrator.u_tmp, new_size)
end

