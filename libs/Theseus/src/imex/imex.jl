abstract type SimpleImplicitExplicitAlgorithm{N} end

abstract type RKIMEX{N} <: SimpleImplicitExplicitAlgorithm{N} end

stages(::RKIMEX{N}) where {N} = N

function (::RKIMEX{N})(res, uₙ, Δt, f1!, du, du_tmp, u, p, t, stages_ex, stages_im, stage, RK) where {N}
    @. res = u
    for j in 1:(stage - 1)
        @. res = res - RK.a_ex[stage, j] * stages_ex[j] - RK.a_im[stage, j] * stages_im[j]
    end
    @. du = u * Δt + uₙ
    f1!(du_tmp, du, p, t + RK.c_im[stage] * Δt)
    @. res = res - RK.a_im[stage, stage] * du_tmp
    return res
end

function nonlinear_problem(alg::SimpleImplicitExplicitAlgorithm, f1::F1) where {F1}
    return (res, u, (uₙ, Δt, du, du_tmp, p, t, stages, stages_im, stage, RK)) -> alg(res, uₙ, Δt, f1, du, du_tmp, u, p, t, stages, stages_im, stage, RK)
end

mutable struct SimpleImplicitExplicitOptions{Callback}
    callback::Callback # callbacks; used in Trixi.jl
    adaptive::Bool # whether the algorithm is adaptive; ignored
    dtmax::Float64 # ignored
    maxiters::Int # maximal number of time steps
    tstops::Vector{Float64} # tstops from https://diffeq.sciml.ai/v6.8/basics/common_solver_opts/#Output-Control-1; ignored
    verbose::Int
    algo::Symbol
    krylov_tol_abs::Float64
    krylov_kwargs::Any
end

function SimpleImplicitExplicitOptions(callback, tspan; maxiters = typemax(Int), verbose = 0, krylov_algo = :gmres, krylov_tol_abs = 1.0e-6, krylov_kwargs = (;), kwargs...)
    return SimpleImplicitExplicitOptions{typeof(callback)}(
        callback, false, Inf, maxiters,
        [last(tspan)],
        verbose,
        krylov_algo,
        krylov_tol_abs,
        krylov_kwargs,
    )
end

mutable struct SimpleImplicitExplicit{
        RealT <: Real, uType, Params, Sol, F, F1, F2, M, Alg <: SimpleImplicitExplicitAlgorithm,
        SimpleImplicitExplicitOptions, RKTableau,
    } <: AbstractTimeIntegrator
    u::uType
    du::uType
    du_tmp::uType
    u_tmp::uType
    stages::NTuple{M, uType}
    stages_im::NTuple{M, uType}
    res::uType
    t::RealT
    dt::RealT # current time step
    dtcache::RealT # ignored
    iter::Int # current number of time steps (iteration)
    p::Params # will be the semidiscretization from Trixi.jl
    sol::Sol # faked
    f::F # sum of f1 and f2 (stiff and non-stiff rhs)
    f1::F1 # stiff part (e.g., `rhs_parabolic!` in Trixi.jl)
    f2::F2 # non-stiff part (e.g., hyperbolic `rhs!` in Trixi.jl)
    alg::Alg # SimpleImplicitAlgorithm
    opts::SimpleImplicitExplicitOptions
    finalstep::Bool # added for convenience
    RK::RKTableau
end


function Base.getproperty(integrator::SimpleImplicitExplicit, field::Symbol)
    if field === :stats
        return (naccept = getfield(integrator, :iter),)
    end
    # general fallback
    return getfield(integrator, field)
end

function init(
        ode::ODEProblem, alg::SimpleImplicitExplicitAlgorithm{N};
        dt, callback::Union{CallbackSet, Nothing} = nothing, kwargs...,
    ) where {N}
    u = copy(ode.u0)
    du = zero(u)
    res = zero(u)
    u_tmp = similar(u)
    stages = ntuple(_ -> similar(u), Val(N))
    stages_im = ntuple(_ -> similar(u), Val(N))
    t = first(ode.tspan)
    iter = 0
    integrator = SimpleImplicitExplicit(
        u, du, copy(du), u_tmp, stages, stages_im, res, t, dt, zero(dt), iter, ode.p,
        (prob = ode,), ode.f,
        ode.f.f1, ode.f.f2, alg,
        SimpleImplicitExplicitOptions(
            callback, ode.tspan;
            kwargs...,
        ), false, RKTableau(alg, eltype(u))
    )

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
        ode::ODEProblem, alg::SimpleImplicitExplicitAlgorithm;
        dt, callback = nothing, kwargs...,
    )
    integrator = init(ode, alg, dt = dt, callback = callback; kwargs...)

    # Start actual solve
    return solve!(integrator)
end

function solve!(integrator::SimpleImplicitExplicit)
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


function step!(integrator::SimpleImplicitExplicit)
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


function stage!(integrator, alg::RKIMEX)
    @. integrator.u_tmp = 0
    for stage in 1:stages(alg)
        if iszero(integrator.RK.a_im[stage, stage])
            @. integrator.u_tmp = 0
            for j in 1:(stage - 1)
                @. integrator.u_tmp = integrator.u_tmp + integrator.RK.a_ex[stage, j] * integrator.stages[j] + integrator.RK.a_im[stage, j] * integrator.stages_im[j]
            end
        else
            F! = nonlinear_problem(alg, integrator.f1)
            # TODO: Pass in `stages[1:(stage-1)]` or full tuple?
            _, stats = newton_krylov!(
                F!, integrator.u_tmp, (integrator.u, integrator.dt, integrator.du, integrator.du_tmp, integrator.p, integrator.t, integrator.stages, integrator.stages_im, stage, integrator.RK), integrator.res;
                verbose = integrator.opts.verbose, krylov_kwargs = integrator.opts.krylov_kwargs,
                algo = integrator.opts.algo, tol_abs = integrator.opts.krylov_tol_abs
            )
            @assert stats.solved
        end
        # Store the solution for each stage in stages
        # For a split problem, we need to compute the stiff and non-stiff RHS.
        # We solve the non-linear problem in the z variable, thus u_tmp is z_i = (u_i - u_n) / dt
        @. integrator.res = integrator.u_tmp * integrator.dt + integrator.u
        integrator.f2(integrator.du, integrator.res, integrator.p, integrator.t + integrator.RK.c_ex[stage] * integrator.dt)
        integrator.stages[stage] .= integrator.du
        @. integrator.res = integrator.u_tmp
        if iszero(integrator.RK.a_im[stage, stage])
            @. integrator.du = integrator.u_tmp * integrator.dt + integrator.u
            integrator.f1(integrator.stages_im[stage], integrator.du, integrator.p, integrator.t + integrator.RK.c_im[stage] * integrator.dt)
        else
            # We avoid evaluating the stiff operator at the numerical solutions (due to the large Lipschitz constant)
            for j in 1:(stage - 1)
                @. integrator.res = integrator.res - integrator.RK.a_im[stage, j] * integrator.stages_im[j] - integrator.RK.a_ex[stage, j] * integrator.stages[j]
            end
            @. integrator.stages_im[stage] = integrator.res / integrator.RK.a_im[stage, stage]
        end
    end
    # Final update
    fill!(integrator.u_tmp, zero(eltype(integrator.u_tmp)))
    for j in 1:stages(alg)
        Δt_b_ex = integrator.dt * integrator.RK.b_ex[j]
        Δt_b_im = integrator.dt * integrator.RK.b_im[j]
        @. integrator.u_tmp = integrator.u_tmp + Δt_b_ex * integrator.stages[j] + Δt_b_im * integrator.stages_im[j]
    end
    @. integrator.u_tmp = integrator.u + integrator.u_tmp
    return
end

# get a cache where the RHS can be stored
get_du(integrator::SimpleImplicitExplicit) = integrator.du
get_tmp_cache(integrator::SimpleImplicitExplicit) = (integrator.u_tmp,)

# some algorithms from DiffEq like FSAL-ones need to be informed when a callback has modified u
u_modified!(integrator::SimpleImplicitExplicit, ::Bool) = false

# used by adaptive timestepping algorithms in DiffEq
function set_proposed_dt!(integrator::SimpleImplicitExplicit, dt)
    return integrator.dt = dt
end

# Required e.g. for `glm_speed_callback`
function get_proposed_dt(integrator::SimpleImplicitExplicit)
    return integrator.dt
end

# stop the time integration
function terminate!(integrator::SimpleImplicitExplicit)
    integrator.finalstep = true
    return empty!(integrator.opts.tstops)
end

# used for AMR
function Base.resize!(integrator::SimpleImplicitExplicit, new_size)
    resize!(integrator.u, new_size)
    resize!(integrator.du, new_size)
    return resize!(integrator.u_tmp, new_size)
end

include("tableau.jl")
