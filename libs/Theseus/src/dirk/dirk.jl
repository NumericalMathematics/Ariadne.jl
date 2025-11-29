abstract type SimpleDiagonallyImplicitAlgorithm{N} end

abstract type DIRK{N} <: SimpleDiagonallyImplicitAlgorithm{N} end

stages(::DIRK{N}) where {N} = N

# This method calculates the residual of a diagonally implicit Runge-Kutta method
#
#   u^{n+1} = u^n + dt \sum_{i=1}^s b_i f(y^i)
#
# with stages
#
#   y^i = u^n + dt \sum_{j=1}^i a_{ij} f(y^j),
#
# where f is the RHS of the ODE.
# To compute the stages y^i, we formulate the stage equations in terms of the
# update variables
#
#   z^i = (y^i - u^n) / dt,
#
# i.e., we have
#
#   y^i = u^n + dt * z^i.
#
# Thus, the stage equations become
#
#   z^i - \sum_{j=1}^i a_{ij} f(y^j) = 0.
#
# In the implementation below, `u = z` is the unknown for the current `stage`.
function (::DIRK{N})(res, uₙ, Δt, f!, du, du_tmp, u, p, t, stages, stage, RK) where {N}
    @. res = u
    for j in 1:(stage - 1)
        @. res = res - RK.a[stage, j] * stages[j]
    end
    @. du = u * Δt + uₙ
    f!(du_tmp, du, p, t + RK.c[stage] * Δt)
    @. res = res - RK.a[stage, stage] * du_tmp
    return res
end

function nonlinear_problem(alg::SimpleDiagonallyImplicitAlgorithm, f::F) where {F}
    return (res, u, (uₙ, Δt, du, du_tmp, p, t, stages, stage, RK)) -> alg(res, uₙ, Δt, f, du, du_tmp, u, p, t, stages, stage, RK)
end

mutable struct SimpleDiagonallyImplicitOptions{Callback}
    callback::Callback # callbacks; used in Trixi.jl
    adaptive::Bool # whether the algorithm is adaptive; ignored
    dtmax::Float64 # ignored
    maxiters::Int # maximal number of time steps
    tstops::Vector{Float64} # tstops from https://diffeq.sciml.ai/v6.8/basics/common_solver_opts/#Output-Control-1; ignored
    verbose::Int
    newton_tol_abs::Float64
    newton_tol_rel::Float64
    newton_max_niter::Int
    krylov_algo::Symbol
    krylov_kwargs::Any
end

function SimpleDiagonallyImplicitOptions(callback, tspan;
        maxiters = typemax(Int),
        verbose = 0,
        newton_tol_abs = 1.0e-6,
        newton_tol_rel = 1.0e-6,
        newton_max_niter = 50,
        krylov_algo = :gmres,
        krylov_kwargs = (;),
        kwargs...)
    return SimpleDiagonallyImplicitOptions{typeof(callback)}(
        callback, false, Inf, maxiters,
        [last(tspan)],
        verbose,
        newton_tol_abs,
        newton_tol_rel,
        newton_max_niter,
        krylov_algo,
        krylov_kwargs,
    )
end

mutable struct SimpleDiagonallyImplicit{
        RealT <: Real, uType, Params, Sol, F, M, Alg <: SimpleDiagonallyImplicitAlgorithm,
        SimpleDiagonallyImplicitOptions, RKTableau,
    } <: AbstractTimeIntegrator
    u::uType
    du::uType
    du_tmp::uType
    u_tmp::uType
    stages::NTuple{M, uType}
    res::uType
    t::RealT
    dt::RealT # current time step
    dtcache::RealT # ignored
    iter::Int # current number of time steps (iteration)
    p::Params # will be the semidiscretization from Trixi.jl
    sol::Sol # faked
    f::F # sum of f1 and f2 (stiff and non-stiff rhs)
    alg::Alg # SimpleImplicitAlgorithm
    opts::SimpleDiagonallyImplicitOptions
    finalstep::Bool # added for convenience
    RK::RKTableau
end


function Base.getproperty(integrator::SimpleDiagonallyImplicit, field::Symbol)
    if field === :stats
        return (naccept = getfield(integrator, :iter),)
    end
    # general fallback
    return getfield(integrator, field)
end

function init(
        ode::ODEProblem, alg::SimpleDiagonallyImplicitAlgorithm{N};
        dt, callback::Union{CallbackSet, Nothing} = nothing, kwargs...,
    ) where {N}
    u = copy(ode.u0)
    du = zero(u)
    res = zero(u)
    u_tmp = similar(u)
    stages = ntuple(_ -> similar(u), Val(N))
    t = first(ode.tspan)
    iter = 0
    integrator = SimpleDiagonallyImplicit(
        u, du, copy(du), u_tmp, stages, res, t, dt, zero(dt), iter, ode.p,
        (prob = ode,), ode.f,
        alg,
        SimpleDiagonallyImplicitOptions(
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
        ode::ODEProblem, alg::SimpleDiagonallyImplicitAlgorithm;
        dt, callback = nothing, kwargs...,
    )
    integrator = init(ode, alg, dt = dt, callback = callback; kwargs...)

    # Start actual solve
    return solve!(integrator)
end

function solve!(integrator::SimpleDiagonallyImplicit)
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


function step!(integrator::SimpleDiagonallyImplicit)
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


# Compute all stages within one time step
function stage!(integrator, alg::DIRK)
    @. integrator.u_tmp = 0
    for stage in 1:stages(alg)
        # This computes all stages of a diagonally implicit Runge-Kutta method
        #
        #   u^{n+1} = u^n + dt \sum_{i=1}^s b_i f(y^i)
        #
        # with stages
        #
        #   y^i = u^n + dt \sum_{j=1}^i a_{ij} f(y^j),
        #
        # where f is the RHS of the ODE.
        # To compute the stages y^i, we formulate the stage equations in terms of the
        # update variables
        #
        #   z^i = (y^i - u^n) / dt,
        #
        # i.e., we have
        #
        #   y^i = u^n + dt * z^i.
        #
        # Thus, the stage equations become
        #
        #   z^i - \sum_{j=1}^i a_{ij} f(y^j) = 0.
        #
        # The variable `integrator.u_tmp` is the update z^i for the current `stage`.
        if iszero(integrator.RK.a[stage, stage])
            # In this case, the stage is explicit and can be computed directly
            # without solving any (nonlinear) system.
            @. integrator.u_tmp = 0
            for j in 1:(stage - 1)
                @. integrator.u_tmp = integrator.u_tmp + integrator.RK.a[stage, j] * integrator.stages[j]
            end
        else
            # In this case, we have an implicit stage that requires solving a
            # nonlinear system.
            F! = nonlinear_problem(alg, integrator.f)
            # TODO: Pass in `stages[1:(stage-1)]` or full tuple?
            _, stats = newton_krylov!(
                F!, integrator.u_tmp, (integrator.u, integrator.dt, integrator.du, integrator.du_tmp, integrator.p, integrator.t, integrator.stages, stage, integrator.RK), integrator.res;
                verbose = integrator.opts.verbose,
                tol_abs = integrator.opts.newton_tol_abs,
                tol_rel = integrator.opts.newton_tol_rel,
                max_niter = integrator.opts.newton_max_niter,
                algo = integrator.opts.krylov_algo,
                krylov_kwargs = integrator.opts.krylov_kwargs
            )
            if !stats.solved
                @warn "Newton did not converge" stats integrator.t
                errro("Newton did not converge")
            end
        end
        # Store the solution for each stage in stages
        # Next, we need to compute the value of the RHS at the stage.
        # We solve the non-linear problem in the z variable, thus u_tmp is z_i = (u_i - u_n) / dt
        if iszero(integrator.RK.a[stage, stage])
            # In this case, the stage is fully explicit. Thus, we have to evaluate
            # the RHS at the corresponding stage value.
            @. integrator.du = integrator.u_tmp * integrator.dt + integrator.u
            integrator.f(integrator.stages[stage], integrator.du, integrator.p, integrator.t + integrator.RK.c[stage] * integrator.dt)
        else
            # We avoid evaluating the RHS.
            # Thus, we rearrange the stage equation
            #
            #   z^i - \sum_{j=1}^i a_{ij} f(y^j) = 0
            #
            # as
            #
            #   f(y^j) = (z^i - \sum_{j=1}^{i-1} a_{ij} f(y^j)) / a_{ii}.
            #
            # Note that `integrator.res .= integrator.u_tmp` is the solution `z` for the
            # current `stage`.
            @. integrator.res = integrator.u_tmp
            for j in 1:(stage - 1)
                @. integrator.res = integrator.res - integrator.RK.a[stage, j] * integrator.stages[j]
            end
            @. integrator.stages[stage] = integrator.res / integrator.RK.a[stage, stage]
        end
    end
    # Finally, we compute the new step value
    #
    #   u^{n+1} = u^n + dt \sum_{i=1}^s b_i f(y^i).
    #
    # To reduce rounding errors, we first accumulate the RHS values and
    # multiply them by the time step size later.
    fill!(integrator.u_tmp, zero(eltype(integrator.u_tmp)))
    for j in 1:stages(alg)
        b = integrator.RK.b[j]
        @. integrator.u_tmp = integrator.u_tmp + b * integrator.stages[j]
    end
    @. integrator.u_tmp = integrator.u + integrator.dt * integrator.u_tmp
    return
end

# get a cache where the RHS can be stored
get_du(integrator::SimpleDiagonallyImplicit) = integrator.du
get_tmp_cache(integrator::SimpleDiagonallyImplicit) = (integrator.u_tmp,)

# some algorithms from DiffEq like FSAL-ones need to be informed when a callback has modified u
u_modified!(integrator::SimpleDiagonallyImplicit, ::Bool) = false

# used by adaptive timestepping algorithms in DiffEq
function set_proposed_dt!(integrator::SimpleDiagonallyImplicit, dt)
    return integrator.dt = dt
end

# Required e.g. for `glm_speed_callback`
function get_proposed_dt(integrator::SimpleDiagonallyImplicit)
    return integrator.dt
end

# stop the time integration
function terminate!(integrator::SimpleDiagonallyImplicit)
    integrator.finalstep = true
    return empty!(integrator.opts.tstops)
end

# used for AMR
function Base.resize!(integrator::SimpleDiagonallyImplicit, new_size)
    resize!(integrator.u, new_size)
    resize!(integrator.du, new_size)
    return resize!(integrator.u_tmp, new_size)
end

include("tableau.jl")
