# # Implicit time-integration (Spring example)

# ## Necessary packages
using Ariadne
using CairoMakie

# Include the implicit time-steppers from [`Implicit schemes`](@ref implicit_schemes)

include(joinpath(dirname(pathof(Ariadne)), "..", "examples", "implicit.jl"))

# ## Spring equations

function f!(du, u, (γ,), t)
    du[1] = u[2] # dx/dt = v
    du[2] = -γ^2 * u[1] # dv/dt = -γ^2 * x
    return nothing
end

# ## Problem setup
function implicit_spring(G! = G_Euler!, Δt = 0.01; verbose = 0)
    k = 2.0    # spring constant
    m = 1.0    # object's mass
    x0 = 0.1 # initial position
    v0 = 0.0   # initial velocity

    t₀ = 0.0
    tₛ = 40.0

    ts = t₀:Δt:tₛ

    uₙ = [x0, v0]

    γ = sqrt(k / m)

    hist = [copy(uₙ)]
    callback = (u) -> push!(hist, copy(u))
    solve(G!, f!, uₙ, (γ,), Δt, ts, ; callback, verbose)
    return hist, ts[1:length(hist)]
end


# ## Plots
function compare(Δt = 0.01)
    hist_euler, ts_euler = implicit_spring(G_Euler!, Δt)
    v_euler = map(y -> y[1], hist_euler)
    x_euler = map(y -> y[2], hist_euler)

    hist_midpoint, ts_midpoint = implicit_spring(G_Midpoint!, Δt)
    v_midpoint = map(y -> y[1], hist_midpoint)
    x_midpoint = map(y -> y[2], hist_midpoint)


    hist_trapezoid, ts_trapezoid = implicit_spring(G_Trapezoid!, Δt)
    v_trapezoid = map(y -> y[1], hist_trapezoid)
    x_trapezoid = map(y -> y[2], hist_trapezoid)


    fig = Figure()

    lines(fig[1, 1], ts_euler, v_euler, label = "Euler")
    lines(fig[1, 2], ts_euler, x_euler, label = "Euler")

    lines(fig[2, 1], ts_midpoint, v_midpoint, label = "Midpoint")
    lines(fig[2, 2], ts_midpoint, x_midpoint, label = "Midpoint")

    lines(fig[3, 1], ts_trapezoid, v_trapezoid, label = "Trapezoid")
    lines(fig[3, 2], ts_trapezoid, x_trapezoid, label = "Trapezoid")

    return fig
end

# ### t = 0.01
compare(0.01)

# ### t = 0.02
compare(0.02)

# ### t = 0.05
compare(0.05)

# ### t = 0.1
compare(0.1)

# ### t = 1.0
compare(1.0)

# ### t = 1.0
compare(10.0)

# ## Jacobian of implicit step

# ### Implicit Euler

jacobian(G_Euler!, f!, [0.1, 0.0], (sqrt(2.0 / 1.0),), 0.01, 0.0)

# ### Implicit Midpoint

jacobian(G_Midpoint!, f!, [0.1, 0.0], (sqrt(2.0 / 1.0),), 0.1, 0.0)

# ### Implicit Trapezoid

jacobian(G_Trapezoid!, f!, [0.1, 0.0], (sqrt(2.0 / 1.0),), 0.1, 0.0)
