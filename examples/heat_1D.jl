# # Implicit time-integration

# ## Necessary packages
using Ariadne
using CairoMakie

include(joinpath(dirname(pathof(Ariadne)), "..", "examples", "implicit.jl"))

# ## Heat 1D
# $ \frac{\partial u(x, t)}{\partial t} = a * \frac{\partial^2 u(x, t)}{\partial x^2 $

function heat_1D!(du, u, (a, Δx, bc!), t)
    N = length(u)

    ## Enforce the boundary condition
    bc!(u)
    du[1] = 0
    du[end] = 0

    ## Only compute within
    for i in 2:(N - 1)
        du[i] = a * (u[i + 1] - 2u[i] + u[i - 1]) / Δx^2
    end
    return
end


# Dirchlet boundary condition
# f(0) = 0
# f(L) = 0
# L = 1
# x ∈ (0,L)

function bc!(u)
    u[1] = 0
    return u[end] = 0
end

function periodic_bc!(u)
    u[1] = u[end - 1]
    return u[end] = u[2]
end

# initial condition

f(x) = 4x * (1 - x)

a = 0.5

N = 10

using LinearAlgebra

# ## Investigate the Jacobian's

# ### Euler
J = jacobian(G_Euler!, heat_1D!, zeros(N), (a, 1 / (N + 1), bc!), 0.1, 0.0)

# Rank:

rank(J)

# Condition number

cond(Array(J))

# ### Midpoint

J = jacobian(G_Midpoint!, heat_1D!, zeros(N), (a, 1 / (N + 1), bc!), 0.1, 0.0)

# Rank:

rank(J)

# Condition number

cond(Array(J))

# ### Trapezoid

J = jacobian(G_Trapezoid!, heat_1D!, zeros(N), (a, 1 / (N + 1), bc!), 0.1, 0.0)

# Rank:

rank(J)

# Condition number

cond(Array(J))


# ### Euler (Periodic boundary condition)
J = jacobian(G_Euler!, heat_1D!, zeros(N), (a, 1 / (N + 1), periodic_bc!), 0.1, 0.0)


function solve_heat_1D(G!, L, M, a, Δt, t_final, initial_condition, bc!)
    Δx = 1 / (M + 1)
    xs = 0.0:Δx:L
    ts = 0.0:Δt:t_final

    u₀ = initial_condition.(xs)

    hist = [copy(u₀)]
    callback = (u) -> push!(hist, copy(u))
    solve(G!, heat_1D!, u₀, (a, Δx, bc!), Δt, ts; callback)

    return xs, ts[1:length(hist)], hist
end


L = 1.0
M = 100
a = 0.2
Δt = 0.1
t_final = 3.0


function plot_1D(xs, ts, hist)
    fig, ax = lines(xs, hist[1])
    for i in 2:length(hist)
        lines!(ax, xs, hist[i])
    end
    return fig
end

xs, ts, hist = solve_heat_1D(G_Euler!, L, M, a, Δt, t_final, f, bc!);

contour(xs, ts, stack(hist))
plot_1D(xs, ts, hist)

xs, ts, hist = solve_heat_1D(G_Midpoint!, L, M, a, Δt, t_final, f, bc!);

contour(xs, ts, stack(hist))
plot_1D(xs, ts, hist)

xs, ts, hist = solve_heat_1D(G_Trapezoid!, L, M, a, Δt, t_final, f, bc!);

contour(xs, ts, stack(hist))
plot_1D(xs, ts, hist)
