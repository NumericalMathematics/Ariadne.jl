# # Implicit time-integration (With DG)

# ## Necessary packages
using Ariadne
using CairoMakie
using LinearAlgebra
using SummationByPartsOperators

# Include the implicit time-steppers from [`Implicit schemes`](@ref implicit_schemes)

include(joinpath(dirname(pathof(Ariadne)), "..", "examples", "implicit.jl"));

# ## Setting up the space discretization
xmin = 0.0
xmax = 1.0

polydeg = 3
elements = 40

D_local = legendre_derivative_operator(xmin = -1.0, xmax = 1.0, N = polydeg + 1)
mesh = UniformPeriodicMesh1D(; xmin, xmax, Nx = elements)

## D1m = D1p = couple_discontinuously(D_local, mesh, Val(:central))
D1m = couple_discontinuously(D_local, mesh, Val(:minus))
D1p = couple_discontinuously(D_local, mesh, Val(:plus))

x = grid(D1m)

# ## Heat 1D
# $ \frac{\partial u(x, t)}{\partial t} = \frac{\partial^2 u(x, t)}{\partial x^2} $

function heat_1D!(du, u, (D1m, D1p), t)
    du1 = D1p * u
    mul!(du, D1m, du1)
    return
end

# Initial condition:

f(x) = sin(π * x)

# ## Investigate the Jacobian's

# ### Euler
J = jacobian(G_Euler!, heat_1D!, zero(x), (D1m, D1p), 0.1, 0.0)

# ### Midpoint

J = jacobian(G_Midpoint!, heat_1D!, zero(x), (D1m, D1p), 0.1, 0.0)

# ### Trapezoid

J = jacobian(G_Trapezoid!, heat_1D!, zero(x), (D1m, D1p), 0.1, 0.0)

# ## Problem setup

function solve_heat_1D(G!, x, Δt, t_final, initial_condition, p)
    ts = 0.0:Δt:t_final

    u₀ = initial_condition.(x)

    hist = [copy(u₀)]
    callback = (u) -> push!(hist, copy(u))
    solve(G!, heat_1D!, u₀, p, Δt, ts; callback)

    return x, ts[1:length(hist)], hist
end

# ## Problem plotting

function plot_1D(xs, ts, hist)
    fig, ax = lines(xs, hist[1])
    for i in 2:length(hist)
        lines!(ax, xs, hist[i])
    end
    return fig
end

# ## Choose time-parameters

Δt = 0.01
t_final = 50.0

# ## Solving the problem with Implicit Euler

xs, ts, hist = solve_heat_1D(G_Euler!, x, Δt, t_final, f, (D1m, D1p));

# ### Initial and final
let
    fig, ax = lines(x, hist[1])
    lines!(ax, x, hist[end])
    fig
end

# ### Contour
contour(xs, ts, stack(hist))

# ### Time-shift
plot_1D(xs, ts, hist)

# ## Solving the problem with Implicit Midpoint
xs, ts, hist = solve_heat_1D(G_Midpoint!, x, Δt, t_final, f, (D1m, D1p));

# ### Initial and final
let
    fig, ax = lines(x, hist[1])
    lines!(ax, x, hist[end])
    fig
end

# ### Contour
contour(xs, ts, stack(hist))

# ### Time-shift
plot_1D(xs, ts, hist)

# ## Solving the problem with Implicit Trapezoid
xs, ts, hist = solve_heat_1D(G_Trapezoid!, x, Δt, t_final, f, (D1m, D1p));

# ### Initial and final
let
    fig, ax = lines(x, hist[1])
    lines!(ax, x, hist[end])
    fig
end

# ### Contour
contour(xs, ts, stack(hist))

# ### Time-shift
plot_1D(xs, ts, hist)

# ## Upwind operator

# ### Set up the space discretization
nnodes = 120
accuracy_order = 3
D = upwind_operators(
    periodic_derivative_operator;
    accuracy_order, xmin, xmax, N = nnodes
)
D1m = D.minus
D1p = D.plus

x = grid(D1m)

# ### Solve with Implicit Euler
xs, ts, hist = solve_heat_1D(G_Euler!, x, Δt, t_final, f, (D1m, D1p));

# #### Initial and final

let
    fig, ax = lines(x, hist[1])
    lines!(ax, x, hist[end])
    fig
end

# #### Contour
contour(xs, ts, stack(hist))

# #### Time-shift
plot_1D(xs, ts, hist)
