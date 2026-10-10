# # Implicit time-integration

# ## Necessary packages
using Ariadne
using CairoMakie
using OffsetArrays

include(joinpath(dirname(pathof(Ariadne)), "..", "examples", "implicit.jl"))
include(joinpath(dirname(pathof(Ariadne)), "..", "examples", "halovector.jl"))

# ## Diffusion 2D

# ### Boundary

function bc_periodic!(u)
    N, M = size(u)
    N = N - 2
    M = M - 2

    ## (wrap around)
    u[0, :] .= u[N, :]
    u[N + 1, :] .= u[1, :]
    u[:, 0] .= u[:, N]
    u[:, N + 1] .= u[:, 1]
    return nothing
end

function bc_zero!(u)
    N, M = size(u)
    N = N - 2
    M = M - 2

    u[0, :] .= 0
    u[N + 1, :] .= 0
    u[:, 0] .= 0
    u[:, N + 1] .= 0
    return nothing
end


function diffusion!(du::HaloVector, u::HaloVector, p, t)
    return diffusion!(du.data, u.data, p, t)
end

function diffusion!(du, u, (a, Δx, Δy, bc!), _)
    N, M = size(u)
    N = N - 2
    M = M - 2

    ## Enforce boundary conditions
    bc!(u)

    for i in 1:N
        for j in 1:M
            du[i, j] = a * (
                (u[i + 1, j] - 2 * u[i, j] + u[i - 1, j]) / Δx^2 +
                    (u[i, j + 1] - 2 * u[i, j] + u[i, j - 1]) / Δy^2
            )
        end
    end
    return
end

a = 0.01

N = M = 40

Δx = 1 / (N + 1)  # x-grid spacing
Δy = 1 / (M + 1)  # y-grid spacing

## TODO: This is a time-step from explicit
Δt = Δx^2 * Δy^2 / (2.0 * a * (Δx^2 + Δy^2)) # Largest stable time step

let
    N = M = 12
    u₀ = HaloVector(OffsetArray(zeros(N + 2, M + 2), 0:(N + 1), 0:(M + 1)))
    jacobian(G_Euler!, diffusion!, u₀, (a, Δx, Δy, bc_zero!), Δt, 0.0)
end

xs = 0.0:Δx:1.0
ys = 0.0:Δy:1.0

function f(x, y)
    return sin(π * x) .* sin(π * y)
end

u₀ = let
    _u₀ = zeros(N + 2, M + 2)
    _u₀ .= f.(xs, ys')
    HaloVector(OffsetArray(_u₀, 0:(N + 1), 0:(M + 1)))
end

function plot_state(t, u, xs, ys)
    data = @lift $u.data.parent

    xs_line = @lift @view $data[:, M ÷ 2]
    ys_line = @lift @view $data[N ÷ 2, :]

    fig = Figure()

    ax = Axis(fig[1, 1])
    heatmap!(ax, xs, ys, data)

    ax1 = Axis(fig[2, 1])
    lines!(ax1, xs, xs_line)

    ax2 = Axis(fig[2, 2])
    lines!(ax2, ys, ys_line)

    fig[0, :] = Label(fig, @lift("t = $($t)"))

    return fig
end

fig = plot_state(Observable(0.0), Observable(u₀), xs, ys)

function create_video_implicit(filename, G!, f!, xs, ys, u, p, Δt, t_stop, frame_kwargs)
    ts = 0.0:Δt:t_stop
    _u = Observable(u)
    _t = Observable(first(ts))

    fig = plot_state(_t, _u, xs, ys)
    return record(fig, filename; frame_kwargs...) do io
        function callback(__u)
            _u[] = u
            _t[] += Δt
            ## autolimits!(ax) # update limits
            recordframe!(io)
            return yield()
        end
        solve(G!, f!, u, p, Δt, ts; callback, verbose = 1, krylov_kwargs = (; verbose = 1, reorthogonalization = true))
    end
end

create_video_implicit(
    joinpath(@__DIR__, "implicit_euler.mp4"),
    G_Euler!, diffusion!, xs, ys, copy(u₀), (a, Δx, Δy, bc_zero!), Δt, 10.0, (; framerate = 30)
)

create_video_implicit(
    joinpath(@__DIR__, "implicit_midpoint.mp4"),
    G_Midpoint!, diffusion!, xs, ys, copy(u₀), (a, Δx, Δy, bc_zero!), Δt, 10.0, (; framerate = 30)
)

create_video_implicit(
    joinpath(@__DIR__, "implicit_trapezoid.mp4"),
    G_Trapezoid!, diffusion!, xs, ys, copy(u₀), (a, Δx, Δy, bc_zero!), Δt, 10.0, (; framerate = 30)
)

## TODO:
create_video_implicit(
    joinpath(@__DIR__, "implicit_euler_periodic.mp4"),
    G_Trapezoid!, diffusion!, xs, ys, copy(u₀), (a, Δx, Δy, bc_periodic!), Δt, 2 * Δt, (; framerate = 30)
)
