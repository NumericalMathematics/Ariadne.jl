# ## Necessary packages
using NewtonKrylov
using CairoMakie
using OffsetArrays

include(joinpath(dirname(pathof(NewtonKrylov)), "..", "examples", "implicit.jl"))
include(joinpath(dirname(pathof(NewtonKrylov)), "..", "examples", "halovector.jl"))

# ## Burgers Equation 2D

# ### Boundary
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


function burgers!(du::HaloVector, u::HaloVector, p, t)
    return burgers!(du.data, u.data, p, t)
end


function burgers!(dU, U, (dx, dy, μ), t)
    N, M, _ = size(u)
    N -= 2
    M -= 2

    u = view(U, :, :, 1)
    du = view(dU, :, :, 1)
    v = view(U, :, :, 2)
    dv = view(dU, :, :, 2)

    ## Enforce boundary conditions
    bc_zero!(u)
    bc_zero!(v)

    for j in 1:M
        for i in 1:N
            du[i, j] = (
                -u[i, j] / (2 * dx) * (u[i + 1, j] - u[i - 1, j]) -
                    v[i, j] / (2 * dy) * (u[i, j + 1] - u[i, j - 1])
            ) +
                μ * (
                (u[i + 1, j] - 2 * u[i, j] + u[i - 1, j]) / dx^2 +
                    (u[i, j + 1] - 2 * u[i, j] + u[i, j - 1]) / dy^2
            )
            dv[i, j] = (
                -v[i, j] / (2 * dx) * (v[i + 1, j] - v[i - 1, j]) -
                    u[i, j] / (2 * dy) * (v[i, j + 1] - v[i, j - 1])
            ) +
                μ * (
                (v[i + 1, j] - 2 * v[i, j] + v[i - 1, j]) / dx^2 +
                    (v[i, j + 1] - 2 * v[i, j] + v[i, j - 1]) / dy^2
            )
        end
    end
    return
end

let
    N = M = 12
    Δx = 1 / (N + 1)  # x-grid spacing
    Δy = 1 / (M + 1)  # y-grid spacing

    a = 0.01

    ## TODO: This is a time-step from explicit
    Δt = Δx^2 * Δy^2 / (2.0 * a * (Δx^2 + Δy^2)) # Largest stable time step

    u₀ = HaloVector(OffsetArray(zeros(N + 2, M + 2), 0:(N + 1), 0:(M + 1)))
    jacobian(G_Euler!, burgers!, u₀, (Δx, Δy, a), Δt, 0.0)
end

# TODO: HaloVector with underlying 3D datastructure? Or do we better do 2d with SArray?
