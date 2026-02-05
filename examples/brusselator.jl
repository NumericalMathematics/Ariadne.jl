# ## Definition of the Brusselator Equation

# The Brusselator PDE is defined as follows:

# ```math
# \begin{align}
# 0 &= 1 + u^2v - 4.4u + \alpha(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}) + f(x, y, t)\\
# 0 &= 3.4u - u^2v + \alpha(\frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2})
# \end{align}
# ```

# where

# ```math
# f(x, y, t) = \begin{cases}
# 5 & \quad \text{if } (x-0.3)^2+(y-0.6)^2 ≤ 0.1^2 \text{ and } t ≥ 1.1 \\
# 0 & \quad \text{else}
# \end{cases}
# ```

# and the initial conditions are

# ```math
# \begin{align}
# u(x, y, 0) &= 22\cdot (y(1-y))^{3/2} \\
# v(x, y, 0) &= 27\cdot (x(1-x))^{3/2}
# \end{align}
# ```

# with the periodic boundary condition

# ```math
# \begin{align}
# u(x+1,y,t) &= u(x,y,t) \\
# u(x,y+1,t) &= u(x,y,t)
# \end{align}
# ```

# To solve this PDE, we will discretize it into a system of ODEs with the finite difference
# method. We discretize `u` and `v` into arrays of the values at each time point:
# `u[i,j] = u(i*dx,j*dy)` for some choice of `dx`/`dy`, and same for `v`. Then our ODE is
# defined with `U[i,j,k] = [u v]`. The second derivative operator, the Laplacian, discretizes
# to become a tridiagonal matrix with `[1 -2 1]` and a `1` in the top right and bottom left
# corners. The nonlinear functions are then applied at each point in space (they are
# broadcast). Use `dx=dy=1/32`.

# The resulting `NonlinearProblem` definition is:

using NewtonKrylov, LinearAlgebra

const N = 32
const xyd_brusselator = range(0, stop = 1, length = N)

brusselator_f(x, y) = (((x - 0.3)^2 + (y - 0.6)^2) <= 0.1^2) * 5.0
limit(a, N) = a == N + 1 ? 1 : a == 0 ? N : a

function brusselator_2d(du, u, p)
    A, B, alpha, dx = p
    alpha = alpha / dx^2
    return @inbounds for I in CartesianIndices((N, N))
        i, j = Tuple(I)
        x, y = xyd_brusselator[I[1]], xyd_brusselator[I[2]]
        ip1, im1, jp1, jm1 = limit(i + 1, N), limit(i - 1, N), limit(j + 1, N),
            limit(j - 1, N)
        du[i, j, 1] = alpha * (
            u[im1, j, 1] + u[ip1, j, 1] + u[i, jp1, 1] + u[i, jm1, 1] -
                4u[i, j, 1]
        ) +
            B +
            u[i, j, 1]^2 * u[i, j, 2] - (A + 1) * u[i, j, 1] + brusselator_f(x, y)
        du[i, j, 2] = alpha * (
            u[im1, j, 2] + u[ip1, j, 2] + u[i, jp1, 2] + u[i, jm1, 2] -
                4u[i, j, 2]
        ) + A * u[i, j, 1] - u[i, j, 1]^2 * u[i, j, 2]
    end
end

function init_brusselator_2d(xyd)
    N = length(xyd)
    u = zeros(N, N, 2)
    for I in CartesianIndices((N, N))
        x = xyd[I[1]]
        y = xyd[I[2]]
        u[I, 1] = 22 * (y * (1 - y))^(3 / 2)
        u[I, 2] = 27 * (x * (1 - x))^(3 / 2)
    end
    return u
end

const p = (3.4, 1.0, 10.0, step(xyd_brusselator))
u0 = init_brusselator_2d(xyd_brusselator)

J = collect(
    NewtonKrylov.JacobianOperator(
        (du, u) -> brusselator_2d(du, u, p), similar(u0), u0
    )
)

# Corruption of in malloc...
newton_krylov!((du, u) -> brusselator_2d(du, u, p), u0)
