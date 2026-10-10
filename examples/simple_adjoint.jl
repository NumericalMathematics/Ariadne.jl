# # Adjoint of a nonlinear solve
#
# The simple 2D example from [Kelley2003](@cite) with parameters `p`. We compute the gradient
# of a loss function `g(x(p), p)` of the solution `x(p)` of `F(x, p) = 0` with respect to `p`
# by the discrete adjoint method (see, e.g., the
# [notes on adjoint methods by S. G. Johnson](https://math.mit.edu/~stevenj/18.336/adjoint.pdf)),
# instead of differentiating through the Newton-Krylov solver:
# ```math
# \frac{dg}{dp} = \frac{\partial g}{\partial p} - \lambda^T \frac{\partial F}{\partial p},
# \qquad \left(\frac{\partial F}{\partial x}\right)^T \lambda = \left(\frac{\partial g}{\partial x}\right)^T.
# ```

using Ariadne, LinearAlgebra
using Enzyme

function F!(res, x, p)
    res[1] = p[1] * x[1]^2 + p[2] * x[2]^2 - 2
    res[2] = exp(p[1] * x[1] - 1) + p[2] * x[2]^2 - 2
    return nothing
end

p = [1.0, 1.3]
x, result = newton_krylov!(F!, [2.0, 0.5], p; tol_rel = 1.0e-12)
@assert result.solved

# The loss function measures the distance of the solution to a target `x̂`.

const x̂ = [1.0, 1.0]
g(x, p) = sum(abs2, x .- x̂)

# ## Adjoint gradient
#
# [`adjoint_gradient`](@ref) computes `∂g/∂x` and `∂g/∂p` with Enzyme.jl, solves the adjoint
# system with GMRES (with products by `(∂F/∂x)ᵀ` from Enzyme.jl reverse mode), and computes
# the vector-Jacobian product `λᵀ ∂F/∂p`.

r = adjoint_gradient(g, F!, x, p)
@assert r.stats.solved
r.dp

# We compare with central finite differences of the loss of the solution.

function loss(p)
    x, result = newton_krylov!(F!, [2.0, 0.5], p; tol_rel = 1.0e-12)
    return g(x, p)
end

h = 1.0e-6
dp_fd = [(loss(p .+ h .* e) - loss(p .- h .* e)) / 2h for e in ([1.0, 0.0], [0.0, 1.0])]
@assert isapprox(r.dp, dp_fd; rtol = 1.0e-5)
dp_fd

# ## Several loss functions at once
#
# For `N` loss functions, [`adjoint_gradient`](@ref) with `Val(N)` takes a function that
# writes the `N` values into a vector. It computes the `N` gradients with one batched
# Enzyme.jl sweep through the losses, one block GMRES solve of the `N` adjoint systems, and
# one batched sweep through `F!` (this requires Julia 1.11 or later).

function losses!(out, x, p)
    out[1] = g(x, p)
    out[2] = p[2] * x[2]^2
    return nothing
end

r2 = adjoint_gradient(losses!, F!, x, p, Val(2))
@assert r2.solved
@assert r2.dp[1] ≈ r.dp
r2.dp

# ## Differentiating through a solve with Enzyme.jl
#
# An [`ImplicitFunction`](@ref) wraps the solver. Differentiating a function that calls
# [`implicit_solve!`](@ref) with Enzyme.jl applies the implicit function theorem at the
# solution instead of differentiating the Newton iterations.

solve!(x, p) = (newton_krylov!(F!, x, p; tol_rel = 1.0e-12); x)
implicit = ImplicitFunction(F!, solve!, x)

function objective(x, p)
    implicit_solve!(implicit, x, p)
    return g(x, p)
end

dp = zero(p)
autodiff(Reverse, objective, Active, Duplicated([2.0, 0.5], zeros(2)), Duplicated(p, dp))
@assert dp ≈ r.dp
dp
