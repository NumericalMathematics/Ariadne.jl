# # Generalized Rosenbrock

# This example is taken from Fig. 1 of:
# > A. Pal et al., "NonlinearSolve.jl: High-performance and robust solvers for systems
# > of nonlinear equations in Julia," arXiv [math.NA], 24-Mar-2024.
# > https://arxiv.org/abs/2403.16341

# ## Packages

using Ariadne
using LineSearches

# ## Problem definition

# The generalized Rosenbrock function in $N$ dimensions:
# ```math
# F(x)_1 = 1 - x_1, \quad F(x)_i = 10(x_i - x_{i-1}^2), \quad i = 2, \ldots, N
# ```

function generalized_rosenbrock(x, _)
    return vcat(
        1 - x[1],
        10 .* (x[2:end] .- x[1:(end - 1)] .* x[1:(end - 1)])
    )
end

# The standard starting point is $x_1 = -1.2$, $x_i = 1$ for $i \geq 2$.

N = 12
x_start = vcat(-1.2, ones(N - 1))

# ## Without line search

# Solving with GMRES and no line search (`NoLineSearch`).
# The number of iterations required grows quickly with $N$ and the solver
# fails to converge for $N \geq 9$ within the iteration budget.

_, stats = newton_krylov(
    generalized_rosenbrock,
    copy(x_start);
    algo = :gmres,
    linesearch! = NoLineSearch(),
    max_niter = 100_000
)
stats

# ## With backtracking line search

# Using `BacktrackingLineSearch` stabilizes convergence for larger $N$.
# Pal et al. report that their backtracking implementation does not converge for $N = 10$
# (using `abstol = 1e-8`); with `abstol = 1e-12` our implementation converges for all
# $N \leq 12$.

_, stats = newton_krylov(
    generalized_rosenbrock,
    copy(x_start);
    algo = :gmres,
    linesearch! = BacktrackingLineSearch(),
    max_niter = 100_000
)
stats

# ## With LineSearches.jl

_, stats = newton_krylov(
    generalized_rosenbrock,
    copy(x_start);
    algo = :gmres,
    linesearch! = Ariadne.LineSearches.LineSearches_JL(BackTracking()),
    max_niter = 100_000
)

# Surfaces GC error yay...
# _, stats = newton_krylov(
#     generalized_rosenbrock,
#     copy(x_start);
#     algo = :gmres,
#     linesearch! = Ariadne.LineSearches.LineSearches_JL(HagerZhang()),
#     max_niter = 100_000
# )

# _, stats = newton_krylov(
#     generalized_rosenbrock,
#     copy(x_start);
#     algo = :gmres,
#     linesearch! = Ariadne.LineSearches.LineSearches_JL(MoreThuente()),
#     max_niter = 100_000
# )
