# Using the example from:
# A. Pal et al., “NonlinearSolve.Jl: High-performance and robust solvers for systems of nonlinear equations in Julia,” arXiv [math.NA], 24-Mar-2024.
# https://arxiv.org/abs/2403.16341
# Fig 1

function generalized_rosenbrock(x, _)
    vcat(
        1 - x[1],
        10 .* (x[2:end] .- x[1:(end - 1)] .* x[1:(end - 1)])
    )
end


using Ariadne

N = 12 
x_start = vcat(-1.2, ones(N-1))
# for N=6 we require 21 iterations in 7.1211e-5 seconds
# for N=7 we require 66 iterations in 0.000129203 seconds
# for N=8 we require 56 iterations in 0.000106193 seconds
# for N=9 we do not find a solution within 100_000 iterations
# for N=10 we do not find a solution within 100_000 iterations
# for N=11 we do not find a solution within 100_000 iterations

_, stats = newton_krylov(
    generalized_rosenbrock,
    copy(x_start);
    algo = :gmres,
    max_niter = 100_000
)

# Using simple line search: BacktrackingLineSearch

# for N=6 we require 115 iterations in 0.000415178 seconds
# for N=7 we require 182 iterations in 0.003204213 seconds
# for N=8 we require 282 iterations in 0.001153047 seconds
# for N=9 we require 465 iterations in 0.00423094 seconds
# for N=10 we require 884 iterations in 0.006603219 seconds
# NOTE: Pal et.al. report that for N=10 their backtracking implementation does not converge.
#       They use abstol = 1e-8 we use 1e-12
# for N=11 we require 1568 iterations in 0.01126607 seconds
# for N=12 we require 2346 iterations in 0.024341541 seconds
_, stats = newton_krylov(
    generalized_rosenbrock,
    copy(x_start);
    algo = :gmres,
    linesearch! = Ariadne.LineSearches.BacktrackingLineSearch(),
    max_niter = 100_000
)
