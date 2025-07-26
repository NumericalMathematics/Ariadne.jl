using Trixi
using Implicit

trixi_include(joinpath(examples_dir(), "tree_2d_dgsem", "elixir_navierstokes_lid_driven_cavity.jl"), sol = nothing, mu = 0.1);

F!(du, u, p) = Trixi.rhs_parabolic!(du, u, p, 0.0)
u = copy(ode.u0)
du = copy(u)
@show "create Jacobian"
J = Implicit.JacobianOperator(F!, du, u, semi)
@show "create Sparse Jacobian"
Jsp = Implicit.collect(J) # Sparse Matrix Jacobian

@show "create cache for Batched coloring"
batched_updater = BatchedColoredUpdater(Jsp)

@show "Number of colors"
N = length(batched_updater.color_groups)

@show "create batched Jacobian"
J_batched = Implicit.Ariadne.BatchedJacobianOperator{N}(F!, du, u, semi)

@show "mul! with batched"
Implicit.Ariadne.mul!(batched_updater.result_matrix, J_batched, batched_updater.input_matrix)

@show "Benchmarking mul with batched"
@btime Implicit.Ariadne.mul!(batched_updater.result_matrix, J_batched, batched_updater.input_matrix)
