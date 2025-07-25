using Trixi
using Implicit

trixi_include(joinpath(examples_dir(), "tree_2d_dgsem", "elixir_navierstokes_lid_driven_cavity.jl"), sol = nothing, mu = 0.1);

F!(du, u, p) = Trixi.rhs_parabolic!(du, u, p, 0.0)
u = copy(ode.u0)
du = copy(u)

J = Implicit.JacobianOperator(F!, du, u, semi)
Jsp = Implicit.collect(J) # Sparse Matrix Jacobian



batched_updater = BatchedColoredUpdater(Jsp)

N = length(batched_updater.color_groups)

J_batched = Implicit.Ariadne.BatchedJacobianOperator{N}(F!, du, u, semi)
Implicit.Ariadne.mul!(batched_updater.result_matrix, J_batched, batched_updater.input_matrix)

@btime Implicit.Ariadne.mul!(batched_updater.result_matrix, J_batched, batched_updater.input_matrix)