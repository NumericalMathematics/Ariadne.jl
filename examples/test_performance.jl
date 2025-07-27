using Trixi
using Implicit
using CairoMakie
using BenchmarkTools
@assert !Trixi._PREFERENCE_POLYESTER
@assert !Trixi._PREFERENCE_LOOPVECTORIZATION

trixi_include(joinpath(examples_dir(), "tree_2d_dgsem", "elixir_navierstokes_lid_driven_cavity.jl"), sol = nothing);

F!(du, u, p) = Trixi.rhs!(du, u, p, 0.0)
G!(du, u, p) = Trixi.rhs_parabolic!(du, u, p, 0.0)

u0 = copy(ode.u0)
un = similar(u0)
du_cons = copy(u0)
du_par = copy(u0)
Trixi.rhs!(du_cons, u0, ode.p, 0.0)
Trixi.rhs_parabolic!(du_par, u0, ode.p, 0.0)
dt = 0.1
Jcons = Implicit.JacobianOperator(F!, du, u0, semi)
Jpar = Implicit.JacobianOperator(G!, du, u0, semi)
Mcons = Implicit.LMOperator(Jcons, dt)
Mpar = Implicit.LMOperator(Jpar, dt)
@show "Performance for mul with Jacobian"
@time Implicit.mul!(un, Jcons, u0)
@time Implicit.mul!(un, Jpar, u0)

res = copy(du)
@. res = u0 + dt * du_cons

starter = copy(res)

@. starter = res + dt * du_par

kc = Implicit.KrylovConstructor(res)
workspace = Implicit.krylov_workspace(:gmres, kc)
@show "Performance for krylov_solve!"
@time Implicit.krylov_solve!(workspace, Mpar, res)

## Plain Performance:
# 
#
