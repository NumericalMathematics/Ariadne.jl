RK = Implicit.RKTableau(Implicit.RKLSSPIMEX332Z())
Δt = 0.01/8
u = copy(ode.u0)
du = copy(u)
du_tmp = copy(u)
jstages1 = copy(u)
jstages2 = copy(u)
jstages3 = copy(u)
stages1 = copy(u)
stages2 = copy(u)
stages3 = copy(u)
res = copy(u)
p = semi
t = 0.0
stage = 1			
F!(du, u, p) = Trixi.rhs_parabolic!(du, u, p, 0.0) ## parabolic
J = Implicit.JacobianOperator(F!, du, u, semi)
invdt = inv(RK.ah[stage,stage] * Δt)
M = Implicit.LMROperator(J, invdt)
@. res = invdt * RK.d[stage] * u
kc = Implicit.KrylovConstructor(res)
workspace = Implicit.krylov_workspace(:gmres, kc)
Implicit.krylov_solve!(workspace, M, res, atol = 1e-6, rtol = 1e-6)

@. jstages1 = workspace.x
@. res = u+ jstages1/RK.ah[stage,stage] + -1/RK.ah[stage,stage]*RK.d[stage]* u
Trixi.rhs_parabolic!(du, res, semi, t + RK.c[stage] * Δt)
Trixi.rhs!(du_tmp, res, semi, t + RK.c[stage] * Δt)
stages1 = du + du_tmp

stage = 2
invdt = inv(RK.ah[stage,stage] * Δt)
M = Implicit.LMROperator(J, invdt)
@. res = invdt * RK.d[stage] * u - invdt*RK.ah[stage,stage] * RK.gamma[stage,1] *( RK.d[1] *  u - jstages1) + RK.a[stage,1] * stages1
Implicit.krylov_solve!(workspace, M, res, atol = 1e-6, rtol = 1e-6)
@. jstages2 = workspace.x
@. res = u + jstages2/RK.ah[stage,stage] + -1/RK.ah[stage,stage]*RK.d[stage]* u + RK.gamma[stage,1] * (RK.d[1] * u - jstages1) 
Trixi.rhs_parabolic!(du, res, p, t + RK.c[stage] * Δt)
Trixi.rhs!(du_tmp, res, p, t + RK.c[stage] * Δt)
@. stages2 = du + du_tmp

stage = 3
invdt = inv(RK.ah[stage,stage] * Δt)
M = Implicit.LMROperator(J, invdt)
@. res = invdt * RK.d[stage] * u - invdt*RK.ah[stage,stage] * (RK.gamma[stage,1] *( RK.d[1] *  u - jstages1) + RK.gamma[stage,2] * (RK.d[2] * u  - jstages2)) + RK.a[stage,1] * stages1 + RK.a[stage,2] * stages2
@show "krylov solve"
@time a =  Implicit.krylov_solve!(workspace, M, res, jstages2, atol = 1e-6, rtol = 1e-6)
@show a
