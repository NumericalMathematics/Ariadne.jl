abstract type SimpleLinearImplicitExplicitAlgorithm{N} end

abstract type RKLIMEX{N} <: SimpleLinearImplicitExplicitAlgorithm{N} end

abstract type RKLIMEXZ{N} <: SimpleLinearImplicitExplicitAlgorithm{N} end

struct IMEXRKButcher{T1<:AbstractArray,T2<:AbstractArray} <: RKTableau
    a::T1
    b::T2
    c::T2
    ah::T1
    bh::T2
    ch::T2	
end
struct IMEXRKZButcher{T1<:AbstractArray,T2<:AbstractArray} <: RKTableau
    a::T1
    b::T2
    c::T2
    ah::T1
    bh::T2
    ch::T2
    d::T2
    gamma::T1
end

struct RKLinearImplicitExplicitEuler <: RKLIMEX{1} end

struct RKLSSPIMEX332 <: RKLIMEX{3} end

struct RKLSSPIMEX332Z <: RKLIMEXZ{3} end

struct LSDIRK222 <: RKLIMEXZ{2} end

struct AGSA432Z <: RKLIMEXZ{4} end
struct AGSA432 <: RKLIMEX{4} end

function mul!(out::AbstractVector, M::LMOperator, v::AbstractVector)
    # out = (I/dt - J(f,x,p)) * v
    	   mul!(out, M.J, v)
 	   @. out = v - out * M.dt
    return nothing
end

struct LMROperator{JOp}
   J::JOp
invdt::Float64
end

Base.size(M::LMROperator) = size(M.J)
Base.eltype(M::LMROperator) = eltype(M.J)
Base.length(M::LMROperator) = length(M.J)
function mul!(out::AbstractVector, M::LMROperator, v::AbstractVector)
    # out = (I/dt - J(f,x,p)) * v
   	 mul!(out, M.J, v)
    	@. out = v * M.invdt - out
	return nothing
end

function (::RKLinearImplicitExplicitEuler)(res, uₙ, Δt, f1!, f2!, du, du_tmp, u, p, t, stages, ustages, jstages, stage, RK, M, lin_du_tmp, lin_du_tmp1, workspace)
    if stage == 1
        # Stage 1:

	## f2 is the conservative part
	## f1 is the parabolic part
	mul!(lin_du_tmp, M.J, uₙ)
#	mul!(lin_du_tmp1, J, u)	
	f2!(du, uₙ, p, t + RK.c[stage] * Δt)
        f1!(du_tmp, uₙ, p, t + RK.c[stage] * Δt)	
	res .= uₙ .+ RK.a[stage, stage] * Δt .* (du .+ du_tmp .- lin_du_tmp)
	krylov_solve!(workspace, M, res, atol = 1e-6, rtol = 1e-6)
	@. u = workspace.x
#	f2!(du, workspace.x, p, t + RK.c[stage] * Δt)
#       f1!(du_tmp, workspace.x, p, t + RK.c[stage] * Δt)
#    	stages[stage] .= du .+ du_tmp  
#        @. u = uₙ + RK.b[1] * Δt * stages[1]
    end
end


function (::RKLIMEXZ{2})(res, uₙ, Δt, f1!, f2!, du, du_tmp, u, p, t, stages, ustages, jstages, stage, RK, lin_du_tmp, lin_du_tmp1, workspace, assume_p_const)
	@trixi_timeit timer() "F! function inside" F!(du, u, p) = f1!(du, u, p, t) ## parabolic
 	invdt = inv(RK.ah[stage,stage] * Δt)
    if stage == 1			
   	@trixi_timeit timer() "Jacobian inside stage1" J = JacobianOperator(F!, du, uₙ, p, assume_p_const = assume_p_const)
	@trixi_timeit timer() "LM Operator inside stage 1" M = LMROperator(J, invdt)
	@. res = invdt * RK.d[stage] * uₙ
	 @trixi_timeit timer() "krylov solve stage 1" krylov_solve!(workspace, M, res, atol = 1e-6, rtol = 1e-6)
	@. jstages[stage] = workspace.x
	@. res = uₙ+ jstages[stage]/RK.ah[stage,stage] + -1/RK.ah[stage,stage]*RK.d[stage]* uₙ
	@. ustages[stage] = res
	f2!(du, res, p, t + RK.c[stage] * Δt)
    	f1!(du_tmp, res, p, t + RK.c[stage] * Δt)
	@. stages[stage] = du + du_tmp
    elseif stage == 2
	@trixi_timeit timer() "Jacobian inside stage 2" J = JacobianOperator(F!, du, ustages[stage-1], p, assume_p_const = assume_p_const)
	@trixi_timeit timer() "LM Operator inside stage 2" M = LMROperator(J, invdt)
	@. res = invdt * RK.d[stage] * uₙ - invdt*RK.ah[stage,stage] * RK.gamma[stage,1] *( RK.d[1] *  uₙ - jstages[1]) + RK.a[stage,1] * stages[1]
	@trixi_timeit timer() "krylov solve stage 2" krylov_solve!(workspace, M, res, jstages[stage-1], atol = 1e-6, rtol = 1e-6)
	@. jstages[stage] = workspace.x
	@. res = uₙ+ jstages[stage]/RK.ah[stage,stage] + -1/RK.ah[stage,stage]*RK.d[stage]* uₙ + RK.gamma[stage,1] * (RK.d[1] * uₙ - jstages[1]) 
	@. ustages[stage] = res
	f2!(du, res, p, t + RK.c[stage] * Δt)
        f1!(du_tmp, res, p, t + RK.c[stage] * Δt)
	@. stages[stage] = du + du_tmp
#	stage = 1	
#	@trixi_timeit timer() "Jacobian inside stage1" J = JacobianOperator(F!, du, ustages[stage], p, assume_p_const = assume_p_const)
#	mul!(jstages[stage], J, ustages[stage])	
#	stage = 2	
#	@trixi_timeit timer() "Jacobian inside stage1" J = JacobianOperator(F!, du, ustages[stage], p, assume_p_const = assume_p_const)
#	mul!(jstages[stage], J, ustages[stage])	
#	stage = 3	
#	@trixi_timeit timer() "Jacobian inside stage1" J = JacobianOperator(F!, du, ustages[stage], p, assume_p_const = assume_p_const)
#	mul!(jstages[stage], J, ustages[stage])	
	@. u = uₙ + RK.b[1] * Δt * stages[1] + RK.b[2] * Δt * stages[2]#   - RK.b[1] * Δt * jstages[1]  - RK.b[2] * Δt * jstages[2] + RK.bh[1] * Δt * jstages[1]  + RK.bh[2] * Δt * jstages[2] 
    end
end


function (::RKLIMEXZ{3})(res, uₙ, Δt, f1!, f2!, du, du_tmp, u, p, t, stages, ustages, jstages, stage, RK, lin_du_tmp, lin_du_tmp1, workspace, assume_p_const)
	@trixi_timeit timer() "F! function inside" F!(du, u, p) = f1!(du, u, p, t) ## parabolic
 	invdt = inv(RK.ah[stage,stage] * Δt)
    if stage == 1			
   	@trixi_timeit timer() "Jacobian inside stage1" J = JacobianOperator(F!, du, uₙ, p, assume_p_const = assume_p_const)
	@trixi_timeit timer() "LM Operator inside stage 1" M = LMROperator(J, invdt)
	@. res = invdt * RK.d[stage] * uₙ
	 @trixi_timeit timer() "krylov solve stage 1" krylov_solve!(workspace, M, res, atol = 1e-6, rtol = 1e-6)
	@. jstages[stage] = workspace.x
	@. res = uₙ+ jstages[stage]/RK.ah[stage,stage] + -1/RK.ah[stage,stage]*RK.d[stage]* uₙ
	@. ustages[stage] = res
	f2!(du, res, p, t + RK.c[stage] * Δt)
    	f1!(du_tmp, res, p, t + RK.c[stage] * Δt)
	@. stages[stage] = du + du_tmp
    elseif stage == 2
	@trixi_timeit timer() "Jacobian inside stage 2" J = JacobianOperator(F!, du, ustages[stage-1], p, assume_p_const = assume_p_const)
	@trixi_timeit timer() "LM Operator inside stage 2" M = LMROperator(J, invdt)
	@. res = invdt * RK.d[stage] * uₙ - invdt*RK.ah[stage,stage] * RK.gamma[stage,1] *( RK.d[1] *  uₙ - jstages[1]) + RK.a[stage,1] * stages[1]
	@trixi_timeit timer() "krylov solve stage 2" krylov_solve!(workspace, M, res, jstages[stage-1], atol = 1e-6, rtol = 1e-6)
	@. jstages[stage] = workspace.x
	@. res = uₙ+ jstages[stage]/RK.ah[stage,stage] + -1/RK.ah[stage,stage]*RK.d[stage]* uₙ + RK.gamma[stage,1] * (RK.d[1] * uₙ - jstages[1]) 
	@. ustages[stage] = res
	f2!(du, res, p, t + RK.c[stage] * Δt)
        f1!(du_tmp, res, p, t + RK.c[stage] * Δt)
	@. stages[stage] = du + du_tmp
    elseif stage == 3
	@trixi_timeit timer() "Jacobian inside stage 3" J = JacobianOperator(F!, du, ustages[stage-1], p, assume_p_const = assume_p_const)
	@trixi_timeit timer() "LM Operator inside stage 3" M = LMROperator(J, invdt)
	@. res = invdt * RK.d[stage] * uₙ - invdt*RK.ah[stage,stage] * (RK.gamma[stage,1] *( RK.d[1] *  uₙ - jstages[1]) + RK.gamma[stage,2] * (RK.d[2] * uₙ  - jstages[2])) + RK.a[stage,1] * stages[1] + RK.a[stage,2] * stages[2]
	@trixi_timeit timer() "krylov solve stage 3" krylov_solve!(workspace, M, res, jstages[stage-1], atol = 1e-6, rtol = 1e-6)
	@. jstages[stage] = workspace.x
	@. res = uₙ+ jstages[stage]/RK.ah[stage,stage] + -1/RK.ah[stage,stage]*RK.d[stage]* uₙ + RK.gamma[stage,1] *( RK.d[1] *  uₙ - jstages[1]) + RK.gamma[stage,2] * (RK.d[2] * uₙ  - jstages[2] )
	@. ustages[stage] = res
	f2!(du, res, p, t + RK.c[stage] * Δt)
        f1!(du_tmp, res, p, t + RK.c[stage] * Δt)
	@. stages[stage] = du + du_tmp
#	stage = 1	
#	@trixi_timeit timer() "Jacobian inside stage1" J = JacobianOperator(F!, du, ustages[stage], p, assume_p_const = assume_p_const)
#	mul!(jstages[stage], J, ustages[stage])	
#	stage = 2	
#	@trixi_timeit timer() "Jacobian inside stage1" J = JacobianOperator(F!, du, ustages[stage], p, assume_p_const = assume_p_const)
#	mul!(jstages[stage], J, ustages[stage])	
#	stage = 3	
#	@trixi_timeit timer() "Jacobian inside stage1" J = JacobianOperator(F!, du, ustages[stage], p, assume_p_const = assume_p_const)
#	mul!(jstages[stage], J, ustages[stage])	
	@. u = uₙ + RK.b[1] * Δt * stages[1] + RK.b[2] * Δt * stages[2] + RK.b[3] * Δt * stages[3] #- RK.b[1] * Δt * jstages[1]  - RK.b[2] * Δt * jstages[2] - RK.b[3] * Δt * jstages[3]   + RK.bh[1] * Δt * jstages[1]  + RK.bh[2] * Δt * jstages[2] + RK.bh[3] * Δt * jstages[3]
    end
end


function (::RKLIMEXZ{4})(res, uₙ, Δt, f1!, f2!, du, du_tmp, u, p, t, stages, ustages, jstages, stage, RK, lin_du_tmp, lin_du_tmp1, workspace, assume_p_const)
	@trixi_timeit timer() "F! function inside" F!(du, u, p) = f1!(du, u, p, t) ## parabolic
 	invdt = inv(RK.ah[stage,stage] * Δt)
    if stage == 1			
   	@trixi_timeit timer() "Jacobian inside stage1" J = JacobianOperator(F!, du, uₙ, p, assume_p_const = assume_p_const)
	@trixi_timeit timer() "LM Operator inside stage 1" M = LMROperator(J, invdt)
	@. res = invdt * RK.d[stage] * uₙ
	 @trixi_timeit timer() "krylov solve stage 1" krylov_solve!(workspace, M, res, atol = 1e-6, rtol = 1e-6)
	@. jstages[stage] = workspace.x
	@. res = uₙ + jstages[stage]/RK.ah[stage,stage] - 1/RK.ah[stage,stage]*RK.d[stage]* uₙ
	@. ustages[stage] = res
	f2!(du, res, p, t + RK.c[stage] * Δt)
    	f1!(du_tmp, res, p, t + RK.c[stage] * Δt)
	@. stages[stage] = du + du_tmp
    elseif stage == 2
	@trixi_timeit timer() "Jacobian inside stage 2" J = JacobianOperator(F!, du, ustages[stage-1], p, assume_p_const = assume_p_const)
	@trixi_timeit timer() "LM Operator inside stage 2" M = LMROperator(J, invdt)
	@. res = invdt * RK.d[stage] * uₙ - invdt*RK.ah[stage,stage] * RK.gamma[stage,1] *(RK.d[1] *  uₙ - jstages[1]) + RK.a[stage,1] * stages[1]
	@trixi_timeit timer() "krylov solve stage 2" krylov_solve!(workspace, M, res, jstages[stage-1], atol = 1e-6, rtol = 1e-6)
	@. jstages[stage] = workspace.x
	@. res = uₙ + jstages[stage]/RK.ah[stage,stage] - 1/RK.ah[stage,stage]*RK.d[stage]*uₙ + RK.gamma[stage,1] * (RK.d[1] * uₙ - jstages[1]) 
	@. ustages[stage] = res
	f2!(du, res, p, t + RK.c[stage] * Δt)
        f1!(du_tmp, res, p, t + RK.c[stage] * Δt)
	@. stages[stage] = du + du_tmp
    elseif stage == 3
	@trixi_timeit timer() "Jacobian inside stage 3" J = JacobianOperator(F!, du, ustages[stage-1], p, assume_p_const = assume_p_const)
	@trixi_timeit timer() "LM Operator inside stage 3" M = LMROperator(J, invdt)
	@. res = invdt * RK.d[stage] * uₙ - invdt*RK.ah[stage,stage] * (RK.gamma[stage,1] *( RK.d[1] *  uₙ - jstages[1]) + RK.gamma[stage,2] * (RK.d[2] * uₙ  - jstages[2])) + RK.a[stage,1] * stages[1] + RK.a[stage,2] * stages[2]
	@trixi_timeit timer() "krylov solve stage 3" krylov_solve!(workspace, M, res, jstages[stage-1], atol = 1e-6, rtol = 1e-6)
	@. jstages[stage] = workspace.x
	@. res = uₙ + jstages[stage]/RK.ah[stage,stage] - 1/RK.ah[stage,stage]*RK.d[stage]* uₙ + RK.gamma[stage,1] *(RK.d[1] *  uₙ - jstages[1]) + RK.gamma[stage,2] * (RK.d[2] * uₙ  - jstages[2])
	@. ustages[stage] = res
	f2!(du, res, p, t + RK.c[stage] * Δt)
        f1!(du_tmp, res, p, t + RK.c[stage] * Δt)
	@. stages[stage] = du + du_tmp	
    elseif stage == 4
	@trixi_timeit timer() "Jacobian inside stage 3" J = JacobianOperator(F!, du, ustages[stage-1], p, assume_p_const = assume_p_const)
	@trixi_timeit timer() "LM Operator inside stage 3" M = LMROperator(J, invdt)
	@. res = invdt * RK.d[stage] * uₙ - invdt*RK.ah[stage,stage] * (RK.gamma[stage,1] *( RK.d[1] *  uₙ - jstages[1]) + RK.gamma[stage,2] * (RK.d[2] * uₙ  - jstages[2]) + RK.gamma[stage,3] * (RK.d[3] * uₙ - jstages[3])) + RK.a[stage,1] * stages[1] + RK.a[stage,2] * stages[2] + RK.a[stage,3] * stages[3]
	@trixi_timeit timer() "krylov solve stage 4" krylov_solve!(workspace, M, res, jstages[stage-1], atol = 1e-6, rtol = 1e-6)
	@. jstages[stage] = workspace.x
	@. res = uₙ+ jstages[stage]/RK.ah[stage,stage] - 1/RK.ah[stage,stage]*RK.d[stage]* uₙ + RK.gamma[stage,1] *( RK.d[1] *  uₙ - jstages[1]) + RK.gamma[stage,2] * (RK.d[2] * uₙ  - jstages[2]) + RK.gamma[stage,3] * (RK.d[3] * uₙ  - jstages[3] )
	@. ustages[stage] = res
	f2!(du, res, p, t + RK.c[stage] * Δt)
        f1!(du_tmp, res, p, t + RK.c[stage] * Δt)
	@. stages[stage] = du + du_tmp

	stage = 1	
	@trixi_timeit timer() "Jacobian inside stage1" J = JacobianOperator(F!, du, uₙ ,p, assume_p_const = assume_p_const)
	mul!(jstages[stage], J, ustages[stage])	
	stage = 2	
	@trixi_timeit timer() "Jacobian inside stage1" J = JacobianOperator(F!, du, ustages[stage-1], p, assume_p_const = assume_p_const)
	mul!(jstages[stage], J, ustages[stage])	
	stage = 3	
	@trixi_timeit timer() "Jacobian inside stage1" J = JacobianOperator(F!, du, ustages[stage-1], p, assume_p_const = assume_p_const)
	mul!(jstages[stage], J, ustages[stage])	
	stage = 4	
	@trixi_timeit timer() "Jacobian inside stage1" J = JacobianOperator(F!, du, ustages[stage-1], p, assume_p_const = assume_p_const)
	mul!(jstages[stage], J, ustages[stage])	
	@. u = uₙ + RK.b[1] * Δt * stages[1] + RK.b[2] * Δt * stages[2] + RK.b[3] * Δt * stages[3] + RK.b[4] * Δt * stages[4] - RK.b[1] * Δt * jstages[1] - RK.b[2] * Δt * jstages[2] - RK.b[3] * Δt * jstages[3]  - RK.b[4] * Δt * jstages[4]  + RK.bh[1] * Δt * jstages[1]  + RK.bh[2] * Δt * jstages[2] + RK.bh[3] * Δt * jstages[3] + RK.bh[4] * Δt * jstages[4]

    end
end


function (::RKLIMEX{3})(res, uₙ, Δt, f1!, f2!, du, du_tmp, u, p, t, stages, ustages, jstages, stage, RK, lin_du_tmp, lin_du_tmp1, workspace, assume_p_const)
	F!(du, u, p) = f1!(du, u, p, t) ## parabolic
    if stage == 1
        # Stage 1:
	## f2 is the conservative part
	## f1 is the parabolic part
   	J = JacobianOperator(F!, du, uₙ, p, assume_p_const = assume_p_const)
   	M = LMOperator(J, RK.ah[stage,stage] * Δt)
	krylov_solve!(workspace, M, uₙ, atol = 1e-6, rtol = 1e-6)
	@. u = workspace.x
   	J = JacobianOperator(F!, du, u, p, assume_p_const = assume_p_const)
	mul!(jstages[stage], J, u)	
	f2!(du, workspace.x, p, t + RK.c[stage] * Δt)
        f1!(du_tmp, workspace.x, p, t + RK.c[stage] * Δt)
	@. stages[stage] = du + du_tmp - jstages[stage]
	@. ustages[stage] = u	
#        @. u = uₙ + RK.b[1] * Δt * stages[1]
	elseif stage == 2
	J = JacobianOperator(F!, du, ustages[1], p, assume_p_const = assume_p_const)
	M = LMOperator(J, RK.ah[stage,stage] * Δt)
	mul!(lin_du_tmp, M.J, uₙ)
#	mul!(lin_du_tmp1, J, u)	
	@. res = uₙ + RK.a[stage,1] * Δt * stages[1] - Δt * RK.ah[stage,1] * jstages[1]  
	krylov_solve!(workspace, M, res, atol = 1e-6, rtol = 1e-6)
	@. u = workspace.x
   	J = JacobianOperator(F!, du, u, p, assume_p_const = assume_p_const)
	mul!(jstages[stage], J, u)	
	f2!(du, workspace.x, p, t + RK.c[stage] * Δt)
	f1!(du_tmp, workspace.x, p, t + RK.c[stage] * Δt)
	@. stages[stage] = du + du_tmp - jstages[stage]
	@. ustages[stage] = u	

	elseif stage == 3
	J = JacobianOperator(F!, du, ustages[2], p, assume_p_const = assume_p_const)
	M = LMOperator(J, RK.ah[stage,stage] * Δt)
	mul!(lin_du_tmp, M.J, uₙ)
	@. res = uₙ + RK.a[stage,1] * Δt * stages[1] + RK.a[stage,2] * Δt * stages[2] - Δt * RK.ah[stage,1] * jstages[1] - Δt * RK.ah[stage,2] * jstages[2]  
	a = krylov_solve!(workspace, M, res, atol = 1e-6, rtol = 1e-6)
	@. u = workspace.x
   	J = JacobianOperator(F!, du, u, p, assume_p_const = assume_p_const)
	mul!(jstages[stage], J, u)	
	f2!(du, workspace.x, p, t + RK.c[stage] * Δt)
        f1!(du_tmp, workspace.x, p, t + RK.c[stage] * Δt)
	@. stages[stage] = du + du_tmp - jstages[stage]
	@. ustages[stage] = u	

	@. u = uₙ + RK.b[1] * Δt * stages[1] + RK.b[2] * Δt * stages[2] + RK.b[3] * Δt * stages[3] -  RK.bh[1] * Δt * jstages[1]  -  RK.bh[2] * Δt * jstages[2]  -  RK.bh[3] * Δt * jstages[3]  
	end
end


function (::RKLIMEX{4})(res, uₙ, Δt, f1!, f2!, du, du_tmp, u, p, t, stages, ustages, jstages, stage, RK, lin_du_tmp, lin_du_tmp1, workspace)
	F!(du, u, p) = f1!(du, u, p, t) ## parabolic
    if stage == 1
        # Stage 1:
	## f2 is the conservative part
	## f1 is the parabolic part
   	J = JacobianOperator(F!, du, uₙ, p)
   	M = LMOperator(J, RK.ah[stage,stage] * Δt)
	krylov_solve!(workspace, M, uₙ, atol = 1e-6, rtol = 1e-6)
	@. u = workspace.x
   	J = JacobianOperator(F!, du, u, p)
	mul!(jstages[stage], J, u)	
	f2!(du, workspace.x, p, t + RK.c[stage] * Δt)
        f1!(du_tmp, workspace.x, p, t + RK.c[stage] * Δt)
	@. stages[stage] = du + du_tmp - jstages[stage]
	@. ustages[stage] = u	
#        @. u = uₙ + RK.b[1] * Δt * stages[1]
	elseif stage == 2
	J = JacobianOperator(F!, du, ustages[1], p)
	M = LMOperator(J, RK.ah[stage,stage] * Δt)
	mul!(lin_du_tmp, M.J, uₙ)
#	mul!(lin_du_tmp1, J, u)	
	@. res = uₙ + RK.a[stage,1] * Δt * stages[1] - Δt * RK.ah[stage,1] * jstages[1]  
	krylov_solve!(workspace, M, res, atol = 1e-6, rtol = 1e-6)
	@. u = workspace.x
   	J = JacobianOperator(F!, du, u, p)
	mul!(jstages[stage], J, u)	
	f2!(du, workspace.x, p, t + RK.c[stage] * Δt)
	f1!(du_tmp, workspace.x, p, t + RK.c[stage] * Δt)
	@. stages[stage] = du + du_tmp - jstages[stage]
	@. ustages[stage] = u	

	elseif stage == 3
	J = JacobianOperator(F!, du, ustages[2], p)
	M = LMOperator(J, RK.ah[stage,stage] * Δt)
	mul!(lin_du_tmp, M.J, uₙ)
	@. res = uₙ + RK.a[stage,1] * Δt * stages[1] + RK.a[stage,2] * Δt * stages[2] - Δt * RK.ah[stage,1] * jstages[1] - Δt * RK.ah[stage,2] * jstages[2]  
	a = krylov_solve!(workspace, M, res, atol = 1e-6, rtol = 1e-6)
	@. u = workspace.x
   	J = JacobianOperator(F!, du, u, p)
	mul!(jstages[stage], J, u)	
	f2!(du, workspace.x, p, t + RK.c[stage] * Δt)
        f1!(du_tmp, workspace.x, p, t + RK.c[stage] * Δt)
	@. stages[stage] = du + du_tmp - jstages[stage]
	@. ustages[stage] = u	

	elseif stage == 4
	J = JacobianOperator(F!, du, ustages[3], p)
	M = LMOperator(J, RK.ah[stage,stage] * Δt)
	mul!(lin_du_tmp, M.J, uₙ)
	@. res = uₙ + RK.a[stage,1] * Δt * stages[1] + RK.a[stage,2] * Δt * stages[2] + RK.a[stage,3] * Δt * stages[3]- Δt * RK.ah[stage,1] * jstages[1] - Δt * RK.ah[stage,2] * jstages[2]  - Δt * RK.ah[stage,3] * jstages[3]
	a = krylov_solve!(workspace, M, res, atol = 1e-6, rtol = 1e-6)
	@. u = workspace.x
   	J = JacobianOperator(F!, du, u, p)
	mul!(jstages[stage], J, u)	
	f2!(du, workspace.x, p, t + RK.c[stage] * Δt)
        f1!(du_tmp, workspace.x, p, t + RK.c[stage] * Δt)
	@. stages[stage] = du + du_tmp - jstages[stage]
	@. ustages[stage] = u	
	@. u = uₙ + RK.b[1] * Δt * stages[1] + RK.b[2] * Δt * stages[2] + RK.b[3] * Δt * stages[3] + RK.b[4] * Δt * stages[4] -  RK.bh[1] * Δt * jstages[1]  -  RK.bh[2] * Δt * jstages[2]  -  RK.bh[3] * Δt * jstages[3]  -  RK.bh[4] * Δt * jstages[4]
	end
end

stages(::SimpleLinearImplicitExplicitAlgorithm{N}) where {N} = N

function nonlinear_problem(alg::SimpleLinearImplicitExplicitAlgorithm, f2::F2) where {F2}
    return (res, u, (uₙ, Δt, f1, du, du_tmp, p, t, stages, stage, RK, J, lin_du_tmp, lin_du_tmp1)) -> alg(res, uₙ, Δt, f1, f2, du, du_tmp, u, p, t, stages, stage, RK, J, lin_du_tmp, lin_du_tmp1)
end

mutable struct SimpleLinearImplicitExplicitOptions{Callback}
    callback::Callback # callbacks; used in Trixi.jl
    adaptive::Bool # whether the algorithm is adaptive; ignored
    dtmax::Float64 # ignored
    maxiters::Int # maximal number of time steps
    tstops::Vector{Float64} # tstops from https://diffeq.sciml.ai/v6.8/basics/common_solver_opts/#Output-Control-1; ignored
    verbose::Int
    algo::Symbol
    assume_p_const::Bool
    krylov_kwargs::Any
end

function RKTableau(alg::RKLSSPIMEX332)
return RKLSSPIMEX332Tableau()
end

function RKTableau(alg::AGSA432Z)
return AGSA432ZTableau()
end

function RKTableau(alg::AGSA432)
return AGSA432Tableau()
end

function RKTableau(alg::LSDIRK222)
return LSDIRK222Tableau()
end

function RKTableau(alg::RKLSSPIMEX332Z)
return RKLSSPIMEX332ZTableau()
end

function LSDIRK222Tableau()

    nstage = 2
	gamma = 1 - sqrt(2)/2
	sigma = 1/(2*gamma)
    a = zeros(Float64, nstage, nstage)
    a[2, 1] = gamma
    
    b = zeros(Float64, nstage)
    b[1] = 1 - gamma
    b[2] = 1

    c = zeros(Float64, nstage)
    c[2] = gamma
    ah = zeros(Float64, nstage, nstage)
    ah[1, 1] = gamma
    ah[2, 2] = gamma
    ah[2, 1] = 1-gamma
    
    bh = zeros(Float64, nstage)
    bh[1] = 1-gamma
    bh[2] = 1

    ch = zeros(Float64, nstage)
    ch[1] = gamma 
    ch[2] = 1
    d = zeros(Float64, nstage)
    @. d = ch - c

    gamma = zeros(Float64, nstage, nstage)

    gamma = diagm(diag(ah).^(-1)) - inv(ah - a)

    return IMEXRKZButcher(a, b, c, ah, bh, ch,d, gamma)
end

function AGSA432ZTableau()

    nstage = 4
    a = zeros(Float64, nstage, nstage)
        a[2,1] = (-139833537) / 38613965
	a[3,1] = 85870407 / 49798258
	a[3,2] = (-121251843) / 1756367063
	a[4,2] = 1/6
	a[4,3] = 2/3
	a[4,1] = 1 - a[4,2] - a[4,3]
    
    b = zeros(Float64, nstage)
    b[1] = 1 - 1/6 - 2/3
    b[2] = 1/6
    b[3] = 2/3	 

    c = zeros(Float64, nstage)
	c[2] = a[2,1]
	c[3] = a[3,1] + a[3,2]
	c[4] = a[4,1] + a[4,2] + a[4,3]
    ah = zeros(Float64, nstage, nstage)
	ah[1,1] = 168999711 / 74248304
	ah[2,1] = 44004295 / 24775207
	ah[2,2] = 202439144 / 118586105
	ah[3,1] = (-6418119) / 169001713
	ah[3,2] = (-748951821) / 1043823139
	ah[3,3] = 12015439 / 183058594
	ah[4,2] = 1 / 3
	ah[4,3] = 0
	ah[4,1] = 1 - ah[2,2] - 1/3
	ah[4,4] = ah[2,2]
    
    bh = zeros(Float64, nstage)
	bh[1] = ah[4,1]
	bh[2] = ah[4,2]
	bh[3] = ah[4,3]
	bh[4] = ah[4,4]

    ch = zeros(Float64, nstage)
	ch[1] = ah[1,1]
	ch[2] = ah[2,1] + ah[2,2]
	ch[3] = ah[3,1] + ah[3,2] + ah[3,3]
	ch[4] = ah[4,1] + ah[4,2] + ah[4,3] + ah[4,4]
    d = zeros(Float64, nstage)
    @. d = ch - c

    gamma = zeros(Float64, nstage, nstage)

    gamma = diagm(diag(ah).^(-1)) - inv(ah - a)

    return IMEXRKZButcher(a, b, c, ah, bh, ch,d, gamma)
end

function RKLSSPIMEX332ZTableau()

    nstage = 3
    a = zeros(Float64, nstage, nstage)
    a[2, 1] = 0.5
    a[3, 1] = 0.5
    a[3, 2] = 0.5
    
    b = zeros(Float64, nstage)
    b[1] = 1/3
    b[2] = 1/3
    b[3] = 1/3	 

    c = zeros(Float64, nstage)
    c[2] = 0.5
    c[3] = 1.0
    ah = zeros(Float64, nstage, nstage)
    ah[1, 1] = 1/4
    ah[2, 2] = 1/4
    ah[3, 1] = 1/3
    ah[3, 2] = 1/3
    ah[3, 3] = 1/3
    
    bh = zeros(Float64, nstage)
    bh[1] = 1/3
    bh[2] = 1/3
    bh[3] = 1/3	 

    ch = zeros(Float64, nstage)
    ch[1] = 1/4
    ch[2] = 1/4
    ch[3] = 1.0
    d = zeros(Float64, nstage)
    @. d = ch - c

    gamma = zeros(Float64, nstage, nstage)

    gamma = diagm(diag(ah).^(-1)) - inv(ah - a)

    return IMEXRKZButcher(a, b, c, ah, bh, ch,d, gamma)
end
function RKLSSPIMEX332Tableau()

    nstage = 3
    a = zeros(Float64, nstage, nstage)
    a[2, 1] = 0.5
    a[3, 1] = 0.5
    a[3, 2] = 0.5
    
    b = zeros(Float64, nstage)
    b[1] = 1/3
    b[2] = 1/3
    b[3] = 1/3	 

    c = zeros(Float64, nstage)
    c[2] = 0.5
    c[3] = 1.0
    ah = zeros(Float64, nstage, nstage)
    ah[1, 1] = 1/4
    ah[2, 2] = 1/4
    ah[3, 1] = 1/3
    ah[3, 2] = 1/3
    ah[3, 3] = 1/3
    
    bh = zeros(Float64, nstage)
    bh[1] = 1/3
    bh[2] = 1/3
    bh[3] = 1/3	 

    ch = zeros(Float64, nstage)
	ch[1] = 1/4
    ch[2] = 1/4
    ch[3] = 1.0
    return IMEXRKButcher(a, b, c, ah, bh, ch)
end

function AGSA432Tableau()

    nstage = 4
    a = zeros(Float64, nstage, nstage)
        a[2,1] = (-139833537) / 38613965
	a[3,1] = 85870407 / 49798258
	a[3,2] = (-121251843) / 1756367063
	a[4,2] = 1/6
	a[4,3] = 2/3
	a[4,1] = 1 - a[4,2] - a[4,3]
    
    b = zeros(Float64, nstage)
    b[1] = 1 - 1/6 - 2/3
    b[2] = 1/6
    b[3] = 2/3	 

    c = zeros(Float64, nstage)
	c[2] = a[2,1]
	c[3] = a[3,1] + a[3,2]
	c[4] = a[4,1] + a[4,2] + a[4,3]
    ah = zeros(Float64, nstage, nstage)
	ah[1,1] = 168999711 / 74248304
	ah[2,1] = 44004295 / 24775207
	ah[2,2] = 202439144 / 118586105
	ah[3,1] = (-6418119) / 169001713
	ah[3,2] = (-748951821) / 1043823139
	ah[3,3] = 12015439 / 183058594
	ah[4,2] = 1 / 3
	ah[4,3] = 0
	ah[4,1] = 1 - ah[2,2] - 1/3
	ah[4,4] = ah[2,2]
    
    bh = zeros(Float64, nstage)
	bh[1] = ah[4,1]
	bh[2] = ah[4,2]
	bh[3] = ah[4,3]
	bh[4] = ah[4,4]

    ch = zeros(Float64, nstage)
	ch[1] = ah[1,1]
	ch[2] = ah[2,1] + ah[2,2]
	ch[3] = ah[3,1] + ah[3,2] + ah[3,3]
	ch[4] = ah[4,1] + ah[4,2] + ah[4,3] + ah[4,4]

    return IMEXRKButcher(a, b, c, ah, bh, ch)
end

function RKTableau(alg::RKLinearImplicitExplicitEuler)
    return LinearImplicitExplicitEulerTableau()
end

function LinearImplicitExplicitEulerTableau()

    nstage = 1
    a = zeros(Float64, nstage, nstage)
    a[1, 1] = 1

    b = zeros(Float64, nstage)
    b[1] = 1

    c = zeros(Float64, nstage)
    c[1] = 1
    return DIRKButcher(a, b, c)
end


function SimpleLinearImplicitExplicitOptions(callback, tspan; maxiters=typemax(Int), verbose=0, krylov_algo=:gmres, assume_p_const = true, krylov_kwargs=(;), kwargs...)
    return SimpleLinearImplicitExplicitOptions{typeof(callback)}(
        callback, false, Inf, maxiters,
        [last(tspan)],
        verbose,
        krylov_algo,
	assume_p_const,
        krylov_kwargs,
    )
end

mutable struct SimpleLinearImplicitExplicit{
    RealT<:Real,uType,Params,Sol,F,F1,F2,M,Alg<:SimpleLinearImplicitExplicitAlgorithm,
    SimpleLinearImplicitExplicitOptions,RKTableau,
} <: AbstractTimeIntegrator
    u::uType
    du::uType
    du_tmp::uType
    lin_du_tmp::uType
    lin_du_tmp1::uType
    u_tmp::uType
    stages::NTuple{M,uType}
    ustages::NTuple{M,uType}
    jstages::NTuple{M,uType}
    res::uType
    t::RealT
    dt::RealT # current time step
    dtcache::RealT # ignored
    iter::Int # current number of time steps (iteration)
    p::Params # will be the semidiscretization from Trixi.jl
    sol::Sol # faked
    f::F #TODO: that should be sum of f1 and f2
    f1::F1 # `rhs!` parabolic
    f2::F2 # rhs! conservative
    alg::Alg # SimpleImplicitAlgorithm
    opts::SimpleLinearImplicitExplicitOptions
    finalstep::Bool # added for convenience
    RK::RKTableau
end


function Base.getproperty(integrator::SimpleLinearImplicitExplicit, field::Symbol)
    if field === :stats
        return (naccept=getfield(integrator, :iter),)
    end
    # general fallback
    return getfield(integrator, field)
end

function init(
    ode::ODEProblem, alg::SimpleLinearImplicitExplicitAlgorithm{N};
    dt, callback::Union{CallbackSet,Nothing}=nothing, kwargs...,
) where {N}
    u = copy(ode.u0)
    du = zero(u)
    res = zero(u)
    u_tmp = similar(u)
    stages = ntuple(_ -> similar(u), Val(N))
    ustages = ntuple(_ -> similar(u), Val(N))
    jstages = ntuple(_ -> similar(u), Val(N))
    t = first(ode.tspan)
    iter = 0
    integrator = SimpleLinearImplicitExplicit(
		u, du, copy(du),copy(du), copy(du), u_tmp,stages, ustages, jstages, res, t, dt, zero(dt), iter, ode.p,
        (prob=ode,), ode.f.f1, ode.f.f1, ode.f.f2, alg,
        SimpleLinearImplicitExplicitOptions(
            callback, ode.tspan;
            kwargs...,
        ), false, RKTableau(alg))

    # initialize callbacks
    if callback isa CallbackSet
        foreach(callback.continuous_callbacks) do cb
            throw(ArgumentError("Continuous callbacks are unsupported with the implicit time integration methods."))
        end
        foreach(callback.discrete_callbacks) do cb
            cb.initialize(cb, integrator.u, integrator.t, integrator)
        end
    end

    return integrator
end

# Fakes `solve`: https://diffeq.sciml.ai/v6.8/basics/overview/#Solving-the-Problems-1
function solve(
    ode::ODEProblem, alg::SimpleLinearImplicitExplicitAlgorithm;
    dt, callback=nothing, kwargs...,
)
    integrator = init(ode, alg, dt=dt, callback=callback; kwargs...)

    # Start actual solve
    return solve!(integrator)
end

function solve!(integrator::SimpleLinearImplicitExplicit)
    @unpack prob = integrator.sol

    integrator.finalstep = false
       @trixi_timeit timer() "kc Constructor" kc = KrylovConstructor(integrator.res)
   @trixi_timeit timer() "workspace" workspace = krylov_workspace(:gmres, kc)
    while !integrator.finalstep
        step!(integrator, workspace)
    end # "main loop" timer

    finalize_callbacks(integrator)

    return TimeIntegratorSolution(
        (first(prob.tspan), integrator.t),
        (prob.u0, integrator.u),
        integrator.sol.prob,
    )
end


function step!(integrator::SimpleLinearImplicitExplicit, workspace)
    @unpack prob = integrator.sol
    @unpack alg = integrator
    t_end = last(prob.tspan)
    callbacks = integrator.opts.callback

    @assert !integrator.finalstep
    if isnan(integrator.dt)
        error("time step size `dt` is NaN")
    end

    # if the next iteration would push the simulation beyond the end time, set dt accordingly
    if integrator.t + integrator.dt > t_end ||
       isapprox(integrator.t + integrator.dt, t_end)
        integrator.dt = t_end - integrator.t
        terminate!(integrator)
    end

    # one time step
    integrator.u_tmp .= integrator.u

    stage!(integrator, alg, workspace)

    integrator.u .= integrator.u_tmp

    integrator.iter += 1
    integrator.t += integrator.dt

    begin
        # handle callbacks
        if callbacks isa CallbackSet
            foreach(callbacks.discrete_callbacks) do cb
                if cb.condition(integrator.u, integrator.t, integrator)
                    cb.affect!(integrator)
                end
                return nothing
            end
        end
    end

    # respect maximum number of iterations
    return if integrator.iter >= integrator.opts.maxiters && !integrator.finalstep
        @warn "Interrupted. Larger maxiters is needed."
        terminate!(integrator)
    end
end

function stage!(integrator, alg::RKLIMEXZ, workspace)
#   @trixi_timeit timer() "F! function" F!(du, u, p) = integrator.f1(du, u, p, integrator.t) ## parabolic
#    @trixi_timeit timer() "Jacobian outside" J = JacobianOperator(F!, integrator.du, integrator.u, integrator.p)
#    @trixi_timeit timer() "LM Operator outside"	M = LMOperator(J, integrator.dt)

	for stage in 1:stages(alg)
        # Store the solution for each stage in stages
	## For a split Problem we need to compute rhs_conservative and rhs_parabolic
	@trixi_timeit timer() "alg"	alg(integrator.res, integrator.u, integrator.dt, integrator.f1, integrator.f2, integrator.du, integrator.du_tmp, integrator.u_tmp, integrator.p, integrator.t, integrator.stages, integrator.ustages, integrator.jstages, stage, integrator.RK, integrator.lin_du_tmp, integrator.lin_du_tmp1, workspace, integrator.opts.assume_p_const)

    end
end
function stage!(integrator, alg::RKLIMEX, workspace)
	for stage in 1:stages(alg)
        # Store the solution for each stage in stages
	## For a split Problem we need to compute rhs_conservative and rhs_parabolic
		alg(integrator.res, integrator.u, integrator.dt, integrator.f1, integrator.f2, integrator.du, integrator.du_tmp, integrator.u_tmp, integrator.p, integrator.t, integrator.stages, integrator.ustages, integrator.jstages, stage, integrator.RK, integrator.lin_du_tmp, integrator.lin_du_tmp1, workspace, integrator.opts.assume_p_const)
    end
end

# get a cache where the RHS can be stored
get_du(integrator::SimpleLinearImplicitExplicit) = integrator.du
get_tmp_cache(integrator::SimpleLinearImplicitExplicit) = (integrator.u_tmp,)

# some algorithms from DiffEq like FSAL-ones need to be informed when a callback has modified u
u_modified!(integrator::SimpleLinearImplicitExplicit, ::Bool) = false

# used by adaptive timestepping algorithms in DiffEq
function set_proposed_dt!(integrator::SimpleLinearImplicitExplicit, dt)
    return integrator.dt = dt
end

# Required e.g. for `glm_speed_callback`
function get_proposed_dt(integrator::SimpleLinearImplicitExplicit)
    return integrator.dt
end

# stop the time integration
function terminate!(integrator::SimpleLinearImplicitExplicit)
    integrator.finalstep = true
    return empty!(integrator.opts.tstops)
end

# used for AMR
function Base.resize!(integrator::SimpleLinearImplicitExplicit, new_size)
    resize!(integrator.u, new_size)
    resize!(integrator.du, new_size)
    return resize!(integrator.u_tmp, new_size)
end

