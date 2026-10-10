"""
    Asterion

Steady states `f(u, p) = 0` by pseudo-transient continuation with the Jacobian-free
Newton-Krylov solvers of Ariadne.jl.
"""
module Asterion

using Ariadne
using Ariadne: JacobianOperator, NewtonKrylovWorkspace, newton_krylov!, LaggedPreconditioner,
    AbstractPreconditioner, refresh!, variable_residual_ratio, SparseJacobian, assemble!,
    PerTaskParameters, NoLineSearch
using LinearAlgebra
using SparseArrays
using Printf
using SparseMatrixColorings: SparseMatrixColorings, GreedyColoringAlgorithm
import Enzyme

include("assembled_preconditioner.jl")
include("steady_state.jl")

export PseudoTransientNewtonKrylov, pseudo_transient!, PseudoTransientWorkspace
export ptc_start!, ptc_step!, ptc_reset_reference!
export SER, LodaresSER, steady_jacobian, jacobian_assembler
export AssembledJacobianPreconditioner, assembled_preconditioner, RowScaled

end # module Asterion
