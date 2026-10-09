# Asterion.jl

Asterion.jl computes steady states `f(u, p) = 0` of large nonlinear systems, e.g.,
discretized PDEs, by pseudo-transient continuation (PTC) with the Jacobian-free
Newton-Krylov solvers of Ariadne.jl. Each pseudo-time step is an implicit Euler step of
`du/dτ = σ f(u, p)`, solved by a few inexact Newton-Krylov iterations; the pseudo-time step
is controlled by the switched evolution relaxation (SER) of the residual.

It is developed in `libs/Asterion` of the Ariadne.jl repository.

```@docs
Asterion
```

## Solvers

```@docs
PseudoTransientNewtonKrylov
pseudo_transient!
PseudoTransientWorkspace
Asterion.PseudoTransientStats
ptc_start!
ptc_step!
ptc_reset_reference!
Asterion.successful
```

## CFL strategies

```@docs
Asterion.AbstractCFLStrategy
SER
LodaresSER
Asterion.update_cfl
Asterion.reject_cfl
Asterion.cfl_too_small
Asterion.limit_cycle!
```

## Preconditioners from assembled Jacobians

```@docs
AssembledJacobianPreconditioner
assembled_preconditioner
Asterion.AssembledJacobianBuilder
RowScaled
```

## Internals and adjoints

```@docs
Asterion.PseudoTransientResidual
Asterion.PseudoTransientParameters
steady_jacobian
jacobian_assembler
```
