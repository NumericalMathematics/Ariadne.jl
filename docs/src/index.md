# Ariadne.jl

Newton Method using Krylov.jl (montoison-orban-2023)[@cite]

## API

```@docs
newton_krylov!
newton_krylov
NewtonKrylovWorkspace
```

### Line Searches

```@docs
Ariadne.LineSearches.AbstractLineSearch
NoLineSearch
BacktrackingLineSearch
```

### Statistics

```@docs
Ariadne.Stats
```

### Adjoints and implicit functions

```@docs
adjoint_solve
adjoint_gradient
TransposedOperator
TransposedPreconditioner
transpose_ldiv!
parameter_vjp
parameter_vjp!
ImplicitFunction
implicit_solve!
```

### Parameters

```@docs
Ariadne.Forcing
Ariadne.Fixed
Ariadne.EisenstatWalker
```

### Internal

```@docs
Ariadne.JacobianOperator
Ariadne.BatchedJacobianOperator
Ariadne.evaluate!
```

## Bibliography

```@bibliography
```