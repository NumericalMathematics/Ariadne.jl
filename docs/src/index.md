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

### Parameters

```@docs
Ariadne.Forcing
Ariadne.Fixed
Ariadne.EisenstatWalker
```

### Sparse Jacobians

```@docs
SparseJacobian
assemble!
Ariadne.jacobian_sparsity
greedy_column_coloring
is_column_coloring
Ariadne.column_coloring_lower_bound
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