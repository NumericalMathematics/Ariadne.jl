# Theseus.jl

Theseus.jl provides implicit and implicit-explicit (IMEX) time integration methods that use
Ariadne.jl's Newton–Krylov solver internally.  All methods implement the
[DifferentialEquations.jl integrator interface](https://docs.sciml.ai/DiffEqDocs/stable/basics/integrator/)
and can be used with `ODEProblem` (and `SplitODEProblem` for IMEX).

## Nonlinear Implicit Methods

These single-step methods solve one nonlinear system per stage via Newton–Krylov.
They accept an `ODEProblem` and the keyword argument `dt` (fixed time step).

```@docs
Theseus.ImplicitEuler
Theseus.ImplicitMidpoint
Theseus.ImplicitTrapezoid
Theseus.TRBDF2
```

## Diagonally Implicit Runge–Kutta (DIRK) Methods

DIRK methods use a lower-triangular Butcher tableau.  Each implicit stage requires
one Newton–Krylov solve.  They accept an `ODEProblem`.

```@docs
Theseus.LobattoIIIA2
Theseus.Crouzeix32
Theseus.DIRK43
```

## Implicit–Explicit (IMEX) Runge–Kutta Methods

IMEX methods split the right-hand side into a stiff part ``f_1`` and a non-stiff
part ``f_2``.  They accept a `SplitODEProblem(f1!, f2!, u0, tspan)`.

### Type I methods (Pareschi–Russo)

```@docs
Theseus.SP111
Theseus.H222
Theseus.SSP2222
Theseus.SSP2322
Theseus.SSP2332
Theseus.SSP3332
Theseus.SSP3433
```

### Type II methods (Ascher–Ruuth–Spiteri)

```@docs
Theseus.HT222
Theseus.ARS111
Theseus.ARS222
Theseus.ARS233
Theseus.ARS443
```

## Rosenbrock-W Methods

Rosenbrock-W methods linearise the implicit system and solve one linear system
(via a Krylov method) per stage instead of a full Newton solve.  The Jacobian
approximation makes them *W methods*: the exact Jacobian is not required.
They accept an `ODEProblem`.

```@docs
Theseus.SSPKnoth
Theseus.ROS2
```
