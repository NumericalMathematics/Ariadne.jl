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
Theseus.CooperSayfy5
Theseus.CrouzeixRaviart34
Theseus.HairerWannerSDIRK4
Theseus.ESDIRK43SA2
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
Theseus.AGSA342
```

### Type II methods (Ascher–Ruuth–Spiteri and Kennedy–Carpenter)

```@docs
Theseus.HT222
Theseus.ARS111
Theseus.ARS222
Theseus.ARS233
Theseus.ARS443
Theseus.KenCarpARK324L2SA
Theseus.KenCarpARK436L2SA
Theseus.KenCarpARK437
Theseus.KenCarpARK548
Theseus.BHR553G1
Theseus.BHR553G2
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

## Newton options

The implicit stages of the DIRK and IMEX methods are solved by `newton_krylov!` with the
keyword arguments `newton_tol_abs = 1e-6`, `newton_tol_rel = 1e-6`,
`newton_max_niter = 50`, `newton_tol_step = 1e-10` (passed as `tol_step`), `krylov_algo`,
`krylov_kwargs`, and any further `newton_kwargs` (e.g., a line search) of `solve`.

By default (`newton_scaling = :jacobian`), the rows of the stage residuals are weighted by
their stiffness, so that rows of very different stiffness, e.g., a stiff relaxation term
next to a non-stiff equation, are all resolved to the Newton tolerance. Use
`newton_scaling = :none` to disable this, or pass a `norm` and a left preconditioner `M`
in `newton_kwargs`.

```@docs
Theseus.newton_scaling_kwargs
```

## Utilities

```@docs
Theseus.jacobian
```
