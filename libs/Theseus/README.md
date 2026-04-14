# Theseus.jl: Implicit time integration methods using Jacobian-free Newton-Krylov solvers from Ariadne.jl

[![Docs-dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://NumericalMathematics.github.io/Ariadne.jl/dev/)
[![Build Status](https://github.com/NumericalMathematics/Ariadne.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/NumericalMathematics/Ariadne.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](https://opensource.org/licenses/MIT)

This package is designed to solve ordinary differential equations of the form `u'(t) = f(t, u(t))` using fully/semi/linearly implicit time integration methods.
It relies on the Jacobian-free (Newton-) Krylov solvers from Ariadne.jl to solve the systems that arise at each time step.
Theseus.jl is designed for big problems like time-dependent PDEs (e.g., with [Trixi.jl](https://github.com/trixi-framework/Trixi.jl)) where forming an explicit Jacobian is impractical, and where stiffness demands implicit treatment.

Theseus.jl is designed to be largely compatible with the OrdinaryDiffEq.jl ecosystem; in particular, it uses the same `ODEProblem` specification and supports the same (discrete) callback interface.
The primary goal of Theseus.jl is to provide a flexible framework for experimenting with new methods, for teaching, and for exploring new approaches before incorporating them into larger application-focused packages.
