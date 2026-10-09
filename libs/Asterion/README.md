# Asterion.jl: Steady states by pseudo-transient continuation with Jacobian-free Newton-Krylov solvers from Ariadne.jl

[![Docs-dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://NumericalMathematics.github.io/Ariadne.jl/dev/)
[![Build Status](https://github.com/NumericalMathematics/Ariadne.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/NumericalMathematics/Ariadne.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](https://opensource.org/licenses/MIT)

This package computes steady states `f(u, p) = 0` of large nonlinear systems, e.g., discretized PDEs, by pseudo-transient continuation (PTC).
Each pseudo-time step is an implicit Euler step of `du/dτ = σ f(u, p)`, solved by a few inexact Newton-Krylov iterations with the Jacobian-free solvers of Ariadne.jl; the pseudo-time step is controlled by the switched evolution relaxation (SER) of the residual.
Preconditioners can be built from sparse Jacobians assembled by colored forward-mode AD with Enzyme.jl.

It is used with `pseudo_transient!`.
