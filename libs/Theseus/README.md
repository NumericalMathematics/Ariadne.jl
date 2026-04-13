# Theseus.jl

[![Docs-dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://NumericalMathematics.github.io/Ariadne.jl/dev/)
[![Build Status](https://github.com/NumericalMathematics/Ariadne.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/NumericalMathematics/Ariadne.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](https://opensource.org/licenses/MIT)

Theseus is a time integration library for ordinary differential equations (ODEs), built as a companion package to Ariadne.jl. It provides implicit and semi-implicit Runge-Kutta methods that use Ariadne's matrix-free Newton-Krylov solver for the nonlinear stage equations.

Theseus is designed for time-dependent PDEs (e.g. with [Trixi.jl](https://github.com/trixi-framework/Trixi.jl)) where forming an explicit Jacobian is impractical, and where stiffness demands implicit treatment.
