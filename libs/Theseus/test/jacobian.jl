using Test
using Theseus
using LinearAlgebra: I, Diagonal

@testset "jacobian" begin
    # du/dt = -p u²
    f!(du, u, p, t) = (du .= .-p .* u .^ 2; nothing)
    uₙ = [1.0, 2.0]
    p = [3.0, 5.0]
    Δt = 0.1
    ode = ODEProblem(f!, uₙ, (0.0, 1.0), p)
    ∂f = Diagonal(-2 .* p .* uₙ)

    # res = uₙ + Δt f(u) - u
    @test Theseus.jacobian(Theseus.ImplicitEuler(), ode, Δt) ≈ Δt * ∂f - I
    # res = uₙ + Δt f((uₙ + u) / 2) - u, at u = uₙ
    @test Theseus.jacobian(Theseus.ImplicitMidpoint(), ode, Δt) ≈ Δt / 2 * ∂f - I
    # res = uₙ + Δt / 2 (f(uₙ) + f(u)) - u
    @test Theseus.jacobian(Theseus.ImplicitTrapezoid(), ode, Δt) ≈ Δt / 2 * ∂f - I

    # TR-BDF2: the first stage is a trapezoidal step of size γΔt,
    # the second stage has the factor γ₂ Δt in front of f(u)
    γ = 2 - √2
    γ₂ = (1 - γ) / (2 - γ)
    @test Theseus.jacobian(Theseus.TRBDF2(), ode, Δt) ≈ γ / 2 * Δt * ∂f - I
    @test Theseus.jacobian(Theseus.TRBDF2(), ode, Δt; stage = 2) ≈ γ₂ * Δt * ∂f - I
end
