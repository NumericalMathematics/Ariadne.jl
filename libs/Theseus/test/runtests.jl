using Test
using LinearAlgebra: norm
using Statistics: mean

using Theseus

# Utility functions used for testing
function compute_eoc(dts, errors)
    eocs = similar(errors)
    eocs[begin] = 0 # no EOC defined for the first grid
    for idx in Iterators.drop(eachindex(errors, dts, eocs), 1)
        eocs[idx] = log(errors[idx] / errors[idx - 1]) /
                    log(dts[idx] / dts[idx - 1])
    end
    return mean(eocs[(begin + 1):end])
end

function compute_errors(prob, u_ana, alg, dts; kwargs...)
    errors = similar(dts)
    for (i, dt) in enumerate(dts)
        sol = @inferred solve(prob, alg;
                              dt, adaptive = false, kwargs...)
        u_num = sol.u[end]
        errors[i] = norm(u_num - u_ana)
    end
    return errors
end


@testset "Theseus" begin
    @testset "Convergence quadrature cos" begin
        ode = ODEProblem([0.0], (0.0, 1.0)) do du, u, p, t
            du[begin] = cos(t)
        end
        u_ana = [sin(ode.tspan[end])]

        @testset "DIRK methods" begin
            @testset "LobattoIIIA2" begin
                alg = Theseus.LobattoIIIA2()
                order = 2
                dts = 2.0 .^ (-2:-1:-6)
                errors = compute_errors(ode, u_ana, alg, dts)
                eoc = compute_eoc(dts, errors)
                @test isapprox(eoc, order; atol = 0.1)
            end

            @testset "Crouzeix32" begin
                alg = Theseus.Crouzeix32()
                order = 3 + 1 # Gaussian quadrature
                dts = 2.0 .^ (-2:-1:-6)
                errors = compute_errors(ode, u_ana, alg, dts)
                eoc = compute_eoc(dts, errors)
                @test isapprox(eoc, order; atol = 0.1)
            end

            @testset "DIRK43" begin
                alg = Theseus.DIRK43()
                order = 4
                dts = 2.0 .^ (-2:-1:-6)
                errors = compute_errors(ode, u_ana, alg, dts)
                eoc = compute_eoc(dts, errors)
                @test isapprox(eoc, order; atol = 0.1)
            end
        end # DIRK methods

        @testset "Rosenbrock methods" begin
            @testset "SSPKnoth" begin
                alg = Theseus.SSPKnoth()
                order = 2
                dts = 2.0 .^ (-2:-1:-6)
                errors = compute_errors(ode, u_ana, alg, dts)
                eoc = compute_eoc(dts, errors)
                @test_broken isapprox(eoc, order; atol = 0.1)
            end

            @testset "ROS2" begin
                alg = Theseus.ROS2()
                order = 2
                dts = 2.0 .^ (-2:-1:-6)
                errors = compute_errors(ode, u_ana, alg, dts)
                eoc = compute_eoc(dts, errors)
                @test_broken isapprox(eoc, order; atol = 0.1)
            end
        end # Rosenbrock methods
    end

    @testset "Convergence linear system" begin
        ode = ODEProblem([1.0, 0.0, 1.0], (0.0, 1.0)) do du, u, p, t
            du[1] = -u[2]
            du[2] = u[1]
            du[3] = -u[3]
            return nothing
        end
        u_ana = [cos(ode.tspan[end]),
                 sin(ode.tspan[end]),
                 exp(-ode.tspan[end])]

        @testset "DIRK methods" begin
            @testset "LobattoIIIA2" begin
                alg = Theseus.LobattoIIIA2()
                order = 2
                dts = 2.0 .^ (-2:-1:-6)
                errors = compute_errors(ode, u_ana, alg, dts)
                eoc = compute_eoc(dts, errors)
                @test isapprox(eoc, order; atol = 0.1)
            end

            @testset "Crouzeix32" begin
                alg = Theseus.Crouzeix32()
                order = 3
                dts = 2.0 .^ (-2:-1:-6)
                errors = compute_errors(ode, u_ana, alg, dts;
                                        krylov_tol_abs = 1.0e-8)
                eoc = compute_eoc(dts, errors)
                @test isapprox(eoc, order; atol = 0.1)
            end

            @testset "DIRK43" begin
                alg = Theseus.DIRK43()
                order = 4
                dts = 2.0 .^ (-2:-1:-6)
                errors = compute_errors(ode, u_ana, alg, dts)
                eoc = compute_eoc(dts, errors)
                @test_broken isapprox(eoc, order; atol = 0.1)
                # TODO: Why can't we get closer to machine precision?
                #=
                julia> dts = 2.0 .^ (-3:-1:-10)
                8-element Vector{Float64}:
                0.125
                0.0625
                0.03125
                0.015625
                0.0078125
                0.00390625
                0.001953125
                0.0009765625

                julia> errors = compute_errors(ode, u_ana, alg, dts; krylov_tol_abs = 1.0e-14)
                8-element Vector{Float64}:
                6.902787265585153e-5
                4.536884404070524e-6
                2.312461892802833e-7
                2.1474333974267422e-8
                6.248482209851633e-8
                7.765078477201435e-9
                7.009412413653584e-9
                1.453399550413022e-7
                =#
            end
        end # DIRK methods

        @testset "Rosenbrock methods" begin
            @testset "SSPKnoth" begin
                alg = Theseus.SSPKnoth()
                order = 2
                dts = 2.0 .^ (-2:-1:-6)
                errors = compute_errors(ode, u_ana, alg, dts)
                eoc = compute_eoc(dts, errors)
                @test isapprox(eoc, order; atol = 0.1)
            end

            @testset "ROS2" begin
                alg = Theseus.ROS2()
                order = 2
                dts = 2.0 .^ (-2:-1:-6)
                errors = compute_errors(ode, u_ana, alg, dts)
                eoc = compute_eoc(dts, errors)
                @test_broken isapprox(eoc, order; atol = 0.1)
                # TODO: Is this a second- or a third-order method?
            end
        end # Rosenbrock methods
    end
end