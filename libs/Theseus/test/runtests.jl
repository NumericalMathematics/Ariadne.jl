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

function compute_errors(prob, u_ana, alg, dts)
    errors = similar(dts)
    for (i, dt) in enumerate(dts)
        sol = @inferred solve(prob, alg; dt, adaptive = false)
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
        end

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
        end
    end
end