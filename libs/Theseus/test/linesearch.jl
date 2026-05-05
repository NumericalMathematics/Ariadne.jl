using Theseus, Ariadne
using Test

function rhs_nonstiff!(du, u, parameters, t)
    u1, u2 = u
    du[1] = u2 - u1 - u1^2
    du[2] = -2 * u2
    return nothing
end

function rhs_stiff!(du, u, parameters, t)
        (; epsilon) = parameters
        u1, u2 = u
        du[1] = 0
        du[2] = (u1^2 - u2) / epsilon
        return nothing
end

@testset "Linesearch" begin
    ode = SplitODEProblem{true}(
            rhs_stiff!, rhs_nonstiff!,
            [1.5, 1.0], (0.0, 1.0),
            (; epsilon = 1.0e-11))

    @test_throws ErrorException("Newton did not converge") begin
        solve(ode, Theseus.ARS443(); 
              dt = 0.02,
              newton_kwargs = (; linesearch! = NoLineSearch())
        )
    end

    solve(ode, Theseus.ARS443(); 
        dt = 0.02,
        newton_kwargs = (; linesearch! = BacktrackingLineSearch())
    )
end
