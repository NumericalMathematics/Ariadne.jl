using Theseus, Ariadne
using LinearAlgebra
using Test

# Stiff split problem from #135: u1' = u2 - u1 - u1^2, u2' = (u1^2 - u2) / ε - 2 u2.
# For ε → 0, u2 = u1^2 and u1' = -u1, i.e., u1(1) ≈ 1.5 exp(-1).
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

@testset "Newton keyword arguments (#135)" begin
    epsilon = 1.0e-11
    ode = SplitODEProblem{true}(
        rhs_stiff!, rhs_nonstiff!,
        [1.5, 1.0], (0.0, 1.0),
        (; epsilon)
    )

    # The residual of the second stage equation is ~1/ε larger than the first one. In the
    # Euclidean norm, the tolerance is below the rounding floor of the stiff row, and
    # line searches cannot help.
    @test_throws ErrorException("Newton did not converge") solve(
        ode, Theseus.ARS443(); dt = 0.02,
        newton_kwargs = (; linesearch! = BacktrackingLineSearch())
    )

    # Weight the rows of the residual by their stiffness, in the norm of the termination
    # criterion and (as left preconditioner) in GMRES
    scale = 1 / epsilon
    W = Diagonal([1.0, 1 / scale])
    for dt in (0.02, 0.01), linesearch! in (NoLineSearch(), BacktrackingLineSearch())
        sol = solve(
            ode, Theseus.ARS443(); dt,
            newton_kwargs = (; norm = ScaledNorm((1.0, scale)), M = J -> W, linesearch!)
        )
        u1, u2 = sol.u[end]
        @test isapprox(u1, 1.5 * exp(-1); rtol = 1.0e-2)
        @test isapprox(u2, u1^2; rtol = 1.0e-8)
    end
end
