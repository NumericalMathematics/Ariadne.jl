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

    # With the defaults, the rows of the stage residual are weighted by their stiffness
    # (`newton_scaling = :jacobian`) and Newton stops at the rounding floor of the
    # residual (`newton_tol_step`)
    for dt in (0.02, 0.01, 0.005)
        u1, u2 = solve(ode, Theseus.ARS443(); dt).u[end]
        @test isapprox(u1, 1.5 * exp(-1); rtol = 1.0e-2)
        @test isapprox(u2, u1^2; rtol = 1.0e-6)
    end

    # Without the scaling, the residual of the second stage equation is ~1/ε larger than
    # the first one: the Newton tolerance is below the rounding floor of the stiff row, and
    # line searches cannot help
    @test_throws ErrorException("Newton did not converge") solve(
        ode, Theseus.ARS443(); dt = 0.02, newton_scaling = :none, newton_tol_step = 0.0,
        newton_kwargs = (; linesearch! = BacktrackingLineSearch())
    )

    # User-provided scaling of the residual (rows weighted in the norm of the termination
    # criterion and, as its inner product, in GMRES)
    scale = 1 / epsilon
    for dt in (0.02, 0.01), linesearch! in (NoLineSearch(), BacktrackingLineSearch())
        sol = solve(
            ode, Theseus.ARS443(); dt,
            newton_kwargs = (; norm = ScaledNorm((1.0, scale)), linesearch!)
        )
        u1, u2 = sol.u[end]
        @test isapprox(u1, 1.5 * exp(-1); rtol = 1.0e-2)
        @test isapprox(u2, u1^2; rtol = 1.0e-8)
    end
end
