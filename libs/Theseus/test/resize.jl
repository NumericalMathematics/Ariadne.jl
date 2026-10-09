using Test
using Theseus
import Theseus: CallbackSet
import Theseus.DiffEqBase: DiscreteCallback

# Grow the state from 2 to 4 entries after the third time step, as AMR does
function resize_callback()
    condition(u, t, integrator) = integrator.iter == 3
    function affect!(integrator)
        resize!(integrator, 4)
        integrator.u[3:4] .= (3.0, 4.0)
        return nothing
    end
    return CallbackSet(DiscreteCallback(condition, affect!; save_positions = (false, false)))
end

@testset "resize!" begin
    # Decoupled linear decay: the first two entries do not see the resize
    rhs!(du, u, p, t) = (du .= .-u; nothing)
    rhs_split!(du, u, p, t) = (du .= .-u ./ 2; nothing)
    ode = ODEProblem(rhs!, [1.0, 2.0], (0.0, 1.0))
    ode_split = SplitODEProblem(rhs_split!, rhs_split!, [1.0, 2.0], (0.0, 1.0))

    for (alg, prob) in (
            (Theseus.ImplicitEuler(), ode),
            (Theseus.TRBDF2(), ode),
            (Theseus.Crouzeix32(), ode),
            (Theseus.ROS2(), ode),
            (Theseus.ARS222(), ode_split),
        )
        @testset "$(nameof(typeof(alg)))" begin
            sol_ref = solve(prob, alg; dt = 0.1, adaptive = false)
            sol = solve(prob, alg; dt = 0.1, adaptive = false, callback = resize_callback())
            u = sol.u[end]
            @test length(u) == 4
            @test u[1:2] ≈ sol_ref.u[end] rtol = 1.0e-5
            @test all(isfinite, u)
            # Entries 3 and 4 decay with the same rate
            @test u[3] / u[4] ≈ 3 / 4 rtol = 1.0e-5
            @test 0 < u[3] < 3
        end
    end
end
