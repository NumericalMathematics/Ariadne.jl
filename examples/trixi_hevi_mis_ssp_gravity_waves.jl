# # Using an implicit-explicit (IMEX) Runge-Kutta solver based on Ariadne with Trixi.jl

using Trixi
using TrixiAtmo
using Theseus
using OrdinaryDiffEqSSPRK
using CairoMakie

# Notes:
# You must disable both Polyester and LoopVectorization for Enzyme to be able to differentiate Trixi.jl.
#
# LocalPreferences.jl
# ```toml
# [Trixi]
# loop_vectorization = false
# backend = "static"
# ```

@assert Trixi._PREFERENCE_THREADING !== :polyester
@assert !Trixi._PREFERENCE_LOOPVECTORIZATION

# First call to load callbacks
trixi_include(@__MODULE__, joinpath(TrixiAtmo.examples_dir(), "euler/dry_air/buoyancy", "elixir_potential_temperature_inertia_gravity_waves.jl"), sol = nothing);


# The standard Trixi.jl implementation of the slip wall boundary condition is not directly 
# compatible with this general splitting approach, since it is based on Toro's Riemann solver 
# which always returns boundary condition values for the entire right-hand side. 
# This function computes the boundary condition based on the surface flux function of the 
# explicit and implicit parts, where the splitting has been defined and thus accounts for it.
@inline function boundary_condition_slip_wall_simple(u_inner,
                                                     normal_direction::AbstractVector,
                                                     x, t,
                                                     surface_flux_function,
                                                     equations::CompressibleEulerPotentialTemperatureEquationsWithGravity2D)
    # normalize the outward pointing direction
    normal = normal_direction / Trixi.norm(normal_direction)

    # compute the normal momentum from
    # u = (rho, rho v1, rho v2, rho e)
    u_normal = normal[1] * u_inner[2] + normal[2] * u_inner[3]

    # create the "external" boundary solution state
    u_boundary = SVector(u_inner[1],
                         u_inner[2] - 2 * u_normal * normal[1],
                         u_inner[3] - 2 * u_normal * normal[2],
			 u_inner[4], u_inner[5])

    # calculate the boundary flux
    flux, _ = surface_flux_function(u_inner, u_boundary, normal_direction, equations)

    return flux
end

# The total flux is split into:
# - Fast (implicit/stiff) part: Contains all pressure-related terms responsible for acoustic waves.
#   Uses LMARS for surface fluxes and Kennedy-Gruber for volume fluxes.
# - Slow (explicit/non-stiff) part: Contains convective terms (advection).
@inline function flux_lmars_fast(u_ll, u_rr, normal_direction::AbstractVector,
                                 equations::CompressibleEulerPotentialTemperatureEquationsWithGravity2D)
    # Reference speed of sound
    a = 340
    # Unpack left and right state
    rho_ll, v1_ll, v2_ll, p_ll = cons2prim(u_ll, equations)
    rho_rr, v1_rr, v2_rr, p_rr = cons2prim(u_rr, equations)

    v_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2]
    v_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]
    v_ll_vert = v2_ll * normal_direction[2]
    v_rr_vert = v2_rr * normal_direction[2]

    norm_ = Trixi.norm(normal_direction)

    rho = 0.5f0 * (rho_ll + rho_rr)

    p_interface = 0.5f0 * (p_ll + p_rr) - 0.5f0 * a * rho * (v_rr - v_ll) / norm_
    v_interface = 0.5f0 * (v_ll + v_rr) - 1 / (2 * a * rho) * (p_rr - p_ll) * norm_
    v_interface_vert = 0.5f0 * (v_ll_vert + v_rr_vert) - 1 / (2 * a * rho) * (p_rr - p_ll) * norm_

    if (v_interface_vert > 0)
	f1 = u_ll[1] * v_interface_vert
	f4 = u_ll[4] * v_interface_vert
    else
	f1 = u_rr[1] * v_interface_vert
	f4 = u_rr[4] * v_interface_vert
    end

    flux = SVector(f1,
		   zero(eltype(u_ll)),
                   p_interface * normal_direction[2],
		   f4, zero(eltype(u_ll)))
    return flux, flux
end

 @inline Trixi.combine_conservative_and_nonconservative_fluxes(::typeof(flux_lmars_fast),
    equations::CompressibleEulerPotentialTemperatureEquationsWithGravity2D) = Trixi.True()


@inline function flux_lmars_slow(u_ll, u_rr, normal_direction::AbstractVector,
                                 equations::CompressibleEulerPotentialTemperatureEquationsWithGravity2D)
    # Reference speed of sound
    a = 340
    # Unpack left and right state
    rho_ll, v1_ll, v2_ll, p_ll = cons2prim(u_ll, equations)
    rho_rr, v1_rr, v2_rr, p_rr = cons2prim(u_rr, equations)

    v_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2]
    v_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]

    v_ll_hor = v1_ll * normal_direction[1]
    v_rr_hor = v1_rr * normal_direction[1]

    norm_ = Trixi.norm(normal_direction)

    rho = 0.5f0 * (rho_ll + rho_rr)

    p_interface = 0.5f0 * (p_ll + p_rr) - 0.5f0 * a * rho * (v_rr - v_ll) / norm_
    v_interface = 0.5f0 * (v_ll + v_rr) - 1 / (2 * a * rho) * (p_rr - p_ll) * norm_
    v_interface_hor = 0.5f0 * (v_ll_hor + v_rr_hor)# - 1 / (2 * a * rho) * (p_rr - p_ll) * norm_

    if (v_interface > 0)
	f2 = u_ll[2] * v_interface
	f3 = u_ll[3] * v_interface
    else
	f2 = u_rr[2] * v_interface
	f3 = u_rr[3] * v_interface
    end
    if (v_interface_hor > 0)
	f1 = u_ll[1] * v_interface_hor
	f4 = u_ll[4] * v_interface_hor
    else
	f1 = u_rr[1] * v_interface_hor
	f4 = u_rr[4] * v_interface_hor
    end
    f2 = f2 + p_interface * normal_direction[1]
    flux = SVector(f1, f2, f3, f4, zero(eltype(u_ll)))
    return flux, flux
end
 @inline Trixi.combine_conservative_and_nonconservative_fluxes(::typeof(flux_lmars_slow),
    equations::CompressibleEulerPotentialTemperatureEquationsWithGravity2D) = Trixi.True()

@inline function flux_kennedy_gruber_slow(u_ll, u_rr, normal_direction::AbstractVector,
                                          equations::CompressibleEulerPotentialTemperatureEquationsWithGravity2D)
    # Unpack left and right state
    rho_ll, v1_ll, v2_ll, p_ll = cons2prim(u_ll, equations)
    rho_rr, v1_rr, v2_rr, p_rr = cons2prim(u_rr, equations)

    # Average each factor of products in flux
    rho_avg = 0.5f0 * (rho_ll + rho_rr)
    v1_avg = 0.5f0 * (v1_ll + v1_rr)
    v2_avg = 0.5f0 * (v2_ll + v2_rr)
    v_dot_n_avg_hor = v1_avg * normal_direction[1]
    v_dot_n_avg = v1_avg * normal_direction[1] + v2_avg * normal_direction[2]
    p_avg = 0.5f0 * (p_ll + p_rr)
    rho_theta_ll = u_ll[4]
    rho_theta_rr = u_rr[4]
    rho_theta_avg = 0.5f0 * (rho_theta_ll+rho_theta_rr)

    # Calculate fluxes depending on normal_direction
    f1 = rho_avg * v_dot_n_avg_hor
    f2 = rho_avg * v_dot_n_avg * v1_avg + p_avg * normal_direction[1]
    f3 = rho_avg * v_dot_n_avg * v2_avg
    f4 = rho_theta_avg * v_dot_n_avg_hor
    flux = SVector(f1, f2, f3, f4, zero(eltype(u_ll)))
    return flux, flux 
end

 @inline Trixi.combine_conservative_and_nonconservative_fluxes(::typeof(flux_kennedy_gruber_slow),
    equations::CompressibleEulerPotentialTemperatureEquationsWithGravity2D) = Trixi.True()
@inline function flux_kennedy_gruber_fast(u_ll, u_rr, normal_direction::AbstractVector,
                                          equations::CompressibleEulerPotentialTemperatureEquationsWithGravity2D)
    # Unpack left and right state
    rho_ll, v1_ll, v2_ll, p_ll, phi_ll = cons2prim(u_ll, equations)
    rho_rr, v1_rr, v2_rr, p_rr, phi_rr = cons2prim(u_rr, equations)

    # Average each factor of products in flux
    rho_avg = 0.5f0 * (rho_rr + rho_ll)
    v1_avg = 0.5f0 * (v1_ll + v1_rr)
    v2_avg = 0.5f0 * (v2_ll + v2_rr)
    v_dot_n_avg = v2_avg * normal_direction[2]
    p_avg = 0.5f0 * (p_ll + p_rr)
    rho_theta_ll = u_ll[4]
    rho_theta_rr = u_rr[4]
    rho_theta_avg = 0.5f0 * (rho_theta_ll+rho_theta_rr)
    # Calculate fluxes depending on normal_direction
    f1 = rho_avg * v_dot_n_avg
    f2 = zero(eltype(u_ll))
    f3 = p_avg * normal_direction[2]
    f4 = rho_theta_avg * v_dot_n_avg

    gravity = rho_avg * (phi_rr - phi_ll)
    g2 = gravity * normal_direction[1]
    g3 = gravity * normal_direction[2] 

    return SVector(f1, f2 + 0.5f0 * g2, f3 + 0.5f0 * g3, f4, zero(eltype(u_ll))),
    SVector(f1, f2 - 0.5f0 * g2, f3 - 0.5f0 * g3, f4, zero(eltype(u_ll)))
end

 @inline Trixi.combine_conservative_and_nonconservative_fluxes(::typeof(flux_kennedy_gruber_fast),
    equations::CompressibleEulerPotentialTemperatureEquationsWithGravity2D) = Trixi.True()
volume_integral_nonstiff = VolumeIntegralFluxDifferencing(flux_kennedy_gruber_slow)
solver_nonstiff = DGSEM(polydeg = polydeg, surface_flux = flux_lmars_slow, volume_integral = volume_integral_nonstiff)

volume_integral_stiff = VolumeIntegralFluxDifferencing(flux_kennedy_gruber_fast)
solver_stiff = DGSEM(polydeg = polydeg, surface_flux =flux_lmars_fast, volume_integral = volume_integral_stiff)

boundary_conditions = (;
                       y_neg = boundary_condition_slip_wall_simple,
                       y_pos = boundary_condition_slip_wall_simple)

initial_condition = initial_condition_gravity_waves

coordinates_min = (0.0, 0.0)
coordinates_max = (300_000.0, 10_000.0)
trees_per_dimension = (60, 8)
mesh = P4estMesh(trees_per_dimension; polydeg = polydeg,
                 coordinates_min = coordinates_min, coordinates_max = coordinates_max,
                 periodicity = (true, false), initial_refinement_level = 0)

semi = SemidiscretizationHyperbolicSplit(mesh,
                                         (equations, equations),
                                         initial_condition,
                                         (solver_stiff, solver_nonstiff);
                                         boundary_conditions = (boundary_conditions,
                                                                boundary_conditions))

alive_callback = AliveCallback(analysis_interval = 10000)
stepsize_callback = nothing
cfl = 3.0
trixi_include(@__MODULE__, joinpath(TrixiAtmo.examples_dir(), "euler/dry_air/buoyancy", "elixir_potential_temperature_inertia_gravity_waves.jl"), sol = nothing, semi = semi, mesh = mesh, alive_callback = alive_callback, stepsize_callback = stepsize_callback);
###############################################################################
# run the simulation

sol = solve(
    ode,
    (Theseus.MISRK3(), SSPRK43());
    dt = 3.0, dt_fast = 0.78,# solve needs some value here but it will be overwritten by the stepsize_callback
    ode_default_options()..., callback = callbacks, adaptive = false,
);
