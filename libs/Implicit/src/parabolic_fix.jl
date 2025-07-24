using Trixi: @threaded, eachelement, eachnode, eachvariable, get_node_vars, multiply_add_to_node_vars!, reset_du!, prolong2interfaces!, prolong2boundaries!, prolong2mortars!, calc_gradient_interface_flux!, calc_boundary_flux_gradients!, calc_mortar_flux!, apply_jacobian_parabolic!
using Trixi: TreeMesh, DG, Gradient, CompressibleNavierStokesDiffusion2D, @trixi_timeit, timer, nnodes
import Trixi: calc_gradient!

function calc_gradient!(gradients, u_transformed, t,
                        mesh::TreeMesh{2}, equations_parabolic::CompressibleNavierStokesDiffusion2D,
                        boundary_conditions_parabolic, dg::DG, parabolic_scheme,
                        cache, cache_parabolic)
    gradients_x, gradients_y = gradients

    # Reset du
    @trixi_timeit timer() "reset gradients" begin
        reset_du!(gradients_x, dg, cache)
        reset_du!(gradients_y, dg, cache)
    end

    # Calculate volume integral
    @trixi_timeit timer() "volume integral" begin
        @unpack derivative_dhat = dg.basis
        @threaded for element in eachelement(dg, cache)

            # Calculate volume terms in one element
            for j in eachnode(dg), i in eachnode(dg)
                u_node = get_node_vars(u_transformed, equations_parabolic, dg,
                                       i, j, element)

                for ii in eachnode(dg)
                    multiply_add_to_node_vars!(gradients_x, derivative_dhat[ii, i],
                                               u_node, equations_parabolic, dg,
                                               ii, j, element)
                end

                for jj in eachnode(dg)
                    multiply_add_to_node_vars!(gradients_y, derivative_dhat[jj, j],
                                               u_node, equations_parabolic, dg,
                                               i, jj, element)
                end
            end
        end
    end

    # Prolong solution to interfaces
    @trixi_timeit timer() "prolong2interfaces" begin
        prolong2interfaces!(cache_parabolic, u_transformed, mesh,
                            equations_parabolic, dg)
    end

    # Calculate interface fluxes
    @trixi_timeit timer() "interface flux" begin
        @unpack surface_flux_values = cache_parabolic.elements
        calc_gradient_interface_flux!(surface_flux_values, mesh, equations_parabolic,
                                      dg, parabolic_scheme,
                                      cache, cache_parabolic)
    end

    # Prolong solution to boundaries
    @trixi_timeit timer() "prolong2boundaries" begin
        prolong2boundaries!(cache_parabolic, u_transformed, mesh, equations_parabolic,
                            dg.surface_integral, dg)
    end

    # Calculate boundary fluxes
    @trixi_timeit timer() "boundary flux" begin
        calc_boundary_flux_gradients!(cache_parabolic, t,
                                      boundary_conditions_parabolic, mesh,
                                      equations_parabolic,
                                      dg.surface_integral, dg)
    end

    # Prolong solution to mortars
    # NOTE: This re-uses the implementation for hyperbolic terms in "dg_2d.jl"
    @trixi_timeit timer() "prolong2mortars" begin
        prolong2mortars!(cache, u_transformed, mesh, equations_parabolic,
                         dg.mortar, dg)
    end

    # Calculate mortar fluxes
    @trixi_timeit timer() "mortar flux" begin
        calc_mortar_flux!(surface_flux_values, mesh, equations_parabolic,
                          dg.mortar, dg.surface_integral, dg,
                          parabolic_scheme, Gradient(), cache)
    end

    # Calculate surface integrals
    @trixi_timeit timer() "surface integral" begin
    calc_surface_integral_parabolic!(cache_parabolic.elements.surface_flux_values, gradients_x, gradients_y,
                                           dg.basis.boundary_interpolation, dg, equations_parabolic,
                                           cache)
    end

    # Apply Jacobian from mapping to reference element
    @trixi_timeit timer() "Jacobian" begin
        apply_jacobian_parabolic!(gradients_x, mesh, equations_parabolic, dg,
                                  cache_parabolic)
        apply_jacobian_parabolic!(gradients_y, mesh, equations_parabolic, dg,
                                  cache_parabolic)
    end

    return nothing
end

function calc_surface_integral_parabolic!(surface_flux_values, gradients_x, gradients_y,
                                           boundary_interpolation, dg, equations_parabolic,
                                           cache)


        # Note that all fluxes have been computed with outward-pointing normal vectors.
        # Access the factors only once before beginning the loop to increase performance.
        # We also use explicit assignments instead of `+=` to let `@muladd` turn these
        # into FMAs (see comment at the top of the file).
        factor_1 = boundary_interpolation[1, 1]
        factor_2 = boundary_interpolation[nnodes(dg), 2]
        @threaded for element in eachelement(dg, cache)
            for l in eachnode(dg)
                for v in eachvariable(equations_parabolic)
                    # surface at -x
                    gradients_x[v, 1, l, element] = (gradients_x[v, 1, l, element] -
                                                     surface_flux_values[v, l, 1,
                                                                         element] *
                                                     factor_1)

                    # surface at +x
                    gradients_x[v, nnodes(dg), l, element] = (gradients_x[v, nnodes(dg),
                                                                          l, element] +
                                                              surface_flux_values[v, l,
                                                                                  2,
                                                                                  element] *
                                                              factor_2)

                    # surface at -y
                    gradients_y[v, l, 1, element] = (gradients_y[v, l, 1, element] -
                                                     surface_flux_values[v, l, 3,
                                                                         element] *
                                                     factor_1)

                    # surface at +y
                    gradients_y[v, l, nnodes(dg), element] = (gradients_y[v, l,
                                                                          nnodes(dg),
                                                                          element] +
                                                              surface_flux_values[v, l,
                                                                                  4,
                                                                                  element] *
                                                              factor_2)
                end
            end
        end
end