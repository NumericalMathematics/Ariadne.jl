## Simple 2D example from (Kelley2003)[@cite]

using Ariadne, LinearAlgebra
using CairoMakie

function F!(res, x, _)
    res[1] = x[1]^2 + x[2]^2 - 2
    return res[2] = exp(x[1] - 1) + x[2]^2 - 2
end

function F(x, p)
    res = similar(x)
    F!(res, x, p)
    return res
end


xs = LinRange(-3, 8, 1000)
ys = LinRange(-15, 10, 1000)

levels = [0.1, 0.25, 0.5:2:10..., 10.0:10:200..., 200:100:4000...]

fig, ax = contour(xs, ys, (x, y) -> norm(F([x, y], nothing)); levels)

trace_1 = let x₀ = [2.0, 0.5]
    xs = Vector{Tuple{Float64, Float64}}(undef, 0)
    hist(x, res, norm_res) = (push!(xs, (x[1], x[2])); nothing)
    x, stats = newton_krylov!(F!, x₀, nothing, callback = hist)
    xs
end
lines!(ax, trace_1)

trace_2 = let x₀ = [2.5, 3.0]
    xs = Vector{Tuple{Float64, Float64}}(undef, 0)
    hist(x, res, norm_res) = (push!(xs, (x[1], x[2])); nothing)
    x, stats = newton_krylov!(F!, x₀, nothing, callback = hist)
    xs
end
lines!(ax, trace_2)

trace_3 = let x₀ = [3.0, 4.0]
    xs = Vector{Tuple{Float64, Float64}}(undef, 0)
    hist(x, res, norm_res) = (push!(xs, (x[1], x[2])); nothing)
    x, stats = newton_krylov!(F!, x₀, nothing, callback = hist, forcing = Ariadne.EisenstatWalker(η_max = 0.68949), verbose = 1)
    @show stats.solved
    xs
end
lines!(ax, trace_3)

fig
