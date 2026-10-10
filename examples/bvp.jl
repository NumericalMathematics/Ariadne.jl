# BVP from (Kelley2022)[@cite]

using Ariadne, Krylov, LinearAlgebra

function Phi(t, tdag, vp, v)
    phi = 4.0 * tdag * vp + (t * v - 1.0) * v
    return phi
end

function Fbvp!(res, U, (force, tv, tvdag, h, n))
    @assert 2n == length(U)
    res[1] = U[2]
    res[2n] = U[2n - 1]
    v = view(U, 1:2:(2n - 1))
    vp = view(U, 2:2:2n)
    force .= Phi.(tv, tvdag, vp, v)
    h2 = 0.5 * h
    @inbounds @simd for ip in 1:(n - 1)
        res[2 * ip + 1] = v[ip + 1] - v[ip] - h2 * (vp[ip] + vp[ip + 1])
        res[2 * ip] = vp[ip + 1] - vp[ip] + h2 * (force[ip] + force[ip + 1])
    end
    return nothing
end

function BVP_U0!(U0, n, tv)
    view(U0, 1:2:(2n - 1)) .= exp.(-0.1 .* tv .* tv)
    return view(U0, 2:2:2n) .= -0.2 .* view(U0, 1:2:(2n - 1)) .* tv
end

struct GmresPreconditioner{JOp}
    J::JOp
    itmax::Int
end

function LinearAlgebra.mul!(y, P::GmresPreconditioner, x)
    sol, _ = gmres(P.J, x; P.itmax)
    return copyto!(y, sol)
end

function BVP_solve(n = 801, T = Float64)
    U0 = zeros(T, 2n)
    res = zeros(T, 2n)

    h = 20.0 / (n - 1)
    tv = collect(0:h:20.0)

    tvdag = collect(0:h:20.0)
    @views tvdag[2:n] .= (1.0 ./ tv[2:n])

    force = zeros(n)

    BVP_U0!(U0, n, tv)

    bvpout, stats = newton_krylov!(
        Fbvp!, U0, (force, tv, tvdag, h, n), res,
        algo = :fgmres,
        N = (J) -> GmresPreconditioner(J, 30),
    )
    return (; bvpout, tv, stats)
end

BVP_solve()
