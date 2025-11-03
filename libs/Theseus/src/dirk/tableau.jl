struct DIRKButcher{T1 <: AbstractArray, T2 <: AbstractArray} <: RKTableau
    a::T1
    b::T2
    c::T2
end


"""
    Crouzeix23()

A third-order A and L-stable DIRK method developed by Crouzeix (2005).

## References
- Michel Crouzeix (1975)
  *Sur l’approximation des équations différentielles opérationnelles linéaires par des méthodes de Runge–Kutta.*
  *Thèse de 3ᵉ cycle, Université de Paris VI (Pierre et Marie Curie).*
"""
struct Crouzeix23 <: DIRK{2} end
function RKTableau(alg::Crouzeix23, RealT)
    nstage = 2
    sqrt3 = sqrt(convert(RealT, 3))
    a = zeros(RealT, nstage, nstage)
    a[1, 1] = 1 // 2 + sqrt3 / 6
    a[2, 1] = -sqrt3 / 3
    a[2, 2] = 1 // 2 + sqrt3 / 6
    b = zeros(RealT, nstage)
    b[1] = 1 // 2
    b[2] = 1 // 2

    c = zeros(Float64, nstage)
    c[1] = 1 // 2 + sqrt(3) / 6
    c[2] = 1 // 2 - sqrt(3) / 6
    return DIRKButcher(a, b, c)
end
