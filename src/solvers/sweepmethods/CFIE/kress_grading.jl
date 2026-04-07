#  Reference:
#      R. Kress, "Boundary integral equations in time-harmonic acoustic scattering",
#      Math. Comput. Modelling 15 (1991), 229–243.
#
#  The grading transformation is given by (1 / q - 1 / 2) * ((pi - s) / pi)^3 + 1 / q * (s - pi) / pi + 1 / 2, where q is the grading parameter.
#  The nodes are then obtained by applying the transformation to the equispaced nodes σ_k = kπ/n, k=1,...,2n-1, and the weights are computed as (π/n) * w'(σ_k), where w(s) is defined in terms of the grading transformation.
#  The function `kress_graded_nodes_weights` computes the graded nodes and weights for a given number of points `N` and grading parameter `q`, ensuring that `N` is odd to satisfy the requirements of the Kress grading formula.
const TWO_PI=2*pi

# v(s,q)
@inline function _kress_v(s::T,q::T) where {T<:Real}
    x=(pi-s)/pi
    return (inv(q)-T(0.5))*x^3+inv(q)*((s-pi)/pi)+T(0.5) # (1 / q - 1 / 2) * ((pi - s) / pi)^3 + 1 / q * (s - pi) / pi + 1 / 2
end
# dv(s,q)/ds
@inline function _kress_vprime(s::T, q::T) where {T<:Real}
    x=(pi-s)/pi
    return -(T(3)/pi)*(inv(q)-T(0.5))*x^2+inv(q)/pi # d/ds x^3 = -3 x^2 / pi
end
# w(s,q) = v(s,q)^q / (v(s,q)^q + v(2π-s,q)^q)
@inline function _kress_w(s::T,q::T) where {T<:Real}
    a=_kress_v(s,q)^q
    b=_kress_v(TWO_PI-s,q)^q
    return TWO_PI*a/(a+b)
end
# dw(s,q)/ds = 2π * (dv(s,q)/ds * (v(s,q)^q + v(2π-s,q)^q) - v(s,q)^q * (dv(s,q)/ds + dv(2π-s,q)/ds)) / (v(s,q)^q + v(2π-s,q)^q)^2
@inline function _kress_wprime(s::T,q::T) where {T<:Real}
    vs=_kress_v(s,q)
    vsp=_kress_vprime(s,q)
    vt=_kress_v(TWO_PI-s,q)
    # d/ds v(2π-s) = -v'(2π-s)
    vtp=-_kress_vprime(TWO_PI-s,q)
    a=vs^q
    b=vt^q
    ap=q*vs^(q-1)*vsp
    bp=q*vt^(q-1)*vtp
    den=a+b
    return TWO_PI*(ap*den-a*(ap+bp))/den^2
end
# Computes the Kress graded nodes and weights for a given number of points `N` and grading parameter `q`.
function kress_graded_nodes_weights(::Type{T},N::Int;q=8) where {T<:Real}
    isodd(N) || error("Use odd N = 2n-1 for the Kress corner grading formula.")
    n=(N+1)÷2
    qT=T(q)
    s=Vector{T}(undef,N)
    w=Vector{T}(undef,N)
    for k in 1:N
        σ=T(k)*π/T(n) # σ_k = kπ/n, k=1,...,2n-1
        s[k]=_kress_w(σ,qT)
        w[k]=(π/T(n))*_kress_wprime(σ,qT)
    end
    return s,w
end
