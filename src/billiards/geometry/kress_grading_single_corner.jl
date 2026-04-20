#  Reference:
#      R. Kress, "Boundary integral equations in time-harmonic acoustic scattering",
#      Math. Comput. Modelling 15 (1991), 229–243.
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
@inline function _kress_vdoubleprime(s::T,q::T) where {T<:Real}
    x=(pi-s)/pi
    return T((6/(pi^2))*(inv(q)-0.5)*x)
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
@inline function _kress_wdoubleprime(s::T,q::T) where {T<:Real}
    vs=_kress_v(s,q)
    vsp=_kress_vprime(s,q)
    vspp=_kress_vdoubleprime(s,q)
    vt=_kress_v(TWO_PI-s,q)
    vtp=-_kress_vprime(TWO_PI-s,q)
    vtpp=_kress_vdoubleprime(TWO_PI-s,q)
    a=vs^q
    b=vt^q
    ap=q*vs^(q-1)*vsp
    bp=q*vt^(q-1)*vtp
    app=q*((q-one(T))*vs^(q-2)*vsp^2+vs^(q-1)*vspp)
    bpp=q*((q-one(T))*vt^(q-2)*vtp^2+vt^(q-1)*vtpp)
    den=a+b
    denp=ap+bp
    num=ap*b-a*bp
    nump=app*b-a*bpp
    return T(TWO_PI*(nump*den-2*num*denp)/den^3)
end
# Computes the Kress graded nodes and weights for a given number of points `N`
# and grading parameter `q`, using a periodic grid that works for both even and
# odd N.
function kress_graded_nodes_data(::Type{T},N::Int;q=8) where {T<:Real}
    qT=T(q)
    qT>one(T) || error("Require q>1 for Kress grading.")
    σ=Vector{T}(undef,N)
    s=Vector{T}(undef,N)
    a=Vector{T}(undef,N)
    a2=Vector{T}(undef,N)
    wq=Vector{T}(undef,N)
    h=T(TWO_PI)/T(N)
    δ=h/T(2)   # midpoint shift: avoids sampling the corner exactly
    for k in 1:N
        σ[k]=δ + T(k-1)*h
        s[k]=_kress_w(σ[k],qT)
        a[k]=_kress_wprime(σ[k],qT)
        a2[k]=_kress_wdoubleprime(σ[k],qT)
        wq[k]=h*a[k]
    end
    return σ,s,a,a2,wq
end

