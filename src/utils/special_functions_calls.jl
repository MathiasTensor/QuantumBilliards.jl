################# GENERAL ####################

@inline H(n::Int,x::T) where {T<:Real}=Bessels.hankelh1(n,x)
@inline H(n::Int,x::Complex{T}) where {T<:Real}=SpecialFunctions.besselh(n,1,x)
@inline function hankel_pair01(x);h0=H(0,x);h1=H(1,x);return h0,h1;end
@inline Φ_helmholtz(k::T,r::T) where {T<:Real}=(im/4)*Bessels.hankelh1(0,k*r) # used for CFIE solvers to get the normal derivative of the wavefunctions
# this is the slp kernel used in the hypersingular maue kress formula for the normal derivative of the DLP kernel.

#################################################################
################# DLP WAVEFUNCTION CONSTRUCTION #################
#################################################################

"""
    ϕ_slp(x::T, y::T, k::T, bd::BoundaryPoints{T}, u::AbstractVector{T}) where {T<:Real} -> T

Wavefunction at (x,y) via the boundary integral:
    Ψ(x,y) = (1/4) ∮ Y₀(k|q-q_s|) u(s) ds

Specialized for real `u` to keep everything in real arithmetic.
"""
@inline function ϕ_slp(x::T,y::T,k::T,bd::Union{BoundaryPoints{T},BoundaryPointsCFIE{T}},u::AbstractVector{T}) where {T<:Real}
    xy=bd.xy;ds=bd.ds
    s=zero(T)
    @inbounds @simd for j in eachindex(u)
        p=xy[j]
        r=hypot(x-p[1],y-p[2]) 
        y0=r<10^2*eps(T) ? zero(T) : Bessels.bessely0(k*r)
        s=muladd(y0*u[j],ds[j],s)
    end
    return s*T(0.25)
end

"""
    ϕ_slp(x::T, y::T, k::T, bd::BoundaryPoints{T}, u::AbstractVector{Complex{T}}) where {T<:Real} -> Complex{T}

Same integral, but with complex boundary data `u`. Uses real kernel and
accumulates real/imag parts separately to avoid unnecessary complex multiplies.
"""
@inline function ϕ_slp(x::T,y::T,k::T,bd::Union{BoundaryPoints{T},BoundaryPointsCFIE{T}},u::AbstractVector{Complex{T}}) where {T<:Real}
    xy=bd.xy;ds=bd.ds
    sr=zero(T); si=zero(T)
    @inbounds @simd for j in eachindex(u)
        p=xy[j]
        r=hypot(x-p[1],y-p[2])
        w=r<10^2*eps(T) ? zero(T) : Bessels.bessely0(k*r)*ds[j] # real weight
        uj=u[j]
        sr=muladd(w,real(uj),sr)
        si=muladd(w,imag(uj),si)
    end
    return Complex(sr,si)*T(0.25)
end

"""
    ϕ_slp_float32_bessel(x::T, y::T, k::T, bd::BoundaryPoints{T}, u::AbstractVector{T}) where {T<:Real} -> T

As `ϕ_slp`, but calls `bessely0` in Float32 for speed; returns in `T`.
"""
@inline function ϕ_slp_float32_bessel(x::T,y::T,k::T,bd::Union{BoundaryPoints{T},BoundaryPointsCFIE{T}},u::AbstractVector{T}) where {T<:Real}
    xy=bd.xy;ds=bd.ds
    s=zero(T)
    @inbounds @simd for j in eachindex(u)
        p=xy[j]
        r=hypot(x-p[1],y-p[2])
        y0=r<10^2*eps(Float32) ? zero(T) : T(Bessels.bessely0(Float32(k*r))) # compute in Float32, cast back
        s=muladd(y0*u[j],ds[j],s)
    end
    return s*T(0.25)
end

"""
    ϕ_slp_float32_bessel(x::T, y::T, k::T, bd::BoundaryPoints{T}, u::AbstractVector{Complex{T}}) where {T<:Real} -> Complex{T}

Float32-Bessel variant for complex `u`. Accumulates real/imag parts separately.
"""
@inline function ϕ_slp_float32_bessel(x::T,y::T,k::T,bd::Union{BoundaryPoints{T},BoundaryPointsCFIE{T}},u::AbstractVector{Complex{T}}) where {T<:Real}
    xy=bd.xy;ds=bd.ds
    sr=zero(T); si=zero(T)
    @inbounds @simd for j in eachindex(u)
        p=xy[j]
        r=hypot(x-p[1],y-p[2])
        w=r<10^2*eps(Float32) ? zero(T) : T(Bessels.bessely0(Float32(k*r)))*ds[j]
        uj=u[j]
        sr=muladd(w,real(uj),sr)
        si=muladd(w,imag(uj),si)
    end
    return Complex(sr,si)*T(0.25)
end