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

#####################################################################
################## CFIE WAVEFUNCTION CONSTRUCTION ###################
#####################################################################

# Flatten the CFIE_kress boundary points into a single cache for faster wavefunction reconstruction, and then evaluate the CFIE_kress wavefunction at many points from the flattened cache and boundary density `u`.
struct CFIEWavefunctionCache{T<:Real}
    x::Vector{T} # boundary x_j
    y::Vector{T} # boundary y_j
    tx::Vector{T} # tangent x-component
    ty::Vector{T} # tangent y-component
    sj::Vector{T} # |tangent_j|
    w::Vector{T} # quadrature weight w_j
end

"""
    flatten_cfie_wavefunction_cache(comps::Vector{BoundaryPointsCFIE{T}}) where {T<:Real} -> CFIEWavefunctionCache{T}

Contains just enough information to evaluate the CFIE wavefunction at many points without needing 
to reconstruct the `BoundaryPointsCFIE` objects or do any extra computations.

# Inputs:
- `comps`: Vector of `BoundaryPointsCFIE` objects, one for each component of the boundary.

# Outputs:
- `CFIEWavefunctionCache{T}`: A struct containing flattened vectors of boundary coordinates, tangents, quadrature weights, etc., for efficient CFIE wavefunction evaluation.
"""
function flatten_cfie_wavefunction_cache(comps::Vector{BoundaryPointsCFIE{T}}) where {T<:Real}
    N=sum(length(c.xy) for c in comps)
    x=Vector{T}(undef,N)
    y=Vector{T}(undef,N)
    tx=Vector{T}(undef,N)
    ty=Vector{T}(undef,N)
    sj=Vector{T}(undef,N)
    w=Vector{T}(undef,N)
    p=1
    @inbounds for c in comps
        for j in eachindex(c.xy)
            q=c.xy[j]
            t=c.tangent[j]
            txj=t[1]
            tyj=t[2]
            x[p]=q[1]
            y[p]=q[2]
            tx[p]=txj
            ty[p]=tyj
            sj[p]=sqrt(txj*txj+tyj*tyj)
            w[p]=c.ws[j]
            p+=1
        end
    end
    return CFIEWavefunctionCache(x,y,tx,ty,sj,w)
end

"""
    ϕ_cfie(xp::T, yp::T, k::T,cache::CFIEWavefunctionCache{T},u::AbstractVector{Complex{T}};float32_bessel::Bool=false) where {T<:Real} -> Complex{T}

Evaluate the CFIE reconstructed wavefunction at point `(xp, yp)` from a
flattened boundary cache and boundary density `u`.

This uses the same kernel as the CFIE assembly:

    ψ(x) = -∑_j w_j u_j [ (i k / 2) * inn * H1(k r) / r + i k * (i/2) * H0(k r) * s_j ]

where
    inn = t_y (x-x_j) - t_x (y-y_j)

# Arguments
- `xp, yp` : evaluation point p = SVector(xp, yp)
- `k::T`      : real wavenumber
- `cache::CFIEWavefunctionCache{T}`  : flattened CFIE geometry cache
- `u::AbstractVector{Complex{T}}`      : complex boundary density, same ordering as flattening
- `float32_bessel::Bool`     : evaluate Hankels in Float32 and cast back

# Returns
- `Complex{T}`: the reconstructed wavefunction value at (xp, yp)
"""
@inline function ϕ_cfie(xp::T,yp::T,k::T,cache::CFIEWavefunctionCache{T},u::AbstractVector{Complex{T}};float32_bessel::Bool=false) where {T<:Real}
    x=cache.x
    y=cache.y
    tx=cache.tx
    ty=cache.ty
    sj=cache.sj
    w=cache.w
    N=length(x)
    @assert length(u)==N
    ψr=zero(T)
    ψi=zero(T)
    # Constants:
    # dterm = (i k / 2) * inn * H1 / r
    # sterm = (i / 2) * H0 * sj
    # contribution = -(w*u) * (dterm + i k * sterm)
    #
    # Since i*k*sterm = i*k*(i/2) H0 sj = -(k/2) H0 sj,
    # the kernel is
    #   K = (i k / 2) * inn * H1 / r  -  (k / 2) * H0 * sj
    khalf=k*T(0.5)
    h=minimum(cache.w) # minimal arc-length spacing
    tol2=(0.5*h)^2 # for near boundary skipping since we have log singularity for H0 
    @inbounds @fastmath for j in 1:N
        dx=xp-x[j]
        dy=yp-y[j]
        r2=muladd(dx,dx,dy*dy)
        r2<=tol2 && continue # skip near-boundary points
        r=sqrt(r2)
        invr=inv(r)
        inn=muladd(ty[j],dx,-(tx[j]*dy)) # ty*dx - tx*dy
        if float32_bessel
            zf=Float32(k*r)
            h0=Complex{T}(Bessels.hankelh1(0,zf))
            h1=Complex{T}(Bessels.hankelh1(1,zf))
        else
            z=k*r
            h0=Bessels.hankelh1(0,z)
            h1=Bessels.hankelh1(1,z)
        end
        # Kernel:
        # K = (i k / 2) * inn * H1 / r - (k / 2) * H0 * sj
        #
        # Let A = (k/2) * inn/r, B = (k/2) * sj
        # Then
        #   K = i*A*h1 - B*h0
        A=khalf*inn*invr
        #B=khalf*sj[j]
        B=khalf*sj[j]
        # i*A*h1 = (-A*imag(h1)) + i*(A*real(h1))
        Kr=muladd(-A,imag(h1),-B*real(h0))
        Ki=muladd(A,real(h1),-B*imag(h0))
        # contribution = -(w*u)*K
        uj=u[j]
        wr=w[j]*real(uj)
        wi=w[j]*imag(uj)
        # -(wr+i wi)(Kr+i Ki)
        ψr-= wr*Kr-wi*Ki
        ψi-= wr*Ki+wi*Kr
    end
    return Complex{T}(ψr, ψi)
end