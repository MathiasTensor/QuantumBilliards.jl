#  Reference:
#      R. Kress, "Boundary integral equations in time-harmonic acoustic scattering",
#      Math. Comput. Modelling 15 (1991), 229–243.
#
#  Global periodic multi-corner grading as a natural extension of the one-corner
#  Kress grading. The corners are given as locations c_j in the uniform
#  computational variable σ ∈ [0,2π).
#
#  The grading map is defined implicitly by
#
#      w(σ)=2π * ( ∫_0^σ ρ(τ)dτ ) / ( ∫_0^(2π) ρ(τ)dτ ),
#
#  where ρ is a positive periodic density which vanishes at the prescribed
#  corner locations. Thus w'(σ) is small near each corner and the nodes are
#  clustered there.
#
#  We use
#
#      ρ(σ)=∏_j |2 sin((σ-c_j)/2)|^(q-1),
#
#  with grading parameter q>1.
#
#  The computational nodes are chosen on a uniformly spaced periodic grid
#  with spacing h=2π/N, shifted by a small constant δ so that no corner is
#  sampled exactly. This works for both even and odd N and preserves the
#  Kress log-split structure, since only differences σ_i-σ_j matter.
#TODO Make a better one than this -> Kress's teardrop drops from 1e-14 - 1e-15 accuracy to 1e-10 if we use the global multi corner solver!
const TWO_PI=2*pi

@inline function _wrap_to_2pi(x::T) where {T<:Real}
    y=mod(x,T(TWO_PI))
    return y<zero(T) ? y+T(TWO_PI) : y
end

@inline function _periodic_dist(x::T,y::T) where {T<:Real}
    d=abs(x-y)
    return min(d,T(TWO_PI)-d)
end

@inline function _sort_unique_corners(::Type{T},corners_in::AbstractVector) where {T<:Real}
    isempty(corners_in) && return T[]
    cs=T[_wrap_to_2pi(T(c)) for c in corners_in]
    sort!(cs)
    out=T[cs[1]]
    @inbounds for j in 2:length(cs)
        abs(cs[j]-out[end])>sqrt(eps(T)) && push!(out,cs[j])
    end
    return out
end

function _choose_sigma_shift(::Type{T},h::T,corners::AbstractVector{T}) where {T<:Real}
    isempty(corners) && return h/T(2)
    candidates=T[h/T(2),h/T(3),T(2)*h/T(3),h/T(5),T(2)*h/T(5),T(3)*h/T(5)]
    bestδ=candidates[1]
    bestd=-one(T)
    for δ in candidates
        mind=typemax(T)
        for c in corners
            r=mod(c-δ,h)
            d=min(r,h-r)
            mind=min(mind,d)
        end
        if mind>bestd
            bestd=mind
            bestδ=δ
        end
    end
    return bestδ
end

# ρ(σ)=∏_j |2 sin((σ-c_j)/2)|^(q-1)
@inline function _multi_kress_rho(s::T,corners::AbstractVector{T},q::T) where {T<:Real}
    isempty(corners) && return one(T)
    p=q-one(T)
    ρ=one(T)
    @inbounds for c in corners
        ρ*=abs(T(2)*sin((s-c)/T(2)))^p
    end
    return ρ
end

# (ρ'/ρ)(σ)=((q-1)/2) * Σ_j cot((σ-c_j)/2)
@inline function _multi_kress_rho_prime_over_rho(s::T,corners::AbstractVector{T},q::T) where {T<:Real}
    isempty(corners) && return zero(T)
    acc=zero(T)
    @inbounds for c in corners
        acc+=cot((s-c)/T(2))
    end
    return (q-one(T))*acc/T(2)
end

# ρ'(σ)=ρ(σ)*(ρ'/ρ)(σ)
@inline function _multi_kress_rho_prime(s::T,corners::AbstractVector{T},q::T) where {T<:Real}
    ρ=_multi_kress_rho(s,corners,q)
    return iszero(ρ) ? zero(T) : ρ*_multi_kress_rho_prime_over_rho(s,corners,q)
end

# cumulative trapezoidal primitive on an ordered grid
function _cumtrapz(xs::AbstractVector{T},ys::AbstractVector{T}) where {T<:Real}
    N=length(xs)
    N==length(ys) || error("xs and ys must have same length.")
    out=zeros(T,N)
    @inbounds for k in 2:N
        out[k]=out[k-1]+(xs[k]-xs[k-1])*(ys[k-1]+ys[k])/T(2)
    end
    return out
end

function multi_kress_graded_nodes_data(::Type{T},N::Int,corners_in::AbstractVector;q=4,quadN_factor::Int=16) where {T<:Real}
    qT=T(q)
    qT>one(T) || error("Require q>1 for multi-corner grading.")
    corners=_sort_unique_corners(T,corners_in)
    h=T(TWO_PI)/T(N)
    σ=Vector{T}(undef,N)
    δ=_choose_sigma_shift(T,h,corners)
    @inbounds for k in 1:N
        σ[k]=_wrap_to_2pi(δ+T(k-1)*h)
    end
    sort!(σ)
    isempty(corners) && begin
        s=copy(σ)
        a=ones(T,N)
        a2=zeros(T,N)
        wq=fill(h,N)
        return σ,s,a,a2,wq
    end
    # dense auxiliary uniform grid for cumulative quadrature of ρ
    M=max(8*N,quadN_factor*N)
    ξ=Vector{T}(undef,M+1)
    ρξ=Vector{T}(undef,M+1)
    Δ=T(TWO_PI)/T(M)
    @inbounds for j in 0:M
        ξ[j+1]=T(j)*Δ
        ρξ[j+1]=_multi_kress_rho(ξ[j+1],corners,qT)
    end
    F=_cumtrapz(ξ,ρξ)
    Z=F[end]
    Z<=zero(T) && error("Normalization integral for multi-corner grading is nonpositive.")
    s=Vector{T}(undef,N)
    a=Vector{T}(undef,N)
    a2=Vector{T}(undef,N)
    wq=Vector{T}(undef,N)
    # linear interpolation of the cumulative primitive
    @inbounds for k in 1:N
        x=σ[k]
        jf=Int(floor(x/Δ))
        jf=clamp(jf,0,M-1)
        j=jf+1
        t=(x-ξ[j])/(ξ[j+1]-ξ[j])
        Fx=(one(T)-t)*F[j]+t*F[j+1]
        ρx=_multi_kress_rho(x,corners,qT)
        ρpx=_multi_kress_rho_prime(x,corners,qT)
        s[k]=T(TWO_PI)*Fx/Z
        a[k]=T(TWO_PI)*ρx/Z
        a2[k]=T(TWO_PI)*ρpx/Z
        wq[k]=h*a[k]
    end
    return σ,s,a,a2,wq
end