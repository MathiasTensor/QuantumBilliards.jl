#    multi_kress_graded_nodes_data(T,N,corners_in;q=4)
#
# Construct a globally periodic Kress-type graded parametrization for a closed
# boundary with a finite set of true geometric corners.
#
# Mathematical idea
# We build a monotone periodic map
#    t = t(σ),   σ ∈ [0,2π)
# such that:
# - t(c_j) = c_j for every true corner location c_j,
# - dt/dσ → 0 at each corner, clustering nodes there,
# - the map is smooth and strictly increasing between corners.
#
# Unlike density-based global grading, this construction is *corner-fixing*:
# each corner is treated locally on its adjacent interval, ensuring that
# node clustering occurs at the correct physical locations.
#
# Construction
# Let {c_j} be sorted corner locations. For each σ we:
# 1. Find the enclosing interval [c_j, c_{j+1}] (periodic).
# 2. Map σ → u ∈ [0,1] on that interval.
# 3. Apply a Kress-type smoothing map
#       v(u) = u^q / (u^q + (1-u)^q),
#   which satisfies v'(0)=v'(1)=0.
# 4. Set
#       t(σ) = c_j + (c_{j+1}-c_j) * v(u).
#
# Outputs
# - σ    : uniform periodic grid (shifted to avoid corners)
# - tmap : graded parameter values
# - jac  : dt/dσ
# - jac2 : d²t/dσ²
# - wq   : quadrature weights h * jac
#
# Properties
# - Preserves periodic Kress log structure (depends only on σ differences)
# - Enforces exact corner alignment
# - Applies grading only at true tangent-discontinuous corners
# - Reduces to uniform parametrization if no corners are present
#
# Usage
# Use this for piecewise-smooth closed curves (rectangles, polygons,
# mushrooms, half-stadium with flat edges, etc.). Smooth joins must NOT
# be included in `corners_in`.
#
# Reference
# Kress (1991), extended here to multiple corners via local interval mapping.

const TWO_PI=2*pi

@inline _wrap_to_2pi(x::T) where {T<:Real}=(y=mod(x,T(TWO_PI));y<zero(T) ? y+T(TWO_PI):y)

@inline function _sort_unique_corners(::Type{T},corners_in) where {T<:Real}
    isempty(corners_in)&&return T[]
    cs=T[_wrap_to_2pi(T(c)) for c in corners_in];sort!(cs)
    out=T[cs[1]]
    @inbounds for j in 2:length(cs)
        abs(cs[j]-out[end])>sqrt(eps(T)) && push!(out,cs[j])
    end
    return out
end

@inline function _choose_sigma_shift(::Type{T},h::T,corners) where {T<:Real}
    isempty(corners)&&return h/T(2)
    candidates=T[h/T(2),h/T(3),T(2)*h/T(3),h/T(5),T(2)*h/T(5),T(3)*h/T(5)]
    bestδ=candidates[1];bestd=-one(T)
    for δ in candidates
        mind=typemax(T)
        for c in corners
            r=mod(c-δ,h);d=min(r,h-r);mind=min(mind,d)
        end
        if mind>bestd;bestd=mind;bestδ=δ;end
    end
    return bestδ
end

@inline function _kress_smoothstep(u::T,q::T) where {T<:Real}
    u=clamp(u,zero(T),one(T));a=u^q;b=(one(T)-u)^q;return a/(a+b)
end

@inline function _kress_smoothstep_prime(u::T,q::T) where {T<:Real}
    u=clamp(u,eps(T),one(T)-eps(T))
    a=u^q;b=(one(T)-u)^q;d=a+b
    return q*u^(q-one(T))*(one(T)-u)^(q-one(T))/d^2
end

@inline function _kress_smoothstep_doubleprime(u::T,q::T) where {T<:Real}
    u=clamp(u,eps(T),one(T)-eps(T))
    vp=_kress_smoothstep_prime(u,q)
    d=u^q+(one(T)-u)^q
    dp=q*(u^(q-one(T))-(one(T)-u)^(q-one(T)))
    logder=(q-one(T))/u-(q-one(T))/(one(T)-u)-T(2)*dp/d
    return vp*logder
end

function _corner_interval(corners::Vector{T},x::T) where {T<:Real}
    m=length(corners)
    if m==1;return corners[1],corners[1]+T(TWO_PI);end
    j=searchsortedlast(corners,x)
    if j==0
        left=corners[end]-T(TWO_PI);right=corners[1]
    elseif j==m
        left=corners[end];right=corners[1]+T(TWO_PI)
    else
        left=corners[j];right=corners[j+1]
    end
    return left,right
end

function multi_kress_graded_nodes_data(::Type{T},N::Int,corners_in;q=4) where {T<:Real}
    qT=T(q);qT>one(T)||error("Require q>1.")
    corners=_sort_unique_corners(T,corners_in)
    h=T(TWO_PI)/T(N)
    σ=Vector{T}(undef,N)
    δ=_choose_sigma_shift(T,h,corners)
    @inbounds for k in 1:N;σ[k]=_wrap_to_2pi(δ+T(k-1)*h);end
    sort!(σ)
    if isempty(corners)
        return σ,copy(σ),ones(T,N),zeros(T,N),fill(h,N)
    end
    tmap=Vector{T}(undef,N)
    jac=Vector{T}(undef,N)
    jac2=Vector{T}(undef,N)
    wq=Vector{T}(undef,N)
    @inbounds for i in 1:N
        x=σ[i]
        left,right=_corner_interval(corners,x)
        L=right-left
        u=(x-left)/L
        v=_kress_smoothstep(u,qT)
        vp=_kress_smoothstep_prime(u,qT)
        vpp=_kress_smoothstep_doubleprime(u,qT)
        tmap[i]=_wrap_to_2pi(left+L*v)
        jac[i]=vp
        jac2[i]=vpp/L
        wq[i]=h*jac[i]
    end
    return σ,tmap,jac,jac2,wq
end