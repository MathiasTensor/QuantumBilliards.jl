#------------------------------------------------------------------------------
# Hyperbolic area/arclength/Weyl utilities for billiards in the Poincaré disk.
#
# Area logic:
#   1. If the origin lies strictly inside the billiard, integrate over θ∈[0,2π).
#   2. If the origin lies on the boundary, e.g. at a triangle vertex, integrate
#      only over the angular sector occupied by the billiard.
#   3. Rays are sampled at angular midpoints, so they avoid boundary sides.
#   4. Ray/segment intersections include endpoint hits but de-duplicate vertex
#      double-counts from adjacent segments.
#
# Hyperbolic polar area formula:
#   dA_H = 4r/(1-r^2)^2 dr dθ
#   ∫_0^ρ 4r/(1-r^2)^2 dr = 2/(1-ρ^2)-2
#------------------------------------------------------------------------------

@inline function cross2(ax::T,ay::T,bx::T,by::T)::T where {T<:Real}
    return muladd(ax,by,-ay*bx)
end

@inline function _angle01(x::T,y::T)::T where {T<:Real}
    return mod2pi(atan(y,x))
end

@inline function _same_t(a::T,b::T,eps::T)::Bool where {T<:Real}
    return abs(a-b)<=eps*max(one(T),abs(a),abs(b))
end

"""
    point_in_poly(xs,ys)::Bool

Return `true` if the origin lies strictly inside the closed polygon/polyline
`(xs,ys)`. The last point is assumed to repeat the first point.

This is the standard horizontal-ray parity test. It is used only for the
strictly-interior case. If the origin lies exactly on the boundary, use
`origin_on_poly`.
"""
@inline function point_in_poly(xs::AbstractVector{T},ys::AbstractVector{T})::Bool where {T<:Real}
    inside=false
    @inbounds for i in 1:length(xs)-1
        ax=xs[i];ay=ys[i]
        bx=xs[i+1];by=ys[i+1]
        c=((ay>zero(T))!=(by>zero(T))) && (zero(T)<(bx-ax)*(-ay)/(by-ay+eps(T))+ax)
        inside⊻=c
    end
    return inside
end

"""
    origin_on_segment(ax,ay,bx,by;eps=1e-13)::Bool

Return `true` if the origin lies on the Euclidean segment from `(ax,ay)` to
`(bx,by)`, up to tolerance. This detects the important case where the polar
center is a boundary vertex or lies on a symmetry edge.
"""
@inline function origin_on_segment(ax::T,ay::T,bx::T,by::T;eps::T=T(1e-13))::Bool where {T<:Real}
    abs(cross2(ax,ay,bx,by))<=eps || return false
    return min(ax,bx)-eps<=zero(T)<=max(ax,bx)+eps &&
           min(ay,by)-eps<=zero(T)<=max(ay,by)+eps
end

"""
    origin_on_poly(xs,ys;eps=1e-13)::Bool

Return `true` if the origin lies on any segment of the closed polyline.
"""
function origin_on_poly(xs::AbstractVector{T},ys::AbstractVector{T};eps::T=T(1e-13))::Bool where {T<:Real}
    @inbounds for i in 1:length(xs)-1
        origin_on_segment(xs[i],ys[i],xs[i+1],ys[i+1];eps=eps) && return true
    end
    return false
end

"""
    angular_support_from_origin(xs,ys;eps=1e-13) -> θ0, Δθ

Determine the angular sector occupied by a star-shaped domain when the origin
lies on the boundary.

The method takes all nonzero boundary sample angles, removes duplicates, finds
the largest empty angular gap, and returns the complementary interval. For a
triangle with one vertex at the origin, this returns exactly the wedge between
the two sides emanating from the origin.
"""
function angular_support_from_origin(xs::AbstractVector{T},ys::AbstractVector{T};eps::T=T(1e-13))::Tuple{T,T} where {T<:Real}
    θs=T[]
    eps2=eps*eps
    @inbounds for i in 1:length(xs)-1
        r2=muladd(xs[i],xs[i],ys[i]*ys[i])
        r2>eps2 && push!(θs,_angle01(xs[i],ys[i]))
    end
    isempty(θs) && return zero(T),T(2)*T(pi)
    sort!(θs)
    clean=T[θs[1]]
    atol=T(100)*eps
    @inbounds for i in 2:length(θs)
        abs(θs[i]-clean[end])>atol && push!(clean,θs[i])
    end
    length(clean)<2 && return clean[1],zero(T)
    n=length(clean)
    maxgap=clean[1]+T(2)*T(pi)-clean[end]
    imax=n
    @inbounds for i in 1:n-1
        gap=clean[i+1]-clean[i]
        if gap>maxgap
            maxgap=gap
            imax=i
        end
    end
    if imax==n
        θ0=clean[1]
        Δθ=clean[end]-clean[1]
    else
        θ0=clean[imax+1]
        Δθ=clean[imax]+T(2)*T(pi)-clean[imax+1]
    end
    return θ0,Δθ
end

"""
    ray_seg_intersect_t(ux,uy,ax,ay,bx,by;...) -> t

Return the positive ray parameter `t` for the intersection

    t*(ux,uy) = (ax,ay) + s*((bx,by)-(ax,ay)),

or `Inf` if there is no robust intersection. Endpoint hits are allowed, because
rays can hit polygon vertices; these are later de-duplicated.
"""
@inline function ray_seg_intersect_t(ux::T,uy::T,ax::T,ay::T,bx::T,by::T;epsden::T=T(1e-14),epss::T=T(1e-11),epst::T=T(1e-13))::T where {T<:Real}
    vx=bx-ax
    vy=by-ay
    den=cross2(ux,uy,vx,vy)
    abs(den)<=epsden && return T(Inf)
    t=cross2(ax,ay,vx,vy)/den
    s=cross2(ax,ay,ux,uy)/den
    return (t>epst && s>=-epss && s<=one(T)+epss) ? t : T(Inf)
end

"""
    ray_hits_min_t(ux,uy,xs,ys;epshit=1e-10) -> tmin, nh

Intersect the ray from the origin in direction `(ux,uy)` with the closed
polyline. Returns the smallest positive hit and the number of distinct hit
radii.

Adjacent segments meeting at a vertex can produce the same hit twice; such
duplicate hits are merged. For a valid star-shaped radial direction, `nh==1`.
"""
@inline function ray_hits_min_t(ux::T,uy::T,xs::AbstractVector{T},ys::AbstractVector{T};epshit::T=T(1e-10))::Tuple{T,Int} where {T<:Real}
    tmin=T(Inf)
    nh=0
    @inbounds for i in 1:length(xs)-1
        t=ray_seg_intersect_t(ux,uy,xs[i],ys[i],xs[i+1],ys[i+1])
        isfinite(t) || continue
        if nh==0
            tmin=t
            nh=1
        elseif !_same_t(t,tmin,epshit)
            return min(tmin,t),2
        else
            t<tmin && (tmin=t)
        end
    end
    return tmin,nh
end

"""
    hyperbolic_area(billiard;tol=1e-6,Nθ0=2048,maxit=12,check_star=true,
                    check_inside=true,kref=1000) -> A, err, Nθ, ok

Compute the hyperbolic area of a star-shaped billiard in the Poincaré disk by
radial integration.

The routine supports two geometries:

  * origin strictly inside the billiard:
      integrate over the full angular range `[0,2π)`;

  * origin on the boundary, e.g. Schmit triangle with one vertex at `(0,0)`:
      determine the visible angular wedge and integrate only over that wedge.

The radial integrand is

    2/(1-r(θ)^2)-2,

where `r(θ)` is the first positive boundary hit along the ray. Angular midpoint
quadrature is used to avoid rays exactly coinciding with boundary edges. The
success flag `ok=false` indicates that the geometry was not compatible with this
star-shaped radial representation or that the sampled boundary left the unit
disk.
"""
function hyperbolic_area(billiard::Bi;tol::Real=1e-6,Nθ0::Int=2048,maxit::Int=12,check_star::Bool=true,check_inside::Bool=true,kref::T=T(1000))::Tuple{T,T,Int,Bool} where {Bi<:AbsBilliard,T<:Real}
    solver=BIM_hyperbolic(T(10),symmetry=nothing)
    pre=precompute_hyperbolic_boundary_cdfs(solver,billiard)
    bd=evaluate_points(solver,billiard,kref,pre)
    xy=bd.xy
    N=length(xy)
    xs=Vector{T}(undef,N+1)
    ys=Vector{T}(undef,N+1)
    @inbounds for i in 1:N
        xs[i]=T(xy[i][1])
        ys[i]=T(xy[i][2])
    end
    xs[N+1]=xs[1]
    ys[N+1]=ys[1]

    epsgeom=T(1e-12)
    inside=point_in_poly(xs,ys)
    onorigin=origin_on_poly(xs,ys;eps=epsgeom)
    check_inside && !(inside||onorigin) && return (T(NaN),T(Inf),0,false)

    @inbounds for i in 1:N
        r2=muladd(xs[i],xs[i],ys[i]*ys[i])
        (!isfinite(r2)||r2>=one(T)-T(100)*eps(T)) && return (T(NaN),T(Inf),0,false)
    end

    twoπ=T(2)*T(pi)
    θ0,Δθ=inside ? (zero(T),twoπ) : angular_support_from_origin(xs,ys;eps=epsgeom)
    (!isfinite(θ0)||!isfinite(Δθ)||Δθ<=zero(T)||Δθ>twoπ+T(1e-10)) && return (T(NaN),T(Inf),0,false)

    function area_rule(Nθ::Int;only_check::Bool=false)::T
        h=Δθ/T(Nθ)
        s=zero(T)
        @inbounds for j in 0:Nθ-1
            θ=θ0+h*(T(j)+T(0.5))
            ux=cos(θ)
            uy=sin(θ)
            tmin,nh=ray_hits_min_t(ux,uy,xs,ys)
            (nh!=1||!isfinite(tmin)||tmin<=zero(T)||tmin>=one(T)) && return T(NaN)
            only_check && continue
            r2=tmin*tmin
            (!isfinite(r2)||r2>=one(T)) && return T(NaN)
            s+=T(2)/(one(T)-r2)-T(2)
        end
        return only_check ? zero(T) : s*h
    end

    if check_star
        isfinite(area_rule(max(2048,Nθ0);only_check=true)) || return (T(NaN),T(Inf),0,false)
    end

    Aprev=area_rule(Nθ0)
    isfinite(Aprev) || return (T(NaN),T(Inf),0,false)
    tolt=T(tol)

    @inbounds for it in 1:maxit
        Nθ=Nθ0<<it
        A=area_rule(Nθ)
        isfinite(A) || return (T(NaN),T(Inf),0,false)
        Aext=(T(4)*A-Aprev)/T(3)
        err=abs(Aext-A)
        err<=tolt && return (Aext,err,Nθ,true)
        Aprev=A
    end

    Nθ=Nθ0<<maxit
    A=area_rule(Nθ)
    isfinite(A) || return (T(NaN),T(Inf),0,false)
    Aext=(T(4)*A-Aprev)/T(3)
    return (Aext,abs(Aext-A),Nθ,true)
end

"""
    hyperbolic_area_fundamental(solver,billiard;...) -> A_fd

Return the hyperbolic area of the symmetry-reduced fundamental domain implied
by `solver.symmetry`.

Supported symmetry logic:

  * `nothing`      : return full area;
  * `Reflection`   : divide by `2`, or by `2*length(parity)` for combined
                     reflection sectors;
  * `Rotation(n,...)`: divide by `n`.
"""
function hyperbolic_area_fundamental(solver::Union{BIM_hyperbolic,DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners,DLP_hyperbolic_log_product},billiard::Bi;tol::Real=1e-6,Nθ0::Int=2048,maxit::Int=12,check_star::Bool=true,check_inside::Bool=true,kref::T=T(1000.0)) where {Bi<:AbsBilliard,T<:Real}
    A,_,_,ok=hyperbolic_area(billiard;tol=tol,Nθ0=Nθ0,maxit=maxit,check_star=check_star,check_inside=check_inside,kref=kref)
    ok || error("Failed to compute hyperbolic area for symmetry-adapted Weyl estimate.")
    sym=solver.symmetry
    isnothing(sym) && return A
    if sym isa Reflection
        l=sym.parity isa Integer ? 1 : length(sym.parity)
        return A/(2*l)
    elseif sym isa Rotation
        return A/sym.n
    else
        error("Unknown symmetry type for symmetry-adapted hyperbolic area.")
    end
end

"""
    hyperbolic_arclength(billiard;kref=1000) -> LH

Return the full hyperbolic arclength of the physical billiard boundary.
For symmetry sectors use `symmetry_adapted_hyperbolic_arclength`.
"""
function hyperbolic_arclength(billiard::Bi;kref::T=T(1000.0))::T where {Bi<:AbsBilliard,T<:Real}
    solver=BIM_hyperbolic(T(10),symmetry=nothing)
    pre=precompute_hyperbolic_boundary_cdfs(solver,billiard)
    bd=evaluate_points(solver,billiard,kref,pre)
    return T(bd.LH)
end

"""
    _L_axis0(a)

Exact hyperbolic arclength from the origin to Euclidean coordinate `a` along
a coordinate axis in the Poincaré disk:

    L = log((1+a)/(1-a)).
"""
@inline function _L_axis0(a::T)::T where {T<:Real}
    return log((one(T)+a)/(one(T)-a))
end

"""
    _axis_intersect_pos(xs,ys,which;eps=1e-14) -> best

Find the positive coordinate where the full boundary intersects a coordinate
axis.

  * `which == :x0`: return max positive `y` at crossings with `x=0`;
  * `which == :y0`: return max positive `x` at crossings with `y=0`.

This is used to reconstruct virtual symmetry-cut edge lengths.
"""
function _axis_intersect_pos(xs::Vector{T},ys::Vector{T},which::Symbol;eps::T=T(1e-14)) where {T<:Real}
    best=zero(T)
    @inbounds for i in 1:length(xs)-1
        x1=xs[i];y1=ys[i]
        x2=xs[i+1];y2=ys[i+1]
        if which===:x0
            abs(x1)<=eps && y1>best && (best=y1)
            if (x1>eps&&x2<-eps)||(x1<-eps&&x2>eps)
                t=x1/(x1-x2)
                y=y1+t*(y2-y1)
                y>best && (best=y)
            end
        elseif which===:y0
            abs(y1)<=eps && x1>best && (best=x1)
            if (y1>eps&&y2<-eps)||(y1<-eps&&y2>eps)
                t=y1/(y1-y2)
                x=x1+t*(x2-x1)
                x>best && (best=x)
            end
        else
            error("which must be :x0 or :y0")
        end
    end
    return best
end

"""
    _physical_LH_fundamental(solver,billiard;kref=1000) -> LH

Return the hyperbolic length of the physical billiard walls actually
discretized by the symmetry-adapted solver. This excludes virtual symmetry
cut edges, which are added or subtracted separately in
`symmetry_adapted_hyperbolic_arclength`.
"""
function _physical_LH_fundamental(solver::Union{BIM_hyperbolic,DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners,DLP_hyperbolic_log_product},billiard::Bi;kref::T=T(1000.0)) where {Bi<:AbsBilliard,T<:Real}
    pre=precompute_hyperbolic_boundary_cdfs(solver,billiard;M_cdf_base=4000,safety=1e-14)
    bd=evaluate_points(solver,billiard,kref,pre;safety=1e-14,threaded=true)
    return T(bd.LH)
end

"""
    symmetry_adapted_hyperbolic_arclength(solver,billiard;kref=1000) -> Leff

Return the effective perimeter term entering the two-term Weyl law for the
specified symmetry sector.

For reflection reductions, virtual symmetry edges contribute with boundary
condition sign:

  * odd parity / Dirichlet edge: add the cut length;
  * even parity / Neumann edge: subtract the cut length.

For pure rotations the discretized physical wedge boundary is already the
correct perimeter contribution.
"""
function symmetry_adapted_hyperbolic_arclength(solver::Union{BIM_hyperbolic,DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners,DLP_hyperbolic_log_product},billiard::Bi;kref::T=T(1000.0)) where {Bi<:AbsBilliard,T<:Real}
    Lphys=_physical_LH_fundamental(solver,billiard;kref=kref)
    sym=solver.symmetry
    isnothing(sym) && return Lphys
    !(sym isa Reflection) && return Lphys

    par=sym.parity
    solver0=BIM_hyperbolic(T(10),symmetry=nothing)
    pre0=precompute_hyperbolic_boundary_cdfs(solver0,billiard;M_cdf_base=4000,safety=1e-14)
    bd0=evaluate_points(solver0,billiard,kref,pre0;safety=1e-14,threaded=true)
    xy=bd0.xy
    N=length(xy)
    xs=Vector{T}(undef,N+1)
    ys=Vector{T}(undef,N+1)
    @inbounds for i in 1:N
        xs[i]=T(xy[i][1])
        ys[i]=T(xy[i][2])
    end
    xs[N+1]=xs[1]
    ys[N+1]=ys[1]

    Lcut=zero(T)
    if sym.axis===:y_axis
        ymax=_axis_intersect_pos(xs,ys,:x0)
        Lcut-=par*_L_axis0(ymax)
    elseif sym.axis===:x_axis
        xmax=_axis_intersect_pos(xs,ys,:y0)
        Lcut-=par*_L_axis0(xmax)
    elseif sym.axis===:origin
        ymax=_axis_intersect_pos(xs,ys,:x0)
        xmax=_axis_intersect_pos(xs,ys,:y0)
        Lcut-=par[1]*_L_axis0(ymax)+par[2]*_L_axis0(xmax)
    else
        error("Incompatible reflection axis.")
    end
    return Lphys+Lcut
end

"""
    weyl_law_hyp(k,A,L;C=0)

Two-term Weyl estimate for the hyperbolic Laplacian convention

    λ = 1/4 + k^2.

The estimate is

    N(k) ≈ A λ/(4π) - L sqrt(λ)/(4π) + C.
"""
@inline function weyl_law_hyp(k::T,A::T,L::T;C::T=zero(T))::T where {T<:Real}
    λ=k*k+T(0.25)
    return (A*λ-L*sqrt(λ))/(T(4)*T(pi))+C
end

"""
    weyl_law_hyp(solver,billiard,k;kref=1000,C=0)

Compute the symmetry-adapted two-term Weyl count using

  * `hyperbolic_area_fundamental` for the area term;
  * `symmetry_adapted_hyperbolic_arclength` for the perimeter term.

This is the correct wrapper for desymmetrized reflection sectors because the
virtual symmetry edges have Dirichlet/Neumann signs in the perimeter term.
"""
function weyl_law_hyp(solver::Union{BIM_hyperbolic,DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners,DLP_hyperbolic_log_product},billiard::Bi,k::T;kref::T=T(1000.0),C::T=zero(T))::T where {Bi<:AbsBilliard,T<:Real}
    A=hyperbolic_area_fundamental(solver,billiard;kref=kref)
    L=symmetry_adapted_hyperbolic_arclength(solver,billiard;kref=kref)
    return weyl_law_hyp(k,A,L;C=C)
end