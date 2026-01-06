#------------------------------------------------------------------------------
# Utilities for Poincaré-disk hyperbolic billiards: star-shaped area
# integration (via radial ray hits) and hyperbolic arclength/Weyl counting
# estimates for unfolding and level-count checks.
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# cross2(ax,ay,bx,by)::T
#
# INPUTS:
#   ax::T, ay::T    components of vector a
#   bx::T, by::T    components of vector b
#
# OUTPUTS:
#   c::T            scalar cross product a×b = ax*by - ay*bx
#------------------------------------------------------------------------------
@inline function cross2(ax::T,ay::T,bx::T,by::T)::T where {T<:Real}
    return muladd(ax,by,-ay*bx)
end

#------------------------------------------------------------------------------
# ray_seg_intersect_t(ux,uy,ax,ay,bx,by)::T
#
# INPUTS:
#   ux::T, uy::T    ray direction u (ray starts at origin)
#   ax::T, ay::T    segment endpoint a
#   bx::T, by::T    segment endpoint b
#
# OUTPUTS:
#   t::T            ray parameter of intersection point t*u; returns Inf if no hit
#------------------------------------------------------------------------------
@inline function ray_seg_intersect_t(ux::T,uy::T,ax::T,ay::T,bx::T,by::T)::T where {T<:Real}
    vx=bx-ax;vy=by-ay
    den=cross2(ux,uy,vx,vy)
    abs(den)<T(1e-15) && return T(Inf)
    t=cross2(ax,ay,vx,vy)/den
    s=cross2(ax,ay,ux,uy)/den
    epss=T(1e-12)
    (t>zero(T) && s>epss && s<one(T)-epss) ? t : T(Inf)
end

#------------------------------------------------------------------------------
# ray_hits_min_t(ux,uy,xs,ys)::Tuple{T,Int}
#
# INPUTS:
#   ux::T, uy::T                  ray direction u (ray starts at origin)
#   xs::AbstractVector{T}         polygon x coords, length N+1, closed
#   ys::AbstractVector{T}         polygon y coords, length N+1, closed
#
# OUTPUTS:
#   tmin::T                       minimal positive intersection parameter (Inf if none)
#   nh::Int                       number of valid intersections found
#------------------------------------------------------------------------------
@inline function ray_hits_min_t(ux::T,uy::T,xs::AbstractVector{T},ys::AbstractVector{T})::Tuple{T,Int} where {T<:Real}
    tmin=T(Inf);nh=0
    @inbounds for i in 1:length(xs)-1
        t=ray_seg_intersect_t(ux,uy,xs[i],ys[i],xs[i+1],ys[i+1])
        if isfinite(t)
            nh+=1
            t<tmin && (tmin=t)
            nh>1 && return tmin,nh
        end
    end
    return tmin,nh
end

#------------------------------------------------------------------------------
# point_in_poly(xs,ys)::Bool
#
# INPUTS:
#   xs::AbstractVector{T}         polygon x coords, length N+1, closed
#   ys::AbstractVector{T}         polygon y coords, length N+1, closed
#
# OUTPUTS:
#   inside::Bool                  true iff origin (0,0) is inside polygon
#------------------------------------------------------------------------------
@inline function point_in_poly(xs::AbstractVector{T},ys::AbstractVector{T})::Bool where {T<:Real}
    inside=false
    @inbounds for i in 1:length(xs)-1
        ax=xs[i];ay=ys[i];bx=xs[i+1];by=ys[i+1]
        c=((ay>zero(T))!=(by>zero(T))) && (zero(T)<(bx-ax)*(-ay)/(by-ay+eps(T))+ax)
        inside ⊻= c
    end
    return inside
end

#------------------------------------------------------------------------------
# hyperbolic_area(billiard;tol,Nθ0,maxit,check_star,check_inside,kref)
#   ::Tuple{Real,Real,Int,Bool}
#
# INPUTS:
#   billiard::Bi                   AbsBilliard inside Poincaré disk
#   tol::Real=1e-6                 target abs tolerance for Richardson err
#   Nθ0::Int=2048                  initial angular resolution
#   maxit::Int=12                  max doublings of Nθ (Nθ=Nθ0<<it)
#   check_star::Bool=true          verify star-shaped wrt origin via rays
#   check_inside::Bool=true        verify origin is inside sampled polygon
#   kref=1000.0                 sampling reference for boundary points
#
# OUTPUTS:
#   A::Real                          extrapolated hyperbolic area
#   err::Real                         last Richardson error estimate
#   Nθ::Int                        final angular resolution used
#   ok::Bool                       false if checks fail (NaN/Inf outputs)
#
# NOTES:
#   Star-shaped polar formula around origin with r(θ)=first ray hit:
#     A = ∫₀^{2π} ( 2/(1-r(θ)^2) - 2 ) dθ
#------------------------------------------------------------------------------
function hyperbolic_area(billiard::Bi;tol::Real=1e-6,Nθ0::Int=2048,maxit::Int=12,check_star::Bool=true,check_inside::Bool=true,kref::T=T(1000))::Tuple{T,T,Int,Bool} where {Bi<:AbsBilliard,T<:Real}
    solver=BIM_hyperbolic(T(10),symmetry=nothing)
    pre=precompute_hyperbolic_boundary_cdfs(solver,billiard)
    bd=evaluate_points(solver,billiard,kref,pre)
    xy=bd.xy;N=length(xy)
    xs=Vector{T}(undef,N+1);ys=Vector{T}(undef,N+1)
    @inbounds for i in 1:N;xs[i]=T(xy[i][1]);ys[i]=T(xy[i][2]);end
    xs[N+1]=xs[1];ys[N+1]=ys[1]
    check_inside && !point_in_poly(xs,ys) && return (T(NaN),T(Inf),0,false)
    @inbounds for i in 1:N;muladd(xs[i],xs[i],ys[i]*ys[i])>=one(T) && return (T(NaN),T(Inf),0,false);end
    twoπ=T(2)*T(pi)
    function area_trap(Nθ::Int;only_check::Bool=false)::T
        h=twoπ/T(Nθ);s=zero(T)
        @inbounds for j in 0:Nθ-1
            θ=h*T(j);ux=cos(θ);uy=sin(θ)
            tmin,nh=ray_hits_min_t(ux,uy,xs,ys)
            (nh!=1||!isfinite(tmin)||tmin>=one(T)) && return T(NaN)
            only_check && continue
            r2=tmin*tmin
            r2>=one(T) && return T(NaN)
            s+=T(2)/(one(T)-r2)-T(2)
        end
        return only_check ? zero(T) : s*h
    end
    if check_star;isfinite(area_trap(max(2048,Nθ0);only_check=true))||return (T(NaN),T(Inf),0,false);end
    Aprev=area_trap(Nθ0);isfinite(Aprev)||return (T(NaN),T(Inf),0,false)
    tolt=T(tol)
    @inbounds for it in 1:maxit
        Nθ=Nθ0<<it;A=area_trap(Nθ);isfinite(A)||return (T(NaN),T(Inf),0,false)
        Aext=(T(4)*A-Aprev)/T(3);err=abs(Aext-A)
        err<=tolt && return (Aext,err,Nθ,true)
        Aprev=A
    end
    Nθ=Nθ0<<maxit;A=area_trap(Nθ);Aext=(T(4)*A-Aprev)/T(3)
    return (Aext,abs(Aext-A),Nθ,true)
end

#------------------------------------------------------------------------------
# hyperbolic_area_fundamental(solver,billiard; ...) -> A_fd
#------------------------------------------------------------------------------
# PURPOSE
#   Return the hyperbolic area of the symmetry-reduced (fundamental) domain.
#
# INPUTS
#   solver :: BIM_hyperbolic
#     - Uses solver.symmetry (either `nothing` or a Vector of length 1):
#         * nothing                      -> no reduction
#         * Reflection(parity=Int)        -> divide area by 2
#         * Reflection(parity=Vector)     -> divide area by 2*length(parity)
#         * Rotation(n, m, center)        -> divide area by n
#
#   billiard :: Bi where Bi<:AbsBilliard
#     - Geometry inside the Poincaré disk; forwarded to `hyperbolic_area`.
#
# KEYWORDS  (forwarded to hyperbolic_area)
#   tol::Real=1e-6, Nθ0::Int=2048, maxit::Int=12,
#   check_star::Bool=true, check_inside::Bool=true, kref::T=T(1000.0)
#
# OUTPUT
#   A_fd :: T
#     Hyperbolic area of the fundamental domain implied by `solver.symmetry`.
#
#------------------------------------------------------------------------------
function hyperbolic_area_fundamental(solver::BIM_hyperbolic,billiard::Bi;tol::Real=1e-6,Nθ0::Int=2048,maxit::Int=12,check_star::Bool=true,check_inside::Bool=true,kref::T=T(1000.0)) where {Bi<:AbsBilliard,T<:Real}
    A,_,_,ok=hyperbolic_area(billiard;tol=tol,Nθ0=Nθ0,maxit=maxit,check_star=check_star,check_inside=check_inside,kref=kref)
    !ok && return error("Failed to compute hyperbolic area for symmetry-adapted Weyl estimate.")
    symmetry=solver.symmetry
    isnothing(symmetry) && return A
    sym=symmetry[1]
    if sym isa Reflection
        l=sym.parity isa Integer ? 1 : length(sym.parity)
        return A/(2*l)
    elseif sym isa Rotation
        n=sym.n
        return A/n
    else
        @error("Unknown symmetry type for symmetry-adapted hyperbolic area.")
    end
end

#------------------------------------------------------------------------------
# hyperbolic_arclength(billiard;kref)::T
# This is the total hyperbolic arclength of the billiard boundary. For the symmetry-adapted one which takes
# into account the fundamental domain, use `hyperbolic_arclength_fundamental`.
#
# INPUTS:
#   billiard::Bi                   AbsBilliard inside Poincaré disk
#   kref::T=1000.0                 sampling reference for boundary points
#
# OUTPUTS:
#   LH::T                          hyperbolic boundary length
#------------------------------------------------------------------------------
function hyperbolic_arclength(billiard::Bi;kref::T=T(1000.0))::T where {Bi<:AbsBilliard,T<:Real}
    solver=BIM_hyperbolic(10.0,symmetry=nothing)
    pre=precompute_hyperbolic_boundary_cdfs(solver,billiard)
    bd=evaluate_points(solver,billiard,kref,pre)
    return T(bd.LH)
end

# ------------------------------------------------------------------------------
# _L_axis0
#
# Hyperbolic arclength along a coordinate axis from 0 to a (a > 0),
# in the Poincaré disk model.
#
# This is the exact hyperbolic distance from (0,0) to (0,a) or (a,0)
# along an axis. It is used to convert a Euclidean axis-intersection
# coordinate into a hyperbolic length contribution for symmetry cut edges.
#
# Formula:
#   L(a) = log((1 + a) / (1 - a))
# ------------------------------------------------------------------------------
@inline _L_axis0(a::T) where {T<:Real}=log((one(T)+a)/(one(T)-a))

# ------------------------------------------------------------------------------
# _axis_intersect_pos
#
# Find the maximum positive intersection coordinate of a closed polyline
# with a coordinate axis.
#
# Given a closed boundary polyline (xs, ys), this scans all segments and
# returns:
# - for which == :x0 : max y > 0 where the boundary crosses x = 0
# - for which == :y0 : max x > 0 where the boundary crosses y = 0
#
# This is purely Euclidean geometry; the result is later converted to a
# hyperbolic length via _L_axis0.
#
# Arguments:
# - xs, ys : closed polyline coordinates (last point should repeat first)
# - which  : :x0 or :y0
#
# Keyword:
# - eps    : tolerance for detecting axis crossings and on-axis vertices
#
# Returns:
# - best   : maximum positive intersection coordinate (0 if none found)
#
# Used for:
# - Determining the extent of virtual symmetry edges in reflection sectors
# ------------------------------------------------------------------------------
function _axis_intersect_pos(xs::Vector{T},ys::Vector{T},which::Symbol;eps::T=T(1e-14)) where {T<:Real}
    N=length(xs)
    best=zero(T)
    @inbounds for i in 1:(N-1)
        x1=xs[i];y1=ys[i]
        x2=xs[i+1];y2=ys[i+1]
        if which===:x0
            #segment crosses x=0
            if abs(x1)<=eps && y1>best;best=y1;end
            if (x1>eps && x2<-eps)||(x1<-eps && x2>eps)
                t=x1/(x1-x2)
                y=y1+t*(y2-y1)
                if y>best;best=y;end
            end
        elseif which===:y0
            if abs(y1)<=eps && x1>best;best=x1;end
            if (y1>eps && y2<-eps)||(y1<-eps && y2>eps)
                t=y1/(y1-y2)
                x=x1+t*(x2-x1)
                if x>best;best=x;end
            end
        else
            error("which must be :x0 or :y0")
        end
    end
    return best
end

# ------------------------------------------------------------------------------
# _physical_LH_fundamental
#
# Hyperbolic boundary length of the physical boundary actually discretized
# in the fundamental domain.
#
# This includes only true billiard walls, and excludes any virtual symmetry cut edges.
#
# Used as the baseline perimeter contribution in the Weyl law before
# symmetry corrections due to virtual edges of reflections are applied.
#
# Arguments:
# - solver   : BIM_hyperbolic with symmetry already applied
# - billiard : billiard geometry
#
# Keyword:
# - kref     : reference k used only for accurate boundary sampling
#
# Returns:
# - LH       : hyperbolic boundary length of the physical boundary
# ------------------------------------------------------------------------------
function _physical_LH_fundamental(solver::BIM_hyperbolic,billiard::Bi;kref::T=T(1000.0)) where {Bi<:AbsBilliard,T<:Real}
    pre=precompute_hyperbolic_boundary_cdfs(solver,billiard;M_cdf_base=4000,safety=1e-14)
    bd=evaluate_points(solver,billiard,kref,pre;safety=1e-14,threaded=true)
    return T(bd.LH)
end

# ------------------------------------------------------------------------------
# symmetry_adapted_hyperbolic_arclength
#
# Effective hyperbolic perimeter length entering the Weyl law for a given
# symmetry sector (with all reflection corrections).
#
# Logic:
# - Start from Lphys = physical boundary length (Dirichlet)
# - If no symmetry → return Lphys
# - If Reflection symmetry:
#     * identify missing virtual symmetry edges along x=0 and/or y=0
#     * compute their hyperbolic lengths via _L_axis0
#     * apply parity:
#         parity = -1 → Dirichlet  → add length
#         parity = +1 → Neumann    → subtract length
# - If Rotation symmetry → return Lphys (wedge handling is implicit)
#
# The minus sign is absorbed here so the returned L can be directly used
# in the Weyl perimeter term without further sign handling.
#
# Arguments:
# - solver   : BIM_hyperbolic (with symmetry)
# - billiard : geometry
#
# Keyword:
# - kref     : reference k for boundary sampling
#
# Returns:
# - Leff     : effective hyperbolic perimeter length
# ------------------------------------------------------------------------------
function symmetry_adapted_hyperbolic_arclength(solver::BIM_hyperbolic,billiard::Bi;kref::T=T(1000.0)) where {Bi<:AbsBilliard,T<:Real}
    Lphys=_physical_LH_fundamental(solver,billiard;kref=kref)
    symm=solver.symmetry
    isnothing(symm) && return Lphys
    sym=symm[1]
    par=symm[1].parity
    !(sym isa Reflection) && return Lphys # Rotation already has the correct wedge handling
    #get a closed polyline of the FULL boundary (no symmetry),to locate axis intersections
    solver0=BIM_hyperbolic(10.0,symmetry=nothing)
    pre0=precompute_hyperbolic_boundary_cdfs(solver0,billiard;M_cdf_base=4000,safety=1e-14)
    bd0=evaluate_points(solver0,billiard,kref,pre0;safety=1e-14,threaded=true)
    xy=bd0.xy;N=length(xy)
    xs=Vector{T}(undef,N+1);ys=Vector{T}(undef,N+1)
    @inbounds for i in 1:N
        xs[i]=T(xy[i][1]);ys[i]=T(xy[i][2])
    end
    xs[N+1]=xs[1];ys[N+1]=ys[1]
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
        error("Incompatible reflection (not usable for rotations)")
    end
    return Lphys+Lcut
end

#------------------------------------------------------------------------------
# weyl_law_hyp(k,A,L;C=zero(T))::T
#
# INPUTS:
#   k::T                         solver spectral parameter (your solver outputs k)
#   A::T                         hyperbolic area
#   L::T                         hyperbolic boundary length
#   C::T=zero(T)                 constant shift in Weyl law
#
# OUTPUTS:
#   Nweyl::T                     2-term Dirichlet Weyl count estimate
#
# NOTES:
#   For hyperbolic convention ν=-1/2+ik (Legendre Q kernel), λ=1/4+k^2.
#   N(λ) ≈ (A/(4π))*λ - (L/(4π))*sqrt(λ).
#------------------------------------------------------------------------------
@inline function weyl_law_hyp(k::T,A::T,L::T;C::T=zero(T))::T where {T<:Real}
    λ=(k*k+T(0.25))
    return (A*λ-L*sqrt(λ))/(T(4)*T(pi))+C
end

#------------------------------------------------------------------------------
# weyl_law_hyp(billiard,k;kref=1000.0,C=zero(T))::T
#
# INPUTS:
#   billiard::Bi                  AbsBilliard inside Poincaré disk
#   k::T                          solver spectral parameter
#   kref::T=1000.0                sampling reference for A/L estimation
#   C::T=zero(T)                  constant shift in Weyl law
#
# OUTPUTS:
#   Nweyl::T                      Weyl count estimate (NaN if failure)
#------------------------------------------------------------------------------
function weyl_law_hyp(solver::BIM_hyperbolic,billiard::Bi,k::T;kref::T=T(1000.0),C::T=zero(T))::T where {Bi<:AbsBilliard,T<:Real}
    A=hyperbolic_area_fundamental(solver,billiard;kref=kref)
    L=hyperbolic_arclength(billiard;kref=kref)
    return weyl_law_hyp(k,A,L;C=C)
end