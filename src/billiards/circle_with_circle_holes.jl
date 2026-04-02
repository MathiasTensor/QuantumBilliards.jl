"""
    validate_holes(R::T, x0::T, y0::T, holes::Vector{Tuple{T,T,T}}) where {T<:Real}

Validate that all holes:
- have positive radius,
- lie strictly inside the outer circle of radius `R` centered at `(x0,y0)`,
- do not overlap or touch each other.
"""
function validate_holes(R::T,x0::T,y0::T,holes::Vector{Tuple{T,T,T}}) where {T<:Real}
    R<=zero(T) && error("Outer radius must be positive.")
    nh=length(holes)
    @inbounds for i in 1:nh
        ri,xi,yi=holes[i]
        ri<=zero(T) && error("Hole $i has non-positive radius.")
        hypot(xi-x0,yi-y0)+ri>=R && error("Hole $i is not strictly inside the outer circle.")
    end
    @inbounds for i in 1:nh
        ri,xi,yi=holes[i]
        for j in (i+1):nh
            rj,xj,yj=holes[j]
            hypot(xi-xj,yi-yj)<=ri+rj && error("Holes $i and $j overlap or touch.")
        end
    end
    return nothing
end

"""
    make_circle_with_holes(R::T; x0=zero(T), y0=zero(T), rot_angle=zero(T), holes=Tuple{T,T,T}[]) where {T<:Real}

Construct a multiply connected billiard boundary consisting of:
- one outer circle of radius `R`,
- zero or more inner circular holes.

Each hole is given as `(r, hx, hy)`, where
- `r`  = hole radius,
- `(hx,hy)` = hole center.

# Returns
- `(boundary, outer_center)` where
  - `boundary[1]` is the outer circle,
  - `boundary[2:end]` are the hole boundaries.
"""
function make_circle_with_holes(R::T;x0=zero(T),y0=zero(T),rot_angle=zero(T),holes::Vector{Tuple{T,T,T}}=Tuple{T,T,T}[]) where {T<:Real}
    validate_holes(R,x0,y0,holes)
    outer_origin=SVector(x0,y0)
    outer=CircleSegment(R,2*pi,zero(T),zero(T),zero(T);origin=outer_origin,rot_angle=rot_angle)
    boundary=Vector{CircleSegment{T}}(undef,1+length(holes))
    boundary[1]=outer
    @inbounds for (i,(r,hx,hy)) in enumerate(holes)
        origin=SVector(hx,hy)
        boundary[i+1]=CircleSegment(r,2*pi,zero(T),zero(T),zero(T);origin=origin,rot_angle=rot_angle)
    end
    return boundary,SVector(x0,y0)
end

"""
    make_quarter_full_annulus_boundary(
        R::T, r::T;
        x0=zero(T), y0=zero(T),
        xh=x0, yh=y0,
        rot_angle=zero(T)
    ) where {T<:Real}

Construct the desymmetrized real boundary for a quarter annulus:
- quarter outer arc
- quarter inner arc

This is what CFIE_alpert / CFIE_kress with symmetry-images should use as
`desymmetrized_full_boundary` in the annulus case.
"""
function make_quarter_full_annulus_boundary(R::T,r::T;x0=zero(T),y0=zero(T),xh=x0,yh=y0,rot_angle=zero(T)) where {T<:Real}
    outer_origin=SVector(x0,y0)
    inner_origin=SVector(xh,yh)
    center=SVector(x0,y0)
    outer_q=CircleSegment(R,pi/2,zero(T),zero(T),zero(T);origin=outer_origin,rot_angle=rot_angle)
    inner_q=CircleSegment(r,pi/2,zero(T),zero(T),zero(T);origin=inner_origin,rot_angle=rot_angle)
    boundary=[[outer_q],[inner_q]]
    return boundary,center
end

"""
    _group_full_boundary_components(boundary::Vector{CircleSegment{T}}) where {T<:Real}

Group each closed circle boundary into its own component, in the format expected by
Kress / Alpert composite logic:
    Vector{Vector{CircleSegment{T}}}

For these circular-hole billiards:
- outer circle is one component
- each hole circle is one component
"""
function _group_full_boundary_components(boundary::Vector{CircleSegment{T}}) where {T<:Real}
    return [[seg] for seg in boundary]
end

"""
    struct CircularHoleBilliard{T} <: AbsBilliard where {T<:Real}

A circular outer billiard with zero or more circular holes.

Convention:
- `full_boundary[1]` is the outer boundary,
- `full_boundary[2:end]` are inner holes.
- `desymmetrized_full_boundary` is grouped by connected component.

If no symmetry reduction is built in, then the desymmetrized/fundamental boundary
is just the full boundary grouped componentwise.
"""
struct CircularHoleBilliard{T}<:AbsBilliard where {T<:Real}
    full_boundary::Vector{CircleSegment{T}}
    desymmetrized_full_boundary::Vector{Vector{CircleSegment{T}}}
    length::T
    length_fundamental::T
    area::T
    outer_radius::T
    outer_center::SVector{2,T}
    hole_radii::Vector{T}
    hole_centers::Vector{SVector{2,T}}
    area_fundamental::T
    angles::Vector{T}
    angles_fundamental::Vector{T}
    s_shift::T
end

"""
    CircularHoleBilliard(R::T; x0=zero(T), y0=zero(T), rot_angle=zero(T), holes=Tuple{T,T,T}[]) where {T<:Real}

Construct a circular billiard with zero or more circular holes.

No symmetry reduction is built into this general constructor, so:
- `desymmetrized_full_boundary` is just the full boundary grouped by component,
- `area_fundamental = area`,
- `length_fundamental = length`.
"""
function CircularHoleBilliard(R::T;x0=zero(T),y0=zero(T),rot_angle=zero(T),holes::Vector{Tuple{T,T,T}}=Tuple{T,T,T}[]) where {T<:Real}
    full_boundary,center=make_circle_with_holes(R;x0=x0,y0=y0,rot_angle=rot_angle,holes=holes)
    desymmetrized_full_boundary=_group_full_boundary_components(full_boundary)

    hole_radii=T[h[1] for h in holes]
    hole_centers=SVector{2,T}[SVector(h[2],h[3]) for h in holes]

    area=pi*R^2-sum(pi*rr^2 for rr in hole_radii)
    length=2*pi*R+sum(2*pi*rr for rr in hole_radii)

    area_fundamental=area
    length_fundamental=length

    angles=T[]
    angles_fundamental=T[]
    s_shift=zero(T)

    return CircularHoleBilliard(
        full_boundary,
        desymmetrized_full_boundary,
        length,
        length_fundamental,
        area,
        R,
        center,
        hole_radii,
        hole_centers,
        area_fundamental,
        angles,
        angles_fundamental,
        s_shift
    )
end

"""
    struct AnnularBilliard{T}<:AbsBilliard where {T<:Real}

A one-hole circular billiard.
- If the inner hole is centered at the outer center, this is a true annulus.
- If not, this is an off-center circular-hole billiard.

This constructor uses the quarter annulus as the desymmetrized boundary, intended
for symmetry-reduced CFIE computations with XY reflection symmetry.
"""
struct AnnularBilliard{T}<:AbsBilliard where {T<:Real}
    full_boundary::Vector{CircleSegment{T}}
    desymmetrized_full_boundary::Vector{Vector{CircleSegment{T}}}
    length::T
    length_fundamental::T
    area::T
    outer_radius::T
    inner_radius::T
    outer_center::SVector{2,T}
    inner_center::SVector{2,T}
    area_fundamental::T
    angles::Vector{T}
    angles_fundamental::Vector{T}
    s_shift::T
end

"""
    AnnularBilliard(R::T, r::T; xh=zero(T), yh=zero(T),
                    x0=zero(T), y0=zero(T), rot_angle=zero(T)) where {T<:Real}

Construct a circular billiard with one circular hole.

This is the symmetry-aware special case:
- `full_boundary` is the full outer+inner circles,
- `desymmetrized_full_boundary` is the quarter outer arc + quarter inner arc,
- `area_fundamental = area/4`.
"""
function AnnularBilliard(R::T,r::T;xh=zero(T),yh=zero(T),x0=zero(T),y0=zero(T),rot_angle=zero(T)) where {T<:Real}
    full_boundary,outer_center=make_circle_with_holes(R;x0=x0,y0=y0,rot_angle=rot_angle,holes=[(r,xh,yh)])
    desymmetrized_full_boundary,_=make_quarter_full_annulus_boundary(R,r;x0=x0,y0=y0,xh=xh,yh=yh,rot_angle=rot_angle)

    area=pi*R^2-pi*r^2
    length=2*pi*(R+r)

    area_fundamental=area/4
    length_fundamental=sum(crv.length for comp in desymmetrized_full_boundary for crv in comp)

    angles=T[]
    angles_fundamental=T[]
    s_shift=zero(T)

    return AnnularBilliard(
        full_boundary,
        desymmetrized_full_boundary,
        length,
        length_fundamental,
        area,
        R,
        r,
        SVector(x0,y0),
        SVector(xh,yh),
        area_fundamental,
        angles,
        angles_fundamental,
        s_shift
    )
end

"""
    MultiHoleBilliard(R::T, holes::Vector{Tuple{T,T,T}};
                      x0=zero(T), y0=zero(T), rot_angle=zero(T)) where {T<:Real}

Convenience wrapper for a circular outer boundary with multiple circular holes.

Each hole is `(r, hx, hy)`.
"""
function MultiHoleBilliard(R::T,holes::Vector{Tuple{T,T,T}};x0=zero(T),y0=zero(T),rot_angle=zero(T)) where {T<:Real}
    return CircularHoleBilliard(R;x0=x0,y0=y0,rot_angle=rot_angle,holes=holes)
end

"""
    make_annulus_and_basis(R::T, r::T; xh=zero(T), yh=zero(T), x0=zero(T), y0=zero(T), rot_angle=zero(T))

Construct a one-hole circular billiard.
"""
function make_annulus_and_basis(R::T,r::T;xh=zero(T),yh=zero(T),x0=zero(T),y0=zero(T),rot_angle=zero(T)) where {T<:Real}
    billiard=AnnularBilliard(R,r;xh=xh,yh=yh,x0=x0,y0=y0,rot_angle=rot_angle)
    basis=AbstractHankelBasis()
    return billiard,basis
end

"""
    make_circle_with_holes_and_basis(R::T; x0=zero(T), y0=zero(T), rot_angle=zero(T), holes=Tuple{T,T,T}[]) where {T<:Real}

Construct a multi-hole circular billiard with no built-in symmetry reduction.
"""
function make_circle_with_holes_and_basis(R::T;x0=zero(T),y0=zero(T),rot_angle=zero(T),holes::Vector{Tuple{T,T,T}}=Tuple{T,T,T}[]) where {T<:Real}
    billiard=CircularHoleBilliard(R;x0=x0,y0=y0,rot_angle=rot_angle,holes=holes)
    basis=AbstractHankelBasis()
    return billiard,basis
end

"""
    make_multihole_and_basis(R::T, holes::Vector{Tuple{T,T,T}}; x0=zero(T), y0=zero(T), rot_angle=zero(T)) where {T<:Real}

Construct a multi-hole circular billiard with no built-in symmetry reduction.
"""
function make_multihole_and_basis(R::T,holes::Vector{Tuple{T,T,T}};x0=zero(T),y0=zero(T),rot_angle=zero(T)) where {T<:Real}
    billiard=MultiHoleBilliard(R,holes;x0=x0,y0=y0,rot_angle=rot_angle)
    basis=AbstractHankelBasis()
    return billiard,basis
end