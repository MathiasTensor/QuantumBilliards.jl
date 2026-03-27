"""
    make_circle(radius::T; x0=zero(T), y0=zero(T), rot_angle=zero(T)) where {T<:Real}

Construct a full circle boundary as a single `CircleSegment`.

# Arguments
- `radius::T`: Circle radius.
- `x0::T=zero(T)`: x-coordinate of the center.
- `y0::T=zero(T)`: y-coordinate of the center.
- `rot_angle::T=zero(T)`: Optional rotation angle.

# Returns
- `(boundary, center)` where
  - `boundary::Vector{CircleSegment{T}}`
  - `center::SVector{2,T}`
"""
function make_circle(radius::T;x0=zero(T),y0=zero(T),rot_angle=zero(T)) where {T<:Real}
    radius<=zero(T) && error("Circle radius must be positive.")
    origin=SVector(x0,y0)
    circle=CircleSegment(radius,2p*i,zero(T),zero(T),zero(T);origin=origin,rot_angle=rot_angle)
    boundary=[circle]
    center=SVector(x0,y0)
    return boundary,center
end

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
        for j in (i + 1):nh
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
    struct CircularHoleBilliard{T} <: AbsBilliard where {T<:Real}

A circular outer billiard with zero or more circular holes.

Convention:
- `full_boundary[1]` is the outer boundary,
- `full_boundary[2:end]` are inner holes.

This is designed to match the CFIE machinery for multiply connected domains.
"""
struct CircularHoleBilliard{T}<:AbsBilliard where {T<:Real}
    full_boundary::Vector{CircleSegment{T}}
    fundamental_boundary::Vector{CircleSegment{T}}
    desymmetrized_full_boundary::Vector{CircleSegment{T}}
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

# Arguments
- `R::T`: Outer radius.
- `x0::T=zero(T)`, `y0::T=zero(T)`: Outer-circle center.
- `rot_angle::T=zero(T)`: Optional rotation angle.
- `holes::Vector{Tuple{T,T,T}}=Tuple{T,T,T}[]`:
    hole list `(r, hx, hy)`.

# Notes
For CFIE use, we set
- `fundamental_boundary = full_boundary`
- `desymmetrized_full_boundary = full_boundary`

since no symmetry reduction is used here.
"""
function CircularHoleBilliard(R::T;x0=zero(T),y0=zero(T),rot_angle=zero(T),holes::Vector{Tuple{T,T,T}}=Tuple{T,T,T}[]) where {T<:Real}
    full_boundary,center=make_circle_with_holes(R;x0=x0,y0=y0,rot_angle=rot_angle,holes=holes)
    hole_radii=T[h[1] for h in holes]
    hole_centers=SVector{2,T}[SVector(h[2],h[3]) for h in holes]
    area=pi*R^2-sum(pi*r^2 for r in hole_radii)
    length=2*pi*R+sum(2*pi*r for r in hole_radii)
    fundamental_boundary=full_boundary
    desymmetrized_full_boundary=full_boundary
    length_fundamental=length
    area_fundamental=area
    angles=T[]
    angles_fundamental=T[]
    s_shift=zero(T)
    return CircularHoleBilliard(full_boundary,fundamental_boundary,desymmetrized_full_boundary,length,length_fundamental,area,R,center,hole_radii,hole_centers,area_fundamental,angles,angles_fundamental,s_shift)
end

"""
    struct AnnularBilliard{T}<:AbsBilliard where {T<:Real}

A one-hole circular billiard.
- If the inner hole is centered at the outer center, this is a true annulus.
- If not, this is an off-center circular-hole billiard.

Stored in the same boundary convention as `CircularHoleBilliard`.
"""
struct AnnularBilliard{T}<:AbsBilliard where {T<:Real}
full_boundary::Vector{CircleSegment{T}}
    fundamental_boundary::Vector{CircleSegment{T}}
    desymmetrized_full_boundary::Vector{CircleSegment{T}}
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

# Arguments
- `R`: outer radius
- `r`: inner-hole radius
- `xh`, `yh`: hole center
- `x0`, `y0`: outer center
"""
function AnnularBilliard(R::T,r::T;xh=zero(T),yh=zero(T),x0=zero(T),y0=zero(T),rot_angle=zero(T)) where {T<:Real}
    boundary,outer_center=make_circle_with_holes(R;x0=x0,y0=y0,rot_angle=rot_angle,holes=[(r,xh,yh)])
    area=pi*R^2-pi*r^2
    length=2*pi*(R+r)
    return AnnularBilliard(boundary,boundary,boundary,length,length,area,R,r,SVector(x0,y0),SVector(xh,yh),area,T[],T[],zero(T))
end

"""
    MultiHoleBilliard(R::T, holes::Vector{Tuple{T,T,T}};
                      x0=zero(T), y0=zero(T), rot_angle=zero(T)) where {T<:Real}

Convenience wrapper for a circular outer boundary with multiple circular holes.

Each hole is `(r, hx, hy)`.
"""
function MultiHoleBilliard(
    R::T,
    holes::Vector{Tuple{T,T,T}};
    x0=zero(T),
    y0=zero(T),
    rot_angle=zero(T)
) where {T<:Real}
    return CircularHoleBilliard(R; x0=x0, y0=y0, rot_angle=rot_angle, holes=holes)
end

"""
    make_annulus_and_basis(R::T, r::T; xh=zero(T), yh=zero(T),  x0=zero(T), y0=zero(T), rot_angle=zero(T))

Construct a one-hole circular billiard together with a placeholder
`AbstractHankelBasis()` object for Beyn / CFIE workflows.
"""
function make_annulus_and_basis(R::T,r::T;xh=zero(T),yh=zero(T),x0=zero(T),y0=zero(T),rot_angle=zero(T)) where {T<:Real}
    billiard=AnnularBilliard(R,r;xh=xh,yh=yh,x0=x0,y0=y0,rot_angle=rot_angle)
    basis=AbstractHankelBasis()
    return billiard,basis
end

"""
    make_circle_with_holes_and_basis(R::T; x0=zero(T), y0=zero(T), rot_angle=zero(T), holes=Tuple{T,T,T}[]) where {T<:Real}

Construct a multi-hole circular billiard together with a placeholder
`AbstractHankelBasis()`.
"""
function make_circle_with_holes_and_basis(R::T;x0=zero(T),y0=zero(T),rot_angle=zero(T),holes::Vector{Tuple{T,T,T}}=Tuple{T,T,T}[]) where {T<:Real}
    billiard=CircularHoleBilliard(R;x0=x0,y0=y0,rot_angle=rot_angle,holes=holes)
    basis=AbstractHankelBasis()
    return billiard,basis
end

"""
    make_multihole_and_basis(R::T, holes::Vector{Tuple{T,T,T}}; x0=zero(T), y0=zero(T), rot_angle=zero(T)) where {T<:Real}

Construct a multi-hole circular billiard together with `AbstractHankelBasis()`.
"""
function make_multihole_and_basis(R::T,holes::Vector{Tuple{T,T,T}};x0=zero(T),y0=zero(T),rot_angle=zero(T)) where {T<:Real}
    billiard=MultiHoleBilliard(R,holes;x0=x0,y0=y0,rot_angle=rot_angle)
    basis=AbstractHankelBasis()
    return billiard,basis
end

############################################################
######################## EXAMPLES ##########################
############################################################

# Example 1: concentric annulus
# billiard, basis = make_annulus_and_basis(1.0, 0.25)

# Example 2: off-center hole
# billiard, basis = make_annulus_and_basis(1.0, 0.2; xh=0.25, yh=0.0)

# Example 3: two holes
# holes = [(0.15, -0.25, 0.0), (0.10, 0.28, 0.12)]
# billiard, basis = make_multihole_and_basis(1.0, holes)

# Example 4: no holes, same type family
# billiard, basis = make_circle_with_holes_and_basis(1.0)