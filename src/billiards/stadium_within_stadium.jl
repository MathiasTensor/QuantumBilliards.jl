# Stadium billiard with optional hole, constructed as a union of two stadiums
# (one for the outer boundary and one for the inner boundary if the hole is present).

"""
    make_desymmetrized_stadium_component(half_width;radius=one(half_width),x0=zero(half_width),y0=zero(half_width),rot_angle=zero(half_width))

Construct the XY-desymmetrized stadium boundary component:
- one quarter circular arc,
- one horizontal line segment.

Returns `(boundary,corners)`.
"""
function make_desymmetrized_stadium_component(half_width;radius=one(half_width),x0=zero(half_width),y0=zero(half_width),rot_angle=zero(half_width))
    origin=SVector(x0,y0)
    T=typeof(half_width)
    circle=CircleSegment(radius,T(pi/2),zero(T),half_width,zero(T);origin=origin,rot_angle=rot_angle)
    corners=[SVector(half_width,radius),SVector(zero(T),radius)]
    line1=LineSegment(corners[1],corners[2];origin=origin,rot_angle=rot_angle)
    boundary=Union{LineSegment,CircleSegment}[circle,line1]
    return boundary,corners
end

"""
    make_full_stadium_component(half_width;radius=one(half_width),x0=zero(half_width),y0=zero(half_width),rot_angle=zero(half_width))

Construct the full stadium boundary component:
- right semicircle,
- top line,
- left semicircle,
- bottom line.

Returns `(boundary,corners)`.
"""
function make_full_stadium_component(half_width;radius=one(half_width),x0=zero(half_width),y0=zero(half_width),rot_angle=zero(half_width))
    origin=SVector(x0,y0)
    T=typeof(half_width)
    corners=[
        SVector( half_width, radius),
        SVector(-half_width, radius),
        SVector(-half_width,-radius),
        SVector( half_width,-radius)]
    circle1=CircleSegment(radius,T(pi),T(-pi/2), half_width,zero(T);origin=origin,rot_angle=rot_angle)
    line1=LineSegment(corners[1],corners[2];origin=origin,rot_angle=rot_angle)
    circle2=CircleSegment(radius,T(pi),T(pi/2),-half_width,zero(T);origin=origin,rot_angle=rot_angle)
    line2=LineSegment(corners[3],corners[4];origin=origin,rot_angle=rot_angle)
    boundary=Union{LineSegment,CircleSegment}[circle1,line1,circle2,line2]
    return boundary,corners
end

"""
    StadiumWithOptionalHole{T}

Stadium billiard with an optional stadium-shaped hole.

Boundary convention:
- no hole:
    `full_boundary = outer_full`
    `desymmetrized_full_boundary = outer_desym`
- with hole:
    `full_boundary = [outer_full,inner_full]`
    `desymmetrized_full_boundary = [outer_desym,inner_desym]`
"""
struct StadiumWithOptionalHole{T}<:AbsBilliard where {T<:Real}
    full_boundary
    desymmetrized_full_boundary
    length::T
    length_fundamental::T
    area::T
    area_fundamental::T
    outer_half_width::T
    outer_radius::T
    inner_half_width::Union{T,Nothing}
    inner_radius::Union{T,Nothing}
    corners::Vector{SVector{2,T}}
    angles::Vector{T}
    angles_fundamental::Vector{T}
end

"""
    StadiumWithOptionalHole(outer_half_width;outer_radius=one(outer_half_width),inner_half_width=nothing,inner_radius=nothing,x0=zero(outer_half_width),y0=zero(outer_half_width))

Construct a stadium billiard, optionally with a concentric stadium hole.
"""
function StadiumWithOptionalHole(outer_half_width;outer_radius=one(outer_half_width),inner_half_width=nothing,inner_radius=nothing,x0=zero(outer_half_width),y0=zero(outer_half_width))
    T=typeof(outer_half_width)
    outer_full,corners=make_full_stadium_component(outer_half_width;radius=outer_radius,x0=x0,y0=y0)
    outer_desym,_=make_desymmetrized_stadium_component(outer_half_width;radius=outer_radius,x0=x0,y0=y0)
    has_hole=!(inner_half_width===nothing || inner_radius===nothing)
    if has_hole
        ihw=T(inner_half_width)
        ir=T(inner_radius)
        inner_full,_=make_full_stadium_component(ihw;radius=ir,x0=x0,y0=y0)
        inner_desym,_=make_desymmetrized_stadium_component(ihw;radius=ir,x0=x0,y0=y0)
        full_boundary=[outer_full,inner_full]
        desymmetrized_full_boundary=[outer_desym,inner_desym]
        outer_area=T(4)*T(outer_half_width)*T(outer_radius)+T(pi)*T(outer_radius)^2
        inner_area=T(4)*ihw*ir+T(pi)*ir^2
        area=outer_area-inner_area
        outer_length=sum(crv.length for crv in outer_full)
        inner_length=sum(crv.length for crv in inner_full)
        length=outer_length+inner_length
        length_fundamental=sum(crv.length for crv in outer_desym)+sum(crv.length for crv in inner_desym)
        inner_half_width_val=ihw
        inner_radius_val=ir
    else
        full_boundary=outer_full
        desymmetrized_full_boundary=outer_desym
        area=T(4)*T(outer_half_width)*T(outer_radius)+T(pi)*T(outer_radius)^2
        length=sum(crv.length for crv in outer_full)
        length_fundamental=sum(crv.length for crv in outer_desym)
        inner_half_width_val=nothing
        inner_radius_val=nothing
    end
    area_fundamental=area/T(4)
    angles=T[]
    angles_fundamental=T[T(pi/2),T(pi/2)]
    return StadiumWithOptionalHole(
        full_boundary,
        desymmetrized_full_boundary,
        T(length),
        T(length_fundamental),
        T(area),
        T(area_fundamental),
        T(outer_half_width),
        T(outer_radius),
        inner_half_width_val,
        inner_radius_val,
        corners,
        angles,
        angles_fundamental)
end

"""
    make_stadium_with_optional_hole_and_basis(outer_half_width;outer_radius=one(outer_half_width),inner_half_width=nothing,inner_radius=nothing,x0=zero(outer_half_width),y0=zero(outer_half_width),basis_type=:cafb)

Construct a stadium-with-optional-hole billiard together with a symmetry-adapted basis.
"""
function make_stadium_with_optional_hole_and_basis(outer_half_width;outer_radius=one(outer_half_width),inner_half_width=nothing,inner_radius=nothing,x0=zero(outer_half_width),y0=zero(outer_half_width),basis_type=:cafb)
    billiard=StadiumWithOptionalHole(outer_half_width;outer_radius=outer_radius,inner_half_width=inner_half_width,inner_radius=inner_radius,x0=x0,y0=y0)
    symmetry=XYReflection(-1,-1)
    if basis_type==:rpw
        basis=RealPlaneWaves(10,symmetry;angle_arc=Float64(pi/2))
    elseif basis_type==:cafb
        basis=CornerAdaptedFourierBessel(10,pi/2,SVector(0.0,0.0),0.0,symmetry)
    else
        error("Non-valid basis")
    end
    return billiard,basis
end