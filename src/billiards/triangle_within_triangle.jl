"""
    make_equilateral_triangle_component(side;rot_angle=zero(side),x0=zero(side),y0=zero(side))

Construct one equilateral triangle boundary component from three `LineSegment`s.

# Inputs
- `side::T`: side length.
- `rot_angle::T`: rotation angle. `0` gives one vertex pointing upward.
- `x0::T`, `y0::T`: triangle center.

# Returns
- `boundary::Vector{LineSegment}`: three line segments in counter-clockwise order.
- `corners::Vector{SVector{2,T}}`: triangle vertices.
"""
function make_equilateral_triangle_component(side;rot_angle=zero(side),x0=zero(side),y0=zero(side))
    T=typeof(side)
    c=SVector(T(x0),T(y0))
    R=T(side)/sqrt(T(3))
    θ0=T(pi)/T(2)+T(rot_angle)
    corners=[
        c+R*SVector(cos(θ0),sin(θ0)),
        c+R*SVector(cos(θ0+T(2pi)/T(3)),sin(θ0+T(2pi)/T(3))),
        c+R*SVector(cos(θ0+T(4pi)/T(3)),sin(θ0+T(4pi)/T(3)))]
    boundary=LineSegment[
        LineSegment(corners[1],corners[2]),
        LineSegment(corners[2],corners[3]),
        LineSegment(corners[3],corners[1])]
    return boundary,corners
end

"""
    TriangleInTriangle{T} <: AbsBilliard

Equilateral triangle billiard with an equilateral triangular hole.

Default convention:
- outer triangle points upward,
- inner triangle is rotated by π, so it points downward,
- both components are stored counter-clockwise; hole orientation is handled by
  the CFIE/DLP `evaluate_points` machinery via `_reverse_component_orientation`.

# Fields
- `full_boundary`: `[outer_boundary,inner_boundary]`.
- `desymmetrized_full_boundary`: currently same as `full_boundary`.
- `length::T`: total boundary length.
- `length_fundamental::T`: currently equal to `length`.
- `area::T`: outer area minus inner area.
- `area_fundamental::T`: currently equal to `area`.
- `outer_side::T`: side length of outer triangle.
- `inner_side::T`: side length of inner triangle.
- `corners::Vector{SVector{2,T}}`: all six vertices, outer first.
- `angles::Vector{T}`: physical corner angles.
- `angles_fundamental::Vector{T}`: same as `angles`.
"""
struct TriangleInTriangle{T<:Real} <: AbsBilliard
    full_boundary
    desymmetrized_full_boundary
    length::T
    length_fundamental::T
    area::T
    area_fundamental::T
    outer_side::T
    inner_side::T
    corners::Vector{SVector{2,T}}
    angles::Vector{T}
    angles_fundamental::Vector{T}
end

"""
    TriangleInTriangle(outer_side;inner_side=outer_side/3,x0=zero(outer_side),y0=zero(outer_side),outer_rot=zero(outer_side),inner_rot=pi)

Construct an equilateral triangle with a concentric equilateral triangular hole.

# Inputs
- `outer_side::T`: outer triangle side length.
- `inner_side::T`: inner triangle side length.
- `x0::T`, `y0::T`: common center.
- `outer_rot::T`: rotation of the outer triangle.
- `inner_rot::T`: rotation of the inner triangle. Default is `π`.

# Returns
- `TriangleInTriangle{T}`.
"""
function TriangleInTriangle(outer_side;inner_side=outer_side/3,x0=zero(outer_side),y0=zero(outer_side),outer_rot=zero(outer_side),inner_rot=pi)
    T=typeof(outer_side)
    os=T(outer_side)
    is=T(inner_side)
    outer,corners_outer=make_equilateral_triangle_component(os;rot_angle=T(outer_rot),x0=T(x0),y0=T(y0))
    inner,corners_inner=make_equilateral_triangle_component(is;rot_angle=T(inner_rot),x0=T(x0),y0=T(y0))
    full_boundary=[outer,inner]
    desymmetrized_full_boundary=full_boundary
    outer_area=sqrt(T(3))*os^2/T(4)
    inner_area=sqrt(T(3))*is^2/T(4)
    area=outer_area-inner_area
    length=T(3)*(os+is)
    corners=vcat(corners_outer,corners_inner)
    angles=fill(T(pi)/T(3),6)
    return TriangleInTriangle(
        full_boundary,
        desymmetrized_full_boundary,
        length,
        length,
        area,
        area,
        os,
        is,
        corners,
        angles,
        copy(angles))
end

"""
    make_triangle_in_triangle_and_basis(outer_side;inner_side=outer_side/3,x0=zero(outer_side),y0=zero(outer_side))

Construct a `TriangleInTriangle` billiard and return `(billiard,basis)`.

Currently returns `basis = nothing`, since this geometry is mainly intended for
boundary-integral solvers rather than basis methods.
"""
function make_triangle_in_triangle_and_basis(outer_side;inner_side=outer_side/3,x0=zero(outer_side),y0=zero(outer_side))
    billiard=TriangleInTriangle(outer_side;inner_side=inner_side,x0=x0,y0=y0)
    basis=nothing
    return billiard,basis
end