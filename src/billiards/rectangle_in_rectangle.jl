"""
    RectangleWithinRectangleBilliard{T} <: BilliardGeometry.AbsBilliard

Multiply connected billiard formed by a rectangular outer wall and a smaller
rectangular interior obstacle.

The physical boundary consists of two connected components,

    full_boundary = [outer_boundary,inner_boundary],

where each component is represented by four straight boundary segments. The
first component is the outer rectangle and the second component is the
rectangular hole.

## Attributes
* `full_boundary::Vector{Vector{BilliardGeometry.AbsCurve}}`: Physical boundary components.
* `symmetries::Vector{BilliardGeometry.AbsSymmetry}`: Geometric symmetries of the billiard.
"""
struct RectangleWithinRectangleBilliard{T}<:BilliardGeometry.AbsBilliard
    full_boundary::Vector{Vector{BilliardGeometry.AbsCurve}}
    symmetries::Vector{BilliardGeometry.AbsSymmetry}
end

"""
    RectangleWithinRectangleBilliard(
        a_outer::T,
        b_outer::T,
        a_inner::T,
        b_inner::T;
        center=SVector{2,T}(zero(T),zero(T)),
    ) where {T<:Real} → RectangleWithinRectangleBilliard{T}

Construct a centered rectangular billiard containing a smaller centered
rectangular obstacle.

The outer rectangle has dimensions

    2a_outer × 2b_outer,

while the inner rectangular obstacle has dimensions

    2a_inner × 2b_inner.

Both rectangles are centered at `center`.

## Arguments
* `a_outer::T`: Half-width of the outer rectangle.
* `b_outer::T`: Half-height of the outer rectangle.
* `a_inner::T`: Half-width of the inner rectangular obstacle.
* `b_inner::T`: Half-height of the inner rectangular obstacle.

## Keyword Arguments
* `center::SVector{2,T}`: Common center of the outer and inner rectangles.

## Returns
* `billiard::RectangleWithinRectangleBilliard{T}`: Constructed multiply connected billiard.

## Notes
The boundary topology is

    outer_boundary = [bottom,right,top,left]
    inner_boundary = [bottom,right,top,left]

and

    full_boundary = [outer_boundary,inner_boundary].

Thus each rectangle is one connected physical boundary component even though it
is represented by four curve segments.

For CFIE discretization, both components can naturally use
[`CFIE_kress_global_corners`](@ref), potentially with different resolutions or
grading orders through [`CFIE_kress_composite`](@ref).
"""
function RectangleWithinRectangleBilliard(a_outer::T,b_outer::T,a_inner::T,b_inner::T;center=SVector{2,T}(zero(T),zero(T))) where {T<:Real}
    c=SVector{2,T}(center)
    o1=c+SVector{2,T}(-a_outer,-b_outer)
    o2=c+SVector{2,T}( a_outer,-b_outer)
    o3=c+SVector{2,T}( a_outer, b_outer)
    o4=c+SVector{2,T}(-a_outer, b_outer)
    i1=c+SVector{2,T}(-a_inner,-b_inner)
    i2=c+SVector{2,T}( a_inner,-b_inner)
    i3=c+SVector{2,T}( a_inner, b_inner)
    i4=c+SVector{2,T}(-a_inner, b_inner)
    outer_bottom=BilliardGeometry.LineSegment(o1,o2;bc=BilliardGeometry.SpecularReflection(),domain_id=1,segment_id=1)
    outer_right=BilliardGeometry.LineSegment(o2,o3;bc=BilliardGeometry.SpecularReflection(),domain_id=1,segment_id=2)
    outer_top=BilliardGeometry.LineSegment(o3,o4;bc=BilliardGeometry.SpecularReflection(),domain_id=1,segment_id=3)
    outer_left=BilliardGeometry.LineSegment(o4,o1;bc=BilliardGeometry.SpecularReflection(),domain_id=1,segment_id=4)
    inner_bottom=BilliardGeometry.LineSegment(i1,i2;bc=BilliardGeometry.SpecularReflection(),domain_id=1,segment_id=5)
    inner_right=BilliardGeometry.LineSegment(i2,i3;bc=BilliardGeometry.SpecularReflection(),domain_id=1,segment_id=6)
    inner_top=BilliardGeometry.LineSegment(i3,i4;bc=BilliardGeometry.SpecularReflection(),domain_id=1,segment_id=7)
    inner_left=BilliardGeometry.LineSegment(i4,i1;bc=BilliardGeometry.SpecularReflection(),domain_id=1,segment_id=8)
    full_boundary=[BilliardGeometry.AbsCurve[outer_bottom,outer_right,outer_top,outer_left],BilliardGeometry.AbsCurve[inner_bottom,inner_right,inner_top,inner_left]]
    symmetries=BilliardGeometry.AbsSymmetry[BilliardGeometry.D2_symmetry...]
    return RectangleWithinRectangleBilliard{T}(full_boundary,symmetries)
end