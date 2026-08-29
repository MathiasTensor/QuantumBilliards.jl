"""
    SinaiBilliard{T} <: BilliardGeometry.AbsBilliard

Sinai billiard formed by a rectangular outer wall and a circular interior
obstacle.

The physical boundary consists of two connected components,

    full_boundary = [outer_boundary,inner_boundary],

where the outer component is the four-sided rectangle and the inner component is
the circular obstacle. The outer rectangle is represented by four
[`BilliardGeometry.LineSegment`](@ref) curves, while the obstacle is represented
by one [`BilliardGeometry.CircleSegment`](@ref).

Boundary orientation of the inner component is handled during boundary-point
generation.

## Attributes
* `full_boundary::Vector{Vector{BilliardGeometry.AbsCurve}}`: Physical boundary components.
* `symmetries::Vector{BilliardGeometry.AbsSymmetry}`: Geometric symmetries of the billiard.
"""
struct SinaiBilliard{T}<:BilliardGeometry.AbsBilliard
    full_boundary::Vector{Vector{BilliardGeometry.AbsCurve}}
    symmetries::Vector{BilliardGeometry.AbsSymmetry}
end

"""
    SinaiBilliard(a::T,b::T,R_inner::T;center=SVector{2,T}(zero(T),zero(T))) where {T<:Real} → SinaiBilliard{T}

Construct a centered Sinai billiard with rectangular outer dimensions `2a×2b`
and a circular obstacle of radius `R_inner`.

The outer rectangle has corners

    center + (-a,-b),
    center + ( a,-b),
    center + ( a, b),
    center + (-a, b),

and the circular obstacle is centered at `center`.

## Arguments
* `a::T`: Half-width of the rectangular outer boundary.
* `b::T`: Half-height of the rectangular outer boundary.
* `R_inner::T`: Radius of the circular interior obstacle.

## Keyword Arguments
* `center::SVector{2,T}`: Common center of the rectangle and circular obstacle.

## Returns
* `billiard::SinaiBilliard{T}`: Constructed Sinai billiard.

## Notes
The rectangular outer boundary forms one connected physical component despite
being represented by four line segments,

    outer_boundary = [bottom,right,top,left].

The circular obstacle forms the second connected component. This layout allows
the outer rectangle and inner circle to use different CFIE-Kress
discretizations, for example `CFIE_kress_global_corners` on the rectangle and
`CFIE_kress` on the circle.
"""
function SinaiBilliard(a::T,b::T,R_inner::T;center=SVector{2,T}(zero(T),zero(T))) where {T<:Real}
    c=SVector{2,T}(center)
    p1=c+SVector{2,T}(-a,-b)
    p2=c+SVector{2,T}(a,-b)
    p3=c+SVector{2,T}(a,b)
    p4=c+SVector{2,T}(-a,b)
    bottom=BilliardGeometry.LineSegment(p1,p2;bc=BilliardGeometry.SpecularReflection(),domain_id=1,segment_id=1)
    right=BilliardGeometry.LineSegment(p2,p3;bc=BilliardGeometry.SpecularReflection(),domain_id=1,segment_id=2)
    top=BilliardGeometry.LineSegment(p3,p4;bc=BilliardGeometry.SpecularReflection(),domain_id=1,segment_id=3)
    left=BilliardGeometry.LineSegment(p4,p1;bc=BilliardGeometry.SpecularReflection(),domain_id=1,segment_id=4)
    inner=BilliardGeometry.CircleSegment(R_inner,T(2*pi),zero(T),c;bc=BilliardGeometry.SpecularReflection(),domain_id=1,segment_id=5)
    full_boundary=[BilliardGeometry.AbsCurve[bottom,right,top,left],BilliardGeometry.AbsCurve[inner]]
    symmetries=BilliardGeometry.AbsSymmetry[BilliardGeometry.D2_symmetry...]
    return SinaiBilliard{T}(full_boundary,symmetries)
end