"""
    SinaiBilliard{T} <: BilliardGeometry.AbsBilliard

Sinai billiard formed by a centered rectangular outer wall and a centered
circular obstacle.

The rectangular component begins at its positive x-axis midpoint and is
traversed counterclockwise. Its right side is split into two smooth pieces so
that the complete periodic component obeys the canonical exact symmetry-index
convention. The circular obstacle uses zero angular shift and therefore has the
same periodic origin.
"""
struct SinaiBilliard{T}<:BilliardGeometry.AbsBilliard
    full_boundary::Vector{Vector{BilliardGeometry.AbsCurve}}
    symmetries::Vector{BilliardGeometry.AbsSymmetry}
end

"""
    SinaiBilliard(
        a::T,
        b::T,
        R_inner::T;
        center=SVector{2,T}(zero(T),zero(T)),
    ) where {T<:Real} → SinaiBilliard{T}

Construct a centered Sinai billiard with outer dimensions `2a×2b` and a
centered circular obstacle of radius `R_inner`.

Both connected components use the canonical symmetry-compatible periodic
origin on the positive x-axis.
"""
function SinaiBilliard(a::T,b::T,R_inner::T;center=SVector{2,T}(zero(T),zero(T))) where {T<:Real}
    c=SVector{2,T}(center)
    p0=c+SVector{2,T}(a,zero(T))
    p1=c+SVector{2,T}(a,b)
    p2=c+SVector{2,T}(-a,b)
    p3=c+SVector{2,T}(-a,-b)
    p4=c+SVector{2,T}(a,-b)
    bc=BilliardGeometry.SpecularReflection()
    right_upper=BilliardGeometry.LineSegment(p0,p1;bc=bc,domain_id=1,segment_id=1)
    top=BilliardGeometry.LineSegment(p1,p2;bc=bc,domain_id=1,segment_id=2)
    left=BilliardGeometry.LineSegment(p2,p3;bc=bc,domain_id=1,segment_id=3)
    bottom=BilliardGeometry.LineSegment(p3,p4;bc=bc,domain_id=1,segment_id=4)
    right_lower=BilliardGeometry.LineSegment(p4,p0;bc=bc,domain_id=1,segment_id=5)
    inner=BilliardGeometry.CircleSegment(R_inner,T(2pi),zero(T),c;bc=bc,domain_id=1,segment_id=6)
    full_boundary=[
        BilliardGeometry.AbsCurve[right_upper,top,left,bottom,right_lower],
        BilliardGeometry.AbsCurve[inner]
    ]
    symmetries=BilliardGeometry.AbsSymmetry[BilliardGeometry.D2_symmetry...]
    return SinaiBilliard{T}(full_boundary,symmetries)
end