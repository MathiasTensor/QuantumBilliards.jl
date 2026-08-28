struct RectangleBilliard{T}<:BilliardGeometry.AbsBilliard
    fundamental_domain::BilliardGeometry.SimpleDomain{T}
    full_boundary::Vector{BilliardGeometry.AbsCurve}
    symmetries::Vector{BilliardGeometry.AbsSymmetry}
end

"""
    RectangleBilliard(a::T=one(T),b::T=one(T);center=SVector{2,T}(zero(T),zero(T))) where {T<:Real}

Construct a rectangle of half-width `a` and half-height `b` with D2 reflection symmetry.

The complete physical billiard is `[-a,a]×[-b,b]`, translated by `center`.
The fundamental domain is the first-quadrant rectangle `[0,a]×[0,b]`.

## Arguments
* `a::T`: Half-width in the x direction.
* `b::T`: Half-height in the y direction.

## Keyword Arguments
* `center::SVector{2,T}`: Rectangle center.

## Returns
* `billiard::RectangleBilliard{T}`: Rectangle with full boundary, first-quadrant fundamental domain and D2 symmetry descriptors.
"""
function RectangleBilliard(a::T=one(T),b::T=one(T);center=SVector{2,T}(zero(T),zero(T))) where {T<:Real}
    a>0||throw(ArgumentError("a must be positive"));b>0||throw(ArgumentError("b must be positive"))
    c=SVector{2,T}(center);cx,cy=c
    p1=SVector{2,T}(cx-a,cy-b);p2=SVector{2,T}(cx+a,cy-b);p3=SVector{2,T}(cx+a,cy+b);p4=SVector{2,T}(cx-a,cy+b)
    q0=c;q1=SVector{2,T}(cx+a,cy);q2=SVector{2,T}(cx+a,cy+b);q3=SVector{2,T}(cx,cy+b)
    full_boundary=BilliardGeometry.AbsCurve[
        BilliardGeometry.LineSegment(p1,p2;bc=BilliardGeometry.SpecularReflection(),domain_id=1,segment_id=1),
        BilliardGeometry.LineSegment(p2,p3;bc=BilliardGeometry.SpecularReflection(),domain_id=1,segment_id=2),
        BilliardGeometry.LineSegment(p3,p4;bc=BilliardGeometry.SpecularReflection(),domain_id=1,segment_id=3),
        BilliardGeometry.LineSegment(p4,p1;bc=BilliardGeometry.SpecularReflection(),domain_id=1,segment_id=4)
    ]
    top=BilliardGeometry.LineSegment(q2,q3;bc=BilliardGeometry.SpecularReflection(),domain_id=1,segment_id=1)
    ywall=BilliardGeometry.LineSegment(q3,q0;bc=BilliardGeometry.ReflectionSymmetry(BilliardGeometry.YAxisReflection(),4),domain_id=1,segment_id=2)
    xwall=BilliardGeometry.LineSegment(q0,q1;bc=BilliardGeometry.ReflectionSymmetry(BilliardGeometry.XAxisReflection(),4),domain_id=1,segment_id=3)
    right=BilliardGeometry.LineSegment(q1,q2;bc=BilliardGeometry.SpecularReflection(),domain_id=1,segment_id=4)
    fundamental_domain=BilliardGeometry.SimpleDomain{T}(BilliardGeometry.AbsCurve[right,top,ywall,xwall],SVector{2,T}[q1,q2,q3,q0],1)
    symmetries=BilliardGeometry.AbsSymmetry[BilliardGeometry.D2_symmetry...]
    return RectangleBilliard{T}(fundamental_domain,full_boundary,symmetries)
end