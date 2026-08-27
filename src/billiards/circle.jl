struct CircleBilliard{T}<:BilliardGeometry.AbsBilliard
    fundamental_domain::BilliardGeometry.SimpleDomain{T}
    full_boundary::Vector{BilliardGeometry.CircleSegment}
    symmetries::Vector{BilliardGeometry.AbsSymmetry}
end

"""
    CircleBilliard(R::T=one(T);center=SVector{2,T}(zero(T),zero(T))) where {T<:Real}

Construct a circular billiard of radius `R` with D2 reflection symmetry.

## Description
The complete physical boundary is represented by one full
`BilliardGeometry.CircleSegment`.

The fundamental domain is the first-quadrant quarter disk. Its boundary consists
of one physical quarter-circle arc and two reflection boundaries along the
positive coordinate axes.

## Arguments
* `R::T`: Circle radius.

## Keyword Arguments
* `center::SVector{2,T}`: Circle center.

## Returns
* `billiard::CircleBilliard{T}`: Circular billiard containing the fundamental
  domain, complete physical boundary and D2 symmetry descriptors.
"""
function CircleBilliard(R::T=one(T);center=SVector{2,T}(zero(T),zero(T))) where {T<:Real}
    c=SVector{2,T}(center);cx,cy=c
    arc=BilliardGeometry.CircleSegment(R,T(pi/2),zero(T),c;bc=BilliardGeometry.SpecularReflection(),domain_id=1,segment_id=1)
    ywall=BilliardGeometry.LineSegment(SVector{2,T}(cx,cy+R),c;bc=BilliardGeometry.ReflectionSymmetry(BilliardGeometry.YAxisReflection(),4),domain_id=1,segment_id=2)
    xwall=BilliardGeometry.LineSegment(c,SVector{2,T}(cx+R,cy);bc=BilliardGeometry.ReflectionSymmetry(BilliardGeometry.XAxisReflection(),4),domain_id=1,segment_id=3)
    fundamental_domain=BilliardGeometry.SimpleDomain{T}(BilliardGeometry.AbsCurve[arc,ywall,xwall],SVector{2,T}[SVector{2,T}(cx+R,cy),SVector{2,T}(cx,cy+R),c],1)
    full_boundary=[BilliardGeometry.CircleSegment(R,T(2pi),zero(T),c;bc=BilliardGeometry.SpecularReflection(),domain_id=1,segment_id=1)]
    symmetries=BilliardGeometry.AbsSymmetry[BilliardGeometry.D2_symmetry...]
    return CircleBilliard{T}(fundamental_domain,full_boundary,symmetries)
end