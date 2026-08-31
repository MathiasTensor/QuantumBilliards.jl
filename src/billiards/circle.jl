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

"""
    CircleWedgeBilliard{T}<:BilliardGeometry.AbsBilliard

Circular billiard with a sector removed along the positive x-axis.

The removed wedge has opening angle `2α`. The remaining boundary consists of
one circular arc and two radial line segments and has `XAxisReflection`
symmetry.
"""
struct CircleWedgeBilliard{T}<:BilliardGeometry.AbsBilliard
    fundamental_domain::BilliardGeometry.SimpleDomain{T}
    full_boundary::Vector{BilliardGeometry.AbsCurve}
    symmetries::Vector{BilliardGeometry.AbsSymmetry}
end

"""
    CircleWedgeBilliard(α::T,R::T=one(T)) where {T<:Real}

Construct a circle of radius `R` with a wedge of opening angle `2α` removed
around the positive x-axis.

## Arguments
* `R::T`: Circle radius.
* `α::T`: Half-angle of the removed wedge.

## Returns
* `CircleWedgeBilliard{T}`: Constructed billiard.
"""
function CircleWedgeBilliard(α::T;R::T=one(T)) where {T<:Real}
    zero(T)<α<T(pi)||throw(ArgumentError("α must satisfy 0<α<π; received α=$α"))
    c=SVector{2,T}(zero(T),zero(T))
    pplus=R*SVector{2,T}(cos(α),sin(α))
    pminus=R*SVector{2,T}(cos(α),-sin(α))
    pleft=SVector{2,T}(-R,zero(T))
    bc=BilliardGeometry.SpecularReflection()
    arc=BilliardGeometry.CircleSegment(R,T(2pi)-T(2)*α;shift_angle=α,center=c,bc=bc,domain_id=1,segment_id=1)
    radial_minus=BilliardGeometry.LineSegment(pminus,c;bc=bc,domain_id=1,segment_id=2)
    radial_plus=BilliardGeometry.LineSegment(c,pplus;bc=bc,domain_id=1,segment_id=3)
    full_boundary=BilliardGeometry.AbsCurve[arc,radial_minus,radial_plus]
    sym=BilliardGeometry.XAxisReflection()
    arc_half=BilliardGeometry.CircleSegment(R,T(pi)-α;shift_angle=α,center=c,bc=bc,domain_id=1,segment_id=1)
    xwall=BilliardGeometry.LineSegment(pleft,c;bc=BilliardGeometry.ReflectionSymmetry(sym,3),domain_id=1,segment_id=2)
    radial=BilliardGeometry.LineSegment(c,pplus;bc=bc,domain_id=1,segment_id=3)
    fundamental_boundary=BilliardGeometry.AbsCurve[arc_half,xwall,radial]
    vertices=SVector{2,T}[pplus,pleft,c]
    fundamental_domain=BilliardGeometry.SimpleDomain{T}(fundamental_boundary,vertices,1)
    symmetries=BilliardGeometry.AbsSymmetry[sym]
    return CircleWedgeBilliard{T}(fundamental_domain,full_boundary,symmetries)
end