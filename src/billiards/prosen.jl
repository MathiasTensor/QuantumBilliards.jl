"""
    ProsenBilliard{T} <: BilliardGeometry.AbsBilliard

Prosen billiard with polar boundary

    r(φ)=1+a*cos(4φ).

The billiard has fourfold rotational symmetry. The stored fundamental domain is
one quarter of the full billiard, spanning an angular interval of `π/2`.

## Attributes
* `fundamental_domain::BilliardGeometry.PolarDomain{T}`: Quarter-domain fundamental region.
* `full_boundary::Vector{BilliardGeometry.AbsCurve}`: Complete physical boundary.
* `symmetries::Vector{BilliardGeometry.AbsSymmetry}`: Geometric symmetries of the billiard.
"""
struct ProsenBilliard{T}<:BilliardGeometry.AbsBilliard
    fundamental_domain::BilliardGeometry.PolarDomain{T}
    full_boundary::Vector{BilliardGeometry.AbsCurve}
    symmetries::Vector{BilliardGeometry.AbsSymmetry}
end

"""
    ProsenBilliard(a::T;center=SVector{2,T}(zero(T),zero(T))) where {T<:Real} → ProsenBilliard{T}

Construct the Prosen billiard

    r(φ)=1+a*cos(4φ).

## Arguments
* `a::T`: Amplitude of the fourfold radial deformation.

## Keyword Arguments
* `center::SVector{2,T}`: Billiard center.

## Returns
* `billiard::ProsenBilliard{T}`: Constructed Prosen billiard.
"""
function ProsenBilliard(a::T;center=SVector{2,T}(zero(T),zero(T))) where {T<:Real}
    coef=SVector{8,T}(zero(T),zero(T),zero(T),zero(T),zero(T),zero(T),zero(T),a)
    arc=BilliardGeometry.PolarSegment(coef;R=one(T),arc_angle=T(pi/2),shift_angle=zero(T),center=center,bc=BilliardGeometry.SpecularReflection(),domain_id=1,segment_id=1)
    p0=BilliardGeometry.curve(arc,zero(T))
    p1=BilliardGeometry.curve(arc,one(T))
    wall1=BilliardGeometry.LineSegment(center,p0;bc=BilliardGeometry.ReflectionSymmetry(BilliardGeometry.XAxisReflection(),4),domain_id=1,segment_id=2)
    wall2=BilliardGeometry.LineSegment(p1,center;bc=BilliardGeometry.ReflectionSymmetry(BilliardGeometry.YAxisReflection(),4),domain_id=1,segment_id=3)
    fundamental_domain=BilliardGeometry.PolarDomain{T}(BilliardGeometry.AbsCurve[arc,wall2,wall1],SVector{2,T}[p0,p1,center],1)
    full=BilliardGeometry.PolarSegment(coef;R=one(T),arc_angle=T(2pi),shift_angle=zero(T),center=center,bc=BilliardGeometry.SpecularReflection(),domain_id=1,segment_id=1)
    full_boundary=BilliardGeometry.AbsCurve[full]
    symmetries=BilliardGeometry.AbsSymmetry[
        BilliardGeometry.XAxisReflection(),
        BilliardGeometry.YAxisReflection(),
        BilliardGeometry.XYAxisReflection(),
        BilliardGeometry.NFoldRotation(4,0)
    ]
    return ProsenBilliard{T}(fundamental_domain,full_boundary,symmetries)
end