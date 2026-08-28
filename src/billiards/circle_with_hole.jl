"""
    AnnularBilliard{T} <: BilliardGeometry.AbsBilliard

Concentric annular billiard bounded by two circles with radii `R_outer` and
`R_inner`.

The physical boundary consists of two connected components,

    full_boundary = [outer_boundary,inner_boundary],

where the first component is the outer wall and the second component is the
interior circular obstacle. Boundary orientation of the inner component is
handled during boundary-point generation.

## Attributes
* `full_boundary::Vector{Vector{BilliardGeometry.AbsCurve}}`: Physical boundary components.
* `symmetries::Vector{BilliardGeometry.AbsSymmetry}`: Geometric symmetries of the billiard.
"""
struct AnnularBilliard{T}<:BilliardGeometry.AbsBilliard
    full_boundary::Vector{Vector{BilliardGeometry.AbsCurve}}
    symmetries::Vector{BilliardGeometry.AbsSymmetry}
end

"""
    AnnularBilliard(R_outer::T,R_inner::T;center=SVector{2,T}(zero(T),zero(T))) where {T<:Real} → AnnularBilliard{T}

Construct a concentric annular billiard with outer radius `R_outer` and inner
radius `R_inner`.

## Arguments
* `R_outer::T`: Radius of the outer circular boundary.
* `R_inner::T`: Radius of the inner circular obstacle.

## Keyword Arguments
* `center::SVector{2,T}`: Common center of the two circular boundaries.

## Returns
* `billiard::AnnularBilliard{T}`: Constructed annular billiard.

## Throws
* `ArgumentError`: If `R_outer≤0`, `R_inner≤0`, or `R_inner≥R_outer`.
"""
function AnnularBilliard(R_outer::T,R_inner::T;center=SVector{2,T}(zero(T),zero(T))) where {T<:Real}
    R_outer>zero(T)||throw(ArgumentError("R_outer must be positive; received $R_outer"))
    R_inner>zero(T)||throw(ArgumentError("R_inner must be positive; received $R_inner"))
    R_inner<R_outer||throw(ArgumentError("R_inner must be smaller than R_outer; received R_inner=$R_inner, R_outer=$R_outer"))
    c=SVector{2,T}(center)
    outer=BilliardGeometry.CircleSegment(R_outer,T(2pi),zero(T),c;bc=BilliardGeometry.SpecularReflection(),domain_id=1,segment_id=1)
    inner=BilliardGeometry.CircleSegment(R_inner,T(2pi),zero(T),c;bc=BilliardGeometry.SpecularReflection(),domain_id=1,segment_id=2)
    full_boundary=[
        BilliardGeometry.AbsCurve[outer],
        BilliardGeometry.AbsCurve[inner]
    ]
    symmetries=BilliardGeometry.AbsSymmetry[BilliardGeometry.D2_symmetry...]
    return AnnularBilliard{T}(full_boundary,symmetries)
end

"""
    CircleStarBilliard{T} <: BilliardGeometry.AbsBilliard

Multiply connected billiard formed by an outer circular wall and a centered
star-shaped polar obstacle

    r(φ)=R_inner+a*cos(nφ).

The physical boundary consists of two connected components,

    full_boundary = [outer_boundary,inner_boundary],

with the star-shaped obstacle represented exactly by a
[`BilliardGeometry.PolarSegment`](@ref).

## Attributes
* `full_boundary::Vector{Vector{BilliardGeometry.AbsCurve}}`: Physical boundary components.
* `symmetries::Vector{BilliardGeometry.AbsSymmetry}`: Geometric symmetries of the billiard.
"""
struct CircleStarBilliard{T}<:BilliardGeometry.AbsBilliard
    full_boundary::Vector{Vector{BilliardGeometry.AbsCurve}}
    symmetries::Vector{BilliardGeometry.AbsSymmetry}
end

"""
    CircleStarBilliard(R_outer::T,R_inner::T,a::T,n::Int;center=SVector{2,T}(zero(T),zero(T))) where {T<:Real} → CircleStarBilliard{T}

Construct a circular billiard containing the centered star-shaped obstacle

    r(φ)=R_inner+a*cos(nφ).

## Arguments
* `R_outer::T`: Radius of the outer circular boundary.
* `R_inner::T`: Mean radius of the inner star-shaped obstacle.
* `a::T`: Amplitude of the radial deformation.
* `n::Int`: Number of angular lobes.

## Keyword Arguments
* `center::SVector{2,T}`: Common center of the outer boundary and inner obstacle.

## Returns
* `billiard::CircleStarBilliard{T}`: Constructed star billiard.

## Throws
* `ArgumentError`: If the radii, deformation, or lobe number do not define a
  positive inner obstacle lying strictly inside the outer circle.
"""
function CircleStarBilliard(R_outer::T,R_inner::T,a::T,n::Int;center=SVector{2,T}(zero(T),zero(T))) where {T<:Real}
    R_outer>zero(T)||throw(ArgumentError("R_outer must be positive; received $R_outer"))
    R_inner>zero(T)||throw(ArgumentError("R_inner must be positive; received $R_inner"))
    n>=2||throw(ArgumentError("n must be at least 2; received $n"))
    R_inner-abs(a)>zero(T)||throw(ArgumentError("R_inner-|a| must be positive; received R_inner=$R_inner, a=$a"))
    R_inner+abs(a)<R_outer||throw(ArgumentError("The inner obstacle must lie strictly inside the outer circle; received R_inner+|a|=$(R_inner+abs(a)), R_outer=$R_outer"))
    c=SVector{2,T}(center)
    coef=SVector{2*n,T}(ntuple(i->i==2*n ? a : zero(T),2*n))
    outer=BilliardGeometry.CircleSegment(R_outer,T(2pi),zero(T),c;bc=BilliardGeometry.SpecularReflection(),domain_id=1,segment_id=1)
    inner=BilliardGeometry.PolarSegment(coef;R=R_inner,arc_angle=T(2pi),shift_angle=zero(T),center=c,bc=BilliardGeometry.SpecularReflection(),domain_id=1,segment_id=2)
    full_boundary=[
        BilliardGeometry.AbsCurve[outer],
        BilliardGeometry.AbsCurve[inner]
    ]
    symmetries=BilliardGeometry.AbsSymmetry[BilliardGeometry.XAxisReflection(),BilliardGeometry.Cn_symmetry(n)...]
    if iseven(n)
        push!(symmetries,BilliardGeometry.YAxisReflection())
        push!(symmetries,BilliardGeometry.XYAxisReflection())
    end
    return CircleStarBilliard{T}(full_boundary,symmetries)
end