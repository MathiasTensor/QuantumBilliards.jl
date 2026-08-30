"""
    StarBilliard{T} <: BilliardGeometry.AbsBilliard

Simply connected star-shaped billiard

    r(φ)=R+a*cos(nφ),

with `n`-fold rotational symmetry. The boundary is one smooth closed
`PolarSegment` starting at the positive x-axis.
"""
struct StarBilliard{T}<:BilliardGeometry.AbsBilliard
    full_boundary::Vector{BilliardGeometry.AbsCurve}
    symmetries::Vector{BilliardGeometry.AbsSymmetry}
end

"""
    StarBilliard(R::T,a::T,n::Int;center=SVector{2,T}(zero(T),zero(T))) where {T<:Real}

Construct the smooth star billiard `r(φ)=R+a*cos(nφ)`. Requires `n≥2` and
`R>|a|`, so the radial function remains positive.
"""
function StarBilliard(R::T,a::T,n::Int;center=SVector{2,T}(zero(T),zero(T))) where {T<:Real}
    R>zero(T)||throw(ArgumentError("R must be positive; received $R"))
    n>=2||throw(ArgumentError("n must be at least 2; received $n"))
    abs(a)<R||throw(ArgumentError("Require |a|<R; received R=$R, a=$a"))
    c=SVector{2,T}(center)
    coef=SVector{2*n,T}(ntuple(i->i==2*n ? a : zero(T),2*n))
    boundary=BilliardGeometry.PolarSegment(coef;R=R,arc_angle=T(2pi),shift_angle=zero(T),center=c,bc=BilliardGeometry.SpecularReflection(),domain_id=1,segment_id=1)
    full_boundary=BilliardGeometry.AbsCurve[boundary]
    symmetries=BilliardGeometry.AbsSymmetry[BilliardGeometry.XAxisReflection()]
    if iseven(n)
        push!(symmetries,BilliardGeometry.YAxisReflection())
        push!(symmetries,BilliardGeometry.XYAxisReflection())
    end
    append!(symmetries,BilliardGeometry.Cn_symmetry(n))
    return StarBilliard{T}(full_boundary,symmetries)
end