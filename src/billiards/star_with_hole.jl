using StaticArrays,LinearAlgebra

"""
    struct StarHoleBilliard{T}<:AbsBilliard where {T<:Real}

Outer star-shaped billiard with one inner star-shaped hole.
The first boundary component is the outer boundary, the second is the hole.
"""
struct StarHoleBilliard{T}<:AbsBilliard where {T<:Real}
    full_boundary::Vector{PolarSegment{T}}
    fundamental_boundary::Vector{PolarSegment{T}}
    desymmetrized_full_boundary::Vector{PolarSegment{T}}
    length::T
    length_fundamental::T
    area::T
    area_fundamental::T
    center::SVector{2,T}
    angles::Vector{T}
    angles_fundamental::Vector{T}
    s_shift::T
end

@inline function _star_r(R::T,ϵ::T,m::Int,θ::T,φ::T) where {T<:Real}
    return R*(one(T)+ϵ*cos(m*(θ-φ)))
end

function make_star_polar_boundary(R::T,ϵ::T,m::Int;x0::T=zero(T),y0::T=zero(T),rot_angle::T=zero(T)) where {T<:Real}
    rfun=t->begin
        θ=T(2π)*t
        r=_star_r(R,ϵ,m,θ,zero(T))
        SVector(r*cos(θ),r*sin(θ))
    end
    crv=PolarSegment(rfun;origin=SVector(x0,y0),rot_angle=rot_angle)
    return [crv],SVector(x0,y0)
end

function _max_star_radius(R::T,ϵ::T) where {T<:Real}
    return R*(one(T)+abs(ϵ))
end

function _min_star_radius(R::T,ϵ::T) where {T<:Real}
    return R*(one(T)-abs(ϵ))
end

"""
    StarHoleBilliard(;Rout,ϵout,mout,xout=0,yout=0,φout=0,Rin,ϵin,min,xin=0,yin=0,φin=0)

Construct a billiard with an outer star and one inner star hole.
The hole must fit strictly inside the outer boundary.
"""
function StarHoleBilliard(;Rout::T,ϵout::T,mout::Int,xout::T=zero(T),yout::T=zero(T),φout::T=zero(T),Rin::T,ϵin::T,min::Int,xin::T=zero(T),yin::T=zero(T),φin::T=zero(T)) where {T<:Real}
    outer,_=make_star_polar_boundary(Rout,ϵout,mout;x0=xout,y0=yout,rot_angle=φout)
    inner,_=make_star_polar_boundary(Rin,ϵin,min;x0=xin,y0=yin,rot_angle=φin)
    dcent=hypot(xin-xout,yin-yout)
    dcent+_max_star_radius(Rin,ϵin)>=_min_star_radius(Rout,ϵout) && error("Inner star hole is not fully inside outer star.")
    full_boundary=[outer[1],inner[1]]
    Aout=compute_area(outer[1])
    Ain=compute_area(inner[1])
    Lout=outer[1].length
    Lin=inner[1].length
    area=Aout-Ain
    length=Lout+Lin
    center=SVector(xout,yout)
    return StarHoleBilliard(full_boundary,full_boundary,full_boundary,length,length,area,area,center,Any[],Any[],zero(T))
end

"""
    make_star_hole_and_basis(;kwargs...,basis_type=:rpw)

Convenience constructor returning `(billiard,basis)`.
For hole geometries this is intended for CFIE, so symmetry reduction is not used.
"""
function make_star_hole_and_basis(;Rout::T=1.0,ϵout::T=0.20,mout::Int=5,xout::T=zero(T),yout::T=zero(T),φout::T=zero(T),Rin::T=0.20,ϵin::T=0.10,min::Int=7,xin::T=T(0.12),yin::T=T(-0.08),φin::T=T(0.31)) where {T<:Real}
    billiard=StarHoleBilliard(Rout=Rout,ϵout=ϵout,mout=mout,xout=xout,yout=yout,φout=φout,Rin=Rin,ϵin=ϵin,min=min,xin=xin,yin=yin,φin=φin)
    return billiard,AbstractHankelBasis()
end