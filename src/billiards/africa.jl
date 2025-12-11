###########################
# AFRICA BILLIARD (1-fold)#
###########################

function make_africa_full(a::T;x0::T=zero(T),y0::T=zero(T),
    rot_angle::T=zero(T),scale::T=one(T)) where {T<:Real}
    origin=SVector(x0,y0)
    r_func=t->begin
        φ=T(2π)*t
        r=scale*(one(T)+a*cos(φ)+a*cos(T(2)*φ)+a*cos(T(3)*φ+T(π)/T(3)))
        SVector(r*cos(φ),r*sin(φ))
    end
    seg=PolarSegment(r_func;origin=origin,rot_angle=rot_angle)
    area=compute_area(seg)
    return PolarSegment[seg],SVector{2,T}[],area
end

function make_africa_fundamental(a::T;x0::T=zero(T),y0::T=zero(T),
    rot_angle::T=zero(T),scale::T=one(T)) where {T<:Real}
    # For compatibility with C3: "fundamental" = just the full boundary,
    # with two virtual closing segments to itself (degenerate wedge)
    full,corners,area=make_africa_full(a;x0=x0,y0=y0,rot_angle=rot_angle,scale=scale)
    seg=full[1]
    pt0=curve(seg,zero(T))
    pt1=curve(seg,one(T))
    e1=VirtualLineSegment(pt1,SVector(x0,y0);origin=SVector(x0,y0),rot_angle=rot_angle)
    e0=VirtualLineSegment(SVector(x0,y0),pt0;origin=SVector(x0,y0),rot_angle=rot_angle)
    fundamental=Union{PolarSegment,VirtualLineSegment}[seg,e1,e0]
    return fundamental,[pt0,pt1,SVector(x0,y0)],area
end

function make_africa_desym_full(a::T;x0::T=zero(T),y0::T=zero(T),
    rot_angle::T=zero(T),scale::T=one(T)) where {T<:Real}
    full,corners,area=make_africa_full(a;x0=x0,y0=y0,rot_angle=rot_angle,scale=scale)
    return full,Vector{SVector{2,T}}()
end

struct AfricaBilliard{T} <: AbsBilliard where {T<:Real}
    fundamental_boundary::Vector{Union{PolarSegment,VirtualLineSegment}}
    full_boundary::Vector{PolarSegment}
    desymmetrized_full_boundary::Vector{PolarSegment}
    length::T
    length_fundamental::T
    a::T
    corners::Vector{SVector{2,T}}
    area::T
    area_fundamental::T
    angles::Vector
    angles_fundamental::Vector
    s_shift::T
end

function AfricaBilliard(a::T;x0::T=zero(T),y0::T=zero(T),rot_angle::T=zero(T),scale::T=one(T)) where {T<:Real}
    fundamental,corners_fund,area_full=make_africa_fundamental(a;x0=x0,y0=y0,rot_angle=rot_angle,scale=scale)
    full,_,_=make_africa_full(a;x0=x0,y0=y0,rot_angle=rot_angle,scale=scale)
    desym,_=make_africa_desym_full(a;x0=x0,y0=y0,rot_angle=rot_angle,scale=scale)
    length=sum(crv.length for crv in full)
    length_fundamental=length
    area_fundamental=area_full
    return AfricaBilliard(fundamental,full,desym,length,length_fundamental,a,corners_fund,area_full,area_fundamental,Any[],Any[],zero(T))
end

function make_africa_and_basis(a::T;x0::T=zero(T),y0::T=zero(T),rot_angle::T=zero(T),scale::T=one(T),nbasis::Int=14) where {T<:Real}
    billiard=AfricaBilliard(a;x0=x0,y0=y0,rot_angle=rot_angle,scale=scale)
    basis=CornerAdaptedFourierBessel(nbasis,Float64(2*pi/3),SVector(x0,y0),rot_angle)
    return billiard,basis
end