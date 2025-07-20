
function make_desymmetrized_star(n::Int64;x0=zero(Float64),y0=zero(Float64),rot_angle=zero(Float64))
    origin=SVector(x0,y0)
    r_func=t -> begin
        φ=1/n*2*π*t  # φ ranges from 0 to 2π/n
        s,c=sincos(φ)
        fac=1+sin(n*φ)/4
        return SVector(fac*c,fac*s)
    end
    r_func_rev(t)=r_func(1-t)
    desymmetrized_star=PolarSegment(r_func_rev;origin=origin,rot_angle=rot_angle)
    pt0=curve(desymmetrized_star,zero(Float64)) 
    pt1=curve(desymmetrized_star,one(Float64)) 
    line_segment1=VirtualLineSegment(pt1,origin;origin=origin,rot_angle=rot_angle)
    line_segment2=VirtualLineSegment(origin,pt0;origin=origin,rot_angle=rot_angle)
    boundary=Union{PolarSegment,VirtualLineSegment}[desymmetrized_star,line_segment1,line_segment2]
    corners=[SVector(x0,y0)]
    return boundary,corners
end

function make_star_desymmetrized_full_boundary(n::Int64;x0=zero(Float64),y0=zero(Float64),rot_angle=zero(Float64))
    origin=SVector(x0,y0)
    r_func=t -> begin
        φ=1/n*2*π*t  # φ ranges from 0 to 2π/n
        s,c=sincos(φ)
        fac=1+sin(n*φ)/4
        return SVector(fac*c,fac*s)
    end
    r_func_rev(t)=r_func(1-t)
    desymmetrized_star=PolarSegment(r_func_rev;origin=origin,rot_angle=rot_angle)
    boundary=PolarSegment[desymmetrized_star]
    return boundary,[]
end

function make_full_star(n::Int64;x0=zero(Float64),y0=zero(Float64),rot_angle=zero(Float64))
    origin=SVector(x0,y0)
    r_func=t -> begin
        φ=2*π*t  # φ ranges from 0 to 2π/n
        s,c=sincos(φ)
        fac=1+sin(n*φ)/4
        return SVector(fac*c,fac*s)
    end
    r_func_rev(t)=r_func(1-t)
    desymmetrized_star=PolarSegment(r_func_rev;origin=origin,rot_angle=rot_angle)
    area_full=compute_area(desymmetrized_star)
    boundary=PolarSegment[desymmetrized_star] 
    return boundary,[],area_full
end

struct StarBilliard{T} <: AbsBilliard where {T<:Real}
    fundamental_boundary::Vector{Union{PolarSegment,VirtualLineSegment}}
    full_boundary::Vector{PolarSegment}
    desymmetrized_full_boundary::Vector{PolarSegment}
    length::T
    length_fundamental::T
    corners::Vector{SVector{2,T}}
    area::T
    area_fundamental::T
    angles::Vector
    angles_fundamental::Vector
    s_shift::T
end

function StarBilliard(n::Int64;x0=zero(Float64),y0=zero(Float64),rot_angle=zero(Float64)) :: StarBilliard 
    fundamental_boundary,corners=make_desymmetrized_star(n;x0=x0,y0=y0,rot_angle=rot_angle)
    full_boundary,_,area_full=make_full_star(n;x0=x0,y0=y0,rot_angle=rot_angle)
    desymmetrized_full_boundary,_=make_star_desymmetrized_full_boundary(n;x0=x0,y0=y0,rot_angle=rot_angle)
    area_fundamental=area_full/n
    length=sum([crv.length for crv in full_boundary])
    length_fundamental=symmetry_accounted_fundamental_boundary_length(fundamental_boundary)
    angles=[2*pi/n]
    angles_fundamental=[2*pi/n]
    s_shift=0.0
    return StarBilliard(fundamental_boundary,full_boundary,desymmetrized_full_boundary,length,length_fundamental,corners,area_full,area_fundamental,angles,angles_fundamental,s_shift)
end

function make_star_and_basis(n::Int64;x0=zero(Float64),y0=zero(Float64),rot_angle=zero(Float64)) 
    star_billiard=StarBilliard(n::Int64;x0=x0,y0=y0,rot_angle=rot_angle)
    symmetry=Vector{Any}([]) # not yet implemented
    basis=CornerAdaptedFourierBessel(10,Float64(2*pi/n),SVector(x0,y0),rot_angle,symmetry)
    return star_billiard,basis
end