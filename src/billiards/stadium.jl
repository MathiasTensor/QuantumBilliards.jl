function make_quarter_stadium(half_width;radius=one(half_width),x0=zero(half_width),y0=zero(half_width),rot_angle=zero(half_width))
    origin=SVector(x0,y0)
    type=typeof(half_width)
    circle=CircleSegment(radius,pi/2,zero(type), half_width, zero(type); origin=origin, rot_angle = rot_angle)
    corners=[SVector(half_width,radius),SVector(zero(type),radius),SVector(zero(type),zero(type)),SVector(half_width+radius,zero(type))]
    line1=LineSegment(corners[1],corners[2];origin=origin,rot_angle=rot_angle)
    line2=VirtualLineSegment(corners[2],corners[3];origin=origin,rot_angle=rot_angle)
    line3=VirtualLineSegment(corners[3],corners[4];origin=origin,rot_angle=rot_angle)
    boundary=[circle,line1,line2,line3]
    return boundary,corners
end

function make_desymmetrized_full_boundary_stadium(half_width;radius=one(half_width),x0=zero(half_width),y0=zero(half_width),rot_angle=zero(half_width))
    origin=SVector(x0,y0)
    type=typeof(half_width)
    circle=CircleSegment(radius,pi/2,zero(type), half_width, zero(type); origin=origin, rot_angle = rot_angle)
    corners=[SVector(half_width, radius), SVector(zero(type), radius)]
    line1=LineSegment(corners[1],corners[2];origin=origin,rot_angle=rot_angle)
    boundary=[circle,line1]
    return boundary,corners
end

function make_full_stadium(half_width;radius=one(half_width),x0=zero(half_width),y0=zero(half_width),rot_angle=zero(half_width))
    origin=SVector(x0,y0)
    type=typeof(half_width)
    corners=[SVector(half_width,radius),SVector(-half_width,radius),SVector(-half_width,-radius),SVector(half_width,-radius)]
    circle1=CircleSegment(radius,1.0*pi, -pi*0.5,half_width,zero(type);origin=origin,rot_angle=rot_angle)
    line1=LineSegment(corners[1],corners[2];origin=origin,rot_angle=rot_angle)
    circle2=CircleSegment(radius,1.0*pi, pi*0.5,-half_width,zero(type);origin=origin,rot_angle=rot_angle)
    line2=LineSegment(corners[3],corners[4];origin=origin,rot_angle=rot_angle)
    boundary=[circle1,line1,circle2,line2]
    return boundary,corners
end

struct Stadium{T} <: AbsBilliard where {T<:Real}
    fundamental_boundary::Vector{Union{LineSegment{T},CircleSegment{T},VirtualLineSegment{T}}}
    full_boundary::Vector{Union{LineSegment{T},CircleSegment{T}}}
    desymmetrized_full_boundary::Vector{LineSegment{T},CircleSegment{T}}
    length::T
    length_fundamental::T
    area::T
    area_fundamental::T
    half_width::T
    radius::T
    corners::Vector{SVector{2,T}}
    angles::Vector
    angles_fundamental::Vector
end

function Stadium(half_width;radius=1.0,x0=0.0,y0=0.0)
    full_boundary,corners=make_full_stadium(half_width;radius=radius,x0=x0,y0=y0)
    area=4.0*half_width*radius+(pi*radius^2)
    area_fundamental=0.25*area
    fundamental_boundary,_=make_quarter_stadium(half_width;radius=radius,x0=x0,y0=y0)
    desymmetrized_full_boundary,_=make_desymmetrized_full_boundary_stadium(half_width;radius=radius,x0=x0,y0=y0)
    length=sum([crv.length for crv in full_boundary])
    length_fundamental=symmetry_accounted_fundamental_boundary_length(fundamental_boundary)
    angles=[]
    angles_fundamental=[pi/2,pi/2]
    return Stadium(fundamental_boundary,full_boundary,desymmetrized_full_boundary,length,length_fundamental,area,area_fundamental,half_width,radius,corners,angles,angles_fundamental)
end 

function make_stadium_and_basis(half_width;radius=1.0,x0=zero(half_width),y0=zero(half_width),rot_angle=zero(half_width),basis_type=:rpw)
    billiard=Stadium(half_width;radius=radius,x0=x0,y0=y0)
    symmetry=Vector{Any}([XYReflection(-1,-1)])
    if basis_type==:rpw
        basis=RealPlaneWaves(10,symmetry;angle_arc=Float64(pi/2))
    elseif basis_type==:bessel
        basis=CornerAdaptedFourierBessel(10,pi/2,SVector(0.0,0.0),0.0,symmetry)
    else
        @error "Non-valid basis"
    end
    return billiard,basis 
end

