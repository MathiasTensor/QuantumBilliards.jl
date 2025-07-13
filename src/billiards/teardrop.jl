
function make_half_teardrop(;x0=zero(Float64),y0=zero(Float64),rot_angle=zero(Float64))
    origin=SVector(x0,y0)
    φ_multiplier=1.0  # t from 0 to 1 maps to φ from 0 to π
    r_func=t -> begin
        φ=φ_multiplier*π*t  # φ ranges from 0 to π
        SVector(2*sin(φ/2),sin(φ))
    end
    half_teardrop_segment=PolarSegment(r_func;origin=origin,rot_angle=rot_angle)
    pt0=curve(quarter_prosen_segment,zero(T))  # Start point at φ = 0
    pt1=curve(quarter_prosen_segment,one(T))   # End point at φ = π
    line_segment1=VirtualLineSegment(pt1,SVector{2,T}(x0,y0);origin=origin,rot_angle=rot_angle)
    # Construct the boundary
    boundary=Union{PolarSegment,VirtualLineSegment}[half_teardrop_segment,line_segment1]
    # Corners are pt0, pt1, and the origin
    corners=[SVector(x0,y0)]
    return boundary,corners
end

function make_teardrop_desymmetrized_full_boundary(;x0=zero(Float64),y0=zero(Float64),rot_angle=zero(Float64))
    origin=SVector(x0,y0)
    φ_multiplier=1.0  # t from 0 to 1 maps to φ from 0 to π/2
    r_func=t -> begin
        φ=φ_multiplier*π*t  # φ ranges from 0 to π
        SVector(2*sin(φ/2),sin(φ))
    end
    half_teardrop_segment=PolarSegment(r_func;origin=origin,rot_angle=rot_angle)
    boundary=PolarSegment[half_teardrop_segment]
    return boundary,[]
end

function make_full_teardrop(;x0=zero(Float64),y0=zero(Float64),rot_angle=zero(Float64))
    origin=SVector(x0,y0)
    φ_multiplier=2.0  # t from 0 to 1 maps to φ from 0 to 2π
    r_func=t -> begin
        φ=φ_multiplier*π*t  # φ ranges from 0 to π
        SVector(2*sin(φ/2),sin(φ))
    end
    full_teardrop_segment=PolarSegment(r_func;origin=origin,rot_angle=rot_angle)
    area_full=compute_area(full_teardrop_segment)
    boundary=PolarSegment[full_teardrop_segment]
    corners=[origin] 
    return boundary,corners,area_full
end

struct TeardropBilliard{T} <: AbsBilliard where {T<:Real}
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

function TeardropBilliard(;x0=zero(Float64),y0=zero(Float64),rot_angle=zero(Float64)) :: ProsenBilliard 
    fundamental_boundary,_=make_half_teardrop(;x0=x0,y0=y0,rot_angle=rot_angle)
    full_boundary,corners,area_full=make_full_teardrop(;x0=x0,y0=y0,rot_angle=rot_angle)
    desymmetrized_full_boundary,_=make_teardrop_desymmetrized_full_boundary(;x0=x0,y0=y0,rot_angle=rot_angle)
    area_fundamental=area_full*0.5
    length=sum([crv.length for crv in full_boundary])
    length_fundamental=symmetry_accounted_fundamental_boundary_length(fundamental_boundary)
    angles=[]
    angles_fundamental=[]
    s_shift=0.0
    return TeardropBilliard(fundamental_boundary,full_boundary,desymmetrized_full_boundary,length,length_fundamental,corners,area_full,area_fundamental,angles,angles_fundamental,s_shift)
end

function make_teardrop_and_basis(;x0=zero(Float64),y0=zero(Float64),rot_angle=zero(Float64),basis_type=:cafb) 
    teardrop_billiard=TeardropBilliard(;x0=x0,y0=y0,rot_angle=rot_angle)
    symmetry=Vector{Any}([YReflection(-1)])
    if basis_type==:cafb
        basis=CornerAdaptedFourierBessel(10,Float64(pi/2),SVector(x0,y0),rot_angle,symmetry)
    elseif basis_type==:rpw
        basis=RealPlaneWaves(10,symmetry;angle_arc=Float64(pi/2))
    else
        throw(ArgumentError("basis_type must be either :rpw or :cafb"))
    end
    return teardrop_billiard,basis
end










