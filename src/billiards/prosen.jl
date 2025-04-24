"""
    make_quarter_prosen(a::T; x0=zero(T), y0=zero(T), rot_angle=zero(T)) where {T<:Real}

Constructs a desymmetrized (quarter) Prosen billiard.

# Arguments
- `a::T`: The deformation parameter.
- `x0::T=zero(T)`: The x-coordinate of the center of the billiard.
- `y0::T=zero(T)`: The y-coordinate of the center of the billiard.
- `rot_angle::T=zero(T)`: The rotation angle of the billiard.

# Returns
- A tuple containing:
  - `boundary::Vector{Union{PolarSegment{T}, VirtualLineSegment{T}}}`: The boundary segments of the quarter Prosen billiard.
  - `corners::Vector{SVector{2,T}}`: The corner points of the quarter billiard.
"""
function make_quarter_prosen(a::T;x0=zero(T),y0=zero(T),rot_angle=zero(T)) where {T<:Real}
    origin=SVector(x0,y0)
    φ_multiplier=0.5  # t from 0 to 1 maps to φ from 0 to π/2
    # Define the radial function for the quarter Prosen billiard
    r_func=t -> begin
        φ=φ_multiplier*π*t  # φ ranges from 0 to π/2
        r=one(T)+a*cos(4*φ)
        SVector(r*cos(φ),r*sin(φ))
    end
    quarter_prosen_segment=PolarSegment(r_func;origin=origin,rot_angle=rot_angle) # quarter Prosen billiard segment
    pt0=curve(quarter_prosen_segment,zero(T))  # Start point at φ = 0
    pt1=curve(quarter_prosen_segment,one(T))   # End point at φ = π/2
    # Create virtual line segments to close the quarter billiard
    line_segment1=VirtualLineSegment(pt1,SVector{2,T}(x0,y0);origin=origin,rot_angle=rot_angle)
    line_segment2=VirtualLineSegment(SVector{2,T}(x0,y0),pt0;origin=origin,rot_angle=rot_angle)
    # Construct the boundary
    boundary=Union{PolarSegment,VirtualLineSegment}[quarter_prosen_segment,line_segment1,line_segment2]
    # Corners are pt0, pt1, and the origin
    corners=[pt0,pt1,SVector(x0,y0)]
    return boundary,corners
end

"""
    make_prosen_desymmetrized_full_boundary(a::T; x0=zero(T), y0=zero(T), rot_angle=zero(T)) where {T<:Real}

Constructs the quarter of the actual full boundary for the `ProsenBilliard`. Used for the construction of the boundary function through `boundary_function`.

# Arguments
- `a::T`: The deformation parameter.
- `x0::T=zero(T)`: The x-coordinate of the center of the billiard.
- `y0::T=zero(T)`: The y-coordinate of the center of the billiard.
- `rot_angle::T=zero(T)`: The rotation angle of the billiard.

# Returns
- A tuple containing:
  - `boundary::Vector{PolarSegment{T}}`: The `PolarSegment` boundary segment of the quarter `ProsenBilliard`.
  - `nothing`: An empty `Nothing` value (no corners for the quarter billiard).
"""
function make_prosen_desymmetrized_full_boundary(a::T;x0=zero(T),y0=zero(T),rot_angle=zero(T)) where {T<:Real}
    origin=SVector(x0,y0)
    φ_multiplier=0.5  # t from 0 to 1 maps to φ from 0 to π/2
    # Define the radial function for the quarter Prosen billiard
    r_func=t -> begin
        φ=φ_multiplier*π*t  # φ ranges from 0 to π/2
        r=one(T)+a*cos(4*φ)
        SVector(r*cos(φ),r*sin(φ))
    end
    quarter_prosen_segment=PolarSegment(r_func;origin=origin,rot_angle=rot_angle)
    pt0=curve(quarter_prosen_segment,zero(T))  # Start point at φ = 0
    pt1=curve(quarter_prosen_segment,one(T))   # End point at φ = π/2
    boundary=PolarSegment[quarter_prosen_segment]
    return boundary,[]
end



"""
    make_full_prosen(a::T; x0=zero(T), y0=zero(T), rot_angle=zero(T)) where {T<:Real}

Constructs a full Prosen billiard.

# Arguments
- `a::T`: The deformation parameter.
- `x0::T=zero(T)`: The x-coordinate of the center of the billiard.
- `y0::T=zero(T)`: The y-coordinate of the center of the billiard.
- `rot_angle::T=zero(T)`: The rotation angle of the billiard.

# Returns
- A tuple containing:
  - `boundary::Vector{PolarSegment{T}}`: The boundary segments of the full Prosen billiard.
  - `corners::Vector{SVector{2,T}}`: An empty vector (no corners for the full billiard).
  - `area_full::T`: The area of the full Prosen billiard.
"""
function make_full_prosen(a::T;x0=zero(T),y0=zero(T),rot_angle=zero(T)) where {T<:Real}
    origin=SVector(x0,y0)
    θ_multiplier=2.0  # t from 0 to 1 maps to φ from 0 to 2π
    # Define the radial function for the full Prosen billiard
    r_func=t -> begin
        φ=θ_multiplier*π*t  # φ ranges from 0 to 2π
        r=one(T)+a*cos(4*φ)
        SVector(r*cos(φ),r*sin(φ))
    end
    full_prosen_segment=PolarSegment(r_func;origin=origin,rot_angle=rot_angle)
    area_full=compute_area(full_prosen_segment)
    boundary=PolarSegment[full_prosen_segment]
    corners=[] 
    return boundary,corners,area_full
end


"""
    struct ProsenBilliard{T} <: QuantumBilliards.AbsBilliard where {T<:Real}

Defines a Prosen billiard with quarter, half, and full boundaries.

# Fields
- `fundamental_boundary::Vector`: The boundary segments of the quarter Prosen billiard.
- `full_boundary::Vector`: The boundary segments of the full Prosen billiard.
- `length::T`: The total length of the full boundary.
- `a::T`: The deformation parameter.
- `corners::Vector{SVector{2,T}}`: The corner points of the billiard.
"""
struct ProsenBilliard{T} <: AbsBilliard where {T<:Real}
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


"""
    ProsenBilliard(a::T; x0=zero(T), y0=zero(T), rot_angle=zero(T)) :: ProsenBilliard where {T<:Real}

Constructs a Prosen billiard.

# Arguments
- `a::T`: The deformation parameter.
- `x0::T=zero(T)`: The x-coordinate of the center of the billiard.
- `y0::T=zero(T)`: The y-coordinate of the center of the billiard.
- `rot_angle::T=zero(T)`: The rotation angle of the billiard.

# Returns
- An instance of the `ProsenBilliard` struct.
"""
function ProsenBilliard(a::T;x0=zero(T),y0=zero(T),rot_angle=zero(T)) :: ProsenBilliard where {T<:Real}
    fundamental_boundary,corners=make_quarter_prosen(a;x0=x0,y0=y0,rot_angle=rot_angle)
    full_boundary,_,area_full=make_full_prosen(a;x0=x0,y0=y0,rot_angle=rot_angle)
    desymmetrized_full_boundary,_=make_prosen_desymmetrized_full_boundary(a; x0=x0, y0=y0, rot_angle=rot_angle)
    area_fundamental=area_full*0.25
    length=sum([crv.length for crv in full_boundary])
    length_fundamental=symmetry_accounted_fundamental_boundary_length(fundamental_boundary)
    angles=[]
    angles_fundamental=[pi/2]
    s_shift=0.0
    return ProsenBilliard(fundamental_boundary,full_boundary,desymmetrized_full_boundary,length,length_fundamental,a,corners,area_full,area_fundamental,angles,angles_fundamental,s_shift)
end



"""
    make_prosen_and_basis(a::T; x0=zero(T), y0=zero(T), rot_angle=zero(T)) :: Tuple{ProsenBilliard{T}, Ba} where {T<:Real, Ba<:AbsBasis}

Constructs a Prosen billiard and the corresponding symmetry-adapted basis.

# Arguments
- `a::T`: The deformation parameter.
- `x0::T=zero(T)`: The x-coordinate of the center of the billiard.
- `y0::T=zero(T)`: The y-coordinate of the center of the billiard.
- `rot_angle::T=zero(T)`: The rotation angle of the billiard.
- `basis_type::Symbol=:cafb`: The type of basis to use, possible are :rpw and :cafb.

# Returns
- A tuple containing:
  - `prosen_billiard::ProsenBilliard`: The constructed Prosen billiard.
  - `basis<:AbsBasis`: The symmetry-adapted basis.
"""
function make_prosen_and_basis(a::T;x0=zero(T),y0=zero(T),rot_angle=zero(T),basis_type=:cafb) where {T<:Real}
    prosen_billiard=ProsenBilliard(a;x0=x0,y0=y0,rot_angle=rot_angle)
    symmetry=Vector{Any}([XYReflection(-1,-1)])
    if basis_type==:cafb
        basis=CornerAdaptedFourierBessel(10,Float64(pi/2),SVector(x0,y0),rot_angle,symmetry)
    elseif basis_type==:rpw
        basis=RealPlaneWaves(10,symmetry;angle_arc=Float64(pi/2))
    else
        throw(ArgumentError("basis_type must be either :rpw or :cafb"))
    end
    return prosen_billiard,basis
end










