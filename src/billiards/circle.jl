"""
    make_quarter_circle(radius; x0=zero(T), y0=zero(T), rot_angle=zero(T)) where {T<:Real}

Constructs a quarter circle from a given radius

# Arguments
- `radius::T`: The radius of the quarter circle.
- `x0::T=zero(T)`: The x-coordinate of the origin.
- `y0::T=zero(T)`: The y-coordinate of the origin.
- `rot_angle::T=zero(T)`: The rotation angle of the billiard.

# Returns
- A tuple containing:
  - `boundary::Vector{Union{CircleSegment{T}, VirtualLineSegment{T}}}`: The boundary segments of the quarter circle.
  - `center::SVector{2,T}`: The center point of the quarter circle.
"""
function make_quarter_circle(radius::T; x0=zero(T), y0=zero(T), rot_angle=zero(T)) where {T<:Real}
    origin = SVector(x0, y0)
    center = SVector(x0, y0)
    start_point_qCircle = SVector(zero(T), radius)
    end_point_qCircle = SVector(radius, zero(T))
    quarter_circle = CircleSegment(radius, pi/2, zero(T), zero(T), zero(T); origin=origin, rot_angle=rot_angle)
    virtual_segment_horizontal = VirtualLineSegment(center, end_point_qCircle)
    virtual_segment_vertical = VirtualLineSegment(start_point_qCircle, center)
    boundary = [quarter_circle, virtual_segment_vertical, virtual_segment_horizontal]
    return boundary, center
end

"""
    make_quarter_full_circle_boundary(radius::T; x0=zero(T), y0=zero(T), rot_angle=zero(T)) where {T<:Real}

Constructs the quarter of the full boundary of the circle. This one will be called in the constructor for the `desymmetrized_full_boundary` function for the `boundary_function` calculation.

# Arguments
- `radius::T`: The radius of the circle.
- `x0::T=zero(T)`: The x-coordinate of the origin.
- `y0::T=zero(T)`: The y-coordinate of the origin.
- `rot_angle::T=zero(T)`: The rotation angle of the billiard.

# Returns
- A tuple containing:
  - `boundary::Vector{CircleSegment{T}}`: The boundary segment of the quarter circle.
  - `center::SVector{2,T}`: The center point of the quarter circle.
"""
function make_quarter_full_circle_boundary(radius::T; x0=zero(T), y0=zero(T), rot_angle=zero(T)) where {T<:Real}
    origin = SVector(x0, y0)
    center = SVector(x0, y0)
    quarter_circle = CircleSegment(radius, pi/2, zero(T), zero(T), zero(T); origin=origin, rot_angle=rot_angle)
    boundary = [quarter_circle]
    return boundary, center
end

"""
    make_circle_billiard(radius; x0=zero(T), y0=zero(T), rot_angle=zero(T)) where {T<:Real}

Constructs a circular billiard with a given radius.

# Arguments
- `radius::T`: The radius of the circle.
- `x0::T=zero(T)`: The x-coordinate of the origin.
- `y0::T=zero(T)`: The y-coordinate of the origin.
- `rot_angle::T=zero(T)`: The rotation angle of the billiard.

# Returns
- A tuple containing:
  - `boundary::Vector{CircleSegment{T}}`: The boundary segments of the circle.
  - `center::SVector{2,T}`: The center point of the circle.
"""
function make_circle(radius::T; x0=zero(T), y0=zero(T), rot_angle=zero(T)) where {T<:Real}
    origin = SVector(x0, y0)
    circle = CircleSegment(radius, 2*pi, zero(T), zero(T), zero(T); origin=origin, rot_angle=rot_angle)
    boundary = [circle]
    center = SVector(x0, y0)
    return boundary, center
end

"""
    struct CircleBilliard{T} <: AbsBilliard where {T<:Real}

Defines a circular billiard.

# Fields
- `boundary::Vector{CircleSegment{T}}`: The boundary segments of the circle.
- `length::T`: The total length of the boundary.
- `area::T`: The total area of the circle.
- `radius::T`: The radius of the circle.
- `center::SVector{2,T}`: The center point of the circle.
- `area_fundamental::T`: The total area of the fundamental boundary.
"""
struct CircleBilliard{T} <: AbsBilliard where {T<:Real}
    full_boundary::Vector
    fundamental_boundary::Vector
    desymmetrized_full_boundary::Vector
    length::T
    length_fundamental::T
    area::T
    radius::T
    center::SVector{2,T}
    area_fundamental::T
    angles::Vector
    angles_fundamental::Vector
    s_shift::T
end

"""
    CircleBilliard(radius; x0=zero(T), y0=zero(T)) where {T<:Real}

Constructs a circular billiard with a given radius.

# Arguments
- `radius::T`: The radius of the circle.
- `x0::T=zero(T)`: The x-coordinate of the origin.
- `y0::T=zero(T)`: The y-coordinate of the origin.

# Returns
- An instance of the `CircleBilliard` struct.
"""
function CircleBilliard(radius::T; x0=zero(T), y0=zero(T), rot_angle=zero(T)) :: CircleBilliard where {T<:Real}
    boundary, center = make_circle(radius; x0=x0, y0=y0, rot_angle=rot_angle)
    fundamental_boundary, _ = make_quarter_circle(radius; x0=x0, y0=y0, rot_angle=rot_angle)
    desymmetrized_full_boundary, _ = make_quarter_full_circle_boundary(radius; x0=x0, y0=y0, rot_angle=rot_angle)
    area = pi * radius^2
    area_fundamental = area * 0.25
    length = 2 * pi * radius
    length_fundamental = symmetry_accounted_fundamental_boundary_length(fundamental_boundary)
    angles = []
    angles_fundamental = [pi/2]
    s_shift = 0.0
    return CircleBilliard(boundary, fundamental_boundary, desymmetrized_full_boundary, length, length_fundamental, area, radius, center, area_fundamental, angles, angles_fundamental, s_shift)
end


function make_circle_and_basis(radius::T; x0=zero(T), y0=zero(T), rot_angle=zero(T), basis_type=:rpw) :: Tuple{CircleBilliard{T}, Union{RealPlaneWaves, CornerAdaptedFourierBessel}} where {T<:Real}
    billiard = CircleBilliard(radius; x0=x0, y0=y0, rot_angle=rot_angle)
    symmetry = Vector{Any}([XYReflection(-1, -1)])
    if basis_type == :rpw
        basis = RealPlaneWaves(10, symmetry; angle_arc=Float64(pi/2))
    elseif basis_type == :bessel
        basis = CornerAdaptedFourierBessel(10, pi/2, SVector(0.0, 0.0), 0.0, symmetry)
    else
        @error "Non-valid basis"
    end
    return billiard, basis 
end