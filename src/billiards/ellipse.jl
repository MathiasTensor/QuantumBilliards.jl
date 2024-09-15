





"""
    make_quarter_ellipse(a::T, b::T; x0=zero(T), y0=zero(T), rot_angle=zero(T)) where {T<:Real}

Constructs a quarter ellipse billiard with virtual line segments along the x and y axes.

# Arguments
- `a::T`: The semi-major axis.
- `b::T`: The semi-minor axis.
- `x0::T=zero(T)`: The x-coordinate of the center of the ellipse.
- `y0::T=zero(T)`: The y-coordinate of the center of the ellipse.
- `rot_angle::T=zero(T)`: The rotation angle of the ellipse.

# Returns
- A tuple containing:
  - `boundary::Vector{Union{PolarSegment, VirtualLineSegment}}`: The boundary segments of the quarter ellipse.
  - `corners::Vector{SVector{2,T}}`: The corner points on the x- and y-axes.
"""
function make_quarter_ellipse(a::T, b::T; x0=zero(T), y0=zero(T), rot_angle=zero(T)) where {T<:Real}
    origin = SVector(x0, y0)

    # Define the parametric equation for the quarter ellipse (t runs from 0 to 1 for the full quarter)
    r_func = t -> SVector(a * cos(0.5 * pi * t), b * sin(0.5 * pi * t))

    # Create the quarter ellipse segment
    quarter_ellipse_segment = PolarSegment(r_func; origin=origin, rot_angle=rot_angle)

    # Define the virtual line segments: from (0, b) to (0, 0), and (0, 0) to (a, 0)
    y_axis_segment = VirtualLineSegment(SVector(zero(T), b), SVector(zero(T), zero(T)); origin=origin, rot_angle=rot_angle)
    x_axis_segment = VirtualLineSegment(SVector(zero(T), zero(T)), SVector(a, zero(T)); origin=origin, rot_angle=rot_angle)

    # The segments should be ordered counterclockwise: PolarSegment -> y_axis_segment -> x_axis_segment
    boundary = Union{PolarSegment, VirtualLineSegment}[quarter_ellipse_segment, y_axis_segment, x_axis_segment]
    
    # Corners at the ends of the virtual line segments
    corners = [SVector(a, zero(T)), SVector(zero(T), b)]

    return boundary, corners
end


"""
    make_full_ellipse(a::T, b::T; x0=zero(T), y0=zero(T), rot_angle=zero(T)) where {T<:Real}

Constructs a full ellipse billiard.

# Arguments
- `a::T`: The semi-major axis of the ellipse.
- `b::T`: The semi-minor axis of the ellipse.
- `x0::T=zero(T)`: The x-coordinate of the origin (center of the ellipse).
- `y0::T=zero(T)`: The y-coordinate of the origin (center of the ellipse).
- `rot_angle::T=zero(T)`: The rotation angle of the billiard table.

# Returns
- A tuple containing:
  - `boundary::Vector{PolarSegment}`: The boundary segments of the full ellipse.
"""
function make_full_ellipse(a::T, b::T; x0=zero(T), y0=zero(T), rot_angle=zero(T)) where {T<:Real}
    origin = SVector(x0, y0)
    
    # Define the full ellipse boundary
    r_func = t -> SVector(a * cos(2 * pi * t), b * sin(2 * pi * t))  # t from 0 to 1 for the full ellipse
    full_ellipse_segment = PolarSegment(r_func; origin=origin, rot_angle=rot_angle)
    
    boundary = [full_ellipse_segment]
    corners = [SVector(-a, zero(T)), SVector(a, zero(T))]  # The corner points on the x-axis
    
    return boundary, corners
end



"""
    struct Ellipse{T} <: AbsBilliard where {T<:Real}

Defines an Ellipse billiard with a full and half boundary.

# Fields
- `fundamental_boundary::Vector`: The boundary segments of the half ellipse.
- `full_boundary::Vector`: The boundary segments of the full ellipse.
- `length::T`: The total length of the boundary.
- `area::T`: The total area of the ellipse.
- `semi_major_axis::T`: The semi-major axis of the ellipse.
- `semi_minor_axis::T`: The semi-minor axis of the ellipse.
"""
struct Ellipse{T} <: AbsBilliard where {T<:Real}
    fundamental_boundary::Vector
    full_boundary::Vector
    length::T
    area::T
    semi_major_axis::T
    semi_minor_axis::T
    corners::Vector{SVector{2,T}}
end


"""
    Ellipse(a::T, b::T; x0=zero(T), y0=zero(T), rot_angle=zero(T)) :: Ellipse where {T<:Real}

Constructs an Ellipse billiard.

# Arguments
- `a::T`: The semi-major axis.
- `b::T`: The semi-minor axis.
- `x0::T=zero(T)`: The x-coordinate of the center of the ellipse.
- `y0::T=zero(T)`: The y-coordinate of the center of the ellipse.

# Returns
- An instance of the `Ellipse` struct.
"""
function Ellipse(a::T, b::T; x0=zero(T), y0=zero(T), rot_angle=zero(T)) :: Ellipse where {T<:Real}
    fundamental_boundary, _ = make_quarter_ellipse(a, b; x0=x0, y0=y0, rot_angle=rot_angle)
    full_boundary, corners = make_full_ellipse(a, b; x0=x0, y0=y0, rot_angle=rot_angle)
    area = pi * a * b
    length = sum([crv.length for crv in full_boundary])
    return Ellipse(fundamental_boundary, full_boundary, length, area, a, b, corners)
end


"""
    make_ellipse_and_basis(a::T, b::T; x0=zero(T), y0=zero(T), rot_angle=zero(T)) :: Tuple{Ellipse, QuantumBilliards.RealPlaneWaves} where {T<:Real}

Constructs an ellipse billiard and the corresponding symmetry-adapted basis.

# Arguments
- `a::T`: The semi-major axis.
- `b::T`: The semi-minor axis.
- `x0::T=zero(T)`: The x-coordinate of the center of the ellipse.
- `y0::T=zero(T)`: The y-coordinate of the center of the ellipse.
- `rot_angle::T=zero(T)`: The rotation angle of the ellipse.

# Returns
- A tuple containing:
  - `ellipse::Ellipse`: The constructed ellipse billiard.
  - `basis::QuantumBilliards.RealPlaneWaves`: The symmetry-adapted basis.
"""
function make_ellipse_and_basis(a::T, b::T; x0=zero(T), y0=zero(T), rot_angle=zero(T)) :: Tuple{Ellipse{T}, RealPlaneWaves} where {T<:Real}
    ellipse = Ellipse(a, b; x0=x0, y0=y0, rot_angle=rot_angle)
    symmetry = Vector{Any}([XYReflection(-1, -1)])
    basis = RealPlaneWaves(10, symmetry; angle_arc=Float64(pi/2))
    return ellipse, basis
end

