




"""
    make_fundamental_equilateral_triangle(h; x0=0.0, y0=0.0, rot_angle=0.0)

Constructs the fundamental domain of an equilateral triangle billiard with specified h length (from 0,0 -> h,0). The fundamental domain is defined by the points `(h, 0)`, rotated by 120 degrees, and `(0, 0)`.

# Arguments
- `h::T`: The height of the triangle from the center.
- `x0::T=0.0`: The x-coordinate of the center of the triangle.
- `y0::T=0.0`: The y-coordinate of the center of the triangle.
- `rot_angle::T=0.0`: The rotation angle of the billiard table.

# Returns
- A tuple containing:
  - `boundary::Vector{Union{LineSegment{T}, VirtualLineSegment{T}}}`: The boundary segments of the fundamental domain.
  - `corners::Vector{SVector{2,T}}`: The corner points of the fundamental domain.
"""
function make_fundamental_equilateral_triangle(h::T; x0::T=0.0, y0::T=0.0, rot_angle::T=0.0) where {T<:Real}
    origin = SVector(x0, y0)

    # Define the corners of the fundamental triangle
    p1 = SVector(h, 0.0)
    angle = 2π / 3  # 120 degrees in radians
    p2 = SVector(
        h * cos(angle),
        h * sin(angle)
    )
    p3 = SVector(0.0, 0.0)

    # Line segments for the fundamental triangle
    side1 = LineSegment(p1, p2; origin=origin, rot_angle=rot_angle)         # Reflective side
    side2 = VirtualLineSegment(p2, p3; origin=origin, rot_angle=rot_angle)  # Virtual side
    side3 = VirtualLineSegment(p3, p1; origin=origin, rot_angle=rot_angle)  # Virtual side

    # Counterclockwise order starting from p1
    boundary = Union{LineSegment{T}, VirtualLineSegment{T}}[
        side1, side2, side3
    ]

    corners = [p1, p2, p3]
    return boundary, corners
end

"""
    make_full_equilateral_triangle(h; x0=0.0, y0=0.0, rot_angle=0.0)

Constructs the full equilateral triangle billiard with specified h. The triangle is centered at `(x0, y0)` with one vertex at `(h, 0)`.

# Arguments
- `h::T`: The height of the triangle from the center.
- `x0::T=0.0`: The x-coordinate of the center of the triangle.
- `y0::T=0.0`: The y-coordinate of the center of the triangle.
- `rot_angle::T=0.0`: The rotation angle of the billiard table.

# Returns
- A tuple containing:
  - `boundary::Vector{LineSegment{T}}`: The boundary segments of the full triangle.
  - `corners::Vector{SVector{2,T}}`: The corner points of the full triangle.
"""
function make_full_equilateral_triangle(h::T; x0::T=0.0, y0::T=0.0, rot_angle::T=0.0) where {T<:Real}
    origin = SVector(x0, y0)

    # Define the corners of the full triangle
    angles = [0, 2π / 3, 4π / 3]  # 0°, 120°, 240°
    corners = [SVector(
        x0 + h * cos(ang + rot_angle),
        y0 + h * sin(ang + rot_angle)
    ) for ang in angles]

    # Line segments for the full triangle in counterclockwise order starting from (h, 0)
    side1 = LineSegment(corners[1], corners[2]; origin=origin, rot_angle=rot_angle)
    side2 = LineSegment(corners[2], corners[3]; origin=origin, rot_angle=rot_angle)
    side3 = LineSegment(corners[3], corners[1]; origin=origin, rot_angle=rot_angle)

    boundary = Vector{LineSegment{T}}[
        side1, side2, side3
    ]

    return boundary, corners
end

"""
    struct EquilateralTriangleBilliard{T} <: AbsBilliard where {T<:Real}

Defines an equilateral triangle billiard with specified h. It includes both the full and fundamental boundaries.

# Fields
- `fundamental_boundary::Vector{Union{LineSegment{T}, VirtualLineSegment{T}}}`: The boundary segments of the fundamental triangle.
- `full_boundary::Vector{LineSegment{T}}`: The boundary segments of the full triangle.
- `length::T`: The total length of the boundary.
- `length_fundamental::T`: The total length of the fundamental boundary.
- `area::T`: The total area of the triangle.
- `area_fundamental::T`: The area of the fundamental domain.
- `side_length::T`: The length of each side of the triangle.
- `corners::Vector{SVector{2,T}}`: The corner points of the full triangle.
- `angles::Vector{T}`: The internal angles of the triangle.
- `angles_fundamental::Vector{T}`: The internal angles of the fundamental domain.
"""
struct EquilateralTriangleBilliard{T} <: AbsBilliard where {T<:Real}
    fundamental_boundary::Vector{Union{LineSegment{T}, VirtualLineSegment{T}}}
    full_boundary::Vector{LineSegment{T}}
    length::T
    length_fundamental::T
    area::T
    area_fundamental::T
    side_length::T
    corners::Vector{SVector{2,T}}
    angles::Vector{T}
    angles_fundamental::Vector{T}
end

"""
    TriangleBilliard(h; x0=0.0, y0=0.0, rot_angle=0.0)

Constructs an equilateral triangle billiard with specified h.

# Arguments
- `h::T`: The height of the triangle from the center.
- `x0::T=0.0`: The x-coordinate of the center of the triangle.
- `y0::T=0.0`: The y-coordinate of the center of the triangle.
- `rot_angle::T=0.0`: The rotation angle of the billiard table.

# Returns
- An instance of the `EquilateralTriangleBilliard` struct.
"""
function EquilateralTriangleBilliard(h::T; x0::T=0.0, y0::T=0.0, rot_angle::T=0.0) where {T<:Real}
    fundamental_boundary, corners = make_fundamental_triangle(h; x0=x0, y0=y0, rot_angle=rot_angle)
    full_boundary, _ = make_full_triangle(h; x0=x0, y0=y0, rot_angle=rot_angle)

    side_length = sum([line.length for line in fundamental_boundary if line isa LineSegment]) # from the real one extract the length of the side
    # Calculate area and lengths
    area = (sqrt(3) / 4) * side_length^2
    area_fundamental = area / 3  # Fundamental domain is 1/3 of the full triangle
    length = 3 * side_length
    length_fundamental = symmetry_accounted_fundamental_boundary_length(fundamental_boundary)

    # Internal angles
    angles = [π / 3, π / 3, π / 3]  # 60 degrees each
    angles_fundamental = [2*π / 3, π / 6, π / 6] 

    return EquilateralTriangleBilliard(
        fundamental_boundary,
        full_boundary,
        length,
        length_fundamental,
        area,
        area_fundamental,
        side_length,
        corners,
        angles,
        angles_fundamental
    )
end

"""
    make_equilateral_triangle_and_basis(h; x0=0.0, y0=0.0, rot_angle=0.0) 
    :: Tuple{EquilateralTriangleBilliard, CornerAdaptedFourierBessel} where {T<:Real}

Constructs an equilateral triangle billiard and a symmetry-adapted CornerAdaptedFourierBessel basis.

# Arguments
- `h`: The height of the triangle from the center.
- `x0`: x-coordinate of the center (default = 0.0).
- `y0`: y-coordinate of the center (default = 0.0).
- `rot_angle`: Rotation angle of the coordinate system (default = 0.0).

# Returns
- A tuple with the triangle billiard and the basis of real plane waves.
"""
function make_equilateral_triangle_and_basis(h::T; x0::T=0.0, y0::T=0.0, rot_angle::T=0.0) :: Tuple{EquilateralTriangleBilliard, CornerAdaptedFourierBessel} where {T<:Real}
    triangle = EquilateralTriangleBilliard(h; x0=x0, y0=y0, rot_angle=rot_angle)
    symmetry = Vector{Any}([Rotation(3, 3)])  # C3 rotational symmetry
    basis = CornerAdaptedFourierBessel(10, 2*pi/3, SVector(zero(T), zero(T)), 0.0, symmetry) # just the origin, rotation angle and symmetry for correct rotations
    return triangle, basis
end