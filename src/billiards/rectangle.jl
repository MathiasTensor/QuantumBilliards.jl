





"""
    make_quarter_rectangle(width, height; x0=0.0, y0=0.0, rot_angle=0.0)

Constructs a quarter of a rectangle billiard with specified width and height. The quarter is defined in the first quadrant (positive x and y axes).

# Arguments
- `width::T`: The width of the full rectangle.
- `height::T`: The height of the full rectangle.
- `x0::T=0.0`: The x-coordinate of the center of the full rectangle.
- `y0::T=0.0`: The y-coordinate of the center of the full rectangle.
- `rot_angle::T=0.0`: The rotation angle of the billiard table.

# Returns
- A tuple containing:
  - `boundary::Vector{Union{LineSegment{T}, VirtualLineSegment{T}}}`: The boundary segments of the quarter rectangle.
  - `corners::Vector{SVector{2,T}}`: The corner points of the quarter rectangle.
"""
function make_quarter_rectangle(width, height; x0=0.0, y0=0.0, rot_angle=0.0)
    type = typeof(width)
    origin = SVector(x0, y0)
    
    # Define the corners of the quarter rectangle
    half_width = width / 2
    half_height = height / 2
    bottom_left = SVector(0.0, 0.0)
    bottom_right = SVector(half_width, 0.0)
    top_right = SVector(half_width, half_height)
    top_left = SVector(0.0, half_height)
    
    # Line segments for the quarter rectangle
    right_side = LineSegment(bottom_right, top_right; origin=origin, rot_angle=rot_angle)
    top_side = LineSegment(top_right, top_left; origin=origin, rot_angle=rot_angle)
    left_side = VirtualLineSegment(top_left, bottom_left; origin=origin, rot_angle=rot_angle)
    bottom_side = VirtualLineSegment(bottom_left, bottom_right; origin=origin, rot_angle=rot_angle)
    
    # Counterclockwise order
    boundary = Union{LineSegment{type}, VirtualLineSegment{type}}[
        right_side, top_side, left_side, bottom_side
    ]
    
    corners = [bottom_left, bottom_right, top_right, top_left]
    return boundary, corners
end

"""
    make_full_rectangle(width, height; x0=0.0, y0=0.0, rot_angle=0.0)

Constructs the full rectangle billiard with specified width and height.

# Arguments
- `width::T`: The width of the rectangle.
- `height::T`: The height of the rectangle.
- `x0::T=0.0`: The x-coordinate of the center of the rectangle.
- `y0::T=0.0`: The y-coordinate of the center of the rectangle.
- `rot_angle::T=0.0`: The rotation angle of the billiard table.

# Returns
- A tuple containing:
  - `boundary::Vector{LineSegment{T}}`: The boundary segments of the full rectangle.
  - `corners::Vector{SVector{2,T}}`: The corner points of the full rectangle.
"""
function make_full_rectangle(width, height; x0=0.0, y0=0.0, rot_angle=0.0)
    type = typeof(width)
    origin = SVector(x0, y0)
    
    # Define the corners of the full rectangle
    half_width = width / 2
    half_height = height / 2
    bottom_left = SVector(-half_width, -half_height)
    bottom_right = SVector(half_width, -half_height)
    top_right = SVector(half_width, half_height)
    top_left = SVector(-half_width, half_height)
    
    # Line segments for the full rectangle
    bottom_side = LineSegment(bottom_left, bottom_right; origin=origin, rot_angle=rot_angle)
    right_side = LineSegment(bottom_right, top_right; origin=origin, rot_angle=rot_angle)
    top_side = LineSegment(top_right, top_left; origin=origin, rot_angle=rot_angle)
    left_side = LineSegment(top_left, bottom_left; origin=origin, rot_angle=rot_angle)
    
    # Counterclockwise
    boundary = Union{LineSegment{type}}[
        bottom_side, right_side, top_side, left_side
    ]
    
    corners = [bottom_left, bottom_right, top_right, top_left]
    return boundary, corners
end

"""
    struct RectangleBilliard{T} <: AbsBilliard where {T<:Real}

Defines a Rectangle billiard with specified width and height. It includes both the full and fundamental boundaries.

# Fields
- `fundamental_boundary::Vector{Union{LineSegment{T}, VirtualLineSegment{T}}}`: The boundary segments of the quarter rectangle.
- `full_boundary::Vector{LineSegment{T}}`: The boundary segments of the full rectangle.
- `length::T`: The total length of the boundary.
- `area::T`: The total area of the rectangle.
- `width::T`: The width of the rectangle.
- `height::T`: The height of the rectangle.
- `corners::Vector{SVector{2,T}}`: The corner points of the full rectangle.
"""
struct RectangleBilliard{T} <: AbsBilliard where {T<:Real}
    fundamental_boundary::Vector
    full_boundary::Vector
    length::T
    length_fundamental::T
    area::T
    area_fundamental::T
    width::T
    height::T
    corners::Vector{SVector{2,T}}
end

"""
    RectangleBilliard(width, height; x0=0.0, y0=0.0, rot_angle=0.0)

Constructs a Rectangle billiard with specified width and height.

# Arguments
- `width::T`: The width of the rectangle.
- `height::T`: The height of the rectangle.
- `x0::T=0.0`: The x-coordinate of the center of the rectangle.
- `y0::T=0.0`: The y-coordinate of the center of the rectangle.

# Returns
- An instance of the `RectangleBilliard` struct.
"""
function RectangleBilliard(width, height; x0=0.0, y0=0.0, rot_angle=0.0)
    fundamental_boundary, _ = make_quarter_rectangle(width, height; x0=x0, y0=y0, rot_angle=rot_angle)
    full_boundary, corners = make_full_rectangle(width, height; x0=x0, y0=y0, rot_angle=rot_angle)
    area = width * height
    area_fundamental = area * 0*25
    length = sum([crv.length for crv in full_boundary])
    length_fundamental = sum([crv.length for crv in fundamental_boundary])
    return RectangleBilliard(fundamental_boundary, full_boundary, length, length_fundamental, area, area_fundamental, width, height, corners)
end

"""
    make_rectangle_and_basis(width, height; x0=0.0, y0=0.0, rot_angle=0.0) 
    :: Tuple{RectangleBilliard{T}, RealPlaneWaves} where {T<:Real}

Constructs a rectangle billiard and a symmetry-adapted basis of real plane waves.

# Arguments
- `width`: Width of the rectangle.
- `height`: Height of the rectangle.
- `x0`: x-coordinate of the center (default = 0.0).
- `y0`: y-coordinate of the center (default = 0.0).
- `rot_angle`: Rotation angle of the coordinate system (default = 0.0).

# Returns
- A tuple with the rectangle billiard and the basis of real plane waves.
"""
function make_rectangle_and_basis(width::T, height::T; x0=zero(T), y0=zero(T), rot_angle=zero(T))  :: Tuple{RectangleBilliard{T}, RealPlaneWaves} where {T<:Real}
    rectangle = RectangleBilliard(width, height; x0=x0, y0=y0, rot_angle=rot_angle)
    symmetry = Vector{Any}([XYReflection(-1, -1)])
    basis = RealPlaneWaves(10, symmetry; angle_arc=pi/2.0)
    return rectangle, basis
end

