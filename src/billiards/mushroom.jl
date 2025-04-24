"""
    make_half_mushroom(stem_width::T, stem_height::T, cap_radius::T; x0=zero(T), y0=zero(T), rot_angle=zero(T)) where {T<:Real}

Constructs a half mushroom billiard with a rectangular stem and a circular cap. For all solutions we should have a virtual line segment on the x reflection symmetry but due to the fact that it is translated a bit this is not possible in the current framework that uses LinearMap for symmetries

# Arguments
- `stem_width::T`: The width of the stem.
- `stem_height::T`: The height of the stem.
- `cap_radius::T`: The radius of the circular cap.
- `x0::T=zero(T)`: The x-coordinate of the origin (center of the semicircle cap).
- `y0::T=zero(T)`: The y-coordinate of the origin (center of the semicircle cap).
- `rot_angle::T=zero(T)`: The rotation angle of the billiard table.

# Returns
- A tuple containing:
  - `boundary::Vector{Union{LineSegment, CircleSegment}}`: The boundary segments of the half mushroom.
  - `corners::Vector{SVector{2,T}}`: The corner points of the stem.
"""
function make_half_mushroom(stem_width::T, stem_height::T, cap_radius::T; x0=zero(T), y0=zero(T), rot_angle=zero(T)) where {T<:Real}
    stem_width = stem_width/2 # Because desymmetrized we only take half of the full width
    origin = SVector(x0, y0)

    # Define the cap: a quarter circle with radius `cap_radius` centered at (-stem_width + cap_radius, 0)
    cap_center_point = SVector(stem_width, zero(T))
    cap_segment = CircleSegment(cap_radius, 0.5*pi, 0.5*pi, cap_center_point[1], cap_center_point[2]; origin=origin, rot_angle=rot_angle)
    
    # Define the stem: a rectangle with width `stem_width` and height `stem_height`
    stem_top_right_corner = SVector(stem_width, cap_radius)
    stem_bottom_right_corner = SVector(stem_width, -stem_height)
    stem_bottom_left_corner = SVector(zero(T), -stem_height)
    stem_top_left_corner = SVector(zero(T), zero(T))
    
    # Line segments for the stem
    stem_right_side = LineSegment(stem_bottom_right_corner, stem_top_right_corner; origin=origin, rot_angle=rot_angle)
    stem_bottom_side = LineSegment(stem_bottom_left_corner, stem_bottom_right_corner; origin=origin, rot_angle=rot_angle)
    stem_left_side = VirtualLineSegment(stem_top_left_corner, stem_bottom_left_corner; origin=origin, rot_angle=rot_angle)
    cap_stem_connector = VirtualLineSegment(SVector(-(cap_radius - stem_width), zero(T)), stem_top_left_corner; origin=origin, rot_angle=rot_angle)
    
    # Starts with AbsRealCurve Counterclockwise
    boundary = Union{LineSegment, CircleSegment, VirtualLineSegment}[stem_bottom_side, stem_right_side, cap_segment, cap_stem_connector, stem_left_side]

    corners = [stem_top_right_corner, stem_bottom_right_corner, stem_bottom_left_corner, stem_top_left_corner]
    return boundary, corners
end

"""
    make_full_mushroom(stem_width::T, stem_height::T, cap_radius::T; x0=zero(T), y0=zero(T), rot_angle=zero(T)) where {T<:Real}

Constructs a full mushroom billiard with a rectangular stem and a circular cap.

# Arguments
- `stem_width::T`: The width of the stem.
- `stem_height::T`: The height of the stem.
- `cap_radius::T`: The radius of the circular cap.
- `x0::T=zero(T)`: The x-coordinate of the origin (center of the semicircle cap).
- `y0::T=zero(T)`: The y-coordinate of the origin (center of the semicircle cap).
- `rot_angle::T=zero(T)`: The rotation angle of the billiard table.

# Returns
- A tuple containing:
  - `boundary::Vector{Union{LineSegment{T}, CircleSegment{T}}}`: The boundary segments of the full mushroom.
  - `corners::Vector{SVector{2,T}}`: The corner points of the stem.
"""
function make_full_mushroom(stem_width::T, stem_height::T, cap_radius::T; x0=zero(T), y0=zero(T), rot_angle=zero(T)) where {T<:Real}
    origin = SVector(x0 + stem_width/2, y0)
    # Define the cap: a half circle with radius `cap_radius` centered at (0, 0)
    cap_segment = CircleSegment(cap_radius, Float64(pi), Float64(2*pi), zero(T), zero(T); origin=origin, rot_angle=rot_angle) # corrected from before wrong -pi, pi for shift angle and arc
    
    # Define the stem: a rectangle with width `stem_width` and height `stem_height`
    stem_top_right_corner = SVector(stem_width/2, zero(T))
    stem_bottom_right_corner = SVector(stem_width/2, -stem_height)
    stem_bottom_left_corner = SVector(-stem_width/2, -stem_height)
    stem_top_left_corner = SVector(-stem_width/2, zero(T))
    
    # Line segments for the stem
    stem_right_side = LineSegment(stem_bottom_right_corner, stem_top_right_corner; origin=origin, rot_angle=rot_angle)
    stem_bottom_side = LineSegment(stem_bottom_left_corner, stem_bottom_right_corner; origin=origin, rot_angle=rot_angle)
    stem_left_side = LineSegment(stem_top_left_corner, stem_bottom_left_corner; origin=origin, rot_angle=rot_angle)
    cap_connector_right = LineSegment(stem_top_right_corner, SVector(cap_radius, zero(T)); origin=origin, rot_angle=rot_angle) # wrong: cap_connector_right = LineSegment(SVector(cap_radius, zero(T)), stem_top_right_corner; origin=origin, rot_angle=rot_angle)
    cap_connector_left = LineSegment(SVector(-cap_radius, zero(T)), stem_top_left_corner; origin=origin, rot_angle=rot_angle)

    # Starts with AbsRealCurve counterclockwise
    boundary = Union{LineSegment, CircleSegment, VirtualLineSegment}[stem_bottom_side, stem_right_side, cap_connector_right, cap_segment, cap_connector_left, stem_left_side]
    corners = [stem_top_right_corner, stem_bottom_right_corner, stem_bottom_left_corner, stem_top_left_corner]
    return boundary, corners
end

"""
    make_half_full_boundary_mushroom(stem_width::T, stem_height::T, cap_radius::T; x0=zero(T), y0=zero(T), rot_angle=zero(T)) where {T<:Real}

Constructs a half mushroom billiard with a rectangular stem and a circular cap. The virtual line segments become real line segments and the symmetry axis is removed. This mimics the true desymmetrized mushroom. This one is used for construction of the boundary function.

# Arguments
- `stem_width::T`: The width of the stem.
- `stem_height::T`: The height of the stem.
- `cap_radius::T`: The radius of the circular cap.
- `x0::T=zero(T)`: The x-coordinate of the origin (center of the semicircle cap).
- `y0::T=zero(T)`: The y-coordinate of the origin (center of the semicircle cap).
- `rot_angle::T=zero(T)`: The rotation angle of the billiard table.

# Returns
- A tuple containing:
  - `boundary::Vector{Union{LineSegment, CircleSegment}}`: The boundary segments of the half mushroom with no symmetry axis and only real segments.
  - `corners::Vector{SVector{2,T}}`: The corners of this geometry.
"""
function make_half_full_boundary_mushroom(stem_width::T, stem_height::T, cap_radius::T; x0=zero(T), y0=zero(T), rot_angle=zero(T)) where {T<:Real}
    stem_width = stem_width/2 # Because desymmetrized we only take half of the full width
    origin = SVector(x0, y0)

    # Define the cap: a quarter circle with radius `cap_radius` centered at (-stem_width + cap_radius, 0)
    cap_center_point = SVector(stem_width, zero(T))
    cap_segment = CircleSegment(cap_radius, 0.5*pi, 0.5*pi, cap_center_point[1], cap_center_point[2]; origin=origin, rot_angle=rot_angle)
    
    stem_bottom_right_corner = SVector(stem_width, -stem_height)
    stem_bottom_left_corner = SVector(zero(T), -stem_height)
    stem_top_left_corner = SVector(zero(T), zero(T))
    
    # Line segments for the stem that are not symmetry axes
    stem_bottom_side = LineSegment(stem_bottom_left_corner, stem_bottom_right_corner; origin=origin, rot_angle=rot_angle)
    stem_left_side = LineSegment(stem_top_left_corner, stem_bottom_left_corner; origin=origin, rot_angle=rot_angle)
    cap_stem_connector = LineSegment(SVector(-(cap_radius - stem_width), zero(T)), stem_top_left_corner; origin=origin, rot_angle=rot_angle)
    # Anticlockwise ordering of the real half mushroom boundary
    boundary = Union{LineSegment, CircleSegment}[cap_segment, cap_stem_connector, stem_left_side, stem_bottom_side]

    corners = [stem_bottom_left_corner, stem_top_left_corner] # only 2 corners
    return boundary, corners
end

"""
    struct Mushroom{T} <: AbsBilliard where {T<:Real}

Defines a Mushroom billiard with a rectangular stem and a circular cap.

# Fields
- `fundamental_boundary::Vector{Union{LineSegment, CircleSegment}}`: The boundary segments of the half mushroom.
- `full_boundary::Vector{Union{LineSegment, CircleSegment}}`: The boundary segments of the full mushroom.
- `desymmetrized_full_boundary::Vector{Union{LineSegment, CircleSegment}}`: The real half mushroom boundary that is used to construct the boundary function.
- `length::T`: The total length of the boundary.
- `area::T`: The total area of the mushroom.
- `stem_width::T`: The width of the stem.
- `stem_height::T`: The height of the stem.
- `cap_radius::T`: The radius of the circular cap.
- `corners::Vector{SVector{2,T}}`: The corner points of the stem.
- `angles::Vector`: The angles of the boundary segments in radians.
- `angles_fundamental::Vector`: The angles of the fundamental boundary segments in radians.
- `x_axis::T`: The actual "axis" of reflection, shifted from the origin.
- `shift_s::T`: For shifting the arclengths for the husimi function construction.
"""
struct Mushroom{T} <: AbsBilliard where {T<:Real}
    fundamental_boundary::Vector{Union{LineSegment, CircleSegment, VirtualLineSegment}}
    full_boundary::Vector{Union{LineSegment, CircleSegment, VirtualLineSegment}}
    desymmetrized_full_boundary::Vector{Union{LineSegment, CircleSegment}}
    length::T
    length_fundamental::T
    area::T
    area_fundamental::T
    stem_width::T
    stem_height::T
    cap_radius::T
    corners::Vector{SVector{2,T}}
    angles::Vector
    angles_fundamental::Vector
    x_axis::T # For correct reflection. This is the actual "axis" of reflection
    shift_s::T # for shifting the arclenthgs for the husimi function.
end

function make_mushroom_and_basis(stem_width::T, stem_height::T, cap_radius::T; x0=zero(T), y0=zero(T), rot_angle=zero(T)) :: Tuple{Mushroom, CornerAdaptedFourierBessel} where {T<:Real}
    x_axis = stem_width/2
    mushroom = Mushroom(stem_width, stem_height, cap_radius; x0=x0, y0=y0, rot_angle=rot_angle, x_axis_reflection=x_axis)
    symmetry = Vector{Any}([XReflection(-1)])
    basis = CornerAdaptedFourierBessel(10, 3*pi/2, SVector(zero(T), zero(T)), Float64(pi), symmetry; rotation_angle_discontinuity=Float64(3*pi/4))
    return mushroom, basis
end

"""
    Mushroom(stem_width::T, stem_height::T, cap_radius::T; x0=zero(T), y0=zero(T), rot_angle=zero(T)) :: Mushroom where {T<:Real}

Constructs a Mushroom billiard with a rectangular stem and a circular cap.

# Arguments
- `stem_width::T`: The width of the stem.
- `stem_height::T`: The height of the stem.
- `cap_radius::T`: The radius of the circular cap.
- `x0::T=zero(T)`: The x-coordinate of the origin (center of the semicircle cap).
- `y0::T=zero(T)`: The y-coordinate of the origin (center of the semicircle cap).

# Returns
- An instance of the `Mushroom` struct.
"""
function Mushroom(stem_width::T, stem_height::T, cap_radius::T; x0=zero(T), y0=zero(T), rot_angle=zero(T), x_axis_reflection=zero(T)) :: Mushroom where {T<:Real}
    fundamental_boundary, _ = make_half_mushroom(stem_width, stem_height, cap_radius; x0=x0, y0=y0, rot_angle=rot_angle)
    full_boundary, corners = make_full_mushroom(stem_width, stem_height, cap_radius; x0=x0, y0=y0, rot_angle=rot_angle)
    area = stem_width * stem_height + 0.5 * pi * cap_radius^2
    area_fundamental = area/2
    length = sum([crv.length for crv in full_boundary])
    desymmetrized_full_boundary, _ = make_half_full_boundary_mushroom(stem_width, stem_height, cap_radius; x0=x0, y0=y0, rot_angle=rot_angle)
    length_fundamental = symmetry_accounted_fundamental_boundary_length(fundamental_boundary)
    angles = [3*pi/2, pi/2, pi/2, 3*pi/2]
    angles_fundamental = [3*pi/2, pi/2, pi/2]
    shift_s = sum(crv.length for crv in desymmetrized_full_boundary[1:3]) # so that the start of the arclength will be at the bottom left corner of the full mushroom boundary
    return Mushroom(fundamental_boundary, full_boundary, desymmetrized_full_boundary, length, length_fundamental, area, area_fundamental, stem_width, stem_height, cap_radius, corners, angles, angles_fundamental, x_axis_reflection, shift_s)
end