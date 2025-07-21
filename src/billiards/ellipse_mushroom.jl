"""
    make_half_ellipse_mushroom(stem_width::T, stem_height::T, cap_radius::T; x0=zero(T), y0=zero(T), rot_angle=zero(T)) where {T<:Real}

Constructs a half ellipse mushroom billiard with a rectangular stem and an elliptic cap. For all solutions we should have a virtual line segment on the x reflection symmetry but due to the fact that it is translated a bit this is not possible in the current framework that uses LinearMap for symmetries

# Arguments
- `stem_width::T`: The width of the stem.
- `stem_height::T`: The height of the stem.
- `ellipse_cap_height::T`: The height of the quarter ellipse cap.
- `ellipse_cap_width::T`: The width of the quarter ellipse cap.
- `x0::T=zero(T)`: The x-coordinate of the origin (center of the semicircle cap).
- `y0::T=zero(T)`: The y-coordinate of the origin (center of the semicircle cap).
- `rot_angle::T=zero(T)`: The rotation angle of the billiard table.
- `triangle_stem::Bool=false`: Whether to use a triangle stem instead of a rectangle. This is needed to remove bouncing ball modes.

# Returns
- A tuple containing:
  - `boundary::Vector{Union{LineSegment,PolarSegment}}`: The boundary segments of the half mushroom.
  - `corners::Vector{SVector{2,T}}`: The corner points of the stem.
"""
function make_half_ellipse_mushroom(stem_width::T,stem_height::T,ellipse_cap_height::T,ellipse_cap_width::T;x0=zero(T),y0=zero(T),rot_angle=zero(T),triangle_stem::Bool=false) where {T<:Real}
    stem_width=stem_width/2
    origin=SVector(x0,y0)
    r_func=t->SVector(ellipse_cap_width*cos(0.5*pi*t+pi/2),ellipse_cap_height*sin(0.5*pi*t+pi/2))
    cap_segment=PolarSegment(r_func;origin=origin,rot_angle=rot_angle)
    # Define the stem: a rectangle with width `stem_width` and height `stem_height` or a triangle with base `stem_width` and height `stem_height`
    stem_top_right_corner=SVector(stem_width,ellipse_cap_height)
    stem_bottom_right_corner=SVector(stem_width,-stem_height)
    stem_bottom_left_corner=SVector(zero(T),-stem_height)
    stem_top_left_corner=SVector(zero(T),zero(T))
    # Line segments for the stem
    if triangle_stem
        stem_right_side=LineSegment(stem_bottom_right_corner,stem_top_right_corner;origin=origin,rot_angle=rot_angle)
        stem_left_side=VirtualLineSegment(stem_top_left_corner,stem_bottom_right_corner;origin=origin,rot_angle=rot_angle)
        cap_stem_connector=VirtualLineSegment(SVector(-(ellipse_cap_width-stem_width),zero(T)),stem_top_left_corner;origin=origin,rot_angle=rot_angle)
        boundary=Union{LineSegment,PolarSegment,VirtualLineSegment}[stem_left_side,stem_right_side,cap_segment,cap_stem_connector]
        corners=[stem_top_right_corner,stem_bottom_right_corner,stem_bottom_left_corner,stem_top_left_corner]
    else
        stem_right_side=LineSegment(stem_bottom_right_corner,stem_top_right_corner;origin=origin,rot_angle=rot_angle)
        stem_bottom_side=LineSegment(stem_bottom_left_corner,stem_bottom_right_corner;origin=origin,rot_angle=rot_angle)
        stem_left_side=VirtualLineSegment(stem_top_left_corner,stem_bottom_left_corner;origin=origin,rot_angle=rot_angle)
        cap_stem_connector=VirtualLineSegment(SVector(-(ellipse_cap_width-stem_width),zero(T)),stem_top_left_corner;origin=origin,rot_angle=rot_angle)
        boundary=Union{LineSegment,PolarSegment,VirtualLineSegment}[stem_bottom_side,stem_right_side,cap_segment,cap_stem_connector,stem_left_side]
        corners=[stem_top_right_corner,stem_bottom_right_corner,stem_bottom_left_corner,stem_top_left_corner]
    end
    return boundary,corners
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
function make_full_ellipse_mushroom(stem_width::T,stem_height::T,ellipse_cap_height::T,ellipse_cap_width::T;x0=zero(T),y0=zero(T),rot_angle=zero(T),triangle_stem::Bool=false) where {T<:Real}
    origin=SVector(x0+stem_width/2,y0)
    r_func=t->SVector(ellipse_cap_width*cos(pi*t),ellipse_cap_height*sin(pi*t))
    cap_segment=PolarSegment(r_func;origin=origin,rot_angle=rot_angle)
    stem_top_right_corner=SVector(stem_width/2,zero(T))
    stem_bottom_right_corner=SVector(stem_width/2,-stem_height)
    stem_bottom_left_corner=SVector(-stem_width/2,-stem_height)
    stem_top_left_corner=SVector(-stem_width/2,zero(T))
    if triangle_stem
        stem_bottom_middle_corner=SVector(zero(T),-stem_height)
        stem_right_side=LineSegment(stem_bottom_middle_corner,stem_top_right_corner;origin=origin,rot_angle=rot_angle)
        stem_left_side=VirtualLineSegment(stem_top_left_corner,stem_bottom_middle_corner;origin=origin,rot_angle=rot_angle)
        cap_connector_right=LineSegment(stem_top_right_corner,SVector(ellipse_cap_width,zero(T));origin=origin,rot_angle=rot_angle)
        cap_connector_left=VirtualLineSegment(SVector(-ellipse_cap_width,zero(T)),stem_top_left_corner;origin=origin,rot_angle=rot_angle)
        boundary=Union{LineSegment,PolarSegment,VirtualLineSegment}[stem_left_side,stem_right_side,cap_connector_right,cap_segment,cap_connector_left]
        corners=[stem_top_right_corner,stem_bottom_right_corner,stem_bottom_left_corner,stem_top_left_corner]
    else # Line segments for the rectangle stem
        stem_right_side=LineSegment(stem_bottom_right_corner,stem_top_right_corner;origin=origin,rot_angle=rot_angle)
        stem_bottom_side=LineSegment(stem_bottom_left_corner,stem_bottom_right_corner;origin=origin,rot_angle=rot_angle)
        stem_left_side=VirtualLineSegment(stem_top_left_corner,stem_bottom_left_corner;origin=origin,rot_angle=rot_angle)
        cap_connector_right=LineSegment(stem_top_right_corner,SVector(ellipse_cap_width,zero(T));origin=origin,rot_angle=rot_angle)
        cap_connector_left=VirtualLineSegment(SVector(-ellipse_cap_width,zero(T)),stem_top_left_corner;origin=origin,rot_angle=rot_angle)
        boundary=Union{LineSegment,PolarSegment,VirtualLineSegment}[stem_bottom_side,stem_right_side,cap_connector_right,cap_segment,cap_connector_left,stem_left_side]
        corners=[stem_top_right_corner,stem_bottom_right_corner,stem_bottom_left_corner,stem_top_left_corner]
    end
    return boundary,corners
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
function make_half_full_boundary_ellipse_mushroom(stem_width::T,stem_height::T,ellipse_cap_height::T,ellipse_cap_width::T;x0=zero(T),y0=zero(T),rot_angle=zero(T),triangle_stem::Bool=false) where {T<:Real}
    stem_width=stem_width/2
    origin=SVector(x0,y0)
    r_func=t->SVector(ellipse_cap_width*cos(0.5*pi*t+pi/2),ellipse_cap_height*sin(0.5*pi*t+pi/2))
    cap_segment=PolarSegment(r_func;origin=origin,rot_angle=rot_angle)
    # Define the stem: a rectangle with width `stem_width` and height `stem_height` or a triangle with base `stem_width` and height `stem_height`
    stem_top_right_corner=SVector(stem_width,ellipse_cap_height)
    stem_bottom_right_corner=SVector(stem_width,-stem_height)
    stem_bottom_left_corner=SVector(zero(T),-stem_height)
    stem_top_left_corner=SVector(zero(T),zero(T))
    # Line segments for the stem
    if triangle_stem
        stem_right_side=LineSegment(stem_bottom_right_corner,stem_top_right_corner;origin=origin,rot_angle=rot_angle)
        stem_left_side=LineSegment(stem_top_left_corner,stem_bottom_right_corner;origin=origin,rot_angle=rot_angle)
        cap_stem_connector=LineSegment(SVector(-(ellipse_cap_width-stem_width),zero(T)),stem_top_left_corner;origin=origin,rot_angle=rot_angle)
        boundary=Union{LineSegment,PolarSegment}[stem_left_side,stem_right_side,cap_segment,cap_stem_connector]
        corners=[stem_top_right_corner,stem_bottom_right_corner,stem_bottom_left_corner,stem_top_left_corner]
    else
        stem_right_side=LineSegment(stem_bottom_right_corner,stem_top_right_corner;origin=origin,rot_angle=rot_angle)
        stem_bottom_side=LineSegment(stem_bottom_left_corner,stem_bottom_right_corner;origin=origin,rot_angle=rot_angle)
        stem_left_side=LineSegment(stem_top_left_corner,stem_bottom_left_corner;origin=origin,rot_angle=rot_angle)
        cap_stem_connector=LineSegment(SVector(-(ellipse_cap_width-stem_width),zero(T)),stem_top_left_corner;origin=origin,rot_angle=rot_angle)
        boundary=Union{LineSegment,PolarSegment}[stem_bottom_side,stem_right_side,cap_segment,cap_stem_connector,stem_left_side]
        corners=[stem_top_right_corner,stem_bottom_right_corner,stem_bottom_left_corner,stem_top_left_corner]
    end
    return boundary,corners
end

"""
    struct Mushroom{T} <: AbsBilliard where {T<:Real}

Defines a Mushroom billiard with a rectangular stem and a circular cap.

# Fields
- `fundamental_boundary::Vector{Union{LineSegment, PolarSegment}}`: The boundary segments of the half mushroom.
- `full_boundary::Vector{Union{LineSegment, PolarSegment}}`: The boundary segments of the full mushroom.
- `desymmetrized_full_boundary::Vector{Union{LineSegment, PolarSegment}}`: The real half mushroom boundary that is used to construct the boundary function.
- `length::T`: The total length of the boundary.
- `area::T`: The total area of the mushroom.
- `stem_width::T`: The width of the stem.
- `stem_height::T`: The height of the stem.
- `corners::Vector{SVector{2,T}}`: The corner points of the stem.
- `angles::Vector`: The angles of the boundary segments in radians.
- `angles_fundamental::Vector`: The angles of the fundamental boundary segments in radians.
- `x_axis::T`: The actual "axis" of reflection, shifted from the origin.
- `shift_s::T`: For shifting the arclengths for the husimi function construction.
"""
struct EllipseMushroom{T} <: AbsBilliard where {T<:Real}
    fundamental_boundary::Vector{Union{LineSegment,PolarSegment,VirtualLineSegment}}
    full_boundary::Vector{Union{LineSegment,PolarSegment,VirtualLineSegment}}
    desymmetrized_full_boundary::Vector{Union{LineSegment,PolarSegment}}
    length::T
    length_fundamental::T
    area::T
    area_fundamental::T
    stem_width::T
    stem_height::T
    cap_radius::T
    ellipse_cap_height::T
    ellipse_cap_width::T
    corners::Vector{SVector{2,T}}
    angles::Vector
    angles_fundamental::Vector
    x_axis::T # For correct reflection. This is the actual "axis" of reflection
    shift_s::T # for shifting the arclenthgs for the husimi function.
end

function make_ellipse_mushroom_and_basis(stem_width::T,stem_height::T,ellipse_cap_height::T,ellipse_cap_width::T;x0=zero(T),y0=zero(T),rot_angle=zero(T),triangle_stem::Bool=false) :: Tuple{EllipseMushroom,CornerAdaptedFourierBessel} where {T<:Real}
    x_axis=stem_width/2
    ellipse_mushroom=EllipseMushroom(stem_width,stem_height,ellipse_cap_height,ellipse_cap_width;x0=x0,y0=y0,rot_angle=rot_angle,x_axis_reflection=x_axis,triangle_stem=triangle_stem)
    symmetry=Vector{Any}([XReflection(-1)])
    α=ifelse(triangle_stem,3*pi/2-atan((stem_width/2)/stem_height),3*pi/2)
    basis=CornerAdaptedFourierBessel(10,α,SVector(zero(T),zero(T)),Float64(pi),symmetry;rotation_angle_discontinuity=Float64(3*pi/4))
    return ellipse_mushroom,basis
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
function EllipseMushroom(stem_width::T,stem_height::T,ellipse_cap_height::T,ellipse_cap_width::T;x0=zero(T),y0=zero(T),rot_angle=zero(T),triangle_stem::Bool=false,x_axis_reflection=zero(T)) :: Mushroom where {T<:Real}
    fundamental_boundary,corners=make_half_ellipse_mushroom(stem_width,stem_height,ellipse_cap_height,ellipse_cap_width;x0=x0,y0=y0,rot_angle=rot_angle,triangle_stem=triangle_stem)
    full_boundary,_=make_full_ellipse_mushroom(stem_width,stem_height,ellipse_cap_height,ellipse_cap_width;x0=x0,y0=y0,rot_angle=rot_angle,triangle_stem=triangle_stem)
    area=compute_area(PolarSegment(t->SVector(ellipse_cap_width*cos(2*pi*t),ellipse_cap_height*sin(2*pi*t))))
    area_fundamental=area/2
    length=sum([crv.length for crv in full_boundary])
    desymmetrized_full_boundary,_=make_half_full_boundary_ellipse_mushroom(stem_width,stem_height,ellipse_cap_height,ellipse_cap_width;x0=x0,y0=y0,rot_angle=rot_angle,triangle_stem=triangle_stem)
    length_fundamental=symmetry_accounted_fundamental_boundary_length(fundamental_boundary)
    α=ifelse(triangle_stem,3*pi/2-atan((stem_width/2)/stem_height),3*pi/2)
    angles=[α,pi/2,pi/2,α]
    angles_fundamental=[α,pi/2,pi/2]
    shift_s=sum(crv.length for crv in desymmetrized_full_boundary[1:3]) # so that the start of the arclength will be at the bottom left corner of the full mushroom boundary
    return EllipseMushroom(fundamental_boundary,full_boundary,desymmetrized_full_boundary,length,length_fundamental,area,area_fundamental,stem_width,stem_height,ellipse_cap_height,ellipse_cap_width,corners,angles,angles_fundamental,x_axis_reflection,shift_s)
end