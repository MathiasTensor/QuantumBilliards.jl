"""
    circle_eq(R::T, arc_angle::T, shift_angle::T, center::SVector{2,T}, t::T) :: SVector{2,T} where T<:Real

Compute a point on a circular arc parameterized by `t ∈ [0, 1]`.

# Arguments
- `R::T`: Radius of the circle.
- `arc_angle::T`: Total angular span of the segment (in radians).
- `shift_angle::T`: Starting angle offset (in radians).
- `center::SVector{2,T}`: Center coordinates `(x, y)` of the circle.
- `t::T`: Parameter in `[0,1]` indicating position along the arc.

# Returns
- `SVector{2,T}`: Cartesian coordinates `(x, y)` on the arc corresponding to parameter `t`.
"""
function circle_eq(R::T, arc_angle::T, shift_angle::T, center::SVector{2,T}, t) where T<:Real
    return SVector(R*cos(arc_angle*t+shift_angle) + center[1], R*sin(arc_angle*t+shift_angle)+center[2])
end

"""
    struct CircleSegment{T} <: AbsRealCurve where T<:Real

Represents a parameterized circular‐arc segment in a billiard boundary.

Fields:
- `cs::PolarCS{T}`: Coordinate system for translation/rotation.
- `radius::T`: Radius of the circle.
- `arc_angle::T`: Total angular span (radians).
- `shift_angle::T`: Starting angle offset (radians).
- `center::SVector{2,T}`: Center of the circle.
- `orientation::Int64`: +1 or –1 indicating interior side.
- `length::T`: Arc length, `abs(radius * arc_angle)`.
"""
struct CircleSegment{T} <: AbsRealCurve where T<:Real
    cs::PolarCS{T}
    radius::T
    arc_angle::T
    shift_angle::T
    center::SVector{2,T}
    orientation::Int64
    length::T
end

"""
    CircleSegment(R::T, arc_angle::T, shift_angle::T, x0::T, y0::T; origin::Tuple{T,T} = (zero(T), zero(T)), rot_angle::T = zero(T), orientation::Int64 = 1) :: CircleSegment{T} where T<:Real

Construct a `CircleSegment{T}` of radius `R::T` centered at `(x0, y0)` with angular span.

# Arguments
- `R::T`: Radius of circle.
- `arc_angle::T`: Total angular span (radians).
- `shift_angle::T`: Starting angle offset (radians).
- `x0::T, y0::T`: Center coordinates.
- `origin::Tuple{T,T}`: Translation of coordinate system before rotation (default `(0,0)`).
- `rot_angle::T`: Rotation of segment (radians) (default `0`).
- `orientation::Int64`: +1 or –1 for interior side (default `1`).

# Returns
- `CircleSegment{T}`: A new circular‐arc segment with `length = abs(R * arc_angle)`.
"""
function CircleSegment(R,arc_angle,shift_angle,x0,y0;origin=(zero(x0),zero(x0)),rot_angle=zero(x0),orientation::Int=1)
    cs=PolarCS(SVector(origin...),rot_angle)
    center=SVector(x0,y0)
    L=abs(R*arc_angle)
    return CircleSegment(cs,R,arc_angle,shift_angle,center,orientation,L)
end

"""
    struct VirtualCircleSegment{T} <: AbsVirtualCurve where T<:Real

Represents a “virtual” circular‐arc segment (for boundary conditions) in a billiard.

Fields:
- `cs::PolarCS{T}`: Coordinate system.
- `radius::T`: Circle radius.
- `arc_angle::T`: Total angular span (radians).
- `shift_angle::T`: Starting angle offset (radians).
- `center::SVector{2,T}`: Center of the circle.
- `orientation::Int64`: +1 or –1 for interior side.
- `length::T`: Arc length = `abs(radius * arc_angle)`.
- `symmetry_type::Symbol`: Boundary condition type (`:Dirichlet` or `:Neumann`).
"""
struct VirtualCircleSegment{T} <: AbsVirtualCurve where T<:Real
    cs::PolarCS{T}
    radius::T
    arc_angle::T
    shift_angle::T
    center::SVector{2,T}
    orientation::Int64
    length::T
    symmetry_type::Symbol
end

"""
    VirtualCircleSegment(R::T, arc_angle::T, shift_angle::T, x0::T, y0::T; symmetry_type::Symbol = :Dirichlet, origin::Tuple{T,T} = (zero(T), zero(T)), rot_angle::T = zero(T), orientation::Int64 = 1) :: VirtualCircleSegment{T} where T<:Real

Construct a `VirtualCircleSegment{T}` of radius `R::T` centered at `(x0, y0)` for boundary conditions.

# Arguments
- `R::T`: Circle radius.
- `arc_angle::T`: Total angular span (radians).
- `shift_angle::T`: Starting angular offset (radians).
- `x0::T, y0::T`: Center coordinates.
- `symmetry_type::Symbol`: `:Dirichlet` or `:Neumann` (default `:Dirichlet`).
- `origin::Tuple{T,T}`: Translation of coordinate system (default `(0,0)`).
- `rot_angle::T`: Rotation of segment (radians) (default `0`).
- `orientation::Int64`: +1 or –1 for interior side (default `1`).

# Returns
- `VirtualCircleSegment{T}`: A new virtual circular‐arc segment with `length = abs(R * arc_angle)`.
"""
function VirtualCircleSegment(R,arc_angle,shift_angle,x0,y0;symmetry_type=:Dirichlet,origin=(zero(x0),zero(x0)),rot_angle=zero(x0),orientation=1)
    cs=PolarCS(SVector(origin...),rot_angle)
    center=SVector(x0,y0)
    L=abs(R*arc_angle)
    return VirtualCircleSegment(cs,R,arc_angle,shift_angle,center,orientation,L,symmetry_type)
end

# Union type alias for convenience
CircleSegments{T}=Union{CircleSegment{T},VirtualCircleSegment{T}} where T<:Real

"""
    curve(circle::L, ts::AbstractVector{T}) :: Vector{SVector{2,T}} where {T<:Real, L<:CircleSegments{T}}

Compute points along the circular segment for each parameter `t ∈ ts`.

# Arguments
- `circle::L`: A `CircleSegment{T}` or `VirtualCircleSegment{T}`.
- `ts::AbstractVector{T}`: Vector of parameters ∈ `[0,1]`.

# Returns
- `Vector{SVector{2,T}}`: Array of `(x, y)` coordinates on the arc for each `t`.
"""
function curve(circle::L,ts::AbstractArray{T,1}) where {T<:Real,L<:CircleSegments{T}}
    let affine_map=circle.cs.affine_map,R=circle.radius,c=circle.center,a=circle.arc_angle,s=circle.shift_angle 
        return collect(affine_map(circle_eq(R,a,s,c,t)) for t in ts)
    end
end

"""
    curve(circle::L, t::T) :: SVector{2,T} where {T<:Real, L<:CircleSegments{T}}

Compute a single point on the circular segment at parameter `t ∈ [0,1]`.

# Arguments
- `circle::L`: A `CircleSegment{T}` or `VirtualCircleSegment{T}`.
- `t::T`: Parameter ∈ `[0,1]`.

# Returns
- `SVector{2,T}`: Cartesian coordinates `(x,y)` at that parameter.
"""
function curve(circle::L,t) where {T<:Real,L<:CircleSegments{T}}
    let affine_map=circle.cs.affine_map,R=circle.radius,c=circle.center,a=circle.arc_angle,s=circle.shift_angle 
        return affine_map(circle_eq(R,a,s,c,t))
    end
end

"""
    arc_length(circle::L, ts::AbstractVector{T}) :: Vector{T} where {T<:Real, L<:CircleSegments{T}}

Compute arc‐length coordinates for parameters `ts`.

# Arguments
- `circle::L`: A `CircleSegment{T}` or `VirtualCircleSegment{T}`.
- `ts::AbstractVector{T}`: Vector of parameters ∈ `[0,1]`.

# Returns
- `Vector{T}`: Lengths along the arc: `circle.length * t` for each `t ∈ ts`.
"""
function arc_length(circle::L,ts::AbstractArray{T,1}) where {T<:Real,L<:CircleSegments{T}}
    s::Vector{T}=circle.length.*ts 
    return s
end

"""
    tangent(circle::L, ts::AbstractVector{T}) :: Vector{SVector{2,T}} where {T<:Real, L<:CircleSegments{T}}

Compute tangent vectors along the circular segment at each `t ∈ ts`.

# Arguments
- `circle::L`: A `CircleSegment{T}` or `VirtualCircleSegment{T}`.
- `ts::AbstractVector{T}`: Vector of parameters ∈ `[0,1]`.

# Returns
- `Vector{SVector{2,T}}`: Tangent vectors (derivatives) at each `t`.
"""
function tangent(circle::L,ts::AbstractArray{T,1}) where {T<:Real,L<:CircleSegments{T}}
    let affine_map=circle.cs.affine_map,R=circle.radius,c=circle.center,a=circle.arc_angle,s=circle.shift_angle 
        orient=circle.orientation
        r(t)=affine_map(circle_eq(R,a,s,c,t))
        #ForwardDiff.derivative(r, t)
        return collect(orient*ForwardDiff.derivative(r,t) for t in ts)
    end
end

"""
    circle_domain(R::T, center::SVector{2,T}, x::T, y::T) :: T where T<:Real

Compute signed distance from a full circle of radius `R` centered at `center` to point `(x,y)`.

# Arguments
- `R::T`: Radius of the circle.
- `center::SVector{2,T}`: Center coordinates `(x_center, y_center)`.
- `x::T, y::T`: Coordinates of evaluation point.

# Returns
- `T`: Signed‐distance = `√((x-center[1])^2 + (y-center[2])^2) - R`. Negative if inside circle, positive if outside.
"""
circle_domain(R,center,x,y)=@. hypot(y-center[2],x-center[1])-R

"""
    circle_segment_domain(pt0::SVector{2,T}, pt1::SVector{2,T}, R::T, center::SVector{2,T}, 
                          orient::Int64, x::T, y::T) :: T where T<:Real

Compute the signed “domain” value for a point `(x,y)` relative to the circular segment closed by the chord from `pt0` to `pt1`.

# Arguments
- `pt0::SVector{2,T}`: First endpoint of chord.
- `pt1::SVector{2,T}`: Second endpoint of chord.
- `R::T`: Radius of circle.
- `center::SVector{2,T}`: Center `(x_center, y_center)`.
- `orient::Int64`: +1 or –1 indicating interior side.
- `x::T, y::T`: Coordinates of test point.

# Returns
- `T`: Negative if inside the circular segment, positive if outside. Chooses chord‐line distance if point lies “outside” chord, otherwise circle distance.
"""
function circle_segment_domain(pt0::SVector{2,T},pt1::SVector{2,T},R::T,center::SVector{2,T},orient::Int,x,y) where {T<:Real}
    let cd=orient*circle_domain(R,center,x,y)
        ld=orient*line_domain(pt0[1],pt0[2],pt1[1],pt1[2],x,y)
        if orient>0
            return ld<zero(ld) ? ld : cd
        else
            return ld>zero(ld) ? ld : cd
        end
    end
end

"""
    domain(circle::L, pts::AbstractVector{SVector{2,T}}) :: Vector{T} where {T<:Real, L<:CircleSegments{T}}

Compute signed‐distance values for a collection of points relative to the circular segment.

# Arguments
- `circle::L`: A `CircleSegment{T}` or `VirtualCircleSegment{T}`.
- `pts::AbstractVector{SVector{2,T}}`: List of points `(x,y)` to test.

# Returns
- `Vector{T}`: Signed distances: negative for inside the segment, positive for outside.
"""
function domain(circle::L,pts::AbstractArray) where {T<:Real,L<:CircleSegments{T}}
    let pt0=curve(circle,zero(T)),pt1=curve(circle,one(T)),R=circle.radius,orient=circle.orientation
        center=circle.cs.affine_map(circle.center) #move center to correct position
        return collect(circle_segment_domain(pt0,pt1,R,center,orient,pt[1],pt[2]) for pt in pts)
    end
end

"""
    is_inside(circle::L, pts::AbstractVector{SVector{2,T}}) :: Vector{Bool} where {T<:Real, L<:CircleSegments{T}}

Determine if each point in `pts` lies inside the circular segment.

# Arguments
- `circle::L`: A `CircleSegment{T}` or `VirtualCircleSegment{T}`.
- `pts::AbstractVector{SVector{2,T}}`: List of points to test.

# Returns
- `Vector{Bool}`: `true` where `domain(circle, pt) < 0` (inside), otherwise `false`.
"""
function is_inside(circle::L,pts::AbstractArray) where {T<:Real,L<:CircleSegments{T}}
    let pt0=curve(circle,zero(T)),pt1=curve(circle,one(T)),R=circle.radius,orient=circle.orientation
        center=circle.cs.affine_map(circle.center) #move center to correct position
        return collect(circle_segment_domain(pt0,pt1,R,center,orient,pt[1],pt[2]) < zero(eltype(pt0)) for pt in pts)
    end
end
