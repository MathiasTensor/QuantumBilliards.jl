
"""
    circle_eq_reversed(R::T, arc_angle::T, shift_angle::T, center::SVector{2,T}, t::T) :: SVector{2,T} where T<:Real

Compute a point on a “reversed” circular arc parameterized by `t ∈ [0,1]`, reflecting around the vertical axis.

# Arguments
- `R::T`: Radius of the circle.
- `arc_angle::T`: Total angular span (radians).
- `shift_angle::T`: Angular offset (radians).
- `center::SVector{2,T}`: Center coordinates `(x, y)`.
- `t::T`: Parameter in `[0,1]` indicating position along the arc.

# Returns
- `SVector{2,T}`: Cartesian coordinates `(x, y)` on the reversed arc at parameter `t`.
"""
function circle_eq_reversed(R::T,arc_angle::T,shift_angle::T,center::SVector{2,T},t) where T<:Real
    return SVector(-R*cos(arc_angle*t-shift_angle)+center[1],R*sin(arc_angle*t-shift_angle)+center[2])
end


"""
    struct DispersingCircleSegment{T} <: AbsRealCurve where T<:Real

Represents a dispersing (concave‐inwards) circular‐arc segment in a billiard boundary.

Fields:
- `cs::PolarCS{T}`: Coordinate system (translation + rotation).
- `radius::T`: Circle radius.
- `arc_angle::T`: Angular span of the segment (radians).
- `shift_angle::T`: Angular offset (radians).
- `center::SVector{2,T}`: Center point `(x,y)`.
- `orientation::Int64`: +1 or –1 indicating interior side.
- `length::T`: Arc length = `abs(radius * arc_angle)`.
"""
struct DispersingCircleSegment{T} <: AbsRealCurve where T<:Real
    cs::PolarCS{T}
    radius::T
    arc_angle::T
    shift_angle::T
    center::SVector{2,T}
    orientation::Int64
    length::T
end

"""
    DispersingCircleSegment(R::T, arc_angle::T, shift_angle::T, x0::T, y0::T; 
                            origin::Tuple{T,T} = (zero(T), zero(T)), 
                            rot_angle::T = zero(T), 
                            orientation::Int64 = 1) :: DispersingCircleSegment{T} where T<:Real

Construct a `DispersingCircleSegment{T}` of radius `R` centered at `(x0,y0)` with on reverse parametrized segment.

# Arguments
- `R::T`: Circle radius.
- `arc_angle::T`: Total angular span (radians).
- `shift_angle::T`: Angular offset (radians).
- `x0::T, y0::T`: Center coordinates.
- `origin::Tuple{T,T}`: Translation for coordinate system (default `(0,0)`).
- `rot_angle::T`: Rotation (radians) (default `0`).
- `orientation::Int64`: +1 or –1 for interior side (default `1`).

# Returns
- `DispersingCircleSegment{T}`: New dispersing (concave) circular‐arc segment with `length = abs(R * arc_angle)`.
"""
function DispersingCircleSegment(R::T,arc_angle::T,shift_angle::T,x0::T,y0::T;origin=(zero(x0),zero(x0)),rot_angle=zero(x0),orientation::Int=1) where {T<:Real}
    cs=PolarCS(SVector(origin...),rot_angle)
    center=SVector(x0,y0)
    L=abs(R*arc_angle)
    return DispersingCircleSegment(cs,R,arc_angle,shift_angle,center,orientation,L)
end

"""
    struct VirtualDispersingCircleSegment{T} <: AbsVirtualCurve where T<:Real

Represents a “virtual” (boundary‐condition) dispersing circular‐arc segment.

Fields:
- `cs::PolarCS{T}`: Coordinate system.
- `radius::T`: Circle radius.
- `arc_angle::T`: Angular span (radians).
- `shift_angle::T`: Angular offset (radians).
- `center::SVector{2,T}`: Center `(x,y)`.
- `orientation::Int64`: +1 or –1 interior side.
- `length::T`: Arc length = `abs(radius * arc_angle)`.
- `symmetry_type::Symbol`: Boundary condition (`:Dirichlet` or `:Neumann`).
"""
struct VirtualDispersingCircleSegment{T} <: AbsVirtualCurve where T<:Real
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
    VirtualDispersingCircleSegment(R::T, arc_angle::T, shift_angle::T, x0::T, y0::T; symmetry_type::Symbol = :Dirichlet, origin::Tuple{T,T} = (zero(T), zero(T)), rot_angle::T = zero(T), orientation::Int64 = 1) :: VirtualDispersingCircleSegment{T} where T<:Real

Construct a `VirtualDispersingCircleSegment{T}` of radius `R` centered at `(x0,y0)` for boundary conditions.

# Arguments
- `R::T`: Circle radius.
- `arc_angle::T`: Angular span (radians).
- `shift_angle::T`: Angular offset (radians).
- `x0::T, y0::T`: Center coordinates.
- `symmetry_type::Symbol`: `:Dirichlet` or `:Neumann` (default `:Dirichlet`).
- `origin::Tuple{T,T}`: Translation (default `(0,0)`).
- `rot_angle::T`: Rotation (radians) (default `0`).
- `orientation::Int64`: +1 or –1 for interior side (default `1`).

# Returns
- `VirtualDispersingCircleSegment{T}`: New virtual dispersing arc with `length = abs(R * arc_angle)`.
"""
function VirtualCircleSegment(R::T,arc_angle::T,shift_angle::T,x0::T,y0::T;symmetry_type=:Dirichlet,origin=(zero(x0),zero(x0)),rot_angle=zero(x0),orientation=1) where {T<:Real}
    cs=PolarCS(SVector(origin...),rot_angle)
    center=SVector(x0,y0)
    L=abs(R*arc_angle)
    return VirtualDispersingCircleSegment(cs,R,arc_angle,shift_angle,center,orientation,L,symmetry_type)
end

# Union alias that combined both concrete and virtual circle segments
DispersingCircleSegments{T}=Union{DispersingCircleSegment{T},VirtualDispersingCircleSegment{T}} where T<:Real

"""
    curve(circle::L, ts::AbstractVector{T}) :: Vector{SVector{2,T}} where {T<:Real, L<:DispersingCircleSegments{T}}

Compute points along the dispersing (reversed) circle segment for each `t ∈ ts`.

# Arguments
- `circle::L`: A `DispersingCircleSegment{T}` or `VirtualDispersingCircleSegment{T}`.
- `ts::AbstractVector{T}`: Vector of parameters ∈ `[0,1]`.

# Returns
- `Vector{SVector{2,T}}`: Array of `(x,y)` coordinates on the reversed arc for each `t`.
"""
function curve(circle::L,ts::AbstractArray{T,1}) where {T<:Real,L<:DispersingCircleSegments{T}}
    let affine_map=circle.cs.affine_map,R=circle.radius,c=circle.center,a=circle.arc_angle,s=circle.shift_angle 
        return collect(affine_map(circle_eq_reversed(R,a,s,c,t)) for t in ts)
    end
end

"""
    curve(circle::L, t::T) :: SVector{2,T} where {T<:Real, L<:DispersingCircleSegments{T}}

Compute a single point on the dispersing (reversed) circle segment at `t ∈ [0,1]`.

# Arguments
- `circle::L`: A `DispersingCircleSegment{T}` or `VirtualDispersingCircleSegment{T}`.
- `t::T`: Parameter ∈ `[0,1]`.

# Returns
- `SVector{2,T}`: Coordinates `(x,y)` at that parameter.
"""
function curve(circle::L,t::T) where {T<:Real,L<:DispersingCircleSegments{T}}
    let affine_map=circle.cs.affine_map,R=circle.radius,c=circle.center,a=circle.arc_angle,s=circle.shift_angle 
        return affine_map(circle_eq_reversed(R,a,s,c,t))
    end
end

"""
    arc_length(circle::L, ts::AbstractVector{T}) :: Vector{T} where {T<:Real, L<:DispersingCircleSegments{T}}

Compute arc‐length coordinates for parameters `ts` along the dispersing segment.

# Arguments
- `circle::L`: A `DispersingCircleSegment{T}` or `VirtualDispersingCircleSegment{T}`.
- `ts::AbstractVector{T}`: Vector of parameters ∈ `[0,1]`.

# Returns
- `Vector{T}`: Arc lengths = `circle.length * t` for each `t`.
"""
function arc_length(circle::L,ts::AbstractArray{T,1}) where {T<:Real,L<:DispersingCircleSegments{T}}
    s::Vector{T}=circle.length.*ts 
    return s
end

"""
    tangent(circle::L, ts::AbstractVector{T}) :: Vector{SVector{2,T}} where {T<:Real, L<:DispersingCircleSegments{T}}

Compute tangent vectors along the dispersing segment at each `t ∈ ts`.

# Arguments
- `circle::L`: A `DispersingCircleSegment{T}` or `VirtualDispersingCircleSegment{T}`.
- `ts::AbstractVector{T}`: Vector of parameters ∈ `[0,1]`.

# Returns
- `Vector{SVector{2,T}}`: Tangent vectors (derivatives) at each `t`.
"""
function tangent(circle::L,ts::AbstractArray{T,1}) where {T<:Real,L<:DispersingCircleSegments{T}}
    let affine_map=circle.cs.affine_map,R=circle.radius,c=circle.center,a=circle.arc_angle,s=circle.shift_angle 
        orient=circle.orientation
        r(t)=affine_map(circle_eq_reversed(R,a,s,c,t))
        return collect(orient*ForwardDiff.derivative(r,t) for t in ts)
    end
end

"""
    tangent(circle::L, t::T) :: SVector{2,T} where {T<:Real, L<:DispersingCircleSegments{T}}

Compute the tangent vector (first derivative) of the dispersing circle segment at a single parameter `t`.

# Arguments
- `circle::L`: A `DispersingCircleSegment{T}` or `VirtualDispersingCircleSegment{T}`.
- `t::T`: A parameter in `[0,1]`.

# Returns
- `SVector{2,T}`: The tangent vector at `t`, scaled by `circle.orientation`.  
"""
function tangent(circle::L,t) where {T<:Real,L<:DispersingCircleSegments{T}}
    affine_map=circle.cs.affine_map,R=circle.radius,c=circle.center,a=circle.arc_angle,s=circle.shift_angle 
    orient=circle.orientation
    r(t)=affine_map(circle_eq_reversed(R,a,s,c,t))
    return orient*ForwardDiff.derivative(r,t)
end

"""
    tangent_2(circle::L, t::T) :: SVector{2,T} where {T<:Real, L<:DispersingCircleSegments{T}}

Compute the second derivative of the dispersing circle segment at a single parameter `t`.

# Arguments
- `circle::L`: A `DispersingCircleSegment{T}` or `VirtualDispersingCircleSegment{T}`.
- `t::T`: A parameter in `[0,1]`.

# Returns
- `SVector{2,T}`: The second derivative vector at `t`.  
"""
function tangent_2(circle::L,t) where {T<:Real,L<:DispersingCircleSegments{T}}
    return ForwardDiff.derivative(u->tangent(circle,u),t)
end

"""
    tangent_2(circle::L, ts::AbstractVector{T}) :: Vector{SVector{2,T}} where {T<:Real, L<:DispersingCircleSegments{T}}

Compute the second derivative vectors of the dispersing circle segment at each parameter in `ts`.

# Arguments
- `circle::L`: A `DispersingCircleSegment{T}` or `VirtualDispersingCircleSegment{T}`.
- `ts::AbstractVector{T}`: Vector of parameters in `[0,1]`.

# Returns
- `Vector{SVector{2,T}}`: Second derivative vectors at each `t` in `ts`.  
"""
function tangent_2(circle::L,ts::AbstractArray{T,1}) where {T<:Real,L<:DispersingCircleSegments{T}}
    return collect(tangent_2(circle,t) for t in ts)
end

"""
    reversed_circle_segment_domain(pt0::SVector{2,T}, pt1::SVector{2,T}, R::T, center::SVector{2,T}, orient::Int64, x::T, y::T) :: T where T<:Real

Compute signed‐distance relative to a reversed (dispersing) circular segment bounded by chord `pt0 → pt1`.

# Arguments
- `pt0::SVector{2,T}`: First chord endpoint.
- `pt1::SVector{2,T}`: Second chord endpoint.
- `R::T`: Circle radius.
- `center::SVector{2,T}`: Center of circle.
- `orient::Int64`: +1 or −1 for interior side.
- `x::T, y::T`: Point coordinates to test.

# Returns
- `T`: Signed distance: inside if negative, outside if positive, using reversed sign convention.
"""
function reversed_circle_segment_domain(pt0::SVector{2,T},pt1::SVector{2,T},R::T,center::SVector{2,T},orient::Int,x,y) where {T<:Real}
    let cd= -orient*circle_domain(R,center,x,y)
        ld=orient*line_domain(pt0[1],pt0[2],pt1[1],pt1[2],x,y)
        if orient>0
            return ld>zero(ld) ? ld : cd
        else
            return ld<zero(ld) ? ld : cd
        end
    end
end

"""
    domain(circle::L, pts::AbstractVector{SVector{2,T}}) :: Vector{T} where {T<:Real, L<:DispersingCircleSegments{T}}

Compute signed‐distance values for points relative to the dispersing circular segment.

# Arguments
- `circle::L`: A `DispersingCircleSegment{T}` or `VirtualDispersingCircleSegment{T}`.
- `pts::AbstractVector{SVector{2,T}}`: List of points `(x,y)` to test.

# Returns
- `Vector{T}`: Signed distances: negative if inside the dispersing segment, positive if outside.
"""
function domain(circle::L,pts::AbstractArray) where {T<:Real,L<:DispersingCircleSegments{T}}
    let pt0=curve(circle,zero(T)),pt1=curve(circle,one(T)),R=circle.radius,orient=circle.orientation
        center=circle.cs.affine_map(circle.center) #move center to correct position
        return collect(reversed_circle_segment_domain(pt0,pt1,R,center,orient,pt[1],pt[2]) for pt in pts)
    end
end

"""
    is_inside(circle::L, pts::AbstractVector{SVector{2,T}}) :: Vector{Bool} where {T<:Real, L<:DispersingCircleSegments{T}}

Determine if each point is inside the dispersing (concave) circular segment.

# Arguments
- `circle::L`: A `DispersingCircleSegment{T}` or `VirtualDispersingCircleSegment{T}`.
- `pts::AbstractVector{SVector{2,T}}`: List of points `(x,y)` to test.

# Returns
- `Vector{Bool}`: `true` where `domain(circle, pt) < 0` (inside), otherwise `false`.
"""
function is_inside(circle::L,pts::AbstractArray) where {T<:Real,L<:DispersingCircleSegments{T}}
    let pt0=curve(circle,zero(T)),pt1=curve(circle,one(T)),R=circle.radius,orient=circle.orientation
        center=circle.cs.affine_map(circle.center) #move center to correct position
        return collect(reversed_circle_segment_domain(pt0,pt1,R,center,orient,pt[1],pt[2]) < zero(eltype(pt0)) for pt in pts)
    end
end
