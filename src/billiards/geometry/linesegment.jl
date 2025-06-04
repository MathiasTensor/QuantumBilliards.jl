
"""
    line_eq(pt0::SVector{2,T}, pt1::SVector{2,T}, t::T) :: SVector{2,T} where T<:Real

Compute a point on the straight line between `pt0` and `pt1`, parameterized by `t ∈ [0,1]`.

# Arguments
- `pt0::SVector{2,T}`: Starting endpoint `(x₀, y₀)`.
- `pt1::SVector{2,T}`: Ending endpoint `(x₁, y₁)`.
- `t::T`: Parameter in `[0,1]` indicating position along the segment.

# Returns
- `SVector{2,T}`: Coordinates `(x, y)` on the line at parameter `t`:  
  `x = (1−t)*pt0[1] + t*pt1[1]`,  
  `y = (1−t)*pt0[2] + t*pt1[2]`.
"""
line_eq(pt0::SVector{2,T},pt1::SVector{2,T},t) where {T<:Real}=@. (pt1-pt0)*t+pt0

"""
    line_domain(x0::T, y0::T, x1::T, y1::T, x::T, y::T) :: T where T<:Real

Compute the signed‐distance‐type value from the infinite line through `(x0,y0) → (x1,y1)` to a test point `(x,y)`.

# Arguments
- `x0::T, y0::T`: Coordinates of the first point on the line.
- `x1::T, y1::T`: Coordinates of the second point on the line.
- `x::T, y::T`: Coordinates of the test point.

# Returns
- `T`: The value `( (y1 − y0)*x − (x1 − x0)*y + x1*y0 − y1*x0 )`.  
  Positive on one side of the directed line, negative on the other.
"""
line_domain(x0,y0,x1,y1,x,y)=((y1-y0)*x-(x1-x0)*y+x1*y0-y1*x0)

"""
    struct LineSegment{T} <: AbsRealCurve where T<:Real

Represents a straight, “real” line‐segment portion of a billiard boundary.

Fields:
- `cs::CartesianCS{T}`: Coordinate‐system (translation + rotation).
- `pt0::SVector{2,T}`: First endpoint `(x₀, y₀)`.
- `pt1::SVector{2,T}`: Second endpoint `(x₁, y₁)`.
- `orientation::Int64`: +1 or −1 indicating which side is “inside”.
- `length::T`: Euclidean length of the segment = `hypot(pt1 − pt0)`.
"""
struct LineSegment{T} <: AbsRealCurve where T<:Real
    cs::CartesianCS{T}
    pt0::SVector{2,T}
    pt1::SVector{2,T}
    orientation::Int64
    length::T
end

"""
    LineSegment(pt0::SVector{2,T}, pt1::SVector{2,T}; origin::Tuple{T,T} = zero(pt0), rot_angle::T = zero(T), orientation::Int64 = 1) :: LineSegment{T} where T<:Real

Construct a `LineSegment{T}` between two points `pt0` and `pt1`, with a Cartesian coordinate‐system.

# Arguments
- `pt0::SVector{2,T}`: Starting endpoint `(x₀, y₀)`.
- `pt1::SVector{2,T}`: Ending endpoint `(x₁, y₁)`.
- `origin::Tuple{T,T}`: Translation of the coordinate system (default `(0,0)`).
- `rot_angle::T`: Rotation of the coordinate system (radians) (default `0.0`).
- `orientation::Int64`: +1 or −1 indicating “inside” direction (default `1`).

# Returns
- `LineSegment{T}`: New segment with `length = hypot(pt1 − pt0)`.
"""
function LineSegment(pt0::SVector{2,T},pt1::SVector{2,T};origin=zero(pt0),rot_angle=zero(eltype(pt0)),orientation=1) where T<:Real
    cs=CartesianCS(SVector(origin...),rot_angle)
    x,y=pt1.-pt0        
    L=hypot(x,y)
    return LineSegment(cs,pt0,pt1,orientation,L)
end

"""
    struct VirtualLineSegment{T} <: AbsVirtualCurve where T<:Real

Represents a “virtual” (boundary‐condition) line segment.  
Same fields as `LineSegment{T}`, plus `symmetry_type` for Dirichlet/Neumann.

Fields:
- `cs::CartesianCS{T}`: Coordinate‐system.
- `pt0::SVector{2,T}`: First endpoint `(x₀, y₀)`.
- `pt1::SVector{2,T}`: Second endpoint `(x₁, y₁)`.
- `orientation::Int64`: +1 or −1 for “inside”.
- `length::T`: Segment length = `hypot(pt1 − pt0)`.
- `symmetry_type::Symbol`: `:Dirichlet` or `:Neumann`.
"""
struct VirtualLineSegment{T} <: AbsVirtualCurve where T<:Real
    cs::CartesianCS{T}
    pt0::SVector{2,T}
    pt1::SVector{2,T}
    orientation::Int64
    length::T
    symmetry_type::Symbol
end

"""
    VirtualLineSegment(pt0::SVector{2,T}, pt1::SVector{2,T}; symmetry_type::Symbol = :Dirichlet, origin::Tuple{T,T} = zero(pt0), rot_angle::T = zero(T), orientation::Int64 = 1) :: VirtualLineSegment{T} where T<:Real

Construct a `VirtualLineSegment{T}` for boundary conditions.

# Arguments
- `pt0::SVector{2,T}`: Starting endpoint `(x₀, y₀)`.
- `pt1::SVector{2,T}`: Ending endpoint `(x₁, y₁)`.
- `symmetry_type::Symbol`: Either `:Dirichlet` or `:Neumann` (default `:Dirichlet`).
- `origin::Tuple{T,T}`: Translation (default `(0,0)`).
- `rot_angle::T`: Rotation (radians) (default `0.0`).
- `orientation::Int64`: +1 or −1 for “inside” (default `1`).

# Returns
- `VirtualLineSegment{T}`: New virtual segment with `length = hypot(pt1 − pt0)`.
"""
function VirtualLineSegment(pt0::SVector{2,T},pt1::SVector{2,T};symmetry_type=:Dirichlet,origin=zero(pt0),rot_angle=zero(eltype(pt0)),orientation=1) where T<:Real
    cs=CartesianCS(SVector(origin...),rot_angle)
    x,y=pt1.-pt0        
    L=hypot(x,y)
    return VirtualLineSegment(cs,pt0,pt1,orientation,L,symmetry_type)
end

# union type alias for convenience
LineSegments{T}=Union{LineSegment{T},VirtualLineSegment{T}} where T<:Real

"""
    curve(line::L, ts::AbstractVector{T}) :: Vector{SVector{2,T}} 
        where {T<:Real, L<:LineSegments{T}}

Compute points along the (real or virtual) line segment for each parameter `t ∈ ts`.

# Arguments
- `line::L`: A `LineSegment{T}` or `VirtualLineSegment{T}`.
- `ts::AbstractVector{T}`: Vector of `t` values ∈ `[0,1]`.

# Returns
- `Vector{SVector{2,T}}`: List of coordinates `(x,y)` along the segment.
"""
function curve(line::L,ts::AbstractArray{T,1}) where {T<:Real,L<:LineSegments{T}}
    let pt0=line.cs.affine_map(line.pt0),pt1=line.cs.affine_map(line.pt1)
    return collect(line_eq(pt0,pt1,t) for t in ts)
    end
end

"""
    curve(line::L, t::T) :: SVector{2,T} where {T<:Real, L<:LineSegments{T}}

Compute a single point on the (real or virtual) line segment at parameter `t ∈ [0,1]`.

# Arguments
- `line::L`: A `LineSegment{T}` or `VirtualLineSegment{T}`.
- `t::T`: Parameter ∈ `[0,1]`.

# Returns
- `SVector{2,T}`: Coordinates `(x,y)` at that parameter.
"""
function curve(line::L,t) where {T<:Real,L<:LineSegments{T}}
    let pt0=line.cs.affine_map(line.pt0),pt1=line.cs.affine_map(line.pt1)
    return line_eq(pt0,pt1,t)
    end
end

"""
    arc_length(line::L, ts::AbstractVector{T}) :: Vector{T} where {T<:Real, L<:LineSegments{T}}

Compute arc‐length values for each `t ∈ ts` along the segment.  
Length = `line.length * t`.

# Arguments
- `line::L`: A `LineSegment{T}` or `VirtualLineSegment{T}`.
- `ts::AbstractVector{T}`: Vector of parameters ∈ `[0,1]`.

# Returns
- `Vector{T}`: Arc lengths = `line.length * t` for each `t`.
"""
function arc_length(line::L,ts::AbstractArray{T,1}) where {T<:Real,L<:LineSegments{T}}
    s::Vector{T}=line.length.*ts
    return s
end

"""
    tangent(line::L, ts::AbstractVector{T}) :: Vector{SVector{2,T}} where {T<:Real, L<:LineSegments{T}}

Compute tangent (derivative) vectors along the segment at each `t ∈ ts`.

# Arguments
- `line::L`: A `LineSegment{T}` or `VirtualLineSegment{T}`.
- `ts::AbstractVector{T}`: Vector of parameters ∈ `[0,1]`.

# Returns
- `Vector{SVector{2,T}}`: Tangent vectors at each `t`, scaled by `orientation`.
"""
function tangent(line::L,ts::AbstractArray{T,1}) where {T<:Real,L<:LineSegments{T}}
    let pt0=line.cs.affine_map(line.pt0),pt1=line.cs.affine_map(line.pt1),orient=line.orientation
        r(t)=line_eq(pt0,pt1,t)
        return collect(orient*ForwardDiff.derivative(r,t) for t in ts)
    end
end

"""
    domain(line::L, pts::AbstractVector{<:SVector{2,T}}) :: Vector{T} where {T<:Real, L<:LineSegments{T}}

Compute signed‐distance values for points relative to the infinite line containing the segment.

# Arguments
- `line::L`: A `LineSegment{T}` or `VirtualLineSegment{T}`.
- `pts::AbstractVector{SVector{2,T}}`: List of points `(x,y)`.

# Returns
- `Vector{T}`: Signed values = `line_domain(pt0_x, pt0_y, pt1_x, pt1_y, x, y) * orientation`.  
  Negative if “inside” side, positive if “outside,” by `orientation`.
"""
function domain(line::L,pts::AbstractArray) where {T<:Real,L<:LineSegments{T}}
    let pt0=line.cs.affine_map(line.pt0) 
        pt1=line.cs.affine_map(line.pt1)
        orientation=line.orientation
    return collect(line_domain(pt0[1],pt0[2],pt1[1],pt1[2],pt[1],pt[2])*orientation for pt in pts)
    end
end

"""
    is_inside(line::L, pts::AbstractVector{<:SVector{2,T}}) :: Vector{Bool} where {T<:Real, L<:LineSegments{T}}

Determine if each point is inside (on the “interior” side of) the segment’s line.

# Arguments
- `line::L`: A `LineSegment{T}` or `VirtualLineSegment{T}`.
- `pts::AbstractVector{SVector{2,T}}`: List of points `(x,y)`.

# Returns
- `Vector{Bool}`: `true` where `domain(line,pt) < 0`, otherwise `false`.
"""
function is_inside(line::L,pts::AbstractArray) where {T<:Real,L<:LineSegments{T}}
    let pt0=line.cs.affine_map(line.pt0) 
        pt1=line.cs.affine_map(line.pt1)
        orientation=line.orientation
    return collect(line_domain(pt0[1],pt0[2],pt1[1],pt1[2],pt[1],pt[2])*orientation < zero(eltype(pt0)) for pt in pts)
    end
end

