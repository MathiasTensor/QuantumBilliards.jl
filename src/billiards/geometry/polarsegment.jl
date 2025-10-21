using QuadGK
using ForwardDiff
using StaticArrays
using LinearAlgebra
using CoordinateTransformations, Rotations

"""
    struct PolarSegment{T} <: AbsRealCurve where T<:Real

Represents a boundary segment defined in polar coordinates by a parametric function `r_func(t)`,
with `t ∈ [0,1]`, mapping to `(x,y)`. Internally stored in a `PolarCS` coordinate system.

# Fields
- `cs::PolarCS{T}`: Polar‐coordinate affine map (origin + rotation).
- `r_func::Function`: User‐provided function `t -> SVector{2,T}` returning `(x(t), y(t))` in local coords.
- `length::T`: Precomputed total arc length of the segment from `t=0` to `t=1`.
"""
struct PolarSegment{T} <: AbsRealCurve where {T<:Real}
    cs::PolarCS{T}
    r_func::Function   # The radial function r(t) provided by the user t -> x,y
    length::T           
end

"""
    struct VirtualPolarSegment{T} <: AbsVirtualCurve where T<:Real

Represents a “virtual” (boundary‐condition) version of a polar segment. Same as `PolarSegment`,
plus a symmetry type (`:Dirichlet` or `:Neumann`).

# Fields
- `cs::PolarCS{T}`: Polar‐coordinate affine map.
- `r_func::Function`: Parametric function `t -> SVector{2,T}` returning `(x(t), y(t))`.
- `length::T`: Arc length from `t=0` to `t=1`.
- `symmetry_type::Symbol`: Either `:Dirichlet` or `:Neumann`.
"""
struct VirtualPolarSegment{T} <: AbsVirtualCurve where {T<:Real}
    cs::PolarCS{T}
    r_func::Function   # The radial function r(t) provided by the user t -> x,y
    length::T 
    symmetry_type::Symbol           
end

# union type alias helper for convenience
PolarSegments{T}=Union{PolarSegment{T},VirtualPolarSegment{T}} where T<:Real

"""
    PolarSegment(r_func::Function; origin::SVector{2,T}=SVector(0.0,0.0), rot_angle::T=0.0) :: PolarSegment{T} where T<:Real

Construct a `PolarSegment{T}` given a parametric function `r_func: t -> SVector{2,T}` mapping `t ∈ [0,1]`
to local coordinates `(x(t), y(t))`. The arc length is computed via numerical quadrature.

# Arguments
- `r_func::Function`: Function `t -> SVector{2,T}` defining the boundary curve in local coords.
- `origin::SVector{2,T}`: Center of the polar coordinate system (default `(0.0,0.0)`).
- `rot_angle::T`: Rotation angle (radians) applied to the entire segment (default `0.0`).

# Returns
- `PolarSegment{T}`: New real polar segment with `length = ∫₀¹ ‖d/dt(affine_map ∘ r_func)(t)‖ dt`.
"""
function PolarSegment(r_func::Function;origin=SVector(0.0,0.0),rot_angle=0.0)
    cs=PolarCS(SVector(origin...),rot_angle)
    L=compute_arc_length_constructor(r_func,cs.affine_map,1.0)
    return PolarSegment(cs,r_func,L)
end

"""
    VirtualPolarSegment(r_func::Function; symmetry_type::Symbol = :Dirichlet, origin::SVector{2,T}=SVector(0.0,0.0), rot_angle::T=0.0) :: VirtualPolarSegment{T} where T<:Real

Construct a `VirtualPolarSegment{T}` given a parametric function `r_func`, plus boundary‐condition type.

# Arguments
- `r_func::Function`: Parametric function `t -> SVector{2,T}` for `(x(t), y(t))`.
- `symmetry_type::Symbol`: Either `:Dirichlet` or `:Neumann` (default `:Dirichlet`).
- `origin::SVector{2,T}`: Center of polar coordinates (default `(0.0,0.0)`).
- `rot_angle::T`: Rotation angle in radians (default `0.0`).

# Returns
- `VirtualPolarSegment{T}`: New virtual segment with computed arc length.
"""
function VirtualPolarSegment(r_func::Function;symmetry_type=:Dirichlet,origin=SVector(0.0,0.0),rot_angle=0.0)
    cs=PolarCS(SVector(origin...),rot_angle)
    L=compute_arc_length_constructor(r_func,cs.affine_map,1.0)
    return VirtualPolarSegment(cs,r_func,L,symmetry_type)
end

"""
    curve(polar::PolarSegment{T}, t::T) :: SVector{2,T} where T<:Real

Compute a single point on the polar segment at parameter `t ∈ [0,1]`, in global Cartesian coords.

# Arguments
- `polar::PolarSegment{T}`: The polar segment.
- `t::T`: Parameter in `[0,1]`.

# Returns
- `SVector{2,T}`: The point `(x,y)` = `affine_map(r_func(t))`.
"""
function curve(polar::L,t) where {T,L<:PolarSegments{T}}
    affine_map=polar.cs.affine_map
    return affine_map(polar.r_func(t))
end

"""
    curve(polar::PolarSegment{T}, ts::AbstractVector{T}) :: Vector{SVector{2,T}} where T<:Real

Compute multiple points along the polar segment for each `t ∈ ts`, in global Cartesian coords.

# Arguments
- `polar::PolarSegment{T}`: The polar segment.
- `ts::AbstractVector{T}`: Vector of parameters in `[0,1]`.

# Returns
- `Vector{SVector{2,T}}`: List of points `affine_map(r_func(t))` for each `t`.
"""
function curve(polar::L,ts::AbstractArray) where {T,L<:PolarSegments{T}}
    affine_map=polar.cs.affine_map
    return collect(affine_map(polar.r_func(t)) for t in ts)
end

"""
    tangent(polar::PolarSegment{T},t) :: SVector{2,T} where T<:Real

Compute the tangent vector (derivative w.r.t. `t`) of the polar segment at `t`, in global coords.

# Arguments
- `polar::PolarSegment{T}`: The polar segment.
- `t::T`: Parameter in `[0,1]`.

# Returns
- `SVector{2,T}`: The derivative `d/dt [ affine_map( r_func(t) ) ]`.
"""
function tangent(polar::L,t) where {T,L<:PolarSegments{T}}
    affine_map=polar.cs.affine_map
    r_func=polar.r_func
    return ForwardDiff.derivative(l->affine_map(r_func(l)),t)
end

"""
    tangent_2(polar::PolarSegment{T},t) :: SVector{2,T} where T<:Real

Compute the tangent vector derivative (2nd derivative w.r.t. `t`) of the polar segment at `t`, in global coords.

# Arguments
- `polar::PolarSegment{T}`: The polar segment.
- `t::T`: Parameter in `[0,1]`.

# Returns
- `SVector{2,T}`: The derivative `d^2/dt^2 [ affine_map( r_func(t) ) ]`.
"""
function tangent_2(polar::L,t) where {T,L<:PolarSegments{T}}
    return ForwardDiff.derivative(u->tangent(polar,u),t)
end

"""
    tangent(polar::PolarSegment{T}, ts::AbstractVector{T}) :: Vector{SVector{2,T}} where T<:Real

Compute tangent vectors at multiple `t ∈ ts`.

# Arguments
- `polar::PolarSegment{T}`: The polar segment.
- `ts::AbstractVector{T}`: Vector of parameters in `[0,1]`.

# Returns
- `Vector{SVector{2,T}}`: Tangent vectors at each `t`.
"""
function tangent(polar::L,ts::AbstractArray) where {T,L<:PolarSegments{T}}
    return collect(tangent(polar,t) for t in ts)
end

"""
    tangent_2(polar::PolarSegment{T}, ts::AbstractVector{T}) :: Vector{SVector{2,T}} where T<:Real

Compute derivative of tangent vectors at multiple `t ∈ ts`.

# Arguments
- `polar::PolarSegment{T}`: The polar segment.
- `ts::AbstractVector{T}`: Vector of parameters in `[0,1]`.

# Returns
- `Vector{SVector{2,T}}`: Derivatives of tangent vectors at each `t`.
"""
function tangent_2(polar::L,ts::AbstractArray) where {T,L<:PolarSegments{T}}
    return collect(tangent_2(polar,t) for t in ts)
end

"""
    arc_length(polar::PolarSegment{T}, t::T) :: T where T<:Real

Compute the arc length from `0` up to parameter `t` via numerical integration.

# Arguments
- `polar::PolarSegment{T}`: The polar segment.
- `t::T`: Upper limit parameter ∈ `[0,1]`.

# Returns
- `T`: `∫₀ᵗ ‖ tangent(polar, s) ‖ ds`.
"""
function arc_length(polar::L,t) where {T<:Real,L<:PolarSegments{T}}
    # Arc length calculation with handling for ForwardDiff.Dual types
    r_prime(l)=tangent(polar,l)
    integrand(l)=sqrt(r_prime(l)[1]^2+r_prime(l)[2]^2)
    length,_=quadgk(integrand,0.0,t)
    return length
end

"""
    arc_length(polar::PolarSegment{T}, ts::AbstractVector{T}) :: Vector{T} where T<:Real

Compute arc lengths for each `t ∈ ts`.

# Arguments
- `polar::PolarSegment{T}`: The polar segment.
- `ts::AbstractVector{T}`: Vector of parameters in `[0,1]`.

# Returns
- `Vector{T}`: List of lengths `arc_length(polar, t)` for each `t`.
"""
function arc_length(polar::L,ts::AbstractArray) where {T<:Real,L<:PolarSegments{T}}
    return collect(arc_length(polar,t) for t in ts)
end

#TODO Change!!!
""" 
    compute_arc_length_constructor(r_func::Function, affine_map::Function, t::T) :: T where T<:Real

Helper to compute total arc length at construction time (from `0` to `t`), using numerical quadrature.

# Arguments
- `r_func::Function`: Parametric function `t -> SVector{2,T}`.
- `affine_map::Function`: Affine map from local to global coords.
- `t::T`: Upper limit parameter (usually `1.0`).

# Returns
- `T`: Arc length `∫₀ᵗ ‖ d/dτ [ affine_map(r_func(τ)) ] ‖ dτ`.
"""
function compute_arc_length_constructor(r_func::Function,affine_map::AffineMap,t)
    r_prime_x(l)=ForwardDiff.derivative(k->affine_map(r_func(k))[1],l)
    r_prime_y(l)=ForwardDiff.derivative(k->affine_map(r_func(k))[2],l)    
    integrand(l)=sqrt(r_prime_x(l)^2+r_prime_y(l)^2)
    length,_=quadgk(integrand,0.0,t)
    return length
end


"""
    compute_area(polar::PolarSegment{T}) :: T where T<:Real

Compute the area enclosed by a closed polar segment (assumed to form a simple, oriented loop)
using Green’s theorem:  
`Area = ∮ (x dy − y dx)/2 = ∫₀¹ 0.5 [ x(t)*y′(t) − y(t)*x′(t) ] dt`.

# Arguments
- `polar::PolarSegment{T}`: Polar segment defining a closed curve.

# Returns
- `T`: Absolute area enclosed by the curve.
"""
function compute_area(polar::L) where {T<:Real,L<:PolarSegments{T}}
    # Integrand function for the area
    function integrand(t)
        pt=curve(polar,t)           # (x(t), y(t))
        tangent_pt=tangent(polar,t) # (x'(t), y'(t))
        # Compute 0.5 * (x(t) * y'(t) - y(t) * x'(t))
        return 0.5*(pt[1]*tangent_pt[2]-pt[2]*tangent_pt[1])
    end
    area,_=quadgk(integrand,0.0,1.0) # numerical integration over t from 0 to 1
    return abs(area)  # Take absolute value to ensure positive area
end

#TODO This is breaking for PSM
"""
    is_inside(polar::PolarSegment{T}, pts::AbstractVector{SVector{2,T}}) :: Vector{Bool} where T<:Real

Test whether each point in `pts` lies inside the region enclosed by the closed polar segment.
Approximates the boundary as a polygon sampled at `num_samples` points and uses a standard point‐in‐polygon test.

# Arguments
- `polar::PolarSegment{T}`: Polar segment forming a closed loop.
- `pts::AbstractVector{SVector{2,T}}`: Points to test.

# Returns
- `Vector{Bool}`: `true` if the point is inside the polygon, `false` otherwise.
"""
function is_inside(polar::L,pts::AbstractArray{SVector{2,T}}) where {T<:Real,L<:PolarSegments{T}}
    num_samples=500  # Adjust for desired accuracy
    ts=range(0.0,1.0,length=num_samples)
    affine_map=polar.cs.affine_map
    # Sample points along the curve in the global coordinate system
    curve_points=[curve(polar,t) for t in ts]
    # Construct the polygon (include the transformed origin)
    polygon=[affine_map(SVector{2,T}(0.0,0.0))]  # Start with the transformed origin
    append!(polygon,curve_points)
    push!(polygon,affine_map(SVector{2,T}(0.0,0.0)))  # Close the polygon
    return [is_point_in_polygon(polygon, pt) for pt in pts]
end







