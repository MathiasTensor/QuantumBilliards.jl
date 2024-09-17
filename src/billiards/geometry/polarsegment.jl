using QuadGK
using ForwardDiff
using StaticArrays
using LinearAlgebra
using CoordinateTransformations, Rotations

# Struct definitions
struct PolarSegment{T} <: AbsRealCurve where {T<:Real}
    cs::PolarCS{T}
    r_func::Function   # The radial function r(t) provided by the user t -> x,y
    length::T           
end

struct VirtualPolarSegment{T} <: AbsVirtualCurve where {T<:Real}
    cs::PolarCS{T}
    r_func::Function   # The radial function r(t) provided by the user t -> x,y
    length::T 
    symmetry_type::Symbol           
end

PolarSegments{T} = Union{PolarSegment{T}, VirtualPolarSegment{T}} where T<:Real

# Constructor for PolarSegment
function PolarSegment(r_func::Function; origin=SVector(0.0, 0.0), rot_angle=0.0) where {T<:Real}
    cs = PolarCS(SVector(origin...), rot_angle)
    L = compute_arc_length_constructor(r_func, cs.affine_map, 1.0)
    return PolarSegment(cs, r_func, L)
end

# Constructor for VirtualPolarSegment
function VirtualPolarSegment(r_func::Function; symmetry_type=:Dirichlet, origin=SVector(0.0, 0.0), rot_angle=0.0) where {T<:Real}
    cs = PolarCS(SVector(origin...), rot_angle)
    L = compute_arc_length_constructor(r_func, cs.affine_map, 1.0)
    return VirtualPolarSegment(cs, r_func, L, symmetry_type)
end

# Curve function
function curve(polar::L, t) where {T, L<:PolarSegments{T}}
    affine_map = polar.cs.affine_map
    return affine_map(polar.r_func(t))
end

function curve(polar::L, ts::AbstractArray) where {T, L<:PolarSegments{T}}
    affine_map = polar.cs.affine_map
    return collect(affine_map(polar.r_func(t)) for t in ts)
end

# Tangent function
function tangent(polar::L, t) where {T, L<:PolarSegments{T}}
    affine_map = polar.cs.affine_map
    r_func = polar.r_func
    return ForwardDiff.derivative(l -> affine_map(r_func(l)), t)
end

function tangent(polar::L, ts::AbstractArray) where {T, L<:PolarSegments{T}}
    return collect(tangent(polar, t) for t in ts)
end


# Arc length calculation with handling for ForwardDiff.Dual types
function arc_length(polar::L, t) where {T<:Real,L<:PolarSegments{T}}
    r_prime(l) = tangent(polar, l)
    integrand(l) = sqrt(r_prime(l)[1]^2 + r_prime(l)[2]^2)
    length, _ = quadgk(integrand, 0.0, t)
    return length
end

function arc_length(polar::L, ts::AbstractArray) where {T<:Real,L<:PolarSegments{T}}
    return collect(arc_length(polar, t) for t in ts)
end

# Helper function for arc length during construction
function compute_arc_length_constructor(r_func::Function, affine_map::AffineMap, t) where {T<:Real}
    r_prime_x(l) = ForwardDiff.derivative(k -> affine_map(r_func(k))[1], l)
    r_prime_y(l) = ForwardDiff.derivative(k -> affine_map(r_func(k))[2], l)    
    integrand(l) = sqrt(r_prime_x(l)^2 + r_prime_y(l)^2)
    length, _ = quadgk(integrand, 0.0, t)
    return length
end

# Function to compute the area enclosed by a closed PolarSegment !!!!!!
function compute_area(polar::L) where {T<:Real, L<:PolarSegments{T}}
    # Integrand function for the area
    function integrand(t)
        pt = curve(polar, t)           # (x(t), y(t))
        tangent_pt = tangent(polar, t) # (x'(t), y'(t))
        # Compute 0.5 * (x(t) * y'(t) - y(t) * x'(t))
        return 0.5 * (pt[1] * tangent_pt[2] - pt[2] * tangent_pt[1])
    end
    # Perform numerical integration over t from 0 to 1
    area, _ = quadgk(integrand, 0.0, 1.0)
    return abs(area)  # Take absolute value to ensure positive area
end





# THIS NEEDS BETTER LOGIC










function is_inside(polar::L, pts::AbstractArray{SVector{2,T}}) where {T<:Real, L<:PolarSegments{T}}
    num_samples = 500  # Adjust for desired accuracy
    ts = range(0.0, 1.0, length=num_samples)
    affine_map = polar.cs.affine_map
    # Sample points along the curve in the global coordinate system
    curve_points = [curve(polar, t) for t in ts]
    # Construct the polygon (include the transformed origin)
    polygon = [affine_map(SVector{2,T}(0.0, 0.0))]  # Start with the transformed origin
    append!(polygon, curve_points)
    push!(polygon, affine_map(SVector{2,T}(0.0, 0.0)))  # Close the polygon
    return [is_point_in_polygon(polygon, pt) for pt in pts]
end







