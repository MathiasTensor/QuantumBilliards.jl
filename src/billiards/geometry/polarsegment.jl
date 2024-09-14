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
end

PolarSegments{T} = Union{PolarSegment{T}, VirtualPolarSegment{T}} where T<:Real

# Constructor for PolarSegment
function PolarSegment(r_func::Function; origin=SVector(0.0, 0.0), rot_angle=0.0) where {T<:Real}
    cs = PolarCS(SVector(origin...), rot_angle)
    L = compute_arc_length_constructor(r_func, cs.affine_map, 1.0)
    return PolarSegment(cs, r_func, L)
end

# Constructor for VirtualPolarSegment
function VirtualPolarSegment(r_func::Function; origin=SVector(0.0, 0.0), rot_angle=0.0) where {T<:Real}
    cs = PolarCS(SVector(origin...), rot_angle)
    L = compute_arc_length_constructor(r_func, cs.affine_map, 1.0)
    return VirtualPolarSegment(cs, r_func, L)
end

# Curve function
function curve(polar::L, t::T) where {T<:Real, L<:PolarSegments{T}}
    affine_map = polar.cs.affine_map
    return affine_map(polar.r_func(t))
end

function curve(polar::L, ts::AbstractArray{T,1}) where {T<:Real, L<:PolarSegments{T}}
    affine_map = polar.cs.affine_map
    return collect(affine_map(polar.r_func(t)) for t in ts)
end

# Tangent function
function tangent(polar::L, t) where {T, L<:PolarSegments{T}}
    affine_map = polar.cs.affine_map
    r_func = polar.r_func
    # Get the curve at time t (returns SVector)
    curve_at_t = r_func(t)
    # Differentiate each component of the SVector separately
    x_prime = ForwardDiff.derivative(t -> affine_map(r_func(t))[1], t)
    y_prime = ForwardDiff.derivative(t -> affine_map(r_func(t))[2], t)
    # Return the differentiated components as a new SVector
    return SVector(x_prime, y_prime)
end

function tangent(polar::L, ts::AbstractArray{T,1}) where {T, L<:PolarSegments{T}}
    return collect(tangent(polar, t) for t in ts)
end


# Arc length calculation with handling for ForwardDiff.Dual types
function arc_length(polar::L, t::T) where {T<:Real,L<:PolarSegments{T}}
    r_prime(l) = tangent(polar, l)
    integrand(l) = sqrt(value(r_prime(l)[1])^2 + value(r_prime(l)[2])^2)
    length, _ = quadgk(integrand, 0.0, t)
    return length
end

function arc_length(polar::L, ts::AbstractArray{T,1}) where {T<:Real,L<:PolarSegments{T}}
    return collect(arc_length(polar, t) for t in ts)
end

# Helper function for arc length during construction
function compute_arc_length_constructor(r_func::Function, affine_map::AffineMap, t::T) where {T<:Real}
   # Differentiate the x and y components individually
   r_x(l) = affine_map(r_func(l))[1]
   r_y(l) = affine_map(r_func(l))[2]

   println(typeof(r_x(0.5)))
   println(typeof(r_y(0.5)))

   # Compute the partial derivatives of r with respect to l
   r_prime_x(l) = ForwardDiff.derivative(t -> affine_map(r_func(t))[1], l)
   r_prime_y(l) = ForwardDiff.derivative(t -> affine_map(r_func(t))[2], l)

    # Print the types for debugging
    println("Type of r_prime_x(0.5): ", typeof(r_prime_x(0.5)))
    println("Type of r_prime_y(0.5): ", typeof(r_prime_y(0.5)))
    
    # Compute the integrand for arc length calculation
    integrand(l) = sqrt(value(r_prime_x(l))^2 + value(r_prime_y(l))^2)
    
    # Perform the quadrature
    length, _ = quadgk(integrand, 0.0, t)
    return length
end







