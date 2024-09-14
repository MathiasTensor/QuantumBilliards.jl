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
function tangent(polar::L, t::T) where {T<:Real, L<:PolarSegments{T}}
    affine_map = polar.cs.affine_map 
    r(t) = affine_map(polar.r_func(t))
    return ForwardDiff.derivative(r, t)
end

function tangent(polar::L, ts::AbstractArray{T,1}) where {T<:Real, L<:PolarSegments{T}}
    affine_map = polar.cs.affine_map
    r(t) = affine_map(polar.r_func(t))
    return collect(ForwardDiff.derivative(r, t) for t in ts)
end

# Arc length calculation with handling for ForwardDiff.Dual types
function arc_length(polar::L, t::T) where {T<:Real,L<:PolarSegments{T}}
    r_prime(l) = tangent(polar, l)
    
    integrand(l) = sqrt(value(r_prime(l)[1])^2 + value(r_prime(l)[2])^2)  # Extracting the values from Dual numbers
    length, _ = quadgk(integrand, 0.0, t)
    return length
end

function arc_length(polar::L, ts::AbstractArray{T,1}) where {T<:Real,L<:PolarSegments{T}}
    r_prime(l) = tangent(polar, l)
    
    integrand(l) = sqrt(value(r_prime(l)[1])^2 + value(r_prime(l)[2])^2)  # Handling Dual numbers
    return collect(quadgk(integrand, 0.0, t)[1] for t in ts)
end

# Helper function for arc length during construction
function compute_arc_length_constructor(r_func::Function, affine_map::AffineMap, t::T) where {T<:Real}
    r_prime(l) = ForwardDiff.derivative(l -> affine_map(r_func(l)), l)
    
    integrand(l) = sqrt(value(r_prime(l)[1])^2 + value(r_prime(l)[2])^2)  # Extracting Dual values
    length, _ = quadgk(integrand, 0.0, t)
    return length
end







