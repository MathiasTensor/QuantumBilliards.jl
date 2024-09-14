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
function PolarSegment(r_func::Function, origin=(zero(T), zero(T)), rot_angle=zero(T)) where {T<:Real}
    cs = PolarCS(SVector(origin...), rot_angle)
    L = compute_arc_length_constructor(r_func, cs.affine_map, T(1.0))
    return PolarSegment(cs, r_func, L)
end

# Constructor for VirtualPolarSegment
function VirtualPolarSegment(r_func::Function, origin=(zero(T), zero(T)), rot_angle=zero(T)) where {T<:Real}
    cs = PolarCS(SVector(origin...), rot_angle)
    L = compute_arc_length_constructor(r_func, cs.affine_map, T(1.0))
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

# Arc length calculation
function arc_length(polar::L, t::T) where {T<:Real,L<:PolarSegments{T}}
    r_prime(l) = tangent(polar, l)
    integrand(l) = sqrt(r_prime(l)[1]^2 + r_prime(l)[2]^2)
    length, _ = quadgk(integrand, 0.0, t)
    return length
end

function arc_length(polar::L, ts::AbstractArray{T,1}) where {T<:Real,L<:PolarSegments{T}}
    r_prime(l) = tangent(polar, l)
    integrand(l) = sqrt(r_prime(l)[1]^2 + r_prime(l)[2]^2)
    return collect(quadgk(integrand, 0.0, t)[1] for t in ts)
end

# Helper function for arc length during construction
function compute_arc_length_constructor(r_func::Function, affine_map::AffineMap, t::T) where {T<:Real}
    r_prime(l) = ForwardDiff.derivative(l -> affine_map(r_func(l)), l)
    integrand(l) = sqrt(r_prime(l)[1]^2 + r_prime(l)[2]^2)
    length, _ = quadgk(integrand, 0.0, t)
    return length
end










