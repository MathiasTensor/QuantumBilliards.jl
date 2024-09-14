using QuadGK
using ForwardDiff
using StaticArrays
using LinearAlgebra
using CoordinateTransformations, Rotations

struct PolarSegment{T} <: AbsRealCurve where {T<:Real}
    cs::PolarCS{T}
    r_func::Function   # The radial function r(t) provided by the user. It has the signature t -> x(t),y(t)
    length::T            
end

struct VirtualPolarSegment{T} <: AbsVirtualCurve where {T<:Real}
    cs::PolarCS{T}
    r_func::Function   # The radial function r(t) provided by the user. It has the signature t -> x(t),y(t)
    length::T            
end

function PolarSegment(r_func::Function, origin=(zero(T),zero(T)), rot_angle=zero(T)) where {T<:Real}
    cs = PolarCS(SVector(origin...),rot_angle)
    L = compute_arc_length_constructor(r_func, cs.affine_map, 1.0)
    return PolarSegment(cs, r_func, L)
end

function VirtualPolarSegment(r_func::Function, origin=(zero(T),zero(T)), rot_angle=zero(T)) where {T<:Real}
    cs = PolarCS(SVector(origin...),rot_angle)
    L = compute_arc_length_constructor(r_func, cs.affine_map, 1.0)
    return PolarSegment(cs, r_func, L)
end

PolarSegments{T} = Union{PolarSegment{T},VirtualPolarSegment{T}} where T<:Real

function curve(polar::L, t) where {T<:Real, L<:PolarSegments{T}}
    let affine_map = polar.cs.affine_map
        return affine_map(polar.r_func(t))
    end
end

function curve(polar::L, ts) where {T<:Real, L<:PolarSegments{T}}
    let affine_map = polar.cs.affine_map
        return collect(affine_map(polar.r_func(t)) for t in ts)
    end
end

function tangent(polar::L, t) where {T<:Real, L<:PolarSegments{T}}
    let affine_map = circle.cs.affine_map 
        r(t) = affine_map(polar.r_func(t))
        return ForwardDiff.derivative(r, t)
    end
end

function tangent(polar::L, ts::AbstractArray) where {T<:Real, L<:PolarSegments{T}}
    let affine_map = polar.cs.affine_map
        r(t) = affine_map(polar.r_func(t))
        return collect(ForwardDiff.derivative(r, t) for t in ts)
    end
end

function arc_length(polar::L, t) where {T<:Real,L<:PolarSegments{T}}
    r_prime(l) = tangent(polar, l)
    integrand(l) = sqrt(r_prime(l)[1]^2 + r_prime(l)[2]^2)
    length, _ = quadgk(integrand, 0.0, t)
    return length
end

function arc_length(polar::L, ts::AbstractArray) where {T<:Real,L<:PolarSegments{T}}
    r_prime(l) = tangent(polar, l)
    integrand(l) = sqrt(r_prime(l)[1]^2 + r_prime(l)[2]^2)
    return collect(quadgk(integrand, 0.0, t)[1] for t in ts)
end

# helper function hack
function compute_arc_length_constructor(r_func::Function, affine_map::AffineMap, t) where {T<:Real}
    r_prime(l) = ForwardDiff.derivative(t -> affine_map(r_func(t)), l)
    integrand(l) = sqrt(r_prime(l)[1]^2 + r_prime(l)[2]^2)
    length, _ = quadgk(integrand, 0.0, t)
    return length
end













