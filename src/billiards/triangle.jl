

"""
    triangle_corners(angles, x0, y0, h) -> Vector{SVector{2,T}}

Compute the coordinates of the three corners of a triangle defined by its internal angles.

# Arguments
- `angles::Vector{<:Real}`: A vector of the three internal angles `[α, β, γ]` (in radians), which must sum to π.
- `x0::Real`: x-coordinate of the corner with angle γ (used as the origin reference).
- `y0::Real`: y-coordinate of the corner with angle γ.
- `h::Real`: The height of the triangle from the base to the corner with angle γ.

# Returns
- `Vector{SVector{2,T}}`: The triangle corners `[A, B, C]`, where C is the corner at `(x0, y0)` with angle γ.
"""
function triangle_corners(angles,x0,y0,h) #x0, y0 position of gamma corner
    alpha,beta,gamma=angles
    B=SVector((-h/tan(beta+alpha))-x0,h-y0)
    A=SVector(h/tan(alpha)+B[1],-y0)
    C=SVector(-x0,-y0)
    return [A,B,C]
end

"""
    struct Triangle{T} <: AbsBilliard

A data structure representing a triangle-shaped billiard, with optional symmetry-reduced boundaries.

# Fields
- `fundamental_boundary::Vector{Union{LineSegment, VirtualLineSegment}}`: The boundary used in calculations, can contain virtual edges for symmetry.
- `full_boundary::Vector{LineSegment}`: The full triangle boundary, all edges are real.
- `desymmetrized_full_boundary::Vector{LineSegment}`: Same as `full_boundary`, can be used to store desymmetrized versions.
- `length::T`: Total perimeter length of the triangle.
- `length_fundamental::T`: Length of the symmetry-reduced boundary (if using virtual edges).
- `area::T`: Full area of the triangle.
- `area_fundamental::T`: Area of the fundamental domain (usually equals `area`).
- `corners::Vector{SVector{2,T}}`: Coordinates of the triangle's three corners `[A, B, C]`.
- `angles::Vector`: List of angles `[α, β, γ]` in the full triangle.
- `angles_fundamental::Vector`: Angles used for the fundamental domain (may match `angles`).
"""
struct Triangle{T} <: AbsBilliard where {T<:Real}
    fundamental_boundary::Vector{Union{LineSegment,VirtualLineSegment}}
    full_boundary::Vector{LineSegment}
    desymmetrized_full_boundary::Vector{LineSegment}
    length::T
    length_fundamental::T
    area::T
    area_fundamental::T
    corners::Vector{SVector{2,T}}
    angles::Vector
    angles_fundamental::Vector
end

"""
    Triangle(gamma, chi; curve_types=[:Real, :Virtual, :Virtual], x0=0.0, y0=0.0, h=1.0) -> Triangle

Construct a triangle given one internal angle `γ` and a ratio `χ = β / α` that determines the triangle's shape.

# Arguments
- `gamma::Real`: Internal angle at corner C (placed at `(x0, y0)`).
- `chi::Real`: The ratio β/α, used to define triangle shape.
- `curve_types::Vector{Symbol}`: Optional list of 3 boundary types (e.g., `:Real` or `:Virtual`) for symmetry reductions. Best kept default as this corresponds to CAFB simplification.
- `x0::Real`: x-position of corner C (angle γ).
- `y0::Real`: y-position of corner C.
- `h::Real`: Height of triangle from base to corner C.

# Returns
- `Triangle`: A fully initialized `Triangle` object with boundary, area, and geometric metadata.
"""
function Triangle(gamma,chi;curve_types=[:Real,:Virtual,:Virtual],x0=zero(gamma),y0=zero(gamma),h=one(gamma))
    alpha=(pi-gamma)/(1+chi)
    beta=alpha*chi
    angles_old=[alpha,beta,gamma]
    corners=triangle_corners(angles_old,x0,y0,h)
    fundamental_boundary=make_polygon(corners,curve_types)
    full_boundary=make_polygon(corners,[:Real,:Real,:Real]) # make all the virtual real
    desymmetrized_full_boundary=full_boundary # same as full boundary
    length=sum([crv.length for crv in full_boundary])
    length_fundamental=symmetry_accounted_fundamental_boundary_length(fundamental_boundary)
    area=0.5*h*abs(corners[1][1]-corners[3][1])
    area_fundamental=area
    angles_new=[alpha,beta,gamma]
    return Triangle(fundamental_boundary,full_boundary,desymmetrized_full_boundary,length,length_fundamental,area,area_fundamental,corners,angles_new,angles_new)
end

"""
    adapt_basis(triangle::Triangle, i::Integer) -> Tuple{Real, PolarCS, Nothing}

Construct a polar coordinate system centered at the `i`-th corner of a triangle for use in corner-adapted basis functions.

# Arguments
- `triangle::Triangle`: The triangle geometry object.
- `i::Integer`: The index of the edge (1-based) for which to compute the adapted coordinate system.

# Returns
- `angle::Real`: The internal angle at the selected corner.
- `cs::PolarCS`: A polar coordinate system with origin at the corner and angle-aligned axis.
- `symmetry::Nothing`: Placeholder for symmetry information (not used currently).
"""
function adapt_basis(triangle::T,i::Ti) where {T<:Triangle,Ti<:Integer}
    N=3
    c=triangle.corners
    i0=mod1(i,N)
    i1=mod1(i+1,N)
    a=c[i1]-c[i0]
    rot_angle=atan(a[2],a[1])#angle(x_axis, a)
    origin=c[i0]
    cs=PolarCS(origin,rot_angle)
    return triangle.angles[i0],cs,nothing
end

"""
    make_triangle_and_basis(gamma, chi; edge_i=1) -> Tuple{Triangle, CornerAdaptedFourierBessel}

Convenience function to create a triangle and construct a corner-adapted Fourier-Bessel basis at a selected edge.

# Arguments
- `gamma::Real`: Internal angle at the base corner (γ).
- `chi::Real`: Shape control parameter, defines ratio β/α.
- `edge_i::Integer=1`: Index of the real edge used to place and adapt the basis.

# Returns
- `tr::Triangle`: The constructed triangle object with virtual edges applied.
- `basis::CornerAdaptedFourierBessel`: A Fourier-Bessel basis adapted to the corner opposite edge `edge_i`.
"""
function make_triangle_and_basis(gamma,chi; edge_i=1)
    cor=Triangle(gamma,chi).corners
    x0,y0=cor[mod1(edge_i+2,3)]
    re=[:Virtua,:Virtual,:Virtual]
    re[edge_i]=:Real 
    tr=Triangle(gamma,chi;curve_types=re,x0=x0,y0=y0)
    angle,cs,symmetry=adapt_basis(tr,edge_i+2)
    basis=CornerAdaptedFourierBessel(10,angle,cs,symmetry)
    return tr,basis 
end
