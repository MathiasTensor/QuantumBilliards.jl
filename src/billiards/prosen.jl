








"""
    make_quarter_prosen(a::T; x0=zero(T), y0=zero(T), rot_angle=zero(T)) where {T<:Real}

Constructs a quarter Prosen billiard.

# Arguments
- `a::T`: The deformation parameter.
- `x0::T=zero(T)`: The x-coordinate of the center of the billiard.
- `y0::T=zero(T)`: The y-coordinate of the center of the billiard.
- `rot_angle::T=zero(T)`: The rotation angle of the billiard.

# Returns
- A tuple containing:
  - `boundary::Vector{Union{PolarSegment{T}, VirtualLineSegment{T}}}`: The boundary segments of the quarter Prosen billiard.
  - `corners::Vector{SVector{2,T}}`: The corner points of the quarter billiard.
"""
function make_quarter_prosen(a::T; x0=zero(T), y0=zero(T), rot_angle=zero(T)) where {T<:Real}
    origin = SVector(x0, y0)
    φ_multiplier = 0.5  # t from 0 to 1 maps to φ from 0 to π/2

    # Define the radial function for the quarter Prosen billiard
    r_func = t -> begin
        φ = φ_multiplier * π * t  # φ ranges from 0 to π/2
        r = one(T) + a * cos(4 * φ)
        x = r * cos(φ)
        y = r * sin(φ)
        SVector(x, y)
    end

    # Create the quarter Prosen billiard segment
    quarter_prosen_segment = PolarSegment(r_func; origin=origin, rot_angle=rot_angle)

    # Compute the start and end points of the quarter segment
    pt0 = curve(quarter_prosen_segment, zero(T))  # Start point at φ = 0
    pt1 = curve(quarter_prosen_segment, one(T))   # End point at φ = π/2

    # Create virtual line segments to close the quarter billiard
    line_segment1 = VirtualLineSegment(pt1, SVector{2,T}(x0, y0); origin=origin, rot_angle=rot_angle)
    line_segment2 = VirtualLineSegment(SVector{2,T}(x0, y0), pt0; origin=origin, rot_angle=rot_angle)

    # Construct the boundary
    boundary = Union{PolarSegment{T}, VirtualLineSegment{T}}[
        quarter_prosen_segment,
        line_segment1,
        line_segment2
    ]

    # Corners are pt0, pt1, and the origin
    corners = [pt0, pt1, SVector(x0, y0)]

    return boundary, corners
end



"""
    make_full_prosen(a::T; x0=zero(T), y0=zero(T), rot_angle=zero(T)) where {T<:Real}

Constructs a full Prosen billiard.

# Arguments
- `a::T`: The deformation parameter.
- `x0::T=zero(T)`: The x-coordinate of the center of the billiard.
- `y0::T=zero(T)`: The y-coordinate of the center of the billiard.
- `rot_angle::T=zero(T)`: The rotation angle of the billiard.

# Returns
- A tuple containing:
  - `boundary::Vector{PolarSegment{T}}`: The boundary segments of the full Prosen billiard.
  - `corners::Vector{SVector{2,T}}`: An empty vector (no corners for the full billiard).
  - `area_full::T`: The area of the full Prosen billiard.
"""
function make_full_prosen(a::T; x0=zero(T), y0=zero(T), rot_angle=zero(T)) where {T<:Real}
    origin = SVector(x0, y0)
    θ_multiplier = 2.0  # t from 0 to 1 maps to φ from 0 to 2π

    # Define the radial function for the full Prosen billiard
    r_func = t -> begin
        φ = θ_multiplier * π * t  # φ ranges from 0 to 2π
        r = one(T) + a * cos(4 * φ)
        x = r * cos(φ)
        y = r * sin(φ)
        SVector(x, y)
    end

    # Create the full Prosen billiard segment
    full_prosen_segment = PolarSegment(r_func; origin=origin, rot_angle=rot_angle)
    area_full = compute_area(full_prosen_segment)

    boundary = [full_prosen_segment]
    corners = []  # Empty vector for corners

    return boundary, corners, area_full
end


"""
    struct ProsenBilliard{T} <: QuantumBilliards.AbsBilliard where {T<:Real}

Defines a Prosen billiard with quarter, half, and full boundaries.

# Fields
- `fundamental_boundary::Vector`: The boundary segments of the quarter Prosen billiard.
- `full_boundary::Vector`: The boundary segments of the full Prosen billiard.
- `length::T`: The total length of the full boundary.
- `a::T`: The deformation parameter.
- `corners::Vector{SVector{2,T}}`: The corner points of the billiard.
"""
struct ProsenBilliard{T} <: AbsBilliard where {T<:Real}
    fundamental_boundary::Vector
    full_boundary::Vector
    length::T
    length_fundamental::T
    a::T
    corners::Vector{SVector{2,T}}
    area::T
    area_fundamental::T
end


"""
    ProsenBilliard(a::T; x0=zero(T), y0=zero(T), rot_angle=zero(T)) :: ProsenBilliard where {T<:Real}

Constructs a Prosen billiard.

# Arguments
- `a::T`: The deformation parameter.
- `x0::T=zero(T)`: The x-coordinate of the center of the billiard.
- `y0::T=zero(T)`: The y-coordinate of the center of the billiard.
- `rot_angle::T=zero(T)`: The rotation angle of the billiard.

# Returns
- An instance of the `ProsenBilliard` struct.
"""
function ProsenBilliard(a::T; x0=zero(T), y0=zero(T), rot_angle=zero(T)) :: ProsenBilliard where {T<:Real}
    # Create the quarter and full boundaries
    fundamental_boundary, corners = make_quarter_prosen(a; x0=x0, y0=y0, rot_angle=rot_angle)
    full_boundary, _, area_full = make_full_prosen(a; x0=x0, y0=y0, rot_angle=rot_angle)
    area_fundamental = area_full * 0.25
    length = sum([crv.length for crv in full_boundary])
    length_fundamental = sum([crv.length for crv in fundamental_boundary])
    return ProsenBilliard(fundamental_boundary, full_boundary, length, length_fundamental, a, corners, area_full, area_fundamental)
end



"""
    make_prosen_and_basis(a::T; x0=zero(T), y0=zero(T), rot_angle=zero(T)) :: Tuple{ProsenBilliard{T}, QuantumBilliards.RealPlaneWaves} where {T<:Real}

Constructs a Prosen billiard and the corresponding symmetry-adapted basis.

# Arguments
- `a::T`: The deformation parameter.
- `x0::T=zero(T)`: The x-coordinate of the center of the billiard.
- `y0::T=zero(T)`: The y-coordinate of the center of the billiard.
- `rot_angle::T=zero(T)`: The rotation angle of the billiard.

# Returns
- A tuple containing:
  - `prosen_billiard::ProsenBilliard`: The constructed Prosen billiard.
  - `basis::QuantumBilliards.RealPlaneWaves`: The symmetry-adapted basis.
"""
function make_prosen_and_basis(a::T; x0=zero(T), y0=zero(T), rot_angle=zero(T)) :: Tuple{ProsenBilliard{T}, RealPlaneWaves} where {T<:Real}
    prosen_billiard = ProsenBilliard(a; x0=x0, y0=y0, rot_angle=rot_angle)
    symmetry = Vector{Any}([XYReflection(-1, -1)])
    basis = RealPlaneWaves(10, symmetry; angle_arc=Float64(pi/2))
    return prosen_billiard, basis
end










