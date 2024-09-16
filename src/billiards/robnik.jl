






"""
    make_full_robnik(ε::T; x0=zero(T), y0=zero(T), rot_angle=zero(T)) where {T<:Real}

Constructs a full Robnik billiard.

# Arguments
- `ε::T`: The deformation parameter (ε ≥ 0).
- `x0::T=zero(T)`: The x-coordinate of the center of the billiard.
- `y0::T=zero(T)`: The y-coordinate of the center of the billiard.
- `rot_angle::T=zero(T)`: The rotation angle of the billiard.

# Returns
- A tuple containing:
  - `boundary::Vector{PolarSegment{T}}`: The boundary segments of the full Robnik billiard.
  - `corners::Vector{SVector{2,T}}`: An empty vector (no corners for the full billiard).
"""
function make_full_robnik(ε::T; x0=zero(T), y0=zero(T), rot_angle=zero(T)) where {T<:Real}
    if ε < zero(T)
        error("ε must be non-negative.")
    end

    origin = (x0, y0)
    θ_multiplier = 2.0  # t from 0 to 1 maps to θ from 0 to 2π

    # Define the radial function for the full Robnik billiard
    r_func = t -> begin
        θ = θ_multiplier * π * t  # θ ranges from 0 to 2π
        r = one(T) + ε * cos(θ)
        x = r * cos(θ)
        y = r * sin(θ)
        SVector(x, y)
    end

    # Create the full Robnik billiard segment
    full_robnik_segment = PolarSegment(r_func; origin=origin, rot_angle=rot_angle)

    boundary = [full_robnik_segment]
    corners = []  # Empty vector for corners

    return boundary, corners
end

"""
    make_half_robnik(ε::T; x0=zero(T), y0=zero(T), rot_angle=zero(T)) where {T<:Real}

Constructs a half Robnik billiard.

# Arguments
- `ε::T`: The deformation parameter (ε ≥ 0).
- `x0::T=zero(T)`: The x-coordinate of the center of the billiard.
- `y0::T=zero(T)`: The y-coordinate of the center of the billiard.
- `rot_angle::T=zero(T)`: The rotation angle of the billiard.

# Returns
- A tuple containing:
  - `boundary::Vector{Union{PolarSegment{T}, VirtualLineSegment{T}}}`: The boundary segments of the half Robnik billiard.
  - `corners::Vector{SVector{2,T}}`: The corner points of the half billiard.
"""
function make_half_robnik(ε::T; x0=zero(T), y0=zero(T), rot_angle=zero(T)) where {T<:Real}
    if ε < zero(T)
        error("ε must be non-negative.")
    end

    origin = SVector(x0, y0)
    θ_multiplier = 1.0  # t from 0 to 1 maps to θ from 0 to π

    # Define the radial function for the half Robnik billiard
    r_func = t -> begin
        θ = θ_multiplier * π * t  # θ ranges from 0 to π
        r = one(T) + ε * cos(θ)
        x = r * cos(θ)
        y = r * sin(θ)
        SVector(x, y)
    end

    # Create the half Robnik billiard segment
    half_robnik_segment = PolarSegment(r_func; origin=origin, rot_angle=rot_angle)

    # Compute the start and end points of the half segment
    pt0 = curve(half_robnik_segment, zero(T))  # Start point at θ = 0
    pt1 = curve(half_robnik_segment, one(T))   # End point at θ = π

    # Create a virtual line segment to close the half billiard from pt1 to pt0
    line_segment = VirtualLineSegment(pt1, pt0; origin=origin, rot_angle=rot_angle)

    # Construct the boundary
    boundary = Union{PolarSegment{T}, VirtualLineSegment{T}}[half_robnik_segment, line_segment]

    # Corners are pt0 and pt1
    corners = [pt0, pt1]

    return boundary, corners
end



"""
    struct RobnikBilliard{T} <: QuantumBilliards.AbsBilliard where {T<:Real}

Defines a Robnik billiard with a full and half boundary.

# Fields
- `fundamental_boundary::Vector`: The boundary segments of the half Robnik billiard.
- `full_boundary::Vector`: The boundary segments of the full Robnik billiard.
- `length::T`: The total length of the boundary.
- `area::T`: The total area of the Robnik billiard.
- `epsilon::T`: The deformation parameter ε.
- `corners::Vector{SVector{2,T}}`: The corner points of the billiard.
"""
struct RobnikBilliard{T} <: AbsBilliard where {T<:Real}
    fundamental_boundary::Vector
    full_boundary::Vector
    length::T
    epsilon::T
    corners::Vector{SVector{2,T}}
end




"""
    RobnikBilliard(ε::T; x0=zero(T), y0=zero(T), rot_angle=zero(T)) :: RobnikBilliard where {T<:Real}

Constructs a Robnik billiard.

# Arguments
- `ε::T`: The deformation parameter (ε ≥ 0).
- `x0::T=zero(T)`: The x-coordinate of the center of the billiard.
- `y0::T=zero(T)`: The y-coordinate of the center of the billiard.
- `rot_angle::T=zero(T)`: The rotation angle of the billiard.

# Returns
- An instance of the `RobnikBilliard` struct.
"""
function RobnikBilliard(ε::T; x0=zero(T), y0=zero(T), rot_angle=zero(T)) :: RobnikBilliard where {T<:Real}
    # Create the half and full boundaries
    fundamental_boundary, corners = make_half_robnik(ε; x0=x0, y0=y0, rot_angle=rot_angle)
    full_boundary, _ = make_full_robnik(ε; x0=x0, y0=y0, rot_angle=rot_angle)
    length = sum([crv.length for crv in full_boundary])
    return RobnikBilliard(fundamental_boundary, full_boundary, length, ε, corners)
end




"""
    make_robnik_and_basis(ε::T; x0=zero(T), y0=zero(T), rot_angle=zero(T)) :: Tuple{RobnikBilliard{T}, QuantumBilliards.RealPlaneWaves} where {T<:Real}

Constructs a Robnik billiard and the corresponding symmetry-adapted basis.

# Arguments
- `ε::T`: The deformation parameter (ε ≥ 0).
- `x0::T=zero(T)`: The x-coordinate of the center of the billiard.
- `y0::T=zero(T)`: The y-coordinate of the center of the billiard.
- `rot_angle::T=zero(T)`: The rotation angle of the billiard.

# Returns
- A tuple containing:
  - `robnik_billiard::RobnikBilliard`: The constructed Robnik billiard.
  - `basis::QuantumBilliards.RealPlaneWaves`: The symmetry-adapted basis.
"""
function make_robnik_and_basis(ε::T; x0=zero(T), y0=zero(T), rot_angle=zero(T)) :: Tuple{RobnikBilliard{T}, RealPlaneWaves} where {T<:Real}
    robnik_billiard = RobnikBilliard(ε; x0=x0, y0=y0, rot_angle=rot_angle)
    # Define symmetry operations if any (e.g., reflection symmetry)
    symmetry = Vector{Any}([XReflection(-1)])
    basis = RealPlaneWaves(10, symmetry; angle_arc=Float64(pi/2))
    return robnik_billiard, basis
end