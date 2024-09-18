using QuadGK

"""
    corner_correction(corner_angles::AbstractVector{<:Real}) -> Real

Calculates the corner correction term for Weyl's law based on the internal angles at the corners of the billiard.

# Arguments
- `corner_angles::AbstractVector{<:Real}`: A vector of internal angles (in radians) at each corner of the billiard.

# Returns
- `corner_term::Real`: The sum of the corner correction terms.

"""
corner_correction(corner_angles) =  sum([(pi^2 - c^2)/(24*pi*c) for c in corner_angles])

"""
    curvature_correction(billiard::Bi) -> Real where {Bi<:AbsBilliard}

Computes the curvature correction term for Weyl's law based on the curvature along the boundary of the billiard.

# Arguments
- `billiard::Bi`: An instance of a billiard (must be a subtype of `AbsBilliard`), containing the `full_boundary` attribute.

# Returns
- `curvature_term::Real`: The total curvature correction.
"""
function curvature_correction(billiard::Bi) where {Bi<:AbsBilliard}
    let segments = billiard.full_boundary
        curvat = 0.0
        for seg in segments 
            if seg isa PolarSegment
                curvat += quadgk(t -> curvature(seg, t), 0.0, 1.0)[1]
            end
            if seg isa CircleSegment
                curvat += 1/(12*pi)*(1/seg.radius)*seg.length
            end
        end
        return curvat
    end
end

"""
    weyl_law(ks::Vector{<:Real}, billiard::Bi; fundamental::Bool=true) -> Vector{Real}

Computes the leading-order Weyl's law term for the eigenvalue counting function based on the provided wave numbers.

# Arguments
- `ks::Vector{<:Real}`: The wave numbers for which Weyl's law is calculated.
- `billiard::Bi`: The billiard instance containing the area and length information.
- `fundamental::Bool=true`: If `true`, uses `area_fundamental` and `length_fundamental` of the billiard; if `false`, uses `area` and `length`.

# Returns
- `N(ks)::Vector{Real}`: A vector of estimated eigenvalue counts for each wave number in `ks`.

# Description
Calculates the leading-order approximation: N(k) = A/(4*pi)*k^2 - L/(4*pi)*k, using either the fundamental or full area and length.
"""
function weyl_law(ks::Vector, billiard::Bi; fundamental::Bool = true) where {Bi<:AbsBilliard}
    A = fundamental ? billiard.area_fundamental : billiard.area
    L = fundamental ? billiard.length_fundamental : billiard.length
    return @. (A * ks^2 - L * ks) / (4 * π)
end

"""
    weyl_law(ks::Vector{<:Real}, billiard::Bi, corner_angles::AbstractVector{<:Real}; fundamental::Bool=true) -> Vector{Real}

Computes Weyl's law for a set of wave numbers, including corner corrections.

# Arguments
- `ks::Vector{<:Real}`: The wave numbers for which Weyl's law is calculated.
- `billiard::Bi`: The billiard instance containing the area and length information.
- `corner_angles::AbstractVector{<:Real}`: The internal angles at the billiard's corners (in radians).
- `fundamental::Bool=true`: If `true`, uses `area_fundamental` and `length_fundamental`; otherwise, uses `area` and `length`.

# Returns
- `N(ks)::Vector{Real}`: A vector of estimated eigenvalue counts for each wave number in `ks`, with corner corrections applied.

# Description
Includes the corner correction term in the eigenvalue count estimate: N(k) = A/(4*pi)*k^2 - L/(4*pi)*k + corner_correction.
"""
function weyl_law(ks::Vector, billiard::Bi, corner_angles::AbstractVector{<:Real}; fundamental::Bool = true) where {Bi<:AbsBilliard}
    N = weyl_law(ks, billiard; fundamental=fundamental)
    N_correction = corner_correction(corner_angles)
    return N .+ N_correction
end

"""
    weyl_law(ks::Vector{<:Real}, billiard::Bi; fundamental::Bool=true) -> Vector{Real}

Computes Weyl's law for a set of wave numbers, including both corner and curvature corrections.

# Arguments
- `ks::Vector{<:Real}`: The wave numbers for which Weyl's law is calculated.
- `billiard::Bi`: The billiard instance containing the area, length, and curvature information.
- `fundamental::Bool=true`: If `true`, uses `area_fundamental` and `length_fundamental`; otherwise, uses `area` and `length`.

# Returns
- `N(ks)::Vector{Real}`: A vector of estimated eigenvalue counts for each wave number in `ks`, with both corner and curvature corrections applied.

# Description
This function adds both corner and curvature corrections to the Weyl's law estimation:
N(k) = A/(4*pi)*k^2 - L/(4*pi)*k + corner_correction + curvature_correction.
If the billiard has a field `angles`, corner corrections are applied automatically.
"""
function weyl_law(ks::Vector{<:Real}, billiard::Bi; fundamental::Bool = true) where {Bi<:AbsBilliard}
    A = fundamental ? billiard.area_fundamental : billiard.area
    L = fundamental ? billiard.length_fundamental : billiard.length
    N = @. (A * ks^2 - L * ks) / (4 * π)
    if hasfield(billiard, :angles)
        N .+= corner_correction(billiard.angles)
    end
    N .+= curvature_correction(billiard)
    return N
end

"""
    k_at_state(state::Int, billiard::Bi; fundamental::Bool=true) -> Real where {Bi<:AbsBilliard}

Calculates the wave number `k` corresponding to a given state number using the inverted Weyl's law with a potential curvature correction.

# Arguments
- `state::Int`: The eigenvalue index (state number).
- `billiard::Bi`: The billiard instance containing area and length information.
- `fundamental::Bool=true`: If `true`, uses `area_fundamental` and `length_fundamental`; otherwise, uses `area` and `length`.

# Returns
- `k::Real`: The estimated wave number corresponding to the given state.

# Description
Solves the quadratic equation: A/(4*pi)*k^2 - L/(4*pi)*k + curvature_correction - state = 0 to find `k`.
"""
function k_at_state(state::Int, billiard::Bi; fundamental::Bool=true) where {Bi<:AbsBilliard}
    A = fundamental ? billiard.area_fundamental : billiard.area
    L = fundamental ? billiard.length_fundamental : billiard.length
    a = A
    b = -L
    c = (curvature_correction(billiard) - state) * 4 * π
    dis = sqrt(b^2 - 4 * a * c)
    return (-b + dis) / (2 * a)
end

"""
    k_at_state(state::Int, billiard::Bi, corner_angles::AbstractVector{<:Real}; fundamental::Bool=true) -> Real where {Bi<:AbsBilliard}

Calculates the wave number `k` corresponding to a given state number using the inverted Weyl's law with corner corrections and potential curvature_correction.

# Arguments
- `state::Int`: The eigenvalue index (state number).
- `billiard::Bi`: The billiard instance containing area and length information.
- `corner_angles::AbstractVector{<:Real}`: Internal angles at the corners (in radians).
- `fundamental::Bool=true`: If `true`, uses `area_fundamental` and `length_fundamental`; otherwise, uses `area` and `length`.

# Returns
- `k::Real`: The estimated wave number.

# Description
Solves the quadratic equation: A/(4*pi)*k^2 - L/(4*pi)*k + (corner_correction + curvature_correction) - state = 0 to find `k`.
"""
function k_at_state(state::Int, billiard::Bi, corner_angles::AbstractVector{<:Real}; fundamental::Bool=true) where {Bi<:AbsBilliard}
    A = fundamental ? billiard.area_fundamental : billiard.area
    L = fundamental ? billiard.length_fundamental : billiard.length
    a = A
    b = -L
    c = (corner_correction(corner_angles) + curvature_correction(billiard) - state) * 4 * π
    dis = sqrt(b^2 - 4 * a * c)
    return (-b + dis) / (2 * a)
end