using QuadGK

"""
    corner_correction(corner_angles::AbstractVector{<:Real}) -> Real

Calculates the corner correction term for Weyl's law based on the internal angles at the corners of the billiard.

# Arguments
- `corner_angles::AbstractVector{<:Real}`: A vector of internal angles (in radians) at each corner of the billiard.

# Returns
- `corner_term::Real`: The sum of the corner correction terms.

"""
corner_correction(corner_angles) = isempty(corner_angles) ? 0.0 : sum([(pi^2 - c^2)/(24*pi*c) for c in corner_angles])

"""
    curvature_correction(billiard::Bi) -> Real where {Bi<:AbsBilliard}

Computes the curvature correction term for Weyl's law based on the curvature along the boundary of the billiard.

# Arguments
- `billiard::Bi`: An instance of a billiard (must be a subtype of `AbsBilliard`), containing the `full_boundary` attribute.

# Returns
- `curvature_term::Real`: The total curvature correction.
"""
function curvature_correction(billiard::Bi; fundamental::Bool=true) where {Bi<:AbsBilliard}
    let segments = fundamental ? billiard.fundamental_boundary : billiard.full_boundary
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
    weyl_law(ks::Vector, billiard::Bi; fundamental::Bool=true) -> Vector where {Bi<:AbsBilliard}

Computes the eigenvalue counting function `N(k)` using Weyl's law, with corrections for corners and curvature.

# Arguments
- `ks::Vector{Real}`: The wave numbers to evaluate.
- `billiard::Bi`: The billiard instance containing area, length, and angle information.
- `fundamental::Bool=true`: If `true`, uses `area_fundamental`, `length_fundamental`, and `angles_fundamental`; otherwise, uses full geometry properties.

# Returns
- `N(ks)::Vector`: The estimated number of eigenvalues less than or equal to each `k`, including corner and curvature corrections.
"""
function weyl_law(ks::Vector, billiard::Bi; fundamental::Bool=true) where {Bi<:AbsBilliard}
    A = fundamental ? billiard.area_fundamental : billiard.area
    L = fundamental ? billiard.length_fundamental : billiard.length
    angles = fundamental ? billiard.angles_fundamental : billiard.angles
    
    N_ks = (A * ks.^2 .- L .* ks) ./ (4π)
    N_ks .+= corner_correction(angles)
    N_ks .+= curvature_correction(billiard)

    return N_ks
end

"""
    k_at_state(state::Int, billiard::Bi; fundamental::Bool=true) -> Real where {Bi<:AbsBilliard}

Calculates the wave number `k` corresponding to a given state number using the inverted Weyl's law, including corner and curvature corrections.

# Arguments
- `state::Int`: The eigenvalue index (state number).
- `billiard::Bi`: The billiard instance containing area, length, and angle information.
- `fundamental::Bool=true`: If `true`, uses `area_fundamental`, `length_fundamental`, and `angles_fundamental`; otherwise, uses full geometry properties.

# Returns
- `k::Real`: The estimated wave number corresponding to the given state.
"""
function k_at_state(state::Int, billiard::Bi; fundamental::Bool=true) where {Bi<:AbsBilliard}
    A = fundamental ? billiard.area_fundamental : billiard.area
    L = fundamental ? billiard.length_fundamental : billiard.length
    angles = fundamental ? billiard.angles_fundamental : billiard.angles
    
    a = A
    b = -L
    c = -state * 4π
    
    c += corner_correction(angles) * 4π
    c += curvature_correction(billiard) * 4π
    
    dis = sqrt(b^2 - 4 * a * c)
    return (-b + dis) / (2 * a)
end