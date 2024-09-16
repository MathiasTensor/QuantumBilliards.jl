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
    weyl_law(k::Real, A::Real, L::Real) -> Real

Computes the leading-order Weyl's law term for the eigenvalue counting function.

# Arguments
- `k::Real`: The wave number.
- `A::Real`: The area of the billiard domain.
- `L::Real`: The perimeter (length of the boundary) of the billiard.

# Returns
- `N(k)::Real`: The estimated number of eigenvalues less than `k`.

# Description
Calculates the leading-order approximation: N(k) = A/(4*pi)*k^2 - L/(4*pi)*k.
"""
weyl_law(k,A,L) =  @. (A * k^2 - L * k)/(4*pi)

"""
    weyl_law(k::Real, A::Real, L::Real, corner_angles::AbstractVector{<:Real}) -> Real

Computes Weyl's law including corner corrections.

# Arguments
- `k::Real`: The wave number.
- `A::Real`: The area of the billiard domain.
- `L::Real`: The perimeter of the billiard.
- `corner_angles::AbstractVector{<:Real}`: Internal angles at the corners (in radians).

# Returns
- `N(k)::Real`: The estimated number of eigenvalues less than `k`, including corner corrections.

# Description
Includes the corner correction term: N(k) = A/(4*pi)*k^2 - L/(4*pi)*k + corner_correction.
"""
weyl_law(k,A,L,corner_angles) =  weyl_law(k,A,L) .+ corner_correction(corner_angles)

"""
    weyl_law(k::Real, A::Real, L::Real, corner_angles::AbstractVector{<:Real}, billiard::Bi) -> Real where {Bi<:AbsBilliard}

Computes Weyl's law including corner and curvature corrections.

# Arguments
- `k::Real`: The wave number.
- `A::Real`: The area of the billiard domain.
- `L::Real`: The perimeter of the billiard.
- `corner_angles::AbstractVector{<:Real}`: Internal angles at the corners (in radians).
- `billiard::Bi`: The billiard instance for curvature corrections.

# Returns
- `N(k)::Real`: The estimated number of eigenvalues less than `k`, including all corrections.

# Description
Adds both corner and curvature correction terms: N(k) = A/(4*pi)*k^2 - L/(4*pi)*k + corner_correction + curvature_correction.
"""
weyl_law(k,A,L,corner_angles,billiard) =  weyl_law(k,A,L) .+ corner_correction(corner_angles) .+ curvature_correction(billiard)

"""
    k_at_state(state::Int, A::Real, L::Real) -> Real

Calculates the wave number `k` corresponding to a given state number using the inverted Weyl's law without corrections.

# Arguments
- `state::Int`: The eigenvalue index (state number).
- `A::Real`: The area of the billiard domain.
- `L::Real`: The perimeter of the billiard.

# Returns
- `k::Real`: The estimated wave number corresponding to the given state.

# Description
Solves the quadratic equation: A/(4*pi)*k^2 - L/(4*pi)*k - state = 0 to find `k`.
"""
function k_at_state(state, A, L)
    a = A
    b = -L
    c = -state*4*pi
    dis = sqrt(b^2-4*a*c)
    return (-b+dis)/(2*a)
end

"""
    k_at_state(state::Int, A::Real, L::Real, corner_angles::AbstractVector{<:Real}) -> Real

Calculates the wave number `k` corresponding to a given state number using the inverted Weyl's law with corner corrections.

# Arguments
- `state::Int`: The eigenvalue index (state number).
- `A::Real`: The area of the billiard domain.
- `L::Real`: The perimeter of the billiard.
- `corner_angles::AbstractVector{<:Real}`: Internal angles at the corners.

# Returns
- `k::Real`: The estimated wave number.

# Description
Solves the quadratic equation: A/(4*pi)*k^2 - L/(4*pi)*k + corner_correction - state = 0 to find `k`.
"""
function k_at_state(state, A, L, corner_angles)
    a = A
    b = -L
    c = (corner_correction(corner_angles)-state)*4*pi 
    dis = sqrt(b^2-4*a*c)
    return (-b+dis)/(2*a)
end

"""
    k_at_state(state::Int, A::Real, L::Real, corner_angles::AbstractVector{<:Real}, billiard::Bi) -> Real where {Bi<:AbsBilliard}

Calculates the wave number `k` corresponding to a given state number using the inverted Weyl's law with both corner and curvature corrections.

# Arguments
- `state::Int`: The eigenvalue index (state number).
- `A::Real`: The area of the billiard domain.
- `L::Real`: The perimeter of the billiard.
- `corner_angles::AbstractVector{<:Real}`: Internal angles at the corners.
- `billiard::Bi`: The billiard instance for curvature corrections.

# Returns
- `k::Real`: The estimated wave number.

# Description
Solves the quadratic equation: A/(4*pi)*k^2 - L/(4*pi)*k + (corner_correction + curvature_correction) - state = 0 to find `k`.
"""
function k_at_state(state, A, L, corner_angles, billiard)
    a = A
    b = -L
    c = (curvature_correction(billiard) + corner_correction(corner_angles)-state)*4*pi 
    dis = sqrt(b^2-4*a*c)
    return (-b+dis)/(2*a)
end