

# We will use the convention of using the 2πħ=1 like in Lozej, Batistić, Robnik. The same holds for the classical phase space volume for counting the number of the classical chaotic domain, so Nc=classical_phase_space_vol
"""
    localization_entropy(H::Matrix{T}, classical_phase_space_vol::T) where {T<:Real}

Calculates the localization entropy of a quantum eigenstate's Husimi matrix. It uses the convention 2πħ=1 like in Lozej, Batistić, Robnik. The same holds for the classical phase space volume for counting the number of the classical chaotic domain, so Nc=classical_phase_space_vol.

# Arguments
- `H::Matrix`: The Husimi matrix of the quantum eigenstate
- `classical_phase_space_vol`: The volume of the classical phase space.

# Returns
- `A<:Real`: The localization entropy A of the quantum eigenstate
"""
function localization_entropy(H::Matrix{T}, classical_phase_space_vol::T) where {T<:Real}
    H = H ./ sum(H) # normalize H
    lnH = log.(H)
    Im = sum(broadcast(*, H, lnH))
    A = 1.0/classical_phase_space_vol*exp(-Im)
    return A
end

"""
    normalized_inverse_participation_ratio_R(H::Matrix) where {T<:Real}

Calculates the normalized inverse participation ratio R.

# Arguments
- `H::Matrix`: The Husimi matrix of the quantum eigenstate

# Returns
- `R<:Real`: The normalized inverse participation ratio R of the quantum eigenstate
"""
function normalized_inverse_participation_ratio_R(H::Matrix{T}) where {T<:Real}
    H = H ./ sum(H)
    R = 1/(prod(size(H))*sum(H.^2)) # the prod(size(H)) is the grid count directly from the size of the matrix
    return R
end

# try to fit the beta distribution
function P_localization_entropy(Hs::Vector{Matrix{T}}, classical_phase_space_vol::T) where {T<:Real}
    localization_entropies = [localization_entropy(H, classical_phase_space_vol) for H in Hs]
end