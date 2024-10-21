using LsqFit, QuadGK

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
    H = H ./ sum(H)  # normalize H
    non_zero_idxs = findall(H .> 0)  # Find indices of non-zero elements
    Im = sum(H[non_zero_idxs] .* log.(H[non_zero_idxs]))  # Compute the entropy for non-zero elements
    A = 1.0 / classical_phase_space_vol * exp(-Im)
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

"""
    P_localization_entropy_pdf_data(Hs::Vector{Matrix{T}}, classical_phase_space_vol::T; nbins=50) where {T<:Real}

Calculates the pdf data for plotting the localization entropy A using a histogram.

# Arguments
- `Hs::Vector{Matrix{T}}`: The Husimi matrices of the quantum eigenstates
- `classical_phase_space_vol::T`: The volume of the classical phase space
- `nbins=50`: The number of bins for the histogram

# Returns
- `bin_centers::Vector{T}`: The centers of the histogram bins
- `bin_counts::Vector{T}`: The counts of the histogram bins (normalized)
"""
function P_localization_entropy_pdf_data(Hs::Vector{Matrix{T}}, classical_phase_space_vol::T; nbins=50) where {T<:Real}
    localization_entropies = [localization_entropy(H, classical_phase_space_vol) for H in Hs]
    hist = Distributions.fit(StatsBase.Histogram, localization_entropies; nbins=nbins)
    bin_centers = (hist.edges[1][1:end-1] .+ hist.edges[1][2:end]) / 2
    bin_counts = hist.weights ./ sum(hist.weights) / diff(hist.edges[1])[1]
    return bin_centers, bin_counts
end

"""
    plot_P_localization_entropy_pdf(ax::Axis, Hs::Vector{Matrix{T}}, classical_phase_space_vol::T; nbins=50) where {T<:Real}

Plots the probability density function (PDF) of the localization entropy A using a histogram.

# Arguments
- `ax::Axis`: The Makie Axis object to plot on
- `Hs::Vector{Matrix{T}}`: The Husimi matrices of the quantum eigenstates
- `classical_phase_space_vol::T`: The volume of the classical phase space
- `nbins=50`: The number of bins for the histogram
- `color::Symbol=:lightblue`: The color of the histogram bars

# Returns
- `Nothing`
"""
function plot_P_localization_entropy_pdf!(ax::Axis, Hs::Vector, classical_phase_space_vol::T; nbins=50, color::Symbol=:lightblue) where {T<:Real}
    bin_centers, bin_counts = P_localization_entropy_pdf_data(Hs, classical_phase_space_vol; nbins=nbins)
    barplot!(ax, bin_centers, bin_counts, label="M distribution", color=color, gap=0, strokecolor=:black, strokewidth=1)
    xlims!(ax, (0.0, 1.0))
    axislegend(ax, position=:ct)
end

"""
    fit_P_localization_entropy_to_beta(Hs::Vector{Matrix{T}}, classical_phase_space_vol::T; nbins=50) :: LsqFitResult where {T<:Real}

Fits the beta distribution P(A) = C*A^a*(A0-A)^b to the numerical data.

# Arguments
- `Hs::Vector{Matrix{T}}`: The Husimi matrices of the quantum eigenstates
- `classical_phase_space_vol::T`: The volume of the classical phase space
- `nbins=50`: The number of bins for the histogram

# Returns
- `fit_result::LsqFitResult`: The result of the curve fitting using LsqFit. To get the `A0`, `a`, `b` take `fit_result.param`
"""
function fit_P_localization_entropy_to_beta(Hs::Vector{Matrix{T}}, classical_phase_space_vol::T; nbins=50) where {T<:Real}
    bin_centers, bin_counts = P_localization_entropy_pdf_data(Hs, classical_phase_space_vol; nbins=nbins)
    function model(A, p) # A scalar, p vector
        A0, a, b = p # unfold the param vector
        # Define the unnormalized function
        unnormalized_f(A) = A^a * (A0 - A)^b # assume a beta disitribution based on paper Batistić, Lozej, Robnik
        C, _ = quadgk(unnormalized_f, 0.0, A0) # use quadGK to get C
        return A .^ a .* (A0 .- A) .^ b ./ C # normalized model
    end
    initial_A0 = maximum(bin_centers) # the max val in As we have from data
    initial_params = [initial_A0, 10.0, 10.0] # just a guess
    fit_result = curve_fit(model, bin_centers, bin_counts, initial_params)
    return fit_result
end