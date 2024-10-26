using LsqFit, QuadGK, SpecialFunctions

"""
    localization_entropy(H::Matrix{T}, chaotic_classical_phase_space_vol_fraction::T) where {T<:Real}

Calculates the localization entropy of a quantum eigenstate's Husimi matrix. It uses the convention 2πħ=1 like in Lozej, Batistić, Robnik. The same holds for the classical phase space volume for counting the number of the classical chaotic domain, so Nc=classical_phase_space_vol.

# Arguments
- `H::Matrix`: The Husimi matrix of the quantum eigenstate
- `chaotic_classical_phase_space_vol_fraction`: The fraction of the chaotic classical phase space (The fraction of chaotic grid cells for the PH Matrix).

# Returns
- `A<:Real`: The localization entropy A of the quantum eigenstate
"""
function localization_entropy(H::Matrix{T}, chaotic_classical_phase_space_vol_fraction::T) where {T<:Real}
    H = H ./ sum(H)  # normalize H
    non_zero_idxs = findall(H .> 0.0)  # Find indices of non-zero elements
    Im = sum(H[non_zero_idxs] .* log.(H[non_zero_idxs]))  # Compute the entropy for non-zero elements
    A = 1.0 / (T(prod(size(H)))*chaotic_classical_phase_space_vol_fraction) * exp(-Im)
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
    P_localization_entropy_pdf_data(Hs::Vector{Matrix{T}}, chaotic_classical_phase_space_vol_fraction::T; nbins=50) where {T<:Real}

Calculates the pdf data for plotting the localization entropy A using a histogram.

# Arguments
- `Hs::Vector{Matrix{T}}`: The Husimi matrices of the quantum eigenstates
- `chaotic_classical_phase_space_vol_fraction::T`: The fraction of the chaotic classical phase space (The fraction of chaotic grid cells for the PH Matrix).
- `nbins=50`: The number of bins for the histogram

# Returns
- `bin_centers::Vector{T}`: The centers of the histogram bins
- `bin_counts::Vector{T}`: The counts of the histogram bins (normalized)
"""
function P_localization_entropy_pdf_data(Hs::Vector{Matrix{T}}, chaotic_classical_phase_space_vol_fraction::T; nbins=50) where {T<:Real}
    localization_entropies = [localization_entropy(H, chaotic_classical_phase_space_vol_fraction) for H in Hs]
    hist = Distributions.fit(StatsBase.Histogram, localization_entropies; nbins=nbins)
    bin_centers = (hist.edges[1][1:end-1] .+ hist.edges[1][2:end]) / 2
    bin_counts = hist.weights ./ sum(hist.weights) / diff(hist.edges[1])[1]
    # Normalize bin_counts to make it a probability density function
    total_area = sum(bin_counts)
    bin_counts = bin_counts ./ total_area
    return bin_centers, bin_counts
end

"""
    plot_P_localization_entropy_pdf(ax::Axis, Hs::Vector{Matrix{T}}, chaotic_classical_phase_space_vol_fraction::T; nbins=50, color::Symbol=:lightblue, fit_beta::Bool=false) where {T<:Real}

Plots the probability density function (PDF) of the localization entropy A using a histogram.

# Arguments
- `ax::Axis`: The Makie Axis object to plot on
- `Hs::Vector{Matrix{T}}`: The Husimi matrices of the quantum eigenstates
- `chaotic_classical_phase_space_vol_fraction::T`: The fraction of the chaotic classical phase space (The fraction of chaotic grid cells for the PH Matrix).
- `nbins=50`: The number of bins for the histogram
- `color::Symbol=:lightblue`: The color of the histogram bars
- `fit_beta::Bool=false`: Whether to fit the beta distribution P(A) = C*A^a*(A0-A)^b to the numerical data (used only for close to ergodic systems otherwise it will obviously fail)

# Returns
- `Nothing`
"""
function plot_P_localization_entropy_pdf!(ax::Axis, Hs::Vector, chaotic_classical_phase_space_vol_fraction::T; nbins=50, color::Symbol=:lightblue, fit_beta::Bool=false) where {T<:Real}
    bin_centers, bin_counts = P_localization_entropy_pdf_data(Hs, chaotic_classical_phase_space_vol_fraction; nbins=nbins)
    barplot!(ax, bin_centers, bin_counts, label="A distribution", color=color, gap=0, strokecolor=:black, strokewidth=1)
    if fit_beta
        fit_data = fit_P_localization_entropy_to_beta(Hs, chaotic_classical_phase_space_vol_fraction, nbins=nbins)
        A0, a, b = fit_data.param
        # x scale for the beta distributin will be from 0.0 to A0
        xs = collect(range(0.0, A0, 200))
        ys = @. xs^a * (A0 - xs)^b # non-normalized
        param_label = "Beta dist.fit: [A0=$(round(A0, digits=2)), a=$(round(a, digits=2)), b=$(round(b, digits=2))]"
        lines!(ax,xs,ys,label=param_label,color=:red)
    end
    axislegend(ax, position=:rt)
    xlims!(ax, (0.0, 1.0))
end

"""
    fit_P_localization_entropy_to_beta(Hs::Vector{Matrix{T}}, chaotic_classical_phase_space_vol_fraction::T; nbins=50) :: LsqFitResult where {T<:Real}

Fits the beta distribution P(A) = C*A^a*(A0-A)^b to the numerical data.

# Arguments
- `Hs::Vector{Matrix{T}}`: The Husimi matrices of the quantum eigenstates
- `chaotic_classical_phase_space_vol_fraction::T`: The fraction of the chaotic classical phase space (The fraction of chaotic grid cells for the PH Matrix).
- `nbins=50`: The number of bins for the histogram

# Returns
- `fit_result::LsqFitResult`: The result of the curve fitting using LsqFit. To get the `A0`, `a`, `b` take `fit_result.param`. It is returned in that order.
"""
function fit_P_localization_entropy_to_beta(Hs::Vector, chaotic_classical_phase_space_vol_fraction::T; nbins=50) where {T<:Real}
    bin_centers, bin_counts = P_localization_entropy_pdf_data(Hs, chaotic_classical_phase_space_vol_fraction; nbins=nbins)
    #=
    function model(A, p) # A scalar, p vector
        A0, a, b = p # unfold the param vector
        # Define the unnormalized function
        unnormalized_f(A) = A^a * (A0 - A)^b # assume a beta disitribution based on paper Batistić, Lozej, Robnik
        #C, _ = quadgk(unnormalized_f, 0.0, A0) # use quadGK to get C, depreceated
        B(x,y)=gamma(x)*gamma(y)/gamma(x+y)
        C = (A^(a+b+1)*B(a+1,b+1))
        return A .^ a .* (A0 .- A) .^ b ./ C # normalized model
    end
    =#
    function model(A, p)
        A0, a, b = p
        epsilon = 1e-6  # Small offset to avoid domain issues
        # Normalization constant using the Beta function B(a+1, b+1)
        B(x, y) = gamma(x) * gamma(y) / gamma(x + y)
        C = real(A0)^(a + b + 1) * real(B(a + 1, b + 1))
        result = Vector{Float64}(undef, length(A))
        for i in eachindex(A)
            base1 = max(A[i] + epsilon, epsilon)
            base2 = max(A0 - A[i] + epsilon, epsilon)
            result[i] = real((base1)^a * (base2)^b) / C
        end
        return result
    end
    initial_A0 = maximum(bin_centers) # the max val in As we have from data
    println("initial_A0 = ", initial_A0)
    println("bin_centers = ", bin_centers)
    println("bin_counts: ", bin_counts)
    initial_params = [initial_A0, 10.0, 10.0] # just a guess
    fit_result = curve_fit(model, bin_centers, bin_counts, initial_params)
    return fit_result
end