using Makie

"""
INTERNAL FOR PLOTTING. Uses the def of σ² = (mean of squares) - (mean)^2 to determine the NV of a vector of unfolded energies in the energy window L
"""
function number_variance(E::Vector{T}, L::T) where {T<:Real}
    # By Tomaž Prosen
    Ave1 = zero(T)
    Ave2 = zero(T)
    j = 2
    k = 2
    x = E[1]
    N = length(E)
    largest_energy = E[end - min(Int(ceil(L) + 10), N-1)]  # Ensure largest_energy is within bounds

    while x < largest_energy
        while k < N && E[k] < x + L  # Ensure k does not exceed bounds with k<N = num of energies
            k += 1
        end

        # Adjusting the interval so that it slides/moves across the energy spectrum, a moving window statistic
        
        d1 = E[j] - x # The difference between the start of interval x and the first energy in the interval E[j]. This difference indicates how much the interval [x, E[j]] deviates from the exact interval [x, x + L]. d1 > 0
        d2 = E[k] - (x + L) # Difference between the end of the interval x + L and the first energy level beyond the interval [E[k]]. This difference shows how far the last energy level included in the interval is from the exact boundary x + L. d2 > 0
        cn = k - j # The number of energy levels in the interval (num of indexes for energies in the interval)
        
        # Interval Adjustment (d1 < d2):
        if d1 < d2
            # If the difference d1 (between x and E[j]) is smaller than d2 (between E[k] and x + L), the interval is adjusted by moving x to the next energy level E[j].
            x = E[j]
            # Since the difference between x and E[j] is smaller set the shift of the interval for x
            s = d1
            # Since d1 was smaller the updated x was for the start o the interval associated with the j index, so +1 it
            j += 1
        else
            # d2 was smaller so the new x is updated at the back of the interval
            x = E[k] - L
            # Analogues to the up case, just the smaller shift stored was d2
            s = d2
            # k is associated with the end interval
            k += 1
        end

        # Accumulations:
        # Ave1 = (1 / Total Length) * Σ(s_i * n(x_i, L)) where the number of energies in the interval n(x_i, L) is given by cn=k-j
        # Ave2 = (1 / Total Length) * Σ(s_i * n(x_i, L)) where the number of energies in the interval n(x_i, L) is given by cn=k-j
        Ave1 += s * cn
        Ave2 += s * cn^2

        # Ensure j does not exceed bounds
        if j >= N || k >= N # This condition checks if either j or k have reached or exceeded the total number of energy levels (N), j >= N || k >= N: If either index exceeds the bounds of the array, the loop is terminated to prevent out-of-bounds errors.
            break
        end
    end

    # See the formula at accumulations for reasons.
    total_length = largest_energy - E[1]
    Ave1 /= total_length
    Ave2 /= total_length

    # Calculate the variance σ² using the accumulated values Ave1 and Ave2. 
    # Variance = mean of squares - (mean)^2
    AveSig = Ave2 - Ave1^2
    return AveSig
end



"""
    probability_berry_robnik(s::T, rho::T) -> T where {T <: Real}

Computes the Berry-Robnik distribution for a given spacing `s` and mixing parameter `rho`.

# Arguments
- `s::T`: The spacing value (must be of a real number type).
- `rho::T`: The "mixing" parameter (0 < rho < 1), also of a real number type. For rho = 1, we have Poisson, and for rho = 0, we have the Wigner (GOE) distribution.

# Returns
- The value of the Berry-Robnik distribution at spacing `s` for the given `rho`.
"""
function probability_berry_robnik(s::T, rho::T) :: T where {T <: Real}
    rho1 = rho
    rho2 = one(T) - rho 
    term1 = (rho1^2) * erfc(sqrt(π / T(2)) * rho2 * s)
    term2 = (T(2) * rho1 * rho2 + (π / T(2)) * (rho2^3) * s) * exp(-(π / T(4)) * (rho2^2) * s^2)
    result = (term1 + term2) * exp(-rho1 * s)
    return result
end



"""
    cumulative_berry_robnik(s::T, rho::T) -> T where {T <: Real}

Computes the cumulative Berry-Robnik distribution function (CDF) for a given spacing `s` and mixing parameter `rho`.
The CDF is obtained by integrating the PDF from 0 to `s`.

# Arguments
- `s::T`: The spacing value (must be of a real number type).
- `rho::T`: The "mixing" parameter (0 < rho < 1).

# Returns
- The cumulative probability for the Berry-Robnik distribution at spacing `s`.
"""
function cumulative_berry_robnik(s::T, rho::T) :: T where {T <: Real}
    # Use quadgk to integrate the Berry-Robnik PDF from 0 to s
    result, _ = quadgk(x -> probability_berry_robnik(x, rho), T(0), s, rtol=1e-9, atol=1e-12)
    return result
end



"""
    compare_level_count_to_weyl(arr::Vector{T}, billiard::Bi; fundamental::Bool=true) where {T<:Real, Bi<:AbsBilliard}

Compares the numerical level count of energy levels with Weyl's law prediction for a given billiard geometry, and plots the comparison.

# Arguments
- `arr::Vector{T}`: A vector of energy levels.
- `billiard::Bi`: The billiard geometry, which must be a subtype of `AbsBilliard`. This provides the area, perimeter, and other characteristics of the billiard table.
- `fundamental::Bool=true`: Whether to use the fundamental domain of the billiard. If `true`, it uses the fundamental area and length, otherwise the full geometry is used. Defaults to `true`.

"""
function compare_level_count_to_weyl(arr::Vector{T}, billiard::Bi; fundamental::Bool=true) where {T<:Real, Bi<:AbsBilliard}
    A = T(fundamental ? billiard.area_fundamental : billiard.area)
    L = T(fundamental ? billiard.length_fundamental : billiard.length)
    C = T(curvature_and_corner_corrections(billiard; fundamental=fundamental))
    ys = [count(_k -> _k < k, arr) for k in arr]
    Ns = [(A/(4*pi)*k^2 - L/(4*pi)*k + C) for k in arr]
    # Create the plot
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="k", ylabel="N(k)", title="Level Count vs. Weyl's Law")
    scatter!(ax, arr, ys, label="Numerical", color=:blue, marker=:circle, markersize=3)
    lines!(ax, arr, Ns, label="Weyl", color=:red)
    axislegend(ax)
    return fig
end

"""
    count_levels_and_avg_fluctuation_in_interval(arr::Vector{T}, billiard::Bi; fundamental::Bool=true) -> Bool

This function computes the average fluctuation of the empirical number of energy levels in a given array `arr` relative to the theoretical number of levels predicted by Weyl's law for a given `billiard` system. It returns `false` if the average fluctuation exceeds ±0.1% of the theoretical number of levels in the interval; otherwise, it returns `true`.

### Arguments:
- `arr::Vector{T}`: A vector of wave numbers of energy levels.
- `billiard::Bi`: A billiard object of type <: AbsBilliard.
- `fundamental::Bool=true`: A boolean flag indicating whether to use the fundamental domain's geometrical data (`true`) or the full billiard system (`false`).

### Returns:
- `Bool`: Returns `true` if the average fluctuation is within the allowed range (±0.1% of the theoretical number of levels), otherwise returns `false`.
"""
function count_levels_and_avg_fluctuation_in_interval(arr::Vector{T}, billiard::Bi; fundamental::Bool=true) where {T<:Real, Bi<:AbsBilliard}
    # geometrical data
    A = T(fundamental ? billiard.area_fundamental : billiard.area)
    L = T(fundamental ? billiard.length_fundamental : billiard.length)
    C = T(curvature_and_corner_corrections(billiard; fundamental=fundamental))
    # empirical data and weyl data
    ys = [count(_k -> _k < k, arr) for k in arr]
    Ns = [(A/(4*pi)*k^2 - L/(4*pi)*k + C) for k in arr]
    new_ys = [y - N for (y, N) in zip(ys, Ns)]
    # Averaging over the entire segment
    avg_new_y = T(mean(new_ys))  # Calculate the mean fluctuation over the whole range
    # Calculate the fluctuation thresholds (±0.1% of the theoretical levels)
    fluctuation_threshold = 0.001 * (weyl_law(arr[end], billiard; fundamental=fundamental) - weyl_law(arr[1], billiard; fundamental=fundamental)) # Number of levels in the given interval
    # Check if the average fluctuation is within ±0.1% of the theoretical number of levels
    if avg_new_y > fluctuation_threshold || avg_new_y < -fluctuation_threshold
        return false  # The average fluctuation exceeds the allowed range
    else
        return true 
    end
end



"""
    plot_nnls(unfolded_energies::Vector{T}; nbins::Int=200, rho::Union{Nothing, T}=nothing) where {T <: Real}

Plots the nearest-neighbor level spacing (NNLS) distribution from unfolded energy levels, along with theoretical distributions (Poisson, GOE, GUE). Optionally, the Berry-Robnik distribution can also be included if a `rho` value is provided.

# Arguments
- `unfolded_energies::Vector{T}`: A vector of unfolded energy eigenvalues.
- `nbins::Int=200`: The number of bins for the histogram of spacings. Defaults to `200`.
- `rho::Union{Nothing, T}=nothing`: The Berry-Robnik parameter. If provided, the Berry-Robnik distribution is plotted. If set to `nothing`, the Berry-Robnik distribution is excluded.

# Returns
- A `Figure` object containing the NNLS distribution plot, showing the empirical histogram and theoretical curves (Poisson, GOE, GUE). The Berry-Robnik curve is added if `rho` is provided.

"""
function plot_nnls(unfolded_energies::Vector{T}; nbins::Int=200, rho::Union{Nothing, T}=nothing) where {T <: Real}
    # Compute nearest neighbor spacings
    spacings = diff(unfolded_energies)
    # Create a normalized histogram
    hist = Distributions.fit(StatsBase.Histogram, spacings; nbins=nbins)
    bin_centers = (hist.edges[1][1:end-1] .+ hist.edges[1][2:end]) / 2
    bin_counts = hist.weights ./ sum(hist.weights) / diff(hist.edges[1])[1]
    # Theoretical distributions
    poisson_pdf = x -> exp(-x)
    goe_pdf = x -> (π / T(2)) * x * exp(-π * x^2 / T(4))
    gue_pdf = x -> (T(32) / (π^2)) * x^2 * exp(-T(4) * x^2 / π)
    # Optionally include Berry-Robnik distribution if rho is provided
    berry_robnik_pdf = rho !== nothing ? (x -> probability_berry_robnik(x, rho)) : nothing
    fig = Figure(resolution=(800, 600))
    ax = Axis(fig[1, 1], title="NNLS", xlabel="Spacing (s)", ylabel="Probability Density")
    scatter!(ax, bin_centers, bin_counts, label="Empirical", color=:black, marker=:cross, markersize=10)
    s_values = range(0, stop=maximum(bin_centers), length=1000)
    lines!(ax, s_values, poisson_pdf.(s_values), label="Poisson", color=:blue, linestyle=:dash, linewidth=1)
    lines!(ax, s_values, goe_pdf.(s_values), label="GOE", color=:green, linestyle=:dot, linewidth=1)
    lines!(ax, s_values, gue_pdf.(s_values), label="GUE", color=:red, linestyle=:dashdot, linewidth=1)
    if berry_robnik_pdf !== nothing
        lines!(ax, s_values, berry_robnik_pdf.(s_values), label="Berry-Robnik, rho=$(round(rho; sigdigits=5))", color=:black, linestyle=:solid, linewidth=1)
    end
    xlims!(ax, extrema(s_values))
    axislegend(ax, position=:rt)

    return fig
end

function plot_brody_fit(unfolded_energies:Vector{T}) where {T<:Real}
    # Compute nearest neighbor spacings
    spacings = diff(unfolded_energies)
    # Create a normalized histogram
    hist = Distributions.fit(StatsBase.Histogram, spacings; nbins=nbins)
    bin_centers = (hist.edges[1][1:end-1] .+ hist.edges[1][2:end]) / 2
    bin_counts = hist.weights ./ sum(hist.weights) / diff(hist.edges[1])[1]
    # Theoretical form
    
end


"""
    plot_cumulative_spacing_distribution(unfolded_energy_eigenvalues::Vector{T}; rho::Union{Nothing, T}=nothing) where {T <: Real}

Plots the cumulative distribution function (CDF) of the nearest-neighbor level spacings (NNLS) for unfolded energy eigenvalues. Optionally, the Berry-Robnik CDF can be plotted if a `rho` value is provided.

# Arguments
- `unfolded_energy_eigenvalues::Vector{T}`: A vector of unfolded energy eigenvalues.
- `rho::Union{Nothing, T}=nothing`: The mixing parameter for the Berry-Robnik distribution. If `nothing`, the Berry-Robnik CDF is not plotted. Defaults to `nothing`.
- `plot_GUE::Bool=false`: Whether to plot the GUE curve. Defaults to `false`.

"""
function plot_cumulative_spacing_distribution(unfolded_energy_eigenvalues::Vector{T}; rho::Union{Nothing, T}=nothing, plot_GUE=false) where {T <: Real}
    # Compute nearest neighbor spacings and sort them
    spacings = diff(sort(unfolded_energy_eigenvalues))
    sorted_spacings = sort(spacings)
    N = length(sorted_spacings)
    # Compute the empirical CDF
    empirical_cdf = [i / N for i in 1:N]
    # Helper functions for theoretical CDFs
    poisson_cdf = s -> 1 - exp(-s)
    goe_cdf = s -> 1 - exp(-π * s^2 / 4)
    gue_cdf = s -> 1 - exp(-4 * s^2 / π) * (1 + 4 * s^2 / π)
    # If `rho` is provided, define the Berry-Robnik CDF with (s, rho)
    berry_robnik_cdf = (s, rho) -> cumulative_berry_robnik(s, rho)
    # Compute the theoretical CDFs
    num_points = 1000
    max_s = maximum(sorted_spacings)
    s_values = range(0, stop=max_s, length=num_points)
    poisson_cdf_values = poisson_cdf.(s_values)
    goe_cdf_values = goe_cdf.(s_values)
    gue_cdf_values = gue_cdf.(s_values)
    # Compute Berry-Robnik CDF values if `rho` is provided
    berry_robnik_cdf_values = rho !== nothing ? [berry_robnik_cdf(s, rho) for s in s_values] : nothing
    fig = Figure(resolution = (800, 600))
    ax = Axis(fig[1, 1], xlabel="Spacing (s)", ylabel="Cumulative Probability", title="Cumulative Distribution of Nearest Neighbor Spacings")
    scatter!(ax, sorted_spacings, empirical_cdf, label="Empirical CDF", color=:blue, markersize=2)
    lines!(ax, s_values, poisson_cdf_values, label="Poisson CDF", color=:red, linewidth=1, linestyle=:dot)
    lines!(ax, s_values, goe_cdf_values, label="GOE CDF", color=:green, linewidth=1, linestyle=:dot)
    if plot_GUE
        lines!(ax, s_values, gue_cdf_values, label="GUE CDF", color=:purple, linewidth=1, linestyle=:dot)
    end
    # Plot the Berry-Robnik CDF if `rho` is provided
    if berry_robnik_cdf_values !== nothing
        lines!(ax, s_values, berry_robnik_cdf_values, label="Berry-Robnik CDF", color=:black, linewidth=1)
    end
    axislegend(ax)
    return fig
end



"""
    plot_subtract_level_counts_from_weyl(arr::Vector{T}, billiard::Bi; bin_size::T = T(20.0), fundamental::Bool=true) where {Bi<:AbsBilliard, T<:Real}

Plots the difference between the empirical level count `N_count(k)` and the Weyl law prediction `N_weyl(k)` for a given billiard, incorporating curvature and corner corrections. Additionally, it computes and plots the averaged differences over specified intervals.

# Arguments
- `arr::Vector{T}`: A vector of `k` values. These represent the wavenumbers or eigenvalues.
- `billiard::Bi`: An instance of a subtype of `AbsBilliard`, representing the billiard's geometric configuration.
- `bin_size::T`: The size of the binning interval for averaging the differences between the empirical level count and Weyl's law prediction. Defaults to `20.0` (or the appropriate type `T`).
- `fundamental::Bool=true`: Whether to use the area and length of the fundamental region of the billiard (`true`) or the full billiard (`false`). Defaults to `true`.

# Returns
- A `Figure` object that plots:
    1. The difference `N_count(k) - N_weyl(k)` for each `k` value as a scatter plot.
    2. The averaged difference over intervals of size `bin_size` as a line plot.

"""
function plot_subtract_level_counts_from_weyl(arr::Vector{T}, billiard::Bi; bin_size::T = T(20.0), fundamental::Bool=true) where {Bi<:AbsBilliard, T<:Real} 
    A = T(fundamental ? billiard.area_fundamental : billiard.area)
    L = T(fundamental ? billiard.length_fundamental : billiard.length)
    C = T(curvature_and_corner_corrections(billiard; fundamental=fundamental))
    
    # Standard Weyl procedure
    ys = [count(_k -> _k < k, arr) for k in arr]
    Ns = [(A/(4*pi)*k^2 - L/(4*pi)*k + C) for k in arr]
    new_ys = [y - N for (y, N) in zip(ys, Ns)]

    # Binning Algorithm
    # Determine the range and bin size
    min_k, max_k = minimum(arr), maximum(arr)
    bins = collect(min_k:bin_size:max_k)  # Create intervals
    
    # To store the results
    bin_centers = T[]
    averaged_new_ys = T[]
    
    # Iterate over each bin
    for i in 1:(length(bins)-1)
        bin_start, bin_end = bins[i], bins[i+1]
        # Find the indices of `arr` that fall within this bin
        indices_in_bin = findall(k -> bin_start <= k < bin_end, arr)
        
        # If there are values in this bin, calculate the average new_ys
        if !isempty(indices_in_bin)
            avg_new_y = T(mean(new_ys[indices_in_bin]))
            push!(averaged_new_ys, avg_new_y)
            # Store the midpoint of the bin as the x-coordinate
            push!(bin_centers, (bin_start + bin_end) / 2)
        end
    end

    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="k", ylabel="N_count(k) - N_weyl(k)", title="Level Count - Weyl's Law")
    # Scatter plot for the numerical data
    scatter!(ax, arr, new_ys, label="Numerical", color=:blue, marker=:circle, markersize=2)
    lines!(ax, bin_centers, averaged_new_ys, color=:red, label="Averaged over k-interval $(bin_size)", linewidth=2)
    axislegend(ax)
    return fig
end

#=
"""
    plot_spectral_rigidity!(arr::Vector{T}, L_min::T, L_max::T; N::Int=100) where {T<:Real}

Plots the spectral rigidity (Δ₃) for an array of energy levels and compares it to the theoretical predictions for Poisson, GOE, and GUE statistics.

# Arguments
- `arr::Vector{T}`: A vector of energy levels.
- `L_min::T`: The minimum value of `L` for which to compute the spectral rigidity.
- `L_max::T`: The maximum value of `L` for which to compute the spectral rigidity.
- `N::Int=100`: The number of points for which to compute the spectral rigidity between `L_min` and `L_max`. Defaults to `100`.

"""
function plot_spectral_rigidity!(arr::Vector{T}, L_min::T, L_max::T; N::Int=100) where {T<:Real}
    Ls = range(L_min, L_max, length=N)
    Δ3_values = [spectral_rigidity(arr, L) for L in Ls]
    # Theoretical Spectral Rigidities
    poissonDeltaL = L -> L / 15.0 
    goeDeltaL = L -> (1 / (π^2)) * (log(2π * L) + 0.57721566490153286060 - 5.0 / 4.0 - (π^2 / 8))
    gueDeltaL = L -> (1 / (2 * π^2)) * (log(2π * L) + 0.57721566490153286060 - 5.0 / 4.0)
    fig = Figure()
    ax = Axis(fig[1, 1])
    # Plot numerical spectral rigidity
    lines!(ax, Ls, Δ3_values, color=:black)
    lines!(ax, Ls, [poissonDeltaL(L) for L in Ls], label="Poisson", color=:blue, linestyle=:dash)
    lines!(ax, Ls, [goeDeltaL(L) for L in Ls], label="GOE", color=:green, linestyle=:dot)
    lines!(ax, Ls, [gueDeltaL(L) for L in Ls], label="GUE", color=:red, linestyle=:dashdot)
    ax.ylabel = L"Δ_\text{3}"
    ax.xlabel = "Spacing (s)"
    ax.title = "Spectral Rigidity"
end

=#