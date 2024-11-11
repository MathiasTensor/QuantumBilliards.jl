using Makie, SpecialFunctions, ForwardDiff, LsqFit

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

# INTERNAL Second derivative of the joint gap probability between Poisson and Brody
function probability_berry_robnik_brody(s::T, rho::T, β::T) where {T<:Real}
    Γ_factor = gamma(1 + 1 / (1 + β))
    C2 = Γ_factor^(β + 1)
    T1 = (1 / Γ_factor) * exp(-rho * s)
    T2 = (1 / s) * exp((-1 + rho) * s * (real((Complex(s - rho * s))^β) * C2))
    # Separate out complex components to prevent negative powers leading to complex values
    inner_term = -((-1 + rho) * s * real((Complex(s - rho * s))^β) * C2)
    T3 = real(Complex(inner_term)^(1 / (1 + β)))
    T4 = 2 * rho - (1 + β) * (-1 + rho) * real((Complex(s - rho * s))^β) * C2
    T5 = (rho^2 * gamma(1 / (1 + abs(β)), abs(inner_term))) / (1 + β)
    return T1 * (T2 * T3 * T4 + T5)
end

"""
    cumulative_berry_robnik(s::T, rho::T) -> T where {T <: Real}

Computes the cumulative Berry-Robnik distribution function (CDF) for a given spacing `s` and mixing parameter `rho`.
The CDF is obtained analytically.

# Arguments
- `s::T`: The spacing value.
- `rho::T`: Liouville regular phase space portion.

# Returns
- The cumulative probability for the Berry-Robnik distribution at spacing `s`.
"""
function cumulative_berry_robnik(s::T, rho::T) :: T where {T <: Real}
    E_c(x) = erfc(sqrt(pi)*x/2.0)
    E(x) = exp(-rho*x)*E_c((1.0-rho)*x)
    #return ForwardDiff.derivative(x -> E(x), s) - ForwardDiff.derivative(x -> E(x), 0.0)
    function dE_dx(x::T, rho::T) :: T where {T <: Real}
        exp_term = exp(-rho * x)
        erfc_term = E_c((1.0 - rho) * x)
        inner_exp_term = exp(-π * (1.0 - rho)^2 * x^2 / 4.0)
        return -exp_term * (rho * erfc_term + (1.0 - rho) * inner_exp_term)
    end
    return dE_dx(s, rho) - dE_dx(0.0, rho)
end

"""
    cumulative_berry_robnik_brody(s::T, rho::T, β::T) where {T<:Real}

Comuptes the cumulative Berry-Robnik_Brody cumulative distribution. The expression used is based on the analytical expression from Mathematica.

# Arguments
- `s::T`: The spacing value.
- `rho::T`: Liouville regular phase space portion.
- `β::T`: Brody exponent.

# Returns
- The cumulative probability for the Berry-Robnik_Brody distribution at spacing `s`.
"""
function cumulative_berry_robnik_brody(s::T, rho::T, β::T) where {T<:Real}
    function E_joint_derivative(s, rho, β) 
        Γ_factor = gamma((β + 2) / (β + 1))
        term1 = -exp(-rho * s - real(Complex((1 - rho) * s)^(1 + β)) * Γ_factor^(1 + β))
        term2 = (1 - rho) * real(Complex((1 - rho) * s)^β) * Γ_factor^β
        term3 = (real(Complex((1 - rho) * s)^(1+β)) * Γ_factor^(1+β))^(-1+1/(1+β))
        term4 = exp(-rho*s) * rho * gamma(1/(1+β)) * gamma_inc(1 / (1 + abs(β)), real(Complex((1 - abs(rho)) * s)^(1 + β)) * Γ_factor^(1 + β))[2] / ((1+β)*Γ_factor)
        return term1 * term2 * term3 - term4
    end
    small_offset = 1e-10
    return E_joint_derivative(s, rho, β) - E_joint_derivative(small_offset, rho, β)
end

# INTERNAL U transformation
function U(ws::Vector)
    return @. 2/pi*acos(sqrt(1-ws))
end


# THIS ONE IS NOT OK, HAS CONVERGENCE PROBLEMS
"""
    cumulative_berry_robnik_numerical_integration(s::T, rho::T) -> T where {T <: Real}

Computes the cumulative Berry-Robnik distribution function (CDF) for a given spacing `s` and mixing parameter `rho`.
The CDF is obtained by numerically integrating the PDF from 0 to `s`.

# Arguments
- `s::T`: The spacing value (must be of a real number type).
- `rho::T`: The "mixing" parameter (0 < rho < 1).

# Returns
- The cumulative probability for the Berry-Robnik distribution at spacing `s`.
"""
function cumulative_berry_robnik_numerical_integration(s::T, rho::T) :: T where {T <: Real}
    result, _ = quadgk_count(x -> probability_berry_robnik(x, rho), 0.0, s, rtol=1e-12, atol=1e-15, maxevals=1e7, order=21)
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

# INTERNAL, returns the optimal (ρ,β) parameters
function fit_brb_to_data(bin_centers::Vector, bin_counts::Vector, rho::T) where {T<:Real}
    function brb_model(s_vals::Vector, params)
        ρ, β = params
        return [probability_berry_robnik_brody(s,ρ,β) for s in s_vals]
    end
    init_params = [rho+0.1,1.0] # beta init 1.0
    fit_result = curve_fit((s_vals, params) -> brb_model(s_vals, params), bin_centers, bin_counts, init_params)
    return fit_result.param
end

function fit_brb_only_beta(bin_centers::Vector, bin_counts::Vector, rho::T) where {T<:Real}
    function brb_model(s_vals::Vector, params)
        β = params[1]
        return [probability_berry_robnik_brody(s,rho,β) for s in s_vals]
    end
    init_params = [1.0] # beta init 1.0
    fit_result = curve_fit((s_vals, params) -> brb_model(s_vals, params), bin_centers, bin_counts, init_params)
    return fit_result.param[1]
end

# INTERNAL, returns optimal (ρ,β) parameters for cumulative
function fit_brb_cumulative_to_data(s_values::Vector, ws::Vector, rho::T) where {T<:Real}
    function brb_cumul_model(s_vals::Vector, params)
        ρ, β = params
        return [cumulative_berry_robnik_brody(s,ρ,β) for s in s_vals]
    end
    init_params = [rho+0.1,1.0] # beta init 1.0, rho little bigger than theoretical regular phase space
    fit_result = curve_fit((s_vals, params) -> brb_cumul_model(s_vals, params), s_values, ws, init_params)
    return fit_result.param
end

function fit_brb_cumulative_to_data_only_beta(s_values::Vector, ws::Vector, rho::T) where {T<:Real}
    function brb_cumul_model(s_vals::Vector, params)
        β = params[1]
        return [cumulative_berry_robnik_brody(s,rho,β) for s in s_vals]
    end
    init_params = [1.0]
    fit_result = curve_fit((s_vals, params) -> brb_cumul_model(s_vals, params), s_values, ws, init_params)
    return fit_result.param[1]
end

"""
    plot_nnls(unfolded_energies::Vector{T}; nbins::Int=200, rho::Union{Nothing, T}=nothing, fit_brb::Bool=false) where {T <: Real}

Plots the nearest-neighbor level spacing (NNLS) distribution from unfolded energy levels, along with theoretical distributions (Poisson, GOE, GUE). Optionally, the Berry-Robnik distribution can also be included if a `rho` value is provided.

# Arguments
- `unfolded_energies::Vector{T}`: A vector of unfolded energy eigenvalues.
- `nbins::Int=200`: The number of bins for the histogram of spacings. Defaults to `200`.
- `rho::Union{Nothing, T}=nothing`: The Berry-Robnik parameter. If provided, the Berry-Robnik distribution is plotted. If set to `nothing`, the Berry-Robnik distribution is excluded.
- `fit_brb::Bool=false`: If the numerical data requires a fitting of the Berry-Robnik_Brody P(s) distribution, displaying the optimal beta and rho parameter in the legend.
- `fit_only_beta::Bool=false`: If `true`, only the Berry-Robnik-Brody distribution's β parameter is fitted to the data, ρ is as given initially.
- `log_scale::Bool=false`: If we plot log(P(s)) for observing small differences

# Returns
- A `Figure` object containing the NNLS distribution plot, showing the empirical histogram and theoretical curves (Poisson, GOE, GUE). The Berry-Robnik curve is added if `rho` is provided.

"""
function plot_nnls(unfolded_energies::Vector{T}; nbins::Int=200, rho::Union{Nothing, T}=nothing, fit_brb::Bool=false, fit_only_beta=false, log_scale=false, fited_rho::Union{Nothing, T} = nothing) where {T <: Real}
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
    if log_scale
        ax = Axis(fig[1, 1], title="NNLS", xlabel="Spacing (s)", ylabel="log10(P(s))")
    else
        ax = Axis(fig[1, 1], title="NNLS", xlabel="Spacing (s)", ylabel="P(s)")
    end
    if log_scale
        scatter!(ax, bin_centers, log10.(bin_counts), label="Empirical", color=:black, marker=:cross, markersize=10)
    else
        scatter!(ax, bin_centers, bin_counts, label="Empirical", color=:black, marker=:cross, markersize=10)
    end
    s_values = range(0, stop=maximum(bin_centers), length=1000)
    if log_scale
        lines!(ax, s_values, log10.(poisson_pdf.(abs.(s_values))), label="Poisson", color=:blue, linestyle=:dash, linewidth=1)
        lines!(ax, s_values, log10.(goe_pdf.(abs.(s_values))), label="GOE", color=:green, linestyle=:dot, linewidth=1)
        lines!(ax, s_values, log10.(gue_pdf.(abs.(s_values))), label="GUE", color=:red, linestyle=:dashdot, linewidth=1)
    else
        lines!(ax, s_values, poisson_pdf.(abs.(s_values)), label="Poisson", color=:blue, linestyle=:dash, linewidth=1)
        lines!(ax, s_values, goe_pdf.(abs.(s_values)), label="GOE", color=:green, linestyle=:dot, linewidth=1)
        lines!(ax, s_values, gue_pdf.(abs.(s_values)), label="GUE", color=:red, linestyle=:dashdot, linewidth=1)
    end
    
    if berry_robnik_pdf !== nothing
        if log_scale
            lines!(ax, s_values, log10.(abs.(berry_robnik_pdf.(abs.(s_values)))), label="Berry-Robnik, rho=$(round(rho; sigdigits=5))", color=:black, linestyle=:solid, linewidth=1)
        else
            lines!(ax, s_values, berry_robnik_pdf.(abs.(s_values)), label="Berry-Robnik, rho=$(round(rho; sigdigits=5))", color=:black, linestyle=:solid, linewidth=1)
        end
    end
    if fit_brb && !isnothing(rho)
        if fit_only_beta
            fited_rho = isnothing(fited_rho) ? rho : fited_rho
            β_opt = fit_brb_only_beta(collect(bin_centers), collect(bin_counts), fited_rho)
            brb_pdf = s -> probability_berry_robnik_brody(s, fited_rho, β_opt)
            if log_scale
                lines!(ax, s_values, log10.(abs.(brb_pdf.(abs.(s_values)))), label="Berry-Robnik-Brody, ρ_fit=$(round(fited_rho; sigdigits=5)), β_fit=$(round(β_opt; sigdigits=5))", color=:orange, linestyle=:solid, linewidth=1)
            else
                lines!(ax, s_values, brb_pdf.(abs.(s_values)), label="Berry-Robnik-Brody, ρ_fit=$(round(fited_rho; sigdigits=5)), β_fit=$(round(β_opt; sigdigits=5))", color=:orange, linestyle=:solid, linewidth=1)
            end
        else
            ρ_opt, β_opt = fit_brb_to_data(collect(bin_centers), collect(bin_counts), rho)
            brb_pdf = s -> probability_berry_robnik_brody(s, ρ_opt, β_opt)
            if log_scale
                lines!(ax, s_values, log10.(abs.(brb_pdf.(abs.(s_values)))), label="Berry-Robnik-Brody, ρ_fit=$(round(ρ_opt; sigdigits=5)), β_fit=$(round(β_opt; sigdigits=5))", color=:orange, linestyle=:solid, linewidth=1)
            else
                lines!(ax, s_values, brb_pdf.(abs.(s_values)), label="Berry-Robnik-Brody, ρ_fit=$(round(ρ_opt; sigdigits=5)), β_fit=$(round(β_opt; sigdigits=5))", color=:orange, linestyle=:solid, linewidth=1)
            end
        end
    end
    xlims!(ax, extrema(s_values))
    if log_scale
        axislegend(ax, position=:lb)
    else
        axislegend(ax, position=:rt)
    end
    return fig
end

"""
    plot_cumulative_spacing_distribution(unfolded_energy_eigenvalues::Vector{T}; rho::Union{Nothing, T}=nothing, plot_GUE=false, plot_inset=true fit_brb_cumul::Bool=false, fit_only_beta=false, fited_rho::Union{Nothing, T} = nothing, plot_log::Bool=false) where {T <: Real}

Plots the cumulative distribution function (CDF) of the nearest-neighbor level spacings (NNLS) for unfolded energy eigenvalues. Optionally, the Berry-Robnik CDF can be plotted if a `rho` value is provided.

# Arguments
- `unfolded_energy_eigenvalues::Vector{T}`: A vector of unfolded energy eigenvalues.
- `rho::Union{Nothing, T}=nothing`: The Liouville reg. phase space portion for the Berry-Robnik distribution. If `nothing`, the Berry-Robnik CDF is not plotted. Defaults to `nothing`.
- `plot_GUE::Bool=false`: Whether to plot the GUE curve. Defaults to `false`.
- `plot_inset::Bool=true`: Whether to plot an inset of small spacings. Defaults to `true`.
- `fit_brb_cumul::Bool=false`: Whether to fit the Berry-Robnik-Brody CDF to the data and display the optimal rho parameter in the legend. Defaults to `false`.
- `fit_only_beta::Bool=false`: If `true`, only the Berry-Robnik-Brody distribution's β parameter is fitted to the data, ρ is as given initially.
- `plot_log::Bool=false`: If we y axis will be in log scale -> log-lin plot. Defaults to `false`.

# Returns
- `Figure`.
"""
function plot_cumulative_spacing_distribution(unfolded_energy_eigenvalues::Vector{T}; rho::Union{Nothing, T}=nothing, plot_GUE=false, plot_inset=true, fit_brb_cumul::Bool=false, fit_only_beta=false, fited_rho::Union{Nothing, T} = nothing, plot_log::Bool=false) where {T <: Real}
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

    # If `rho` is provided, define the Berry-Robnik CDF
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
    # Compute Berry-Robnik-Brody values if `rho` is provided and also `fit_brb_cumul` is true
    ρ_opt, β_opt = nothing, nothing
    if !isnothing(rho) && fit_brb_cumul
        if fit_only_beta
            fited_rho = isnothing(fited_rho) ? rho : fited_rho
            empirial_spacings = Vector(collect(sorted_spacings))
            β_opt_fit = fit_brb_cumulative_to_data_only_beta(empirial_spacings, empirical_cdf, fited_rho)
            berry_robnik_brody_cdf_values = [cumulative_berry_robnik_brody(s, fited_rho, β_opt_fit) for s in s_values]
            ρ_opt, β_opt = fited_rho, β_opt_fit
        else
            empirial_spacings = Vector(collect(sorted_spacings))
            ρ_opt_fit, β_opt_fit = fit_brb_cumulative_to_data(empirial_spacings, empirical_cdf, rho)
            berry_robnik_brody_cdf_values = [cumulative_berry_robnik_brody(s, ρ_opt_fit, β_opt_fit) for s in s_values]
            ρ_opt, β_opt = ρ_opt_fit, β_opt_fit
        end
    else
        berry_robnik_brody_cdf_values = nothing
    end

    # Determine cutoff point in `s_values` based on GOE reaching 0.5
    max_s_index_goe = findfirst(x -> x > 0.5, goe_cdf_values)
    if max_s_index_goe !== nothing
        s_cutoff = s_values[max_s_index_goe]
    else
        s_cutoff = max_s  # If GOE CDF never reaches 0.5, use max_s as cutoff
        max_s_index_goe = length(s_values)  # Set to the end of the array
    end

    # Ensure all arrays are truncated to the same length
    max_index = min(max_s_index_goe, length(s_values), length(poisson_cdf_values), length(goe_cdf_values))
    if berry_robnik_cdf_values !== nothing
        max_index = min(max_index, length(berry_robnik_cdf_values))
    end

    # Create the main figure and axis
    fig = Figure(resolution = (1000, 1000))
    if plot_log
        ax = Axis(fig[1, 1], xlabel="Spacing (s)", ylabel="log10(W(s))", title="Cumulative Distribution of Nearest Neighbor Spacings")
    else
        ax = Axis(fig[1, 1], xlabel="Spacing (s)", ylabel="W(s)", title="Cumulative Distribution of Nearest Neighbor Spacings")
    end

    if plot_log
        # Plot the empirical CDF
        scatter!(ax, sorted_spacings, log10.(abs.(1.0.-empirical_cdf)), label="Empirical CDF", color=:blue, markersize=2)
        # Plot theoretical CDFs
        lines!(ax, s_values, log10.(abs.(1.0.-poisson_cdf_values)), label="Poisson CDF", color=:red, linewidth=1, linestyle=:dot)
        lines!(ax, s_values, log10.(abs.(1.0.-goe_cdf_values)), label="GOE CDF", color=:green, linewidth=1, linestyle=:dot)
        if plot_GUE
            lines!(ax, s_values, log10.(abs.(1.0.-gue_cdf_values)), label="GUE CDF", color=:purple, linewidth=1, linestyle=:dot)
        end
        if berry_robnik_cdf_values !== nothing
            lines!(ax, s_values, log10.(abs.(1.0.-berry_robnik_cdf_values)), label="BR: ρ_reg=$(round(rho; sigdigits=4))", color=:black, linewidth=2)
        end
        if berry_robnik_brody_cdf_values !== nothing
            lines!(ax,  s_values, log10.(abs.(1.0.-berry_robnik_brody_cdf_values)), label="BRB: ρ_reg=$(round(ρ_opt; sigdigits=4)), β=$(round(β_opt; sigdigits=4))", color=:orange, linewidth=2)
        end
        xlims!(ax, (2.0, 10.0))
    else
        # Plot the empirical CDF
        scatter!(ax, sorted_spacings, empirical_cdf, label="Empirical CDF", color=:blue, markersize=2)
        # Plot theoretical CDFs
        lines!(ax, s_values, poisson_cdf_values, label="Poisson CDF", color=:red, linewidth=1, linestyle=:dot)
        lines!(ax, s_values, goe_cdf_values, label="GOE CDF", color=:green, linewidth=1, linestyle=:dot)
        if plot_GUE
            lines!(ax, s_values, gue_cdf_values, label="GUE CDF", color=:purple, linewidth=1, linestyle=:dot)
        end
        if berry_robnik_cdf_values !== nothing
            lines!(ax, s_values, berry_robnik_cdf_values, label="BR: ρ_reg=$(round(rho; sigdigits=4))", color=:black, linewidth=2)
        end
        if berry_robnik_brody_cdf_values !== nothing
            lines!(ax,  s_values, berry_robnik_brody_cdf_values, label="BRB: ρ_reg=$(round(ρ_opt; sigdigits=4)), β=$(round(β_opt; sigdigits=4))", color=:orange, linewidth=2)
        end
    end
    axislegend(ax, position=:rb)

    # Inset plot settings
    if plot_inset
        inset_ax = Axis(fig[1, 1], width=Relative(0.5), height=Relative(0.5), halign=0.95, valign=0.5, xlabel="Spacing (s)", ylabel="Cumulative Probability")
        # Offset inset to bring it in front of the main axis content
        translate!(inset_ax.scene, 0, 0, 10)
        translate!(inset_ax.elements[:background], 0, 0, 9)
        # Plot cumulative distributions in the inset, limited to the cutoff
        max_s_cutoff_index = findfirst(s -> s >= s_cutoff, sorted_spacings)
        if max_s_cutoff_index === nothing
            max_s_cutoff_index = N  # Use full range if s_cutoff is beyond data range
        end
        if plot_log
            scatter!(inset_ax, sorted_spacings[1:max_s_cutoff_index], log10.(empirical_cdf[1:max_s_cutoff_index]), label="Empirical CDF", color=:blue, markersize=2)
            lines!(inset_ax, s_values[1:max_index], log10.(poisson_cdf_values[1:max_index]), label="Poisson CDF", color=:red, linewidth=1, linestyle=:dot)
            lines!(inset_ax, s_values[1:max_index], log10.(goe_cdf_values[1:max_index]), label="GOE CDF", color=:green, linewidth=1, linestyle=:dot)
            if plot_GUE
                lines!(inset_ax, s_values[1:max_index], log10.(gue_cdf_values[1:max_index]), label="GUE CDF", color=:purple, linewidth=1, linestyle=:dot)
            end
            if berry_robnik_cdf_values !== nothing
                lines!(inset_ax, s_values[1:max_index], log10.(berry_robnik_cdf_values[1:max_index]), label="BR CDF", color=:black, linewidth=1)
            end
            if berry_robnik_brody_cdf_values !== nothing
                lines!(inset_ax, s_values[1:max_index], log10.(berry_robnik_brody_cdf_values[1:max_index]), label="BRB CDF", color=:orange, linewidth=1)
            end
            # Set inset x and y limits to fit the range [0, s_cutoff] and [0, 0.5]
            xlims!(inset_ax, 0.0, s_cutoff)
            ylims!(inset_ax, log10.(1e-5), log10.(0.5))
        else
            scatter!(inset_ax, sorted_spacings[1:max_s_cutoff_index], empirical_cdf[1:max_s_cutoff_index], label="Empirical CDF", color=:blue, markersize=2)
            lines!(inset_ax, s_values[1:max_index], poisson_cdf_values[1:max_index], label="Poisson CDF", color=:red, linewidth=1, linestyle=:dot)
            lines!(inset_ax, s_values[1:max_index], goe_cdf_values[1:max_index], label="GOE CDF", color=:green, linewidth=1, linestyle=:dot)
            if plot_GUE
                lines!(inset_ax, s_values[1:max_index], gue_cdf_values[1:max_index], label="GUE CDF", color=:purple, linewidth=1, linestyle=:dot)
            end
            if berry_robnik_cdf_values !== nothing
                lines!(inset_ax, s_values[1:max_index], berry_robnik_cdf_values[1:max_index], label="BR CDF", color=:black, linewidth=1)
            end
            if berry_robnik_brody_cdf_values !== nothing
                lines!(inset_ax, s_values[1:max_index], berry_robnik_brody_cdf_values[1:max_index], label="BRB CDF", color=:orange, linewidth=1)
            end
            # Set inset x and y limits to fit the range [0, s_cutoff] and [0, 0.5]
            xlims!(inset_ax, 0.0, s_cutoff)
            ylims!(inset_ax, 1e-5, 0.5)
        end
    end
    return fig
end

function plot_U_diff(unfolded_energy_eigenvalues::Vector{T}; rho::T, fit_brb_cumul::Bool=false, fit_only_beta=false, num_bins = 100, fited_rho::Union{Nothing, T} = nothing) where {T <: Real}
    # Compute nearest neighbor spacings and sort them
    spacings = diff(sort(unfolded_energy_eigenvalues))
    sorted_spacings = collect(sort(spacings))
    N = length(sorted_spacings)
    # Compute the empirical CDF
    empirical_cdf = [i / N for i in 1:N]
    berry_robnik_cdf = (s, rho) -> cumulative_berry_robnik(s, rho)
    berry_robnik_cdf_values = [berry_robnik_cdf(s, rho) for s in sorted_spacings]
    
    # Compute Berry-Robnik-Brody values if requested
    ρ_opt, β_opt = nothing, nothing
    if fit_brb_cumul
        if fit_only_beta
            fited_rho = isnothing(fited_rho) ? rho : fited_rho
            β_opt_fit = fit_brb_cumulative_to_data_only_beta(sorted_spacings, empirical_cdf, fited_rho)
            berry_robnik_brody_cdf_values = [cumulative_berry_robnik_brody(s, fited_rho, β_opt_fit) for s in sorted_spacings]
            ρ_opt, β_opt = fited_rho, β_opt_fit
        else
            ρ_opt_fit, β_opt_fit = fit_brb_cumulative_to_data(sorted_spacings, empirical_cdf, rho)
            berry_robnik_brody_cdf_values = [cumulative_berry_robnik_brody(s, ρ_opt_fit, β_opt_fit) for s in sorted_spacings]
            ρ_opt, β_opt = ρ_opt_fit, β_opt_fit
        end
    else
        berry_robnik_brody_cdf_values = nothing
    end
    
    # Calculate U transformations
    U_numerical = U(empirical_cdf)
    U_berry_robnik = U(berry_robnik_cdf_values)
    
    # Calculate U differences
    dU_num_br = U_numerical .- U_berry_robnik
    if fit_brb_cumul && berry_robnik_brody_cdf_values !== nothing
        U_berry_robnik_brody = U(berry_robnik_brody_cdf_values)
        dU_num_brb = U_numerical .- U_berry_robnik_brody
    else
        dU_num_brb = nothing
    end

    # Bin the data and calculate the standard deviation within each bin
    bins = range(0, stop=1.0, length=num_bins + 1)
    bin_indices = searchsortedlast.(Ref(empirical_cdf), bins[1:end-1])
    bin_std_devs = Float64[]

    for i in 1:num_bins
        # Extract dU values in the current bin
        start_idx = i == 1 ? 1 : bin_indices[i-1] + 1
        end_idx = bin_indices[i]
        if start_idx <= end_idx
            dU_values_in_bin = dU_num_br[start_idx:end_idx]
            push!(bin_std_devs, std(dU_values_in_bin))
        end
    end

    # Calculate the maximum standard deviation across bins
    max_std_dev = maximum(bin_std_devs)
    
    fig = Figure(resolution = (2000, 1500), size=(2000,1500))
    w_cutoff = 1e-4
    u_cutoff = 0.025
    w_ticks = w_cutoff:0.1:(1.0-w_cutoff)
    u_ticks = (-u_cutoff):0.005:u_cutoff
    ax = Axis(fig[1, 1], xlabel="W(s)", ylabel=L"U - U(β,ρ)", title="U(s) transformation of W(s)", xticks=w_ticks, yticks=u_ticks)
    xlims!(ax, w_cutoff, 1.0-w_cutoff)
    ylims!(ax, -u_cutoff, u_cutoff)
    #lines!(ax, empirical_cdf, dU_num_br, label="BR: ρ_reg=$(round(rho; sigdigits=4))", color=:black, linewidth=2)
    #band!(ax, empirical_cdf, dU_num_br .+ max_std_dev, dU_num_br .- max_std_dev, color=:lightgray)
    if fit_brb_cumul && dU_num_brb !== nothing
        band!(ax, empirical_cdf, dU_num_br .+ max_std_dev, dU_num_br .- max_std_dev, color=(:lightgray, 0.3))
        band!(ax, empirical_cdf, dU_num_brb .+ max_std_dev, dU_num_brb .- max_std_dev, color=(:lightgray, 0.3))
        lines!(ax, empirical_cdf, dU_num_br, label="BR: ρ_reg=$(round(rho; sigdigits=4))", color=:black, linewidth=2)
        lines!(ax, empirical_cdf, dU_num_brb, label="BRB: ρ_reg=$(round(ρ_opt; sigdigits=4)), β=$(round(β_opt; sigdigits=4))", color=:orange, linewidth=2)
    else
        band!(ax, empirical_cdf, dU_num_br .+ max_std_dev, dU_num_br .- max_std_dev, color=(:lightgray, 0.3))
        lines!(ax, empirical_cdf, dU_num_br, label="BR: ρ_reg=$(round(rho; sigdigits=4))", color=:black, linewidth=2)
    end
    lines!(ax, [w_cutoff, 1.0-w_cutoff], [0.0,0.0], color=:red, linewidth=1)
    axislegend(ax, position=:rt)
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

"""
    length_spectrum(energies::Vector{T}, l::T, billiard::Bi; fundamental::Bool=true, regular_idx::Union{Nothing, Vector{Int}}=nothing, reg_or_cha=true) :: T where {T<:Real, Bi<:AbsBilliard}

Calculates the length spectrum for a particular `l`, using raw (non-unfolded) energy levels and Weyl's law.

# Arguments
- `energies::Vector`: A vector of raw energy levels.
- `l::T`: Length for which to compute the length spectrum.
- `billiard::Bi`: Billiard geometry, which provides the area, perimeter, and curvature/corner corrections.
- `fundamental::Bool`: Whether to use the fundamental domain of the billiard. Defaults to `true`.
- `regular_idx::Union{Nothing, Vector{Int}}`: Optional vector of regular indexes (must be regular otherwise we get complement).
- `reg_or_cha::Bool`: Whether to use regular or chaotic indexes. Defaults to `true` for regular idxs.

# Returns
- `T`: The length spectrum value for the given `l`, energies, and billiard geometry.
"""
function length_spectrum(energies::Vector, l::T, billiard::Bi; fundamental::Bool=true, regular_idx::Union{Nothing, Vector{Int}}=nothing, reg_or_cha=true) where {T<:Real, Bi<:AbsBilliard}
    A = T(fundamental ? billiard.area_fundamental : billiard.area)
    L = T(fundamental ? billiard.length_fundamental : billiard.length)
    C = T(curvature_and_corner_corrections(billiard; fundamental=fundamental))
    l_spec = complex(0.0)
    N_weyl = [(A/(4*pi)*k^2 - L/(4*pi)*k + C) for k in energies]
    N_numeric = [count(_k -> _k < k, energies) for k in energies]
    rho_fluct = N_numeric .- N_weyl
    # unfold then separate
    if !isnothing(regular_idx)
        if reg_or_cha == true # do regular
            rho_fluct = rho_fluct[regular_idx]
            energies = energies[regular_idx]
        else # do chaotic
            all_indices = Set(1:length(rho_fluct))  # Set of all indices
            chaotic_idx = sort(collect(setdiff(all_indices, regular_idx)))  # Find chaotic indices
            rho_fluct = rho_fluct[collect(chaotic_idx)]
            energies = energies[collect(chaotic_idx)]
        end    
    end
    for (rho, k) in zip(rho_fluct, energies)
        l_spec += rho * exp(im * k * l)
    end
    norm = length(energies)
    return abs(l_spec) / norm
end

"""
    length_spectrum(energies::Vector{T}, ls::Vector{T}, billiard::Bi; fundamental::Bool=true, regular_idx::Union{Nothing, Vector{Int}}=nothing, reg_or_cha=true) :: Vector{T} where {T<:Real, Bi<:AbsBilliard}

Calculates the length spectrum for a range of lengths `ls`, using raw (non-unfolded) energy levels and Weyl's law.

# Arguments
- `energies::Vector`: A vector of raw energy levels.
- `ls::Vector{T}`: A vector of lengths for which to compute the length spectrum.
- `billiard::Bi`: Billiard geometry, which provides the area, perimeter, and curvature/corner corrections.
- `fundamental::Bool`: Whether to use the fundamental domain of the billiard. Defaults to `true`.
- `regular_idx::Union{Nothing, Vector{Int}}`: Optional vector of regular indexes (must be regular otherwise we get complement).
- `reg_or_cha::Bool`: Whether to use regular or chaotic indexes. Defaults to `true` for regular idxs.

# Returns
- `Vector{T}`: A vector of length spectrum values corresponding to the given `ls` and `energies`.
"""
function length_spectrum(energies::Vector{T}, ls::Vector{T}, billiard::Bi; fundamental::Bool=true, regular_idx::Union{Nothing, Vector{Int}}=nothing, reg_or_cha=true) where {T<:Real, Bi<:AbsBilliard}
    l_spec = zeros(T, length(ls))
    Threads.@threads for i in 1:length(ls)
        l_spec[i] = length_spectrum(energies, ls[i], billiard; fundamental=fundamental, regular_idx=regular_idx, reg_or_cha=reg_or_cha)
    end
    return l_spec
end

"""
    plot_length_spectrum!(ax::Axis, energies::Vector, ls::Vector{T}, billiard::Bi; fundamental::Bool=true, regular_idx::Union{Nothing, Vector{Int}}=nothing, reg_or_cha=true) where {T<:Real, Bi<:AbsBilliard}

Simple high-level plotting of length spectrum.

# Arguments
- `ax::Axis`: The axis to plot on.
- `energies::Vector`: A vector of energy levels.
- `billiard::Bi`: Billiard geometry, which provides the area, perimeter, and curvature/corner corrections.
- `fundamental::Bool`: Whether to use the fundamental domain of the billiard. Defaults to `true`.
- `regular_idx::Union{Nothing, Vector{Int}}`: Optional vector of regular indexes (must be regular otherwise we get complement).
- `reg_or_cha::Bool`: Whether to use regular or chaotic indexes. Defaults to `true` for regular idxs.

# Returns
- `Nothing`
"""
function plot_length_spectrum!(ax::Axis, energies::Vector, ls::Vector{T}, billiard::Bi; fundamental::Bool=true, regular_idx::Union{Nothing, Vector{Int}}=nothing, reg_or_cha=true) where {T<:Real, Bi<:AbsBilliard}
    lines!(ax, ls, length_spectrum(energies, ls, billiard; fundamental=fundamental, regular_idx=regular_idx, reg_or_cha=reg_or_cha), linewidth=1)
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