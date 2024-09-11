using LinearAlgebra
using Statistics
using Makie
using Polynomials
using Threads
using Distributions
using StatsBase

function number_variance(E::Vector{T}, L::T) where {T<:Real}
    # By Tomaž Prosen
    Ave1 = 0.0
    Ave2 = 0.0
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

function spectral_rigidity(E::Vector{T}, L::T) where {T<:Real}
    N = length(E)
    Ave = Threads.Atomic{Float64}(0.0)  # Use atomic operations to safely update the shared variable
    largest_energy = E[end - min(Int(ceil(L) + 10), N-1)]  # Ensure largest_energy is within bounds
    
    # Parallelize the main loop
    Threads.@threads for idx in 1:N
        j = idx
        k = j
        x = E[j]
        while x < largest_energy && j < N && k < N
            while k < N && E[k] < x + L  # Ensure k does not exceed bounds
                k += 1
            end
            
            d1 = E[j] - x
            d2 = E[k] - (x + L)
            cn = k - j  # Number of levels in the interval n(x_i, L)
            
            if cn < 2
                if d1 < d2
                    x = E[j]
                    j += 1
                else
                    x = E[k] - L
                    k += 1
                end
                continue
            end

            # Get the energy levels in the current interval
            E_interval = E[j:k-1]
            nE = 1:length(E_interval)

            # Perform a linear fit to n(E) = a + b*E
            p = Polynomials.fit(E_interval, nE, 1)
            
            # Calculate the deviation from the linear fit
            fit_values = [p(e) for e in E_interval]
            deviation = sum((nE .- fit_values).^2) / L

            # Accumulate the deviation for the interval
            s = min(d1, d2)
            Threads.atomic_add!(Ave, s * deviation)

            # Move to the next interval
            if d1 < d2
                x = E[j]
                j += 1
            else
                x = E[k] - L
                k += 1
            end
        end
    end

    # Normalize by the total length of the energy spectrum considered
    total_length = largest_energy - E[1]
    Ave_value = Ave[] / total_length

    return Ave_value  # This is the spectral rigidity, Δ3(L)
end

function compare_level_count_to_weyl(arr::Vector{T}, A::T, L::T, C::T) where {T<:Real}
    xs = arr
    ys = [count(_k -> _k < k, xs) for k in xs]
    Ns = [(A/(4*pi)*k^2 - L/(4*pi)*k + C) for k in xs]
    # Create the plot
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="k", ylabel="N(k)", title="Level Count vs. Weyl's Law")
    # Scatter plot for the numerical data
    scatter!(ax, xs, ys, label="Numerical", color=:blue, marker=:circle, markersize=3)
    # Line plot for the Weyl prediction
    lines!(ax, xs, Ns, label="Weyl", color=:red)
    axislegend(ax)
    return fig
end

function unfold_with_weyl(arr::Vector{T}, A::T, L::T, C::T) where {T<:Real}
    res = [A/(4*pi)*k^2 - L/(4*pi)*k + C for k in arr]
    res_diff = [abs(res[i+1] - res[i]) for i in 1:(length(res)-1)]
    return res, mean(res_diff) # Also the mean level spacing for good unfolding
end

function plot_number_variance(unfolded_energies::Vector{T}, L_min::T, L_max::T; N::Int=100) where {T<:Real}
    # Theoretical number variance functions
    function poissonSigmaL(L::Float64)
        return L
    end
    function goeSigmaL(L::Float64)
        return (2.0 / (π * π)) * (log(2.0 * π * L) + 0.57721566490153286060 + 1.0 - (π * π / 8.0))
    end
    function gueSigmaL(L::Float64)
        return (1.0 / (π * π)) * (log(2.0 * π * L) + 0.57721566490153286060 + 1.0)
    end
    Ls = collect(range(L_min, L_max, N))
    NVs = [number_variance(unfolded_energies, L) for L in Ls]
    # Theoretical curves
    poisson_NV = [poissonSigmaL(L) for L in Ls]
    goe_NV = [goeSigmaL(L) for L in Ls]
    gue_NV = [gueSigmaL(L) for L in Ls]

    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="L", ylabel="Number Variance")
    lines!(ax, Ls, NVs, label="Numeric", color=:black)
    lines!(ax, Ls, poisson_NV, label="Poisson", linestyle=:dash, color=:red)
    lines!(ax, Ls, goe_NV, label="GOE", linestyle=:dot, color=:blue)
    lines!(ax, Ls, gue_NV, label="GUE", linestyle=:dashdot, color=:green)
    axislegend(ax)
    return fig
end

function plot_spectral_rigidity!(ax::GLMakie.Axis, unfolded_energies::Vector{T}, k::T, L_min::T, L_max::T; N::Int=100)
    Ls = range(L_min, L_max, length=N)
    println("Calculating spectral rigity to k: ", k)
    Δ3_values = [spectral_rigidity(unfolded_energies, L) for L in Ls]
    # Theoretical Spectral Rigidities
    poissonDeltaL = L -> L / 15.0 
    goeDeltaL = L -> (1 / (π^2)) * (log(2π * L) + 0.57721566490153286060 - 5.0 / 4.0 - (π^2 / 8))
    gueDeltaL = L -> (1 / (2 * π^2)) * (log(2π * L) + 0.57721566490153286060 - 5.0 / 4.0)
    # Plot numerical spectral rigidity
    label = "Numerical k_max=" * string(round(Int, k))
    lines!(ax, Ls, Δ3_values, color=:black, label=label)
    # Plot theoretical predictions
    lines!(ax, Ls, [poissonDeltaL(L) for L in Ls], label="Poisson", color=:blue, linestyle=:dash)
    lines!(ax, Ls, [goeDeltaL(L) for L in Ls], label="GOE", color=:green, linestyle=:dot)
    lines!(ax, Ls, [gueDeltaL(L) for L in Ls], label="GUE", color=:red, linestyle=:dashdot)
end

function plot_nnls(unfolded_energies::Vector{T}, k::T; nbins::Int=200) where {T<:Real}
    # Compute nearest neighbor spacings
    spacings = diff(unfolded_energies)
    # Create a normalized histogram
    hist = Distributions.fit(StatsBase.Histogram, spacings; nbins=nbins)
    bin_centers = (hist.edges[1][1:end-1] .+ hist.edges[1][2:end]) / 2
    bin_counts = hist.weights ./ sum(hist.weights) / diff(hist.edges[1])[1]
    # Theoretical distributions
    poisson_pdf = x -> exp(-x)
    goe_pdf = x -> (π / 2) * x * exp(-π * x^2 / 4)
    gue_pdf = x -> (32 / (π^2)) * x^2 * exp(-4 * x^2 / π)
    # Plotting
    fig = Figure(resolution=(800, 600))
    ax = Axis(fig[1, 1], title="k_max = $(round(k, digits=2))", xlabel="Spacing (s)", ylabel="Probability Density")
    # Plot the empirical histogram
    scatter!(ax, bin_centers, bin_counts, label="Empirical", color=:black, marker=:cross, markersize=10)
    # Plot the theoretical distributions
    s_values = range(0, stop=maximum(bin_centers), length=1000)
    lines!(ax, s_values, poisson_pdf.(s_values), label="Poisson", color=:blue, linestyle=:dash, linewidth=1)
    lines!(ax, s_values, goe_pdf.(s_values), label="GOE", color=:green, linestyle=:dot, linewidth=1)
    lines!(ax, s_values, gue_pdf.(s_values), label="GUE", color=:red, linestyle=:dashdot, linewidth=1)
    xlims!(ax, [0.0, 10.0])
    # Add legend
    axislegend(ax, position=:rt)
    return fig
end