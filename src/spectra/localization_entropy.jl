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
        A0, a, b, beta_model = fit_data
        params = a, b
        xs = collect(range(0.0, A0, 200)) # x scale for the beta distributin will be from 0.0 to A0
        ys = beta_model(xs, params)
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
- `beta_model::Function`: The best fitting beta distribution function.
"""
function fit_P_localization_entropy_to_beta(Hs::Vector, chaotic_classical_phase_space_vol_fraction::T; nbins=50) where {T<:Real}
    bin_centers, bin_counts = P_localization_entropy_pdf_data(Hs, chaotic_classical_phase_space_vol_fraction; nbins=nbins)
    bin_centers = collect(bin_centers)
    println("bin_centers, ", bin_centers) # debug
    println("bin_counts, ", bin_counts) # debug
    A0 = maximum(bin_centers)+0.05  # Fix A0
    function beta_model(A, params)
        a, b = params  # Only a and b are optimized
        B(x, y) = gamma(x) * gamma(y) / gamma(x + y)
        C = 1 / (A0^(a + b + 1) * B(a + 1, b + 1)) 
        result = Vector{Float64}(undef, length(A))
        for i in eachindex(A)
            term1 = (A[i] + 0im)^a
            term2 = (A0 - A[i] + 0im)^b
            term = C*term1 * term2 #/ C
            result[i] = isreal(term) ? real(term) : 0.0  # Ensure real output
        end
        return result
    end
    initial_guess = [30.0, 6.0]  # Initial guesses for a and b
    fit_result = curve_fit((A, params) -> beta_model(A, params), bin_centers, bin_counts, initial_guess)
    optimal_a, optimal_b = fit_result.param
    return A0, optimal_a, optimal_b, beta_model
end

# INTERNAL convert integer to Roman numeral (up to 16, otherwise arabic to string)
function int_to_roman(n::Int)
    romans = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii", "xiii", "xiv", "xv", "xvi"]
    return n <= length(romans) ? romans[n] : string(n)
end

# INTERNAL construct a gray chaotic background for the husimi plots
function husimi_with_chaotic_background(H::Matrix, projection_grid::Matrix)
     # Create a binary mask for chaotic regions
     chaotic_mask = projection_grid .== 1
     H_bg = H
     return H_bg, chaotic_mask
end

"""
    heatmap_M_vs_A_2d(Hs_list::Vector, qs_list::Vector, ps_list::Vector, classical_chaotic_s_vals::Vector, classical_chaotic_p_vals::Vector, chaotic_classical_phase_space_vol_fraction::T) where {T<:Real}

Plots the P(M,A) 2d heatmap along with 16 random representative chaotic Poincare-Husimi functions for that joint probability distributions.

# Arguments
- `Hs_list::Vector{Matrix}`: A list of Husimi function (matrices).
- `qs_list::Vector{Vector}`: Vector of Vectors that represent the qs for each Husimi matrix.
- `ps_list::Vector{Vector}`: Vector of Vectors that represent the ps for each Husimi matrix.
- `classical_chaotic_s_vals::Vector`: Vector of classical chaotic s values for a trajectory.
- `classical_chaotic_p_vals::Vector`: Vector of classical chaotic p values for a trajectory.
- `chaotic_classical_phase_space_vol_fraction::T`: The chaotic classical phase space volume fraction.

# Returns
- `fig::Figure`: Figure object from Makie to save or display.
"""
#=
function heatmap_M_vs_A_2d(Hs_list::Vector, qs_list::Vector, ps_list::Vector, classical_chaotic_s_vals::Vector, classical_chaotic_p_vals::Vector, chaotic_classical_phase_space_vol_fraction::T) where {T<:Real}
    Ms = compute_overlaps(Hs_list, qs_list, ps_list, classical_chaotic_s_vals, classical_chaotic_p_vals)
    As = [localization_entropy(H, chaotic_classical_phase_space_vol_fraction) for H in Hs_list]
    As_grid = collect(range(0.0, maximum(As), length=100)) # Create 2D grid for M and A with 100x100 bins
    Ms_grid = collect(range(-1.0, 1.0, length=100))
    grid = fill(0, length(Ms_grid), length(As_grid))  # Integer grid for counts
    H_to_bin = Dict{Int, Tuple{Int, Int}}() # Initialize a dictionary to map each Husimi matrix to its (M_index, A_index) bin
    
    # Dict for H -> (M,A)
    for (i, (M, A)) in enumerate(zip(Ms, As))
        A_index = findfirst(x -> x >= A, As_grid) - 1
        M_index = findfirst(x -> x >= M, Ms_grid) - 1
        if A_index in eachindex(As_grid) && M_index in eachindex(Ms_grid) # indices withing bounds, sanity check for findfirst
            grid[M_index, A_index] += 1
            H_to_bin[i] = (M_index, A_index)
        end
    end

    # Main grid P(A,M)
    fig = Figure(resolution=(1500, 1000), size=(1500,1000))
    ax = Axis(fig[1, 1], title="P(A,M)", xlabel="A", ylabel="M")
    heatmap!(ax, As_grid, Ms_grid, grid; colormap=Reverse(:gist_heat))

    selected_indices = rand(1:length(Hs_list), 16) # Choose 16 random Husimi matrices and label them with Roman numerals
    for (j, random_index) in enumerate(selected_indices)
        bin_coords = H_to_bin[random_index]
        M_index, A_index = bin_coords
        roman_label = int_to_roman(j)
        text!(ax, As_grid[A_index], Ms_grid[M_index], text=roman_label, color=:red, align=(:center, :center), fontsize=10)
    end

    # get the classical phase space matrix so we can make the gray spots on the chaotic grid whenever there is a 0.0 value of the chaotic husimi on it
    husimi_grid = fig[2:3, 1] = GridLayout()
    row = 1
    col = 1
    for (j, random_index) in enumerate(selected_indices)
        H = Hs_list[random_index]
        qs_i = qs_list[random_index]
        ps_i = ps_list[random_index]
        projection_grid = classical_phase_space_matrix(classical_chaotic_s_vals, classical_chaotic_p_vals, qs_i, ps_i)
        H_bg, chaotic_mask = husimi_with_chaotic_background(H, projection_grid)
        roman_label = int_to_roman(j)
        ax_husimi = Axis(husimi_grid[row, col], title=roman_label, xticksvisible=false, yticksvisible=false, xgridvisible=false, ygridvisible=false, xticklabelsvisible=false, yticklabelsvisible=false)
        #heatmap!(ax_husimi, H; colormap=Reverse(:gist_heat), nan_color=:lightgray) # Plot the Husimi matrix with NaN values as light gray
        #heatmap!(ax_husimi, H; colormap=Reverse(:gist_heat), colorrange=(0.0, maximum(H)))

        # Overlay the chaotic mask with transparency
        colormap = cgrad([:white, :black])  # Linear gradient from white to black
        heatmap!(ax_husimi, H_bg; colormap=Reverse(:gist_heat), colorrange=(0.0, maximum(H_bg)))
        heatmap!(ax_husimi, chaotic_mask; colormap=colormap, alpha=0.05, colorrange=(0, 1))

        text!(ax_husimi, 0.5, 0.1, text=roman_label, color=:black, fontsize=10) # Label the top left corner with the Roman numeral
        col += 1
        if col > 4  # Move to the next row after 4 columns
            col = 1
            row += 1
        end
    end
    rowgap!(husimi_grid, 5)
    colgap!(husimi_grid, 5)
    return fig
end
=#


#=
function heatmap_M_vs_A_2d(Hs_list::Vector,qs_list::Vector, ps_list::Vector, classical_chaotic_s_vals::Vector, classical_chaotic_p_vals::Vector, chaotic_classical_phase_space_vol_fraction::T; N_to_plot=12) where {T<:Real}
    # Compute R and A values
    Rs = [normalized_inverse_participation_ratio_R(H) for H in Hs_list]
    As = [localization_entropy(H, chaotic_classical_phase_space_vol_fraction) for H in Hs_list]
    max_A = maximum(As)
    A_max_range = max(0.7, max_A)  # Extend to the maximum A value if needed
    R_min = minimum(Rs) * 0.8
    R_max = maximum(Rs) * 1.2

    # Define bin edges and centers for A and R
    As_edges = collect(range(0.0, A_max_range, length=101))  # Dynamically adjusted bins for A-axis
    Rs_edges = collect(range(R_min, R_max, length=101))       # Dynamically adjusted bins for R-axis
    As_bin_centers = [(As_edges[i] + As_edges[i + 1]) / 2 for i in 1:(length(As_edges) - 1)]
    Rs_bin_centers = [(Rs_edges[i] + Rs_edges[i + 1]) / 2 for i in 1:(length(Rs_edges) - 1)]

    # Initialize the grid with swapped dimensions due to Makie
    grid = fill(0, length(As_bin_centers), length(Rs_bin_centers))
    H_to_bin = Dict{Int, Tuple{Int, Int}}()

    # Map each Husimi function to its bin (A_bin, R_bin)
    for (i, (R, A)) in enumerate(zip(Rs, As))
        A_index = findfirst(x -> x > A, As_edges)
        R_index = findfirst(x -> x > R, Rs_edges)
        # Handle cases where indices are out of bounds
        A_index = A_index === nothing ? length(As_bin_centers) : max(1, A_index - 1)
        R_index = R_index === nothing ? length(Rs_bin_centers) : max(1, R_index - 1)
        if A_index in 1:length(As_bin_centers) && R_index in 1:length(Rs_bin_centers)
            grid[A_index, R_index] += 1  # Swap indices here
            H_to_bin[i] = (A_index, R_index)  # Swap indices here
        else
            println("DEBUG: Skipped invalid bin for Husimi index $i (A=$A, R=$R, A_index=$A_index, R_index=$R_index)")
        end
    end

    fig = Figure(resolution=(2000, 1500),size=(2000,1500))
    ax = Axis(fig[1, 1], title="P(A,R)", xlabel="A", ylabel="R", xticks=As_bin_centers[1:10:end], yticks=Rs_bin_centers[1:10:end], xtickformat="{:.1f}", ytickformat="{:.1f}")
    heatmap!(ax, As_bin_centers, Rs_bin_centers, grid; colormap=Reverse(:gist_heat))

    # Select 16 random Husimi matrices and label them
    selected_indices = rand(1:length(Hs_list), N_to_plot)
    for (j, random_index) in enumerate(selected_indices)
        bin_coords = H_to_bin[random_index]
        A_index, R_index = bin_coords  # Swap indices here
        roman_label = int_to_roman(j)

        # Use bin centers for label placement
        R_center = Rs_bin_centers[R_index]
        A_center = As_bin_centers[A_index]
        # Plot a black square marker (outline) at the data point with transparent fill
        scatter!(ax, [A_center], [R_center],marker=:rect, color=:transparent, markersize=8, strokecolor=:black, strokewidth=1.5)
        if isodd(j) # for better viewing
            angle = 2*pi/3
        else
            angle = -pi/3
        end
        # Set fixed distance for label offset
        label_distance = 0.02 * sqrt((maximum(As_bin_centers) - minimum(As_bin_centers))^2 + (maximum(Rs_bin_centers) - minimum(Rs_bin_centers))^2)
        label_offset = (label_distance * cos(angle),label_distance * sin(angle))
        label_position = (A_center + label_offset[1], R_center + label_offset[2])
        # Add the text label at the offset position
        text!(ax, label_position[1], label_position[2], text=roman_label, color=:black, fontsize=30, halign=:center, valign=:center)
        # Draw a line from the data point to the label
        lines!(ax, [A_center, label_position[1]], [R_center, label_position[2]], color=:black)
    end

    # Husimi function grid layout
    husimi_grid = fig[2, 1] = GridLayout()
    row = 1
    col = 1
    for (j, random_index) in enumerate(selected_indices)
        H = Hs_list[random_index]
        qs_i = qs_list[random_index]
        ps_i = ps_list[random_index]

        # Create projection grid and chaotic mask
        projection_grid = classical_phase_space_matrix(classical_chaotic_s_vals, classical_chaotic_p_vals, qs_i, ps_i)
        H_bg, chaotic_mask = husimi_with_chaotic_background(H, projection_grid)
        roman_label = int_to_roman(j)

        # Plot individual Husimi functions
        ax_husimi = Axis(
            husimi_grid[row, col],
            title=roman_label,
            xticksvisible=false,
            yticksvisible=false,
            xgridvisible=false,
            ygridvisible=false,
            xticklabelsvisible=false,
            yticklabelsvisible=false
        )
        heatmap!(ax_husimi, H_bg; colormap=Reverse(:gist_heat), colorrange=(0.0, maximum(H_bg)))
        heatmap!(ax_husimi, chaotic_mask; colormap=cgrad([:white, :black]), alpha=0.05, colorrange=(0, 1))
        text!(ax_husimi, 0.5, 0.1, text=roman_label, color=:black, fontsize=10)

        col += 1
        if col > 4
            col = 1
            row += 1
        end
    end
    rowgap!(husimi_grid, 5)
    colgap!(husimi_grid, 5)

    return fig
end
=#

function heatmap_R_vs_A_2d( Hs_list::Vector,qs_list::Vector,ps_list::Vector,classical_chaotic_s_vals::Vector,classical_chaotic_p_vals::Vector,chaotic_classical_phase_space_vol_fraction::T;desired_samples::Int = 12) where {T<:Real}
    Rs = [normalized_inverse_participation_ratio_R(H) for H in Hs_list]
    As = [localization_entropy(H, chaotic_classical_phase_space_vol_fraction) for H in Hs_list]
    max_A = maximum(As)
    min_A = minimum(As)
    A_max_range = max(0.7, max_A)  # Extend to the maximum A value if needed
    R_min = minimum(Rs) * 0.8
    R_max = maximum(Rs) * 1.2

    # Define the number of bins along the A-axis
    num_bins_A = round(Int, sqrt(desired_samples))  # e.g., 4 bins
    As_edges = collect(range(min_A, max_A, length=num_bins_A + 1))

    # Initialize bin mappings
    bin_to_indices = Dict{Int, Vector{Int}}()

    # Map each Husimi function to its bin (A_bin)
    for (i, A) in enumerate(As)
        A_index = findfirst(x -> x > A, As_edges)
        A_index = A_index === nothing ? num_bins_A : max(1, A_index - 1)
        # Handle edge cases
        A_index = clamp(A_index, 1, num_bins_A)

        # Add index to the bin
        bin_to_indices[A_index] = get(bin_to_indices, A_index, Int[])
        push!(bin_to_indices[A_index], i)
    end

    # Initialize selected indices and bins
    selected_indices = []
    bins_available = collect(keys(bin_to_indices))

    # Iteratively select data points from bins
    while length(selected_indices) < desired_samples && !isempty(bins_available)
        # Make a copy of bins_available to iterate over
        bins_to_iterate = copy(bins_available)
        for bin_index in bins_to_iterate
            indices_in_bin = bin_to_indices[bin_index]
            if !isempty(indices_in_bin)
                # Randomly select a data point from the bin
                random_pos = rand(1:length(indices_in_bin))
                selected_index = indices_in_bin[random_pos]
                push!(selected_indices, selected_index)
                # Remove selected index from bin
                deleteat!(indices_in_bin, random_pos)
                # If bin is empty, remove it from bins_available
                if isempty(indices_in_bin)
                    delete!(bin_to_indices, bin_index)
                    filter!(x -> x != bin_index, bins_available)
                end
                # Check if we have reached desired samples
                if length(selected_indices) >= desired_samples
                    break
                end
            else
                # Bin is empty, remove it from bins_available
                delete!(bin_to_indices, bin_index)
                filter!(x -> x != bin_index, bins_available)
            end
        end
    end

    # Now proceed with plotting
    fig = Figure(resolution=(2000, 1500), size=(2000, 1500))
    ax = Axis(fig[1, 1],title="P(A,R)",xlabel="A",ylabel="R",xtickformat="{:.2f}",ytickformat="{:.2f}")
    # Plot the heatmap
    As_edges_heatmap = collect(range(0.0, A_max_range, length=201))  # For heatmap
    Rs_edges_heatmap = collect(range(R_min, R_max, length=201))      # For heatmap
    As_bin_centers_heatmap = [(As_edges_heatmap[i] + As_edges_heatmap[i + 1]) / 2 for i in 1:(length(As_edges_heatmap) - 1)]
    Rs_bin_centers_heatmap = [(Rs_edges_heatmap[i] + Rs_edges_heatmap[i + 1]) / 2 for i in 1:(length(Rs_edges_heatmap) - 1)]
    grid = zeros(length(As_bin_centers_heatmap), length(Rs_bin_centers_heatmap))
    # Map data points to heatmap grid
    for (i, (A, R)) in enumerate(zip(As, Rs))
        A_index = findfirst(x -> x > A, As_edges_heatmap)
        R_index = findfirst(x -> x > R, Rs_edges_heatmap)
        # Adjust indices
        A_index = A_index === nothing ? length(As_bin_centers_heatmap) : max(1, A_index - 1)
        R_index = R_index === nothing ? length(Rs_bin_centers_heatmap) : max(1, R_index - 1)
        if A_index in 1:length(As_bin_centers_heatmap) && R_index in 1:length(Rs_bin_centers_heatmap)
            grid[A_index, R_index] += 1
        end
    end

    heatmap!(ax, As_bin_centers_heatmap, Rs_bin_centers_heatmap, grid; colormap=Reverse(:gist_heat), alpha=0.7)

    # Now use selected_indices for labeling and plotting
    for (j, selected_index) in enumerate(selected_indices)
        A = As[selected_index]
        R = Rs[selected_index]
        roman_label = int_to_roman(j)
        # Plot a black square marker (outline) at the data point with transparent fill
        scatter!(ax,[A],[R],marker=:rect,color=:transparent,markersize=8,strokecolor=:black,strokewidth=1.5)
        # Alternate angles for label placement
        if isodd(j)
            angle = 2π / 3
        else
            angle = -π / 3
        end
        # Set fixed distance for label offset
        label_distance = 0.0
        label_offset = (label_distance * cos(angle),label_distance * sin(angle))
        label_position = (A + label_offset[1], R + label_offset[2])
        # Place the text inside the square
        text!(ax,label_position[1],label_position[2],text=roman_label,color=:black,fontsize=20)
    end

    # Husimi function grid layout
    husimi_grid = fig[2, 1] = GridLayout()
    row = 1
    col = 1
    for (j, selected_index) in enumerate(selected_indices)
        H = Hs_list[selected_index]
        qs_i = qs_list[selected_index]
        ps_i = ps_list[selected_index]
        # Create projection grid and chaotic mask
        projection_grid = classical_phase_space_matrix(classical_chaotic_s_vals,classical_chaotic_p_vals,qs_i,ps_i)
        H_bg, chaotic_mask = husimi_with_chaotic_background(H, projection_grid)
        roman_label = int_to_roman(j)
        # Plot individual Husimi functions
        ax_husimi = Axis(husimi_grid[row, col],title=roman_label,xticksvisible=false,yticksvisible=false,xgridvisible=false,ygridvisible=false,xticklabelsvisible=false,yticklabelsvisible=false
        )
        heatmap!(ax_husimi, H_bg; colormap=Reverse(:gist_heat), colorrange=(0.0, maximum(H_bg)))
        heatmap!(ax_husimi,chaotic_mask;colormap=cgrad([:white, :black]),alpha=0.05,colorrange=(0, 1))
        text!(ax_husimi, 0.5, 0.1, text=roman_label, color=:black, fontsize=10)
        col += 1
        if col > 4
            col = 1
            row += 1
        end
    end
    rowgap!(husimi_grid, 5)
    colgap!(husimi_grid, 5)

    return fig
end

function heatmap_M_vs_A_2d( Hs_list::Vector,qs_list::Vector,ps_list::Vector,classical_chaotic_s_vals::Vector,classical_chaotic_p_vals::Vector,chaotic_classical_phase_space_vol_fraction::T;desired_samples::Int = 12) where {T<:Real}

    # Compute R and A values
    Ms = compute_overlaps(Hs_list, qs_list, ps_list, classical_chaotic_s_vals, classical_chaotic_p_vals)
    As = [localization_entropy(H, chaotic_classical_phase_space_vol_fraction) for H in Hs_list]
    max_A = maximum(As)
    min_A = minimum(As)
    A_max_range = max(0.7, max_A)  # Extend to the maximum A value if needed
    M_min = minimum(Ms) * 0.8
    M_max = maximum(Ms) * 1.2

    # Define the number of bins along the A-axis
    num_bins_A = round(Int, sqrt(desired_samples))  # e.g., 4 bins
    As_edges = collect(range(min_A, max_A, length=num_bins_A + 1))

    # Initialize bin mappings
    bin_to_indices = Dict{Int, Vector{Int}}()

    # Map each Husimi function to its bin (A_bin)
    for (i, A) in enumerate(As)
        A_index = findfirst(x -> x > A, As_edges)
        A_index = A_index === nothing ? num_bins_A : max(1, A_index - 1)
        # Handle edge cases
        A_index = clamp(A_index, 1, num_bins_A)

        # Add index to the bin
        bin_to_indices[A_index] = get(bin_to_indices, A_index, Int[])
        push!(bin_to_indices[A_index], i)
    end

    # Initialize selected indices and bins
    selected_indices = []
    bins_available = collect(keys(bin_to_indices))

    # Iteratively select data points from bins, when exhausting one remove or from the pool in next iteration of the while loop
    while length(selected_indices) < desired_samples && !isempty(bins_available)
        # Make a copy of bins_available to iterate over
        bins_to_iterate = copy(bins_available)
        for bin_index in bins_to_iterate
            indices_in_bin = bin_to_indices[bin_index]
            if !isempty(indices_in_bin)
                # Randomly select a data point from the bin
                random_pos = rand(1:length(indices_in_bin))
                selected_index = indices_in_bin[random_pos]
                push!(selected_indices, selected_index)
                # Remove selected index from bin
                deleteat!(indices_in_bin, random_pos)
                # If bin is empty, remove it from bins_available
                if isempty(indices_in_bin)
                    delete!(bin_to_indices, bin_index)
                    filter!(x -> x != bin_index, bins_available)
                end
                # Check if we have reached desired samples
                if length(selected_indices) >= desired_samples
                    break
                end
            else
                # Bin is empty, remove it from bins_available
                delete!(bin_to_indices, bin_index)
                filter!(x -> x != bin_index, bins_available)
            end
        end
    end

    # Now proceed with plotting
    fig = Figure(resolution=(2000, 1500), size=(2000, 1500))
    ax = Axis(fig[1, 1],title="P(A,M)",xlabel="A",ylabel="M",xtickformat="{:.2f}",ytickformat="{:.2f}")
    # Plot the heatmap
    As_edges_heatmap = collect(range(0.0, A_max_range, length=201))  # For heatmap
    Ms_edges_heatmap = collect(range(M_min, M_max, length=201))      # For heatmap
    As_bin_centers_heatmap = [(As_edges_heatmap[i] + As_edges_heatmap[i + 1]) / 2 for i in 1:(length(As_edges_heatmap) - 1)]
    Ms_bin_centers_heatmap = [(Ms_edges_heatmap[i] + Ms_edges_heatmap[i + 1]) / 2 for i in 1:(length(Ms_edges_heatmap) - 1)]
    grid = zeros(length(As_bin_centers_heatmap), length(Ms_bin_centers_heatmap))
    # Map data points to heatmap grid
    for (i, (A, R)) in enumerate(zip(As, Ms))
        A_index = findfirst(x -> x > A, As_edges_heatmap)
        M_index = findfirst(x -> x > R, Ms_edges_heatmap)
        # Adjust indices
        A_index = A_index === nothing ? length(As_bin_centers_heatmap) : max(1, A_index - 1)
        M_index = M_index === nothing ? length(Ms_bin_centers_heatmap) : max(1, M_index - 1)
        if A_index in 1:length(As_bin_centers_heatmap) && M_index in 1:length(Ms_bin_centers_heatmap)
            grid[A_index, M_index] += 1
        end
    end

    heatmap!(ax, As_bin_centers_heatmap, Ms_bin_centers_heatmap, grid; colormap=Reverse(:gist_heat), alpha=0.7)

    # Now use selected_indices for labeling and plotting
    for (j, selected_index) in enumerate(selected_indices)
        A = As[selected_index]
        M = Ms[selected_index]
        roman_label = int_to_roman(j)
        # Plot a black square marker (outline) at the data point with transparent fill
        scatter!(ax,[A],[M],marker=:rect,color=:transparent,markersize=8,strokecolor=:black,strokewidth=1.5)
        # Alternate angles for label placement
        if isodd(j)
            angle = 2π / 3
        else
            angle = -π / 3
        end
        # Set fixed distance for label offset
        label_distance = 0.0
        label_offset = (label_distance * cos(angle),label_distance * sin(angle))
        label_position = (A + label_offset[1], M + label_offset[2])
        # Place the text inside the square
        text!(ax,label_position[1],label_position[2],text=roman_label,color=:black,fontsize=20)
    end

    # Husimi function grid layout
    husimi_grid = fig[2, 1] = GridLayout()
    row = 1
    col = 1
    for (j, selected_index) in enumerate(selected_indices)
        H = Hs_list[selected_index]
        qs_i = qs_list[selected_index]
        ps_i = ps_list[selected_index]
        # Create projection grid and chaotic mask
        projection_grid = classical_phase_space_matrix(classical_chaotic_s_vals,classical_chaotic_p_vals,qs_i,ps_i)
        H_bg, chaotic_mask = husimi_with_chaotic_background(H, projection_grid)
        roman_label = int_to_roman(j)
        # Plot individual Husimi functions
        ax_husimi = Axis(husimi_grid[row, col],title=roman_label,xticksvisible=false,yticksvisible=false,xgridvisible=false,ygridvisible=false,xticklabelsvisible=false,yticklabelsvisible=false
        )
        heatmap!(ax_husimi, H_bg; colormap=Reverse(:gist_heat), colorrange=(0.0, maximum(H_bg)))
        heatmap!(ax_husimi,chaotic_mask;colormap=cgrad([:white, :black]),alpha=0.05,colorrange=(0, 1))
        text!(ax_husimi, 0.5, 0.1, text=roman_label, color=:black, fontsize=10)
        col += 1
        if col > 4
            col = 1
            row += 1
        end
    end
    rowgap!(husimi_grid, 5)
    colgap!(husimi_grid, 5)

    return fig
end