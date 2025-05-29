using LsqFit, QuadGK, SpecialFunctions, ProgressMeter

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
    H=H./sum(H)  # normalize H
    non_zero_idxs=findall(H.>0.0)  # Find indices of non-zero elements
    Im=sum(H[non_zero_idxs].*log.(H[non_zero_idxs]))  # Compute the entropy for non-zero elements
    A=1.0/(T(prod(size(H)))*chaotic_classical_phase_space_vol_fraction)*exp(-Im)
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
    H=H./sum(H)
    R=1/(prod(size(H))*sum(H.^2)) # the prod(size(H)) is the grid count directly from the size of the matrix
    return R
end

"""
    correlation_matrix(H1::Matrix{T}, H2::Matrix{T})

Computes the corelation matrix between the Husimi functions.

```math
Cₙₘ = 1/{QₙQₘ}∑HⁿᵢⱼHᵐᵢⱼ
```

# Arguments
- `H1::Matrix{T}`: First Husimi function.
- `H2::Matrix{T}`: Second Husimi function.

# Returns
- `T`: The value of the nm-correlation matrix element.
""" 
function correlation_matrix(H1::Matrix{T},H2::Matrix{T}) where {T<:Real}
    numerator=sum(H1.*H2)
    denominator=sqrt(sum(H1.^2)*sum(H2.^2))
    if denominator==0.0
        throw(ErrorException("Denominator cannot be zero"))
    end
    return numerator/denominator
end

"""
    correlation_matrix_and_average(H_list::Vector{Matrix{T}}) where {T<:Real}

Computes the correlation matrix and it's average for a sequence of consecutive Husimi matrices.

# Arguments
- `H_list::Vector{Matrix{T}}`: Vector of Husimi matrices.

# Returns
- `corr_mat::Matrix{T}`: The correlation matrix for visualization.
- `avg_corr::T`: The average correlation value for the given Husimi matrices.
"""
function correlation_matrix_and_average(H_list::Vector)
    n=length(H_list)
    corr_mat=Matrix{Float64}(undef,n,n)
    norms=[sqrt(sum(H.^2)) for H in H_list] 
    total_corr=Threads.Atomic{Float64}(0.0) 
    count=Threads.Atomic{Int}(0) 
    @showprogress desc="Computing correlation matrices N=$n" Threads.@threads for i in 1:n
        @inbounds for j in i:n  # Only loop over upper triangular matrix
            numerator=sum(H_list[i].*H_list[j])
            denominator=norms[i]*norms[j]
            corr=numerator/denominator
            corr_mat[i,j]=corr
            corr_mat[j,i]=corr  # Symmetric
            Threads.atomic_add!(total_corr,corr)
            Threads.atomic_add!(count, 1)
        end
    end
    avg_corr=total_corr[]/count[] 
    return corr_mat,avg_corr
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
function P_localization_entropy_pdf_data(Hs::Vector{Matrix{T}},chaotic_classical_phase_space_vol_fraction::T;nbins=50) where {T<:Real}
    localization_entropies=[localization_entropy(H,chaotic_classical_phase_space_vol_fraction) for H in Hs]
    hist=Distributions.fit(StatsBase.Histogram,localization_entropies;nbins=nbins)
    bin_centers=(hist.edges[1][1:end-1].+hist.edges[1][2:end])/2
    bin_counts=hist.weights./sum(hist.weights)/diff(hist.edges[1])[1]
    return bin_centers,bin_counts
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
function plot_P_localization_entropy_pdf!(ax::Axis,Hs::Vector,chaotic_classical_phase_space_vol_fraction::T;nbins=50,color::Symbol=:lightblue,fit_beta::Bool=false) where {T<:Real}
    bin_centers,bin_counts=P_localization_entropy_pdf_data(Hs,chaotic_classical_phase_space_vol_fraction;nbins=nbins)
    barplot!(ax,bin_centers,bin_counts,label="A distribution",color=color,gap=0,strokecolor=:black,strokewidth=1)
    if fit_beta
        fit_data=fit_P_localization_entropy_to_beta(Hs,chaotic_classical_phase_space_vol_fraction,nbins=nbins)
        A0,a,b,beta_model=fit_data
        params=a,b
        xs=collect(range(0.0,A0,200)) # x scale for the beta distributin will be from 0.0 to A0
        ys=beta_model(xs,params)
        param_label="Beta dist.fit: [A0=$(round(A0, digits=2)), a=$(round(a, digits=2)), b=$(round(b, digits=2))]"
        lines!(ax,xs,ys,label=param_label,color=:red)
    end
    axislegend(ax,position=:rt)
    xlims!(ax,(0.0,1.0))
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
function fit_P_localization_entropy_to_beta(Hs::Vector,chaotic_classical_phase_space_vol_fraction::T;nbins=50) where {T<:Real}
    bin_centers,bin_counts=P_localization_entropy_pdf_data(Hs,chaotic_classical_phase_space_vol_fraction;nbins=nbins)
    bin_centers=collect(bin_centers)
    #println("bin_centers, ",bin_centers) # debug
    #println("bin_counts, ",bin_counts) # debug
    A0=maximum(bin_centers)+0.05  # Fix A0
    function beta_model(A,params)
        a,b=params  # Only a and b are optimized
        B(x,y)=gamma(x)*gamma(y)/gamma(x+y)
        C=1/(A0^(a+b+1)*B(a+1,b+1)) 
        result=Vector{Float64}(undef,length(A))
        for i in eachindex(A)
            term1=(A[i]+0im)^a
            term2=(A0-A[i]+0im)^b
            term=C*term1*term2 #/ C
            result[i]= isreal(term) ? real(term) : 0.0  # Ensure real output
        end
        return result
    end
    initial_guess=[30.0,6.0]  # Initial guesses for a and b
    fit_result=curve_fit((A,params) -> beta_model(A,params),bin_centers,bin_counts,initial_guess)
    optimal_a,optimal_b=fit_result.param
    return A0,optimal_a,optimal_b,beta_model
end

# INTERNAL convert integer to Roman numeral (up to 16, otherwise arabic to string)
function int_to_roman(n::Int)
    romans = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii", "xiii", "xiv", "xv", "xvi"]
    return n<=length(romans) ? romans[n] : string(n)
end

# INTERNAL construct a gray chaotic background for the husimi plots
function husimi_with_chaotic_background(H::Matrix,projection_grid::Matrix)
     # Create a binary mask for chaotic regions
     chaotic_mask = projection_grid .== 1
     H_bg=H
     return H_bg,chaotic_mask
end

"""
    heatmap_R_vs_A_2d(Hs_list::Vector, qs_list::Vector, ps_list::Vector, classical_chaotic_s_vals::Vector, classical_chaotic_p_vals::Vector, chaotic_classical_phase_space_vol_fraction::T; desired_samples::Int = 12) where {T<:Real}

Plots the P(R,A) 2d heatmap along with random representative chaotic Poincare-Husimi functions for that joint probability distributions (R,A), where R is the normalized inverse participation ratio.

# Arguments
- `Hs_list::Vector{Matrix}`: A list of Husimi function (matrices).
- `qs_list::Vector{Vector}`: Vector of Vectors that represent the qs for each Husimi matrix.
- `ps_list::Vector{Vector}`: Vector of Vectors that represent the ps for each Husimi matrix.
- `classical_chaotic_s_vals::Vector`: Vector of classical chaotic s values for a trajectory.
- `classical_chaotic_p_vals::Vector`: Vector of classical chaotic p values for a trajectory.
- `chaotic_classical_phase_space_vol_fraction::T`: The chaotic classical phase space volume fraction.
- `desired_samples::Integer=12`: The number of husimi data samples to choose from and display.

# Returns
- `fig::Figure`: Figure object from Makie to save or display.
"""
function heatmap_R_vs_A_2d( Hs_list::Vector,qs_list::Vector,ps_list::Vector,classical_chaotic_s_vals::Vector,classical_chaotic_p_vals::Vector,chaotic_classical_phase_space_vol_fraction::T;desired_samples::Int = 12) where {T<:Real}
    Rs = [normalized_inverse_participation_ratio_R(H) for H in Hs_list]
    As = [localization_entropy(H, chaotic_classical_phase_space_vol_fraction) for H in Hs_list]
    max_A = maximum(As)
    min_A = minimum(As)
    A_max_range = max(0.7, max_A)  # Extend to the maximum A value if needed
    R_min = minimum(Rs) * 0.8
    R_max = maximum(Rs) * 1.2

    # Define the number of bins along the A-axis
    num_bins_A = 2*round(Int, sqrt(desired_samples))  # e.g., 4 bins
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
    ax.xticks = range(min_A, max_A, 20) 
    ax.yticks = range(R_min, R_max, 10) 
    ax.xticklabelrotation = pi/2
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
        scatter!(ax,[A],[R],marker=:rect,color=:transparent,markersize=8,strokecolor=:black,strokewidth=3.5)
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
        text!(ax,label_position[1],label_position[2],text=roman_label,color=:black,fontsize=30)
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

"""
    heatmap_M_vs_A_2d(Hs_list::Vector, qs_list::Vector, ps_list::Vector, classical_chaotic_s_vals::Vector, classical_chaotic_p_vals::Vector, chaotic_classical_phase_space_vol_fraction::T; desired_samples::Int = 12) where {T<:Real}

Plots the P(M,A) 2d heatmap along with random representative chaotic Poincare-Husimi functions for that joint probability distributions (M,A), where M is the overlap index of the Poincare-Husimi function.

# Arguments
- `Hs_list::Vector{Matrix}`: A list of Husimi function (matrices).
- `qs_list::Vector{Vector}`: Vector of Vectors that represent the qs for each Husimi matrix.
- `ps_list::Vector{Vector}`: Vector of Vectors that represent the ps for each Husimi matrix.
- `classical_chaotic_s_vals::Vector`: Vector of classical chaotic s values for a trajectory.
- `classical_chaotic_p_vals::Vector`: Vector of classical chaotic p values for a trajectory.
- `chaotic_classical_phase_space_vol_fraction::T`: The chaotic classical phase space volume fraction.
- `desired_samples::Integer=12`: The number of husimi data samples to choose from and display.

# Returns
- `fig::Figure`: Figure object from Makie to save or display.
"""
function heatmap_M_vs_A_2d( Hs_list::Vector,qs_list::Vector,ps_list::Vector,classical_chaotic_s_vals::Vector,classical_chaotic_p_vals::Vector,chaotic_classical_phase_space_vol_fraction::T; desired_samples::Int = 12) where {T<:Real}

    # Compute R and A values
    Ms = compute_overlaps(Hs_list, qs_list, ps_list, classical_chaotic_s_vals, classical_chaotic_p_vals)
    As = [localization_entropy(H, chaotic_classical_phase_space_vol_fraction) for H in Hs_list]
    max_A = maximum(As)
    min_A = minimum(As)
    A_max_range = max(0.7, max_A)  # Extend to the maximum A value if needed
    M_min = minimum(Ms) * 0.8
    M_max = maximum(Ms) * 1.2

    # Define the number of bins along the A-axis
    num_bins_A = 2*round(Int, sqrt(desired_samples))  # e.g., 4 bins
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
    ax.xticks = range(min_A, max_A, 20) 
    ax.yticks = range(M_min, M_max, 10) 
    ax.xticklabelrotation = pi/2
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
        scatter!(ax,[A],[M],marker=:rect,color=:transparent,markersize=8,strokecolor=:black,strokewidth=3.5)
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
        text!(ax,label_position[1],label_position[2],text=roman_label,color=:black,fontsize=30)
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

"""
    combined_heatmaps_with_husimi(Hs_list::Vector, qs_list::Vector, ps_list::Vector, classical_chaotic_s_vals::Vector, classical_chaotic_p_vals::Vector, chaotic_classical_phase_space_vol_fraction::T; desired_samples::Int = 12) where {T<:Real}

Plots the P(M,A) 2d heatmap and P(R,A) 2d heatmap along with random representative chaotic Poincare-Husimi functions for that joint probability distributions (M/R,A), where M is the overlap index of the Poincare-Husimi function and R is the normalized inverse participation ratio.

# Arguments
- `Hs_list::Vector{Matrix}`: A list of Husimi function (matrices).
- `qs_list::Vector{Vector}`: Vector of Vectors that represent the qs for each Husimi matrix.
- `ps_list::Vector{Vector}`: Vector of Vectors that represent the ps for each Husimi matrix.
- `classical_chaotic_s_vals::Vector`: Vector of classical chaotic s values for a trajectory.
- `classical_chaotic_p_vals::Vector`: Vector of classical chaotic p values for a trajectory.
- `chaotic_classical_phase_space_vol_fraction::T`: The chaotic classical phase space volume fraction.
- `desired_samples::Integer=12`: The number of husimi data samples to choose from and display.

# Returns
- `fig::Figure`: Figure object from Makie to save or display.
"""
function combined_heatmaps_with_husimi(Hs_list::Vector, qs_list::Vector, ps_list::Vector, classical_chaotic_s_vals::Vector, classical_chaotic_p_vals::Vector, chaotic_classical_phase_space_vol_fraction::T; desired_samples::Int = 12) where {T<:Real}

    # Compute M, R, and A values
    Ms = compute_overlaps(Hs_list, qs_list, ps_list, classical_chaotic_s_vals, classical_chaotic_p_vals)
    Rs = [normalized_inverse_participation_ratio_R(H) for H in Hs_list]
    As = [localization_entropy(H, chaotic_classical_phase_space_vol_fraction) for H in Hs_list]

    println("Ms = ", Ms)
    println("  any NaN in Ms? ", any(isnan, Ms))
    println("Rs = ", Rs)
    println("  any NaN in Rs? ", any(isnan, Rs))
    println("As = ", As)
    println("  any NaN in As? ", any(isnan, As))

    @assert all(isfinite, Ms)   "compute_overlaps returned NaN"
    @assert all(isfinite, Rs)   "R calculation returned NaN"
    @assert all(isfinite, As)   "A calculation returned NaN"

    max_A = maximum(As)
    min_A = minimum(As)
    A_max_range = max(0.7, max_A)  # Extend to the maximum A value if needed
    M_min = minimum(Ms) #* 0.8
    M_max = maximum(Ms) * 1.05
    R_min = minimum(Rs) #* 0.8
    R_max = maximum(Rs) * 1.05

    # Define the number of bins along the A-axis
    num_bins_A = 2 * round(Int, sqrt(desired_samples))  # e.g., 4 bins
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
    fig = Figure(resolution=(2000, 2500), size=(2000, 2500))  # Adjusted size for three plots

    ### Top Plot: P(A, M) ###
    ax_top = Axis(fig[1, 1], title="P(A, M)", xlabel="A", ylabel="M", xtickformat="{:.2f}", ytickformat="{:.2f}")
    #ax_top.xticks = range(min_A, max_A, 20) 
    ax_top.yticks = range(M_min, M_max, 10) 
    ax_top.xticklabelrotation = pi/2
    # Plot the heatmap for P(A, M)
    As_edges_heatmap_M = collect(range(0.0, A_max_range, length=201))
    Ms_edges_heatmap = collect(range(M_min, M_max, length=201))
    As_bin_centers_heatmap_M = [(As_edges_heatmap_M[i] + As_edges_heatmap_M[i + 1]) / 2 for i in 1:(length(As_edges_heatmap_M) - 1)]
    Ms_bin_centers_heatmap = [(Ms_edges_heatmap[i] + Ms_edges_heatmap[i + 1]) / 2 for i in 1:(length(Ms_edges_heatmap) - 1)]
    grid_M = zeros(length(As_bin_centers_heatmap_M), length(Ms_bin_centers_heatmap))
    # Map data points to heatmap grid for P(A, M)
    for (i, (A, M)) in enumerate(zip(As, Ms))
        A_index = findfirst(x -> x > A, As_edges_heatmap_M)
        M_index = findfirst(x -> x > M, Ms_edges_heatmap)
        # Adjust indices
        A_index = A_index === nothing ? length(As_bin_centers_heatmap_M) : max(1, A_index - 1)
        M_index = M_index === nothing ? length(Ms_bin_centers_heatmap) : max(1, M_index - 1)
        if A_index in 1:length(As_bin_centers_heatmap_M) && M_index in 1:length(Ms_bin_centers_heatmap)
            grid_M[A_index, M_index] += 1
        end
    end

    A_lim_min = minimum(As_bin_centers_heatmap_M)
    A_lim_max = maximum(As_bin_centers_heatmap_M)
    xlims!(ax_top, (A_lim_min, A_lim_max))
    ax_top.xticks = range(A_lim_min, A_lim_max, length=20)
    ylims!(ax_top, (M_min, M_max))
    heatmap!(ax_top, As_bin_centers_heatmap_M, Ms_bin_centers_heatmap, grid_M; colormap=Reverse(:gist_heat), alpha=0.7)

    # Label the selected points on the P(A, M) plot
    for (j, selected_index) in enumerate(selected_indices)
        A = As[selected_index]
        M = Ms[selected_index]
        roman_label = int_to_roman(j)
        # Plot a black square marker (outline) at the data point with transparent fill
        scatter!(ax_top, [A], [M], marker=:rect, color=:transparent, markersize=8, strokecolor=:black, strokewidth=3.5)
        # Place the text inside the square
        text!(ax_top, A, M, text=roman_label, color=:black, fontsize=30, halign=:center, valign=:center)
    end

    ### Middle Plot: P(A, R) ###
    ax_middle = Axis(fig[2, 1], title="P(A, R)", xlabel="A", ylabel="R", xtickformat="{:.2f}", ytickformat="{:.2f}")
    #ax_middle.xticks = range(min_A, max_A, 20) 
    ax_middle.yticks = range(R_min, R_max, 10) 
    ax_middle.xticklabelrotation = pi/2
    # Plot the heatmap for P(A, R)
    As_edges_heatmap_R = collect(range(0.0, A_max_range, length=201))
    Rs_edges_heatmap = collect(range(R_min, R_max, length=201))
    As_bin_centers_heatmap_R = [(As_edges_heatmap_R[i] + As_edges_heatmap_R[i + 1]) / 2 for i in 1:(length(As_edges_heatmap_R) - 1)]
    Rs_bin_centers_heatmap = [(Rs_edges_heatmap[i] + Rs_edges_heatmap[i + 1]) / 2 for i in 1:(length(Rs_edges_heatmap) - 1)]
    grid_R = zeros(length(As_bin_centers_heatmap_R), length(Rs_bin_centers_heatmap))
    # Map data points to heatmap grid for P(A, R)
    for (i, (A, R)) in enumerate(zip(As, Rs))
        A_index = findfirst(x -> x > A, As_edges_heatmap_R)
        R_index = findfirst(x -> x > R, Rs_edges_heatmap)
        # Adjust indices
        A_index = A_index === nothing ? length(As_bin_centers_heatmap_R) : max(1, A_index - 1)
        R_index = R_index === nothing ? length(Rs_bin_centers_heatmap) : max(1, R_index - 1)
        if A_index in 1:length(As_bin_centers_heatmap_R) && R_index in 1:length(Rs_bin_centers_heatmap)
            grid_R[A_index, R_index] += 1
        end
    end
    A_lim_min = minimum(As_bin_centers_heatmap_R)
    A_lim_max = maximum(As_bin_centers_heatmap_R)
    xlims!(ax_middle, (A_lim_min, A_lim_max))
    ax_middle.xticks = range(A_lim_min, A_lim_max, length=20)
    ylims!(ax_middle, (R_min, R_max))
    heatmap!(ax_middle, As_bin_centers_heatmap_R, Rs_bin_centers_heatmap, grid_R; colormap=Reverse(:gist_heat), alpha=0.7)

    # Label the selected points on the P(A, R) plot
    for (j, selected_index) in enumerate(selected_indices)
        A = As[selected_index]
        R = Rs[selected_index]
        roman_label = int_to_roman(j)
        # Plot a black square marker (outline) at the data point with transparent fill
        scatter!(ax_middle, [A], [R], marker=:rect, color=:transparent, markersize=8, strokecolor=:black, strokewidth=3.5)
        # Place the text inside the square
        text!(ax_middle, A, R, text=roman_label, color=:black, fontsize=30, halign=:center, valign=:center)
    end

    ### Husimi Functions at the Bottom ###
    # Husimi function grid layout at the bottom
    husimi_grid = fig[3, 1] = GridLayout()
    row = 1
    col = 1
    for (j, selected_index) in enumerate(selected_indices)
        H = Hs_list[selected_index]
        qs_i = qs_list[selected_index]
        ps_i = ps_list[selected_index]
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
        H_bg ./= maximum(H_bg)
        Threads.@threads for i in axes(H_bg,1)
            for j in axes(H_bg,2)
                if projection_grid[i,j] == -1 # if it is regular
                    H_bg[i,j] = ifelse(H_bg[i,j] > 1e-1, H_bg[i,j], NaN) 
                else # the chaotic region
                    H_bg[i,j] = H_bg[i, j] 
                end
            end
        end
        heatmap!(ax_husimi, H_bg; colormap=Reverse(:gist_heat), colorrange=(0.0, 1.0), nan_color=:lightgray)
        # Label
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

"""
    combined_heatmaps_with_husimi( Hs_list::Vector, qs_list::Vector, ps_list::Vector, classical_chaotic_s_vals::Vector, classical_chaotic_p_vals::Vector, chaotic_classical_phase_space_vol_fraction::T, Psi2ds::Vector{Matrix}, x_grid::Vector{<:Real}, y_grid::Vector{<:Real}, billiard::Bi; desired_samples::Int = 12,
) where {T<:Real, Bi<:AbsBilliard}

Plots the P(M,A) 2d heatmap and P(R,A) 2d heatmap along with random representative chaotic Poincare-Husimi functions for that joint probability distributions (M/R,A), where M is the overlap index of the Poincare-Husimi function and R is the normalized inverse participation ratio. Additionaly also on the sides plots the wavefunctions that correspond to these husimi plots.

# Arguments
- `Hs_list::Vector{Matrix}`: A list of Husimi function (matrices).
- `qs_list::Vector{Vector}`: Vector of Vectors that represent the qs for each Husimi matrix.
- `ps_list::Vector{Vector}`: Vector of Vectors that represent the ps for each Husimi matrix.
- `classical_chaotic_s_vals::Vector`: Vector of classical chaotic s values for a trajectory.
- `classical_chaotic_p_vals::Vector`: Vector of classical chaotic p values for a trajectory.
- `chaotic_classical_phase_space_vol_fraction::T`: The chaotic classical phase space volume fraction.
- `Psi2ds::Vector{Matrix}`: A list of wavefunctions (matrices).
- `x_grid::Vector{<:Real}`: Vector of x grid points for the wavefunctions.
- `y_grid::Vector{<:Real}`: Vector of y grid points for the wavefunctions.
- `billiard<:AbsBilliard`: The billiard object.
- `desired_samples::Integer=12`: The number of husimi data samples to choose from and display.

# Returns
- `fig::Figure`: Figure object from Makie to save or display.
"""
function combined_heatmaps_with_husimi( Hs_list::Vector, qs_list::Vector, ps_list::Vector, classical_chaotic_s_vals::Vector, classical_chaotic_p_vals::Vector, chaotic_classical_phase_space_vol_fraction::T, Psi2ds::Vector, x_grid::Vector, y_grid::Vector, billiard::Bi; desired_samples::Int = 12,
) where {T<:Real, Bi<:AbsBilliard}

    # Compute M, R, and A values
    Ms = compute_overlaps(Hs_list, qs_list, ps_list, classical_chaotic_s_vals, classical_chaotic_p_vals)
    Rs = [normalized_inverse_participation_ratio_R(H) for H in Hs_list]
    As = [localization_entropy(H, chaotic_classical_phase_space_vol_fraction) for H in Hs_list]

    max_A = maximum(As)
    min_A = minimum(As)
    A_max_range = max(0.7, max_A)  # Extend to the maximum A value if needed
    M_min = minimum(Ms) * 0.8
    M_max = maximum(Ms) * 1.2
    R_min = minimum(Rs) * 0.8
    R_max = maximum(Rs) * 1.2

    # Define the number of bins along the A-axis
    num_bins_A = 2 * round(Int, sqrt(desired_samples))  # e.g., 4 bins
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

    # Prepare for plotting
    # Figure layout with left and right columns for wavefunctions, middle column for heatmaps and Husimi functions
    fig = Figure(resolution=(2000, 2500), fontsize=14)

    # Define the grid layout
    grid = fig[1, 1] = GridLayout()

    # Left wavefunction plots
    left_grid = grid[1, 1] = GridLayout()
    # Middle plots (heatmaps and Husimi functions)
    middle_grid = grid[1, 2] = GridLayout()
    # Right wavefunction plots
    right_grid = grid[1, 3] = GridLayout()

    # Adjust column widths
    colsize!(grid, 1, Relative(1))  # Left wavefunctions
    colsize!(grid, 2, Relative(2))  # Middle plots
    colsize!(grid, 3, Relative(1))  # Right wavefunctions

    ### Plot Left Wavefunctions ###
    # Left side wavefunctions (2 columns, 3 rows)
    num_left = div(desired_samples, 2)
    rows_left = ceil(Int, num_left / 2)
    row = 1
    col = 1
    for i in 1:num_left
        idx = selected_indices[i]
        roman_label = int_to_roman(i)
        ax_wf = Axis(
            left_grid[row, col],
            title=roman_label,
            aspect=DataAspect(),
            xticksvisible=false,
            yticksvisible=false,
            xgridvisible=false,
            ygridvisible=false
        )
        # Plot the wavefunction
        Psi = Psi2ds[idx]
        hm = heatmap!(ax_wf, x_grid, y_grid, Psi; colormap=:balance, colorrange=(-maximum(abs, Psi), maximum(abs, Psi)))
        # Plot billiard boundary
        plot_boundary!(ax_wf, billiard; fundamental_domain=true, plot_normal=false)
        col += 1
        if col > 2
            col = 1
            row += 1
        end
    end

    ### Plot Middle Heatmaps and Husimi Functions ###
    # Middle Grid
    middle_grid[1, 1] = GridLayout()
    middle_subgrid = middle_grid[1, 1]

    # Top Plot: P(A, M)
    ax_top = Axis(middle_subgrid[1, 1], title="P(A, M)", xlabel="A", ylabel="M", xtickformat="{:.2f}", ytickformat="{:.2f}")
    # Plot the heatmap for P(A, M)
    As_edges_heatmap_M = collect(range(0.0, A_max_range, length=201))
    Ms_edges_heatmap = collect(range(M_min, M_max, length=201))
    As_bin_centers_heatmap_M = [(As_edges_heatmap_M[i] + As_edges_heatmap_M[i + 1]) / 2 for i in 1:(length(As_edges_heatmap_M) - 1)]
    Ms_bin_centers_heatmap = [(Ms_edges_heatmap[i] + Ms_edges_heatmap[i + 1]) / 2 for i in 1:(length(Ms_edges_heatmap) - 1)]
    grid_M = zeros(length(As_bin_centers_heatmap_M), length(Ms_bin_centers_heatmap))
    # Map data points to heatmap grid for P(A, M)
    for (i, (A, M)) in enumerate(zip(As, Ms))
        A_index = findfirst(x -> x > A, As_edges_heatmap_M)
        M_index = findfirst(x -> x > M, Ms_edges_heatmap)
        # Adjust indices
        A_index = A_index === nothing ? length(As_bin_centers_heatmap_M) : max(1, A_index - 1)
        M_index = M_index === nothing ? length(Ms_bin_centers_heatmap) : max(1, M_index - 1)
        if A_index in 1:length(As_bin_centers_heatmap_M) && M_index in 1:length(Ms_bin_centers_heatmap)
            grid_M[A_index, M_index] += 1
        end
    end
    heatmap!(ax_top, As_bin_centers_heatmap_M, Ms_bin_centers_heatmap, grid_M; colormap=Reverse(:gist_heat), alpha=0.7)

    # Label the selected points on the P(A, M) plot
    for (j, selected_index) in enumerate(selected_indices)
        A = As[selected_index]
        M = Ms[selected_index]
        roman_label = int_to_roman(j)
        # Plot a black square marker (outline) at the data point with transparent fill
        scatter!(ax_top, [A], [M], marker=:rect, color=:transparent, markersize=8, strokecolor=:black, strokewidth=3.5)
        # Place the text inside the square
        text!(ax_top, A, M, text=roman_label, color=:black, fontsize=30, halign=:center, valign=:center)
    end

    # Middle Plot: P(A, R)
    ax_middle = Axis(middle_subgrid[2, 1], title="P(A, R)", xlabel="A", ylabel="R", xtickformat="{:.2f}", ytickformat="{:.2f}")
    # Plot the heatmap for P(A, R)
    As_edges_heatmap_R = collect(range(0.0, A_max_range, length=201))
    Rs_edges_heatmap = collect(range(R_min, R_max, length=201))
    As_bin_centers_heatmap_R = [(As_edges_heatmap_R[i] + As_edges_heatmap_R[i + 1]) / 2 for i in 1:(length(As_edges_heatmap_R) - 1)]
    Rs_bin_centers_heatmap = [(Rs_edges_heatmap[i] + Rs_edges_heatmap[i + 1]) / 2 for i in 1:(length(Rs_edges_heatmap) - 1)]
    grid_R = zeros(length(As_bin_centers_heatmap_R), length(Rs_bin_centers_heatmap))
    # Map data points to heatmap grid for P(A, R)
    for (i, (A, R)) in enumerate(zip(As, Rs))
        A_index = findfirst(x -> x > A, As_edges_heatmap_R)
        R_index = findfirst(x -> x > R, Rs_edges_heatmap)
        # Adjust indices
        A_index = A_index === nothing ? length(As_bin_centers_heatmap_R) : max(1, A_index - 1)
        R_index = R_index === nothing ? length(Rs_bin_centers_heatmap) : max(1, R_index - 1)
        if A_index in 1:length(As_bin_centers_heatmap_R) && R_index in 1:length(Rs_bin_centers_heatmap)
            grid_R[A_index, R_index] += 1
        end
    end
    heatmap!(ax_middle, As_bin_centers_heatmap_R, Rs_bin_centers_heatmap, grid_R; colormap=Reverse(:gist_heat), alpha=0.7)

    # Label the selected points on the P(A, R) plot
    for (j, selected_index) in enumerate(selected_indices)
        A = As[selected_index]
        R = Rs[selected_index]
        roman_label = int_to_roman(j)
        # Plot a black square marker (outline) at the data point with transparent fill
        scatter!(ax_middle, [A], [R], marker=:rect, color=:transparent, markersize=8, strokecolor=:black, strokewidth=3.5)
        # Place the text inside the square
        text!(ax_middle, A, R, text=roman_label, color=:black, fontsize=30, halign=:center, valign=:center)
    end

    # Husimi Functions at the Bottom
    husimi_grid = middle_subgrid[3, 1] = GridLayout()
    row = 1
    col = 1
    for (j, selected_index) in enumerate(selected_indices)
        H = Hs_list[selected_index]
        qs_i = qs_list[selected_index]
        ps_i = ps_list[selected_index]
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

    ### Plot Right Wavefunctions ###
    # Right side wavefunctions (2 columns, 3 rows)
    num_right = desired_samples - num_left
    rows_right = ceil(Int, num_right / 2)
    row = 1
    col = 1
    for i in (num_left + 1):desired_samples
        idx = selected_indices[i]
        roman_label = int_to_roman(i)
        ax_wf = Axis(
            right_grid[row, col],
            title=roman_label,
            aspect=DataAspect(),
            xticksvisible=false,
            yticksvisible=false,
            xgridvisible=false,
            ygridvisible=false
        )
        # Plot the wavefunction
        Psi = Psi2ds[idx]
        hm = heatmap!(ax_wf, x_grid, y_grid, Psi; colormap=:balance, colorrange=(-maximum(abs, Psi), maximum(abs, Psi)))
        # Plot billiard boundary
        plot_boundary!(ax_wf, billiard; fundamental_domain=true, plot_normal=false)
        col += 1
        if col > 2
            col = 1
            row += 1
        end
    end

    return fig
end