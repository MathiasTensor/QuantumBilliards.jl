using JLD2, Makie

# PRELIMINARY FUNCTIONS

"""
    shift_s_vals_poincare_birkhoff(s_vals::Vector{T}, s_shift::T, boundary_length::T) where {T<:Real}

Shift the classical arc-length values `s_vals` by a specified shift `s_shift`, 
ensuring that the shifted values are periodic and wrap around a closed boundary.

# Arguments
- `s_vals::Vector{T}`: A vector of arc-length values representing classical positions along the boundary of the billiard. These values correspond to the Poincaré-Birkhoff coordinates in classical phase space.
- `s_shift::T`: The shift to be applied to each value in `s_vals`, aligning the classical coordinates with quantum conventions.
- `boundary_length::T`: The total length of the boundary of the billiard. This ensures that the shifted `s_vals` are wrapped around the boundary.

# Returns
- `shifted_s_vals::Vector{T}`: A vector containing the shifted arc-length values. Each `s_val` is shifted by `s_shift` and wrapped around the boundary using modulo arithmetic with `boundary_length`.
"""
function shift_s_vals_poincare_birkhoff(s_vals::Vector{T}, s_shift::T, boundary_length::T) where {T<:Real}
    shifted_s_vals = Vector{T}(undef, length(s_vals))
    for i in 1:length(s_vals)
        shifted_s_vals[i] = (s_vals[i] + s_shift) % boundary_length
    end
    return shifted_s_vals
end

"""
    classical_phase_space_matrix(classical_s_vals::Vector{T}, classical_p_vals::Vector{T}, qs::Vector{T}, ps::Vector{T}) where {T<:Real}

Project a classical chaotic trajectory (represented by `classical_s_vals` and `classical_p_vals`) onto a Husimi grid (defined by `qs` and `ps`), marking the Husimi grid cells that contain 
classical points with a `+1`, while keeping other cells as `-1`.

# Arguments
- `classical_s_vals::Vector{T}`: Classical arc-length values `s` representing the position on the boundary.
- `classical_p_vals::Vector{T}`: Classical momentum values `p`.
- `qs::Vector{T}`: Husimi grid values in the `s` coordinate (arc-length).
- `ps::Vector{T}`: Husimi grid values in the `p` coordinate (momentum).

# Returns
- A 2D matrix `Matrix` with dimensions `length(qs) × length(ps)`:
    - Each cell is `+1` if a classical chaotic point falls inside the corresponding Husimi cell.
    - Cells that do not contain chaotic classical points are set to `-1`.

In this implementation, we use a brute-force approach to check whether each classical point 
(`s`, `p`) falls inside a grid cell. The reason for this is to accommodate irregular spacing 
in the Husimi grid, particularly for the `qs` grid, which may be constructed via Fourier sampling or other non-uniform methods. 
"""
function classical_phase_space_matrix(classical_s_vals::Vector{T}, classical_p_vals::Vector{T}, qs::Vector{T}, ps::Vector{T}) where {T<:Real}
    s_grid_size = length(qs)
    p_grid_size = length(ps)
    # Initialize the projection grid with -1
    projection_grid = fill(-1, s_grid_size, p_grid_size)
    # Use an efficient function to determine the correct grid index wrt mapping (s,p) -> (qs, ps)
    function find_index(val, grid)
        idx = searchsortedlast(grid, val)  # find the index where val would fit in the grid
        return clamp(idx, 1, length(grid))  # clamp to ensure it's within grid bounds, otherwise problem
    end
    
    # Mark the projection grid based on classical_s_vals and classical_p_vals
    Threads.@threads for i in 1:length(classical_s_vals)
        classical_s = classical_s_vals[i]
        classical_p = classical_p_vals[i]
        # Find correct grid index for classical_s and classical_p using searchsortedlast. Could still have some artifacts for small dimensional matrices
        s_idx = find_index(classical_s, qs)
        p_idx = find_index(classical_p, ps)
        # Mark the corresponding grid cell as chaotic (1)
        projection_grid[s_idx, p_idx] = 1
    end
    return projection_grid
end

"""
    compute_M(projection_grid::Matrix, H::Matrix)

Compute the overlap function M between a classical projection grid and a Husimi function matrix. H is normalized internally.

# Arguments
- `projection_grid::Matrix`: A matrix with values:
    - `+1` for cells corresponding to chaotic classical points.
    - `-1` for cells corresponding to regular classical points.
- `H::Matrix`: A Husimi function matrix of the same size as `projection_grid`.

# Returns
- `M`: The computed overlap value, which is the sum of the element-wise product (Kroenecker product) of `projection_grid` and `H`.
"""
function compute_M(projection_grid::Matrix, H::Matrix)
    @assert size(projection_grid) == size(H) "H and projection grid must be same size"
    H = H ./ sum(H) # normalize
    M = sum(broadcast(*, projection_grid, H))
    return M
end

# VISUALISATIONS

"""
    visualize_overlap(projection_grid::Matrix, H::Matrix)

Visualize the overlap between a classical phase space projection grid and a Husimi function matrix. 

# Arguments
- `projection_grid::Matrix`: A matrix with values:
    - `+1` for cells corresponding to chaotic classical points.
    - `-1` for cells corresponding to regular classical points.
- `H::Matrix`: A Husimi function matrix of the same size as `projection_grid`.

# Returns
- `M`: A matrix representing the element-wise product of the `projection_grid` and the normalized Husimi function `H`.
"""
function visualize_overlap(projection_grid::Matrix, H::Matrix)
    @assert size(projection_grid) == size(H) "H and projection grid must be same size"
    H = H ./ sum(H) # normalize
    M = broadcast(*, projection_grid, H)
    return M
end

"""
    visualize_quantum_classical_overlap_of_levels!(ks::Vector, H_list::Vector{Matrix}, qs_list::Vector{Vector}, ps_list::Vector{Vector}, classical_chaotic_s_vals::Vector, classical_chaotic_p_vals::Vector)

Generates and saves visualizations of the quantum-classical overlap for each level based on precomputed Husimi functions.

# Arguments
- `ks::Vector`:  
  Vector of wavenumbers corresponding to each quantum level.
- `H_list::Vector{Matrix}`:  
  Vector of Husimi functions for each quantum level.
- `qs_list::Vector{Vector}`:  
  Vector of position grids (`q`) for each Husimi function.
- `ps_list::Vector{Vector}`:  
  Vector of momentum grids (`p`) for each Husimi function.
- `classical_chaotic_s_vals::Vector`:  
  Vector of classical chaotic `s` values used to compute the projection grid.
- `classical_chaotic_p_vals::Vector`:  
  Vector of classical chaotic `p` values used to compute the projection grid.

# Returns
- `Nothing`
"""
function visualize_quantum_classical_overlap_of_levels!(ks::Vector, H_list::Vector{Matrix}, qs_list::Vector{Vector}, 
    ps_list::Vector{Vector}, classical_chaotic_s_vals::Vector, classical_chaotic_p_vals::Vector)
    if !isdir("Overlap_visualization")
        mkdir("Overlap_visualization")
    end
    progress_computing = Progress(length(ks), desc = "Computing overlaps...")
    progress_saving = Progress(length(ks), desc = "Saving overlap visualizations...")
    Ms = Vector{Float64}(undef, length(ks))
    projection_grids = Vector{Matrix}(undef, length(ks))
    overlaps = Vector{Matrix}(undef, length(ks))
    Threads.@threads for i in eachindex(ks) # only this can multithread, precompute data
        try
            H = H_list[i]
            qs = qs_list[i]
            ps = ps_list[i]
            proj_grid = classical_phase_space_matrix(classical_chaotic_s_vals, classical_chaotic_p_vals, qs, ps)
            M_val = compute_M(proj_grid, H)
            overlap_val = visualize_overlap(proj_grid, H)
            projection_grids[i] = proj_grid
            Ms[i] = M_val
            overlaps[i] = overlap_val
        catch e
            @warn "Failed to compute overlap for k = $(ks[i]): $(e)"
        end
        next!(progress_computing)
    end

    for i in eachindex(ks) # do not multithread, memory corruption problem
        try
            f = Figure(resolution = (800, 1500))
            ax_H = Axis(f[1,1])
            hmap = heatmap!(ax_H, H_list[i]; colormap=Reverse(:gist_heat))
            Colorbar(f[1,2], hmap)
            ax_projection = Axis(f[2,1])
            hmap = heatmap!(ax_projection, projection_grids[i])
            Colorbar(f[2,2], hmap)
            ax_overlap = Axis(f[3,1], title="k = $(round(ks[i]; sigdigits=8)), Overlap = $(round(Ms[i]; sigdigits=4))")
            hmap = heatmap!(ax_overlap, overlaps[i]; colormap=:balance)
            Colorbar(f[3,2], hmap)
            colsize!(f.layout, 1, Aspect(3, 1))
            save("Overlap_visualization/$(ks[i])_overlap.png", f)
        catch e
            @warn "Failed to save overlap for k = $(ks[i]): $(e)"
        end
        next!(progress_saving)
    end
end

"""
    visualize_quantum_classical_overlap_of_levels!(ks::Vector, H_list::Vector{Matrix}, qs_list::Vector{Vector}, 
    ps_list::Vector{Vector}, classical_chaotic_s_vals::Vector, classical_chaotic_p_vals::Vector, 
    state_data::StateData, billiard::Bi, basis::Ba; 
    b = 5.0, inside_only = true, fundamental_domain = true, memory_limit = 10.0e9)

Generates and saves visualizations of the quantum-classical overlap for each level, and also plots wavefunction heatmaps. This one is for visualization of a small number of levels due to the wavefunction computations.

# Arguments
- `ks::Vector`:  
  Vector of wavenumbers corresponding to each quantum level.
- `H_list::Vector{Matrix}`:  
  Vector of Husimi functions for each quantum level.
- `qs_list::Vector{Vector}`:  
  Vector of position grids (`q`) for each Husimi function.
- `ps_list::Vector{Vector}`:  
  Vector of momentum grids (`p`) for each Husimi function.
- `classical_chaotic_s_vals::Vector`:  
  Vector of classical chaotic `s` values used to compute the projection grid.
- `classical_chaotic_p_vals::Vector`:  
  Vector of classical chaotic `p` values used to compute the projection grid.
- `state_data::StateData`:  
  The `StateData` object containing wavefunction information from the spectrum calculations.
- `billiard::Bi`:  
  The billiard object for computing wavefunctions.
- `basis::Ba`:  
  The basis object for computing wavefunctions.
- `b::Float64`:  
  Point scaling factor for the wavefunctions (default 5.0).
- `inside_only::Bool`:  
  Whether to only include points inside the billiard for wavefunction (default true).
- `fundamental_domain::Bool`:  
  Whether to compute wavefunctions only in the fundamental domain (default true).
- `memory_limit::Float64`:  
  The memory limit for constructing the wavefunction (default 10.0e9 bytes).

# Returns
- `Nothing`
"""
function visualize_quantum_classical_overlap_of_levels!(ks::Vector, H_list::Vector{Matrix}, qs_list::Vector{Vector}, 
    ps_list::Vector{Vector}, classical_chaotic_s_vals::Vector, classical_chaotic_p_vals::Vector, 
    state_data::StateData, billiard::Bi, basis::Ba; 
    b = 5.0, inside_only = true, fundamental_domain = true, memory_limit = 10.0e9) where {Bi<:AbsBilliard, Ba<:AbsBasis}
    if !isdir("Overlap_visualization")
        mkdir("Overlap_visualization")
    end
    progress_computing = Progress(length(ks), desc = "Computing overlaps...")
    progress_saving = Progress(length(ks), desc = "Saving overlap visualizations...")
    Ms = Vector{Float64}(undef, length(ks))
    projection_grids = Vector{Matrix}(undef, length(ks))
    overlaps = Vector{Matrix}(undef, length(ks))
    ks_wf, Psi2ds, x_grids, y_grids = wavefunctions(state_data, billiard, basis; b=b, inside_only=inside_only, fundamental_domain=fundamental_domain, memory_limit=memory_limit)

    Threads.@threads for i in eachindex(ks) # only this can multithread, precompute data
        try
            H = H_list[i]
            qs = qs_list[i]
            ps = ps_list[i]
            proj_grid = classical_phase_space_matrix(classical_chaotic_s_vals, classical_chaotic_p_vals, qs, ps)
            println("Size of projection grid: ", size(proj_grid), " , size of H: ", size(H))
            M_val = compute_M(proj_grid, H)
            overlap_val = visualize_overlap(proj_grid, H)
            projection_grids[i] = proj_grid
            Ms[i] = M_val
            overlaps[i] = overlap_val
        catch e
            @warn "Failed to compute overlap for k = $(ks[i]): $(e)"
        end
        next!(progress_computing)
    end

    for i in eachindex(ks) # do not multithread, memory corruption problem
        try
            f = Figure(size = (1000, 2500), resolution=(1000, 2500))
            ax_H = Axis(f[1,1])
            hmap = heatmap!(ax_H, H_list[i]; colormap=Reverse(:gist_heat))
            Colorbar(f[1,2], hmap)
            ax_projection = Axis(f[2,1])
            hmap = heatmap!(ax_projection, projection_grids[i])
            Colorbar(f[2,2], hmap)
            ax_overlap = Axis(f[3,1], title="k = $(round(ks[i]; sigdigits=8)), Overlap = $(round(Ms[i]; sigdigits=4))")
            hmap = heatmap!(ax_overlap, overlaps[i]; colormap=:balance)
            Colorbar(f[3,2], hmap)
            ax_wave = Axis(f[4,1], title="Wavefunction Heatmap for k = $(round(ks[i]; sigdigits=8))")
            hmap = heatmap!(ax_wave, x_grids[i], y_grids[i], Psi2ds[i]; colormap=:balance)
            #plot_boundary!(ax_wave, billiard; fundamental_domain=fundamental_domain) # add later b/c DataAspect from plot_boundary! makes not good image
            Colorbar(f[4,2], hmap)
            colsize!(f.layout, 1, Aspect(4, 1))
            save("Overlap_visualization/$(ks[i])_overlap_w_wavefunctions.png", f)
        catch e
            @warn "Failed to save overlap for k = $(ks[i]): $(e)"
        end
        next!(progress_saving)
    end
end

# M COMPUTATIONS

"""
    compute_overlaps(H_list::Vector{Matrix}, qs_list::Vector{Vector}, ps_list::Vector{Vector}, classical_chaotic_s_vals::Vector, classical_chaotic_p_vals::Vector)

Computes the overlaps of the classical phase space matrix of {+1,-1} depending on whether we are on the chaotic region or regular with the Husimi function matrix.

# Arguments
- `H_list::Vector{Matrix}`:  
  Vector of Husimi functions for each quantum level.
- `qs_list::Vector{Vector}`: Vector of position grid points corresponding to each Husimi function.
- `ps_list::Vector{Vector}`: Vector of momentum grid points corresponding to each Husimi function.
- `classical_chaotic_s_vals::Vector`: Vector of classical chaotic `s` values used to compute the projection grid.
- `classical_chaotic_p_vals::Vector`: Vector of classical chaotic `p` values used to compute the projection grid.

# Returns
- `Ms::Vector{Float64}`: Vector of overlaps for each Husimi function with the classical phase space grid
"""
function compute_overlaps(H_list::Vector{Matrix}, qs_list::Vector{Vector}, ps_list::Vector{Vector}, classical_chaotic_s_vals::Vector, classical_chaotic_p_vals::Vector)
    @assert (length(H_list) == length(qs_list)) && (length(qs_list) == length(ps_list)) "The lists are not the same length"
    Ms = Vector{Union{Float64, Nothing}}(undef, length(H_list))
    Threads.@threads for i in eachindex(qs_list) 
        try
            H = H_list[i]
            qs = qs_list[i]
            ps = ps_list[i]
            proj_grid = classical_phase_space_matrix(classical_chaotic_s_vals, classical_chaotic_p_vals, qs, ps)
            M_val = compute_M(proj_grid, H)
            Ms[i] = M_val
        catch e
            @warn "Failed to compute overlap for idx=$(i): $(e)"
            Ms[i] = nothing
        end
    end
    filter!(x -> !isnothing(x), Ms)
    return convert(Vector{Float64}, Ms)
end

function fraction_of_mixed_states(Ms::Vector; l_bound=-0.8, u_bound=0.8)
    idxs = findall(x -> (x > l_bound)&&(x < u_bound), Ms)
    return length(idxs) / length(Ms)
end

"""
    separate_regular_and_chaotic_states(ks::Vector, H_list::Vector{Matrix}, qs_list::Vector{Vector}, ps_list::Vector{Vector}, classical_chaotic_s_vals::Vector, classical_chaotic_p_vals::Vector, ρ_regular_classic::Float64) :: Tuple{Vector, Vector, Vector}

Separates the regular from the chaotic states based on the classical criterion where the fraction of the number of quantum states classified as regular by their Husimi functions is the same as the classical regular phase space volume.

# Arguments
- `ks::Vector`: Vector of wavenumbers.
- `H_list::Vector{Matrix}`: Vector of Husimi function matrices corresponding to each wavenumber.
- `qs_list::Vector{Vector}`: Vector of position grid points corresponding to each Husimi function.
- `ps_list::Vector{Vector}`: Vector of momentum grid points corresponding to each Husimi function.
- `classical_chaotic_s_vals::Vector`: Vector of classical chaotic `s` values used to compute the projection grid.
- `classical_chaotic_p_vals::Vector`: Vector of classical chaotic `p` values used to compute the projection grid.
- `ρ_regular_classic::Float64`: The volume fraction of the classical phase space.
- `decrease_step_size`: By how much each iteration we decrease the M_thresh until we get the correct volume fraction of the classical phase space.

# Returns
- `Tuple{Vector, Vector, Vector}`: A tuple containing:
- `Ms::Vector`: The thresholds for which calculation were done for plotting purposes.
- `ρs::Vector`: The calculated volumes for each M_thresh.
- `regular_idx::Vector`: The indices of the regular states for which M_thresh produced the correct volume fraction of the classical phase space. This can then be used on the initial `ks` to get the regular ones.
"""
function separate_regular_and_chaotic_states(ks::Vector, H_list::Vector{Matrix}, qs_list::Vector{Vector}, ps_list::Vector{Vector}, classical_chaotic_s_vals::Vector, classical_chaotic_p_vals::Vector, ρ_regular_classic::Float64; decrease_step_size=1e-3)
    @assert (length(H_list) == length(qs_list)) && (length(qs_list) == length(ps_list)) "The lists are not the same length"
    function calc_ρ(M_thresh) # helper for each M_thresh iteration
        local_regular_idx = Threads.Atomic{Vector{Int}}(Vector{Int}())  # thread-safe
        Threads.@threads for i in eachindex(ks) 
            try
                H = H_list[i]
                qs = qs_list[i]
                ps = ps_list[i]
                proj_grid = classical_phase_space_matrix(classical_chaotic_s_vals, classical_chaotic_p_vals, qs, ps)
                M_val = compute_M(proj_grid, H)
                
                if M_val < M_thresh
                    # Append in a thread-safe way
                    Threads.atomic_push!(local_regular_idx, i)
                end
            catch e
                @warn "Failed to compute overlap for k = $(ks[i]): $(e)"
            end
        end
        regular_idx = copy(local_regular_idx[])  # Retrieve final indices list
        return length(regular_idx) / length(ks), regular_idx
    end

    M_thresh = 0.99 #first guess
    ρ_numeric_reg, regular_idx = calc_ρ(M_thresh)
    Ms = Float64[]
    ρs = Float64[]
    push!(Ms, M_thresh) # push the first ones
    push!(ρs, ρ_numeric_reg) # push the first ones
    while ρ_numeric_reg > ρ_regular_classic # as it will only decrease as there will be more chaotic states with the decrease of M_thresh
        M_thresh -= decrease_step_size # slightly decrease the M_thresh
        ρ_numeric_reg, reg_idx_loop = calc_ρ(M_thresh)
        push!(Ms, M_thresh)
        push!(ρs, ρ_numeric_reg)
        if M_thresh < 0.0
            throw(ArgumentError("M_thresh must be positive"))
            break
        end
        if ρ_numeric_reg < ρ_regular_classic # the first one that passes the classical one
            regular_idx = reg_idx_loop
        end
    end
    return Ms, ρs, regular_idx
end

# HISTOGRAMS

"""
    plot_hist_M_distribution(ax::Axis, Ms::Vector; nbins::Int=50, color::Symbol=:blue)

Plots a histogram (pdf) of the distribution of overlap indexes `Ms`

# Arguments
- `ax::Axis`: The axis to plot on.
- `Ms::Vector`: The overlap indexes.
- `nbins::Int`: The number of bins for the histogram.
- `color::Symbol`: The color of the histogram.

# Returns
- `Nothing`
"""
function plot_hist_M_distribution!(ax::Axis, Ms::Vector; nbins::Int=50, color::Symbol=:blue)
    hist = Distributions.fit(StatsBase.Histogram, Ms; nbins=nbins)
    bin_edges = hist.edges[1]
    bin_widths = diff(bin_edges)
    bin_centers = bin_edges[1:end-1] .+ bin_widths / 2
    bin_counts = hist.weights ./ sum(hist.weights) / bin_widths
    barplot!(ax, bin_centers, bin_counts, label="M distribution", color=color, gap=0)
    xlims!(ax, (-1.0, 1.0))
    axislegend(ax, position=:ct)
end

