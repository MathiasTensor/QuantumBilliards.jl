using JLD2, Makie, ProgressMeter

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
    ps_list::Vector{Vector}, classical_chaotic_s_vals::Vector, classical_chaotic_p_vals::Vector; save_path::String = "Overlap_visualization")
    if !isdir(save_path)
        mkdir(save_path)
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
            max_value = maximum(abs, overlaps[i])
            hmap = heatmap!(ax_overlap, overlaps[i]; colormap=:balance,colorrange=(-max_value, max_value))
            Colorbar(f[3,2], hmap)
            colsize!(f.layout, 1, Aspect(3, 1))
            save("$save_path/$(ks[i])_overlap_w_wavefunctions.png", f)
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
    b = 5.0, inside_only = true, fundamental_domain = true, memory_limit = 10.0e9, save_path::String = "Overlap_visualization") where {Bi<:AbsBilliard, Ba<:AbsBasis}
    if !isdir(save_path)
        mkdir(save_path)
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
            max_value = maximum(abs, overlaps[i])
            hmap = heatmap!(ax_overlap, overlaps[i]; colormap=:balance,colorrange=(-max_value, max_value))
            Colorbar(f[3,2], hmap)
            ax_wave = Axis(f[4,1], title="Wavefunction Heatmap for k = $(round(ks[i]; sigdigits=8))")
            hmap = heatmap!(ax_wave, x_grids[i], y_grids[i], Psi2ds[i]; colormap=:balance)
            #plot_boundary!(ax_wave, billiard; fundamental_domain=fundamental_domain) # add later b/c DataAspect from plot_boundary! makes not good image
            Colorbar(f[4,2], hmap)
            colsize!(f.layout, 1, Aspect(4, 1)) 
            save("$save_path/$(ks[i])_overlap_w_wavefunctions.png", f)
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
- `decrease_step_size=0.05`: By how much each iteration we decrease the M_thresh until we get the correct volume fraction of the classical phase space.

# Returns
- `Tuple{Vector, Vector, Vector}`: A tuple containing:
- `Ms::Vector`: The thresholds for which calculation were done for plotting purposes.
- `ρs::Vector`: The calculated volumes for each M_thresh.
- `regular_idx::Vector`: The indices of the regular states for which M_thresh produced the correct volume fraction of the classical phase space. This can then be used on the initial `ks` to get the regular ones.
"""
function separate_regular_and_chaotic_states(
    ks::Vector,
    H_list::Vector{Matrix},
    qs_list::Vector{Vector},
    ps_list::Vector{Vector},
    classical_chaotic_s_vals::Vector,
    classical_chaotic_p_vals::Vector,
    ρ_regular_classic::Float64;
    decrease_step_size=0.05
)
    @assert (length(H_list) == length(qs_list)) && (length(qs_list) == length(ps_list)) "The lists are not the same length"

    function calc_ρ(M_thresh)
        n = length(ks)
        regular_mask = zeros(Bool, n)
        progress = Progress(n; desc="Calculating for M_thresh = $(round(M_thresh, digits=3))")
        nthreads = Threads.nthreads()
        # Initialize per-thread caches
        thread_caches = [Dict{UInt64, Any}() for _ in 1:nthreads]
        Threads.@threads for i in 1:n
            try
                thread_id = Threads.threadid()
                cache = thread_caches[thread_id]
                H = H_list[i]
                qs = qs_list[i]
                ps = ps_list[i]
                # Create a hash key based on the size of H, qs, and ps
                key = hash((size(H), qs, ps))
                if haskey(cache, key)
                    proj_grid = cache[key]
                else
                    proj_grid = classical_phase_space_matrix(classical_chaotic_s_vals, classical_chaotic_p_vals, qs, ps)
                    cache[key] = proj_grid
                end
                M_val = compute_M(proj_grid, H)
                if M_val < M_thresh
                    regular_mask[i] = true
                end
            catch e
                @warn "Failed to compute overlap for k = $(ks[i]): $(e)"
            end
            next!(progress)
        end

        ρ_numeric_reg = count(regular_mask) / n
        regular_idx = findall(regular_mask)
        return ρ_numeric_reg, regular_idx
    end

    # Initial setup
    M_thresh = 0.99
    Ms = Float64[]
    ρs = Float64[]
    ρ_numeric_reg, regular_idx = calc_ρ(M_thresh)

    println("Current ρ_numeric_reg: $(round(ρ_numeric_reg, digits=6))") # for checking purposes
    println("Theoretical ρ_reg: $(round(ρ_regular_classic, digits=6))")
    relative_closeness = abs(ρ_numeric_reg - ρ_regular_classic) / ρ_regular_classic * 100
    println("Relative closeness: $(round(relative_closeness, digits=4))%")
    
    prev_num_ρ = ρ_numeric_reg # for preventing inf cycles in outer loop
    push!(Ms, M_thresh)
    push!(ρs, ρ_numeric_reg)
    M_thresh -= decrease_step_size # to not calculate twice in outer loop

    max_iterations = 1000  # To prevent infinite loops
    iteration = 0
    inner_iteration = false

    while true
        iteration += 1
        if iteration > max_iterations
            @warn "Maximum iterations reached."
            break
        end

        ρ_numeric_reg, reg_idx_loop = calc_ρ(M_thresh)
        push!(Ms, M_thresh)
        push!(ρs, ρ_numeric_reg)
        regular_idx = reg_idx_loop # this is returned for separation purposes

        println("Current ρ_numeric_reg: $(round(ρ_numeric_reg, digits=6))")
        println("Theoretical ρ_reg: $(round(ρ_regular_classic, digits=6))")
        relative_closeness = abs(ρ_numeric_reg - ρ_regular_classic) / ρ_regular_classic * 100
        println("Relative closeness: $(round(relative_closeness, digits=4))%")

        # Adjust decrease_step_size if ρ_numeric_reg has gone below ρ_regular_classic
        if ρ_numeric_reg < ρ_regular_classic
            decrease_step_size *= 0.9
            inner_iteration = true
            println("Adjusted decrease_step_size: $(round(decrease_step_size, digits=6))")
        end
        if (abs(prev_num_ρ - ρ_numeric_reg)/ρ_numeric_reg * 100) < 1 && (relative_closeness < 5)
            println("No more precise, breaking!")
            break
        end
        prev_num_ρ = ρ_numeric_reg
        # Update M_thresh based on outer/inner loop
        if !inner_iteration
            M_thresh -= decrease_step_size
        else
            M_thresh += decrease_step_size
            inner_iteration = false
        end
        if M_thresh <= 0.0
            throw(ArgumentError("M_thresh must be positive"))
        end
    end

    return Ms, ρs, regular_idx
end

"""
    separate_ks_by_classical_indices(ks::Vector, regular_idx::Vector{Int})

Separate `ks` values into regular and chaotic based on the indices of regular states.

# Arguments:
- `ks::Vector`: A vector containing the `ks` wavenumbers.
- `regular_idx::Vector{Int}`: A vector of indices corresponding to the regular states.

# Returns:
- `ks_regular::Vector`: The subset of `ks` corresponding to the regular states.
- `ks_chaotic::Vector`: The subset of `ks` corresponding to the chaotic states.
"""
function separate_ks_by_classical_indices(ks::Vector, regular_idx::Vector{Int})
    ks_regular = ks[regular_idx]
    all_indices = Set(1:length(ks))  # Set of all indices
    chaotic_idx = sort(collect(setdiff(all_indices, regular_idx)))  # Find chaotic indices
    ks_chaotic = ks[collect(chaotic_idx)]  # Extract ks_chaotic
    return ks_regular, ks_chaotic
end

"""
    separate_Hs_by_classical_indices(Hs_list::Vector, regular_idx::Vector{Int})

Separates the Husimi function matrices `Hs_list` into regular and chaotic based on the indices of regular states.

# Arguments:
- `Hs_list::Vector{Matrix}`: A vector containing the Husimi function matrices.
- `regular_idx::Vector{Int}`: A vector of indices corresponding to the regular states.

# Returns:
- `Hs_regular::Vector{Matrix}`: A vector containing the Husimi function matrices corresponding to the regular states.
- `Hs_chaotic::Vector{Matrix}`: A vector containing the Husimi function matrices corresponding to the chaotic states.
"""
function separate_Hs_by_classical_indices(Hs_list::Vector, regular_idx::Vector{Int})
    Hs_regular = Hs_list[regular_idx]
    all_indices = Set(1:length(Hs_list))
    chaotic_idx = sort(collect(setdiff(all_indices, regular_idx)))  # Find chaotic indices
    Hs_chaotic = Hs_list[collect(chaotic_idx)]  # Extract Hs_chaotic
    return Hs_regular, Hs_chaotic
end

"""
    separate_ks_and_Hs_by_classical_indices(ks::Vector, Hs_list::Vector, regular_idx::Vector{Int})

 Separates `ks` and `Hs_list` into regular and chaotic based on the indices of regular states.

 # Arguments:
 - `ks::Vector`: A vector containing the `ks` wavenumbers.
 - `Hs_list::Vector{Matrix}`: A vector containing the Husimi function matrices.
 - `regular_idx::Vector{Int}`: A vector of indices corresponding to the regular states.

 # Returns:
 - `ks_regular::Vector`: The subset of `ks` corresponding to the regular states.
 - `ks_chaotic::Vector`: The subset of `ks` corresponding to the chaotic states.
 - `Hs_regular::Vector{Matrix}`: The subset of `Hs_list` corresponding to regular states.
 - `Hs_chaotic::Vector{Matrix}`: The subset of `Hs_list` corresponding to chaotic states.
"""
function separate_ks_and_Hs_by_classical_indices(ks::Vector, Hs_list::Vector, regular_idx::Vector{Int})
    ks_regular = ks[regular_idx]
    Hs_regular = Hs_list[regular_idx]
    all_indices = Set(1:length(ks))  # Set of all indices
    chaotic_idx = sort(collect(setdiff(all_indices, regular_idx)))  # Find chaotic indices
    ks_chaotic = ks[collect(chaotic_idx)]
    Hs_chaotic = Hs_list[collect(chaotic_idx)]
    return ks_regular, ks_chaotic, Hs_regular, Hs_chaotic
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
function plot_hist_M_distribution!(ax::Axis, Ms::Vector; nbins::Int=50, color::Symbol=:lightblue)
    hist = Distributions.fit(StatsBase.Histogram, Ms; nbins=nbins)
    bin_centers = (hist.edges[1][1:end-1] .+ hist.edges[1][2:end]) / 2
    bin_counts = hist.weights ./ sum(hist.weights) / diff(hist.edges[1])[1]
    barplot!(ax, bin_centers, bin_counts, label="M distribution", color=color, gap=0, strokecolor=:black, strokewidth=1)
    xlims!(ax, (-1.0, 1.0))
    axislegend(ax, position=:ct)
end

"""
    fraction_of_mixed_states(Ms::Vector; l_bound=-0.8, u_bound=0.8)

Computes the fraxtion of mixed states as the number of Ms between l_bound and u_bound.

# Arguments
- `Ms::Vector`: The overlap indexes.
- `l_bound::Float64`: Lower bound for the fraction of mixed states.
- `u_bound::Float64`: Upper bound for the fraction of mixed states.

 # Returns
- `<:Real`: The fraction of mixed states.
"""
function fraction_of_mixed_states(Ms::Vector; l_bound=-0.8, u_bound=0.8)
    idxs = findall(x -> (x > l_bound)&&(x < u_bound), Ms)
    return length(idxs) / length(Ms)
end

"""
    get_mixed_states(Ms::Vector; l_bound=-0.8, u_bound=0.8)

Gives us overlap indexes of mixed states between l_bound and u_bound and their corresponding ks.

# Arguments
- `Ms::Vector`: The overlap indexes.
- `ks::Vector`: The wavenumbers.
- `l_bound::Float64`: Lower bound for the mixed states.
- `u_bound::Float64`: Upper bound for the mixed states.

 # Returns
- `Vector{<:Real}`: Vector of overlap indexes of mixed states.
- `Vector{<:Real}`: Vector of wavenumbers corresponding to the mixed states.
"""
function get_mixed_states(Ms::Vector, ks::Vector; l_bound=-0.8, u_bound=0.8)
    idxs = findall(x -> (x > l_bound)&&(x < u_bound), Ms)
    return Ms[idxs], ks[idxs]
end

"""
    coefficient_of_fraction_of_eigenstates_vs_k(χ_Ms::Vector{T}, ks_M::Vector{T}) where {T<:Real}

Computes the coefficient of the fraction of mixed eigenstates (ζ) for the relation M_mixed=k^-α via the built in linear fitting.

# Arguments
- `χ_Ms::Vector{T}`: The fraction of mixed states for each k in ks.
- `ks_M::Vector{T}`: The wavenumbers that accompany the mixed states.

 # Returns
 - `<:Real`: The coefficient of the fraction of mixed eigenstates (ζ).
"""
function coefficient_of_fraction_of_mixed_eigenstates_vs_k(χ_Ms::Vector{T}, ks_M::Vector{T}) where {T<:Real}
    # Make the log-log data of the χ_Ms and ks
    log_χ_Ms = log.(χ_Ms)
    log_ks_M = log.(ks_M)
    # Built in lapack linear fitting
    ζ = -(log_ks_M / log_χ_Ms)
    return ζ
end

"""
    plot_fraction_of_mixed_eigenstates_vs_k(ax::Axis, χ_Ms::Vector{T}, ks::Vector{T}) where {T<:Real}

PLots the fraction of mixed states vs. the wavenumber k. We expect a power-law decay of the type log(χ) = -ζ*log(k). Also plots the fitted line with the ζ coefficient from linear fitting.

# Arguments
- `ax::Axis`: The Makie axis object to plot on.
- `χ_Ms::Vector{T}`: The fraction of mixed states for each k in ks.
- `ks_M::Vector{T}`: The wavenumbers that accompany the mixed states.

# Returns
- `Nothing`
"""
function plot_fraction_of_mixed_eigenstates_vs_k(ax::Axis, χ_Ms::Vector{T}, ks_M::Vector{T}) where {T<:Real}
    log_ks_M = log.(ks_M)
    log_χ_Ms = log.(χ_Ms)
    # First plot the scatter log-log plot then fit
    ζ = coefficient_of_fraction_of_mixed_eigenstates_vs_k(χ_Ms, ks_M)
    scatter!(ax, log_ks_M, log_χ_Ms, label="Numerical data", color=:blue, marker=:circle, msize=4)
    lines!(ax, log_ks_M, -ζ*log_ks, label="ζ=$(round(ζ; digits=3))", color=:red, linewidth=2) # do not forget the minus here. The fraction of mixed eigenstates should decay with increasing k
end

