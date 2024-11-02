using ProgressMeter


"""
    S(bmap::Vector, L::T; max_collisions::Int=10^8, num_bins::Int=1000) where {T<:Real}

The S-plot matrix (of size (num_bins+1)x(num_bins+1)) along with the s and p edges for heatmap plotting. Good for showing stickiness regions.

# Arguments
- `bmap::Vector{SVector}`: A `Vector` of `SVector(s, p)` coordinates representing the trajectories for each collision up to `max_collisions`.
- `L::T`: The total length of the phase space.
- `max_collisions::Int=10^8`: (Optional) The maximum number of collisions to process.
- `num_bins::Int=1000`: (Optional) The number of bins in the phase space grid, same in s and p direction.

# Returns
- `S_grid::Matrix`: The S-plot matrix of size (num_bins+1)x(num_bins+1)
- `s_edges::Vector`: The edges of the phase space grid in the s direction of length (num_bins+1).
- `p_edges::Vector`: The edges of the phase space grid in the p direction of length (num_bins+1).
"""
function S(bmap::Vector, L::T; max_collisions::Int=10^8, num_bins::Int=1000) where {T<:Real}
    # phase space grid (s, p)
    total_arclength = L
    s_edges = range(0, total_arclength; length = num_bins + 1)
    p_edges = range(-1.0, 1.0; length = num_bins + 1)
    
    # Initialize grid and recurrence tracking
    S_grid = fill(0.0, num_bins, num_bins)       # Fill with 0.0 to represent regular regions
    recurrence_times = Dict{Tuple{Int, Int}, Vector{Int}}()  # Main dictionary to accumulate recurrence times

    # Set up thread-local dictionaries b/c we the dictionary is not thread safe so we need local storages with atomic additions and subsequent merging
    local_recurrence_times = Vector{Dict{Tuple{Int, Int}, Vector{Int}}}(undef, Threads.nthreads())
    local_last_visit = Vector{Dict{Tuple{Int, Int}, Int}}(undef, Threads.nthreads())
    for i in 1:Threads.nthreads()
        local_recurrence_times[i] = Dict{Tuple{Int, Int}, Vector{Int}}()
        local_last_visit[i] = Dict{Tuple{Int, Int}, Int}()
    end

    # Progress bar
    progress = Progress(max_collisions, desc="Processing collisions")
    counter = Threads.Atomic{Int}(0)

    Threads.@threads for collision in eachindex(bmap)
        s, p_coord = bmap[collision]
        
        # Determine cell indices in the grid
        #s_idx = findfirst(x -> x >= s, s_edges) - 1 # the previous one is therefore the wanted one
        #p_idx = findfirst(x -> x >= p_coord, p_edges) - 1 # the previous one is therefore the wanted one
        s_idx = findfirst(x -> x >= s, s_edges)
        p_idx = findfirst(x -> x >= p_coord, p_edges)

        # Check if `findfirst` returned `nothing` and adjust accordingly. This is a hack since s or p_coord can go slightly out of bounds (now always)
        s_idx = isnothing(s_idx) ? num_bins : max(1, s_idx - 1)  # Use last bin if out of bounds, or decrement by 1 as logic above (uncommented)
        p_idx = isnothing(p_idx) ? num_bins : max(1, p_idx - 1)  # Same logic for p

        # Check if the particle is within the grid bounds -> sanity check
        if s_idx in 1:num_bins && p_idx in 1:num_bins
            cell = (s_idx, p_idx)
            thread_id = Threads.threadid() 
            # Calculate recurrence time if this cell was visited before on this thread
            if haskey(local_last_visit[thread_id], cell)
                #recurrence_time = collision - local_last_visit[thread_id][cell]
                #push!(get!(local_recurrence_times[thread_id], cell, []), recurrence_time)
                last_time = local_last_visit[thread_id][cell]
                if last_time !== nothing
                    recurrence_time = collision - last_time
                    push!(get!(local_recurrence_times[thread_id], cell, []), recurrence_time)
                end
            end
            # Update the last visit time for the current cell on this thread
            local_last_visit[thread_id][cell] = collision
        end

        # Update only after 100 collisions since bloat otherwise
        if collision % 100 == 0
            Threads.atomic_add!(counter, 100)
            ProgressMeter.update!(progress, counter[])
        end
    end

    ProgressMeter.update!(progress, max_collisions) # Ensure progress reaches 100%
    # Combine thread-local recurrence times into the main dictionary
    for t in 1:Threads.nthreads()
        for (cell, times) in local_recurrence_times[t]
            recurrence_times[cell] = get!(recurrence_times, cell, [])  # Ensure entry exists
            append!(recurrence_times[cell], times)
        end
    end

    # Calculate S for each cell using combined recurrence times
    for ((s_idx, p_idx), times) in recurrence_times
        if !isempty(times)
            # Compute standard deviation and mean of recurrence times
            sigma = std(times)
            τ = mean(times)
            S_grid[s_idx, p_idx] = sigma / τ # store as 2d grid
        end
    end
    return S_grid, collect(s_edges), collect(p_edges) 
end

"""
    S(bmap_func::Function; max_collisions::Int=10^8, num_bins::Int=1000)

High-level wrapper for the S-plot function. It just takes the function that for a given initial condition and geometry generates a chaotic trajectory. The function's signature is given in Arguments. Use this one if the chaotic trajectory spans limiting s values of the phase space plot.

# Arguments
- `bmap_func::Function`: Signature: `(max_collisions::Int) -> Vector{SVector{2,<:Real}}`. A function that takes the number of collisions as input and returns a `Vector` of `SVector(s, p)` coordinates representing the trajectories for each collision up to `max_collisions`.
- `max_collisions::Int=10^8`: (Optional) The maximum number of collisions to process.
- `num_bins::Int=1000`: (Optional) The number of bins in the phase space grid, same in s and p direction.

# Returns
- `S_grid::Matrix`: The S-plot matrix of size (num_bins+1)x(num_bins+1)
- `s_edges::Vector`: The edges of the phase space grid in the s direction of length (num_bins+1).
- `p_edges::Vector`: The edges of the phase space grid in the p direction of length (num_bins+1).

# Example
Here is an example of such a wrapper function for the mushroom billiard using `DynamicalBilliards.jl` and it's `MushroomTools`.

```julia
using DynamicalBilliards

function bmap_wrapper(l,w,r)
    p = MushroomTools.randomchaotic(l,w,r)
    mushroom = billiard_mushroom(l,w,r)
    bmap_func = function (max_collisions)
        bmap, _ = boundarymap(p, mushroom, max_collisions)
        return bmap
    end
    return bmap_func
end
```
"""
function S(bmap_func::Function; max_collisions::Int=10^8, num_bins::Int=1000)
    println("Processing collisions...")
    @time bmap = bmap_func(max_collisions)
    println("Done w/ collisions...")
    L, _ = findmax(s_p[1] for s_p in bmap) # finds largest s if chaotic spans 0 -> L, equal to whole s length
    S(bmap, L, max_collisions=max_collisions, num_bins=num_bins)
end

"""
    S(bmap_func::Function, L::T; max_collisions::Int=10^8, num_bins::Int=1000) where {T<:Real}

High-level wrapper for the S-plot function. It just takes the function that for a given initial condition and geometry generates a chaotic trajectory. The function's signature is given in Arguments. 

# Arguments
- `bmap_func::Function`: Signature: `(max_collisions::Int) -> Vector{SVector{2,<:Real}}`. A function that takes the number of collisions as input and returns a `Vector` of `SVector(s, p)` coordinates representing the trajectories for each collision up to `max_collisions`.
- `L::T`: The total length of the phase space.
- `max_collisions::Int=10^8`: (Optional) The maximum number of collisions to process.
- `num_bins::Int=1000`: (Optional) The number of bins in the phase space grid, same in s and p direction.

# Returns
- `S_grid::Matrix`: The S-plot matrix of size (num_bins+1)x(num_bins+1)
- `s_edges::Vector`: The edges of the phase space grid in the s direction of length (num_bins+1).
- `p_edges::Vector`: The edges of the phase space grid in the p direction of length (num_bins+1).

# Example
Here is an example of such a wrapper function for the mushroom billiard using `DynamicalBilliards.jl` and it's `MushroomTools`.

```julia
using DynamicalBilliards

function bmap_wrapper(l,w,r)
    p = MushroomTools.randomchaotic(l,w,r)
    mushroom = billiard_mushroom(l,w,r)
    bmap_func = function (max_collisions)
        bmap, _ = boundarymap(p, mushroom, max_collisions)
        return bmap
    end
    return bmap_func
end
```
"""
function S(bmap_func::Function, L::T; max_collisions::Int=10^8, num_bins::Int=1000) where {T<:Real}
    println("Processing collisions...")
    @time bmap = bmap_func(max_collisions)
    println("Done w/ collisions...")
    S(bmap, L, max_collisions=max_collisions, num_bins=num_bins)
end

"""
    plot_S_heatmap!(f::Figure, S_grid::Matrix, s_edges::Vector, p_edges::Vector)

Plots the S matrix with the s and p edges w/ a Colorbar.

# Arguments
- `f::Figure`: The figure to plot on.
- `S_grid::Matrix`: The S-plot matrix of size (num_bins+1)x(num_bins+1).
- `s_edges::Vector`: The edges of the phase space grid in the s direction.
- `p_edges::Vector`: The edges of the phase space grid in the p direction.

# Returns
- `Nothing`
"""
function plot_S_heatmap!(f::Figure, S_grid::Matrix, s_edges::Vector, p_edges::Vector; additional_text::String="")
    ax = Axis(f[1, 1], title="S-plot " * additional_text, xlabel="s", ylabel="p")
    hmap = heatmap!(ax, s_edges, p_edges, S_grid; colormap=Reverse(:gist_heat), nan_color=:white, colorrange=(0, 15))
    Colorbar(f[1, 2], hmap, ticks=0:1:15)
    colsize!(f.layout, 2, Relative(1/10))
end

"""
    plot_S_heatmap!(f::Figure, S_grid::Vector, s_edges::Vector, p_edges::Vector; max_cols::Int=4)

Plots the vector of S matrices with the the index wise corresponding s and p edges w/ a Colorbar.

# Arguements
- `f::Figure`: The figure to plot on.
- `S_grids::Vector{Matrix}`: The vector of S-plot matrices of size (num_bins+1)x(num_bins+1).
- `s_edges::Vector{Vector}`: The edges of the phase space grid in the s direction for each S_grid.
- `p_edges::Vector{Vector}`: The edges of the phase space grid in the p direction for each S_grid.
- `max_cols::Int=4`: (Optional) The maximum number of columns to plot in the figure.

# Returns
- `Nothing`
"""
function plot_S_heatmaps!(f::Figure, S_grids::Vector, s_edges::Vector, p_edges::Vector; max_cols::Int=4, additional_texts::Vector{String}=fill("", length(S_grids)))
    @assert (length(S_grids) == length(s_edges)) && (length(S_grids) == length(p_edges)) "All must be constructed with same num_bind param"
    row = 1
    col = 1
    heatmaps = []
    for idx in eachindex(S_grids) 
        ax = Axis(f[row, col], title="S-plot " * additional_texts[idx], xlabel="s", ylabel="p")
        hmap = heatmap!(ax, S_grids[idx], s_edges[idx], p_edges[idx]; colormap=Reverse(:gist_heat), nan_color=:white, colorrange=(0, 15))
        push!(heatmaps, hmap)
        col += 1
        if col > max_cols
            col = 1
            row += 1
        end
    end
    colorbar = Colorbar(fig[row + 1, 1:max_cols], heatmaps[1], vertical=:false)
    rowgap!(fig.layout, row + 1, Relative(1/10))
end