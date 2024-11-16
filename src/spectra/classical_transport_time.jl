


"""
    generate_intervals_from_limits(limits::Vector{T}; numerical_cutoff=1e-2) where {T<:Real}

Convenience function that generates intervals from provided limits.

# Arguements
- `limits::Vector{<:Real}`: A vector of elements representing the lower and upper bounds of the interval in that order. Each time 2 are taken to construct the lower and upper bounds of an interval.
- `numerical_cutoff::Real=1e-2`: A numerical value used to prevent initial conditions that have ill defined trajectory behaviouir. Can happen id there are sharp changes in the geometry.

# Returns
- `Vector{Vector{T}}`: A vector of vectors, where each inner vector contains two elements representing the lower and upper bounds of an interval in that order.
"""
function generate_intervals_from_limits(limits::Vector{T}; numerical_cutoff=1e-2) where {T<:Real}
    sort!(limits)
    return [[limits[i]+numerical_cutoff, limits[i+1]-numerical_cutoff] for i in 1:(length(limits)-1)]
end

"""
    generate_p_0_chaotic_init_conditions(interval::Vector{T}; N_total::Integer=10_000) where {T<:Real}

Uniformly distributes the initial conditions in the supplied interval.

# Arguments
- `interval::Vector{<:Real}`: A vector of two elements representing the lower and upper bounds of the interval in that order.
- `N_total::INteger=10_000`: The total number of initial conditions to generate.

# Returns
- `Vector{T}`: A vector of initial conditions uniformly distributed within the supplied interval.
"""
function generate_p_0_chaotic_init_conditions(interval::Vector{T}; N_total::Integer=10_000) where {T<:Real}
    interval_length = abs(interval[2] - interval[1])
    s_vals = if N_total > 1
        range(interval[1], interval[2], length=N_total)
    else
        [interval[1]]
    end
    return [s for s in s_vals]
end

"""
    generate_p_0_chaotic_init_conditions(intervals::Vector{Vector{T}}; N_total::Integer=10_000) where {T<:Real}

Uniformly distributes the initial conditions for all intervals provided. All the intervals must have the first value as a lower bound and second upper bound. The result is concatenated into a single vector.

# Arguments
- `intervals::Vector{Vector{T}}`: A vector of vectors, where each inner vector contains two elements representing the lower and upper bounds of an interval in that order.
- `N_total::Integer=10_000`: The total number of initial conditions to generate.

# Returns
- `Vector{T}`: A vector of initial conditions uniformly distributed within all the intervals provided.
"""
function generate_p_0_chaotic_init_conditions(intervals::Vector{Vector{T}}; N_total::Integer=10_000) where {T<:Real}
    interval_lengths = [abs(interval[2] - interval[1]) for interval in intervals]
    total_length = sum(interval_lengths)
    fractions = [len / total_length for len in interval_lengths]
    N_interval_points = [Int(floor(N_total * fraction)) for fraction in fractions] # initial allocation of points per interval
    remaining_points = N_total - sum(N_interval_points)
    if remaining_points != 0 # Distribute any remaining points to ensure the total matches N_total
        sorted_indices = sortperm(fractions, rev=true)  # Sort intervals by largest fraction
        for i in 1:abs(remaining_points)
            idx = sorted_indices[(i - 1) % length(sorted_indices) + 1]
            N_interval_points[idx] += sign(remaining_points)
        end
    end
    init_conditions = Vector{T}()
    for i in eachindex(intervals)
        interval = intervals[i]
        N_points = N_interval_points[i]
        s_vals = N_points > 1 ? range(interval[1], interval[2], length=N_points) : [interval[1]]
        append!(init_conditions, s_vals)
    end
    return init_conditions
end

"""
    convert_p_0_chaotic_init_conditions_to_cartesian(init_conditions::Vector{T}, from_bcoords_to_cartesian::Function; extra_args...) where {T<:Real}

Transforms Poincare-Birkhoff initial conditions to Cartesian using the provided function. An example function is provided in Example.

# Example 
```julia
# Convenience function from Poincare-Birkhoff to Cartesian. Requires the dependancy DynamicalBilliards.
function from_bcoords_to_cartesian_mushroom(s::T, p::T, billiard) :: Vector{Tuple{SVector{2, T}, SVector{2, T}}} where {T<:Real}
    return DynamicalBilliards.from_bcoords(s, p, billiard)
end
```

# Arguments
- `init_conditions::Vector{T}`: A vector of initial conditions in Poincare-Birkhoff coordinates. This is just the s value in this case since the p's are fixed in PB coordinates.
- `from_bcoords_to_cartesian::Function`: A function that takes s and p as arguments and returns a tuple of Cartesian coordinates. I's formal signature is: `(s::T, p::T; extra_args...) ->  Vector{Tuple{SVector{2, T}, SVector{2, T}}}`, that is a `Vector` of (pos, vel) which are both `SVector{2,T}`. The extra arguments are there if one needs additional geometry information for creating this wrapper function.
- `extra_args...`: A vector of extra arguments (usually geometry) that are the extra parameters for `from_bcoords_to_cartesian`

# Returns
- `Vector{Tuple{SVector{2, T}, SVector{2, T}}}`: A vector of Cartesian coordinates and Cartesian momenta corresponding to the provided Poincare-Birkhoff initial conditions.
"""
function convert_p_0_chaotic_init_conditions_to_cartesian(init_conditions::Vector{T}, from_bcoords_to_cartesian::Function, extra_args...) where {T<:Real}
    cartesian_conditions = Vector{Tuple{SVector{2, Float64}, SVector{2, Float64}}}(undef, length(init_conditions))
    Threads.@threads for i in eachindex(init_conditions)
        s = init_conditions[i]
        p = 0.0  # Fixed p = 0 for all initial conditions (perpendicular velocity)
        pos, vel = from_bcoords_to_cartesian(s, p, extra_args...)
        cartesian_conditions[i] = (pos, vel)
    end
    return cartesian_conditions
end

"""
    simulate_trajectories(cartesian_conditions::Vector{Tuple{SVector{2, T}, SVector{2, T}}}, boundarymap_function::Function; 
    extra_args..., N_collisions::Int=1_000_000) where {T<:Real}

Simulates the boundary map of particles in a billiard geometry using a provided `boundarymap_function`. 

# Arguments
- `cartesian_conditions::Vector{Tuple{SVector{2, T}, SVector{2, T}}}`: A vector of initial conditions in Cartesian coordinates, where each element is `(pos, vel)`.
- `boundarymap_function::Function`: A function that computes the boundary map for a particle. Its signature should be:
  `boundarymap_function(pos::SVector{2, T}, vel::SVector{2, T}, N_collisions::Int; extra_args...) -> Vector{Tuple{T,T}}`, where each element is a`(s,p)` pair for a trajectory that starts at `pos,vel`.
- `extra_args...`: Additional arguments required by the `boundarymap_function`.
- `N_collisions::Integer=1_000_000`: Number of boundary collisions to simulate for each trajectory.

# Returns
- `s_vals_all::Vector{Vector{T}}`: A vector of `s` values for each trajectory across collisions.
- `p_vals_all::Vector{Vector{T}}`: A vector of `p` values for each trajectory across collisions.

# Example

Here’s an example wrapper function for `boundarymap` from the `DynamicalBilliards` library:
```julia
# A wrapper around `DynamicalBilliards.boundarymap`
function boundarymap_wrapper(pos::SVector{2, T}, vel::SVector{2, T}, N_collisions::Int, billiard) where {T<:Real}
    particle = Particle(pos, vel)
    bmap, _ = DynamicalBilliards.boundarymap(particle, billiard, N_collisions)
    return bmap
end
"""
function simulate_trajectories(cartesian_conditions::Vector{Tuple{SVector{2, T}, SVector{2, T}}}, boundarymap_function::Function, extra_args...; 
    N_collisions::Int=1_000_000) where {T<:Real}
    total_particles = length(cartesian_conditions)
    s_vals_all = Vector{Vector{T}}(undef, total_particles)
    p_vals_all = Vector{Vector{T}}(undef, total_particles)
    progress = Progress(total_particles, desc="Simulating trajectories")
    Threads.@threads for particle_idx in eachindex(cartesian_conditions)
        pos, vel = cartesian_conditions[particle_idx]
        success = false
        while !success
            try
                bmap = boundarymap_function(pos, vel, N_collisions, extra_args...)
                s_vals_all[particle_idx] = [bmap[i][1] for i in eachindex(bmap)]
                p_vals_all[particle_idx] = [bmap[i][2] for i in eachindex(bmap)]
                success = true
            catch e
                println("Warning: Error encountered for particle $particle_idx, retrying...")
                println("ERROR: ", e)
            end
        end
        next!(progress)
    end
    return s_vals_all, p_vals_all
end

"""
    calculate_p2_averages(p_vals_all::Vector{Vector{T}}, N_collisions::Int) where {T<:Real}

Calculates the average p^2 value for all particles for each collision.

# Arguements
- `p_vals_all::Vector{Vector{T}}`: A vector of vectors where each element is a vector of p^2 values for a particle across collisions.

# Returns
- `iterations::Vector{T}`: A vector of integers representing the collision numbers.
- `p_squared_averages::Vector{T}`: A vector of numbers representing the average p^2 values for each collision.
"""
function calculate_p2_averages(p_vals_all::Vector{Vector{T}}) where {T<:Real}
    N_collisions = length(p_vals_all[1])
    p_squared_averages = Vector{T}(undef, N_collisions)
    for collision_idx in 1:N_collisions
        p_values = [p_vals_all[particle_idx][collision_idx] for particle_idx in eachindex(p_vals_all)]
        p_squared_averages[collision_idx] = sum(p^2 for p in p_values)/length(p_values)
    end
    return collect(1:N_collisions), p_squared_averages
end

# SIMPLE PLOTTING WRAPPER
function plot_p2_stats!(ax::Axis, p2_averages::Vector{T}) where {T<:Real}
    iterations = 1:length(p2_averages)
    scatter!(ax, log10.(iterations), p2_averages, markersize=4, color=:blue)
end