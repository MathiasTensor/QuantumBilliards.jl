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
function shift_s_vals_poincare_birkhoff(s_vals::Vector{T},s_shift::T,boundary_length::T) where {T<:Real}
    shifted_s_vals=Vector{T}(undef,length(s_vals))
    for i in 1:length(s_vals)
        shifted_s_vals[i]=(s_vals[i]+s_shift)%boundary_length
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
function classical_phase_space_matrix(classical_s_vals::Vector{T},classical_p_vals::Vector{T},qs::Vector{T},ps::Vector{T}) where {T<:Real}
    s_grid_size=length(qs)
    p_grid_size=length(ps)
    # Initialize the projection grid with -1
    projection_grid=fill(-1,s_grid_size,p_grid_size)
    # Use an efficient function to determine the correct grid index wrt mapping (s,p) -> (qs, ps)
    function find_index(val,grid)
        idx=searchsortedlast(grid,val)  # find the index where val would fit in the grid
        return clamp(idx,1,length(grid))  # clamp to ensure it's within grid bounds, otherwise problem
    end
    
    # Mark the projection grid based on classical_s_vals and classical_p_vals
    Threads.@threads for i in 1:length(classical_s_vals)
        classical_s=classical_s_vals[i]
        classical_p=classical_p_vals[i]
        # Find correct grid index for classical_s and classical_p using searchsortedlast. Could still have some artifacts for small dimensional matrices
        s_idx=find_index(classical_s,qs)
        p_idx=find_index(classical_p,ps)
        # Mark the corresponding grid cell as chaotic (1)
        projection_grid[s_idx,p_idx]=1
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
function compute_M(projection_grid::Matrix,H::Matrix)
    @assert size(projection_grid)==size(H) "M computation: H and projection grid must be same size"
    H=H./sum(H) # normalize
    M=sum(broadcast(*,projection_grid,H)) # element wise multiplication
    return M
end

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
function visualize_overlap(projection_grid::Matrix,H::Matrix)
    @assert size(projection_grid)==size(H) "H and projection grid must be same size"
    H=H./sum(H) # normalize
    M=broadcast(*,projection_grid,H) # element wise multiplication
    return M
end

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
function compute_overlaps(H_list::Vector,qs_list::Vector,ps_list::Vector,classical_chaotic_s_vals::Vector,classical_chaotic_p_vals::Vector)
    @assert (length(H_list)==length(qs_list)) && (length(qs_list)==length(ps_list)) "The lists are not the same length"
    Ms=Vector{Union{Float64,Nothing}}(undef,length(H_list))
    @showprogress desc="Computing overlaps..." Threads.@threads for i in eachindex(qs_list) 
        try
            H=H_list[i]
            qs=qs_list[i]
            ps=ps_list[i]
            proj_grid=classical_phase_space_matrix(classical_chaotic_s_vals,classical_chaotic_p_vals,qs,ps)
            M_val=compute_M(proj_grid,H)
            Ms[i]=M_val
        catch e
            @warn "Failed to compute overlap for idx=$(i): $(e)"
            Ms[i]=nothing
        end
    end
    filter!(x -> !isnothing(x),Ms)
    return convert(Vector{Float64},Ms)
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
- `relative_closeness_perc=5`: What is (percentage) the acceptable relative tolerance for the numerical ρ wrt. theoretical ρ.

# Returns
- `Tuple{Vector, Vector, Vector}`: A tuple containing:
- `Ms::Vector`: The thresholds for which calculation were done for plotting purposes.
- `ρs::Vector`: The calculated volumes for each M_thresh.
- `regular_idx::Vector`: The indices of the regular states for which M_thresh produced the correct volume fraction of the classical phase space. This can then be used on the initial `ks` to get the regular ones.
"""
function separate_regular_and_chaotic_states(ks::Vector,H_list::Vector,qs_list::Vector,ps_list::Vector,classical_chaotic_s_vals::Vector,classical_chaotic_p_vals::Vector,ρ_regular_classic::Float64;decrease_step_size=0.05,relative_closeness_perc=5)
    @assert (length(H_list)==length(qs_list)) && (length(qs_list)==length(ps_list)) "Separation of states: The lists are not the same length"
    n=length(ks)
    M_vals=zeros(Float64,n)

    # Precompute M_vals once
    progress=Progress(n;desc="Computing M_vals")
    Threads.@threads for i in 1:n
        H=H_list[i]
        qs=qs_list[i]
        ps=ps_list[i]
        proj_grid=classical_phase_space_matrix(classical_chaotic_s_vals,classical_chaotic_p_vals,qs,ps)
        M_vals[i]=compute_M(proj_grid,H)
        next!(progress)
    end

    function calc_ρ(M_thresh)
        regular_mask=M_vals.<M_thresh
        ρ_numeric_reg=count(regular_mask)/n
        regular_idx=findall(regular_mask)
        return ρ_numeric_reg,regular_idx
    end

    # Initial setup
    M_thresh=0.99
    Ms=Float64[]
    ρs=Float64[]
    ρ_numeric_reg,regular_idx=calc_ρ(M_thresh)

    println("Current ρ_numeric_reg: $(round(ρ_numeric_reg,digits=6))") # for checking purposes
    println("Theoretical ρ_reg: $(round(ρ_regular_classic,digits=6))")
    relative_closeness=abs(ρ_numeric_reg-ρ_regular_classic)/ρ_regular_classic*100
    println("Relative closeness: $(round(relative_closeness,digits=4))%")
    
    prev_num_ρ=ρ_numeric_reg # for preventing inf cycles in outer loop
    push!(Ms,M_thresh)
    push!(ρs,ρ_numeric_reg)
    M_thresh-=decrease_step_size # to not calculate twice in outer loop

    max_iterations=1000  # To prevent infinite loops
    max_inner_iterations=5 # max number of iterations when we are doing small step iterations when close to ρ_regular_classic
    iteration=0
    inner_iterations=0
    inner_iteration=false

    while true
        iteration+=1
        if iteration>max_iterations
            @warn "Maximum iterations reached."
            break
        end

        ρ_numeric_reg,reg_idx_loop=calc_ρ(M_thresh)
        push!(Ms,M_thresh)
        push!(ρs,ρ_numeric_reg)
        regular_idx=reg_idx_loop # this is returned for separation purposes

        println("Current ρ_numeric_reg: $(round(ρ_numeric_reg,digits=6))")
        println("Theoretical ρ_reg: $(round(ρ_regular_classic,digits=6))")
        relative_closeness=abs(ρ_numeric_reg-ρ_regular_classic)/ρ_regular_classic*100
        println("Relative closeness: $(round(relative_closeness,digits=4))%")
        if relative_closeness<5
            break
        end

        # Adjust decrease_step_size if ρ_numeric_reg has gone below ρ_regular_classic
        if ρ_numeric_reg<ρ_regular_classic
            if relative_closeness<5 # if 5% close to ρ_regular_classic execute inner iteration loop
                break
            end
            if inner_iterations>max_inner_iterations
                println("Max inner iterations achieved, breaking")
                break
            end
            decrease_step_size*=0.9
            inner_iteration=true
            println("Adjusted decrease_step_size: $(round(decrease_step_size, digits=6))")
            inner_iterations+=1
        end
        if (abs(prev_num_ρ-ρ_numeric_reg)/ρ_numeric_reg*100)<1 && (relative_closeness<relative_closeness_perc)
            println("No more precise, breaking!")
            break
        end
        prev_num_ρ=ρ_numeric_reg
        # Update M_thresh based on outer/inner loop
        if !inner_iteration
            M_thresh-=decrease_step_size # depedning on the inenr/outer loop
        else
            M_thresh+=decrease_step_size # depedning on the inenr/outer loop
            inner_iteration=false
        end
        if M_thresh<=0.0
            throw(ArgumentError("M_thresh must be positive"))
        end
    end
    return Ms,ρs,regular_idx
end