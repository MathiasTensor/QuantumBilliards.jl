"""
    get_virtual_line_segment_as_vectors(billiard::Bi; center::SVector=SVector{2,Float64}(0.0,0.0))

Extracts and returns the non-origin vectors from the virtual line segments in the fundamental boundary of the given billiard. 
Optionally, the vectors are shifted by the specified center.

# Arguments
- `billiard::Bi`: The billiard object, where `Bi` is a subtype of `AbsBilliard`.
- `center::SVector`: Optional, specifies the center for shifting the vectors (default is (0.0, 0.0)).

# Returns
- A vector of non-origin points from the virtual line segments.
"""
function get_virtual_line_segment_as_vectors(billiard::Bi; center::SVector{2,Float64}=SVector(0.0,0.0)) where {Bi<:AbsBilliard}
    origin_vl_segments = SVector{2, Float64}[] # contains the non-origin points of the virtual line segment
    for seg in billiard.fundamental_boundary # only in the fundamental boundary
        if seg isa VirtualLineSegment
            if !isapprox(norm(center), 0.0; atol=1e-8)
                pt0_shifted = seg.pt0 .- center # shift the pt0 of the virtual line
                pt1_shifted = seg.pt1 .- center # shift the pt1 of the virtual line
            else
                pt0_shifted = seg.pt0
                pt1_shifted = seg.pt1
            end
            if isapprox(norm(pt0_shifted), 0.0; atol=1e-8) # if shifted pt0 is the origin
                vector = pt1_shifted # push the point inside
                push!(origin_vl_segments, vector)
            elseif isapprox(norm(pt1_shifted), 0.0; atol=1e-8) # if shifted pt1 is the origin
                vector = pt0_shifted # push the point inside
                push!(origin_vl_segments, vector)
            end
        end
    end
    return origin_vl_segments
end


"""
    compute_vector_angles(vectors::Vector{SVector{2,T}}) where {T<:Real}

Calculates the angles of the given vectors (non-origin points of virtual line segments) relative to the origin.

# Arguments
- `vectors::Vector{SVector{2,T}}`: A vector of 2D vectors (points) where each point represents a non-origin vector.
  
# Returns
- A vector of angles (in radians) of the input vectors, each angle in the range `[0, 2π)`.
"""
function compute_vector_angles(vectors::Vector{SVector{2,T}}) where {T<:Real}
    vector_angles = T[]
    for v in vectors
        θ = atan(v[2], v[1]) # get the angle of the non-origin vector
        if θ < 0
            θ += 2π
        end
        push!(vector_angles, θ)
    end
    return vector_angles
end

"""
    sorted_vector_angles(vector_angles::Vector{T}) where T<:Real

Sorts a vector of angles in ascending order.

# Arguments
- `vector_angles::Vector{T}`: A vector of angles (in radians).

# Returns
- A sorted vector of angles.
"""
function sorted_vector_angles(vector_angles::Vector{T}) where T<:Real
    sorted_angles = sort(vector_angles)
    return sorted_angles
end

"""
    get_fundamental_area_indices(x_grid::Vector{T}, y_grid::Vector{T}, billiard::Bi; center::SVector{2, T}=SVector{2, T}(0.0, 0.0)) where {T<:Real, Bi<:AbsBilliard}

Identifies the indices of the fundamental area within a rectangular grid defined by `x_grid` and `y_grid`. The fundamental area 
is defined by the angles between two vectors extracted from the billiard's boundary.

# Arguments
- `x_grid::Vector{T}`: The grid of x-coordinates.
- `y_grid::Vector{T}`: The grid of y-coordinates.
- `billiard::Bi`: The billiard object, where `Bi` is a subtype of `AbsBilliard`.
- `center::SVector{2, T}`: Optional, specifies the center for the angles computation (default is (0.0, 0.0)).

# Returns
- A tuple containing:
  - `fund_indices`: The indices of the points that fall within the fundamental area.
  - `coords_flat`: The flattened coordinates of the grid.
  - `indices_flat`: The corresponding grid indices.
"""
function get_fundamental_area_indices(x_grid::Vector{T}, y_grid::Vector{T}, billiard::Bi; center::SVector{2, T}=SVector(0.0, 0.0)) where {T<:Real, Bi<:AbsBilliard}
    vectors = get_virtual_line_segment_as_vectors(billiard; center=center)
    vector_angles = compute_vector_angles(vectors)
    sort_vec_angles = sorted_vector_angles(vector_angles)
    θ1 = sort_vec_angles[1]
    θ2 = sort_vec_angles[2]
    x_indices = collect(1:length(x_grid))
    y_indices = collect(1:length(y_grid))
    coords = [SVector{2, T}(x_grid[ix], y_grid[iy]) for ix in x_indices, iy in y_indices]
    coords_flat = vec(coords)
    indices_flat = [(ix, iy) for ix in x_indices, iy in y_indices]
    indices_flat = vec(indices_flat)
    # Initialize list for indices of fundamental area points
    fund_indices = Int[]
    for idx in eachindex(coords_flat)
        coord = coords_flat[idx]
        θ = atan(coord[2] - center[2], coord[1] - center[1])
        if θ < zero(T)
            θ += 2 * pi
        end
        if θ > θ1 && θ < θ2
            push!(fund_indices, idx)
        end
    end
    return fund_indices, coords_flat, indices_flat
    end
    
"""
get_full_area(x_grid::Vector{T}, y_grid::Vector{T}, Psi_grid::Matrix{Complex{T}}, billiard::Bi, n::Int, m::Int; center::SVector{2, T}=SVector{2, T}(0.0, 0.0)) where {T<:Real, Bi<:AbsBilliard}

Generates the full set of coordinates and Psi values for the entire billiard by rotating the fundamental area `n` times and applying parity phase factor.

# Arguments
- `x_grid::Vector{T}`: The grid of x-coordinates.
- `y_grid::Vector{T}`: The grid of y-coordinates.
- `Psi_grid::Matrix{Complex{T}}`: The matrix of Psi values.
- `billiard::Bi`: The billiard object, where `Bi` is a subtype of `AbsBilliard`.
- `n::Int`: The number of times to rotate the fundamental area.
- `m::Int`: The mode number for the phase factor.
- `center::SVector{2, T}`: Optional, specifies the center for the rotations (default is (0.0, 0.0)).

# Returns
- A tuple containing:
  - `full_coords`: The full set of rotated coordinates.
  - `full_Psi`: The corresponding Psi values.
"""
function get_full_area(x_grid::Vector{T}, y_grid::Vector{T}, Psi_grid::Matrix{Complex{T}}, billiard::Bi, n::Int, m::Int; center::SVector{2, T}=SVector(0.0, 0.0)) where {T<:Real, Bi<:AbsBilliard}
    # Get the indices and coordinates of the fundamental area
    fund_indices, coords_flat, indices_flat = get_fundamental_area_indices(x_grid, y_grid, billiard; center=center)
    # Extract fundamental area points and associated values
    fund_coords = coords_flat[fund_indices]
    fund_indices_pairs = indices_flat[fund_indices]
    fund_Psi = [Psi_grid[ix, iy] for (ix, iy) in fund_indices_pairs]
    # Initialize full_coords and full_Psi with fundamental area data
    full_coords = copy(fund_coords)
    full_Psi = copy(fund_Psi)
    # Rotate the fundamental coordinates and associated values
    Threads.@threads for i in 1:(n - 1)
        θ = 2 * pi * i / n  # Compute the rotation angle
        R = SMatrix{2, 2, T}([cos(θ) -sin(θ); sin(θ)  cos(θ)])
        # Compute phase factor
        phase_factor = cis(m * θ)  # e^(i * m * θ)
        # Rotate coordinates and apply phase factor to Psi
        rotated_coords = [R * (coord .- center) .+ center for coord in fund_coords]
        rotated_Psi = phase_factor .* fund_Psi
        # Append to full_coords and full_Psi
        append!(full_coords, rotated_coords)
        append!(full_Psi, rotated_Psi)
    end
    return full_coords, full_Psi
end

"""
get_full_area_with_manual_binning(x_grid::Vector{T}, y_grid::Vector{T}, Psi_grid::Matrix{Complex{T}}, billiard::Bi, n::Int, m::Int; center::SVector{2, T}=SVector{2, T}(0.0, 0.0), grid_size::Int = ceil(Int, 0.7*length(x_grid))) where {T<:Real, Bi<:AbsBilliard}

Generates a rectangular grid over the billiard area and computes the averaged Psi values for each cell. Ensures that no two regions overlap within a single cell. Optionally, the grid size can be specified.

# Arguments
- `x_grid::Vector{T}`: The grid of x-coordinates.
- `y_grid::Vector{T}`: The grid of y-coordinates.
- `Psi_grid::Matrix{Complex{T}}`: The matrix of Psi values.
- `billiard::Bi`: The billiard object, where `Bi` is a subtype of `AbsBilliard`.
- `n::Int`: The number of times to rotate the fundamental area.
- `m::Int`: The mode number for the phase factor.
- `center::SVector{2, T}`: Optional, specifies the center for the rotations (default is (0.0, 0.0)).
- `grid_size::Int`: Optional, specifies the size of the rectangular grid (default is 70% of the original grid size).

# Returns
- A tuple containing:
- `new_x_grid`: The newly generated x-coordinates of the rectangular grid.
- `new_y_grid`: The newly generated y-coordinates of the rectangular grid.
- `new_Psi_grid`: The matrix of averaged Psi values.
"""
function get_full_area_with_manual_binning(x_grid::Vector{T}, y_grid::Vector{T}, Psi_grid::Matrix{Complex{T}}, billiard::Bi, n::Int, m::Int; center::SVector{2, T}=SVector(0.0, 0.0),grid_size::Int = ceil(Int, 0.7*length(x_grid))) where {T<:Real, Bi<:AbsBilliard} # grid_size should be less than length of full grid or we get wrong empty pixels

    # Get the original full_coords and full_Psi
    full_coords, full_Psi = get_full_area(x_grid, y_grid, Psi_grid, billiard, n, m; center=center)
    
    # Find min and max values for new grid
    x_min, x_max = extrema([coord[1] for coord in full_coords])
    y_min, y_max = extrema([coord[2] for coord in full_coords])

    # Create a rectangular grid
    new_x_grid = collect(range(x_min, stop=x_max, length=grid_size))
    new_y_grid = collect(range(y_min, stop=y_max, length=grid_size))

    # Initialize the new Psi grid and count grid (for averaging)
    new_Psi_grid = Matrix{Complex{T}}(zeros(Complex{T}, length(new_x_grid), length(new_y_grid)))
    count_grid = Matrix{Int}(zeros(Int, length(new_x_grid), length(new_y_grid)))
    # To track region for each cell
    region_grid = Matrix{Int}(zeros(Int, length(new_x_grid), length(new_y_grid)))

    # Iterate over the full coordinates and assign them to cells
    Threads.@threads for k in eachindex(full_coords)
        # Find the nearest x and y indices in the rectangular grid
        x_idx = searchsortedfirst(new_x_grid, full_coords[k][1])
        y_idx = searchsortedfirst(new_y_grid, full_coords[k][2])

        # Check bounds (some values might be slightly out of bounds due to precision)
        if x_idx > 0 && x_idx <= length(new_x_grid) && y_idx > 0 && y_idx <= length(new_y_grid)
            if count_grid[x_idx, y_idx] == 0 || region_grid[x_idx, y_idx] == k  # Either the cell is empty, or it's from the same region
                # Accumulate Psi value in the cell and increment the count
                new_Psi_grid[x_idx, y_idx] += full_Psi[k]
                count_grid[x_idx, y_idx] += 1
                region_grid[x_idx, y_idx] = k  # Assign the current region
            end
        end
    end

    # Average the Psi values for each cell
    Threads.@threads for i in eachindex(new_x_grid)
        for j in 1:length(new_y_grid)
            if count_grid[i, j] > 0
                new_Psi_grid[i, j] /= count_grid[i, j]
            else
                new_Psi_grid[i, j] = 0.0 + im * 0.0  # Assign 0 if no coordinates fall into the cell
            end
        end
    end

    return new_x_grid, new_y_grid, new_Psi_grid
end