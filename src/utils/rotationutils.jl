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
function get_virtual_line_segment_as_vectors(billiard::Bi; center::SVector{2, Float64}=SVector{2, Float64}(0.0, 0.0)) where {Bi<:AbsBilliard}
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
function get_fundamental_area_indices(x_grid::Vector{T}, y_grid::Vector{T}, billiard::Bi; center::SVector{2, Float64}=SVector{2, Float64}(0.0, 0.0)) where {T<:Real, Bi<:AbsBilliard}
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
get_full_area(x_grid::Vector{T}, y_grid::Vector{T}, Psi_grid::Matrix, billiard::Bi, n::Int, m::Int; center::SVector{2, T}=SVector{2, T}(0.0, 0.0)) where {T<:Real, Bi<:AbsBilliard}

Generates the full set of coordinates and Psi values for the entire billiard by rotating the fundamental area `n` times and applying parity phase factor.

# Arguments
- `x_grid::Vector{T}`: The grid of x-coordinates.
- `y_grid::Vector{T}`: The grid of y-coordinates.
- `Psi_grid::Matrix`: The matrix of Psi values.
- `billiard::Bi`: The billiard object, where `Bi` is a subtype of `AbsBilliard`.
- `n::Int`: The number of times to rotate the fundamental area.
- `m::Int`: The mode number for the phase factor.
- `center::SVector{2, T}`: Optional, specifies the center for the rotations (default is (0.0, 0.0)).

# Returns
- A tuple containing:
  - `full_coords`: The full set of rotated coordinates.
  - `full_Psi`: The corresponding Psi values.
"""
function get_full_area(x_grid::Vector{T}, y_grid::Vector{T}, Psi_grid::Matrix, billiard::Bi, n::Int, m::Int; center::SVector{2, Float64}=SVector{2, Float64}(0.0, 0.0)) where {T<:Real, Bi<:AbsBilliard}
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
    for i in 1:(n - 1)
        θ = 2 * pi * i / n  # Compute the rotation angle
        R = SMatrix{2, 2, T}([cos(θ) -sin(θ); sin(θ)  cos(θ)])
        # Compute phase factor
        phase_factor = cis(m * θ)  # e^(i * m * θ)
        # Rotate coordinates and apply phase factor to Psi
        rotated_coords = [R * (coord .- center) .+ center for coord in fund_coords]
        #rotated_Psi = phase_factor .* fund_Psi
        rotated_Psi = fund_Psi
        # Append to full_coords and full_Psi
        append!(full_coords, rotated_coords)
        append!(full_Psi, rotated_Psi) # ordering should be imporant here, perhaps use indexed threads ?
    end
    return full_coords, full_Psi
end

# helper for cropping the final x and y grids with the Psi matrix
function crop_grid_with_full_boundary(Psi::Matrix{T}, x_grid::Vector{T}, y_grid::Vector{T}, billiard::Bi) where {T<:Real, Bi<:AbsBilliard}
    xy_vec = billiard_polygon(billiard, 256; fundamental_domain=false)
    xy_vec = vcat(xy_vec...)
    x_min, x_max = extrema([xy[1] for xy in xy_vec])
    y_min, y_max = extrema([xy[2] for xy in xy_vec])
    x_in_bounds = (x_min[1] .<= x_grid) .& (x_grid .<= x_max[1])
    y_in_bounds = (y_min[1] .<= y_grid) .& (y_grid .<= y_max[1])
    cropped_x_grid = x_grid[x_in_bounds]
    cropped_y_grid = y_grid[y_in_bounds]
    cropped_Psi = Psi[x_in_bounds, y_in_bounds]
    return cropped_x_grid, cropped_y_grid, cropped_Psi
end


"""
get_full_area_with_manual_binning(x_grid::Vector{T}, y_grid::Vector{T}, Psi_grid::Matrix, billiard::Bi, n::Int, m::Int; center::SVector{2, T}=SVector{2, T}(0.0, 0.0), grid_size::Int = ceil(Int, 0.7*length(x_grid))) where {T<:Real, Bi<:AbsBilliard}

Generates a rectangular grid over the billiard area and computes the averaged Psi values for each cell. Ensures that no two regions overlap within a single cell. Optionally, the grid size can be specified.

# Arguments
- `x_grid::Vector{T}`: The grid of x-coordinates.
- `y_grid::Vector{T}`: The grid of y-coordinates.
- `Psi_grid::Matrix`: The matrix of Psi values.
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
function get_full_area_with_manual_binning(x_grid::Vector{T}, y_grid::Vector{T}, Psi_grid::Matrix, billiard::Bi, n::Int, m::Int; center::SVector{2, Float64}=SVector{2, Float64}(0.0, 0.0),grid_size::Int = ceil(Int, 0.7*length(x_grid))) where {T<:Real, Bi<:AbsBilliard} # grid_size should be less than length of full grid or we get wrong empty pixels
    
    # Get the original full_coords and full_Psi
    full_coords, full_Psi = get_full_area(x_grid, y_grid, Psi_grid, billiard, n, m; center=center)
    
    # Find min and max values for new grid
    x_min, x_max = extrema([coord[1] for coord in full_coords])
    y_min, y_max = extrema([coord[2] for coord in full_coords])

    # Create a rectangular grid
    new_x_grid = collect(range(x_min, stop=x_max, length=grid_size))
    new_y_grid = collect(range(y_min, stop=y_max, length=grid_size))

    # Initialize the new Psi grid and count grid (for averaging)
    new_Psi_grid = Matrix{T}(zeros(T, length(new_x_grid), length(new_y_grid)))
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
                new_Psi_grid[i, j] = 0.0  # Assign 0 if no coordinates fall into the cell
            end
        end
    end
    #return new_x_grid, new_y_grid, new_Psi_grid
    return crop_grid_with_full_boundary(new_Psi_grid, new_x_grid, new_y_grid, billiard)
end

"""
    apply_symmetries_to_boundary_points(pts::BoundaryPoints{T}, symmetries::Union{Vector{Any},Nothing}, billiard::Bi; same_direction::Bool=true) where {Bi<:AbsBilliard, T<:Real}

Convenience function that construct new BoundaryPoints object from the old BoundaryPoints object such that it extends it with the correct symmetries.

# Arguments
- `pts`: Original BoundaryPoints object
- `symmetries`: Vector of symmetry operations (Reflection, Rotation) to be applied. If `Nothing`, symmetries are not applied. Do not mix and match rotation/reflections.
- `billiard`: Billiard object used to determine if the symmetry axes are shifted wrt origin.
- `same_direction::Bool=true`: Reverse the direction of the points construction such that the direction of the new points is anti-clockwise. If false it creates a perfect symmetry (e.g. for reflection the point direction will also be mirror reflected)

# Returns
- `BoundaryPoints`: The new symmetry adapted boundary points.
"""
function apply_symmetries_to_boundary_points(pts::BoundaryPoints{T}, symmetries::Union{Vector{Any},Nothing}, billiard::Bi; same_direction::Bool=true) where {Bi<:AbsBilliard, T<:Real}
    if isnothing(symmetries)
        return pts
    end
    
    # desymmetrized boundary points
    full_xy = copy(pts.xy)
    full_normal = copy(pts.normal)
    full_s = copy(pts.s)
    full_ds = copy(pts.ds)
    max_s = maximum(pts.s)

    # shifts for symmetry axes when the billiard's reflections centers do not include the origin
    x_axis, y_axis = 0.0, 0.0
    if hasproperty(billiard, :x_axis)
        x_axis = billiard.x_axis
    end
    if hasproperty(billiard, :y_axis)
        y_axis = billiard.y_axis
    end

    # pre-shift to align symmetry axes with the origin, does nothing if no shifts
    shifted_xy = [SVector(v[1]-x_axis, v[2]-y_axis) for v in full_xy]
    shifted_normal = full_normal  # normals are unaffected by shifts, no need for this

    for sym in symmetries
        if sym isa Reflection
            if sym.axis == :y_axis
                # reflect across the vertical reflection axis (at x = x_axis)
                reflected_xy = [SVector(-v[1], v[2]) for v in shifted_xy]
                reflected_normal = [SVector(-n[1], n[2]) for n in shifted_normal]
                reflected_s = 2*max_s .- reverse(full_s)
                reflected_ds = reverse(full_ds)
                # shift back reflected coordinates back to the original alignment
                shifted_back_xy = [SVector(v[1]+x_axis, v[2]+y_axis) for v in reflected_xy]
                # Reverse direction if same_direction=true
                if same_direction # reverse only for xy and normals since s and ds are already correctly set up
                    shifted_back_xy = reverse(shifted_back_xy)
                    reflected_normal = reverse(reflected_normal)
                end
            elseif sym.axis == :x_axis
                # reflect across the horizontal reflection axis (at y = y_axis)
                reflected_xy = [SVector(v[1], -v[2]) for v in shifted_xy]
                reflected_normal = [SVector(n[1], -n[2]) for n in shifted_normal]
                reflected_s = 2*max_s .- reverse(full_s)
                reflected_ds = reverse(full_ds)
                # shift back reflected coordinates back to the original alignment
                shifted_back_xy = [SVector(v[1]+x_axis, v[2]+y_axis) for v in reflected_xy]
                # Reverse direction if same_direction=true
                if same_direction # reverse only for xy and normals since s and ds are already correctly set up
                    shifted_back_xy = reverse(shifted_back_xy)
                    reflected_normal = reverse(reflected_normal)
                end
            elseif sym.axis == :origin
                # reflect across the vertical axis (x-axis reflection at x = x_axis)
                vertical_reflected_xy = [SVector(-v[1], v[2]) for v in shifted_xy]
                vertical_reflected_normal = [SVector(-n[1], n[2]) for n in shifted_normal]
                vertical_reflected_s = 2 * max_s .- full_s
                vertical_reflected_ds = full_ds
                # resolve direction if `same_direction=true` for vertical reflection
                if same_direction
                    vertical_reflected_xy = reverse(vertical_reflected_xy)
                    vertical_reflected_normal = reverse(vertical_reflected_normal)
                end
                # reflect the vertically reflected points across the horizontal axis (y-axis reflection at y = y_axis)
                horizontal_reflected_xy = [SVector(v[1], -v[2]) for v in vertical_reflected_xy]
                horizontal_reflected_normal = [SVector(n[1], -n[2]) for n in vertical_reflected_normal]
                horizontal_reflected_s = 2*(2*max_s) .- vertical_reflected_s
                horizontal_reflected_ds = vertical_reflected_ds
                # resolve direction if `same_direction=true` for horizontal reflection
                if same_direction
                    horizontal_reflected_xy = reverse(horizontal_reflected_xy)
                    horizontal_reflected_normal = reverse(horizontal_reflected_normal)
                end
                # shift back reflected coordinates back to the original alignment
                shifted_back_xy = [SVector(v[1]+x_axis, v[2]+y_axis) for v in horizontal_reflected_xy]
                # Update reflected values do to compounded logic
                reflected_xy = shifted_back_xy
                reflected_normal = horizontal_reflected_normal
                reflected_s = horizontal_reflected_s
                reflected_ds = horizontal_reflected_ds     
            end
            
            # combine the reflected components with the originals
            full_xy = vcat(full_xy, shifted_back_xy)
            full_normal = vcat(full_normal, reflected_normal)
            full_s = vcat(full_s, reflected_s)
            full_ds = vcat(full_ds, reflected_ds)
        
        elseif sym isa Rotation
            θ = 2π / sym.n 
            R = SMatrix{2, 2, Float64}([cos(θ) -sin(θ); sin(θ) cos(θ)])
            # Apply the rotation sym.n - 1 times
            current_xy = shifted_xy
            current_normal = shifted_normal
            for i in 1:(sym.n-1)
                # Rotate points and normals
                rotated_xy = [R*SVector(v[1], v[2]) for v in current_xy]
                rotated_normal = [R*SVector(n[1], n[2]) for n in current_normal]
                # shift rotated coordinates back to the original alignment
                shifted_back_rot_xy = [SVector(v[1]+x_axis, v[2]+y_axis) for v in rotated_xy]
                full_xy = vcat(full_xy, shifted_back_rot_xy)
                full_normal = vcat(full_normal, rotated_normal)
                # Extend s arc length and ds
                rotated_s = full_s .+ maximum(full_s)
                full_s = vcat(full_s, rotated_s)
                full_ds = vcat(full_ds, full_ds)
                current_xy = rotated_xy  # Update current_xy and current_normal for the next rotation
                current_normal = rotated_normal
            end
        else
            error("Unknown symmetry type: $(typeof(sym))")
        end
    end
    return BoundaryPoints(full_xy, full_normal, full_s, full_ds)
end

"""
    difference_boundary_points(pts_new::BoundaryPoints{T}, pts_old::BoundaryPoints{T}) where {T<:Real}

Compute the difference between two `BoundaryPoints` objects. The returned `BoundaryPoints` object contains
only the elements in `pts_new` that are not present in `pts_old`. Throws errors if the lengths of the `Vector` fields in either the old or new inputs is not the same

# Arguments
- `pts_new::BoundaryPoints`: The updated `BoundaryPoints` object that contains all the points.
- `pts_old::BoundaryPoints`: The original `BoundaryPoints` object whose data is a subset of `pts_new`.

# Returns
- A new `BoundaryPoints` object containing the fields (`xy`, `normal`, `s`, `ds`) from `pts_new`
  starting from the index where `pts_old` ends.
"""
function difference_boundary_points(pts_new::BoundaryPoints{T}, pts_old::BoundaryPoints{T}) where {T<:Real}
    if !(length(pts_new.xy) == length(pts_new.normal) == length(pts_new.s) == length(pts_new.ds))
        println("length xy: ", length(pts_new.xy))
        println("length normal: ", length(pts_new.normal))
        println("length s: ", length(pts_new.s))
        println("length ds: ", length(pts_new.ds))
        error("Fields of pts_new are not consistent in length")
    end
    if !(length(pts_old.xy) == length(pts_old.normal) == length(pts_old.s) == length(pts_old.ds))
        println("length xy: ", length(pts_old.xy))
        println("length normal: ", length(pts_old.normal))
        println("length s: ", length(pts_old.s))
        println("length ds: ", length(pts_old.ds))
        error("Fields of pts_old are not consistent in length")
    end
    start_index = length(pts_old.xy)+1
    new_xy = pts_new.xy[start_index:end]
    new_normal = pts_new.normal[start_index:end]
    new_s = pts_new.s[start_index:end]
    new_ds = pts_new.ds[start_index:end]
    return BoundaryPoints(new_xy, new_normal, new_s, new_ds)
end