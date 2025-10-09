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
                # Reflect across the vertical axis (x-axis reflection at x = x_axis)
                vertical_reflected_xy = [SVector(-v[1], v[2]) for v in shifted_xy]
                vertical_reflected_normal = [SVector(-n[1], n[2]) for n in shifted_normal]
                vertical_reflected_s = 2*max_s .- reverse(full_s)
                vertical_reflected_ds = reverse(full_ds)
                # Resolve direction if `same_direction=true` for vertical reflection
                if same_direction
                    vertical_reflected_xy = reverse(vertical_reflected_xy)
                    vertical_reflected_normal = reverse(vertical_reflected_normal)
                end
                # Combine original and vertical reflection
                combined_xy = vcat(shifted_xy, vertical_reflected_xy)
                combined_normal = vcat(shifted_normal, vertical_reflected_normal)
                combined_s = vcat(full_s, vertical_reflected_s)
                combined_ds = vcat(full_ds, vertical_reflected_ds)

                # Reflect the combined result across the horizontal axis (y-axis reflection at y = y_axis)
                horizontal_reflected_xy = [SVector(v[1], -v[2]) for v in combined_xy]
                horizontal_reflected_normal = [SVector(n[1], -n[2]) for n in combined_normal]
                horizontal_reflected_s = 2 * max_s .+ combined_s
                horizontal_reflected_ds = combined_ds

                # Resolve direction if `same_direction=true` for horizontal reflection
                if same_direction
                    horizontal_reflected_xy = reverse(horizontal_reflected_xy)
                    horizontal_reflected_normal = reverse(horizontal_reflected_normal)
                end

                # Combine the vertical and horizontal reflections
                final_xy = vcat(vertical_reflected_xy, horizontal_reflected_xy)
                final_normal = vcat(vertical_reflected_normal, horizontal_reflected_normal)
                final_s = vcat(vertical_reflected_s, horizontal_reflected_s)
                final_ds = vcat(vertical_reflected_ds, horizontal_reflected_ds)

                # Shift back reflected coordinates to the original alignment
                shifted_back_xy = [SVector(v[1] + x_axis, v[2] + y_axis) for v in final_xy]

                # Update reflected values
                reflected_xy = shifted_back_xy
                reflected_normal = final_normal
                reflected_s = final_s
                reflected_ds = final_ds  
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
    if !(length(pts_new.xy)==length(pts_new.normal)==length(pts_new.s)==length(pts_new.ds))
        println("length xy: ", length(pts_new.xy))
        println("length normal: ", length(pts_new.normal))
        println("length s: ", length(pts_new.s))
        println("length ds: ", length(pts_new.ds))
        error("Fields of pts_new are not consistent in length")
    end
    if !(length(pts_old.xy)==length(pts_old.normal)==length(pts_old.s)==length(pts_old.ds))
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

"""
    apply_symmetries_to_boundary_function(u::Vector{T}, symmetries::Union{Vector{Any}, Nothing}) where {T<:Real}

Applies symmetries to the desymmetrized_full_boundary BoundaryPoints constructed u(s) boundary function. Needed for the correct whole boundary.

# Arguments
- `u::Vector{T}`: The desymmetrized boundary function `u(s)`.
- `symmetries::Union{Vector{Any}, Nothing}`: The symmetries to apply to the boundary function. If `Nothing`, no symmetries are applied.

# Returns
- A `Vector{T}` containing the symmetrized boundary function `u(s)`.
"""
function apply_symmetries_to_boundary_function(u::Vector{T}, symmetries::Union{Vector{Any}, Nothing}) where {T<:Real}
    if isnothing(symmetries)
        return u
    end
    full_u = copy(u)
    for sym in symmetries
        if sym isa Reflection
            if sym.axis == :y_axis
                reflected_u = sym.parity .* reverse(u)
            elseif sym.axis == :x_axis
                reflected_u = sym.parity .* reverse(u)
            elseif sym.axis == :origin
                vertical_reflected_u = sym.parity[1] .* reverse(u)
                combined_u = vcat(u, vertical_reflected_u)
                reflected_u = sym.parity[2] .* reverse(combined_u)
                reflected_u = vcat(vertical_reflected_u, reflected_u) # same trick as the boundary point case
            end
            full_u = vcat(full_u, reflected_u)
        elseif sym isa Rotation
            current_u = u
            for i in 1:(sym.n-1)
                rotated_u = sym.parity .* current_u
                full_u = vcat(full_u, rotated_u)
                current_u = rotated_u
            end
        else
            error("Unknown symmetry type: $(typeof(sym))")
        end
    end
    return full_u
end