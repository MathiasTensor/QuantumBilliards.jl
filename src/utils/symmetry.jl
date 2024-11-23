include("../billiards/boundarypoints.jl")
using CoordinateTransformations
using StaticArrays
struct BoundaryPoints{T} <: AbsPoints where {T<:Real}; end

reflect_x = LinearMap(SMatrix{2,2}([-1.0 0.0;0.0 1.0]))
reflect_y = LinearMap(SMatrix{2,2}([1.0 0.0;0.0 -1.0]))

struct Reflection <: AbsSymmetry
    sym_map::LinearMap{SMatrix{2, 2, Float64, 4}}
    parity::Union{Int64, Vector{Int64}}
    axis::Symbol
end

struct Rotation <: AbsSymmetry
    n::Int
    parity::Int
end

function XReflection(parity)
    return Reflection(reflect_x, parity, :y_axis)
end

function YReflection(parity)
    return Reflection(reflect_y, parity, :x_axis)
end

function XYReflection(parity_x, parity_y)
    return Reflection(reflect_x ∘ reflect_y, [parity_x, parity_y], :origin)
end

function reflect_wavefunction(Psi,x_grid,y_grid,symmetries; x_axis=0.0, y_axis=0.0)
    #println("x_grid before: ", extrema(x_grid))
    #println("y_grid before: ", extrema(y_grid))
    #println("length of grids before: ", length(x_grid))
    #println("Size of Psi before: ", size(Psi))
    x_grid = x_grid .- x_axis  # Shift the grid to move the reflection axis to x=0
    y_grid = y_grid .- y_axis  # Shift the grid to move the reflection axis to y=0
    #println("x_grid after shift: ", extrema(x_grid))
    #println("y_grid after shift: ", extrema(y_grid))
    for sym in symmetries
        if sym.axis == :y_axis
            x = -reverse(x_grid)
            Psi_ref = reverse(sym.parity.*Psi; dims=1)

            Psi = vcat(Psi,Psi_ref)
            x_grid = append!(x,x_grid)
            sorted_indices = sortperm(x_grid)
            x_grid = x_grid[sorted_indices]
            #println("x_grid after appending: ", extrema(x_grid))
        end
        if sym.axis == :x_axis
            y = -reverse(y_grid)
            Psi_ref = reverse(sym.parity.*Psi; dims=2)

            Psi = hcat(Psi_ref,Psi) 
            y_grid = append!(y,y_grid)
            sorted_indices = sortperm(y_grid)
            y_grid = y_grid[sorted_indices]
            #println("y_grid after appending: ", extrema(y_grid))
        end
        if sym.axis == :origin
            # Reflect over both axes (x -> -x, y -> -y)
            # First, reflect over y-axis
            x_reflected = -reverse(x_grid)
            Psi_y_reflected = reverse(sym.parity[1] .* Psi; dims=2)
            Psi_y_combined = [Psi_y_reflected Psi]
            x_grid_combined = [x_reflected; x_grid]
            
            # Then, reflect over x-axis
            y_reflected = -reverse(y_grid)
            Psi_x_reflected = reverse(sym.parity[2] .* Psi_y_combined; dims=1)
            Psi_x_combined = [Psi_x_reflected; Psi_y_combined]
            y_grid_combined = [y_reflected; y_grid]

            # Permute the indexes
            sorted_indices = sortperm(x_grid_combined)
            x_grid_combined = x_grid_combined[sorted_indices]
            sorted_indices = sortperm(y_grid_combined)
            y_grid_combined = y_grid_combined[sorted_indices]
            
            # Update Psi and grids
            Psi = Psi_x_combined
            x_grid = x_grid_combined
            y_grid = y_grid_combined
        end
    end
    # Shift the grids back to their original positions before returning
    x_grid = x_grid .+ x_axis
    y_grid = y_grid .+ y_axis
    #println("x_grid after shift back: ", extrema(x_grid))
    #println("y_grid after shift back: ", extrema(y_grid))
    #println("length of grids after: ", length(x_grid))
    #println("Size of Psi after: ", size(Psi))
    return Psi, x_grid, y_grid
end

function rotate_wavefunction(Psi_grid::Matrix, x_grid::Vector{T}, y_grid::Vector{T}, rotation::Rotation, billiard::Bi; center::SVector{2, Float64}=SVector{2, Float64}(0.0, 0.0),grid_size::Int = ceil(Int, 0.7*length(x_grid))) where {T<:Real, Bi<:AbsBilliard}
    let n = rotation.n, m = rotation.parity
        new_x_grid, new_y_grid, new_Psi_grid = get_full_area_with_manual_binning(x_grid, y_grid, Psi_grid, billiard, n, m; center=center, grid_size=grid_size)
        return new_Psi_grid, new_x_grid, new_y_grid 
    end
end

"""
    apply_symmetries_to_boundary_points(pts::BoundaryPoints{T}, symmetries::Union{Vector{Any},Nothing}, billiard::Bi) where {Bi<:AbsBilliard, T<:Real}

Convenience function that construct new BoundaryPoints object from the old BoundaryPoints object such that it extends it with the correct symmetries.

# Arguments
- `pts`: Original BoundaryPoints object
- `symmetries`: Vector of symmetry operations (Reflection, Rotation) to be applied. If `Nothing`, symmetries are not applied. Do not mix and match rotation/reflections.
- `billiard`: Billiard object used to determine if the symmetry axes are shifted wrt origin.

# Returns
- `BoundaryPoints`: The new symmetry adapted boundary points.
"""
function apply_symmetries_to_boundary_points(pts::BoundaryPoints{T}, symmetries::Union{Vector{Any},Nothing}, billiard::Bi) where {Bi<:AbsBilliard, T<:Real}
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
    shifted_xy = [SVector(x-x_axis, y-y_axis) for SVector(x, y) in full_xy]
    #shifted_normal = [SVector(nx, ny) for SVector(nx, ny) in full_normal]
    shifted_normal = full_normal  # normals are unaffected by shifts, no need for this

    for sym in symmetries
        if sym isa Reflection
            if sym.axis == :y_axis
                # reflect across the vertical reflection axis (at x = x_axis)
                reflected_xy = [SVector(-x, y) for SVector(x, y) in shifted_xy]
                reflected_normal = [SVector(-nx, ny) for SVector(nx, ny) in shifted_normal]
                reflected_s = 2*max_s .- reverse(full_s[1:end-1])
                reflected_ds = reverse(full_ds[1:end-1])
            elseif sym.axis == :x_axis
                # reflect across the horizontal reflection axis (at y = y_axis)
                reflected_xy = [SVector(x, -y) for SVector(x, y) in shifted_xy]
                reflected_normal = [SVector(nx, -ny) for SVector(nx, ny) in shifted_normal]
                reflected_s = 2*max_s .- reverse(full_s[1:end-1])
                reflected_ds = reverse(full_ds[1:end-1])
            elseif sym.axis == :origin
                # reflect across both axes (origin reflection, combining shifts)
                reflected_xy = [SVector(-x, -y) for SVector(x, y) in shifted_xy]
                reflected_normal = [SVector(-nx, -ny) for SVector(nx, ny) in shifted_normal]
                reflected_s = 2*(2*max_s) .- reverse(full_s[1:end-1]) 
                reflected_ds = reverse(full_ds[1:end-1])     
            end
            # shift back reflected coordinates back to the original alignment
            shifted_back_xy = [SVector(x + x_axis, y + y_axis) for SVector(x, y) in reflected_xy]
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
                rotated_xy = [R*SVector(x, y) for SVector(x, y) in current_xy]
                rotated_normal = [R*SVector(nx, ny) for SVector(nx, ny) in current_normal]
                # shift rotated coordinates back to the original alignment
                shifted_back_rot_xy = [SVector(x + x_axis, y + y_axis) for SVector(x, y) in rotated_xy]
                full_xy = vcat(full_xy, shifted_back_rot_xy)
                full_normal = vcat(full_normal, rotated_normal)
                # Extend s arc length and ds
                rotated_s = full_s[1:end-1] .+ maximum(full_s)
                full_s = vcat(full_s, rotated_s)
                full_ds = vcat(full_ds, full_ds[1:end-1])
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
        error("Fields of pts_new are not consistent in length")
    end
    if !(length(pts_old.xy) == length(pts_old.normal) == length(pts_old.s) == length(pts_old.ds))
        error("Fields of pts_old are not consistent in length")
    end
    start_index = length(pts_old.xy)+1
    new_xy = pts_new.xy[start_index:end]
    new_normal = pts_new.normal[start_index:end]
    new_s = pts_new.s[start_index:end]
    new_ds = pts_new.ds[start_index:end]
    return BoundaryPoints(new_xy, new_normal, new_s, new_ds)
end

#=
function reflect_wavefunction(Psi,x_grid,y_grid,symmetries; x_axis=0.0, y_axis=0.0)
    for sym in symmetries
        if sym.axis == :y_axis
            x = -reverse(x_grid)
            Psi_ref = reverse(sym.parity.*Psi; dims=1)

            Psi = vcat(Psi_ref,Psi)
            println("x_grid before: ", x_grid)
            x_grid = append!(x,x_grid)
            println("x_grid after: ", x_grid)
        end
        if sym.axis == :x_axis
            y = -reverse(y_grid)
            Psi_ref = reverse(sym.parity.*Psi; dims=2)

            Psi = hcat(Psi_ref,Psi) 
            println("y_grid before: ", y_grid)
            y_grid = append!(y,y_grid)
            println("y_grid after: ", y_grid)
        end
        if sym.axis == :origin
            # Reflect over both axes (x -> -x, y -> -y)
            # First, reflect over y-axis
            x_reflected = -reverse(x_grid)
            Psi_y_reflected = reverse(sym.parity[1] .* Psi; dims=2)
            Psi_y_combined = [Psi_y_reflected Psi]
            x_grid_combined = [x_reflected; x_grid]
            
            # Then, reflect over x-axis
            y_reflected = -reverse(y_grid)
            Psi_x_reflected = reverse(sym.parity[2] .* Psi_y_combined; dims=1)
            Psi_x_combined = [Psi_x_reflected; Psi_y_combined]
            y_grid_combined = [y_reflected; y_grid]
            
            # Update Psi and grids
            Psi = Psi_x_combined
            x_grid = x_grid_combined
            y_grid = y_grid_combined
        end
    end
    return Psi, x_grid, y_grid
end
=#


#=
function reflect_wavefunction(Psi,x_grid,y_grid,symmetries)
    for sym in symmetries
        if sym.axis == :y_axis
            if x_grid[1] == zero(eltype(x_grid))
                x = -reverse(x_grid[2:end])
                Psi_ref = reverse(sym.parity.*Psi[2:end,:]; dims=1)
            else
                x = -reverse(x_grid)
                Psi_ref = reverse(sym.parity.*Psi; dims=1)
            end
            Psi = vcat(Psi_ref,Psi)
            x_grid = append!(x,x_grid)
        end
        if sym.axis == :x_axis
            if y_grid[1] == zero(eltype(y_grid))
                y = -reverse(y_grid[2:end])
                Psi_ref = reverse(sym.parity.*Psi[:,2:end]; dims=2)
            else
                y = -reverse(y_grid)
                Psi_ref = reverse(sym.parity.*Psi; dims=2)
            end
            Psi = hcat(Psi_ref,Psi)
            y_grid = append!(y,y_grid)
        end
    end
    return Psi, x_grid, y_grid
end
=#