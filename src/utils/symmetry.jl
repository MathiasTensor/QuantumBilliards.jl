
using CoordinateTransformations
using StaticArrays


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

function rotate_wavefunction(Psi_grid::Matrix{Complex{T}}, x_grid::Vector{T}, y_grid::Vector{T}, rotation::Rotation, billiard::Bi, center::SVector{2, T}=SVector{2, T}(0.0, 0.0),grid_size::Int = ceil(Int, 0.7*length(x_grid))) where {T<:Real, Bi<:AbsBilliard}
    let n = rotation.n, m = rotation.parity
        new_x_grid, new_y_grid, new_Psi_grid = get_full_area_with_manual_binning(x_grid, y_grid, Psi_grid, billiard, n, m; center=center, grid_size=grid_size)
        return new_Psi_grid, new_x_grid, new_y_grid
    end
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