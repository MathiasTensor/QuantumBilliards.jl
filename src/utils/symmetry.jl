
using CoordinateTransformations
using StaticArrays

reflect_x=LinearMap(SMatrix{2,2}([-1.0 0.0;0.0 1.0]))
reflect_y=LinearMap(SMatrix{2,2}([1.0 0.0;0.0 -1.0]))

"""
    Reflection

Represents a geometric reflection symmetry.

# Fields
- `sym_map::LinearMap{SMatrix{2,2,Float64,4}}`: The linear map defining the reflection.
- `parity::Union{Int64,Vector{Int64}}`: The parity of the wavefunction under reflection -> is vector for xy - reflection.
- `axis::Symbol`: The axis of reflection (`:x_axis`, `:y_axis`, or `:origin`).
"""
struct Reflection <: AbsSymmetry
    sym_map::LinearMap{SMatrix{2,2,Float64,4}}
    parity::Union{Int64,Vector{Int64}}
    axis::Symbol
end

"""
    Rotation

Represents a geometric rotation symmetry.

# Fields
- `n::Int`: The order of the rotation (e.g., 2 for 180° rotation, 4 for 90° rotation).
- `parity::Int`: The parity of the wavefunction under rotation.
"""
struct Rotation <: AbsSymmetry
    n::Int
    parity::Int
end

"""
    XReflection(parity::Int)

Creates a `Reflection` object that reflects over the `y`-axis.

# Arguments
- `parity::Int`: The wavefunction's parity under the reflection.

# Returns
- `Reflection`: A reflection object for the `y`-axis.

"""
function XReflection(parity)
    return Reflection(reflect_x,parity,:y_axis)
end

"""
    YReflection(parity::Int)

Creates a `Reflection` object that reflects over the `x`-axis.

# Arguments
- `parity::Int`: The wavefunction's parity under the reflection.

# Returns
- `Reflection`: A reflection object for the `x`-axis.
"""
function YReflection(parity)
    return Reflection(reflect_y,parity,:x_axis)
end

"""
    XYReflection(parity_x::Int, parity_y::Int)

Creates a `Reflection` object that reflects over both axes (origin symmetry).

# Arguments
- `parity_x::Int`: The wavefunction's parity under reflection over the `y`-axis.
- `parity_y::Int`: The wavefunction's parity under reflection over the `x`-axis.

# Returns
- `Reflection`: A reflection object for the origin symmetry.
"""
function XYReflection(parity_x,parity_y)
    return Reflection(reflect_x∘reflect_y,[parity_x,parity_y],:origin)
end

"""
    reflect_wavefunction(Psi::Matrix, x_grid::Vector, y_grid::Vector, symmetries::Sy; x_axis=0.0, y_axis=0.0) -> (Psi::Matrix, x_grid::Vector, y_grid::Vector)

Applies reflection symmetries to a wavefunction and its grid.

# Arguments
- `Psi::Matrix`: The wavefunction defined on the `x_grid` and `y_grid`.
- `x_grid::Vector`: The x-coordinates of the grid points.
- `y_grid::Vector`: The y-coordinates of the grid points.
- `symmetries::Vector{Any}`: A vector of `Reflection` objects to apply. Should contain only 1 element.
- `x_axis=0.0`: The x-coordinate of the reflection axis. Default is 0.0.
- `y_axis=0.0`: The y-coordinate of the reflection axis. Default is 0.0.

# Returns
- `Psi::Matrix`: The modified wavefunction after applying the reflections.
- `x_grid::Vector`: The updated x-coordinates of the grid points.
- `y_grid::Vector`: The updated y-coordinates of the grid points.
"""
function reflect_wavefunction(Psi,x_grid,y_grid,symmetries;x_axis=0.0,y_axis=0.0)
    x_grid=x_grid.-x_axis  # Shift the grid to move the reflection axis to x=0
    y_grid=y_grid.-y_axis  # Shift the grid to move the reflection axis to y=0
    for sym in symmetries
        if sym.axis==:y_axis
            x= -reverse(x_grid)
            Psi_ref=reverse(sym.parity.*Psi;dims=1)

            Psi=vcat(Psi,Psi_ref)
            x_grid=append!(x,x_grid)
            sorted_indices=sortperm(x_grid)
            x_grid=x_grid[sorted_indices]
        end
        if sym.axis==:x_axis
            y= -reverse(y_grid)
            Psi_ref=reverse(sym.parity.*Psi;dims=2)
            Psi=hcat(Psi_ref,Psi) 
            y_grid=append!(y,y_grid)
            sorted_indices=sortperm(y_grid)
            y_grid=y_grid[sorted_indices]
        end
        if sym.axis==:origin
            # Reflect over both axes (x -> -x, y -> -y)
            # First, reflect over y-axis
            x_reflected= -reverse(x_grid)
            Psi_y_reflected=reverse(sym.parity[1].*Psi;dims=2)
            Psi_y_combined=[Psi_y_reflected Psi]
            x_grid_combined=[x_reflected; x_grid]
            
            # Then, reflect over x-axis
            y_reflected= -reverse(y_grid)
            Psi_x_reflected=reverse(sym.parity[2].*Psi_y_combined;dims=1)
            Psi_x_combined=[Psi_x_reflected; Psi_y_combined]
            y_grid_combined=[y_reflected; y_grid]

            # Permute the indexes
            sorted_indices=sortperm(x_grid_combined)
            x_grid_combined=x_grid_combined[sorted_indices]
            sorted_indices=sortperm(y_grid_combined)
            y_grid_combined=y_grid_combined[sorted_indices]
            
            # Update Psi and grids
            Psi=Psi_x_combined
            x_grid=x_grid_combined
            y_grid=y_grid_combined
        end
    end
    # Shift the grids back to their original positions before returning
    x_grid=x_grid.+x_axis
    y_grid=y_grid.+y_axis
    return Psi,x_grid,y_grid
end

# INTERNAL 
function rotate_wavefunction(Psi_grid::Matrix, x_grid::Vector{T}, y_grid::Vector{T}, rotation::Rotation, billiard::Bi; center::SVector{2, Float64}=SVector{2, Float64}(0.0, 0.0),grid_size::Int = ceil(Int, 0.7*length(x_grid))) where {T<:Real, Bi<:AbsBilliard}
    let n=rotation.n,m=rotation.parity
        new_x_grid,new_y_grid,new_Psi_grid=get_full_area_with_manual_binning(x_grid,y_grid,Psi_grid,billiard,n,m;center=center,grid_size=grid_size)
        return new_Psi_grid,new_x_grid,new_y_grid 
    end
end
