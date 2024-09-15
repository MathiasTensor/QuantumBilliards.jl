
using CoordinateTransformations
using StaticArrays


reflect_x = LinearMap(SMatrix{2,2}([-1.0 0.0;0.0 1.0]))
reflect_y = LinearMap(SMatrix{2,2}([1.0 0.0;0.0 -1.0]))

struct Reflection <: AbsSymmetry
    sym_map::LinearMap{SMatrix{2, 2, Float64, 4}}
    parity::Union{Int64, Vector{Int64}}
    axis::Symbol
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

function reflect_wavefunction(Psi,x_grid,y_grid,symmetries)
    for sym in symmetries
        if sym.axis == :y_axis
            x = -reverse(x_grid)
            Psi_ref = reverse(sym.parity.*Psi; dims=1)

            Psi = vcat(Psi_ref,Psi)
            x_grid = append!(x,x_grid)
        end
        if sym.axis == :x_axis
            y = -reverse(y_grid)
            Psi_ref = reverse(sym.parity.*Psi; dims=2)

            Psi = hcat(Psi_ref,Psi)
            y_grid = append!(y,y_grid)
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