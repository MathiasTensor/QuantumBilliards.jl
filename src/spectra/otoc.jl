include("../states/wavefunctions.jl")

# Calculates the matrix element x_nm = <n|x|m> for real wavefunctions in billiard systems.
function x_nm(state_n::S, state_m::S) where {S<:AbsState}
    k = max(state_m.k, state_n.k) 
    nx = max(round(Int, k*dx*b/(2*pi)),512)
    ny = max(round(Int, k*dy*b/(2*pi)),512)
    # Get wavefunctions and grids for both states on the same grid
    Psi2d_m, x_grid, y_grid = wavefunction_fixed_grid_(state_m, nx, ny)
    Psi2d_n, _, _ = wavefunction_fixed_grid_(state_n, nx, ny)
    deltax = x_grid[2] - x_grid[1]
    deltay = y_grid[2] - y_grid[1]
    integral_vals = fill(0.0, Threads.nthreads())
    Threads.@threads for j in eachindex(y_grid)
        for i in eachindex(x_grid)
            x = x_grid[i]
            integral_vals[Threads.threadid()] += x * Psi2d_n[i, j] * Psi2d_m[i, j] * deltax * deltay
        end
    end
    integral = sum(integral_vals)
    return integral
end