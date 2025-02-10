include("scalingmethod.jl")

"""
    solve_wavenumber(solver::AcceleratedSolver,basis::AbsBasis,billiard::AbsBilliard,k::T,dk::T) where {T<:Real}

Solves the wavenumber for an `AcceleratedSolver` by finding the one closest to the given reference wavenumber `k` if found in the `dk` interval.

# Arguments
- `solver<:AcceleratedSolver`: An instance of `AcceleratedSolver`.
- `basis<:AbsBasis`: The basis of the problem, no additional restrictions.
- `billiard<:AbsBilliard`: billiard instance, geometrical information.
- `k::T`: The reference wavenumber.
- `dk::T`: The interval in which to find the closest wavenumber.

# Returns
- `T`: The closest wavenumber found in the `dk` interval.
- `T`: The corresponding tension found for the closest wavenumber (by construction smallest tension in the given interval via findmin in the code).
"""
function solve_wavenumber(solver::AcceleratedSolver,basis::AbsBasis,billiard::AbsBilliard,k,dk)
    dim=max(solver.min_dim,round(Int,billiard.length*k*solver.dim_scaling_factor/(2*pi)))
    new_basis=resize_basis(basis,billiard,dim,k)
    pts=evaluate_points(solver,billiard,k)
    ks,ts=solve(solver,new_basis,pts,k,dk)
    idx=findmin(abs.(ks.-k))[2]
    return ks[idx],ts[idx]
end

"""
    solve_spectrum(solver::AcceleratedSolver,basis::AbsBasis,billiard::AbsBilliard,k::T,dk::T) where {T<:Real}

Solves for all the wavenumbers and corresponding tensions that `solve(<:AcceleratedSolver...)` gives us in the given interval `dk`.

# Arguments
- `solver<:AcceleratedSolver`: An instance of `AcceleratedSolver`.
- `basis<:AbsBasis`: The basis of the problem, no additional restrictions.
- `billiard<:AbsBilliard`: billiard instance, geometrical information.
- `k::T`: The reference wavenumber.
- `dk::T`: The interval in which to find the wavenumbers and corresponding tensions.

# Returns
- `Vector{T}`: The wavenumbers found in the `dk` interval.
- `Vector{T}`: The corresponding tensions found for the wavenumbers in the `dk` interval.
"""
function solve_spectrum(solver::AcceleratedSolver,basis::AbsBasis,billiard::AbsBilliard,k,dk)
    dim=max(solver.min_dim,round(Int,billiard.length*k*solver.dim_scaling_factor/(2*pi)))
    new_basis=resize_basis(basis,billiard,dim,k)
    pts=evaluate_points(solver, billiard,k)
    ks,ts=solve(solver,new_basis,pts,k,dk)
    return ks,ts
end