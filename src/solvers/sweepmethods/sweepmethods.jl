#include("../../abstracttypes.jl")
#include("../../utils/billiardutils.jl")
#include("decompositions.jl")
#include("../samplers.jl")
include("particularsolutionsmethod.jl")
include("decompositionmethod.jl")
include("boundaryintegralmethod.jl")
using LinearAlgebra, Optim, ProgressMeter

"""
    solve_wavenumber(solver::SweepSolver,basis::AbsBasis,billiard::AbsBilliard,k::Real,dk::Real) -> Tuple{Real, Real}

Solves for the wavenumbers `k0` its corresponding tension `t0` within a given range `dk`.

# Arguments
- `solver::SweepSolver`: The solver configuration for performing the sweep.
- `basis::AbsBasis`: The basis to be used. Can use also the AbstractHankelBasis() when solver is `BoundaryIntegralMethod`.
- `billiard::AbsBilliard`: The billiard configuration.
- `k::Real`: Central wavenumber for the optimization.
- `dk::Real`: Range to search for the lowest tension.

# Returns
- `Tuple{Real, Real}`: 
  - `k0`: Wavenumber with lowest tension.
  - `t0`: Corresponding lowest tension.
"""
function solve_wavenumber(solver::SweepSolver,basis::AbsBasis, billiard::AbsBilliard, k, dk)
    dim = max(solver.min_dim,round(Int, billiard.length*k*solver.dim_scaling_factor/(2*pi)))
    new_basis = resize_basis(basis,billiard,dim,k)
    pts = evaluate_points(solver, billiard, k)
    function f(k)
        return solve(solver,new_basis,pts,k)
    end
    res =  optimize(f, k-0.5*dk, k+0.5*dk)
    k0,t0 = res.minimizer, res.minimum
    return k0, t0
end

"""
    k_sweep(solver::SweepSolver,basis::AbsBasis,billiard::AbsBilliard,ks::Vector{Real}) -> Vector{Real}

Performs a sweep over a range of wavenumbers `ks` and computes tensions for `res` each.

# Arguments
- `solver::SweepSolver`: The solver configuration for performing the sweep.
- `basis::AbsBasis`: The basis to be used. Can use also the AbstractHankelBasis() when solver is `BoundaryIntegralMethod`.
- `billiard::AbsBilliard`: The billiard configuration.
- `ks::Vector{Real}`: Vector of wavenumbers over which to perform the sweep.

# Returns
- `Vector{Real}`: Tensions of the `solve` function for each wavenumber in `ks`.
"""
function k_sweep(solver::SweepSolver, basis::AbsBasis, billiard::AbsBilliard, ks)
    k = maximum(ks)
    dim = max(solver.min_dim,round(Int, billiard.length*k*solver.dim_scaling_factor/(2*pi)))
    new_basis = resize_basis(basis,billiard,dim,k)
    pts = evaluate_points(solver, billiard, k)
    res = similar(ks)
    num_intervals = length(ks)
    println("$(nameof(typeof(solver))) sweep...")
    p = Progress(num_intervals, 1)
    Threads.@threads for i in eachindex(ks)
        res[i] = solve(solver,new_basis,pts,ks[i])
        next!(p)
    end
    return res
end

"""
    solve_spectrum(solver::ExpandedBoundaryIntegralMethod,billiard::Bi,k1::T,k2::T;dk::Function=(k) -> (0.05 * k^(-1/3))) -> Tuple{Vector{T}, Vector{T}}

Computes the spectrum of the expanded BIM and their corresponding tensions for a given billiard problem within a specified wavenumber range.

# Arguments
- `solver::ExpandedBoundaryIntegralMethod`: The solver configuration for the expanded boundary integral method.
- `billiard::Bi`: The billiard configuration, a subtype of `AbsBilliard`.
- `k1::T`: Starting wavenumber for the spectrum calculation.
- `k2::T`: Ending wavenumber for the spectrum calculation.
- `dk::Function`: Custom function to calculate the wavenumber step size. Defaults to a scaling law inspired by Veble's paper.

# Returns
- `Tuple{Vector{T}, Vector{T}}`: 
  - First element is a vector of corrected eigenvalues (`λ`).
  - Second element is a vector of corresponding tensions.
"""
function solve_spectrum(solver::ExpandedBoundaryIntegralMethod,billiard::Bi,k1::T,k2::T;dk::Function=(k) -> (0.05*k^(-1/3))) where {T<:Real,Bi<:AbsBilliard}
    basis=AbstractHankelBasis()
    bim_solver=BoundaryIntegralMethod(solver.dim_scaling_factor,solver.pts_scaling_factor,solver.sampler,solver.eps,solver.min_dim,solver.min_pts,solver.rule)
    ks=T[]
    k=k1
    while k<k2
        push!(ks,k)
        k+=dk(k)/2 
    end
    λs_all=T[] 
    tensions_all=T[]
    @showprogress for k in ks
        λs,tensions=solve(solver,basis,evaluate_points(bim_solver,billiard,k),k,dk(k))
        if !isempty(λs)
            append!(λs_all,λs)
            append!(tensions_all,tensions) 
        end
    end
    if isempty(λs_all) # Handle case of no eigenvalues found
        λs_all=[0.0]
        tensions_all=[0.0]
    end
    return λs_all,tensions_all
end