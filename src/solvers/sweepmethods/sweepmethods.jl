#include("../../abstracttypes.jl")
#include("../../utils/billiardutils.jl")
#include("decompositions.jl")
#include("../samplers.jl")
include("particularsolutionsmethod.jl")
include("decompositionmethod.jl")
include("boundaryintegralmethod.jl")
include("expanded_boundary_integral_method.jl")
using LinearAlgebra, Optim, ProgressMeter

"""
    solve_wavenumber(solver::SweepSolver,basis::AbsBasis,billiard::AbsBilliard,k::Real,dk::Real) -> Tuple{Real, Real}

Solves for the wavenumbers `k0` its corresponding tension `t0` within a given range `dk`. It returns the one with the lowest tension and not all those that are in the interval.

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
function solve_wavenumber(solver::SweepSolver,basis::AbsBasis,billiard::AbsBilliard,k,dk)
    dim=max(solver.min_dim,round(Int,billiard.length*k*solver.dim_scaling_factor/(2*pi)))
    new_basis=resize_basis(basis,billiard,dim,k)
    pts=evaluate_points(solver,billiard,k)
    function f(k)
        return solve(solver,new_basis,pts,k)
    end
    res=optimize(f,k-0.5*dk,k+0.5*dk)
    k0,t0=res.minimizer,res.minimum
    return k0,t0
end

"""
    k_sweep(solver::SweepSolver,basis::AbsBasis,billiard::AbsBilliard,ks::Vector{Real}) -> Vector{Real}

Performs a sweep over a range of wavenumbers `ks` and computes tensions for `res` each. If one wants to use different kernels in the BoundaryIntegralMethod one needs to change the kernel_fun which is the Second Layer potential of the original differential equation. By default the SL potential of the Helmholtz equation is used.

# Arguments
- `solver::SweepSolver`: The solver configuration for performing the sweep.
- `basis::AbsBasis`: The basis to be used. Can use also the AbstractHankelBasis() when solver is `BoundaryIntegralMethod`.
- `billiard::AbsBilliard`: The billiard configuration.
- `ks::Vector{Real}`: Vector of wavenumbers over which to perform the sweep.
- `kernel_fun::Union{Symbol, Function}`: Kernel function to use in the boundary integral method. Defaults to `:default`.

# Returns
- `Vector{Real}`: Tensions of the `solve` function for each wavenumber in `ks`.
"""
function k_sweep(solver::SweepSolver,basis::AbsBasis,billiard::AbsBilliard,ks;kernel_fun::Union{Symbol,Function}=:default)
    k=maximum(ks)
    dim=max(solver.min_dim,round(Int,billiard.length*k*solver.dim_scaling_factor/(2*pi)))
    new_basis=resize_basis(basis,billiard,dim,k)
    pts=evaluate_points(solver, billiard, k)
    res=similar(ks)
    num_intervals=length(ks)
    println("$(nameof(typeof(solver))) sweep...")
    p=Progress(num_intervals,1)
    if solver isa BoundaryIntegralMethod
        res[1]=solve_INFO(solver,new_basis,pts,ks[1],kernel_fun=kernel_fun)
        for i in eachindex(ks)[2:end]
            res[i]=solve(solver,new_basis,pts,ks[i],kernel_fun=kernel_fun)
            next!(p)
        end
    else
        res[1]=solve_INFO(solver,new_basis,pts,ks[1],kernel_fun=kernel_fun)
        for i in eachindex(ks)[2:end]
            res[i]=solve(solver,new_basis,pts,ks[i])
            next!(p)
        end
    end
    return res
end
