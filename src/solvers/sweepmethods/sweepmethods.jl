#include("../../abstracttypes.jl")
#include("../../utils/billiardutils.jl")
#include("decompositions.jl")
#include("../samplers.jl")
include("particularsolutionsmethod.jl")
include("decompositionmethod.jl")
include("boundaryintegralmethod.jl")
using LinearAlgebra, Optim, ProgressMeter

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

function solve_spectrum(solver::ExpandedBoundaryIntegralMethod,billiard::Bi,k1,k2) where {Bi<:AbsBilliard}
    basis=AbstractHankelBasis()
    bim_solver=BoundaryIntegralMethod(solver.dim_scaling_factor,solver.pts_scaling_factor,solver.sampler,solver.eps,solver.min_dim,solver.min_pts,solver.rule)
    λs_all = Vector{Float64}()  # To accumulate all eigenvalues
    tensions_all = Vector{Float64}()  # To accumulate corresponding tensions
    k = k1
    
    # Iterate over the wavenumber range
    @showprogress while k < k2
        dk = 0.025 * k^(-1/3)  # Interval size from Veble's paper
        λs, tensions = solve(solver, basis, evaluate_points(bim_solver, billiard, k), k, dk)
        
        # Append results only if there are valid eigenvalues
        if !isempty(λs)
            append!(λs_all, λs)  # Append eigenvalues
            append!(tensions_all, tensions)  # Append corresponding tensions
        end
        
        # Increment to the next interval
        k += dk
    end
    
    # Ensure both lists have the same length
    if length(λs_all) != length(tensions_all)
        error("Mismatch between lengths of eigenvalues and tensions.")
    end
    
    # Handle case of no eigenvalues found
    if isempty(λs_all)
        λs_all = [0.0]  # Assign a single zero value to avoid issues with `log`
        tensions_all = [0.0]
    end
    
    return λs_all, tensions_all
end