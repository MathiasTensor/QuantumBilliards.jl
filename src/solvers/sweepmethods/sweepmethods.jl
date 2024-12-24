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

function solve_spectrum(solver::ExpandedBoundaryIntegralMethod,billiard::Bi,k1,k2;dk::Function=(k) -> (0.025*k^(-1/3))) where {Bi<:AbsBilliard}
    basis=AbstractHankelBasis()
    bim_solver=BoundaryIntegralMethod(solver.dim_scaling_factor,solver.pts_scaling_factor,solver.sampler,solver.eps,solver.min_dim,solver.min_pts,solver.rule)
    ks=[]
    k=k1
    while k < k2
        push!(ks, k)
        k+=dk(k)  # Increment by interval size from Veble's paper
    end
    λs_all=Float64[] 
    tensions_all=Float64[]
    @showprogress for k in ks
        λs, tensions = solve(solver,basis,evaluate_points(bim_solver,billiard,k),k,dk(k)/2)
        if !isempty(λs)
            append!(λs_all,λs)
            append!(tensions_all,tensions) 
        end
    end
    if length(λs_all) != length(tensions_all)
        error("Mismatch between lengths of eigenvalues and tensions.")
    end
    if isempty(λs_all) # Handle case of no eigenvalues found
        λs_all=[0.0]  # Assign a single zero value to avoid issues with `log`
        tensions_all=[0.0]
    end
    
    return λs_all,tensions_all
end