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

function solve_spectrum(solver::ExpandedBoundaryIntegralMethod,basis::Ba,billiard::Bi,k,dk) where {Ba<:AbsBasis,Bi<:AbsBilliard}
    
end