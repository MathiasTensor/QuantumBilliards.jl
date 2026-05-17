#include("../abstracttypes.jl")
#include("../utils/billiardutils.jl")
#include("../utils/typeutils.jl")

struct Eigenstate{K,T,S,Bi,Ba} <: StationaryState
    k::K
    k_basis::K
    vec::Vector{K}
    ten::T
    dim::Int64
    eps::T
    solver::S
    basis::Ba
    billiard::Bi
end

function Eigenstate(k, vec, ten, solver, basis, billiard)  
    eps = set_precision(vec[1])
    if eltype(vec) <: Real
        filtered_vec = eltype(vec).([abs(v)>eps ? v : zero(vec[1]) for v in vec])
    else 
        filtered_vec = vec
    end
    return Eigenstate(k, k, filtered_vec,ten, length(vec), eps, solver, basis, billiard)
end

function Eigenstate(k, k_basis, vec, ten, solver, basis, billiard)  
    eps = set_precision(vec[1])
    if eltype(vec) <: Real
        filtered_vec = eltype(vec).([abs(v)>eps ? v : zero(vec[1]) for v in vec])
    else 
        filtered_vec = vec
    end
    return Eigenstate(k, k_basis, filtered_vec, ten, length(vec), eps, solver, basis, billiard)
end

function compute_eigenstate(solver::SweepSolver, basis::AbsBasis, billiard::AbsBilliard,k; multithreaded = true)
    L = CompositeCurve(get_boundary_curves(billiard)).length
    dim = max(solver.min_dim,round(Int, L*k*solver.dim_scaling_factor/(2*pi)))
    basis_new = resize_basis(basis,billiard, dim, k)
    pts = evaluate_points(solver, billiard, k)
    ten, vec = solve_vect(solver, basis_new, pts, k; multithreaded)
    return Eigenstate(k, vec, ten, solver, basis_new, billiard)
end

function compute_eigenstate(solver::AcceleratedSolver, basis::AbsBasis, billiard::AbsBilliard, k; dk = 0.1, multithreaded = true)
    L = CompositeCurve(get_boundary_curves(billiard)).length
    dim = max(solver.min_dim,round(Int, L*k*solver.dim_scaling_factor/(2*pi)))
    basis_new = resize_basis(basis,billiard,dim,k)
    pts = evaluate_points(solver, billiard, k)
    ks, tens, X = solve_vectors(solver,basis_new, pts, k, dk; multithreaded)
    idx = findmin(abs.(ks.-k))[2]
    k_state = ks[idx]
    ten = tens[idx]
    vec = X[:,idx]
    return Eigenstate(k_state, k, vec, ten, solver, basis_new, billiard)
end
