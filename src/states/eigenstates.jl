#include("../abstracttypes.jl")
#include("../utils/billiardutils.jl")
#include("../utils/typeutils.jl")

struct Eigenstate{K,T,Ba,Bi} <: StationaryState
    k::K
    k_basis::K
    vec::Vector{T}
    ten::T
    dim::Int64
    eps::T
    basis::Ba
    billiard::Bi
end

function Eigenstate(k, vec, ten, basis, billiard)  
    eps = set_precision(vec[1])
    if eltype(vec) <: Real
        filtered_vec = eltype(vec).([abs(v)>eps ? v : zero(vec[1]) for v in vec])
    else 
        filtered_vec = vec
    end
    return Eigenstate(k, k, filtered_vec,ten, length(vec), eps, basis, billiard)
end

function Eigenstate(k, k_basis, vec, ten, basis, billiard)  
    eps = set_precision(vec[1])
    if eltype(vec) <: Real
        filtered_vec = eltype(vec).([abs(v)>eps ? v : zero(vec[1]) for v in vec])
    else 
        filtered_vec = vec
    end
    return Eigenstate(k, k_basis, filtered_vec, ten, length(vec), eps, basis, billiard)
end

function compute_eigenstate(solver::SweepSolver, basis::AbsBasis, billiard::AbsBilliard,k)
    L = billiard.length
    dim = max(solver.min_dim,round(Int, L*k*solver.dim_scaling_factor/(2*pi)))
    basis_new = resize_basis(basis,billiard, dim, k)
    pts = evaluate_points(solver, billiard, k)
    ten, vec = solve_vect(solver, basis_new, pts, k)
    return Eigenstate(k, vec, ten, basis_new, billiard)
end

function compute_eigenstate(solver::AcceleratedSolver, basis::AbsBasis, billiard::AbsBilliard, k; dk = 0.1)
    L = billiard.length
    dim = max(solver.min_dim,round(Int, L*k*solver.dim_scaling_factor/(2*pi)))
    basis_new = resize_basis(basis,billiard,dim,k)
    pts = evaluate_points(solver, billiard, k)
    ks, tens, X = solve_vectors(solver,basis_new, pts, k, dk)
    idx = findmin(abs.(ks.-k))[2]
    k_state = ks[idx]
    ten = tens[idx]
    vec = X[:,idx]
    return Eigenstate(k_state, k, vec, ten, basis_new, billiard)
end

struct EigenstateBundle{K,T,Ba,Bi} <: AbsState 
    ks::Vector{K}
    k_basis::K
    X::Matrix{T}
    tens::Vector{T}
    dim::Int64
    eps::T
    basis::Ba
    billiard::Bi
end

function EigenstateBundle(ks, k_basis, X, tens, basis, billiard)  
    eps = set_precision(X[1,1])
    type = eltype(X)
    if  type <: Real
        filtered_array = type.([abs(x)>eps ? x : zero(type) for x in X])
    else 
        filtered_array = X
    end
    return EigenstateBundle(ks, k_basis, filtered_array, tens, length(X[:,1]), eps, basis, billiard)
end

function compute_eigenstate_bundle(solver::AcceleratedSolver, basis::AbsBasis, billiard::AbsBilliard, k; dk = 0.1, tol=1e-5)
    L = billiard.length
    dim = max(solver.min_dim,round(Int, L*k*solver.dim_scaling_factor/(2*pi)))
    basis_new = resize_basis(basis,billiard, dim,k)
    pts = evaluate_points(solver, billiard, k)
    ks, tens, X = solve_vectors(solver,basis_new, pts, k, dk)
    idx = abs.(tens) .< tol
    ks = ks[idx]
    tens = tens[idx]
    X = X[:,idx]
    return EigenstateBundle(ks, k, X, tens, basis_new, billiard)
end





###### NEW ######


# no need for basis and billiard data due to complex nested hierarchy
"""
    struct StateData{K,T} <: AbsState 

Convenience wrapper for all the relevant results from the computation of a spectrum. It saves the wavenumbers, the tensions and the expansion coefficient for the basis stored as a Vector
"""
struct StateData{K,T} <: AbsState 
    ks::Vector{K}
    X::Vector{Vector{T}}  # Changed from Matrix{T}
    tens::Vector{T}
end

# constructor for the saved data with no billiard or basis information
"""
    StateData(ks::Vector, X::Vector{Matrix}, tens::Vector) :: StateData

Constructor for the convenience wrapper `StateData`. Under the hood it filters the coefficients that are very small (sets them to zero(T) if the val is smaller than eps(T)) so as to get better representation of the wavefunction

# Arguments
- `ks::Vector`: The wavenumbers for which the wavefunction was computed.
- `X::Vector{Matrix}`: The expansion coefficients for the basis stored as a Vector of vectors.
- `tens::Vector`: The tension minima for which the wavefunction was computed.
"""
function StateData(ks::Vector, X::Vector{Matrix}, tens::Vector)  
    # Access the first element of the first vector in X
    eps = set_precision(X[1][1])
    type = eltype(X[1])
    if type <: Real
        filtered_array = [
            [abs(x) > eps ? x : zero(type) for x in vec] for vec in X
        ] # Filter each vector in X individually
    else
        filtered_array = X
    end
    # dim can be gained for each k in ks separately as they all do not have the same dimension as the X vector of vectors has a different dimension for each k
    return StateData(ks, filtered_array, tens)
end

# this is basically the new solve where we incur the smallest penalty for getting the ks and the relevant state information for saving the husimi functions but it is much more efficient than doint it again once we have the eigenvalues
"""
    function solve_state_data_bundle(solver::Sol, basis::Ba, billiard::Bi, k, dk) where {Sol<:AbsSolver, Ba<:AbsBasis, Bi<:AbsBilliard} :: StateData

Solves the generalized eigenvalue problem in a small interval `[k0-dk, k0+dk]` and constructs the `StateData` object in that small interval. This function is iteratively called in the `compute_spectrum` function version that also computes the `StateData` object. The advantage of this version of the function from the regular `solve(solver...)` is that we get the eigenvectors here witjh minimal additional computational cost.

# Arguments
- `solver<:AbsSolver`: The solver object to use for the eigenvalue problem.
- `basis<:AbsBasis`: The basis object to use for the eigenvalue problem.
- `billiard<:AbsBilliard`: The billiard object to use for the eigenvalue problem.
- `k<:Real`: The center of the interval for which to solve the eigenvalue problem.
- `dk<:Real`: The width of the interval for which to solve the eigenvalue problem.

# Returns
A `StateData` object containing the wavenumbers, the tensions and the expansion coefficients for the basis stored as a Vector of Vectors after a generalized eigenvalue problem computation.
"""
function solve_state_data_bundle(solver::Sol, basis::Ba, billiard::Bi, k, dk) where {Sol<:AbsSolver, Ba<:AbsBasis, Bi<:AbsBilliard}
    L = billiard.length
    dim = max(solver.min_dim,round(Int, L*k*solver.dim_scaling_factor/(2*pi)))
    basis_new = resize_basis(basis,billiard, dim,k)
    pts = evaluate_points(solver, billiard, k)
    ks, tens, X_matrix = solve_vectors(solver,basis_new, pts, k, dk) # this one filters the ks that are outside k+-dk and gives us the filtered out ks, tensions and X matrix of filtered vectors. No need to store dim as we can get it from the length(X[1])
    # Extract columns of X_matrix and store them as a Vector of Vectors b/c it is easier to merge them in the top function -> compute_spectrum_with_state
    X_vectors = collect(eachcol(X_matrix))
    return StateData(ks, X_vectors, tens)
end