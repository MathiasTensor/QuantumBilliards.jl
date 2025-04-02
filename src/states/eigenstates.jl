
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

"""
    Eigenstate(k::T, vec::Vector{T}, ten::T, basis::Ba, billiard::Bi) where {T<:Real, Ba<:AbsBasis, Bi<:AbsBilliard}

Constructs an `Eigenstate` object representing a stationary state of the wavefunction.

# Arguments
- `k::T`: The wavenumber of the eigenstate.
- `vec::Vector{T}`: The coefficients of the linear expansion of the wavefunction in the given basis.
- `ten::T`: The tension associated with the eigenstate.
- `basis::Ba`: The basis in which the eigenstate is represented.
- `billiard::Bi`: The billiard domain for the eigenstate.

# Returns
An `Eigenstate` object with normalized coefficients.
"""
function Eigenstate(k,vec,ten,basis,billiard)  
    eps=set_precision(vec[1])
    if eltype(vec) <: Real
        filtered_vec=eltype(vec).([abs(v)>eps ? v : zero(vec[1]) for v in vec])
    else 
        filtered_vec=vec
    end
    return Eigenstate(k,k,filtered_vec,ten,length(vec),eps,basis,billiard)
end

"""
    Eigenstate(k::T, k_basis::T, vec::Vector{T}, ten::T, basis::Ba, billiard::Bi) where {T<:Real, Ba<:AbsBasis, Bi<:AbsBilliard}

Constructs an `Eigenstate` object, allowing for separate wavenumbers for the eigenstate (`k`) and its basis (`k_basis`).

# Arguments
- `k::T`: The wavenumber of the eigenstate.
- `k_basis::T`: The wavenumber associated with the basis.
- `vec::Vector{T}`: The coefficients of the linear expansion of the wavefunction in the given basis.
- `ten::T`: The tension associated with the eigenstate.
- `basis::Ba`: The basis in which the eigenstate is represented.
- `billiard::Bi`: The billiard domain for the eigenstate.

# Returns
An `Eigenstate` object with normalized coefficients.
"""
function Eigenstate(k,k_basis,vec,ten,basis,billiard)  
    eps=set_precision(vec[1])
    if eltype(vec) <: Real
        filtered_vec=eltype(vec).([abs(v)>eps ? v : zero(vec[1]) for v in vec])
    else 
        filtered_vec=vec
    end
    return Eigenstate(k,k_basis,filtered_vec,ten,length(vec),eps,basis,billiard)
end

"""
    compute_eigenstate(solver::SweepSolver, basis::AbsBasis, billiard::AbsBilliard, k::T) where {T<:Real}

Computes a single eigenstate for a given wavenumber `k`.

# Arguments
- `solver::SweepSolver`: The solver object to compute the eigenstate.
- `basis::AbsBasis`: The basis in which the eigenstate is represented.
- `billiard::AbsBilliard`: The billiard domain for the eigenstate.
- `k::T`: The wavenumber of the eigenstate.

# Returns
An `Eigenstate` object representing the computed eigenstate.
"""
function compute_eigenstate(solver::SweepSolver,basis::AbsBasis,billiard::AbsBilliard,k)
    L=billiard.length
    dim=max(solver.min_dim,round(Int, L*k*solver.dim_scaling_factor/(2*pi)))
    basis_new=resize_basis(basis,billiard,dim,k)
    pts=evaluate_points(solver,billiard,k)
    ten,vec=solve_vect(solver,basis_new,pts,k)
    return Eigenstate(k,vec,ten,basis_new,billiard)
end

"""
    compute_eigenstate(solver::AcceleratedSolver, basis::AbsBasis, billiard::AbsBilliard, k::T; dk::T=0.1) where {T<:Real}

Computes a single eigenstate for a given wavenumber `k` using an accelerated solver. Based on the inputted k it finds the closest one. In principle this will find an even better precide eigenvalue k for which we get it's basis coefficients expansion (`Eigenstate`).

# Arguments
- `solver::AcceleratedSolver`: The solver object to compute the eigenstate.
- `basis::AbsBasis`: The basis in which the eigenstate is represented.
- `billiard::AbsBilliard`: The billiard domain for the eigenstate.
- `k::T`: The wavenumber of the eigenstate.
- `dk::T`: The step size for the solver (default: `0.1`).

# Returns
An `Eigenstate` object representing the computed eigenstate.
"""
function compute_eigenstate(solver::AcceleratedSolver,basis::AbsBasis,billiard::AbsBilliard,k;dk=0.1)
    L=billiard.length
    dim=max(solver.min_dim,round(Int, L*k*solver.dim_scaling_factor/(2*pi)))
    basis_new=resize_basis(basis,billiard,dim,k)
    pts=evaluate_points(solver,billiard,k)
    ks,tens,X=solve_vectors(solver,basis_new,pts,k,dk)
    idx=findmin(abs.(ks.-k))[2]
    k_state=ks[idx]
    ten=tens[idx]
    vec=X[:,idx]
    return Eigenstate(k_state,k,vec,ten,basis_new,billiard)
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

"""
    EigenstateBundle(ks::Vector{K}, k_basis::K, X::Matrix{T}, tens::Vector{T}, basis::Ba, billiard::Bi) where {K, T, Ba<:AbsBasis, Bi<:AbsBilliard}

Constructs an `EigenstateBundle` object representing multiple eigenstates arising from the same generalized eigenvalue problem.

# Arguments
- `ks::Vector{K}`: The wavenumbers for the eigenstates.
- `k_basis::K`: The wavenumber associated with the basis.
- `X::Matrix{T}`: The coefficients of the linear expansion for the wavefunctions.
- `tens::Vector{T}`: The tensions associated with the eigenstates.
- `basis::Ba`: The basis in which the eigenstates are represented.
- `billiard::Bi`: The billiard domain for the eigenstates.

# Returns
An `EigenstateBundle` object containing the computed eigenstates.
"""
function EigenstateBundle(ks,k_basis,X,tens,basis,billiard)  
    eps=set_precision(X[1,1])
    type=eltype(X)
    if type <: Real
        filtered_array=type.([abs(x)>eps ? x : zero(type) for x in X])
    else 
        filtered_array=X
    end
    return EigenstateBundle(ks,k_basis,filtered_array,tens,length(X[:,1]),eps,basis,billiard)
end

"""
    compute_eigenstate_bundle(solver::AcceleratedSolver, basis::AbsBasis, billiard::AbsBilliard, k::T; dk::T=0.1, tol::T=1e-5) where {T<:Real}

Computes a bundle of eigenstates within a small interval `[k-dk, k+dk]`.

# Arguments
- `solver::AcceleratedSolver`: The solver object to compute the eigenstates.
- `basis::AbsBasis`: The basis in which the eigenstates are represented.
- `billiard::AbsBilliard`: The billiard domain for the eigenstates.
- `k::T`: The center wavenumber for the computation.
- `dk::T`: The width of the interval for the computation (default: `0.1`).
- `tol::T`: The tolerance for filtering out eigenstates with high tensions (default: `1e-5`).

# Returns
An `EigenstateBundle` object containing the computed eigenstates within the specified interval.
"""
function compute_eigenstate_bundle(solver::AcceleratedSolver, basis::AbsBasis, billiard::AbsBilliard, k; dk = 0.1, tol=1e-5)
    L=billiard.length
    dim=max(solver.min_dim,round(Int,L*k*solver.dim_scaling_factor/(2*pi)))
    basis_new=resize_basis(basis,billiard, dim,k)
    pts=evaluate_points(solver, billiard, k)
    ks,tens,X=solve_vectors(solver,basis_new,pts,k,dk)
    idx= abs.(tens) .< tol
    ks=ks[idx]
    tens=tens[idx]
    X=X[:,idx]
    return EigenstateBundle(ks,k,X,tens,basis_new,billiard)
end

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
function StateData(ks::Vector,X::Vector{Matrix},tens::Vector)  
    # Access the first element of the first vector in X
    eps=set_precision(X[1][1])
    type=eltype(X[1])
    if type <: Real
        filtered_array=[[abs(x) > eps ? x : zero(type) for x in vec] for vec in X] # Filter each vector in X individually
    else
        filtered_array=X
    end
    # dim can be gained for each k in ks separately as they all do not have the same dimension as the X vector of vectors has a different dimension for each k
    return StateData(ks,filtered_array,tens)
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
function solve_state_data_bundle(solver::Sol,basis::Ba,billiard::Bi,k,dk) where {Sol<:AbsSolver, Ba<:AbsBasis, Bi<:AbsBilliard}
    L=billiard.length
    dim=max(solver.min_dim,round(Int,L*k*solver.dim_scaling_factor/(2*pi)))
    basis_new=resize_basis(basis,billiard,dim,k)
    pts=evaluate_points(solver,billiard, k)
    ks,tens,X_matrix=solve_vectors(solver,basis_new,pts,k,dk) # this one filters the ks that are outside k+-dk and gives us the filtered out ks, tensions and X matrix of filtered vectors. No need to store dim as we can get it from the length(X[1])
    # Extract columns of X_matrix and store them as a Vector of Vectors b/c it is easier to merge them in the top function -> compute_spectrum_with_state
    X_vectors=[Vector(col) for col in eachcol(X_matrix)]
    return StateData(ks,X_vectors,tens)
end

#### INTERNAL FUNCTION FOR TESTING TIME AND ALLOCATIONS OF MATRIX CONSTRUCTIONS AND EIGENVALUE SOLVING ####
function solve_state_data_bundle_with_INFO(solver::Sol,basis::Ba,billiard::Bi,k,dk) where {Sol<:AbsSolver, Ba<:AbsBasis, Bi<:AbsBilliard}
    L=billiard.length
    dim=max(solver.min_dim,round(Int,L*k*solver.dim_scaling_factor/(2*pi)))
    @info "Basis resizing..."
    @time basis_new=resize_basis(basis,billiard,dim,k)
    @info "Pts on boundary evaluation..."
    @time pts=evaluate_points(solver,billiard, k)
    @info "F & dF/dk matrix construction..."
    @time F,Fk=construct_matrices(solver,basis_new,pts,k)
    println("Condition num. F: ",cond(F))
    println("Condition num. dF/dk: ",cond(Fk))
    @info "Eigenvalue problem..."
    @time mu,Z,C=generalized_eigen(Symmetric(F),Symmetric(Fk);eps=solver.eps)
    ks,ten=sm_results(mu,k)
    idx=abs.(ks.-k).<dk
    ks=ks[idx]
    ten=ten[idx]
    Z=Z[:,idx]
    X=C*Z #transform into original basis 
    X=(sqrt.(ten))' .* X # Use the automatic normalization via tension values as described in Barnett's thesis. Maybe also use X = X .* reshape(sqrt.(ten), 1, :) ?
    p=sortperm(ks)
    ks,ten,X= ks[p],ten[p],X[:,p]
    # Extract columns of X_matrix and store them as a Vector of Vectors b/c it is easier to merge them in the top function -> compute_spectrum_with_state
    X_vectors=[Vector(col) for col in eachcol(X)]
    return StateData(ks,X_vectors,ten)
end