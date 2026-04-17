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

