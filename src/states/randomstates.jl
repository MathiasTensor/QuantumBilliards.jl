#include("../abstracttypes.jl")
#include("../utils/typeutils.jl")

using Random, Distributions

"""
    GaussianRandomState{K,T} <: AbsState where {K<:Number, T<:Real}

A struct representing the case where we wish to visualize the Gaussian distributed coefficients of the vector expansion.

# Fields
- `k::K`: The wavevector (just for precision type and compatibility).
- `k_basis::K`: The basis wavevector (just for compatibility).
- `vec::Vector{T}`: The vector of Gaussian coefficients.
- `dim::Int64`: The dimension of the vector.
- `eps::T`: The precision of the wavevector from k.
- `basis<:AbsBasis`: The underlying basis we will construct
"""
struct GaussianRandomState{K,T} <: AbsState where {K<:Number, T<:Real, Ba<:AbsBasis, Bi<:AbsBilliard}
    k::K
    billiard::Bi
    k_basis::K
    vec::Vector{T}
    dim::Int64
    eps::T
    basis::Ba
end

"""
    GaussianRandomState(k::T,dim::Integer) where {T<:Real}

Constructs the Gaussian random state with the linear expansion coefficients of a given basis. This should create an instance of `<:AbsState` so it is compatible as a true `Eigenstate`.

# Arguments
- `k::T`: The wavevector (just for precision type and compatibility).
- `billiard::Bi`: The billiard object.
- `dim::Integer`: The dimension of the vector.
- `basis::Ba`: The underlying basis we will use for construction.

# Returns
- A `GaussianRandomState` instance with Gaussian coefficients.
"""
function GaussianRandomState(k::T, billiard::Bi, dim::Integer, basis::Ba) where {T<:Real, Ba<:AbsBasis, Bi<:AbsBilliard}
    d = Distributions.Normal()
    vec = rand(d, N)
    eps = set_precision(k)
    return GaussianRandomState(k,billiard,k,vec,dim,eps,basis)
end