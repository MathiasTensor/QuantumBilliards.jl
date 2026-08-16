#include("../abstracttypes.jl")
#include("../utils/typeutils.jl")

"""
    GaussianRandomState{K,T} <: AbsState

`GaussianRandomState` is a concrete type representing a random state with
Gaussian-distributed coefficients, used e.g. as a reference ensemble for
random-wave conjecture studies.

## Description
Unlike [`Eigenstate`](@ref) and [`BasisState`](@ref), a `GaussianRandomState`
is not associated with a specific basis; its coefficient vector `vec` is
simply drawn from a standard normal distribution.

## Attributes
* `k`: The wavenumber associated with the random state.
* `k_basis`: The wavenumber associated with the random state (equal to `k`).
* `vec`: Gaussian-distributed random coefficients.
* `dim`: Dimension of `vec`.
* `eps`: Numerical precision threshold, given by `set_precision`.
"""
struct GaussianRandomState{K,T} <: AbsState where {K<:Number, T<:Real}
    k::K
    k_basis::K
    vec::Vector{T}
    dim::Int64
    eps::T
    #basis type
end

"""
    GaussianRandomState(k, dim) → state::GaussianRandomState

Construct a [`GaussianRandomState`](@ref) of dimension `dim` at wavenumber
`k`, with coefficients drawn independently from a standard normal
distribution.

## Arguments
* `k`: The wavenumber associated with the random state.
* `dim`: The number of random coefficients to generate.

## Returns
*  `state` : A new [`GaussianRandomState`](@ref).
"""
function GaussianRandomState(k,dim)
    d = Distributions.Normal()
    vec = rand(d, dim)
    eps = set_precision(k)
    #norm = sum(abs.(vec))
    return GaussianRandomState(k,k, vec, dim,eps)
end