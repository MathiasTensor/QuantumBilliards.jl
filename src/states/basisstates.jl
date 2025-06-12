"""
    BasisState{K,T,Ba} <: StationaryState 

Constructs a state that represent a single indexed basis element that is useful for visualizing it/plotting.

# Fields
- `k`: The wave vector of the basis state.
- `k_basis`: The wave vector of the basis.
- `vec`: The vector representation of the basis state.
- `idx`: The index of the basis state in the basis.
- `dim`: The dimension of the basis.
- `eps`: The precision of calculations.
- `basis`: The basis in which the state is defined.
"""
struct BasisState{K,T,Ba} <: StationaryState 
    k::K
    k_basis::K
    vec::Vector{T}
    idx::Int64
    dim::Int64
    eps::T
    basis::Ba
end

"""
    BasisState(basis::Ba, k::T, i::Integer) where {T<:Real, Ba<:AbsBasis} 

Constructs the basis state of the given index i.

# Arguements
- `basis`: The basis in which the state is defined.
- `k`: The wave vector of the basis state.
- `i`: The index of the basis state in the basis.

# Returns
-`BasisState` object representing the basis state.
"""
function BasisState(basis::Ba, k::T, i::Integer) where {T<:Real, Ba<:AbsBasis} 
    dim = basis.dim
    typ = typeof(k)
    eps = set_precision(k)
    vec = zeros(typ,dim)
    vec[i] = one(typ)
    return BasisState(k,k, vec, i, dim, eps, basis)
end
