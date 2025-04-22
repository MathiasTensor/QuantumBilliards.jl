include("evanescent/evanescent_pw.jl")
include("fourierbessel/corneradapted.jl")
include("planewaves/realplanewaves.jl")

#############################
###### COMPOSITE BASIS ######
#############################

# Neccesery functions to correctly add to a main basis the evanescent plave wave basis. The dim is only for the main basis, the indices for the evanescent basis are determined directly as 1:basis.evanescent.dim due to compatibility reasons.

"""
    CompositeBasis{T<:Real, Ba<:AbsBasis} <: AbsBasis

A composite basis type that combines a primary (main) basis and an evanescent plane wave basis.

# Fields
- `main::Ba`: The main basis, subtype of `AbsBasis`.
- `evanescent::EvanescentPlaneWaves{T}`: The evanescent basis with elements parameterized by real type `T`.
"""
struct CompositeBasis{T<:Real,Ba<:AbsBasis} <: AbsBasis
    dim::Int
    main::Ba
    evanescent::EvanescentPlaneWaves{T}
    symmetries::Union{Vector{Any},Nothing}
end

"""
    CompositeBasis(main::Ba, evanescent::EvanescentPlaneWaves{T}) -> CompositeBasis{T,Ba}

Constructs a `CompositeBasis` from a main basis and an evanescent plane wave basis.

# Arguments
- `main::Ba`: The main basis, which must be a subtype of `AbsBasis`. Its `dim` field sets the overall dimension of the composite basis.
- `evanescent::EvanescentPlaneWaves{T}`: The evanescent plane wave basis, parameterized by real type `T`.

# Returns
- `CompositeBasis{T,Ba}`: A new composite basis object containing both `main` and `evanescent` basis components.
"""
function CompositeBasis(main::Ba,evanescent::EvanescentPlaneWaves{T}) where {T<:Real,Ba<:AbsBasis}
    return CompositeBasis{T,typeof(main)}(main.dim+evanescent.dim,main,evanescent,main.symmetries)
end

# dim corresponds to the main basis, evanescent basis has custom dim scaling based on k. dim in evanescent is placeholder
"""
    resize_basis(basis::CompositeBasis, billiard::Bi, dim::Int, k) -> CompositeBasis
 
Resizes both the main and evanescent components of the composite basis. The new basis must hold the dim field as the total dimension, so add the dim of the main basis and add the dim of the evanescent basis. This is to ensure proper scaling is made (Only the dim of the main basis is b and k dependant, the dim of the evanescent basis is dependant only on k)
 
# Arguments
- `basis`: A `CompositeBasis` instance.
- `billiard`: An object of type `<: AbsBilliard`.
- `dim::Int`: Number of basis functions to retain in the main basis. The evanescent basis is resized automatically based on internal rules.
- `k`: Wavenumber used for scaling both bases.
 
# Returns
- `CompositeBasis`: A new instance of the composite basis with resized components.
"""
function resize_basis(basis::CompositeBasis,billiard::Bi,dim::Int,k) where {Bi<:AbsBilliard}
    # dim in resize_basis(basis.evanescent,billiard,dim,k) is just a placeholder as dim is determined uniqely with the max_i algorithm in evanescent_pw.jl
    return CompositeBasis(dim+basis.evanescent.dim,resize_basis(basis.main,billiard,dim,k),resize_basis(basis.evanescent,billiard,dim,k),basis.main.symmetries)
end

"""
    basis_fun(basis::CompositeBasis{T}, indices::AbstractArray, k::T, pts::AbstractArray; multithreaded=true) -> Matrix{Complex{T}}

Evaluates the composite basis functions (main + evanescent) at given spatial points.

# Arguments
- `basis::CompositeBasis{T}`: The composite basis structure.
- `indices`: Index range for the full basis (including evanescent tail).
- `k::T`: The wavenumber.
- `pts::AbstractArray`: Points at which the functions are evaluated.
- `multithreaded::Bool=true`: Whether to use multithreaded execution.

# Returns
- `Matrix{Complex{T}}`: Matrix where each column corresponds to a basis function evaluated at all points.
"""
function basis_fun(basis::CompositeBasis{T},indices::AbstractArray,k::T,pts::AbstractArray;multithreaded::Bool=true) where {T<:Real}
    # indices_main = indices - epw indices
    indices=indices[1:end-basis.evanescent.dim]
    f_main=basis_fun(basis.main,indices,k,pts;multithreaded=multithreaded)
    f_epw=basis_fun(basis.evanescent,1:basis.evanescent.dim,k,pts;multithreaded=multithreaded)
    return reduce(hcat,[f_main,f_epw])
end

"""
    dk_fun(basis::CompositeBasis{T}, indices::AbstractArray, k::T, pts::AbstractArray; multithreaded=true) -> Matrix{Complex{T}}

Computes derivatives of composite basis functions with respect to wavenumber `k`.

# Arguments
- `basis::CompositeBasis{T}`: The composite basis structure.
- `indices`: Index range for the full basis (including evanescent tail).
- `k::T`: The wavenumber.
- `pts::AbstractArray`: Points at which the functions are evaluated.
- `multithreaded::Bool=true`: Whether to use multithreaded execution.

# Returns
- `Matrix{Complex{T}}`: Derivatives of basis functions w.r.t. `k` at each point.
"""
function dk_fun(basis::CompositeBasis{T},indices::AbstractArray,k::T,pts::AbstractArray;multithreaded::Bool=true) where {T<:Real}
    # indices_main = indices - epw indices
    indices=indices[1:end-basis.evanescent.dim]
    f_main=dk_fun(basis.main,indices,k,pts;multithreaded=multithreaded)
    f_epw=dk_fun(basis.evanescent,1:basis.evanescent.dim,k,pts;multithreaded=multithreaded)
    return reduce(hcat,[f_main,f_epw])
end

"""
    gradient(basis::CompositeBasis{T}, indices::AbstractArray, k::T, pts::AbstractArray; multithreaded=true) -> Tuple{Matrix{T}, Matrix{T}}

Computes gradients of the composite basis in x and y directions.

# Arguments
- `basis::CompositeBasis{T}`: The composite basis structure.
- `indices`: Index range for the full basis (including evanescent tail).
- `k::T`: The wavenumber.
- `pts::AbstractArray`: Points at which the gradients are computed.
- `multithreaded::Bool=true`: Whether to use multithreaded execution.

# Returns
- `Tuple{Matrix{T}, Matrix{T}}`: A tuple `(dx, dy)` representing the gradients in the x and y directions.
"""
function gradient(basis::CompositeBasis{T},indices::AbstractArray,k::T,pts::AbstractArray;multithreaded::Bool=true) where {T<:Real}
    # indices_main = indices - epw indices
    indices=indices[1:end-basis.evanescent.dim]
    main_dX,main_dY=gradient(basis.main,indices,k,pts;multithreaded=multithreaded)
    epw_dX,epw_dY=gradient(basis.evanescent,1:basis.evanescent.dim,k,pts;multithreaded=multithreaded)
    return reduce(hcat,[main_dX,epw_dX]),reduce(hcat,[main_dY,epw_dY])
end

"""
    basis_and_gradient(basis::CompositeBasis{T}, indices::AbstractArray, k::T, pts::AbstractArray; multithreaded=true) -> Tuple{Matrix{T}, Matrix{T}, Matrix{T}}

Evaluates both the values and gradients of the composite basis functions.

# Arguments
- `basis::CompositeBasis{T}`: The composite basis structure.
- `indices`: Index range for the full basis (including evanescent tail).
- `k::T`: The wavenumber.
- `pts::AbstractArray`: Points at which evaluation is performed.
- `multithreaded::Bool=true`: Whether to use multithreaded execution.

# Returns
- `Tuple{Matrix{T}, Matrix{T}, Matrix{T}}`: A tuple `(values, dx, dy)` containing the evaluated basis functions and their gradients.
"""
function basis_and_gradient(basis::CompositeBasis{T},indices::AbstractArray,k::T,pts::AbstractArray;multithreaded::Bool=true) where {T<:Real}
    # indices_main = indices - epw indices
    indices=indices[1:end-basis.evanescent.dim]
    main_vec=basis_fun(basis.main,indices,k,pts;multithreaded=multithreaded)
    epw_vec=basis_fun(basis.evanescent,1:basis.evanescent.dim,k,pts;multithreaded=multithreaded)
    main_dX,main_dY=gradient(basis.main,indices,k,pts;multithreaded=multithreaded)
    epw_dX,epw_dY=gradient(basis.evanescent,1:basis.evanescent.dim,k,pts;multithreaded=multithreaded)
    return reduce(hcat,[main_vec,epw_vec]),reduce(hcat,[main_dX,epw_dX]),reduce(hcat,[main_dY,epw_dY])
end
