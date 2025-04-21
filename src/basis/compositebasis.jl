include("evanescent/evanescent_pw.jl")
include("fourierbessel/corneradapted.jl")
include("planewaves/realplanewaves.jl")

#############################
###### COMPOSITE BASIS ######
#############################

# Neccesery functions to correctly add to a main basis the evanescent plave wave basis. The dim is only for the main basis, the indices for the evanescent basis are determined directly as 1:basis.evanescent.dim due to compatibility reasons.

struct CompositeBasis{T<:Real,Ba<:AbsBasis} <: AbsBasis
    main::Ba
    evanescent::EvanescentPlaneWaves{T}
end

# dim corresponds to the main basis, evanescent basis has custom dim scaling based on k. dim in evanescent is placeholder
function resize_basis(basis::CompositeBasis,billiard::Bi,dim::Int,k) where {Bi<:AbsBilliard}
    return CompositeBasis(resize_basis(basis.main,billiard,dim,k),resize_basis(basis.evanescent,billiard,dim,k))
end

function basis_fun(basis::CompositeBasis{T},indices::AbstractArray,k::T,pts::AbstractArray;multithreaded::Bool=true) where {T<:Real}
   f_main=basis_fun(basis.main,indices,k,pts;multithreaded=multithreaded)
   f_epw=basis_fun(basis.evanescent,1:basis.evanescent.dim,k,pts;multithreaded=multithreaded)
   return reduce(hcat,[f_main,f_epw])
end

function dk_fun(basis::CompositeBasis{T},indices::AbstractArray,k::T,pts::AbstractArray;multithreaded::Bool=true) where {T<:Real}
    f_main=dk_fun(basis.main,indices,k,pts;multithreaded=multithreaded)
    f_epw=dk_fun(basis.evanescent,1:basis.evanescent.dim,k,pts;multithreaded=multithreaded)
    return reduce(hcat,[f_main,f_epw])
end

function gradient(basis::CompositeBasis{T},indices::AbstractArray,k::T,pts::AbstractArray;multithreaded::Bool=true) where {T<:Real}
    main_dX,main_dY=gradient(basis.main,indices,k,pts;multithreaded=multithreaded)
    epw_dX,epw_dY=gradient(basis.evanescent,1:basis.evanescent.dim,k,pts;multithreaded=multithreaded)
    return reduce(hcat,[main_dX,epw_dX]),reduce(hcat,[main_dY,epw_dY])
end

function basis_and_gradient(basis::CompositeBasis{T},indices::AbstractArray,k::T,pts::AbstractArray;multithreaded::Bool=true) where {T<:Real}
    main_vec=basis_fun(basis.main,indices,k,pts;multithreaded=multithreaded)
    epw_vec=basis_fun(basis.evanescent,1:basis.evanescent.dim,k,pts;multithreaded=multithreaded)
    main_dX,main_dY=gradient(basis.main,indices,k,pts;multithreaded=multithreaded)
    epw_dX,epw_dY=gradient(basis.evanescent,1:basis.evanescent.dim,k,pts;multithreaded=multithreaded)
    return main_vec.+epw_vec,main_dX.+epw_dX,main_dY.+epw_dY
end
