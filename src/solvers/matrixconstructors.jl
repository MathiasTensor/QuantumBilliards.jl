
"""
    filter_matrix!(M::AbstractMatrix; ϵ::Real = eps(eltype(M)))

Filters a matrix `M` by setting all elements with an absolute value less than or equal to `ϵ` to zero.
The reason we do this is to get rid of numerical artifacts that would make the final wavefunction have anomalues Matrix.

# Arguments
- `M::Matrix`: The matrix to filter.
- `ϵ::<:Real`: The threshold for filtering small values (default: `eps(eltype(M))`).

# Returns
The modified matrix `M` with small values replaced by zero.
"""
function filter_matrix!(M;ϵ=eps(real(eltype(M))))
    type=eltype(M)
    k=1
    @inbounds Threads.@threads for t in eachindex(M)
        if abs.(M[t])<=ϵ
            M[t]=zero(type)
            k+=1
        end
    end
    return(M)
end

#this will be usefull for basis sets containing several functions (plane and evanscent waves etc.)
"""
    basis_matrix(basis::Ba,k,pts::Vector{SVector{2,T}};multithreaded::Bool=true) where {T<:Real,Ba<:AbsBasis}

Computes the basis matrix for a given basis set, wave number, and set of points, and filters out small values.

# Arguments
- `basis::Ba`: The basis object of type `Ba <: AbsBasis`.
- `k::T`: The wavenumber for which the basis functions are evaluated.
- `pts::Vector{SVector{2,T}}`: A vector of 2D points where the basis functions are evaluated.
- `multithreaded::Bool=true`: If the matrix construction should be multithreaded.

# Returns
- `Matrix` : The filtered basis matrix after removing elements smaller than the specified threshold.
"""
function basis_matrix(basis::Ba,k,pts::Vector{SVector{2,T}};multithreaded::Bool=true) where {T<:Real,Ba<:AbsBasis}
    let dim=basis.dim
        B=basis_fun(basis,1:dim,k,pts,multithreaded=multithreaded)
        return filter_matrix!(B)
    end
end

"""
    gradient_matrices(basis::Ba,k,pts::Vector{SVector{2,T}};multithreaded::Bool=true) where {T<:Real,Ba<:AbsBasis}

Computes the gradient matrices (partial derivatives with respect to `x` and `y`) for a given basis, wave number, and set of points, and filters out small values.

# Arguments
- `basis::Ba`: The basis object of type `Ba <: AbsBasis`.
- `k::T`: The wavenumber for which the gradients are computed.
- `pts::Vector{SVector{2,T}}`: A vector of 2D points where the gradients are evaluated.
- `multithreaded::Bool=true`: If the matrix construction should be multithreaded.

# Returns
A tuple `(dB_dx::Matrix, dB_dy::MAtrix)` of filtered gradient matrices with respect to `x` and `y`.
"""
function gradient_matrices(basis::Ba,k,pts::Vector{SVector{2,T}};multithreaded::Bool=true) where {T<:Real,Ba<:AbsBasis}
    let dim=basis.dim
        dB_dx,dB_dy=gradient(basis,1:dim,k,pts;multithreaded=multithreaded)
        return filter_matrix!(dB_dx),filter_matrix!(dB_dy)
    end
end

"""
    basis_and_gradient_matrices(basis::Ba,k,pts::Vector{SVector{2,T}};multithreaded::Bool=true) where {T<:Real,Ba<:AbsBasis}

Computes the basis matrix and its gradient matrices (partial derivatives with respect to `x` and `y`) for a given basis, wave number, and set of points, and filters out small values.

# Arguments
- `basis::Ba`: The basis object of type `Ba <: AbsBasis`.
- `k::T`: The wavenumber for which the basis and gradients are computed.
- `pts::Vector{SVector{2,T}}`: A vector of 2D points where the basis and gradients are evaluated.
- `multithreaded::Bool=true`: If the matrix construction should be multithreaded.

# Returns
A tuple `(B::Matrix, dB_dx::Matrix, dB_dy::Matrix)` of the filtered basis matrix and gradient matrices.
"""
function basis_and_gradient_matrices(basis::Ba,k,pts::Vector{SVector{2,T}};multithreaded::Bool=true) where {T<:Real,Ba<:AbsBasis}
    let dim=basis.dim
        B,dB_dx,dB_dy=basis_and_gradient(basis,1:dim,k,pts;multithreaded=multithreaded)
        return filter_matrix!(B),filter_matrix!(dB_dx),filter_matrix!(dB_dy)
    end
end

"""
    dk_matrix(basis::Ba,k,pts::Vector{SVector{2,T}};multithreaded::Bool=true) where {T<:Real,Ba<:AbsBasis}

Computes the derivative of the basis matrix with respect to the wave number `k` for a given basis, wave number, and set of points, and filters out small values.

# Arguments
- `basis::Ba`: The basis object of type `Ba <: AbsBasis`.
- `k::T`: The wavenumber for which the derivative is computed.
- `pts::Vector{SVector{2,T}}`: A vector of 2D points where the derivative is evaluated.
- `multithreaded::Bool=true`: If the matrix construction should be multithreaded.

# Returns
- `Matrix` : The filtered derivative matrix with respect to the wave number `k`.
"""
function dk_matrix(basis::Ba,k,pts::Vector{SVector{2,T}};multithreaded::Bool=true) where {T<:Real,Ba<:AbsBasis}
    let dim=basis.dim
        dB_dk=dk_fun(basis,1:dim,k,pts;multithreaded=multithreaded)
        return filter_matrix!(dB_dk)
    end
end

























#=
#for now unnecesary
function basis_matrix(basis::Ba, k, pts::Vector{SVector{2,T}}, indices::AbstractArray) where {T<:Real, Ba<:AbsBasis}
    let N = basis.dim
        M =  length(pts)
        B = zeros(T,M,N) 
        B1 = basis_fun(basis,indices,k,pts)
        #println(size(B))
        #println(size(B1))
        for i in indices
            B[:,i] .= B1[:,i]
        end
        return filter_matrix!(B)
    end
end

function basis_and_gradient_matrices(basis::Ba, k, pts::Vector{SVector{2,T}}, indices::AbstractArray) where {T<:Real, Ba<:AbsBasis}
    let N = basis.dim
        M =  length(pts)
        dX = zeros(T,M,N)
        dY = zeros(T,M,N)
        B1 = zeros(T,M,N)
        B, dB_dx, dB_dy = basis_and_gradient(basis,indices,k,pts)
        filter_matrix!(B)
        filter_matrix!(dB_dx)
        filter_matrix!(dB_dy)
        #println(size(B))
        #println(size(B1))
        for i in indices
            B1[:,i] .= B[:,i]
            dX[:,i] .= dB_dx[:,i]
            dY[:,i] .= dB_dy[:,i]
        end
        return B1, dX, dY
    end
end



function gradient_matrices(basis::Ba, k, pts::Vector{SVector{2,T}}, indices::AbstractArray) where {T<:Real, Ba<:AbsBasis}
    let N = basis.dim
        M =  length(pts)
        dX = zeros(T,M,N)
        dY = zeros(T,M,N)  
        dB_dx, dB_dy = gradient(basis,indices,k,pts)
        filter_matrix!(dB_dx)
        filter_matrix!(dB_dy)
        #println(size(B))
        #println(size(B1))
        for i in indices
            dX[:,i] .= dB_dx[:,i]
            dY[:,i] .= dB_dy[:,i]
        end
        return dX, dY
    end
end

function dk_matrix(basis::Ba, k, pts::Vector{SVector{2,T}}, indices::AbstractArray) where {T<:Real, Ba<:AbsBasis}
    let N = basis.dim
        M =  length(pts)
        dB1 = zeros(T,M,N) 
        dB_dk = dk_fun(basis,indices,k,pts)
        filter_matrix!(dB_dk)
        #println(size(B))
        #println(size(B1))
        for i in indices
            dB1[:,i] .= dB_dk[:,i]
        end
        return dB1
    end
end
=#
#=
#rework these
function basis_matrix(basis::Ba, k, x_grid::AbstractArray, y_grid::AbstractArray) where {T<:Real, Ba<:AbsBasis}
    let dim = basis.dim
        pts = collect(SVector(x,y) for y in y_grid for x in x_grid) 
        return basis_fun(basis,1:dim,k,pts)
    end
end

function basis_matrix(basis::Ba, k, x_grid::AbstractArray, y_grid::AbstractArray, indices::AbstractArray) where {T<:Real, Ba<:AbsBasis}
    let N = basis.dim
        M =  length(x_grid)*length(y_grid)
        B = zeros(eltype(x_grid),M,N) 
        pts = collect(SVector(x,y) for y in y_grid for x in x_grid)
        B1 = basis_fun(basis,indices,k,pts)
        #println(size(B))
        #println(size(B1))
        for i in indices
            B[:,i] .= B1[:,i]
        end
        return B
    end
end

function basis_matrix(basis::Ba, k, x_grid::AbstractArray, y_grid::AbstractArray, indices::AbstractArray) where {T<:Real, Ba<:AbsBasis}
    let N = basis.dim
        M =  length(x_grid)*length(y_grid)
        B = zeros(eltype(x_grid),M,N)
        pts = collect(SVector(x,y) for y in y_grid for x in x_grid) 
        B1 = basis_fun(basis,indices,k,pts)
        #println(size(B))
        #println(size(B1))
        for i in indices
            B[:,i] .= B1[:,i]
        end
        return B
    end
end
=#