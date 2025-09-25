
"""
    filter_matrix!(M::AbstractMatrix; ϵ::Real = eps(eltype(M)))

Filters a matrix `M` by setting all elements with an absolute value less than or equal to `ϵ` to zero.
The reason we do this is to get rid of numerical artifacts that would make the final wavefunction have anomalies.

# Arguments
- `M::Matrix`: The matrix to filter.
- `ϵ::<:Real`: The threshold for filtering small values (default: `eps(eltype(M))`).

# Returns
The modified matrix `M` with small values replaced by zero.
"""
function filter_matrix!(M;ϵ=eps(real(eltype(M))))
    typ=eltype(M)
    k=1
    @inbounds Threads.@threads for t in eachindex(M)
        if abs.(M[t])<=ϵ
            M[t]=zero(typ)
            k+=1
        end
    end
    return(M)
end

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