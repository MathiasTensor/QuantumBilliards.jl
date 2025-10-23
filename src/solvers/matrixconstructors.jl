
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
@inline function filter_matrix!(M;ϵ=eps(real(eltype(M))))
    T=eltype(M)
    @inbounds @simd for t in eachindex(M)
        if abs(M[t])<=ϵ; M[t]=zero(T); end
    end
    return M
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
    dim=basis.dim
    B=basis_fun(basis,1:dim,k,pts,multithreaded=multithreaded)
    return filter_matrix!(B)
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
    dim=basis.dim
    dB_dx,dB_dy=gradient(basis,1:dim,k,pts;multithreaded=multithreaded)
    return filter_matrix!(dB_dx),filter_matrix!(dB_dy)
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
    dim=basis.dim
    B,dB_dx,dB_dy=basis_and_gradient(basis,1:dim,k,pts;multithreaded=multithreaded)
    return filter_matrix!(B),filter_matrix!(dB_dx),filter_matrix!(dB_dy)
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
    dim=basis.dim
    dB_dk=dk_fun(basis,1:dim,k,pts;multithreaded=multithreaded)
    return filter_matrix!(dB_dk)
end

# INTERNAL FUNCTIONS
###################################################
#### INPLACE FUNCTIONS FOR MATRIX CONSTRUCTION ####
###################################################


# scale rows of A by weights w (inplace)
@inline function _scale_rows!(A::AbstractMatrix,w)
    @inbounds for i in axes(A,1)
        @views A[i,:].*=w[i]
    end
    return A
end

@inline function scale_cols!(Y::AbstractMatrix, X::AbstractMatrix, s::AbstractVector)
    @assert size(X,2)==length(s)
    @inbounds for j in eachindex(s)
        @views Y[:,j].=X[:,j].*s[j]
    end
    return Y
end

# sqrt(W) scaling for syrk type problems of the form A'*W*A -> (A'*sqrt(W))*(sqrt(W)*A) -> BLAS.syrk!
@inline function _scale_rows_sqrtw!(A::AbstractMatrix,w,nsym)
    α=sqrt(one(eltype(w))*nsym)
    @inbounds for i in axes(A,1)
        fi=α*sqrt(w[i])
        for j in axes(A,2)
            A[i,j]*=fi
        end
    end
    return A
end

# mirror upper triangle into lower (inplace)
@inline function _symmetrize_from_upper!(S::StridedMatrix)
    @inbounds for j in axes(S,2)
        for i in 1:j-1
            S[j,i]=S[i,j]
        end
    end
    return S
end

# dX <- nx*dX + ny*dY (done inplace on dX)
@inline function _build_Bn_inplace!(dX::AbstractMatrix,dY::AbstractMatrix,normals)
    @inbounds for i in axes(dX,1)
        nx=normals[i][1]
        ny=normals[i][2]
        for j in axes(dX, 2)
            dX[i,j]=nx*dX[i,j]+ny*dY[i,j]
        end
    end
    return dX
end