using LinearAlgebra

"""
    generalized_eigen(A::Symmetric,B::Symmetric;eps=1e-15)

Computes the generalized eigenvalues and eigenvectors of the system `A * x = λ * B * x`
using a truncated basis where eigenvalues of `A` smaller than `eps * max(eigenvalues(A))` 
are ignored. This optimized implementation minimizes memory allocations.

Reference: https://users.flatironinstitute.org/~ahb/thesis_html/node60.html
| Step                                      | Code Line                                                                                     | Explanation                                                                                                      |
|-------------------------------------------|-----------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| Diagonalize A                             | `d, S = eigen(Symmetric(A))`                                                                  | Compute the eigenvalues (`d`) and eigenvectors (`S`) of `A`.                                                    |
| Truncate eigenvectors                     | `idx = d .> eps * maximum(d)`                                                                | Identify eigenvalues greater than `eps * max(eigenvalues(A))`.                                                  |
| Construct truncated eigenvector matrix C  | `C = @view S[:, idx]`                                                                         | Extract eigenvectors corresponding to retained eigenvalues.                                                     |
| Scale eigenvectors by Λ^(-1/2)            | `q = 1.0 ./ sqrt.(d[idx]); C_scaled = C .* q'`                                                | Scale selected eigenvectors by the inverse square root of the retained eigenvalues.                             |
| Form reduced matrix E                     | `mul!(tmp, B, C_scaled); mul!(E, C_scaled', tmp); E = Symmetric(E)`                            | Compute the reduced matrix `E = Λ^(-1/2) * C' * B * C * Λ^(-1/2)`.                                              |
| Solve reduced eigenproblem                | `mu, Z = eigen(Symmetric(E))`                                                                | Solve the reduced eigenproblem for eigenvalues (`mu`) and eigenvectors (`Z`).                                   |

# Arguments
- `A::Symmetric`: Symmetric matrix `A` in the generalized eigenproblem.
- `B::Symmetric`: Symmetric matrix `B` in the generalized eigenproblem.
- `eps`: Relative tolerance for filtering small eigenvalues of `A`. Default is `1e-15`.

# Returns
- `mu::Vector`: Vector of generalized eigenvalues.
- `Z::Matrix`: Matrix of eigenvectors in the reduced space.
- `C_scaled::Matrix`: Scaled eigenvector matrix corresponding to the truncated basis.
"""
function generalized_eigen(A,B;eps=1e-15)
    d,S=eigen(Symmetric(A))
    idx=d.>eps*maximum(d)
    q=1.0./sqrt.(d[idx])
    C=@view S[:,idx]
    C_scaled=C.*q'
    n=size(C_scaled,2)
    tmp=Matrix{eltype(B)}(undef,size(B,1),n)
    E=Matrix{eltype(B)}(undef,n,n)
    mul!(tmp,B,C_scaled)
    mul!(E,C_scaled',tmp)
    mu,Z=eigen(Symmetric(E))
    return mu,Z,C_scaled
end

"""
    generalized_eigvals(A::Symmetric,B::Symmetric;eps=1e-15)

Computes the generalized eigenvalues of the system `A * x = λ * B * x`
using a truncated basis where eigenvalues of `A` smaller than `eps * max(eigenvalues(A))` 
are ignored.

Reference: https://users.flatironinstitute.org/~ahb/thesis_html/node60.html
| Step                                      | Code Line                                                                                     | Explanation                                                                                                      |
|-------------------------------------------|-----------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| Diagonalize A                             | `d, S = eigen(A)`                                                                             | Compute the eigenvalues (`d`) and eigenvectors (`S`) of `A`.                                                    |
| Truncate eigenvectors                     | `idx = d .> eps * maxd`                                                                       | Identify eigenvalues greater than `eps * maxd`.                                                                 |
| Construct truncated eigenvector matrix C  | `C = @view S[:, idx]` or `C = S[:, idx]`                                                     | Extract eigenvectors corresponding to retained eigenvalues.                                                     |
| Scale eigenvectors by Lambda_r^(-1/2)     | `q = 1.0 ./ sqrt.(d[idx]); C_scaled = C .* q'`                                                | Scale selected eigenvectors by the inverse square root of the retained eigenvalues.                             |
| Form reduced matrix G'_r                  | `mul!(tmp, B, C_scaled); mul!(E, C_scaled', tmp); E = Symmetric(E)`                            | Compute the reduced matrix `G'_r = Lambda_r^(-1/2) * V_r^T * G * V_r * Lambda_r^(-1/2)`.                        |
| Solve reduced eigenproblem                | `return eigvals(E)`                                                                           | Solve the reduced eigenproblem for the eigenvalues of the system.                                               |

# Arguments
- `A::Symmetric`: Symmetric matrix `A` in the generalized eigenproblem.
- `B::Symmetric`: Symmetric matrix `B` in the generalized eigenproblem.
- `eps`: Relative tolerance for filtering small eigenvalues of `A`. Default is `1e-15`.

# Returns
- `mu::Vector`: Vector of generalized eigenvalues.
"""
function generalized_eigvals(A,B;eps=1e-15)    
    d,S=eigen(Symmetric(A))
    maxd=maximum(d)
    idx=d.>eps*maxd 
    q=1.0./sqrt.(d[idx])
    C=@view S[:,idx]
    C_scaled=C.*q'
    n=size(C_scaled,2)
    tmp=Matrix{eltype(B)}(undef,size(B,1),n)  
    E=Matrix{eltype(B)}(undef,n,n) 
    mul!(tmp,B,C_scaled)
    mul!(E,C_scaled',tmp)
    return eigvals(Symmetric(E))  
end

"""
    generalized_eigen_all(A::AbstractMatrix, B::AbstractMatrix) -> (Vector{Complex{T}}, Matrix{Complex{T}}, Matrix{Complex{T}}) where T <: Real

Computes the generalized eigenvalues and both left and right eigenvectors of the pair of matrices `(A, B)`. There are no further restrictions on the types of matrices `(A, B)`.

```math
A * u = λ * B * u     ->     A * u = λ * dA/dk * u
```

It is important to filter the eigenvalues λ for Inf or NaN since eigen internally calls ggev3/ggev which uses QZ algorithm which gives the diagonal elements of the triangular form of A and B (they are simulatenously transformed as T_A = Q * A * Z & T_B = Q * B * Z) as vectors α (diagonals of T_A) and vectors β (diagonals of T_B). The key observation here is that matrix B in EBIM is ill-conditioned (cond(B) > 1e16) and singular (since the diagonals are all zero in using the helmholtz kernel) and therefore the QZ algorithm returns many Inf values for β. And when we construct the final eigenvalues as λ=α./β this becomes problematic, hence the need to for this check. Only when we are close to an actual eigenvalue do we have generalized eigenvalues in the dk range where the problem is constructed.

# Arguments
- `A::AbstractMatrix`: Square matrix.
- `B::AbstractMatrix`: Square matrix.

# Returns
- `λ::Vector{Complex{T}}`: Vector of ordered filtered eigenvalues (excluding `NaN` and `Inf` values).
- `VR::Matrix{Complex{T}}`: Complex matrix where each column is a right eigenvector.
- `VL::Matrix{Complex{T}}`: Complex matrix where each column is a left eigenvector.
"""
function generalized_eigen_all(A,B)
    F=eigen(A,B)
    λ=F.values
    VR=F.vectors # right eigenvectors
    F_adj=eigen(A',B') # adjoint problem to find left eigenvectors
    VL=F_adj.vectors 
    valid_indices=.!isnan.(λ).&.!isinf.(λ)  # for singular matrices give NaN λ
    λ=λ[valid_indices]
    VR=VR[:,valid_indices]
    VL=VL[:,valid_indices]
    sort_order=sortperm(abs.(λ)) 
    λ=λ[sort_order]
    VR=VR[:,sort_order]
    VL=VL[:,sort_order]
    return λ,VR,VL
end

"""
    generalized_eigen_all_LAPACK_LEGACY(A::AbstractMatrix, B::AbstractMatrix) -> (Vector{Complex{T}}, Matrix{Complex{T}}, Matrix{Complex{T}}) where T <: Real

Computes the generalized eigenvalues and both left and right eigenvectors of the pair of matrices `(A, B)` using LAPACK's `ggev!` function.

This function is optimized for speed on small matrices (`dim < 350`) and can handle general square matrices.
!!! This function gives differently scaled correct solution for the same eigenvalue to the generalized eigenproblem than `eigen(A,B)` does!!!

```math
A * u = λ * B * u
```

It is important to filter the eigenvalues λ for Inf or NaN since ggev3/ggev which uses QZ algorithm which gives the diagonal elements of the triangular form of A and B (they are simulatenously transformed as T_A = Q * A * Z & T_B = Q * B * Z) as vectors α (diagonals of T_A) and vectors β (diagonals of T_B). The key observation here is that matrix B in EBIM is ill-conditioned (cond(B) > 1e16) and singular (since the diagonals are all zero in using the helmholtz kernel) and therefore the QZ algorithm returns many Inf values for β. And when we construct the final eigenvalues as λ=α./β this becomes problematic, hence the need to for this check. Only when we are close to an actual eigenvalue do we have generalized eigenvalues in the dk range where the problem is constructed.

# Arguments
- `A::AbstractMatrix`: Square matrix.
- `B::AbstractMatrix`: Square matrix.

# Returns
- `λ::Vector{Complex{T}}`: Vector of ordered filtered eigenvalues (excluding `NaN` and `Inf` values).
- `VR::Matrix{Complex{T}}`: Complex matrix where each column is a right eigenvector.
- `VL::Matrix{Complex{T}}`: Complex matrix where each column is a left eigenvector.
"""
function generalized_eigen_all_LAPACK_LEGACY(A,B) 
    if LAPACK.version()<v"3.6.0"
        α,β,VL,VR=LAPACK.ggev!('V','V',A,copy(B)) # dA needs to be copied since we still need it after inplace modification for 2nd order corrections
    else
        α,β,VL,VR=LAPACK.ggev3!('V','V',A,copy(B)) # dA needs to be copied since we still need it after inplace modification for 2nd order corrections
    end
    λ=α./β
    #valid_indices=.!isnan.(λ).&.!isinf.(λ)
    valid_indices=.!isnan.(λ).&.!isinf.(λ).& (abs.(β).>1e-7)
    λ=λ[valid_indices]
    VR=VR[:,valid_indices]
    VL=VL[:,valid_indices]
    sort_order=sortperm(abs.(λ)) 
    λ=λ[sort_order]
    VR=VR[:,sort_order]
    VL=VL[:,sort_order]
    return λ,VR,VL
end

"""
    generalized_eigen_symmetric_LAPACK_LEGACY(A::AbstractMatrix, B::AbstractMatrix) -> (Vector{Real}, Matrix{Complex{T}}, Matrix{Complex{T}}) where T <: Real

! VERY EFFICIENT, at least a 2X improvement over the general LAPACK one and the general eigen!(A,B)!
Computes the generalized eigenvalues and eigenvectors for symmetric or Hermitian matrices `(A, B)` using LAPACK's `sygvd!` function. B MUST BE POSITIVE DEFINITE.

This function is optimized for symmetric and Hermitian matrices. The left eigenvectors are identical to the right eigenvectors. This function gives the same scaled eigenvectors as `eigen(Symmetric(A),Symmetric)`.

```math
A * u = λ * B * u
```

# Arguments
- `A::AbstractMatrix`: Symmetric or Hermitian square matrix. DON'T USE `Symmetric(...)` or `Hermitian(...)` since LAPACK does not support it, use `Matrix(...)`
- `B::AbstractMatrix`: Symmetric or Hermitian square matrix. DON'T USE `Symmetric(...)` or `Hermitian(...)` since LAPACK does not support it, use `Matrix(...)`

# Returns
- `λ::Vector{Real}`: Vector of ordered filtered eigenvalues (excluding `NaN` and `Inf` values).
- `VR::Matrix{Complex{T}}`: Complex matrix where each column is a right eigenvector.
- `VL::Matrix{Complex{T}}`: Same as `VR`, since left eigenvectors = right eigenvectors for symmetric matrices.
"""
function generalized_eigen_symmetric_LAPACK_LEGACY(A,B)
    λ,VR=LAPACK.sygvd!(1,'V','U',A,copy(B)) # dA needs to be copied since we still need it after inplace modification for 2nd order corrections
    valid_indices=.!isnan.(λ).&.!isinf.(λ)
    λ=λ[valid_indices]
    VR=VR[:,valid_indices]
    sort_order=sortperm(abs.(λ)) 
    λ=λ[sort_order]
    VR=VR[:,sort_order]
    return λ,VR,VL
end

"""
    directsum(A::Matrix, B::Matrix)

Constructs the direct sum of two matrices `A` and `B`. The result is a block diagonal matrix 
where `A` occupies the top-left block, `B` occupies the bottom-right block, and the off-diagonal 
blocks are filled with zeros.

```math
⎡A (m × n)  0 (m × q)⎤
⎣0 (p × n)  B (p × q)⎦
```

# Arguments
- `A::Matrix{T}`: A matrix of size `m × n`.
- `B::Matrix{T}`: A matrix of size `p × q`.

# Returns
A block matrix of size `(m + p) × (n + q)`.
"""
directsum(A::Matrix,B::Matrix) = [A zeros(size(A,1), size(B,2)); zeros(size(B,1), size(A,2)) B]

"""
    adjust_scaling_and_samplers(solver::AbsSolver, billiard::AbsBilliard)

Adjusts the scaling factors and samplers of the solver to match the number of fundamental 
boundary curves in the billiard (for each curve one b and sampler). This ensures that the solver has the appropriate number of 
scaling factors and samplers, filling in defaults where necessary.

# Arguments
- `solver::AbsSolver`: The solver whose scaling factors and samplers need adjustment.
- `billiard::AbsBilliard`: The billiard object, which defines the fundamental boundary curves.

# Returns
A tuple `(bs, samplers)` where:
- `bs::Vector{eltype{solver.pts_scaling_factor}}`: The adjusted vector of scaling factors, with length equal to the number of fundamental boundary curves.
- `samplers::Vector{<:AbsSampler}`: The adjusted vector of samplers, with length equal to the number of fundamental boundary curves.
"""
function adjust_scaling_and_samplers(solver::AbsSolver,billiard::AbsBilliard)
    bs=solver.pts_scaling_factor
    samplers=solver.sampler
    default=samplers[1]
    n_curves=length(billiard.fundamental_boundary)
    b_min=minimum(bs)
    while length(bs)<n_curves
        push!(bs,b_min)
    end
    while length(samplers)<n_curves
        push!(samplers,default)
    end
    return bs,samplers
end
