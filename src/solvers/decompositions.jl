using LinearAlgebra
#=
function GESVDVALS(A, B; eps=1e-14)
    M = reduce(vcat, [A, B]) #concatenate columns
    Q, R =  qr(M)
    _ , sv_r, Vt_r = svd(R)
    mask = sv_r .> eps
    #println(mask)
    V1t = Vt_r[ : , mask]

    return svdvals((A * V1t), (B * V1t))
end
=#

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
    
    d, S = eigen(A)
    idx = (d/maximum(d)) .> eps
    q = 1.0 ./ sqrt.(d[idx])
    #println(length(q))
    C = q' .* S[:,idx] 
    D = B * C
    #println(size(D))
    E = Symmetric(C' * D)
    #println(size(E))
    mu, Z = eigen(E) #check eigenvectors
    return mu, Z, C
    
    #=
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
    =#
end

"""
    generalized_eigvals(A::Symmetric,B::Symmetric;eps=1e-15)

Computes the generalized eigenvalues of the system `A * x = λ * B * x`
using a truncated basis where eigenvalues of `A` smaller than `eps * max(eigenvalues(A))` 
are ignored. This optimized implementation minimizes memory allocations.

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
    
    d, S = eigen(A)
    idx = (d/maximum(d)) .> eps
    q = 1.0 ./ sqrt.(d[idx])
    #println(length(q))
    C = q' .* S[:,idx] 
    #println(size(C))
    D = B * C
    #println(size(D))
    E = Symmetric(C' * D)
    #println(size(E))
    mu = eigvals(E)
    return mu
    
    #=
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
    =#
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
function adjust_scaling_and_samplers(solver::AbsSolver, billiard::AbsBilliard)
    bs = solver.pts_scaling_factor
    samplers = solver.sampler
    default = samplers[1]
    n_curves = length(billiard.fundamental_boundary)
    b_min = minimum(bs)
    while length(bs)<n_curves
        push!(bs, b_min)
    end
    
    while length(samplers)<n_curves
        push!(samplers, default)
    end
    return bs, samplers
end
