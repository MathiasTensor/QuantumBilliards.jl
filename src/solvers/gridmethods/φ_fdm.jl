using LinearAlgebra, SparseArrays, Arpack
include("fdm.jl")


"""
    compute_extended_index(x_grid::Vector{T}, y_grid::Vector{T}, mask::Vector{Bool}) where T<:Real

Further reading: https://hal.science/hal-04731164v1/file/phiFD.pdf

Computes an extended index mapping from a Cartesian grid to the active nodes for use in ϕ‐FD methods.
The extended grid Ωₕ is defined as the set of grid nodes that are either inside the domain Ω (as indicated by 
the Boolean vector `mask`) or adjacent (in one of the four cardinal directions) to at least one node inside Ω. 
This extended indexing is necessary because the ϕ‐FD method applies standard finite difference stencils that may 
reach outside Ω, so those extra nodes (even if not strictly inside Ω) must be included to correctly enforce the 
Dirichlet boundary conditions via penalty and stabilization terms.

# Arguments
- `x_grid::Vector{T}`: A vector of x-coordinate values for the grid nodes.
- `y_grid::Vector{T}`: A vector of y-coordinate values for the grid nodes.
- `mask::Vector{Bool}`: A Boolean vector of length (length(x_grid)*length(y_grid)) indicating whether each grid 
  node is inside the domain Ω (`true`) or not (`false`).

# Returns
A tuple `(ext_idx, count)` where:
- `ext_idx::Vector{Int}` is a vector of the same length as `mask` that maps each grid node (by its linear index) 
  to a unique positive integer if the node is active (i.e. either inside Ω or adjacent to an inside node), or 
  0 if not.
- `count::Int` is the total number of active nodes (i.e. the number of nonzero entries in `ext_idx`).

# Details
The function first computes the grid dimensions (Nx and Ny) from the lengths of `x_grid` and `y_grid`. It then 
loops over each node using a helper function `idx(i,j)=i+(j-1)*Nx` to compute the linear index. For each node, 
if the node is not inside Ω (i.e. `mask[lin]` is `false`), the function checks its four immediate neighbors. 
If any neighbor is inside Ω, the node is marked as active. Finally, each active node is assigned a unique 
sequential index and the total count is returned.
"""
function compute_extended_index(x_grid::Vector,y_grid::Vector,mask::Vector{Bool})
    Nx=length(x_grid)
    Ny=length(y_grid)
    ext_idx=zeros(Int,Nx*Ny)
    count=0
    # Helper to convert (i,j) to linear index:
    idx(i,j)=i+(j-1)*Nx
    for j in 1:Ny
        for i in 1:Nx
            lin=idx(i,j)
            # Check if current node is inside or has a neighbor that is inside:
            inside=mask[lin]
            if !inside
                for (di,dj) in [(1,0),(-1,0),(0,1),(0,-1)] # 4 possible directions to go in 2D
                    i2,j2=i+di,j+dj
                    if i2≥1 && i2≤Nx && j2≥1 && j2≤Ny
                        if mask[idx(i2,j2)]
                            inside=true
                            break
                        end
                    end
                end
            end
            if inside
                count+=1
                ext_idx[lin]=count
            end
        end
    end
    return ext_idx,count
end

"""
    phiFD_Hamiltonian(fem::FiniteElementMethod{T}, phi::Function, gamma::T, sigma::T) where T<:Real

Further reading: https://hal.science/hal-04731164v1/file/phiFD.pdf

Constructs the discrete Hamiltonian matrix for the Poisson problem with homogeneous Dirichlet boundary 
conditions using the ϕ-FD (phi-FD) finite difference method. This method is designed to work on a fixed 
Cartesian grid that “covers” a complex geometry described by a level‐set function φ, and it avoids the 
need for body-fitted meshes by working on an extended grid Ωₕ. The extended grid includes all nodes that 
are either inside the physical domain Ω (where φ(x,y) < 0) or adjacent to an inside node. The boundary 
conditions are enforced weakly via two additional terms: a penalty term bₕ and a stabilization term jₕ.

The optimal values of γ and σ can vary quite a bit depending on the geometry and grid resolution. 
In many of the numerical tests reported in the literature on ϕ‐FD methods (for example, in  ￼), for simple, 
smooth geometries (such as circles or rectangles) one might choose γ on the order of 1 and σ around 0.01. 
However, for more complex or highly curved geometries, 
you may need to increase γ and/or σ to robustly enforce the boundary conditions
and control the derivative jumps—this ensures that the penalty is strong enough to force the solution 
to vanish on the boundary while still keeping the matrix well conditioned. 
In practice, one usually tunes these parameters (often via a convergence study) 
so that the error and condition number meet the desired criteria.

# Arguments
- `fem::FiniteElementMethod{T}`: A structure containing grid and domain information. It must include:
- `fem.x_grid::Vector{T}`: The vector of x-coordinates for the grid nodes.
- `fem.y_grid::Vector{T}`: The vector of y-coordinates for the grid nodes.
- `fem.Nx::Int` and `fem.Ny::Int`: The number of grid points in the x and y directions.
- `fem.dx::T` and `fem.dy::T`: The grid spacing in x and y.
- `fem.mask::Vector{Bool}`: A Boolean vector (of length Nx*Ny) indicating if a node is inside Ω.
- `phi::Function`: A level-set function of signature `phi(x::T, y::T) -> T or Bool` that defines the domain Ω 
  via Ω = { (x,y) ∈ O : phi(x,y) < 0 }. This function is also used to weight the penalty term. For example,
  for a rectangle one might define `phi(x,y) = -min(x - a, b - x, y - c, d - y)`, and for a circle, 
  `phi(x,y) = (x - x₀)^2 + (y - y₀)^2 - R^2`.
- `gamma::T`: The penalization parameter. This scalar controls the strength of the penalty term that is 
  applied on edges where adjacent grid nodes lie on opposite sides of the boundary. A larger value of 
  γ forces the discrete solution to adhere more closely to the Dirichlet condition (u = 0 on ∂Ω).
- `sigma::T`: The stabilization parameter. This scalar scales the stabilization term which penalizes 
  large discrete second derivatives near the boundary. This term improves the accuracy and conditioning 
  of the matrix where the standard finite difference stencil may be less accurate (i.e. at nodes adjacent 
  to the boundary).

# Returns
A sparse matrix `A::SparseMatrixCSC{T,Int}` of size Q_ext × Q_ext, where Q_ext is the number of 
active (extended) nodes. This matrix represents the Hamiltonian with the following components:
  
  1. **Standard Laplacian:** The 5-point finite difference approximation:
         (–∆u)_ij ≈ [4u_ij – u_{i–1,j} – u_{i+1,j} – u_{i,j–1} – u_{i,j+1}]/dx².
     Contributions are added only when both the central node and the neighbor are active (i.e. in Ωₕ).

  2. **Penalty Term (bₕ):** For each edge (horizontal or vertical) that crosses the boundary 
     (detected when the Boolean mask differs between adjacent nodes), the function evaluates φ at 
     both endpoints and computes local coefficients:
         C₁₁ = γ/(2·dx²) · (φ₂²/(φ₁²+φ₂²))
         C₂₂ = γ/(2·dx²) · (φ₁²/(φ₁²+φ₂²))
         C₁₂ = –γ/(2·dx²) · (φ₁ φ₂/(φ₁²+φ₂²))
     (or analogous expressions using dy for vertical edges). These coefficients are then added 
     to both diagonal and off-diagonal entries to weakly enforce u = 0 on the boundary.

  3. **Stabilization Term (jₕ):** For nodes inside Ω that have at least one neighbor outside Ω, 
     a stabilization term is added to penalize the jump in the finite difference approximation of the 
     derivative. For example, in the horizontal direction, if node (i,j) has an adjacent node outside, 
     a term proportional to the discrete second derivative (–u_{i–1,j} + 2u_{i,j} – u_{i+1,j})/dx is 
     added with weight σ/(dx²). A similar procedure is used in the vertical direction.

# Detailed Process
1. **Extended Grid Construction:**  
   The function first computes an extended index mapping via `compute_extended_index` (not shown here) 
   which labels all nodes that are either inside Ω (fem.mask is true) or adjacent (by one grid spacing) 
   to an inside node. This permits the use of standard finite difference stencils across the entire 
   extended domain Ωₕ.

2. **Assembly of the Standard Laplacian:**  
   The code loops over all grid nodes in Ωₕ. For each active node, it applies the 5-point stencil to 
   approximate –∆u. Contributions from neighbors are only included if those neighbors are also active.

3. **Application of the Penalty Term:**  
   For each horizontal edge (between (i,j) and (i+1,j)) and vertical edge (between (i,j) and (i,j+1)) 
   where the mask indicates that one node is inside and the other is outside, φ is evaluated at both 
   endpoints. The coefficients C₁₁, C₂₂, and C₁₂ are computed to create a local quadratic form that 
   penalizes any mismatch in u across the boundary. This term is crucial for enforcing the Dirichlet 
   condition in a weak sense on a non-conforming grid.

4. **Addition of the Stabilization Term:**  
   For nodes that are adjacent to the boundary (i.e. where one horizontal or vertical neighbor is 
   outside Ω), a stabilization term is added. This term uses a discrete second-derivative approximation 
   to penalize irregularities and improve the accuracy and conditioning of the scheme.

5. **Sparse Matrix Assembly:**  
   Finally, all contributions are collected into arrays and assembled into a sparse matrix A, which is 
   returned as the discretized Hamiltonian on the extended grid.
"""
function phiFD_Hamiltonian(fem::FiniteElementMethod,phi::Function,gamma::T,sigma::T) where {T<:Real}
    Nx,Ny=fem.Nx,fem.Ny
    dx,dy=fem.dx,fem.dy
    x_grid,y_grid=fem.x_grid,fem.y_grid
    Ntot=Nx*Ny
    ℏ,m=fem.ℏ,fem.m
    A_const=ℏ^2/(2*m)
    # Build extended index for all nodes in Ω_h:
    ext_idx,Q_ext=compute_extended_index(x_grid,y_grid,fem.mask)
    # Preallocate arrays for sparse matrix assembly
    rows=Int[]; cols=Int[]; vals=T[]
    # Helper to get linear index:
    idx(i,j)=i+(j-1)*Nx
    # --- Standard Laplacian (central 5-point stencil) ---
    for j in 1:Ny
        for i in 1:Nx
            lin=idx(i,j)
            α=ext_idx[lin]
            if α==0
                continue
            end
            diag=zero(T)
            # Distinguish horizontal vs. vertical neighbors. f di != 0 (horizontal neighbor) or dj != 0 (vertical neighbor) and use dx^2 or dy^2 accordingly
            for (di,dj,fac) in [(-1,0,-1.0),(1,0,-1.0),(0,-1,-1.0),(0,1,-1.0)] # all possible non diagonal directions in 2d starting at (i,j), so check all combinations. The diagonal ones are treated separately
                i2,j2=i+di,j+dj
                if i2>=1 && i2<=Nx && j2>=1 && j2<=Ny # we are not ouf bounds
                    lin2=idx(i2,j2)
                    β=ext_idx[lin2] # check if lin2 index is inside the extended boundary
                    if β!=0 # if it is inside the extended boundary or adjacent to an inside cell (so it is part of the extended boundary but not in the interior)
                        if di!=0
                            # horizontal neighbor => use dx
                            push!(rows,α);push!(cols,β);push!(vals,fac/(dx^2))
                            diag-=fac/(dx^2)
                        else
                            # vertical neighbor => use dy
                            push!(rows,α);push!(cols,β);push!(vals,fac/(dy^2))
                            diag-=fac/(dy^2)
                        end
                    end
                end
            end
            push!(rows,α);push!(cols,α);push!(vals,diag)
        end
    end
    # --- Penalization term on edges ---
    # Horizontal edges => dx
    for j in 1:Ny
        for i in 1:(Nx-1)
            lin1=idx(i,j); lin2=idx(i+1,j) # we only check the i -> i + 1
            if ext_idx[lin1]!=0 && ext_idx[lin2]!=0 # if both are part of the extended boundary
                if fem.mask[lin1]!=fem.mask[lin2] # if we are on the boundary (this condition says that one must be true and one false -> one is part of extended boundary and one outside)
                    phi1=phi(x_grid[i],y_grid[j])
                    phi2=phi(x_grid[i+1],y_grid[j])
                    denom=phi1^2+phi2^2
                    if denom==0; continue; end
                    C11=gamma/(2*dx^2)*(phi2^2/denom) # as per paper the penalization function
                    C22=gamma/(2*dx^2)*(phi1^2/denom)  # as per paper the penalization function
                    C12=-gamma/(2*dx^2)*(phi1*phi2/denom)  # as per paper the penalization function
                    α=ext_idx[lin1]; β=ext_idx[lin2]
                    push!(rows,α);push!(cols,α);push!(vals,C11) # like above we manually construct the matrix at the indexes where a penalization procedure was performed
                    push!(rows,β);push!(cols,β);push!(vals,C22)
                    push!(rows,α);push!(cols,β);push!(vals,C12) # symmetric wrt α <-> β 
                    push!(rows,β);push!(cols,α);push!(vals,C12) # symmetric wrt α <-> β 
                end
            end
        end 
    end
    # Vertical edges => dy
    for j in 1:(Ny-1)
        for i in 1:Nx
            lin1=idx(i,j); lin2=idx(i,j+1) # we only check the j -> j + 1
            if ext_idx[lin1]!=0 && ext_idx[lin2]!=0 # if both are part of the extended boundary
                if fem.mask[lin1]!=fem.mask[lin2] # if we are on the boundary (this condition says that one must be true and one false -> one is part of extended boundary and one outside)
                    phi1=phi(x_grid[i],y_grid[j])
                    phi2=phi(x_grid[i],y_grid[j+1])
                    denom=phi1^2+phi2^2
                    if denom==0; continue; end
                    C11=gamma/(2*dy^2)*(phi2^2/denom)  # as per paper the penalization function
                    C22=gamma/(2*dy^2)*(phi1^2/denom)  # as per paper the penalization function
                    C12=-gamma/(2*dy^2)*(phi1*phi2/denom)  # as per paper the penalization function
                    α=ext_idx[lin1]; β=ext_idx[lin2]  # as per paper the penalization function
                    push!(rows,α);push!(cols,α);push!(vals,C11) # check above
                    push!(rows,β);push!(cols,β);push!(vals,C22)
                    push!(rows,α);push!(cols,β);push!(vals,C12) # symmetric like above
                    push!(rows,β);push!(cols,α);push!(vals,C12)
                end
            end
        end
    end
    # --- Stabilization term j_h ---
    # Horizontal stabilization => use dx
    for j in 1:Ny
        for i in 2:(Nx-1)
            lin=idx(i,j)
            if ext_idx[lin]==0 || !fem.mask[lin] # if we are not part of the extended boundary
                continue
            end
            if (!fem.mask[idx(i-1,j)]) || (!fem.mask[idx(i+1,j)]) # if either its right or left neighboor is not part of the interior boundary (more strict than part of extended boundary. otherwise no need to stabilize)
                coeff=sigma/(dx^2)
                α=ext_idx[idx(i-1,j)] # is left in extended
                β=ext_idx[lin] # is i,j in extended
                γ_=ext_idx[idx(i+1,j)] # is right extended
                if α!=0 && β!=0 && γ_!=0 # if all of them are in the extended boundary
                    # stencil modifications, please check the referenced paper for details on how to do this
                    push!(rows,α);push!(cols,α);push!(vals,coeff)
                    push!(rows,β);push!(cols,β);push!(vals,4*coeff)
                    push!(rows,γ_);push!(cols,γ_);push!(vals,coeff)
                    push!(rows,α);push!(cols,β);push!(vals,-2*coeff)
                    push!(rows,β);push!(cols,α);push!(vals,-2*coeff)
                    push!(rows,α);push!(cols,γ_);push!(vals,coeff)
                    push!(rows,γ_);push!(cols,α);push!(vals,coeff)
                    push!(rows,β);push!(cols,γ_);push!(vals,-2*coeff)
                    push!(rows,γ_);push!(cols,β);push!(vals,-2*coeff)
                end
            end
        end
    end
    # Vertical stabilization => use dy (check above for comments)
    for j in 2:(Ny-1)
        for i in 1:Nx
            lin=idx(i,j)
            if ext_idx[lin]==0 || !fem.mask[lin]
                continue
            end
            if (!fem.mask[idx(i,j-1)]) || (!fem.mask[idx(i,j+1)])
                coeff=sigma/(dy^2)
                α=ext_idx[idx(i,j-1)]
                β=ext_idx[lin]
                γ_=ext_idx[idx(i,j+1)]
                if α!=0 && β!=0 && γ_!=0
                    push!(rows,α);push!(cols,α);push!(vals,coeff)
                    push!(rows,β);push!(cols,β);push!(vals,4*coeff)
                    push!(rows,γ_);push!(cols,γ_);push!(vals,coeff)
                    push!(rows,α);push!(cols,β);push!(vals,-2*coeff)
                    push!(rows,β);push!(cols,α);push!(vals,-2*coeff)
                    push!(rows,α);push!(cols,γ_);push!(vals,coeff)
                    push!(rows,γ_);push!(cols,α);push!(vals,coeff)
                    push!(rows,β);push!(cols,γ_);push!(vals,-2*coeff)
                    push!(rows,γ_);push!(cols,β);push!(vals,-2*coeff)
                end
            end
        end
    end
    # Assemble the global sparse matrix
    A=-A_const*sparse(rows,cols,vals,Q_ext,Q_ext)
    return A
end

"""
    reconstruct_wavefunction(evec::Vector{T},ext_idx::Vector{Int},Nx::Int,Ny::Int) where {T<:Real}

This function reconstructs a 2D wavefunction on a Cartesian grid from a 1D eigenvector obtained from a finite-difference discretization of the Schrödinger equation. The function maps the values of evec, which is defined only on the active nodes (i.e., those inside or near the boundary of the domain -> exterior domain problem), back onto the full 2D computational grid.

# Arguments
- `evec::Vector{T}`: a 1D array representing the eigenvector obtained from the finite-difference discretization of the Schrödinger equation.
- `ext_idx::Vector{Int}`: a 1D array representing the extended index mapping.
- `Nx::Int`: the number of grid points in the x-direction.
- `Ny::Int`: the number of grid points in the y-direction.

# Returns
- `wf::Matrix{T}`: a 2D array representing the reconstructed wavefunction on the full 2D computational grid from the external domain.
"""
function reconstruct_wavefunction(evec::Vector{T},ext_idx::Vector{Int},Nx::Int,Ny::Int) where {T<:Real}
    wf=zeros(T,Nx,Ny)
    for j in 1:Ny
        for i in 1:Nx
            lin=i+(j-1)*Nx
            idx_active=ext_idx[lin]
            if idx_active!=0
                wf[i,j]=evec[idx_active]
            end
        end
    end
    return wf
end

"""
    compute_ϕ_fem_eigenmodes(fem::FiniteElementMethod{T}; nev::Int=100, maxiter=100000, tol=1e-8) -> Tuple{Vector{T}, Vector{Matrix{T}}}

Computes the lowest `nev` eigenvalues and wavefunctions of the FEM Hamiltonian.

# Arguments:
- `fem::FiniteElementMethod{T}`: The FEM instance.
- `nev::Int=100`: Number of eigenvalues to compute.
- `maxiter::Int=100000`: Maximum iterations for eigensolver. Can be lower since it should not get past the default limit.
- `tol::T=1e-8`: Tolerance for eigensolver convergence.
- `phi::Function`: Type signature `(x<:Real,y<:Real) -> Bool`. The function that describes the billiard geometry. Needs to be implemented as we need to calculate for specific exterior domain the values there. 

# Returns:
- `evals::Vector{T}`: Computed eigenvalues (square root for energy).
- `wavefunctions::Vector{Matrix{T}}`: Corresponding wavefunctions.
"""
function compute_ϕ_fem_eigenmodes(fem::FiniteElementMethod{T},phi::Function,gamma::T,sigma::T;nev::Int=100,maxiter=100000,tol=1e-8) where {T<:Real}
    H=phiFD_Hamiltonian(fem,phi,gamma,sigma)
    nev=min(nev,fem.Q-1)  # Prevent requesting more eigenvalues than available
    evals,evecs=eigs(H,nev=nev,which=:SM,tol=tol,maxiter=maxiter)
    wavefunctions=[zeros(T,fem.Nx,fem.Ny) for _ in 1:nev]
    Threads.@threads for i in 1:fem.Nx
        for j in 1:fem.Ny
            α=fem.interior_idx[i+(j-1)*fem.Nx] 
            if α>0  # Ensure only interior points are accessed
                for n in 1:nev
                    wavefunctions[n][i,j]=evecs[α,n]
                end
            end
        end
    end
    return evals,wavefunctions
end