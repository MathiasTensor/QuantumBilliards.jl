using LinearAlgebra,SparseArrays,Arpack

"""
    struct FiniteElementMethod{T<:Real,Bi<:AbsBilliard}

A Finite Element Method (FEM) solver for quantum billiards.

# Fields:
- `billiard::Bi`: The billiard geometry.
- `ℏ::T`: Reduced Planck's constant.
- `m::T`: Particle mass.
- `Nx::Int, Ny::Int`: Number of grid points in x and y directions.
- `Lx::T, Ly::T`: Domain lengths in x and y directions.
- `dx::T, dy::T`: Grid spacing in x and y directions.
- `x_grid::Vector{T}, y_grid::Vector{T}`: Discretized coordinate grids.
- `mask::Vector{Bool}`: Mask indicating valid points inside the billiard.
- `interior_idx::Vector{Int}`: Index map of interior nodes.
- `Q::Int`: Number of interior nodes.
"""
struct FiniteElementMethod{T<:Real,Bi<:AbsBilliard}
    billiard::Bi
    ℏ::T
    m::T
    Nx::Int
    Ny::Int
    Lx::T
    Ly::T
    dx::T
    dy::T
    x_grid::Vector{T}
    y_grid::Vector{T}
    mask::Vector{Bool}
    interior_idx::Vector{Int}
    Q::Int
end

"""
    compute_interior_index(mask::Vector{Bool}) -> Tuple{Vector{Int}, Int}

Computes an index mapping of interior points inside the billiard. This follows the logic of reference: 
Classical and Quantum Chaos in the Diamond Shaped Billiard, R. Salazar (1), G. Téllez (1), D. Jaramillo (1), D. L. González (2) ((1) Departamento de Física, Universidad de los Andes (2) Department of Physics, University of Maryland)
https://doi.org/10.48550/arXiv.1205.4990

# Arguments:
- `mask::Vector{Bool}`: Boolean vector indicating interior points.

# Returns:
- `interior_idx::Vector{Int}`: Indexed mapping of interior points.
- `Q::Int`: Total number of interior points.
"""
function compute_interior_index(mask::Vector{Bool})
    interior_idx=zeros(Int,length(mask));count=0
    for i in 1:length(mask)
        if mask[i]
            count+=1
            interior_idx[i]=count
        end
    end
    return interior_idx,count
end

"""
    compute_boundary(interior_idx::Vector{Ti},Nx::Ti,Ny::Ti) -> Matrix{Bool} where {Ti<:Integer}

Computes the boundary of a domain given the `interior_idx` matrix, which is derived from `compute_interior_index(mask)`. 

A boundary cell is any interior cell (value > 0) that has at least one neighbor that is an exterior cell (value = 0).

# Arguments
- `interior_idx::Vector{Ti}`: A **flattened** 1D vector of size `Nx * Ny`, where each interior node is assigned a unique index (> 0) and exterior nodes are marked as `0`.
- `Nx::Ti`: Number of grid points in the x-direction.
- `Ny::Ti`: Number of grid points in the y-direction.

# Returns
- `boundary::Matrix{Bool}`: A `Nx × Ny` Boolean matrix where `true` marks boundary points and `false` marks non-boundary points.
"""
function compute_boundary(interior_idx::Vector{Ti},Nx::Ti,Ny::Ti) where {Ti<:Integer}
    boundary=Matrix{Bool}(undef,Nx,Ny)
    boundary.=false
    interior=reshape(interior_idx,Nx,Ny)
    for j in 1:Ny, i in 1:Nx
        if interior[i,j] > 0  # Only check if it's an interior point
            # Check if any neighbor is outside (interior_idx = 0)
            if (i>1 && interior[i-1,j]==0) ||  # Left
               (i<Nx && interior[i+1,j]==0) ||  # Right
               (j>1 && interior[i,j-1]==0) ||  # Below
               (j<Ny && interior[i,j+1]==0)     # Above
                boundary[i,j]=true
            end
        end
    end
    return boundary::Matrix{Bool}
end

"""
    FiniteElementMethod(billiard::Bi, Nx::Int, Ny::Int, b::T; ℏ::T=1.0, m::T=1.0, fundamental=false, k_max=100.0) where {T<:Real, Bi<:AbsBilliard}

Initializes a FEM solver for a quantum billiard.

# Arguments:
- `billiard::Bi`: The billiard geometry.
- `Nx::Int, Ny::Int`: Number of grid points in x and y directions.
- `ℏ::T=1.0`: Reduced Planck's constant.
- `m::T=1.0`: Particle mass.
- `fundamental::Bool=false`: Use fundamental domain (if applicable).
- `k_max::T=100.0`: Wavevector-based scaling parameter.
- `offset_x_symmetric::T=0.1`: Symmetric +/- offset in x direction. This is crucial as in certain methods like φ-FDM we need cells outside the formal interior. Do not set to 0.0
- `offset_y_symmetric::T=0.1`: Symmetric +/- offset in y direction. This is crucial as in certain methods like φ-FDM we need cells outside the formal interior. Do not set to 0.0

# Returns:
- `FiniteElementMethod` instance with computed grid and index mapping.
"""
function FiniteElementMethod(billiard::Bi,Nx::Int,Ny::Int;ℏ::T=1.0,m::T=1.0,fundamental=false,k_max=100.0,offset_x_symmetric=0.1,offset_y_symmetric=0.1) where {T<:Real,Bi<:AbsBilliard}
    boundary=fundamental ? billiard.fundamental_boundary : billiard.full_boundary
    L=billiard.length
    typ=eltype(L)
    xlim::Tuple{typ,typ},ylim::Tuple{typ,typ}=boundary_limits(boundary;grd=max(1000,round(Int,k_max*L*5.0/(2*pi)))) # b=5.0, not needed here
    xlim=(xlim[1]-offset_x_symmetric,xlim[2]+offset_x_symmetric)
    ylim=(ylim[1]-offset_y_symmetric,ylim[2]+offset_y_symmetric)
    Lx,Ly=xlim[2]-xlim[1],ylim[2]-ylim[1] # domain lengths in x and y
    dx,dy=Lx/(Nx-1),Ly/(Ny-1)
    nx,ny=Nx,Ny
    x_grid,y_grid=collect(typ,range(xlim..., nx)),collect(typ,range(ylim..., ny))
    pts=collect(SVector(x,y) for y in y_grid for x in x_grid)
    sz=length(pts)
    mask=points_in_billiard_polygon(pts,billiard,round(Int,sqrt(sz));fundamental_domain=fundamental)
    interior_idx,Q=compute_interior_index(mask)
    return FiniteElementMethod(billiard,ℏ,m,Nx,Ny,Lx,Ly,dx,dy,x_grid,y_grid,mask,interior_idx,Q)
end

# NEEDS PRECONDITIONING
"""
    FEM_Hamiltonian(fem::FiniteElementMethod{T}) -> SparseMatrixCSC{T,Int}

Constructs the free FEM Hamiltonian matrix using the 2D Laplacian. If optionally V::Matrix is supplied then it incorporates the potential energy into the Hamiltonian matrix. 
The Hamiltonian is symmetric either way since the potential only comes as digonal terms in the final matrix.

# Arguments:
- `fem::FiniteElementMethod{T}`: The FEM instance.

# Returns:
- `H::SparseMatrixCSC{T,Int}`: The Hamiltonian matrix.
"""
function FEM_Hamiltonian(fem::FiniteElementMethod{T}) where {T<:Real}
    ℏ,m=fem.ℏ,fem.m;A_const=ℏ^2/(2m)
    dx²,dy²=fem.dx^2,fem.dy^2;Nx,Ny=fem.Nx,fem.Ny
    rows,cols,vals=Int[],Int[],T[]

    for i in 2:(Nx-1),j in 2:(Ny-1)
        α=fem.interior_idx[i + (j-1) * Nx];if α==0;continue;end
        for (ni,nj,factor) in [(i+1,j,-A_const/dx²),(i-1,j,-A_const/dx²),
                               (i,j+1,-A_const/dy²),(i,j-1,-A_const/dy²)]
            β=fem.interior_idx[ni + (nj-1) * Nx]
            if β!=0
                push!(rows,α);push!(cols,β);push!(vals,factor)
            end
        end
        push!(rows,α);push!(cols,α);push!(vals,A_const*(2/dx²+2/dy²))
    end
    return sparse(rows,cols,vals,fem.Q,fem.Q)
end

# NEEDS PRECONDITIONING - NOT CURRENTLY FOR USE
function FEM_Hamiltonian(fem::FiniteElementMethod{T},V::Matrix{T}) where {T<:Real}
    ℏ,m=fem.ℏ,fem.m;A_const=ℏ^2/(2m)
    dx²,dy²=fem.dx^2,fem.dy^2;Nx,Ny=fem.Nx,fem.Ny
    rows,cols,vals=Int[],Int[],T[]

    for i in 2:(Nx-1),j in 2:(Ny-1)
        α=fem.interior_idx[i + (j-1) * Nx];if α==0;continue;end
        for (ni,nj,factor) in [(i+1,j,-A_const/dx²),(i-1,j,-A_const/dx²),
                               (i,j+1,-A_const/dy²),(i,j-1,-A_const/dy²)]
            β=fem.interior_idx[ni + (nj-1) * Nx]
            if β!=0
                push!(rows,α);push!(cols,β);push!(vals,factor)
            end
        end
        push!(rows,α);push!(cols,α);push!(vals,A_const*(2/dx²+2/dy²)+V[i,j])
    end
    return sparse(rows,cols,vals,fem.Q,fem.Q)
end

"""
    compute_fem_eigenmodes(fem::FiniteElementMethod{T}; nev::Int=100, maxiter=100000, tol=1e-8) -> Tuple{Vector{T}, Vector{Matrix{T}}}

Computes the lowest `nev` eigenvalues and wavefunctions of the FEM Hamiltonian. It uses the standard 5 point stencil

# Arguments:
- `fem::FiniteElementMethod{T}`: The FEM instance.
- `nev::Int=100`: Number of eigenvalues to compute.
- `maxiter::Int=100000`: Maximum iterations for eigensolver. Can be lower since it should not get past the default limit.
- `tol::T=1e-8`: Tolerance for eigensolver convergence.

# Returns:
- `evals::Vector{T}`: Computed eigenvalues (square root for energy).
- `wavefunctions::Vector{Matrix{T}}`: Corresponding wavefunctions.
"""
function compute_fem_eigenmodes(fem::FiniteElementMethod{T};nev::Int=100,maxiter=100000,tol=1e-8) where {T<:Real}
    H=FEM_Hamiltonian(fem)
    nev=min(nev,fem.Q-1)  # Prevent requesting more eigenvalues than available
    evals,evecs=eigs(Symmetric(H),nev=nev,which=:SR,tol=tol,maxiter=maxiter)
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

"""
    compute_boundary_tension(ψ::Matrix{T}, boundary_mask::Matrix{Bool}) -> T where {T<:Real}

Computes the boundary tension of a wavefunction `ψ` by summing its squared magnitude only at the boundary points. 
It's purpose is to gauge the "badness" of the resulting wavefunction. Ideally should be 0.0.

# Arguments
- `ψ::Matrix{T}`: A Nx × Ny matrix representing the wavefunction over the computational domain -> from `compute_fem_eigenmodes`
- `boundary_mask::Matrix{Bool}`: `Matrix` of the same size as `ψ`, where `true` marks boundary points and `false` marks interior/exterior points.

# Returns
- `T`: boundary tension, computed as `∑ |ψ(i, j)|²` over the boundary.
"""
function compute_boundary_tension(ψ::Matrix{T},boundary_mask::Matrix{Bool}) where {T<:Real}
    ψ.=ψ./norm(ψ)
    return sum(abs2.(ψ)[boundary_mask])  # Sum |ψ|² only at boundary points
end