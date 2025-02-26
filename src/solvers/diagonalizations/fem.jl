using LinearAlgebra, SparseArrays, Arpack

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

function FiniteElementMethod(billiard::Bi,Nx::Int,Ny::Int,b::T;ℏ::T=1.0,m::T=1.0,fundamental=false,k_max=100.0) where {T<:Real,Bi<:AbsBilliard}
    boundary=fundamental ? billiard.fundamental_boundary : billiard.full_boundary
    L=billiard.length
    typ=eltype(L)
    xlim::Tuple{typ,typ},ylim::Tuple{typ,typ}=boundary_limits(boundary;grd=max(1000,round(Int,k_max*L*b/(2*pi))))
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

function FEM_Hamiltonian(fem::FiniteElementMethod{T}) where {T<:Real}
    ℏ,m=fem.ℏ,fem.m;A_const=ℏ^2/(2m)
    dx²,dy²=fem.dx^2,fem.dy^2;Nx,Ny=fem.Nx,fem.Ny
    rows,cols,vals=Int[],Int[],Float64[]

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

function FEM_Hamiltonian(fem::FiniteElementMethod{T},V::Matrix{T}) where {T<:Real}
    ℏ,m=fem.ℏ,fem.m;A_const=ℏ^2/(2m)
    dx²,dy²=fem.dx^2,fem.dy^2;Nx,Ny=fem.Nx,fem.Ny
    rows,cols,vals=Int[],Int[],Float64[]

    for i in 2:(Nx-1),j in 2:(Ny-1)
        α=fem.interior_idx[i + (j-1) * Nx];if α==0;continue;end
        for (ni,nj,factor) in [(i+1,j,-A_const/dx²),(i-1,j,-A_const/dx²),
                               (i,j+1,-A_const/dy²),(i,j-1,-A_const/dy²)]
            β=fem.interior_idx[ni + (nj-1) * Nx]
            if β!=0
                push!(rows,α);push!(cols,β);push!(vals,factor)
            end
        end
        push!(rows,α);push!(cols,α);push!(vals,A_const*(2/dx²+2/dy²) + V[i,j])
    end
    return sparse(rows,cols,vals,fem.Q,fem.Q)
end

function compute_fem_eigenmodes(fem::FiniteElementMethod{T};nev::Int=100,maxiter=100000) where {T<:Real}
    H=FEM_Hamiltonian(fem)
    nev=min(nev,fem.Q-1)  # Prevent requesting more eigenvalues than available
    evals,evecs=eigs(H,nev=nev,which=:SR,tol=1e-8,maxiter=maxiter)
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