using Bessels, LinearAlgebra, ProgressMeter

#include("../billiards/boundarypoints.jl")
#include("../states/wavefunctions.jl")

#=
Contains functions for computing the 2-point and 4-point out-of-time-order correlators (OTOCs) in billiard systems. It also supports wavepacket projections onto the eigenbasis of the billiard and their OTOC computations.
Based on the paper by Hashimoto et. al.: Out-of-time-order correlators in quantum mechanics, link: https://arxiv.org/abs/1703.09435
=#


##############################
#### GENERAL 2-POINT OTOC ####
##############################


"""
    Xmat(Psis::Vector{<:AbstractMatrix{T}},xgrid::Vector{T},ygrid::Vector{T};memory_efficient::Bool=false,n_blas_threads::Int=ceil(Int,Threads.nthreads()/2),direction::Symbol=:x) where {T<:Real}

Compute the “X” overlap matrix for a set of discretized wavefunctions.

This function builds the Hermitian matrix

Xₘₙ = ∑_{i,j} ψₘ[i,j] * xgrid[i] * Δx * Δy * ψₙ[i,j]

either by
1.	a memory-efficient diagonal-block BLAS call (when memory_efficient=true), or
2.	a BLAS-GEMM accelerated approach (when memory_efficient=false, via Xmat_gemm).

# Arguments
- `Psis::Psis::Vector{<:AbstractMatrix{T}}`: length-N Vector of real matrices, each of size (nx×ny), giving ψₘ[i,j].
- `xgrid::Vector{T}, ygrid::Vector{T}`: 1D coordinate vectors of lengths nx and ny, respectively. Must be uniformly spaced.
- `memory_efficient::Bool=false`: If true, uses X_block_diag to trade extra compute for lower peak memory.
- `n_blas_threads::Int=ceil(Int,Threads.nthreads()/2)`: number of BLAS threads to use when memory_efficient=false.
- `direction::Symbol=:x`: Direction of the overlap matrix computation, possibility is :x or :y.

# Returns
- `X::Symmetric{T}`: N×N Hermitian overlap matrix.
"""
function Xmat(Psis::Vector{<:AbstractMatrix{T}},xgrid::Vector{T},ygrid::Vector{T};memory_efficient::Bool=false,n_blas_threads::Int=ceil(Int,Threads.nthreads()/2),direction::Symbol=:x) where {T<:Real}
    if memory_efficient
        return X_block_diag(Psis,xgrid,ygrid,direction=direction)
    else
        return Xmat_gemm(Psis,xgrid,ygrid,n_blas_threads=n_blas_threads,direction=direction)
    end
end

"""
    X_standard(Psis::Vector{<:AbstractMatrix{T}},xgrid::Vector{T},ygrid::Vector{T};multithreaded::Bool=true,direction::Symbol=:x) where {T<:Real}

Compute the “X” overlap matrix by direct weighted summation over all grid points. One should not use this function in large scale calculations but just to check their results since this version is a brute-force approach.

This function builds the Hermitian matrix

Xₘₙ = ∑₍ᵢ,ⱼ₎ ψₘ[i,j] * xgrid[i] * Δx * Δy * ψₙ[i,j]

using nested loops and optional multithreading.

# Arguments
- `Psis::Vector{<:AbstractMatrix{T}}`: length-N Vector of real matrices, each of size (nx×ny), giving ψₘ[i,j].
- `xgrid::Vector{T}`, `ygrid::Vector{T}`: 1D coordinate vectors of lengths nx and ny, respectively. Must be uniformly spaced.
- `multithreaded::Bool=true`: If true, parallelize the outer loops over m,n using `Threads.@threads`.
- `direction::Symbol=:x`: Direction of the overlap matrix computation, possibility is :x or :y.

# Returns
- `X::Symmetric{T}`: N×N Hermitian overlap matrix.
"""
function X_standard(Psis::Vector{<:AbstractMatrix{T}},xgrid::Vector{T},ygrid::Vector{T};multithreaded::Bool=true,direction::Symbol=:x) where {T<:Real}
    N=length(Psis)
    @assert N>0 "Need at least one wavefunction"
    nx,ny=length(xgrid),length(ygrid)
    @assert size(Psis[1])==(nx,ny) "Each ψ must be (nx×ny) matching xgrid, ygrid"
    Δx=xgrid[2]-xgrid[1]
    Δy=ygrid[2]-ygrid[1]
    area=Δx*Δy
    # Pre-compute the weight matrix:
    if direction==:x
        W=xgrid.*area # W[i,j]=xgrid[j]*Δx*Δy so that sum( conj(ψm) .* (W .* ψn) ) ≈ X_{mn}
    elseif direction==:y
        W=ygrid.*area # W[i,j]=ygrid[i]*Δx*Δy so that sum( conj(ψm) .* (W .* ψn) ) ≈ X_{mn}
    else
        throw(ArgumentError("direction must be either :x or :y"))
    end
    Xmat=Matrix{Complex{T}}(undef,N,N)
    @fastmath begin
        @use_threads multithreading=multithreaded for m in 1:N
            ψm=Psis[m]
            ψm_conj=conj.(ψm)
            @inbounds for n in m:N
                ψn=Psis[n]
                # sum over i=1:nx, j=1:ny of conj(ψm[i,j]) * xgrid[i]*ΔxΔy * ψn[i,j]
                val=sum(ψm_conj.*(W.*ψn))
                Xmat[m,n]=val
                Xmat[n,m]=val
            end
        end
    end
    return Symmetric(Xmat)
end

"""
   X_block_diag(Psis::Vector{<:AbstractMatrix{T}},xgrid::Vector{T},ygrid::Vector{T};direction::Symbol=:x) where {T<:Real}

Compute the “X” overlap matrix via one diagonal-block BLAS call. This one has smaller overhead then `Xmat_gemm` and is more memory efficient than `X_standard`, but slightly less efficient than `Xmat_gemm` for large N.

This routine constructs

P = hcat(vec.(Psis)…) # (nx*ny)×N
Wbig = vec(repeat(xgrid, outer=ny)) * (Δx * Δy)
X = P' * Diagonal(Wbig) * P

in a single efficient BLAS operation.

# Arguments
- `Psis::Vector{<:AbstractMatrix{T}}`: length-N Vector of real matrices, each of size (nx×ny), giving ψₘ[i,j].
- `xgrid::Vector{T}`, `ygrid::Vector{T}`: 1D coordinate vectors of lengths nx and ny, respectively. Must be uniformly spaced.
- `direction::Symbol=:x`: Direction of the overlap matrix computation, possibility is :x or :y.

# Returns
- `X::Symmetric{T}`: N×N real overlap matrix.
"""
function X_block_diag(Psis::Vector{<:AbstractMatrix{T}},xgrid::Vector{T},ygrid::Vector{T};direction::Symbol=:x) where {T<:Real}
    dx,dy=xgrid[2]-xgrid[1], ygrid[2]-ygrid[1]
    area=dx*dy
    nx,ny=size(Psis[1])
    if direction==:x
        Wbig=repeat(xgrid,outer=ny).*area  # length‐M weight vector in column-major order
    elseif direction==:y
        Wbig=repeat(ygrid,inner=nx).*area  # length‐M weight vector in column-major order
    else
        throw(ArgumentError("direction must be either :x or :y"))
    end
    P=hcat(vec.(Psis)...) # stack all flattened ψ’s into an M×N matrix
    # compute X = P' * diag(Wbig) * P in one go, diagonal-block multiply + GEMM
    return Symmetric(P'*(Diagonal(Wbig)*P))
end

"""
     Xmat_gemm(Psis::Vector{<:AbstractMatrix{T}},xgrid::Vector{T},ygrid::Vector{T};n_blas_threads::Int=ceil(Int,Threads.nthreads()/2),direction::Symbol=:x) where {T<:Real}

Compute the “X” overlap matrix using a BLAS-GEMM accelerated approach. It is faster but has a higher RAM cast since it needs to allocate two large matrices P and Y.
Builds two matrices P and Y (both M×N), where Y has each row k scaled by its weight,
and then performs one GEMM X = P’ * Y.

# Arguments
- `Psis::Vector{<:AbstractMatrix{T}}`: length-N Vector of real matrices, each of size (nx×ny), giving ψₘ[i,j].
- `xgrid::Vector{T}`, `ygrid::Vector{T}`: 1D coordinate vectors of lengths nx and ny, respectively. Must be uniformly spaced.
- `n_blas_threads::Int=ceil(Int,Threads.nthreads()/2)`: number of BLAS threads to use.
- `direction::Symbol=:x`: Direction of the overlap matrix computation, possibility is :x or :y.

# Returns
- `X::Matrix{T}`: N×N real overlap matrix.
"""
function Xmat_gemm(Psis::Vector{<:AbstractMatrix{T}},xgrid::Vector{T},ygrid::Vector{T};n_blas_threads::Int=ceil(Int,Threads.nthreads()/2),direction::Symbol=:x) where {T<:Real}
    BLAS.set_num_threads(n_blas_threads) 
    nx,ny=length(xgrid),length(ygrid)
    N=length(Psis)
    M=nx*ny
    # auxillary matrices, raised RAM cost wrt X_block_diag but faster in the end due to single optimized GEMM call
    P=Matrix{T}(undef,M,N)# for vec(Psis)
    Wbig=Vector{T}(undef,M)# for weights
    X=Matrix{T}(undef,N,N)# for result
    # build weight‐vector exactly aligned with vec(P)
    dx,dy=xgrid[2]-xgrid[1],ygrid[2]-ygrid[1]
    @fastmath begin 
        # make an (nx×ny) weight‐matrix whose (i,j)=xgrid[i], then flatten
        if direction==:x
            Wmat=repeat(xgrid,1,ny) # size (nx×ny),  Wmat[i,j] == xgrid[i]
        elseif direction==:y
            Wmat=repeat(ygrid',nx,1) # size (nx×ny), Wmat[i,j] == ygrid[j]
        else
            throw(ArgumentError("direction must be either :x or :y"))
        end
        Wbig.=vec(Wmat).*(dx*dy)      # length-M
        @inbounds Threads.@threads for j in 1:N
            P[:,j] .=vec(Psis[j]) # fill the un-weighted P (M×N) once
        end
        # make a *separate* copy Y of P and weight *only* its rows
        Y=copy(P)
        Threads.@threads for k in 1:M
            BLAS.scal!(Wbig[k],view(Y,k,:))
        end
        # one call to GEMM:  X = Pᵀ * Y
        BLAS.gemm!('T','N',one(T),P,Y,zero(T),X)
    end
    return Symmetric(X)
end

"""
    Bmat(X::AbstractMatrix{T},E::Vector{T},t::Real) where {T<:Real}

Cosntructs the b_{nm} matrix in the OTOC paper. With this one can construct the microcanonical OTOC. The b_{nm} matrix is defined as:

b_{nm} = 0.5 * (∑_k (X[n,k]*exp(im*(E_n - E_k)*t)) * (X[k,m]*(E_k - E_m))  -  ∑_k (X[n,k]*(E_n - E_k)) * (X[k,m]*exp(im*(E_k - E_m)*t)))

where we can write this as an efficient matrix multiplication with LEVEL 3 BLAS implementation of the form:

B_{nm} = 0.5 * (T1 - T2)

where T1 and T2 are defined as:

T1[n,m] = ∑_k (X[n,k]*exp(im*(E_n - E_k)*t)) * (X[k,m]*(E_k - E_m))

T2[n,m] = ∑_k (X[n,k]*(E_n - E_k)) * (X[k,m]*exp(im*(E_k - E_m)*t))

# Arguments
- `X::AbstractMatrix{T}`: The matrix X representing the scalar products between the wavefunctions and the x or y grid.
- `E::Vector{T}`: Vector of eigenvalues (energies) corresponding to the wavefunctions.
- `t::T`: Time value for the OTOC computation.

# Returns
- `Hermitian{Complex{T}}`: The computed b_{nm} matrix at time t.
"""
function Bmat(X::AbstractMatrix{T},E::Vector{T},t::T) where {T<:Real}
    N=length(E)
    En=reshape(E,N,1) # column vector
    Em=reshape(E,1,N) # row vector
    # energy‐difference matrices
    ΔE_nk=En.-Em # (n,k): E_n – E_k
    ΔE_km=En.-Em # (k,m) interpreted in second GEMM
    Φ_nk=@. cis(ΔE_nk*t)
    Φ_km=@. cis(ΔE_km*t)
    # two GEMM calls
    # T1[n,m] = ∑_k (X[n,k]*Φ_nk[n,k]) * (X[k,m]*ΔE_km[k,m])
    T1=(X.*Φ_nk)*(X.*ΔE_km) # for each inner bracket elementwise product, then finally matrix product
    # T2[n,m] = ∑_k (X[n,k]*ΔE_nk[n,k]) * (X[k,m]*Φ_km[k,m])
    T2=(X.*ΔE_nk)*(X.*Φ_km) # for each inner bracket elementwise product, then finally matrix product
    return Hermitian(0.5*(T1.-T2))
end

"""
    C(Bmat::AbstractMatrix{T}) where {T<:Real}

Compute the microcanonical out-of-time-order correlator vector for all eigenstates at the time `t` that has the corresponding `Bmat`.
Given the commutator matrix Bmat with elements bₙₘ(t) = –i⟨n|[x(t), p(0)]|m⟩, this returns the length-N vector

cₙ(t) = ∑ₘ |Bmat[n, m]|².

# Arguments
- `Bmat::AbstractMatrix{Complex{T}}`: An N×N matrix of real or complex values, where entry (n,m) is bₙₘ(t).

# Returns
- `c::Vector{T}`: length-N real vector whose nth entry is sum(abs2, Bmat[n, :]) at single time `t`.
"""
function C(Bmat::AbstractMatrix{Complex{T}}) where {T<:Real}
    return sum(abs2,Bmat,dims=2)  # returns an N×1 vector c[n] = ∑_m |B[n,m]|^2
end


"""
    C(Bmat::AbstractMatrix{T},n::Int) where {T<:Real}

Compute the microcanonical out-of-time-order correlator for a single eigenstate n.
Returns the scalar

cₙ(t) = ∑ₘ |Bmat[n, m]|²

for the specified row n of the commutator matrix.

# Arguments
- `Bmat::AbstractMatrix{T}`: An N×N matrix of real or complex values, where entry (n,m) is bₙₘ(t) at a single time `t`.
- `n::Int`: Index of the eigenstate (1 ≤ n ≤ size(Bmat,1)).

# Returns
- `cₙ::T`: real scalar equal to sum(abs2, Bmat[n, :]) at time `t`.
"""
function C(Bmat::AbstractMatrix{Complex{T}},n::Int) where {T<:Real}
    return sum(abs2,view(Bmat[n,:]))
end

"""
    C_evolution(Psis::Vector{<:AbstractMatrix{T}},Es::Vector{T},xgrid::Vector{T},ygrid::Vector{T},ts::Vector{T};memory_efficient::Bool=false,n_blas_threads::Int=ceil(Int,Threads.nthreads()/2),direction::Symbol=:x) where {T<:Real}

Compute the microcanonical out-of-time-order correlator cₙ(t) for each eigenstate over a sequence of times.
Construct the overlap matrix X (via Xmat), then for each time t in ts, the commutator matrix B = Bmat(X, Es, t), and then the vector
cₙ(t) = ∑ₘ |B[n,m]|².  Returns all cₙ values as the columns of a matrix.

# Arguments
- `Psis::Vector{<:AbstractMatrix{T}}`: length-N vector of discretized wavefunction matrices (size nx×ny).
- `Es::Vector{T}`: length-N vector of eigenvalues Eₙ.
- `xgrid::Vector{T}, ygrid::Vector{T}`: 1D coordinate arrays of lengths nx and ny (uniformly spaced).
- `ts::Vector{T}`: array of time points at which to evaluate cₙ(t).
- `memory_efficient::Bool=false`: if true, uses the low-memory diagonal-block method for Xmat.
- `n_blas_threads::Int`: number of BLAS threads to use when memory_efficient=false.
- `direction::Symbol=:x`: choose :x or :y direction for the overlap matrix.

# Returns
- `Cmat::Matrix{T} : N×length(ts) real array whose (n,i) entry is cₙ(ts[i]) = ∑ₘ |Bn,m|².
"""
function C_evolution(Psis::Vector{<:AbstractMatrix{T}},Es::Vector{T},xgrid::Vector{T},ygrid::Vector{T},ts::Vector{T};memory_efficient::Bool=false,n_blas_threads::Int=ceil(Int,Threads.nthreads()/2),direction::Symbol=:x) where {T<:Real}
    Cmat=Matrix{T}(undef,length(Psis),length(ts)) # N×T matrix
    X=Xmat(Psis,xgrid,ygrid;direction=direction,memory_efficient=memory_efficient,n_blas_threads=n_blas_threads)
    for i in eachindex(ts) 
        B=Bmat(X,Es,ts[i])
        Cs=C(B)
        Cmat[:,i]=vec(Cs) # store the cₙ(t) vector as the i-th column
    end
    return Cmat
end


#########################################
#### WAVAPACKET PROJECTIONS AND OTOC ####
#########################################

"""
    α_n(ks::Vector{T},vec_us::Vector{Vector{T}},vec_bdPoints::Vector{BoundaryPoints{T}},billiard::Bi,packet::Wavepacket{T};b::Float64=5.0,fundamental_domain::Bool=true) where {Bi<:AbsBilliard, T<:Real}

Compute the projection of a Gaussian‐localized wavepacket onto the billiard eigenbasis.

This routine calls `gaussian_coefficients(...)` to build:
1. `Psi2ds` — a Vector of 2D Husimi‐like discretizations of the wavepacket on the (nx×ny) grid,
2. `α_ns` — the complex amplitudes ⍺ₙ = ⟨n|ψ⟩ in the eigenbasis,
3. `x_grid`,`y_grid` — the 1D coordinate arrays of lengths nx and ny,
4. `pts_mask` — a boolean mask of valid interior points (true inside the fundamental domain),
5. `dx`,`dy` — the grid spacings.

# Arguments
- `ks::Vector{T}`: Vector of eigen-wavenumbers kₙ (length N).
- `vec_us::Vector{Vector{T}}`: Vector of boundary point parameterizations for each eigenstate.
- `vec_bdPoints::Vector{BoundaryPoints{T}}`: Vector of boundary geometry objects for each state.
- `billiard::Bi`: A subtype of `AbsBilliard` that defines the billiard domain.
- `packet::Wavepacket{T}`: Parameters of the initial Gaussian wavepacket (center, width, momentum).
- `b::Float64=5.0`: Optional Gaussian width parameter.
- `fundamental_domain::Bool=true`: If true, restrict the computation to the fundamental billiard domain.

# Returns
- `Psi2ds::Vector{Matrix{T}}`: Length-N Vector of nx×ny real matrices representing the 2D wavepacket values.
- `α_ns::Vector{Complex{T}}`: Length-N complex amplitudes of the packet in the eigenbasis.
- `x_grid::Vector{T}`: 1D array of x-coordinates (length nx).
- `y_grid::Vector{T}`: 1D array of y-coordinates (length ny).
- `pts_mask::Matrix{Bool}`: (nx×ny) boolean mask marking interior points of the fundamental domain.
- `dx::T`,`dy::T`: Scalar grid spacings in x and y.
"""
function α_n(ks::Vector{T},vec_us::Vector{Vector{T}},vec_bdPoints::Vector{BoundaryPoints{T}},billiard::Bi,packet::Wavepacket{T};b::Float64=5.0,fundamental_domain=true) where {Bi<:AbsBilliard,T<:Real}
    Psi2ds,α_ns,x_grid,y_grid,pts_mask,dx,dy=gaussian_coefficients(ks,vec_us,vec_bdPoints,billiard,packet;b=b,fundamental_domain=fundamental_domain)
    return Psi2ds,α_ns,x_grid,y_grid,pts_mask,dx,dy
end


"""
    b_wavepacket(α_ns::Vector{Complex{T}},Psi2ds::Vector{<:AbstractMatrix{T}},xgrid::Vector{T},ygrid::Vector{T},ks::Vector{T},t::T;memory_efficient::Bool=true,direction::Symbol=:x) where {T<:Real}

Compute the 2-point OTOC b(t) = α† B(t) α at a single time t.

Steps:
1. Build the overlap matrix X of size N×N by summing over the 2D grids:
   - If `direction == :x`, weight by xgrid[i] * Δx * Δy.
   - If `direction == :y`, weight by ygrid[j] * Δx * Δy.
   Two options for X:
   - `X_block_diag` (lower memory, uses a block-diagonal BLAS approach).
   - `Xmat_gemm` (uses a single GEMM call, faster for large N).
2. Compute eigenenergies `Es = ks .^ 2`.
3. Call `Bmat(X, Es, t)` to form Bₙₘ(t) = −i ⟨n|[x(t), p(0)]|m⟩.
4. Return the scalar b(t) = α† * B * α.

# Arguments
- `α_ns::Vector{Complex{T}}`: Length-N complex amplitudes αₙ = ⟨n|ψ⟩.
- `Psi2ds::Vector{<:AbstractMatrix{T}}`: Length-N vector of ψₙ[i,j] on an nx×ny grid.
- `xgrid::Vector{T}`, `ygrid::Vector{T}`: 1D coordinate arrays (lengths nx and ny).
- `ks::Vector{T}`: Length-N vector of wavenumbers (energies = ks.^2).
- `t::T`: Time at which to evaluate b(t).
- `memory_efficient::Bool=true`: If true, use `X_block_diag`; otherwise `Xmat_gemm`.
- `direction::Symbol=:x`: Choose `:x` (weight by x) or `:y` (weight by y).

# Returns
- `b_val::Complex{T}`: Complex scalar b(t) = α† * B(t) * α.
"""
function b_wavepacket(α_ns::Vector{Complex{T}},Psi2ds::Vector{<:AbstractMatrix{T}},xgrid::Vector{T},ygrid::Vector{T},ks::Vector{T},t::T;memory_efficient::Bool=true,direction::Symbol=:x) where {T<:Real}
    X=ifelse(memory_efficient,X_block_diag(Psi2ds,xgrid,ygrid,direction=direction),Xmat_gemm(Psi2ds,xgrid,ygrid,direction=direction))
    Es=ks.^2 # E=k^2
    B=Bmat(X,Es,t)
    return (conj(α_ns)'*B)*α_ns # scalar b(t) = α_n^† B α_n
end

"""
    b_wavepacket(α_ns::Vector{Complex{T}},Psi2ds::Vector{<:AbstractMatrix{T}},xgrid::Vector{T},ygrid::Vector{T},ks::Vector{T},ts::Vector{T};memory_efficient::Bool=true,direction::Symbol=:x) where {T<:Real}

Compute the 2-point OTOC b(t) = α† B(t) α for multiple time points in parallel.

Internally, X is constructed only once, then for each time t in ts:
- Call `Bmat(X, Es, t)` to form B(t).
- Compute b(t) = α† * B(t) * α.
A progress bar shows the construction of B(t) across ts.

# Arguments
- `α_ns::Vector{Complex{T}}`: Projection amplitudes αₙ.
- `Psi2ds::Vector{<:AbstractMatrix{T}}`: Vector of ψₙ[i,j] grids.
- `xgrid::Vector{T}`,`ygrid::Vector{T}`: 1D coordinate arrays.
- `ks::Vector{T}`: Wavenumbers (energies = ks.^2).
- `ts::Vector{T}`: Array of time values.
- `memory_efficient::Bool=true`: If true, use `X_block_diag`; otherwise `Xmat_gemm`.
- `direction::Symbol=:x`: Weight direction (`:x` or `:y`).

# Returns
- `bs::Vector{Complex{T}}`: b(t_i) for each t_i.
"""
function b_wavepacket(α_ns::Vector{Complex{T}},Psi2ds::Vector{<:AbstractMatrix{T}},xgrid::Vector{T},ygrid::Vector{T},ks::Vector{T},ts::Vector{T};memory_efficient::Bool=true,direction::Symbol=:x) where {T<:Real}
    X=ifelse(memory_efficient,X_block_diag(Psi2ds,xgrid,ygrid,direction=direction),Xmat_gemm(Psi2ds,xgrid,ygrid,direction=direction))
    Es=ks.^2 # E=k^2
    bs=Vector{Complex{T}}(undef,length(ts))
    @showprogress desc="b(t) construction for ts..." Threads.@threads for i in eachindex(ts)
        B=Bmat(X,Es,ts[i])
        bs[i]=(conj(α_ns)'*B)*α_ns # scalar b(t) = α_n^† B α_n
    end
    return bs # return vector of b(t) values for each t in ts
end

"""
    c_wavepacket(α_ns::Vector{Complex{T}},Psi2ds::Vector{<:AbstractMatrix{T}},xgrid::Vector{T},ygrid::Vector{T},ks::Vector{T},t::T;memory_efficient::Bool=true,direction::Symbol=:x) where {T<:Real}

Compute the 4-point OTOC c(t) = α† B(t) B(t) α at a single time t.

Steps:
1. Build X once (same as in `b_wavepacket`).
2. Form B(t) = Bmat(X, Es, t).
3. Return c(t) = α† * B(t) * B(t) * α.

# Arguments
- `α_ns::Vector{Complex{T}}`: Projection amplitudes αₙ.
- `Psi2ds::Vector{<:AbstractMatrix{T}}`: Vector of ψₙ[i,j] grids.
- `xgrid::Vector{T}`,`ygrid::Vector{T}`: 1D coordinate arrays.
- `ks::Vector{T}`: Wavenumbers (energies = ks.^2).
- `t::T`: Time at which to evaluate c(t).
- `memory_efficient::Bool=true`: If true, use `X_block_diag`; otherwise `Xmat_gemm`.
- `direction::Symbol=:x`: Weight direction (`:x` or `:y`).

# Returns
- `c_val::Complex{T}`: Complex scalar c(t) = α† * B(t) * B(t) * α.
"""
function c_wavepacket(α_ns::Vector{Complex{T}},Psi2ds::Vector{<:AbstractMatrix{T}},xgrid::Vector{T},ygrid::Vector{T},ks::Vector{T},t::T;memory_efficient::Bool=true,direction::Symbol=:x) where {T<:Real}
    X=ifelse(memory_efficient,X_block_diag(Psi2ds,xgrid,ygrid,direction=direction),Xmat_gemm(Psi2ds,xgrid,ygrid,direction=direction))
    Es=ks.^2 # E=k^2
    B=Bmat(X,Es,t)
    return ((conj(α_ns)'*B)*B)*α_ns # scalar c(t) = α_n^† B α_n B α_n
end

"""
    c_wavepacket_evolution(α_ns::Vector{Complex{T}},Psi2ds::Vector{<:AbstractMatrix{T}},xgrid::Vector{T},ygrid::Vector{T},ks::Vector{T},ts::Vector{T};memory_efficient::Bool=true,direction::Symbol=:x) where {T<:Real}

Compute the 4-point OTOC c(t) = α† B(t) B(t) α over multiple times.

This reuses the overlap matrix X once, then for each t ∈ ts:
- Form B(t) = Bmat(X, Es, t).
- Compute c(t) = α† * B(t) * B(t) * α.
Displays a progress bar during the B(t) constructions.

# Arguments
- `α_ns::Vector{Complex{T}}`: Projection amplitudes αₙ.
- `Psi2ds::Vector{<:AbstractMatrix{T}}`: Vector of ψₙ[i,j] grids.
- `xgrid::Vector{T}`,`ygrid::Vector{T}`: 1D coordinate arrays.
- `ks::Vector{T}`: Wavenumbers (energies = ks.^2).
- `ts::Vector{T}`: Array of time points.
- `memory_efficient::Bool=true`: If true, use `X_block_diag`; otherwise `Xmat_gemm`.
- `direction::Symbol=:x`: Weight direction (`:x` or `:y`).

# Returns
- `cs::Vector{Complex{T}}`: c(t_i) for each t_i.
"""
function c_wavepacket_evolution(α_ns::Vector{Complex{T}},Psi2ds::Vector{<:AbstractMatrix{T}},xgrid::Vector{T},ygrid::Vector{T},ks::Vector{T},ts::Vector{T};memory_efficient::Bool=true,direction::Symbol=:x) where {T<:Real}
    X=ifelse(memory_efficient,X_block_diag(Psi2ds,xgrid,ygrid,direction=direction),Xmat_gemm(Psi2ds,xgrid,ygrid,direction=direction))
    Es=ks.^2 # E=k^2
    cs=Vector{Complex{T}}(undef,length(ts))
    @showprogress desc="c(t) construction for ts..." Threads.@threads for i in eachindex(ts)
        B=Bmat(X,Es,ts[i])
        cs[i]=((conj(α_ns)'*B)*B)*α_ns # scalar c(t) = α_n^† B α_n B α_n
    end
    return cs # return vector of c(t) values for each t in ts
end
