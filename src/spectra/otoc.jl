using Bessels, LinearAlgebra, ProgressMeter

include("../billiards/boundarypoints.jl")
include("../states/wavefunctions.jl")

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
- `X::Matrix{Complex{T}}`: N×N Hermitian overlap matrix.
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
- `X::Matrix{Complex{T}}`: N×N Hermitian overlap matrix.
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
    return Xmat
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
- `X::Matrix{T}`: N×N real overlap matrix.
"""
function X_block_diag(Psis::Vector{<:AbstractMatrix{T}},xgrid::Vector{T},ygrid::Vector{T};direction::Symbol=:x) where {T<:Real}
    dx,dy=xgrid[2]-xgrid[1], ygrid[2]-ygrid[1]
    area=dx*dy
    _,ny=size(Psis[1])
    if direction==:x
        Wbig=repeat(xgrid,outer=ny).*area  # length‐M weight vector in column-major order
    elseif direction==:y
        Wbig=repeat(ygrid,inner=nx).*area  # length‐M weight vector in column-major order
    else
        throw(ArgumentError("direction must be either :x or :y"))
    end
    P=hcat(vec.(Psis)...) # stack all flattened ψ’s into an M×N matrix
    # compute X = P' * diag(Wbig) * P in one go, diagonal-block multiply + GEMM
    return P'*(Diagonal(Wbig)*P)
end

"""
     Xmat_gemm(Psis::Vector{<:AbstractMatrix{T}},xgrid::Vector{T},ygrid::Vector{T};n_blas_threads::Int=ceil(Int,Threads.nthreads()/2),direction::Symbol=:x) where {T<:Real}

Compute the “X” overlap matrix using a BLAS-GEMM accelerated approach.
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
    begin 
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
        @inbounds for i in 1:N, j in i+1:N
            X[j,i]=X[i,j] # symmetric
        end
    end
    return X
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
- `Matrix{Complex{T}}`: The computed b_{nm} matrix at time t.
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
    T1=(X.*Φ_nk)*(X.*ΔE_km)
    # T2[n,m] = ∑_k (X[n,k]*ΔE_nk[n,k]) * (X[k,m]*Φ_km[k,m])
    T2=(X.*ΔE_nk)*(X.*Φ_km)
    return 0.5*(T1.-T2)
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
    C_evolution(Psis::Vector{<:AbstractMatrix{T}},Es::Vector{T},xgrid::Vector{T},ygrid::Vector{T},ts::Vector{T};memory_efficient::Bool=false,n_blas_threads::Int=ceil(Int,Threads.nthread()/2),direction::Symbol=:x) where {T<:Real}

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
        Cmat[:,i]
    end
    return Cmat
end
