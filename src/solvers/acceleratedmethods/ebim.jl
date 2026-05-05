"""

    EBIMSolver

Union type collecting all solver backends that support the EBIM correction workflow.
Included solvers are:
- `BoundaryIntegralMethod`
- `DLP_kress`
- `DLP_kress_global_corners`
- `CFIE_kress`
- `CFIE_kress_corners`
- `CFIE_kress_global_corners`
- `CFIE_alpert`
The EBIM routines only require that the solver can assemble:
- the Fredholm matrix `A(k)`,
- its first derivative `dA/dk`,
- its second derivative `d²A/dk²`.
"""
const EBIMSolver=Union{BoundaryIntegralMethod,DLP_kress,DLP_kress_global_corners,CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners,CFIE_alpert}

###################################################################
################# CHEBYSHEV INTERPOLATION PATHWAY #################
###################################################################

"""

    EBIMChebBatchCache{W}

Reusable Chebyshev batch cache for EBIM matrix construction.
This cache stores:
- `ws`: the solver-specific Chebyshev workspace,
- `ks`: the complex wavenumbers for which the workspace was built.
The workspace type `W` is left parametric because each solver family has its own
Chebyshev workspace type.
"""
struct EBIMChebBatchCache{W}
    ws::W
    ks::Vector{ComplexF64}
end

# EBIM needs 3 matrix families (A,dA,ddA), so the effective batch cap is reduced.
@inline _ebim_batch_cap(max_batch_matrices::Int)=max(1,max_batch_matrices÷3)
@inline _ebim_complex_ks(ks)=ComplexF64.(ks) # to guarantee it gets the correct type for cache
@inline function allocate_ebim_cheb_matrices(cache::EBIMChebBatchCache,pts)
    N=boundary_matrix_size(pts)
    Mk=length(cache.ks)
    As=[Matrix{ComplexF64}(undef,N,N) for _ in 1:Mk]
    dAs=[Matrix{ComplexF64}(undef,N,N) for _ in 1:Mk]
    ddAs=[Matrix{ComplexF64}(undef,N,N) for _ in 1:Mk]
    return As,dAs,ddAs
end

"""
    build_ebim_cheb_cache(solver,pts,ks;kwargs...)

Build the solver-specific Chebyshev workspace needed for EBIM batch matrix
construction over the wavenumbers `ks`.

Returns:
- `EBIMChebBatchCache`
"""
function build_ebim_cheb_cache(solver::BoundaryIntegralMethod,pts::BoundaryPoints{T},ks;n_panels::Int=15000,M::Int=5,grading::Symbol=:uniform,geo_ratio::Real=1.05,timeit::Bool=false) where {T<:Real}
    zks=_ebim_complex_ks(ks)
    rmin,rmax=estimate_rmin_rmax(pts,solver.symmetry)
    plans0=Vector{ChebHankelPlanH}(undef,length(zks))
    plans1=Vector{ChebHankelPlanH}(undef,length(zks))
    @benchit timeit=timeit "EBIM BIM deriv plans" Threads.@threads for i in eachindex(zks)
        plans0[i]=plan_h(0,1,zks[i],rmin,rmax;npanels=n_panels,M=M,grading=grading,geo_ratio=geo_ratio)
        plans1[i]=plan_h(1,1,zks[i],rmin,rmax;npanels=n_panels,M=M,grading=grading,geo_ratio=geo_ratio)
    end
    ws=DLPDerivChebWorkspace(T,length(zks))
    return EBIMChebBatchCache((plans0,plans1,ws),zks)
end

function build_ebim_cheb_cache(solver::Union{DLP_kress,DLP_kress_global_corners},pts::BoundaryPointsCFIE{T},ks;n_panels::Int=15000,M::Int=5,grading::Symbol=:uniform,geo_ratio::Real=1.05,plan_nthreads::Int=Threads.nthreads(),ntls::Int=Threads.nthreads(),timeit::Bool=false) where {T<:Real}
    zks=_ebim_complex_ks(ks)
    @benchit timeit=timeit "EBIM DLP_kress direct workspace" directws=build_dlp_kress_workspace(solver,pts)
    @benchit timeit=timeit "EBIM DLP_kress cheb workspace" ws=build_dlp_kress_cheb_workspace(solver,pts,directws,zks;npanels=n_panels,M=M,grading=grading,geo_ratio=geo_ratio,plan_nthreads=plan_nthreads,ntls=ntls)
    return EBIMChebBatchCache(ws,zks)
end

function build_ebim_cheb_cache(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},pts::Vector{BoundaryPointsCFIE{T}},ks;n_panels::Int=15000,M::Int=5,grading::Symbol=:uniform,geo_ratio::Real=1.05,plan_nthreads::Int=Threads.nthreads(),ntls::Int=Threads.nthreads(),timeit::Bool=false) where {T<:Real}
    zks=_ebim_complex_ks(ks)
    @benchit timeit=timeit "EBIM CFIE_kress cheb block cache" block_cache=build_cfie_kress_block_caches(solver,pts;npanels=n_panels,M=M,grading=grading,geo_ratio=geo_ratio)
    @benchit timeit=timeit "EBIM CFIE_kress plans" plans0,plans1,plans2,plans3=build_CFIE_plans_kress(zks,block_cache.rmin,block_cache.rmax;npanels=n_panels,M=M,grading=grading,geo_ratio=geo_ratio,nthreads=plan_nthreads)
    ws=(block_cache=block_cache,plans0=plans0,plans1=plans1,plans2=plans2,plans3=plans3,bessel_ws=CFIE_H0_H1_J0_J1_BesselWorkspace(length(zks);ntls=ntls))
    return EBIMChebBatchCache(ws,zks)
end

function build_ebim_cheb_cache(solver::CFIE_alpert{T},pts::Vector{BoundaryPointsCFIE{T}},ks;
n_panels::Int=15000,M::Int=5,grading::Symbol=:uniform,geo_ratio::Real=1.05,pad=(T(0.95),T(1.05)),plan_nthreads::Int=Threads.nthreads(),ntls::Int=Threads.nthreads(),timeit::Bool=false) where {T<:Real}
    zks=_ebim_complex_ks(ks)
    @benchit timeit=timeit "EBIM CFIE_alpert direct workspace" directws=build_cfie_alpert_workspace(solver,pts)
    @benchit timeit=timeit "EBIM CFIE_alpert cheb workspace" ws=build_cfie_alpert_cheb_workspace(solver,pts,directws,zks;npanels=n_panels,M=M,grading=grading,geo_ratio=geo_ratio,pad=pad,plan_nthreads=plan_nthreads,ntls=ntls)
    return EBIMChebBatchCache(ws,zks)
end

"""
    construct_ebim_cheb_matrices!(As,dAs,ddAs,solver,pts,cache;multithreaded=true)

Construct EBIM matrix triples `(A,dA,ddA)` for all wavenumbers stored in
`cache.ks`, writing them in place.
"""
function construct_ebim_cheb_matrices!(As::Vector{Matrix{ComplexF64}},dAs::Vector{Matrix{ComplexF64}},ddAs::Vector{Matrix{ComplexF64}},solver::BoundaryIntegralMethod,pts::BoundaryPoints{T},cache::EBIMChebBatchCache;multithreaded::Bool=true) where {T<:Real}
    plans0,plans1,ws=cache.ws
    compute_kernel_matrices_DLP_chebyshev_derivatives!(As,dAs,ddAs,pts,solver.symmetry,plans0,plans1;multithreaded=multithreaded,ws=ws)
    assemble_fredholm_matrices_with_derivatives!(As,dAs,ddAs,pts)
    return nothing
end

function construct_ebim_cheb_matrices!(As::Vector{Matrix{ComplexF64}},dAs::Vector{Matrix{ComplexF64}},ddAs::Vector{Matrix{ComplexF64}},solver::Union{DLP_kress,DLP_kress_global_corners},pts::BoundaryPointsCFIE{T},cache::EBIMChebBatchCache;multithreaded::Bool=true) where {T<:Real}
    construct_dlp_kress_matrices_derivatives_chebyshev!(As,dAs,ddAs,pts,cache.ws;multithreaded=multithreaded)
    return nothing
end

function construct_ebim_cheb_matrices!(As::Vector{Matrix{ComplexF64}},dAs::Vector{Matrix{ComplexF64}},ddAs::Vector{Matrix{ComplexF64}},solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},pts::Vector{BoundaryPointsCFIE{T}},cache::EBIMChebBatchCache;multithreaded::Bool=true) where {T<:Real}
    ws=cache.ws
    compute_kernel_matrices_CFIE_kress_chebyshev!(As,dAs,ddAs,pts,ws.plans0,ws.plans1,ws.plans2,ws.plans3,ws.bessel_ws.h0_tls,ws.bessel_ws.h1_tls,ws.bessel_ws.j0_tls,ws.bessel_ws.j1_tls,ws.block_cache;multithreaded=multithreaded)
    return nothing
end

function construct_ebim_cheb_matrices!(As::Vector{Matrix{ComplexF64}},dAs::Vector{Matrix{ComplexF64}},ddAs::Vector{Matrix{ComplexF64}},solver::CFIE_alpert{T},pts::Vector{BoundaryPointsCFIE{T}},cache::EBIMChebBatchCache;multithreaded::Bool=true) where {T<:Real}
    compute_kernel_matrices_CFIE_alpert_chebyshev!(As,dAs,ddAs,solver,pts,cache.ws;multithreaded=multithreaded)
    return nothing
end

"""

    construct_ebim_cheb_matrices(solver,pts,cache;multithreaded=true)

Allocate and construct EBIM Chebyshev matrix triples `(A,dA,ddA)` for all
wavenumbers stored in `cache.ks`.

Returns:
- `(As,dAs,ddAs)`
"""
function construct_ebim_cheb_matrices(solver::EBIMSolver,pts,cache::EBIMChebBatchCache;multithreaded::Bool=true)
    As,dAs,ddAs=allocate_ebim_cheb_matrices(cache,pts)
    construct_ebim_cheb_matrices!(As,dAs,ddAs,solver,pts,cache;multithreaded=multithreaded)
    return As,dAs,ddAs
end

"""
    construct_ebim_cheb_matrix_at!(A,dA,ddA,solver,pts,cache,idx;multithreaded=true)

Construct the EBIM matrix triple for the `idx`-th wavenumber stored in a batch
Chebyshev cache, writing into the preallocated matrices `A`, `dA`, `ddA`.
"""
function construct_ebim_cheb_matrix_at!(A::Matrix{ComplexF64},dA::Matrix{ComplexF64},ddA::Matrix{ComplexF64},
    solver::BoundaryIntegralMethod,pts::BoundaryPoints{T},cache::EBIMChebBatchCache,idx::Int;multithreaded::Bool=true) where {T<:Real}
    plans0,plans1,_=cache.ws
    sym=solver.symmetry
    if isnothing(sym)
        compute_kernel_matrices_DLP_chebyshev_derivatives!(A,dA,ddA,pts,plans0[idx],plans1[idx];multithreaded=multithreaded)
    elseif sym isa Reflection
        ws1=DLPDerivChebWorkspace(T,1)
        compute_kernel_matrices_DLP_chebyshev_derivatives!(A,dA,ddA,pts,sym,plans0[idx],plans1[idx];multithreaded=multithreaded,ws=ws1)
    elseif sym isa Rotation
        ws1=DLPDerivChebWorkspace(T,1)
        compute_kernel_matrices_DLP_chebyshev_derivatives!(A,dA,ddA,pts,sym,plans0[idx],plans1[idx];multithreaded=multithreaded,ws=ws1)
    else
        error("Unsupported BIM symmetry $(typeof(sym))")
    end
    assemble_fredholm_matrices_with_derivatives!(A,dA,ddA,pts)
    return nothing
end

function construct_ebim_cheb_matrix_at!(A::Matrix{ComplexF64},dA::Matrix{ComplexF64},ddA::Matrix{ComplexF64},solver::Union{DLP_kress,DLP_kress_global_corners},pts::BoundaryPointsCFIE{T},cache::EBIMChebBatchCache,idx::Int;multithreaded::Bool=true) where {T<:Real}
    ws=cache.ws
    ws1=typeof(ws)(ws.direct,ws.block_cache,[ws.plans0[idx]],[ws.plans1[idx]],[ws.plansj0[idx]],[ws.plansj1[idx]],DLP_H0_H1_J0_J1_BesselWorkspace(1;ntls=length(ws.bessel_ws.h0_tls)),[ws.ks[idx]],1)
    construct_dlp_kress_matrices_derivatives_chebyshev!([A],[dA],[ddA],pts,ws1;multithreaded=multithreaded)
    return nothing
end

function construct_ebim_cheb_matrix_at!(A::Matrix{ComplexF64},dA::Matrix{ComplexF64},ddA::Matrix{ComplexF64},solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},pts::Vector{BoundaryPointsCFIE{T}},cache::EBIMChebBatchCache,idx::Int;multithreaded::Bool=true) where {T<:Real}
    ws=cache.ws
    compute_kernel_matrices_CFIE_kress_chebyshev!(A,dA,ddA,pts,ws.plans0[idx],ws.plans1[idx],ws.plans2[idx],ws.plans3[idx],ws.bessel_ws.h0_tls,ws.bessel_ws.h1_tls,ws.bessel_ws.j0_tls,ws.bessel_ws.j1_tls,ws.block_cache;multithreaded=multithreaded)
    return nothing
end

function construct_ebim_cheb_matrix_at!(A::Matrix{ComplexF64},dA::Matrix{ComplexF64},ddA::Matrix{ComplexF64},solver::CFIE_alpert{T},pts::Vector{BoundaryPointsCFIE{T}},cache::EBIMChebBatchCache,idx::Int;multithreaded::Bool=true) where {T<:Real}
    ws=cache.ws
    ws1=CFIEAlpertChebWorkspace{T}(ws.direct,ws.block_cache,[ws.plans0[idx]],[ws.plans1[idx]],CFIE_H0_H1_BesselWorkspace(1;ntls=length(ws.bessel_ws.h0_tls)),[ws.ks[idx]],1)
    compute_kernel_matrices_CFIE_alpert_chebyshev!([A],[dA],[ddA],solver,pts,ws1;multithreaded=multithreaded)
    return nothing
end

"""
    solve!(solver,A,dA,ddA,pts,k,dk,cache,idx;
           use_lapack_raw=false,multithreaded=true,use_krylov=true,nev=5)

EBIM solve using the `idx`-th Chebyshev-cached matrix triple from `cache`.
"""
function solve!(solver::EBIMSolver,A::AbstractMatrix{ComplexF64},dA::AbstractMatrix{ComplexF64},ddA::AbstractMatrix{ComplexF64},pts,k,dk,cache::EBIMChebBatchCache,idx::Int;use_lapack_raw::Bool=false,multithreaded::Bool=true,use_krylov::Bool=true,nev::Int=5,return_imag_part::Bool=false)
    fill!(A,0.0+0.0im)
    fill!(dA,0.0+0.0im)
    fill!(ddA,0.0+0.0im)
    construct_ebim_cheb_matrix_at!(A,dA,ddA,solver,pts,cache,idx;multithreaded=multithreaded)
    if use_krylov
        return solve_krylov!(solver,A,dA,ddA,pts,k,dk;multithreaded=multithreaded,nev=nev,return_imag_part=return_imag_part)
    else
        return solve_full!(solver,A,dA,ddA,pts,k,dk;use_lapack_raw=use_lapack_raw,multithreaded=multithreaded,return_imag_part=return_imag_part)
    end
end

###################################################################
######################## DIRECT PATHWAY ###########################
###################################################################

"""
    solve_full!(solver, A, dA, ddA, pts, k, dk; use_lapack_raw=false, multithreaded=true)

Dense generalized-eigenvalue EBIM solve using the fully assembled matrices
`A(k)`, `dA/dk`, and `d²A/dk²`.

This function implements the “full” EBIM correction step after the three
matrices have already been assembled at a trial wavenumber `k`. It solves the
generalized eigenproblem

    A v = λ dA v

and interprets the small generalized eigenvalues `λ` as first-order local
corrections around `k`.

The basic local model is:

    A(k + ε) ≈ A(k) + ε dA(k) + 1/2 ε² ddA(k).

If `A(k + ε)` is singular, then to first order one gets

    A v ≈ -ε dA v,

hence the generalized eigenvalue relation with

    λ ≈ -ε.

This function then adds a second-order correction using the left/right
generalized eigenvectors.

# Mathematical correction formula
For each accepted generalized eigenpair, the code uses

- first-order correction:
    ε₁ = -λ

- second-order correction:
    ε₂ = -1/2 * ε₁² * [(u† ddA v) / (u† dA v)]

- (Optional Re) of the final wavenumber for error estimate

where:
- `v` is the right generalized eigenvector,
- `u` is the left generalized eigenvector,
- `†` denotes conjugate transpose.

The final corrected eigenvalue estimate is

    k_corrected = k + ε₁ + ε₂

and the associated “tension-like” scalar returned by this function is

    tension = |ε₁ + ε₂|.

# Inputs
- `solver::EBIMSolver`:
  Included for interface consistency. It is not used directly inside this
  function, since the matrices are assumed already assembled.
- `A::AbstractMatrix{Complex{T}}`:
  Fredholm matrix `A(k)`.
- `dA::AbstractMatrix{Complex{T}}`:
  First derivative `dA/dk`.
- `ddA::AbstractMatrix{Complex{T}}`:
  Second derivative `d²A/dk²`.
- `pts`:
  Boundary discretization object(s). Present for interface uniformity; not used
  directly here.
- `k`:
  Center wavenumber about which the EBIM correction is performed.
- `dk`:
  Acceptance window for the generalized eigenvalues. Only eigenvalues with
  both `|Re(λ)| < dk` and `|Im(λ)| < dk` are retained.
- `use_lapack_raw::Bool=false`:
  If `true`, use the low-level LAPACK generalized eigensolver backend.
  Otherwise use the higher-level Julia/general wrapper.
- `multithreaded::Bool=true`:
  Currently included for API consistency. Matrix assembly does not happen here,
  so this flag is not used internally in this function.
- `return_imag_part::Bool=false`: Whether to return the imaginary part of the corrected eigenvalues (wavenumbers) in the output. By default, only the real part is returned, but setting this to true will include the imaginary part as well, which can be useful for errors.

# Returns
- `λ_corrected::Vector{RT}`:
  Corrected real wavenumber estimates near `k`.
- `tens::Vector{RT}`:
  Absolute EBIM corrections `|ε₁ + ε₂|`, often used as a local spectral
  quality/tension measure.

# Acceptance logic
After solving the generalized eigenproblem, the function filters to eigenvalues
inside the `dk` box in the complex plane. If none survive, it returns two empty
vectors.

# Performance notes
This is the dense all-eigenpairs variant. It is appropriate when:
- the matrix size is moderate,
- many nearby eigenvalues may matter,
- one wants the most direct generalized-eigenvalue solve.

For larger systems or when only a few nearby eigenvalues are needed, see
`solve_krylov!`.
"""
function solve_full!(solver::EBIMSolver,A::AbstractMatrix{Complex{T}},dA::AbstractMatrix{Complex{T}},ddA::AbstractMatrix{Complex{T}},pts,k,dk;use_lapack_raw::Bool=false,multithreaded::Bool=true,return_imag_part::Bool=false) where {T<:Real}
    if use_lapack_raw
        @blas_multi MAX_BLAS_THREADS λ,VR,VL=generalized_eigen_all_LAPACK_LEGACY(A,dA)
    else
        @blas_multi MAX_BLAS_THREADS λ,VR,VL=generalized_eigen_all(A,dA)
    end
    CT=eltype(λ)
    RT=real(CT)
    KT=return_imag_part ? CT : RT
    valid=(abs.(real.(λ)).<dk).&(abs.(imag.(λ)).<dk)
    if !any(valid)
        return Vector{KT}(),Vector{RT}()
    end
    λ=λ[valid]
    VR=VR[:,valid]
    VL=VL[:,valid]
    corr_1=Vector{CT}(undef,length(λ))
    corr_2=Vector{CT}(undef,length(λ))
    buf=similar(VR,eltype(ddA),size(ddA,1))  # reusable buffer
    for i in eachindex(λ)
        v_right=VR[:,i]
        v_left=VL[:,i]
        # ddA * v_right → buf
        mul!(buf,ddA,v_right)
        numerator=dot(v_left,buf)   # u† * (ddA v)
        # dA * v_right → buf (overwrite)
        mul!(buf,dA,v_right)
        denominator=dot(v_left, buf) # u† * (dA v)
        corr_1[i]= -λ[i]
        # optional safety guard 
        if abs(denominator)>1e-15
            corr_2[i]= -0.5*corr_1[i]^2*(numerator/denominator)
        else
            corr_2[i]=zero(CT)
        end
    end
    λ_corrected=k.+(return_imag_part ? corr_1.+corr_2 : real.(corr_1.+corr_2))
    tens=abs.(corr_1.+corr_2)
    return λ_corrected,tens
end


"""
    solve_krylov!(solver, A, dA, ddA, pts, k, dk; multithreaded=true, nev=5, tol=1e-14, maxiter=5000, krylovdim=min(40,max(40,2*nev+1)), return_imag_part::Bool=false)

Shift-invert Krylov EBIM solve for a small number of nearby eigenvalues. 
This variant is usually preferable when the matrix is too large for a full generalized eigendecomposition.

This function is the iterative counterpart of `solve_full!`. Instead of solving
the full dense generalized eigenproblem

    A v = λ dA v,

it applies Krylov methods to the shift-invert operator

    C = A^{-1} dA.

Since generalized eigenpairs satisfy

    A v = λ dA v
    => A^{-1} dA v = (1/λ) v,

small `|λ|` correspond to large `|μ|` where `μ = 1/λ`. Therefore an iterative
eigensolve for the largest-magnitude eigenvalues of `C` recovers the desired
small generalized eigenvalues near the current `k`.

# Left and right eigenvectors
The second-order EBIM correction requires both right and left generalized
eigenvectors. Therefore the function solves:
- the right problem for `C = A^{-1} dA`,
- the left/adjoint problem for
      C_L = (A')^{-1} (dA'),

then pairs the two eigenvector families by eigenvalue proximity.

# Mathematical correction formula
Exactly as in `solve_full!`, the function uses

- ε₁ = -(λ)
- ε₂ = -1/2 * ε₁² * [(u† ddA v) / (u† dA v)]
- (Optional Re) of the final wavenumber for error estimate

and returns

- corrected eigenvalue estimate `k + ε₁ + ε₂`,
- tension `|ε₁ + ε₂|`.

The real part is optionally explicitly used in the first-order correction since the target
is a real wavenumber correction.

# Inputs
- `solver::EBIMSolver`:
  Interface placeholder; the matrices are assumed already assembled.
- `A, dA, ddA`:
  Matrix, first derivative, and second derivative at the trial `k`.
  Note that `A` is factorized in place through `lu!`, so it should be treated as
  disposable inside this function.
- `pts`:
  Present for API consistency; not used directly.
- `k`:
  Trial wavenumber.
- `dk`:
  Acceptance window for corrected eigenvalues.
- `multithreaded::Bool=true`:
  Kept for interface compatibility; no assembly occurs here.
- `nev::Int=5`:
  Number of Krylov eigenpairs requested from both right and left solves.
  !!! Check with Weyl that you have enough Krylov space to get the desired number of pairs after filtering with `dk`.
- `tol=1e-14`:
  Convergence tolerance for `eigsolve`.
- `maxiter=5000`:
  Maximum Krylov iterations.
- `krylovdim`:
  Krylov subspace dimension used by `eigsolve`.
- `return_imag_part::Bool=false`: Whether to return the imaginary part of the corrected eigenvalues (wavenumbers) in the output. By default, only the real part is returned, but setting this to true will include the imaginary part as well, which can be useful for errors.

# Returns
- `λ_out::Vector{Union{T,Complex{T}}}`:
  Corrected wavenumber estimates accepted inside the `dk` window.
- `tens_out::Vector{Union{T,Complex{T}}}`:
  Corresponding EBIM tensions.

# Acceptance logic
After recovering `λ = 1/μ`, only candidates with
- `|Re(λ)| < dk`
- `|Im(λ)| < dk`
are accepted.
"""
function solve_krylov!(solver::EBIMSolver,A::AbstractMatrix{Complex{T}},dA::AbstractMatrix{Complex{T}},ddA::AbstractMatrix{Complex{T}},pts,k,dk;multithreaded::Bool=true,nev::Int=5,tol=1e-14,maxiter=5000,krylovdim::Int=min(40,max(40,2*nev+1)),return_imag_part::Bool=false) where {T<:Real}
    CT=eltype(A)
    RT=real(CT)
    KT=return_imag_part ? CT : RT
    n=size(A,1)
    @blas_multi MAX_BLAS_THREADS F=lu!(A) # enables fast solves with A (shift–invert) by creating triangular matrices to internally act on vectors. This is an expensive n(O^3) operation. Reuses A's storage; adjoint(F) gives fast solves with A'. We use lu! since A is not reused in this scope 
    @blas_1 begin
        Ft=adjoint(F) # define outside op_l! for reuse 
        dAt=adjoint(dA) # define outside op_l! for reuse
        tmp=zeros(CT,n) # # reusable work buffer to avoid allocations in operator applications
        # shift–invert map C := A^{-1} dA 
        # Mathematics: linearize A(k+ε) ≈ A + ε dA. Singularity A+ε dA ≈ 0 -> A v = -ε dA v ⇒ (generalized EVP) A v = λ dA v with λ=-ε
        # Hence (A^{-1} dA) v = μ v with μ = 1/λ. Small |λ| correspond to large |μ|
        function op_r!(y,x)
            mul!(y,dA,x)  # y <- dA * x  without extra allocations
            ldiv!(F,y) # y <- A \ y  (using LU) without extra allocations
            return y
        end
        C=LinearMaps.LinearMap{CT}(op_r!,n,n;ismutating=true) # LinearMaps wraps the op for Krylov without forming A^{-1}dA explicitly. Crucial to reduce allocations
        μr,VRlist,_=eigsolve(C,n,nev,:LM;tol=tol,maxiter=maxiter,krylovdim=krylovdim) # compute the largest |μ| -> smallest |λ|
        λ=inv.(μr) # # map back via λ = 1/μ                          
        ord=sortperm(abs.(λ))
        λ=λ[ord]
        μr=μr[ord]
        VRlist=VRlist[ord]
        # left shift–invert map C_L = (A')^{-1} (dA') acting on column vectors u. This is solving the adjoint eigenproblem
        # If C u = μ u is the right EVP, then (A')^{-1} (dA') u = μ u gives the corresponding left EVP for the pair (A,dA).
        # Those u are (up to scaling) left generalized eigenvectors of A v = λ dA v: u' A = λ u' dA with λ = 1/μ.
        function op_l!(y,x)
            copyto!(tmp,x) # tmp <- x  (so we can reuse tmp in-place)
            ldiv!(Ft,tmp)  # tmp <- (A') \ tmp without extra allocations
            mul!(y,dAt,tmp) # y <- (dA') * tmp = (dA') * (A')^{-1} * x  without extra allocations
            return y
        end
        Cl=LinearMaps.LinearMap{CT}(op_l!,n,n;ismutating=true) # adjoint-side LinearMap (no explicit transposed matrices formed beyond dA', A')
        #w0=zeros(CT,n);randn!(rng,w0) # random complex starting vector for krylov
        μl,ULlist,_=eigsolve(Cl,n,nev,:LM;tol=tol,maxiter=maxiter,krylovdim=krylovdim) # left eigenvalues should match μr (up to num. noise), reuse v0
        # Pair left and right sets by closeness in μ (using conjugation to be robust for complex arithmetic)
        perm=@inbounds [argmin(abs.(μl.-conj(μrj))) for μrj in μr] # if stable solve then this should perfectly align the left and right eigenvectors
        ULlist=ULlist[perm]
        λ_out=Vector{KT}(undef,nev) # at most we will have nev
        tens_out=Vector{RT}(undef,nev)
        m=0 # keeps track of valid eigvals, if in the end 0 empty interval
        buf=zeros(CT,n) # reusable temp array used with mul! to always overwrite previous result
        @inbounds for j in 1:nev
            λj=λ[j]
            if !(abs(real(λj))<dk && abs(imag(λj))<dk)
                continue
            end
            v=VRlist[j];u=ULlist[j]
            @blas_multi MAX_BLAS_THREADS mul!(buf,ddA,v) # buf <- ddA * v
            num=dot(u,buf)  # numerator = u' * ddA * v
            @blas_multi MAX_BLAS_THREADS mul!(buf,dA,v)  # buf <- dA * v, overwrites previous buf
            den=dot(u,buf)   # denominator = u' * dA * v  (bi-orthogonal pairing; scaling cancels in the ratio)
            # first-order: ε1 = -λ  (since A v = λ dA v with λ = -ε to first order)
            # second-order: ε2 = -0.5 ε1^2 * (u' ddA v)/(u' dA v)
            c1=-(λj)
            c2=zero(CT)
            if abs(den)>1e-15 # soft guard
                c2-=0.5*c1^2*(num/den) # second-order correction (scale-invariant thanks to the ratio)
            end
            t=c1+c2
            abst=abs(t)
            kc=complex(k)+t
            if abs(real(kc)-k)<dk # accept corrected root whose real part lies in the local k-window
                m+=1
                λ_out[m]=return_imag_part ? kc : real(kc) # corrected k = k + ε1 + ε2 
                tens_out[m]=abst # tension ≈ |ε1 + ε2|
            end
        end
        if m==0;return KT[],RT[];end # if it happens to be empty solve in dk, return empty. tens is real
        resize!(λ_out,m);resize!(tens_out,m) # since nev > expected eigvals in dk due to added padding, trim it
        return λ_out,tens_out
    end
end


"""
    gev_eigconds(A, B, λ, VR::AbstractMatrix, VL::AbstractMatrix; p=2)
    gev_eigconds(A, B, λ, VR::Vector{<:AbstractVector}, VL::Vector{<:AbstractVector}; p=2)

Estimate condition numbers for generalized eigenvalues of the matrix pencil
`A - λ B`.

Mathematical meaning
--------------------
For a generalized eigenpair of

    A x = λ B x
    y† A = λ y† B

with right eigenvector `x` and left eigenvector `y`, an estimate
of the condition number of `λ` is given by

    κ(λ) ≈ ( ||x|| ||y|| (||A|| + |λ| ||B||) ) / ( |λ| |y† B x| ).

# Inputs
- `A, B`:
  Matrices defining the generalized eigenvalue problem.
- `λ`:
  Vector of generalized eigenvalues.
- `VR, VL`:
  Right and left eigenvectors, either:
  - as column-stacked matrices, or
  - as vectors of vectors.
- `p=2`:
  Norm used in `opnorm`.

# Returns
- `κ::Vector{Float64}`: Estimated condition number.
"""
function gev_eigconds(A,B,λ,VR::AbstractMatrix,VL::AbstractMatrix;p=2)
    nA=opnorm(A,p)
    nB=opnorm(B,p)
    n=length(λ)
    κ=Vector{Float64}(undef,n)
    for j in 1:n
        x=VR[:,j]
        y=VL[:,j]
        v=norm(x)*norm(y)
        d=abs(dot(y,B*x))
        κ[j]=(d==0||λ[j]==0) ? Inf : (v*(nA+abs(λ[j])*nB)/(abs(λ[j])*d))
    end
    return κ
end
function gev_eigconds(A,B,λ,VR::Vector{<:AbstractVector},VL::Vector{<:AbstractVector};p=2)
    nA=opnorm(A,p)
    nB=opnorm(B,p)
    n=length(λ)
    κ=Vector{Float64}(undef,n)
    for j in 1:n
        x=VR[j]
        y=VL[j]
        v=norm(x)*norm(y)
        d=abs(dot(y,B*x))
        κ[j]=(d==0||λ[j]==0) ? Inf : (v*(nA+abs(λ[j])*nB)/(abs(λ[j])*d))
    end
    return κ
end

"""
    solve_full_INFO!(solver, A, dA, ddA, pts, k, dk;
                     use_lapack_raw=false, multithreaded=true)

Diagnostic/instrumented version of `solve_full!`.

Purpose
-------
This function performs the same dense EBIM generalized-eigenvalue correction as
`solve_full!`, but with detailed logging and timing breakdowns intended for:
- debugging,
- benchmarking,
- checking matrix conditioning,
- comparing generalized eigensolver backends.

# What it reports
Depending on the backend, the function reports:
- assembly time for `A`, `dA`, `ddA`,
- condition numbers of `A` and `dA`,
- generalized eigensolver timing,
- percentage of valid generalized eigenvalues,
- smallest generalized eigenvalue magnitude,
- median generalized eigenvalue condition number,
- timing spent in second-order correction formulas,
- total runtime summary.

# Backend behavior
- `use_lapack_raw=true`:
  Uses low-level LAPACK generalized eigensolvers (`ggev!` or `ggev3!`
  depending on LAPACK version).
- `use_lapack_raw=false`:
  Uses Julia's generalized eigen wrapper for the right problem and again on the
  adjoint pair `(A', dA')` to obtain left eigenvectors.

# Outputs
Same as `solve_full!`:
- corrected wavenumber estimates,
- tensions.
"""
function solve_full_INFO!(solver::EBIMSolver,A::AbstractMatrix{Complex{T}},dA::AbstractMatrix{Complex{T}},ddA::AbstractMatrix{Complex{T}},pts,k,dk;use_lapack_raw::Bool=false,multithreaded::Bool=true) where {T<:Real}
    s=time()
    s_constr=time()
    basis=AbstractHankelBasis()
    fill!(A,Complex{T}(zero(T),zero(T)))
    fill!(dA,Complex{T}(zero(T),zero(T)))
    fill!(ddA,Complex{T}(zero(T),zero(T)))
    @info "Constructing A,dA,ddA Fredholm matrix and its derivatives..."
    @time construct_matrices!(solver,basis,A,dA,ddA,pts,k;multithreaded=multithreaded)
    e_constr=time()
    if use_lapack_raw
        if LAPACK.version()<v"3.6.0"
            s_gev=time()
            @info "Doing ggev!"
            @info "Matrix condition numbers: cond(A) = $(cond(A)), cond(dA) = $(cond(dA))"
            @blas_multi MAX_BLAS_THREADS α,β,VL,VR=LAPACK.ggev!('V','V',copy(A),copy(dA))
            e_gev=time()
        else
            s_gev=time()
            @info "Doing ggev3!"
            @info "Matrix condition numbers: cond(A) = $(cond(A)), cond(dA) = $(cond(dA))"
            @blas_multi MAX_BLAS_THREADS α,β,VL,VR=LAPACK.ggev3!('V','V',copy(A),copy(dA))
            e_gev=time()
        end
        λ=α./β
        valid_indices=.!isnan.(λ).&.!isinf.(λ)
        @info "% of valid indices: $(count(valid_indices)/length(λ))"
        λ=λ[valid_indices]
        VR=VR[:,valid_indices]
        VL=VL[:,valid_indices]
        sort_order=sortperm(abs.(λ))
        λ=λ[sort_order]
        @info "Smallest eigenvalue: $(minimum(abs.(λ)))"
        VR=VR[:,sort_order]
        VL=VL[:,sort_order]
        normalize!(VR)
        normalize!(VL)
        κ_all=gev_eigconds(A,dA,λ,VR,VL;p=2)
        @info "Median eigenvalue condition number: $(median(κ_all))"
    else
        @info "Solving Julia's ggev for A,dA"
        s_gev=time()
        @blas_multi MAX_BLAS_THREADS F=eigen(A,dA)
        λ=F.values
        VR=F.vectors
        @info "Solving Julia's ggev for A' and dA' for the left eigenvectors"
        @blas_multi MAX_BLAS_THREADS F_adj=eigen(A',dA')
        e_gev=time()
        VL=F_adj.vectors
        valid_indices=.!isnan.(λ).&.!isinf.(λ)
        @info "Number of valid indices: $(count(valid_indices))"
        λ=λ[valid_indices]
        VR=VR[:,valid_indices]
        VL=VL[:,valid_indices]
        sort_order=sortperm(abs.(λ))
        λ=λ[sort_order]
        @info "Smallest eigenvalue: $(minimum(abs.(λ)))"
        VR=VR[:,sort_order]
        VL=VL[:,sort_order]
        normalize!(VR)
        normalize!(VL)
        κ_all=gev_eigconds(A,dA,λ,VR,VL;p=2)
        @info "Median eigenvalue condition number: $(median(κ_all))"
    end
    L=eltype(real.(λ))
    valid=(abs.(real.(λ)).<dk).&(abs.(imag.(λ)).<dk)
    if !any(valid)
        total_time=time()-s
        @info "Final computation time without extrema of SVD for cond calculation: $(total_time) s"
        println("%%%%% SUMMARY %%%%%")
        println("Percentage of total time (most relevant ones): ")
        println("A,dA,ddA construction: $(100*(e_constr-s_constr)/total_time) %")
        println("Generalized eigen: $(100*(e_gev-s_gev)/total_time) %")
        println("%%%%%%%%%%%%%%%%%%%")
        return Vector{L}(),Vector{L}()
    end
    λ=λ[valid]
    CT=eltype(λ)
    RT=real(CT)
    VR=VR[:,valid]
    VL=VL[:,valid]
    corr_1=Vector{CT}(undef,length(λ))
    corr_2=Vector{CT}(undef,length(λ))
    @info "Corrections to the eigenvalues and eigenvectors..."
    s_corr=time()
    @time for i in eachindex(λ)
        v_right=VR[:,i]
        v_left=VL[:,i]
        buf=similar(v_right)
        mul!(buf,ddA,v_right)
        numerator=dot(v_left,buf)
        mul!(buf,dA,v_right)
        denominator=dot(v_left,buf)
        @info "Denominator for index $i : $denominator"
        corr_1[i]=-λ[i]
        corr_2[i]=abs(denominator)>1e-15 ? -0.5*corr_1[i]^2*(numerator/denominator) : zero(CT)
        @info "Correction for index $i : ε₁ = $(corr_1[i]), ε₂ = $(corr_2[i]), total = $(corr_1[i]+corr_2[i])"
    end
    e_corr=time()
    λ_corrected=complex(k).+corr_1.+corr_2
    tens=abs.(corr_1.+corr_2)
    e=time()
    total_time=e-s
    @info "Final computation time without extrema of SVD for cond calculation: $(total_time) s"
    println("%%%%% SUMMARY %%%%%")
    println("Percentage of total time (most relevant ones): ")
    println("A,dA,ddA construction: $(100*(e_constr-s_constr)/total_time) %")
    println("Generalized eigen: $(100*(e_gev-s_gev)/total_time) %")
    println("2nd order corrections: $(100*(e_corr-s_corr)/total_time) %")
    println("%%%%%%%%%%%%%%%%%%%")
    return λ_corrected,tens
end

"""
    solve_krylov_INFO!(solver, A, dA, ddA, pts, k, dk;
                       multithreaded=true, nev=5, tol=1e-14,
                       maxiter=5000, krylovdim=min(40,max(40,2*nev+1)))

Diagnostic/instrumented version of `solve_krylov!`.

Purpose
-------
This function runs the shift-invert Krylov EBIM algorithm while printing a
detailed trace of:
- matrix assembly,
- LU factorization,
- right and left Krylov eigensolves,
- number of accepted eigenvalues,
- eigenvalue conditioning,
- second-order correction timing.

# What it reports
Among other things, the function logs:
- matrix sizes and element type,
- Krylov convergence counts and iteration numbers,
- number of accepted eigenvalues inside the `dk` window,
- median generalized eigenvalue condition number,
- median lower bound on relative eigenvalue error,
- per-index correction denominators,
- timing split across assembly, factorization/eigensolves, and corrections.

# Returns
Same as `solve_krylov!`:
- corrected real wavenumber estimates,
- tensions.
"""
function solve_krylov_INFO!(solver::EBIMSolver,A::AbstractMatrix{Complex{T}},dA::AbstractMatrix{Complex{T}},ddA::AbstractMatrix{Complex{T}},pts,k,dk;multithreaded::Bool=true,nev::Int=5,tol=1e-14,maxiter=5000,krylovdim::Int=min(40,max(40,2*nev+1))) where {T<:Real}
    fill!(A,Complex{T}(zero(T),zero(T)))
    fill!(dA,Complex{T}(zero(T),zero(T)))
    fill!(ddA,Complex{T}(zero(T),zero(T)))
    basis=AbstractHankelBasis()
    t0=time()
    @info "Constructing A,dA,ddA at k=$k..."
    @time construct_matrices!(solver,basis,A,dA,ddA,pts,k;multithreaded=multithreaded)
    n=size(A,1)
    CT=eltype(A)
    RT=real(eltype(A))
    @info "Sizes: A=$(size(A)) dA=$(size(dA)) ddA=$(size(ddA)) eltype=$CT"
    Afac=copy(A)
    @info "LU factorization of A (shift-invert)..."
    t1=time()
    @time F=lu!(Afac)
    Ft=adjoint(F)
    dAt=adjoint(dA)
    tmp=zeros(CT,n)
    buf=zeros(CT,n)
    function op_r!(y,x)
        mul!(y,dA,x)
        ldiv!(F,y)
        return y
    end
    C=LinearMaps.LinearMap{CT}(op_r!,n,n;ismutating=true)
    @info "Right eigsolve on A^{-1}dA (nev=$nev tol=$tol krylovdim=$krylovdim)..."
    @time μr,VR,infoR=eigsolve(C,n,nev,:LM;tol=tol,maxiter=maxiter,krylovdim=krylovdim)
    λ=inv.(μr)
    ord=sortperm(abs.(λ))
    λ=λ[ord]
    μr=μr[ord]
    VR=VR[ord]
    @info "Right eigsolve: converged=$(infoR.converged) iters=$(infoR.numiter)"
    function op_l!(y,x)
        copyto!(tmp,x)
        ldiv!(Ft,tmp)
        mul!(y,dAt,tmp)
        return y
    end
    Cl=LinearMaps.LinearMap{CT}(op_l!,n,n;ismutating=true)
    @info "Left eigsolve on (A')^{-1}(dA')..."
    @time μl,UL,infoL=eigsolve(Cl,n,nev,:LM;tol=tol,maxiter=maxiter,krylovdim=krylovdim)
    @info "Left eigsolve: converged=$(infoL.converged) iters=$(infoL.numiter)"
    perm=@inbounds [argmin(abs.(μl.-conj(μrj))) for μrj in μr]
    UL=UL[perm]
    acc=@. (abs(real(λ))<dk) & (abs(imag(λ))<dk)
    nacc=count(acc)
    @info "Accepted in dk-window: $nacc / $nev"
    if nacc==0
        t2=time()
        @info "Timings: construct=$(t1-t0)s, factor+eigs=$(t2-t1)s, total=$(t2-t0)s"
        return CT[],RT[]
    end
    λ=λ[acc]
    VR=VR[acc]
    UL=UL[acc]
    κ_all=gev_eigconds(A,dA,λ,VR,UL;p=2)
    rel_bound_all=κ_all.*eps(RT)
    @info "Median eigenvalue condition number: $(median(κ_all))"
    @info "Median lower bound on relative eigenvalue error: $(median(rel_bound_all))"
    λ_out=Vector{CT}(undef,nacc)
    tens=Vector{RT}(undef,nacc)
    m=0
    @info "Second-order corrections..."
    t2=time()
    @time for j in 1:nacc
        v=VR[j]
        u=UL[j]
        λj=λ[j]
        mul!(buf,ddA,v)
        num=dot(u,buf)
        mul!(buf,dA,v)
        den=dot(u,buf)
        @info "Denominator for index $j : $den"
        c1=-λj
        c2=zero(CT)
        if abs(den)>1e-15
            c2-=0.5*c1^2*(num/den)
        end
        kc=complex(k)+c1+c2
        m+=1
        λ_out[m]=kc
        tens[m]=abs(c1+c2)
        @info "Correction" j=j λ=λj k_corrected=kc imag_k=imag(kc) tension=tens[m]
    end
    t3=time()
    @info "Timings: construct=$(t1-t0)s, factor+eigs=$(t2-t1)s, corrections=$(t3-t2)s, total=$(t3-t0)s"
    return λ_out,tens
end

"""
    solve!(solver, A, dA, ddA, pts, k, dk;
           use_lapack_raw=false, multithreaded=true,
           use_krylov=true, nev=5)

Unified high-level EBIM solve entry point.
This function first assembles the matrix triple

- `A(k)`
- `dA/dk`
- `d²A/dk²`

for the supplied solver and boundary discretization, then dispatches to either:
- `solve_krylov!` if `use_krylov=true`,
- `solve_full!` otherwise.

# Workflow
1. Construct `A`, `dA`, `ddA` at trial wavenumber `k`.
2. Run the chosen EBIM correction backend.
3. Return corrected nearby eigenvalue estimates and tensions.

# Inputs
- `solver::EBIMSolver`:
  Any solver supporting derivative-aware matrix assembly.
- `A, dA, ddA`:
  Preallocated complex matrices.
- `pts`:
  Boundary discretization(s).
- `k`:
  Trial wavenumber.
- `dk`:
  Acceptance window for EBIM corrections.
- `use_lapack_raw::Bool=false`:
  Passed to the dense/full backend.
- `multithreaded::Bool=true`:
  Passed to matrix assembly and the chosen backend.
- `use_krylov::Bool=true`:
  Selects Krylov or full generalized eigensolve.
- `nev::Int=5`:
  Number of requested eigenpairs for the Krylov backend.
- `return_imag_part::Bool=false`:
  Whether to return the imaginary part of the corrected eigenvalues (wavenumbers) in the output. By default, only the real part is returned, but setting this to true will include the imaginary part as well, which can be useful for errors.  

# Returns
- corrected wavenumber estimates,
- tensions.

# Recommended usage
- Use `use_krylov=true` for large matrices and local searches.
- Use `use_krylov=false` for smaller dense problems.
"""
function solve!(solver::EBIMSolver,A::AbstractMatrix{Complex{T}},dA::AbstractMatrix{Complex{T}},ddA::AbstractMatrix{Complex{T}},pts,k,dk;use_lapack_raw::Bool=false,multithreaded::Bool=true,use_krylov::Bool=true,nev::Int=5,return_imag_part::Bool=false) where {T<:Real}
    basis=AbstractHankelBasis()
    @blas_1 construct_matrices!(solver,basis,A,dA,ddA,pts,k;multithreaded=multithreaded)
    if use_krylov
        return solve_krylov!(solver,A,dA,ddA,pts,k,dk;multithreaded=multithreaded,nev=nev,return_imag_part=return_imag_part)
    else
        return solve_full!(solver,A,dA,ddA,pts,k,dk;use_lapack_raw=use_lapack_raw,multithreaded=multithreaded,return_imag_part=return_imag_part)
    end
end

"""
    solve_INFO!(solver, A, dA, ddA, pts, k, dk;
                use_lapack_raw=false, multithreaded=true,
                use_krylov=true)

Diagnostic wrapper around `solve!`.

Purpose
-------
This routine chooses between the instrumented EBIM backends:
- `solve_krylov_INFO!` when `use_krylov=true`,
- `solve_full_INFO!` otherwise.

It is intended for development and profiling rather than routine production
use.

# Returns
Same as `solve!`:
- corrected eigenvalue estimates,
- tensions.

# Use case
Call this when you want detailed logging of:
- assembly cost,
- eigensolver behavior,
- conditioning,
- correction denominators,
- timing breakdowns.
"""
function solve_INFO!(solver::EBIMSolver,A::AbstractMatrix{Complex{T}},dA::AbstractMatrix{Complex{T}},ddA::AbstractMatrix{Complex{T}},pts,k,dk;use_lapack_raw::Bool=false,multithreaded::Bool=true,use_krylov::Bool=true) where {T<:Real}
    if use_krylov
        return solve_krylov_INFO!(solver,A,dA,ddA,pts,k,dk;multithreaded=multithreaded)
    else
        return solve_full_INFO!(solver,A,dA,ddA,pts,k,dk;use_lapack_raw=use_lapack_raw,multithreaded=multithreaded)
    end
end

function overlap_and_merge_ebim!(k_left::Vector{K},ten_left::Vector{T},k_right::Vector{K},ten_right::Vector{T},control_left::Vector{Bool},kl::T,kr::T;tol::T=T(1e-5)) where {K<:Number,T<:Real}
    isempty(k_right) && return nothing
    if isempty(k_left)
        append!(k_left,k_right)
        append!(ten_left,ten_right)
        append!(control_left,fill(false,length(k_right)))
        return nothing
    end
    for j in eachindex(k_right)
        kj=k_right[j]
        tj=ten_right[j]
        hit=false
        for i in eachindex(k_left)
            if abs(real(k_left[i])-real(kj))<=tol
                hit=true
                if tj<ten_left[i]
                    k_left[i]=kj
                    ten_left[i]=tj
                end
                control_left[i]=true
                break
            end
        end
        if !hit
            push!(k_left,kj)
            push!(ten_left,tj)
            push!(control_left,false)
        end
    end
    p=sortperm(real.(k_left))
    k_left[:]=k_left[p]
    ten_left[:]=ten_left[p]
    control_left[:]=control_left[p]
    return nothing
end

"""
    compute_spectrum(solver::EBIMSolver,billiard::Bi,k1::T,k2::T;dk=(k)->0.05*k^(-1/3),tol=1e-4,use_lapack_raw=false,multithreaded_matrices=false,use_krylov=true,seg_reuse_frac=0.95,solve_info=true,use_chebyshev=false,n_panels=15000,M=5) -> Tuple{Vector{T},Vector{T}}

Compute the spectrum and corresponding tensions of a billiard problem using the expanded boundary integral method (EBIM) over the wavenumber interval `[k1,k2]`.

The interval is partitioned into segments, and for each segment the boundary geometry is constructed only once (at the segment’s upper wavenumber) and reused for all `k` values within that segment. This reduces geometric and allocation overhead while maintaining accuracy.

# Arguments
- `solver::EBIMSolver`: Boundary integral solver (e.g. BIM, Kress, Alpert variants).
- `billiard::Bi`: Billiard geometry.
- `k1::T`: Lower bound of the wavenumber interval.
- `k2::T`: Upper bound of the wavenumber interval.

# Keyword arguments
- `dk::Function`: Step-size function for generating the `k` grid. Default follows the scaling law `0.05*k^(-1/3)`.
- `tol::T=1e-4`: Tolerance for merging overlapping eigenvalues.
- `use_lapack_raw::Bool=false`: Use raw LAPACK `ggev` instead of Julia’s `eigen(A,B)`.
- `multithreaded_matrices::Bool=false`: Enable multithreading in matrix construction.
- `use_krylov::Bool=true`: Use Krylov-based shift–invert solver instead of full generalized EVP.
- `seg_reuse_frac::T=0.95`: Controls segment size; geometry is reused while `k` stays within this fraction of the segment’s upper bound.
- `solve_info::Bool=true`: Print detailed information during the solve process.
- `return_imag_part::Bool=false`: Whether to return the imaginary part of the corrected eigenvalues (wavenumbers) in the output. By default, only the real part is returned, but setting this to true will include the imaginary part as well, which can be useful for errors.

# Chebyshev specific kwargs:
- `use_chebyshev::Bool=false`: Use Chebyshev interpolation for matrix assembly across segments.
- `n_panels::Int=15000`: Number of panels for Chebyshev interpolation if `use_chebyshev=true`.
- `M::Int=5`: Number of Chebyshev modes for interpolation if `use_chebyshev=true`.
- `cheb_param_strategy::Symbol=:global`: Strategy for Chebyshev parameter selection. Options are `:global` for using the spectrum's maximum k value to decide panelization and M for Chebyshev polynomials, then `:segment` if we want to adaptively choose parameters for each segment (good for large intervals with varying k density). Third is `:manual` where it will provide the user's initial kwarg `n_panels` and `M` for all segments without adaptation.
- `cheb_tol::Real=1e-13`: Tolerance for Chebyshev parameter tuning (if using Chebyshev).
- `max_iter::Int=20`: Maximum iterations for Chebyshev parameter tuning (if using Chebyshev).
- `sampling_points::Int=50_000`: Number of points to sample for Chebyshev parameter tuning (if using Chebyshev).
- `grading::Symbol=:uniform`: Grading strategy for Chebyshev panels, by default uniform, can be `:uniform` or `:geometric` (if using Chebyshev).
- `grow_panels::Real=1.5`: Growth factor for number of panels during Chebyshev parameter tuning (if using Chebyshev).
- `grow_M::Int=2`: Growth factor for degree of Chebyshev polynomials during Chebyshev parameter tuning (if using Chebyshev).
- `verbose_cheb_panelization::Bool=false`: Whether to print detailed information during Chebyshev panelization (if using Chebyshev).

# Returns
- `λs::Vector{T} or Vector{Complex{T}}`: Corrected eigenvalues (wavenumbers).
- `tensions::Vector{T}`: Corresponding tension values (error estimates).

# Notes
- The EBIM formulation solves the generalized eigenproblem
  `A(k₀) v = λ dA(k₀) v` and applies second-order corrections.
- Left/right eigenvectors are complex in general; Hermitian pairing is used internally.
- Matrix buffers are reused within each segment to avoid repeated allocations.
"""
function compute_spectrum_ebim(solver::EBIMSolver,billiard::Bi,k1::T,k2::T;dk::Function=(k->0.05*k^(-1/3)),tol=T(1e-4),use_lapack_raw::Bool=false,multithreaded_matrices::Bool=false,use_krylov::Bool=true,seg_reuse_frac::T=T(0.95),solve_info::Bool=true,use_chebyshev::Bool=false,n_panels::Int=15000,M::Int=5,cheb_param_strategy::Symbol=:global,cheb_tol::Real=1e-13,max_iter::Int=20,sampling_points::Int=50_000,grading::Symbol=:uniform,grow_panels::Real=1.5,grow_M::Int=2,verbose_cheb_panelization::Bool=false,return_imag_part::Bool=false) where {T<:Real,Bi<:AbsBilliard}
    ks=T[] # these are pts on the real axis (centers of EBIM windows)
    dks=T[] # these are the half-widths of the EBIM windows, which can be k-dependent
    k=k1
    while k<k2 # populate the k grid with variable step size dk(k)
        push!(ks,k)
        Δk=dk(k)
        push!(dks,Δk)
        k+=Δk
    end
    nevs=Int[] # number of eigenvalues to request from the solver at each k, based on Weyl's law with padding. Is it a crude estimate
    for i in eachindex(ks)
        k=ks[i]
        Δk=dks[i]
        push!(nevs,Int(ceil((billiard.area*k/(2*pi)-billiard.length/(4*pi))*Δk))+10)
    end
    K=return_imag_part ? Complex{T} : T
    isempty(ks) && return K[],T[]
    pts0=evaluate_points(solver,billiard,ks[1])
    println("compute_spectrum...")
    println("Total k points: $(length(ks))")
    N0=boundary_matrix_size(pts0)
    A0=Matrix{ComplexF64}(undef,N0,N0)
    dA0=Matrix{ComplexF64}(undef,N0,N0)
    ddA0=Matrix{ComplexF64}(undef,N0,N0)
    if solve_info
        #TODO Chebyshev pathway solve_INFO
        solve_INFO!(solver,A0,dA0,ddA0,pts0,ks[1],dks[1];use_lapack_raw=use_lapack_raw,multithreaded=multithreaded_matrices,use_krylov=use_krylov)
    end
    if use_chebyshev && cheb_param_strategy==:global
        kref=ks[end]
        pts_ref=evaluate_points(solver,billiard,kref)
        n_panels,M,_=chebyshev_params(solver,pts_ref,ComplexF64[kref];tol=cheb_tol,n_panels_init=n_panels,M_init=M,grading=grading,sampling_points=sampling_points,max_iter=max_iter,grow_panels=grow_panels,grow_M=grow_M,verbose=verbose_cheb_panelization)
    elseif use_chebyshev && cheb_param_strategy==:manual
        @info "Using manual Chebyshev parameters: n_panels=$n_panels, M=$M for all segments."
    end
    results=Vector{Tuple{Vector{K},Vector{T}}}(undef,length(ks))
    p=Progress(length(ks),1)
    seg_first=1
    while seg_first<=length(ks)
        seg_last=seg_first
        while seg_last<length(ks) && ks[seg_last+1]<=ks[seg_first]/seg_reuse_frac
            seg_last+=1
        end
        pts=seg_first==1 ? pts0 : evaluate_points(solver,billiard,ks[seg_last])
        N=boundary_matrix_size(pts)
        A=Matrix{ComplexF64}(undef,N,N)
        dA=Matrix{ComplexF64}(undef,N,N)
        ddA=Matrix{ComplexF64}(undef,N,N)
        if use_chebyshev
            segks=ks[seg_first:seg_last]
            if cheb_param_strategy==:segment
                kref=segks[end]
                np_init=n_panels
                M_init=M
                n_panels,M,_=chebyshev_params(solver,pts,ComplexF64[kref];tol=cheb_tol,n_panels_init=np_init,M_init=M_init,grading=grading,sampling_points=sampling_points,max_iter=max_iter,grow_panels=grow_panels,grow_M=grow_M,verbose=verbose_cheb_panelization)
            end
            cache=build_ebim_cheb_cache(solver,pts,segks;n_panels=n_panels,M=M,grading=grading)
            for (loc,i) in enumerate(seg_first:seg_last)
                λs,tens=solve!(solver,A,dA,ddA,pts,ks[i],dks[i],cache,loc;use_lapack_raw=use_lapack_raw,multithreaded=multithreaded_matrices,use_krylov=use_krylov,nev=nevs[i],return_imag_part=return_imag_part)
                results[i]=(λs,tens)
                next!(p)
            end
        else
            for i in seg_first:seg_last
                λs,tens=solve!(solver,A,dA,ddA,pts,ks[i],dks[i];use_lapack_raw=use_lapack_raw,multithreaded=multithreaded_matrices,use_krylov=use_krylov,nev=nevs[i],return_imag_part=return_imag_part)
                results[i]=(λs,tens)
                next!(p)
            end
        end
        seg_first=seg_last+1
    end
    λs_all=K[]
    tensions_all=T[]
    control=Bool[]
    for i in eachindex(ks)
        λs,tens=results[i]
        isempty(λs) && continue
        overlap_and_merge_ebim!(λs_all,tensions_all,λs,tens,control,ks[i]-dks[i],ks[i]+dks[i];tol=tol)
    end
    isempty(λs_all) && return K[],T[]
    keep=[k1<=real(λ)<=k2 for λ in λs_all] # since im part is a discretization error we should only check if Re part is inside the interval
    # weird hack? that can cause issues if we want to glue together 2 results from compute_spectrum_ebim on the edges of the interval, but for now we can live with it and if we want to do something more robust we can add a small buffer to k1 and k2 here
    λs_all=λs_all[keep]
    tensions_all=tensions_all[keep]
    return λs_all,tensions_all
end

"""
    solve_DEBUG_w_2nd_order_corrections(solver::ExpandedBoundaryIntegralMethod,pts,k;multithreaded::Bool=true)

A debug routine that solves the generalized eigenproblem `(A, dA)` at wavenumber `k`, then applies
both first- and second-order corrections to refine the approximate roots. Specifically,
it extracts λ from `A*x = λ dA*x`, then does:

  corr₁[i] = -λ[i]
  corr₂[i] = -0.5 * corr₁[i]^2 * ( (v_leftᵀ ddA v_right) / (v_leftᵀ dA v_right) )

Hence two sets of corrected wavenumbers: `k + corr₁` and `k + corr₁ + corr₂`. Tensions are `|corr₁|`
and `|corr₁ + corr₂|`.

# Arguments
- `solver::ExpandedBoundaryIntegralMethod`: The EBIM solver config.
- `pts`: Boundary geometry.
- `k`: Wavenumber for the eigenproblem.
- `multithreaded::Bool=true`: If the matrix construction should be multithreaded.

# Returns (all complex since we want to capture the imaginary part of the corrections as well, which can be useful for error estimation):
- `(λ_corrected_1, tens_1, λ_corrected_2, tens_2)`: 
   1. `λ_corrected_1 = k + corr₁` (1st-order),
   2. `tens_1 = corr₁`,
   3. `λ_corrected_2 = k + corr₁ + corr₂` (2nd-order),
   4. `tens_2 = corr₁ + corr₂`.
"""
function solve_DEBUG_w_2nd_order_corrections(solver::EBIMSolver,basis::Ba,pts,k;multithreaded::Bool=true) where {Ba<:AbstractHankelBasis}
    A,dA,ddA=construct_matrices(solver,basis,pts,k;multithreaded=multithreaded)
    λ,VR,VL=generalized_eigen_all(A,dA)
    valid_indices=.!isnan.(λ).&.!isinf.(λ)
    λ=λ[valid_indices]
    sort_order=sortperm(abs.(λ)) 
    λ=λ[sort_order]
    VR=VR[:,sort_order]
    VL=VL[:,sort_order]
    T=eltype(λ)
    corr_1=Vector{T}(undef,length(λ))
    corr_2=Vector{T}(undef,length(λ))
    for i in eachindex(λ)
        v_right=VR[:,i]
        v_left=VL[:,i]
        buf=similar(v_right)
        mul!(buf,ddA,v_right)
        numerator=dot(v_left,buf)
        mul!(buf,dA,v_right)
        denominator=dot(v_left,buf)
        corr_1[i]=-λ[i]
        corr_2[i]=-0.5*corr_1[i]^2*(numerator/denominator)
    end
    λ_corrected_1=k.+corr_1
    λ_corrected_2=λ_corrected_1.+corr_2
    tens_1=corr_1
    tens_2=corr_1.+corr_2
    return λ_corrected_1,tens_1,λ_corrected_2,tens_2
end

"""
    ebim_inv_diff(kvals::Vector{T}) where {T<:Number}

Computes the inverse of the differences between consecutive elements in `kvals`. This inverts the small differences between the ks very close to the correct eigenvalues and serves as a visual aid or potential criteria for finding missing levels.

# Arguments
- `kvals::Vector{T}`: A vector of values for which differences are calculated.

# Returns
- `Vector{T}`: The `kvals` vector excluding its last element.
- `Vector{T}`: The inverse of the differences between consecutive elements in `kvals`.
"""
function ebim_inv_diff(kvals::Vector{K},tens::Vector{C}) where {K<:Number,C<:Number}
    p=sortperm(real.(kvals))
    kvals=kvals[p]
    tens=tens[p]
    dr=diff(real.(kvals))
    invspacing=one(real(K))./dr
    return kvals[1:end-1],invspacing,tens[1:end-1]
end

"""
    visualize_ebim_sweep(solver::EBIMSolver,billiard::Bi,k1,k2;dk=(k)->0.05*k^(-1/3),multithreaded::Bool=false,multithreaded_ks::Bool=true,tension_cutoff=1.0)

Debugging and visualization tool for EBIM spectral searches over a wavenumber interval.

This routine performs a sweep over `k ∈ [k1,k2]`, solves the EBIM generalized
eigenproblem at each sample point, and extracts the smallest first- and second-order
corrections to identify nearby eigenvalues.

Unlike the production solver, this function keeps the full complex corrections,
which is crucial for diagnosing:
- discretization error,
- conditioning issues,
- spurious roots,
- convergence quality of the EBIM expansion.

For each `k`:
- the smallest-magnitude correction (first and second order),
- filtered by `|ε| < tension_cutoff`.

The output is then postprocessed by:
- sorting by `Re(k)`,
- computing inverse spacings:
  
      1 / Δ(Re(k))

which highlight eigenvalue clustering.

# Returns
The function returns:
    (kplot_1, invspacing_1, tens_1,
     kplot_2, invspacing_2, tens_2)
where:
- `kplot_1`, `kplot_2`:
    Complex corrected wavenumbers (first and second order),
    sorted by real part and trimmed for spacing analysis.
- `invspacing_1`, `invspacing_2`:
    Inverse spacings computed from `Re(k)`:
        invspacing[i] = 1 / (Re(k[i+1]) - Re(k[i]))
    Large values indicate clustering → likely eigenvalues.
- `tens_1`, `tens_2`:
    Complex corrections:
        tens_1 = ε₁
        tens_2 = ε₁ + ε₂
"""
function visualize_ebim_sweep(solver::EBIMSolver,billiard::Bi,k1::T,k2::T;dk=(k)->0.05*k^(-1/3),multithreaded::Bool=false,multithreaded_ks::Bool=true,tension_cutoff::Real=1.0) where {T<:Real,Bi<:AbsBilliard}
    ks=T[]
    k=k1
    while k<k2
        push!(ks,k)
        k+=dk(k)
    end
    push!(ks,k2)
    CT=Complex{T}
    ks_all_1=Vector{Union{CT,Missing}}(missing,length(ks))
    ks_all_2=Vector{Union{CT,Missing}}(missing,length(ks))
    tens_all_1=Vector{Union{CT,Missing}}(missing,length(ks))
    tens_all_2=Vector{Union{CT,Missing}}(missing,length(ks))
    all_pts=Vector{typeof(evaluate_points(solver,billiard,ks[1]))}(undef,length(ks))
    @showprogress desc="Calculating boundary points..." for i in eachindex(ks)
        all_pts[i]=evaluate_points(solver,billiard,ks[i])
    end
    @info "EBIM smallest complex corrections..."
    pbar=Progress(length(ks),1)
    @use_threads multithreading=multithreaded_ks for i in eachindex(ks)
        k1c,t1,k2c,t2=solve_DEBUG_w_2nd_order_corrections(
            solver,
            AbstractHankelBasis(),
            all_pts[i],
            ks[i];
            multithreaded=multithreaded,
        )
        idx1=findmin(abs.(t1))[2]
        idx2=findmin(abs.(t2))[2]
        if abs(t1[idx1])<tension_cutoff
            ks_all_1[i]=k1c[idx1]
            tens_all_1[i]=t1[idx1]
        end
        if abs(t2[idx2])<tension_cutoff
            ks_all_2[i]=k2c[idx2]
            tens_all_2[i]=t2[idx2]
        end
        next!(pbar)
    end
    ks_all_1=collect(skipmissing(ks_all_1))
    tens_all_1=collect(skipmissing(tens_all_1))
    ks_all_2=collect(skipmissing(ks_all_2))
    tens_all_2=collect(skipmissing(tens_all_2))
    kplot_1,invspacing_1,tplot_1=ebim_inv_diff(ks_all_1,tens_all_1)
    kplot_2,invspacing_2,tplot_2=ebim_inv_diff(ks_all_2,tens_all_2)
    keep1=findall(isfinite,invspacing_1)
    keep2=findall(isfinite,invspacing_2)
    return kplot_1[keep1],invspacing_1[keep1],tplot_1[keep1],kplot_2[keep2],invspacing_2[keep2],tplot_2[keep2]
end