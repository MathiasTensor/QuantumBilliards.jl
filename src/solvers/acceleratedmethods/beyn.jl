#=
#############################################
########### BEYN CONTOUR METHOD #############
#############################################

MAIN REFERENCE: Beyn, Wolf-Jurgen, An integral method for solving nonlinear eigenvalue problems, 2018

- A Beyn-type contour-integral method for nonlinear eigenproblems T(k)φ=0 arising from BIM.
- On each disk Γ: center k0, radius R, we build
A0 = (1/2πi)∮ T(z)^{-1}V dz,  A1 = (1/2πi)∮ z T(z)^{-1}V dz
then project with the rank-revealing SVD of A0 to a small dense B, and solve eigen!(B).
- Returned eigenpairs are filtered: (i) |k−k0|≤R and (ii) residual ‖T(k)φ‖ below a tolerance.

Workflow - short

- Call compute_spectrum with the BIM solver and try first with default kwargs. If a lot of warnings about a potential eigenvalue with tension just a bit below tolerance is printed then increase nq until they disappear.

Workflow - long:

1) Choose geometry, basis, and symmetry:
- Build billiard, basis, and (optional) symmetry for your problem.
- Construct the BIM solver: solver=BoundaryIntegralMethod(…, billiard, symmetry=…).

2) Plan windows (disks) that each contain ≈m levels:
- Call intervals=plan_weyl_windows(billiard,k1,k2; m, Rmax, fundamental).
- Convert to disks: (k0,R)=beyn_disks_from_windows(intervals).
- R is automatically ≤Rmax, so each disk’s radius is controlled.

3) Collocate once per window:
- For each k0[i], call pts[i]=evaluate_points(solver,billiard,real(k0[i])).

4) Pick numerical contour nodes (geometry dependent):
- nq : number of contour nodes (trapezoid on circle). Start at 30–60; increase if residuals are large.
- r : probe rank (≥ expected #levels per disk). Start at m+10…m+20.
- svd_tol: SVD cutoff on Σ(A0). 1e-12…1e-14 is safe; inspect Σ tail in solve_INFO.
- res_tol: residual tolerance. 1e-10…1e-12 typical; tighten if needed.

5) Sanity-check one disk before the sweep:
- Run solve_INFO on the last (or a representative) disk with nq-10, nq, nq+10.
- Inspect:
* singular values Σ(A0) and detected rank,
* kept/dropped counts and residuals,
* whether increasing nq reduces residuals by orders of magnitude.
- If residuals are too big or spurious roots appear, increase nq (and, if m is larger, also r).

6) Solve all disks:
- Use solve_vect or the high-level compute_spectrum to get ks and densities Φ.

7) Compute “tensions” (post-process, scale-invariant):
- Use compute_normalized_tensions to get  t_i = ‖T(k_i)φ_i‖ / (‖T(k_i)‖ · ‖φ_i‖)  (default 1-norm).

Practical guidance

- If you increase m (more levels per disk), increase both r (≥ m by a margin) and usually nq.
- Non-analytic boundaries converge slower with nq; expect to use higher nq and/or smaller R.
- Use solve_INFO logs to diagnose: Σ(A0) gaps, rank rk, counts kept/dropped, and residual histogram.
- Typical robust defaults: m≈8–12, Rmax≈0.5, nq≈64–96, r≈m+15, svd_tol≈1e-13, res_tol≈1e-10.
- For very high k or intricate geometries, start conservative (smaller R, larger nq) and relax if safe.

Added support for BoundaryPointsCFIE to handle domains with holes (e.g., annulus) where the CFIE_kress formulation is needed.
MO 29/3/26
=#

#################
#### HELPERS ####
#################

# Internal helpers to compute the area-based estimate of the counting function N(k) for a given billiard, optionally using the fundamental domain geometry. This is enouhg for the upper bound due to annoying billiard struct fundamental_boundary missing for complex geometries with no bases.
@inline function area_count_estimate(k::T,billiard::Bi;fundamental::Bool=true) where {T<:Real,Bi<:AbsBilliard}
    A=fundamental ? billiard.area_fundamental : billiard.area
    return A*k^2/(4*pi)
end
@inline function delta_area_count_estimate(billiard::Bi,k::T,Δk::T;fundamental::Bool=true) where {T<:Real,Bi<:AbsBilliard}
    A=fundamental ? billiard.area_fundamental : billiard.area
    return A*((k+Δk)^2-k^2)/(4*pi)
end
@inline function dos_area_estimate(k::T,billiard::Bi;fundamental::Bool=true) where {T<:Real,Bi<:AbsBilliard}
    A=fundamental ? billiard.area_fundamental : billiard.area
    return A*k/(2*pi)
end

# ΔN(k,Δk) = N(k+Δk) - N(k)
# Uses Weyl’s law as a fast estimator of eigenvalue count N(k).
# Returns the estimated number of levels in the interval [k, k+Δk].
function delta_weyl(billiard::Bi,k::T,Δk::T;fundamental::Bool=true) where {T<:Real,Bi<:AbsBilliard}
    # Use Weyl’s law as a fast estimator of counting function N(k); difference gives # levels in [k,k+Δk]
    delta_area_count_estimate(billiard,k,Δk;fundamental=fundamental)
end

# initial_step_from_dos
# Compute an initial guess for Δk from the density of states ρ(k).
# Formula: Δk₀ ≈ m / ρ(k). 
# Ensures Δk₀ is not too small by flooring with min_step.
function initial_step_from_dos(billiard::Bi,k::T,m::Int;fundamental::Bool=true,min_step::Real=1e-6) where {T<:Real,Bi<:AbsBilliard}
    ρ=max(dos_area_estimate(k,billiard;fundamental=fundamental),1e-12) # estimate DOS ρ(k); clamp to avoid division by tiny numbers
    max(m/ρ,min_step) # target m levels -> Δk≈m/ρ; enforce a minimum to avoid Δk≈0 in sparse regions
end

# grow_upper_bound
# Start from Δk₀ and geometrically increase Δk until ΔN(k,Δk) ≥ m.
# Capped by the remaining interval length. 
# Returns (Δk, success_flag), where success_flag -> true if m levels reached.
function grow_upper_bound(billiard::Bi,k::T,m,Δk0::T,remaining::T;fundamental::Bool=true,max_grows::Int=60) where {T<:Real,Bi<:AbsBilliard}
    Δk=min(remaining,max(Δk0,eps(k))) # start from Δk0 but: (i) not below machine eps, (ii) not above remaining span
    for _ in 1:max_grows
        if delta_weyl(billiard,k,Δk;fundamental=fundamental)≥m-eps();return Δk,true end # stop once we bracket ≥m levels
        if Δk≥0.999999*remaining;return remaining,false end # cannot grow further—signal that we hit the cap
        Δk=min(remaining,2Δk) # after cap whether we met target
    end
    return Δk,delta_weyl(billiard,k,Δk;fundamental=fundamental)≥m
end

# bisect_for_delta_k
# Given bracket [lo,hi] where ΔN(lo) < m ≤ ΔN(hi), perform bisection to find Δk ≈ m levels.
# Stops when tolerance in level count is satisfied or bracket is sufficiently small.
# Returns the Δk that yields ΔN(k,Δk) ≈ m.
function bisect_for_delta_k(billiard::Bi,k::T,m,lo::T,hi::T;fundamental::Bool=true,tol_levels=0.1,maxit::Int=50) where {T<:Real,Bi<:AbsBilliard}
    @assert hi>lo
    Nlo=delta_weyl(billiard,k,lo;fundamental=fundamental) # evaluate count at lo
    if abs(Nlo-m)≤tol_levels;return lo end # early accept if already within tolerance
    Nhi=delta_weyl(billiard,k,hi;fundamental=fundamental) # evaluate count at hi
    if abs(Nhi-m)≤tol_levels;return hi end #  early accept at hi
    @assert Nhi≥m
    for _ in 1:maxit
        mid=0.5*(lo+hi)  # bisection on Δk
        Nmid=delta_weyl(billiard,k,mid;fundamental=fundamental)  # count at midpoint
        if abs(Nmid-m)≤tol_levels;return mid end # tolerance satisfied
        if Nmid<m;lo=mid else hi=mid end # shrink bracket toward target
        if hi-lo≤max(1e-12,1e-9*max(1.0,k));return 0.5*(lo+hi) end # stop when bracket is tiny (abs or relative to k)
    end
    return 0.5*(lo+hi) # fallback: return midpoint of final bracket
end

# plan_weyl_windows
# Cover interval [k1,k2] with windows each containing ≈ m eigenvalues by Weyl estimate.
# Each window length is capped at ≤ 2*Rmax (so that disk radius R ≤ Rmax).
# Uses: initial_step_from_dos, grow_upper_bound, bisect_for_delta_k.
# Returns a list of (kL,kR) windows.
function plan_weyl_windows(billiard::Bi,k1::T,k2::T; m::Int=10, Rmax::Real=1.0,fundamental::Bool=true, tol_levels=0.1, maxit::Int=50) where {T<:Real,Bi<:AbsBilliard}
    iv=Vector{Tuple{T,T}}() # accumulator of (kL,kR) windows
    k=k1 # left cursor that advances to cover [k1,k2]
    while k<k2-eps() # loop until we reach the right end
        rem_raw=k2-k # remaining span
        rem=rem_raw>2*Rmax ? T(2*Rmax) : rem_raw # enforce max window length 2 * Rmax (so disk radius R≤Rmax)
        if delta_weyl(billiard,k,rem;fundamental=fundamental)≤m+tol_levels && rem==rem_raw
            push!(iv,(k,k2)); break # if tail holds ≤m levels and fits entirely, close with a single window
        end
        Δk0=initial_step_from_dos(billiard,k,m;fundamental=fundamental) # DOS-based initial guess for ~m levels
        hi,ok=grow_upper_bound(billiard,k,m,Δk0,rem;fundamental=fundamental) # geometrically grow Δk until we bracket ≥m or hit cap
        if !ok
            push!(iv,(k,k+hi)); k+=hi; continue # couldn’t hit ≥m due to length cap: accept best effort and advance
        end
        Δk=bisect_for_delta_k(billiard,k,m,0.0,hi;fundamental=fundamental,tol_levels=tol_levels,maxit=maxit) # refine to ≈m levels
        kR=min(k+Δk,k+rem) # still respect max window length
        push!(iv,(k,kR)); k=kR # record window and advance cursor
    end
    return iv # list of windows covering [k1,k2]
end

# beyn_disks_from_windows
# Convert a list of windows [kL,kR] into Beyn method disks:
# center k0 -> midpoint of window, radius R -> half the window length.
# Guarantees R ≤ Rmax because windows were capped.
# Returns (k0,R) arrays for use in contour integration.
function beyn_disks_from_windows(iv::Vector{Tuple{T,T}}) where {T<:Real}
    k0=Vector{Complex{T}}(undef,length(iv)); R=Vector{T}(undef,length(iv)) # disk centers (complex, imag=0) and radii
    @inbounds for (i,(kL,kR)) in pairs(iv) # center = midpoint of window -> matches Beyn circle center
        k0[i]=complex(0.5*(kL+kR)); R[i]=0.5*(kR-kL) # radius = half window length → guarantees |window| ≤ 2Rmax -> R≤Rmax
    end
    return k0,R
end

#################################
#### HELPERS FOR BEYN METHOD ####
#################################

# Constructs the buffer matrices for Beyn's method
#
# Inputs:
#   - T: Type of the elements (Real or Complex)
#   - N: Size of the matrices
#   - r: Number of rows in the buffer matrices
#   - rng: Random number generator
#
# Outputs:
#   - V::Matrix{Complex{T}}: Buffer matrix for the right-hand side
#   - X::Matrix{Complex{T}}: Buffer matrix for the solution
#   - A0::Matrix{Complex{T}}: Buffer matrix for the first moment
#   - A1::Matrix{Complex{T}}: Buffer matrix for the second moment
function beyn_buffer_matrices(::Type{T},N::Int64,r::Int64,rng::G) where {T<:Real,G}
    V=randn(rng,Complex{T},N,r) # best leave as Complex even for Real problems to avoid issues in ldiv!
    X=similar(V)
    A0=zeros(Complex{T},N,r)
    A1=zeros(Complex{T},N,r)
    return V,X,A0,A1
end

# Applies the projection onto the symmetry subspace defined by `symmetry` to the buffer matrix `V` in the case of CFIE_kress, where the boundary points are represented by `BoundaryPointsCFIE`.
# This is necessary because CFIE_kress requires working with the full domain, and we need to ensure that the buffer matrix respects the symmetry of the problem.
#
# Inputs:
#   - solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners,CFIE_alpert}: The solver instance which may contain symmetry information.
#   - pts::Vector{BoundaryPointsCFIE{T}}: The boundary points for the CFIE_kress / alpert method, which may consist of multiple components (e.g., outer boundary and holes).
#   - V::Matrix{Complex{T}}: The buffer matrix to be projected.
#   - W::Matrix{Complex{T}}: Buffer matrix.
#
# Output:
#   - The function modifies `V` in-place to contain the projected values.
function _CFIE_project_V_subspace!(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners,CFIE_alpert},pts::Vector{BoundaryPointsCFIE{T}},V::AbstractMatrix{Complex{T}},W::AbstractMatrix{Complex{T}}) where {T<:Real}
    isnothing(solver.symmetry) && return V
    maps=build_symmetry_maps(pts,solver.symmetry)
    apply_projection!(V,W,maps,solver.symmetry)
    return V
end
# Overload for the case where we have a single BoundaryPointsCFIE instead of a vector, which can happen in simpler geometries without holes. This ensures that the projection is applied correctly even when the input is not a vector of boundary point sets.
function _CFIE_project_V_subspace!(solver::Union{DLP_kress,DLP_kress_global_corners},pts::BoundaryPointsCFIE{T},V::AbstractMatrix{Complex{T}},W::AbstractMatrix{Complex{T}}) where {T<:Real}
    isnothing(solver.symmetry) && return V
    maps=build_symmetry_maps(pts,solver.symmetry)
    apply_projection!(V,W,maps,solver.symmetry)
    return V
end
# this one has desymmetrization already built in in matrix construction stage, so dont do anything here
function _CFIE_project_V_subspace!(solver::Union{BoundaryIntegralMethod},pts::BoundaryPoints{T},V::AbstractMatrix{Complex{T}},W::AbstractMatrix{Complex{T}}) where {T<:Real}
    return V
end

# construct the B matrix as described in Beyn's paper using the Chebyshev Hankel evaluations to circumvent allocations for complex argument Hankel functions. For high k this is unavoidable.
#
# Inputs:
#   - solver::Union{BoundaryIntegralMethod,CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners,CFIE_alpert,DLP_kress,DLP_kress_global_corners}: The solver instance which defines the type of boundary integral formulation and may contain symmetry information.
#   - pts::Union{BoundaryPoints{T},Vector{BoundaryPointsCFIE{T}}}: The boundary points for both the standard domain with outer boundary (BoundaryPoints) and the CFIE_kress or CFIE_alpert formulation with inner and outer boundaries (BoundaryPointsCFIE)
#   - N::Int: Size of the Fredholm matrices
#   - k0::Complex{T}: Center of the contour
#   - R::T: Radius of the contour
#   - nq::Int: Number of quadrature points on the contour (if analytic boundary one can use less due to spectral convergence)
#   - r::Int: number of expected eigenvalues inside + some padding to their number
#   - svd_tol::Real: Tolerance for the SVD truncation
#   - rng::AbstractRNG: Random number generator
#   - use_chebyshev::Bool: Whether to use the Chebyshev Hankel evaluation or the standard one (for low k one can use no interpolations)
#   - n_panels::Int: Number of Chebyshev panels to use (try 2000 for k≈3000 and rmax≈4.0)
#   - M::Int: Degree of Chebyshev polynomials to use (try 200 for k≈3000 and rmax≈4.0)
#   - info::Bool: Whether to print info messages
#   - multithreaded::Bool: Whether to use multithreading in the kernel matrix assembly
#
# Notes:
#   1) Forms A0 = (1/2πi)∮ T(z)^{-1} V dz and A1 = (1/2πi)∮ z T(z)^{-1} V dz via LU solves.
#   2) Rank rk determined by Σ[i] ≥ svd_tol (strict absolute threshold).
#   3) If rk == 0, return empty matrices to signal “no roots in window”.
function construct_B_matrix(solver::Union{BoundaryIntegralMethod,CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners,CFIE_alpert,DLP_kress,DLP_kress_global_corners},pts::Union{BoundaryPoints{T},Vector{BoundaryPointsCFIE{T}}},N::Int,k0::Complex{T},R::T;nq::Int=64,r::Int=48,svd_tol=1e-14,rng=MersenneTwister(0),use_chebyshev=true,n_panels=15000,M=5,info::Bool=false,multithreaded::Bool=true) where {T<:Real}
    θ=range(zero(T),TWO_PI;length=nq+1)[1:end-1] # remove last point
    ej=cis.(θ) # unit circle points
    zj=k0.+R.*ej # contour points
    wj=(R/nq).*ej # contour weights
    #TODO Make the Fredholm matrices working buffers from outside to prevent large allocations in a loop (only for RAM critical applications since this is a small part of the actual execution time)
    Tbufs1=[zeros(Complex{T},N,N) for _ in 1:nq] 
    construct_boundary_matrices!(Tbufs1,solver,pts,zj;multithreaded=multithreaded,use_chebyshev=use_chebyshev,n_panels=n_panels,M=M,timeit=info) # construct the T(zj) matrices for each contour point zj.
    # Allocate the buffers for the Beyn method. These are used in the matrix construction and then in the contour integrations to avoid repeated allocations. The matrices are sized according to the expected number of eigenvalues r and the size of the Fredholm matrices N.
    V,X,A0,A1=beyn_buffer_matrices(T,N,r,rng)
    if solver isa CFIE_kress
        W=similar(V)
        _CFIE_project_V_subspace!(solver,pts,V,W) # for CFIE_kress we need to project the random V onto the symmetry subspace to ensure it is in the correct function space for the problem. For standard BIM this is not needed since we are already working with the outer boundary points which are the relevant ones for the eigenvalue problem.
    end
    # Now perform the Beyn contour integrations to form A0 and A1. To do this we need to solve T(zj) X = V for each zj and accumulate A0 += wj[j] * X, A1 += wj[j] * zj[j] * X. So as the first step we LU factor all T(zj) matrices to get the Fj factors which are used for ldiv! to efficiently solve the systems.
    @blas_multi MAX_BLAS_THREADS F1=lu!(Tbufs1[1];check=false) # just to get the type
    Fs=Vector{typeof(F1)}(undef,nq)
    Fs[1]=F1
    @benchit timeit=info "LU factorization" begin
        @blas_multi_then_1 MAX_BLAS_THREADS @inbounds for j in 2:nq # LU factor all T(zj) matrices
            Fs[j]=lu!(Tbufs1[j];check=false)
        end
    end
    xv=reshape(X,:);a0v=reshape(A0,:);a1v=reshape(A1,:) # vector views for BLAS.axpy! operations, to avoid allocations in the loop via reshaping the matrices each time in the loop
    @benchit timeit=info "Contour integration - ldiv! + axpy!" begin
        @blas_multi_then_1 MAX_BLAS_THREADS @inbounds for j in eachindex(zj)
            ldiv!(X,Fs[j],V) # make efficient inverse
            BLAS.axpy!(wj[j],xv,a0v) # A0 += wj[j] * X
            BLAS.axpy!(wj[j]*zj[j],xv,a1v) # A1 += wj[j] * zj[j] * X
        end
    end
    @blas_multi_then_1 MAX_BLAS_THREADS U,Σ,W=svd!(A0;full=false) # thin SVD of A0, revealing rank. The singular values > svd_tol correspond to eigenvalues. If all sv > svd_tol then maybe increase r (expected eigenvalue count) or reduce R (contour around k0), but if increasing r careful with nq. Check ref. section 3 eq. 22
    rk=count(>=(svd_tol),Σ) # filter out those that correspond to actual eigenvalues
    if rk==0 # if nothing found early return
        return Matrix{Complex{T}}(undef,N,0),Matrix{Complex{T}}(undef,N,0)
    end
    if rk==r # increase the eigvals dimension for A0 and reasonably adjust svd_tol if needed!
        r_tmp=r+r
        while r_tmp<N # do again the ldiv + axpy accumulation with larger r until some sv < svd_tol. This does not require another Fredholm matrix construction since the same T(zj) can be used for larger r.
            V,X,A0,A1=beyn_buffer_matrices(T,N,r_tmp,rng)
            xv=reshape(X,:);a0v=reshape(A0,:);a1v=reshape(A1,:)
            @blas_multi_then_1 MAX_BLAS_THREADS @inbounds for j in eachindex(zj)  
                ldiv!(X,Fs[j],V)
                BLAS.axpy!(wj[j],xv,a0v)
                BLAS.axpy!(wj[j]*zj[j],xv,a1v)
            end
            U,Σ,W=svd!(A0;full=false)
            rk=count(>=(svd_tol),Σ)
            rk<r_tmp && break
            r_tmp+=r
            r_tmp>N && throw(ArgumentError("r > N is impossible: requested r=$(r_tmp), N=$(N)"));break
        end
    end
    Uk=@view U[:,1:rk] # take the relevant ones corresponding to eigenvalues as in Integral algorithm 1 on p14 of ref
    Wk=@view W[:,1:rk] # take the relevant ones corresponding to eigenvalues as in Integral algorithm 1 on p14 of ref
    Σk=@view Σ[1:rk] # take the relevant ones corresponding to eigenvalues as in Integral algorithm 1 on p14 of ref
    # form B = adjoint(U) * A1 * W * Σ^{-1} as in the reference, p14, integral algorithm 1
    tmp=Matrix{Complex{T}}(undef,N,rk)
    @blas_multi_then_1 MAX_BLAS_THREADS mul!(tmp,A1,Wk)  # tmp := A1 * Wk, not weighted by inverse diagonal Σk
    @inbounds @simd for j in 1:rk # right-divide by diagonal Σk
        @views tmp[:,j]./=Σk[j]
    end
    B=Matrix{Complex{T}}(undef,rk,rk)
    @blas_multi_then_1 MAX_BLAS_THREADS mul!(B,adjoint(Uk),tmp) # B := Uk'*tmp, the final step
    return B,Uk
end

# Performs one contour integration of the Beyn algorithm for a
# single circular window centered at k with radius dk.
# Constructs the B matrix via contour integration (using Chebyshev
# Hankel acceleration if enabled), computes its eigenvalues λ and
# eigenvectors Y, and returns both the spectral data and internal
# matrices needed for residual testing.
#
# Inputs:
#   - solver::Union{BoundaryIntegralMethod,CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners,CFIE_alpert,DLP_kress,DLP_kress_global_corners} : The solver instance which defines the type of boundary integral formulation and may contain symmetry information.
#   - basis::Ba                      : Hankel basis type
#   - pts::Union{BoundaryPoints{T},Vector{BoundaryPointsCFIE{T}}} : Boundary discretization
#   - k::Complex{T}                  : Center of contour
#   - dk::T                          : Radius of contour
#
# Keyword options:
#   - nq::Int          : # of quadrature nodes on contour (≥15 recommended)
#   - r::Int           : # of random probe vectors (≥ expected # of roots)
#   - svd_tol::Real    : SVD truncation tolerance for rank detection
#   - res_tol::Real    : residual tolerance (not used directly here)
#   - rng              : random generator
#   - use_chebyshev    : use Chebyshev Hankel evaluation (fast at large k)
#   - n_panels, M      : Chebyshev grid parameters
#   - multithreaded    : whether to use multithreading in kernel assembly
#   - auto_discard_spurious : unused - legacy
#   - info::Bool       : whether to print time execution messages during execution
#
# Outputs:
#   - λ::Vector{Complex{T}}  : eigenvalues inside the contour
#   - Uk::Matrix{Complex{T}} : left singular vectors (A0 basis)
#   - Y::Matrix{Complex{T}}  : eigenvectors of B (small matrix)
#   - k::Complex{T}          : center of window
#   - dk::T                  : radius of window
#   - pts::Union{BoundaryPoints{T},Vector{BoundaryPointsCFIE{T}}} : geometry of this window
#
# Notes:
#   - This function does not check residuals or remove spurious λ.
#     Use `residual_and_norm_select` afterwards for filtering.
function solve_vect(solver::Union{BoundaryIntegralMethod,CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners,CFIE_alpert,DLP_kress,DLP_kress_global_corners},basis::Ba,pts::Union{BoundaryPoints{T},Vector{BoundaryPointsCFIE{T}}},k::Complex{T},dk::T;multithreaded::Bool=true,nq::Int=32,r::Int=48,svd_tol::Real=1e-14,res_tol::Real=1e-8,rng=MersenneTwister(0),auto_discard_spurious::Bool=true,use_chebyshev::Bool=true,n_panels=15000,M=5,info::Bool=false) where {Ba<:AbstractHankelBasis} where {T<:Real}
    N=boundary_matrix_size(pts) # get the size of the boundary matrix based on the type of pts (BoundaryPoints or Vector{BoundaryPointsCFIE})
    B,Uk=construct_B_matrix(solver,pts,N,k,dk,nq=nq,r=r,svd_tol=svd_tol,rng=rng,use_chebyshev=use_chebyshev,n_panels=n_panels,M=M,multithreaded=multithreaded,info=info) # here is where the core of the algorithm is found. Constructs B from step 5 in ref p.14
    if isempty(B) # rk==0
        @info "no_roots_in_window" k0=k R=dk nq=nq svd_tol=svd_tol
        return Complex{T}[],Uk,Matrix{Complex{T}}(undef,0,0),k,dk,pts
    end
    @blas_multi_then_1 MAX_BLAS_THREADS λ,Y=eigen!(B) # small dense eigendecomposition to get eigenvalues λ are the eigenvalues and v(λ) are the eigenvectors
    # Now form only relevant cols of Φ = U * Y since A0 = U Σ W*, we have A0 * W Σ^{-1} Y = U Y. Each column is now an eigenvector of of T(λ)v(λ) = 0. This is the second layer potential boundary operator now!
    #println("Eigenvalues found in window k0=$(k), R=$(dk): ",λ)
    return λ,Uk,Y,k,dk,pts
end

# Lightweight version of `solve_vect` returning only eigenvalues λ.
# Intended for diagnostic or quick scans, not for production use.
#
# WARNING:
#  - Does not compute residuals or discard spurious eigenvalues.
#  - Always verify λ with `residual_and_norm_select` before using
#     them in statistics or physical interpretation.
# -------------------------------------------------------------

function solve(solver::Union{BoundaryIntegralMethod,CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners,CFIE_alpert,DLP_kress,DLP_kress_global_corners},basis::Ba,pts::Union{BoundaryPoints{T},Vector{BoundaryPointsCFIE{T}}},k::Complex{T},dk::T;multithreaded::Bool=true,nq::Int=32,r::Int=48,svd_tol::Real=1e-14,res_tol::Real=1e-8,rng=MersenneTwister(0),auto_discard_spurious::Bool=true,use_chebyshev::Bool=true,n_panels::Int=15000,M::Int=5,info::Bool=false) where {Ba<:AbstractHankelBasis} where {T<:Real}
    N=boundary_matrix_size(pts) # get the size of the boundary matrix based on the type of pts (BoundaryPoints or Vector{BoundaryPointsCFIE})
    B,Uk=construct_B_matrix(solver,pts,N,k,dk,nq=nq,r=r,svd_tol=svd_tol,rng=rng,use_chebyshev=use_chebyshev,n_panels=n_panels,M=M,multithreaded=multithreaded,info=info) # here is where the core of the algorithm is found. Constructs B from step 5 in ref p.14
    if isempty(B) # rk==0
        @info "no_roots_in_window" k0=k R=dk nq=nq svd_tol=svd_tol
        return Complex{T}[]
    end
    @blas_multi_then_1 MAX_BLAS_THREADS λ,Y=eigen!(B) # small dense eigendecomposition to get eigenvalues λ are the eigenvalues and v(λ) are the eigenvectors
    # Now form only relevant cols of Φ = U * Y since A0 = U Σ W*, we have A0 * W Σ^{-1} Y = U Y. Each column is now an eigenvector of of T(λ)v(λ) = 0. This is the second layer potential boundary operator now!
    #println("Eigenvalues found in window k0=$(k), R=$(dk): ",λ)
    return λ
end

# One Beyn solve with more information (and slower) than solve_vect. Provides useful information about residuals and automatic spurious eigenvalue discarding and sheds a light on the internal workings of the algorithm and if we are using a large enough nq for the contour integration.
# NOTE: Does not use chebyshev hankel evaluations for now, only standard hankel evaluations for clarity.
#
# Inputs:
#   - solver::Union{BoundaryIntegralMethod,CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners,CFIE_alpert,DLP_kress,DLP_kress_global_corners}: The solver instance which defines the type of boundary integral formulation and may contain symmetry information.
#   - basis::Ba: The basis type
#   - pts::Union{BoundaryPoints{T},Vector{BoundaryPointsCFIE{T}}}: The boundary points (standard with just outer boundary) or CFIE_kress/CFIE_alpert type (w/ inner holes)
#   - k0::Complex{T}: Center of the contour
#   - R::T: Radius of the contour
#   - multithreaded::Bool: Whether to use multithreading
#   - nq::Int: Number of quadrature points on the contour
#   - r::Int: number of expected eigenvalues inside + some padding to their number
#   - svd_tol::Real: Tolerance for the SVD truncation
#   - res_tol::Real: Tolerance for the residual ||A(k)v(k)||
#   - rng::AbstractRNG: Random number generator
#   - use_adaptive_svd_tol::Bool: Whether to use adaptive svd tolerance based on maximum singular value
#   - auto_discard_spurious::Bool: Whether to automatically discard spurious eigenvalues based on residuals
#   - use_chebyshev::Bool: Whether to use chebyshev hankel evaluations 
#   - n_panels::Int: Number of chebyshev panels to use
#   - M::Int: Degree of chebyshev polynomials to use
#
# Outputs:
#   - λ::Vector{Complex{T}}: The eigenvalues found inside the contour
#   - Phi::Matrix{Complex{T}}: The eigenvectors corresponding to the eigenvalues
#   - tens::Vector{T}: The residuals ||A(k)v(k)||
function solve_INFO(solver::Union{BoundaryIntegralMethod,CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners,CFIE_alpert,DLP_kress,DLP_kress_global_corners},basis::Ba,pts::Union{BoundaryPoints{T},Vector{BoundaryPointsCFIE{T}}},k0::Complex{T},R::T;multithreaded::Bool=true,nq::Int=48,r::Int=48,svd_tol::Real=1e-10,res_tol::Real=1e-10,rng=MersenneTwister(0),use_adaptive_svd_tol=false,auto_discard_spurious=false,use_chebyshev=true,n_panels=15000,M=5) where {Ba<:AbstractHankelBasis,T<:Real}
    N=boundary_matrix_size(pts) # get the size of the boundary matrix based on the type of pts (BoundaryPoints or Vector{BoundaryPointsCFIE})
    θ=range(zero(T),TWO_PI;length=nq+1);θ=θ[1:end-1];ej=cis.(θ);zj=k0.+R.*ej;wj=(R/nq).*ej # contour points and weights
    V,X,A0,A1=beyn_buffer_matrices(T,N,r,rng)
    @info "beyn:start" k0=k0 R=R nq=nq N=N r=r
    Tbufs1=[zeros(Complex{T},N,N) for _ in 1:nq] 
    construct_boundary_matrices!(Tbufs1,solver,pts,zj;multithreaded=multithreaded,use_chebyshev=use_chebyshev,n_panels=n_panels,M=M,timeit=true) # construct the T(zj) matrices for each contour point zj.
    @blas_multi MAX_BLAS_THREADS F1=lu!(Tbufs1[1];check=false) # just to get the type
    Fs=Vector{typeof(F1)}(undef,nq)
    Fs[1]=F1
    @blas_multi_then_1 MAX_BLAS_THREADS @inbounds begin
        @showprogress desc="lu!" for j in 2:nq # LU factor all T(zj) matrices
            Fs[j]=lu!(Tbufs1[j];check=false)
        end
    end
    xv=reshape(X,:);a0v=reshape(A0,:);a1v=reshape(A1,:) # vector views for BLAS.axpy! operations, to avoid allocations in the loop via reshaping the matrices each time in the loop
    @blas_multi_then_1 MAX_BLAS_THREADS @inbounds begin
        @showprogress desc="ldiv! + axpy!" for j in eachindex(zj)
            ldiv!(X,Fs[j],V) # make efficient inverse
            BLAS.axpy!(wj[j],xv,a0v) # A0 += wj[j] * X
            BLAS.axpy!(wj[j]*zj[j],xv,a1v) # A1 += wj[j] * zj[j] * X
        end
    end
    @time "SVD" @blas_multi_then_1 MAX_BLAS_THREADS U,Σ,W=svd!(A0;full=false)
    println("Singular values (<1e-10 tail inspection): ",Σ)
    rk=0
    svd_tol=use_adaptive_svd_tol ? maximum(Σ)*1e-15 : svd_tol
    @inbounds for i in eachindex(Σ)
        if Σ[i]≥svd_tol
            rk+=1
        else
            break
        end
    end
    rk==r && @warn "All singular values are above svd_tol = $(svd_tol), r = $(r) needs to be increased" # in the actual implementation where B matrix is constructed this will increase r by a fixed amount and do the procedure again until we have some singular values under tolerance!
    rk==0 && return Complex{T}[],Matrix{Complex{T}}(undef,N,0),T[]
    Uk=@view U[:,1:rk]
    Wk=@view W[:,1:rk]
    Σk=@view Σ[1:rk]
    tmp=Matrix{Complex{T}}(undef,N,rk)
    @blas_multi MAX_BLAS_THREADS mul!(tmp,A1,Wk)
    @inbounds for j in 1:rk
        @views tmp[:,j]./=Σk[j]
    end
    B=Matrix{Complex{T}}(undef,rk,rk)
    @blas_multi MAX_BLAS_THREADS mul!(B,adjoint(Uk),tmp)
    @time "eigen" @blas_multi_then_1 MAX_BLAS_THREADS ev=eigen!(B)
    λ=ev.values;Y=ev.vectors;Phi=Uk*Y
    keep=trues(length(λ))
    tens=Vector{T}()
    ybuf=Vector{Complex{T}}(undef,size(Phi,1))
    dropped_out=0
    dropped_res=0
    res_keep=T[]
    Tbuf_check=[zeros(Complex{T},N,N)]
    begin 
        @inbounds for j in eachindex(λ)
            d=abs(λ[j]-k0)
            if d>R
                keep[j]=false
                dropped_out+=1
                continue
            end
            fill!(Tbuf_check[1],0.0+0.0im)
            construct_boundary_matrices!(Tbuf_check,solver,pts,[λ[j]];multithreaded=multithreaded,use_chebyshev=use_chebyshev,n_panels=n_panels,M=M,timeit=false) # construct the T(λ[j]) matrix for the eigenvalue λ[j] to check the residual
            @blas_multi_then_1 MAX_BLAS_THREADS mul!(ybuf,Tbuf_check[1],@view(Phi[:,j]))
            ybuf_norm=norm(ybuf)
            @info "k=$((λ[j])) ||A(k)v(k)|| = $(ybuf_norm)"
            if auto_discard_spurious
                if ybuf_norm≥res_tol
                    keep[j]=false
                    dropped_res+=1
                    if ybuf_norm>1e-8
                        if ybuf_norm>1e-6 # heuristic for when usually it is spurious sqrt(eps())
                            @warn "k=$((λ[j])) ||A(k)v(k)|| = $(ybuf_norm) > $res_tol , definitely spurious" 
                        else # gray zone
                            @warn "k=$((λ[j])) ||A(k)v(k)|| = $(ybuf_norm) > $res_tol , most probably eigenvalue but too low nq" 
                        end
                    else
                        @warn "k=$((λ[j])) ||A(k)v(k)|| = $(ybuf_norm) > $res_tol , could be spurious or try increasing nq (usually spurious) or lowering residual tolerance" 
                    end
                    continue
                end
            end
            push!(tens,ybuf_norm)
            push!(res_keep,ybuf_norm)
        end
        kept=count(keep)
        if kept>0
            @info "STATUS: " kept=kept dropped_outside=dropped_out dropped_residual=dropped_res max_residual=maximum(res_keep)
        else
            @info "STATUS: " kept=0 dropped_outside=dropped_out dropped_residual=dropped_res
        end
    end
    return λ[keep],Phi[:,keep],tens
end

#####################################
#### RESIDUAL CALCULATION HELPER ####
#####################################

# Computes the residuals ||A(k)φ|| and normalized residuals for a set of eigenpairs (λ,φ) obtained from Beyn's method. 
# Inputs:
#   - solver::Union{BoundaryIntegralMethod,CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners,CFIE_alpert,DLP_kress,DLP_kress_global_corners}
#   - λ::AbstractVector{Complex{T}}: Eigenvalues obtained from Beyn
#   - Uk::AbstractMatrix{Complex{T}}: Left singular vectors from Beyn's method
#   - Y::AbstractMatrix{Complex{T}}: Eigenvectors from the small B
#   - k0::Complex{T}: Center of the contour
#   - R::T: Radius of the contour
#   - pts::Union{BoundaryPoints{T},Vector{BoundaryPointsCFIE{T}}}: Boundary points for either standard or CFIE_kress formulation
#   - symmetry::Union{Vector{Any},Nothing}: Symmetry information for the BIM
#   - res_tol::T: Residual tolerance for discarding spurious eigenvalues
#   - matnorm::Symbol=:one: Matrix norm to use for normalization (:one, :two, :inf)
#   - epss::Real=1e-15: Small value to avoid division by zero
#   - auto_discard_spurious::Bool=true: Whether to automatically discard spurious eigenvalues based on residuals
#   - collect_logs::Bool=false: Whether to collect logs of the selection process
#   - use_chebyshev::Bool=true: Whether to use Chebyshev Hankel evaluations
#   - n_panels::Int=2000: Number of Chebyshev panels
#   - M::Int=300: Degree of Chebyshev polynomials
#   - multithreaded::Bool=true: Whether to use multithreading in kernel assembly
#
# Outputs:
#   - idx::Vector{Int}: Indices of kept eigenpairs
#   - Φ_kept::AbstractMatrix{Complex{T}}: Kept eigenvectors
#   - tens::Vector{T}: Residual norms ||Aφ||
#   - tensN::Vector{T}: Normalized residuals
#   - logs::Vector{String}: Logs of the selection process (if collect_logs is true)
function residual_and_norm_select(solver::Union{BoundaryIntegralMethod,CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners,CFIE_alpert,DLP_kress,DLP_kress_global_corners},λ::AbstractVector{Complex{T}},Uk::AbstractMatrix{Complex{T}},Y::AbstractMatrix{Complex{T}},k0::Complex{T},R::T,pts::Union{BoundaryPoints{T},Vector{BoundaryPointsCFIE{T}}};res_tol::T,matnorm::Symbol=:one,epss::Real=1e-15,auto_discard_spurious::Bool=true,collect_logs::Bool=false,use_chebyshev::Bool=true,n_panels::Int=15000,M::Int=5,multithreaded::Bool=true) where {T<:Real}
    N,rk=size(Uk)
    Φtmp=Matrix{Complex{T}}(undef,N,rk)
    y=Vector{Complex{T}}(undef,N)
    keep=falses(rk)
    tens=Vector{T}(undef,rk)
    tensN=Vector{T}(undef,rk)
    logs=collect_logs ? String[] : nothing
    Tbufs=[zeros(Complex{T},N,N) for _ in eachindex(λ)]
    construct_boundary_matrices!(Tbufs,solver,pts,λ;multithreaded=multithreaded,use_chebyshev=use_chebyshev,n_panels=n_panels,M=M,timeit=false)
    vecnorm= matnorm===:one ? (v->norm(v,1)) : matnorm===:two ? (v->norm(v)) : (v->norm(v,Inf))
    @inbounds for j in 1:rk
        λj=λ[j]
        if abs(λj-k0)>R
            tens[j]=T(NaN)
            tensN[j]=T(NaN)
            continue
        end
        @blas_multi_then_1 MAX_BLAS_THREADS mul!(@view(Φtmp[:,j]),Uk,@view(Y[:,j]))
        @blas_multi_then_1 MAX_BLAS_THREADS mul!(y,Tbufs[j],@view(Φtmp[:,j]))
        rj=norm(y)
        tens[j]=rj
        nA= matnorm===:one ? opnorm(Tbufs[j],1) : matnorm===:two ? opnorm(Tbufs[j],2) : opnorm(Tbufs[j],Inf)
        φn=vecnorm(@view(Φtmp[:,j]))
        yn=vecnorm(y)
        tensN[j]=yn/(nA*(φn+epss)+epss)
        if auto_discard_spurious && rj>=res_tol
            collect_logs && push!(logs,"λ=$(λj) ||Aφ||=$(rj) > $res_tol → DROP")
        else
            keep[j]=true
            collect_logs && push!(logs,"λ=$(λj) ||Aφ||=$(rj) < $res_tol ← KEEP")
        end
    end
    idx=findall(keep)
    Φ_kept=isempty(idx) ? Matrix{Complex{T}}(undef,N,0) : Φtmp[:,idx]
    return idx,Φ_kept,tens[idx],tensN[idx],(collect_logs ? logs : String[])
end

########################
#### HIGH LEVEL API ####
########################

"""
    compute_spectrum_beyn(solver::Union{BoundaryIntegralMethod,CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners,CFIE_alpert,DLP_kress,DLP_kress_global_corners},basis::Ba,billiard::Bi,k1::T,k2::T;kwargs...)

Compute all eigenpairs in [k1,k2] via a two-phase Beyn workflow:
Phase 1: (per Weyl-balanced disk) build Fredholm matrices along the contour, run Beyn to get provisional eigenvalues λ and subspaces (Uk, Y).
Phase 2: validate each λ by computing residuals ||A(λ)φ|| (φ=Uk*Y[:,j]), keep those inside the disk with residual < res_tol.

# Inputs:
- solver::Union{BoundaryIntegralMethod,CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners,CFIE_alpert,DLP_kress,DLP_kress_global_corners}: The BIM solver object
- basis::Ba: The basis type
- billiard::Bi: The billiard domain
- k1::T: Lower bound of the wavenumber interval
- k2::T: Upper bound of the wavenumber interval

# Keyword arguments:
- m::Int=10: Wanted number of eigenvalues per Weyl window
- Rmax::T=one(T): Maximum radius of the Weyl windows (careful not to choose m too large to go over the max windows size)
- nq::Int=48: Number of quadrature points on the contour
- r::Int=m+15: Number of expected eigenvalues inside each contour + padding to avoid rank saturation
- svd_tol::Real=1e-12: Tolerance for the SVD truncation in Beyn
- res_tol::Real=1e-9: Residual tolerance for discarding
- spurious::Bool=true: Whether to discard spurious eigenvalues
- multithreaded_matrix::Bool=false: Whether to use multithreading in matrix assembly
- use_adaptive_svd_tol::Bool=false: Whether to use adaptive svd tolerance based on maximum singular value
- use_chebyshev::Bool=true: Whether to use Chebyshev Hankel evaluations
- n_panels_init::Int=2000: Number of Chebyshev panels as initial guess
- M_init::Int=300: Degree of Chebyshev polynomials as initial guess
- do_INFO_init::Bool=true: Whether to run a diagnostic Beyn solve on the last window
- do_per_solve_INFO::Bool=false: Whether to run the diagnostics on each solve
- cheb_tol::Real=1e-10: Tolerance for Chebyshev parameter tuning
- max_iter::Int=10: Maximum iterations for Chebyshev parameter tuning
- sampling_points::Int=50_000: Number of points to sample for Chebyshev parameter tuning
- grading::Symbol=:uniform: Grading strategy for Chebyshev panels, by default uniform, can be :uniform or :geometric
- grow_panels::Real=1.5: Growth factor for number of panels
- grow_M::Int=2: Growth factor for degree of Chebyshev polynomials

# Returns:
ks      :: Vector{T}                   – kept real wavenumbers Re(λ)
tens    :: Vector{T}                   – raw residuals ||A(λ)φ||
us      :: Vector{Vector{Complex{T}}}  – kept DLP densities φ (one per eigenvalue)
pts     :: Vector{BoundaryPoints{T}}– matching boundary points object per φ
tensN   :: Vector{T}                   – normalized residuals (scale-free)

# Notes:
   • m controls the expected #eigs per Weyl window; Rmax caps disk radius.
   • nq should not be tiny (spectral conv. on analytic boundaries, but use ≥15).
   • r is the probe rank for Beyn (auto-bumped internally if saturated).
   • use_chebyshev turns on Chebyshev Hankel evaluation (faster at large k).
"""
function compute_spectrum_beyn(solver::Union{BoundaryIntegralMethod,CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners,CFIE_alpert,DLP_kress,DLP_kress_global_corners},basis::Ba,billiard::Bi,k1::T,k2::T;m::Int=50,Rmax::T=one(T),nq::Int=48,r::Int=m+15,svd_tol::Real=1e-12,res_tol::Real=1e-9,auto_discard_spurious::Bool=true,multithreaded_matrix::Bool=true,use_adaptive_svd_tol::Bool=false,use_chebyshev::Bool=true,n_panels_init=15000,M_init=5,do_INFO_init::Bool=true,do_per_solve_INFO::Bool=true,cheb_tol::Real=1e-10,max_iter::Int=10,sampling_points::Int=50_000,grading::Symbol=:uniform,grow_panels::Real=1.5,grow_M::Int=2) where {T<:Real,Bi<:AbsBilliard,Ba<:AbstractHankelBasis}
    fundamental=!isnothing(solver.symmetry)
    intervals=plan_weyl_windows(billiard,k1,k2;m=m,fundamental=fundamental,Rmax=Rmax)
    if length(intervals)>=2
        kL2,kR2=intervals[end-1]
        kL3,kR3=intervals[end]
        len3=kR3-kL3
        if len3<=max(100*eps(k2),1e-9*max(k2,one(T)))
            if (kR3-kL2)<=2*Rmax+10*eps(k2)
                intervals[end-1]=(kL2,kR3)
                pop!(intervals)
            else
                pop!(intervals)
            end
        end
    end
    k0,R=beyn_disks_from_windows(intervals)
    isempty(k0) && return T[],T[],Vector{Vector{Complex{T}}}(),Vector{Union{BoundaryPoints{T},Vector{BoundaryPointsCFIE{T}}}}(),T[]
    do_INFO_init && @info "Weyl windows planned" intervals=intervals k0=k0 R=R
    nq<=15 && error("Do not use less than 15 contour nodes")
    all_pts=Vector{Union{BoundaryPoints{T},Vector{BoundaryPointsCFIE{T}}}}(undef,length(k0))
    @benchit timeit=do_per_solve_INFO "Point evaluation" for i in eachindex(k0)
        all_pts[i]=evaluate_points(solver,billiard,real(k0[i]))
    end
    λs=Vector{Vector{Complex{T}}}(undef,length(k0))
    Uks=Vector{Matrix{Complex{T}}}(undef,length(k0))
    Ys=Vector{Matrix{Complex{T}}}(undef,length(k0))
    k0s=Vector{Complex{T}}(undef,length(k0))
    Rs=Vector{T}(undef,length(k0))
    n_panels=n_panels_init
    M=M_init
    if use_chebyshev
        imax=argmax(real.(k0).+ R)
        θref=range(zero(T),TWO_PI;length=nq+1)[1:end-1]
        zj_ref=k0[imax].+R[imax].*cis.(θref)
        n_panels,M,_... =chebyshev_params(solver,all_pts[imax],zj_ref;tol=cheb_tol,n_panels_init=n_panels_init,M_init=M_init,grading=grading,sampling_points=sampling_points,max_iter=max_iter,grow_panels=grow_panels,grow_M=grow_M,verbose=do_per_solve_INFO)
    end
    if do_INFO_init
        mid=cld(length(k0),2)
        @benchit timeit=do_INFO_init "solve_INFO representative disk" begin
            _=solve_INFO(solver,basis,all_pts[mid],complex(k0[mid]),R[mid];nq=nq,r=r,svd_tol=svd_tol,res_tol=res_tol,use_adaptive_svd_tol=use_adaptive_svd_tol,multithreaded=multithreaded_matrix,use_chebyshev=use_chebyshev,n_panels=n_panels,M=M)
        end
    end
    p=Progress(length(k0),1)
    @time "Beyn pass (all disks)" begin
        @inbounds for i in eachindex(k0)
            λ,Uk,Y,κ,δ,ptsi=solve_vect(solver,basis,all_pts[i],complex(k0[i]),R[i];nq=nq,r=r,svd_tol=svd_tol,res_tol=res_tol,rng=MersenneTwister(0),auto_discard_spurious=auto_discard_spurious,multithreaded=multithreaded_matrix,use_chebyshev=use_chebyshev,n_panels=n_panels,M=M,info=do_per_solve_INFO)
            λs[i]=λ
            Uks[i]=Uk
            Ys[i]=Y
            k0s[i]=κ
            Rs[i]=δ
            all_pts[i]=ptsi
            next!(p)
        end
    end
    ks_list=Vector{Vector{T}}(undef,length(k0))
    tens_list=Vector{Vector{T}}(undef,length(k0))
    tensN_list=Vector{Vector{T}}(undef,length(k0))
    phi_list=Vector{Matrix{Complex{T}}}(undef,length(k0))
    @benchit timeit=do_per_solve_INFO "Residuals/tensions pass" begin
        @inbounds @showprogress for i in eachindex(k0)
            if isempty(λs[i])
                ks_list[i]=T[]
                tens_list[i]=T[]
                tensN_list[i]=T[]
                phi_list[i]=Matrix{Complex{T}}(undef,boundary_matrix_size(all_pts[i]),0)
                continue
            end
            idx,Φ_kept,traw,tnorm,_=residual_and_norm_select(solver,λs[i],Uks[i],Ys[i],k0s[i],Rs[i],all_pts[i];res_tol=T(res_tol),matnorm=:one,epss=1e-15,auto_discard_spurious=auto_discard_spurious,collect_logs=false,use_chebyshev=use_chebyshev,n_panels=n_panels,M=M,multithreaded=multithreaded_matrix)
            ks_list[i]=real.(λs[i][idx])
            tens_list[i]=traw
            tensN_list[i]=tnorm
            phi_list[i]=Matrix(Φ_kept)
        end
    end
    nw=length(phi_list)
    n_by_win=Vector{Int}(undef,nw)
    @inbounds for i in 1:nw
        n_by_win[i]=size(phi_list[i],2)
    end
    offs=zeros(Int,nw)
    @inbounds for i in 2:nw
        offs[i]=offs[i-1]+n_by_win[i-1]
    end
    ntot=offs[end]+n_by_win[end]
    ks_all=Vector{T}(undef,ntot)
    tens_all=Vector{T}(undef,ntot)
    tensN_all=Vector{T}(undef,ntot)
    us_all=Vector{Vector{Complex{T}}}(undef,ntot)
    pts_all=(solver isa CFIE) ? Vector{Vector{BoundaryPointsCFIE{T}}}(undef,ntot) : Vector{BoundaryPoints{T}}(undef,ntot)
    Threads.@threads for i in 1:nw
        n=n_by_win[i]
        n==0 && continue
        off=offs[i]
        ksi=ks_list[i]
        tr=tens_list[i]
        tn=tensN_list[i]
        Φ=phi_list[i]
        ptsi=all_pts[i]
        @inbounds for j in 1:n
            ks_all[off+j]=ksi[j]
            tens_all[off+j]=tr[j]
            tensN_all[off+j]=tn[j]
            us_all[off+j]=vec(@view Φ[:,j])
            pts_all[off+j]=ptsi
        end
    end
    return ks_all,tens_all,us_all,pts_all,tensN_all
end