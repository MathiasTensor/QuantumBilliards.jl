# ===============================================================================
#   End-to-end spectrum computation for 2D quantum billiards in the Poincaré disk
#   (constant curvature -1) using a boundary-integral formulation of the
#   hyperbolic Helmholtz eigenproblem
#
#       (Δ_H + 1/4 + k^2) u = 0
#
#   with Dirichlet boundary conditions on a billiard domain Γ.
#
#   The nonlinear eigenvalue problem is solved by a Beyn contour-integral method:
#     - For each complex contour disk Γ(k0,R) in the k-plane:
#         * assemble the hyperbolic DLP Fredholm matrix T(k)
#         * factor T(k) for each quadrature node on the contour
#         * compute Beyn moments A0, A1 and reduce to a small matrix B
#         * extract candidate eigenvalues from eig(B)
#         * (optional/standard) validate candidates by residual ||T(k)v||
#
#   The hyperbolic Green kernel is evaluated via precomputed Legendre-Q Taylor
#   tables in distance d:
#
#       G_k(d) = (1/(2π)) Q_ν(cosh d),   ν = -1/2 + i k
#
#   so that kernel assembly is dominated by dense linear algebra (LU/SVD), not
#   special-function evaluation.
#
# CONTENTS (MAIN ENTRY)
#   - plan_k_windows_hyp         : choose a set of Beyn disks covering [k1,k2]
#   - compute_spectrum_hyp       : full pipeline (plan -> discretize -> Beyn -> filter)
#
# SUPPORT ROUTINES
#   - construct_B_matrix_hyp     : builds reduced Beyn matrix B and Uk for one disk
#   - solve_vect_hyp / solve_hyp : one-disk solves
#   - residual_and_norm_select_hyp : residual-based pruning of candidates
#   - solve_INFO_hyp             : verbose diagnostic version for debugging
#
# INPUT / OUTPUT CONVENTIONS
#   - Geometry points/normals/curvature are Euclidean in the disk model.
#   - Hyperbolic DLP assembly uses Euclidean ds weights and adds the -1/2 jump.
#   - Symmetry handling (none / Reflection / Rotation) is supported via dispatch.
#
# PERFORMANCE NOTES
#   - Main costs: nq dense N×N LU factorizations per disk + residual rebuilds.
#   - Special functions are amortized by QTaylorTable; only one mpmath seed per k.
#
# AUTHOR / DATE
#   MO / 22-12-2025
# ===============================================================================

# ===============================================================================
# plan_k_windows_hyp
#
# PURPOSE
#   Partition the real k-interval [k1,k2] into a list of non-overlapping Beyn
#   contour disks (centers k0s[i], radii Rs[i]) for the hyperbolic BIM.
#
#   Each disk is intended to contain O(M) eigenvalues (on average), using a
#   Weyl-like mean density for the hyperbolic Helmholtz operator:
#
#        (Δ_H + 1/4 + k^2) u = 0
#
#   with area A of the fundamental domain. We use the leading term
#
#        N(k) ~ (A / (4π)) k^2    =>    dN/dk ~ ρA(k) = (A / (2π)) k
#
#   and choose disk radii R(k0) so that the arc length 2R captures about M
#   eigenvalues:
#
#        expected count in [k0-R, k0+R]  ~  2R * ρA(k0)  ~  M
#        => R(k0) ~ M / (2 ρA(k0)).
#
#   The algorithm marches from left=k1 to k2, greedily choosing a radius
#   constrained by (i) remaining interval length and (ii) user bounds.
#
# INPUTS
#   solver :: BIM_hyperbolic
#       Hyperbolic BIM solver handle (contains symmetry info etc.).
#
#   billiard :: Bi   where Bi <: AbsBilliard
#       Geometry definition (fundamental domain); used only to compute area.
#
#   k1 :: T,  k2 :: T   where T <: Real
#       Start and end of spectral interval in real k (k2 > k1 required).
#
# KEYWORD INPUTS
#   M :: Int = 50
#       Target mean number of eigenvalues per disk (density heuristic).
#
#   Rmax :: T = 0.8
#       Upper bound on disk radius. Prevents overly large disks where Beyn
#       becomes less stable (too many eigenvalues; worse conditioning).
#
#   Rfloor :: T = 1e-6
#       Lower bound on disk radius. Prevents infinite loops or degenerate disks.
#
#   kref :: T = 1000
#       Reference k for the area computation routine (if it uses sampling).
#
#   tolA :: Real = 1e-8
#       Tolerance passed to hyperbolic_area_fundamental.
#
#   iters :: Int = 8
#       Fixed-point refinement iterations for R = min(Rof(k0), rem/2)
#       because k0 depends on R (k0 = left + R).
#
# OUTPUTS
#   k0s :: Vector{T}
#       Disk centers. Guaranteed increasing. Each disk covers [k0-R, k0+R].
#
#   Rs  :: Vector{T}
#       Disk radii corresponding to k0s.
# ===============================================================================
function plan_k_windows_hyp(solver::BIM_hyperbolic,billiard::Bi,k1::T,k2::T;M::Int=50,Rmax::T=T(0.8),Rfloor::T=T(1e-6),kref::T=T(1000),tolA::Real=1e-8,iters::Int=8) where {Bi<:AbsBilliard,T<:Real}
    L=k2-k1
    (L<=zero(T) || Rmax<=zero(T)) && return T[],T[]
    A=hyperbolic_area_fundamental(solver,billiard;tol=tolA,kref=kref)
    ρA(k)=max((A/TWO_PI)*k,T(1e-12))
    Rof(k)=clamp(M/(2*ρA(k)),Rfloor,Rmax)
    k0s=T[];Rs=T[]
    left=k1
    while left<k2-T(10)*eps(k2)
        rem=k2-left
        rem<=zero(T) && break
        R=clamp(rem/2,Rfloor,Rmax)
        @inbounds for _ in 1:iters
            k0=left+R
            R=min(Rof(k0),rem/2)
            R=clamp(R,Rfloor,Rmax)
        end
        push!(k0s,left+R)
        push!(Rs,R)
        left+=2*R
    end
    return k0s,Rs
end

# ===============================================================================
# construct_B_matrix_hyp
#
# PURPOSE
#   Construct the reduced Beyn matrix B (and the projection basis Uk) for one
#   contour disk Γ in the complex k-plane, for the hyperbolic BIM nonlinear EVP:
#
#       T(k) φ = 0,
#
#   where T(k) is the assembled boundary integral operator matrix (here: DLP
#   Fredholm matrix including the -1/2 jump, quadrature weights, and filtering).
#
#   Beyn moment integrals on Γ (circle):
#       A0 = (1/2πi) ∮ T(z)^{-1} V dz
#       A1 = (1/2πi) ∮ z T(z)^{-1} V dz
#
#   with random probing matrix V (N×r). Discretized with nq quadrature points:
#       z_j = k0 + R e^{iθ_j},   θ_j = 2π(j-1/2)/nq
#       w_j = (R/nq) e^{iθ_j}    (trapezoidal rule on circle)
#
#   Then compute SVD(A0) to obtain rank rk and form:
#       B = Uk^H A1 Wk Σk^{-1}
#
# INPUTS
#   solver :: BIM_hyperbolic
#       Hyperbolic BIM solver (symmetry rules, boundary sampling policy, etc.).
#
#   pts :: BoundaryPointsHypBIM{T}
#       Boundary discretization in the hyperbolic setting.
#       NOTE: you convert to Euclidean BoundaryPointsBIM for kernel assembly.
#
#   N :: Int
#       Number of boundary DOFs (must match length(pts.xy)).
#
#   k0 :: Complex{T}
#       Center of Beyn contour disk in complex k-plane.
#
#   R :: T
#       Radius of Beyn contour disk.
#
# KEYWORD INPUTS (Beyn)
#   nq :: Int = 64
#       Number of quadrature nodes on the contour circle.
#
#   r :: Int = 48
#       Number of probing vectors (columns of V). Controls rank detection.
#
#   svd_tol :: Real = 1e-14
#       Singular-value cutoff for rank determination from Σ(A0).
#
#   rng :: MersenneTwister
#       RNG for generating random probing matrix V.
#
#   multithreaded :: Bool = true
#       Enables threading inside kernel assembly routines.
#
# KEYWORD INPUTS (Q-table / LegendreQ engine)
#   h :: Real = 1e-4
#       Taylor patch spacing in d for QTaylorTable.
#
#   P :: Int = 30
#       Taylor degree per patch.
#
#   mp_dps :: Int = 60
#       Decimal digits used in mpmath seeding at dmin (ONE seed per k on Γ).
#
#   leg_type :: Int = 3
#       mpmath LegendreQ definition selector.
#
# OUTPUTS
#   (B, Uk) :: Tuple{Matrix{Complex{T}}, Matrix{Complex{T}}}
#
#   B  :: Matrix{Complex{T}}  size (rk × rk)
#       Reduced Beyn matrix whose eigenvalues approximate eigenvalues of T(k).
#
#   Uk :: Matrix{Complex{T}}  size (N × rk)
#       Left singular subspace basis of A0 associated with kept singular values.
#
# ALGORITHM
#   1) Build contour nodes zj and weights wj (Complex{T}).
#   2) For each quadrature node z_j:
#        a) Build QTaylorTable tab_j for ν=-1/2 + i z_j  (hyperbolic Green kernel)
#        b) Assemble DLP matrix Tbufs[j] (with symmetry if needed)
#        c) Apply assemble_DLP_hyperbolic! (weights, -1/2 jump, filtering)
#        d) LU factorization Fs[j] = lu!(Tbufs[j])
#   3) Accumulate A0,A1 by repeated solves:
#        X := T(z_j)^{-1} V   via ldiv!(X, Fs[j], V)
#        A0 += w_j * X
#        A1 += w_j * z_j * X
#   4) SVD(A0) => rank rk by svd_tol.
#   5) Build reduced B via Uk,Wk,Σk and A1.
# ===============================================================================
function construct_B_matrix_hyp(solver::BIM_hyperbolic,pts::BoundaryPointsHypBIM{T},N::Int,k0::Complex{T},R::T;nq::Int=64,r::Int=48,svd_tol=1e-14,rng=MersenneTwister(0),multithreaded::Bool=true,h=1e-4,P=30,mp_dps::Int=60,leg_type::Int=3)::Tuple{Matrix{Complex{T}},Matrix{Complex{T}}} where {T<:Real}
    @info "Constructing B matrix (hyp) with N=$N, k0=$k0, R=$R, nq=$nq, r=$r"
    θ=(TWO_PI/nq).*(collect(0:nq-1).+0.5)
    ej=cis.(θ);zj=k0.+R.*ej;wj=(R/nq).*ej
    ks=ComplexF64.(zj)
    Tbufs=[zeros(Complex{T},N,N) for _ in 1:nq]
    dmin,dmax=d_bounds_hyp(pts,solver.symmetry)
    pts_eucl=_BoundaryPointsHypBIM_to_BoundaryPointsBIM(pts)
    dmin=max(dmin,1e-3)
    #pre=build_QTaylorPrecomp(dmin=dmin,dmax=dmax,h=h,P=P)
    #tabs=alloc_QTaylorTables(pre,nq;k=ks[1])
    #ws=QTaylorWorkspace(P;threaded=multithreaded)
    #build_QTaylorTable!(tabs,pre,ws,ks;mp_dps=mp_dps,leg_type=leg_type,threaded=multithreaded)
    tabs=Vector{QTaylorTable}(undef,nq)
    for j in 1:nq
        tabs[j]=build_QTaylorTable(ks[j],dmin=dmin,dmax=dmax,h=h,P=P,mp_dps=mp_dps,leg_type=leg_type)
    end
    compute_kernel_matrices_DLP_hyperbolic!(Tbufs,pts_eucl,solver.symmetry,tabs;multithreaded=multithreaded)
    assemble_DLP_hyperbolic!(Tbufs,pts_eucl)
    @blas_multi MAX_BLAS_THREADS F1=lu!(Tbufs[1];check=false)
    Fs=Vector{typeof(F1)}(undef,nq);Fs[1]=F1
    @blas_multi_then_1 MAX_BLAS_THREADS @inbounds for j in 2:nq
        Fs[j]=lu!(Tbufs[j];check=false)
    end
    function accum_moments!(A0::Matrix{Complex{T}},A1::Matrix{Complex{T}},X::Matrix{Complex{T}},V::Matrix{Complex{T}})
        xv=reshape(X,:);a0v=reshape(A0,:);a1v=reshape(A1,:)
        @blas_multi_then_1 MAX_BLAS_THREADS @inbounds for j in 1:nq
            ldiv!(X,Fs[j],V)
            BLAS.axpy!(wj[j],xv,a0v)
            BLAS.axpy!(wj[j]*zj[j],xv,a1v)
        end
        return nothing
    end
    V,X,A0,A1=beyn_buffer_matrices(T,N,r,rng)
    accum_moments!(A0,A1,X,V)
    @blas_multi_then_1 MAX_BLAS_THREADS U,Σ,W=svd!(A0;full=false)
    rk=count(>=(svd_tol),Σ)
    rk==0 && return Matrix{Complex{T}}(undef,N,0),Matrix{Complex{T}}(undef,N,0)
    if rk==r
        r_tmp=r+r
        while r_tmp<N
            V,X,A0,A1=beyn_buffer_matrices(T,N,r_tmp,rng)
            accum_moments!(A0,A1,X,V)
            @blas_multi_then_1 MAX_BLAS_THREADS U,Σ,W=svd!(A0;full=false)
            rk=count(>=(svd_tol),Σ)
            rk<r_tmp && break
            r_tmp+=r
            r_tmp>N && throw(ArgumentError("r > N is impossible: requested r=$(r_tmp), N=$(N)"))
        end
        rk==r_tmp && @warn "All singular values ≥ svd_tol=$(svd_tol); consider increasing r or decreasing R"
    end
    Uk=@view U[:,1:rk]
    Wk=@view W[:,1:rk]
    Σk=@view Σ[1:rk]
    tmp=Matrix{Complex{T}}(undef,N,rk)
    @blas_multi_then_1 MAX_BLAS_THREADS mul!(tmp,A1,Wk)
    @inbounds @simd for j in 1:rk
        @views tmp[:,j]./=Σk[j]
    end
    B=Matrix{Complex{T}}(undef,rk,rk)
    @blas_multi_then_1 MAX_BLAS_THREADS mul!(B,adjoint(Uk),tmp)
    return B,Uk
end

# ===============================================================================
# solve_vect_hyp
#
# PURPOSE
#   Execute the Beyn reduction on one contour disk Γ and return:
#     - candidate eigenvalues λ inside Γ (eigenvalues of reduced B),
#     - the subspace Uk and eigenvectors Y of B (needed to reconstruct boundary
#       densities / null vectors in the original N-dimensional space).
#
# INPUTS
#   solver :: BIM_hyperbolic
#
#   basis  :: Ba   where Ba <: AbstractHankelBasis
#       Currently unused in this function body (kept for uniform API).
#
#   pts    :: BoundaryPointsHypBIM{T}
#   k0     :: Complex{T}
#   R      :: T
#
# KEYWORD INPUTS
#   nq, r, svd_tol, res_tol, rng, multithreaded, h, P, mp_dps, leg_type
#       Passed through to construct_B_matrix_hyp.
#
# OUTPUTS
#   (λ, Uk, Y, k0, R, pts)
#
#   λ  :: Vector{Complex{T}}    length rk
#       Eigenvalues of B (approximate k-eigenvalues of the BIM operator).
#
#   Uk :: Matrix{Complex{T}}    size N×rk
#   Y  :: Matrix{Complex{T}}    size rk×rk
#       So that boundary densities Φ ≈ Uk*Y.
#
#   k0, R, pts returned for convenience / logging.
#
# NOTES
#   - If rk==0 (empty B), returns empty λ and empty matrices.
# ===============================================================================
function solve_vect_hyp(solver::BIM_hyperbolic,basis::Ba,pts::BoundaryPointsHypBIM{T},k0::Complex{T},R::T;nq::Int=64,r::Int=48,svd_tol::Real=1e-14,res_tol::Real=1e-8,rng=MersenneTwister(0),multithreaded::Bool=true,h::T=T(1e-4),P::Int=30,mp_dps::Int=60,leg_type::Int=3) where {Ba<:AbstractHankelBasis,T<:Real}
    N=length(pts.xy)
    B,Uk=construct_B_matrix_hyp(solver,pts,N,k0,R;nq=nq,r=r,svd_tol=svd_tol,rng=rng,multithreaded=multithreaded,h=h,P=P,mp_dps=mp_dps,leg_type=leg_type)
    if isempty(B)
        return Complex{T}[],Uk,Matrix{Complex{T}}(undef,0,0),k0,R,pts
    end
    @blas_multi_then_1 MAX_BLAS_THREADS λ,Y=eigen!(B)
    return λ,Uk,Y,k0,R,pts
end

# ===============================================================================
# solve_hyp
#
# PURPOSE
#   Convenience wrapper returning only the eigenvalue candidates λ for one disk.
#
# INPUTS / KEYWORDS
#   Same as solve_vect_hyp.
#
# OUTPUTS
#   λ :: Vector{Complex{T}}
# ===============================================================================
@inline function solve_hyp(solver::BIM_hyperbolic,basis::Ba,pts::BoundaryPointsHypBIM{T},k0::Complex{T},R::T;nq::Int=64,r::Int=48,svd_tol::Real=1e-14,res_tol::Real=1e-8,rng=MersenneTwister(0),multithreaded::Bool=true,h::T=T(1e-4),P::Int=30,mp_dps::Int=60,leg_type::Int=3) where {Ba<:AbstractHankelBasis,T<:Real}
    λ,_,_,_,_,_=solve_vect_hyp(solver,basis,pts,k0,R;nq=nq,r=r,svd_tol=svd_tol,res_tol=res_tol,rng=rng,multithreaded=multithreaded,h=h,P=P,mp_dps=mp_dps,leg_type=leg_type)
    return λ
end

# ===============================================================================
# residual_and_norm_select_hyp
#
# PURPOSE
#   Post-processing / filtering of candidate eigenvalues λ from one Beyn disk.
#
#   For each candidate λ_j inside the disk, reconstruct a boundary vector
#       Φ_j = Uk * Y[:,j]
#   and evaluate the residual / “tension”:
#       y = T(λ_j) Φ_j,
#       r_j = ||y||_2
#
#   Optionally compute a normalized residual measure:
#       tensN = ||y|| / ( ||T|| * (||Φ|| + epss) + epss )
#   using selectable matrix/vector norms.
#
#   Candidates are kept if r_j < res_tol (unless auto_discard_spurious=false).
#
# INPUTS
#   solver :: BIM_hyperbolic
#
#   λ  :: AbstractVector{Complex{T}}
#       Candidate eigenvalues (from reduced B).
#
#   Uk :: AbstractMatrix{Complex{T}}    size N×rk
#   Y  :: AbstractMatrix{Complex{T}}    size rk×rk
#
#   k0 :: Complex{T},  R :: T
#       Disk definition used for "inside disk" check |λ-k0| <= R.
#
#   pts :: BoundaryPointsHypBIM{T}
#       Boundary discretization for residual evaluation at each λ_j.
#
# KEYWORD INPUTS
#   res_tol :: T
#       Residual cutoff for accepting eigenvalues.
#
#   matnorm :: Symbol = :one
#       Which operator norm to estimate for T(λ) in tensN: :one, :two, :inf.
#
#   epss :: Real = 1e-15
#       Stabilizer for normalized residual (avoid division by 0).
#
#   auto_discard_spurious :: Bool = true
#       If true, drop λ_j when r_j >= res_tol.
#
#   collect_logs :: Bool = false
#       If true, return verbose string logs for each candidate.
#
#   multithreaded :: Bool = true
#       Threaded assembly for the per-λ residual matrix build.
#
#   h, P, mp_dps, leg_type
#       Q-table parameters. NOTE: this function currently rebuilds tab per λ_j.
#
# OUTPUTS
#   idx     :: Vector{Int}
#       Indices of kept candidates in λ (subset of 1:rk).
#
#   Φ_kept  :: Matrix{Complex{T}}   size N×nkeep
#       Kept boundary vectors.
#
#   tens    :: Vector{T}            length nkeep
#       Raw residual norms ||T(λ)Φ|| (2-norm in current code).
#
#   tensN   :: Vector{T}            length nkeep
#       Normalized residual metric.
#
#   logs    :: Vector{String}
#       Empty unless collect_logs=true.
#
# PERFORMANCE NOTES (IMPORTANT)
#   - This is expensive because it builds a FULL N×N matrix T(λ_j) for each λ_j.
#     That is often the dominant cost after Beyn if many candidates exist.
# ===============================================================================
function residual_and_norm_select_hyp(solver::BIM_hyperbolic,λ::AbstractVector{Complex{T}},Uk::AbstractMatrix{Complex{T}},Y::AbstractMatrix{Complex{T}},k0::Complex{T},R::T,pts::BoundaryPointsHypBIM{T};res_tol::T,matnorm::Symbol=:one,epss::Real=1e-15,auto_discard_spurious::Bool=true,collect_logs::Bool=false,multithreaded::Bool=true,h::T=T(1e-4),P::Int=30,mp_dps::Int=60,leg_type::Int=3) where {T<:Real}
    N,rk=size(Uk)
    Φtmp=Matrix{Complex{T}}(undef,N,rk)
    y=Vector{Complex{T}}(undef,N)
    keep=falses(rk)
    tens=Vector{T}(undef,rk)
    tensN=Vector{T}(undef,rk)
    logs=collect_logs ? String[] : nothing
    A_buf=fill(zero(Complex{T}),N,N)
    pts_eucl=_BoundaryPointsHypBIM_to_BoundaryPointsBIM(pts)
    dmin,dmax=d_bounds_hyp(pts,solver.symmetry)
    dmin=max(dmin,1e-3)
    #pre=build_QTaylorPrecomp(dmin=dmin,dmax=dmax,h=T(h),P=P)
    #tab=alloc_QTaylorTable(pre;k=ComplexF64(k0))
    #ws=QTaylorWorkspace(P;threaded=false)
    vecnorm = matnorm===:one ? (v->norm(v,1)) : matnorm===:two ? (v->norm(v)) : (v->norm(v,Inf))
    @inbounds for j in 1:rk
        λj=λ[j]
        abs(λj-k0)>R && (tens[j]=T(NaN);tensN[j]=T(NaN);continue)
        @blas_multi_then_1 MAX_BLAS_THREADS mul!(@view(Φtmp[:,j]),Uk,@view(Y[:,j]))
        #build_QTaylorTable!(tab,pre,ws,ComplexF64(λj);mp_dps=mp_dps,leg_type=leg_type)
        tab=build_QTaylorTable(ComplexF64(λj),dmin=dmin,dmax=dmax,h=h,P=P,mp_dps=mp_dps,leg_type=leg_type)
        compute_kernel_matrices_DLP_hyperbolic!(A_buf,pts_eucl,solver.symmetry,tab;multithreaded=multithreaded)
        assemble_DLP_hyperbolic!(A_buf,pts_eucl)
        @blas_multi_then_1 MAX_BLAS_THREADS mul!(y,A_buf,@view(Φtmp[:,j]))
        rj=norm(y)
        tens[j]=rj
        nA=matnorm===:one ? opnorm(A_buf,1) : matnorm===:two ? opnorm(A_buf,2) : opnorm(A_buf,Inf)
        φn=vecnorm(@view(Φtmp[:,j]))
        yn=vecnorm(y)
        tensN[j]=yn/(nA*(φn+epss)+epss)
        if auto_discard_spurious && rj≥res_tol
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

# ===============================================================================
# solve_INFO_hyp
#
# PURPOSE
#   Diagnostic solver for one Beyn disk in hyperbolic geometry.
#   Intended to print timing, conditioning, singular values, and residual checks
#   to debug stability issues (e.g. LU conditioning, SVD rank, tab building).
#
#   Returns kept eigenvalues, reconstructed boundary vectors, and tensions.
#
# INPUTS / KEYWORDS
#   Same physical meaning as solve_vect_hyp, plus:
#
#   use_adaptive_svd_tol :: Bool
#       If true, set svd_tol_eff = max(Σ) * 1e-15.
#
#   auto_discard_spurious :: Bool
#       If true, residual test is applied inside this function too.
#
# OUTPUTS
#   λ_keep :: Vector{Complex{T}}
#   Φ_keep :: Matrix{Complex{T}}  size N×nkeep
#   tens   :: Vector{T}           residual norms for kept values
#
# INTERNAL STEPS
#   1) Build nq QTaylorTable objects for contour nodes.
#   2) Assemble all T(z_j) and LU factor them, printing condition diagnostics.
#   3) Accumulate A0,A1, compute SVD(A0) and build reduced B.
#   4) Compute eigen(B) => λ candidates.
#   5) Optionally rebuild T(λ_j) and compute residual ||T(λ_j) Φ_j||.
# ===============================================================================
function solve_INFO_hyp(solver::BIM_hyperbolic,basis::Ba,pts::BoundaryPointsHypBIM{T},k0::Complex{T},R::T;multithreaded::Bool=true,nq::Int=64,r::Int=48,svd_tol::Real=1e-10,res_tol::Real=1e-10,rng=MersenneTwister(0),use_adaptive_svd_tol::Bool=false,auto_discard_spurious::Bool=false,h::T=T(1e-4),P::Int=30,mp_dps::Int=60,leg_type::Int=3) where {Ba<:AbstractHankelBasis,T<:Real}
    N=length(pts.xy)
    θ=(TWO_PI/nq).*(collect(0:nq-1).+0.5)
    ej=cis.(θ);zj=k0.+R.*ej;wj=(R/nq).*ej
    ks=ComplexF64.(zj)
    V,X,A0,A1=beyn_buffer_matrices(T,N,r,rng)
    @info "beyn:start(hyp)" k0=k0 R=R nq=nq N=N r=r
    Tbufs=[zeros(Complex{T},N,N) for _ in 1:nq]
    dmin,dmax=d_bounds_hyp(pts,solver.symmetry)
    pts_eucl=_BoundaryPointsHypBIM_to_BoundaryPointsBIM(pts)
    xy=pts_eucl.xy
    if norm(xy[1]-xy[end])<1e-14
        @warn "Duplicate endpoint in boundary points; drop last point!" N=length(xy)
    end
    dmin=max(dmin,1e-3)
    tabs=Vector{QTaylorTable}(undef,nq)
    for j in 1:nq
        tabs[j]=build_QTaylorTable(ks[j],dmin=dmin,dmax=dmax,h=h,P=P,mp_dps=mp_dps,leg_type=leg_type)
    end
    @time "DLP(hyp):kernel+assemble" begin
        compute_kernel_matrices_DLP_hyperbolic!(Tbufs,pts_eucl,solver.symmetry,tabs;multithreaded=multithreaded)
        assemble_DLP_hyperbolic!(Tbufs,pts_eucl)
    end
    @blas_multi MAX_BLAS_THREADS F1=lu!(Tbufs[1];check=false)
    Fs=Vector{typeof(F1)}(undef,nq);Fs[1]=F1
    @blas_multi_then_1 MAX_BLAS_THREADS @inbounds @showprogress desc="lu!(hyp)" for j in 2:nq
        an=opnorm(Tbufs[j],1)
        Fs[j]=lu!(Tbufs[j];check=false)
        rc=LAPACK.gecon!('1',Fs[j].factors,an)
        println("LAPACK.gecon! = ",rc)
    end
    function accum_moments!(A0::Matrix{Complex{T}},A1::Matrix{Complex{T}},X::Matrix{Complex{T}},V::Matrix{Complex{T}})
        xv=reshape(X,:);a0v=reshape(A0,:);a1v=reshape(A1,:)
        @time "ldiv!+axpy!(hyp)" begin
            @blas_multi_then_1 MAX_BLAS_THREADS @inbounds @showprogress desc="ldiv!+axpy!(hyp)" for j in 1:nq
                ldiv!(X,Fs[j],V)
                BLAS.axpy!(wj[j],xv,a0v)
                BLAS.axpy!(wj[j]*zj[j],xv,a1v)
            end
            return nothing
        end
    end
    accum_moments!(A0,A1,X,V)
    @show typeof(A0) size(A0) strides(A0)
    @show A0 isa StridedMatrix
    @show stride(A0,1) stride(A0,2)
    @assert size(A0,1)>0 && size(A0,2)>0
    @assert stride(A0,2)>=max(1,size(A0,1))
    @assert all(isfinite,A0)
    @time "SVD(hyp)" @blas_multi_then_1 MAX_BLAS_THREADS U,Σ,W=svd!(A0;full=false)
    println("Singular values (<1e-10 tail inspection): ",Σ)
    svd_tol_eff=use_adaptive_svd_tol ? maximum(Σ)*1e-15 : svd_tol
    rk=0;@inbounds for i in eachindex(Σ)
        if Σ[i]≥svd_tol_eff;rk+=1 else;break end
    end
    rk==r && @warn "All singular values are above svd_tol=$(svd_tol_eff), r=$(r) needs to be increased"
    rk==0 && return Complex{T}[],Matrix{Complex{T}}(undef,N,0),T[]
    Uk=@view U[:,1:rk]
    Wk=@view W[:,1:rk]
    Σk=@view Σ[1:rk]
    tmp=Matrix{Complex{T}}(undef,N,rk);@blas_multi MAX_BLAS_THREADS mul!(tmp,A1,Wk)
    @inbounds for j in 1:rk
        @views tmp[:,j]./=Σk[j] 
    end
    B=Matrix{Complex{T}}(undef,rk,rk);@blas_multi MAX_BLAS_THREADS mul!(B,adjoint(Uk),tmp)
    @time "eigen(hyp)" @blas_multi_then_1 MAX_BLAS_THREADS ev=eigen!(B)
    λ=ev.values;Y=ev.vectors;Phi=Uk*Y
    keep=trues(length(λ));tens=T[];res_keep=T[]
    ybuf=Vector{Complex{T}}(undef,N);A_buf=Matrix{Complex{T}}(undef,N,N)
    dropped_out=0
    dropped_res=0
    @inbounds for j in eachindex(λ)
        d=abs(λ[j]-k0)
        if d>R;keep[j]=false;dropped_out+=1;continue end
        tab=build_QTaylorTable(ComplexF64(λ[j]);dmin=dmin,dmax=dmax,h=h,P=P,mp_dps=mp_dps,leg_type=leg_type)
        fill!(A_buf,zero(eltype(A_buf)))
        compute_kernel_matrices_DLP_hyperbolic!(A_buf,pts_eucl,solver.symmetry,tab;multithreaded=multithreaded)
        assemble_DLP_hyperbolic!(A_buf,pts_eucl)
        @blas_multi_then_1 MAX_BLAS_THREADS mul!(ybuf,A_buf,@view(Phi[:,j]))
        ybn=norm(ybuf)
        @info "k=$(λ[j]) ||A(k)v(k)|| = $(ybn) vs. res_tol $res_tol"
        if auto_discard_spurious && ybn≥res_tol
            keep[j]=false;dropped_res+=1
            if ybn>1e-8
                if ybn>1e-6;@warn "k=$(λ[j]) ||A(k)v(k)||=$(ybn) > $res_tol , definitely spurious"
                else;@warn "k=$(λ[j]) ||A(k)v(k)||=$(ybn) > $res_tol , most probably eigenvalue but too low nq" end
            else
                @warn "k=$(λ[j]) ||A(k)v(k)||=$(ybn) > $res_tol , could be spurious or increase nq"
            end
            continue
        end
        push!(tens,T(ybn));push!(res_keep,T(ybn))
    end
    kept=count(keep)
    kept>0 ? @info("STATUS(hyp): ",kept=kept,dropped_outside=dropped_out,dropped_residual=dropped_res,max_residual=maximum(res_keep)) :
             @info("STATUS(hyp): ",kept=0,dropped_outside=dropped_out,dropped_residual=dropped_res)
    return λ[keep],Phi[:,keep],tens
end

# ===============================================================================
# compute_spectrum_hyp
#
# PURPOSE
#   High-level function to compute a spectral list of hyperbolic billiard
#   eigenvalues on [k1,k2] using:
#     - window planning (plan_k_windows_hyp),
#     - boundary discretization at each window center,
#     - Beyn contour solve per window,
#     - residual/tension filtering per candidate,
#     - concatenation of results into global arrays.
#
# INPUTS
#   solver  :: BIM_hyperbolic
#   basis   :: Ba   where Ba <: AbstractHankelBasis
#   billiard:: Bi   where Bi <: AbsBilliard
#   k1,k2   :: T
#
# KEYWORD INPUTS (high level)
#   m :: Int = 10
#       Target eigenvalues per disk (passed to plan_k_windows_hyp as M=m).
#
#   Rmax, Rfloor
#       Planning bounds for disk radii.
#
# KEYWORD INPUTS (Beyn)
#   nq, r, svd_tol, res_tol, auto_discard_spurious
#
# KEYWORD INPUTS (parallelism)
#   multithreaded_matrix :: Bool
#       Used for point evaluation and kernel assembly.
#
# KEYWORD INPUTS (Q-table)
#   h, P, mp_dps, leg_type
#
# KEYWORD INPUTS (diagnostics)
#   do_INFO :: Bool = true
#       If true, runs solve_INFO_hyp on a representative disk (middle one).
#
# OUTPUTS
#   ks_all   :: Vector{T}
#       All accepted real eigenvalues (concatenated over windows).
#
#   tens_all :: Vector{T}
#       Residual norms associated with each accepted eigenvalue.
#
#   us_all   :: Vector{Vector{Complex{T}}}
#       Stored boundary vectors u (Dirichlet densities / boundary solutions),
#       one per accepted eigenvalue.
#
#   pts_all  :: Vector{BoundaryPointsHypBIM{T}}
#       Boundary discretization used for each accepted eigenvalue.
#       NOTE: repeated entries (same window’s pts reused across eigenvalues).
#
#   tensN_all :: Vector{T}
#       Normalized residuals (if computed in residual stage).
# ===============================================================================
function compute_spectrum_hyp(solver::BIM_hyperbolic,basis::Ba,billiard::Bi,k1::T,k2::T;m::Int=10,Rmax::T=T(0.8),nq::Int=64,r::Int=m+15,svd_tol::Real=1e-12,res_tol::Real=1e-9,auto_discard_spurious::Bool=true,multithreaded_matrix::Bool=true,h::T=T(1e-4),P::Int=30,mp_dps::Int=60,leg_type::Int=3,kref::T=T(1000.0),do_INFO::Bool=true,Rfloor::T=T(1e-6)) where {T<:Real,Bi<:AbsBilliard,Ba<:AbstractHankelBasis}
    @time "k-windows (hyp)" k0s,Rs=plan_k_windows_hyp(solver,billiard,k1,k2;M=m,Rmax=Rmax,Rfloor=Rfloor,kref=kref)
    idx=findall(>(max(zero(T),Rfloor)),Rs)
    k0s=isempty(idx) ? T[] : k0s[idx]
    Rs =isempty(idx) ? T[] : Rs[idx]
    nw=length(k0s);nw==0 && return T[],T[],Vector{Vector{Complex{T}}}(),Vector{BoundaryPointsHypBIM{T}}(),T[]
    println("Number of windows: ",nw);println("Average R: ",sum(Rs)/T(nw))
    all_pts=Vector{BoundaryPointsHypBIM{T}}(undef,nw)
    pre=precompute_hyperbolic_boundary_cdfs(solver,billiard;M_cdf_base=4000,safety=1e-14)
    @time "Point evaluation" @inbounds for i in 1:nw
        all_pts[i]=evaluate_points(solver,billiard,k0s[i],pre;safety=1e-14,threaded=multithreaded_matrix)
        dmin,dmax=d_bounds_hyp(all_pts[i],solver.symmetry)
        @show i k0s[i] Rs[i] length(all_pts[i].xy) dmin dmax
    end
    if do_INFO
        iinfo=cld(nw,2)
        @time "solve_INFO last disk (hyp)" begin
            _=solve_INFO_hyp(solver,basis,all_pts[iinfo],complex(k0s[iinfo],zero(T)),Rs[iinfo];multithreaded=multithreaded_matrix,nq=nq,r=r,svd_tol=svd_tol,res_tol=res_tol,rng=MersenneTwister(0),use_adaptive_svd_tol=false,auto_discard_spurious=false,h=h,P=P,mp_dps=mp_dps,leg_type=leg_type)
        end
    end
    λs=Vector{Vector{Complex{T}}}(undef,nw);Uks=Vector{Matrix{Complex{T}}}(undef,nw);Ys=Vector{Matrix{Complex{T}}}(undef,nw)
    p=Progress(nw,1)
    @time "Beyn pass (all disks) (hyp)" @inbounds for i in 1:nw
        λ,Uk,Y,_,_,_=solve_vect_hyp(solver,basis,all_pts[i],complex(k0s[i],zero(T)),Rs[i];nq=nq,r=r,svd_tol=svd_tol,res_tol=res_tol,rng=MersenneTwister(0),multithreaded=multithreaded_matrix,h=h,P=P,mp_dps=mp_dps,leg_type=leg_type)
        λs[i]=λ;Uks[i]=Uk;Ys[i]=Y
        next!(p)
    end
    ks_list=Vector{Vector{T}}(undef,nw);tens_list=Vector{Vector{T}}(undef,nw);tensN_list=Vector{Vector{T}}(undef,nw);phi_list=Vector{Matrix{Complex{T}}}(undef,nw)
    @time "Residuals/tensions pass (hyp)" @inbounds @showprogress desc="Residuals/tensions (hyp)" for i in 1:nw
        if isempty(λs[i])
            ks_list[i]=T[];tens_list[i]=T[];tensN_list[i]=T[];phi_list[i]=Matrix{Complex{T}}(undef,length(all_pts[i].xy),0);continue
        end
        idx2,Φ_kept,traw,tnorm,_=residual_and_norm_select_hyp(solver,λs[i],Uks[i],Ys[i],complex(k0s[i],zero(T)),Rs[i],all_pts[i];res_tol=T(res_tol),matnorm=:one,epss=1e-15,auto_discard_spurious=auto_discard_spurious,collect_logs=false,multithreaded=multithreaded_matrix,h=h,P=P,mp_dps=mp_dps,leg_type=leg_type)
        ks_list[i]=real.(λs[i][idx2]);tens_list[i]=traw;tensN_list[i]=tnorm;phi_list[i]=Matrix(Φ_kept)
    end
    n_by_win=Vector{Int}(undef,nw);@inbounds for i in 1:nw;n_by_win[i]=size(phi_list[i],2);end
    offs=zeros(Int,nw);@inbounds for i in 2:nw;offs[i]=offs[i-1]+n_by_win[i-1];end
    ntot=offs[end]+n_by_win[end]
    ks_all=Vector{T}(undef,ntot);tens_all=Vector{T}(undef,ntot);tensN_all=Vector{T}(undef,ntot)
    us_all=Vector{Vector{Complex{T}}}(undef,ntot);pts_all=Vector{BoundaryPointsHypBIM{T}}(undef,ntot)
    Threads.@threads for i in 1:nw
        n=n_by_win[i];n==0 && continue
        off=offs[i];ksi=ks_list[i];tr=tens_list[i];tn=tensN_list[i];Φ=phi_list[i];pts=all_pts[i]
        @inbounds for j in 1:n
            ks_all[off+j]=ksi[j];tens_all[off+j]=tr[j];tensN_all[off+j]=tn[j];us_all[off+j]=vec(@view Φ[:,j]);pts_all[off+j]=pts
        end
    end
    return ks_all,tens_all,us_all,pts_all,tensN_all
end