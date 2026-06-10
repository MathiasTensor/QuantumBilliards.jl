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
#   MO 9/6/2026
# ===============================================================================
const HyperbolicBoundarySolver=Union{BIM_hyperbolic,DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners,DLP_hyperbolic_log_product}
const HyperbolicBeynPoints=Union{BoundaryPointsHyp,DLPHypLogDiscretization}

@inline hyp_bp(pts::BoundaryPointsHyp)=pts
@inline hyp_bp(disc::DLPHypLogDiscretization)=disc.bp
@inline _hyp_precompute_points(solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners},billiard)=precompute_hyperbolic_boundary_cdfs(solver,billiard;M_cdf_base=4000,safety=1e-14)
@inline _hyp_precompute_points(solver::DLP_hyperbolic_log_product,billiard)=nothing
@inline _hyp_precompute_points(solver::BIM_hyperbolic,billiard)=precompute_hyperbolic_boundary_cdfs(solver,billiard;M_cdf_base=4000,safety=1e-14)
@inline _hyp_evaluate_points(solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners},billiard,k,pre;threaded=true)=evaluate_points(solver,billiard,k,pre;safety=1e-14,threaded=threaded)
@inline _hyp_evaluate_points(solver::DLP_hyperbolic_log_product,billiard,k,pre;threaded=true)=evaluate_points(solver,billiard,k)
@inline _hyp_evaluate_points(solver::BIM_hyperbolic,billiard,k,pre;threaded=true)=evaluate_points(solver,billiard,k,pre;safety=1e-14,threaded=threaded)

struct HypContourPrecomp{T,W}
    zj::Vector{ComplexF64}
    wj::Vector{Complex{T}}
    ws::Vector{W}
end

_hyp_contour_cache(solver,pts)=nothing
_hyp_contour_cache(solver::DLP_hyperbolic_log_product,pts::DLPHypLogDiscretization)=build_dlp_hyp_log_geom_cache(solver,pts)
function _hyp_contour_cache(solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners},pts)
    return build_dlp_hyperbolic_kress_geom_workspace(solver,pts)
end

function _hyp_contour_workspace(solver::DLP_hyperbolic_log_product,pts::DLPHypLogDiscretization,k,cache;mp_dps::Int=80,leg_type::Int=3)
    return build_dlp_hyp_log_workspace(solver,pts,cache,k)
end

function _hyp_contour_workspace(solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners},pts,k,gws;mp_dps::Int=80,leg_type::Int=3)
    kws=build_dlp_hyperbolic_kress_k_workspace(solver,pts,k;mp_dps=mp_dps,leg_type=leg_type)
    return gws,kws
end

@inline _hyp_contour_workspace(solver::BIM_hyperbolic,pts::BoundaryPointsHyp,k,cache;mp_dps::Int=80,leg_type::Int=3)=nothing

function precompute_hyp_contour(solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners,DLP_hyperbolic_log_product},pts::HyperbolicBeynPoints,k0::Complex{T},R::T;nq::Int=64,mp_dps::Int=80,leg_type::Int=3) where {T<:Real}
    θ=(TWO_PI/nq).*(collect(0:nq-1).+T(0.5))
    ej=cis.(θ)
    zj=ComplexF64.(k0.+R.*ej)
    wj=Complex{T}.((R/nq).*ej)
    cache=_hyp_contour_cache(solver,pts)
    ws1=_hyp_contour_workspace(solver,pts,zj[1],cache;mp_dps=mp_dps,leg_type=leg_type)
    ws=Vector{typeof(ws1)}(undef,nq)
    ws[1]=ws1
    @inbounds for q in 2:nq
        ws[q]=_hyp_contour_workspace(solver,pts,zj[q],cache;mp_dps=mp_dps,leg_type=leg_type)
    end
    return HypContourPrecomp{T,typeof(ws1)}(zj,wj,ws)
end

function construct_boundary_matrices_precomputed!(Tbufs::Vector{Matrix{ComplexF64}},solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners,DLP_hyperbolic_log_product},pts::HyperbolicBeynPoints,pc::HypContourPrecomp;multithreaded::Bool=true,timeit::Bool=false,adjoint_mode::Symbol=:direct)
    @blas_1 begin
        @inbounds for q in eachindex(pc.zj)
            n=_workspace_dim(pc.ws[q])
            @assert size(Tbufs[q])==(n,n) "Tbufs[$q] has size $(size(Tbufs[q])), but solver requires ($n,$n)."
            fill!(Tbufs[q],0.0+0.0im)
            @benchit timeit=timeit "hyp precomputed assembly" construct_matrices!(solver,Tbufs[q],pts,pc.ws[q],pc.zj[q];multithreaded=multithreaded,adjoint_mode=adjoint_mode)
        end
    end
    return nothing
end

function construct_boundary_matrices_precomputed!(Tbufs::Vector{Matrix{ComplexF64}},solver::BIM_hyperbolic,pts::HyperbolicBeynPoints,pc;multithreaded::Bool=true,timeit::Bool=false,adjoint_mode::Symbol=:direct)
    construct_boundary_matrices!(Tbufs,solver,pts,pc.zj;multithreaded=multithreaded,timeit=timeit,adjoint_mode=adjoint_mode)
    return nothing
end

function precompute_hyp_contour(solver::BIM_hyperbolic,pts::HyperbolicBeynPoints,k0::Complex{T},R::T;nq::Int=64,mp_dps::Int=80,leg_type::Int=3) where {T<:Real}
    θ=(TWO_PI/nq).*(collect(0:nq-1).+T(0.5))
    ej=cis.(θ)
    zj=ComplexF64.(k0.+R.*ej)
    wj=Complex{T}.((R/nq).*ej)
    return HypContourPrecomp{T,Nothing}(zj,wj,Nothing[])
end

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
function plan_k_windows_hyp(solver::HyperbolicBoundarySolver,billiard::Bi,k1::T,k2::T;M::Int=50,Rmax::T=T(0.8),Rfloor::T=T(1e-6),kref::T=T(1000),tolA::Real=1e-8,iters::Int=8,origin_vertex_tol::Real=0.0) where {Bi<:AbsBilliard,T<:Real}
    L=k2-k1
    (L<=zero(T)||Rmax<=zero(T))&&return T[],T[]
    A=hyperbolic_area_fundamental(solver,billiard;tol=tolA,kref=kref,origin_vertex_tol=origin_vertex_tol)
    ρA(k)=max((A/TWO_PI)*k,T(1e-12))
    Rof(k)=clamp(M/(2*ρA(k)),Rfloor,Rmax)
    k0s=T[]
    Rs=T[]
    left=k1
    while left<k2-T(10)*eps(k2)
        rem=k2-left
        rem<=zero(T)&&break
        R=clamp(rem/2,Rfloor,Rmax)
        @inbounds for _ in 1:iters
            k0=left+R
            R=min(Rof(k0),rem/2)
            R=clamp(R,Rfloor,Rmax)
        end
        push!(k0s,left+R)
        push!(Rs,R)
        left+=2R
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
#   pts :: HyperbolicBeynPoints
#       Boundary discretization in the hyperbolic setting.
#       NOTE: you convert to Euclidean BoundaryPoints for kernel assembly.
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
function construct_B_matrix_hyp(solver::HyperbolicBoundarySolver,pts::HyperbolicBeynPoints,N::Int,pc::HypContourPrecomp;r::Int=48,svd_tol=1e-14,rng=MersenneTwister(0),multithreaded::Bool=true,timeit::Bool=false,adjoint_mode::Symbol=:direct)
    nq=length(pc.zj)
    zj=pc.zj
    wj=pc.wj
    Tbufs=[zeros(ComplexF64,N,N) for _ in 1:nq]
    construct_boundary_matrices_precomputed!(Tbufs,solver,pts,pc;multithreaded=multithreaded,timeit=timeit,adjoint_mode=adjoint_mode)
    @blas_multi MAX_BLAS_THREADS F1=lu!(Tbufs[1];check=false)
    Fs=Vector{typeof(F1)}(undef,nq);Fs[1]=F1
    @blas_multi_then_1 MAX_BLAS_THREADS @inbounds for j in 2:nq
        Fs[j]=lu!(Tbufs[j];check=false)
    end
    function accum_moments!(A0,A1,X,V)
        xv=reshape(X,:);a0v=reshape(A0,:);a1v=reshape(A1,:)
        @blas_multi_then_1 MAX_BLAS_THREADS @inbounds for j in 1:nq
            ldiv!(X,Fs[j],V)
            BLAS.axpy!(wj[j],xv,a0v)
            BLAS.axpy!(wj[j]*zj[j],xv,a1v)
        end
        return nothing
    end
    T=real(eltype(wj))
    N=size(Tbufs[1],1)
    V,X,A0,A1=beyn_buffer_matrices(T,N,r,rng)
    accum_moments!(A0,A1,X,V)
    @blas_multi_then_1 MAX_BLAS_THREADS U,Σ,W=svd!(A0;full=false)
    rk=count(>=(svd_tol),Σ)
    rk==0&&return Matrix{Complex{T}}(undef,N,0),Matrix{Complex{T}}(undef,N,0)
    Uk=@view U[:,1:rk]
    Wk=@view W[:,1:rk]
    Σk=@view Σ[1:rk]
    tmp=Matrix{Complex{T}}(undef,N,rk)
    @blas_multi_then_1 MAX_BLAS_THREADS mul!(tmp,A1,Wk)
    @inbounds for j in 1:rk
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
#   pts    :: HyperbolicBeynPoints
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
function solve_vect_hyp(solver::HyperbolicBoundarySolver,basis::Ba,pts::HyperbolicBeynPoints,pc::HypContourPrecomp;r::Int=48,svd_tol::Real=1e-14,rng=MersenneTwister(0),multithreaded::Bool=true,timeit::Bool=false,adjoint_mode::Symbol=:direct) where {Ba<:AbstractHankelBasis}
    N=_hyp_beyn_dim(solver,pts,pc.zj[1])
    B,Uk=construct_B_matrix_hyp(solver,pts,N,pc;r=r,svd_tol=svd_tol,rng=rng,multithreaded=multithreaded,timeit=timeit,adjoint_mode=adjoint_mode)
    isempty(B)&&return ComplexF64[],Uk,Matrix{ComplexF64}(undef,0,0),pc.zj[1],zero(real(eltype(pc.wj))),pts
    @blas_multi_then_1 MAX_BLAS_THREADS λ,Y=eigen!(B)
    return λ,Uk,Y,pc.zj[1],zero(real(eltype(pc.wj))),pts
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
@inline function solve_hyp(solver::HyperbolicBoundarySolver,basis::Ba,pts::HyperbolicBeynPoints,pc::HypContourPrecomp;r::Int=48,svd_tol::Real=1e-14,rng=MersenneTwister(0),multithreaded::Bool=true,timeit::Bool=false,adjoint_mode::Symbol=:direct) where {Ba<:AbstractHankelBasis}
    λ,_,_,_,_,_=solve_vect_hyp(solver,basis,pts,pc;r=r,svd_tol=svd_tol,rng=rng,multithreaded=multithreaded,timeit=timeit,adjoint_mode=adjoint_mode)
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
#   pts :: HyperbolicBeynPoints
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
function residual_and_norm_select_hyp(solver::HyperbolicBoundarySolver,λ::AbstractVector{Complex{T}},Uk::AbstractMatrix{Complex{T}},Y::AbstractMatrix{Complex{T}},k0::Complex{T},R::T,pts::HyperbolicBeynPoints;res_tol::T,matnorm::Symbol=:one,epss::Real=1e-15,auto_discard_spurious::Bool=true,collect_logs::Bool=false,multithreaded::Bool=true,timeit::Bool=false,adjoint_mode::Symbol=:direct) where {T<:Real}
    cache=_hyp_contour_cache(solver,pts)
    N,rk=size(Uk)
    Φtmp=Matrix{Complex{T}}(undef,N,rk)
    y=Vector{Complex{T}}(undef,N)
    keep=falses(rk)
    tens=Vector{T}(undef,rk)
    tensN=Vector{T}(undef,rk)
    logs=collect_logs ? String[] : nothing
    A_buf=zeros(ComplexF64,N,N)
    vecnorm=matnorm===:one ? (v->norm(v,1)) : matnorm===:two ? (v->norm(v)) : (v->norm(v,Inf))
    @inbounds for j in 1:rk
        λj=λ[j]
        abs(λj-k0)>R&&(tens[j]=T(NaN);tensN[j]=T(NaN);continue)
        @blas_multi_then_1 MAX_BLAS_THREADS mul!(@view(Φtmp[:,j]),Uk,@view(Y[:,j]))
        wsj=_hyp_contour_workspace(solver,pts,ComplexF64(λj),cache)
        construct_matrices!(solver,A_buf,pts,wsj,λj;multithreaded=multithreaded,adjoint_mode=adjoint_mode)
        @blas_multi_then_1 MAX_BLAS_THREADS mul!(y,A_buf,@view(Φtmp[:,j]))
        rj=norm(y)
        tens[j]=rj
        nA=matnorm===:one ? opnorm(A_buf,1) : matnorm===:two ? opnorm(A_buf,2) : opnorm(A_buf,Inf)
        φn=vecnorm(@view(Φtmp[:,j]))
        yn=vecnorm(y)
        tensN[j]=yn/(nA*(φn+epss)+epss)
        if auto_discard_spurious&&rj>=res_tol
            collect_logs&&push!(logs,"λ=$(λj) ||Aφ||=$(rj) > $res_tol → DROP")
        else
            keep[j]=true
            collect_logs&&push!(logs,"λ=$(λj) ||Aφ||=$(rj) < $res_tol ← KEEP")
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
function solve_INFO_hyp(solver::HyperbolicBoundarySolver,basis::Ba,pts::HyperbolicBeynPoints,k0::Complex{T},R::T;multithreaded::Bool=true,nq::Int=64,r::Int=48,svd_tol::Real=1e-10,res_tol::Real=1e-10,rng=MersenneTwister(0),use_adaptive_svd_tol::Bool=false,auto_discard_spurious::Bool=false,timeit::Bool=false,adjoint_mode::Symbol=:direct) where {Ba<:AbstractHankelBasis,T<:Real}
    N=_hyp_beyn_dim(solver,pts,k0)
    θ=(TWO_PI/nq).*(collect(0:nq-1).+T(0.5))
    ej=cis.(θ)
    zj=ComplexF64.(k0.+R.*ej)
    wj=Complex{T}.((R/nq).*ej)
    V,X,A0,A1=beyn_buffer_matrices(T,N,r,rng)
    @info "beyn:start(hyp)" k0=k0 R=R nq=nq N=N r=r adjoint_mode=adjoint_mode
    Tbufs=[zeros(ComplexF64,N,N) for _ in 1:nq]
    @time "Boundary matrices (hyp)" construct_boundary_matrices!(Tbufs,solver,pts,zj;multithreaded=multithreaded,timeit=timeit,adjoint_mode=adjoint_mode)
    @blas_multi MAX_BLAS_THREADS F1=lu!(Tbufs[1];check=false)
    Fs=Vector{typeof(F1)}(undef,nq)
    Fs[1]=F1
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
                BLAS.axpy!(wj[j]*Complex{T}(zj[j]),xv,a1v)
            end
        end
        return nothing
    end
    accum_moments!(A0,A1,X,V)
    @show typeof(A0) size(A0) strides(A0)
    @show A0 isa StridedMatrix
    @show stride(A0,1) stride(A0,2)
    @assert size(A0,1)>0&&size(A0,2)>0
    @assert stride(A0,2)>=max(1,size(A0,1))
    @assert all(isfinite,A0)
    @time "SVD(hyp)" @blas_multi_then_1 MAX_BLAS_THREADS U,Σ,W=svd!(A0;full=false)
    println("Singular values (<1e-10 tail inspection): ",Σ)
    svd_tol_eff=use_adaptive_svd_tol ? maximum(Σ)*1e-15 : svd_tol
    rk=0
    @inbounds for i in eachindex(Σ)
        if Σ[i]>=svd_tol_eff
            rk+=1
        else
            break
        end
    end
    rk==r&&@warn "All singular values are above svd_tol=$(svd_tol_eff), r=$(r) needs to be increased"
    rk==0&&return Complex{T}[],Matrix{Complex{T}}(undef,N,0),T[]
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
    @time "eigen(hyp)" @blas_multi_then_1 MAX_BLAS_THREADS ev=eigen!(B)
    λ=ev.values
    Y=ev.vectors
    Phi=Uk*Y
    keep=trues(length(λ))
    tens=T[]
    res_keep=T[]
    ybuf=Vector{Complex{T}}(undef,N)
    A_buf=zeros(ComplexF64,N,N)
    dropped_out=0
    dropped_res=0
    cache=_hyp_contour_cache(solver,pts)
    @inbounds for j in eachindex(λ)
        if abs(λ[j]-k0)>R
            keep[j]=false
            dropped_out+=1
            continue
        end
        wsj=_hyp_contour_workspace(solver,pts,ComplexF64(λ[j]),cache)
        construct_matrices!(solver,A_buf,pts,wsj,λ[j];multithreaded=multithreaded,adjoint_mode=adjoint_mode)
        @blas_multi_then_1 MAX_BLAS_THREADS mul!(ybuf,A_buf,@view(Phi[:,j]))
        ybn=norm(ybuf)
        @info "k=$(λ[j]) ||A(k)v(k)|| = $(ybn) vs. res_tol $res_tol" adjoint_mode=adjoint_mode
        if auto_discard_spurious&&ybn>=res_tol
            keep[j]=false
            dropped_res+=1
            if ybn>1e-8
                ybn>1e-6 ? @warn("k=$(λ[j]) ||A(k)v(k)||=$(ybn) > $res_tol , definitely spurious") : @warn("k=$(λ[j]) ||A(k)v(k)||=$(ybn) > $res_tol , most probably eigenvalue but too low nq")
            else
                @warn "k=$(λ[j]) ||A(k)v(k)||=$(ybn) > $res_tol , could be spurious or increase nq"
            end
            continue
        end
        push!(tens,T(ybn))
        push!(res_keep,T(ybn))
    end
    kept=count(keep)
    kept>0 ? @info("STATUS(hyp): ",kept=kept,dropped_outside=dropped_out,dropped_residual=dropped_res,max_residual=maximum(res_keep),adjoint_mode=adjoint_mode) : @info("STATUS(hyp): ",kept=0,dropped_outside=dropped_out,dropped_residual=dropped_res,adjoint_mode=adjoint_mode)
    return λ[keep],Phi[:,keep],tens
end

# Experimental imag-tail residual filter for hyperbolic Beyn solves.
#
# Motivation:
#   Genuine eigenvalues typically have very small |Im λ|, while spurious Beyn
#   roots tend to populate the large-|Im λ| tail. Instead of residual-checking
#   every candidate, we sort all contour-inside roots by decreasing |Im λ|,
#   residual-check only this suspicious tail, and stop once `pad`
#   consecutive candidates pass the residual test.
#
# Inputs:
#   - solver::HyperbolicBoundarySolver
#   - λs,Uks,Ys : Beyn outputs for all windows
#   - k0s,Rs    : contour centers/radii
#   - all_pts   : discretizations used in each window
#   - res_tol   : residual threshold ||A(λ)φ||
#
# Keywords:
#   - pad::Int=20         : stop after this many consecutive good candidates
#   - group_size::Int=64  : batch size in global |Im λ| ordering
#   - multithreaded=true  : threaded matrix assembly
#   - verbose=true        : print diagnostics
#   - mp_dps,leg_type     : forwarded to hyperbolic workspaces
#
# Returns:
#   idx_keep  :: Vector{Vector{Int}}
#       Indices of accepted eigenvalues in each window.
#
#   residuals :: Vector{Vector{T}}
#       Residuals aligned with idx_keep. Unchecked roots remain NaN.
#
# Notes:
#   - Reuses one geometry cache per window.
#   - Works for all hyperbolic solvers implementing:
#         _hyp_contour_cache
#         _hyp_contour_workspace
#         construct_matrices!
#   - Experimental heuristic: validate against full residual checks when
#     changing geometry, discretization, or quadrature parameters.
function imag_k_check_hyp_EXPERIMENTAL(solver::HyperbolicBoundarySolver,λs::Vector{Vector{Complex{T}}},Uks::Vector{Matrix{Complex{T}}},Ys::Vector{Matrix{Complex{T}}},k0s::Vector{Complex{T}},Rs::Vector{T},all_pts;res_tol::T,pad::Int=20,group_size::Int=64,multithreaded::Bool=true,verbose::Bool=true,mp_dps::Int=80,leg_type::Int=3,adjoint_mode::Symbol=:direct) where {T<:Real}
    nw=length(λs)
    idx_inside=Vector{Vector{Int}}(undef,nw)
    idx_keep=Vector{Vector{Int}}(undef,nw)
    residuals=Vector{Vector{T}}(undef,nw)
    local_pos=Dict{Tuple{Int,Int},Int}()
    candidates=Tuple{Int,Int,T,T}[]
    @inbounds for i in 1:nw
        λi=λs[i]
        idx_inside[i]=findall(j->abs(λi[j]-k0s[i])<=Rs[i],eachindex(λi))
        idx_keep[i]=copy(idx_inside[i])
        residuals[i]=fill(T(NaN),length(idx_inside[i]))
        for (lp,j) in pairs(idx_inside[i])
            local_pos[(i,j)]=lp
            push!(candidates,(i,j,abs(imag(λi[j])),real(λi[j])))
        end
    end
    sort!(candidates;by=c->c[3],rev=true)
    verbose&&!isempty(candidates)&&@info "top hyp imag candidates" first_candidates=candidates[1:min(10,length(candidates))]
    caches=Vector{Any}(fill(nothing,nw))
    drop=Dict{Tuple{Int,Int},Bool}()
    checked=0
    dropped=0
    good_streak=0
    pos=1
    A=Matrix{Complex{T}}(undef,0,0)
    y=Vector{Complex{T}}(undef,0)
    φ=Vector{Complex{T}}(undef,0)
    while pos<=length(candidates)
        stop=min(pos+group_size-1,length(candidates))
        group=@view candidates[pos:stop]
        rdict=Dict{Tuple{Int,Int},T}()
        for iwin in unique(c[1] for c in group)
            sub=[c for c in group if c[1]==iwin]
            isempty(sub)&&continue
            pts=all_pts[iwin]
            caches[iwin]===nothing&&(caches[iwin]=_hyp_contour_cache(solver,pts))
            cache=caches[iwin]
            N=_hyp_beyn_dim(solver,pts,k0s[iwin])
            size(A)!=(N,N)&&(A=Matrix{Complex{T}}(undef,N,N))
            length(y)!=N&&(y=Vector{Complex{T}}(undef,N))
            length(φ)!=N&&(φ=Vector{Complex{T}}(undef,N))
            @inbounds for c in sub
                i,j,_,_=c
                λj=λs[i][j]
                wsj=_hyp_contour_workspace(solver,pts,ComplexF64(λj),cache;mp_dps=mp_dps,leg_type=leg_type)
                fill!(A,zero(Complex{T}))
                construct_matrices!(solver,A,pts,wsj,λj;multithreaded=multithreaded,adjoint_mode=adjoint_mode)
                mul!(φ,Uks[i],@view Ys[i][:,j])
                mul!(y,A,φ)
                rdict[(i,j)]=norm(y)
            end
        end
        @inbounds for c in group
            i,j,imj,_=c
            rj=rdict[(i,j)]
            checked+=1
            residuals[i][local_pos[(i,j)]]=rj
            if rj>=res_tol
                drop[(i,j)]=true
                dropped+=1
                good_streak=0
                verbose&&@info "DROP hyp candidate" i=i j=j k=λs[i][j] abs_imag=imj residual=rj
            else
                good_streak+=1
                if good_streak>=pad
                    verbose&&@info "hyp imag-tail check stopped" checked=checked dropped=dropped good_streak=good_streak last_imag=imj last_residual=rj
                    break
                end
            end
        end
        good_streak>=pad&&break
        pos=stop+1
    end
    @inbounds for i in 1:nw
        old=idx_inside[i]
        mask=[!get(drop,(i,j),false) for j in old]
        idx_keep[i]=old[mask]
        residuals[i]=residuals[i][mask]
    end
    verbose&&@info "hyp imag-tail summary" checked=checked dropped=dropped total_candidates=length(candidates)
    return idx_keep,residuals
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
#   pts_all  :: Vector{BoundaryPointsHyp{T}}
#       Boundary discretization used for each accepted eigenvalue.
#       NOTE: repeated entries (same window’s pts reused across eigenvalues).
#
#   tensN_all :: Vector{T}
#       Normalized residuals (if computed in residual stage).
# ===============================================================================
function compute_spectrum_hyp(solver::HyperbolicBoundarySolver,basis::Ba,billiard::Bi,k1::T,k2::T;m::Int=10,Rmax::T=T(0.8),nq::Int=64,r::Int=m+15,svd_tol::Real=1e-12,res_tol::Real=1e-9,auto_discard_spurious::Bool=true,multithreaded_matrix::Bool=true,kref::T=T(1000.0),do_INFO::Bool=true,Rfloor::T=T(1e-6),timeit::Bool=false,return_imag_part::Bool=true,origin_vertex_tol::Real=0.0,use_imag_residual_check::Bool=true,adjoint_mode::Symbol=:direct) where {T<:Real,Bi<:AbsBilliard,Ba<:AbstractHankelBasis}
    @time "k-windows (hyp)" k0s,Rs=plan_k_windows_hyp(solver,billiard,k1,k2;M=m,Rmax=Rmax,Rfloor=Rfloor,kref=kref,origin_vertex_tol=origin_vertex_tol)
    idx=findall(>(max(zero(T),Rfloor)),Rs)
    k0s=isempty(idx) ? T[] : k0s[idx]
    Rs=isempty(idx) ? T[] : Rs[idx]
    nw=length(k0s)
    nw==0&&return (return_imag_part ? Complex{T}[] : T[]),T[],Vector{Vector{Complex{T}}}(),Vector{HyperbolicBeynPoints}(),T[]
    println("Number of windows: ",nw)
    println("Average R: ",sum(Rs)/T(nw))
    pre=_hyp_precompute_points(solver,billiard)
    first_pts=_hyp_evaluate_points(solver,billiard,k0s[1],pre;threaded=multithreaded_matrix)
    PtsT=typeof(first_pts)
    all_pts=Vector{PtsT}(undef,nw)
    all_pts[1]=first_pts
    @time "Point evaluation" begin
        @showprogress desc="pts construction" Threads.@threads for i in 1:nw
            i>1&&(all_pts[i]=_hyp_evaluate_points(solver,billiard,k0s[i],pre;threaded=multithreaded_matrix))
            bp=hyp_bp(all_pts[i])
            dmin,dmax=d_bounds_hyp(bp,solver.symmetry)
        end
    end
    if do_INFO
        iinfo=cld(nw,2)
        @time "solve_INFO middle disk (hyp)" begin
            _=solve_INFO_hyp(solver,basis,all_pts[iinfo],complex(k0s[iinfo],zero(T)),Rs[iinfo];multithreaded=multithreaded_matrix,nq=nq,r=r,svd_tol=svd_tol,res_tol=res_tol,rng=MersenneTwister(0),use_adaptive_svd_tol=false,auto_discard_spurious=false,timeit=timeit,adjoint_mode=adjoint_mode)
        end
    end
    λs=Vector{Vector{Complex{T}}}(undef,nw)
    Uks=Vector{Matrix{Complex{T}}}(undef,nw)
    Ys=Vector{Matrix{Complex{T}}}(undef,nw)
    p=Progress(nw,1)
    @time "Beyn pass (all disks) (hyp)" @inbounds for i in 1:nw
        pc=precompute_hyp_contour(solver,all_pts[i],complex(k0s[i],zero(T)),Rs[i];nq=nq)
        λ,Uk,Y,_,_,_=solve_vect_hyp(solver,basis,all_pts[i],pc;r=r,svd_tol=svd_tol,rng=MersenneTwister(0),multithreaded=multithreaded_matrix,timeit=timeit,adjoint_mode=adjoint_mode)
        λs[i]=λ
        Uks[i]=Uk
        Ys[i]=Y
        pc=nothing
        next!(p)
    end
    ks_list=Vector{Vector{Complex{T}}}(undef,nw)
    tens_list=Vector{Vector{T}}(undef,nw)
    tensN_list=Vector{Vector{T}}(undef,nw)
    phi_list=Vector{Matrix{Complex{T}}}(undef,nw)
    if use_imag_residual_check
        idx_keep,residuals=imag_k_check_hyp_EXPERIMENTAL(solver,λs,Uks,Ys,complex.(k0s,zero(T)),Rs,all_pts;res_tol=T(res_tol),pad=20,group_size=64,multithreaded=multithreaded_matrix,verbose=timeit,adjoint_mode=adjoint_mode)
        @time "Imag-check selection pass (hyp)" @inbounds @showprogress desc="Imag-check selection (hyp)" for i in 1:nw
            idx=idx_keep[i]
            if isempty(idx)
                ks_list[i]=Complex{T}[]
                tens_list[i]=T[]
                tensN_list[i]=T[]
                bp=hyp_bp(all_pts[i])
                phi_list[i]=Matrix{Complex{T}}(undef,length(bp.xy),0)
                continue
            end
            N=size(Uks[i],1)
            Φ_kept=Matrix{Complex{T}}(undef,N,length(idx))
            @inbounds for (jj,j) in pairs(idx)
                mul!(@view(Φ_kept[:,jj]),Uks[i],@view(Ys[i][:,j]))
            end
            ks_list[i]=λs[i][idx]
            tens_list[i]=abs.(imag.(λs[i][idx]))
            tensN_list[i]=residuals[i]
            phi_list[i]=Φ_kept
        end
    else
        @time "Residuals/tensions pass (hyp)" @inbounds @showprogress desc="Residuals/tensions (hyp)" for i in 1:nw
            if isempty(λs[i])
                ks_list[i]=Complex{T}[]
                tens_list[i]=T[]
                tensN_list[i]=T[]
                bp=hyp_bp(all_pts[i])
                phi_list[i]=Matrix{Complex{T}}(undef,length(bp.xy),0)
                continue
            end
            idx2,Φ_kept,traw,tnorm,_=residual_and_norm_select_hyp(solver,λs[i],Uks[i],Ys[i],complex(k0s[i],zero(T)),Rs[i],all_pts[i];res_tol=T(res_tol),matnorm=:one,epss=1e-15,auto_discard_spurious=auto_discard_spurious,collect_logs=false,multithreaded=multithreaded_matrix,timeit=timeit,adjoint_mode=adjoint_mode)
            ks_list[i]=λs[i][idx2]
            tens_list[i]=traw
            tensN_list[i]=tnorm
            phi_list[i]=Matrix(Φ_kept)
        end
    end
    GC.gc()
    GC.gc()
    n_by_win=Vector{Int}(undef,nw)
    @inbounds for i in 1:nw
        n_by_win[i]=size(phi_list[i],2)
    end
    offs=zeros(Int,nw)
    @inbounds for i in 2:nw
        offs[i]=offs[i-1]+n_by_win[i-1]
    end
    ntot=offs[end]+n_by_win[end]
    ks_all=Vector{Complex{T}}(undef,ntot)
    tens_all=Vector{T}(undef,ntot)
    tensN_all=Vector{T}(undef,ntot)
    us_all=Vector{Vector{Complex{T}}}(undef,ntot)
    pts_all=Vector{PtsT}(undef,ntot)
    Threads.@threads for i in 1:nw
        n=n_by_win[i]
        n==0&&continue
        off=offs[i]
        ksi=ks_list[i]
        tr=tens_list[i]
        tn=tensN_list[i]
        Φ=phi_list[i]
        pts=all_pts[i]
        @inbounds for j in 1:n
            ks_all[off+j]=ksi[j]
            tens_all[off+j]=tr[j]
            tensN_all[off+j]=tn[j]
            us_all[off+j]=vec(@view Φ[:,j])
            pts_all[off+j]=pts
        end
    end
    ks_out=return_imag_part ? ks_all : real.(ks_all)
    return ks_out,tens_all,us_all,pts_all,tensN_all
end