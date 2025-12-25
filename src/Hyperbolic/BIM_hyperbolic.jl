# ==============================================================================
# HYPERBOLIC BIM (POINCARÉ DISK)
# ==============================================================================
#
# GOAL
#   Compute Dirichlet eigen-wavenumbers k of the hyperbolic Laplace–Beltrami
#   operator on a billiard Ω embedded in the Poincaré disk, and extract the
#   boundary density u(s)=∂_n ψ(s) required for:
#     • interior wavefunction reconstruction ψ(x,y) via SLP
#     • Husimi and Poincaré–Husimi constructions
#
# MODEL
#   Domain: Ω ⊂ { (x,y): x^2+y^2 < 1 } with Euclidean coordinates (x,y).
#   Metric: Poincaré disk metric
#       ds_H = λ(x,y) ds_E,    λ(x,y) = 2 / (1 - (x^2 + y^2)).
#
# PDE (DIRICHLET EIGENPROBLEM)
#   Find nontrivial ψ and k such that
#       (Δ_H + (k^2 + 1/4)) ψ = 0   in Ω  
#       ψ = 0  on ∂Ω
#
#   The Green kernel is parameterized by ν=-1/2+i k through Q_ν(cosh d), with d
#   the hyperbolic distance in the disk model. The exact mapping is determined
#   by QTaylorTable / Q_ν evaluation routines.
#
# BOUNDARY INTEGRAL FORMULATION (DLP SINGULARITY TEST)
#   Let G_k(x,q) be the hyperbolic Green kernel:
#       G_k(x,q) = (1/2π) Q_ν(cosh d(x,q)),   ν = -1/2 + i k.
#
#   A standard Dirichlet representation uses a double-layer potential:
#       ψ(x) = ∮_{∂Ω} ∂_{n_q} G_k(x,q)  ϕ(q)  ds_H(q),
#
#   and boundary limiting gives a Fredholm equation:
#       (±1/2 I + K_k) ϕ = 0,
#   assembled as a dense matrix K(k) after the package’s diagonal/limit
#   conventions (principal value, diagonal correction, ...).
#
#   Eigen-wavenumbers are detected by the smallest singular value:
#       σ_min(K(k)) ≈ 0  ⇔  k near an eigen-wavenumber.
#
#   For wavefunctions/Husimis, the required boundary data is u(s)=∂_n ψ(s).
#   In this workflow, the LEFT singular vector associated with σ_min(K(k)) is
#   taken as the discrete representation of u(s) (up to scaling), then normalized
#   in the hyperbolic boundary inner product (see Normalization).
#
# WHAT IS COMPUTED
#   • k_sweep : σ_min(k) over a k-grid (for root finding / eigenvalue extraction)
#   • solve   : σ_min(k) at a single k (probe)
#   • solve_vect : (σ_min(k), u_left) where u_left is normalized with ds_H weights
#
# QUADRATURE + DISCRETIZATION
#   Boundary nodes are distributed uniformly in HYPERBOLIC arclength (to satisfy Nystrom oversampling homogenously) on each real
#   boundary curve r(t), t∈[0,1]:
#       dℓ_H = λ(r(t)) |r'(t)| dt.
#
#   Construction strategy:
#     1) Precompute a dense CDF per curve using trapezoidal integration:
#          F(t) = (∫_0^t λ(r(τ))|r'(τ)| dτ) / (∫_0^1 λ(r(τ))|r'(τ)| dτ).
#     2) Invert the monotone CDF at midpoint targets u_i=(i-0.5)/N via
#        piecewise-linear interpolation to obtain parameter nodes t_i.
#     3) Form Euclidean weights and hyperbolic weights:
#          ds_E[i] = |r'(t_i)| Δt_i,
#          ds_H[i] = λ(r(t_i)) ds_E[i].
#
#   Geometric quantities stored for BIM assembly are Euclidean by design:
#     • normal[i]    : Euclidean outward unit normal
#     • curvature[i] : Euclidean curvature κ_E
#   Hyperbolic measure is incorporated via dsH (and optionally in kernels).
#
# NORMALIZATION (CRITICAL)
#   The extracted boundary vector u is normalized in the hyperbolic boundary L2
#   inner product:
#       ∥u∥^2_{∂Ω,H}  ≈  Σ_i |u_i|^2 ds_H[i].
#   Enforce:
#       Σ_i |u_i|^2 ds_H[i] = 1,
#   ensuring consistent scaling for wavefunction reconstruction and Husimi
#   integrals across k.
#
# SYMMETRY
#   Only origin-based discrete symmetries are supported in the Poincaré disk
#   setting (rotations about the origin; reflections about x-axis, y-axis, or
#   through the origin). Symmetry affects distance bounds via
#   estimate_rmin_rmax(pts, solver.symmetry).
#
# PERFORMANCE MODEL
#   For each k:
#     • Build/update Q-table for this k
#     • Assemble dense N×N DLP matrix K(k) (dominant O(N^2))
#     • Extract σ_min(K(k)) via:
#         - KrylovKit svdsolve (recommended), or
#         - full SVD/svdvals (expensive)
#
# RECOMMENDED SETTINGS
#   • use_krylov=true for sweeps and for solve_vect (left vector required)
#   • h=1e-4, P=30 are robust defaults for QTaylorTable resolution
#   • M_cdf_base≈4000 for stable hyperbolic-uniform boundary sampling
#
# ==============================================================================

struct BIM_hyperbolic{T} <: SweepSolver where {T<:Real}
    dim_scaling_factor::T
    pts_scaling_factor::Vector{T}
    sampler::Vector
    eps::T
    min_dim::Int64 
    min_pts::Int64
    symmetry::Union{Vector{Any},Nothing}
end

#------------------------------------------------------------------------------
# BIM_hyperbolic(pts_scaling_factor;min_pts=20,symmetry=nothing)->solver
#
# PURPOSE
#   Convenience constructor. Accepts either a scalar b or a vector of b_i.
#
# INPUTS
#   pts_scaling_factor
#     Either:
#       • b::T              -> stored as Vector{T}([b])
#       • bs::Vector{T}     -> stored directly
#
# KEYWORDS
#   min_pts::Int
#     Minimum points per real curve.
#   symmetry::Union{Vector{Any},Nothing}
#     Symmetry specification forwarded into solver.symmetry.
#
# OUTPUTS
#   solver::BIM_hyperbolic{T}
#------------------------------------------------------------------------------
function BIM_hyperbolic(pts_scaling_factor::Union{T,Vector{T}};min_pts=20,symmetry::Union{Vector{Any},Nothing}=nothing) where {T<:Real}
    bs=typeof(pts_scaling_factor)==T ? [pts_scaling_factor] : pts_scaling_factor
    sampler=[Hyperbolic()]
    return BIM_hyperbolic{T}(1.0,bs,sampler,eps(T),min_pts,min_pts,symmetry)
end

# ==============================================================================
# BoundaryPointsHypBIM
# ==============================================================================
#
#   • xy[j]        : Euclidean coordinates in unit disk.
#   • normal[j]    : Euclidean outward unit normal (what BIM needs everywhere).
#   • curvature[j] : Euclidean curvature κ_E(s) of the boundary curve.
#   • ds[j]        : Euclidean quadrature weight (ds_E) at node j.
#
#   • λ[j]         : Poincaré conformal factor at node j:
#                     λ(x,y)=2/(1-(x^2+y^2)).
#   • dsH[j]       : Hyperbolic quadrature weight at node j:
#                     dsH[j]=λ[j]*ds[j]  (i.e. ds_H = λ ds_E).
#   • ξ[j]         : Cumulative hyperbolic arclength coordinate along the
#                     concatenated boundary nodes (same ordering as xy).
#                     Typically ξ[1]=0 and ξ[end]≈LH.
#   • LH           : Total hyperbolic boundary length (≈sum(dsH)).
# ==============================================================================
struct BoundaryPointsHypBIM{T} <: AbsPoints where{T<:Real}
    xy::Vector{SVector{2,T}}
    normal::Vector{SVector{2,T}}
    curvature::Vector{T}
    ds::Vector{T}
    λ::Vector{T}
    dsH::Vector{T}
    ξ::Vector{T}
    LH::T
end

#------------------------------------------------------------------------------
# smallest_svd_triplet(K;tol=1e-12,maxiter=2000)->(σmin,u,v,info)
#
# PURPOSE
#   Compute smallest singular value σmin of K and associated left/right singular
#   vectors u,v such that:
#     K*v ≈ σmin*u
#     K'*u ≈ σmin*v
#
# INPUTS
#   K::AbstractMatrix{<:Number}
#
# KEYWORDS
#   tol::Real
#     KrylovKit stopping tolerance passed to svdsolve.
#   maxiter::Int
#     KrylovKit maximum iterations passed to svdsolve.
#
# OUTPUTS
#   σmin::Real
#     Smallest singular value.
#   u::AbstractVector
#     Left singular vector (length size(K,1)).
#   v::AbstractVector
#     Right singular vector (length size(K,2)).
#   info
#     KrylovKit info object from svdsolve.
#------------------------------------------------------------------------------
@inline function smallest_svd_triplet(K;tol=1e-12,maxiter=2000)
    S,U,V,info=svdsolve(K,1,:SR;tol=tol,maxiter=maxiter)
    return S[1],U[1],V[1],info
end

#------------------------------------------------------------------------------
# normalize_boundary_left!(u,dsH)->u
#
# PURPOSE
#   Normalize boundary vector u with hyperbolic boundary quadrature weights dsH:
#     u <- u / sqrt( sum_j |u_j|^2 * dsH[j] + eps() )
#
# INPUTS
#   u::AbstractVector{<:Number}
#     Boundary vector (typically left singular vector from SVD).
#   dsH::AbstractVector{<:Real}
#     Hyperbolic boundary weights at the same nodes as u.
#
# OUTPUTS
#   u (modified in place)
#------------------------------------------------------------------------------
@inline function normalize_boundary_left!(u::AbstractVector{<:Number},dsH::AbstractVector{<:Real})
    u./=sqrt(sum(abs2.(u).*dsH)+eps())
    return u
end

#------------------------------------------------------------------------------
# solve(solver,basis,pts_hyp,k;...)->σmin
#
# PURPOSE
#   Assemble the hyperbolic DLP Fredholm matrix K(k) on the boundary and return
#   its smallest singular value σmin(k).
#
# INPUTS
#   solver::BIM_hyperbolic
#     Hyperbolic BIM “sweep solver” settings. Uses:
#       solver.symmetry          : symmetry specification passed to rmin/rmax logic
#       solver.pts_scaling_factor: type carrier (already used when pts were built)
#
#   basis::Ba
#     Currently unused (kept for SweepSolver API compatibility). Can be used later
#     if you want basis-dependent assembly; safe to ignore for now.
#
#   pts_hyp::BoundaryPointsHypBIM{T}
#     Hyperbolic boundary container for THIS k (typically constructed once at kmax
#     and reused across k-sweep). Fields used here:
#       pts_hyp.xy,normal,curvature,ds  (via conversion to BoundaryPointsBIM)
#     Hyperbolic extras (λ,dsH,ξ,LH) are not used by matrix assembly.
#
#   k
#     Wavenumber for which K(k) is assembled (Real/Complex accepted by complex(k)).
#
# KEYWORDS
#   multithreaded::Bool
#     Passed into compute_kernel_matrices_DLP_hyperbolic! for threaded assembly.
#
#   h::Real, P::Int
#     Parameters used to build the QTaylorTable (Legendre-Q evaluation):
#       h : distance grid spacing in d for the table, 1e-4 is good till k=4000 for 12-14 digits
#       P : Taylor/series order (per your QTaylor implementation) 30 is good for 12-14 digits up to k=4000
#
#   use_krylov::Bool
#     If true: use KrylovKit svdsolve to estimate smallest singular value (fast,
#     no full SVD). If false: compute full singular spectrum via svdvals (slow).
#
#   tol::Real, maxiter::Int
#     Passed to KrylovKit svdsolve when use_krylov=true.
#
# OUTPUTS
#   σmin::Real (Float64 in krylov branch, otherwise eltype of svdvals)
#     Smallest singular value of K(k).
#------------------------------------------------------------------------------
function solve(solver::BIM_hyperbolic,basis::Ba,pts_hyp::BoundaryPointsHypBIM,k;multithreaded::Bool=true,h=1e-4,P::Int=30,use_krylov::Bool=true,tol=1e-12,maxiter::Int=2000) where {Ba<:AbstractHankelBasis}
    symmetry=solver.symmetry
    pts=_BoundaryPointsHypBIM_to_BoundaryPointsBIM(pts_hyp)
    N=length(pts.xy)
    dmin_m,dmax_m=estimate_rmin_rmax(pts,symmetry)
    tab=build_QTaylorTable(complex(k);dmin=dmin_m,dmax=dmax_m,h=h,P=P)
    K=Matrix{ComplexF64}(undef,N,N)
    compute_kernel_matrices_DLP_hyperbolic!(K,pts,symmetry,tab;multithreaded=multithreaded)
    assemble_DLP_hyperbolic!(K,pts)
    if use_krylov
        σ,_,_,_=smallest_svd_triplet(K;tol=tol,maxiter=maxiter)
        return σ
    else
        @blas_multi_then_1 MAX_BLAS_THREADS S=svdvals(K)
        return S[end]
    end
end

#------------------------------------------------------------------------------
# solve_vect(solver,basis,pts_hyp,k;...)->(σmin,u)
#
# PURPOSE
#   Assemble K(k) and return:
#     • σmin(k)  : smallest singular value
#     • u        : corresponding LEFT singular vector (boundary data)
#   This u is exactly the Dirichlet boundary density
#   u(s)=∂nψ(s) for SLP wavefunction reconstruction and Husimi/Poincaré–Husimi.
#
# INPUTS
#   solver::BIM_hyperbolic
#     Uses solver.symmetry for rmin/rmax estimation.
#
#   basis::Ba
#     Currently unused (API compatibility).
#
#   pts_hyp::BoundaryPointsHypBIM{T}
#     Boundary discretization container. Used as:
#       - Euclidean geometry (xy,normal,κ,ds) for DLP assembly
#       - Hyperbolic weights (dsH) ONLY for u normalization after SVD
#
#   k
#     Wavenumber at which to assemble K(k).
#
# KEYWORDS
#   multithreaded::Bool
#     Threaded kernel assembly.
#
#   h::Real, P::Int
#     Parameters used to build the QTaylorTable (Legendre-Q evaluation):
#       h : distance grid spacing in d for the table, 1e-4 is good till k=4000 for 12-14 digits
#       P : Taylor/series order (per your QTaylor implementation) 30 is good for 12-14 digits up to k=4000
#
#   use_krylov::Bool
#     If true: compute smallest singular triplet using KrylovKit svdsolve.
#     If false: compute economy SVD via LAPACK.gesvd!('S','S',K).
#
#   tol::Real, maxiter::Int
#     Passed to KrylovKit when use_krylov=true.
#
# OUTPUTS
#   σmin::Real
#     Smallest singular value of K(k).
#
#   u::Vector{ComplexF64} (or eltype from KrylovKit)
#     Left singular vector corresponding to σmin. Normalized with hyperbolic
#     quadrature weights:
#       ∑_j |u_j|^2 dsH[j] = 1
#------------------------------------------------------------------------------
function solve_vect(solver::BIM_hyperbolic,basis::Ba,pts_hyp::BoundaryPointsHypBIM,k;multithreaded::Bool=true,h=1e-4,P::Int=30,use_krylov::Bool=true,tol=1e-12,maxiter::Int=2000) where {Ba<:AbstractHankelBasis}
    symmetry=solver.symmetry
    pts=_BoundaryPointsHypBIM_to_BoundaryPointsBIM(pts_hyp)
    N=length(pts.xy)
    dmin_m,dmax_m=estimate_rmin_rmax(pts,symmetry)
    tab=build_QTaylorTable(complex(k);dmin=dmin_m,dmax=dmax_m,h=h,P=P)
    K=Matrix{ComplexF64}(undef,N,N)
    compute_kernel_matrices_DLP_hyperbolic!(K,pts,symmetry,tab;multithreaded=multithreaded)
    assemble_DLP_hyperbolic!(K,pts)
    if use_krylov
        σ,u,_,_=smallest_svd_triplet(K;tol=tol,maxiter=maxiter)
        normalize_boundary_left!(u,pts_hyp.dsH) # ∑|u|² dsH = 1
        return σ,u
    else
        @blas_multi_then_1 MAX_BLAS_THREADS U,S,Vt=LAPACK.gesvd!('S','S',K)
        idx=findmin(S)[2]
        σ=S[idx]
        u=copy(view(U,:,idx)) 
        normalize_boundary_left!(u,pts_hyp.dsH)# ∑|u|² dsH = 1
        return σ,u
    end
end

#------------------------------------------------------------------------------
# k_sweep(solver,basis,billiard,ks;...)->σmins
#
# PURPOSE
#   Compute σmin(k) over a grid of wavenumbers ks for eigenvalue detection.
#   Boundary points are constructed ONCE (at kmax) and reused for all k in ks.
#
# INPUTS
#   solver::BIM_hyperbolic
#     Uses:
#       solver.symmetry   : passed into estimate_rmin_rmax
#       solver.min_pts    : affects boundary sampling via evaluate_points_hyperbolic
#
#   basis::Ba
#     Currently unused (API compatibility).
#
#   billiard::AbsBilliard
#     Geometry. Used to:
#       - build hyperbolic boundary points (via evaluate_points_hyperbolic)
#       - supply boundary curves used by the solver
#
#   ks::AbstractVector{T}
#     Wavenumber grid. Only maximum(ks) is used to select boundary resolution
#     (points per curve), then the same pts are reused for all entries in ks.
#
# KEYWORDS
#   multithreaded_matrices::Bool
#     Controls threading in both:
#       - evaluate_points_hyperbolic(...;threaded=...)
#       - compute_kernel_matrices_DLP_hyperbolic!(...;multithreaded=...)
#
#   use_krylov::Bool
#     If true: σmin via KrylovKit svdsolve for each k (recommended).
#     If false: full svdvals(K) for each k (very expensive).
#
#   tol::Real, maxiter::Int
#     KrylovKit parameters for σmin extraction (use_krylov=true).
#
#   h::Real, P::Int
#     QTaylor table resolution parameters. A QTaylorPrecomp + workspace is built
#     once and reused; tab is rebuilt in-place for each k. Check solve function for more information.
#
#   M_cdf_base::Int, safety::Real
#     Passed to precompute_hyperbolic_boundary_cdfs / evaluate_points.
#     These control the dense CDF resolution and numerical safety near r→1.
#
# OUTPUTS
#   σmins::Vector{T}
#     σmins[i] = σmin(ks[i]) 
#------------------------------------------------------------------------------
function k_sweep(solver::BIM_hyperbolic,basis::Ba,billiard::AbsBilliard,ks::AbstractVector{T};multithreaded_matrices::Bool=true,use_krylov::Bool=true,tol=1e-12,maxiter::Int=2000,h=1e-4,P::Int=30,M_cdf_base::Int=4000,safety::Real=1e-14) where {T<:Real,Ba<:AbstractHankelBasis}
    symmetry=solver.symmetry
    kmax=maximum(ks)
    pre=precompute_hyperbolic_boundary_cdfs(solver,billiard;M_cdf_base=M_cdf_base,safety=safety)
    pts_hyp=evaluate_points(solver,billiard,real(kmax),pre;safety=safety,threaded=multithreaded_matrices)
    pts=_BoundaryPointsHypBIM_to_BoundaryPointsBIM(pts_hyp)
    N=length(pts.xy)
    dmin_m,dmax_m=estimate_rmin_rmax(pts,symmetry)
    preQ=build_QTaylorPrecomp(dmin=dmin_m,dmax=dmax_m,h=h,P=P)
    ws=QTaylorWorkspace(preQ.P;threaded=false)
    tab=alloc_QTaylorTable(preQ)
    K=Matrix{ComplexF64}(undef,N,N)
    σmins=Vector{T}(undef,length(ks))
    num_intervals=length(ks)
    p=Progress(num_intervals,1)
    @inbounds for i in eachindex(ks)
        k=ks[i]
        build_QTaylorTable!(tab,preQ,ws,complex(k);mp_dps=60,leg_type=3)
        compute_kernel_matrices_DLP_hyperbolic!(K,pts,symmetry,tab;multithreaded=multithreaded_matrices)
        assemble_DLP_hyperbolic!(K,pts)
        if use_krylov
            σ,_,_,_=smallest_svd_triplet(K;tol=tol,maxiter=maxiter)
            σmins[i]=T(σ)
        else
            @blas_multi_then_1 MAX_BLAS_THREADS S=svdvals(K)
            σmins[i]=T(S[end])
        end
        next!(p)
    end
    return σmins
end

# ==============================================================================
# Helpers: periodic wrap + hyperbolic distance bounds (Poincaré disk)
# ==============================================================================
#
# _wrap(i::Int, N::Int) -> Int
#   Purpose
#     Periodic index wrap into 1:N for boundary-neighbor access on a closed curve.
#   Inputs
#     i::Int    : candidate index (may be outside 1:N)
#     N::Int    : number of boundary nodes
#   Output
#     j::Int in 1:N
#   Logic
#     If i<1 add N; if i>N subtract N; otherwise keep i.
#
#
# _dmax_from_rho(ρ::T) where {T<:Real} -> T
#   Purpose
#     Convert a Euclidean radius bound ρ=max‖x‖ (points inside the unit disk)
#     into a safe upper bound on hyperbolic distances d_H between any two points
#     with ‖x‖≤ρ in the Poincaré disk.
#   Inputs
#     ρ::T : Euclidean radius bound (0 ≤ ρ < 1)
#   Output
#     dmax::T : hyperbolic distance upper bound
#   Logic
#     The maximum distance within radius ρ occurs for opposite points ±ρ:
#         dmax = d_H(ρ, -ρ) = 4*atanh(ρ).
#     Clamp ρ below 1 to avoid atanh(1) and overflow near the boundary.
#
#
# d_bounds_hyp(bp::BoundaryPointsHypBIM{T}, symmetry; K=3, dmin_floor=1e-6, pad=1.1)
#     where {T<:Real} -> (dmin::T, dmax::T)
#
#   Purpose
#     Determine safe hyperbolic distance bounds for building/evaluating tables
#     of Q_ν(cosh d) and y(d)=d/dd Q_ν(cosh d) in hyperbolic BIM on the Poincaré disk.
#
#   Inputs
#     bp::BoundaryPointsHypBIM{T}
#       Uses:
#         bp.xy::Vector{SVector{2,T}}  boundary nodes in Euclidean disk coords.
#     symmetry
#       - nothing, or Vector{Any} with symmetry[1] being:
#           * Reflection  (images via x↦-x and/or y↦-y; shift is assumed 0 in disk)
#           * Rotation    (Cn images about center s.center)
#
#   Keyword inputs
#     K::Int
#       Neighbor stencil size for dmin estimation:
#         compares each node i with i±1,...,i±K (and their symmetry images).
#     dmin_floor::T
#       Hard lower floor returned for dmin to avoid d≈0 issues in kernels/tables.
#     pad::T
#       Multiplicative safety padding applied to dmax.
#
#   Outputs
#     dmin::T
#       Smallest nonzero sampled hyperbolic distance between (nearby) boundary nodes,
#       including symmetry images; lower-bounded by dmin_floor.
#     dmax::T
#       Safe upper bound on hyperbolic distances needed by the kernel, computed from
#       a maximal radius bound ρ and padded by pad.
# ==============================================================================

@inline _wrap(i::Int,N::Int)=i<1 ? i+N : (i>N ? i-N : i)
@inline function _dmax_from_rho(ρ::T) where {T<:Real}
    δ=max(T(1e-12),sqrt(eps(one(T))))
    ρc=min(ρ,one(T)-δ)
    return T(4)*atanh(ρc)  # = acosh(1+8ρ^2/(1-ρ^2)^2) 
end

function d_bounds_hyp(bp::BoundaryPointsHypBIM{T},symmetry;K::Int=3,dmin_floor::T=T(1e-6),pad::T=T(1.1)) where {T<:Real}
    xy=bp.xy
    N=length(xy)
    tol2=(T(64)*eps(one(T)))^2
    ρ=zero(T)
    @inbounds for i in 1:N
        x=xy[i][1];y=xy[i][2]
        ρ=max(ρ,hypot(x,y))
    end
    if symmetry!==nothing
        s=symmetry[1]
        if s isa Rotation
            cx=s.center[1];cy=s.center[2];n=s.n
            ctab,stab,_=_rotation_tables(T,n,mod(s.m,n))
            @inbounds for i in 1:N
                x0=xy[i][1]-cx
                y0=xy[i][2]-cy
                @inbounds for l in 2:n
                    ρ=max(ρ,hypot(ctab[l]*x0-stab[l]*y0+cx,stab[l]*x0+ctab[l]*y0+cy))
                end
            end
        end
    end
    dmax=pad*_dmax_from_rho(ρ)
    dmin=typemax(T)
    @inline function upd!(xi,yi,xj,yj)
        dx=xi-xj;dy=yi-yj
        muladd(dx,dx,dy*dy)<=tol2 && return nothing
        d=hyperbolic_distance_poincare(xi,yi,xj,yj)
        (d>zero(T) && d<dmin) && (dmin=d)
        return nothing
    end
    if symmetry===nothing
        @inbounds for i in 1:N
            xi=xy[i][1];yi=xy[i][2]
            @inbounds for t in 1:K
                j=_wrap(i+t,N);upd!(xi,yi,xy[j][1],xy[j][2])
                j=_wrap(i-t,N);upd!(xi,yi,xy[j][1],xy[j][2])
            end
        end
        return max(dmin_floor,dmin),dmax
    end
    s=symmetry[1]
    if s isa Reflection
        ops=_reflect_ops_and_scales(T,s)
        @inbounds for i in 1:N
            xi=xy[i][1];yi=xy[i][2]
            @inbounds for t in 1:K
                @inbounds for j in (_wrap(i+t,N),_wrap(i-t,N))
                    x0=xy[j][1];y0=xy[j][2]
                    upd!(xi,yi,x0,y0)
                    @inbounds for (op,_) in ops
                        x=(op==1||op==3) ? -x0 : x0
                        y=(op==2||op==3) ? -y0 : y0
                        upd!(xi,yi,x,y)
                    end
                end
            end
        end
    elseif s isa Rotation
        cx=s.center[1];cy=s.center[2];n=s.n
        ctab,stab,_=_rotation_tables(T,n,mod(s.m,n))
        @inbounds for i in 1:N
            xi=xy[i][1];yi=xy[i][2]
            @inbounds for t in 1:K
                @inbounds for j in (_wrap(i+t,N),_wrap(i-t,N))
                    x0=xy[j][1];y0=xy[j][2]
                    upd!(xi,yi,x0,y0)
                    x=x0-cx;y=y0-cy
                    @inbounds for l in 2:n
                        upd!(xi,yi,ctab[l]*x-stab[l]*y+cx,stab[l]*x+ctab[l]*y+cy)
                    end
                end
            end
        end
    else
        error("Unsupported symmetry type $(typeof(s))")
    end
    return max(dmin_floor,dmin),dmax
end
