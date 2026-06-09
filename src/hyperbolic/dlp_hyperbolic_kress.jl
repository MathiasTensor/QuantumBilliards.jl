# ==============================================================================
# Hyperbolic DLP-Kress solver
# ==============================================================================
#
# This file implements a Kress-corrected double-layer boundary integral solver
# for two-dimensional quantum billiards in the hyperbolic plane, represented in
# the Poincare disk model.
#
# The hyperbolic metric in the disk is conformal to the Euclidean metric:
#
#     ds_H = λ ds_E,
#     λ(x,y)=2/(1-x^2-y^2).
#
# The solver uses this conformal structure in an important way. Although the
# Green function is hyperbolic, the boundary integral measure can be written
# using Euclidean normals and Euclidean arclength weights because
#
#     ∂ₙᴴG ds_H = ∂ₙᴱG ds_E.
#
# Therefore the matrix assembly uses:
#
#     pts.normal  : Euclidean outward unit normal,
#     pts.ds      : Euclidean boundary quadrature weight,
#
# while the hyperbolic quantities
#
#     pts.λ, pts.dsH, pts.ξ, pts.LH
#
# are kept for hyperbolic node placement, diagnostics, and boundary-Husimi
# postprocessing.
#
# The continuous source-normal double-layer operator is
#
#     D(k)φ(x) = ∫_Γ ∂_{n_y} G_k(d_H(x,y)) φ(y) ds_E(y),
#
# and the Fredholm matrix assembled for eigenvalue searches is
#
#     A(k) = I - D(k).
#
# The singularity is logarithmic, as in the Euclidean Helmholtz problem. For
# points on the same smooth periodic boundary component, the kernel is split as
#
#     K(t,s) = L1(t,s) log(4sin²((t-s)/2)) + L2(t,s),
#
# where `L1` is the logarithmic coefficient and `L2` is smooth. The singular
# logarithmic part is integrated with the periodic Kress matrix `Rmat`, while
# the smooth part is integrated with the ordinary periodic quadrature weight.
#
# The hyperbolic Green function and its normal derivative are evaluated using
# Legendre-Q Taylor tables. The logarithmic coefficient is evaluated using the
# corresponding P-function table. Since these special-function tables depend on
# k but not on the source/target indices, they are stored in the solver
# workspace and reused during matrix assembly.
#
# Diagonal treatment
# ------------------
# The diagonal entry is the limiting value of the smooth Kress remainder L2.
# With the present source-normal convention, the diagonal limit is
#
#     L2(x,x) = (-κ_E(x) - ∂ₙ log λ(x)) / (2π),
#
# where κ_E is the Euclidean signed curvature in the boundary convention used by
# the library, and ∂ₙ log λ is the Euclidean outward normal derivative of the
# Poincare conformal factor.
#
# This sign is important. It corresponds to the source-normal convention used by
# `_∂n_d`. A wrong sign in the curvature term produces a visible
# failure of the L2 diagonal convergence test.
#
# Symmetry handling
# -----------------
# If `solver.symmetry === nothing`, the full Fredholm matrix is assembled.
#
# If a symmetry is supplied, the code follows the same strategy as the Euclidean
# DLP-Kress solver:
#
#   1. build the full periodic Kress discretization,
#   2. keep the full logarithmic quadrature structure,
#   3. assemble only the fundamental-domain block,
#   4. add the remaining symmetry images as regular nonsingular contributions.
#
# This is important because the Kress logarithmic correction is tied to the full
# periodic boundary indexing. The image terms are away from the singular source
# copy, so they are added using the ordinary off-diagonal hyperbolic DLP kernel.
#
# Adjoint / boundary-function path
# --------------------------------
# The assembled DLP matrix uses the source normal, i.e. ∂_{n_y}G. Therefore the
# null vector of A(k)=I-D(k) is a double-layer density, not directly the physical
# boundary normal derivative u=∂ₙψ.
#
# For Husimi and boundary-function postprocessing, we instead solve the weighted
# adjoint problem. In discrete form,
#
#     D'ᵢⱼ = Dⱼᵢ dsⱼ / dsᵢ,
#
# and the adjoint Fredholm matrix is
#
#     A' = I - D'.
#
# The corresponding null vector is the boundary function u=∂ₙψ in the same
# convention used by the Euclidean DLP-Kress solver.
#
# Main public API
# ---------------
#
#   DLP_hyperbolic_kress(...)
#       Construct the solver.
#
#   precompute_hyperbolic_boundary_cdfs(...)
#       Precompute geometry-only hyperbolic arclength CDFs.
#
#   evaluate_points(...)
#       Build `BoundaryPointsHyp` nodes, uniform in hyperbolic arclength.
#
#   build_dlp_hyperbolic_kress_workspace(...)
#       Build the reusable Kress geometry and Taylor-table workspace.
#
#   construct_matrices!(...)
#       Assemble A(k)=I-D(k).
#
#   solve(...)
#       Return a scalar spectral diagnostic, e.g. smallest singular value or
#       determinant-based quantity.
#
#   solve_vect(...)
#       Return the smallest singular value and adjoint boundary vector for
#       Husimi / boundary-function construction.
#
# Low-level routines
# Functions such as `hyp_L1_kress`, `hyp_L2_kress`, `hyp_raw_dlp`,
# `construct_dlp_hyperbolic_kress_matrix!`, and
# `_regular_hyp_dlp_image_D` are internal assembly kernels. They are intentionally
# kept simple and close to the formulas so that the singular split can be checked
# directly.
# MO 26/5/26
# ==============================================================================

const TWO_PI=2*pi
const INV_TWO_PI=1/TWO_PI
const INV_FOUR_PI=1/(2*TWO_PI)

struct DLP_hyperbolic_kress{T<:Real,Bi<:AbsBilliard,Sym}<:SweepSolver
    sampler::Vector{Hyperbolic}
    pts_scaling_factor::Vector{T}
    dim_scaling_factor::T
    eps::T
    min_dim::Int64
    min_pts::Int64
    billiard::Bi
    symmetry::Sym
end

struct DLP_hyperbolic_kress_global_corners{T<:Real,Bi<:AbsBilliard,Sym}<:SweepSolver
    sampler::Vector{Hyperbolic}
    pts_scaling_factor::Vector{T}
    dim_scaling_factor::T
    eps::T
    min_dim::Int64
    min_pts::Int64
    billiard::Bi
    symmetry::Sym
    kressq::Int
    min_t_spacing::T
end

"""
    DLP_hyperbolic_kress(pts_scaling_factor,billiard;min_pts=20,eps=1e-15,symmetry=nothing)

Construct a Kress quadrature scheme hyperbolic double-layer solver.

The solver discretizes the source-normal hyperbolic double-layer Fredholm
operator

    A(k)=I-D(k),

where the hyperbolic Green function depends on the Poincare-disk distance and
the boundary measure is represented using Euclidean weights, exploiting the
conformal cancellation

    ∂ₙᴴG ds_H = ∂ₙᴱG ds_E.

The point density is controlled by `pts_scaling_factor` through the rule

    N ≈ k LH b / (2π),

where `LH` is the hyperbolic boundary length of each real boundary component.

If `symmetry` is supplied, the full Kress discretization is still constructed,
but later matrix assembly uses the reduced symmetry-image workspace.
"""
function DLP_hyperbolic_kress(pts_scaling_factor::Union{T,Vector{T}},billiard::Bi;min_pts=20,eps=T(1e-15),symmetry::Union{Nothing,AbsSymmetry}=nothing) where {T<:Real,Bi<:AbsBilliard}
    bs=pts_scaling_factor isa T ? [pts_scaling_factor] : pts_scaling_factor
    sampler=[Hyperbolic()]
    Sym=typeof(symmetry)
    return DLP_hyperbolic_kress{T,Bi,Sym}(sampler,bs,bs[1],eps,min_pts,min_pts,billiard,symmetry)
end

"""
    DLP_hyperbolic_kress_global_corners(pts_scaling_factor,billiard;
        min_pts=20,eps=1e-15,symmetry=nothing,kressq=4,min_t_spacing=1e-12)

Construct the corner-capable hyperbolic DLP-Kress solver.

This solver is the hyperbolic analogue of `DLP_kress_global_corners`. It is used
for a single composite outer boundary, i.e. a boundary represented by several
curve pieces joined at corners.

The nodes are generated by a global Kress grading map

    t = t(σ),

where `σ` is the computational periodic Kress variable and `t` is the global
geometric boundary parameter. The grading clusters points near the corner
locations returned by `_component_corner_locations`.

For non-composite smooth boundaries this solver falls back to
`DLP_hyperbolic_kress`, because in this library a non-composite boundary is
assumed smooth and has no corners.

The returned `BoundaryPointsHyp` stores:
- `ts = σ`, the graded computational Kress nodes,
- `original_ts = tmap`, the physical global parameter,
- `ws = 2π/N`, the base Kress quadrature weights,
- `ws_der = dt/dσ`, the grading Jacobian,
- `ds`, Euclidean arclength weights,
- `dsH`, hyperbolic arclength weights.

The Kress matrix builder detects nontrivial grading through `ws_der` and uses
`kress_R_corner!` when needed.
"""
function DLP_hyperbolic_kress_global_corners(pts_scaling_factor::Union{T,Vector{T}},billiard::Bi;min_pts=20,eps=T(1e-15),symmetry::Union{Nothing,AbsSymmetry}=nothing,kressq::Int=4,min_t_spacing::T=T(1e-12)) where {T<:Real,Bi<:AbsBilliard}
    bs=pts_scaling_factor isa T ? [pts_scaling_factor] : pts_scaling_factor
    sampler=[Hyperbolic()]
    Sym=typeof(symmetry)
    return DLP_hyperbolic_kress_global_corners{T,Bi,Sym}(sampler,bs,bs[1],eps,min_pts,min_pts,billiard,symmetry,kressq,min_t_spacing)
end

struct DLPHyperbolicKressGeomCache{T<:Real}
    d::Matrix{T}
    dnd::Matrix{T}
    logterm::Matrix{T}
    kappaE::Vector{T}
    dnlogλ::Vector{T}
end

struct DLPHyperbolicKressGeomWorkspace{T<:Real,M<:AbstractMatrix{T}}
    Rmat::M
    G::DLPHyperbolicKressGeomCache{T}
    N::Int
end

struct DLPHyperbolicKressTaylorWorkspace
    pre::QTaylorPrecomp
    qws::QTaylorWorkspace
    qtab::QTaylorTable
    ptab::PTaylorTable
    k::ComplexF64
end

struct DLPHyperbolicKressTaylorOnlyWorkspace
    taylor::DLPHyperbolicKressTaylorWorkspace
    k::ComplexF64
end

struct DLPHyperbolicKressReducedGeomWorkspace{T<:Real,M<:AbstractMatrix{T}}
    full::DLPHyperbolicKressGeomWorkspace{T,M}
    Ifund::Vector{Int}
    full_to_fund::Vector{Int}
    full_to_scale::Vector{Complex{T}}
    fund_to_full::Vector{Vector{Int}}
    fund_to_scale::Vector{Vector{Complex{T}}}
    xs::Vector{T}
    ys::Vector{T}
    nx::Vector{T}
    ny::Vector{T}
    wE::Vector{T}
    m::Int
end

@inline _workspace_dim(ws::DLPHyperbolicKressGeomWorkspace)=ws.N
@inline _workspace_dim(ws::DLPHyperbolicKressReducedGeomWorkspace)=ws.m
@inline _workspace_dim(ws::Tuple{<:Union{DLPHyperbolicKressGeomWorkspace,DLPHyperbolicKressReducedGeomWorkspace},DLPHyperbolicKressTaylorOnlyWorkspace})=_workspace_dim(ws[1])
@inline _is_dlp_hyp_kress_graded(::DLP_hyperbolic_kress,pts::BoundaryPointsHyp)=false
@inline _is_dlp_hyp_kress_graded(::DLP_hyperbolic_kress_global_corners,pts::BoundaryPointsHyp)=_is_nontrivial_hyp_grading(pts)
@inline function _is_nontrivial_hyp_grading(pts::BoundaryPointsHyp{T}) where {T<:Real}
    tol=sqrt(eps(T))
    @inbounds for x in pts.ws_der
        abs(x-one(T))>tol && return true
    end
    return false
end
@inline _dlp_hyp_kress_use_reduced(solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners})=!isnothing(solver.symmetry)
# name compatibility with BIM_hyperbolic
function _hyp_beyn_dim(solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners},pts::BoundaryPointsHyp,k)
    if isnothing(solver.symmetry)
        return length(pts.xy)
    else
        Ifund,_,_,_,_=symmetry_index_orbits(eltype(pts.ds),pts,solver.symmetry,solver.billiard)
        return length(Ifund)
    end
end

"""
    evaluate_points(solver,billiard,k,precomps;threaded=true)
    evaluate_points(solver,billiard,k;M_cdf_base=4000,threaded=true)

Construct hyperbolic boundary quadrature nodes for `solver`.

The nodes are placed uniformly in hyperbolic arclength on each real boundary
component. The returned `BoundaryPointsHyp` stores both Euclidean and hyperbolic
geometry:

- `xy`: boundary points in Euclidean disk coordinates,
- `normal`: Euclidean outward unit normals,
- `kappa`: Euclidean signed curvature,
- `ds`: Euclidean quadrature weights,
- `λ`: Poincare conformal factor,
- `dsH`: hyperbolic quadrature weights,
- `ξ`: cumulative hyperbolic arclength coordinate,
- `LH`: total hyperbolic boundary length,
- `ts`: periodic Kress parameter,
- `ws`: periodic Kress base weights.

The Kress parameter is uniform on each component, while the physical points are
obtained by inverting the precomputed hyperbolic arclength CDF. This lets the
Kress logarithmic correction use a clean periodic parameter while still giving
approximately uniform hyperbolic resolution.
"""
function evaluate_points(solver::DLP_hyperbolic_kress,billiard::Bi,k::Real,precomps::Vector{HyperArcCDFPrecomp{Float64}};safety::Real=1e-14,threaded::Bool=true) where {Bi<:AbsBilliard}
    curves=billiard.full_boundary
    real_idxs=findall(crv->typeof(crv)<:AbsRealCurve,curves)
    nreal=length(real_idxs)
    nreal==1 || error("DLP_hyperbolic_kress expects one smooth periodic real boundary.")
    T=eltype(solver.pts_scaling_factor)
    bs0=solver.pts_scaling_factor
    bs=length(bs0)==1 ? bs0[1] :
       error("DLP_hyperbolic_kress expects scalar pts_scaling_factor.")
    Lh=T(precomps[1].Lh)
    N=max(solver.min_pts,round(Int,real(k)*Lh*bs/TWO_PI))
    needed=2
    if !isnothing(solver.symmetry)
        sym=solver.symmetry
        sym isa Rotation && (needed=lcm(needed,sym.n))
        sym isa Reflection && (needed=lcm(needed,4))
    end
    remN=mod(N,needed)
    remN!=0 && (N+=needed-remN)
    crv=curves[real_idxs[1]]
    h=T(TWO_PI)/T(N)
    σ=[TWO_PI*(T(j)-T(0.5))/T(N) for j in 1:N]
    us=σ./TWO_PI
    pts=curve(crv,us)
    ta=tangent(crv,us)
    t2=tangent_2(crv,us)
    tu,speeds=_unit_tangents_and_speeds(ta)
    nrm=_normals_from_unit_tangents(tu)
    xy=Vector{SVector{2,T}}(undef,N)
    normal=Vector{SVector{2,T}}(undef,N)
    kappa=Vector{T}(undef,N)
    ds=Vector{T}(undef,N)
    λs=Vector{T}(undef,N)
    dsH=Vector{T}(undef,N)
    ξ=Vector{T}(undef,N)
    tangent_1st=Vector{SVector{2,T}}(undef,N)
    tangent_2nd=Vector{SVector{2,T}}(undef,N)
    ws=fill(h,N)
    ws_der=ones(T,N)
    @inbounds for i in 1:N
        p=pts[i]
        x=T(p[1]);y=T(p[2])
        den=max(one(T)-muladd(x,x,y*y),T(1e-15))
        λ=T(2)/den
        sp=T(speeds[i])
        dse=sp*h
        xy[i]=SVector(x,y)
        γu=SVector{2,T}(ta[i])
        γuu=SVector{2,T}(t2[i])
        γσ=γu/T(TWO_PI)
        γσσ=γuu/T(TWO_PI)^2
        sp=hypot(γσ[1],γσ[2])
        normal[i]=SVector(γσ[2]/sp,-γσ[1]/sp)
        kappa[i]=(γσ[1]*γσσ[2]-γσ[2]*γσσ[1])/sp^3
        ds[i]=sp*h
        tangent_1st[i]=γσ
        tangent_2nd[i]=γσσ
        λs[i]=λ
        dsH[i]=λ*ds[i]
    end
    s=zero(T)
    @inbounds for i in 1:N
        ξ[i]=s
        s+=dsH[i]
    end
    return BoundaryPointsHyp{T}(xy,normal,kappa,ds,λs,dsH,ξ,s,tangent_1st,tangent_2nd,σ,copy(us),ws,ws_der)
end

"""
    precompute_hyperbolic_boundary_cdfs(solver,billiard;M_cdf_base=4000,safety=1e-14)

Precompute hyperbolic arclength CDFs for the boundary curves used by `solver`.

Each curve is sampled densely and converted into a monotone CDF for

    dℓ_H = λ(r(t)) |r'(t)| dt.

The resulting precomputation is geometry-only and should be reused across
multiple calls to `evaluate_points`, k-sweeps, or refinement passes.
"""
function precompute_hyperbolic_boundary_cdfs(solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners},billiard::Bi;M_cdf_base::Int=4000,safety::Real=1e-14) where {Bi<:AbsBilliard}
    curves=billiard.full_boundary
    pre=HyperArcCDFPrecomp{Float64}[]
    for crv in curves
        typeof(crv)<:AbsRealCurve && push!(pre,precompute_hyper_cdf(crv;M=M_cdf_base,safety=safety))
    end
    return pre
end

"""
    evaluate_points(solver::DLP_hyperbolic_kress_global_corners,billiard,k;threaded=true)

Construct a globally graded hyperbolic boundary discretization for a composite
cornered boundary.

If the boundary is smooth and non-composite, the method automatically falls back
to `DLP_hyperbolic_kress`.

For composite boundaries, the method:
1. detects corner locations in the global periodic parameter,
2. builds a Kress grading map `t=t(σ)`,
3. evaluates the composite geometry at `t`,
4. applies the chain rule to obtain derivatives with respect to `σ`,
5. computes Euclidean weights `ds` and hyperbolic weights `dsH=λ ds`.

This is the correct corner path because the Kress grading must act on the global
periodic boundary parameter used by the logarithmic quadrature, while the metric
conversion to hyperbolic arclength is applied afterwards through `λ`.
"""
function evaluate_points(solver::DLP_hyperbolic_kress_global_corners,billiard::Bi,k::Real;M_cdf_base::Int=4000,safety::Real=1e-14,threaded::Bool=true) where {Bi<:AbsBilliard}
    boundary=billiard.full_boundary
    if length(boundary)==1 && !(boundary[1] isa AbstractVector)
        base=DLP_hyperbolic_kress(solver.pts_scaling_factor,solver.billiard;min_pts=solver.min_pts,eps=solver.eps,symmetry=solver.symmetry)
        return evaluate_points(base,billiard,k;M_cdf_base=M_cdf_base,safety=safety,threaded=threaded)
    end
    _is_single_composite_boundary(boundary) || error("DLP_hyperbolic_kress_global_corners supports exactly one composite outer boundary.")
    return _evaluate_points_hyp_global_corners(solver,boundary,k,1;threaded=threaded)
end

# fallback when we dont need corner graading but still is a composite boundary. Se we pretend it is just one periodic segment
function _evaluate_points_hyp_smooth_composite(solver::DLP_hyperbolic_kress_global_corners{T},comp::Vector{C},k::Real,idx::Int;threaded::Bool=true) where {T<:Real,C<:AbsCurve}
    pres=[precompute_hyper_cdf(crv;M=5_000,safety=1e-14) for crv in comp]
    Lh=sum(T(pre.Lh) for pre in pres)
    N=max(solver.min_pts,round(Int,real(k)*Lh*solver.pts_scaling_factor[1]/TWO_PI))
    needed=2
    if !isnothing(solver.symmetry)
        sym=solver.symmetry
        sym isa Rotation && (needed=lcm(needed,sym.n))
        sym isa Reflection && (needed=lcm(needed,4))
    end
    remN=mod(N,needed)
    remN!=0 && (N+=needed-remN)
    h=T(TWO_PI)/T(N)
    σ=[TWO_PI*(T(j)-T(0.5))/T(N) for j in 1:N]
    xy=Vector{SVector{2,T}}(undef,N)
    normal=Vector{SVector{2,T}}(undef,N)
    κE=Vector{T}(undef,N)
    ds=Vector{T}(undef,N)
    λs=Vector{T}(undef,N)
    dsH=Vector{T}(undef,N)
    ξ=Vector{T}(undef,N)
    tangent_1st=Vector{SVector{2,T}}(undef,N)
    tangent_2nd=Vector{SVector{2,T}}(undef,N)
    @inbounds for i in 1:N
        q,γσ,γσσ=_eval_composite_geom_global_t(T,comp,σ[i])
        x=q[1]; y=q[2]
        sp=hypot(γσ[1],γσ[2])
        λ=λ_poincare(x,y)
        xy[i]=q
        tangent_1st[i]=γσ
        tangent_2nd[i]=γσσ
        normal[i]=SVector(γσ[2]/sp,-γσ[1]/sp)
        κE[i]=(γσ[1]*γσσ[2]-γσ[2]*γσσ[1])/(sp^3)
        ds[i]=sp*h
        λs[i]=λ
        dsH[i]=λ*ds[i]
    end
    s=zero(T)
    @inbounds for i in 1:N
        ξ[i]=s
        s+=dsH[i]
    end
    ws=fill(h,N)
    ws_der=ones(T,N)
    return BoundaryPointsHyp{T}(xy,normal,κE,ds,λs,dsH,ξ,s,tangent_1st,tangent_2nd,σ,copy(σ),ws,ws_der)
end

"""
    _evaluate_points_hyp_global_corners(solver::DLP_hyperbolic_kress_global_corners{T},comp::Vector{C},k::Real,idx::Int;threaded::Bool=true) where {T<:Real,C<:AbsCurve}

Construct an ungraded hyperbolic Kress discretization for a smooth composite
boundary.

This is the fallback path for `DLP_hyperbolic_kress_global_corners` when the
boundary is represented by several curve pieces, but `_component_corner_locations`
detects no genuine corners.

The boundary is treated as one smooth periodic composite curve with global
parameter `t ∈ [0,2π)`. Since no grading is needed, the computational Kress
parameter equals the geometric parameter:

    σ = t.

The routine evaluates

    γ(t), γ_t(t), γ_tt(t),

computes the Euclidean quadrature weight

    ds = |γ_t| dt,

and then converts it to the hyperbolic weight

    dsH = λ ds.

The returned `BoundaryPointsHyp` has `ws_der = 1`, so the standard smooth Kress
matrix `kress_R!` is used instead of the corner-graded matrix.
"""
function _evaluate_points_hyp_global_corners(solver::DLP_hyperbolic_kress_global_corners{T},comp::Vector{C},k::Real,idx::Int;threaded::Bool=true) where {T<:Real,C<:AbsCurve}
    corners=_component_corner_locations(T,comp)
    isempty(corners) && return _evaluate_points_hyp_smooth_composite(solver,comp,k,idx;threaded=threaded)
    pres=[precompute_hyper_cdf(crv;M=20_000,safety=1e-14) for crv in comp]
    Lh=sum(T(pre.Lh) for pre in pres)
    N=max(solver.min_pts,round(Int,real(k)*Lh*solver.pts_scaling_factor[1]/TWO_PI))
    needed=1
    if !isnothing(solver.symmetry)
        sym=solver.symmetry
        sym isa Rotation && (needed=lcm(needed,sym.n))
        sym isa Reflection && (needed=lcm(needed,4))
    end
    remN=mod(N,needed)
    remN!=0 && (N+=needed-remN)
    σ,τ,jac,jac2,_=multi_kress_graded_nodes_data(T,N,corners;q=solver.kressq,minsep_tol=solver.min_t_spacing)
    xy=Vector{SVector{2,T}}(undef,N)
    normal=Vector{SVector{2,T}}(undef,N)
    κE=Vector{T}(undef,N)
    ds=Vector{T}(undef,N)
    λs=Vector{T}(undef,N)
    dsH=Vector{T}(undef,N)
    ξ=Vector{T}(undef,N)
    tangent_1st=Vector{SVector{2,T}}(undef,N)
    tangent_2nd=Vector{SVector{2,T}}(undef,N)
    h=T(TWO_PI)/T(N)
    @inbounds for i in 1:N
        q,γt,γtt=_eval_composite_geom_global_t(T,comp,τ[i])
        γσ=γt*jac[i]
        γσσ=γtt*(jac[i]^2)+γt*jac2[i]
        sp=hypot(γσ[1],γσ[2])
        x=q[1]
        y=q[2]
        λ=λ_poincare(x,y)
        xy[i]=q
        tangent_1st[i]=γσ
        tangent_2nd[i]=γσσ
        normal[i]=SVector(γσ[2]/sp,-γσ[1]/sp)
        κE[i]=(γσ[1]*γσσ[2]-γσ[2]*γσσ[1])/(sp^3)
        ds[i]=sp*h
        λs[i]=λ
        dsH[i]=λ*ds[i]
    end
    s=zero(T)
    @inbounds for i in 1:N
        ξ[i]=s
        s+=dsH[i]
    end
    ws=fill(h,N)
    ws_der=jac
    return BoundaryPointsHyp{T}(xy,normal,κE,ds,λs,dsH,ξ,s,tangent_1st,tangent_2nd,σ,τ,ws,ws_der)
end

function evaluate_points(solver::DLP_hyperbolic_kress_global_corners,billiard::Bi,k::Real,precomps::Vector{HyperArcCDFPrecomp{Float64}};safety::Real=1e-14,threaded::Bool=true) where {Bi<:AbsBilliard}
    boundary=billiard.full_boundary
    isempty(boundary) && error("Boundary cannot be empty.")
    if length(boundary)==1 && !(boundary[1] isa AbstractVector)
        base=DLP_hyperbolic_kress(solver.pts_scaling_factor,solver.billiard;min_pts=solver.min_pts,eps=solver.eps,symmetry=solver.symmetry)
        return evaluate_points(base,billiard,k,precomps;safety=safety,threaded=threaded)
    end
    _is_single_composite_boundary(boundary) || error("DLP_hyperbolic_kress_global_corners supports exactly one composite outer boundary.")
    return _evaluate_points_hyp_global_corners(solver,boundary,k,1;threaded=threaded)
end

function build_Rmat_dlp_hyperbolic_kress(solver::DLP_hyperbolic_kress,pts::BoundaryPointsHyp{T}) where {T<:Real}
    R=zeros(T,length(pts.xy),length(pts.xy))
    kress_R!(R)
    return R
end

function build_Rmat_dlp_hyperbolic_kress(solver::DLP_hyperbolic_kress_global_corners,pts::BoundaryPointsHyp{T}) where {T<:Real}
    R=zeros(T,length(pts.xy),length(pts.xy))
    _is_nontrivial_hyp_grading(pts) ? kress_R_corner!(R) : kress_R!(R)
    return R
end

# Periodic logarithmic kernel used by the Kress decomposition:
#
#     log(4 sin²((t-s)/2)).
#
# The Kress quadrature integrates this singular contribution analytically via
# the precomputed convolution matrix R.
#
# The diagonal value is never evaluated directly.
@inline function hyp_logterm_periodic(ti::T,tj::T) where {T<:Real}
    return log(4*sin((ti-tj)/2)^2)
end

@inline function hyp_dnlogλ(x::T,y::T,nx::T,ny::T) where {T<:Real}
    den=max(one(T)-muladd(x,x,y*y),T(1e-15))
    return T(2)*muladd(x,nx,y*ny)/den
end

# Precompute geometry data used repeatedly during hyperbolic DLP assembly.
#
# Stored quantities:
#
#   d[i,j] Hyperbolic distance between target node i and source node j.
#
#   dnd[i,j]
#       Source-normal derivative of the hyperbolic distance: ∂_{n_y} d_H(x_i,y_j). This fixes the operator convention to source-normal DLP.
#   logterm[i,j] : Periodic Kress logarithmic singular kernel : log(4 sin²((t_i-t_j)/2)).
#   kappaE[i] : Euclidean signed curvature of the boundary.
#   dnlogλ[i] : Euclidean outward normal derivative of the Poincare conformal factor.
#
# This cache is geometry-only and independent of k.
function build_dlp_hyperbolic_kress_geom_cache(pts::BoundaryPointsHyp{T}) where {T<:Real}
    N=length(pts.xy)
    d=Matrix{T}(undef,N,N)
    dnd=Matrix{T}(undef,N,N)
    logterm=Matrix{T}(undef,N,N)
    dnlogλ=Vector{T}(undef,N)
    @inbounds for i in 1:N
        xi,yi=pts.xy[i]
        nxi,nyi=pts.normal[i]
        dnlogλ[i]=hyp_dnlogλ(xi,yi,nxi,nyi)
    end
    @inbounds for j in 1:N
        xj,yj=pts.xy[j]
        nxj,nyj=pts.normal[j]
        tj=pts.ts[j]
        for i in 1:N
            xi,yi=pts.xy[i]
            if i==j
                d[i,j]=zero(T)
                dnd[i,j]=zero(T)
                logterm[i,j]=zero(T)
            else
                ti=pts.ts[i]
                d[i,j]=hyperbolic_distance_poincare(xi,yi,xj,yj)
                dnd[i,j]=_∂n_d(xi,yi,xj,yj,nxj,nyj)
                logterm[i,j]=hyp_logterm_periodic(ti,tj)
            end
        end
    end
    return DLPHyperbolicKressGeomCache(d,dnd,logterm,pts.kappa,dnlogλ)
end

# Build the k-dependent special-function workspace for hyperbolic Green kernels.
#
# The hyperbolic Green function is represented through Legendre functions:
#
#   Q-table: stores the Green kernel contribution G_H(d) ~ Q(...)
#
#   P-table:  stores the logarithmic coefficient needed for the Kress split.
#
# The singular kernel decomposition requires both:
#
#     K = L1 log(4 sin²((t-s)/2)) + L2.
#
# where:
#   L1 smooth coefficient multiplying the universal logarithmic singularity.
#   L2  smooth remainder after subtraction.
# These tables depend on k but not on matrix indices.
function build_dlp_hyperbolic_kress_taylor_workspace(pts::BoundaryPointsHyp{T},solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners},k;mp_dps::Int=80,leg_type::Int=3) where {T<:Real}
    dmin,dmax=d_bounds_hyp(pts,solver.symmetry;dmin_floor=T(1e-15),pad_max=T(1.1))
    pre=build_QTaylorPrecomp(;dmin=legendre_d_threshold(),dmax=Float64(dmax)*1.05)
    qws=QTaylorWorkspace(;threaded=false)
    qtab=alloc_QTaylorTable(pre;k=ComplexF64(k))
    ptab=alloc_PTaylorTable(pre;k=ComplexF64(k))
    build_QTaylorTable!(qtab,pre,qws,ComplexF64(k);mp_dps=mp_dps,leg_type=leg_type)
    build_PTaylorTable!(ptab,pre,qws,ComplexF64(k);mp_dps=mp_dps)
    return DLPHyperbolicKressTaylorWorkspace(pre,qws,qtab,ptab,ComplexF64(k))
end

function build_dlp_hyperbolic_kress_geom_workspace_full(solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners},pts::BoundaryPointsHyp{T}) where {T<:Real}
    Rmat=build_Rmat_dlp_hyperbolic_kress(solver,pts)
    G=build_dlp_hyperbolic_kress_geom_cache(pts)
    return DLPHyperbolicKressGeomWorkspace(Rmat,G,length(pts.xy))
end

function build_dlp_hyperbolic_kress_geom_workspace_reduced(solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners},pts::BoundaryPointsHyp{T}) where {T<:Real}
    full=build_dlp_hyperbolic_kress_geom_workspace_full(solver,pts)
    Ifund,full_to_fund,full_to_scale,fund_to_full,fund_to_scale=symmetry_index_orbits(T,pts,solver.symmetry,solver.billiard)
    N=length(pts.xy)
    xs=Vector{T}(undef,N);ys=Vector{T}(undef,N)
    nx=Vector{T}(undef,N);ny=Vector{T}(undef,N)
    wE=Vector{T}(undef,N)
    @inbounds for i in 1:N
        xs[i]=pts.xy[i][1];ys[i]=pts.xy[i][2]
        nx[i]=pts.normal[i][1];ny[i]=pts.normal[i][2]
        wE[i]=pts.ds[i]
    end
    return DLPHyperbolicKressReducedGeomWorkspace(full,Ifund,full_to_fund,full_to_scale,fund_to_full,fund_to_scale,xs,ys,nx,ny,wE,length(Ifund))
end

"""
    build_dlp_hyperbolic_kress_workspace(solver,pts,k;mp_dps=80,leg_type=3)

Build the reusable workspace for hyperbolic DLP-Kress assembly.

The workspace stores the geometry-dependent data and the k-dependent Taylor
tables needed for repeated assembly at the same discretization and wavenumber.

For the full workspace this includes:

- the periodic Kress logarithmic matrix `Rmat`,
- pairwise hyperbolic distances,
- source-normal derivatives of the hyperbolic distance,
- periodic logarithmic terms,
- diagonal curvature/conformal data,
- Legendre-Q/P Taylor tables at `k`.

If `solver.symmetry !== nothing`, the function returns a reduced workspace.
The reduced workspace keeps the full Kress geometry but assembles only the
fundamental-domain block. The missing symmetry copies are added as regular
off-diagonal image terms.
"""
function build_dlp_hyperbolic_kress_workspace(solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners},pts::BoundaryPointsHyp{T},k;mp_dps::Int=80,leg_type::Int=3) where {T<:Real}
    gws=build_dlp_hyperbolic_kress_geom_workspace(solver,pts)
    kws=build_dlp_hyperbolic_kress_k_workspace(solver,pts,k;mp_dps=mp_dps,leg_type=leg_type)
    return gws,kws
end

function build_dlp_hyperbolic_kress_geom_workspace(solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners},pts::BoundaryPointsHyp{T}) where {T<:Real}
    _dlp_hyp_kress_use_reduced(solver) ?
    build_dlp_hyperbolic_kress_geom_workspace_reduced(solver,pts) :
    build_dlp_hyperbolic_kress_geom_workspace_full(solver,pts)
end

function build_dlp_hyperbolic_kress_k_workspace(solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners},pts::BoundaryPointsHyp{T},k;mp_dps::Int=80,leg_type::Int=3) where {T<:Real}
    taylor=build_dlp_hyperbolic_kress_taylor_workspace(pts,solver,k;mp_dps=mp_dps,leg_type=leg_type)
    return DLPHyperbolicKressTaylorOnlyWorkspace(taylor,ComplexF64(k))
end

# Diagonal limit of the smooth Kress remainder.
# Since the logarithmic singularity has already been removed into L1, the
# diagonal contribution comes from the finite limit
#
#     lim_{y→x} L2(x,y).
#
# For the source-normal convention used here:
#
#     L2(x,x) = (-κ_E(x) - ∂ₙ log λ(x)) / (2π).
#
# Terms:
#   κ_E : Euclidean signed curvature contribution.
#   ∂ₙ log λ : conformal correction from the Poincare metric.
@inline function hyp_L2_diag_Kress(G::DLPHyperbolicKressGeomCache{T},i::Int) where {T<:Real}
    return -Complex{T}((G.kappaE[i]+G.dnlogλ[i])*INV_TWO_PI,zero(T))
end

# Full off-diagonal source-normal hyperbolic DLP kernel:
#
# K(x,y) = ∂_{n_y} G_H(d_H(x,y)).
#
# This still contains the logarithmic singular structure before Kress splitting.
# The Green kernel comes from the Legendre-Q representation.
# Jump-scaled DLP kernel used in A = I - D.
# This is 2*∂_{n_y}G, not ∂_{n_y}G.
@inline function hyp_raw_dlp(qtab::QTaylorTable,d::Float64,dn::T) where {T<:Real}
    return _eval_dQdd(qtab,d)*dn*(2*INV_TWO_PI)
end

#=
@inline hyp_L1_singlelog(ptab::PTaylorTable,d::Float64,dn::T) where {T<:Real}=2*hyperbolic_Alog_d(ptab,d)*dn
@inline hyp_L1_kress(ptab::PTaylorTable,d::Float64,dn::T) where {T<:Real}=hyperbolic_Alog_d(ptab,d)*dn
@inline function hyp_L2_kress(qtab::QTaylorTable,ptab::PTaylorTable,d::Float64,dn::T,logterm::T) where {T<:Real}
    l1=hyp_L1_kress(ptab,d,dn)
    return hyp_raw_dlp(qtab,d,dn)-l1*logterm
end
=#

@inline hyp_L1_kress(ptab::PTaylorTable,d::Float64,dn::T) where {T<:Real}=2*hyperbolic_Alog_d(ptab,d)*dn
@inline function hyp_L2_kress(qtab::QTaylorTable,ptab::PTaylorTable,d::Float64,dn::T,logterm::T) where {T<:Real}
    l1=hyp_L1_kress(ptab,d,dn)
    return hyp_raw_dlp(qtab,d,dn)-l1*logterm
end

@inline function hyp_kress_entry(qtab::QTaylorTable,ptab::PTaylorTable,Rij::T,dsj::T,wsj::T,d::Float64,dn::T,logterm::T) where {T<:Real}
    l1=2*hyperbolic_Alog_d(ptab,d)*dn
    raw=_eval_dQdd(qtab,d)*dn*(2*INV_TWO_PI)
    l2=raw-l1*logterm
    return Rij*(l1*(dsj/wsj))+dsj*l2
end

# Assemble the full source-normal hyperbolic DLP matrix.
# The continuous operator is
#
#     Dφ(x)=∫_Γ ∂_{n_y}G_H(d_H(x,y)) φ(y) ds_E(y).
#
# Off-diagonal same-copy interactions are split as
#
#     K = L1 logterm + L2.
#
# Discretization:
#     singular part: R * L1
#     smooth part: trapezoidal / periodic quadrature on L2
# ! This constructs the primal DLP operator, not its adjoint.
function construct_dlp_hyperbolic_kress_matrix!(D::AbstractMatrix{Complex{T}},solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners},pts::BoundaryPointsHyp{T},gws::DLPHyperbolicKressGeomWorkspace{T},kws::DLPHyperbolicKressTaylorOnlyWorkspace;multithreaded::Bool=true) where {T<:Real}
    N=gws.N
    R=gws.Rmat
    G=gws.G
    qtab=kws.taylor.qtab
    ptab=kws.taylor.ptab
    fill!(D,zero(Complex{T}))
    @inbounds for i in 1:N
        D[i,i]=pts.ds[i]*hyp_L2_diag_Kress(G,i)
    end
    @use_threads multithreading=(multithreaded && N>=32) for j in 2:N
        dsj=pts.ds[j]
        wsj=pts.ws[j]
        @inbounds for i in 1:j-1
            D[i,j]=hyp_kress_entry(qtab,ptab,R[i,j],dsj,wsj,Float64(G.d[i,j]),G.dnd[i,j],G.logterm[i,j])
            D[j,i]=hyp_kress_entry(qtab,ptab,R[j,i],pts.ds[i],pts.ws[i],Float64(G.d[j,i]),G.dnd[j,i],G.logterm[j,i])
        end
    end
    return D
end

# Regular DLP contribution from a symmetry-image source. These sources are geometrically separated from the physical singular copy, so no Kress logarithmic correction is needed. This is simply the ordinary off-diagonal hyperbolic source-normal DLP kernel.
@inline function _regular_hyp_dlp_image_D(qtab::QTaylorTable,xi::T,yi::T,xj::T,yj::T,nxj::T,nyj::T,wj::T,scale::Complex{T}) where {T<:Real}
    d=hyperbolic_distance_poincare(xi,yi,xj,yj)
    d<=eps(T) && return zero(Complex{T})
    dn=_∂n_d(xi,yi,xj,yj,nxj,nyj)
    return scale*hyp_raw_dlp(qtab,Float64(d),dn)*wj
end

# Assemble the symmetry-reduced DLP operator.
# Strategy:
#
#   1. same-copy singular interactions: assembled with the full Kress logarithmic treatment.
#   2. all additional symmetry copies: added explicitly as regular image interactions.
#
# This preserves the exact singular quadrature structure while reducing matrix
# size to the fundamental domain.
function construct_dlp_hyperbolic_kress_matrix!(D::AbstractMatrix{Complex{T}},solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners},pts::BoundaryPointsHyp{T},rgws::DLPHyperbolicKressReducedGeomWorkspace{T},kws::DLPHyperbolicKressTaylorOnlyWorkspace;multithreaded::Bool=true) where {T<:Real}
    m=rgws.m
    @assert size(D,1)==m && size(D,2)==m
    full=rgws.full
    R=full.Rmat
    G=full.G
    qtab=kws.taylor.qtab
    ptab=kws.taylor.ptab
    Ifund=rgws.Ifund
    fill!(D,zero(Complex{T}))
    @use_threads multithreading=(multithreaded && m>=32) for b in 1:m
        j=Ifund[b]
        dsj=pts.ds[j]
        wsj=pts.ws[j]
        @inbounds for a in 1:m
            i=Ifund[a]
            if i==j
                D[a,b]=dsj*hyp_L2_diag_Kress(G,i)
            else
                D[a,b]=hyp_kress_entry(qtab,ptab,R[i,j],dsj,wsj,Float64(G.d[i,j]),G.dnd[i,j],G.logterm[i,j])
            end
        end
    end
    @inbounds for b in 1:m
        j=Ifund[b]
        imgs=rgws.fund_to_full[b]
        scales=rgws.fund_to_scale[b]
        for a in 1:m
            i=Ifund[a]
            xi=rgws.xs[i]
            yi=rgws.ys[i]
            s=zero(Complex{T})
            for l in eachindex(imgs)
                q=imgs[l]
                q==j && continue
                s+=_regular_hyp_dlp_image_D(qtab,xi,yi,rgws.xs[q],rgws.ys[q],rgws.nx[q],rgws.ny[q],rgws.wE[q],scales[l])
            end
            D[a,b]+=s
        end
    end
    return D
end

function construct_fredholm_hyperbolic_kress_matrix!(A::AbstractMatrix{Complex{T}},solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners},pts::BoundaryPointsHyp{T},gws::Union{DLPHyperbolicKressGeomWorkspace{T},DLPHyperbolicKressReducedGeomWorkspace{T}},kws::DLPHyperbolicKressTaylorOnlyWorkspace;multithreaded::Bool=true) where {T<:Real}
    construct_dlp_hyperbolic_kress_matrix!(A,solver,pts,gws,kws;multithreaded=multithreaded)
    @inbounds for j in axes(A,2), i in axes(A,1)
        A[i,j]*=-1
    end
    @inbounds for i in axes(A,1)
        A[i,i]+=one(Complex{T})
    end
    return A
end

# Assemble the weighted adjoint Fredholm matrix.
# The operator uses source normals:
#
#     Dφ = ∫ ∂_{n_y}G φ ds.
#
# The layer density is obtained from the adjoint:
#
#     D'ᵢⱼ = Dⱼᵢ dsⱼ/dsᵢ.
#
# Therefore solve_vect() uses this matrix instead of the source-normal Fredholm matrix.
# The Euclidean weights ds are correct because ∂ₙᴴG ds_H = ∂ₙᴱG ds_E.
function adjoint_fredholm_matrix!(A::AbstractMatrix{Complex{T}},D::AbstractMatrix{Complex{T}},solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners},pts::BoundaryPointsHyp{T},gws::DLPHyperbolicKressGeomWorkspace{T},kws::DLPHyperbolicKressTaylorOnlyWorkspace;multithreaded::Bool=true) where {T<:Real}
    construct_dlp_hyperbolic_kress_matrix!(D,solver,pts,gws,kws;multithreaded=multithreaded)
    ds=pts.ds
    N=gws.N
    @inbounds for i in 1:N, j in 1:N
        A[i,j]=-D[j,i]*ds[j]/ds[i]
    end
    @inbounds for i in 1:N
        A[i,i]+=one(Complex{T})
    end
    return A
end

function adjoint_fredholm_matrix!(A::AbstractMatrix{Complex{T}},D::AbstractMatrix{Complex{T}},solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners},pts::BoundaryPointsHyp{T},rgws::DLPHyperbolicKressReducedGeomWorkspace{T},kws::DLPHyperbolicKressTaylorOnlyWorkspace;multithreaded::Bool=true) where {T<:Real}
    construct_dlp_hyperbolic_kress_matrix!(D,solver,pts,rgws,kws;multithreaded=multithreaded)
    ds=pts.ds
    Ifund=rgws.Ifund
    m=rgws.m
    @inbounds for b in 1:m, a in 1:m
        i=Ifund[a]
        j=Ifund[b]
        A[a,b]=-D[b,a]*ds[j]/ds[i]
    end
    @inbounds for a in 1:m
        A[a,a]+=one(Complex{T})
    end
    return A
end

################################
#### TARGET NORMAL APPROACH ####
################################

# Direct assembly of the weighted adjoint Kress entry.
#
# This is algebraically identical to first assembling the source-normal DLP
# entry
#
#     D[j,i] = R[j,i] L1[j,i] ds_i/ws_i + ds_i L2[j,i],
#
# and then applying the weighted adjoint relation
#
#     D'ᵢⱼ = Dⱼᵢ dsⱼ/dsᵢ.
#
# The cancellation of dsᵢ is kept in the formula in this explicit form so that
# the code mirrors the derivation and is easy to compare against the reference
# `adjoint_fredholm_matrix!` path.
@inline function hyp_kress_entry_weighted_adjoint(qtab::QTaylorTable,ptab::PTaylorTable,Rji::T,dsi::T,dsj::T,wsi::T,d::Float64,dn::T,logterm::T) where {T<:Real}
    l1=2*hyperbolic_Alog_d(ptab,d)*dn
    raw=_eval_dQdd(qtab,d)*dn*(2*INV_TWO_PI)
    l2=raw-l1*logterm
    return (Rji*(l1*(dsi/wsi))+dsi*l2)*(dsj/dsi)
end

"""
    construct_adjoint_fredholm_hyperbolic_kress_matrix_direct!(A,solver,pts,gws,kws;multithreaded=true)

Assemble the weighted adjoint Fredholm matrix directly, without first building
the primal source-normal DLP matrix.

The source-normal discretization has entries

    Dᵢⱼ ≈ ∂_{n_j}G(xᵢ,xⱼ) dsⱼ.

The boundary function needed for Husimi/postprocessing is obtained from the
weighted adjoint operator

    D'ᵢⱼ = Dⱼᵢ dsⱼ/dsᵢ,

and the Fredholm matrix is

    A' = I - D'.

This routine constructs `A'` directly by using the symmetry of the Green
function,

    ∂_{n_i}G(xᵢ,xⱼ) = ∂_{n_j}G(xⱼ,xᵢ),

which means that the target-normal entry is obtained by reusing the
source-normal kernel with swapped indices `(j,i)`.

For off-diagonal same-copy interactions the Kress split is also used with
swapped indices:

    R[j,i], d[j,i], dnd[j,i], logterm[j,i].

The diagonal is unchanged by the weighted adjoint relation, since
`dsᵢ/dsᵢ = 1`.

This is the fast production path for `solve_vect`; the older `:via_D` path is
kept as a reference/regression implementation.
"""
function construct_adjoint_fredholm_hyperbolic_kress_matrix_direct!(A::AbstractMatrix{Complex{T}},solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners},pts::BoundaryPointsHyp{T},gws::DLPHyperbolicKressGeomWorkspace{T},kws::DLPHyperbolicKressTaylorOnlyWorkspace;multithreaded::Bool=true) where {T<:Real}
    N=gws.N
    R=gws.Rmat
    G=gws.G
    qtab=kws.taylor.qtab
    ptab=kws.taylor.ptab
    ds=pts.ds
    fill!(A,zero(Complex{T}))
    @inbounds for i in 1:N
        A[i,i]=one(Complex{T})-ds[i]*hyp_L2_diag_Kress(G,i)
    end
    @use_threads multithreading=(multithreaded && N>=32) for j in 1:N
        dsj=ds[j]
        @inbounds for i in 1:N
            i==j && continue
            A[i,j]=-hyp_kress_entry_weighted_adjoint(qtab,ptab,R[j,i],ds[i],dsj,pts.ws[i],Float64(G.d[j,i]),G.dnd[j,i],G.logterm[j,i])
        end
    end
    return A
end

"""
    construct_adjoint_fredholm_hyperbolic_kress_matrix_direct!(A,solver,pts,rgws,kws;multithreaded=true)

Directly assemble the symmetry-reduced weighted adjoint Fredholm matrix.

The reduced source-normal DLP matrix is assembled on the fundamental-domain
indices `Ifund`. Its weighted adjoint is

    D'ₐᵦ = Dᵦₐ dsⱼ/dsᵢ,

where

    i = Ifund[a],
    j = Ifund[b].

The same-copy singular contribution is handled by the full periodic Kress data
with swapped full indices `(j,i)`. This preserves the same logarithmic
quadrature structure as the source-normal assembly.

The additional symmetry-image contributions are regular. For the direct adjoint
assembly they are obtained by swapping source/target roles relative to the
primal reduced assembly and then applying the same weight factor `dsⱼ/dsᵢ`.

This routine should agree with adjoint_fredholm_matrix!(A,D,solver,pts,rgws,kws)
up to roundoff, while avoiding allocation and assembly of the intermediate `D`.
"""
function construct_adjoint_fredholm_hyperbolic_kress_matrix_direct!(A::AbstractMatrix{Complex{T}},solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners},pts::BoundaryPointsHyp{T},rgws::DLPHyperbolicKressReducedGeomWorkspace{T},kws::DLPHyperbolicKressTaylorOnlyWorkspace;multithreaded::Bool=true) where {T<:Real}
    m=rgws.m
    @assert size(A,1)==m && size(A,2)==m
    full=rgws.full
    R=full.Rmat
    G=full.G
    qtab=kws.taylor.qtab
    ptab=kws.taylor.ptab
    Ifund=rgws.Ifund
    ds=pts.ds
    fill!(A,zero(Complex{T}))
    @use_threads multithreading=(multithreaded && m>=32) for b in 1:m
        j=Ifund[b]
        dsj=ds[j]
        @inbounds for a in 1:m
            i=Ifund[a]
            dsi=ds[i]
            if i==j
                A[a,b]=-dsi*hyp_L2_diag_Kress(G,i)
            else
                A[a,b]=-hyp_kress_entry_weighted_adjoint(qtab,ptab,R[j,i],dsi,dsj,pts.ws[i],Float64(G.d[j,i]),G.dnd[j,i],G.logterm[j,i])
            end
        end
    end
    @inbounds for b in 1:m
        j=Ifund[b]
        dsj=ds[j]
        for a in 1:m
            i=Ifund[a]
            dsi=ds[i]
            imgs=rgws.fund_to_full[a]
            scales=rgws.fund_to_scale[a]
            simg=zero(Complex{T})
            for l in eachindex(imgs)
                q=imgs[l]
                q==i && continue
                simg+=_regular_hyp_dlp_image_D(qtab,rgws.xs[j],rgws.ys[j],rgws.xs[q],rgws.ys[q],rgws.nx[q],rgws.ny[q],rgws.wE[q],scales[l])
            end
            A[a,b]-=simg*(dsj/dsi)
        end
    end
    @inbounds for a in 1:m
        A[a,a]+=one(Complex{T})
    end
    return A
end

"""
    construct_adjoint_fredholm_hyperbolic_kress_matrix!(A,solver,pts,gws,kws;multithreaded=true,adjoint_mode=:direct)

Assemble the weighted adjoint Fredholm matrix used by `solve_vect`.
Keyword `adjoint_mode` selects the implementation:

- `:direct`
    Assemble `A' = I-D'` directly using swapped Kress/source-normal data.
    This avoids allocating and assembling an intermediate primal DLP matrix.

- `:via_D`
    Reference path. Assemble the primal source-normal DLP matrix `D`, then form

        D'ᵢⱼ = Dⱼᵢ dsⱼ/dsᵢ.

    This is slower and allocates `D`, but is useful for regression testing the
    direct implementation.

Both modes should produce the same matrix up to roundoff.
"""
function construct_adjoint_fredholm_hyperbolic_kress_matrix!(A::AbstractMatrix{Complex{T}},solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners},pts::BoundaryPointsHyp{T},gws::Union{DLPHyperbolicKressGeomWorkspace{T},DLPHyperbolicKressReducedGeomWorkspace{T}},kws::DLPHyperbolicKressTaylorOnlyWorkspace;multithreaded::Bool=true,adjoint_mode::Symbol=:direct) where {T<:Real}
    if adjoint_mode===:direct
        construct_adjoint_fredholm_hyperbolic_kress_matrix_direct!(A,solver,pts,gws,kws;multithreaded=multithreaded)
    elseif adjoint_mode===:via_D
        D=similar(A)
        adjoint_fredholm_matrix!(A,D,solver,pts,gws,kws;multithreaded=multithreaded)
    else
        error("Unknown adjoint_mode=$(adjoint_mode). Use :direct or :via_D.")
    end
    return A
end

################################

function construct_matrices!(solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners},A::AbstractMatrix{Complex{T}},pts::BoundaryPointsHyp{T},gws::Union{DLPHyperbolicKressGeomWorkspace{T},DLPHyperbolicKressReducedGeomWorkspace{T}},kws::DLPHyperbolicKressTaylorOnlyWorkspace,k;multithreaded::Bool=true,adjoint_mode::Symbol=:direct) where {T<:Real}
    if adjoint_mode===:source
        construct_fredholm_hyperbolic_kress_matrix!(A,solver,pts,gws,kws;multithreaded=multithreaded)
    elseif adjoint_mode===:direct || adjoint_mode===:via_D
        construct_adjoint_fredholm_hyperbolic_kress_matrix!(A,solver,pts,gws,kws;multithreaded=multithreaded,adjoint_mode=adjoint_mode)
    else
        error("Invalid adjoint_mode: $adjoint_mode. Expected :source, :direct, or :via_D.")
    end
    return A
end

function construct_matrices!(solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners},A::AbstractMatrix{Complex{T}},pts::BoundaryPointsHyp{T},ws::Tuple{<:Union{DLPHyperbolicKressGeomWorkspace,DLPHyperbolicKressReducedGeomWorkspace},DLPHyperbolicKressTaylorOnlyWorkspace},k;multithreaded::Bool=true,adjoint_mode::Symbol=:direct) where {T<:Real}
    construct_matrices!(solver,A,pts,ws[1],ws[2],k;multithreaded=multithreaded,adjoint_mode=adjoint_mode)
end

function construct_boundary_matrices!(Tbufs::Vector{Matrix{ComplexF64}},solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners},pts::BoundaryPointsHyp{T},zj::AbstractVector{ComplexF64};multithreaded::Bool=true,timeit::Bool=false,adjoint_mode::Symbol=:direct) where {T<:Real}
    @blas_1 begin
        @benchit timeit=timeit "DLP_hyperbolic_kress GeometryWorkspace" begin
            gws=build_dlp_hyperbolic_kress_geom_workspace(solver,pts)
        end
        n=_workspace_dim(gws)
        @inbounds for q in eachindex(zj)
            @assert size(Tbufs[q])==(n,n) "Tbufs[$q] has size $(size(Tbufs[q])), but DLP-hyperbolic-Kress requires ($n,$n)."
            @benchit timeit=timeit "DLP_hyperbolic_kress TaylorWorkspace" begin
                kws=build_dlp_hyperbolic_kress_k_workspace(solver,pts,zj[q])
            end
            fill!(Tbufs[q],0.0+0.0im)
            @benchit timeit=timeit "DLP_hyperbolic_kress Assembly" begin
                construct_matrices!(solver,Tbufs[q],pts,gws,kws,zj[q];multithreaded=multithreaded,adjoint_mode=adjoint_mode)
            end
        end
    end

    return nothing
end

"""
    solve(solver,basis,pts,k; kwargs...)
    solve(solver,basis,pts,ws,k; kwargs...)
    solve(solver,basis,A,pts,ws,k; kwargs...)

Compute a scalar spectral diagnostic for the hyperbolic DLP-Kress Fredholm
matrix.

This is the standard high-level entry point for detecting eigenvalues. Depending
on `which`, the returned scalar is typically:

- the smallest singular value (`which = :svd`),
- a determinant-based diagnostic (`which = :det`).

By default this solver assembles the weighted adjoint Fredholm matrix

    A' = I - D',

rather than the primal source-normal matrix

    A = I - D.

This is intentional: `A` and `A'` have the same eigenvalue condition, while the
null vectors of `A'` are the physical boundary functions needed for
Poincaré--Husimi and boundary postprocessing.

Method variants
---------------
The signatures differ only in how much reusable data is supplied:

    solve(solver,basis,pts,k)

Build workspace and matrix storage internally.

    solve(solver,basis,pts,ws,k)

Reuse a previously constructed workspace.

    solve(solver,basis,A,pts,ws,k)

Reuse both matrix storage `A` and workspace `ws`.

Keyword arguments
-----------------
- `multithreaded=true`:
    Enable threaded matrix assembly when supported.

- `use_krylov=true`:
    Use Krylov iterative solver for the SVD-based diagnostic.

- `which=:svd`:
    Spectral diagnostic to compute.

- `adjoint_mode=:direct`:
    Select how the weighted adjoint Fredholm matrix is assembled.
    `:direct` assembles `A'` directly using swapped Kress/source-normal data.
    `:via_D` first assembles the source-normal DLP matrix `D` and then forms
    `D'ᵢⱼ = Dⱼᵢ dsⱼ/dsᵢ`.

Returns
-------
A scalar diagnostic whose minima indicate eigenvalues.
"""
function solve(solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners},basis::Ba,A::AbstractMatrix{Complex{T}},pts::BoundaryPointsHyp{T},gws::Union{DLPHyperbolicKressGeomWorkspace{T},DLPHyperbolicKressReducedGeomWorkspace{T}},kws::DLPHyperbolicKressTaylorOnlyWorkspace,k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:svd,adjoint_mode::Symbol=:direct) where {T<:Real,Ba<:AbsBasis}
    @blas_1 construct_matrices!(solver,A,pts,gws,kws,k;multithreaded=multithreaded,adjoint_mode=adjoint_mode)
    @svd_or_det_solve A use_krylov which MAX_BLAS_THREADS
end

function solve(solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners},basis::Ba,pts::BoundaryPointsHyp{T},gws::Union{DLPHyperbolicKressGeomWorkspace{T},DLPHyperbolicKressReducedGeomWorkspace{T}},kws::DLPHyperbolicKressTaylorOnlyWorkspace,k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:svd,adjoint_mode::Symbol=:direct) where {T<:Real,Ba<:AbsBasis}
    n=_workspace_dim(gws)
    A=Matrix{Complex{T}}(undef,n,n)
    return solve(solver,basis,A,pts,gws,kws,k;multithreaded=multithreaded,use_krylov=use_krylov,which=which,adjoint_mode=adjoint_mode)
end

function solve(solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners},basis::Ba,pts::BoundaryPointsHyp{T},ws::Tuple{<:Union{DLPHyperbolicKressGeomWorkspace,DLPHyperbolicKressReducedGeomWorkspace},DLPHyperbolicKressTaylorOnlyWorkspace},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:svd,adjoint_mode::Symbol=:direct) where {T<:Real,Ba<:AbsBasis}
    return solve(solver,basis,pts,ws[1],ws[2],k;multithreaded=multithreaded,use_krylov=use_krylov,which=which,adjoint_mode=adjoint_mode)
end

function solve(solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners},basis::Ba,A::AbstractMatrix{Complex{T}},pts::BoundaryPointsHyp{T},ws::Tuple{<:Union{DLPHyperbolicKressGeomWorkspace,DLPHyperbolicKressReducedGeomWorkspace},DLPHyperbolicKressTaylorOnlyWorkspace},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:svd,adjoint_mode::Symbol=:direct) where {T<:Real,Ba<:AbsBasis}
    return solve(solver,basis,A,pts,ws[1],ws[2],k;multithreaded=multithreaded,use_krylov=use_krylov,which=which,adjoint_mode=adjoint_mode)
end

function solve(solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners},basis::Ba,pts::BoundaryPointsHyp{T},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:svd,mp_dps::Int=80,leg_type::Int=3,adjoint_mode::Symbol=:direct) where {T<:Real,Ba<:AbsBasis}
    gws=build_dlp_hyperbolic_kress_geom_workspace(solver,pts)
    kws=build_dlp_hyperbolic_kress_k_workspace(solver,pts,k;mp_dps=mp_dps,leg_type=leg_type)
    return solve(solver,basis,pts,gws,kws,k;multithreaded=multithreaded,use_krylov=use_krylov,which=which,adjoint_mode=adjoint_mode)
end

"""
    solve_vect(solver,basis,pts,k; kwargs...)
    solve_vect(solver,basis,pts,ws,k; kwargs...)
    solve_vect(solver,basis,A,pts,ws,k; kwargs...)

If adjoint_mode=:source, the returned vector is the DLP layer density.
If adjoint_mode=:direct or :via_D, the returned vector is the physical
boundary function u=∂ₙψ.

This function is used when the boundary representation of an eigenstate is
required, rather than only a scalar spectral diagnostic.

For the source-normal DLP formulation, the primal Fredholm matrix

    A = I - D

has a null vector corresponding to the double-layer density. For Husimi,
wavefunction reconstruction, and boundary-function diagnostics, we instead need
the physical boundary function

    u = ∂ₙψ.

Therefore `solve_vect` solves the weighted adjoint Fredholm problem

    A' = I - D',

where

    D'ᵢⱼ = Dⱼᵢ dsⱼ/dsᵢ.

Equivalently, this is the target-normal formulation

    D'u(xᵢ) ≈ ∫_Γ ∂_{n_i}G(xᵢ,y)u(y)ds_y.

The Euclidean weights `ds` are used because of the conformal cancellation

    ∂ₙᴴG ds_H = ∂ₙᴱG ds_E.

Method variants
---------------
The signatures differ only in how much reusable data is supplied:

    solve_vect(solver,basis,pts,k)

Build workspace and matrix storage internally.

    solve_vect(solver,basis,pts,ws,k)

Reuse a previously constructed workspace.

    solve_vect(solver,basis,A,pts,ws,k)

Fully allocation-minimizing variant with preallocated matrix storage.

Keyword arguments
-----------------
- `multithreaded=true`:
    Enable threaded matrix assembly where supported.

- `tol=1e-12`:
    Null-vector convergence tolerance.

- `maxiter=2000`:
    Maximum Krylov iterations.

- `krylovdim=40`:
    Krylov subspace dimension.

- `adjoint_mode=:direct`:
    Select how the weighted adjoint Fredholm matrix is assembled.

    `:direct` assembles `A' = I-D'` directly using swapped Kress/source-normal
    data. This avoids allocating and assembling an intermediate primal DLP
    matrix and is the preferred production path.

    `:via_D` first assembles the source-normal DLP matrix `D` and then forms
    `D'ᵢⱼ = Dⱼᵢ dsⱼ/dsᵢ`. This is mainly useful for regression tests.

Returns
-------
`(σ,u)`, where:

- `σ` is the smallest null singular value,
- `u` is the associated physical boundary vector.

This is the correct function for:

- boundary Husimi construction,
- wavefunction reconstruction,
- boundary current analysis,
- localization diagnostics based on boundary data.

If only eigenvalue detection is needed, [`solve`] is cheaper.
"""
function solve_vect(solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners},basis::Ba,A::AbstractMatrix{Complex{T}},pts::BoundaryPointsHyp{T},gws::Union{DLPHyperbolicKressGeomWorkspace{T},DLPHyperbolicKressReducedGeomWorkspace{T}},kws::DLPHyperbolicKressTaylorOnlyWorkspace,k;multithreaded::Bool=true,tol=1e-12,maxiter::Int=2000,krylovdim::Int=40,adjoint_mode::Symbol=:direct) where {T<:Real,Ba<:AbsBasis}
    @blas_1 construct_matrices!(solver,A,pts,gws,kws,k;multithreaded=multithreaded,adjoint_mode=adjoint_mode)
    σ,u,_=smallest_nullvec_krylov!(A;nev=1,tol=tol,maxiter=maxiter,krylovdim=krylovdim)
    return σ,u
end

function solve_vect(solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners},basis::Ba,pts::BoundaryPointsHyp{T},gws::Union{DLPHyperbolicKressGeomWorkspace{T},DLPHyperbolicKressReducedGeomWorkspace{T}},kws::DLPHyperbolicKressTaylorOnlyWorkspace,k;multithreaded::Bool=true,tol=1e-12,maxiter::Int=2000,krylovdim::Int=40,adjoint_mode::Symbol=:direct) where {T<:Real,Ba<:AbsBasis}
    n=_workspace_dim(gws)
    A=Matrix{Complex{T}}(undef,n,n)
    return solve_vect(solver,basis,A,pts,gws,kws,k;multithreaded=multithreaded,tol=tol,maxiter=maxiter,krylovdim=krylovdim,adjoint_mode=adjoint_mode)
end

function solve_vect(solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners},basis::Ba,pts::BoundaryPointsHyp{T},ws::Tuple{<:Union{DLPHyperbolicKressGeomWorkspace,DLPHyperbolicKressReducedGeomWorkspace},DLPHyperbolicKressTaylorOnlyWorkspace},k;multithreaded::Bool=true,tol=1e-12,maxiter::Int=2000,krylovdim::Int=40,adjoint_mode::Symbol=:direct) where {T<:Real,Ba<:AbsBasis}
    return solve_vect(solver,basis,pts,ws[1],ws[2],k;multithreaded=multithreaded,tol=tol,maxiter=maxiter,krylovdim=krylovdim,adjoint_mode=adjoint_mode)
end

function solve_vect(solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners},basis::Ba,A::AbstractMatrix{Complex{T}},pts::BoundaryPointsHyp{T},ws::Tuple{<:Union{DLPHyperbolicKressGeomWorkspace,DLPHyperbolicKressReducedGeomWorkspace},DLPHyperbolicKressTaylorOnlyWorkspace},k;multithreaded::Bool=true,tol=1e-12,maxiter::Int=2000,krylovdim::Int=40,adjoint_mode::Symbol=:direct) where {T<:Real,Ba<:AbsBasis}
    return solve_vect(solver,basis,A,pts,ws[1],ws[2],k;multithreaded=multithreaded,tol=tol,maxiter=maxiter,krylovdim=krylovdim,adjoint_mode=adjoint_mode)
end

function solve_vect(solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners},basis::Ba,pts::BoundaryPointsHyp{T},k;multithreaded::Bool=true,tol=1e-12,maxiter::Int=2000,krylovdim::Int=40,mp_dps::Int=80,leg_type::Int=3,adjoint_mode::Symbol=:direct) where {T<:Real,Ba<:AbsBasis}
    gws=build_dlp_hyperbolic_kress_geom_workspace(solver,pts)
    kws=build_dlp_hyperbolic_kress_k_workspace(solver,pts,k;mp_dps=mp_dps,leg_type=leg_type)
    return solve_vect(solver,basis,pts,gws,kws,k;multithreaded=multithreaded,tol=tol,maxiter=maxiter,krylovdim=krylovdim,adjoint_mode=adjoint_mode)
end