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
# `hyperbolic_dn_d_source`. A wrong sign in the curvature term produces a visible
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

struct DLPHyperbolicKressTaylorWorkspace
    pre::QTaylorPrecomp
    qws::QTaylorWorkspace
    qtab::QTaylorTable
    ptab::PTaylorTable
    k::ComplexF64
end

struct DLPHyperbolicKressWorkspace{T<:Real,M<:AbstractMatrix{T}}
    Rmat::M
    G::DLPHyperbolicKressGeomCache{T}
    taylor::DLPHyperbolicKressTaylorWorkspace
    N::Int
end

struct DLPHyperbolicKressReducedWorkspace{T<:Real,M<:AbstractMatrix{T}}
    full::DLPHyperbolicKressWorkspace{T,M}
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

@inline _workspace_dim(ws::DLPHyperbolicKressWorkspace)=ws.N
@inline _workspace_dim(ws::DLPHyperbolicKressReducedWorkspace)=ws.m
@inline _is_dlp_hyp_kress_graded(::DLP_hyperbolic_kress,pts::BoundaryPointsHyp)=false
@inline _is_dlp_hyp_kress_graded(::DLP_hyperbolic_kress_global_corners,pts::BoundaryPointsHyp)=_is_nontrivial_hyp_grading(pts)
@inline function _is_nontrivial_hyp_grading(pts::BoundaryPointsHyp{T}) where {T<:Real};return maximum(abs.(pts.ws_der.-one(T)))>sqrt(eps(T));end
@inline _dlp_hyp_kress_use_reduced(solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners})=!isnothing(solver.symmetry)
# name compatibility with BIM_hyperbolic
function _hyp_beyn_dim(solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners},pts::BoundaryPointsHyp,k)
    ws=build_dlp_hyperbolic_kress_workspace(solver,pts,k)
    return _workspace_dim(ws)
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
#=
function evaluate_points(solver::DLP_hyperbolic_kress,billiard::Bi,k::Real,precomps::Vector{HyperArcCDFPrecomp{Float64}};safety::Real=1e-14,threaded::Bool=true) where {Bi<:AbsBilliard}
    curves=billiard.full_boundary
    real_idxs=findall(crv->typeof(crv)<:AbsRealCurve,curves)
    nreal=length(real_idxs)
    @assert nreal==length(precomps)
    T=eltype(solver.pts_scaling_factor)
    bs0=solver.pts_scaling_factor
    bs=length(bs0)==1 ? fill(bs0[1],nreal) : length(bs0)==nreal ? copy(bs0) : error("Expected scalar b or one b per real curve.")
    Ni=Vector{Int}(undef,nreal)
    for r in 1:nreal
        Ni[r]=max(solver.min_pts,round(Int,real(k)*precomps[r].Lh*bs[r]/TWO_PI))
    end
    needed=2
    if !isnothing(solver.symmetry)
        sym=solver.symmetry
        sym isa Rotation && (needed=lcm(needed,sym.n))
        sym isa Reflection && (needed=lcm(needed,4))
    end
    @inbounds for r in 1:nreal
        remN=mod(Ni[r],needed)
        remN!=0 && (Ni[r]+=needed-remN)
    end
    offs=Vector{Int}(undef,nreal+1)
    offs[1]=1
    @inbounds for r in 1:nreal
        offs[r+1]=offs[r]+Ni[r]
    end
    Ntot=offs[end]-1
    xy_all=Vector{SVector{2,T}}(undef,Ntot)
    normal_all=Vector{SVector{2,T}}(undef,Ntot)
    kappa_all=Vector{T}(undef,Ntot)
    ds_all=Vector{T}(undef,Ntot)
    λ_all=Vector{T}(undef,Ntot)
    dsH_all=Vector{T}(undef,Ntot)
    ξ_all=Vector{T}(undef,Ntot)
    tangent_all=Vector{SVector{2,T}}(undef,Ntot)
    tangent2_all=Vector{SVector{2,T}}(undef,Ntot)
    ts_all=Vector{T}(undef,Ntot)
    original_ts_all=Vector{T}(undef,Ntot)
    ws_all=Vector{T}(undef,Ntot)
    ws_der_all=Vector{T}(undef,Ntot)
    ranges=[offs[r]:(offs[r+1]-1) for r in 1:nreal]
    fill_one!(r)=begin
        crv=curves[real_idxs[r]]
        pre=precomps[r]
        Nr=Ni[r]
        rng=ranges[r]
        t,_=invert_cdf_midpoints(pre.ts_dense,pre.F_dense,Nr)
        pts=curve(crv,t)
        ta=tangent(crv,t)
        t2=tangent_2(crv,t)
        tu,_=_unit_tangents_and_speeds(ta)
        nrm=_normals_from_unit_tangents(tu)
        κE=curvature(crv,t)
        hH=T(pre.Lh)/T(Nr)
        hσ=TWO_PI/T(Nr)
        j0=first(rng)
        @inbounds for n in 1:Nr
            idx=j0+n-1
            p=pts[n]
            x=T(p[1])
            y=T(p[2])
            den=max(one(T)-muladd(x,x,y*y),T(1e-15))
            λ=T(2)/den
            σ=hσ*(T(n)-T(0.5))
            xy_all[idx]=SVector(x,y)
            normal_all[idx]=SVector(T(nrm[n][1]),T(nrm[n][2]))
            kappa_all[idx]=T(κE[n])
            dsH_all[idx]=hH
            λ_all[idx]=λ
            ds_all[idx]=hH/λ
            tangent_all[idx]=SVector(T(ta[n][1]),T(ta[n][2]))
            tangent2_all[idx]=SVector(T(t2[n][1]),T(t2[n][2]))
            ts_all[idx]=σ
            original_ts_all[idx]=T(t[n])
            ws_all[idx]=hσ
            ws_der_all[idx]=one(T)
        end
        return nothing
    end
    if threaded && Threads.nthreads()>1
        Threads.@threads for r in 1:nreal
            fill_one!(r)
        end
    else
        for r in 1:nreal
            fill_one!(r)
        end
    end
    s=zero(T)
    @inbounds for j in 1:Ntot
        ξ_all[j]=s
        s+=dsH_all[j]
    end
    return BoundaryPointsHyp{T}(xy_all,normal_all,kappa_all,ds_all,λ_all,dsH_all,ξ_all,s,tangent_all,tangent2_all,ts_all,original_ts_all,ws_all,ws_der_all)
end
=#
function evaluate_points(solver::DLP_hyperbolic_kress,billiard::Bi,k::Real,precomps::Vector{HyperArcCDFPrecomp{Float64}};safety::Real=1e-14,threaded::Bool=true,curv_alpha::Real=0.0,curv_kappa0::Real=1e-12,Mmap::Int=20000) where {Bi<:AbsBilliard}
    curves=billiard.full_boundary
    real_idxs=findall(crv->typeof(crv)<:AbsRealCurve,curves)
    nreal=length(real_idxs)
    @assert nreal==length(precomps)
    T=eltype(solver.pts_scaling_factor)
    bs0=solver.pts_scaling_factor
    bs=length(bs0)==1 ? fill(bs0[1],nreal) : length(bs0)==nreal ? copy(bs0) : error("Expected scalar b or one b per real curve.")
    Lρ=Vector{T}(undef,nreal)
    maps=Vector{Any}(undef,nreal)
    for r in 1:nreal
        crv=curves[real_idxs[r]]
        M=max(Mmap,4solver.min_pts)
        td=collect(range(0.0,2pi,length=M+1))[1:M]
        q=curve(crv,td); γt=tangent(crv,td); γtt=tangent_2(crv,td)
        ρ=Vector{Float64}(undef,M)
        @inbounds for i in 1:M
            x=Float64(q[i][1]);y=Float64(q[i][2])
            tx=Float64(γt[i][1]);ty=Float64(γt[i][2])
            txx=Float64(γtt[i][1]);tyy=Float64(γtt[i][2])
            sp=hypot(tx,ty)
            nx=ty/sp;ny=-tx/sp
            den=max(1.0-x*x-y*y,1e-15)
            λ=2.0/den
            κE=-(tx*tyy-ty*txx)/(sp^3)
            dnlogλ=2.0*(x*nx+y*ny)/den
            κH=(κE+dnlogλ)/λ
            ρ[i]=λ*sp*(1.0+Float64(curv_alpha)*sqrt(κH^2+Float64(curv_kappa0)^2)/Float64(k))
        end
        F=zeros(Float64,M+1)
        h=2pi/M
        @inbounds for i in 1:M
            ip=i==M ? 1 : i+1
            F[i+1]=F[i]+0.5h*(ρ[i]+ρ[ip])
        end
        maps[r]=(td=td,ρ=ρ,F=F,L=F[end],h=h)
        Lρ[r]=T(F[end])
    end
    Ni=Vector{Int}(undef,nreal)
    for r in 1:nreal
        Ni[r]=max(solver.min_pts,round(Int,real(k)*Lρ[r]*bs[r]/TWO_PI))
    end
    needed=2
    if !isnothing(solver.symmetry)
        sym=solver.symmetry
        sym isa Rotation && (needed=lcm(needed,sym.n))
        sym isa Reflection && (needed=lcm(needed,4))
    end
    @inbounds for r in 1:nreal
        remN=mod(Ni[r],needed)
        remN!=0 && (Ni[r]+=needed-remN)
    end
    offs=Vector{Int}(undef,nreal+1);offs[1]=1
    @inbounds for r in 1:nreal
        offs[r+1]=offs[r]+Ni[r]
    end
    Ntot=offs[end]-1
    xy_all=Vector{SVector{2,T}}(undef,Ntot)
    normal_all=Vector{SVector{2,T}}(undef,Ntot)
    kappa_all=Vector{T}(undef,Ntot)
    ds_all=Vector{T}(undef,Ntot)
    λ_all=Vector{T}(undef,Ntot)
    dsH_all=Vector{T}(undef,Ntot)
    ξ_all=Vector{T}(undef,Ntot)
    tangent_all=Vector{SVector{2,T}}(undef,Ntot)
    tangent2_all=Vector{SVector{2,T}}(undef,Ntot)
    ts_all=Vector{T}(undef,Ntot)
    original_ts_all=Vector{T}(undef,Ntot)
    ws_all=Vector{T}(undef,Ntot)
    ws_der_all=Vector{T}(undef,Ntot)
    fill_one!(r)=begin
        crv=curves[real_idxs[r]]
        Nr=Ni[r];j0=offs[r]
        td,ρ,F,L,h=maps[r]
        hσ=TWO_PI/T(Nr)
        @inbounds for n in 1:Nr
            σ=hσ*(T(n)-T(0.5))
            target=Float64(σ)/TWO_PI*L
            m=searchsortedlast(F,target)
            m=clamp(m,1,length(ρ))
            mp=m==length(ρ) ? 1 : m+1
            a=F[m];b=F[m+1]
            θ=(target-a)/(b-a)
            t=td[m]+θ*h
            t>=TWO_PI && (t-=TWO_PI)
            ρt=(ρ[mp]-ρ[m])/h
            ρloc=(1-θ)*ρ[m]+θ*ρ[mp]
            R=L
            tσ=R/(TWO_PI*ρloc)
            tσσ=-(R^2)*ρt/(TWO_PI^2*ρloc^3)
            q=curve(crv,t)
            γt_raw=tangent(crv,t)
            γtt_raw=tangent_2(crv,t)
            x=T(q[1]);y=T(q[2])
            γt=SVector(T(γt_raw[1]),T(γt_raw[2]))
            γtt=SVector(T(γtt_raw[1]),T(γtt_raw[2]))
            γσ=γt*T(tσ)
            γσσ=γtt*T(tσ^2)+γt*T(tσσ)
            sp=hypot(γσ[1],γσ[2])
            den=max(one(T)-muladd(x,x,y*y),T(1e-15))
            λ=T(2)/den
            idx=j0+n-1
            xy_all[idx]=SVector(x,y)
            tangent_all[idx]=γσ
            tangent2_all[idx]=γσσ
            normal_all[idx]=SVector(γσ[2]/sp,-γσ[1]/sp)
            kappa_all[idx]=-(γσ[1]*γσσ[2]-γσ[2]*γσσ[1])/(sp^3)
            ds_all[idx]=sp*hσ
            λ_all[idx]=λ
            dsH_all[idx]=λ*ds_all[idx]
            ts_all[idx]=σ
            original_ts_all[idx]=T(t)
            ws_all[idx]=hσ
            ws_der_all[idx]=T(tσ)
        end
        nothing
    end
    if threaded && Threads.nthreads()>1
        Threads.@threads for r in 1:nreal
            fill_one!(r)
        end
    else
        for r in 1:nreal
            fill_one!(r)
        end
    end
    s=zero(T)
    @inbounds for j in 1:Ntot
        ξ_all[j]=s
        s+=dsH_all[j]
    end
    return BoundaryPointsHyp{T}(xy_all,normal_all,kappa_all,ds_all,λ_all,dsH_all,ξ_all,s,tangent_all,tangent2_all,ts_all,original_ts_all,ws_all,ws_der_all)
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

"""
    _evaluate_points_hyp_global_corners(solver,comp,k,idx;threaded=true)

Construct the globally graded hyperbolic boundary discretization for one
composite cornered boundary component.

This is the corner-specific point generator used by
`DLP_hyperbolic_kress_global_corners`. The boundary is assumed to be a single
closed composite curve `comp`, represented by several smooth curve pieces joined
at corners.

The construction follows the Euclidean `DLP_kress_global_corners` logic:

    σ -> t = t(σ) -> γ(t),

where `σ` is the computational Kress variable and `t` is the global geometric
parameter of the composite boundary. The map `t(σ)` is generated by
`multi_kress_graded_nodes_data`, using the corner locations returned by
`_component_corner_locations`.

The chain rule gives

    γ_σ  = γ_t  * dt/dσ,
    γ_σσ = γ_tt * (dt/dσ)^2 + γ_t * d²t/dσ².

The Euclidean quadrature weight is

    ds = |γ_σ| dσ,

and the hyperbolic weight is then

    dsH = λ ds.

This order is intentional: the Kress grading regularizes the corner singularity
in the boundary parameter, while the Poincare conformal factor is applied only
after the Euclidean geometry has been transformed.

The returned `BoundaryPointsHyp` stores `ts = σ`, `original_ts = tmap`,
`ws = dσ`, and `ws_der = dt/dσ`. The latter lets the Kress matrix builder detect
nontrivial grading and use `kress_R_corner!`.
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
        u=τ[i]/TWO_PI
        acc=zero(T)
        crv_idx=length(comp)
        local_u=one(T)
        for j in eachindex(comp)
            frac=T(pres[j].Lh)/Lh
            if u<=acc+frac || j==lastindex(comp)
                crv_idx=j
                local_u=clamp((u-acc)/frac,zero(T),one(T))
                break
            end
            acc+=frac
        end
        pre=pres[crv_idx]
        M=length(pre.ts_dense)
        jj=2
        while jj<M && pre.F_dense[jj]<local_u
            jj+=1
        end
        f1=T(pre.F_dense[jj-1]); f2=T(pre.F_dense[jj])
        t1=T(pre.ts_dense[jj-1]); t2=T(pre.ts_dense[jj])
        α=(f2==f1) ? zero(T) : (local_u-f1)/(f2-f1)
        tt=muladd(α,t2-t1,t1)
        q=curve(comp[crv_idx],tt)
        γt_raw=tangent(comp[crv_idx],tt)
        γtt_raw=tangent_2(comp[crv_idx],tt)
        x=T(q[1]); y=T(q[2])
        γt=SVector(T(γt_raw[1]),T(γt_raw[2]))
        γtt=SVector(T(γtt_raw[1]),T(γtt_raw[2]))
        λ=λ_poincare(x,y)
        sp_raw=hypot(γt[1],γt[2])
        den=max(one(T)-muladd(x,x,y*y),T(1e-15))
        dλdt=T(4)*dot(SVector(x,y),γt)/(den^2)
        dspdt=dot(γt,γtt)/sp_raw
        Fp=λ*sp_raw/Lh
        Fpp=(dλdt*sp_raw+λ*dspdt)/Lh
        dt_dτ=inv(TWO_PI*Fp)
        d2t_dτ2=-Fpp/(TWO_PI^2*Fp^3)
        dt_dσ=dt_dτ*jac[i]
        d2t_dσ2=d2t_dτ2*(jac[i]^2)+dt_dτ*jac2[i]
        γσ=γt*dt_dσ
        γσσ=γtt*(dt_dσ^2)+γt*d2t_dσ2
        sp=hypot(γσ[1],γσ[2])
        xy[i]=SVector(x,y)
        tangent_1st[i]=γσ
        tangent_2nd[i]=γσσ
        normal[i]=SVector(γσ[2]/sp,-γσ[1]/sp)
        κE[i]=-(γσ[1]*γσσ[2]-γσ[2]*γσσ[1])/(sp^3)
        dsH[i]=T(Lh)*jac[i]*h/TWO_PI
        ds[i]=dsH[i]/λ
        λs[i]=λ
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

"""
    _evaluate_points_hyp_smooth_composite(solver,comp,k,idx;threaded=true)

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
function _evaluate_points_hyp_smooth_composite(solver::DLP_hyperbolic_kress_global_corners{T},comp::Vector{C},k::Real,idx::Int;threaded::Bool=true) where {T<:Real,C<:AbsCurve}
    pres=[precompute_hyper_cdf(crv;M=20_000,safety=1e-14) for crv in comp]
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
    ts=[TWO_PI*(T(j)-T(0.5))/T(N) for j in 1:N]
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
        q,γt,γtt=_eval_composite_geom_global_t(T,comp,ts[i])
        x=q[1];y=q[2]
        den=max(one(T)-muladd(x,x,y*y),T(1e-15))
        λ=T(2)/den
        sp=hypot(γt[1],γt[2])
        xy[i]=q
        tangent_1st[i]=γt
        tangent_2nd[i]=γtt
        normal[i]=SVector(γt[2]/sp,-γt[1]/sp)
        κE[i]=-(γt[1]*γtt[2]-γt[2]*γtt[1])/(sp^3)
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
    return BoundaryPointsHyp{T}(xy,normal,κE,ds,λs,dsH,ξ,s,tangent_1st,tangent_2nd,ts,copy(ts),ws,ws_der)
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
                dnd[i,j]=hyperbolic_dn_d_source(xi,yi,xj,yj,nxj,nyj)
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
    pre=build_QTaylorPrecomp(;dmin=max(Float64(dmin)*0.25,1e-15),dmax=Float64(dmax)*1.05)
    qws=QTaylorWorkspace(;threaded=false)
    qtab=alloc_QTaylorTable(pre;k=ComplexF64(k))
    ptab=alloc_PTaylorTable(pre;k=ComplexF64(k))
    build_QTaylorTable!(qtab,pre,qws,ComplexF64(k);mp_dps=mp_dps,leg_type=leg_type)
    build_PTaylorTable!(ptab,pre,qws,ComplexF64(k);mp_dps=mp_dps)
    return DLPHyperbolicKressTaylorWorkspace(pre,qws,qtab,ptab,ComplexF64(k))
end

function build_dlp_hyperbolic_kress_workspace_full(solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners},pts::BoundaryPointsHyp{T},k;mp_dps::Int=80,leg_type::Int=3) where {T<:Real}
    Rmat=build_Rmat_dlp_hyperbolic_kress(solver,pts)
    G=build_dlp_hyperbolic_kress_geom_cache(pts)
    taylor=build_dlp_hyperbolic_kress_taylor_workspace(pts,solver,k;mp_dps=mp_dps,leg_type=leg_type)
    return DLPHyperbolicKressWorkspace(Rmat,G,taylor,length(pts.xy))
end

function build_dlp_hyperbolic_kress_reduced_workspace(solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners},pts::BoundaryPointsHyp{T},k;mp_dps::Int=80,leg_type::Int=3) where {T<:Real}
    full=build_dlp_hyperbolic_kress_workspace_full(solver,pts,k;mp_dps=mp_dps,leg_type=leg_type)
    Ifund,full_to_fund,full_to_scale,fund_to_full,fund_to_scale=symmetry_index_orbits(T,pts,solver.symmetry,solver.billiard)
    xs=getindex.(pts.xy,1)
    ys=getindex.(pts.xy,2)
    nx=getindex.(pts.normal,1)
    ny=getindex.(pts.normal,2)
    wE=copy(pts.ds)
    return DLPHyperbolicKressReducedWorkspace(full,Ifund,full_to_fund,full_to_scale,fund_to_full,fund_to_scale,xs,ys,nx,ny,wE,length(Ifund))
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
    _dlp_hyp_kress_use_reduced(solver) ?
    build_dlp_hyperbolic_kress_reduced_workspace(solver,pts,k;mp_dps=mp_dps,leg_type=leg_type) :
    build_dlp_hyperbolic_kress_workspace_full(solver,pts,k;mp_dps=mp_dps,leg_type=leg_type)
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
    return Complex{T}((-G.kappaE[i]-G.dnlogλ[i])*INV_TWO_PI,zero(T))
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
function construct_dlp_hyperbolic_kress_matrix!(D::AbstractMatrix{Complex{T}},solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners},pts::BoundaryPointsHyp{T},ws::DLPHyperbolicKressWorkspace{T};multithreaded::Bool=true) where {T<:Real}
    N=ws.N
    R=ws.Rmat
    G=ws.G
    qtab=ws.taylor.qtab
    ptab=ws.taylor.ptab
    fill!(D,zero(Complex{T}))
    @inbounds for i in 1:N
        D[i,i]=pts.ds[i]*hyp_L2_diag_Kress(G,i)
    end
    @use_threads multithreading=(multithreaded && N>=32) for j in 2:N
        @inbounds for i in 1:j-1
            dij=Float64(G.d[i,j])
            l1ij=hyp_L1_kress(ptab,dij,G.dnd[i,j])
            l2ij=hyp_L2_kress(qtab,ptab,dij,G.dnd[i,j],G.logterm[i,j])
            hj=pts.ws[j]
            JEj=pts.ds[j]/hj
            D[i,j]=R[i,j]*(l1ij*JEj)+hj*(l2ij*JEj)
            dji=Float64(G.d[j,i])
            l1ji=hyp_L1_kress(ptab,dji,G.dnd[j,i])
            l2ji=hyp_L2_kress(qtab,ptab,dji,G.dnd[j,i],G.logterm[j,i])
            hi=pts.ws[i]
            JEi=pts.ds[i]/hi
            D[j,i]=R[j,i]*(l1ji*JEi)+hi*(l2ji*JEi)
        end
    end
    return D
end
# Convert the DLP operator into the Fredholm matrix : A = I - D. Eigenvalues are detected from singularity / near-singularity of A(k).
function construct_fredholm_hyperbolic_kress_matrix!(A::AbstractMatrix{Complex{T}},solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners},pts::BoundaryPointsHyp{T},ws::DLPHyperbolicKressWorkspace{T};multithreaded::Bool=true) where {T<:Real}
    construct_dlp_hyperbolic_kress_matrix!(A,solver,pts,ws;multithreaded=multithreaded)
    @inbounds for j in axes(A,2), i in axes(A,1)
        A[i,j]*=-1
    end
    @inbounds for i in axes(A,1)
        A[i,i]+=one(Complex{T})
    end
    return A
end

# Regular DLP contribution from a symmetry-image source. These sources are geometrically separated from the physical singular copy, so no Kress logarithmic correction is needed. This is simply the ordinary off-diagonal hyperbolic source-normal DLP kernel.
@inline function _regular_hyp_dlp_image_D(qtab::QTaylorTable,xi::T,yi::T,xj::T,yj::T,nxj::T,nyj::T,wj::T,scale::Complex{T}) where {T<:Real}
    d=hyperbolic_distance_poincare(xi,yi,xj,yj)
    d<=eps(T) && return zero(Complex{T})
    dn=hyperbolic_dn_d_source(xi,yi,xj,yj,nxj,nyj)
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
function construct_dlp_hyperbolic_kress_matrix!(D::AbstractMatrix{Complex{T}},solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners},pts::BoundaryPointsHyp{T},rws::DLPHyperbolicKressReducedWorkspace{T};multithreaded::Bool=true) where {T<:Real}
    m=rws.m
    @assert size(D,1)==m && size(D,2)==m
    full=rws.full
    R=full.Rmat
    G=full.G
    qtab=full.taylor.qtab
    ptab=full.taylor.ptab
    Ifund=rws.Ifund
    fill!(D,zero(Complex{T}))
    @use_threads multithreading=(multithreaded && m>=32) for b in 1:m
        j=Ifund[b]
        @inbounds for a in 1:m
            i=Ifund[a]
            if i==j
                D[a,b]=pts.ds[i]*hyp_L2_diag_Kress(G,i)
            else
                d=Float64(G.d[i,j])
                l1=hyp_L1_kress(ptab,d,G.dnd[i,j])
                l2=hyp_L2_kress(qtab,ptab,d,G.dnd[i,j],G.logterm[i,j])
                hj=pts.ws[j]
                JEj=pts.ds[j]/hj
                D[a,b]=R[i,j]*(l1*JEj)+hj*(l2*JEj)
            end
        end
    end
    for b in 1:m
        j=Ifund[b]
        @inbounds for a in 1:m
            i=Ifund[a]
            xi=rws.xs[i]
            yi=rws.ys[i]
            for l in eachindex(rws.fund_to_full[b])
                q=rws.fund_to_full[b][l]
                q==j && continue
                scale=rws.fund_to_scale[b][l]
                D[a,b]+=_regular_hyp_dlp_image_D(qtab,xi,yi,rws.xs[q],rws.ys[q],rws.nx[q],rws.ny[q],rws.wE[q],scale)
            end
        end
    end
    return D
end

function construct_fredholm_hyperbolic_kress_matrix!(A::AbstractMatrix{Complex{T}},solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners},pts::BoundaryPointsHyp{T},rws::DLPHyperbolicKressReducedWorkspace{T};multithreaded::Bool=true) where {T<:Real}
    construct_dlp_hyperbolic_kress_matrix!(A,solver,pts,rws;multithreaded=multithreaded)
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
function adjoint_fredholm_matrix!(A::AbstractMatrix{Complex{T}},D::AbstractMatrix{Complex{T}},solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners},pts::BoundaryPointsHyp{T},ws::DLPHyperbolicKressWorkspace{T};multithreaded::Bool=true) where {T<:Real}
    construct_dlp_hyperbolic_kress_matrix!(D,solver,pts,ws;multithreaded=multithreaded)
    ds=pts.ds
    N=ws.N
    @inbounds for i in 1:N, j in 1:N
        A[i,j]=-D[j,i]*ds[j]/ds[i]
    end
    @inbounds for i in 1:N
        A[i,i]+=one(Complex{T})
    end
    return A
end
function adjoint_fredholm_matrix!(A::AbstractMatrix{Complex{T}},D::AbstractMatrix{Complex{T}},solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners},pts::BoundaryPointsHyp{T},rws::DLPHyperbolicKressReducedWorkspace{T};multithreaded::Bool=true) where {T<:Real}
    construct_dlp_hyperbolic_kress_matrix!(D,solver,pts,rws;multithreaded=multithreaded)
    ds=pts.ds
    Ifund=rws.Ifund
    m=rws.m
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

function construct_matrices!(solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners},A::AbstractMatrix{Complex{T}},pts::BoundaryPointsHyp{T},ws::Union{DLPHyperbolicKressWorkspace{T},DLPHyperbolicKressReducedWorkspace{T}},k;multithreaded::Bool=true) where {T<:Real}
    construct_fredholm_hyperbolic_kress_matrix!(A,solver,pts,ws;multithreaded=multithreaded)
end

function construct_boundary_matrices!(Tbufs::Vector{Matrix{ComplexF64}},solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners},pts::BoundaryPointsHyp{T},zj::AbstractVector{ComplexF64};multithreaded::Bool=true,timeit::Bool=false) where {T<:Real}
    @blas_1 begin
        @inbounds for q in eachindex(zj)
            @benchit timeit=timeit "DLP_hyperbolic_kress Workspace" ws=build_dlp_hyperbolic_kress_workspace(solver,pts,zj[q])
            n=_workspace_dim(ws)
            @assert size(Tbufs[q])==(n,n) "Tbufs[$q] has size $(size(Tbufs[q])), but DLP-hyperbolic-Kress requires ($n,$n)."
            fill!(Tbufs[q],0.0+0.0im)
            @benchit timeit=timeit "DLP_hyperbolic_kress Assembly" construct_matrices!(solver,Tbufs[q],pts,ws,zj[q];multithreaded=multithreaded)
        end
    end
    return nothing
end

"""
    solve(solver,basis,pts,k; kwargs...)
    solve(solver,basis,pts,ws,k; kwargs...)
    solve(solver,basis,A,pts,ws,k; kwargs...)

Compute a scalar spectral diagnostic for a boundary integral eigenvalue problem.

This is the standard high-level entry point for detecting eigenvalues of the
Fredholm operator associated with `solver`. Depending on the chosen `which`
option, the returned scalar is typically:

- the smallest singular value (`which = :svd`),
- a determinant-based diagnostic (`which = :det`),

Method variants
---------------
The different method signatures allow progressive reuse of precomputed data:

    solve(solver,basis,pts,k)

Builds the solver workspace internally and constructs the Fredholm matrix.

    solve(solver,basis,pts,ws,k)

Reuses a previously constructed workspace `ws`, avoiding repeated geometry and
kernel precomputation.

    solve(solver,basis,A,pts,ws,k)

Fully allocation-minimizing variant. Reuses both the matrix storage `A` and the
workspace `ws`.

Arguments
---------
- `solver`:
    Boundary integral solver (e.g. DLP, CFIE, hyperbolic DLP-Kress).
- `basis`:
    Basis object controlling the spectral reduction strategy.
- `pts`:
    Boundary discretization points.
- `ws`:
    Optional reusable solver workspace.
- `A`:
    Optional preallocated Fredholm matrix storage.
- `k`:
    Spectral parameter / wavenumber.

Keyword arguments
-----------------
- `multithreaded=true`:
    Enable threaded matrix assembly when supported.
- `use_krylov=true`:
    Use Krylov iterative solver for the SVD-based diagnostic. This is typically faster for large matrices.
- `which=:svd`:
    Spectral diagnostic to compute.

Returns
-------
A scalar diagnostic whose minima indicate eigenvalues.
For boundary-function recovery (e.g. Husimi or wavefunction reconstruction),
use [`solve_vect`] instead.
"""
function solve(solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners},basis::Ba,A::AbstractMatrix{Complex{T}},pts::BoundaryPointsHyp{T},ws::Union{DLPHyperbolicKressWorkspace{T},DLPHyperbolicKressReducedWorkspace{T}},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:svd) where {T<:Real,Ba<:AbsBasis}
    @blas_1 construct_matrices!(solver,A,pts,ws,k;multithreaded=multithreaded)
    @svd_or_det_solve A use_krylov which MAX_BLAS_THREADS
end

function solve(solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners},basis::Ba,pts::BoundaryPointsHyp{T},ws::Union{DLPHyperbolicKressWorkspace{T},DLPHyperbolicKressReducedWorkspace{T}},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:svd) where {T<:Real,Ba<:AbsBasis}
    n=_workspace_dim(ws)
    A=Matrix{Complex{T}}(undef,n,n)
    return solve(solver,basis,A,pts,ws,k;multithreaded=multithreaded,use_krylov=use_krylov,which=which)
end

function solve(solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners},basis::Ba,pts::BoundaryPointsHyp{T},k;multithreaded::Bool=true,use_krylov::Bool=true, which::Symbol=:svd,mp_dps::Int=80,leg_type::Int=3) where {T<:Real,Ba<:AbsBasis}
    ws=build_dlp_hyperbolic_kress_workspace(solver,pts,k;mp_dps=mp_dps,leg_type=leg_type)
    return solve(solver,basis,pts,ws,k;multithreaded=multithreaded,use_krylov=use_krylov,which=which)
end

"""
    solve_vect(solver,basis,pts,k; kwargs...)
    solve_vect(solver,basis,pts,ws,k; kwargs...)
    solve_vect(solver,basis,A,pts,ws,k; kwargs...)

Compute the smallest null singular value together with the corresponding
boundary vector.

This function is used when the boundary representation of an eigenstate is
required, rather than only a scalar spectral diagnostic.

Unlike [`solve`], which returns only a scalar indicator, `solve_vect`
constructs and solves the adjoint Fredholm problem so that the returned vector
corresponds to the physically meaningful boundary function.

For source-normal DLP formulations this means solving

    A' = I - D'

rather than the primal operator

    A = I - D,

where

    D'ᵢⱼ = Dⱼᵢ wⱼ / wᵢ

with the appropriate quadrature weights for the formulation.

Method variants
---------------
The signatures differ only in how much reusable data is supplied:

    solve_vect(solver,basis,pts,k)

Build workspace and matrix storage internally.

    solve_vect(solver,basis,pts,ws,k)

Reuse a previously constructed workspace.

    solve_vect(solver,basis,A,pts,ws,k)

Fully allocation-minimizing variant with preallocated matrix storage.

Arguments
---------
- `solver`:
    Boundary integral solver.
- `basis`:
    Basis object controlling null-vector extraction.
- `pts`:
    Boundary discretization points.
- `ws`:
    Optional reusable solver workspace.
- `A`:
    Optional preallocated matrix storage.
- `k`:
    Spectral parameter / wavenumber.

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

Returns
-------
`(σ,u)`, where:

- `σ` is the smallest null singular value,
- `u` is the associated boundary vector.

This is the correct function for:

- boundary Husimi construction,
- wavefunction reconstruction,
- boundary current analysis,
- localization diagnostics based on boundary data.

If only eigenvalue detection is needed, [`solve`] is cheaper.
"""
function solve_vect(solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners},basis::Ba,A::AbstractMatrix{Complex{T}},pts::BoundaryPointsHyp{T},ws::Union{DLPHyperbolicKressWorkspace{T},DLPHyperbolicKressReducedWorkspace{T}},k;multithreaded::Bool=true,tol=1e-12,maxiter::Int=2000,krylovdim::Int=40) where {T<:Real,Ba<:AbsBasis}
    D=similar(A)
    @blas_1 adjoint_fredholm_matrix!(A,D,solver,pts,ws;multithreaded=multithreaded)
    σ,u,_=smallest_nullvec_krylov!(A;nev=1,tol=tol,maxiter=maxiter,krylovdim=krylovdim)
    return σ,u
end

function solve_vect(solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners},basis::Ba,pts::BoundaryPointsHyp{T},ws::Union{DLPHyperbolicKressWorkspace{T},DLPHyperbolicKressReducedWorkspace{T}},k;multithreaded::Bool=true,tol=1e-12,maxiter::Int=2000,krylovdim::Int=40) where {T<:Real,Ba<:AbsBasis}
    n=_workspace_dim(ws)
    A=Matrix{Complex{T}}(undef,n,n)
    return solve_vect(solver,basis,A,pts,ws,k;multithreaded=multithreaded,tol=tol,maxiter=maxiter,krylovdim=krylovdim)
end

function solve_vect(solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners},basis::Ba,pts::BoundaryPointsHyp{T},k;multithreaded::Bool=true,tol=1e-12,maxiter::Int=2000,krylovdim::Int=40,mp_dps::Int=80,leg_type::Int=3) where {T<:Real,Ba<:AbsBasis}
    ws=build_dlp_hyperbolic_kress_workspace(solver,pts,k;mp_dps=mp_dps,leg_type=leg_type)
    return solve_vect(solver,basis,pts,ws,k;multithreaded=multithreaded,tol=tol,maxiter=maxiter,krylovdim=krylovdim)
end