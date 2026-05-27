struct Hyperbolic<:AbsSampler end 

struct BIM_hyperbolic{T<:Real,Sym}<:SweepSolver 
    dim_scaling_factor::T
    pts_scaling_factor::Vector{T}
    sampler::Vector
    eps::T
    min_dim::Int64 
    min_pts::Int64
    symmetry::Sym
end

# ==============================================================================
# BoundaryPointsHyp
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
struct BoundaryPointsHyp{T<:Real}
    xy::Vector{SVector{2,T}}
    normal::Vector{SVector{2,T}}
    kappa::Vector{T}
    ds::Vector{T}
    λ::Vector{T}
    dsH::Vector{T}
    ξ::Vector{T}
    LH::T
    tangent::Vector{SVector{2,T}}
    tangent_2::Vector{SVector{2,T}}
    ts::Vector{T}
    original_ts::Vector{T}
    ws::Vector{T}
    ws_der::Vector{T}
end

function _boundary_curves_for_solver(billiard::Bi,solver::BIM_hyperbolic) where {Bi<:AbsBilliard}
    if isnothing(solver.symmetry)
        hasproperty(billiard,:full_boundary) && return getfield(billiard,:full_boundary)
    end
    hasproperty(billiard,:desymmetrized_full_boundary) && return getfield(billiard,:desymmetrized_full_boundary)
    hasproperty(billiard,:full_boundary) && return getfield(billiard,:full_boundary)
    error("No usable boundary field found in $(typeof(billiard))")
end

#------------------------------------------------------------------------------
# _BoundaryPointsHypBIM_to_BoundaryPoints(bph)->bp
#
# PURPOSE
#   Convert a hyperbolic boundary-point container (BoundaryPointsHyp) into the
#   standard Euclidean points (BoundaryPoints) expected by low-level
#   Fredholm / layer-kernel matrix constructors.
#
#   PRESERVED (copied by reference, no allocation for vectors):
#     • xy        : Euclidean boundary nodes
#     • normal    : Euclidean unit normals
#     • curvature : Euclidean curvature κ_E
#     • ds        : Euclidean quadrature weights ds_E
#
#   DROPPED (ignored):
#     • λ, dsH, ξ, LH (hyperbolic extras)
#
# INPUTS
#   bph::BoundaryPointsHyp{T}
#     Fields used:
#       bph.xy, bph.normal, bph.curvature, bph.ds
#
# OUTPUTS
#   bp::BoundaryPoints{T}
#     Uses the same vector objects for xy/normal/curvature/ds (zero-copy).
#     shift_x and shift_y are set to 0 (no shifts in Poincaré disk workflow).
#------------------------------------------------------------------------------
@inline function _BoundaryPointsHypBIM_to_BoundaryPoints(bph::BoundaryPointsHyp{T}) where {T<:Real}
    return BoundaryPoints{T}(bph.xy,bph.normal,Vector{T}(),bph.ds,Vector{T}(),Vector{T}(),bph.kappa,Vector{SVector{2,T}}(),zero(T),zero(T))
end

#------------------------------------------------------------------------------
# HyperArcCDFPrecomp{T,C}
#
# PURPOSE
#   Geometry-only precomputation for ONE real boundary curve. Stores a dense
#   approximation to the hyperbolic arclength CDF F(t) over curve parameter t.
#
# FIELDS
#   crv      : curve object (AbsRealCurve subtype)
#   ts_dense : dense parameter grid t in [0,1] (Vector{T}, length M)
#   F_dense  : monotone CDF values in [0,1] (Vector{T}, length M)
#              F_dense[i]≈(hyperbolic arclength from 0 to ts_dense[i]) / Lh
#   Lh       : total hyperbolic length of the curve (T)
#
# USED BY
#   invert_cdf_midpoints(...) to place N quadrature nodes at equal hyperbolic
#   arclength spacing (midpoint rule in CDF-space).
#------------------------------------------------------------------------------
struct HyperArcCDFPrecomp{T<:Real,C}
    crv::C
    ts_dense::Vector{T}
    F_dense::Vector{T}
    Lh::T
end

#------------------------------------------------------------------------------
# _unit_tangents_and_speeds(ta)->(tu,sp)
#
# PURPOSE
#   Convert tangents r'(t) into:
#     tu(t)=r'(t)/|r'(t)|  (unit tangent)
#     sp(t)=|r'(t)|        (Euclidean speed)
#
# INPUTS
#   ta : Vector{SVector{2,T}}
#        ta[i]=r'(t_i) for a set of parameter nodes t_i.
#
# OUTPUTS
#   tu : Vector{SVector{2,T}}
#        tu[i]=unit tangent at t_i.
#   sp : Vector{T}
#        sp[i]=Euclidean speed |r'(t_i)|.
#------------------------------------------------------------------------------
@inline function _unit_tangents_and_speeds(ta::AbstractVector{SVector{2,T}}) where{T<:Real}
    N=length(ta)
    sp=Vector{T}(undef,N)
    tu=Vector{SVector{2,T}}(undef,N)
    @inbounds for i in 1:N
        s=norm(ta[i])
        sp[i]=s
        tu[i]=ta[i]/s
    end
    return tu,sp
end

#------------------------------------------------------------------------------
# _normals_from_unit_tangents(tu)->nrm
#
# PURPOSE
#   Build Euclidean unit outward normals from Euclidean unit tangents by a
#   +90° rotation: n=(ty,-tx).
#
# INPUTS
#   tu : Vector{SVector{2,T}}
#        Euclidean unit tangents.
#
# OUTPUTS
#   nrm : Vector{SVector{2,T}}
#         Euclidean unit normals corresponding to tu.
#------------------------------------------------------------------------------
@inline function _normals_from_unit_tangents(tu::AbstractVector{SVector{2,T}}) where{T<:Real}
    N=length(tu)
    nrm=Vector{SVector{2,T}}(undef,N)
    @inbounds for i in 1:N
        ti=tu[i]
        nrm[i]=SVector(ti[2],-ti[1])
    end
    return nrm
end

#------------------------------------------------------------------------------
# precompute_hyper_cdf(crv;M=4000,safety=1e-14)->HyperArcCDFPrecomp
#
# PURPOSE
#   Precompute dense hyperbolic arclength CDF for ONE curve:
#     dℓ_H = λ(r(t))*|r'(t)| dt
#   Then:
#     F(t)= (∫_0^t dℓ_H) / (∫_0^1 dℓ_H)  ∈ [0,1]
#
# INPUTS
#   crv : AbsRealCurve subtype
#         Must support curve(crv,t), tangent(crv,t).
#
# KEYWORDS
#   M     : number of dense samples on [0,1] used for trapezoidal integration.
#   safety: currently unused (reserved).
#
# OUTPUTS
#   HyperArcCDFPrecomp{Float64,typeof(crv)}
#------------------------------------------------------------------------------
function precompute_hyper_cdf(crv;M::Int=4000,safety::Real=1e-14)
    T=Float64
    ts=collect(range(T(0),T(1),length=M))
    ps=curve(crv,ts)
    ta=tangent(crv,ts)  
    w=Vector{T}(undef,M) 
    @inbounds for i in 1:M
        p=ps[i]
        lam=λ_poincare(T(p[1]),T(p[2]))
        w[i]=lam*norm(ta[i])
    end
    F=zeros(T,M)
    @inbounds for i in 2:M
        dt=ts[i]-ts[i-1]
        F[i]=F[i-1]+(w[i-1]+w[i])*dt/2
    end
    Lh=F[end]
    if !(Lh>0)
        F.=range(T(0),T(1),length=M)
        Lh=one(T)
    else
        F./=Lh
        F[end]=one(T)
    end
    return HyperArcCDFPrecomp{T,typeof(crv)}(crv,ts,F,Lh)
end

#------------------------------------------------------------------------------
# precompute_hyperbolic_boundary_cdfs(solver,billiard;...)->precomps
#
# PURPOSE
#   Build geometry-only HyperArcCDFPrecomp for each real boundary curve used
#   by the solver on this billiard. This is done ONCE and reused for many k.
#
# INPUTS
#   solver  : BoundaryIntegralMethod
#   billiard: AbsBilliard
#
# KEYWORDS
#   M_cdf_base : dense sampling size per curve for CDF construction
#   safety     : reserved
#
# OUTPUTS
#   precomps::Vector{HyperArcCDFPrecomp{Float64}}
#------------------------------------------------------------------------------
function precompute_hyperbolic_boundary_cdfs(solver::BIM_hyperbolic,billiard::Bi;M_cdf_base::Int=4000,safety::Real=1e-14) where{Bi<:AbsBilliard}
    curves=_boundary_curves_for_solver(billiard,solver)
    pre=HyperArcCDFPrecomp{Float64}[]
    for crv in curves
        if typeof(crv)<:AbsRealCurve
            push!(pre,precompute_hyper_cdf(crv;M=M_cdf_base,safety=safety))
        end
    end
    return pre
end

#------------------------------------------------------------------------------
# evaluate_points(solver,billiard,k,precomps;...)->BoundaryPointsHyp
#
# PURPOSE
#   Construct boundary quadrature nodes distributed UNIFORMLY in HYPERBOLIC
#   arclength along each real curve, but store Euclidean geometric quantities
#   (xy,normal,curvature) and Euclidean ds weights.
#
# INPUTS
#   solver::BIM_hyperbolic
#     Uses:
#       solver.min_pts  : minimum points per curve
#       solver.pts_scaling_factor (eltype sets output floating type T)
#
#   billiard::AbsBilliard
#     Used to fetch the boundary curves used by the solver.
#
#   k::Real
#     Wavenumber used only for point counts:
#       Ni=max(min_pts,round(k*Lh_curve*bs[i]/(2π))).
#
#   precomps::Vector{HyperArcCDFPrecomp{Float64}}
#     Geometry-only CDF objects. Must correspond 1-to-1 with real curves.
#
# KEYWORDS
#   safety::Real  : reserved
#   threaded::Bool: if true, parallelize across curves
#
# OUTPUTS  (BoundaryPointsHyp{T})
#   xy[j]        : Euclidean boundary nodes (SVector{2,T})
#   normal[j]    : Euclidean unit normals (SVector{2,T})
#   curvature[j] : Euclidean curvature κ_E (T)
#   ds[j]        : Euclidean boundary weights ds_E (T)
#   λ[j]         : Poincaré conformal factor at xy[j] (T)
#   dsH[j]       : Hyperbolic boundary weights ds_H=λ*ds (T)
#   ξ[j]         : cumulative hyperbolic coordinate (T), ξ[1]=0
#   LH           : total hyperbolic boundary length, LH≈sum(dsH)
#------------------------------------------------------------------------------
function evaluate_points(solver::BIM_hyperbolic,billiard::Bi,k::Real,precomps::Vector{HyperArcCDFPrecomp{Float64}};safety::Real=1e-14,threaded::Bool=true) where {Bi<:AbsBilliard}
    curves=_boundary_curves_for_solver(billiard,solver)
    real_idxs=Int[]
    for i in eachindex(curves)
        typeof(curves[i])<:AbsRealCurve && push!(real_idxs,i)
    end
    nreal=length(real_idxs)
    @assert nreal==length(precomps)
    T=eltype(solver.pts_scaling_factor)
    bs0=solver.pts_scaling_factor
    bs=length(bs0)==1 ? fill(bs0[1],nreal) :
       length(bs0)==nreal ? copy(bs0) :
       error("Expected scalar b or one b per real curve; got $(length(bs0)), expected 1 or $nreal.")
    Ni=Vector{Int}(undef,nreal)
    for r in 1:nreal
        Lh=precomps[r].Lh
        Ni[r]=max(solver.min_pts,round(Int,real(k)*Lh*bs[r]/(2π)))
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
        Nloc=Ni[r]
        rng=ranges[r]
        t,dt=invert_cdf_midpoints(pre.ts_dense,pre.F_dense,Nloc)
        pts=curve(crv,t)
        ta=tangent(crv,t)
        t2=tangent_2(crv,t)
        tu,sp=_unit_tangents_and_speeds(ta)
        nrm=_normals_from_unit_tangents(tu)
        κE=curvature(crv,t)
        h=T(2π)/T(Nloc)
        j0=first(rng)
        @inbounds for n in 1:Nloc
            idx=j0+n-1
            p=pts[n]
            x=T(p[1])
            y=T(p[2])
            den=max(one(T)-muladd(x,x,y*y),T(1e-15))
            lam=T(2)/den
            dsH=T(pre.Lh)/T(Nloc)
            dsE=dsH/lam
            xy_all[idx]=SVector{2,T}(p)
            normal_all[idx]=SVector{2,T}(nrm[n])
            kappa_all[idx]=T(κE[n])
            ds_all[idx]=dsE
            λ_all[idx]=lam
            dsH_all[idx]=dsH
            tangent_all[idx]=SVector{2,T}(ta[n])
            tangent2_all[idx]=SVector{2,T}(t2[n])
            ts_all[idx]=T(2π)*T(t[n])
            original_ts_all[idx]=ts_all[idx]
            ws_all[idx]=h
            ws_der_all[idx]=dsE/(sp[n]*h)
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