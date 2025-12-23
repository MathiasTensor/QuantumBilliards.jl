struct Hyperbolic<:AbsSampler end 

function _boundary_curves_for_solver(billiard::Bi,solver::BIM_hyperbolic) where {Bi<:AbsBilliard}
    if solver.symmetry === nothing
        hasproperty(billiard,:full_boundary) && return getfield(billiard,:full_boundary)
    end
    hasproperty(billiard,:desymmetrized_full_boundary) && return getfield(billiard,:desymmetrized_full_boundary)
    hasproperty(billiard,:full_boundary) && return getfield(billiard,:full_boundary)
    error("No usable boundary field found in $(typeof(billiard))")
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
# invert_cdf_midpoints(ts,F,N)->(tnodes,dtnodes)
#
# PURPOSE
#   Invert monotone CDF F(t) to obtain N parameter nodes placed at midpoint
#   targets in CDF space:
#     u_i=(i-0.5)/N,   i=1..N
#   Solve F(t_i)=u_i by piecewise-linear interpolation on (ts,F).
#
# INPUTS
#   ts : Vector{T} (length M), strictly increasing, typically in [0,1]
#   F  : Vector{T} (length M), nondecreasing, with F[1]≈0 and F[end]=1
#   N  : number of output nodes
#
# OUTPUTS
#   tnodes  : Vector{T} (length N), parameter nodes in [0,1]
#   dtnodes : Vector{T} (length N), parameter cell widths for midpoint rule:
#            dtnodes[i]=tnodes[i+1]-tnodes[i] (i=1..N-1)
#            dtnodes[N]=(1-tnodes[N])+tnodes[1] (wrap closure)
#
# NOTES
#   This dt is in parameter space. Euclidean ds uses |r'(t)|*dt.
#------------------------------------------------------------------------------
function invert_cdf_midpoints(ts::Vector{T},F::Vector{T},N::Int) where{T<:Real}
    N<=0 && return T[],T[]
    M=length(ts)
    tnodes=Vector{T}(undef,N)
    dtnodes=Vector{T}(undef,N)
    j=2
    @inbounds for i in 1:N
        ui=(T(i)-T(0.5))/T(N)
        while j<M && F[j]<ui
            j+=1
        end
        f1=F[j-1];f2=F[j]
        t1=ts[j-1];t2=ts[j]
        α=(f2==f1) ? zero(T) : (ui-f1)/(f2-f1)
        tnodes[i]=muladd(α,(t2-t1),t1)
    end
    @inbounds for i in 1:N-1
        dtnodes[i]=tnodes[i+1]-tnodes[i]
    end
    dtnodes[end]=(one(T)-tnodes[end])+tnodes[1]
    return tnodes,dtnodes
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
function precompute_hyperbolic_boundary_cdfs(solver::BIM_hyperbolic,billiard::Bi;M_cdf_base::Int=4000,safety::Real=1e-14) where{Bi<:QuantumBilliards.AbsBilliard}
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
# evaluate_points_hyperbolic(solver,billiard,k,precomps;...)->BoundaryPointsHypBIM
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
# OUTPUTS  (BoundaryPointsHypBIM{T})
#   xy[j]        : Euclidean boundary nodes (SVector{2,T})
#   normal[j]    : Euclidean unit normals (SVector{2,T})
#   curvature[j] : Euclidean curvature κ_E (T)
#   ds[j]        : Euclidean boundary weights ds_E (T)
#   λ[j]         : Poincaré conformal factor at xy[j] (T)
#   dsH[j]       : Hyperbolic boundary weights ds_H=λ*ds (T)
#   ξ[j]         : cumulative hyperbolic coordinate (T), ξ[1]=0
#   LH           : total hyperbolic boundary length, LH≈sum(dsH)
#------------------------------------------------------------------------------
function evaluate_points_hyperbolic(solver::BIM_hyperbolic,billiard::Bi,k::Real,precomps::Vector{HyperArcCDFPrecomp{Float64}};safety::Real=1e-14,threaded::Bool=true) where{Bi<:QuantumBilliards.AbsBilliard}
    bs,_=QuantumBilliards.adjust_scaling_and_samplers(solver,billiard)
    curves=QuantumBilliards._boundary_curves_for_solver(billiard,solver)
    T=eltype(solver.pts_scaling_factor)
    real_idxs=Int[]
    for i in eachindex(curves)
        crv=curves[i]
        (typeof(crv)<:AbsRealCurve) && push!(real_idxs,i)
    end
    nreal=length(real_idxs)
    @assert nreal==length(precomps)
    Ni=Vector{Int}(undef,nreal)
    Lh_curves=Vector{T}(undef,nreal)
    for r in 1:nreal
        i=real_idxs[r]
        pre=precomps[r]
        Lh=pre.Lh
        Lh_curves[r]=T(Lh)
        Ni[r]=max(solver.min_pts,round(Int,(real(k)*Lh*bs[i])/(2π)))
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
    ranges=Vector{UnitRange{Int}}(undef,nreal)
    @inbounds for r in 1:nreal
        ranges[r]=offs[r]:(offs[r+1]-1)
    end
    fill_one!(r)=begin
        i=real_idxs[r]
        crv=curves[i]
        pre=precomps[r]
        Ni_r=Ni[r]
        rng=ranges[r]
        t,dt=invert_cdf_midpoints(pre.ts_dense,pre.F_dense,Ni_r)
        pts=curve(crv,t)
        ta=tangent(crv,t)
        tu,sp=_unit_tangents_and_speeds(ta)
        nrm=_normals_from_unit_tangents(tu)
        κE=curvature(crv,t)
        @inbounds begin
            copyto!(view(xy_all,rng),pts)
            copyto!(view(normal_all,rng),nrm)
            copyto!(view(kappa_all,rng),κE)
        end
        j0=first(rng)
        @inbounds for n in 1:Ni_r
            p=pts[n]
            x=T(p[1]);y=T(p[2])
            den=one(T)-muladd(x,x,y*y);den=max(den,T(1e-15))
            lam=T(2)/den
            dsE=T(sp[n]*dt[n])
            idx=j0+n-1
            ds_all[idx]=dsE
            λ_all[idx]=lam
            dsH_all[idx]=lam*dsE
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
    LH=s
    return BoundaryPointsHypBIM{T}(xy_all,normal_all,kappa_all,ds_all,λ_all,dsH_all,ξ_all,LH)
end