#################################################################
#   CHEBYSHEV-BASED SLP/DLP EVALUATION FOR CFIE_alpert ASSEMBLY IN 2D EUCLIDEAN HELMHOLTZ
# Functions to build Chebyshev-based SLP/DLP evaluation plans for multiple wavenumbers,
# and to compute the CFIE_alpert matrix blocks using these plans.
#
# Logic:
# - Build Chebyshev-based Hankel/Bessel evaluation plans for the SLP and DLP kernels.
# - Precompute pairwise geometry data (R,invR,inner,speed,weights,panel ids,tloc) for all blocks.
# - Reuse the existing Alpert self-panel logic, but replace direct H0/H1/J0/J1 calls by Chebyshev evaluations.
# - For off-diagonal / image / symmetry blocks, use the same geometry loops as the source code, but read kernels from plans.
#
# API:
# - `build_CFIE_alpert_plans(...)`
# - `build_cfie_alpert_cheb_block_caches(...)`
# - `compute_kernel_matrices_CFIE_alpert_chebyshev!(...)`
#
# USUALLY NOT CALLED DIRECTLY:
# - `_one_k_nosymm_CFIE_alpert_chebyshev!(...)`
# - `_all_k_nosymm_CFIE_alpert_chebyshev!(...)`
# - `construct_matrices_symmetry_chebyshev!(...)`
#
# Workflow:
# 1. Call `build_cfie_alpert_cheb_block_caches(...)`
# 2. Call `build_CFIE_alpert_plans(...)`
# 3. Create `CFIEMultiBesselWorkspace(...)`
# 4. Call `compute_kernel_matrices_CFIE_alpert_chebyshev!(...)`
#
# MO 26/3/26
#################################################################

_TWO_PI=2*pi
_INV_TWO_PI=1/_TWO_PI

#######################
#### MULTI-K H0/H1 ####
#######################

@inline function bessels_multi_ks_at_r!(h0vals::AbstractVector{ComplexF64},h1vals::AbstractVector{ComplexF64},j0vals::AbstractVector{ComplexF64},j1vals::AbstractVector{ComplexF64},plans0::AbstractVector{ChebHankelPlanH},plans1::AbstractVector{ChebHankelPlanH},plansj0::AbstractVector{ChebJPlan},plansj1::AbstractVector{ChebJPlan},pidx::Int32,t::Float64)
    @inbounds for m in eachindex(plans0)
        h0vals[m]=_cheb_clenshaw(plans0[m].panels[pidx].c,t)
        h1vals[m]=_cheb_clenshaw(plans1[m].panels[pidx].c,t)
        j0vals[m]=_cheb_clenshaw(plansj0[m].panels[pidx].c,t)
        j1vals[m]=_cheb_clenshaw(plansj1[m].panels[pidx].c,t)
    end
    return nothing
end

@inline function hankels_multi_ks_at_r!(h0vals::AbstractVector{ComplexF64},h1vals::AbstractVector{ComplexF64},plans0::AbstractVector{ChebHankelPlanH},plans1::AbstractVector{ChebHankelPlanH},pidx::Int32,t::Float64)
    @inbounds for m in eachindex(plans0)
        h0vals[m]=_cheb_clenshaw(plans0[m].panels[pidx].c,t)
        h1vals[m]=_cheb_clenshaw(plans1[m].panels[pidx].c,t)
    end
    return nothing
end

###############################
#### SINGLE-K HELPER EVALS ####
###############################

@inline function _bessels_one_k_at_r(plan0::ChebHankelPlanH,plan1::ChebHankelPlanH,planj0::ChebJPlan,planj1::ChebJPlan,pidx::Int32,t::Float64)
    h0=_cheb_clenshaw(plan0.panels[pidx].c,t)
    h1=_cheb_clenshaw(plan1.panels[pidx].c,t)
    j0=_cheb_clenshaw(planj0.panels[pidx].c,t)
    j1=_cheb_clenshaw(planj1.panels[pidx].c,t)
    return h0,h1,j0,j1
end

@inline function _hankels_one_k_at_r(plan0::ChebHankelPlanH,plan1::ChebHankelPlanH,pidx::Int32,t::Float64)
    h0=_cheb_clenshaw(plan0.panels[pidx].c,t)
    h1=_cheb_clenshaw(plan1.panels[pidx].c,t)
    return h0,h1
end

################################
#### PLAN BUILDERS FOR ALPERT ###
################################

function build_CFIE_alpert_plans(ks::AbstractVector{<:Number},rmin::Float64,rmax::Float64;npanels::Int=10000,M::Int=5,grading::Symbol=:uniform,geo_ratio::Real=1.05,nthreads::Int=1)
    Mk=length(ks)
    plans0=Vector{ChebHankelPlanH}(undef,Mk)
    plans1=Vector{ChebHankelPlanH}(undef,Mk)
    plansj0=Vector{ChebJPlan}(undef,Mk)
    plansj1=Vector{ChebJPlan}(undef,Mk)
    if nthreads<=1 || Mk==1
        @inbounds for m in 1:Mk
            k=ComplexF64(ks[m])
            plans0[m]=plan_h(0,1,k,rmin,rmax;npanels=npanels,M=M,grading=grading,geo_ratio=geo_ratio)
            plans1[m]=plan_h(1,1,k,rmin,rmax;npanels=npanels,M=M,grading=grading,geo_ratio=geo_ratio)
            plansj0[m]=plan_j(0,k,rmin,rmax;npanels=npanels,M=M,grading=grading,geo_ratio=geo_ratio)
            plansj1[m]=plan_j(1,k,rmin,rmax;npanels=npanels,M=M,grading=grading,geo_ratio=geo_ratio)
        end
    else
        nt=min(nthreads,Mk)
        chunks=Vector{UnitRange{Int}}(undef,nt)
        base=div(Mk,nt)
        remn=rem(Mk,nt)
        s=1
        for t in 1:nt
            len=base+(t<=remn ? 1 : 0)
            chunks[t]=s:(s+len-1)
            s+=len
        end
        Threads.@threads for tid in 1:nt
            @inbounds for m in chunks[tid]
                k=ComplexF64(ks[m])
                plans0[m]=plan_h(0,1,k,rmin,rmax;npanels=npanels,M=M,grading=grading,geo_ratio=geo_ratio)
                plans1[m]=plan_h(1,1,k,rmin,rmax;npanels=npanels,M=M,grading=grading,geo_ratio=geo_ratio)
                plansj0[m]=plan_j(0,k,rmin,rmax;npanels=npanels,M=M,grading=grading,geo_ratio=geo_ratio)
                plansj1[m]=plan_j(1,k,rmin,rmax;npanels=npanels,M=M,grading=grading,geo_ratio=geo_ratio)
            end
        end
    end
    return plans0,plans1,plansj0,plansj1
end

#################################
#### COMPONENT / BLOCK CACHE ####
#################################

struct CFIEAlpertBlockCache{T<:Real}
    row_offset::Int
    col_offset::Int
    Ni::Int
    Nj::Int
    R::Matrix{T}
    invR::Matrix{T}
    inner::Matrix{T}
    speed_j::Vector{T}
    wj::Vector{T}
    same::Bool
    pidx::Matrix{Int32}
    tloc::Matrix{Float64}
end

struct CFIEAlpertBlockSystemCache{T<:Real}
    blocks::Matrix{CFIEAlpertBlockCache{T}}
    offsets::Vector{Int}
    rmin::Float64
    rmax::Float64
end

function build_cfie_alpert_cheb_block_caches(comps::Vector{BoundaryPointsCFIE{T}};npanels::Int=10000,M::Int=5,grading::Symbol=:uniform,geo_ratio::Real=1.05,pad=(T(0.95),T(1.05))) where {T<:Real}
    nc=length(comps)
    offs=component_offsets(comps)
    blocks=Matrix{CFIEAlpertBlockCache{T}}(undef,nc,nc)
    global_rmin=typemax(T)
    global_rmax=zero(T)
    for a in 1:nc, b in 1:nc
        pa=comps[a]
        pb=comps[b]
        Ni=length(pa.xy)
        Nj=length(pb.xy)
        Xa=getindex.(pa.xy,1)
        Ya=getindex.(pa.xy,2)
        Xb=getindex.(pb.xy,1)
        Yb=getindex.(pb.xy,2)
        dXb=getindex.(pb.tangent,1)
        dYb=getindex.(pb.tangent,2)
        ΔX=reshape(Xa,Ni,1).-reshape(Xb,1,Nj)
        ΔY=reshape(Ya,Ni,1).-reshape(Yb,1,Nj)
        R=hypot.(ΔX,ΔY)
        invR=similar(R)
        @inbounds for j in 1:Nj, i in 1:Ni
            rij=R[i,j]
            invR[i,j]=rij>eps(T) ? inv(rij) : zero(T)
        end
        dXbr=reshape(dXb,1,Nj)
        dYbr=reshape(dYb,1,Nj)
        inner=dYbr.*ΔX.-dXbr.*ΔY
        speed_j=sqrt.(dXb.^2 .+ dYb.^2)
        wj=copy(pb.ws)
        same=(a==b)
        rmin_blk=typemax(T)
        rmax_blk=zero(T)
        @inbounds for j in 1:Nj, i in 1:Ni
            if same && i==j
                continue
            end
            rij=R[i,j]
            if rij>eps(T)
                rij<rmin_blk && (rmin_blk=rij)
                rij>rmax_blk && (rmax_blk=rij)
            end
        end
        @assert isfinite(rmin_blk) && rmax_blk>zero(T)
        rmin_blk=pad[1]*rmin_blk
        rmax_blk=pad[2]*rmax_blk
        global_rmin=min(global_rmin,rmin_blk)
        global_rmax=max(global_rmax,rmax_blk)
        pidx=Matrix{Int32}(undef,Ni,Nj)
        tloc=Matrix{Float64}(undef,Ni,Nj)
        blocks[a,b]=CFIEAlpertBlockCache{T}(offs[a],offs[b],Ni,Nj,R,invR,inner,speed_j,wj,same,pidx,tloc)
    end
    pref_plan=plan_h(0,1,1.0+0im,Float64(global_rmin),Float64(global_rmax);npanels=npanels,M=M,grading=grading,geo_ratio=geo_ratio)
    pans=pref_plan.panels
    for a in 1:nc, b in 1:nc
        blk=blocks[a,b]
        @inbounds for j in 1:blk.Nj, i in 1:blk.Ni
            if blk.same && i==j
                blk.pidx[i,j]=Int32(1)
                blk.tloc[i,j]=0.0
            else
                rij=Float64(blk.R[i,j])
                p=_find_panel(pref_plan,rij)
                P=pans[p]
                blk.pidx[i,j]=Int32(p)
                blk.tloc[i,j]=(2*rij-(P.b+P.a))/(P.b-P.a)
            end
        end
    end
    return CFIEAlpertBlockSystemCache{T}(blocks,offs,Float64(global_rmin),Float64(global_rmax))
end

##########################
#### BESSEL WORKSPACE ####
##########################

struct CFIEMultiBesselWorkspace
    h0_tls::Vector{Vector{ComplexF64}}
    h1_tls::Vector{Vector{ComplexF64}}
    j0_tls::Vector{Vector{ComplexF64}}
    j1_tls::Vector{Vector{ComplexF64}}
end

function CFIEMultiBesselWorkspace(Mk::Int;ntls::Int=Threads.nthreads())
    h0_tls=[Vector{ComplexF64}(undef,Mk) for _ in 1:ntls]
    h1_tls=[Vector{ComplexF64}(undef,Mk) for _ in 1:ntls]
    j0_tls=[Vector{ComplexF64}(undef,Mk) for _ in 1:ntls]
    j1_tls=[Vector{ComplexF64}(undef,Mk) for _ in 1:ntls]
    return CFIEMultiBesselWorkspace(h0_tls,h1_tls,j0_tls,j1_tls)
end

###############################
#### CHEB ALPERT WORKSPACE ####
###############################

struct CFIEAlpertChebWorkspace{T<:Real,C}
    alpert_ws::CFIEAlpertWorkspace{T,C}
    cheb_blocks::CFIEAlpertBlockSystemCache{T}
end

function build_cfie_alpert_cheb_workspace(solver::CFIE_alpert{T},pts::Vector{BoundaryPointsCFIE{T}};npanels::Int=10000,M::Int=5,grading::Symbol=:uniform,geo_ratio::Real=1.05,pad=(T(0.95),T(1.05))) where {T<:Real}
    aws=build_cfie_alpert_workspace(solver,pts)
    cws=build_cfie_alpert_cheb_block_caches(pts;npanels=npanels,M=M,grading=grading,geo_ratio=geo_ratio,pad=pad)
    return CFIEAlpertChebWorkspace{T,eltype(aws.Cs)}(aws,cws)
end

#################
#### HELPERS ####
#################

@inline dlp_weight(pts::BoundaryPointsCFIE,j::Int)=pts.ws[j]
@inline function slp_weight(pts::BoundaryPointsCFIE{T},j::Int,sj::T) where {T<:Real}
    return pts.ws[j]*sj
end

@inline function _component_id_of_panel(a::Int,gmaps::Vector{Vector{Int}})
    @inbounds for c in eachindex(gmaps)
        a in gmaps[c] && return c
    end
    return 0
end

@inline function _panel_xy_tangent_arrays(pts::BoundaryPointsCFIE{T}) where {T<:Real}
    X=getindex.(pts.xy,1)
    Y=getindex.(pts.xy,2)
    dX=getindex.(pts.tangent,1)
    dY=getindex.(pts.tangent,2)
    return X,Y,dX,dY
end

@inline function _scatter_local4!(A::AbstractMatrix{Complex{T}},gi::Int,col_range::UnitRange{Int},coeff::Complex{T},idx,wt) where {T<:Real}
    @inbounds for m in 1:4
        q=idx[m]
        A[gi,col_range[q]]+=coeff*wt[m]
    end
    return nothing
end

@inline function _right_neighbor_excluded_count(i::Int,N::Int,a::Int)
    return max(0,i+a-1-N)
end

@inline function _left_neighbor_excluded_count(i::Int,a::Int)
    return max(0,a-i)
end

@inline function _eval_on_open_panel_local4(pts::BoundaryPointsCFIE{T},u::T) where {T<:Real}
    X,Y,dX,dY=_panel_xy_tangent_arrays(pts)
    h=pts.ws[1]
    return _eval_shifted_source_smooth_panel_local4(u,h,X,Y,dX,dY)
end

@inline function _check_r(r,name,i,j)
    if !(isfinite(r)) || r<=sqrt(eps(eltype(r)))
        @warn "Bad distance in $name at i=$i j=$j : r=$r"
    end
end

###########################################################
################ SELF ALPERT ASSEMBLY #####################
###########################################################

function _assemble_self_alpert_periodic_cheb!(A::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},G::CFIEGeomCache{T},C::AlpertPeriodicCache{T},row_range::UnitRange{Int},k::T,rule::AlpertLogRule{T},blk::CFIEAlpertBlockCache{T},plan0::ChebHankelPlanH,plan1::ChebHankelPlanH,planj0::ChebJPlan,planj1::ChebJPlan;multithreaded::Bool=true) where {T<:Real}
    αD=Complex{T}(0,k/2)
    αS=Complex{T}(0,one(T)/2)
    ik=Complex{T}(0,k)
    X=getindex.(pts.xy,1)
    Y=getindex.(pts.xy,2)
    N=length(pts.ts)
    a=rule.a
    jcorr=rule.j
    h=pts.ws[1]
    @use_threads multithreading=multithreaded for i in 1:N
        gi=row_range[i]
        xi=X[i]
        yi=Y[i]
        si=G.speed[i]
        κi=G.kappa[i]
        A[gi,gi]+=one(Complex{T})-Complex{T}(h*si*κi,zero(T))
        @inbounds for j in 1:N
            j==i && continue
            gj=row_range[j]
            p=blk.pidx[i,j]
            t=blk.tloc[i,j]
            _,h1,_,_=_bessels_one_k_at_r(plan0,plan1,planj0,planj1,p,t)
            rij=G.R[i,j]
            inn=G.inner[i,j]
            invr=G.invR[i,j]
            A[gi,gj]-=h*(αD*inn*h1*invr)
        end
        @inbounds for j in 1:N
            j==i && continue
            m=j-i
            m>N÷2 && (m-=N)
            m<-N÷2 && (m+=N)
            abs(m)<a && continue
            gj=row_range[j]
            p=blk.pidx[i,j]
            t=blk.tloc[i,j]
            h0,_=_hankels_one_k_at_r(plan0,plan1,p,t)
            A[gi,gj]-=ik*(h*(αS*h0*G.speed[j]))
        end
        @inbounds for pidx_al in 1:jcorr
            fac=h*rule.w[pidx_al]
            dx=xi-C.xp[pidx_al,i]
            dy=yi-C.yp[pidx_al,i]
            r=hypot(dx,dy)
            if r>sqrt(eps(T))
                p=Int32(_find_panel(plan0,Float64(r)))
                P=plan0.panels[p]
                t=(2*Float64(r)-(P.b+P.a))/(P.b-P.a)
                h0,_=_hankels_one_k_at_r(plan0,plan1,p,t)
                coeff=-ik*(fac*(αS*h0*C.sp[pidx_al,i]))
                for m4 in 1:4
                    q=C.idxp[pidx_al,i,m4]
                    A[gi,row_range[q]]+=coeff*C.wtp[pidx_al,i,m4]
                end
            end
            dx=xi-C.xm[pidx_al,i]
            dy=yi-C.ym[pidx_al,i]
            r=hypot(dx,dy)
            if r>sqrt(eps(T))
                p=Int32(_find_panel(plan0,Float64(r)))
                P=plan0.panels[p]
                t=(2*Float64(r)-(P.b+P.a))/(P.b-P.a)
                h0,_=_hankels_one_k_at_r(plan0,plan1,p,t)
                coeff=-ik*(fac*(αS*h0*C.sm[pidx_al,i]))
                for m4 in 1:4
                    q=C.idxm[pidx_al,i,m4]
                    A[gi,row_range[q]]+=coeff*C.wtm[pidx_al,i,m4]
                end
            end
        end
    end
    return A
end

function _assemble_self_alpert_smooth_panel_cheb!(solver::CFIE_alpert{T},A::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},G::CFIEGeomCache{T},C::AlpertSmoothPanelCache{T},row_range::UnitRange{Int},k::T,rule::AlpertLogRule{T},blk::CFIEAlpertBlockCache{T},plan0::ChebHankelPlanH,plan1::ChebHankelPlanH,planj0::ChebJPlan,planj1::ChebJPlan;multithreaded::Bool=true) where {T<:Real}
    αD=Complex{T}(0,k/2)
    αS=Complex{T}(0,one(T)/2)
    ik=Complex{T}(0,k)
    X=getindex.(pts.xy,1)
    Y=getindex.(pts.xy,2)
    N=length(X)
    h=pts.ws[1]
    a=rule.a
    jcorr=rule.j
    @use_threads multithreading=multithreaded for i in 1:N
        gi=row_range[i]
        xi=X[i]
        yi=Y[i]
        si=G.speed[i]
        κi=G.kappa[i]
        A[gi,gi]+=one(Complex{T})-Complex{T}(h*si*κi,zero(T))
        @inbounds for j in 1:N
            j==i && continue
            gj=row_range[j]
            p=blk.pidx[i,j]
            t=blk.tloc[i,j]
            _,h1,_,_=_bessels_one_k_at_r(plan0,plan1,planj0,planj1,p,t)
            rij=G.R[i,j]
            inn=G.inner[i,j]
            invr=G.invR[i,j]
            A[gi,gj]-=h*(αD*inn*h1*invr)
        end
        @inbounds for j in 1:N
            j==i && continue
            abs(j-i)<a && continue
            gj=row_range[j]
            p=blk.pidx[i,j]
            t=blk.tloc[i,j]
            h0,_=_hankels_one_k_at_r(plan0,plan1,p,t)
            A[gi,gj]-=ik*(h*(αS*h0*G.speed[j]))
        end
        @inbounds for pidx_al in 1:jcorr
            fac=h*rule.w[pidx_al]
            Δu=h*rule.x[pidx_al]
            ui=C.us[i]
            if ui+Δu<one(T)
                dx=xi-C.xp[pidx_al,i]
                dy=yi-C.yp[pidx_al,i]
                r=hypot(dx,dy)
                if isfinite(r) && r>sqrt(eps(T))
                    p=Int32(_find_panel(plan0,Float64(r)))
                    P=plan0.panels[p]
                    t=(2*Float64(r)-(P.b+P.a))/(P.b-P.a)
                    h0,_=_hankels_one_k_at_r(plan0,plan1,p,t)
                    coeff=-ik*(fac*(αS*h0*C.sp[pidx_al,i]))
                    for m4 in 1:4
                        q=C.idxp[pidx_al,i,m4]
                        A[gi,row_range[q]]+=coeff*C.wtp[pidx_al,i,m4]
                    end
                end
            end
            if ui-Δu>zero(T)
                dx=xi-C.xm[pidx_al,i]
                dy=yi-C.ym[pidx_al,i]
                r=hypot(dx,dy)
                if isfinite(r) && r>sqrt(eps(T))
                    p=Int32(_find_panel(plan0,Float64(r)))
                    P=plan0.panels[p]
                    t=(2*Float64(r)-(P.b+P.a))/(P.b-P.a)
                    h0,_=_hankels_one_k_at_r(plan0,plan1,p,t)
                    coeff=-ik*(fac*(αS*h0*C.sm[pidx_al,i]))
                    for m4 in 1:4
                        q=C.idxm[pidx_al,i,m4]
                        A[gi,row_range[q]]+=coeff*C.wtm[pidx_al,i,m4]
                    end
                end
            end
        end
    end
    return A
end

function _assemble_self_alpert_cheb!(solver::CFIE_alpert{T},A::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},G::CFIEGeomCache{T},C,row_range::UnitRange{Int},k::T,rule::AlpertLogRule{T},blk::CFIEAlpertBlockCache{T},plan0::ChebHankelPlanH,plan1::ChebHankelPlanH,planj0::ChebJPlan,planj1::ChebJPlan;multithreaded::Bool=true) where {T<:Real}
    return pts.is_periodic ? _assemble_self_alpert_periodic_cheb!(A,pts,G,C,row_range,k,rule,blk,plan0,plan1,planj0,planj1;multithreaded=multithreaded) : _assemble_self_alpert_smooth_panel_cheb!(solver,A,pts,G,C,row_range,k,rule,blk,plan0,plan1,planj0,planj1;multithreaded=multithreaded)
end

function _assemble_self_alpert_composite_component_cheb!(solver::CFIE_alpert{T},A::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},Gs::Vector{CFIEGeomCache{T}},Cs,offs::Vector{Int},k::T,rule::AlpertLogRule{T},topo::AlpertCompositeTopology{T},gmap::Vector{Int},blocks::Matrix{CFIEAlpertBlockCache{T}},plan0::ChebHankelPlanH,plan1::ChebHankelPlanH,planj0::ChebJPlan,planj1::ChebJPlan;multithreaded::Bool=true) where {T<:Real}
    αD=Complex{T}(0,k/2)
    αS=Complex{T}(0,one(T)/2)
    ik=Complex{T}(0,k)
    a=rule.a
    jcorr=rule.j
    @inbounds for l in eachindex(gmap)
        aidx=gmap[l]
        pa=pts[aidx]
        Ga=Gs[aidx]
        Ca=Cs[aidx]
        ra=offs[aidx]:(offs[aidx+1]-1)
        Xa=getindex.(pa.xy,1)
        Ya=getindex.(pa.xy,2)
        Na=length(pa.xy)
        ha=pa.ws[1]
        left_smooth=topo.left_kind[l]===:smooth
        right_smooth=topo.right_kind[l]===:smooth
        lprev=topo.prev[l]
        lnext=topo.next[l]
        prev_idx=(lprev==0) ? 0 : gmap[lprev]
        next_idx=(lnext==0) ? 0 : gmap[lnext]
        prev_pts=(prev_idx==0) ? nothing : pts[prev_idx]
        next_pts=(next_idx==0) ? nothing : pts[next_idx]
        prev_ra=(prev_idx==0) ? (1:0) : (offs[prev_idx]:(offs[prev_idx+1]-1))
        next_ra=(next_idx==0) ? (1:0) : (offs[next_idx]:(offs[next_idx+1]-1))
        @use_threads multithreading=multithreaded for i in 1:Na
            gi=ra[i]
            xi=Xa[i]
            yi=Ya[i]
            si=Ga.speed[i]
            κi=Ga.kappa[i]
            ui=Ca.us[i]
            A[gi,gi]+=one(Complex{T})-Complex{T}(ha*si*κi,zero(T))
            for m in eachindex(gmap)
                bidx=gmap[m]
                pb=pts[bidx]
                rb=offs[bidx]:(offs[bidx+1]-1)
                Nb=length(pb.xy)
                blk=blocks[aidx,bidx]
                for j in 1:Nb
                    gj=rb[j]
                    if !(bidx==aidx && j==i)
                        p=blk.pidx[i,j]
                        t=blk.tloc[i,j]
                        _,h1,_,_=_bessels_one_k_at_r(plan0,plan1,planj0,planj1,p,t)
                        dx=xi-pb.xy[j][1]
                        dy=yi-pb.xy[j][2]
                        r2=muladd(dx,dx,dy*dy)
                        if r2>(eps(T))^2
                            r=sqrt(r2)
                            invr=inv(r)
                            inn=pb.tangent[j][2]*dx-pb.tangent[j][1]*dy
                            A[gi,gj]-=pb.ws[j]*(αD*inn*h1*invr)
                        end
                    end
                    skip_slp=false
                    if bidx==aidx
                        skip_slp=abs(j-i)<a
                    elseif right_smooth && bidx==next_idx
                        nr=_right_neighbor_excluded_count(i,Na,a)
                        skip_slp=(j<=nr)
                    elseif left_smooth && bidx==prev_idx
                        nl=_left_neighbor_excluded_count(i,a)
                        skip_slp=(j>Nb-nl)
                    end
                    if !skip_slp
                        p=blk.pidx[i,j]
                        t=blk.tloc[i,j]
                        h0,_=_hankels_one_k_at_r(plan0,plan1,p,t)
                        dx=xi-pb.xy[j][1]
                        dy=yi-pb.xy[j][2]
                        r2=muladd(dx,dx,dy*dy)
                        if r2>(eps(T))^2
                            A[gi,gj]-=ik*(pb.ws[j]*(αS*h0*sqrt(pb.tangent[j][1]^2+pb.tangent[j][2]^2)))
                        end
                    end
                end
            end
            for pidx_al in 1:jcorr
                fac=ha*rule.w[pidx_al]
                Δu=ha*rule.x[pidx_al]
                if ui+Δu<one(T)
                    dx=xi-Ca.xp[pidx_al,i]
                    dy=yi-Ca.yp[pidx_al,i]
                    r=hypot(dx,dy)
                    if isfinite(r) && r>sqrt(eps(T))
                        p=Int32(_find_panel(plan0,Float64(r)))
                        P=plan0.panels[p]
                        t=(2*Float64(r)-(P.b+P.a))/(P.b-P.a)
                        h0,_=_hankels_one_k_at_r(plan0,plan1,p,t)
                        coeff=-ik*(fac*(αS*h0*Ca.sp[pidx_al,i]))
                        for m4 in 1:4
                            q=Ca.idxp[pidx_al,i,m4]
                            A[gi,ra[q]]+=coeff*Ca.wtp[pidx_al,i,m4]
                        end
                    end
                end
                if ui-Δu>zero(T)
                    dx=xi-Ca.xm[pidx_al,i]
                    dy=yi-Ca.ym[pidx_al,i]
                    r=hypot(dx,dy)
                    if isfinite(r) && r>sqrt(eps(T))
                        p=Int32(_find_panel(plan0,Float64(r)))
                        P=plan0.panels[p]
                        t=(2*Float64(r)-(P.b+P.a))/(P.b-P.a)
                        h0,_=_hankels_one_k_at_r(plan0,plan1,p,t)
                        coeff=-ik*(fac*(αS*h0*Ca.sm[pidx_al,i]))
                        for m4 in 1:4
                            q=Ca.idxm[pidx_al,i,m4]
                            A[gi,ra[q]]+=coeff*Ca.wtm[pidx_al,i,m4]
                        end
                    end
                end
            end
            if right_smooth && next_idx!=0
                for pidx_al in 1:jcorr
                    Δu=ha*rule.x[pidx_al]
                    if ui+Δu>=one(T)
                        u2=ui+Δu-one(T)
                        x,y,tx,ty,s2,idx2,wt2=_eval_on_open_panel_local4(next_pts,u2)
                        dx=xi-x
                        dy=yi-y
                        r=hypot(dx,dy)
                        if isfinite(r) && r>sqrt(eps(T))
                            fac=ha*rule.w[pidx_al]
                            p=Int32(_find_panel(plan0,Float64(r)))
                            P=plan0.panels[p]
                            t=(2*Float64(r)-(P.b+P.a))/(P.b-P.a)
                            h0,_=_hankels_one_k_at_r(plan0,plan1,p,t)
                            coeff=-ik*(fac*(αS*h0*s2))
                            _scatter_local4!(A,gi,next_ra,coeff,idx2,wt2)
                        end
                    end
                end
            end
            if left_smooth && prev_idx!=0
                for pidx_al in 1:jcorr
                    Δu=ha*rule.x[pidx_al]
                    if ui-Δu<=zero(T)
                        u2=one(T)+ui-Δu
                        x,y,tx,ty,s2,idx2,wt2=_eval_on_open_panel_local4(prev_pts,u2)
                        dx=xi-x
                        dy=yi-y
                        r=hypot(dx,dy)
                        if isfinite(r) && r>sqrt(eps(T))
                            fac=ha*rule.w[pidx_al]
                            p=Int32(_find_panel(plan0,Float64(r)))
                            P=plan0.panels[p]
                            t=(2*Float64(r)-(P.b+P.a))/(P.b-P.a)
                            h0,_=_hankels_one_k_at_r(plan0,plan1,p,t)
                            coeff=-ik*(fac*(αS*h0*s2))
                            _scatter_local4!(A,gi,prev_ra,coeff,idx2,wt2)
                        end
                    end
                end
            end
        end
    end
    return A
end

function _assemble_all_self_alpert_composite_cheb!(solver::CFIE_alpert{T},A::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},Gs::Vector{CFIEGeomCache{T}},Cs,offs::Vector{Int},k::T,rule::AlpertLogRule{T},topos::Vector{AlpertCompositeTopology{T}},gmaps::Vector{Vector{Int}},blocks::Matrix{CFIEAlpertBlockCache{T}},plan0::ChebHankelPlanH,plan1::ChebHankelPlanH,planj0::ChebJPlan,planj1::ChebJPlan;multithreaded::Bool=true) where {T<:Real}
    @inbounds for c in eachindex(gmaps)
        gmap=gmaps[c]
        if length(gmap)==1 && pts[gmap[1]].is_periodic
            a=gmap[1]
            ra=offs[a]:(offs[a+1]-1)
            _assemble_self_alpert_cheb!(solver,A,pts[a],Gs[a],Cs[a],ra,k,rule,blocks[a,a],plan0,plan1,planj0,planj1;multithreaded=multithreaded)
        else
            _assemble_self_alpert_composite_component_cheb!(solver,A,pts,Gs,Cs,offs,k,rule,topos[c],gmap,blocks,plan0,plan1,planj0,planj1;multithreaded=multithreaded)
        end
    end
    return A
end

#############################################################
#### DIRECT NO-SYMMETRY CFIE_alpert ASSEMBLY: ALL k / ONE k #
#############################################################

function _all_k_nosymm_CFIE_alpert_chebyshev!(As::Vector{Matrix{ComplexF64}},solver::CFIE_alpert{T},pts::Vector{BoundaryPointsCFIE{T}},plans0::Vector{ChebHankelPlanH},plans1::Vector{ChebHankelPlanH},plansj0::Vector{ChebJPlan},plansj1::Vector{ChebJPlan},cheb_ws::CFIEAlpertChebWorkspace{T},bws::CFIEMultiBesselWorkspace;multithreaded::Bool=true) where {T<:Real}
    aws=cheb_ws.alpert_ws
    cws=cheb_ws.cheb_blocks
    offs=aws.offs
    Gs=aws.Gs
    Cs=aws.Cs
    rule=aws.rule
    topos=aws.topos
    gmaps=aws.gmaps
    panel_to_comp=aws.panel_to_comp
    blocks=cws.blocks
    Mk=length(plans0)
    @assert length(As)==Mk
    @inbounds for m in 1:Mk
        fill!(As[m],0)
        if topos===nothing
            for a in eachindex(pts)
                ra=offs[a]:(offs[a+1]-1)
                _assemble_self_alpert_cheb!(solver,As[m],pts[a],Gs[a],Cs[a],ra,T(real(plans0[m].k)),rule,blocks[a,a],plans0[m],plans1[m],plansj0[m],plansj1[m];multithreaded=multithreaded)
            end
        else
            _assemble_all_self_alpert_composite_cheb!(solver,As[m],pts,Gs,Cs,offs,T(real(plans0[m].k)),rule,topos,gmaps,blocks,plans0[m],plans1[m],plansj0[m],plansj1[m];multithreaded=multithreaded)
        end
    end
    for a in eachindex(pts), b in eachindex(pts)
        a==b && continue
        if panel_to_comp!==nothing
            ca=panel_to_comp[a]
            cb=panel_to_comp[b]
            ca!=0 && ca==cb && continue
        end
        blk=blocks[a,b]
        @use_threads multithreading=multithreaded for j in 1:blk.Nj
            tid=Threads.threadid()
            h0vals=bws.h0_tls[tid]
            h1vals=bws.h1_tls[tid]
            hankels_multi_ks_at_r!(h0vals,h1vals,plans0,plans1,blk.pidx[1,j],blk.tloc[1,j])
            for i in 1:blk.Ni
                p=blk.pidx[i,j]
                t=blk.tloc[i,j]
                hankels_multi_ks_at_r!(h0vals,h1vals,plans0,plans1,p,t)
                gi=blk.row_offset+i-1
                gj=blk.col_offset+j-1
                invr=blk.invR[i,j]
                inn=blk.inner[i,j]
                sj=blk.speed_j[j]
                wj=blk.wj[j]
                @inbounds for m in 1:Mk
                    αD=0.5im*ComplexF64(plans0[m].k)
                    αS=0.5im
                    ik=1im*ComplexF64(plans0[m].k)
                    dval=wj*(αD*inn*h1vals[m]*invr)
                    sval=wj*sj*(αS*h0vals[m])
                    As[m][gi,gj]-=(dval+ik*sval)
                end
            end
        end
    end
    return nothing
end

function _one_k_nosymm_CFIE_alpert_chebyshev!(A::Matrix{ComplexF64},solver::CFIE_alpert{T},pts::Vector{BoundaryPointsCFIE{T}},plan0::ChebHankelPlanH,plan1::ChebHankelPlanH,planj0::ChebJPlan,planj1::ChebJPlan,cheb_ws::CFIEAlpertChebWorkspace{T},bws::CFIEMultiBesselWorkspace;multithreaded::Bool=true) where {T<:Real}
    _all_k_nosymm_CFIE_alpert_chebyshev!([A],solver,pts,[plan0],[plan1],[planj0],[planj1],cheb_ws,bws;multithreaded=multithreaded)
    return nothing
end

##############################
#### DESYMMETRIZED KERNEL ####
##############################

@inline function _pair_panel_t_from_r(plan0::ChebHankelPlanH,r::Float64)
    p=Int32(_find_panel(plan0,r))
    P=plan0.panels[p]
    t=(2*r-(P.b+P.a))/(P.b-P.a)
    return p,t
end

function _add_image_block_cheb!(A::AbstractMatrix{Complex{T}},ra::UnitRange{Int},rb::UnitRange{Int},pa::BoundaryPointsCFIE{T},pb::BoundaryPointsCFIE{T},k::T,plan0::ChebHankelPlanH,plan1::ChebHankelPlanH,qfun,tfun,weight;multithreaded::Bool=true) where {T<:Real}
    αD=Complex{T}(0,k/2)
    αS=Complex{T}(0,one(T)/2)
    ik=Complex{T}(0,k)
    Na=length(pa.xy)
    Nb=length(pb.xy)
    Xa=getindex.(pa.xy,1)
    Ya=getindex.(pa.xy,2)
    @use_threads multithreading=multithreaded for j in 1:Nb
        gj=rb[j]
        qimg=qfun(pb.xy[j])
        timg=tfun(pb.tangent[j])
        xj=qimg[1]
        yj=qimg[2]
        txj=timg[1]
        tyj=timg[2]
        sj=sqrt(txj*txj+tyj*tyj)
        wd=dlp_weight(pb,j)
        ws=slp_weight(pb,j,sj)
        @inbounds for i in 1:Na
            gi=ra[i]
            dx=Xa[i]-xj
            dy=Ya[i]-yj
            r2=muladd(dx,dx,dy*dy)
            r2<=(eps(T))^2 && continue
            r=sqrt(r2)
            _check_r(r,"image-block-cheb",i,j)
            invr=inv(r)
            inn=tyj*dx-txj*dy
            p,t=_pair_panel_t_from_r(plan0,Float64(r))
            h0,h1=_hankels_one_k_at_r(plan0,plan1,p,t)
            dval=weight*wd*(αD*inn*h1*invr)
            sval=weight*ws*(αS*h0)
            A[gi,gj]-=(dval+ik*sval)
        end
    end
    return A
end

function _assemble_reflection_images_cheb!(A::AbstractMatrix{Complex{T}},ra::UnitRange{Int},rb::UnitRange{Int},pa::BoundaryPointsCFIE{T},pb::BoundaryPointsCFIE{T},solver::CFIE_alpert{T},billiard::Bi,k::T,plan0::ChebHankelPlanH,plan1::ChebHankelPlanH,sym::Reflection;multithreaded::Bool=true) where {T<:Real,Bi<:AbsBilliard}
    if sym.axis==:y_axis
        _add_image_block_cheb!(A,ra,rb,pa,pb,k,plan0,plan1,q->image_point_x(q,billiard),t->image_tangent_x(t),image_weight(sym);multithreaded=multithreaded)
    elseif sym.axis==:x_axis
        _add_image_block_cheb!(A,ra,rb,pa,pb,k,plan0,plan1,q->image_point_y(q,billiard),t->image_tangent_y(t),image_weight(sym);multithreaded=multithreaded)
    elseif sym.axis==:origin
        σx=image_weight_x(sym)
        σy=image_weight_y(sym)
        σxy=image_weight_xy(sym)
        _add_image_block_cheb!(A,ra,rb,pa,pb,k,plan0,plan1,q->image_point_x(q,billiard),t->image_tangent_x(t),σx;multithreaded=multithreaded)
        _add_image_block_cheb!(A,ra,rb,pa,pb,k,plan0,plan1,q->image_point_y(q,billiard),t->image_tangent_y(t),σy;multithreaded=multithreaded)
        _add_image_block_cheb!(A,ra,rb,pa,pb,k,plan0,plan1,q->image_point_xy(q,billiard),t->image_tangent_xy(t),σxy;multithreaded=multithreaded)
    else
        error("Unknown reflection axis $(sym.axis)")
    end
    return A
end

function _assemble_rotation_images_cheb!(A::AbstractMatrix{Complex{T}},ra::UnitRange{Int},rb::UnitRange{Int},pa::BoundaryPointsCFIE{T},pb::BoundaryPointsCFIE{T},k::T,plan0::ChebHankelPlanH,plan1::ChebHankelPlanH,sym::Rotation,costab,sintab,χ;multithreaded::Bool=true) where {T<:Real}
    for l in 1:(sym.n-1)
        phase=χ[l+1]
        _add_image_block_cheb!(A,ra,rb,pa,pb,k,plan0,plan1,q->image_point(sym,q,l,costab,sintab),t->image_tangent(sym,t,l,costab,sintab),phase;multithreaded=multithreaded)
    end
    return A
end

##########################################
#### SYMMETRY ASSEMBLY: DIRECT + CHEB ####
##########################################

function construct_matrices_symmetry_chebyshev!(solver::CFIE_alpert{T},A::Matrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},k::T,plan0::ChebHankelPlanH,plan1::ChebHankelPlanH,planj0::ChebJPlan,planj1::ChebJPlan,cheb_ws::CFIEAlpertChebWorkspace{T};multithreaded::Bool=true) where {T<:Real}
    symmetry=solver.symmetry
    isnothing(symmetry) && error("construct_matrices_symmetry_chebyshev! called with symmetry = nothing")
    fill!(A,zero(Complex{T}))
    aws=cheb_ws.alpert_ws
    cws=cheb_ws.cheb_blocks
    offs=aws.offs
    Gs=aws.Gs
    Cs=aws.Cs
    rule=aws.rule
    topos=aws.topos
    gmaps=aws.gmaps
    panel_to_comp=aws.panel_to_comp
    blocks=cws.blocks
    nc=length(pts)
    if topos===nothing
        @inbounds for a in 1:nc
            ra=offs[a]:(offs[a+1]-1)
            _assemble_self_alpert_cheb!(solver,A,pts[a],Gs[a],Cs[a],ra,k,rule,blocks[a,a],plan0,plan1,planj0,planj1;multithreaded=multithreaded)
        end
    else
        _assemble_all_self_alpert_composite_cheb!(solver,A,pts,Gs,Cs,offs,k,rule,topos,gmaps,blocks,plan0,plan1,planj0,planj1;multithreaded=multithreaded)
    end
    αD=Complex{T}(0,k/2)
    αS=Complex{T}(0,one(T)/2)
    ik=Complex{T}(0,k)
    for a in 1:nc, b in 1:nc
        a==b && continue
        if panel_to_comp!==nothing
            ca=panel_to_comp[a]
            cb=panel_to_comp[b]
            ca!=0 && ca==cb && continue
        end
        pa=pts[a]
        pb=pts[b]
        blk=blocks[a,b]
        Na=length(pa.xy)
        Nb=length(pb.xy)
        ra=offs[a]:(offs[a+1]-1)
        rb=offs[b]:(offs[b+1]-1)
        @use_threads multithreading=multithreaded for j in 1:Nb
            gj=rb[j]
            xj=pb.xy[j][1]
            yj=pb.xy[j][2]
            txj=pb.tangent[j][1]
            tyj=pb.tangent[j][2]
            sj=blk.speed_j[j]
            wd=dlp_weight(pb,j)
            ws=slp_weight(pb,j,sj)
            @inbounds for i in 1:Na
                gi=ra[i]
                dx=pa.xy[i][1]-xj
                dy=pa.xy[i][2]-yj
                r2=muladd(dx,dx,dy*dy)
                r2<=(eps(T))^2 && continue
                p=blk.pidx[i,j]
                t=blk.tloc[i,j]
                h0,h1=_hankels_one_k_at_r(plan0,plan1,p,t)
                invr=blk.invR[i,j]
                inn=tyj*dx-txj*dy
                dval=wd*(αD*inn*h1*invr)
                sval=ws*(αS*h0)
                A[gi,gj]-=(dval+ik*sval)
            end
        end
    end
    for sym in symmetry
        if sym isa Reflection
            for a in 1:nc, b in 1:nc
                ra=offs[a]:(offs[a+1]-1)
                rb=offs[b]:(offs[b+1]-1)
                _assemble_reflection_images_cheb!(A,ra,rb,pts[a],pts[b],solver,solver.billiard,k,plan0,plan1,sym;multithreaded=multithreaded)
            end
        elseif sym isa Rotation
            costab,sintab,χ=_rotation_tables(T,sym.n,sym.m)
            for a in 1:nc, b in 1:nc
                ra=offs[a]:(offs[a+1]-1)
                rb=offs[b]:(offs[b+1]-1)
                _assemble_rotation_images_cheb!(A,ra,rb,pts[a],pts[b],k,plan0,plan1,sym,costab,sintab,χ;multithreaded=multithreaded)
            end
        else
            error("Unknown symmetry type $(typeof(sym))")
        end
    end
    return A
end

function construct_matrices_symmetry_chebyshev!(solver::CFIE_alpert{T},As::Vector{Matrix{Complex{T}}},pts::Vector{BoundaryPointsCFIE{T}},plans0::Vector{ChebHankelPlanH},plans1::Vector{ChebHankelPlanH},plansj0::Vector{ChebJPlan},plansj1::Vector{ChebJPlan},cheb_ws::CFIEAlpertChebWorkspace{T};multithreaded::Bool=true) where {T<:Real}
    @inbounds for m in eachindex(As)
        construct_matrices_symmetry_chebyshev!(solver,As[m],pts,T(real(plans0[m].k)),plans0[m],plans1[m],plansj0[m],plansj1[m],cheb_ws;multithreaded=multithreaded)
    end
    return nothing
end

##############################
#### DESYMMETRIZED KERNEL ####
##############################

@inline function _pair_panel_t_from_r(plan0::ChebHankelPlanH,r::Float64)
    p=Int32(_find_panel(plan0,r))
    P=plan0.panels[p]
    t=(2*r-(P.b+P.a))/(P.b-P.a)
    return p,t
end

function _add_image_block_cheb!(A::AbstractMatrix{Complex{T}},ra::UnitRange{Int},rb::UnitRange{Int},pa::BoundaryPointsCFIE{T},pb::BoundaryPointsCFIE{T},k::T,plan0::ChebHankelPlanH,plan1::ChebHankelPlanH,qfun,tfun,weight;multithreaded::Bool=true) where {T<:Real}
    αD=Complex{T}(0,k/2)
    αS=Complex{T}(0,one(T)/2)
    ik=Complex{T}(0,k)
    Na=length(pa.xy)
    Nb=length(pb.xy)
    Xa=getindex.(pa.xy,1)
    Ya=getindex.(pa.xy,2)
    @use_threads multithreading=multithreaded for j in 1:Nb
        gj=rb[j]
        qimg=qfun(pb.xy[j])
        timg=tfun(pb.tangent[j])
        xj=qimg[1]
        yj=qimg[2]
        txj=timg[1]
        tyj=timg[2]
        sj=sqrt(txj*txj+tyj*tyj)
        wd=dlp_weight(pb,j)
        ws=slp_weight(pb,j,sj)
        @inbounds for i in 1:Na
            gi=ra[i]
            dx=Xa[i]-xj
            dy=Ya[i]-yj
            r2=muladd(dx,dx,dy*dy)
            r2<=(eps(T))^2 && continue
            r=sqrt(r2)
            _check_r(r,"image-block-cheb",i,j)
            invr=inv(r)
            inn=tyj*dx-txj*dy
            p,t=_pair_panel_t_from_r(plan0,Float64(r))
            h0,h1=_hankels_one_k_at_r(plan0,plan1,p,t)
            dval=weight*wd*(αD*inn*h1*invr)
            sval=weight*ws*(αS*h0)
            A[gi,gj]-=(dval+ik*sval)
        end
    end
    return A
end

function _assemble_reflection_images_cheb!(A::AbstractMatrix{Complex{T}},ra::UnitRange{Int},rb::UnitRange{Int},pa::BoundaryPointsCFIE{T},pb::BoundaryPointsCFIE{T},solver::CFIE_alpert{T},billiard::Bi,k::T,plan0::ChebHankelPlanH,plan1::ChebHankelPlanH,sym::Reflection;multithreaded::Bool=true) where {T<:Real,Bi<:AbsBilliard}
    if sym.axis==:y_axis
        _add_image_block_cheb!(A,ra,rb,pa,pb,k,plan0,plan1,q->image_point_x(q,billiard),t->image_tangent_x(t),image_weight(sym);multithreaded=multithreaded)
    elseif sym.axis==:x_axis
        _add_image_block_cheb!(A,ra,rb,pa,pb,k,plan0,plan1,q->image_point_y(q,billiard),t->image_tangent_y(t),image_weight(sym);multithreaded=multithreaded)
    elseif sym.axis==:origin
        σx=image_weight_x(sym)
        σy=image_weight_y(sym)
        σxy=image_weight_xy(sym)
        _add_image_block_cheb!(A,ra,rb,pa,pb,k,plan0,plan1,q->image_point_x(q,billiard),t->image_tangent_x(t),σx;multithreaded=multithreaded)
        _add_image_block_cheb!(A,ra,rb,pa,pb,k,plan0,plan1,q->image_point_y(q,billiard),t->image_tangent_y(t),σy;multithreaded=multithreaded)
        _add_image_block_cheb!(A,ra,rb,pa,pb,k,plan0,plan1,q->image_point_xy(q,billiard),t->image_tangent_xy(t),σxy;multithreaded=multithreaded)
    else
        error("Unknown reflection axis $(sym.axis)")
    end
    return A
end

function _assemble_rotation_images_cheb!(A::AbstractMatrix{Complex{T}},ra::UnitRange{Int},rb::UnitRange{Int},pa::BoundaryPointsCFIE{T},pb::BoundaryPointsCFIE{T},k::T,plan0::ChebHankelPlanH,plan1::ChebHankelPlanH,sym::Rotation,costab,sintab,χ;multithreaded::Bool=true) where {T<:Real}
    for l in 1:(sym.n-1)
        phase=χ[l+1]
        _add_image_block_cheb!(A,ra,rb,pa,pb,k,plan0,plan1,q->image_point(sym,q,l,costab,sintab),t->image_tangent(sym,t,l,costab,sintab),phase;multithreaded=multithreaded)
    end
    return A
end

##########################################
#### SYMMETRY ASSEMBLY: DIRECT + CHEB ####
##########################################

function construct_matrices_symmetry_chebyshev!(solver::CFIE_alpert{T},A::Matrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},k::T,plan0::ChebHankelPlanH,plan1::ChebHankelPlanH,planj0::ChebJPlan,planj1::ChebJPlan,cheb_ws::CFIEAlpertChebWorkspace{T};multithreaded::Bool=true) where {T<:Real}
    symmetry=solver.symmetry
    isnothing(symmetry) && error("construct_matrices_symmetry_chebyshev! called with symmetry = nothing")
    fill!(A,zero(Complex{T}))
    aws=cheb_ws.alpert_ws
    cws=cheb_ws.cheb_blocks
    offs=aws.offs
    Gs=aws.Gs
    Cs=aws.Cs
    rule=aws.rule
    topos=aws.topos
    gmaps=aws.gmaps
    panel_to_comp=aws.panel_to_comp
    blocks=cws.blocks
    nc=length(pts)
    if topos===nothing
        @inbounds for a in 1:nc
            ra=offs[a]:(offs[a+1]-1)
            _assemble_self_alpert_cheb!(solver,A,pts[a],Gs[a],Cs[a],ra,k,rule,blocks[a,a],plan0,plan1,planj0,planj1;multithreaded=multithreaded)
        end
    else
        _assemble_all_self_alpert_composite_cheb!(solver,A,pts,Gs,Cs,offs,k,rule,topos,gmaps,blocks,plan0,plan1,planj0,planj1;multithreaded=multithreaded)
    end
    αD=Complex{T}(0,k/2)
    αS=Complex{T}(0,one(T)/2)
    ik=Complex{T}(0,k)
    for a in 1:nc, b in 1:nc
        a==b && continue
        if panel_to_comp!==nothing
            ca=panel_to_comp[a]
            cb=panel_to_comp[b]
            ca!=0 && ca==cb && continue
        end
        pa=pts[a]
        pb=pts[b]
        blk=blocks[a,b]
        Na=length(pa.xy)
        Nb=length(pb.xy)
        ra=offs[a]:(offs[a+1]-1)
        rb=offs[b]:(offs[b+1]-1)
        @use_threads multithreading=multithreaded for j in 1:Nb
            gj=rb[j]
            xj=pb.xy[j][1]
            yj=pb.xy[j][2]
            txj=pb.tangent[j][1]
            tyj=pb.tangent[j][2]
            sj=blk.speed_j[j]
            wd=dlp_weight(pb,j)
            ws=slp_weight(pb,j,sj)
            @inbounds for i in 1:Na
                gi=ra[i]
                dx=pa.xy[i][1]-xj
                dy=pa.xy[i][2]-yj
                r2=muladd(dx,dx,dy*dy)
                r2<=(eps(T))^2 && continue
                p=blk.pidx[i,j]
                t=blk.tloc[i,j]
                h0,h1=_hankels_one_k_at_r(plan0,plan1,p,t)
                invr=blk.invR[i,j]
                inn=tyj*dx-txj*dy
                dval=wd*(αD*inn*h1*invr)
                sval=ws*(αS*h0)
                A[gi,gj]-=(dval+ik*sval)
            end
        end
    end
    for sym in symmetry
        if sym isa Reflection
            for a in 1:nc, b in 1:nc
                ra=offs[a]:(offs[a+1]-1)
                rb=offs[b]:(offs[b+1]-1)
                _assemble_reflection_images_cheb!(A,ra,rb,pts[a],pts[b],solver,solver.billiard,k,plan0,plan1,sym;multithreaded=multithreaded)
            end
        elseif sym isa Rotation
            costab,sintab,χ=_rotation_tables(T,sym.n,sym.m)
            for a in 1:nc, b in 1:nc
                ra=offs[a]:(offs[a+1]-1)
                rb=offs[b]:(offs[b+1]-1)
                _assemble_rotation_images_cheb!(A,ra,rb,pts[a],pts[b],k,plan0,plan1,sym,costab,sintab,χ;multithreaded=multithreaded)
            end
        else
            error("Unknown symmetry type $(typeof(sym))")
        end
    end
    return A
end

function construct_matrices_symmetry_chebyshev!(solver::CFIE_alpert{T},As::Vector{Matrix{Complex{T}}},pts::Vector{BoundaryPointsCFIE{T}},plans0::Vector{ChebHankelPlanH},plans1::Vector{ChebHankelPlanH},plansj0::Vector{ChebJPlan},plansj1::Vector{ChebJPlan},cheb_ws::CFIEAlpertChebWorkspace{T};multithreaded::Bool=true) where {T<:Real}
    @inbounds for m in eachindex(As)
        construct_matrices_symmetry_chebyshev!(solver,As[m],pts,T(real(plans0[m].k)),plans0[m],plans1[m],plansj0[m],plansj1[m],cheb_ws;multithreaded=multithreaded)
    end
    return nothing
end

#################################
#### HIGH LEVEL ENTRY POINTS ####
#################################

function compute_kernel_matrices_CFIE_alpert_chebyshev!(As::Vector{Matrix{ComplexF64}},solver::CFIE_alpert{T},pts::Vector{BoundaryPointsCFIE{T}},plans0::Vector{ChebHankelPlanH},plans1::Vector{ChebHankelPlanH},plansj0::Vector{ChebJPlan},plansj1::Vector{ChebJPlan},cheb_ws::CFIEAlpertChebWorkspace{T},bws::CFIEMultiBesselWorkspace;multithreaded::Bool=true) where {T<:Real}
    if isnothing(solver.symmetry)
        _all_k_nosymm_CFIE_alpert_chebyshev!(As,solver,pts,plans0,plans1,plansj0,plansj1,cheb_ws,bws;multithreaded=multithreaded)
    else
        construct_matrices_symmetry_chebyshev!(solver,As,pts,plans0,plans1,plansj0,plansj1,cheb_ws;multithreaded=multithreaded)
    end
    return nothing
end

function compute_kernel_matrices_CFIE_alpert_chebyshev!(A::Matrix{ComplexF64},solver::CFIE_alpert{T},pts::Vector{BoundaryPointsCFIE{T}},plan0::ChebHankelPlanH,plan1::ChebHankelPlanH,planj0::ChebJPlan,planj1::ChebJPlan,cheb_ws::CFIEAlpertChebWorkspace{T},bws::CFIEMultiBesselWorkspace;multithreaded::Bool=true) where {T<:Real}
    if isnothing(solver.symmetry)
        _one_k_nosymm_CFIE_alpert_chebyshev!(A,solver,pts,plan0,plan1,planj0,planj1,cheb_ws,bws;multithreaded=multithreaded)
    else
        construct_matrices_symmetry_chebyshev!(solver,A,pts,T(real(plan0.k)),plan0,plan1,planj0,planj1,cheb_ws;multithreaded=multithreaded)
    end
    return nothing
end

########################
#### HIGH LEVEL API ####
########################

function construct_matrices_chebyshev!(solver::CFIE_alpert{T},A::Matrix{ComplexF64},pts::Vector{BoundaryPointsCFIE{T}},k::T,cheb_ws::CFIEAlpertChebWorkspace{T};npanels::Int=10000,M::Int=5,grading::Symbol=:uniform,geo_ratio::Real=1.05,multithreaded::Bool=true) where {T<:Real}
    plans0,plans1,plansj0,plansj1=build_CFIE_alpert_plans([k],cheb_ws.cheb_blocks.rmin,cheb_ws.cheb_blocks.rmax;npanels=npanels,M=M,grading=grading,geo_ratio=geo_ratio)
    bws=CFIEMultiBesselWorkspace(1)
    compute_kernel_matrices_CFIE_alpert_chebyshev!(A,solver,pts,plans0[1],plans1[1],plansj0[1],plansj1[1],cheb_ws,bws;multithreaded=multithreaded)
    return A
end

function construct_matrices_chebyshev(solver::CFIE_alpert{T},pts::Vector{BoundaryPointsCFIE{T}},k::T,cheb_ws::CFIEAlpertChebWorkspace{T};npanels::Int=10000,M::Int=5,grading::Symbol=:uniform,geo_ratio::Real=1.05,multithreaded::Bool=true) where {T<:Real}
    A=Matrix{ComplexF64}(undef,cheb_ws.alpert_ws.Ntot,cheb_ws.alpert_ws.Ntot)
    construct_matrices_chebyshev!(solver,A,pts,k,cheb_ws;npanels=npanels,M=M,grading=grading,geo_ratio=geo_ratio,multithreaded=multithreaded)
    return A
end

function construct_matrices_chebyshev!(solver::CFIE_alpert{T},As::Vector{Matrix{ComplexF64}},pts::Vector{BoundaryPointsCFIE{T}},ks::AbstractVector{<:Number},cheb_ws::CFIEAlpertChebWorkspace{T};npanels::Int=10000,M::Int=5,grading::Symbol=:uniform,geo_ratio::Real=1.05,nthreads::Int=1,multithreaded::Bool=true) where {T<:Real}
    plans0,plans1,plansj0,plansj1=build_CFIE_alpert_plans(ks,cheb_ws.cheb_blocks.rmin,cheb_ws.cheb_blocks.rmax;npanels=npanels,M=M,grading=grading,geo_ratio=geo_ratio,nthreads=nthreads)
    bws=CFIEMultiBesselWorkspace(length(ks))
    compute_kernel_matrices_CFIE_alpert_chebyshev!(As,solver,pts,plans0,plans1,plansj0,plansj1,cheb_ws,bws;multithreaded=multithreaded)
    return As
end

function construct_matrices_chebyshev(solver::CFIE_alpert{T},pts::Vector{BoundaryPointsCFIE{T}},ks::AbstractVector{<:Number},cheb_ws::CFIEAlpertChebWorkspace{T};npanels::Int=10000,M::Int=5,grading::Symbol=:uniform,geo_ratio::Real=1.05,nthreads::Int=1,multithreaded::Bool=true) where {T<:Real}
    As=[Matrix{ComplexF64}(undef,cheb_ws.alpert_ws.Ntot,cheb_ws.alpert_ws.Ntot) for _ in eachindex(ks)]
    construct_matrices_chebyshev!(solver,As,pts,ks,cheb_ws;npanels=npanels,M=M,grading=grading,geo_ratio=geo_ratio,nthreads=nthreads,multithreaded=multithreaded)
    return As
end

############################
#### SOLVE WRAPPERS ########
############################

function solve_chebyshev(solver::CFIE_alpert,basis::Ba,pts::Vector{BoundaryPointsCFIE{T}},k,cheb_ws::CFIEAlpertChebWorkspace{T};npanels::Int=10000,M::Int=5,grading::Symbol=:uniform,geo_ratio::Real=1.05,multithreaded::Bool=true,use_krylov::Bool=true) where {T<:Real,Ba<:AbsBasis}
    A=construct_matrices_chebyshev(solver,pts,T(k),cheb_ws;npanels=npanels,M=M,grading=grading,geo_ratio=geo_ratio,multithreaded=multithreaded)
    if use_krylov
        @blas_multi_then_1 MAX_BLAS_THREADS mu,_,_,_=svdsolve(A,1,:SR)
        return mu[1]
    else
        s=svdvals(A)
        return s[end]
    end
end

function solve_vect_chebyshev(solver::CFIE_alpert,basis::Ba,pts::Vector{BoundaryPointsCFIE{T}},k,cheb_ws::CFIEAlpertChebWorkspace{T};npanels::Int=10000,M::Int=5,grading::Symbol=:uniform,geo_ratio::Real=1.05,multithreaded::Bool=true) where {T<:Real,Ba<:AbsBasis}
    A=construct_matrices_chebyshev(solver,pts,T(k),cheb_ws;npanels=npanels,M=M,grading=grading,geo_ratio=geo_ratio,multithreaded=multithreaded)
    _,S,Vt=LAPACK.gesvd!('A','A',A)
    idx=findmin(S)[2]
    return S[idx],conj.(Vt[idx,:])
end

function solve_INFO_chebyshev(solver::CFIE_alpert,basis::Ba,pts::Vector{BoundaryPointsCFIE{T}},k,cheb_ws::CFIEAlpertChebWorkspace{T};npanels::Int=10000,M::Int=5,grading::Symbol=:uniform,geo_ratio::Real=1.05,multithreaded::Bool=true,use_krylov::Bool=true) where {T<:Real,Ba<:AbsBasis}
    A=Matrix{ComplexF64}(undef,cheb_ws.alpert_ws.Ntot,cheb_ws.alpert_ws.Ntot)
    t0=time()
    @info "Building boundary operator A with Chebyshev kernels..."
    construct_matrices_chebyshev!(solver,A,pts,T(k),cheb_ws;npanels=npanels,M=M,grading=grading,geo_ratio=geo_ratio,multithreaded=multithreaded)
    any(isnan.(A)) && error("NaN detected in system matrix A; check geometry and quadrature.")
    t1=time()
    cA=cond(A)
    @info "Condition number of A: $(round(cA;sigdigits=4))"
    @info "Performing SVD..."
    t2=time()
    if use_krylov
        @blas_multi_then_1 MAX_BLAS_THREADS s,_,_,_=svdsolve(A,1,:SR)
        reverse!(s)
    else
        s=svdvals(A)
    end
    t3=time()
    build_A=t1-t0
    svd_time=t3-t2
    total=build_A+svd_time
    println("────────── SOLVE_INFO_CHEB SUMMARY ──────────")
    println("A-matrix build: ",100*build_A/total," %")
    println("SVD: ",100*svd_time/total," %")
    println("(total: ",total," s)")
    println("─────────────────────────────────────────────")
    return s[end]
end

##############################
#### OPTIONAL LEGACY ALIAS ####
##############################

const build_CFIE_plans_alpert=build_CFIE_alpert_plans