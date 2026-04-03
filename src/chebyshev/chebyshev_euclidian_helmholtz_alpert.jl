#################################################################
#   CHEBYSHEV-BASED SLP/DLP EVALUATION FOR CFIE_alpert ASSEMBLY IN 2D EUCLIDEAN HELMHOLTZ 
# Functions to build Chebyshev-based SLP/DLP evaluation plans for multiple wavenumbers, and to compute the CFIE_alpert matrix blocks using these plans. 
# Logic:
# - Build Chebyshev-based Hankel evaluation plans for the SLP and DLP kernels for each wavenumber.
# - Precompute the geometry-related terms (R, invR, inner product, speed, quadrature weights) for each block of the CFIE_alpert matrix.
# - For each block, use the precomputed geometry and the Chebyshev plans to evaluate the SLP and DLP contributions for each pair of points, and accumulate the CFIE_alpert matrix entries.
#
# API: 
# - `build_CFIE_alpert_plans(...)`: Builds Chebyshev-based Hankel evaluation plans for the given wavenumbers and geometry.
# - `build_cfie_alpert_block_caches(...)`: Precomputes the geometry-related terms for each block of the CFIE_alpert matrix.
# - `h01_multi_ks_at_r!(...)`: Evaluates the SLP and DLP Hankel functions for multiple wavenumbers at given distances.
# - `compute_kernel_matrices_CFIE_chebyshev!(...)`: Main function to compute the CFIE_alpert matrix blocks for all wavenumbers, using the appropriate method based on the presence of symmetries and the number of wavenumbers.
# USUALLY NOT CALLED DIRECTLY: 
# - `_one_k_nosymm_CFIE_chebyshev!(...)`: Computes the CFIE_alpert matrix blocks for a single wavenumber without using symmetries, using Chebyshev-based SLP/DLP evaluation.
# - `_all_k_nosymm_CFIE_chebyshev!(...)`: Computes the CFIE_alpert matrix blocks for all wavenumbers without using symmetries, using Chebyshev-based SLP/DLP evaluation.
#
# Workflow: 
# 1. Call `build_cfie_alpert_workspace` to create the Alpert workspace, which includes building the CFIE_alpert block caches with `build_cfie_alpert_block_caches`.
# 2. Call `build_cfie_alpert_cheb_workspace` to create the Chebyshev workspace, which includes building the Chebyshev plans for the SLP and DLP kernels for all wavenumbers with `build_CFIE_alpert_plans`.
# 3. Call `compute_kernel_matrices_CFIE_alpert_chebyshev!` to compute the CFIE_alpert matrix blocks for all wavenumbers, which will internally call the appropriate function based on the presence of symmetries and the number of wavenumbers.
#
# MO 2/4/26
#################################################################

_TWO_PI=2*pi
_INV_TWO_PI=1/_TWO_PI

############################
#### MULTI-K H0 / H1 AT r ##
############################

@inline function h0_h1_multi_ks_at_r!(h0vals::AbstractVector{ComplexF64},h1vals::AbstractVector{ComplexF64},plans0::AbstractVector{ChebHankelPlanH},plans1::AbstractVector{ChebHankelPlanH},pidx::Int32,t::Float64)
    @inbounds for m in eachindex(plans0)
        h0vals[m]=_cheb_clenshaw(plans0[m].panels[pidx].c,t)
        h1vals[m]=_cheb_clenshaw(plans1[m].panels[pidx].c,t)
    end
    return nothing
end

#######################################
#### PLAN BUILDERS FOR CFIE_alpert ####
#######################################

function build_CFIE_alpert_plans(ks::AbstractVector{ComplexF64},rmin::Float64,rmax::Float64;npanels::Int=10000,M::Int=5,grading::Symbol=:uniform,geo_ratio::Real=1.05,nthreads::Int=1)
    Mk=length(ks)
    plans0=Vector{ChebHankelPlanH}(undef,Mk)
    plans1=Vector{ChebHankelPlanH}(undef,Mk)
    if nthreads<=1 || Mk==1
        @inbounds for m in 1:Mk
            k=ks[m]
            plans0[m]=plan_h(0,1,k,rmin,rmax;npanels=npanels,M=M,grading=grading,geo_ratio=geo_ratio)
            plans1[m]=plan_h(1,1,k,rmin,rmax;npanels=npanels,M=M,grading=grading,geo_ratio=geo_ratio)
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
                k=ks[m]
                plans0[m]=plan_h(0,1,k,rmin,rmax;npanels=npanels,M=M,grading=grading,geo_ratio=geo_ratio)
                plans1[m]=plan_h(1,1,k,rmin,rmax;npanels=npanels,M=M,grading=grading,geo_ratio=geo_ratio)
            end
        end
    end
    return plans0,plans1
end

#################################
#### COMPONENT / BLOCK CACHE ####
#################################

struct CFIE_alpert_BlockCache{T<:Real}
    same::Bool
    row_offset::Int
    col_offset::Int
    Ni::Int
    Nj::Int
    R::Matrix{T}
    invR::Matrix{T}
    inner::Matrix{T}
    speed_j::Vector{T}
    wj::Vector{T}
    pidx::Matrix{Int32}
    tloc::Matrix{Float64}
end

struct CFIEAlpertBlockSystemCache{T<:Real}
    blocks::Matrix{CFIE_alpert_BlockCache{T}}
    offsets::Vector{Int}
    rmin::Float64
    rmax::Float64
end

function build_cfie_alpert_cheb_block_caches(comps::Vector{BoundaryPointsCFIE{T}};npanels::Int=10000,M::Int=5,grading::Symbol=:uniform,geo_ratio::Real=1.05,pad=(T(0.95),T(1.05))) where {T<:Real}
    nc=length(comps)
    offs=component_offsets(comps)
    blocks=Matrix{CFIE_alpert_BlockCache{T}}(undef,nc,nc)
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
        blocks[a,b]=CFIE_alpert_BlockCache{T}(same,offs[a],offs[b],Ni,Nj,R,invR,inner,speed_j,wj,pidx,tloc)
    end
    pref_plan=plan_h(0,1,1.0+0im,Float64(global_rmin),Float64(global_rmax);npanels=npanels,M=M,grading=grading,geo_ratio=geo_ratio)
    pans=pref_plan.panels
    for a in 1:nc, b in 1:nc
        blk=blocks[a,b]
        same=blk.same
        @inbounds for j in 1:blk.Nj, i in 1:blk.Ni
            if same && i==j
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

struct CFIE_H0_H1_BesselWorkspace
    h0_tls::Vector{Vector{ComplexF64}}
    h1_tls::Vector{Vector{ComplexF64}}
    coeff_tls::Vector{Vector{ComplexF64}}
end

function CFIE_H0_H1_BesselWorkspace(Mk::Int;ntls::Int=(Threads.nthreads()))
    h0_tls=[Vector{ComplexF64}(undef,Mk) for _ in 1:ntls]
    h1_tls=[Vector{ComplexF64}(undef,Mk) for _ in 1:ntls]
    coeff_tls=[Vector{ComplexF64}(undef,Mk) for _ in 1:ntls]
    return CFIE_H0_H1_BesselWorkspace(h0_tls,h1_tls,coeff_tls)
end

#########################################
#### FULL CHEB WORKSPACE FOR ALPERT #####
#########################################

struct CFIEAlpertChebWorkspace{T<:Real,C}
    direct::CFIEAlpertWorkspace{T,C}
    block_cache::CFIEAlpertBlockSystemCache{T}
    plans0::Vector{ChebHankelPlanH}
    plans1::Vector{ChebHankelPlanH}
    bessel_ws::CFIE_H0_H1_BesselWorkspace
    ks::Vector{ComplexF64}
    Mk::Int
end

############################################
#### CHEB LOOKUP HELPERS FOR ONE ENTRY #####
############################################

@inline function _h0_h1_at_entry!(h0vals::AbstractVector{ComplexF64},h1vals::AbstractVector{ComplexF64},blk::CFIE_alpert_BlockCache,i::Int,j::Int,plans0::AbstractVector{ChebHankelPlanH},plans1::AbstractVector{ChebHankelPlanH})
    pidx=blk.pidx[i,j]
    t=blk.tloc[i,j]
    h0_h1_multi_ks_at_r!(h0vals,h1vals,plans0,plans1,pidx,t)
    return nothing
end

@inline function _h0_h1_at_r!(h0vals::AbstractVector{ComplexF64},h1vals::AbstractVector{ComplexF64},r::Float64,plans0::AbstractVector{ChebHankelPlanH},plans1::AbstractVector{ChebHankelPlanH})
    pidx=_find_panel(plans0[1],r)
    P=plans0[1].panels[pidx]
    t=(2r-(P.b+P.a))/(P.b-P.a)
    h0_h1_multi_ks_at_r!(h0vals,h1vals,plans0,plans1,Int32(pidx),t)
    return nothing
end

########################################################
#### SELF-ALPERT ASSEMBLY, CHEBYSHEV, MULTI-k ONLY #####
########################################################

function _assemble_self_alpert_periodic_cheb!(As::Vector{<:AbstractMatrix{ComplexF64}},pts::BoundaryPointsCFIE{T},G::CFIEGeomCache{T},C::AlpertPeriodicCache{T},row_range::UnitRange{Int},ks::Vector{ComplexF64},rule::AlpertLogRule{T},plans0::Vector{ChebHankelPlanH},plans1::Vector{ChebHankelPlanH},h0_tls::Vector{Vector{ComplexF64}},h1_tls::Vector{Vector{ComplexF64}},coeff_tls::Vector{Vector{ComplexF64}};multithreaded::Bool=true) where {T<:Real}
    Mk=length(ks)
    αD=Vector{ComplexF64}(undef,Mk)
    khalf=Vector{ComplexF64}(undef,Mk)
    @inbounds for m in 1:Mk
        αD[m]=0.5im*ks[m]
        khalf[m]=0.5*ks[m]
    end
    X=getindex.(pts.xy,1)
    Y=getindex.(pts.xy,2)
    N=length(pts.ts)
    a=rule.a
    jcorr=rule.j
    h=pts.ws[1]
    @use_threads multithreading=multithreaded for i in 1:N
        tid=Threads.threadid()
        h0vals=h0_tls[tid]
        h1vals=h1_tls[tid]
        gi=row_range[i]
        xi=X[i]
        yi=Y[i]
        si=G.speed[i]
        κi=G.kappa[i]
        diagv=1.0-(h*si*κi)
        @inbounds for m in 1:Mk
            As[m][gi,gi]+=diagv
        end
        @inbounds for j in 1:N
            j==i && continue
            gj=row_range[j]
            rij=Float64(G.R[i,j])
            inn=G.inner[i,j]
            invr=G.invR[i,j]
            _h0_h1_at_r!(h0vals,h1vals,rij,plans0,plans1)
            for m in 1:Mk
                As[m][gi,gj]-=h*(αD[m]*inn*h1vals[m]*invr)
            end
        end
        @inbounds for j in 1:N
            j==i && continue
            s=j-i
            s>N÷2 && (s-=N)
            s<-N÷2 && (s+=N)
            abs(s)<a && continue
            gj=row_range[j]
            rij=Float64(G.R[i,j])
            _h0_h1_at_r!(h0vals,h1vals,rij,plans0,plans1)
            sj=G.speed[j]
            for m in 1:Mk
                As[m][gi,gj]+=khalf[m]*(h*sj*h0vals[m])
            end
        end
        @inbounds for p in 1:jcorr
            fac=h*rule.w[p]
            dx=xi-C.xp[p,i]
            dy=yi-C.yp[p,i]
            r=sqrt(dx*dx+dy*dy)
            _h0_h1_at_r!(h0vals,h1vals,Float64(r),plans0,plans1)
            for m in 1:Mk
                coeff=khalf[m]*(fac*C.sp[p,i]*h0vals[m])
                for qid in 1:4
                    q=C.idxp[p,i,qid]
                    As[m][gi,row_range[q]]+=coeff*C.wtp[p,i,qid]
                end
            end
            dx=xi-C.xm[p,i]
            dy=yi-C.ym[p,i]
            r=sqrt(dx*dx+dy*dy)
            _h0_h1_at_r!(h0vals,h1vals,Float64(r),plans0,plans1)
            for m in 1:Mk
                coeff=khalf[m]*(fac*C.sm[p,i]*h0vals[m])
                for qid in 1:4
                    q=C.idxm[p,i,qid]
                    As[m][gi,row_range[q]]+=coeff*C.wtm[p,i,qid]
                end
            end
        end
    end
    return nothing
end

function _assemble_self_alpert_smooth_panel_cheb!(solver::CFIE_alpert{T},As::Vector{<:AbstractMatrix{ComplexF64}},pts::BoundaryPointsCFIE{T},G::CFIEGeomCache{T},C::AlpertSmoothPanelCache{T},row_range::UnitRange{Int},ks::Vector{ComplexF64},rule::AlpertLogRule{T},plans0::Vector{ChebHankelPlanH},plans1::Vector{ChebHankelPlanH},h0_tls::Vector{Vector{ComplexF64}},h1_tls::Vector{Vector{ComplexF64}},coeff_tls::Vector{Vector{ComplexF64}};multithreaded::Bool=true) where {T<:Real}
    Mk=length(ks)
    αD=Vector{ComplexF64}(undef,Mk)
    khalf=Vector{ComplexF64}(undef,Mk)
    @inbounds for m in 1:Mk
        αD[m]=0.5im*ks[m]
        khalf[m]=0.5*ks[m]
    end
    X=getindex.(pts.xy,1)
    Y=getindex.(pts.xy,2)
    N=length(X)
    h=pts.ws[1]
    a=rule.a
    jcorr=rule.j
    @use_threads multithreading=multithreaded for i in 1:N
        tid=Threads.threadid()
        h0vals=h0_tls[tid]
        h1vals=h1_tls[tid]
        gi=row_range[i]
        xi=X[i]
        yi=Y[i]
        si=G.speed[i]
        κi=G.kappa[i]
        ui=C.us[i]
        diagv=1.0-(h*si*κi)
        @inbounds for m in 1:Mk
            As[m][gi,gi]+=diagv
        end
        @inbounds for j in 1:N
            j==i && continue
            gj=row_range[j]
            rij=Float64(G.R[i,j])
            inn=G.inner[i,j]
            invr=G.invR[i,j]
            _h0_h1_at_r!(h0vals,h1vals,rij,plans0,plans1)
            for m in 1:Mk
                As[m][gi,gj]-=h*(αD[m]*inn*h1vals[m]*invr)
            end
        end
        @inbounds for j in 1:N
            j==i && continue
            abs(j-i)<a && continue
            gj=row_range[j]
            rij=Float64(G.R[i,j])
            _h0_h1_at_r!(h0vals,h1vals,rij,plans0,plans1)
            sj=G.speed[j]
            for m in 1:Mk
                As[m][gi,gj]+=khalf[m]*(h*sj*h0vals[m])
            end
        end
        @inbounds for p in 1:jcorr
            fac=h*rule.w[p]
            Δu=h*rule.x[p]
            if ui+Δu<one(T)
                dx=xi-C.xp[p,i]
                dy=yi-C.yp[p,i]
                r=sqrt(dx*dx+dy*dy)
                if isfinite(r) && r>sqrt(eps(T))
                    _h0_h1_at_r!(h0vals,h1vals,Float64(r),plans0,plans1)
                    for m in 1:Mk
                        coeff=khalf[m]*(fac*C.sp[p,i]*h0vals[m])
                        for qid in 1:4
                            q=C.idxp[p,i,qid]
                            As[m][gi,row_range[q]]+=coeff*C.wtp[p,i,qid]
                        end
                    end
                end
            end
            if ui-Δu>zero(T)
                dx=xi-C.xm[p,i]
                dy=yi-C.ym[p,i]
                r=sqrt(dx*dx+dy*dy)
                if isfinite(r) && r>sqrt(eps(T))
                    _h0_h1_at_r!(h0vals,h1vals,Float64(r),plans0,plans1)
                    for m in 1:Mk
                        coeff=khalf[m]*(fac*C.sm[p,i]*h0vals[m])
                        for qid in 1:4
                            q=C.idxm[p,i,qid]
                            As[m][gi,row_range[q]]+=coeff*C.wtm[p,i,qid]
                        end
                    end
                end
            end
        end
    end
    return nothing
end

function _scatter_local4_multi!(As::Vector{<:AbstractMatrix{ComplexF64}},gi::Int,col_range::UnitRange{Int},coeffs::AbstractVector{ComplexF64},idx,wt)
    @inbounds for m in eachindex(As)
        coeff=coeffs[m]
        for qid in 1:4
            q=idx[qid]
            As[m][gi,col_range[q]]+=coeff*wt[qid]
        end
    end
    return nothing
end

function _assemble_self_alpert_cheb!(solver::CFIE_alpert{T},As::Vector{<:AbstractMatrix{ComplexF64}},pts::BoundaryPointsCFIE{T},G::CFIEGeomCache{T},C,row_range::UnitRange{Int},ks::Vector{ComplexF64},rule::AlpertLogRule{T},plans0::Vector{ChebHankelPlanH},plans1::Vector{ChebHankelPlanH},h0_tls::Vector{Vector{ComplexF64}},h1_tls::Vector{Vector{ComplexF64}},coeff_tls::Vector{Vector{ComplexF64}};multithreaded::Bool=true) where {T<:Real}
    return pts.is_periodic ? _assemble_self_alpert_periodic_cheb!(As,pts,G,C,row_range,ks,rule,plans0,plans1,h0_tls,h1_tls,coeff_tls;multithreaded=multithreaded) : _assemble_self_alpert_smooth_panel_cheb!(solver,As,pts,G,C,row_range,ks,rule,plans0,plans1,h0_tls,h1_tls,coeff_tls;multithreaded=multithreaded)
end

##################################################################
#### COMPOSITE SELF-ASSEMBLY + IMAGE BLOCKS, CHEB, MULTI-k #######
##################################################################

function _assemble_self_alpert_composite_component_cheb!(solver::CFIE_alpert{T},As::Vector{<:AbstractMatrix{ComplexF64}},pts::Vector{BoundaryPointsCFIE{T}},Gs::Vector{CFIEGeomCache{T}},Cs,offs::Vector{Int},ks::Vector{ComplexF64},rule::AlpertLogRule{T},topo::AlpertCompositeTopology{T},gmap::Vector{Int},plans0::Vector{ChebHankelPlanH},plans1::Vector{ChebHankelPlanH},h0_tls::Vector{Vector{ComplexF64}},h1_tls::Vector{Vector{ComplexF64}},coeff_tls::Vector{Vector{ComplexF64}};multithreaded::Bool=true) where {T<:Real}
    Mk=length(ks)
    αD=Vector{ComplexF64}(undef,Mk)
    khalf=Vector{ComplexF64}(undef,Mk)
    @inbounds for m in 1:Mk
        αD[m]=0.5im*ks[m]
        khalf[m]=0.5*ks[m]
    end
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
            tid=Threads.threadid()
            h0vals=h0_tls[tid]
            h1vals=h1_tls[tid]
            coeffs=coeff_tls[tid]
            gi=ra[i]
            xi=Xa[i]
            yi=Ya[i]
            si=Ga.speed[i]
            κi=Ga.kappa[i]
            ui=Ca.us[i]
            diagv=1.0-(ha*si*κi)
            @inbounds for m in 1:Mk
                As[m][gi,gi]+=diagv
            end
            for sidx in eachindex(gmap)
                bidx=gmap[sidx]
                pb=pts[bidx]
                rb=offs[bidx]:(offs[bidx+1]-1)
                Xb=getindex.(pb.xy,1)
                Yb=getindex.(pb.xy,2)
                dXb=getindex.(pb.tangent,1)
                dYb=getindex.(pb.tangent,2)
                sb=@. sqrt(dXb^2+dYb^2)
                Nb=length(pb.xy)
                for j in 1:Nb
                    gj=rb[j]
                    if !(bidx==aidx && j==i)
                        dx=xi-Xb[j]
                        dy=yi-Yb[j]
                        r2=muladd(dx,dx,dy*dy)
                        if r2>(eps(T))^2
                            r=sqrt(r2)
                            invr=inv(r)
                            inn=dYb[j]*dx-dXb[j]*dy
                            _h0_h1_at_r!(h0vals,h1vals,Float64(r),plans0,plans1)
                            @inbounds for m in 1:Mk
                                As[m][gi,gj]-=pb.ws[j]*(αD[m]*inn*h1vals[m]*invr)
                            end
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
                        dx=xi-Xb[j]
                        dy=yi-Yb[j]
                        r2=muladd(dx,dx,dy*dy)
                        if r2>(eps(T))^2
                            r=sqrt(r2)
                            _h0_h1_at_r!(h0vals,h1vals,Float64(r),plans0,plans1)
                            sj=sb[j]
                            @inbounds for m in 1:Mk
                                As[m][gi,gj]+=khalf[m]*(pb.ws[j]*sj*h0vals[m])
                            end
                        end
                    end
                end
            end
            for p in 1:jcorr
                fac=ha*rule.w[p]
                Δu=ha*rule.x[p]
                if ui+Δu<one(T)
                    dx=xi-Ca.xp[p,i]
                    dy=yi-Ca.yp[p,i]
                    r=sqrt(dx*dx+dy*dy)
                    if isfinite(r) && r>sqrt(eps(T))
                        _h0_h1_at_r!(h0vals,h1vals,Float64(r),plans0,plans1)
                        @inbounds for m in 1:Mk
                            coeffs[m]=khalf[m]*(fac*Ca.sp[p,i]*h0vals[m])
                        end
                        _scatter_local4_multi!(As,gi,ra,coeffs,Ca.idxp[p,i,:],Ca.wtp[p,i,:])
                    end
                end
                if ui-Δu>zero(T)
                    dx=xi-Ca.xm[p,i]
                    dy=yi-Ca.ym[p,i]
                    r=sqrt(dx*dx+dy*dy)
                    if isfinite(r) && r>sqrt(eps(T))
                        _h0_h1_at_r!(h0vals,h1vals,Float64(r),plans0,plans1)
                        @inbounds for m in 1:Mk
                            coeffs[m]=khalf[m]*(fac*Ca.sm[p,i]*h0vals[m])
                        end
                        _scatter_local4_multi!(As,gi,ra,coeffs,Ca.idxm[p,i,:],Ca.wtm[p,i,:])
                    end
                end
            end
            if right_smooth && next_idx!=0
                for p in 1:jcorr
                    Δu=ha*rule.x[p]
                    if ui+Δu>=one(T)
                        u2=ui+Δu-one(T)
                        x,y,tx,ty,s2,idx2,wt2=_eval_on_open_panel_local4(next_pts,u2)
                        dx=xi-x
                        dy=yi-y
                        r=sqrt(dx*dx+dy*dy)
                        if isfinite(r) && r>sqrt(eps(T))
                            fac=ha*rule.w[p]
                            _h0_h1_at_r!(h0vals,h1vals,Float64(r),plans0,plans1)
                            @inbounds for m in 1:Mk
                                coeffs[m]=khalf[m]*(fac*s2*h0vals[m])
                            end
                            _scatter_local4_multi!(As,gi,next_ra,coeffs,idx2,wt2)
                        end
                    end
                end
            end
            if left_smooth && prev_idx!=0
                for p in 1:jcorr
                    Δu=ha*rule.x[p]
                    if ui-Δu<=zero(T)
                        u2=one(T)+ui-Δu
                        x,y,tx,ty,s2,idx2,wt2=_eval_on_open_panel_local4(prev_pts,u2)
                        dx=xi-x
                        dy=yi-y
                        r=sqrt(dx*dx+dy*dy)
                        if isfinite(r) && r>sqrt(eps(T))
                            fac=ha*rule.w[p]
                            _h0_h1_at_r!(h0vals,h1vals,Float64(r),plans0,plans1)
                            @inbounds for m in 1:Mk
                                coeffs[m]=khalf[m]*(fac*s2*h0vals[m])
                            end
                            _scatter_local4_multi!(As,gi,prev_ra,coeffs,idx2,wt2)
                        end
                    end
                end
            end
        end
    end
    return nothing
end

function _assemble_all_self_alpert_composite_cheb!(solver::CFIE_alpert{T},As::Vector{<:AbstractMatrix{ComplexF64}},pts::Vector{BoundaryPointsCFIE{T}},Gs::Vector{CFIEGeomCache{T}},Cs,offs::Vector{Int},ks::Vector{ComplexF64},rule::AlpertLogRule{T},topos::Vector{AlpertCompositeTopology{T}},gmaps::Vector{Vector{Int}},plans0::Vector{ChebHankelPlanH},plans1::Vector{ChebHankelPlanH},h0_tls::Vector{Vector{ComplexF64}},h1_tls::Vector{Vector{ComplexF64}},coeff_tls::Vector{Vector{ComplexF64}};multithreaded::Bool=true) where {T<:Real}
    @inbounds for c in eachindex(gmaps)
        gmap=gmaps[c]
        if length(gmap)==1 && pts[gmap[1]].is_periodic
            a=gmap[1]
            ra=offs[a]:(offs[a+1]-1)
            _assemble_self_alpert_cheb!(solver,As,pts[a],Gs[a],Cs[a],ra,ks,rule,plans0,plans1,h0_tls,h1_tls,coeff_tls;multithreaded=multithreaded)
        else
            _assemble_self_alpert_composite_component_cheb!(solver,As,pts,Gs,Cs,offs,ks,rule,topos[c],gmap,plans0,plans1,h0_tls,h1_tls,coeff_tls;multithreaded=multithreaded)
        end
    end
    return nothing
end

################################
#### IMAGE / SYMMETRY BLOCKS ####
################################

function _add_image_block_cheb!(As::Vector{<:AbstractMatrix{ComplexF64}},ra::UnitRange{Int},rb::UnitRange{Int},pa::BoundaryPointsCFIE{T},pb::BoundaryPointsCFIE{T},ks::Vector{ComplexF64},plans0::Vector{ChebHankelPlanH},plans1::Vector{ChebHankelPlanH},h0_tls::Vector{Vector{ComplexF64}},h1_tls::Vector{Vector{ComplexF64}},qfun,tfun,weight;multithreaded::Bool=true) where {T<:Real}
    Mk=length(ks)
    αD=Vector{ComplexF64}(undef,Mk)
    iks=Vector{ComplexF64}(undef,Mk)
    αS=0.5im
    @inbounds for m in 1:Mk
        αD[m]=0.5im*ks[m]
        iks[m]=1im*ks[m]
    end
    Na=length(pa.xy)
    Nb=length(pb.xy)
    Xa=getindex.(pa.xy,1)
    Ya=getindex.(pa.xy,2)
    @use_threads multithreading=multithreaded for j in 1:Nb
        tid=Threads.threadid()
        h0vals=h0_tls[tid]
        h1vals=h1_tls[tid]
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
            _check_r(r,"image-block",i,j)
            invr=inv(r)
            inn=tyj*dx-txj*dy
            _h0_h1_at_r!(h0vals,h1vals,Float64(r),plans0,plans1)
            for m in 1:Mk
                dval=weight*wd*(αD[m]*inn*h1vals[m]*invr)
                sval=weight*ws*(αS*h0vals[m])
                As[m][gi,gj]-=(dval+iks[m]*sval)
            end
        end
    end
    return nothing
end

function _assemble_reflection_images_cheb!(As::Vector{<:AbstractMatrix{ComplexF64}},ra::UnitRange{Int},rb::UnitRange{Int},pa::BoundaryPointsCFIE{T},pb::BoundaryPointsCFIE{T},solver::CFIE_alpert{T},billiard::Bi,ks::Vector{ComplexF64},plans0::Vector{ChebHankelPlanH},plans1::Vector{ChebHankelPlanH},h0_tls::Vector{Vector{ComplexF64}},h1_tls::Vector{Vector{ComplexF64}},sym::Reflection;multithreaded::Bool=true) where {T<:Real,Bi<:AbsBilliard}
    if sym.axis==:y_axis
        _add_image_block_cheb!(As,ra,rb,pa,pb,ks,plans0,plans1,h0_tls,h1_tls,q->image_point_x(q,billiard),t->image_tangent_x(t),image_weight(sym);multithreaded=multithreaded)
    elseif sym.axis==:x_axis
        _add_image_block_cheb!(As,ra,rb,pa,pb,ks,plans0,plans1,h0_tls,h1_tls,q->image_point_y(q,billiard),t->image_tangent_y(t),image_weight(sym);multithreaded=multithreaded)
    elseif sym.axis==:origin
        σx=image_weight_x(sym)
        σy=image_weight_y(sym)
        σxy=image_weight_xy(sym)
        _add_image_block_cheb!(As,ra,rb,pa,pb,ks,plans0,plans1,h0_tls,h1_tls,q->image_point_x(q,billiard),t->image_tangent_x(t),σx;multithreaded=multithreaded)
        _add_image_block_cheb!(As,ra,rb,pa,pb,ks,plans0,plans1,h0_tls,h1_tls,q->image_point_y(q,billiard),t->image_tangent_y(t),σy;multithreaded=multithreaded)
        _add_image_block_cheb!(As,ra,rb,pa,pb,ks,plans0,plans1,h0_tls,h1_tls,q->image_point_xy(q,billiard),t->image_tangent_xy(t),σxy;multithreaded=multithreaded)
    else
        error("Unknown reflection axis $(sym.axis)")
    end
    return nothing
end

function _assemble_rotation_images_cheb!(As::Vector{<:AbstractMatrix{ComplexF64}},ra::UnitRange{Int},rb::UnitRange{Int},pa::BoundaryPointsCFIE{T},pb::BoundaryPointsCFIE{T},ks::Vector{ComplexF64},plans0::Vector{ChebHankelPlanH},plans1::Vector{ChebHankelPlanH},h0_tls::Vector{Vector{ComplexF64}},h1_tls::Vector{Vector{ComplexF64}},sym::Rotation,costab,sintab,χ;multithreaded::Bool=true) where {T<:Real}
    for l in 1:(sym.n-1)
        phase=χ[l+1]
        _add_image_block_cheb!(As,ra,rb,pa,pb,ks,plans0,plans1,h0_tls,h1_tls,q->image_point(sym,q,l,costab,sintab),t->image_tangent(sym,t,l,costab,sintab),phase;multithreaded=multithreaded)
    end
    return nothing
end

############################################################
#### TOP-LEVEL MATRIX ASSEMBLY (CHEB, MULTI-k, NO SYM) #####
############################################################

function compute_kernel_matrices_CFIE_alpert_chebyshev!(As::Vector{<:AbstractMatrix{ComplexF64}},solver::CFIE_alpert{T},pts::Vector{BoundaryPointsCFIE{T}},ws::CFIEAlpertChebWorkspace{T},ks::Vector{ComplexF64};multithreaded::Bool=true) where {T<:Real}
    Mk=length(ks)
    @assert length(As)==Mk
    for m in 1:Mk
        fill!(As[m],0.0+0im)
    end
    direct=ws.direct
    offs=direct.offs
    Gs=direct.Gs
    Cs=direct.Cs
    rule=direct.rule
    topos=direct.topos
    gmaps=direct.gmaps
    panel_to_comp=direct.panel_to_comp
    plans0=ws.plans0
    plans1=ws.plans1
    h0_tls=ws.bessel_ws.h0_tls
    h1_tls=ws.bessel_ws.h1_tls
    coeff_tls=ws.bessel_ws.coeff_tls
    nc=length(pts)
    if topos===nothing
        @inbounds for a in 1:nc
            ra=offs[a]:(offs[a+1]-1)
            _assemble_self_alpert_cheb!(solver,As,pts[a],Gs[a],Cs[a],ra,ks,rule,plans0,plans1,h0_tls,h1_tls,coeff_tls;multithreaded=multithreaded)
        end
    else
        _assemble_all_self_alpert_composite_cheb!(solver,As,pts,Gs,Cs,offs,ks,rule,topos,gmaps,plans0,plans1,h0_tls,h1_tls,coeff_tls;multithreaded=multithreaded)
    end
    αD=Vector{ComplexF64}(undef,Mk)
    @inbounds for m in 1:Mk
        αD[m]=0.5im*ks[m]
    end
    for a in 1:nc, b in 1:nc
        a==b && continue
        if panel_to_comp!==nothing
            ca=panel_to_comp[a]
            cb=panel_to_comp[b]
            ca!=0 && ca==cb && continue
        end
        pa=pts[a]
        pb=pts[b]
        Na=length(pa.xy)
        Nb=length(pb.xy)
        ra=offs[a]:(offs[a+1]-1)
        rb=offs[b]:(offs[b+1]-1)
        Xa=getindex.(pa.xy,1)
        Ya=getindex.(pa.xy,2)
        Xb=getindex.(pb.xy,1)
        Yb=getindex.(pb.xy,2)
        dXb=getindex.(pb.tangent,1)
        dYb=getindex.(pb.tangent,2)
        sb=@. sqrt(dXb^2+dYb^2)
        @use_threads multithreading=multithreaded for j in 1:Nb
            tid=Threads.threadid()
            h0vals=h0_tls[tid]
            h1vals=h1_tls[tid]
            gj=rb[j]
            xj=Xb[j]
            yj=Yb[j]
            txj=dXb[j]
            tyj=dYb[j]
            sj=sb[j]
            wd=pb.ws[j]
            wsj=pb.ws[j]*sj
            @inbounds for i in 1:Na
                gi=ra[i]
                dx=Xa[i]-xj
                dy=Ya[i]-yj
                r2=muladd(dx,dx,dy*dy)
                r2<=(eps(T))^2 && continue
                r=sqrt(r2)
                inn=tyj*dx-txj*dy
                invr=inv(r)
                _h0_h1_at_r!(h0vals,h1vals,Float64(r),plans0,plans1)
                for m in 1:Mk
                    dval=wd*(αD[m]*inn*h1vals[m]*invr)
                    sval=wsj*(0.5im*h0vals[m])
                    As[m][gi,gj]-=(dval+(1im*ks[m])*sval)
                end
            end
        end
    end
    return nothing
end

############################################################
#### WITH SYMMETRY (CHEB, MULTI-k) ##########################
############################################################

function compute_kernel_matrices_CFIE_alpert_chebyshev_symmetry!(As::Vector{<:AbstractMatrix{ComplexF64}},solver::CFIE_alpert{T},pts::Vector{BoundaryPointsCFIE{T}},ws::CFIEAlpertChebWorkspace{T},ks::Vector{ComplexF64};multithreaded::Bool=true) where {T<:Real}
    symmetry=solver.symmetry
    isnothing(symmetry) && error("called symmetry version without symmetry")
    Mk=length(ks)
    for m in 1:Mk
        fill!(As[m],0.0+0im)
    end
    direct=ws.direct
    offs=direct.offs
    Gs=direct.Gs
    Cs=direct.Cs
    rule=direct.rule
    topos=direct.topos
    gmaps=direct.gmaps
    panel_to_comp=direct.panel_to_comp
    plans0=ws.plans0
    plans1=ws.plans1
    h0_tls=ws.bessel_ws.h0_tls
    h1_tls=ws.bessel_ws.h1_tls
    coeff_tls=ws.bessel_ws.coeff_tls
    nc=length(pts)
    if topos===nothing
        @inbounds for a in 1:nc
            ra=offs[a]:(offs[a+1]-1)
            _assemble_self_alpert_cheb!(solver,As,pts[a],Gs[a],Cs[a],ra,ks,rule,plans0,plans1,h0_tls,h1_tls,coeff_tls;multithreaded=multithreaded)
        end
    else
        _assemble_all_self_alpert_composite_cheb!(solver,As,pts,Gs,Cs,offs,ks,rule,topos,gmaps,plans0,plans1,h0_tls,h1_tls,coeff_tls;multithreaded=multithreaded)
    end
    αD=Vector{ComplexF64}(undef,Mk)
    @inbounds for m in 1:Mk
        αD[m]=0.5im*ks[m]
    end
    for a in 1:nc, b in 1:nc
        a==b && continue
        if panel_to_comp!==nothing
            ca=panel_to_comp[a]
            cb=panel_to_comp[b]
            ca!=0 && ca==cb && continue
        end
        pa=pts[a]
        pb=pts[b]
        Na=length(pa.xy)
        Nb=length(pb.xy)
        ra=offs[a]:(offs[a+1]-1)
        rb=offs[b]:(offs[b+1]-1)
        Xa=getindex.(pa.xy,1)
        Ya=getindex.(pa.xy,2)
        Xb=getindex.(pb.xy,1)
        Yb=getindex.(pb.xy,2)
        dXb=getindex.(pb.tangent,1)
        dYb=getindex.(pb.tangent,2)
        sb=@. sqrt(dXb^2+dYb^2)
        @use_threads multithreading=multithreaded for j in 1:Nb
            tid=Threads.threadid()
            h0vals=h0_tls[tid]
            h1vals=h1_tls[tid]
            gj=rb[j]
            xj=Xb[j]
            yj=Yb[j]
            txj=dXb[j]
            tyj=dYb[j]
            sj=sb[j]
            wd=pb.ws[j]
            wsj=pb.ws[j]*sj
            @inbounds for i in 1:Na
                gi=ra[i]
                dx=Xa[i]-xj
                dy=Ya[i]-yj
                r2=muladd(dx,dx,dy*dy)
                r2<=(eps(T))^2 && continue
                r=sqrt(r2)
                _check_r(r,"symmetry before images",i,j)
                invr=inv(r)
                inn=tyj*dx-txj*dy
                _h0_h1_at_r!(h0vals,h1vals,Float64(r),plans0,plans1)
                for m in 1:Mk
                    dval=wd*(αD[m]*inn*h1vals[m]*invr)
                    sval=wsj*(0.5im*h0vals[m])
                    As[m][gi,gj]-=(dval+(1im*ks[m])*sval)
                end
            end
        end
    end
    for sym in symmetry
        if sym isa Reflection
            for a in 1:nc, b in 1:nc
                ra=offs[a]:(offs[a+1]-1)
                rb=offs[b]:(offs[b+1]-1)
                _assemble_reflection_images_cheb!(As,ra,rb,pts[a],pts[b],solver,solver.billiard,ks,plans0,plans1,h0_tls,h1_tls,sym;multithreaded=multithreaded)
            end
        elseif sym isa Rotation
            costab,sintab,χ=_rotation_tables(T,sym.n,sym.m)
            for a in 1:nc, b in 1:nc
                ra=offs[a]:(offs[a+1]-1)
                rb=offs[b]:(offs[b+1]-1)
                _assemble_rotation_images_cheb!(As,ra,rb,pts[a],pts[b],ks,plans0,plans1,h0_tls,h1_tls,sym,costab,sintab,χ;multithreaded=multithreaded)
            end
        else
            error("Unknown symmetry type")
        end
    end
    return nothing
end

############################################################
#### GLOBAL rmin / rmax FOR CFIE_alpert CHEB PLANS #########
############################################################

@inline function _rbounds_update!(rminmax::Base.RefValue{Tuple{Float64,Float64}},r)
    if isfinite(r) && r>0
        rmin,rmax=rminmax[]
        rr=Float64(r)
        rr<rmin && (rmin=rr)
        rr>rmax && (rmax=rr)
        rminmax[]=(rmin,rmax)
    end
    return nothing
end

function estimate_cfie_alpert_cheb_rbounds(solver::CFIE_alpert{T},pts::Vector{BoundaryPointsCFIE{T}},directws::CFIEAlpertWorkspace{T};pad=(T(0.95),T(1.05))) where {T<:Real}
    rminmax=Ref((Inf,0.0))
    offs=directws.offs
    Gs=directws.Gs
    Cs=directws.Cs
    rule=directws.rule
    topos=directws.topos
    gmaps=directws.gmaps
    nc=length(pts)
    #-------------------------------------------------------
    # 1. Direct boundary node-to-node distances
    #-------------------------------------------------------
    for a in 1:nc, b in 1:nc
        pa=pts[a]
        pb=pts[b]
        Xa=getindex.(pa.xy,1)
        Ya=getindex.(pa.xy,2)
        Xb=getindex.(pb.xy,1)
        Yb=getindex.(pb.xy,2)
        Na=length(Xa)
        Nb=length(Xb)
        @inbounds for j in 1:Nb, i in 1:Na
            if a==b && i==j
                continue
            end
            dx=Xa[i]-Xb[j]
            dy=Ya[i]-Yb[j]
            _rbounds_update!(rminmax,sqrt(dx*dx+dy*dy))
        end
    end
    #-------------------------------------------------------
    # 2. Alpert shifted source points from caches
    #-------------------------------------------------------
    for a in 1:nc
        pa=pts[a]
        Ca=Cs[a]
        Xa=getindex.(pa.xy,1)
        Ya=getindex.(pa.xy,2)
        Na=length(pa.xy)
        if Ca isa AlpertPeriodicCache{T}
            jcorr=size(Ca.xp,1)
            @inbounds for i in 1:Na, p in 1:jcorr
                dx=Xa[i]-Ca.xp[p,i]
                dy=Ya[i]-Ca.yp[p,i]
                _rbounds_update!(rminmax,sqrt(dx*dx+dy*dy))
                dx=Xa[i]-Ca.xm[p,i]
                dy=Ya[i]-Ca.ym[p,i]
                _rbounds_update!(rminmax,sqrt(dx*dx+dy*dy))
            end
        elseif Ca isa AlpertSmoothPanelCache{T}
            jcorr=size(Ca.xp,1)
            @inbounds for i in 1:Na, p in 1:jcorr
                dx=Xa[i]-Ca.xp[p,i]
                dy=Ya[i]-Ca.yp[p,i]
                _rbounds_update!(rminmax,sqrt(dx*dx+dy*dy))
                dx=Xa[i]-Ca.xm[p,i]
                dy=Ya[i]-Ca.ym[p,i]
                _rbounds_update!(rminmax,sqrt(dx*dx+dy*dy))
            end
        else
            error("Unknown Alpert cache type $(typeof(Ca))")
        end
    end
    #-------------------------------------------------------
    # 3. Smooth continuation across composite joins
    #-------------------------------------------------------
    if !(topos===nothing || gmaps===nothing)
        @inbounds for c in eachindex(gmaps)
            topo=topos[c]
            gmap=gmaps[c]
            for l in eachindex(gmap)
                aidx=gmap[l]
                pa=pts[aidx]
                Ca=Cs[aidx]
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
                if !(Ca isa AlpertSmoothPanelCache{T})
                    continue
                end
                jcorr=rule.j
                @inbounds for i in 1:Na
                    xi=Xa[i]
                    yi=Ya[i]
                    ui=Ca.us[i]
                    if right_smooth && next_idx!=0
                        for p in 1:jcorr
                            Δu=ha*rule.x[p]
                            if ui+Δu>=one(T)
                                u2=ui+Δu-one(T)
                                x,y,tx,ty,s2,idx2,wt2=_eval_on_open_panel_local4(next_pts,u2)
                                dx=xi-x
                                dy=yi-y
                                _rbounds_update!(rminmax,sqrt(dx*dx+dy*dy))
                            end
                        end
                    end
                    if left_smooth && prev_idx!=0
                        for p in 1:jcorr
                            Δu=ha*rule.x[p]
                            if ui-Δu<=zero(T)
                                u2=one(T)+ui-Δu
                                x,y,tx,ty,s2,idx2,wt2=_eval_on_open_panel_local4(prev_pts,u2)
                                dx=xi-x
                                dy=yi-y
                                _rbounds_update!(rminmax,sqrt(dx*dx+dy*dy))
                            end
                        end
                    end
                end
            end
        end
    end
    #-------------------------------------------------------
    # 4. Symmetry image distances
    #-------------------------------------------------------
    symmetry=solver.symmetry
    if !isnothing(symmetry)
        for sym in symmetry
            if sym isa Reflection
                for a in 1:nc, b in 1:nc
                    pa=pts[a]
                    pb=pts[b]
                    Xa=getindex.(pa.xy,1)
                    Ya=getindex.(pa.xy,2)
                    Na=length(pa.xy)
                    Nb=length(pb.xy)
                    if sym.axis==:y_axis
                        @inbounds for j in 1:Nb
                            qimg=image_point_x(pb.xy[j],solver.billiard)
                            xj=qimg[1]
                            yj=qimg[2]
                            for i in 1:Na
                                dx=Xa[i]-xj
                                dy=Ya[i]-yj
                                _rbounds_update!(rminmax,sqrt(dx*dx+dy*dy))
                            end
                        end
                    elseif sym.axis==:x_axis
                        @inbounds for j in 1:Nb
                            qimg=image_point_y(pb.xy[j],solver.billiard)
                            xj=qimg[1]
                            yj=qimg[2]
                            for i in 1:Na
                                dx=Xa[i]-xj
                                dy=Ya[i]-yj
                                _rbounds_update!(rminmax,sqrt(dx*dx+dy*dy))
                            end
                        end
                    elseif sym.axis==:origin
                        @inbounds for j in 1:Nb
                            qx=image_point_x(pb.xy[j],solver.billiard)
                            qy=image_point_y(pb.xy[j],solver.billiard)
                            qxy=image_point_xy(pb.xy[j],solver.billiard)
                            for i in 1:Na
                                dx=Xa[i]-qx[1]
                                dy=Ya[i]-qx[2]
                                _rbounds_update!(rminmax,sqrt(dx*dx+dy*dy))
                                dx=Xa[i]-qy[1]
                                dy=Ya[i]-qy[2]
                                _rbounds_update!(rminmax,sqrt(dx*dx+dy*dy))
                                dx=Xa[i]-qxy[1]
                                dy=Ya[i]-qxy[2]
                                _rbounds_update!(rminmax,sqrt(dx*dx+dy*dy))
                            end
                        end
                    else
                        error("Unknown reflection axis $(sym.axis)")
                    end
                end
            elseif sym isa Rotation
                costab,sintab,χ=_rotation_tables(T,sym.n,sym.m)
                for a in 1:nc, b in 1:nc
                    pa=pts[a]
                    pb=pts[b]
                    Xa=getindex.(pa.xy,1)
                    Ya=getindex.(pa.xy,2)
                    Na=length(pa.xy)
                    Nb=length(pb.xy)
                    for l in 1:(sym.n-1)
                        @inbounds for j in 1:Nb
                            qimg=image_point(sym,pb.xy[j],l,costab,sintab)
                            xj=qimg[1]
                            yj=qimg[2]
                            for i in 1:Na
                                dx=Xa[i]-xj
                                dy=Ya[i]-yj
                                _rbounds_update!(rminmax,sqrt(dx*dx+dy*dy))
                            end
                        end
                    end
                end
            else
                error("Unknown symmetry type $(typeof(sym))")
            end
        end
    end
    rmin,rmax=rminmax[]
    @assert isfinite(rmin) && rmax>0 "Could not determine valid CFIE_alpert Chebyshev r-bounds"
    return Float64(pad[1])*rmin,Float64(pad[2])*rmax
end

###################################
#### WORKSPACE CONSTRUCTOR ########
###################################

function build_cfie_alpert_cheb_workspace(solver::CFIE_alpert{T},pts::Vector{BoundaryPointsCFIE{T}},direct::CFIEAlpertWorkspace{T,C},ks::Vector{ComplexF64};npanels::Int=10000,M::Int=5,grading::Symbol=:uniform,geo_ratio::Real=1.05,pad=(T(0.95),T(1.05)),plan_nthreads::Int=1,ntls::Int=Threads.nthreads()) where {T<:Real,C}
    block_cache=build_cfie_alpert_cheb_block_caches(pts;npanels=npanels,M=M,grading=grading,geo_ratio=geo_ratio,pad=pad)
    rmin,rmax=estimate_cfie_alpert_cheb_rbounds(solver,pts,direct;pad=pad)
    plans0,plans1=build_CFIE_alpert_plans(ks,rmin,rmax;npanels=npanels,M=M,grading=grading,geo_ratio=geo_ratio,nthreads=plan_nthreads)
    bessel_ws=CFIE_H0_H1_BesselWorkspace(length(ks);ntls=ntls)
    return CFIEAlpertChebWorkspace{T,C}(direct,block_cache,plans0,plans1,bessel_ws,ks,length(ks))
end

###########################################################
############## HIGH LEVEL ENTRY POINTS ####################
###########################################################

function compute_kernel_matrices_CFIE_alpert_chebyshev!(As::Vector{<:AbstractMatrix{ComplexF64}},solver::CFIE_alpert{T},pts::Vector{BoundaryPointsCFIE{T}},ws::CFIEAlpertChebWorkspace{T};multithreaded::Bool=true) where {T<:Real}
    if isnothing(solver.symmetry)
        return compute_kernel_matrices_CFIE_alpert_chebyshev!(As,solver,pts,ws,ws.ks;multithreaded=multithreaded)
    else
        return compute_kernel_matrices_CFIE_alpert_chebyshev_symmetry!(As,solver,pts,ws,ws.ks;multithreaded=multithreaded)
    end
end

function compute_kernel_matrices_CFIE_alpert_chebyshev(solver::CFIE_alpert{T},pts::Vector{BoundaryPointsCFIE{T}},ws::CFIEAlpertChebWorkspace{T};multithreaded::Bool=true) where {T<:Real}
    Ntot=ws.direct.Ntot
    Mk=ws.Mk
    As=[Matrix{ComplexF64}(undef,Ntot,Ntot) for _ in 1:Mk]
    compute_kernel_matrices_CFIE_alpert_chebyshev!(As,solver,pts,ws;multithreaded=multithreaded)
    return As
end
