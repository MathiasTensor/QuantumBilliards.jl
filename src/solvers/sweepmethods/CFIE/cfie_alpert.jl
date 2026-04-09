struct AlpertPeriodicCache{T<:Real}
    xp::Matrix{T}
    yp::Matrix{T}
    txp::Matrix{T}
    typ::Matrix{T}
    sp::Matrix{T}
    xm::Matrix{T}
    ym::Matrix{T}
    txm::Matrix{T}
    tym::Matrix{T}
    sm::Matrix{T}
    offsp::Matrix{Int}
    wtp::Matrix{T}
    offsm::Matrix{Int}
    wtm::Matrix{T}
    ninterp::Int
end

struct AlpertSmoothPanelCache{T<:Real}
    us::Vector{T}
    xp::Matrix{T}
    yp::Matrix{T}
    txp::Matrix{T}
    typ::Matrix{T}
    sp::Matrix{T}
    xm::Matrix{T}
    ym::Matrix{T}
    txm::Matrix{T}
    tym::Matrix{T}
    sm::Matrix{T}
    idxp::Array{Int,3}
    wtp::Array{T,3}
    idxm::Array{Int,3}
    wtm::Array{T,3}
end

struct CFIEAlpertWorkspace{T<:Real,C}
    rule::AlpertLogRule{T}
    offs::Vector{Int}
    Gs::Vector{CFIEGeomCache{T}}
    Cs::Vector{C}
    topos::Union{Nothing,Vector{AlpertCompositeTopology{T}}}
    gmaps::Union{Nothing,Vector{Vector{Int}}}
    panel_to_comp::Union{Nothing,Vector{Int}}
    Ntot::Int
end

@inline function _scatter_localp!(A::AbstractMatrix{Complex{T}},gi::Int,col_range::UnitRange{Int},coeff::Complex{T},idx,wt) where {T<:Real}
    @inbounds for m in eachindex(idx)
        A[gi,col_range[idx[m]]]+=coeff*wt[m]
    end
    nothing
end

@inline function _wrap01(u::T) where {T<:Real}
    v=mod(u,one(T))
    v<zero(T) ? v+one(T) : v
end

@inline function wrap_angle(t::T) where {T<:Real}
    tp=mod(t,two_pi)
    tp==zero(T) ? T(two_pi) : tp
end

@inline function wrap_diff(t::T) where {T<:Real}
    mod(t+T(pi),two_pi)-T(pi)
end

@inline function _lagrange_weights(ξ::T,nodes::AbstractVector{T}) where {T<:Real}
    m=length(nodes)
    w=Vector{T}(undef,m)
    @inbounds for j in 1:m
        num=one(T)
        den=one(T)
        xj=nodes[j]
        for l in 1:m
            l==j && continue
            xl=nodes[l]
            num*=ξ-xl
            den*=xj-xl
        end
        w[j]=num/den
    end
    return w
end

@inline function _alpert_interp_offsets_weights(ξ::T,ninterp::Int) where {T<:Real}
    j0=floor(Int,ξ-T(ninterp)/2+one(T))
    offs=collect(j0:(j0+ninterp-1))
    wt=_lagrange_weights(ξ,T.(offs))
    return offs,wt
end

@inline function _local_offsets(p::Int)
    iseven(p) || error("Interpolation stencil size p must be even.")
    q=p÷2
    return collect(-(q-1):q)
end

@inline function _periodic_orientation_sign(ts::AbstractVector{T}) where {T<:Real}
    N=length(ts)
    N<2 && return 1
    Δ=wrap_diff(ts[mod1(2,N)]-ts[1])
    Δ>=zero(T) ? 1 : -1
end

@inline function _dinner(dx,dy,tx,ty)
    ty*dx-tx*dy
end

@inline function _panel_us(::Type{T},N::Int) where {T<:Real}
    return collect(midpoints(range(zero(T),one(T),length=N+1)))
end

@inline function _panel_smooth_localp_midpoint_data(u::T,h::T,N::Int,p::Int) where {T<:Real}
    iseven(p) || error("p must be even.")
    p<=N || error("p must satisfy p <= N.")
    q=p÷2
    s=u/h-T(1)/2
    j0=floor(Int,s)+1
    η=s-floor(T,s)
    j0=clamp(j0,q,N-q)
    offs=_local_offsets(p)
    wt=_lagrange_weights(η,T.(offs))
    idx=Vector{Int}(undef,p)
    @inbounds for m in 1:p
        idx[m]=j0+offs[m]
    end
    return idx,wt
end

@inline function _eval_shifted_source_smooth_panel_localp(u::T,h::T,X::AbstractVector{T},Y::AbstractVector{T},dX::AbstractVector{T},dY::AbstractVector{T},p::Int) where {T<:Real}
    N=length(X)
    idx,wt=_panel_smooth_localp_midpoint_data(u,h,N,p)
    x=zero(T)
    y=zero(T)
    tx=zero(T)
    ty=zero(T)
    @inbounds for m in eachindex(idx)
        q=idx[m]
        wm=wt[m]
        x+=wm*X[q]
        y+=wm*Y[q]
        tx+=wm*dX[q]
        ty+=wm*dY[q]
    end
    s=sqrt(tx*tx+ty*ty)
    return x,y,tx,ty,s,idx,wt
end

function _build_alpert_periodic_cache(solver::CFIE_alpert{T},crv::C,pts::BoundaryPointsCFIE{T},rule::AlpertLogRule{T},ord::Int) where {T<:Real,C<:AbsCurve}
    N=length(pts.xy)
    jcorr=rule.j
    ninterp=ord+3
    σ=_periodic_orientation_sign(pts.ts)
    xp=Matrix{T}(undef,jcorr,N)
    yp=similar(xp)
    txp=similar(xp)
    typ=similar(xp)
    sp=similar(xp)
    xm=Matrix{T}(undef,jcorr,N)
    ym=similar(xm)
    txm=similar(xm)
    tym=similar(xm)
    sm=similar(xm)
    offsp=Matrix{Int}(undef,jcorr,ninterp)
    wtp=Matrix{T}(undef,jcorr,ninterp)
    offsm=Matrix{Int}(undef,jcorr,ninterp)
    wtm=Matrix{T}(undef,jcorr,ninterp)
    @inbounds for p in 1:jcorr
        ξ=rule.x[p]
        ξidx=T(σ)*ξ
        op,wp=_alpert_interp_offsets_weights( ξidx,ninterp)
        om,wm=_alpert_interp_offsets_weights(-ξidx,ninterp)
        for m in 1:ninterp
            offsp[p,m]=op[m]
            wtp[p,m]=wp[m]
            offsm[p,m]=om[m]
            wtm[p,m]=wm[m]
        end
        δu=ξ/T(N)
        for i in 1:N
            ui=pts.ts[i]/T(two_pi)
            up=_wrap01(ui+δu)
            qp=curve(crv,up)
            tp=tangent(crv,up)/T(two_pi)
            xp[p,i]=qp[1]
            yp[p,i]=qp[2]
            txp[p,i]=tp[1]
            typ[p,i]=tp[2]
            sp[p,i]=sqrt(tp[1]^2+tp[2]^2)
            um=_wrap01(ui-δu)
            qm=curve(crv,um)
            tm=tangent(crv,um)/T(two_pi)
            xm[p,i]=qm[1]
            ym[p,i]=qm[2]
            txm[p,i]=tm[1]
            tym[p,i]=tm[2]
            sm[p,i]=sqrt(tm[1]^2+tm[2]^2)
        end
    end
    return AlpertPeriodicCache(xp,yp,txp,typ,sp,xm,ym,txm,tym,sm,offsp,wtp,offsm,wtm,ninterp)
end

function _build_alpert_smooth_panel_cache(pts::BoundaryPointsCFIE{T},rule::AlpertLogRule{T},p::Int) where {T<:Real}
    iseven(p) || error("Smooth-panel Alpert interpolation stencil size p must be even.")
    X=getindex.(pts.xy,1)
    Y=getindex.(pts.xy,2)
    dX=getindex.(pts.tangent,1)
    dY=getindex.(pts.tangent,2)
    N=length(X)
    p<=N || error("Smooth-panel Alpert interpolation stencil size p must satisfy p <= N.")
    h=pts.ws[1]
    jcorr=rule.j
    xp=Matrix{T}(undef,jcorr,N)
    yp=similar(xp)
    txp=similar(xp)
    typ=similar(xp)
    sp=similar(xp)
    xm=similar(xp)
    ym=similar(xp)
    txm=similar(xp)
    tym=similar(xp)
    sm=similar(xp)
    idxp=Array{Int,3}(undef,jcorr,N,p)
    idxm=Array{Int,3}(undef,jcorr,N,p)
    wtp=Array{T,3}(undef,jcorr,N,p)
    wtm=Array{T,3}(undef,jcorr,N,p)
    us=_panel_us(T,N)
    @inbounds for q in 1:jcorr
        Δu=h*rule.x[q]
        for i in 1:N
            up=us[i]+Δu
            x,y,tx,ty,s,idx,wt=_eval_shifted_source_smooth_panel_localp(up,h,X,Y,dX,dY,p)
            xp[q,i]=x
            yp[q,i]=y
            txp[q,i]=tx
            typ[q,i]=ty
            sp[q,i]=s
            for m in 1:p
                idxp[q,i,m]=idx[m]
                wtp[q,i,m]=wt[m]
            end
            um=us[i] - Δu
            x,y,tx,ty,s,idx,wt=_eval_shifted_source_smooth_panel_localp(um,h,X,Y,dX,dY,p)
            xm[q,i]=x
            ym[q,i]=y
            txm[q,i]=tx
            tym[q,i]=ty
            sm[q,i]=s
            for m in 1:p
                idxm[q,i,m]=idx[m]
                wtm[q,i,m]=wt[m]
            end
        end
    end
    return AlpertSmoothPanelCache(us,xp,yp,txp,typ,sp,xm,ym,txm,tym,sm,idxp,wtp,idxm,wtm)
end

function _build_alpert_component_cache(solver::CFIE_alpert{T},crv,pts::BoundaryPointsCFIE{T},rule::AlpertLogRule{T},ord::Int) where {T<:Real}
    pts.is_periodic ? _build_alpert_periodic_cache(solver,crv,pts,rule,ord) : _build_alpert_smooth_panel_cache(pts,rule,ord)
end

function _assemble_self_alpert_periodic!(A::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},G::CFIEGeomCache{T},C::AlpertPeriodicCache{T},row_range::UnitRange{Int},k::T,rule::AlpertLogRule{T};multithreaded::Bool=true) where {T<:Real}
    αD=Complex{T}(0,k/2)
    αS=Complex{T}(0,one(T)/2)
    ik=Complex{T}(0,k)
    X=getindex.(pts.xy,1)
    Y=getindex.(pts.xy,2)
    N=length(pts.ts)
    h=pts.ws[1]
    a=rule.a
    jcorr=rule.j
    ninterp=C.ninterp
    @use_threads multithreading=multithreaded for i in 1:N
        gi=row_range[i]
        xi=X[i]
        yi=Y[i]
        A[gi,gi]+=one(Complex{T})
        @inbounds for j in 1:N
            j==i && continue
            gj=row_range[j]
            rij=G.R[i,j]
            inn=G.inner[i,j]
            invr=G.invR[i,j]
            A[gi,gj] -= h*(αD*inn*H(1,k*rij)*invr)
        end
        @inbounds for m in (-a+1):(a-1)
            m==0 && continue
            j=mod1(i+m,N)
            gj=row_range[j]
            rij=G.R[i,j]
            inn=G.inner[i,j]
            invr=G.invR[i,j]
            A[gi,gj] += h*(αD*inn*H(1,k*rij)*invr)
        end
        @inbounds for p in 1:jcorr
            fac=h*rule.w[p]
            dx=xi-C.xp[p,i]
            dy=yi-C.yp[p,i]
            r2=muladd(dx,dx,dy*dy)
            if isfinite(r2) && r2>(eps(T))^2
                r=sqrt(r2)
                inn=_dinner(dx,dy,C.txp[p,i],C.typ[p,i])
                coeff= -(fac*(αD*inn*H(1,k*r)/r))
                for m in 1:ninterp
                    q=mod1(i+C.offsp[p,m],N)
                    A[gi,row_range[q]] += coeff*C.wtp[p,m]
                end
            end
            dx=xi-C.xm[p,i]
            dy=yi-C.ym[p,i]
            r2=muladd(dx,dx,dy*dy)
            if isfinite(r2) && r2>(eps(T))^2
                r=sqrt(r2)
                inn=_dinner(dx,dy,C.txm[p,i],C.tym[p,i])
                coeff= -(fac*(αD*inn*H(1,k*r)/r))
                for m in 1:ninterp
                    q=mod1(i+C.offsm[p,m],N)
                    A[gi,row_range[q]] += coeff*C.wtm[p,m]
                end
            end
        end
        @inbounds for j in 1:N
            j==i && continue
            gj=row_range[j]
            A[gi,gj] -= ik*(h*(αS*H(0,k*G.R[i,j])*G.speed[j]))
        end
        @inbounds for m in (-a+1):(a-1)
            m==0 && continue
            j=mod1(i+m,N)
            gj=row_range[j]
            A[gi,gj] += ik*(h*(αS*H(0,k*G.R[i,j])*G.speed[j]))
        end
        @inbounds for p in 1:jcorr
            fac=h*rule.w[p]
            dx=xi-C.xp[p,i]
            dy=yi-C.yp[p,i]
            r2=muladd(dx,dx,dy*dy)
            if isfinite(r2) && r2>(eps(T))^2
                r=sqrt(r2)
                coeff= -ik*(fac*(αS*H(0,k*r)*C.sp[p,i]))
                for m in 1:ninterp
                    q=mod1(i+C.offsp[p,m],N)
                    A[gi,row_range[q]] += coeff*C.wtp[p,m]
                end
            end
            dx=xi-C.xm[p,i]
            dy=yi-C.ym[p,i]
            r2=muladd(dx,dx,dy*dy)
            if isfinite(r2) && r2>(eps(T))^2
                r=sqrt(r2)
                coeff= -ik*(fac*(αS*H(0,k*r)*C.sm[p,i]))
                for m in 1:ninterp
                    q=mod1(i+C.offsm[p,m],N)
                    A[gi,row_range[q]] += coeff*C.wtm[p,m]
                end
            end
        end
    end
    return A
end

function _assemble_self_alpert_smooth_panel!(solver::CFIE_alpert{T},A::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},G::CFIEGeomCache{T},C::AlpertSmoothPanelCache{T},row_range::UnitRange{Int},k::T,rule::AlpertLogRule{T};multithreaded::Bool=true) where {T<:Real}
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
        ui=C.us[i]
        A[gi,gi] += one(Complex{T})
        @inbounds for j in 1:N
            j==i && continue
            gj=row_range[j]
            rij=G.R[i,j]
            inn=G.inner[i,j]
            invr=G.invR[i,j]
            A[gi,gj] -= h*(αD*inn*H(1,k*rij)*invr)
        end
        @inbounds for j in 1:N
            j==i && continue
            abs(j-i) < a || continue
            gj=row_range[j]
            rij=G.R[i,j]
            inn=G.inner[i,j]
            invr=G.invR[i,j]
            A[gi,gj] += h*(αD*inn*H(1,k*rij)*invr)
        end
        @inbounds for p in 1:jcorr
            fac=h*rule.w[p]
            Δu=h*rule.x[p]
            if ui+Δu<one(T)
                dx=xi-C.xp[p,i]
                dy=yi-C.yp[p,i]
                r=sqrt(dx*dx+dy*dy)
                if isfinite(r) && r>sqrt(eps(T))
                    inn=_dinner(dx,dy,C.txp[p,i],C.typ[p,i])
                    coeff= -(fac*(αD*inn*H(1,k*r)/r))
                    for m in axes(C.idxp,3)
                        q=C.idxp[p,i,m]
                        A[gi,row_range[q]] += coeff*C.wtp[p,i,m]
                    end
                end
            end
            if ui - Δu > zero(T)
                dx=xi-C.xm[p,i]
                dy=yi-C.ym[p,i]
                r=sqrt(dx*dx+dy*dy)
                if isfinite(r) && r>sqrt(eps(T))
                    inn=_dinner(dx,dy,C.txm[p,i],C.tym[p,i])
                    coeff= -(fac*(αD*inn*H(1,k*r)/r))
                    for m in axes(C.idxm,3)
                        q=C.idxm[p,i,m]
                        A[gi,row_range[q]] += coeff*C.wtm[p,i,m]
                    end
                end
            end
        end
        @inbounds for j in 1:N
            j==i && continue
            gj=row_range[j]
            abs(j-i)<a && continue
            A[gi,gj] -= ik*(h*(αS*H(0,k*G.R[i,j])*G.speed[j]))
        end
        @inbounds for p in 1:jcorr
            fac=h*rule.w[p]
            Δu=h*rule.x[p]
            if ui+Δu<one(T)
                dx=xi-C.xp[p,i]
                dy=yi-C.yp[p,i]
                r=sqrt(dx*dx+dy*dy)
                if isfinite(r) && r>sqrt(eps(T))
                    coeff= -ik*(fac*(αS*H(0,k*r)*C.sp[p,i]))
                    for m in axes(C.idxp,3)
                        q=C.idxp[p,i,m]
                        A[gi,row_range[q]] += coeff*C.wtp[p,i,m]
                    end
                end
            end
            if ui-Δu>zero(T)
                dx=xi-C.xm[p,i]
                dy=yi-C.ym[p,i]
                r=sqrt(dx*dx+dy*dy)
                if isfinite(r) && r>sqrt(eps(T))
                    coeff= -ik*(fac*(αS*H(0,k*r)*C.sm[p,i]))
                    for m in axes(C.idxm,3)
                        q=C.idxm[p,i,m]
                        A[gi,row_range[q]] += coeff*C.wtm[p,i,m]
                    end
                end
            end
        end
    end
    return A
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

@inline function _right_neighbor_excluded_count(i::Int,N::Int,a::Int)
    max(0,i+a-1-N)
end

@inline function _left_neighbor_excluded_count(i::Int,a::Int)
    max(0,a-i)
end

@inline function _eval_on_open_panel_localp(pts::BoundaryPointsCFIE{T},u::T,p::Int) where {T<:Real}
    X,Y,dX,dY=_panel_xy_tangent_arrays(pts)
    h=pts.ws[1]
    _eval_shifted_source_smooth_panel_localp(u,h,X,Y,dX,dY,p)
end

function _assemble_self_alpert_composite_component!(solver::CFIE_alpert{T},A::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},Gs::Vector{CFIEGeomCache{T}},Cs,offs::Vector{Int},k::T,rule::AlpertLogRule{T},topo::AlpertCompositeTopology{T},gmap::Vector{Int};multithreaded::Bool=true) where {T<:Real}
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
        ui_list=Ca.us
        left_smooth=topo.left_kind[l] === :smooth
        right_smooth=topo.right_kind[l] === :smooth
        lprev=topo.prev[l]
        lnext=topo.next[l]
        prev_idx=lprev == 0 ? 0 : gmap[lprev]
        next_idx=lnext == 0 ? 0 : gmap[lnext]
        prev_pts=prev_idx == 0 ? nothing : pts[prev_idx]
        next_pts=next_idx == 0 ? nothing : pts[next_idx]
        prev_ra=prev_idx == 0 ? (1:0) : (offs[prev_idx]:(offs[prev_idx+1]-1))
        next_ra=next_idx == 0 ? (1:0) : (offs[next_idx]:(offs[next_idx+1]-1))
        @use_threads multithreading=multithreaded for i in 1:Na
            gi=ra[i]
            xi=Xa[i]
            yi=Ya[i]
            ui=ui_list[i]
            A[gi,gi] += one(Complex{T})
            for m in eachindex(gmap)
                bidx=gmap[m]
                pb=pts[bidx]
                rb=offs[bidx]:(offs[bidx+1]-1)
                Xb=getindex.(pb.xy,1)
                Yb=getindex.(pb.xy,2)
                dXb=getindex.(pb.tangent,1)
                dYb=getindex.(pb.tangent,2)
                sb=@. sqrt(dXb^2 + dYb^2)
                Nb=length(pb.xy)
                for j in 1:Nb
                    gj=rb[j]
                    dx=xi-Xb[j]
                    dy=yi-Yb[j]
                    r2=muladd(dx,dx,dy*dy)
                    same_self=(bidx==aidx && j==i)
                    skip_near=false
                    if bidx==aidx
                        skip_near= abs(j-i)<a
                    elseif right_smooth && bidx==next_idx
                        nr=_right_neighbor_excluded_count(i,Na,a)
                        skip_near= j<=nr
                    elseif left_smooth && bidx==prev_idx
                        nl=_left_neighbor_excluded_count(i,a)
                        skip_near = j>Nb-nl
                    end
                    if !same_self && !skip_near && r2>(eps(T))^2
                        r=sqrt(r2)
                        invr=inv(r)
                        inn=_dinner(dx,dy,dXb[j],dYb[j])
                        A[gi,gj] -= pb.ws[j]*(αD*inn*H(1,k*r)*invr)
                    end
                    if !same_self && !skip_near && r2>(eps(T))^2
                        r=sqrt(r2)
                        A[gi,gj] -= ik*(pb.ws[j]*(αS*H(0,k*r)*sb[j]))
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
                        inn=_dinner(dx,dy,Ca.txp[p,i],Ca.typ[p,i])
                        coeffD= -(fac*(αD*inn*H(1,k*r)/r))
                        coeffS= -ik*(fac*(αS*H(0,k*r)*Ca.sp[p,i]))
                        for m in axes(Ca.idxp,3)
                            q=Ca.idxp[p,i,m]
                            w=Ca.wtp[p,i,m]
                            A[gi,ra[q]] += coeffD*w
                            A[gi,ra[q]] += coeffS*w
                        end
                    end
                end
                if ui-Δu>zero(T)
                    dx=xi-Ca.xm[p,i]
                    dy=yi-Ca.ym[p,i]
                    r=sqrt(dx*dx+dy*dy)
                    if isfinite(r) && r>sqrt(eps(T))
                        inn=_dinner(dx,dy,Ca.txm[p,i],Ca.tym[p,i])
                        coeffD= -(fac*(αD*inn*H(1,k*r)/r))
                        coeffS= -ik*(fac*(αS*H(0,k*r)*Ca.sm[p,i]))
                        for m in axes(Ca.idxm,3)
                            q=Ca.idxm[p,i,m]
                            w=Ca.wtm[p,i,m]
                            A[gi,ra[q]] += coeffD*w
                            A[gi,ra[q]] += coeffS*w
                        end
                    end
                end
            end
            pinterp=size(Ca.idxp,3)
            if right_smooth && next_idx!=0
                for p in 1:jcorr
                    Δu=ha*rule.x[p]
                    if ui+Δu>=one(T)
                        u2=ui+Δu-one(T)
                        x,y,tx,ty,s2,idx2,wt2=_eval_on_open_panel_localp(next_pts,u2,pinterp)
                        dx=xi-x
                        dy=yi-y
                        r=sqrt(dx*dx+dy*dy)
                        if isfinite(r) && r>sqrt(eps(T))
                            fac=ha*rule.w[p]
                            inn=_dinner(dx,dy,tx,ty)
                            coeffD= -(fac*(αD*inn*H(1,k*r)/r))
                            coeffS= -ik*(fac*(αS*H(0,k*r)*s2))
                            _scatter_localp!(A,gi,next_ra,coeffD,idx2,wt2)
                            _scatter_localp!(A,gi,next_ra,coeffS,idx2,wt2)
                        end
                    end
                end
            end
            if left_smooth && prev_idx != 0
                for p in 1:jcorr
                    Δu=ha*rule.x[p]
                    if ui-Δu<=zero(T)
                        u2=one(T)+ui-Δu
                        x,y,tx,ty,s2,idx2,wt2=_eval_on_open_panel_localp(prev_pts,u2,pinterp)
                        dx=xi-x
                        dy=yi-y
                        r=sqrt(dx*dx+dy*dy)
                        if isfinite(r) && r>sqrt(eps(T))
                            fac=ha*rule.w[p]
                            inn=_dinner(dx,dy,tx,ty)
                            coeffD= -(fac*(αD*inn*H(1,k*r)/r))
                            coeffS= -ik*(fac*(αS*H(0,k*r)*s2))
                            _scatter_localp!(A,gi,prev_ra,coeffD,idx2,wt2)
                            _scatter_localp!(A,gi,prev_ra,coeffS,idx2,wt2)
                        end
                    end
                end
            end
        end
    end
    return A
end


function _assemble_self_alpert!(solver::CFIE_alpert{T},A::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},G::CFIEGeomCache{T},C,row_range::UnitRange{Int},k::T,rule::AlpertLogRule{T};multithreaded::Bool=true) where {T<:Real}
    pts.is_periodic ?
        _assemble_self_alpert_periodic!(A,pts,G,C,row_range,k,rule;multithreaded=multithreaded) :
        _assemble_self_alpert_smooth_panel!(solver,A,pts,G,C,row_range,k,rule;multithreaded=multithreaded)
end

function _assemble_all_self_alpert_composite!(solver::CFIE_alpert{T},A::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},Gs::Vector{CFIEGeomCache{T}},Cs,offs::Vector{Int},k::T,rule::AlpertLogRule{T},topos::Vector{AlpertCompositeTopology{T}},gmaps::Vector{Vector{Int}};multithreaded::Bool=true) where {T<:Real}
    @inbounds for c in eachindex(gmaps)
        gmap=gmaps[c]
        if length(gmap)==1 && pts[gmap[1]].is_periodic
            a=gmap[1]
            ra=offs[a]:(offs[a+1]-1)
            _assemble_self_alpert!(solver,A,pts[a],Gs[a],Cs[a],ra,k,rule;multithreaded=multithreaded)
        else
            _assemble_self_alpert_composite_component!(solver,A,pts,Gs,Cs,offs,k,rule,topos[c],gmap;multithreaded=multithreaded)
        end
    end
    return A
end

struct CFIEAlpertWorkspace{T<:Real,C}
    rule::AlpertLogRule{T}
    offs::Vector{Int}
    Gs::Vector{CFIEGeomCache{T}}
    Cs::Vector{C}
    topos::Union{Nothing,Vector{AlpertCompositeTopology{T}}}
    gmaps::Union{Nothing,Vector{Vector{Int}}}
    panel_to_comp::Union{Nothing,Vector{Int}}
    Ntot::Int
end

function build_cfie_alpert_workspace(solver::CFIE_alpert{T},pts::Vector{BoundaryPointsCFIE{T}}) where {T<:Real}
    rule=alpert_log_rule(T,solver.alpert_order)
    offs=component_offsets(pts)
    Gs=[cfie_geom_cache(p) for p in pts]
    boundary= isnothing(solver.symmetry) ? solver.billiard.full_boundary : solver.billiard.desymmetrized_full_boundary
    flat_boundary= boundary[1] isa AbstractVector ? reduce(vcat,boundary) : boundary
    Cs=[_build_alpert_component_cache(solver,flat_boundary[a],pts[a],rule,solver.alpert_order) for a in eachindex(pts)]
    topo_data=build_join_topology(pts)
    if isnothing(topo_data)
        topos=nothing
        gmaps=nothing
        panel_to_comp=nothing
    else
        topos,gmaps=topo_data
        panel_to_comp=zeros(Int,length(pts))
        @inbounds for c in eachindex(gmaps), a in gmaps[c]
            panel_to_comp[a]=c
        end
    end
    Ntot=offs[end]-1
    CFIEAlpertWorkspace(rule,offs,Gs,Cs,topos,gmaps,panel_to_comp,Ntot)
end

@inline function _check_r(r,name,i,j)
    if !(isfinite(r)) || r<=sqrt(eps(eltype(r)))
        @warn "Bad distance in $name at i=$i j=$j : r=$r"
    end
end

@inline dlp_weight(pts::BoundaryPointsCFIE,j::Int)=pts.ws[j]
@inline function slp_weight(pts::BoundaryPointsCFIE{T},j::Int,sj::T) where {T<:Real}
    pts.ws[j]*sj
end

function _add_image_block!(A::AbstractMatrix{Complex{T}},ra::UnitRange{Int},rb::UnitRange{Int},pa::BoundaryPointsCFIE{T},pb::BoundaryPointsCFIE{T},k::T,qfun,tfun,weight;multithreaded::Bool=true) where {T<:Real}
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
            _check_r(r,"image-block",i,j)
            invr=inv(r)
            inn=_dinner(dx,dy,txj,tyj)
            dval=weight*wd*(αD*inn*H(1,k*r)*invr)
            sval=weight*ws*(αS*H(0,k*r))
            A[gi,gj] -= dval+ik*sval
        end
    end
    return A
end

function _assemble_reflection_images!(A::AbstractMatrix{Complex{T}},ra::UnitRange{Int},rb::UnitRange{Int},pa::BoundaryPointsCFIE{T},pb::BoundaryPointsCFIE{T},solver::CFIE_alpert{T},billiard::Bi,k::T,sym::Reflection;multithreaded::Bool=true) where {T<:Real,Bi<:AbsBilliard}
    if sym.axis==:y_axis
        _add_image_block!(A,ra,rb,pa,pb,k,q->image_point_x(q,billiard),t->image_tangent_x(t),image_weight(sym);multithreaded=multithreaded)
    elseif sym.axis==:x_axis
        _add_image_block!(A,ra,rb,pa,pb,k,q->image_point_y(q,billiard),t->image_tangent_y(t),image_weight(sym);multithreaded=multithreaded)
    elseif sym.axis==:origin
        σx=image_weight_x(sym)
        σy=image_weight_y(sym)
        σxy=image_weight_xy(sym)
        _add_image_block!(A,ra,rb,pa,pb,k,q->image_point_x(q,billiard),t->image_tangent_x(t),σx;multithreaded=multithreaded)
        _add_image_block!(A,ra,rb,pa,pb,k,q->image_point_y(q,billiard),t->image_tangent_y(t),σy;multithreaded=multithreaded)
        _add_image_block!(A,ra,rb,pa,pb,k,q->image_point_xy(q,billiard),t->image_tangent_xy(t),σxy;multithreaded=multithreaded)
    else
        error("Unknown reflection axis $(sym.axis)")
    end
    return A
end

function _assemble_rotation_images!(A::AbstractMatrix{Complex{T}},ra::UnitRange{Int},rb::UnitRange{Int},pa::BoundaryPointsCFIE{T},pb::BoundaryPointsCFIE{T},k::T,sym::Rotation,costab,sintab,χ;multithreaded::Bool=true) where {T<:Real}
    for l in 1:(sym.n-1)
        phase=χ[l+1]
        _add_image_block!(A,ra,rb,pa,pb,k,q->image_point(sym,q,l,costab,sintab),t->image_tangent(sym,t,l,costab,sintab),phase;multithreaded=multithreaded)
    end
    return A
end

function construct_matrices_symmetry!(solver::CFIE_alpert{T},A::Matrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},k::T;multithreaded::Bool=true) where {T<:Real}
    symmetry=solver.symmetry
    isnothing(symmetry) && error("construct_matrices_symmetry! called with symmetry = nothing")
    offs=component_offsets(pts)
    fill!(A,zero(Complex{T}))
    rule=alpert_log_rule(T,solver.alpert_order)
    Gs=[cfie_geom_cache(p) for p in pts]
    boundary=solver.billiard.desymmetrized_full_boundary
    flat_boundary=boundary[1] isa AbstractVector ? reduce(vcat,boundary) : boundary
    Cs=[_build_alpert_component_cache(solver,flat_boundary[a],pts[a],rule,solver.alpert_order) for a in eachindex(pts)]
    nc=length(pts)
    topo_data=build_join_topology(pts)
    gmaps= isnothing(topo_data) ? nothing : topo_data[2]
    if isnothing(topo_data)
        for a in 1:nc
            ra=offs[a]:(offs[a+1]-1)
            _assemble_self_alpert!(solver,A,pts[a],Gs[a],Cs[a],ra,k,rule;multithreaded=multithreaded)
        end
    else
        topos,gmaps=topo_data
        _assemble_all_self_alpert_composite!(solver,A,pts,Gs,Cs,offs,k,rule,topos,gmaps;multithreaded=multithreaded)
    end
    αD=Complex{T}(0,k/2)
    αS=Complex{T}(0,one(T)/2)
    ik=Complex{T}(0,k)
    for a in 1:nc, b in 1:nc
        a==b && continue
        if gmaps !== nothing
            ca=_component_id_of_panel(a,gmaps)
            cb=_component_id_of_panel(b,gmaps)
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
            gj=rb[j]
            xj=Xb[j]
            yj=Yb[j]
            txj=dXb[j]
            tyj=dYb[j]
            sj=sb[j]
            wd=dlp_weight(pb,j)
            ws=slp_weight(pb,j,sj)
            @inbounds for i in 1:Na
                gi=ra[i]
                dx=Xa[i]-xj
                dy=Ya[i]-yj
                r2=muladd(dx,dx,dy*dy)
                r2<=(eps(T))^2 && continue
                r=sqrt(r2)
                _check_r(r,"symmetry before images",i,j)
                invr=inv(r)
                inn=_dinner(dx,dy,txj,tyj)
                dval=wd*(αD*inn*H(1,k*r)*invr)
                sval=ws*(αS*H(0,k*r))
                A[gi,gj] -= dval+ik*sval
            end
        end
    end
    if symmetry isa Reflection
        for a in 1:nc, b in 1:nc
            ra=offs[a]:(offs[a+1]-1)
            rb=offs[b]:(offs[b+1]-1)
            _assemble_reflection_images!(A,ra,rb,pts[a],pts[b],solver,solver.billiard,k,symmetry;multithreaded=multithreaded)
        end
    elseif symmetry isa Rotation
        costab,sintab,χ=_rotation_tables(T,symmetry.n,symmetry.m)
        for a in 1:nc, b in 1:nc
            ra=offs[a]:(offs[a+1]-1)
            rb=offs[b]:(offs[b+1]-1)
            _assemble_rotation_images!(A,ra,rb,pts[a],pts[b],k,symmetry,costab,sintab,χ;multithreaded=multithreaded)
        end
    else
        error("Unknown symmetry type $(typeof(symmetry))")
    end
    return A
end

function construct_matrices_symmetry!(solver::CFIE_alpert{T},A::Matrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},ws::CFIEAlpertWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}
    symmetry=solver.symmetry
    isnothing(symmetry) && error("construct_matrices_symmetry! called with symmetry = nothing")
    fill!(A,zero(Complex{T}))
    offs=ws.offs
    Gs=ws.Gs
    Cs=ws.Cs
    rule=ws.rule
    topos=ws.topos
    gmaps=ws.gmaps
    panel_to_comp=ws.panel_to_comp
    nc=length(pts)
    if isnothing(topos)
        @inbounds for a in 1:nc
            ra=offs[a]:(offs[a+1]-1)
            _assemble_self_alpert!(solver,A,pts[a],Gs[a],Cs[a],ra,k,rule;multithreaded=multithreaded)
        end
    else
        _assemble_all_self_alpert_composite!(solver,A,pts,Gs,Cs,offs,k,rule,topos,gmaps;multithreaded=multithreaded)
    end
    αD=Complex{T}(0,k/2)
    αS=Complex{T}(0,one(T)/2)
    ik=Complex{T}(0,k)
    for a in 1:nc, b in 1:nc
        a==b && continue
        if !isnothing(panel_to_comp)
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
                inn=_dinner(dx,dy,txj,tyj)
                dval= wd*(αD*inn*H(1,k*r)*invr)
                sval= wsj*(αS*H(0,k*r))
                A[gi,gj] -= dval+ik*sval
            end
        end
    end
    if symmetry isa Reflection
        for a in 1:nc, b in 1:nc
            ra=offs[a]:(offs[a+1]-1)
            rb=offs[b]:(offs[b+1]-1)
            _assemble_reflection_images!(A,ra,rb,pts[a],pts[b],solver,solver.billiard,k,symmetry;multithreaded=multithreaded)
        end
    elseif symmetry isa Rotation
        costab,sintab,χ=_rotation_tables(T,symmetry.n,symmetry.m)
        for a in 1:nc, b in 1:nc
            ra=offs[a]:(offs[a+1]-1)
            rb=offs[b]:(offs[b+1]-1)
            _assemble_rotation_images!(A,ra,rb,pts[a],pts[b],k,symmetry,costab,sintab,χ;multithreaded=multithreaded)
        end
    else
        error("Unknown symmetry type $(typeof(symmetry))")
    end
    return A
end

function construct_matrices!(solver::CFIE_alpert{T},A::Matrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},k::T;multithreaded::Bool=true) where {T<:Real}
    if isnothing(solver.symmetry)
        offs=component_offsets(pts)
        αD=Complex{T}(0,k/2)
        αS=Complex{T}(0,one(T)/2)
        ik=Complex{T}(0,k)
        fill!(A,zero(Complex{T}))
        Gs=[cfie_geom_cache(p) for p in pts]
        rule=alpert_log_rule(T,solver.alpert_order)
        boundary=solver.billiard.full_boundary
        flat_boundary= boundary[1] isa AbstractVector ? reduce(vcat,boundary) : boundary
        Cs=[_build_alpert_component_cache(solver,flat_boundary[a],pts[a],rule,solver.alpert_order) for a in eachindex(pts)]
        nc=length(pts)
        topo_data=build_join_topology(pts)
        gmaps= isnothing(topo_data) ? nothing : topo_data[2]
        if isnothing(topo_data)
            for a in 1:nc
                ra=offs[a]:(offs[a+1]-1)
                _assemble_self_alpert!(solver,A,pts[a],Gs[a],Cs[a],ra,k,rule;multithreaded=multithreaded)
            end
        else
            topos,gmaps=topo_data
            _assemble_all_self_alpert_composite!(solver,A,pts,Gs,Cs,offs,k,rule,topos,gmaps;multithreaded=multithreaded)
        end
        for a in 1:nc, b in 1:nc
            a==b && continue
            if !isnothing(gmaps)
                ca=_component_id_of_panel(a,gmaps)
                cb=_component_id_of_panel(b,gmaps)
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
            sb=@. sqrt(dXb^2 + dYb^2)
            @use_threads multithreading=multithreaded for j in 1:Nb
                gj=rb[j]
                xj=Xb[j]
                yj=Yb[j]
                txj=dXb[j]
                tyj=dYb[j]
                sj=sb[j]
                wd=dlp_weight(pb,j)
                ws=slp_weight(pb,j,sj)
                @inbounds for i in 1:Na
                    gi=ra[i]
                    dx=Xa[i]-xj
                    dy=Ya[i]-yj
                    r2=muladd(dx,dx,dy*dy)
                    r2<=(eps(T))^2 && continue
                    r=sqrt(r2)
                    invr=inv(r)
                    inn=_dinner(dx,dy,txj,tyj)
                    dval= wd*(αD*inn*H(1,k*r)*invr)
                    sval= ws*(αS*H(0,k*r))
                    A[gi,gj] -= dval+ik*sval
                end
            end
        end
        return A
    else
        construct_matrices_symmetry!(solver,A,pts,k;multithreaded=multithreaded)
    end
end

function construct_matrices!(solver::CFIE_alpert{T},A::Matrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},ws::CFIEAlpertWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}
    if isnothing(solver.symmetry)
        fill!(A,zero(Complex{T}))
        offs=ws.offs
        Gs=ws.Gs
        Cs=ws.Cs
        rule=ws.rule
        topos=ws.topos
        gmaps=ws.gmaps
        panel_to_comp=ws.panel_to_comp
        αD=Complex{T}(0,k/2)
        αS=Complex{T}(0,one(T)/2)
        ik=Complex{T}(0,k)
        if isnothing(topos)
            @inbounds for a in eachindex(pts)
                ra=offs[a]:(offs[a+1]-1)
                _assemble_self_alpert!(solver,A,pts[a],Gs[a],Cs[a],ra,k,rule;multithreaded=multithreaded)
            end
        else
            _assemble_all_self_alpert_composite!(solver,A,pts,Gs,Cs,offs,k,rule,topos,gmaps;multithreaded=multithreaded)
        end
        for a in eachindex(pts), b in eachindex(pts)
            a==b && continue
            if panel_to_comp !== nothing
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
                    invr=inv(r)
                    inn=_dinner(dx,dy,txj,tyj)
                    dval= wd*(αD*inn*H(1,k*r)*invr)
                    sval= wsj*(αS*H(0,k*r))
                    A[gi,gj] -= dval+ik*sval
                end
            end
        end
        return A
    else
        construct_matrices_symmetry!(solver,A,pts,ws,k;multithreaded=multithreaded)
    end
end

"""
    construct_matrices(solver::CFIE_alpert,pts::Vector{BoundaryPointsCFIE{T}},k::T;multithreaded::Bool=true) where {T<:Real}

High-level wrapper to construct the CFIE Alpert system matrix. This function checks for symmetry and dispatches to the appropriate assembly routine. It is mostly legacy since it is better to precompute the workspace.

# Inputs:
- `solver::CFIE_alpert{T}` : The CFIE_alpert solver object containing parameters and symmetry information.
- `pts::Vector{BoundaryPointsCFIE{T}}` : The boundary points for the entire geometry.
- `k::T` : The real wavenumber.
- `multithreaded::Bool=true` : Whether to use multithreading for assembly.

# Outputs:
- `A::Matrix{Complex{T}}` containing the assembled system matrix for the CFIE Alpert solver.
"""
function construct_matrices(solver::CFIE_alpert,pts::Vector{BoundaryPointsCFIE{T}},k::T;multithreaded::Bool=true) where {T<:Real}
    Ntot=boundary_matrix_size(pts)
    A=Matrix{Complex{T}}(undef,Ntot,Ntot)
    @blas_1 construct_matrices!(solver,A,pts,k;multithreaded=multithreaded)
    return A
end

"""
    construct_matrices(solver::CFIE_alpert,pts::Vector{BoundaryPointsCFIE{T}},ws::CFIEAlpertWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}

High-level wrapper to construct the CFIE Alpert system matrix using a precomputed workspace. This is the recommended interface for efficient assembly, as it avoids redundant precomputation of geometry caches and Alpert component caches.

# Inputs:
- `solver::CFIE_alpert{T}` : The CFIE_alpert solver object containing parameters and symmetry information.
- `pts::Vector{BoundaryPointsCFIE{T}}` : The boundary points for the entire geometry.
- `ws::CFIEAlpertWorkspace{T}` : The precomputed workspace containing all necessary data for assembly.
- `k::T` : The real wavenumber.
- `multithreaded::Bool=true` : Whether to use multithreading for assembly. Whether to use multithreading for assembly.
# Outputs:
- `A::Matrix{Complex{T}}` containing the assembled system matrix for the CFIE Alpert solver.
"""
function construct_matrices(solver::CFIE_alpert,pts::Vector{BoundaryPointsCFIE{T}},ws::CFIEAlpertWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}
    A=Matrix{Complex{T}}(undef,ws.Ntot,ws.Ntot)
    construct_matrices!(solver,A,pts,ws,k;multithreaded=multithreaded)
    return A
end

"""
    solve(solver::CFIE_alpert,basis::Ba,pts::Vector{BoundaryPointsCFIE{T}},ws::CFIEAlpertWorkspace{T},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det_argmin) where {T<:Real,Ba<:AbsBasis}

# High-level function to solve the CFIE eigenvalue problem using the Alpert-based discretization. This function constructs the system matrix and then computes the smallest singular value, which corresponds to the eigenvalue of interest.

# Inputs:
- `solver::CFIE_alpert{T}` : The CFIE_alpert solver object containing parameters and symmetry information.
- `basis::Ba` : The basis object (not used in this implementation but included for API consistency).
- `pts::Vector{BoundaryPointsCFIE{T}}` : The boundary points for the entire geometry.
- `ws::CFIEAlpertWorkspace{T}` (optional) : The precomputed workspace containing all necessary data for assembly. If not provided, the system matrix will be constructed without using a workspace, which may be less efficient.
- `k::T` : The real wavenumber.
- `multithreaded::Bool=true` : Whether to use multithreading for assembly.
- `use_krylov::Bool=true` : Whether to use a Krylov method (svdsolve) to compute the smallest singular value, which can be more efficient for large systems. If false, it will compute the full SVD and return the smallest singular value, which is more expensive. 
- `which::Symbol=:det_argmin` : Which method to use for computing the eigenvalue. Options include `:det`, `:svd`, and `:det_argmin`. Note that the Krylov method does not support determinant calculation and will fall back to SVD if `:det` is selected.

# Outputs:
- The smallest singular value of the system matrix, which corresponds to the eigenvalue of interest for the CFIE problem.
"""
function solve(solver::CFIE_alpert,basis::Ba,pts::Vector{BoundaryPointsCFIE{T}},ws::CFIEAlpertWorkspace{T},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det_argmin) where {T<:Real,Ba<:AbsBasis}
    A=Matrix{Complex{T}}(undef,ws.Ntot,ws.Ntot)
    @blas_1 construct_matrices!(solver,A,pts,ws,k;multithreaded=multithreaded)
    @svd_or_det_solve A use_krylov which MAX_BLAS_THREADS
end

"""
    solve_vect(solver::CFIE_alpert,basis::Ba,pts::Vector{BoundaryPointsCFIE{T}},ws::CFIEAlpertWorkspace{T},k;multithreaded::Bool=true) where {T<:Real,Ba<:AbsBasis}

High-level function to solve the CFIE eigenvalue problem and return both the smallest singular value and the corresponding right singular vector (eigenfunction). This function constructs the system matrix and then computes the SVD to extract the smallest singular value and its associated singular vector.

# Inputs:
- `solver::CFIE_alpert{T} : The CFIE_alpert solver object containing parameters and symmetry information.
- `basis::Ba` : The basis object (not used in this implementation but included for API consistency).
- `pts::Vector{BoundaryPointsCFIE{T}}` : The boundary points for the entire geometry.
- `ws::CFIEAlpertWorkspace{T}` (optional) : The precomputed workspace containing all necessary data for assembly. If not provided, the system matrix will be constructed without using a workspace, which may be less efficient.
- `k::T` : The real wavenumber.
- `multithreaded::Bool=true` : Whether to use multithreading for assembly. 

# Outputs:
 - A tuple containing the smallest singular value and the corresponding right singular vector (eigenfunction) of the system matrix.
"""
function solve_vect(solver::CFIE_alpert,basis::Ba,pts::Vector{BoundaryPointsCFIE{T}},ws::CFIEAlpertWorkspace{T},k;multithreaded::Bool=true) where {T<:Real,Ba<:AbsBasis}
    A=Matrix{Complex{T}}(undef,ws.Ntot,ws.Ntot)
    @blas_1 construct_matrices!(solver,A,pts,ws,k;multithreaded=multithreaded)
    @blas_multi_then_1 MAX_BLAS_THREADS _,S,Vt=LAPACK.gesvd!('A','A',A)
    idx=findmin(S)[2]
    return S[idx],conj.(Vt[idx,:])
end

"""
    solve_INFO(solver::CFIE_alpert,basis::Ba,pts::Vector{BoundaryPointsCFIE{T}},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det) where {T<:Real,Ba<:AbsBasis}

High-level function to solve the CFIE eigenvalue problem while also providing detailed timing and condition number information. This function constructs the system matrix, computes its condition number, performs the SVD, and then reports the time taken for each step as well as the condition number of the matrix. 

# Inputs:
- `solver::CFIE_alpert{T}` : The CFIE_alpert solver object containing parameters and symmetry information.
- `basis::Ba` : The basis object (not used in this implementation but included for API consistency).
- `pts::Vector{BoundaryPointsCFIE{T}}` : The boundary points for the entire geometry.
- `k::T` : The real wavenumber.
- `multithreaded::Bool=true` : Whether to use multithreading for assembly.
- `use_krylov::Bool=true` : Whether to use a Krylov method (svdsolve) to compute the smallest singular value, which can be more efficient for large systems. If false, it will compute the full SVD and return the smallest singular value, which is more expensive.
- `which::Symbol=:det_argmin` : Which SVD method to use if `use_krylov` is false. This is passed to the `@svd_or_det_solve` macro to determine the SVD computation method. Options include `:det`, `:svd`, and `:det_argmin`.   

# Outputs:
- The smallest singular value of the system matrix, which corresponds to the eigenvalue of interest for the CFIE problem, along with printed information about the condition number of the matrix and the time taken.
"""
function solve_INFO(solver::CFIE_alpert,basis::Ba,pts::Vector{BoundaryPointsCFIE{T}},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det_argmin) where {T<:Real,Ba<:AbsBasis}
    offs=component_offsets(pts)
    Ntot=offs[end]-1
    A=Matrix{Complex{T}}(undef,Ntot,Ntot)
    t0=time()
    @info "Building boundary operator A..."
    @blas_1 construct_matrices!(solver,A,pts,k;multithreaded=multithreaded)
    any(isnan.(A)) && error("NaN detected in system matrix A; check geometry and quadrature.")
    t1=time()
    cA=cond(A)
    @info "Condition number of A: $(round(cA;sigdigits=4))"
    t2=time()
    @svd_or_det_solve A use_krylov which MAX_BLAS_THREADS
    t3=time()
    build_A=t1-t0
    svd_time=t3-t2
    total=build_A+svd_time
    println("────────── SOLVE_INFO SUMMARY ──────────")
    println("A-matrix build: ",100*build_A/total," %")
    println("SVD: ",100*svd_time/total," %")
    println("(total: ",total," s)")
    println("────────────────────────────────────────")
    return s[end]
end

################
#### LEGACY ####
################

function solve(solver::CFIE_alpert,basis::Ba,pts::Vector{BoundaryPointsCFIE{T}},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det_argmin) where {T<:Real,Ba<:AbsBasis}
    @blas_1 A=construct_matrices(solver,pts,k;multithreaded=multithreaded)
    @svd_or_det_solve A use_krylov which MAX_BLAS_THREADS
end

function solve_vect(solver::CFIE_alpert,basis::Ba,pts::Vector{BoundaryPointsCFIE{T}},k;multithreaded::Bool=true) where {T<:Real,Ba<:AbsBasis}
    @blas_1 A=construct_matrices(solver,pts,k;multithreaded=multithreaded)
    @blas_multi_then_1 MAX_BLAS_THREADS _,S,Vt=LAPACK.gesvd!('A','A',A)
    idx=findmin(S)[2]
    return S[idx],conj.(Vt[idx,:])
end