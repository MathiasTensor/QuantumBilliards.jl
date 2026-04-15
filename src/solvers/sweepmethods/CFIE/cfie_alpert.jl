struct CFIEPanelArrays{T<:Real}
    X::Vector{T}
    Y::Vector{T}
    dX::Vector{T}
    dY::Vector{T}
    s::Vector{T}
end

@inline function _panel_arrays_cache(pts::BoundaryPointsCFIE{T}) where {T<:Real}
    X=getindex.(pts.xy,1)
    Y=getindex.(pts.xy,2)
    dX=getindex.(pts.tangent,1)
    dY=getindex.(pts.tangent,2)
    s=@. sqrt(dX^2+dY^2)
    return CFIEPanelArrays(X,Y,dX,dY,s)
end

struct CFIEAlpertWorkspace{T<:Real,C}
    rule::AlpertLogRule{T}
    offs::Vector{Int}
    Gs::Vector{CFIEGeomCache{T}}
    Cs::Vector{C}
    parr::Vector{CFIEPanelArrays{T}}
    Ntot::Int
end

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
    crv::Any
    sig::Vector{T}
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
    wtp::Array{Complex{T},3}
    idxm::Array{Int,3}
    wtm::Array{Complex{T},3}
end

@inline function _dlp_terms(TT,k,r,inn,invr,w)
    h0,h1=hankel_pair01(k*r)
    αD=Complex{TT}(0,k/2)
    d0=w*(αD*inn*h1*invr)
    d1=w*((Complex{TT}(0,1)/2)*inn*k*h0)
    d2=w*((Complex{TT}(0,1)/2)*inn*(h0-k*r*h1))
    return d0,d1,d2,h0,h1
end

@inline function _slp_terms(TT,k,r,s,w,h0,h1)
    αS=Complex{TT}(0,one(TT)/2)
    s0=w*(αS*h0*s)
    s1=w*(-(Complex{TT}(0,1)/2)*r*h1*s)
    s2=w*((Complex{TT}(0,1)/2)*r*(h1-k*r*h0)*s/k)
    return s0,s1,s2
end

@inline function _wrap01(u::T) where {T<:Real}
    v=mod(u,one(T))
    v<zero(T) ? v+one(T) : v
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
    collect(-(q-1):q)
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

@inline function _speed(v::SVector{2,T}) where {T<:Real}
    s=sqrt(v[1]^2+v[2]^2)
    s<eps(T) ? eps(T) : s
end

@inline function _panel_arrays(pts::BoundaryPointsCFIE{T}) where {T<:Real}
    X=getindex.(pts.xy,1)
    Y=getindex.(pts.xy,2)
    dX=getindex.(pts.tangent,1)
    dY=getindex.(pts.tangent,2)
    s=@. sqrt(dX^2+dY^2)
    return X,Y,dX,dY,s
end

function _add_naive_panel_block!(A::AbstractMatrix{Complex{T}},gi::Int,xi::T,yi::T,rb::UnitRange{Int},pb::BoundaryPointsCFIE{T},Pb::CFIEPanelArrays{T},k::T,αD::Complex{T},αS::Complex{T},ik::Complex{T};skip_pred=(j->false)) where {T<:Real}
    Xb=Pb.X
    Yb=Pb.Y
    dXb=Pb.dX
    dYb=Pb.dY
    sb=Pb.s
    Nb=length(Xb)
    @inbounds for j in 1:Nb
        skip_pred(j) && continue
        gj=rb[j]
        dx=xi-Xb[j]
        dy=yi-Yb[j]
        r2=muladd(dx,dx,dy*dy)
        r2<=(eps(T))^2 && continue
        r=sqrt(r2)
        invr=inv(r)
        inn=_dinner(dx,dy,dXb[j],dYb[j])
        wd=pb.ws[j]
        ws=pb.ws[j]*sb[j]
        dval=wd*(αD*inn*H(1,k*r)*invr)
        sval=ws*(αS*H(0,k*r))
        A[gi,gj]-=dval+ik*sval
    end
    return A
end

function _add_self_panel_alpert_correction!(A::AbstractMatrix{Complex{T}},gi::Int,xi::T,yi::T,i::Int,ra::UnitRange{Int},Ca::AlpertSmoothPanelCache{T},hσ::T,k::T,αD::Complex{T},αS::Complex{T},ik::Complex{T},rule::AlpertLogRule{T}) where {T<:Real}
    jcorr=rule.j
    @inbounds for p in 1:jcorr
        fac=hσ*rule.w[p]
        dx=xi-Ca.xp[p,i]
        dy=yi-Ca.yp[p,i]
        r2=muladd(dx,dx,dy*dy)
        if isfinite(r2) && r2>(eps(T))^2
            r=sqrt(r2)
            inn=_dinner(dx,dy,Ca.txp[p,i],Ca.typ[p,i])
            coeffD=-(fac*(αD*inn*H(1,k*r)/r))
            coeffS=-ik*(fac*(αS*H(0,k*r)*Ca.sp[p,i]))
            for m in axes(Ca.idxp,3)
                q=Ca.idxp[p,i,m]
                w=Ca.wtp[p,i,m]
                A[gi,ra[q]] += coeffD*w+coeffS*w
            end
        end
        dx=xi-Ca.xm[p,i]
        dy=yi-Ca.ym[p,i]
        r2=muladd(dx,dx,dy*dy)
        if isfinite(r2) && r2>(eps(T))^2
            r=sqrt(r2)
            inn=_dinner(dx,dy,Ca.txm[p,i],Ca.tym[p,i])
            coeffD=-(fac*(αD*inn*H(1,k*r)/r))
            coeffS=-ik*(fac*(αS*H(0,k*r)*Ca.sm[p,i]))
            for m in axes(Ca.idxm,3)
                q=Ca.idxm[p,i,m]
                w=Ca.wtm[p,i,m]
                A[gi,ra[q]] += coeffD*w+coeffS*w
            end
        end
    end
    return A
end

@inline function _panel_interp_midpoint_data(σ::T,hσ::T,N::Int,p::Int) where {T<:Real}
    iseven(p) || error("p must be even.")
    p<=N || error("p must satisfy p <= N.")
    q=p÷2
    s=σ/hσ-T(1)/2
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

@inline function _eval_open_panel_geom_exact(crv,u::T) where {T<:Real}
    q=curve(crv,u)
    t=tangent(crv,u)
    s=sqrt(t[1]^2+t[2]^2)
    return q[1],q[2],t[1],t[2],s
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
        op,wp=_alpert_interp_offsets_weights(ξ,ninterp)
        om,wm=_alpert_interp_offsets_weights(-ξ,ninterp)
        for m in 1:ninterp
            offsp[p,m]=op[m]
            wtp[p,m]=wp[m]
            offsm[p,m]=om[m]
            wtm[p,m]=wm[m]
        end
        δu=T(σ)*ξ/T(N)
        for i in 1:N
            ui=pts.ts[i]/T(two_pi)
            up=_wrap01(ui+δu)
            qp=curve(crv,up)
            tp=T(σ)*tangent(crv,up)/T(two_pi)
            xp[p,i]=qp[1]
            yp[p,i]=qp[2]
            txp[p,i]=tp[1]
            typ[p,i]=tp[2]
            sp[p,i]=sqrt(tp[1]^2+tp[2]^2)
            um=_wrap01(ui-δu)
            qm=curve(crv,um)
            tm=T(σ)*tangent(crv,um)/T(two_pi)
            xm[p,i]=qm[1]
            ym[p,i]=qm[2]
            txm[p,i]=tm[1]
            tym[p,i]=tm[2]
            sm[p,i]=sqrt(tm[1]^2+tm[2]^2)
        end
    end
    return AlpertPeriodicCache(xp,yp,txp,typ,sp,xm,ym,txm,tym,sm,offsp,wtp,offsm,wtm,ninterp)
end

@inline _fold_open_reflection_right(::Type{T},σ::T) where {T<:Real}=T(2)-σ
@inline _fold_open_reflection_left(::Type{T},σ::T) where {T<:Real}=-σ

@inline function _reflection_character(::Type{T},sym) where {T<:Real}
    sym isa Reflection || error("Only Reflection supported here.")
    if sym.axis===:x_axis || sym.axis===:y_axis
        return Complex{T}(T(sym.parity),zero(T))
    elseif sym.axis===:origin
        return Complex{T}(T(sym.parity[1]*sym.parity[2]),zero(T))
    else
        error("Unknown reflection axis $(sym.axis)")
    end
end

function _continued_panel_interp_data(solver::CFIE_alpert{T},σ::T,N::Int,p::Int) where {T<:Real}
    if zero(T)<σ<one(T)
        idx,wt=_panel_interp_midpoint_data(σ,inv(T(N)),N,p)
        return idx,Complex{T}.(wt)
    end
    sym=solver.symmetry
    sym isa Reflection || error("Open-panel continuation currently implemented only for reflections.")
    σred=σ>=one(T) ? _fold_open_reflection_right(T,σ) : _fold_open_reflection_left(T,σ)
    zero(T)<σred<one(T) || error("Continuation too far outside panel: σ=$σ mapped to σred=$σred")
    idx,wt=_panel_interp_midpoint_data(σred,inv(T(N)),N,p)
    χ=_reflection_character(T,sym)
    cwt=Complex{T}.(wt)
    cwt .*= χ
    return idx,cwt
end

function _eval_open_panel_geom_continued(solver::CFIE_alpert{T},crv,uσ::T) where {T<:Real}
    σcl=clamp(uσ,zero(T),one(T))
    u,jac,jac2=_panel_sigma_to_u_jac(solver,σcl)
    q=curve(crv,u)
    tu=tangent(crv,u)
    su=sqrt(tu[1]^2+tu[2]^2)
    x=q[1];y=q[2]
    tx=tu[1]*jac;ty=tu[2]*jac
    s=su*jac
    if zero(T)<uσ<one(T)
        return x,y,tx,ty,s
    end
    sym=solver.symmetry
    sym isa Reflection || error("Open-panel continuation currently implemented only for reflections.")
    sx=hasproperty(solver.billiard,:x_axis) ? T(getfield(solver.billiard,:x_axis)) : zero(T)
    sy=hasproperty(solver.billiard,:y_axis) ? T(getfield(solver.billiard,:y_axis)) : zero(T)
    if sym.axis===:y_axis
        xr=_x_reflect(x,sx);yr=y
        txr,tyr=_x_reflect_tangent(tx,ty)
        return xr,yr,txr,tyr,s
    elseif sym.axis===:x_axis
        xr=x;yr=_y_reflect(y,sy)
        txr,tyr=_y_reflect_tangent(tx,ty)
        return xr,yr,txr,tyr,s
    elseif sym.axis===:origin
        xr=_x_reflect(x,sx);yr=_y_reflect(y,sy)
        txr,tyr=_xy_reflect_tangent(tx,ty)
        return xr,yr,txr,tyr,s
    else
        error("Unknown reflection axis $(sym.axis)")
    end
end

function _build_alpert_smooth_panel_cache(solver::CFIE_alpert{T},crv,pts::BoundaryPointsCFIE{T},rule::AlpertLogRule{T},p::Int) where {T<:Real}
    iseven(p) || error("Smooth-panel Alpert interpolation stencil size p must be even.")
    N=length(pts.xy)
    p<=N || error("Smooth-panel Alpert interpolation stencil size p must satisfy p <= N.")
    hσ=pts.ws[1]
    jcorr=rule.j
    sig=copy(pts.ts)
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
    wtp=Array{Complex{T},3}(undef,jcorr,N,p)
    wtm=Array{Complex{T},3}(undef,jcorr,N,p)
    @inbounds for q in 1:jcorr
        Δσ=hσ*rule.x[q]
        for i in 1:N
            σp=sig[i]+Δσ
            xp[q,i],yp[q,i],txp[q,i],typ[q,i],sp[q,i]=_eval_open_panel_geom_continued(solver,crv,σp)
            idx,wt=_continued_panel_interp_data(solver,σp,N,p)
            for m in 1:p
                idxp[q,i,m]=idx[m]
                wtp[q,i,m]=wt[m]
            end
            σm=sig[i]-Δσ
            xm[q,i],ym[q,i],txm[q,i],tym[q,i],sm[q,i]=_eval_open_panel_geom_continued(solver,crv,σm)
            idx,wt=_continued_panel_interp_data(solver,σm,N,p)
            for m in 1:p
                idxm[q,i,m]=idx[m]
                wtm[q,i,m]=wt[m]
            end
        end
    end
    return AlpertSmoothPanelCache(crv,sig,xp,yp,txp,typ,sp,xm,ym,txm,tym,sm,idxp,wtp,idxm,wtm)
end

function _build_alpert_component_cache(solver::CFIE_alpert{T},crv,pts::BoundaryPointsCFIE{T},rule::AlpertLogRule{T},ord::Int) where {T<:Real}
    if pts.is_periodic
        return _build_alpert_periodic_cache(solver,crv,pts,rule,ord)
    else
        pinterp=max(8,ord+3)
        iseven(pinterp) || (pinterp+=1)
        pinterp=min(pinterp,length(pts.xy))
        isodd(pinterp) && (pinterp-=1)
        pinterp>=4 || error("Interpolation stencil too small for smooth-panel Alpert cache.")
        return _build_alpert_smooth_panel_cache(solver,crv,pts,rule,pinterp)
    end
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
            A[gi,gj]-=h*(αD*inn*H(1,k*rij)*invr)
        end
        @inbounds for m in (-a+1):(a-1)
            m==0 && continue
            j=mod1(i+m,N)
            gj=row_range[j]
            rij=G.R[i,j]
            inn=G.inner[i,j]
            invr=G.invR[i,j]
            A[gi,gj]+=h*(αD*inn*H(1,k*rij)*invr)
        end
        @inbounds for p in 1:jcorr
            fac=h*rule.w[p]
            dx=xi-C.xp[p,i]
            dy=yi-C.yp[p,i]
            r2=muladd(dx,dx,dy*dy)
            if isfinite(r2) && r2>(eps(T))^2
                r=sqrt(r2)
                inn=_dinner(dx,dy,C.txp[p,i],C.typ[p,i])
                coeff=-(fac*(αD*inn*H(1,k*r)/r))
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
                coeff=-(fac*(αD*inn*H(1,k*r)/r))
                for m in 1:ninterp
                    q=mod1(i+C.offsm[p,m],N)
                    A[gi,row_range[q]] += coeff*C.wtm[p,m]
                end
            end
        end
        @inbounds for j in 1:N
            j==i && continue
            gj=row_range[j]
            A[gi,gj]-=ik*(h*(αS*H(0,k*G.R[i,j])*G.speed[j]))
        end
        @inbounds for m in (-a+1):(a-1)
            m==0 && continue
            j=mod1(i+m,N)
            gj=row_range[j]
            A[gi,gj]+=ik*(h*(αS*H(0,k*G.R[i,j])*G.speed[j]))
        end
        @inbounds for p in 1:jcorr
            fac=h*rule.w[p]
            dx=xi-C.xp[p,i]
            dy=yi-C.yp[p,i]
            r2=muladd(dx,dx,dy*dy)
            if isfinite(r2) && r2>(eps(T))^2
                r=sqrt(r2)
                coeff=-ik*(fac*(αS*H(0,k*r)*C.sp[p,i]))
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
                coeff=-ik*(fac*(αS*H(0,k*r)*C.sm[p,i]))
                for m in 1:ninterp
                    q=mod1(i+C.offsm[p,m],N)
                    A[gi,row_range[q]] += coeff*C.wtm[p,m]
                end
            end
        end
    end
    return A
end

function _assemble_self_alpert_periodic_deriv!(A::AbstractMatrix{Complex{T}},A1::AbstractMatrix{Complex{T}},A2::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},G::CFIEGeomCache{T},C::AlpertPeriodicCache{T},P::CFIEPanelArrays{T},row_range::UnitRange{Int},k::T,rule::AlpertLogRule{T};multithreaded::Bool=true) where {T<:Real}
    ik=Complex{T}(0,k)
    X=P.X
    Y=P.Y
    N=length(X)
    h=pts.ws[1]
    a=rule.a
    jcorr=rule.j
    ninterp=C.ninterp
    QuantumBilliards.@use_threads multithreading=(multithreaded && N>=16) for i in 1:N
        gi=row_range[i]
        xi=X[i]
        yi=Y[i]
        A[gi,gi]+=one(Complex{T})
        @inbounds for j in 1:N
            j==i && continue
            gj=row_range[j]
            r=G.R[i,j]
            inn=G.inner[i,j]
            invr=G.invR[i,j]
            d0,d1,d2,h0,h1=_dlp_terms(T,k,r,inn,invr,h)
            s0,s1,s2=_slp_terms(T,k,r,G.speed[j],h,h0,h1)
            A[gi,gj]-=d0+ik*s0
            A1[gi,gj]-=d1+Complex{T}(0,1)*s0+ik*s1
            A2[gi,gj]-=d2+Complex{T}(0,2)*s1+ik*s2
        end
        @inbounds for m in (-a+1):(a-1)
            m==0 && continue
            j=mod1(i+m,N)
            gj=row_range[j]
            r=G.R[i,j]
            inn=G.inner[i,j]
            invr=G.invR[i,j]
            d0,d1,d2,h0,h1=_dlp_terms(T,k,r,inn,invr,h)
            s0,s1,s2=_slp_terms(T,k,r,G.speed[j],h,h0,h1)
            A[gi,gj]+=d0+ik*s0
            A1[gi,gj]+=d1+Complex{T}(0,1)*s0+ik*s1
            A2[gi,gj]+=d2+Complex{T}(0,2)*s1+ik*s2
        end
        @inbounds for p in 1:jcorr
            fac=h*rule.w[p]
            dx=xi-C.xp[p,i]
            dy=yi-C.yp[p,i]
            r2=muladd(dx,dx,dy*dy)
            if isfinite(r2) && r2>(eps(T))^2
                r=sqrt(r2)
                inn=C.typ[p,i]*dx-C.txp[p,i]*dy
                d0,d1,d2,h0,h1=_dlp_terms(T,k,r,inn,inv(r),fac)
                s0,s1,s2=_slp_terms(T,k,r,C.sp[p,i],fac,h0,h1)
                for m in 1:ninterp
                    q=mod1(i+C.offsp[p,m],N)
                    w=C.wtp[p,m]
                    gq=row_range[q]
                    A[gi,gq]-=(d0+ik*s0)*w
                    A1[gi,gq]-=(d1+Complex{T}(0,1)*s0+ik*s1)*w
                    A2[gi,gq]-=(d2+Complex{T}(0,2)*s1+ik*s2)*w
                end
            end
            dx=xi-C.xm[p,i]
            dy=yi-C.ym[p,i]
            r2=muladd(dx,dx,dy*dy)
            if isfinite(r2) && r2>(eps(T))^2
                r=sqrt(r2)
                inn=C.tym[p,i]*dx-C.txm[p,i]*dy
                d0,d1,d2,h0,h1=_dlp_terms(T,k,r,inn,inv(r),fac)
                s0,s1,s2=_slp_terms(T,k,r,C.sm[p,i],fac,h0,h1)
                for m in 1:ninterp
                    q=mod1(i+C.offsm[p,m],N)
                    w=C.wtm[p,m]
                    gq=row_range[q]
                    A[gi,gq]-=(d0+ik*s0)*w
                    A1[gi,gq]-=(d1+Complex{T}(0,1)*s0+ik*s1)*w
                    A2[gi,gq]-=(d2+Complex{T}(0,2)*s1+ik*s2)*w
                end
            end
        end
    end
    return A,A1,A2
end

function _assemble_self_alpert_smooth_panel!(solver::CFIE_alpert{T},A::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},G::CFIEGeomCache{T},C::AlpertSmoothPanelCache{T},row_range::UnitRange{Int},k::T,rule::AlpertLogRule{T};multithreaded::Bool=true) where {T<:Real}
    αD=Complex{T}(0,k/2)
    αS=Complex{T}(0,one(T)/2)
    ik=Complex{T}(0,k)
    X=getindex.(pts.xy,1)
    Y=getindex.(pts.xy,2)
    N=length(X)
    hσ=pts.ws[1]
    a=rule.a
    jcorr=rule.j
    @use_threads multithreading=multithreaded for i in 1:N
        gi=row_range[i]
        xi=X[i]
        yi=Y[i]
        ui=C.sig[i]
        A[gi,gi]+=one(Complex{T})
        @inbounds for j in 1:N
            j==i && continue
            gj=row_range[j]
            rij=G.R[i,j]
            inn=G.inner[i,j]
            invr=G.invR[i,j]
            wd=pts.ws[j]
            A[gi,gj]-=wd*(αD*inn*H(1,k*rij)*invr)
        end
        @inbounds for j in 1:N
            j==i && continue
            abs(j-i)<a || continue
            gj=row_range[j]
            rij=G.R[i,j]
            inn=G.inner[i,j]
            invr=G.invR[i,j]
            wd=pts.ws[j]
            A[gi,gj]+=wd*(αD*inn*H(1,k*rij)*invr)
        end
        @inbounds for p in 1:jcorr
            fac=hσ*rule.w[p]
            Δσ=hσ*rule.x[p]
            if ui+Δσ<one(T)
                dx=xi-C.xp[p,i]
                dy=yi-C.yp[p,i]
                r=sqrt(dx*dx+dy*dy)
                if isfinite(r) && r>sqrt(eps(T))
                    inn=_dinner(dx,dy,C.txp[p,i],C.typ[p,i])
                    coeff=-(fac*(αD*inn*H(1,k*r)/r))
                    for m in axes(C.idxp,3)
                        q=C.idxp[p,i,m]
                        A[gi,row_range[q]]+=coeff*C.wtp[p,i,m]
                    end
                end
            end
            if ui-Δσ>zero(T)
                dx=xi-C.xm[p,i]
                dy=yi-C.ym[p,i]
                r=sqrt(dx*dx+dy*dy)
                if isfinite(r) && r>sqrt(eps(T))
                    inn=_dinner(dx,dy,C.txm[p,i],C.tym[p,i])
                    coeff=-(fac*(αD*inn*H(1,k*r)/r))
                    for m in axes(C.idxm,3)
                        q=C.idxm[p,i,m]
                        A[gi,row_range[q]]+=coeff*C.wtm[p,i,m]
                    end
                end
            end
        end
        @inbounds for j in 1:N
            j==i && continue
            gj=row_range[j]
            abs(j-i)<a && continue
            wsj=pts.ws[j]*G.speed[j]
            A[gi,gj]-=ik*(wsj*(αS*H(0,k*G.R[i,j])))
        end
        @inbounds for p in 1:jcorr
            fac=hσ*rule.w[p]
            Δσ=hσ*rule.x[p]
            if ui+Δσ<one(T)
                dx=xi-C.xp[p,i]
                dy=yi-C.yp[p,i]
                r=sqrt(dx*dx+dy*dy)
                if isfinite(r) && r>sqrt(eps(T))
                    coeff=-ik*(fac*(αS*H(0,k*r)*C.sp[p,i]))
                    for m in axes(C.idxp,3)
                        q=C.idxp[p,i,m]
                        A[gi,row_range[q]]+=coeff*C.wtp[p,i,m]
                    end
                end
            end
            if ui-Δσ>zero(T)
                dx=xi-C.xm[p,i]
                dy=yi-C.ym[p,i]
                r=sqrt(dx*dx+dy*dy)
                if isfinite(r) && r>sqrt(eps(T))
                    coeff=-ik*(fac*(αS*H(0,k*r)*C.sm[p,i]))
                    for m in axes(C.idxm,3)
                        q=C.idxm[p,i,m]
                        A[gi,row_range[q]]+=coeff*C.wtm[p,i,m]
                    end
                end
            end
        end
    end
    return A
end

function _assemble_self_alpert_smooth_panel_deriv!(A::AbstractMatrix{Complex{T}},A1::AbstractMatrix{Complex{T}},A2::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},G::CFIEGeomCache{T},C::AlpertSmoothPanelCache{T},P::CFIEPanelArrays{T},row_range::UnitRange{Int},k::T,rule::AlpertLogRule{T};multithreaded::Bool=true) where {T<:Real}
    ik=Complex{T}(0,k)
    X=P.X
    Y=P.Y
    N=length(X)
    hσ=pts.ws[1]
    a=rule.a
    jcorr=rule.j
    QuantumBilliards.@use_threads multithreading=(multithreaded && N>=16) for i in 1:N
        gi=row_range[i]
        xi=X[i]
        yi=Y[i]
        ui=C.sig[i]
        A[gi,gi]+=one(Complex{T})
        @inbounds for j in 1:N
            j==i && continue
            gj=row_range[j]
            r=G.R[i,j]
            inn=G.inner[i,j]
            invr=G.invR[i,j]
            wd=pts.ws[j]
            d0,d1,d2,h0,h1=_dlp_terms(T,k,r,inn,invr,wd)
            if abs(j-i)<a
                A[gi,gj]+=d0
                A1[gi,gj]+=d1
                A2[gi,gj]+=d2
            else
                ws=pts.ws[j]*G.speed[j]
                s0,s1,s2=_slp_terms(T,k,r,one(T),ws,h0,h1)
                A[gi,gj]-=d0+ik*s0
                A1[gi,gj]-=d1+Complex{T}(0,1)*s0+ik*s1
                A2[gi,gj]-=d2+Complex{T}(0,2)*s1+ik*s2
            end
        end
        @inbounds for p in 1:jcorr
            fac=hσ*rule.w[p]
            Δσ=hσ*rule.x[p]
            if ui+Δσ<one(T)
                dx=xi-C.xp[p,i]
                dy=yi-C.yp[p,i]
                r2=muladd(dx,dx,dy*dy)
                if isfinite(r2) && r2>(eps(T))^2
                    r=sqrt(r2)
                    inn=C.typ[p,i]*dx-C.txp[p,i]*dy
                    d0,d1,d2,h0,h1=_dlp_terms(T,k,r,inn,inv(r),fac)
                    s0,s1,s2=_slp_terms(T,k,r,C.sp[p,i],fac,h0,h1)
                    for m in axes(C.idxp,3)
                        q=C.idxp[p,i,m]
                        w=C.wtp[p,i,m]
                        gq=row_range[q]
                        A[gi,gq]+=(d0-ik*s0)*w
                        A1[gi,gq]+=(d1-(Complex{T}(0,1)*s0+ik*s1))*w
                        A2[gi,gq]+=(d2-(Complex{T}(0,2)*s1+ik*s2))*w
                    end
                end
            end
            if ui-Δσ>zero(T)
                dx=xi-C.xm[p,i]
                dy=yi-C.ym[p,i]
                r2=muladd(dx,dx,dy*dy)
                if isfinite(r2) && r2>(eps(T))^2
                    r=sqrt(r2)
                    inn=C.tym[p,i]*dx-C.txm[p,i]*dy
                    d0,d1,d2,h0,h1=_dlp_terms(T,k,r,inn,inv(r),fac)
                    s0,s1,s2=_slp_terms(T,k,r,C.sm[p,i],fac,h0,h1)
                    for m in axes(C.idxm,3)
                        q=C.idxm[p,i,m]
                        w=C.wtm[p,i,m]
                        gq=row_range[q]
                        A[gi,gq]+=(d0-ik*s0)*w
                        A1[gi,gq]+=(d1-(Complex{T}(0,1)*s0+ik*s1))*w
                        A2[gi,gq]+=(d2-(Complex{T}(0,2)*s1+ik*s2))*w
                    end
                end
            end
        end
    end
    return A,A1,A2
end

function _assemble_self_alpert!(solver::CFIE_alpert{T},A::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},G::CFIEGeomCache{T},C,row_range::UnitRange{Int},k::T,rule::AlpertLogRule{T};multithreaded::Bool=true) where {T<:Real}
    pts.is_periodic ?
        _assemble_self_alpert_periodic!(A,pts,G,C,row_range,k,rule;multithreaded=multithreaded) :
        _assemble_self_alpert_smooth_panel!(solver,A,pts,G,C,row_range,k,rule;multithreaded=multithreaded)
end

function _assemble_self_alpert_composite!(solver::CFIE_alpert{T},A::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},Gs::Vector{CFIEGeomCache{T}},Cs,offs::Vector{Int},parr::Vector{CFIEPanelArrays{T}},k::T,rule::AlpertLogRule{T};multithreaded::Bool=true) where {T<:Real}
    αD=Complex{T}(0,k/2)
    αS=Complex{T}(0,one(T)/2)
    ik=Complex{T}(0,k)
    a=rule.a
    for aidx in eachindex(pts)
        pa=pts[aidx]
        pa.is_periodic && continue
        Ca=Cs[aidx]
        Pa=parr[aidx]
        ra=offs[aidx]:(offs[aidx+1]-1)
        Xa=Pa.X
        Ya=Pa.Y
        Na=length(Xa)
        hσ=pa.ws[1]
        @use_threads multithreading=multithreaded for i in 1:Na
            gi=ra[i]
            xi=Xa[i]
            yi=Ya[i]
            A[gi,gi]+=one(Complex{T})
            _add_naive_panel_block!(A,gi,xi,yi,ra,pa,Pa,k,αD,αS,ik;skip_pred=j->(j==i || abs(j-i)<a))
            _add_self_panel_alpert_correction!(A,gi,xi,yi,i,ra,Ca,hσ,k,αD,αS,ik,rule)
        end
    end
    return A
end

function _assemble_all_offpanel_naive!(A::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},offs::Vector{Int},parr::Vector{CFIEPanelArrays{T}},k::T;multithreaded::Bool=true) where {T<:Real}
    αD=Complex{T}(0,k/2)
    αS=Complex{T}(0,one(T)/2)
    ik=Complex{T}(0,k)
    for aidx in eachindex(pts)
        pa=pts[aidx]
        ra=offs[aidx]:(offs[aidx+1]-1)
        Pa=parr[aidx]
        Xa=Pa.X
        Ya=Pa.Y
        Na=length(Xa)
        for bidx in eachindex(pts)
            bidx==aidx && continue
            pb=pts[bidx]
            rb=offs[bidx]:(offs[bidx+1]-1)
            Pb=parr[bidx]
            Xb=Pb.X
            Yb=Pb.Y
            dXb=Pb.dX
            dYb=Pb.dY
            sb=Pb.s
            Nb=length(Xb)
            @use_threads multithreading=(multithreaded && Na>=16) for i in 1:Na
                gi=ra[i]
                xi=Xa[i]
                yi=Ya[i]
                @inbounds for j in 1:Nb
                    gj=rb[j]
                    dx=xi-Xb[j]
                    dy=yi-Yb[j]
                    r2=muladd(dx,dx,dy*dy)
                    r2<=(eps(T))^2 && continue
                    r=sqrt(r2)
                    invr=inv(r)
                    inn=dYb[j]*dx-dXb[j]*dy
                    wd=pb.ws[j]
                    ws=pb.ws[j]*sb[j]
                    dval=wd*(αD*inn*H(1,k*r)*invr)
                    sval=ws*(αS*H(0,k*r))
                    A[gi,gj]-=dval+ik*sval
                end
            end
        end
    end
    return A
end

function _assemble_all_offpanel_naive_deriv!(A::AbstractMatrix{Complex{T}},A1::AbstractMatrix{Complex{T}},A2::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},offs::Vector{Int},parr::Vector{CFIEPanelArrays{T}},k::T;multithreaded::Bool=true,
) where {T<:Real}
    ik=Complex{T}(0,k)
    for aidx in eachindex(pts)
        pa=pts[aidx]
        ra=offs[aidx]:(offs[aidx+1]-1)
        Pa=parr[aidx]
        Xa=Pa.X
        Ya=Pa.Y
        Na=length(Xa)
        for bidx in eachindex(pts)
            bidx==aidx && continue
            pb=pts[bidx]
            rb=offs[bidx]:(offs[bidx+1]-1)
            Pb=parr[bidx]
            Xb=Pb.X
            Yb=Pb.Y
            dXb=Pb.dX
            dYb=Pb.dY
            sb=Pb.s
            Nb=length(Xb)
            QuantumBilliards.@use_threads multithreading=(multithreaded && Na>=16) for i in 1:Na
                gi=ra[i]
                xi=Xa[i]
                yi=Ya[i]
                @inbounds for j in 1:Nb
                    gj=rb[j]
                    dx=xi-Xb[j]
                    dy=yi-Yb[j]
                    r2=muladd(dx,dx,dy*dy)
                    r2<=(eps(T))^2 && continue
                    r=sqrt(r2)
                    invr=inv(r)
                    inn=dYb[j]*dx-dXb[j]*dy
                    wd=pb.ws[j]
                    ws=pb.ws[j]*sb[j]
                    d0,d1,d2,h0,h1=_dlp_terms(T,k,r,inn,invr,wd)
                    s0,s1,s2=_slp_terms(T,k,r,one(T),ws,h0,h1)
                    A[gi,gj]-=d0+ik*s0
                    A1[gi,gj]-=d1+Complex{T}(0,1)*s0+ik*s1
                    A2[gi,gj]-=d2+Complex{T}(0,2)*s1+ik*s2
                end
            end
        end
    end
    return A,A1,A2
end

function _assemble_all_image_naive!(solver::CFIE_alpert{T},A::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},offs::Vector{Int},parr::Vector{CFIEPanelArrays{T}},k::T;multithreaded::Bool=true) where {T<:Real}
    sym=solver.symmetry
    isnothing(sym) && return A
    αD=Complex{T}(0,k/2)
    αS=Complex{T}(0,one(T)/2)
    ik=Complex{T}(0,k)
    sx=hasproperty(solver.billiard,:x_axis) ? T(getfield(solver.billiard,:x_axis)) : zero(T)
    sy=hasproperty(solver.billiard,:y_axis) ? T(getfield(solver.billiard,:y_axis)) : zero(T)
    have_rot=sym isa Rotation
    if have_rot
        ctab,stab,χtab=_rotation_tables(T,sym.n,sym.m)
        cx=T(sym.center[1]);cy=T(sym.center[2])
    end
    for aidx in eachindex(pts)
        ra=offs[aidx]:(offs[aidx+1]-1)
        Pa=parr[aidx]
        Xa=Pa.X;Ya=Pa.Y
        Na=length(Xa)
        for bidx in eachindex(pts)
            rb=offs[bidx]:(offs[bidx+1]-1)
            pb=pts[bidx]
            Pb=parr[bidx]
            Xb=Pb.X;Yb=Pb.Y;dXb=Pb.dX;dYb=Pb.dY;sb=Pb.s
            Nb=length(Xb)
            @use_threads multithreading=(multithreaded && Na>=16) for i in 1:Na
                gi=ra[i]
                xi=Xa[i]
                yi=Ya[i]
                @inbounds for j in 1:Nb
                    if aidx == bidx
                        continue
                    end
                    gj=rb[j]
                    wd=pb.ws[j]
                    ws=pb.ws[j]*sb[j]
                    xj=Xb[j];yj=Yb[j];txj=dXb[j];tyj=dYb[j]
                    if sym isa Reflection
                        if sym.axis===:y_axis
                            xr=_x_reflect(xj,sx);yr=yj
                            txr,tyr=_x_reflect_tangent(txj,tyj)
                            χ=T(sym.parity)
                            dx=xi-xr;dy=yi-yr
                            r2=muladd(dx,dx,dy*dy)
                            r2<=(eps(T))^2 && continue
                            r=sqrt(r2);invr=inv(r)
                            inn=_dinner(dx,dy,txr,tyr)
                            dval= wd*(αD*inn*H(1,k*r)*invr)
                            sval=ws*(αS*H(0,k*r))
                            A[gi,gj]-=χ*(dval+ik*sval)
                        elseif sym.axis===:x_axis
                            xr=xj;yr=_y_reflect(yj,sy)
                            txr,tyr=_y_reflect_tangent(txj,tyj)
                            χ=T(sym.parity)
                            dx=xi-xr;dy=yi-yr
                            r2=muladd(dx,dx,dy*dy)
                            r2<=(eps(T))^2 && continue
                            r=sqrt(r2);invr=inv(r)
                            inn=_dinner(dx,dy,txr,tyr)
                            dval= wd*(αD*inn*H(1,k*r)*invr)
                            sval=ws*(αS*H(0,k*r))
                            A[gi,gj]-=χ*(dval+ik*sval)
                        elseif sym.axis===:origin
                            χx=T(sym.parity[1])
                            χy=T(sym.parity[2])
                            χxy=T(sym.parity[1]*sym.parity[2])
                            xr=_x_reflect(xj,sx);yr=yj
                            txr,tyr=_x_reflect_tangent(txj,tyj)
                            dx=xi-xr;dy=yi-yr
                            r2=muladd(dx,dx,dy*dy)
                            if r2>(eps(T))^2
                                r=sqrt(r2);invr=inv(r)
                                inn=_dinner(dx,dy,txr,tyr)
                                dval= wd*(αD*inn*H(1,k*r)*invr)
                                sval=ws*(αS*H(0,k*r))
                                A[gi,gj]-=χx*(dval+ik*sval)
                            end
                            xr=xj;yr=_y_reflect(yj,sy)
                            txr,tyr=_y_reflect_tangent(txj,tyj)
                            dx=xi-xr;dy=yi-yr
                            r2=muladd(dx,dx,dy*dy)
                            if r2>(eps(T))^2
                                r=sqrt(r2);invr=inv(r)
                                inn=_dinner(dx,dy,txr,tyr)
                                dval= wd*(αD*inn*H(1,k*r)*invr)
                                sval=ws*(αS*H(0,k*r))
                                A[gi,gj]-=χy*(dval+ik*sval)
                            end
                            xr=_x_reflect(xj,sx);yr=_y_reflect(yj,sy)
                            txr,tyr=_xy_reflect_tangent(txj,tyj)
                            dx=xi-xr;dy=yi-yr
                            r2=muladd(dx,dx,dy*dy)
                            if r2>(eps(T))^2
                                r=sqrt(r2);invr=inv(r)
                                inn=_dinner(dx,dy,txr,tyr)
                                dval= wd*(αD*inn*H(1,k*r)*invr)
                                sval=ws*(αS*H(0,k*r))
                                A[gi,gj]-=χxy*(dval+ik*sval)
                            end
                        else
                            error("Unknown reflection axis $(sym.axis)")
                        end
                    else
                        @inbounds for l in 2:sym.n
                            c=ctab[l];s=stab[l]
                            xr,yr=_rot_point(xj,yj,cx,cy,c,s)
                            txr,tyr=_rot_vec(txj,tyj,c,s)
                            χ=χtab[l]
                            dx=xi-xr;dy=yi-yr
                            r2=muladd(dx,dx,dy*dy)
                            r2<=(eps(T))^2 && continue
                            r=sqrt(r2);invr=inv(r)
                            inn=_dinner(dx,dy,txr,tyr)
                            dval=wd*(αD*inn*H(1,k*r)*invr)
                            sval=ws*(αS*H(0,k*r))
                            A[gi,gj]-=χ*(dval+ik*sval)
                        end
                    end
                end
            end
        end
    end
    return A
end

function _assemble_all_image_naive_deriv!(solver::CFIE_alpert{T},A::AbstractMatrix{Complex{T}},A1::AbstractMatrix{Complex{T}},A2::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},offs::Vector{Int},parr::Vector{CFIEPanelArrays{T}},k::T;multithreaded::Bool=true) where {T<:Real}
    sym=solver.symmetry
    isnothing(sym) && return A,A1,A2
    ik=Complex{T}(0,k)
    sx=hasproperty(solver.billiard,:x_axis) ? T(getfield(solver.billiard,:x_axis)) : zero(T)
    sy=hasproperty(solver.billiard,:y_axis) ? T(getfield(solver.billiard,:y_axis)) : zero(T)
    have_rot=sym isa Rotation
    if have_rot
        ctab,stab,χtab=_rotation_tables(T,sym.n,sym.m)
        cx=T(sym.center[1]);cy=T(sym.center[2])
    end
    for aidx in eachindex(pts)
        ra=offs[aidx]:(offs[aidx+1]-1)
        Pa=parr[aidx]
        Xa=Pa.X;Ya=Pa.Y
        Na=length(Xa)
        for bidx in eachindex(pts)
            rb=offs[bidx]:(offs[bidx+1]-1)
            pb=pts[bidx]
            Pb=parr[bidx]
            Xb=Pb.X;Yb=Pb.Y;dXb=Pb.dX;dYb=Pb.dY;sb=Pb.s
            Nb=length(Xb)
            QuantumBilliards.@use_threads multithreading=(multithreaded && Na>=16) for i in 1:Na
                gi=ra[i]
                xi=Xa[i]
                yi=Ya[i]
                @inbounds for j in 1:Nb
                    gj=rb[j]
                    wd=pb.ws[j]
                    ws=pb.ws[j]*sb[j]
                    xj=Xb[j];yj=Yb[j];txj=dXb[j];tyj=dYb[j]
                    if sym isa Reflection
                        if sym.axis===:y_axis
                            xr=_x_reflect(xj,sx);yr=yj
                            txr,tyr=_x_reflect_tangent(txj,tyj)
                            χ=Complex{T}(T(sym.parity),zero(T))
                            dx=xi-xr;dy=yi-yr
                            r2=muladd(dx,dx,dy*dy)
                            r2<=(eps(T))^2 && continue
                            r=sqrt(r2);invr=inv(r)
                            inn=_dinner(dx,dy,txr,tyr)
                            d0,d1,d2,h0,h1=_dlp_terms(T,k,r,inn,invr,wd)
                            s0,s1,s2=_slp_terms(T,k,r,one(T),ws,h0,h1)
                            A[gi,gj]-=χ*(d0+ik*s0)
                            A1[gi,gj]-=χ*(d1+Complex{T}(0,1)*s0+ik*s1)
                            A2[gi,gj]-=χ*(d2+Complex{T}(0,2)*s1+ik*s2)
                        elseif sym.axis===:x_axis
                            xr=xj;yr=_y_reflect(yj,sy)
                            txr,tyr=_y_reflect_tangent(txj,tyj)
                            χ=Complex{T}(T(sym.parity),zero(T))
                            dx=xi-xr;dy=yi-yr
                            r2=muladd(dx,dx,dy*dy)
                            r2<=(eps(T))^2 && continue
                            r=sqrt(r2);invr=inv(r)
                            inn=_dinner(dx,dy,txr,tyr)
                            d0,d1,d2,h0,h1=_dlp_terms(T,k,r,inn,invr,wd)
                            s0,s1,s2=_slp_terms(T,k,r,one(T),ws,h0,h1)
                            A[gi,gj]-=χ*(d0+ik*s0)
                            A1[gi,gj]-=χ*(d1+Complex{T}(0,1)*s0+ik*s1)
                            A2[gi,gj]-=χ*(d2+Complex{T}(0,2)*s1+ik*s2)
                        elseif sym.axis===:origin
                            xr=_x_reflect(xj,sx);yr=_y_reflect(yj,sy)
                            txr,tyr=_xy_reflect_tangent(txj,tyj)
                            χ=Complex{T}(T(sym.parity[1]*sym.parity[2]),zero(T))
                            dx=xi-xr;dy=yi-yr
                            r2=muladd(dx,dx,dy*dy)
                            r2<=(eps(T))^2 && continue
                            r=sqrt(r2);invr=inv(r)
                            inn=_dinner(dx,dy,txr,tyr)
                            d0,d1,d2,h0,h1=_dlp_terms(T,k,r,inn,invr,wd)
                            s0,s1,s2=_slp_terms(T,k,r,one(T),ws,h0,h1)
                            A[gi,gj]-=χ*(d0+ik*s0)
                            A1[gi,gj]-=χ*(d1+Complex{T}(0,1)*s0+ik*s1)
                            A2[gi,gj]-=χ*(d2+Complex{T}(0,2)*s1+ik*s2)
                        else
                            error("Unknown reflection axis $(sym.axis)")
                        end
                    else
                        @inbounds for l in 2:sym.n
                            c=ctab[l];s=stab[l]
                            xr,yr=_rot_point(xj,yj,cx,cy,c,s)
                            txr,tyr=_rot_vec(txj,tyj,c,s)
                            χ=χtab[l]
                            dx=xi-xr;dy=yi-yr
                            r2=muladd(dx,dx,dy*dy)
                            r2<=(eps(T))^2 && continue
                            r=sqrt(r2);invr=inv(r)
                            inn=_dinner(dx,dy,txr,tyr)
                            d0,d1,d2,h0,h1=_dlp_terms(T,k,r,inn,invr,wd)
                            s0,s1,s2=_slp_terms(T,k,r,one(T),ws,h0,h1)
                            A[gi,gj]-=χ*(d0+ik*s0)
                            A1[gi,gj]-=χ*(d1+Complex{T}(0,1)*s0+ik*s1)
                            A2[gi,gj]-=χ*(d2+Complex{T}(0,2)*s1+ik*s2)
                        end
                    end
                end
            end
        end
    end
    return A,A1,A2
end

function build_cfie_alpert_workspace(solver::CFIE_alpert{T},pts::Vector{BoundaryPointsCFIE{T}}) where {T<:Real}
    rule=alpert_log_rule(T,solver.alpert_order)
    offs=component_offsets(pts)
    Gs=[cfie_geom_cache(p) for p in pts]
    boundary=isnothing(solver.symmetry) ? solver.billiard.full_boundary : solver.billiard.desymmetrized_full_boundary
    flat_boundary=boundary[1] isa AbstractVector ? reduce(vcat,boundary) : boundary
    Cs=[_build_alpert_component_cache(solver,flat_boundary[a],pts[a],rule,solver.alpert_order) for a in eachindex(pts)]
    parr=[_panel_arrays_cache(p) for p in pts]
    Ntot=offs[end]-1
    return CFIEAlpertWorkspace(rule,offs,Gs,Cs,parr,Ntot)
end

function construct_matrices!(solver::CFIE_alpert{T},A::Matrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},k::T;multithreaded::Bool=true) where {T<:Real}
    fill!(A,zero(Complex{T}))
    offs=component_offsets(pts)
    Gs=[cfie_geom_cache(p) for p in pts]
    rule=alpert_log_rule(T,solver.alpert_order)
    boundary=isnothing(solver.symmetry) ? solver.billiard.full_boundary : solver.billiard.desymmetrized_full_boundary
    flat_boundary=boundary[1] isa AbstractVector ? reduce(vcat,boundary) : boundary
    Cs=[_build_alpert_component_cache(solver,flat_boundary[a],pts[a],rule,solver.alpert_order) for a in eachindex(pts)]
    parr=[_panel_arrays_cache(p) for p in pts]
    @inbounds for a in eachindex(pts)
        pts[a].is_periodic || continue
        ra=offs[a]:(offs[a+1]-1)
        _assemble_self_alpert!(solver,A,pts[a],Gs[a],Cs[a],ra,k,rule;multithreaded=multithreaded)
    end
    _assemble_self_alpert_composite!(solver,A,pts,Gs,Cs,offs,parr,k,rule;multithreaded=multithreaded)
    _assemble_all_offpanel_naive!(A,pts,offs,parr,k;multithreaded=multithreaded)
    _assemble_all_image_naive!(solver,A,pts,offs,parr,k;multithreaded=multithreaded)
    return A
end

function construct_matrices!(solver::CFIE_alpert{T},A::Matrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},ws::CFIEAlpertWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}
    fill!(A,zero(Complex{T}))
    offs=ws.offs
    Gs=ws.Gs
    Cs=ws.Cs
    parr=ws.parr
    rule=ws.rule
    @inbounds for a in eachindex(pts)
        pts[a].is_periodic || continue
        ra=offs[a]:(offs[a+1]-1)
        _assemble_self_alpert!(solver,A,pts[a],Gs[a],Cs[a],ra,k,rule;multithreaded=multithreaded)
    end
    _assemble_self_alpert_composite!(solver,A,pts,Gs,Cs,offs,parr,k,rule;multithreaded=multithreaded)
    _assemble_all_offpanel_naive!(A,pts,offs,parr,k;multithreaded=multithreaded)
    _assemble_all_image_naive!(solver,A,pts,offs,parr,k;multithreaded=multithreaded)
    return A
end

function construct_matrices!(solver::CFIE_alpert{T},basis::AbstractHankelBasis,A::AbstractMatrix{Complex{T}},dA::AbstractMatrix{Complex{T}},ddA::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},k::T;multithreaded::Bool=true) where {T<:Real}
    ws=build_cfie_alpert_workspace(solver,pts)
    construct_matrices_with_derivatives!(solver,A,dA,ddA,pts,ws,k;multithreaded=multithreaded)
    return A,dA,ddA
end

function construct_matrices!(solver::CFIE_alpert{T},basis::AbstractHankelBasis,A::AbstractMatrix{Complex{T}},dA::AbstractMatrix{Complex{T}},ddA::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},ws::CFIEAlpertWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}
    construct_matrices_with_derivatives!(solver,A,dA,ddA,pts,ws,k;multithreaded=multithreaded)
    return A,dA,ddA
end

function construct_matrices(solver::CFIE_alpert{T},basis::AbstractHankelBasis,pts::Vector{BoundaryPointsCFIE{T}},k::T;multithreaded::Bool=true) where {T<:Real}
    ws=build_cfie_alpert_workspace(solver,pts)
    return construct_matrices_with_derivatives(solver,pts,ws,k;multithreaded=multithreaded)
end

function construct_matrices(solver::CFIE_alpert{T},basis::AbstractHankelBasis,pts::Vector{BoundaryPointsCFIE{T}},ws::CFIEAlpertWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}
    return construct_matrices_with_derivatives(solver,pts,ws,k;multithreaded=multithreaded)
end

function construct_matrices(solver::CFIE_alpert,pts::Vector{BoundaryPointsCFIE{T}},k::T;multithreaded::Bool=true) where {T<:Real}
    Ntot=boundary_matrix_size(pts)
    A=Matrix{Complex{T}}(undef,Ntot,Ntot)
    @blas_1 construct_matrices!(solver,A,pts,k;multithreaded=multithreaded)
    return A
end

function construct_matrices(solver::CFIE_alpert,pts::Vector{BoundaryPointsCFIE{T}},ws::CFIEAlpertWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}
    A=Matrix{Complex{T}}(undef,ws.Ntot,ws.Ntot)
    construct_matrices!(solver,A,pts,ws,k;multithreaded=multithreaded)
    return A
end

function construct_matrices_with_derivatives!(solver::CFIE_alpert{T},A::Matrix{Complex{T}},A1::Matrix{Complex{T}},A2::Matrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},ws::CFIEAlpertWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}
    fill!(A,zero(Complex{T}))
    fill!(A1,zero(Complex{T}))
    fill!(A2,zero(Complex{T}))
    offs=ws.offs
    Gs=ws.Gs
    Cs=ws.Cs
    parr=ws.parr
    rule=ws.rule
    @inbounds for a in eachindex(pts)
        ra=offs[a]:(offs[a+1]-1)
        if pts[a].is_periodic
            _assemble_self_alpert_periodic_deriv!(A,A1,A2,pts[a],Gs[a],Cs[a],parr[a],ra,k,rule;multithreaded=multithreaded)
        else
            _assemble_self_alpert_smooth_panel_deriv!(A,A1,A2,pts[a],Gs[a],Cs[a],parr[a],ra,k,rule;multithreaded=multithreaded)
        end
    end
    _assemble_all_offpanel_naive_deriv!(A,A1,A2,pts,offs,parr,k;multithreaded=multithreaded)
    _assemble_all_image_naive_deriv!(solver,A,A1,A2,pts,offs,parr,k;multithreaded=multithreaded)
    return A,A1,A2
end

function construct_matrices_with_derivatives(solver::CFIE_alpert{T},pts::Vector{BoundaryPointsCFIE{T}},ws::CFIEAlpertWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}
    A=Matrix{Complex{T}}(undef,ws.Ntot,ws.Ntot)
    A1=Matrix{Complex{T}}(undef,ws.Ntot,ws.Ntot)
    A2=Matrix{Complex{T}}(undef,ws.Ntot,ws.Ntot)
    construct_matrices_with_derivatives!(solver,A,A1,A2,pts,ws,k;multithreaded=multithreaded)
    return A,A1,A2
end

function solve(solver::CFIE_alpert,basis::Ba,pts::Vector{BoundaryPointsCFIE{T}},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det_argmin) where {T<:Real,Ba<:AbsBasis}
    A=construct_matrices(solver,pts,k;multithreaded=multithreaded)
    @svd_or_det_solve A use_krylov which MAX_BLAS_THREADS
end

function solve(solver::CFIE_alpert,basis::Ba,pts::Vector{BoundaryPointsCFIE{T}},ws::CFIEAlpertWorkspace{T},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det_argmin) where {T<:Real,Ba<:AbsBasis}
    A=Matrix{Complex{T}}(undef,ws.Ntot,ws.Ntot)
    @blas_1 construct_matrices!(solver,A,pts,ws,k;multithreaded=multithreaded)
    @svd_or_det_solve A use_krylov which MAX_BLAS_THREADS
end

function solve(solver::CFIE_alpert,basis::Ba,A::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det_argmin) where {T<:Real,Ba<:AbsBasis}
    @blas_1 construct_matrices!(solver,A,pts,k;multithreaded=multithreaded)
    @svd_or_det_solve A use_krylov which MAX_BLAS_THREADS
end

function solve(solver::CFIE_alpert,basis::Ba,A::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},ws::CFIEAlpertWorkspace{T},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det_argmin) where {T<:Real,Ba<:AbsBasis}
    @blas_1 construct_matrices!(solver,A,pts,ws,k;multithreaded=multithreaded)
    @svd_or_det_solve A use_krylov which MAX_BLAS_THREADS
end

function solve_vect(solver::CFIE_alpert,basis::Ba,pts::Vector{BoundaryPointsCFIE{T}},k;multithreaded::Bool=true) where {T<:Real,Ba<:AbsBasis}
    @blas_1 A=construct_matrices(solver,pts,k;multithreaded=multithreaded)
    @blas_multi_then_1 MAX_BLAS_THREADS _,S,Vt=LAPACK.gesvd!('A','A',A)
    idx=findmin(S)[2]
    return S[idx],conj.(Vt[idx,:])
end

function solve_vect(solver::CFIE_alpert,basis::Ba,pts::Vector{BoundaryPointsCFIE{T}},ws::CFIEAlpertWorkspace{T},k;multithreaded::Bool=true) where {T<:Real,Ba<:AbsBasis}
    A=Matrix{Complex{T}}(undef,ws.Ntot,ws.Ntot)
    @blas_1 construct_matrices!(solver,A,pts,ws,k;multithreaded=multithreaded)
    @blas_multi_then_1 MAX_BLAS_THREADS _,S,Vt=LAPACK.gesvd!('A','A',A)
    idx=findmin(S)[2]
    return S[idx],conj.(Vt[idx,:])
end

function solve_vect(solver::CFIE_alpert,basis::Ba,A::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},k;multithreaded::Bool=true) where {T<:Real,Ba<:AbsBasis}
    @blas_1 construct_matrices!(solver,A,pts,k;multithreaded=multithreaded)
    @blas_multi_then_1 MAX_BLAS_THREADS _,S,Vt=LAPACK.gesvd!('A','A',A)
    idx=findmin(S)[2]
    return S[idx],conj.(Vt[idx,:])
end

function solve_vect(solver::CFIE_alpert,basis::Ba,A::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},ws::CFIEAlpertWorkspace{T},k;multithreaded::Bool=true) where {T<:Real,Ba<:AbsBasis}
    @blas_1 construct_matrices!(solver,A,pts,ws,k;multithreaded=multithreaded)
    @blas_multi_then_1 MAX_BLAS_THREADS _,S,Vt=LAPACK.gesvd!('A','A',A)
    idx=findmin(S)[2]
    return S[idx],conj.(Vt[idx,:])
end

function solve_INFO(solver::CFIE_alpert,basis::Ba,pts::Vector{BoundaryPointsCFIE{T}},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det_argmin) where {T<:Real,Ba<:AbsBasis}
    offs=component_offsets(pts)
    Ntot=offs[end]-1
    A=Matrix{Complex{T}}(undef,Ntot,Ntot)
    t0=time()
    @info "Building boundary operator A..."
    @blas_1 construct_matrices!(solver,A,pts,k;multithreaded=multithreaded)
    any(isnan.(A))&&error("NaN detected in system matrix A; check geometry and quadrature.")
    t1=time()
    cA=cond(A)
    @info "Condition number of A: $(round(cA;sigdigits=4))"
    t2=time()
    s=@svd_or_det_solve A use_krylov which MAX_BLAS_THREADS
    t3=time()
    build_A=t1-t0
    svd_time=t3-t2
    total=build_A+svd_time
    println("────────── SOLVE_INFO SUMMARY ──────────")
    println("A-matrix build: ",100*build_A/total," %")
    println("SVD: ",100*svd_time/total," %")
    println("(total: ",total," s)")
    println("────────────────────────────────────────")
    return s
end

function solve_INFO(solver::CFIE_alpert,basis::Ba,pts::Vector{BoundaryPointsCFIE{T}},ws::CFIEAlpertWorkspace{T},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det_argmin) where {T<:Real,Ba<:AbsBasis}
    A=Matrix{Complex{T}}(undef,ws.Ntot,ws.Ntot)
    t0=time()
    @info "Building boundary operator A (cached Alpert)..."
    @blas_1 construct_matrices!(solver,A,pts,ws,k;multithreaded=multithreaded)
    any(isnan.(A))&&error("NaN detected in system matrix A; check geometry and quadrature.")
    t1=time()
    cA=cond(A)
    @info "Condition number of A: $(round(cA;sigdigits=4))"
    t2=time()
    s=@svd_or_det_solve A use_krylov which MAX_BLAS_THREADS
    t3=time()
    build_A=t1-t0
    svd_time=t3-t2
    total=build_A+svd_time
    println("────────── SOLVE_INFO SUMMARY ──────────")
    println("A-matrix build: ",100*build_A/total," %")
    println("SVD: ",100*svd_time/total," %")
    println("(total: ",total," s)")
    println("────────────────────────────────────────")
    return s
end