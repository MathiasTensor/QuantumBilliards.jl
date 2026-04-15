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
    rp::Matrix{T}
    rm::Matrix{T}
    innp::Matrix{T}
    innm::Matrix{T}
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
    rp::Matrix{T}
    rm::Matrix{T}
    innp::Matrix{T}
    innm::Matrix{T}
    idxp::Array{Int,3}
    wtp::Array{T,3}
    idxm::Array{Int,3}
    wtm::Array{T,3}
end

const AlpertCache{T}=Union{AlpertPeriodicCache{T},AlpertSmoothPanelCache{T}}

struct CFIEAlpertWorkspace{T<:Real}
    rule::AlpertLogRule{T}
    offs::Vector{Int}
    Gs::Vector{CFIEGeomCache{T}}
    Cs::Vector{AlpertCache{T}}
    parr::Vector{CFIEPanelArrays{T}}
    Ntot::Int
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

@inline function _panel_sigma_wrap(σ::T) where {T<:Real}
    v=mod(σ,one(T))
    v<zero(T) ? v+one(T) : v
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
    Xb=Pb.X;Yb=Pb.Y;dXb=Pb.dX;dYb=Pb.dY;sb=Pb.s;wb=pb.ws
    Nb=length(Xb)
    @inbounds for j in 1:Nb
        skip_pred(j) && continue
        dx=xi-Xb[j];dy=yi-Yb[j]
        r2=muladd(dx,dx,dy*dy)
        r2<=(eps(T))^2 && continue
        r=sqrt(r2);invr=inv(r)
        inn=_dinner(dx,dy,dXb[j],dYb[j])
        h0,h1=hankel_pair01(k*r)
        wd=wb[j];ws=wd*sb[j]
        A[gi,rb[j]]-=wd*(αD*inn*h1*invr)+ik*(ws*(αS*h0))
    end
    return A
end

function _add_self_panel_alpert_correction!(A::AbstractMatrix{Complex{T}},gi::Int,xi::T,yi::T,i::Int,ra::UnitRange{Int},Ca::AlpertSmoothPanelCache{T},hσ::T,k::T,αD::Complex{T},αS::Complex{T},ik::Complex{T},rule::AlpertLogRule{T}) where {T<:Real}
    jcorr=rule.j
    @inbounds for p in 1:jcorr
        fac=hσ*rule.w[p]
        dx=xi-Ca.xp[p,i];dy=yi-Ca.yp[p,i]
        r2=muladd(dx,dx,dy*dy)
        if isfinite(r2) && r2>(eps(T))^2
            r=sqrt(r2)
            inn=_dinner(dx,dy,Ca.txp[p,i],Ca.typ[p,i])
            h0,h1=hankel_pair01(k*r)
            coeff=-(fac*(αD*inn*h1/r))-ik*(fac*(αS*h0*Ca.sp[p,i]))
            for m in axes(Ca.idxp,3)
                A[gi,ra[Ca.idxp[p,i,m]]]+=coeff*Ca.wtp[p,i,m]
            end
        end
        dx=xi-Ca.xm[p,i];dy=yi-Ca.ym[p,i]
        r2=muladd(dx,dx,dy*dy)
        if isfinite(r2) && r2>(eps(T))^2
            r=sqrt(r2)
            inn=_dinner(dx,dy,Ca.txm[p,i],Ca.tym[p,i])
            h0,h1=hankel_pair01(k*r)
            coeff=-(fac*(αD*inn*h1/r))-ik*(fac*(αS*h0*Ca.sm[p,i]))
            for m in axes(Ca.idxm,3)
                A[gi,ra[Ca.idxm[p,i,m]]]+=coeff*Ca.wtm[p,i,m]
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
    X=getindex.(pts.xy,1)
    Y=getindex.(pts.xy,2)
    xp=Matrix{T}(undef,jcorr,N);yp=similar(xp);txp=similar(xp);typ=similar(xp);sp=similar(xp)
    xm=Matrix{T}(undef,jcorr,N);ym=similar(xm);txm=similar(xm);tym=similar(xm);sm=similar(xm)
    rp=Matrix{T}(undef,jcorr,N);rm=similar(rp);innp=similar(rp);innm=similar(rp)
    offsp=Matrix{Int}(undef,jcorr,ninterp);wtp=Matrix{T}(undef,jcorr,ninterp)
    offsm=Matrix{Int}(undef,jcorr,ninterp);wtm=Matrix{T}(undef,jcorr,ninterp)
    bad=T(Inf)
    @inbounds for p in 1:jcorr
        ξ=rule.x[p]
        op,wp=_alpert_interp_offsets_weights(ξ,ninterp)
        om,wm=_alpert_interp_offsets_weights(-ξ,ninterp)
        for m in 1:ninterp
            offsp[p,m]=op[m];wtp[p,m]=wp[m]
            offsm[p,m]=om[m];wtm[p,m]=wm[m]
        end
        δu=T(σ)*ξ/T(N)
        for i in 1:N
            xi=X[i];yi=Y[i]
            ui=pts.ts[i]/T(two_pi)
            up=_wrap01(ui+δu)
            qp=curve(crv,up)
            tp=T(σ)*tangent(crv,up)/T(two_pi)
            xpi=qp[1];ypi=qp[2];txpi=tp[1];typi=tp[2]
            xp[p,i]=xpi;yp[p,i]=ypi;txp[p,i]=txpi;typ[p,i]=typi;sp[p,i]=sqrt(txpi^2+typi^2)
            dx=xi-xpi;dy=yi-ypi
            r2=muladd(dx,dx,dy*dy)
            if isfinite(r2) && r2>(eps(T))^2
                rp[p,i]=sqrt(r2)
                innp[p,i]=typi*dx-txpi*dy
            else
                rp[p,i]=bad
                innp[p,i]=zero(T)
            end
            um=_wrap01(ui-δu)
            qm=curve(crv,um)
            tm=T(σ)*tangent(crv,um)/T(two_pi)
            xmi=qm[1];ymi=qm[2];txmi=tm[1];tymi=tm[2]
            xm[p,i]=xmi;ym[p,i]=ymi;txm[p,i]=txmi;tym[p,i]=tymi;sm[p,i]=sqrt(txmi^2+tymi^2)
            dx=xi-xmi;dy=yi-ymi
            r2=muladd(dx,dx,dy*dy)
            if isfinite(r2) && r2>(eps(T))^2
                rm[p,i]=sqrt(r2)
                innm[p,i]=tymi*dx-txmi*dy
            else
                rm[p,i]=bad
                innm[p,i]=zero(T)
            end
        end
    end
    return AlpertPeriodicCache(xp,yp,txp,typ,sp,xm,ym,txm,tym,sm,rp,rm,innp,innm,offsp,wtp,offsm,wtm,ninterp)
end

function _build_alpert_smooth_panel_cache(solver::CFIE_alpert{T},crv,pts::BoundaryPointsCFIE{T},rule::AlpertLogRule{T},p::Int) where {T<:Real}
    iseven(p) || error("Smooth-panel Alpert interpolation stencil size p must be even.")
    N=length(pts.xy)
    p<=N || error("Smooth-panel Alpert interpolation stencil size p must satisfy p <= N.")
    hσ=pts.ws[1]
    jcorr=rule.j
    sig=copy(pts.ts)
    X=getindex.(pts.xy,1)
    Y=getindex.(pts.xy,2)
    xp=Matrix{T}(undef,jcorr,N);yp=similar(xp);txp=similar(xp);typ=similar(xp);sp=similar(xp)
    xm=similar(xp);ym=similar(xp);txm=similar(xp);tym=similar(xp);sm=similar(xp)
    rp=similar(xp);rm=similar(xp);innp=similar(xp);innm=similar(xp)
    idxp=Array{Int,3}(undef,jcorr,N,p);idxm=Array{Int,3}(undef,jcorr,N,p)
    wtp=Array{T,3}(undef,jcorr,N,p);wtm=Array{T,3}(undef,jcorr,N,p)
    bad=T(Inf)
    @inbounds for q in 1:jcorr
        Δσ=hσ*rule.x[q]
        for i in 1:N
            xi=X[i];yi=Y[i]
            σp=_panel_sigma_wrap(sig[i]+Δσ)
            up,jp,_=_panel_sigma_to_u_jac(solver,σp)
            x,y,tu,tv,su=_eval_open_panel_geom_exact(crv,up)
            idx,wt=_panel_interp_midpoint_data(σp,hσ,N,p)
            tx=tu*jp;ty=tv*jp
            xp[q,i]=x;yp[q,i]=y;txp[q,i]=tx;typ[q,i]=ty;sp[q,i]=su*jp
            dx=xi-x;dy=yi-y
            r2=muladd(dx,dx,dy*dy)
            if isfinite(r2) && r2>(eps(T))^2
                rp[q,i]=sqrt(r2)
                innp[q,i]=ty*dx-tx*dy
            else
                rp[q,i]=bad
                innp[q,i]=zero(T)
            end
            for m in 1:p
                idxp[q,i,m]=idx[m]
                wtp[q,i,m]=wt[m]
            end
            σm=_panel_sigma_wrap(sig[i]-Δσ)
            um,jm,_=_panel_sigma_to_u_jac(solver,σm)
            x,y,tu,tv,su=_eval_open_panel_geom_exact(crv,um)
            idx,wt=_panel_interp_midpoint_data(σm,hσ,N,p)
            tx=tu*jm;ty=tv*jm
            xm[q,i]=x;ym[q,i]=y;txm[q,i]=tx;tym[q,i]=ty;sm[q,i]=su*jm
            dx=xi-x;dy=yi-y
            r2=muladd(dx,dx,dy*dy)
            if isfinite(r2) && r2>(eps(T))^2
                rm[q,i]=sqrt(r2)
                innm[q,i]=ty*dx-tx*dy
            else
                rm[q,i]=bad
                innm[q,i]=zero(T)
            end
            for m in 1:p
                idxm[q,i,m]=idx[m]
                wtm[q,i,m]=wt[m]
            end
        end
    end
    return AlpertSmoothPanelCache(crv,sig,xp,yp,txp,typ,sp,xm,ym,txm,tym,sm,rp,rm,innp,innm,idxp,wtp,idxm,wtm)
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

function _assemble_self_alpert_periodic!(A::Matrix{Complex{T}},pts::BoundaryPointsCFIE{T},G::CFIEGeomCache{T},C::AlpertPeriodicCache{T},row_range::UnitRange{Int},k::T,rule::AlpertLogRule{T};multithreaded::Bool=true) where {T<:Real}
    αD=Complex{T}(0,k/2);αS=Complex{T}(0,one(T)/2);ik=Complex{T}(0,k)
    X=getindex.(pts.xy,1);Y=getindex.(pts.xy,2)
    R=G.R;invR=G.invR;inner=G.inner;speed=G.speed
    rp=C.rp;rm=C.rm;innp=C.innp;innm=C.innm;sp=C.sp;sm=C.sm
    offsp=C.offsp;wtp=C.wtp;offsm=C.offsm;wtm=C.wtm
    r0=first(row_range)-1
    N=length(X);h=pts.ws[1];a=rule.a;jcorr=rule.j;ninterp=C.ninterp
    @use_threads multithreading=(multithreaded && N>=16) for i in 1:N
        gi=r0+i
        A[gi,gi]+=one(Complex{T})
        @inbounds for j in 1:N
            j==i && continue
            r=R[i,j]
            h0,h1=hankel_pair01(k*r)
            A[gi,r0+j]-=h*(αD*inner[i,j]*h1*invR[i,j])+ik*(h*(αS*h0*speed[j]))
        end
        @inbounds for s in (-a+1):(a-1)
            s==0 && continue
            j=mod1(i+s,N)
            r=R[i,j]
            h0,h1=hankel_pair01(k*r)
            A[gi,r0+j]+=h*(αD*inner[i,j]*h1*invR[i,j])+ik*(h*(αS*h0*speed[j]))
        end
        @inbounds for p in 1:jcorr
            fac=h*rule.w[p]
            r=rp[p,i]
            if isfinite(r)
                h0,h1=hankel_pair01(k*r)
                coeff=-(fac*(αD*innp[p,i]*h1/r))-ik*(fac*(αS*h0*sp[p,i]))
                for m in 1:ninterp
                    A[gi,r0+mod1(i+offsp[p,m],N)]+=coeff*wtp[p,m]
                end
            end
            r=rm[p,i]
            if isfinite(r)
                h0,h1=hankel_pair01(k*r)
                coeff=-(fac*(αD*innm[p,i]*h1/r))-ik*(fac*(αS*h0*sm[p,i]))
                for m in 1:ninterp
                    A[gi,r0+mod1(i+offsm[p,m],N)]+=coeff*wtm[p,m]
                end
            end
        end
    end
    return A
end

function _assemble_self_alpert_periodic_deriv!(A::AbstractMatrix{Complex{T}},A1::AbstractMatrix{Complex{T}},A2::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},G::CFIEGeomCache{T},C::AlpertPeriodicCache{T},P::CFIEPanelArrays{T},row_range::UnitRange{Int},k::T,rule::AlpertLogRule{T};multithreaded::Bool=true) where {T<:Real}
    ik=Complex{T}(0,k)
    X=P.X;Y=P.Y
    R=G.R;invR=G.invR;inner=G.inner;speed=G.speed
    rp=C.rp;rm=C.rm;innp=C.innp;innm=C.innm;sp=C.sp;sm=C.sm
    offsp=C.offsp;wtp=C.wtp;offsm=C.offsm;wtm=C.wtm
    N=length(X);h=pts.ws[1];a=rule.a;jcorr=rule.j;ninterp=C.ninterp
    QuantumBilliards.@use_threads multithreading=(multithreaded && N>=16) for i in 1:N
        gi=row_range[i]
        A[gi,gi]+=one(Complex{T})
        @inbounds for j in 1:N
            j==i && continue
            gj=row_range[j]
            d0,d1,d2,h0,h1=_dlp_terms(T,k,R[i,j],inner[i,j],invR[i,j],h)
            s0,s1,s2=_slp_terms(T,k,R[i,j],speed[j],h,h0,h1)
            A[gi,gj]-=d0+ik*s0
            A1[gi,gj]-=d1+Complex{T}(0,1)*s0+ik*s1
            A2[gi,gj]-=d2+Complex{T}(0,2)*s1+ik*s2
        end
        @inbounds for m in (-a+1):(a-1)
            m==0 && continue
            j=mod1(i+m,N);gj=row_range[j]
            d0,d1,d2,h0,h1=_dlp_terms(T,k,R[i,j],inner[i,j],invR[i,j],h)
            s0,s1,s2=_slp_terms(T,k,R[i,j],speed[j],h,h0,h1)
            A[gi,gj]+=d0+ik*s0
            A1[gi,gj]+=d1+Complex{T}(0,1)*s0+ik*s1
            A2[gi,gj]+=d2+Complex{T}(0,2)*s1+ik*s2
        end
        @inbounds for p in 1:jcorr
            fac=h*rule.w[p]
            r=rp[p,i]
            if isfinite(r)
                d0,d1,d2,h0,h1=_dlp_terms(T,k,r,innp[p,i],inv(r),fac)
                s0,s1,s2=_slp_terms(T,k,r,sp[p,i],fac,h0,h1)
                for m in 1:ninterp
                    gq=row_range[mod1(i+offsp[p,m],N)];ww=wtp[p,m]
                    A[gi,gq]-=(d0+ik*s0)*ww
                    A1[gi,gq]-=(d1+Complex{T}(0,1)*s0+ik*s1)*ww
                    A2[gi,gq]-=(d2+Complex{T}(0,2)*s1+ik*s2)*ww
                end
            end
            r=rm[p,i]
            if isfinite(r)
                d0,d1,d2,h0,h1=_dlp_terms(T,k,r,innm[p,i],inv(r),fac)
                s0,s1,s2=_slp_terms(T,k,r,sm[p,i],fac,h0,h1)
                for m in 1:ninterp
                    gq=row_range[mod1(i+offsm[p,m],N)];ww=wtm[p,m]
                    A[gi,gq]-=(d0+ik*s0)*ww
                    A1[gi,gq]-=(d1+Complex{T}(0,1)*s0+ik*s1)*ww
                    A2[gi,gq]-=(d2+Complex{T}(0,2)*s1+ik*s2)*ww
                end
            end
        end
    end
    return A,A1,A2
end

function _assemble_self_alpert_smooth_panel!(A::Matrix{Complex{T}},pts::BoundaryPointsCFIE{T},G::CFIEGeomCache{T},C::AlpertSmoothPanelCache{T},row_range::UnitRange{Int},k::T,rule::AlpertLogRule{T};multithreaded::Bool=true) where {T<:Real}
    αD=Complex{T}(0,k/2);αS=Complex{T}(0,one(T)/2);ik=Complex{T}(0,k)
    X=getindex.(pts.xy,1);Y=getindex.(pts.xy,2);w=pts.ws
    R=G.R;invR=G.invR;inner=G.inner;speed=G.speed
    rp=C.rp;rm=C.rm;innp=C.innp;innm=C.innm;sp=C.sp;sm=C.sm
    idxp=C.idxp;wtp=C.wtp;idxm=C.idxm;wtm=C.wtm
    r0=first(row_range)-1
    N=length(X);hσ=w[1];a=rule.a;jcorr=rule.j;pinterp=size(idxp,3)
    @use_threads multithreading=(multithreaded && N>=16) for i in 1:N
        gi=r0+i
        A[gi,gi]+=one(Complex{T})
        @inbounds for j in 1:N
            j==i && continue
            r=R[i,j]
            h0,h1=hankel_pair01(k*r)
            if abs(j-i)<a
                A[gi,r0+j]+=w[j]*(αD*inner[i,j]*h1*invR[i,j])
            else
                A[gi,r0+j]-=w[j]*(αD*inner[i,j]*h1*invR[i,j])+ik*((w[j]*speed[j])*(αS*h0))
            end
        end
        @inbounds for p in 1:jcorr
            fac=hσ*rule.w[p]
            r=rp[p,i]
            if isfinite(r)
                h0,h1=hankel_pair01(k*r)
                coeff=-(fac*(αD*innp[p,i]*h1/r))-ik*(fac*(αS*h0*sp[p,i]))
                for m in 1:pinterp
                    A[gi,r0+idxp[p,i,m]]+=coeff*wtp[p,i,m]
                end
            end
            r=rm[p,i]
            if isfinite(r)
                h0,h1=hankel_pair01(k*r)
                coeff=-(fac*(αD*innm[p,i]*h1/r))-ik*(fac*(αS*h0*sm[p,i]))
                for m in 1:pinterp
                    A[gi,r0+idxm[p,i,m]]+=coeff*wtm[p,i,m]
                end
            end
        end
    end
    return A
end

function _assemble_self_alpert_smooth_panel_deriv!(A::AbstractMatrix{Complex{T}},A1::AbstractMatrix{Complex{T}},A2::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},G::CFIEGeomCache{T},C::AlpertSmoothPanelCache{T},P::CFIEPanelArrays{T},row_range::UnitRange{Int},k::T,rule::AlpertLogRule{T};multithreaded::Bool=true) where {T<:Real}
    ik=Complex{T}(0,k)
    X=P.X;Y=P.Y;w=pts.ws
    R=G.R;invR=G.invR;inner=G.inner;speed=G.speed
    rp=C.rp;rm=C.rm;innp=C.innp;innm=C.innm;sp=C.sp;sm=C.sm
    idxp=C.idxp;wtp=C.wtp;idxm=C.idxm;wtm=C.wtm
    N=length(X);hσ=w[1];a=rule.a;jcorr=rule.j
    QuantumBilliards.@use_threads multithreading=(multithreaded && N>=16) for i in 1:N
        gi=row_range[i]
        A[gi,gi]+=one(Complex{T})
        @inbounds for j in 1:N
            j==i && continue
            gj=row_range[j]
            d0,d1,d2,h0,h1=_dlp_terms(T,k,R[i,j],inner[i,j],invR[i,j],w[j])
            if abs(j-i)<a
                A[gi,gj]+=d0
                A1[gi,gj]+=d1
                A2[gi,gj]+=d2
            else
                s0,s1,s2=_slp_terms(T,k,R[i,j],one(T),w[j]*speed[j],h0,h1)
                A[gi,gj]-=d0+ik*s0
                A1[gi,gj]-=d1+Complex{T}(0,1)*s0+ik*s1
                A2[gi,gj]-=d2+Complex{T}(0,2)*s1+ik*s2
            end
        end
        @inbounds for p in 1:jcorr
            fac=hσ*rule.w[p]
            r=rp[p,i]
            if isfinite(r)
                d0,d1,d2,h0,h1=_dlp_terms(T,k,r,innp[p,i],inv(r),fac)
                s0,s1,s2=_slp_terms(T,k,r,sp[p,i],fac,h0,h1)
                for m in axes(idxp,3)
                    gq=row_range[idxp[p,i,m]];ww=wtp[p,i,m]
                    A[gi,gq]+=(d0-ik*s0)*ww
                    A1[gi,gq]+=(d1-(Complex{T}(0,1)*s0+ik*s1))*ww
                    A2[gi,gq]+=(d2-(Complex{T}(0,2)*s1+ik*s2))*ww
                end
            end
            r=rm[p,i]
            if isfinite(r)
                d0,d1,d2,h0,h1=_dlp_terms(T,k,r,innm[p,i],inv(r),fac)
                s0,s1,s2=_slp_terms(T,k,r,sm[p,i],fac,h0,h1)
                for m in axes(idxm,3)
                    gq=row_range[idxm[p,i,m]];ww=wtm[p,i,m]
                    A[gi,gq]+=(d0-ik*s0)*ww
                    A1[gi,gq]+=(d1-(Complex{T}(0,1)*s0+ik*s1))*ww
                    A2[gi,gq]+=(d2-(Complex{T}(0,2)*s1+ik*s2))*ww
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

function _assemble_all_offpanel_naive!(A::Matrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},offs::Vector{Int},parr::Vector{CFIEPanelArrays{T}},k::T;multithreaded::Bool=true) where {T<:Real}
    αD=Complex{T}(0,k/2);αS=Complex{T}(0,one(T)/2);ik=Complex{T}(0,k)
    for aidx in eachindex(pts)
        ra=offs[aidx]:(offs[aidx+1]-1);r0a=first(ra)-1;Pa=parr[aidx]
        Xa=Pa.X;Ya=Pa.Y;Na=length(Xa)
        for bidx in eachindex(pts)
            bidx==aidx && continue
            pb=pts[bidx];rb=offs[bidx]:(offs[bidx+1]-1);r0b=first(rb)-1;Pb=parr[bidx]
            Xb=Pb.X;Yb=Pb.Y;dXb=Pb.dX;dYb=Pb.dY;sb=Pb.s;wb=pb.ws;Nb=length(Xb)
            @use_threads multithreading=(multithreaded && Na>=16) for i in 1:Na
                gi=r0a+i
                xi=Xa[i];yi=Ya[i]
                @inbounds for j in 1:Nb
                    dx=xi-Xb[j];dy=yi-Yb[j]
                    r2=muladd(dx,dx,dy*dy)
                    r2<=(eps(T))^2 && continue
                    r=sqrt(r2)
                    h0,h1=hankel_pair01(k*r)
                    wd=wb[j]
                    A[gi,r0b+j]-=wd*(αD*(dYb[j]*dx-dXb[j]*dy)*h1/r)+ik*((wd*sb[j])*(αS*h0))
                end
            end
        end
    end
    return A
end

function _assemble_all_offpanel_naive_deriv!(A::Matrix{Complex{T}},A1::Matrix{Complex{T}},A2::Matrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},offs::Vector{Int},parr::Vector{CFIEPanelArrays{T}},k::T;multithreaded::Bool=true) where {T<:Real}
    ik=Complex{T}(0,k)
    for aidx in eachindex(pts)
        ra=offs[aidx]:(offs[aidx+1]-1);r0a=first(ra)-1;Pa=parr[aidx]
        Xa=Pa.X;Ya=Pa.Y;Na=length(Xa)
        for bidx in eachindex(pts)
            bidx==aidx && continue
            pb=pts[bidx];rb=offs[bidx]:(offs[bidx+1]-1);r0b=first(rb)-1;Pb=parr[bidx]
            Xb=Pb.X;Yb=Pb.Y;dXb=Pb.dX;dYb=Pb.dY;sb=Pb.s;wb=pb.ws;Nb=length(Xb)
            QuantumBilliards.@use_threads multithreading=(multithreaded && Na>=16) for i in 1:Na
                gi=r0a+i
                xi=Xa[i];yi=Ya[i]
                @inbounds for j in 1:Nb
                    dx=xi-Xb[j];dy=yi-Yb[j]
                    r2=muladd(dx,dx,dy*dy)
                    r2<=(eps(T))^2 && continue
                    r=sqrt(r2);invr=inv(r)
                    wd=wb[j]
                    d0,d1,d2,h0,h1=_dlp_terms(T,k,r,dYb[j]*dx-dXb[j]*dy,invr,wd)
                    s0,s1,s2=_slp_terms(T,k,r,one(T),wd*sb[j],h0,h1)
                    gj=r0b+j
                    A[gi,gj]-=d0+ik*s0
                    A1[gi,gj]-=d1+Complex{T}(0,1)*s0+ik*s1
                    A2[gi,gj]-=d2+Complex{T}(0,2)*s1+ik*s2
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
    boundary=solver.billiard.full_boundary
    flat_boundary=boundary[1] isa AbstractVector ? reduce(vcat,boundary) : boundary
    Cs=Vector{AlpertCache{T}}(undef,length(pts))
    @inbounds for a in eachindex(pts)
        Cs[a]=_build_alpert_component_cache(solver,flat_boundary[a],pts[a],rule,solver.alpert_order)
    end
    parr=[_panel_arrays_cache(p) for p in pts]
    return CFIEAlpertWorkspace(rule,offs,Gs,Cs,parr,offs[end]-1)
end

@inline function _construct_matrices_cached!(A::Matrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},ws::CFIEAlpertWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}
    fill!(A,zero(Complex{T}))
    offs=ws.offs;Gs=ws.Gs;Cs=ws.Cs;parr=ws.parr;rule=ws.rule
    @inbounds for a in eachindex(pts)
        ra=offs[a]:(offs[a+1]-1)
        if pts[a].is_periodic
            _assemble_self_alpert_periodic!(A,pts[a],Gs[a],Cs[a]::AlpertPeriodicCache{T},ra,k,rule;multithreaded=multithreaded)
        else
            _assemble_self_alpert_smooth_panel!(A,pts[a],Gs[a],Cs[a]::AlpertSmoothPanelCache{T},ra,k,rule;multithreaded=multithreaded)
        end
    end
    _assemble_all_offpanel_naive!(A,pts,offs,parr,k;multithreaded=multithreaded)
    return A
end

@inline function _construct_matrices_deriv_cached!(A::Matrix{Complex{T}},A1::Matrix{Complex{T}},A2::Matrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},ws::CFIEAlpertWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}
    fill!(A,zero(Complex{T}))
    fill!(A1,zero(Complex{T}))
    fill!(A2,zero(Complex{T}))
    offs=ws.offs;Gs=ws.Gs;Cs=ws.Cs;parr=ws.parr;rule=ws.rule
    @inbounds for a in eachindex(pts)
        ra=offs[a]:(offs[a+1]-1)
        if pts[a].is_periodic
            _assemble_self_alpert_periodic_deriv!(A,A1,A2,pts[a],Gs[a],Cs[a]::AlpertPeriodicCache{T},parr[a],ra,k,rule;multithreaded=multithreaded)
        else
            _assemble_self_alpert_smooth_panel_deriv!(A,A1,A2,pts[a],Gs[a],Cs[a]::AlpertSmoothPanelCache{T},parr[a],ra,k,rule;multithreaded=multithreaded)
        end
    end
    _assemble_all_offpanel_naive_deriv!(A,A1,A2,pts,offs,parr,k;multithreaded=multithreaded)
    return A,A1,A2
end

function construct_matrices!(solver::CFIE_alpert{T},A::Matrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},ws::CFIEAlpertWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}
    _construct_matrices_cached!(A,pts,ws,k;multithreaded=multithreaded)
end

function construct_matrices!(solver::CFIE_alpert{T},A::Matrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},k::T;multithreaded::Bool=true) where {T<:Real}
    ws=build_cfie_alpert_workspace(solver,pts)
    _construct_matrices_cached!(A,pts,ws,k;multithreaded=multithreaded)
end

function construct_matrices(solver::CFIE_alpert{T},pts::Vector{BoundaryPointsCFIE{T}},ws::CFIEAlpertWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}
    A=Matrix{Complex{T}}(undef,ws.Ntot,ws.Ntot)
    _construct_matrices_cached!(A,pts,ws,k;multithreaded=multithreaded)
    return A
end

function construct_matrices(solver::CFIE_alpert{T},pts::Vector{BoundaryPointsCFIE{T}},k::T;multithreaded::Bool=true) where {T<:Real}
    ws=build_cfie_alpert_workspace(solver,pts)
    construct_matrices(solver,pts,ws,k;multithreaded=multithreaded)
end

function construct_matrices!(solver::CFIE_alpert{T},A::Matrix{Complex{T}},A1::Matrix{Complex{T}},A2::Matrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},ws::CFIEAlpertWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}
    _construct_matrices_deriv_cached!(A,A1,A2,pts,ws,k;multithreaded=multithreaded)
end

function construct_matrices!(solver::CFIE_alpert{T},A::Matrix{Complex{T}},A1::Matrix{Complex{T}},A2::Matrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},k::T;multithreaded::Bool=true) where {T<:Real}
    ws=build_cfie_alpert_workspace(solver,pts)
    _construct_matrices_deriv_cached!(A,A1,A2,pts,ws,k;multithreaded=multithreaded)
end

function construct_matrices!(solver::CFIE_alpert{T},basis::AbstractHankelBasis,A::Matrix{Complex{T}},A1::Matrix{Complex{T}},A2::Matrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},ws::CFIEAlpertWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}
    _construct_matrices_deriv_cached!(A,A1,A2,pts,ws,k;multithreaded=multithreaded)
end

function construct_matrices!(solver::CFIE_alpert{T},basis::AbstractHankelBasis,A::Matrix{Complex{T}},A1::Matrix{Complex{T}},A2::Matrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},k::T;multithreaded::Bool=true) where {T<:Real}
    ws=build_cfie_alpert_workspace(solver,pts)
    _construct_matrices_deriv_cached!(A,A1,A2,pts,ws,k;multithreaded=multithreaded)
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
    @info "Building boundary operator A..."
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