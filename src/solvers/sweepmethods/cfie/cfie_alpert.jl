"""
    AlpertPeriodicCache{T}

Cache for Alpert near-singular self-panel correction on a closed periodic
boundary component.

## Description
This cache stores all geometry and interpolation data required by the Alpert
hybrid Gauss-trapezoidal correction for one smooth closed boundary component
sampled on a periodic midpoint grid.

For every target node `i` and Alpert abscissa `ξ_p`, correction points are
constructed at signed parameter offsets

    ±ξ_p/N.

Because the component is periodic, interpolation from these off-grid correction
points back to the native discretization can be represented by integer offsets
modulo the number of nodes.

## Attributes
* `xp`, `yp`: Coordinates of the positive-side Alpert correction nodes.
* `txp`, `typ`: Tangent components at the positive-side correction nodes.
* `sp`: Speed at the positive-side correction nodes.
* `xm`, `ym`: Coordinates of the negative-side correction nodes.
* `txm`, `tym`: Tangent components at the negative-side correction nodes.
* `sm`: Speed at the negative-side correction nodes.
* `rp`, `rm`: Target-to-correction-node distances. Degenerate distances are
  stored as `Inf`.
* `innp`, `innm`: Oriented DLP numerators at the positive and negative
  correction nodes.
* `offsp`, `offsm`: Periodic interpolation offsets for the positive and negative
  correction nodes.
* `wtp`, `wtm`: Corresponding interpolation weights.
* `ninterp`: Number of interpolation nodes in each periodic stencil.

## Notes
This cache is intended only for boundary discretizations satisfying

    pts.is_periodic == true.
"""
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

"""
    AlpertSmoothPanelCache{T}

Cache for Alpert near-singular self-panel correction on an open smooth panel.

## Description
This is the open-panel analogue of [`AlpertPeriodicCache`](@ref).

For an open panel, the Alpert correction nodes are generated in the panel's
computational coordinate and then mapped through the panel grading
transformation. Since the panel is not periodic, interpolation stencils cannot
wrap modulo the number of nodes. Explicit source indices and interpolation
weights must therefore be stored separately for every target and correction
node.

## Attributes
* `crv`: Underlying geometric curve associated with the panel.
* `sig`: Native computational midpoint nodes of the panel.
* `xp`, `yp`: Positive correction-node coordinates.
* `txp`, `typ`: Positive correction-node tangent components.
* `sp`: Positive correction-node speeds.
* `xm`, `ym`: Negative correction-node coordinates.
* `txm`, `tym`: Negative correction-node tangent components.
* `sm`: Negative correction-node speeds.
* `rp`, `rm`: Distances from the target nodes to the positive and negative
  correction nodes.
* `innp`, `innm`: Oriented DLP numerators at the correction nodes.
* `idxp`, `idxm`: Explicit interpolation indices for each correction node.
* `wtp`, `wtm`: Corresponding interpolation weights.

## Notes
This cache is intended only for boundary discretizations satisfying

    pts.is_periodic == false.
"""
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

"""
    CFIEAlpertWorkspace{T}

Reusable workspace for CFIE-Alpert assembly on a fixed boundary discretization.

## Description
The workspace contains all data that depend only on the boundary geometry,
sampling, and selected Alpert rule, but not on the current wavenumber `k`.

Repeated matrix constructions can therefore reuse:
* the Alpert logarithmic quadrature rule,
* global component offsets,
* pairwise geometry caches,
* Alpert correction-node caches,
* flattened panel geometry arrays.

## Attributes
* `rule`: Selected [`AlpertLogRule`](@ref).
* `offs`: Global offsets of the individual panels/components.
* `Gs`: One [`BoundaryGeomCache`](@ref) for each panel/component.
* `Cs`: One periodic or open-panel Alpert correction cache per panel/component.
* `parr`: One [`BoundaryPanelArrays`](@ref) object per panel/component.
* `Ntot`: Total matrix dimension.
"""
struct CFIEAlpertWorkspace{T<:Real}
    rule::AlpertLogRule{T}
    offs::Vector{Int}
    Gs::Vector{BoundaryGeomCache{T}}
    Cs::Vector{AlpertCache{T}}
    parr::Vector{BoundaryPanelArrays{T}}
    Ntot::Int
end

"""
    _dlp_terms(TT,k,r,inn,invr,w) → d0,d1,d2,h0,h1

Evaluate one weighted off-diagonal DLP contribution and its first two
wavenumber derivatives.

## Description
For source-target distance `r`, oriented tangent numerator `inn`, and source
quadrature weight `w`, the contribution is

    D(k)=w (ik/2) inn H₁⁽¹⁾(kr)/r.

The function also returns the Hankel values needed by the SLP contribution so
that they do not need to be evaluated a second time.

## Arguments
* `TT`: Real scalar type used by the geometry and destination matrices.
* `k`: Real wavenumber.
* `r`: Source-target distance.
* `inn`: Oriented DLP numerator.
* `invr`: Reciprocal distance `1/r`.
* `w`: Source quadrature weight.

## Returns
* `d0`: DLP contribution `D(k)`.
* `d1`: First derivative `dD/dk`.
* `d2`: Second derivative `d²D/dk²`.
* `h0`: `H₀⁽¹⁾(kr)`.
* `h1`: `H₁⁽¹⁾(kr)`.
"""
@inline function _dlp_terms(TT,k,r,inn,invr,w)
    h0,h1=hankel_pair01(k*r)
    αD=Complex{TT}(0,k/2)
    d0=w*(αD*inn*h1*invr)
    d1=w*(Complex{TT}(0,one(TT)/2)*inn*k*h0)
    d2=w*(Complex{TT}(0,one(TT)/2)*inn*(h0-k*r*h1))
    return d0,d1,d2,h0,h1
end

"""
    _slp_terms(TT,k,r,s,w,h0,h1) → s0,s1,s2

Evaluate one weighted off-diagonal SLP contribution and its first two
wavenumber derivatives.

## Description
The weighted single-layer contribution is

    S(k)=w (i/2) H₀⁽¹⁾(kr) s,

where `s` is the source-point speed.

The Hankel values are supplied by the caller so that a preceding DLP evaluation
can be reused.

## Arguments
* `TT`: Real scalar type.
* `k`: Real wavenumber.
* `r`: Source-target distance.
* `s`: Source-point speed.
* `w`: Quadrature weight.
* `h0`: `H₀⁽¹⁾(kr)`.
* `h1`: `H₁⁽¹⁾(kr)`.

## Returns
* `s0`: SLP contribution `S(k)`.
* `s1`: First derivative `dS/dk`.
* `s2`: Second derivative `d²S/dk²`.
"""
@inline function _slp_terms(TT,k,r,s,w,h0,h1)
    αS=Complex{TT}(0,one(TT)/2)
    s0=w*(αS*h0*s)
    s1=w*(-Complex{TT}(0,one(TT)/2)*r*h1*s)
    s2=w*(Complex{TT}(0,one(TT)/2)*r*(h1-k*r*h0)*s/k)
    return s0,s1,s2
end

"""
    _wrap01(u) → v

Wrap a real parameter to the half-open interval `[0,1)`.

## Arguments
* `u`: Real parameter value.

## Returns
* `v`: Periodically equivalent value satisfying `0 ≤ v < 1`.
"""
@inline function _wrap01(u::T) where {T<:Real}
    v=mod(u,one(T))
    v<zero(T) ? v+one(T) : v
end

"""
    wrap_diff(t) → Δ

Wrap an angular difference to the interval `[-π,π)`.

## Arguments
* `t`: Angular difference.

## Returns
* `Δ`: Wrapped angular difference.
"""
@inline function wrap_diff(t::T) where {T<:Real}
    return mod(t+T(pi),T(two_pi))-T(pi)
end

"""
    _panel_sigma_wrap(σ) → σw

Wrap a panel computational coordinate to `[0,1)`.

## Arguments
* `σ`: Computational panel coordinate.

## Returns
* `σw`: Wrapped coordinate satisfying `0 ≤ σw < 1`.
"""
@inline function _panel_sigma_wrap(σ::T) where {T<:Real}
    v=mod(σ,one(T))
    v<zero(T) ? v+one(T) : v
end

"""
    _lagrange_weights(ξ,nodes) → w

Compute Lagrange interpolation weights at `ξ`.

## Description
For interpolation nodes `x_j`, the returned weights satisfy

    ℓ_j(ξ)=∏_{l≠j}(ξ-x_l)/(x_j-x_l),

so that

    f(ξ)≈∑_j ℓ_j(ξ)f(x_j).

## Arguments
* `ξ`: Evaluation point.
* `nodes`: Interpolation nodes.

## Returns
* `w`: Lagrange basis weights evaluated at `ξ`.
"""
@inline function _lagrange_weights(ξ::T,nodes::AbstractVector{T}) where {T<:Real}
    m=length(nodes)
    w=Vector{T}(undef,m)
    @inbounds for j in 1:m
        num=one(T)
        den=one(T)
        xj=nodes[j]
        for l in 1:m
            l==j&&continue
            xl=nodes[l]
            num*=ξ-xl
            den*=xj-xl
        end
        w[j]=num/den
    end
    return w
end

"""
    _alpert_interp_offsets_weights(ξ,ninterp) → offs,wt

Construct a periodic interpolation stencil for an off-grid Alpert node.

## Arguments
* `ξ`: Noninteger displacement in native-grid index units.
* `ninterp`: Number of interpolation nodes.

## Returns
* `offs`: Integer stencil offsets.
* `wt`: Lagrange interpolation weights on those offsets.
"""
@inline function _alpert_interp_offsets_weights(ξ::T,ninterp::Int) where {T<:Real}
    j0=floor(Int,ξ-T(ninterp)/2+one(T))
    offs=collect(j0:(j0+ninterp-1))
    wt=_lagrange_weights(ξ,T.(offs))
    return offs,wt
end

"""
    _local_offsets(p) → offsets

Construct the centered even interpolation stencil used for open panels.

## Arguments
* `p`: Even interpolation-stencil size.

## Returns
* `offsets`: Integer local offsets of length `p`.

## Throws
* `ErrorException`: If `p` is odd.
"""
@inline function _local_offsets(p::Int)
    iseven(p)||error("Interpolation stencil size p must be even.")
    q=p÷2
    return collect(-(q-1):q)
end

"""
    _periodic_orientation_sign(ts) → sign

Determine the orientation of a periodic parameter sequence.

## Description
The wrapped difference between the first two stored parameter values determines
whether the stored node ordering follows increasing or decreasing periodic
parameter.

## Arguments
* `ts`: Periodic parameter nodes.

## Returns
* `sign`: `1` for increasing orientation and `-1` for decreasing orientation.
"""
@inline function _periodic_orientation_sign(ts::AbstractVector{T}) where {T<:Real}
    N=length(ts)
    N<2&&return 1
    Δ=wrap_diff(ts[mod1(2,N)]-ts[1])
    return Δ>=zero(T) ? 1 : -1
end

"""
    _dinner(dx,dy,tx,ty) → inn

Compute the oriented tangent numerator used by the DLP kernel.

## Arguments
* `dx`, `dy`: Source-to-target displacement components.
* `tx`, `ty`: Source tangent components.

## Returns
* `inn`: Scalar `ty*dx-tx*dy`.
"""
@inline function _dinner(dx,dy,tx,ty)
    return ty*dx-tx*dy
end

"""
    _speed(v) → s

Compute a numerically safe tangent speed.

## Arguments
* `v`: Two-dimensional tangent vector.

## Returns
* `s`: Euclidean norm of `v`, bounded below by `eps(T)`.
"""
@inline function _speed(v::SVector{2,T}) where {T<:Real}
    s=hypot(v[1],v[2])
    return max(s,eps(T))
end

"""
    _panel_arrays(pts) → X,Y,dX,dY,s

Extract flattened coordinate and tangent arrays from a boundary discretization.

## Arguments
* `pts`: Boundary discretization.

## Returns
* `X`: x-coordinates.
* `Y`: y-coordinates.
* `dX`: x-components of the tangent.
* `dY`: y-components of the tangent.
* `s`: Tangent speeds.
"""
@inline function _panel_arrays(pts::BoundaryPoints{T}) where {T<:Real}
    X=getindex.(pts.xy,1)
    Y=getindex.(pts.xy,2)
    dX=getindex.(pts.tangent,1)
    dY=getindex.(pts.tangent,2)
    s=@. hypot(dX,dY)
    return X,Y,dX,dY,s
end

"""
    _add_naive_panel_block!(A,gi,xi,yi,rb,pb,Pb,k,αD,αS,ik;skip_pred=(j->false)) → A

Add one smooth off-panel CFIE block contribution to a fixed target row.

## Description
For each source node, the function adds

    -(D_ij+ikS_ij)

using ordinary quadrature. It is intended for interactions that require no
Alpert self-correction.

## Arguments
* `A`: Global destination matrix.
* `gi`: Global target-row index.
* `xi`, `yi`: Target coordinates.
* `rb`: Global source-block index range.
* `pb`: Source boundary discretization.
* `Pb`: Flattened source geometry arrays.
* `k`: Real wavenumber.
* `αD`: DLP prefactor.
* `αS`: SLP prefactor.
* `ik`: Combined-field factor `ik`.

## Keyword arguments
* `skip_pred`: Optional predicate returning `true` for source indices that
  should be omitted.

## Returns
* `A`: Matrix modified in place.
"""
function _add_naive_panel_block!(A::AbstractMatrix{Complex{T}},gi::Int,xi::T,yi::T,rb::UnitRange{Int},pb::BoundaryPoints{T},Pb::BoundaryPanelArrays{T},k::T,αD::Complex{T},αS::Complex{T},ik::Complex{T};skip_pred=(j->false)) where {T<:Real}
    Xb=Pb.X;Yb=Pb.Y;dXb=Pb.dX;dYb=Pb.dY;sb=Pb.s;wb=pb.ws
    Nb=length(Xb)
    @inbounds for j in 1:Nb
        skip_pred(j)&&continue
        dx=xi-Xb[j];dy=yi-Yb[j]
        r2=muladd(dx,dx,dy*dy)
        r2<=(eps(T))^2&&continue
        r=sqrt(r2);invr=inv(r)
        inn=_dinner(dx,dy,dXb[j],dYb[j])
        h0,h1=hankel_pair01(k*r)
        wd=wb[j];ws=wd*sb[j]
        A[gi,rb[j]]-=wd*(αD*inn*h1*invr)+ik*(ws*(αS*h0))
    end
    return A
end

"""
    _add_self_panel_alpert_correction!(A,gi,xi,yi,i,ra,Ca,hσ,k,αD,αS,ik,rule) → A

Add the Alpert correction contribution for one target on an open panel.

## Arguments
* `A`: Global destination matrix.
* `gi`: Global target index.
* `xi`, `yi`: Target coordinates.
* `i`: Local target index.
* `ra`: Global range of the current panel.
* `Ca`: Open-panel Alpert correction cache.
* `hσ`: Base computational panel spacing.
* `k`: Real wavenumber.
* `αD`: DLP prefactor.
* `αS`: SLP prefactor.
* `ik`: Combined-field factor.
* `rule`: Alpert logarithmic quadrature rule.

## Returns
* `A`: Matrix modified in place.
"""
function _add_self_panel_alpert_correction!(A::AbstractMatrix{Complex{T}},gi::Int,xi::T,yi::T,i::Int,ra::UnitRange{Int},Ca::AlpertSmoothPanelCache{T},hσ::T,k::T,αD::Complex{T},αS::Complex{T},ik::Complex{T},rule::AlpertLogRule{T}) where {T<:Real}
    jcorr=rule.j
    @inbounds for p in 1:jcorr
        fac=hσ*rule.w[p]
        dx=xi-Ca.xp[p,i];dy=yi-Ca.yp[p,i]
        r2=muladd(dx,dx,dy*dy)
        if isfinite(r2)&&r2>(eps(T))^2
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
        if isfinite(r2)&&r2>(eps(T))^2
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

"""
    _panel_interp_midpoint_data(σ,hσ,N,p) → idx,wt

Construct an endpoint-aware interpolation stencil on an open midpoint panel.

## Arguments
* `σ`: Off-grid computational coordinate.
* `hσ`: Native midpoint spacing.
* `N`: Number of panel nodes.
* `p`: Even interpolation-stencil size.

## Returns
* `idx`: Native panel indices used by the interpolation stencil.
* `wt`: Corresponding Lagrange interpolation weights.

## Throws
* `ErrorException`: If `p` is odd or exceeds `N`.
"""
@inline function _panel_interp_midpoint_data(σ::T,hσ::T,N::Int,p::Int) where {T<:Real}
    iseven(p)||error("p must be even.")
    p<=N||error("p must satisfy p <= N.")
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

"""
    _eval_open_panel_geom_exact(crv,u) → x,y,tx,ty,s

Evaluate the exact point, tangent, and speed of an open panel.

## Arguments
* `crv`: Underlying curve.
* `u`: Curve parameter.

## Returns
* `x`, `y`: Curve coordinates.
* `tx`, `ty`: Tangent components.
* `s`: Tangent speed.
"""
@inline function _eval_open_panel_geom_exact(crv,u::T) where {T<:Real}
    q=curve(crv,u)
    t=tangent(crv,u)
    s=hypot(t[1],t[2])
    return q[1],q[2],t[1],t[2],s
end

"""
    _build_alpert_periodic_cache(solver,crv,pts,rule,ord) → cache

Build the Alpert self-correction cache for one periodic boundary component.

## Description
For every target and Alpert correction abscissa, this function evaluates the
positive and negative periodic correction points, their tangents, speeds,
distances, DLP numerators, and periodic interpolation stencils.

The interpolation stencil size is chosen as

    ninterp=ord+3.

## Arguments
* `solver`: CFIE-Alpert solver.
* `crv`: Underlying smooth periodic curve.
* `pts`: Periodic boundary discretization.
* `rule`: Alpert logarithmic quadrature rule.
* `ord`: Formal Alpert order.

## Returns
* `cache`: [`AlpertPeriodicCache`](@ref).
"""
function _build_alpert_periodic_cache(solver::CFIE_alpert{T},crv::C,pts::BoundaryPoints{T},rule::AlpertLogRule{T},ord::Int) where {T<:Real,C<:AbsCurve}
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
            xp[p,i]=xpi;yp[p,i]=ypi;txp[p,i]=txpi;typ[p,i]=typi;sp[p,i]=hypot(txpi,typi)
            dx=xi-xpi;dy=yi-ypi
            r2=muladd(dx,dx,dy*dy)
            if isfinite(r2)&&r2>(eps(T))^2
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
            xm[p,i]=xmi;ym[p,i]=ymi;txm[p,i]=txmi;tym[p,i]=tymi;sm[p,i]=hypot(txmi,tymi)
            dx=xi-xmi;dy=yi-ymi
            r2=muladd(dx,dx,dy*dy)
            if isfinite(r2)&&r2>(eps(T))^2
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

"""
    _build_alpert_smooth_panel_cache(solver,crv,pts,rule,p) → cache

Build the Alpert self-correction cache for one open smooth panel.

## Description
For each target midpoint and each Alpert correction abscissa, this function:
1. displaces the computational coordinate by `±Δσ`,
2. maps the displaced coordinate through the panel grading map,
3. evaluates the exact panel geometry,
4. transforms the tangent by the grading Jacobian,
5. computes the source-target geometry,
6. constructs an endpoint-aware interpolation stencil.

## Arguments
* `solver`: CFIE-Alpert solver.
* `crv`: Underlying open curve.
* `pts`: Open-panel boundary discretization.
* `rule`: Alpert logarithmic quadrature rule.
* `p`: Even interpolation-stencil size.

## Returns
* `cache`: [`AlpertSmoothPanelCache`](@ref).
"""
function _build_alpert_smooth_panel_cache(solver::CFIE_alpert{T},crv,pts::BoundaryPoints{T},rule::AlpertLogRule{T},p::Int) where {T<:Real}
    iseven(p)||error("Smooth-panel Alpert interpolation stencil size p must be even.")
    N=length(pts.xy)
    p<=N||error("Smooth-panel Alpert interpolation stencil size p must satisfy p <= N.")
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
            if isfinite(r2)&&r2>(eps(T))^2
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
            if isfinite(r2)&&r2>(eps(T))^2
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

"""
    _build_alpert_component_cache(solver,crv,pts,rule,ord) → cache

Build the appropriate Alpert self-correction cache for one boundary object.

## Arguments
* `solver`: CFIE-Alpert solver.
* `crv`: Underlying geometric curve.
* `pts`: Boundary discretization.
* `rule`: Alpert logarithmic quadrature rule.
* `ord`: Formal Alpert order.

## Returns
* `cache`: [`AlpertPeriodicCache`](@ref) when `pts.is_periodic` is true,
  otherwise [`AlpertSmoothPanelCache`](@ref).
"""
function _build_alpert_component_cache(solver::CFIE_alpert{T},crv,pts::BoundaryPoints{T},rule::AlpertLogRule{T},ord::Int) where {T<:Real}
    if pts.is_periodic
        return _build_alpert_periodic_cache(solver,crv,pts,rule,ord)
    else
        pinterp=max(8,ord+3)
        iseven(pinterp)||(pinterp+=1)
        pinterp=min(pinterp,length(pts.xy))
        isodd(pinterp)&&(pinterp-=1)
        pinterp>=4||error("Interpolation stencil too small for smooth-panel Alpert cache.")
        return _build_alpert_smooth_panel_cache(solver,crv,pts,rule,pinterp)
    end
end

"""
    _assemble_self_alpert_periodic!(A,pts,G,C,row_range,k,rule;multithreaded=true) → A

Assemble one periodic Alpert-corrected CFIE self-interaction block.

## Description
The assembly consists of:
1. adding the identity,
2. adding the ordinary periodic quadrature for all regular off-diagonal pairs,
3. removing the inaccurate near-band trapezoidal contribution,
4. replacing that near band by the Alpert correction-node contribution.

## Arguments
* `A`: Global destination matrix.
* `pts`: Periodic boundary discretization.
* `G`: Geometry cache for the component.
* `C`: Periodic Alpert correction cache.
* `row_range`: Global index range associated with the component.
* `k`: Real wavenumber.
* `rule`: Alpert logarithmic quadrature rule.

## Keyword arguments
* `multithreaded`: Whether to thread the target-row loop.

## Returns
* `A`: Matrix modified in place.
"""
function _assemble_self_alpert_periodic!(A::AbstractMatrix{Complex{T}},pts::BoundaryPoints{T},G::BoundaryGeomCache{T},C::AlpertPeriodicCache{T},row_range::UnitRange{Int},k::T,rule::AlpertLogRule{T};multithreaded::Bool=true) where {T<:Real}
    αD=Complex{T}(0,k/2);αS=Complex{T}(0,one(T)/2);ik=Complex{T}(0,k)
    R=G.R;invR=G.invR;inner=G.inner;speed=G.speed
    rp=C.rp;rm=C.rm;innp=C.innp;innm=C.innm;sp=C.sp;sm=C.sm
    offsp=C.offsp;wtp=C.wtp;offsm=C.offsm;wtm=C.wtm
    r0=first(row_range)-1
    N=length(pts);h=pts.ws[1];a=rule.a;jcorr=rule.j;ninterp=C.ninterp
    @use_threads multithreading=(multithreaded&&N>=16) for i in 1:N
        gi=r0+i
        A[gi,gi]+=one(Complex{T})
        @inbounds for j in 1:N
            j==i&&continue
            r=R[i,j]
            h0,h1=hankel_pair01(k*r)
            A[gi,r0+j]-=h*(αD*inner[i,j]*h1*invR[i,j])+ik*(h*(αS*h0*speed[j]))
        end
        @inbounds for s in (-a+1):(a-1)
            s==0&&continue
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

"""
    _assemble_self_alpert_periodic_deriv!(A,A1,A2,pts,G,C,P,row_range,k,rule;multithreaded=true) → A,A1,A2

Assemble one periodic Alpert self-block and its first two wavenumber derivatives.

## Arguments
* `A`: Destination matrix for the CFIE block.
* `A1`: Destination matrix for the first derivative.
* `A2`: Destination matrix for the second derivative.
* `pts`: Periodic boundary discretization.
* `G`: Geometry cache.
* `C`: Periodic Alpert correction cache.
* `P`: Flattened boundary panel arrays.
* `row_range`: Global index range of the component.
* `k`: Real wavenumber.
* `rule`: Alpert logarithmic quadrature rule.

## Keyword arguments
* `multithreaded`: Whether to thread the target-row loop.

## Returns
* `A`: CFIE matrix modified in place.
* `A1`: First derivative modified in place.
* `A2`: Second derivative modified in place.
"""
function _assemble_self_alpert_periodic_deriv!(A::AbstractMatrix{Complex{T}},A1::AbstractMatrix{Complex{T}},A2::AbstractMatrix{Complex{T}},pts::BoundaryPoints{T},G::BoundaryGeomCache{T},C::AlpertPeriodicCache{T},P::BoundaryPanelArrays{T},row_range::UnitRange{Int},k::T,rule::AlpertLogRule{T};multithreaded::Bool=true) where {T<:Real}
    ik=Complex{T}(0,k)
    R=G.R;invR=G.invR;inner=G.inner;speed=G.speed
    rp=C.rp;rm=C.rm;innp=C.innp;innm=C.innm;sp=C.sp;sm=C.sm
    offsp=C.offsp;wtp=C.wtp;offsm=C.offsm;wtm=C.wtm
    N=length(P.X);h=pts.ws[1];a=rule.a;jcorr=rule.j;ninterp=C.ninterp
    @use_threads multithreading=(multithreaded&&N>=16) for i in 1:N
        gi=row_range[i]
        A[gi,gi]+=one(Complex{T})
        @inbounds for j in 1:N
            j==i&&continue
            gj=row_range[j]
            d0,d1,d2,h0,h1=_dlp_terms(T,k,R[i,j],inner[i,j],invR[i,j],h)
            s0,s1,s2=_slp_terms(T,k,R[i,j],speed[j],h,h0,h1)
            A[gi,gj]-=d0+ik*s0
            A1[gi,gj]-=d1+Complex{T}(0,1)*s0+ik*s1
            A2[gi,gj]-=d2+Complex{T}(0,2)*s1+ik*s2
        end
        @inbounds for m in (-a+1):(a-1)
            m==0&&continue
            j=mod1(i+m,N)
            gj=row_range[j]
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
                    gq=row_range[mod1(i+offsp[p,m],N)]
                    ww=wtp[p,m]
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
                    gq=row_range[mod1(i+offsm[p,m],N)]
                    ww=wtm[p,m]
                    A[gi,gq]-=(d0+ik*s0)*ww
                    A1[gi,gq]-=(d1+Complex{T}(0,1)*s0+ik*s1)*ww
                    A2[gi,gq]-=(d2+Complex{T}(0,2)*s1+ik*s2)*ww
                end
            end
        end
    end
    return A,A1,A2
end

"""
    _assemble_self_alpert_smooth_panel!(A,pts,G,C,row_range,k,rule;multithreaded=true) → A

Assemble one open-panel Alpert-corrected CFIE self-interaction block.

## Description
The ordinary midpoint rule is retained only outside the near band

    abs(j-i)<rule.a.

The near-band contribution is replaced by Alpert correction-node evaluations
and endpoint-aware interpolation.

## Arguments
* `A`: Global destination matrix.
* `pts`: Open-panel boundary discretization.
* `G`: Panel geometry cache.
* `C`: Open-panel Alpert cache.
* `row_range`: Global index range of the panel.
* `k`: Real wavenumber.
* `rule`: Alpert logarithmic quadrature rule.

## Keyword arguments
* `multithreaded`: Whether to thread the target-row loop.

## Returns
* `A`: Matrix modified in place.
"""
function _assemble_self_alpert_smooth_panel!(A::AbstractMatrix{Complex{T}},pts::BoundaryPoints{T},G::BoundaryGeomCache{T},C::AlpertSmoothPanelCache{T},row_range::UnitRange{Int},k::T,rule::AlpertLogRule{T};multithreaded::Bool=true) where {T<:Real}
    αD=Complex{T}(0,k/2)
    αS=Complex{T}(0,one(T)/2)
    ik=Complex{T}(0,k)
    w=pts.ws
    R=G.R
    invR=G.invR
    inner=G.inner
    speed=G.speed
    rp=C.rp
    rm=C.rm
    innp=C.innp
    innm=C.innm
    sp=C.sp
    sm=C.sm
    idxp=C.idxp
    wtp=C.wtp
    idxm=C.idxm
    wtm=C.wtm
    r0=first(row_range)-1
    N=length(pts)
    hσ=w[1]
    jcorr=rule.j
    @use_threads multithreading=(multithreaded&&N>=16) for i in 1:N
        gi=r0+i
        A[gi,gi]+=one(Complex{T})
        @inbounds for j in 1:N
            j==i&&continue
            abs(j-i)<rule.a&&continue
            r=R[i,j]
            h0,h1=hankel_pair01(k*r)
            A[gi,r0+j]-=w[j]*(αD*inner[i,j]*h1*invR[i,j])+ik*((w[j]*speed[j])*(αS*h0))
        end
        @inbounds for p in 1:jcorr
            fac=hσ*rule.w[p]
            r=rp[p,i]
            if isfinite(r)
                h0,h1=hankel_pair01(k*r)
                coeff=-(fac*(αD*innp[p,i]*h1/r))-ik*(fac*(αS*h0*sp[p,i]))
                for m in axes(idxp,3)
                    A[gi,r0+idxp[p,i,m]]+=coeff*wtp[p,i,m]
                end
            end
            r=rm[p,i]
            if isfinite(r)
                h0,h1=hankel_pair01(k*r)
                coeff=-(fac*(αD*innm[p,i]*h1/r))-ik*(fac*(αS*h0*sm[p,i]))
                for m in axes(idxm,3)
                    A[gi,r0+idxm[p,i,m]]+=coeff*wtm[p,i,m]
                end
            end
        end
    end
    return A
end

"""
    _assemble_self_alpert_smooth_panel_deriv!(A,A1,A2,pts,G,C,P,row_range,k,rule;multithreaded=true) → A,A1,A2

Assemble one open-panel Alpert self-block and its first two wavenumber
derivatives.

## Arguments
* `A`: Destination matrix for the CFIE block.
* `A1`: Destination matrix for the first derivative.
* `A2`: Destination matrix for the second derivative.
* `pts`: Open-panel boundary discretization.
* `G`: Geometry cache.
* `C`: Open-panel Alpert cache.
* `P`: Flattened panel arrays.
* `row_range`: Global panel index range.
* `k`: Real wavenumber.
* `rule`: Alpert logarithmic quadrature rule.

## Keyword arguments
* `multithreaded`: Whether to thread the target-row loop.

## Returns
* `A`: CFIE matrix modified in place.
* `A1`: First derivative modified in place.
* `A2`: Second derivative modified in place.
"""
function _assemble_self_alpert_smooth_panel_deriv!(A::AbstractMatrix{Complex{T}},A1::AbstractMatrix{Complex{T}},A2::AbstractMatrix{Complex{T}},pts::BoundaryPoints{T},G::BoundaryGeomCache{T},C::AlpertSmoothPanelCache{T},P::BoundaryPanelArrays{T},row_range::UnitRange{Int},k::T,rule::AlpertLogRule{T};multithreaded::Bool=true) where {T<:Real}
    ik=Complex{T}(0,k)
    w=pts.ws
    R=G.R
    invR=G.invR
    inner=G.inner
    speed=G.speed
    rp=C.rp
    rm=C.rm
    innp=C.innp
    innm=C.innm
    sp=C.sp
    sm=C.sm
    idxp=C.idxp
    wtp=C.wtp
    idxm=C.idxm
    wtm=C.wtm
    N=length(P.X)
    hσ=w[1]
    jcorr=rule.j
    @use_threads multithreading=(multithreaded&&N>=16) for i in 1:N
        gi=row_range[i]
        A[gi,gi]+=one(Complex{T})
        @inbounds for j in 1:N
            j==i&&continue
            abs(j-i)<rule.a&&continue
            gj=row_range[j]
            d0,d1,d2,h0,h1=_dlp_terms(T,k,R[i,j],inner[i,j],invR[i,j],w[j])
            s0,s1,s2=_slp_terms(T,k,R[i,j],speed[j],w[j],h0,h1)
            A[gi,gj]-=d0+ik*s0
            A1[gi,gj]-=d1+Complex{T}(0,1)*s0+ik*s1
            A2[gi,gj]-=d2+Complex{T}(0,2)*s1+ik*s2
        end
        @inbounds for p in 1:jcorr
            fac=hσ*rule.w[p]
            r=rp[p,i]
            if isfinite(r)
                d0,d1,d2,h0,h1=_dlp_terms(T,k,r,innp[p,i],inv(r),fac)
                s0,s1,s2=_slp_terms(T,k,r,sp[p,i],fac,h0,h1)
                for m in axes(idxp,3)
                    gq=row_range[idxp[p,i,m]]
                    ww=wtp[p,i,m]
                    A[gi,gq]-=(d0+ik*s0)*ww
                    A1[gi,gq]-=(d1+Complex{T}(0,1)*s0+ik*s1)*ww
                    A2[gi,gq]-=(d2+Complex{T}(0,2)*s1+ik*s2)*ww
                end
            end
            r=rm[p,i]
            if isfinite(r)
                d0,d1,d2,h0,h1=_dlp_terms(T,k,r,innm[p,i],inv(r),fac)
                s0,s1,s2=_slp_terms(T,k,r,sm[p,i],fac,h0,h1)
                for m in axes(idxm,3)
                    gq=row_range[idxm[p,i,m]]
                    ww=wtm[p,i,m]
                    A[gi,gq]-=(d0+ik*s0)*ww
                    A1[gi,gq]-=(d1+Complex{T}(0,1)*s0+ik*s1)*ww
                    A2[gi,gq]-=(d2+Complex{T}(0,2)*s1+ik*s2)*ww
                end
            end
        end
    end
    return A,A1,A2
end

"""
    _assemble_self_alpert!(solver,A,pts,G,C,row_range,k,rule;multithreaded=true) → A

Dispatch to the periodic or open-panel Alpert self-block assembler.

## Arguments
* `solver`: CFIE-Alpert solver.
* `A`: Global destination matrix.
* `pts`: Boundary discretization.
* `G`: Geometry cache.
* `C`: Alpert correction cache.
* `row_range`: Global index range.
* `k`: Real wavenumber.
* `rule`: Alpert logarithmic quadrature rule.

## Keyword arguments
* `multithreaded`: Whether to thread the self-block assembly.

## Returns
* `A`: Matrix modified in place.
"""
function _assemble_self_alpert!(solver::CFIE_alpert{T},A::AbstractMatrix{Complex{T}},pts::BoundaryPoints{T},G::BoundaryGeomCache{T},C,row_range::UnitRange{Int},k::T,rule::AlpertLogRule{T};multithreaded::Bool=true) where {T<:Real}
    pts.is_periodic ?
        _assemble_self_alpert_periodic!(A,pts,G,C,row_range,k,rule;multithreaded=multithreaded) :
        _assemble_self_alpert_smooth_panel!(A,pts,G,C,row_range,k,rule;multithreaded=multithreaded)
end

"""
    _assemble_all_offpanel_naive!(A,pts,offs,parr,k;multithreaded=true) → A

Assemble all interactions between distinct panels/components.

## Arguments
* `A`: Global destination matrix.
* `pts`: Boundary discretizations.
* `offs`: Global panel/component offsets.
* `parr`: Flattened panel geometry arrays.
* `k`: Real wavenumber.

## Keyword arguments
* `multithreaded`: Whether to thread target-row loops.

## Returns
* `A`: Matrix modified in place.
"""
function _assemble_all_offpanel_naive!(A::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPoints{T}},offs::Vector{Int},parr::Vector{BoundaryPanelArrays{T}},k::T;multithreaded::Bool=true) where {T<:Real}
    αD=Complex{T}(0,k/2);αS=Complex{T}(0,one(T)/2);ik=Complex{T}(0,k)
    for aidx in eachindex(pts)
        ra=offs[aidx]:(offs[aidx+1]-1);r0a=first(ra)-1;Pa=parr[aidx]
        Xa=Pa.X;Ya=Pa.Y;Na=length(Xa)
        for bidx in eachindex(pts)
            bidx==aidx&&continue
            pb=pts[bidx];rb=offs[bidx]:(offs[bidx+1]-1);r0b=first(rb)-1;Pb=parr[bidx]
            Xb=Pb.X;Yb=Pb.Y;dXb=Pb.dX;dYb=Pb.dY;sb=Pb.s;wb=pb.ws;Nb=length(Xb)
            @use_threads multithreading=(multithreaded&&Na>=16) for i in 1:Na
                gi=r0a+i
                xi=Xa[i];yi=Ya[i]
                @inbounds for j in 1:Nb
                    dx=xi-Xb[j];dy=yi-Yb[j]
                    r2=muladd(dx,dx,dy*dy)
                    r2<=(eps(T))^2&&continue
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

"""
    _assemble_all_offpanel_naive_deriv!(A,A1,A2,pts,offs,parr,k;multithreaded=true) → A,A1,A2

Assemble all smooth off-panel interactions and their first two wavenumber
derivatives.

## Arguments
* `A`: Destination matrix.
* `A1`: First derivative destination.
* `A2`: Second derivative destination.
* `pts`: Boundary discretizations.
* `offs`: Global offsets.
* `parr`: Flattened panel geometry arrays.
* `k`: Real wavenumber.

## Keyword arguments
* `multithreaded`: Whether to thread target-row loops.

## Returns
* `A`: Matrix modified in place.
* `A1`: First derivative modified in place.
* `A2`: Second derivative modified in place.
"""
function _assemble_all_offpanel_naive_deriv!(A::AbstractMatrix{Complex{T}},A1::AbstractMatrix{Complex{T}},A2::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPoints{T}},offs::Vector{Int},parr::Vector{BoundaryPanelArrays{T}},k::T;multithreaded::Bool=true) where {T<:Real}
    ik=Complex{T}(0,k)
    for aidx in eachindex(pts)
        ra=offs[aidx]:(offs[aidx+1]-1);r0a=first(ra)-1;Pa=parr[aidx]
        Xa=Pa.X;Ya=Pa.Y;Na=length(Xa)
        for bidx in eachindex(pts)
            bidx==aidx&&continue
            pb=pts[bidx];rb=offs[bidx]:(offs[bidx+1]-1);r0b=first(rb)-1;Pb=parr[bidx]
            Xb=Pb.X;Yb=Pb.Y;dXb=Pb.dX;dYb=Pb.dY;sb=Pb.s;wb=pb.ws;Nb=length(Xb)
            @use_threads multithreading=(multithreaded&&Na>=16) for i in 1:Na
                gi=r0a+i
                xi=Xa[i];yi=Ya[i]
                @inbounds for j in 1:Nb
                    dx=xi-Xb[j];dy=yi-Yb[j]
                    r2=muladd(dx,dx,dy*dy)
                    r2<=(eps(T))^2&&continue
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

"""
    build_cfie_alpert_workspace(solver,pts) → ws

Build a reusable CFIE-Alpert workspace.

## Description
The workspace builder constructs:
* the selected Alpert logarithmic rule,
* global panel/component offsets,
* pairwise geometry caches,
* periodic or open-panel Alpert correction caches,
* flattened panel geometry arrays.

The geometric boundary description is flattened when necessary so that each
entry of `pts` remains aligned with the corresponding underlying curve segment.

## Arguments
* `solver`: CFIE-Alpert solver.
* `pts`: Boundary discretizations.

## Returns
* `ws`: [`CFIEAlpertWorkspace`](@ref).
"""
function build_cfie_alpert_workspace(solver::CFIE_alpert{T},pts::Vector{BoundaryPoints{T}}) where {T<:Real}
    rule=alpert_log_rule(T,solver.alpert_order)
    offs=component_offsets(pts)
    Gs=[boundary_geom_cache(p) for p in pts]
    boundary=solver.billiard.full_boundary
    flat_boundary=boundary[1] isa AbstractVector ? reduce(vcat,boundary) : boundary
    Cs=Vector{AlpertCache{T}}(undef,length(pts))
    @inbounds for a in eachindex(pts)
        Cs[a]=_build_alpert_component_cache(solver,flat_boundary[a],pts[a],rule,solver.alpert_order)
    end
    parr=[_boundary_panel_arrays_cache(p) for p in pts]
    return CFIEAlpertWorkspace(rule,offs,Gs,Cs,parr,offs[end]-1)
end

"""
    _construct_matrices_cached!(A,pts,ws,k;multithreaded=true) → A

Assemble the complete CFIE-Alpert matrix from a cached workspace.

## Description
Each same-panel block is assembled using the appropriate Alpert correction.
Interactions between distinct panels/components are smooth and are assembled
with ordinary quadrature.

## Arguments
* `A`: Preallocated destination matrix.
* `pts`: Boundary discretizations.
* `ws`: Cached CFIE-Alpert workspace.
* `k`: Real wavenumber.

## Keyword arguments
* `multithreaded`: Whether to thread low-level assembly loops.

## Returns
* `A`: Assembled matrix.
"""
@inline function _construct_matrices_cached!(A::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPoints{T}},ws::CFIEAlpertWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}
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

"""
    _construct_matrices_deriv_cached!(A,A1,A2,pts,ws,k;multithreaded=true) → A,A1,A2

Assemble the complete CFIE-Alpert matrix and its first two wavenumber
derivatives from a cached workspace.

## Arguments
* `A`: Destination matrix.
* `A1`: First derivative destination.
* `A2`: Second derivative destination.
* `pts`: Boundary discretizations.
* `ws`: Cached CFIE-Alpert workspace.
* `k`: Real wavenumber.

## Keyword arguments
* `multithreaded`: Whether to thread low-level assembly loops.

## Returns
* `A`: CFIE-Alpert matrix.
* `A1`: First derivative.
* `A2`: Second derivative.
"""
@inline function _construct_matrices_deriv_cached!(A::AbstractMatrix{Complex{T}},A1::AbstractMatrix{Complex{T}},A2::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPoints{T}},ws::CFIEAlpertWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}
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

"""
    construct_matrices!(solver::CFIE_alpert,A,pts,ws,k;multithreaded=true)
    construct_matrices!(solver::CFIE_alpert,A,pts,k;multithreaded=true)
    construct_matrices(solver::CFIE_alpert,pts,ws,k;multithreaded=true)
    construct_matrices(solver::CFIE_alpert,pts,k;multithreaded=true)
    construct_matrices!(solver::CFIE_alpert,A,A1,A2,pts,ws,k;multithreaded=true)
    construct_matrices!(solver::CFIE_alpert,A,A1,A2,pts,k;multithreaded=true)

High-level CFIE-Alpert matrix assembly interface.

## Arguments
* `solver`: CFIE-Alpert solver.
* `A`: Destination matrix.
* `A1`: Optional first derivative destination.
* `A2`: Optional second derivative destination.
* `pts`: Boundary discretizations.
* `ws`: Optional cached CFIE-Alpert workspace.
* `basis`: Placeholder basis required by the common solver interface.
* `k`: Real wavenumber.

## Keyword arguments
* `multithreaded`: Whether to thread matrix assembly.

## Returns
* Matrix-only overloads return `A`.
* Derivative overloads return `(A,A1,A2)`.
"""
function construct_matrices!(solver::CFIE_alpert{T},A::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPoints{T}},ws::CFIEAlpertWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}
    return _construct_matrices_cached!(A,pts,ws,k;multithreaded=multithreaded)
end

function construct_matrices!(solver::CFIE_alpert{T},A::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPoints{T}},k::T;multithreaded::Bool=true) where {T<:Real}
    ws=build_cfie_alpert_workspace(solver,pts)
    return _construct_matrices_cached!(A,pts,ws,k;multithreaded=multithreaded)
end

function construct_matrices(solver::CFIE_alpert{T},pts::Vector{BoundaryPoints{T}},ws::CFIEAlpertWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}
    A=Matrix{Complex{T}}(undef,ws.Ntot,ws.Ntot)
    _construct_matrices_cached!(A,pts,ws,k;multithreaded=multithreaded)
    return A
end

function construct_matrices(solver::CFIE_alpert{T},pts::Vector{BoundaryPoints{T}},k::T;multithreaded::Bool=true) where {T<:Real}
    ws=build_cfie_alpert_workspace(solver,pts)
    return construct_matrices(solver,pts,ws,k;multithreaded=multithreaded)
end

function construct_matrices!(solver::CFIE_alpert{T},A::AbstractMatrix{Complex{T}},A1::AbstractMatrix{Complex{T}},A2::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPoints{T}},ws::CFIEAlpertWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}
    return _construct_matrices_deriv_cached!(A,A1,A2,pts,ws,k;multithreaded=multithreaded)
end

function construct_matrices!(solver::CFIE_alpert{T},A::AbstractMatrix{Complex{T}},A1::AbstractMatrix{Complex{T}},A2::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPoints{T}},k::T;multithreaded::Bool=true) where {T<:Real}
    ws=build_cfie_alpert_workspace(solver,pts)
    return _construct_matrices_deriv_cached!(A,A1,A2,pts,ws,k;multithreaded=multithreaded)
end

function construct_matrices!(solver::CFIE_alpert{T},basis::AbstractHankelBasis,A::AbstractMatrix{Complex{T}},A1::AbstractMatrix{Complex{T}},A2::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPoints{T}},ws::CFIEAlpertWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}
    return _construct_matrices_deriv_cached!(A,A1,A2,pts,ws,k;multithreaded=multithreaded)
end

function construct_matrices!(solver::CFIE_alpert{T},basis::AbstractHankelBasis,A::AbstractMatrix{Complex{T}},A1::AbstractMatrix{Complex{T}},A2::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPoints{T}},k::T;multithreaded::Bool=true) where {T<:Real}
    ws=build_cfie_alpert_workspace(solver,pts)
    return _construct_matrices_deriv_cached!(A,A1,A2,pts,ws,k;multithreaded=multithreaded)
end

"""
    solve(solver::CFIE_alpert,basis,pts,k;multithreaded=true,use_krylov=true,which=:det_argmin)
    solve(solver::CFIE_alpert,basis,pts,ws,k;multithreaded=true,use_krylov=true,which=:det_argmin)
    solve(solver::CFIE_alpert,basis,A,pts,k;multithreaded=true,use_krylov=true,which=:det_argmin)
    solve(solver::CFIE_alpert,basis,A,pts,ws,k;multithreaded=true,use_krylov=true,which=:det_argmin)

Evaluate a scalar spectral diagnostic of the CFIE-Alpert matrix.

## Arguments
* `solver`: CFIE-Alpert solver.
* `basis`: Basis placeholder retained for API compatibility.
* `A`: Optional preallocated matrix.
* `pts`: Boundary discretizations.
* `ws`: Optional cached CFIE-Alpert workspace.
* `k`: Real wavenumber.

## Keyword arguments
* `multithreaded`: Whether to thread matrix construction.
* `use_krylov`: Forwarded to the common scalar reduction backend.
* `which`: Requested scalar diagnostic.

## Returns
A scalar spectral diagnostic whose interpretation depends on `which`.
"""
function solve(solver::CFIE_alpert,basis::Ba,pts::Vector{BoundaryPoints{T}},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det_argmin) where {T<:Real,Ba<:AbsBasis}
    A=construct_matrices(solver,pts,k;multithreaded=multithreaded)
    @svd_or_det_solve A use_krylov which MAX_BLAS_THREADS
end

function solve(solver::CFIE_alpert,basis::Ba,pts::Vector{BoundaryPoints{T}},ws::CFIEAlpertWorkspace{T},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det_argmin) where {T<:Real,Ba<:AbsBasis}
    A=Matrix{Complex{T}}(undef,ws.Ntot,ws.Ntot)
    @blas_1 construct_matrices!(solver,A,pts,ws,k;multithreaded=multithreaded)
    @svd_or_det_solve A use_krylov which MAX_BLAS_THREADS
end

function solve(solver::CFIE_alpert,basis::Ba,A::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPoints{T}},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det_argmin) where {T<:Real,Ba<:AbsBasis}
    @blas_1 construct_matrices!(solver,A,pts,k;multithreaded=multithreaded)
    @svd_or_det_solve A use_krylov which MAX_BLAS_THREADS
end

function solve(solver::CFIE_alpert,basis::Ba,A::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPoints{T}},ws::CFIEAlpertWorkspace{T},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det_argmin) where {T<:Real,Ba<:AbsBasis}
    @blas_1 construct_matrices!(solver,A,pts,ws,k;multithreaded=multithreaded)
    @svd_or_det_solve A use_krylov which MAX_BLAS_THREADS
end

"""
    solve_vect(solver::CFIE_alpert,billiard,basis,pts,k;multithreaded=true)
    solve_vect(solver::CFIE_alpert,billiard,basis,ks;multithreaded=true)
    solve_vect(solver::CFIE_alpert,basis,pts,ws,k;multithreaded=true)
    solve_vect(solver::CFIE_alpert,basis,A,pts,k;multithreaded=true)
    solve_vect(solver::CFIE_alpert,basis,A,pts,ws,k;multithreaded=true)

Compute the smallest singular value and corresponding right singular vector of
the CFIE-Alpert matrix.

## Description
These overloads use a dense SVD

    A=UΣV*,

and therefore return the actual smallest singular value and its right singular
vector.

## Arguments
* `solver`: CFIE-Alpert solver.
* `billiard`: Billiard geometry for convenience/batched overloads.
* `basis`: Basis placeholder retained for the common solver interface.
* `A`: Optional preallocated matrix.
* `pts`: Boundary discretizations.
* `ws`: Optional cached CFIE-Alpert workspace.
* `k`: Single real wavenumber.
* `ks`: Vector of real wavenumbers.

## Keyword arguments
* `multithreaded`: Whether to thread matrix assembly.

## Returns
Single-wavenumber overloads:
* `μ`: Smallest singular value.
* `u`: Corresponding right singular vector.

Batched overload:
* `us_all`: Right singular vectors.
* `pts_all`: Boundary discretizations used for the corresponding wavenumbers.
"""
function solve_vect(solver::CFIE_alpert,billiard::Bi,basis::Ba,pts::Vector{BoundaryPoints{T}},k;multithreaded::Bool=true) where {T<:Real,Ba<:AbsBasis,Bi<:AbsBilliard}
    @blas_1 A=construct_matrices(solver,pts,k;multithreaded=multithreaded)
    @blas_multi_then_1 MAX_BLAS_THREADS _,S,Vt=LAPACK.gesvd!('A','A',A)
    idx=findmin(S)[2]
    return S[idx],conj.(Vt[idx,:])
end

function solve_vect(solver::CFIE_alpert,billiard::Bi,basis::Ba,ks::Vector{T};multithreaded::Bool=true) where {T<:Real,Ba<:AbsBasis,Bi<:AbsBilliard}
    us_all=Vector{Vector{Complex{T}}}(undef,length(ks))
    pts_all=Vector{Vector{BoundaryPoints{T}}}(undef,length(ks))
    @showprogress "solve_vect CFIE Alpert" for i in eachindex(ks)
        pts=evaluate_points(solver,billiard,ks[i])
        _,u=solve_vect(solver,billiard,basis,pts,ks[i];multithreaded=multithreaded)
        us_all[i]=u
        pts_all[i]=pts
    end
    return us_all,pts_all
end

function solve_vect(solver::CFIE_alpert,basis::Ba,pts::Vector{BoundaryPoints{T}},ws::CFIEAlpertWorkspace{T},k;multithreaded::Bool=true) where {T<:Real,Ba<:AbsBasis}
    A=Matrix{Complex{T}}(undef,ws.Ntot,ws.Ntot)
    @blas_1 construct_matrices!(solver,A,pts,ws,k;multithreaded=multithreaded)
    @blas_multi_then_1 MAX_BLAS_THREADS _,S,Vt=LAPACK.gesvd!('A','A',A)
    idx=findmin(S)[2]
    return S[idx],conj.(Vt[idx,:])
end

function solve_vect(solver::CFIE_alpert,basis::Ba,A::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPoints{T}},k;multithreaded::Bool=true) where {T<:Real,Ba<:AbsBasis}
    @blas_1 construct_matrices!(solver,A,pts,k;multithreaded=multithreaded)
    @blas_multi_then_1 MAX_BLAS_THREADS _,S,Vt=LAPACK.gesvd!('A','A',A)
    idx=findmin(S)[2]
    return S[idx],conj.(Vt[idx,:])
end

function solve_vect(solver::CFIE_alpert,basis::Ba,A::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPoints{T}},ws::CFIEAlpertWorkspace{T},k;multithreaded::Bool=true) where {T<:Real,Ba<:AbsBasis}
    @blas_1 construct_matrices!(solver,A,pts,ws,k;multithreaded=multithreaded)
    @blas_multi_then_1 MAX_BLAS_THREADS _,S,Vt=LAPACK.gesvd!('A','A',A)
    idx=findmin(S)[2]
    return S[idx],conj.(Vt[idx,:])
end

# INTERNAL - for benchmarking and diagnostic purposes only; not part of the public API.
function solve_INFO(solver::CFIE_alpert,basis::Ba,pts::Vector{BoundaryPoints{T}},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det_argmin) where {T<:Real,Ba<:AbsBasis}
    Ntot=boundary_matrix_size(pts)
    A=Matrix{Complex{T}}(undef,Ntot,Ntot)
    t0=time()
    @info "Building boundary operator A..."
    @blas_1 construct_matrices!(solver,A,pts,k;multithreaded=multithreaded)
    any(isnan,A)&&error("NaN detected in system matrix A; check geometry and quadrature.")
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

# INTERNAL - cached-workspace benchmarking/diagnostic variant.
function solve_INFO(solver::CFIE_alpert,basis::Ba,pts::Vector{BoundaryPoints{T}},ws::CFIEAlpertWorkspace{T},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det_argmin) where {T<:Real,Ba<:AbsBasis}
    A=Matrix{Complex{T}}(undef,ws.Ntot,ws.Ntot)
    t0=time()
    @info "Building boundary operator A..."
    @blas_1 construct_matrices!(solver,A,pts,ws,k;multithreaded=multithreaded)
    any(isnan,A)&&error("NaN detected in system matrix A; check geometry and quadrature.")
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

"""
    estimate_cfie_alpert_cheb_rbounds(ws;pad=(0.95,1.05)) → rmin,rmax

Estimate a safe radial interval for Chebyshev interpolation from a complete
CFIE-Alpert workspace.

## Description
Alpert correction nodes can lie closer to a target than any pair of native
boundary nodes. A Chebyshev radial interval based only on the native geometry
can therefore miss distances that occur during Alpert matrix construction.

This helper scans:
1. same-block native geometry distances,
2. all inter-block native geometry distances,
3. all positive and negative Alpert correction-node distances.

The resulting interval is padded multiplicatively before being returned.

## Arguments
* `ws`: Cached CFIE-Alpert workspace.

## Keyword arguments
* `pad`: Multiplicative `(lower,upper)` padding applied to the detected interval.

## Returns
* `rmin`: Padded lower radial bound as `Float64`.
* `rmax`: Padded upper radial bound as `Float64`.
"""
function estimate_cfie_alpert_cheb_rbounds(ws::CFIEAlpertWorkspace{T};pad=(T(0.95),T(1.05))) where {T<:Real}
    rmin=typemax(T)
    rmax=zero(T)

    for G in ws.Gs
        R=G.R
        @inbounds for j in axes(R,2),i in axes(R,1)
            i==j&&continue
            r=R[i,j]
            if isfinite(r)&&r>eps(T)
                rmin=min(rmin,r)
                rmax=max(rmax,r)
            end
        end
    end

    parr=ws.parr
    nc=length(parr)
    @inbounds for a in 1:nc
        Pa=parr[a]
        Xa=Pa.X
        Ya=Pa.Y
        Na=length(Xa)
        for b in 1:nc
            b==a&&continue
            Pb=parr[b]
            Xb=Pb.X
            Yb=Pb.Y
            Nb=length(Xb)
            for j in 1:Nb,i in 1:Na
                dx=Xa[i]-Xb[j]
                dy=Ya[i]-Yb[j]
                r2=muladd(dx,dx,dy*dy)
                if isfinite(r2)&&r2>(eps(T))^2
                    r=sqrt(r2)
                    rmin=min(rmin,r)
                    rmax=max(rmax,r)
                end
            end
        end
    end

    for C in ws.Cs
        @inbounds for r in C.rp
            if isfinite(r)&&r>eps(T)
                rmin=min(rmin,r)
                rmax=max(rmax,r)
            end
        end
        @inbounds for r in C.rm
            if isfinite(r)&&r>eps(T)
                rmin=min(rmin,r)
                rmax=max(rmax,r)
            end
        end
    end

    @assert isfinite(rmin)&&rmax>zero(T)
    return Float64(pad[1]*rmin),Float64(pad[2]*rmax)
end