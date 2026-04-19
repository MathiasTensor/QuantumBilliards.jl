"""
    AlpertPeriodicCache{T}

Cache for Alpert near-singular self-panel correction on a closed periodic
boundary component.

This cache stores all geometry and interpolation data needed to apply the
Alpert hybrid Gauss-trapezoidal correction for a single periodic panel, that
is, a whole smooth closed boundary component sampled on a periodic midpoint
grid.

It is designed for the case where the boundary is represented by one periodic
parameter and the Alpert correction points lie at signed offsets ±ξ/N from
each target node. Because the panel is periodic, all interpolation back to the
native grid is also periodic, and the correction can be expressed using simple
integer offsets modulo N.

# Fields
- `xp, yp`:
  Coordinates of the positive-side Alpert correction nodes. Entry `(p,i)`
  corresponds to the point reached from target node `i` by moving a signed
  parameter distance `+ξ_p/N`.
- `txp, typ`:
  First derivative components of the boundary parameterization at those
  positive correction nodes, already rescaled to the computational parameter
  used by the periodic discretization.
- `sp`:
  Speed `|γ'(t)|` at the positive correction nodes.
- `xm, ym`:
  Coordinates of the negative-side correction nodes, using `-ξ_p/N`.
- `txm, tym`:
  Tangent components at the negative correction nodes.
- `sm`:
  Speed at the negative correction nodes.
- `rp, rm`:
  Distances from each target node to the positive/negative correction nodes.
  Entries are set to `Inf` when the distance is numerically degenerate, so the
  assembly code can skip them safely.
- `innp, innm`:
  Oriented DLP numerators at the correction nodes, i.e. the scalar
  `t_y*dx - t_x*dy` evaluated using the correction-node tangent. These are the
  geometric numerators entering the double-layer kernel.
- `offsp, offsm`:
  Integer periodic interpolation offsets for the positive/negative correction
  nodes. For each Alpert node index `p`, these determine which nearby periodic
  grid points receive the interpolated correction.
- `wtp, wtm`:
  Lagrange interpolation weights corresponding to `offsp` and `offsm`.
- `ninterp`:
  Interpolation stencil size used for each correction node.

# Applicability
Use this cache only when `pts.is_periodic == true`, i.e. for a single smooth
closed component discretized on a periodic midpoint grid.
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

This is the open-panel analogue of `AlpertPeriodicCache`. It is used when the
boundary is represented as one smooth segment with endpoints, rather than as a
closed periodic curve. In that case, the Alpert correction nodes are generated
in a graded panel parameter `σ ∈ (0,1)`, and interpolation back to the native
midpoint grid must use explicit local index stencils rather than periodic wrap.

# Fields
- `crv`:
  The underlying curve object for the panel. Stored so the cache remains tied
  to the exact geometric segment it was built from.
- `sig`:
  Computational midpoint nodes on the open panel. These are the panel-native
  coordinates from which correction offsets are formed.
- `xp, yp`, `xm, ym`:
  Positive and negative Alpert correction-node coordinates for every target
  node and every Alpert abscissa.
- `txp, typ`, `txm, tym`:
  Tangent components at those correction nodes, after applying the chain rule
  through the panel grading map.
- `sp`, `sm`:
  Speeds at the positive/negative correction nodes.
- `rp`, `rm`:
  Distances from the target node to the correction node.
- `innp`, `innm`:
  Oriented DLP numerators at the correction nodes.
- `idxp`, `idxm`:
  Explicit interpolation indices for the positive/negative correction nodes.
  Shape is `(jcorr, N, p)`, where `p` is the interpolation stencil size.
- `wtp`, `wtm`:
  Corresponding interpolation weights.
  Same shape as the index arrays.

# Difference from the periodic cache
The periodic cache can describe interpolation through integer offsets mod N.
The open-panel cache cannot, because endpoint proximity matters. Therefore it
stores the full interpolation indexes and their weights for each target
node separately.

# Applicability
Use this cache only when `pts.is_periodic == false`, i.e. for smooth panels in
the composite-panel Alpert discretization.
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

Workspace for repeated CFIE-Alpert assembly on a fixed discretized boundary.
This object collects all `k`-independent data needed by the Alpert-based CFIE
assembly.

# Fields
- `rule::AlpertLogRule{T}`:
  The chosen Alpert logarithmic quadrature rule, containing the correction
  abscissae, weights, correction width `a`, and number of correction nodes `j`.
- `offs::Vector{Int}`:
  Global component offsets in the assembled matrix. If the full system has
  total size `Ntot`, then component `a` occupies rows/cols
  `offs[a] : offs[a+1]-1`.
- `Gs::Vector{CFIEGeomCache{T}}`:
  One geometric cache per boundary component/panel, storing pairwise
  distances, inverse distances, kernel numerators, speeds, etc.
- `Cs::Vector{AlpertCache{T}}`:
  One Alpert self-correction cache per component, periodic or open-panel
  depending on the geometry.
- `parr::Vector{CFIEPanelArrays{T}}`:
  Flat coordinate/tangent arrays used for fast off-panel assembly.
- `Ntot::Int`:
  Total system size after concatenating all components.
"""
struct CFIEAlpertWorkspace{T<:Real}
    rule::AlpertLogRule{T}
    offs::Vector{Int}
    Gs::Vector{CFIEGeomCache{T}}
    Cs::Vector{AlpertCache{T}}
    parr::Vector{CFIEPanelArrays{T}}
    Ntot::Int
end

"""
    _dlp_terms(TT, k, r, inn, invr, w)

Return the off-diagonal double-layer contribution and its first two
k-derivatives for a single source-target pair.

This helper evaluates the complex Helmholtz DLP kernel contribution already
multiplied by the relevant quadrature weight `w`, and also returns the Hankel
values used so that the SLP part can reuse them without recomputation.

# Mathematical meaning
For the 2D Helmholtz Green function, the DLP kernel contribution entering the
CFIE has the form

    D(r) = w * (i k / 2) * inn * H₁^(1)(k r) / r

where:
- `r` is the source-target distance,
- `inn` is the oriented tangent numerator,
- `w` is the quadrature weight.

The returned values are:
- `d0 = D(k)`
- `d1 = dD/dk`
- `d2 = d²D/dk²`

with the derivative formulas written in a way that reuses `H₀` and `H₁`.

# Arguments
- `TT`:
  Real scalar type.
- `k`:
  Wavenumber.
- `r`:
  Distance between source and target.
- `inn`:
  Oriented DLP numerator.
- `invr`:
  Reciprocal distance `1/r`.
- `w`:
  Quadrature weight already appropriate for the source point/correction point.

# Returns
`(d0, d1, d2, h0, h1)` where:
- `d0, d1, d2` are complex scalars,
- `h0 = H₀^(1)(k r)`,
- `h1 = H₁^(1)(k r)`.
"""
@inline function _dlp_terms(TT,k,r,inn,invr,w)
    h0,h1=hankel_pair01(k*r)
    αD=Complex{TT}(0,k/2)
    d0=w*(αD*inn*h1*invr)
    d1=w*((Complex{TT}(0,1)/2)*inn*k*h0)
    d2=w*((Complex{TT}(0,1)/2)*inn*(h0-k*r*h1))
    return d0,d1,d2,h0,h1
end

"""
    _slp_terms(TT, k, r, s, w, h0, h1)

Return the off-diagonal single-layer contribution and its first two
k-derivatives for one source-target pair.

# Mathematical meaning
The SLP contribution appearing in the CFIE has the form

    S(r) = w * (i / 2) * H₀^(1)(k r) * s

where:
- `s` is the source-point speed factor,
- `w` is the quadrature weight.

The returned values are:
- `s0 = S(k)`
- `s1 = dS/dk`
- `s2 = d²S/dk²`

with derivative formulas written using the already available Hankel values.

# Arguments
- `TT`:
  Real scalar type.
- `k`:
  Wavenumber.
- `r`:
  Distance between source and target.
- `s`:
  Source-point speed factor entering the arclength measure.
- `w`:
  Quadrature weight.
- `h0, h1`:
  Hankel values `H₀^(1)(k r)` and `H₁^(1)(k r)`.

# Returns
`(s0, s1, s2)` as complex scalars.
"""
@inline function _slp_terms(TT,k,r,s,w,h0,h1)
    αS=Complex{TT}(0,one(TT)/2)
    s0=w*(αS*h0*s)
    s1=w*(-(Complex{TT}(0,1)/2)*r*h1*s)
    s2=w*((Complex{TT}(0,1)/2)*r*(h1-k*r*h0)*s/k)
    return s0,s1,s2
end

"""
    _wrap01(u)

Wrap a real parameter to the half-open interval [0,1).

This is the unit-interval analogue of periodic angular wrapping. It is used
when periodic correction nodes for closed curves step outside the base
parameter range.

# Returns
A value equivalent to `u mod 1`, but guaranteed to lie in `[0,1)`.

# Use case
Needed by periodic Alpert caches when a correction node at
`u_i ± ξ/N` crosses the periodic seam.
"""
@inline function _wrap01(u::T) where {T<:Real}
    v=mod(u,one(T))
    v<zero(T) ? v+one(T) : v
end

"""
    wrap_diff(t)

Wrap an angular difference to the interval [-π, π).

This helper is used to determine the orientation of a periodic parameter grid
from successive samples, without ambiguity from the `2π` branch cut.

# Returns
The wrapped angular difference.

# Use case
Used in `_periodic_orientation_sign` to determine whether the periodic panel
parameter increases or decreases with the stored node ordering.
"""
@inline function wrap_diff(t::T) where {T<:Real}
    mod(t+T(pi),two_pi)-T(pi)
end

"""
    _panel_sigma_wrap(σ)

Wrap an open-panel computational parameter into [0,1).

This is used for midpoint-grid panel coordinates when forming Alpert correction
nodes displaced by ±Δσ.

# Returns
Wrapped panel coordinate in `[0,1)`.
"""
@inline function _panel_sigma_wrap(σ::T) where {T<:Real}
    v=mod(σ,one(T))
    v<zero(T) ? v+one(T) : v
end

"""
    _lagrange_weights(ξ, nodes)

Compute Lagrange interpolation weights at evaluation point `ξ` for the given
interpolation nodes.

    ℓ_j(ξ) = ∏_{l≠j} (ξ - x_l) / (x_j - x_l)

so that for any interpolated quantity `f`

    f(ξ) ≈ Σ_j ℓ_j(ξ) f(x_j).

# Arguments
- `ξ`:
  Evaluation point.
- `nodes`:
  Stencil nodes.

# Returns
- `w::Vector{T}`:
  The Lagrange basis weights evaluated at `ξ`.

# Use case in Alpert
These weights are used to transfer kernel values evaluated at off-grid Alpert
correction nodes back to the nearby native discretization nodes.
"""
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

"""
    _alpert_interp_offsets_weights(ξ, ninterp)

Build periodic interpolation offsets and weights for an Alpert correction node
located at non-integer offset `ξ`.

# Arguments
- `ξ`:
  Non-integer offset in grid-index units.
- `ninterp`:
  Interpolation stencil size.

# Returns
- `offs`:
  Integer offsets describing the chosen local stencil.
- `wt`:
  Interpolation weights on that stencil.

# Use case
Stored in the periodic Alpert cache and later applied modulo N.
"""
@inline function _alpert_interp_offsets_weights(ξ::T,ninterp::Int) where {T<:Real}
    j0=floor(Int,ξ-T(ninterp)/2+one(T))
    offs=collect(j0:(j0+ninterp-1))
    wt=_lagrange_weights(ξ,T.(offs))
    return offs,wt
end

"""
    _local_offsets(p)

Return a centered even interpolation stencil of width `p`.

For example, if `p = 8`, the offsets are
`[-3, -2, -1, 0, 1, 2, 3, 4]`.

# Requirements
- `p` must be even.

# Use case
This is the canonical local stencil used for open-panel midpoint interpolation.
"""
@inline function _local_offsets(p::Int)
    iseven(p) || error("Interpolation stencil size p must be even.")
    q=p÷2
    return collect(-(q-1):q)
end


"""
    _periodic_orientation_sign(ts)

Determine the orientation sign of a periodic node sequence.

# Mathematical meaning
Given periodic parameter nodes `ts`, this function inspects the wrapped
difference between the first two nodes and returns:
- `+1` if the parameter increases in the stored ordering,
- `-1` if it decreases.

# Why this matters
For periodic Alpert correction nodes one must know whether positive correction
correspond to forward or backward motion along the stored parameter
order. This sign ensures the cache respects the actual orientation of the
component.

# Returns
`1` or `-1`.
"""
@inline function _periodic_orientation_sign(ts::AbstractVector{T}) where {T<:Real}
    N=length(ts)
    N<2 && return 1
    Δ=wrap_diff(ts[mod1(2,N)]-ts[1])
    Δ>=zero(T) ? 1 : -1
end

"""
    _dinner(dx, dy, tx, ty)

`ty*dx - tx*dy`.

# Returns
A scalar numerator for the DLP kernel.
"""
@inline function _dinner(dx,dy,tx,ty)
    return ty*dx-tx*dy
end

"""
    _speed(v)

Return the Euclidean norm of a tangent vector.

# Returns
`max(|v|, eps(T))`.
"""
@inline function _speed(v::SVector{2,T}) where {T<:Real}
    s=sqrt(v[1]^2+v[2]^2)
    s<eps(T) ? eps(T) : s
end

"""
    _panel_arrays(pts)

Extract flat coordinate and tangent arrays from `BoundaryPointsCFIE`.

# Returns
`(X, Y, dX, dY, s)` where:
- `X, Y` are point coordinates,
- `dX, dY` are tangent components,
- `s` is the speed array.

This is a lightweight unpacking helper used in tight assembly loops that prefer
plain vectors over repeated field access on small static vectors.
"""
@inline function _panel_arrays(pts::BoundaryPointsCFIE{T}) where {T<:Real}
    X=getindex.(pts.xy,1)
    Y=getindex.(pts.xy,2)
    dX=getindex.(pts.tangent,1)
    dY=getindex.(pts.tangent,2)
    s=@. sqrt(dX^2+dY^2)
    return X,Y,dX,dY,s
end

"""
    _add_naive_panel_block!(A, gi, xi, yi, rb, pb, Pb, k, αD, αS, ik; skip_pred=(j->false))

Add the naive off-panel CFIE contribution from one source panel to a fixed
target row.

This helper is used for interactions that do not need Alpert self-correction,
typically distinct panels/components

# Mathematical meaning
For each source node `j` in the source block `rb`, this adds

    - [ D_ij + i k S_ij ]

using the standard off-diagonal DLP and SLP kernels with source weights from
`pb.ws`.

# Arguments
- `A`:
  Global system matrix.
- `gi`:
  Global row index of the target point.
- `xi, yi`:
  Target coordinates.
- `rb`:
  Global index range of the source block.
- `pb`:
  Source boundary-point object.
- `Pb`:
  Flat source panel arrays.
- `k, αD, αS, ik`:
  Wavenumber and kernel prefactors.
- `skip_pred`:
  Optional predicate to omit selected source indices.

# Returns
- `A`, modified in place.
"""
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


"""
    _add_self_panel_alpert_correction!(A, gi, xi, yi, i, ra, Ca, hσ, k, αD, αS, ik, rule)

Add the Alpert self-panel correction for one target node on an open smooth
panel. It implements the replacement of the inaccurate near-diagonal part by
a corrected hybrid Gauss-trapezoid contribution.

# Arguments
- `A`:
  Global system matrix.
- `gi`:
  Global row index of the target.
- `xi, yi`:
  Target coordinates.
- `i`:
  Local target index within the panel.
- `ra`:
  Global row/column range of the panel.
- `Ca`:
  `AlpertSmoothPanelCache`.
- `hσ`:
  Base panel midpoint spacing in computational parameter.
- `k`:
  Wavenumber.
- `αD, αS, ik`:
  Kernel prefactors.
- `rule`:
  Alpert logarithmic correction rule.

# Returns
- `A`, modified in place.
"""
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

"""
    _panel_interp_midpoint_data(σ, hσ, N, p)

Build interpolation indices and weights for an off-grid point on an open
midpoint panel.

# Mathematical idea
An off-grid coordinate `σ` is expressed relative to the midpoint grid spacing
`hσ`. One chooses a centered even stencil of width `p`, clamps it inside the
panel, and computes Lagrange weights on that stencil.

# Arguments
- `σ`:
  Off-grid computational coordinate.
- `hσ`:
  Midpoint spacing.
- `N`:
  Number of panel nodes.
- `p`:
  Even interpolation stencil size.

# Returns
- `idx`:
  Concrete interpolation indices on the panel.
- `wt`:
  Lagrange interpolation weights.
"""
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

"""
    _eval_open_panel_geom_exact(crv, u)

Evaluate exact open-panel geometry data at parameter value `u`.

# Returns
`(x, y, tx, ty, s)` where:
- `(x,y)` is the point on the curve,
- `(tx,ty)` is the tangent,
- `s = |γ'(u)|` is the speed.
"""
@inline function _eval_open_panel_geom_exact(crv,u::T) where {T<:Real}
    q=curve(crv,u)
    t=tangent(crv,u)
    s=sqrt(t[1]^2+t[2]^2)
    return q[1],q[2],t[1],t[2],s
end

"""
    _build_alpert_periodic_cache(solver, crv, pts, rule, ord)

Build the periodic Alpert self-correction cache for one closed smooth component.

# What this computes
For each target node `i` and each Alpert correction abscissa `ξ_p`, this
function:
1. constructs the positive and negative correction nodes on the periodic curve,
2. evaluates coordinates, tangents, speeds, distances, and DLP numerators,
3. constructs periodic interpolation offsets and weights of size `ord+3`.

# Arguments
- `solver`:
  `CFIE_alpert` solver.
- `crv`:
  Underlying smooth periodic curve.
- `pts`:
  Boundary discretization for this component.
- `rule`:
  Alpert logarithmic rule.
- `ord`:
  Alpert correction order.

# Returns
- `AlpertPeriodicCache{T}`

# Notes
The interpolation width is chosen as `ninterp = ord + 3`, mirroring the
standard practical choice of using a  slightly wider interpolation stencil.
"""
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

"""
    _build_alpert_smooth_panel_cache(solver, crv, pts, rule, p)

Build the open-panel Alpert self-correction cache for one smooth panel.

# What this computes
For each target midpoint node and each Alpert correction, this
function:
1. shifts the panel computational coordinate by ±Δσ,
2. maps that shifted coordinate through the panel grading map,
3. evaluates the exact geometry and its chain-rule-transformed tangent,
4. computes distances and DLP numerators,
5. builds a local interpolation stencil of width `p`.

# Arguments
- `solver`:
  `CFIE_alpert` solver.
- `crv`:
  Underlying open curve segment.
- `pts`:
  Boundary discretization for this panel.
- `rule`:
  Alpert logarithmic rule.
- `p`:
  Even interpolation stencil size.

# Returns
- `AlpertSmoothPanelCache{T}`

# Notes
Unlike the periodic cache, interpolation data must be stored separately for
each target node because the stencil is clamped near endpoints.
"""
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

"""
    _build_alpert_component_cache(solver, crv, pts, rule, ord)

Build the appropriate Alpert self-correction cache for one component.

# Dispatch logic
- If `pts.is_periodic`, returns `AlpertPeriodicCache`.
- Otherwise returns `AlpertSmoothPanelCache`.

# Returns
- `AlpertCache{T}`

# Why this helper exists
It centralizes the cache-type selection so the workspace builder does not need
to know whether each component is periodic or open.
"""
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

"""
    _assemble_self_alpert_periodic!(A, pts, G, C, row_range, k, rule; multithreaded=true)

Assemble the self-interaction block for one periodic component using Alpert’s
hybrid correction.

# Mathematical structure
This function forms the self-block in three conceptual steps:

1. Add the identity contribution on the diagonal.
2. Add the full naive periodic quadrature contribution for all off-diagonal
   source nodes.
3. Undo the inaccurate near-neighbor trapezoid contribution in the band
   `|j-i| < a`, where `a = rule.a`.
4. Replace that near band with the Alpert correction-node contribution,
   redistributed through the periodic interpolation weights.

Thus the final block is the corrected self-panel CFIE block for a smooth closed
periodic component.

# Arguments
- `A`:
  Global system matrix.
- `pts`:
  Periodic boundary component.
- `G`:
  Geometric cache for this component.
- `C`:
  `AlpertPeriodicCache`.
- `row_range`:
  Global row/column range for this component.
- `k`:
  Wavenumber.
- `rule`:
  Alpert logarithmic correction rule.
- `multithreaded`:
  Enables row-parallel assembly.

# Returns
- `A`, modified in place.
"""
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

"""
    _assemble_self_alpert_periodic_deriv!(A, A1, A2, pts, G, C, P, row_range, k, rule; multithreaded=true)

Assemble the periodic Alpert self-block together with its first and second
derivatives with respect to `k`.
This is the derivative-aware version of `_assemble_self_alpert_periodic!`.
Every naive contribution, near-band subtraction, and Alpert correction-node
replacement is mirrored consistently at the level of:
- kernel values,
- first derivatives,
- second derivatives.

# Returns
- `A`:
  CFIE block.
- `A1`:
  First k-derivative.
- `A2`:
  Second k-derivative.
"""
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

"""
    _assemble_self_alpert_smooth_panel!(A, pts, G, C, row_range, k, rule; multithreaded=true)

Assemble the self-interaction block for one open smooth panel using Alpert
endpoint-aware correction.

# Mathematical structure
For open panels, the near correction differs from the periodic case:
- the near band `|j-i| < a` is treated differently because endpoint effects
  break periodic symmetry,
- the local correction nodes are interpolated through explicit per-target
  interpolation.

The function:
1. adds the identity,
2. adds either the naive or near-modified DLP/CFIE contribution depending on
   whether `|j-i| < a`,
3. adds the positive and negative Alpert correction-node contributions through
   the open-panel interpolation.

# Returns
- `A`, modified in place.
"""
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

"""
    _assemble_self_alpert_smooth_panel_deriv!(A, A1, A2, pts, G, C, P, row_range, k, rule; multithreaded=true)

Derivative version of the open-panel Alpert self-block assembly.

# Returns
- `A`, `A1`, `A2` assembled in place.

# Mathematical role
This extends `_assemble_self_alpert_smooth_panel!` to the first and second
k-derivatives, with all near-band logic and correction-node interpolation
performed as before.
"""
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


"""
    _assemble_self_alpert!(solver, A, pts, G, C, row_range, k, rule; multithreaded=true)

Dispatch helper for Alpert self-block assembly.

# Behavior
- If `pts.is_periodic`, dispatches to `_assemble_self_alpert_periodic!`.
- Otherwise dispatches to `_assemble_self_alpert_smooth_panel!`.

# Purpose
Provides a unified entry point when the caller does not want to branch
explicitly on cache type.
"""
function _assemble_self_alpert!(solver::CFIE_alpert{T},A::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},G::CFIEGeomCache{T},C,row_range::UnitRange{Int},k::T,rule::AlpertLogRule{T};multithreaded::Bool=true) where {T<:Real}
    pts.is_periodic ?
        _assemble_self_alpert_periodic!(A,pts,G,C,row_range,k,rule;multithreaded=multithreaded) :
        _assemble_self_alpert_smooth_panel!(solver,A,pts,G,C,row_range,k,rule;multithreaded=multithreaded)
end


"""
    _assemble_all_offpanel_naive!(A, pts, offs, parr, k; multithreaded=true)

Assemble all off-panel and off-component interactions using naive quadrature.

# Arguments
- `A`:
  Global system matrix.
- `pts`:
  Vector of component/panel discretizations.
- `offs`:
  Global component offsets.
- `parr`:
  Flat source arrays for each panel.
- `k`:
  Wavenumber.

# Returns
- `A`, modified in place.
"""
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


"""
    _assemble_all_offpanel_naive_deriv!(A, A1, A2, pts, offs, parr, k; multithreaded=true)

Derivative-aware version of `_assemble_all_offpanel_naive!`.

# Returns
- `A`, `A1`, `A2` modified in place.
"""
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

"""
    build_cfie_alpert_workspace(solver, pts)

Build the reusable Alpert workspace for a fixed boundary discretization.
The function matches each `pts[a]` to the corresponding curve segment from the
solver’s boundary description. For composite boundaries, the boundary is first
flattened so that the cache list and panel list remain aligned.

# What this builds
- the Alpert logarithmic rule,
- global component offsets,
- one geometry cache per component,
- one Alpert self-correction cache per component,
- one flat panel-array cache per component,
- the total matrix size.

# Arguments
- `solver::CFIE_alpert`
- `pts::Vector{BoundaryPointsCFIE{T}}`

# Returns
- `CFIEAlpertWorkspace{T}`
"""
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

"""
    _construct_matrices_cached!(A, pts, ws, k; multithreaded=true)

Assemble the full CFIE-Alpert matrix using a prebuilt workspace.

# Mathematical decomposition
The global matrix is assembled as

    A = Σ self-blocks + Σ off-panel blocks

where:
- self-blocks are corrected by Alpert hybrid quadrature,
- off-panel blocks are assembled naively because they are smooth.

# Arguments
- `A`:
  Preallocated global matrix.
- `pts`:
  Boundary discretizations.
- `ws`:
  `CFIEAlpertWorkspace`.
- `k`:
  Wavenumber.

# Returns
- `A`, modified in place.
"""
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

"""
    _construct_matrices_deriv_cached!(A, A1, A2, pts, ws, k; multithreaded=true)

Assemble the full CFIE-Alpert matrix and its first two k-derivatives using a
prebuilt workspace.
This is the derivative-aware counterpart of `_construct_matrices_cached!`,
combining:
- derivative-aware self-block Alpert corrections,
- derivative-aware naive off-panel assembly.

# Returns
- `A`, `A1`, `A2` modified in place.
"""
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

"""
    construct_matrices!(solver::CFIE_alpert, A, pts, ws, k; multithreaded=true)
    construct_matrices!(solver::CFIE_alpert, A, pts, k; multithreaded=true)
    construct_matrices(solver::CFIE_alpert, pts, ws, k; multithreaded=true)
    construct_matrices(solver::CFIE_alpert, pts, k; multithreaded=true)
    construct_matrices!(solver::CFIE_alpert, A, A1, A2, pts, ws, k; multithreaded=true)
    construct_matrices!(solver::CFIE_alpert, A, A1, A2, pts, k; multithreaded=true)
    construct_matrices!(solver::CFIE_alpert, basis::AbstractHankelBasis, A, A1, A2, pts, ws, k; multithreaded=true)
    construct_matrices!(solver::CFIE_alpert, basis::AbstractHankelBasis, A, A1, A2, pts, k; multithreaded=true)

High-level matrix assembly interface for the CFIE-Alpert solver family.

# What is assembled
All these overloads construct the same global CFIE boundary operator matrix,
with optional first and second derivatives with respect to `k`.

The value-only forms return or overwrite:
- `A(k)`

The derivative-aware forms additionally assemble:
- `A1(k) = dA/dk`
- `A2(k) = d²A/dk²`

# Overload philosophy
- Forms with `ws` reuse a prebuilt `CFIEAlpertWorkspace`.
- Forms without `ws` build the workspace internally.
- Bang forms write into preallocated matrices.
- Non-bang forms allocate and return new matrices.
- Basis-accepting forms exist for compatibility with the generic
  interface used elsewhere in the codebase.

# Performance guidance
For repeated use over many wavenumbers, prefer:
1. build `pts`,
2. build `ws = build_cfie_alpert_workspace(...)`,
3. reuse `construct_matrices!` with preallocated `A`, `A1`, `A2`.

# Returns
Depending on the overload:
- `A`
- `(A, A1, A2)`
"""
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

"""
    solve(solver::CFIE_alpert, basis, pts, k; multithreaded=true, use_krylov=true, which=:det_argmin)
    solve(solver::CFIE_alpert, basis, pts, ws, k; multithreaded=true, use_krylov=true, which=:det_argmin)
    solve(solver::CFIE_alpert, basis, A, pts, k; multithreaded=true, use_krylov=true, which=:det_argmin)
    solve(solver::CFIE_alpert, basis, A, pts, ws, k; multithreaded=true, use_krylov=true, which=:det_argmin)

High-level scalar solve interface for the CFIE-Alpert discretization.

# Purpose
These methods assemble the Alpert-corrected CFIE matrix and reduce it to a
single scalar diagnostic using the common backend `@svd_or_det_solve`.

Typical choices of `which` include:
- `:svd` for smallest singular value,
- `:det` for determinant,
- `:det_argmin` for determinant-based minimization logic.

# Overloads
- no `ws`, no `A`:
  simplest one-shot form;
- with `ws`:
  reuse cached geometric/alpert preprocessing;
- with `A`:
  reuse matrix storage;
- with both `A` and `ws`:
  most efficient repeated-use form.

# Returns
A scalar diagnostic whose exact meaning depends on `which`.
"""
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

"""
    solve_vect(solver::CFIE_alpert, basis, pts, k; multithreaded=true)
    solve_vect(solver::CFIE_alpert, basis, pts, ws, k; multithreaded=true)
    solve_vect(solver::CFIE_alpert, basis, A, pts, k; multithreaded=true)
    solve_vect(solver::CFIE_alpert, basis, A, pts, ws, k; multithreaded=true)

Compute the smallest singular value of the Alpert CFIE matrix and the
associated right singular vector.

# Mathematical meaning
Given the assembled matrix `A`, these routines compute its full SVD

    A = U Σ V*

and return:
- `σ_min(A)`,
- the corresponding right singular vector.

# Overloads
They mirror the `solve` interface:
- one-shot,
- cached workspace,
- reusable matrix buffer,
- reusable matrix buffer plus workspace.

# Returns
`(mu, u_mu)` where:
- `mu` is the smallest singular value,
- `u_mu` is the corresponding right singular vector.
"""
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

# INTERNAL - for benchmarking and diagnostic purposes only; not part of the public API
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

# INTERNAL - for benchmarking and diagnostic purposes only; not part of the public API
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

# this hack is really annoying, some off gird alpert nodes can be below the rmin of the actual geometry, so need to use the 
# workspace to actually get the accurate bounds or else chebyshev interpolation will fail catastrophically. 
# We pad the bounds a bit to be safe, but this is really just a hack until we have a better way to do this.
function estimate_cfie_alpert_cheb_rbounds(ws::CFIEAlpertWorkspace{T};pad=(T(0.95),T(1.05))) where {T<:Real}
    rmin=typemax(T)
    rmax=zero(T)
    # 1) Include all same-block geometric distances already stored in the direct caches.
    for G in ws.Gs
        R=G.R
        @inbounds for j in axes(R,2), i in axes(R,1)
            i==j && continue
            r=R[i,j]
            if isfinite(r) && r>eps(T)
                rmin=min(rmin,r)
                rmax=max(rmax,r)
            end
        end
    end
    # 2) Include all off-block geometric distances explicitly.
    #    This is required for composite corner geometries, where the smallest
    #    Chebyshev-relevant distance can occur between different blocks/panels.
    parr=ws.parr
    nc=length(parr)
    @inbounds for a in 1:nc
        Pa=parr[a]
        Xa=Pa.X; Ya=Pa.Y
        Na=length(Xa)
        for b in 1:nc
            b==a && continue
            Pb=parr[b]
            Xb=Pb.X; Yb=Pb.Y
            Nb=length(Xb)
            for j in 1:Nb, i in 1:Na
                dx=Xa[i]-Xb[j]
                dy=Ya[i]-Yb[j]
                r2=muladd(dx,dx,dy*dy)
                if isfinite(r2) && r2>(eps(T))^2
                    r=sqrt(r2)
                    rmin=min(rmin,r)
                    rmax=max(rmax,r)
                end
            end
        end
    end
    # 3) Include all Alpert correction-node distances from the direct caches.
    #    These can be much smaller than node-node distances and must be part of
    #    the admissible Chebyshev radius interval.
    for C in ws.Cs
        @inbounds for r in C.rp
            if isfinite(r) && r>eps(T)
                rmin=min(rmin,r)
                rmax=max(rmax,r)
            end
        end
        @inbounds for r in C.rm
            if isfinite(r) && r>eps(T)
                rmin=min(rmin,r)
                rmax=max(rmax,r)
            end
        end
    end
    @assert isfinite(rmin) && rmax>zero(T)
    return Float64(pad[1]*rmin),Float64(pad[2]*rmax)
end