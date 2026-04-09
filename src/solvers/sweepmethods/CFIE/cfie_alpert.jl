# =============================================================================
#  ALPERT-CORRECTED CFIE ASSEMBLY: OVERVIEW AND IMPLEMENTATION NOTES
# =============================================================================
#
#  Reference:
#      B. Alpert, "Hybrid Gauss-Trapezoidal Quadrature Rules", SIAM J. Sci. Comput.
#      20 (1999), 1551–1584.
#
#  This file implements an Alpert-corrected assembly of the CFIE boundary matrix
#  for panelwise smooth 2D billiard boundaries.
#
#  The main issue is the self-interaction singular / nearly singular part of the
#  single-layer potential (SLP). For source and target points that are well
#  separated, the ordinary trapezoidal rule is highly accurate on smooth periodic
#  curves. However, when the source point approaches the target point on the same
#  boundary component, the logarithmic singularity in the kernel destroys the 
#  the plain trapezoid rule.
#
#  Alpert’s idea is:
#
#      far interactions     -> use ordinary trapezoidal quadrature
#      near self interactions -> replace the local trapezoid contribution by a
#      corrected quadrature rule with shifted nodes
#
#  In this implementation, the corrected shifted source locations are not added
#  as new true boundary unknowns. Instead, each shifted source point is locally
#  interpolated back onto the native boundary grid by a 4-point stencil.
#
# -----------------------------------------------------------------------------
#  HIGH-LEVEL PICTURE
# -----------------------------------------------------------------------------
#
#  For one target point x_i on the boundary, the self SLP contribution is split:
#
#      ∫_Γ K(x_i, y) σ(y) ds(y)
#        = far part + near part
#
#  where:
#
#      - the far part is evaluated by the ordinary trapezoid rule over standard
#        boundary nodes y_j,
#      - the near part is replaced by an Alpert correction using shifted source
#        points y_i ± Δ_p near the target.
#
#
#  Sketch: periodic closed curve
#
#  For one fixed target row i:
#
#  target: x_i = y(θ_i)
#
#  source grid on the boundary:
#
#      ...  y_{i-1} ---- y_i ---- y_{i+1} ---- y_{i+2}  ...
#
#  near-source corrected evaluation point:
#
#      y(θ_i + Δ_p) -> *
#
#  This star is:
#      - a SOURCE point
#      - near y_i
#      - not one of the grid nodes
#
#  So we evaluate the kernel there:
#
#      K(x_i, y(θ_i + Δ_p))
#
#  and approximate the density there by interpolation (since we dont know it at shifted points):
#
#      σ(θ_i + Δ_p) ≈ w0 σ_{i-1} + w1 σ_i + w2 σ_{i+1} + w3 σ_{i+2}
#
#  In other words, the shifted source point is used for the quadrature, but its
#  density value is represented by nearby grid-node unknowns.
#  Therefore one shifted-source evaluation does not contribute to a single matrix
#  column. Instead, its contribution is redistributed onto the local 4-point
#  source stencil with the interpolation weights:
#
#      A[i,i-1] += coeff * w0
#      A[i,i  ] += coeff * w1
#      A[i,i+1] += coeff * w2
#      A[i,i+2] += coeff * w3
#
#  where `coeff` contains the kernel evaluation, the Alpert quadrature weight,
#  and the local speed / Jacobian factor at the shifted source point.
#
#  Thus, the near integral is approximated by evaluating the kernel at shifted
#  off-grid source points, then redistributing each shifted-source contribution
#  back onto the nearby source-grid columns.
# -----------------------------------------------------------------------------
#  WHAT IS CORRECTED AND WHAT IS NOT
# -----------------------------------------------------------------------------
#
#   - DLP diagonal / off-diagonal: assembled directly with the existing CFIE formulas
#
#    - SLP far part: plain trapezoidal rule
#
#    - SLP near self part: Alpert correction
#
#  Thus the assembly is a hybrid. This is why the correction logic appears only in the near-self assembly blocks.
#
# -----------------------------------------------------------------------------
#  TWO GEOMETRIC MODES
# -----------------------------------------------------------------------------
#
#  There are two versions of the cache and near-correction logic:
#
#      (A) periodic closed boundary: AlpertPeriodicCache
#
#      (B) one open smooth panel: AlpertSmoothPanelCache
#
#  and a third layer:
#
#      (C) composite boundary made of several smooth panels joined together:
#          _assemble_self_alpert_composite_component!
#
# -----------------------------------------------------------------------------
#  PERIODIC MODE
# -----------------------------------------------------------------------------
#
#  Used when one boundary component is a single smooth periodic curve.
#
#  Let the boundary be parametrized by θ ∈ (0,2π], and let the native source/target
#  quadrature nodes be
#
#      θ_j = ts[j],    j = 1,...,N
#
#  with corresponding boundary points
#
#      y_j = y(θ_j).
#
#  Consider one fixed target row i. The target point is
#
#      x_i = y(θ_i),   where θ_i = ts[i].
#
#  The self SLP row for x_i contains contributions from source points y_j on the
#  same periodic curve. Away from y_i, the ordinary trapezoidal rule is used.
#  Near y_i, the trapezoidal rule is replaced by the Alpert correction.
#
#  For each Alpert correction node p, we define two shifted SOURCE parameter values:
#
#      θ_i^+ = θ_i + h * rule.x[p]
#      θ_i^- = θ_i - h * rule.x[p]
#
#  where:
#
#      - rule.x[p] ∈ (0,1) are the Alpert nodes,
#      - h = 2π / N is the parameter spacing.
#
#  These are wrapped periodically back into (0,2π]. In general, θ_i^± do not coincide
#  with grid nodes, so the geometry must be reconstructed.
#
#  In this implementation, the shifted source geometry is obtained by local
#  4-point periodic interpolation using nearby nodes:
#
#      θ_{i-1}, θ_i, θ_{i+1}, θ_{i+2}
#
#  yielding:
#
#      y(θ_i^±), tangent(θ_i^±), speed(θ_i^±).
#
#  These are stored in the periodic cache:
#
#      xp, yp, txp, typ, sp   for the + shifts
#      xm, ym, txm, tym, sm   for the - shifts
#
#
#  The density at the shifted source point is also represented by local interpolation:
#
#      σ(θ_i^+) ≈ w0 σ_{i-1} + w1 σ_i + w2 σ_{i+1} + w3 σ_{i+2}
#
#  and similarly for σ(θ_i^-).
#
#  Therefore, one shifted source evaluation does NOT correspond to a single
#  source column. Instead, its contribution is distributed over 4 nearby source
#  columns using interpolation weights.
#
#  This information is stored in:
#
#      idxp / idxm   = indices of the 4 stencil nodes
#      wtp  / wtm    = interpolation weights
#
#
#  In summary:
#
#      fixed target x_i
#         → evaluate kernel at shifted source points near y_i
#         → represent σ at those points via interpolation
#         → distribute (scatter) the contribution onto 4 source columns
#
#
#  Schematic (for one target row i):
#
#      source grid:
#
#          ... y_{i-1} ---- y_i ---- y_{i+1} ---- y_{i+2} ...
#
#      target:
#
#          x_i = y_i
#
#      shifted source point:
#
#          * = y(θ_i + Δ_p)
#
#      interpolation:
#
#          σ(*) ≈ w0 σ_{i-1} + w1 σ_i + w2 σ_{i+1} + w3 σ_{i+2}
#
#      contribution to matrix row:
#
#          coeff = K(x_i, *) * (Alpert weight) * (speed factor)
#
#          A[i,i-1] += coeff * w0
#          A[i,i  ] += coeff * w1
#          A[i,i+1] += coeff * w2
#          A[i,i+2] += coeff * w3
#
# -----------------------------------------------------------------------------
#  OPEN SMOOTH PANEL MODE
# -----------------------------------------------------------------------------
#
#  This is similar to the periodic case, but the parameter domain is now u ∈ [0,1]
#  and there is no periodic wrapping.
#
#  The key difference is that shifted source points must remain inside the panel:
#
#      use the + correction only if   u_i^+ < 1
#      use the - correction only if   u_i^- > 0
#
#  If a shifted source point leaves the panel, the current panel cannot supply
#  that correction locally. In the composite-boundary case (next section),
#  this contribution may instead be handled by a neighboring panel.
#
# -----------------------------------------------------------------------------
#  COMPOSITE PANEL MODE
# -----------------------------------------------------------------------------
#
#  For boundaries composed of multiple smooth panels, each panel is treated locally.
#  However, near smooth joins, a shifted source point may lie on a neighboring panel.
#
#  This is handled by:
#
#      _assemble_self_alpert_composite_component!
#
#  The topology object determines:
#
#      - whether neighboring panels exist,
#      - whether the join is smooth.
#
#
#  Sketch:
#
#      panel a                         panel b
#      |------------------------------||-----------------------------|
#         o   o   o   o   o             *   *   *   *   *
#                          ● target
#
#      shifted source crosses interface:
#
#          - evaluate geometry on the correct panel
#          - represent σ via that panel's local stencil
#          - distribute contribution onto that panel's source columns
#
#  In this case:
#
#      - far SLP skips overlapping near nodes,
#      - shifted source is evaluated on the neighboring panel,
#      - contribution is distributed onto that panel’s local stencil.
#
# -----------------------------------------------------------------------------
#  CACHES
# -----------------------------------------------------------------------------
#
#  The Alpert correction repeatedly requires, for each target i and node p:
#
#      - shifted source point y(θ_i ± Δ_p) (or y(u_i ± Δ_p))
#      - tangent and speed at that point
#      - local 4-point interpolation stencil
#      - interpolation weights
#
#  Computing these inside the assembly loop would be expensive, so they are
#  precomputed and stored in caches.
#
#
#  .1 Periodic cache
#
#  For each (p,i):
#
#      xp[p,i], yp[p,i]     = y(θ_i + Δ_p)
#      txp[p,i], typ[p,i]   = tangent at θ_i + Δ_p
#      sp[p,i]              = speed at θ_i + Δ_p
#
#      xm[p,i], ym[p,i]     = y(θ_i - Δ_p)
#      txm[p,i], tym[p,i]   = tangent at θ_i - Δ_p
#      sm[p,i]              = speed at θ_i - Δ_p
#
#      idxp[p,i,1:4]        = stencil indices for +
#      wtp[p,i,1:4]         = interpolation weights for +
#
#      idxm[p,i,1:4]        = stencil indices for -
#      wtm[p,i,1:4]         = interpolation weights for -
#
#
#  .2 Smooth panel cache
#
#  Same structure, but using panel parameter u instead of θ.
#
#
#  .3 Assembly workflow (per target row i)
#
#      (1) Add diagonal term
#      (2) Add DLP off-diagonal terms
#      (3) Add SLP far terms (skip near region)
#      (4) Add SLP near correction:
#
#              for each Alpert node p:
#
#                  - read shifted source from cache
#                  - evaluate kernel K(x_i, shifted source)
#                  - distribute contribution onto 4 source columns
#
#
#  .4 Near region definition
#
#      periodic:
#          |j - i| (mod N) < a
#
#      panel:
#          abs(j - i) < a
#
#      where a = rule.a (Alpert exclusion width)
#
#
#  CFIEAlpertWorkspace collects:
#
#      rule          : Alpert rule
#      offs          : global offsets
#      Gs            : geometry caches
#      Cs            : Alpert caches
#      topos/gmaps   : topology info
#      panel_to_comp : component lookup
#      Ntot          : system size
#
#  enabling repeated assembly without recomputing geometry.
# -----------------------------------------------------------------------------
#  LIMITATIONS OF THIS IMPLEMENTATION
# -----------------------------------------------------------------------------
#
#  This implementation is designed for:
#
#      - smooth boundaries
#      - periodic closed curves
#      - smooth open panels
#      - composite boundaries with smooth joins
#
#  It is not a full corner-aware generalized Alpert implementation.
#
#  In particular, no corner singularity resolution is included here and 
#  the interpolation is local 4-point Lagrange.
# =============================================================================

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


@inline function _scatter_localp!(A::AbstractMatrix{Complex{T}},gi::Int,col_range::UnitRange{Int},coeff::Complex{T},idx,wt) where {T<:Real}
    @inbounds for m in eachindex(idx)
        A[gi,col_range[idx[m]]] += coeff*wt[m]
    end
    return nothing
end

@inline function _wrap01(u::T) where {T<:Real}
    v=mod(u,one(T))
    return v<zero(T) ? v+one(T) : v
end

# wrap_angle
# Wrap an angle to the interval (0, 2π].
# Inputs:
#   - t::T : Angle to wrap.
# Outputs:
#   - Wrapped angle in (0, 2π].
@inline function wrap_angle(t::T) where {T<:Real}
    tp=mod(t,two_pi)
    return tp==zero(T) ? T(two_pi) : tp
end

# wrap_diff
# Wrap a difference of angles to the interval (-π, π].
# Inputs:
#   - t::T : Angle difference to wrap.
# Outputs:
#   - Wrapped angle difference in (-π, π].
@inline function wrap_diff(t::T) where {T<:Real}
    return mod(t+T(pi),two_pi)-T(pi)
end

# Generic Lagrange weights on arbitrary local nodes.
# Inputs:
#   η     : local coordinate where interpolation is evaluated
#   nodes : interpolation nodes in the same local coordinate system
# Output:
#   w     : Lagrange weights so that f(η) ≈ sum_j w[j] f(nodes[j])
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

# Build a symmetric local stencil of size p around the interval anchor.
# For p even, this returns offsets:
#   -(p÷2-1), ..., -1, 0, 1, ..., p÷2
# Example:
#   p=4  -> (-1,0,1,2)
#   p=8  -> (-3,-2,-1,0,1,2,3,4)
@inline function _local_offsets(p::Int)
    iseven(p) || error("Interpolation stencil size p must be even.")
    q=p÷2
    return collect(-(q-1):q)
end

function _build_alpert_periodic_cache(solver::CFIE_alpert{T},crv::C,pts::BoundaryPointsCFIE{T},rule::AlpertLogRule{T},ord::Int) where {T<:Real,C<:AbsCurve}
    N=length(pts.xy)
    h=pts.ws[1]
    jcorr=rule.j
    ninterp=ord+3

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

        δu=ξ/T(N)

        for i in 1:N
            ui=(T(i)-T(1)/2)/T(N)

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

###########################################
########### PANEL LOGIC ALPERT ############
###########################################

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

# _panel_us
# Compute the local parameter values at the midpoints of the panels for a given number of points N.
# Inputs:
#   - T : Real type for the parameter values.
#   - N : Number of points (panels) to compute the midpoints for.
# Outputs:
#   - Vector{T} : A vector containing the local parameter values at the midpoints
@inline function _panel_us(::Type{T},N::Int) where {T<:Real}
    return collect(midpoints(range(zero(T),one(T),length=N+1)))
end


# Smooth open-panel local-p midpoint interpolation metadata.
#
# Local panel nodes are midpoint-indexed. If
#   s = u/h - 1/2
# and j0 = floor(s)+1,
# then η = s - floor(s) ∈ [0,1)
# and we interpolate on offsets centered around j0.
#
# Outputs:
#   idx : Vector{Int} of length p
#   wt  : Vector{T}   of length p
@inline function _panel_smooth_localp_midpoint_data(u::T,h::T,N::Int,p::Int) where {T<:Real}
    iseven(p) || error("p must be even.")
    p<=N || error("p must satisfy p <= N.")
    q=p÷2
    s=u/h-T(1)/2
    j0=floor(Int,s)+1
    η=s-floor(T,s)
    j0=clamp(j0,q,N-q)
    offs=_local_offsets(p)
    nodes=T.(offs)
    wt=_lagrange_weights(η,nodes)
    idx=Vector{Int}(undef,p)
    @inbounds for m in 1:p
        idx[m]=j0+offs[m]
    end
    return idx,wt
end


# Smooth open-panel local-p interpolation of shifted source data.
#
# Inputs:
#   u      : shifted local panel parameter
#   h      : panel midpoint spacing
#   X,Y    : sampled geometry arrays
#   dX,dY  : sampled tangent arrays
#   p      : even stencil size
#
# Outputs:
#   x,y,tx,ty,s,idx,wt
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

# _build_alpert_component_cache
# Precompute only endpoint-special Alpert data for one smooth panel.
#
# Inputs:
#   - pts::BoundaryPointsCFIE{T} :
#       One smooth panel.
#   - rule::AlpertLogRule{T} :
#       Log-singular Alpert rule.
#
# Outputs:
#   - C::AlpertComponentCache{T} : Cache storing interpolation metadata and endpoint rules.
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
            um=us[i]-Δu
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
    return pts.is_periodic ? _build_alpert_periodic_cache(solver,crv,pts,rule,ord) : _build_alpert_smooth_panel_cache(pts,rule,ord)
end

###########################################################
################ SELF ALPERT ASSEMBLY #####################
###########################################################

function _assemble_self_alpert_periodic!(A::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},G::CFIEGeomCache{T},C::AlpertPeriodicCache{T},row_range::UnitRange{Int},k::T,rule::AlpertLogRule{T};multithreaded::Bool=true) where {T<:Real}
    αD=Complex{T}(0,k/2)
    αS=Complex{T}(0,one(T)/2)
    ik=Complex{T}(0,k)
    X=getindex.(pts.xy,1)
    Y=getindex.(pts.xy,2)
    N=length(pts.ts)
    h=pts.ws[1]
    nskip=rule.a
    jcorr=rule.j
    ninterp=C.ninterp

    @use_threads multithreading=multithreaded for i in 1:N
        gi=row_range[i]
        xi=X[i]
        yi=Y[i]
        si=G.speed[i]
        κi=G.kappa[i]

        # diagonal: I - D_diag
        A[gi,gi]+=one(Complex{T})-Complex{T}(h*si*κi,zero(T))

        # DLP off-diagonal: keep naive everywhere
        @inbounds for j in 1:N
            j==i && continue
            gj=row_range[j]
            rij=G.R[i,j]
            inn=G.inner[i,j]
            invr=G.invR[i,j]
            A[gi,gj]-=h*(αD*inn*H(1,k*rij)*invr)
        end

        # SLP off-diagonal: start from naive everywhere
        @inbounds for j in 1:N
            j==i && continue
            gj=row_range[j]
            A[gi,gj]-=ik*(h*(αS*H(0,k*G.R[i,j])*G.speed[j]))
        end

        # remove only the wrapped near band of the naive SLP
        @inbounds for m in (-nskip+1):(nskip-1)
            m==0 && continue
            j=mod1(i+m,N)
            gj=row_range[j]
            A[gi,gj]+=ik*(h*(αS*H(0,k*G.R[i,j])*G.speed[j]))
        end

        # add only the Alpert SLP correction
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
                    A[gi,row_range[q]]+=coeff*C.wtp[p,m]
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
                    A[gi,row_range[q]]+=coeff*C.wtm[p,m]
                end
            end
        end
    end
    return A
end

# _assemble_self_alpert_smooth_panel!
# Assemble the self-interaction block of the CFIE matrix for a single smooth panel using the panel-based Alpert rule. This includes the standard diagonal and off-diagonal contributions from the DLP and SLP, as well as the near correction using the precomputed shifted interpolation vectors from the AlpertSmoothPanelCache.
# Inputs:
#   - A::AbstractMatrix{Complex{T}} : Matrix to assemble into (modified in place).
#   - pts::BoundaryPointsCFIE{T} : Boundary points for the CFIE discretization (should correspond to a single smooth panel).
#   - G::CFIEGeomCache{T} : Precomputed geometric quantities for the CFIE assembly.
#   - C::AlpertSmoothPanelCache{T} : Precomputed cache for the panel-based Alpert rule.
#   - row_range::UnitRange{Int} : Row indices corresponding to the current panel.
#   - k::T : Wave number.
#   - rule::AlpertLogRule{T} : Alpert quadrature rule.
#   - multithreaded::Bool : Whether to use multithreading for assembly.
# Outputs:
#   - A : Modified in place with the self-interaction block assembled using the panel-based Alpert rule.
function _assemble_self_alpert_smooth_panel!(solver::CFIE_alpert{T},A::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},G::CFIEGeomCache{T},C::AlpertSmoothPanelCache{T},row_range::UnitRange{Int},k::T,rule::AlpertLogRule{T};multithreaded::Bool=true) where {T<:Real}
    αD=Complex{T}(0,k/2)
    αS=Complex{T}(0,one(T)/2)
    ik=Complex{T}(0,k)
    X=getindex.(pts.xy,1)
    Y=getindex.(pts.xy,2)
    N=length(X)
    @info "self alpert smooth panel"
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
            rij=G.R[i,j]
            inn=G.inner[i,j]
            invr=G.invR[i,j]
            A[gi,gj]-=h*(αD*inn*H(1,k*rij)*invr)
        end
        @inbounds for j in 1:N
            j==i && continue
            abs(j-i)<a && continue
            gj=row_range[j]
            A[gi,gj]-=ik*(h*(αS*H(0,k*G.R[i,j])*G.speed[j]))
        end
        @inbounds for p in 1:jcorr
            fac=h*rule.w[p]
            Δu=h*rule.x[p]
            ui=C.us[i]
            # plus side exists only if still inside panel
            if ui+Δu<one(T)
                dx=xi-C.xp[p,i]
                dy=yi-C.yp[p,i]
                r=sqrt(dx*dx+dy*dy)
                if isfinite(r) && r>sqrt(eps(T))
                    coeff= -ik*(fac*(αS*H(0,k*r)*C.sp[p,i]))
                    for m in axes(C.idxp,3)
                        q=C.idxp[p,i,m]
                        A[gi,row_range[q]]+=coeff*C.wtp[p,i,m]
                    end
                end
            end
            # minus side exists only if still inside panel
            if ui-Δu>zero(T)
                dx=xi-C.xm[p,i]
                dy=yi-C.ym[p,i]
                r=sqrt(dx*dx+dy*dy)
                if isfinite(r) && r>sqrt(eps(T))
                    coeff= -ik*(fac*(αS*H(0,k*r)*C.sm[p,i]))
                    for m in axes(C.idxp,3)
                        q=C.idxm[p,i,m]
                        A[gi,row_range[q]]+=coeff*C.wtm[p,i,m]
                    end
                end
            end
        end
    end
    return A
end

###############################
#### COMPOSITE ALPERT HELP ####
###############################

# _component_id_of_panel
# Determine which component a given panel index belongs to based on the provided groupings (gmaps). This is used to identify the correct component for applying the appropriate Alpert correction during assembly.
# Inputs:
#   - a::Int : Panel index to check.
#   - gmaps::Vector{Vector{Int}} : A vector of vectors, where each inner vector contains the panel indices that belong to a particular component.
# Outputs:
#   - Int : The index of the component that panel a belongs to, or 0 if it does not belong to any component.
@inline function _component_id_of_panel(a::Int,gmaps::Vector{Vector{Int}})
    @inbounds for c in eachindex(gmaps)
        a in gmaps[c] && return c
    end
    return 0
end

# _panel_xy_tangent_arrays
# Extract the x and y coordinates of the geometry and the tangent vectors from the BoundaryPointsCFIE struct for a given panel. This is used to prepare the data for interpolation when applying the Alpert correction for panel-based rules.
# Inputs:
#   - pts::BoundaryPointsCFIE{T} : Boundary points for the CFIE discretization, corresponding to a single panel.
# Outputs:
#   - Tuple of vectors (X, Y, dX, dY) where X and Y are the x and y coordinates of the geometry, and dX and dY are the x and y components of the tangent vectors at the sampled points on the panel.
@inline function _panel_xy_tangent_arrays(pts::BoundaryPointsCFIE{T}) where {T<:Real}
    X=getindex.(pts.xy,1)
    Y=getindex.(pts.xy,2)
    dX=getindex.(pts.tangent,1)
    dY=getindex.(pts.tangent,2)
    return X,Y,dX,dY
end

# _right_neighbor_excluded_count
# Compute the number of points to exclude from the right neighbor panel when applying the Alpert correction for a point near the right endpoint of the current panel. This is used to determine which points in the neighboring panel should be skipped when applying the SLP contribution in the assembly.
# Inputs:
#   - i::Int : Local index of the point within the current panel.
#   - N::Int : Total number of points in the current panel.
#   - a::Int : Alpert parameter indicating the number of near points to exclude.
# Outputs:
#   - Int : The number of points to exclude from the right neighbor panel.
@inline function _right_neighbor_excluded_count(i::Int,N::Int,a::Int)
    return max(0,i+a-1-N)
end

# _left_neighbor_excluded_count
# Compute the number of points to exclude from the left neighbor panel when applying the Alpert correction for a point near the left endpoint of the current panel. This is used to determine which points in the neighboring panel should be skipped when applying the SLP contribution in the assembly.
# Inputs:
#   - i::Int : Local index of the point within the current panel.
#   - a::Int : Alpert parameter indicating the number of near points to exclude.
# Outputs:
#   - Int : The number of points to exclude from the left neighbor panel.   
@inline function _left_neighbor_excluded_count(i::Int,a::Int)
    return max(0,a-i)
end

@inline function _eval_on_open_panel_localp(pts::BoundaryPointsCFIE{T},u::T,p::Int) where {T<:Real}
    X,Y,dX,dY=_panel_xy_tangent_arrays(pts)
    h=pts.ws[1]
    return _eval_shifted_source_smooth_panel_localp(u,h,X,Y,dX,dY,p)
end

# _assemble_self_alpert_composite_component!
# Assemble the self-interaction block of the CFIE matrix for a composite component consisting of multiple panels, using the appropriate Alpert correction for each point based on its location within the component and its proximity to the panel endpoints. This function handles the logic for determining which points to apply the Alpert correction to, as well as the contributions from neighboring panels when the target point is near a panel endpoint.
# Inputs:
#   - A::AbstractMatrix{Complex{T}} : Matrix to assemble into (modified in place).
#   - pts::Vector{BoundaryPointsCFIE{T}} : Vector of boundary points for each panel in the composite component.
#   - Gs::Vector{CFIEGeomCache{T}} : Vector of precomputed geometric quantities for each panel.
#   - Cs : Vector of AlpertComponentCache for each panel, containing the precomputed data for the Alpert correction.
#   - offs::Vector{Int} : Vector of offsets indicating the global index range for each panel in the composite component.
#   - k::T : Wave number.
#   - rule::AlpertLogRule{T} : Alpert quadrature rule.
#   - topo::AlpertCompositeTopology{T} : Topology information for the composite component, including the types of neighboring panels and their relationships.
#   - gmap::Vector{Int} : Mapping from global panel indices to the corresponding indices in the pts, Gs, and Cs vectors.
#   - multithreaded::Bool : Whether to use multithreading for assembly.
# Outputs:
#   - A : Modified in place with the self-interaction block for the composite component assembled using the appropriate Alpert corrections.
function _assemble_self_alpert_composite_component!(solver::CFIE_alpert{T},A::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},Gs::Vector{CFIEGeomCache{T}},Cs,offs::Vector{Int},k::T,rule::AlpertLogRule{T},topo::AlpertCompositeTopology{T},gmap::Vector{Int};multithreaded::Bool=true) where {T<:Real}
    αD=Complex{T}(0,k/2)
    αS=Complex{T}(0,one(T)/2)
    ik=Complex{T}(0,k)
    @info "self alpert composite component"
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
                            A[gi,gj]-=pb.ws[j]*(αD*inn*H(1,k*r)*invr)
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
                            A[gi,gj]-=ik*(pb.ws[j]*(αS*H(0,k*r)*sb[j]))
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
                        coeff=-ik*(fac*(αS*H(0,k*r)*Ca.sp[p,i]))
                        for m in axes(Ca.idxp,3)
                            q=Ca.idxp[p,i,m]
                            A[gi,ra[q]]+=coeff*Ca.wtp[p,i,m]
                        end
                    end
                end
                if ui-Δu>zero(T)
                    dx=xi-Ca.xm[p,i]
                    dy=yi-Ca.ym[p,i]
                    r=sqrt(dx*dx+dy*dy)
                    if isfinite(r) && r>sqrt(eps(T))
                        coeff=-ik*(fac*(αS*H(0,k*r)*Ca.sm[p,i]))
                        for m in axes(Ca.idxm,3)
                            q=Ca.idxm[p,i,m]
                            A[gi,ra[q]]+=coeff*Ca.wtm[p,i,m]
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
                            coeff=-ik*(fac*(αS*H(0,k*r)*s2))
                            _scatter_localp!(A,gi,next_ra,coeff,idx2,wt2)
                        end
                    end
                end
            end
            if left_smooth && prev_idx!=0
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
                            coeff=-ik*(fac*(αS*H(0,k*r)*s2))
                            _scatter_localp!(A,gi,prev_ra,coeff,idx2,wt2)
                        end
                    end
                end
            end
        end
    end
    return A
end

###############################################
############## ASSEMBLY DISPATCH ##############
###############################################

function _assemble_self_alpert!(solver::CFIE_alpert{T},A::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},G::CFIEGeomCache{T},C,row_range::UnitRange{Int},k::T,rule::AlpertLogRule{T};multithreaded::Bool=true) where {T<:Real}
    return pts.is_periodic ? _assemble_self_alpert_periodic!(A,pts,G,C,row_range,k,rule;multithreaded=multithreaded) : _assemble_self_alpert_smooth_panel!(solver,A,pts,G,C,row_range,k,rule;multithreaded=multithreaded)
end

# _assemble_all_self_alpert_composite!
# Assemble all self-interaction blocks for a composite boundary using the appropriate Alpert assembly for each component. This dispatches to `_assemble_self_alpert_composite_component!` for each component, which handles the logic for smooth joins and near corrections based on the composite topology.
# Inputs:
#   - solver,A,pts,G,C,offs,k,rule,topos,gmaps :
#       Standard assembly data for the composite case, including the topological information and global index maps.
#   - multithreaded::Bool=true :
#       Whether to thread over target rows within each component.
# Outputs:
#   - Modifies `A` in place with all self-interaction blocks assembled for the composite boundary.
function _assemble_all_self_alpert_composite!(solver::CFIE_alpert{T},A::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},Gs::Vector{CFIEGeomCache{T}},Cs,offs::Vector{Int},k::T,rule::AlpertLogRule{T},topos::Vector{AlpertCompositeTopology{T}},gmaps::Vector{Vector{Int}};multithreaded::Bool=true) where {T<:Real}
    @inbounds for c in eachindex(gmaps) # gmaps is a vector of vectors, where each inner vector contains the component indices that belong to that composite component. So we go closed componentwise and pull out the relevant component indices for each composite component, then dispatch to the appropriate assembly routine based on whether it's a single periodic component or a multi-component composite.
        gmap=gmaps[c]
        if length(gmap)==1 && pts[gmap[1]].is_periodic # first check for the simple periodic case (one component, periodic geometry) and use the cheaper periodic Alpert assembly if so
            a=gmap[1] # component index of the first (and only) component in this composite component
            ra=offs[a]:(offs[a+1]-1)
            _assemble_self_alpert!(solver,A,pts[a],Gs[a],Cs[a],ra,k,rule;multithreaded=multithreaded) # if it's just one periodic component, we can use the cheaper periodic Alpert assembly
        else # otherwise, we have to use the more expensive composite component assembly which can handle multiple components and smooth joins
            _assemble_self_alpert_composite_component!(solver,A,pts,Gs,Cs,offs,k,rule,topos[c],gmap;multithreaded=multithreaded)
        end
    end
    return A
end

#################
#### HELPERS ####
#################

# CFIEAlpertWorkspace
# Struct to hold all precomputed data for CFIE Alpert assembly, including:
#   - Alpert quadrature rule
#   - Component offsets for global indexing
#   - Geometry caches for each component
#   - Alpert component caches for each component
#   - Composite topology information (if non-periodic)
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

# build_cfie_alpert_workspace
# Build the CFIEAlpertWorkspace for a given CFIE_alpert solver and boundary points. This precomputes all necessary data for efficient assembly, including geometry caches and Alpert component caches, and also analyzes the composite topology if applicable.
# Inputs:
#   - solver::CFIE_alpert{T} :
#       The CFIE_alpert solver object containing parameters like Alpert order and symmetry.
#   - pts::Vector{BoundaryPointsCFIE{T}} :
#       The boundary points for the entire geometry, which may consist of multiple components.
# Outputs:
#   - CFIEAlpertWorkspace containing all precomputed data for assembly.
function build_cfie_alpert_workspace(solver::CFIE_alpert{T},pts::Vector{BoundaryPointsCFIE{T}}) where {T<:Real}
    rule=alpert_log_rule(T,solver.alpert_order)
    offs=component_offsets(pts)
    Gs=[cfie_geom_cache(p) for p in pts]

    boundary=isnothing(solver.symmetry) ? solver.billiard.full_boundary : solver.billiard.desymmetrized_full_boundary
    flat_boundary=boundary[1] isa AbstractVector ? reduce(vcat,boundary) : boundary

    Cs=[_build_alpert_component_cache(solver,flat_boundary[a],pts[a],rule,solver.alpert_order) for a in eachindex(pts)]

    topo_data=build_join_topology(pts)
    if topo_data===nothing
        topos=nothing
        gmaps=nothing
        panel_to_comp=nothing
    else
        topos,gmaps=topo_data
        panel_to_comp=zeros(Int,length(pts))
        @inbounds for c in eachindex(gmaps),a in gmaps[c]
            panel_to_comp[a]=c
        end
    end
    Ntot=offs[end]-1
    return CFIEAlpertWorkspace(rule,offs,Gs,Cs,topos,gmaps,panel_to_comp,Ntot)
end

@inline function _check_r(r,name,i,j)
    if !(isfinite(r)) || r <= sqrt(eps(eltype(r)))
        @warn "Bad distance in $name at i=$i j=$j : r=$r"
    end
end
@inline dlp_weight(pts::BoundaryPointsCFIE,j::Int)=pts.ws[j]
@inline function slp_weight(pts::BoundaryPointsCFIE{T},j::Int,sj::T) where {T<:Real}
    return pts.ws[j]*sj
end

##############################
#### DESYMMETRIZED KERNEL ####
##############################

# _add_image_block!
# Add one smooth image contribution from source component `pb` into the
# target/source block (a,b) of the desymmetrized Alpert matrix.
#
# Inputs:
#   - A::AbstractMatrix{Complex{T}} :
#       Global system matrix being assembled.
#   - ra::UnitRange{Int} :
#       Global row range of the target component.
#   - rb::UnitRange{Int} :
#       Global column range of the source component.
#   - pa::BoundaryPointsCFIE{T} :
#       Target component points.
#   - pb::BoundaryPointsCFIE{T} :
#       Source component points.
#   - k::T :
#       Real wavenumber.
#   - qfun :
#       Function mapping a source point q -> image point qimg.
#   - tfun :
#       Function mapping a source tangent t -> image tangent timg.
#   - weight :
#       Scalar symmetry/image weight.
#   - multithreaded::Bool=true :
#       Whether to thread over source columns.
#
# Outputs:
#   - Modifies `A` in place by adding the smooth image contribution
#       -(D_img + i k S_img)
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
            inn=tyj*dx-txj*dy
            dval=weight*wd*(αD*inn*H(1,k*r)*invr)
            sval=weight*ws*(αS*H(0,k*r))
            A[gi,gj]-=(dval+ik*sval)
        end
    end
    return A
end

# _assemble_reflection_images!
# Add all reflection image contributions for one source component block.
#
# Inputs:
#   - A, ra, rb, pa, pb, solver, billiard, k :
#       Standard assembly data.
#   - sym::Reflection :
#       Reflection symmetry object.
#
# Outputs:
#   - Modifies `A` in place by adding the corresponding reflected image terms.
#
# Notes:
#   - For `:y_axis` and `:x_axis`, there is a single reflected image.
#   - For `:origin` (XYReflection), this expands into three image terms:
#       x-image, y-image, and xy-image,
#     with weights σx, σy, and σx*σy respectively.
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


# _assemble_rotation_images!
# Add all nontrivial rotation images for one source component block.
#
# Inputs:
#   - A, ra, rb, pa, pb, k :
#       Standard assembly data.
#   - sym::Rotation :
#       Rotation symmetry descriptor.
#   - costab, sintab, χ :
#       Rotation tables from `_rotation_tables(T, sym.n, sym.m)`.
#
# Outputs:
#   - Modifies `A` in place by adding the smooth rotated image terms.
#
# Notes:
#   - Direct l=0 contribution is not included here; it is assembled separately
#   - Adds l=1,...,n-1 images only.
function _assemble_rotation_images!(A::AbstractMatrix{Complex{T}},ra::UnitRange{Int},rb::UnitRange{Int},pa::BoundaryPointsCFIE{T},pb::BoundaryPointsCFIE{T},k::T,sym::Rotation,costab,sintab,χ;multithreaded::Bool=true) where {T<:Real}
    for l in 1:(sym.n-1)
        phase=χ[l+1]
        _add_image_block!(A,ra,rb,pa,pb,k,q->image_point(sym,q,l,costab,sintab),t->image_tangent(sym,t,l,costab,sintab),phase;multithreaded=multithreaded)
    end
    return A
end

# construct_matrices_symmetry!
# Assemble the CFIE_Alpert matrix on the desymmetrized/fundamental boundary.
#
# Inputs:
#   - solver::CFIE_alpert{T} :
#       Alpert-based CFIE solver with nontrivial symmetry.
#   - A::Matrix{Complex{T}} :
#       Output system matrix.
#   - pts::Vector{BoundaryPointsCFIE{T}} :
#       Boundary points on the desymmetrized boundary only.
#   - k::T :
#       Real wavenumber.
#   - multithreaded::Bool=true :
#       Whether to use threaded assembly.
#
# Outputs:
#   - Modifies `A` in place to contain the desymmetrized operator matrix.
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
    gmaps=topo_data===nothing ? nothing : topo_data[2]
    if topo_data===nothing
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
        if gmaps!==nothing
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
                inn=tyj*dx-txj*dy
                dval=wd*(αD*inn*H(1,k*r)*invr)
                sval=ws*(αS*H(0,k*r))
                A[gi,gj]-=(dval+ik*sval)
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

# construct_matrices_symmetry!
# Workspace-backed symmetry assembly for CFIE_alpert.
#
# This assembles the desymmetrized operator directly from the fundamental
# boundary data stored in `pts`, using precomputed geometry/alpert caches
# from `ws`.
#
# Inputs:
#   - solver::CFIE_alpert{T}
#   - A::Matrix{Complex{T}}
#   - pts::Vector{BoundaryPointsCFIE{T}}
#   - ws::CFIEAlpertWorkspace{T}
#   - k::T
#   - multithreaded::Bool=true
#
# Outputs:
#   - Modifies `A` in place.
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
    if topos===nothing
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
                dval=wd*(αD*inn*H(1,k*r)*invr)
                sval=wsj*(αS*H(0,k*r))
                A[gi,gj]-=(dval+ik*sval)
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

########################
#### HIGH LEVEL API ####
########################

# construct_matrices!
# High-level function to construct the CFIE Alpert system matrix. This function checks for symmetry. It is mostly legacy since it is better to precomoute the workspace and call the version of construct_matrices! that takes the workspace as an argument, but it is still useful for quick prototyping and testing.
function construct_matrices!(solver::CFIE_alpert{T},A::Matrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},k::T;multithreaded::Bool=true) where {T<:Real}
    if isnothing(solver.symmetry)
        offs=component_offsets(pts)
        αL2=Complex{T}(0,k/2)
        αM2=Complex{T}(0,one(T)/2)
        ik=Complex{T}(0,k)
        fill!(A,zero(Complex{T}))
        Gs=[cfie_geom_cache(p) for p in pts]
        rule=alpert_log_rule(T,solver.alpert_order)
        boundary=solver.billiard.full_boundary
        flat_boundary=boundary[1] isa AbstractVector ? reduce(vcat,boundary) : boundary
        Cs=[_build_alpert_component_cache(solver,flat_boundary[a],pts[a],rule,solver.alpert_order) for a in eachindex(pts)]
        nc=length(pts)
        topo_data=build_join_topology(pts)
        gmaps=topo_data===nothing ? nothing : topo_data[2]
        if topo_data===nothing
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
            if gmaps!==nothing
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
                    inn=tyj*dx-txj*dy
                    invr=inv(r)
                    dval=wd*(αL2*inn*H(1,k*r)*invr)
                    sval=ws*(αM2*H(0,k*r))
                    A[gi,gj]-=(dval+ik*sval)
                end
            end
        end
        return A
    else
        return construct_matrices_symmetry!(solver,A,pts,k;multithreaded=multithreaded)
    end
end

# construct_matrices!
# High-level function to construct the CFIE Alpert system matrix. This function handles both the symmetry and non-symmetry cases, dispatching to the appropriate assembly routines. It uses the CFIEAlpertWorkspace to access all precomputed data for efficient assembly.
# Inputs:
#   - solver::CFIE_alpert{T} :
#       The CFIE_alpert solver object containing parameters and symmetry information.
#   - A::Matrix{Complex{T}} :
#       The output system matrix to be filled.
#   - pts::Vector{BoundaryPointsCFIE{T}} :
#       The boundary points for the entire geometry.
#   - k::T :
#       The real wavenumber.
#   - ws::CFIEAlpertWorkspace{T} :
#       The precomputed workspace containing all necessary data for assembly.
#   - multithreaded::Bool=true :
#       Whether to use multithreading for assembly.
# Outputs:
#   - Modifies `A` in place to contain the assembled system matrix for the CFIE Alpert solver. 
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
        if topos===nothing
            @inbounds for a in eachindex(pts)
                ra=offs[a]:(offs[a+1]-1)
                _assemble_self_alpert!(solver,A,pts[a],Gs[a],Cs[a],ra,k,rule;multithreaded=multithreaded)
            end
        else
            _assemble_all_self_alpert_composite!(solver,A,pts,Gs,Cs,offs,k,rule,topos,gmaps;multithreaded=multithreaded)
        end
        for a in eachindex(pts), b in eachindex(pts)
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
                    dval=wd*(αD*inn*H(1,k*r)*invr)
                    sval=wsj*(αS*H(0,k*r))
                    A[gi,gj]-=(dval+ik*sval)
                end
            end
        end
        return A
    else
        return construct_matrices_symmetry!(solver,A,pts,ws,k;multithreaded=multithreaded)
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