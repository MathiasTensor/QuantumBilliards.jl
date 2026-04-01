# Ref: HYBRID GAUSS-TRAPEZOIDAL QUADRATURE RULES, Alpert B., 1999

# Cache for simple billiards where we can use the periodic Alpert rule on the whole boundary. This is not corner-aware and does not support multiple segments.
struct AlpertPeriodicCache{T<:Real}
    xp::Matrix{T}      # jcorr × N
    yp::Matrix{T}
    txp::Matrix{T}
    typ::Matrix{T}
    sp::Matrix{T}
    xm::Matrix{T}
    ym::Matrix{T}
    txm::Matrix{T}
    tym::Matrix{T}
    sm::Matrix{T}
    idxp::Array{Int,3} # jcorr × N × 4
    wtp::Array{T,3}    # jcorr × N × 4
    idxm::Array{Int,3} # jcorr × N × 4
    wtm::Array{T,3}    # jcorr × N × 4
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

# trig_cardinal_weights!
# Compute the trigonometric cardinal weights for a given query angle and periodic nodes. This is used to construct the shifted interpolation vectors for the periodic Alpert rule. The output vector `l` is modified in place.
# Inputs:
#   - l::AbstractVector{T} : Output vector to store the weights (length N).
#   - θ::T : Angle for which to compute the weights.
#   - ts::AbstractVector{T} : Periodic nodes on [0,1] mapped to angles (length N).
# Outputs:
#   - l : Modified in place to contain the trigonometric cardinal weights for interpolation at angle θ.
function trig_cardinal_weights!(l::AbstractVector{T}, θ::T, ts::AbstractVector{T}) where {T<:Real}
    N=length(ts)
    fill!(l,zero(T))
    @inbounds for j in 1:N
        δ=wrap_diff(θ-ts[j])
        if abs(δ)<=64*eps(T)
            l[j]=one(T)
            return l
        end
    end
    invN=inv(T(N))
    if isodd(N)
        @inbounds for j in 1:N
            δ=wrap_diff(θ-ts[j])
            l[j]=invN*sin(T(N)*δ/2)/sin(δ/2)
        end
    else
        @inbounds for j in 1:N
            δ=wrap_diff(θ-ts[j])
            l[j]=invN*sin(T(N)*δ/2)*cot(δ/2)
        end
    end
    ssum=sum(l)
    abs(ssum)>0 && (@. l=l/ssum)
    return l
end


# 4-point equispaced Lagrange weights on local coordinate η ∈ [0,1),
# using nodes at offsets -1, 0, 1, 2 relative to the local left node.
#
# So if θ lies between ts[i] and ts[i+1], we interpolate using:
#   ts[i-1], ts[i], ts[i+1], ts[i+2]
#
# with local coordinate η = (θ - ts[i]) / h.
@inline function _lagrange4_weights(η::T) where {T<:Real}
    w0=-(η)*(η-one(T))*(η-T(2))/T(6) # node -1
    w1=(η+one(T))*(η-one(T))*(η-T(2))/T(2) # node  0
    w2=-(η+one(T))*(η)*(η-T(2))/T(2) # node  1
    w3=(η+one(T))*(η)*(η-one(T))/T(6) # node  2
    return w0,w1,w2,w3
end

# Evaluate one periodic shifted source point/tangent/speed using a local
# 4-point periodic stencil.
#
# Inputs:
#   θ  : target shifted angle in (0, 2π]
#   ts : equispaced periodic nodes
#   h  : angular step = 2π/N
#   X,Y,dX,dY : sampled geometry/tangent arrays at ts
#
# Output:
#   x,y,tx,ty,s : interpolated point, tangent, and speed at angle θ
@inline function _eval_shifted_source_periodic_local4(θ::T,ts::AbstractVector{T},h::T,X::AbstractVector{T},Y::AbstractVector{T},dX::AbstractVector{T},dY::AbstractVector{T}) where {T<:Real}
    N = length(ts)
    # Convert θ to fractional grid coordinate in [0, N)
    u=θ/h-one(T)
    u<zero(T) && (u+=T(N))
    u>=T(N) && (u-=T(N))
    # With ts[j] = j*h for j=1..N, the interval [ts[i], ts[i+1]) corresponds to i = floor(u)
    i0=floor(Int,u)
    η=u-T(i0) # local coordinate in [0,1)
    i=i0+1
    i>N && (i-=N)
    im1=mod1(i-1,N)
    i0j=i
    ip1=mod1(i+1,N)
    ip2=mod1(i+2,N)
    w0,w1,w2,w3=_lagrange4_weights(η)
    x=w0*X[im1]+w1*X[i0j]+w2*X[ip1]+w3*X[ip2]
    y=w0*Y[im1]+w1*Y[i0j]+w2*Y[ip1]+w3*Y[ip2]
    tx=w0*dX[im1]+w1*dX[i0j]+w2*dX[ip1]+w3*dX[ip2]
    ty=w0*dY[im1]+w1*dY[i0j]+w2*dY[ip1]+w3*dY[ip2]
    s=sqrt(tx*tx+ty*ty)
    idx=(im1,i0j,ip1,ip2)
    wt=(w0,w1,w2,w3)
    return x,y,tx,ty,s,idx,wt
end

# _build_alpert_periodic_cache
# Precompute the necessary data for applying the periodic Alpert rule to a simple billiard boundary. This includes the shifted interpolation vectors and the corresponding geometry for the shifted points. This cache is used to efficiently apply the periodic Alpert correction during matrix assembly.
# Inputs:
#   - pts::BoundaryPointsCFIE{T} : Boundary points for the CFIE discretization.
#   - rule::AlpertLogRule{T} : Alpert quadrature rule.
# Outputs:
#   - AlpertPeriodicCache{T} : Precomputed cache for the periodic Alpert rule.
function _build_alpert_periodic_cache(pts::BoundaryPointsCFIE{T}, rule::AlpertLogRule{T}) where {T<:Real}
    X=getindex.(pts.xy,1)
    Y=getindex.(pts.xy,2)
    dX=getindex.(pts.tangent,1)
    dY=getindex.(pts.tangent,2)
    ts=pts.ts
    N=length(ts)
    jcorr=rule.j
    h=pts.ws[1]
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
    idxp=Array{Int,3}(undef,jcorr,N,4)
    idxm=Array{Int,3}(undef,jcorr,N,4)
    wtp=Array{T,3}(undef,jcorr,N,4)
    wtm=Array{T,3}(undef,jcorr,N,4)
    @inbounds for p in 1:jcorr
        Δt=h*rule.x[p]
        for i in 1:N
            θp=wrap_angle(ts[i] + Δt)
            x,y,tx,ty,s,idx,wt = _eval_shifted_source_periodic_local4(θp,ts,h,X,Y,dX,dY)
            xp[p,i]=x
            yp[p,i]=y
            txp[p,i]=tx
            typ[p,i]=ty
            sp[p,i]=s
            idxp[p,i,1]=idx[1]
            idxp[p,i,2]=idx[2]
            idxp[p,i,3]=idx[3]
            idxp[p,i,4]=idx[4]
            wtp[p,i,1]=wt[1]
            wtp[p,i,2]=wt[2]
            wtp[p,i,3]=wt[3]
            wtp[p,i,4]=wt[4]
            θm=wrap_angle(ts[i]-Δt)
            x,y,tx,ty,s,idx,wt=_eval_shifted_source_periodic_local4(θm,ts,h,X,Y,dX,dY)
            xm[p,i]=x
            ym[p,i]=y
            txm[p,i]=tx
            tym[p,i]=ty
            sm[p,i]=s
            idxm[p,i,1]=idx[1]
            idxm[p,i,2]=idx[2]
            idxm[p,i,3]=idx[3]
            idxm[p,i,4]=idx[4]
            wtm[p,i,1]=wt[1]
            wtm[p,i,2]=wt[2]
            wtm[p,i,3]=wt[3]
            wtm[p,i,4]=wt[4]
        end
    end
    return AlpertPeriodicCache(xp,yp,txp,typ,sp,xm,ym,txm,tym,sm,idxp,wtp,idxm,wtm)
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

# _panel_smooth_local4_midpoint_data
# Compute the indices and weights for a local 4-point stencil centered at the midpoint of a panel. This is used for applying the Alpert correction at the panel midpoints.
# Inputs:
#   - u : Local parameter value for which to compute the stencil (should be in [0,1] for a panel).
#   - h : Step size (length of the panel divided by N).
#   - N : Number of points in the panel.
# Outputs:
#   - Tuple of indices (jm1, j, jp1, jp2) corresponding to the points used in the stencil.
#   - Tuple of weights (w0, w1, w2, w3) corresponding to the Lagrange interpolation weights for the local coordinate η.
@inline function _panel_smooth_local4_midpoint_data(u::T,h::T,N::Int) where {T<:Real}
    s=u/h-T(1)/2
    j0=floor(Int,s)+1
    η=s-floor(T,s)
    j0=clamp(j0,2,N-2)
    jm1=j0-1
    j=j0
    jp1=j0+1
    jp2=j0+2
    w0,w1,w2,w3=_lagrange4_weights(η)
    return (jm1,j,jp1,jp2),(w0,w1,w2,w3)
end

# _eval_shifted_source_smooth_panel_local4
# Evaluate the geometry, tangent, and speed at a shifted source point using a local 4-point stencil for a panel. This is used to compute the Alpert correction for points near the panel midpoints.
# Inputs:
#   - u : Local parameter value for the shifted source point.
#   - h : Step size (length of the panel divided by N).
#   - X, Y : Vectors containing the x and y coordinates of the geometry at the sampled points on the panel.
#   - dX, dY : Vectors containing the x and y components of the tangent vectors at the sampled points on the panel.
# Outputs:
#   - x, y : Interpolated x and y coordinates of the geometry at the shifted source point.
#   - tx, ty : Interpolated x and y components of the tangent vector at the shifted source point.
#   - s : Interpolated speed (magnitude of the tangent vector) at the shifted source point.
#   - idx : Tuple of indices used for the interpolation stencil.
#   - wt : Tuple of weights used for the interpolation.
@inline function _eval_shifted_source_smooth_panel_local4(u::T,h::T,X::AbstractVector{T},Y::AbstractVector{T},dX::AbstractVector{T},dY::AbstractVector{T}) where {T<:Real}
    N=length(X)
    idx,wt=_panel_smooth_local4_midpoint_data(u,h,N)
    i1,i2,i3,i4=idx
    w1,w2,w3,w4=wt
    x=w1*X[i1]+w2*X[i2]+w3*X[i3]+w4*X[i4]
    y=w1*Y[i1]+w2*Y[i2]+w3*Y[i3]+w4*Y[i4]
    tx=w1*dX[i1]+w2*dX[i2]+w3*dX[i3]+w4*dX[i4]
    ty=w1*dY[i1]+w2*dY[i2]+w3*dY[i3]+w4*dY[i4]
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
function _build_alpert_smooth_panel_cache(pts::BoundaryPointsCFIE{T},rule::AlpertLogRule{T}) where {T<:Real}
    X=getindex.(pts.xy,1)
    Y=getindex.(pts.xy,2)
    dX=getindex.(pts.tangent,1)
    dY=getindex.(pts.tangent,2)
    N=length(X)
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
    idxp=Array{Int,3}(undef,jcorr,N,4)
    idxm=Array{Int,3}(undef,jcorr,N,4)
    wtp=Array{T,3}(undef,jcorr,N,4)
    wtm=Array{T,3}(undef,jcorr,N,4)
    us=_panel_us(T,N)
    @inbounds for p in 1:jcorr
        Δu=h*rule.x[p]
        for i in 1:N
            up=us[i]+Δu
            x,y,tx,ty,s,idx,wt=_eval_shifted_source_smooth_panel_local4(up,h,X,Y,dX,dY)
            xp[p,i]=x
            yp[p,i]=y
            txp[p,i]=tx
            typ[p,i]=ty
            sp[p,i]=s
            idxp[p,i,1]=idx[1]
            idxp[p,i,2]=idx[2]
            idxp[p,i,3]=idx[3]
            idxp[p,i,4]=idx[4]
            wtp[p,i,1]=wt[1]
            wtp[p,i,2]=wt[2]
            wtp[p,i,3]=wt[3]
            wtp[p,i,4]=wt[4]
            um=us[i]-Δu
            x,y,tx,ty,s,idx,wt=_eval_shifted_source_smooth_panel_local4(um,h,X,Y,dX,dY)
            xm[p,i]=x
            ym[p,i]=y
            txm[p,i]=tx
            tym[p,i]=ty
            sm[p,i]=s
            idxm[p,i,1]=idx[1]
            idxm[p,i,2]=idx[2]
            idxm[p,i,3]=idx[3]
            idxm[p,i,4]=idx[4]
            wtm[p,i,1]=wt[1]
            wtm[p,i,2]=wt[2]
            wtm[p,i,3]=wt[3]
            wtm[p,i,4]=wt[4]
        end
    end
    return AlpertSmoothPanelCache(us,xp,yp,txp,typ,sp,xm,ym,txm,tym,sm,idxp,wtp,idxm,wtm)
end

# _build_alpert_component_cache
# Dispatch to the appropriate cache builder based on whether the boundary is periodic or panelized.
# Inputs:
#   - pts::BoundaryPointsCFIE{T} : Boundary points for the CFIE discretization.
#   - rule::AlpertLogRule{T} : Alpert quadrature rule.
# Outputs:
#   - AlpertComponentCache{T} : Precomputed cache for the Alpert rule, either periodic or panel-based.
function _build_alpert_component_cache(pts::BoundaryPointsCFIE{T},rule::AlpertLogRule{T}) where {T<:Real}
    return pts.is_periodic ? _build_alpert_periodic_cache(pts,rule) : _build_alpert_smooth_panel_cache(pts,rule)
end

###########################################################
################ SELF ALPERT ASSEMBLY #####################
###########################################################

# _assemble_self_alpert_periodic!
# Assemble the self-interaction block of the CFIE matrix for a periodic boundary using the periodic Alpert rule. This includes the standard diagonal and off-diagonal contributions from the DLP and SLP, as well as the near correction using the precomputed shifted interpolation vectors from the AlpertPeriodicCache.
# Inputs:
#   - A::AbstractMatrix{Complex{T}} : Matrix to assemble into (modified in place).
#   - pts::BoundaryPointsCFIE{T} : Boundary points for the CFIE discretization.
#   - G::CFIEGeomCache{T} : Precomputed geometric quantities for the CFIE assembly.
#   - C::AlpertPeriodicCache{T} : Precomputed cache for the periodic Alpert rule.
#   - row_range::UnitRange{Int} : Row indices corresponding to the current boundary component.
#   - k::T : Wave number.
#   - rule::AlpertLogRule{T} : Alpert quadrature rule.
#   - multithreaded::Bool : Whether to use multithreading for assembly.
# Outputs:
#   - A : Modified in place with the self-interaction block assembled using the periodic Alpert rule.
function _assemble_self_alpert_periodic!(A::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},G::CFIEGeomCache{T},C::AlpertPeriodicCache{T},row_range::UnitRange{Int},k::T,rule::AlpertLogRule{T};multithreaded::Bool=true) where {T<:Real}
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
        # diagonal
        A[gi,gi]+=one(Complex{T})-Complex{T}(h*si*κi,zero(T))
        # DLP off-diagonal
        @inbounds for j in 1:N
            j==i && continue
            gj=row_range[j]
            rij=G.R[i,j]
            inn=G.inner[i,j]
            invr=G.invR[i,j]
            A[gi,gj]-=h*(αD*inn*H(1,k*rij)*invr)
        end
        # SLP far part
        @inbounds for j in 1:N
            j==i && continue
            m=j-i
            m>N÷2 && (m-=N)
            m<-N÷2 && (m+=N)
            abs(m)<a && continue
            gj=row_range[j]
            A[gi,gj]-=ik*(h*(αS*H(0,k*G.R[i,j])*G.speed[j]))
        end
        # Near correction: scatter onto the 4-point source stencil
        @inbounds for p in 1:jcorr
            fac=h*rule.w[p]
            dx=xi-C.xp[p,i]
            dy=yi-C.yp[p,i]
            r=sqrt(dx*dx+dy*dy)
            coeff= -ik*(fac*(αS*H(0,k*r)*C.sp[p,i]))
            for m in 1:4
                q=C.idxp[p,i,m]
                A[gi,row_range[q]]+=coeff*C.wtp[p,i,m]
            end
            dx=xi-C.xm[p,i]
            dy=yi-C.ym[p,i]
            r=sqrt(dx*dx+dy*dy)
            coeff= -ik*(fac*(αS*H(0,k*r)*C.sm[p,i]))
            for m in 1:4
                q=C.idxm[p,i,m]
                A[gi,row_range[q]]+=coeff*C.wtm[p,i,m]
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
                    for m in 1:4
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
                    for m in 1:4
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

function _reverse_component_curves(comp::Vector{C}) where {C<:AbsCurve}
    return reverse(comp)
end

function build_join_topology(solver::CFIE_alpert{T},billiard::Bi;xtol::T=T(1e-10),angtol::T=T(1e-8)) where {T<:Real,Bi<:AbsBilliard}
    boundary=isnothing(solver.symmetry) ? billiard.full_boundary : billiard.desymmetrized_full_boundary
    if !(boundary[1] isa AbstractVector)
        return nothing
    end
    topos=Vector{AlpertCompositeTopology{T}}(undef,length(boundary))
    gmaps=Vector{Vector{Int}}(undef,length(boundary))
    pos=1
    @inbounds for c in eachindex(boundary)
        comp=(c==1) ? boundary[c] : _reverse_component_curves(boundary[c])
        nseg=length(comp)
        gmaps[c]=collect(pos:(pos+nseg-1))
        pos+=nseg
        periodic=_is_component_closed(comp,xtol)
        topos[c]=build_component_join_topology(comp;xtol=xtol,angtol=angtol,periodic=periodic)
    end
    return topos,gmaps
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
                        for m4 in 1:4
                            q=Ca.idxp[p,i,m4]
                            A[gi,ra[q]]+=coeff*Ca.wtp[p,i,m4]
                        end
                    end
                end
                if ui-Δu>zero(T)
                    dx=xi-Ca.xm[p,i]
                    dy=yi-Ca.ym[p,i]
                    r=sqrt(dx*dx+dy*dy)
                    if isfinite(r) && r>sqrt(eps(T))
                        coeff=-ik*(fac*(αS*H(0,k*r)*Ca.sm[p,i]))
                        for m4 in 1:4
                            q=Ca.idxm[p,i,m4]
                            A[gi,ra[q]]+=coeff*Ca.wtm[p,i,m4]
                        end
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
                            coeff=-ik*(fac*(αS*H(0,k*r)*s2))
                            _scatter_local4!(A,gi,next_ra,coeff,idx2,wt2)
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
                            coeff=-ik*(fac*(αS*H(0,k*r)*s2))
                            _scatter_local4!(A,gi,prev_ra,coeff,idx2,wt2)
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

# _assemble_self_alpert!
# Assemble the self-panel CFIE block using:
#   - plain trapezoid for DLP off-diagonal
#   - plain trapezoid for far SLP
#   - endpoint-special Alpert corrections near the left/right ends
#   - on-the-fly interior Alpert corrections away from the ends
#
# Inputs:
#   - solver,A,pts,G,C,row_range,k,rule :
#       Standard self-block assembly data.
#   - multithreaded::Bool=true :
#       Whether to thread over target rows.
#
# Outputs:
#   - Modifies `A` in place.
#
# Notes:
#   - This assumes `pts` is ONE smooth panel.
#   - It is not yet a corner-aware or multi-segment panelwise implementation.
# ---------------------------------------------------------
function _assemble_self_alpert!(solver::CFIE_alpert{T},A::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},G::CFIEGeomCache{T},C,row_range::UnitRange{Int},k::T,rule::AlpertLogRule{T};multithreaded::Bool=true) where {T<:Real}
    return  pts.is_periodic ? _assemble_self_alpert_periodic!(A,pts,G,C,row_range,k,rule;multithreaded=multithreaded) : _assemble_self_alpert_smooth_panel!(solver,A,pts,G,C,row_range,k,rule;multithreaded=multithreaded)
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
    @inbounds for c in eachindex(gmaps)
        _assemble_self_alpert_composite_component!(solver,A,pts,Gs,Cs,offs,k,rule,topos[c],gmaps[c];multithreaded=multithreaded)
    end
    return A
end

##############################
#### DESYMMETRIZED KERNEL ####
##############################

@inline function _check_r(r,name,i,j)
    if !(isfinite(r)) || r <= sqrt(eps(eltype(r)))
        @warn "Bad distance in $name at i=$i j=$j : r=$r"
    end
end

@inline dlp_weight(pts::BoundaryPointsCFIE,j::Int)=pts.ws[j]
@inline function slp_weight(pts::BoundaryPointsCFIE{T},j::Int,sj::T) where {T<:Real}
    return pts.ws[j]*sj
end

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
#=
function construct_matrices_symmetry!(solver::CFIE_alpert{T},A::Matrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},k::T;multithreaded::Bool=true) where {T<:Real}
    symmetry=solver.symmetry
    isnothing(symmetry) && error("construct_matrices_symmetry! called with symmetry = nothing")
    offs=component_offsets(pts)
    fill!(A,zero(Complex{T}))
    rule=alpert_log_rule(T,solver.alpert_order)
    Gs=[cfie_geom_cache(p) for p in pts]
    Cs=[_build_alpert_component_cache(pts[a],rule) for a in eachindex(pts)]
    nc=length(pts)
    for a in 1:nc
        ra=offs[a]:(offs[a+1]-1)
        _assemble_self_alpert!(solver,A,pts[a],Gs[a],Cs[a],ra,k,rule;multithreaded=multithreaded)
    end
    αD=Complex{T}(0,k/2)
    αS=Complex{T}(0,one(T)/2)
    ik=Complex{T}(0,k)
    for a in 1:nc, b in 1:nc
        a==b && continue
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
    # symmetry image contributions
    for sym in symmetry
        if sym isa Reflection
            for a in 1:nc, b in 1:nc
                ra=offs[a]:(offs[a+1]-1)
                rb=offs[b]:(offs[b+1]-1)
                _assemble_reflection_images!(A,ra,rb,pts[a],pts[b],solver,solver.billiard,k,sym;multithreaded=multithreaded)
            end
        elseif sym isa Rotation
            costab,sintab,χ=_rotation_tables(T,sym.n,sym.m)
            for a in 1:nc, b in 1:nc
                ra=offs[a]:(offs[a+1]-1)
                rb=offs[b]:(offs[b+1]-1)
                _assemble_rotation_images!(A,ra,rb,pts[a],pts[b],k,sym,costab,sintab,χ;multithreaded=multithreaded)
            end
        else
            error("Unknown symmetry type $(typeof(sym))")
        end
    end
    return A
end
=#
function construct_matrices_symmetry!(solver::CFIE_alpert{T},A::Matrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},k::T;multithreaded::Bool=true) where {T<:Real}
    symmetry=solver.symmetry
    isnothing(symmetry) && error("construct_matrices_symmetry! called with symmetry = nothing")
    offs=component_offsets(pts)
    fill!(A,zero(Complex{T}))
    rule=alpert_log_rule(T,solver.alpert_order)
    Gs=[cfie_geom_cache(p) for p in pts]
    Cs=[_build_alpert_component_cache(pts[a],rule) for a in eachindex(pts)]
    nc=length(pts)
    topo_data=build_join_topology(solver,solver.billiard)
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
    for sym in symmetry
        if sym isa Reflection
            for a in 1:nc, b in 1:nc
                ra=offs[a]:(offs[a+1]-1)
                rb=offs[b]:(offs[b+1]-1)
                _assemble_reflection_images!(A,ra,rb,pts[a],pts[b],solver,solver.billiard,k,sym;multithreaded=multithreaded)
            end
        elseif sym isa Rotation
            costab,sintab,χ=_rotation_tables(T,sym.n,sym.m)
            for a in 1:nc, b in 1:nc
                ra=offs[a]:(offs[a+1]-1)
                rb=offs[b]:(offs[b+1]-1)
                _assemble_rotation_images!(A,ra,rb,pts[a],pts[b],k,sym,costab,sintab,χ;multithreaded=multithreaded)
            end
        else
            error("Unknown symmetry type $(typeof(sym))")
        end
    end
    return A
end

########################
#### HIGH LEVEL API ####
########################

#=
function construct_matrices!(solver::CFIE_alpert{T},A::Matrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},k::T;multithreaded::Bool=true) where {T<:Real}
    if isnothing(solver.symmetry)
        offs=component_offsets(pts)
        αL2=Complex{T}(0,k/2)
        αM2=Complex{T}(0,one(T)/2)
        ik=Complex{T}(0,k)
        fill!(A,zero(Complex{T}))
        Gs=[cfie_geom_cache(p) for p in pts]
        rule=alpert_log_rule(T,solver.alpert_order)
        Cs=[_build_alpert_component_cache(pts[a],rule) for a in eachindex(pts)]
        nc=length(pts)
        for a in 1:nc
            ra=offs[a]:(offs[a+1]-1)
            _assemble_self_alpert!(solver,A,pts[a],Gs[a],Cs[a],ra,k,rule;multithreaded=multithreaded)
        end
        for a in 1:nc, b in 1:nc
            a==b && continue
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
=#
function construct_matrices!(solver::CFIE_alpert{T},A::Matrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},k::T;multithreaded::Bool=true) where {T<:Real}
    if isnothing(solver.symmetry)
        offs=component_offsets(pts)
        αL2=Complex{T}(0,k/2)
        αM2=Complex{T}(0,one(T)/2)
        ik=Complex{T}(0,k)
        fill!(A,zero(Complex{T}))
        Gs=[cfie_geom_cache(p) for p in pts]
        rule=alpert_log_rule(T,solver.alpert_order)
        Cs=[_build_alpert_component_cache(pts[a],rule) for a in eachindex(pts)]
        nc=length(pts)
        topo_data=build_join_topology(solver,solver.billiard)
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

function construct_matrices(solver::CFIE_alpert,pts::Vector{BoundaryPointsCFIE{T}},k::T;multithreaded::Bool=true) where {T<:Real}
    Ntot=boundary_matrix_size(pts)
    A=Matrix{Complex{T}}(undef,Ntot,Ntot)
    construct_matrices!(solver,A,pts,k;multithreaded=multithreaded)
    return A
end

############################
#### SOLVE WRAPPERS ########
############################

function solve(solver::CFIE_alpert,basis::Ba,pts::Vector{BoundaryPointsCFIE{T}},k;multithreaded::Bool=true,use_krylov::Bool=true) where {T<:Real,Ba<:AbsBasis}
    A=construct_matrices(solver,pts,k;multithreaded=multithreaded)
    if use_krylov
        @blas_multi_then_1 MAX_BLAS_THREADS mu,_,_,_=svdsolve(A,1,:SR)
        return mu[1]
    else
        s=svdvals(A)
        return s[end]
    end
end

function solve_vect(solver::CFIE_alpert,basis::Ba,pts::Vector{BoundaryPointsCFIE{T}},k;multithreaded::Bool=true) where {T<:Real,Ba<:AbsBasis}
    A=construct_matrices(solver,pts,k;multithreaded=multithreaded)
    _,S,Vt=LAPACK.gesvd!('A','A',A)
    idx=findmin(S)[2]
    return S[idx],real.(Vt[idx,:])
end

function solve_INFO(solver::CFIE_alpert,basis::Ba,pts::Vector{BoundaryPointsCFIE{T}},k;multithreaded::Bool=true,use_krylov::Bool=true) where {T<:Real,Ba<:AbsBasis}
    offs=component_offsets(pts)
    Ntot=offs[end]-1
    A=Matrix{Complex{T}}(undef,Ntot,Ntot)
    t0=time()
    @info "Building boundary operator A..."
    construct_matrices!(solver,A,pts,k;multithreaded=multithreaded)
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
    println("────────── SOLVE_INFO SUMMARY ──────────")
    println("A-matrix build: ",100*build_A/total," %")
    println("SVD: ",100*svd_time/total," %")
    println("(total: ",total," s)")
    println("────────────────────────────────────────")
    return s[end]
end