# ==============================================================================
# POINCARÉ–HUSIMI ON ∂Ω IN THE POINCARÉ DISK
# ==============================================================================
#
# GOAL
#   Compute the (Poincaré–)Husimi phase-space density H(q,p) on the boundary ∂Ω
#   for Dirichlet hyperbolic billiards embedded in the Poincaré disk model.
#   Input data is the boundary density u(s)=∂nψ(s) obtained from the hyperbolic
#   BIM (LEFT singular vector associated with σmin(K(k))).
#
# MODEL (PoinCARÉ DISK)
#   Domain Ω ⊂ {(x,y): x^2+y^2<1} with Euclidean coordinates (x,y).
#   Metric: ds_H = λ(x,y) ds_E,  λ(x,y) = 2/(1-(x^2+y^2)).
#
# BOUNDARY COORDINATE
#   The boundary is parameterized by hyperbolic arclength ξ ∈ [0, L_H):
#       ξ[j] ≈ Σ_{m<j} ds_H[m],   L_H ≈ Σ_j ds_H[j].
#
# COHERENT STATE (BOUNDARY GAUSSIAN WAVE PACKET)
#   For each boundary phase-space point (q,p), build a boundary wavepacket
#   localized near ξ=q with width ~ 1/√k and with tangential phase exp(i k p (ξ-q)).
#   The implementation follows the standard boundary Husimi construction:
#
#       c(q,p) = ∫ u(ξ)  exp( -k(ξ-q)^2/2 )  exp( i k p (ξ-q) ) dξ
#       H(q,p) = |c(q,p)|^2
#
#   Discretization:
#       dξ is the hyperbolic boundary weight ds_H at node j.
#       The integral is approximated by Σ_j u[j] * gaussian * phase * dξ[j].
#
# NORMALIZATION
#   Returned H is normalized to unity:
#       H ./= sum(H)
#
# OUTPUT GRID CONVENTIONS
#   - full_p=true:
#       H is returned on the (qs × ps) grid with the natural ordering of ps. This is needed for e.g. GUE type symmetries.
#   - full_p=false:
#       negative p is appended by reflection to mimic the Euclidean boundary PH
#       plotting convention:
#           ps_out = [-reverse(ps)[1:end-1]; ps]
#       and H is returned on (qs × ps_out) with matching ordering. This is most cases where we have time-reversal symmetry.
#
# API
#   Two implementations are provided:
#     (A) husimi_on_grid_hyp(...)      : simple, allocates periodic vectors
#     (B) _husimi_on_grid_hyp!(...)          : allocation-minimal fast path
# ==============================================================================

# ------------------------------------------------------------------------------
# husimi_on_grid_hyp (reference implementation)
#
# PURPOSE
#   Compute boundary Husimi H(q,p) for a single state on (qs,ps).
#
# INPUTS
#   k::T
#     Wavenumber.
#   ξ::AbstractVector{T}
#     Hyperbolic boundary arclength coordinates (monotone, length n).
#   u::AbstractVector{Num}
#     Boundary density u(ξ)=∂nψ on the same nodes as ξ (length n).
#   Lh::T
#     Total hyperbolic boundary length.
#   dξ::AbstractVector{T}
#     Hyperbolic quadrature weights (typically ds_H), same length as ξ.
#   qs::AbstractVector{T}
#     q grid (boundary position coordinate).
#   ps::AbstractVector{T}
#     p grid (dimensionless tangential momentum coordinate).
#
# KEYWORDS
#   full_p::Bool=false
#     If false, append negative p branch (false by default in case of forgetting of  e.g. not having time-reversal symmetry).
#
# OUTPUTS
#   H::Matrix{T}
#     Normalized Husimi density on the (qs,ps_out) grid.
#   qs::AbstractVector{T}
#     Returned unchanged for convenience.
#   ps_out::Vector{T}
#     p grid actually used 
# ------------------------------------------------------------------------------
function husimi_on_grid_hyp(k::T,ξ::AbstractVector{T},u::AbstractVector{Num},Lh::T,dξ::AbstractVector{T},qs::AbstractVector{T},ps::AbstractVector{T};full_p::Bool=false) where{T<:Real,Num<:Number}
    n=length(ξ)
    ξ_ext=vcat(ξ.-Lh,ξ,ξ.+Lh)
    u_ext=vcat(u,u,u)
    dξ_ext=vcat(dξ,dξ,dξ)
    nx=length(qs);ny=length(ps)
    Hp=zeros(T,ny,nx)
    nf=sqrt(sqrt(k/pi))
    width=4/sqrt(k)
    c_re=Vector{T}(undef,0)
    c_im=Vector{T}(undef,0)
    si=Vector{T}(undef,0)
    @inbounds for iq in 1:nx
        q=qs[iq]+Lh
        lo=searchsortedfirst(ξ_ext,q-width)
        hi=searchsortedlast(ξ_ext,q+width)
        W=max(0,hi-lo+1)
        if length(c_re)<W
            resize!(c_re,W);resize!(c_im,W);resize!(si,W)
        end
        @inbounds for t=0:W-1
            j=lo+t
            sdiff=ξ_ext[j]-q
            si[t+1]=sdiff
            w=nf*exp(-0.5*k*sdiff*sdiff)*dξ_ext[j]
            uj=u_ext[j]
            if uj isa Real
                c_re[t+1]=w*uj
                c_im[t+1]=zero(T)
            else
                c_re[t+1]=w*real(uj)
                c_im[t+1]=w*imag(uj)
            end
        end
        @inbounds for ip in 1:ny
            kp=k*ps[ip]
            sracc=zero(T);siacc=zero(T)
            @inbounds for t in 1:W
                θ=kp*si[t]
                s_,c_=sincos(θ)
                a=c_re[t];b=c_im[t]
                sracc+=a*c_+b*s_
                siacc+=b*c_-a*s_
            end
            Hp[ip,iq]=(sracc*sracc+siacc*siacc)
        end
    end
    if full_p
        H=permutedims(Hp)
        ps_out=collect(ps)
    else
        H=vcat(reverse(Hp;dims=1),Hp[2:end,:])|>permutedims
        ps_out=vcat(-reverse(ps)[1:end-1],ps)
    end
    H./=sum(H)
    return H,qs,ps_out
end

# ------------------------------------------------------------------------------
# husimi_on_grid_hyperbolic(bp::BoundaryPointsHypBIM,...)
#
# PURPOSE
#   Convenience wrapper for single-state Husimi using BoundaryPointsHypBIM fields.
#
# INPUTS
#   bp.ξ   : hyperbolic arclength coordinate on ∂Ω
#   bp.LH  : total hyperbolic boundary length
#   bp.dsH : hyperbolic quadrature weights (ds_H)
#
# OUTPUTS
#   Same as husimi_on_grid_hyperbolic(k,ξ,u,Lh,dξ,qs,ps;...)
# ------------------------------------------------------------------------------
function husimi_on_grid_hyp(k::T,bp::BoundaryPointsHypBIM{T},u::AbstractVector{Num},qs::AbstractVector{T},ps::AbstractVector{T};full_p::Bool=false) where {T<:Real,Num<:Number}
    return husimi_on_grid_hyp(k,bp.ξ,u,bp.LH,bp.dsH,qs,ps;full_p=full_p)
end

# ------------------------------------------------------------------------------
# _HusimiWorkspace (internal)
#
# PURPOSE
#   Minimize allocations for large scale Husimi computation by storing temporary vectors:
#     - periodic ξ and dξ buffers (length 3*n)
#     - per-q wavepacket buffers (length up to 3*n):
#         packet_s   : s = ξ - q
#         packet_re  : weighted real part contributions
#         packet_im  : weighted imag part contributions
#
# NOTES
#   This is internal (not part of the public API).
# ------------------------------------------------------------------------------
mutable struct _HusimiWorkspace{T}
    ξ_periodic::Vector{T}
    dξ_periodic::Vector{T}
    packet_s::Vector{T}
    packet_re::Vector{T}
    packet_im::Vector{T}
end
_HusimiWorkspace(::Type{T}) where {T<:Real}=_HusimiWorkspace{T}(T[],T[],T[],T[],T[])

const _HUSIMI_TLS=[Dict{DataType,Any}() for _ in 1:Threads.nthreads()]

# ------------------------------------------------------------------------------
# _tls_husimi(T) (internal)
#
# PURPOSE
#   Return per-thread workspace for element type T. Allocates once per thread.
# ------------------------------------------------------------------------------
@inline function _tls_husimi(::Type{T}) where {T<:Real}
    d=_HUSIMI_TLS[Threads.threadid()]
    return get!(()->_HusimiWorkspace(T), d, T)::_HusimiWorkspace{T}
end

# ==============================================================================
# 3× PERIODIC EXTENSION (ξ_periodic length = 3n)
# ==============================================================================
#
# CONTEXT
#   The boundary coordinate ξ is periodic: ξ ≡ ξ + L_H, where L_H is the total
#   hyperbolic boundary length. The Husimi amplitude at (q,p) uses a Gaussian
#   wavepacket centered at ξ=q with finite window ~ width = O(1/√k):
#
#       c(q,p) = ∫ u(ξ) exp(-k(ξ-q)^2/2) exp(i k p (ξ-q)) dξ
#
#   Numerically we restrict the sum to ξ ∈ [q-width, q+width] because the Gaussian
#   suppresses contributions outside this interval.
#
# PROBLEM (EDGE / WRAP-AROUND)
#   If q is close to the periodic boundary (near 0 or near L_H), the interval
#   [q-width, q+width] may cross the boundary and split into two pieces:
#
#     - q ≈ 0:   [q-width, q+width] overlaps negative ξ, which should wrap to ξ+L_H
#     - q ≈ L_H: [q-width, q+width] overlaps ξ>L_H, which should wrap to ξ-L_H
#
#   Handling this directly requires modular indexing and two disjoint index ranges,
#   which complicates the code and typically prevents using a single searchsorted
#   range on a monotone ξ array.
#
# SOLUTION (CONTIGUOUS WINDOW VIA 3× EXTENSION)
#   Build a 3× periodic extension of the boundary arrays:
#
#       ξ_periodic  = [ξ - L_H;  ξ;  ξ + L_H]     (length 3n)
#       dξ_periodic = [dξ;       dξ; dξ]          (length 3n)
#
#   Then we shift the query center into the middle copy:
#
#       q̃ = q + L_H
#
#   and always search for indices in the single contiguous interval:
#
#       q̃ - width ≤ ξ_periodic[j] ≤ q̃ + width .
#
#   This guarantees that the contributing boundary nodes are found as one
#   contiguous index range [jlo:jhi] in ξ_periodic, regardless of where q lies
#   on the original periodic boundary.
# ==============================================================================
@inline function _ensure_husimi_ws!(ws::_HusimiWorkspace{T},n::Int) where {T<:Real}
    m=3*n
    length(ws.ξ_periodic)<m && resize!(ws.ξ_periodic,m)
    length(ws.dξ_periodic)<m && resize!(ws.dξ_periodic,m)
    length(ws.packet_s)<m && resize!(ws.packet_s,m)
    length(ws.packet_re)<m && resize!(ws.packet_re,m)
    length(ws.packet_im)<m && resize!(ws.packet_im,m)
    return ws
end

# ------------------------------------------------------------------------------
# _wrapidx(j,n) (internal)
#
# PURPOSE
#   Map index j in [1,3*n] back to the base index in [1,n] for periodic u access.
# ------------------------------------------------------------------------------
@inline _wrapidx(j::Int,n::Int)=1+mod(j-1,n)

# ------------------------------------------------------------------------------
# _husimi_on_grid_fast!(H,k,ξ,u,Lh,dξ,qs,ps;full_p=false) (internal)
#
# PURPOSE
#   Allocation-minimal Husimi evaluation for a single state into a preallocated H.
#   Intended for mass computation; logic matches the reference method.
#
# INPUTS
#   H::Matrix{T}
#     Output buffer. Must have size:
#       full_p=true  -> (length(qs), length(ps))
#       full_p=false -> (length(qs), 2*length(ps)-1)
#
#   Remaining inputs are identical to husimi_on_grid_hyperbolic.
#
# OUTPUTS
#   H (filled and normalized in place), returned for convenience.
#
# NOTES
#   - Avoids vcat by building periodic ξ and dξ in workspace.
#   - Uses searchsortedfirst/last on the periodic ξ buffer.
#   - Only temporary arrays live in thread-local workspace.
# ------------------------------------------------------------------------------
function _husimi_on_grid_hyp!(H::Matrix{T},k::T,ξ::AbstractVector{T},u::AbstractVector{Num},Lh::T,dξ::AbstractVector{T},qs::AbstractVector{T},ps::AbstractVector{T};full_p::Bool=false) where {T<:Real,Num<:Number}
    n=length(ξ);nq=length(qs);np=length(ps)
    ws=_tls_husimi(T);_ensure_husimi_ws!(ws,n)
    ξ_periodic=ws.ξ_periodic
    dξ_periodic=ws.dξ_periodic
    packet_s=ws.packet_s
    packet_re=ws.packet_re
    packet_im=ws.packet_im
    @inbounds for j in 1:n
        ξj=ξ[j];dξj=dξ[j]
        ξ_periodic[j]=ξj-Lh;ξ_periodic[j+n]=ξj;ξ_periodic[j+2n]=ξj+Lh
        dξ_periodic[j]=dξj;dξ_periodic[j+n]=dξj;dξ_periodic[j+2n]=dξj
    end
    pref=sqrt(sqrt(k/pi))
    window=4/sqrt(k)
    fill!(H,zero(T))
    @inbounds for iq in 1:nq
        q=qs[iq]+Lh
        jlo=searchsortedfirst(ξ_periodic,q-window)
        jhi=searchsortedlast(ξ_periodic,q+window)
        W=jhi>=jlo ? (jhi-jlo+1) : 0
        @inbounds for t in 1:W
            j=jlo+t-1
            s=ξ_periodic[j]-q
            packet_s[t]=s
            uj=u[_wrapidx(j,n)]
            w=pref*exp(-0.5*k*s*s)*dξ_periodic[j]
            packet_re[t]=w*real(uj)
            packet_im[t]=w*imag(uj)
        end
        if full_p
            @inbounds for ip in 1:np
                kp=k*ps[ip]
                reacc=zero(T);imacc=zero(T)
                @inbounds for t in 1:W
                    s_,c_=sincos(kp*packet_s[t])
                    a=packet_re[t];b=packet_im[t]
                    reacc+=a*c_+b*s_
                    imacc+=b*c_-a*s_
                end
                H[iq,ip]=reacc*reacc+imacc*imacc
            end
        else
            @inbounds for col in 1:(np-1)
                ip=np-col+1
                kp=k*ps[ip]
                reacc=zero(T);imacc=zero(T)
                @inbounds for t in 1:W
                    s_,c_=sincos(kp*packet_s[t])
                    a=packet_re[t];b=packet_im[t]
                    reacc+=a*c_+b*s_
                    imacc+=b*c_-a*s_
                end
                H[iq,col]=reacc*reacc+imacc*imacc
            end
            @inbounds for col in np:(2np-1)
                ip=col-np+1
                kp=k*ps[ip]
                reacc=zero(T);imacc=zero(T)
                @inbounds for t in 1:W
                    s_,c_=sincos(kp*packet_s[t])
                    a=packet_re[t];b=packet_im[t]
                    reacc+=a*c_+b*s_
                    imacc+=b*c_-a*s_
                end
                H[iq,col]=reacc*reacc+imacc*imacc
            end
        end
    end
    s=sum(H)
    if s>0
        invs=inv(s)
        @inbounds for j in eachindex(H)
            H[j]*=invs
        end
    end
    return H
end

# ------------------------------------------------------------------------------
# husimi_on_grid_hyp(ks,bps,us,qs,ps;...)
#
# PURPOSE
#   Compute Husimi matrices for many states with minimal allocations.
#
# INPUTS
#   ks::AbstractVector{T}
#     Wavenumbers (one per state).
#   bps::AbstractVector{BoundaryPointsHypBIM{T}}
#     Boundary containers (one per state; dimensions may differ).
#   us::AbstractVector{<:AbstractVector{Num}}
#     Boundary densities u for each state (matching bp node count).
#   qs::AbstractVector{<:AbstractVector{T}}
#     q-grids for each state (can differ per state).
#   ps::AbstractVector{<:AbstractVector{T}}
#     p-grids for each state (can differ per state).
#
# KEYWORDS
#   full_p::Bool=false
#     If false, output includes negative p branch (2np-1 columns).
#   show_progress::Bool=true
#     If true, show ProgressMeter progress bar (thread-safe update).
#
# OUTPUTS
#   Hs::Vector{Matrix{T}}
#     Husimi matrices for successful states (failed states are dropped).
# ------------------------------------------------------------------------------
function husimi_on_grid_hyp(ks::AbstractVector{T},bps::AbstractVector{BoundaryPointsHypBIM{T}},us::AbstractVector{<:AbstractVector{Num}},qs::AbstractVector{<:AbstractVector{T}},ps::AbstractVector{<:AbstractVector{T}};full_p::Bool=false,show_progress::Bool=true) where {T<:Real,Num<:Number}
    n=length(ks)
    @assert length(bps)==n && length(us)==n && length(qs)==n && length(ps)==n
    Hs=Vector{Matrix{T}}(undef,n)
    ok=trues(n)
    pbar=show_progress ? Progress(n;desc="Husimi N=$n") : nothing
    Threads.@threads for i in 1:n
        try
            qgrid=qs[i];pgrid=ps[i]
            nq=length(qgrid);np=length(pgrid)
            H=full_p ? Matrix{T}(undef,nq,np) : Matrix{T}(undef,nq,2np-1)
            bp=bps[i]
            _husimi_on_grid_hyp!(H,ks[i],bp.ξ,us[i],bp.LH,bp.dsH,qgrid,pgrid;full_p=full_p)
            Hs[i]=H
        catch
            ok[i]=false
        end
        if show_progress
            next!(pbar)
        end
    end
    return Hs[ok]
end