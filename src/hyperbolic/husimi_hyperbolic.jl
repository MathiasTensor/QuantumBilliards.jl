# ==============================================================================
# POINCARE-HUSIMI FUNCTIONS ON ∂Ω IN THE POINCARE DISK
# ==============================================================================
#
# Hyperbolic billiard in the Poincare disk:
#
#     ds_H = λ ds_E,
#     λ(x,y) = 2/(1-x^2-y^2).
#
# The boundary coordinate used for the coherent state is hyperbolic arclength
#
#     ξ ∈ [0,LH).
#
# The boundary function used here is assumed to come from the adjoint-kernel
# construction, hence
#
#     u = u_E = ∂_{n_E}ψ.
#
# Therefore the correct invariant product is
#
#     u_H ds_H = u_E ds_E,
#
# and the Husimi integral is discretized with Euclidean boundary weights `ds`,
# not hyperbolic weights `dsH`.
#
# The boundary coherent state is the periodic 1D coherent state used in the
# standard boundary Poincare-Husimi construction:
#
#     c_{q,p}(ξ) ∼ exp[-k(ξ-q)^2/2] exp[i k p (ξ-q)].
#
# Bäcker--Fürstberger--Schubert use the same boundary-function idea: project
# u_n(s)=∂nψ_n onto periodic coherent states in boundary arclength.
# MO 4/6/26
# ==============================================================================

"""
    make_qp_grids_hyp(ks,bps; q_oversample=2.0,nq_min=1000,np=1000,pmax=one(T),full_p=false)

Construct per-state Poincare-Husimi grids.
Inputs:
- `ks::AbstractVector{T}`: Wavenumbers.
- `bps::AbstractVector{BoundaryPointsHyp{T}}`: Boundary containers with `LH`.
Keywords:
- `q_oversample::Float64`: Oversampling relative to `k*LH/(2π)`.
- `nq_min::Int`: Minimum number of q-points.
- `np::Int`: Number of p-points.
- `pmax::T`: Maximum absolute tangential momentum.
- `full_p::Bool`: If true, use `p∈[-pmax,pmax]`; otherwise use `p∈[0,pmax]`.
Outputs:
- `qs::Vector{Vector{T}}`: Hyperbolic arclength grids `q∈[0,LH)`.
- `ps::Vector{Vector{T}}`: Momentum grids.
Mathematical convention:
- `q` is hyperbolic arclength.
- `p=sin(χ)` is the dimensionless tangential momentum.
"""
function make_qp_grids_hyp(ks::AbstractVector{T},bps::AbstractVector{BoundaryPointsHyp{T}};q_oversample::Float64=2.0,nq_min::Int=1000,np::Int=1000,pmax::T=one(T),full_p::Bool=false) where {T<:Real}
    n=length(ks)
    qs=Vector{Vector{T}}(undef,n)
    ps=Vector{Vector{T}}(undef,n)
    @inbounds for i in 1:n
        k=ks[i]
        LH=bps[i].LH
        # Semiclassical resolution in q:
        # one boundary wavelength is ≈2π/k, so k*LH/(2π) is the natural
        # number of boundary oscillations along ∂Ω.
        nq=max(nq_min,ceil(Int,q_oversample*k*LH/(2π)))
        # Periodic q-grid on [0,LH); endpoint is dropped to avoid duplication.
        qs[i]=collect(range(zero(T),LH;length=nq+1))[1:end-1]
        # p=sinχ is dimensionless tangential momentum.
        # full_p=true is needed for complex sectors where H(q,p)≠H(q,-p).
        ps[i]=full_p ? collect(range(-pmax,pmax;length=np)) : collect(range(zero(T),pmax;length=np))
    end
    return qs,ps
end

"""
    husimi_on_grid_hyp(k,ξ,u,Lh,w,qs,ps; full_p=false)

Compute one normalized boundary Poincare-Husimi density.
Inputs:
- `k::T`: Wavenumber.
- `ξ::AbstractVector{T}`: Hyperbolic arclength nodes.
- `u::AbstractVector{Num}`: Boundary function samples.
- `Lh::T`: Total hyperbolic boundary length.
- `w::AbstractVector{T}`: Quadrature weights paired with `u`.
- `qs::AbstractVector{T}`: q-grid in hyperbolic arclength.
- `ps::AbstractVector{T}`: p-grid.
Output:
- `(H,qs,ps_out)`: Normalized Husimi matrix and grids.
Mathematical convention:
- Computes
      h(q,p) = ∑_j u_j exp[-k(ξ_j-q)^2/2] exp[-i k p(ξ_j-q)] w_j
  and returns `H(q,p)=|h(q,p)|²`, normalized by `sum(H)`.
- For the current adjoint-kernel convention, `u=∂ₙᴱψ`, hence use `w=ds_E`.
- If instead `u=∂ₙᴴψ`, then use `w=ds_H`.
"""
function husimi_on_grid_hyp(k::T,ξ::AbstractVector{T},u::AbstractVector{Num},Lh::T,w::AbstractVector{T},qs::AbstractVector{T},ps::AbstractVector{T};full_p::Bool=false) where {T<:Real,Num<:Number}
    n=length(ξ)
    # Periodic extension:
    # The coherent state is localized in ξ but lives on a circle of length Lh.
    # The 3-copy array makes wrap-around windows near q≈0 and q≈Lh contiguous.
    ξ_ext=vcat(ξ.-Lh,ξ,ξ.+Lh)
    u_ext=vcat(u,u,u)
    w_ext=vcat(w,w,w)
    nq=length(qs)
    np=length(ps)
    Hp=zeros(T,np,nq)
    # 1D coherent-state normalization factor for
    # exp[-k(ξ-q)^2/2]. The global normalization is not critical because H is
    # normalized by sum(H), but keeping this factor makes amplitudes comparable.
    pref=sqrt(sqrt(k/pi))
    # Truncate the Gaussian at 4 standard widths.
    # Since width σ≈1/sqrt(k), the omitted tail is exponentially small.
    window=4/sqrt(k)
    c_re=Vector{T}(undef,0)
    c_im=Vector{T}(undef,0)
    sbuf=Vector{T}(undef,0)
    @inbounds for iq in 1:nq
        q=qs[iq]
        # Find the active Gaussian support in the periodic extension.
        lo=searchsortedfirst(ξ_ext,q-window)
        hi=searchsortedlast(ξ_ext,q+window)
        W=max(0,hi-lo+1)
        if length(c_re)<W
            resize!(c_re,W)
            resize!(c_im,W)
            resize!(sbuf,W)
        end
        @inbounds for t in 1:W
            j=lo+t-1
            s=ξ_ext[j]-q
            # For the adjoint-kernel convention:
            #   u[j] = ∂_{n_E}ψ(j)
            # and the correct pairing is u_E ds_E.
            # Hence callers should pass w=bp.ds, not bp.dsH.
            amp=pref*exp(-0.5*k*s*s)*w_ext[j]
            uj=u_ext[j]
            sbuf[t]=s
            c_re[t]=amp*real(uj)
            c_im[t]=amp*imag(uj)
        end
        @inbounds for ip in 1:np
            kp=k*ps[ip]
            reacc=zero(T)
            imacc=zero(T)
            @inbounds for t in 1:W
                # We compute projection against exp[-i k p(ξ-q)].
                # If the opposite sign convention is used, the p-axis is reversed
                # for complex states; |h|² is unchanged in TRI symmetric cases.
                sθ,cθ=sincos(kp*sbuf[t])
                a=c_re[t]
                b=c_im[t]
                # (a+ib)*exp(-iθ) = (a cosθ + b sinθ) + i(b cosθ - a sinθ).
                reacc+=a*cθ+b*sθ
                imacc+=b*cθ-a*sθ
            end
            Hp[ip,iq]=reacc*reacc+imacc*imacc
        end
    end
    if full_p
        H=permutedims(Hp)
        ps_out=collect(ps)
    else
        # Mirror positive branch to negative p.
        # This should only be used when the state/symmetry sector satisfies
        # H(q,p)=H(q,-p), e.g. real TRI sectors.
        H=permutedims(vcat(reverse(Hp;dims=1),Hp[2:end,:]))
        ps_out=vcat(-reverse(ps)[1:end-1],ps)
    end
    # Normalize for plotting/statistics.
    # This is not the same as the semiclassical invariant measure normalization.
    sH=sum(H)
    sH>zero(T) && (H./=sH)

    return H,qs,ps_out
end

"""
    husimi_on_grid_hyp(k,bp,u,qs,ps; full_p=false)

Convenience wrapper for one hyperbolic boundary state.

Inputs:
- `k::T`: Wavenumber.
- `bp::BoundaryPointsHyp{T}`: Boundary data with `ξ`, `LH`, and `ds`.
- `u::AbstractVector{Num}`: Boundary function from the adjoint kernel, `u=∂ₙᴱψ`.
- `qs::AbstractVector{T}`: q-grid.
- `ps::AbstractVector{T}`: p-grid.
Output:
- `(H,qs,ps_out)`: Normalized Husimi matrix and grids.
Important:
- Uses `bp.ds`, not `bp.dsH`.
- This is correct because the adjoint-kernel boundary function is Euclidean-normal.
"""
function husimi_on_grid_hyp(k::T,bp::BoundaryPointsHyp{T},u::AbstractVector{Num},qs::AbstractVector{T},ps::AbstractVector{T};full_p::Bool=false) where {T<:Real,Num<:Number}
    # Important:
    # The adjoint hyperbolic DLP kernel returns u_E=∂_{n_E}ψ.
    # Therefore the coherent-state integral uses u_E ds_E.
    # The coordinate is still ξ=s_H, so we pass bp.ξ but bp.ds.
    return husimi_on_grid_hyp(k,bp.ξ,u,bp.LH,bp.ds,qs,ps;full_p=full_p)
end

"""
    _HusimiWorkspace{T}

Thread-local workspace for allocation-minimal Husimi evaluation.

Fields:
- `ξ_periodic::Vector{T}`: Three-copy periodic arclength buffer.
- `w_periodic::Vector{T}`: Three-copy quadrature-weight buffer.
- `packet_s::Vector{T}`: Local differences `ξ-q`.
- `packet_re::Vector{T}`: Weighted real contributions.
- `packet_im::Vector{T}`: Weighted imaginary contributions.
Purpose:
- Avoid repeated allocations in large multi-state Husimi computation.
"""
mutable struct _HusimiWorkspace{T}
    ξ_periodic::Vector{T}
    w_periodic::Vector{T}
    packet_s::Vector{T}
    packet_re::Vector{T}
    packet_im::Vector{T}
end

_HusimiWorkspace(::Type{T}) where {T<:Real}=_HusimiWorkspace{T}(T[],T[],T[],T[],T[])
const _HUSIMI_WS=Dict{Tuple{Int,DataType},Any}()
const _HUSIMI_WS_LOCK=ReentrantLock()

"""
    _tls_husimi(T) -> _HusimiWorkspace{T}

Return the thread-local Husimi workspace for element type `T`.
"""
@inline function _tls_husimi(::Type{T}) where {T<:Real}
    key=(Threads.threadid(),T)
    ws=get(_HUSIMI_WS,key,nothing)
    ws===nothing || return ws::_HusimiWorkspace{T}
    lock(_HUSIMI_WS_LOCK) do
        return get!(_HUSIMI_WS,key) do
            _HusimiWorkspace(T)
        end::_HusimiWorkspace{T}
    end
end

"""
    _ensure_husimi_ws!(ws,n) -> _HusimiWorkspace{T}

Resize workspace buffers for a boundary with `n` nodes.
Mathematical reason:
- The boundary coordinate is periodic.
- The Gaussian packet near `q≈0` or `q≈LH` needs wrap-around contributions.
- A three-copy buffer `[ξ-LH; ξ; ξ+LH]` makes the active window contiguous.
"""
@inline function _ensure_husimi_ws!(ws::_HusimiWorkspace{T},n::Int) where {T<:Real}
    m=3*n
    # Need 3n because we store [ξ-Lh; ξ; ξ+Lh].
    length(ws.ξ_periodic)<m && resize!(ws.ξ_periodic,m)
    length(ws.w_periodic)<m && resize!(ws.w_periodic,m)
    # These only need the active packet length W, but allocating 3n once avoids
    # repeated resizing and keeps threaded calls allocation-light.
    length(ws.packet_s)<m && resize!(ws.packet_s,m)
    length(ws.packet_re)<m && resize!(ws.packet_re,m)
    length(ws.packet_im)<m && resize!(ws.packet_im,m)

    return ws
end

"""
    _wrapidx(j,n) -> Int

Map an index from the three-copy periodic buffer back to the original boundary
index range `1:n`.
"""
@inline _wrapidx(j::Int,n::Int)=1+mod(j-1,n)

"""
    _husimi_on_grid_hyp!(H,k,ξ,u,Lh,w,qs,ps; full_p=false) -> Matrix{T}

Allocation-minimal boundary Poincare-Husimi evaluation.

Inputs:
- `H::Matrix{T}`: Output matrix.
- `k::T`: Wavenumber.
- `ξ::AbstractVector{T}`: Hyperbolic arclength nodes.
- `u::AbstractVector{Num}`: Boundary function samples.
- `Lh::T`: Total hyperbolic boundary length.
- `w::AbstractVector{T}`: Quadrature weights paired with `u`.
- `qs::AbstractVector{T}`: q-grid.
- `ps::AbstractVector{T}`: p-grid.
Output:
- `H::Matrix{T}`: Filled and normalized in place.
Mathematical convention:
- Uses the packet
      exp[-k(ξ-q)^2/2] exp[-i k p(ξ-q)].
- The sign in the phase is immaterial after `abs2` for real/TRI sectors, but for
  complex sectors it fixes the plotted orientation in p.
- With adjoint-kernel boundary functions, pass `w=bp.ds`.
"""
function _husimi_on_grid_hyp!(H::Matrix{T},k::T,ξ::AbstractVector{T},u::AbstractVector{Num},Lh::T,w::AbstractVector{T},qs::AbstractVector{T},ps::AbstractVector{T};full_p::Bool=false) where {T<:Real,Num<:Number}
    n=length(ξ)
    nq=length(qs)
    np=length(ps)
    ws=_tls_husimi(T)
    _ensure_husimi_ws!(ws,n)
    ξ_periodic=ws.ξ_periodic
    w_periodic=ws.w_periodic
    packet_s=ws.packet_s
    packet_re=ws.packet_re
    packet_im=ws.packet_im
    @inbounds for j in 1:n
        ξj=ξ[j]
        wj=w[j]
        # Periodic coordinate copies.
        ξ_periodic[j]=ξj-Lh
        ξ_periodic[j+n]=ξj
        ξ_periodic[j+2n]=ξj+Lh
        # Weight copies. For adjoint-kernel u, w=ds_E.
        w_periodic[j]=wj
        w_periodic[j+n]=wj
        w_periodic[j+2n]=wj
    end
    pref=sqrt(sqrt(k/pi))
    window=4/sqrt(k)
    fill!(H,zero(T))
    @inbounds for iq in 1:nq
        q=qs[iq]
        # Active local packet support in the periodic coordinate.
        jlo=searchsortedfirst(ξ_periodic,q-window)
        jhi=searchsortedlast(ξ_periodic,q+window)
        W=jhi>=jlo ? jhi-jlo+1 : 0
        @inbounds for t in 1:W
            j=jlo+t-1
            s=ξ_periodic[j]-q
            # Pull u from the original boundary index.
            # The periodic copies represent the same boundary value.
            uj=u[_wrapidx(j,n)]
            # Boundary coherent-state envelope:
            #   exp[-k s^2/2],
            # with s=ξ-q in hyperbolic arclength.
            amp=pref*exp(-0.5*k*s*s)*w_periodic[j]
            packet_s[t]=s
            packet_re[t]=amp*real(uj)
            packet_im[t]=amp*imag(uj)
        end
        if full_p
            @inbounds for ip in 1:np
                kp=k*ps[ip]
                reacc=zero(T)
                imacc=zero(T)
                @inbounds for t in 1:W
                    sθ,cθ=sincos(kp*packet_s[t])
                    a=packet_re[t]
                    b=packet_im[t]
                    # Projection against exp[-i k p s].
                    reacc+=a*cθ+b*sθ
                    imacc+=b*cθ-a*sθ
                end
                H[iq,ip]=reacc*reacc+imacc*imacc
            end
        else
            @inbounds for col in 1:(np-1)
                # Negative-p branch by reflection.
                # This avoids recomputation, but should only be used when the
                # symmetry sector really has p↦-p symmetry.
                ip=np-col+1
                kp=k*ps[ip]
                reacc=zero(T)
                imacc=zero(T)
                @inbounds for t in 1:W
                    sθ,cθ=sincos(kp*packet_s[t])
                    a=packet_re[t]
                    b=packet_im[t]
                    reacc+=a*cθ+b*sθ
                    imacc+=b*cθ-a*sθ
                end
                H[iq,col]=reacc*reacc+imacc*imacc
            end
            @inbounds for col in np:(2np-1)
                # Nonnegative-p branch.
                ip=col-np+1
                kp=k*ps[ip]
                reacc=zero(T)
                imacc=zero(T)
                @inbounds for t in 1:W
                    sθ,cθ=sincos(kp*packet_s[t])
                    a=packet_re[t]
                    b=packet_im[t]
                    reacc+=a*cθ+b*sθ
                    imacc+=b*cθ-a*sθ
                end
                H[iq,col]=reacc*reacc+imacc*imacc
            end
        end
    end
    # Normalize density for downstream entropy/IPR/statistical analysis.
    sH=sum(H)
    if sH>zero(T)
        invsH=inv(sH)
        @inbounds for j in eachindex(H)
            H[j]*=invsH
        end
    end
    return H
end



"""
    husimi_on_grid_hyp(ks,bps,us,qs,ps; full_p=false,show_progress=true) -> Vector{Matrix{T}}

Compute boundary Poincare-Husimi matrices for many states.
Inputs:
- `ks::AbstractVector{T}`: Wavenumbers.
- `bps::AbstractVector{BoundaryPointsHyp{T}}`: Boundary containers.
- `us::AbstractVector{<:AbstractVector{Num}}`: Adjoint-kernel boundary functions `u=∂ₙᴱψ`.
- `qs::AbstractVector{<:AbstractVector{T}}`: q-grids.
- `ps::AbstractVector{<:AbstractVector{T}}`: p-grids.
Keywords:
- `full_p::Bool`: If true, use the supplied p-grid directly.
- `show_progress::Bool`: Show progress bar.
Output:
- `Hs::Vector{Matrix{T}}`: Husimi matrices for successful states.
Important:
- Uses `bp.ds`, not `bp.dsH`.
- This is the correct choice for boundary functions constructed from the adjoint
  hyperbolic DLP kernel.
"""
function husimi_on_grid_hyp(ks::AbstractVector{T},bps::AbstractVector{BoundaryPointsHyp{T}},us::AbstractVector{<:AbstractVector{Num}},qs::AbstractVector{<:AbstractVector{T}},ps::AbstractVector{<:AbstractVector{T}};full_p::Bool=false,show_progress::Bool=true) where {T<:Real,Num<:Number}
    n=length(ks)
    @assert length(bps)==n
    @assert length(us)==n
    @assert length(qs)==n
    @assert length(ps)==n
    Hs=Vector{Matrix{T}}(undef,n)
    ok=trues(n)
    pbar=show_progress ? Progress(n;desc="Husimi N=$n") : nothing
    Threads.@threads for i in 1:n
        try
            qgrid=qs[i]
            pgrid=ps[i]
            nq=length(qgrid)
            np=length(pgrid)
            H=full_p ? Matrix{T}(undef,nq,np) : Matrix{T}(undef,nq,2np-1)
            bp=bps[i]
            # The supplied u is from the adjoint-kernel path:
            #   u = ∂_{n_E}ψ.
            # Hence use bp.ds, while the packet coordinate remains bp.ξ.
            _husimi_on_grid_hyp!(H,ks[i],bp.ξ,us[i],bp.LH,bp.ds,qgrid,pgrid;full_p=full_p)
            Hs[i]=H
        catch
            ok[i]=false
        end
        show_progress && next!(pbar)
    end
    return Hs[ok]
end