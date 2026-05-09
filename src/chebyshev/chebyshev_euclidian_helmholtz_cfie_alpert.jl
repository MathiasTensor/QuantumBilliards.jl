_TWO_PI=2*pi
_INV_TWO_PI=1/_TWO_PI

"""
    build_CFIE_plans_alpert(ks,rmin,rmax;npanels=10000,M=5,nthreads=1)

Build Chebyshev interpolation plans for the special functions needed by
CFIE-Alpert assembly over a collection of wavenumbers.

For each `k`, this constructs plans for:
- `H₀^(1)(k r)`
- `H₁^(1)(k r)`

Unlike the Kress case, the Alpert formulation does not need separate `J₀`/`J₁`
plans in the self-correction formulas, because the logarithmic singularity is
handled by Alpert’s hybrid quadrature rather than analytic Kress splitting.

# Arguments
- `ks`:
  Wavenumbers for which the CFIE-Alpert operator will be assembled.
- `rmin, rmax`:
  Distance interval on which the Chebyshev interpolation must be valid.
- `npanels, M`:
  Parameters passed to the Chebyshev Hankel plan builder.
- `nthreads::Int=1`:
  Number of threads used when building the plans.

# Returns
- `(plans0,plans1)`:
  Chebyshev plans for `H₀^(1)` and `H₁^(1)` for all supplied wavenumbers.
"""
function build_CFIE_plans_alpert(ks::AbstractVector{<:Number},rmin::Float64,rmax::Float64;npanels::Int=10000,M::Int=5,nthreads::Int=1)
    Mk=length(ks)
    plans0=Vector{ChebHankelPlanH}(undef,Mk)
    plans1=Vector{ChebHankelPlanH}(undef,Mk)
    if nthreads<=1 || Mk==1
        @inbounds for m in 1:Mk
            k=ComplexF64(ks[m])
            plans0[m]=plan_h(0,1,k,rmin,rmax;npanels=npanels,M=M)
            plans1[m]=plan_h(1,1,k,rmin,rmax;npanels=npanels,M=M)
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
                k=ComplexF64(ks[m])
                plans0[m]=plan_h(0,1,k,rmin,rmax;npanels=npanels,M=M)
                plans1[m]=plan_h(1,1,k,rmin,rmax;npanels=npanels,M=M)
            end
        end
    end
    return plans0,plans1
end

"""
    CFIE_H0_H1_BesselWorkspace

Thread-local scratch workspace for Chebyshev-based CFIE-Alpert special-function
evaluation over multiple wavenumbers.

This workspace stores reusable temporary arrays for:
- `H₀^(1)(k r)`
- `H₁^(1)(k r)`

evaluated across a collection of wavenumbers at one geometric distance `r`
represented by a panel index and local Chebyshev coordinate.

In addition, it stores one thread-local complex work vector used for temporary
kernel coefficients during multi-`k` Alpert correction scattering.

# Fields
- `h0_tls`:
  Thread-local storage for interpolated `H₀^(1)` values.
- `h1_tls`:
  Thread-local storage for interpolated `H₁^(1)` values.
- `coeff_tls`:
  Thread-local temporary coefficient vectors used when scattering correction
  contributions to several interpolation nodes for all `k`.
"""
struct CFIE_H0_H1_BesselWorkspace
    h0_tls::Vector{Vector{ComplexF64}}
    h1_tls::Vector{Vector{ComplexF64}}
    coeff_tls::Vector{Vector{ComplexF64}}
end

function CFIE_H0_H1_BesselWorkspace(Mk::Int;ntls::Int=Threads.nthreads())
    h0_tls=[Vector{ComplexF64}(undef,Mk) for _ in 1:ntls]
    h1_tls=[Vector{ComplexF64}(undef,Mk) for _ in 1:ntls]
    coeff_tls=[Vector{ComplexF64}(undef,Mk) for _ in 1:ntls]
    return CFIE_H0_H1_BesselWorkspace(h0_tls,h1_tls,coeff_tls)
end

"""
    CFIE_alpert_BlockCache{T}

Geometry-and-interpolation cache for one ordered block of the CFIE-Alpert
matrix.

This cache represents a single ordered target/source block `(Γ_a,Γ_b)` of the
global CFIE-Alpert operator. It stores all quantities that depend only on the
boundary geometry and node placement, but not on the wavenumber `k`.

There are two qualitatively different cases:

1. `same == false`
   The block is an off-component interaction block. The kernel is smooth and
   the block is assembled directly from the standard DLP/SLP formulas.

2. `same == true`
   The block is a same-component self-interaction block. In this case the
   Alpert near-singular correction must be applied, and `selfcache` stores the
   corresponding `AlpertPeriodicCache` or `AlpertSmoothPanelCache`.

# Fields
- `same::Bool`:
  Whether this is a self-block (`a == b`).
- `row_offset::Int`:
  Global row offset of the target component in the assembled matrix.
- `col_offset::Int`:
  Global column offset of the source component in the assembled matrix.
- `Ni`, `Nj`:
  Number of target and source nodes in this block.
- `R::Matrix{T}`:
  Pairwise distance matrix for the block.
- `invR::Matrix{T}`:
  Pairwise inverse-distance matrix, with safe zero handling where needed.
- `inner::Matrix{T}`:
  Oriented DLP numerator matrix for this ordered target/source pairing.
- `speed_i`, `speed_j`:
  Target-side and source-side speed arrays.
- `wi`, `wj`:
  Target-side and source-side quadrature weights.
- `pidx::Matrix{Int32}`:
  Chebyshev panel index for each pair `(i,j)`.
- `tloc::Matrix{Float64}`:
  Local Chebyshev coordinate in `[-1,1]` for each pair `(i,j)`.
- `selfcache::Union{Nothing,AlpertCache{T}}`:
  Alpert self-correction cache for self-blocks; `nothing` for off-blocks.
"""
struct CFIE_alpert_BlockCache{T<:Real}
    same::Bool
    row_offset::Int
    col_offset::Int
    Ni::Int
    Nj::Int
    R::Matrix{T}
    invR::Matrix{T}
    inner::Matrix{T}
    speed_i::Vector{T}
    speed_j::Vector{T}
    wi::Vector{T}
    wj::Vector{T}
    pidx::Matrix{Int32}
    tloc::Matrix{Float64}
    selfcache::Union{Nothing,AlpertCache{T}}
end

"""
    CFIEAlpertBlockSystemCache{T}

Global block-cache system for Chebyshev-based CFIE-Alpert assembly.

This object gathers all ordered block caches for the full CFIE-Alpert matrix,
together with flat coordinate/tangent arrays and the global distance interval
required by the Chebyshev Hankel plans.

It is the Chebyshev-side companion to `CFIEAlpertWorkspace`:
- `CFIEAlpertWorkspace` stores the non-Chebyshev `k`-independent Alpert data,
- `CFIEAlpertBlockSystemCache` stores the additional blockwise geometry and
  interpolation metadata needed for Chebyshev special-function evaluation.

# Fields
- `blocks::Matrix{CFIE_alpert_BlockCache{T}}`:
  Ordered block caches for all component pairs.
- `offsets::Vector{Int}`:
  Global component offsets in the assembled matrix.
- `Xs, Ys`:
  Flat coordinate arrays for each component.
- `dXs, dYs`:
  Flat tangent-component arrays for each component.
- `ss`:
  Flat speed arrays for each component.
- `rmin, rmax`:
  Global minimum and maximum distances on which the Chebyshev plans must be
  valid, after safety padding.
"""
struct CFIEAlpertBlockSystemCache{T<:Real}
    blocks::Matrix{CFIE_alpert_BlockCache{T}}
    offsets::Vector{Int}
    Xs::Vector{Vector{T}}
    Ys::Vector{Vector{T}}
    dXs::Vector{Vector{T}}
    dYs::Vector{Vector{T}}
    ss::Vector{Vector{T}}
    rmin::Float64
    rmax::Float64
end
    
"""
    build_cfie_alpert_block_caches(solver,pts;npanels=10000,M=5,pad=(0.95,1.05))

Build the Chebyshev block-cache system for CFIE-Alpert assembly.

For each ordered block of the global matrix this precomputes:
- pairwise distances and inverse distances,
- DLP numerators,
- source/target speeds and quadrature weights,
- Chebyshev panel lookup indices and local coordinates,
- and, for self-blocks, the corresponding Alpert self-correction cache.

The resulting cache contains all `k`-independent blockwise information needed by
the Chebyshev-accelerated Alpert assembly.

# Arguments
- `solver::CFIE_alpert`
- `pts::Vector{BoundaryPointsCFIE{T}}`
- `rmin, rmax::T`: These global rmin and rmax must come from outside as they correct bounds due to off-grid Alpert nodes, and are used to build the Chebyshev plans. They can be estimated from the direct Alpert workspace using `estimate_cfie_alpert_cheb_rbounds`.
- `npanels, M`:
  Parameters passed to the Chebyshev Hankel plan builder, used here to determine
  the panel layout for the blockwise interpolation metadata.
- `pad`:
  Safety padding factors for the minimum and maximum distances used in the
  Chebyshev plans, applied to the blockwise extrema.

# Returns
- `CFIEAlpertBlockSystemCache{T}`
"""
function build_cfie_alpert_block_caches(solver::CFIE_alpert{T},pts::Vector{BoundaryPointsCFIE{T}},rmin::T,rmax::T;npanels::Int=10000,M::Int=5,pad=(T(0.95),T(1.05))) where {T<:Real}
    offs=component_offsets(pts)
    Gs=[cfie_geom_cache(p) for p in pts]
    rule=alpert_log_rule(T,solver.alpert_order)
    boundary=solver.billiard.full_boundary
    flat_boundary=boundary[1] isa AbstractVector ? reduce(vcat,boundary) : boundary
    nc=length(pts)
    Xs=Vector{Vector{T}}(undef,nc)
    Ys=Vector{Vector{T}}(undef,nc)
    dXs=Vector{Vector{T}}(undef,nc)
    dYs=Vector{Vector{T}}(undef,nc)
    ss=Vector{Vector{T}}(undef,nc)
    @inbounds for a in 1:nc
        X,Y,dX,dY,s=_panel_arrays(pts[a])
        Xs[a]=X;Ys[a]=Y;dXs[a]=dX;dYs[a]=dY;ss[a]=s
    end
    Cs=Vector{AlpertCache{T}}(undef,nc)
    @inbounds for a in 1:nc
        Cs[a]=_build_alpert_component_cache(solver,flat_boundary[a],pts[a],rule,solver.alpert_order)
    end
    blocks=Matrix{CFIE_alpert_BlockCache{T}}(undef,nc,nc)
    for a in 1:nc,b in 1:nc
        pa=pts[a];pb=pts[b]
        Ga=Gs[a];Gb=Gs[b]
        Ni=length(pa.xy);Nj=length(pb.xy)
        same=(a==b)
        if same
            R=Ga.R
            invR=Ga.invR
            inner=Ga.inner
            speed_i=Ga.speed
            speed_j=Ga.speed
            wi=pa.ws
            wj=pa.ws
            selfcache=Cs[a]
        else
            Xa=Xs[a];Ya=Ys[a]
            Xb=Xs[b];Yb=Ys[b]
            dXb=dXs[b];dYb=dYs[b]
            R=Matrix{T}(undef,Ni,Nj)
            invR=Matrix{T}(undef,Ni,Nj)
            inner=Matrix{T}(undef,Ni,Nj)
            @inbounds for j in 1:Nj,i in 1:Ni
                dx=Xa[i]-Xb[j]
                dy=Ya[i]-Yb[j]
                rij=hypot(dx,dy)
                R[i,j]=rij
                invR[i,j]=rij>eps(T) ? inv(rij) : zero(T)
                inner[i,j]=dYb[j]*dx-dXb[j]*dy
            end
            speed_i=Ga.speed
            speed_j=Gb.speed
            wi=pa.ws
            wj=pb.ws
            selfcache=nothing
        end
        pidx=Matrix{Int32}(undef,Ni,Nj)
        tloc=Matrix{Float64}(undef,Ni,Nj)
        blocks[a,b]=CFIE_alpert_BlockCache{T}(same,offs[a],offs[b],Ni,Nj,R,invR,inner,speed_i,speed_j,wi,wj,pidx,tloc,selfcache)
    end
    pref_plan=plan_h(0,1,1.0+0im,Float64(rmin),Float64(rmax);npanels=npanels,M=M)
    pans=pref_plan.panels
    for a in 1:nc,b in 1:nc
        blk=blocks[a,b]
        @inbounds for j in 1:blk.Nj,i in 1:blk.Ni
            if blk.same && i==j
                blk.pidx[i,j]=Int32(0)
                blk.tloc[i,j]=0.0
            else
                rij=Float64(blk.R[i,j])
                if rij<rmin
                    blk.pidx[i,j]=Int32(0)
                    blk.tloc[i,j]=0.0
                else
                    p=_find_panel(pref_plan,rij)
                    P=pans[p]
                    blk.pidx[i,j]=Int32(p)
                    blk.tloc[i,j]=(2*rij-(P.b+P.a))/(P.b-P.a)
                end
            end
        end
    end
    return CFIEAlpertBlockSystemCache{T}(blocks,offs,Xs,Ys,dXs,dYs,ss,rmin,rmax)
end

"""
    CFIEAlpertChebWorkspace{T}

Reusable workspace for Chebyshev-accelerated CFIE-Alpert assembly on a fixed
boundary discretization and a fixed set of wavenumbers.

This object combines:
- the original direct Alpert workspace,
- the Chebyshev block cache,
- the Hankel interpolation plans,
- the thread-local temporary special-function buffers.

# Fields
- `direct::CFIEAlpertWorkspace{T}`:
  Non-Chebyshev Alpert workspace containing geometry caches, Alpert self-caches,
  offsets, and flat panel arrays.
- `block_cache::CFIEAlpertBlockSystemCache{T}`:
  Chebyshev block-system cache for blockwise geometry and interpolation lookup.
- `plans0`, `plans1`:
  Chebyshev Hankel plans for `H₀^(1)` and `H₁^(1)`.
- `bessel_ws::CFIE_H0_H1_BesselWorkspace`:
  Thread-local temporary storage for interpolated Hankel values and
  coefficient vectors.
- `ks::Vector{ComplexF64}`:
  Wavenumbers associated with the plans in this workspace.
- `Mk::Int`:
  Number of wavenumbers.
"""
struct CFIEAlpertChebWorkspace{T<:Real}
    direct::CFIEAlpertWorkspace{T}
    block_cache::CFIEAlpertBlockSystemCache{T}
    plans0::Vector{ChebHankelPlanH}
    plans1::Vector{ChebHankelPlanH}
    bessel_ws::CFIE_H0_H1_BesselWorkspace
    ks::Vector{ComplexF64}
    Mk::Int
end

"""
    build_cfie_alpert_cheb_workspace(solver,pts,direct,ks;npanels=10000,M=5,pad=(0.95,1.05),plan_nthreads=1,ntls=Threads.nthreads())

Build the full reusable Chebyshev workspace for CFIE-Alpert assembly.

This function combines:
- a prebuilt direct Alpert workspace `direct`,
- a Chebyshev block cache,
- Chebyshev Hankel plans over the required distance interval,
- thread-local temporary buffers.

# Arguments
- `solver::CFIE_alpert`
- `pts::Vector{BoundaryPointsCFIE{T}}`
- `direct::CFIEAlpertWorkspace{T}`:
  Prebuilt direct Alpert workspace.
- `ks::Vector{ComplexF64}`:
  Wavenumbers for which the workspace will be used.
- `npanels, M`:
  Parameters passed to the Chebyshev Hankel plan builder.
- `pad`:
  Safety padding factors for the minimum and maximum distances used in the Chebyshev plans, applied to the blockwise extrema.
- `plan_nthreads`:
  Number of threads to use when building the Chebyshev Hankel plans.
- `ntls`:
  Number of thread-local buffers to allocate in the Bessel workspace; typically set to `Threads.nthreads()`.

# Returns
- `CFIEAlpertChebWorkspace{T}`
"""
function build_cfie_alpert_cheb_workspace(solver::CFIE_alpert{T},pts::Vector{BoundaryPointsCFIE{T}},direct::CFIEAlpertWorkspace{T},ks::Vector{ComplexF64};npanels::Int=10000,M::Int=5,pad=(T(0.95),T(1.05)),plan_nthreads::Int=1,ntls::Int=Threads.nthreads()) where {T<:Real}
    rmin_raw,rmax=estimate_cfie_alpert_cheb_rbounds(direct;pad=pad)
    rmin_cheb=minimum(hankel_z_chebyshev_cutoff./abs.(ks))
    rmin=max(rmin_raw,rmin_cheb)
    block_cache=build_cfie_alpert_block_caches(solver,pts,rmin,rmax;npanels=npanels,M=M,pad=pad)
    plans0,plans1=build_CFIE_plans_alpert(ks,rmin,rmax;npanels=npanels,M=M,nthreads=plan_nthreads)
    bessel_ws=CFIE_H0_H1_BesselWorkspace(length(ks);ntls=ntls)
    return CFIEAlpertChebWorkspace{T}(direct,block_cache,plans0,plans1,bessel_ws,ks,length(ks))
end

############################################
#### CHEB LOOKUP HELPERS FOR ONE ENTRY #####
############################################

@inline function _h0_h1_at_pidx_t!(h0vals::AbstractVector{ComplexF64},h1vals::AbstractVector{ComplexF64},pidx::Int32,t::Float64,r::Float64,plans0::AbstractVector{ChebHankelPlanH},plans1::AbstractVector{ChebHankelPlanH})
    h0_h1_multi_ks_at_r!(h0vals,h1vals,plans0,plans1,pidx,t,r)
    return nothing
end

"""
    _h0_h1_at_entry!(h0vals,h1vals,blk,i,j,plans0,plans1)

Evaluate `H₀^(1)` and `H₁^(1)` for all wavenumbers at the block entry `(i,j)`
using the precomputed Chebyshev panel index and local coordinate stored in
`blk`.

# Returns
- `nothing`
"""
@inline function _h0_h1_at_entry!(h0vals::AbstractVector{ComplexF64},h1vals::AbstractVector{ComplexF64},blk::CFIE_alpert_BlockCache,i::Int,j::Int,plans0::AbstractVector{ChebHankelPlanH},plans1::AbstractVector{ChebHankelPlanH})
    pidx=blk.pidx[i,j]
    r=Float64(blk.R[i,j])
    if pidx!=0
        _h0_h1_at_pidx_t!(h0vals,h1vals,pidx,blk.tloc[i,j],r,plans0,plans1)
    else
        @inbounds for m in eachindex(plans0)
            z=ComplexF64(plans0[m].k)*r
            az=abs(z)
            if az<hankel_z_chebyshev_cutoff_small_z
                h0vals[m]=_small_h0_series(z)
                h1vals[m]=_small_h1_series(z)
            else
                h0vals[m]=SpecialFunctions.besselh(0,1,z)
                h1vals[m]=SpecialFunctions.besselh(1,1,z)
            end
        end
    end
    return nothing
end

"""
    _h0_h1_at_r!(h0vals,h1vals,r,plans0,plans1)

Evaluate `H₀^(1)` and `H₁^(1)` for all wavenumbers at a raw distance `r`,
locating the corresponding Chebyshev panel on the fly from `plans0[1]`.

# Returns
- `nothing`
"""
@inline function _h0_h1_at_r!(h0vals::AbstractVector{ComplexF64},h1vals::AbstractVector{ComplexF64},r::Float64,plans0::AbstractVector{ChebHankelPlanH},plans1::AbstractVector{ChebHankelPlanH})
    if r<plans0[1].rmin
        @inbounds for m in eachindex(plans0)
            z=ComplexF64(plans0[m].k)*r
            az=abs(z)
            if az<hankel_z_chebyshev_cutoff_small_z
                h0vals[m]=_small_h0_series(z)
                h1vals[m]=_small_h1_series(z)
            else
                h0vals[m]=SpecialFunctions.besselh(0,1,z)
                h1vals[m]=SpecialFunctions.besselh(1,1,z)
            end
        end
    else
        pidx=_find_panel(plans0[1],r)
        P=plans0[1].panels[pidx]
        t=(2*r-(P.b+P.a))/(P.b-P.a)
        _h0_h1_at_pidx_t!(h0vals,h1vals,Int32(pidx),t,r,plans0,plans1)
    end
    return nothing
end

# these are used for the Alpert correction nodes, which are off-grid and thus require raw `r`-based evaluation rather than direct Chebyshev lookup
# the distances `r` are typically small, so we use the small-`z` series expansions.

"""
    _dlp_terms_h01(TT,k,r,inn,invr,w,h0,h1)

Return the DLP contribution and its first two `k`-derivatives using already
available Hankel values `h0 = H₀^(1)(k r)` and `h1 = H₁^(1)(k r)`.

# Returns
`(d0,d1,d2)`.
"""
@inline function _dlp_terms_h01(TT,k,r,inn,invr,w,h0,h1)
    αD=Complex{TT}(0,k/2)
    d0=w*(αD*inn*h1*invr)
    d1=w*((Complex{TT}(0,1)/2)*inn*k*h0)
    d2=w*((Complex{TT}(0,1)/2)*inn*(h0-k*r*h1))
    return d0,d1,d2
end

"""
    _slp_terms_h01(TT,k,r,s,w,h0,h1)

Return the SLP contribution and its first two `k`-derivatives using already
available Hankel values `h0 = H₀^(1)(k r)` and `h1 = H₁^(1)(k r)`.

# Returns
`(s0,s1,s2)`.
"""
@inline function _slp_terms_h01(TT,k,r,s,w,h0,h1)
    αS=Complex{TT}(0,one(TT)/2)
    s0=w*(αS*h0*s)
    s1=w*(-(Complex{TT}(0,1)/2)*r*h1*s)
    s2=w*((Complex{TT}(0,1)/2)*r*(h1-k*r*h0)*s/k)
    return s0,s1,s2
end

# Return `true` if nodes `i` and `j` are within the excluded Alpert near
# stencil on a periodic component.

# The distance is measured cyclically,

#    dper = min(abs(i-j), N - abs(i-j)),
#
# and the pair is treated as near-singular if `dper < a`.
#
# This predicate is used in the optimized periodic self-block assembly to skip
# the naive trapezoidal contribution on the Alpert replacement stencil. The
# missing near entries are then supplied only through the Alpert correction nodes
# and interpolation weights.
@inline _periodic_near(i::Int,j::Int,N::Int,a::Int)=begin
    d=abs(i-j)
    min(d,N-d)<a
end

# Assemble the value-only periodic self-interaction block for CFIE-Alpert using
# Chebyshev-interpolated Hankel values.
# 1. inserts the identity contribution on the diagonal,
# 2. adds the naive trapezoidal CFIE contribution only for pairs outside the
#    periodic Alpert near stencil,
# 3. adds the positive and negative Alpert correction-node contributions using
#    the interpolation weights in `C`.
function _assemble_self_alpert_periodic_cheb!(As::Vector{Matrix{ComplexF64}},pts::BoundaryPointsCFIE{T},G::CFIEGeomCache{T},C::AlpertPeriodicCache{T},blk::CFIE_alpert_BlockCache{T},row_range::UnitRange{Int},ks::Vector{ComplexF64},rule::AlpertLogRule{T},plans0::Vector{ChebHankelPlanH},plans1::Vector{ChebHankelPlanH},h0_tls::Vector{Vector{ComplexF64}},h1_tls::Vector{Vector{ComplexF64}};multithreaded::Bool=true) where {T<:Real}
    Mk=length(ks)
    αD=Vector{ComplexF64}(undef,Mk)
    iks=Vector{ComplexF64}(undef,Mk)
    αS=ComplexF64(0,0.5)
    @inbounds for m in 1:Mk
        k=ks[m]
        αD[m]=0.5im*k
        iks[m]=1im*k
    end
    R=G.R;invR=G.invR;inner=G.inner;speed=G.speed
    rp=C.rp;rm=C.rm;innp=C.innp;innm=C.innm;sp=C.sp;sm=C.sm
    offsp=C.offsp;wtp=C.wtp;offsm=C.offsm;wtm=C.wtm
    N=length(pts.xy);h=pts.ws[1];a=rule.a;jcorr=rule.j;ninterp=C.ninterp
    r0=first(row_range)-1
    @use_threads multithreading=multithreaded for i in 1:N
        tid=Threads.threadid()
        h0vals=h0_tls[tid]
        h1vals=h1_tls[tid]
        gi=r0+i
        @inbounds for m in 1:Mk
            As[m][gi,gi]+=1.0+0im
        end
        @inbounds for j in 1:N
            (j==i || _periodic_near(i,j,N,a)) && continue
            gj=r0+j
            _h0_h1_at_entry!(h0vals,h1vals,blk,i,j,plans0,plans1)
            cD=h*inner[i,j]*invR[i,j]
            cS=h*speed[j]
            for m in 1:Mk
                As[m][gi,gj]-=cD*αD[m]*h1vals[m]+iks[m]*(cS*αS*h0vals[m])
            end
        end
        @inbounds for p in 1:jcorr
            fac=h*rule.w[p]
            r=rp[p,i]
            if isfinite(r)
                _h0_h1_at_r!(h0vals,h1vals,Float64(r),plans0,plans1)
                cD=fac*innp[p,i]/r
                cS=fac*sp[p,i]
                for m in 1:Mk
                    coeff=-(cD*αD[m]*h1vals[m]+iks[m]*(cS*αS*h0vals[m]))
                    for q in 1:ninterp
                        As[m][gi,r0+mod1(i+offsp[p,q],N)]+=coeff*wtp[p,q]
                    end
                end
            end
            r=rm[p,i]
            if isfinite(r)
                _h0_h1_at_r!(h0vals,h1vals,Float64(r),plans0,plans1)
                cD=fac*innm[p,i]/r
                cS=fac*sm[p,i]
                for m in 1:Mk
                    coeff=-(cD*αD[m]*h1vals[m]+iks[m]*(cS*αS*h0vals[m]))
                    for q in 1:ninterp
                        As[m][gi,r0+mod1(i+offsm[p,q],N)]+=coeff*wtm[p,q]
                    end
                end
            end
        end
    end
    return nothing
end

# Assemble the periodic CFIE-Alpert self-interaction block and its first two
# wavenumber derivatives.
#
# For every `k = ks[m]`, this fills As[m]  = A(k), A1s[m] = dA/dk, A2s[m] = d²A/dk²,
# 
# for the local periodic component block. The operator convention is
#     A(k) = I - (D(k) + i*k*S(k)),
# so the derivatives are
#     A′(k)  = -(D′(k) + i*S(k) + i*k*S′(k)),
#     A′′(k) = -(D′′(k) + 2i*S′(k) + i*k*S′′(k)).
function _assemble_self_alpert_periodic_cheb_deriv!(As::Vector{Matrix{ComplexF64}},A1s::Vector{Matrix{ComplexF64}},A2s::Vector{Matrix{ComplexF64}},pts::BoundaryPointsCFIE{T},G::CFIEGeomCache{T},C::AlpertPeriodicCache{T},blk::CFIE_alpert_BlockCache{T},row_range::UnitRange{Int},ks::Vector{ComplexF64},rule::AlpertLogRule{T},plans0::Vector{ChebHankelPlanH},plans1::Vector{ChebHankelPlanH},h0_tls::Vector{Vector{ComplexF64}},h1_tls::Vector{Vector{ComplexF64}};multithreaded::Bool=true) where {T<:Real}
    Mk=length(ks)
    iks=Vector{ComplexF64}(undef,Mk)
    @inbounds for m in 1:Mk
        iks[m]=1im*ks[m]
    end
    im1=ComplexF64(0,1)
    im2=ComplexF64(0,2)
    R=G.R;invR=G.invR;inner=G.inner;speed=G.speed
    rp=C.rp;rm=C.rm;innp=C.innp;innm=C.innm;sp=C.sp;sm=C.sm
    offsp=C.offsp;wtp=C.wtp;offsm=C.offsm;wtm=C.wtm
    N=length(pts.xy);h=pts.ws[1];a=rule.a;jcorr=rule.j;ninterp=C.ninterp
    r0=first(row_range)-1
    @use_threads multithreading=multithreaded for i in 1:N
        tid=Threads.threadid()
        h0vals=h0_tls[tid]
        h1vals=h1_tls[tid]
        gi=r0+i
        @inbounds for m in 1:Mk
            As[m][gi,gi]+=1.0+0im
        end
        @inbounds for j in 1:N
            (j==i || _periodic_near(i,j,N,a)) && continue
            gj=r0+j
            _h0_h1_at_entry!(h0vals,h1vals,blk,i,j,plans0,plans1)
            r=R[i,j];invr=invR[i,j];inn=inner[i,j];sj=speed[j]
            for m in 1:Mk
                d0,d1,d2=_dlp_terms_h01(T,ks[m],r,inn,invr,h,h0vals[m],h1vals[m])
                s0,s1,s2=_slp_terms_h01(T,ks[m],r,sj,h,h0vals[m],h1vals[m])
                As[m][gi,gj]-=d0+iks[m]*s0
                A1s[m][gi,gj]-=d1+im1*s0+iks[m]*s1
                A2s[m][gi,gj]-=d2+im2*s1+iks[m]*s2
            end
        end
        @inbounds for p in 1:jcorr
            fac=h*rule.w[p]
            r=rp[p,i]
            if isfinite(r)
                _h0_h1_at_r!(h0vals,h1vals,Float64(r),plans0,plans1)
                invr=inv(r)
                inn=innp[p,i]
                spp=sp[p,i]
                for m in 1:Mk
                    d0,d1,d2=_dlp_terms_h01(T,ks[m],r,inn,invr,fac,h0vals[m],h1vals[m])
                    s0,s1,s2=_slp_terms_h01(T,ks[m],r,spp,fac,h0vals[m],h1vals[m])
                    a0=-(d0+iks[m]*s0)
                    a1=-(d1+im1*s0+iks[m]*s1)
                    a2=-(d2+im2*s1+iks[m]*s2)
                    for q in 1:ninterp
                        gq=r0+mod1(i+offsp[p,q],N)
                        ww=wtp[p,q]
                        As[m][gi,gq]+=a0*ww
                        A1s[m][gi,gq]+=a1*ww
                        A2s[m][gi,gq]+=a2*ww
                    end
                end
            end
            r=rm[p,i]
            if isfinite(r)
                _h0_h1_at_r!(h0vals,h1vals,Float64(r),plans0,plans1)
                invr=inv(r)
                inn=innm[p,i]
                smm=sm[p,i]
                for m in 1:Mk
                    d0,d1,d2=_dlp_terms_h01(T,ks[m],r,inn,invr,fac,h0vals[m],h1vals[m])
                    s0,s1,s2=_slp_terms_h01(T,ks[m],r,smm,fac,h0vals[m],h1vals[m])
                    a0=-(d0+iks[m]*s0)
                    a1=-(d1+im1*s0+iks[m]*s1)
                    a2=-(d2+im2*s1+iks[m]*s2)
                    for q in 1:ninterp
                        gq=r0+mod1(i+offsm[p,q],N)
                        ww=wtm[p,q]
                        As[m][gi,gq]+=a0*ww
                        A1s[m][gi,gq]+=a1*ww
                        A2s[m][gi,gq]+=a2*ww
                    end
                end
            end
        end
    end
    return nothing
end

# For each target node inserts the identity contribution, adds the
# naive CFIE trapezoidal interaction only outside the local Alpert exclusion
# stencil `abs(i-j) < rule.a`, and then scatters the Alpert correction-node
# contributions using the interpolation indices and weights stored in `C`.
function _assemble_self_alpert_smooth_panel_cheb!(solver::CFIE_alpert{T},As::Vector{Matrix{ComplexF64}},pts::BoundaryPointsCFIE{T},G::CFIEGeomCache{T},C::AlpertSmoothPanelCache{T},X::Vector{T},Y::Vector{T},row_range::UnitRange{Int},ks::Vector{ComplexF64},rule::AlpertLogRule{T},plans0::Vector{ChebHankelPlanH},plans1::Vector{ChebHankelPlanH},h0_tls::Vector{Vector{ComplexF64}},h1_tls::Vector{Vector{ComplexF64}},coeff_tls::Vector{Vector{ComplexF64}};multithreaded::Bool=true) where {T<:Real}
    Mk=length(ks)
    αD=Vector{ComplexF64}(undef,Mk)
    iks=Vector{ComplexF64}(undef,Mk)
    αS=ComplexF64(0,0.5)
    @inbounds for m in 1:Mk
        k=ks[m]
        αD[m]=0.5im*k
        iks[m]=1im*k
    end
    R=G.R;invR=G.invR;inner=G.inner;speed=G.speed
    rp=C.rp;rm=C.rm;innp=C.innp;innm=C.innm;sp=C.sp;sm=C.sm
    idxp=C.idxp;wtp=C.wtp;idxm=C.idxm;wtm=C.wtm
    w=pts.ws
    N=length(X)
    hσ=w[1]
    a=rule.a
    jcorr=rule.j
    pinterp=size(idxp,3)
    r0=first(row_range)-1
    @use_threads multithreading=multithreaded for i in 1:N
        tid=Threads.threadid()
        h0vals=h0_tls[tid]
        h1vals=h1_tls[tid]
        gi=r0+i
        @inbounds for m in 1:Mk
            As[m][gi,gi]+=1.0+0im
        end
        @inbounds for j in 1:N
            (j==i || abs(j-i)<a) && continue
            gj=r0+j
            r=R[i,j]
            _h0_h1_at_r!(h0vals,h1vals,Float64(r),plans0,plans1)
            cD=w[j]*inner[i,j]*invR[i,j]
            cS=w[j]*speed[j]
            for m in 1:Mk
                As[m][gi,gj]-=cD*αD[m]*h1vals[m]+iks[m]*(cS*αS*h0vals[m])
            end
        end
        @inbounds for p in 1:jcorr
            fac=hσ*rule.w[p]
            r=rp[p,i]
            if isfinite(r)
                _h0_h1_at_r!(h0vals,h1vals,Float64(r),plans0,plans1)
                cD=fac*innp[p,i]*inv(r)
                cS=fac*sp[p,i]
                for m in 1:Mk
                    coeff=-(cD*αD[m]*h1vals[m]+iks[m]*(cS*αS*h0vals[m]))
                    for q in 1:pinterp
                        As[m][gi,row_range[idxp[p,i,q]]]+=coeff*wtp[p,i,q]
                    end
                end
            end
            r=rm[p,i]
            if isfinite(r)
                _h0_h1_at_r!(h0vals,h1vals,Float64(r),plans0,plans1)
                cD=fac*innm[p,i]*inv(r)
                cS=fac*sm[p,i]
                for m in 1:Mk
                    coeff=-(cD*αD[m]*h1vals[m]+iks[m]*(cS*αS*h0vals[m]))
                    for q in 1:pinterp
                        As[m][gi,row_range[idxm[p,i,q]]]+=coeff*wtm[p,i,q]
                    end
                end
            end
        end
    end
    return nothing
end

# For each `k = ks[m]`, fills As[m]  = A(k), A1s[m] = dA/dk, A2s[m] = d²A/dk²,
#     A(k) = I - (D(k) + i*k*S(k)).
# The naive trapezoidal loop excludes the local Alpert stencil
# `abs(i-j) < rule.a`; the excluded contribution is replaced by the Alpert
# correction-node quadrature stored in `C`.
#
# The derivative formulas use the already-interpolated values
#   `H₀^(1)(k*r)` and `H₁^(1)(k*r)` and apply
#    A′(k)  = -(D′(k) + i*S(k) + i*k*S′(k)),
#    A′′(k) = -(D′′(k) + 2i*S′(k) + i*k*S′′(k)).
function _assemble_self_alpert_smooth_panel_cheb_deriv!(solver::CFIE_alpert{T},As::Vector{Matrix{ComplexF64}},A1s::Vector{Matrix{ComplexF64}},A2s::Vector{Matrix{ComplexF64}},pts::BoundaryPointsCFIE{T},G::CFIEGeomCache{T},C::AlpertSmoothPanelCache{T},P::CFIEPanelArrays{T},row_range::UnitRange{Int},ks::Vector{ComplexF64},rule::AlpertLogRule{T},plans0::Vector{ChebHankelPlanH},plans1::Vector{ChebHankelPlanH},h0_tls::Vector{Vector{ComplexF64}},h1_tls::Vector{Vector{ComplexF64}};multithreaded::Bool=true) where {T<:Real}
    Mk=length(ks)
    iks=Vector{ComplexF64}(undef,Mk)
    @inbounds for m in 1:Mk
        iks[m]=1im*ks[m]
    end
    im1=ComplexF64(0,1)
    im2=ComplexF64(0,2)
    halfim=ComplexF64(0,0.5)
    mhalfim=ComplexF64(0,-0.5)
    X=P.X
    w=pts.ws
    R=G.R;invR=G.invR;inner=G.inner;speed=G.speed
    rp=C.rp;rm=C.rm;innp=C.innp;innm=C.innm;sp=C.sp;sm=C.sm
    idxp=C.idxp;wtp=C.wtp;idxm=C.idxm;wtm=C.wtm
    N=length(X)
    hσ=w[1]
    a=rule.a
    jcorr=rule.j
    pinterp=size(idxp,3)
    @use_threads multithreading=multithreaded for i in 1:N
        tid=Threads.threadid()
        h0vals=h0_tls[tid]
        h1vals=h1_tls[tid]
        gi=row_range[i]
        @inbounds for m in 1:Mk
            As[m][gi,gi]+=1.0+0im
        end
        @inbounds for j in 1:N
            (j==i || abs(j-i)<a) && continue
            gj=row_range[j]
            r=R[i,j]
            _h0_h1_at_r!(h0vals,h1vals,Float64(r),plans0,plans1)
            wd=w[j]
            cD=wd*inner[i,j]
            cDoverR=cD*invR[i,j]
            cS=wd*speed[j]
            for m in 1:Mk
                k=ks[m]
                h0=h0vals[m]
                h1=h1vals[m]
                ik=iks[m]
                d0=halfim*k*cDoverR*h1
                d1=halfim*cD*k*h0
                d2=halfim*cD*(h0-k*r*h1)
                s0=halfim*cS*h0
                s1=mhalfim*cS*r*h1
                s2=halfim*cS*r*(h1-k*r*h0)/k
                As[m][gi,gj]-=d0+ik*s0
                A1s[m][gi,gj]-=d1+im1*s0+ik*s1
                A2s[m][gi,gj]-=d2+im2*s1+ik*s2
            end
        end
        @inbounds for p in 1:jcorr
            fac=hσ*rule.w[p]
            r=rp[p,i]
            if isfinite(r)
                _h0_h1_at_r!(h0vals,h1vals,Float64(r),plans0,plans1)
                cD=fac*innp[p,i]
                cDoverR=cD*inv(r)
                cS=fac*sp[p,i]
                for m in 1:Mk
                    k=ks[m]
                    h0=h0vals[m]
                    h1=h1vals[m]
                    ik=iks[m]
                    d0=halfim*k*cDoverR*h1
                    d1=halfim*cD*k*h0
                    d2=halfim*cD*(h0-k*r*h1)
                    s0=halfim*cS*h0
                    s1=mhalfim*cS*r*h1
                    s2=halfim*cS*r*(h1-k*r*h0)/k
                    a0=-(d0+ik*s0)
                    a1=-(d1+im1*s0+ik*s1)
                    a2=-(d2+im2*s1+ik*s2)
                    for q in 1:pinterp
                        gq=row_range[idxp[p,i,q]]
                        ww=wtp[p,i,q]
                        As[m][gi,gq]+=a0*ww
                        A1s[m][gi,gq]+=a1*ww
                        A2s[m][gi,gq]+=a2*ww
                    end
                end
            end
            r=rm[p,i]
            if isfinite(r)
                _h0_h1_at_r!(h0vals,h1vals,Float64(r),plans0,plans1)
                cD=fac*innm[p,i]
                cDoverR=cD*inv(r)
                cS=fac*sm[p,i]
                for m in 1:Mk
                    k=ks[m]
                    h0=h0vals[m]
                    h1=h1vals[m]
                    ik=iks[m]
                    d0=halfim*k*cDoverR*h1
                    d1=halfim*cD*k*h0
                    d2=halfim*cD*(h0-k*r*h1)
                    s0=halfim*cS*h0
                    s1=mhalfim*cS*r*h1
                    s2=halfim*cS*r*(h1-k*r*h0)/k
                    a0=-(d0+ik*s0)
                    a1=-(d1+im1*s0+ik*s1)
                    a2=-(d2+im2*s1+ik*s2)
                    for q in 1:pinterp
                        gq=row_range[idxm[p,i,q]]
                        ww=wtm[p,i,q]
                        As[m][gi,gq]+=a0*ww
                        A1s[m][gi,gq]+=a1*ww
                        A2s[m][gi,gq]+=a2*ww
                    end
                end
            end
        end
    end
    return nothing
end

# It uses `CFIEAlpertBlockSystemCache` to reuse all geometry-dependent quantities:
# distances, inverse distances, DLP numerators, source weights, source speeds,
# global row/column offsets, and Chebyshev panel lookup data.
# For each off-diagonal block `(a,b)`, `a != b`, it writes
#     A_ij(k) -= D_ij(k) + i*k*S_ij(k),
# where
#     D_ij(k) = w_j * (i*k/2) * inner_ij * H₁^(1)(k*r_ij) / r_ij,
#     S_ij(k) = w_j * (i/2)   * speed_j * H₀^(1)(k*r_ij).
function _assemble_all_offpanel_blockcache!(As::Vector{Matrix{ComplexF64}},block_cache::CFIEAlpertBlockSystemCache{T},ks::Vector{ComplexF64},plans0::Vector{ChebHankelPlanH},plans1::Vector{ChebHankelPlanH},h0_tls::Vector{Vector{ComplexF64}},h1_tls::Vector{Vector{ComplexF64}};multithreaded::Bool=true) where {T<:Real}
    Mk=length(ks)
    αD=Vector{ComplexF64}(undef,Mk)
    iks=Vector{ComplexF64}(undef,Mk)
    αS=ComplexF64(0,0.5)
    @inbounds for m in 1:Mk
        k=ks[m]
        αD[m]=0.5im*k
        iks[m]=1im*k
    end
    blocks=block_cache.blocks
    nc=size(blocks,1)
    @inbounds for a in 1:nc, b in 1:nc
        a==b && continue
        blk=blocks[a,b]
        R=blk.R
        invR=blk.invR
        inner=blk.inner
        wj=blk.wj
        sj=blk.speed_j
        ro=blk.row_offset
        co=blk.col_offset
        @use_threads multithreading=multithreaded for i in 1:blk.Ni
            tid=Threads.threadid()
            h0vals=h0_tls[tid]
            h1vals=h1_tls[tid]
            gi=ro+i-1
            @inbounds for j in 1:blk.Nj
                r=R[i,j]
                r<=eps(T) && continue
                gj=co+j-1
                _h0_h1_at_entry!(h0vals,h1vals,blk,i,j,plans0,plans1)
                cD=wj[j]*inner[i,j]*invR[i,j]
                cS=wj[j]*sj[j]
                for m in 1:Mk
                    As[m][gi,gj]-=cD*αD[m]*h1vals[m]+iks[m]*(cS*αS*h0vals[m])
                end
            end
        end
    end

    return nothing
end

# For every `k = ks[m]`, fills the off-block contributions to
#    As[m]  = A(k),
#    A1s[m] = dA/dk,
#    A2s[m] = d²A/dk²,
# with the operator convention
#    A(k) = I - (D(k) + i*k*S(k)).
# The derivative formulas are
#    A′(k)  = -(D′(k) + i*S(k) + i*k*S′(k)),
#    A′′(k) = -(D′′(k) + 2i*S′(k) + i*k*S′′(k)).
function _assemble_all_offpanel_blockcache_deriv!(As::Vector{Matrix{ComplexF64}},A1s::Vector{Matrix{ComplexF64}},A2s::Vector{Matrix{ComplexF64}},block_cache::CFIEAlpertBlockSystemCache{T},ks::Vector{ComplexF64},plans0::Vector{ChebHankelPlanH},plans1::Vector{ChebHankelPlanH},h0_tls::Vector{Vector{ComplexF64}},h1_tls::Vector{Vector{ComplexF64}};multithreaded::Bool=true) where {T<:Real}
    Mk=length(ks)
    iks=Vector{ComplexF64}(undef,Mk)
    @inbounds for m in 1:Mk
        iks[m]=1im*ks[m]
    end
    im1=ComplexF64(0,1)
    im2=ComplexF64(0,2)
    halfim=ComplexF64(0,0.5)
    mhalfim=ComplexF64(0,-0.5)
    blocks=block_cache.blocks
    nc=size(blocks,1)
    @inbounds for a in 1:nc, b in 1:nc
        a==b && continue
        blk=blocks[a,b]
        R=blk.R
        invR=blk.invR
        inner=blk.inner
        wj=blk.wj
        sj=blk.speed_j
        ro=blk.row_offset
        co=blk.col_offset
        @use_threads multithreading=multithreaded for i in 1:blk.Ni
            tid=Threads.threadid()
            h0vals=h0_tls[tid]
            h1vals=h1_tls[tid]
            gi=ro+i-1
            @inbounds for j in 1:blk.Nj
                r=R[i,j]
                r<=eps(T) && continue
                gj=co+j-1
                invr=invR[i,j]
                _h0_h1_at_entry!(h0vals,h1vals,blk,i,j,plans0,plans1)
                cD=wj[j]*inner[i,j]
                cDoverR=cD*invr
                cS=wj[j]*sj[j]
                for m in 1:Mk
                    k=ks[m]
                    h0=h0vals[m]
                    h1=h1vals[m]
                    ik=iks[m]
                    d0=halfim*k*cDoverR*h1
                    d1=halfim*cD*k*h0
                    d2=halfim*cD*(h0-k*r*h1)
                    s0=halfim*cS*h0
                    s1=mhalfim*cS*r*h1
                    s2=halfim*cS*r*(h1-k*r*h0)/k
                    As[m][gi,gj]-=d0+ik*s0
                    A1s[m][gi,gj]-=d1+im1*s0+ik*s1
                    A2s[m][gi,gj]-=d2+im2*s1+ik*s2
                end
            end
        end
    end
    return nothing
end

"""
    compute_kernel_matrices_CFIE_alpert_chebyshev!(As,solver,pts,ws;multithreaded=true)

Assemble the CFIE-Alpert matrices for all wavenumbers stored in a prebuilt
Chebyshev workspace, writing the results in place.

This is the main value-only entry point for Chebyshev-accelerated CFIE-Alpert
assembly. It fills one dense global matrix per wavenumber, using:
- Alpert-corrected self-block assembly on each component,
- Chebyshev-interpolated Hankel evaluation,
- naive smooth off-component assembly for all off-diagonal blocks.

The assembled operator is the same CFIE boundary operator as in the direct
Alpert route, but with the special-function evaluations accelerated through the
Chebyshev plans stored in `ws`.

# Arguments
- `As::Vector{Matrix{ComplexF64}}`: Output matrices, one for each wavenumber in `ws.ks`. Each matrix is overwritten in place.
- `solver::CFIE_alpert{T}`: CFIE-Alpert solver whose geometry and Alpert order define the discretization.
- `pts::Vector{BoundaryPointsCFIE{T}}`: Boundary discretization for all components/panels.
- `ws::CFIEAlpertChebWorkspace{T}`: Prebuilt Chebyshev workspace containing:
- `multithreaded::Bool=true`:
  Enables threaded assembly where the low-level kernels support it.

# Returns
- `nothing`
"""
function compute_kernel_matrices_CFIE_alpert_chebyshev!(As::Vector{Matrix{ComplexF64}},solver::CFIE_alpert{T},pts::Vector{BoundaryPointsCFIE{T}},ws::CFIEAlpertChebWorkspace{T};multithreaded::Bool=true) where {T<:Real}
    Mk=ws.Mk
    length(As)==Mk || throw(DimensionMismatch("length(As) must equal ws.Mk"))
    @inbounds for m in 1:Mk
        fill!(As[m],0.0+0im)
    end
    direct=ws.direct
    offs=direct.offs
    Gs=direct.Gs
    Cs=direct.Cs
    parr=direct.parr
    rule=direct.rule
    plans0=ws.plans0
    plans1=ws.plans1
    h0_tls=ws.bessel_ws.h0_tls
    h1_tls=ws.bessel_ws.h1_tls
    coeff_tls=ws.bessel_ws.coeff_tls
    ks=ws.ks
    blocks=ws.block_cache.blocks
    nc=length(pts)
    @inbounds for a in 1:nc
        ra=offs[a]:(offs[a+1]-1)
        blk=blocks[a,a]
        if pts[a].is_periodic
            _assemble_self_alpert_periodic_cheb!(As,pts[a],Gs[a],Cs[a]::AlpertPeriodicCache{T},blk,ra,ks,rule,plans0,plans1,h0_tls,h1_tls;multithreaded=multithreaded)
        else
            _assemble_self_alpert_smooth_panel_cheb!(solver,As,pts[a],Gs[a],Cs[a]::AlpertSmoothPanelCache{T},ws.block_cache.Xs[a],ws.block_cache.Ys[a],ra,ks,rule,plans0,plans1,h0_tls,h1_tls,coeff_tls;multithreaded=multithreaded)
        end
    end
    _assemble_all_offpanel_blockcache!(As,ws.block_cache,ks,plans0,plans1,h0_tls,h1_tls;multithreaded=multithreaded)
    return nothing
end

"""
    compute_kernel_matrices_CFIE_alpert_chebyshev!(As,A1s,A2s,solver,pts,ws;multithreaded=true)

Assemble the CFIE-Alpert matrices and their first two k derivatives
for all wavenumbers stored in a prebuilt Chebyshev workspace, writing the
results in place.

# Arguments
- `As::Vector{Matrix{ComplexF64}}`:
  Output matrices for `A(k)`.
- `A1s::Vector{Matrix{ComplexF64}}`:
  Output matrices for `dA/dk`.
- `A2s::Vector{Matrix{ComplexF64}}`:
  Output matrices for `d²A/dk²`.
- `solver::CFIE_alpert{T}`:
  CFIE-Alpert solver.
- `pts::Vector{BoundaryPointsCFIE{T}}`:
  Boundary discretization for all components/panels.
- `ws::CFIEAlpertChebWorkspace{T}`:
  Prebuilt Chebyshev workspace.
- `multithreaded::Bool=true`:
  Enables threaded assembly where supported.

# Returns
- `nothing`
"""
function compute_kernel_matrices_CFIE_alpert_chebyshev!(As::Vector{Matrix{ComplexF64}},A1s::Vector{Matrix{ComplexF64}},A2s::Vector{Matrix{ComplexF64}},solver::CFIE_alpert{T},pts::Vector{BoundaryPointsCFIE{T}},ws::CFIEAlpertChebWorkspace{T};multithreaded::Bool=true) where {T<:Real}
    Mk=ws.Mk
    @inbounds for m in 1:Mk
        fill!(As[m],0.0+0im)
        fill!(A1s[m],0.0+0im)
        fill!(A2s[m],0.0+0im)
    end
    direct=ws.direct
    offs=direct.offs
    Gs=direct.Gs
    Cs=direct.Cs
    parr=direct.parr
    rule=direct.rule
    plans0=ws.plans0
    plans1=ws.plans1
    h0_tls=ws.bessel_ws.h0_tls
    h1_tls=ws.bessel_ws.h1_tls
    ks=ws.ks
    blocks=ws.block_cache.blocks
    nc=length(pts)
    @inbounds for a in 1:nc
        ra=offs[a]:(offs[a+1]-1)
        blk=blocks[a,a]
        if pts[a].is_periodic
            _assemble_self_alpert_periodic_cheb_deriv!(As,A1s,A2s,pts[a],Gs[a],Cs[a]::AlpertPeriodicCache{T},blk,ra,ks,rule,plans0,plans1,h0_tls,h1_tls;multithreaded=multithreaded)
        else
            _assemble_self_alpert_smooth_panel_cheb_deriv!(solver,As,A1s,A2s,pts[a],Gs[a],Cs[a]::AlpertSmoothPanelCache{T},parr[a],ra,ks,rule,plans0,plans1,h0_tls,h1_tls;multithreaded=multithreaded)
        end
    end
    _assemble_all_offpanel_blockcache_deriv!(As,A1s,A2s,ws.block_cache,ks,plans0,plans1,h0_tls,h1_tls;multithreaded=multithreaded)
    return nothing
end

"""
    compute_kernel_matrices_CFIE_alpert_chebyshev(solver,pts,ws;multithreaded=true)
    compute_kernel_matrices_CFIE_alpert_chebyshev(solver,pts,ws,::Val{:deriv};multithreaded=true)
    compute_kernel_matrices_CFIE_alpert_chebyshev!(A,solver,pts,ws;multithreaded=true)
    compute_kernel_matrices_CFIE_alpert_chebyshev!(A,A1,A2,solver,pts,ws;multithreaded=true)

Convenience entry points for Chebyshev-accelerated CFIE-Alpert matrix assembly.
# Overload families
There are four convenience forms:

1. Allocating, value-only:
   allocates and returns one matrix per wavenumber in `ws.ks`.

2. Allocating, with derivatives:
   allocates and returns one matrix triple `(A,A1,A2)` per wavenumber, where
   `A1 = dA/dk` and `A2 = d²A/dk²`.

3. In-place, single-`k`, value-only:
   fills one preallocated matrix `A` using the first wavenumber and first
   Chebyshev plans stored in `ws`.

4. In-place, single-`k`, with derivatives:
   fills preallocated matrices `A`, `A1`, and `A2` using the first wavenumber
   and first Chebyshev plans stored in `ws`.

# Arguments
Common arguments:
- `solver::CFIE_alpert{T}`:
  CFIE-Alpert solver.
- `pts::Vector{BoundaryPointsCFIE{T}}`:
  Boundary discretization for all components/panels.
- `ws::CFIEAlpertChebWorkspace{T}`:
  Prebuilt Chebyshev workspace reused across assemblies.
- `multithreaded::Bool=true`:
  Enables threaded assembly where supported by the low-level kernels.

Allocating value-only form:
- returns `As::Vector{Matrix{ComplexF64}}`, one matrix for each wavenumber in
  `ws.ks`.

Allocating derivative form:
- `::Val{:deriv}` selects the derivative-aware overload,
- returns `(As,A1s,A2s)`, with one matrix triple per wavenumber.

Single-`k` in-place value-only form:
- `A::Matrix{ComplexF64}`:
  Destination matrix for the first wavenumber stored in `ws`.

Single-`k` in-place derivative form:
- `A::Matrix{ComplexF64}`:
  Destination matrix for the operator.
- `A1::Matrix{ComplexF64}`:
  Destination matrix for the first derivative.
- `A2::Matrix{ComplexF64}`:
  Destination matrix for the second derivative.

# Returns
- `compute_kernel_matrices_CFIE_alpert_chebyshev(solver,pts,ws)` returns
  `As`.
- `compute_kernel_matrices_CFIE_alpert_chebyshev(solver,pts,ws,Val(:deriv))`
  returns `(As,A1s,A2s)`.
- The two bang forms return `nothing`, with their destination matrices filled
  in place.
"""
function compute_kernel_matrices_CFIE_alpert_chebyshev(solver::CFIE_alpert{T},pts::Vector{BoundaryPointsCFIE{T}},ws::CFIEAlpertChebWorkspace{T};multithreaded::Bool=true) where {T<:Real}
    Ntot=ws.direct.Ntot
    Mk=ws.Mk
    As=[Matrix{ComplexF64}(undef,Ntot,Ntot) for _ in 1:Mk]
    compute_kernel_matrices_CFIE_alpert_chebyshev!(As,solver,pts,ws;multithreaded=multithreaded)
    return As
end

function compute_kernel_matrices_CFIE_alpert_chebyshev(solver::CFIE_alpert{T},pts::Vector{BoundaryPointsCFIE{T}},ws::CFIEAlpertChebWorkspace{T},::Val{:deriv};multithreaded::Bool=true) where {T<:Real}
    Ntot=ws.direct.Ntot
    Mk=ws.Mk
    As=[Matrix{ComplexF64}(undef,Ntot,Ntot) for _ in 1:Mk]
    A1s=[Matrix{ComplexF64}(undef,Ntot,Ntot) for _ in 1:Mk]
    A2s=[Matrix{ComplexF64}(undef,Ntot,Ntot) for _ in 1:Mk]
    compute_kernel_matrices_CFIE_alpert_chebyshev!(As,A1s,A2s,solver,pts,ws;multithreaded=multithreaded)
    return As,A1s,A2s
end

function compute_kernel_matrices_CFIE_alpert_chebyshev!(A::Matrix{ComplexF64},solver::CFIE_alpert{T},pts::Vector{BoundaryPointsCFIE{T}},ws::CFIEAlpertChebWorkspace{T};multithreaded::Bool=true) where {T<:Real}
    compute_kernel_matrices_CFIE_alpert_chebyshev!([A],solver,pts,CFIEAlpertChebWorkspace{T}(ws.direct,ws.block_cache,[ws.plans0[1]],[ws.plans1[1]],CFIE_H0_H1_BesselWorkspace(1;ntls=length(ws.bessel_ws.h0_tls)),[ws.ks[1]],1);multithreaded=multithreaded)
    return nothing
end

function compute_kernel_matrices_CFIE_alpert_chebyshev!(A::Matrix{ComplexF64},A1::Matrix{ComplexF64},A2::Matrix{ComplexF64},solver::CFIE_alpert{T},pts::Vector{BoundaryPointsCFIE{T}},ws::CFIEAlpertChebWorkspace{T};multithreaded::Bool=true) where {T<:Real}
    ws1=CFIEAlpertChebWorkspace{T}(ws.direct,ws.block_cache,[ws.plans0[1]],[ws.plans1[1]],CFIE_H0_H1_BesselWorkspace(1;ntls=length(ws.bessel_ws.h0_tls)),[ws.ks[1]],1)
    compute_kernel_matrices_CFIE_alpert_chebyshev!([A],[A1],[A2],solver,pts,ws1;multithreaded=multithreaded)
    return nothing
end

"""
    construct_boundary_matrices!(Tbufs,solver,pts,zj;multithreaded=true,use_chebyshev=true,n_panels=15000,M=5,timeit=false)

Assemble CFIE-Alpert boundary matrices for a collection of complex
wavenumbers, writing the results in place.

# Arguments
- `Tbufs::Vector{Matrix{ComplexF64}}`:
  Output matrices, one per wavenumber. Must be preallocated and of
  consistent size with the boundary discretization.
- `solver::CFIE_alpert`:
  CFIE-Alpert solver specifying the quadrature and formulation.
- `pts::Vector{BoundaryPointsCFIE{T}}`:
  Boundary discretization (possibly multi-component).
- `zj::Vector{ComplexF64}`:
  Wavenumbers at which the matrices are assembled.

# Keyword Arguments
- `multithreaded::Bool=true`:
  Enables threaded off-diagonal assembly when beneficial.
- `use_chebyshev::Bool=true`:
  If `true`, uses Chebyshev interpolation for special-function evaluation.
  Currently this is the only supported path for complex wavenumbers.
- `n_panels::Int=15000`:
  Number of panels used in the Chebyshev interpolation of special functions.
- `M::Int=5`:
  Chebyshev interpolation order per panel.
- `timeit::Bool=false`:
  If `true`, enables timing instrumentation via `@benchit`.

# Returns
- `nothing`
"""
function construct_boundary_matrices!(Tbufs::Vector{Matrix{ComplexF64}},solver::CFIE_alpert,pts::Vector{BoundaryPointsCFIE{T}},zj::Vector{ComplexF64};multithreaded::Bool=true,use_chebyshev::Bool=true,n_panels_h::Int=15000,M_h::Int=5,n_panels_j::Int=3000,M_j::Int=5,timeit::Bool=false) where {T<:Real}
    Mk=length(zj)
    @assert length(Tbufs)==Mk
    if use_chebyshev
        @blas_1 begin
            @benchit timeit=timeit "CFIE_alpert Direct Workspace" directws=build_cfie_alpert_workspace(solver,pts)
            @benchit timeit=timeit "CFIE_alpert Chebyshev Workspace" chebws=build_cfie_alpert_cheb_workspace(solver,pts,directws,ComplexF64.(zj);npanels=n_panels_h,M=M_h,plan_nthreads=Threads.nthreads(),ntls=Threads.nthreads())
            @inbounds for j in eachindex(Tbufs)
                fill!(Tbufs[j],0.0+0.0im)
            end
            @benchit timeit=timeit "CFIE_alpert Chebyshev" compute_kernel_matrices_CFIE_alpert_chebyshev!(Tbufs,solver,pts,chebws;multithreaded=multithreaded)
        end
    else
        @error("Direct matrix construction is only for real k currently")
    end
    return nothing
end

"""
    construct_boundary_matrices_with_derivatives!(Tbufs::Vector{Matrix{ComplexF64}},dTbufs::Vector{Matrix{ComplexF64}},ddTbufs::Vector{Matrix{ComplexF64}},solver::CFIE_alpert,pts::Vector{BoundaryPointsCFIE{T}},zj::Vector{ComplexF64};multithreaded::Bool=true,use_chebyshev::Bool=true,n_panels::Int=15000,M::Int=5,timeit::Bool=false) where {T<:Real}

Assemble CFIE-Alpert boundary matrices and their first two derivatives with
respect to the wavenumber for a collection of complex wavenumbers, writing the
results in place.

For each stored wavenumber `zj[m]`, this function fills:
- `Tbufs[m]`   with the boundary matrix `A(zj[m])`,
- `dTbufs[m]`  with the first derivative `dA/dk`,
- `ddTbufs[m]` with the second derivative `d²A/dk²`.

When `use_chebyshev=true`, the construction uses the Chebyshev-accelerated
CFIE-Alpert pathway based on interpolation plans for:
- `H₀^(1)(k r)`
- `H₁^(1)(k r)`

# Arguments
- `Tbufs::Vector{Matrix{ComplexF64}}`:
  Output matrices for `A(k)`, one per wavenumber.
- `dTbufs::Vector{Matrix{ComplexF64}}`:
  Output matrices for `dA/dk`, one per wavenumber.
- `ddTbufs::Vector{Matrix{ComplexF64}}`:
  Output matrices for `d²A/dk²`, one per wavenumber.
- `solver::CFIE_alpert`:
  CFIE-Alpert solver specifying the quadrature and formulation.
- `pts::Vector{BoundaryPointsCFIE{T}}`:
  Boundary discretization (possibly multi-component).
- `zj::Vector{ComplexF64}`:
  Wavenumbers at which the matrices and derivatives are assembled.

# Keyword Arguments
- `multithreaded::Bool=true`:
  Enables threaded off-diagonal assembly when beneficial.
- `use_chebyshev::Bool=true`:
  If `true`, uses Chebyshev interpolation for special-function evaluation.
  Currently this is the only supported path for complex wavenumbers.
- `n_panels_h::Int=15000`:
  Number of panels used in the Chebyshev interpolation of special functions for the H₀^(1) and H₁^(1) kernels.
- `M_h::Int=5`:
  Chebyshev interpolation order per panel for the H₀^(1) and H₁^(1) kernels.
- `n_panels_j::Int=3000`:
  Number of panels used in the Chebyshev interpolation of special functions for the J₀ and J₁ kernels. UNUSED HERE
- `M_j::Int=5`:
  Chebyshev interpolation order per panel for the J₀ and J₁ kernels. UNUSED HERE
- `timeit::Bool=false`:
  If `true`, enables timing instrumentation via `@benchit`.

# Returns
- `nothing`
"""
function construct_boundary_matrices_with_derivatives!(Tbufs::Vector{Matrix{ComplexF64}},dTbufs::Vector{Matrix{ComplexF64}},ddTbufs::Vector{Matrix{ComplexF64}},solver::CFIE_alpert,pts::Vector{BoundaryPointsCFIE{T}},zj::Vector{ComplexF64};multithreaded::Bool=true,use_chebyshev::Bool=true,n_panels_h::Int=15000,M_h::Int=5,n_panels_j::Int=3000,M_j::Int=5,timeit::Bool=false) where {T<:Real}
    Mk=length(zj)
    @assert length(Tbufs)==Mk
    @assert length(dTbufs)==Mk
    @assert length(ddTbufs)==Mk
    if use_chebyshev
        @blas_1 begin
            @benchit timeit=timeit "CFIE_alpert Direct Workspace" directws=build_cfie_alpert_workspace(solver,pts)
            @benchit timeit=timeit "CFIE_alpert Chebyshev Workspace" chebws=build_cfie_alpert_cheb_workspace(solver,pts,directws,ComplexF64.(zj);npanels=n_panels_h,M=M_h,plan_nthreads=Threads.nthreads(),ntls=Threads.nthreads())
            @inbounds for j in eachindex(Tbufs)
                fill!(Tbufs[j],0.0+0.0im)
                fill!(dTbufs[j],0.0+0.0im)
                fill!(ddTbufs[j],0.0+0.0im)
            end
            @benchit timeit=timeit "CFIE_alpert Derivatives Chebyshev" compute_kernel_matrices_CFIE_alpert_chebyshev!(Tbufs,dTbufs,ddTbufs,solver,pts,chebws;multithreaded=multithreaded)
        end
    else
        @error("Direct derivative matrix construction is only for real k currently")
    end
    return nothing
end