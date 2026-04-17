_TWO_PI=2*pi
_INV_TWO_PI=1/_TWO_PI

"""
    build_CFIE_plans_alpert(ks,rmin,rmax;npanels=10000,M=5,grading=:uniform,geo_ratio=1.05,nthreads=1)

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
- `npanels, M, grading, geo_ratio`:
  Parameters passed to the Chebyshev Hankel plan builder.
- `nthreads::Int=1`:
  Number of threads used when building the plans.
- `r_switch::Float64=0.0`:
  Chebyshev cutoff radius for small-argument series patch.

# Returns
- `(plans0,plans1)`:
  Chebyshev plans for `H₀^(1)` and `H₁^(1)` for all supplied wavenumbers.
"""
function build_CFIE_plans_alpert(ks::AbstractVector{<:Number},rmin::Float64,rmax::Float64;npanels::Int=10000,M::Int=5,grading::Symbol=:uniform,geo_ratio::Real=1.05,nthreads::Int=1,r_switch::Float64=0.0)
    Mk=length(ks)
    plans0=Vector{ChebHankelPlanH}(undef,Mk)
    plans1=Vector{ChebHankelPlanH}(undef,Mk)
    if nthreads<=1 || Mk==1
        @inbounds for m in 1:Mk
            k=ComplexF64(ks[m])
            plans0[m]=plan_h(0,1,k,rmin,rmax;npanels=npanels,M=M,grading=grading,geo_ratio=geo_ratio,r_switch=r_switch)
            plans1[m]=plan_h(1,1,k,rmin,rmax;npanels=npanels,M=M,grading=grading,geo_ratio=geo_ratio,r_switch=r_switch)
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
                plans0[m]=plan_h(0,1,k,rmin,rmax;npanels=npanels,M=M,grading=grading,geo_ratio=geo_ratio,r_switch=r_switch)
                plans1[m]=plan_h(1,1,k,rmin,rmax;npanels=npanels,M=M,grading=grading,geo_ratio=geo_ratio,r_switch=r_switch)
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
    build_cfie_alpert_block_caches(solver,pts;npanels=10000,M=5,grading=:uniform,geo_ratio=1.05,pad=(0.95,1.05))

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

# Returns
- `CFIEAlpertBlockSystemCache{T}`
"""
function build_cfie_alpert_block_caches(solver::CFIE_alpert{T},pts::Vector{BoundaryPointsCFIE{T}};npanels::Int=10000,M::Int=5,grading::Symbol=:uniform,geo_ratio::Real=1.05,pad=(T(0.95),T(1.05))) where {T<:Real}
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
    global_rmin=typemax(T)
    global_rmax=zero(T)
    for a in 1:nc,b in 1:nc
        pa=pts[a];pb=pts[b]
        Ga=Gs[a];Gb=Gs[b]
        Ni=length(pa.xy);Nj=length(pb.xy)
        same=(a==b)
        if same
            R=copy(Ga.R)
            invR=copy(Ga.invR)
            inner=copy(Ga.inner)
            speed_i=copy(Ga.speed)
            speed_j=copy(Ga.speed)
            wi=copy(pa.ws)
            wj=copy(pa.ws)
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
            speed_i=copy(Ga.speed)
            speed_j=copy(Gb.speed)
            wi=copy(pa.ws)
            wj=copy(pb.ws)
            selfcache=nothing
        end

        rmin_blk=typemax(T)
        rmax_blk=zero(T)
        @inbounds for j in 1:Nj,i in 1:Ni
            if same && i==j
                continue
            end
            rij=R[i,j]
            if rij>eps(T)
                rij<rmin_blk && (rmin_blk=rij)
                rij>rmax_blk && (rmax_blk=rij)
            end
        end
        @assert isfinite(rmin_blk) && rmax_blk>zero(T)
        rmin_blk=pad[1]*rmin_blk
        rmax_blk=pad[2]*rmax_blk
        global_rmin=min(global_rmin,rmin_blk)
        global_rmax=max(global_rmax,rmax_blk)
        pidx=Matrix{Int32}(undef,Ni,Nj)
        tloc=Matrix{Float64}(undef,Ni,Nj)
        blocks[a,b]=CFIE_alpert_BlockCache{T}(same,offs[a],offs[b],Ni,Nj,R,invR,inner,speed_i,speed_j,wi,wj,pidx,tloc,selfcache)
    end
    pref_plan=plan_h(0,1,1.0+0im,Float64(global_rmin),Float64(global_rmax);npanels=npanels,M=M,grading=grading,geo_ratio=geo_ratio)
    pans=pref_plan.panels
    for a in 1:nc,b in 1:nc
        blk=blocks[a,b]
        @inbounds for j in 1:blk.Nj,i in 1:blk.Ni
            if blk.same && i==j
                blk.pidx[i,j]=Int32(1)
                blk.tloc[i,j]=0.0
            else
                rij=Float64(blk.R[i,j])
                p=_find_panel(pref_plan,rij)
                P=pans[p]
                blk.pidx[i,j]=Int32(p)
                blk.tloc[i,j]=(2*rij-(P.b+P.a))/(P.b-P.a)
            end
        end
    end
    return CFIEAlpertBlockSystemCache{T}(blocks,offs,Xs,Ys,dXs,dYs,ss,Float64(global_rmin),Float64(global_rmax))
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
    build_cfie_alpert_cheb_workspace(solver,pts,direct,ks;npanels=10000,M=5,grading=:uniform,geo_ratio=1.05,pad=(0.95,1.05),plan_nthreads=1,ntls=Threads.nthreads())

Build the full reusable Chebyshev workspace for CFIE-Alpert assembly.

This routine combines:
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
- `npanels, M, grading, geo_ratio`:
  Parameters passed to the Chebyshev Hankel plan builder.
- `pad`:
  Safety padding factors for the minimum and maximum distances used in the Chebyshev plans, applied to the blockwise extrema.
- `plan_nthreads`:
  Number of threads to use when building the Chebyshev Hankel plans.
- `ntls`:
  Number of thread-local buffers to allocate in the Bessel workspace; typically set to `Threads.nthreads()`.
- `r_switch`:
  Chebyshev cutoff radius for small-argument series patch in the Hankel plans.

# Returns
- `CFIEAlpertChebWorkspace{T}`
"""
function build_cfie_alpert_cheb_workspace(solver::CFIE_alpert{T},pts::Vector{BoundaryPointsCFIE{T}},direct::CFIEAlpertWorkspace{T},ks::Vector{ComplexF64};npanels::Int=10000,M::Int=5,grading::Symbol=:uniform,geo_ratio::Real=1.05,pad=(T(0.95),T(1.05)),plan_nthreads::Int=1,ntls::Int=Threads.nthreads(),r_switch::Float64=0.0) where {T<:Real}
    block_cache=build_cfie_alpert_block_caches(solver,pts;npanels=npanels,M=M,grading=grading,geo_ratio=geo_ratio,pad=pad)
    plans0,plans1=build_CFIE_plans_alpert(ks,block_cache.rmin,block_cache.rmax;npanels=npanels,M=M,grading=grading,geo_ratio=geo_ratio,nthreads=plan_nthreads,r_switch=r_switch)
    bessel_ws=CFIE_H0_H1_BesselWorkspace(length(ks);ntls=ntls)
    return CFIEAlpertChebWorkspace{T}(direct,block_cache,plans0,plans1,bessel_ws,ks,length(ks))
end

############################################
#### CHEB LOOKUP HELPERS FOR ONE ENTRY #####
############################################

@inline function _h0_h1_at_pidx_t!(h0vals::AbstractVector{ComplexF64},h1vals::AbstractVector{ComplexF64},pidx::Int32,t::Float64,plans0::AbstractVector{ChebHankelPlanH},plans1::AbstractVector{ChebHankelPlanH})
    h0_h1_multi_ks_at_r!(h0vals,h1vals,plans0,plans1,pidx,t)
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
    _h0_h1_at_pidx_t!(h0vals,h1vals,blk.pidx[i,j],blk.tloc[i,j],plans0,plans1)
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
    pidx=_find_panel(plans0[1],r)
    P=plans0[1].panels[pidx]
    t=(2r-(P.b+P.a))/(P.b-P.a)
    _h0_h1_at_pidx_t!(h0vals,h1vals,Int32(pidx),t,plans0,plans1)
    return nothing
end

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

"""
    _assemble_self_alpert_periodic_cheb!(As,pts,G,C,blk,row_range,ks,rule,plans0,plans1,h0_tls,h1_tls;multithreaded=true)
    _assemble_self_alpert_periodic_cheb_deriv!(As,A1s,A2s,pts,G,C,blk,row_range,ks,rule,plans0,plans1,h0_tls,h1_tls;multithreaded=true)

Assemble the self-interaction block for one periodic CFIE-Alpert component
using Chebyshev-accelerated Hankel evaluation.

The value-only form assembles:
- `A(k)`

The derivative-aware form additionally assembles:
- `A1(k)=dA/dk`
- `A2(k)=d²A/dk²`

# Arguments
- `As`, `A1s`, `A2s`: Output matrices, one per wavenumber.
- `pts::BoundaryPointsCFIE{T}`:
  Periodic boundary component.
- `G::CFIEGeomCache{T}`: Pairwise geometry cache for this component.
- `C::AlpertPeriodicCache{T}`: Periodic Alpert self-correction cache.
- `blk::CFIE_alpert_BlockCache{T}`: Block cache for the self block, used for Chebyshev lookup on node-node pairs.
- `row_range::UnitRange{Int}`: Global row/column range occupied by this component.
- `ks::Vector{ComplexF64}`: Wavenumbers being assembled simultaneously.
- `rule::AlpertLogRule{T}`: Alpert logarithmic correction rule.
- `plans0`, `plans1`: Chebyshev Hankel plans for `H₀^(1)` and `H₁^(1)`.
- `h0_tls`, `h1_tls`: Thread-local temporary arrays for interpolated Hankel values.
- `multithreaded::Bool=true`: Enables threaded row-parallel assembly.

# Returns
- `nothing`
"""
function _assemble_self_alpert_periodic_cheb!(As::Vector{Matrix{ComplexF64}},pts::BoundaryPointsCFIE{T},G::CFIEGeomCache{T},C::AlpertPeriodicCache{T},blk::CFIE_alpert_BlockCache{T},row_range::UnitRange{Int},ks::Vector{ComplexF64},rule::AlpertLogRule{T},plans0::Vector{ChebHankelPlanH},plans1::Vector{ChebHankelPlanH},h0_tls::Vector{Vector{ComplexF64}},h1_tls::Vector{Vector{ComplexF64}};multithreaded::Bool=true) where {T<:Real}
    Mk=length(ks)
    αD=Vector{ComplexF64}(undef,Mk)
    αS=ComplexF64(0,0.5)
    iks=Vector{ComplexF64}(undef,Mk)
    @inbounds for m in 1:Mk
        αD[m]=0.5im*ks[m]
        iks[m]=1im*ks[m]
    end
    R=G.R;invR=G.invR;inner=G.inner;speed=G.speed
    rp=C.rp;rm=C.rm;innp=C.innp;innm=C.innm;sp=C.sp;sm=C.sm
    offsp=C.offsp;wtp=C.wtp;offsm=C.offsm;wtm=C.wtm
    N=length(pts.xy);h=pts.ws[1];a=rule.a;jcorr=rule.j;ninterp=C.ninterp
    r0=first(row_range)-1
    @use_threads multithreading=(multithreaded && N>=16) for i in 1:N
        tid=Threads.threadid()
        h0vals=h0_tls[tid]
        h1vals=h1_tls[tid]
        gi=r0+i
        @inbounds for m in 1:Mk
            As[m][gi,gi]+=one(ComplexF64)
        end
        @inbounds for j in 1:N
            j==i && continue
            gj=r0+j
            _h0_h1_at_entry!(h0vals,h1vals,blk,i,j,plans0,plans1)
            inn=inner[i,j]
            invrij=invR[i,j]
            sj=speed[j]
            for m in 1:Mk
                As[m][gi,gj]-=h*(αD[m]*inn*h1vals[m]*invrij)+iks[m]*(h*(αS*h0vals[m]*sj))
            end
        end
        @inbounds for s in (-a+1):(a-1)
            s==0 && continue
            j=mod1(i+s,N)
            gj=r0+j
            _h0_h1_at_entry!(h0vals,h1vals,blk,i,j,plans0,plans1)
            inn=inner[i,j]
            invrij=invR[i,j]
            sj=speed[j]
            for m in 1:Mk
                As[m][gi,gj]+=h*(αD[m]*inn*h1vals[m]*invrij)+iks[m]*(h*(αS*h0vals[m]*sj))
            end
        end
        @inbounds for p in 1:jcorr
            fac=h*rule.w[p]
            r=rp[p,i]
            if isfinite(r)
                _h0_h1_at_r!(h0vals,h1vals,Float64(r),plans0,plans1)
                for m in 1:Mk
                    coeff=-(fac*(αD[m]*innp[p,i]*h1vals[m]/r))-iks[m]*(fac*(αS*h0vals[m]*sp[p,i]))
                    for q in 1:ninterp
                        As[m][gi,r0+mod1(i+offsp[p,q],N)]+=coeff*wtp[p,q]
                    end
                end
            end
            r=rm[p,i]
            if isfinite(r)
                _h0_h1_at_r!(h0vals,h1vals,Float64(r),plans0,plans1)
                for m in 1:Mk
                    coeff=-(fac*(αD[m]*innm[p,i]*h1vals[m]/r))-iks[m]*(fac*(αS*h0vals[m]*sm[p,i]))
                    for q in 1:ninterp
                        As[m][gi,r0+mod1(i+offsm[p,q],N)]+=coeff*wtm[p,q]
                    end
                end
            end
        end
    end
    return nothing
end

function _assemble_self_alpert_periodic_cheb_deriv!(As::Vector{Matrix{ComplexF64}},A1s::Vector{Matrix{ComplexF64}},A2s::Vector{Matrix{ComplexF64}},pts::BoundaryPointsCFIE{T},G::CFIEGeomCache{T},C::AlpertPeriodicCache{T},blk::CFIE_alpert_BlockCache{T},row_range::UnitRange{Int},ks::Vector{ComplexF64},rule::AlpertLogRule{T},plans0::Vector{ChebHankelPlanH},plans1::Vector{ChebHankelPlanH},h0_tls::Vector{Vector{ComplexF64}},h1_tls::Vector{Vector{ComplexF64}};multithreaded::Bool=true) where {T<:Real}
    Mk=length(ks)
    iks=Vector{ComplexF64}(undef,Mk)
    @inbounds for m in 1:Mk
        iks[m]=1im*ks[m]
    end
    R=G.R;invR=G.invR;inner=G.inner;speed=G.speed
    rp=C.rp;rm=C.rm;innp=C.innp;innm=C.innm;sp=C.sp;sm=C.sm
    offsp=C.offsp;wtp=C.wtp;offsm=C.offsm;wtm=C.wtm
    N=length(pts.xy);h=pts.ws[1];a=rule.a;jcorr=rule.j;ninterp=C.ninterp
    r0=first(row_range)-1
    @use_threads multithreading=(multithreaded && N>=16) for i in 1:N
        tid=Threads.threadid()
        h0vals=h0_tls[tid]
        h1vals=h1_tls[tid]
        gi=r0+i
        @inbounds for m in 1:Mk
            As[m][gi,gi]+=one(ComplexF64)
        end
        @inbounds for j in 1:N
            j==i && continue
            gj=r0+j
            _h0_h1_at_entry!(h0vals,h1vals,blk,i,j,plans0,plans1)
            r=R[i,j];invr=invR[i,j];inn=inner[i,j];sj=speed[j]
            for m in 1:Mk
                d0,d1,d2=_dlp_terms_h01(T,ks[m],r,inn,invr,h,h0vals[m],h1vals[m])
                s0,s1,s2=_slp_terms_h01(T,ks[m],r,sj,h,h0vals[m],h1vals[m])
                As[m][gi,gj]-=d0+iks[m]*s0
                A1s[m][gi,gj]-=d1+ComplexF64(0,1)*s0+iks[m]*s1
                A2s[m][gi,gj]-=d2+ComplexF64(0,2)*s1+iks[m]*s2
            end
        end
        @inbounds for q in (-a+1):(a-1)
            q==0 && continue
            j=mod1(i+q,N);gj=r0+j
            _h0_h1_at_entry!(h0vals,h1vals,blk,i,j,plans0,plans1)
            r=R[i,j];invr=invR[i,j];inn=inner[i,j];sj=speed[j]
            for m in 1:Mk
                d0,d1,d2=_dlp_terms_h01(T,ks[m],r,inn,invr,h,h0vals[m],h1vals[m])
                s0,s1,s2=_slp_terms_h01(T,ks[m],r,sj,h,h0vals[m],h1vals[m])
                As[m][gi,gj]+=d0+iks[m]*s0
                A1s[m][gi,gj]+=d1+ComplexF64(0,1)*s0+iks[m]*s1
                A2s[m][gi,gj]+=d2+ComplexF64(0,2)*s1+iks[m]*s2
            end
        end
        @inbounds for p in 1:jcorr
            fac=h*rule.w[p]
            r=rp[p,i]
            if isfinite(r)
                _h0_h1_at_r!(h0vals,h1vals,Float64(r),plans0,plans1)
                for m in 1:Mk
                    d0,d1,d2=_dlp_terms_h01(T,ks[m],r,innp[p,i],inv(r),fac,h0vals[m],h1vals[m])
                    s0,s1,s2=_slp_terms_h01(T,ks[m],r,sp[p,i],fac,h0vals[m],h1vals[m])
                    for q in 1:ninterp
                        gq=r0+mod1(i+offsp[p,q],N);ww=wtp[p,q]
                        As[m][gi,gq]-=(d0+iks[m]*s0)*ww
                        A1s[m][gi,gq]-=(d1+ComplexF64(0,1)*s0+iks[m]*s1)*ww
                        A2s[m][gi,gq]-=(d2+ComplexF64(0,2)*s1+iks[m]*s2)*ww
                    end
                end
            end
            r=rm[p,i]
            if isfinite(r)
                _h0_h1_at_r!(h0vals,h1vals,Float64(r),plans0,plans1)
                for m in 1:Mk
                    d0,d1,d2=_dlp_terms_h01(T,ks[m],r,innm[p,i],inv(r),fac,h0vals[m],h1vals[m])
                    s0,s1,s2=_slp_terms_h01(T,ks[m],r,sm[p,i],fac,h0vals[m],h1vals[m])
                    for q in 1:ninterp
                        gq=r0+mod1(i+offsm[p,q],N);ww=wtm[p,q]
                        As[m][gi,gq]-=(d0+iks[m]*s0)*ww
                        A1s[m][gi,gq]-=(d1+ComplexF64(0,1)*s0+iks[m]*s1)*ww
                        A2s[m][gi,gq]-=(d2+ComplexF64(0,2)*s1+iks[m]*s2)*ww
                    end
                end
            end
        end
    end
    return nothing
end

"""
    _assemble_self_alpert_smooth_panel_cheb!(solver,As,pts,G,C,X,Y,row_range,ks,rule,plans0,plans1,h0_tls,h1_tls,coeff_tls;multithreaded=true)
    _assemble_self_alpert_smooth_panel_cheb_deriv!(solver,As,A1s,A2s,pts,G,C,P,row_range,ks,rule,plans0,plans1,h0_tls,h1_tls;multithreaded=true)

Assemble the self-interaction block for one open smooth CFIE-Alpert panel
using Chebyshev-accelerated Hankel evaluation.

The value-only form assembles:
- `A(k)`

The derivative-aware form additionally assembles:
- `A1(k)=dA/dk`
- `A2(k)=d²A/dk²`

# Arguments
- `solver::CFIE_alpert{T}`: CFIE-Alpert solver.
- `As`, `A1s`, `A2s`: Output matrices, one per wavenumber.
- `pts::BoundaryPointsCFIE{T}`: Open smooth boundary panel.
- `G::CFIEGeomCache{T}`: Pairwise geometry cache for this panel.
- `C::AlpertSmoothPanelCache{T}`: Open-panel Alpert self-correction cache.
- `X`, `Y`: Flat coordinate arrays for the panel, used by the value-only form.
- `P::CFIEPanelArrays{T}`: Flat panel arrays used by the derivative-aware form.
- `row_range::UnitRange{Int}`: Global row/column range occupied by this panel.
- `ks::Vector{ComplexF64}`: Wavenumbers being assembled simultaneously.
- `rule::AlpertLogRule{T}`: Alpert logarithmic correction rule.
- `plans0`, `plans1`: Chebyshev Hankel plans for `H₀^(1)` and `H₁^(1)`.
- `h0_tls`, `h1_tls`: Thread-local temporary arrays for interpolated Hankel values.
- `coeff_tls`: Thread-local coefficient work vectors used during correction scattering in the value-only form.
- `multithreaded::Bool=true`: Enables threaded row-parallel assembly.

# Returns
- `nothing`
"""
function _assemble_self_alpert_smooth_panel_cheb!(solver::CFIE_alpert{T},As::Vector{Matrix{ComplexF64}},pts::BoundaryPointsCFIE{T},G::CFIEGeomCache{T},C::AlpertSmoothPanelCache{T},X::Vector{T},Y::Vector{T},row_range::UnitRange{Int},ks::Vector{ComplexF64},rule::AlpertLogRule{T},plans0::Vector{ChebHankelPlanH},plans1::Vector{ChebHankelPlanH},h0_tls::Vector{Vector{ComplexF64}},h1_tls::Vector{Vector{ComplexF64}},coeff_tls::Vector{Vector{ComplexF64}};multithreaded::Bool=true) where {T<:Real}
    Mk=length(ks)
    αD=Vector{ComplexF64}(undef,Mk)
    iks=Vector{ComplexF64}(undef,Mk)
    αS=0.5im
    @inbounds for m in 1:Mk
        αD[m]=0.5im*ks[m]
        iks[m]=1im*ks[m]
    end
    R=G.R;invR=G.invR;inner=G.inner;speed=G.speed
    rp=C.rp;rm=C.rm;innp=C.innp;innm=C.innm;sp=C.sp;sm=C.sm
    idxp=C.idxp;wtp=C.wtp;idxm=C.idxm;wtm=C.wtm
    N=length(X);hσ=pts.ws[1];a=rule.a;jcorr=rule.j;pinterp=size(idxp,3)
    r0=first(row_range)-1
    @use_threads multithreading=(multithreaded && N>=16) for i in 1:N
        tid=Threads.threadid()
        h0vals=h0_tls[tid]
        h1vals=h1_tls[tid]
        gi=r0+i
        xi=X[i];yi=Y[i]
        @inbounds for m in 1:Mk
            As[m][gi,gi]+=1.0+0im
        end
        @inbounds for j in 1:N
            j==i && continue
            gj=r0+j
            _h0_h1_at_r!(h0vals,h1vals,Float64(R[i,j]),plans0,plans1)
            if abs(j-i)<a
                wij=pts.ws[j]
                inn=inner[i,j]
                invrij=invR[i,j]
                for m in 1:Mk
                    As[m][gi,gj]+=wij*(αD[m]*inn*h1vals[m]*invrij)
                end
            else
                wij=pts.ws[j]
                inn=inner[i,j]
                invrij=invR[i,j]
                sj=speed[j]
                for m in 1:Mk
                    dval=wij*(αD[m]*inn*h1vals[m]*invrij)
                    sval=(wij*sj)*(αS*h0vals[m])
                    As[m][gi,gj]-=(dval+iks[m]*sval)
                end
            end
        end
        @inbounds for p in 1:jcorr
            fac=hσ*rule.w[p]
            r=rp[p,i]
            if isfinite(r)
                _h0_h1_at_r!(h0vals,h1vals,Float64(r),plans0,plans1)
                invr=inv(r)
                inn=innp[p,i]
                spp=sp[p,i]
                for m in 1:Mk
                    coeff=-(fac*(αD[m]*inn*h1vals[m]*invr))-iks[m]*(fac*(αS*h0vals[m]*spp))
                    for q in 1:pinterp
                        As[m][gi,row_range[idxp[p,i,q]]]+=coeff*wtp[p,i,q]
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
                    coeff=-(fac*(αD[m]*inn*h1vals[m]*invr))-iks[m]*(fac*(αS*h0vals[m]*smm))
                    for q in 1:pinterp
                        As[m][gi,row_range[idxm[p,i,q]]]+=coeff*wtm[p,i,q]
                    end
                end
            end
        end
    end
    return nothing
end

function _assemble_self_alpert_smooth_panel_cheb_deriv!(solver::CFIE_alpert{T},As::Vector{Matrix{ComplexF64}},A1s::Vector{Matrix{ComplexF64}},A2s::Vector{Matrix{ComplexF64}},pts::BoundaryPointsCFIE{T},G::CFIEGeomCache{T},C::AlpertSmoothPanelCache{T},P::CFIEPanelArrays{T},row_range::UnitRange{Int},ks::Vector{ComplexF64},rule::AlpertLogRule{T},plans0::Vector{ChebHankelPlanH},plans1::Vector{ChebHankelPlanH},h0_tls::Vector{Vector{ComplexF64}},h1_tls::Vector{Vector{ComplexF64}};multithreaded::Bool=true) where {T<:Real}
    Mk=length(ks)
    iks=Vector{ComplexF64}(undef,Mk)
    @inbounds for m in 1:Mk
        iks[m]=1im*ks[m]
    end
    X=P.X;w=pts.ws
    R=G.R;invR=G.invR;inner=G.inner;speed=G.speed
    rp=C.rp;rm=C.rm;innp=C.innp;innm=C.innm;sp=C.sp;sm=C.sm
    idxp=C.idxp;wtp=C.wtp;idxm=C.idxm;wtm=C.wtm
    N=length(X);hσ=w[1];a=rule.a;jcorr=rule.j
    @use_threads multithreading=(multithreaded && N>=16) for i in 1:N
        tid=Threads.threadid()
        h0vals=h0_tls[tid]
        h1vals=h1_tls[tid]
        gi=row_range[i]
        @inbounds for m in 1:Mk
            As[m][gi,gi]+=1.0+0im
        end
        @inbounds for j in 1:N
            j==i && continue
            gj=row_range[j]
            rij=R[i,j]
            _h0_h1_at_r!(h0vals,h1vals,Float64(rij),plans0,plans1)
            d0j,d1j,d2j=zero(ComplexF64),zero(ComplexF64),zero(ComplexF64)
            if abs(j-i)<a
                for m in 1:Mk
                    d0,d1,d2=_dlp_terms_h01(T,ks[m],rij,inner[i,j],invR[i,j],w[j],h0vals[m],h1vals[m])
                    As[m][gi,gj]+=d0
                    A1s[m][gi,gj]+=d1
                    A2s[m][gi,gj]+=d2
                end
            else
                for m in 1:Mk
                    d0,d1,d2=_dlp_terms_h01(T,ks[m],rij,inner[i,j],invR[i,j],w[j],h0vals[m],h1vals[m])
                    s0,s1,s2=_slp_terms_h01(T,ks[m],rij,one(T),w[j]*speed[j],h0vals[m],h1vals[m])
                    As[m][gi,gj]-=d0+iks[m]*s0
                    A1s[m][gi,gj]-=d1+ComplexF64(0,1)*s0+iks[m]*s1
                    A2s[m][gi,gj]-=d2+ComplexF64(0,2)*s1+iks[m]*s2
                end
            end
        end
        @inbounds for p in 1:jcorr
            fac=hσ*rule.w[p]
            r=rp[p,i]
            if isfinite(r)
                _h0_h1_at_r!(h0vals,h1vals,Float64(r),plans0,plans1)
                invr=inv(r)
                for m in 1:Mk
                    d0,d1,d2=_dlp_terms_h01(T,ks[m],r,innp[p,i],invr,fac,h0vals[m],h1vals[m])
                    s0,s1,s2=_slp_terms_h01(T,ks[m],r,sp[p,i],fac,h0vals[m],h1vals[m])
                    for q in axes(idxp,3)
                        gq=row_range[idxp[p,i,q]]
                        ww=wtp[p,i,q]
                        As[m][gi,gq]+=(d0-iks[m]*s0)*ww
                        A1s[m][gi,gq]+=(d1-(ComplexF64(0,1)*s0+iks[m]*s1))*ww
                        A2s[m][gi,gq]+=(d2-(ComplexF64(0,2)*s1+iks[m]*s2))*ww
                    end
                end
            end
            r=rm[p,i]
            if isfinite(r)
                _h0_h1_at_r!(h0vals,h1vals,Float64(r),plans0,plans1)
                invr=inv(r)
                for m in 1:Mk
                    d0,d1,d2=_dlp_terms_h01(T,ks[m],r,innm[p,i],invr,fac,h0vals[m],h1vals[m])
                    s0,s1,s2=_slp_terms_h01(T,ks[m],r,sm[p,i],fac,h0vals[m],h1vals[m])
                    for q in axes(idxm,3)
                        gq=row_range[idxm[p,i,q]]
                        ww=wtm[p,i,q]
                        As[m][gi,gq]+=(d0-iks[m]*s0)*ww
                        A1s[m][gi,gq]+=(d1-(ComplexF64(0,1)*s0+iks[m]*s1))*ww
                        A2s[m][gi,gq]+=(d2-(ComplexF64(0,2)*s1+iks[m]*s2))*ww
                    end
                end
            end
        end
    end
    return nothing
end

"""
    _assemble_all_offpanel_naive!(As,pts,offs,parr,ks,plans0,plans1,h0_tls,h1_tls;multithreaded=true)
    _assemble_all_offpanel_naive_deriv!(As,A1s,A2s,pts,offs,parr,ks,plans0,plans1,h0_tls,h1_tls;multithreaded=true)

Assemble all smooth off-panel and off-component CFIE-Alpert interactions using
Chebyshev-accelerated Hankel evaluation.

The value-only form assembles:
- `A(k)`

The derivative-aware form additionally assembles:
- `A1(k)=dA/dk`
- `A2(k)=d²A/dk²`

# Arguments
- `As`, `A1s`, `A2s`: Output matrices, one per wavenumber.
- `pts::Vector{BoundaryPointsCFIE{T}}`: Boundary discretization for all components/panels.
- `offs::Vector{Int}`: Global component offsets in the assembled matrix.
- `parr::Vector{CFIEPanelArrays{T}}`: Flat panel-array caches for all components/panels.
- `ks::Vector{ComplexF64}`: Wavenumbers being assembled simultaneously.
- `plans0`, `plans1`: Chebyshev Hankel plans for `H₀^(1)` and `H₁^(1)`.
- `h0_tls`, `h1_tls`: Thread-local temporary arrays for interpolated Hankel values.
- `multithreaded::Bool=true`: Enables threaded row-parallel assembly within each ordered block.

# Returns
- `nothing`
"""
function _assemble_all_offpanel_naive!(As::Vector{Matrix{ComplexF64}},pts::Vector{BoundaryPointsCFIE{T}},offs::Vector{Int},parr::Vector{CFIEPanelArrays{T}},ks::Vector{ComplexF64},plans0::Vector{ChebHankelPlanH},plans1::Vector{ChebHankelPlanH},h0_tls::Vector{Vector{ComplexF64}},h1_tls::Vector{Vector{ComplexF64}};multithreaded::Bool=true) where {T<:Real}
    Mk=length(ks)
    αD=Vector{ComplexF64}(undef,Mk)
    iks=Vector{ComplexF64}(undef,Mk)
    @inbounds for m in 1:Mk
        αD[m]=0.5im*ks[m]
        iks[m]=1im*ks[m]
    end
    αS=0.5im
    for aidx in eachindex(pts)
        ra=offs[aidx]:(offs[aidx+1]-1);Pa=parr[aidx]
        Xa=Pa.X;Ya=Pa.Y;Na=length(Xa)
        for bidx in eachindex(pts)
            bidx==aidx && continue
            pb=pts[bidx];rb=offs[bidx]:(offs[bidx+1]-1);Pb=parr[bidx]
            Xb=Pb.X;Yb=Pb.Y;dXb=Pb.dX;dYb=Pb.dY;sb=Pb.s;wb=pb.ws;Nb=length(Xb)
            @use_threads multithreading=(multithreaded && Na>=16) for i in 1:Na
                tid=Threads.threadid()
                h0vals=h0_tls[tid]
                h1vals=h1_tls[tid]
                gi=ra[i]
                xi=Xa[i];yi=Ya[i]
                @inbounds for j in 1:Nb
                    dx=xi-Xb[j];dy=yi-Yb[j]
                    r2=muladd(dx,dx,dy*dy)
                    r2<=(eps(T))^2 && continue
                    r=sqrt(r2);invr=inv(r)
                    inn=dYb[j]*dx-dXb[j]*dy
                    _h0_h1_at_r!(h0vals,h1vals,Float64(r),plans0,plans1)
                    wd=wb[j];wsj=wd*sb[j]
                    gj=rb[j]
                    for m in 1:Mk
                        dval=wd*(αD[m]*inn*h1vals[m]*invr)
                        sval=wsj*(αS*h0vals[m])
                        As[m][gi,gj]-=dval+iks[m]*sval
                    end
                end
            end
        end
    end
    return nothing
end

function _assemble_all_offpanel_naive_deriv!(As::Vector{Matrix{ComplexF64}},A1s::Vector{Matrix{ComplexF64}},A2s::Vector{Matrix{ComplexF64}},pts::Vector{BoundaryPointsCFIE{T}},offs::Vector{Int},parr::Vector{CFIEPanelArrays{T}},ks::Vector{ComplexF64},plans0::Vector{ChebHankelPlanH},plans1::Vector{ChebHankelPlanH},h0_tls::Vector{Vector{ComplexF64}},h1_tls::Vector{Vector{ComplexF64}};multithreaded::Bool=true) where {T<:Real}
    Mk=length(ks)
    iks=Vector{ComplexF64}(undef,Mk)
    @inbounds for m in 1:Mk
        iks[m]=1im*ks[m]
    end
    for aidx in eachindex(pts)
        ra=offs[aidx]:(offs[aidx+1]-1);Pa=parr[aidx]
        Xa=Pa.X;Ya=Pa.Y;Na=length(Xa)
        for bidx in eachindex(pts)
            bidx==aidx && continue
            pb=pts[bidx];rb=offs[bidx]:(offs[bidx+1]-1);Pb=parr[bidx]
            Xb=Pb.X;Yb=Pb.Y;dXb=Pb.dX;dYb=Pb.dY;sb=Pb.s;wb=pb.ws;Nb=length(Xb)
            @use_threads multithreading=(multithreaded && Na>=16) for i in 1:Na
                tid=Threads.threadid()
                h0vals=h0_tls[tid]
                h1vals=h1_tls[tid]
                gi=ra[i]
                xi=Xa[i];yi=Ya[i]
                @inbounds for j in 1:Nb
                    dx=xi-Xb[j];dy=yi-Yb[j]
                    r2=muladd(dx,dx,dy*dy)
                    r2<=(eps(T))^2 && continue
                    r=sqrt(r2);invr=inv(r)
                    inn=dYb[j]*dx-dXb[j]*dy
                    _h0_h1_at_r!(h0vals,h1vals,Float64(r),plans0,plans1)
                    wd=wb[j]
                    gj=rb[j]
                    for m in 1:Mk
                        d0,d1,d2=_dlp_terms_h01(T,ks[m],r,inn,invr,wd,h0vals[m],h1vals[m])
                        s0,s1,s2=_slp_terms_h01(T,ks[m],r,one(T),wd*sb[j],h0vals[m],h1vals[m])
                        As[m][gi,gj]-=d0+iks[m]*s0
                        A1s[m][gi,gj]-=d1+ComplexF64(0,1)*s0+iks[m]*s1
                        A2s[m][gi,gj]-=d2+ComplexF64(0,2)*s1+iks[m]*s2
                    end
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
    _assemble_all_offpanel_naive!(As,pts,offs,parr,ks,plans0,plans1,h0_tls,h1_tls;multithreaded=multithreaded)
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
    _assemble_all_offpanel_naive_deriv!(As,A1s,A2s,pts,offs,parr,ks,plans0,plans1,h0_tls,h1_tls;multithreaded=multithreaded)
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
function construct_boundary_matrices!(Tbufs::Vector{Matrix{ComplexF64}},solver::CFIE_alpert,pts::Vector{BoundaryPointsCFIE{T}},zj::Vector{ComplexF64};multithreaded::Bool=true,use_chebyshev::Bool=true,n_panels::Int=15000,M::Int=5,timeit::Bool=false) where {T<:Real}
    Mk=length(zj)
    @assert length(Tbufs)==Mk
    if use_chebyshev
        @blas_1 begin
            @benchit timeit=timeit "CFIE_alpert Direct Workspace" directws=build_cfie_alpert_workspace(solver,pts)
            @benchit timeit=timeit "CFIE_alpert Chebyshev Workspace" chebws=build_cfie_alpert_cheb_workspace(solver,pts,directws,ComplexF64.(zj);npanels=n_panels,M=M,grading=:uniform,plan_nthreads=Threads.nthreads(),ntls=Threads.nthreads())
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