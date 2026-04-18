_TWO_PI=2*pi
_INV_TWO_PI=1/_TWO_PI
_EULER_OVER_PI=MathConstants.eulergamma/pi

"""
    CFIE_kress_BlockCache{T}

Geometry-and-interpolation cache for one ordered block of the CFIE-Kress matrix.

This cache represents a single pair of boundary components `(Γ_a, Γ_b)` in the
global CFIE-Kress assembly. It stores all quantities that depend only on the
geometry and boundary discretization, but not on the wavenumber `k`. Its main
purpose is to let the Chebyshev-based assembly routines evaluate the Helmholtz
special-function part cheaply while reusing all geometric data.

Conceptually, each block cache corresponds to one matrix subblock of the global
operator

    A(k) = I - ( D(k) + i k S(k) ),

restricted to target nodes on component `a` and source nodes on component `b`.

There are two qualitatively different cases:

1. `same == false`
   Off-component interaction block (`a ≠ b`).
   The kernel is smooth, so no Kress logarithmic correction is required.
   The block is assembled directly from the smooth DLP and SLP kernels.

2. `same == true`
   Same-component block (`a = b`).
   The kernel has the usual logarithmic singularity and diagonal limits.
   In this case the cache additionally stores:
   - the Kress logarithmic term `logterm`,
   - the diagonal curvature limit `kappa_i`,
   - the local Kress correction matrix block `Rkress`.

This makes the block cache the exact Chebyshev analogue of the non-Chebyshev
low-level assembly data used in the original CFIE-Kress implementation.

# Mathematical role
For a fixed pair `(i,j)` of target/source nodes in this block, the assembly only
needs:
- the distance `r = |x_i - x_j|`,
- the inverse distance `1/r`,
- the oriented DLP numerator `inner`,
- source speed and quadrature weight,
- and, in same-component blocks, the Kress split data.

The Chebyshev machinery then replaces expensive runtime evaluation of
`H₀^(1)(k r)`, `H₁^(1)(k r)`, `J₀(k r)`, `J₁(k r)` by interpolation on a
precomputed distance panel.

# Fields
- `same::Bool`:
  Whether this block is a same-component block (`a == b`). If `true`, the block
  requires Kress singular correction; if `false`, it is a smooth inter-component
  block.
- `row_offset::Int`:
  Global row offset of the target component inside the full assembled matrix.
  The global row index corresponding to local row `i` is `row_offset + i - 1`.
- `col_offset::Int`:
  Global column offset of the source component inside the full assembled matrix.
- `Ni::Int`, `Nj::Int`:
  Number of target and source nodes in this block.
- `R::Matrix{T}`:
  Pairwise distance matrix for this block, with `R[i,j] = |x_i - x_j|`.
- `invR::Matrix{T}`:
  Pairwise inverse distances, with safe zero handling on the diagonal if needed.
- `inner::Matrix{T}`:
  Oriented DLP numerator matrix. In the code this is the tangent-based quantity
  used in the standard 2D DLP formula, equivalent to the source-normal numerator
  up to the usual parameterization conventions.
- `speed_i::Vector{T}`:
  Target-side boundary speed values for same-component mirrored assembly.
- `speed_j::Vector{T}`:
  Source-side boundary speed values.
- `wi::Vector{T}`:
  Target-side quadrature weights, needed because same-component blocks fill both
  `(i,j)` and `(j,i)` entries.
- `wj::Vector{T}`:
  Source-side quadrature weights.
- `pidx::Matrix{Int32}`:
  Chebyshev panel index for each pair `(i,j)`. This tells the interpolation
  evaluator which distance panel contains `R[i,j]`.
- `tloc::Matrix{Float64}`:
  Local Chebyshev coordinate in `[-1,1]` for each pair `(i,j)` inside the chosen
  panel.
- `logterm::Union{Nothing,Matrix{T}}`:
  Same-component logarithmic Kress split term. `nothing` for off-component
  blocks.
- `kappa_i::Union{Nothing,Vector{T}}`:
  Diagonal curvature-limit contribution for same-component blocks. `nothing` for
  off-component blocks.
- `Rkress::Union{Nothing,Matrix{T}}`:
  Local Kress correction matrix block for same-component interactions. This is
  the smooth-periodic `R` matrix for `CFIE_kress` and the corner-aware version
  for `CFIE_kress_corners` / `CFIE_kress_global_corners`.
"""
struct CFIE_kress_BlockCache{T<:Real}
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
    logterm::Union{Nothing,Matrix{T}}
    kappa_i::Union{Nothing,Vector{T}}
    Rkress::Union{Nothing,Matrix{T}}
end

"""
    CFIEBlockSystemCache{T}

Global block-cache system for Chebyshev-based CFIE-Kress assembly.

This object gathers all `CFIE_kress_BlockCache` objects for all ordered pairs of
boundary components, together with the global component offsets and the distance
interval on which the Chebyshev special-function plans must be valid.

It is the geometry-side companion to the Hankel/Bessel Chebyshev plans:
- the plans know how to evaluate the special functions efficiently on
  `[rmin,rmax]`,
- the block system cache knows which geometric pair `(i,j)` needs which panel
  and local coordinate, and provides all non-special-function coefficients.

If the boundary has `nc` connected components, then the global CFIE matrix is
partitioned into `nc × nc` blocks. This cache stores all those blocks in
`blocks[a,b]`, so that the assembly kernels can loop over:
- same-component singular-corrected blocks,
- off-component smooth blocks, without rebuilding geometry information.

# Fields
- `blocks::Matrix{CFIE_kress_BlockCache{T}}`:
  Matrix of block caches, one for each ordered component pair `(a,b)`.
- `offsets::Vector{Int}`:
  Global component offsets into the full matrix. If component `a` occupies
  `offsets[a]:(offsets[a+1]-1)`, then `offsets[end]-1` is the global dimension.
- `rmin::Float64`:
  Minimum distance used to build the Chebyshev interpolation plans, after safety
  padding.
- `rmax::Float64`:
  Maximum distance used to build the Chebyshev interpolation plans, after safety
  padding.
"""
struct CFIEBlockSystemCache{T<:Real}
    blocks::Matrix{CFIE_kress_BlockCache{T}}
    offsets::Vector{Int}
    rmin::Float64
    rmax::Float64
end

"""
    CFIE_H0_H1_J0_J1_BesselWorkspace

Thread-local scratch workspace for Chebyshev-based CFIE-Kress special-function
evaluation over multiple wavenumbers.

This workspace stores reusable temporary arrays for:
- `H₀^(1)(k r)`
- `H₁^(1)(k r)`
- `J₀(k r)`
- `J₁(k r)`

evaluated across a collection of wavenumbers at one geometric distance `r`
represented by a panel index and local Chebyshev coordinate.

# Why this workspace exists
The multi-`k` Chebyshev assembly repeatedly evaluates the same four special
functions for many matrix entries. Allocating fresh vectors inside the inner
loops would be expensive and would defeat much of the performance benefit of the
Chebyshev interpolation strategy.

Instead, each thread owns one small set of scratch vectors:
- one complex vector for all `H₀` values at the active `r`,
- one for `H₁`,
- one for `J₀`,
- one for `J₁`.

The assembly kernels then fill these vectors in place and immediately consume
them.

# Fields
- `h0_tls::Vector{Vector{ComplexF64}}`:
  Thread-local storage for interpolated `H₀^(1)` values, one vector per thread.
- `h1_tls::Vector{Vector{ComplexF64}}`:
  Thread-local storage for interpolated `H₁^(1)` values.
- `j0_tls::Vector{Vector{ComplexF64}}`:
  Thread-local storage for interpolated `J₀` values.
- `j1_tls::Vector{Vector{ComplexF64}}`:
  Thread-local storage for interpolated `J₁` values.
"""
struct CFIE_H0_H1_J0_J1_BesselWorkspace
    h0_tls::Vector{Vector{ComplexF64}}
    h1_tls::Vector{Vector{ComplexF64}}
    j0_tls::Vector{Vector{ComplexF64}}
    j1_tls::Vector{Vector{ComplexF64}}
end

function CFIE_H0_H1_J0_J1_BesselWorkspace(Mk::Int;ntls::Int=Threads.nthreads())
    h0_tls=[Vector{ComplexF64}(undef,Mk) for _ in 1:ntls]
    h1_tls=[Vector{ComplexF64}(undef,Mk) for _ in 1:ntls]
    j0_tls=[Vector{ComplexF64}(undef,Mk) for _ in 1:ntls]
    j1_tls=[Vector{ComplexF64}(undef,Mk) for _ in 1:ntls]
    return CFIE_H0_H1_J0_J1_BesselWorkspace(h0_tls,h1_tls,j0_tls,j1_tls)
end

"""
    build_CFIE_plans_kress(ks,rmin,rmax;npanels=10000,M=5,grading=:uniform,geo_ratio=1.05,nthreads=1)

Build Chebyshev interpolation plans for all special functions needed by
CFIE-Kress assembly over a collection of wavenumbers.

For each `k` this constructs four plans over the common distance interval
`[rmin,rmax]`:
- `H₀^(1)(k r)`
- `H₁^(1)(k r)`
- `J₀(k r)`
- `J₁(k r)`

The Bessel-J plans are needed because, for complex `k`, the Kress split should
not be implemented by taking real parts of the Hankels.

# Returns
- `(plans0,plans1,plansj0,plansj1)`
"""
function build_CFIE_plans_kress(ks::AbstractVector{<:Number},rmin::Float64,rmax::Float64;npanels::Int=10000,M::Int=5,grading::Symbol=:uniform,geo_ratio::Real=1.05,nthreads::Int=1)
    Mk=length(ks)
    plans0=Vector{ChebHankelPlanH}(undef,Mk)
    plans1=Vector{ChebHankelPlanH}(undef,Mk)
    plansj0=Vector{ChebJPlan}(undef,Mk)
    plansj1=Vector{ChebJPlan}(undef,Mk)
    if nthreads<=1 || Mk==1
        @inbounds for m in 1:Mk
            k=ComplexF64(ks[m])
            plans0[m]=plan_h(0,1,k,rmin,rmax;npanels=npanels,M=M,grading=grading,geo_ratio=geo_ratio)
            plans1[m]=plan_h(1,1,k,rmin,rmax;npanels=npanels,M=M,grading=grading,geo_ratio=geo_ratio)
            plansj0[m]=plan_j(0,k,rmin,rmax;npanels=npanels,M=M,grading=grading,geo_ratio=geo_ratio)
            plansj1[m]=plan_j(1,k,rmin,rmax;npanels=npanels,M=M,grading=grading,geo_ratio=geo_ratio)
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
                plans0[m]=plan_h(0,1,k,rmin,rmax;npanels=npanels,M=M,grading=grading,geo_ratio=geo_ratio)
                plans1[m]=plan_h(1,1,k,rmin,rmax;npanels=npanels,M=M,grading=grading,geo_ratio=geo_ratio)
                plansj0[m]=plan_j(0,k,rmin,rmax;npanels=npanels,M=M,grading=grading,geo_ratio=geo_ratio)
                plansj1[m]=plan_j(1,k,rmin,rmax;npanels=npanels,M=M,grading=grading,geo_ratio=geo_ratio)
            end
        end
    end
    return plans0,plans1,plansj0,plansj1
end

"""
    build_cfie_kress_block_caches(comps;npanels=10000,M=5,grading=:uniform,geo_ratio=1.05,pad=(0.95,1.05))

Build the Chebyshev block-cache system for CFIE-Kress assembly.

This precomputes, for every ordered component pair `(a,b)`:
- pairwise distances `R`,
- inverse distances `invR`,
- DLP oriented numerators `inner`,
- source speeds and quadrature weights,
- panel lookup indices `pidx`,
- local Chebyshev coordinates `tloc`,
- and, for same-component blocks only, the Kress logarithmic data
  `logterm`, diagonal curvature limits `kappa_i`, and local Kress matrix block
  `Rkress`.

The resulting object is exactly what the Chebyshev assembly kernels need in
order to avoid recomputing geometry-dependent quantities at every `k`.

# Arguments
- `solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners}`:
  The CFIE-Kress solver type determines whether the same-component blocks are
  built with the smooth-periodic Kress correction or the corner-aware version.
- `comps::Vector{BoundaryPointsCFIE{T}}` : 
  The boundary discretization, given as a vector of `BoundaryPointsCFIE` objects,
  one for each connected component. Each `BoundaryPointsCFIE` contains the
  geometry and quadrature data for that component.
- `npanels::Int`:
  Number of Chebyshev panels to use for the distance interpolation (default: `10000`).
- `M::Int`:
  Chebyshev interpolation order (default: `5`).
- `grading::Symbol`:
  Grading type for the distance panels (default: `:uniform`).
- `geo_ratio::Real`:
  Geometric growth ratio for panel sizes if grading is used (default: `1.05`).
- `pad::Tuple{T,T}`:
  Safety padding factors for the minimum and maximum distances when determining the interpolation interval (default: `(0.95, 1.05)`).

# Returns
- `CFIEBlockSystemCache{T}`
"""
function build_cfie_kress_block_caches(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},comps::Vector{BoundaryPointsCFIE{T}};npanels::Int=10000,M::Int=5,grading::Symbol=:uniform,geo_ratio::Real=1.05,pad=(T(0.95),T(1.05)),rmin_cheb::Union{Nothing,Float64}=nothing) where {T<:Real}
    nc=length(comps)
    offs=component_offsets(comps)
    Gs=_is_kress_graded(solver) ? [cfie_geom_cache(p,true) for p in comps] : [cfie_geom_cache(p,false) for p in comps]
    blocks=Matrix{CFIE_kress_BlockCache{T}}(undef,nc,nc)
    global_rmin=typemax(T)
    global_rmax=zero(T)
    for a in 1:nc, b in 1:nc
        pa=comps[a]
        pb=comps[b]
        Ga=Gs[a]
        Gb=Gs[b]
        Ni=length(pa.xy)
        Nj=length(pb.xy)
        same=(a==b)
        if same
            R=copy(Ga.R)
            invR=copy(Ga.invR)
            inner=copy(Ga.inner)
            speed_i=copy(Ga.speed)
            speed_j=copy(Ga.speed)
            wi=copy(pa.ws)
            wj=copy(pa.ws)
            rmin_blk=typemax(T)
            rmax_blk=zero(T)
            @inbounds for j in 1:Nj, i in 1:Ni
                i==j && continue
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
            logterm=copy(Ga.logterm)
            kappa_i=copy(Ga.kappa)
            Rkress=zeros(T,Ni,Ni)
            if solver isa CFIE_kress
                kress_R!(Rkress)
            else
                kress_R_corner!(Rkress)
            end
            blocks[a,b]=CFIE_kress_BlockCache{T}(true,offs[a],offs[b],Ni,Nj,R,invR,inner,speed_i,speed_j,wi,wj,pidx,tloc,logterm,kappa_i,Rkress)
        else
            Xa=getindex.(pa.xy,1)
            Ya=getindex.(pa.xy,2)
            Xb=getindex.(pb.xy,1)
            Yb=getindex.(pb.xy,2)
            dXb=getindex.(pb.tangent,1)
            dYb=getindex.(pb.tangent,2)
            R=Matrix{T}(undef,Ni,Nj)
            invR=Matrix{T}(undef,Ni,Nj)
            inner=Matrix{T}(undef,Ni,Nj)
            @inbounds for j in 1:Nj, i in 1:Ni
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
            rmin_blk=typemax(T)
            rmax_blk=zero(T)
            @inbounds for j in 1:Nj, i in 1:Ni
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
            blocks[a,b]=CFIE_kress_BlockCache{T}(false,offs[a],offs[b],Ni,Nj,R,invR,inner,speed_i,speed_j,wi,wj,pidx,tloc,nothing,nothing,nothing)
        end
    end
    global_rmin_geom=Float64(global_rmin)
    global_rmax_geom=Float64(global_rmax)
    global_rmin_cheb=isnothing(rmin_cheb) ? global_rmin_geom : max(Float64(rmin_cheb),global_rmin_geom)
    pref_plan=plan_h(0,1,1.0+0im,global_rmin_cheb,global_rmax_geom;npanels=npanels,M=M,grading=grading,geo_ratio=geo_ratio)
    pans=pref_plan.panels
    for a in 1:nc, b in 1:nc
        blk=blocks[a,b]
        @inbounds for j in 1:blk.Nj, i in 1:blk.Ni
            if blk.same && i==j
                blk.pidx[i,j]=Int32(1)
                blk.tloc[i,j]=0.0
            else
                rij=Float64(blk.R[i,j])
                if rij<global_rmin_cheb
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
    return CFIEBlockSystemCache{T}(blocks,offs,global_rmin_cheb,global_rmax_geom)
end

"""
    _all_k_nosymm_CFIE_chebyshev!(As,pts,plans0,plans1,plans2,plans3,h0_tls,h1_tls,j0_tls,j1_tls,block_cache;multithreaded=true)

Assemble the CFIE-Kress Fredholm matrices for all supplied wavenumbers using
Chebyshev-interpolated special functions, without applying any symmetry
reduction.

This is the core multi-`k` Chebyshev assembly kernel for the non-derivative
operator. It fills, in place, one global matrix per wavenumber:

    A(k_m) = I - ( D(k_m) + i k_m S(k_m) ).

# Arguments
- `As::Vector{Matrix{ComplexF64}}`:
  Output matrices, one per wavenumber. Each matrix is overwritten in place.
- `pts::Vector{BoundaryPointsCFIE{T}}`:
  Boundary discretization for all connected components.
- `plans0::Vector{ChebHankelPlanH}`:
  Multi-`k` Chebyshev plans for `H₀^(1)`.
- `plans1::Vector{ChebHankelPlanH}`:
  Multi-`k` Chebyshev plans for `H₁^(1)`.
- `plans2::Vector{ChebJPlan}`:
  Multi-`k` Chebyshev plans for `J₀`.
- `plans3::Vector{ChebJPlan}`:
  Multi-`k` Chebyshev plans for `J₁`.
- `h0_tls`, `h1_tls`, `j0_tls`, `j1_tls`:
  Thread-local temporary buffers, typically taken from
  `CFIE_H0_H1_J0_J1_BesselWorkspace`.
- `block_cache::CFIEBlockSystemCache{T}`:
  Precomputed block geometry/interpolation cache.
- `multithreaded::Bool=true`:
  Enables threaded columnwise assembly when beneficial.

# Returns
- `nothing`
"""
function _all_k_nosymm_CFIE_chebyshev!(As::Vector{Matrix{ComplexF64}},pts::Vector{BoundaryPointsCFIE{T}},plans0::Vector{ChebHankelPlanH},plans1::Vector{ChebHankelPlanH},plans2::Vector{ChebJPlan},plans3::Vector{ChebJPlan},h0_tls::Vector{Vector{ComplexF64}},h1_tls::Vector{Vector{ComplexF64}},j0_tls::Vector{Vector{ComplexF64}},j1_tls::Vector{Vector{ComplexF64}},block_cache::CFIEBlockSystemCache{T};multithreaded::Bool=true) where {T<:Real}
    Mk=length(plans0)
    ks=Vector{ComplexF64}(undef,Mk)
    αL1=Vector{ComplexF64}(undef,Mk)
    αL2=Vector{ComplexF64}(undef,Mk)
    iks=Vector{ComplexF64}(undef,Mk)
    @inbounds for m in 1:Mk
        km=ComplexF64(plans1[m].k)
        ks[m]=km
        αL1[m]=-km*_INV_TWO_PI
        αL2[m]=0.5im*km
        iks[m]=1im*km
        fill!(As[m],0)
    end
    αM1=-_INV_TWO_PI
    αM2=0.5im
    blocks=block_cache.blocks
    nc=size(blocks,1)
    function same_block_col!(blk::CFIE_kress_BlockCache{T},j::Int,h0vals::Vector{ComplexF64},h1vals::Vector{ComplexF64},j0vals::Vector{ComplexF64},j1vals::Vector{ComplexF64}) where {T<:Real}
        ro=blk.row_offset
        co=blk.col_offset
        sj=blk.speed_j[j]
        wj=blk.wj[j]
        gj=co+j-1
        gi=ro+j-1
        κj=blk.kappa_i[j]
        rjj=blk.Rkress[j,j]
        @inbounds for m in 1:Mk
            km=ks[m]
            dval=ComplexF64(wj*κj,0.0)
            m1=αM1*sj
            m2=((0.5im-_EULER_OVER_PI)-_INV_TWO_PI*log((km^2/4)*(sj^2)))*sj
            sval=ComplexF64(rjj*m1,0.0)+wj*m2
            As[m][gi,gj]=1.0-(dval+iks[m]*sval)
        end
        @inbounds for i in (j+1):blk.Ni
            gi=ro+i-1
            invr=blk.invR[i,j]
            r=blk.R[i,j]
            p=blk.pidx[i,j]
            t=blk.tloc[i,j]
            lt=blk.logterm[i,j]
            rijR=blk.Rkress[i,j]
            inn_ij=blk.inner[i,j]
            inn_ji=blk.inner[j,i]
            si=blk.speed_i[i]
            wi=blk.wi[i]
            h0_h1_j0_j1_multi_ks_at_r!(h0vals,h1vals,j0vals,j1vals,plans0,plans1,plans2,plans3,p,t,Float64(r))
            for m in 1:Mk
                h0=h0vals[m]
                h1=h1vals[m]
                j0=j0vals[m]
                j1=j1vals[m]
                βL1=αL1[m]*j1*invr
                βL2=αL2[m]*h1*invr
                βM1=αM1*j0
                βM2=αM2*h0
                l1ij=βL1*inn_ij
                l1ji=βL1*inn_ji
                l2ij=βL2*inn_ij-l1ij*lt
                l2ji=βL2*inn_ji-l1ji*lt
                dvalij=rijR*l1ij+wj*l2ij
                dvalji=rijR*l1ji+wi*l2ji
                m1j=βM1*sj
                m1i=βM1*si
                m2j=βM2*sj-m1j*lt
                m2i=βM2*si-m1i*lt
                svalij=rijR*m1j+wj*m2j
                svalji=rijR*m1i+wi*m2i
                As[m][gi,gj]=-(dvalij+iks[m]*svalij)
                As[m][gj,gi]=-(dvalji+iks[m]*svalji)
            end
        end
        return nothing
    end
    function off_block_col!(blk::CFIE_kress_BlockCache{T},j::Int,h0vals::Vector{ComplexF64},h1vals::Vector{ComplexF64}) where {T<:Real}
        ro=blk.row_offset
        co=blk.col_offset
        sj=blk.speed_j[j]
        wj=blk.wj[j]
        gj=co+j-1
        @inbounds for i in 1:blk.Ni
            gi=ro+i-1
            invr=blk.invR[i,j]
            inn=blk.inner[i,j]
            r=blk.R[i,j]
            p=blk.pidx[i,j]
            t=blk.tloc[i,j]
            h0_h1_multi_ks_at_r!(h0vals,h1vals,plans0,plans1,p,t,Float64(r))
            for m in 1:Mk
                h0=h0vals[m]
                h1=h1vals[m]
                dval=wj*(αL2[m]*inn*h1*invr)
                sval=wj*(αM2*h0*sj)
                As[m][gi,gj]=-(dval+iks[m]*sval)
            end
        end
        return nothing
    end
    for a in 1:nc
        blk=blocks[a,a]
        @use_threads multithreading=multithreaded for j in 1:blk.Nj
            tid=Threads.threadid()
            same_block_col!(blk,j,h0_tls[tid],h1_tls[tid],j0_tls[tid],j1_tls[tid])
        end
    end
    for a in 1:nc, b in 1:nc
        a==b && continue
        blk=blocks[a,b]
        @use_threads multithreading=multithreaded for j in 1:blk.Nj
            tid=Threads.threadid()
            off_block_col!(blk,j,h0_tls[tid],h1_tls[tid])
        end
    end
    return nothing
end

# Convenience wrapper for the single-matrix case.
function _one_k_nosymm_CFIE_chebyshev!(A::Matrix{ComplexF64},pts::Vector{BoundaryPointsCFIE{T}},plan0::ChebHankelPlanH,plan1::ChebHankelPlanH,plan2::ChebJPlan,plan3::ChebJPlan,h0_tls::Vector{Vector{ComplexF64}},h1_tls::Vector{Vector{ComplexF64}},j0_tls::Vector{Vector{ComplexF64}},j1_tls::Vector{Vector{ComplexF64}},block_cache::CFIEBlockSystemCache{T};multithreaded::Bool=true) where {T<:Real}
    _all_k_nosymm_CFIE_chebyshev!([A],pts,[plan0],[plan1],[plan2],[plan3],h0_tls,h1_tls,j0_tls,j1_tls,block_cache;multithreaded=multithreaded)
    return nothing
end

"""
    _all_k_nosymm_CFIE_chebyshev_deriv!(As,A1s,A2s,pts,plans0,plans1,plans2,plans3,h0_tls,h1_tls,j0_tls,j1_tls,block_cache;multithreaded=true)

Assemble the CFIE-Kress Fredholm matrices and their first two wavenumber
derivatives for all supplied wavenumbers, using Chebyshev-interpolated special
functions and no symmetry reduction.

For each wavenumber `k_m`, this routine computes in place:

- `A(k_m)`,
- `A₁(k_m) = dA/dk`,
- `A₂(k_m) = d²A/dk²`.

The assembled operator is again

    A(k) = I - ( D(k) + i k S(k) ),

so the derivatives follow the exact formulas used in the original non-Chebyshev
assembly:

    A′(k)  = -( D′(k) + i S(k) + i k S′(k) )
    A′′(k) = -( D′′(k) + 2 i S′(k) + i k S′′(k) ).

# Arguments
- `As::Vector{Matrix{ComplexF64}}`:
  Output matrices for `A(k_m)`.
- `A1s::Vector{Matrix{ComplexF64}}`:
  Output matrices for `dA/dk`.
- `A2s::Vector{Matrix{ComplexF64}}`:
  Output matrices for `d²A/dk²`.
- `pts::Vector{BoundaryPointsCFIE{T}}`:
  Boundary discretization for all connected components.
- `plans0::Vector{ChebHankelPlanH}`:
  Chebyshev plans for `H₀^(1)`.
- `plans1::Vector{ChebHankelPlanH}`:
  Chebyshev plans for `H₁^(1)`.
- `plans2::Vector{ChebJPlan}`:
  Chebyshev plans for `J₀`.
- `plans3::Vector{ChebJPlan}`:
  Chebyshev plans for `J₁`.
- `h0_tls`, `h1_tls`, `j0_tls`, `j1_tls`:
  Thread-local temporary storage.
- `block_cache::CFIEBlockSystemCache{T}`:
  Geometry/interpolation cache for all blocks.
- `multithreaded::Bool=true`:
  Enables threaded block-column assembly.

# Returns
- `nothing`
"""
function _all_k_nosymm_CFIE_chebyshev_deriv!(As::Vector{Matrix{ComplexF64}},A1s::Vector{Matrix{ComplexF64}},A2s::Vector{Matrix{ComplexF64}},pts::Vector{BoundaryPointsCFIE{T}},plans0::Vector{ChebHankelPlanH},plans1::Vector{ChebHankelPlanH},plans2::Vector{ChebJPlan},plans3::Vector{ChebJPlan},h0_tls::Vector{Vector{ComplexF64}},h1_tls::Vector{Vector{ComplexF64}},j0_tls::Vector{Vector{ComplexF64}},j1_tls::Vector{Vector{ComplexF64}},block_cache::CFIEBlockSystemCache{T};multithreaded::Bool=true) where {T<:Real}
    Mk=length(plans0)
    ks=Vector{ComplexF64}(undef,Mk)
    αL1=Vector{ComplexF64}(undef,Mk)
    αL2=Vector{ComplexF64}(undef,Mk)
    iks=Vector{ComplexF64}(undef,Mk)
    @inbounds for m in 1:Mk
        km=ComplexF64(plans1[m].k)
        ks[m]=km
        αL1[m]=-km*_INV_TWO_PI
        αL2[m]=0.5im*km
        iks[m]=1im*km
        fill!(As[m],0)
        fill!(A1s[m],0)
        fill!(A2s[m],0)
    end
    αM1=-_INV_TWO_PI
    αM2=0.5im
    blocks=block_cache.blocks
    nc=size(blocks,1)
    function same_block_col_deriv!(blk::CFIE_kress_BlockCache{T},j::Int,h0vals::Vector{ComplexF64},h1vals::Vector{ComplexF64},j0vals::Vector{ComplexF64},j1vals::Vector{ComplexF64}) where {T<:Real}
        ro=blk.row_offset
        co=blk.col_offset
        sj=blk.speed_j[j]
        wj=blk.wj[j]
        gj=co+j-1
        gi=ro+j-1
        κj=blk.kappa_i[j]
        rjj=blk.Rkress[j,j]
        @inbounds for m in 1:Mk
            km=ks[m]
            dval=ComplexF64(wj*κj,0.0)
            m1=αM1*sj
            m2=((0.5im-_EULER_OVER_PI)-_INV_TWO_PI*log((km^2/4)*(sj^2)))*sj
            sval=ComplexF64(rjj*m1,0.0)+wj*m2
            m2_1=-(sj/(π*km))
            m2_2=(sj/(π*km^2))
            sval1=wj*m2_1
            sval2=wj*m2_2
            As[m][gi,gj]=1.0-(dval+iks[m]*sval)
            A1s[m][gi,gj]=-(1im*sval+iks[m]*sval1)
            A2s[m][gi,gj]=-(2im*sval1+iks[m]*sval2)
        end
        @inbounds for i in (j+1):blk.Ni
            gi=ro+i-1
            invr=blk.invR[i,j]
            r=blk.R[i,j]
            p=blk.pidx[i,j]
            t=blk.tloc[i,j]
            lt=blk.logterm[i,j]
            rijR=blk.Rkress[i,j]
            inn_ij=blk.inner[i,j]
            inn_ji=blk.inner[j,i]
            si=blk.speed_i[i]
            wi=blk.wi[i]
            h0_h1_j0_j1_multi_ks_at_r!(h0vals,h1vals,j0vals,j1vals,plans0,plans1,plans2,plans3,p,t,Float64(r))
            for m in 1:Mk
                km=ks[m]
                h0=h0vals[m]
                h1=h1vals[m]
                j0=j0vals[m]
                j1=j1vals[m]
                l1_ij=αL1[m]*inn_ij*j1*invr
                l2_ij=αL2[m]*inn_ij*h1*invr-l1_ij*lt
                dval_ij=rijR*l1_ij+wj*l2_ij
                l1_ij_1=-(inn_ij*km*j0)*_INV_TWO_PI
                l1_ij_2=(inn_ij*(km*r*j1-j0))*_INV_TWO_PI
                l2_ij_1=(inn_ij*km*(lt*j0+1im*π*h0))*_INV_TWO_PI
                l2_ij_2=(inn_ij*(lt*(j0-km*r*j1)+1im*π*(h0-km*r*h1)))*_INV_TWO_PI
                dval_ij_1=rijR*l1_ij_1+wj*l2_ij_1
                dval_ij_2=rijR*l1_ij_2+wj*l2_ij_2
                l1_ji=αL1[m]*inn_ji*j1*invr
                l2_ji=αL2[m]*inn_ji*h1*invr-l1_ji*lt
                dval_ji=rijR*l1_ji+wi*l2_ji
                l1_ji_1=-(inn_ji*km*j0)*_INV_TWO_PI
                l1_ji_2=(inn_ji*(km*r*j1-j0))*_INV_TWO_PI
                l2_ji_1=(inn_ji*km*(lt*j0+1im*π*h0))*_INV_TWO_PI
                l2_ji_2=(inn_ji*(lt*(j0-km*r*j1)+1im*π*(h0-km*r*h1)))*_INV_TWO_PI
                dval_ji_1=rijR*l1_ji_1+wi*l2_ji_1
                dval_ji_2=rijR*l1_ji_2+wi*l2_ji_2
                m1_ij=αM1*j0*sj
                m2_ij=αM2*h0*sj-m1_ij*lt
                sval_ij=rijR*m1_ij+wj*m2_ij
                m1_ij_1=(r*sj*j1)*_INV_TWO_PI
                m1_ij_2=(r*sj*(km*r*j0-j1))*_INV_TWO_PI/km
                m2_ij_1=-(r*sj*(lt*j1+1im*π*h1))*_INV_TWO_PI
                m2_ij_2=(r*sj*(lt*(j1-km*r*j0)-1im*π*km*r*h0+1im*π*h1))*_INV_TWO_PI/km
                sval_ij_1=rijR*m1_ij_1+wj*m2_ij_1
                sval_ij_2=rijR*m1_ij_2+wj*m2_ij_2
                m1_ji=αM1*j0*si
                m2_ji=αM2*h0*si-m1_ji*lt
                sval_ji=rijR*m1_ji+wi*m2_ji
                m1_ji_1=(r*si*j1)*_INV_TWO_PI
                m1_ji_2=(r*si*(km*r*j0-j1))*_INV_TWO_PI/km
                m2_ji_1=-(r*si*(lt*j1+1im*π*h1))*_INV_TWO_PI
                m2_ji_2=(r*si*(lt*(j1-km*r*j0)-1im*π*km*r*h0+1im*π*h1))*_INV_TWO_PI/km
                sval_ji_1=rijR*m1_ji_1+wi*m2_ji_1
                sval_ji_2=rijR*m1_ji_2+wi*m2_ji_2
                As[m][gi,gj]=-(dval_ij+iks[m]*sval_ij)
                A1s[m][gi,gj]=-(dval_ij_1+1im*sval_ij+iks[m]*sval_ij_1)
                A2s[m][gi,gj]=-(dval_ij_2+2im*sval_ij_1+iks[m]*sval_ij_2)
                As[m][gj,gi]=-(dval_ji+iks[m]*sval_ji)
                A1s[m][gj,gi]=-(dval_ji_1+1im*sval_ji+iks[m]*sval_ji_1)
                A2s[m][gj,gi]=-(dval_ji_2+2im*sval_ji_1+iks[m]*sval_ji_2)
            end
        end
        return nothing
    end
    function off_block_col_deriv!(blk::CFIE_kress_BlockCache{T},j::Int,h0vals::Vector{ComplexF64},h1vals::Vector{ComplexF64}) where {T<:Real}
        ro=blk.row_offset
        co=blk.col_offset
        sj=blk.speed_j[j]
        wj=blk.wj[j]
        gj=co+j-1
        @inbounds for i in 1:blk.Ni
            gi=ro+i-1
            r=blk.R[i,j]
            invr=blk.invR[i,j]
            inn=blk.inner[i,j]
            p=blk.pidx[i,j]
            t=blk.tloc[i,j]
            h0_h1_multi_ks_at_r!(h0vals,h1vals,plans0,plans1,p,t,Float64(r))
            for m in 1:Mk
                km=ks[m]
                h0=h0vals[m]
                h1=h1vals[m]
                dval=wj*(0.5im*km*inn*h1*invr)
                dval1=wj*(-(0.5im)*inn*km*h0)
                dval2=wj*(-(0.5im)*inn*(h0-km*r*h1))
                sval=wj*(0.5im*h0*sj)
                sval1=wj*(-(0.5im)*r*h1*sj)
                sval2=wj*((0.5im)*r*(h1-km*r*h0)*sj/km)
                As[m][gi,gj]=-(dval+iks[m]*sval)
                A1s[m][gi,gj]=-(dval1+1im*sval+iks[m]*sval1)
                A2s[m][gi,gj]=-(dval2+2im*sval1+iks[m]*sval2)
            end
        end
        return nothing
    end
    for a in 1:nc
        blk=blocks[a,a]
        @use_threads multithreading=multithreaded for j in 1:blk.Nj
            tid=Threads.threadid()
            same_block_col_deriv!(blk,j,h0_tls[tid],h1_tls[tid],j0_tls[tid],j1_tls[tid])
        end
    end
    for a in 1:nc, b in 1:nc
        a==b && continue
        blk=blocks[a,b]
        @use_threads multithreading=multithreaded for j in 1:blk.Nj
            tid=Threads.threadid()
            off_block_col_deriv!(blk,j,h0_tls[tid],h1_tls[tid])
        end
    end
    return nothing
end

# Convenience wrapper for single `k` when derivatives are needed.
function _one_k_nosymm_CFIE_chebyshev_deriv!(A::Matrix{ComplexF64},A1::Matrix{ComplexF64},A2::Matrix{ComplexF64},pts::Vector{BoundaryPointsCFIE{T}},plan0::ChebHankelPlanH,plan1::ChebHankelPlanH,plan2::ChebJPlan,plan3::ChebJPlan,h0_tls::Vector{Vector{ComplexF64}},h1_tls::Vector{Vector{ComplexF64}},j0_tls::Vector{Vector{ComplexF64}},j1_tls::Vector{Vector{ComplexF64}},block_cache::CFIEBlockSystemCache{T};multithreaded::Bool=true) where {T<:Real}
    _all_k_nosymm_CFIE_chebyshev_deriv!([A],[A1],[A2],pts,[plan0],[plan1],[plan2],[plan3],h0_tls,h1_tls,j0_tls,j1_tls,block_cache;multithreaded=multithreaded)
    return nothing
end

#################################
#### HIGH LEVEL ENTRY POINTS ####
#################################

"""
    compute_kernel_matrices_CFIE_kress_chebyshev!(As,pts,plans0,plans1,plans2,plans3,h0_tls,h1_tls,j0_tls,j1_tls,block_cache;multithreaded=true)
    compute_kernel_matrices_CFIE_kress_chebyshev!(A,pts,plan0,plan1,plan2,plan3,h0_tls,h1_tls,j0_tls,j1_tls,block_cache;multithreaded=true)
    compute_kernel_matrices_CFIE_kress_chebyshev!(As,A1s,A2s,pts,plans0,plans1,plans2,plans3,h0_tls,h1_tls,j0_tls,j1_tls,block_cache;multithreaded=true)
    compute_kernel_matrices_CFIE_kress_chebyshev!(A,A1,A2,pts,plan0,plan1,plan2,plan3,h0_tls,h1_tls,j0_tls,j1_tls,block_cache;multithreaded=true)

High-level Chebyshev-accelerated CFIE-Kress kernel assembly interface.
These methods are the public entry points for assembling the CFIE-Kress Fredholm
matrices using Chebyshev interpolation of the special functions. They provide a
thin API layer over the lower-level internal kernels

- `_all_k_nosymm_CFIE_chebyshev!`
- `_one_k_nosymm_CFIE_chebyshev!`
- `_all_k_nosymm_CFIE_chebyshev_deriv!`
- `_one_k_nosymm_CFIE_chebyshev_deriv!`

and preserve the same assembly target as the original non-Chebyshev CFIE-Kress
code, namely

    A(k) = I - ( D(k) + i k S(k) ),

together, when requested, with its first and second derivatives with respect to
the wavenumber `k`.

Overview
--------
There are four overload families:

1. Multi-`k`, matrix-only:
   assembles one Fredholm matrix for each wavenumber in a supplied list of
   Chebyshev plans.

2. Single-`k`, matrix-only:
   assembles one Fredholm matrix using one set of single-`k` plans.

3. Multi-`k`, matrix plus derivatives:
   assembles `A(k_m)`, `A₁(k_m)`, and `A₂(k_m)` for every supplied wavenumber.

4. Single-`k`, matrix plus derivatives:
   assembles `A(k)`, `A₁(k)`, and `A₂(k)` for one wavenumber.

# Arguments
Common arguments:
- `pts::Vector{BoundaryPointsCFIE{T}}`:
  Boundary discretization for all connected components.
- `block_cache::CFIEBlockSystemCache{T}`:
  Precomputed geometry/interpolation block cache containing all pairwise
  geometry data, Kress split terms, and panel lookup data.
- `h0_tls`, `h1_tls`, `j0_tls`, `j1_tls`:
  Thread-local scratch arrays for temporary Chebyshev evaluations of
  `H₀^(1)`, `H₁^(1)`, `J₀`, and `J₁`.
- `multithreaded::Bool=true`:
  Enables threaded block-column assembly when beneficial.

Matrix-only multi-`k` form:
- `As::Vector{Matrix{ComplexF64}}`:
  Output matrices, one per wavenumber.
- `plans0::Vector{ChebHankelPlanH}`:
  Multi-`k` plans for `H₀^(1)`.
- `plans1::Vector{ChebHankelPlanH}`:
  Multi-`k` plans for `H₁^(1)`.
- `plans2::Vector{ChebJPlan}`:
  Multi-`k` plans for `J₀`.
- `plans3::Vector{ChebJPlan}`:
  Multi-`k` plans for `J₁`.

Matrix-only single-`k` form:
- `A::Matrix{ComplexF64}`:
  Output Fredholm matrix.
- `plan0::ChebHankelPlanH`:
  Single-`k` plan for `H₀^(1)`.
- `plan1::ChebHankelPlanH`:
  Single-`k` plan for `H₁^(1)`.
- `plan2::ChebJPlan`:
  Single-`k` plan for `J₀`.
- `plan3::ChebJPlan`:
  Single-`k` plan for `J₁`.

Derivative multi-`k` form:
- `As::Vector{Matrix{ComplexF64}}`:
  Output matrices for `A(k_m)`.
- `A1s::Vector{Matrix{ComplexF64}}`:
  Output matrices for `dA/dk`.
- `A2s::Vector{Matrix{ComplexF64}}`:
  Output matrices for `d²A/dk²`.
- `plans0`, `plans1`, `plans2`, `plans3`:
  Multi-`k` Chebyshev plans as above.

Derivative single-`k` form:
- `A::Matrix{ComplexF64}`:
  Output matrix for `A(k)`.
- `A1::Matrix{ComplexF64}`:
  Output matrix for `dA/dk`.
- `A2::Matrix{ComplexF64}`:
  Output matrix for `d²A/dk²`.
- `plan0`, `plan1`, `plan2`, `plan3`:
  Single-`k` Chebyshev plans as above.

# Returns
- Matrix-only forms return `nothing`, with destination matrices filled in place.
- Derivative forms also return `nothing`, with all destination matrices filled
  in place.
"""
function compute_kernel_matrices_CFIE_kress_chebyshev!(As::Vector{Matrix{ComplexF64}},pts::Vector{BoundaryPointsCFIE{T}},plans0::Vector{ChebHankelPlanH},plans1::Vector{ChebHankelPlanH},plans2::Vector{ChebJPlan},plans3::Vector{ChebJPlan},h0_tls::Vector{Vector{ComplexF64}},h1_tls::Vector{Vector{ComplexF64}},j0_tls::Vector{Vector{ComplexF64}},j1_tls::Vector{Vector{ComplexF64}},block_cache::CFIEBlockSystemCache{T};multithreaded::Bool=true) where {T<:Real}
    _all_k_nosymm_CFIE_chebyshev!(As,pts,plans0,plans1,plans2,plans3,h0_tls,h1_tls,j0_tls,j1_tls,block_cache;multithreaded=multithreaded)
    return nothing
end

function compute_kernel_matrices_CFIE_kress_chebyshev!(A::Matrix{ComplexF64},pts::Vector{BoundaryPointsCFIE{T}},plan0::ChebHankelPlanH,plan1::ChebHankelPlanH,plan2::ChebJPlan,plan3::ChebJPlan,h0_tls::Vector{Vector{ComplexF64}},h1_tls::Vector{Vector{ComplexF64}},j0_tls::Vector{Vector{ComplexF64}},j1_tls::Vector{Vector{ComplexF64}},block_cache::CFIEBlockSystemCache{T};multithreaded::Bool=true) where {T<:Real}
    _one_k_nosymm_CFIE_chebyshev!(A,pts,plan0,plan1,plan2,plan3,h0_tls,h1_tls,j0_tls,j1_tls,block_cache;multithreaded=multithreaded)
    return nothing
end

function compute_kernel_matrices_CFIE_kress_chebyshev!(As::Vector{Matrix{ComplexF64}},A1s::Vector{Matrix{ComplexF64}},A2s::Vector{Matrix{ComplexF64}},pts::Vector{BoundaryPointsCFIE{T}},plans0::Vector{ChebHankelPlanH},plans1::Vector{ChebHankelPlanH},plans2::Vector{ChebJPlan},plans3::Vector{ChebJPlan},h0_tls::Vector{Vector{ComplexF64}},h1_tls::Vector{Vector{ComplexF64}},j0_tls::Vector{Vector{ComplexF64}},j1_tls::Vector{Vector{ComplexF64}},block_cache::CFIEBlockSystemCache{T};multithreaded::Bool=true) where {T<:Real}
    _all_k_nosymm_CFIE_chebyshev_deriv!(As,A1s,A2s,pts,plans0,plans1,plans2,plans3,h0_tls,h1_tls,j0_tls,j1_tls,block_cache;multithreaded=multithreaded)
    return nothing
end

function compute_kernel_matrices_CFIE_kress_chebyshev!(A::Matrix{ComplexF64},A1::Matrix{ComplexF64},A2::Matrix{ComplexF64},pts::Vector{BoundaryPointsCFIE{T}},plan0::ChebHankelPlanH,plan1::ChebHankelPlanH,plan2::ChebJPlan,plan3::ChebJPlan,h0_tls::Vector{Vector{ComplexF64}},h1_tls::Vector{Vector{ComplexF64}},j0_tls::Vector{Vector{ComplexF64}},j1_tls::Vector{Vector{ComplexF64}},block_cache::CFIEBlockSystemCache{T};multithreaded::Bool=true) where {T<:Real}
    _one_k_nosymm_CFIE_chebyshev_deriv!(A,A1,A2,pts,plan0,plan1,plan2,plan3,h0_tls,h1_tls,j0_tls,j1_tls,block_cache;multithreaded=multithreaded)
    return nothing
end

"""
    construct_boundary_matrices!(Tbufs,solver,pts,zj;multithreaded=true,use_chebyshev=true,n_panels=15000,M=5,timeit=false)

Assemble CFIE-Kress boundary matrices for a collection of complex
wavenumbers, writing the results in place.

# Arguments
- `Tbufs::Vector{Matrix{Complex{T}}}`:
  Output matrices, one per wavenumber. Must be preallocated and consistent
  with the boundary discretization.
- `solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners}`:
  CFIE-Kress solver specifying whether the geometry is smooth, cornered, or
  globally corner-graded.
- `pts::Vector{BoundaryPointsCFIE{T}}`:
  Boundary discretization for all connected components.
- `zj::AbstractVector{Complex{T}}`:
  Wavenumbers at which the matrices are assembled.

# Keyword Arguments
- `multithreaded::Bool=true`:
  Enables threaded assembly when beneficial.
- `use_chebyshev::Bool=true`:
  If `true`, uses Chebyshev interpolation for the Hankel and Bessel special
  functions. Currently this is the only supported path for complex
  wavenumbers.
- `n_panels::Int=15000`:
  Number of distance panels used in the Chebyshev interpolation plans.
- `M::Int=5`:
  Chebyshev interpolation order per panel.
- `timeit::Bool=false`:
  If `true`, enables timing instrumentation via `@benchit`.

# Returns
- `nothing`
"""
function construct_boundary_matrices!(Tbufs::Vector{Matrix{Complex{T}}},solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},pts::Vector{BoundaryPointsCFIE{T}},zj::AbstractVector{Complex{T}};multithreaded::Bool=true,use_chebyshev::Bool=true,n_panels::Int=15000,M::Int=5,timeit::Bool=false) where {T<:Real}
    Mk=length(zj)
    @assert length(Tbufs)==Mk
    if use_chebyshev
        @blas_1 begin
            @benchit timeit=timeit "CFIE_kress Block Caches" block_cache=build_cfie_kress_block_caches(solver,pts;npanels=n_panels,M=M,grading=:uniform)
            rmin_interp=max(block_cache.rmin,rsw)
            @benchit timeit=timeit "CFIE_kress Plans" plans0,plans1,plans2,plans3=build_CFIE_plans_kress(zj,rmin_interp,block_cache.rmax;npanels=n_panels,M=M,grading=:uniform,nthreads=Threads.nthreads())
            @benchit timeit=timeit "CFIE_kress Workspace" ws=CFIE_H0_H1_J0_J1_BesselWorkspace(Mk;ntls=Threads.nthreads())
            @inbounds for j in eachindex(Tbufs)
                fill!(Tbufs[j],0.0+0.0im)
            end
            @benchit timeit=timeit "CFIE_kress Chebyshev" compute_kernel_matrices_CFIE_kress_chebyshev!(Tbufs,pts,plans0,plans1,plans2,plans3,ws.h0_tls,ws.h1_tls,ws.j0_tls,ws.j1_tls,block_cache;multithreaded=multithreaded)
        end
    else
        @error("Direct matrix construction is only for real k currently")
    end
    return nothing
end