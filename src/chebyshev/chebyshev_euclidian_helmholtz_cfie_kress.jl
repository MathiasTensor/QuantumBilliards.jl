_TWO_PI=2*pi
_INV_TWO_PI=1/_TWO_PI
_EULER_OVER_PI=MathConstants.eulergamma/pi

"""
    CFIE_kress_BlockCache{T}

Geometry-and-interpolation cache for one ordered block of the CFIE-Kress matrix.

This cache represents a single pair of boundary components `(Γ_a, Γ_b)` in the
global CFIE-Kress assembly. It stores all quantities that depend only on the
geometry and boundary discretization, but not on the wavenumber `k`. Its main
purpose is to let the Chebyshev-based assembly functions evaluate the Helmholtz
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
  Chebyshev panel index for each pair `(i,j)` for Hankels. This tells the interpolation
  evaluator which distance panel contains `R[i,j]`.
- `tloc::Matrix{Float64}`:
  Local Chebyshev coordinate in `[-1,1]` for each pair `(i,j)` for Hankels inside the chosen
  panel.
- `pidxj::Matrix{Int32}`:
  Chebyshev panel index for each pair `(j,i)` for Bessel J. This tells the interpolation
  evaluator which distance panel contains `R[j,i]`.
- `tlocj::Matrix{Float64}`:
  Local Chebyshev coordinate in `[-1,1]` for each pair `(j,i)` for Bessel J inside the chosen
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
    pidxj::Matrix{Int32}
    tlocj::Matrix{Float64}
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
    ReducedOrbitInfo{T}

Precomputed symmetry-orbit metadata for one reduced boundary degree of freedom.

In a symmetry-reduced boundary discretization, a single reduced unknown does
not correspond to a single full boundary node, but rather to an orbit of nodes
generated by the action of the symmetry group.

If the full boundary basis indices are denoted by `g`, and the symmetry group
maps one representative node into orbit images

    {g₁, g₂, ..., g_m},

with corresponding symmetry weights

    {χ₁, χ₂, ..., χ_m},

then a reduced matrix entry is assembled as a symmetry-weighted sum over the
full-space orbit contributions.

This struct caches all data needed for that summation so that the reduced
assembly kernels do not repeatedly:
- reconstruct symmetry orbits,
- recompute symmetry scaling coefficients,
- convert global indices into component/local block coordinates.

If the full CFIE operator is

    A_full(k),

then a reduced operator entry takes the form

    A_red(a,b)
    =
    ∑_ℓ χ_ℓ A_full(i_a, j_ℓ),

where:
- `i_a` is the full representative node for reduced row `a`,
- `j_ℓ` runs over the symmetry orbit of reduced column `b`,
- `χ_ℓ` is the associated symmetry character / phase factor.

This struct stores the orbit metadata needed for efficient evaluation of that
sum.

# Fields
- `redidx::Int`:
  Reduced index associated with this orbit representative.
- `full::Vector{Int}`:
  Full global boundary indices belonging to the orbit.
- `scales::Vector{Complex{T}}`:
  Symmetry scaling coefficients multiplying each orbit contribution.
- `cscales::Vector{Complex{T}}`:
  Complex-conjugated symmetry scaling coefficients, useful for adjoint
  formulations.
- `blocks::Vector{Int}`:
  Boundary component index for each orbit image.
- `locals::Vector{Int}`:
  Local node index inside the corresponding component block.
"""
struct ReducedOrbitInfo{T<:Real}
    redidx::Int
    full::Vector{Int}
    scales::Vector{Complex{T}}
    cscales::Vector{Complex{T}}
    blocks::Vector{Int}
    locals::Vector{Int}
end

"""
    build_reduced_orbit_infos(...)

Construct cached symmetry-orbit metadata for reduced CFIE-Kress assembly.

This function converts the raw symmetry bookkeeping:
- full-to-reduced mappings,
- reduced-to-full orbit mappings,
- symmetry scaling factors,
- global component ownership,

into a compact cache of `ReducedOrbitInfo` objects.

For a reduced matrix assembly, each reduced source degree of freedom requires
evaluation of a sum over its full symmetry orbit:

    ∑_ℓ χ_ℓ A(i,j_ℓ).

Rather than repeatedly reconstructing:
- orbit members `j_ℓ`,
- their symmetry weights `χ_ℓ`,
- and their component/local coordinates,

this function precomputes all of that once.

# Returns
- `Vector{ReducedOrbitInfo{T}}`
"""
function build_reduced_orbit_infos(Ifund::Vector{Int},full_to_fund::Vector{Int},fund_to_full::Vector{Vector{Int}},fund_to_scale::Vector{Vector{Complex{T}}},global_to_block::Vector{Int},global_to_local::Vector{Int}) where {T<:Real}
    n=length(Ifund)
    infos=Vector{ReducedOrbitInfo{T}}(undef,n)
    @inbounds for a in 1:n
        g0=Ifund[a]
        orbit=fund_to_full[a]
        scales=fund_to_scale[a]
        m=length(orbit)
        blocks=Vector{Int}(undef,m)
        locals=Vector{Int}(undef,m)
        cscales=Vector{Complex{T}}(undef,m)
        for q in 1:m
            g=orbit[q]
            blocks[q]=global_to_block[g]
            locals[q]=global_to_local[g]
            cscales[q]=conj(scales[q])
        end
        infos[a]=ReducedOrbitInfo{T}(full_to_fund[g0],orbit,scales,cscales,blocks,locals)
    end
    return infos
end

"""
    CFIEKressReducedWorkspace{T,S}

Precomputed workspace for symmetry-reduced CFIE-Kress Chebyshev assembly.

This object combines:
- the full geometric CFIE block cache,
- symmetry metadata,
- mappings between full and reduced boundary indices,
- precomputed symmetry orbits,
- component/local block lookup information.

It serves as the reduced-space companion to `CFIEBlockSystemCache`.
If reduced basis states correspond to symmetry-adapted combinations of full
boundary unknowns, then matrix entries are assembled as

    A_red(a,b) = ∑_ℓ χ_ℓ A_full(i_a,j_ℓ),

where:
- `i_a` is the full representative node for reduced row `a`,
- `{j_ℓ}` is the orbit of reduced column `b`,
- `χ_ℓ` are symmetry scaling coefficients.

# Used by
- `_all_k_symm_CFIE_chebyshev!`
- `_all_k_symm_CFIE_chebyshev_deriv!`
"""
struct CFIEKressReducedWorkspace{T<:Real,S<:AbsSymmetry}
    block_cache::CFIEBlockSystemCache{T}
    sym::S
    Ifund::Vector{Int}
    full_to_fund::Vector{Int}
    full_to_scale::Vector{Complex{T}}
    fund_to_full::Vector{Vector{Int}}
    fund_to_scale::Vector{Vector{Complex{T}}}
    global_to_block::Vector{Int}
    global_to_local::Vector{Int}
    reduced_orbits::Vector{ReducedOrbitInfo{T}}
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
    build_CFIE_plans_kress(ks::AbstractVector{<:Number},rmin::Float64,rmax::Float64;npanels_h::Int=10000,M_h::Int=5,npanels_j::Int=3000,M_j::Int=5,nthreads::Int=1)

Build Chebyshev interpolation plans for all special functions needed by
CFIE-Kress assembly over a collection of wavenumbers.

For each `k`, this constructs:
- `H₀^(1)(k r)` on `[rmin,rmax]`,
- `H₁^(1)(k r)` on `[rmin,rmax]`,
- `J₀(k r)` on `[0,rmax]`,
- `J₁(k r)` on `[0,rmax]`.

The Hankel plans use the positive cutoff interval because the Hankels are
singular near zero. The Bessel-J plans start at zero because `J₀` and `J₁` are
regular there.

# Returns
- `(plans0,plans1,plansj0,plansj1)`
"""
function build_CFIE_plans_kress(ks::AbstractVector{<:Number},rmin::Float64,rmax::Float64;npanels_h::Int=10000,M_h::Int=5,npanels_j::Int=3000,M_j::Int=5,nthreads::Int=1)
    Mk=length(ks)
    plans0=Vector{ChebHankelPlanH}(undef,Mk)
    plans1=Vector{ChebHankelPlanH}(undef,Mk)
    plansj0=Vector{ChebJPlan}(undef,Mk)
    plansj1=Vector{ChebJPlan}(undef,Mk)
    if nthreads<=1 || Mk==1
        @inbounds for m in 1:Mk
            k=ComplexF64(ks[m])
            plans0[m]=plan_h(0,1,k,rmin,rmax;npanels=npanels_h,M=M_h)
            plans1[m]=plan_h(1,1,k,rmin,rmax;npanels=npanels_h,M=M_h)
            plansj0[m]=plan_j(0,k,0.0,rmax;npanels=npanels_j,M=M_j)
            plansj1[m]=plan_j(1,k,0.0,rmax;npanels=npanels_j,M=M_j)
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
                plans0[m]=plan_h(0,1,k,rmin,rmax;npanels=npanels_h,M=M_h)
                plans1[m]=plan_h(1,1,k,rmin,rmax;npanels=npanels_h,M=M_h)
                plansj0[m]=plan_j(0,k,0.0,rmax;npanels=npanels_j,M=M_j)
                plansj1[m]=plan_j(1,k,0.0,rmax;npanels=npanels_j,M=M_j)
            end
        end
    end
    return plans0,plans1,plansj0,plansj1
end

"""
    build_cfie_kress_block_caches(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},comps::Vector{BoundaryPointsCFIE{T}};npanels_h::Int=10000,M_h::Int=5,npanels_j::Int=3000,M_j::Int=5,pad=(T(0.95),T(1.05)),rmin_cheb::Union{Nothing,Float64}=nothing) where {T<:Real}

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
- `npanels_h::Int`:
  Number of Chebyshev panels for the Hankel-based interpolation (default: `10000`).
- `M_h::Int`:
  Chebyshev interpolation order for the Hankel-based interpolation (default: `5`).
- `npanels_j::Int`:
  Number of Chebyshev panels for the Bessel-J-based interpolation (default: `3000`).
- `M_j::Int`:
  Chebyshev interpolation order for the Bessel-J-based interpolation (default: `5`).
- `pad::Tuple{T,T}`:
  Safety padding factors for the minimum and maximum distances when determining the interpolation interval (default: `(0.95, 1.05)`).

# Returns
- `CFIEBlockSystemCache{T}`
"""
function build_cfie_kress_block_caches(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},comps::Vector{BoundaryPointsCFIE{T}};npanels_h::Int=10000,M_h::Int=5,npanels_j::Int=3000,M_j::Int=5,pad=(T(0.95),T(1.05)),rmin_cheb::Union{Nothing,Float64}=nothing) where {T<:Real}
    nc=length(comps)
    offs=component_offsets(comps)
    Gs=[cfie_geom_cache(p,_is_nontrivial_grading(p)) for p in comps]
    blocks=Matrix{CFIE_kress_BlockCache{T}}(undef,nc,nc)
    global_rmin=typemax(T)
    global_rmax=zero(T)
    for a in 1:nc,b in 1:nc
        pa=comps[a]; pb=comps[b]; Ga=Gs[a]; Gb=Gs[b]
        Ni=length(pa.xy); Nj=length(pb.xy); same=(a==b)
        if same
            R=Ga.R;invR=Ga.invR;inner=Ga.inner
            speed_i=Ga.speed;speed_j=Ga.speed
            wi=pa.ws;wj=pa.ws
            rmin_blk=typemax(T); rmax_blk=zero(T)
            @inbounds for j in 1:Nj,i in 1:Ni
                i==j && continue
                rij=R[i,j]
                if rij>eps(T)
                    rij<rmin_blk && (rmin_blk=rij)
                    rij>rmax_blk && (rmax_blk=rij)
                end
            end
            rmin_blk*=pad[1];rmax_blk*=pad[2]
            global_rmin=min(global_rmin,rmin_blk);global_rmax=max(global_rmax,rmax_blk)
            pidx=Matrix{Int32}(undef,Ni,Nj);tloc=Matrix{Float64}(undef,Ni,Nj)
            pidxj=Matrix{Int32}(undef,Ni,Nj);tlocj=Matrix{Float64}(undef,Ni,Nj)
            logterm=Ga.logterm;kappa_i=Ga.kappa
            Rkress=zeros(T,Ni,Ni)
            _is_nontrivial_grading(pa) ? kress_R_corner!(Rkress) : kress_R!(Rkress)
            blocks[a,b]=CFIE_kress_BlockCache{T}(true,offs[a],offs[b],Ni,Nj,R,invR,inner,speed_i,speed_j,wi,wj,pidx,tloc,pidxj,tlocj,logterm,kappa_i,Rkress)
        else
            Xa=getindex.(pa.xy,1);Ya=getindex.(pa.xy,2)
            Xb=getindex.(pb.xy,1);Yb=getindex.(pb.xy,2)
            dXb=getindex.(pb.tangent,1);dYb=getindex.(pb.tangent,2)
            R=Matrix{T}(undef,Ni,Nj);invR=Matrix{T}(undef,Ni,Nj);inner=Matrix{T}(undef,Ni,Nj)
            @inbounds for j in 1:Nj,i in 1:Ni
                dx=Xa[i]-Xb[j];dy=Ya[i]-Yb[j];rij=hypot(dx,dy)
                R[i,j]=rij;invR[i,j]=rij>eps(T) ? inv(rij) : zero(T)
                inner[i,j]=dYb[j]*dx-dXb[j]*dy
            end
            speed_i=Ga.speed;speed_j=Gb.speed
            wi=pa.ws;wj=pb.ws
            rmin_blk=typemax(T); rmax_blk=zero(T)
            @inbounds for j in 1:Nj,i in 1:Ni
                rij=R[i,j]
                if rij>eps(T)
                    rij<rmin_blk && (rmin_blk=rij)
                    rij>rmax_blk && (rmax_blk=rij)
                end
            end
            rmin_blk*=pad[1];rmax_blk*=pad[2]
            global_rmin=min(global_rmin,rmin_blk);global_rmax=max(global_rmax,rmax_blk)
            pidx=Matrix{Int32}(undef,Ni,Nj);tloc=Matrix{Float64}(undef,Ni,Nj)
            pidxj=Matrix{Int32}(undef,Ni,Nj);tlocj=Matrix{Float64}(undef,Ni,Nj)
            blocks[a,b]=CFIE_kress_BlockCache{T}(false,offs[a],offs[b],Ni,Nj,R,invR,inner,speed_i,speed_j,wi,wj,pidx,tloc,pidxj,tlocj,nothing,nothing,nothing)
        end
    end
    global_rmin_geom=Float64(global_rmin)
    global_rmax_geom=Float64(global_rmax)
    global_rmin_cheb=isnothing(rmin_cheb) ? global_rmin_geom : max(Float64(rmin_cheb),global_rmin_geom)
    pref_h=plan_h(0,1,1.0+0im,global_rmin_cheb,global_rmax_geom;npanels=npanels_h,M=M_h)
    pref_j=plan_j(0,1.0+0im,0.0,global_rmax_geom;npanels=npanels_j,M=M_j)
    pansh=pref_h.panels; pansj=pref_j.panels
    for a in 1:nc,b in 1:nc
        blk=blocks[a,b]
        @inbounds for j in 1:blk.Nj,i in 1:blk.Ni
            if blk.same && i==j
                blk.pidx[i,j]=Int32(1);blk.tloc[i,j]=0.0
                blk.pidxj[i,j]=Int32(1);blk.tlocj[i,j]=0.0
            else
                rij=Float64(blk.R[i,j])
                if rij<global_rmin_cheb
                    blk.pidx[i,j]=Int32(0);blk.tloc[i,j]=0.0
                else
                    p=_find_panel(pref_h,rij);P=pansh[p]
                    blk.pidx[i,j]=Int32(p)
                    blk.tloc[i,j]=(2*rij-(P.b+P.a))/(P.b-P.a)
                end
                pj=_find_panel(pref_j,rij);Pj=pansj[pj]
                blk.pidxj[i,j]=Int32(pj)
                blk.tlocj[i,j]=(2*rij-(Pj.b+Pj.a))/(Pj.b-Pj.a)
            end
        end
    end
    return CFIEBlockSystemCache{T}(blocks,offs,global_rmin_cheb,global_rmax_geom)
end

"""
    build_CFIE_kress_reduced_workspace(...)

Construct the full symmetry-reduced workspace for CFIE-Kress Chebyshev
assembly.

- the full geometric CFIE block cache,
- component ownership mappings,
- global-to-local block coordinates,
- symmetry orbit decompositions,
- symmetry scaling coefficients,
- cached reduced-orbit metadata.

# Returns
- `CFIEKressReducedWorkspace{T,S}`
"""
function build_CFIE_kress_reduced_workspace(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},pts::Vector{BoundaryPointsCFIE{T}},sym::S;npanels_h::Int=10000,M_h::Int=5,npanels_j::Int=3000,M_j::Int=5,pad=(T(0.95),T(1.05)),rmin_cheb::Union{Nothing,Float64}=nothing) where {T<:Real,S<:AbsSymmetry}
    block_cache=build_cfie_kress_block_caches(solver,pts;npanels_h=npanels_h,M_h=M_h,npanels_j=npanels_j,M_j=M_j,pad=pad,rmin_cheb=rmin_cheb)
    Nfull=boundary_matrix_size(pts)
    offs=component_offsets(pts)
    global_to_block=Vector{Int}(undef,Nfull)
    global_to_local=Vector{Int}(undef,Nfull)
    @inbounds for a in 1:length(pts)
        lo=offs[a]
        hi=offs[a+1]-1
        for g in lo:hi
            global_to_block[g]=a
            global_to_local[g]=g-lo+1
        end
    end
    Ifund=Int[]
    full_to_fund=zeros(Int,Nfull)
    full_to_scale=zeros(Complex{T},Nfull)
    fund_to_full=Vector{Vector{Int}}()
    fund_to_scale=Vector{Vector{Complex{T}}}()
    visited=falses(Nfull)
    @inbounds for g in 1:Nfull
        visited[g] && continue
        orb=apply_symmetry_orbit(sym,g,Nfull)
        orb_inds=orb.indices
        orb_scales=Complex{T}.(orb.scales)
        push!(Ifund,g)
        redcol=length(Ifund)
        push!(fund_to_full,orb_inds)
        push!(fund_to_scale,orb_scales)
        for q in eachindex(orb_inds)
            gg=orb_inds[q]
            visited[gg]=true
            full_to_fund[gg]=redcol
            full_to_scale[gg]=orb_scales[q]
        end
    end
    reduced_orbits=build_reduced_orbit_infos(Ifund,full_to_fund,fund_to_full,fund_to_scale,global_to_block,global_to_local)
    return CFIEKressReducedWorkspace(block_cache,sym,Ifund,full_to_fund,full_to_scale,fund_to_full,fund_to_scale,global_to_block,global_to_local,reduced_orbits)
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
        fill!(As[m],0.0+0.0im)
    end
    αM1=-_INV_TWO_PI
    αM2=0.5im
    blocks=block_cache.blocks
    nc=size(blocks,1)
    # -------------------------------------------------------------------------
    # Assemble one source-column `j` of a same-component CFIE block.
    #
    # This is the singular Kress-corrected case (`Γ_target = Γ_source`).
    #
    # The assembled operator is
    #
    #     A(k) = I - ( D(k) + i k S(k) ).
    #
    # Diagonal entries use the analytic singular limits:
    #
    #   - the DLP curvature jump contribution,
    #   - the finite-part Helmholtz SLP diagonal.
    #
    # Off-diagonal entries use the Kress logarithmic split:
    #
    #     kernel = smooth Hankel part + logarithmic Bessel correction.
    # -------------------------------------------------------------------------
    function same_block_col!(blk::CFIE_kress_BlockCache{T},j::Int,h0vals::Vector{ComplexF64},h1vals::Vector{ComplexF64},j0vals::Vector{ComplexF64},j1vals::Vector{ComplexF64}) where {T<:Real}
        ro=blk.row_offset;co=blk.col_offset
        sj=blk.speed_j[j];wj=blk.wj[j]
        gj=co+j-1;gi=ro+j-1
        κj=blk.kappa_i[j];rjj=blk.Rkress[j,j]
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
            r=blk.R[i,j];invr=blk.invR[i,j];lt=blk.logterm[i,j];Rij=blk.Rkress[i,j]
            inn_ij=blk.inner[i,j];inn_ji=blk.inner[j,i]
            si=blk.speed_i[i];wi=blk.wi[i]
            h0_h1_j0_j1_multi_ks_at_r!(h0vals,h1vals,j0vals,j1vals,plans0,plans1,plans2,plans3,blk.pidx[i,j],blk.tloc[i,j],blk.pidxj[i,j],blk.tlocj[i,j],Float64(r))
            cD1ij=Rij*inn_ij*invr;cD2ij=wj*inn_ij*invr;cD3ij=wj*lt*inn_ij*invr
            cD1ji=Rij*inn_ji*invr;cD2ji=wi*inn_ji*invr;cD3ji=wi*lt*inn_ji*invr
            cS1j=Rij*sj;cS2j=wj*sj;cS3j=wj*sj*lt
            cS1i=Rij*si;cS2i=wi*si;cS3i=wi*si*lt
            for m in 1:Mk
                h0=h0vals[m];h1=h1vals[m];j0=j0vals[m];j1=j1vals[m]
                L1=αL1[m]*j1;L2=αL2[m]*h1
                M1=αM1*j0;M2=αM2*h0
                dvalij=cD1ij*L1+cD2ij*L2-cD3ij*L1
                dvalji=cD1ji*L1+cD2ji*L2-cD3ji*L1
                svalij=cS1j*M1+cS2j*M2-cS3j*M1
                svalji=cS1i*M1+cS2i*M2-cS3i*M1
                As[m][gi,gj]=-(dvalij+iks[m]*svalij)
                As[m][gj,gi]=-(dvalji+iks[m]*svalji)
            end
        end
        return nothing
    end
    # -------------------------------------------------------------------------
    # Assemble one source-column `j` of a smooth off-component CFIE block.
    #
    # This is the nonsingular case (`Γ_target ≠ Γ_source`).
    #
    # Since source and target lie on distinct boundary components, the Helmholtz
    # kernels are smooth:
    #
    #     r = |x_i - x_j| > 0.
    #
    # No diagonal singular treatment or Kress logarithmic splitting is required.
    #
    # The block contribution is assembled directly from:
    #
    #     D(k) + i k S(k).
    # -------------------------------------------------------------------------
    function off_block_col!(blk::CFIE_kress_BlockCache{T},j::Int,h0vals::Vector{ComplexF64},h1vals::Vector{ComplexF64}) where {T<:Real}
        ro=blk.row_offset;co=blk.col_offset
        sj=blk.speed_j[j];wj=blk.wj[j];gj=co+j-1
        @inbounds for i in 1:blk.Ni
            gi=ro+i-1
            r=blk.R[i,j];invr=blk.invR[i,j];inn=blk.inner[i,j]
            h0_h1_multi_ks_at_r!(h0vals,h1vals,plans0,plans1,blk.pidx[i,j],blk.tloc[i,j],Float64(r))
            cD=wj*inn*invr
            cS=wj*sj
            for m in 1:Mk
                dval=cD*αL2[m]*h1vals[m]
                sval=cS*αM2*h0vals[m]
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
    for a in 1:nc,b in 1:nc
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
    _all_k_symm_CFIE_chebyshev!(...)

Assemble symmetry-reduced CFIE-Kress Fredholm matrices for multiple
wavenumbers using Chebyshev-interpolated special functions. 
Constructs the reduced operator by summing symmetry-weighted contributions over
full-space orbit images.

The assembled operator is

    A(k) = I - ( D(k) + i k S(k) ),

projected into the supplied symmetry-adapted basis.
Entries are assembled as

    A_red(a,b) = ∑_ℓ χ_ℓ A_full(i_a,j_ℓ),

where:
- `i_a` is the representative full boundary node for reduced row `a`,
- `{j_ℓ}` are the orbit images of reduced column `b`,
- `χ_ℓ` are the corresponding symmetry weights.

# Returns
- `nothing`
"""
function _all_k_symm_CFIE_chebyshev!(As::Vector{Matrix{ComplexF64}},plans0::Vector{ChebHankelPlanH},plans1::Vector{ChebHankelPlanH},plans2::Vector{ChebJPlan},plans3::Vector{ChebJPlan},h0_tls::Vector{Vector{ComplexF64}},h1_tls::Vector{Vector{ComplexF64}},j0_tls::Vector{Vector{ComplexF64}},j1_tls::Vector{Vector{ComplexF64}},ws::CFIEKressReducedWorkspace{T};multithreaded::Bool=true) where {T<:Real}
    Mk=length(plans0)
    mred=length(ws.Ifund)
    ks=Vector{ComplexF64}(undef,Mk)
    αL1=Vector{ComplexF64}(undef,Mk)
    αL2=Vector{ComplexF64}(undef,Mk)
    iks=Vector{ComplexF64}(undef,Mk)
    @inbounds for q in 1:Mk
        k=ComplexF64(plans1[q].k)
        ks[q]=k
        αL1[q]=-k*_INV_TWO_PI
        αL2[q]=0.5im*k
        iks[q]=1im*k
        fill!(As[q],0.0+0.0im)
    end
    αM1=-_INV_TWO_PI
    αM2=0.5im
    blocks=ws.block_cache.blocks
    Ifund=ws.Ifund
    g2b=ws.global_to_block
    g2l=ws.global_to_local
    orbits=ws.reduced_orbits
    ntls=length(h0_tls)
    acc_tls=[Vector{ComplexF64}(undef,Mk) for _ in 1:ntls]
    # -------------------------------------------------------------------------
    # Reduced symmetry assembly.
    #
    # For each reduced matrix entry (a,b), assemble the symmetry-projected
    # operator entry by summing over the full symmetry orbit of the reduced
    # source degree of freedom:
    #
    #     A_red(a,b) = ∑_ℓ χ_ℓ A_full(i_a, j_ℓ),
    #
    # where:
    #
    #   i_a   = representative full boundary node for reduced row a,
    #   j_ℓ   = orbit images of reduced column b,
    #   χ_ℓ   = symmetry scaling coefficients.
    #
    # Full-space kernel interactions are evaluated directly from the geometric
    # block cache and accumulated into the reduced matrix.
    # -------------------------------------------------------------------------
    @use_threads multithreading=multithreaded for b in 1:mred
        tid=Threads.threadid()
        h0vals=h0_tls[tid]
        h1vals=h1_tls[tid]
        j0vals=j0_tls[tid]
        j1vals=j1_tls[tid]
        acc=acc_tls[tid]
        orb=orbits[b]
        imgs=orb.full
        scales=orb.scales
        img_blocks=orb.blocks
        img_locals=orb.locals
        @inbounds for a in 1:mred
            fill!(acc,0.0+0.0im)
            ig=Ifund[a]
            ib=g2b[ig]
            i=g2l[ig]
            for l in eachindex(imgs)
                scale=ComplexF64(scales[l])
                jb=img_blocks[l]
                j=img_locals[l]
                blk=blocks[ib,jb]
                if blk.same
                    sj=blk.speed_j[j]
                    wj=blk.wj[j]
                    if i==j
                        κj=blk.kappa_i[j]
                        rjj=blk.Rkress[j,j]
                        for q in 1:Mk
                            k=ks[q]
                            dval=ComplexF64(wj*κj,0.0)
                            m1=αM1*sj
                            m2=((0.5im-_EULER_OVER_PI)-_INV_TWO_PI*log((k^2/4)*(sj^2)))*sj
                            sval=ComplexF64(rjj*m1,0.0)+wj*m2
                            acc[q]+=scale*(1.0-(dval+iks[q]*sval))
                        end
                    else
                        r=blk.R[i,j]
                        invr=blk.invR[i,j]
                        lt=blk.logterm[i,j]
                        Rij=blk.Rkress[i,j]
                        inn=blk.inner[i,j]
                        h0_h1_j0_j1_multi_ks_at_r!(h0vals,h1vals,j0vals,j1vals,plans0,plans1,plans2,plans3,blk.pidx[i,j],blk.tloc[i,j],blk.pidxj[i,j],blk.tlocj[i,j],Float64(r))
                        cD1=scale*Rij*inn*invr
                        cD2=scale*wj*inn*invr
                        cD3=cD2*lt
                        cS1=scale*Rij*sj
                        cS2=scale*wj*sj
                        cS3=cS2*lt
                        for q in 1:Mk
                            h0=h0vals[q]
                            h1=h1vals[q]
                            j0=j0vals[q]
                            j1=j1vals[q]
                            L1=αL1[q]*j1
                            M1=αM1*j0
                            M2=αM2*h0
                            dval=cD1*L1+cD2*αL2[q]*h1-cD3*L1
                            sval=cS1*M1+cS2*M2-cS3*M1
                            acc[q]+=-(dval+iks[q]*sval)
                        end
                    end
                else
                    sj=blk.speed_j[j]
                    wj=blk.wj[j]
                    r=blk.R[i,j]
                    invr=blk.invR[i,j]
                    inn=blk.inner[i,j]
                    h0_h1_multi_ks_at_r!(h0vals,h1vals,plans0,plans1,blk.pidx[i,j],blk.tloc[i,j],Float64(r))
                    cD=scale*wj*inn*invr
                    cS=scale*wj*sj
                    for q in 1:Mk
                        dval=cD*αL2[q]*h1vals[q]
                        sval=cS*αM2*h0vals[q]
                        acc[q]+=-(dval+iks[q]*sval)
                    end
                end
            end
            for q in 1:Mk
                As[q][a,b]=acc[q]
            end
        end
    end
    return nothing
end

function _one_k_symm_CFIE_chebyshev!(A::Matrix{ComplexF64},plan0::ChebHankelPlanH,plan1::ChebHankelPlanH,plan2::ChebJPlan,plan3::ChebJPlan,h0_tls::Vector{Vector{ComplexF64}},h1_tls::Vector{Vector{ComplexF64}},j0_tls::Vector{Vector{ComplexF64}},j1_tls::Vector{Vector{ComplexF64}},ws::CFIEKressReducedWorkspace{T};multithreaded::Bool=true) where {T<:Real}
    _all_k_symm_CFIE_chebyshev!([A],[plan0],[plan1],[plan2],[plan3],h0_tls,h1_tls,j0_tls,j1_tls,ws;multithreaded=multithreaded)
    return nothing
end

"""
    _all_k_nosymm_CFIE_chebyshev_deriv!(As,A1s,A2s,pts,plans0,plans1,plans2,plans3,h0_tls,h1_tls,j0_tls,j1_tls,block_cache;multithreaded=true)

Assemble the CFIE-Kress Fredholm matrices and their first two wavenumber
derivatives for all supplied wavenumbers, using Chebyshev-interpolated special
functions and no symmetry reduction.

For each wavenumber `k_m`, this function computes in place:

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
    # -------------------------------------------------------------------------
    # Assemble one source-column `j` of a same-component CFIE block together
    # with its first and second derivatives with respect to the wavenumber.
    #
    # Computes:
    #     A(k), A′(k), A′′(k),
    #
    # for the singular same-component Kress-corrected block.
    #
    # The operator derivatives are
    #
    #     A′(k) = -( D′(k) + i S(k) + i k S′(k) ),
    #
    #     A′′(k) = -( D′′(k) + 2 i S′(k) + i k S′′(k) ).
    #
    # Diagonal entries use exact analytic singular expressions of the
    # finite-part Helmholtz diagonal terms.
    #
    # Off-diagonal entries use analytic expression of the Kress split
    # kernels.
    # -------------------------------------------------------------------------
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
            dval=ComplexF64(wj*κj,0)
            m1=αM1*sj
            m2=((0.5im-_EULER_OVER_PI)-_INV_TWO_PI*log((km^2/4)*(sj^2)))*sj
            sval=ComplexF64(rjj*m1,0)+wj*m2
            sval1=wj*(-sj/(pi*km))
            sval2=wj*(sj/(pi*km^2))
            As[m][gi,gj]=1-(dval+iks[m]*sval)
            A1s[m][gi,gj]=-(1im*sval+iks[m]*sval1)
            A2s[m][gi,gj]=-(2im*sval1+iks[m]*sval2)
        end
        @inbounds for i in (j+1):blk.Ni
            gi=ro+i-1
            r=blk.R[i,j]
            invr=blk.invR[i,j]
            lt=blk.logterm[i,j]
            Rij=blk.Rkress[i,j]
            inn_ij=blk.inner[i,j]
            inn_ji=blk.inner[j,i]
            si=blk.speed_i[i]
            wi=blk.wi[i]
            h0_h1_j0_j1_multi_ks_at_r!(h0vals,h1vals,j0vals,j1vals,plans0,plans1,plans2,plans3,blk.pidx[i,j],blk.tloc[i,j],blk.pidxj[i,j],blk.tlocj[i,j],Float64(r))
            cDRij=Rij*inn_ij*invr
            cDWij=wj*inn_ij*invr
            cDLij=wj*lt*inn_ij*invr
            cDRji=Rij*inn_ji*invr
            cDWji=wi*inn_ji*invr
            cDLji=wi*lt*inn_ji*invr
            cRij=Rij*inn_ij*_INV_TWO_PI
            cWij=wj*inn_ij*_INV_TWO_PI
            cRji=Rij*inn_ji*_INV_TWO_PI
            cWji=wi*inn_ji*_INV_TWO_PI
            cSij=wj*sj
            cSji=wi*si
            cSRij=Rij*sj
            cSRji=Rij*si
            @inbounds for m in 1:Mk
                km=ks[m]
                h0=h0vals[m]
                h1=h1vals[m]
                j0=j0vals[m]
                j1=j1vals[m]
                kr=km*r
                L1=αL1[m]*j1
                M1=αM1*j0
                M2=αM2*h0
                dval_ij=cDRij*L1+cDWij*αL2[m]*h1-cDLij*L1
                dval_ji=cDRji*L1+cDWji*αL2[m]*h1-cDLji*L1
                dval_ij_1=-cRij*km*j0+cWij*km*(lt*j0+1im*pi*h0)
                dval_ji_1=-cRji*km*j0+cWji*km*(lt*j0+1im*pi*h0)
                dval_ij_2=cRij*(kr*j1-j0)+cWij*(lt*(j0-kr*j1)+1im*pi*(h0-kr*h1))
                dval_ji_2=cRji*(kr*j1-j0)+cWji*(lt*(j0-kr*j1)+1im*pi*(h0-kr*h1))
                sval_ij=cSRij*M1+cSij*M2-cSij*lt*M1
                sval_ji=cSRji*M1+cSji*M2-cSji*lt*M1
                sval_ij_1=(r*sj*_INV_TWO_PI)*(Rij*j1-wj*(lt*j1+1im*pi*h1))
                sval_ji_1=(r*si*_INV_TWO_PI)*(Rij*j1-wi*(lt*j1+1im*pi*h1))
                sval_ij_2=(r*sj*_INV_TWO_PI/km)*(Rij*(kr*j0-j1)+wj*(lt*(j1-kr*j0)+1im*pi*(h1-kr*h0)))
                sval_ji_2=(r*si*_INV_TWO_PI/km)*(Rij*(kr*j0-j1)+wi*(lt*(j1-kr*j0)+1im*pi*(h1-kr*h0)))
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
    # -------------------------------------------------------------------------
    # Assemble one source-column `j` of a smooth off-component CFIE block and
    # its first and second wavenumber derivatives.
    #
    # Since source and target belong to distinct boundary components, the
    # kernels are smooth and no singular correction is required.
    #
    # Evaluates: A(k), A′(k), A′′(k),
    #
    # using exact analytic expressions identities for the Helmholtz kernels.
    # -------------------------------------------------------------------------
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
            h0_h1_multi_ks_at_r!(h0vals,h1vals,plans0,plans1,blk.pidx[i,j],blk.tloc[i,j],Float64(r))
            cD=wj*inn*invr
            cD1=wj*inn
            cS=wj*sj
            @inbounds for m in 1:Mk
                km=ks[m]
                h0=h0vals[m]
                h1=h1vals[m]
                dval=cD*αL2[m]*h1
                dval1=-(0.5im)*cD1*km*h0
                dval2=-(0.5im)*cD1*(h0-km*r*h1)
                sval=cS*αM2*h0
                sval1=-(0.5im)*cS*r*h1
                sval2=(0.5im)*cS*r*(h1-km*r*h0)/km
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
    for a in 1:nc,b in 1:nc
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

"""
    _all_k_symm_CFIE_chebyshev_deriv!(...)

Assemble symmetry-reduced CFIE-Kress Fredholm matrices together with their
first and second derivatives with respect to the wavenumber.

For each supplied wavenumber `k`, this function computes:

    A(k),
    A′(k),
    A′′(k),

for the symmetry-reduced CFIE operator.
Reduced entries are assembled via symmetry-orbit summation:

    A_red(a,b) = ∑_ℓ χ_ℓ A_full(i_a,j_ℓ).

# Returns
- `nothing`
"""
function _all_k_symm_CFIE_chebyshev_deriv!(As::Vector{Matrix{ComplexF64}},A1s::Vector{Matrix{ComplexF64}},A2s::Vector{Matrix{ComplexF64}},plans0::Vector{ChebHankelPlanH},plans1::Vector{ChebHankelPlanH},plans2::Vector{ChebJPlan},plans3::Vector{ChebJPlan},h0_tls::Vector{Vector{ComplexF64}},h1_tls::Vector{Vector{ComplexF64}},j0_tls::Vector{Vector{ComplexF64}},j1_tls::Vector{Vector{ComplexF64}},ws::CFIEKressReducedWorkspace{T};multithreaded::Bool=true) where {T<:Real}
    Mk=length(plans0)
    mred=length(ws.Ifund)
    ks=Vector{ComplexF64}(undef,Mk)
    αL1=Vector{ComplexF64}(undef,Mk)
    αL2=Vector{ComplexF64}(undef,Mk)
    iks=Vector{ComplexF64}(undef,Mk)
    @inbounds for q in 1:Mk
        k=ComplexF64(plans1[q].k)
        ks[q]=k
        αL1[q]=-k*_INV_TWO_PI
        αL2[q]=0.5im*k
        iks[q]=1im*k
        fill!(As[q],0.0+0.0im)
        fill!(A1s[q],0.0+0.0im)
        fill!(A2s[q],0.0+0.0im)
    end
    αM1=-_INV_TWO_PI
    αM2=0.5im
    blocks=ws.block_cache.blocks
    Ifund=ws.Ifund
    g2b=ws.global_to_block
    g2l=ws.global_to_local
    orbits=ws.reduced_orbits
    ntls=length(h0_tls)
    acc_tls=[Vector{ComplexF64}(undef,Mk) for _ in 1:ntls]
    acc1_tls=[Vector{ComplexF64}(undef,Mk) for _ in 1:ntls]
    acc2_tls=[Vector{ComplexF64}(undef,Mk) for _ in 1:ntls]
    # -------------------------------------------------------------------------
    # Reduced symmetry derivative assembly.
    #
    # As in the matrix-only reduced assembly, each reduced entry is formed via
    # symmetry-weighted orbit summation:
    #
    #     A_red(a,b)
    #     =
    #     ∑_ℓ χ_ℓ A_full(i_a,j_ℓ).
    #
    # Aassembles also:  A′(k), A′′(k),
    #
    # using analytic expressions of the Helmholtz kernels.
    # -------------------------------------------------------------------------
    @use_threads multithreading=multithreaded for b in 1:mred
        tid=Threads.threadid()
        h0vals=h0_tls[tid]
        h1vals=h1_tls[tid]
        j0vals=j0_tls[tid]
        j1vals=j1_tls[tid]
        acc=acc_tls[tid]
        acc1=acc1_tls[tid]
        acc2=acc2_tls[tid]
        orb=orbits[b]
        imgs=orb.full
        scales=orb.scales
        img_blocks=orb.blocks
        img_locals=orb.locals
        @inbounds for a in 1:mred
            fill!(acc,0.0+0.0im)
            fill!(acc1,0.0+0.0im)
            fill!(acc2,0.0+0.0im)
            ig=Ifund[a]
            ib=g2b[ig]
            i=g2l[ig]
            for l in eachindex(imgs)
                scale=ComplexF64(scales[l])
                jb=img_blocks[l]
                j=img_locals[l]
                blk=blocks[ib,jb]
                if blk.same
                    sj=blk.speed_j[j]
                    wj=blk.wj[j]
                    if i==j
                        κj=blk.kappa_i[j]
                        rjj=blk.Rkress[j,j]
                        for q in 1:Mk
                            k=ks[q]
                            dval=ComplexF64(wj*κj,0.0)
                            m1=αM1*sj
                            m2=((0.5im-_EULER_OVER_PI)-_INV_TWO_PI*log((k^2/4)*(sj^2)))*sj
                            sval=ComplexF64(rjj*m1,0.0)+wj*m2
                            sval1=wj*(-sj/(pi*k))
                            sval2=wj*(sj/(pi*k^2))
                            acc[q]+=scale*(1.0-(dval+iks[q]*sval))
                            acc1[q]+=scale*(-(1im*sval+iks[q]*sval1))
                            acc2[q]+=scale*(-(2im*sval1+iks[q]*sval2))
                        end
                    else
                        r=blk.R[i,j]
                        invr=blk.invR[i,j]
                        lt=blk.logterm[i,j]
                        Rij=blk.Rkress[i,j]
                        inn=blk.inner[i,j]
                        h0_h1_j0_j1_multi_ks_at_r!(h0vals,h1vals,j0vals,j1vals,plans0,plans1,plans2,plans3,blk.pidx[i,j],blk.tloc[i,j],blk.pidxj[i,j],blk.tlocj[i,j],Float64(r))
                        cDR=scale*Rij*inn*invr
                        cDW=scale*wj*inn*invr
                        cDL=scale*wj*lt*inn*invr
                        cR=scale*Rij*inn*_INV_TWO_PI
                        cW=scale*wj*inn*_INV_TWO_PI
                        cS=scale*wj*sj
                        cSR=scale*Rij*sj
                        for q in 1:Mk
                            k=ks[q]
                            h0=h0vals[q]
                            h1=h1vals[q]
                            j0=j0vals[q]
                            j1=j1vals[q]
                            kr=k*r
                            L1=αL1[q]*j1
                            M1=αM1*j0
                            M2=αM2*h0
                            dval=cDR*L1+cDW*αL2[q]*h1-cDL*L1
                            dval1=-cR*k*j0+cW*k*(lt*j0+1im*pi*h0)
                            dval2=cR*(kr*j1-j0)+cW*(lt*(j0-kr*j1)+1im*pi*(h0-kr*h1))
                            sval=cSR*M1+cS*M2-cS*lt*M1
                            sval1=(r*scale*sj*_INV_TWO_PI)*(Rij*j1-wj*(lt*j1+1im*pi*h1))
                            sval2=(r*scale*sj*_INV_TWO_PI/k)*(Rij*(kr*j0-j1)+wj*(lt*(j1-kr*j0)+1im*pi*(h1-kr*h0)))
                            acc[q]+=-(dval+iks[q]*sval)
                            acc1[q]+=-(dval1+1im*sval+iks[q]*sval1)
                            acc2[q]+=-(dval2+2im*sval1+iks[q]*sval2)
                        end
                    end
                else
                    sj=blk.speed_j[j]
                    wj=blk.wj[j]
                    r=blk.R[i,j]
                    invr=blk.invR[i,j]
                    inn=blk.inner[i,j]
                    h0_h1_multi_ks_at_r!(h0vals,h1vals,plans0,plans1,blk.pidx[i,j],blk.tloc[i,j],Float64(r))
                    cD=scale*wj*inn*invr
                    cD1=scale*wj*inn
                    cS=scale*wj*sj
                    for q in 1:Mk
                        k=ks[q]
                        h0=h0vals[q]
                        h1=h1vals[q]
                        dval=cD*αL2[q]*h1
                        dval1=-(0.5im)*cD1*k*h0
                        dval2=-(0.5im)*cD1*(h0-k*r*h1)
                        sval=cS*αM2*h0
                        sval1=-(0.5im)*cS*r*h1
                        sval2=(0.5im)*cS*r*(h1-k*r*h0)/k
                        acc[q]+=-(dval+iks[q]*sval)
                        acc1[q]+=-(dval1+1im*sval+iks[q]*sval1)
                        acc2[q]+=-(dval2+2im*sval1+iks[q]*sval2)
                    end
                end
            end
            for q in 1:Mk
                As[q][a,b]=acc[q]
                A1s[q][a,b]=acc1[q]
                A2s[q][a,b]=acc2[q]
            end
        end
    end
    return nothing
end

function _one_k_symm_CFIE_chebyshev_deriv!(A::Matrix{ComplexF64},A1::Matrix{ComplexF64},A2::Matrix{ComplexF64},plan0::ChebHankelPlanH,plan1::ChebHankelPlanH,plan2::ChebJPlan,plan3::ChebJPlan,h0_tls::Vector{Vector{ComplexF64}},h1_tls::Vector{Vector{ComplexF64}},j0_tls::Vector{Vector{ComplexF64}},j1_tls::Vector{Vector{ComplexF64}},ws::CFIEKressReducedWorkspace{T};multithreaded::Bool=true) where {T<:Real}
    _all_k_symm_CFIE_chebyshev_deriv!([A],[A1],[A2],[plan0],[plan1],[plan2],[plan3],h0_tls,h1_tls,j0_tls,j1_tls,ws;multithreaded=multithreaded)
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
    construct_boundary_matrices!(Tbufs::Vector{Matrix{Complex{T}}},solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},pts::Vector{BoundaryPointsCFIE{T}},zj::AbstractVector{Complex{T}};multithreaded::Bool=true,use_chebyshev::Bool=true,n_panels_h::Int=15000,M_h::Int=5,n_panels_j::Int=3000,M_j::Int=5,timeit::Bool=false) where {T<:Real}

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
- `n_panels_h::Int=15000`:
  Number of distance panels used in the Chebyshev interpolation plans for Hankel functions.
- `M_h::Int=5`:
  Chebyshev interpolation order per panel for Hankel functions.
- `n_panels_j::Int=3000`:
  Number of distance panels used in the Chebyshev interpolation plans for Bessel functions.
- `M_j::Int=5`:
  Chebyshev interpolation order per panel for Bessel functions.
- `timeit::Bool=false`:
  If `true`, enables timing instrumentation via `@benchit`.

# Returns
- `nothing`
"""
function construct_boundary_matrices!(Tbufs::Vector{Matrix{Complex{T}}},solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},pts::Vector{BoundaryPointsCFIE{T}},zj::AbstractVector{Complex{T}};multithreaded::Bool=true,use_chebyshev::Bool=true,n_panels_h::Int=15000,M_h::Int=5,n_panels_j::Int=3000,M_j::Int=5,timeit::Bool=false) where {T<:Real}
    Mk=length(zj)
    @assert length(Tbufs)==Mk
    if use_chebyshev
        @blas_1 begin
            @benchit timeit=timeit "CFIE_kress Block Caches" block_cache=build_cfie_kress_block_caches(solver,pts;npanels_h=n_panels_h,M_h=M_h,npanels_j=n_panels_j,M_j=M_j)
            @benchit timeit=timeit "CFIE_kress Plans" plans0,plans1,plans2,plans3=build_CFIE_plans_kress(zj,block_cache.rmin,block_cache.rmax;npanels_h=n_panels_h,M_h=M_h,npanels_j=n_panels_j,M_j=M_j,nthreads=Threads.nthreads())
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

"""
    construct_boundary_matrices_with_derivatives!(Tbufs::Vector{Matrix{Complex{T}}},dTbufs::Vector{Matrix{Complex{T}}},ddTbufs::Vector{Matrix{Complex{T}}},solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},pts::Vector{BoundaryPointsCFIE{T}},zj::AbstractVector{Complex{T}};multithreaded::Bool=true,use_chebyshev::Bool=true,n_panels_h::Int=15000,M_h::Int=5,n_panels_j::Int=3000,M_j::Int=5,timeit::Bool=false) where {T<:Real}

Assemble CFIE-Kress boundary matrices and their first two derivatives with
respect to the wavenumber for a collection of complex wavenumbers, writing the
results in place.

For each stored wavenumber `zj[m]`, this function fills:
- `Tbufs[m]`   with the Fredholm matrix `A(zj[m])`,
- `dTbufs[m]`  with the first derivative `dA/dk`,
- `ddTbufs[m]` with the second derivative `d²A/dk²`.

When `use_chebyshev=true`, the construction uses the Chebyshev-accelerated
CFIE-Kress pathway based on interpolation plans for:
- `H₀^(1)(k r)`
- `H₁^(1)(k r)`
- `J₀(k r)`
- `J₁(k r)`

# Arguments
- `Tbufs::Vector{Matrix{Complex{T}}}`:
  Output matrices for `A(k)`, one per wavenumber.
- `dTbufs::Vector{Matrix{Complex{T}}}`:
  Output matrices for `dA/dk`, one per wavenumber.
- `ddTbufs::Vector{Matrix{Complex{T}}}`:
  Output matrices for `d²A/dk²`, one per wavenumber.
- `solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners}`:
  CFIE-Kress solver specifying whether the geometry is smooth, cornered, or
  globally corner-graded.
- `pts::Vector{BoundaryPointsCFIE{T}}`:
  Boundary discretization for all connected components.
- `zj::AbstractVector{Complex{T}}`:
  Wavenumbers at which the matrices and derivatives are assembled.

# Keyword Arguments
- `multithreaded::Bool=true`:
  Enables threaded assembly when beneficial.
- `use_chebyshev::Bool=true`:
  If `true`, uses Chebyshev interpolation for the Hankel and Bessel special
  functions. Currently this is the only supported path for complex
  wavenumbers.
- `n_panels_h::Int=15000`:
  Number of distance panels used in the Chebyshev interpolation plans for Hankel functions.
- `M_h::Int=5`:
  Chebyshev interpolation order per panel for Hankel functions.
- `n_panels_j::Int=3000`:
  Number of distance panels used in the Chebyshev interpolation plans for Bessel functions.
- `M_j::Int=5`:
  Chebyshev interpolation order per panel for Bessel functions.
- `timeit::Bool=false`:
  If `true`, enables timing instrumentation via `@benchit`.

# Returns
- `nothing`
"""
function construct_boundary_matrices_with_derivatives!(Tbufs::Vector{Matrix{Complex{T}}},dTbufs::Vector{Matrix{Complex{T}}},ddTbufs::Vector{Matrix{Complex{T}}},solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},pts::Vector{BoundaryPointsCFIE{T}},zj::AbstractVector{Complex{T}};multithreaded::Bool=true,use_chebyshev::Bool=true,n_panels_h::Int=15000,M_h::Int=5,n_panels_j::Int=3000,M_j::Int=5,timeit::Bool=false) where {T<:Real}
    Mk=length(zj)
    @assert length(Tbufs)==Mk
    @assert length(dTbufs)==Mk
    @assert length(ddTbufs)==Mk
    if use_chebyshev
        @blas_1 begin
            @benchit timeit=timeit "CFIE_kress Block Caches" block_cache=build_cfie_kress_block_caches(solver,pts;npanels_h=n_panels_h,M_h=M_h,npanels_j=n_panels_j,M_j=M_j)
            @benchit timeit=timeit "CFIE_kress Plans" plans0,plans1,plans2,plans3=build_CFIE_plans_kress(zj,block_cache.rmin,block_cache.rmax;npanels_h=n_panels_h,M_h=M_h,npanels_j=n_panels_j,M_j=M_j,nthreads=Threads.nthreads())
            @benchit timeit=timeit "CFIE_kress Workspace" ws=CFIE_H0_H1_J0_J1_BesselWorkspace(Mk;ntls=Threads.nthreads())
            @inbounds for j in eachindex(Tbufs)
                fill!(Tbufs[j],0.0+0.0im)
                fill!(dTbufs[j],0.0+0.0im)
                fill!(ddTbufs[j],0.0+0.0im)
            end
            @benchit timeit=timeit "CFIE_kress Derivatives Chebyshev" compute_kernel_matrices_CFIE_kress_chebyshev!(Tbufs,dTbufs,ddTbufs,pts,plans0,plans1,plans2,plans3,ws.h0_tls,ws.h1_tls,ws.j0_tls,ws.j1_tls,block_cache;multithreaded=multithreaded)
        end
    else
        @error("Direct derivative matrix construction is only for real k currently")
    end
    return nothing
end