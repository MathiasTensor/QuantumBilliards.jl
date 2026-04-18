_TWO_PI=2*pi
_INV_TWO_PI=1/_TWO_PI
_EULER_OVER_PI=MathConstants.eulergamma/pi

"""
    DLP_kress_BlockCache{T}

Geometry-and-interpolation cache for Chebyshev-accelerated DLP-Kress assembly
on a single boundary component.

This cache stores all `k`-independent data needed to assemble the Kress-corrected
double-layer matrix using Chebyshev-interpolated special functions. Since the
DLP-Kress solver acts on exactly one closed boundary component, only one block is
needed, unlike the multi-component CFIE case.

The cache combines:
- pairwise geometric data,
- Kress logarithmic-split data,
- and Chebyshev panel lookup metadata.

For a fixed node pair `(i,j)`, the matrix assembly only needs:
- the distance `r = |x_i-x_j|`,
- the inverse distance `1/r`,
- the oriented DLP numerator,
- the quadrature weights,
- the Kress logarithmic term and correction matrix,
- and the Chebyshev panel/local coordinate for evaluating the special functions.

# Fields
- `N::Int`:
  Matrix size, equal to the number of boundary nodes.
- `R::Matrix{T}`:
  Pairwise distance matrix.
- `invR::Matrix{T}`:
  Pairwise inverse-distance matrix, with safe diagonal handling.
- `inner::Matrix{T}`:
  Oriented DLP numerator matrix.
- `wi::Vector{T}`:
  Quadrature weights.
- `pidx::Matrix{Int32}`:
  Chebyshev panel index for each pair `(i,j)`.
- `tloc::Matrix{Float64}`:
  Local Chebyshev coordinate in `[-1,1]` for each pair `(i,j)`.
- `logterm::Matrix{T}`:
  Kress logarithmic split term.
- `kappa::Vector{T}`:
  Diagonal curvature-limit contribution.
- `Rkress::Matrix{T}`:
  Kress correction matrix for the logarithmic singular part.
"""
struct DLP_kress_BlockCache{T<:Real}
    N::Int
    R::Matrix{T}
    invR::Matrix{T}
    inner::Matrix{T}
    wi::Vector{T}
    pidx::Matrix{Int32}
    tloc::Matrix{Float64}
    logterm::Matrix{T}
    kappa::Vector{T}
    Rkress::Matrix{T}
end

"""
    DLPKressBlockSystemCache{T}

Global geometry/interpolation cache for Chebyshev-based DLP-Kress assembly.

Since DLP-Kress acts on a single connected boundary component, the full cache
consists of one `DLP_kress_BlockCache` together with the global distance interval
required by the Chebyshev interpolation plans.

This object is the geometry-side companion to the Chebyshev Hankel/Bessel plans:
- `block` stores all `k`-independent geometry and Kress-split data,
- `rmin` and `rmax` define the distance interval on which the plans are valid.

# Fields
- `block::DLP_kress_BlockCache{T}`:
  Single-component Kress block cache.
- `rmin::Float64`:
  Minimum interpolation distance after safety padding.
- `rmax::Float64`:
  Maximum interpolation distance after safety padding.
"""
struct DLPKressBlockSystemCache{T<:Real}
    block::DLP_kress_BlockCache{T}
    rmin::Float64
    rmax::Float64
end

"""
    build_dlp_kress_block_cache(solver,pts;npanels=10000,M=5,grading=:uniform,geo_ratio=1.05,pad=(0.95,1.05))

Build the Chebyshev geometry/interpolation cache for DLP-Kress assembly on a
fixed boundary discretization.

# Arguments
- `solver::Union{DLP_kress,DLP_kress_global_corners}`:
  Determines whether the smooth or corner-graded Kress correction is used.
- `pts::BoundaryPointsCFIE{T}`:
  Boundary discretization for the single outer boundary.
- `npanels, M, grading, geo_ratio`:
  Parameters used to build a reference Chebyshev panelization over the distance
  interval.
- `pad`:
  Multiplicative safety padding applied to the minimum and maximum off-diagonal
  distances before constructing the interpolation interval.

# Returns
- `DLPKressBlockSystemCache{T}`
"""
function build_dlp_kress_block_cache(solver::Union{DLP_kress,DLP_kress_global_corners},pts::BoundaryPointsCFIE{T};npanels::Int=10000,M::Int=5,grading::Symbol=:uniform,geo_ratio::Real=1.05,pad=(T(0.95),T(1.05)),rmin_cheb::Union{Nothing,Float64}=nothing) where {T<:Real}
    G=_is_dlp_kress_graded(solver) ? cfie_geom_cache(pts,true) : cfie_geom_cache(pts,false)
    N=length(pts.xy)
    R=copy(G.R)
    invR=copy(G.invR)
    inner=copy(G.inner)
    wi=copy(pts.ws)
    logterm=copy(G.logterm)
    kappa=copy(G.kappa)
    Rkress=zeros(T,N,N)
    if solver isa DLP_kress
        kress_R!(Rkress)
    else
        kress_R_corner!(Rkress)
    end
    rmin0=typemax(T)
    rmax0=zero(T)
    @inbounds for j in 1:N,i in 1:N
        i==j && continue
        rij=R[i,j]
        if isfinite(rij) && rij>eps(T)
            rij<rmin0 && (rmin0=rij)
            rij>rmax0 && (rmax0=rij)
        end
    end
    @assert isfinite(rmin0) && rmax0>zero(T)
    rrmin=Float64(pad[1]*rmin0)
    rrmax=Float64(pad[2]*rmax0)
    rmin_cheb_loc=isnothing(rmin_cheb) ? rrmin : max(Float64(rmin_cheb),rrmin)
    pidx=Matrix{Int32}(undef,N,N)
    tloc=Matrix{Float64}(undef,N,N)
    pref_plan=plan_h(0,1,1.0+0im,rmin_cheb_loc,rrmax;npanels=npanels,M=M,grading=grading,geo_ratio=geo_ratio)
    pans=pref_plan.panels
    @inbounds for j in 1:N,i in 1:N
        if i==j
            pidx[i,j]=Int32(1)
            tloc[i,j]=0.0
        else
            rij=Float64(R[i,j])
            if rij<rmin_cheb_loc
                pidx[i,j]=Int32(0)
                tloc[i,j]=0.0
            else
                p=_find_panel(pref_plan,rij)
                P=pans[p]
                pidx[i,j]=Int32(p)
                tloc[i,j]=(2*rij-(P.b+P.a))/(P.b-P.a)
            end
        end
    end
    blk=DLP_kress_BlockCache{T}(N,R,invR,inner,wi,pidx,tloc,logterm,kappa,Rkress)
    return DLPKressBlockSystemCache{T}(blk,rmin_cheb_loc,rrmax)
end

"""
    build_DLP_kress_plans(ks,rmin,rmax;npanels=10000,M=5,grading=:uniform,geo_ratio=1.05,nthreads=1)

Build Chebyshev interpolation plans for the special functions required by the
DLP-Kress assembly over a collection of wavenumbers.

For each `k`, this constructs plans for:
- `H₀^(1)(k r)`
- `H₁^(1)(k r)`
- `J₀(k r)`
- `J₁(k r)`

Unlike the Alpert case, the Kress logarithmic split genuinely requires the
Bessel `J` functions in addition to the Hankel functions. In particular, for
complex wavenumbers one must not replace `J₀`/`J₁` by `real(H₀^(1))` or
`real(H₁^(1))`, so separate Chebyshev plans are built.

# Arguments
- `ks`:
  Wavenumbers for which the DLP-Kress operator will be assembled.
- `rmin, rmax`:
  Distance interval on which the Chebyshev interpolation must be valid.
- `npanels, M, grading, geo_ratio`:
  Parameters passed to the Chebyshev plan builders.
- `nthreads::Int=1`:
  Number of threads used when building the plans.

# Returns
- `(plans0,plans1,plansj0,plansj1)`:
  Chebyshev plans for `H₀^(1)`, `H₁^(1)`, `J₀`, and `J₁`.
"""
function build_DLP_kress_plans(ks::AbstractVector{<:Number},rmin::Float64,rmax::Float64;npanels::Int=10000,M::Int=5,grading::Symbol=:uniform,geo_ratio::Real=1.05,nthreads::Int=1)
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
    DLP_H0_H1_J0_J1_BesselWorkspace

Thread-local scratch workspace for Chebyshev-based DLP-Kress special-function
evaluation over multiple wavenumbers.

This workspace stores reusable temporary arrays for:
- `H₀^(1)(k r)`
- `H₁^(1)(k r)`
- `J₀(k r)`
- `J₁(k r)`

evaluated at one geometric distance `r` across a collection of wavenumbers.

The Kress logarithmic split genuinely requires both Hankel and Bessel `J`
functions for complex k, so all four temporary vectors are stored explicitly.

# Fields
- `h0_tls`:
  Thread-local storage for interpolated `H₀^(1)` values.
- `h1_tls`:
  Thread-local storage for interpolated `H₁^(1)` values.
- `j0_tls`:
  Thread-local storage for interpolated `J₀` values.
- `j1_tls`:
  Thread-local storage for interpolated `J₁` values.
"""
struct DLP_H0_H1_J0_J1_BesselWorkspace
    h0_tls::Vector{Vector{ComplexF64}}
    h1_tls::Vector{Vector{ComplexF64}}
    j0_tls::Vector{Vector{ComplexF64}}
    j1_tls::Vector{Vector{ComplexF64}}
end

function DLP_H0_H1_J0_J1_BesselWorkspace(Mk::Int;ntls::Int=Threads.nthreads())
    h0_tls=[Vector{ComplexF64}(undef,Mk) for _ in 1:ntls]
    h1_tls=[Vector{ComplexF64}(undef,Mk) for _ in 1:ntls]
    j0_tls=[Vector{ComplexF64}(undef,Mk) for _ in 1:ntls]
    j1_tls=[Vector{ComplexF64}(undef,Mk) for _ in 1:ntls]
    return DLP_H0_H1_J0_J1_BesselWorkspace(h0_tls,h1_tls,j0_tls,j1_tls)
end

"""
    DLPKressChebWorkspace{T,MatT}

Reusable workspace for Chebyshev-accelerated DLP-Kress assembly on a fixed
boundary discretization and a fixed set of wavenumbers.

# Fields
- `direct::DLPKressWorkspace{T,MatT}`:
  Original direct DLP-Kress workspace containing geometry and Kress data used by
  the non-Chebyshev assembly route.
- `block_cache::DLPKressBlockSystemCache{T}`:
  Chebyshev-side geometry/interpolation cache.
- `plans0`, `plans1`:
  Chebyshev plans for `H₀^(1)` and `H₁^(1)`.
- `plansj0`, `plansj1`:
  Chebyshev plans for `J₀` and `J₁`.
- `bessel_ws::DLP_H0_H1_J0_J1_BesselWorkspace`:
  Thread-local temporary storage for interpolated special-function values.
- `ks::Vector{ComplexF64}`:
  Wavenumbers associated with the plans in this workspace.
- `Mk::Int`:
  Number of wavenumbers.
"""
struct DLPKressChebWorkspace{T<:Real,MatT<:AbstractMatrix{T}}
    direct::DLPKressWorkspace{T,MatT}
    block_cache::DLPKressBlockSystemCache{T}
    plans0::Vector{ChebHankelPlanH}
    plans1::Vector{ChebHankelPlanH}
    plansj0::Vector{ChebJPlan}
    plansj1::Vector{ChebJPlan}
    bessel_ws::DLP_H0_H1_J0_J1_BesselWorkspace
    ks::Vector{ComplexF64}
    Mk::Int
end

"""
    build_dlp_kress_cheb_workspace(solver,pts,direct,ks;npanels=10000,M=5,grading=:uniform,geo_ratio=1.05,pad=(0.95,1.05),plan_nthreads=1,ntls=Threads.nthreads())

Build the reusable Chebyshev workspace for DLP-Kress assembly on a fixed
boundary discretization and a fixed set of wavenumbers.

This combines:
- the direct DLP-Kress workspace,
- a Kress-aware geometry/interpolation cache,
- Chebyshev plans for `H₀^(1)`, `H₁^(1)`, `J₀`, and `J₁`,
- thread-local temporary buffers for multi-`k` assembly.
# Arguments
- `solver::Union{DLP_kress{T},DLP_kress_global_corners{T}}`:
  Determines whether the smooth or corner-graded Kress correction is used.
- `pts::BoundaryPointsCFIE{T}`:
  Boundary discretization for the single outer boundary.
- `direct::DLPKressWorkspace{T,MatT}`:
  Prebuilt direct DLP-Kress workspace for the same geometry, used to extract the original geometry and Kress data for reuse in the Chebyshev assembly.
- `ks::Vector{ComplexF64}`:
  Wavenumbers for which the DLP-Kress operator will be assembled, and for which the Chebyshev plans will be built.
- `npanels, M, grading, geo_ratio`:
  Parameters passed to the Chebyshev plan builders.
- `pad`:
  Multiplicative safety padding applied to the minimum and maximum off-diagonal distances before constructing the interpolation interval.
- `plan_nthreads::Int=1`:
  Number of threads used when building the Chebyshev plans.
- `ntls::Int=Threads.nthreads()`:
  Number of thread-local buffers allocated for the Bessel/Hankel workspace, which should ideally match the number of threads used during assembly to avoid contention.

# Returns
- `DLPKressChebWorkspace{T,MatT}`
"""
function build_dlp_kress_cheb_workspace(solver::Union{DLP_kress{T},DLP_kress_global_corners{T}},pts::BoundaryPointsCFIE{T},direct::DLPKressWorkspace{T,MatT},ks::Vector{ComplexF64};npanels::Int=10000,M::Int=5,grading::Symbol=:uniform,geo_ratio::Real=1.05,pad=(T(0.95),T(1.05)),plan_nthreads::Int=1,ntls::Int=Threads.nthreads()) where {T<:Real,MatT<:AbstractMatrix{T}}
    block_cache=build_dlp_kress_block_cache(solver,pts;npanels=npanels,M=M,grading=grading,geo_ratio=geo_ratio,pad=pad)
    plans0,plans1,plansj0,plansj1=build_DLP_kress_plans(ks,block_cache.rmin,block_cache.rmax;npanels=npanels,M=M,grading=grading,geo_ratio=geo_ratio,nthreads=plan_nthreads)
    bessel_ws=DLP_H0_H1_J0_J1_BesselWorkspace(length(ks);ntls=ntls)
    return DLPKressChebWorkspace{T,MatT}(direct,block_cache,plans0,plans1,plansj0,plansj1,bessel_ws,ks,length(ks))
end

"""
    _h0_h1_j0_j1_at_pidx_t!(h0vals,h1vals,j0vals,j1vals,pidx,t,r,plans0,plans1,plansj0,plansj1)

Evaluate `H₀^(1)`, `H₁^(1)`, `J₀`, and `J₁` for all wavenumbers at one fixed
Chebyshev panel/location, writing the results in place.

This is the small inner evaluator used by the multi-`k` DLP-Kress assembly.
The geometric pair `(i,j)` has already been mapped to:
- a panel index `pidx`,
- a local coordinate `t ∈ [-1,1]`,
- and a physical distance `r`.

For each stored wavenumber `k_m`, the routine interpolates:
- `H₀^(1)(k_m r)`
- `H₁^(1)(k_m r)`
- `J₀(k_m r)`
- `J₁(k_m r)`

If pidx is zero, the distance is below the interpolation cutoff and direct evaluation is used instead to avoid accuracy issues and having too many panels to resolve the small z=k*r region.

# Returns
- `nothing`
"""
@inline function _h0_h1_j0_j1_at_pidx_t!(h0vals::AbstractVector{ComplexF64},h1vals::AbstractVector{ComplexF64},j0vals::AbstractVector{ComplexF64},j1vals::AbstractVector{ComplexF64},pidx::Int32,t::Float64,r::Float64,plans0::AbstractVector{ChebHankelPlanH},plans1::AbstractVector{ChebHankelPlanH},plansj0::AbstractVector{ChebJPlan},plansj1::AbstractVector{ChebJPlan})
    h0_h1_j0_j1_multi_ks_at_r!(h0vals,h1vals,j0vals,j1vals,plans0,plans1,plansj0,plansj1,pidx,t,r)
    return nothing
end

"""
    _h1_j1_at_pidx_t!(h1vals,j1vals,pidx,t,r,plans1,plansj1)

Evaluate `H₁^(1)` and `J₁` for all wavenumbers at one fixed Chebyshev
panel/location, writing the results in place. 

If pidx is zero, the distance is below the interpolation cutoff and direct evaluation is used instead to avoid accuracy issues and having too many panels to resolve the small z=k*r region.

# Returns
- `nothing`
"""
@inline function _h1_j1_at_pidx_t!(h1vals::AbstractVector{ComplexF64},j1vals::AbstractVector{ComplexF64},pidx::Int32,t::Float64,r::Float64,plans1::AbstractVector{ChebHankelPlanH},plansj1::AbstractVector{ChebJPlan})
    h1_j1_multi_ks_at_r!(h1vals,j1vals,plans1,plansj1,pidx,t,r)
    return nothing
end

"""
    construct_dlp_kress_matrices_chebyshev!(Ds,pts,ws;multithreaded=true)

Assemble the Kress-corrected DLP matrices for all wavenumbers stored in a
prebuilt Chebyshev workspace, writing the results in place.
This is the multi-`k` value-only Chebyshev assembly routine for the raw
double-layer operator `D(k)`. One dense matrix is filled for each wavenumber in
`ws.ks`.

# Arguments
- `Ds::Vector{<:AbstractMatrix{ComplexF64}}`:
  Output matrices, one for each wavenumber in `ws.ks`.
- `pts::BoundaryPointsCFIE{T}`:
  Boundary discretization. Included for API consistency with the direct route.
- `ws::DLPKressChebWorkspace{T}`:
  Prebuilt Chebyshev workspace.
- `multithreaded::Bool=true`:
  Enables threaded off-diagonal assembly when beneficial.

# Returns
- `nothing`
"""
function _construct_dlp_kress_matrices_chebyshev!(Ds::Vector{<:AbstractMatrix{ComplexF64}},pts::BoundaryPointsCFIE{T},ws::DLPKressChebWorkspace{T};multithreaded::Bool=true) where {T<:Real}
    Mk=ws.Mk
    blk=ws.block_cache.block
    N=blk.N
    @inbounds for m in 1:Mk
        fill!(Ds[m],0.0+0im)
        for i in 1:N
            Ds[m][i,i]=ComplexF64(blk.wi[i]*blk.kappa[i],0.0)
        end
    end
    h1_tls=ws.bessel_ws.h1_tls
    j1_tls=ws.bessel_ws.j1_tls
    ks=ws.ks
    @use_threads multithreading=(multithreaded && N>=32) for j in 2:N
        tid=Threads.threadid()
        h1vals=h1_tls[tid]
        j1vals=j1_tls[tid]
        @inbounds for i in 1:j-1
            r=blk.R[i,j]
            _h1_j1_at_pidx_t!(h1vals,j1vals,blk.pidx[i,j],blk.tloc[i,j],r,ws.plans1,ws.plansj1)
            invr=blk.invR[i,j]
            lt=blk.logterm[i,j]
            inn_ij=blk.inner[i,j]
            inn_ji=blk.inner[j,i]
            Rij=blk.Rkress[i,j]
            wj=blk.wi[j]
            wi=blk.wi[i]
            for m in 1:Mk
                k=ks[m]
                αL1=-k*_INV_TWO_PI
                αL2=0.5im*k
                h1=h1vals[m]
                j1=j1vals[m]
                l1_ij=αL1*inn_ij*j1*invr
                l2_ij=αL2*inn_ij*h1*invr-l1_ij*lt
                Ds[m][i,j]=Rij*l1_ij+wj*l2_ij
                l1_ji=αL1*inn_ji*j1*invr
                l2_ji=αL2*inn_ji*h1*invr-l1_ji*lt
                Ds[m][j,i]=Rij*l1_ji+wi*l2_ji
            end
        end
    end
    return nothing
end

"""
    construct_dlp_kress_matrices_derivatives_chebyshev!(Ds,D1s,D2s,pts,ws;multithreaded=true)

Assemble the Kress-corrected DLP matrices and their first two derivatives with
respect to the wavenumber for all wavenumbers stored in a prebuilt Chebyshev
workspace, writing the results in place.

For each stored wavenumber `k_m`, this routine computes:
- `D(k_m)`
- `D₁(k_m)=dD/dk`
- `D₂(k_m)=d²D/dk²`

using Chebyshev-interpolated evaluations of `H₀^(1)`, `H₁^(1)`, `J₀`, and
`J₁`.

# Arguments
- `Ds::Vector{<:AbstractMatrix{ComplexF64}}`:
  Output matrices for `D(k)`.
- `D1s::Vector{<:AbstractMatrix{ComplexF64}}`:
  Output matrices for `dD/dk`.
- `D2s::Vector{<:AbstractMatrix{ComplexF64}}`:
  Output matrices for `d²D/dk²`.
- `pts::BoundaryPointsCFIE{T}`:
  Boundary discretization. Included for API consistency.
- `ws::DLPKressChebWorkspace{T}`:
  Prebuilt Chebyshev workspace.
- `multithreaded::Bool=true`:
  Enables threaded off-diagonal assembly when beneficial.

# Returns
- `nothing`
"""
function _construct_dlp_kress_matrices_derivatives_chebyshev!(Ds::Vector{<:AbstractMatrix{ComplexF64}},D1s::Vector{<:AbstractMatrix{ComplexF64}},D2s::Vector{<:AbstractMatrix{ComplexF64}},pts::BoundaryPointsCFIE{T},ws::DLPKressChebWorkspace{T};multithreaded::Bool=true) where {T<:Real}
    Mk=ws.Mk
    blk=ws.block_cache.block
    N=blk.N
    @inbounds for m in 1:Mk
        fill!(Ds[m],0.0+0im)
        fill!(D1s[m],0.0+0im)
        fill!(D2s[m],0.0+0im)
        for i in 1:N
            Ds[m][i,i]=ComplexF64(blk.wi[i]*blk.kappa[i],0.0)
        end
    end
    h0_tls=ws.bessel_ws.h0_tls
    h1_tls=ws.bessel_ws.h1_tls
    j0_tls=ws.bessel_ws.j0_tls
    j1_tls=ws.bessel_ws.j1_tls
    ks=ws.ks
    @use_threads multithreading=(multithreaded && N>=32) for j in 2:N
        tid=Threads.threadid()
        h0vals=h0_tls[tid]
        h1vals=h1_tls[tid]
        j0vals=j0_tls[tid]
        j1vals=j1_tls[tid]
        @inbounds for i in 1:j-1
            r=blk.R[i,j]
            _h0_h1_j0_j1_at_pidx_t!(h0vals,h1vals,j0vals,j1vals,blk.pidx[i,j],blk.tloc[i,j],r,ws.plans0,ws.plans1,ws.plansj0,ws.plansj1)
            invr=blk.invR[i,j]
            lt=blk.logterm[i,j]
            inn_ij=blk.inner[i,j]
            inn_ji=blk.inner[j,i]
            Rij=blk.Rkress[i,j]
            wj=blk.wi[j]
            wi=blk.wi[i]
            for m in 1:Mk
                k=ks[m]
                αL1=-k*_INV_TWO_PI
                αL2=0.5im*k
                h0=h0vals[m]
                h1=h1vals[m]
                j0=j0vals[m]
                j1=j1vals[m]
                l1_ij=αL1*inn_ij*j1*invr
                l2_ij=αL2*inn_ij*h1*invr-l1_ij*lt
                Ds[m][i,j]=Rij*l1_ij+wj*l2_ij
                l1_ij_1=-(inn_ij*k*j0)*_INV_TWO_PI
                l1_ij_2=(inn_ij*(k*r*j1-j0))*_INV_TWO_PI
                l2_ij_1=(inn_ij*k*(lt*j0+im*pi*h0))*_INV_TWO_PI
                l2_ij_2=(inn_ij*(lt*(j0-k*r*j1)+im*pi*(h0-k*r*h1)))*_INV_TWO_PI
                D1s[m][i,j]=Rij*l1_ij_1+wj*l2_ij_1
                D2s[m][i,j]=Rij*l1_ij_2+wj*l2_ij_2
                l1_ji=αL1*inn_ji*j1*invr
                l2_ji=αL2*inn_ji*h1*invr-l1_ji*lt
                Ds[m][j,i]=Rij*l1_ji+wi*l2_ji
                l1_ji_1=-(inn_ji*k*j0)*_INV_TWO_PI
                l1_ji_2=(inn_ji*(k*r*j1-j0))*_INV_TWO_PI
                l2_ji_1=(inn_ji*k*(lt*j0+im*pi*h0))*_INV_TWO_PI
                l2_ji_2=(inn_ji*(lt*(j0-k*r*j1)+im*pi*(h0-k*r*h1)))*_INV_TWO_PI
                D1s[m][j,i]=Rij*l1_ji_1+wi*l2_ji_1
                D2s[m][j,i]=Rij*l1_ji_2+wi*l2_ji_2
            end
        end
    end
    return nothing
end

"""
    construct_dlp_kress_matrices_chebyshev!(Fs,pts,ws;multithreaded=true)
    construct_dlp_kress_matrices_derivatives_chebyshev!(Fs,F1s,F2s,pts,ws;multithreaded=true)

Assemble the Kress Fredholm second-kind matrices and, optionally, their first
two wavenumber derivatives using a prebuilt Chebyshev workspace.
These are the Fredholm counterparts of the raw DLP assembly routines. They first
assemble the Kress-corrected DLP matrices and then convert them in place to

    F(k)=I-D(k),
    F₁(k)=-D₁(k),
    F₂(k)=-D₂(k).

# Arguments
- `Fs::Vector{<:AbstractMatrix{ComplexF64}}`:
  Output matrices for `F(k)`.
- `F1s::Vector{<:AbstractMatrix{ComplexF64}}`:
  Output matrices for `dF/dk` in the derivative-aware form.
- `F2s::Vector{<:AbstractMatrix{ComplexF64}}`:
  Output matrices for `d²F/dk²` in the derivative-aware form.
- `pts::BoundaryPointsCFIE{T}`:
  Boundary discretization. Included for API consistency.
- `ws::DLPKressChebWorkspace{T}`:
  Prebuilt Chebyshev workspace.
- `multithreaded::Bool=true`:
  Enables threaded assembly where beneficial.

# Returns
- `nothing`
"""
function construct_dlp_kress_matrices_chebyshev!(Fs::Vector{<:AbstractMatrix{ComplexF64}},pts::BoundaryPointsCFIE{T},ws::DLPKressChebWorkspace{T};multithreaded::Bool=true) where {T<:Real}
    _construct_dlp_kress_matrices_chebyshev!(Fs,pts,ws;multithreaded=multithreaded)
    @inbounds for m in eachindex(Fs),j in axes(Fs[m],2),i in axes(Fs[m],1)
        Fs[m][i,j]*=-1
    end
    @inbounds for m in eachindex(Fs),i in axes(Fs[m],1)
        Fs[m][i,i]+=1.0+0im
    end
    return nothing
end

function construct_dlp_kress_matrices_derivatives_chebyshev!(Fs::Vector{<:AbstractMatrix{ComplexF64}},F1s::Vector{<:AbstractMatrix{ComplexF64}},F2s::Vector{<:AbstractMatrix{ComplexF64}},pts::BoundaryPointsCFIE{T},ws::DLPKressChebWorkspace{T};multithreaded::Bool=true) where {T<:Real}
    _construct_dlp_kress_matrices_derivatives_chebyshev!(Fs,F1s,F2s,pts,ws;multithreaded=multithreaded)
    @inbounds for m in eachindex(Fs),j in axes(Fs[m],2),i in axes(Fs[m],1)
        Fs[m][i,j]*=-1
        F1s[m][i,j]*=-1
        F2s[m][i,j]*=-1
    end
    @inbounds for m in eachindex(Fs),i in axes(Fs[m],1)
        Fs[m][i,i]+=1.0+0im
    end
    return nothing
end

"""
    construct_boundary_matrices!(Tbufs,solver,pts,zj;multithreaded=true,use_chebyshev=true,n_panels=15000,M=5,timeit=false)

Assemble DLP-Kress Fredholm boundary matrices for a collection of complex
wavenumbers, writing the results in place.

# Arguments
- `Tbufs::Vector{Matrix{ComplexF64}}`:
  Output matrices, one for each wavenumber in `zj`. Each matrix is overwritten.
- `solver::Union{DLP_kress,DLP_kress_global_corners}`:
  DLP-Kress solver describing the smooth or globally corner-graded boundary
  discretization.
- `pts::BoundaryPointsCFIE{T}`:
  Boundary discretization for the single outer boundary component.
- `zj::AbstractVector{ComplexF64}`:
  Wavenumbers at which the Fredholm matrices are assembled.

# Keyword Arguments
- `multithreaded::Bool=true`:
  Enables threaded off-diagonal assembly when beneficial.
- `use_chebyshev::Bool=true`:
  If `true`, uses the Chebyshev-accelerated complex-`k` assembly path.
  Currently this is the only supported route for complex wavenumbers.
- `n_panels::Int=15000`:
  Number of distance panels used in the Chebyshev interpolation plans.
- `M::Int=5`:
  Chebyshev interpolation order per panel.
- `timeit::Bool=false`:
  Enables timing instrumentation through `@benchit`.

# Returns
- `nothing`
"""
function construct_boundary_matrices!(Tbufs::Vector{Matrix{ComplexF64}},solver::Union{DLP_kress,DLP_kress_global_corners},pts::BoundaryPointsCFIE{T},zj::AbstractVector{ComplexF64};multithreaded::Bool=true,use_chebyshev::Bool=true,n_panels::Int=15000,M::Int=5,timeit::Bool=false) where {T<:Real}
    Mk=length(zj)
    @assert length(Tbufs)==Mk
    if use_chebyshev
        @blas_1 begin
            @benchit timeit=timeit "DLP_kress Workspace" directws=build_dlp_kress_workspace(solver,pts)
            @benchit timeit=timeit "DLP_kress Chebyshev Workspace" chebws=build_dlp_kress_cheb_workspace(solver,pts,directws,ComplexF64.(zj);npanels=n_panels,M=M,grading=:uniform,plan_nthreads=Threads.nthreads(),ntls=Threads.nthreads())
            @inbounds for j in eachindex(Tbufs)
                fill!(Tbufs[j],0.0+0.0im)
            end
            @benchit timeit=timeit "DLP_kress Chebyshev" construct_dlp_kress_matrices_chebyshev!(Tbufs,pts,chebws;multithreaded=multithreaded)
        end
    else
        @error("Direct matrix construction is only for real k currently")
    end
    return nothing
end