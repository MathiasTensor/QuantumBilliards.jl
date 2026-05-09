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
  Chebyshev panel index for H Bessel of each pair `(i,j)`.
- `tloc::Matrix{Float64}`:
  Local Chebyshev coordinate in `[-1,1]` for H Bessel of each pair `(i,j)`.
- `pidxj::Matrix{Int32}`:
  Chebyshev panel index for the J Bessel of each pair `(i,j)`.
- `tlocj::Matrix{Float64}`:
  Local Chebyshev coordinate in `[-1,1]` for the J Bessel of each pair `(i,j)`.
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
    pidxj::Matrix{Int32}
    tlocj::Matrix{Float64}
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
    build_dlp_kress_block_cache(solver,pts;npanels=10000,M=5,pad=(0.95,1.05))

Build the Chebyshev geometry/interpolation cache for DLP-Kress assembly on a
fixed boundary discretization.

# Arguments
- `solver::Union{DLP_kress,DLP_kress_global_corners}`:
  Determines whether the smooth or corner-graded Kress correction is used.
- `pts::BoundaryPointsCFIE{T}`:
  Boundary discretization for the single outer boundary.
- `npanels_h::Int=10000, M_h::Int=5`:
  Chebyshev plan parameters for the Hankel function interpolation.
- `npanels_j::Int=10000, M_j::Int=5`:
  Chebyshev plan parameters for the Bessel function interpolation.
- `pad`:
  Multiplicative safety padding applied to the minimum and maximum off-diagonal
  distances before constructing the interpolation interval.

# Returns
- `DLPKressBlockSystemCache{T}`
"""
function build_dlp_kress_block_cache(solver::Union{DLP_kress,DLP_kress_global_corners},pts::BoundaryPointsCFIE{T};npanels_h::Int=10000,npanels_j::Int=10000,M_h::Int=5,M_j::Int=5,pad=(T(0.95),T(1.05)),rmin_cheb::Union{Nothing,Float64}=nothing) where {T<:Real}
    G=_is_dlp_kress_graded(solver,pts) ? cfie_geom_cache(pts,true) : cfie_geom_cache(pts,false)
    N=length(pts.xy)
    R=copy(G.R)
    invR=copy(G.invR)
    inner=copy(G.inner)
    wi=copy(pts.ws)
    logterm=copy(G.logterm)
    kappa=copy(G.kappa)
    Rkress=zeros(T,N,N)
    if _is_dlp_kress_graded(solver,pts)
        kress_R_corner!(Rkress)
    else
        kress_R!(Rkress)
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
    pidx=Matrix{Int32}(undef,N,N) # H Bessel panel index
    tloc=Matrix{Float64}(undef,N,N) # H Bessel local coordinate
    pidxj=Matrix{Int32}(undef,N,N) # J Bessel panel index
    tlocj=Matrix{Float64}(undef,N,N) # J Bessel local coordinate
    pref_plan_h=plan_h(0,1,1.0+0im,rmin_cheb_loc,rrmax;npanels=npanels_h,M=M_h)
    pref_plan_j=plan_j(1,1.0+0im,0.0,rrmax;npanels=npanels_j,M=M_j) 
    pansh=pref_plan_h.panels
    pansj=pref_plan_j.panels
    @inbounds for j in 1:N, i in 1:N
        if i==j
            pidx[i,j]=Int32(1)
            tloc[i,j]=0.0
            pidxj[i,j]=Int32(1)
            tlocj[i,j]=0.0
        else
            rij=Float64(R[i,j])
            if rij<rmin_cheb_loc
                pidx[i,j]=Int32(0)
                tloc[i,j]=0.0
            else
                p=_find_panel(pref_plan_h,rij)
                P=pansh[p]
                pidx[i,j]=Int32(p)
                tloc[i,j]=(2*rij-(P.b+P.a))/(P.b-P.a)
            end
            pj=_find_panel(pref_plan_j,rij)
            Pj=pansj[pj]
            pidxj[i,j]=Int32(pj)
            tlocj[i,j]=(2*rij-(Pj.b+Pj.a))/(Pj.b-Pj.a)
        end

    end
    blk=DLP_kress_BlockCache{T}(N,R,invR,inner,wi,pidx,tloc,pidxj,tlocj,logterm,kappa,Rkress)
    return DLPKressBlockSystemCache{T}(blk,rmin_cheb_loc,rrmax)
end

"""
    build_DLP_kress_plans_h1_j1(ks,rmin,rmax;npanels_h=10000,npanels_j=10000,M_h=5,M_j=5,nthreads=1)

Build Chebyshev interpolation plans for the special functions required by the
DLP-Kress assembly over a collection of wavenumbers.

For each `k`, this constructs plans for:
- `H₁^(1)(k r)`
- `J₁(k r)`

Unlike the Alpert case, the Kress logarithmic split genuinely requires the
Bessel `J` functions in addition to the Hankel functions. In particular, for
complex wavenumbers one must not replace `J₁` by `real(H₁^(1))`, 
so separate Chebyshev plans are built.

# Arguments
- `ks`:
  Wavenumbers for which the DLP-Kress operator will be assembled.
- `rmin, rmax`:
  Distance interval on which the Chebyshev interpolation must be valid.
- `npanels_h, npanels_j, M_h, M_j`:
  Note: the `npanels` and `M` parameters are decoupled for the Hankel and Bessel plans since the Bessel functions 
  require fewer panels/terms for the same accuracy, so separate parameters are provided for each.
  Parameters passed to the Chebyshev plan builders.
- `nthreads::Int=1`:
  Number of threads used when building the plans.

# Returns
- `(plans1,plansj1)`:
  Chebyshev plans for `H₁^(1)` and `J₁`.
"""
function build_DLP_kress_plans_h1_j1(ks::AbstractVector{<:Number},rmin::Float64,rmax::Float64;npanels_h::Int=10000,npanels_j::Int=10000,M_h::Int=5,M_j::Int=5,nthreads::Int=1)
    Mk=length(ks)
    plans1=Vector{ChebHankelPlanH}(undef,Mk)
    plansj1=Vector{ChebJPlan}(undef,Mk)
    if nthreads<=1 || Mk==1
        @inbounds for m in 1:Mk
            k=ComplexF64(ks[m])
            plans1[m]=plan_h(1,1,k,rmin,rmax;npanels=npanels_h,M=M_h)
            plansj1[m]=plan_j(1,k,0.0,rmax;npanels=npanels_j,M=M_j) # rmin=0 since J has no issue there
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
                plans1[m]=plan_h(1,1,k,rmin,rmax;npanels=npanels_h,M=M_h)
                plansj1[m]=plan_j(1,k,0.0,rmax;npanels=npanels_j,M=M_j)  # rmin=0 since J has no issue there
            end
        end
    end
    return plans1,plansj1
end

"""
    build_DLP_kress_plans_h0_h1_j0_j1(ks,rmin,rmax;npanels_h=10000,npanels_j=10000,M_h=5,M_j=5,nthreads=1)

Build Chebyshev interpolation plans for the special functions required by the
DLP-Kress assembly over a collection of wavenumbers. For use in EBIM where the
derivatives require both `H₀^(1)` and `H₁^(1)` as well as `J₀` and `J₁`.

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
- `npanels_h, npanels_j, M_h, M_j`: 
  Note: the `npanels` and `M` parameters are decoupled for the Hankel and Bessel plans since the Bessel functions 
  require fewer panels/terms for the same accuracy, so separate parameters are provided for each.  Parameters passed to the Chebyshev plan builders.
- `nthreads::Int=1`:
  Number of threads used when building the plans.

# Returns
- `(plans0,plans1,plansj0,plansj1)`:
  Chebyshev plans for `H₀^(1)`, `H₁^(1)`, `J₀`, and `J₁`.
"""
function build_DLP_kress_plans_h0_h1_j0_j1(ks::AbstractVector{<:Number},rmin::Float64,rmax::Float64;npanels_h::Int=10000,npanels_j::Int=10000,M_h::Int=5,M_j::Int=5,nthreads::Int=1)
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
            plansj0[m]=plan_j(0,k,0.0,rmax;npanels=npanels_j,M=M_j) # rmin=0 since J has no issue there
            plansj1[m]=plan_j(1,k,0.0,rmax;npanels=npanels_j,M=M_j)  # rmin=0 since J has no issue there
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
                plansj0[m]=plan_j(0,k,0.0,rmax;npanels=npanels_j,M=M_j) # rmin=0 since J has no issue there
                plansj1[m]=plan_j(1,k,0.0,rmax;npanels=npanels_j,M=M_j)  # rmin=0 since J has no issue there
            end
        end
    end
    return plans0,plans1,plansj0,plansj1
end

"""
    DLP_H1_J1_BesselWorkspace

Thread-local scratch workspace for Chebyshev-based DLP-Kress special-function
evaluation over multiple wavenumbers (`Mk` of them). 

This workspace stores reusable temporary arrays for:
- `H₁^(1)(k r)`
- `J₁(k r)`

evaluated at one geometric distance `r` across a collection of wavenumbers.

The Kress logarithmic split genuinely requires both Hankel and Bessel `J`
functions for complex k, so all four temporary vectors are stored explicitly.

# Fields
- `h1_tls`:
  Thread-local storage for interpolated `H₁^(1)` values.
- `j1_tls`:
  Thread-local storage for interpolated `J₁` values.
"""
struct DLP_H1_J1_BesselWorkspace
    h1_tls::Vector{Vector{ComplexF64}}
    j1_tls::Vector{Vector{ComplexF64}}
end

function DLP_H1_J1_BesselWorkspace(Mk::Int;ntls::Int=Threads.nthreads())
    h1_tls=[Vector{ComplexF64}(undef,Mk) for _ in 1:ntls]
    j1_tls=[Vector{ComplexF64}(undef,Mk) for _ in 1:ntls]
    return DLP_H1_J1_BesselWorkspace(h1_tls,j1_tls)
end

"""
    DLP_H0_H1_J0_J1_BesselWorkspace

Thread-local scratch workspace for Chebyshev-based DLP-Kress special-function
evaluation over multiple wavenumbers (`Mk` of them). Used in EBIM since the derivatives
require all four bessel/hankel evaluations.

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
    DLPKressH1J1ChebWorkspace{T,MatT}

Reusable value-only Chebyshev workspace for DLP-Kress assembly on the full
boundary.

This workspace is used when only the Fredholm matrix `F(k)=I-D(k)` is needed.
The Kress logarithmic split for the DLP value requires only

    H₁^(1)(k r),   J₁(k r),

so this workspace stores only the corresponding Chebyshev plans and temporary
thread-local buffers. This avoids building unnecessary `H₀` and `J₀` plans.

# Fields
- `direct::DLPKressWorkspace{T,MatT}`:
  Direct full-boundary DLP-Kress workspace for the same geometry.
- `block_cache::DLPKressBlockSystemCache{T}`:
  Geometry, Kress, and Hankel-panel lookup cache.
- `plans1::Vector{ChebHankelPlanH}`:
  Chebyshev plans for `H₁^(1)(k r)`.
- `plansj1::Vector{ChebJPlan}`:
  Chebyshev plans for `J₁(k r)`.
- `bessel_ws::DLP_H1_J1_BesselWorkspace`:
  Thread-local buffers for `H₁` and `J₁`.
- `ks::Vector{ComplexF64}`:
  Wavenumbers associated with the plans.
- `Mk::Int`:
  Number of wavenumbers.
"""
struct DLPKressH1J1ChebWorkspace{T<:Real,MatT<:AbstractMatrix{T}}
    direct::DLPKressWorkspace{T,MatT}
    block_cache::DLPKressBlockSystemCache{T}
    plans1::Vector{ChebHankelPlanH}
    plansj1::Vector{ChebJPlan}
    bessel_ws::DLP_H1_J1_BesselWorkspace
    ks::Vector{ComplexF64}
    Mk::Int
end

"""
    DLPKressH0H1J0J1ChebWorkspace{T,MatT}

Reusable derivative-aware Chebyshev workspace for DLP-Kress assembly on the full
boundary.

This workspace is used when assembling `D(k)` together with its first two
wavenumber derivatives. The derivative formulas require

    H₀^(1)(k r), H₁^(1)(k r), J₀(k r), J₁(k r),

so all four plan families and thread-local buffers are stored.

# Fields
- `direct::DLPKressWorkspace{T,MatT}`:
  Direct full-boundary DLP-Kress workspace for the same geometry.
- `block_cache::DLPKressBlockSystemCache{T}`:
  Geometry, Kress, and Hankel-panel lookup cache.
- `plans0`, `plans1`:
  Chebyshev plans for `H₀^(1)` and `H₁^(1)`.
- `plansj0`, `plansj1`:
  Chebyshev plans for `J₀` and `J₁`.
- `bessel_ws::DLP_H0_H1_J0_J1_BesselWorkspace`:
  Thread-local buffers for all four special functions.
- `ks::Vector{ComplexF64}`:
  Wavenumbers associated with the plans.
- `Mk::Int`:
  Number of wavenumbers.
"""
struct DLPKressH0H1J0J1ChebWorkspace{T<:Real,MatT<:AbstractMatrix{T}}
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
    DLPKressReducedH1J1ChebWorkspace{T,MatT}

Reduced-symmetry value-only Chebyshev workspace for DLP-Kress assembly.

This wraps a full-boundary `DLPKressH1J1ChebWorkspace` and a reduced direct
workspace. The full workspace supplies geometry, Kress data, and special-function
plans, while the reduced workspace supplies the fundamental-index/image mapping.

# Fields
- `direct::DLPKressReducedWorkspace{T,MatT}`:
  Reduced direct workspace containing symmetry/image data.
- `fullcheb::DLPKressH1J1ChebWorkspace{T,MatT}`:
  Full-boundary value-only Chebyshev workspace.
- `m::Int`:
  Reduced matrix dimension.
"""
struct DLPKressReducedH1J1ChebWorkspace{T<:Real,MatT<:AbstractMatrix{T}}
    direct::DLPKressReducedWorkspace{T,MatT}
    fullcheb::DLPKressH1J1ChebWorkspace{T,MatT}
    m::Int
end

"""
    DLPKressReducedH0H1J0J1ChebWorkspace{T,MatT}

Reduced-symmetry derivative-aware Chebyshev workspace for DLP-Kress assembly.

This wraps a full-boundary derivative workspace and a reduced direct workspace.
It is used when assembling reduced Fredholm matrices and their first two
wavenumber derivatives.

# Fields
- `direct::DLPKressReducedWorkspace{T,MatT}`:
  Reduced direct workspace containing symmetry/image data.
- `fullcheb::DLPKressH0H1J0J1ChebWorkspace{T,MatT}`:
  Full-boundary derivative-aware Chebyshev workspace.
- `m::Int`:
  Reduced matrix dimension.
"""
struct DLPKressReducedH0H1J0J1ChebWorkspace{T<:Real,MatT<:AbstractMatrix{T}}
    direct::DLPKressReducedWorkspace{T,MatT}
    fullcheb::DLPKressH0H1J0J1ChebWorkspace{T,MatT}
    m::Int
end

const DLPKressValueChebWorkspace=Union{DLPKressH1J1ChebWorkspace,DLPKressReducedH1J1ChebWorkspace}
const DLPKressDerivativeChebWorkspace=Union{DLPKressH0H1J0J1ChebWorkspace,DLPKressReducedH0H1J0J1ChebWorkspace}

@inline _cheb_workspace_dim(ws::DLPKressH1J1ChebWorkspace)=ws.block_cache.block.N
@inline _cheb_workspace_dim(ws::DLPKressH0H1J0J1ChebWorkspace)=ws.block_cache.block.N
@inline _cheb_workspace_dim(ws::DLPKressReducedH1J1ChebWorkspace)=ws.m
@inline _cheb_workspace_dim(ws::DLPKressReducedH0H1J0J1ChebWorkspace)=ws.m

"""
    build_dlp_kress_h1_j1_cheb_workspace(solver::Union{DLP_kress{T},DLP_kress_global_corners{T}},pts::BoundaryPointsCFIE{T},direct::DLPKressWorkspace{T,MatT},ks::Vector{ComplexF64};npanels_h::Int=10000,npanels_j::Int=2000,M_h::Int=5,M_j::Int=5,pad=(T(0.95),T(1.05)),rmin_cheb::Union{Nothing,Float64}=nothing,plan_nthreads::Int=1,ntls::Int=Threads.nthreads()) where {T<:Real,MatT<:AbstractMatrix{T}}

Build the value-only Chebyshev workspace for DLP-Kress assembly.

This constructs:
- the DLP-Kress geometry/block cache,
- Chebyshev plans for `H₁^(1)(k r)`,
- Chebyshev plans for `J₁(k r)`,
- thread-local buffers for value-only multi-`k` assembly.

Use this when only the Fredholm matrices `F(k)=I-D(k)` are required.

# Arguments
- `solver`:
  DLP-Kress solver, smooth or globally corner-graded.
- `pts::BoundaryPointsCFIE{T}`:
  Boundary discretization.
- `direct::DLPKressWorkspace{T,MatT}`:
  Full direct DLP-Kress workspace for the same geometry.
- `ks::Vector{ComplexF64}`:
  Wavenumbers for which plans are built.
- `npanels_h`, `npanels_j`:
  Number of Chebyshev panels for Hankel and Bessel-J plans.
- `M_h`, `M_j`:
  Chebyshev degree for Hankel and Bessel-J plans.
- `pad`:
  Multiplicative padding for the global Hankel interpolation interval.
- `rmin_cheb`:
  Optional lower cutoff for Hankel interpolation. Distances below this use the
  low-`z`/direct path.
- `plan_nthreads`:
  Number of threads used to build the plans.
- `ntls`:
  Number of thread-local special-function buffers.

# Returns
- `DLPKressH1J1ChebWorkspace{T,MatT}`
"""
function build_dlp_kress_h1_j1_cheb_workspace(solver::Union{DLP_kress{T},DLP_kress_global_corners{T}},pts::BoundaryPointsCFIE{T},direct::DLPKressWorkspace{T,MatT},ks::Vector{ComplexF64};npanels_h::Int=10000,npanels_j::Int=2000,M_h::Int=5,M_j::Int=5,pad=(T(0.95),T(1.05)),rmin_cheb::Union{Nothing,Float64}=nothing,plan_nthreads::Int=1,ntls::Int=Threads.nthreads()) where {T<:Real,MatT<:AbstractMatrix{T}}
    block_cache=build_dlp_kress_block_cache(solver,pts;npanels_h=npanels_h,npanels_j=npanels_j,M_h=M_h,M_j=M_j,pad=pad,rmin_cheb=rmin_cheb)
    plans1,plansj1=build_DLP_kress_plans_h1_j1(ks,block_cache.rmin,block_cache.rmax;npanels_h=npanels_h,npanels_j=npanels_j,M_h=M_h,M_j=M_j,nthreads=plan_nthreads)
    bessel_ws=DLP_H1_J1_BesselWorkspace(length(ks);ntls=ntls)
    return DLPKressH1J1ChebWorkspace{T,MatT}(direct,block_cache,plans1,plansj1,bessel_ws,ks,length(ks))
end

"""
    build_dlp_kress_h0_h1_j0_j1_cheb_workspace(solver::Union{DLP_kress{T},DLP_kress_global_corners{T}},pts::BoundaryPointsCFIE{T},direct::DLPKressWorkspace{T,MatT},ks::Vector{ComplexF64};npanels_h::Int=10000,npanels_j::Int=2000,M_h::Int=5,M_j::Int=5,pad=(T(0.95),T(1.05)),rmin_cheb::Union{Nothing,Float64}=nothing,plan_nthreads::Int=1,ntls::Int=Threads.nthreads()) where {T<:Real,MatT<:AbstractMatrix{T}}

Build the derivative-aware Chebyshev workspace for DLP-Kress assembly.

This constructs:
- the DLP-Kress geometry/block cache,
- Chebyshev plans for `H₀^(1)(k r)` and `H₁^(1)(k r)`,
- Chebyshev plans for `J₀(k r)` and `J₁(k r)`,
- thread-local buffers for derivative-aware multi-`k` assembly.

Use this when `F(k)`, `dF/dk`, and `d²F/dk²` are required.

# Returns
- `DLPKressH0H1J0J1ChebWorkspace{T,MatT}`
"""
function build_dlp_kress_h0_h1_j0_j1_cheb_workspace(solver::Union{DLP_kress{T},DLP_kress_global_corners{T}},pts::BoundaryPointsCFIE{T},direct::DLPKressWorkspace{T,MatT},ks::Vector{ComplexF64};npanels_h::Int=10000,npanels_j::Int=2000,M_h::Int=5,M_j::Int=5,pad=(T(0.95),T(1.05)),rmin_cheb::Union{Nothing,Float64}=nothing,plan_nthreads::Int=1,ntls::Int=Threads.nthreads()) where {T<:Real,MatT<:AbstractMatrix{T}}
    block_cache=build_dlp_kress_block_cache(solver,pts;npanels_h=npanels_h,npanels_j=npanels_j,M_h=M_h,M_j=M_j,pad=pad,rmin_cheb=rmin_cheb)
    plans0,plans1,plansj0,plansj1=build_DLP_kress_plans_h0_h1_j0_j1(ks,block_cache.rmin,block_cache.rmax;npanels_h=npanels_h,npanels_j=npanels_j,M_h=M_h,M_j=M_j,nthreads=plan_nthreads)
    bessel_ws=DLP_H0_H1_J0_J1_BesselWorkspace(length(ks);ntls=ntls)
    return DLPKressH0H1J0J1ChebWorkspace{T,MatT}(direct,block_cache,plans0,plans1,plansj0,plansj1,bessel_ws,ks,length(ks))
end

"""
    build_dlp_kress_h1_j1_cheb_workspace(solver::Union{DLP_kress{T},DLP_kress_global_corners{T}},pts::BoundaryPointsCFIE{T},direct::DLPKressReducedWorkspace{T,MatT},ks::Vector{ComplexF64};kwargs...) where {T<:Real,MatT<:AbstractMatrix{T}}

Build the reduced-symmetry value-only Chebyshev workspace.

This first builds the full-boundary `H₁/J₁` Chebyshev workspace using
`direct.full`, then wraps it with the reduced workspace metadata.

# Returns
- `DLPKressReducedH1J1ChebWorkspace`
"""
function build_dlp_kress_h1_j1_cheb_workspace(solver::Union{DLP_kress{T},DLP_kress_global_corners{T}},pts::BoundaryPointsCFIE{T},direct::DLPKressReducedWorkspace{T,MatT},ks::Vector{ComplexF64};kwargs...) where {T<:Real,MatT<:AbstractMatrix{T}}
    fullcheb=build_dlp_kress_h1_j1_cheb_workspace(solver,pts,direct.full,ks;kwargs...)
    return DLPKressReducedH1J1ChebWorkspace{T,MatT}(direct,fullcheb,direct.m)
end

"""
    build_dlp_kress_h0_h1_j0_j1_cheb_workspace(solver::Union{DLP_kress{T},DLP_kress_global_corners{T}},pts::BoundaryPointsCFIE{T},direct::DLPKressReducedWorkspace{T,MatT},ks::Vector{ComplexF64};kwargs...) where {T<:Real,MatT<:AbstractMatrix{T}}

Build the reduced-symmetry derivative-aware Chebyshev workspace.

This first builds the full-boundary `H₀/H₁/J₀/J₁` Chebyshev workspace using
`direct.full`, then wraps it with the reduced workspace metadata.

# Returns
- `DLPKressReducedH0H1J0J1ChebWorkspace`
"""
function build_dlp_kress_h0_h1_j0_j1_cheb_workspace(solver::Union{DLP_kress{T},DLP_kress_global_corners{T}},pts::BoundaryPointsCFIE{T},direct::DLPKressReducedWorkspace{T,MatT},ks::Vector{ComplexF64};kwargs...) where {T<:Real,MatT<:AbstractMatrix{T}}
    fullcheb=build_dlp_kress_h0_h1_j0_j1_cheb_workspace(solver,pts,direct.full,ks;kwargs...)
    return DLPKressReducedH0H1J0J1ChebWorkspace{T,MatT}(direct,fullcheb,direct.m)
end

"""
    _h0_h1_j0_j1_at_pidx_t!(h0vals::AbstractVector{ComplexF64},h1vals::AbstractVector{ComplexF64},j0vals::AbstractVector{ComplexF64},j1vals::AbstractVector{ComplexF64},pidx_h::Int32,t_h::Float64,pidx_j::Int32,t_j::Float64,r::Float64,plans0::AbstractVector{ChebHankelPlanH},plans1::AbstractVector{ChebHankelPlanH},plansj0::AbstractVector{ChebJPlan},plansj1::AbstractVector{ChebJPlan})

Evaluate `H₀^(1)`, `H₁^(1)`, `J₀`, and `J₁` for all wavenumbers at one fixed
Chebyshev panel/location, writing the results in place.

This is the small inner evaluator used by the multi-`k` DLP-Kress assembly.
The geometric pair `(i,j)` has already been mapped to:
- a panel index `pidx`,
- a local coordinate `t ∈ [-1,1]`,
- and a physical distance `r`.

For each stored wavenumber `k_m`, the function interpolates:
- `H₀^(1)(k_m r)`
- `H₁^(1)(k_m r)`
- `J₀(k_m r)`
- `J₁(k_m r)`

If `pidx_h` is zero (Hankels), the distance is below the interpolation cutoff and direct evaluation is used instead to avoid accuracy issues and having too many panels to resolve the small z=k*r region.

# Returns
- `nothing`
"""
@inline function _h0_h1_j0_j1_at_pidx_t!(h0vals::AbstractVector{ComplexF64},h1vals::AbstractVector{ComplexF64},j0vals::AbstractVector{ComplexF64},j1vals::AbstractVector{ComplexF64},pidx_h::Int32,t_h::Float64,pidx_j::Int32,t_j::Float64,r::Float64,plans0::AbstractVector{ChebHankelPlanH},plans1::AbstractVector{ChebHankelPlanH},plansj0::AbstractVector{ChebJPlan},plansj1::AbstractVector{ChebJPlan})
    h0_h1_j0_j1_multi_ks_at_r!(h0vals,h1vals,j0vals,j1vals,plans0,plans1,plansj0,plansj1,pidx_h,t_h,pidx_j,t_j,r)
    return nothing
end

"""
    function _h1_j1_at_pidx_t!(h1vals::AbstractVector{ComplexF64},j1vals::AbstractVector{ComplexF64},pidx_h::Int32,t_h::Float64,pidx_j::Int32,t_j::Float64,r::Float64,plans1::AbstractVector{ChebHankelPlanH},plansj1::AbstractVector{ChebJPlan})

Evaluate `H₁^(1)` and `J₁` for all stored wavenumbers at one fixed distance
`r`, writing the results in place.

The Hankel and Bessel-J interpolants may use different radial panelizations, so
this helper receives separate cached Chebyshev panel data:
- `(pidx_h, t_h)` for the Hankel plan,
- `(pidx_j, t_j)` for the Bessel-J plan.

If `pidx_h == 0`, the Hankel evaluation falls back to the small-`z` / direct
path. The Bessel-J plan is regular at `r = 0`, so it normally uses its cached
J-panel directly.

# Returns
- `nothing`
"""
@inline function _h1_j1_at_pidx_t!(h1vals::AbstractVector{ComplexF64},j1vals::AbstractVector{ComplexF64},pidx_h::Int32,t_h::Float64,pidx_j::Int32,t_j::Float64,r::Float64,plans1::AbstractVector{ChebHankelPlanH},plansj1::AbstractVector{ChebJPlan})
    h1_j1_multi_ks_at_r!(h1vals,j1vals,plans1,plansj1,pidx_h,t_h,pidx_j,t_j,r)
    return nothing
end

"""
    construct_dlp_kress_matrices_chebyshev!(Ds,pts,ws;multithreaded=true)

Assemble the Kress-corrected DLP matrices for all wavenumbers stored in a
prebuilt Chebyshev workspace, writing the results in place.
This is the multi-`k` value-only Chebyshev assembly function for the raw
double-layer operator `D(k)`. One dense matrix is filled for each wavenumber in
`ws.ks`.

# Arguments
- `Ds::Vector{Matrix{ComplexF64}}`:
  Output matrices, one for each wavenumber in `ws.ks`.
- `pts::BoundaryPointsCFIE{T}`:
  Boundary discretization. Included for API consistency with the direct route.
- `ws::DLPKressH1J1ChebWorkspace{T}`:
  Prebuilt Chebyshev workspace.
- `multithreaded::Bool=true`:
  Enables threaded off-diagonal assembly when beneficial.

# Returns
- `nothing`
"""
function _construct_dlp_kress_matrices_chebyshev!(Ds::Vector{Matrix{ComplexF64}},pts::BoundaryPointsCFIE{T},ws::DLPKressH1J1ChebWorkspace{T};multithreaded::Bool=true) where {T<:Real}
    Mk=ws.Mk
    blk=ws.block_cache.block
    N=blk.N
    ks=ws.ks
    @inbounds for q in 1:Mk
        fill!(Ds[q],0.0+0.0im)
    end
    αL1s=Vector{ComplexF64}(undef,Mk)
    αL2s=Vector{ComplexF64}(undef,Mk)
    @inbounds for q in 1:Mk
        k=ks[q]
        αL1s[q]=-k*_INV_TWO_PI
        αL2s[q]=0.5im*k
    end
    h1_tls=ws.bessel_ws.h1_tls
    j1_tls=ws.bessel_ws.j1_tls
    plans1=ws.plans1
    plansj1=ws.plansj1
    R=blk.R
    invRmat=blk.invR
    logterm=blk.logterm
    inner=blk.inner
    Rkress=blk.Rkress
    wi=blk.wi
    pidx=blk.pidx
    tloc=blk.tloc
    pidxj=blk.pidxj
    tlocj=blk.tlocj
    kappa=blk.kappa
    @inbounds for q in 1:Mk
        Dq=Ds[q]
        for i in 1:N
            Dq[i,i]=ComplexF64(wi[i]*kappa[i],0.0)
        end
    end
    @use_threads multithreading=multithreaded for j in 2:N
        tid=Threads.threadid()
        h1vals=h1_tls[tid]
        j1vals=j1_tls[tid]
        @inbounds for i in 1:j-1
            r=R[i,j]
            _h1_j1_at_pidx_t!(h1vals,j1vals,pidx[i,j],tloc[i,j],pidxj[i,j],tlocj[i,j],r,plans1,plansj1)
            invr=invRmat[i,j]
            lt=logterm[i,j]
            inn_ij=inner[i,j]
            inn_ji=inner[j,i]
            Rij=Rkress[i,j]
            wj=wi[j]
            wii=wi[i]
            c1ij=Rij*inn_ij*invr
            c2ij=wj*inn_ij*invr
            c3ij=wj*lt
            c1ji=Rij*inn_ji*invr
            c2ji=wii*inn_ji*invr
            c3ji=wii*lt
            for q in 1:Mk
                l1=αL1s[q]*j1vals[q]
                hterm=αL2s[q]*h1vals[q]
                Ds[q][i,j]=c1ij*l1+c2ij*hterm-c3ij*l1
                Ds[q][j,i]=c1ji*l1+c2ji*hterm-c3ji*l1
            end
        end
    end
    return nothing
end

function _construct_dlp_kress_matrices_chebyshev!(Ds::Vector{Matrix{ComplexF64}},pts::BoundaryPointsCFIE{T},rws::DLPKressReducedH1J1ChebWorkspace{T};multithreaded::Bool=true) where {T<:Real}
    fullws=rws.fullcheb
    blk=fullws.block_cache.block
    rw=rws.direct
    Mk=fullws.Mk
    m=rw.m
    Ifund=rw.Ifund
    ks=fullws.ks
    @inbounds for q in 1:Mk
        fill!(Ds[q],0.0+0.0im)
    end
    αL1s=Vector{ComplexF64}(undef,Mk)
    αL2s=Vector{ComplexF64}(undef,Mk)
    @inbounds for q in 1:Mk
        k=ks[q]
        αL1s[q]=-k*_INV_TWO_PI
        αL2s[q]=0.5im*k
    end
    ntls=length(fullws.bessel_ws.h1_tls)
    acc_tls=[Vector{ComplexF64}(undef,Mk) for _ in 1:ntls]
    h1_tls=fullws.bessel_ws.h1_tls
    j1_tls=fullws.bessel_ws.j1_tls
    plans1=fullws.plans1
    plansj1=fullws.plansj1
    R=blk.R
    invRmat=blk.invR
    logterm=blk.logterm
    inner=blk.inner
    Rkress=blk.Rkress
    wi=blk.wi
    pidx=blk.pidx
    tloc=blk.tloc
    pidxj=blk.pidxj
    tlocj=blk.tlocj
    kappa=blk.kappa
    @use_threads multithreading=multithreaded for b in 1:m
        tid=Threads.threadid()
        h1vals=h1_tls[tid]
        j1vals=j1_tls[tid]
        accs=acc_tls[tid]
        imgs=rw.fund_to_full[b]
        scales=rw.fund_to_scale[b]
        @inbounds for a in 1:m
            fill!(accs,0.0+0.0im)
            i=Ifund[a]
            for l in eachindex(imgs)
                j=imgs[l]
                scale=ComplexF64(scales[l])
                if i==j
                    d0=scale*ComplexF64(wi[i]*kappa[i],0.0)
                    for q in 1:Mk
                        accs[q]+=d0
                    end
                    continue
                end
                r=R[i,j]
                invr=invRmat[i,j]
                lt=logterm[i,j]
                inn=inner[i,j]
                Rij=Rkress[i,j]
                wj=wi[j]
                _h1_j1_at_pidx_t!(h1vals,j1vals,pidx[i,j],tloc[i,j],pidxj[i,j],tlocj[i,j],r,plans1,plansj1)
                c1=scale*Rij*inn*invr
                c2=scale*wj*inn*invr
                c3=c2*lt
                for q in 1:Mk
                    l1=αL1s[q]*j1vals[q]
                    accs[q]+=c1*l1+c2*αL2s[q]*h1vals[q]-c3*l1
                end
            end
            for q in 1:Mk
                Ds[q][a,b]=accs[q]
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

For each stored wavenumber `k_m`, this function computes:
- `D(k_m)`
- `D₁(k_m)=dD/dk`
- `D₂(k_m)=d²D/dk²`

using Chebyshev-interpolated evaluations of `H₀^(1)`, `H₁^(1)`, `J₀`, and
`J₁`.

# Arguments
- `Ds::Vector{Matrix{ComplexF64}}`:
  Output matrices for `D(k)`.
- `D1s::Vector{Matrix{ComplexF64}}`:
  Output matrices for `dD/dk`.
- `D2s::Vector{Matrix{ComplexF64}}`:
  Output matrices for `d²D/dk²`.
- `pts::BoundaryPointsCFIE{T}`:
  Boundary discretization. Included for API consistency.
- `ws::DLPKressH0H1J0J1ChebWorkspace{T}`:
  Prebuilt Chebyshev workspace.
- `multithreaded::Bool=true`:
  Enables threaded off-diagonal assembly when beneficial.

# Returns
- `nothing`
"""
function _construct_dlp_kress_matrices_derivatives_chebyshev!(Ds::Vector{<:AbstractMatrix{ComplexF64}},D1s::Vector{<:AbstractMatrix{ComplexF64}},D2s::Vector{<:AbstractMatrix{ComplexF64}},pts::BoundaryPointsCFIE{T},ws::DLPKressH0H1J0J1ChebWorkspace{T};multithreaded::Bool=true) where {T<:Real}
    Mk=ws.Mk
    blk=ws.block_cache.block
    N=blk.N
    ks=ws.ks
    R=blk.R;invRmat=blk.invR;logterm=blk.logterm;inner=blk.inner;Rkress=blk.Rkress;wi=blk.wi;kappa=blk.kappa
    pidx=blk.pidx;tloc=blk.tloc;pidxj=blk.pidxj;tlocj=blk.tlocj
    plans0=ws.plans0;plans1=ws.plans1;plansj0=ws.plansj0;plansj1=ws.plansj1
    αL1s=Vector{ComplexF64}(undef,Mk)
    αL2s=Vector{ComplexF64}(undef,Mk)
    kcs=Vector{ComplexF64}(undef,Mk)
    @inbounds for q in 1:Mk
        fill!(Ds[q],0.0+0.0im)
        fill!(D1s[q],0.0+0.0im)
        fill!(D2s[q],0.0+0.0im)
        k=ks[q]
        αL1s[q]=-k*_INV_TWO_PI
        αL2s[q]=0.5im*k
        kcs[q]=k*_INV_TWO_PI
        for i in 1:N
            Ds[q][i,i]=ComplexF64(wi[i]*kappa[i],0.0)
        end
    end
    h0_tls=ws.bessel_ws.h0_tls
    h1_tls=ws.bessel_ws.h1_tls
    j0_tls=ws.bessel_ws.j0_tls
    j1_tls=ws.bessel_ws.j1_tls
    @use_threads multithreading=multithreaded for j in 2:N
        tid=Threads.threadid()
        h0vals=h0_tls[tid];h1vals=h1_tls[tid];j0vals=j0_tls[tid];j1vals=j1_tls[tid]
        @inbounds for i in 1:j-1
            r=R[i,j]
            _h0_h1_j0_j1_at_pidx_t!(h0vals,h1vals,j0vals,j1vals,pidx[i,j],tloc[i,j],pidxj[i,j],tlocj[i,j],r,plans0,plans1,plansj0,plansj1)
            invr=invRmat[i,j]
            lt=logterm[i,j]
            Rij=Rkress[i,j]
            wj=wi[j]
            wii=wi[i]
            inn_ij=inner[i,j]
            inn_ji=inner[j,i]
            cD_ij=inn_ij*invr
            cD_ji=inn_ji*invr
            c1_ij=Rij*cD_ij
            c1_ji=Rij*cD_ji
            c2_ij=wj*cD_ij
            c2_ji=wii*cD_ji
            c3_ij=wj*lt*cD_ij
            c3_ji=wii*lt*cD_ji
            d1j0_ij=-inn_ij
            d1j0_ji=-inn_ji
            d1h0_ij=im*pi*inn_ij
            d1h0_ji=im*pi*inn_ji
            d2pref_ij=_INV_TWO_PI*inn_ij
            d2pref_ji=_INV_TWO_PI*inn_ji
            @inbounds for q in 1:Mk
                k=ks[q]
                h0=h0vals[q];h1=h1vals[q];j0=j0vals[q];j1=j1vals[q]
                l1=αL1s[q]*j1
                Ds[q][i,j]=c1_ij*l1+c2_ij*αL2s[q]*h1-c3_ij*l1
                Ds[q][j,i]=c1_ji*l1+c2_ji*αL2s[q]*h1-c3_ji*l1
                D1s[q][i,j]=kcs[q]*(Rij*d1j0_ij*j0+wj*(lt*inn_ij*j0+d1h0_ij*h0))
                D1s[q][j,i]=kcs[q]*(Rij*d1j0_ji*j0+wii*(lt*inn_ji*j0+d1h0_ji*h0))
                u=j0-k*r*j1
                v=h0-k*r*h1
                D2s[q][i,j]=d2pref_ij*(Rij*(-u)+wj*(lt*u+im*pi*v))
                D2s[q][j,i]=d2pref_ji*(Rij*(-u)+wii*(lt*u+im*pi*v))
            end
        end
    end
    return nothing
end

function _construct_dlp_kress_matrices_derivatives_chebyshev!(Ds::Vector{Matrix{ComplexF64}},D1s::Vector{Matrix{ComplexF64}},D2s::Vector{Matrix{ComplexF64}},pts::BoundaryPointsCFIE{T},rws::DLPKressReducedH0H1J0J1ChebWorkspace{T};multithreaded::Bool=true) where {T<:Real}
    fullws=rws.fullcheb
    blk=fullws.block_cache.block
    rw=rws.direct
    Mk=fullws.Mk
    m=rw.m
    Ifund=rw.Ifund
    ks=fullws.ks
    @inbounds for q in 1:Mk
        fill!(Ds[q],0.0+0.0im)
        fill!(D1s[q],0.0+0.0im)
        fill!(D2s[q],0.0+0.0im)
    end
    αL1s=Vector{ComplexF64}(undef,Mk)
    αL2s=Vector{ComplexF64}(undef,Mk)
    @inbounds for q in 1:Mk
        k=ks[q]
        αL1s[q]=-k*_INV_TWO_PI
        αL2s[q]=0.5im*k
    end
    ntls=length(fullws.bessel_ws.h0_tls)
    acc_tls=[Vector{ComplexF64}(undef,Mk) for _ in 1:ntls]
    acc1_tls=[Vector{ComplexF64}(undef,Mk) for _ in 1:ntls]
    acc2_tls=[Vector{ComplexF64}(undef,Mk) for _ in 1:ntls]
    h0_tls=fullws.bessel_ws.h0_tls
    h1_tls=fullws.bessel_ws.h1_tls
    j0_tls=fullws.bessel_ws.j0_tls
    j1_tls=fullws.bessel_ws.j1_tls
    plans0=fullws.plans0
    plans1=fullws.plans1
    plansj0=fullws.plansj0
    plansj1=fullws.plansj1
    R=blk.R
    invRmat=blk.invR
    logterm=blk.logterm
    inner=blk.inner
    Rkress=blk.Rkress
    wi=blk.wi
    pidx=blk.pidx
    tloc=blk.tloc
    pidxj=blk.pidxj
    tlocj=blk.tlocj
    kappa=blk.kappa
    @use_threads multithreading=multithreaded for b in 1:m
        tid=Threads.threadid()
        h0vals=h0_tls[tid]
        h1vals=h1_tls[tid]
        j0vals=j0_tls[tid]
        j1vals=j1_tls[tid]
        acc=acc_tls[tid]
        acc1=acc1_tls[tid]
        acc2=acc2_tls[tid]
        imgs=rw.fund_to_full[b]
        scales=rw.fund_to_scale[b]
        @inbounds for a in 1:m
            fill!(acc,0.0+0.0im)
            fill!(acc1,0.0+0.0im)
            fill!(acc2,0.0+0.0im)
            i=Ifund[a]
            for l in eachindex(imgs)
                j=imgs[l]
                scale=ComplexF64(scales[l])
                if i==j
                    d0=scale*ComplexF64(wi[i]*kappa[i],0.0)
                    for q in 1:Mk
                        acc[q]+=d0
                    end
                    continue
                end
                r=R[i,j]
                invr=invRmat[i,j]
                lt=logterm[i,j]
                inn=inner[i,j]
                Rij=Rkress[i,j]
                wj=wi[j]
                _h0_h1_j0_j1_at_pidx_t!(h0vals,h1vals,j0vals,j1vals,pidx[i,j],tloc[i,j],pidxj[i,j],tlocj[i,j],r,plans0,plans1,plansj0,plansj1)
                cD1=scale*Rij*inn*invr
                cD2=scale*wj*inn*invr
                cD3=scale*wj*inn*invr*lt
                cR=scale*Rij*inn*_INV_TWO_PI
                cW=scale*wj*inn*_INV_TWO_PI
                wr=k*r
                for q in 1:Mk
                    kq=ks[q]
                    h0=h0vals[q]
                    h1=h1vals[q]
                    j0=j0vals[q]
                    j1=j1vals[q]
                    l1=αL1s[q]*j1
                    acc[q]+=cD1*l1+cD2*αL2s[q]*h1-cD3*l1
                    acc1[q]+=cR*(-kq*j0)+cW*(kq*(lt*j0+im*pi*h0))
                    acc2[q]+=cR*(kq*r*j1-j0)+cW*(lt*(j0-kq*r*j1)+im*pi*(h0-kq*r*h1))
                end
            end
            for q in 1:Mk
                Ds[q][a,b]=acc[q]
                D1s[q][a,b]=acc1[q]
                D2s[q][a,b]=acc2[q]
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
- `ws::DLPKressValueChebWorkspace` and `DLPKressDerivativeChebWorkspace`:
  Prebuilt Chebyshev workspace based on if we need derivatives or not.
- `multithreaded::Bool=true`:
  Enables threaded assembly where beneficial.

# Returns
- `nothing`
"""
function construct_dlp_kress_matrices_chebyshev!(Fs::Vector{<:AbstractMatrix{ComplexF64}},pts::BoundaryPointsCFIE{T},ws::Union{DLPKressH1J1ChebWorkspace{T},DLPKressReducedH1J1ChebWorkspace{T}};multithreaded::Bool=true) where {T<:Real}
    _construct_dlp_kress_matrices_chebyshev!(Fs,pts,ws;multithreaded=multithreaded)
    @inbounds for m in eachindex(Fs),j in axes(Fs[m],2),i in axes(Fs[m],1)
        Fs[m][i,j]*=-1
    end
    @inbounds for m in eachindex(Fs),i in axes(Fs[m],1)
        Fs[m][i,i]+=1.0+0im
    end
    return nothing
end

function construct_dlp_kress_matrices_derivatives_chebyshev!(Fs::Vector{<:AbstractMatrix{ComplexF64}},F1s::Vector{<:AbstractMatrix{ComplexF64}},F2s::Vector{<:AbstractMatrix{ComplexF64}},pts::BoundaryPointsCFIE{T},ws::Union{DLPKressH0H1J0J1ChebWorkspace{T},DLPKressReducedH0H1J0J1ChebWorkspace{T}};multithreaded::Bool=true) where {T<:Real}
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

function construct_boundary_matrices!(Tbufs::Vector{Matrix{ComplexF64}},solver::Union{DLP_kress,DLP_kress_global_corners},pts::BoundaryPointsCFIE{T},zj::AbstractVector{ComplexF64};multithreaded::Bool=true,use_chebyshev::Bool=true,n_panels_h::Int=15000,M_h::Int=5,n_panels_j::Int=3000,M_j::Int=5,timeit::Bool=false) where {T<:Real}
    use_chebyshev || error("Direct DLP-Kress complex-k construction is not implemented.")
    @blas_1 begin
        @benchit timeit=timeit "DLP_kress Workspace" directws=build_dlp_kress_workspace(solver,pts)
        @benchit timeit=timeit "DLP_kress H1/J1 Chebyshev Workspace" chebws=build_dlp_kress_h1_j1_cheb_workspace(solver,pts,directws,ComplexF64.(zj);npanels_h=n_panels_h,npanels_j=n_panels_j,M_h=M_h,M_j=M_j,plan_nthreads=Threads.nthreads(),ntls=Threads.nthreads())
        n=_cheb_workspace_dim(chebws)
        @inbounds for q in eachindex(Tbufs)
            @assert size(Tbufs[q])==(n,n) "Tbufs[$q] has size $(size(Tbufs[q])), but DLP-Kress workspace requires ($n,$n)."
            fill!(Tbufs[q],0.0+0.0im)
        end
        @benchit timeit=timeit "DLP_kress H1/J1 Chebyshev" construct_dlp_kress_matrices_chebyshev!(Tbufs,pts,chebws;multithreaded=multithreaded)
    end

    return nothing
end

function construct_boundary_matrices_with_derivatives!(Tbufs::Vector{Matrix{ComplexF64}},dTbufs::Vector{Matrix{ComplexF64}},ddTbufs::Vector{Matrix{ComplexF64}},solver::Union{DLP_kress,DLP_kress_global_corners},pts::BoundaryPointsCFIE{T},zj::AbstractVector{ComplexF64};multithreaded::Bool=true,use_chebyshev::Bool=true,n_panels_h::Int=15000,M_h::Int=5,n_panels_j::Int=3000,M_j::Int=5,timeit::Bool=false) where {T<:Real}
    use_chebyshev || error("Direct DLP-Kress complex-k derivative construction is not implemented.")
    @blas_1 begin
        @benchit timeit=timeit "DLP_kress Workspace" directws=build_dlp_kress_workspace(solver,pts)
        @benchit timeit=timeit "DLP_kress H0/H1/J0/J1 Chebyshev Workspace" chebws=
            build_dlp_kress_h0_h1_j0_j1_cheb_workspace(solver,pts,directws,ComplexF64.(zj);npanels_h=n_panels_h,npanels_j=n_panels_j,M_h=M_h,M_j=M_j,plan_nthreads=Threads.nthreads(),ntls=Threads.nthreads())
        n=_cheb_workspace_dim(chebws)
        @inbounds for q in eachindex(Tbufs)
            @assert size(Tbufs[q])==(n,n) "Tbufs[$q] has size $(size(Tbufs[q])), but DLP-Kress workspace requires ($n,$n)."
            @assert size(dTbufs[q])==(n,n)
            @assert size(ddTbufs[q])==(n,n)
            fill!(Tbufs[q],0.0+0.0im)
            fill!(dTbufs[q],0.0+0.0im)
            fill!(ddTbufs[q],0.0+0.0im)
        end
        @benchit timeit=timeit "DLP_kress H0/H1/J0/J1 Derivatives Chebyshev" construct_dlp_kress_matrices_derivatives_chebyshev!(Tbufs,dTbufs,ddTbufs,pts,chebws;multithreaded=multithreaded)
    end
    return nothing
end