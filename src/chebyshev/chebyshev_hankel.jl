#############################################################################
# Piecewise-Chebyshev approximation of Hankel functions H_ν^(1)(k r).
#
# 1. Complex-k scaled route (legacy Beyn path)
#    For complex k we interpolate the scaled Hankel
#        H1x(z) = exp(-i z) * H1^(1)(z),   z = k r
#    and store Chebyshev coefficients of
#        G1(r) = sqrt(r) * H1x(k r).
#    The sqrt(r) factor regularizes the small-r behavior, and the scaling
#    avoids overflow/underflow for large |Im(k r)|.
#
# 2. Real-k unscaled route
#    For real k we interpolate the ordinary unscaled Hankel
#        H_ν^(1)(k r)
#    directly, since no exponential scaling is needed and Bessels.jl is
#    faster for real arguments.
#
# In both cases the interval [rmin,rmax] is split into panels, each panel is
# approximated by a Chebyshev series, and evaluation is done with Clenshaw’s
# recurrence after mapping r to the local panel coordinate t ∈ [-1,1].
#
# Main API
# - plan_h1x : build a complex-k scaled H1 plan
# - plan_h1  : build a real-k unscaled H_ν plan
# - panel_indices / panel_and_geom : precompute panel locations and local coords
# - eval_h1x! / eval_h1x : evaluate the stored scaled quantity
# - eval_h1!  / eval_h1  : evaluate the physical unscaled Hankel
# - precompute_phase : precompute exp(i k r) for the complex-k route
#
# MO/20/3/26
#############################################################################

struct ChebHankelTableH1x
    a::Float64 # start of panel 
    b::Float64 # end of panel
    M::Int # order of the chebyshev polynomial for panel [a,b]
    c1::Vector{ComplexF64}   # coeffs of Gν(r) = √r * Hνx(k r)
end

struct ChebHankelTableH1
    a::Float64 # start of panel 
    b::Float64 # end of panel
    M::Int # order of the chebyshev polynomial for panel [a,b]
    ν::Int # order of the Hankel type 1 function
    c::Vector{ComplexF64}   # coeffs of Gν(r) = √r * Hνx(k r)
end

# =============================================================================
# Construct a Chebyshev table on a single panel [a,b] for the scaled Hankel:
#   H1x(z) = exp(-i z) * H1^(1)(z),  with  z = k*r.
# We approximate
#   G1(r) = sqrt(r) * H1x(k r)
# on Chebyshev–Lobatto nodes t_j = cos(π j / M), j=0..M, mapped to r_j ∈ [a,b].
# The sqrt(r) factor regularizes the r→0 behavior.
#
# Inputs
#   k :: ComplexF64         # (fixed) complex wavenumber
#   a,b :: Float64          # panel endpoints, 0 < a < b
#   M :: Int                # Chebyshev degree (stores M+1 coeffs)
#
# Output
#   ChebHankelTableH1x(a,b,M,c) where c are the Chebyshev coeffs of G1.
# =============================================================================
function _build_table_h1x!(k::ComplexF64,a::Float64,b::Float64;M::Int=16)::ChebHankelTableH1x
    @assert a>0 && b>a "a=$(a), b=$(b)" # sanity check to keep the interval bounded above 0 and to not be degenerate/reversed
    f1=Vector{ComplexF64}(undef,M+1) # preallocate the vector storing function evalutions at chebyshev nodes
    @inbounds for j in 0:M
        t=cospi(j/M) # chebyshev node in [-1,1]
        r=((b+a)+(b-a)*t)/2 # affine map to [a,b] so we are in the correct sector
        z=k*r # argument for Hankel in local coordinates
        H1x=besselhx(1,1,z) # scaled: exp(-i z) * H1^(1)(z) since AMOS provides this directly and is more stable than computing the unscaled version due to underflow/overflow issues with large/small |Im(z)|.
        f1[j+1]=sqrt(r)*H1x # G1(r) since for small r this is well behaved
    end
    c1=Vector{ComplexF64}(undef,M+1) # preallocate chebyshev coeffs
    _chebfit!(c1,f1) # fit the chebyshev coeffs to the chebyshev node evaluations
    return ChebHankelTableH1x(a,b,M,c1) # construct the table for that particular panel with local cheby polynomial 
end

# =============================================================================
# Construct a Chebyshev table on a single panel [a,b] for the unscaled Hankel:
#   Hν(z) = H_ν^(1)(z),  with  z = k*r and k real.
#
# We approximate the function Fν(r) = H_ν^(1)(k r) on Chebyshev nodes t_j = cos(π j / M), j=0..M, mapped to r_j ∈ [a,b].
#
# Unlike the complex-k scaled route, this real-k path stores the unscaled Hankel
# values directly.
#
# Inputs
#   ν :: Int                 # Hankel order (typically 0 or 1)
#   k :: Float64             # fixed real wavenumber
#   a,b :: Float64           # panel endpoints, 0 < a < b
#   M :: Int                 # Chebyshev degree (stores M+1 coeffs)
#
# Output
#   ChebHankelTableH1(a,b,M,ν,c) where c are the Chebyshev coeffs of Fν.
# =============================================================================
function _build_table_h1!(ν::Int,k::Float64,a::Float64,b::Float64;M::Int=16)::ChebHankelTableH1
    @assert a>0 && b>a "a=$(a), b=$(b)" # sanity check to keep the interval bounded above 0 and to not be degenerate/reversed
    f1=Vector{ComplexF64}(undef,M+1) # preallocate the vector storing function evalutions at chebyshev nodes
    @inbounds for j in 0:M
        t=cospi(j/M) # chebyshev node in [-1,1]
        r=((b+a)+(b-a)*t)/2 # affine map to [a,b] so we are in the correct sector
        z=k*r # argument for Hankel in local coordinates
        f1[j+1]=ComplexF64(Bessels.hankelh1(ν,z)) # unscaled real-argument Hankel from Bessels.jl
    end
    c=Vector{ComplexF64}(undef,M+1) # preallocate chebyshev coeffs
    _chebfit!(c,f1) # fit the chebyshev coeffs to the chebyshev node evaluations
    return ChebHankelTableH1(a,b,M,ν,c) # construct the table for that particular panel with local cheby polynomial 
end

# multiple dispatch version of above with complex k. Maybe redundant since we have the scaled version but this one does not need the exp mul in the end and therefore faster. Other one is LEGACY made for original formulation for Beyn's method.
function _build_table_h1!(ν::Int,k::ComplexF64,a::Float64,b::Float64;M::Int=16)::ChebHankelTableH1
    @assert a>0 && b>a "a=$(a), b=$(b)" # sanity check to keep the interval bounded above 0 and to not be degenerate/reversed
    f1=Vector{ComplexF64}(undef,M+1) # preallocate the vector storing function evalutions at chebyshev nodes
    @inbounds for j in 0:M
        t=cospi(j/M) # chebyshev node in [-1,1]
        r=((b+a)+(b-a)*t)/2 # affine map to [a,b] so we are in the correct sector
        z=k*r # argument for Hankel in local coordinates
        f1[j+1]=besselh(ν,1,z) # unscaled complex-argument Hankel from AMOS. We use this for complex k since Bessels.jl doesn't support complex arguments and AMOS is more stable for complex arguments. Just dont use this here with large im part for z !!! 
        #FIXME It is hacky since for _build_table_h1x! we used the scaled version primarily for Beyn's method since we did not know the size of the imaginary part, but this seems not safe
    end
    c=Vector{ComplexF64}(undef,M+1) # preallocate chebyshev coeffs
    _chebfit!(c,f1) # fit the chebyshev coeffs to the chebyshev node evaluations
    return ChebHankelTableH1(a,b,M,ν,c) # construct the table for that particular panel with local cheby polynomial 
end

# LEGACY USED FOR BEYN; need it for backwards compatibility
struct ChebHankelPlanH1x
    k::ComplexF64
    panels::Vector{ChebHankelTableH1x}
    rmin::Float64
    rmax::Float64
    grading::Symbol
    dr::Float64
    invdr::Float64
    npanels::Int
end

struct ChebHankelPlanH1
    k::Union{Float64,ComplexF64}
    ν::Int
    panels::Vector{ChebHankelTableH1}
    rmin::Float64
    rmax::Float64
    grading::Symbol
    dr::Float64
    invdr::Float64
    npanels::Int
end

# =============================================================================
# Build a piecewise-Chebyshev plan for G1(r) = sqrt(r) * H1x(k * r) over r ∈ [rmin,rmax].
# The interval is split into `npanels` either uniformly or with geometric growth.
# Each panel stores M+1 Chebyshev coefficients (degree M). This plan is reused for the unscaled hankel function.
#
# Inputs
#   k :: ComplexF64
#   rmin,rmax :: Float64      # 0 < rmin < rmax
#   npanels :: Int
#   M :: Int
#   grading :: Symbol         # :uniform or :geometric
#   geo_ratio :: Real         # panel-size ratio for :geometric
#
# Output
#   ChebHankelPlanH1x(k, panels)
# =============================================================================
function plan_h1x(k::ComplexF64,rmin::Float64,rmax::Float64;npanels::Int=64,M::Int=16,grading::Symbol=:uniform,geo_ratio::Real=1.05)::ChebHankelPlanH1x
    @assert rmin>0 && rmax>rmin
    br= grading===:uniform ?
        _breaks_uniform(rmin,rmax,npanels) :
        _breaks_geometric(rmin,rmax,npanels;ratio=geo_ratio) # breakpoints for panels. These are the same for all k, so we can reuse them if we build multiple plans for different k's but the same geometry.
    panels=Vector{ChebHankelTableH1x}(undef,npanels)
    @inbounds Threads.@threads for i in 1:npanels
        panels[i]=_build_table_h1x!(k,br[i],br[i+1];M=M)
    end
    dr= grading===:uniform ? (rmax-rmin)/npanels : 0.0 # only used for uniform grading, geometric grading doesn't have a fixed panel width so we set dr=0 and invdr=0 
    invdr= grading===:uniform ? inv(dr) : 0.0
    return ChebHankelPlanH1x(k,panels,rmin,rmax,grading,dr,invdr,npanels)
end

# =============================================================================
# Build a piecewise-Chebyshev plan for the unscaled Hankel
#   H_ν^(1)(k r)
# over r ∈ [rmin,rmax], with k real.
#
# The interval is split into `npanels` either uniformly or with geometric growth.
# Each panel stores M+1 Chebyshev coefficients (degree M) for the direct,
# unscaled Hankel values. This path is intended for real-k applications such as
# CFIE/BIM, where the exponential scaling used in the complex-k route is not
# needed.
#
# Inputs
#   ν :: Int
#       # Hankel order (typically 0 for SLP or 1 for DLP)
#   k :: Float64
#       # fixed real wavenumber
#   rmin,rmax :: Float64
#       # 0 < rmin < rmax
#   npanels :: Int
#   M :: Int
#   grading :: Symbol
#       # :uniform or :geometric
#   geo_ratio :: Real
#       # panel-size ratio for :geometric
#
# Output
#   ChebHankelPlanH1(k,ν,panels,...) containing the panelized Chebyshev tables.
# =============================================================================
function plan_h1(ν::Int,k::Union{Float64,ComplexF64},rmin::Float64,rmax::Float64;npanels::Int=64,M::Int=16,grading::Symbol=:uniform,geo_ratio::Real=1.05)::ChebHankelPlanH1
    @assert rmin>0 && rmax>rmin
    br= grading===:uniform ?
        _breaks_uniform(rmin,rmax,npanels) :
        _breaks_geometric(rmin,rmax,npanels;ratio=geo_ratio) # breakpoints for panels. These are the same for all k, so we can reuse them if we build multiple plans for different k's but the same geometry.
    panels=Vector{ChebHankelTableH1}(undef,npanels)
    @inbounds Threads.@threads for i in 1:npanels
        panels[i]=_build_table_h1!(ν,k,br[i],br[i+1];M=M)
    end
    dr= grading===:uniform ? (rmax-rmin)/npanels : 0.0 # only used for uniform grading, geometric grading doesn't have a fixed panel width so we set dr=0 and invdr=0 
    invdr= grading===:uniform ? inv(dr) : 0.0
    return ChebHankelPlanH1(k,ν,panels,rmin,rmax,grading,dr,invdr,npanels)
end

# =============================================================================
# Locate the panel index p such that panels[p].a ≤ r ≤ panels[p].b.
#
# This is a scalar binary search used to identify which Chebyshev panel
# contains the given radius `r`.  Each panel corresponds to one interval [a,b]
# on which the Chebyshev coefficients for G₁(r) = √r * H₁x(k r) were fitted.
#
# Inputs
#   panels :: Union{Vector{ChebHankelTableH1x},Vector{ChebHankelTableH1}}   # vector of panel structs
#   r      :: Float64                      # query radius (must be positive)
#
# Output
#   Int                                 # index p such that r ∈ [panels[p].a, panels[p].b]
#
# Behavior
#   - If r lies within the covered domain, return its panel index.
#   - If r is outside the plan’s total range, an error is thrown.
#
# Implementation details
#   - O(log Np) comparisons.
#   - @inbounds avoids bounds checks for speed.
#   - Typical Np = 50–300, so this routine is negligible cost compared to Hankel evaluation.
# =============================================================================
@inline function _find_panel_binary(panels::Union{Vector{ChebHankelTableH1x},Vector{ChebHankelTableH1}},r::Float64)::Int
    lo=1;hi=length(panels)
    @inbounds while lo≤hi
        mid=(lo+hi)>>>1
        P=panels[mid]
        if r<P.a
            hi=mid-1
        elseif r>P.b
            lo=mid+1
        else
            return mid
        end
    end
    error("r=$r outside plan range")
end

# =============================================================================
# Fast panel lookup for uniformly graded Chebyshev plans.
#
# For plans constructed with grading === :uniform, the panels are equally spaced
# over [rmin, rmax], so the panel index can be obtained in O(1) time using a
# direct arithmetic mapping instead of a binary search.
#
# The mapping is:
#   p = floor((r - rmin) / dr) + 1
# where dr = (rmax - rmin) / npanels.
#
# Inputs
#   pl :: Union{ChebHankelPlanH1x,ChebHankelPlanH1}
#          # plan with uniform panel spacing (pl.grading === :uniform)
#          # must contain fields rmin, invdr, npanels
#   r  :: Float64
#          # query radius (must lie within [pl.rmin, pl.rmax])
#
# Output
#   Int
#          # panel index p such that r lies in panel p
#
# Behavior
#   - Returns a clamped index in [1, pl.npanels].
#   - No bounds error is thrown; values outside range are projected.
#
# Implementation details
#   - O(1) cost: one multiply, one floor, and clamping.
# =============================================================================

@inline function _find_panel_uniform(pl::Union{ChebHankelPlanH1x,ChebHankelPlanH1},r::Float64)::Int
    p=Int(floor((r-pl.rmin)*pl.invdr))+1
    return ifelse(p<1,1,ifelse(p>pl.npanels,pl.npanels,p))
end

# =============================================================================
# Unified panel lookup for Chebyshev plans.
#
# Dispatches to either:
#   - O(1) arithmetic lookup for uniform panels
#   - O(log Np) binary search for nonuniform (e.g. geometric) panels
#
# This function provides a consistent interface for locating the panel index
# p such that r ∈ [panels[p].a, panels[p].b], regardless of grading type.
#
# Inputs
#   pl :: Union{ChebHankelPlanH1x,ChebHankelPlanH1}
#          # Chebyshev plan containing panels and grading information
#   r  :: Float64
#          # query radius (must lie within plan range)
#
# Output
#   Int
#          # panel index p corresponding to r
#
# Behavior
#   - If pl.grading === :uniform:
#         uses direct O(1) lookup (_find_panel_uniform)
#   - Otherwise:
#         uses binary search over panel intervals (_find_panel_binary)
# =============================================================================
@inline function _find_panel(pl::Union{ChebHankelPlanH1x,ChebHankelPlanH1},r::Float64)::Int
    if pl.grading===:uniform
        return _find_panel_uniform(pl,r)
    else
        return _find_panel_binary(pl.panels,r)
    end
end

# =============================================================================
# Vectorized panel search: determine for each r[i] ∈ rvec which panel
# [a,b] of the Chebyshev plan `pl` contains it.
#
# This creates an Int32 vector of panel indices to enable later vectorized
# evaluation of H₁x or H₁.
#
# Inputs
#   pl    :: Union{ChebHankelPlanH1x,ChebHankelPlanH1}
#             # plan containing `panels::Union{Vector{ChebHankelTableH1x},Vector{ChebHankelTableH1}}`
#   rvec  :: AbstractVector{Float64}
#             # radii (each must satisfy pl.panels[1].a ≤ r ≤ pl.panels[end].b)
#
# Output
#   pidx  :: Vector{Int32}
#             # pidx[i] = index of panel that contains rvec[i]
#
# Implementation details
#   - Each thread processes a chunk of `rvec` independently.
#   - Calls `_find_panel` for each rvec[i].
#   - All outputs are computed in place; no allocations except the result vector.
#   - Complexity: O(length(rvec) * log Np).
# =============================================================================
function panel_indices(pl::Union{ChebHankelPlanH1x,ChebHankelPlanH1},rvec::AbstractVector{Float64})::Vector{Int32}
    p=similar(rvec,Int32)
    @inbounds Threads.@threads for i in eachindex(rvec)
        p[i]=Int32(_find_panel(pl,rvec[i]))
    end
    return p
end

# =============================================================================
# Precompute Chebyshev coordinates and normalization factors for the given
# radii rvec, to enable fast, k-independent evaluation of
#   G₁(r) = √r * H₁x(k r)
# and hence H₁x and H₁.
#
# For each r[i], this function computes:
#   • t[i]        = (2r[i] − (a+b)) / (b−a)     # mapped Chebyshev coordinate ∈ [−1,1]
#   • invsqrt[i]  = 1 / √r[i]
#
# Inputs
#   pl     :: Union{ChebHankelPlanH1x,ChebHankelPlanH1}  # plan containing panels [a,b]
#   rvec   :: AbstractVector{Float64}  # radii (must lie in total range of plan)
#   pidx   :: AbstractVector{Int32}    # panel indices for each r[i]
#
# Outputs
#   (t, invsqrt)
#       t        :: Vector{Float64}     # mapped Chebyshev coordinates
#       invsqrt  :: Vector{Float64}     # 1/√r for normalization
# =============================================================================
function precompute_geom(pl::Union{ChebHankelPlanH1x,ChebHankelPlanH1},rvec::AbstractVector{Float64},pidx::AbstractVector{Int32})::Tuple{Vector{Float64},Vector{Float64}}
    @assert length(rvec)==length(pidx)
    n=length(rvec)
    t=Vector{Float64}(undef,n)
    invsqrt=Vector{Float64}(undef, n)
    pans=pl.panels
    @inbounds Threads.@threads for i in eachindex(rvec)
        rr=rvec[i]
        P=pans[pidx[i]]
        t[i]=(2*rr-(P.b+P.a))/(P.b-P.a) # aff_map_inv
        invsqrt[i]=inv(sqrt(rr))
    end
    return t, invsqrt
end

# =============================================================================
# Single-pass, threaded precompute of:
#   - pidx[i]   : panel index for rvec[i]
#   - t[i]      : mapped Chebyshev coordinate in [-1,1] on that panel
#   - invsqrt[i]: 1 / √(rvec[i])
#
# This merges the work of `panel_indices` and `precompute_geom` to avoid
# an extra pass over rvec and redundant memory traffic.
#
# Inputs
#   pl   :: Union{ChebHankelPlanH1x,ChebHankelPlanH1}
#   rvec :: Vector{Float64}  # radii (must lie in [panels[1].a, panels[end].b])
#
# Outputs
#   pidx    :: Vector{Int32}
#   t       :: Vector{Float64}
#   invsqrt :: Vector{Float64}
#
# Notes
#   - Uses an inlined binary search per element (same as _find_panel) to
#     minimize overhead.
#   - Forces no allocations inside the threaded loop.
# =============================================================================
function panel_and_geom(pl::Union{ChebHankelPlanH1x,ChebHankelPlanH1},rvec::AbstractVector{Float64})::Tuple{Vector{Int32},Vector{Float64},Vector{Float64}}
    n=length(rvec)
    pidx=Vector{Int32}(undef,n)
    t=Vector{Float64}(undef,n)
    invsqrt=Vector{Float64}(undef,n)
    if pl.grading===:uniform
        rmin=pl.rmin
        dr=pl.dr
        invdr=pl.invdr
        np=pl.npanels
        @inbounds Threads.@threads for i in eachindex(rvec)
            r=rvec[i]
            p=Int(floor((r-rmin)*invdr))+1
            p=ifelse(p<1,1,ifelse(p>np,np,p))
            pidx[i]=Int32(p)
            center=rmin+(p-0.5)*dr
            t[i]=2*(r-center)*invdr
            invsqrt[i]=inv(sqrt(r))
        end
    else
        pans=pl.panels
        nP=length(pans)
        @inbounds Threads.@threads for i in eachindex(rvec)
            r=rvec[i]
            lo=1
            hi=nP
            mid=0
            while lo<=hi
                mid=(lo+hi)>>>1
                P=pans[mid]
                if r<P.a
                    hi=mid-1
                elseif r>P.b
                    lo=mid+1
                else
                    break
                end
            end
            if !(pans[mid].a<=r<=pans[mid].b)
                error("r=$r outside plan range [$(pans[1].a), $(pans[end].b)]")
            end
            P=pans[mid]
            pidx[i]=Int32(mid)
            t[i]=(2*r-(P.b+P.a))/(P.b-P.a)
            invsqrt[i]=inv(sqrt(r))
        end
    end
    return pidx,t,invsqrt
end

# Cheap exp(i k r) with k = a + i b:
# exp(i k r) = exp(-b r) * (cos(a r) + i sin(a r))
@inline function _exp_ikr(a::Float64,b::Float64,r::Float64)::ComplexF64
    u=exp(-b*r)
    ar=a*r
    return ComplexF64(u*cos(ar),u*sin(ar))
end

# Optional: precompute phase once if you'll reuse the same k & r many times
function precompute_phase(k::ComplexF64,r::AbstractVector{Float64})::Vector{ComplexF64}
    a=real(k);b=imag(k)
    φ=Vector{ComplexF64}(undef,length(r))
    @inbounds Threads.@threads for i in eachindex(r)
        φ[i]=_exp_ikr(a,b,r[i])
    end
    return φ
end

##################################################################
###################### EVALUATION FUNCTIONS ######################
##################################################################

# =============================================================================
# Fast evaluation of the *scaled* Hankel:
#   H1x(z) = exp(-i z) * H1^(1)(z),  with z = k*r.
#
# Storage model
# -------------
# Each panel stores Chebyshev coefficients of
#   G1(r) = √r * H1x(k r)
#
# With k-independent precomputations:
#   - pidx[i]   : which panel r_i belongs to
#   - t[i]      : mapped Chebyshev coordinate in [-1,1] on that panel
#   - invsqrt[i]: 1/√r_i
#
# we evaluate
#   H1x(k r_i) ≈ _cheb_clenshaw(c_panel, t[i]) * invsqrt[i].
#
# Inputs
#   H1x     :: Vector{ComplexF64}        # output (length = length(t))
#   pl      :: ChebHankelPlanH1x  # plan at a fixed complex k
#   r       :: AbstractVector{Float64}   # radii (only used for indexing)
#   pidx    :: AbstractVector{Int32}     # panel index for each i
#   t       :: AbstractVector{Float64}   # Chebyshev coordinate for each i
#   invsqrt :: AbstractVector{Float64}   # 1/√r for each i
#
# Output
#   Fills H1x in place.
# =============================================================================
function eval_h1x!(H1x::AbstractVector{ComplexF64},pl::ChebHankelPlanH1x,r::AbstractVector{Float64},pidx::AbstractVector{Int32},t::AbstractVector{Float64},invsqrt::AbstractVector{Float64})
    pans=pl.panels
    @inbounds Threads.@threads for i in eachindex(r)
        T=pans[pidx[i]]
        g1=_cheb_clenshaw(T.c1,t[i])
        H1x[i]=g1*invsqrt[i]
    end
    return nothing
end

# =============================================================================
# Fast evaluation of the unscaled Hankel for the real-k plan:
#   H_ν^(1)(k r),  with k real.
#
# Each panel stores Chebyshev coefficients of the direct unscaled Hankel
# values on that interval. No exponential scaling and no 1/sqrt(r)
# normalization are used in this real-k route.
#
# Inputs
#   H1   :: AbstractVector{ComplexF64}
#           # output
#   pl   :: ChebHankelPlanH1
#           # plan at fixed real k
#   r    :: AbstractVector{Float64}
#           # radii (unused; kept for API compatibility)
#   pidx :: AbstractVector{Int32}
#           # per-point panel index
#   t    :: AbstractVector{Float64}
#           # per-point Chebyshev coordinate
#
# Output
#   Fills H1 in place with H_ν^(1)(k r_i).
# =============================================================================
function eval_h1!(H1::AbstractVector{ComplexF64},pl::ChebHankelPlanH1,r::AbstractVector{Float64},pidx::AbstractVector{Int32},t::AbstractVector{Float64})
    pans=pl.panels
    @inbounds Threads.@threads for i in eachindex(r)
        T=pans[pidx[i]]
        H1[i]=_cheb_clenshaw(T.c,t[i])
    end
    return nothing
end

# =============================================================================
# ----------------------------------
# Evaluate the *scaled* Hankel function
#       H1x(z) = exp(-i z) * H1^(1)(z)
# for a single point z = k*r using the Chebyshev approximation table stored
# in a ChebHankelPlanH1x.
#
# The scaled form H1x is used to avoid overflow/underflow in AMOS for large
# |Im(k r)|.  The plan stores Chebyshev coefficients of
#
#       G1(r) = √r * H1x(k r)
#
# on each panel [a,b].  We evaluate G1(r) using the Clenshaw recurrence on
# Chebyshev coefficients, then divide by √r to recover H1x(k r).
#
# Inputs
#   pl       :: ChebHankelPlanH1x   # precomputed Chebyshev plan for given k
#   pidx     :: Int32               # panel index such that r ∈ [a,b]
#   t        :: Float64             # mapped Chebyshev coordinate in [-1,1]
#   invsqrt  :: Float64             # 1 / √r
#
# Output
#   ComplexF64                     # approximation of H1x(k r)
@inline function eval_h1x(pl::ChebHankelPlanH1x,pidx::Int32,t::Float64,invsqrt::Float64)
    return _cheb_clenshaw(pl.panels[pidx].c1,t)*invsqrt
end

# =============================================================================
# Evaluate the unscaled Hankel
#   H_ν^(1)(k r),  with k real,
# at a single point using the real-k Chebyshev plan.
#
# Inputs
#   pl   :: ChebHankelPlanH1
#           # precomputed Chebyshev plan for fixed real k
#   pidx :: Int32
#           # panel index such that r ∈ [a,b]
#   t    :: Float64
#           # mapped Chebyshev coordinate in [-1,1]
#
# Output
#   ComplexF64
#           # approximation of H_ν^(1)(k r)
# =============================================================================
@inline function eval_h1(pl::ChebHankelPlanH1,pidx::Int32,t::Float64)
    return _cheb_clenshaw(pl.panels[pidx].c,t)
end

# =============================================================================
# Evaluate the *scaled* Hankel H1x(k r) for the *same radius r* across multiple
# complex wavenumbers (one per plan). Each plan must have been built over the
# *same panel breaks* so that `pidx` and `t` are valid for every plan.
#
# Storage model:
#   Each ChebHankelPlanH1x stores Chebyshev coeffs c for G1(r) = √r * H1x(k r).
# For a fixed r we reuse:
#   - pidx    : panel index s.t. panels[pidx].a ≤ r ≤ panels[pidx].b
#   - t       : mapped Chebyshev coord in [-1,1] for that panel
#   - invsqrt : 1 / √r
#
# Inputs
#   out    :: AbstractVector{ComplexF64}             # length == length(plans)
#   plans  :: AbstractVector{ChebHankelPlanH1x}      # k-dependent Cheb plans
#   pidx   :: Int32                                  # panel index for r
#   t      :: Float64                                # Chebyshev coord for r
#   invsqrt:: Float64                                # 1/√r
#
# Output
#   out[m] = H1x_{k_m}(r) for m = 1..length(plans).
# =============================================================================
function eval_h1x_multi_ks!(out::AbstractVector{ComplexF64},plans::AbstractVector{ChebHankelPlanH1x},pidx::Int32,t::Float64,invsqrt::Float64)
    @inbounds for m in eachindex(plans)
        out[m]=_cheb_clenshaw(plans[m].panels[pidx].c1,t)*invsqrt
    end
    return nothing
end

# =============================================================================
# Evaluate the unscaled Hankel H_ν^(1)(k r) for the same radius r across
# multiple real wavenumbers (one per plan).
#
# Each plan stores the direct unscaled Hankel coefficients on the same panel
# partition, so the same `pidx` and `t` may be reused across all plans.
#
# Inputs
#   out  :: AbstractVector{ComplexF64}
#           # length == length(plans)
#   plans:: AbstractVector{ChebHankelPlanH1}
#           # real-k Chebyshev plans
#   r    :: Float64
#           # radius for this evaluation point (unused; kept for API symmetry)
#   pidx :: Int32
#           # panel index for r
#   t    :: Float64
#           # Chebyshev coordinate for r
#
# Output
#   out[m] = H_ν^(1)(k_m r) for m = 1..length(plans).
# =============================================================================
function eval_h1_multi_ks!(out::AbstractVector{ComplexF64},plans::AbstractVector{ChebHankelPlanH1},r::Float64,pidx::Int32,t::Float64)
    @inbounds for m in eachindex(plans)
        plan_m=plans[m]
        out[m]=_cheb_clenshaw(plan_m.panels[pidx].c,t)
    end
    return nothing
end
