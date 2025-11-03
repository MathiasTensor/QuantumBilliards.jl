#############################################################################
# Chebyshev piecewise approximation of the scaled Hankel function H1^(1)(z) 
# for complex arguments z = k*r, with k complex and r real positive. The scaled
# Hankel is defined as: H1x(z) = exp(-i z) * H1^(1)(z) and we use it to prevent 
# AMOS overflow/underflow issues for large |Im(z)| which inevitably occur in Beyn's
# method.
#
# The idea is to build a piecewise Chebyshev approximation of the function
#   G1(r) = sqrt(r) * H1x(k r) = sqrt(r) * exp(-i k r) * H1^(1)(k r) because
# the sqrt(r) factor stabilizes the region around r=0. We then have:
#   H1x(k r) ≈ _cheb_clenshaw(c1, t) / sqrt(r) where t = aff_map_inv(a,b,r) is 
# the mapped Chebyshev coordinate on panel [a,b], cleb_clenshaw is the Clenshaw algorithm
# evaluation of the Chebyshev series with coefficients c1. This is to avoid the Runge phenomenom with
# high order interpolations and edges of panels.
#
# API
# - plan_h1x: builds a piecewise Chebyshev plan for G1(r) over [rmin,rmax] which is k dependant. This is the most crucial one since
# it computes the expensive function evaluations at Chebyshev nodes and fits the Chebyshev coefficients on each panel.
# - eval_h1!: evaluates H1^(1)(k r) for a vector of r values using the Chebyshev plan
# - eval_h1x!: evaluates the scaled Hankel H1x(k r) for a vector of r values using the Chebyshev plan
# - precompute_phase: precomputes exp(i k r) for a vector of r values to speed up eval_h1!
# - precompute_geom: precomputes the panel indices and mapped Chebyshev coordinates for a vector of r values to speed up eval_h1! and eval_h1x!
#
# Procedure: 
# 1. Estimate r_min and r_max from the given geometry for a given k (or use the ones at largest k)
# 2. Construct scaled hankel plans plan_h1x(k,r_min,r_max;npanels=2000,M=200,grading=:uniform,geo_ratio=1.05). Choose number of panels npanels & 
# chebyshev polynomial degree M such that one gets the wanted precision wrt SpecialFunctions.jl.
# 3. Construct panels and geometry panel_and_geom(plan,rs) to get for all r in rs the panel on which they live, the index of that panel and the chebyshev
# interval variable there + the inverse sqrt(r) since it is useful for inverting the G1(r).
# 4. Use eval_h1x or eval_h1 (either the scalar or vector inplace versions) with the above data depending on whether one wants the scaled or
# unscaled versions.
#
# NOTE - Implemented only with double precision ComplexF64 throughout as the SpecialFuntions.jl for evaluating Hankel functions (scaled/unscaled) only supports this type (AMOS algorithm).
# MO/30/10/25
#############################################################################

# tries to fit a piecewise Chebyshev polynomial approximation of order M on each panel [a,b] for the function G1(r) = √r * H1x(k r) where H1x(z) = exp(-i z) * H1^(1)(z).
# We evaluated the function at Chebyshev–Lobatto nodes at each panel and then fit the Chebyshev coefficients. c is the vector of Chebyshev coefficients and f is the vector of function evaluations at Chebyshev nodes
# This is all done with clenshaw's reverse recurrence algorithm for numerical stability.
@inline function _cheb_clenshaw(c::AbstractVector{ComplexF64},t::Float64)::ComplexF64
    b1=0.0+0.0im
    b2=0.0+0.0im
    twot=2*t
    @inbounds for k in (length(c)-1):-1:1
        b=muladd(twot,b1,c[k+1]-b2)
        b2=b1
        b1=b
    end
    return muladd(t,b1,c[1])-b2
end

# =============================================================================
# Compute Chebyshev expansion coefficients c (in place) from samples f taken at
# Chebyshev–Lobatto nodes on [-1,1]. This is a direct O(M^2) Discrete Cosine
# Transform of type I with endpoint half-weights.
#
# Inputs
#   c :: Vector{ComplexF64}   # output coefficients (length M+1), filled in place
#   f :: Vector{ComplexF64}   # samples at Lobatto nodes, length M+1 with t_j = cos(π j / M), j=0..M
#
# Construction
#   Given f_j = f(t_j), this computes coefficients c_m (m=0..M) such that
#       f(t) ≈ ∑_{m=0}^M c_m T_m(t)
#   via the weighted projection
#       c_m = (2/M) * ∑_{j=0}^M w_j * f_j * cos(π j m / M),
#   with Lobatto endpoint weights
#       w_0 = w_M = 1/2,   and   w_j = 1 for j=1..M-1.
#
# Normalization note (matches _cheb_clenshaw):
#   After the DCT-I sum, we set c_0 ← c_0/2 (i.e. c[1] *= 0.5) so that
#       f(t) ≈ c_0 + c_1 T_1(t) + … + c_M T_M(t)
# =============================================================================
@inline function _chebfit!(c::Vector{ComplexF64},f::Vector{ComplexF64})::Vector{ComplexF64}
    M=length(f)-1
    @inbounds for m in 0:M
        s=0.0+0.0im
        s+=0.5*f[1] # endpoint j = 0: cospi(0) = 1, w0 = 0.5
        s+=0.5*((isodd(m) ? -1.0 : 1.0)*f[M+1]) # endpoint j = M: cospi(m) = (-1)^m
        for j in 1:M-1
            s+=f[j+1]*cospi((j*m)/M)
        end
        c[m+1]=(2/M)*s
    end
    c[1]*=0.5
    return c
end

# ------------- H1x-only tables & plan -------------
struct ChebHankelTableH1x
    a::Float64 # start of panel 
    b::Float64 # end of panel
    M::Int # order of the chebyshev polynomial for panel [a,b]
    c1::Vector{ComplexF64}   # Chebyshev coeffs of G1(r) = √r * H1x(k r)
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
#   ChebHankelTableH1x(a,b,M,c1) where c1 are the Chebyshev coeffs of G1.
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

struct ChebHankelPlanH1x
    k::ComplexF64
    panels::Vector{ChebHankelTableH1x}
    rmin::Float64
    rmax::Float64
end

# =============================================================================
# Generate uniformly spaced panel breakpoints between rmin and rmax.
# The interval [rmin, rmax] is divided into np equal panels, returning
# np+1 breakpoints suitable for piecewise-Chebyshev interpolation.
#
# Inputs
#   rmin :: Float64    # lower bound (must be positive)
#   rmax :: Float64    # upper bound (must be > rmin)
#   np   :: Int        # number of panels
#
# Output
#   b :: Vector{Float64}   # length np+1, with b[1]=rmin, b[end]=rmax, and uniform spacing h = (rmax - rmin)/np
# =============================================================================
@inline function _breaks_uniform(rmin::Float64,rmax::Float64,np::Int)::Vector{Float64}
    h=(rmax-rmin)/np
    b=Vector{Float64}(undef,np+1)
    @inbounds for i in 0:np
        b[i+1]=muladd(i,h,rmin)
    end
    b[end]=rmax
    return b
end

# =============================================================================
# Generate geometrically graded panel breakpoints between rmin and rmax.
# The interval [rmin, rmax] is divided into np panels with exponentially
# increasing widths controlled by the ratio parameter.
#
# Panel i width ∝ ratio^(i-1), i=1..np
#
# Inputs
#   rmin  :: Float64    # lower bound (>0)
#   rmax  :: Float64    # upper bound (>rmin)
#   np    :: Int        # number of panels
#   ratio :: Float64=1.05 # geometric growth factor (>1 for expanding panels)
#
# Output
#   b :: Vector{Float64}  # length np+1, monotonically increasing, b[1]=rmin,
#                         # b[end]=rmax
#
# Implementation details
#   - Constructs relative widths `s[i]` as a geometric sequence.
#   - Normalizes total length by scaling with base = (rmax - rmin)/sum(s).
#   - Returns exact rmax at the last breakpoint to avoid accumulation error.
#   - The mild geometric grading helps resolve steep near-origin behavior
#     (e.g., oscillatory Bessel/Hankel terms) with fewer panels.
# =============================================================================
@inline function _breaks_geometric(rmin::Float64,rmax::Float64,np::Int;ratio::Float64=1.05)::Vector{Float64}
    s=ones(Float64,np)
    @inbounds for i in 2:np
        s[i]=s[i-1]*ratio
    end
    base=(rmax-rmin)/sum(s)
    b=Vector{Float64}(undef,np+1)
    b[1]=rmin
    @inbounds for i in 1:np
        b[i+1]=b[i]+base*s[i]
    end
    b[end]=rmax
    return b
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
    br=grading===:uniform ?
        _breaks_uniform(rmin,rmax,npanels) :
        _breaks_geometric(rmin,rmax,npanels;ratio=geo_ratio)
    panels=Vector{ChebHankelTableH1x}(undef,npanels)
    @inbounds Threads.@threads for i in 1:npanels
        panels[i]=_build_table_h1x!(k,br[i],br[i+1];M=M)
    end
    return ChebHankelPlanH1x(k,panels,rmin,rmax)
end

# =============================================================================
# Locate the panel index p such that panels[p].a ≤ r ≤ panels[p].b.
#
# This is a scalar binary search used to identify which Chebyshev panel
# contains the given radius `r`.  Each panel corresponds to one interval [a,b]
# on which the Chebyshev coefficients for G₁(r) = √r * H₁x(k r) were fitted.
#
# Inputs
#   panels :: Vector{ChebHankelTableH1x}   # vector of panel structs
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
#   - Uses a branchless integer binary search: O(log Np) comparisons.
#   - @inbounds avoids bounds checks for speed.
#   - Typical Np = 50–300, so this routine is negligible cost compared to Hankel evaluation.
# =============================================================================
@inline function _find_panel(panels::Vector{ChebHankelTableH1x},r::Float64)::Int
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
# Vectorized panel search: determine for each r[i] ∈ rvec which panel
# [a,b] of the Chebyshev plan `pl` contains it.
#
# This creates an Int32 vector of panel indices to enable later vectorized
# evaluation of H₁x or H₁.
#
# Inputs
#   pl    :: ChebHankelPlanH1x
#             # plan containing `panels::Vector{ChebHankelTableH1x}`
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
function panel_indices(pl::ChebHankelPlanH1x,rvec::AbstractVector{Float64})::Vector{Int32}
    p=similar(rvec,Int32)
    pans=pl.panels
    @inbounds Threads.@threads for i in eachindex(rvec)
        p[i]=_find_panel(pans,rvec[i])
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
#   pl     :: ChebHankelPlanH1x        # plan containing panels [a,b]
#   rvec   :: AbstractVector{Float64}  # radii (must lie in total range of plan)
#   pidx   :: AbstractVector{Int32}    # panel indices for each r[i]
#
# Outputs
#   (t, invsqrt)
#       t        :: Vector{Float64}     # mapped Chebyshev coordinates
#       invsqrt  :: Vector{Float64}     # 1/√r for normalization
# =============================================================================
function precompute_geom(pl::ChebHankelPlanH1x,rvec::AbstractVector{Float64},pidx::AbstractVector{Int32})::Tuple{Vector{Float64},Vector{Float64}}
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
#   pl   :: ChebHankelPlanH1x
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
function panel_and_geom(pl::ChebHankelPlanH1x,rvec::AbstractVector{Float64})::Tuple{Vector{Int32},Vector{Float64},Vector{Float64}}
    n=length(rvec)
    pidx=Vector{Int32}(undef,n)
    t=Vector{Float64}(undef,n)
    invsqrt=Vector{Float64}(undef,n)
    pans=pl.panels
    nP=length(pans)
    @inbounds Threads.@threads for i in eachindex(rvec)
        r=rvec[i]
        lo=1;hi=nP
        mid=0
        while lo<=hi
            mid=(lo+hi)>>>1
            P=pans[mid]
            if r<P.a
                hi=mid-1
            elseif r>P.b
                lo=mid+1
            else
                break # found panel = mid
            end
        end
        if !(pans[mid].a<=r<=pans[mid].b) # If not found, this will be out of range
            error("r=$r outside plan range [$(pans[1].a), $(pans[end].b)]")
        end
        P=pans[mid]
        pidx[i]=Int32(mid)
        t[i]=(2*r-(P.b+P.a))/(P.b-P.a)  # aff_map_inv
        invsqrt[i]=inv(sqrt(r))
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

# =============================================================================
# Fast evaluation of the *scaled* Hankel:
#   H1x(z) = exp(-i z) * H1^(1)(z),  with z = k*r.
#
# Storage model
# -------------
# Each panel stores Chebyshev coefficients of
#   G1(r) = √r * H1x(k r).
#
# With k-independent precomputations:
#   - pidx[i]   : which panel r_i belongs to
#   - t[i]      : mapped Chebyshev coordinate in [-1,1] on that panel
#   - invsqrt[i]: 1/√r_i
#
# we evaluate
#   H1x(k r_i) ≈ _cheb_clenshaw(c1_panel, t[i]) * invsqrt[i].
#
# Inputs
#   H1x     :: Vector{ComplexF64}        # output (length = length(t))
#   pl      :: ChebHankelPlanH1x         # plan at a *fixed* complex k
#   r       :: AbstractVector{Float64}   # radii (only used for indexing; can be omitted)
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
@inline function eval_h1x!(pl::ChebHankelPlanH1x,pidx::Int32,t::Float64,invsqrt::Float64)
    return _cheb_clenshaw(pl.panels[pidx].c1,t)*invsqrt
end

# =============================================================================
# Fast evaluation of the *unscaled* Hankel:
#   H1^(1)(k r) = exp(i k r) * H1x(k r).
#
# We reuse the same G1(r) tables (scaled representation) and multiply by the
# phase exp(i k r). For numerical stability at large |Im(k)|, pass precomputed
# 'phase[i] = exp(1im * k * r[i])'.
#
# Inputs
#   H1      :: Vector{ComplexF64}         # output
#   pl      :: ChebHankelPlanH1x          # plan at fixed k
#   pidx    :: AbstractVector{Int32}      # per-point panel index
#   t       :: AbstractVector{Float64}    # per-point Chebyshev coord
#   invsqrt :: AbstractVector{Float64}    # per-point 1/√r
#   phase   :: AbstractVector{ComplexF64} # exp(1im*k*r[i]) for the *same* k
#
# Output
#   Fills H1 in place.
# =============================================================================
function eval_h1!(H1::AbstractVector{ComplexF64},pl::ChebHankelPlanH1x,pidx::AbstractVector{Int32},t::AbstractVector{Float64},invsqrt::AbstractVector{Float64},phase::AbstractVector{ComplexF64})
    pans=pl.panels
    @inbounds Threads.@threads for i in eachindex(t)
        T=pans[pidx[i]]
        H1[i]=phase[i]*_cheb_clenshaw(T.c1,t[i])*invsqrt[i] # H1 = e^{ikr} * H1x
    end
    return nothing
end

# =============================================================================
# Evaluate the *unscaled* Hankel function
#       H1^(1)(k r)
# for a single point z = k*r using the scaled Chebyshev plan.
#
# The unscaled Hankel is reconstructed from the scaled one via
#       H1^(1)(k r) = exp(i k r) * H1x(k r)
#
# where H1x is obtained from the same Chebyshev table of
#       G1(r) = √r * H1x(k r)
# stored in the plan.  The exponential factor exp(i k r) (the *phase*) must
# be supplied externally to prevent recomputation and potential underflow for
# large Im(k).
#
# Inputs
#   pl       :: ChebHankelPlanH1x     # precomputed Chebyshev plan for given k
#   pidx     :: Int32                 # panel index such that r ∈ [a,b]
#   t        :: Float64               # mapped Chebyshev coordinate in [-1,1]
#   invsqrt  :: Float64               # 1 / √r
#   phase    :: ComplexF64 (keyword)  # precomputed exp(i * k * r)
#
# Output
#   ComplexF64                       # approximation of H1^(1)(k r)
# =============================================================================
@inline function eval_h1(pl::ChebHankelPlanH1x,pidx::Int32,t::Float64,invsqrt::Float64;phase::ComplexF64)
    return phase*_cheb_clenshaw(pl.panels[pidx].c1,t)*invsqrt
end

# =============================================================================
# Evaluate the *scaled* Hankel H1x(k r) for the *same radius r* across multiple
# complex wavenumbers (one per plan). Each plan must have been built over the
# *same panel breaks* so that `pidx` and `t` are valid for every plan.
#
# Storage model:
#   Each ChebHankelPlanH1x stores Chebyshev coeffs c1 for G1(r) = √r * H1x(k r).
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
# Evaluate the *unscaled* Hankel H₁^{(1)}(k r) for a fixed radius r across
# multiple complex wavenumbers (one per plan). Combines the scaled Chebyshev
# evaluation with the per-k phase:
#      H1(k r) = exp(i k r) * H1x(k r).
#
# Inputs
#   out    :: AbstractVector{ComplexF64}             # length == length(plans)
#   plans  :: AbstractVector{ChebHankelPlanH1x}      # k-dependent Cheb plans
#   r      :: Float64                                # distance for this (i,j)
#   pidx   :: Int32                                  # panel index for r
#   t      :: Float64                                # Chebyshev coord for r
#   invsqrt:: Float64                                # 1/√r
#
# Output
#   out[m] = H₁^{(1)}(k_m r) for m = 1..length(plans).
# =============================================================================
function eval_h1_multi_ks!(out::AbstractVector{ComplexF64},plans::AbstractVector{ChebHankelPlanH1x},r::Float64,pidx::Int32,t::Float64,invsqrt::Float64)
    @inbounds for m in eachindex(plans)
        plan_m=plans[m]
        out[m]=_exp_ikr(real(plan_m.k),imag(plan_m.k),r)*_cheb_clenshaw(plan_m.panels[pidx].c1,t)*invsqrt
    end
    return nothing
end


# =============================================================================
# Evaluate the *unscaled* Hankel H₁^{(1)}(k r) for a fixed radius r across
# multiple complex wavenumbers (one per plan). Combines the scaled Chebyshev
# evaluation with the per-k phase:
#      H1(k r) = exp(i k r) * H1x(k r).
#
# Inputs
#   out    :: AbstractVector{ComplexF64}             # length == length(plans)
#   plans  :: AbstractVector{ChebHankelPlanH1x}      # k-dependent Cheb plans
#   r      :: Float64                                # distance for this (i,j)
#   pidx   :: Int32                                  # panel index for r
#   t      :: Float64                                # Chebyshev coord for r
#   invsqrt:: Float64                                # 1/√r for that point
#   phases :: AbstractVector{ComplexF64}             # Phases exp(i k_m r) for each plan                            
#
# Output
#   out[m] = H₁^{(1)}(k_m r) for m = 1..length(plans).
# =============================================================================
function eval_h1_multi_ks!(out::AbstractVector{ComplexF64},plans::AbstractVector{ChebHankelPlanH1x},pidx::Int32,t::Float64,invsqrt::Float64;phases::AbstractVector{ComplexF64})
    @inbounds for m in eachindex(plans)
        out[m]=phases[m]*_cheb_clenshaw(plans[m].panels[pidx].c1,t)*invsqrt
    end
    return nothing
end