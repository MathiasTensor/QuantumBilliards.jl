#############################################################################
# Piecewise-Chebyshev special-function core for Helmholtz boundary integral
# operators in 2D.
#
# This file provides reusable plan construction and evaluation routines for
# Hankel and Bessel functions of the form
#
#     H_ν^(κ)(k r),   J_ν(k r),
#
# where `r ≥ 0` is a geometric distance and `k` may be real or complex.
#
# The main purpose is to separate:
#   1. geometric work        (distances, panel lookup, block caches),
#   2. special-function work (Hankel/Bessel evaluation),
# so that matrix assembly code can stay clean and only call a small evaluator
# API at the point where values are needed.
#
# ---------------------------------------------------------------------------
# DESIGN OVERVIEW
# ---------------------------------------------------------------------------
#
# The core idea is to approximate the distance dependence of the kernels on a
# fixed interval `[rmin, rmax]` by piecewise Chebyshev expansions. Since many
# matrix entries reuse the same distance-to-special-function map, this is much
# cheaper than repeatedly calling AMOS / Bessels.jl inside inner assembly loops.
#
# We support three related plan families:
#
# 1. ChebHankelPlanH1x
#    Complex-k scaled H1 plan for the legacy Beyn / EBIM route:
#
#        H1x(z) = exp(-i z) H_1^(1)(z),    z = k r.
#
#    On each panel we interpolate
#
#        G1(r) = sqrt(r) * H1x(k r),
#
#    because the `sqrt(r)` factor regularizes the small-r behavior and the
#    exponential scaling improves stability for complex arguments with nonzero
#    imaginary part.
#
# 2. ChebHankelPlanH
#    Unscaled Hankel plan for
#
#        H_ν^(κ)(k r),
#
#    used in the CFIE / DLP Kress and Alpert code paths. This is the standard
#    route when the physical unscaled Hankel values are needed directly.
#
# 3. ChebJPlan
#    Unscaled Bessel-J plan for
#
#        J_ν(k r),
#
#    needed especially in the Kress logarithmic split, where for complex `k`
#    one must not replace `J_ν` by `real(H_ν^(1))`.
#
# ---------------------------------------------------------------------------
# SMALL-ARGUMENT STRATEGY
# ---------------------------------------------------------------------------
#
# The singular / nearly singular regime near `z = k r = 0` is handled outside
# the Chebyshev interpolation itself.
#
# This file implements a layered strategy:
#
#   - for very small |z|:
#         use explicit small-argument series for H0 / H1,
#
#   - for a slightly larger near-zero window:
#         optionally use direct special-function evaluation,
#
#   - otherwise:
#         use the Chebyshev panel expansions.
#
# This avoids polluting the whole interpolation interval with excessively many
# tiny panels just to resolve the singular structure near zero. In practice this
# is especially important for:
#
#   - !!! Alpert self-correction nodes, where some off-grid distances can be very
#     small !!!
#
# In higher-level code, this is typically represented by:
#
#   pidx == 0
#
# meaning:
#   “this distance is outside the interpolated radial range and should be
#    evaluated through the low-z patch instead of panel interpolation.”
#
# That convention lets block caches and evaluators carry the switching logic,
# rather than pushing it into matrix assembly.
#
# ---------------------------------------------------------------------------
# PANELIZATION AND GEOMETRIC PRECOMPUTATION
# ---------------------------------------------------------------------------
#
# For interpolated distances, each radius `r` is mapped to:
#
#   - a panel index `pidx`,
#   - a local Chebyshev coordinate `t ∈ [-1,1]`,
#   - and, for the scaled H1x route, `invsqrt = 1/sqrt(r)`.
#
# These quantities are geometry-dependent but k-independent, so they can be
# precomputed once and then reused across:
#
#   - many contour points,
#   - many matrix entries with repeated geometry,
#   - derivative evaluations,
#   - multi-k assembly loops.
#
# This file therefore exposes both scalar and vectorized lookup helpers:
#
#   - _find_panel / panel_indices
#   - precompute_geom
#   - panel_and_geom
#
# Uniform panel layouts use O(1) arithmetic lookup, while nonuniform layouts
# use a binary search over panel endpoints.
#
# ---------------------------------------------------------------------------
# MULTI-k EVALUATION API
# ---------------------------------------------------------------------------
#
# A major use case is evaluating the same special function at one fixed distance
# `r`, but for many wavenumbers `k_m` simultaneously. This appears throughout:
#
#   - Beyn contour matrix construction,
#   - CFIE-Kress same-block assembly,
#   - CFIE-Alpert correction nodes,
#   - DLP-Kress derivative assembly.
#
# To support that efficiently, this file provides low-level multi-k evaluators
# which fill user-provided output buffers in place, for example:
#
#   - eval_h1x_multi_ks!
#   - eval_h_multi_ks!
#   - eval_j_multi_ks!
#   - h0_h1_multi_ks_at_r!
#   - h0_h1_j0_j1_multi_ks_at_r!
#   - h1_j1_multi_ks_at_r!
#   - h1_multi_ks_at_r!
#
# These routines are intentionally written as the “special-function layer” under
# the matrix constructors, so that future changes in the low-z strategy,
# interpolation cutoffs, or direct-evaluation policy can be made here without
# touching the assembly code.
#
# ---------------------------------------------------------------------------
# THREADING / PERFORMANCE NOTES
# ---------------------------------------------------------------------------
#
# Plan construction over panels is threaded.
# Vectorized lookup and evaluation kernels are also threaded where beneficial.
#
# In practice, the dominant runtime is usually controlled by:
#
#   - number of distance panels,
#   - Chebyshev degree per panel,
#   - number of special-function evaluations per matrix entry,
#
# rather than by the tiny extra branch used to distinguish:
#
#   interpolated region   vs   small-z patched region.
#
# The branch cost is negligible compared to a Hankel / Bessel evaluation or a
# Clenshaw recurrence, so keeping the switching logic in this low-level core is
# the correct tradeoff for maintainability.
#
# ---------------------------------------------------------------------------
# MAIN PUBLIC / SEMI-PUBLIC API
# ---------------------------------------------------------------------------
#
# Plan builders
#   - plan_h1x
#   - plan_h
#   - plan_j
#
# Panel / geometry helpers
#   - panel_indices
#   - precompute_geom
#   - panel_and_geom
#   - precompute_phase
#
# Scalar evaluators
#   - eval_h1x! / eval_h1x
#   - eval_h!   / eval_h
#   - eval_j!   / eval_j
#
# Multi-k evaluators
#   - eval_h1x_multi_ks!
#   - eval_h_multi_ks!
#   - eval_j_multi_ks!
#   - h0_h1_multi_ks_at_r!
#   - h0_h1_j0_j1_multi_ks_at_r!
#   - h1_j1_multi_ks_at_r!
#   - h1_multi_ks_at_r!
#   - h1_at_r
#
# Low-z support
#   - _small_h0_series
#   - _small_h1_series
#
# ---------------------------------------------------------------------------
# FILE ROLE IN THE LARGER CODEBASE
# ---------------------------------------------------------------------------
#
# This is the central special-function backend shared by:
#
#   - legacy EBIM / Beyn DLP code,
#   - CFIE-Kress code paths,
#   - DLP-Kress code paths,
#   - CFIE-Alpert code paths.
#
# The goal is that assembly files should not contain detailed logic about when
# to interpolate, when to use a series, or when to call direct special
# functions. They should only pass:
#
#   (pidx, t, r, invsqrt, plans, output buffers)
#
# and let this core decide how the function value is actually produced.
#
# MO/18/4/26
#############################################################################

const γ=MathConstants.eulergamma
const hankel_z_chebyshev_cutoff_small_z=0.001 # for z=k*r below this we use the small-argument series expansions for the Hankel functions instead of the Chebyshev evaluation, since the Chebyshev approximation is not accurate near zero due to the singularity. This is a bit hacky but it works and is fast since we only need to evaluate a few terms in the series expansion for small z. We can afford to be conservative here since this only affects a small portion of the domain near r=0, and we want to ensure high accuracy there.
const hankel_z_chebyshev_cutoff=0.1 # this is the region between [hankel_z_chebyshev_cutoff_small_z, hankel_z_chebyshev_cutoff] where we switch from the small-argument series expansions to the direct evaluation since only 12th degree small z poly.
#TODO Differentiation of Chebyshev polynomials to get other orders of hankels. Is this even good to do for performance, how much does it really matter?

struct ChebHankelTableH1x
    a::Float64 # start of panel 
    b::Float64 # end of panel
    M::Int # order of the chebyshev polynomial for panel [a,b]
    c1::Vector{ComplexF64}   # coeffs of Gν(r) = √r * Hνx(k r)
end

struct ChebHankelTableH
    a::Float64 # start of panel 
    b::Float64 # end of panel
    M::Int # order of the chebyshev polynomial for panel [a,b]
    ν::Int # degree of the Hankel type function
    κ::Int # order/type of the Hankel function 
    c::Vector{ComplexF64} 
end

struct ChebJTable
    a::Float64 # start of panel 
    b::Float64 # end of panel
    M::Int # order of the chebyshev polynomial for panel [a,b]
    ν::Int # degree of the J
    c::Vector{ComplexF64} 
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
        H1x=SpecialFunctions.besselhx(1,1,z) # scaled: exp(-i z) * H1^(1)(z) since AMOS provides this directly and is more stable than computing the unscaled version due to underflow/overflow issues with large/small |Im(z)|.
        f1[j+1]=sqrt(r)*H1x # G1(r) since for small r this is well behaved
    end
    c1=Vector{ComplexF64}(undef,M+1) # preallocate chebyshev coeffs
    _chebfit!(c1,f1) # fit the chebyshev coeffs to the chebyshev node evaluations
    return ChebHankelTableH1x(a,b,M,c1) # construct the table for that particular panel with local cheby polynomial 
end

# =============================================================================
# Construct a Chebyshev table on a single panel [a,b] for the unscaled Hankel:
#   Hν(z) = H_ν^(κ)(z),  with  z = k*r and k real.
#
# We approximate the function Fν(r) = H_ν^(κ)(k r) on Chebyshev nodes t_j = cos(π j / M), j=0..M, mapped to r_j ∈ [a,b].
#
# Unlike the complex-k scaled route, this real-k path stores the unscaled Hankel
# values directly.
#
# Inputs
#   ν :: Int                 # Hankel order (typically 0 or 1)
#   κ :: Int                 # Hankel type (1 or 2 etc.)
#   k :: Float64             # fixed real wavenumber
#   a,b :: Float64           # panel endpoints, 0 < a < b
#   M :: Int                 # Chebyshev degree (stores M+1 coeffs)
#
# Output
#   ChebHankelTableH(a,b,M,ν,c) where c are the Chebyshev coeffs of Fν.
# =============================================================================
function _build_table_h!(ν::Int,κ::Int,k::Float64,a::Float64,b::Float64;M::Int=16)::ChebHankelTableH
    @assert a>0 && b>a "a=$(a), b=$(b)" # sanity check to keep the interval bounded above 0 and to not be degenerate/reversed
    f1=Vector{ComplexF64}(undef,M+1) # preallocate the vector storing function evalutions at chebyshev nodes
    @inbounds for j in 0:M
        t=cospi(j/M) # chebyshev node in [-1,1]
        r=((b+a)+(b-a)*t)/2 # affine map to [a,b] so we are in the correct sector
        z=k*r # argument for Hankel in local coordinates
        f1[j+1]=ComplexF64(Bessels.hankelh(ν,κ,z)) # unscaled real-argument Hankel from Bessels.jl
    end
    c=Vector{ComplexF64}(undef,M+1) # preallocate chebyshev coeffs
    _chebfit!(c,f1) # fit the chebyshev coeffs to the chebyshev node evaluations
    return ChebHankelTableH(a,b,M,ν,κ,c) # construct the table for that particular panel with local cheby polynomial 
end

# multiple dispatch version of above with complex k. Maybe redundant since we have the scaled version but this one does not need the exp mul in the end and therefore faster. Other one is LEGACY made for original formulation for Beyn's method.
function _build_table_h!(ν::Int,κ::Int,k::ComplexF64,a::Float64,b::Float64;M::Int=16)::ChebHankelTableH
    @assert a>0 && b>a "a=$(a), b=$(b)" # sanity check to keep the interval bounded above 0 and to not be degenerate/reversed
    f1=Vector{ComplexF64}(undef,M+1) # preallocate the vector storing function evalutions at chebyshev nodes
    @inbounds for j in 0:M
        t=cospi(j/M) # chebyshev node in [-1,1]
        r=((b+a)+(b-a)*t)/2 # affine map to [a,b] so we are in the correct sector
        z=k*r # argument for Hankel in local coordinates
        f1[j+1]=SpecialFunctions.besselh(ν,κ,z) # unscaled complex-argument Hankel from AMOS. We use this for complex k since Bessels.jl doesn't support complex arguments and AMOS is more stable for complex arguments. Just dont use this here with large im part for z !!! 
        #FIXME It is hacky since for _build_table_h1x! we used the scaled version primarily for Beyn's method since we did not know the size of the imaginary part, but this seems not safe
    end
    c=Vector{ComplexF64}(undef,M+1) # preallocate chebyshev coeffs
    _chebfit!(c,f1) # fit the chebyshev coeffs to the chebyshev node evaluations
    return ChebHankelTableH(a,b,M,ν,κ,c) # construct the table for that particular panel with local cheby polynomial 
end

function _build_table_j!(ν::Int,k::Float64,a::Float64,b::Float64;M::Int=16)::ChebJTable
    @assert a>0 && b>a "a=$(a), b=$(b)" 
    f1=Vector{ComplexF64}(undef,M+1)
    @inbounds for j in 0:M
        t=cospi(j/M) 
        r=((b+a)+(b-a)*t)/2 
        z=k*r 
        f1[j+1]=Bessels.besselj(ν,z) 
    end
    c=Vector{ComplexF64}(undef,M+1) 
    _chebfit!(c,f1) 
    return ChebJTable(a,b,M,ν,c)
end

function _build_table_j!(ν::Int,k::ComplexF64,a::Float64,b::Float64;M::Int=16)::ChebJTable
    @assert a>0 && b>a "a=$(a), b=$(b)" 
    f1=Vector{ComplexF64}(undef,M+1)
    @inbounds for j in 0:M
        t=cospi(j/M) 
        r=((b+a)+(b-a)*t)/2 
        z=k*r 
        f1[j+1]=SpecialFunctions.besselj(ν,z)
    end
    c=Vector{ComplexF64}(undef,M+1) 
    _chebfit!(c,f1) 
    return ChebJTable(a,b,M,ν,c)
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

struct ChebHankelPlanH
    k::Union{Float64,ComplexF64}
    ν::Int
    κ::Int
    panels::Vector{ChebHankelTableH}
    rmin::Float64
    rmax::Float64
    grading::Symbol
    dr::Float64
    invdr::Float64
    npanels::Int
end

struct ChebJPlan
    k::Union{Float64,ComplexF64}
    ν::Int
    panels::Vector{ChebJTable}
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
#   κ :: Int
#       # Hankel type (1 for H^(1), 2 for H^(2), etc.)
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
#   ChebHankelPlanH(κ,k,ν,panels,...) containing the panelized Chebyshev tables.
# =============================================================================
function plan_h(ν::Int,κ::Int,k::Union{Float64,ComplexF64},rmin::Float64,rmax::Float64;npanels::Int=64,M::Int=16,grading::Symbol=:uniform,geo_ratio::Real=1.05)::ChebHankelPlanH
    @assert rmin>0 && rmax>rmin
    br= grading===:uniform ?
        _breaks_uniform(rmin,rmax,npanels) :
        _breaks_geometric(rmin,rmax,npanels;ratio=geo_ratio) # breakpoints for panels. These are the same for all k, so we can reuse them if we build multiple plans for different k's but the same geometry.
    panels=Vector{ChebHankelTableH}(undef,npanels)
    @inbounds Threads.@threads for i in 1:npanels
        panels[i]=_build_table_h!(ν,κ,k,br[i],br[i+1];M=M)
    end
    dr= grading===:uniform ? (rmax-rmin)/npanels : 0.0 # only used for uniform grading, geometric grading doesn't have a fixed panel width so we set dr=0 and invdr=0 
    invdr= grading===:uniform ? inv(dr) : 0.0
    return ChebHankelPlanH(k,ν,κ,panels,rmin,rmax,grading,dr,invdr,npanels)
end

# =============================================================================
# Build a piecewise-Chebyshev plan for the Bessel function J_ν(k r) over r ∈ [rmin,rmax].
#
# The interval is split into `npanels` either uniformly or with geometric growth. 
# Each panel stores M+1 Chebyshev coefficients (degree M) for the Bessel function values.
# This is intended for use in the CFIE when evaluating the interior Dirichlet Green’s function, which involves J_ν(k r).
# 
# Inputs
#   ν :: Int
#       # Bessel order need both 0 and 1 for CFIE
#   k :: Union{Float64,ComplexF64}
#       # fixed wavenumber (real or complex)
#   rmin,rmax :: Float64
#       # 0 < rmin < rmax
#   npanels :: Int
#   M :: Int
#   grading :: Symbol
#       # :uniform or :geometric
#   geo_ratio :: Real
#       # panel-size ratio for :geometric
# Output
#   ChebJPlan(k,ν,panels,...) containing the panelized Chebyshev tables for J_ν(k r).
# =============================================================================
function plan_j(ν::Int,k::Union{Float64,ComplexF64},rmin::Float64,rmax::Float64;npanels::Int=64,M::Int=16,grading::Symbol=:uniform,geo_ratio::Real=1.05)::ChebJPlan
    @assert rmin>0 && rmax>rmin
    br= grading===:uniform ?
        _breaks_uniform(rmin,rmax,npanels) :
        _breaks_geometric(rmin,rmax,npanels;ratio=geo_ratio) # breakpoints for panels. These are the same for all k, so we can reuse them if we build multiple plans for different k's but the same geometry.
    panels=Vector{ChebJTable}(undef,npanels)
    @inbounds Threads.@threads for i in 1:npanels
        panels[i]=_build_table_j!(ν,k,br[i],br[i+1];M=M)
    end
    dr= grading===:uniform ? (rmax-rmin)/npanels : 0.0 # only used for uniform grading, geometric grading doesn't have a fixed panel width so we set dr=0 and invdr=0 
    invdr= grading===:uniform ? inv(dr) : 0.0
    return ChebJPlan(k,ν,panels,rmin,rmax,grading,dr,invdr,npanels)
end

# =============================================================================
# Locate the panel index p such that panels[p].a ≤ r ≤ panels[p].b.
#
# This is a scalar binary search used to identify which Chebyshev panel
# contains the given radius `r`.  Each panel corresponds to one interval [a,b]
# on which the Chebyshev coefficients for G₁(r) = √r * H₁x(k r) were fitted.
#
# Inputs
#   panels :: Union{Vector{ChebHankelTableH1x},Vector{ChebHankelTableH},Vector{ChebJTable}}   # vector of panel structs
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
#   - Typical Np = 50–300, so this function is negligible cost compared to Hankel evaluation.
# =============================================================================
@inline function _find_panel_binary(panels::Union{Vector{ChebHankelTableH1x},Vector{ChebHankelTableH},Vector{ChebJTable}},r::Float64)::Int
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
#   pl :: Union{ChebHankelPlanH1x,ChebHankelPlanH,ChebJPlan}
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

@inline function _find_panel_uniform(pl::Union{ChebHankelPlanH1x,ChebHankelPlanH,ChebJPlan},r::Float64)::Int
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
#   pl :: Union{ChebHankelPlanH1x,ChebHankelPlanH,ChebJPlan}
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
@inline function _find_panel(pl::Union{ChebHankelPlanH1x,ChebHankelPlanH,ChebJPlan},r::Float64)::Int
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
#   pl    :: Union{ChebHankelPlanH1x,ChebHankelPlanH,ChebJPlan}
#             # plan containing `panels::Union{Vector{ChebHankelTableH1x},Vector{ChebHankelTableH},Vector{ChebJTable}` and grading information
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
function panel_indices(pl::Union{ChebHankelPlanH1x,ChebHankelPlanH,ChebJPlan},rvec::AbstractVector{Float64})::Vector{Int32}
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
#   pl     :: Union{ChebHankelPlanH1x,ChebHankelPlanH,ChebJPlan}  # plan containing panels [a,b]
#   rvec   :: AbstractVector{Float64}  # radii (must lie in total range of plan)
#   pidx   :: AbstractVector{Int32}    # panel indices for each r[i]
#
# Outputs
#   (t, invsqrt)
#       t        :: Vector{Float64}     # mapped Chebyshev coordinates
#       invsqrt  :: Vector{Float64}     # 1/√r for normalization
# =============================================================================
function precompute_geom(pl::Union{ChebHankelPlanH1x,ChebHankelPlanH,ChebJPlan},rvec::AbstractVector{Float64},pidx::AbstractVector{Int32})::Tuple{Vector{Float64},Vector{Float64}}
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
#   pl   :: Union{ChebHankelPlanH1x,ChebHankelPlanH,ChebJPlan}  # Chebyshev plan with panels and grading info
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
function panel_and_geom(pl::Union{ChebHankelPlanH1x,ChebHankelPlanH,ChebJPlan},rvec::AbstractVector{Float64})::Tuple{Vector{Int32},Vector{Float64},Vector{Float64}}
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
############## NEAR 0 EXPANSIONS FOR H0 AND H1 ###################
##################################################################

# For small arguments, the Hankel functions can be approximated by their series expansions. These are used to handle near-singular cases where the argument z = k*r is close to zero, which can cause numerical instability in the chebyshev evaluation of the Hankel functions.
# Up to O(z^12) for both, hopefully with defaul cutoff 1e-3 this is good enough for near machine precision.

@inline function _small_h0_series(z::ComplexF64)
    zz=z*z
    P=2123366400+zz*(-530841600+zz*(33177600+zz*(-921600+zz*(14400+zz*(-144+zz)))))
    Q=10616832000+zz*(-995328000+zz*(33792000+zz*(-600000+zz*(6576+zz*(-49)))))
    return (((10*pi+20*im*γ)*P+im*zz*Q)/(21233664000*pi))+(im*P/(1061683200*pi))*log(z/2)
end

@inline function _small_h0_series(z::T) where T<:Number
    return _small_h0_series(ComplexF64(z))
end

@inline function _small_h1_series(z::ComplexF64)
    zz=z*z
    A=-4161798144000+
    zz*(1040449536000*(-1+2*γ-1im*pi)+
    zz*(-65028096000*(-5+4*γ-2im*pi)+
    zz*(1806336000*(-10+6*γ-3im*pi)+
    zz*(-9408000*(-47+24*γ-12im*pi)+
    zz*(47040*(-131+60*γ-30im*pi)+
    zz*(-784*(-71+30*γ-15im*pi)+
    zz*(-353+140*γ-70im*pi)))))))
    R=14863564800+zz*(-1857945600+zz*(77414400+zz*(-1612800+zz*(20160+zz*(-168+zz)))))
    return (im*A/(2080899072000*pi*z))+(im*z*R/(14863564800*pi))*log(z/2)
end

@inline function _small_h1_series(z::T) where T<:Number
    return _small_h1_series(ComplexF64(z))
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
    k=pl.k
    @inbounds Threads.@threads for i in eachindex(r)
        z=k*r[i]
        if pidx[i]==0
            if abs(z)<hankel_z_chebyshev_cutoff_small_z
                H1x[i]=exp(-1*im*z)*_small_h1_series(z)
            else
                H1x[i]=SpecialFunctions.besselhx(1,1,z)
            end
        else
            T=pans[pidx[i]]
            g1=_cheb_clenshaw(T.c1,t[i])
            H1x[i]=g1*invsqrt[i]
        end
    end
    return nothing
end

# =============================================================================
# Fast evaluation of the unscaled Hankel for the real-k plan:
#   H_ν^(κ)(k r),  with k real.
#
# Each panel stores Chebyshev coefficients of the direct unscaled Hankel
# values on that interval. No exponential scaling and no 1/sqrt(r)
# normalization are used in this real-k route.
#
# Inputs
#   H1   :: AbstractVector{ComplexF64}
#           # output
#   pl   :: ChebHankelPlanH
#           # plan at fixed real k
#   r    :: AbstractVector{Float64}
#           # radii (unused; kept for API compatibility)
#   pidx :: AbstractVector{Int32}
#           # per-point panel index
#   t    :: AbstractVector{Float64}
#           # per-point Chebyshev coordinate
#
# Output
#   Fills H1 in place with H_ν^(κ)(k r_i).
# =============================================================================
function eval_h!(H1::AbstractVector{ComplexF64},pl::ChebHankelPlanH,r::AbstractVector{Float64},pidx::AbstractVector{Int32},t::AbstractVector{Float64})
    pans=pl.panels
    k=ComplexF64(pl.k)
    ν=pl.ν
    @inbounds Threads.@threads for i in eachindex(r)
        z=k*r[i]
        if pidx[i]==0
            if ν==0
                if abs(z)<hankel_z_chebyshev_cutoff_small_z
                    H1[i]=_small_h0_series(z)
                else
                    H1[i]=SpecialFunctions.besselh(0,pl.κ,z)
                end
            elseif ν==1
                if abs(z)<hankel_z_chebyshev_cutoff_small_z
                    H1[i]=_small_h1_series(z)
                else
                    H1[i]=SpecialFunctions.besselh(1,pl.κ,z)
                end
            else
                H1[i]=SpecialFunctions.besselh(ν,pl.κ,z)
            end
        else
            T=pans[pidx[i]]
            H1[i]=_cheb_clenshaw(T.c,t[i])
        end
    end
    return nothing
end

# =============================================================================
# Fast evaluation of the Bessel function J_ν(k r) for the same radius r across
# multiple real or complex wavenumbers (one per plan).
#
# Each plan stores the direct unscaled Bessel values on the same panel
# partition, so the same `pidx` and `t` may be reused across all plans.
#
# Inputs
#   J     :: AbstractVector{ComplexF64}
#           # output vector (length = length(plans))   
#   pl    :: ChebJPlan
#           # Chebyshev plan for J_ν(k r) at fixed k
#   r     :: AbstractVector{Float64}
#           # radii (unused; kept for API compatibility)
#   pidx  :: AbstractVector{Int32}
#           # per-point panel index
#   t     :: AbstractVector{Float64}
#           # per-point Chebyshev coordinate
# Output
#   Fills J in place with J_ν(k r_i) for each plan.
# =============================================================================
function eval_j!(J::AbstractVector{ComplexF64},pl::ChebJPlan,r::AbstractVector{Float64},pidx::AbstractVector{Int32},t::AbstractVector{Float64})
    pans=pl.panels
    ν=pl.ν
    k=ComplexF64(pl.k)
    @inbounds Threads.@threads for i in eachindex(r)
        if pidx[i]==0
            J[i]=SpecialFunctions.besselj(ν,k*r[i])
        else
            T=pans[pidx[i]]
            J[i]=_cheb_clenshaw(T.c,t[i])
        end
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
    r=1/invsqrt^2
    z=pl.k*r
    if pidx==0
        if abs(z)<hankel_z_chebyshev_cutoff_small_z
            return exp(-1*im*z)*_small_h1_series(z)
        else
            return SpecialFunctions.besselhx(1,1,z)
        end
    end
    return _cheb_clenshaw(pl.panels[pidx].c1,t)*invsqrt
end

# =============================================================================
# Evaluate the unscaled Hankel
#   H_ν^(κ)(k r),  with k real,
# at a single point using the real-k Chebyshev plan.
#
# Inputs
#   pl   :: ChebHankelPlanH
#           # precomputed Chebyshev plan for fixed real k
#   pidx :: Int32
#           # panel index such that r ∈ [a,b]
#   t    :: Float64
#           # mapped Chebyshev coordinate in [-1,1]
#   r    :: Float64
#           # radius for this evaluation point - for small z asymptotics
# Output
#   ComplexF64
#           # approximation of H_ν^(κ)(k r)
# =============================================================================
@inline function eval_h(pl::ChebHankelPlanH,pidx::Int32,t::Float64,r::Float64)
    z=ComplexF64(pl.k)*r
    if pidx==0
        if pl.ν==0
            if abs(z)<hankel_z_chebyshev_cutoff_small_z
                return _small_h0_series(z)
            else
                return SpecialFunctions.besselh(0,pl.κ,z)
            end
        elseif pl.ν==1
            if abs(z)<hankel_z_chebyshev_cutoff_small_z
                return _small_h1_series(z)
            else
                return SpecialFunctions.besselh(1,pl.κ,z)
            end
        else
            return SpecialFunctions.besselh(pl.ν,pl.κ,z)
        end
    end
    return _cheb_clenshaw(pl.panels[pidx].c,t)
end

# =============================================================================
# Evaluate the Bessel function J_ν(k r) for a single point using the Chebyshev plan for J.
#
# Inputs
#   pl   :: ChebJPlan
#           # precomputed Chebyshev plan for J_ν(k r) at fixed k
#   pidx :: Int32
#           # panel index such that r ∈ [a,b]
#   t    :: Float64
#           # mapped Chebyshev coordinate in [-1,1]
# Output
#   ComplexF64
#           # approximation of J_ν(k r)
# =============================================================================
@inline function eval_j(pl::ChebJPlan,pidx::Int32,t::Float64,r::Float64)
    if pidx==0
        return SpecialFunctions.besselj(pl.ν,ComplexF64(pl.k)*r)
    end
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
    r=1/invsqrt^2
    @inbounds for m in eachindex(plans)
        z=plans[m].k*r
        if pidx==0
            if abs(z)<hankel_z_chebyshev_cutoff_small_z
                out[m]=exp(-1*im*z)*_small_h1_series(z)
            else
                out[m]=SpecialFunctions.besselhx(1,1,z)
            end
        else
            out[m]=_cheb_clenshaw(plans[m].panels[pidx].c1,t)*invsqrt
        end
    end
    return nothing
end

# =============================================================================
# Evaluate the unscaled Hankel H_ν^(κ)(k r) for the same radius r across
# multiple real wavenumbers (one per plan).
#
# Each plan stores the direct unscaled Hankel coefficients on the same panel
# partition, so the same `pidx` and `t` may be reused across all plans.
#
# Inputs
#   out  :: AbstractVector{ComplexF64}
#           # length == length(plans)
#   plans:: AbstractVector{ChebHankelPlanH}
#           # real-k Chebyshev plans
#   r    :: Float64
#           # radius for this evaluation point (unused; kept for API symmetry)
#   pidx :: Int32
#           # panel index for r
#   t    :: Float64
#           # Chebyshev coordinate for r
#
# Output
#   out[m] = H_ν^(κ)(k_m r) for m = 1..length(plans).
# =============================================================================
function eval_h_multi_ks!(out::AbstractVector{ComplexF64},plans::AbstractVector{ChebHankelPlanH},r::Float64,pidx::Int32,t::Float64)
    @inbounds for m in eachindex(plans)
        plan_m=plans[m]
        z=ComplexF64(plan_m.k)*r
        if pidx==0
            if plan_m.ν==0
                if abs(z)<hankel_z_chebyshev_cutoff_small_z
                    out[m]=_small_h0_series(z)
                else
                    out[m]=SpecialFunctions.besselh(0,plan_m.κ,z)
                end
            elseif plan_m.ν==1
                if abs(z)<hankel_z_chebyshev_cutoff_small_z
                    out[m]=_small_h1_series(z)
                else
                    out[m]=SpecialFunctions.besselh(1,plan_m.κ,z)
                end
            else
                out[m]=SpecialFunctions.besselh(plan_m.ν,plan_m.κ,z)
            end
        else
            out[m]=_cheb_clenshaw(plan_m.panels[pidx].c,t)
        end
    end
    return nothing
end

# =============================================================================
# Evaluate the Bessel function J_ν(k r) for the same radius r across multiple
# wavenumbers (one per plan).
#
# Each plan stores the direct unscaled Bessel coefficients on the same panel
# partition, so the same `pidx` and `t` may be reused across all plans.
#
# Inputs
#   out  :: AbstractVector{ComplexF64}
#           # length == length(plans)
#   plans:: AbstractVector{ChebJPlan}
#           # Chebyshev plans for J_ν(k r) at fixed k
#   r    :: Float64
#           # radius for this evaluation point (unused; kept for API symmetry)
#   pidx :: Int32
#           # panel index for r
#   t    :: Float64
#           # Chebyshev coordinate for r
# Output
#   out[m] = J_ν(k_m r) for m = 1..length(plans).
# =============================================================================
function eval_j_multi_ks!(out::AbstractVector{ComplexF64},plans::AbstractVector{ChebJPlan},r::Float64,pidx::Int32,t::Float64)
    @inbounds for m in eachindex(plans)
        plan_m=plans[m]
        if pidx==0
            out[m]=SpecialFunctions.besselj(plan_m.ν,ComplexF64(plan_m.k)*r)
        else
            out[m]=_cheb_clenshaw(plan_m.panels[pidx].c,t)
        end
    end
    return nothing
end

############################################################
##################### HIGH LEVEL API #######################
############################################################

"""
    h0_h1_j0_j1_multi_ks_at_r!(h0vals,h1vals,j0vals,j1vals,plans0,plans1,plansj0,plansj1,pidx,t)

Evaluate `H₀^(1)`, `H₁^(1)`, `J₀`, and `J₁` for all wavenumbers at one fixed
distance panel/location, writing the results in place.

This function is the small inner evaluator used by the multi-`k` same-component
CFIE-Kress assembly. The geometric pair `(i,j)` has already been mapped to:
- a Chebyshev panel index `pidx`,
- a local coordinate `t ∈ [-1,1]`.

For each wavenumber `k_m`, the function interpolates:
- `H₀^(1)(k_m r)`
- `H₁^(1)(k_m r)`
- `J₀(k_m r)`
- `J₁(k_m r)`

by evaluating the corresponding Chebyshev expansions with Clenshaw recurrence.
For real wavenumbers one may sometimes identify `J_n` with `real(H_n^(1))`,
but for complex `k` that is not valid. The Kress split formulas genuinely
require `J₀` and `J₁`, so they are interpolated separately.

# Arguments
- `h0vals::AbstractVector{ComplexF64}`:
  Output vector for the `H₀^(1)` values.
- `h1vals::AbstractVector{ComplexF64}`:
  Output vector for the `H₁^(1)` values.
- `j0vals::AbstractVector{ComplexF64}`:
  Output vector for the `J₀` values.
- `j1vals::AbstractVector{ComplexF64}`:
  Output vector for the `J₁` values.
- `plans0::AbstractVector{ChebHankelPlanH}`:
  Chebyshev plans for `H₀^(1)`.
- `plans1::AbstractVector{ChebHankelPlanH}`:
  Chebyshev plans for `H₁^(1)`.
- `plansj0::AbstractVector{ChebJPlan}`:
  Chebyshev plans for `J₀`.
- `plansj1::AbstractVector{ChebJPlan}`:
  Chebyshev plans for `J₁`.
- `pidx::Int32`:
  Panel index containing the current distance.
- `t::Float64`:
  Local Chebyshev coordinate in that panel.

# Returns
- `nothing`
"""
@inline function h0_h1_j0_j1_multi_ks_at_r!(h0vals::AbstractVector{ComplexF64},h1vals::AbstractVector{ComplexF64},j0vals::AbstractVector{ComplexF64},j1vals::AbstractVector{ComplexF64},plans0::AbstractVector{ChebHankelPlanH},plans1::AbstractVector{ChebHankelPlanH},plansj0::AbstractVector{ChebJPlan},plansj1::AbstractVector{ChebJPlan},pidx::Int32,t::Float64,r::Float64)
    @inbounds for m in eachindex(plans0)
        z=ComplexF64(plans0[m].k)*r
        az=abs(z)
        if az<hankel_z_chebyshev_cutoff_small_z
            h0vals[m]=_small_h0_series(z)
            h1vals[m]=_small_h1_series(z)
            j0vals[m]=SpecialFunctions.besselj(0,z)
            j1vals[m]=SpecialFunctions.besselj(1,z)
        elseif az<hankel_z_chebyshev_cutoff || pidx==0
            h0vals[m]=SpecialFunctions.besselh(0,1,z)
            h1vals[m]=SpecialFunctions.besselh(1,1,z)
            j0vals[m]=SpecialFunctions.besselj(0,z)
            j1vals[m]=SpecialFunctions.besselj(1,z)
        else
            h0vals[m]=_cheb_clenshaw(plans0[m].panels[pidx].c,t)
            h1vals[m]=_cheb_clenshaw(plans1[m].panels[pidx].c,t)
            j0vals[m]=_cheb_clenshaw(plansj0[m].panels[pidx].c,t)
            j1vals[m]=_cheb_clenshaw(plansj1[m].panels[pidx].c,t)
        end
    end
    return nothing
end

"""
    h0_h1_multi_ks_at_r!(h0vals,h1vals,plans0,plans1,pidx,t)

Evaluate `H₀^(1)` and `H₁^(1)` for all wavenumbers at one fixed distance
panel/location, writing the results in place.

This is the reduced special-function evaluator used in off-component CFIE-Kress
blocks, where the kernel is smooth and no Kress logarithmic split is needed.
Since the smooth inter-component assembly uses only the Hankel terms, the Bessel
`J₀/J₁` values are not required.

# Arguments
- `h0vals::AbstractVector{ComplexF64}`:
  Output vector for the `H₀^(1)` values.
- `h1vals::AbstractVector{ComplexF64}`:
  Output vector for the `H₁^(1)` values.
- `plans0::AbstractVector{ChebHankelPlanH}`:
  Chebyshev plans for `H₀^(1)`.
- `plans1::AbstractVector{ChebHankelPlanH}`:
  Chebyshev plans for `H₁^(1)`.
- `pidx::Int32`:
  Panel index for the active distance.
- `t::Float64`:
  Local Chebyshev coordinate in that panel.

# Returns
- `nothing`
"""
@inline function h0_h1_multi_ks_at_r!(h0vals::AbstractVector{ComplexF64},h1vals::AbstractVector{ComplexF64},plans0::AbstractVector{ChebHankelPlanH},plans1::AbstractVector{ChebHankelPlanH},pidx::Int32,t::Float64,r::Float64)
    @inbounds for m in eachindex(plans0)
        z=ComplexF64(plans0[m].k)*r
        az=abs(z)
        if az<hankel_z_chebyshev_cutoff_small_z
            h0vals[m]=_small_h0_series(z)
            h1vals[m]=_small_h1_series(z)
        elseif az<hankel_z_chebyshev_cutoff || pidx==0
            h0vals[m]=SpecialFunctions.besselh(0,1,z)
            h1vals[m]=SpecialFunctions.besselh(1,1,z)
        else
            h0vals[m]=_cheb_clenshaw(plans0[m].panels[pidx].c,t)
            h1vals[m]=_cheb_clenshaw(plans1[m].panels[pidx].c,t)
        end
    end
    return nothing
end

"""
        h1_j1_multi_ks_at_r!(h1vals,j1vals,plans1,plansj1,pidx,t,r)

Evaluate `H₁^(1)` and `J₁` for all wavenumbers at one fixed distance panel/location, writing the results in place.
This is the reduced special-function evaluator used in off-component CFIE-Kress blocks, where the kernel is smooth and no Kress logarithmic split is needed. Since the smooth inter-component assembly uses only the Hankel terms, the Bessel `J₀` values are not required.

If `pidx==0`, the evaluation point is close to zero and the small-argument series expansions are used for all wavenumbers. Otherwise, the Chebyshev expansions are evaluated with Clenshaw recurrence. This is to avoid too many panels, which would cause unnecessary overhead in the Chebyshev evaluation and also workspace construction.

# Arguments
- `h1vals::AbstractVector{ComplexF64}`:
  Output vector for the `H₁^(1)` values.
- `j1vals::AbstractVector{ComplexF64}`:
  Output vector for the `J₁` values.
- `plans1::AbstractVector{ChebHankelPlanH}`:
  Chebyshev plans for `H₁^(1)`.
- `plansj1::AbstractVector{ChebJPlan}`:
  Chebyshev plans for `J₁`.
- `pidx::Int32`:
  Panel index for the active distance.
- `t::Float64`:
  Local Chebyshev coordinate in that panel.
- `r::Float64`:
  Distance at which to evaluate the functions.

# Returns
- `nothing`
"""
@inline function h1_j1_multi_ks_at_r!(h1vals::AbstractVector{ComplexF64},j1vals::AbstractVector{ComplexF64},plans1::AbstractVector{ChebHankelPlanH},plansj1::AbstractVector{ChebJPlan},pidx::Int32,t::Float64,r::Float64)
    @inbounds for m in eachindex(plans1)
        z=ComplexF64(plans1[m].k)*r
        az=abs(z)
        if az<hankel_z_chebyshev_cutoff_small_z
            h1vals[m]=_small_h1_series(z)
            j1vals[m]=SpecialFunctions.besselj(1,z)
        elseif az<hankel_z_chebyshev_cutoff || pidx==0
            h1vals[m]=SpecialFunctions.hankelh1(1,z)
            j1vals[m]=SpecialFunctions.besselj(1,z)
        else
            h1vals[m]=_cheb_clenshaw(plans1[m].panels[pidx].c,t)
            j1vals[m]=_cheb_clenshaw(plansj1[m].panels[pidx].c,t)
        end
    end
    return nothing
end

"""
    h0_h1_h2_at_r(plan0,plan1,pidx,t,r)

Evaluate `H₀^(1)`, `H₁^(1)`, and `H₂^(1)` at one fixed distance for a single
wavenumber, returning the values directly.

This is the small scalar evaluator used by the single-`k` derivative DLP
Chebyshev pathway. The geometric pair `(i,j)` has already been mapped to:
- a Chebyshev panel index `pidx`,
- a local coordinate `t ∈ [-1,1]`,
- and the physical distance `r`.

For the interpolated regime, the function evaluates:
- `H₀^(1)(k r)` from `plan0`,
- `H₁^(1)(k r)` from `plan1`,

and then recovers `H₂^(1)(k r)` by the recurrence

    H₂^(1)(z) = (2/z) H₁^(1)(z) - H₀^(1)(z).

Near zero, the function switches to the small-argument series or direct special-
function evaluation.

# Arguments
- `plan0::ChebHankelPlanH`:
  Chebyshev plan for `H₀^(1)`.
- `plan1::ChebHankelPlanH`:
  Chebyshev plan for `H₁^(1)`.
- `pidx::Int32`:
  Panel index containing the current distance. If `pidx==0`, the evaluation is
  taken from the near-zero patch instead of the Chebyshev panels.
- `t::Float64`:
  Local Chebyshev coordinate in the active panel.
- `r::Float64`:
  Distance at which to evaluate the Hankel functions.

# Returns
- `(H0,H1,H2)::Tuple{ComplexF64,ComplexF64,ComplexF64}`
"""
@inline function h0_h1_h2_at_r(plan0::ChebHankelPlanH,plan1::ChebHankelPlanH,pidx::Int32,t::Float64,r::Float64)
    z=ComplexF64(plan0.k)*r
    az=abs(z)
    if az<hankel_z_chebyshev_cutoff_small_z
        H0=_small_h0_series(z)
        H1=_small_h1_series(z)
        H2=SpecialFunctions.besselh(2,1,z)
    elseif az<hankel_z_chebyshev_cutoff||pidx==0
        H0=SpecialFunctions.besselh(0,1,z)
        H1=SpecialFunctions.besselh(1,1,z)
        H2=SpecialFunctions.besselh(2,1,z)
    else
        H0=_cheb_clenshaw(plan0.panels[pidx].c,t)
        H1=_cheb_clenshaw(plan1.panels[pidx].c,t)
        H2=(2/z)*H1-H0
    end
    return H0,H1,H2
end

"""
    h0_h1_h2_multi_ks_at_r!(h0vals,h1vals,h2vals,plans0,plans1,pidx,t,r)

Evaluate `H₀^(1)`, `H₁^(1)`, and `H₂^(1)` for all wavenumbers at one fixed
distance panel/location, writing the results in place.

This is the reduced special-function evaluator used by the multi-`k` derivative
DLP Chebyshev assembly. The geometric pair `(i,j)` has already been mapped to:
- a Chebyshev panel index `pidx`,
- a local coordinate `t ∈ [-1,1]`,
- and the physical distance `r`.

For each wavenumber `k_m`, the function evaluates:
- `H₀^(1)(k_m r)`,
- `H₁^(1)(k_m r)`,

and then recovers `H₂^(1)(k_m r)` by the recurrence

    H₂^(1)(z) = (2/z) H₁^(1)(z) - H₀^(1)(z),

except in the small-argument regime, where it switches to series or direct
evaluation to avoid loss of accuracy.

# Arguments
- `h0vals::AbstractVector{ComplexF64}`:
  Output vector for the `H₀^(1)` values.
- `h1vals::AbstractVector{ComplexF64}`:
  Output vector for the `H₁^(1)` values.
- `h2vals::AbstractVector{ComplexF64}`:
  Output vector for the `H₂^(1)` values.
- `plans0::AbstractVector{ChebHankelPlanH}`:
  Chebyshev plans for `H₀^(1)`.
- `plans1::AbstractVector{ChebHankelPlanH}`:
  Chebyshev plans for `H₁^(1)`.
- `pidx::Int32`:
  Panel index containing the current distance. If `pidx==0`, the evaluation is
  taken from the near-zero patch instead of the Chebyshev panels.
- `t::Float64`:
  Local Chebyshev coordinate in the active panel.
- `r::Float64`:
  Distance at which to evaluate the Hankel functions.

# Returns
- `nothing`
"""
@inline function h0_h1_h2_multi_ks_at_r!(h0vals::AbstractVector{ComplexF64},h1vals::AbstractVector{ComplexF64},h2vals::AbstractVector{ComplexF64},
    plans0::AbstractVector{ChebHankelPlanH},plans1::AbstractVector{ChebHankelPlanH},pidx::Int32,t::Float64,r::Float64)
    @inbounds for m in eachindex(plans0)
        z=ComplexF64(plans0[m].k)*r
        az=abs(z)
        if az<hankel_z_chebyshev_cutoff_small_z
            h0vals[m]=_small_h0_series(z)
            h1vals[m]=_small_h1_series(z)
            h2vals[m]=SpecialFunctions.besselh(2,1,z)
        elseif az<hankel_z_chebyshev_cutoff||pidx==0
            h0vals[m]=SpecialFunctions.besselh(0,1,z)
            h1vals[m]=SpecialFunctions.besselh(1,1,z)
            h2vals[m]=SpecialFunctions.besselh(2,1,z)
        else
            h0vals[m]=_cheb_clenshaw(plans0[m].panels[pidx].c,t)
            h1vals[m]=_cheb_clenshaw(plans1[m].panels[pidx].c,t)
            h2vals[m]=(2/z)*h1vals[m]-h0vals[m]
        end
    end
    return nothing
end

"""
        h1_multi_ks_at_r!(h1vals,phases,plans,pidx,t,invsqrt,r,ab)

Evaluate `H₁^(1)` for all wavenumbers at one fixed distance panel/location, writing the results in place. Only used for standard DLP kernel where we need only H₁^(1) and not J₁, but we want to use the scaled Chebyshev expansions for complex k. The `phases` vector is used to store the exp(-i k r) factors for each wavenumber, which can be reused across multiple evaluations at the same radius.

If `pidx==0`, the evaluation point is close to zero and the small-argument series expansion is used for all wavenumbers. Otherwise, the Chebyshev expansions are evaluated with Clenshaw recurrence. This is to avoid too many panels, which would cause unnecessary overhead in the Chebyshev evaluation and also workspace construction.

# Arguments
- `h1vals::AbstractVector{ComplexF64}`:
  Output vector for the `H₁^(1)` values.
- `phases::AbstractVector{ComplexF64}`:
  Workspace vector for the `exp(-i k r)` factors for each wavenumber.
- `plans::AbstractVector{ChebHankelPlanH1x}`:
  Chebyshev plans for the scaled Hankel `H1x`.
- `pidx::Int32`:
  Panel index for the active distance.
- `t::Float64`:
  Local Chebyshev coordinate in that panel.
- `invsqrt::Float64`:
  1 / √r for normalization.
- `r::Float64`:
  Distance at which to evaluate the functions.
- `ab::AbstractVector{NTuple{2,Float64}}`:
  Real and Imag parts of k for each plan, pre-extracted.

# Returns
- `nothing`
"""
@inline function h1_multi_ks_at_r!(hvals::AbstractVector{ComplexF64},phases::AbstractVector{ComplexF64},plans::AbstractVector{ChebHankelPlanH1x},pidx::Int32,t::Float64,invsqrt::Float64,r::Float64,ab::AbstractVector{NTuple{2,Float64}})
    @inbounds for m in eachindex(plans)
        a,b=ab[m]
        phases[m]=_exp_ikr(a,b,r)
        z=ComplexF64(plans[m].k)*r
        az=abs(z)
        if az<hankel_z_chebyshev_cutoff_small_z
            hvals[m]=phases[m]*_small_h1_series(z)
        elseif az<hankel_z_chebyshev_cutoff || pidx==0
            hvals[m]=phases[m]*SpecialFunctions.hankelh1(1,z)
        else
            hvals[m]=phases[m]*_cheb_clenshaw(plans[m].panels[pidx].c1,t)*invsqrt
        end
    end
    return nothing
end

"""
    h1_at_r(plan,pidx,t,invsqrt,r,a,b)

Evaluate `H₁^(1)` at one fixed distance for a single complex wavenumber using
the scaled-Hankel Chebyshev plan, returning the unscaled value directly.

    H1x(z) = exp(-i z) H₁^(1)(z),

with `z = k r`.

For the interpolated regime, the function evaluates the stored Chebyshev
approximation to `H1x(k r)` and then restores the unscaled Hankel value by
multiplying with

    exp(i k r).

Near zero, it switches to the small-argument series or direct special-function
evaluation.

# Arguments
- `plan::ChebHankelPlanH1x`:
  Chebyshev plan for the scaled Hankel function `H1x`.
- `pidx::Int32`:
  Panel index containing the current distance. If `pidx==0`, the evaluation is
  taken from the near-zero patch instead of the Chebyshev panels.
- `t::Float64`:
  Local Chebyshev coordinate in the active panel.
- `invsqrt::Float64`:
  The factor `1/sqrt(r)` used to undo the radial regularization stored in the
  plan coefficients.
- `r::Float64`:
  Distance at which to evaluate the Hankel function.
- `a::Float64`:
  Real part of the complex wavenumber `k`.
- `b::Float64`:
  Imaginary part of the complex wavenumber `k`.

# Returns
- `ComplexF64`
"""
@inline function h1_at_r(plan::ChebHankelPlanH1x,pidx::Int32,t::Float64,invsqrt::Float64,r::Float64,a::Float64,b::Float64)
    phase=_exp_ikr(a,b,r)
    z=ComplexF64(plan.k)*r
    az=abs(z)
    if az<hankel_z_chebyshev_cutoff_small_z
        return phase*_small_h1_series(z)
    elseif az<hankel_z_chebyshev_cutoff || pidx==0
        return phase*SpecialFunctions.hankelh1(1,z)
    else
        return phase*_cheb_clenshaw(plan.panels[pidx].c1,t)*invsqrt
    end
end