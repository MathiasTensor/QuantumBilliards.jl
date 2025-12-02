###############################################################################
# Core Chebyshev routines for piecewise spectral approximation.
#
# This file provides the low–level building blocks used throughout the BIM /
# Beyn machinery to approximate special functions (Hankel, Legendre Q, Green’s
# functions, etc.) on many geometric panels with high accuracy and stability.
#
# Contents
# --------
# 1.  _cheb_clenshaw(c,t)
#       Stable evaluation of a Chebyshev series
#           f(t) = ∑_{m=0}^M c[m] T_m(t)
#     using Clenshaw’s reverse recurrence.  This routine is called tens of
#     millions of times inside matrix–vector products, so it is optimized for
#     zero allocations and minimal branches.
#
# 2.  _chebfit!(c,f)
#       Compute Chebyshev coefficients c from function samples f evaluated at
#       Chebyshev–Lobatto nodes t_j = cos(π j / M).  This is a direct O(M²)
#       cosine transform (DCT-I) with endpoint half-weights:
#           c_m = (2/M) * ∑ w_j f_j cos(π j m / M),  w_0=w_M=1/2.
#       After fitting, c[1] is halved so that the normalization matches
#       _cheb_clenshaw.
#
# 3.  _breaks_uniform(rmin,rmax,np)
#       Generate np uniformly sized panels, returning np+1 breakpoints.
#
# 4.  _breaks_geometric(rmin,rmax,np;ratio)
#       Generate geometrically graded panels whose widths grow as ratio^(i−1).
#       This grading is critical for resolving near-singular and strongly
#       oscillatory kernels (Hankel, Legendre Q) without excessive panel count.
#
# Usage Pattern
# -------------
# These routines are intentionally minimal: they know nothing about the
# underlying special function or geometry.  High-level modules (H1x Hankel
# tables, Q_ν tables, Green’s function tables, etc.) call these primitives to:
#
#   • evaluate f(r) at Chebyshev nodes on each panel;
#   • use _chebfit! to obtain Chebyshev coefficients;
#   • evaluate the function later with _cheb_clenshaw;
#   • divide the domain using _breaks_uniform or _breaks_geometric depending on
#     the expected oscillatory behavior.
#
# All functions are allocation-free inside tight loops and suitable for
# multi-threaded vectorized usage.
#
# MO / 02-12-25
###############################################################################

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


