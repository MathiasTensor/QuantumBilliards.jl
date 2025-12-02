# =============================================================================
# Chebyshev piecewise approximation of the Legendre function Q_ν(z),
# with ν = -1/2 + i k (k real or complex) and z = cosh(d), d>0.
#
# Context:
#   In the hyperbolic Helmholtz Green's function we have
#       G_k(d) = (1/(2π)) Q_{-1/2 + i k}(cosh d) ,
#   and we need to evaluate this for many distances d and a fixed (or few)
#   complex wavenumbers k, repeatedly inside BIM/Beyn contour integrals. This
#   is expensive if we do the integral representation of Q_ν(z) every time.
#   It is better to do this computation and then use the analytic recurrence 
#   relations:
#       ∂/∂z Q_ν(z) = (z^2-1)^-1 * (Q_(ν+1) - Q_ν) 
#   to get derivatives as needed. Otherwise the it's integrand is even more 
#   oscillatory and expensive to compute.
#
# Strategy:
#   1. For each k (fixed ν = -1/2 + i k), build a Gauss–Legendre quadrature
#      rule on t ∈ [0, T] for the integral representation
#          Q_ν(z) = ∫₀^∞ (z + √(z²-1) cosh t)^(-(ν+1)) dt,  z>1
#      truncated at T = T_cutoff chosen from a target bottom tolerance.
#
#   2. Use this quadrature to define a scalar function F(d) = Q_ν(cosh d)
#      and build a piecewise Chebyshev approximation of F(d) on d ∈ [dmin,dmax].
#
#   3. At runtime, evaluate F(d) from the Chebyshev plan using Clenshaw's
#      recurrence, which is much cheaper than doing the full integral every
#      time. Near z≈1 (d≈0) we switch to a small-d asymptotic expansion.
#
# API:
#   - plan_Q_GL(kmax; tol_bottom, scaling):
#       Build a Gauss–Legendre quadrature rule for the Q_ν integral, tuned
#       to kmax (maximum |k| along the contour in case of Beyn's method).
#
#   - legendre_Q(Qplan, ν, z):
#       Evaluate Q_ν(z) using the fixed GL rule (with small-d asymptotics
#       near z≈1).
#
#   - plan_Q_cheb(f_eval, dmin, dmax; npanels, M, grading, geo_ratio):
#       Build a Chebyshev plan in d for a complex function f_eval(d)
#       (typically f_eval(d) = Q_ν(cosh d) or G_k(d)).
#
#   - eval_Q(pl, d):
#       Scalar evaluation F(d) from a Chebyshev plan.
#
#   - eval_Q!(out, pl, dvec):
#       Vectorized (threaded) evaluation of F(d[i]) into out[i].
#
# Notes:
#   - Chebyshev core routines (_cheb_clenshaw, _chebfit!, _breaks_uniform,
#     _breaks_geometric) live in chebyshev_core.jl and are not duplicated here.
#
# MO / 02-12-25
# =============================================================================

#############################
#   Q_ν INTEGRAL QUADRATURE #
#############################

# Global threshold for switching to small-d asymptotics near z≈1.
# In terms of d we have z = cosh(d), and z = 1 + O(d²); for z < Z_threshold
# we fall back to a series expansion in d.
Z_threshold=1.0+1e-14

# =============================================================================
# _reference_legendre_Q:
#   Reference implementation for Q_ν(z) based on the integral representation:
#
#     Q_ν(z) = ∫₀^∞ (z + √(z²-1) cosh t)^(-(ν+1)) dt ,   z>1.
#
#   This uses quadgk on (0,∞) and is intended ONLY for testing against
#   the fixed Gauss–Legendre rule plus small-d asymptotic expansion. 
#   Pushing z smaller than 1e-8 will lead to QuadGK crashing due to singularity
#   because z = cosh(d) is 1+d^2/2 + O(d^4), therefore hitting machine precision
#   and hitting the pole.
#
# Inputs
#   ν   :: Complex      # ν index
#   z   :: Real         # argument, z>1
#   rtol:: Real=1e-15   # relative tolerance
#
# Output
#   ComplexF64          # approximate Q_ν(z)
# =============================================================================
function _reference_legendre_Q(ν::Complex{T},z::T;rtol::Real=1e-15)::Complex{T} where {T<:Real}
    zf=float(z)
    root=sqrt(zf^2-1)
    f(t)=(zf+root*cosh(t))^(-(ν+1))
    val,_=quadgk(f,0.0,Inf;rtol=rtol,order=15)
    return val
end

# =============================================================================
# QInfRule:
#   Gauss–Legendre quadrature rule on t ∈ [0, Ttrunc], stored as nodes/weights
#   mapped from [-1,1] to [0, Ttrunc].
#
# Fields
#   x      :: Vector{T}   # GL nodes on [-1,1]
#   w      :: Vector{T}   # GL weights on [-1,1]
#   Ttrunc :: T           # truncation length for the t-integral
# =============================================================================
struct QInfRule{T<:Real}
    x::Vector{T}
    w::Vector{T}
    Ttrunc::T
end

# =============================================================================
# QInfRule(N; T_cutoff):
#   Build a Gauss–Legendre rule with N nodes on t ∈ [0,T_cutoff], stored as
#   nodes and weights on [-1,1] together with the truncation T_cutoff.
#
# Inputs
#   N         :: Int      # number of GL nodes
#   T_cutoff  :: Real     # upper integration limit for t
#
# Output
#   QInfRule{Float64}
# =============================================================================
function QInfRule(N::Int;T_cutoff::Real=100.0)
    x,w=gausslegendre(N) # nodes/weights on [-1,1]
    return QInfRule{Float64}(x,w,float(T_cutoff))
end

# =============================================================================
# QInfPlan:
#   Container for the fixed-∞ quadrature tuned to a maximum |k|.
#
# Fields
#   rule       :: QInfRule{TR}     # underlying GL nodes/weights and Ttrunc
#   T          :: TR               # same as rule.Ttrunc (for convenience)
#   kmax       :: TC               # maximum |k| for which rule is tuned
#   tol_bottom :: TR               # target amplitude at t = T_cutoff
#   scaling    :: TR               # factor in N ≈ scaling * (T/π) * |kmax|
# =============================================================================
struct QInfPlan{TR<:Real,TC<:Complex}
    rule::QInfRule{TR}
    T::TR
    kmax::TC
    tol_bottom::TR
    scaling::TR
end

# =============================================================================
# plan_Q(kmax; tol_bottom, scaling, verbose):
#   Build a Gauss–Legendre quadrature rule for the Q_ν integral, tuned for
#   ν = -1/2 + i k with |k| ≤ |kmax|. The truncation T_cutoff is chosen such
#   that the integrand has decayed down to tol_bottom at t = T_cutoff
#   (heuristically via exp(-(0.5 - Im(kmax)) T_cutoff) ≈ tol_bottom).
#
# Inputs
#   kmax       :: Complex{T}   # maximum k along contour (T real type)
#   tol_bottom :: Real=1e-18   # target minimum integrand magnitude
#   scaling    :: Real=3.0     # factor in node count formula
#   verbose    :: Bool=false   # info printout
#
# Output
#   QInfPlan{Float64,ComplexF64}
# =============================================================================
function plan_Q(kmax::Complex{T};tol_bottom::Real=1e-18,scaling::Real=3.0,verbose::Bool=false) where {T<:Real}
    T_cutoff=-log(tol_bottom)/(0.5-imag(kmax))
    N=round(Int,scaling*(T_cutoff/π)*abs(kmax))
    verbose && @info "Q-plan" kmax=kmax T_cutoff=T_cutoff N=N
    rule=QInfRule(N;T_cutoff=T_cutoff)
    return QInfPlan(rule,float(T_cutoff),complex(kmax),float(tol_bottom),float(scaling))
end

# =============================================================================
# ν(k):
#   Helper function returning ν = -1/2 + i k from k.
# =============================================================================
@inline ν(k)=-0.5+im*k

#############################################################################
# Small-d expansion of G_k(d) = (1/(2π)) Q_{-1/2 + i k}(cosh d):
#
#   G_k(d) = term0 + term2 d² + term4 d⁴ + O(d⁶),
#
# with coefficients generated symbolically (Mathematica) and tested against
# mpmath's implementation. Valid for extremely small d (e.g. d≲1e-7–1e-6
# in the high-k regime) up to 14 decimal places. Use it only when z = cosh(d) < Z_threshold.
#
# NOTE:
#   - This routine returns Q_{-1/2 + i k}(cosh d), NOT the Green's function
#     G_k(d); hence the final factor 2π in the return value below.
#############################################################################
@inline function _small_z_Q(k::C,d::T)::C where {C<:Complex,T<:Real}
    H=MathConstants.eulergamma+digamma(0.5+im*k)  # H(-1/2+ik) Harmonic number
    Ld=log(d)
    L2=log(2.0)
    L4=log(4.0)
    L8=log(8.0)
    k2=k*k
    k4=k2*k2
    term0=(-H+L2-Ld)/(2*pi)
    term2=(1-12*k2+3*H+12*k2*H-3*L2-4*k2*L8+3*Ld+12*k2*Ld)*d^2/(96*pi)
    term4=(-193+1560*k2+2160*k4-330*H-1680*k2*H-1440*k4*H+165*L4+840*k2*L4+720*k4*L4-330*Ld-1680*k2*Ld-1440*k4*Ld)*d^4/(184320*pi)
    return 2*pi*(term0+term2+term4)
end

# Real-k convenience wrapper for _small_z_Q
@inline function _small_z_Q(k::T,d::T) where {T<:Real}
    return _small_z_Q(complex(k,0.0),d)
end

# =============================================================================
# legendre_Q(plan, ν, z):
#   Evaluate Q_ν(z) using the precomputed Gauss–Legendre rule stored in
#   QInfPlan. If z is extremely close to 1 (z<Z_threshold) we switch to the
#   small-d asymptotic expansion with d = acosh(z).
#
# Inputs
#   plan :: QInfPlan           # quadrature plan
#   ν    :: Complex            # index ν (typically ν = -1/2 + i k)
#   z    :: Real               # argument z = cosh(d), z>1
#
# Output
#   ComplexF64                 # Q_ν(z)
#
# Implementation details
#   - Uses Kahan summation to reduce roundoff in the GL sum.
#   - t ∈ [0,T] is obtained from x∈[-1,1] via t = (T/2)(x+1).
# =============================================================================
@inline function legendre_Q(plan::QInfPlan,ν::C,z::T)::C where {C<:Complex,T<:Real}
    z<Z_threshold && return _small_z_Q(imag(ν+0.5),acosh(z))
    x=plan.rule.x
    w=plan.rule.w
    Tt=plan.rule.Ttrunc
    jac=Tt/2
    root=sqrt(z*z-1)
    s=zero(C)
    c=zero(C)
    @inbounds for i in eachindex(x)
        t=(Tt/2)*(x[i]+1)
        base=z+root*cosh(t)
        y=w[i]*base^(-(ν+1))-c  # Kahan step
        tmp=s+y
        c=(tmp-s)-y
        s=tmp
    end
    return jac*s
end

# =============================================================================
# ChebQTable:
#   Chebyshev table on a single panel [a,b] in d for a complex function F(d).
#
# Fields
#   a :: T                # left endpoint of panel
#   b :: T                # right endpoint of panel
#   M :: Int                    # Chebyshev polynomial degree
#   c :: Vector{Complex{T}}     # Chebyshev coefficients (length M+1)
# =============================================================================
struct ChebQTable{T<:Real}
    a::T
    b::T
    M::Int
    c::Vector{Complex{T}}
end

# =============================================================================
# _build_table_Q!(Qplan, ν, a, b; M):
#   Construct a Chebyshev table for
#       F(d) = Q_ν(cosh d) 
#   on the panel [a,b], using Chebyshev–Lobatto nodes t_j = cos(π j / M).
#
# Inputs
#   Qplan  :: QInfPlan          # Gauss–Legendre plan for Q_ν
#   ν      :: Complex           # ν = -1/2 + i k
#   a,b    :: Float64           # panel endpoints, 0 < a < b
#   M      :: Int               # Chebyshev degree
#
# Output
#   ChebQTable(a,b,M,c)         # with c the Chebyshev coefficients
# =============================================================================
function _build_table_Q!(Qplan::QInfPlan,ν::Complex,a::T,b::T;M::Int=300)::ChebQTable{T} where {T<:Real}
    @assert a>0 && b>a "invalid panel [a,b] = [$a,$b]"
    f=Vector{Complex{T}}(undef,M+1)
    @inbounds for j in 0:M
        t=cospi(j/M)                    # Chebyshev node in [-1,1]
        d=((b+a)+(b-a)*t)/2       # affine map to [a,b]
        z=cosh(d)
        f[j+1]=legendre_Q(Qplan,ν,z)
    end
    c=Vector{Complex{T}}(undef,M+1)
    _chebfit!(c,f)
    return ChebQTable(a,b,M,c)
end

# =============================================================================
# ChebQPlan:
#   Piecewise Chebyshev plan over d ∈ [dmin,dmax], represented as a collection
#   of ChebQTable panels.
#
# Fields
#   panels :: Vector{ChebQTable{T}} # per-panel Chebyshev tables
#   dmin   :: T            # global lower bound
#   dmax   :: T            # global upper bound
# =============================================================================
struct ChebQPlan{T<:Real}
    panels::Vector{ChebQTable{T}}
    dmin::T
    dmax::T
end

# =============================================================================
# plan_Q_cheb(Qplan, ν, dmin, dmax;, npanels, M, grading, geo_ratio):
#   Build a piecewise-Chebyshev plan for F(d) = Q_ν(cosh d) on [dmin,dmax].
#
# Inputs
#   Qplan    :: QInfPlan            # GL quadrature plan for Q_ν
#   ν        :: Complex             # ν = -1/2 + i k
#   dmin     :: T            # lower bound, dmin>0
#   dmax     :: T            # upper bound, dmax>dmin
#   npanels  :: Int=200             # number of panels
#   M        :: Int=40              # Chebyshev degree per panel
#   grading  :: Symbol=:geometric   # :uniform or :geometric
#   geo_ratio:: Real=1.03           # geometric panel-size ratio
#
# Output
#   ChebQPlan(panels,dmin,dmax)
#
# Notes
#   - Panels are generated using _breaks_uniform or _breaks_geometric from
#     chebyshev_core.jl.
#   - Each panel table is built independently (threaded).
# =============================================================================
function plan_Q_cheb(Qplan::QInfPlan,ν::Complex{T},dmin::T,dmax::T;npanels::Int=2500,M::Int=300,grading::Symbol=:geometric,geo_ratio::Real=1.03)::ChebQPlan{T} where {T<:Real}
    @assert dmin>0 && dmax>dmin
    br=grading===:uniform ? _breaks_uniform(dmin,dmax,npanels) : _breaks_geometric(dmin,dmax,npanels;ratio=geo_ratio)
    panels=Vector{ChebQTable{T}}(undef,npanels)
    @inbounds Threads.@threads for i in 1:npanels
        panels[i]=_build_table_Q!(Qplan,ν,br[i],br[i+1];M=M)
    end
    return ChebQPlan(panels,dmin,dmax)
end

# =============================================================================
# _find_panel_Q(panels, d):
#   Locate the index p such that panels[p].a ≤ d ≤ panels[p].b via a binary
#   search over the panel list.
#
# Inputs
#   panels :: Vector{ChebQTable{T}} where T
#   d      :: Float64
#
# Output
#   Int                         # panel index
#
# Behavior
#   - Throws an error if d lies outside [panels[1].a, panels[end].b].
# =============================================================================
@inline function _find_panel_Q(panels::Vector{ChebQTable{T}},d::T)::Int where {T<:Real}
    lo=1
    hi=length(panels)
    @inbounds while lo ≤ hi
        mid=(lo+hi)>>>1
        P=panels[mid]
        if d<P.a
            hi=mid-1
        elseif d>P.b
            lo=mid+1
        else
            return mid
        end
    end
    error("d=$d outside Chebyshev plan range")
end

# =============================================================================
# eval_Q(plan, d):
#   Scalar evaluation of F(d) from a Chebyshev plan.
#
# Inputs
#   plan :: ChebQPlan
#   d    :: Float64
#
# Output
#   ComplexF64       # F(d), either Q_ν(cosh d) or (1/(2π))Q_ν(cosh d)
#
# Implementation
#   - Find panel index p s.t. d ∈ [a,b].
#   - Map d → t ∈ [-1,1] via affine map.
#   - Evaluate Chebyshev series using _cheb_clenshaw from chebyshev_core.jl.
# =============================================================================
@inline function eval_Q(plan::ChebQPlan{T},d::T)::Complex{T} where {T<:Real}
    p=_find_panel_Q(plan.panels,d)
    P=plan.panels[p]
    t=(2*d-(P.b+P.a))/(P.b-P.a)
    return _cheb_clenshaw(P.c,t)
end

# =============================================================================
# eval_Q!(out, plan, dvec):
#   Vectorized/threaded evaluation of F(d_i) into out[i].
#
# Inputs
#   out  :: AbstractVector{Complex{T}}   # output (length = length(dvec))
#   plan :: ChebQPlan{T}                 # Chebyshev plan in d
#   dvec :: AbstractVector{T}            # input d values
#
# Output
#   Fills out[i] = F(dvec[i]) in place.
#
# Implementation details
#   - Each thread processes a chunk of dvec independently.
#   - Per element:
#       * find panel p via _find_panel_Q
#       * compute mapped t
#       * call _cheb_clenshaw on P.c, t
# =============================================================================
function eval_Q!(out::AbstractVector{Complex{T}},plan::ChebQPlan{T},dvec::AbstractVector{T}) where {T<:Real}
    panels=plan.panels
    @inbounds Threads.@threads for i in eachindex(dvec)
        d=dvec[i]
        p=_find_panel_Q(panels,d)
        P=panels[p]
        t=(2*d-(P.b+P.a))/(P.b-P.a)
        out[i]=_cheb_clenshaw(P.c,t)
    end
    return nothing
end

# =============================================================================
# Convenience wrapper: ChebPlanQ_GL
#
#   Small wrapper to keep the same external API as before, but with the
#   function F(d) now hard-wired to legendre_Q(Qplan,ν,cosh d).
# =============================================================================
struct ChebPlanQ_GL{T<:Real}
    plan::ChebQPlan{T}
end

# =============================================================================
# build_Q_cheb_from_GL(Qplan, ν, dmin, dmax; npanels, M, grading, geo_ratio)
#
#   Build a Chebyshev plan for: F(d) = Q_ν(cosh d) 
#
# Inputs/Output exactly as before, now just forwarding to plan_Q_cheb.
# =============================================================================
function build_Q_cheb_from_GL(Qplan::QInfPlan,ν::Complex{T},dmin::T,dmax::T;npanels::Int=2500,M::Int=300,grading::Symbol=:geometric,geo_ratio::Real=1.03)::ChebPlanQ_GL{T} where {T<:Real}
    cp=plan_Q_cheb(Qplan,ν,dmin,dmax;npanels=npanels,M=M,grading=grading,geo_ratio=geo_ratio)
    return ChebPlanQ_GL(cp)
end

# =============================================================================
# eval_Q(pl_GL, d):
#   Scalar evaluation of F(d) = Q_ν(cosh d) from ChebPlanQ_GL.
# Inputs:
#   pl_GL :: ChebPlanQ_GL
#   d     :: Real
# Output:
#   Complex{Real}
# =============================================================================
@inline function eval_Q(pl_GL::ChebPlanQ_GL,d::T)::Complex{T} where {T<:Real}
    return eval_Q(pl_GL.plan,d)
end

# =============================================================================
# eval_Q!(out, pl_GL, dvec):
#   Vectorized/threaded evaluation of F(d_i) = Q_ν(cosh d_i) into out[i].
# Inputs:
#   out   :: AbstractVector{Complex{T}}   # output (length = length(dvec))
#   pl_GL :: ChebPlanQ_GL{T}              # Chebyshev plan in d
#   dvec  :: AbstractVector{T}            # input d values
# Output:
#   Fills out[i] = F(dvec[i]) in place.
# =============================================================================
function eval_Q!(out::AbstractVector{Complex{T}},pl_GL::ChebPlanQ_GL,dvec::AbstractVector{T}) where {T<:Real}
    eval_Q!(out,pl_GL.plan,dvec)
    return nothing
end