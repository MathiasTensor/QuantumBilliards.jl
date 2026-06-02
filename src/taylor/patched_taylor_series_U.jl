# =============================================================================
# Fast Taylor-patch evaluator for the Landau-regularized magnetic Green radial
# factor
# =============================================================================
# This file provides fast, allocation-free evaluation of the gauge-independent
# radial factor
#
#     F(z;ν) = -exp(-z/2) U(1/2-ν,1,z) / (4 Γ(ν+1/2)),
#
# where
#
#     z = |x-y|² / b².
#
# This is the radial part of the regularized magnetic Green function
#
#     G_B(x,y;ν) = exp(iΦ(x,y)) F(|x-y|²/b²),
#
# with magnetic phase handled elsewhere.
#
# What this file contains
# -----------------------
#   - mpmath seeding for F and its derivative,
#   - Taylor-patch propagation in s=sqrt(z),
#   - small-z logarithmic expansion,
#   - fast evaluators for F(z), Rν(z), and Fz(z).
#
# Mathematical split
# ------------------
# Near z=0,
#
#     F(z;ν) = aν log(z) + Rν(z),
#     aν = cos(πν)/(4π),
#
# with finite part
#
#     Rν(0) = aν[ψ(ν+1/2)-2ψ(1)] - sin(πν)/4.
#
# The evaluator provides both F(z) and Rν(z). The Kress-specific combination
# involving log(|x(t)-x(τ)|²/(4b²sin²((t-τ)/2))) belongs in the separate Kress
# file.
#
# Taylor-patch strategy
# Define
#
#     s = sqrt(z),
#     G(s;ν) = F(s²;ν).
#
# The transformed radial Green factor satisfies
#
#     G''(s) = -G'(s)/s + (s^2 - 4ν)G(s),     s = sqrt(z).
#
# Hence the formal turning point is
#
#     s_t = 2sqrt(|ν|),     z_t = 4|ν|.
#
# For tables whose z-interval approaches or crosses z_t, one-sided Taylor
# propagation from a single seed can lose accuracy across the Airy transition
# region and over long post-turning intervals. The builder therefore
# automatically switches to a dynamic multi-seed strategy. It seeds near s_t and
# inserts additional seeds up to smax, so no propagated segment becomes too
# long. This affects only table construction; runtime evaluation is unchanged.
# MO 31/5/26
# =============================================================================

# uncomment these when testing this file in isolation, but they are already imported in QuantumBilliards.jl
#using PyCall
#using SpecialFunctions
#using LinearAlgebra
#using Test
#using Random
#using BenchmarkTools

# also import the mpmath objects (since QuantumBilliards.jl has these in pycall_init.jl)
const inv4π=1/(4*pi)

#------------------------------------------------------------------------------
# UConfluentTaylorConfig
#
# Global numerical configuration for the Taylor-patch evaluator of
#
#     F(z;ν) = -exp(-z/2)U(1/2-ν,1,z)/(4Γ(ν+1/2)).
#
# The runtime table is built in s=sqrt(z), where G(s)=F(s²) satisfies
#
#     G''(s) = -G'(s)/s + (s²-4ν)G(s).
#
# The formal turning point is
#
#     s_t = 2sqrt(|ν|),        z_t = 4|ν|.
#
# If the table extends close to or beyond z_t, long one-sided propagation can
# accumulate error. The multi-seed parameters below limit that propagation
# length by inserting additional high-precision mpmath seeds.
#
# Fields
#   h_patch          spacing of Taylor centers in s
#   P_patch          Taylor degree per local patch
#   turning_eta_crit activate multi-seeding when η=zmax/(4|ν|) exceeds this
#   min_seed_span    lower bound for seed spacing in s
#   max_seed_span    upper bound for seed spacing in s
#   target_accuracy  intended validation target; does not alter evaluation
#------------------------------------------------------------------------------
Base.@kwdef mutable struct UConfluentTaylorConfig
    h_patch::Float64=1e-5
    P_patch::Int=8
    turning_eta_crit::Float64=0.8
    min_seed_span::Float64=0.1
    max_seed_span::Float64=0.2
    target_accuracy::Float64=1e-13
end
const U_CONFLUENT_TAYLOR_CONFIG=UConfluentTaylorConfig()

@inline u_confluent_h_patch()=U_CONFLUENT_TAYLOR_CONFIG.h_patch
@inline u_confluent_P_patch()=U_CONFLUENT_TAYLOR_CONFIG.P_patch
@inline magnetic_turning_eta_crit()=U_CONFLUENT_TAYLOR_CONFIG.turning_eta_crit
@inline magnetic_min_seed_span()=U_CONFLUENT_TAYLOR_CONFIG.min_seed_span
@inline magnetic_max_seed_span()=U_CONFLUENT_TAYLOR_CONFIG.max_seed_span
@inline magnetic_target_accuracy()=U_CONFLUENT_TAYLOR_CONFIG.target_accuracy

@inline function confluent_U_params()
    cfg=U_CONFLUENT_TAYLOR_CONFIG
    return cfg.h_patch,cfg.P_patch
end

@inline function magnetic_seed_params()
    cfg=U_CONFLUENT_TAYLOR_CONFIG
    return cfg.turning_eta_crit,cfg.min_seed_span,cfg.max_seed_span,cfg.target_accuracy
end

function confluent_U_set_h!(h::Real)
    h>0 || error("h_patch must be positive.")
    U_CONFLUENT_TAYLOR_CONFIG.h_patch=Float64(h)
    return U_CONFLUENT_TAYLOR_CONFIG
end

function confluent_U_set_P!(P::Integer)
    P≥1 || error("P_patch must be at least 1.")
    U_CONFLUENT_TAYLOR_CONFIG.P_patch=Int(P)
    return U_CONFLUENT_TAYLOR_CONFIG
end

function magnetic_set_turning_eta_crit!(η::Real)
    η>0 || error("turning_eta_crit must be positive.")
    U_CONFLUENT_TAYLOR_CONFIG.turning_eta_crit=Float64(η)
    return U_CONFLUENT_TAYLOR_CONFIG
end

#------------------------------------------------------------------------------
# magnetic_set_seed_span!
#
# Modify the admissible spacing between neighboring high-precision seeds.
#
# Smaller seed spacings reduce accumulated analytic-continuation error when
# Taylor patches are propagated over long intervals, particularly when the
# evaluation range extends significantly beyond the turning point
#
#     z_t = 4|ν|.
#
# Decreasing the seed spacing generally improves robustness at the expense of
# increased table-construction time.
#
# Runtime evaluation cost is unchanged.
#
# Returns
# -------
# Updated UConfluentTaylorConfig object.
#------------------------------------------------------------------------------
function magnetic_set_seed_span!(;min_seed_span=nothing,max_seed_span=nothing)
    cfg=U_CONFLUENT_TAYLOR_CONFIG
    smin=isnothing(min_seed_span) ? cfg.min_seed_span : Float64(min_seed_span)
    smax=isnothing(max_seed_span) ? cfg.max_seed_span : Float64(max_seed_span)
    smin>0 || error("min_seed_span must be positive.")
    smax>0 || error("max_seed_span must be positive.")
    smin<=smax || error("min_seed_span must be <= max_seed_span.")
    cfg.min_seed_span=smin
    cfg.max_seed_span=smax
    return cfg
end

#------------------------------------------------------------------------------
# magnetic_set_target_accuracy!
#
# Set the target relative accuracy associated with the magnetic Taylor-table
# machinery.
#
# This parameter is intended for validation, regression testing, and future
# automatic parameter-selection routines. It does not currently alter the
# discretization parameters automatically.
#
# Typical values
# --------------
# 1e-10 : exploratory calculations
# 1e-12 : production-quality calculations
# 1e-13 : recommended default
# 1e-14 : near machine precision
#
# Returns
# -------
# Updated UConfluentTaylorConfig object.
#------------------------------------------------------------------------------
function magnetic_set_target_accuracy!(tol::Real)
    0<tol<1 || error("target_accuracy must satisfy 0 < target_accuracy < 1.")
    U_CONFLUENT_TAYLOR_CONFIG.target_accuracy=Float64(tol)
    return U_CONFLUENT_TAYLOR_CONFIG
end

#------------------------------------------------------------------------------
# confluent_U_set_taylor_params!
#
# Purpose
# Update the global configuration used by the Landau-regularized magnetic Green
# Taylor-table machinery.
#
# The evaluated radial factor is
#
#     F(z;ν) = -exp(-z/2)U(1/2-ν,1,z)/(4Γ(ν+1/2)),
#
# with
#
#     z = |x-y|²/b²,        s = sqrt(z),        G(s;ν)=F(s²;ν).
#
# The table stores local Taylor expansions in s. The coefficients are generated
# from the transformed ODE
#
#     G''(s) = -G'(s)/s + (s²-4ν)G(s).
#
# The formal turning point is
#
#     s_t = 2sqrt(|ν|),        z_t = 4|ν|.
#
# Therefore the ratio
#
#     η = zmax/(4|ν|)
#
# measures whether the table reaches the turning region. When η is large,
# propagation from one seed may accumulate error. The multi-seed parameters below
# control how often high-precision mpmath seeds are inserted.
#
# Keyword arguments
# -----------------
# h_patch
#     Uniform Taylor-center spacing in s=sqrt(z). Smaller h_patch improves local
#     Taylor accuracy, but increases Npatch and memory.
#
# P_patch
#     Taylor degree on each patch. Larger P_patch improves local accuracy, but
#     increases coefficient storage and patch-construction work.
#
# turning_eta_crit
#     Activates dynamic multi-seeding when
#
#         zmax/(4|ν|) > turning_eta_crit.
#
# min_seed_span, max_seed_span
#     Lower and upper bounds for the s-distance between neighboring mpmath
#     seeds. Smaller max_seed_span improves robustness over long post-turning
#     intervals, but increases table build time. Runtime evaluation is unchanged.
#
# target_accuracy
#     Intended relative accuracy for validation and regression tests. This does
#     not automatically change h_patch, P_patch, or seed spacing.
#
# validate
#     If true, run magnetic_validate_taylor_config! after applying the new
#     parameters. If validation fails, the old configuration is restored and the
#     validation error is rethrown.
#
# validation_kwargs...
#     Keyword arguments forwarded to `magnetic_validate_taylor_config!`.
#
#     These are used only when
#
#         validate = true.
#
#     They define the independent numerical check that is run after changing the
#     global Taylor-table parameters. A typical validator should build one or
#     more tables with the updated configuration, compare `_eval_F`, `_eval_Fz`,
#     `_eval_Alog`, `_eval_Alog_z`, and `_eval_Blog` against mpmath references,
#     and require errors below `magnetic_target_accuracy()`.
#
#     Accepted validation keywords:
#
#     nus
#         Vector of ComplexF64 test values of ν. Should include small ν near the
#         turning-sensitive regime, large ν, half-integers, integers, and one
#         complex Beyn-like value.
#
#     zmax
#         Maximum z value used in the validation table. Should be at least as
#         large as the largest production value expected from
#         `magnetic_zmax(pts,bmag;safety=...)`.
#
#     zmin
#         Lower endpoint of the Taylor table. Usually 1e-3.
#
#     zsmall
#         Threshold below which the Frobenius/log small-z branch is used.
#         Usually equal to zmin.
#
#     Msmall
#         Order of the small-z logarithmic expansion.
#
#     mp_dps
#         mpmath precision used for reference values and table seeds.
#
#     ztests
#         Optional explicit vector of z values. If omitted, the validator should
#         test logarithmically small z, zsmall, moderate z, the turning region,
#         and fractions of zmax.
#
#     rtol
#         Relative tolerance. Default should be `magnetic_target_accuracy()`.
#
#     atol
#         Absolute tolerance, useful near zeros of F, A, or derivatives.
#
#     verbose
#         Print per-ν and per-z diagnostic errors.
#
#     test_derivatives
#         Whether to test `_eval_Fz` and `_eval_Alog_z`. Derivative tests are
#         stricter and can reveal propagation errors not visible in F alone.
#
#     test_split
#         Whether to test the exact split
#
#             F(z) = Aν(z)log(z) + Bν(z).
#
# Example:
#
#     confluent_U_set_taylor_params!(
#         min_seed_span=0.1,
#         max_seed_span=0.2,
#         target_accuracy=1e-13,
#         validate=true,
#         nus=ComplexF64[20+0im,40+0im,80+0im,600+0im,2003.37+0.5im],
#         zmax=500.0,
#         zmin=1e-3,
#         zsmall=1e-3,
#         Msmall=40,
#         mp_dps=160,
#         verbose=true,
#     )
#------------------------------------------------------------------------------
function confluent_U_set_taylor_params!(;h_patch=nothing,P_patch=nothing,turning_eta_crit=nothing,min_seed_span=nothing,max_seed_span=nothing,target_accuracy=nothing,validate::Bool=false,validation_kwargs...)
    old=deepcopy(U_CONFLUENT_TAYLOR_CONFIG)
    try
        !isnothing(h_patch) && confluent_U_set_h!(h_patch)
        !isnothing(P_patch) && confluent_U_set_P!(P_patch)
        !isnothing(turning_eta_crit) && magnetic_set_turning_eta_crit!(turning_eta_crit)
        (!isnothing(min_seed_span) || !isnothing(max_seed_span)) && magnetic_set_seed_span!(;min_seed_span=min_seed_span,max_seed_span=max_seed_span)
        !isnothing(target_accuracy) && magnetic_set_target_accuracy!(target_accuracy)
        validate && magnetic_validate_taylor_config!(;validation_kwargs...)
        return U_CONFLUENT_TAYLOR_CONFIG
    catch err
        U_CONFLUENT_TAYLOR_CONFIG.h_patch=old.h_patch
        U_CONFLUENT_TAYLOR_CONFIG.P_patch=old.P_patch
        U_CONFLUENT_TAYLOR_CONFIG.turning_eta_crit=old.turning_eta_crit
        U_CONFLUENT_TAYLOR_CONFIG.min_seed_span=old.min_seed_span
        U_CONFLUENT_TAYLOR_CONFIG.max_seed_span=old.max_seed_span
        U_CONFLUENT_TAYLOR_CONFIG.target_accuracy=old.target_accuracy
        rethrow(err)
    end
end

# =============================================================================
# magnetic_log_coeff
#
#   Return the coefficient of the logarithmic singularity of the regularized
#   magnetic Green radial factor.
#
#   Near z=0,
#
#   F(z;ν) = a_ν log(z) + R_ν(z),
#
#   where R_ν is finite at z=0 and
#
#   a_ν = cos(πν)/(4π).
#
# Inputs
#   ν::ComplexF64 Complex spectral parameter.
#
# Output
#   ComplexF64 Logarithmic coefficient a_ν.
#
# Notes
#   This is the radial coefficient only. The magnetic phase exp(iΦ) is applied
#   separately during boundary-kernel assembly.
# =============================================================================
@inline magnetic_log_coeff(ν::ComplexF64)=cospi(ν)*inv4π

# =============================================================================
# magnetic_R0
#
# Purpose
# Return the diagonal finite part R_ν(0) of the regularized magnetic Green
# radial factor.
#
#   F(z;ν)=a_ν log(z)+R_ν(z),
#
#   the finite part is 
#
#   R_ν(0) = a_ν[ψ(ν+1/2)-2ψ(1)] - sin(πν)/4.
#
# Inputs
#   ν::ComplexF64 Complex spectral parameter.
#
# Output
#   ComplexF64 Finite part R_ν(0).
#
# Notes
#   This Float64/SpecialFunctions version is cheap and sufficient for many
#   uses. For high-accuracy small-z series construction, use
#   `magnetic_R0_mpmath`.
# =============================================================================
@inline function magnetic_R0(ν::ComplexF64)
    a=magnetic_log_coeff(ν)
    return a*(digamma(ν+0.5)-2*digamma(1.0+0.0im))-sin(pi*ν)/4
end

# =============================================================================
# magnetic_log_coeff_mpmath
#
# Purpose
# High-precision mpmath version of `magnetic_log_coeff`.
#
# Why this exists
# For large |ν|, computing a_ν in ComplexF64 can introduce a small error floor
# in the z≈0 logarithmic branch. Since a_ν multiplies log(z), even a tiny
# coefficient error can become visible at z≈1e-15.
#
# Inputs
#   ν::ComplexF64 Complex spectral parameter.
#
# Keyword arguments
#   dps::Int=100 Decimal precision used by mpmath.
#
# Output
#   ComplexF64 High-precision-computed a_ν, rounded back to ComplexF64.
# =============================================================================
function magnetic_log_coeff_mpmath(ν::ComplexF64;dps::Int=100)
    _mpctx[].dps=dps
    νp=_mpc[](real(ν),imag(ν))
    a=_mp_cos[](_mp_pi[]*νp)/(4*_mp_pi[])
    return ComplexF64(pycall(_pyfloat[],Float64,a.real),pycall(_pyfloat[],Float64,a.imag))
end

# =============================================================================
# magnetic_R0_mpmath
#
# Purpose
# High-precision mpmath version of `magnetic_R0`.
#
#       R_ν(0) = a_ν[ψ(ν+1/2)-2ψ(1)] - sin(πν)/4
#
# using mpmath arithmetic.
#
# Why this exists
# The small-z branch is anchored by R_ν(0). If R_ν(0) is only computed with
# ordinary ComplexF64 arithmetic, the small-z series inherits that error.
#
# Inputs
#   ν::ComplexF64 Complex spectral parameter.
#
# Keyword arguments
#   dps::Int=100 Decimal precision used by mpmath.
#
# Output
#   ComplexF64 High-precision-computed R_ν(0), rounded back to ComplexF64.
# =============================================================================
function magnetic_R0_mpmath(ν::ComplexF64;dps::Int=100)
    _mpctx[].dps=dps
    νp=_mpc[](real(ν),imag(ν))
    a=_mp_cos[](_mp_pi[]*νp)/(4*_mp_pi[])
    R0=a*(_mp_digamma[](νp+_mpf[](0.5))-2*_mp_digamma[](_mpf[](1)))-_mp_sin[](_mp_pi[]*νp)/4
    return ComplexF64(pycall(_pyfloat[],Float64,R0.real),pycall(_pyfloat[],Float64,R0.imag))
end

# =============================================================================
# seed_G_Gp_mpmath
#
# Purpose
#   Compute one high-precision seed for the Taylor-patch propagation of
#
#       G(s;ν) = F(s²;ν),
#
#   together with its derivative with respect to s:
#
#       G'(s;ν) = 2s F_z(s²;ν).
#
# The radial Green factor is
#
#       F(z;ν) = -exp(-z/2) U(1/2-ν,1,z) / (4Γ(ν+1/2)).
#
#   Differentiating with respect to z gives
#
#       F_z(z;ν) =
#           C exp(-z/2) [-1/2 U(a,1,z) - a U(a+1,2,z)],
#
#   where
#
#       a = 1/2 - ν,
#       C = -1/(4Γ(ν+1/2)).
#
# Why this is needed
#   Direct calls to HypergeometricU are too expensive for every matrix entry.
#   We therefore call mpmath only once per ν to seed the ODE propagation.
#
# Inputs
#   s0::Float64 Seed location in the variable s=sqrt(z). Must satisfy s0>0.
#
#   ν::ComplexF64 Complex spectral parameter. For Beyn this will generally be complex.
#
# Keyword arguments
#   dps::Int=80 Decimal precision used by mpmath for the seed evaluation.
#
# Output
#   (G0,Gp0)::Tuple{ComplexF64,ComplexF64}
#    G0  = G(s0;ν)
#    Gp0 = dG/ds(s0;ν)
# =============================================================================
function seed_G_Gp_mpmath(s0::Float64,ν::ComplexF64;dps::Int=80)
    lock(PYCALL_MPMATH_LOCK)
    try
        _mpctx[].dps=dps
        s_py=_mpf[](s0)
        z=s_py*s_py
        ν_py=_mpc[](real(ν),imag(ν))
        a_py=_mpf[](0.5)-ν_py
        C=-1/(4*_mp_gamma[](ν_py+_mpf[](0.5)))
        U0=_mp_hyperu[](a_py,1,z)
        U1=_mp_hyperu[](a_py+1,2,z)
        ez=_mp_exp[](-z/2)
        F=C*ez*U0
        Fz=C*ez*(-_mpf[](0.5)*U0-a_py*U1)
        G=F
        Gp=2*s_py*Fz
        Gre=pycall(_pyfloat[],Float64,G.real);Gim=pycall(_pyfloat[],Float64,G.imag)
        Gpre=pycall(_pyfloat[],Float64,Gp.real);Gpim=pycall(_pyfloat[],Float64,Gp.imag)
        return ComplexF64(Gre,Gim),ComplexF64(Gpre,Gpim)
    finally
        unlock(PYCALL_MPMATH_LOCK)
    end
end

# =============================================================================
# seed_A_Ap_mpmath
#
# Purpose
# Compute one high-precision seed for
#
#     H(s;ν)=Aν(s²),
#
# where
#
#     Aν(z)=cos(πν)/(4π)*exp(-z/2)*1F1(1/2-ν;1;z).
#
# This is the full z-dependent logarithmic coefficient in
#
#     F(z;ν)=Aν(z)log(z)+Bν(z).
#
# Output
#   (A0,Ap0), where Ap0=dH/ds(s0).
# =============================================================================
function seed_A_Ap_mpmath(s0::Float64,ν::ComplexF64;dps::Int=80)
    lock(PYCALL_MPMATH_LOCK)
    try
        _mpctx[].dps=dps
        s_py=_mpf[](s0)
        z=s_py*s_py
        ν_py=_mpc[](real(ν),imag(ν))
        a_py=_mpf[](0.5)-ν_py
        c=_mp_cos[](_mp_pi[]*ν_py)/(4*_mp_pi[])
        M0=_mp_hyp1f1[](a_py,1,z)
        M1=_mp_hyp1f1[](a_py+1,2,z)
        ez=_mp_exp[](-z/2)
        A=c*ez*M0
        Az=c*ez*(a_py*M1-_mpf[](0.5)*M0)
        Ap=2*s_py*Az
        Are=pycall(_pyfloat[],Float64,A.real);Aim=pycall(_pyfloat[],Float64,A.imag)
        Apre=pycall(_pyfloat[],Float64,Ap.real);Apim=pycall(_pyfloat[],Float64,Ap.imag)
        return ComplexF64(Are,Aim),ComplexF64(Apre,Apim)
    finally
        unlock(PYCALL_MPMATH_LOCK)
    end
end

# =============================================================================
# horner_deriv_col
#
# Purpose
# Evaluate the derivative of a Taylor polynomial stored as one column of a
# coefficient matrix.
#
# Coefficient convention
#
#   p(x) = Σ_{n=0}^P A[n+1,j] x^n,
#
#   then this routine evaluates
#
#   p'(x) = Σ_{n=1}^P n A[n+1,j] x^(n-1).
#
# Inputs
#   A::Matrix{ComplexF64} Matrix of Taylor coefficients.
#
#   j::Int Patch index / matrix column.
#
#   x::Float64 Local coordinate x=s-centers[j].
#
# Output
#   ComplexF64 Derivative p'(x).
#
# Usage
#   Used for G'(s) and hence F_z(z)=G'(sqrt(z))/(2sqrt(z)).
# =============================================================================
@inline function horner_deriv_col(A::Matrix{ComplexF64},j::Int,x::Float64)
    P=size(A,1)-1
    P==0 && return ComplexF64(0.0,0.0)
    xx=ComplexF64(x,0.0)
    acc=ComplexF64(P,0.0)*A[P+1,j]
    @inbounds for n in (P-1):-1:1
        acc=muladd(acc,xx,ComplexF64(n,0.0)*A[n+1,j])
    end
    return acc
end

# =============================================================================
# horner_eval_vec
#
# Purpose
#   Evaluate a polynomial stored in a coefficient vector.
#
# Coefficient convention
#   v[n+1] is the coefficient of x^n, so the routine evaluates
#
#   p(x) = Σ_{n=0}^P v[n+1] x^n.
#
# Inputs
#   v::Vector{ComplexF64} Coefficient vector.
#
#   x::Float64 Evaluation point.
#
# Output
#   ComplexF64 Polynomial value.
#
# Usage
#   Used mainly for the small-z power series A(z), B(z).
# =============================================================================
@inline function horner_eval_vec(v::Vector{ComplexF64},x::Float64)
    xx=ComplexF64(x,0.0)
    acc=ComplexF64(0.0,0.0)
    @inbounds for n in length(v):-1:1
        acc=muladd(acc,xx,v[n])
    end
    return acc
end

# =============================================================================
# horner_deriv_vec
#
# Purpose
# Evaluate the derivative of a polynomial stored in a coefficient vector.
#
# Coefficient convention
#
#  p(x)=Σ_{n=0}^P v[n+1]x^n,
#
#  then this routine evaluates p'(x).
#
# Inputs
#   v::Vector{ComplexF64} Coefficient vector.
#
#   x::Float64 Evaluation point.
#
# Output
#   ComplexF64 Derivative p'(x).
#
# Usage
# Used in the small-z derivative F_z(z).
# =============================================================================
@inline function horner_deriv_vec(v::Vector{ComplexF64},x::Float64)
    P=length(v)-1
    P==0 && return ComplexF64(0.0,0.0)
    xx=ComplexF64(x,0.0)
    acc=ComplexF64(P,0.0)*v[P+1]
    @inbounds for n in (P-1):-1:1
        acc=muladd(acc,xx,ComplexF64(n,0.0)*v[n+1])
    end
    return acc
end

# =============================================================================
# build_magnetic_patch_coeffs!
#
# Purpose
# Generate one local Taylor patch for
#
#       G(s;ν)=F(s²;ν)
#
#   around a center s0.
#
# Differential equation
# The transformed radial Green factor satisfies
#
#       G''(s) = -G'(s)/s + (s²-4ν)G(s).
#
#   We write
#
#       G(s0+h)=Σ g_n h^n.
#
#   The recurrence is obtained by expanding
#
#       1/(s0+h),
#       (s0+h)²,
#       G(s0+h),
#       G'(s0+h),
#       G''(s0+h),
#
#    and matching powers of h.
#
# Inputs
#   g::Vector{ComplexF64} Output coefficient buffer of length P+1. On return,
#   g[n+1] is the coefficient of h^n.
#
#   invs::Vector{ComplexF64} Scratch buffer of length P+1 storing coefficients of 1/(s0+h).
#   Supplied externally to avoid allocations inside the patch loop.
#
#   ν::ComplexF64 Complex spectral parameter.
#
#   G0::ComplexF64 Value G(s0).
#
#   Gp0::ComplexF64 Derivative G'(s0).
#
#   s0::Float64 Patch center. Must be positive.
# =============================================================================
@inline function build_magnetic_patch_coeffs!(g::Vector{ComplexF64},invs::Vector{ComplexF64},ν::ComplexF64,G0::ComplexF64,Gp0::ComplexF64,s0::Float64)
    P=length(g)-1
    g[1]=G0
    g[2]=Gp0
    cc=ComplexF64(s0,0.0)
    invcc=inv(cc)
    invs[1]=invcc
    @inbounds for m in 1:P
        invs[m+1]=-invs[m]*invcc
    end
    c2=cc*cc
    fourν=4ν
    @inbounds for n in 0:(P-2)
        rhs=ComplexF64(0.0,0.0)
        for m in 0:n
            rhs-=invs[m+1]*ComplexF64(n-m+1,0.0)*g[n-m+2]
        end
        rhs+=(c2-fourν)*g[n+1]
        n>=1 && (rhs+=2cc*g[n])
        n>=2 && (rhs+=g[n-1])
        g[n+3]=rhs/ComplexF64((n+2)*(n+1),0.0)
    end
    return nothing
end

# =============================================================================
# build_small_z_coeffs
#
# Purpose
# Build coefficients for the small-z log representation
#
#       F(z;ν)=A(z)log(z)+B(z),
#
# where
#
#       A(z)=Σ A_m z^m,
#       B(z)=Σ B_m z^m.
#
# Initial values
# The leading coefficients are
#
#       A_0 = a_ν,
#       B_0 = R_ν(0).
#
# Recurrence
#   The coefficients are generated from the differential equation satisfied by
#   F(z;ν). The logarithmic structure is necessary because z=0 is a
#   singular point of the confluent hypergeometric equation.
#
# Inputs
#   ν::ComplexF64 Complex spectral parameter.
#
#   R0::ComplexF64 Finite part R_ν(0), preferably computed with mpmath for accuracy.
#
# Keyword arguments
#   M::Int=24 Maximum order of the small-z expansion. The returned arrays have length M+1.
#
#   prec::Int=256 BigFloat precision used while building the coefficients.
#
# Output
#   (A,B)::Tuple{Vector{ComplexF64},Vector{ComplexF64}} Coefficient vectors for A(z) and B(z), rounded to ComplexF64.
#
# Notes
#   In `_update_small_z!`, A[1] and B[1] are overwritten with mpmath-computed
#   a_ν and R_ν(0). This prevents the small-z branch from inheriting a Float64
#   error floor.
# =============================================================================
function build_small_z_coeffs(ν::ComplexF64,R0::ComplexF64;a0::ComplexF64=magnetic_log_coeff(ν),M::Int=24,prec::Int=256)
    setprecision(BigFloat,prec) do
        νb=Complex{BigFloat}(BigFloat(real(ν)),BigFloat(imag(ν)))
        A=Vector{Complex{BigFloat}}(undef,M+1)
        B=Vector{Complex{BigFloat}}(undef,M+1)
        A[1]=Complex{BigFloat}(BigFloat(real(a0)),BigFloat(imag(a0)))
        B[1]=Complex{BigFloat}(BigFloat(real(R0)),BigFloat(imag(R0)))
        am1=zero(Complex{BigFloat})
        bm1=zero(Complex{BigFloat})
        for m in 0:(M-1)
            den=Complex{BigFloat}(BigFloat((m+1)^2),zero(BigFloat))
            ap1=(BigFloat("0.25")*am1-νb*A[m+1])/den
            bp1=(-BigFloat(2*(m+1))*ap1-νb*B[m+1]+BigFloat("0.25")*bm1)/den
            am1=A[m+1]
            bm1=B[m+1]
            A[m+2] = ap1
            B[m+2] = bp1
        end
        return ComplexF64.(A), ComplexF64.(B)
    end
end

# =============================================================================
# MagneticGreenSTaylorTable
#
# Purpose
#   Runtime table for fast evaluation of the Landau-regularized magnetic Green
#   radial factor at one fixed value of ν.
#
# Stored function
#   The table stores Taylor coefficients of
#
#       G(s;ν) = F(s²;ν),
#
#   where
#
#       F(z;ν) = -exp(-z/2)U(1/2-ν,1,z)/(4Γ(ν+1/2)).
#
# Small-z representation
#   For z<zsmall, the table does not extrapolate the s-Taylor patches. Instead
#   it evaluates the Frobenius/log expansion
#
#       F(z) = A(z)log(z)+B(z),
#
#   where the coefficients of A and B are stored in smallA and smallB.
#
# Fields
#   ν::ComplexF64 Complex spectral parameter for which the table is built.
#
#   zmin,zmax::Float64 Valid z-interval for the s-Taylor patches.
#
#   zsmall::Float64 Threshold below which the small-z log-series branch is used. Usually zsmall=zmin.
#
#   smin,smax::Float64 Corresponding interval in s=sqrt(z).
#
#   h::Float64 Uniform patch spacing in s.
#
#   P::Int Taylor degree per patch.
#
#   centers::Vector{Float64} Patch centers in s.
#
#   gcoeffs::Matrix{ComplexF64} Taylor coefficients of G(s). Column j stores the coefficients for the
#   patch centered at centers[j].
#
#   a_log::ComplexF64 Logarithmic coefficient aν=cos(πν)/(4π).
#
#   R0::ComplexF64 Finite regular part Rν(0).
#
#   smallA,smallB::Vector{ComplexF64}
#   Coefficients of A(z) and B(z) in the small-z expansion
#   F(z)=A(z)log(z)+B(z).
# =============================================================================
mutable struct MagneticGreenSTaylorTable
    ν::ComplexF64
    zmin::Float64
    zmax::Float64
    zsmall::Float64
    smin::Float64
    smax::Float64
    h::Float64
    P::Int
    centers::Vector{Float64}
    gcoeffs::Matrix{ComplexF64}
    acoeffs::Matrix{ComplexF64}
    a_log::ComplexF64
    R0::ComplexF64
    smallA::Vector{ComplexF64}
    smallB::Vector{ComplexF64}
end

# =============================================================================
# MagneticGreenSPrecomp
#
# Purpose
#  Store ν-independent table geometry for the s-Taylor representation.
#
# Fields
#   zmin,zmax::Float64 The s-Taylor table covers z∈[zmin,zmax].
#
#   zsmall::Float64 Threshold below which the small-z log series is used.
#
#   smin,smax::Float64 sqrt(zmin), sqrt(zmax).
#
#   h::Float64 Uniform spacing of patch centers in s.
#
#   P::Int Taylor degree per patch.
#
#   Msmall::Int Number of small-z log-series coefficients minus one.
#
#   Npatch::Int Number of s-Taylor patches.
#
#   centers::Vector{Float64} Patch centers in the s variable.
#
# Usage
#   Reuse this object for all ν values on a Beyn contour when zmin,zmax,h,P are
#   fixed.
# =============================================================================
struct MagneticGreenSPrecomp
    zmin::Float64
    zmax::Float64
    zsmall::Float64
    smin::Float64
    smax::Float64
    h::Float64
    P::Int
    Msmall::Int
    Npatch::Int
    centers::Vector{Float64}
end

# =============================================================================
# MagneticGreenSWorkspace
#
# Purpose
# Scratch storage for Taylor-table construction.
#
# Fields
#   gcoef::Vector{ComplexF64} Single-thread scratch buffer for one patch coefficient vector.
#
#   invs::Vector{ComplexF64} Single-thread scratch buffer for the expansion of 1/(s0+h).
#
#   gcoef_tls::Vector{Vector{ComplexF64}} Thread-local coefficient buffers for batched construction.
#
#   invs_tls::Vector{Vector{ComplexF64}} Thread-local inverse-series buffers for batched construction.
#
# Why it matters
# Without these buffers, every patch construction would allocate temporary
# vectors. In Beyn runs this would be catastrophic.
# =============================================================================
struct MagneticGreenSWorkspace
    gcoef::Vector{ComplexF64}
    invs::Vector{ComplexF64}
    gcoef_tls::Vector{Vector{ComplexF64}}
    invs_tls::Vector{Vector{ComplexF64}}
end

# =============================================================================
# MagneticGreenSWorkspace(P; threaded=true)
#
# Purpose
# Allocate scratch buffers for table construction.
#
# Inputs
#   P::Int Taylor degree. Buffers of length P+1 are allocated.
#
# Keyword arguments
#   threaded::Bool=true If true, allocate one pair of scratch buffers per Julia thread.
#
# Output
#   MagneticGreenSWorkspace Workspace object used by `build_MagneticGreenSTaylorTable!`.
# =============================================================================
@inline function MagneticGreenSWorkspace(;threaded::Bool=true)
    h_patch,P_patch=confluent_U_params()
    NT=threaded ? Threads.nthreads() : 1
    return MagneticGreenSWorkspace(
        Vector{ComplexF64}(undef,P_patch+1),
        Vector{ComplexF64}(undef,P_patch+1),
        [Vector{ComplexF64}(undef,P_patch+1) for _ in 1:NT],
        [Vector{ComplexF64}(undef,P_patch+1) for _ in 1:NT],
    )
end

# =============================================================================
# _magnetic_turning_s
#
# Purpose
# Return the clipped turning-point location in the s=sqrt(z) variable.
#
# The Taylor ODE
#
#     G''(s) = -G'(s)/s + (s^2 - 4ν)G(s)
#
# has its formal turning point at
#
#     s_t = 2sqrt(|ν|).
#
# This is the natural anchor for stable Taylor propagation. The value is clipped
# to the precomputed table interval [smin,smax].
# =============================================================================
@inline function _magnetic_turning_s(pre::MagneticGreenSPrecomp,ν::ComplexF64)
    return clamp(2sqrt(max(abs(ν),eps(Float64))),pre.smin,pre.smax)
end

# =============================================================================
# _magnetic_anchor_index
#
# Purpose
# Return the primary Taylor seed index.
#
# If anchor_s is supplied, it is used directly after clipping to
# [pre.smin,pre.smax]. Otherwise the automatic anchor is the ODE turning point
#
#     s_t = 2sqrt(|ν|).
#
# This replaces the older sqrt(|ν|) anchor. The factor 2 is essential because
# the ODE coefficient changes sign at s^2 = 4ν.
# =============================================================================
@inline function _magnetic_anchor_index(pre::MagneticGreenSPrecomp,ν::ComplexF64,anchor_s::Union{Nothing,Float64})
    s0=isnothing(anchor_s) ? _magnetic_turning_s(pre,ν) : anchor_s
    s0=clamp(s0,pre.smin,pre.smax)
    return clamp(Int(round((s0-pre.smin)/pre.h))+1,1,pre.Npatch)
end

# =============================================================================
# _magnetic_eta
#
# Purpose
# Return the dimensionless turning-point coverage ratio
#
#     η = zmax/(4|ν|).
#
# Since the transformed ODE has formal turning point z_t=4|ν|, η<1 means the
# table ends before the turning region, while η>1 means the table extends beyond
# it. This is used only to choose the numerical seeding strategy.
# =============================================================================
@inline function _magnetic_eta(pre::MagneticGreenSPrecomp,ν::ComplexF64)
    return pre.zmax/(4max(abs(ν),eps(Float64)))
end

# =============================================================================
# _magnetic_use_multi_seed
#
# Purpose
# Decide whether automatic multi-seed construction should be used.
#
# Multi-seeding is enabled when the table reaches sufficiently close to or past
# the turning point z_t=4|ν|. It is disabled when anchor_s is supplied, because
# a manual anchor means the caller explicitly requested single-seed propagation.
# =============================================================================
@inline function _magnetic_use_multi_seed(pre::MagneticGreenSPrecomp,ν::ComplexF64)
    return _magnetic_eta(pre,ν)>magnetic_turning_eta_crit()
end

# =============================================================================
# _magnetic_seed_span
#
# Purpose
# Choose the approximate spacing in s between successive mpmath seeds.
#
# The span is based on the remaining interval [s_t,smax] and the coverage ratio
# η=zmax/(4|ν|). Small ν or large zmax gives large η and therefore more seeds.
# The result is clamped between MAGNETIC_MIN_SEED_SPAN and
# MAGNETIC_MAX_SEED_SPAN to avoid both excessive seed density and overly long
# unstable propagation segments.
# =============================================================================
@inline function _magnetic_seed_span(pre::MagneticGreenSPrecomp,ν::ComplexF64,jt::Int)
    span=pre.smax-pre.centers[jt]
    span<=0 && return Inf
    nη=max(1,ceil(Int,_magnetic_eta(pre,ν)))
    return clamp(span/nη,magnetic_min_seed_span(),magnetic_max_seed_span())
end

# =============================================================================
# _magnetic_seed_indices
#
# Purpose
# Construct the list of Taylor-patch indices where mpmath seeds are inserted.
#
# If no multi-seeding is needed, this returns only the primary anchor index. If
# multi-seeding is active, the first seed is placed near the turning point
# s_t=2sqrt(|ν|), additional seeds are spaced toward smax, and the final seed is
# forced to be the last patch. Neighboring seeded intervals are later filled
# from both sides and joined at midpoint indices.
# =============================================================================
function _magnetic_seed_indices(pre::MagneticGreenSPrecomp,ν::ComplexF64,anchor_s::Union{Nothing,Float64})
    jt=_magnetic_anchor_index(pre,ν,anchor_s)
    if anchor_s!==nothing || !_magnetic_use_multi_seed(pre,ν) || jt>=pre.Npatch-1
        return [jt]
    end
    Δs=_magnetic_seed_span(pre,ν,jt)
    inds=Int[jt]
    while true
        snext=pre.centers[inds[end]]+Δs
        snext>=pre.smax-pre.h && break
        j=clamp(Int(round((snext-pre.smin)/pre.h))+1,inds[end]+1,pre.Npatch)
        j>=pre.Npatch && break
        push!(inds,j)
    end
    inds[end]!=pre.Npatch && push!(inds,pre.Npatch)
    return inds
end

# =============================================================================
# _store_seed_patch!
#
# Purpose
# Evaluate one high-precision mpmath seed and store the corresponding local
# Taylor patch in column j of coefficient matrix C.
#
# The seedfun argument is either seed_G_Gp_mpmath for G(s)=F(s²), or
# seed_A_Ap_mpmath for H(s)=Aν(s²). This makes the propagation code shared by
# both tables.
# =============================================================================
@inline function _store_seed_patch!(C::Matrix{ComplexF64},pre::MagneticGreenSPrecomp,ws::MagneticGreenSWorkspace,ν::ComplexF64,seedfun,j::Int,mp_dps::Int)
    V,Vp=seedfun(pre.centers[j],ν;dps=mp_dps)
    build_magnetic_patch_coeffs!(ws.gcoef,ws.invs,ν,V,Vp,pre.centers[j])
    @inbounds for n in 1:(pre.P+1)
        C[n,j]=ws.gcoef[n]
    end
    return nothing
end

# =============================================================================
# _propagate_right! / _propagate_left!
#
# Purpose
# Fill Taylor patches by local analytic continuation from an already known
# neighboring patch.
#
# Right propagation evaluates the previous patch at the next center; left
# propagation evaluates the next patch at the previous center. In both cases the
# value and derivative are used as Cauchy data for the same second-order ODE,
# and build_magnetic_patch_coeffs! generates the new local Taylor coefficients.
# =============================================================================
function _propagate_right!(C::Matrix{ComplexF64},pre::MagneticGreenSPrecomp,ws::MagneticGreenSWorkspace,ν::ComplexF64,j0::Int,j1::Int)
    j1<=j0 && return nothing
    @inbounds for j in (j0+1):j1
        h=pre.centers[j]-pre.centers[j-1]
        V=horner_eval_col(C,j-1,h)
        Vp=horner_deriv_col(C,j-1,h)
        build_magnetic_patch_coeffs!(ws.gcoef,ws.invs,ν,V,Vp,pre.centers[j])
        for n in 1:(pre.P+1)
            C[n,j]=ws.gcoef[n]
        end
    end
    return nothing
end
function _propagate_left!(C::Matrix{ComplexF64},pre::MagneticGreenSPrecomp,ws::MagneticGreenSWorkspace,ν::ComplexF64,j0::Int,j1::Int)
    j1>=j0 && return nothing
    @inbounds for j in (j0-1):-1:j1
        h=pre.centers[j]-pre.centers[j+1]
        V=horner_eval_col(C,j+1,h)
        Vp=horner_deriv_col(C,j+1,h)
        build_magnetic_patch_coeffs!(ws.gcoef,ws.invs,ν,V,Vp,pre.centers[j])
        for n in 1:(pre.P+1)
            C[n,j]=ws.gcoef[n]
        end
    end
    return nothing
end

# =============================================================================
# build_MagneticGreenSPrecomp
#
# Purpose
# Build the ν-independent s-grid and table configuration.
#
# Inputs / keyword arguments
#   zmin::Float64=1e-3 Lower z value for the s-Taylor table.
#
#   zmax::Float64=900.0 Upper z value for the s-Taylor table. In production this should satisfy zmax ≥ max_{i,j}|x_i-x_j|²/b².
#
#   zsmall::Float64=zmin Threshold for switching to the small-z log-series branch.
#
#   h::Float64=0.00001 Uniform spacing in s=sqrt(z).
#
#   P::Int=6 Taylor degree per patch.
#
#   Msmall::Int=16 Order of the small-z log expansion.
#
# Output
#   MagneticGreenSPrecomp Precomputed table layout.
# =============================================================================
function build_MagneticGreenSPrecomp(;zmin::Float64=1e-3,zmax::Float64=900.0,zsmall::Float64=zmin,Msmall::Int=16)
    h_patch,P_patch=confluent_U_params()
    @assert zmin>0 && zmax>zmin && h_patch>0 && P_patch>=2
    @assert 0<zsmall<=zmin
    smin=sqrt(zmin);smax=sqrt(zmax)
    Npatch=Int(ceil((smax-smin)/h_patch))+1
    centers=Vector{Float64}(undef,Npatch)
    @inbounds for i in 1:Npatch
        centers[i]=smin+(i-1)*h_patch
    end
    centers[end]=smax
    return MagneticGreenSPrecomp(zmin,zmax,zsmall,smin,smax,h_patch,P_patch,Msmall,Npatch,centers)
end

# =============================================================================
# alloc_MagneticGreenSTaylorTable
#
# Purpose
# Allocate one runtime Taylor table compatible with a precomputed s-grid.
#
# Inputs
#   pre::MagneticGreenSPrecomp Table layout produced by `build_MagneticGreenSPrecomp`.
#
# Keyword arguments
#   ν::ComplexF64=0+0im Initial spectral parameter used to initialize constants and small-z
#   coefficients. The table should still be filled later with
#   `build_MagneticGreenSTaylorTable!`.
#
# Output
#   MagneticGreenSTaylorTable Table with allocated coefficient matrix.
# =============================================================================
@inline function alloc_MagneticGreenSTaylorTable(pre::MagneticGreenSPrecomp;ν::ComplexF64=0.0+0.0im)
    gcoeffs=Matrix{ComplexF64}(undef,pre.P+1,pre.Npatch)
    acoeffs=Matrix{ComplexF64}(undef,pre.P+1,pre.Npatch)
    R0=magnetic_R0(ν)
    A,B=build_small_z_coeffs(ν,R0;M=pre.Msmall)
    return MagneticGreenSTaylorTable(ν,pre.zmin,pre.zmax,pre.zsmall,pre.smin,pre.smax,pre.h,pre.P,pre.centers,gcoeffs,acoeffs,magnetic_log_coeff(ν),R0,A,B)
end

# =============================================================================
# alloc_MagneticGreenSTaylorTables
#
# Purpose
# Allocate several Taylor tables sharing the same s-grid.
#
# Inputs
#   pre::MagneticGreenSPrecomp Shared table layout.
#
#   Nν::Int Number of tables to allocate.
#
# Keyword arguments
#   ν::ComplexF64=0+0im Initial dummy spectral parameter.
#
# Output
#   Vector{MagneticGreenSTaylorTable} One table per requested ν.
# =============================================================================
@inline function alloc_MagneticGreenSTaylorTables(pre::MagneticGreenSPrecomp,Nν::Int;ν::ComplexF64=0.0+0.0im)
    tabs=Vector{MagneticGreenSTaylorTable}(undef,Nν)
    @inbounds for i in 1:Nν
        tabs[i]=alloc_MagneticGreenSTaylorTable(pre;ν=ν)
    end
    return tabs
end

# =============================================================================
# _update_small_z!
#
# Purpose
# Update all ν-dependent small-z data inside an existing table.
#
# What is updated
#   - tab.ν
#   - tab.a_log
#   - tab.R0
#   - tab.smallA
#   - tab.smallB
#
# Inputs
#   tab::MagneticGreenSTaylorTable Table to mutate.
#
#   pre::MagneticGreenSPrecomp Table layout, used mainly for Msmall.
#
#   ν::ComplexF64 New spectral parameter.
#
# Keyword arguments
#   mp_dps::Int=100 mpmath precision for a_ν and Rν(0).
# =============================================================================
function _update_small_z!(tab::MagneticGreenSTaylorTable,pre::MagneticGreenSPrecomp,ν::ComplexF64;mp_dps::Int=100)
    tab.ν=ν
    tab.a_log=magnetic_log_coeff_mpmath(ν;dps=mp_dps)
    tab.R0=magnetic_R0_mpmath(ν;dps=mp_dps)
    A,B=build_small_z_coeffs(ν,tab.R0;a0=tab.a_log,M=pre.Msmall)
    A[1]=tab.a_log
    B[1]=tab.R0
    resize!(tab.smallA,length(A))
    resize!(tab.smallB,length(B))
    copyto!(tab.smallA,A)
    copyto!(tab.smallB,B)
    return nothing
end

# =============================================================================
# _build_coeff_table!
#
# Purpose
# Build one complete Taylor coefficient table using either single-seed or
# dynamic multi-seed propagation.
#
# For one seed, the table is propagated left and right from the anchor. For
# multiple seeds, every seed patch is first computed directly with mpmath. Each
# interval between neighboring seeds is then filled from both sides and joined
# at the midpoint. This limits accumulated propagation error over long
# post-turning intervals while preserving allocation-free runtime evaluation.
#
# The same routine is used for:
#
#     G(s;ν)=F(s²;ν),
#     H(s;ν)=Aν(s²).
# =============================================================================
function _build_coeff_table!(C::Matrix{ComplexF64},pre::MagneticGreenSPrecomp,ws::MagneticGreenSWorkspace,ν::ComplexF64,seedfun;mp_dps::Int=80,anchor_s::Union{Nothing,Float64}=nothing)
    seeds=_magnetic_seed_indices(pre,ν,anchor_s)
    if length(seeds)==1
        j0=seeds[1]
        _store_seed_patch!(C,pre,ws,ν,seedfun,j0,mp_dps)
        _propagate_right!(C,pre,ws,ν,j0,pre.Npatch)
        _propagate_left!(C,pre,ws,ν,j0,1)
        return nothing
    end
    @inbounds for j in seeds
        _store_seed_patch!(C,pre,ws,ν,seedfun,j,mp_dps)
    end
    _propagate_left!(C,pre,ws,ν,seeds[1],1)
    @inbounds for k in 1:(length(seeds)-1)
        jl=seeds[k]
        jr=seeds[k+1]
        jc=(jl+jr)>>>1
        _propagate_right!(C,pre,ws,ν,jl,jc)
        _propagate_left!(C,pre,ws,ν,jr,jc+1)
    end
    return nothing
end

# =============================================================================
# build_A_coeff_table!
#
# Purpose
# Build Taylor patches for
#
#     H(s;ν)=Aν(s²),
#
# where Aν(z) is the z-dependent logarithmic coefficient in
#
#     F(z;ν)=Aν(z)log(z)+Bν(z).
#
# The function H satisfies the same s-ODE as G(s;ν)=F(s²;ν),
#
#     H''(s) = -H'(s)/s + (s^2 - 4ν)H(s).
#
# Propagation strategy
#   - If anchor_s is supplied, use single-seed propagation from that anchor.
#   - Otherwise anchor first near the turning point s_t=2sqrt(|ν|).
#   - If zmax/(4|ν|) is large, insert additional mpmath seeds between s_t and
#     smax. The intervals between neighboring seeds are filled from both sides.
# =============================================================================
function build_A_coeff_table!(tab::MagneticGreenSTaylorTable,pre::MagneticGreenSPrecomp,ws::MagneticGreenSWorkspace,ν::ComplexF64;mp_dps::Int=80,anchor_s::Union{Nothing,Float64}=nothing)
    _build_coeff_table!(tab.acoeffs,pre,ws,ν,seed_A_Ap_mpmath;mp_dps=mp_dps,anchor_s=anchor_s)
    return nothing
end

# =============================================================================
# build_MagneticGreenSTaylorTable!
#
# Purpose
# Build the full Taylor table for one ν in preallocated storage.
#
# What is updated
#   1. Small-z logarithmic data: aν, Rν(0), A(z), B(z).
#   2. Taylor patches for G(s;ν)=F(s²;ν).
#   3. Taylor patches for Aν(s²), the z-dependent logarithmic coefficient.
#
# Propagation strategy
# The ODE
#
#     G''(s) = -G'(s)/s + (s^2 - 4ν)G(s)
#
# has formal turning point s_t=2sqrt(|ν|). The builder anchors there by default.
# If the table extends far beyond z_t=4|ν|, it uses dynamic multi-seeding: one
# seed near s_t plus additional seeds toward smax. This is important for small ν
# and large zmax, where a two-seed construction is still too long.
#
# If anchor_s is provided, automatic multi-seeding is disabled.
# =============================================================================
function build_MagneticGreenSTaylorTable!(tab::MagneticGreenSTaylorTable,pre::MagneticGreenSPrecomp,ws::MagneticGreenSWorkspace,ν::ComplexF64;mp_dps::Int=80,anchor_s::Union{Nothing,Float64}=nothing)
    @assert pre.centers===tab.centers
    @assert pre.P==tab.P && pre.Npatch==size(tab.gcoeffs,2) && pre.Npatch==size(tab.acoeffs,2)
    _update_small_z!(tab,pre,ν;mp_dps=mp_dps)
    _build_coeff_table!(tab.gcoeffs,pre,ws,ν,seed_G_Gp_mpmath;mp_dps=mp_dps,anchor_s=anchor_s)
    _build_coeff_table!(tab.acoeffs,pre,ws,ν,seed_A_Ap_mpmath;mp_dps=mp_dps,anchor_s=anchor_s)
    return nothing
end

# =============================================================================
# build_MagneticGreenSTaylorTable
#
# Purpose
# Convenience allocation-and-build wrapper for a single ν.
#
# Inputs
#   ν::ComplexF64 Spectral parameter.
#
# Keyword arguments
#   zmin,zmax,zsmall,h,P,Msmall,mp_dps,anchor_s Forwarded to the precomp/table builder.
#
# Output
#   MagneticGreenSTaylorTable Fully built table.
#
# Notes
# For production Beyn loops, prefer the in-place version to avoid repeated
# allocations.
# =============================================================================
function build_MagneticGreenSTaylorTable(ν::ComplexF64;zmin::Float64=1e-3,zmax::Float64=900.0,zsmall::Float64=zmin,Msmall::Int=16,mp_dps::Int=80,anchor_s::Union{Nothing,Float64}=nothing)
    h_patch,P_patch=confluent_U_params()
    pre=build_MagneticGreenSPrecomp(;zmin=zmin,zmax=zmax,zsmall=zsmall,Msmall=Msmall)
    ws=MagneticGreenSWorkspace(;threaded=false)
    tab=alloc_MagneticGreenSTaylorTable(pre;ν=ν)
    build_MagneticGreenSTaylorTable!(tab,pre,ws,ν;mp_dps=mp_dps,anchor_s=anchor_s)
    return tab
end

# =============================================================================
# _mag_patch_index
#
# Purpose
# Map an s value to a Taylor patch index.
#
# Inputs
#   tab::MagneticGreenSTaylorTable Runtime table.
#
#   s::Float64 Evaluation point in s=sqrt(z).
#
# Output
#   Int Patch index.
# =============================================================================
@inline function _mag_patch_index(tab::MagneticGreenSTaylorTable,s::Float64)
    if s<=tab.smin
        return 1
    elseif s>=tab.smax
        return length(tab.centers)
    else
        t=(s-tab.smin)/tab.h
        idx=Int(floor(t))+1
        abs(t-round(t))<64*eps(t) && (idx=Int(round(t))+1)
        return clamp(idx,1,length(tab.centers))
    end
end

# =============================================================================
# _small_F
#
# Purpose
# Evaluate F(z;ν) for z<zsmall using the small-z log expansion.
#
# Formula
# F(z)=A(z)log(z)+B(z).
#
# Inputs
#   tab::MagneticGreenSTaylorTable Table containing smallA/smallB.
#
#   z::Float64 Positive small z.
#
# Output
#   ComplexF64  F(z;ν)
# =============================================================================
@inline function _small_F(tab::MagneticGreenSTaylorTable,z::Float64)
    L=log(z)
    A=horner_eval_vec(tab.smallA,z)
    B=horner_eval_vec(tab.smallB,z)
    return A*L+B
end

# =============================================================================
# _small_R
#
# Purpose
# Evaluate the regular remainder Rν(z) for z<zsmall.
#
# Formula
# Rν(z)=F(z)-aν log(z) = (A(z)-aν)log(z)+B(z).
#
# Inputs
#   tab::MagneticGreenSTaylorTable Table containing small-z coefficients.
#
#   z::Float64 Positive small z.
#
# Output
#   ComplexF64 Regular remainder Rν(z).
# =============================================================================
@inline function _small_R(tab::MagneticGreenSTaylorTable,z::Float64)
    L=log(z)
    A=horner_eval_vec(tab.smallA,z)
    B=horner_eval_vec(tab.smallB,z)
    return (A-tab.a_log)*L+B
end

# =============================================================================
# _small_Fz
#
# Purpose
# Evaluate dF/dz for z<zsmall.
#
# Formula
#
#   F(z)=A(z)log(z)+B(z),
#
#   then
#
#   F_z(z)=A'(z)log(z)+A(z)/z+B'(z).
#
# Inputs
#   tab::MagneticGreenSTaylorTable Table containing small-z coefficients.
#
#   z::Float64 Positive small z.
#
# Output
#   ComplexF64 Derivative F_z(z;ν).
# =============================================================================
@inline function _small_Fz(tab::MagneticGreenSTaylorTable,z::Float64)
    L=log(z)
    A=horner_eval_vec(tab.smallA,z)
    Ap=horner_deriv_vec(tab.smallA,z)
    Bp=horner_deriv_vec(tab.smallB,z)
    return Ap*L+A/z+Bp
end

# =============================================================================
# _eval_G
#
# Purpose
# Evaluate G(s;ν)=F(s²;ν) from the s-Taylor table.
#
# Inputs
#   tab::MagneticGreenSTaylorTable Runtime table.
#
#   s::Float64 Positive s value.
#
# Output
#   ComplexF64 G(s;ν).
# =============================================================================
@inline function _eval_G(tab::MagneticGreenSTaylorTable,s::Float64)
    idx=_mag_patch_index(tab,s)
    return horner_eval_col(tab.gcoeffs,idx,s-tab.centers[idx])
end

# =============================================================================
# _eval_dGds
#
# Purpose
# Evaluate dG/ds from the s-Taylor table.
#
# Inputs
#   tab::MagneticGreenSTaylorTable Runtime table.
#
#   s::Float64 Positive s value.
#
# Output
#   ComplexF64 G'(s;ν).
# =============================================================================
@inline function _eval_dGds(tab::MagneticGreenSTaylorTable,s::Float64)
    idx=_mag_patch_index(tab,s)
    return horner_deriv_col(tab.gcoeffs,idx,s-tab.centers[idx])
end

# =============================================================================
# _eval_F
#
# Purpose
# Evaluate the regularized radial magnetic Green factor F(z;ν).
#
# Strategy
#   - If z<zsmall, use the small-z log expansion.
#   - Otherwise, use G(s)=F(s²) with s=sqrt(z).
#
# Inputs
#   tab::MagneticGreenSTaylorTable Runtime table.
#
#   z::Float64 Dimensionless squared distance z=r²/b².
#
# Output
#   ComplexF64 F(z;ν).
# =============================================================================
@inline function _eval_F(tab::MagneticGreenSTaylorTable,z::Float64)
    z==0.0 && return ComplexF64(Inf,0.0)
    z<tab.zsmall && return _small_F(tab,z)
    return _eval_G(tab,sqrt(z))
end

# =============================================================================
# _eval_Alog
#
# Purpose
# Evaluate the full logarithmic coefficient
#
#     Aν(z)=cos(πν)/(4π)*exp(-z/2)*1F1(1/2-ν;1;z),
#
# appearing in the exact local split
#
#     F(z;ν)=Aν(z)log(z)+Bν(z).
#
# This is the coefficient that should multiply the Kress logarithm
#
#     log(4sin²((t-τ)/2)).
#
# Notes
#   At z=0, Aν(0)=aν=cos(πν)/(4π).
# =============================================================================
@inline function _eval_Alog(tab::MagneticGreenSTaylorTable,z::Float64)
    z==0.0 && return tab.a_log
    z<tab.zsmall && return horner_eval_vec(tab.smallA,z)
    s=sqrt(z)
    idx=_mag_patch_index(tab,s)
    return horner_eval_col(tab.acoeffs,idx,s-tab.centers[idx])
end

# =============================================================================
# _eval_Blog
#
# Purpose
# Evaluate the Kress-smooth finite part
#
#     Bν(z)=F(z;ν)-Aν(z)log(z).
#
# This differs from _eval_R, which subtracts only the constant coefficient
# aν log(z). For high-order Kress assembly, use _eval_Alog and _eval_Blog,
# not _eval_R.
#
# Notes
#   At z=0, Bν(0)=Rν(0).
# =============================================================================
@inline function _eval_Blog(tab::MagneticGreenSTaylorTable,z::Float64)
    z==0.0 && return tab.R0
    z<tab.zsmall && return horner_eval_vec(tab.smallB,z)
    return _eval_F(tab,z)-_eval_Alog(tab,z)*log(z)
end

# =============================================================================
# _eval_Alog_z
#
# Purpose
# Evaluate dAν/dz.
#
# This is useful for differentiated Kress splits or for constructing DLP
# kernels in a form where the derivative of the logarithmic coefficient is
# separated explicitly.
#
# Notes
#   The derivative is finite at z=0, but this routine intentionally errors at
#   z=0 because the finite value should be taken from the small-z coefficient
#   A₁ if needed.
# =============================================================================
@inline function _eval_Alog_z(tab::MagneticGreenSTaylorTable,z::Float64)
    z==0.0 && error("Alog_z is finite at z=0; use tab.smallA[2] explicitly if needed.")
    z<tab.zsmall && return horner_deriv_vec(tab.smallA,z)
    s=sqrt(z)
    idx=_mag_patch_index(tab,s)
    return horner_deriv_col(tab.acoeffs,idx,s-tab.centers[idx])/(2s)
end

# =============================================================================
# _eval_R
#
# Purpose
# Evaluate the constant-log remainder
#
#     Rν(z)=F(z;ν)-aν log(z).
#
#   This is algebraically useful and finite at z=0, but it is not the
#   Kress-smooth finite part. For Kress assembly use
#
#     _eval_Alog(tab,z), _eval_Blog(tab,z).
# =============================================================================
@inline function _eval_R(tab::MagneticGreenSTaylorTable,z::Float64)
    z==0.0 && return tab.R0
    z<tab.zsmall && return _small_R(tab,z)
    return _eval_F(tab,z)-tab.a_log*log(z)
end

# =============================================================================
# _eval_Fz
#
# Purpose
# Evaluate dF/dz.
#
# Strategy
#   - If z<zsmall, use the differentiated small-z log series.
#   - Otherwise, use
#
#  F_z(z)=G'(sqrt(z))/(2sqrt(z)).
#
# Inputs
#   tab::MagneticGreenSTaylorTable Runtime table.
#
#   z::Float64 Positive dimensionless squared distance.
#
# Output
#   ComplexF64 F_z(z;ν).
#
# Notes
#   Not needed for SLP assembly, but useful for DLP/normal derivatives later.
# =============================================================================
@inline function _eval_Fz(tab::MagneticGreenSTaylorTable,z::Float64)
    z==0.0 && error("Fz is singular at z=0.")
    z<tab.zsmall && return _small_Fz(tab,z)
    s=sqrt(z)
    return _eval_dGds(tab,s)/(2s)
end

# =============================================================================
# _eval_F! / _eval_R! / _eval_Fz!
#
# Purpose
# Vectorized, allocation-free evaluation for many z values.
#
# Inputs
#   out::AbstractVector{ComplexF64} Preallocated output vector.
#
#   tab::MagneticGreenSTaylorTable Runtime table.
#
#   zvec::AbstractVector{Float64} Input z values.
# =============================================================================
function _eval_F!(out::AbstractVector{ComplexF64},tab::MagneticGreenSTaylorTable,zvec::AbstractVector{Float64})
    @inbounds for i in eachindex(zvec)
        out[i]=_eval_F(tab,zvec[i])
    end
    return nothing
end
function _eval_R!(out::AbstractVector{ComplexF64},tab::MagneticGreenSTaylorTable,zvec::AbstractVector{Float64})
    @inbounds for i in eachindex(zvec)
        out[i]=_eval_R(tab,zvec[i])
    end
    return nothing
end
function _eval_Fz!(out::AbstractVector{ComplexF64},tab::MagneticGreenSTaylorTable,zvec::AbstractVector{Float64})
    @inbounds for i in eachindex(zvec)
        out[i]=_eval_Fz(tab,zvec[i])
    end
    return nothing
end
function _eval_Alog!(out::AbstractVector{ComplexF64},tab::MagneticGreenSTaylorTable,zvec::AbstractVector{Float64})
    @inbounds for i in eachindex(zvec)
        out[i]=_eval_Alog(tab,zvec[i])
    end
    return nothing
end
function _eval_Blog!(out::AbstractVector{ComplexF64},tab::MagneticGreenSTaylorTable,zvec::AbstractVector{Float64})
    @inbounds for i in eachindex(zvec)
        out[i]=_eval_Blog(tab,zvec[i])
    end
    return nothing
end

#################################
#### LANDAU POLE SUBTRACTION ####
#################################

@inline landau_pole(n::Int)=n+0.5

@inline function laguerreL0(n::Int,z::Float64)
    n==0 && return 1.0
    Lm2=1.0
    Lm1=1.0-z
    n==1 && return Lm1
    @inbounds for k in 2:n
        L=((2*k-1-z)*Lm1-(k-1)*Lm2)/k
        Lm2,Lm1=Lm1,L
    end
    return Lm1
end

@inline function laguerreLα1(n::Int,z::Float64)
    n==0 && return 1.0
    Lm2=1.0
    Lm1=2.0-z
    n==1 && return Lm1
    @inbounds for k in 2:n
        L=((2*k-z)*Lm1-k*Lm2)/k
        Lm2,Lm1=Lm1,L
    end
    return Lm1
end

@inline function landau_residue(n::Int,z::Float64)
    return -exp(-0.5z)*laguerreL0(n,z)*inv4π
end

@inline function landau_residue_z(n::Int,z::Float64)
    L=laguerreL0(n,z)
    Lz=n==0 ? 0.0 : -laguerreLα1(n-1,z)
    return -exp(-0.5z)*(Lz-0.5L)*inv4π
end

@inline function _pole_correction(tab::MagneticGreenSTaylorTable,z::Float64)
    isempty(tab.pole_ns) && return 0.0+0.0im
    ν=tab.ν
    s=0.0+0.0im
    @inbounds for n in tab.pole_ns
        s+=landau_residue(n,z)/(ν-landau_pole(n))
    end
    return s
end

@inline function _pole_correction_z(tab::MagneticGreenSTaylorTable,z::Float64)
    isempty(tab.pole_ns) && return 0.0+0.0im
    ν=tab.ν
    s=0.0+0.0im
    @inbounds for n in tab.pole_ns
        s+=landau_residue_z(n,z)/(ν-landau_pole(n))
    end
    return s
end

############################################################
##### FOR TESTING: REFERENCE EVALUATION WITH MP MPMATH #####
############################################################

function Fz_ref_mpmath(z::Float64,ν::ComplexF64;dps::Int=100)
    _mpctx[].dps=dps
    z_py=_mpf[](z);ν_py=_mpc[](real(ν),imag(ν));a_py=_mpf[](0.5)-ν_py
    C=-1/(4*_mp_gamma[](ν_py+_mpf[](0.5)))
    U0=_mp_hyperu[](a_py,1,z_py)
    U1=_mp_hyperu[](a_py+1,2,z_py)
    Fz=C*_mp_exp[](-z_py/2)*(-_mpf[](0.5)*U0-a_py*U1)
    return ComplexF64(pycall(_pyfloat[],Float64,Fz.real),pycall(_pyfloat[],Float64,Fz.imag))
end

function Alog_z_ref_mpmath(z::Float64,ν::ComplexF64;dps::Int=100)
    _mpctx[].dps=dps
    z_py=_mpf[](z);ν_py=_mpc[](real(ν),imag(ν));a_py=_mpf[](0.5)-ν_py
    c=_mp_cos[](_mp_pi[]*ν_py)/(4*_mp_pi[])
    M0=_mp_hyp1f1[](a_py,1,z_py)
    M1=_mp_hyp1f1[](a_py+1,2,z_py)
    Az=c*_mp_exp[](-z_py/2)*(a_py*M1-_mpf[](0.5)*M0)
    return ComplexF64(pycall(_pyfloat[],Float64,Az.real),pycall(_pyfloat[],Float64,Az.imag))
end

function Blog_ref_mpmath(z::Float64,ν::ComplexF64;dps::Int=100)
    return F_ref_mpmath(z,ν;dps=dps)-Alog_ref_mpmath(z,ν;dps=dps)*log(z)
end

function magnetic_validate_taylor_config!(;nus=ComplexF64[20+0im,40+0im,80+0im,600+0im,2000+0im,2003.37+0.5im],zmax::Float64=500.0,zmin::Float64=1e-3,zsmall::Float64=1e-3,Msmall::Int=30,mp_dps::Int=160,ztests=nothing,rtol::Float64=magnetic_target_accuracy(),atol::Float64=1e-14,verbose::Bool=false,test_derivatives::Bool=true,test_split::Bool=true)
    zs=isnothing(ztests) ? Float64[
        1e-15,1e-12,1e-9,1e-6,1e-4,zsmall,1.5zsmall,
        1e-2,3e-2,0.12,0.35,0.75,1.0,
        0.25zmax,0.5zmax,0.75zmax,zmax,
    ] : Float64.(ztests)
    ws=MagneticGreenSWorkspace(;threaded=false)
    pre=build_MagneticGreenSPrecomp(;zmin=zmin,zmax=zmax,zsmall=zsmall,Msmall=Msmall)
    for ν in nus
        tab=alloc_MagneticGreenSTaylorTable(pre;ν=ν)
        build_MagneticGreenSTaylorTable!(tab,pre,ws,ν;mp_dps=mp_dps)
        for z in zs
            0<z<=zmax || continue
            Fref=F_ref_mpmath(z,ν;dps=mp_dps)
            Fval=_eval_F(tab,z)
            Aval=_eval_Alog(tab,z)
            Aref=Alog_ref_mpmath(z,ν;dps=mp_dps)
            errF=abs(Fval-Fref); relF=errF/max(abs(Fref),eps(Float64))
            errA=abs(Aval-Aref); relA=errA/max(abs(Aref),eps(Float64))
            verbose && println("ν=",ν," z=",z," relF=",relF," relA=",relA)
            (relF<=rtol || errF<=atol) || error("F validation failed: ν=$ν z=$z rel=$relF abs=$errF")
            (relA<=rtol || errA<=atol) || error("Alog validation failed: ν=$ν z=$z rel=$relA abs=$errA")
            if test_derivatives
                Fzref=Fz_ref_mpmath(z,ν;dps=mp_dps)
                Fzval=_eval_Fz(tab,z)
                Azref=Alog_z_ref_mpmath(z,ν;dps=mp_dps)
                Azval=_eval_Alog_z(tab,z)
                errFz=abs(Fzval-Fzref); relFz=errFz/max(abs(Fzref),eps(Float64))
                errAz=abs(Azval-Azref); relAz=errAz/max(abs(Azref),eps(Float64))
                verbose && println("ν=",ν," z=",z," relFz=",relFz," relAz=",relAz)
                (relFz<=max(10rtol,1e-12) || errFz<=10atol) || error("Fz validation failed: ν=$ν z=$z rel=$relFz abs=$errFz")
                (relAz<=max(10rtol,1e-12) || errAz<=10atol) || error("Alog_z validation failed: ν=$ν z=$z rel=$relAz abs=$errAz")
            end
            if test_split
                Bval=_eval_Blog(tab,z)
                scale=max(abs(Fval),abs(Aval*log(z)),abs(Bval),1.0)
                err=abs(Fval-(Aval*log(z)+Bval))/scale
                err<=max(10rtol,100eps(Float64)) || error("Split validation failed: ν=$ν z=$z err=$err")
            end
        end
    end
    return true
end










##############################
########## TESTING ###########
##############################

# run only if file ran as a script
if abspath(PROGRAM_FILE)==@__FILE__

@inline function magnetic_test_zmax(b::T;D::T=T(10.0),safety::T=T(1.2)) where {T<:Real}
    return safety*(D/b)^2
end

function magnetic_ztests(zmax::Float64;zsmall::Float64=1e-3)
    candidates=[1e-15,1e-12,1e-9,1e-6,zsmall,1.5*zsmall,1e-2,3e-2,0.12,0.35,0.75,1.5,3.0,7.0,9.5,20.0,30.0,40.0,100.0,500.0,900.0,0.25*zmax,0.5*zmax,0.75*zmax,zmax]
    return candidates[candidates.<=zmax]
end

function run_magnetic_green_taylor_test(;ν=ComplexF64(2003.37,0.5),zmin=1e-3,zsmall=1e-3,Msmall=20,mp_dps=100,b=2.0)
    ws=MagneticGreenSWorkspace(;threaded=false)
    zmax=magnetic_test_zmax(b;D=10.0,safety=1.2)
    pre=build_MagneticGreenSPrecomp(;zmin=zmin,zmax=zmax,zsmall=zsmall,Msmall=Msmall)
    tab=alloc_MagneticGreenSTaylorTable(pre;ν=ν)

    println("\nBuild:")
    @time build_MagneticGreenSTaylorTable!(tab,pre,ws,ν;mp_dps=mp_dps)

    ztests=magnetic_ztests(zmax)
    N=1_000_000_00

    maxrel=0.0
    println("\nAccuracy:")
    for z in ztests
        ref=F_ref_mpmath(z,ν;dps=mp_dps)
        val=_eval_F(tab,z)
        err=abs(val-ref)
        rel=err/max(abs(ref),eps(Float64))
        maxrel=max(maxrel,rel)
        println("z = ",z,"  abs err = ",err,"  rel err = ",rel)
        @test rel < 5e-11 || err < 5e-13
    end

    for z in ztests
        ref=F_ref_mpmath(z,ν;dps=mp_dps)-tab.a_log*log(z)
        val=_eval_R(tab,z)
        @test abs(val-ref)/max(abs(ref),eps(Float64))<5e-10
    end

    for z in ztests
        z==0.0 && continue
        Aref=Alog_ref_mpmath(z,ν;dps=mp_dps)
        Aval=_eval_Alog(tab,z)
        Bref=Blog_ref_mpmath(z,ν;dps=mp_dps)
        Bval=_eval_Blog(tab,z)
        relA=abs(Aval-Aref)/max(abs(Aref),eps(Float64))
        relB=abs(Bval-Bref)/max(abs(Bref),eps(Float64))
        println("A/B z = ",z,"  relA = ",relA,"  relB = ",relB)
        @test relA < 5e-11 || abs(Aval-Aref)<5e-13
        @test relB < 5e-10 || abs(Bval-Bref)<5e-12
    end

    for z in ztests
        z==0.0 && continue
        F1=_eval_F(tab,z)
        F2=_eval_Alog(tab,z)*log(z)+_eval_Blog(tab,z)
        @test abs(F1-F2)/max(abs(F1),eps(Float64))<5e-13
    end

    @test isfinite(real(tab.R0))
    @test isfinite(imag(tab.R0))

    rng=MersenneTwister(1234)
    zvec=exp.(log(1e-15).+(log(zmax)-log(1e-15)).*rand(rng,N))
    out=Vector{ComplexF64}(undef,length(zvec))

    println("\nBenchmark scalar _eval_F:")
    @btime _eval_F($tab,1.234)

    println("\nBenchmark vector _eval_F! : $(N) evals:")
    @btime _eval_F!($out,$tab,$zvec)

    println("\nBenchmark vector _eval_R! : $(N) evals:")
    @btime _eval_R!($out,$tab,$zvec)

    println("\nBenchmark vector _eval_Fz! : $(N) evals:")
    @btime _eval_Fz!($out,$tab,$zvec)

    println("\nBenchmark vector _eval_Alog! : $(N) evals:")
    @btime _eval_Alog!($out,$tab,$zvec)

    println("\nBenchmark vector _eval_Blog! : $(N) evals:")
    @btime _eval_Blog!($out,$tab,$zvec)

    t=@belapsed _eval_F!($out,$tab,$zvec)
    println("\nThroughput _eval_F!: ",length(zvec)/t/1e6," million evals/s")
    println("ns/eval: ",1e9*t/length(zvec))
    println("max relative error = ",maxrel)

    return tab
end

@testset "MagneticGreenSTaylorTable" begin
    tab=run_magnetic_green_taylor_test(;Msmall=30,mp_dps=120)
end

end