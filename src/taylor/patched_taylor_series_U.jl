# RADIAL GREEN FACTOR
#   F(z;ν)=-Γ(1/2-ν)exp(-z/2)U(1/2-ν,1,z)/(4π),
#   z=|x-y|²/b².
#
# THE FULL MAGNETIC GREEN FUNCTION IS
#
#   G_B(x,y;ν)=exp(iΦ(x,y))F(z;ν),
#
# WITH THE MAGNETIC PHASE ASSEMBLED ELSEWHERE.
#
# NEAR z=0,
#
#   F(z;ν)=Aν(z)log(z)+Bν(z),
#   Aν(0)=1/(4π),
#   Bν(0)=[ψ(1/2-ν)+2γ]/(4π).
#
# FOR z>=zsmall WE PROPAGATE IN s=sqrt(z),
#
#   G(s;ν)=F(s²;ν),
#   G''(s)=-G'(s)/s+(s²-4ν)G(s).
# By: chatGPT 5.6 Sol, based on the library's old code
# Edits: MO 23/8/26

# USAGE HIERARCHY
#   build_MagneticGreenSPrecomp
#       -> alloc_MagneticGreenSTaylorTable
#       -> build_MagneticGreenSTaylorTable!
#           -> _update_small_z!
#           -> _build_coeff_table!
#               -> _magnetic_seed_indices
#               -> _store_seed_patch!
#               -> _propagate_left! / _propagate_right!
#       -> _eval_F / _eval_Fz / _eval_Alog / _eval_Alog_z / _eval_Blog
#
# validation:
#   magnetic_validate_taylor_config!
#       -> build_MagneticGreenSTaylorTable!
#       -> *_ref_mpmath
#
# TYPICAL USAGE
#   pre=build_MagneticGreenSPrecomp(;zmin=1e-3,zmax=500.0)
#   ws=MagneticGreenSWorkspace(;threaded=false)
#   tab=alloc_MagneticGreenSTaylorTable(pre;ν=ν)
#   build_MagneticGreenSTaylorTable!(tab,pre,ws,ν)
#   F=_eval_F(tab,z); Fz=_eval_Fz(tab,z)
#   A=_eval_Alog(tab,z); Az=_eval_Alog_z(tab,z); B=_eval_Blog(tab,z)

# for out of library testing, uncomment these imports. They are already imported in QuantumBilliards.jl
#using PyCall
#using SpecialFunctions
#using LinearAlgebra
#using Test
#using Random
#using BenchmarkTools
#using QuantumBilliards

#PYCALL_MPMATH_LOCK=QuantumBilliards.PYCALL_MPMATH_LOCK
#_mpctx=QuantumBilliards._mpctx
#_mpf=QuantumBilliards._mpf
#_mpc=QuantumBilliards._mpc
#_mp_digamma=QuantumBilliards._mp_digamma
#_mp_pi=QuantumBilliards._mp_pi
#_pyfloat=QuantumBilliards._pyfloat
#_mp_hyperu=QuantumBilliards._mp_hyperu
#_mp_exp=QuantumBilliards._mp_exp
#_mp_gamma=QuantumBilliards._mp_gamma
#_mp_hyp1f1=QuantumBilliards._mp_hyp1f1
#horner_eval_col=QuantumBilliards.horner_eval_col

const inv4π=1/(4*pi)

# =============================================================================
# TAYLOR-TABLE NUMERICAL CONFIGURATION
# =============================================================================
#
# h_patch
#   Spacing between neighboring Taylor centers in s=sqrt(z).
#   Smaller values reduce the local continuation distance, but increase the
#   number of patches, memory usage, build time, and accumulated roundoff.
#
# P_patch
#   Degree of each local Taylor polynomial. A patch stores coefficients
#   g_0,...,g_P, so the number of stored Taylor coefficients is P_patch+1.
#
# turning_eta_crit
#   Threshold for activating dynamic multi-seeding. The turning-region
#   coverage parameter is η=zmax/(4|ν|), since the formal turning point 
#   of the s-ODE is z_t=4|ν|. Multi-seeding is enabled when η>turning_eta_crit.
#
# min_seed_span, max_seed_span
#   Lower and upper bounds on the separation in s between additional
#   high-precision mpmath seeds used beyond the turning region. Smaller spans
#   improve propagation robustness at the cost of more expensive table builds.
#
# target_accuracy
#   Desired relative accuracy used by the validation routines. It is currently
#   a validation target only and does not automatically modify h_patch,
#   P_patch, or the seed spacing.
# =============================================================================
Base.@kwdef mutable struct UConfluentTaylorConfig
    h_patch::Float64=1e-4
    P_patch::Int=8
    turning_eta_crit::Float64=0.8
    min_seed_span::Float64=0.1
    max_seed_span::Float64=0.2
    target_accuracy::Float64=1e-12
end

const U_CONFLUENT_TAYLOR_CONFIG=UConfluentTaylorConfig()

# Individual configuration accessors.
@inline u_confluent_h_patch()=U_CONFLUENT_TAYLOR_CONFIG.h_patch
@inline u_confluent_P_patch()=U_CONFLUENT_TAYLOR_CONFIG.P_patch
@inline magnetic_turning_eta_crit()=U_CONFLUENT_TAYLOR_CONFIG.turning_eta_crit
@inline magnetic_min_seed_span()=U_CONFLUENT_TAYLOR_CONFIG.min_seed_span
@inline magnetic_max_seed_span()=U_CONFLUENT_TAYLOR_CONFIG.max_seed_span
@inline magnetic_target_accuracy()=U_CONFLUENT_TAYLOR_CONFIG.target_accuracy
@inline confluent_U_params()=(u_confluent_h_patch(),u_confluent_P_patch())
@inline magnetic_seed_params()=(magnetic_turning_eta_crit(),magnetic_min_seed_span(),magnetic_max_seed_span(),magnetic_target_accuracy())

# Set the uniform Taylor-center spacing h in s=sqrt(z).
function confluent_U_set_h!(h::Real)
    h>0||error("h_patch must be positive.")
    U_CONFLUENT_TAYLOR_CONFIG.h_patch=Float64(h)
    return U_CONFLUENT_TAYLOR_CONFIG
end

# Set the local Taylor polynomial degree.
function confluent_U_set_P!(P::Integer)
    P>=1||error("P_patch must be at least 1.")
    U_CONFLUENT_TAYLOR_CONFIG.P_patch=Int(P)
    return U_CONFLUENT_TAYLOR_CONFIG
end

# Set the η threshold above which dynamic multi-seeding is enabled.
function magnetic_set_turning_eta_crit!(η::Real)
    η>0||error("turning_eta_crit must be positive.")
    U_CONFLUENT_TAYLOR_CONFIG.turning_eta_crit=Float64(η)
    return U_CONFLUENT_TAYLOR_CONFIG
end

# Set the admissible range of separations between additional mpmath seeds.
# A keyword left as `nothing` keeps its current value.
function magnetic_set_seed_span!(;min_seed_span=nothing,max_seed_span=nothing)
    cfg=U_CONFLUENT_TAYLOR_CONFIG
    smin=isnothing(min_seed_span) ? cfg.min_seed_span : Float64(min_seed_span)
    smax=isnothing(max_seed_span) ? cfg.max_seed_span : Float64(max_seed_span)
    smin>0||error("min_seed_span must be positive.")
    smax>0||error("max_seed_span must be positive.")
    smin<=smax||error("min_seed_span must be <= max_seed_span.")
    cfg.min_seed_span=smin
    cfg.max_seed_span=smax
    return cfg
end

# Set the validation target used by magnetic_validate_taylor_config!.
function magnetic_set_target_accuracy!(tol::Real)
    0<tol<1||error("target_accuracy must satisfy 0 < target_accuracy < 1.")
    U_CONFLUENT_TAYLOR_CONFIG.target_accuracy=Float64(tol)
    return U_CONFLUENT_TAYLOR_CONFIG
end

# Update any subset of the Taylor-table parameters.
# If validate=true, the new configuration is tested immediately with
# magnetic_validate_taylor_config!. If validation fails, every parameter is
# restored to its previous value before the exception is rethrown.
function confluent_U_set_taylor_params!(;h_patch=nothing,P_patch=nothing,turning_eta_crit=nothing,min_seed_span=nothing,max_seed_span=nothing,target_accuracy=nothing,validate::Bool=false,validation_kwargs...)
    old=deepcopy(U_CONFLUENT_TAYLOR_CONFIG)
    try
        !isnothing(h_patch)&&confluent_U_set_h!(h_patch)
        !isnothing(P_patch)&&confluent_U_set_P!(P_patch)
        !isnothing(turning_eta_crit)&&magnetic_set_turning_eta_crit!(turning_eta_crit)
        (!isnothing(min_seed_span)||!isnothing(max_seed_span))&&magnetic_set_seed_span!(;min_seed_span=min_seed_span,max_seed_span=max_seed_span)
        !isnothing(target_accuracy)&&magnetic_set_target_accuracy!(target_accuracy)
        validate&&magnetic_validate_taylor_config!(;validation_kwargs...)
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
# HIGH-PRECISION CONSTANTS AND SEEDS
# =============================================================================

@inline _mp_to_c64(x)=ComplexF64(pycall(_pyfloat[],Float64,x.real),pycall(_pyfloat[],Float64,x.imag))

# Physical logarithmic coefficient in F(z;ν)=Aν(z)log(z)+Bν(z).
# For the physical Green function Aν(0)=1/(4π), independent of ν
@inline magnetic_log_coeff(::ComplexF64)=ComplexF64(inv4π,0.0)

# High-precision constants for the small-z expansion, with a=1/2-ν.
# Returns A0=1/(4π), R0=[ψ(a)+2γ]/(4π).
function magnetic_constants_mpmath(ν::ComplexF64;dps::Int=80)
    lock(PYCALL_MPMATH_LOCK)
    try
        _mpctx[].dps=dps
        a=_mpf[](0.5)-_mpc[](real(ν),imag(ν))
        ψ=_mp_digamma[](a)
        c=_mpf[](1)/(4*_mp_pi[])
        R0=c*(ψ+2*MathConstants.eulergamma)
        return _mp_to_c64(c),_mp_to_c64(R0)
    finally
        unlock(PYCALL_MPMATH_LOCK)
    end
end

# Finite diagonal value Bν(0)=R0.
@inline magnetic_R0(ν::ComplexF64)=magnetic_R0_mpmath(ν;dps=80)
function magnetic_R0_mpmath(ν::ComplexF64;dps::Int=80)
    _,R0=magnetic_constants_mpmath(ν;dps=dps)
    return R0
end

# High-precision seed for G(s)=F(s²) and G'(s)=2sFz(s²).
function seed_G_Gp_mpmath(s0::Float64,ν::ComplexF64;dps::Int=80)
    lock(PYCALL_MPMATH_LOCK)
    try
        _mpctx[].dps=dps
        s=_mpf[](s0)
        z=s*s
        a=_mpf[](0.5)-_mpc[](real(ν),imag(ν))
        U0=_mp_hyperu[](a,1,z)
        U1=_mp_hyperu[](a+1,2,z)
        ez=_mp_exp[](-z/2)
        C=-_mp_gamma[](a)/(4*_mp_pi[])
        F=C*ez*U0
        Fz=C*ez*(-_mpf[](0.5)*U0-a*U1)
        return _mp_to_c64(F),_mp_to_c64(2*s*Fz)
    finally
        unlock(PYCALL_MPMATH_LOCK)
    end
end

# High-precision seed for Aν(s²), where
# Aν(z)=exp(-z/2)1F1(1/2-ν;1;z)/(4π), together with dAν/ds=2sAν,z.
function seed_A_Ap_mpmath(s0::Float64,ν::ComplexF64;dps::Int=80)
    lock(PYCALL_MPMATH_LOCK)
    try
        _mpctx[].dps=dps
        s=_mpf[](s0)
        z=s*s
        a=_mpf[](0.5)-_mpc[](real(ν),imag(ν))
        M0=_mp_hyp1f1[](a,1,z)
        M1=_mp_hyp1f1[](a+1,2,z)
        ez=_mp_exp[](-z/2)
        c=_mpf[](1)/(4*_mp_pi[])
        A=c*ez*M0
        Az=c*ez*(a*M1-_mpf[](0.5)*M0)
        return _mp_to_c64(A),_mp_to_c64(2*s*Az)
    finally
        unlock(PYCALL_MPMATH_LOCK)
    end
end

# Evaluate the derivative of the Taylor polynomial stored in column j.
# If A[n+1,j] is the coefficient of x^n, this returns d/dx Σ A[n+1,j]x^n
# using Horner evaluation.
@inline function horner_deriv_col(A::Matrix{ComplexF64},j::Int,x::Float64)
    P=size(A,1)-1
    P==0&&return ComplexF64(0.0,0.0)
    xx=ComplexF64(x,0.0)
    acc=ComplexF64(P,0.0)*A[P+1,j]
    @inbounds for n in (P-1):-1:1
        acc=muladd(acc,xx,ComplexF64(n,0.0)*A[n+1,j])
    end
    return acc
end

# Evaluate a polynomial with coefficients v[n+1] multiplying x^n
# using Horner's rule.
@inline function horner_eval_vec(v::Vector{ComplexF64},x::Float64)
    xx=ComplexF64(x,0.0)
    acc=ComplexF64(0.0,0.0)
    @inbounds for n in length(v):-1:1
        acc=muladd(acc,xx,v[n])
    end
    return acc
end

# Evaluate the derivative of the polynomial stored in v using Horner's rule.
# For v[n+1] multiplying x^n, this evaluates Σ n*v[n+1]*x^(n-1).
@inline function horner_deriv_vec(v::Vector{ComplexF64},x::Float64)
    P=length(v)-1
    P==0&&return ComplexF64(0.0,0.0)
    xx=ComplexF64(x,0.0)
    acc=ComplexF64(P,0.0)*v[P+1]
    @inbounds for n in (P-1):-1:1
        acc=muladd(acc,xx,ComplexF64(n,0.0)*v[n+1])
    end
    return acc
end

# =============================================================================
# LOCAL TAYLOR RECURRENCE
# =============================================================================
#
# On each patch centered at s0,
#
#   G(s0+h)=Σ g_n h^n,
#
# with G(s)=F(s²). The coefficients are generated directly from
#
#   G''=-G'/s+(s²-4ν)G.
#
# The array invs stores the Taylor coefficients of 1/(s0+h), which are
# needed by the -G'/s term.
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
        n>=1&&(rhs+=2cc*g[n])
        n>=2&&(rhs+=g[n-1])
        g[n+3]=rhs/ComplexF64((n+2)*(n+1),0.0)
    end
    return nothing
end

# Near z=0 the physical Green function is represented as
# F(z)=A(z)log(z)+B(z),
# A(z)=Σ a_m z^m,
# B(z)=Σ b_m z^m.
# The recurrence follows from the radial differential equation after inserting
# this logarithmic ansatz. It is evaluated in BigFloat to avoid loss of
# accuracy near the singular point and converted to ComplexF64 afterwards.
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
            A[m+2]=ap1
            B[m+2]=bp1
        end
        return ComplexF64.(A),ComplexF64.(B)
    end
end

# Complete table for one ν. gcoeffs stores the local Taylor expansions of
# G(s)=F(s²), while acoeffs stores the corresponding expansions of Aν(s²).
# smallA and smallB hold the direct small-z Frobenius coefficients.
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

# Geometry-independent table layout shared by all ν values: z/s ranges,
# Taylor spacing and degree, small-z order, and the common patch centers.
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

# Scratch storage used while generating Taylor coefficients. The TLS arrays
# provide one coefficient buffer per Julia thread when threaded construction
# is requested.
struct MagneticGreenSWorkspace
    gcoef::Vector{ComplexF64}
    invs::Vector{ComplexF64}
end

# Allocate the reusable coefficient buffers for serial or threaded table builds.
@inline function MagneticGreenSWorkspace(;threaded::Bool=false)
    _,P_patch=confluent_U_params()
    return MagneticGreenSWorkspace(Vector{ComplexF64}(undef,P_patch+1),Vector{ComplexF64}(undef,P_patch+1))
end

# TURNING-POINT AND MULTI-SEED STRATEGY
# Characteristic turning scale for nearly real positive ν:
# s_t≈2sqrt(|ν|), z_t≈4|ν|.
# If zmax reaches sufficiently far beyond z_t, extra mpmath seeds are added
# to the right and the intervals between seeds are filled from both sides.

# Use the turning point when it lies inside the table; if it lies beyond the
# tabulated range, seed from the left to avoid a full backward propagation.
@inline function _magnetic_turning_s(pre::MagneticGreenSPrecomp,ν::ComplexF64)
    st=2*sqrt(max(abs(ν),eps(Float64)))
    return st>=pre.smax ? pre.smin : clamp(st,pre.smin,pre.smax)
end

# Convert either the automatic turning point or a user anchor to a patch index.
@inline function _magnetic_anchor_index(pre::MagneticGreenSPrecomp,ν::ComplexF64,anchor_s::Union{Nothing,Float64})
    s0=isnothing(anchor_s) ? _magnetic_turning_s(pre,ν) : anchor_s
    s0=clamp(s0,pre.smin,pre.smax)
    return clamp(Int(round((s0-pre.smin)/pre.h))+1,1,pre.Npatch)
end

# η measures how far zmax extends relative to the formal turning point z_t=4|ν|.
@inline _magnetic_eta(pre::MagneticGreenSPrecomp,ν::ComplexF64)=pre.zmax/(4*max(abs(ν),eps(Float64)))
# Use multiple high-precision seeds only when the turning region is relevant.
@inline _magnetic_use_multi_seed(pre::MagneticGreenSPrecomp,ν::ComplexF64)=_magnetic_eta(pre,ν)>magnetic_turning_eta_crit()

# Choose the extra-seed spacing in s, bounded by the configured limits.
@inline function _magnetic_seed_span(pre::MagneticGreenSPrecomp,ν::ComplexF64,jt::Int)
    span=pre.smax-pre.centers[jt]
    span<=0&&return Inf
    nη=max(1,ceil(Int,_magnetic_eta(pre,ν)))
    return clamp(span/nη,magnetic_min_seed_span(),magnetic_max_seed_span())
end

# Return the patch indices at which independent mpmath seeds are computed.
function _magnetic_seed_indices(pre::MagneticGreenSPrecomp,ν::ComplexF64,anchor_s::Union{Nothing,Float64})
    jt=_magnetic_anchor_index(pre,ν,anchor_s)
    if anchor_s!==nothing||!_magnetic_use_multi_seed(pre,ν)||jt>=pre.Npatch-1
        return [jt]
    end
    Δs=_magnetic_seed_span(pre,ν,jt)
    inds=Int[jt]
    while true
        snext=pre.centers[inds[end]]+Δs
        snext>=pre.smax-pre.h&&break
        j=clamp(Int(round((snext-pre.smin)/pre.h))+1,inds[end]+1,pre.Npatch)
        j>=pre.Npatch&&break
        push!(inds,j)
    end
    inds[end]!=pre.Npatch&&push!(inds,pre.Npatch)
    return inds
end

# Build and store one Taylor patch from an independent high-precision seed.
@inline function _store_seed_patch!(C::Matrix{ComplexF64},pre::MagneticGreenSPrecomp,ws::MagneticGreenSWorkspace,ν::ComplexF64,seedfun,j::Int,mp_dps::Int)
    V,Vp=seedfun(pre.centers[j],ν;dps=mp_dps)
    build_magnetic_patch_coeffs!(ws.gcoef,ws.invs,ν,V,Vp,pre.centers[j])
    @inbounds for n in 1:(pre.P+1)
        C[n,j]=ws.gcoef[n]
    end
    return nothing
end

# Propagate Taylor data from patch j0 to larger s.
function _propagate_right!(C::Matrix{ComplexF64},pre::MagneticGreenSPrecomp,ws::MagneticGreenSWorkspace,ν::ComplexF64,j0::Int,j1::Int)
    j1<=j0&&return nothing
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

# Propagate Taylor data from patch j0 to smaller s.
function _propagate_left!(C::Matrix{ComplexF64},pre::MagneticGreenSPrecomp,ws::MagneticGreenSWorkspace,ν::ComplexF64,j0::Int,j1::Int)
    j1>=j0&&return nothing
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

# Build the common s-grid and table layout used for all ν values.
# The Taylor patches cover s∈[sqrt(zmin),sqrt(zmax)] with spacing h_patch.
function build_MagneticGreenSPrecomp(;zmin::Float64=1e-3,zmax::Float64=900.0,zsmall::Float64=zmin,Msmall::Int=16)
    h_patch,P_patch=confluent_U_params()
    @assert zmin>0&&zmax>zmin&&h_patch>0&&P_patch>=2
    @assert 0<zsmall<=zmin
    smin=sqrt(zmin)
    smax=sqrt(zmax)
    Npatch=Int(ceil((smax-smin)/h_patch))+1
    centers=Vector{Float64}(undef,Npatch)
    @inbounds for i in 1:Npatch
        centers[i]=smin+(i-1)*h_patch
    end
    centers[end]=smax
    return MagneticGreenSPrecomp(zmin,zmax,zsmall,smin,smax,h_patch,P_patch,Msmall,Npatch,centers)
end

# Allocate one Green-function table for a fixed ν.
# gcoeffs stores F(s²), acoeffs stores Aν(s²), and smallA/smallB store
# the direct small-z logarithmic expansion.
@inline function alloc_MagneticGreenSTaylorTable(pre::MagneticGreenSPrecomp;ν::ComplexF64=0.0+0.0im)
    gcoeffs=Matrix{ComplexF64}(undef,pre.P+1,pre.Npatch)
    acoeffs=Matrix{ComplexF64}(undef,pre.P+1,pre.Npatch)
    R0=magnetic_R0(ν)
    a0=magnetic_log_coeff(ν)
    A,B=build_small_z_coeffs(ν,R0;a0=a0,M=pre.Msmall)
    return MagneticGreenSTaylorTable(ν,pre.zmin,pre.zmax,pre.zsmall,pre.smin,pre.smax,pre.h,pre.P,pre.centers,gcoeffs,acoeffs,a0,R0,A,B)
end

# Allocate several tables sharing the same precomputed patch layout.
@inline function alloc_MagneticGreenSTaylorTables(pre::MagneticGreenSPrecomp,Nν::Int;ν::ComplexF64=0.0+0.0im)
    tabs=Vector{MagneticGreenSTaylorTable}(undef,Nν)
    @inbounds for i in 1:Nν
        tabs[i]=alloc_MagneticGreenSTaylorTable(pre;ν=ν)
    end
    return tabs
end

# Recompute the ν-dependent small-z constants and Frobenius coefficients.
function _update_small_z!(tab::MagneticGreenSTaylorTable,pre::MagneticGreenSPrecomp,ν::ComplexF64;mp_dps::Int=100)
    tab.ν=ν
    tab.a_log,tab.R0=magnetic_constants_mpmath(ν;dps=mp_dps)
    A,B=build_small_z_coeffs(ν,tab.R0;a0=tab.a_log,M=pre.Msmall)
    A[1]=tab.a_log
    B[1]=tab.R0
    resize!(tab.smallA,length(A))
    resize!(tab.smallB,length(B))
    copyto!(tab.smallA,A)
    copyto!(tab.smallB,B)
    return nothing
end

# Build one complete coefficient table from high-precision seeds.
# With one seed the table is propagated left and right from that point.
# With multiple seeds, every seed interval is filled from both ends.
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

# Build only the Taylor table for the logarithmic coefficient Aν(z).
function build_A_coeff_table!(tab::MagneticGreenSTaylorTable,pre::MagneticGreenSPrecomp,ws::MagneticGreenSWorkspace,ν::ComplexF64;mp_dps::Int=80,anchor_s::Union{Nothing,Float64}=nothing)
    _build_coeff_table!(tab.acoeffs,pre,ws,ν,seed_A_Ap_mpmath;mp_dps=mp_dps,anchor_s=anchor_s)
    return nothing
end

# Rebuild an existing table in place for ν, including the small-z expansion,
# the physical Green-function coefficients, and the Kress log coefficients.
function build_MagneticGreenSTaylorTable!(tab::MagneticGreenSTaylorTable,pre::MagneticGreenSPrecomp,ws::MagneticGreenSWorkspace,ν::ComplexF64;mp_dps::Int=80,anchor_s::Union{Nothing,Float64}=nothing)
    @assert pre.centers===tab.centers
    @assert pre.P==tab.P&&pre.Npatch==size(tab.gcoeffs,2)&&pre.Npatch==size(tab.acoeffs,2)
    _update_small_z!(tab,pre,ν;mp_dps=mp_dps)
    _build_coeff_table!(tab.gcoeffs,pre,ws,ν,seed_G_Gp_mpmath;mp_dps=mp_dps,anchor_s=anchor_s)
    _build_coeff_table!(tab.acoeffs,pre,ws,ν,seed_A_Ap_mpmath;mp_dps=mp_dps,anchor_s=anchor_s)
    return nothing
end

# Convenience constructor that allocates the layout, workspace, and table,
# then builds the complete physical Green-function table for one ν.
function build_MagneticGreenSTaylorTable(ν::ComplexF64;zmin::Float64=1e-3,zmax::Float64=900.0,zsmall::Float64=zmin,Msmall::Int=16,mp_dps::Int=80,anchor_s::Union{Nothing,Float64}=nothing)
    pre=build_MagneticGreenSPrecomp(;zmin=zmin,zmax=zmax,zsmall=zsmall,Msmall=Msmall)
    ws=MagneticGreenSWorkspace(;threaded=false)
    tab=alloc_MagneticGreenSTaylorTable(pre;ν=ν)
    build_MagneticGreenSTaylorTable!(tab,pre,ws,ν;mp_dps=mp_dps,anchor_s=anchor_s)
    return tab
end

# Return the Taylor patch containing s. Values outside the tabulated range are
# clamped to the first or last patch; the roundoff check stabilizes points that
# lie numerically on a patch center.
@inline function _mag_patch_index(tab::MagneticGreenSTaylorTable,s::Float64)
    if s<=tab.smin
        return 1
    elseif s>=tab.smax
        return length(tab.centers)
    else
        t=(s-tab.smin)/tab.h
        idx=Int(floor(t))+1
        abs(t-round(t))<64*eps(t)&&(idx=Int(round(t))+1)
        return clamp(idx,1,length(tab.centers))
    end
end

# Direct small-z evaluation of F=A log(z)+B from the Frobenius coefficients.
@inline _small_F(tab::MagneticGreenSTaylorTable,z::Float64)=horner_eval_vec(tab.smallA,z)*log(z)+horner_eval_vec(tab.smallB,z)

# Small-z constant-log remainder R=F-a_log log(z), finite at z=0.
@inline function _small_R(tab::MagneticGreenSTaylorTable,z::Float64)
    A=horner_eval_vec(tab.smallA,z)
    B=horner_eval_vec(tab.smallB,z)
    return (A-tab.a_log)*log(z)+B
end

# z-derivative of the small-z representation F=A log(z)+B.
@inline function _small_Fz(tab::MagneticGreenSTaylorTable,z::Float64)
    L=log(z)
    A=horner_eval_vec(tab.smallA,z)
    Ap=horner_deriv_vec(tab.smallA,z)
    Bp=horner_deriv_vec(tab.smallB,z)
    return Ap*L+A/z+Bp
end

# Evaluate G(s)=F(s²) from the local Taylor patch.
@inline function _eval_G(tab::MagneticGreenSTaylorTable,s::Float64)
    idx=_mag_patch_index(tab,s)
    return horner_eval_col(tab.gcoeffs,idx,s-tab.centers[idx])
end

# Evaluate dG/ds from the derivative of the local Taylor polynomial.
@inline function _eval_dGds(tab::MagneticGreenSTaylorTable,s::Float64)
    idx=_mag_patch_index(tab,s)
    return horner_deriv_col(tab.gcoeffs,idx,s-tab.centers[idx])
end

# Evaluate the physical radial Green factor. The Frobenius series is used below
# zsmall and the propagated s-table elsewhere.
@inline function _eval_F(tab::MagneticGreenSTaylorTable,z::Float64)
    z==0.0&&return ComplexF64(Inf,0.0)
    z<tab.zsmall&&return _small_F(tab,z)
    return _eval_G(tab,sqrt(z))
end

# Exact Kress logarithmic coefficient Aν(z).
@inline function _eval_Alog(tab::MagneticGreenSTaylorTable,z::Float64)
    z==0.0&&return tab.a_log
    z<tab.zsmall&&return horner_eval_vec(tab.smallA,z)
    s=sqrt(z)
    idx=_mag_patch_index(tab,s)
    return horner_eval_col(tab.acoeffs,idx,s-tab.centers[idx])
end

# Smooth Kress remainder Bν(z)=F(z)-Aν(z)log(z).
@inline function _eval_Blog(tab::MagneticGreenSTaylorTable,z::Float64)
    z==0.0&&return tab.R0
    z<tab.zsmall&&return horner_eval_vec(tab.smallB,z)
    return _eval_F(tab,z)-_eval_Alog(tab,z)*log(z)
end

# z-derivative of Aν(z); convert d/ds to d/dz using dz/ds=2s.
@inline function _eval_Alog_z(tab::MagneticGreenSTaylorTable,z::Float64)
    z==0.0&&error("Alog_z is finite at z=0; use tab.smallA[2].")
    z<tab.zsmall&&return horner_deriv_vec(tab.smallA,z)
    s=sqrt(z)
    idx=_mag_patch_index(tab,s)
    return horner_deriv_col(tab.acoeffs,idx,s-tab.centers[idx])/(2s)
end

# Constant-log remainder Rν(z)=F(z)-a_log log(z), with Rν(0)=R0.
@inline function _eval_R(tab::MagneticGreenSTaylorTable,z::Float64)
    z==0.0&&return tab.R0
    z<tab.zsmall&&return _small_R(tab,z)
    return _eval_F(tab,z)-tab.a_log*log(z)
end

# z-derivative of F; for the Taylor region use Fz=(dG/ds)/(2s).
@inline function _eval_Fz(tab::MagneticGreenSTaylorTable,z::Float64)
    z==0.0&&error("Fz is singular at z=0.")
    z<tab.zsmall&&return _small_Fz(tab,z)
    s=sqrt(z)
    return _eval_dGds(tab,s)/(2s)
end

# In-place vector evaluations used in matrix assembly.
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

# Direct high-precision physical Green factor used only for validation.
function F_ref_mpmath(z::Float64,ν::ComplexF64;dps::Int=100)
    lock(PYCALL_MPMATH_LOCK)
    try
        _mpctx[].dps=dps
        zp=_mpf[](z)
        a=_mpf[](0.5)-_mpc[](real(ν),imag(ν))
        return _mp_to_c64(-_mp_gamma[](a)/(4*_mp_pi[])*_mp_exp[](-zp/2)*_mp_hyperu[](a,1,zp))
    finally
        unlock(PYCALL_MPMATH_LOCK)
    end
end

# Direct high-precision z-derivative of the physical Green factor.
function Fz_ref_mpmath(z::Float64,ν::ComplexF64;dps::Int=100)
    lock(PYCALL_MPMATH_LOCK)
    try
        _mpctx[].dps=dps
        zp=_mpf[](z)
        a=_mpf[](0.5)-_mpc[](real(ν),imag(ν))
        U0=_mp_hyperu[](a,1,zp)
        U1=_mp_hyperu[](a+1,2,zp)
        C=-_mp_gamma[](a)/(4*_mp_pi[])
        return _mp_to_c64(C*_mp_exp[](-zp/2)*(-_mpf[](0.5)*U0-a*U1))
    finally
        unlock(PYCALL_MPMATH_LOCK)
    end
end

# Direct high-precision Kress log coefficient Aν(z).
function Alog_ref_mpmath(z::Float64,ν::ComplexF64;dps::Int=100)
    lock(PYCALL_MPMATH_LOCK)
    try
        _mpctx[].dps=dps
        zp=_mpf[](z)
        a=_mpf[](0.5)-_mpc[](real(ν),imag(ν))
        return _mp_to_c64(_mp_exp[](-zp/2)*_mp_hyp1f1[](a,1,zp)/(4*_mp_pi[]))
    finally
        unlock(PYCALL_MPMATH_LOCK)
    end
end

# Direct high-precision z-derivative of Aν(z).
function Alog_z_ref_mpmath(z::Float64,ν::ComplexF64;dps::Int=100)
    lock(PYCALL_MPMATH_LOCK)
    try
        _mpctx[].dps=dps
        zp=_mpf[](z)
        a=_mpf[](0.5)-_mpc[](real(ν),imag(ν))
        M0=_mp_hyp1f1[](a,1,zp)
        M1=_mp_hyp1f1[](a+1,2,zp)
        return _mp_to_c64(_mp_exp[](-zp/2)*(a*M1-_mpf[](0.5)*M0)/(4*_mp_pi[]))
    finally
        unlock(PYCALL_MPMATH_LOCK)
    end
end

# Reference smooth remainder Bν(z)=F(z)-Aν(z)log(z).
@inline Blog_ref_mpmath(z::Float64,ν::ComplexF64;dps::Int=100)=F_ref_mpmath(z,ν;dps=dps)-Alog_ref_mpmath(z,ν;dps=dps)*log(z)

# VALIDATION
function magnetic_validate_taylor_config!(;nus=ComplexF64[20+0im,40+0im,80+0im,600+0im,2000+0im,20.5-0.01im,80.5-0.01im,100-1im,500-2im,1000-5im,2000-10im,5000-20im],zmax::Float64=500.0,zmin::Float64=1e-3,zsmall::Float64=1e-3,Msmall::Int=30,mp_dps::Int=160,ztests=nothing,rtol::Float64=magnetic_target_accuracy(),atol::Float64=1e-12,verbose::Bool=false,test_derivatives::Bool=true,test_split::Bool=true,test_timing::Bool=true,timing_batch::Int=10000)
    zs=isnothing(ztests) ? Float64[1e-15,1e-12,1e-9,1e-6,1e-4,zsmall,1.5zsmall,1e-2,3e-2,0.12,0.35,0.75,1.0,0.25zmax,0.5zmax,0.75zmax,zmax] : Float64.(ztests)
    ws=MagneticGreenSWorkspace(;threaded=false)
    pre=build_MagneticGreenSPrecomp(;zmin=zmin,zmax=zmax,zsmall=zsmall,Msmall=Msmall)
    rng=MersenneTwister(12345)

    for ν in nus
        tab=alloc_MagneticGreenSTaylorTable(pre;ν=ν)
        build_MagneticGreenSTaylorTable!(tab,pre,ws,ν;mp_dps=mp_dps)

        abs(tab.a_log-ComplexF64(inv4π,0.0))<=100eps(Float64)||error("Physical log coefficient failed at ν=$ν")

        for z in zs
            0<z<=zmax||continue

            Fref=F_ref_mpmath(z,ν;dps=mp_dps)
            Fval=_eval_F(tab,z)
            Aref=Alog_ref_mpmath(z,ν;dps=mp_dps)
            Aval=_eval_Alog(tab,z)

            errF=abs(Fval-Fref)
            relF=errF/max(abs(Fref),eps(Float64))
            errA=abs(Aval-Aref)
            relA=errA/max(abs(Aref),eps(Float64))

            verbose&&println("ν=",ν," z=",z," relF=",relF," relA=",relA)

            (relF<=rtol||errF<=atol)||error("F validation failed: ν=$ν z=$z rel=$relF abs=$errF")
            (relA<=rtol||errA<=atol)||error("Alog validation failed: ν=$ν z=$z rel=$relA abs=$errA")

            if test_derivatives
                Fzref=Fz_ref_mpmath(z,ν;dps=mp_dps)
                Fzval=_eval_Fz(tab,z)
                Azref=Alog_z_ref_mpmath(z,ν;dps=mp_dps)
                Azval=_eval_Alog_z(tab,z)

                errFz=abs(Fzval-Fzref)
                relFz=errFz/max(abs(Fzref),eps(Float64))
                errAz=abs(Azval-Azref)
                relAz=errAz/max(abs(Azref),eps(Float64))

                (relFz<=max(10rtol,1e-12)||errFz<=10atol)||error("Fz validation failed: ν=$ν z=$z rel=$relFz abs=$errFz")
                (relAz<=max(10rtol,1e-12)||errAz<=10atol)||error("Alog_z validation failed: ν=$ν z=$z rel=$relAz abs=$errAz")
            end

            if test_split
                Bval=_eval_Blog(tab,z)
                scale=max(abs(Fval),abs(Aval*log(z)),abs(Bval),1.0)
                err=abs(Fval-(Aval*log(z)+Bval))/scale
                err<=max(10rtol,100eps(Float64))||error("Split validation failed: ν=$ν z=$z err=$err")
            end
        end

        if test_timing
            for zbench in (0.1zsmall,min(1.0,zmax))
                tF=@belapsed _eval_F($tab,$zbench)
                tFz=@belapsed _eval_Fz($tab,$zbench)
                tA=@belapsed _eval_Alog($tab,$zbench)
                tAz=@belapsed _eval_Alog_z($tab,$zbench)
                tB=@belapsed _eval_Blog($tab,$zbench)

                println("TIMING ν=",ν," z=",zbench,
                    " F=",round(1e9*tF,digits=2)," ns",
                    " Fz=",round(1e9*tFz,digits=2)," ns",
                    " A=",round(1e9*tA,digits=2)," ns",
                    " Az=",round(1e9*tAz,digits=2)," ns",
                    " B=",round(1e9*tB,digits=2)," ns")
            end

            zbench=zsmall .+ (zmax-zsmall).*rand(rng,timing_batch)
            out=Vector{ComplexF64}(undef,timing_batch)

            tF=@belapsed _eval_F!($out,$tab,$zbench)
            tFz=@belapsed _eval_Fz!($out,$tab,$zbench)
            tA=@belapsed _eval_Alog!($out,$tab,$zbench)
            tB=@belapsed _eval_Blog!($out,$tab,$zbench)

            println("BATCH ν=",ν," N=",timing_batch,
                " F=",round(1e9*tF/timing_batch,digits=2)," ns/eval",
                " Fz=",round(1e9*tFz/timing_batch,digits=2)," ns/eval",
                " A=",round(1e9*tA/timing_batch,digits=2)," ns/eval",
                " B=",round(1e9*tB/timing_batch,digits=2)," ns/eval")
        end
    end
    return true
end

if abspath(PROGRAM_FILE)==@__FILE__
    magnetic_validate_taylor_config!(zmax=500.0,zmin=1e-3,zsmall=1e-3,Msmall=30,mp_dps=160,rtol=magnetic_target_accuracy(),atol=1e-14,verbose=true,test_derivatives=true,test_split=true)
end
