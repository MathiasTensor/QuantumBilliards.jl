# =============================================================================
# Fast tables for the Legendre function Q_ν(cosh d) and its radial derivative
# y(d)=d/dd Q_ν(cosh d), with ν=-1/2+i k.
#
# Motivation (hyperbolic BIM / Beyn):
#   - We need extremely fast evaluations of Q and dQ/dd at many distances d.
#   - Instead of GL quadrature + Chebyshev panels, we build a uniform-step
#     Taylor-patch table by propagating the ODE system in d:
#         u(d)=Q_ν(cosh d)
#         y(d)=u'(d)
#         u'=y
#         y'=ν(ν+1)u-coth(d)y
#
#   - Only ONE high-precision seed (u(dmin),y(dmin)) is needed, obtained from
#     mpmath.legenq via PyCall.
#   - Everything else is pure Julia and extremely fast.
#
# Notes:
#   - This file provides radial derivatives dQ/dd. The normal derivative
#     needed for the DLP is obtained by chain rule:
#         ∂_{n'}Q_ν(cosh d)=(dQ/dd)(d)*∂_{n'}d
#     Geometry-specific computation of ∂_{n'}d lives elsewhere.
#
# MO / 15-12-25
# =============================================================================

using PyCall
using LinearAlgebra
const _mp=pyimport("mpmath")
const _pyfloat=pybuiltin("float")

# -----------------------------
# Small-d fallbacks
# -----------------------------

const Z_threshold=1.0+1e-14
const d_threshold=1e-3

@inline function _small_z_Q(k::C,d::T)::C where{C<:Complex,T<:Real}
    H=MathConstants.eulergamma+digamma(0.5+im*k)
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

@inline function _small_z_Q(k::T,d::T) where{T<:Real}
    return _small_z_Q(complex(k,0.0),d)
end

@inline function _small_z_dQ(k::C,d::T)::C where{C<:Complex,T<:Real};nu=-T(0.5)+im*k;Hnu=Base.MathConstants.eulergamma+SpecialFunctions.digamma(nu+one(nu))
    return C(-(one(T)/d)+(d*(2+3*nu*(1+nu)))/12+(d^3*(-56+15*nu*(1+nu)*(-2+15*nu*(1+nu))))/2880+(d^5*(248+21*nu*(1+nu)*(8+5*nu*(1+nu)*(-7+5*nu*(1+nu)))))/120960+(d^7*(-48768+5*nu*(1+nu)*(-7568+21*nu*(1+nu)*(1268+5*nu*(1+nu)*(-156+47*nu*(1+nu))))))/2.322432e8+(d^9*(1308160+11*nu*(1+nu)*(98432+3*nu*(1+nu)*(-104864+7*nu*(1+nu)*(8284+nu*(1+nu)*(-2320+393*nu*(1+nu)))))))/6.13122048e10+(d^11*(-724212224+273*nu*(1+nu)*(-2267520+11*nu*(1+nu)*(628720+nu*(1+nu)*(-327944+21*nu*(1+nu)*(3968+nu*(1+nu)*(-650+71*nu*(1+nu))))))))/3.34764638208e14+(d^13*(3522785280+nu*(1+nu)*(3064070144+429*nu*(1+nu)*(-21245568+nu*(1+nu)*(10734208+nu*(1+nu)*(-2570552+3*nu*(1+nu)*(127876+nu*(1+nu)*(-13846+1059*nu*(1+nu)))))))))/1.6068702633984e16+(d^15*(-81555718766592+17*nu*(1+nu)*(-4215787018240+3*nu*(1+nu)*(4119262721280+143*nu*(1+nu)*(-14288101248+nu*(1+nu)*(3308766736+3*nu*(1+nu)*(-154931200+nu*(1+nu)*(15324344+45*nu*(1+nu)*(-26264+1487*nu*(1+nu))))))))))/3.6713771778126643e21+(d^17*(7536235717591040+57*nu*(1+nu)*(116947251986432+17*nu*(1+nu)*(-19979202338816+nu*(1+nu)*(9796767888640+13*nu*(1+nu)*(-171086931456+11*nu*(1+nu)*(2109795216+nu*(1+nu)*(-196236224+nu*(1+nu)*(13852776+5*nu*(1+nu)*(-160176+6989*nu*(1+nu)))))))))))/3.34829598616515e24+(d^19*(-3023786723765649408+55*nu*(1+nu)*(-48844705828831232+57*nu*(1+nu)*(2473269691121664+17*nu*(1+nu)*(-70796702697984+nu*(1+nu)*(15871802558208+13*nu*(1+nu)*(-162189398880+11*nu*(1+nu)*(1322981232+5*nu*(1+nu)*(-17552464+nu*(1+nu)*(930264+nu*(1+nu)*(-41850+1451*nu*(1+nu))))))))))))/1.3259252105213994e28+(d^21*(5919143148921500467200+23*nu*(1+nu)*(229361823479855316992+15*nu*(1+nu)*(-43938582105369018368+19*nu*(1+nu)*(1119337009242789888+17*nu*(1+nu)*(-14636176930816512+nu*(1+nu)*(1918454135937408+13*nu*(1+nu)*(-12957794542560+nu*(1+nu)*(828616686128+nu*(1+nu)*(-41247566448+7*nu*(1+nu)*(243316128+nu*(1+nu)*(-8764690+247353*nu*(1+nu)))))))))))))/2.5616875067273434e32+(d^23*(-686096493620974804008960+13*nu*(1+nu)*(-47144867132796858793984+23*nu*(1+nu)*(5871102997047666982912+5*nu*(1+nu)*(-566138921659181092864+57*nu*(1+nu)*(2194468161804355584+17*nu*(1+nu)*(-16765943320964352+nu*(1+nu)*(1451742074957120+13*nu*(1+nu)*(-6984481999040+nu*(1+nu)*(334973138896+nu*(1+nu)*(-12986455152+7*nu*(1+nu)*(61403188+nu*(1+nu)*(-1811612+42433*nu*(1+nu))))))))))))))/2.930570507696081e35+(d^25*(33367733728285762089123840+nu*(1+nu)*(29859320862789776542007296+5*nu*(1+nu)*(-17063269414016842458202112+23*nu*(1+nu)*(356639815691069055434752+nu*(1+nu)*(-78443654495831951527936+57*nu*(1+nu)*(177569554987496819712+17*nu*(1+nu)*(-895795952130801408+nu*(1+nu)*(55227923257381696+nu*(1+nu)*(-2589559892863424+nu*(1+nu)*(96684098805008+nu*(1+nu)*(-3003546729040+7*nu*(1+nu)*(11641920004+5*nu*(1+nu)*(-57309356+1132133*nu*(1+nu)))))))))))))))/1.4066738436941188e38+(d^27*(-91770018091370053888741736448+29*nu*(1+nu)*(-2835615745800674524836921344+3*nu*(1+nu)*(2695606115502739701208973312+5*nu*(1+nu)*(-258579031590302542450655232+23*nu*(1+nu)*(2464338533704759823015936+nu*(1+nu)*(-316405030616016052463616+19*nu*(1+nu)*(1418259159879823898112+17*nu*(1+nu)*(-5092635130610819200+nu*(1+nu)*(235309612303988928+nu*(1+nu)*(-8586706803750816+nu*(1+nu)*(256808887011248+3*nu*(1+nu)*(-2179494838664+nu*(1+nu)*(49370502636+5*nu*(1+nu)*(-205868026+3476589*nu*(1+nu))))))))))))))))/3.818275481323316e42-(d*nu*(1+nu)*(182684914765469984565271461888000000+7611871448561249356886310912000000*d^2*(-2+3*nu*(1+nu))+190296786214031233922157772800000*d^4*(8+5*(-1+nu)*nu*(1+nu)*(2+nu))+566359482779854862863564800000*d^6*(-272+7*nu*(1+nu)*(44+5*nu*(1+nu)*(-4+nu+nu^2)))+3933051963748992103219200000*d^8*(3968+3*nu*(1+nu)*(-1424+7*nu*(1+nu)*(84+nu*(1+nu)*(-20+3*nu*(1+nu)))))+8938754463065891143680000*d^10*(-176896+11*nu*(1+nu)*(16864+nu*(1+nu)*(-6584+21*nu*(1+nu)*(68+nu*(1+nu)*(-10+nu+nu^2)))))+28649854048288112640000*d^12*(5592064+13*nu*(1+nu)*(-444544+11*nu*(1+nu)*(15296+nu*(1+nu)*(-3128+3*nu*(1+nu)*(140+nu*(1+nu)*(-14+nu+nu^2))))))+8526742276276224000*d^14*(-1903757312+3*nu*(1+nu)*(650078976+13*nu*(1+nu)*(-18589056+11*nu*(1+nu)*(334192+nu*(1+nu)*(-42240+nu*(1+nu)*(3864+5*nu*(1+nu)*(-56+3*nu*(1+nu))))))))+15674158596096000*d^16*(104932671488+17*nu*(1+nu)*(-6287587328+nu*(1+nu)*(2311237888+13*nu*(1+nu)*(-34468224+11*nu*(1+nu)*(382416+nu*(1+nu)*(-32896+nu*(1+nu)*(2184+5*nu*(1+nu)*(-24+nu+nu^2))))))))+11457718272000*d^18*(-14544442556416+19*nu*(1+nu)*(776768475136+17*nu*(1+nu)*(-16670893568+nu*(1+nu)*(3191642624+13*nu*(1+nu)*(-29341344+11*nu*(1+nu)*(221328+nu*(1+nu)*(-13808+nu*(1+nu)*(696+(-5+nu)*nu*(1+nu)*(6+nu)))))))))+13640140800*d^20*(1237874513281024+nu*(1+nu)*(-1252648497168384+19*nu*(1+nu)*(23928414824448+17*nu*(1+nu)*(-267191996928+nu*(1+nu)*(31506758784+13*nu*(1+nu)*(-196781088+nu*(1+nu)*(11833360+nu*(1+nu)*(-560208+7*nu*(1+nu)*(3168+nu*(1+nu)*(-110+3*nu*(1+nu)))))))))))+3369600*d^22*(-507711943253426176+23*nu*(1+nu)*(22292423254048768+nu*(1+nu)*(-8059883338280960+19*nu*(1+nu)*(80038242321408+17*nu*(1+nu)*(-550103120640+nu*(1+nu)*(44044491328+13*nu*(1+nu)*(-199258304+nu*(1+nu)*(9087760+nu*(1+nu)*(-337744+7*nu*(1+nu)*(1540+nu*(1+nu)*(-44+nu+nu^2)))))))))))+2808*d^24*(61730370047551995904+5*nu*(1+nu)*(-12448661250518024192+23*nu*(1+nu)*(195122177890779136+nu*(1+nu)*(-36649899662381056+19*nu*(1+nu)*(223895553555456+17*nu*(1+nu)*(-1044390696192+nu*(1+nu)*(60545295936+nu*(1+nu)*(-2699564608+nu*(1+nu)*(96615376+nu*(1+nu)*(-2894320+7*nu*(1+nu)*(10868+5*nu*(1+nu)*(-52+nu+nu^2))))))))))))+d^26*(-17562900400985989971968+3*nu*(1+nu)*(5895807142545951555584+5*nu*(1+nu)*(-424139144014000029696+23*nu*(1+nu)*(3451830441237979136+nu*(1+nu)*(-398682053674530816+19*nu*(1+nu)*(1652375735883264+17*nu*(1+nu)*(-5578990587776+nu*(1+nu)*(245125459008+nu*(1+nu)*(-8573995872+nu*(1+nu)*(247266448+3*nu*(1+nu)*(-2032888+nu*(1+nu)*(44772+5*nu*(1+nu)*(-182+3*nu*(1+nu)))))))))))))))*(Hnu+log(d/2)))/3.6536982953093996e35)
end

@inline function _small_z_dQ(k::T,d::T) where{T<:Real}
    return _small_z_dQ(complex(k,0.0),d)
end

# ---------------------------------
# Python / mpmath seeding (PyCall)
# ---------------------------------

# =============================================================================
# ν(k)
#
# Convenience map from wavenumber k to Legendre index
#   ν=-1/2+i k.
#
# Input
#   k::Complex/Real
#
# Output
#   ν::Complex
# =============================================================================
@inline ν(k)=-0.5+im*k

# =============================================================================
# seed_u_y_mpmath
#
# Compute the seed values at a single point d0 using mpmath.legenq:
#   u0=Q_ν(cosh(d0)),
#   y0=d/dd Q_ν(cosh(d0)).
#
# Derivative evaluation
#   We avoid differentiating a hypergeometric representation directly.
#   Using the stable 3-term identity (in z-space) and chain rule yields
#       y(d)=(ν+1)*(Q_{ν+1}(z)-z Q_ν(z))/sinh(d),   z=cosh(d).
#
# Inputs
#   nu::ComplexF64   - ν index (typically ν=-1/2+i k)
#   d0::Float64      - seed location (d0>0)
#   dps::Int         - mpmath working precision (decimal digits)
#   leg_type::Int    - mpmath LegendreQ definition selector
#
# Outputs
#   (u0,y0)::Tuple{ComplexF64,ComplexF64}
# =============================================================================
function seed_u_y_mpmath(nu::ComplexF64,d0::Float64;dps::Int=60,leg_type::Int=3)
    _mp.mp.dps=dps
    nu_py=_mp.mpc(real(nu),imag(nu))
    d_py=_mp.mpf(d0)
    z=_mp.cosh(d_py)
    sh=_mp.sinh(d_py)
    Qnu=_mp.legenq(nu_py,0,z,type=leg_type)
    Qnu1=_mp.legenq(nu_py+1,0,z,type=leg_type)
    y0=(nu_py+1)*(Qnu1-z*Qnu)/sh
    u_re=pycall(_pyfloat,Float64,Qnu.real)
    u_im=pycall(_pyfloat,Float64,Qnu.imag)
    y_re=pycall(_pyfloat,Float64,y0.real)
    y_im=pycall(_pyfloat,Float64,y0.imag)
    return ComplexF64(u_re,u_im),ComplexF64(y_re,y_im)
end

# =============================================================================
# series_sinh_cosh!
#
# Build Taylor coefficients about δ=0 for
#   sinh(δ)=Σ_{n=0}^P sδ[n+1] δ^n,
#   cosh(δ)=Σ_{n=0}^P cδ[n+1] δ^n,
# with exact Float64 coefficients 1/n! on the appropriate parity.
#
# Inputs
#   sδ::Vector{Float64} length P+1  (output buffer)
#   cδ::Vector{Float64} length P+1  (output buffer)
#
# Output
#   nothing (fills sδ,cδ in place)
# =============================================================================
@inline function series_sinh_cosh!(sδ::Vector{Float64},cδ::Vector{Float64})
    P=length(sδ)-1
    fill!(sδ,0.0)
    fill!(cδ,0.0)
    fact=1.0
    @inbounds for n in 0:P
        if n>0
            fact*=n
        end
        if iseven(n)
            cδ[n+1]=1.0/fact
        else
            sδ[n+1]=1.0/fact
        end
    end
    return nothing
end

# =============================================================================
# series_div!
#
# Power-series division q=num/den about 0 up to order P, assuming den[1]≠0.
# Coefficients are stored as q[n+1]=q_n.
#
# Inputs
#   q::Vector{Float64}   length P+1 (output)
#   num::Vector{Float64} length P+1
#   den::Vector{Float64} length P+1 with den[1]≠0
#
# Output
#   nothing (fills q in place)
# =============================================================================
@inline function series_div!(q::Vector{Float64},num::Vector{Float64},den::Vector{Float64})
    P=length(q)-1
    invd0=1.0/den[1]
    @inbounds q[1]=num[1]*invd0
    @inbounds for n in 1:P
        s=num[n+1]
        @inbounds for k in 1:n
            s-=den[k+1]*q[n-k+1]
        end
        q[n+1]=s*invd0
    end
    return nothing
end

# =============================================================================
# coth_series_from_sinhcosh!
#
# Construct Taylor coefficients about δ=0 for
#   coth(d0+δ)=cosh(d0+δ)/sinh(d0+δ),
# given sinh(d0)=sinh0 and cosh(d0)=cosh0.
#
# Using addition formulas:
#   sinh(d0+δ)=sinh0*cosh(δ)+cosh0*sinh(δ),
#   cosh(d0+δ)=cosh0*cosh(δ)+sinh0*sinh(δ),
# followed by series division.
#
# Inputs
#   coth::Vector{Float64} length P+1 (output)
#   sinh0,cosh0::Float64  values at d0
#   sδ,cδ::Vector{Float64} base series for sinh/cosh(δ)
#   sinh_series,cosh_series::Vector{Float64} scratch buffers
#
# Output
#   nothing (fills coth in place)
# =============================================================================
@inline function coth_series_from_sinhcosh!(coth::Vector{Float64},sinh0::Float64,cosh0::Float64,sδ::Vector{Float64},cδ::Vector{Float64},sinh_series::Vector{Float64},cosh_series::Vector{Float64})
    P=length(coth)-1
    @inbounds for i in 1:(P+1)
        sinh_series[i]=sinh0*cδ[i]+cosh0*sδ[i]
        cosh_series[i]=cosh0*cδ[i]+sinh0*sδ[i]
    end
    series_div!(coth,cosh_series,sinh_series)
    return nothing
end

# =============================================================================
# horner_eval
#
# Evaluate a complex Taylor polynomial
#   p(x)=Σ_{n=0}^P a_n x^n
# stored as coeffs[n+1]=a_n, using Horner form with fused muladd.
#
# Inputs
#   coeffs::AbstractVector{ComplexF64}
#   x::Float64
#
# Output
#   ComplexF64
# =============================================================================
@inline function horner_eval(coeffs::AbstractVector{ComplexF64},x::Float64)
    xx=ComplexF64(x,0.0)
    acc=ComplexF64(0.0,0.0)
    @inbounds for i in length(coeffs):-1:1
        acc=muladd(acc,xx,coeffs[i])
    end
    return acc
end

# =============================================================================
# build_patch_coeffs!
#
# Generate Taylor coefficients (u_n,y_n) about a patch center d0 for the ODE
# system
#   u'=y,
#   y'=ν(ν+1)u-coth(d)y,
# assuming coth(d0+δ)=Σ c_n δ^n is provided.
#
# Recurrences for n=0..P-1 (coeffs store u_n=u[n+1],y_n=y[n+1]):
#   u_{n+1}=y_n/(n+1),
#   y_{n+1}=(ν(ν+1)u_n-Σ_{j=0}^n c_{n-j}y_j)/(n+1).
#
# Inputs
#   u,y::Vector{ComplexF64} length P+1 (output)
#   nu::ComplexF64          ν index
#   u0,y0::ComplexF64       values at δ=0
#   coth::Vector{Float64}   c_n coefficients of coth(d0+δ)
#
# Output
#   nothing (fills u,y in place)
# =============================================================================
@inline function build_patch_coeffs!(u::Vector{ComplexF64},y::Vector{ComplexF64},nu::ComplexF64,u0::ComplexF64,y0::ComplexF64,coth::Vector{Float64})
    P=length(u)-1
    u[1]=u0
    y[1]=y0
    lam=nu*(nu+1)
    @inbounds for n in 0:(P-1)
        u[n+2]=y[n+1]/ComplexF64(n+1,0.0)
        conv=ComplexF64(0.0,0.0)
        @inbounds for j in 0:n
            conv+=ComplexF64(coth[n-j+1],0.0)*y[j+1]
        end
        y[n+2]=(lam*u[n+1]-conv)/ComplexF64(n+1,0.0)
    end
    return nothing
end

# =============================================================================
# QTaylorTable
#
# Purpose
#   Store fast, uniform-step Taylor-patch tables for the Legendre function
#       u(d)=Q_ν(cosh(d))
#   and its radial derivative
#       y(d)=d/dd Q_ν(cosh(d))=u'(d),
#   where
#       ν=-1/2+i k.
#
# Mathematical model
#   For z=cosh(d)>1, Q_ν(z) solves the associated Legendre ODE. After the change
#   of variables z=cosh(d), the pair (u,y) satisfies the first-order system
#       u'(d)=y(d),
#       y'(d)=ν(ν+1)u(d)-coth(d)y(d).
#
# Discretization strategy (Taylor patches)
#   Choose centers d_i=dmin+(i-1)h and represent u and y locally by Taylor series
#       u(d_i+δ)=Σ_{n=0}^P u_n δ^n,
#       y(d_i+δ)=Σ_{n=0}^P y_n δ^n,
#   with coefficients generated by recurrence from the ODE and the local Taylor
#   series of coth(d_i+δ). We store the coefficients for each patch and evaluate
#   by Horner/Clenshaw-like nested multiplication.
#
# Fields
#   nu      ::ComplexF64      # ν=-1/2+i k
#   dmin    ::Float64         # lower d bound (>0)
#   dmax    ::Float64         # upper d bound
#   h       ::Float64         # uniform patch spacing
#   P       ::Int             # Taylor degree per patch
#   centers ::Vector{Float64} # d_i centers, length Npatch
#   ucoeffs ::Matrix{ComplexF64} # size (P+1,Npatch), u-series coeffs per patch
#   ycoeffs ::Matrix{ComplexF64} # size (P+1,Npatch), y-series coeffs per patch
#
# Notes
#   - Only ONE high-precision seed (u(dmin),y(dmin)) is required.
#   - For d<d_threshold we switch to validated small-d asymptotics.
# =============================================================================
struct QTaylorTable
    nu::ComplexF64
    dmin::Float64
    dmax::Float64
    h::Float64
    P::Int
    centers::Vector{Float64}
    ucoeffs::Matrix{ComplexF64}
    ycoeffs::Matrix{ComplexF64}
end

# =============================================================================
# _patch_index
#
# Map a distance d to a Taylor patch index i.
# Uses uniform spacing h and snaps to the nearest integer index when d lies
# extremely close to a patch boundary (to avoid off-by-one from floating error).
#
# Inputs
#   tab::QTaylorTable
#   d::Float64
#
# Output
#   i::Int in 1:length(tab.centers)
# =============================================================================
@inline function _patch_index(tab::QTaylorTable,d::Float64)
    if d<=tab.dmin
        return 1
    elseif d>=tab.dmax
        return length(tab.centers)
    else
        t=(d-tab.dmin)/tab.h
        idx=Int(floor(t))+1
        if abs(t-round(t))<64*eps(t)
            idx=Int(round(t))+1
        end
        return clamp(idx,1,length(tab.centers))
    end
end

# =============================================================================
# eval_Q
#
# Evaluate u(d)=Q_ν(cosh(d)) from the uniform Taylor-patch table.
#   - Choose patch index i such that d≈centers[i].
#   - Evaluate the stored Taylor polynomial u(d)=Σ u_n (d-centers[i])^n.
#   - For d<d_threshold use the small-d expansion _small_z_Q.
#
# Inputs
#   tab::QTaylorTable
#   d::Float64
#
# Output
#   ComplexF64  (approximation to Q_ν(cosh(d)))
# =============================================================================
@inline function _eval_Q(tab::QTaylorTable,d::Float64)
    dd=Float64(d)
    if dd<d_threshold
        k=imag(tab.nu+0.5)
        return _small_z_Q(k,dd)
    end
    idx=_patch_index(tab,dd)
    x=dd-tab.centers[idx]
    return horner_eval(view(tab.ucoeffs,:,idx),x)
end

# =============================================================================
# eval_dQdd
#
# Evaluate y(d)=d/dd Q_ν(cosh(d)) from the uniform Taylor-patch table.
#   - Same patch selection and Horner evaluation as eval_Q, but using ycoeffs.
#   - For d<d_threshold use the small-d expansion _small_z_dQ.
#
# Inputs
#   tab::QTaylorTable
#   d::Float64
#
# Output
#   ComplexF64  (approximation to d/dd Q_ν(cosh(d)))
# =============================================================================
@inline function _eval_dQdd(tab::QTaylorTable,d::Float64)
    dd=Float64(d)
    if dd<d_threshold
        k=imag(tab.nu+0.5)
        return _small_z_dQ(k,dd)
    end
    idx=_patch_index(tab,dd)
    x=dd-tab.centers[idx]
    return horner_eval(view(tab.ycoeffs,:,idx),x)
end

# =============================================================================
# eval_Q!
#
# Threaded batched evaluation of u(d)=Q_ν(cosh(d)) for a vector of distances.
#
# Inputs
#   out::AbstractVector{ComplexF64} preallocated
#   tab::QTaylorTable
#   dvec::AbstractVector{Float64}
#
# Output
#   nothing (fills out in place)
# =============================================================================
function _eval_Q!(out::AbstractVector{ComplexF64},tab::QTaylorTable,dvec::AbstractVector{Float64})
    @inbounds Threads.@threads for i in eachindex(dvec)
        out[i]=_eval_Q(tab,dvec[i])
    end
    return nothing
end

# =============================================================================
# eval_dQdd!
#
# Threaded batched evaluation of y(d)=d/dd Q_ν(cosh(d)) for a vector of distances.
#
# Inputs
#   out::AbstractVector{ComplexF64} preallocated
#   tab::QTaylorTable
#   dvec::AbstractVector{Float64}
#
# Output
#   nothing (fills out in place)
# =============================================================================
function _eval_dQdd!(out::AbstractVector{ComplexF64},tab::QTaylorTable,dvec::AbstractVector{Float64})
    @inbounds Threads.@threads for i in eachindex(dvec)
        out[i]=_eval_dQdd(tab,dvec[i])
    end
    return nothing
end

# =============================================================================
# build_QTaylorTable
#
# Build a QTaylorTable for a single complex wavenumber k.
#
# Parameters
#   ν=-1/2+i k.
#
# Algorithm
#   1) Seed at dmin using mpmath (2 calls to legenq):
#        u(dmin)=Q_ν(cosh(dmin)),
#        y(dmin)=d/dd Q_ν(cosh(dmin)).
#   2) For each patch center d_i=dmin+(i-1)h:
#        - Expand coth(d_i+δ) in δ to order P (Float64 coeffs).
#        - Generate Taylor coefficients (u_n,y_n) using ODE recurrences.
#        - Store coefficients as columns of ucoeffs,ycoeffs.
#        - Advance the state (u0,y0) to the next center via Horner eval at δ=h.
#
# Complexity (per k)
#   - Python: O(1) calls (seed only).
#   - Julia: O(Npatch*P^2) arithmetic, no allocations in the inner loop.
#
# Inputs (keywords)
#   k::ComplexF64   - wavenumber (can have small Im part)
#   dmin,dmax::Float64 - domain in d, dmin>0
#   h::Float64      - patch spacing
#   P::Int          - Taylor degree
#   mp_dps::Int     - mpmath precision for seeding
#   leg_type::Int=3   - mpmath LegendreQ definition selector
#
# Output
#   QTaylorTable
# =============================================================================
function build_QTaylorTable(k::ComplexF64;dmin::Float64=1e-6,dmax::Float64=5.0,h::Float64=1e-4,P::Int=30,mp_dps::Int=60,leg_type::Int=3)
    @assert dmax>dmin
    @assert h>0
    @assert P≥1
    nu=ComplexF64(-0.5,0.0)+1im*k
    sδ=Vector{Float64}(undef,P+1)
    cδ=Vector{Float64}(undef,P+1)
    series_sinh_cosh!(sδ,cδ)
    u0,y0=seed_u_y_mpmath(nu,dmin;dps=mp_dps,leg_type=leg_type)
    Npatch=Int(ceil((dmax-dmin)/h))
    centers=Vector{Float64}(undef,Npatch)
    @inbounds for i in 1:Npatch
        centers[i]=dmin+(i-1)*h
    end
    ucoeffs=Matrix{ComplexF64}(undef,P+1,Npatch)
    ycoeffs=Matrix{ComplexF64}(undef,P+1,Npatch)
    coth=Vector{Float64}(undef,P+1)
    sinh_series=Vector{Float64}(undef,P+1)
    cosh_series=Vector{Float64}(undef,P+1)
    ucoef=Vector{ComplexF64}(undef,P+1)
    ycoef=Vector{ComplexF64}(undef,P+1)
    sh0=sinh(dmin)
    ch0=cosh(dmin)
    shh=sinh(h)
    chh=cosh(h)
    @inbounds for i in 1:Npatch
        coth_series_from_sinhcosh!(coth,sh0,ch0,sδ,cδ,sinh_series,cosh_series)
        build_patch_coeffs!(ucoef,ycoef,nu,u0,y0,coth)
        @inbounds for n in 1:(P+1)
            ucoeffs[n,i]=ucoef[n]
            ycoeffs[n,i]=ycoef[n]
        end
        if i<Npatch
            u0=horner_eval(ucoef,h)
            y0=horner_eval(ycoef,h)

            sh1=sh0*chh+ch0*shh
            ch1=ch0*chh+sh0*shh
            sh0,ch0=sh1,ch1
        end
    end
    return QTaylorTable(nu,dmin,dmax,h,P,centers,ucoeffs,ycoeffs)
end

# =============================================================================
# build_QTaylorTable (vector-ks overload)
#
# Build QTaylorTable objects for many complex wavenumbers ks.
#
# Parameters
#   ν_i=-1/2+i ks[i].
#
# Algorithm
#   0) Precompute geometry-independent data once:
#        - patch centers d_p = dmin+(p-1)h,  p=1..Npatch
#        - Taylor coefficients of coth(d_p+δ) up to order P for every patch p
#          (stored in coth_coeffs[:,p]).
#   1) Seed each ν_i at dmin using mpmath (2 calls to legenq per k):
#        u_i(dmin)=Q_{ν_i}(cosh(dmin)),
#        y_i(dmin)=d/dd Q_{ν_i}(cosh(dmin)).
#   2) For each i (optionally threaded), build the full table by sweeping patches:
#        For p=1..Npatch:
#          - Generate Taylor coeffs (u_n,y_n) at center d_p via ODE recurrences
#            using precomputed coth_coeffs[:,p].
#          - Store as columns ucoeffs[:,p], ycoeffs[:,p].
#          - Advance (u0,y0) to next center by Horner eval at δ=h.
#
# Complexity
#   - Python: O(Nk) calls (seed only; 2 legenq calls per k).
#   - Julia: O(Nk*Npatch*P^2) arithmetic.
#   - Memory: O(Nk*(P+1)*Npatch) complex numbers for coeffs + shared real buffers.
#
# Threading / allocations
#   - Seeding is done serially (PyCall/mpmath).
#   - The per-k build phase may be threaded; each thread reuses its own
#     (P+1)-length ComplexF64 scratch vectors (ucoef/ycoef) to avoid allocs.
#   - The large ucoeffs/ycoeffs matrices are necessarily allocated per k
#     because each QTaylorTable must own its coefficient storage.
#
# Inputs (keywords)
#   ks::AbstractVector{ComplexF64}   - wavenumbers (may have small Im parts)
#   dmin,dmax::Float64              - domain in d, dmin>0
#   h::Float64                      - patch spacing
#   P::Int                          - Taylor degree
#   mp_dps::Int                     - mpmath precision for seeding
#   leg_type::Int=3                 - mpmath LegendreQ definition selector
#   threaded::Bool=true             - build tables in parallel (Julia only)
#
# Output
#   Vector{QTaylorTable} length Nk, one table per ks[i]
# =============================================================================

function build_QTaylorTable(ks::AbstractVector{ComplexF64};dmin::Float64=1e-6,dmax::Float64=5.0,h::Float64=1e-4,P::Int=30,mp_dps::Int=60,leg_type::Int=3,threaded::Bool=true)
    @assert dmax>dmin && h>0 && P≥1
    Nk=length(ks)
    Npatch=Int(ceil((dmax-dmin)/h))
    centers=Vector{Float64}(undef,Npatch)
    @inbounds for i in 1:Npatch
        centers[i]=dmin+(i-1)*h
    end
    sδ=Vector{Float64}(undef,P+1);cδ=Vector{Float64}(undef,P+1)
    series_sinh_cosh!(sδ,cδ)
    coth_coeffs=Matrix{Float64}(undef,P+1,Npatch)
    sinh_series=Vector{Float64}(undef,P+1);cosh_series=Vector{Float64}(undef,P+1)
    coth=Vector{Float64}(undef,P+1)
    sh0=sinh(dmin);ch0=cosh(dmin)
    shh=sinh(h);chh=cosh(h)
    @inbounds for p in 1:Npatch
        coth_series_from_sinhcosh!(coth,sh0,ch0,sδ,cδ,sinh_series,cosh_series)
        @views copyto!(coth_coeffs[:,p],coth)
        if p<Npatch
            sh1=sh0*chh+ch0*shh
            ch1=ch0*chh+sh0*shh
            sh0,ch0=sh1,ch1
        end
    end
    nus=Vector{ComplexF64}(undef,Nk)
    u0s=Vector{ComplexF64}(undef,Nk)
    y0s=Vector{ComplexF64}(undef,Nk)
    for i in 1:Nk
        nu=ComplexF64(-0.5,0.0)+1im*ks[i]
        nus[i]=nu
        u0s[i],y0s[i]=seed_u_y_mpmath(nu,dmin;dps=mp_dps,leg_type=leg_type)
    end
    tabs=Vector{QTaylorTable}(undef,Nk)
    NT=threaded ? Threads.nthreads() : 1
    ucoef_tls=[Vector{ComplexF64}(undef,P+1) for _ in 1:NT]
    ycoef_tls=[Vector{ComplexF64}(undef,P+1) for _ in 1:NT]
    function build_one!(i,tid)
        nu=nus[i];u0=u0s[i];y0=y0s[i]
        ucoeffs=Matrix{ComplexF64}(undef,P+1,Npatch)
        ycoeffs=Matrix{ComplexF64}(undef,P+1,Npatch)
        ucoef=ucoef_tls[tid];ycoef=ycoef_tls[tid]
        @inbounds for p in 1:Npatch
            build_patch_coeffs!(ucoef,ycoef,nu,u0,y0,@view(coth_coeffs[:,p]))
            @views copyto!(ucoeffs[:,p],ucoef)
            @views copyto!(ycoeffs[:,p],ycoef)
            if p<Npatch
                u0=horner_eval(ucoef,h)
                y0=horner_eval(ycoef,h)
            end
        end
        tabs[i]=QTaylorTable(nu,dmin,dmax,h,P,centers,ucoeffs,ycoeffs)
        return nothing
    end
    if threaded && Threads.nthreads()>1
        Threads.@threads for i in 1:Nk
            build_one!(i,Threads.threadid())
        end
    else
        for i in 1:Nk
            build_one!(i,1)
        end
    end
    return tabs
end

# =============================================================================
# Construction of hyperbolic distances 
# =============================================================================

# =============================================================================
# Euclidean normal-derivative of the hyperbolic distance d in the Poincaré disk.
#
# Geometry
#   Points x=(x1,x2),x'=(xp1,xp2) lie in the unit disk. The hyperbolic distance
#   satisfies
#       cosh(d)=1+2|x'-x|^2/((1-|x|^2)(1-|x'|^2)).
#
# This routine returns ∂_{n'_E}d, the derivative of d w.r.t. the *Euclidean*
# outward unit normal at the source point x' (no hyperbolic normal factors).
#
# Inputs
#   x,y::Float64      - target point x in disk
#   xp,yp::Float64    - source point x' in disk
#   nx,ny::Float64      - Euclidean unit normal at x'
#
# Output
#   Float64             - ∂_{n'_E}d(x,x')
# =============================================================================
@inline function _∂n_d(x::T,y::T,xp::T,yp::T,nx::T,ny::T) where {T<:Real}
    ax=one(T)-muladd(x,x,y*y)
    bx=one(T)-muladd(xp,xp,yp*yp)
    dx=xp-x;dy = yp-y
    r2=muladd(dx,dx,dy*dy)
    c=one(T)+2*r2/(ax*bx)               # = cosh(d)
    sh=sqrt(max(c*c-one(T),zero(T)))       # = sinh(d)
    dotdxn=muladd(dx,nx,dy*ny)             # (x' - x)·n'
    dotxpn=muladd(xp,nx,yp*ny)             # x'·n'
    return (4/(ax*bx))*dotdxn/sh+(4*r2/(ax*bx*bx))*dotxpn/sh
end

# =============================================================================
# build_dvec_disk!
#
# Construct hyperbolic distances d(x_i,xp_i) for many pairs in the Poincaré disk.
#
# Math
#   cosh(d)=1+2|xp-x|^2/((1-|x|^2)(1-|xp|^2)),  d=acosh(cosh(d)).
#
# Inputs
#   dvec::Vector{Float64}          preallocated length N
#   x1,x2::Vector{Float64}         target coords length N
#   xp1,xp2::Vector{Float64}       source coords length N
#
# Output
#   nothing (fills dvec)
# =============================================================================
@inline function build_dvec_disk!(dvec::Vector{Float64},x1::Vector{Float64},x2::Vector{Float64},xp1::Vector{Float64},xp2::Vector{Float64})
    @inbounds Threads.@threads for i in eachindex(dvec)
        x1i=x1[i];x2i=x2[i];xp1i=xp1[i];xp2i=xp2[i]
        ax=1.0-(x1i*x1i+x2i*x2i)
        bx=1.0-(xp1i*xp1i+xp2i*xp2i)
        dx1=xp1i-x1i;dx2=xp2i-x2i
        r2=dx1*dx1+dx2*dx2
        c=1.0+2.0*r2/(ax*bx)
        dvec[i]=acosh(c)
    end
    return nothing
end

# =============================================================================
# build_dnvec_disk!
#
# Construct Euclidean-normal derivatives ∂_{n'_E}d(x_i,xp_i) for many pairs.
# Uses your scalar ∂n_d(...) which is the only geometry-dependent ingredient.
#
# Inputs
#   dnvec::Vector{Float64}         preallocated length N
#   x1,x2,xp1,xp2::Vector{Float64} coords length N
#   n1,n2::Vector{Float64}         Euclidean unit normals at sources length N
#
# Output
#   nothing (fills dnvec)
# =============================================================================
@inline function build_dn_vec_disk!(dnvec::Vector{Float64},x1::Vector{Float64},x2::Vector{Float64},xp1::Vector{Float64},xp2::Vector{Float64},n1::Vector{Float64},n2::Vector{Float64})
    @inbounds Threads.@threads for i in eachindex(dnvec)
        dnvec[i]=_∂n_d(x1[i],x2[i],xp1[i],xp2[i],n1[i],n2[i])
    end
    return nothing
end

@inline function slp_hyperbolic_kernel(tab::QTaylorTable,x1::Float64,x2::Float64,xp1::Float64,xp2::Float64)
    return 1.0/TWO_PI*_eval_Q(tab,acosh(1.0+2.0*((xp1-x1)^2+(xp2-x2)^2)/((1.0-(x1^2+x2^2))*(1.0-(xp1^2+xp2^2)))))
end

# =============================================================================
# dlp_kernel_disk
#
# Hyperbolic DLP kernel in the Poincaré disk using the Taylor table for y(d):
#   y(d)=d/dd Q_ν(cosh(d)).
#
# Kernel factorization
#   For the Green's function G_k(d)=(1/(2π))Q_ν(cosh(d)), the DLP kernel involves
#       ∂_{n'}G_k(d)=(1/(2π))y(d)*∂_{n'_E}d,
# where ∂_{n'_E}d is the Euclidean-normal derivative returned by dnd_euclid_disk.
#
# Inputs
#   tab::QTaylorTable   - precomputed tables for Q and dQ/dd at fixed ν
#   x1,x2::Float64      - target point x in disk
#   xp1,xp2::Float64    - source point x' in disk
#   n1,n2::Float64      - Euclidean unit normal at x'
#
# Output
#   ComplexF64          - (1/(2π))y(d)*∂_{n'_E}d
# =============================================================================
@inline function dlp_hyperbolic_kernel(tab::QTaylorTable,x1::Float64,x2::Float64,xp1::Float64,xp2::Float64,n1::Float64,n2::Float64)
    ax=1.0-(x1*x1+x2*x2)
    bx=1.0-(xp1*xp1+xp2*xp2)
    dx1=xp1-x1;dx2=xp2-x2
    r2=dx1*dx1+dx2*dx2
    c=1.0+2.0*r2/(ax*bx)
    d=acosh(c)
    y=_eval_dQdd(tab,d)
    dn=_∂n_d(x1,x2,xp1,xp2,n1,n2)
    return (y*dn)/(2*pi)
end

# =============================================================================
# slp_hyperbolic_kernel!
#
# K_SLP(x,x')=(1/(2π))Q_ν(cosh(d(x,x'))).
#
# Inputs
#   out::Vector{ComplexF64}  preallocated length N
#   tab::QTaylorTable
#   dvec::Vector{Float64}    precomputed distances
# =============================================================================
@inline function slp_hyperbolic_kernel!(out::Vector{ComplexF64},tab::QTaylorTable,dvec::Vector{Float64})
    _eval_Q!(out,tab,dvec)
    inv2π=1.0/TWO_PI
    @inbounds for i in eachindex(out)
        out[i]*=inv2π
    end
    return nothing
end

# =============================================================================
# dlp_hyperbolic_kernel!
#
# K_DLP(x,x')=(1/(2π))y(d)*∂_{n'_E}d,  y(d)=d/dd Q_ν(cosh(d)).
#
# Inputs
#   out::Vector{ComplexF64}  preallocated length N
#   tab::QTaylorTable
#   dvec::Vector{Float64}    precomputed distances
#   dnvec::Vector{Float64}   precomputed ∂n d
# =============================================================================
@inline function dlp_hyperbolic_kernel!(out::Vector{ComplexF64},tab::QTaylorTable,dvec::Vector{Float64},dnvec::Vector{Float64})
    _eval_dQdd!(out,tab,dvec)
    inv2π=1.0/TWO_PI
    @inbounds for i in eachindex(out)
        out[i]*=(dnvec[i]*inv2π)
    end
    return nothing
end