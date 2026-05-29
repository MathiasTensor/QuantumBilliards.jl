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
# MO / 30/12/25
# =============================================================================

# uncomment these when testing this file in isolation, but they are already imported in QuantumBilliards.jl
#using PyCall
#using SpecialFunctions
#using LinearAlgebra
#using Test
#using Random
#using BenchmarkTools

# also import the mpmath objects (since QuantumBilliards.jl has these in pycall_init.jl)

const TWO_PI=2*pi
const FOUR_PI=4*pi

# Import mpmath functions via PyCall for high-precision seeding (at d_min calls 2 calls to legenq are needed)

const inv2π=1.0/TWO_PI
const inv4π=1.0/FOUR_PI
const Z_threshold=1.0+1e-14 # threshold for |z-1| smallness for the Legendre Q required for the SLP kernel
const d_threshold=1e-3

Base.@kwdef mutable struct LegendreTaylorConfig
    h_patch::Float64=1e-5 # patch half-width. This is the main parameter controlling the table size and accuracy. Smaller h means more patches and more accuracy, but also more memory and longer precomputation time. 1e-5 is a good default for double precision and k up to a few thousands, giving rel accuracy of 1e-13 at d=1e-3.
    P_patch::Int=8 # patch order. This is the order of the Taylor expansion in each patch. Higher order means more accuracy but also more memory and longer precomputation time. 8 is a good default for double precision and k up to a few thousands, giving rel accuracy of 1e-13 at d=1e-3.
    d_threshold::Float64=1e-3 # corresponding d threshold for cosh(d)=z close to 1 (if smaller than 1e-3 invalidates tables)
    P_small_terms::Int=48 # usually this is enough for P even up to d=1e-2 for k in the thousands in double precision.
    dQ_small_terms::Int=24 # this is good enought fo the default d_threshold of 1e-3.
end

const LEGENDRE_TAYLOR_CONFIG=LegendreTaylorConfig()

@inline legendre_Q_h_patch()=LEGENDRE_TAYLOR_CONFIG.h_patch
@inline legendre_Q_P_patch()=LEGENDRE_TAYLOR_CONFIG.P_patch
@inline legendre_d_threshold()=LEGENDRE_TAYLOR_CONFIG.d_threshold
@inline legendre_P_small_terms()=LEGENDRE_TAYLOR_CONFIG.P_small_terms
@inline legendre_dQ_small_terms()=LEGENDRE_TAYLOR_CONFIG.dQ_small_terms
@inline function legendre_Q_params()
    cfg=LEGENDRE_TAYLOR_CONFIG
    return cfg.h_patch,cfg.P_patch
end

function legendre_Q_set_h!(h::Real)
    h>0 || error("h_patch must be positive.")
    LEGENDRE_TAYLOR_CONFIG.h_patch=Float64(h)
    return LEGENDRE_TAYLOR_CONFIG
end

function legendre_Q_set_P!(P::Integer)
    P>=1 || error("P_patch must be at least 1.")
    LEGENDRE_TAYLOR_CONFIG.P_patch=Int(P)
    return LEGENDRE_TAYLOR_CONFIG
end

function legendre_Q_set_d_threshold!(d::Real)
    d>0 || error("d_threshold must be positive.")
    LEGENDRE_TAYLOR_CONFIG.d_threshold=Float64(d)
    return LEGENDRE_TAYLOR_CONFIG
end

function legendre_Q_set_P_small_terms!(n::Integer)
    n>=2 || error("P_small_terms must be at least 2.")
    LEGENDRE_TAYLOR_CONFIG.P_small_terms=Int(n)
    return LEGENDRE_TAYLOR_CONFIG
end

function legendre_Q_set_dQ_small_terms!(n::Integer)
    n>=2 || error("dQ_small_terms must be at least 2.")
    LEGENDRE_TAYLOR_CONFIG.dQ_small_terms=Int(n)
    return LEGENDRE_TAYLOR_CONFIG
end

function legendre_Q_set_taylor_params!(;h_patch=nothing,P_patch=nothing,d_threshold=nothing,P_small_terms=nothing,dQ_small_terms=nothing)
    !isnothing(h_patch) && legendre_Q_set_h!(h_patch)
    !isnothing(P_patch) && legendre_Q_set_P!(P_patch)
    !isnothing(d_threshold) && legendre_Q_set_d_threshold!(d_threshold)
    !isnothing(P_small_terms) && legendre_Q_set_P_small_terms!(P_small_terms)
    !isnothing(dQ_small_terms) && legendre_Q_set_dQ_small_terms!(dQ_small_terms)
    return LEGENDRE_TAYLOR_CONFIG
end

# Thread-local storage for the small-z Taylor expansion buffers, to avoid allocations in the small-d regime where we use a Taylor expansion instead of the patch table. The size is determined by legendre_P_small_terms() since we need that many terms in the expansion.
const SMALL_P_TLS=Ref{Vector{Vector{ComplexF64}}}()
const SMALL_QD_TLS=Ref{Vector{Tuple{Vector{ComplexF64},Vector{ComplexF64}}}}()

function init_small_p_tls!()
    n=legendre_P_small_terms()+1
    SMALL_P_TLS[]=[Vector{ComplexF64}(undef,n) for _ in 1:Threads.nthreads()]
    return nothing
end

function init_small_qd_tls!()
    n=legendre_dQ_small_terms()+1
    SMALL_QD_TLS[]=[(Vector{ComplexF64}(undef,n),Vector{ComplexF64}(undef,n)) for _ in 1:Threads.nthreads()]
    return nothing
end

@inline function small_p_buffer(terms::Int)
    if !isassigned(SMALL_P_TLS)
        init_small_p_tls!()
    end
    tls=SMALL_P_TLS[]
    if Threads.threadid()>length(tls) || terms+1>length(tls[Threads.threadid()])
        init_small_p_tls!()
        tls=SMALL_P_TLS[]
    end
    return tls[Threads.threadid()]
end

@inline function small_qd_buffers(terms::Int)
    if !isassigned(SMALL_QD_TLS)
        init_small_qd_tls!()
    end
    tls=SMALL_QD_TLS[]
    tid=Threads.threadid()
    if tid>length(tls) || terms+1>length(tls[tid][1])
        init_small_qd_tls!()
        tls=SMALL_QD_TLS[]
    end
    return tls[tid]
end

# Coefficients in the even-power expansion
#
#     coth(d) - 1/d = Σ c_n d^(2n+1),
#
# used in the Taylor recurrence for P_ν(cosh d)
# and also the Taylor recurrence for d/dd P_ν(cosh d).
const _COTH_SERIES_COEFFS=Float64[
    1/3,
    -1/45,
    2/945,
    -1/4725,
    2/93555,
    -1382/638512875,
    4/18243225,
    -3617/325641566250,
    43867/38979295480125,
    -174611/1531329465290625,
    155366/13447856940643125,
    -236364091/201919571963756521875,
    1315862/11094481976030578125,
    -6785560294/564653660170076273671875,
    6892673020804/5660878804669082674070015625,
    -9459803781912212/76410959832894957292156621875000
]

# =============================================
# _small_z_Q
# Series expansions for Q_ν(cosh d) at small d.
# The order is chosen to give good accuracy (rel accuracy 12-13 digits) at d~1e-3 (order O(d^4)).
#
# Input
#   k::ComplexF64
#   d::Real, small distance
#
# Output
#   Q_ν(cosh d)
# =============================================
@inline function _small_z_Q(k::ComplexF64,d::T)::ComplexF64 where{T<:Real}
    H=MathConstants.eulergamma+SpecialFunctions.digamma(ComplexF64(0.5,0.0)+1im*k)
    Ld=log(d)
    L2=log(2.0)
    L4=2*L2 # L4=log(4)
    L8=3*L2 # L8=log(8)
    k2=k*k
    k4=k2*k2
    term0=(-H+L2-Ld)/(2*pi)
    term2=(1-12*k2+3*H+12*k2*H-3*L2-4*k2*L8+3*Ld+12*k2*Ld)*d^2/(96*pi)
    term4=(-193+1560*k2+2160*k4-330*H-1680*k2*H-1440*k4*H+165*L4+840*k2*L4+720*k4*L4-330*Ld-1680*k2*Ld-1440*k4*Ld)*d^4/(184320*pi)
    return 2*pi*(term0+term2+term4)
end

# Real k wrapper for _small_z_Q
@inline function _small_z_Q(k::T,d::T) where{T<:Real}
    return _small_z_Q(complex(k,0.0),d)
end

# =============================================
# _small_z_dQ_LEGACY
# Series expansions for d/dd Q_ν(cosh d) at small d.
# The order is chosen to give good accuracy (rel accuracy 12-13 digits) at d~1e-3 (order O(d^27) since it is highly oscillatory).
#
# Input
#   k::ComplexF64
#   d::Real, small distance
#
# Output
#   d/dd Q_ν(cosh d)
@inline function _small_z_dQ_LEGACY(k::ComplexF64,d::T)::ComplexF64 where{T<:Real}
    nu=ν(k)
    Hnu=Base.MathConstants.eulergamma+SpecialFunctions.digamma(nu+one(nu))
    return ComplexF64(-(one(T)/d)+(d*(2+3*nu*(1+nu)))/12+(d^3*(-56+15*nu*(1+nu)*(-2+15*nu*(1+nu))))/2880+(d^5*(248+21*nu*(1+nu)*(8+5*nu*(1+nu)*(-7+5*nu*(1+nu)))))/120960+(d^7*(-48768+5*nu*(1+nu)*(-7568+21*nu*(1+nu)*(1268+5*nu*(1+nu)*(-156+47*nu*(1+nu))))))/2.322432e8+(d^9*(1308160+11*nu*(1+nu)*(98432+3*nu*(1+nu)*(-104864+7*nu*(1+nu)*(8284+nu*(1+nu)*(-2320+393*nu*(1+nu)))))))/6.13122048e10+(d^11*(-724212224+273*nu*(1+nu)*(-2267520+11*nu*(1+nu)*(628720+nu*(1+nu)*(-327944+21*nu*(1+nu)*(3968+nu*(1+nu)*(-650+71*nu*(1+nu))))))))/3.34764638208e14+(d^13*(3522785280+nu*(1+nu)*(3064070144+429*nu*(1+nu)*(-21245568+nu*(1+nu)*(10734208+nu*(1+nu)*(-2570552+3*nu*(1+nu)*(127876+nu*(1+nu)*(-13846+1059*nu*(1+nu)))))))))/1.6068702633984e16+(d^15*(-81555718766592+17*nu*(1+nu)*(-4215787018240+3*nu*(1+nu)*(4119262721280+143*nu*(1+nu)*(-14288101248+nu*(1+nu)*(3308766736+3*nu*(1+nu)*(-154931200+nu*(1+nu)*(15324344+45*nu*(1+nu)*(-26264+1487*nu*(1+nu))))))))))/3.6713771778126643e21+(d^17*(7536235717591040+57*nu*(1+nu)*(116947251986432+17*nu*(1+nu)*(-19979202338816+nu*(1+nu)*(9796767888640+13*nu*(1+nu)*(-171086931456+11*nu*(1+nu)*(2109795216+nu*(1+nu)*(-196236224+nu*(1+nu)*(13852776+5*nu*(1+nu)*(-160176+6989*nu*(1+nu)))))))))))/3.34829598616515e24+(d^19*(-3023786723765649408+55*nu*(1+nu)*(-48844705828831232+57*nu*(1+nu)*(2473269691121664+17*nu*(1+nu)*(-70796702697984+nu*(1+nu)*(15871802558208+13*nu*(1+nu)*(-162189398880+11*nu*(1+nu)*(1322981232+5*nu*(1+nu)*(-17552464+nu*(1+nu)*(930264+nu*(1+nu)*(-41850+1451*nu*(1+nu))))))))))))/1.3259252105213994e28+(d^21*(5919143148921500467200+23*nu*(1+nu)*(229361823479855316992+15*nu*(1+nu)*(-43938582105369018368+19*nu*(1+nu)*(1119337009242789888+17*nu*(1+nu)*(-14636176930816512+nu*(1+nu)*(1918454135937408+13*nu*(1+nu)*(-12957794542560+nu*(1+nu)*(828616686128+nu*(1+nu)*(-41247566448+7*nu*(1+nu)*(243316128+nu*(1+nu)*(-8764690+247353*nu*(1+nu)))))))))))))/2.5616875067273434e32+(d^23*(-686096493620974804008960+13*nu*(1+nu)*(-47144867132796858793984+23*nu*(1+nu)*(5871102997047666982912+5*nu*(1+nu)*(-566138921659181092864+57*nu*(1+nu)*(2194468161804355584+17*nu*(1+nu)*(-16765943320964352+nu*(1+nu)*(1451742074957120+13*nu*(1+nu)*(-6984481999040+nu*(1+nu)*(334973138896+nu*(1+nu)*(-12986455152+7*nu*(1+nu)*(61403188+nu*(1+nu)*(-1811612+42433*nu*(1+nu))))))))))))))/2.930570507696081e35+(d^25*(33367733728285762089123840+nu*(1+nu)*(29859320862789776542007296+5*nu*(1+nu)*(-17063269414016842458202112+23*nu*(1+nu)*(356639815691069055434752+nu*(1+nu)*(-78443654495831951527936+57*nu*(1+nu)*(177569554987496819712+17*nu*(1+nu)*(-895795952130801408+nu*(1+nu)*(55227923257381696+nu*(1+nu)*(-2589559892863424+nu*(1+nu)*(96684098805008+nu*(1+nu)*(-3003546729040+7*nu*(1+nu)*(11641920004+5*nu*(1+nu)*(-57309356+1132133*nu*(1+nu)))))))))))))))/1.4066738436941188e38+(d^27*(-91770018091370053888741736448+29*nu*(1+nu)*(-2835615745800674524836921344+3*nu*(1+nu)*(2695606115502739701208973312+5*nu*(1+nu)*(-258579031590302542450655232+23*nu*(1+nu)*(2464338533704759823015936+nu*(1+nu)*(-316405030616016052463616+19*nu*(1+nu)*(1418259159879823898112+17*nu*(1+nu)*(-5092635130610819200+nu*(1+nu)*(235309612303988928+nu*(1+nu)*(-8586706803750816+nu*(1+nu)*(256808887011248+3*nu*(1+nu)*(-2179494838664+nu*(1+nu)*(49370502636+5*nu*(1+nu)*(-205868026+3476589*nu*(1+nu))))))))))))))))/3.818275481323316e42-(d*nu*(1+nu)*(182684914765469984565271461888000000+7611871448561249356886310912000000*d^2*(-2+3*nu*(1+nu))+190296786214031233922157772800000*d^4*(8+5*(-1+nu)*nu*(1+nu)*(2+nu))+566359482779854862863564800000*d^6*(-272+7*nu*(1+nu)*(44+5*nu*(1+nu)*(-4+nu+nu^2)))+3933051963748992103219200000*d^8*(3968+3*nu*(1+nu)*(-1424+7*nu*(1+nu)*(84+nu*(1+nu)*(-20+3*nu*(1+nu)))))+8938754463065891143680000*d^10*(-176896+11*nu*(1+nu)*(16864+nu*(1+nu)*(-6584+21*nu*(1+nu)*(68+nu*(1+nu)*(-10+nu+nu^2)))))+28649854048288112640000*d^12*(5592064+13*nu*(1+nu)*(-444544+11*nu*(1+nu)*(15296+nu*(1+nu)*(-3128+3*nu*(1+nu)*(140+nu*(1+nu)*(-14+nu+nu^2))))))+8526742276276224000*d^14*(-1903757312+3*nu*(1+nu)*(650078976+13*nu*(1+nu)*(-18589056+11*nu*(1+nu)*(334192+nu*(1+nu)*(-42240+nu*(1+nu)*(3864+5*nu*(1+nu)*(-56+3*nu*(1+nu))))))))+15674158596096000*d^16*(104932671488+17*nu*(1+nu)*(-6287587328+nu*(1+nu)*(2311237888+13*nu*(1+nu)*(-34468224+11*nu*(1+nu)*(382416+nu*(1+nu)*(-32896+nu*(1+nu)*(2184+5*nu*(1+nu)*(-24+nu+nu^2))))))))+11457718272000*d^18*(-14544442556416+19*nu*(1+nu)*(776768475136+17*nu*(1+nu)*(-16670893568+nu*(1+nu)*(3191642624+13*nu*(1+nu)*(-29341344+11*nu*(1+nu)*(221328+nu*(1+nu)*(-13808+nu*(1+nu)*(696+(-5+nu)*nu*(1+nu)*(6+nu)))))))))+13640140800*d^20*(1237874513281024+nu*(1+nu)*(-1252648497168384+19*nu*(1+nu)*(23928414824448+17*nu*(1+nu)*(-267191996928+nu*(1+nu)*(31506758784+13*nu*(1+nu)*(-196781088+nu*(1+nu)*(11833360+nu*(1+nu)*(-560208+7*nu*(1+nu)*(3168+nu*(1+nu)*(-110+3*nu*(1+nu)))))))))))+3369600*d^22*(-507711943253426176+23*nu*(1+nu)*(22292423254048768+nu*(1+nu)*(-8059883338280960+19*nu*(1+nu)*(80038242321408+17*nu*(1+nu)*(-550103120640+nu*(1+nu)*(44044491328+13*nu*(1+nu)*(-199258304+nu*(1+nu)*(9087760+nu*(1+nu)*(-337744+7*nu*(1+nu)*(1540+nu*(1+nu)*(-44+nu+nu^2)))))))))))+2808*d^24*(61730370047551995904+5*nu*(1+nu)*(-12448661250518024192+23*nu*(1+nu)*(195122177890779136+nu*(1+nu)*(-36649899662381056+19*nu*(1+nu)*(223895553555456+17*nu*(1+nu)*(-1044390696192+nu*(1+nu)*(60545295936+nu*(1+nu)*(-2699564608+nu*(1+nu)*(96615376+nu*(1+nu)*(-2894320+7*nu*(1+nu)*(10868+5*nu*(1+nu)*(-52+nu+nu^2))))))))))))+d^26*(-17562900400985989971968+3*nu*(1+nu)*(5895807142545951555584+5*nu*(1+nu)*(-424139144014000029696+23*nu*(1+nu)*(3451830441237979136+nu*(1+nu)*(-398682053674530816+19*nu*(1+nu)*(1652375735883264+17*nu*(1+nu)*(-5578990587776+nu*(1+nu)*(245125459008+nu*(1+nu)*(-8573995872+nu*(1+nu)*(247266448+3*nu*(1+nu)*(-2032888+nu*(1+nu)*(44772+5*nu*(1+nu)*(-182+3*nu*(1+nu)))))))))))))))*(Hnu+log(d/2)))/3.6536982953093996e35)
end

# =============================================================================
# _small_z_dQ
#
# Small-distance expansion for
#
#     d/dd Q_ν(cosh d).
#
# Mathematical background
#   Near d=0 the Legendre Q solution admits the Frobenius form
#
#       Q_ν(cosh d) = -P_ν(cosh d) log(d/2) + B_ν(d),
#
#   where both
#
#       P_ν(cosh d)
#
#   and the analytic remainder
#
#       B_ν(d)
#
#   satisfy the radial Legendre equation
#
#       u'' + coth(d)u' - ν(ν+1)u = 0.
#
#   Writing
#
#       P_ν(cosh d)
#       = Σ a_n d^(2n),
#
#       B_ν(d)
#       = Σ b_n d^(2n),
#
#   and substituting into the ODE yields coupled recurrences for the
#   coefficients a_n and b_n. The coefficients are generated on-the-fly
#   using the Taylor series of
#
#       coth(d)-1/d.
#
# Derivative formula
#   Differentiating
#
#       Q_ν(cosh d) = -P_ν(cosh d) log(d/2) + B_ν(d)
#
#   gives
#
#       dQ/dd = -P_ν(d)/d - P_ν'(d) log(d/2) + B_ν'(d).
#
#   The three terms are evaluated from Horner sums in d².
#
# Inputs
#   k::ComplexF64
#       Wavenumber parameter with : ν = -1/2 + i k.
#
#   d::Real : Hyperbolic distance, assumed small.
#
# Keywords
#   terms::Int=legendre_P_small_terms()
#   Number of even-power coefficients retained in the Frobenius expansion.
#
# Output
#   ComplexF64 : Approximation of d/dd Q_ν(cosh d).
# =============================================================================
@inline function _small_z_dQ(k::ComplexF64,d::T;terms::Int=legendre_dQ_small_terms())::ComplexF64 where {T<:Real}
    a,b=small_qd_buffers(terms)
    nu=ν(k)
    lam=nu*(nu+1)
    Hnu=Base.MathConstants.eulergamma+SpecialFunctions.digamma(nu+one(nu))
    a[1]=1.0+0.0im
    @inbounds for m in 0:terms-1
        s=0.0+0.0im
        for r in 1:min(m,length(_COTH_SERIES_COEFFS))
            n=m-r+1
            s+=ComplexF64(_COTH_SERIES_COEFFS[r],0.0)*ComplexF64(2n,0.0)*a[n+1]
        end
        a[m+2]=(lam*a[m+1]-s)/ComplexF64(4*(m+1)^2,0.0)
    end
    b[1]=-Hnu
    @inbounds for m in 0:terms-1
        sb=0.0+0.0im
        for r in 1:min(m,length(_COTH_SERIES_COEFFS))
            n=m-r+1
            sb+=ComplexF64(_COTH_SERIES_COEFFS[r],0.0)*ComplexF64(2n,0.0)*b[n+1]
        end
        rhs=ComplexF64(4*(m+1),0.0)*a[m+2]
        for r in 1:min(m+1,length(_COTH_SERIES_COEFFS))
            n=m-r+1
            rhs+=ComplexF64(_COTH_SERIES_COEFFS[r],0.0)*a[n+1]
        end
        b[m+2]=(lam*b[m+1]+rhs-sb)/ComplexF64(4*(m+1)^2,0.0)
    end
    x=ComplexF64(d*d,0.0)
    L=ComplexF64(log(Float64(d)/2),0.0)
    p=0.0+0.0im
    dp=0.0+0.0im
    db=0.0+0.0im
    @inbounds for n in terms:-1:0
        p=muladd(p,x,a[n+1])
    end
    @inbounds for n in terms:-1:1
        c=ComplexF64(2n,0.0)
        dp=muladd(dp,x,c*a[n+1])
        db=muladd(db,x,c*b[n+1])
    end
    dd=ComplexF64(d,0.0)
    return -p/dd-L*(dd*dp)+dd*db
end

# Real k wrapper for _small_z_dQ
@inline function _small_z_dQ(k::T,d::T) where{T<:Real}
    return _small_z_dQ(ComplexF64(k,0.0),d)
end

# Small-distance Taylor expansion for
#
#     P_ν(cosh d)
#
# obtained from the Frobenius expansion of the radial Legendre equation
#
#     P'' + coth(d) P' - ν(ν+1) P = 0
#
# around d=0.
#
# The solution is expanded as
#
#     P_ν(cosh d) = Σ a_n d^(2n),
#
# and the coefficients a_n are generated recursively using the
# series expansion of coth(d)-1/d.
#
# This branch is used only for very small d, where direct special-function
# evaluation loses relative accuracy due to cancellation.
@inline function _small_z_P(k::ComplexF64,d::T;terms::Int=legendre_P_small_terms()) where {T<:Real}
    a=small_p_buffer(terms)
    nu=ν(k)
    lam=nu*(nu+1)
    x=ComplexF64(d*d,0.0)
    a[1]=1.0+0.0im
    @inbounds for m in 0:terms-1
        s=0.0+0.0im
        rmax=min(m,length(_COTH_SERIES_COEFFS))
        for r in 1:rmax
            n=m-r+1
            s+=ComplexF64(_COTH_SERIES_COEFFS[r],0.0)*ComplexF64(2n,0.0)*a[n+1]
        end
        a[m+2]=(lam*a[m+1]-s)/ComplexF64(4*(m+1)^2,0.0)
    end
    acc=0.0+0.0im
    @inbounds for n in terms:-1:0
        acc=muladd(acc,x,a[n+1])
    end
    return acc
end

# Small-distance derivative expansion
#
#     d/dd P_ν(cosh d).
#
# Differentiating
#
#     P_ν(cosh d) = Σ a_n d^(2n)
#
# gives
#
#     P'(d) = Σ 2n a_n d^(2n-1).
#
# The same recursively generated Taylor coefficients are reused.
#
# This is primarily needed for stable evaluation of logarithmic
# kernel coefficients in the hyperbolic Kress splitting near d=0.
@inline function _small_z_dP(k::ComplexF64,d::T;terms::Int=legendre_P_small_terms()) where {T<:Real}
    a=small_p_buffer(terms)
    nu=ν(k)
    lam=nu*(nu+1)
    x=ComplexF64(d*d,0.0)
    a[1]=1.0+0.0im
    @inbounds for m in 0:terms-1
        s=0.0+0.0im
        rmax=min(m,length(_COTH_SERIES_COEFFS))
        for r in 1:rmax
            n=m-r+1
            s+=ComplexF64(_COTH_SERIES_COEFFS[r],0.0)*ComplexF64(2n,0.0)*a[n+1]
        end
        a[m+2]=(lam*a[m+1]-s)/ComplexF64(4*(m+1)^2,0.0)
    end
    acc=0.0+0.0im
    @inbounds for n in terms:-1:1
        acc=muladd(acc,x,ComplexF64(2n,0.0)*a[n+1])
    end
    return ComplexF64(d,0.0)*acc
end

@inline _small_z_P(k::T,d::T;terms::Int=legendre_P_small_terms()) where {T<:Real}=_small_z_P(ComplexF64(k,0.0),d;terms=terms)
@inline _small_z_dP(k::T,d::T;terms::Int=legendre_P_small_terms()) where {T<:Real}=_small_z_dP(ComplexF64(k,0.0),d;terms=terms)

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
@inline ν(k::ComplexF64)=-0.5+im*k
@inline k_from_ν(nu::ComplexF64)=-1im*(nu+0.5)

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
    lock(PYCALL_MPMATH_LOCK)
    try
        _mpctx[].dps=dps
        nu_py=_mpc[](real(nu),imag(nu))
        d_py=_mpf[](d0)
        z=_mp_cosh[](d_py)
        sh=_mp_sinh[](d_py)
        Qnu=_mp_legenq[](nu_py,0,z;type=leg_type)
        Qnu1=_mp_legenq[](nu_py+1,0,z;type=leg_type)
        y0=(nu_py+1)*(Qnu1-z*Qnu)/sh
        u_re=pycall(_pyfloat[],Float64,Qnu.real)
        u_im=pycall(_pyfloat[],Float64,Qnu.imag)
        y_re=pycall(_pyfloat[],Float64,y0.real)
        y_im=pycall(_pyfloat[],Float64,y0.imag)
        ComplexF64(u_re,u_im),ComplexF64(y_re,y_im)
    finally
        unlock(PYCALL_MPMATH_LOCK)
    end
end

# =============================================================================
# series_sinh_cosh!
#
# Build Taylor coefficients about δ=0 for
#   sinh(δ)=Σ_{n=0}^P sδ[n+1] δ^n,
#   cosh(δ)=Σ_{n=0}^P cδ[n+1] δ^n,
# with coefficients 1/n! on the appropriate parity. 
# This way we get series for sinh(δ) and cosh(δ) cheaply simultaneously.
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
# Power-series division q=num/den up to order P, assuming den[1]≠0.
# Coefficients are stored as q[n+1]=q_n. The algirith is the standard recurrence:
#   q_0=num_0/den_0,
#   q_n=(num_n - Σ_{k=1}^n den_k q_{n-k})/den_0,   n≥1.
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
# Construct Taylor coefficients about δ for
#   coth(d0+δ)=cosh(d0+δ)/sinh(d0+δ),
# given sinh(d0)=sinh0 and cosh(d0)=cosh0. The idea is that 
# d0 is the patch center and δ is the local variable within the patch.
# Therefore we need only calculate the Taylor series of sinh(δ) and cosh(δ)
# efficiently.
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
# Starts at the higest degree and build up the sum:
#   p(x)=((...((a_P x + a_{P-1}) x + a_{P-2}) x + ... ) x + a_1) x + a_0.
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
# horner_eval_col
#
# Evaluate a Taylor polynomial stored as a column of a coefficient matrix.
#
# Mathematical definition
#   If column j of A stores
#
#       a₀,a₁,...,a_P,
#
#   then this function evaluates
#
#       p(x)=Σ_{n=0}^P a_n x^n
#
#   using Horner accumulation:
#
#       p(x)=(((a_P x+a_{P-1})x+a_{P-2})x+⋯)x+a₀.
#
#   It avoids creating SubArray views
#   such as
#
#       @view A[:,j]
#
#   in hot evaluation paths.
#
# Inputs
#   A::Matrix{ComplexF64}
#       Coefficient matrix whose columns store Taylor coefficients.
#
#   j::Int
#       Column index.
#
#   x::Float64
#       Evaluation point.
#
# Output
#   ComplexF64
#       Polynomial value p(x).
# =============================================================================
@inline function horner_eval_col(A::Matrix{ComplexF64},j::Int,x::Float64)
    xx=ComplexF64(x,0.0)
    acc=0.0+0.0im
    @inbounds for i in size(A,1):-1:1
        acc=muladd(acc,xx,A[i,j])
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
# These follow from substituting the series into the ODE and matching powers.
# Explicitly ((n+1) comes from differentiation):
#   Σ_{n=0}^P (n+1)u_{n+1} δ^n = Σ_{n=0}^P y_n δ^n,
#   Σ_{n=0}^P (n+1)y_{n+1} δ^n = ν(ν+1)Σ_{n=0}^P u_n δ^n - (Σ_{m=0}^P c_m δ^m)(Σ_{l=0}^P y_l δ^l).
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
@inline function build_patch_coeffs!(u::Vector{ComplexF64},y::Vector{ComplexF64},nu::ComplexF64,u0::ComplexF64,y0::ComplexF64,coth::AbstractVector{Float64})
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
mutable struct QTaylorTable
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
# QTaylorPrecomp
#
# Precomputation data for building QTaylorTable objects.
#   Stores coth-series coefficients for each patch center.
# Fields
#   dmin    ::Float64         # lower d bound (>d_threshold)
#   dmax    ::Float64         # upper d bound
#   h       ::Float64         # uniform patch spacing
#   P       ::Int             # Taylor degree per patch
#   Npatch  ::Int             # number of patches
#   centers ::Vector{Float64} # d_i centers, length Npatch
#   coth_coeffs ::Matrix{Float64} #(P+1)×Npatch, coth-series coeffs per patch
# =============================================================================
struct QTaylorPrecomp
    dmin::Float64
    dmax::Float64
    h::Float64
    P::Int
    Npatch::Int
    centers::Vector{Float64}
    coth_coeffs::Matrix{Float64}  #(P+1)×Npatch
end

# =============================================================================
# QTaylorWorkspace
#
# Workspace buffers for building QTaylorTable objects.
# Fields
#   ucoef::Vector{ComplexF64}       length P+1
#   ycoef::Vector{ComplexF64}       length P+1
#   ucoef_tls::Vector{Vector{ComplexF64}}  length Npatch, each length P+1
#   ycoef_tls::Vector{Vector{ComplexF64}}  length Npatch, each length P+1
# =============================================================================
struct QTaylorWorkspace
    ucoef::Vector{ComplexF64}
    ycoef::Vector{ComplexF64}
    ucoef_tls::Vector{Vector{ComplexF64}}
    ycoef_tls::Vector{Vector{ComplexF64}}
end

# =============================================================================
# _patch_index
#
# Map a distance d to a Taylor patch index i.
# Uses uniform spacing h and snaps to the nearest integer index when d lies
# close to a patch boundary (to avoid off-by-one from floating error).
#
# Inputs
#   tab::QTaylorTable
#   d::Float64
#
# Output
#   i::Int in 1:length(tab.centers)
# =============================================================================
#=
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
=#
@inline function _patch_index(tab::QTaylorTable,d::Float64)
    @boundscheck tab.dmin <= d <= tab.dmax || error(
        "QTaylorTable distance d=$d outside [$(tab.dmin), $(tab.dmax)]"
    )
    t = (d - tab.dmin)/tab.h
    idx = Int(floor(t)) + 1
    abs(t-round(t)) < 64eps(t) && (idx = Int(round(t)) + 1)
    return clamp(idx,1,length(tab.centers))
end

# =============================================================================
# eval_Q
#
# Evaluate u(d)=Q_ν(cosh(d)) from the uniform Taylor-patch table.
#   - Choose patch index i such that d≈centers[i].
#   - Let x=d-centers[i] (local patch coordinate).
#   - Evaluate the stored Taylor polynomial u(d)=Σ u_n (x)^n.
#   - For d<d_threshold use the small-d expansion _small_z_Q.
#   - For small d the tables are unreliable so asymptotics are used instead.
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
    if dd<legendre_d_threshold()
        return _small_z_Q(k_from_ν(tab.nu),dd)
    end
    idx=_patch_index(tab,dd)
    return horner_eval_col(tab.ucoeffs,idx,dd-tab.centers[idx])
end

# =============================================================================
# eval_dQdd
#
# Evaluate y(d)=d/dd Q_ν(cosh(d)) from the uniform Taylor-patch table.
#   - Same patch selection and Horner evaluation as eval_Q, but using ycoeffs (more info there).
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
    if dd<legendre_d_threshold()
        return _small_z_dQ(k_from_ν(tab.nu),dd)
    end
    idx=_patch_index(tab,dd)
    return horner_eval_col(tab.ycoeffs,idx,dd-tab.centers[idx])
end

# =============================================================================
# eval_Q!
#
# Evaluation of u(d)=Q_ν(cosh(d)) for a vector of distances.
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
    @inbounds for i in eachindex(dvec)
        out[i]=_eval_Q(tab,dvec[i])
    end
    return nothing
end

# =============================================================================
# eval_dQdd!
#
# Evaluation of y(d)=d/dd Q_ν(cosh(d)) for a vector of distances.
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
    @inbounds for i in eachindex(dvec)
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
#   2) For each patch center d_i=dmin+(i-1)h_patch:
#        - Expand coth(d_i+δ) in δ to order P_patch (Float64 coeffs).
#        - Generate Taylor coefficients (u_n,y_n) using ODE recurrences.
#        - Store coefficients as columns of ucoeffs,ycoeffs.
#        - Advance the state (u0,y0) to the next center via Horner eval at δ=h_patch.
#
# Complexity (per k)
#   - Python: O(1) calls (seed only).
#   - Julia: O(Npatch*P_patch^2) arithmetic, no allocations in the inner loop.
#
# Inputs (keywords)
#   k::ComplexF64   - wavenumber (can have small Im part)
#   dmin,dmax::Float64 - domain in d, dmin>0
#   mp_dps::Int     - mpmath precision for seeding
#   leg_type::Int=3   - mpmath LegendreQ definition selector
#
# Output
#   QTaylorTable
# =============================================================================
function build_QTaylorTable(k::ComplexF64;dmin::Float64=1e-3,dmax::Float64=5.0,mp_dps::Int=60,leg_type::Int=3)
    h_patch,P_patch=legendre_Q_params()
    @assert dmax>dmin
    @assert h_patch>0
    @assert P_patch≥1
    nu=ν(k)
    sδ=Vector{Float64}(undef,P_patch+1) # sinh coeffs for each patch replaced in loop
    cδ=Vector{Float64}(undef,P_patch+1) # cosh coeffs for each patch replaced in loop
    series_sinh_cosh!(sδ,cδ) # base series for sinh(δ),cosh(δ), same order as P
    u0,y0=seed_u_y_mpmath(nu,dmin;dps=mp_dps,leg_type=leg_type) # seed at dmin (high-precision needs d>1e-3)
    Npatch=Int(ceil((dmax-dmin)/h_patch))  # number of patches linearly spaced
    centers=Vector{Float64}(undef,Npatch) # patch centers
    @inbounds for i in 1:Npatch
        centers[i]=dmin+(i-1)*h_patch # each patch center sits at dmin+(i-1)h_patch
    end
    ucoeffs=Matrix{ComplexF64}(undef,P_patch+1,Npatch) # for each patch in Npatch store P_patch+1 coeffs for u
    ycoeffs=Matrix{ComplexF64}(undef,P_patch+1,Npatch) # for each patch in Npatch store P_patch+1 coeffs for y
    coth=Vector{Float64}(undef,P_patch+1) # coth series per patch, replaced in loop
    sinh_series=Vector{Float64}(undef,P_patch+1) # scratch buffer for series division
    cosh_series=Vector{Float64}(undef,P_patch+1) # scratch buffer for series division
    ucoef=Vector{ComplexF64}(undef,P_patch+1) # scratch buffer for patch coeffs
    ycoef=Vector{ComplexF64}(undef,P_patch+1) # scratch buffer for patch coeffs
    sh0=sinh(dmin) # initial sinh/cosh at dmin
    ch0=cosh(dmin) 
    shh=sinh(h_patch) # sinh/cosh at patch step h (reused for every patch since step is uniform)
    chh=cosh(h_patch) # sinh/cosh at patch step h (reused for every patch since step is uniform)
    @inbounds for i in 1:Npatch
        coth_series_from_sinhcosh!(coth,sh0,ch0,sδ,cδ,sinh_series,cosh_series) # build coth series at patch center 
        build_patch_coeffs!(ucoef,ycoef,nu,u0,y0,coth) # build patch coeffs for u,y at this center
        @inbounds for n in 1:(P_patch+1) # store coeffs in the big matrices for each patch
            ucoeffs[n,i]=ucoef[n]
            ycoeffs[n,i]=ycoef[n]
        end
        if i<Npatch # advance u0,y0 to next patch center via Horner eval at δ=h 
            u0=horner_eval(ucoef,h_patch) # the new u0 which is at d_i+h
            y0=horner_eval(ycoef,h_patch) # the new y0 which is at d_i+h
            # also advance sinh/cosh to next patch center
            sh1=sh0*chh+ch0*shh
            ch1=ch0*chh+sh0*shh
            sh0,ch0=sh1,ch1 # update for next iteration
        end
    end
    return QTaylorTable(nu,dmin,dmax,h_patch,P_patch,centers,ucoeffs,ycoeffs)
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
#   mp_dps::Int                     - mpmath precision for seeding
#   leg_type::Int=3                 - mpmath LegendreQ definition selector
#   threaded::Bool=true             - build tables in parallel (Julia only)
#
# Output
#   Vector{QTaylorTable} length Nk, one table per ks[i]
# =============================================================================
function build_QTaylorTable(ks::AbstractVector{ComplexF64};dmin::Float64=1e-3,dmax::Float64=5.0,mp_dps::Int=60,leg_type::Int=3,threaded::Bool=true)
    h_patch,P_patch=legendre_Q_params()
    @assert dmax>dmin && h_patch>0 && P_patch≥1
    Nk=length(ks)
    Npatch=Int(ceil((dmax-dmin)/h_patch))
    centers=Vector{Float64}(undef,Npatch)
    @inbounds for i in 1:Npatch
        centers[i]=dmin+(i-1)*h_patch
    end
    sδ=Vector{Float64}(undef,P_patch+1);cδ=Vector{Float64}(undef,P_patch+1)
    series_sinh_cosh!(sδ,cδ)
    coth_coeffs=Matrix{Float64}(undef,P_patch+1,Npatch)
    sinh_series=Vector{Float64}(undef,P_patch+1);cosh_series=Vector{Float64}(undef,P_patch+1)
    coth=Vector{Float64}(undef,P_patch+1)
    sh0=sinh(dmin);ch0=cosh(dmin)
    shh=sinh(h_patch);chh=cosh(h_patch)
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
        nu=ν(ks[i])
        nus[i]=nu
        u0s[i],y0s[i]=seed_u_y_mpmath(nu,dmin;dps=mp_dps,leg_type=leg_type)
    end
    tabs=Vector{QTaylorTable}(undef,Nk)
    NT=threaded ? Threads.nthreads() : 1
    ucoef_tls=[Vector{ComplexF64}(undef,P_patch+1) for _ in 1:NT]
    ycoef_tls=[Vector{ComplexF64}(undef,P_patch+1) for _ in 1:NT]
    function build_one!(i,tid)
        nu=nus[i];u0=u0s[i];y0=y0s[i]
        ucoeffs=Matrix{ComplexF64}(undef,P_patch+1,Npatch)
        ycoeffs=Matrix{ComplexF64}(undef,P_patch+1,Npatch)
        ucoef=ucoef_tls[tid];ycoef=ycoef_tls[tid]
        @inbounds for p in 1:Npatch
            build_patch_coeffs!(ucoef,ycoef,nu,u0,y0,@view(coth_coeffs[:,p]))
            @views copyto!(ucoeffs[:,p],ucoef)
            @views copyto!(ycoeffs[:,p],ycoef)
            if p<Npatch
                u0=horner_eval(ucoef,h_patch)
                y0=horner_eval(ycoef,h_patch)
            end
        end
        tabs[i]=QTaylorTable(nu,dmin,dmax,h_patch,P_patch,centers,ucoeffs,ycoeffs)
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
# QTaylorWorkspace(P;threaded=true)
#
# Purpose
#   Allocate scratch buffers for Taylor-coefficient recurrences so that
#   build_QTaylorTable! performs no per-call allocations in the hot loop.
#
# Input
#   threaded::Bool  if true, allocate per-thread buffers for Threads.nthreads()
#
# Output
#   QTaylorWorkspace
# =============================================================================
@inline function QTaylorWorkspace(;threaded::Bool=true)
    NT=threaded ? Threads.nthreads() : 1
    h_patch,P_patch=legendre_Q_params()
    u=Vector{ComplexF64}(undef,P_patch+1)
    y=Vector{ComplexF64}(undef,P_patch+1)
    u_tls=[Vector{ComplexF64}(undef,P_patch+1) for _ in 1:NT]
    y_tls=[Vector{ComplexF64}(undef,P_patch+1) for _ in 1:NT]
    return QTaylorWorkspace(u,y,u_tls,y_tls)
end

# =============================================================================
# build_QTaylorPrecomp(;dmin,dmax)
#
# Purpose
#   Precompute the geometry-independent data (without Legendre Q data) for Taylor-patch propagation:
#     - patch centers d_p=dmin+(p-1)h_patch
#     - Taylor coefficients of coth(d_p+δ) for every patch p up to order P_patch
#
# Math
#   coth(d_p+δ)=cosh(d_p+δ)/sinh(d_p+δ), expanded in δ about 0.
#   The addition formulas reduce the problem to fixed base series for sinh/cosh.
#
# Input (keywords)
#   dmin,dmax::Float64  d-domain, with dmin>0
#
# Output
#   QTaylorPrecomp(dmin,dmax,h_patch,P_patch,Npatch,centers,coth_coeffs)
# =============================================================================
@inline function build_QTaylorPrecomp(;dmin::Float64=1e-3,dmax::Float64=5.0)
    h_patch,P_patch=legendre_Q_params()
    Npatch=Int(ceil((dmax-dmin)/h_patch))
    centers=Vector{Float64}(undef,Npatch)
    @inbounds for p in 1:Npatch
        centers[p]=dmin+(p-1)*h_patch
    end
    sδ=Vector{Float64}(undef,P_patch+1);cδ=Vector{Float64}(undef,P_patch+1)
    series_sinh_cosh!(sδ,cδ) 
    coth_coeffs=Matrix{Float64}(undef,P_patch+1,Npatch)
    sinh_series=Vector{Float64}(undef,P_patch+1);cosh_series=Vector{Float64}(undef,P_patch+1)
    coth=Vector{Float64}(undef,P_patch+1)
    sh0=sinh(dmin);ch0=cosh(dmin)
    shh=sinh(h_patch);chh=cosh(h_patch)
    @inbounds for p in 1:Npatch
        coth_series_from_sinhcosh!(coth,sh0,ch0,sδ,cδ,sinh_series,cosh_series)
        @views copyto!(coth_coeffs[:,p],coth)
        if p<Npatch
            sh1=sh0*chh+ch0*shh
            ch1=ch0*chh+sh0*shh
            sh0,ch0=sh1,ch1
        end
    end
    return QTaylorPrecomp(dmin,dmax,h_patch,P_patch,Npatch,centers,coth_coeffs)
end

# =============================================================================
# alloc_QTaylorTable(pre;k=0)
#
# Purpose
#   Allocate coefficient storage for one QTaylorTable compatible with a given
#   QTaylorPrecomp. The returned table shares pre.centers (no copies).
#
# Input
#   pre::QTaylorPrecomp
#   k::ComplexF64   dummy k used only to initialize nu=-1/2+i k
#
# Output
#   QTaylorTable with uninitialized ucoeffs/ycoeffs (filled by build_QTaylorTable!)
# =============================================================================
@inline function alloc_QTaylorTable(pre::QTaylorPrecomp;k::ComplexF64=0.0+0.0im)
    ucoeffs=Matrix{ComplexF64}(undef,pre.P+1,pre.Npatch)
    ycoeffs=Matrix{ComplexF64}(undef,pre.P+1,pre.Npatch)
    nu=ν(k)
    return QTaylorTable(nu,pre.dmin,pre.dmax,pre.h,pre.P,pre.centers,ucoeffs,ycoeffs)
end

# =============================================================================
# alloc_QTaylorTables(pre,Nk;k=0)
#
# Purpose
#   Allocate a vector of QTaylorTable objects sharing the same centers array
#   from QTaylorPrecomp, with independent coefficient matrices per table.
#
# Input
#   pre::QTaylorPrecomp
#   Nk::Int
#   k::ComplexF64 dummy initializer for nu
#
# Output
#   Vector{QTaylorTable} length Nk
# =============================================================================
@inline function alloc_QTaylorTables(pre::QTaylorPrecomp,Nk::Int;k::ComplexF64=0.0+0.0im)
    tabs=Vector{QTaylorTable}(undef,Nk)
    @inbounds for i in 1:Nk
        tabs[i]=alloc_QTaylorTable(pre;k=k)
    end
    return tabs
end

# =============================================================================
# build_QTaylorTable!(tab,pre,ws,k;mp_dps,leg_type)
#
# Purpose
#   In-place construction of Taylor-patch coefficients for fixed ν=-1/2+i k,
#   writing into an existing QTaylorTable (no coefficient-matrix allocations).
#
# Algorithm
#   - Seed (u(dmin),y(dmin)) once by mpmath.legenq (2 calls).
#   - For p=1..Npatch:
#       * use precomputed coth-series for patch p
#       * generate (u_n,y_n) via ODE recurrences
#       * store in tab.ucoeffs[:,p],tab.ycoeffs[:,p]
#       * propagate (u0,y0) to next center by Horner evaluation at δ=h
#
# Input
#   tab::QTaylorTable       coefficient buffers (must match pre)
#   pre::QTaylorPrecomp     shared centers and coth-series
#   ws::QTaylorWorkspace    scratch vectors
#   k::ComplexF64
#   mp_dps::Int             mpmath digits for seeding
#   leg_type::Int           mpmath selector for LegendreQ
#
# Output
#   nothing (mutates tab)
# =============================================================================
function build_QTaylorTable!(tab::QTaylorTable,pre::QTaylorPrecomp,ws::QTaylorWorkspace,k::ComplexF64;mp_dps::Int=60,leg_type::Int=3)
    @assert pre.P==tab.P
    @assert pre.Npatch==size(tab.ucoeffs,2)
    @assert pre.Npatch==size(tab.ycoeffs,2)
    @assert pre.Npatch==length(tab.centers)
    @assert pre.dmin==tab.dmin && pre.dmax==tab.dmax && pre.h==tab.h
    @assert pre.centers===tab.centers
    nu=ν(k)
    u0,y0=seed_u_y_mpmath(nu,pre.dmin;dps=mp_dps,leg_type=leg_type) # seed at dmin, same as regular build_QTaylorTable
    P=pre.P;Npatch=pre.Npatch;h=pre.h
    ucoef=ws.ucoef;ycoef=ws.ycoef;ucoeffs=tab.ucoeffs;ycoeffs=tab.ycoeffs;coth=pre.coth_coeffs
    @inbounds for p in 1:Npatch
        build_patch_coeffs!(ucoef,ycoef,nu,u0,y0,@view(coth[:,p])) # build patch coeffs for u,y at this center directly from QTaylorWorkspace and precomp
        @inbounds for n in 1:(P+1)
            ucoeffs[n,p]=ucoef[n]
            ycoeffs[n,p]=ycoef[n]
        end
        if p<Npatch
            u0=horner_eval(ucoef,h)
            y0=horner_eval(ycoef,h)
        end
    end
    tab.nu=nu
    return nothing
end

@inline function _patch_index_Q(tab,d::Float64)
    if d<=tab.dmin
        return 1
    elseif d>=tab.dmax
        return length(tab.centers)
    else
        t=(d-tab.dmin)/tab.h
        idx=Int(floor(t))+1
        abs(t-round(t))<64*eps(t) && (idx=Int(round(t))+1)
        return clamp(idx,1,length(tab.centers))
    end
end

# =============================================================================
# build_QTaylorTable!(tabs,pre,ws,ks;mp_dps,leg_type,threaded)
#
# Purpose
#   Batched in-place construction of many QTaylorTable objects sharing the same
#   QTaylorPrecomp. Seeding is serial (PyCall), coefficient propagation can be
#   threaded in Julia.
#
# Threading model
#   - Python seeding: serial to avoid PyCall contention.
#   - Table propagation: Threads.@threads over i=1..Nk with thread-local scratch
#     ws.ucoef_tls/ws.ycoef_tls.
#
# Input
#   tabs::Vector{QTaylorTable}
#   pre::QTaylorPrecomp
#   ws::QTaylorWorkspace
#   ks::AbstractVector{ComplexF64}
#   threaded::Bool
#
# Output
#   nothing (mutates tabs)
# =============================================================================
function build_QTaylorTable!(tabs::Vector{QTaylorTable},pre::QTaylorPrecomp,ws::QTaylorWorkspace,ks::AbstractVector{ComplexF64};mp_dps::Int=60,leg_type::Int=3,threaded::Bool=true)
    Nk=length(ks);@assert length(tabs)==Nk
    @inbounds for i in 1:Nk
        t=tabs[i]
        @assert pre.P==t.P
        @assert pre.Npatch==size(t.ucoeffs,2)==size(t.ycoeffs,2)==length(t.centers)
        @assert pre.dmin==t.dmin && pre.dmax==t.dmax && pre.h==t.h
        @assert pre.centers===t.centers
    end
    nus=Vector{ComplexF64}(undef,Nk);u0s=Vector{ComplexF64}(undef,Nk);y0s=Vector{ComplexF64}(undef,Nk)
    @inbounds for i in 1:Nk
        nu=ν(ks[i]);nus[i]=nu
        u0s[i],y0s[i]=seed_u_y_mpmath(nu,pre.dmin;dps=mp_dps,leg_type=leg_type)
    end
    P=pre.P;Npatch=pre.Npatch;h=pre.h
    if threaded && Threads.nthreads()>1
        Threads.@threads :static for i in 1:Nk
            tid=Threads.threadid();ucoef=ws.ucoef_tls[tid];ycoef=ws.ycoef_tls[tid]
            t=tabs[i];ucoeffs=t.ucoeffs;ycoeffs=t.ycoeffs;nu=nus[i];u0=u0s[i];y0=y0s[i]
            @inbounds for p in 1:Npatch
                build_patch_coeffs!(ucoef,ycoef,nu,u0,y0,@view(pre.coth_coeffs[:,p]))
                @inbounds for n in 1:(P+1)
                    ucoeffs[n,p]=ucoef[n];ycoeffs[n,p]=ycoef[n]
                end
                if p<Npatch
                    u0=horner_eval(ucoef,h)
                    y0=horner_eval(ycoef,h)
                end
            end
            t.nu=nu
        end
    else
        ucoef=ws.ucoef;ycoef=ws.ycoef
        @inbounds for i in 1:Nk
            t=tabs[i];ucoeffs=t.ucoeffs;ycoeffs=t.ycoeffs;nu=nus[i];u0=u0s[i];y0=y0s[i]
            @inbounds for p in 1:Npatch
                build_patch_coeffs!(ucoef,ycoef,nu,u0,y0,@view(pre.coth_coeffs[:,p]))
                @inbounds for n in 1:(P+1)
                    ucoeffs[n,p]=ucoef[n];ycoeffs[n,p]=ycoef[n]
                end
                if p<Npatch
                    u0=horner_eval(ucoef,h)
                    y0=horner_eval(ycoef,h)
                end
            end
            t.nu=nu
        end
    end
    return nothing
end

# =============================================================================
# PTaylorTable
#
# Purpose
#   Store fast, uniform-step Taylor-patch tables for the Legendre function
#       p(d)=P_ν(cosh(d))
#   and its radial derivative
#       yp(d)=d/dd P_ν(cosh(d))=p'(d),
#   where
#       ν=-1/2+i k.
#
# Motivation (hyperbolic Kress DLP split)
#   In the hyperbolic logarithmic Kress decomposition of the DLP kernel,
#   the singular logarithmic coefficient is proportional to
#
#       A(d)=-(1/(4π)) P_ν(cosh(d)).
#
#   Therefore, besides Q_ν(cosh(d)) (already handled by QTaylorTable),
#   we also require extremely fast evaluations of P_ν(cosh(d)) and
#   its radial derivative.
#
# Mathematical model
#   The Legendre function P_ν(z) satisfies the same differential equation
#   as Q_ν(z). After the substitution
#
#       z=cosh(d),
#
#   the radial system becomes
#
#       p'(d)=yp(d),
#       yp'(d)=ν(ν+1)p(d)-coth(d) yp(d).
#
# Since this is identical to the Q-system, we reuse the same Taylor-patch
# propagation machinery and only change the initial high-precision seed.
#
# Discretization strategy (Taylor patches)
#   Choose centers
#
#       d_i=dmin+(i-1)h
#
#   and represent locally
#
#       p(d_i+δ)=Σ_{n=0}^P p_n δ^n,
#       yp(d_i+δ)=Σ_{n=0}^P yp_n δ^n.
#
# Coefficients are generated recursively from the ODE using precomputed
# Taylor expansions of coth(d_i+δ).
#
# Fields
#   nu       ::ComplexF64          ν=-1/2+i k
#   dmin     ::Float64             lower d bound (>0)
#   dmax     ::Float64             upper d bound
#   h        ::Float64             uniform patch spacing
#   P        ::Int                 Taylor degree
#   centers  ::Vector{Float64}     patch centers
#   pcoeffs  ::Matrix{ComplexF64}  Taylor coeffs for P_ν(cosh(d))
#   dpcoeffs ::Matrix{ComplexF64}  Taylor coeffs for d/dd P_ν(cosh(d))
#
# Notes
#   - Only ONE high-precision seed at dmin is required.
#   - The same coth precomputation as QTaylorTable is reused.
# =============================================================================
mutable struct PTaylorTable
    nu::ComplexF64
    dmin::Float64
    dmax::Float64
    h::Float64
    P::Int
    centers::Vector{Float64}
    pcoeffs::Matrix{ComplexF64}
    dpcoeffs::Matrix{ComplexF64}
end

# =============================================================================
# seed_p_dp_mpmath
#
# Compute the high-precision seed values
#
#   p(d0)=P_ν(cosh(d0)),
#   yp(d0)=d/dd P_ν(cosh(d0)),
#
# using mpmath.legenp through PyCall.
#
# Derivative evaluation
#   Using the stable Legendre recurrence identity and the chain rule:
#
#       d/dd P_ν(cosh(d))
#       =
#       (ν+1)(P_{ν+1}(z)-z P_ν(z))/sinh(d),
#
#   where
#
#       z=cosh(d).
#
# Inputs
#   nu::ComplexF64
#       Legendre index (typically ν=-1/2+i k)
#
#   d0::Float64
#       Seed distance (must satisfy d0>0)
#
# Keywords
#   dps::Int=60
#       Decimal precision used by mpmath
#
# Outputs
#   (p0,yp0)::Tuple{ComplexF64,ComplexF64}
#
# Notes
#   Only a single seed point is required; all other values are propagated
#   internally via Taylor recurrence.
# =============================================================================
function seed_p_dp_mpmath(nu::ComplexF64,d0::Float64;dps::Int=60)
    lock(PYCALL_MPMATH_LOCK)
    try
        _mpctx[].dps=dps
        nup=_mpc[](real(nu),imag(nu))
        dp=_mpf[](d0)
        z=_mp_cosh[](dp)
        sh=_mp_sinh[](dp)
        P0=_mp_legenp[](nup,0,z)
        P1=_mp_legenp[](nup+1,0,z)
        yp=(nup+1)*(P1-z*P0)/sh
        p_re=pycall(_pyfloat[],Float64,P0.real)
        p_im=pycall(_pyfloat[],Float64,P0.imag)
        y_re=pycall(_pyfloat[],Float64,yp.real)
        y_im=pycall(_pyfloat[],Float64,yp.imag)
        return ComplexF64(p_re,p_im),ComplexF64(y_re,y_im)
    finally
        unlock(PYCALL_MPMATH_LOCK)
    end
end

# =============================================================================
# alloc_PTaylorTable
#
# Allocate coefficient storage for a PTaylorTable compatible with a given
# QTaylorPrecomp.
#
# Purpose
#   This mirrors alloc_QTaylorTable, but stores Legendre P-series instead
#   of Legendre Q-series.
#
# Inputs
#   pre::QTaylorPrecomp
#       Shared geometry-independent precomputation:
#         - patch centers
#         - coth Taylor expansions
#
# Keyword
#   k::ComplexF64=0+0im
#       Dummy initialization value used only to define ν=-1/2+i k.
#
# Output
#   PTaylorTable
#
# Notes
#   The returned table contains allocated coefficient matrices but no
#   populated data until build_PTaylorTable! is called.
# =============================================================================
@inline function alloc_PTaylorTable(pre::QTaylorPrecomp;k::ComplexF64=0.0+0.0im)
    pcoeffs=Matrix{ComplexF64}(undef,pre.P+1,pre.Npatch)
    dpcoeffs=Matrix{ComplexF64}(undef,pre.P+1,pre.Npatch)
    return PTaylorTable(ν(k),pre.dmin,pre.dmax,pre.h,pre.P,pre.centers,pcoeffs,dpcoeffs)
end

# =============================================================================
# PTaylorTable
#
# Purpose
#   Store fast Taylor-patch tables for
#
#       p(d)  = P_ν(cosh d),
#       dp(d) = d/dd P_ν(cosh d),
#
#   with ν = -1/2 + i k.
#
# Motivation
#   The hyperbolic Kress DLP split needs the logarithmic coefficient
#
#       A(d) = -P_ν(cosh d)/(4π),
#
#   and its radial derivative A'(d). These are supplied by PTaylorTable.
#
# Difference from QTaylorTable
#   Q_ν is singular at d=0, so QTaylorTable uses small-d asymptotics below
#   d_threshold. P_ν is regular at d=0, but to avoid relying on threshold
#   heuristics we seed P at an interior anchor distance and propagate the Taylor
#   patches both forward and backward.
#
# Mathematical model
#   P_ν(cosh d) satisfies the same radial ODE as Q_ν(cosh d):
#
#       p'  = dp,
#       dp' = ν(ν+1)p - coth(d)dp.
#
# Fields
#   nu       :: ComplexF64
#   dmin     :: Float64
#   dmax     :: Float64
#   h_patch  :: Float64
#   P_patch  :: Int
#   centers  :: Vector{Float64}
#   pcoeffs  :: Matrix{ComplexF64}
#   dpcoeffs :: Matrix{ComplexF64}
# =============================================================================
function build_PTaylorTable!(tab::PTaylorTable,pre::QTaylorPrecomp,ws::QTaylorWorkspace,k::ComplexF64;mp_dps::Int=80,anchor_d::Float64=clamp(0.05,pre.dmin,pre.dmax))
    nu=ν(k)
    P=pre.P
    Npatch=pre.Npatch
    h=pre.h
    j0=clamp(Int(round((anchor_d-pre.dmin)/h))+1,1,Npatch)
    d0=pre.centers[j0]
    p0,dp0=seed_p_dp_mpmath(nu,d0;dps=mp_dps)
    pcoef=ws.ucoef
    dpcoef=ws.ycoef
    build_patch_coeffs!(pcoef,dpcoef,nu,p0,dp0,@view(pre.coth_coeffs[:,j0]))
    @inbounds for n in 1:(P+1)
        tab.pcoeffs[n,j0]=pcoef[n]
        tab.dpcoeffs[n,j0]=dpcoef[n]
    end
    pL=p0
    dpL=dp0
    @inbounds for q in (j0-1):-1:1
        pL=horner_eval(@view(tab.pcoeffs[:,q+1]),-h)
        dpL=horner_eval(@view(tab.dpcoeffs[:,q+1]),-h)
        build_patch_coeffs!(pcoef,dpcoef,nu,pL,dpL,@view(pre.coth_coeffs[:,q]))
        for n in 1:(P+1)
            tab.pcoeffs[n,q]=pcoef[n]
            tab.dpcoeffs[n,q]=dpcoef[n]
        end
    end
    pR=p0
    dpR=dp0
    @inbounds for q in (j0+1):Npatch
        pR=horner_eval(@view(tab.pcoeffs[:,q-1]),h)
        dpR=horner_eval(@view(tab.dpcoeffs[:,q-1]),h)
        build_patch_coeffs!(pcoef,dpcoef,nu,pR,dpR,@view(pre.coth_coeffs[:,q]))
        for n in 1:(P+1)
            tab.pcoeffs[n,q]=pcoef[n]
            tab.dpcoeffs[n,q]=dpcoef[n]
        end
    end
    tab.nu=nu
    return nothing
end

# =============================================================================
# _eval_Pleg
#
# Evaluate
#
#   P_ν(cosh(d))
#
# from the Taylor-patch table.
#
# Algorithm
#   - locate nearest Taylor patch
#   - form local coordinate
#
#       δ=d-d_i
#
#   - evaluate stored Taylor polynomial by Horner accumulation
#
# Inputs
#   tab::PTaylorTable
#   d::Float64
#
# Output
#   ComplexF64
# =============================================================================
@inline function _eval_Pleg(tab::PTaylorTable,d::Float64)
    dd=Float64(d)
    if dd<legendre_d_threshold()
        return _small_z_P(k_from_ν(tab.nu),dd)
    end
    @boundscheck tab.dmin<=dd<=tab.dmax || error("P table d=$dd outside [$(tab.dmin), $(tab.dmax)]")
    idx=_patch_index_Q(tab,dd)
    return horner_eval_col(tab.pcoeffs,idx,dd-tab.centers[idx])
end

# =============================================================================
# _eval_dPlegdd
#
# Evaluate
#
#   d/dd P_ν(cosh(d))
#
# from the Taylor-patch table.
#
# Inputs
#   tab::PTaylorTable
#   d::Float64
#
# Output
#   ComplexF64
# =============================================================================
@inline function _eval_dPlegdd(tab::PTaylorTable,d::Float64)
    dd=Float64(d)
    if dd<legendre_d_threshold()
        return _small_z_dP(k_from_ν(tab.nu),dd)
    end
    @boundscheck tab.dmin<=dd<=tab.dmax || error("P table d=$dd outside [$(tab.dmin), $(tab.dmax)]")
    idx=_patch_index_Q(tab,dd)
    return horner_eval_col(tab.dpcoeffs,idx,dd-tab.centers[idx])
end

# =============================================================================
# hyperbolic_Alog
#
# Logarithmic coefficient for the hyperbolic Kress DLP split.
#
# Mathematical definition
#
#   A(d)=-(1/(4π)) P_ν(cosh(d)).
#
# This is the coefficient multiplying the explicit logarithmic singularity
# in the hyperbolic double-layer decomposition.
#
# Inputs
#   ptab::PTaylorTable
#   d::Float64
#
# Output
#   ComplexF64
# =============================================================================
@inline hyperbolic_Alog(ptab::PTaylorTable,d::Float64)=-_eval_Pleg(ptab,d)*inv4π
# =============================================================================
# hyperbolic_Alog_d
#
# Radial derivative of the logarithmic coefficient
#
#   A'(d)=-(1/(4π)) d/dd P_ν(cosh(d)).
#
# Needed in the Kress split regular remainder construction.
#
# Inputs
#   ptab::PTaylorTable
#   d::Float64
#
# Output
#   ComplexF64
# =============================================================================
@inline hyperbolic_Alog_d(ptab::PTaylorTable,d::Float64)=-_eval_dPlegdd(ptab,d)*inv4π

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
# This function returns ∂_{n'_E}d, the derivative of d w.r.t. the *Euclidean*
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
        c=max(c,1.0)
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
    return inv2π*_eval_Q(tab,acosh(1.0+2.0*((xp1-x1)^2+(xp2-x2)^2)/((1.0-(x1^2+x2^2))*(1.0-(xp1^2+xp2^2)))))
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
    c=max(c,1.0)
    d=acosh(c)
    y=_eval_dQdd(tab,d)
    dn=_∂n_d(x1,x2,xp1,xp2,n1,n2)
    return (y*dn)*inv2π
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
    @inbounds for i in eachindex(out)
        out[i]*=(dnvec[i]*inv2π)
    end
    return nothing
end





############################
######### TESTING ##########
############################

if abspath(PROGRAM_FILE)==@__FILE__

function Q_ref_mpmath(nu::ComplexF64,d::Float64;dps::Int=80,leg_type::Int=3)
    _mpctx[].dps=dps
    nup=_mpc[](real(nu),imag(nu))
    dp=_mpf[](d)
    z=_mp_cosh[](dp)
    Q=_mp_legenq[](nup,0,z;type=leg_type)
    return ComplexF64(pycall(_pyfloat[],Float64,Q.real),pycall(_pyfloat[],Float64,Q.imag))
end

function dQdd_ref_mpmath(nu::ComplexF64,d::Float64;dps::Int=80,leg_type::Int=3)
    _mpctx[].dps=dps
    nup=_mpc[](real(nu),imag(nu))
    dp=_mpf[](d)
    z=_mp_cosh[](dp)
    sh=_mp_sinh[](dp)
    Q0=_mp_legenq[](nup,0,z;type=leg_type)
    Q1=_mp_legenq[](nup+1,0,z;type=leg_type)
    y=(nup+1)*(Q1-z*Q0)/sh
    return ComplexF64(pycall(_pyfloat[],Float64,y.real),pycall(_pyfloat[],Float64,y.imag))
end

function run_QTaylorTable_test(;k=ComplexF64(120.0,0.2),dmin=1e-3,dmax=8.0,mp_dps=90,leg_type=3,Nbench=1_000_000_00)
    pre=build_QTaylorPrecomp(;dmin=dmin,dmax=dmax)
    ws=QTaylorWorkspace(;threaded=false)
    tab=alloc_QTaylorTable(pre;k=k)
    println("\nBuild:")
    @time build_QTaylorTable!(tab,pre,ws,k;mp_dps=mp_dps,leg_type=leg_type)
    nu=ν(k)
    dtests=sort(unique([1e-8,1e-6,1e-4,5e-4,1e-3,1.5e-3,3e-3,1e-2,3e-2,0.1,0.3,0.75,1.5,3.0,5.0,0.25*dmax,0.5*dmax,0.75*dmax,dmax]))
    maxrelQ=0.0
    maxreldQ=0.0
    println("\nAccuracy:")
    for d in dtests
        Qr=Q_ref_mpmath(nu,d;dps=mp_dps,leg_type=leg_type)
        dQr=dQdd_ref_mpmath(nu,d;dps=mp_dps,leg_type=leg_type)
        Qv=_eval_Q(tab,d)
        dQv=_eval_dQdd(tab,d)
        errQ=abs(Qv-Qr)
        reQ=errQ/max(abs(Qr),eps(Float64))
        errdQ=abs(dQv-dQr)
        redQ=errdQ/max(abs(dQr),eps(Float64))
        maxrelQ=max(maxrelQ,reQ)
        maxreldQ=max(maxreldQ,redQ)
        println("d = ",d)
        println("  Q abs err    = ",errQ,"  rel err = ",reQ)
        println("  dQ abs err   = ",errdQ,"  rel err = ",redQ)
        @test reQ<5e-10 || errQ<5e-12
        @test redQ<5e-9 || errdQ<5e-9
    end
    rng=MersenneTwister(1234)
    dvec=exp.(log(dmin).+(log(dmax)-log(dmin)).*rand(rng,Nbench))
    out=Vector{ComplexF64}(undef,Nbench)
    println("\nVector smoke tests:")
    _eval_Q!(out,tab,dvec)
    @test all(isfinite,real.(out))
    @test all(isfinite,imag.(out))
    _eval_dQdd!(out,tab,dvec)
    @test all(isfinite,real.(out))
    @test all(isfinite,imag.(out))
    println("\nBenchmark scalar _eval_Q:")
    @btime _eval_Q($tab,1.234)
    println("\nBenchmark scalar _eval_dQdd:")
    @btime _eval_dQdd($tab,1.234)
    println("\nBenchmark vector _eval_Q! : $(Nbench) evals")
    @btime _eval_Q!($out,$tab,$dvec)
    println("\nBenchmark vector _eval_dQdd! : $(Nbench) evals")
    @btime _eval_dQdd!($out,$tab,$dvec)
    println("\nmax rel Q    = ",maxrelQ)
    println("max rel dQdd = ",maxreldQ)
    return tab
end

@testset "QTaylorTable Legendre Q" begin
    tab=run_QTaylorTable_test()
end

end