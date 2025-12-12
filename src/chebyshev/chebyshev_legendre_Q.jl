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
d_threshold=1e-3 # this is for the dQ evaluation which has worse accuracy close to d=0

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

##################################################################################
# Small-d expansion for d/dd Q_ν(cosh d), ν = -1/2 + i k obtained from Mathematica up to d^27:
#
# NOTE:
#   - This returns d/dd Q_ν(cosh d), WITHOUT the (1/(2π)) factor.
#   - Use only for very small d (e.g. d ≲ 1e-3) inside the Z_threshold branch chosen such that accuracy ≈ 1e-14.
##################################################################################
@inline function _small_z_dQ(k::C,d::T)::C where{C<:Complex,T<:Real};nu=-T(0.5)+im*k;Hnu=Base.MathConstants.eulergamma+SpecialFunctions.digamma(nu+one(nu))
    return C(-(one(T)/d)+(d*(2+3*nu*(1+nu)))/12+(d^3*(-56+15*nu*(1+nu)*(-2+15*nu*(1+nu))))/2880+(d^5*(248+21*nu*(1+nu)*(8+5*nu*(1+nu)*(-7+5*nu*(1+nu)))))/120960+(d^7*(-48768+5*nu*(1+nu)*(-7568+21*nu*(1+nu)*(1268+5*nu*(1+nu)*(-156+47*nu*(1+nu))))))/2.322432e8+(d^9*(1308160+11*nu*(1+nu)*(98432+3*nu*(1+nu)*(-104864+7*nu*(1+nu)*(8284+nu*(1+nu)*(-2320+393*nu*(1+nu)))))))/6.13122048e10+(d^11*(-724212224+273*nu*(1+nu)*(-2267520+11*nu*(1+nu)*(628720+nu*(1+nu)*(-327944+21*nu*(1+nu)*(3968+nu*(1+nu)*(-650+71*nu*(1+nu))))))))/3.34764638208e14+(d^13*(3522785280+nu*(1+nu)*(3064070144+429*nu*(1+nu)*(-21245568+nu*(1+nu)*(10734208+nu*(1+nu)*(-2570552+3*nu*(1+nu)*(127876+nu*(1+nu)*(-13846+1059*nu*(1+nu)))))))))/1.6068702633984e16+(d^15*(-81555718766592+17*nu*(1+nu)*(-4215787018240+3*nu*(1+nu)*(4119262721280+143*nu*(1+nu)*(-14288101248+nu*(1+nu)*(3308766736+3*nu*(1+nu)*(-154931200+nu*(1+nu)*(15324344+45*nu*(1+nu)*(-26264+1487*nu*(1+nu))))))))))/3.6713771778126643e21+(d^17*(7536235717591040+57*nu*(1+nu)*(116947251986432+17*nu*(1+nu)*(-19979202338816+nu*(1+nu)*(9796767888640+13*nu*(1+nu)*(-171086931456+11*nu*(1+nu)*(2109795216+nu*(1+nu)*(-196236224+nu*(1+nu)*(13852776+5*nu*(1+nu)*(-160176+6989*nu*(1+nu)))))))))))/3.34829598616515e24+(d^19*(-3023786723765649408+55*nu*(1+nu)*(-48844705828831232+57*nu*(1+nu)*(2473269691121664+17*nu*(1+nu)*(-70796702697984+nu*(1+nu)*(15871802558208+13*nu*(1+nu)*(-162189398880+11*nu*(1+nu)*(1322981232+5*nu*(1+nu)*(-17552464+nu*(1+nu)*(930264+nu*(1+nu)*(-41850+1451*nu*(1+nu))))))))))))/1.3259252105213994e28+(d^21*(5919143148921500467200+23*nu*(1+nu)*(229361823479855316992+15*nu*(1+nu)*(-43938582105369018368+19*nu*(1+nu)*(1119337009242789888+17*nu*(1+nu)*(-14636176930816512+nu*(1+nu)*(1918454135937408+13*nu*(1+nu)*(-12957794542560+nu*(1+nu)*(828616686128+nu*(1+nu)*(-41247566448+7*nu*(1+nu)*(243316128+nu*(1+nu)*(-8764690+247353*nu*(1+nu)))))))))))))/2.5616875067273434e32+(d^23*(-686096493620974804008960+13*nu*(1+nu)*(-47144867132796858793984+23*nu*(1+nu)*(5871102997047666982912+5*nu*(1+nu)*(-566138921659181092864+57*nu*(1+nu)*(2194468161804355584+17*nu*(1+nu)*(-16765943320964352+nu*(1+nu)*(1451742074957120+13*nu*(1+nu)*(-6984481999040+nu*(1+nu)*(334973138896+nu*(1+nu)*(-12986455152+7*nu*(1+nu)*(61403188+nu*(1+nu)*(-1811612+42433*nu*(1+nu))))))))))))))/2.930570507696081e35+(d^25*(33367733728285762089123840+nu*(1+nu)*(29859320862789776542007296+5*nu*(1+nu)*(-17063269414016842458202112+23*nu*(1+nu)*(356639815691069055434752+nu*(1+nu)*(-78443654495831951527936+57*nu*(1+nu)*(177569554987496819712+17*nu*(1+nu)*(-895795952130801408+nu*(1+nu)*(55227923257381696+nu*(1+nu)*(-2589559892863424+nu*(1+nu)*(96684098805008+nu*(1+nu)*(-3003546729040+7*nu*(1+nu)*(11641920004+5*nu*(1+nu)*(-57309356+1132133*nu*(1+nu)))))))))))))))/1.4066738436941188e38+(d^27*(-91770018091370053888741736448+29*nu*(1+nu)*(-2835615745800674524836921344+3*nu*(1+nu)*(2695606115502739701208973312+5*nu*(1+nu)*(-258579031590302542450655232+23*nu*(1+nu)*(2464338533704759823015936+nu*(1+nu)*(-316405030616016052463616+19*nu*(1+nu)*(1418259159879823898112+17*nu*(1+nu)*(-5092635130610819200+nu*(1+nu)*(235309612303988928+nu*(1+nu)*(-8586706803750816+nu*(1+nu)*(256808887011248+3*nu*(1+nu)*(-2179494838664+nu*(1+nu)*(49370502636+5*nu*(1+nu)*(-205868026+3476589*nu*(1+nu))))))))))))))))/3.818275481323316e42-(d*nu*(1+nu)*(182684914765469984565271461888000000+7611871448561249356886310912000000*d^2*(-2+3*nu*(1+nu))+190296786214031233922157772800000*d^4*(8+5*(-1+nu)*nu*(1+nu)*(2+nu))+566359482779854862863564800000*d^6*(-272+7*nu*(1+nu)*(44+5*nu*(1+nu)*(-4+nu+nu^2)))+3933051963748992103219200000*d^8*(3968+3*nu*(1+nu)*(-1424+7*nu*(1+nu)*(84+nu*(1+nu)*(-20+3*nu*(1+nu)))))+8938754463065891143680000*d^10*(-176896+11*nu*(1+nu)*(16864+nu*(1+nu)*(-6584+21*nu*(1+nu)*(68+nu*(1+nu)*(-10+nu+nu^2)))))+28649854048288112640000*d^12*(5592064+13*nu*(1+nu)*(-444544+11*nu*(1+nu)*(15296+nu*(1+nu)*(-3128+3*nu*(1+nu)*(140+nu*(1+nu)*(-14+nu+nu^2))))))+8526742276276224000*d^14*(-1903757312+3*nu*(1+nu)*(650078976+13*nu*(1+nu)*(-18589056+11*nu*(1+nu)*(334192+nu*(1+nu)*(-42240+nu*(1+nu)*(3864+5*nu*(1+nu)*(-56+3*nu*(1+nu))))))))+15674158596096000*d^16*(104932671488+17*nu*(1+nu)*(-6287587328+nu*(1+nu)*(2311237888+13*nu*(1+nu)*(-34468224+11*nu*(1+nu)*(382416+nu*(1+nu)*(-32896+nu*(1+nu)*(2184+5*nu*(1+nu)*(-24+nu+nu^2))))))))+11457718272000*d^18*(-14544442556416+19*nu*(1+nu)*(776768475136+17*nu*(1+nu)*(-16670893568+nu*(1+nu)*(3191642624+13*nu*(1+nu)*(-29341344+11*nu*(1+nu)*(221328+nu*(1+nu)*(-13808+nu*(1+nu)*(696+(-5+nu)*nu*(1+nu)*(6+nu)))))))))+13640140800*d^20*(1237874513281024+nu*(1+nu)*(-1252648497168384+19*nu*(1+nu)*(23928414824448+17*nu*(1+nu)*(-267191996928+nu*(1+nu)*(31506758784+13*nu*(1+nu)*(-196781088+nu*(1+nu)*(11833360+nu*(1+nu)*(-560208+7*nu*(1+nu)*(3168+nu*(1+nu)*(-110+3*nu*(1+nu)))))))))))+3369600*d^22*(-507711943253426176+23*nu*(1+nu)*(22292423254048768+nu*(1+nu)*(-8059883338280960+19*nu*(1+nu)*(80038242321408+17*nu*(1+nu)*(-550103120640+nu*(1+nu)*(44044491328+13*nu*(1+nu)*(-199258304+nu*(1+nu)*(9087760+nu*(1+nu)*(-337744+7*nu*(1+nu)*(1540+nu*(1+nu)*(-44+nu+nu^2)))))))))))+2808*d^24*(61730370047551995904+5*nu*(1+nu)*(-12448661250518024192+23*nu*(1+nu)*(195122177890779136+nu*(1+nu)*(-36649899662381056+19*nu*(1+nu)*(223895553555456+17*nu*(1+nu)*(-1044390696192+nu*(1+nu)*(60545295936+nu*(1+nu)*(-2699564608+nu*(1+nu)*(96615376+nu*(1+nu)*(-2894320+7*nu*(1+nu)*(10868+5*nu*(1+nu)*(-52+nu+nu^2))))))))))))+d^26*(-17562900400985989971968+3*nu*(1+nu)*(5895807142545951555584+5*nu*(1+nu)*(-424139144014000029696+23*nu*(1+nu)*(3451830441237979136+nu*(1+nu)*(-398682053674530816+19*nu*(1+nu)*(1652375735883264+17*nu*(1+nu)*(-5578990587776+nu*(1+nu)*(245125459008+nu*(1+nu)*(-8573995872+nu*(1+nu)*(247266448+3*nu*(1+nu)*(-2032888+nu*(1+nu)*(44772+5*nu*(1+nu)*(-182+3*nu*(1+nu)))))))))))))))*(Hnu+log(d/2)))/3.6536982953093996e35)
end

@inline function _small_z_dQ(k::T,d::T) where {T<:Real}
    return _small_z_dQ(complex(k,0.0),d)
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

##################################################################################
# legendre_dQ_dd(plan, ν, d):
#   Evaluate d/dd Q_ν(cosh d) using the same fixed Gauss–Legendre rule used
#   for Q_ν, but with the *differentiated integrand* (Mathematica-verified):
#
#       d/dd Q_ν(cosh d)
#       = - (ν+1) ∫₀^T (sinh d + cosh d cosh t)
#                        (cosh d + sinh d cosh t)^(-(ν+2)) dt
#
#   where T = plan.rule.Ttrunc and the integral on [0,T] is approximated by
#   the GL rule on [-1,1] via t = (T/2)(x+1).
#
# Inputs
#   plan :: QInfPlan     # quadrature plan (same as for legendre_Q)
#   ν    :: Complex      # index ν (typically ν = -1/2 + i k)
#   d    :: Real         # distance d>0, with z = cosh(d)
#
# Output
#   Complex              # d/dd Q_ν(cosh d)
#
# Notes
#   - For now we do not switch to _small_z_dQ; we will plug that in later
#     for very small d (e.g. d < 1e-3).
##################################################################################
@inline function legendre_dQ_dd(plan::QInfPlan,ν::C,d::T)::C where {C<:Complex,T<:Real}
    d<d_threshold && return _small_z_dQ(imag(ν+0.5),d)
    x=plan.rule.x
    w=plan.rule.w
    Tt=plan.rule.Ttrunc
    jac=Tt/2
    sh,ch=sinhcosh(d)
    νp1=ν+one(C)
    νp2=ν+2  # ν + 2
    s=zero(C)
    cacc=zero(C)
    @inbounds for i in eachindex(x)
        t=(Tt/2)*(x[i]+1)
        ct=cosh(t)
        base=ch+sh*ct
        num=sh+ch*ct
        integrand=-(νp1)*num*base^(-νp2)
        y=w[i]*integrand-cacc # Kahan summation
        tmp=s+y
        cacc=(tmp-s)-y
        s=tmp
    end
    return jac*s
end

@inline function legendre_dQ_dd(plan::QInfPlan,k::Ck,d::T) where {Ck<:Complex,T<:Real}
    return legendre_dQ_dd(plan,ν(k),d)
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

##################################################################################
# _build_table_dQ!(Qplan, ν, a, b; M):
#   Construct a Chebyshev table for
#       F(d) = d/dd Q_ν(cosh d)
#   on the panel [a,b], using Chebyshev–Lobatto nodes t_j = cos(π j / M).
#
# Inputs
#   Qplan  :: QInfPlan          # Gauss–Legendre plan for Q_ν / dQ/dd
#   ν      :: Complex           # ν = -1/2 + i k
#   a,b    :: Float64           # panel endpoints, 0 < a < b
#   M      :: Int               # Chebyshev degree
#
# Output
#   ChebQTable(a,b,M,c)         # with c the Chebyshev coefficients of dQ/dd
##################################################################################
function _build_table_dQ!(Qplan::QInfPlan,ν::Complex,a::T,b::T;M::Int=300)::ChebQTable{T} where {T<:Real}
    @assert a>0 && b>a "invalid panel [a,b] = [$a,$b]"
    f=Vector{Complex{T}}(undef,M+1)
    @inbounds for j in 0:M
        t=cospi(j/M) # Chebyshev node in [-1,1]
        d=((b+a)+(b-a)*t)/2 # affine map to [a,b]
        f[j+1]=legendre_dQ_dd(Qplan,ν,d)
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

##################################################################################
# plan_dQ_cheb(Qplan, ν, dmin, dmax; npanels, M, grading, geo_ratio):
#   Build a piecewise-Chebyshev plan for
#       F(d) = d/dd Q_ν(cosh d)
#   on [dmin,dmax].
#
# Inputs
#   Qplan    :: QInfPlan            # GL quadrature plan
#   ν        :: Complex             # ν = -1/2 + i k
#   dmin     :: T                   # lower bound, dmin>0
#   dmax     :: T                   # upper bound, dmax>dmin
#   npanels  :: Int=2500            # number of panels
#   M        :: Int=300             # Chebyshev degree per panel
#   grading  :: Symbol=:geometric   # :uniform or :geometric
#   geo_ratio:: Real=1.03           # geometric panel-size ratio
#
# Output
#   ChebQPlan(panels,dmin,dmax)     # tables for d/dd Q_ν(cosh d)
##################################################################################
function plan_dQ_cheb(Qplan::QInfPlan,ν::Complex{T},dmin::T,dmax::T;npanels::Int=2500,M::Int=300,grading::Symbol=:geometric,geo_ratio::Real=1.03)::ChebQPlan{T} where {T<:Real}
    @assert dmin>0 && dmax>dmin
    br=grading===:uniform ? _breaks_uniform(dmin,dmax,npanels) : _breaks_geometric(dmin,dmax,npanels;ratio=geo_ratio)
    panels=Vector{ChebQTable{T}}(undef,npanels)
    @inbounds Threads.@threads for i in 1:npanels
        panels[i]=_build_table_dQ!(Qplan,ν,br[i],br[i+1];M=M)
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

# ===============================================================================
#   Scalar evaluation of
#       F(d) = d/dd Q_ν(cosh d)
#   from a ChebQPlan built by plan_dQ_cheb / build_dQ_cheb_from_GL.
#
# Inputs
#   plan :: ChebQPlan{T}
#   d    :: T
#
# Output
#   Complex{T}  value of d/dd Q_ν(cosh d) at d
# ===============================================================================
@inline function eval_dQ(plan::ChebQPlan{T},d::T)::Complex{T} where {T<:Real}
    p=_find_panel_Q(plan.panels,d)
    P=plan.panels[p]
    t=(2*d-(P.b+P.a))/(P.b-P.a)
    return _cheb_clenshaw(P.c,t)
end

# ===============================================================================
# eval_dQ!(out, plan, dvec):
#   Vectorized/threaded evaluation of
#       F(d_i) = d/dd Q_ν(cosh d_i)
#   for all elements of dvec.
#
# Inputs
#   out   :: AbstractVector{Complex{T}}   (preallocated)
#   plan  :: ChebQPlan{T}
#   dvec  :: AbstractVector{T}
#
# Output
#   out[i] = d/dd Q_ν(cosh(dvec[i]))  in place
# ===============================================================================
function eval_dQ!(out::AbstractVector{Complex{T}},plan::ChebQPlan{T},dvec::AbstractVector{T}) where {T<:Real}
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