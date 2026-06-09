#############################################################################
# Gravity Green function Φ(a,b): asymptotic / local-Hankel evaluator
#
# We evaluate the oscillatory integral
#
#     Φ(a,b) = (1/4π) ∫ exp(iψ(t)) dt/t,
#     ψ(t) = a/t + b t - t^3/12.
#
# The high-frequency formula is split into two analytically distinct pieces:
#
#   1. Local Hankel part:
#
#        Φ_loc(a,b)
#        =
#        Σ_{m=0}^{Mloc}
#        1/(m! 12^m)
#        ∂_b^{3m} [ (i/4) H_0^(1)(2√(ab)) ].
#
#   2. Large-saddle part:
#
#        Φ_sad(a,b)
#        ~
#        (1/4π) exp(iψ(t_+))
#        sqrt(2π i / ψ''(t_+))
#        Σ_n p_n μ_n.
#
# The final outgoing evaluator is
#
#        Φ ≈ Φ_loc + S Φ_sad,
#
# with a lower-half-plane suppression rule for exponentially exploding saddles.
#
# In billiard use:
#
#        a = |x-y|^2 / 4,
#        b = E + geometric/linear shift.
#
# MO 4/6/26
#############################################################################

const INV4PI=1/(4π)

@inline gravity_phase(t,a,b)=a/t+b*t-t^3/12
@inline gravity_h0(z)=SpecialFunctions.besselh(0,1,z)
@inline gravity_h1(z)=SpecialFunctions.besselh(1,1,z)

# Saddle points of ψ(t).
#
# They solve
#
#     ψ'(t) = -a/t^2 + b - t^2/4 = 0,
#
# hence
#
#     t^2 = 2(b ± √(b^2-a)).
#
# We return both saddles in logarithmic coordinates s=log(t), together with the
# corresponding t-values. The pair is ordered by real(log(t)), so the smaller
# saddle appears first and the larger saddle second.
@inline function gravity_saddles(a::Float64,b::ComplexF64)
    root=sqrt(b*b-a+0im)
    tm=sqrt(2*(b-root)+0im)
    tp=sqrt(2*(b+root)+0im)
    real(tm)<0 && (tm=-tm)
    real(tp)<0 && (tp=-tp)
    sm=log(tm)
    sp=log(tp)
    real(sm)>real(sp) && ((sm,sp)=(sp,sm);(tm,tp)=(tp,tm))
    return sm,sp,tm,tp
end

# Derivatives ψ^(r)(t).
#
# For
#
#     ψ(t)=a/t+bt-t^3/12,
#
# the derivative structure is simple:
#
#     ψ'(t)  = -a/t^2 + b - t^2/4,
#     ψ''(t) =  2a/t^3 - t/2,
#     ψ'''(t)= -6a/t^4 - 1/2,
#
# and for r≥4 only the a/t term contributes:
#
#     d^r/dt^r (a/t) = (-1)^r r! a/t^(r+1).
@inline function gravity_phase_derivative(r::Int,t,a,b)
    v=(-1)^r*factorial(r)*a/t^(r+1)
    r==1 && (v+=b-t^2/4)
    r==2 && (v+=-t/2)
    r==3 && (v+=-1/2)
    return v
end

# Taylor coefficient of 1/(t+u).
#
# Around a saddle t, we use
#
#     1/(t+u) = Σ_{m≥0} (-1)^m u^m / t^(m+1).
@inline gravity_inverse_taylor_coef(m::Int,t)=(-1)^m/t^(m+1)

# Odd double factorial.
#
# Needed for Gaussian moments:
#
#     (2m-1)!! = 1·3·5···(2m-1).
@inline function gravity_doublefactorial_odd(n::Int)
    n<=0 && return 1.0
    p=1.0
    for k in 1:2:n
        p*=k
    end
    return p
end

# Gaussian moments for the saddle expansion.
#
# After extracting the quadratic phase
#
#     exp(i ψ'' u^2/2),
#
# the even moments are
#
#     μ_{2m} = (2m-1)!! (i/ψ'')^m,
#
# while odd moments vanish by symmetry.
@inline function gravity_gaussian_moment(n::Int,ψ2)
    isodd(n) && return 0im
    m=n÷2
    return gravity_doublefactorial_odd(2m-1)*(im/ψ2)^m
end

# Exponential Bell-type coefficients.
#
# If
#
#     exp(q_1 u + q_2 u^2 + ... ) = Σ e_n u^n,
#
# then
#
#     n e_n = Σ_{k=1}^n k q_k e_{n-k}.
#
# In the saddle expansion q_1=q_2=0 because the linear term vanishes at the
# saddle and the quadratic term is kept inside the Gaussian.
function gravity_exp_poly_coeff(q::Vector{ComplexF64})
    N=length(q)-1
    e=zeros(ComplexF64,N+1)
    e[1]=1
    @inbounds for n in 1:N
        s=0im
        for k in 1:n
            s+=k*q[k+1]*e[n-k+1]
        end
        e[n+1]=s/n
    end
    return e
end

# Large-saddle steepest-descent expansion.
#
# Around t=t_+, write
#
#     ψ(t_+ + u)
#       =
#       ψ(t_+) + ψ''(t_+)u^2/2
#       + ψ'''(t_+)u^3/6 + ...
#
# and
#
#     1/(t_+ + u) = Σ f_m u^m.
#
# Multiplying the Taylor series of the non-Gaussian exponential correction and
# the amplitude 1/t gives coefficients p_n. The saddle contribution is
#
#     Φ_sad ~
#     (1/4π) exp(iψ(t_+))
#     sqrt(2π i / ψ''(t_+))
#     Σ_n p_n μ_n.
function large_saddle_asym(a::Float64,b::ComplexF64;N=12)
    _,_,_,tp=gravity_saddles(a,b)
    t=tp
    ψ0=gravity_phase(t,a,b)
    ψ2=gravity_phase_derivative(2,t,a,b)

    q=zeros(ComplexF64,N+1)
    @inbounds for r in 3:N
        q[r+1]=im*gravity_phase_derivative(r,t,a,b)/factorial(r)
    end

    e=gravity_exp_poly_coeff(q)
    fc=ComplexF64[gravity_inverse_taylor_coef(m,t) for m in 0:N]

    p=zeros(ComplexF64,N+1)
    @inbounds for n in 0:N
        s=0im
        for m in 0:n
            s+=fc[m+1]*e[n-m+1]
        end
        p[n+1]=s
    end

    A=0im
    @inbounds for n in 0:N
        A+=p[n+1]*gravity_gaussian_moment(n,ψ2)
    end

    return INV4PI*exp(im*ψ0)*sqrt(2π*im/ψ2)*A
end

# Coefficient in the local expansion:
#
#     1/(m! 12^m).
#
# gamma(m+1) avoids integer factorial overflow. In practice Mloc is small,
# usually 2 or 3, so overflow is not a realistic concern.
@inline function gravity_local_series_coef(m::Int)::Float64
    return inv(gamma(m+1)*12.0^m)
end

#############################################################################
# Sparse symbolic polynomials for local b-derivative recurrence
#
# We represent coefficients A_n(z,b), B_n(z,b) in
#
#     ∂_b^n [ (i/4) H_0^(1)(z) ]
#     =
#     A_n(z,b) H_0^(1)(z) + B_n(z,b) H_1^(1)(z),
#
# where
#
#     z = 2√(ab).
#
# Each coefficient is stored as a sparse sum
#
#     A(z,b) = Σ c[p,q] z^p / b^q.
#
# The derivative identities are:
#
#     ∂_b z = z/(2b),
#     ∂_z H_0^(1)(z) = -H_1^(1)(z),
#     ∂_z H_1^(1)(z) = H_0^(1)(z) - H_1^(1)(z)/z.
#
# Hence
#
#     ∂_b(AH_0 + BH_1)
#       =
#       (D_b A + zB/(2b)) H_0
#       +
#       (D_b B - zA/(2b) - B/(2b)) H_1.
#
# Therefore
#
#     A_{n+1} = D_b A_n + zB_n/(2b),
#     B_{n+1} = D_b B_n - zA_n/(2b) - B_n/(2b),
#
# with
#
#     A_0 = i/4,   B_0 = 0.
#############################################################################

struct GravityLocalCoeffPoly
    terms::Dict{Tuple{Int,Int},ComplexF64}
end

GravityLocalCoeffPoly()=GravityLocalCoeffPoly(Dict{Tuple{Int,Int},ComplexF64}())
GravityLocalCoeffPoly(v::Number)=GravityLocalCoeffPoly(Dict((0,0)=>ComplexF64(v)))

# Runtime-compressed representation.
#
# Dicts are convenient for one-time symbolic construction, but slow and
# allocation-heavy in repeated evaluation. Therefore after construction each
# polynomial is compressed into three arrays:
#
#     p[i], q[i], c[i]  representing c[i] z^p[i] / b^q[i].
struct GravityLocalCoeffTerms
    p::Vector{Int}
    q::Vector{Int}
    c::Vector{ComplexF64}
end

# Insert or accumulate one sparse monomial term:
#
#     c z^p / b^q.
function gravity_poly_addterm!(P::GravityLocalCoeffPoly,p::Int,q::Int,c::ComplexF64)
    abs(c)==0 && return P
    key=(p,q)
    P.terms[key]=get(P.terms,key,0im)+c
    return P
end

# Polynomial addition.
function gravity_poly_add(A::GravityLocalCoeffPoly,B::GravityLocalCoeffPoly)
    C=GravityLocalCoeffPoly(copy(A.terms))
    for ((p,q),v) in B.terms
        gravity_poly_addterm!(C,p,q,v)
    end
    return C
end

# Polynomial subtraction.
function gravity_poly_sub(A::GravityLocalCoeffPoly,B::GravityLocalCoeffPoly)
    C=GravityLocalCoeffPoly(copy(A.terms))
    for ((p,q),v) in B.terms
        gravity_poly_addterm!(C,p,q,-v)
    end
    return C
end

# Explicit b-derivative D_b acting on z^p b^{-q}.
#
# Since z=2√(ab),
#
#     ∂_b z^p = (p/2) z^p / b,
#
# and
#
#     ∂_b b^{-q} = -q b^{-q-1}.
#
# Therefore
# D_b [z^p b^{-q}] = (p/2-q) z^p b^{-q-1}.
function gravity_poly_db(A::GravityLocalCoeffPoly)
    C=GravityLocalCoeffPoly()
    for ((p,q),v) in A.terms
        gravity_poly_addterm!(C,p,q+1,v*(p/2-q))
    end
    return C
end

# Multiplication by z/(2b):
# z^p b^{-q} -> (1/2) z^{p+1} b^{-(q+1)}.
function gravity_poly_z_over_2b(A::GravityLocalCoeffPoly)
    C=GravityLocalCoeffPoly()
    for ((p,q),v) in A.terms
        gravity_poly_addterm!(C,p+1,q+1,0.5*v)
    end
    return C
end

# Multiplication by 1/(2b):
# z^p b^{-q} -> (1/2) z^p b^{-(q+1)}.
function gravity_poly_inv_2b(A::GravityLocalCoeffPoly)
    C=GravityLocalCoeffPoly()
    for ((p,q),v) in A.terms
        gravity_poly_addterm!(C,p,q+1,0.5*v)
    end
    return C
end

# Convert the symbolic Dict representation into compact runtime arrays.
function gravity_poly_compress(A::GravityLocalCoeffPoly)
    p=Int[]
    q=Int[]
    c=ComplexF64[]
    for ((pp,qq),v) in A.terms
        abs(v)==0 && continue
        push!(p,pp)
        push!(q,qq)
        push!(c,v)
    end
    return GravityLocalCoeffTerms(p,q,c)
end

# Evaluate one compressed coefficient polynomial:
# Σ c_i z^{p_i} / b^{q_i}.
@inline function gravity_poly_eval(A::GravityLocalCoeffTerms,z::ComplexF64,b::ComplexF64)
    s=0im
    @inbounds for i in eachindex(A.c)
        s+=A.c[i]*z^A.p[i]/b^A.q[i]
    end
    return s
end

const GRAVITY_LOCAL_COEFF_CACHE=Dict{Int,Tuple{Vector{GravityLocalCoeffTerms},Vector{GravityLocalCoeffTerms}}}()

# Build and cache local recurrence coefficients up to derivative order maxder.
# The cache stores compressed arrays, so the symbolic Dict algebra is paid only
# once per maxder. This matters because phi_local_rec is evaluated many times
# during Chebyshev table construction.
function gravity_local_coeffs(maxder::Int)
    haskey(GRAVITY_LOCAL_COEFF_CACHE,maxder) && return GRAVITY_LOCAL_COEFF_CACHE[maxder]
    A=GravityLocalCoeffPoly(im/4)
    B=GravityLocalCoeffPoly(0im)
    Apoly=Vector{GravityLocalCoeffPoly}(undef,maxder+1)
    Bpoly=Vector{GravityLocalCoeffPoly}(undef,maxder+1)
    Apoly[1]=A
    Bpoly[1]=B
    for n in 1:maxder
        An=gravity_poly_add(gravity_poly_db(A),gravity_poly_z_over_2b(B))
        Bn=gravity_poly_sub(gravity_poly_sub(gravity_poly_db(B),gravity_poly_z_over_2b(A)),gravity_poly_inv_2b(B))
        A=An
        B=Bn
        Apoly[n+1]=A
        Bpoly[n+1]=B
    end
    Aterms=[gravity_poly_compress(Apoly[i]) for i in eachindex(Apoly)]
    Bterms=[gravity_poly_compress(Bpoly[i]) for i in eachindex(Bpoly)]
    GRAVITY_LOCAL_COEFF_CACHE[maxder]=(Aterms,Bterms)
    return Aterms,Bterms
end

# Local-Hankel expansion:
#
#     Φ_loc(a,b)
#     =
#     Σ_{m=0}^{Mloc}
#     1/(m! 12^m)
#     ∂_b^{3m} [ (i/4) H_0^(1)(2√(ab)) ].
#
# Only derivative orders 0,3,6,... occur because the difference between the
# exact gravity phase and the local Helmholtz phase is generated by the cubic
# term -t^3/12.
function phi_local_rec(a::Float64,b::ComplexF64;Mloc=2)
    z=2*sqrt(a*b)
    h0=gravity_h0(z)
    h1=gravity_h1(z)
    Aterms,Bterms=gravity_local_coeffs(3*Mloc)
    out=0im
    @inbounds for m in 0:Mloc
        n=3*m
        coef=gravity_local_series_coef(m)
        A=gravity_poly_eval(Aterms[n+1],z,b)
        B=gravity_poly_eval(Bterms[n+1],z,b)
        out+=coef*(A*h0+B*h1)
    end
    return out
end

# Stokes / outgoing-continuation safety rule.
#
# The saddle amplitude contains exp(iψ(t_+)). Its magnitude is
#
#     |exp(iψ)| = exp(real(iψ)).
#
# In the lower half-plane this saddle may become exponentially huge even though
# the outgoing physical branch has switched it off. Numerically this gives
# meaningless overflow-sized contributions.
#
# Therefore, if
#
#     imag(b) < 0
#     real(iψ(t_+)) > growth_cut,
#
# we suppress the large saddle and keep only the local Hankel part.
function large_saddle_safe(a::Float64,b::ComplexF64;N=12,growth_cut=250.0)
    _,_,_,tp=gravity_saddles(a,b)
    growth=real(im*gravity_phase(tp,a,b))
    if imag(b)<0 && growth>growth_cut
        return 0im,growth,true
    end
    sad=large_saddle_asym(a,b;N=N)
    return sad,growth,false
end

# Full outgoing asymptotic evaluator.
#
# Returns:
#
#     total = Φ_loc + Φ_sad_or_suppressed,
#     loc   = Φ_loc,
#     sad   = included saddle contribution,
#     growth= real(iψ(t_+)),
#     supp  = true if the lower-half-plane saddle was suppressed.
function phi_asym_safe(a::Float64,b::ComplexF64;Mloc=2,Nsad=12,growth_cut=250.0)
    loc=phi_local_rec(a,b;Mloc=Mloc)
    sad,growth,supp=large_saddle_safe(a,b;N=Nsad,growth_cut=growth_cut)
    return loc+sad,loc,sad,growth,supp
end

# Scalar production evaluator when diagnostics are not needed.
@inline function phi_asym(a::Float64,b::ComplexF64;Mloc=2,Nsad=12,growth_cut=250.0)
    return phi_asym_safe(a,b;Mloc=Mloc,Nsad=Nsad,growth_cut=growth_cut)[1]
end

# Evaluate ∂_a of one compressed coefficient polynomial:
#
#     P(z,b)=Σ c_i z^{p_i}/b^{q_i},    z=2√(ab).
# Since ∂_a z = z/(2a) -> ∂_a z^p = (p/(2a)) z^p.
@inline function gravity_poly_eval_da(P::GravityLocalCoeffTerms,z::ComplexF64,b::ComplexF64,a::Float64)
    s=0im
    inv2a=0.5/a
    @inbounds for i in eachindex(P.c)
        P.p[i]==0 && continue
        s+=P.c[i]*(P.p[i]*inv2a)*z^P.p[i]/b^P.q[i]
    end
    return s
end

# Local-Hankel expansion for Φ_a.
#
# Starting from
#
#     Φ_loc = Σ_m c_m ∂_b^{3m}[(i/4)H0(z)],
#     z = 2√(ab),
#
# each cached derivative has the form
#
#     A(z,b)H0(z)+B(z,b)H1(z).
#
# Differentiate w.r.t. a:
#
#     ∂_a z = z/(2a),
#     ∂_z H0 = -H1,
#     ∂_z H1 = H0 - H1/z.
#
# Hence
#
#     ∂_a(AH0+BH1) =
#     [A_a + B z/(2a)] H0
#     +
#     [B_a - A z/(2a) - B/(2a)] H1.
function phi_a_local_rec(a::Float64,b::ComplexF64;Mloc=2)
    z=2*sqrt(a*b)
    h0=gravity_h0(z)
    h1=gravity_h1(z)
    Aterms,Bterms=gravity_local_coeffs(3*Mloc)

    out=0im
    inv2a=0.5/a

    @inbounds for m in 0:Mloc
        n=3*m
        coef=gravity_local_series_coef(m)

        A=gravity_poly_eval(Aterms[n+1],z,b)
        B=gravity_poly_eval(Bterms[n+1],z,b)

        Aa=gravity_poly_eval_da(Aterms[n+1],z,b,a)
        Ba=gravity_poly_eval_da(Bterms[n+1],z,b,a)

        C0=Aa+B*z*inv2a
        C1=Ba-A*z*inv2a-B*inv2a

        out+=coef*(C0*h0+C1*h1)
    end

    return out
end

# Taylor coefficient for the differentiated saddle amplitude.
#
# For Φ:
#
#     amplitude = 1/t.
#
# For Φ_a:
#
#     ∂_a exp(iψ) = i(∂_aψ)exp(iψ) = i exp(iψ)/t,
#
# so the total amplitude becomes
#
#     i/t^2.
#
# Around t=t_+:
#
#     i/(t+u)^2 = i Σ_{m≥0} (-1)^m (m+1) u^m / t^{m+2}.
@inline gravity_inverse_taylor_coef_da(m::Int,t)=im*(-1)^m*(m+1)/t^(m+2)

# Large-saddle steepest-descent expansion for Φ_a.
#
# This is the saddle expansion of the differentiated integral
#
#     Φ_a = (1/4π) ∫ i exp(iψ(t)) dt/t^2.
#
# Therefore the phase and Gaussian moments are unchanged from Φ_sad; only the
# amplitude Taylor coefficients change from 1/t to i/t^2.
function large_saddle_asym_da(a::Float64,b::ComplexF64;N=12)
    _,_,_,tp=gravity_saddles(a,b)
    t=tp
    ψ0=gravity_phase(t,a,b)
    ψ2=gravity_phase_derivative(2,t,a,b)

    q=zeros(ComplexF64,N+1)
    @inbounds for r in 3:N
        q[r+1]=im*gravity_phase_derivative(r,t,a,b)/factorial(r)
    end

    e=gravity_exp_poly_coeff(q)
    fc=ComplexF64[gravity_inverse_taylor_coef_da(m,t) for m in 0:N]

    p=zeros(ComplexF64,N+1)
    @inbounds for n in 0:N
        s=0im
        for m in 0:n
            s+=fc[m+1]*e[n-m+1]
        end
        p[n+1]=s
    end

    A=0im
    @inbounds for n in 0:N
        A+=p[n+1]*gravity_gaussian_moment(n,ψ2)
    end

    return INV4PI*exp(im*ψ0)*sqrt(2π*im/ψ2)*A
end

# Stokes / outgoing-continuation safety rule for Φ_a.
#
# The differentiated saddle contribution
#
#     Φ_{a,sad}
#
# inherits exactly the same exponential factor
#
#     exp(iψ(t_+))
#
# as the undifferentiated saddle.
#
# Therefore the same Stokes suppression criterion applies:
#
#     imag(b) < 0
#
# together with
#
#     real(iψ(t_+)) > growth_cut.
#
# In this regime the outgoing continuation has already switched off the large
# saddle, while the formal asymptotic series would instead produce an
# exponentially huge contribution.
#
# Returns
#
#     (sad,growth,suppressed)
#
# where
#
#     sad        = retained saddle contribution
#     growth     = real(iψ(t_+))
#     suppressed = true if the saddle was discarded.
function large_saddle_safe_da(a::Float64,b::ComplexF64;N=12,growth_cut=250.0)
    _,_,_,tp=gravity_saddles(a,b)
    growth=real(im*gravity_phase(tp,a,b))
    if imag(b)<0 && growth>growth_cut
        return 0im,growth,true
    end
    sad=large_saddle_asym_da(a,b;N=N)
    return sad,growth,false
end

# Full asymptotic evaluator for
#
#     Φ_a(a,b) = ∂Φ/∂a.
#
# The approximation is decomposed into
#
#     Φ_a ≈ Φ_{a,loc} + Φ_{a,sad},
#
# where Φ_{a,loc}
# is obtained by differentiating the local Hankel expansion and Φ_{a,sad} is obtained from steepest descent with amplitude i/t^2 instead of the original 1/t.
function phi_a_asym_safe(a::Float64,b::ComplexF64;Mloc=2,Nsad=12,growth_cut=250.0)
    loc=phi_a_local_rec(a,b;Mloc=Mloc)
    sad,growth,supp=large_saddle_safe_da(a,b;N=Nsad,growth_cut=growth_cut)
    return loc+sad,loc,sad,growth,supp
end

# Production evaluator for
#
#     Φ_a(a,b) = ∂Φ/∂a.
#
# This is the quantity required by the double-layer potential kernel.
# Internally this combines: local Hankel expansion and large-saddle asymptotics.
# Only the total value is returned.
@inline function phi_a_asym(a::Float64,b::ComplexF64;Mloc=2,Nsad=12,growth_cut=250.0)
    return phi_a_asym_safe(a,b;Mloc=Mloc,Nsad=Nsad,growth_cut=growth_cut)[1]
end

#############################################################################
# Airy-CFU evaluator near the turning line b^2=a
#
# The separated saddle expansion becomes non-uniform when the two saddles
# coalesce:
#
#     b^2-a ≈ 0.
#
# In that regime we use the Chester--Friedman--Ursell cubic normal form
#
#     ψ(t(w)) = A + w^3/3 - ηw,
#
# whose saddles are at
#
#     w = ±sqrt(η).
#
# The canonical Airy moments are
#
#     I_0(η) = ∫ exp(i(w^3/3-ηw)) dw = 2π Ai(-η),
#     I_1(η) = ∫ w exp(i(w^3/3-ηw)) dw = 2π i Ai'(-η).
#
# We use mpmath only for Ai and Ai'. The CFU algebra itself is ComplexF64.
#
# For Φ:
#
#     G(w)=t'(w)/t(w).
#
# For Φ_a:
#
#     G_a(w)=i t'(w)/t(w)^2.
#
# The same CFU recurrence is then used for both.
#############################################################################

# Raw saddle points of ψ(t).
#
# Unlike gravity_saddles, this does not return logarithmic coordinates and does
# not reorder by real(log(t)). CFU needs the two coalescing saddles directly.
@inline function gravity_saddles_raw(a::Float64,b::ComplexF64)
    root=sqrt(b*b-a+0im)
    tm=sqrt(2*(b-root)+0im)
    tp=sqrt(2*(b+root)+0im)
    real(tm)<0 && (tm=-tm)
    real(tp)<0 && (tp=-tp)
    return tm,tp
end

# Airy moments for the CFU cubic normal form.
#
# The required canonical moments are
#
#     I_0(η)=2π Ai(-η),
#     I_1(η)=2π i Ai'(-η).
#
# mpmath is used only here, guarded by the shared PyCall lock. The derivative is
# requested directly by mpmath's derivative argument:
#
#     airyai(z,1).
function gravity_airy_moments(η::ComplexF64;dps::Int=50)
    lock(PYCALL_MPMATH_LOCK)
    try
        _mpctx[].dps=dps
        z=-η
        zmp=_mpc[](real(z),imag(z))
        Ai=_mp_airyai[](zmp)
        Aip=_mp_airyai[](zmp,1)
        I0=2π*ComplexF64(_pyfloat[](Ai.real),_pyfloat[](Ai.imag))
        I1=2π*im*ComplexF64(_pyfloat[](Aip.real),_pyfloat[](Aip.imag))
        return I0,I1
    finally
        unlock(PYCALL_MPMATH_LOCK)
    end
end

# Cubic-root branch factor for η.
# Since η is defined by a 2/3 power, three equivalent branches are possible.
# The factor exp(4πi branch/3) selects the CFU/Stokes branch.
@inline gravity_cfu_branch_factor(branch::Int)=cis(4*π*branch/3)

# Default Airy branch choice.
# The shadow side b^2-a<0 uses branch 1; the oscillatory side uses branch 0.
# In production we mainly call Airy only on the oscillatory side.
@inline gravity_airy_branch(a::Float64,b::ComplexF64)=real(b*b-a)<0 ? 1 : 0

# CFU normal-form parameters.
#
# The two saddle values ψ_- and ψ_+ are mapped to
#
#     ψ(t(w)) = A + w^3/3 - ηw.
#
# Matching the two saddle values gives
#
#     A = (ψ_+ + ψ_-)/2,
#     η = [3(ψ_+ - ψ_-)/4]^(2/3),
#
# up to the cubic-root branch.
function gravity_cfu_parameters(a::Float64,b::ComplexF64;branch::Int=0)
    tm,tp=gravity_saddles_raw(a,b)
    ψm=gravity_phase(tm,a,b)
    ψp=gravity_phase(tp,a,b)
    A=0.5*(ψp+ψm)
    Δ=ψp-ψm
    η=(0.75*Δ)^(2/3)*gravity_cfu_branch_factor(branch)
    return A,η,sqrt(η),tm,tp
end

# Multiply two truncated Taylor series.
#
# The series convention throughout the CFU code is
#
#     x(v)=Σ_{n=0}^N x[n+1] v^n,
#     y(v)=Σ_{n=0}^N y[n+1] v^n.
#
# This returns the truncated Cauchy product
#
#     z(v)=x(v)y(v) mod v^{N+1}.
#
# Terms of degree larger than N are discarded. This is used only during local
# symbolic coefficient construction, not in the high-throughput kernel path.
function gravity_series_mul(x::Vector{ComplexF64},y::Vector{ComplexF64},N::Int)
    z=zeros(ComplexF64,N+1)
    @inbounds for i in 0:N
        xi=x[i+1]
        xi==0 && continue
        for j in 0:N-i
            z[i+j+1]+=xi*y[j+1]
        end
    end
    return z
end

# Integer power of a truncated Taylor series.
#
# Returns
#
#     x(v)^k mod v^{N+1}.
#
# The case k=0 returns the constant series 1. This deliberately uses repeated
# truncated multiplication because N is small in the CFU patch, typically
# N≈30-50, and clarity is more important than asymptotic polynomial efficiency.
function gravity_series_pow(x::Vector{ComplexF64},k::Int,N::Int)
    y=zeros(ComplexF64,N+1)
    y[1]=1
    for _ in 1:k
        y=gravity_series_mul(y,x,N)
    end
    return y
end

# Derivative of a truncated Taylor series.
#
# If
#     x(v)=Σ_{n=0}^N x[n+1] v^n,
# then this returns
#     x'(v)=Σ_{n=1}^N n x[n+1] v^{n-1}.
# The returned vector therefore has length N rather than N+1.
function gravity_series_derivative(x::Vector{ComplexF64})
    N=length(x)-1
    N==0 && return ComplexF64[0]
    y=zeros(ComplexF64,N)
    @inbounds for n in 1:N
        y[n]=n*x[n+1]
    end
    return y
end

# Divide two truncated power series: q = num/den.
# Requires den[1] != 0.
function gravity_series_divide(num::Vector{ComplexF64},den::Vector{ComplexF64},N::Int)
    q=zeros(ComplexF64,N+1)
    @inbounds for n in 0:N
        s=num[n+1]
        for k in 0:n-1
            s-=q[k+1]*den[n-k+1]
        end
        q[n+1]=s/den[1]
    end
    return q
end

# Phase series around one saddle.
#
# Given
#     t = tσ + u(v),
# return the coefficients of
#     ψ(tσ+u(v))-ψ(tσ).
function gravity_phase_series_at_saddle(a::Float64,b::ComplexF64,tσ::ComplexF64,u::Vector{ComplexF64},N::Int)
    out=zeros(ComplexF64,N+1)
    for r in 2:N
        ur=gravity_series_pow(u,r,N)
        fac=gravity_phase_derivative(r,tσ,a,b)/factorial(r)
        @inbounds for n in 0:N
            out[n+1]+=fac*ur[n+1]
        end
    end
    return out
end

# Local inverse map t(w) near one saddle.
#
# Let
#     v = w - wσ,
#     t = tσ + u(v),
# and solve
#     ψ(tσ+u(v))-ψ(tσ) = -wσ v^2 - v^3/3.
# The coefficient c1 is fixed by the quadratic term, and higher coefficients
# are solved recursively by linearization.
function gravity_cfu_local_inverse_series(a::Float64,b::ComplexF64,tσ::ComplexF64,wσ::ComplexF64,N::Int,sgn::Int)
    c=zeros(ComplexF64,N+1)
    c[2]=sgn*sqrt((-2*wσ)/gravity_phase_derivative(2,tσ,a,b))
    rhs=zeros(ComplexF64,N+1)
    N>=2 && (rhs[3]=-wσ)
    N>=3 && (rhs[4]=-1/3)
    for n in 2:N
        d=n+1
        d>N && break
        c[n+1]=0
        known=gravity_phase_series_at_saddle(a,b,tσ,c,N)[d+1]
        c[n+1]=1
        withone=gravity_phase_series_at_saddle(a,b,tσ,c,N)[d+1]
        c[n+1]=(rhs[d+1]-known)/(withone-known)
    end
    return c
end

# CFU amplitude series for Φ.
#
# Since
#     Φ = (1/4π)∫ exp(iψ(t)) dt/t,
# the transformed amplitude is
#     G(w) = t'(w)/t(w).
function gravity_cfu_phi_amplitude_series(a::Float64,b::ComplexF64,tσ::ComplexF64,wσ::ComplexF64,N::Int,sgn::Int)
    u=gravity_cfu_local_inverse_series(a,b,tσ,wσ,N+1,sgn)
    up=gravity_series_derivative(u)
    den=zeros(ComplexF64,N+1)
    num=zeros(ComplexF64,N+1)
    den[1]=tσ
    @inbounds for n in 1:N
        den[n+1]=u[n+1]
    end
    @inbounds for n in 0:min(N,length(up)-1)
        num[n+1]=up[n+1]
    end
    return gravity_series_divide(num,den,N)
end

# CFU amplitude series for Φ_a.
#
# Since
#     Φ_a = ∂_a Φ = (1/4π)∫ i exp(iψ(t)) dt/t^2,
# the transformed amplitude is
#     G_a(w) = i t'(w)/t(w)^2.
function gravity_cfu_phi_a_amplitude_series(a::Float64,b::ComplexF64,tσ::ComplexF64,wσ::ComplexF64,N::Int,sgn::Int)
    u=gravity_cfu_local_inverse_series(a,b,tσ,wσ,N+1,sgn)
    up=gravity_series_derivative(u)
    den=zeros(ComplexF64,N+1)
    num=zeros(ComplexF64,N+1)
    den[1]=tσ
    @inbounds for n in 1:N
        den[n+1]=u[n+1]
    end
    @inbounds for n in 0:min(N,length(up)-1)
        num[n+1]=im*up[n+1]
    end
    q=gravity_series_divide(num,den,N)
    return gravity_series_divide(q,den,N)
end

# Divide by the local factor w^2-η.
# Around one saddle w=wσ, with v=w-wσ,
#
#     w^2-η = v(2wσ+v).
#
# If R(v)=v(2wσ+v)H(v), this recovers H(v).
function gravity_cfu_divide_by_w2_minus_eta(R::Vector{ComplexF64},wσ::ComplexF64,N::Int)
    H=zeros(ComplexF64,N)
    old=0im
    @inbounds for n in 1:N
        H[n]=(R[n+1]-old)/(2*wσ)
        old=H[n]
    end
    return H
end

# One CFU integration-by-parts step.
# Decompose
#
#     G(w)=α+βw+(w^2-η)H(w).
#
# Since
#     d/dw exp(i(w^3/3-ηw)) = i(w^2-η) exp(i(...)),
# integration by parts gives the next amplitude
#     G_next(w) = -i H'(w).
function gravity_cfu_step(Gm::Vector{ComplexF64},Gp::Vector{ComplexF64},s::ComplexF64,N::Int)
    wm=-s
    wp=s
    α=0.5*(Gp[1]+Gm[1])
    β=0.5*(Gp[1]-Gm[1])/s
    Rm=copy(Gm)
    Rp=copy(Gp)
    Rm[1]-=α+β*wm
    Rm[2]-=β
    Rp[1]-=α+β*wp
    Rp[2]-=β
    Hm=gravity_cfu_divide_by_w2_minus_eta(Rm,wm,N)
    Hp=gravity_cfu_divide_by_w2_minus_eta(Rp,wp,N)
    Gm_next=(-im).*gravity_series_derivative(Hm)
    Gp_next=(-im).*gravity_series_derivative(Hp)
    M=N-1
    Gmn=zeros(ComplexF64,M+1)
    Gpn=zeros(ComplexF64,M+1)
    @inbounds for i in eachindex(Gm_next)
        Gmn[i]=Gm_next[i]
        Gpn[i]=Gp_next[i]
    end
    return α,β,Gmn,Gpn
end

# Evaluate the CFU expansion.
#
# The asymptotic series is truncated at the first increase of the term norm.
# This gives the usual optimal truncation for an asymptotic expansion.
function gravity_cfu_sum(A::ComplexF64,η::ComplexF64,s::ComplexF64,Gm::Vector{ComplexF64},Gp::Vector{ComplexF64};N::Int=40,Jmax::Int=20,tol::Float64=1e-14,airy_dps::Int=50)
    I0,I1=gravity_airy_moments(η;dps=airy_dps)
    total=0im
    prevterm=Inf
    Nj=N
    @inbounds for _ in 1:Jmax
        α,β,Gm,Gp=gravity_cfu_step(Gm,Gp,s,Nj)
        term=α*I0+β*I1
        tnorm=abs(term)
        tnorm>prevterm && break
        total+=term
        tnorm/max(abs(total),1e-300)<tol && break
        prevterm=tnorm
        Nj-=1
        Nj<=4 && break
    end
    return INV4PI*exp(im*A)*total
end

# Airy-CFU evaluator for Φ near b^2=a.
#
# This evaluates the uniform approximation
#
#     Φ(a,b) ≈ (1/4π) exp(iA) Σ_j [α_j I_0(η)+β_j I_1(η)],

# where the CFU normal form is
#     ψ(t(w)) = A + w^3/3 - ηw.
# The initial amplitude is
#     G_0(w)=t'(w)/t(w),
# corresponding to the original integral dt/t. The two local Taylor series at
# w=±sqrt(η) are constructed from the inverse map t(w), and the shared CFU
# recurrence produces the coefficients α_j,β_j.
#
# Keyword notes:
#
#     N        local Taylor order for the inverse-map/amplitude series,
#     Jmax     maximum number of CFU integration-by-parts steps,
#     tol      early stop when the last term is small relative to the sum,
#     airy_dps mpmath precision used only for Ai and Ai',
#     branch   optional manual Stokes branch for η,
#     sgnm/p   signs of the local inverse-map square roots at the two saddles.
function phi_airy_cfu(a::Float64,b::ComplexF64;N::Int=40,Jmax::Int=20,tol::Float64=1e-14,airy_dps::Int=50,branch::Union{Nothing,Int}=nothing,sgnm::Int=1,sgnp::Int=1)
    br=branch===nothing ? gravity_airy_branch(a,b) : branch
    A,η,s,tm,tp=gravity_cfu_parameters(a,b;branch=br)
    abs(s)<1e-14 && error("Too close to exact η=0")
    Gm=gravity_cfu_phi_amplitude_series(a,b,tm,-s,N,sgnm)
    Gp=gravity_cfu_phi_amplitude_series(a,b,tp,s,N,sgnp)
    return gravity_cfu_sum(A,η,s,Gm,Gp;N=N,Jmax=Jmax,tol=tol,airy_dps=airy_dps)
end

# Airy-CFU evaluator for Φ_a near b^2=a.
#
# This is the uniform approximation for
#
#     Φ_a(a,b)=∂Φ/∂a.
#
# Since
#     ∂_a exp(iψ(t)) = i/t * exp(iψ(t)),
# the differentiated integral has amplitude
#     i dt/t^2.
# Therefore the CFU initial amplitude is
#     G_{0,a}(w)=i t'(w)/t(w)^2.
# Everything else is identical to phi_airy_cfu: same cubic normal form, same
# Airy moments, same recurrence, and same optimal-truncation rule.
function phi_a_airy_cfu(a::Float64,b::ComplexF64;N::Int=40,Jmax::Int=20,tol::Float64=1e-14,airy_dps::Int=50,branch::Union{Nothing,Int}=nothing,sgnm::Int=1,sgnp::Int=1)
    br=branch===nothing ? gravity_airy_branch(a,b) : branch
    A,η,s,tm,tp=gravity_cfu_parameters(a,b;branch=br)
    abs(s)<1e-14 && error("Too close to exact η=0")
    Gm=gravity_cfu_phi_a_amplitude_series(a,b,tm,-s,N,sgnm)
    Gp=gravity_cfu_phi_a_amplitude_series(a,b,tp,s,N,sgnp)
    return gravity_cfu_sum(A,η,s,Gm,Gp;N=N,Jmax=Jmax,tol=tol,airy_dps=airy_dps)
end

# Production switch for the Airy patch.
# The normalized distance from the turning line is
#
#     |b^2-a| / max(|b^2|,|a|,1).
#
# Airy is enabled only on the oscillatory side, where the previous tests showed
# 1e-12-level behavior. The shadow side remains handled by the local Hankel /
# Stokes-suppressed asymptotic evaluator.
@inline function gravity_near_turning(a::Float64,b::ComplexF64;τ::Float64=0.05)
    d=abs(b*b-a)
    scale=max(abs(b*b),abs(a),1.0)
    return d/scale<τ && real(b*b-a)>0
end

# Production evaluator for Φ.
# Uses Airy-CFU near the oscillatory turning line and otherwise falls back to
# the local-Hankel plus separated large-saddle asymptotic evaluator.
@inline function phi_gravity(a::Float64,b::ComplexF64;Mloc=2,Nsad=12,growth_cut=250.0)
    gravity_near_turning(a,b) && return phi_airy_cfu(a,b)
    return phi_asym(a,b;Mloc=Mloc,Nsad=Nsad,growth_cut=growth_cut)
end

# Production evaluator for Φ_a.
# This is the version required by DLP. Near the oscillatory turning line it uses
# the Airy-CFU amplitude G_a(w)=i t'(w)/t(w)^2.
@inline function phi_a_gravity(a::Float64,b::ComplexF64;Mloc=2,Nsad=12,growth_cut=250.0)
    gravity_near_turning(a,b) && return phi_a_airy_cfu(a,b)
    return phi_a_asym(a,b;Mloc=Mloc,Nsad=Nsad,growth_cut=growth_cut)
end

# Double-layer potential kernel.
#
# The gravity Green function depends on
#
#     a = |x-y|^2 / 4.
#
# Assuming b is independent of the source point y,
#
#     ∂_{n_y} Φ = Φ_a ∂_{n_y} a.
#
# Since
#
#     a = |x-y|^2 / 4,
#
# one finds
#
#     ∂_{n_y} a = -(x-y)·n_y / 2.
#
# Therefore the DLP kernel is
#
#     K_DLP(x,y) = -((x-y)·n_y)/2 * Φ_a(a,b).
#
# Arguments
#
#     (xi,yi)   target point
#     (xj,yj)   source point
#     (nxj,nyj) outward source normal
#
# Returns ∂_{n_y} Φ(a,b).
@inline function gravity_dlp_kernel(xi,yi,xj,yj,nxj,nyj,b::ComplexF64;Mloc=2,Nsad=12,growth_cut=250.0)
    dx=xi-xj
    dy=yi-yj
    a=0.25*(dx*dx+dy*dy)
    dadn=-0.5*(dx*nxj+dy*nyj)
    return dadn*phi_a_gravity(a,b;Mloc=Mloc,Nsad=Nsad,growth_cut=growth_cut)
end

# Single-layer potential kernel.
#
# The gravity Green function depends only on
#
#     a = |x-y|^2 / 4
#
# and the spectral parameter b.
# The SLP kernel is therefore simply
#
#     K_SLP(x,y) = Φ(a,b).
#
# Arguments
#
#     (xi,yi) target point
#     (xj,yj) source point
#
# Returns Φ(a,b).
@inline function gravity_slp_kernel(xi::Float64,yi::Float64,xj::Float64,yj::Float64,b::ComplexF64;Mloc=2,Nsad=12,growth_cut=250.0)
    dx=xi-xj
    dy=yi-yj
    a=0.25*(dx*dx+dy*dy)
    return phi_gravity(a,b;Mloc=Mloc,Nsad=Nsad,growth_cut=growth_cut)
end

# Single-layer potential kernel evaluated directly from
#
#     a = |x-y|^2 / 4.
#
# This version avoids recomputing the distance when a has already been
# assembled elsewhere (for example in cached geometry workspaces).
#
# Returns Φ(a,b).
@inline function gravity_slp_kernel_from_a(a::Float64,b::ComplexF64;Mloc=2,Nsad=12,growth_cut=250.0)
    return phi_gravity(a,b;Mloc=Mloc,Nsad=Nsad,growth_cut=growth_cut)
end

















if abspath(PROGRAM_FILE)==@__FILE__

const GRAVITY_TEST_A1=1/pi+1/2
const GRAVITY_TEST_B1=pi/4-1/2
const GRAVITY_TEST_A2=1/pi+1/6
const GRAVITY_TEST_B2=pi/12-1/2
const GRAVITY_TEST_CMINUS=0.5*tan(GRAVITY_TEST_B1/GRAVITY_TEST_A1)
const GRAVITY_TEST_CPLUS=0.25*tan(GRAVITY_TEST_B2/GRAVITY_TEST_A2)

@inline gravity_relerr(x,y)=abs(x-y)/max(abs(y),eps(Float64))

@inline function gravity_allowed_g_gp(α,sm,sp,Eeff)
    x1=2*(α-real(sm)+GRAVITY_TEST_CMINUS)
    x2=-4*(α-real(sp)-GRAVITY_TEST_CPLUS)
    f1=GRAVITY_TEST_A1*atan(x1)-GRAVITY_TEST_B1
    f2=GRAVITY_TEST_A2*atan(x2)-GRAVITY_TEST_B2
    f1p=GRAVITY_TEST_A1*2/(1+x1*x1)
    f2p=GRAVITY_TEST_A2*(-4)/(1+x2*x2)
    return f1*f2,f1p*f2+f1*f2p
end

function gravity_contour_ref(a::Float64,b::ComplexF64;rtol=1e-12,atol=1e-14,margin=10.0,cap=250.0)
    sm,sp,_,_=gravity_saddles(a,b)
    L=real(sm)-margin
    R=real(sp)+margin
    Eeff=max(real(b),1e-15)
    f(α)=begin
        g,gp=gravity_allowed_g_gp(α,sm,sp,Eeff)
        s=complex(α,g)
        ds=complex(1,gp)
        abs(real(s))>700 && return 0im
        t=exp(s)
        z=im*gravity_phase(t,a,b)
        (!isfinite(real(z)) || real(z)>cap) && return 0im
        return exp(z)*ds
    end
    val,err=quadgk(f,L,R;rtol=rtol,atol=atol,maxevals=10^7)
    return INV4PI*val,INV4PI*err
end

function gravity_contour_ref_da(a::Float64,b::ComplexF64;rtol=1e-12,atol=1e-14,margin=10.0,cap=250.0)
    sm,sp,_,_=gravity_saddles(a,b)
    L=real(sm)-margin
    R=real(sp)+margin
    Eeff=max(real(b),1e-15)
    f(α)=begin
        g,gp=gravity_allowed_g_gp(α,sm,sp,Eeff)
        s=complex(α,g)
        ds=complex(1,gp)
        abs(real(s))>700 && return 0im
        t=exp(s)
        z=im*gravity_phase(t,a,b)
        (!isfinite(real(z)) || real(z)>cap) && return 0im
        return im*exp(z)*ds/t
    end
    val,err=quadgk(f,L,R;rtol=rtol,atol=atol,maxevals=10^7)
    return INV4PI*val,INV4PI*err
end

function run_gravity_airy_cfu_test(;alist=(1.0,12.5),deltas=(-0.2,-0.1,-0.03,-0.01,0.01,0.03,0.1,0.2),imbs=(0.0,1e-6,-1e-6),N=40,Jmax=20,airy_dps=70)
    println("\n"*"="^160)
    println("Gravity Airy-CFU Φ / Φ_a test")
    println("="^160)
    @printf("%10s %11s %11s %15s %11s %11s %11s %11s %11s\n",
        "a","delta","Imb","b²-a","relΦ","absΦ","relΦa","absΦa","|Φref|")
    maxrelΦ=0.0
    maxrelΦa=0.0
    for a in alist
        bc=sqrt(a)
        for imb in imbs,δ in deltas
            b=ComplexF64(bc*(1+δ)+im*imb)
            q,qe=gravity_contour_ref(a,b)
            qa,qae=gravity_contour_ref_da(a,b)
            branch=real(b*b-a)<0 ? 1 : 0
            v=phi_airy_cfu(a,b;N=N,Jmax=Jmax,airy_dps=airy_dps,branch=branch)
            va=phi_a_airy_cfu(a,b;N=N,Jmax=Jmax,airy_dps=airy_dps,branch=branch)
            relΦ=gravity_relerr(v,q)
            relΦa=gravity_relerr(va,qa)
            maxrelΦ=max(maxrelΦ,relΦ)
            maxrelΦa=max(maxrelΦa,relΦa)
            @printf("%10.3e %11.3e %11.3e %15.6e %11.3e %11.3e %11.3e %11.3e %11.3e\n",
                a,δ,imb,real(b*b-a),relΦ,abs(v-q),relΦa,abs(va-qa),abs(q))
            if real(b*b-a)>0
                @test relΦ<5e-10 || abs(v-q)<5e-11
                @test relΦa<5e-8 || abs(va-qa)<5e-8
            else
                @test isfinite(real(v)) && isfinite(imag(v))
                @test isfinite(real(va)) && isfinite(imag(va))
            end
        end
    end
    println("\nmax rel Φ  = ",maxrelΦ)
    println("max rel Φa = ",maxrelΦa)
    return maxrelΦ,maxrelΦa
end

function run_gravity_switch_continuity_test(;a=12.5,δs=(-0.08,-0.04,-0.02,0.02,0.04,0.08),N=40,Jmax=20)
    println("\n"*"="^160)
    println("Gravity switching continuity test")
    println("="^160)
    @printf("%11s %15s %12s %12s %12s %12s\n","delta","b²-a","rel Φ","abs Φ","rel Φa","abs Φa")
    bc=sqrt(a)
    maxjumpΦ=0.0
    maxjumpΦa=0.0
    for δ in δs
        b=ComplexF64(bc*(1+δ))
        airy=phi_airy_cfu(a,b;N=N,Jmax=Jmax,branch=gravity_airy_branch(a,b))
        asym=phi_asym(a,b)
        airya=phi_a_airy_cfu(a,b;N=N,Jmax=Jmax,branch=gravity_airy_branch(a,b))
        asyma=phi_a_asym(a,b)
        e=gravity_relerr(airy,asym)
        ea=gravity_relerr(airya,asyma)
        maxjumpΦ=max(maxjumpΦ,e)
        maxjumpΦa=max(maxjumpΦa,ea)
        @printf("%11.3e %15.6e %12.3e %12.3e %12.3e %12.3e\n",
            δ,real(b*b-a),e,abs(airy-asym),ea,abs(airya-asyma))
    end
    println("\nmax switch rel Φ  = ",maxjumpΦ)
    println("max switch rel Φa = ",maxjumpΦa)
    return maxjumpΦ,maxjumpΦa
end

function run_gravity_kernel_smoke_test(;E=ComplexF64(100.0^2,0.0),nsamp=10_000)
    rng=MersenneTwister(1234)
    vals=Vector{ComplexF64}(undef,nsamp)
    valsa=Vector{ComplexF64}(undef,nsamp)
    for i in 1:nsamp
        xi=2rand(rng)-1
        yi=2rand(rng)-1
        xj=2rand(rng)-1
        yj=2rand(rng)-1
        θ=2π*rand(rng)
        nx=cos(θ)
        ny=sin(θ)
        b=E+0.2*(yi+yj)
        vals[i]=gravity_slp_kernel(xi,yi,xj,yj,b)
        valsa[i]=gravity_dlp_kernel(xi,yi,xj,yj,nx,ny,b)
    end
    @test all(isfinite,real.(vals))
    @test all(isfinite,imag.(vals))
    @test all(isfinite,real.(valsa))
    @test all(isfinite,imag.(valsa))
    println("\nKernel smoke test passed for ",nsamp," random points.")
    return vals,valsa
end

@testset "Gravity Airy-CFU" begin
    run_gravity_airy_cfu_test()
    run_gravity_switch_continuity_test()
    run_gravity_kernel_smoke_test()
end

end