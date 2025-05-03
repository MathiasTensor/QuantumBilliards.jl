using LinearAlgebra,FFTW,Test

"""
Root‐Finding Utilities via Boyd’s Method and Interval Subdivision

This file provides three main tools:

1. `companion_matrix(coeffs::Vector{Complex{T}})`:  
   Build the companion matrix of a monic polynomial  
   p(z)=a_0 + a_1 z + ... + a_{n-1} z^{n-1} + z^n,
   so that its eigenvalues are the roots of p.

2. `interval_roots_boyd(f, a, b; Ftol, Mmax, print_error, rtol)`:  
   Find all real roots of an analytic function f in the interval (a,b)  
   to near‐machine precision, using Boyd’s degree‐doubling Fourier method.  

3. `subdivide_intervals(Nfunc, a, b; target_count, max_delta, tol_bisection, tol_ΔN)`:  
   Subdivide (a,b) into subintervals (a_i,b_i) so that  
   - the estimated root‐count Nfunc(b_i) - Nfunc(a_i) ≈ target_count},  
   - each (b_i-a_i) <= max_delta}.  
   This guarantees that Boyd’s method sees only a few roots per piece.
"""

########################
#### MAIN FUNCTIONS ####
########################

"""
    companion_matrix(coeffs::Vector{Complex{T}}) where {T<:Real}

Construct the companion matrix of the monic polynomial (final coeff must be one(T) to satisfy the construction)
Notes: https://math.mit.edu/~edelman/publications/polynomial_roots.pdf

    p(z) = a_0 + a_1*z + ... + a_{n-1} z^{n-1} + z^n  

# Arguments
- `coeffs::Vector{Complex{T}}`: A length-n+1 vector of complex coefficients [ a_0, a_1, …, a_{n-1}, 1 ]

# Returns
- `C::Matrix{Complex{T}}`: Companion matrix whose characteristic polynomial is (p(z)). Its eigenvalues are exactly the
roots of p.
"""
function companion_matrix(coeffs::Vector{Complex{T}}) where {T<:Real}
    n=length(coeffs)-1 
    C=zeros(Complex{T},n,n)
    C[diagind(C,-1)].=one(T)
    C[:,end].=-coeffs[1:n]
    return C
end

"""
    interval_roots_boyd(f::Function,a::T,b::T;Ftol::Float64=1e-12,Mmax::Int=1024,print_error::Bool=false,rtol::Float64=1e-8) where {T<:Real}

BY ALEX BARNETT - Check https://github.com/ahbarnett/mpspack/blob/master/%40utils/intervalrootsboyd.m
Find all real roots of the analytic function `f` in the interval `(a,b)`
to near machine precision, using Boyd's degree-doubling method.
Example usage: https://users.flatironinstitute.org/~ahb/dartmouth/papers/zhaodet.pdf

# Arguments
- `f::Function`: T -> Complex{T}
- `a::T`: Lower interval bound
- `b::T`: Upper interval bound

Returns
- `xs::Vector{T}`: estimated roots in (a,b)

Keyword Arguments
- `Ftol`: relative tolerance on the Fourier‐coefficient tail (default 1e-12). This is also a first filter for close roots but the user will need to filter the "too-close" ones manually based on output (many times gives slight variation in the final digit up to Ftol)
- `Nmax`: maximum half-samples on [0,π] (so up to `2*Nmax` total func‐evals)
- `rtol`: Compare next M iteration roots abs diff vs. the previous iterations's results.
"""
function interval_roots_boyd(f::Function,a::T,b::T;Ftol::Float64=1e-12,Mmax::Int=1024,print_error::Bool=false,rtol::Float64=1e-8) where {T<:Real}
    @assert Ftol<1e-10 "Ftol should be larger than 1e-10, default is 1e-12. Using Ftol=1e-14 or smaller usually crashes so best not use it!"
    rad=(b-a)/2
    cen=(b+a)/2
    M=32 # og code says 4, but let's be safe here...
    last_roots=Vector{Complex{T}}()
    while M≤Mmax
        N=2*M
        θ=@. π*(0:M)/M # length M+1, goes from 0→π
        y=@. cen+rad*cos(θ)
        u=Vector{Complex{T}}(undef,N)
        for j in 1:(M+1) # fill the first M+1 entries
            u[j]=f(y[j])
        end
        @inbounds for j in 2:M # mirror to fill the remaining M−1 entries
            u[M+j]=u[M+2-j] # we want u[M+2] = u[M], u[M+3] = u[M−1], …, u[2M] = u[2]
        end
        U=fft(u) # F[1]=c_{-M}, F[M+1]=c_0, F[end]=c_{M-1}
        F=fftshift(U)
        F./=F[end] # normalize to make the polynomial monic, now F[end]=1+0im
        if abs(F[1]/F[M+1])<Ftol # check tail |c_{-M}| / |c_0| < tol
            C=companion_matrix(F) # build companion‐matrix from length‐N vector F = [c_{-M},…c_{M-1}]
            μ=eigvals(C)
            mask=abs.(abs.(μ).-1).<1e-6 # keep only those μ near the unit circle -> μ≈exp(iϕ)
            μ=μ[mask]
            θr=Base.angle.(μ)  # map back to real k = cen + rad * Re(μ)
            roots=cen.+rad*cos.(θr)
            sort!(roots)
            if !isempty(last_roots) &&  # stable two‐in‐a‐row, we’re done
                length(roots)==length(last_roots) &&
                all(abs.(roots.-last_roots).<rtol)
                return roots
             end
            last_roots=copy(roots) # still missing roots, these are the new roots for comparison
        end # not yet converged -> double M
        M*=2
    end
    print_error && error("No convergence for Mmax = $(Mmax) on [ $(a) , $(b) ] for Fourier coeff tol = $(Ftol)")
    return Vector{Complex{T}}()
end

"""
    subdivide_intervals(Nfunc::Function,a::T,b::T;target_count::Int=3,max_delta::Float64=0.1,tol_bisection::Float64=1e-8,tol_ΔN::Float64=1e-6) where {T<:Real}

Subdivide the interval `[a, b]` into smaller subintervals so that in each subinterval:

1. The estimated number of roots, `Nfunc(b_i) - Nfunc(a_i)`, is at most `target_count`.
2. The length of the subinterval `b_i - a_i` does not exceed `max_delta`.

This ensures Boyd’s method on each piece operates on only a few roots and on a well-resolved domain.

# Arguments
- `Nfunc::Function`: a monotonically increasing “root‐count” estimator, so that `Nfunc(x)`≈ number of roots in `(initial_a, x]`.
- `a::T`: full interval start.
- `b::T`: full interval end.

# Keyword Arguments
- `target_count::Int=5`: desired maximum number of roots per subinterval.
- `max_delta::Float64=0.1`: maximum allowed length of any subinterval.
- `tol_bisection::Float64=1e-8`: bisection tolerance for finding end‐points when the direct jump would exceed `target_count`.
- `tol_ΔN::Float64=1e-6`: Tolerance for level counting to prevent extremely small intervals at the ends.                      
# Returns
- `intervals::Vector{Tuple{Float64,Float64}}`: A list of `(a_i, b_i)` subintervals covering `[a,b]`, each satisfying the two bounds above.
"""
function subdivide_intervals(Nfunc::Function,a::T,b::T;target_count::Int=3,max_delta::Float64=0.1,tol_bisection::Float64=1e-8,tol_ΔN::Float64=1e-6) where {T<:Real}
    intervals=Vector{Tuple{T,T}}()
    x0=a
    N0=Nfunc(x0)
    while x0<b
        x_high=min(x0+max_delta,b) # cap the right by max_deltsa, needed for low ks
        ΔN=Nfunc(x_high)-N0 # N0 will iteratively change as the previous N_high
        if ΔN<=target_count+tol_ΔN
            push!(intervals,(x0,x_high)) # if estimated roots ≤ target_count, accept [x0, x_high]
            x0=x_high
            N0=Nfunc(x0)
        else # too many roots → bisect until we isolate ≈ target_count roots
            lo,hi=x0,x_high
            # define function whose zero is where N(x)-N0 = target_count
            g(x)=(Nfunc(x)-N0)-target_count
            @assert g(lo)≤0≤g(hi) # bracket: g(lo) ≤ 0, g(hi) ≥ 0, i.e. there is a root so no cycle. Must exist since we could not have gotten inside this else part
            while hi-lo>tol_bisection # bisection loop
                mid=(lo+hi)/2
                if g(mid)<0
                    lo=mid
                else
                    hi=mid
                end
            end
            x_mid=(lo+hi)/2 # midpoint is the end
            push!(intervals,(x0,x_mid))
            x0=x_mid
            N0=Nfunc(x0) # relabel the ending of this interval to be the start of the next one
        end
    end
    # Check last interval if too small due to rounding
    if abs(intervals[end][2]-intervals[end][1])<tol_ΔN
       deleteat!(intervals,length(intervals))
    end
    return intervals
end

###############################
#### END OF MAIN FUNCTIONS ####
###############################





#################
#### TESTING ####
#################

####################
#### POYLNOMIAL ####
####################

@info "Polyonimal"

N=5 # usually best if the number of roots is kept small, say N≈5 or max N=10. Use Weyl's law to determine [a,b] to have 1-10 roots, best 1-5 roots
a=1.0
b=1.1 # intervals should be kept small, say δ=0.1, for larger ones precision is still ok but missing roots start to emerge!
roots=a.+(b-a).*rand(N) # random real roots
sort!(roots)
f=x->prod(x.-roots) # Nth degree polynomial
xs=interval_roots_boyd(f,a,b;Ftol=1e-12,Mmax=512) # play around with the Ftol
xs_filtered=Complex{Float64}[]
push!(xs_filtered,xs[1])
for x in xs[2:end] 
    if abs(x-xs_filtered[end])>1e-8
        push!(xs_filtered,x)
    end
end
@show xs_filtered
# Compare to the exact roots #
println()
println("True roots:  ",sort(roots))
println()
println("Found roots: ",xs_filtered)
println()
println("Errors: ",abs.(xs_filtered.-sort(roots)))
println()

###################
#### COS & SIN ####
###################

@info "Cos"

a=0.0
b=1.0 # intervals should be kept small, say δ=0.1, for larger ones precision is still ok but missing roots start to emerge!
ω=5.0
f=x->cos(ω*x)

# generate true roots for comparison
roots=[]
k=0
while true
    global k
    x=(2k+1)*pi/(2*ω)
    if x>b
        break
    end
    push!(roots,x)
    k+=1
end

xs=interval_roots_boyd(f,a,b;Ftol=1e-12,Mmax=512) # play around with the Ftol
xs_filtered=Complex{Float64}[]
push!(xs_filtered,xs[1])
for x in xs[2:end] 
    if abs(x-xs_filtered[end])>1e-8
        push!(xs_filtered,x)
    end
end

@show xs_filtered

# Compare to the exact roots #
println()
println("True roots:  ",sort(roots))
println()
println("Found roots: ",xs_filtered)
println()
println("Errors: ",abs.(xs_filtered.-sort(roots)))
println()

#############################
#### INTERVAL SUBDIVISON ####
#############################

tol=1e-6 # for checking interval bound equalities, cannot expect machine precision equality
# helper to check tuple equality
function tuples_approx(A::Vector{Tuple{T,T}},B::Vector{Tuple{T,T}},atol) where {T<:Real}
    @assert length(A)==length(B)
    for ((a1,a2),(b1,b2)) in zip(A,B)
      if abs(a1-b1)>atol || abs(a2-b2)>atol
        return false
      end
    end
    return true
end
function tuples_approx(A::Tuple{T,T},B::Tuple{T,T},atol) where {T<:Real}
    (a1,a2),(b1,b2)=A,B
    if abs(a1-b1)>atol || abs(a2-b2)>atol
        return false
    end
    return true
end

# (1) Trivial case: if Nfunc≡0 there are never any “roots,” so
# no need to cut up the interval:
ints=subdivide_intervals(x->0.0,0.0,1.0;target_count=3,max_delta=2.0)
@test tuples_approx(ints,[(0.0,1.0)],tol)

# (2) A linear root count N(x)=x on [0,10], with target_count=2, should carve it into chunks of length ≈2, up to max_delta=5:
ints=subdivide_intervals(x->x,0.0,10.0;target_count=2,max_delta=5.0)
# EXPECT: [0–2], [2–4], [4–6], [6–8], [8–10]
@test length(ints)==5
@test tuples_approx(ints[1],(0.0,2.0),tol)
@test tuples_approx(ints[end],(8.0,10.0),tol)

# (3) A “staircase” Nfunc(x)=floor(10x) on [0,1], target_count=3, floor(10x) steps by 1 every 0.1 in x, so you can fit up to 3 steps in Δx=0.3, but max_delta=0.1 will break it into 10 pieces:
ints=subdivide_intervals(x->floor(10x),0.0,1.0;target_count=3,max_delta=0.1)
@test length(ints)==10
@test all(b-a<=0.1+tol for (a,b) in ints) # tolerance to roundoff

# (4) Check that the union of all the little pieces exactly covers [a,b]:
a,b=0.2,1.7
ints=subdivide_intervals(x->2x,a,b;target_count=1,max_delta=0.5)
# N(x)=2x so ΔN≤1 ⇒ Δx≤0.5, max_delta=0.5 the same
# we should get [(0.2,0.7),(0.7,1.2),(1.2,1.7)]
@test isapprox(ints[1][1],a;atol=tol)   # first interval’s left endpoint == a
@test isapprox(ints[end][2],b;atol=tol) # last interval’s right endpoint == b
