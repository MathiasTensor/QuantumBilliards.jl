using GSL # for the derivatives of the Bessel functions
using Bessels # for the Bessel functions themselves due to GSL underflow
using QuadGK # for the radial integrand of the conformal mapping matrix
using FastGaussQuadrature # for the radial integrand which is smooth on [0,1]
using LinearAlgebra
using CairoMakie # for plotting, especially tricontourf
using ProgressMeter # useful for progress bars
using SparseArrays # for A matrix and SparseMatrixCSC routines due to block diagonal form of A
using Arpack # for large matrices very useful due to lower 25% of the eigenvalues correct
using ForwardDiff # numerical differentiation, used in area, perimeter and boundary function calculations
using PyCall # for the symbolic routines to calculate the a(r,φ) coefficients
using CircularArrays # for the husimi function construction where the window at beggining and end overlaps with indexes normally out of bounds
using CSV,DataFrames # for saving the eigenvalues and/or their differences. If one does not need to save the eigenvalues one can comment this out and comment out/remove the comparison of the eigenvalues part in the examples section.

# For Symbolic routines below
sympy=pyimport("sympy") # use Python's sympy API
z,w=sympy.symbols("z w") # symbol for z, w (both complex)

#TODO New solve function which uses ARPACK to solve say 2000 eignevalues if a sigma inverse shifted matrix where we vary the sigma so we can get the lowest 25% of levels of A with iterating this process and merging the results.


#=
CONFORMAL MAPPING METHOD
This module provides functions for conformal mapping of the unit disk via a complex Polynomial.
The mapping is defined by a polynomial f(z) = coeffs[1] + coeffs[2]*z + … + coeffs[end]*z^(end-1).

# For usage of this method see the below exhausting EXAMPLE section with detailed comments and self-explanatory code (altough there are comments in the code as well).
=#






#################################################################
####### SYMBOLIC FUNCTIONS FOR GENERAL CONFORMAL MAPPINGS #######
#################################################################


"""
    make_polynomial(coeffs::Vector{PyObject}) :: PyObject

Construct a Sympy polynomial f_z from a list of coefficients.

# Arguments
- `coeffs::Vector{PyObject}`  
  A vector of Sympy numbers or expressions.  
  `coeffs[i]` is the coefficient of `z^(i-1)`.

# Returns
- `f_z::PyObject`  
  The Sympy expression for coeffs[1] + coeffs[2]*z + coeffs[3]*z^2 + … + coeffs[end]*z^(length(coeffs)-1)
"""
@inline function make_polynomial(coeffs::Vector{PyObject})
    f_z=sympy.Integer(0)
    for (i,c) in enumerate(coeffs) # build f_z by summing c[i]*z^(i)
        f_z=f_z+c*z^i
    end
    return f_z
end

"""
    integrand_coeff_dictionary(coeffs::Vector{PyObject}) :: Vector{Pair{Tuple{Int,Int},PyObject}}

Given the coefficients of f(z), compute the nonzero (m,n) → a_{m,n} terms
in the expansion of |f′(r e^{iφ})|² as a sum of r^n cos(mφ) modes.

# Arguments
- `coeffs::Vector{PyObject}`  
  Sympy coefficients defining f(z) = coeffs[1] + coeffs[2]*z + … + coeffs[end]*z^(end-1).

# Returns
- `pairs::Vector{Pair{(m,n),a}}`  
  A sorted vector of pairs `((m,n) => a)` where `a` is the Sympy coefficient
  multiplying `r^n * cos(m*φ)`. Only nonzero coefficients are included, ordered by m.
"""
function integrand_coeff_dictionary(coeffs::Vector{PyObject})
    f_z=make_polynomial(coeffs)
    fprime_z=sympy.diff(f_z,z) # differentiate df/dz
    fprime_rφ=fprime_z.subs(z,r*sympy.exp(im*φ)) #  substitute z → r*exp(iφ)
    a=sympy.simplify(sympy.conjugate(fprime_rφ)*fprime_rφ) # a(r,φ) = |f'|^2
    a=sympy.expand(a)
    a=sympy.expand_complex(a) # rewrite exp(iφ)→cosφ+ i sinφ
    a=sympy.trigsimp(a) # clean up trig expressions to have only cos
    @info "a(r,φ) = $a"
    all_symbols=Vector{PyObject}()
    for sym in a.free_symbols # to avoid having to declare Symbols top‐level
        push!(all_symbols,sym)
    end
    push!(all_symbols,sympy.pi) # add π to the free symbols
    function extract_an(a_expr::PyObject,M::Int) # function to extract aₙ(r) via orthogonality
        coefs=Dict{Int,PyObject}()
        I0=sympy.integrate(a_expr,(φ,0,2*sympy.pi))/(2*sympy.pi) # a₀(r)
        coefs[0]=sympy.simplify(I0)
        for n in 1:M # aₙ(r) for n=1…M
            In=sympy.integrate(a_expr*sympy.cos(n*φ),(φ,0,2*sympy.pi))/sympy.pi
            coefs[n]=sympy.simplify(In)
        end
        return coefs
    end
    function clean!(coefs::Dict{Int,PyObject};tolerance=1e-10) # clean 1e-15 junk from integration via nsimplify
        for (n,expr) in coefs
            coefs[n]=sympy.nsimplify(expr,all_symbols;tolerance=tolerance)  # tell nsimplify we only expect r,λ,γ,δ,π,... in the final answer
        end
    end
    M=length(coeffs) # heurstic n = 1:M is actully the n in cos(nφ) !!!
    radial=extract_an(a,M)
    clean!(radial) # numerical integration artifacts
    radial_coefs=Dict{Int,Vector{PyObject}}()
    for (n, a_expr) in radial
        P=sympy.Poly(a_expr,r)  # build the univariate polynomial in r
        coeffs_high2low=P.all_coeffs() # all_coeffs() returns [c_deg, c_{deg-1}, …, c_0]
        coeffs=reverse(coeffs_high2low) # reverse to get [c_0, c_1, …, c_deg]
        radial_coefs[n]=coeffs
    end
    coeff_dict=Dict{Tuple{Int,Int},PyObject}()
    for (m,coeffs) in radial_coefs
        for (k,c) in enumerate(coeffs)
            coeff_dict[(m,k-1)]=c  # m is the cos‐harmonic index, k-1 is the power of r
        end
    end
    pairs=collect(coeff_dict)
    sort!(pairs,by=kv->kv[1]) # sort by the key (the first element of each pair)
    return pairs
end

"""
    make_A_mask(pairs::Vector{Pair{Tuple{Int,Int},PyObject}},ordered_modes::Vector{Tuple{Int,Int}}) :: Tuple{Matrix{Bool}, Vector{Pair{Tuple{Int,Int},PyObject}}}

From the list of (m,n)→a coefficients and a list of (k,l) modes, build:

1. `mask::Matrix{Bool}`: an N×N boolean array where mask[i,j] is true
   if |k_i - k_j| matches one of the m values with nonzero coefficient.
2. `pairs_reduced::Vector{Pair}`: the subset of `pairs` with nonzero a.

# Arguments
- `pairs::Vector{Pair{(m,n),PyObject}}`  
  All extracted coefficient pairs.
- `ordered_modes::Vector{Tuple{Int,Int}}`  
  The list of (k,l) mode indices in the order they will appear in the matrix.

# Returns
- `(mask, pairs_reduced)`  
  - `mask::Matrix{Bool}`: true entries indicate allowed nonzero matrix blocks.  
  - `pairs_reduced::Vector{Pair}`: filtered coefficient list.
"""
@inline function make_A_mask(pairs::Vector{Pair{Tuple{Int,Int},PyObject}},ordered_modes::Vector{Tuple{Int,Int}})
    N=length(ordered_modes)
    mask=fill(false,N,N)
    pairs_reduced=Vector{Pair{Tuple{Int,Int},PyObject}}()
    zero_sym=sympy.Integer(0)
    Ms=Set{Int}()
    for p in pairs # figure out which angular‐difference bands m survive
        m,_=p.first
        if p.second!=zero_sym
            push!(Ms,m)
            push!(pairs_reduced,p)
        end
    end
    for i in 1:N # set mask[i,j]=true whenever |k_i - k_j| in Ms
        k_i,_=ordered_modes[i]
        for j in 1:N
            k_j,_=ordered_modes[j]
            if abs(k_i-k_j) in Ms # if the angular difference is in Ms
                mask[i,j]=true
            end
        end
    end
    return mask,pairs_reduced
end

@inline function evaluate_pairs(pairs::Vector{Pair{Tuple{Int,Int},PyObject}},param_vals::Dict{PyObject,T}) where {T<:Real}
    numeric_pairs=Vector{Tuple{Tuple{Int,Int},T}}(undef,length(pairs))
    for (i,((m,n),a_sym)) in enumerate(pairs)
        tmp=a_sym
        for (sym,val) in param_vals
            tmp=tmp.subs(sym,val)
        end
        a_num=tmp.evalf()
        numeric_pairs[i]=((m,n),a_num)
    end
    return numeric_pairs
end








###########################################
####### CONFORMAL MAPPING FUNCTIONS #######
###########################################


"""
    delta(m::Int, n::Int) :: Int

Return 1 if `m == n`, otherwise 0. Used as the Kronecker delta function.

# Arguments
- `m::Int`: first index  
- `n::Int`: second index  

# Returns
- `1` if `m == n`, else `0`
"""
function delta(m::Int,n::Int)
    if m==n
        return 1
    else
        return 0
    end
end

"""
    angular_radial_order(modes::Vector{Tuple{Int,Int}}) :: (Vector{Tuple{Int,Int}}, Dict{Tuple{Int,Int},Int})

Given a list of (k,l) mode indices, produce a standardized ordering that
cycles k=0,–1,+1,–2,+2,…, each with its l sorted ascending.

# Arguments
- `modes::Vector{Tuple{Int,Int}}`: unsorted list of (k,l) pairs

# Returns
- `ordered_modes::Vector{Tuple{Int,Int}}`: modes in the sequence
    (0,ℓ…),(–1,ℓ…),(+1,ℓ…),(–2,ℓ…),…  
- `idx::Dict{Tuple{Int,Int},Int}`: map from each (k,l) to its 1-based position
"""
@inline function angular_radial_order(modes::Vector{Tuple{Int,Int}})
    groups=Dict{Int,Vector{Int}}() # Group l–values by angular index k, that is create an empty dictionary whose keys will be the different k values, and whose values will be vectors of the corresponding l values.
    for (k,l) in modes
        push!(get!(groups,k,Int[]),l) # a bit hacky, get!(groups, k, Int[]) says: if groups already has a key k, return its vector; otherwise create an empty Int[], insert it under k, and return that. In either case, push! adds l to the vector that either contained some l values from the previous modes or an empty vector and for this k this is the first l value.
    end # groups[k] is now the (unsorted) list of all l’s paired with that k. So we sort the l’s for each k.
    for v in values(groups)
        sort!(v)
    end
    ang_seq=Int[] # start with an empty list of k-values
    if haskey(groups,0) # if we have k=0 stick 0 into the angular part since it has no +/- part (l=0)
        push!(ang_seq,0)
    end
    Nmax=maximum(abs.(keys(groups))) # Find the largest magnitude of any angular index (if k = –3,-2,…,+2,+3 -> Nmax = 3)
    for d in 1:Nmax # for each d = 1, 2,…, Nmax-1, Nmax we check if we have any k with that value k+d or k–d in groups and add it the angular sequence k list
        if haskey(groups,-d) 
            push!(ang_seq,-d)
        end
        if haskey(groups,+d)
            push!(ang_seq,+d)
        end
    end
    ordered=Tuple{Int,Int}[]  # Assemble the ordered (k,l) list. We do this by iterating over the angular sequence and for each k in the sequence we iterate over the l values in groups[k] (all l values possible for that k) and push them into the ordered list.
    for k in ang_seq, l in groups[k] 
        push!(ordered,(k,l))
    end
    idx=Dict{Tuple{Int,Int},Int}((mode=>i) for (i,mode) in enumerate(ordered)) # idx[(k,l)] to find mode that lives in i in the flattened orderder modes. This allows one to find where the (k,l,k',l') tetradic index lives in the matrix via:
    # i = idx[(k,l)]
    # j = idx[(k′,l′)]
    # A[i,j] = …
    return ordered,idx
end

"""
    all_bessel_J_nu_roots(kmin::Int, kmax::Int,lmin::Int, lmax::Int) :: Dict{Tuple{Int,Int}, T}

Compute the first `lmax` positive roots of the Bessel function Jₖ(x)
for each integer order k in `kmin:kmax`.

# Arguments
- `kmin::Int`, `kmax::Int`: angular orders  
- `lmin::Int`, `lmax::Int`: radial indices

# Returns
- `roots_dict::Dict{(k,l) => root}`: map from (k,l) to the l-th zero of Jₖ
"""
function all_bessel_J_nu_roots(kmin::Int,kmax::Int,lmin::Int,lmax::Int)
    k_range=kmin:kmax
    l_range=lmin:lmax
    all_roots=Matrix{Float64}(undef,length(k_range),length(l_range))
    @showprogress desc="Calculating all roots" Threads.@threads for k in kmin:kmax
        for l in lmin:lmax 
            if k==0
                all_roots[k,l]=sf_bessel_zero_J0(l)
            else
                all_roots[k,l]=sf_bessel_zero_Jnu(k,l)
            end
        end
    end
    roots_dict=Dict{Tuple{Int,Int},eltype(all_roots[1,1])}((k,l)=>all_roots[k,l] for k in k_range for l in l_range)
    return roots_dict
end

#TODO Use Steer's reccurence formula here
"""
    bessel_j_nu_der(k::Int, x::T) where {T<:Real} :: T

Evaluate the derivative Jₖ′(x) of the Bessel function of order k at x,
using the identity Jₖ′(x) = k/x*Jₖ(x) − Jₖ₊₁(x).

# Arguments
- `k::Int`: order  
- `x::Real`: evaluation point

# Returns
- `Jₖ′(x)::Real`
"""
function bessel_j_nu_der(k::Int,x::T) where {T<:Real}
    return k/x*Bessels.besselj(k,x)-Bessels.besselj(k+1,x)
end

"""
    normalization(k::Int, l::Int,roots::Dict{Tuple{Int,Int},T}) where {T<:Real} :: T

Compute the L²-normalization constant for the basis function
Jₖ(αₖₗ r) on the unit disk, where αₖₗ is the l-th zero of Jₖ.

# Arguments
- `k::Int`, `l::Int`: Bessel order and zero index  
- `roots::Dict{(k,l)=>α}`: precomputed zeros

# Returns
- `N::Real`: normalization factor so ∫₀¹∫₀²π |N Jₖ(αₖₗ r)e^{ikφ}|² r dr dφ = 1
"""
function normalization(k::Int,l::Int,all_roots::Dict{Tuple{Int,Int},T}) where {T<:Real}
    inv_sqrt_pi=1/sqrt(pi)
    if k==0
        return inv_sqrt_pi*1/bessel_j_nu_der(k,all_roots[k,l])
    else
        return sqrt(2)*inv_sqrt_pi*1/bessel_j_nu_der(k,all_roots[k,l])
    end
end

"""
    angular_integral_cos(m::Int, k::Int, kp::Int)

Compute ∫₀²π cos(mφ) cos(kφ) cos(kpφ) dφ analytically:
- if m=0: returns 2π δₖ,ₖₚ or π δₖ,ₖₚ when k>0
- else: (π/2)[δₘ,|k−kp| + δₘ,k+kp]

# Arguments
- `m::Int`: harmonic index  
- `k, kp::Int`: angular mode orders

# Returns
- value of the angular integral as a `T`
"""
@inline function angular_integral_cos(m::Int,k::Int,kp::Int)
    if m==0
        if k==0
            return 2*pi*delta(k,kp)
        else
            return pi*delta(k,kp)
        end
    else
        return (pi/2)*(delta(m,abs(k-kp))+delta(m,k+kp))
    end
end

"""
    angular_integral_sin(m::Int, k::Int, kp::Int)

Compute ∫₀²π cos(mφ) · sin(kφ) · sin(kpφ) dφ analytically:

- if m = 0: returns π·δₖ,ₖₚ  
- else: (π/2)·[δₘ,|k−kp| − δₘ,k+kp]

# Arguments
- `m::Int`: harmonic index  
- `k, kp::Int`: angular mode orders

# Returns
- value of the angular integral as a `Float64`
"""
@inline function angular_integral_sin(m::Int,k::Int,kp::Int)
    if m==0 # ∫ cos(0·x) sin(kx) sin(kp x) dx = ∫ sin(kx) sin(kp x) dx = π δ_{k,kp}
        return pi*delta(k,kp)
    else 
        return pi/2*(delta(m,abs(k-kp))-delta(m,k+kp))
    end
end

"""
    angular_integral(m::Int, k::Int, kp::Int, symmetry::Symbol = :EVEN)

Dispatch to the appropriate angular overlap integral based on basis symmetry:

- `symmetry == :EVEN` uses `angular_integral_cos` (cos–cos–cos block)
- `symmetry == :ODD`  uses `angular_integral_sin` (cos–sin–sin block)

# Arguments
- `m::Int`— harmonic index  
- `k::Int`,`kp::Int`— angular mode orders  
- `symmetry::Symbol`—`:EVEN` or `:ODD`

# Returns
- value of the integral ∫₀²π cos(mφ)·[cos(kφ)cos(kpφ) or sin(kφ)sin(kpφ)] dφ  
"""
@inline function angular_integral(m::Int,k::Int,kp::Int,symmetry::Symbol)
    if symmetry==:EVEN
        return angular_integral_cos(m,k,kp)
    else
        return angular_integral_sin(m,k,kp)
    end
end

"""
    integrand_rn(n::Int,k::Int, l::Int,kp::Int, lp::Int,roots::Dict{Tuple{Int,Int},T}) where {T<:Real} :: T

Gives the functional form the radial overlap integrand `f(r) = r^n Jₖ(αₖₗ r) Jₖₚ(αₖₚ,ₗₚ r)`.

# Arguments
- `n::Int`: power of r  
- `(k::Int,l::Int)`, `(kp::Int,lp::Int)`: Bessel orders and zero indices  
- `roots::Dict{Tuple{Int,Int},T}`: map of zeros α

# Returns
- numerical value of the integral as `T`
"""
function integrand_rn(n::Int,k::Int,l::Int,kp::Int,lp::Int,all_roots::Dict{Tuple{Int,Int},T}) where {T<:Real}
    f(r)=r^n*Bessels.besselj(k,all_roots[k,l]*r)*Bessels.besselj(kp,all_roots[kp,lp]*r)
    return f
end

"""
    integrate_rn(n::Int,k::Int, l::Int,kp::Int, lp::Int,roots::Dict{Tuple{Int,Int},T}) where {T<:Real} :: T

Compute the radial overlap integral
∫₀¹ r^n Jₖ(αₖₗ r) Jₖₚ(αₖₚ,ₗₚ r) dr via adaptive quadrature.

# Arguments
- `n::Int`: power of r  
- `(k::Int,l::Int)`, `(kp::Int,lp::Int)`: Bessel orders and zero indices  
- `roots::Dict{Tuple{Int,Int},T}`: map of zeros α

# Returns
- numerical value of the integral as `T`
"""
function integrate_rn(n::Int,k::Int,l::Int,kp::Int,lp::Int,all_roots::Dict{Tuple{Int,Int},T}) where {T<:Real}
    f(r)=r^n*Bessels.besselj(k,all_roots[k,l]*r)*Bessels.besselj(kp,all_roots[kp,lp]*r)
    return quadgk(f,10*eps(T),1.0)[1]
end

"""
    construct_A(reduced_pairs::Vector{Pair{Tuple{Int,Int},PyObject}},ordered_modes::Vector{Tuple{Int,Int}},all_roots::Dict{Tuple{Int,Int},T},mask::Matrix{Bool};symmetry::Symbol=:EVEN,quadrature::Symbol=:GAUSS_KRONROD_ADAPTIVE) where {T<:Real}

Assemble the sparse symmetric matrix A for the conformal map,
using the pre-extracted (m,n) → aₘₙ pairs and a Boolean mask.

# Arguments
- `reduced_pairs`: list of nonzero `((m,n)=>a)` symbolic coefficients. These should be reduced so no zero coefficients ones are included 
- `ordered_modes`: list of (k,l) modes  
- `roots`: map of Bessel zeros  
- `mask[i,j]`: true if A[i,j] may be nonzero
- `param_vals`: dictionary of parameter values for the symbolic coefficients
- `symmetry`: `:EVEN` or `:ODD` for the angular integral. Must be then used the same for the wavefunction construction
- `quadrature::Symbol=:GAUSS_LEGENDRE_FIXED`: `:GAUSS_KRONROD_ADAPTIVE` or `:GAUSS_LEGENDRE_FIXED` for the quadrature method
- `n_q_MIN::Int=12000`: minimum number of quadrature points for the Gauss–Legendre quadrature.

# Returns
- `A::SparseMatrixCSC`: the assembled symmetric matrix
"""
function construct_A(reduced_pairs::Vector{Pair{Tuple{Int,Int},PyObject}},ordered_modes::Vector{Tuple{Int,Int}},all_roots::Dict{Tuple{Int,Int},T},mask::Matrix{Bool},param_vals::Dict{PyObject,T};symmetry::Symbol=:EVEN,quadrature::Symbol=:GAUSS_KRONROD_ADAPTIVE,n_q_MIN::Int=12000) where {T<:Real}
    @assert symmetry in [:EVEN,:ODD] "Symmetry must be :EVEN or :ODD"
    @info "All pairs: (cos(mϕ),r^n) -> aₘₙ: $(reduced_pairs)"
    ang_int(m::Int,k::Int,kp::Int)=angular_integral(m,k,kp,symmetry)
    N=length(ordered_modes)
    A=zeros(N,N)
    numeric_pairs=evaluate_pairs(reduced_pairs,param_vals)
    println("Numeric pairs: ",numeric_pairs)
    @info "Filled matrix: $(length(findall(x->x==true,mask))/prod(size(mask))*100) %"
    if quadrature==:GAUSS_KRONROD_ADAPTIVE
        @showprogress desc="Constructing matrix" Threads.@threads for i in eachindex(ordered_modes)
            k,l=ordered_modes[i]
            for j in i:N # only calculate j>i since A is symmetric
                kp,lp=ordered_modes[j]
                if !mask[i,j]
                    continue
                end
                Ni=normalization(k,l,all_roots)
                Nj=normalization(kp,lp,all_roots)
                S=(k==kp && l==lp ? one(T) : zero(T))
                for ((m,n),a) in numeric_pairs
                    if m==0 && n==0
                        continue
                    end
                    if isapprox(a,0.0)
                        continue
                    end
                    ang=ang_int(m,k,kp)  # angular piece
                    if isapprox(ang,0.0)
                        continue
                    end
                    R=integrate_rn(n+1,k,l,kp,lp,all_roots) # radial piece, n+1 due to dS=r*dr*dϕ
                    S+=a*ang*Ni*Nj*R
                end
                A[i,j]=(1/all_roots[kp,lp])^2*S
            end
        end
    elseif quadrature==:GAUSS_LEGENDRE_FIXED
        n_q=ceil(Int,10*maximum(values(all_roots))) # number of quadrature points is ≈ 10*max(roots)
        n_q=max(n_q,n_q_MIN) # at least n_q_MIN quadrature points, probably overkill
        @info "Number of G-L quadrature points: $n_q"
        numeric_pairs=[(mn,a) for (mn,a) in numeric_pairs if mn!=(0,0) && a!=0.0]
        xg,wg=gausslegendre(n_q) # Gauss–Legendre on [0,1]
        r_q=@. (xg+1)/2
        w_q=wg./2
        needed_ns=unique(n+1 for ((m,n),_) in numeric_pairs)
        Rpows=Dict{Int,Vector{T}}()
        for n1 in needed_ns
            Rpows[n1]=r_q.^n1
        end
        jobs=Vector{Tuple{Int,Int}}()
        for i in 1:N, j in i:N
            mask[i,j] && push!(jobs,(i,j))
        end
        idxs=unique(vcat([i for (i,_) in jobs], [j for (_,j) in jobs]))
        norm_vec=Vector{T}(undef,N)
        for i in idxs
            k,l=ordered_modes[i]
            norm_vec[i]=normalization(k,l,all_roots)
        end
        J_cache=Vector{Vector{T}}(undef,N)
        @showprogress desc="Calculating Bessel grid for G-L..." Threads.@threads for i in idxs
            k,l=ordered_modes[i]
            α=all_roots[(k,l)]
            J_cache[i]=Bessels.besselj.(k,α.*r_q)
        end
        @showprogress desc="Fast Gauss–Legendre..." Threads.@threads for (i,j) in jobs
            k,l=ordered_modes[i]
            kp,lp=ordered_modes[j]
            Ni,Nj=norm_vec[i],norm_vec[j]
            S=(k==kp && l==lp ? one(T) : 0.0)
            JJ=J_cache[i].*J_cache[j] # elementwise product of the two J‐vectors
            for ((m,n),a) in numeric_pairs
              ang=ang_int(m,k,kp)
              ang==0.0 && continue
              R=dot(Rpows[n+1].*JJ,w_q) # single BLAS dot for the radial integral
              S+=a*ang*Ni*Nj*R
            end
            A[i,j]=(1/all_roots[(kp,lp)])^2*S
        end
    else
        error("Quadrature method not implemented")
    end
    return sparse(Symmetric(A))
end

"""
    solve(A::SparseMatrixCSC{T,Int};fraction_of_levels::T=0.2,use_ARPACK::Bool=false) :: (Vector{T}, Matrix{T})

Compute the lowest eigenvalues and eigenvectors of A, using either dense or ARPACK.

# Arguments
- `A`: sparse symmetric matrix  
- `fraction_of_levels`: fraction of total modes to return. This is only applicable if use_ARPACK=true since otherwise the full dense matrix is used.
- `use_ARPACK`: if true, use `eigs`; otherwise use dense `eigen`

# Returns
- `eigenvalues::Vector{T}`: sorted ascending  
- `C::Matrix{T}`: corresponding eigenvectors as columns
"""
function solve(A::SparseMatrixCSC;fraction_of_levels=0.2,use_ARPACK=false)
    N=ceil(Int,size(A,1)*fraction_of_levels)
    if use_ARPACK
        vals,C=eigs(A,nev=N,which=:LR,maxiter=prod(size(A))) # Must use :LR since we want the lowest eigenvalues and therefore we need largest vals from eigs
    else
        vals,C=eigen(Matrix(A))
    end
    eigenvalues=@. 1.0/vals
    idxs=sortperm(eigenvalues)
    eigenvalues=eigenvalues[idxs]
    C=C[:,idxs]
    return eigenvalues[1:N],C[:,1:N]
end

"""
    ϕ(k::Int, l::Int,roots::Dict{Tuple{Int,Int},T},r::T, φ::T;symmetry::Symbol=:EVEN) :: {T<:Real}

Evaluate the disk basis function φ_{k,l}(r,φ) = Nₖₗ Jₖ(αₖₗ r)*[cos(kφ)|sin(kφ)].

# Arguments
- `k,l`: Bessel order and zero index  
- `roots`: map of Bessel zeros  
- `r,φ`: polar coordinates (r∈[0,1], φ∈[0,2π])  
- `symmetry`: `:EVEN` for cosine, `:ODD` for sine angular part

# Returns
- value of φ_{k,l}(r,φ) as `T`
"""
function ϕ(k::Int,l::Int,roots::Dict{Tuple{Int,Int},T},r::T,φ::T;symmetry::Symbol=:EVEN) where {T<:Real}
    j=roots[(k,l)]
    inv_pi=1/sqrt(pi)
    N=(k==0 ? inv_pi : sqrt(2)*inv_pi)/(0.5*bessel_j_nu_der(k,j))
    return N*Bessels.besselj(k,j*r)*(symmetry==:ODD ? sin(k*φ) : cos(k*φ))
end

"""
    make_inv_newton_fd(f::Function;tol::T=1e-12,maxiter::Int=1000) :: Function

Return a function `invf(w)` that approximates the inverse of `f(z)` via Newton’s method. This is now a LEGACY function that was used to invert the (x,y) Cartesian grid coordinates back to the domain D coordinates via f^-1.

# Arguments
- `f::Function`: mapping from z to w  
- `tol`: convergence tolerance for |Δxy|  
- `maxiter`: maximum Newton iterations

# Returns
- `invf::Function`: takes `w::Complex` and returns `z≈f⁻¹(w)` as `Complex`
"""
function make_inv_newton_fd(f::Function;tol=1e-12,maxiter=1000)
    function invf(w::Complex{<:Real})
        xy=[real(w),imag(w)] # represent z = x + i y as a real vector [x,y].
        function F(xy::Vector{T}) where {T<:Real} #  real‐valued residual F(xy) = [Re(f(z)-w), Im(f(z)-w)].
            z=xy[1]+im*xy[2]
            fz=f(z)-w
            return [real(fz),imag(fz)]
        end
        for _ in 1:maxiter # Newton loop
            Fv=F(xy)
            J=ForwardDiff.jacobian(F,xy) # eval root minimizer F and Jacobian F at current xy
            δ=J\Fv # solve J · δ = Fv
            xy.-=δ
            if norm(δ)<tol
                break
            end
        end
        return xy[1]+im*xy[2]
    end
    return invf
end

"""
    wavefunctions(C::Matrix{T},ordered_modes::Vector{Tuple{Int,Int}},roots::Dict{Tuple{Int,Int},T},f::Function;symmetry=:EVEN,Nr::Int=200,Nφ::Int=400) where {T<:Real}

Compute the wavefunctions on a polar grid in the conformal image f(D) of the unit disk. This is then used in plotting via DelaunayTriangulation.

# Arguments
- `C::Matrix{T}`: N×nstates eigenvector matrix  
- `ordered_modes::Vector{Tuple{Int,Int}}: list of (k,l) pairs in matrix order
- `roots::Dict{Tuple{Int,Int},T}`: map of Bessel zeros
- `conformal_mapping(w)::Function`: conformal map z=f(w)  
- `symmetry::Symbol`: `:EVEN` or `:ODD` for angular part, chooses cos or sin basis
- `Nr::Int=200`: Polar grid r spacing
- `Nφ::Int=400`: Polar grid φ spacing

# Returns
- `Zs_flat::Vector{Vector{T}}`: flattened wavefunction matrices for each eigenvector
- `Xs::Vector{T}`: x-coordinates of the grid points in f(D)
- `Ys::Vector{T}`: y-coordinates of the grid points in f(D)
- `xb::Vector{T}`: x-coordinates of the boundary curve for f(z)
- `yb::Vector{T}`: y-coordinates of the boundary curve for f(z)
"""
function wavefunctions(C::Matrix{T},ordered_modes::Vector{Tuple{Int,Int}},roots::Dict{Tuple{Int,Int},T},f::Function;symmetry=:EVEN,Nr::Int=200,Nφ::Int=400) where {T<:Real}
    rs=range(sqrt(eps(T)),one(T);length=Nr) # build flattened polar grid
    phs=range(zero(T),2*pi;length=Nφ+1)[1:end-1]
    P=Nr*Nφ
    rgrid=repeat(rs,inner=Nφ)
    phigrid=repeat(phs,outer=Nr)
    zgrid=rgrid.*cis.(phigrid) # build the domain D
    wgrid=f.(zgrid)  # push forward once from D -> f(D)
    Xs=reshape(real.(wgrid),Nr,Nφ) # reshape to cartesian grid in f(D)
    Ys=reshape(imag.(wgrid),Nr,Nφ) # reshape to cartesian grid in f(D)
    M=length(ordered_modes) # number of modes, same as length(col(C)), coeffs of eigenvector expansion
    B=Matrix{T}(undef,M,P) # build basis matrix B (M×P)
    @fastmath begin # we are far away from NaNs and Inf 
        @showprogress desc="Constructing wavefunction matrices..." Threads.@threads for m in 1:M
            k,l=ordered_modes[m] # (0,ℓ…),(–1,ℓ…),(+1,ℓ…),(–2,ℓ…),… constructed as eachcol(C)
            B[m,:].=ϕ.(k,l,Ref(roots),rgrid,phigrid;symmetry=symmetry)
        end
    end
    Zflat=abs.(B'*C) # BLAS: compute abs.(B' * C) => P×nstates
    nstates=size(C,2)
    Zs=Vector{Matrix{T}}(undef,nstates)
    for s in 1:nstates
        Zs[s]=reshape(Zflat[:,s],Nr,Nφ) # split into a Vector of Nr×Nφ matrices
    end
    # build boundary curve for plotting f(z)=f(e^{iφ}) in the unit disk
    φb=range(zero(T),2*pi;length=Nφ)
    zs=f.(cis.(φb))
    xb=real.(zs)
    yb=imag.(zs)
    Xs=vec(Xs) # flatten for DelaunayTriangulation
    Ys=vec(Ys) # flatten for DelaunayTriangulation
    Zs_flat=Vector{Vector{T}}(undef,length(Zs))
    for i in eachindex(Zs)
        Zs_flat[i]=vec(Zs[i]) # # flatten for DelaunayTriangulation every matrix
    end
    return Zs_flat,Xs,Ys,xb,yb
end

"""
    conformal_arclength(f::Function, ϕ::T) where {T<:Real} -> T

Compute the arclength along the image of the unit‐circle boundary under
the conformal map `f`, from φ=0 up to φ=ϕ:

    L(ϕ) = ∫₀^ϕ |d/dθ [ f(e^{iθ}) ]|  dθ.

# Arguments
- `f::Function`: mapping ℂ → ℂ (conformal map on the unit disk)  
- `ϕ::T`: end angle in [0,2π]  

# Returns
- `L::T`: the partial arclength from θ=0 to θ=ϕ
"""
function conformal_arclength(f::Function,ϕ::T) where {T<:Real}
    u(θ)=real(f(cis(θ)))
    v(θ)=imag(f(cis(θ)))
    du(θ)=ForwardDiff.derivative(u,θ)
    dv(θ)=ForwardDiff.derivative(v,θ)
    integrand(θ)=sqrt(du(θ)^2+dv(θ)^2)
    return quadgk(integrand,zero(T),ϕ)[1]
end

"""
    function boundary_function(C::Matrix{T},ordered_modes::Vector{Tuple{Int,Int}},roots::Dict{Tuple{Int,Int},T},f::Function;symmetry::Symbol=:EVEN,Nφ::Int=1000) where {T<:Real}

Compute the outward normal derivative ∂Ψ/∂n of each eigenfunction Ψₛ on
the physical boundary w = f(e^{iφ}), for φ sampled uniformly.

Internally we use
1.  ∂ₙΨ = (1/|f′(e^{iφ})|) · ∂ᵣψ(r,φ)|_{r=1},
2.  ∂ᵣΦₖₗ(r,φ)|_{r=1} = Nₖₗ·αₖₗ·Jₖ′(αₖₗ)·[cos(kφ)|sin(kφ)],
3.  a single M×nstates matrix–vector multiply per φ,
4.  `Threads.@threads` over φ,
5.  BLAS for `C_scaled' * ang`.

# Arguments
- `C::Matrix{T}`: each **column** is one eigenvector (length M)
- `ordered_modes::Vector{(k,l)}`: length‐M list of Bessel modes
- `roots::Dict{(k,l)=>α}`: αₖₗ = l-th zero of Jₖ
- `f::Function`: conformal map z ↦ f(z)
- `symmetry::Symbol`: `:EVEN` (cos) or `:ODD` (sin) angular part
- `Nφ::Int=1000`: how many φ‐points around the circle

# Returns
- `φs::Vector{T}` : grid of angles, length Nφ  
- `Ls::Vector{T}` : partial arclengths L(φs) of the boundary, length Nφ  
- `Bnd::Matrix{T}`: Nφ×nstates matrix of ∂Ψₛ/∂n at each φₖ
"""
function boundary_function(C::Matrix{T},ordered_modes::Vector{Tuple{Int,Int}},roots::Dict{Tuple{Int,Int},T},f::Function;symmetry::Symbol=:EVEN,Nφ::Int=1000) where {T<:Real}
    M,nstates=size(C)
    ks=Vector{Int}(undef,M) # ks in nrows of C
    αs=Vector{T}(undef,M) # roots in nrows of C
    Threads.@threads for i in 1:M
        k,l=ordered_modes[i]
        ks[i]=k
        αs[i]=roots[(k,l)]
    end
    radial_prefac=Vector{T}(undef,M) # precompute radial prefactors: Nₖₗ·αₖₗ·Jₖ′(αₖₗ)
    Threads.@threads for i in 1:M
        k=ks[i]
        α=αs[i]
        Nkl=normalization(k,Int(round((ordered_modes[i][2]))),roots)
        radial_prefac[i]=Nkl*α*bessel_j_nu_der(k,α)
    end
    # scale C by those prefactors once: M×nstates
    C_scaled=C.*radial_prefac[:,1]  # broadcast down each row
    φs=range(zero(T),2*pi;length=Nφ+1)[1:end-1] # sample φ and prep output
    Bnd=Matrix{Float64}(undef,Nφ,nstates)
    g(φ)=f(cis(φ)) # define g and its derivative g'(φ) = d/dφ f(e^{iφ})
    dg(φ)=ForwardDiff.derivative(g,φ)
    @showprogress desc="Calculating boundary function..." Threads.@threads for j in 1:Nφ
        φ=φs[j]
        fp=dg(φ)/(im*cis(φ)) # physical stretching |f'(e^{iφ})|
        norm_fac=abs(fp)
        if symmetry==:EVEN # build angular vector of length M
            ang=cos.(ks.*φ)
        else
            ang=sin.(ks.*φ)
        end
        # BLAS dψ/dr at r=1 for each state: length‐nstates = (C_scaled') * ang
        dpsidr=C_scaled' *ang
        @inbounds for s in 1:nstates # divide by |f'| to get ∂/∂n
            Bnd[j,s]=dpsidr[s]/norm_fac
        end
    end
    Ls=conformal_arclength.(f,φs)
    return φs,Ls,Bnd
end

"""
    plot_wavefuntions(Zs_flat::Vector{Vector{T}},Xs::Vector{T},Ys::Vector{T},xb::Vector{T},yb::Vector{T};width=500,height=500,max_col=5) where {T<:Real}

Plot the wavefunctions on a grid in the conformal image f(D) of the unit disk. This uses DelaunayTriangulation to plot the wavefunctions via tricontourf since the grid is not uniform.

# Arguments
- `Zs_flat::Vector{Vector{T}}`: flattened wavefunction matrices for each eigenvector
- `Xs::Vector{T}`: x-coordinates of the grid points in f(D)
- `Ys::Vector{T}`: y-coordinates of the grid points in f(D)
- `xb::Vector{T}`: x-coordinates of the boundary curve for f(z)
- `yb::Vector{T}`: y-coordinates of the boundary curve for f(z)
- `width::Int=500`: width of each plot in pixels
- `height::Int=500`: height of each plot in pixels
- `max_col::Int=5`: maximum number of plots per row
- `Nlevels::Int=10`: number of contour levels

# Returns
- `fig::Figure`: the figure containing all the wavefunction plots
"""
function plot_wavefuntions(Zs_flat::Vector{Vector{T}},Xs::Vector{T},Ys::Vector{T},xb::Vector{T},yb::Vector{T};width::Int=500,height::Int=500,max_col::Int=5,Nlevels::Int=10) where {T<:Real}
    nstates=length(Zs_flat)
    fig=Figure(size=(width*(max_col+2),height*(2+ceil(Int,nstates/max_col))),resolution=(width*(max_col+2),height*(2+ceil(Int,nstates/max_col))))
    r=1
    c=1
    @showprogress desc="Plotting wavefunctions..." for i in eachindex(Zs_flat)
        r=div(i-1,max_col)+1
        c=(i-1) % max_col+1
        ax=Axis(fig[r,c][1,1],width=width,height=height)
        # This triangulation is really annoying and slow but since non uniform xy grid I don't know of a better way
        #TODO Find a faster way to plot this...
        tr=tricontourf!(ax,Xs,Ys,abs.(Zs_flat[i]);levels=Nlevels,colormap=Reverse(:gist_heat))
        lines!(ax,xb,yb;color=:red,linewidth=2)
        Colorbar(fig[r,c][1,2],tr)
    end
    return fig
end

function antisym_vec(x)
    v=reverse(-x[2:end])
    return append!(v,x)
end

"""
    husimi_function(k::T,u::Vector{T},s::Vector{T},L::T;c=10.0,w=7.0) where {T<:Real}

Calculates the Husimi function on a grid defined by the boundary `s`. The logic is that from the arclengths `s` we construct the qs with the help of the width of the Gaussian at that k (width = 1/√k) and the parameter c (which determines how many q evaluations we will have per this width at k -> defines the step 1/(√k*c) so we will have tham many more q evaluations in peak). Analogously we do for the ps, where the step size (matrix size in ps direction) is range(0.0,1.0,step=1/(√k*c)) (we do only for half since symmetric with p=0 implies we can use antisym_vec(ps) to recreate the whole -1.0 to 1.0 range while also symmetrizing the Husimi function that is constructed from p=0.0 to 1.0 with the logic perscribed above). The w (how many sigmas we will take) is used in construction of the ds summation weights.
Comment: Due to the Poincare map being an involution, i.e. (ξ,p) →(ξ,−p) we construct only the p=0 to p=1 matrix and then via symmetry automatically obtain the p=-1 to p=0 also.
Comment: Original algorithm by Črt Lozej

# Arguments
- `k<:Real`: Wavenumber of the eigenstate.
- `u::Vector{<:Real}`: Array of boundary function values.
- `s::Vector{<:Real}`: Array of boundary points.
- `L<:Real`: Total length of the billiard boundary.
- `c<:Real`: Density of points in the coherent state peak (default: `10.0`).
- `w<:Real`: Width in units of `σ` (default: `7.0`).

# Returns
- `H::Matrix`: Husimi function matrix.
- `qs::Vector`: Array of position coordinates on the grid.
- `ps::Vector`: Array of momentum coordinates on the grid.
"""
function husimi_function(k::T,u::Vector{T},s::Vector{T},L::T;c=10.0,w=7.0) where {T<:Real}
    #c density of points in coherent state peak, w width in units of sigma
    #L is the boundary length for periodization
    #compute coherrent state weights
    N=length(s)
    sig=one(k)/sqrt(k) # width of the Gaussian
    x=s[s.<=w*sig]
    idx=length(x) # do not change order here
    x=antisym_vec(x)
    a=one(k)/(2*pi*sqrt(pi*k)) # normalization factor (Husimi not normalized to 1)
    ds=(x[end]-x[1])/length(x) # integration weight
    uc=CircularVector(u) # allows circular indexing
    gauss=@. exp(-k/2*x^2)*ds
    gauss_l=@. exp(-k/2*(x+L)^2)*ds
    gauss_r=@. exp(-k/2*(x-L)^2)*ds
    ps=collect(range(0.0,1.0,step=sig/c)) # evaluation points in p coordinate
    q_stride=length(s[s.<=sig/c])
    q_idx=collect(1:q_stride:N)
    push!(q_idx,N) # add last point
    qs=s[q_idx] # evaluation points in q coordinate
    H=zeros(typeof(k),length(qs),length(ps))
    for i in eachindex(ps)
        cs=@. exp(im*ps[i]*k*x)*gauss + exp(im*ps[i]*k*(x+L))*gauss_l + exp(im*ps[i]*k*(x-L))*gauss_r
        for j in eachindex(q_idx)
            u_w=uc[q_idx[j]-idx+1:q_idx[j]+idx-1] # window with relevant values of u
            h=sum(cs.*u_w)
            H[j,i]=a*abs2(h)
        end
    end
    ps=antisym_vec(ps)
    H_ref=reverse(H[:,2:end];dims=2)
    H=hcat(H_ref,H)
    return H,qs,ps
end

"""
    plot_wavefunctions_with_husimi(Zs_flat::Vector{Vector{T}},Xs::Vector{T},Ys::Vector{T},xb::Vector{T},yb::Vector{T},ks::Vector{T},Ls::Vector{T},Bnd::Matrix{T};plot_boundary_function::Bool=false,width::Int=500,height::Int=500,max_col::Int=5,Nlevels::Int=10) where {T<:Real}

Plot a gallery of conformally-mapped quantum wavefunctions alongside their
boundary Husimi distributions.

For each eigenstate:
1. Uses `tricontourf!` on the non-uniform grid `(Xs, Ys)` with values
   `abs.(Zs_flat[i])`, drawing the conformal billiard boundary `(xb, yb)` in red.
2. Computes the Husimi function via `husimi_function(ks[i], Bnd[:,i], Ls, maximum(Ls))`
   and displays it as a `heatmap!` on the conjugate Poincaré section.
3. Optionally, if `plot_boundary_function` is `true`, appends a line plot of
   the normal derivative `∂Ψ/∂n` versus boundary arclength `Ls` below each pair.

# Arguments
- `Zs_flat::Vector{Vector{T}}`: Flattened wavefunction values for each eigenstate on the Delaunay grid.
- `Xs, Ys::Vector{T}`: x- and y-coordinates of the scattered grid points in the physical billiard.
- `xb, yb::Vector{T}`: x- and y-coordinates of the billiard boundary curve.
- `ks::Vector{T}`: Wave-numbers k_s for each state, used to build Husimi widths.
- `Ls::Vector{T}`: Cumulative boundary arclengths at which `Bnd` is sampled.
- `Bnd::Matrix{T}`: Normal-derivative values ∂Ψ/∂n on the boundary.

# Keyword Arguments
- `plot_boundary_function::Bool=false`: If `true`, plot ∂Ψ/∂n arclength beneath each Husimi map.
- `width, height::Int=500`: Pixel size for each subplot.
- `max_col::Int=5`: Maximum number of wavefunction/Husimi pairs per row.
- `Nlevels::Int=10`: Number of contour levels for the `tricontourf!` wavefunction plot.

# Returns
- `fig::Figure`: A `CairoMakie.Figure` containing a grid of wavefunction (left) and Husimi
  section (right) panels, with optional boundary-function traces below.
"""
function plot_wavefunctions_with_husimi(Zs_flat::Vector{Vector{T}},Xs::Vector{T},Ys::Vector{T},xb::Vector{T},yb::Vector{T},ks::Vector{T},Ls::Vector{T},Bnd::Matrix{T};plot_boundary_function::Bool=false,width::Int=500,height::Int=500,max_col::Int=5,Nlevels::Int=10) where {T<:Real}
    nstates=length(Zs_flat)
    fig=Figure(size=(2*width*(max_col+2),2*height*(2+ceil(Int,nstates/max_col))),resolution=(2*width*(max_col+2),2*height*(2+ceil(Int,nstates/max_col))))
    r=1
    c=1
    @showprogress desc="Plotting wavefunctions..." for i in eachindex(Zs_flat)
        r=div(i-1,max_col)+1
        c=(i-1) % max_col+1
        ax=Axis(fig[r,c][1,1],width=width,height=height)
        # This triangulation is really annoying and slow but since non uniform xy grid I don't know of a better way
        #TODO Find a faster way to plot this...
        tr=tricontourf!(ax,Xs,Ys,abs.(Zs_flat[i]);levels=Nlevels,colormap=Reverse(:gist_heat))
        lines!(ax,xb,yb;color=:red,linewidth=2)
        H,qs,ps=husimi_function(ks[i],Bnd[:,i],Ls,maximum(Ls))
        ax_h=Axis(fig[r,c][1,2],width=width,height=height,xlabel=L"s",ylabel=L"p")
        heatmap!(ax_h,qs,ps,H;colormap=Reverse(:gist_heat),colorrange=(0.0,maximum(H)))
        if plot_boundary_function
            ax_b=Axis(fig[r,c][2,1:2],width=2*width,height=height/2)
            lines!(ax_b,Ls,Bnd[:,i],xlabel=L"φ",ylabel=L"∂/∂n Ψₛ",color=:blue,linewidth=2)
        end
    end
    return fig
end






##############################
#### HELPERS AND WRAPPERS ####
##############################

"""
    A_construction_wrapper(kmin::Int, kmax::Int,lmin::Int, lmax::Int,coeffs::Vector{PyObject},param_vals::Dict{PyObject,T},symmetry::Symbol) where {T<:Real} :: (SparseMatrixCSC{T,Int},Vector{Tuple{Int,Int}},Dict{Tuple{Int,Int},Int},Dict{Tuple{Int,Int},T})

Build the Galerkin matrix A for a given conformal map specified by its polynomial
coefficients and parameter values.

# Arguments
- `kmin, kmax::Int`: minimum and maximum angular orders for Bessel modes  
- `lmin, lmax::Int`: minimum and maximum radial indices for Bessel zeros  
- `coeffs::Vector{PyObject}`: symbolic coefficients of the map polynomial  
- `param_vals::Dict{PyObject,T}`: mapping from each Sympy symbol to its numeric value  
- `symmetry::Symbol`: `:EVEN` for cosine basis or `:ODD` for sine basis  
- `quadrature::Symbol=:GAUSS_KRONROD_ADAPTIVE`: `:GAUSS_KRONROD_ADAPTIVE` or `:GAUSS_LEGENDRE_FIXED` for the quadrature method
- `n_q_MIN::Int=12000`: minimum number of quadrature points for the Gauss–Legendre quadrature.

# Returns
A tuple `(A, ordered_modes, idx, all_roots)` where:
- `A::SparseMatrixCSC{T,Int}`: assembled symmetric matrix  
- `ordered_modes::Vector{Tuple{Int,Int}}`: list of (k,l) pairs in matrix order  
- `idx::Dict{Tuple{Int,Int},Int}`: map from (k,l) → its row/column index  
- `all_roots::Dict{Tuple{Int,Int},T}`: precomputed Bessel zeros for each (k,l)
- `pairs_reduced::Vector{Pair{Tuple{Int,Int},PyObject}}`: list of nonzero pairs numerically evaluated
"""
function A_construction_wrapper(kmin::Int,kmax::Int,lmin::Int,lmax::Int,coeffs::Vector{PyObject},param_vals::Dict{PyObject,T},symmetry::Symbol;quadrature::Symbol=:GAUSS_KRONROD_ADAPTIVE,n_q_MIN::Int=12000) where {T<:Real}
    modes=[(k,l) for k in kmin:kmax for l in lmin:lmax]
    ordered_modes,idx=angular_radial_order(modes)
    pairs=integrand_coeff_dictionary(coeffs)
    A_mask,pairs_reduced=make_A_mask(pairs,ordered_modes)
    all_roots=all_bessel_J_nu_roots(kmin,kmax,lmin,lmax)
    A=construct_A(pairs_reduced,ordered_modes,all_roots,A_mask,param_vals,symmetry=symmetry,quadrature=quadrature,n_q_MIN=n_q_MIN)
    return A,ordered_modes,idx,all_roots,pairs_reduced
end

"""
    plot_sparsity_pattern!(ax::Axis, A::SparseMatrixCSC)

Display the sparsity pattern of a sparse matrix by plotting a point at each nonzero entry.

# Arguments
- `ax::Axis`: the Makie axis on which to draw  
- `A::SparseMatrixCSC`: sparse matrix whose nonzeros will be shown  

# Returns
- `nothing`: modifies the axis in place
"""
function plot_sparsity_pattern!(ax::Axis,A::SparseMatrixCSC)
    I,J,_=findnz(A)
    scatter!(ax,J,I) # scatter the i,j ∈ I, J
    return nothing
end

"""
    area(f::Function)

Estimate the area of the image f({|z| ≤ 1}) by the boundary integral

    A = ½ ∫₀²π [ u(θ) * v′(θ)  –  v(θ) * u′(θ) ] dθ,

where f(e^{iθ}) = u(θ) + i·v(θ).

# Arguments
- `f::Function`: Conformal map z ↦ f(z), defined on the unit circle.

# Returns
- `A::Float64`: Estimated area enclosed by f(|z| = 1).
"""
function area(f::Function)
    u(θ)=real(f(cis(θ)))
    v(θ)=imag(f(cis(θ)))
    du(θ)=ForwardDiff.derivative(u,θ)
    dv(θ)=ForwardDiff.derivative(v,θ)
    integrand(θ)=u(θ)*dv(θ)-v(θ)*du(θ)  # integrand for Green's theorem
    I=quadgk(integrand,0.0,2*pi)[1]
    return 0.5*I
end

"""
    perimeter(f::Function)

Estimate the perimeter of the image of the unit circle under the conformal map `f(z)`
by computing

    L = ∫₀²π |d/dθ f(e^{iθ})| dθ.

We write f(e^{iθ}) = u(θ) + i·v(θ) and note

    |d/dθ f(e^{iθ})| = sqrt( (u′(θ))² + (v′(θ))² ).

# Arguments
- `f::Function`: mapping `z -> f(z)` on the complex plane

# Returns
- `L::Real`: estimated perimeter
"""
function perimeter(f::Function)
    return conformal_arclength(f,2*pi)
end

"""
    unfold(f::Function, ks::Vector{T}) where {T<:Real} -> Vector{T}

Compute the unfolded “cumulative level number” N(k) for a set of wave-numbers `ks`
using the leading Weyl term for a billiard with conformal map `f`.

N(k) is defined by

    N(k) = [A * k^2  –  L * k]  / (4π)

where
- A is the area of the billiard (computed via `area(f)`),
- L is its perimeter (computed via `perimeter(f)`).

# Arguments
- `f::Function`: Conformal map from the unit disk to the billiard domain.
- `ks::Vector{T}`: Monotonically increasing vector of wave-numbers k (T<:Real).

# Returns
- `N_ks::Vector{T}`  
  The unfolded level counts N(k) evaluated at each entry of `ks`.
"""
function unfold(f::Function,ks::Vector{T}) where {T<:Real}
    A=area(f)
    L=perimeter(f)
    N_ks=(A*ks.^2 .- L .* ks)./(4π)
    return N_ks
end


"""
    calculate_spacings(f::Function, ks::Vector{T}) where {T<:Real} -> Vector{T}

Compute the nearest-neighbor spacings of unfolded wave-numbers.

This function first calls `unfold(f, ks)` to obtain the unfolded sequence N(k),
sorts it, and then returns the differences between successive levels.

# Arguments
- `f::Function`: Conformal map from the unit disk to the billiard domain.
- `ks::Vector{T}` :Vector of wave-numbers k at which to compute spacings.

# Returns
- `spacings::Vector{T}`  
  Sorted nearest-neighbor spacings ΔN = N_{i+1} − N_i of the unfolded levels.
"""
function calculate_spacings(f::Function,ks::Vector{T}) where {T<:Real}
    unfolded_energies=unfold(f,ks)
    return diff(sort(unfolded_energies))
end










##################
#### EXAMPLES ####
##################








############################
#### INITIAL PARAMETERS ####
############################

# k - indexes order of bessel functions
# l - indexes the root of the bessel function up to kmax
kmin=1 # starting order of bessel functions for basis expansion. For now use kmin>0
kmax=200 # final order of bessel functions for basis expansion
lmin=1 # starting index of bessel function roots. For now use lmin>0
lmax=200 # final index of bessel function roots
λ_value=0.05 # value of λ in the standard Robnik conformal mapping. If the mapping uses more parameters, their values should be added and also their symbols!
γ_value=0.1 # example of adding a new value
δ_value=0.1 # example of adding a new value
symmetry=:ODD # :EVEN (cos) or :ODD (sin) for the angular part of the basis expansion
quadrature=:GAUSS_LEGENDRE_FIXED # :GAUSS_KRONROD_ADAPTIVE or :GAUSS_LEGENDRE_FIXED for the quadrature method


########################
#### CREATE SYMBOLS ####
########################

#!!! If adding new parameters (symbols) one must add them to the top of the file as for SymPy to know that they need to be treated as symbolic variables for automatic differentiation/integration
r,φ,λ,γ,δ=sympy.symbols("r φ λ γ δ",real=true) # create real symbols for those used in the mappings. If used creates mappings that have more e.g greek letters this needs to be updated


##########################
#### EXAMPLE MAPPINGS ####
##########################


# The following mappings are examples of conformal mappings that can be used in the code. It is required as input in most functions and is used to calculate the eigenvalues and eigenvectors of the system. The mappings are defined as functions that take a complex number z and return a complex number w=f(z). The mappings can be defined in terms of their coefficients, which are used to construct the matrix A.

# param_vals is a dictionary that maps the symbols used in the mapping to their values. This is used to evaluate the mapping at specific points
# coeffs is a vector of PyObjects that represent the symbolic coefficients of the mapping. 

# Unit circle mapping for testing 
function conformal_mapping_1()
    return (z)->z
end
mapping_1=conformal_mapping_1()
param_vals_1=Dict{PyObject,Float64}() # no parameters in the mapping
coeffs_1=[PyObject(1)] # just circle

# Robnik's Limacon
function conformal_mapping_2(λ::T) where {T<:Real}
    return (z)->begin
        (z+λ*z^2)
    end
end
mapping_2=conformal_mapping_2(λ_value)
param_vals_2=Dict(λ=>λ_value) # store the value of λ symbol
coeffs_2=[PyObject(1),PyObject(λ)] # coefficients of the mapping as symbols. Needed for symbolic integration/differentiation and formation of a(r,φ)

# Adding γ*z^3 to the Limacon mapping
function conformal_mapping_3(λ::T,γ::T) where {T<:Real}
    return (z)->begin
        (z+λ*z^2+γ*z^3)
    end
end
mapping_3=conformal_mapping_3(λ_value,γ_value)
param_vals_3=Dict(λ=>λ_value,γ=>γ_value) # store the value of λ and γ symbol
coeffs_3=[PyObject(1),PyObject(λ),PyObject(γ)]

# 3 leaf clover mapping
function conformal_mapping_4(λ::T) where {T<:Real}
    return (z)->begin
        (z+λ*z^4)
    end
end
mapping_4=conformal_mapping_4(λ_value)
param_vals_4=Dict(λ=>λ_value)
coeffs_4=[PyObject(1),PyObject(0),PyObject(0),PyObject(λ)] 

# Adding cubic and quartic terms to the Limacon mapping
function conformal_mapping_5(γ::T,δ::T,λ::T) where {T<:Real}
    return (z)->begin
        (z+γ*z^2+δ*z^3+λ*z^4)
    end
end
mapping_5=conformal_mapping_5(γ_value,δ_value,λ_value)
param_vals_5=Dict(γ=>γ_value,δ=>δ_value,λ=>λ_value) # store the value of λ symbol
coeffs_5=[PyObject(1),PyObject(γ),PyObject(δ),PyObject(λ)] # coefficients of the mapping as symbols. Needed for symbolic integration/differentiation and formation of a(r,φ)

function conformal_mapping_6(λ::T) where {T<:Real}
    return (z)->begin
        (z+λ*z^9)
    end
end
mapping_6=conformal_mapping_6(λ_value)
param_vals_6=Dict(λ=>λ_value)
coeffs_6=[PyObject(1),PyObject(0),PyObject(0),PyObject(0),PyObject(0),PyObject(0),PyObject(0),PyObject(0),PyObject(λ)] 

####

# Here we assign to generic variables used below their unique identifiers (aka {...}_i for the i-th mapping etc.)
mapping=mapping_6
param_vals=param_vals_6
coeffs=coeffs_6


#############################
#### EXAMPLE CALCULATION ####
#############################


# ! One needs to be careful that we always use the same symmetry in all functions, otherwise we will get wrong results

# High level wrapper to calculate all the relevant parameters:
# Sparse matrix A, ordered modes, index of the modes, all roots of the bessel functions and the pairs of (m,n) with their coefficients (determines which parts in the integrands survive)
A,ordered_modes,idx,all_roots,pairs_reduced=A_construction_wrapper(kmin,kmax,lmin,lmax,coeffs,param_vals,symmetry,quadrature=quadrature)

# Check the Sparsity pattern of the matrix partitioned into block. The ammount of block diagonals (and their spacing between each other) is determined by the complexity of the mapping.
f=Figure()
ax=Axis(f[1,1])
plot_sparsity_pattern!(ax,A)
save("Sparsity_pattern_$(param_vals).png",f)

#### eigenvalues and eigenvectors ####

use_ARPACK=true # useful for large matrices where we know that only say 25% of levels are correct
fraction_of_levels=0.25 # fraction of levels to return. This is useful for large matrices where we know that only say 25% of lower levels are correct
@time eigenvalues,C=solve(A,use_ARPACK=use_ARPACK,fraction_of_levels=fraction_of_levels)
ks=sqrt.(eigenvalues) # because we need the ks for statistics and Husimi functions
# C is the matrix whose columns indexwise (C[:,i]) are the i-th eigenvectors of the i-th eigenvalue (eigenvalues[i])
@info "Eigenvalues: $(ks[1:10])"
# This is useful to check for mapping_1 (Unit circle) as it should give the Bessel roots 


##### Check the eigenvalues diff between :GAUSS_KRONROD_ADAPTIVE or :GAUSS_LEGENDRE_FIXED ####
# Small debugging check to test the convergence of GAUSS_LEGENDRE_FIXED vs. GAUSS_KRONROD_ADAPTIVE
check_convergence=false # check the convergence of the two quadrature methods
if check_convergence
A1,_,_,_,_=A_construction_wrapper(kmin,kmax,lmin,lmax,coeffs,param_vals,symmetry,quadrature=:GAUSS_KRONROD_ADAPTIVE)
A2,_,_,_,_=A_construction_wrapper(kmin,kmax,lmin,lmax,coeffs,param_vals,symmetry,quadrature=:GAUSS_LEGENDRE_FIXED)
@time eigenvalues1,C1=solve(A1,use_ARPACK=use_ARPACK,fraction_of_levels=fraction_of_levels)
@time eigenvalues2,C2=solve(A2,use_ARPACK=use_ARPACK,fraction_of_levels=fraction_of_levels)
df=DataFrame(k1=sqrt.(eigenvalues1),k2=sqrt.(eigenvalues2),ds=abs.(sqrt.(eigenvalues1).-sqrt.(eigenvalues2)))
CSV.write("Eigenvalues_diff_$(param_vals).csv",df)
end
#### wavefunction plotting ####

# for plotting lower levels
#N_min=1 # starting index of the wavefunction to plot
#N_max=20 # final index of the wavefunction to plot

# for plotting highest accepted levels
N_min=floor(Int,fraction_of_levels*kmax*lmax-21) 
N_max=floor(Int,fraction_of_levels*kmax*lmax-1)

# Bnd is the boundary function of the wavefunctions on the physical boundary. We need it to plot the husimi functions. We choose a subset of all the eigenvector expansions C[:,N_min:N_max]
φs,Ls,Bnd=boundary_function(C[:,N_min:N_max],ordered_modes,all_roots,mapping;symmetry=symmetry,Nφ=10000)

# Create the wavefunctions on the grid in the conformal image of the unit disk
Zs_flat,Xs,Ys,xb,yb=wavefunctions(C[:,N_min:N_max],ordered_modes,all_roots,mapping,symmetry=symmetry)

# Plot the wavefunctions and the husimi functions through the boundary function
f=plot_wavefunctions_with_husimi(Zs_flat,Xs,Ys,xb,yb,ks,Ls,Bnd,plot_boundary_function=true,Nlevels=20,width=500,height=500,max_col=5)
save("wavefunctions_husimi_$(param_vals).png",f)
