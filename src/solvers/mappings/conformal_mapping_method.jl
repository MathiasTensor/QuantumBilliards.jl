using GSL
using QuadGK
using LinearAlgebra
using CairoMakie
using ProgressMeter
using SparseArrays
using Arpack
using ForwardDiff
using PyCall

#=
CONFORMAL MAPPING METHOD
This module provides functions for conformal mapping of the unit disk via a complex Polynomial.
The mapping is defined by a polynomial f(z) = coeffs[1] + coeffs[2]*z + … + coeffs[end]*z^(end-1).

# For usage of this method see the below exhausting EXAMPLE section with detailed comments and self-explanatory code (altough there are comments in the code as well).
=#

# For Symbolic routines below
sympy=pyimport("sympy") # use Python's sympy API
r,φ,λ,γ=sympy.symbols("r φ λ γ",real=true) # create real symbols
z,w=sympy.symbols("z w") # symbol for z, w (both complex)


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
    a=sympy.simplify(a)
    a=sympy.expand_complex(a) # rewrite exp(iφ)→cosφ+ i sinφ
    a=sympy.trigsimp(a) # clean up trig expressions to have only cos
    @info "a(r,φ) = $a"
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
            coefs[n]=sympy.nsimplify(expr,[r,λ,γ,sympy.pi];tolerance=tolerance)  # tell nsimplify we only expect r,λ,γ,π in the final answer
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
    groups=Dict{Int,Vector{Int}}() # Group l–values by angular index k
    for (k,l) in modes
        push!(get!(groups,k,Int[]),l)
    end
    for v in values(groups)
        sort!(v)
    end
    ang_seq=Int[] # Build the angular sequence 0, -1,+1, -2,+2, …
    if haskey(groups,0)
        push!(ang_seq,0)
    end
    Nmax=maximum(abs.(keys(groups)))
    for d in 1:Nmax
        if haskey(groups,-d)
            push!(ang_seq,-d)
        end
        if haskey(groups,+d)
            push!(ang_seq,+d)
        end
    end
    ordered=Tuple{Int,Int}[]  # Assemble the ordered (k,l) list
    for k in ang_seq, l in groups[k]
        push!(ordered,(k,l))
    end
    idx=Dict{Tuple{Int,Int},Int}((mode=>i) for (i,mode) in enumerate(ordered))
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
    return k/x*sf_bessel_Jnu(k,x)-sf_bessel_Jnu(k+1,x)
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

Compute the radial overlap integral
∫₀¹ r^n Jₖ(αₖₗ r) Jₖₚ(αₖₚ,ₗₚ r) dr via adaptive quadrature.

# Arguments
- `n::Int`: power of r  
- `(k::Int,l::Int)`, `(kp::Int,lp::Int)`: Bessel orders and zero indices  
- `roots::Dict{Tuple{Int,Int},T}`: map of zeros α

# Returns
- numerical value of the integral as `T`
"""
function integrand_rn(n::Int,k::Int,l::Int,kp::Int,lp::Int,all_roots::Dict{Tuple{Int,Int},T}) where {T<:Real}
    f(r)=r^n*sf_bessel_Jnu(k,all_roots[k,l]*r)*sf_bessel_Jnu(kp,all_roots[kp,lp]*r)
    return quadgk(f,eps(T),1.0)[1]
end

"""
    construct_A(reduced_pairs::Vector{Pair{Tuple{Int,Int},PyObject}},ordered_modes::Vector{Tuple{Int,Int}},all_roots::Dict{Tuple{Int,Int},T},mask::Matrix{Bool};symmetry::Symbol=:EVEN) where {T<:Real}

Assemble the sparse symmetric matrix A for the conformal map,
using the pre-extracted (m,n) → aₘₙ pairs and a Boolean mask.

# Arguments
- `reduced_pairs`: list of nonzero `((m,n)=>a)` symbolic coefficients  
- `ordered_modes`: list of (k,l) modes  
- `roots`: map of Bessel zeros  
- `mask[i,j]`: true if A[i,j] may be nonzero
- `param_vals`: dictionary of parameter values for the symbolic coefficients
- `symmetry`: `:EVEN` or `:ODD` for the angular integral. Must be then used the same for the wavefunction construction

# Returns
- `A::SparseMatrixCSC`: the assembled symmetric matrix
"""
function construct_A(reduced_pairs::Vector{Pair{Tuple{Int,Int},PyObject}},ordered_modes::Vector{Tuple{Int,Int}},all_roots::Dict{Tuple{Int,Int},T},mask::Matrix{Bool},param_vals::Dict{PyObject,T};symmetry::Symbol=:EVEN) where {T<:Real}
    @assert symmetry in [:EVEN,:ODD] "Symmetry must be :EVEN or :ODD"
    @info "All pairs: (cos(mϕ),r^n) -> aₘₙ: $(reduced_pairs)"
    ang_int(m::Int,k::Int,kp::Int)=angular_integral(m,k,kp,symmetry)
    N=length(ordered_modes)
    A=zeros(N,N)
    #numeric_pairs=Vector{Tuple{Tuple{Int,Int},T}}(undef,length(reduced_pairs))
    #for (i,((m,n),a_sym)) in enumerate(reduced_pairs)
    #    tmp=a_sym
    #    for (sym,val) in param_vals
    #        tmp=tmp.subs(sym,val)
    #    end
    #   a_num=tmp.evalf()
    #    numeric_pairs[i]=((m,n),a_num)
    #end
    numeric_pairs=evaluate_pairs(reduced_pairs,param_vals)
    println("Numeric pairs: ",numeric_pairs)
    @info "Filled matrix: $(length(findall(x->x==true,mask))/prod(size(mask))*100) %"
    @showprogress desc="Constructing matrix" Threads.@threads for i in eachindex(ordered_modes)
        k,l=ordered_modes[i]
        for (j,(kp,lp)) in enumerate(ordered_modes)
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
                R=integrand_rn(n+1,k,l,kp,lp,all_roots) # radial piece, n+1 due to dS=r*dr*dϕ
                S+=a*ang*Ni*Nj*R
            end
            A[i,j]=(1/all_roots[kp,lp])^2*S
        end
    end
    return sparse(Symmetric(A))
end

"""
    solve(A::SparseMatrixCSC{T,Int};fraction_of_levels::T=0.2,use_ARPACK::Bool=false) :: (Vector{T}, Matrix{T})

Compute the lowest eigenvalues and eigenvectors of A, using either dense or ARPACK.

# Arguments
- `A`: sparse symmetric matrix  
- `fraction_of_levels`: fraction of total modes to return  
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
    return N*sf_bessel_Jnu(k,j*r)*(symmetry==:ODD ? sin(k*φ) : cos(k*φ))
end

"""
    make_inv_newton_fd(f::Function;tol::T=1e-12,maxiter::Int=1000) :: Function

Return a function `invf(w)` that approximates the inverse of `f(z)` via Newton’s method.

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
    wavefunctions(C::Matrix{T},ordered_modes::Vector{Tuple{Int,Int}},idx::Dict{Tuple{Int,Int},Int},roots::Dict{Tuple{Int,Int},T},inv_map::Function;symmetry::Symbol=:ODD,res::Int=100,xrange::Tuple{T,T}=(-1.5,1.5),plot_outside::Bool=false) :: (Vector{Matrix{T}}, Vector{T}, Vector{T})

Compute the spatial wavefunction matrices for each eigenvector C on a uniform
grid in (x,y), optionally including the exterior region via `plot_outside`.

# Arguments
- `C::Matrix{T}`: N×nstates eigenvector matrix  
- `ordered_modes::Vector{Tuple{Int,Int}}, idx::Dict{Tuple{Int,Int},Int}, roots::Dict{Tuple{Int,Int},T}`: basis setup  
- `conformal_mapping(w)::Function`: conformal map z=f(w)  
- `symmetry::Symbol`: `:EVEN` or `:ODD` for angular part, chooses cos or sin basis
- `res::Int`: grid resolution per axis  
- `xrange::Tuple{Float64,Float64}`: (xmin,xmax) for both x and y  
- `plot_outside::Bool`: if true, evaluate everywhere; else restrict to |z|≤1

# Returns
- `Psis::Matrix`: vector of res×res matrices, each |ψ(x,y)| values  
- `xs, ys`: coordinate vectors of length `res`
"""
function wavefunctions(C::Matrix{T},ordered_modes::Vector{Tuple{Int,Int}},idx::Dict{Tuple{Int,Int},Int},all_roots::Dict{Tuple{Int,Int},T},conformal_mapping::Function;symmetry::Symbol=:EVEN,res::Int=100,xrange::Tuple{Float64,Float64}=(-1.5,1.5),plot_outside=false) where T<:Real
    nstates=size(C,2)
    xmin,xmax=xrange
    xs=range(xmin,xmax,length=res)
    ys=collect(xs)
    Psis=Vector{Matrix{T}}(undef,nstates)
    @fastmath begin
        @inbounds @showprogress desc="Constructing wavefunctions" Threads.@threads for n in 1:nstates
            vec=C[:,n]
            mat=zeros(T,res,res)
            @inbounds Threads.@threads for ix in 1:res
                for iy in 1:res
                    x,y=xs[ix],ys[iy]
                    w=x+im*y
                    z=make_inv_newton_fd(conformal_mapping)(w)
                    r=abs(z)
                    if r<sqrt(eps(T))
                        r=sqrt(eps(T)) # numerical undeflow around 0 for GSL Bessel functions
                    end
                    φ=atan(imag(z),real(z))
                    acc=zero(T)
                    if plot_outside
                        @inbounds for (k,l) in ordered_modes
                            i=idx[(k,l)]
                            acc+=vec[i]*ϕ(k,l,all_roots,r,φ;symmetry=symmetry)
                        end
                        mat[iy,ix]=acc
                    else
                        if abs(z)≤1
                            @inbounds for (k,l) in ordered_modes
                                i=idx[(k,l)]
                                acc+=vec[i]*ϕ(k,l,all_roots,r,φ;symmetry=symmetry)
                            end
                            mat[iy,ix]=acc
                        end
                    end
                end
            end
            Psis[n]=mat
        end
    end
  return Psis,xs,ys
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

# Returns
A tuple `(A, ordered_modes, idx, all_roots)` where:
- `A::SparseMatrixCSC{T,Int}`: assembled symmetric matrix  
- `ordered_modes::Vector{Tuple{Int,Int}}`: list of (k,l) pairs in matrix order  
- `idx::Dict{Tuple{Int,Int},Int}`: map from (k,l) → its row/column index  
- `all_roots::Dict{Tuple{Int,Int},T}`: precomputed Bessel zeros for each (k,l)
- `pairs_reduced::Vector{Pair{Tuple{Int,Int},PyObject}}`: list of nonzero pairs numerically evaluated
"""
function A_construction_wrapper(kmin::Int,kmax::Int,lmin::Int,lmax::Int,coeffs::Vector{PyObject},param_vals::Dict{PyObject,T},symmetry::Symbol) where {T<:Real}
    modes=[(k,l) for k in kmin:kmax for l in lmin:lmax]
    ordered_modes,idx=angular_radial_order(modes)
    pairs=integrand_coeff_dictionary(coeffs)
    A_mask,pairs_reduced=make_A_mask(pairs,ordered_modes)
    all_roots=all_bessel_J_nu_roots(kmin,kmax,lmin,lmax)
    A=construct_A(pairs_reduced,ordered_modes,all_roots,A_mask,param_vals,symmetry=symmetry)
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
    u(θ)=real(f(exp(im*θ)))
    v(θ)=imag(f(exp(im*θ)))
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
    u(θ)=real(f(exp(im*θ)))
    v(θ)=imag(f(exp(im*θ)))
    du(θ)=ForwardDiff.derivative(u,θ)
    dv(θ)=ForwardDiff.derivative(v,θ)
    integrand(θ)=sqrt(du(θ)^2+dv(θ)^2)
    return quadgk(integrand,0.0,2*pi)[1]
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
    N_ks=(A*ks.^2 .- L .* ks)./(4π) .+ 1/4
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






#### INITIAL PARAMETERS

# k - indexes order of bessel functions
# l - indexes the root of the bessel function up to kmax
kmin=1 # starting order of bessel functions for basis expansion
kmax=20 # final order of bessel functions for basis expansion
lmin=1 # starting index of bessel function roots
lmax=20 # final index of bessel function roots
λ_value=0.9 # value of λ in the standard Robnik conformal mapping. If the mapping uses more parameters, their values should be added and also their symbols!
symmetry=:ODD # :EVEN (cos) or :ODD (sin) for the angular part of the basis expansion
γ_value=0.2 # example of adding a new value
#!!! If adding new parameters (symbols) one must add them to the top of the file as for SymPy to know that they need to be treated as symbolic variables for automatic differentiation/integration

####





#### EXAMPLE MAPPINGS

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
coeffs_3=[PyObject(1),PyObject(λ)#=,PyObject(γ)=#]

####




#### EXAMPLE CALCULATION
# High level wrapper to calculate all the relevant parameters:
# Sparse matrix A, ordered modes, index of the modes, all roots of the bessel functions and the pairs of (m,n) with their coefficients (determines which parts in the integrands survive)
A,ordered_modes,idx,all_roots,pairs_reduced=A_construction_wrapper(kmin,kmax,lmin,lmax,coeffs_2,param_vals_2,symmetry)

# Check the Sparsity pattern of the matrix partitioned into block. The ammount of block diagonals (and their spacing between each other) is determined by the complexity of the mapping.
f=Figure()
ax=Axis(f[1,1])
plot_sparsity_pattern!(ax,A)
save("Limacon_Sparsity_$(λ_value)_NEW_GENERAL.png",f)

use_ARPACK=true # useful for large matrices where we know that only say 20% of levels are correct
@time eigenvalues,C=solve(A,use_ARPACK=use_ARPACK,fraction_of_levels=0.2)
# C is the matrix whose columns indexwise (C[:,i]) are the i-th eigenvectors of the i-th eigenvalue (eigenvalues[i])
eigenvalues=sqrt.(eigenvalues) # because we need the ks and not the energies Es 

@info "Eigenvalues: $(eigenvalues[1:10])"
# This is useful to check for mapping_1 (Unit circle) as it should give the Bessel roots 



#### wavefunction plotting

N_min=1 # starting index of the wavefunction to plot
N_max=10 # final index of the wavefunction to plot
N=N_max-N_min+1 # number of wavefunctions to plot
Psis,xs,ys=wavefunctions(C[:,1:N],ordered_modes,idx,all_roots,mapping_2,symmetry=symmetry,res=256,plot_outside=true,xrange=(-2.0,2.0))
# if plot_outside=true, the wavefunction is calculated outside the boundary as well for checking boundary condition being Dirichlet

max_col=5 
r=1
c=1
f=Figure(size=(600*max_col,ceil(Int,N/max_col)*600),resolution=(600*max_col,ceil(Int,N/max_col)*600))
for i in N_min:N_max
    global r,c
    if c>max_col
        r+=1
        c=1
    end
    local ax=Axis(f[r,c][1,1],width=400,height=400)
    hmap=heatmap!(ax,ys,xs,abs.(Psis[i]),colormap=Reverse(:gist_heat))
    Colorbar(f[r,c][1,2],hmap)
    c+=1
end
save("wavefunctions_$(λ_value)_$(γ_value)_$(symmetry).png",f)
