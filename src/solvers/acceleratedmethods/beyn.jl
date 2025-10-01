
#####################################
#### CONSTRUCTORS FOR COMPLEX ks ####
#####################################

@inline function _add_pair_default_complex!(M::AbstractMatrix{Complex{T}},i::Int,j::Int,xi::T,yi::T,nxi::T,nyi::T,xj::T,yj::T,nxj::T,nyj::T,k::Complex{T},tol2::T,pref::Complex{T};scale::T=one(T)) where {T<:Real}
    dx=xi-xj;dy=yi-yj
    d2=muladd(dx,dx,dy*dy)
    if d2<=tol2
        return false
    end
    d=sqrt(d2)
    invd=inv(d)
    h=pref*SpecialFunctions.hankelh1(1,k*d)
    @inbounds begin
        M[i,j]+=scale*((nxi*dx+nyi*dy)*invd)*h
    end
    return true
end

@inline function _add_pair_custom_complex!(M::AbstractMatrix{Complex{T}},i::Int,j::Int,xi::T,yi::T,nxi::T,nyi::T,xj::T,yj::T,nxj::T,nyj::T,k::Complex{T},kernel_fun;scale::T=one(T)) where {T<:Real}
    val_ij=kernel_fun(i,j,xi,yi,nxi,nyi,xj,yj,nxj,nyj,k)*scale
    @inbounds begin
        M[i,j]+=val_ij
    end
    return true
end

@inline _x_reflect(x::T,sx::T) where {T<:Real}=(2*sx-x)
@inline _y_reflect(y::T,sy::T) where {T<:Real}=(2*sy-y)

function compute_kernel_matrix_complex_k(bp::BoundaryPointsBIM{T},k::Complex{T};multithreaded::Bool=true,kernel_fun::Union{Symbol,Function}=:default) where {T<:Real}
    xy=bp.xy;nrm=bp.normal;κ=bp.curvature;N=length(xy)
    K=Matrix{Complex{T}}(undef,N,N)
    xs=getindex.(xy,1);ys=getindex.(xy,2);nx=getindex.(nrm,1);ny=getindex.(nrm,2)
    tol2=(eps(T))^2;pref=-im*k/2;TW= T(2π)
    if kernel_fun===:default
        QuantumBilliards.@use_threads multithreading=multithreaded for i in 1:N
            xi=xs[i];yi=ys[i];nxi=nx[i];nyi=ny[i]
            @inbounds for j in 1:i
                dx=xi-xs[j];dy=yi-ys[j];d2=muladd(dx,dx,dy*dy)
                if d2≤tol2
                    K[i,j]=Complex{T}(κ[i]/TW)
                else
                    d=sqrt(d2);invd=inv(d);h=pref*SpecialFunctions.hankelh1(1,k*d)
                    K[i,j]=(nxi*dx+nyi*dy)*invd*h
                    if i!=j
                        K[j,i]=(nx[j]*(-dx)+ny[j]*(-dy))*invd*h
                    end
                end
            end
        end
    else
        QuantumBilliards.@use_threads multithreading=multithreaded for i in 1:N
            xi=xs[i];yi=ys[i];nxi=nx[i];nyi=ny[i]
            @inbounds for j in 1:N
                xj=xs[j];yj=ys[j];nxj=nx[j];nyj=ny[j]
                K[i,j]=kernel_fun(i,j,xi,yi,nxi,nyi,xj,yj,nxj,nyj,k)
            end
        end
    end
    return K
end

function compute_kernel_matrix_complex_k(bp::BoundaryPointsBIM{T},symmetry::Vector{Any},k::Complex{T};multithreaded::Bool=true,kernel_fun::Union{Symbol,Function}=:default) where {T<:Real}
    xy=bp.xy
    nrm=bp.normal
    κ=bp.curvature 
    N=length(xy)
    K=zeros(Complex{T},N,N)
    tol2=(eps(T))^2
    pref=-im*k/2
    add_x=false;add_y=false;add_xy=false # true if the symmetry is present
    sxgn=one(T);sygn=one(T);sxy=one(T) # the scalings +/- depending on the symmetry considerations
    shift_x=bp.shift_x;shift_y=bp.shift_y # the reflection axes shifts from billiard geometry
    @inbounds for s in symmetry # symmetry here is always != nothing
        if s.axis==:y_axis;add_x=true;sxgn=(s.parity==-1 ? -one(T) : one(T)); end
        if s.axis==:x_axis;add_y=true;sygn=(s.parity==-1 ? -one(T) : one(T)); end
        if s.axis==:origin
            add_x=true;add_y=true;add_xy=true
            sxgn=(s.parity[1]==-1 ? -one(T) : one(T))
            sygn=(s.parity[2]==-1 ? -one(T) : one(T))
            sxy=sxgn*sygn
        end
    end
    isdef=(kernel_fun===:default)
    QuantumBilliards.@use_threads multithreading=multithreaded for i in 1:N
        xi=xy[i][1]; yi=xy[i][2]; nxi=nrm[i][1]; nyi=nrm[i][2]
        @inbounds for j in 1:N # since it has non-trivial symmetry we have to do both loops over all indices, not just the upper triangular
            xj=xy[j][1];yj=xy[j][2];nxj=nrm[j][1];nyj=nrm[j][2]
            if isdef
                ok=_add_pair_default_complex!(K,i,j,xi,yi,nxi,nyi,xj,yj,nxj,nyj,k,tol2,pref)
                if !ok; K[i,j]+=Complex(κ[i]/TWO_PI); end
            else
                _add_pair_custom_complex!(K,i,j,xi,yi,nxi,nyi,xj,yj,nxj,nyj,k,kernel_fun)
            end
            if add_x # reflect only over the x axis
                xr=_x_reflect(xj,shift_x);yr=yj
                isdef ? _add_pair_default_complex!(K,i,j,xi,yi,nxi,nyi,xr,yr,nxj,nyj,k,tol2,pref;scale=sxgn) :
                        _add_pair_custom_complex!(K,i,j,xi,yi,nxi,nyi,xr,yr,nxj,nyj,k,kernel_fun;scale=sxgn)
            end
            if add_y # reflect only over the y axis
                xr=xj;yr=_y_reflect(yj,shift_y)
                isdef ? _add_pair_default_complex!(K,i,j,xi,yi,nxi,nyi,xr,yr,nxj,nyj,k,tol2,pref;scale=sygn) :
                        _add_pair_custom_complex!(K,i,j,xi,yi,nxi,nyi,xr,yr,nxj,nyj,k,kernel_fun;scale=sygn)
            end
            if add_xy # reflect over both the axes
                xr=_x_reflect(xj,shift_x);yr=_y_reflect(yj,shift_y)
                isdef ? _add_pair_default_complex!(K,i,j,xi,yi,nxi,nyi,xr,yr,nxj,nyj,k,tol2,pref;scale=sxy) :
                        _add_pair_custom_complex!(K,i,j,xi,yi,nxi,nyi,xr,yr,nxj,nyj,k,kernel_fun;scale=sxy)
            end
        end
    end
    return K
end

function fredholm_matrix_complex_k(bp::BoundaryPointsBIM{T},symmetry::Union{Vector{Any},Nothing},k::Complex{T};multithreaded::Bool=true,kernel_fun::Union{Symbol,Function}=:default) where {T<:Real}
    K=isnothing(symmetry) ?
        compute_kernel_matrix_complex_k(bp,k;multithreaded=multithreaded,kernel_fun=kernel_fun) :
        compute_kernel_matrix_complex_k(bp,symmetry,k,multithreaded=multithreaded,kernel_fun=kernel_fun)
    ds=bp.ds
    @inbounds for j in eachindex(ds)
        @views K[:,j].*=ds[j]
    end
    K.*=-one(T)
    @inbounds for i in axes(K,1)
        K[i,i]+=one(T)
    end
    return QuantumBilliards.filter_matrix!(K)
end

#################
#### HELPERS ####
#################

# ΔN(k,Δk)=N(k+Δk)-N(k)
function delta_weyl(billiard::Bi,k::T,Δk::T;fundamental::Bool=true) where {T<:Real,Bi<:QuantumBilliards.AbsBilliard}
    # Use Weyl’s law as a fast estimator of counting function N(k); difference gives # levels in [k,k+Δk]
    weyl_law(k+Δk,billiard;fundamental=fundamental)-weyl_law(k,billiard;fundamental=fundamental)
end

# Δk₀≈m/ρ(k); floor to avoid zero
function initial_step_from_dos(billiard::Bi,k::T,m::Int;fundamental::Bool=true,min_step::Real=1e-6) where {T<:Real,Bi<:QuantumBilliards.AbsBilliard}
    ρ=max(dos_weyl(k,billiard;fundamental=fundamental),1e-12) # estimate DOS ρ(k); clamp to avoid division by tiny numbers
    max(m/ρ,min_step) # target m levels -> Δk≈m/ρ; enforce a minimum to avoid Δk≈0 in sparse regions
end

# grow Δk until ΔN(k,Δk)≥m or we hit remaining; here 'remaining' is already clamped
function grow_upper_bound(billiard::Bi,k::T,m,Δk0::T,remaining::T;fundamental::Bool=true,max_grows::Int=60) where {T<:Real,Bi<:QuantumBilliards.AbsBilliard}
    Δk=min(remaining,max(Δk0,eps(k))) # start from Δk0 but: (i) not below machine eps, (ii) not above remaining span
    for _ in 1:max_grows
        if delta_weyl(billiard,k,Δk;fundamental=fundamental)≥m-eps();return Δk,true end # stop once we bracket ≥m levels
        if Δk≥0.999999*remaining;return remaining,false end # cannot grow further—signal that we hit the cap
        Δk=min(remaining,2Δk) # after cap whether we met target
    end
    return Δk,delta_weyl(billiard,k,Δk;fundamental=fundamental)≥m
end

# bisection over Δk∈[lo,hi] to solve ΔN(k,Δk)≈m
function bisect_for_delta_k(billiard::Bi,k::T,m,lo::T,hi::T;fundamental::Bool=true,tol_levels=0.1,maxit::Int=50) where {T<:Real,Bi<:QuantumBilliards.AbsBilliard}
    @assert hi>lo
    Nlo=delta_weyl(billiard,k,lo;fundamental=fundamental) # evaluate count at lo
    if abs(Nlo-m)≤tol_levels;return lo end # early accept if already within tolerance
    Nhi=delta_weyl(billiard,k,hi;fundamental=fundamental) # evaluate count at hi
    if abs(Nhi-m)≤tol_levels;return hi end #  early accept at hi
    @assert Nhi≥m
    for _ in 1:maxit
        mid=0.5*(lo+hi)  # bisection on Δk
        Nmid=delta_weyl(billiard,k,mid;fundamental=fundamental)  # count at midpoint
        if abs(Nmid-m)≤tol_levels;return mid end # tolerance satisfied
        if Nmid<m;lo=mid else hi=mid end # shrink bracket toward target
        if hi-lo≤max(1e-12,1e-9*max(1.0,k));return 0.5*(lo+hi) end # stop when bracket is tiny (abs or relative to k)
    end
    return 0.5*(lo+hi) # fallback: return midpoint of final bracket
end

# cover [k1,k2] with windows of ≈m levels, AND enforce |window|≤2Rmax (so R≤Rmax)
function plan_weyl_windows(billiard::Bi,k1::T,k2::T; m::Int=10, Rmax::Real=1.0,
                           fundamental::Bool=true, tol_levels=0.1, maxit::Int=50) where {T<:Real,Bi<:QuantumBilliards.AbsBilliard}
    iv=Vector{Tuple{T,T}}() # accumulator of (kL,kR) windows
    k=k1 # left cursor that advances to cover [k1,k2]
    while k<k2-eps() # loop until we reach the right end
        rem_raw=k2-k # remaining span
        rem=rem_raw>2*Rmax ? T(2*Rmax) : rem_raw # enforce max window length 2 * Rmax (so disk radius R≤Rmax)
        if delta_weyl(billiard,k,rem;fundamental=fundamental)≤m+tol_levels && rem==rem_raw
            push!(iv,(k,k2)); break # if tail holds ≤m levels and fits entirely, close with a single window
        end
        Δk0=initial_step_from_dos(billiard,k,m;fundamental=fundamental) # DOS-based initial guess for ~m levels
        hi,ok=grow_upper_bound(billiard,k,m,Δk0,rem;fundamental=fundamental) # geometrically grow Δk until we bracket ≥m or hit cap
        if !ok
            push!(iv,(k,k+hi)); k+=hi; continue # couldn’t hit ≥m due to length cap: accept best effort and advance
        end
        Δk=bisect_for_delta_k(billiard,k,m,0.0,hi;fundamental=fundamental,tol_levels=tol_levels,maxit=maxit) # refine to ≈m levels
        kR=min(k+Δk,k+rem) # still respect max window length
        push!(iv,(k,kR)); k=kR # record window and advance cursor
    end
    return iv # list of windows covering [k1,k2]
end

# Beyn disks (centers/radii) from windows; lengths already ≤2Rmax
function beyn_disks_from_windows(iv::Vector{Tuple{T,T}}) where {T<:Real}
    k0=Vector{Complex{T}}(undef,length(iv)); R=Vector{T}(undef,length(iv)) # disk centers (complex, imag=0) and radii
    @inbounds for (i,(kL,kR)) in pairs(iv) # center = midpoint of window -> matches Beyn circle center
        k0[i]=complex(0.5*(kL+kR)); R[i]=0.5*(kR-kL) # radius = half window length → guarantees |window| ≤ 2Rmax -> R≤Rmax
    end
    return k0,R
end

#################################
#### SOLVERS FOR BEYN METHOD ####
#################################

function construct_B_matrix(fun::Fu,k0::Complex{T},R::T;nq::Int=32,r::Int=48,svd_tol=1e-14,rng=MersenneTwister(0)) where {T<:Real,Fu<:Function}
    # Reference: Beyn, Wolf-Jurgen, An integral method for solving nonlinear eigenvalue problems, 2018, especially Integral algorithm 1 on p14
    # quadrature nodes/weights on contor Γ 
    θ=range(zero(T),TWO_PI;length=nq+1) # the angles that form the complex circle, equally spaced since curvature zero for trapezoidal rule
    θ=θ[1:end-1] # make sure we start at 0 -> 2*pi
    ej=cis.(θ) # e^{iθ} via cis, infinitesimal contribution to speed
    zj=k0.+R.*ej # the actual complex nodes where to take the ks, we choose center around k0
    wj=(R/nq).*ej # Δz/(2π*i) absorbed in weighting as per eq 30/31 in Section 3 in ref.
    T0=fun(zj[1]) # initial on real axis calculation
    Tbuf=similar(T0) # workspace allocation for contour additions
    copyto!(Tbuf,T0) # populate working buffer with starting value
    N=size(T0,1) # as per integral alogorithm 1 in refe
    V=randn(rng,Complex{T},N,r) # random matrix reused in inner accumulator loop. It is needed not to miss instances of eigenvectors by the operator being orthogonal to them
    A0=zeros(Complex{T},N,r) # the spectral operator A0 = 1 / (2*π*i) * ∮ T^{-1}(z) * V dz
    A1=zeros(Complex{T},N,r) # the spectral operator A1 = 1 / (2*π*i) * ∮ T^{-1}(z) * V * z dz
    X=similar(V) # RHS workspace for sequential LU decomposition at every zj
    # contour accumulation: A0 += wj * (T(zj) \ V), A1 += (wj*zj) * (T(zj) \ V), instead of forming the inverse directly we create a LU factorization object and use ldiv! on it to get the same algebraic operation
    @fastmath begin # cond(Tz) # actually of the real axis the condition numbers of Fredholm A matrix improve greatly!
        @inbounds for j in eachindex(zj)
            if j>1;copyto!(Tbuf,fun(zj[j]));end # first one already in buffer
            F=lu!(Tbuf,check=false) # LU for the ldiv!
            ldiv!(X,F,V) # make efficient inverse
            α0=wj[j] # 1 / (2*π*i) weight for A0
            α1=wj[j]*zj[j] # 1 / (2*π*i) * z weight for A1
            BLAS.axpy!(α0,vec(X),vec(A0)) # A0 += α0 * X
            BLAS.axpy!(α1,vec(X),vec(A1)) # A1 += α1 * X
        end
    end
    U,Σ,W=svd!(A0;full=false) # thin SVD of A0, revealing rank. The singular values > svd_tol correspond to eigenvalues. If all sv > svd_tol then maybe increase r (expected eigenvalue count) or reduce R (contour around k0), but if increasing r careful with nq. Check ref. section 3 eq. 22
    rk=0
    @inbounds for i in eachindex(Σ)
        if Σ[i]≥svd_tol # filter out those that correspond to actual eigenvalues
            rk+=1 # to determine how big we must construct the matrices below
        else
            break
        end
    end
    rk==0 && return Matrix{Complex{T}}(undef,N,0),Matrix{Complex{T}}(undef,N,0) # if nothing found early return
    Uk=@view U[:,1:rk] # take the relevant ones corresponding to eigenvalues as in Integral algorithm 1 on p14 of ref
    Wk=@view W[:,1:rk]  # take the relevant ones corresponding to eigenvalues as in Integral algorithm 1 on p14 of ref
    Σk=@view Σ[1:rk] # take the relevant ones corresponding to eigenvalues as in Integral algorithm 1 on p14 of ref
    # form B = adjoint(U) * A1 * W * Σ^{-1} as in the reference, p14, integral algorithm 1
    tmp=Matrix{Complex{T}}(undef,N,rk)
    mul!(tmp,A1,Wk) # tmp := A1 * Wk, not weighted by inverse diagonal Σk
    @inbounds for j in 1:rk # right-divide by diagonal Σk
        @views tmp[:,j]./=Σk[j]
    end
    B=Matrix{Complex{T}}(undef,rk,rk)
    mul!(B,adjoint(Uk),tmp) # B := Uk'*tmp, the final step
    return B,Uk
end

function solve_vect(solver::BoundaryIntegralMethod,basis::Ba,pts::BoundaryPointsBIM,k::Complex{T},dk::T;kernel_fun::Union{Symbol,Function}=:default,multithreaded::Bool=true,nq::Int=32,r::Int=48,svd_tol=1e-14,res_tol=1e-8,rng=MersenneTwister(0)) where {Ba<:AbstractHankelBasis} where {T<:Real}
    fun=z->fredholm_matrix_complex_k(pts,solver.symmetry,z;multithreaded=multithreaded,kernel_fun=kernel_fun)
    B,Uk=construct_B_matrix(fun,k,dk,nq=nq,r=r,svd_tol=svd_tol,rng=rng) # here is where the core of the algorithm is found. Constructs B from step 5 in ref p.14
    if isempty(B) # rk==0
        @info "no_roots_in_window" k0=k R=dk nq=nq svd_tol=svd_tol
        return Complex{T}[],Matrix{Complex{T}}(undef,size(Uk,1),0),T[]
    end
    λ,Y=eigen!(B) # small dense eigendecomposition to get eigenvalues λ are the eigenvalues and v(λ) are the eigenvectors
    Phi=Uk*Y # Φ = U * Y since A0 = U Σ W*, we have A0 * W Σ^{-1} Y = U Y. Each column is now an eigenvector of of T(λ)v(λ) = 0. This is the second layer potential boundary operator now!
    keep=trues(length(λ))
    tens=Vector{T}()
    ybuf=Vector{Complex{T}}(undef,length(Phi[:,1])) # all DLP density operators the have same length
    @inbounds for j in eachindex(λ)
        d=abs(λ[j]-k) # take only those found in the radius R where we have the expected eigenvalues for which r was used
        if d>dk
            keep[j]=false
            continue
        end
        mul!(ybuf,fun(λ[j]),@view(Phi[:,j])) # ybuf = T(λ_j)*φ_j, this is a measure of how well we solve the original problem T(λ)v(λ) = 0
        ybuf_norm=norm(ybuf)
        if ybuf_norm≥res_tol # residual criterion, ybuf should be on the order of 1e-13 - 1e-14 for both the imaginary and real part. If larger than that nq must be increased. Check for a small segment with sweep methods like psm/bim/dm at the end of the wanted spectrum to determime of nq is enough for the whole spectrum. If nq large enough use it for whole spectrum
            keep[j]=false
            if ybuf_norm>1e-8
                @info "k=$(real(λ[j])) ||A(k)v(k)|| = $(ybuf_norm) > $res_tol , definitely spurious" 
            else
                @warn "k=$(real(λ[j])) ||A(k)v(k)|| = $(ybuf_norm) > $res_tol , could be spurious or try increasing nq (usually spurious) or lowering residual tolerance" 
            end
            continue
        end
        push!(tens,d)
    end
    return λ[keep],Phi[:,keep],tens # eigenvalues, DLP density operator, "tension - difference from k0, determines badness since for analytic domain it has exponential convergence with exponent nq * N where N is the Fredholm matrix dimension (check ref Abstract)"
end

function solve(solver::BoundaryIntegralMethod,basis::Ba,pts::BoundaryPointsBIM,k::Complex{T},dk::T;kernel_fun::Union{Symbol,Function}=:default,multithreaded::Bool=true,nq::Int=32,r::Int=48,svd_tol=1e-14,res_tol=1e-8,rng=MersenneTwister(0)) where {Ba<:AbstractHankelBasis} where {T<:Real}
    fun=z->fredholm_matrix_complex_k(pts,solver.symmetry,z;multithreaded=multithreaded,kernel_fun=kernel_fun)
    B,Uk=construct_B_matrix(fun,k,dk,nq=nq,r=r,svd_tol=svd_tol,rng=rng) # here is where the core of the algorithm is found. Constructs B from step 5 in ref p.14
    if isempty(B) # rk==0
        @info "no_roots_in_window" k0=k R=dk nq=nq svd_tol=svd_tol
        return Complex{T}[],Matrix{Complex{T}}(undef,size(Uk,1),0),T[]
    end
    λ,Y=eigen!(B) # small dense eigendecomposition to get eigenvalues λ are the eigenvalues and v(λ) are the eigenvectors
    Phi=Uk*Y # Φ = U * Y since A0 = U Σ W*, we have A0 * W Σ^{-1} Y = U Y. Each column is now an eigenvector of of T(λ)v(λ) = 0. This is the second layer potential boundary operator now!
    keep=trues(length(λ))
    tens=Vector{T}()
    ybuf=Vector{Complex{T}}(undef,length(Phi[:,1])) # all DLP density operators the have same length
    @inbounds for j in eachindex(λ)
        d=abs(λ[j]-k) # take only those found in the radius R where we have the expected eigenvalues for which r was used
        if d>dk
            keep[j]=false
            continue
        end
        mul!(ybuf,fun(λ[j]),@view(Phi[:,j])) # ybuf = T(λ_j)*φ_j, this is a measure of how well we solve the original problem T(λ)v(λ) = 0
        ybuf_norm=norm(ybuf)
        if ybuf_norm≥res_tol # residual criterion, ybuf should be on the order of 1e-13 - 1e-14 for both the imaginary and real part. If larger than that nq must be increased. Check for a small segment with sweep methods like psm/bim/dm at the end of the wanted spectrum to determime of nq is enough for the whole spectrum. If nq large enough use it for whole spectrum
            keep[j]=false
            if ybuf_norm>1e-8 # heuristic for when usually it is spurious sqrt(eps())
                @info "k=$(real(λ[j])) ||A(k)v(k)|| = $(ybuf_norm) > $res_tol , definitely spurious" 
            else
                @warn "k=$(real(λ[j])) ||A(k)v(k)|| = $(ybuf_norm) > $res_tol , could be spurious or try increasing nq (usually spurious) or lowering residual tolerance" 
            end
            continue
        end
        push!(tens,d)
    end
    return λ[keep],tens # eigenvalues, DLP density operator, "tension - difference from k0, determines badness since for analytic domain it has exponential convergence with exponent nq * N where N is the Fredholm matrix dimension (check ref Abstract)"
end

function solve_INFO(solver::BoundaryIntegralMethod,basis::Ba,pts::BoundaryPointsBIM,k0::Complex{T},R::T;kernel_fun::Union{Symbol,Function}=:default,multithreaded::Bool=true,nq::Int=48,r::Int=48,svd_tol::Real=1e-10,res_tol::Real=1e-10,rng=MersenneTwister(0)) where {Ba<:AbstractHankelBasis,T<:Real}
    fun=z->fredholm_matrix_complex_k(pts,solver.symmetry,z;multithreaded=multithreaded,kernel_fun=kernel_fun)
    θ=range(zero(T),TWO_PI;length=nq+1);θ=θ[1:end-1];ej=cis.(θ);zj=k0.+R.*ej;wj=(R/nq).*ej
    T0=fun(zj[1]);Tbuf=similar(T0);copyto!(Tbuf,T0)
    N=size(T0,1);V=randn(rng,Complex{T},N,r);A0=zeros(Complex{T},N,r);A1=zeros(Complex{T},N,r);X=similar(V)
    @info "beyn:start" k0=k0 R=R nq=nq N=N r=r
    @inbounds for j in eachindex(zj)
        if j>1;copyto!(Tbuf,fun(zj[j]));end
        F=lu!(Tbuf,check=false)
        ldiv!(X,F,V)
        α0=wj[j];α1=wj[j]*zj[j]
        BLAS.axpy!(α0,vec(X),vec(A0));BLAS.axpy!(α1,vec(X),vec(A1))
    end
    @time "SVD" U,Σ,W=svd!(A0;full=false)
    println("Singular values (<1e-10 tail inspection): ",Σ)
    rk=0
    @inbounds for i in eachindex(Σ)
        if Σ[i]≥svd_tol
            rk+=1
        else
            break
        end
    end
    rk==0 && return Complex{T}[],Matrix{Complex{T}}(undef,N,0),T[]
    Uk=@view U[:,1:rk]
    Wk=@view W[:,1:rk]
    Σk=@view Σ[1:rk]
    tmp=Matrix{Complex{T}}(undef,N,rk)
    mul!(tmp,A1,Wk)
    @inbounds for j in 1:rk
        @views tmp[:,j]./=Σk[j]
    end
    B=Matrix{Complex{T}}(undef,rk,rk)
    mul!(B,adjoint(Uk),tmp)
    @time "eigen" ev=eigen!(B)
    λ=ev.values;Y=ev.vectors;Phi=Uk*Y
    keep=trues(length(λ))
    tens=Vector{T}()
    ybuf=Vector{Complex{T}}(undef,size(Phi,1))
    dropped_out=0
    dropped_res=0
    res_keep=T[]
    @inbounds for j in eachindex(λ)
        d=abs(λ[j]-k0)
        if d>R
            keep[j]=false
            dropped_out+=1
            continue
        end
        mul!(ybuf,fun(λ[j]),@view(Phi[:,j]))
        @info "k=$(real(λ[j])) ||A(k)v(k)|| = $(norm(ybuf)) < $res_tol"
        ybuf_norm=norm(ybuf)
        if ybuf_norm≥res_tol
            keep[j]=false
            dropped_res+=1
            if ybuf_norm>1e-8 # heuristic for when usually it is spurious sqrt(eps())
                @info "k=$(real(λ[j])) ||A(k)v(k)|| = $(ybuf_norm) > $res_tol , definitely spurious" 
            else
                @warn "k=$(real(λ[j])) ||A(k)v(k)|| = $(ybuf_norm) > $res_tol , could be spurious or try increasing nq (usually spurious) or lowering residual tolerance" 
            end
            continue
        end
        push!(tens,d)
        push!(res_keep,norm(ybuf))
    end
    kept=count(keep)
    if kept>0
        @info "STATUS: " kept=kept dropped_outside=dropped_out dropped_residual=dropped_res max_residual=maximum(res_keep)
    else
        @info "STATUS: " kept=0 dropped_outside=dropped_out dropped_residual=dropped_res
    end
    return λ[keep],Phi[:,keep],tens
end