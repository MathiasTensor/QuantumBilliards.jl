using KrylovKit
using LinearMaps
using LinearAlgebra

################################
#### CALLED IMPLEMENTATIONS ####
################################


function solve_krylov(solver::BoundaryIntegralMethod,basis::Ba,pts::BoundaryPointsBIM,k;kernel_fun::Union{Symbol,Function}=:default,multithreaded::Bool=true) where {Ba<:AbstractHankelBasis}
    A=construct_matrices(solver,basis,pts,k;kernel_fun=kernel_fun,multithreaded=multithreaded)
    @blas_multi_then_1 MAX_BLAS_THREADS vals,_,_,_=svdsolve(A,1,:SR)
    return vals[1]
end

function solve_vect_krylov(solver::BoundaryIntegralMethod,basis::Ba,pts::BoundaryPointsBIM,k;kernel_fun::Union{Symbol,Function}=:default,multithreaded::Bool=true) where {Ba<:AbstractHankelBasis}
    A=construct_matrices(solver,basis,pts,k;kernel_fun=kernel_fun,multithreaded=multithreaded)
    @blas_multi_then_1 MAX_BLAS_THREADS vals,_,rvect,_=svdsolve(A,1,:SR) # take the lowest singular value and the associated eigenvector, might add option to choose the number of lowest lying singualr values for symmetry reasons.
    return vals[1],rvect[1]
end

function solve_krylov(solver::ExpandedBoundaryIntegralMethod,basis::Ba,pts::BoundaryPointsBIM,k,dk;kernel_fun::Union{Tuple{Symbol,Symbol,Symbol},Tuple{Function,Function,Function}}=(:default,:first,:second),multithreaded::Bool=true,nev::Int=5,tol=1e-14,maxiter=5000,krylovdim::Int=min(40,max(40,2*nev+1))) where {Ba<:AbstractHankelBasis}
    A,dA,ddA=construct_matrices(solver,basis,pts,k;kernel_fun=kernel_fun,multithreaded=multithreaded)
    CT=eltype(A)
    n=size(A,1)
    @blas_multi MAX_BLAS_THREADS F=lu!(A) # enables fast solves with A (shift–invert) by creating triangular matrices to internally act on vectors. This is an expensive n(O^3) operation. Reuses A's storage; adjoint(F) gives fast solves with A'. We use lu! since A is not reused in this scope 
    @blas_1 begin
        Ft=adjoint(F) # define outside op_l! for reuse 
        dAt=adjoint(dA) # define outside op_l! for reuse
        tmp=zeros(CT,n) # # reusable work buffer to avoid allocations in operator applications
        # shift–invert map C := A^{-1} dA 
        # Mathematics: linearize A(k+ε) ≈ A + ε dA. Singularity A+ε dA ≈ 0 -> A v = -ε dA v ⇒ (generalized EVP) A v = λ dA v with λ=-ε
        # Hence (A^{-1} dA) v = μ v with μ = 1/λ. Small |λ| correspond to large |μ|
        function op_r!(y,x)
            mul!(y,dA,x)  # y <- dA * x  without extra allocations
            ldiv!(F,y) # y <- A \ y  (using LU) without extra allocations
            return y
        end
        C=LinearMaps.LinearMap{CT}(op_r!,n,n;ismutating=true) # LinearMaps wraps the op for Krylov without forming A^{-1}dA explicitly. Crucial to reduce allocations
        μr,VRlist,_=eigsolve(C,n,nev,:LM;tol=tol,maxiter=maxiter,krylovdim=krylovdim) # compute the largest |μ| -> smallest |λ|
        λ=inv.(μr) # # map back via λ = 1/μ                          
        ord=sortperm(abs.(λ))
        λ=λ[ord]
        μr=μr[ord]
        VRlist=VRlist[ord]
        # left shift–invert map C_L = (A')^{-1} (dA') acting on column vectors u. This is solving the adjoint eigenproblem
        # If C u = μ u is the right EVP, then (A')^{-1} (dA') u = μ u gives the corresponding left EVP for the pair (A,dA).
        # Those u are (up to scaling) left generalized eigenvectors of A v = λ dA v: u' A = λ u' dA with λ = 1/μ.
        function op_l!(y,x)
            copyto!(tmp,x) # tmp <- x  (so we can reuse tmp in-place)
            ldiv!(Ft,tmp)  # tmp <- (A') \ tmp without extra allocations
            mul!(y,dAt,tmp) # y <- (dA') * tmp = (dA') * (A')^{-1} * x  without extra allocations
            return y
        end
        Cl=LinearMaps.LinearMap{CT}(op_l!,n,n;ismutating=true) # adjoint-side LinearMap (no explicit transposed matrices formed beyond dA', A')
        #w0=zeros(CT,n);randn!(rng,w0) # random complex starting vector for krylov
        μl,ULlist,_=eigsolve(Cl,n,nev,:LM;tol=tol,maxiter=maxiter,krylovdim=krylovdim) # left eigenvalues should match μr (up to num. noise), reuse v0
        # Pair left and right sets by closeness in μ (using conjugation to be robust for complex arithmetic)
        perm=@inbounds [argmin(abs.(μl.-conj(μrj))) for μrj in μr] # if stable solve then this should perfectly align the left and right eigenvectors
        ULlist=ULlist[perm]
        RT=real(CT)
        λ_out=Vector{RT}(undef,nev) # at most we will have nev
        tens_out=Vector{RT}(undef,nev)
        m=0 # keeps track of valid eigvals, if in the end 0 empty interval
        buf=zeros(CT,n) # reusable temp array used with mul! to always overwrite previous result
        @inbounds for j in 1:nev
            λj=λ[j]
            v=VRlist[j];u=ULlist[j]
            @blas_multi MAX_BLAS_THREADS mul!(buf,ddA,v) # buf <- ddA * v
            num=dot(u,buf)  # numerator = u' * ddA * v
            @blas_multi MAX_BLAS_THREADS mul!(buf,dA,v)  # buf <- dA * v, overwrites previous buf
            den=dot(u,buf)   # denominator = u' * dA * v  (bi-orthogonal pairing; scaling cancels in the ratio)
            # first-order: ε1 = -λ  (since A v = λ dA v with λ = -ε to first order)
            # second-order: ε2 = -0.5 ε1^2 * (u' ddA v)/(u' dA v)
            c1=-real(λj)
            c2=zero(RT)
            if abs(den)>1e-15 # soft guard
                c2-=0.5*c1^2*real(num/den) # second-order correction (scale-invariant thanks to the ratio)
            end
            t=c1+c2
            abst=abs(t)
            if abst<dk # acceptance window in the (Re λ, Im λ) plane
                m+=1
                λ_out[m]=k+t # corrected k = k + ε1 + ε2 
                tens_out[m]=abst # tension ≈ |ε1 + ε2|
            end
        end
        if m==0;return RT[],RT[];end # if it happens to be empty solve in dk, return empty
        resize!(λ_out,m);resize!(tens_out,m) # since nev > expected eigvals in dk due to added padding, trim it
        return λ_out,tens_out
    end
end

##############
#### INFO ####
##############
# INTERNAL FUNCTION THAT GIVES US USEFUL INFORMATION OF THE TIME COMPLEXITY AND STABILITY OF THE ALGORITHM

function solve_krylov_INFO(solver::BoundaryIntegralMethod,basis::Ba,pts::BoundaryPointsBIM,k;kernel_fun::Union{Symbol,Function}=:default,multithreaded::Bool=true) where {Ba<:AbstractHankelBasis}
    t0=time()
    @info "Constructing BIM matrix A(k=$k)…"
    @time A=construct_matrices(solver,basis,pts,k;kernel_fun=kernel_fun,multithreaded)
    @info "A size: $(size(A)) eltype=$(eltype(A)) nnz≈$(count(!iszero,A))"
    t1=time()
    @info "Krylov σmin(A)…"
    @time @blas_multi_then_1 MAX_BLAS_THREADS S,U,V,info=svdsolve(A,1,:SR)
    σ=S[1]; @info "converged=$(info.converged) iters=$(info.numiter) normres=$(info.normres)"
    if !isempty(U) && !isempty(V)
        r=norm(A*V[1]-σ*U[1]); @info "triplet residual ‖A*v-σ*u‖=$r"
    end
    t2=time(); @info "Timings: construct=$(t1-t0)s, krylov=$(t2-t1)s, total=$(t2-t0)s"
    return σ
end

function solve_vect_krylov_INFO(solver::BoundaryIntegralMethod,basis::Ba,pts::BoundaryPointsBIM,k;kernel_fun::Union{Symbol,Function}=:default,multithreaded::Bool=true) where {Ba<:AbstractHankelBasis}
    t0=time(); @info "Constructing BIM matrix A(k=$k)…"
    @time A=construct_matrices(solver,basis,pts,k;kernel_fun=kernel_fun,multithreaded)
    @info "A size: $(size(A)) eltype=$(eltype(A))"
    t1=time(); @info "Krylov σmin(A) + right vector…"
    @time @blas_multi_then_1 MAX_BLAS_THREADS S,U,V,info=svdsolve(A,1,:SR)
    σ=S[1]; v=V[1]
    @info "converged=$(info.converged) iters=$(info.numiter) normres=$(info.normres)"
    if !isempty(U)
        r=norm(A*v-σ*U[1]); @info "triplet residual ‖A*v-σ*u‖=$r"
    else
        r=norm(A*v); @info "vector residual proxy ‖A*v‖=$r"
    end
    t2=time(); @info "Timings: construct=$(t1-t0)s, krylov=$(t2-t1)s, total=$(t2-t0)s"
    return σ,v
end

function solve_krylov_INFO(solver::ExpandedBoundaryIntegralMethod,basis::Ba,pts::BoundaryPointsBIM,k,dk; kernel_fun::Union{Tuple{Symbol,Symbol,Symbol},Tuple{Function,Function,Function}}=(:default,:first,:second),multithreaded::Bool=true,nev::Int=5,tol=1e-14,maxiter=5000,krylovdim::Int=min(40,max(40,2*nev+1))) where {Ba<:AbstractHankelBasis}
    t0=time(); @info "Constructing A,dA,ddA at k=$k…"
    @time A,dA,ddA=construct_matrices(solver,basis,pts,k;kernel_fun=kernel_fun,multithreaded)
    @blas_multi MAX_BLAS_THREADS begin
        n=size(A,1); T=eltype(A); @info "Sizes: A=$(size(A)) dA=$(size(dA)) ddA=$(size(ddA)) eltype=$T"
        t1=time(); @info "LU factorization of A (shift–invert)…"
        @time F=lu!(A); Ft=adjoint(F); dAt=adjoint(dA)
    end
    @blas_1 begin 
        tmp=zeros(T,n); buf=zeros(T,n)
        # right op y = A^{-1} dA x
        function op_r!(y,x); mul!(y,dA,x); ldiv!(F,y); y; end
        C=LinearMaps.LinearMap{T}(op_r!,n,n;ismutating=true)
        @info "Right eigsolve on A^{-1} dA (nev=$nev tol=$tol krylovdim=$krylovdim)…"
        @time μr,VR,infoR=eigsolve(C,n,nev,:LM;tol,maxiter,krylovdim)
        λ=inv.(μr); ord=sortperm(abs.(λ)); λ=λ[ord]; μr=μr[ord]; VR=VR[ord]
        @info "Right eigsolve: converged=$(infoR.converged) iters=$(infoR.numiter)"
        # left op y = (A')^{-1} (dA') x
        function op_l!(y,x); copyto!(tmp,x); ldiv!(Ft,tmp); mul!(y,dAt,tmp); y; end
        Cl=LinearMaps.LinearMap{T}(op_l!,n,n;ismutating=true)
        @info "Left eigsolve on (A')^{-1} (dA')…"
        @time μl,UL,infoL=eigsolve(Cl,n,nev,:LM;tol,maxiter,krylovdim)
        @info "Left eigsolve: converged=$(infoL.converged) iters=$(infoL.numiter)"
        # pair left/right
        perm=@inbounds [argmin(abs.(μl.-conj(μrj))) for μrj in μr]; UL=UL[perm]
        # accept window
        acc=@. (abs(real(λ))<dk) & (abs(imag(λ))<dk)
        nacc=count(acc); @info "Accepted in dk-window: $nacc / $nev"
        if nacc==0
            t2=time(); @info "Timings: construct=$(t1-t0)s, factor+eigs=$(t2-t1)s, total=$(t2-t0)s"
            return real(T)[],real(T)[]
        end
        λ=λ[acc];VR=VR[acc];UL=UL[acc]
        κ_all=gev_eigconds(A,dA,λ,VR,UL;p=2)
        @info "Median eigenvalue condition number: $(median(κ_all))"
        @info "Median lower bound on relative eigenvalue error: $(median(rel_bound_all))"
        # 2nd-order corrections
        λ_out=Vector{real(T)}(undef,nacc); tens=similar(λ_out); m=0
        @info "Second-order corrections…"
        @time for j in 1:nacc
            v=VR[j]; u=UL[j]; λj=λ[j]
            mul!(buf,ddA,v); num=dot(u,buf)
            mul!(buf,dA,v);  den=dot(u,buf)
            c1=-real(λj); c2=-0.5*c1^2*real(num/den)
            m+=1; λ_out[m]=k+c1+c2; tens[m]=abs(c1+c2)
        end
        t2=time(); @info "Timings: construct=$(t1-t0)s, factor+eigs=$(t2-(t1))s, corrections=$(time()-t2)s, total=$(time()-t0)s"
        return λ_out,tens
    end
end

#### NOT YET IMPLEMENTED ####

#TODO EBIM - L and R eigenvectors simultaneously when compatibility is resolved for KrylovKit > 10.1 we can use BiArnoldi method below - bieigsolve that is designed for non-hermitian EVP and will give simultaneously both left and right eigenvectors. Currently we do 2 separate solver for left and right eigenvectors.
#=
function solve_krylov_biarnoldi(solver::ExpandedBoundaryIntegralMethod,basis::Ba,pts::BoundaryPointsBIM,k,dk;kernel_fun::Union{Tuple{Symbol,Symbol,Symbol},Tuple{Function,Function,Function}}=(:default,:first,:second),multithreaded::Bool=true,nev::Int=5,tol::Real=1e-14,maxiter::Int=5000,krylovdim::Int=max(40,min(80,2*nev+1))) where {Ba<:AbstractHankelBasis}
    A,dA,ddA=construct_matrices(solver,basis,pts,k;kernel_fun,multithreaded)
    CT=eltype(A)
    n=size(A,1)
    F=lu!(A)
    Ft=adjoint(F)
    dAt=adjoint(dA)
    # Right map: y = A^{-1} (dA * x)
    op_r=function (x)
        y=similar(x)
        mul!(y,dA,x) # y <- dA*x
        ldiv!(F,y) # y <- A\y
        return y
    end
    # Left map: y = (A')^{-1} (dA') * x
    op_l=function (x)
        t=copy(x) # temp to solve Ft \ x
        ldiv!(Ft,t) # t <- (A')\x
        y=similar(x)
        mul!(y,dAt,t) # y <- (dA')*t
        return y
    end
    v0=randn(CT,n)
    w0=randn(CT,n)
    μ,(Vlist, Wlist),(infoR, infoL)=bieigsolve((op_r,op_l),v0,w0,nev,:LM;tol=tol,maxiter=maxiter,krylovdim=krylovdim)
    if infoR.converged<1
        return real(CT)[],real(CT)[]
    end
    # Map back: λ = 1/μ (generalized EVP A v = λ dA v), sort by |λ|
    λ=inv.(μ)
    ord=sortperm(abs.(λ))
    λ.=λ[ord]
    Vlist=Vlist[ord]
    Wlist=Wlist[ord]
    RT=real(CT)
    λ_out=Vector{RT}(undef,nev)
    tens=Vector{RT}(undef,nev)
    m=0
    buf=zeros(CT,n)# work buffer
    @inbounds for j in 1:nev
        λj=λ[j]
        if abs(real(λj))<dk && abs(imag(λj))<dk
            v=Vlist[j]
            u=Wlist[j]
            mul!(buf,ddA,v)
            num=dot(u,buf) # u' * ddA * v
            mul!(buf,dA,v)
            den=dot(u,buf) # u' * dA  * v
            c1=-real(λj)
            c2=-0.5*c1^2*real(num/den)
            m+=1
            λ_out[m]=k+c1+c2
            tens[m]=abs(c1+c2)
        end
    end
    if m==0
        return RT[], RT[]
    else
        resize!(λ_out,m)
        resize!(tens,m)
        return λ_out,tens
    end
end
=#