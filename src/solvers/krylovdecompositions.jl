using KrylovKit
using LinearMaps
using LinearAlgebra

function solve_krylov(solver::BoundaryIntegralMethod,basis::Ba,pts::BoundaryPointsBIM,k;kernel_fun::Union{Symbol,Function}=:default,multithreaded::Bool=true) where {Ba<:AbstractHankelBasis}
    A=construct_matrices(solver,basis,pts,k;kernel_fun=kernel_fun,multithreaded=multithreaded)
    vals,_,_,_=svdsolve(A,1,:SR)
    return vals[1]
end

function solve_vect_krylov(solver::BoundaryIntegralMethod,basis::Ba,pts::BoundaryPointsBIM,k;kernel_fun::Union{Symbol,Function}=:default,multithreaded::Bool=true) where {Ba<:AbstractHankelBasis}
    A=construct_matrices(solver,basis,pts,k;kernel_fun=kernel_fun,multithreaded=multithreaded)
    vals,_,rvect,_=svdsolve(A,1,:SR) # take the lowest singular value and the associated eigenvector, might add option to choose the number of lowest lying singualr values for symmetry reasons.
    return vals[1],rvect[1]
end

function solve_krylov(solver::ExpandedBoundaryIntegralMethod,basis::Ba,pts::BoundaryPointsBIM,k,dk;kernel_fun::Union{Tuple{Symbol,Symbol,Symbol},Tuple{Function,Function,Function}}=(:default,:first,:second),multithreaded::Bool=true,nev::Int=5,tol=1e-14,maxiter=5000,krylovdim::Int=min(40,max(40,2*nev+1))) where {Ba<:AbstractHankelBasis}
    A,dA,ddA=construct_matrices(solver,basis,pts,k;kernel_fun=kernel_fun,multithreaded=multithreaded)
    CT=eltype(A)
    n=size(A,1)
    F=lu!(A) # enables fast solves with A (shift–invert) by creating triangular matrices to internally act on vectors. This is an expensive n(O^3) operation. Reuses A's storage; adjoint(F) gives fast solves with A'. We use lu! since A is not reused in this scope
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
    C=LinearMap{CT}(op_r!,n,n;ismutating=true) # LinearMaps wraps the op for Krylov without forming A^{-1}dA explicitly. Crucial to reduce allocations
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
    Cl=LinearMap{CT}(op_l!,n,n;ismutating=true) # adjoint-side LinearMap (no explicit transposed matrices formed beyond dA', A')
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
        if abs(real(λj))<dk && abs(imag(λj))<dk # rectangular acceptance window in the (Re λ, Im λ) plane
            v=VRlist[j];u=ULlist[j]
            mul!(buf,ddA,v) # buf ← ddA * v
            num=dot(u,buf)  # numerator = u' * ddA * v
            mul!(buf,dA,v)  # buf ← dA * v, overwrites previous buf
            den=dot(u,buf)   # denominator = u' * dA * v  (bi-orthogonal pairing; scaling cancels in the ratio)
            # first-order: ε1 = -λ  (since A v = λ dA v with λ = -ε to first order)
            # second-order: ε2 = -0.5 ε1^2 * (u' ddA v)/(u' dA v)
            c1=-real(λj)
            c2=-0.5*c1^2*real(num/den) # second-order correction (scale-invariant thanks to the ratio)
            m+=1
            λ_out[m]=k+c1+c2 # corrected k = k + ε1 + ε2 
            tens_out[m]=abs(c1+c2) # tension ≈ |ε1 + ε2|
        end
    end
    if m==0;return RT[],RT[];end # if it happens to be empty solve in dk, return empty
    resize!(λ_out,m);resize!(tens_out,m) # since nev > expected eigvals in dk due to added padding, trim it
    return λ_out,tens_out
end

function solve_vect_krylov(solver::ParticularSolutionsMethod,basis::Ba,pts::PointsPSM,k;multithreaded::Bool=true,maxiter=5000) where {Ba<:AbsBasis}
    tol=1e-14
    B,C=construct_matrices(solver,basis,pts,k;multithreaded)
    T=eltype(B)
    F=qr(C,ColumnNorm()) # rank-revealing QR with column pivoting: C*P = Q*R. This is the main trick
    R=UpperTriangular(F.R) # for fast triangular solves, just in case API changes
    piv=F.p # permutation vector piv such that C[:,piv] = Q*R
    r=findlast(i->abs(R[i,i])>tol*abs(R[1,1]),1:min(size(R)...)) # numerical rank r: keep diagonal entries of R down to a relative threshold and discard near-null interior directions that cause spurious minima
    isnothing(r) && return (Inf,zeros(T,size(B,2))) # in case degenerate fail    
    Rr=R[1:r,1:r] # well-determined r×r block on Q
    Br=B[:,piv[1:r]]/Rr # Br = B[:,piv[1:r]] * Rr^{-1} via triangular solve (stable, no inv)
    vals,_,rvect,_=svdsolve(Br,1,:SR,tol=tol,maxiter=maxiter) # take the lowest singular value and the associated eigenvector, might add option to choose the number of lowest lying singualr values for symmetry reasons.
    y=real.(rvect[1])
    chat=zeros(T,size(B,2))
    chat[piv[1:r]]=Rr\y  # back-substitute: c[piv[1:r]]=Rr^{-1} y; rest are zeros
    return vals[1],chat
end