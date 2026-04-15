using LinearAlgebra,StaticArrays,TimerOutputs,Bessels,ProgressMeter
using KrylovKit
using LinearMaps

const EBIMSolver=Union{BoundaryIntegralMethod,CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners,CFIE_alpert}

function solve_full!(solver::EBIMSolver,A::AbstractMatrix{Complex{T}},dA::AbstractMatrix{Complex{T}},ddA::AbstractMatrix{Complex{T}},pts,k,dk;use_lapack_raw::Bool=false,multithreaded::Bool=true) where {T<:Real}
    if use_lapack_raw
        @blas_multi MAX_BLAS_THREADS λ,VR,VL=generalized_eigen_all_LAPACK_LEGACY(A,dA)
    else
        @blas_multi MAX_BLAS_THREADS λ,VR,VL=generalized_eigen_all(A,dA)
    end
    RT=eltype(real.(λ))
    valid=(abs.(real.(λ)).<dk).&(abs.(imag.(λ)).<dk)
    if !any(valid)
        return Vector{RT}(),Vector{RT}()
    end
    λ=real.(λ[valid])
    VR=VR[:,valid]
    VL=VL[:,valid]
    corr_1=Vector{RT}(undef,length(λ))
    corr_2=Vector{RT}(undef,length(λ))
    buf=similar(VR,eltype(ddA),size(ddA,1))  # reusable buffer
    for i in eachindex(λ)
        v_right=VR[:,i]
        v_left=VL[:,i]
        # ddA * v_right → buf
        mul!(buf,ddA,v_right)
        numerator=dot(v_left,buf)   # u† * (ddA v)
        # dA * v_right → buf (overwrite)
        mul!(buf,dA,v_right)
        denominator=dot(v_left, buf) # u† * (dA v)
        corr_1[i]= -λ[i]
        # optional safety guard 
        if abs(denominator)>1e-15
            corr_2[i]= -0.5*corr_1[i]^2*real(numerator/denominator)
        else
            corr_2[i]=zero(RT)
        end
    end
    λ_corrected=k.+corr_1.+corr_2
    tens=abs.(corr_1.+corr_2)
    return λ_corrected,tens
end

function solve_krylov!(solver::EBIMSolver,A::AbstractMatrix{Complex{T}},dA::AbstractMatrix{Complex{T}},ddA::AbstractMatrix{Complex{T}},pts,k,dk;multithreaded::Bool=true,nev::Int=5,tol=1e-14,maxiter=5000,krylovdim::Int=min(40,max(40,2*nev+1))) where {T<:Real}
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

function solve_full_INFO!(solver::EBIMSolver,A::AbstractMatrix{Complex{T}},dA::AbstractMatrix{Complex{T}},ddA::AbstractMatrix{Complex{T}},pts,k,dk;use_lapack_raw::Bool=false,multithreaded::Bool=true) where {T<:Real}
    s=time()
    s_constr=time()
    basis=AbstractHankelBasis()
    fill!(A,Complex{T}(zero(T),zero(T)))
    fill!(dA,Complex{T}(zero(T),zero(T)))
    fill!(ddA,Complex{T}(zero(T),zero(T)))
    @info "Constructing A,dA,ddA Fredholm matrix and its derivatives..."
    @time construct_matrices!(solver,basis,A,dA,ddA,pts,k;multithreaded=multithreaded)
    e_constr=time()
    if use_lapack_raw
        if LAPACK.version()<v"3.6.0"
            s_gev=time()
            @info "Doing ggev!"
            @info "Matrix condition numbers: cond(A) = $(cond(A)), cond(dA) = $(cond(dA))"
            @blas_multi MAX_BLAS_THREADS α,β,VL,VR=LAPACK.ggev!('V','V',copy(A),copy(dA))
            e_gev=time()
        else
            s_gev=time()
            @info "Doing ggev3!"
            @info "Matrix condition numbers: cond(A) = $(cond(A)), cond(dA) = $(cond(dA))"
            @blas_multi MAX_BLAS_THREADS α,β,VL,VR=LAPACK.ggev3!('V','V',copy(A),copy(dA))
            e_gev=time()
        end
        λ=α./β
        valid_indices=.!isnan.(λ).&.!isinf.(λ)
        @info "% of valid indices: $(count(valid_indices)/length(λ))"
        λ=λ[valid_indices]
        VR=VR[:,valid_indices]
        VL=VL[:,valid_indices]
        sort_order=sortperm(abs.(λ))
        λ=λ[sort_order]
        @info "Smallest eigenvalue: $(minimum(abs.(λ)))"
        VR=VR[:,sort_order]
        VL=VL[:,sort_order]
        normalize!(VR)
        normalize!(VL)
        κ_all=gev_eigconds(A,dA,λ,VR,VL;p=2)
        @info "Median eigenvalue condition number: $(median(κ_all))"
    else
        @info "Solving Julia's ggev for A,dA"
        s_gev=time()
        @blas_multi MAX_BLAS_THREADS F=eigen(A,dA)
        λ=F.values
        VR=F.vectors
        @info "Solving Julia's ggev for A' and dA' for the left eigenvectors"
        @blas_multi MAX_BLAS_THREADS F_adj=eigen(A',dA')
        e_gev=time()
        VL=F_adj.vectors
        valid_indices=.!isnan.(λ).&.!isinf.(λ)
        @info "Number of valid indices: $(count(valid_indices))"
        λ=λ[valid_indices]
        VR=VR[:,valid_indices]
        VL=VL[:,valid_indices]
        sort_order=sortperm(abs.(λ))
        λ=λ[sort_order]
        @info "Smallest eigenvalue: $(minimum(abs.(λ)))"
        VR=VR[:,sort_order]
        VL=VL[:,sort_order]
        normalize!(VR)
        normalize!(VL)
        κ_all=gev_eigconds(A,dA,λ,VR,VL;p=2)
        @info "Median eigenvalue condition number: $(median(κ_all))"
    end
    L=eltype(real.(λ))
    valid=(abs.(real.(λ)).<dk).&(abs.(imag.(λ)).<dk)
    if !any(valid)
        total_time=time()-s
        @info "Final computation time without extrema of SVD for cond calculation: $(total_time) s"
        println("%%%%% SUMMARY %%%%%")
        println("Percentage of total time (most relevant ones): ")
        println("A,dA,ddA construction: $(100*(e_constr-s_constr)/total_time) %")
        println("Generalized eigen: $(100*(e_gev-s_gev)/total_time) %")
        println("%%%%%%%%%%%%%%%%%%%")
        return Vector{L}(),Vector{L}()
    end
    λ=real.(λ[valid])
    VR=VR[:,valid]
    VL=VL[:,valid]
    corr_1=Vector{L}(undef,length(λ))
    corr_2=Vector{L}(undef,length(λ))
    @info "Corrections to the eigenvalues and eigenvectors..."
    s_corr=time()
    @time for i in eachindex(λ)
        v_right=VR[:,i]
        v_left=VL[:,i]
        t_v_left=transpose(v_left)
        numerator=t_v_left*ddA*v_right
        denominator=t_v_left*dA*v_right
        @info "Denominator for index $i : $denominator"
        corr_1[i]=-λ[i]
        corr_2[i]=-0.5*corr_1[i]^2*real(numerator/denominator)
    end
    e_corr=time()
    λ_corrected=k.+corr_1.+corr_2
    tens=abs.(corr_1.+corr_2)
    e=time()
    total_time=e-s
    @info "Final computation time without extrema of SVD for cond calculation: $(total_time) s"
    println("%%%%% SUMMARY %%%%%")
    println("Percentage of total time (most relevant ones): ")
    println("A,dA,ddA construction: $(100*(e_constr-s_constr)/total_time) %")
    println("Generalized eigen: $(100*(e_gev-s_gev)/total_time) %")
    println("2nd order corrections: $(100*(e_corr-s_corr)/total_time) %")
    println("%%%%%%%%%%%%%%%%%%%")
    return λ_corrected,tens
end

function solve_krylov_INFO!(solver::EBIMSolver,A::AbstractMatrix{Complex{T}},dA::AbstractMatrix{Complex{T}},ddA::AbstractMatrix{Complex{T}},pts,k,dk;multithreaded::Bool=true,nev::Int=5,tol=1e-14,maxiter=5000,krylovdim::Int=min(40,max(40,2*nev+1))) where {T<:Real}
    fill!(A,Complex{T}(zero(T),zero(T)))
    fill!(dA,Complex{T}(zero(T),zero(T)))
    fill!(ddA,Complex{T}(zero(T),zero(T)))
    basis=AbstractHankelBasis()
    t0=time()
    @info "Constructing A,dA,ddA at k=$k..."
    @time construct_matrices!(solver,basis,A,dA,ddA,pts,k;multithreaded=multithreaded)
    n=size(A,1)
    CT=eltype(A)
    RT=real(eltype(A))
    @info "Sizes: A=$(size(A)) dA=$(size(dA)) ddA=$(size(ddA)) eltype=$CT"
    Afac=copy(A)
    @info "LU factorization of A (shift-invert)..."
    t1=time()
    @time F=lu!(Afac)
    Ft=adjoint(F)
    dAt=adjoint(dA)
    tmp=zeros(CT,n)
    buf=zeros(CT,n)
    function op_r!(y,x)
        mul!(y,dA,x)
        ldiv!(F,y)
        return y
    end
    C=LinearMaps.LinearMap{CT}(op_r!,n,n;ismutating=true)
    @info "Right eigsolve on A^{-1}dA (nev=$nev tol=$tol krylovdim=$krylovdim)..."
    @time μr,VR,infoR=eigsolve(C,n,nev,:LM;tol=tol,maxiter=maxiter,krylovdim=krylovdim)
    λ=inv.(μr)
    ord=sortperm(abs.(λ))
    λ=λ[ord]
    μr=μr[ord]
    VR=VR[ord]
    @info "Right eigsolve: converged=$(infoR.converged) iters=$(infoR.numiter)"
    function op_l!(y,x)
        copyto!(tmp,x)
        ldiv!(Ft,tmp)
        mul!(y,dAt,tmp)
        return y
    end
    Cl=LinearMaps.LinearMap{CT}(op_l!,n,n;ismutating=true)
    @info "Left eigsolve on (A')^{-1}(dA')..."
    @time μl,UL,infoL=eigsolve(Cl,n,nev,:LM;tol=tol,maxiter=maxiter,krylovdim=krylovdim)
    @info "Left eigsolve: converged=$(infoL.converged) iters=$(infoL.numiter)"
    perm=@inbounds [argmin(abs.(μl.-conj(μrj))) for μrj in μr]
    UL=UL[perm]
    acc=@. (abs(real(λ))<dk) & (abs(imag(λ))<dk)
    nacc=count(acc)
    @info "Accepted in dk-window: $nacc / $nev"
    if nacc==0
        t2=time()
        @info "Timings: construct=$(t1-t0)s, factor+eigs=$(t2-t1)s, total=$(t2-t0)s"
        return RT[],RT[]
    end
    λ=λ[acc]
    VR=VR[acc]
    UL=UL[acc]
    κ_all=gev_eigconds(A,dA,λ,VR,UL;p=2)
    rel_bound_all=κ_all.*eps(RT)
    @info "Median eigenvalue condition number: $(median(κ_all))"
    @info "Median lower bound on relative eigenvalue error: $(median(rel_bound_all))"
    λ_out=Vector{RT}(undef,nacc)
    tens=Vector{RT}(undef,nacc)
    m=0
    @info "Second-order corrections..."
    t2=time()
    @time for j in 1:nacc
        v=VR[j]
        u=UL[j]
        λj=λ[j]
        mul!(buf,ddA,v)
        num=dot(u,buf)
        mul!(buf,dA,v)
        den=dot(u,buf)
        @info "Denominator for index $j : $den"
        c1=-real(λj)
        c2=-0.5*c1^2*real(num/den)
        m+=1
        λ_out[m]=k+c1+c2
        tens[m]=abs(c1+c2)
    end
    t3=time()
    @info "Timings: construct=$(t1-t0)s, factor+eigs=$(t2-t1)s, corrections=$(t3-t2)s, total=$(t3-t0)s"
    return λ_out,tens
end

function solve!(solver::EBIMSolver,A::AbstractMatrix{Complex{T}},dA::AbstractMatrix{Complex{T}},ddA::AbstractMatrix{Complex{T}},pts,k,dk;use_lapack_raw::Bool=false,multithreaded::Bool=true,use_krylov::Bool=true,nev::Int=5) where {T<:Real}
    basis=AbstractHankelBasis()
    @blas_1 construct_matrices!(solver,basis,A,dA,ddA,pts,k;multithreaded=multithreaded)
    if use_krylov
        return solve_krylov!(solver,A,dA,ddA,pts,k,dk;multithreaded=multithreaded,nev=nev)
    else
        return solve_full!(solver,A,dA,ddA,pts,k,dk;use_lapack_raw=use_lapack_raw,multithreaded=multithreaded)
    end
end

function solve_INFO!(solver::EBIMSolver,A::AbstractMatrix{Complex{T}},dA::AbstractMatrix{Complex{T}},ddA::AbstractMatrix{Complex{T}},pts,k,dk;use_lapack_raw::Bool=false,multithreaded::Bool=true,use_krylov::Bool=true) where {T<:Real}
    if use_krylov
        return solve_krylov_INFO!(solver,A,dA,ddA,pts,k,dk;multithreaded=multithreaded)
    else
        return solve_full_INFO!(solver,A,dA,ddA,pts,k,dk;use_lapack_raw=use_lapack_raw,multithreaded=multithreaded)
    end
end

function gev_eigconds(A,B,λ,VR::AbstractMatrix,VL::AbstractMatrix;p=2)
    nA=opnorm(A,p)
    nB=opnorm(B,p)
    n=length(λ)
    κ=Vector{Float64}(undef,n)
    for j in 1:n
        x=VR[:,j]
        y=VL[:,j]
        v=norm(x)*norm(y)
        d=abs(dot(y,B*x))
        κ[j]=(d==0||λ[j]==0) ? Inf : (v*(nA+abs(λ[j])*nB)/(abs(λ[j])*d))
    end
    return κ
end

function gev_eigconds(A,B,λ,VR::Vector{<:AbstractVector},VL::Vector{<:AbstractVector};p=2)
    nA=opnorm(A,p)
    nB=opnorm(B,p)
    n=length(λ)
    κ=Vector{Float64}(undef,n)
    for j in 1:n
        x=VR[j]
        y=VL[j]
        v=norm(x)*norm(y)
        d=abs(dot(y,B*x))
        κ[j]=(d==0||λ[j]==0) ? Inf : (v*(nA+abs(λ[j])*nB)/(abs(λ[j])*d))
    end
    return κ
end