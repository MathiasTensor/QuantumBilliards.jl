using KrylovKit
using LinearMaps
using LinearAlgebra


#### NOT YET IMPLEMENTED ####

#TODO EBIM - L and R eigenvectors simultaneously when compatibility is resolved for KrylovKit > 10.1 we can use BiArnoldi method below - bieigsolve that is designed for non-hermitian EVP and will give simultaneously both left and right eigenvectors. Currently we do 2 separate solver for left and right eigenvectors.
#=
function solve_krylov_biarnoldi(solver::EBIMSolver,basis::Ba,pts::BoundaryPoints,k,dk;multithreaded::Bool=true,nev::Int=5,tol::Real=1e-14,maxiter::Int=5000,krylovdim::Int=max(40,min(80,2*nev+1))) where {Ba<:AbstractHankelBasis}
    A,dA,ddA=construct_matrices(solver,basis,pts,k;multithreaded=multithreaded)
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