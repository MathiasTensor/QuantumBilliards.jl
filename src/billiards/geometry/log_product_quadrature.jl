############################
##### LOG PRODUCT QUAD #####
############################
# EDIT MO 27/5/26 

"""
    log_moments_big(x::BigFloat,n::Int)

Moments M_p(x) up to degree n-1 for handling logarithmic singularities on the same panel.

Compute exact logarithmic moments

    m[r] = ∫_{-1}^{1} t^(r-1) log|t-x| dt,   r=1,...,n,

or equivalently, with zero-based polynomial degree p=r-1,

    M_p(x) = ∫_{-1}^{1} t^p log|t-x| dt,     p=0,...,n-1.

The first entry is the p=0 moment:

    M_0(x) = ∫ log|t-x| dt = (1-x)log|1-x| + (1+x)log|1+x| - 2.

For p≥1, one expands

    t^p = (u+x)^p,  u=t-x,

so

    M_p(x) = Σ_{q=0}^{p} binomial(p,q) x^(p-q) ∫ u^q log|u| du,

where

    ∫ u^q log|u| du = u^(q+1) [ log|u|/(q+1) - 1/(q+1)^2 ].

BigFloat is used due to ill-conditioning as `ngl` grows.
"""
function log_moments_big(x::BigFloat,n::Int)
    m=Vector{BigFloat}(undef,n)
    a=big(1)-x;b=big(1)+x
    m[1]=a*log(abs(a))+b*log(abs(b))-big(2)
    uL=-big(1)-x;uR=big(1)-x
    for p in 1:n-1
        s=big(0)
        for q in 0:p
            p=big(p);q=big(q)
            coeff=BigFloat(binomial(p,q))*x^(p-q)
            qq=BigFloat(q+1)
            F(u)=abs(u)<eps(BigFloat) ? big(0) : u^(q+1)*(log(abs(u))/qq-inv(qq^2))
            s+=coeff*(F(uR)-F(uL))
        end
        m[p+1]=s
    end
    return m
end

"""
    log_weights_matrix(::Type{T},ξ::Vector{T};prec::Int=256) where {T<:Real}

Construct the logarithmic product-integration weight matrix Λ.

Given interpolation nodes:

    ξ₁,...,ξ_n ∈ [-1,1],

we seek weights satisfying:

    ∫_{-1}^{1} p(t) log|t-ξ_i| dt
      =
      Σ_j Λ[i,j] p(ξ_j)

for every polynomial p (t) = Σ_{k=0}^{n-1} c_k t^k of degree ≤ n−1.
Using t = ξ_i we get for p(ξ_i) = Σ_{k=0}^{n-1} c_k ξ_i^k
where the Vandermonde matrix is V[j,k] = ξ_j^(k-1) and the exact 
logarithmic moments are m_i[k] = ∫ t^(k-1) log|t-ξ_i| dt.

That means Λ exactly integrates the logarithmic singularity
against polynomial interpolants up to degree n-1.

Exact logarithmic moments are:

    m_i[k] = ∫ t^(k-1) log|t-ξ_i| dt.

We require Σ_j Λ[i,j] p(ξ_j) = m_i[k] for all p of degree ≤ n-1, which means:

    Λ_i V = m_i^T

so we can use Julia's backslash operator to solve for the log weights:

    Λ_i = m_i^T V^{-1}.

Each row is therefore obtained by solving: V^T λ = m.
"""
function log_weights_matrix(::Type{T},ξ::Vector{T};prec::Int=256) where {T<:Real}
    setprecision(BigFloat,prec) do
        n=length(ξ)
        xb=BigFloat.(ξ)
        V=Matrix{BigFloat}(undef,n,n)
        @inbounds for i in 1:n
            V[i,1]=big(1)
            for j in 2:n
                V[i,j]=V[i,j-1]*xb[i]
            end
        end
        Λb=Matrix{BigFloat}(undef,n,n)
        @inbounds for i in 1:n
            m=log_moments_big(xb[i],n)
            Λb[i,:].=transpose(V)\m
        end
        return T.(Λb)
    end
end