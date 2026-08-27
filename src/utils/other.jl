@inline midpoints(v::AbstractVector)=(v[1:end-1].+v[2:end])./2
@inline midpoints(r::AbstractRange)=r[1:end-1].+step(r)/2

function median(v::AbstractVector{T}) where {T<:Real}
    n=length(v)
    w=copy(v)
    if isodd(n)
        return partialsort!(w,(n+1)÷2)
    else
        k=n÷2
        m1=partialsort!(w,k)
        m2=partialsort!(w,k+1)
        return (m1+m2)/2
    end
end

function quantile(v::AbstractVector{T},p::Real) where {T<:Real}
    n=length(v);n==1&&return v[firstindex(v)]
    h=1+(n-1)*p;k=floor(Int,h);γ=h-k
    w=copy(v);x0=partialsort!(w,k)
    k==n&&return x0
    x1=partialsort!(w,k+1)
    return x0+γ*(x1-x0)
end

# Batch n objects into batches of size batch_size
@inline function _nbatches(n::Int,batch_size::Int)
    return cld(n,batch_size)
end
# Get the first index of the b-th batch
@inline _batch_first(b::Int,batch_size::Int)=1+(b-1)*batch_size
# Get the last index of the b-th batch, ensuring it does not exceed n
@inline _batch_last(b::Int,batch_size::Int,n::Int)=min(b*batch_size,n)