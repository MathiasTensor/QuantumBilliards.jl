@inline H(n::Int,x::T) where {T<:Real}=Bessels.hankelh1(n,x)
@inline H(n::Int,x::Complex{T}) where {T<:Real}=SpecialFunctions.besselh(n,1,x)
@inline function hankel_pair01(x);h0=H(0,x);h1=H(1,x);return h0,h1;end