#  Reference:
#      Two-end panel grading for piecewise smooth boundary segments.
#
#  The grading map is defined on the panel computational variable σ ∈ [0,1] by
#
#      u = w(σ) = I_σ(q,q),
#
#  where I_σ(q,q) is the regularized incomplete beta function.
#
#  This gives symmetric endpoint clustering:
#
#      w'(σ) ~ σ^(q-1)      as σ -> 0,
#      w'(σ) ~ (1-σ)^(q-1)  as σ -> 1.
#
#  The midpoint computational nodes are
#
#      σ_j = (j - 1/2)/N,   j=1,...,N,
#
#  and are mapped to the physical panel parameter u_j = w(σ_j).
#
#  Returned quantities:
#      σ   = midpoint computational nodes in [0,1]
#      u   = graded panel nodes in [0,1]
#      a   = w'(σ)
#      a2  = w''(σ)
#      wq  = h * w'(σ),   with h = 1/N

# regularized incomplete beta grading map
@inline function _panel_kress_w(s::T,q::T) where {T<:Real}
    return T(beta_inc(q,q,s)[1])
end
# w'(σ)=σ^(q-1)(1-σ)^(q-1)/B(q,q)
@inline function _panel_kress_wprime(s::T,q::T) where {T<:Real}
    Bqq=T(beta(q,q)[1])
    return s^(q-one(T))*(one(T)-s)^(q-one(T))/Bqq
end
# w''(σ)=(q-1)(1-2σ)σ^(q-2)(1-σ)^(q-2)/B(q,q)
@inline function _panel_kress_wdoubleprime(s::T,q::T) where {T<:Real}
    if q<=one(T)
        return zero(T)
    end
    Bqq=T(beta(q,q))
    return (q-one(T))*(one(T)-T(2)*s)*s^(q-T(2))*(one(T)-s)^(q-T(2))/Bqq
end
# midpoint nodes on [0,1] with two-end panel grading
function panel_kress_graded_nodes_data(::Type{T},N::Int;q=4) where {T<:Real}
    N>=1 || error("N must be positive.")
    qT=T(q)
    qT>one(T) || error("Require q>1 for panel grading.")
    σ=Vector{T}(undef,N)
    u=Vector{T}(undef,N)
    a=Vector{T}(undef,N)
    a2=Vector{T}(undef,N)
    wq=Vector{T}(undef,N)
    h=inv(T(N))
    @inbounds for j in 1:N
        σ[j]=(T(j)-T(1)/T(2))*h
        u[j]=_panel_kress_w(σ[j],qT)
        a[j]=_panel_kress_wprime(σ[j],qT)
        a2[j]=_panel_kress_wdoubleprime(σ[j],qT)
        wq[j]=h*a[j]
    end
    return σ,u,a,a2,wq
end

