# ============================================================
# PANEL GRADING FOR ALPERT QUADRATURE (OPEN PANELS)
# ============================================================
# This section defines the endpoint grading used for Alpert
# quadrature on open panels.
#
# We use a symmetric algebraic grading map:
#
#     u(σ) = σ^q / (σ^q + (1 - σ)^q),   σ ∈ [0,1],  q ≥ 1
#
# where:
#   - σ is the uniform computational parameter (midpoint grid)
#   - u is the physical panel parameter
#   - q controls clustering strength near endpoints
#
# Properties:
#   - u(0) = 0, u(1) = 1
#   - symmetric clustering near both endpoints
#   - smooth for σ ∈ (0,1)
#   - du/dσ → 0 at σ → 0,1 for q > 1
# ============================================================

# u(σ): grading map from computational parameter σ to panel u
@inline function _panel_grade_map(σ::T,q::Int) where {T<:Real}
    a=σ^q
    b=(one(T)-σ)^q
    return a/(a+b)
end
# du/dσ: first derivative of grading map. This is the Jacobian used to scale tangents and arc-length.
# Closed form: du/dσ = q σ^(q-1) (1-σ)^(q-1) / (σ^q + (1-σ)^q)^2
@inline function _panel_grade_map_prime(σ::T,q::Int) where {T<:Real}
    a=σ^q
    b=(one(T)-σ)^q
    den=a+b
    return q*σ^(q-1)*(one(T)-σ)^(q-1)/(den^2)
end
# d²u/dσ²: second derivative of grading map
# Needed for transforming second derivatives of geometry:
#   γ''(σ) = γ''(u)*(du/dσ)^2 + γ'(u)*(d²u/dσ²)
# Used in tangent_2 construction.
@inline function _panel_grade_map_doubleprime(σ::T,q::Int) where {T<:Real}
    qT=T(q)
    omσ=one(T)-σ
    a=σ^q
    b=omσ^q
    den=a+b
    ap=qT*σ^(q-1)
    bp= -qT*omσ^(q-1)
    denp=ap+bp
    num=qT*σ^(qT-1)*omσ^(qT-1)
    nump=qT*(qT-1)*σ^(qT-2)*omσ^(qT-2)*(one(T)-2*σ)
    return (nump*den-2*num*denp)/(den^3)
end