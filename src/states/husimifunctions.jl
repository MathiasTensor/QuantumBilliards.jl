
"""
    antisym_vec(x::AbstractVector) → v::Vector

Antisymmetrically extends a one-sided vector `x` (assumed sorted with
`x[1]` closest to zero) to negative values, returning `[-reverse(x[2:end]);
x]`.

## Description
This is used to build symmetric grids of coherent-state evaluation points (in
`q` or `p`) around zero from a one-sided grid `x`, by mirroring and negating
all but the first entry and prepending the result to `x`.

## Arguments
* `x`: The one-sided vector to extend, typically non-negative and increasing.

## Returns
*  `v` : The antisymmetrically extended vector `[-reverse(x[2:end]); x]`, of length `2*length(x) - 1`.
"""
function antisym_vec(x)
    v = reverse(-x[2:end])
    return append!(v,x)
end

"""
    husimi_function(k, u::AbstractVector, s::AbstractVector, L::Real; c::Real = 10.0, w::Real = 7.0) → (H::Matrix, qs::Vector, ps::Vector)

Computes the boundary Husimi function of the normal-derivative boundary
function `u(s)` (sampled at equally spaced arc-length points `s`) at
wavenumber `k`, on a boundary of total length `L`, returning the Husimi
density `H` on a grid of arc-length coordinates `qs` and momenta `ps`.

## Description
`u` is projected onto (approximately) minimal-uncertainty coherent-state
wavepackets of Gaussian width `sig = 1/sqrt(k)` in the arc-length coordinate,
truncated to `w` widths (`x = s[s .<= w*sig]`) and evaluated with `c` points
per width along the momentum direction. Because the boundary is periodic with
period `L`, each coherent state is periodized by adding its two nearest
periodic images at `s ± L` (`gauss_l`, `gauss_r`) before contraction with
`u`, giving the overlap

```math
h(q,p) = \\sum_{s} u(s)\\, \\big[g(s-q) + g(s-q+L) + g(s-q-L)\\big]\\, e^{i k p (s-q)},
```

with `g(\\cdot) = \\exp(-k\\, (\\cdot)^2/2)` the Gaussian envelope. The Husimi
density is `H(q,p) = a\\, |h(q,p)|^2`, with normalization constant
`a = 1/(2\\pi\\sqrt{\\pi k})` (not normalized so that `H` integrates to `1`).
Momenta are sampled on `ps ∈ [0,1]` in steps of `sig/c` and then mirrored to
`[-1,1]` via [`antisym_vec`](@ref), and arc-length points `qs` are subsampled
from `s` with the same step `sig/c`.

## Arguments
* `k`: The wavenumber of the eigenstate.
* `u`: The (real) boundary normal-derivative function, typically from [`boundary_function`](@ref).
* `s`: The arc-length coordinates of `u`, assumed equally spaced.
* `L`: The total length of the (periodized) boundary.

## Keyword arguments
*  `c::Real = 10.0` : Number of coherent-state evaluation points per Gaussian width `sig`, controlling the resolution of `qs` and `ps`.
*  `w::Real = 7.0` : Truncation width of the coherent-state Gaussian envelope, in units of `sig`.

## Returns
*  `H` : The Husimi density on the grid `(qs, ps)`.
*  `qs` : Arc-length coordinates at which `H` is sampled.
*  `ps` : Momenta (in units of `k`, ranging over `[-1,1]`) at which `H` is sampled.
"""
function husimi_function(k,u,s,L; c = 10.0, w = 7.0)
    #c density of points in coherent state peak, w width in units of sigma
    #L is the boundary length for periodization
    #compute coherrent state weights
    N = length(s)
    sig = one(k)/sqrt(k) #width of the gaussian
    x = s[s.<=w*sig]
    idx = length(x) #do not change order here
    x = antisym_vec(x)
    a = one(k)/(2*pi*sqrt(pi*k)) #normalization factor in this version Hsimi is not noramlized to 1
    ds = (x[end]-x[1])/length(x) #integration weigth
    uc = CircularVector(u) #allows circular indexing
    gauss = @. exp(-k/2*x^2)*ds
    gauss_l = @. exp(-k/2*(x+L)^2)*ds
    gauss_r = @. exp(-k/2*(x-L)^2)*ds
    #construct evaluation points in p coordinate
    ps = collect(range(0.0,1.0,step = sig/c))
    #construct evaluation points in q coordinate
    q_stride = length(s[s.<=sig/c])
    q_idx = collect(1:q_stride:N)
    push!(q_idx,N) #add last point
    qs = s[q_idx]
    #println(length(qs))
    H = zeros(typeof(k),length(qs),length(ps))
    for i in eachindex(ps)   
        cs = @. exp(im*ps[i]*k*x)*gauss + exp(im*ps[i]*k*(x+L))*gauss_l + exp(im*ps[i]*k*(x-L))*gauss_r#imag part of coherent state
        for j in eachindex(q_idx)
            u_w = uc[q_idx[j]-idx+1:q_idx[j]+idx-1] #window with relevant values of u
            h = sum(cs.*u_w)
            #hi = sum(ci.*u_w)
            H[j,i] = a*abs2(h)
        end
    end

    ps = antisym_vec(ps)
    H_ref = reverse(H[:, 2:end]; dims=2)
    H = hcat(H_ref,H)
     
    return H, qs, ps    
end

"""
    husimi_function(state::S; b::Real = 5.0, c::Real = 10.0, w::Real = 7.0) where {S<:AbsState} → (H::Matrix, qs::Vector, ps::Vector)

Computes the boundary Husimi function of an eigenstate directly from `state`,
by combining [`boundary_function`](@ref) and
[`husimi_function(k, u, s, L)`](@ref husimi_function).

## Arguments
* `state`: The eigenstate for which the boundary Husimi function is computed.

## Keyword arguments
*  `b::Real = 5.0` : Oversampling factor passed to [`boundary_function`](@ref) controlling the boundary point density.
*  `c::Real = 10.0` : Number of coherent-state evaluation points per Gaussian width, passed to [`husimi_function(k, u, s, L)`](@ref husimi_function).
*  `w::Real = 7.0` : Truncation width of the coherent-state Gaussian envelope, passed to [`husimi_function(k, u, s, L)`](@ref husimi_function).

## Returns
*  `H` : The Husimi density on the grid `(qs, ps)`.
*  `qs` : Arc-length coordinates at which `H` is sampled.
*  `ps` : Momenta at which `H` is sampled.
"""
function husimi_function(state::S;  b = 5.0, c = 10.0, w = 7.0) where {S<:AbsState}
    L = state.billiard.length
    k = state.k
    u, s, norm = boundary_function(state; b=b)
    return husimi_function(k,u,s,L; c = c, w = w)
end

