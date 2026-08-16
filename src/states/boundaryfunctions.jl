
"""
    regularize!(u::AbstractVector) → Nothing

Removes `NaN` values from a boundary function `u` in place, mutating `u`
directly.

## Description
The boundary normal derivative computed by [`boundary_function`](@ref) can be
`NaN` at points where the boundary normal is not smooth (e.g. at corners of
the domain). Each such entry is replaced by the average of its two nearest
neighbours along the arc-length parametrization of the boundary. Since the
boundary is a closed curve, the first sample's neighbours are its next entry
and the last entry of `u`.

## Arguments
* `u`: The boundary function values to regularize, modified in place.

## Returns
*  `nothing` : `u` is modified in place; no meaningful value is returned.
"""
function regularize!(u)
    idx = findall(isnan, u)
    for i in idx
        if i != 1
            u[i] = (u[i+1] + u[i-1])/2.0
        else
            u[i] = (u[i+1] + u[end])/2.0
        end
    end
end

"""
    _rellich(pts::BoundaryPoints{T}, u::AbstractVector{N}, k::T) where {N<:Number,T<:Real} → norm::T

Evaluates the Rellich-identity normalization of the boundary function `u`
(the normal derivative of an eigenstate) at wavenumber `k`.

## Description
For a Dirichlet eigenstate normalized to unit interior norm, Rellich's
identity relates that normalization to a boundary integral of the normal
derivative:

```math
\\frac{1}{2k^2}\\int_{\\partial\\Omega} (\\mathbf{n}\\cdot\\mathbf{x})\\, |\\partial_n\\psi|^2\\, ds = 1.
```

This function evaluates the discretized left-hand side, providing a cheap
normalization consistency check that is computed alongside
[`boundary_function`](@ref).

## Arguments
* `pts`: The boundary points at which `u` was evaluated, providing the
  positions `xy`, outward unit normals `normal` and arc-length weights `ds`.
* `u`: The boundary function (normal derivative) values at `pts`.
* `k`: The wavenumber of the eigenstate.

## Returns
*  `norm` : The Rellich-normalized boundary integral, expected to be close to
   `1` for a correctly normalized eigenstate.
"""
function _rellich(pts::BoundaryPoints{T},u::AbstractVector{N},k::T) where {N<:Number,T<:Real}
    acc=zero(T)
    @inbounds @simd for i in eachindex(u)
        n=pts.normal[i]
        xy=pts.xy[i]
        w=(n[1]*xy[1]+n[2]*xy[2])*pts.ds[i]
        acc+=w*abs2(u[i])
    end
    return acc/(2*k^2)
end

"""
    boundary_function(state::S; b::Real = 5.0, multithreaded::Bool = true) where {S<:AbsState} → (u::Vector, s::Vector, norm::Real)

Computes the normal-derivative boundary function \$u(s) = \\partial_n\\psi(s)\$ of
an eigenstate along its boundary, together with the arc-length coordinate `s`
and a Rellich-identity normalization check `norm`. The default `b = 5.0`
gives roughly `5` boundary sample points per de Broglie wavelength.

## Description
The boundary is discretized with `FourierNodes` quadrature on
`N = max(round(Int, k*L*b/(2\\pi)), 512)` points, where `L` is the total
boundary length. The gradient matrices of the basis are evaluated at these
points and contracted with the outward unit normal to give

```math
\\partial_n\\psi(s) = n_x(s)\\,\\partial_x\\psi(s) + n_y(s)\\,\\partial_y\\psi(s).
```

[`regularize!`](@ref) then removes any `NaN` artifacts arising at non-smooth
points of the boundary, and [`_rellich`](@ref) evaluates the Rellich-identity
normalization of the resulting function. If the basis carries symmetries,
the boundary points and function are subsequently unfolded onto the full
boundary via `apply_symmetries_to_boundary_points` and
`apply_symmetries_to_boundary_function`.

## Arguments
* `state`: The eigenstate for which the boundary function is computed.

## Keyword arguments
*  `b::Real = 5.0` : Oversampling factor controlling the boundary point
   density; the boundary is sampled at `max(round(Int, k*L*b/(2\\pi)), 512)`
   points.
*  `multithreaded::Bool = true` : Whether the gradient matrix construction is
   multithreaded.

## Returns
*  `u` : The normal-derivative boundary function values (symmetry-unfolded if
   applicable).
*  `s` : The arc-length coordinates corresponding to `u`.
*  `norm` : The Rellich-identity normalization of the boundary function
   computed by [`_rellich`](@ref), before any symmetry unfolding; should be
   close to `1` for a correctly normalized eigenstate.
"""
function boundary_function(state::S; b=5.0, multithreaded = true) where {S<:AbsState}
    let vec = state.vec, k = state.k, k_basis = state.k_basis, new_basis = state.basis, billiard=state.billiard
        type = eltype(vec)
        boundary = get_boundary_curves_with_ignored(billiard)
        crv_lengths = [crv.length for crv in boundary]
        sampler = FourierNodes([2,3,5],crv_lengths) 
        L = CompositeCurve(boundary).length
        N = max(round(Int, k*L*b/(2*pi)), 512)
        pts = boundary_coords(billiard, sampler, N)
        dX, dY = gradient_matrices(new_basis, k_basis, pts.xy; multithreaded)
        nx = getindex.(pts.normal,1)
        ny = getindex.(pts.normal,2)
        dX = nx .* dX 
        dY = ny .* dY
        U::Array{type,2} = dX .+ dY
        u::Vector{type} = U * vec
        regularize!(u)
        #compute the boundary norm
        w = dot.(pts.normal, pts.xy) .* pts.ds
        norm = _rellich(pts, u, k)
        if isnothing(new_basis.symmetries) == false
            pts = apply_symmetries_to_boundary_points(pts, new_basis.symmetries, billiard)
            u = apply_symmetries_to_boundary_function(u, new_basis.symmetries, new_basis.sym_qnumbers)
        end
        #println(norm)
        return u, pts.s::Vector{type}, norm
    end
end

"""
    momentum_function(u::AbstractVector, s::AbstractVector) → (power::Vector, ks::Vector)

Computes the discrete momentum-space representation of a boundary function
`u(s)` sampled at equally spaced arc-length points `s`, returning the
one-sided power spectral density and the corresponding angular wavenumbers.

## Description
The real FFT of `u` is taken with `rfft`, and the associated frequencies from
`rfftfreq` are rescaled to angular wavenumbers via \$k_s = 2\\pi f\$. Each
coefficient is normalized by \$N^2\$ (with `N = length(u)`) and further
divided by the wavenumber spacing \$\\Delta k = 2\\pi/L\$ (with
`L = s[end]` the total arc length), converting the discrete Fourier
coefficients into a power *spectral density* in `k`. All bins except the DC
term (and the Nyquist term, when `N` is even) are doubled to account for the
folded negative-frequency content, giving a one-sided power spectral density
whose integral over `ks` recovers the mean-square value of `u`:

```math
\\int \\mathrm{power}(k)\\, dk \\;\\approx\\; \\sum_j \\mathrm{power}_j\\, \\Delta k \\;=\\; \\frac{1}{L}\\int_0^L |u(s)|^2\\, ds \\;\\approx\\; \\frac{1}{N}\\sum_i |u_i|^2.
```

## Arguments
* `u`: The boundary function values, typically obtained from
  [`boundary_function`](@ref), sampled at equally spaced arc-length points.
* `s`: The arc-length coordinates of `u`; only the sample spacing
  `diff(s)[1]` (for the sampling rate) and `s[end]` (for the total arc
  length `L`) are used.

## Returns
*  `power` : The one-sided power spectral density of `u` as a function of
   `ks`, normalized so that `sum(power) * (ks[2] - ks[1]) ≈ mean(abs2, u)`.
*  `ks` : The angular wavenumbers corresponding to `power`.
"""
function momentum_function(u,s)
    N = length(u)
    fu = rfft(u)
    power = abs2.(fu) ./ N^2 ./ (2*pi/s[end]) 
    power[2:(iseven(N) ? end-1 : end)] .*= 2
    sr = 1.0/diff(s)[1]
    ks = rfftfreq(N,sr).*(2*pi)
    return power, ks
end

"""
    momentum_function(state::S; b::Real = 5.0, multithreaded::Bool = true) where {S<:AbsState} → (power::Vector, ks::Vector)

Computes the momentum-space representation of an eigenstate's boundary
function directly from `state`, by combining [`boundary_function`](@ref) and
[`momentum_function(u, s)`](@ref momentum_function).

## Arguments
* `state`: The eigenstate for which the boundary momentum function is
  computed.

## Keyword arguments
*  `b::Real = 5.0` : Oversampling factor passed to [`boundary_function`](@ref)
   controlling the boundary point density.
*  `multithreaded::Bool = true` : Whether the underlying gradient matrix
   construction is multithreaded.

## Returns
*  `power` : The normalized power spectrum of the boundary function.
*  `ks` : The angular wavenumbers corresponding to `power`.
"""
function momentum_function(state::S; b=5.0, multithreaded = true) where {S<:AbsState}
    u, s, norm = boundary_function(state; b, multithreaded)
    return momentum_function(u,s)
end
