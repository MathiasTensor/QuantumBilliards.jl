################################################################################
# STATE PROJECTIONS FROM PHYSICAL CORNER DENSITIES
#
# In BIMEigenstate, the weighted adjoint CFIE stores physical nodal q=∂ₙψ
# in vec; u, when available, is the same density after Rellich normalization.
# Unlike the primal CFIE combined-layer density, q reconstructs ψ=S q. A linear
# projection uses q ds=Σⱼ bⱼ(t)qⱼ ds, with the source cardinal matrices owned
# by the boundary space. Its Gaussian source measure differs from the Jacobi
# weights that integrate |q|² ds. The same measure reconstructs ψ=S q and
# evaluates boundary Fourier and Poincare-Husimi projections; no resampling or
# second nullspace solve is required. Ordinary BIM representations keep their
# existing state implementations and quadrature.
################################################################################

"""
    _weighted_corner_boundary_quadrature(pts::BoundaryPoints{T},
        q::AbstractVector{<:Number}) -> Tuple{Vector{T},Vector{Complex{T}}}

Sample the physical source measure `q ds` through its weighted panel cardinals.
`pts` owns the original curve charts; `q` contains full-boundary physical nodal
normal derivatives. Return source arc lengths and integrated complex source
amplitudes. Summing `amplitudes .* f.(s)` approximates `∫q(s)f(s)ds` for a smooth
boundary test function. Gaussian source weights already enter the amplitudes;
they must not be multiplied by the separate Jacobi `|q|²` norm weights.
No keywords are required. Inputs are unchanged and returned vectors are owned.
"""
function _weighted_corner_boundary_quadrature(pts::BoundaryPoints{T},q::AbstractVector{<:Number}) where {T<:Real}
    length(q) == length(pts) || throw(DimensionMismatch("boundary projection needs a full physical density"))
    space = pts.density_space::WeightedDensitySpace{T}
    _weighted_corner_source_plan(pts,Int[])
    rules = lock(space.source_lock) do
        [only(space.source_cache[(0,p,panel.source_order)]) for (p,panel) in enumerate(space.panels)]
    end
    offsets = cumsum(vcat(zero(T),T[curve.length for curve in space.curves]))
    count = sum(length(rule.parameters) for rule in rules)
    samples = Vector{T}(undef,count)
    amplitudes = Vector{Complex{T}}(undef,count)
    next = 1
    for (panel,rule) in zip(space.panels,rules)
        indices = next:(next+length(rule.parameters)-1)
        mul!(@view(amplitudes[indices]),rule.basis,@view(q[panel.indices]))
        curve = space.curves[panel.curve_index]
        for t in rule.parameters
            ξ = panel.reverse ? one(T)-t : t
            native = panel.reverse ? panel.right-(panel.right-panel.left)*ξ^panel.power : panel.left+(panel.right-panel.left)*ξ^panel.power
            samples[next] = offsets[panel.curve_index]+_weighted_corner_arc_length(curve,native)
            next += 1
        end
    end
    return samples, amplitudes
end


"""
    momentum_function(u::AbstractVector{<:Number}, pts::BoundaryPoints{T})
        -> Tuple{Vector{T},Vector{T}}

Project a physical boundary derivative onto `exp(-2πim*s/L)`, with
`m=0:floor(length(u)/2)`. `u` contains full-boundary nodal values and `pts`
contains their geometry. Periodic/graded points use their physical quadrature;
weighted corner points integrate the complete source cardinals through
independent Gaussian samples. This distinction is essential because the
Jacobi norm weights integrate `|u|²`, rather than the linear density.

No keywords are required. Return squared Fourier coefficients and their
angular wavenumbers. Inputs are unchanged; vectors are newly allocated.
"""
function momentum_function(u::AbstractVector{<:Number},pts::BoundaryPoints{T}) where {T<:Real}
    pts.density_space === nothing && return momentum_function(u,pts.s,pts.ds)
    samples, amplitudes = _weighted_corner_boundary_quadrature(pts,u)
    L = T(sum(curve.length for curve in pts.density_space.curves))
    frequencies = T.(2pi/L .* (0:div(length(u),2)))
    power = Vector{T}(undef,length(frequencies))
    @inbounds for m in eachindex(frequencies)
        coefficient = zero(Complex{T})
        for j in eachindex(samples)
            coefficient += amplitudes[j]*cis(-frequencies[m]*samples[j])
        end
        power[m] = abs2(coefficient/L)
    end
    if eltype(u) <: Real
        lastdouble = iseven(length(u)) ? length(power)-1 : length(power)
        lastdouble >= 2 && (power[2:lastdouble] .*= 2)
    end
    return power, frequencies
end

"""
    _weighted_corner_wavefunctions(bds::AbstractVector{<:BoundaryPoints{T}},
        densities::AbstractVector{<:AbstractVector{<:Number}}, ks::AbstractVector{T},
        points::AbstractVector{SVector{2,T}}, indices::AbstractVector{Int},
        nx::Int, ny::Int; inside_only::Bool=true, use_NaN_mask::Bool=false)
        -> Psi::Vector{Matrix{Complex{T}}}

Reconstruct a batch of physical normal densities using `ψ=S q` on the already
constructed Cartesian grid. `bds`, `densities`, and `ks` provide one complete
weighted boundary space, physical nodal `q`, and real wavenumber per state;
`points[indices]` are the retained grid targets. Independent Gaussian source
cardinals and adaptive near-target refinement are shared with matrix assembly.
`nx,ny` give the output grid dimensions. `inside_only` together with
`use_NaN_mask` fills excluded grid entries with `NaN`, otherwise with zero.
All input arrays are unchanged and the returned field matrices are owned.
"""
function _weighted_corner_wavefunctions(bds::AbstractVector{<:BoundaryPoints{T}},
    densities::AbstractVector{<:AbstractVector{<:Number}},ks::AbstractVector{T},
    points::AbstractVector{SVector{2,T}},indices::AbstractVector{Int},nx::Int,ny::Int;
    inside_only::Bool=true,use_NaN_mask::Bool=false) where {T<:Real}
    all(bd.density_space !== nothing for bd in bds) || throw(ArgumentError("reconstruct weighted and periodic states in separate batches"))
    fields = [fill(use_NaN_mask && inside_only ? Complex{T}(NaN) : zero(Complex{T}),nx,ny) for _ in bds]
    targets = points[indices]
    for j in eachindex(bds)
        fields[j][indices] = _weighted_corner_potential(bds[j],densities[j],ks[j],targets)
    end
    return fields
end
