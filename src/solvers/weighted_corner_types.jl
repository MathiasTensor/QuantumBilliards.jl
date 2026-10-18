################################################################################
# OWNED WEIGHTED-DENSITY METADATA
#
# BoundaryPoints keeps its existing fields and constructor conventions; its one
# optional density_space field owns these physical-cardinal interpolation data.
# The records are independent of the spectral parameter and contain no kernel
# values. Geometry, assembly, and state projections share this same ownership.
################################################################################

"""
    WeightedDensityPanel{T<:Real}

One polynomial amplitude panel on a retained physical curve span. `left:right`
is the original curve parameter span; `tleft:tright` is a subinterval of its
power coordinate `t`. For a left corner, `u=left+(right-left)t^power`; for a
right corner, `u=right-(right-left)(1-t)^power`. The physical density is
`q(t)=ξ^(power*(lambda-1))*v(t)`, with `ξ=t` or `1-t` and `lambda=π/α`.

`nodes`, `barycentric_weights`, and `density_factors` define the cardinal
functions of `v`, while `indices` always refer to physical nodal values of
`q`. `curve_index` selects the original curve in `WeightedDensitySpace.curves`.
These numeric records contain no sampled kernel or eigenvalue.
"""
struct WeightedDensityPanel{T<:Real}
    curve_index::Int
    left::T
    right::T
    tleft::T
    tright::T
    reverse::Bool
    lambda::T
    power::T
    source_order::Int
    indices::UnitRange{Int}
    nodes::Vector{T}
    barycentric_weights::Vector{T}
    density_factors::Vector{T}
end

"""
    WeightedSourceRule{T<:Real}

Owned source geometry and integrated physical-density cardinal matrices.
Rows of `basis` are source samples and columns are density unknowns on one
panel. They already include source quadrature weights and physical speed.
`logbasis` contains the additional logarithmic product correction, or is empty
for an ordinary rule. `parameters` are the panel's power-coordinate samples.
The source rule has no spectral dependence and is reused across contour nodes.
Target-owned near/self rules also store `distances` and `normal_projections`.
These retain cancellation-free same-chart chord geometry. Shared far rules
leave those two arrays empty because their geometry depends on the target.
"""
struct WeightedSourceRule{T<:Real}
    parameters::Vector{T}
    xy::Vector{SVector{2,T}}
    basis::Matrix{T}
    logbasis::Matrix{T}
    distances::Vector{T}
    normal_projections::Vector{T}
end

WeightedSourceRule{T}(parameters,xy,basis,logbasis) where {T<:Real} =
    WeightedSourceRule{T}(parameters,xy,basis,logbasis,T[],T[])

"""
    WeightedDensitySpace{T<:Real}

Geometry and interpolation ownership for physical normal-derivative densities.
`panels` and `curves` retain the original physical charts and their coordinate
maps. Curve dispatch occurs once per setup panel; hot kernel loops use only
concrete numeric arrays from `source_cache`. `node_panel` and `node_parameter`
identify the power coordinate of each physical density node.

`norm_weights` integrate the known endpoint power in `|q|² ds` using a Jacobi
rule. Their bare sum is not an accurate independent geometric perimeter
estimate. `source_order` is the ordinary minimum integration count; each corner panel captures its own
larger minimum. Cache keys are `(target_index,panel_index,source_order)`, where
zero target denotes an ordinary full-panel rule. The lock guards preparation;
threaded matrix assembly reads completed immutable rules.
"""
struct WeightedDensitySpace{T<:Real}
    panels::Vector{WeightedDensityPanel{T}}
    curves::Vector{BilliardGeometry.AbsCurve}
    node_panel::Vector{Int}
    node_parameter::Vector{T}
    norm_weights::Vector{T}
    source_order::Int
    source_cache::Dict{NTuple{3,Int},Vector{WeightedSourceRule{T}}}
    source_lock::ReentrantLock
end
