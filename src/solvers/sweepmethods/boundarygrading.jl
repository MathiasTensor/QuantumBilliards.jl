# WEIGHTED-CORNER STARTING RULES
# These are initial resolution choices, not acceptance tolerances. Density and
# source integration are checked independently; increasing source counts does
# not add unknowns. Explicit constructor keywords allow convergence studies.
const WEIGHTED_CORNER_DEFAULTS = (;
    regular_density_nodes=12,       # Degree 11 on smooth spans; practical 8-24, supported 4-32.
    corner_density_nodes=16,        # Degree 15 for the regular corner amplitude; practical 12-24, supported 4-32.
    corner_subpanels=2,             # Two computational cells resolve remaining power/log terms; practical/supported 1-4.
    regular_integration_nodes=32,   # Independent kernel sources, not density unknowns; practical 24-64, supported >= density nodes.
    corner_integration_nodes=96,    # Resolves self logarithms after the power map; practical 64-160, supported >= corner density nodes.
    corner_angle_tolerance=1e-8,    # Radians: roundoff-sized tangent joins count as smooth; practical 1e-10-1e-6, supported (0,0.1).
    minimum_tangent_ratio=1e-12,    # Endpoint speed relative to midpoint speed; detects cusps, practical 1e-14-1e-8, supported (0,1).
)

"""
BoundaryGrading

`BoundaryGrading` is the abstract supertype for the boundary discretization
strategies used by [`SweepBIMSolver`](@ref) concrete types
([`DoubleLayerPotentialSolver`](@ref), [`CombinedFieldIntegralEquationSolver`](@ref)).

## Description
A `BoundaryGrading` selects how `evaluate_points` samples, and optionally
reparametrizes, each boundary curve before the Nyström/Kress-corrected
Fredholm matrix is assembled. Concrete subtypes are
[`SmoothPeriodicGrading`](@ref) (a single smooth closed curve, or a smooth
composite boundary with no true corners, using ungraded periodic
quadrature), [`CornerGrading`](@ref) (a single curve parametrization with
known parametric corner singularities, Kress-graded), and
[`GlobalCornerGrading`](@ref) (a composite/piecewise-smooth boundary,
globally Kress-graded relative to all detected corners), and
[`WeightedCornerGrading`](@ref) (local Gaussian physical-density panels, with
angle-adapted endpoint interpolation).

## API
Concrete subtypes of `BoundaryGrading` are dispatched on by `evaluate_points`
for the [`SweepBIMSolver`](@ref) they are attached to.
"""
abstract type BoundaryGrading end

"""
    SmoothPeriodicGrading <: BoundaryGrading

Ungraded periodic discretization for a single smooth closed boundary curve, or
a smooth composite boundary with no true corners. The periodic Kress
logarithmic singularity correction is applied without any parameter
reclustering. This is the default, and lowest-overhead, grading strategy.
"""
struct SmoothPeriodicGrading <: BoundaryGrading end

"""
    CornerGrading{T<:Real} <: BoundaryGrading

Kress-graded discretization for a single closed curve parametrized on
`[0, 1)` whose corner(s) are known parametric singularities (e.g. a polygon
edge parametrization with a corner at `t = 0`).

## Attributes
* `kressq::Int`: Order of the Kress grading transformation `t = w(σ)`.
* `min_t_spacing::T`: Minimum permitted physical-parameter spacing after grading, guarding against machine-precision node collisions for large `kressq`.
"""
struct CornerGrading{T<:Real} <: BoundaryGrading
    kressq::Int
    min_t_spacing::T
end

"""
    CornerGrading(; kressq::Int = 2, min_t_spacing::Real = 1e-12) → grading::CornerGrading

Constructs a [`CornerGrading`](@ref) with default Kress grading order `2`.

## Keyword arguments
* `kressq::Int = 2`: Order of the Kress grading transformation.
* `min_t_spacing::Real = 1e-12`: Minimum permitted physical-parameter spacing after grading.

## Returns
* `grading`: A [`CornerGrading`](@ref) instance.
"""
CornerGrading(; kressq::Int=2, min_t_spacing::Real=1e-12) = CornerGrading(kressq, min_t_spacing)

"""
    GlobalCornerGrading{T<:Real} <: BoundaryGrading

Kress-graded discretization for a composite/piecewise-smooth closed boundary
built from several joined curve segments. True corners are detected from
tangent discontinuities between adjacent segments and a single global grading
map is built relative to all of them simultaneously; if no corners are
detected, the boundary is discretized as an ungraded uniform periodic mesh
(equivalent to [`SmoothPeriodicGrading`](@ref)).

## Attributes
* `kressq::Int`: Order of the global Kress grading transformation.
* `min_t_spacing::T`: Minimum permitted physical-parameter spacing after grading.
"""
struct GlobalCornerGrading{T<:Real} <: BoundaryGrading
    kressq::Int
    min_t_spacing::T
end

"""
    GlobalCornerGrading(; kressq::Int = 2, min_t_spacing::Real = 1e-12) → grading::GlobalCornerGrading

Constructs a [`GlobalCornerGrading`](@ref) with default Kress grading order `2`.

## Keyword arguments
* `kressq::Int = 2`: Order of the global Kress grading transformation.
* `min_t_spacing::Real = 1e-12`: Minimum permitted physical-parameter spacing after grading.

## Returns
* `grading`: A [`GlobalCornerGrading`](@ref) instance.
"""
GlobalCornerGrading(; kressq::Int=2, min_t_spacing::Real=1e-12) = GlobalCornerGrading(kressq, min_t_spacing)

"""
    _bim_numeric_type(solver::SweepBIMSolver) → Type{<:Real}

Internal helper returning the real scalar type `T` used by a
[`SweepBIMSolver`](@ref) kernel, used by [`BeynSolver`](@ref) and
[`ExpandedBIMSolver`](@ref) to infer their own numeric type from the wrapped
kernel. Every concrete `SweepBIMSolver` is expected to add a method.
"""
function _bim_numeric_type end

"""
    WeightedCornerGrading{T<:Real} <: BoundaryGrading

Local Gaussian panels for the physical boundary normal derivative
`q=∂ₙψ`, with automatic angle-dependent corner coordinates. This opt-in policy
selects the adjoint CFIE density formulation. Existing Kress policies retain
their original combined-layer density and discretization.

In a Dirichlet wedge with interior angle `α`, separation of variables gives
`ψ∼r^(π/α) sin(πθ/α)` and therefore `q∼r^(λ-1)`, `λ=π/α`. On an endpoint
span use `r∝ξ^p` and interpolate the bounded amplitude `v=q/ξ^(p*(λ-1))`.
For a reentrant corner `p=2/λ`, so `q ds∝ξ*v dξ`; for a convex corner
`p=ceil(3λ)/λ`, which keeps the mapped geometry regular. These leading-power
rules do not remove all higher powers or logarithmic terms.

`regular_density_nodes` and `corner_density_nodes` count unknowns per panel.
Integration nodes are independent kernel samples and do not enlarge the
matrix. The complete original boundary is retained, including smooth joins;
transparent interfaces and symmetry walls are excluded by `full_boundary`.

## Attributes
* `regular_density_nodes::Int`: Nodal polynomial order on ordinary spans.
* `corner_density_nodes::Int`: Nodal amplitude order on mapped endpoint cells.
* `corner_subpanels::Int`: Computational cells per physical endpoint span.
* `regular_integration_nodes::Int`: Minimum ordinary Gaussian source count.
* `corner_integration_nodes::Int`: Minimum source count retained by corner cells.
* `corner_angle_tolerance::T`: Radian tolerance for a smooth tangent join.
* `minimum_tangent_ratio::T`: Smallest endpoint/midpoint tangent-speed ratio.
"""
struct WeightedCornerGrading{T<:Real} <: BoundaryGrading
    regular_density_nodes::Int
    corner_density_nodes::Int
    corner_subpanels::Int
    regular_integration_nodes::Int
    corner_integration_nodes::Int
    corner_angle_tolerance::T
    minimum_tangent_ratio::T
end

"""
    WeightedCornerGrading(; regular_density_nodes::Int=12,
        corner_density_nodes::Int=16, corner_subpanels::Int=2,
        regular_integration_nodes::Int=32, corner_integration_nodes::Int=96,
        corner_angle_tolerance::Real=1e-8, minimum_tangent_ratio::Real=1e-12)
        → grading::WeightedCornerGrading

Construct the automatic weighted-corner starting space. Defaults and realistic
study ranges are declared in `WEIGHTED_CORNER_DEFAULTS` at the top of this
file. No angle, wedge exponent, or coordinate power is guessed by the caller.

## Keyword arguments
* `regular_density_nodes::Int=12`: Ordinary panel unknown count, in `4:32`.
* `corner_density_nodes::Int=16`: Corner amplitude unknown count, in `4:32`.
* `corner_subpanels::Int=2`: Endpoint computational subdivision, in `1:4`.
* `regular_integration_nodes::Int=32`: Ordinary source count, at least the ordinary density count.
* `corner_integration_nodes::Int=96`: Corner source count, at least the corner density count.
* `corner_angle_tolerance::Real=1e-8`: Smooth-join tolerance in radians, in `(0,0.1)`.
* `minimum_tangent_ratio::Real=1e-12`: Scale-independent cusp screen, in `(0,1)`.

## Returns
* `grading::WeightedCornerGrading`: Immutable geometry and starting-resolution policy.
"""
function WeightedCornerGrading(;
    regular_density_nodes::Int=WEIGHTED_CORNER_DEFAULTS.regular_density_nodes,
    corner_density_nodes::Int=WEIGHTED_CORNER_DEFAULTS.corner_density_nodes,
    corner_subpanels::Int=WEIGHTED_CORNER_DEFAULTS.corner_subpanels,
    regular_integration_nodes::Int=WEIGHTED_CORNER_DEFAULTS.regular_integration_nodes,
    corner_integration_nodes::Int=WEIGHTED_CORNER_DEFAULTS.corner_integration_nodes,
    corner_angle_tolerance::Real=WEIGHTED_CORNER_DEFAULTS.corner_angle_tolerance,
    minimum_tangent_ratio::Real=WEIGHTED_CORNER_DEFAULTS.minimum_tangent_ratio)
    4 <= regular_density_nodes <= 32 || throw(ArgumentError("regular_density_nodes must be in 4:32"))
    4 <= corner_density_nodes <= 32 || throw(ArgumentError("corner_density_nodes must be in 4:32"))
    1 <= corner_subpanels <= 4 || throw(ArgumentError("corner_subpanels must be in 1:4"))
    regular_integration_nodes >= regular_density_nodes || throw(ArgumentError("ordinary integration must have at least regular_density_nodes samples"))
    corner_integration_nodes >= corner_density_nodes || throw(ArgumentError("corner integration must have at least corner_density_nodes samples"))
    isfinite(corner_angle_tolerance) && 0 < corner_angle_tolerance < 0.1 || throw(ArgumentError("corner_angle_tolerance must be finite and in (0,0.1)"))
    isfinite(minimum_tangent_ratio) && 0 < minimum_tangent_ratio < 1 || throw(ArgumentError("minimum_tangent_ratio must be finite and in (0,1)"))
    T = promote_type(typeof(float(corner_angle_tolerance)), typeof(float(minimum_tangent_ratio)))
    return WeightedCornerGrading{T}(regular_density_nodes, corner_density_nodes,
        corner_subpanels, regular_integration_nodes, corner_integration_nodes,
        T(corner_angle_tolerance), T(minimum_tangent_ratio))
end
