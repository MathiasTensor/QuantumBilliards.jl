################################################################################
# PHYSICAL CORNER DENSITIES: ORIGINAL GEOMETRY, REGULAR AMPLITUDE, LOCAL PANELS
#
# A Dirichlet eigenfunction in a wedge 0<θ<α starts with
# ψ(r,θ)=a r^λ sin(λθ)+..., λ=π/α, because the angular equation is
# f''+λ²f=0 and both wall values vanish. Its physical normal derivative has
# q(r)=r^(λ-1)v(r). A reentrant corner therefore has an integrable divergence.
# The coefficient a is determined by the global eigenstate; no such coefficient
# is guessed or inserted here. Curvature and Helmholtz terms produce higher
# powers, and logarithms can remain, so refinement checks are still required.
#
# On one retained native curve span [a,b], let h=b-a. At a left corner choose
# u=a+h t^p; at a right corner choose u=b-h(1-t)^p. Write ξ=t or 1-t.
# Then q=ξ^e v, e=p(λ-1), and q ds=h p |γ_u| ξ^(pλ-1)v dt.
# Reentrant p=2/λ makes this source measure linear in ξ. Convex p=ceil(3λ)/λ
# retains an integer source power and p>=3. The mushroom's 270-degree corner
# has λ=2/3,p=3,e=-1: q=t^-1 v and q ds=3h|γ_u|t v dt.
#
# The unknowns remain PHYSICAL nodal q_j. The source cardinal functions are
# b_j(t)=L_j(t) ξ(t)^e/ξ(t_j)^e, hence b_j(t_i)=δ_ij. This is a different
# interpolation space, not an invertible diagonal rescaling of a fixed matrix.
# Native spans, mapped coordinates and these cardinals must also be used during
# independent source refinement and interior-field reconstruction.
#
# The boundary norm has |q|²ds ∝ ξ^η |v|²dt, η=2pλ-p-1>-1. The cell touching
# the corner therefore uses a Gauss-Jacobi density rule for ξ^η. Its stored
# physical weights divide out that Jacobi factor; multiplying by |q_j|² then
# restores the correct integral. Other cells use Gauss-Legendre nodes. These
# positive compensated weights target the density norm; summing them alone is
# not an accurate geometric length or area audit.
# Source integration instead samples the physical charts with its own Gaussian
# rules, so integration can improve without adding density unknowns.
#
# Corners are detected on the complete ordered physical boundary using signed
# endpoint tangent turns: α=π-atan(cross(t_-,t_+),dot(t_-,t_+)). Keeping the sign
# distinguishes a 270-degree shoulder from a 90-degree convex join. A tangent
# cusp has no finite wedge angle and is rejected rather than silently weighted.
# Reflection and rotation folding uses these actual panel nodes, because the
# integer permutations of a uniform Kress grid do not apply to this mesh.
################################################################################

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

# Geometry-only policies. These are derived roundoff guards and bounded starting
# choices, not requested spectral accuracy. Node/source defaults belong to the
# public WeightedCornerGrading constructor below.
const WEIGHTED_CORNER_GEOMETRY_DEFAULTS = (;
    minimum_panels_per_curve=2,  # Keeps left/right endpoint spans separate; practical 2-4.
    reentrant_source_power=2,    # pλ=2 makes q ds linear; practical 1-3, 2 keeps 2<p<4.
    convex_minimum_power=3,      # p>=3 regularizes mapped derivatives; practical 2-4.
    integer_snap_factor=32,     # eps-scaled rounding before ceil; practical 8-128.
    endpoint_roundoff_factor=128, # eps times geometry scale for physical joins; practical 32-512.
    cardinal_roundoff_factor=4, # eps-scaled coincident-node test; practical 1-8.
    symmetry_roundoff_factor=512, # eps times geometry scale for node images; practical 128-2048.
    normal_roundoff_factor=4096, # eps-scaled unit-normal check after symmetry; practical 1024-16384.
)

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

"""
    _weighted_panel_geometry(curve::C, panel::WeightedDensityPanel{T}, t::T)
        → (xy::SVector{2,T}, tangent::SVector{2,T}, tangent_2::SVector{2,T})

Evaluate the original physical curve through the panel's captured power map.
With `u'=hpξ^(p-1)` and `u''=±hp(p-1)ξ^(p-2)`, return `γ(u)`, `γ_u*u'`,
and `γ_uu*(u')²+γ_u*u''`. The sign in `u''` is negative for a right corner;
orientation is preserved because `u'>0`. For ordinary panels `p=1` and the
second map derivative is exactly zero. Inputs are unchanged.
"""
@inline function _weighted_panel_geometry(curve::C, panel::WeightedDensityPanel{T}, t::T) where {T<:Real,C<:BilliardGeometry.AbsCurve}
    h = panel.right-panel.left
    ξ = panel.reverse ? one(T)-t : t
    p = panel.power
    u = t == zero(T) ? panel.left : t == one(T) ? panel.right : panel.reverse ? panel.right-h*ξ^p : panel.left+h*ξ^p
    d1 = h*p*ξ^(p-one(T))
    d2 = p == one(T) ? zero(T) : (panel.reverse ? -one(T) : one(T))*h*p*(p-one(T))*ξ^(p-T(2))
    v = SVector{2,T}(tangent(curve,u))
    a = SVector{2,T}(tangent_2(curve,u))
    return SVector{2,T}(BilliardGeometry.curve(curve,u)), v*d1, a*d1^2+v*d2
end

"""
    _weighted_cardinal!(values::AbstractVector{T}, panel::WeightedDensityPanel{T}, t::T)
        → values::AbstractVector{T}

Evaluate all physical-density cardinal functions on one amplitude panel:
`b_j(t)=L_j(t)*(ξ(t)/ξ(t_j))^(p*(λ-1))`. The barycentric denominator forms
`L_j` without solving a Vandermonde system. Coincident density nodes return an
exact unit vector, preserving physical nodal densities and the jump identity.
The caller owns `values`; all other inputs remain unchanged. Source rules use
strictly interior samples, so a divergent corner endpoint is never evaluated.
"""
@inline function _weighted_cardinal!(values::AbstractVector{T}, panel::WeightedDensityPanel{T}, t::T) where {T<:Real}
    length(values) == length(panel.nodes) || throw(DimensionMismatch("cardinal buffer does not match panel density nodes"))
    total = zero(T)
    tolerance = T(WEIGHTED_CORNER_GEOMETRY_DEFAULTS.cardinal_roundoff_factor)*eps(T)*max(one(T),abs(t))
    @inbounds for j in eachindex(values)
        δ = t-panel.nodes[j]
        if abs(δ) <= tolerance
            fill!(values,zero(T)); values[j] = one(T)
            return values
        end
        value = panel.barycentric_weights[j]/δ
        values[j] = value; total += value
    end
    ξ = panel.reverse ? one(T)-t : t
    factor = ξ^(panel.power*(panel.lambda-one(T)))
    @inbounds @simd for j in eachindex(values)
        values[j] *= factor/(total*panel.density_factors[j])
    end
    return values
end

"""
    _weighted_corner_arc_length(curve::AbsCurve, u::T) -> s::T

Return physical arc length from native parameter zero to `u`. Lines and circular
arcs have constant speed `|γ′|=curve.length`, so their integral is exactly
`s=u*curve.length` and needs no numerical quadrature. Other smooth charts keep
BilliardGeometry's adaptive integral with relative tolerance `64eps(T)`; no
geometric approximation or interpolation is introduced. The input is unchanged.
"""
@inline _weighted_corner_arc_length(curve::BilliardGeometry.AbsCurve,u::T) where {T<:Real} =
    T(BilliardGeometry.arc_length(curve,u;rtol=T(64)*eps(T)))
@inline _weighted_corner_arc_length(curve::Union{BilliardGeometry.AbsLine,BilliardGeometry.CircleSegment},u::T) where {T<:Real} =
    u*T(curve.length)

# A function barrier keeps heterogeneous original curve dispatch outside the
# node loop. Every retained panel and source cardinal remains numerically typed.
function _weighted_append_panel!(arrays, panels::Vector{WeightedDensityPanel{T}}, curve::C,
    curve_index::Int, left::T, right::T, tleft::T, tright::T, reverse::Bool,
    lambda::T, power::T, source_order::Int, count::Int, s_offset::T) where {T<:Real,C<:BilliardGeometry.AbsCurve}
    e = power*(lambda-one(T))
    η = T(2)*power*lambda-power-one(T)
    at_corner = e != zero(T) && (reverse ? tright == one(T) : tleft == zero(T))
    if at_corner
        η > -one(T) && isfinite(η) || throw(ArgumentError("corner angle gives a nonintegrable density norm"))
        x,w = FastGaussQuadrature.gaussjacobi(count,reverse ? η : zero(T),reverse ? zero(T) : η)
        ordinary_nodes = T.((x .+ 1)./2)
        quadrature_weights = T.(w./exp2(η+one(T)))
    else
        x,w = FastGaussQuadrature.gausslegendre(count)
        ordinary_nodes = T.((x .+ 1)./2)
        quadrature_weights = T.(w./2)
    end
    nodes = tleft .+(tright-tleft).*ordinary_nodes
    barycentric = Vector{T}(undef,count)
    factors = Vector{T}(undef,count)
    @inbounds for j in 1:count
        product = one(T)
        for l in 1:count
            l == j || (product *= ordinary_nodes[j]-ordinary_nodes[l])
        end
        barycentric[j] = inv(product)
        ξ = reverse ? one(T)-nodes[j] : nodes[j]
        factors[j] = ξ^e
        isfinite(factors[j]) && factors[j] > zero(T) || throw(ArgumentError("corner density factor is not representable; refine physical spans instead"))
    end
    barycentric ./= maximum(abs,barycentric)
    indices = (length(arrays.xy)+1):(length(arrays.xy)+count)
    panel = WeightedDensityPanel(curve_index,left,right,tleft,tright,reverse,lambda,power,source_order,indices,nodes,barycentric,factors)
    panel_index = length(panels)+1
    left_point = SVector{2,T}(BilliardGeometry.curve(curve,left))
    right_point = SVector{2,T}(BilliardGeometry.curve(curve,right))
    @inbounds for j in eachindex(nodes)
        t = nodes[j]
        point,v,a = _weighted_panel_geometry(curve,panel,t)
        speed = hypot(v[1],v[2])
        ξnorm = reverse ? one(T)-ordinary_nodes[j] : ordinary_nodes[j]
        weight = (tright-tleft)*quadrature_weights[j]*speed/(at_corner ? ξnorm^η : one(T))
        h = right-left; ξ = reverse ? one(T)-t : t
        u = reverse ? right-h*ξ^power : left+h*ξ^power
        all(isfinite,point) && all(isfinite,v) && isfinite(speed) && speed > zero(T) && isfinite(weight) && weight > zero(T) || throw(ArgumentError("weighted corner node or norm weight is unresolved"))
        point != left_point && point != right_point || throw(ArgumentError("weighted corner point collapses onto a physical endpoint"))
        isempty(arrays.xy) || point != last(arrays.xy) || throw(ArgumentError("weighted density nodes collapse in physical coordinates"))
        normal = SVector{2,T}(v[2]/speed,-v[1]/speed)
        push!(arrays.xy,point); push!(arrays.normal,normal)
        push!(arrays.tangent,v); push!(arrays.tangent_2,a)
        push!(arrays.ds,weight); push!(arrays.ws,weight/speed)
        push!(arrays.ws_der,h*power*ξ^(power-one(T)))
        push!(arrays.s,s_offset+_weighted_corner_arc_length(curve,u))
        push!(arrays.tphys,u); push!(arrays.ts,t)
        push!(arrays.node_panel,panel_index); push!(arrays.node_parameter,t)
    end
    push!(panels,panel)
    return nothing
end

"""
    _weighted_corner_points(solver::CombinedFieldIntegralEquationSolver,
        comp::Vector, k::T) → pts::BoundaryPoints{T}

Build the complete physical-boundary amplitude space selected by
`WeightedCornerGrading`. Physical endpoints and signed tangent turns determine
all corner angles. The wavelength budget is `max(min_pts,k*L*ppw/(2π))`;
complete panels and the corner safety floor can increase the actual count.
Density nodes exclude endpoints. Source counts do not enlarge this matrix.

The returned `density_space` owns the original curves, power-coordinate spans,
physical-density cardinal metadata, Jacobi norm weights and empty reusable
source caches. `pts.ds` uses the same physical density-norm quadrature; use the
original curves for geometric perimeter/area integration. This constructor
prepares a starting space and does not certify an eigenstate's convergence.
Disconnected components and nonregular cusps require their own construction
and are rejected rather than joined by an artificial segment.
"""
function _weighted_corner_points(solver, comp::Vector, k::T)::BoundaryPoints{T} where {T<:Real}
    T === Float64 || throw(ArgumentError("WeightedCornerGrading currently supports Float64 boundary geometry; legacy Kress policies retain their existing numeric types"))
    grading = solver.grading::WeightedCornerGrading
    isempty(comp) && throw(ArgumentError("weighted density boundary cannot be empty"))
    isfinite(k) && k > zero(T) || throw(ArgumentError("weighted density resolution needs a positive finite wavenumber"))
    lengths = T[c.length for c in comp]
    all(x -> isfinite(x) && x > zero(T),lengths) || throw(ArgumentError("every physical curve needs a positive finite length"))
    perimeter = sum(lengths)
    curves = BilliardGeometry.AbsCurve[c for c in comp]
    angles = fill(T(pi),length(comp))
    policy = WEIGHTED_CORNER_GEOMETRY_DEFAULTS
    scale = max(one(T),maximum(norm(SVector{2,T}(BilliardGeometry.curve(c,zero(T)))) for c in comp))
    endpoint_tolerance = T(policy.endpoint_roundoff_factor)*eps(T)*scale
    for j in eachindex(comp)
        previous = comp[mod1(j-1,length(comp))]; current = comp[j]
        norm(SVector{2,T}(BilliardGeometry.curve(previous,one(T)))-SVector{2,T}(BilliardGeometry.curve(current,zero(T)))) <= endpoint_tolerance || throw(ArgumentError("weighted panels require one ordered closed physical component"))
        left = SVector{2,T}(tangent(previous,one(T)))
        right = SVector{2,T}(tangent(current,zero(T)))
        left_speed = norm(left); right_speed = norm(right)
        left_mid = norm(SVector{2,T}(tangent(previous,T(0.5))))
        right_mid = norm(SVector{2,T}(tangent(current,T(0.5))))
        left_speed > T(grading.minimum_tangent_ratio)*left_mid > zero(T) && right_speed > T(grading.minimum_tangent_ratio)*right_mid > zero(T) || throw(ArgumentError("a cusp or zero endpoint tangent has no regular wedge angle"))
        left /= left_speed; right /= right_speed
        turn = atan(left[1]*right[2]-left[2]*right[1],clamp(dot(left,right),-one(T),one(T)))
        if abs(turn) > T(grading.corner_angle_tolerance)
            angle = T(pi)-turn
            zero(T) < angle < T(2pi) || throw(ArgumentError("corner angle must lie strictly between 0 and 2π"))
            angles[j] = angle
        end
    end
    panels = WeightedDensityPanel{T}[]
    arrays = (;xy=SVector{2,T}[],normal=SVector{2,T}[],tangent=SVector{2,T}[],tangent_2=SVector{2,T}[],
        ds=T[],ws=T[],ws_der=T[],s=T[],ts=T[],tphys=T[],node_panel=Int[],node_parameter=T[])
    requested = max(T(solver.min_pts),k*perimeter*T(solver.pts_scaling_factor[1])/T(2pi))
    ordinary_order = grading.regular_density_nodes
    corner_order = grading.corner_density_nodes
    # Even node counts avoid fixed points on a reflection axis; no density node
    # is dropped or forced to zero when a sector is subsequently selected.
    if solver.symmetry !== nothing
        ordinary_order = cld(ordinary_order,2)*2
        corner_order = cld(corner_order,2)*2
    end
    s_offset = zero(T)
    for index in eachindex(comp)
        left_angle = angles[index]; right_angle = angles[mod1(index+1,length(comp))]
        left_corner = left_angle != T(pi); right_corner = right_angle != T(pi)
        ends = Int(left_corner)+Int(right_corner)
        local_budget = requested*lengths[index]/perimeter
        middle_budget = (local_budget-T(ends*corner_order*grading.corner_subpanels))/T(ordinary_order)
        nearest = round(middle_budget)
        abs(middle_budget-nearest) <= T(policy.integer_snap_factor)*eps(T)*max(one(T),abs(middle_budget)) && (middle_budget=nearest)
        count = max(policy.minimum_panels_per_curve,ends+ceil(Int,middle_budget))
        for cell in 1:count
            left = T(cell-1)/T(count); right = T(cell)/T(count)
            corner = cell == 1 && left_corner || cell == count && right_corner
            reverse = cell == count && right_corner
            lambda = corner ? T(pi)/(reverse ? right_angle : left_angle) : one(T)
            if corner
                if lambda < one(T)
                    power = T(policy.reentrant_source_power)/lambda
                else
                    degree = T(policy.convex_minimum_power)*lambda
                    nearest = round(degree)
                    abs(degree-nearest) <= T(policy.integer_snap_factor)*eps(T)*max(one(T),abs(degree)) && (degree=nearest)
                    power = ceil(degree)/lambda
                end
                for sub in 1:grading.corner_subpanels
                    _weighted_append_panel!(arrays,panels,comp[index],index,left,right,
                        T(sub-1)/T(grading.corner_subpanels),T(sub)/T(grading.corner_subpanels),reverse,
                        lambda,power,grading.corner_integration_nodes,corner_order,s_offset)
                end
            else
                _weighted_append_panel!(arrays,panels,comp[index],index,left,right,zero(T),one(T),false,
                    one(T),one(T),grading.regular_integration_nodes,ordinary_order,s_offset)
            end
        end
        s_offset += lengths[index]
    end
    space = WeightedDensitySpace(panels,curves,arrays.node_panel,arrays.node_parameter,copy(arrays.ds),
        grading.regular_integration_nodes,Dict{NTuple{3,Int},Vector{WeightedSourceRule{T}}}(),ReentrantLock())
    return BoundaryPoints(arrays.xy;normal=arrays.normal,s=arrays.s,ds=arrays.ds,
        tangent=arrays.tangent,tangent_2=arrays.tangent_2,ts=arrays.ts,tphys=arrays.tphys,
        ws=arrays.ws,ws_der=arrays.ws_der,density_space=space)
end

# Match one geometric action once during setup. Quantized buckets make this
# O(N) for well-separated nodes, while neighboring buckets avoid rounding-cell
# boundary errors. Every accepted image is checked in physical coordinates and
# against the transformed physical unit normal before any matrix is folded.
function _weighted_node_permutation(pts::BoundaryPoints{T}, action::S) where {T<:Real,S<:BilliardGeometry.AbsSymmetry}
    N = length(pts)
    scale = max(one(T),maximum(norm,pts.xy))
    tolerance = T(WEIGHTED_CORNER_GEOMETRY_DEFAULTS.symmetry_roundoff_factor)*eps(T)*scale
    normal_tolerance = T(WEIGHTED_CORNER_GEOMETRY_DEFAULTS.normal_roundoff_factor)*eps(T)
    buckets = Dict{Tuple{Int,Int},Vector{Int}}()
    for j in eachindex(pts.xy)
        point = pts.xy[j]
        key = (round(Int,point[1]/tolerance),round(Int,point[2]/tolerance))
        push!(get!(buckets,key,Int[]),j)
    end
    permutation = Vector{Int}(undef,N)
    used = falses(N)
    for j in eachindex(pts.xy)
        image = SVector{2,T}(BilliardGeometry.apply_symmetry(action,pts.xy[j]))
        key = (round(Int,image[1]/tolerance),round(Int,image[2]/tolerance))
        selected = 0; distance = tolerance
        for dx in -1:1, dy in -1:1
            candidates = get(buckets,(key[1]+dx,key[2]+dy),nothing)
            candidates === nothing && continue
            for candidate in candidates
                error = norm(pts.xy[candidate]-image)
                if error <= distance
                    distance = error; selected = candidate
                end
            end
        end
        selected != 0 || throw(ArgumentError("weighted panel mesh is not invariant under $(typeof(action)); no physical image of node $j"))
        !used[selected] || throw(ArgumentError("weighted symmetry node matching is not bijective; physical nodes are too close"))
        expected_normal = SVector{2,T}(BilliardGeometry.apply_symmetry(action,pts.normal[j]))
        norm(pts.normal[selected]-expected_normal) <= normal_tolerance || throw(ArgumentError("weighted symmetry image has an inconsistent physical normal"))
        permutation[j] = selected; used[selected] = true
    end
    return permutation
end

"""
    _fold_boundary(::Type{T}, billiard::AbsBilliard, pts::BoundaryPoints{T},
        symmetry::AbsSymmetry, characters::Tuple) → orbits::SymmetryOrbitMap{T}

Construct the symmetry action on the actual Gaussian density nodes. Legacy
points delegate to the existing integer Kress-grid construction. Weighted
points instead match geometric images using `BilliardGeometry.apply_symmetry`,
check coordinate/normal consistency and bijectivity, then close the generator
permutations and their characters before calling the shared orbit-map builder.

Single reflections use one parity. `XYAxisReflection` follows the existing
solver convention of two independent x/y parities. `CompositeReflection`
closes its reflection generators; an `XYAxisReflection` inside that set applies
its supplied parity to both axes, matching BilliardGeometry. `NFoldRotation`
uses one integer cyclic sector, default zero. Fixed-node or inconsistent
character actions are rejected rather than assigned incorrect source weights.
"""
function _fold_boundary(::Type{T}, billiard::Bi, pts::BoundaryPoints{T}, symmetry::S, characters::Tuple) where {T<:Real,Bi<:BilliardGeometry.AbsBilliard,S<:BilliardGeometry.AbsSymmetry}
    pts.density_space === nothing && return _fold_boundary(T,billiard,length(pts),symmetry,characters)
    actions = BilliardGeometry.AbsSymmetry[]
    generator_scales = Complex{T}[]
    if symmetry isa NFoldRotation
        length(characters) <= 1 || throw(ArgumentError("NFoldRotation requires one integer sector"))
        sector = isempty(characters) ? 0 : only(characters)
        sector isa Int || throw(ArgumentError("NFoldRotation sector must be an Int"))
        0 <= sector < symmetry.order || throw(ArgumentError("rotation sector is outside 0:$(symmetry.order-1)"))
        push!(actions,NFoldRotation(symmetry.order,1;T=T))
        push!(generator_scales,cis(T(2pi)*T(sector)/T(symmetry.order)))
    elseif symmetry isa XYAxisReflection
        length(characters) <= 2 || throw(ArgumentError("XYAxisReflection accepts x and y parities"))
        append!(actions,(XAxisReflection(),YAxisReflection()))
        push!(generator_scales,isempty(characters) ? one(Complex{T}) : Complex{T}(characters[1]))
        push!(generator_scales,length(characters) < 2 ? one(Complex{T}) : Complex{T}(characters[2]))
    elseif symmetry isa CompositeReflection
        isempty(symmetry.reflections) && throw(ArgumentError("CompositeReflection needs at least one generator"))
        isempty(characters) || length(characters) == length(symmetry.reflections) || throw(DimensionMismatch("one character per composite reflection is required"))
        for (j,reflection) in enumerate(symmetry.reflections)
            χ = isempty(characters) ? one(Complex{T}) : Complex{T}(characters[j])
            if reflection isa XYAxisReflection
                append!(actions,(XAxisReflection(),YAxisReflection()))
                append!(generator_scales,(χ,χ))
            elseif reflection isa Union{XAxisReflection,YAxisReflection,DiagonalReflection,AntiDiagonalReflection}
                push!(actions,reflection); push!(generator_scales,χ)
            else
                throw(ArgumentError("unsupported composite reflection $(typeof(reflection))"))
            end
        end
    elseif symmetry isa Union{XAxisReflection,YAxisReflection,DiagonalReflection,AntiDiagonalReflection}
        length(characters) <= 1 || throw(ArgumentError("one reflection accepts one parity"))
        push!(actions,symmetry)
        push!(generator_scales,isempty(characters) ? one(Complex{T}) : Complex{T}(only(characters)))
    else
        throw(ArgumentError("unsupported weighted density symmetry $(typeof(symmetry))"))
    end
    phase_tolerance = T(WEIGHTED_CORNER_GEOMETRY_DEFAULTS.normal_roundoff_factor)*eps(T)
    if !(symmetry isa NFoldRotation)
        all(χ -> abs(χ^2-one(Complex{T})) <= phase_tolerance,generator_scales) || throw(ArgumentError("reflection characters must be ±1"))
    end
    generator_permutations = [_weighted_node_permutation(pts,action) for action in actions]
    permutations = Vector{Int}[collect(1:length(pts))]
    scales = Complex{T}[one(Complex{T})]
    head = 1
    while head <= length(permutations)
        for j in eachindex(generator_permutations)
            composed = BilliardGeometry._compose_index_permutations(generator_permutations[j],permutations[head])
            χ = generator_scales[j]*scales[head]
            previous = findfirst(==(composed),permutations)
            if previous === nothing
                push!(permutations,composed); push!(scales,χ)
                length(permutations) <= length(pts) || throw(ArgumentError("symmetry action does not have free boundary-node orbits"))
            else
                abs(scales[previous]-χ) <= phase_tolerance || throw(ArgumentError("symmetry characters are inconsistent under group composition"))
            end
        end
        head += 1
    end
    return BilliardGeometry._build_symmetry_orbit_map(T,length(pts),permutations,scales)
end
