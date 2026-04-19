# Useful reading:
#  - https://github.com/ahbarnett/mpspack - by Alex Barnett & Timo Betcke (MATLAB)
#  - Kress, R., Boundary integral equations in time-harmonic acoustic scattering. Mathematics Comput. Modelling Vol 15, pp. 229-243). Pergamon Press, 1991, GB.
#  - Barnett, A. H., & Betcke, T. (2007). Stability and convergence of the method of fundamental solutions for Helmholtz problems on analytic domains. Journal of Computational Physics, 227(14), 7003-7026.
#  - Zhao, L., & Barnett, A. (2015). Robust and efficient solution of the drum problem via Nyström approximation of the Fredholm determinant. SIAM Journal on Numerical Analysis, Stable URL: https://www.jstor.org/stable/24512689

two_pi=2*pi
inv_two_pi=1/two_pi
euler_over_pi=MathConstants.eulergamma/pi

################################
#### CONSTRUCTOR CFIE_kress ####
################################

"""
    CFIE_kress{T,Bi,Sym} <: CFIE

Combined-field integral equation (CFIE) solver using Kress's periodic
logarithmic splitting on smooth closed boundary components.

This is the smooth-boundary Kress variant of the CFIE Fredholm formulation.
It is designed for billiards whose boundary components are individually smooth,
closed, and periodic. The method uses Kress's idea of splitting the singular
self-interaction kernels into:

    universal logarithmic part + smooth remainder,

and then treating the logarithmic part analytically through a special dense
correction matrix `R`.

Mathematically, for one smooth component, the discretized operator has the form

    A = I - ( D + i k S ),

where:
- `D` is the double-layer operator,
- `S` is the single-layer operator,
- `i k S` is added to stabilize the Fredholm formulation and remove the
  spurious-nullspace behavior of the pure double-layer formulation
  for domains with holes.

In the Kress method, both `D` and `S` are split into:
- a coefficient multiplying the universal periodic logarithm,
- a smooth remainder evaluated by ordinary quadrature.

This type is intended for:
- circles, ellipses, limacons, and other smooth closed curves,
- multi-component smooth boundaries (e.g. smooth holes), where each component is
  itself a single smooth closed curve,
- high-accuracy determinant or singular-value based spectral computation.

# Fields
- `sampler::Vector{LinearNodes}`:
  Placeholder field kept for consistency with the common solver interface.
  In the periodic Kress method, the actual nodes are generated explicitly as
  equispaced periodic Kress nodes, so this field is not actively used.
- `pts_scaling_factor::Vector{T}`:
  Resolution parameter controlling the number of discretization points per
  wavelength. For a component of length `L`, the node count is chosen roughly as

      N ≈ k * L * b / (2π),

  where `b = pts_scaling_factor[1]` (or the component-specific entry if the
  logic is extended).
- `dim_scaling_factor::T`:
  Compatibility field for the common solver interface. No interior basis is used
  here, so this field is not mathematically active.
- `eps::T`:
  Compatibility / tolerance placeholder.
- `min_dim::Int64`:
  Compatibility field, unused mathematically in CFIE-Kress.
- `min_pts::Int64`:
  Minimum number of discretization points on each boundary component.
- `billiard::Bi`:
  The billiard geometry.
- `symmetry::Sym`:
  Optional symmetry descriptor, used by higher-level symmetry-aware logic.

# When to use this type
Use `CFIE_kress` when:
- every boundary component is smooth and closed,
- you want Kress-corrected singular quadrature for both SLP and DLP,
- you want higher accuracy than the plain BIM on smooth domains.

# When not to use it
Do not use this type if a boundary component is piecewise smooth with corners.
For that, use:
- `CFIE_kress_corners` for single-corner-type grading per closed curve,
- `CFIE_kress_global_corners` for global multi-corner grading on composite
  closed boundaries.
"""
struct CFIE_kress{T<:Real,Bi<:AbsBilliard,Sym}<:CFIE 
    sampler::Vector{LinearNodes} # placeholder since the trapezoidal rule can be changed. Not used currently.
    pts_scaling_factor::Vector{T} # scaling factor for the number of points per wavelength, which is used to determine the number of discretization points on the boundary based on the wavenumber and the length of the boundary. It can be a single value or a vector of values for different components of the boundary.
    dim_scaling_factor::T # UNUSED since no basis. Only for compatibility
    eps::T # UNUSED, for compatibility
    min_dim::Int64 # UNUSED, for compatibility
    min_pts::Int64 # minimum number of discretization points on the boundary, which ensures that even for low wavenumbers or short boundaries, we have enough points to accurately represent the geometry and solve the integral equation.
    billiard::Bi # the billiard domain for which we are solving the CFIE. It contains the geometry of the problem, including the boundary curves and their properties, which are essential for constructing the system matrix and solving the eigenvalue problem.
    symmetry::Sym # symmetry information for the billiard, which can be used to reduce the computational cost by exploiting symmetries in the geometry.
end

"""
    CFIE_kress_corners{T,Bi,Sym} <: CFIE

CFIE solver using Kress corner grading on a single smooth closed curve whose
parameterization contains endpoint singular behavior associated with corners.

This type is the corner-aware counterpart of `CFIE_kress`, but still assumes
that each boundary component is represented as a single closed parameterized
curve. The idea is that the underlying geometric curve may have corner-type
singular behavior in its derivatives, and Kress's grading transformation is used
to cluster nodes near the problematic locations.

Compared with the smooth periodic Kress method, the computational parameter is
no longer the physical periodic variable itself. Instead, one introduces a
graded variable `σ` and a nonlinear map `t = w(σ)` so that the geometric
singularity is regularized in the new variable.

The resulting discrete CFIE still has the formal structure

    A = I - ( D + i k S ),

but the nodes, weights, and derivative data now incorporate the grading map and
its Jacobian.

# Fields
- `sampler::Vector{LinearNodes}`:
  Placeholder field retained for common API compatibility.
- `pts_scaling_factor::Vector{T}`:
  Controls how the total number of nodes grows with the wavenumber.
- `dim_scaling_factor::T`:
  Compatibility field, unused mathematically here.
- `eps::T`:
  Compatibility / tolerance placeholder.
- `min_dim::Int64`:
  Compatibility field.
- `min_pts::Int64`:
  Minimum number of discretization points.
- `billiard::Bi`:
  Billiard geometry.
- `symmetry::Sym`:
  Optional symmetry descriptor.
- `kressq::Int`:
  Kress grading strength. Larger values cluster more points near corners/endpoints.

# Important practical note
For Float64 arithmetic, aggressive grading can become numerically unstable if
`kressq` is pushed too far. In this implementation, values around 4 are the safe
default and are generally recommended.

# When to use this type
Use `CFIE_kress_corners` when:
- the geometry is represented by a single closed curve,
- that curve has corner-type singular behavior,
- you want a Kress-graded discretization without switching to a global
  multi-segment composite-boundary treatment.
"""
struct CFIE_kress_corners{T<:Real,Bi<:AbsBilliard,Sym} <: CFIE
    sampler::Vector{LinearNodes} # placeholder since the trapezoidal rule will be changed by v(s,q) in kress_graging.jl
    pts_scaling_factor::Vector{T} 
    dim_scaling_factor::T # UNUSED since no basis. Only for compatibility
    eps::T # UNUSED, for compatibility
    min_dim::Int64 # UNUSED, for compatibility
    min_pts::Int64
    billiard::Bi 
    symmetry::Sym
    kressq::Int # the grading parameter q in the Kress grading formula, which controls how strongly the nodes are clustered near the corners. A larger value of q results in stronger clustering, which can improve accuracy for problems with sharp corners. Typical values are in the range of 4 to 16, with 8 being a common choice for many problems.
end

"""
    CFIE_kress_global_corners{T,Bi,Sym} <: CFIE

CFIE solver using a global multi-corner Kress grading on a closed composite
boundary component.

This is the most general Kress-based CFIE type in this family. It is designed
for piecewise smooth boundaries represented as several joined curve segments
forming one closed component. Typical examples are:
- rectangles,
- polygons,
- stadium-like composite boundaries,
- general piecewise smooth billiards with finitely many corners.

Unlike `CFIE_kress_corners`, which applies a grading transformation to a single
closed parameterized curve, this type treats the entire closed composite
boundary globally:
- all corner locations are detected,
- one global graded periodic variable is introduced,
- geometry is evaluated on the full composite boundary through a segment-aware
  global map.

The resulting operator is again the CFIE Fredholm matrix

    A = I - ( D + i k S ),

but now assembled on a globally graded periodic mesh.

# Fields
- `sampler::Vector{LinearNodes}`:
  Placeholder field for API consistency.
- `pts_scaling_factor::Vector{T}`:
  Node-density scaling factor(s).
- `dim_scaling_factor::T`:
  Compatibility field.
- `eps::T`:
  Compatibility / tolerance placeholder.
- `min_dim::Int64`:
  Compatibility field.
- `min_pts::Int64`:
  Minimum number of discretization points.
- `billiard::Bi`:
  Billiard geometry.
- `symmetry::Sym`:
  Optional symmetry descriptor.
- `kressq::Int`:
  Global Kress grading strength parameter.

# Why this type exists
For composite boundaries, the Kress logarithmic splitting only works correctly
if one treats each closed component as a single periodic object. Segment-by-
segment treatment would break the periodic singular structure. This type
therefore:
- reconstructs a single global periodic variable for the whole component,
- applies Kress grading relative to all corner locations at once,
- preserves the periodic framework needed for the `R`-matrix logic.

# When to use this type
Use `CFIE_kress_global_corners` when:
- a boundary component consists of multiple segments,
- the closed component contains corners,
- you want a Kress-corrected CFIE rather than Alpert panel quadrature.
"""
struct CFIE_kress_global_corners{T<:Real,Bi<:AbsBilliard,Sym} <: CFIE
    sampler::Vector{LinearNodes}
    pts_scaling_factor::Vector{T}
    dim_scaling_factor::T
    eps::T
    min_dim::Int64
    min_pts::Int64
    billiard::Bi
    symmetry::Sym
    kressq::Int
end

"""
    CFIE_kress(pts_scaling_factor::Union{T,Vector{T}},billiard::Bi;min_pts=20,eps=T(1e-15),symmetry::Sym=nothing)

Constructor for CFIE_kress solver.

# Inputs:
- `pts_scaling_factor`: A scaling factor for the number of points per wavelength, which is used to determine the number of discretization points on the boundary based on the wavenumber and the length of the boundary. It can be a single value or a vector of values for different components of the boundary.
- `billiard`: The billiard domain for which we are solving the CFIE. It contains the geometry of the problem, including the boundary curves and their properties, which are essential for constructing the system matrix and solving the eigenvalue problem.
- `min_pts`: Minimum number of discretization points on the boundary, which ensures that even for low wavenumbers or short boundaries, we have enough points to accurately represent the geometry and solve the integral equation. Default is 20.
- `eps`: Unused internally.
- `symmetry`: Symmetry information for the billiard, which can be used to reduce the computational cost by exploiting symmetries in the geometry. It can be `nothing` if no symmetry is present or an instance of a type that implements `AbsSymmetry` if symmetries are present. Default is `nothing`.

# Output:
- An instance of the `CFIE_kress` solver initialized with the provided parameters.
"""
function CFIE_kress(pts_scaling_factor::Union{T,Vector{T}},billiard::Bi;min_pts=20,eps=T(1e-15),symmetry::Union{Nothing,AbsSymmetry}=nothing) where {T<:Real,Bi<:AbsBilliard}
    bs=typeof(pts_scaling_factor)==T ? [pts_scaling_factor] : pts_scaling_factor
    sampler=[LinearNodes()]
    Sym=typeof(symmetry)
    return CFIE_kress{T,Bi,Sym}(sampler,bs,bs[1],eps,min_pts,min_pts,billiard,symmetry)
end

"""
    CFIE_kress_corners(pts_scaling_factor::Union{T,Vector{T}},billiard::Bi;min_pts=20,eps=T(1e-15),symmetry::Union{Nothing,AbsSymmetry}=nothing,kressq=8)

Constructor for CFIE_kress_corners solver. 

# Inputs:
- `pts_scaling_factor`: A scaling factor for the number of points per wavelength, which is used to determine the number of discretization points on the boundary based on the wavenumber and the length of the boundary. It can be a single value or a vector of values for different components of the boundary.
- `billiard`: The billiard domain for which we are solving the CFIE. It contains the geometry of the problem, including the boundary curves and their properties, which are essential for constructing the system matrix and solving the eigenvalue problem.
- `min_pts`: Minimum number of discretization points on the boundary, which ensures that even for low wavenumbers or short boundaries, we have enough points to accurately represent the geometry and solve the integral equation. Practically irrelevant since we will always be above this value. Default is 20.
- `eps`: Unused internally.
- `symmetry`: Symmetry information for the billiard, which can be used to reduce the computational cost by exploiting symmetries in the geometry. It can be `nothing` if no symmetry is present or an instance of a type that implements `AbsSymmetry` if symmetries are present. Default is `nothing`.
- `kressq`: The grading parameter q in Kress's grading technique, which controls the clustering of discretization points near the corners. Default is 4. !!! DO NOT PUSH THIS PAST 4 IN FLOAT64
"""
function CFIE_kress_corners(pts_scaling_factor::Union{T,Vector{T}},billiard::Bi;min_pts=20,eps=T(1e-15),symmetry::Union{Nothing,AbsSymmetry}=nothing,kressq=4) where {T<:Real,Bi<:AbsBilliard}
    bs=typeof(pts_scaling_factor)==T ? [pts_scaling_factor] : pts_scaling_factor
    sampler=[LinearNodes()]
    Sym=typeof(symmetry)
    return CFIE_kress_corners{T,Bi,Sym}(sampler,bs,bs[1],eps,min_pts,min_pts,billiard,symmetry,kressq)
end

"""
    CFIE_kress_global_corners(pts_scaling_factor::Union{T,Vector{T}},billiard::Bi;min_pts=20,eps=T(1e-15),symmetry::Union{Nothing,AbsSymmetry}=nothing,kressq=8)

Constructor for CFIE_kress_global_corners solver.

# Inputs:
- `pts_scaling_factor`: A scaling factor for the number of points per wavelength, which is used to determine the number of discretization points on the boundary based on the wavenumber and the length of the boundary. It can be a single value or a vector of values for different components of the boundary.
- `billiard`: The billiard domain for which we are solving the CFIE. It contains the geometry of the problem, including the boundary curves and their properties, which are essential for constructing the system matrix and solving the eigenvalue problem.
- `min_pts`: Minimum number of discretization points on the boundary, which ensures that even for low wavenumbers or short boundaries, we have enough points to accurately represent the geometry and solve the integral equation. Practically irrelevant since we will always be above this value. Default is 20.
- `eps`: Unused internally.
- `symmetry`: Symmetry information for the billiard, which can be used to reduce the computational cost by exploiting symmetries in the geometry. It can be `nothing` if no symmetry is present or an instance of a type that implements `AbsSymmetry` if symmetries are present. Default is `nothing`.
- `kressq`: The grading parameter q in the global Kress grading technique, which controls the clustering of discretization points near the corners. Default is 8.
"""
function CFIE_kress_global_corners(pts_scaling_factor::Union{T,Vector{T}},billiard::Bi;min_pts=20,eps=T(1e-15),symmetry::Union{Nothing,AbsSymmetry}=nothing,kressq=4) where {T<:Real,Bi<:AbsBilliard}
    bs=pts_scaling_factor isa T ? [pts_scaling_factor] : pts_scaling_factor
    sampler=[LinearNodes()]
    Sym=typeof(symmetry)
    return CFIE_kress_global_corners{T,Bi,Sym}(sampler,bs,bs[1],eps,min_pts,min_pts,billiard,symmetry,kressq)
end

#################################
#### CONSTRUCTOR CFIE_alpert ####
#################################


"""
    CFIE_alpert{T,Bi,Sym} <: CFIE

Combined-field integral equation solver using Alpert hybrid Gauss-trapezoidal
near-singular quadrature on panelized boundaries.

This type implements the CFIE on boundaries represented panel-wise, with special
near-singular correction handled by Alpert quadrature rather than Kress
periodic logarithmic splitting. It is especially useful for:
- composite boundaries made of multiple smooth segments,
- cornered geometries,
- situations where an open-panel viewpoint is more natural than a global
  periodic Kress discretization.

The underlying Fredholm operator is again

    A = I - ( D + i k S ),

but now the singular/near-singular behavior is treated through local corrected
quadrature rules rather than a global dense `R` matrix.

# Fields
- `sampler::Vector{LinearNodes}`:
  Placeholder field for the common solver API.
- `pts_scaling_factor::Vector{T}`:
  Controls the base sampling density.
- `dim_scaling_factor::T`:
  Compatibility field.
- `eps::T`:
  Compatibility / tolerance placeholder.
- `min_dim::Int64`:
  Compatibility field.
- `min_pts::Int64`:
  Minimum number of points on each panel.
- `billiard::Bi`:
  Billiard geometry.
- `symmetry::Sym`:
  Optional symmetry descriptor.
- `alpert_order::Int`:
  Order of the Alpert correction rule.
- `alpertq::Int`:
  Panel grading strength parameter used in the endpoint clustering map.

# Mathematical viewpoint
Where the Kress method removes the logarithmic singularity through an analytic
periodic split, the Alpert method instead corrects the quadrature locally near
the singularity using specially designed hybrid nodes and weights. This makes it
well suited to open panels and multi-panel piecewise smooth boundaries.

# When to use this type
Use `CFIE_alpert` when:
- the boundary is naturally panelized (but has true corners - polygons),
- you want strong near-singular correction without building a global periodic
  Kress structure,
- you want a method that handles composite boundaries and corners in a natural,
  panel-based way.
"""
struct CFIE_alpert{T<:Real,Bi<:AbsBilliard,Sym}<:CFIE
    sampler::Vector{LinearNodes}
    pts_scaling_factor::Vector{T}
    dim_scaling_factor::T
    eps::T
    min_dim::Int64
    min_pts::Int64
    billiard::Bi
    symmetry::Sym
    alpert_order::Int
    alpertq::Int
end

"""
    _warn_aggressive_alpert(pts_scaling_factor, billiard, alpert_order, alpertq)

Internal heuristic warning helper for CFIE-Alpert parameter choices. It does not prove that a discretization is invalid or
accurate, but it is useful as an early warning that the shortest panel may be
insufficiently resolved. It is based on testings and experience with the Alpert near-correction structure
and is designed to be conservative in its warnings.

Purpose
-------
This function estimates whether the chosen combination of:
- Alpert correction order,
- grading strength,
- base boundary resolution,
- geometry panel lengths,

may be too aggressive for the shortest boundary segment.

Why this matters
----------------
In Alpert-based panel quadrature, the effective near-correction becomes harder
to resolve when:
- the correction order is large,
- the endpoint grading is strong,
- the point density per wavelength is small,
- one panel is much shorter than the others.

Heuristic
---------
The function computes a rough dimensionless ratio

    R ~ (alpert_order * alpertq) / (bmin * Lmin / Lavg),

where:
- `bmin` is the smallest point-scaling factor,
- `Lmin` is the shortest segment length,
- `Lavg` is the average segment length.

Large values of `R` indicate that the near-correction may be too strong for the
available resolution.

# Behavior
- If `R > 6`, it emits a warning.
- If `4 < R <= 6`, it emits an informational message.
- Otherwise it stays silent.

# Arguments
- `pts_scaling_factor`
- `billiard`
- `alpert_order::Int`
- `alpertq::Int`

# Returns
- `nothing`
"""
function _warn_aggressive_alpert(pts_scaling_factor,billiard,alpert_order::Int,alpertq::Int)
    bs=pts_scaling_factor isa AbstractVector ? pts_scaling_factor : [pts_scaling_factor]
    bmin=minimum(bs)
    boundary=billiard.full_boundary
    lens=Float64[]
    if boundary[1] isa AbstractVector
        for comp in boundary;append!(lens,[crv.length for crv in comp]);end
    else;append!(lens,[crv.length for crv in boundary]);end
    Lmin=minimum(lens) # minimum length is the most problematic since the near-correction will be strongest there and we need to make sure we have enough points to resolve it. 
    Lavg=sum(lens)/length(lens) # The average length is also relevant since it gives us a sense of the overall discretization density.
    # heuristic danger ratio:
    # bigger order, bigger q, smaller b, shorter smallest panel => more dangerous
    R=(alpert_order*alpertq)/(bmin*Lmin/Lavg)
    if R>6.0
        b_suggest=(alpert_order*alpertq)/(4.0*Lmin/Lavg)
        q_suggest=(4.0*bmin*Lmin/Lavg)/alpert_order
        @warn "CFIE_alpert: aggressive grading / near-correction may be under-resolved on the shortest boundary segment." b=bmin alpert_order=alpert_order alpertq=alpertq shortest_segment=Lmin average_segment=Lavg ratio=R suggested_min_b=b_suggest suggested_max_q=q_suggest
    elseif R>4.0
        b_suggest=(alpert_order*alpertq)/(4.0*Lmin/Lavg)
        q_suggest=(4.0*bmin*Lmin/Lavg)/alpert_order
        @info "CFIE_alpert: borderline grading / correction strength." b=bmin alpert_order=alpert_order alpertq=alpertq shortest_segment=Lmin average_segment=Lavg ratio=R suggested_min_b=b_suggest suggested_max_q=q_suggest
    end
    return nothing
end

"""
    CFIE_alpert(pts_scaling_factor::Union{T,Vector{T}},billiard::Bi;min_pts=20,eps=T(1e-15),symmetry::Union{Nothing,S}=nothing,alpert_order=16,alpertq=8) where {T<:Real,Bi<:AbsBilliard,S<:AbsSymmetry}

Constructor for CFIE_alpert solver.

# Inputs:
- `pts_scaling_factor`: A scaling factor for the number of points per wavelength, which is used to determine the number of discretization points on the boundary based on the wavenumber and the length of the boundary. It can be a single value or a vector of values for different components of the boundary.
- `billiard`: The billiard domain for which we are solving the CFIE. It contains the geometry of the problem, including the boundary curves and their properties, which are essential for constructing the system matrix and solving the eigenvalue problem.
- `min_pts`: Minimum number of discretization points on the boundary, which ensures that even for low wavenumbers or short boundaries, we have enough points to accurately represent the geometry and solve the integral equation. Default is 20.
- `eps`: Unused internally.
- `symmetry`: Symmetry information for the billiard, which can be used to reduce the computational cost by exploiting symmetries in the geometry. It can be `nothing` if no symmetry is present or an instance of a type that implements `AbsSymmetry` if symmetries are present. Default is `nothing`.
- `alpert_order`: The order of the Alpert quadrature correction to use for near interactions. Supported values are 2, 3, 4, 5, 6, 8, 10, 12, 14, and 16. Default is 12.
- `alpertq`: The grading strength parameter for the Alpert quadrature. Default is 4.

# Output:
- An instance of the `CFIE_alpert` solver initialized with the provided parameters.
"""
function CFIE_alpert(pts_scaling_factor::Union{T,Vector{T}},billiard::Bi;min_pts::Int=20,eps::T=T(1e-15),symmetry::Union{Nothing,AbsSymmetry}=nothing,alpert_order::Int=12,alpertq::Int=4) where {T<:Real,Bi<:AbsBilliard}
    !(alpert_order in (2,3,4,5,6,8,10,12,14,16)) && error("Alpert order not currently supported")
    _warn_aggressive_alpert(pts_scaling_factor,billiard,alpert_order,alpertq)
    _=alpert_log_rule(T,alpert_order)
    bs=pts_scaling_factor isa T ? [pts_scaling_factor] : pts_scaling_factor
    sampler=[LinearNodes()]
    Sym=typeof(symmetry)
    return CFIE_alpert{T,Bi,Sym}(sampler,bs,bs[1],eps,min_pts,min_pts,billiard,symmetry,alpert_order,alpertq)
end

#### use N even for the algorithm - equidistant parameters ####
s(k::Int,N::Int)=two_pi*k/N


"""
    _reverse_component_orientation(solver::CFIE, pts::BoundaryPointsCFIE)

Reverse the orientation of one boundary component while preserving its role as a
closed periodic or open-panel object.

Purpose
-------
In multiply connected billiards, the outer boundary and hole boundaries must
carry opposite orientations for the boundary integral formulation to be
consistent. This helper is used to reverse the point ordering for holes while
also updating the derivative data so that the geometry remains mathematically
correct.

What gets reversed
------------------
- `xy`: point order is reversed,
- `tangent`: reversed and negated,
- `tangent_2`: reversed,
- `ts`, `ws`, `ws_der`, `ds`: reversed,
- endpoint data `xL, xR, tL, tR`: swapped and tangents sign-flipped.

What does not change
--------------------
- `compid` stays the same, because the component identity is unchanged,
- `is_periodic` stays the same.

# Arguments
- `solver::CFIE`
- `pts::BoundaryPointsCFIE`

# Returns
- A new `BoundaryPointsCFIE` with reversed orientation.
"""
function _reverse_component_orientation(solver::S,pts::BoundaryPointsCFIE{T}) where {T<:Real,S<:CFIE}
    xy=reverse(pts.xy)
    tangent=reverse(-pts.tangent)
    tangent_2=reverse(pts.tangent_2)
    ts=reverse(pts.ts)
    ws=reverse(pts.ws)
    ws_der=reverse(pts.ws_der)
    ds=reverse(pts.ds)
    xL=pts.xR
    xR=pts.xL
    tL=-pts.tR
    tR=-pts.tL
    return BoundaryPointsCFIE(xy,tangent,tangent_2,ts,ws,ws_der,ds,pts.compid,pts.is_periodic,xL,xR,tL,tR)
end

###############
#### KRESS ####
###############

"""
    _evaluate_points(solver::CFIE_kress, crv, k, idx)

Construct the periodic Kress discretization for one smooth closed boundary
component.

Purpose
-------
This helper generates the `BoundaryPointsCFIE` representation of one smooth
closed curve for the periodic CFIE-Kress method. It provides:
- sampled boundary points,
- first and second derivatives with respect to the Kress variable,
- periodic quadrature weights,
- geometric arc-length increments.

Resolution logic
----------------
For a component of length `L`, the number of periodic nodes is chosen roughly as

    N ≈ k * L * b / (2π),

subject to:
- `N ≥ min_pts`,
- compatibility with active rotational symmetry,
- evenness constraints used by the periodic Kress infrastructure.

Parameterization
----------------
The actual geometric curves in the library are typically defined on `[0,1]`,
while Kress uses a periodic variable `t ∈ [0, 2π)`. Therefore:
- nodes are generated in the periodic variable `t`,
- those nodes are rescaled to `u = t / (2π)` for geometric evaluation,
- the geometry derivatives are rescaled by the chain rule:

    dγ/dt   = (dγ/du) / (2π)
    d²γ/dt² = (d²γ/du²) / (2π)².

Weights
-------
- `ws = 2π / N` are the periodic trapezoidal weights in the Kress variable,
- `ws_der = 1` because no grading is applied in the smooth periodic case.

# Arguments
- `solver::CFIE_kress`
- `crv::AbsCurve`:
  One smooth closed curve.
- `k`:
  Real wavenumber.
- `idx::Int`:
  Boundary component index.

# Returns
- `BoundaryPointsCFIE`
"""
function _evaluate_points(solver::CFIE_kress{T},crv::C,k::T,idx::Int) where {T<:Real,C<:AbsCurve}
    L=crv.length
    bs=solver.pts_scaling_factor
    N=max(solver.min_pts,round(Int,k*L*bs[1]/(two_pi)))
    needed=2 # need it to. be even number of points for reflections and at same type divisible by rotation order for rotations. A bit hacky but valid for reflections/rotations. If we dont do this build_rotation_maps_components crashes
    if !isnothing(solver.symmetry)
        sym=solver.symmetry
        if sym isa Rotation
            needed=lcm(needed,sym.n)
        end
    end
    remN=mod(N,needed)
    remN!=0 && (N+=needed-remN)
    ts=[s(k,N) for k in 1:N]
    ts_rescaled=ts./two_pi # b/c our curves and tangents are defined on [0,1]
    xy=curve(crv,ts_rescaled) 
    tangent_1st=tangent(crv,ts_rescaled)./(two_pi) # ! Rescaled tangents due to chain rule ∂γ/∂θ = ∂γ/∂u * ∂u/∂θ = ∂γ/∂u * 1/(2π)
    tangent_2nd=tangent_2(crv,ts_rescaled)./(two_pi)^2 # ! Rescaled tangents due to chain rule ∂²γ/∂θ² = ∂²γ/∂u² * (∂u/∂θ)² + ∂γ/∂u * ∂²u/∂θ² = ∂²γ/∂u² * 1/(2π)^2 + ∂γ/∂u * 0 = ∂²γ/∂u² * 1/(2π)^2
    ss=arc_length(crv,ts_rescaled)
    ds=diff(ss)
    append!(ds,L+ss[1]-ss[end])
    ws=fill(T(two_pi/N),N)
    ws_der=ones(T,N) # unused, legacy
    return BoundaryPointsCFIE(xy,tangent_1st,tangent_2nd,ts,ws,ws_der,ds,idx,true,SVector(zero(T),zero(T)),SVector(zero(T),zero(T)),SVector(zero(T),zero(T)),SVector(zero(T),zero(T)))
end

"""
    _evaluate_points(solver::CFIE_kress_corners, crv, k, idx)

Construct a Kress-graded discretization for one closed curve with corner-type
endpoint singular behavior.

Purpose
-------
This helper is the corner-graded analogue of the smooth periodic Kress
discretization. It replaces the uniform periodic parameter with a graded
computational variable `σ`, whose map `t = w(σ)` clusters nodes near the corner
locations.

Resolution logic
----------------
As in the smooth case, the total node count is chosen from the boundary length
and wavenumber, but now:
- odd `N` is enforced,
- rotational-symmetry compatibility is enforced when needed,
- Kress grading data is generated by `kress_graded_nodes_data`.

Grading transformation
----------------------
The grading map returns:
- `σ`: computational nodes,
- `tmap`: mapped periodic variable,
- `jac = dt/dσ`,
- `jac2 = d²t/dσ²`.

Geometry derivatives are transformed by the chain rule:

    γ'(σ)  = γ'(u) * jac / (2π)
    γ''(σ) = γ''(u) * (jac / 2π)^2 + γ'(u) * (jac2 / 2π),

because the underlying geometric parameter is still normalized to `[0,1]`.

Weights
-------
- base quadrature weights are `h = π / ((N+1)/2)`,
- `ws_der = jac` stores the Jacobian of the grading map.

# Arguments
- `solver::CFIE_kress_corners`
- `crv::AbsCurve`
- `k`
- `idx::Int`

# Returns
- `BoundaryPointsCFIE`
"""
function _evaluate_points(solver::CFIE_kress_corners{T},crv::C,k::T,idx::Int) where {T<:Real,C<:AbsCurve}
    L=crv.length
    bs=solver.pts_scaling_factor
    N=max(solver.min_pts,round(Int,k*L*bs[1]/two_pi))
    needed=1
    if !isnothing(solver.symmetry)
        sym=solver.symmetry
        if sym isa Rotation
            iseven(sym.n) && error("Incompatible. If sym.n is even, please use reflections.")
            needed=lcm(needed,sym.n)
        end
    end
    remN=mod(N,needed)
    remN!=0 && (N+=needed-remN)
    iseven(N) && (N+=needed)
    σ,tmap,jac,jac2,_=kress_graded_nodes_data(T,N;q=solver.kressq)
    u=tmap./two_pi
    xy=curve(crv,u)
    γu=tangent(crv,u)
    γuu=tangent_2(crv,u)
    tangent_1st=[γu[i]*(jac[i]/two_pi) for i in eachindex(u)]
    tangent_2nd=[γuu[i]*(jac[i]/two_pi)^2+γu[i]*(jac2[i]/two_pi) for i in eachindex(u)]
    ss=arc_length(crv,u)
    ds=diff(ss)
    append!(ds,L+ss[1]-ss[end])
    h=pi/T((N+1)÷2)
    ts=σ
    ws=fill(h,N)
    ws_der=jac
    return BoundaryPointsCFIE(xy,tangent_1st,tangent_2nd,ts,ws,ws_der,ds,idx,true,SVector(zero(T),zero(T)),SVector(zero(T),zero(T)),SVector(zero(T),zero(T)),SVector(zero(T),zero(T)))
end

############################
#### KRESS MULTI CORNER ####
############################

"""
    _evaluate_points(solver::CFIE_kress_global_corners, comp, k, idx)

Construct one globally graded periodic discretization for a closed composite
boundary component consisting of multiple joined segments.

Purpose
-------
This helper takes a vector of curve segments forming one closed component and
turns them into a single `BoundaryPointsCFIE` object compatible with the global
Kress CFIE assembly.

Why global grading is needed
----------------------------
For Kress splitting to work correctly, the entire closed component must be
treated as one periodic object. If one discretized each segment independently,
the periodic logarithmic structure of the self-interaction kernel would be lost.

# Arguments
- `solver::CFIE_kress_global_corners`
- `comp::Vector{AbsCurve}`:
  Segments forming one closed component.
- `k`
- `idx::Int`

# Returns
- `BoundaryPointsCFIE`
"""
function _evaluate_points(solver::CFIE_kress_global_corners{T},comp::Vector{C},k::T,idx::Int) where {T<:Real,C<:AbsCurve}
    # total length
    _,_,Ltot=component_lengths(comp)
    # choose number of nodes
    bs=solver.pts_scaling_factor
    N=max(solver.min_pts,round(Int,k*Ltot*bs[1]/two_pi))
    # enforce symmetry compatibility - like above 
    needed=1
    if !isnothing(solver.symmetry)
        sym=solver.symmetry
        if sym isa Rotation
            iseven(sym.n) && error("Incompatible. If sym.n is even, please use reflections.")
            needed=lcm(needed,sym.n)
        end
    end
    remN=mod(N,needed)
    remN!=0 && (N+=needed-remN)
    iseven(N) && (N+=needed)  # enforce odd N
    # build corner locations
    corners=_component_corner_locations(T,comp)
    # global graded nodes
    σ,tmap,jac,jac2,_=multi_kress_graded_nodes_data(T,N,corners;q=solver.kressq)
    # allocate outputs
    xy=Vector{SVector{2,T}}(undef,N)
    tangent_1st=Vector{SVector{2,T}}(undef,N)
    tangent_2nd=Vector{SVector{2,T}}(undef,N)
    @inbounds for i in 1:N
        q,γt,γtt=_eval_composite_geom_global_t(T,comp,tmap[i])
        # combine geometry derivatives with grading derivatives
        xy[i]=q
        tangent_1st[i]=γt*jac[i]
        tangent_2nd[i]=γtt*(jac[i]^2)+γt*jac2[i]
    end
    ds=Vector{T}(undef,N)
    # Kress weights
    h=pi/T((N+1)÷2)
    @inbounds for i in 1:N
        tx=tangent_1st[i][1]
        ty=tangent_1st[i][2]
        ds[i]=hypot(tx,ty)*h
    end
    ts=σ  # computational nodes
    ws=fill(h,N) # trapezoidal weights
    ws_der=jac # w'(σ)
    return BoundaryPointsCFIE(xy,tangent_1st,tangent_2nd,ts,ws,ws_der,ds,idx,true,SVector(zero(T),zero(T)),SVector(zero(T),zero(T)),SVector(zero(T),zero(T)),SVector(zero(T),zero(T)))
end

####################
#### HIGH LEVEL ####
####################


"""
    evaluate_points(solver::Union{CFIE_kress,CFIE_kress_corners}, billiard, k)

Construct CFIE-Kress boundary discretizations for all boundary components of the
billiard.

Behavior
--------
This high-level function:
1. extracts the connected boundary components,
2. requires each component to consist of exactly one smooth closed curve,
3. applies the appropriate low-level `_evaluate_points` helper,
4. reverses the orientation of every component except the first one, so that
   holes acquire the correct opposite orientation.

# Arguments
- `solver::CFIE_kress` or `solver::CFIE_kress_corners`
- `billiard::AbsBilliard`
- `k`

# Returns
- `Vector{BoundaryPointsCFIE{T}}`

  where:
  - `pts[1]` is the outer boundary,
  - `pts[2:]` are holes, orientation-reversed.

# Notes
This function does not support composite multi-segment components. For that,
use `CFIE_kress_global_corners`.
"""
function evaluate_points(solver::Union{CFIE_kress{T},CFIE_kress_corners{T}},billiard::Bi,k::T) where {T<:Real,Bi<:AbsBilliard}
    comps=_boundary_components(billiard.full_boundary)
    pts=Vector{BoundaryPointsCFIE{T}}(undef,length(comps))
    for (idx,comp) in enumerate(comps)
        length(comp)==1 || error("Kress requires each boundary component to be a single smooth closed curve.")
        crv=comp[1]
        p=_evaluate_points(solver,crv,k,idx)
        pts[idx]=idx==1 ? p : _reverse_component_orientation(solver,p)
    end
    return pts
end


"""
    evaluate_points(solver::CFIE_kress_global_corners, billiard, k)

Construct CFIE-Kress boundary discretizations for all boundary components using
global multi-corner grading when needed.

Behavior
--------
For each connected component of the boundary:
- if the component has exactly one curve, it falls back to the single-curve
  corner-graded Kress helper,
- if the component is composite, it builds a global graded periodic
  discretization over the whole component,
- components after the first are orientation-reversed to represent holes.

# Arguments
- `solver::CFIE_kress_global_corners`
- `billiard::AbsBilliard`
- `k`

# Returns
- `Vector{BoundaryPointsCFIE{T}}`

# Notes
This is the high-level entry point for the most general Kress-CFIE geometry
handling in this file.
"""
function evaluate_points(solver::CFIE_kress_global_corners{T},billiard::Bi,k::T) where {T<:Real,Bi<:AbsBilliard}
    boundary=billiard.full_boundary
    isempty(boundary) && error("Boundary cannot be empty.")
    # Case 1: one smooth closed curve stored directly
    # full_boundary = [crv]
    if length(boundary)==1 && !(boundary[1] isa AbstractVector)
        crv=boundary[1]
        base=CFIE_kress(solver.pts_scaling_factor,solver.billiard;
            min_pts=solver.min_pts,eps=solver.eps,symmetry=solver.symmetry)
        return [_evaluate_points(base,crv,k,1)]
    end
    # Case 2: one composite outer boundary stored as a flat vector of segments
    # full_boundary = [seg1,seg2,...]
    # rectangle, stadium, polygon, ...
    if _is_single_composite_boundary(boundary)
        return [_evaluate_points(solver,boundary,k,1)]
    end
    # Case 3: multiple boundary components
    # full_boundary = [outer_comp, hole1_comp, ...]
    comps=_boundary_components(boundary)
    pts=Vector{BoundaryPointsCFIE{T}}(undef,length(comps))
    for (idx,comp) in enumerate(comps)
        isempty(comp) && error("Boundary component cannot be empty.")
        p= if length(comp)==1
            # smooth closed component
            base=CFIE_kress(solver.pts_scaling_factor,solver.billiard;
                min_pts=solver.min_pts,eps=solver.eps,symmetry=solver.symmetry)
            _evaluate_points(base,comp[1],k,idx)
        else
            # composite closed component with corners
            _evaluate_points(solver,comp,k,idx)
        end
        pts[idx]=idx==1 ? p : _reverse_component_orientation(solver,p)
    end
    return pts
end

################
#### ALPERT ####
################

"""
    _evaluate_points_periodic(solver::CFIE_alpert, crv, k, idx)

Construct one closed periodic boundary component for the CFIE-Alpert method.

Purpose
-------
Although Alpert quadrature is panel-based, a billiard may also consist of a
single smooth closed curve such as a circle or ellipse. In that case, this
helper builds a periodic closed discretization suitable for the Alpert CFIE
infrastructure.

Sampling choice
---------------
Unlike the Kress periodic helper, this function uses midpoint-like periodic nodes

    t_j = 2π (j - 1/2) / N,

rather than the `2π j / N` convention. This is consistent with the chosen
periodic Alpert implementation.

Derivative rescaling
--------------------
As usual, the curve is geometrically defined on `[0,1]`, so the first and
second derivatives are rescaled by the chain rule:
- first derivative divided by `2π`,
- second derivative divided by `(2π)^2`.

Weights
-------
- `ws = 2π / N`,
- `ws_der = 1`,
- `ds` from arc-length increments around the periodic curve.

# Arguments
- `solver::CFIE_alpert`
- `crv::AbsCurve`
- `k`
- `idx::Int`

# Returns
- `BoundaryPointsCFIE`
"""
function _evaluate_points_periodic(solver::CFIE_alpert{T},crv::C,k::T,idx::Int) where {T<:Real,C<:AbsCurve}
    L=crv.length
    bs=solver.pts_scaling_factor
    N=max(solver.min_pts,round(Int,k*L*bs[1]/two_pi))
    ts=[T(two_pi)*(j-1/2)/N for j in 1:N]
    ts_rescaled=ts./two_pi
    xy=curve(crv,ts_rescaled)
    tangent_1st=tangent(crv,ts_rescaled)./(two_pi)
    tangent_2nd=tangent_2(crv,ts_rescaled)./(two_pi)^2
    ss=arc_length(crv,ts_rescaled)
    ds=diff(ss)
    append!(ds,L+ss[1]-ss[end])
    ws=fill(T(two_pi/N),N)
    ws_der=ones(T,N)
    return BoundaryPointsCFIE(xy,tangent_1st,tangent_2nd,ts,ws,ws_der,ds,idx,true,SVector(zero(T),zero(T)),SVector(zero(T),zero(T)),SVector(zero(T),zero(T)),SVector(zero(T),zero(T)))
end

"""
    _panel_sigma_to_u_jac(solver::CFIE_alpert, σ)

Map an Alpert computational panel coordinate `σ` to the physical panel
parameter `u`, together with the first and second derivatives of the map.

Purpose
-------
The Alpert panel discretization uses a graded coordinate map near endpoints.
This helper provides:
- `u = u(σ)`,
- `du/dσ`,
- `d²u/dσ²`,

which are then used to transform geometry derivatives from the physical panel
parameter to the computational panel variable.

# Arguments
- `solver::CFIE_alpert`
- `σ`:
  Computational coordinate on the panel.

# Returns
- `(u, jac, jac2)`

  where:
  - `u`   is the physical panel coordinate,
  - `jac` is `du/dσ`,
  - `jac2` is `d²u/dσ²`.

# Notes
The grading strength is controlled by `solver.alpertq`.
"""
@inline function _panel_sigma_to_u_jac(solver::CFIE_alpert{T},σ::T) where {T<:Real}
    q=solver.alpertq   # acts as grading strength parameter
    u=_panel_grade_map(σ,q)
    jac=_panel_grade_map_prime(σ,q)
    jac2=_panel_grade_map_doubleprime(σ,q)
    return u,jac,jac2
end

"""
    _evaluate_points_panel(solver::CFIE_alpert, crv, k, idx)

Construct one open-panel discretization for the CFIE-Alpert method.

Purpose
-------
This helper builds the `BoundaryPointsCFIE` object for one panel curve. It is
the fundamental geometry function behind the panel-based Alpert CFIE
implementation.

Sampling logic
--------------
For a panel of length `L`, the node count is chosen roughly as

    N ≈ k * L * b / (2π),

with `N ≥ min_pts`, and at least 2 nodes.

Nodes are placed at panel midpoints in the computational variable `σ`, then
mapped through the Alpert grading transformation to the physical panel
parameter `u`.

Open-panel metadata
-------------------
Because this is an open panel:
- `is_periodic = false`,
- the endpoints `xL`, `xR`,
- and endpoint tangents `tL`, `tR`

are stored explicitly in the returned `BoundaryPointsCFIE`.

# Arguments
- `solver::CFIE_alpert`
- `crv::AbsCurve`
- `k`
- `idx::Int`

# Returns
- `BoundaryPointsCFIE`
"""
function _evaluate_points_panel(solver::CFIE_alpert{T},crv::C,k::T,idx::Int) where {T<:Real,C<:AbsCurve}
    L=crv.length
    bs=solver.pts_scaling_factor
    N=max(solver.min_pts,round(Int,k*L*bs[1]/two_pi))
    N<2 && (N=2)
    hσ=inv(T(N))
    sig=[T(j-0.5)/T(N) for j in 1:N]
    xy=Vector{SVector{2,T}}(undef,N)
    tangent_1st=Vector{SVector{2,T}}(undef,N)
    tangent_2nd=Vector{SVector{2,T}}(undef,N)
    ds=Vector{T}(undef,N)
    @inbounds for j in 1:N
        σ=sig[j]
        u,jac,jac2=_panel_sigma_to_u_jac(solver,σ)
        q=curve(crv,u)
        tu=tangent(crv,u)
        t2u=tangent_2(crv,u)
        xy[j]=q
        tangent_1st[j]=tu*jac
        tangent_2nd[j]=t2u*(jac^2)+tu*jac2
        ds[j]=sqrt((tu[1]*jac)^2+(tu[2]*jac)^2)*hσ
    end
    ws=fill(hσ,N)
    ws_der=ones(T,N)
    xL=curve(crv,zero(T))
    xR=curve(crv,one(T))
    tL=tangent(crv,zero(T))
    tR=tangent(crv,one(T))
    return BoundaryPointsCFIE(xy,tangent_1st,tangent_2nd,sig,ws,ws_der,ds,idx,false,xL,xR,tL,tR)
end

"""
    evaluate_points(solver::CFIE_alpert, billiard, k)

Construct the CFIE-Alpert boundary discretization for the full billiard.

Supported geometry layouts
--------------------------
This high-level function is flexible and supports several boundary descriptions:

1. A vector of smooth closed curves:
   - each curve is treated as one periodic closed component,
   - periodic Alpert discretization is used.

2. A single composite boundary described by several segments:
   - each segment is treated as a separate open panel.

3. Several connected components, each itself consisting of one or more panels:
   - each panel is discretized separately,
   - components after the first are orientation-reversed to represent holes.

Boundary choice and symmetry
----------------------------
If no symmetry is active, the function uses `billiard.full_boundary`.
If symmetry is active, it uses `billiard.desymmetrized_full_boundary`.

# Arguments
- `solver::CFIE_alpert`
- `billiard::AbsBilliard`
- `k`

# Returns
- `Vector{BoundaryPointsCFIE{T}}`

# Output interpretation
Depending on the geometry structure, the returned vector may contain:
- one object per closed component,
- or one object per panel, with `compid` indicating which connected component
  that panel belongs to.
"""
function evaluate_points(solver::CFIE_alpert{T},billiard::Bi,k::T) where {T<:Real,Bi<:AbsBilliard}
    boundary=isnothing(solver.symmetry) ? billiard.full_boundary : billiard.desymmetrized_full_boundary
    if !(boundary[1] isa AbstractVector) && _all_closed_curves(boundary)
        pts=Vector{BoundaryPointsCFIE{T}}(undef,length(boundary))
        for (idx,crv) in enumerate(boundary)
            p=_evaluate_points_periodic(solver,crv,k,idx)
            pts[idx]=(idx==1) ? p : _reverse_component_orientation(solver,p)
        end
        return pts
    end
    if _is_single_composite_boundary(boundary)
        pts=Vector{BoundaryPointsCFIE{T}}(undef,length(boundary))
        for (idx,crv) in enumerate(boundary)
            pts[idx]=_evaluate_points_panel(solver,crv,k,1)
        end
        return pts
    end
    ncomps=length(boundary)
    npanels=sum(length(comp) for comp in boundary)
    pts=Vector{BoundaryPointsCFIE{T}}(undef,npanels)
    pos=1
    for compid in 1:ncomps
        comp=boundary[compid]
        for crv in comp
            p=_evaluate_points_panel(solver,crv,k,compid)
            pts[pos]=(compid==1) ? p : _reverse_component_orientation(solver,p)
            pos+=1
        end
    end
    return pts
end