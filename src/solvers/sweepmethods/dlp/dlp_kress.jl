# Useful reading:
# - Kress, R., Boundary integral equations in time-harmonic acoustic scattering. Math. Comput. Modelling 15 (1991), 229-243.
# - Barnett, A. H. / Betcke, T., mpspack DLP implementation.
# - Zhao, L. / Barnett, A., Robust and efficient solution of the drum problem via Nyström approximation of the Fredholm determinant.

const two_pi=2*pi
const inv_two_pi=1/two_pi

"""
    DLPKressWorkspace{T,M}

Workspace cache for repeated DLP-Kress matrix assembly on a fixed boundary
discretization.

This object stores all geometry-dependent data that can be reused across many
wavenumbers `k`. The key design idea is that for repeated assembly on the same
boundary nodes, the expensive pairwise geometric quantities should be computed
once, while only the truly k-dependent special-function evaluations are updated.

# Fields
- `Rmat::M`:
  Dense real Kress correction matrix for the logarithmic singular part.
  For a smooth periodic boundary this is the standard periodic Kress matrix.
  For globally graded corner discretizations it is the corner-graded version.
- `G::CFIEGeomCache{T}`:
  Geometry cache built from the boundary points. It stores pairwise distances,
  inverse distances, logarithmic singular terms, curvature, and oriented inner
  products needed by the DLP formulas.
- `parr::CFIEPanelArrays{T}`:
  Flat arrays of coordinates and tangents extracted from `pts`, used for cheap
  indexed access inside tight loops.
- `N::Int`:
  Matrix dimension, equal to the number of boundary points.

# Why a workspace matters
For determinant scans, k-sweeps, Newton refinement, and EBIM-style repeated
assembly, rebuilding all geometry data at every `k` would be wasteful. This
workspace isolates the immutable geometry from the mutable spectral parameter.

# Typical use
1. Generate `pts = evaluate_points(solver,billiard,kref)`.
2. Build `ws = build_dlp_kress_workspace(solver,pts)`.
3. Reuse `ws` in repeated calls to `construct_matrices!`, `solve`, or
   `solve_vect`.
"""
struct DLPKressWorkspace{T<:Real,M<:AbstractMatrix{T}}
    Rmat::M
    G::CFIEGeomCache{T}
    parr::CFIEPanelArrays{T}
    N::Int
end

"""
    DLP_kress{T,Bi,Sym} <: SweepSolver

Solver type for the Kress-corrected double-layer Fredholm formulation on a
single smooth closed boundary component.

This is the smooth periodic version of the DLP-based Fredholm determinant
approach. It assumes the outer boundary is one closed `C^∞` or at least
sufficiently smooth periodic curve, parameterized without corners. The singular
self-interaction is treated using Kress's periodic logarithmic quadrature.

The associated Fredholm matrix constructed in this file is

    F(k) = I - D(k),

where `D(k)` is the Nyström discretization of the interior Helmholtz
double-layer boundary operator using Kress's singular splitting.

# Fields
- `sampler::Vector{LinearNodes}`:
  Placeholder sampling descriptor, kept consistent with the library’s solver
  interface. In this implementation the actual periodic node set is built
  directly in `_evaluate_points`.
- `pts_scaling_factor::Vector{T}`:
  Controls boundary resolution as a function of `k`. The default node count
  scales like `N ≈ k*L*b/(2π)`, with `b = pts_scaling_factor[1]`.
- `dim_scaling_factor::T`:
  Included for compatibility with the common sweep/refinement infrastructure.
  DLP itself is boundary-only, but the field is kept so the generic solver
  layer can refine in a uniform way.
- `eps::T`:
  Numerical tolerance placeholder.
- `min_dim::Int64`, `min_pts::Int64`:
  Minimum discretization controls. Here `min_pts` is the essential one;
  `min_dim` is kept for consistency with other sweep solvers.
- `billiard::Bi`:
  Underlying billiard geometry.
- `symmetry::Sym`:
  Optional symmetry reduction descriptor. In practice this mostly affects node
  count compatibility conditions, but can be used with Beyn for proper desymmetrization.

# Mathematical setting
This type is meant for the periodic smooth-boundary Kress machinery, where the
kernel singularity is expressed through the universal periodic logarithm

    log(4 sin^2((t-s)/2)),

and treated by the precomputed Kress `R` matrix.

# Restrictions
- exactly one outer boundary component;
- that component must be a single smooth closed curve;
- holes are not supported here - cfie;
- corners are not supported here.

For cornered geometries use `DLP_kress_global_corners`.
"""
struct DLP_kress{T<:Real,Bi<:AbsBilliard,Sym}<:SweepSolver
    sampler::Vector{LinearNodes}
    pts_scaling_factor::Vector{T}
    dim_scaling_factor::T
    eps::T
    min_dim::Int64
    min_pts::Int64
    billiard::Bi
    symmetry::Sym
end

"""
    DLP_kress_global_corners{T,Bi,Sym} <: SweepSolver

Solver type for the Kress-corrected double-layer Fredholm formulation on a
single outer boundary component represented by several joined curve segments and
globally resolved through Kress grading.

This is the corner-capable counterpart of `DLP_kress`. It targets piecewise
smooth boundaries where the full outer boundary may contain corners, but there
are no holes. The singularity treatment still follows the Kress philosophy, but
the discretization is no longer uniform in the physical parameter: instead, a
graded parameter mesh is introduced so that endpoint/corner singular structure
is regularized.

The associated Fredholm matrix is again

    F(k) = I - D(k),

but now built on a globally graded boundary parameter.

# Fields
- `sampler::Vector{LinearNodes}`:
  Placeholder interface field, analogous to other sweep solvers.
- `pts_scaling_factor::Vector{T}`:
  Resolution scaling factor controlling node growth with `k`.
- `dim_scaling_factor::T`:
  Compatibility field for the generic refinement layer.
- `eps::T`:
  Numerical tolerance placeholder.
- `min_dim::Int64`, `min_pts::Int64`:
  Minimum discretization parameters.
- `billiard::Bi`:
  Underlying billiard.
- `symmetry::Sym`:
  Optional symmetry descriptor.
- `kressq::Int`:
  Grading parameter controlling how strongly nodes are clustered near corners.
  Larger values correspond to stronger smoothing of endpoint singular behavior.

# Mathematical setting
This type assumes:
- one outer boundary component only;
- that component may be composed of several smooth segments;
- corners are handled globally by a Kress-type grading map;
- the corner-log correction matrix must match the graded periodic indexing,
  hence the use of `kress_R_corner!`.

# When to use this type
Use it for rectangles, polygons, stadium-like joined arcs/segments, or any
piecewise smooth single-boundary geometry where the smooth periodic
assumptions of `DLP_kress` do not apply.
"""
struct DLP_kress_global_corners{T<:Real,Bi<:AbsBilliard,Sym}<:SweepSolver
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

function DLP_kress(pts_scaling_factor::Union{T,Vector{T}},billiard::Bi;min_pts=20,eps=T(1e-15),symmetry::Union{Nothing,AbsSymmetry}=nothing) where {T<:Real,Bi<:AbsBilliard}
    bs=pts_scaling_factor isa T ? [pts_scaling_factor] : pts_scaling_factor
    sampler=[LinearNodes()]
    Sym=typeof(symmetry)
    return DLP_kress{T,Bi,Sym}(sampler,bs,bs[1],eps,min_pts,min_pts,billiard,symmetry)
end

function DLP_kress_global_corners(pts_scaling_factor::Union{T,Vector{T}},billiard::Bi;min_pts=20,eps=T(1e-15),symmetry::Union{Nothing,AbsSymmetry}=nothing,kressq=4) where {T<:Real,Bi<:AbsBilliard}
    bs=pts_scaling_factor isa T ? [pts_scaling_factor] : pts_scaling_factor
    sampler=[LinearNodes()]
    Sym=typeof(symmetry)
    return DLP_kress_global_corners{T,Bi,Sym}(sampler,bs,bs[1],eps,min_pts,min_pts,billiard,symmetry,kressq)
end

@inline _is_dlp_kress_graded(::DLP_kress)=false
@inline _is_dlp_kress_graded(::DLP_kress_global_corners)=true

"""
    build_dlp_kress_workspace(solver,pts)

Build and return a `DLPKressWorkspace` for a fixed boundary discretization.

This function collects all geometry-dependent ingredients needed for repeated
matrix assembly:

1. the Kress correction matrix `Rmat`,
2. the pairwise geometry cache `G`,
3. the unpacked panel arrays `parr`,
4. the matrix size `N`.

# Arguments
- `solver::Union{DLP_kress,DLP_kress_global_corners}`:
  Determines whether the smooth or corner-graded Kress correction is built and
  whether the geometry cache includes grading-aware logarithmic corrections.
- `pts::BoundaryPointsCFIE{T}`:
  Boundary discretization returned by `evaluate_points`.

# Returns
- `DLPKressWorkspace{T}`

# Why this function exists
Repeatedly constructing `Rmat` and the pairwise geometry cache is expensive.
This builder should be used whenever the same boundary points will be reused for
multiple `k` values, such as in:
- k-sweeps,
- Newton refinement,
- Krylov-based EBIM routines.

# Notes
`pts` must already be compatible with the solver type:
- smooth periodic for `DLP_kress`,
- odd-sized graded discretization for `DLP_kress_global_corners`.
"""
function build_dlp_kress_workspace(solver::Union{DLP_kress,DLP_kress_global_corners},pts::BoundaryPointsCFIE{T}) where {T<:Real}
    Rmat=build_Rmat_dlp_kress(solver,pts)
    G=_is_dlp_kress_graded(solver) ? cfie_geom_cache(pts,true) : cfie_geom_cache(pts,false)
    parr=_panel_arrays_cache(pts)
    N=length(pts.xy)
    return DLPKressWorkspace(Rmat,G,parr,N)
end

"""
    build_Rmat_dlp_kress(solver::DLP_kress,pts)

Build the periodic Kress correction matrix `Rmat` for a smooth closed boundary.

# Arguments
- `solver::DLP_kress`
- `pts::BoundaryPointsCFIE{T}`

# Returns
- `Rmat::Matrix{T}` of size `N × N`, where `N = length(pts.xy)`.

# Mathematical role
In the Kress decomposition, the singular self-interaction of the double-layer
kernel is written as

    logarithmic coefficient × universal periodic log kernel
    + smooth remainder.

The matrix `Rmat` is the Nyström discretization of that universal periodic
logarithmic kernel on the chosen periodic grid. It is the discrete object that
encodes the singular quadrature correction.

# Structure
For a smooth periodic equispaced grid, `Rmat` is the standard Kress periodic
matrix produced by `kress_R!`. Conceptually it is circulant because the
underlying periodic logarithm depends only on index difference.

# When this is valid
Only for the smooth periodic case. If the boundary is globally graded for
corners, the corner-specific builder must be used instead.
"""
function build_Rmat_dlp_kress(solver::DLP_kress,pts::BoundaryPointsCFIE{T}) where {T<:Real}
    N=length(pts.xy)
    Rmat=zeros(T,N,N)
    kress_R!(Rmat)
    return Rmat
end

"""
    build_Rmat_dlp_kress(solver::DLP_kress_global_corners,pts)

Build the corner-graded Kress correction matrix `Rmat` for a globally graded
piecewise smooth outer boundary.

# Arguments
- `solver::DLP_kress_global_corners`
- `pts::BoundaryPointsCFIE{T}`

# Returns
- `Rmat::Matrix{T}` of size `N × N`, where `N = length(pts.xy)`.

# Mathematical role
For globally graded Kress discretizations, the singular logarithmic term must be
represented on the original periodic corner-compatible indexing rather than the
simple smooth-periodic circulant form. The appropriate discrete correction is
therefore generated with `kress_R_corner!`.

# Why this differs from the smooth case
Even though the DLP singularity is still logarithmic after splitting, the
parameterization is no longer uniform in the physical curve parameter. The
corner-aware correction matrix must therefore reflect the graded periodic mesh.
"""
function build_Rmat_dlp_kress(solver::DLP_kress_global_corners,pts::BoundaryPointsCFIE{T}) where {T<:Real}
    N=length(pts.xy)
    Rmat=zeros(T,N,N)
    kress_R_corner!(Rmat)
    return Rmat
end

"""
    _evaluate_points(solver::DLP_kress{T}, crv::C, k::T, idx::Int)

Construct a smooth periodic Nyström discretization of a single closed curve
for the DLP-Kress solver.

This function builds the full `BoundaryPointsCFIE` object required for assembly
of the Kress-corrected double-layer operator. It assumes a single smooth,
periodic boundary and uses a uniform parameter grid in the Kress variable
`t ∈ [0,2π)`.

# Mathematical role
The boundary is discretized using a periodic trapezoidal rule combined with
Kress logarithmic singularity subtraction. The discretization must therefore
provide:
- boundary coordinates,
- first and second derivatives with respect to the Kress parameter,
- quadrature weights in the periodic variable,
- local arclength increments.

# Resolution strategy
The number of nodes is chosen as:
    N ≈ k * L / (2π)
scaled by `pts_scaling_factor`, and enforced to satisfy:
- N ≥ min_pts,
- compatibility with symmetry (e.g. rotational periodicity),
- evenness required by periodic Kress quadrature.

# Parameterization details
- Nodes are generated in the Kress variable `t ∈ [0,2π)`.
- Geometry is evaluated in a normalized parameter, so derivatives are rescaled:
    γ'(t)  = γ'(σ)/(2π)
    γ''(t) = γ''(σ)/(2π)^2

# Quadrature structure
- `ws = 2π/N` are uniform periodic weights,
- `ws_der = 1` since no grading is used.

# Arguments
- `solver`: smooth DLP-Kress solver instance
- `crv`: smooth closed curve
- `k`: wavenumber controlling resolution
- `idx`: component index label

# Returns
- `BoundaryPointsCFIE{T}` containing geometry, derivatives, and quadrature data

# Notes
This function is valid only for smooth, single-component boundaries.
For corners, use `DLP_kress_global_corners`.
"""
function _evaluate_points(solver::DLP_kress{T},crv::C,k::T,idx::Int) where {T<:Real,C<:AbsCurve}
    L=crv.length
    bs=solver.pts_scaling_factor
    N=max(solver.min_pts,round(Int,k*L*bs[1]/two_pi))
    needed=2
    if !isnothing(solver.symmetry)
        sym=solver.symmetry
        if sym isa Rotation
            needed=lcm(needed,sym.n)
        end
    end
    remN=mod(N,needed)
    remN!=0 && (N+=needed-remN)
    ts=[s(j,N) for j in 1:N]
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
    _evaluate_points(solver::DLP_kress_global_corners{T}, comp::Vector{C}, k::T, idx::Int)

Construct a globally graded Nyström discretization for a composite boundary
with corners using the Kress grading transformation.

This function  extends the smooth periodic discretization to piecewise smooth
boundaries by introducing a grading map that clusters nodes near corner
singularities.

# Mathematical role
The boundary is parameterized by a globally periodic variable σ with a nonlinear
map:
    t = t(σ)
such that:
- points cluster near corner locations,
- the logarithmic singularity is regularized,
- the periodic Kress framework remains applicable.

# Grading transformation
Computed via:
    multi_kress_graded_nodes_data(...; q=kressq)

Produces:
- σ: graded nodes
- tmap: mapped geometric parameter
- jac: dt/dσ
- jac2: d²t/dσ²

# Chain rule
Geometry derivatives are transformed as:
    γ'(σ)  = γ'(t) * jac
    γ''(σ) = γ''(t) * jac^2 + γ'(t) * jac2

# Quadrature structure
- base weights: `ws = h`
- Jacobian weights: `ws_der = jac`

These encode the transformed measure separately.

# Arguments
- `solver`: corner-capable DLP-Kress solver
- `comp`: vector of curve segments forming a closed boundary
- `k`: wavenumber
- `idx`: component label

# Returns
- `BoundaryPointsCFIE{T}`
"""
function _evaluate_points(solver::DLP_kress_global_corners{T},comp::Vector{C},k::T,idx::Int) where {T<:Real,C<:AbsCurve}
    _,_,Ltot=component_lengths(comp)
    bs=solver.pts_scaling_factor
    N=max(solver.min_pts,round(Int,k*Ltot*bs[1]/two_pi))
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
    corners=_component_corner_locations(T,comp)
    σ,tmap,jac,jac2,_=multi_kress_graded_nodes_data(T,N,corners;q=solver.kressq)
    xy=Vector{SVector{2,T}}(undef,N)
    tangent_1st=Vector{SVector{2,T}}(undef,N)
    tangent_2nd=Vector{SVector{2,T}}(undef,N)
    @inbounds for i in 1:N
        q,γt,γtt=_eval_composite_geom_global_t(T,comp,tmap[i])
        xy[i]=q
        tangent_1st[i]=γt*jac[i]
        tangent_2nd[i]=γtt*(jac[i]^2)+γt*jac2[i]
    end
    h=pi/T((N+1)÷2)
    ds=Vector{T}(undef,N)
    @inbounds for i in 1:N
        tx=tangent_1st[i][1]
        ty=tangent_1st[i][2]
        ds[i]=hypot(tx,ty)*h
    end
    ts=σ
    ws=fill(h,N)
    ws_der=jac
    return BoundaryPointsCFIE(xy,tangent_1st,tangent_2nd,ts,ws,ws_der,ds,idx,true,SVector(zero(T),zero(T)),SVector(zero(T),zero(T)),SVector(zero(T),zero(T)),SVector(zero(T),zero(T)))
end

"""
    evaluate_points(solver::DLP_kress, billiard, k)

High-level entry point for constructing boundary discretization for smooth
billiards using DLP-Kress.

# Behavior
- Extracts boundary components from `billiard`
- Enforces:
  - exactly one outer boundary
  - exactly one smooth closed curve
- Delegates to `_evaluate_points` for smooth periodic discretization

# Arguments
- `solver`: DLP_kress solver
- `billiard`: billiard geometry
- `k`: wavenumber

# Returns
- `BoundaryPointsCFIE`
"""
function evaluate_points(solver::DLP_kress{T},billiard::Bi,k::T) where {T<:Real,Bi<:AbsBilliard}
    boundary=billiard.full_boundary
    isempty(boundary) && error("Boundary cannot be empty.")
    # Case 1: one smooth closed curve stored directly
    # full_boundary = [crv]
    # -> valid for DLP_kress
    if length(boundary)==1 && !(boundary[1] isa AbstractVector)
        crv=boundary[1]
        return _evaluate_points(solver,crv,k,1)
    end
    # Case 2: single composite boundary stored as a flat vector of segments
    # full_boundary = [seg1, seg2, ..., segm]
    # -> not valid for smooth DLP_kress
    if _is_single_composite_boundary(boundary)
        error("DLP_kress requires a single smooth closed curve. " *
            "This boundary is piecewise/composite; use DLP_kress_global_corners instead.")
    end
    # Case 3: multiple closed boundary components
    # full_boundary = [[outer...],[hole...],...]
    # -> not valid for DLP_kress
    error("DLP_kress supports exactly one smooth outer boundary component. " *
        "Geometries with holes or multiple closed components are not supported; use a CFIE solver instead.")
end

"""
    evaluate_points(solver::DLP_kress_global_corners, billiard, k)

High-level boundary discretization for general (possibly cornered) geometries.

# Behavior
- Extracts outer boundary
- If boundary is smooth:
    falls back to smooth `DLP_kress`
- If boundary has multiple segments:
    uses graded corner discretization

# Arguments
- `solver`: DLP_kress_global_corners solver
- `billiard`: geometry
- `k`: wavenumber

# Returns
- `BoundaryPointsCFIE`
"""
function evaluate_points(solver::DLP_kress_global_corners{T},billiard::Bi,k::T) where {T<:Real,Bi<:AbsBilliard}
    boundary=billiard.full_boundary
    isempty(boundary) && error("Boundary cannot be empty.")
    # Case 1: one smooth closed curve stored directly in full_boundary
    # full_boundary = [crv]
    # Example: a single PolarSegment-based smooth closed curve.
    # -> fallback to smooth DLP_kress discretization.
    if length(boundary)==1 && !(boundary[1] isa AbstractVector)
        crv=boundary[1]
        base_solver=DLP_kress(solver.pts_scaling_factor,solver.billiard;min_pts=solver.min_pts,eps=solver.eps,symmetry=solver.symmetry)
        return _evaluate_points(base_solver,crv,k,1)
    end
    # Case 2: Single composite outer boundary stored as a flat vector of segments:
    # full_boundary = [seg1, seg2, ..., segm]
    # Example: rectangle, stadium.
    # -> treat as one globally graded composite boundary.
    if _is_single_composite_boundary(boundary)
        comp=boundary
        return _evaluate_points(solver,comp,k,1)
    end
    # 3) Multiple closed boundary components:
    # full_boundary = [[outer_seg1,...],[hole_seg1,...],...]
    # Example: outer flower boundary with an inner polygon hole.
    # -> NOT supported here; DLP_kress_global_corners is for one outer boundary only.
    # Use a CFIE-type solver for multiply connected geometries.
    error("DLP_kress_global_corners supports exactly one outer boundary component.")
end

"""
    construct_dlp_matrix!(solver, D, pts, Rmat, G, k; multithreaded=true)

Assemble in-place the Kress-corrected Nyström matrix for the Helmholtz
double-layer operator on a single closed boundary component.

Mathematical meaning
--------------------
This function discretizes the boundary integral operator

    D(k)φ(x) = ∫_Γ ∂_{n_y} G_k(x,y) φ(y) ds_y

for the 2D Helmholtz Green function

    G_k(x,y) = (i/4) H_0^(1)(k |x-y|).

Differentiating with respect to the source normal n_y gives the kernel

    K_D(x,y;k) = (i k / 4) H_1^(1)(k r) * ((x-y)·n_y) / r,

where r = |x-y|.

On a smooth closed periodic curve, this kernel is weakly singular as x -> y.
Kress' method rewrites the same-component kernel as

    logarithmic singular part + smooth remainder.

More precisely, for nodes i,j on the same component, one isolates a coefficient
l1 multiplying the universal periodic logarithmic kernel, and a smooth part l2.
The discrete matrix entry is then

    D[i,j] = Rmat[i,j] * l1 + w_j * l2,   for i ≠ j,

where:
- Rmat is the Kress quadrature matrix for the periodic log singularity,
- w_j = pts.ws[j] is the base quadrature weight in parameter space,
- l1 carries the singular coefficient,
- l2 is the smooth remainder after subtracting l1 * logterm.

In this implementation the coefficients are

    αL1 = -k / (2π)
    αL2 = i k / 2

and for i ≠ j

    l1 = αL1 * inner * J_1(k r) / r
    l2 = αL2 * inner * H_1^(1)(k r) / r - l1 * logterm,

where:
- inner = G.inner[i,j] is the oriented source-normal numerator,
- J_1 is taken as real(H_1^(1)) in the code, matching the Kress split,
- logterm is the precomputed periodic logarithmic part (different for smooth vs graded cases) stored in G.logterm[i,j],
- r and 1/r come from G.R and G.invR.

# Inputs
- `solver`:
  Either `DLP_kress` or `DLP_kress_global_corners`. The former assumes a smooth
  periodic boundary; the latter uses globally graded nodes for piecewise-smooth
  boundaries with corners.
- `D`:
  Preallocated complex matrix to be overwritten with the assembled DLP matrix.
  Must be N×N where N = length(pts.xy).
- `pts`:
  Boundary discretization for one connected boundary component.
- `Rmat`:
  Precomputed Kress correction matrix. For smooth periodic boundaries this is
  the circulant Kress R-matrix; for globally graded corner meshes it is the
  corresponding corner-corrected logarithmic matrix.
- `G`:
  Geometric cache returned by `cfie_geom_cache`. It stores pairwise distances,
  inverse distances, logarithmic terms, normal inner products, curvature, etc.
- `k`:
  Real wavenumber.
- `multithreaded`:
  If true, the off-diagonal assembly loop may use threading when the matrix is
  large enough.

# Returns
- `D`, modified in place.
"""
function construct_dlp_matrix!(solver::Union{DLP_kress,DLP_kress_global_corners},D::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},Rmat::AbstractMatrix{T},G::CFIEGeomCache{T},k::T;multithreaded::Bool=true) where {T<:Real}
    αL1=-k*inv_two_pi
    αL2=Complex{T}(0,k/2)
    fill!(D,zero(Complex{T}))
    N=length(pts.xy)
    @inbounds for i in 1:N
        D[i,i]=Complex{T}(pts.ws[i]*G.kappa[i],zero(T))
    end
    @use_threads multithreading=(multithreaded && N>=32) for j in 2:N
        @inbounds for i in 1:j-1
            r=G.R[i,j]
            invr=G.invR[i,j]
            lt=G.logterm[i,j]
            inn_ij=G.inner[i,j]
            inn_ji=G.inner[j,i]
            _,h1=hankel_pair01(k*r)
            j1=real(h1)
            l1_ij=αL1*inn_ij*j1*invr
            l2_ij=αL2*inn_ij*h1*invr-l1_ij*lt
            D[i,j]=Rmat[i,j]*l1_ij+pts.ws[j]*l2_ij
            l1_ji=αL1*inn_ji*j1*invr
            l2_ji=αL2*inn_ji*h1*invr-l1_ji*lt
            D[j,i]=Rmat[j,i]*l1_ji+pts.ws[i]*l2_ji
        end
    end
    return D
end

# INTERNAL debugging version that returns the split matrices Dlog and Dsmooth separately.
function construct_dlp_split!(solver::Union{DLP_kress,DLP_kress_global_corners},Dlog::AbstractMatrix{Complex{T}},Dsmooth::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},Rmat::AbstractMatrix{T},G::CFIEGeomCache{T},parr::CFIEPanelArrays{T},k::T;multithreaded::Bool=true) where {T<:Real}
    αL1=-k*inv_two_pi
    αL2=Complex{T}(0,k/2)
    fill!(Dlog,zero(Complex{T}))
    fill!(Dsmooth,zero(Complex{T}))
    N=length(parr.X)
    @inbounds for i in 1:N
        Dsmooth[i,i]=Complex{T}(pts.ws[i]*G.kappa[i],zero(T))
    end
    @use_threads multithreading=(multithreaded && N>=32) for j in 2:N
        @inbounds for i in 1:j-1
            r=G.R[i,j]
            invr=G.invR[i,j]
            lt=G.logterm[i,j]
            inn_ij=G.inner[i,j]
            inn_ji=G.inner[j,i]
            _,h1=hankel_pair01(k*r)
            j1=real(h1)
            l1_ij=αL1*inn_ij*j1*invr
            l2_ij=αL2*inn_ij*h1*invr-l1_ij*lt
            Dlog[i,j]=Rmat[i,j]*l1_ij
            Dsmooth[i,j]=pts.ws[j]*l2_ij
            l1_ji=αL1*inn_ji*j1*invr
            l2_ji=αL2*inn_ji*h1*invr-l1_ji*lt
            Dlog[j,i]=Rmat[j,i]*l1_ji
            Dsmooth[j,i]=pts.ws[i]*l2_ji
        end
    end
    return Dlog,Dsmooth
end

"""
    construct_fredholm_matrix!(solver, F, pts, Rmat, G, parr, k; multithreaded=true)

Assemble in-place the Fredholm second-kind matrix associated with the
double-layer formulation:

    F(k) = I - D(k).

Assembly formula
----------------
If the raw double-layer matrix has entries D[i,j], then this function builds

    F[i,j] = δ_ij - D[i,j].

Using the Kress split, for i ≠ j this becomes

    F[i,j] = -( Rmat[i,j] * l1 + w_j * l2 ),

with the same l1 and l2 as in `construct_dlp_matrix!`:

    l1 = -(k / 2π) * inner * J_1(k r) / r
    l2 = (i k / 2) * inner * H_1^(1)(k r) / r - l1 * logterm.

# Inputs
- `solver`:
  `DLP_kress` or `DLP_kress_global_corners`.
- `F`:
  Preallocated complex N×N matrix to be overwritten.
- `pts`:
  Boundary discretization for one component.
- `Rmat`:
  Kress logarithmic quadrature matrix.
- `G`:
  Geometry cache.
- `parr`:
  Panel cache; mainly used for dimension consistency.
- `k`:
  Real wavenumber.
- `multithreaded`:
  Enables threaded off-diagonal assembly when N is large enough.

# Output
- `F`, modified in place.
"""
function construct_fredholm_matrix!(solver::Union{DLP_kress,DLP_kress_global_corners},F::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},Rmat::AbstractMatrix{T},G::CFIEGeomCache{T},parr::CFIEPanelArrays{T},k::T;multithreaded::Bool=true) where {T<:Real}
    αL1=-k*inv_two_pi
    αL2=Complex{T}(0,k/2)
    fill!(F,zero(Complex{T}))
    N=length(parr.X)
    @inbounds for i in 1:N
        F[i,i]=one(Complex{T})-Complex{T}(pts.ws[i]*G.kappa[i],zero(T))
    end
    @use_threads multithreading=(multithreaded && N>=32) for j in 2:N
        @inbounds for i in 1:j-1
            r=G.R[i,j]
            invr=G.invR[i,j]
            lt=G.logterm[i,j]
            inn_ij=G.inner[i,j]
            inn_ji=G.inner[j,i]
            _,h1=hankel_pair01(k*r)
            j1=real(h1)
            l1_ij=αL1*inn_ij*j1*invr
            l2_ij=αL2*inn_ij*h1*invr-l1_ij*lt
            F[i,j]=-(Rmat[i,j]*l1_ij+pts.ws[j]*l2_ij)
            l1_ji=αL1*inn_ji*j1*invr
            l2_ji=αL2*inn_ji*h1*invr-l1_ji*lt
            F[j,i]=-(Rmat[j,i]*l1_ji+pts.ws[i]*l2_ji)
        end
    end
    return F
end

"""
    construct_dlp_matrix_derivatives!(solver, D, D1, D2, pts, Rmat, G, parr, k; multithreaded=true)

Assemble the raw Kress-corrected DLP matrix together with its first and second
derivatives with respect to the wavenumber k.

Purpose
-------
This function is intended for methods that need local analytic information in k,
especially:
- Newton refinement of singular-value minima,
- EBIM / accelerated spectrum extraction,
- derivative-based determinant methods.

It computes, in place,

    D  = D(k)
    D1 = dD/dk
    D2 = d²D/dk².

# Inputs
- `solver`:
  `DLP_kress` or `DLP_kress_global_corners`.
- `D, D1, D2`:
  Preallocated N×N complex matrices.
- `pts`:
  Boundary discretization.
- `Rmat`:
  Kress correction matrix.
- `G`:
  Geometry cache.
- `parr`:
  Panel cache.
- `k`:
  Real wavenumber at which the matrix and derivatives are evaluated.
- `multithreaded`:
  Enables threaded off-diagonal assembly for large N.

# Output
- `(D, D1, D2)` after in-place assembly.
"""
function construct_dlp_matrix_derivatives!(solver::Union{DLP_kress,DLP_kress_global_corners},D::AbstractMatrix{Complex{T}},D1::AbstractMatrix{Complex{T}},D2::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},Rmat::AbstractMatrix{T},G::CFIEGeomCache{T},parr::CFIEPanelArrays{T},k::T;multithreaded::Bool=true) where {T<:Real}
    αL1=-k*inv_two_pi
    αL2=Complex{T}(0,k/2)
    fill!(D,zero(Complex{T}))
    fill!(D1,zero(Complex{T}))
    fill!(D2,zero(Complex{T}))
    N=length(parr.X)
    @inbounds for i in 1:N
        D[i,i]=Complex{T}(pts.ws[i]*G.kappa[i],zero(T))
        D1[i,i]=zero(Complex{T})
        D2[i,i]=zero(Complex{T})
    end
    @use_threads multithreading=(multithreaded && N>=32) for j in 2:N
        @inbounds for i in 1:j-1
            r=G.R[i,j]
            invr=G.invR[i,j]
            lt=G.logterm[i,j]
            inn_ij=G.inner[i,j]
            inn_ji=G.inner[j,i]
            kr=k*r
            h0,h1=hankel_pair01(kr)
            j0=real(h0)
            j1=real(h1)
            l1_ij=αL1*inn_ij*j1*invr
            l2_ij=αL2*inn_ij*h1*invr-l1_ij*lt
            D[i,j]=Rmat[i,j]*l1_ij+pts.ws[j]*l2_ij
            l1_ij_1=-(inn_ij*k*j0)*inv_two_pi
            l1_ij_2=(inn_ij*(k*r*j1-j0))*inv_two_pi
            l2_ij_1=(inn_ij*k*(lt*j0+im*pi*h0))*inv_two_pi
            l2_ij_2=(inn_ij*(lt*(j0-k*r*j1)+im*pi*(h0-k*r*h1)))*inv_two_pi
            D1[i,j]=Rmat[i,j]*l1_ij_1+pts.ws[j]*l2_ij_1
            D2[i,j]=Rmat[i,j]*l1_ij_2+pts.ws[j]*l2_ij_2
            l1_ji=αL1*inn_ji*j1*invr
            l2_ji=αL2*inn_ji*h1*invr-l1_ji*lt
            D[j,i]=Rmat[j,i]*l1_ji+pts.ws[i]*l2_ji
            l1_ji_1=-(inn_ji*k*j0)*inv_two_pi
            l1_ji_2=(inn_ji*(k*r*j1-j0))*inv_two_pi
            l2_ji_1=(inn_ji*k*(lt*j0+im*pi*h0))*inv_two_pi
            l2_ji_2=(inn_ji*(lt*(j0-k*r*j1)+im*pi*(h0-k*r*h1)))*inv_two_pi
            D1[j,i]=Rmat[j,i]*l1_ji_1+pts.ws[i]*l2_ji_1
            D2[j,i]=Rmat[j,i]*l1_ji_2+pts.ws[i]*l2_ji_2
        end
    end
    return D,D1,D2
end

"""
    construct_fredholm_matrix_derivatives!(solver, F, F1, F2, pts, Rmat, G, parr, k; multithreaded=true)

Assemble the Fredholm second-kind matrix

    F(k) = I - D(k)

together with its first and second derivatives with respect to k.
The function first assembles

    D  = D(k)
    D1 = dD/dk
    D2 = d²D/dk²

by calling `construct_dlp_matrix_derivatives!`, reusing the buffers `F`, `F1`,
and `F2`. It then converts these in place to the Fredholm quantities

    F  = I - D
    F1 = -D1
    F2 = -D2.

# Inputs
- `solver`:
  `DLP_kress` or `DLP_kress_global_corners`.
- `F, F1, F2`:
  Preallocated N×N complex matrices to receive the matrix and its first two
  k-derivatives.
- `pts`:
  Boundary discretization.
- `Rmat`:
  Kress correction matrix.
- `G`:
  Geometry cache.
- `parr`:
  Panel cache.
- `k`:
  Real wavenumber.
- `multithreaded`:
  Enables threaded off-diagonal assembly when beneficial.

# Output
- `(F, F1, F2)` after in-place assembly.
"""
function construct_fredholm_matrix_derivatives!(solver::Union{DLP_kress,DLP_kress_global_corners},F::AbstractMatrix{Complex{T}},F1::AbstractMatrix{Complex{T}},F2::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},Rmat::AbstractMatrix{T},G::CFIEGeomCache{T},parr::CFIEPanelArrays{T},k::T;multithreaded::Bool=true) where {T<:Real}
    construct_dlp_matrix_derivatives!(solver,F,F1,F2,pts,Rmat,G,parr,k;multithreaded=multithreaded)
    @inbounds for j in axes(F,2),i in axes(F,1)
        F[i,j]*=-1
        F1[i,j]*=-1
        F2[i,j]*=-1
    end
    @inbounds for i in axes(F,1)
        F[i,i]+=one(Complex{T})
    end
    return F,F1,F2
end

"""
    construct_matrices!(solver,A,pts,ws,k;multithreaded=true)
    construct_matrices!(solver,A,pts,Rmat,k;multithreaded=true)
    construct_matrices!(solver,A,A1,A2,pts,ws,k;multithreaded=true)
    construct_matrices!(solver,A,A1,A2,pts,Rmat,k;multithreaded=true)
    construct_matrices!(solver,basis,A,dA,ddA,pts,k;multithreaded=true)
    construct_matrices!(solver,basis,A,dA,ddA,pts,ws,k;multithreaded=true)

High-level assembly interface for the DLP-Kress Fredholm operator and, when
requested, its first two derivatives with respect to the wavenumber k.

Overview
--------
These methods are the public assembly layer sitting above the low-level DLP
routines. They all build the same operator family, namely the Fredholm
second-kind matrix

    A(k) = I - D(k),

where D(k) is the Kress-corrected Nyström discretization of the Helmholtz
double-layer boundary operator on a single closed component.

The role of `construct_matrices!` is to hide the details of:
- whether the boundary is smooth-periodic or globally corner-graded,
- whether the auxiliary geometry caches already exist,
- whether only A(k) is needed or also dA/dk and d²A/dk²,
- whether the calling code comes from an SVD/determinant path or from a
  basis-driven spectral function such as EBIM.

All methods overwrite preallocated matrices in place.

What is being assembled
-----------------------
For both `DLP_kress` and `DLP_kress_global_corners`, the object assembled here is

    A(k) = I - D(k),

with D(k) the Kress-corrected DLP discretization. Thus:
- diagonal entries are `1 - D_ii`,
- off-diagonal entries are `-D_ij`.

When derivatives are requested, the assembled matrices are

    A1(k) = dA/dk = - dD/dk
    A2(k) = d²A/dk² = - d²D/dk².

This is the operator you typically want for:
- smallest singular value sweeps,
- Fredholm determinant rootfinding,
- Newton refinement of spectral minima,
- EBIM / accelerated spectrum extraction.

# Inputs
Common positional inputs:
- `solver`:
  Either `DLP_kress` or `DLP_kress_global_corners`.
- `A`:
  Preallocated N×N complex matrix to receive the Fredholm matrix `I - D(k)`.
- `A1`, `A2`:
  Preallocated N×N complex matrices for the first and second derivatives with
  respect to k.
- `pts`:
  Boundary discretization, represented as `BoundaryPointsCFIE{T}` for one
  connected boundary component.
- `ws`:
  A `DLPKressWorkspace` containing the precomputed Kress correction matrix,
  geometry cache, panel arrays, and matrix size.
- `Rmat`:
  Precomputed Kress correction matrix. Used when a full workspace is not
  supplied.
- `basis`:
  A basis object, typically `AbstractHankelBasis()`, included to satisfy the
  generic interface used by accelerated spectral routines.
- `k`:
  Real wavenumber.
- `multithreaded::Bool=true`:
  Passed down to the low-level assembly kernels. If true, off-diagonal assembly
  may use threading for sufficiently large matrices.

# Outputs
- Single-matrix forms return `A`.
- Derivative forms return `(A,A1,A2)`.
- All matrices are modified in place.

Typical usage patterns
----------------------
1. Single matrix, cached workspace:
       pts = evaluate_points(solver,billiard,k0)
       ws  = build_dlp_kress_workspace(solver,pts)
       A   = Matrix{ComplexF64}(undef,ws.N,ws.N)
       construct_matrices!(solver,A,pts,ws,k0)

2. Matrix plus derivatives for Newton/EBIM:
       A   = Matrix{ComplexF64}(undef,ws.N,ws.N)
       dA  = similar(A)
       ddA = similar(A)
       construct_matrices!(solver,A,dA,ddA,pts,ws,k0)
"""
function construct_matrices!(solver::Union{DLP_kress,DLP_kress_global_corners},A::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},ws::DLPKressWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}
    construct_fredholm_matrix!(solver,A,pts,ws.Rmat,ws.G,ws.parr,k;multithreaded=multithreaded)
end

function construct_matrices!(solver::Union{DLP_kress,DLP_kress_global_corners},A::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},Rmat::AbstractMatrix{T},k::T;multithreaded::Bool=true) where {T<:Real}
    G=_is_dlp_kress_graded(solver) ? cfie_geom_cache(pts,true) : cfie_geom_cache(pts,false)
    parr=_panel_arrays_cache(pts)
    construct_fredholm_matrix!(solver,A,pts,Rmat,G,parr,k;multithreaded=multithreaded)
end

function construct_matrices!(solver::Union{DLP_kress,DLP_kress_global_corners},A::AbstractMatrix{Complex{T}},A1::AbstractMatrix{Complex{T}},A2::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},ws::DLPKressWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}
    construct_fredholm_matrix_derivatives!(solver,A,A1,A2,pts,ws.Rmat,ws.G,ws.parr,k;multithreaded=multithreaded)
end

function construct_matrices!(solver::Union{DLP_kress,DLP_kress_global_corners},A::AbstractMatrix{Complex{T}},A1::AbstractMatrix{Complex{T}},A2::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},Rmat::AbstractMatrix{T},k::T;multithreaded::Bool=true) where {T<:Real}
    G=_is_dlp_kress_graded(solver) ? cfie_geom_cache(pts,true) : cfie_geom_cache(pts,false)
    parr=_panel_arrays_cache(pts)
    construct_fredholm_matrix_derivatives!(solver,A,A1,A2,pts,Rmat,G,parr,k;multithreaded=multithreaded)
end

function construct_matrices!(solver::Union{DLP_kress,DLP_kress_global_corners},basis::AbstractHankelBasis,A::AbstractMatrix{Complex{T}},dA::AbstractMatrix{Complex{T}},ddA::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},k::T;multithreaded::Bool=true) where {T<:Real}
    ws=build_dlp_kress_workspace(solver,pts)
    construct_matrices!(solver,A,dA,ddA,pts,ws,k;multithreaded=multithreaded)
    return A,dA,ddA
end

function construct_matrices!(solver::Union{DLP_kress,DLP_kress_global_corners},basis::AbstractHankelBasis,A::AbstractMatrix{Complex{T}},dA::AbstractMatrix{Complex{T}},ddA::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},ws::DLPKressWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}
    construct_matrices!(solver,A,dA,ddA,pts,ws,k;multithreaded=multithreaded)
    return A,dA,ddA
end

"""
    solve(solver,basis,pts,k;multithreaded=true,use_krylov=true,which=:det)
    solve(solver,basis,pts,ws,k;multithreaded=true,use_krylov=true,which=:det_argmin)
    solve(solver,basis,A,pts,k,Rmat;multithreaded=true,use_krylov=true,which=:det_argmin)
    solve(solver,basis,A,pts,ws,k;multithreaded=true,use_krylov=true,which=:det_argmin)

High-level scalar solver interface for the DLP-Kress Fredholm formulation.

Purpose
-------
These methods assemble the Fredholm matrix

    A(k) = I - D(k)

for the DLP-Kress discretization and then reduce that matrix to a single scalar
quantity, depending on the requested mode `which`.

Overload structure
------------------
1. `solve(solver,basis,pts,k; ...)`
   --------------------------------
   Minimal high-level form.
   - Builds the Kress correction matrix internally,
   - allocates a fresh A,
   - assembles A(k),
   - returns the chosen scalar diagnostic.

2. `solve(solver,basis,pts,ws,k; ...)`
   -----------------------------------
   Cached-workspace form.
   - Reuses `ws`,
   - allocates a fresh A,
   - assembles A(k),
   - returns the scalar diagnostic.

   This is the preferred form when the same discretization is reused over many k.

3. `solve(solver,basis,A,pts,k,Rmat; ...)`
   ---------------------------------------
   Reuse-buffer form with preallocated matrix A and cached Rmat.
   Useful in sweeps when matrix allocation should be avoided but a full
   workspace is not being stored.

4. `solve(solver,basis,A,pts,ws,k; ...)`
   -------------------------------------
   Fully cached production form.
   - Reuses both the matrix buffer A and the workspace ws.
   This is generally the fastest variant in long sweeps.

Meaning of `which`
------------------
The keyword `which` is forwarded to the generic backend macro
`@svd_or_det_solve`, so the exact supported options depend on that backend.
In the current usage pattern, the important modes are:

- `which = :svd`
  Return the smallest singular value of A(k).
  This is the most robust quantity for “tension” or spectral proximity.

- `which = :det`
  Return the determinant det(A(k)).
  This is useful for Fredholm determinant rootfinding.

- `which = :det_argmin`
  Return the scalar used by determinant-based minimization logic.
  In practice this is often the most convenient mode for local searches.

# Inputs
- `solver`:
  `DLP_kress` or `DLP_kress_global_corners`.
- `basis`:
  An abstract basis object. For DLP-Kress this is mainly an interface placeholder
  so that the solver fits the same API as the other spectral methods.
- `pts`:
  Boundary discretization for one connected boundary component.
- `ws`:
  Optional `DLPKressWorkspace` for cached assembly.
- `A`:
  Optional preallocated complex matrix buffer.
- `Rmat`:
  Optional precomputed Kress correction matrix.
- `k`:
  Real wavenumber at which the spectral diagnostic is evaluated.

# Keyword arguments
- `multithreaded::Bool=true`
  Passed to matrix assembly.
- `use_krylov::Bool=true`
  Forwarded to the scalar-reduction backend. Depending on the selected `which`,
  this may control whether Krylov-based or dense-LAPACK-based reduction is used.
- `which::Symbol`
  Selects the scalar spectral diagnostic to return.

# Output
Returns a scalar quantity measuring spectral proximity at k.
Its precise meaning depends on `which`:
- for `:svd`, typically a real nonnegative smallest singular value,
- for `:det`, typically a complex determinant,
- for `:det_argmin`, the backend-specific scalar used in determinant minimization.
"""
function solve(solver::Union{DLP_kress,DLP_kress_global_corners},basis::Ba,pts::BoundaryPointsCFIE{T},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det) where {T<:Real,Ba<:AbsBasis}
    N=length(pts.xy)
    A=Matrix{Complex{T}}(undef,N,N)
    @blas_1 Rmat=build_Rmat_dlp_kress(solver,pts)
    @blas_1 construct_matrices!(solver,A,pts,Rmat,k;multithreaded=multithreaded)
    @svd_or_det_solve A use_krylov which MAX_BLAS_THREADS
end

function solve(solver::Union{DLP_kress,DLP_kress_global_corners},basis::Ba,pts::BoundaryPointsCFIE{T},ws::DLPKressWorkspace{T},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det_argmin) where {T<:Real,Ba<:AbsBasis}
    A=Matrix{Complex{T}}(undef,ws.N,ws.N)
    @blas_1 construct_matrices!(solver,A,pts,ws,k;multithreaded=multithreaded)
    @svd_or_det_solve A use_krylov which MAX_BLAS_THREADS
end

function solve(solver::Union{DLP_kress,DLP_kress_global_corners},basis::Ba,A::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},k,Rmat::AbstractMatrix{T};multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det_argmin) where {T<:Real,Ba<:AbsBasis}
    @blas_1 construct_matrices!(solver,A,pts,Rmat,k;multithreaded=multithreaded)
    @svd_or_det_solve A use_krylov which MAX_BLAS_THREADS
end

function solve(solver::Union{DLP_kress,DLP_kress_global_corners},basis::Ba,A::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},ws::DLPKressWorkspace{T},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det_argmin) where {T<:Real,Ba<:AbsBasis}
    @blas_1 construct_matrices!(solver,A,pts,ws,k;multithreaded=multithreaded)
    @svd_or_det_solve A use_krylov which MAX_BLAS_THREADS
end

"""
    solve_vect(solver,basis,A,pts,k,Rmat;multithreaded=true)
    solve_vect(solver,basis,A,pts,ws,k;multithreaded=true)
    solve_vect(solver,basis,pts,ws,k;multithreaded=true)
    solve_vect(solver,basis,pts,k;multithreaded=true)
    solve_vect(solver,basis,ks::Vector{T};multithreaded=true)

Compute the smallest singular value of the DLP-Kress Fredholm matrix together
with the associated right singular vector. 
For a given wavenumber k, these routines compute the singular value decomposition

    A = U Σ V*.

They then select the smallest singular value

    μ = σ_min(A),

and return the corresponding right singular vector

    u_μ = v_min,

implemented in the code as the conjugate of the appropriate row of `Vt`.

Overload structure
------------------
1. `solve_vect(solver,basis,A,pts,k,Rmat; ...)`
   --------------------------------------------
   Reuses both a matrix buffer A and a cached Kress correction matrix.

2. `solve_vect(solver,basis,A,pts,ws,k; ...)`
   ------------------------------------------
   Reuses a matrix buffer and a full cached workspace. This is usually the best
   high-performance form.

3. `solve_vect(solver,basis,pts,ws,k; ...)`
   ----------------------------------------
   Allocates a fresh A but reuses the workspace.

4. `solve_vect(solver,basis,pts,k; ...)`
   -------------------------------------
   Fully self-contained form: builds Rmat, allocates A, assembles, computes SVD.

5. `solve_vect(solver,basis,ks::Vector{T}; ...)`
   ---------------------------------------------
   Convenience batched form for multiple wavenumbers.
   For each k in `ks`, it:
   - evaluates the boundary discretization,
   - computes the smallest singular vector,
   - stores the vector and the boundary points.

   The return is:
       (us_all, pts_all)

   where `us_all[i]` is the singular vector at `ks[i]`, and `pts_all[i]` is the
   corresponding discretization.

# Inputs
- `solver`:
  `DLP_kress` or `DLP_kress_global_corners`.
- `basis`:
  Abstract basis object included for API compatibility.
- `A`:
  Optional preallocated complex matrix buffer.
- `pts`:
  Boundary discretization.
- `ws`:
  Optional `DLPKressWorkspace`.
- `Rmat`:
  Optional Kress correction matrix.
- `k`:
  Real wavenumber.
- `ks`:
  Vector of real wavenumbers in the batched overload.

# Keyword arguments
- `multithreaded::Bool=true`
  Controls matrix assembly threading.

# Outputs
Single-k overloads return:
- `mu`:
  smallest singular value of the Fredholm matrix,
- `u_mu`:
  corresponding right singular vector.
Vector-of-k overload returns:
- `us_all`:
  vector of singular vectors, one per wavenumber,
- `pts_all`:
  vector of boundary discretizations used at those wavenumbers.
"""
function solve_vect(solver::Union{DLP_kress,DLP_kress_global_corners},basis::Ba,A::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},k,Rmat::AbstractMatrix{T};multithreaded::Bool=true) where {T<:Real,Ba<:AbsBasis}
    @blas_1 construct_matrices!(solver,A,pts,Rmat,k;multithreaded=multithreaded)
    @blas_multi_then_1 MAX_BLAS_THREADS _,S,Vt=LAPACK.gesvd!('A','A',A)
    idx=findmin(S)[2]
    mu=S[idx]
    u_mu=conj.(Vt[idx,:])
    return mu,u_mu
end

function solve_vect(solver::Union{DLP_kress,DLP_kress_global_corners},basis::Ba,A::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},ws::DLPKressWorkspace{T},k;multithreaded::Bool=true) where {T<:Real,Ba<:AbsBasis}
    @blas_1 construct_matrices!(solver,A,pts,ws,k;multithreaded=multithreaded)
    @blas_multi_then_1 MAX_BLAS_THREADS _,S,Vt=LAPACK.gesvd!('A','A',A)
    idx=findmin(S)[2]
    mu=S[idx]
    u_mu=conj.(Vt[idx,:])
    return mu,u_mu
end

function solve_vect(solver::Union{DLP_kress,DLP_kress_global_corners},basis::Ba,pts::BoundaryPointsCFIE{T},ws::DLPKressWorkspace{T},k;multithreaded::Bool=true) where {T<:Real,Ba<:AbsBasis}
    A=Matrix{Complex{T}}(undef,ws.N,ws.N)
    @blas_1 construct_matrices!(solver,A,pts,ws,k;multithreaded=multithreaded)
    @blas_multi_then_1 MAX_BLAS_THREADS _,S,Vt=LAPACK.gesvd!('A','A',A)
    idx=findmin(S)[2]
    mu=S[idx]
    u_mu=conj.(Vt[idx,:])
    return mu,u_mu
end

function solve_vect(solver::Union{DLP_kress,DLP_kress_global_corners},basis::Ba,pts::BoundaryPointsCFIE{T},k;multithreaded::Bool=true) where {T<:Real,Ba<:AbsBasis}
    N=length(pts.xy)
    A=Matrix{Complex{T}}(undef,N,N)
    @blas_1 Rmat=build_Rmat_dlp_kress(solver,pts)
    @blas_1 construct_matrices!(solver,A,pts,Rmat,k;multithreaded=multithreaded)
    @blas_multi_then_1 MAX_BLAS_THREADS _,S,Vt=LAPACK.gesvd!('A','A',A)
    idx=findmin(S)[2]
    mu=S[idx]
    u_mu=conj.(Vt[idx,:])
    return mu,u_mu
end

function solve_vect(solver::Union{DLP_kress,DLP_kress_global_corners},basis::Ba,ks::Vector{T};multithreaded::Bool=true) where {T<:Real,Ba<:AbsBasis}
    us_all=Vector{Vector{eltype(ks)}}(undef,length(ks))
    pts_all=Vector{BoundaryPointsCFIE{eltype(ks)}}(undef,length(ks))
    for i in eachindex(ks)
        pts=evaluate_points(solver,solver.billiard,ks[i])
        _,u=solve_vect(solver,basis,pts,ks[i];multithreaded=multithreaded)
        us_all[i]=u
        pts_all[i]=pts
    end
    return us_all,pts_all
end
# INTERNAL - for checking allocation patterns and execution time of the single-k solve variants. Not intended for public use.
function solve_INFO(solver::Union{DLP_kress,DLP_kress_global_corners},basis::Ba,pts::BoundaryPointsCFIE{T},ws::DLPKressWorkspace{T},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det_argmin) where {T<:Real,Ba<:AbsBasis}
    A=Matrix{Complex{T}}(undef,ws.N,ws.N)
    t0=time()
    @info "Building boundary operator A from cached DLP-Kress workspace..."
    @blas_1 construct_matrices!(solver,A,pts,ws,k;multithreaded=multithreaded)
    t1=time()
    cA=cond(A)
    @info "Condition number of A: $(round(cA;sigdigits=4))"
    t2=time()
    s=@svd_or_det_solve A use_krylov which MAX_BLAS_THREADS
    t3=time()
    build_A=t1-t0
    svd_time=t3-t2
    total=build_A+svd_time
    println("────────── SOLVE_INFO SUMMARY ──────────")
    println("A-matrix build: ",100*build_A/total," %")
    println("SVD: ",100*svd_time/total," %")
    println("(total: ",total," s)")
    println("────────────────────────────────────────")
    return s
end

# INTERNAL debugging function to check the consistency of the DLP Kress split.
function debug_dlp_split_error(solver::Union{DLP_kress,DLP_kress_global_corners},pts::BoundaryPointsCFIE{T},ws::DLPKressWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}
    D=Matrix{Complex{T}}(undef,ws.N,ws.N)
    Dlog=Matrix{Complex{T}}(undef,ws.N,ws.N)
    Dsmooth=Matrix{Complex{T}}(undef,ws.N,ws.N)
    construct_dlp_matrix!(solver,D,pts,ws.Rmat,ws.G,ws.parr,k;multithreaded=multithreaded)
    construct_dlp_split!(solver,Dlog,Dsmooth,pts,ws.Rmat,ws.G,ws.parr,k;multithreaded=multithreaded)
    Δ=D-(Dlog+Dsmooth)
    return norm(Δ),maximum(abs.(Δ))
end
# INTERNAL debugging function that builds the workspace internally and checks the split error.
function debug_dlp_split_error(solver::Union{DLP_kress,DLP_kress_global_corners},pts::BoundaryPointsCFIE{T},k::T;multithreaded::Bool=true) where {T<:Real}
    ws=build_dlp_kress_workspace(solver,pts)
    debug_dlp_split_error(solver,pts,ws,k;multithreaded=multithreaded)
end