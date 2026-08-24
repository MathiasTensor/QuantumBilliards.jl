# Useful reading:
# - Kress, R., Boundary integral equations in time-harmonic acoustic scattering. Math. Comput. Modelling 15 (1991), 229-243.
# - Barnett, A. H. / Betcke, T., mpspack DLP implementation.
# - Zhao, L. / Barnett, A., Robust and efficient solution of the drum problem via Nyström approximation of the Fredholm determinant.

const two_pi=2*pi
const inv_two_pi=inv(two_pi)

"""
    DLPKressWorkspace{T,M}

Workspace cache for repeated DLP-Kress matrix assembly on a fixed boundary
discretization.

## Description
This object stores all geometry-dependent data that can be reused across many
wavenumbers `k`. For repeated assembly on the same boundary nodes, the expensive
pairwise geometric quantities are therefore computed only once, while the
k-dependent special-function evaluations are updated for each new wavenumber.

The workspace contains the Kress logarithmic correction matrix, the pairwise
geometry cache, flat panel arrays and the matrix dimension.

## Attributes
* `Rmat`: Dense real Kress correction matrix for the logarithmic singular part.
* `G`: Geometry cache containing pairwise distances, inverse distances,
  logarithmic terms, curvature data and oriented interaction factors.
* `parr`: Flat coordinate and tangent arrays used for efficient indexed access.
* `N`: Number of boundary points and matrix dimension.

## Notes
For determinant scans, k-sweeps, Newton refinement and repeated spectral
assembly, rebuilding the geometric quantities for every `k` would be wasteful.
A workspace should therefore be reused whenever the boundary discretization
remains fixed.
"""
struct DLPKressWorkspace{T<:Real,M<:AbstractMatrix{T}}
    Rmat::M
    G::BoundaryGeomCache{T}
    parr::BoundaryPanelArrays{T}
    N::Int
end

"""
    DLPKressReducedWorkspace{T,M}

Reduced workspace for symmetry-desymmetrized DLP-Kress assembly.

## Description
The boundary points remain the complete Kress boundary discretization, so the
logarithmic splitting and all singular quadrature data are defined on the full
periodic grid.

The output operator is assembled only on the fundamental index set `Ifund`.
Missing symmetry copies of each source point are added as regular image-kernel
contributions. This avoids allocating the full complex Fredholm matrix while
preserving the full-grid Kress treatment of the singular same-copy
interaction.

## Attributes
* `full`: Complete DLP-Kress workspace on the full periodic boundary.
* `Ifund`: Full-grid indices belonging to the fundamental boundary.
* `full_to_fund`: Mapping from full-grid indices to fundamental indices.
* `full_to_scale`: Symmetry factors mapping full-grid nodes to fundamental nodes.
* `fund_to_full`: Full-grid symmetry orbit associated with each fundamental node.
* `fund_to_scale`: Symmetry factors associated with the corresponding orbit.
* `xs`: Full-grid x coordinates.
* `ys`: Full-grid y coordinates.
* `nx`: Full-grid outward-normal x components.
* `ny`: Full-grid outward-normal y components.
* `speed`: Full-grid parametrization speeds.
* `m`: Dimension of the reduced matrix.
"""
struct DLPKressReducedWorkspace{T<:Real,M<:AbstractMatrix{T}}
    full::DLPKressWorkspace{T,M}
    Ifund::Vector{Int}
    full_to_fund::Vector{Int}
    full_to_scale::Vector{Complex{T}}
    fund_to_full::Vector{Vector{Int}}
    fund_to_scale::Vector{Vector{Complex{T}}}
    xs::Vector{T}
    ys::Vector{T}
    nx::Vector{T}
    ny::Vector{T}
    speed::Vector{T}
    m::Int
end

"""
    DLP_kress{T,Bi,Sym} <: SweepSolver

Solver for the smooth periodic Kress-corrected double-layer Fredholm
formulation.

## Description
`DLP_kress` implements the periodic Kress treatment of the Helmholtz
double-layer operator on a single smooth closed boundary.

The associated Fredholm matrix is

    F(k)=I-D(k),

where `D(k)` is the Kress-corrected Nyström discretization of the interior
Helmholtz double-layer operator.

The singular same-boundary interaction is split into a universal periodic
logarithmic kernel

    log(4sin²((t-s)/2))

and a smooth remainder. The singular part is integrated through the precomputed
Kress correction matrix.

## Attributes
* `sampler`: Placeholder sampling descriptor retained for the common solver API.
* `pts_scaling_factor`: Boundary-resolution scaling factors.
* `dim_scaling_factor`: Compatibility field used by generic refinement code.
* `eps`: Numerical tolerance.
* `min_dim`: Minimum dimension compatibility field.
* `min_pts`: Minimum number of boundary points.
* `billiard`: Underlying billiard geometry.
* `symmetry`: Optional reflection or rotation symmetry descriptor.

## Notes
The nominal number of boundary points is

    N ≈ k*L*b/(2π),

where `L` is the boundary length and `b=pts_scaling_factor[1]`.

This solver supports exactly one smooth closed outer boundary. Piecewise smooth
boundaries with corners should use [`DLP_kress_global_corners`](@ref).
Multiply-connected geometries require a formulation supporting multiple
boundary components.
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

Solver for the globally graded Kress-corrected double-layer Fredholm
formulation on a piecewise smooth outer boundary.

## Description
This is the corner-capable counterpart of [`DLP_kress`](@ref). It treats a
single outer boundary represented by one or more joined smooth curve segments.

For boundaries containing true corners, the global periodic parameter is
transformed through a Kress grading map that clusters nodes near the corner
locations. The associated Fredholm matrix remains

    F(k)=I-D(k),

but the geometry is evaluated on the graded parameterization.

If the supplied composite boundary contains no true corners, the implementation
automatically falls back to an ungraded smooth-composite discretization.

## Attributes
* `sampler`: Placeholder sampling descriptor retained for the common solver API.
* `pts_scaling_factor`: Boundary-resolution scaling factors.
* `dim_scaling_factor`: Compatibility field used by generic refinement code.
* `eps`: Numerical tolerance.
* `min_dim`: Minimum dimension compatibility field.
* `min_pts`: Minimum number of boundary points.
* `billiard`: Underlying billiard geometry.
* `symmetry`: Optional reflection or rotation symmetry descriptor.
* `kressq`: Order of the Kress grading map.
* `min_t_spacing`: Minimum allowed spacing after grading.

## Notes
This solver supports one outer closed boundary, which may consist of several
smooth segments. Multiply-connected geometries are not supported by this
formulation.
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
    min_t_spacing::Real
end

function DLP_kress(pts_scaling_factor::Union{T,Vector{T}},billiard::Bi;min_pts=20,eps=T(1e-15),symmetry::Union{Nothing,AbsSymmetry}=nothing) where {T<:Real,Bi<:AbsBilliard}
    bs=pts_scaling_factor isa T ? [pts_scaling_factor] : pts_scaling_factor
    sampler=[LinearNodes()]
    Sym=typeof(symmetry)
    return DLP_kress{T,Bi,Sym}(sampler,bs,bs[1],eps,min_pts,min_pts,billiard,symmetry)
end

function DLP_kress_global_corners(pts_scaling_factor::Union{T,Vector{T}},billiard::Bi;min_pts=20,eps=T(1e-15),symmetry::Union{Nothing,AbsSymmetry}=nothing,kressq=4,min_t_spacing=1e-12) where {T<:Real,Bi<:AbsBilliard}
    bs=pts_scaling_factor isa T ? [pts_scaling_factor] : pts_scaling_factor
    sampler=[LinearNodes()]
    Sym=typeof(symmetry)
    return DLP_kress_global_corners{T,Bi,Sym}(sampler,bs,bs[1],eps,min_pts,min_pts,billiard,symmetry,kressq,min_t_spacing)
end

# This flag refers to the nontrivial parameter grading used for corner treatment.
@inline _is_dlp_kress_graded(::DLP_kress,pts::BoundaryPoints)=false
@inline _is_dlp_kress_graded(::DLP_kress_global_corners,pts::BoundaryPoints)=_is_nontrivial_grading(pts)

@inline function _is_nontrivial_grading(pts::BoundaryPoints{T}) where {T<:Real}
    length(pts.ws_der)==length(pts)||return false
    return maximum(abs.(pts.ws_der.-one(T)))>sqrt(eps(T))
end

# Use a reduced symmetry-image assembly whenever a nontrivial symmetry is active.
@inline _dlp_kress_use_reduced(solver::Union{DLP_kress,DLP_kress_global_corners})=!isnothing(solver.symmetry)

"""
    build_Rmat_dlp_kress(solver::DLP_kress,pts::BoundaryPoints{T}) where {T<:Real} → Rmat::Matrix{T}

Builds the periodic Kress logarithmic correction matrix for a smooth closed
boundary.

## Description
For a smooth periodic discretization, the logarithmically singular part of the
double-layer kernel is represented by the standard periodic Kress quadrature
matrix.

The matrix depends only on the periodic node count and may therefore be cached
and reused for all wavenumbers evaluated on the same discretization.

## Arguments
* `solver`: Smooth periodic DLP-Kress solver.
* `pts`: Smooth periodic boundary discretization.

## Returns
* `Rmat`: Dense `N×N` Kress logarithmic correction matrix.
"""
function build_Rmat_dlp_kress(solver::DLP_kress,pts::BoundaryPoints{T}) where {T<:Real}
    N=length(pts)
    Rmat=zeros(T,N,N)
    kress_R!(Rmat)
    return Rmat
end

"""
    build_Rmat_dlp_kress(solver::DLP_kress_global_corners,pts::BoundaryPoints{T}) where {T<:Real} → Rmat::Matrix{T}

Builds the Kress logarithmic correction matrix for the global-corner solver.

## Description
If the boundary discretization contains a nontrivial grading map,
[`kress_R_corner!`](@ref) is used. If no actual grading is present, the standard
periodic correction matrix is sufficient.

## Arguments
* `solver`: Global-corner DLP-Kress solver.
* `pts`: Boundary discretization.

## Returns
* `Rmat`: Dense `N×N` Kress logarithmic correction matrix.
"""
function build_Rmat_dlp_kress(solver::DLP_kress_global_corners,pts::BoundaryPoints{T}) where {T<:Real}
    N=length(pts)
    Rmat=zeros(T,N,N)
    _is_nontrivial_grading(pts) ? kress_R_corner!(Rmat) : kress_R!(Rmat)
    return Rmat
end

"""
    _evaluate_points(solver::DLP_kress{T},crv::C,k::T,idx::Int) where {T<:Real,C<:AbsCurve} → pts::BoundaryPoints{T}

Constructs the periodic DLP-Kress discretization of a single smooth closed
curve.

## Description
The computational Kress variable is the uniform periodic parameter

    σ_j=2π(j-1/2)/N,

as generated by `s_mid`.

The physical curve parameter is

    u=σ/(2π).

If `γ(u)` denotes the underlying curve parameterization, derivatives with
respect to the computational variable satisfy

    γ_σ=γ_u/(2π),
    γ_σσ=γ_uu/(2π)².

The node count is chosen approximately as

    N ≈ k*L*b/(2π),

and adjusted to satisfy both the periodic Kress requirements and any active
symmetry.

## Arguments
* `solver`: Smooth DLP-Kress solver.
* `crv`: Smooth closed curve.
* `k`: Wavenumber controlling the discretization density.
* `idx`: Boundary-component label.

## Returns
* `pts`: A [`BoundaryPoints`](@ref) instance containing coordinates, normals,
  tangent data, parameter nodes and quadrature weights.
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
        elseif sym isa Reflection
            needed=lcm(needed,4)
        end
    end
    remN=mod(N,needed)
    remN!=0&&(N+=needed-remN)
    ts=[s_mid(j,N) for j in 1:N]
    ts_rescaled=ts./two_pi
    xy=curve(crv,ts_rescaled)
    tangent_1st=tangent(crv,ts_rescaled)./two_pi
    tangent_2nd=tangent_2(crv,ts_rescaled)./(two_pi^2)
    ss=arc_length(crv,ts_rescaled)
    ds=diff(ss)
    append!(ds,L+ss[1]-ss[end])
    ws=fill(T(two_pi/N),N)
    ws_der=ones(T,N)
    z=SVector(zero(T),zero(T))
    return BoundaryPoints(xy,tangent_1st,tangent_2nd,ts,copy(ts),ws,ws_der,ds,idx,true,z,z,z,z)
end

"""
    _evaluate_points_smooth_composite(solver::DLP_kress_global_corners{T},comp::Vector{C},k::T,idx::Int) where {T<:Real,C<:AbsCurve} → pts::BoundaryPoints{T}

Constructs an ungraded periodic Nyström discretization of a smooth composite
boundary.

## Description
This is the fallback path used by [`DLP_kress_global_corners`](@ref) when the
boundary is represented by several curve pieces but no true corners are
detected.

A global periodic variable

    t∈[0,2π)

is mapped to the corresponding segment and local segment parameter by
[`_eval_composite_geom_global_t`](@ref).

Since no grading is active, the computational and physical global parameters
coincide.

## Arguments
* `solver`: Global-corner DLP-Kress solver.
* `comp`: Smooth curve pieces forming one closed composite component.
* `k`: Wavenumber controlling the discretization density.
* `idx`: Boundary-component label.

## Returns
* `pts`: Ungraded periodic [`BoundaryPoints`](@ref) discretization.
"""
function _evaluate_points_smooth_composite(solver::DLP_kress_global_corners{T},comp::Vector{C},k::T,idx::Int) where {T<:Real,C<:AbsCurve}
    _,_,Ltot=component_lengths(comp)
    bs=solver.pts_scaling_factor
    N=max(solver.min_pts,round(Int,k*Ltot*bs[1]/two_pi))
    needed=2
    if !isnothing(solver.symmetry)
        sym=solver.symmetry
        if sym isa Rotation
            needed=lcm(needed,sym.n)
        elseif sym isa Reflection
            needed=lcm(needed,4)
        end
    end
    remN=mod(N,needed)
    remN!=0&&(N+=needed-remN)
    ts=[s_mid(j,N) for j in 1:N]
    xy=Vector{SVector{2,T}}(undef,N)
    tangent_1st=Vector{SVector{2,T}}(undef,N)
    tangent_2nd=Vector{SVector{2,T}}(undef,N)
    h=T(two_pi)/T(N)
    ds=Vector{T}(undef,N)
    @inbounds for i in 1:N
        q,γt,γtt=_eval_composite_geom_global_t(T,comp,ts[i])
        xy[i]=q
        tangent_1st[i]=γt
        tangent_2nd[i]=γtt
        ds[i]=hypot(γt[1],γt[2])*h
    end
    ws=fill(h,N)
    ws_der=ones(T,N)
    z=SVector(zero(T),zero(T))
    return BoundaryPoints(xy,tangent_1st,tangent_2nd,ts,copy(ts),ws,ws_der,ds,idx,true,z,z,z,z)
end

"""
    _evaluate_points(solver::DLP_kress_global_corners{T},comp::Vector{C},k::T,idx::Int) where {T<:Real,C<:AbsCurve} → pts::BoundaryPoints{T}

Constructs a globally graded Nyström discretization for a piecewise smooth
closed boundary.

## Description
True corner locations are first detected from the tangent discontinuities
between neighboring curve segments.

If no true corners are found, the function delegates to
[`_evaluate_points_smooth_composite`](@ref).

Otherwise a global periodic grading map

    t=t(σ)

is generated by `multi_kress_graded_nodes_data`. The map returns the physical
parameter `t`, its first derivative and its second derivative.

If

    γ=γ(t),

then the transformed derivatives are

    γ_σ=γ_t t_σ,

and

    γ_σσ=γ_tt(t_σ)²+γ_t t_σσ.

The resulting speed already contains the grading Jacobian, so the arc-length
quadrature elements are

    ds_j=|γ_σ(σ_j)|h,

where `h=2π/N`.

## Arguments
* `solver`: Global-corner DLP-Kress solver.
* `comp`: Curve segments forming one closed component.
* `k`: Wavenumber controlling the discretization density.
* `idx`: Boundary-component label.

## Returns
* `pts`: Globally graded [`BoundaryPoints`](@ref) discretization.
"""
function _evaluate_points(solver::DLP_kress_global_corners{T},comp::Vector{C},k::T,idx::Int) where {T<:Real,C<:AbsCurve}
    corners=_component_corner_locations(T,comp)
    isempty(corners)&&return _evaluate_points_smooth_composite(solver,comp,k,idx)
    _,_,Ltot=component_lengths(comp)
    bs=solver.pts_scaling_factor
    N=max(solver.min_pts,round(Int,k*Ltot*bs[1]/two_pi))
    needed=1
    if !isnothing(solver.symmetry)
        sym=solver.symmetry
        if sym isa Rotation
            needed=lcm(needed,sym.n)
        elseif sym isa Reflection
            needed=lcm(needed,4)
        end
    end
    remN=mod(N,needed)
    remN!=0&&(N+=needed-remN)
    σ,tmap,jac,jac2,_=multi_kress_graded_nodes_data(T,N,corners;q=solver.kressq,minsep_tol=solver.min_t_spacing)
    xy=Vector{SVector{2,T}}(undef,N)
    tangent_1st=Vector{SVector{2,T}}(undef,N)
    tangent_2nd=Vector{SVector{2,T}}(undef,N)
    @inbounds for i in 1:N
        q,γt,γtt=_eval_composite_geom_global_t(T,comp,tmap[i])
        xy[i]=q
        tangent_1st[i]=γt*jac[i]
        tangent_2nd[i]=γtt*(jac[i]^2)+γt*jac2[i]
    end
    h=T(two_pi)/T(N)
    ds=Vector{T}(undef,N)
    @inbounds for i in 1:N
        ds[i]=hypot(tangent_1st[i][1],tangent_1st[i][2])*h
    end
    ws=fill(h,N)
    z=SVector(zero(T),zero(T))
    return BoundaryPoints(xy,tangent_1st,tangent_2nd,σ,tmap,ws,jac,ds,idx,true,z,z,z,z)
end

"""
    evaluate_points(solver::DLP_kress{T},billiard::Bi,k::T) where {T<:Real,Bi<:AbsBilliard} → pts::BoundaryPoints{T}

Constructs the smooth periodic DLP-Kress boundary discretization of `billiard`.

## Description
The solver requires exactly one smooth closed outer boundary represented by a
single curve.

A flat collection of several joined curve segments is considered a composite
boundary and must instead use [`DLP_kress_global_corners`](@ref).

Multiple closed components are not supported by this formulation.

## Arguments
* `solver`: Smooth DLP-Kress solver.
* `billiard`: Billiard geometry.
* `k`: Wavenumber controlling the discretization density.

## Returns
* `pts`: Smooth periodic [`BoundaryPoints`](@ref) discretization.
"""
function evaluate_points(solver::DLP_kress{T},billiard::Bi,k::T) where {T<:Real,Bi<:AbsBilliard}
    boundary=billiard.full_boundary
    isempty(boundary)&&error("Boundary cannot be empty.")
    if length(boundary)==1&&!(boundary[1] isa AbstractVector)
        return _evaluate_points(solver,boundary[1],k,1)
    end
    if _is_single_composite_boundary(boundary)
        error("DLP_kress requires a single smooth closed curve. This boundary is piecewise/composite; use DLP_kress_global_corners instead.")
    end
    error("DLP_kress supports exactly one smooth outer boundary component. Geometries with holes or multiple closed components require a multiply-connected boundary-integral formulation.")
end

"""
    evaluate_points(solver::DLP_kress_global_corners{T},billiard::Bi,k::T) where {T<:Real,Bi<:AbsBilliard} → pts::BoundaryPoints{T}

Constructs the DLP-Kress discretization of a smooth or piecewise smooth
single-component boundary.

## Description
A boundary represented by one smooth closed curve is delegated to the smooth
periodic DLP-Kress discretization.

A flat vector of joined segments is treated as one composite outer boundary and
is globally graded whenever true corners are detected.

Multiple closed boundary components are not supported.

## Arguments
* `solver`: Global-corner DLP-Kress solver.
* `billiard`: Billiard geometry.
* `k`: Wavenumber controlling the discretization density.

## Returns
* `pts`: Smooth or globally graded [`BoundaryPoints`](@ref) discretization.
"""
function evaluate_points(solver::DLP_kress_global_corners{T},billiard::Bi,k::T) where {T<:Real,Bi<:AbsBilliard}
    boundary=billiard.full_boundary
    isempty(boundary)&&error("Boundary cannot be empty.")
    if length(boundary)==1&&!(boundary[1] isa AbstractVector)
        crv=boundary[1]
        base_solver=DLP_kress(solver.pts_scaling_factor,solver.billiard;min_pts=solver.min_pts,eps=solver.eps,symmetry=solver.symmetry)
        return _evaluate_points(base_solver,crv,k,1)
    end
    if _is_single_composite_boundary(boundary)
        return _evaluate_points(solver,boundary,k,1)
    end
    error("DLP_kress_global_corners supports exactly one outer boundary component. Multiple closed components require a multiply-connected boundary-integral formulation.")
end

"""
    build_dlp_kress_workspace_full(solver,pts) → ws::DLPKressWorkspace

Builds the full geometry workspace for a fixed DLP-Kress discretization.

## Description
The workspace contains all wavenumber-independent data required for repeated
matrix assembly:

1. the Kress logarithmic correction matrix,
2. the pairwise geometry cache,
3. flat panel arrays,
4. the matrix dimension.

For a nontrivially graded boundary, the geometry cache retains the corner-aware
parameter data.

## Arguments
* `solver`: Smooth or global-corner DLP-Kress solver.
* `pts`: Boundary discretization.

## Returns
* `ws`: Full [`DLPKressWorkspace`](@ref).
"""
function build_dlp_kress_workspace_full(solver::Union{DLP_kress,DLP_kress_global_corners},pts::BoundaryPoints{T}) where {T<:Real}
    Rmat=build_Rmat_dlp_kress(solver,pts)
    G=boundary_geom_cache(pts,_is_dlp_kress_graded(solver,pts))
    parr=_boundary_panel_arrays_cache(pts)
    N=length(pts)
    return DLPKressWorkspace(Rmat,G,parr,N)
end

"""
    dlp_kress_component_normals(pts::BoundaryPoints{T}) where {T<:Real}

Returns outward normals and parametrization speeds for a DLP-Kress boundary.

## Arguments
* `pts`: DLP-Kress boundary discretization.

## Returns
* `nx`: Outward-normal x components.
* `ny`: Outward-normal y components.
* `speed`: Parametrization speeds.
"""
function dlp_kress_component_normals(pts::BoundaryPoints{T}) where {T<:Real}
    return component_normals(pts)
end

"""
    build_dlp_kress_reduced_workspace(solver,pts) → rws::DLPKressReducedWorkspace

Builds the symmetry-reduced DLP-Kress workspace.

## Description
The complete full-grid workspace is first constructed. Symmetry index orbits
are then generated and the full coordinates, normals and speeds are stored for
regular image-kernel evaluation.

## Arguments
* `solver`: DLP-Kress solver with active symmetry.
* `pts`: Full periodic boundary discretization.

## Returns
* `rws`: Symmetry-reduced [`DLPKressReducedWorkspace`](@ref).
"""
function build_dlp_kress_reduced_workspace(solver::Union{DLP_kress,DLP_kress_global_corners},pts::BoundaryPoints{T}) where {T<:Real}
    full=build_dlp_kress_workspace_full(solver,pts)
    Ifund,full_to_fund,full_to_scale,fund_to_full,fund_to_scale=symmetry_index_orbits(T,pts,solver.symmetry,solver.billiard)
    xs=getindex.(pts.xy,1)
    ys=getindex.(pts.xy,2)
    nx,ny,speed=dlp_kress_component_normals(pts)
    return DLPKressReducedWorkspace(full,Ifund,full_to_fund,full_to_scale,fund_to_full,fund_to_scale,xs,ys,nx,ny,speed,length(Ifund))
end

"""
    build_dlp_kress_workspace(solver,pts)

Builds either the full or symmetry-reduced DLP-Kress workspace.

## Arguments
* `solver`: Smooth or global-corner DLP-Kress solver.
* `pts`: Boundary discretization.

## Returns
* A [`DLPKressWorkspace`](@ref) if no symmetry is active.
* A [`DLPKressReducedWorkspace`](@ref) if symmetry reduction is active.
"""
function build_dlp_kress_workspace(solver::Union{DLP_kress,DLP_kress_global_corners},pts::BoundaryPoints{T}) where {T<:Real}
    return _dlp_kress_use_reduced(solver) ? build_dlp_kress_reduced_workspace(solver,pts) : build_dlp_kress_workspace_full(solver,pts)
end

@inline _workspace_dim(ws::DLPKressWorkspace)=ws.N
@inline _workspace_dim(ws::DLPKressReducedWorkspace)=ws.m

###############################################
############# NO SYMMETRY PATHWAY #############
###############################################

"""
    construct_dlp_matrix!(solver,D,pts,Rmat,G,k;multithreaded=true) → D

Assembles the Kress-corrected Nyström matrix of the Helmholtz double-layer
operator.

## Description
For the two-dimensional Helmholtz Green function, the source-normal
double-layer kernel in the normalization used here is split into a logarithmic
part and a smooth remainder.

For off-diagonal entries,

    D[i,j]=Rmat[i,j]*l1+pts.ws[j]*l2,

where

    l1=-(k/2π)*inner*J₁(kr)/r,

and

    l2=(ik/2)*inner*H₁^(1)(kr)/r-l1*logterm.

The quantities `r`, `1/r`, `inner` and `logterm` are read from the
precomputed [`BoundaryGeomCache`](@ref).

The diagonal limit is stored in `G.kappa` and contributes

    D[i,i]=pts.ws[i]*G.kappa[i].

## Arguments
* `solver`: Smooth or global-corner DLP-Kress solver.
* `D`: Preallocated destination matrix.
* `pts`: Boundary discretization.
* `Rmat`: Kress logarithmic correction matrix.
* `G`: Pairwise geometry cache.
* `k`: Real wavenumber.

## Keyword arguments
* `multithreaded`: Whether to thread the off-diagonal assembly.

## Returns
* `D`: Assembled Kress-corrected DLP matrix.
"""
function construct_dlp_matrix!(solver::Union{DLP_kress,DLP_kress_global_corners},D::AbstractMatrix{Complex{T}},pts::BoundaryPoints{T},Rmat::AbstractMatrix{T},G::BoundaryGeomCache{T},k::T;multithreaded::Bool=true) where {T<:Real}
    αL1=-k*inv_two_pi
    αL2=Complex{T}(0,k/2)
    fill!(D,zero(Complex{T}))
    N=length(pts)
    @inbounds for i in 1:N
        D[i,i]=Complex{T}(pts.ws[i]*G.kappa[i],zero(T))
    end
    @use_threads multithreading=(multithreaded&&N>=32) for j in 2:N
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

# INTERNAL debugging version returning the logarithmic and smooth contributions separately.
function construct_dlp_split!(solver::Union{DLP_kress,DLP_kress_global_corners},Dlog::AbstractMatrix{Complex{T}},Dsmooth::AbstractMatrix{Complex{T}},pts::BoundaryPoints{T},Rmat::AbstractMatrix{T},G::BoundaryGeomCache{T},parr::BoundaryPanelArrays{T},k::T;multithreaded::Bool=true) where {T<:Real}
    αL1=-k*inv_two_pi
    αL2=Complex{T}(0,k/2)
    fill!(Dlog,zero(Complex{T}))
    fill!(Dsmooth,zero(Complex{T}))
    N=length(parr.X)
    @inbounds for i in 1:N
        Dsmooth[i,i]=Complex{T}(pts.ws[i]*G.kappa[i],zero(T))
    end
    @use_threads multithreading=(multithreaded&&N>=32) for j in 2:N
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
    construct_fredholm_matrix!(solver,F,pts,Rmat,G,parr,k;multithreaded=true) → F

Assembles the DLP-Kress Fredholm matrix

    F(k)=I-D(k).

## Description
The Kress split is evaluated directly into the Fredholm matrix, avoiding a
separate temporary DLP matrix.

For off-diagonal entries,

    F[i,j]=-(Rmat[i,j]*l1+pts.ws[j]*l2),

while the diagonal is

    F[i,i]=1-pts.ws[i]*G.kappa[i].

## Arguments
* `solver`: Smooth or global-corner DLP-Kress solver.
* `F`: Preallocated destination matrix.
* `pts`: Boundary discretization.
* `Rmat`: Kress logarithmic correction matrix.
* `G`: Pairwise geometry cache.
* `parr`: Flat panel-array cache.
* `k`: Real wavenumber.

## Keyword arguments
* `multithreaded`: Whether to thread the off-diagonal assembly.

## Returns
* `F`: Assembled Fredholm matrix.
"""
function construct_fredholm_matrix!(solver::Union{DLP_kress,DLP_kress_global_corners},F::AbstractMatrix{Complex{T}},pts::BoundaryPoints{T},Rmat::AbstractMatrix{T},G::BoundaryGeomCache{T},parr::BoundaryPanelArrays{T},k::T;multithreaded::Bool=true) where {T<:Real}
    αL1=-k*inv_two_pi
    αL2=Complex{T}(0,k/2)
    fill!(F,zero(Complex{T}))
    N=length(parr.X)
    @inbounds for i in 1:N
        F[i,i]=one(Complex{T})-Complex{T}(pts.ws[i]*G.kappa[i],zero(T))
    end
    @use_threads multithreading=(multithreaded&&N>=32) for j in 2:N
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
    construct_dlp_matrix_derivatives!(solver,D,D1,D2,pts,Rmat,G,parr,k;multithreaded=true) → D,D1,D2

Assembles the Kress-corrected DLP matrix and its first two wavenumber
derivatives.

## Description
The function computes

    D=D(k),
    D1=dD/dk,
    D2=d²D/dk².

The logarithmic coefficient and smooth remainder are analytically
differentiated with respect to `k`. Geometry and quadrature data remain fixed.

The diagonal DLP limit is independent of `k`, so

    D1[i,i]=D2[i,i]=0.

## Arguments
* `solver`: Smooth or global-corner DLP-Kress solver.
* `D`: Destination matrix for the DLP operator.
* `D1`: Destination matrix for its first derivative.
* `D2`: Destination matrix for its second derivative.
* `pts`: Boundary discretization.
* `Rmat`: Kress logarithmic correction matrix.
* `G`: Pairwise geometry cache.
* `parr`: Flat panel-array cache.
* `k`: Real wavenumber.

## Keyword arguments
* `multithreaded`: Whether to thread the off-diagonal assembly.

## Returns
* `D`: DLP matrix.
* `D1`: First wavenumber derivative.
* `D2`: Second wavenumber derivative.
"""
function construct_dlp_matrix_derivatives!(solver::Union{DLP_kress,DLP_kress_global_corners},D::AbstractMatrix{Complex{T}},D1::AbstractMatrix{Complex{T}},D2::AbstractMatrix{Complex{T}},pts::BoundaryPoints{T},Rmat::AbstractMatrix{T},G::BoundaryGeomCache{T},parr::BoundaryPanelArrays{T},k::T;multithreaded::Bool=true) where {T<:Real}
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
    @use_threads multithreading=(multithreaded&&N>=32) for j in 2:N
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
    construct_fredholm_matrix_derivatives!(solver,F,F1,F2,pts,Rmat,G,parr,k;multithreaded=true) → F,F1,F2

Assembles the DLP-Kress Fredholm matrix and its first two derivatives.

## Description
The DLP quantities are first assembled into the supplied buffers and converted
in place according to

    F=I-D,
    F1=-D1,
    F2=-D2.

## Arguments
* `solver`: Smooth or global-corner DLP-Kress solver.
* `F`: Destination matrix for the Fredholm operator.
* `F1`: Destination matrix for its first derivative.
* `F2`: Destination matrix for its second derivative.
* `pts`: Boundary discretization.
* `Rmat`: Kress logarithmic correction matrix.
* `G`: Pairwise geometry cache.
* `parr`: Flat panel-array cache.
* `k`: Real wavenumber.

## Keyword arguments
* `multithreaded`: Whether to thread the DLP assembly.

## Returns
* `F`: Fredholm matrix.
* `F1`: First derivative.
* `F2`: Second derivative.
"""
function construct_fredholm_matrix_derivatives!(solver::Union{DLP_kress,DLP_kress_global_corners},F::AbstractMatrix{Complex{T}},F1::AbstractMatrix{Complex{T}},F2::AbstractMatrix{Complex{T}},pts::BoundaryPoints{T},Rmat::AbstractMatrix{T},G::BoundaryGeomCache{T},parr::BoundaryPanelArrays{T},k::T;multithreaded::Bool=true) where {T<:Real}
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

#################################################
############# DESYMMETRIZED PATHWAY #############
#################################################

@inline function _regular_dlp_image_D(xi::T,yi::T,xj::T,yj::T,nxj::T,nyj::T,wj::T,k::T,scale::Complex{T}) where {T<:Real}
    dx=xi-xj;dy=yi-yj
    r=hypot(dx,dy)
    r<eps(T)&&return zero(Complex{T})
    c=(nxj*dx+nyj*dy)/r
    return scale*Complex{T}(0,k/2)*c*H(1,k*r)*wj
end

@inline function _regular_dlp_image_D_derivs(xi::T,yi::T,xj::T,yj::T,nxj::T,nyj::T,wj::T,k::T,scale::Complex{T}) where {T<:Real}
    dx=xi-xj;dy=yi-yj
    r=hypot(dx,dy)
    r<eps(T)&&return zero(Complex{T}),zero(Complex{T}),zero(Complex{T})
    c=(nxj*dx+nyj*dy)/r
    kr=k*r
    h0,h1=hankel_pair01(kr)
    D=scale*Complex{T}(0,k/2)*c*h1*wj
    D1=scale*Complex{T}(0,1/2)*c*(kr*h0)*wj
    D2=scale*Complex{T}(0,1/2)*c*(r*h0-k*r^2*h1)*wj
    return D,D1,D2
end

"""
    construct_dlp_matrix!(solver,D,pts,rws,k;multithreaded=true) → D

Assembles the symmetry-reduced DLP-Kress matrix.

## Description
The complete boundary discretization remains available through `pts` and the
full workspace stored in `rws`.

For fundamental indices `a,b`, the corresponding full-grid indices are

    i=rws.Ifund[a],
    j=rws.Ifund[b].

The same physical copy of the source is evaluated using the full-grid Kress
logarithmic split. All remaining symmetry copies are nonsingular relative to
the fundamental source and are therefore added as ordinary regular DLP image
kernels.

## Arguments
* `solver`: DLP-Kress solver with active symmetry.
* `D`: Preallocated reduced `m×m` destination matrix.
* `pts`: Full boundary discretization.
* `rws`: Reduced DLP-Kress workspace.
* `k`: Real wavenumber.

## Keyword arguments
* `multithreaded`: Whether to thread the same-copy block assembly.

## Returns
* `D`: Reduced DLP matrix.
"""
function construct_dlp_matrix!(solver::Union{DLP_kress,DLP_kress_global_corners},D::AbstractMatrix{Complex{T}},pts::BoundaryPoints{T},rws::DLPKressReducedWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}
    m=rws.m
    @assert size(D,1)==m&&size(D,2)==m
    Rmat=rws.full.Rmat
    G=rws.full.G
    Ifund=rws.Ifund
    αL1=-k*inv_two_pi
    αL2=Complex{T}(0,k/2)
    fill!(D,zero(Complex{T}))
    @use_threads multithreading=(multithreaded&&m>=32) for b in 1:m
        j=Ifund[b]
        @inbounds for a in 1:m
            i=Ifund[a]
            if i==j
                D[a,b]=Complex{T}(pts.ws[i]*G.kappa[i],zero(T))
            else
                r=G.R[i,j]
                invr=G.invR[i,j]
                lt=G.logterm[i,j]
                inn=G.inner[i,j]
                _,h1=hankel_pair01(k*r)
                j1=real(h1)
                l1=αL1*inn*j1*invr
                l2=αL2*inn*h1*invr-l1*lt
                D[a,b]=Rmat[i,j]*l1+pts.ws[j]*l2
            end
        end
    end
    for b in 1:m
        j=Ifund[b]
        @inbounds for a in 1:m
            i=Ifund[a]
            xi=rws.xs[i]
            yi=rws.ys[i]
            for ℓ in eachindex(rws.fund_to_full[b])
                q=rws.fund_to_full[b][ℓ]
                q==j&&continue
                scale=rws.fund_to_scale[b][ℓ]
                wq=rws.speed[q]*pts.ws[q]
                D[a,b]+=_regular_dlp_image_D(xi,yi,rws.xs[q],rws.ys[q],rws.nx[q],rws.ny[q],wq,k,scale)
            end
        end
    end
    return D
end

"""
    construct_fredholm_matrix!(solver,F,pts,rws,k;multithreaded=true) → F

Assembles the symmetry-reduced Fredholm matrix

    F(k)=I-D(k).

## Arguments
* `solver`: DLP-Kress solver with active symmetry.
* `F`: Preallocated reduced destination matrix.
* `pts`: Full boundary discretization.
* `rws`: Reduced DLP-Kress workspace.
* `k`: Real wavenumber.

## Keyword arguments
* `multithreaded`: Whether to thread the underlying DLP assembly.

## Returns
* `F`: Reduced Fredholm matrix.
"""
function construct_fredholm_matrix!(solver::Union{DLP_kress,DLP_kress_global_corners},F::AbstractMatrix{Complex{T}},pts::BoundaryPoints{T},rws::DLPKressReducedWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}
    construct_dlp_matrix!(solver,F,pts,rws,k;multithreaded=multithreaded)
    @inbounds for j in axes(F,2),i in axes(F,1)
        F[i,j]*=-1
    end
    @inbounds for i in axes(F,1)
        F[i,i]+=one(Complex{T})
    end
    return F
end

"""
    construct_dlp_matrix_derivatives!(solver,D,D1,D2,pts,rws,k;multithreaded=true) → D,D1,D2

Assembles the symmetry-reduced DLP matrix and its first two wavenumber
derivatives.

## Description
The same-copy fundamental block is evaluated with the complete full-grid Kress
split. Symmetry-image terms are regular and their derivatives are added through
[`_regular_dlp_image_D_derivs`](@ref).

## Arguments
* `solver`: DLP-Kress solver with active symmetry.
* `D`: Reduced destination matrix for the DLP operator.
* `D1`: Reduced destination matrix for the first derivative.
* `D2`: Reduced destination matrix for the second derivative.
* `pts`: Full boundary discretization.
* `rws`: Reduced DLP-Kress workspace.
* `k`: Real wavenumber.

## Keyword arguments
* `multithreaded`: Whether to thread the same-copy block assembly.

## Returns
* `D`: Reduced DLP matrix.
* `D1`: First derivative.
* `D2`: Second derivative.
"""
function construct_dlp_matrix_derivatives!(solver::Union{DLP_kress,DLP_kress_global_corners},D::AbstractMatrix{Complex{T}},D1::AbstractMatrix{Complex{T}},D2::AbstractMatrix{Complex{T}},pts::BoundaryPoints{T},rws::DLPKressReducedWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}
    m=rws.m
    @assert size(D,1)==m&&size(D,2)==m
    full=rws.full
    Rmat=full.Rmat
    G=full.G
    Ifund=rws.Ifund
    αL1=-k*inv_two_pi
    αL2=Complex{T}(0,k/2)
    fill!(D,zero(Complex{T}))
    fill!(D1,zero(Complex{T}))
    fill!(D2,zero(Complex{T}))
    @use_threads multithreading=(multithreaded&&m>=32) for b in 1:m
        j=Ifund[b]
        @inbounds for a in 1:m
            i=Ifund[a]
            if i==j
                D[a,b]=Complex{T}(pts.ws[i]*G.kappa[i],zero(T))
            else
                r=G.R[i,j]
                invr=G.invR[i,j]
                lt=G.logterm[i,j]
                inn=G.inner[i,j]
                kr=k*r
                h0,h1=hankel_pair01(kr)
                j0=real(h0)
                j1=real(h1)
                l1=αL1*inn*j1*invr
                l2=αL2*inn*h1*invr-l1*lt
                D[a,b]=Rmat[i,j]*l1+pts.ws[j]*l2
                l1_1=-(inn*k*j0)*inv_two_pi
                l1_2=(inn*(k*r*j1-j0))*inv_two_pi
                l2_1=(inn*k*(lt*j0+im*pi*h0))*inv_two_pi
                l2_2=(inn*(lt*(j0-k*r*j1)+im*pi*(h0-k*r*h1)))*inv_two_pi
                D1[a,b]=Rmat[i,j]*l1_1+pts.ws[j]*l2_1
                D2[a,b]=Rmat[i,j]*l1_2+pts.ws[j]*l2_2
            end
        end
    end
    for b in 1:m
        j=Ifund[b]
        @inbounds for a in 1:m
            i=Ifund[a]
            xi=rws.xs[i]
            yi=rws.ys[i]
            for ℓ in eachindex(rws.fund_to_full[b])
                q=rws.fund_to_full[b][ℓ]
                q==j&&continue
                scale=rws.fund_to_scale[b][ℓ]
                wq=rws.speed[q]*pts.ws[q]
                d,d1,d2=_regular_dlp_image_D_derivs(xi,yi,rws.xs[q],rws.ys[q],rws.nx[q],rws.ny[q],wq,k,scale)
                D[a,b]+=d
                D1[a,b]+=d1
                D2[a,b]+=d2
            end
        end
    end
    return D,D1,D2
end

"""
    construct_fredholm_matrix_derivatives!(solver,F,F1,F2,pts,rws,k;multithreaded=true) → F,F1,F2

Assembles the symmetry-reduced Fredholm matrix and its first two derivatives.

## Description
The reduced DLP quantities are converted according to

    F=I-D,
    F1=-D1,
    F2=-D2.

## Arguments
* `solver`: DLP-Kress solver with active symmetry.
* `F`: Reduced destination matrix for the Fredholm operator.
* `F1`: Reduced destination matrix for the first derivative.
* `F2`: Reduced destination matrix for the second derivative.
* `pts`: Full boundary discretization.
* `rws`: Reduced DLP-Kress workspace.
* `k`: Real wavenumber.

## Keyword arguments
* `multithreaded`: Whether to thread the underlying DLP assembly.

## Returns
* `F`: Reduced Fredholm matrix.
* `F1`: First derivative.
* `F2`: Second derivative.
"""
function construct_fredholm_matrix_derivatives!(solver::Union{DLP_kress,DLP_kress_global_corners},F::AbstractMatrix{Complex{T}},F1::AbstractMatrix{Complex{T}},F2::AbstractMatrix{Complex{T}},pts::BoundaryPoints{T},rws::DLPKressReducedWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}
    construct_dlp_matrix_derivatives!(solver,F,F1,F2,pts,rws,k;multithreaded=multithreaded)
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

########################################
######### NEEDED FOR HUSIMIS ###########
########################################

"""
    adjoint_fredholm_matrix!(A,D,solver,pts,ws,k;multithreaded=true) → A

Assembles the adjoint full-grid DLP-Kress Fredholm matrix.

## Description
If `D` is the Kress-corrected source-normal double-layer matrix and

    W=diag(ds),

the discrete adjoint operator is

    D'=W⁻¹DᵀW.

Thus

    A=I-D',

with entries

    A[i,j]=-D[j,i]*ds[j]/ds[i]

before the identity shift.

The right null vector of this matrix corresponds directly to the boundary
normal derivative used for Husimi and boundary-function postprocessing.

## Arguments
* `A`: Preallocated adjoint Fredholm destination matrix.
* `D`: Preallocated DLP workspace matrix.
* `solver`: DLP-Kress solver.
* `pts`: Boundary discretization.
* `ws`: Full DLP-Kress workspace.
* `k`: Real wavenumber.

## Keyword arguments
* `multithreaded`: Whether to thread DLP assembly.

## Returns
* `A`: Adjoint Fredholm matrix.
"""
function adjoint_fredholm_matrix!(A::AbstractMatrix{Complex{T}},D::AbstractMatrix{Complex{T}},solver::Union{DLP_kress,DLP_kress_global_corners},pts::BoundaryPoints{T},ws::DLPKressWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}
    N=ws.N
    construct_dlp_matrix!(solver,D,pts,ws.Rmat,ws.G,k;multithreaded=multithreaded)
    ds=pts.ds
    @inbounds for i in 1:N,j in 1:N
        A[i,j]=-D[j,i]*ds[j]/ds[i]
    end
    @inbounds for i in 1:N
        A[i,i]+=one(Complex{T})
    end
    return A
end

"""
    adjoint_fredholm_matrix!(A,D,solver,pts,rws,k;multithreaded=true) → A

Assembles the symmetry-reduced adjoint DLP-Kress Fredholm matrix.

## Description
The reduced DLP matrix is already indexed by fundamental indices. If

    i=Ifund[a],
    j=Ifund[b],

the adjoint reduced matrix satisfies

    A[a,b]=-D[b,a]*ds[j]/ds[i],

followed by the identity shift.

## Arguments
* `A`: Preallocated reduced adjoint Fredholm matrix.
* `D`: Preallocated reduced DLP workspace matrix.
* `solver`: DLP-Kress solver.
* `pts`: Full boundary discretization.
* `rws`: Reduced DLP-Kress workspace.
* `k`: Real wavenumber.

## Keyword arguments
* `multithreaded`: Whether to thread DLP assembly.

## Returns
* `A`: Reduced adjoint Fredholm matrix.
"""
function adjoint_fredholm_matrix!(A::AbstractMatrix{Complex{T}},D::AbstractMatrix{Complex{T}},solver::Union{DLP_kress,DLP_kress_global_corners},pts::BoundaryPoints{T},rws::DLPKressReducedWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}
    m=rws.m
    construct_dlp_matrix!(solver,D,pts,rws,k;multithreaded=multithreaded)
    Ifund=rws.Ifund
    ds=pts.ds
    @inbounds for b in 1:m,a in 1:m
        i=Ifund[a]
        j=Ifund[b]
        A[a,b]=-D[b,a]*ds[j]/ds[i]
    end
    @inbounds for a in 1:m
        A[a,a]+=one(Complex{T})
    end
    return A
end

"""
    adjoint_fredholm_matrix!(A,D,solver,pts,k;multithreaded=true) → A

Convenience wrapper that constructs the appropriate DLP-Kress workspace
internally.

## Arguments
* `A`: Preallocated adjoint Fredholm matrix.
* `D`: Preallocated DLP workspace matrix.
* `solver`: DLP-Kress solver.
* `pts`: Boundary discretization.
* `k`: Real wavenumber.

## Keyword arguments
* `multithreaded`: Whether to thread DLP assembly.

## Returns
* `A`: Adjoint Fredholm matrix.
"""
function adjoint_fredholm_matrix!(A::AbstractMatrix{Complex{T}},D::AbstractMatrix{Complex{T}},solver::Union{DLP_kress,DLP_kress_global_corners},pts::BoundaryPoints{T},k::T;multithreaded::Bool=true) where {T<:Real}
    ws=build_dlp_kress_workspace(solver,pts)
    return adjoint_fredholm_matrix!(A,D,solver,pts,ws,k;multithreaded=multithreaded)
end

"""
    adjoint_fredholm_matrix!(A,D,solver,pts,Rmat,k;multithreaded=true) → A

Convenience wrapper using a precomputed Kress correction matrix.

## Description
The geometry cache is constructed from `pts`. If symmetry is active, the
corresponding reduced workspace is generated before adjoint assembly.

## Arguments
* `A`: Preallocated adjoint Fredholm matrix.
* `D`: Preallocated DLP workspace matrix.
* `solver`: DLP-Kress solver.
* `pts`: Boundary discretization.
* `Rmat`: Precomputed Kress correction matrix.
* `k`: Real wavenumber.

## Keyword arguments
* `multithreaded`: Whether to thread DLP assembly.

## Returns
* `A`: Adjoint Fredholm matrix.
"""
function adjoint_fredholm_matrix!(A::AbstractMatrix{Complex{T}},D::AbstractMatrix{Complex{T}},solver::Union{DLP_kress,DLP_kress_global_corners},pts::BoundaryPoints{T},Rmat::AbstractMatrix{T},k::T;multithreaded::Bool=true) where {T<:Real}
    G=boundary_geom_cache(pts,_is_dlp_kress_graded(solver,pts))
    parr=_boundary_panel_arrays_cache(pts)
    full=DLPKressWorkspace(Rmat,G,parr,length(pts))
    if isnothing(solver.symmetry)
        return adjoint_fredholm_matrix!(A,D,solver,pts,full,k;multithreaded=multithreaded)
    end
    Ifund,full_to_fund,full_to_scale,fund_to_full,fund_to_scale=symmetry_index_orbits(T,pts,solver.symmetry,solver.billiard)
    xs=getindex.(pts.xy,1)
    ys=getindex.(pts.xy,2)
    nx,ny,speed=dlp_kress_component_normals(pts)
    rws=DLPKressReducedWorkspace(full,Ifund,full_to_fund,full_to_scale,fund_to_full,fund_to_scale,xs,ys,nx,ny,speed,length(Ifund))
    return adjoint_fredholm_matrix!(A,D,solver,pts,rws,k;multithreaded=multithreaded)
end

##########################################

"""
    construct_matrices!(solver,A,pts,ws,k;multithreaded=true)
    construct_matrices!(solver,A,pts,Rmat,k;multithreaded=true)
    construct_matrices!(solver,A,A1,A2,pts,ws,k;multithreaded=true)
    construct_matrices!(solver,A,A1,A2,pts,Rmat,k;multithreaded=true)

High-level in-place assembly interface for the DLP-Kress Fredholm operator.

## Description
The methods assemble

    A(k)=I-D(k),

and, when requested,

    A1(k)=-D'(k),
    A2(k)=-D''(k).

Cached-workspace overloads avoid recomputing the geometry and Kress correction
data during repeated wavenumber sweeps.

## Arguments
* `solver`: Smooth or global-corner DLP-Kress solver.
* `A`: Destination matrix for the Fredholm operator.
* `A1`: Destination matrix for the first derivative.
* `A2`: Destination matrix for the second derivative.
* `pts`: Boundary discretization.
* `ws`: Cached full DLP-Kress workspace.
* `Rmat`: Precomputed Kress logarithmic correction matrix.
* `k`: Real wavenumber.

## Keyword arguments
* `multithreaded`: Whether to thread low-level matrix assembly.

## Returns
Single-matrix overloads return `A`. Derivative overloads return
`(A,A1,A2)`.
"""
function construct_matrices!(solver::Union{DLP_kress,DLP_kress_global_corners},A::AbstractMatrix{Complex{T}},pts::BoundaryPoints{T},ws::DLPKressWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}
    return construct_fredholm_matrix!(solver,A,pts,ws.Rmat,ws.G,ws.parr,k;multithreaded=multithreaded)
end

function construct_matrices!(solver::Union{DLP_kress,DLP_kress_global_corners},A::AbstractMatrix{Complex{T}},pts::BoundaryPoints{T},Rmat::AbstractMatrix{T},k::T;multithreaded::Bool=true) where {T<:Real}
    G=boundary_geom_cache(pts,_is_dlp_kress_graded(solver,pts))
    parr=_boundary_panel_arrays_cache(pts)
    return construct_fredholm_matrix!(solver,A,pts,Rmat,G,parr,k;multithreaded=multithreaded)
end

function construct_matrices!(solver::Union{DLP_kress,DLP_kress_global_corners},A::AbstractMatrix{Complex{T}},A1::AbstractMatrix{Complex{T}},A2::AbstractMatrix{Complex{T}},pts::BoundaryPoints{T},ws::DLPKressWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}
    return construct_fredholm_matrix_derivatives!(solver,A,A1,A2,pts,ws.Rmat,ws.G,ws.parr,k;multithreaded=multithreaded)
end

function construct_matrices!(solver::Union{DLP_kress,DLP_kress_global_corners},A::AbstractMatrix{Complex{T}},A1::AbstractMatrix{Complex{T}},A2::AbstractMatrix{Complex{T}},pts::BoundaryPoints{T},Rmat::AbstractMatrix{T},k::T;multithreaded::Bool=true) where {T<:Real}
    G=boundary_geom_cache(pts,_is_dlp_kress_graded(solver,pts))
    parr=_boundary_panel_arrays_cache(pts)
    return construct_fredholm_matrix_derivatives!(solver,A,A1,A2,pts,Rmat,G,parr,k;multithreaded=multithreaded)
end

function construct_matrices!(solver::Union{DLP_kress,DLP_kress_global_corners},basis::AbstractHankelBasis,A::AbstractMatrix{Complex{T}},dA::AbstractMatrix{Complex{T}},ddA::AbstractMatrix{Complex{T}},pts::BoundaryPoints{T},k::T;multithreaded::Bool=true) where {T<:Real}
    ws=build_dlp_kress_workspace(solver,pts)
    construct_matrices!(solver,A,dA,ddA,pts,ws,k;multithreaded=multithreaded)
    return A,dA,ddA
end

function construct_matrices!(solver::Union{DLP_kress,DLP_kress_global_corners},basis::AbstractHankelBasis,A::AbstractMatrix{Complex{T}},dA::AbstractMatrix{Complex{T}},ddA::AbstractMatrix{Complex{T}},pts::BoundaryPoints{T},ws::DLPKressWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}
    construct_matrices!(solver,A,dA,ddA,pts,ws,k;multithreaded=multithreaded)
    return A,dA,ddA
end

###############################################
############ DESYMMETRIZED PATHWAY ############
###############################################

function construct_matrices!(solver::Union{DLP_kress,DLP_kress_global_corners},A::AbstractMatrix{Complex{T}},pts::BoundaryPoints{T},rws::DLPKressReducedWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}
    return construct_fredholm_matrix!(solver,A,pts,rws,k;multithreaded=multithreaded)
end

function construct_matrices!(solver::Union{DLP_kress,DLP_kress_global_corners},basis::AbstractHankelBasis,A::AbstractMatrix{Complex{T}},pts::BoundaryPoints{T},rws::DLPKressReducedWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}
    construct_matrices!(solver,A,pts,rws,k;multithreaded=multithreaded)
    return A
end

function construct_matrices(solver::Union{DLP_kress,DLP_kress_global_corners},basis::AbstractHankelBasis,pts::BoundaryPoints{T},rws::DLPKressReducedWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}
    A=Matrix{Complex{T}}(undef,rws.m,rws.m)
    construct_matrices!(solver,basis,A,pts,rws,k;multithreaded=multithreaded)
    return A
end

function construct_matrices!(solver::Union{DLP_kress,DLP_kress_global_corners},A::AbstractMatrix{Complex{T}},dA::AbstractMatrix{Complex{T}},ddA::AbstractMatrix{Complex{T}},pts::BoundaryPoints{T},rws::DLPKressReducedWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}
    return construct_fredholm_matrix_derivatives!(solver,A,dA,ddA,pts,rws,k;multithreaded=multithreaded)
end

###############################################

"""
    solve(solver,basis,pts,k;multithreaded=true,use_krylov=true,which=:det)
    solve(solver,basis,pts,ws,k;multithreaded=true,use_krylov=true,which=:det_argmin)
    solve(solver,basis,A,pts,k,Rmat;multithreaded=true,use_krylov=true,which=:det_argmin)
    solve(solver,basis,A,pts,ws,k;multithreaded=true,use_krylov=true,which=:det_argmin)

Evaluates a scalar spectral diagnostic of the DLP-Kress Fredholm matrix.

## Description
The Fredholm matrix

    A(k)=I-D(k)

is assembled and passed to the common SVD/determinant backend.

The different overloads allow the caller to reuse a boundary workspace, a Kress
correction matrix and/or the complex matrix buffer itself.

## Arguments
* `solver`: Smooth or global-corner DLP-Kress solver.
* `basis`: Basis placeholder retained for the common solver interface.
* `pts`: Boundary discretization.
* `ws`: Optional cached workspace.
* `A`: Optional preallocated matrix buffer.
* `Rmat`: Optional precomputed Kress correction matrix.
* `k`: Real wavenumber.

## Keyword arguments
* `multithreaded`: Whether to thread matrix assembly.
* `use_krylov`: Whether the scalar backend may use its Krylov pathway.
* `which`: Scalar diagnostic selected by the common backend.

## Returns
A scalar spectral diagnostic determined by `which`.
"""
function solve(solver::Union{DLP_kress,DLP_kress_global_corners},basis::Ba,pts::BoundaryPoints{T},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det) where {T<:Real,Ba<:AbsBasis}
    ws=build_dlp_kress_workspace(solver,pts)
    n=_workspace_dim(ws)
    A=Matrix{Complex{T}}(undef,n,n)
    @blas_1 construct_matrices!(solver,A,pts,ws,k;multithreaded=multithreaded)
    @svd_or_det_solve A use_krylov which MAX_BLAS_THREADS
end

function solve(solver::Union{DLP_kress,DLP_kress_global_corners},basis::Ba,pts::BoundaryPoints{T},ws::DLPKressWorkspace{T},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det_argmin) where {T<:Real,Ba<:AbsBasis}
    A=Matrix{Complex{T}}(undef,ws.N,ws.N)
    @blas_1 construct_matrices!(solver,A,pts,ws,k;multithreaded=multithreaded)
    @svd_or_det_solve A use_krylov which MAX_BLAS_THREADS
end

function solve(solver::Union{DLP_kress,DLP_kress_global_corners},basis::Ba,A::AbstractMatrix{Complex{T}},pts::BoundaryPoints{T},k,Rmat::AbstractMatrix{T};multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det_argmin) where {T<:Real,Ba<:AbsBasis}
    @blas_1 construct_matrices!(solver,A,pts,Rmat,k;multithreaded=multithreaded)
    @svd_or_det_solve A use_krylov which MAX_BLAS_THREADS
end

function solve(solver::Union{DLP_kress,DLP_kress_global_corners},basis::Ba,A::AbstractMatrix{Complex{T}},pts::BoundaryPoints{T},ws::DLPKressWorkspace{T},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det_argmin) where {T<:Real,Ba<:AbsBasis}
    @blas_1 construct_matrices!(solver,A,pts,ws,k;multithreaded=multithreaded)
    @svd_or_det_solve A use_krylov which MAX_BLAS_THREADS
end

###############################################
############ DESYMMETRIZED PATHWAY ############
###############################################

function solve(solver::Union{DLP_kress,DLP_kress_global_corners},basis::Ba,pts::BoundaryPoints{T},rws::DLPKressReducedWorkspace{T},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det_argmin) where {T<:Real,Ba<:AbsBasis}
    A=Matrix{Complex{T}}(undef,rws.m,rws.m)
    @blas_1 construct_matrices!(solver,A,pts,rws,k;multithreaded=multithreaded)
    @svd_or_det_solve A use_krylov which MAX_BLAS_THREADS
end

function solve(solver::Union{DLP_kress,DLP_kress_global_corners},basis::Ba,A::AbstractMatrix{Complex{T}},pts::BoundaryPoints{T},rws::DLPKressReducedWorkspace{T},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det_argmin) where {T<:Real,Ba<:AbsBasis}
    @blas_1 construct_matrices!(solver,A,pts,rws,k;multithreaded=multithreaded)
    @svd_or_det_solve A use_krylov which MAX_BLAS_THREADS
end

###############################################

"""
    solve_vect(solver,basis,A,pts,k,Rmat;multithreaded=true)
    solve_vect(solver,basis,A,pts,ws,k;multithreaded=true)
    solve_vect(solver,basis,pts,ws,k;multithreaded=true)
    solve_vect(solver,basis,pts,k;multithreaded=true)

Computes the near-null vector of the adjoint DLP-Kress Fredholm matrix.

## Description
The adjoint Fredholm matrix is assembled and passed to
[`smallest_nullvec_krylov!`](@ref), which applies a Krylov eigensolver to the
inverse matrix.

The returned vector is therefore the boundary-function representation required
for Husimi and related postprocessing.

Both full and symmetry-reduced workspaces are supported.

## Arguments
* `solver`: Smooth or global-corner DLP-Kress solver.
* `basis`: Basis placeholder retained for the common solver interface.
* `A`: Optional preallocated matrix buffer.
* `pts`: Boundary discretization.
* `ws`: Full or reduced DLP-Kress workspace.
* `Rmat`: Optional precomputed Kress correction matrix.
* `k`: Real wavenumber.

## Keyword arguments
* `multithreaded`: Whether to thread matrix assembly.
* `tol`: Krylov convergence tolerance.
* `maxiter`: Maximum number of Krylov iterations.
* `krylovdim`: Krylov subspace dimension.

## Returns
* `σ`: Smallest-eigenvalue/singular-value proxy.
* `u`: Corresponding normalized near-null vector.
"""
function solve_vect(solver::Union{DLP_kress,DLP_kress_global_corners},basis::Ba,A::AbstractMatrix{Complex{T}},pts::BoundaryPoints{T},k,Rmat::AbstractMatrix{T};multithreaded::Bool=true,tol=1e-12,maxiter::Int=2000,krylovdim::Int=40) where {T<:Real,Ba<:AbsBasis}
    G=boundary_geom_cache(pts,_is_dlp_kress_graded(solver,pts))
    parr=_boundary_panel_arrays_cache(pts)
    full=DLPKressWorkspace(Rmat,G,parr,length(pts))
    ws=if isnothing(solver.symmetry)
        full
    else
        Ifund,full_to_fund,full_to_scale,fund_to_full,fund_to_scale=symmetry_index_orbits(T,pts,solver.symmetry,solver.billiard)
        xs=getindex.(pts.xy,1)
        ys=getindex.(pts.xy,2)
        nx,ny,speed=dlp_kress_component_normals(pts)
        DLPKressReducedWorkspace(full,Ifund,full_to_fund,full_to_scale,fund_to_full,fund_to_scale,xs,ys,nx,ny,speed,length(Ifund))
    end
    return solve_vect(solver,basis,A,pts,ws,k;multithreaded=multithreaded,tol=tol,maxiter=maxiter,krylovdim=krylovdim)
end

function solve_vect(solver::Union{DLP_kress,DLP_kress_global_corners},basis::Ba,A::AbstractMatrix{Complex{T}},pts::BoundaryPoints{T},ws::Union{DLPKressWorkspace{T},DLPKressReducedWorkspace{T}},k;multithreaded::Bool=true,tol=1e-12,maxiter::Int=2000,krylovdim::Int=40) where {T<:Real,Ba<:AbsBasis}
    D=similar(A)
    @blas_1 adjoint_fredholm_matrix!(A,D,solver,pts,ws,k;multithreaded=multithreaded)
    σ,u,_=smallest_nullvec_krylov!(A;nev=1,tol=tol,maxiter=maxiter,krylovdim=krylovdim)
    return σ,u
end

function solve_vect(solver::Union{DLP_kress,DLP_kress_global_corners},basis::Ba,pts::BoundaryPoints{T},ws::Union{DLPKressWorkspace{T},DLPKressReducedWorkspace{T}},k;multithreaded::Bool=true,tol=1e-12,maxiter::Int=2000,krylovdim::Int=40) where {T<:Real,Ba<:AbsBasis}
    n=_workspace_dim(ws)
    A=Matrix{Complex{T}}(undef,n,n)
    return solve_vect(solver,basis,A,pts,ws,k;multithreaded=multithreaded,tol=tol,maxiter=maxiter,krylovdim=krylovdim)
end

function solve_vect(solver::Union{DLP_kress,DLP_kress_global_corners},basis::Ba,pts::BoundaryPoints{T},k;multithreaded::Bool=true,tol=1e-12,maxiter::Int=2000,krylovdim::Int=40) where {T<:Real,Ba<:AbsBasis}
    ws=build_dlp_kress_workspace(solver,pts)
    return solve_vect(solver,basis,pts,ws,k;multithreaded=multithreaded,tol=tol,maxiter=maxiter,krylovdim=krylovdim)
end

###############################################

# INTERNAL - for checking allocation patterns and execution time of the single-k solve variants.
function solve_INFO(solver::Union{DLP_kress,DLP_kress_global_corners},basis::Ba,pts::BoundaryPoints{T},ws::Union{DLPKressWorkspace{T},DLPKressReducedWorkspace{T}},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det_argmin) where {T<:Real,Ba<:AbsBasis}
    N=_workspace_dim(ws)
    A=Matrix{Complex{T}}(undef,N,N)
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