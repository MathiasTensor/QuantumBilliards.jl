# ============================================================
#   DLP-RCIP solver for cornered billiards
#
#   Helsing-style recursively compressed inverse preconditioning
#   for the interior Dirichlet Helmholtz problem, using the
#   double-layer boundary integral equation.
#
#   The uncompressed Nyström equation is
#
#     A(k)ρ = (I - D(k))ρ,
#
#   where D is the source-normal double-layer operator. In the
#   RCIP convention used here,
#
#     K(k) = -D(k),
#     A(k) = I + K(k).
#
#   Near corners the physical layer density is singular or strongly
#   non-smooth. RCIP replaces it locally by a transformed smooth
#   density
#
#     ρ = R ρ̃,
#
#   where R is built independently on each corner by a dyadic
#   Helsing type-b recursion. The global compressed matrix becomes
#
#     A_rcip(k) = I + K°(k) R(k),
#
#   where K° is the global operator with the unresolved inner
#   corner singular blocks removed. Away from corners, R is the
#   identity.
#
#   Main ingredients:
#   - panelwise Gauss-Legendre Nyström discretization,
#   - same-panel logarithmic product quadrature,
#   - local six-panel type-b corner patches,
#   - dyadic RCIP compression of corner singularities,
#   - optional reflection/rotation symmetry-image projection,
#
#   Gauss--Legendre panelization and singular quadrature:
#   Each smooth boundary segment γ(u), u ∈ [0,1], is first split into
#   coarse panels
#
#       [u_a,u_{a+1}],  a = 0,...,Npan-1.
#
#   On every panel we place `ngl` Gauss--Legendre nodes ξ_j ∈ [-1,1]
#   with weights ω_j. The local panel coordinate is
#
#       τ_j = (ξ_j + 1)/2 ∈ [0,1],
#       u_j = (1-τ_j)u_a + τ_j u_{a+1}.
#
#   The geometric Jacobian is included through
#
#       γ_τ = γ_u (u_{a+1}-u_a),
#       ds_j = (ω_j/2) |γ_τ(τ_j)|.
#
#   This yields the composite Nyström discretization
#
#       ∫_Γ f(s) ds ≈ Σ_panels Σ_{j=1}^{ngl} f(γ(u_j)) ds_j.
#
#   For regular inter-panel interactions the Helmholtz double-layer kernel
#   is evaluated directly with this quadrature. Same-panel interactions
#   require special treatment because the double-layer kernel contains a
#   weak logarithmic singularity in the tangential limit. On each panel we
#   therefore split the kernel into the standard Kress/Helsing form
#
#       K(s,t) = L₁(s,t) log|s-t| + L₂(s,t),
#
#   where L₁ and L₂ are smooth. The logarithmic term is integrated using
#   product integration:
#
#       ∫_{-1}^{1} p(t) log|t-ξ_i| dt
#         =
#         Σ_j Λ[i,j] p(ξ_j),
#
#   where Λ is a precomputed logarithmic weight matrix exact for
#   polynomial interpolants up to degree `ngl-1`. The smooth remainder
#   is then integrated with the ordinary Gauss rule.
#
#   Thus:
#     - inter-panel interactions use standard Gauss--Legendre Nyström,
#     - same-panel singular interactions use logarithmic product quadrature.
#
#   The number of panels on a segment of arclength L_j is chosen as
#
#       Npan_j ≈ ppw * k * L_j / (2π * ngl),
#
#   so the total number of boundary nodes scales like
#
#       N_j = ngl * Npan_j ≈ ppw * k * L_j / (2π).
#
#   Hence `ppw` sets the approximate number of boundary nodes per local
#   wavelength, while `ngl` controls the polynomial order within each panel.
#
#   This panel structure is essential for RCIP: the two coarse panels
#   adjacent to each physical corner can be recursively dyadically refined
#   into Helsing type-b six-panel patches, while the rest of the boundary
#   remains on the coarse global discretization.
#
#   For eigenfunction reconstruction we usually solve the weighted
#   adjoint compressed problem. After adjoint RCIP reconstruction,
#   the resulting density is proportional to the boundary function
#
#     u(s) = ∂ₙψ(q(s)),
#
#   which is then used in the single-layer representation of the
#   interior Dirichlet eigenfunction.
#
#   Computational structure:
#
#   global DLP assembly:         O(N²) - bessel h/j pair per entry
#
#   corner block removal:        O(ncorners * (4ngl)²) - just zeroing out the corner blocks in the global matrix
#
#   local RCIP recursion:        O(ncorners * nsub * (6ngl)³) - the Helsing type-b recursion on the six-panel corner patches, which involves inverting 6ngl x 6ngl matrices at each level of the recursion, and there are nsub levels of recursion for each corner, and ncorners corners
#
#   compressed column updates:   O(ncorners * N * (4ngl)²) - after computing the RCIP transform R for each corner, we need to update the global matrix by multiplying the K°(k) operator with the R(k) transform, which involves multiplying the N x 4ngl K°(k) block with the 4ngl x 4ngl R(k) block for each corner, and there are ncorners corners
#
#   For fixed `ngl`, `nsub`, and number of physical corners, the
#   RCIP corner treatment has essentially fixed cost independent of
#   wavenumber. The dominant scaling therefore remains the dense
#   global boundary integral assembly, with N growing approximately
#   linearly with k for fixed points-per-wavelength resolution.
#   Thus RCIP removes the corner-singularity resolution bottleneck
#   without changing the asymptotic dense O(N²) complexity of the
#   underlying boundary integral method.
#
#   Ref:
#   - Helsing, J. Solving integral equations on piecewise smooth boundaries using the RCIP method: a tutorial; https://arxiv.org/abs/1207.6737
#   - Helsing, J. A Fast and Stable Solver for Singular Integral Equations on Piecewise Smooth Curves; https://doi.org/10.1137/090779218
#   MO 19/5/26 - added solve_vect
#TODO Grouping corners into equaivalence clasees for reuse of rcip R matrix
# ============================================================

const INV_TWO_PI=1/(2*pi)
const INV_PI=1/pi

# helper function to make an identity matrix of type Complex{T} without allocating a new matrix from a given matrix A
function make_identity_matrix!(A::AbstractMatrix{Complex{T}}) where {T<:Real}
    fill!(A,zero(Complex{T}))
    @inbounds for i in 1:min(size(A,1),size(A,2))
        A[i,i]=one(Complex{T})
    end
    return A
end
# helper function to add an identity matrix A without allocating a new matrix from a given matrix A
function add_identity_matrix!(A::AbstractMatrix{Complex{T}}) where {T<:Real}
    @inbounds for i in 1:min(size(A,1),size(A,2))
        A[i,i]+=one(Complex{T})
    end
    return A
end
# helper function to compute the relative error between two matrices A and B without allocating a new matrix for the difference
function relerr_noalloc(A,B)
    s=zero(real(eltype(A)))
    n=zero(real(eltype(A)))
    @inbounds for i in eachindex(A,B)
        d=A[i]-B[i]
        s+=abs2(d)
        n+=abs2(A[i])
    end
    return sqrt(s)/max(sqrt(n),eps(real(eltype(A))))
end

"""
    DLP_rcip{T,Bi,Sym} <: SweepSolver

Helsing-style RCIP double-layer solver for cornered billiards.

It assembles the second-kind boundary integral equation

    A(k) = I - D(k),

but replaces the locally singular corner density by an RCIP-transformed density,

    ρ = R ρ̃,

so the compressed matrix is

    A_rcip(k) = I + K°(k) R(k), K = -D.

The solver uses panelwise Gauss-Legendre quadrature, same-panel logarithmic
product integration, recursive corner compression, and optional symmetry-image
projection through `symmetry`.
"""
struct DLP_rcip{T<:Real,Bi<:AbsBilliard,Sym}<:SweepSolver
    pts_scaling_factor::Vector{T}
    dim_scaling_factor::T
    eps::T
    min_dim::Int64
    min_pts::Int64
    billiard::Bi
    symmetry::Sym
    ngl::Int
    nsub::Int
    logquad_prec::Int
    use_panel_logquad::Bool
end

"""
    DLP_rcip(pts_scaling_factor::Union{T,Vector{T}},billiard::Bi;min_pts=24,eps=T(1e-15),symmetry::Union{Nothing,AbsSymmetry}=nothing,ngl::Int=16,nsub::Int=30,logquad_prec::Int=256,use_panel_logquad::Bool=true) where {T<:Real,Bi<:AbsBilliard}

Constructor for the RCIP double-layer solver.

`pts_scaling_factor` controls the global boundary resolution

    Npan_piece ≈ pts_scaling_factor * k * L_piece / (2π * ngl).

Important parameters:
- `ngl::Int`: Gauss-Legendre nodes per panel.
- `nsub::Int`: number of dyadic RCIP refinement levels.
- `logquad_prec::Int`: BigFloat precision used when building log-product weights.
- `symmetry::Union{Nothing,AbsSymmetry}`: optional reflection/rotation symmetry used by method-of-images desymmetrization.
"""
function DLP_rcip(pts_scaling_factor::Union{T,Vector{T}},billiard::Bi;min_pts=24,eps=T(1e-15),symmetry::Union{Nothing,AbsSymmetry}=nothing,ngl::Int=16,nsub::Int=30,logquad_prec::Int=256,use_panel_logquad::Bool=true) where {T<:Real,Bi<:AbsBilliard}
    bs=pts_scaling_factor isa Vector ? T.(pts_scaling_factor) : [T(pts_scaling_factor)]
    Sym=typeof(symmetry)
    return DLP_rcip{T,Bi,Sym}(bs,bs[1],eps,min_pts,min_pts,billiard,symmetry,ngl,nsub,logquad_prec,use_panel_logquad)
end

"""
    DLPRCIPPanel{T,C}

One coarse RCIP panel on a curve segment.

The panel stores a subinterval `[u0,u1]` of a parent curve parameter `u∈[0,1]`.
A Gauss coordinate `τ∈[0,1]` is mapped by

    u(τ) = (1-τ)u0 + τu1.

Fields:
- `curve<:AbsCurve`: parent curve object.
- `u0,u1`: parameter interval limits on that curve.
- `id`: global panel id in the RCIP discretization.
- `piece_id`: curve-piece id inside the connected boundary component.
- `comp_id`: connected component id.
- `is_physical`: true for physical boundary panels; useful if later adding
  artificial symmetry edges or auxiliary panels.
"""
struct DLPRCIPPanel{T<:Real,C<:AbsCurve}
    curve::C
    u0::T
    u1::T
    id::Int
    piece_id::Int
    comp_id::Int
    is_physical::Bool
end

"""
    DLPRCIPPanelData{T,C}

Panel metadata attached to a `BoundaryPoints` discretization.

Fields:
- `panels`: vector of all coarse RCIP panels.
- `panel_id[j]`: panel containing global node `j`.
- `local_id[j]`: local Gauss index of global node `j` inside its panel.
- `ngl`: number of Gauss nodes per panel.

This metadata is needed for same-panel logarithmic product quadrature and for
constructing local RCIP corner patches.
"""
struct DLPRCIPPanelData{T<:Real,C<:AbsCurve}
    panels::Vector{DLPRCIPPanel{T,C}}
    panel_id::Vector{Int}
    local_id::Vector{Int}
    ngl::Int
end

"""
    DLPRCIPDiscretization{T,C}

Complete RCIP boundary discretization.

Contains:
- `bp`: standard `BoundaryPoints{T}` object used by the library.
- `pdata`: RCIP panel metadata.

This wrapper keeps compatibility with existing `BoundaryIntegralMethod` infrastructure while adding
the panel information required by RCIP.
"""
struct DLPRCIPDiscretization{T<:Real,C<:AbsCurve}
    bp::BoundaryPoints{T}
    pdata::DLPRCIPPanelData{T,C}
end

# helper to get the number of boundary nodes in the RCIP discretization for matrix size
boundary_matrix_size(disc::DLPRCIPDiscretization)=length(disc.bp.xy)

"""
   _panel_curve_eval(p::DLPRCIPPanel{T},τ::T) where {T<:Real}

Evaluate one RCIP panel at local coordinate `τ∈[0,1]`.

The mapping is

    u = (1-τ)u0 + τu1,
    γτ = γu * (u1-u0),
    γττ = γuu * (u1-u0)^2.

Returns:
- boundary point `q`,
- outward unit normal `n = (ty,-tx)/|t|`,
- speed `sp = |γτ|`,
- curvature `κ = -(tx*tty - ty*ttx) / |γτ|^3`.
Note: The sign matches the outward-normal convention for a counter-clockwise boundary.
"""
@inline function _panel_curve_eval(p::DLPRCIPPanel{T},τ::T) where {T<:Real}
    u=(one(T)-τ)*p.u0+τ*p.u1
    du=p.u1-p.u0
    q=SVector{2,T}(curve(p.curve,u)) # just in case type instability with ForwardDiff.jl
    t=SVector{2,T}(tangent(p.curve,u))*du
    tt=SVector{2,T}(tangent_2(p.curve,u))*(du^2)
    sp=hypot(t[1],t[2])
    n=SVector{2,T}(t[2]/sp,-t[1]/sp)
    κ=-(t[1]*tt[2]-t[2]*tt[1])/sp^3 # calculate curvature here, easier than to call curvature(crv,t)
    return q,n,sp,κ
end

"""
    _split_curve_into_rcip_panels(crv::C,comp_id::Int,piece_id::Int,npan::Int,::Type{T}) where {C,T<:Real}

Split one curve segment into `npan` equal parameter panels. This is the Gauss-Legendre panelization used for the coarse RCIP discretization. The panels are later refined by RCIP near corners. Helsing requires panelization to do dyadic refinement for the panels near corners. So we are left needing to use high--order quadrature on the coarse panels, which is the reason logarithmic product integration is implemented in the first place.

For panel `a=0,...,npan-1`, the parameter interval is

    [a/npan, (a+1)/npan].

The resulting panels are later discretized by Gauss-Legendre quadrature.
"""
function _split_curve_into_rcip_panels(crv::C,comp_id::Int,piece_id::Int,npan::Int,::Type{T}) where {C<:AbsCurve,T<:Real}
    out=DLPRCIPPanel{T,C}[]
    for a in 0:npan-1
        # logic is a/npan ≤ u ≤ (a+1)/npan, so u0=a/npan and u1=(a+1)/npan, and the panel id is length(out)+1 since we start with an empty vector and push panels in order. piece_id and comp_id are passed through for later use in panel metadata and dont change since the curve piece and component are not changing when we split into panels.
        push!(out,DLPRCIPPanel{T,C}(crv,T(a)/T(npan),T(a+1)/T(npan),length(out)+1,piece_id,comp_id,true))
    end
    return out
end

"""
    _panelize_component(comp::Vector,comp_id::Int,k::T,b::T,min_panels_per_piece::Int,ngl::Int,::Type{T}) where {T<:Real}

Split one connected boundary component into RCIP panels. min_panels_per_piece ensures that even the shortest curve pieces get at least some panels, while the pts_scaling_factor scaling ensures that the total number of panels grows linearly with k to resolve the oscillations. Helsing's RCIP requires dyadic panel refinement near corners, so we start with a coarse panelization of the entire curve piece and let RCIP refine as needed. But we still need to use high-order quadrature on the coarse panels to get accurate results, which is the reason for the ngl argument and the logarithmic product integration implementation in the first place.

For each curve piece of length `L`, choose approximately

    Npan ≈ b * k * L / (2π * ngl),

with at least `min_panels_per_piece` panels.

The division by `ngl` appears because each panel carries `ngl` Gauss nodes, so
the total nodes per wavelength scale like `b`.
"""
function _panelize_component(comp::Vector,comp_id::Int,k::T,b::T,min_panels_per_piece::Int,ngl::Int,::Type{T}) where {T<:Real}
    # comp is a vector of <:AbsCurve function pieces that can construct boundary segments. The total boundary is made up of these curves that each define a part of the boundary
    C=eltype(comp)
    panels=DLPRCIPPanel{T,C}[]
    pid=0
    for j in eachindex(comp)
        # each segment has a length that we use to get the number of panels for that segment, and then we split that segment into panels and push them to the panels vector. The panel id is just a running count of how many panels we have created so far, and piece_id and comp_id are passed through for later use in panel metadata.
        L=T(comp[j].length)
        Nj=max(min_panels_per_piece,ceil(Int,b*k*L/(T(2*pi)*ngl)))
        # loc is a vector of DLPRCIPPanel objects that represent the panels for this segment. Since we want to push / concatenate the whole boundary and not a vector of panels for each segment, we need to keep track of the global panel id across segments, which is what pid is for. We also need to pass the piece_id and comp_id for each panel for later use in panel metadata.
        loc=_split_curve_into_rcip_panels(comp[j],comp_id,j,Nj,T)
        for p in loc
            pid+=1
            push!(panels,DLPRCIPPanel{T,C}(p.curve,p.u0,p.u1,pid,p.piece_id,p.comp_id,p.is_physical))
        end
    end
    return panels
end

"""
   _discretize_rcip_panels(panels::Vector{DLPRCIPPanel{T,C}},ngl::Int,billiard) where {T<:Real,C}

Apply an `ngl`-point Gauss-Legendre rule to every RCIP panel. This is the semi low--level wrapper around evaluate_points. It computes the standard `BoundaryPoints` fields plus the RCIP panel metadata needed for same-panel log-product quadrature and local RCIP corner patch construction. The symmetry shifts are copied from the billiard when available so that method-of-images desymmetrization works correctly.

For each node,

    τq = (ξq+1)/2,
    dsq = (ωq/2) |γτ(τq)|.

The result is stored as a standard `BoundaryPoints{T}` plus RCIP panel metadata.
The symmetry shifts `shift_x` and `shift_y` are copied from the billiard when
available so that method-of-images desymmetrization works correctly.
"""
function _discretize_rcip_panels(panels::Vector{DLPRCIPPanel{T,C}},ngl::Int,billiard::Bi) where {T<:Real,C<:AbsCurve,Bi<:AbsBilliard}
    # no need to precompute, very cheap since usually 16th order GL
    ξ,ω=gausslegendre(ngl)
    ξ=T.(ξ);ω=T.(ω)
    N=length(panels)*ngl # total number of nodes is number of panels times number of GL points per panel
    xy=Vector{SVector{2,T}}(undef,N)
    normal=Vector{SVector{2,T}}(undef,N)
    s=Vector{T}(undef,N)
    ds=Vector{T}(undef,N)
    curvature=Vector{T}(undef,N)
    panel_id=Vector{Int}(undef,N)
    local_id=Vector{Int}(undef,N)
    soff=zero(T)
    qid=0
    # for each panel, we loop over the GL points and evaluate the panel curve at those points to get the boundary point, normal, speed, curvature, and quadrature weight. We also keep track of the global node id qid, which runs from 1 to N across all panels and GL points. The local_id is just the index of the GL point within the panel, which runs from 1 to ngl for each panel. The panel_id is the global panel id that this node belongs to, which we get from the panel metadata. The s field is the cumulative arclength along the boundary, which we compute by summing up the ds values as we go along.
    @inbounds for (pidx,p) in enumerate(panels)
        local_s=zero(T)
        for q in 1:ngl
            qid+=1
            τ=(ξ[q]+one(T))/T(2)
            xq,nq,sp,κ=_panel_curve_eval(p,τ)
            wq=ω[q]/T(2)
            xy[qid]=xq
            normal[qid]=nq
            ds[qid]=wq*sp
            local_s+=ds[qid]
            s[qid]=soff+local_s
            curvature[qid]=κ
            panel_id[qid]=pidx
            local_id[qid]=q
        end
        soff+=local_s
    end
    # optionally give info for shifting the symmetry axes for method-of-images desymmetrization. Otherwise it will give wrong images. The shifts are just the coordinates of the symmetry axes, which we can get from the billiard if they are defined.
    shift_x=hasproperty(billiard,:x_axis) ? T(billiard.x_axis) : zero(T)
    shift_y=hasproperty(billiard,:y_axis) ? T(billiard.y_axis) : zero(T)
    bp=BoundaryPoints{T}(xy,normal,s,ds,T[],T[],curvature,SVector{2,T}[],shift_x,shift_y)
    pdata=DLPRCIPPanelData{T,C}(panels,panel_id,local_id,ngl)
    return DLPRCIPDiscretization{T,C}(bp,pdata)
end

# helper to get the connected boundary components as vectors of curve pieces. If the input boundary is already given as a vector of components, we just collect them. Otherwise we assume it's a single component and collect it as one vector of curve pieces. This is used to handle both the full boundary and the desymmetrized boundary, which may have different structures.
function _rcip_boundary_components(boundary)
    if boundary[1] isa AbstractVector
        return [collect(comp) for comp in boundary]
    end
    # a bit of a hack to check if the single component is closed by checking if the endpoints of the curve pieces match up. If they do, we can treat each curve piece as its own component for panelization, which is more efficient. If not, we have to treat the whole thing as one component to avoid breaking the continuity at the junctions.
    all_closed=all(crv->begin 
        p0=curve(crv,0.0)
        p1=curve(crv,1.0)
        hypot(p1[1]-p0[1],p1[2]-p0[2])<=1e-8
    end,boundary)
    return all_closed ? [[crv] for crv in boundary] : [collect(boundary)]
end

"""
    evaluate_points(solver::DLP_rcip{T}, billiard::Bi, k::T) where {T<:Real,Bi<:AbsBilliard}

Build the RCIP boundary discretization used by `solver` at wavenumber `k`.

The boundary is chosen as:
- `billiard.full_boundary` if no symmetry is used,
- `billiard.desymmetrized_full_boundary` if a symmetry is supplied.

The component is panelized with `solver.pts_scaling_factor`, `solver.ngl`, and a fixed
minimum number of panels per curve piece. The result is a
`DLPRCIPDiscretization` struct containing `BoundaryPoints` and also `DLPRCIPPanelData`.
"""
function evaluate_points(solver::DLP_rcip{T},billiard::Bi,k::T) where {T<:Real,Bi<:AbsBilliard}
    boundary=isnothing(solver.symmetry) ? billiard.full_boundary : billiard.desymmetrized_full_boundary
    comps=_rcip_boundary_components(boundary)
    length(comps)==1 || error("DLP_rcip currently expects one connected boundary component.")
    panels=_panelize_component(comps[1],1,k,solver.pts_scaling_factor[1],4,solver.ngl,T)
    return _discretize_rcip_panels(panels,solver.ngl,billiard)
end

############################
##### LOG PRODUCT QUAD #####
############################

"""
    log_moments_big(x::BigFloat,n::Int)

Moments M_p(x) up to degree n-1 for handling logarithmic singularities on the same panel.

Compute exact logarithmic moments

    m[r] = ∫_{-1}^{1} t^(r-1) log|t-x| dt,   r=1,...,n,

or equivalently, with zero-based polynomial degree p=r-1,

    M_p(x) = ∫_{-1}^{1} t^p log|t-x| dt,     p=0,...,n-1.

The first entry is the p=0 moment:

    M_0(x) = ∫ log|t-x| dt = (1-x)log|1-x| + (1+x)log|1+x| - 2.

For p≥1, one expands

    t^p = (u+x)^p,  u=t-x,

so

    M_p(x) = Σ_{q=0}^{p} binomial(p,q) x^(p-q) ∫ u^q log|u| du,

where

    ∫ u^q log|u| du = u^(q+1) [ log|u|/(q+1) - 1/(q+1)^2 ].

BigFloat is used due to ill-conditioning as `ngl` grows.
"""
function log_moments_big(x::BigFloat,n::Int)
    m=Vector{BigFloat}(undef,n)
    a=big(1)-x;b=big(1)+x
    m[1]=a*log(abs(a))+b*log(abs(b))-big(2)
    uL=-big(1)-x;uR=big(1)-x
    for p in 1:n-1
        s=big(0)
        for q in 0:p
            coeff=BigFloat(binomial(p,q))*x^(p-q)
            qq=BigFloat(q+1)
            F(u)=abs(u)<eps(BigFloat) ? big(0) : u^(q+1)*(log(abs(u))/qq-inv(qq^2))
            s+=coeff*(F(uR)-F(uL))
        end
        m[p+1]=s
    end
    return m
end

"""
    log_weights_matrix(::Type{T},ξ::Vector{T};prec::Int=256) where {T<:Real}

Construct the logarithmic product-integration weight matrix Λ.

Given interpolation nodes:

    ξ₁,...,ξ_n ∈ [-1,1],

we seek weights satisfying:

    ∫_{-1}^{1} p(t) log|t-ξ_i| dt
      =
      Σ_j Λ[i,j] p(ξ_j)

for every polynomial p (t) = Σ_{k=0}^{n-1} c_k t^k of degree ≤ n−1.
Using t = ξ_i we get for p(ξ_i) = Σ_{k=0}^{n-1} c_k ξ_i^k
where the Vandermonde matrix is V[j,k] = ξ_j^(k-1) and the exact 
logarithmic moments are m_i[k] = ∫ t^(k-1) log|t-ξ_i| dt.

That means Λ exactly integrates the logarithmic singularity
against polynomial interpolants up to degree n-1.

Exact logarithmic moments are:

    m_i[k] = ∫ t^(k-1) log|t-ξ_i| dt.

We require Σ_j Λ[i,j] p(ξ_j) = m_i[k] for all p of degree ≤ n-1, which means:

    Λ_i V = m_i^T

so we can use Julia's backslash operator to solve for the log weights:

    Λ_i = m_i^T V^{-1}.

Each row is therefore obtained by solving: V^T λ = m.
"""
function log_weights_matrix(::Type{T},ξ::Vector{T};prec::Int=256) where {T<:Real}
    setprecision(BigFloat,prec) do
        n=length(ξ)
        xb=BigFloat.(ξ)
        V=Matrix{BigFloat}(undef,n,n)
        @inbounds for i in 1:n
            V[i,1]=big(1)
            for j in 2:n
                V[i,j]=V[i,j-1]*xb[i]
            end
        end
        Λb=Matrix{BigFloat}(undef,n,n)
        @inbounds for i in 1:n
            m=log_moments_big(xb[i],n)
            Λb[i,:].=transpose(V)\m
        end
        return T.(Λb)
    end
end

############################
##### DLP ASSEMBLY #########
############################

# Return: r = |x_i - x_j|
# and directional cosine: q = ((x_i-x_j)·n_j)/r.
# This is the geometric factor in the 2D Helmholtz double-layer kernel: ∂G/∂n_j.
@inline function source_cos(xi,yi,xj,yj,nxj,nyj)
    dx=xi-xj;dy=yi-yj
    r=hypot(dx,dy)
    q=(dx*nxj+dy*nyj)/r
    return r,q
end

# Diagonal Nyström contribution of the double-layer kernel.
# For the current sign convention this returns ds * κ / (2π).
@inline dlp_diag(::Type{T},ds::T,κ::T) where {T<:Real}=Complex{T}(ds*κ*INV_TWO_PI,zero(T))

# Off-diagonal Helmholtz double-layer kernel contribution.
#     r = |x_i - x_j|,
#     q = ((x_i - x_j) ⋅ n_j) / r,
#
# and the quadrature weight `dsj` is included.
@inline dlp_regular(::Type{T},k::T,r::T,q::T,dsj::T) where {T<:Real}=Complex{T}(zero(T),k/2)*H(1,k*r)*q*dsj

# Add one regular double-layer image contribution to `D[i,j]`.
# Used for reflection and rotation image terms in symmetry-reduced sectors.
# Returns `false` if the target/source distance is numerically zero, otherwise
# returns `true`.
@inline function _add_dlp_regular!(D::AbstractMatrix{Complex{T}},i::Int,j::Int,xi::T,yi::T,xj::T,yj::T,nxj::T,nyj::T,k::T,dsj::T;scale::Complex{T}=one(Complex{T})) where {T<:Real}
    dx=xi-xj;dy=yi-yj
    r=hypot(dx,dy)
    r<=eps(T) && return false
    q=(dx*nxj+dy*nyj)/r
    D[i,j]+=scale*dlp_regular(T,k,r,q,dsj)
    return true
end

# construct the source normal DLP Nyström matrix D for the RCIP discretization. The main difference from a standard DLP assembly is that we need to handle same-panel interactions with logarithmic product quadrature, which requires the panel_id and local_id metadata to identify which nodes are on the same panel and to compute the appropriate weights. We also need to handle the diagonal terms with dlp_diag. The symmetry handling is also included here, where we add contributions from reflected or rotated image sources as needed based on the specified symmetry. The use_panel_logquad flag allows us to toggle between using the special log-product quadrature for same-panel interactions or just treating them as regular off-diagonal interactions.
function assemble_dlp_rcip!(D::Matrix{Complex{T}},disc::DLPRCIPDiscretization{T,C},k::T,Λ::Matrix{T},ξ::Vector{T},ω::Vector{T};symmetry=nothing,use_panel_logquad::Bool=true,multithreaded::Bool=true) where {T<:Real,C}
    N=boundary_matrix_size(disc)
    fill!(D,zero(Complex{T}))
    xy=disc.bp.xy
    normal=disc.bp.normal
    ds=disc.bp.ds
    κ=disc.bp.curvature
    panel_id=disc.pdata.panel_id
    local_id=disc.pdata.local_id
    add_x=false;add_y=false;add_xy=false
    sxgn=one(Complex{T});sygn=one(Complex{T});sxy=one(Complex{T})
    have_rot=false;nrot=1;mrot=0
    cx=zero(T);cy=zero(T)
    # handle symmetry if provided. We check if the symmetry is a reflection or a rotation and set the appropriate flags and parameters for adding image contributions. For reflections, we determine which axes are involved and set the signs for the image contributions based on the parity. For rotations, we set up the rotation tables needed to compute the rotated image contributions during assembly.
    if !isnothing(symmetry)
        s=symmetry
        if hasproperty(s,:axis)
            if s.axis==:y_axis
                add_x=true
                sxgn=Complex{T}(s.parity==-1 ? -one(T) : one(T),zero(T))
            elseif s.axis==:x_axis
                add_y=true
                sygn=Complex{T}(s.parity==-1 ? -one(T) : one(T),zero(T))
            elseif s.axis==:origin
                add_x=true;add_y=true;add_xy=true
                px=s.parity[1]==-1 ? -one(T) : one(T)
                py=s.parity[2]==-1 ? -one(T) : one(T)
                sxgn=Complex{T}(px,zero(T))
                sygn=Complex{T}(py,zero(T))
                sxy=Complex{T}(px*py,zero(T))
            end
        elseif s isa Rotation 
            have_rot=true
            nrot=s.n
            mrot=mod(s.m,nrot)
            cx=T(s.center[1]);cy=T(s.center[2])
        end
    end
    ctab=T[]
    stab=T[]
    χ=ComplexF64[]
    if have_rot
        ctab,stab,χ=_rotation_tables(T,nrot,mrot)
    end
    shift_x=disc.bp.shift_x
    shift_y=disc.bp.shift_y
    @use_threads multithreading=multithreaded for j in 1:N
        qj=xy[j];nj=normal[j]
        xj=qj[1];yj=qj[2];nxj=nj[1];nyj=nj[2]
        dsj=ds[j];pj=panel_id[j];jl=local_id[j]
        speed_half_j=dsj/ω[jl]
        for i in 1:N
            qi=xy[i];xi=qi[1];yi=qi[2]
            if i==j # diagonal term is independant of the quadrature chosen
                D[i,j]+=dlp_diag(T,dsj,κ[j])
            else
                r,q=source_cos(xi,yi,xj,yj,nxj,nyj)
                # for same-panel interactions, use the log-product quadrature weights to handle the logarithmic singularity. We check if the source node j is on the same panel as the target node i using the panel_id metadata. If they are on the same panel, we compute the contribution using the precomputed log weights Λ and the local coordinates ξ to get the correct singular behavior. If they are not on the same panel, we just use the regular off-diagonal kernel evaluation.
                if use_panel_logquad && panel_id[i]==pj
                    il=local_id[i]
                    du=abs(ξ[il]-ξ[jl])
                    full=Complex{T}(zero(T),k/2)*H(1,k*r)*q*speed_half_j
                    L1=-(k*INV_PI)*J(1,k*r)*q*speed_half_j
                    L2=full-L1*log(du)
                    D[i,j]+=Λ[il,jl]*L1+ω[jl]*L2
                else
                    D[i,j]+=dlp_regular(T,k,r,q,dsj)
                end
            end
            # since reflections have 1d irreps whose characters are just ±1, the image contributions are just ± the regular kernel evaluation at the reflected point. For rotations, we have to compute the rotated image points and normals and then evaluate the regular kernel at those points with the appropriate rotation factor from the character table. The symmetry shifts are also applied to the reflected points to ensure that the images are placed correctly according to the symmetry axes.
            if add_x
                xr=_x_reflect(xj,shift_x);yr=yj
                nxr,nyr=_x_reflect_normal(nxj,nyj)
                _add_dlp_regular!(D,i,j,xi,yi,xr,yr,nxr,nyr,k,dsj;scale=sxgn)
            end
            if add_y
                xr=xj;yr=_y_reflect(yj,shift_y)
                nxr,nyr=_y_reflect_normal(nxj,nyj)
                _add_dlp_regular!(D,i,j,xi,yi,xr,yr,nxr,nyr,k,dsj;scale=sygn)
            end
            if add_xy
                xr=_x_reflect(xj,shift_x);yr=_y_reflect(yj,shift_y)
                nxr,nyr=_xy_reflect_normal(nxj,nyj)
                _add_dlp_regular!(D,i,j,xi,yi,xr,yr,nxr,nyr,k,dsj;scale=sxy)
            end
            if have_rot
                for l in 1:nrot-1
                    xr,yr=_rot_point(xj,yj,cx,cy,ctab[l+1],stab[l+1])
                    nxr,nyr=_rot_vec(nxj,nyj,ctab[l+1],stab[l+1])
                    _add_dlp_regular!(D,i,j,xi,yi,xr,yr,nxr,nyr,k,dsj;scale=χ[l+1])
                end
            end
        end
    end
    return D
end

############################
##### RCIP CORNER DATA #####
############################

"""
    DLPRCIPCornerMeta{T}

Metadata for one physical RCIP corner.

Fields:
- `cid`: corner id.
- `pleft`, `pright`: coarse panels ending/starting at the corner.
- `patch_panels`: four coarse panels used to build the local type-b patch.
- `patch_nodes`: global node ids belonging to those four panels.
- `point`: physical corner coordinate.
"""
struct DLPRCIPCornerMeta{T<:Real}
    cid::Int
    pleft::Int
    pright::Int
    patch_panels::Vector{Int}
    patch_nodes::Vector{Int}
    point::SVector{2,T}
end

"""
    _same_point(a,b; tol)

Return true if two physical points are equal up to Euclidean tolerance `tol`.
Used to match full-boundary physical corners to panels of the possibly
desymmetrized boundary.
"""
@inline function _same_point(a::SVector{2,T},b::SVector{2,T};tol::T=T(1e-8)) where {T<:Real}
    return hypot(a[1]-b[1],a[2]-b[2])<=tol
end

"""
    _curve_endpoint(crv, side, T)

Return the left or right endpoint of curve `crv`.

`side === :left` evaluates `u=0`; otherwise it evaluates `u=1`.
"""
function _curve_endpoint(crv,side::Symbol,::Type{T}) where {T<:Real}
    u=side===:left ? zero(T) : one(T)
    return SVector{2,T}(curve(crv,u))
end

"""
    _physical_corner_points(comp, T; angle_tol)

Return physical corner coordinates of a connected boundary component.

A junction between consecutive curve pieces is considered a true corner when
the angle between outgoing and incoming unit tangents exceeds `angle_tol`.

Smooth joins are ignored.
"""
function _physical_corner_points(comp::Vector,::Type{T};angle_tol::T=T(1e-8)) where {T<:Real}
    pts=SVector{2,T}[]
    m=length(comp)
    for j in 1:m
        jr=j==m ? 1 : j+1
        _is_true_corner(comp[j],comp[jr],T;angle_tol=angle_tol) && push!(pts,_curve_endpoint(comp[j],:right,T))
    end
    return pts
end

function _physical_corner_points_open(comp::Vector,::Type{T};angle_tol::T=T(1e-8)) where {T<:Real}
    pts=SVector{2,T}[]
    for j in 1:length(comp)-1
        _is_true_corner(comp[j],comp[j+1],T;angle_tol=angle_tol) && push!(pts,_curve_endpoint(comp[j],:right,T))
    end
    return pts
end

# helper to get the left and right endpoints of a panel's curve for matching to physical corners. We use the curve evaluation at u0 and u1 to get the physical coordinates of the panel endpoints, which we can then compare to the detected physical corners to find which panels are adjacent to each corner.
@inline _panel_endpoint_left(p::DLPRCIPPanel{T}) where {T<:Real}=SVector{2,T}(curve(p.curve,p.u0))
@inline _panel_endpoint_right(p::DLPRCIPPanel{T}) where {T<:Real}=SVector{2,T}(curve(p.curve,p.u1))

"""
    _nodes_for_panels(pids, ngl)

Return global node indices belonging to the panel ids `pids`.

Each panel has exactly `ngl` Gauss-Legendre nodes and that panel `p`
occupies indices `(p-1)*ngl+1 : p*ngl`.
"""
function _nodes_for_panels(pids::Vector{Int},ngl::Int)
    ids=Int[]
    sizehint!(ids,length(pids)*ngl)
    for p in pids
        append!(ids,((p-1)*ngl+1):(p*ngl))
    end
    return ids
end

"""
    _find_corner_panel_pair(panels, xcorner; tol)

Find the two coarse panels meeting at the physical corner `xcorner`.

Returns `(pleft, pright)`, where `pleft` ends at the corner and `pright`
starts at the corner. If either panel is not found, its index is returned as
zero.
"""
function _find_corner_panel_pair(panels::Vector{DLPRCIPPanel{T,C}},xcorner::SVector{2,T};tol::T=T(1e-8)) where {T<:Real,C}
    # iterate over all the panels and check if the right endpoint of the panel matches the corner (within tolerance), which would mean that this panel is the left panel adjacent to the corner. Similarly, check if the left endpoint of the panel matches the corner, which would mean that this panel is the right panel adjacent to the corner. We keep track of these indices in pleft and pright, and return them at the end. If we don't find a match for either endpoint, we return zero for that index.
    pleft=0;pright=0
    for p in eachindex(panels)
        _same_point(_panel_endpoint_right(panels[p]),xcorner;tol=tol) && (pleft=p)
        _same_point(_panel_endpoint_left(panels[p]),xcorner;tol=tol) && (pright=p)
    end
    return pleft,pright
end

"""
    rcip_corner_metas(disc::DLPRCIPDiscretization{T,C},billiard;symmetry=nothing,angle_tol::T=T(1e-8),point_tol::T=T(1e-8)) where {T<:Real,C}

High-level function to detect physical corners and build local RCIP corner metadata for the whole boundary.

Corners are detected on the active boundary representation: the full boundary
without symmetry, and the desymmetrized boundary when symmetry is used.
"""
function rcip_corner_metas(disc::DLPRCIPDiscretization{T,C},billiard;symmetry=nothing,angle_tol::T=T(1e-8),point_tol::T=T(1e-8)) where {T<:Real,C}
    panels=disc.pdata.panels
    ngl=disc.pdata.ngl
    Np=length(panels)
    wrap(p)=mod(p-1,Np)+1
    boundary=isnothing(symmetry) ? billiard.full_boundary : billiard.desymmetrized_full_boundary
    comps=_rcip_boundary_components(boundary)
    length(comps)==1 || error("DLP_rcip corner detection currently expects one connected component.")
    is_open_reduced=!isnothing(symmetry) # name for when we are looking at a desymmetrized boundary which may have open corners that are not true physical corners of the full boundary
    xcorners=is_open_reduced ? _physical_corner_points_open(comps[1],T;angle_tol=angle_tol) : _physical_corner_points(comps[1],T;angle_tol=angle_tol)
    metas=DLPRCIPCornerMeta{T}[]
    used_pairs=Set{Tuple{Int,Int}}()
    cid=0
    # For each detected corner of the active boundary representation:
    # - full boundary: periodic closed boundary, so patch panels may wrap around;
    # - desymmetrized boundary: open boundary, so endpoints are not treated as
    #   RCIP corners and patch panels must exist on both sides without wrapping.
    for xc in xcorners
        pL,pR=_find_corner_panel_pair(panels,xc;tol=point_tol)
        (pL==0 || pR==0) && continue
        key=(pL,pR)
        key in used_pairs && continue # duplicate corner should not happen but just in case
        if is_open_reduced
            (pL<=1 || pR>=Np) && continue # for open corners we won't have periodicity to wrap around, so we need to check if the panels are valid before collecting the patch panel ids
            # Example: in a desymmetrized rectangle where only 1 corner is true and the other 2 are Dirichlet symmetry edges
            pids=[pL-1,pL,pR,pR+1]
        else
            pids=[wrap(pL-1),pL,pR,wrap(pR+1)]
        end
        push!(used_pairs,key)
        cid+=1
        push!(metas,DLPRCIPCornerMeta{T}(cid,pL,pR,pids,_nodes_for_panels(pids,ngl),xc))
    end
    return metas
end

############################
##### RCIP LOCAL PATCH #####
############################

# ============================================================
# Helsing RCIP type-b corner patch indexing
#
# RCIP local recursion follows Helsing's "type-b" corner construction.
#
# Starting from the coarse 4-panel boundary patch surrounding one physical
# corner,
#
#     [ outer-left | inner-left ]  CORNER  [ inner-right | outer-right ]
#
# the two panels adjacent to the corner are dyadically split, producing the
# refined 6-panel type-b patch:
#
#     [ outer-left | inner-left(a) | inner-left(b) |
#                    inner-right(a) | inner-right(b) | outer-right ]
#
# Each panel carries `ngl` Gauss-Legendre quadrature nodes, so:
#
#     coarse patch size   = 4 * ngl
#     type-b patch size   = 6 * ngl
#
# The RCIP recursion separates the local operator into:
#
#   inner block:
#       the four corner-adjacent panels inner-left(a), inner-left(b),
#       inner-right(a), inner-right(b)
#       These contain the singular corner interactions and define the
#       compressed unknown transform R.
#
#   outer block:
#       the two interface panels outer-left, outer-right
#       These couple the local corner patch back to the global boundary
#       discretization and are eliminated in the Schur-complement recursion.
#
# Node ordering in the 6-panel patch:
#
#     1:ngl                  -> outer-left
#     ngl+1:2ngl            -> inner-left(a)
#     2ngl+1:3ngl           -> inner-left(b)
#     3ngl+1:4ngl           -> inner-right(a)
#     4ngl+1:5ngl           -> inner-right(b)
#     5ngl+1:6ngl           -> outer-right
#
# Hence:
#
#     inner_typeb = ngl+1 : 5ngl
#     outer_typeb = [1:ngl ; 5ngl+1:6ngl]
# ============================================================

# Number of nodes in the four-panel coarse RCIP corner patch.
@inline ncoarse(ngl::Int)=4*ngl
# Number of nodes in the six-panel refined Helsing type-b patch.
@inline ntypeb(ngl::Int)=6*ngl
# Node indices belonging to the four inner (corner-singular) panels.
@inline inner_typeb(ngl::Int)=(ngl+1):(5*ngl)
# Node indices belonging to the two outer/interface panels.
@inline outer_typeb(ngl::Int)=vcat(1:ngl,(5*ngl+1):(6*ngl))

# Return the subpanel of `p` corresponding to local interval `[a,b]⊂[0,1]`.
@inline function subdivide_panel(p::DLPRCIPPanel{T,C},a::Real,b::Real) where {T<:Real,C}
    aa=T(a);bb=T(b)
    u0=(one(T)-aa)*p.u0+aa*p.u1
    u1=(one(T)-bb)*p.u0+bb*p.u1
    return DLPRCIPPanel{T,C}(p.curve,u0,u1,p.id,p.piece_id,p.comp_id,p.is_physical)
end

# Scale panel `p` about parameter value `uc`. This is used to build geometrically similar local RCIP patches around a corner
# at dyadically decreasing scales.
@inline function scaled_panel_about_u(p::DLPRCIPPanel{T,C},uc::T,scale::T) where {T<:Real,C}
    u0=uc+scale*(p.u0-uc)
    u1=uc+scale*(p.u1-uc)
    return DLPRCIPPanel{T,C}(p.curve,u0,u1,p.id,p.piece_id,p.comp_id,p.is_physical)
end

"""

    typeb_patch_from_four(pids::Vector{Int},panels::Vector{DLPRCIPPanel{T,C}},scale::T) where {T<:Real,C}

Build Helsing's six-panel type-b patch from the four coarse panels surrounding
one corner.
The input `pids` must be ordered as
    [outer-left, inner-left, inner-right, outer-right]
where `inner-left` ends at the corner and `inner-right` starts at the corner.
The two inner panels are split into halves, giving the local boundary ordering
    outer-left,
    inner-left outer half,
    inner-left corner half,
    inner-right corner half,
    inner-right outer half,
    outer-right.
For `scale == 1`, this returns the physical type-b patch at the current coarse
scale. For `scale < 1`, the panels are contracted toward the corner in their
own curve parameters. The left-side panels are scaled about `pLin.u1`, and the
right-side panels about `pRin.u0`, which are the two parameter values
representing the same physical corner.
"""
function typeb_patch_from_four(pids::Vector{Int},panels::Vector{DLPRCIPPanel{T,C}},scale::T) where {T<:Real,C}
    pLout=panels[pids[1]]
    pLin=panels[pids[2]]
    pRin=panels[pids[3]]
    pRout=panels[pids[4]]
    ucL=pLin.u1
    ucR=pRin.u0
    ps=DLPRCIPPanel{T,C}[pLout,subdivide_panel(pLin,0,0.5),subdivide_panel(pLin,0.5,1),subdivide_panel(pRin,0,0.5),subdivide_panel(pRin,0.5,1),pRout]
    scale==one(T) && return ps
    return DLPRCIPPanel{T,C}[scaled_panel_about_u(ps[1],ucL,scale),scaled_panel_about_u(ps[2],ucL,scale),scaled_panel_about_u(ps[3],ucL,scale),scaled_panel_about_u(ps[4],ucR,scale),scaled_panel_about_u(ps[5],ucR,scale),scaled_panel_about_u(ps[6],ucR,scale)]
end

"""

    blockdiag_dense(blocks...)

Construct a dense block-diagonal matrix from the input matrices.
Given blocks B₁ ∈ C^{m₁×n₁},  B₂ ∈ C^{m₂×n₂},  ...,  B_k ∈ C^{m_k×n_k},
this returns the dense matrix A = diag(B₁, B₂, ..., B_k)

    [ B₁   0    ⋯    0

      0    B₂   ⋯    0

      ⋮    ⋮    ⋱    ⋮

      0    0    ⋯    B_k ].
The total matrix size is
    (m₁ + ⋯ + m_k) × (n₁ + ⋯ + n_k).
This helper is used to assemble RCIP prolongation operators from independent
panelwise interpolation blocks.
For Helsing's type-b patch, the prolongation from the coarse four-panel patch
    [ outer-left | inner-left | inner-right | outer-right ]
to the refined six-panel patch
    [ outer-left | split(inner-left) | split(inner-right) | outer-right ]
acts independently on each boundary segment:
- the outer panels are unchanged, so they contribute identity blocks,
- the two inner panels are dyadically refined, so they contribute
  interpolation blocks.
Therefore the prolongation has the block structure
    P = blockdiag(I, IP, IP, I),
where:
    I   : identity on unchanged outer panels,
    IP  : interpolation from one coarse panel to its two child panels.
The weighted prolongation used in RCIP is constructed analogously.

"""
function blockdiag_dense(blocks...)
    nr=sum(size(B,1) for B in blocks)
    nc=sum(size(B,2) for B in blocks)
    A=zeros(promote_type(map(eltype,blocks)...),nr,nc)
    r0=0;c0=0
    for B in blocks
        r,c=size(B)
        @views A[r0+1:r0+r,c0+1:c0+c].=B
        r0+=r;c0+=c
    end
    return A
end

"""
    bary_weights(x::Vector{T}) where {T<:Real}

Compute barycentric interpolation weights for nodes
    x₁,...,x_n.
The barycentric weights are
    λ_j = 1 / ∏_{m≠j} (x_j - x_m).
They define the Lagrange polynomials:
    l_j(x) = [λ_j/(x-x_j)] / Σ_m [λ_m/(x-x_m)].
Compared with solving a Vandermonde system, this is usually more stable for
constructing interpolation matrices at moderate or high `ngl`.
"""
function bary_weights(x::Vector{T}) where {T<:Real}
    n=length(x)
    λ=ones(T,n)
    @inbounds for j in 1:n
        p=one(T)
        for m in 1:n
            m==j && continue
            p*=x[j]-x[m]
        end
        λ[j]=inv(p)
    end
    return λ
end

"""
    interp_matrix(xsrc::Vector{T},xdst::Vector{T}) where {T<:Real}

Build the interpolation matrix P mapping values on source nodes `xsrc`
to values on destination nodes `xdst`.
If `fsrc[j] = f(xsrc[j])`, then
    fdst ≈ P * fsrc,
where
    fdst[i] ≈ f(xdst[i]).
The entries are the Lagrange basis functions
    P[i,j] = l_j(xdst[i]).
Using barycentric interpolation,
    l_j(x) = [λ_j/(x-x_j)] / Σ_m [λ_m/(x-x_m)].
If a destination point coincides with a source point, the corresponding row is
set to a unit vector.
In RCIP, this maps one coarse Gauss panel to two half-panels. That is,
    xsrc ∈ [-1,1],
while
    xdst ∈ [-1,0] ∪ [0,1].
So the output matrix has size 2ngl × ngl.
"""
function interp_matrix(xsrc::Vector{T},xdst::Vector{T}) where {T<:Real}
    n=length(xsrc);m=length(xdst)
    λ=bary_weights(xsrc)
    P=zeros(T,m,n)
    @inbounds for i in 1:m
        hit=0
        for j in 1:n
            if abs(xdst[i]-xsrc[j])<=10*eps(T)*max(one(T),abs(xsrc[j]))
                hit=j
                break
            end
        end
        if hit!=0
            P[i,hit]=one(T)
        else
            den=zero(T)
            for j in 1:n
                den+=λ[j]/(xdst[i]-xsrc[j])
            end
            for j in 1:n
                P[i,j]=(λ[j]/(xdst[i]-xsrc[j]))/den
            end
        end
    end
    return P
end

"""
    Pbc_PWbc(::Type{T},ngl::Int) where {T<:Real}

Construct the Helsing RCIP prolongation matrices P and PW.
The Gauss nodes on one panel are
    x_j ∈ [-1,1],   weights w_j.
After dyadic refinement, one panel becomes two panels with local nodes
    x_left  = (x_j - 1)/2 ∈ [-1,0],
    x_right = (x_j + 1)/2 ∈ [0,1],
and weights
    w_left = w_j/2,
    w_right = w_j/2.
The interpolation matrix
    IP : values on one parent panel → values on two child panels
has size
    2*ngl × ngl.
The weighted prolongation is
    IPW = diag(w_child) IP diag(1/w_parent).
Represents the adjoint-compatible map for weighted Nyström unknowns.
The type-b patch has six panels. The two outer panels are not refined in this
prolongation step, while the two central parent panels are split. Therefore
    P  = diag(I, IP,  IP,  I),
    PW = diag(I, IPW, IPW, I).
Both have size
    6*ngl × 4*ngl.
They map compressed/coarse four-panel data to the six-panel refined type-b patch.
"""
function Pbc_PWbc(::Type{T},ngl::Int) where {T<:Real}
    x,w=gausslegendre(ngl)
    x=T.(x);w=T.(w)
    x2=vcat((x.-one(T))./T(2),(x.+one(T))./T(2))
    w2=vcat(w./T(2),w./T(2))
    IP=interp_matrix(x,x2)
    IPW=IP.*(w2*(one(T)./w)')
    Igl=Matrix{T}(I,ngl,ngl)
    P=blockdiag_dense(Igl,IP,IP,Igl)
    PW=blockdiag_dense(Igl,IPW,IPW,Igl)
    return Complex{T}.(P),Complex{T}.(PW)
end

"""
    Icirc(::Type{T},ngl::Int) where {T<:Real}

Construct the type-b identity mask `I°`.
It is identity on the outer/interface panels and zero on the inner corner block.
In the full RCIP recursion it appears in
    I° + K° + frame(R^{-1}).
"""
function Icirc(::Type{T},ngl::Int) where {T<:Real}
    nt=ntypeb(ngl)
    A=Matrix{Complex{T}}(I,nt,nt)
    @inbounds for i in inner_typeb(ngl)
        A[i,i]=zero(Complex{T})
    end
    return A
end

##############################
##### RCIP CACHE OBJECTS #####
##############################

#Stores the local type-b discretization and reusable matrices:
# K: local operator `K = -D`,
# Kstar: singular inner block, used in the recursion,
# Kcirc: remainder with the inner block removed, used in the recursion,
# Awork: temporary work matrix used in the recursion.
struct DLPRCIPLevelCache{T<:Real,C}
    scale::T
    pts::DLPRCIPDiscretization{T,C}
    K::Matrix{Complex{T}}
    Kstar::Matrix{Complex{T}}
    Kcirc::Matrix{Complex{T}}
    Awork::Matrix{Complex{T}}
end

# Reusable RCIP cache for one physical corner.
# Stores all refinement levels, prolongation matrices P and PW, and the
# coarse identity matrix used in the local recursion.
struct DLPRCIPCornerCache{T<:Real,C}
    meta::DLPRCIPCornerMeta{T}
    levels::Vector{DLPRCIPLevelCache{T,C}}
    P::Matrix{Complex{T}}
    PW::Matrix{Complex{T}}
    Icoarse::Matrix{Complex{T}}
    Tio::Matrix{Complex{T}}
    Tip::Matrix{Complex{T}}
    S::Matrix{Complex{T}}
    RHS::Matrix{Complex{T}}
    Xo::Matrix{Complex{T}}
    Xi::Matrix{Complex{T}}
    Rnew::Matrix{Complex{T}}
    KRloc::Matrix{Complex{T}}
    Rbest::Matrix{Complex{T}}
end

# Contains the global discretization, all corner caches, global matrices K,
# R, Kcirc, temporary work storage, local corner transforms Rlocs, and the
# cached logarithmic product weights.
struct DLPRCIPWorkspace{T<:Real,C,Sym}
    pts::DLPRCIPDiscretization{T,C}
    metas::Vector{DLPRCIPCornerMeta{T}}
    corners::Vector{DLPRCIPCornerCache{T,C}}
    K::Matrix{Complex{T}}
    Kphys::Matrix{Complex{T}}
    R::Matrix{Complex{T}}
    Kcirc::Matrix{Complex{T}}
    Awork::Matrix{Complex{T}}
    Rlocs::Vector{Matrix{Complex{T}}}
    Λ::Matrix{T}
    Λlocal::Matrix{T}
    nsub::Int
    symmetry::Sym
    ξ::Vector{T}
    ω::Vector{T}
end

"""
    make_level_cache(pids::Vector{Int}, panels::Vector{DLPRCIPPanel{T,C}}, scale::T, ngl::Int, billiard::Bi) where {T<:Real,C,Bi<:AbsBilliard}

Construct one local Helsing type-b RCIP refinement level.

The four coarse panels indexed by `pids` are converted into the six-panel
type-b patch

    outer-left | split-left | split-right | outer-right,

then optionally scaled toward the corner by `scale`. The resulting patch is
discretized with `ngl` Gauss-Legendre nodes per panel, giving

    ntypeb(ngl) = 6ngl

local unknowns.

The returned `DLPRCIPLevelCache` preallocates the local matrices

    K, Kstar, Kcirc, Awork ∈ C^{6ngl×6ngl},

where later

    K = -D,
    Kstar = inner-inner singular block,
    Kcirc = K with the inner-inner block removed.
"""
function make_level_cache(pids::Vector{Int},panels::Vector{DLPRCIPPanel{T,C}},scale::T,ngl::Int,billiard::Bi) where {T<:Real,C,Bi<:AbsBilliard}
    ps=typeb_patch_from_four(pids,panels,scale)
    pts=_discretize_rcip_panels(ps,ngl,billiard)
    nt=ntypeb(ngl)
    return DLPRCIPLevelCache{T,C}(scale,pts,zeros(Complex{T},nt,nt),zeros(Complex{T},nt,nt),zeros(Complex{T},nt,nt),zeros(Complex{T},nt,nt))
end

"""
    make_corner_cache(meta::DLPRCIPCornerMeta{T}, pts::DLPRCIPDiscretization{T,C}, nsub::Int, billiard::Bi) where {T<:Real,C,Bi<:AbsBilliard}

Construct all reusable local RCIP data for one physical corner.
The cache contains `nsub+1` type-b refinement levels. The first level is the
smallest patch, with scale
    2^(-(nsub-1)),
and the following levels grow dyadically outward until scale `1`.
It also constructs the Helsing prolongation matrices
    P, PW ∈ C^{6*ngl×4*ngl},
and the four-panel coarse identity
    Icoarse ∈ C^{4*ngl×4*ngl},
used in the local recursion for the compressed corner transform `Rloc`.
"""
function make_corner_cache(meta::DLPRCIPCornerMeta{T},pts::DLPRCIPDiscretization{T,C},nsub::Int,billiard::Bi) where {T<:Real,C,Bi<:AbsBilliard}
    ngl=pts.pdata.ngl
    levels=Vector{DLPRCIPLevelCache{T,C}}(undef,nsub+1)
    scale0=T(2)^(-(nsub-1))
    levels[1]=make_level_cache(meta.patch_panels,pts.pdata.panels,scale0,ngl,billiard)
    for lev in 1:nsub
        levels[lev+1]=make_level_cache(meta.patch_panels,pts.pdata.panels,T(2)^(-(nsub-lev)),ngl,billiard)
    end
    P,PW=Pbc_PWbc(T,ngl)
    Icoarse=Matrix{Complex{T}}(I,ncoarse(ngl),ncoarse(ngl))
    no=2*ngl
    nc=4*ngl
    N=boundary_matrix_size(pts) # for KRloc
    return DLPRCIPCornerCache{T,C}(meta,levels,P,PW,Icoarse,zeros(Complex{T},nc,no),zeros(Complex{T},nc,nc),zeros(Complex{T},no,no),zeros(Complex{T},no,nc),zeros(Complex{T},no,nc),zeros(Complex{T},nc,nc),zeros(Complex{T},nc,nc),zeros(Complex{T},N,nc),zeros(Complex{T},nc,nc))
end

"""
    make_dlp_rcip_workspace(solver::DLP_rcip{T,Bi,Sym},pts::DLPRCIPDiscretization{T,C};corner_angle_tol::T=T(1e-8),point_tol::T=T(1e-8)) where {T<:Real,Bi,Sym,C}

Allocate the reusable global RCIP workspace for a fixed discretization.
This performs all geometry-independent setup for a wavenumber sweep:
  1. builds the same-panel logarithmic product weights `Λ`,
  2. detects physical corners and matches them to coarse panels,
  3. builds one `DLPRCIPCornerCache` per corner,
  4. allocates global dense matrices `K`, `R`, `Kcirc`, `Awork`,
  5. allocates one local compressed transform `Rloc` per corner.
Both `Λ` and `Λlocal` are currently identical because all local type-b patches
use the same `ngl` Gauss rule as the global panels.
"""
function make_dlp_rcip_workspace(solver::DLP_rcip{T,Bi,Sym},pts::DLPRCIPDiscretization{T,C};corner_angle_tol::T=T(1e-8),point_tol::T=T(1e-8)) where {T<:Real,Bi,Sym,C}
    N=boundary_matrix_size(pts)
    ngl=pts.pdata.ngl
    ξ0,ω0=gausslegendre(ngl)
    ξ=T.(ξ0);ω=T.(ω0)
    Λ=log_weights_matrix(T,T.(ξ);prec=solver.logquad_prec)
    metas=rcip_corner_metas(pts,solver.billiard;symmetry=solver.symmetry,angle_tol=corner_angle_tol,point_tol=point_tol)
    corners=[make_corner_cache(m,pts,solver.nsub,solver.billiard) for m in metas]
    Rlocs=[Matrix{Complex{T}}(I,ncoarse(ngl),ncoarse(ngl)) for _ in corners]
    return DLPRCIPWorkspace{T,C,Sym}(pts,metas,corners,zeros(Complex{T},N,N),zeros(Complex{T},N,N),Matrix{Complex{T}}(I,N,N),zeros(Complex{T},N,N),zeros(Complex{T},N,N),Rlocs,Λ,Λ,solver.nsub,solver.symmetry,ξ,ω)
end

# Return true iff every real and imaginary part of every entry of A is finite.
@inline finite_matrix(A)=all(z->isfinite(real(z))&&isfinite(imag(z)),A)

"""
    split_helsing_typeb!(Kstar,Kcirc,K,ngl::Int)

Split a local six-panel Helsing type-b operator into its RCIP singular and
regular parts.
The type-b patch has node ordering
    outer-left | inner-left-a | inner-left-b | inner-right-a | inner-right-b | outer-right
with `ngl` nodes per panel. Hence
    ii = inner_typeb(ngl) = ngl+1 : 5*ngl
selects the four panels adjacent to the corner. These are the only panels
whose mutual interactions contain the unresolved corner singular behavior.
Given the local matrix operator `K`, this function forms
    Kstar = K restricted to the inner-inner block,
    Kcirc = K with the inner-inner block removed.
Equivalently,
    Kstar[ii,ii] = K[ii,ii],
    Kcirc = K,
    Kcirc[ii,ii] = 0.
In Helsing's RCIP notation, `Kstar` is the local singular corner operator
that is compressed into the transformed inverse density, while `Kcirc`
contains the smooth remainder and the coupling to the two outer interface
panels.

The operation is performed in-place and returns `(Kstar,Kcirc)`.
"""
function split_helsing_typeb!(Kstar,Kcirc,K,ngl::Int)
    fill!(Kstar,zero(eltype(Kstar)))
    Kcirc.=K # start with Kcirc = K, then zero out the inner block to get Kcirc = K with inner block removed
    ii=inner_typeb(ngl) # indices of the inner block
    @views Kstar[ii,ii].=K[ii,ii] # Kstar = K restricted to inner block
    @views Kcirc[ii,ii].=zero(eltype(Kcirc)) # zero out inner block in Kcirc
    return Kstar,Kcirc
end

"""
    assemble_typeb_K!(lev,k,Λlocal,ξ,ω;use_panel_logquad=true,multithreaded=true)

Assemble the local RCIP operator for one six-panel type-b refinement level.
The level `lev` stores a geometrically scaled local patch around one corner.
First the physical double-layer Nyström matrix `D` is assembled on this local
patch. The matrix is then converted to the RCIP sign convention
    K = -D,
so that the local Fredholm operator is written as
    I + K.
After assembly, `K` is split into Helsing's singular and regular pieces,
    K = Kstar + Kcirc,
where `Kstar` contains only the four-panel inner corner block and `Kcirc`
contains all remaining interactions.
The matrices are stored in-place as
    lev.K      = K,
    lev.Kstar  = singular inner-inner block,
    lev.Kcirc  = regular/interface remainder.
This function is called at every dyadic RCIP level during the local corner
recursion.
"""
function assemble_typeb_K!(lev::DLPRCIPLevelCache{T,C},k::T,Λlocal::Matrix{T},ξ::Vector{T},ω::Vector{T};use_panel_logquad::Bool=true,multithreaded::Bool=true) where {T<:Real,C}
    assemble_dlp_rcip!(lev.K,lev.pts,k,Λlocal,ξ,ω;symmetry=nothing,use_panel_logquad=use_panel_logquad,multithreaded=multithreaded)
    lev.K.*=-one(T)
    split_helsing_typeb!(lev.Kstar,lev.Kcirc,lev.K,lev.pts.pdata.ngl)
    return lev.Kstar,lev.Kcirc
end

"""
    assemble_physical_K_block!(Kblk,disc,rowids,colids,k,Λ,ξ,ω;use_panel_logquad=true)

Assemble a selected physical block of the global RCIP operator `K = -D`.
The rows and columns are specified by global node index sets `rowids` and
`colids`. This is mainly used after the global matrix has been assembled to
remove the uncompressed singular corner block from `Kcirc`.
Mathematically, this computes
    Kblk[α,β] = -D[rowids[α], colids[β]],
using the same DLP kernel, diagonal convention, quadrature weights, and
same-panel logarithmic product quadrature as the global assembly.
For same-panel interactions the DLP kernel is split into a logarithmic part
and a smooth remainder,
    D[i,j] = Λ[il,jl] * L1 + ω[jl] * L2
where `Λ` contains the precomputed product-integration weights. For different
panels the ordinary weighted off-diagonal DLP kernel is used.
The result is written in-place into `Kblk`.
"""
function assemble_physical_K_block!(Kblk::AbstractMatrix{Complex{T}},disc::DLPRCIPDiscretization{T,C},rowids::AbstractVector{Int},colids::AbstractVector{Int},k::T,Λ::Matrix{T},ξ::Vector{T},ω::Vector{T};use_panel_logquad::Bool=true) where {T<:Real,C}
    xy=disc.bp.xy
    normal=disc.bp.normal
    ds=disc.bp.ds
    κ=disc.bp.curvature
    panel_id=disc.pdata.panel_id
    local_id=disc.pdata.local_id
    fill!(Kblk,zero(Complex{T}))
    @inbounds for β in eachindex(colids)
        j=colids[β]
        qj=xy[j];nj=normal[j]
        xj=qj[1];yj=qj[2];nxj=nj[1];nyj=nj[2]
        dsj=ds[j];pj=panel_id[j];jl=local_id[j]
        speed_half_j=dsj/ω[jl]
        for α in eachindex(rowids)
            i=rowids[α]
            qi=xy[i];xi=qi[1];yi=qi[2]
            if i==j
                Kblk[α,β]-=dlp_diag(T,dsj,κ[j])
            else
                r,q=source_cos(xi,yi,xj,yj,nxj,nyj)
                if use_panel_logquad && panel_id[i]==pj
                    il=local_id[i]
                    du=abs(ξ[il]-ξ[jl])
                    full=Complex{T}(zero(T),k/2)*H(1,k*r)*q*speed_half_j
                    L1=-(k*INV_PI)*J(1,k*r)*q*speed_half_j
                    L2=full-L1*log(du)
                    Kblk[α,β]-=Λ[il,jl]*L1+ω[jl]*L2
                else
                    Kblk[α,β]-=dlp_regular(T,k,r,q,dsj)
                end
            end
        end
    end
    return Kblk
end

"""
    rcip_R_for_corner!(cc,Rloc,k,Λlocal,ξ,ω;use_panel_logquad=true,stop_tol=1e-13,min_scale=5e-14,multithreaded=true)

Compute Helsing's compressed inverse-density transform for one corner.
For the local corner equation
    (I + Kstar + Kcirc)ρ = f,
RCIP eliminates the unresolved corner singularity by writing the physical
density locally as
    ρ = R ρ̃,
where `ρ̃` is a transformed coarse density and `R` contains the singular
corner behavior. The output `Rloc` is this local transform on the four-panel
coarse corner patch.
The recursion starts on the smallest dyadically scaled type-b patch with
    R₀ = (I + Kstar)^{-1},
where `Kstar` is the inner-inner singular block. Each larger level updates
`Rloc` by coupling the inner block to the two outer interface panels through
a Schur-complement elimination. The prolongation matrices `P` and `PW` map
between the four-panel coarse representation and the six-panel type-b patch;
`PW` is the quadrature-weight-compatible version used in the weighted Nyström
formulation.
The update is repeated outward through the precomputed dyadic levels until
either the relative change in `Rloc` is below `stop_tol` or the patch scale
falls below the cutoff.
The final `Rloc ∈ C^{4*ngl × 4*ngl}` maps compressed corner unknowns to the
resolved physical corner density on the coarse corner patch.
"""
function rcip_R_for_corner!(cc::DLPRCIPCornerCache{T,C},Rloc::Matrix{Complex{T}},k::T,Λlocal::Matrix{T},ξ::Vector{T},ω::Vector{T};use_panel_logquad::Bool=true,stop_tol::T=T(1e-13),min_scale::T=T(5e-14),scale_safety::T=T(100),multithreaded::Bool=true) where {T<:Real,C}
    ngl=cc.levels[1].pts.pdata.ngl
    ii=inner_typeb(ngl)
    oo=outer_typeb(ngl)
    scale_floor=scale_safety*min_scale
    istart=findfirst(lev->lev.scale>=scale_floor,cc.levels)
    isnothing(istart) && error("all RCIP levels below safe scale at corner $(cc.meta.cid)")
    lev0=cc.levels[istart]
    assemble_typeb_K!(lev0,k,Λlocal,ξ,ω;use_panel_logquad=use_panel_logquad,multithreaded=multithreaded)
    copyto!(cc.Rnew,cc.Icoarse)
    @views cc.Rnew.+=lev0.Kstar[ii,ii]
    finite_matrix(cc.Rnew) || error("non-finite initial RCIP local system at corner $(cc.meta.cid)")
    F=lu!(cc.Rnew)
    copyto!(Rloc,cc.Icoarse)
    ldiv!(F,Rloc)
    for ell in (istart+1):length(cc.levels)
        lev=cc.levels[ell]
        lev.scale<scale_floor && continue
        assemble_typeb_K!(lev,k,Λlocal,ξ,ω;use_panel_logquad=use_panel_logquad,multithreaded=multithreaded)
        finite_matrix(lev.Kcirc) || error("non-finite Kcirc at corner $(cc.meta.cid), level $ell")
        @views begin
            Koo=lev.Kcirc[oo,oo]
            Koi=lev.Kcirc[oo,ii]
            Kio=lev.Kcirc[ii,oo]
            Po=cc.P[oo,:]
            Pi=cc.P[ii,:]
            PWo=cc.PW[oo,:]
            PWi=cc.PW[ii,:]
        end
        make_identity_matrix!(cc.S)
        cc.S.+=Koo
        mul!(cc.Tio,Rloc,Kio)
        mul!(cc.Tip,Rloc,Pi)
        mul!(cc.S,Koi,cc.Tio,-one(Complex{T}),one(Complex{T}))
        copyto!(cc.RHS,Po)
        mul!(cc.RHS,Koi,cc.Tip,-one(Complex{T}),one(Complex{T}))
        F=lu!(cc.S)
        ldiv!(cc.Xo,F,cc.RHS)
        copyto!(cc.Xi,cc.Tip)
        mul!(cc.Xi,cc.Tio,cc.Xo,-one(Complex{T}),one(Complex{T}))
        mul!(cc.Rnew,adjoint(PWo),cc.Xo)
        mul!(cc.Rnew,adjoint(PWi),cc.Xi,one(Complex{T}),one(Complex{T}))
        rel=relerr_noalloc(cc.Rnew,Rloc)
        copyto!(Rloc,cc.Rnew)
        rel<stop_tol && break
    end
    return Rloc
end

# Diagnostic version of `rcip_R_for_corner!`.
# It performs the same Helsing RCIP recursion for one corner, but prints
# per-level information useful for debugging stability and convergence:
# scale of the local patch,
# relative change of `Rloc`,
# best level reached,
# condition number of the outer Schur system,
# norms of the local blocks,
# norm of the updated transform.
function rcip_R_for_corner_diagnostics!(cc::DLPRCIPCornerCache{T,C},Rloc::Matrix{Complex{T}},k::T,Λlocal::Matrix{T},ξ::Vector{T},ω::Vector{T};use_panel_logquad::Bool=true,stop_tol::T=T(1e-13),min_scale::T=T(5e-14),patience::Int=3,multithreaded::Bool=true) where {T<:Real,C}
    ngl=cc.levels[1].pts.pdata.ngl
    ii=inner_typeb(ngl)
    oo=outer_typeb(ngl)
    lev0=cc.levels[1]
    @info "RCIP corner start" corner=cc.meta.cid k=k ngl=ngl nlevels=length(cc.levels) ncoarse=ncoarse(ngl) ntypeb=ntypeb(ngl)
    assemble_typeb_K!(lev0,k,Λlocal,ξ,ω;use_panel_logquad=use_panel_logquad,multithreaded=multithreaded)
    copyto!(cc.Rnew,cc.Icoarse)
    @views cc.Rnew.+=lev0.Kstar[ii,ii]
    finite_matrix(cc.Rnew) || error("non-finite initial RCIP local system at corner $(cc.meta.cid)")
    @info "RCIP initial solve" corner=cc.meta.cid level=1 scale=lev0.scale norm_Kstar=opnorm(lev0.Kstar[ii,ii]) cond_A0=cond(cc.Rnew)
    F=lu!(cc.Rnew)
    copyto!(Rloc,cc.Icoarse)
    ldiv!(F,Rloc)
    copyto!(cc.Rbest,Rloc)
    best_rel=typemax(T)
    best_level=1
    prev_rel=typemax(T)
    worse_count=0
    stop_reason="completed all levels"
    for ell in 2:length(cc.levels)
        lev=cc.levels[ell]
        if lev.scale<min_scale
            stop_reason="scale below min_scale"
            break
        end
        assemble_typeb_K!(lev,k,Λlocal,ξ,ω;use_panel_logquad=use_panel_logquad,multithreaded=multithreaded)
        finite_matrix(lev.Kcirc) || error("non-finite Kcirc at corner $(cc.meta.cid), level $ell")
        Koo=@view lev.Kcirc[oo,oo]
        Koi=@view lev.Kcirc[oo,ii]
        Kio=@view lev.Kcirc[ii,oo]
        Po=@view cc.P[oo,:]
        Pi=@view cc.P[ii,:]
        PWo=@view cc.PW[oo,:]
        PWi=@view cc.PW[ii,:]
        make_identity_matrix!(cc.S)
        cc.S.+=Koo
        mul!(cc.Tio,Rloc,Kio)
        mul!(cc.Tip,Rloc,Pi)
        mul!(cc.S,Koi,cc.Tio,-one(Complex{T}),one(Complex{T}))
        copyto!(cc.RHS,Po)
        mul!(cc.RHS,Koi,cc.Tip,-one(Complex{T}),one(Complex{T}))
        F=lu!(cc.S)
        ldiv!(cc.Xo,F,cc.RHS)
        copyto!(cc.Xi,cc.Tip)
        mul!(cc.Xi,cc.Tio,cc.Xo,-one(Complex{T}),one(Complex{T}))
        mul!(cc.Rnew,adjoint(PWo),cc.Xo)
        mul!(cc.Rnew,adjoint(PWi),cc.Xi,one(Complex{T}),one(Complex{T}))
        rel=relerr_noalloc(cc.Rnew,Rloc)
        if rel<best_rel
            best_rel=rel
            best_level=ell
            worse_count=0
            copyto!(cc.Rbest,cc.Rnew)
        else
            worse_count+=1
        end
        @info "RCIP level" corner=cc.meta.cid level=ell scale=lev.scale rel=rel prev_rel=prev_rel best_rel=best_rel best_level=best_level worse_count=worse_count cond_S=cond(cc.S) norm_Koo=opnorm(Koo) norm_Koi=opnorm(Koi) norm_Kio=opnorm(Kio) norm_S=opnorm(cc.S) norm_RHS=opnorm(cc.RHS) norm_R=opnorm(Rloc) norm_Rnew=opnorm(cc.Rnew)
        copyto!(Rloc,cc.Rnew)
        prev_rel=rel
        if rel<stop_tol
            stop_reason="rel below stop_tol"
            break
        end
        if worse_count>=patience
            stop_reason="worse_count reached patience"
            break
        end
    end
    copyto!(Rloc,cc.Rbest)
    @info "RCIP corner done" corner=cc.meta.cid stop_reason=stop_reason best_level=best_level best_rel=best_rel final_norm=opnorm(Rloc)
    return Rloc
end

#############################

"""
    build_rcip_blocks!(ws,k;use_panel_logquad=true,verbose=false,multithreaded=true)

Build the global compressed RCIP blocks at wavenumber `k`.
The global physical DLP matrix is first assembled and converted to the RCIP
sign convention
    K = -D.
For each detected corner, the local RCIP transform `Rloc` is computed and
stored in the workspace. The corresponding physical singular block is then
removed from the global operator, producing
    Kcirc = K - local singular corner blocks.
Thus the compressed global Fredholm operator is represented as
    A_rcip = I + Kcirc R,
where `R` is identity away from corners and equal to the local `Rloc` on each
four-panel corner patch.
This function does not form `A_rcip` itself. It prepares the factors `Kcirc`
and the local corner transforms stored in `ws.Rlocs`. Corner patches are
required to be disjoint; overlapping patches indicate that the coarse
panelization is too sparse near corners.
"""
function build_rcip_blocks!(ws::DLPRCIPWorkspace{T,C,Sym},k::T;use_panel_logquad::Bool=true,verbose::Bool=false,multithreaded::Bool=true) where {T<:Real,C,Sym}
    N=boundary_matrix_size(ws.pts)
    assemble_dlp_rcip!(ws.K,ws.pts,k,ws.Λ,ws.ξ,ws.ω;symmetry=ws.symmetry,use_panel_logquad=use_panel_logquad,multithreaded=multithreaded)
    ws.K.*=-one(T)
    ws.Kcirc.=ws.K
    used=falses(N)
    for c in eachindex(ws.corners)
        cc=ws.corners[c]
        ids=cc.meta.patch_nodes
        any(used[ids])&&error("overlapping RCIP patches detected; increase coarse panel count")
        used[ids].=true
        Rloc=ws.Rlocs[c]
        verbose ? rcip_R_for_corner_diagnostics!(cc,Rloc,k,ws.Λlocal,ws.ξ,ws.ω;use_panel_logquad=use_panel_logquad,multithreaded=multithreaded) : rcip_R_for_corner!(cc,Rloc,k,ws.Λlocal,ws.ξ,ws.ω;use_panel_logquad=use_panel_logquad,multithreaded=multithreaded)
        @views begin
            Kp=view(ws.Kphys,1:length(ids),1:length(ids))
            assemble_physical_K_block!(Kp,ws.pts,ids,ids,k,ws.Λ,ws.ξ,ws.ω;use_panel_logquad=use_panel_logquad)
            ws.Kcirc[ids,ids].-=Kp
        end
    end
    return ws.Kcirc
end

"""
    construct_rcip_fredholm!(A,ws,k;use_panel_logquad=true,verbose=false,multithreaded=true)

Construct the full compressed RCIP Fredholm matrix
    A_rcip(k) = I + Kcirc(k) R(k)
in-place.

First `build_rcip_blocks!` assembles the physical operator `K = -D`, computes
the local corner transforms `Rloc`, and removes the corner blocks to
obtain `Kcirc`. Then this routine applies the transform `R` column-wise on
each corner patch:
    A[:,ids] = Kcirc[:,ids] * Rloc,
while columns away from corners remain unchanged. Finally the identity is
added. The result is the dense matrix whose near-null vectors are the compressed
RCIP densities `ρ̃`. The physical DLP layer density is recovered afterward by
    ρ = R ρ̃.
Cost structure:
    global DLP assembly:       O(N²),
    local RCIP recursion:      O(ncorners * nsub * (6ngl)³),
    corner block removal:      O(ncorners * (4ngl)²),
    corner-column update:      O(ncorners * N * (4ngl)²).
For large `k`, `N` grows roughly linearly with `k`, while the local RCIP
corner recursion has fixed size for fixed `ngl`, `nsub`, and number of corners.
"""
function construct_rcip_fredholm!(A::Matrix{Complex{T}},ws::DLPRCIPWorkspace{T,C,Sym},k::T;use_panel_logquad::Bool=true,verbose::Bool=false,multithreaded::Bool=true) where {T<:Real,C,Sym}
    N=boundary_matrix_size(ws.pts)
    @assert size(A)==(N,N)
    build_rcip_blocks!(ws,k;use_panel_logquad=use_panel_logquad,verbose=verbose,multithreaded=multithreaded)
    copyto!(A,ws.Kcirc)
    for c in eachindex(ws.corners)
        ids=ws.corners[c].meta.patch_nodes
        Rloc=ws.Rlocs[c]
        @views begin
            Aids=view(A,:,ids)
            tmp=view(ws.Awork,1:N,1:length(ids))
            mul!(tmp,Aids,Rloc)
            copyto!(Aids,tmp)
        end
    end
    add_identity_matrix!(A)
    return A
end

"""
    construct_matrices!(solver::DLP_rcip{T,Bi,Sym}, basis::AbstractHankelBasis, A::Matrix{Complex{T}}, pts::DLPRCIPDiscretization{T,C}, ws::DLPRCIPWorkspace{T,C,Sym}, k::T; multithreaded::Bool=true, verbose_rcip::Bool=false) where {T<:Real,Bi<:AbsBilliard,Sym,C}

Library-facing in-place matrix constructor using a prebuilt RCIP workspace.

This is the preferred method in a spectral sweep because the expensive
wavenumber-independent data are reused:

    corner metadata,
    type-b level geometries,
    prolongation matrices,
    log-product quadrature weights,
    dense work arrays.

The `basis` argument is unused for DLP-RCIP assembly but kept for compatibility
with the common `construct_matrices!` solver interface.
"""
function construct_matrices!(solver::DLP_rcip{T,Bi,Sym},basis::AbstractHankelBasis,A::Matrix{Complex{T}},pts::DLPRCIPDiscretization{T,C},ws::DLPRCIPWorkspace{T,C,Sym},k::T;multithreaded::Bool=true,verbose_rcip::Bool=false) where {T<:Real,Bi<:AbsBilliard,Sym,C}
    construct_rcip_fredholm!(A,ws,k;use_panel_logquad=solver.use_panel_logquad,verbose=verbose_rcip,multithreaded=multithreaded)
    return A
end

"""
    construct_matrices!(solver::DLP_rcip{T,Bi,Sym}, basis::AbstractHankelBasis, A::Matrix{Complex{T}}, pts::DLPRCIPDiscretization{T,C}, k::T; multithreaded::Bool=true, verbose_rcip::Bool=false) where {T<:Real,Bi<:AbsBilliard,Sym,C}

Convenience in-place constructor that creates a temporary RCIP workspace.

This is suitable for isolated matrix builds. For repeated calls at many `k`,
prefer the method that accepts `ws::DLPRCIPWorkspace`, otherwise the corner
caches and log-product weights are rebuilt every time.
"""
function construct_matrices!(solver::DLP_rcip{T,Bi,Sym},basis::AbstractHankelBasis,A::Matrix{Complex{T}},pts::DLPRCIPDiscretization{T,C},k::T;multithreaded::Bool=true,verbose_rcip::Bool=false) where {T<:Real,Bi<:AbsBilliard,Sym,C}
    ws=make_dlp_rcip_workspace(solver,pts)
    return construct_matrices!(solver,basis,A,pts,ws,k;multithreaded=multithreaded,verbose_rcip=verbose_rcip)
end

"""
    construct_matrices(solver::DLP_rcip{T,Bi,Sym},basis::AbstractHankelBasis,pts::DLPRCIPDiscretization{T,C},ws::DLPRCIPWorkspace{T,C,Sym},k::T;multithreaded::Bool=true,verbose_rcip::Bool=false) where {T<:Real,Bi,Sym,C}

Allocate and return the RCIP Fredholm matrix using an existing workspace.

Equivalent to allocating `A` and calling the workspace-aware
`construct_matrices!`.
"""
function construct_matrices(solver::DLP_rcip{T,Bi,Sym},basis::AbstractHankelBasis,pts::DLPRCIPDiscretization{T,C},ws::DLPRCIPWorkspace{T,C,Sym},k::T;multithreaded::Bool=true,verbose_rcip::Bool=false) where {T<:Real,Bi,Sym,C}
    N=boundary_matrix_size(pts)
    A=Matrix{Complex{T}}(undef,N,N)
    construct_matrices!(solver,basis,A,pts,ws,k;multithreaded=multithreaded,verbose_rcip=verbose_rcip)
    return A
end

"""
    construct_matrices(solver::DLP_rcip{T,Bi,Sym},basis::AbstractHankelBasis,pts::DLPRCIPDiscretization{T,C},k::T;multithreaded::Bool=true,verbose_rcip::Bool=false) where {T<:Real,Bi,Sym,C}

Allocate and return the RCIP Fredholm matrix using a temporary workspace.

This is the highest-level convenience wrapper. It is correct but not optimal
for sweeps over many wavenumbers because `make_dlp_rcip_workspace` is called
inside.
"""
function construct_matrices(solver::DLP_rcip{T,Bi,Sym},basis::AbstractHankelBasis,pts::DLPRCIPDiscretization{T,C},k::T;multithreaded::Bool=true,verbose_rcip::Bool=false) where {T<:Real,Bi,Sym,C}
    ws=make_dlp_rcip_workspace(solver,pts)
    return construct_matrices(solver,basis,pts,ws,k;multithreaded=multithreaded,verbose_rcip=verbose_rcip)
end

"""
    reconstruct_density!(ρ::AbstractVector{Complex{T}},ws::DLPRCIPWorkspace{T,C,Sym},ρtilde::AbstractVector{Complex{T}}) where {T<:Real,C,Sym}

Reconstruct the physical RCIP density ρ = R * ρtilde on the global coarse RCIP nodes.
Here `ρtilde` is the compressed unknown, i.e. the right null vector of

    A_rcip = I + Kcirc * R,

while `ρ` is the physical density to use in layer-potential wavefunction
reconstruction and boundary functions.
"""
function reconstruct_density!(ρ::AbstractVector{Complex{T}},ws::DLPRCIPWorkspace{T,C,Sym},ρtilde::AbstractVector{Complex{T}}) where {T<:Real,C,Sym}
    N=boundary_matrix_size(ws.pts)
    copyto!(ρ,ρtilde)
    for c in eachindex(ws.corners)
        ids=ws.corners[c].meta.patch_nodes
        Rloc=ws.Rlocs[c]
        @views mul!(ρ[ids],Rloc,ρtilde[ids])
    end
    return ρ
end

"""
    reconstruct_adjoint_density!(u, ws, utilde)

Reconstruct the physical adjoint RCIP density from the compressed adjoint
unknown.

If the primal compressed operator is

    A_rcip = I + Kcirc * R,

then the corresponding weighted Nyström adjoint involves the weighted adjoint
of the local RCIP transform. On each corner patch this gives

    u_patch = W^{-1} Rloc' W utilde_patch,

where `W = diag(ds)` is the quadrature-weight matrix on the patch.

This is the density associated with the adjoint DLP formulation. In the
Dirichlet eigenproblem this is the natural boundary-function object,
proportional to the normal derivative of the eigenfunction, not the DLP
layer density used in the primal representation.
"""
function reconstruct_adjoint_density!(u::AbstractVector{Complex{T}},ws::DLPRCIPWorkspace{T,C,Sym},utilde::AbstractVector{Complex{T}}) where {T<:Real,C,Sym}
    copyto!(u,utilde)
    ds=ws.pts.bp.ds
    for c in eachindex(ws.corners)
        ids=ws.corners[c].meta.patch_nodes
        Rloc=ws.Rlocs[c]
        @views begin
            D=Diagonal(ds[ids])
            u[ids].=D\(adjoint(Rloc)*(D*utilde[ids]))
        end
    end
    return u
end

"""
    construct_rcip_adjoint_fredholm!(Aadj, A, ws, k; use_panel_logquad=true, verbose=false, multithreaded=true)

Construct the weighted Nyström adjoint of the RCIP Fredholm matrix.

First builds the primal compressed RCIP matrix

    A = I + Kcirc * R,

at wavenumber `k`. The adjoint matrix is then formed with respect to the
quadrature-weighted boundary inner product

    <f,g>_W = sum_i conj(f_i) g_i ds_i.

Thus the discrete weighted adjoint is

    Aadj = W^{-1} A' W,

with `W = diag(ds)`. Entrywise this is

    Aadj[i,j] = A[j,i] * ds[j] / ds[i].

The resulting matrix is the correct object for computing the adjoint boundary
density, which is the boundary-function/normal-derivative representation used
for SLP wavefunction reconstruction.
"""
function construct_rcip_adjoint_fredholm!(Aadj::Matrix{Complex{T}},A::Matrix{Complex{T}},ws::DLPRCIPWorkspace{T,C,Sym},k::T;use_panel_logquad::Bool=true,verbose::Bool=false,multithreaded::Bool=true) where {T<:Real,C,Sym}
    N=boundary_matrix_size(ws.pts)
    construct_rcip_fredholm!(A,ws,k;use_panel_logquad=use_panel_logquad,verbose=verbose,multithreaded=multithreaded)
    ds=ws.pts.bp.ds
    @inbounds for j in 1:N, i in 1:N
        Aadj[i,j]=A[j,i]*ds[j]/ds[i]
    end
    return Aadj
end

"""
    solve(solver,basis,pts,k;...)

Construct the RCIP Fredholm matrix at wavenumber `k` and solve the
resulting nonlinear eigenproblem using the standard SVD/determinant backend.

This convenience wrapper allocates both the dense matrix and a temporary
RCIP workspace.
"""
function solve(solver::DLP_rcip{T,Bi,Sym},basis::Ba,pts::DLPRCIPDiscretization{T,C},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det_argmin,verbose_rcip::Bool=false) where {T<:Real,Bi<:AbsBilliard,Sym,C,Ba<:AbstractHankelBasis}
    N=boundary_matrix_size(pts)
    A=Matrix{Complex{T}}(undef,N,N)
    ws=make_dlp_rcip_workspace(solver,pts)
    @blas_1 construct_matrices!(solver,basis,A,pts,ws,T(k);multithreaded=multithreaded,verbose_rcip=verbose_rcip)
    @svd_or_det_solve A use_krylov which MAX_BLAS_THREADS
end

"""
    solve(solver,basis,A,pts,k;...)

In-place variant using a caller-provided dense matrix `A`.

A temporary RCIP workspace is still constructed internally.
"""
function solve(solver::DLP_rcip{T,Bi,Sym},basis::Ba,A::AbstractMatrix{Complex{T}},pts::DLPRCIPDiscretization{T,C},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det_argmin,verbose_rcip::Bool=false) where {T<:Real,Bi<:AbsBilliard,Sym,C,Ba<:AbstractHankelBasis}
    ws=make_dlp_rcip_workspace(solver,pts)
    @blas_1 construct_matrices!(solver,basis,A,pts,ws,T(k);multithreaded=multithreaded,verbose_rcip=verbose_rcip)
    @svd_or_det_solve A use_krylov which MAX_BLAS_THREADS
end

"""
    solve(solver,basis,A,pts,ws,k;...)

High-performance in-place solve using a prebuilt RCIP workspace `ws`.

This is the preferred method for sweeps over many wavenumbers since the
corner caches, prolongation operators, and log-product quadrature data
are reused.
"""
function solve(solver::DLP_rcip{T,Bi,Sym},basis::Ba,A::AbstractMatrix{Complex{T}},pts::DLPRCIPDiscretization{T,C},ws::DLPRCIPWorkspace{T,C,Sym},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det_argmin,verbose_rcip::Bool=false) where {T<:Real,Bi<:AbsBilliard,Sym,C,Ba<:AbstractHankelBasis}
    @blas_1 construct_matrices!(solver,basis,A,pts,ws,T(k);multithreaded=multithreaded,verbose_rcip=verbose_rcip)
    @svd_or_det_solve A use_krylov which MAX_BLAS_THREADS
end

"""
    solve_vect(solver::DLP_rcip, billiard, k; multithreaded=true, tol=1e-14, maxiter=2000, krylovdim=40)

Compute the smallest singular vector of the adjoint RCIP DLP formulation at
wavenumber `k`.

The function performs the following steps:

1. Builds the RCIP boundary discretization at `k`.
2. Constructs the reusable RCIP workspace.
3. Forms the primal compressed RCIP Fredholm matrix

       A = I + Kcirc * R.

4. Forms the weighted adjoint matrix

       Aadj = W^{-1} A' W.

5. Computes the smallest right singular vector `utilde` of `Aadj`.
6. Reconstructs the physical adjoint RCIP density

       u = W^{-1} R' W utilde

   on the coarse global boundary nodes.

The returned `u` is the adjoint DLP boundary density. For the Dirichlet
eigenproblem this is the boundary-function representation, proportional to
`∂ₙψ`, and is the appropriate input for SLP-based wavefunction reconstruction.

Returns

    σ, u, pts.bp, ws, A, Aadj

where `σ` is the smallest singular value, `u` is the reconstructed physical
adjoint density, `pts.bp` are the boundary points, `ws` is the RCIP workspace,
and `A`, `Aadj` are the primal and weighted-adjoint matrices.
"""
function solve_vect(solver::DLP_rcip,billiard::Bi,k::T;multithreaded::Bool=true,tol=T(1e-14),maxiter::Int=2000,krylovdim::Int=40) where {T<:Real,Bi<:AbsBilliard}
    pts=evaluate_points(solver,billiard,k)
    ws=make_dlp_rcip_workspace(solver,pts)
    N=boundary_matrix_size(pts)
    A=Matrix{Complex{T}}(undef,N,N)
    Aadj=Matrix{Complex{T}}(undef,N,N)
    construct_rcip_adjoint_fredholm!(Aadj,A,ws,k;use_panel_logquad=solver.use_panel_logquad,multithreaded=multithreaded)
    σ,utilde,_=smallest_nullvec_krylov!(Aadj;nev=1,tol=tol,maxiter=maxiter,krylovdim=krylovdim)
    u=similar(utilde)
    reconstruct_adjoint_density!(u,ws,utilde)
    return σ,u,pts.bp,ws,A,Aadj
end

"""
    solve_vect(solver::DLP_rcip,billiard,ks;multithreaded=true,tol=1e-14,maxiter=2000,krylovdim=40)

Compute adjoint RCIP singular vectors for several wavenumbers using one shared
RCIP discretization/workspace built at `maximum(ks)`. This is more efficient than calling `solve_vect` separately for each `k` because the expensive RCIP setup is amortized over many solves. The returned vectors are all reconstructed on the same physical boundary nodes, so they can be directly compared or used together for subsequent processing.
"""
function solve_vect(solver::DLP_rcip,billiard::Bi,ks::AbstractVector{T};multithreaded::Bool=true,tol::T=T(1e-14),maxiter::Int=2000,krylovdim::Int=40) where {T<:Real,Bi<:AbsBilliard}
    kref=maximum(ks)
    pts=evaluate_points(solver,billiard,kref)
    ws=make_dlp_rcip_workspace(solver,pts)
    N=boundary_matrix_size(pts)
    A=Matrix{Complex{T}}(undef,N,N)
    Aadj=Matrix{Complex{T}}(undef,N,N)
    σs=Vector{T}(undef,length(ks))
    us=Vector{Vector{Complex{T}}}(undef,length(ks))
    for n in eachindex(ks)
        k=ks[n]
        construct_rcip_adjoint_fredholm!(Aadj,A,ws,k;use_panel_logquad=solver.use_panel_logquad,multithreaded=multithreaded)
        σ,utilde,_=smallest_nullvec_krylov!(Aadj;nev=1,tol=tol,maxiter=maxiter,krylovdim=krylovdim)
        u=similar(utilde)
        reconstruct_adjoint_density!(u,ws,utilde)
        σs[n]=σ
        us[n]=u
    end
    return σs,us,pts.bp,ws,A,Aadj
end