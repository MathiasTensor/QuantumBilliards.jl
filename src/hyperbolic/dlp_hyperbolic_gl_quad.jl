# ==============================================================================
# Hyperbolic DLP with panelwise logarithmic product quadrature
# ==============================================================================
#
# This solver discretizes the interior Dirichlet eigenvalue problem in the
# Poincare disk using the source-normal hyperbolic double-layer operator
#
#     D(k)φ(x) = ∫_Γ ∂_{n_y} G_k(d_H(x,y)) φ(y) ds_E(y),
#
# and forms the Fredholm equation
#
#     A(k)φ = (I - D(k))φ = 0.
#
# Although the Green function is hyperbolic, the boundary quadrature uses
# Euclidean normals and Euclidean arclength weights. This follows from the
# conformal form of the Poincare metric,
#
#     ds_H = λ ds_E,        λ(x,y)=2/(1-x²-y²),
#
# together with the cancellation
#
#     ∂ₙᴴG ds_H = ∂ₙᴱG ds_E.
#
# Hence the matrix uses
#
#     pts.normal  = Euclidean outward unit normal,
#     pts.ds      = Euclidean quadrature weight,
#
# while `pts.λ`, `pts.dsH`, and `pts.ξ` are retained for hyperbolic arclength
# placement and postprocessing.
#
# The boundary is split into panels. Panel endpoints are placed approximately
# uniformly in hyperbolic arclength, with
#
#     Npan ≈ b k L_H / (2π ngl),
#
# so the total number of quadrature nodes satisfies
#
#     N ≈ b k L_H / (2π).
#
# Thus `b = pts_scaling_factor` acts as a hyperbolic points-per-wavelength
# parameter. Inside each panel, standard `ngl`-point Gauss-Legendre quadrature
# is used.
#
# Same-panel interactions are weakly singular. Locally, for reference-panel
# coordinates ξ_i, ξ_j ∈ [-1,1], the weighted kernel is split as
#
#     K(ξ_i,ξ_j) = L₁(ξ_i,ξ_j) log|ξ_i-ξ_j| + L₂(ξ_i,ξ_j).
#
# Here
#
#     K(ξ_i,ξ_j)
#       = ∂_{n_y}G_k(d_H(x_i,x_j)) J(ξ_j),
#
# with
#
#     J(ξ_j) = ds_E/dξ.
#
# The hyperbolic Green function has the local logarithmic structure
#
#     G_k(d) = A(d) log(sinh²(d/2)) + B(d),
#
# (or the Martensen/Kusmaul/Kress logarithmic decomposition, just changes L2
#  which does not have simple closed form) where
#
#     A(d) = -(1/(4π)) P_ν(cosh d),
#     ν = -1/2 + i k.
#
# Differentiating gives
#
#     ∂_{n_y}G_k(d)
#       = A'(d) ∂_{n_y}d log(sinh²(d/2))
#         + smooth,
#
# and since, on one smooth panel,
#
#     log(sinh²(d/2)) = 2log|ξ_i-ξ_j| + smooth,
#
# the coefficient of log|ξ_i-ξ_j| is
#
#     L₁(ξ_i,ξ_j)
#       = 2 A'(d_H(x_i,x_j)) ∂_{n_y}d_H(x_i,x_j) J(ξ_j).
#
# The smooth remainder is defined by subtraction:
#
#     L₂(ξ_i,ξ_j)
#       = K(ξ_i,ξ_j) - L₁(ξ_i,ξ_j) log|ξ_i-ξ_j|.
#
# The logarithmic product weights Λ are built for exactly
#
#     ∫_{-1}^{1} p(ξ) log|ξ-ξ_i| dξ
#       ≈ Σ_j Λ[i,j] p(ξ_j),
#
# so same-panel entries are assembled as
#
#     Λ[i,j] L₁ + ω[j] L₂.
#
# Different-panel interactions are smooth and use the ordinary
# Gauss-Legendre Nyström rule.
#
# The hyperbolic Green kernel and its source-normal derivative are evaluated
# through Legendre-Q Taylor tables. The logarithmic coefficient `L₁` is obtained
# from the corresponding P/logarithmic-coefficient table. These tables depend on
# the spectral parameter `k` and are stored in `DLPHypLogWorkspace`.
#
# If a reflection or rotation symmetry is supplied, the discretization is built
# on the desymmetrized boundary. The missing copies are added by the method of
# images. These image terms are geometrically separated from the physical source
# copy, so they are always regular off-panel contributions and need no
# logarithmic product correction.
#
# For eigenfunction postprocessing, the physical boundary function is obtained
# from the weighted adjoint problem
#
#     D†[i,j] = D[j,i] ds[j]/ds[i],
#     A† = I - D†,
#
# whose null vector is proportional to the boundary normal derivative
#
#     u(s) = ∂ₙψ(s).
#
# This is the vector used for Poincare-Husimi plots, IPR/entropy diagnostics,
# and SLP-based wavefunction reconstruction.
#
# MO 27/5/26
# ==============================================================================

const TWO_PI=2*pi
const INV_TWO_PI=1/TWO_PI
const INV_FOUR_PI=1/(2*TWO_PI)

"""
    DLP_hyperbolic_log_product{T,Bi,Sym} <: SweepSolver

Panelwise logarithmic-product quadrature solver for the source-normal
hyperbolic double-layer operator in the Poincare disk.

The boundary is split into Gauss-Legendre panels whose endpoints are distributed
approximately uniformly in hyperbolic arclength. On each panel, `ngl`
Gauss-Legendre nodes are used. Same-panel logarithmic singularities are treated
by product integration with precomputed log-weight matrix `Λ`, while different
panels use the ordinary smooth Nyström rule.

The continuous operator is

    D(k)φ(x) = ∫Γ ∂_{n_y}G_k(d_H(x,y)) φ(y) ds_E(y),

using Euclidean normals and Euclidean arclength weights, relying on the
conformal cancellation `∂ₙᴴG ds_H = ∂ₙᴱG ds_E`.

Fields:
- `pts_scaling_factor`: points-per-wavelength-like resolution factor.
- `dim_scaling_factor`: compatibility field for Beyn/window planning.
- `eps`: numerical tolerance.
- `min_dim`, `min_pts`: minimum dimension / point-count safeguards.
- `billiard`: billiard geometry.
- `symmetry`: optional reflection or rotation symmetry.
- `ngl`: Gauss-Legendre nodes per panel.
- `near_panels`: number of near panels for special quadrature treatment.
- `logquad_prec`: BigFloat precision for logarithmic product weights.
- `mp_dps`: multiprecision digits for Legendre-Q/P Taylor table generation.
"""
struct DLP_hyperbolic_log_product{T<:Real,Bi<:AbsBilliard,Sym}<:SweepSolver
    pts_scaling_factor::Vector{T}
    dim_scaling_factor::T
    eps::T
    min_dim::Int64
    min_pts::Int64
    billiard::Bi
    symmetry::Sym
    ngl::Int
    near_panels::Int
    logquad_prec::Int
    mp_dps::Int
end

"""
    DLP_hyperbolic_log_product(pts_scaling_factor,billiard; kwargs...)

Construct a hyperbolic DLP solver using panelwise Gauss-Legendre quadrature and
same-panel logarithmic product integration.

The resolution is controlled by

    Npan ≈ pts_scaling_factor * k * L_H / (2π * ngl),

where `L_H` is the hyperbolic length of a curve piece. Thus the total number of
nodes scales approximately as

    N ≈ pts_scaling_factor * k * L_H / (2π).

Keyword arguments:
- `min_pts=32`: minimum total panel-node safeguard.
- `eps=1e-15`: numerical tolerance.
- `symmetry=nothing`: optional symmetry image projection.
- `ngl=16`: Gauss-Legendre order per panel.
- `near_panels=0`: number of near panels for special quadrature treatment.
- `logquad_prec=256`: BigFloat precision for log weights.
- `mp_dps=80`: precision used for Legendre-Q/P Taylor tables.
"""
function DLP_hyperbolic_log_product(pts_scaling_factor::Union{T,Vector{T}},billiard::Bi;min_pts=32,eps=T(1e-15),symmetry::Union{Nothing,AbsSymmetry}=nothing,ngl::Int=16,near_panels::Int=0,logquad_prec::Int=256,mp_dps::Int=80) where {T<:Real,Bi<:AbsBilliard}
    bs=pts_scaling_factor isa T ? [pts_scaling_factor] : T.(pts_scaling_factor)
    Sym=typeof(symmetry)
    near_panels<0 && error("near_panels must be nonnegative.")
    return DLP_hyperbolic_log_product{T,Bi,Sym}(bs,bs[1],eps,min_pts,min_pts,billiard,symmetry,ngl,near_panels,logquad_prec,mp_dps)
end

# helpers that help determine adjacent panels and when to apply the logarithmic product correction
@inline function cyclic_panel_distance(p::Int,q::Int,npan::Int)
    d=abs(p-q)
    return min(d,npan-d)
end
@inline function is_near_panel(p::Int,q::Int,npan::Int,near_panels::Int)
    return cyclic_panel_distance(p,q,npan)<=near_panels
end
@inline function cyclic_panel_offset(p_i::Int,p_j::Int,npan::Int)
    r=p_i-p_j
    if r>npan÷2
        r-=npan
    elseif r<-npan÷2
        r+=npan
    end
    return r
end

"""
    DLPHypLogPanel{T}

One hyperbolic log-product quadrature panel.

The panel stores a subinterval `[u0,u1]` of a parent curve parameter. Local
Gauss coordinate `τ ∈ [0,1]` is mapped by

    u(τ) = (1-τ)u0 + τu1.

Fields:
- `curve`: parent curve object.
- `u0`, `u1`: parameter interval endpoints.
- `id`: global panel id.
- `compid`: connected boundary component id.
"""
struct DLPHypLogPanel{T<:Real}
    curve::Any
    u0::T
    u1::T
    id::Int
    compid::Int
end

"""
    DLPHypLogPanelData{T}

Panel metadata attached to a hyperbolic log-product discretization.

Fields:
- `panels`: all panels used in the discretization.
- `panel_id[j]`: panel containing node `j`.
- `local_id[j]`: local Gauss index of node `j` inside its panel.
- `ngl`: number of Gauss-Legendre nodes per panel.

This metadata is required to decide when a source-target pair lies on the same
panel and therefore requires logarithmic product quadrature.
"""
struct DLPHypLogPanelData{T<:Real}
    panels::Vector{DLPHypLogPanel{T}}
    panel_id::Vector{Int}
    local_id::Vector{Int}
    ngl::Int
end

"""
    DLPHypLogDiscretization{T} <: AbsPoints

Complete boundary discretization for `DLP_hyperbolic_log_product`.

Contains:
- `bp`: `BoundaryPointsHyp`, storing geometry, normals, Euclidean weights,
  hyperbolic weights, and hyperbolic arclength coordinates.
- `pdata`: panel metadata needed for same-panel log-product quadrature.
- `Λ`: precomputed logarithmic product weight matrix.
- `nearΛ`: dictionary of near-panel logarithmic product weight matrices.
- `ξ`: local Gauss-Legendre nodes in [-1,1].
- `ω`: local Gauss-Legendre weights.
"""
struct DLPHypLogDiscretization{T<:Real}<:AbsPoints
    bp::BoundaryPointsHyp{T}
    pdata::DLPHypLogPanelData{T}
    Λ::Matrix{T}
    nearΛ::Dict{Int,Matrix{T}}
    ξ::Vector{T}
    ω::Vector{T}
end

"""
    DLPHypLogTaylorWorkspace

Special-function workspace for the hyperbolic Green function.

Stores the Legendre-Q Taylor table used for the hyperbolic Green-function
derivative and the corresponding P/logarithmic-coefficient table used in the
local singular split.

Fields:
- `pre`: distance-domain Taylor precomputation.
- `qws`: temporary Taylor workspace.
- `qtab`: Legendre-Q derivative table.
- `ptab`: logarithmic coefficient / P-function table.
- `k`: complex spectral parameter used to build the tables.
"""
struct DLPHypLogTaylorWorkspace
    pre::QTaylorPrecomp
    qws::QTaylorWorkspace
    qtab::QTaylorTable
    ptab::PTaylorTable
    k::ComplexF64
end

struct DLPHypLogGeomCache{T<:Real}
    d::Matrix{T}
    dn::Matrix{T}
    dnlogλ::Vector{T}
    speed_half::Vector{T}
    dimg::Vector{Matrix{T}}
    dnimg::Vector{Matrix{T}}
    imgscale::Vector{Complex{T}}
end

"""
    DLPHypLogWorkspace{T}

Reusable assembly workspace for `DLP_hyperbolic_log_product`.

Fields:
- `taylor`: k-dependent Legendre-Q/P Taylor tables.
- `G`: geometry cache for the current discretization, storing inter-node distances,
  source-normal derivatives, and image contributions.
- `N`: matrix dimension.
"""
struct DLPHypLogWorkspace{T<:Real}
    taylor::DLPHypLogTaylorWorkspace
    G::DLPHypLogGeomCache{T}
    N::Int
end

# helpers to determine the matrix dimension from the discretization
# imporrant due to symmetries as the matrix sizes get reduced by the symmetry order
boundary_matrix_size(disc::DLPHypLogDiscretization)=length(disc.bp.xy)
@inline _workspace_dim(ws::DLPHypLogWorkspace)=ws.N

"""
    _hyp_log_inv_cdf_values(ts,F,ys)

Invert a monotone tabulated CDF by piecewise-linear interpolation.

Given a grid `ts`, normalized cumulative values `F`, and target values
`ys ∈ [0,1]`, return parameter values `u` such that approximately
`F(u)=ys`.

This is used only to place panel endpoints approximately uniformly in
hyperbolic arclength. The actual quadrature nodes and weights are still the
Gauss-Legendre nodes inside each panel, so this inversion does not need to be
as accurate as a direct node-placement rule.
"""
function _hyp_log_inv_cdf_values(ts,F,ys::AbstractVector{T}) where {T<:Real}
    out=Vector{T}(undef,length(ys))
    @inbounds for n in eachindex(ys)
        y=clamp(ys[n],zero(T),one(T))
        j=searchsortedfirst(F,y)
        if j<=1
            out[n]=T(ts[1])
        elseif j>length(F)
            out[n]=T(ts[end])
        else
            f1=T(F[j-1]);f2=T(F[j])
            t1=T(ts[j-1]);t2=T(ts[j])
            a=(y-f1)/(f2-f1)
            out[n]=muladd(a,t2-t1,t1)
        end
    end
    return out
end

"""
    _hyp_log_panelize_component(comp,compid,k,b,min_panels,ngl)

Split one connected boundary component into hyperbolic-length-adapted panels.

For each curve piece, a hyperbolic arclength CDF is precomputed and the panel
endpoints are placed approximately uniformly in hyperbolic arclength. The number
of panels is chosen as

    np = max(min_panels, ceil(b*k*L_H/(2π*ngl))),

so that the total number of Gauss nodes on the piece is approximately

    ngl*np ≈ b*k*L_H/(2π).

This gives a constant hyperbolic points-per-wavelength rule while retaining
high-order Gauss-Legendre quadrature inside each panel.
"""
function _hyp_log_panelize_component(comp,compid::Int,k::T,b::T,min_panels::Int,ngl::Int) where {T<:Real}
    panels=DLPHypLogPanel{T}[]
    pid=0
    for crv in comp
        pre=precompute_hyper_cdf(crv;M=4000,safety=1e-14)
        Lh=T(pre.Lh)
        np=max(min_panels,ceil(Int,b*k*Lh/(TWO_PI*T(ngl))))
        ys=[T(a)/T(np) for a in 0:np]
        us=_hyp_log_inv_cdf_values(pre.ts_dense,pre.F_dense,ys)
        for a in 1:np
            pid+=1
            push!(panels,DLPHypLogPanel{T}(crv,us[a],us[a+1],pid,compid))
        end
    end
    return panels
end

"""
    _hyp_log_panel_eval(p,τ)

Evaluate one hyperbolic log-product panel at local coordinate `τ ∈ [0,1]`.

Returns

    q, n, γu, γuu, sp, κ, λ

where:
- `q` is the boundary point,
- `n` is the Euclidean outward unit normal,
- `γu` and `γuu` are first and second derivatives with respect to panel
  coordinate `τ`,
- `sp = |γu|` is the Euclidean speed,
- `κ` is the Euclidean signed curvature in the library convention,
- `λ = 2/(1-|q|²)` is the Poincare conformal factor.
"""
@inline function _hyp_log_panel_eval(p::DLPHypLogPanel{T},τ::T) where {T<:Real}
    u=(one(T)-τ)*p.u0+τ*p.u1
    du=p.u1-p.u0
    q=SVector{2,T}(curve(p.curve,u))
    γu=SVector{2,T}(tangent(p.curve,u))*du
    γuu=SVector{2,T}(tangent_2(p.curve,u))*(du^2)
    sp=hypot(γu[1],γu[2])
    n=SVector{2,T}(γu[2]/sp,-γu[1]/sp)
    κ=(γu[1]*γuu[2]-γu[2]*γuu[1])/(sp^3)
    x=q[1];y=q[2]
    λ=λ_poincare(x,y)
    return q,n,γu,γuu,sp,κ,λ
end

function build_dlp_hyp_log_geom_cache(solver::DLP_hyperbolic_log_product,disc::DLPHypLogDiscretization{T}) where {T<:Real}
    pts=disc.bp
    pdata=disc.pdata
    N=length(pts.xy)
    d=Matrix{T}(undef,N,N)
    dn=Matrix{T}(undef,N,N)
    dnlogλ=Vector{T}(undef,N)
    speed_half=Vector{T}(undef,N)
    @inbounds for i in 1:N
        xi,yi=pts.xy[i]
        nxi,nyi=pts.normal[i]
        dnlogλ[i]=hyp_dnlogλ(xi,yi,nxi,nyi)
        speed_half[i]=pts.ds[i]/disc.ω[pdata.local_id[i]]
    end
    @inbounds for j in 1:N
        xj,yj=pts.xy[j]
        nxj,nyj=pts.normal[j]
        for i in 1:N
            if i==j
                d[i,j]=zero(T)
                dn[i,j]=zero(T)
            else
                xi,yi=pts.xy[i]
                d[i,j]=hyperbolic_distance_poincare(xi,yi,xj,yj)
                dn[i,j]=_∂n_d(xi,yi,xj,yj,nxj,nyj)
            end
        end
    end
    ximgs=Vector{Vector{T}}()
    yimgs=Vector{Vector{T}}()
    nximgs=Vector{Vector{T}}()
    nyimgs=Vector{Vector{T}}()
    scales=Complex{T}[]
    sym=solver.symmetry
    shift_x=hasproperty(solver.billiard,:x_axis) ? T(solver.billiard.x_axis) : zero(T)
    shift_y=hasproperty(solver.billiard,:y_axis) ? T(solver.billiard.y_axis) : zero(T)
    function push_image!(kind,scale)
        xi=Vector{T}(undef,N); yi=Vector{T}(undef,N)
        nxi=Vector{T}(undef,N); nyi=Vector{T}(undef,N)
        @inbounds for j in 1:N
            xj,yj=pts.xy[j]
            nxj,nyj=pts.normal[j]
            if kind===:x
                xi[j]=_x_reflect(xj,shift_x); yi[j]=yj
                nxi[j],nyi[j]=_x_reflect_normal(nxj,nyj)
            elseif kind===:y
                xi[j]=xj; yi[j]=_y_reflect(yj,shift_y)
                nxi[j],nyi[j]=_y_reflect_normal(nxj,nyj)
            elseif kind===:xy
                xi[j]=_x_reflect(xj,shift_x); yi[j]=_y_reflect(yj,shift_y)
                nxi[j],nyi[j]=_xy_reflect_normal(nxj,nyj)
            end
        end
        push!(ximgs,xi);push!(yimgs,yi);push!(nximgs,nxi);push!(nyimgs,nyi)
        push!(scales,scale)
    end
    if !isnothing(sym)
        if hasproperty(sym,:axis)
            if sym.axis==:y_axis
                push_image!(:x,Complex{T}(sym.parity==-1 ? -one(T) : one(T),zero(T)))
            elseif sym.axis==:x_axis
                push_image!(:y,Complex{T}(sym.parity==-1 ? -one(T) : one(T),zero(T)))
            elseif sym.axis==:origin
                px=sym.parity[1]==-1 ? -one(T) : one(T)
                py=sym.parity[2]==-1 ? -one(T) : one(T)
                push_image!(:x,Complex{T}(px,zero(T)))
                push_image!(:y,Complex{T}(py,zero(T)))
                push_image!(:xy,Complex{T}(px*py,zero(T)))
            end
        elseif sym isa Rotation
            ctab,stab,χ=_rotation_tables(T,sym.n,mod(sym.m,sym.n))
            cx=T(sym.center[1]); cy=T(sym.center[2])
            for l in 1:sym.n-1
                xi=Vector{T}(undef,N); yi=Vector{T}(undef,N)
                nxi=Vector{T}(undef,N); nyi=Vector{T}(undef,N)
                @inbounds for j in 1:N
                    xj,yj=pts.xy[j]
                    nxj,nyj=pts.normal[j]
                    xi[j],yi[j]=_rot_point(xj,yj,cx,cy,ctab[l+1],stab[l+1])
                    nxi[j],nyi[j]=_rot_vec(nxj,nyj,ctab[l+1],stab[l+1])
                end
                push!(ximgs,xi); push!(yimgs,yi); push!(nximgs,nxi); push!(nyimgs,nyi)
                push!(scales,Complex{T}(χ[l+1]))
            end
        end
    end
    dimg=Matrix{T}[]
    dnimg=Matrix{T}[]
    for r in eachindex(scales)
        Dr=Matrix{T}(undef,N,N)
        Dnr=Matrix{T}(undef,N,N)
        @inbounds for j in 1:N
            xj=ximgs[r][j]; yj=yimgs[r][j]
            nxj=nximgs[r][j]; nyj=nyimgs[r][j]
            for i in 1:N
                xi,yi=pts.xy[i]
                dij=hyperbolic_distance_poincare(xi,yi,xj,yj)
                Dr[i,j]=dij
                Dnr[i,j]=dij<=eps(T) ? zero(T) : _∂n_d(xi,yi,xj,yj,nxj,nyj)
            end
        end
        push!(dimg,Dr);push!(dnimg,Dnr)
    end
    return DLPHypLogGeomCache(d,dn,dnlogλ,speed_half,dimg,dnimg,scales)
end

"""
    evaluate_points(solver::DLP_hyperbolic_log_product,billiard,k)

Build the panelwise hyperbolic boundary discretization at wavenumber `k`.

If `solver.symmetry === nothing`, the full boundary is used. Otherwise the
desymmetrized boundary is used and symmetry images are later added during matrix
assembly.

The returned object contains:
- Euclidean nodes, normals, curvature, and quadrature weights `ds`;
- hyperbolic weights `dsH = λ ds`;
- cumulative hyperbolic arclength coordinates;
- panel ids and local Gauss ids for same-panel singular quadrature.

Panel endpoints are approximately uniform in hyperbolic arclength, while the
nodes inside each panel are standard Gauss-Legendre nodes.
"""
function evaluate_points(solver::DLP_hyperbolic_log_product{T},billiard::Bi,k::Real) where {T<:Real,Bi<:AbsBilliard}
    boundary=isnothing(solver.symmetry) ? billiard.full_boundary : billiard.desymmetrized_full_boundary
    comps=_boundary_components_v2(boundary)
    length(comps)==1 || error("DLP_hyperbolic_log_product currently expects one connected boundary component.")
    panels=_hyp_log_panelize_component(comps[1],1,T(k),solver.pts_scaling_factor[1],max(1,solver.min_pts÷solver.ngl),solver.ngl)
    ξ0,ω0=gausslegendre(solver.ngl)
    ξ=T.(ξ0)
    ω=T.(ω0)
    N=length(panels)*solver.ngl
    xy=Vector{SVector{2,T}}(undef,N)
    normal=Vector{SVector{2,T}}(undef,N)
    kappa=Vector{T}(undef,N)
    ds=Vector{T}(undef,N)
    λs=Vector{T}(undef,N)
    dsH=Vector{T}(undef,N)
    ξH=Vector{T}(undef,N)
    tangent_1=Vector{SVector{2,T}}(undef,N)
    tangent_2=Vector{SVector{2,T}}(undef,N)
    ts=Vector{T}(undef,N)
    original_ts=Vector{T}(undef,N)
    ws=Vector{T}(undef,N)
    ws_der=Vector{T}(undef,N)
    panel_id=Vector{Int}(undef,N)
    local_id=Vector{Int}(undef,N)
    qid=0
    sH=zero(T)
    for p in panels
        for l in 1:solver.ngl
            qid+=1
            τ=(ξ[l]+one(T))/T(2)
            xq,nq,γu,γuu,sp,κ,λ=_hyp_log_panel_eval(p,τ)
            wq=ω[l]/T(2)
            xy[qid]=xq
            normal[qid]=nq
            tangent_1[qid]=γu
            tangent_2[qid]=γuu
            kappa[qid]=κ
            ds[qid]=wq*sp
            λs[qid]=λ
            dsH[qid]=λ*ds[qid]
            ξH[qid]=sH
            sH+=dsH[qid]
            ts[qid]=T(p.id-1)+τ
            original_ts[qid]=ts[qid]
            ws[qid]=ω[l]
            ws_der[qid]=one(T)
            panel_id[qid]=p.id
            local_id[qid]=l
        end
    end
    bp=BoundaryPointsHyp{T}(xy,normal,kappa,ds,λs,dsH,ξH,sH,tangent_1,tangent_2,ts,original_ts,ws,ws_der)
    pdata=DLPHypLogPanelData{T}(panels,panel_id,local_id,solver.ngl)
    Λ=log_weights_matrix(T,ξ;prec=solver.logquad_prec)
    nearΛ=build_near_log_weights(T,ξ,solver.near_panels;prec=solver.logquad_prec)
    return DLPHypLogDiscretization{T}(bp,pdata,Λ,nearΛ,ξ,ω)
end

"""
    build_dlp_hyp_log_taylor_workspace(pts,solver,k)

Build the k-dependent Taylor-table workspace for the hyperbolic DLP kernel.

The distance range is estimated from the discretization and the active symmetry.
The workspace stores:
- a Legendre-Q table for the hyperbolic Green-function derivative;
- a P/logarithmic-coefficient table used in the local singular split.

These tables are reused for all source-target pairs during matrix assembly.
"""
function build_dlp_hyp_log_taylor_workspace(pts::BoundaryPointsHyp{T},solver::DLP_hyperbolic_log_product,k) where {T<:Real}
    dmin,dmax=d_bounds_hyp(pts,solver.symmetry;dmin_floor=T(1e-15),pad_max=T(1.1))
    pre=build_QTaylorPrecomp(;dmin=legendre_d_threshold(),dmax=Float64(dmax)*1.05)
    qws=QTaylorWorkspace(;threaded=false)
    qtab=alloc_QTaylorTable(pre;k=ComplexF64(k))
    ptab=alloc_PTaylorTable(pre;k=ComplexF64(k))
    build_QTaylorTable!(qtab,pre,qws,ComplexF64(k);mp_dps=solver.mp_dps)
    build_PTaylorTable!(ptab,pre,qws,ComplexF64(k);mp_dps=solver.mp_dps)
    return DLPHypLogTaylorWorkspace(pre,qws,qtab,ptab,ComplexF64(k))
end

"""
    build_dlp_hyp_log_workspace(solver,disc,k)

Build the full reusable workspace for one discretization and spectral parameter.

This constructs:
- Gauss-Legendre nodes and weights on the reference panel;
- the logarithmic product-integration matrix `Λ`;
- the hyperbolic Legendre-Q/P Taylor tables.

For repeated matrix assembly at the same `k` and same discretization, reuse this
workspace to avoid rebuilding log weights and special-function tables.
"""
function build_dlp_hyp_log_workspace(solver::DLP_hyperbolic_log_product,disc::DLPHypLogDiscretization{T},k) where {T<:Real}
    taylor=build_dlp_hyp_log_taylor_workspace(disc.bp,solver,k)
    G=build_dlp_hyp_log_geom_cache(solver,disc)
    return DLPHypLogWorkspace{T}(taylor,G,length(disc.bp.xy))
end
function build_dlp_hyp_log_workspace(solver::DLP_hyperbolic_log_product,disc::DLPHypLogDiscretization{T},G::DLPHypLogGeomCache{T},k) where {T<:Real}
    taylor=build_dlp_hyp_log_taylor_workspace(disc.bp,solver,k)
    return DLPHypLogWorkspace{T}(taylor,G,length(disc.bp.xy))
end

"""
    _regular_hyp_log_image_D(qtab,xi,yi,xj,yj,nxj,nyj,wj,scale)

Evaluate one regular hyperbolic DLP image contribution.

This is used for symmetry images. Since image sources are geometrically
separated from the physical source copy, no same-panel logarithmic correction is
needed; the ordinary off-diagonal hyperbolic DLP kernel is sufficient.

Returns
    scale * ∂_{n_y}G_k(d_H(x_i,y_j)) * wj.
"""
@inline function _regular_hyp_log_image_D(qtab::QTaylorTable,xi::T,yi::T,xj::T,yj::T,nxj::T,nyj::T,wj::T,scale::Complex{T}) where {T<:Real}
    d=hyperbolic_distance_poincare(xi,yi,xj,yj)
    d<=eps(T) && return zero(Complex{T})
    dn=_∂n_d(xi,yi,xj,yj,nxj,nyj)
    return scale*hyp_raw_dlp(qtab,Float64(d),dn)*wj
end

# Logarithmic coefficient for product quadrature with log|ξ_i-ξ_j|.
#
# Since
#
#     log(sinh²(d/2)) = 2log|ξ_i-ξ_j| + smooth,
#
# the coefficient multiplying log|ξ_i-ξ_j| is
#
#     L1 = 2 A′(d_H) ∂_{n_y}d_H.
#
# Therefore this function returns the coefficient for a single logarithm
# log|ξ_i-ξ_j|. If one instead writes a Kress split with
# log(4sin²((t-s)/2)), the corresponding coefficient is half of this.
@inline hyp_L1(ptab::PTaylorTable,d::Float64,dn::T) where {T<:Real}=2*hyperbolic_Alog_d(ptab,d)*dn

# Smooth Kress remainder.
# After removing the singular logarithmic part,
#
#     K = L1 logterm + L2,
# so
#     L2 = raw kernel - L1 * logterm.
#
# By construction, L2 remains finite and smooth as the target approaches the
# source along the same smooth boundary.
# This part is integrated with ordinary periodic quadrature weights.
@inline function hyp_L2(qtab::QTaylorTable,ptab::PTaylorTable,d::Float64,dn::T,logterm::T) where {T<:Real}
    l1=hyp_L1(ptab,d,dn)
    raw=hyp_raw_dlp(qtab,d,dn)
    return raw-l1*logterm
end

# the diagonal limit of the L2 coefficient, used for the removable singularity at same-panel nodes
@inline function hyp_L2_diag_GL(kappaE::T,dnlogλ::T) where {T<:Real}
    return Complex{T}((-kappaE-dnlogλ)*INV_TWO_PI,zero(T))
end

"""
    construct_dlp_hyp_log_product_matrix!(D,solver,disc,ws;multithreaded=true)

Assemble the source-normal hyperbolic double-layer matrix `D`.

For source and target nodes on the same panel, the kernel is split as

    K(ξ_i,ξ_j) = L1(ξ_i,ξ_j) log|ξ_i-ξ_j| + L2(ξ_i,ξ_j),

and discretized by

    Λ[i,j] * L1 + ω[j] * L2.

For different panels, the ordinary smooth Nyström rule is used. The diagonal is
the analytic removable DLP limit,

    ds_j * (-κ_E - ∂ₙ log λ)/(2π),

consistent with the source-normal convention.

If a symmetry is active, nonsingular reflected or rotated image contributions
are added as regular off-diagonal hyperbolic DLP terms.
"""
function construct_dlp_hyp_log_product_matrix!(D::AbstractMatrix{Complex{T}},solver::DLP_hyperbolic_log_product,disc::DLPHypLogDiscretization{T},ws::DLPHypLogWorkspace{T};multithreaded::Bool=true) where {T<:Real}
    pts=disc.bp
    pdata=disc.pdata
    npan=length(pdata.panels)
    near_panels=solver.near_panels
    qtab=ws.taylor.qtab
    ptab=ws.taylor.ptab
    Λ=disc.Λ
    nearΛ=disc.nearΛ
    ξ=disc.ξ
    ω=disc.ω
    G=ws.G
    N=length(pts.xy)
    fill!(D,zero(Complex{T}))
    @use_threads multithreading=(multithreaded&&N>=32) for j in 1:N
        p_j=pdata.panel_id[j]
        jl=pdata.local_id[j]
        speed_half_j=G.speed_half[j]
        for i in 1:N
            if i==j
                D[i,j]+=pts.ds[j]*hyp_L2_diag_GL(pts.kappa[i],G.dnlogλ[i])
            else
                d=G.d[i,j]
                dn=G.dn[i,j]
                p_i=pdata.panel_id[i]
                if p_i==p_j
                    il=pdata.local_id[i]
                    l1=2*hyp_L1(ptab,Float64(d),dn)*speed_half_j
                    full=hyp_raw_dlp(qtab,Float64(d),dn)*speed_half_j
                    l2=full-l1*log(abs(ξ[il]-ξ[jl]))
                    D[i,j]+=Λ[il,jl]*l1+ω[jl]*l2
                else
                    r=cyclic_panel_offset(p_i,p_j,npan)
                    if abs(r)<=near_panels && haskey(nearΛ,r)
                        il=pdata.local_id[i]
                        x0=ξ[il]+T(2*r)
                        l1=2*hyp_L1(ptab,Float64(d),dn)*speed_half_j
                        full=hyp_raw_dlp(qtab,Float64(d),dn)*speed_half_j
                        l2=full-l1*log(abs(ξ[jl]-x0))
                        D[i,j]+=nearΛ[r][il,jl]*l1+ω[jl]*l2
                    else
                        D[i,j]+=hyp_raw_dlp(qtab,Float64(d),dn)*pts.ds[j]
                    end
                end
            end
            @inbounds for r in eachindex(G.imgscale)
                dI=G.dimg[r][i,j]
                dI<=eps(T) && continue
                dnI=G.dnimg[r][i,j]
                D[i,j]+=G.imgscale[r]*hyp_raw_dlp(qtab,Float64(dI),dnI)*pts.ds[j]
            end
        end
    end
    return D
end

"""
    construct_fredholm_hyp_log_product_matrix!(A,solver,disc,ws;multithreaded=true)

Construct the Fredholm matrix

    A(k) = I - D(k)

in-place, where `D(k)` is assembled by
`construct_dlp_hyp_log_product_matrix!`.

This is the original source-normal double-layer Fredholm matrix. For boundary
normal-derivative vectors used in Husimi or wavefunction reconstruction, use the
weighted adjoint path instead.
"""
function construct_fredholm_hyp_log_product_matrix!(A::AbstractMatrix{Complex{T}},solver::DLP_hyperbolic_log_product,disc::DLPHypLogDiscretization{T},ws::DLPHypLogWorkspace{T};multithreaded::Bool=true) where {T<:Real}
    construct_dlp_hyp_log_product_matrix!(A,solver,disc,ws;multithreaded=multithreaded)
    @inbounds for j in axes(A,2),i in axes(A,1)
        A[i,j]*=-1
    end
    @inbounds for i in axes(A,1)
        A[i,i]+=one(Complex{T})
    end
    return A
end

"""
    adjoint_fredholm_matrix!(A,D,solver,disc,ws;multithreaded=true)

Construct the weighted Nyström adjoint Fredholm matrix.

The original DLP uses the source-normal kernel. The weighted discrete adjoint is

    D'ᵢⱼ = Dⱼᵢ dsⱼ / dsᵢ,

and the adjoint Fredholm matrix is

    A = I - D'.

This is the appropriate matrix for extracting the boundary function
`u = ∂ₙψ` used in boundary-Husimi and wavefunction reconstruction.
"""
function adjoint_fredholm_matrix!(A::AbstractMatrix{Complex{T}},D::AbstractMatrix{Complex{T}},solver::DLP_hyperbolic_log_product,disc::DLPHypLogDiscretization{T},ws::DLPHypLogWorkspace{T};multithreaded::Bool=true) where {T<:Real}
    construct_dlp_hyp_log_product_matrix!(D,solver,disc,ws;multithreaded=multithreaded)
    ds=disc.bp.ds
    N=length(ds)
    @inbounds for i in 1:N,j in 1:N
        A[i,j]=-D[j,i]*ds[j]/ds[i]
    end
    @inbounds for i in 1:N
        A[i,i]+=one(Complex{T})
    end
    return A
end

"""
    construct_matrices!(solver,A,disc,ws,k;multithreaded=true)

Construct the original hyperbolic DLP Fredholm matrix in-place.

This is the library-facing matrix assembly function for
`DLP_hyperbolic_log_product`. It fills the caller-provided dense matrix `A`
with the Fredholm operator

    A(k) = I - D(k),

where `D(k)` is the source-normal hyperbolic double-layer operator
discretized by panelwise Gauss-Legendre Nyström quadrature with same-panel
logarithmic product integration.

More explicitly:

- off-panel interactions use the regular hyperbolic DLP kernel;
- same-panel interactions use the split

      K = L₁ log|ξ_i-ξ_j| + L₂

  together with the precomputed logarithmic product weights `Λ`;
- diagonal entries use the analytic removable DLP limit;
- if a symmetry sector is active, reflected / rotated image contributions are
  added through the method of images.

The supplied workspace `ws` is assumed to have been built for the same
spectral parameter `k`, since it contains the corresponding Legendre-Q/P
Taylor expansions used for fast hyperbolic kernel evaluation.

This function performs no allocations aside from any hidden BLAS temporaries,
making it the preferred low-level assembly path for repeated solves or contour
methods.

Arguments:
- `solver`: hyperbolic DLP log-product solver configuration.
- `A`: preallocated dense output matrix.
- `disc`: fixed boundary discretization.
- `ws`: k-dependent assembly workspace.
- `k`: spectral parameter (assumed consistent with `ws`).

Keyword arguments:
- `multithreaded=true`: enable threaded dense assembly.

See also:
`construct_fredholm_hyp_log_product_matrix!`,
`build_dlp_hyp_log_workspace`,
`construct_boundary_matrices!`.
"""
function construct_matrices!(solver::DLP_hyperbolic_log_product,A::AbstractMatrix{Complex{T}},disc::DLPHypLogDiscretization{T},ws::DLPHypLogWorkspace{T},k;multithreaded::Bool=true) where {T<:Real}
    construct_fredholm_hyp_log_product_matrix!(A,solver,disc,ws;multithreaded=multithreaded)
end

"""
    construct_boundary_matrices!(Tbufs,solver,disc,zj;multithreaded=true,timeit=false)

Construct boundary-integral matrices for a Beyn contour solve.

For each complex contour node `zj[q]`, this function assembles the nonlinear
Fredholm matrix

    T(zj[q]) = I - D(zj[q]),

where `D(z)` is the hyperbolic source-normal double-layer operator evaluated
at the complex spectral parameter `z`.

This is the standard interface used by contour-based nonlinear eigenvalue
methods such as Beyn's algorithm, where the matrices

    T(z₁), T(z₂), ..., T(z_m)

are required to evaluate contour integrals of the resolvent

    T(z)^(-1).

The boundary discretization `disc` is reused across all contour points, while
the hyperbolic special-function tables are rebuilt because they depend on the
complex spectral parameter.

Arguments:
- `Tbufs`: vector of preallocated dense matrices, one per contour node.
- `solver`: hyperbolic DLP log-product solver.
- `disc`: fixed boundary discretization.
- `zj`: contour nodes in the complex spectral plane.

Keyword arguments:
- `multithreaded=true`: enable threaded matrix assembly.
- `timeit=false`: enable timing diagnostics through benchmarking macros.

Notes:
This is intentionally allocation-aware and optimized for repeated contour
assembly in nonlinear eigenvalue searches.
"""
function construct_boundary_matrices!(Tbufs::Vector{Matrix{ComplexF64}},solver::DLP_hyperbolic_log_product,disc::DLPHypLogDiscretization{T},zj::AbstractVector{ComplexF64};multithreaded::Bool=true,timeit::Bool=false) where {T<:Real}
    @blas_1 begin
        @inbounds for q in eachindex(zj)
            @benchit timeit=timeit "DLP_hyperbolic_log_product Workspace" ws=build_dlp_hyp_log_workspace(solver,disc,zj[q])
            n=_workspace_dim(ws)
            @assert size(Tbufs[q])==(n,n) "Tbufs[$q] has size $(size(Tbufs[q])), but DLP-hyperbolic-log-product requires ($n,$n)."
            fill!(Tbufs[q],0.0+0.0im)
            @benchit timeit=timeit "DLP_hyperbolic_log_product Assembly" construct_matrices!(solver,Tbufs[q],disc,ws,zj[q];multithreaded=multithreaded)
        end
    end
    return nothing
end

function _hyp_beyn_dim(solver::DLP_hyperbolic_log_product,disc::DLPHypLogDiscretization,k)
    return boundary_matrix_size(disc)
end

"""
    solve(solver,basis,A,disc,ws,k;multithreaded=true,use_krylov=true,which=:svd)

Solve the original hyperbolic DLP spectral problem using a preallocated matrix.

This assembles the Fredholm operator

    A(k) = I - D(k),

for the hyperbolic source-normal double-layer formulation and then applies the
standard solver backend (`SVD`, determinant minimization, or Krylov-based null
search depending on `which`).

This is the lowest-allocation solve path and is intended for repeated spectral
queries at fixed discretization.

Arguments:
- `solver`: hyperbolic DLP log-product solver.
- `basis`: compatibility argument for the common solver interface.
- `A`: preallocated dense matrix workspace.
- `disc`: boundary discretization.
- `ws`: prebuilt k-dependent workspace.
- `k`: spectral parameter.

Keyword arguments:
- `multithreaded=true`: threaded assembly.
- `use_krylov=true`: use KrylovKit for smallest singular vector.
"""
function solve(solver::DLP_hyperbolic_log_product,basis::Ba,A::AbstractMatrix{Complex{T}},disc::DLPHypLogDiscretization{T},ws::DLPHypLogWorkspace{T},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:svd) where {T<:Real,Ba<:AbsBasis}
    @blas_1 construct_matrices!(solver,A,disc,ws,k;multithreaded=multithreaded)
    @svd_or_det_solve A use_krylov which MAX_BLAS_THREADS
end

"""
    solve(solver,basis,disc,ws,k; kwargs...)

Solve the original hyperbolic DLP spectral problem using a prebuilt workspace.

Equivalent to

    A = Matrix(...)
    solve(solver,basis,A,disc,ws,k)

but allocates the dense matrix internally.

This is convenient when repeated solves reuse the same hyperbolic Taylor
workspace but explicit matrix management is unnecessary.
"""
function solve(solver::DLP_hyperbolic_log_product,basis::Ba,disc::DLPHypLogDiscretization{T},ws::DLPHypLogWorkspace{T},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:svd) where {T<:Real,Ba<:AbsBasis}
    A=Matrix{Complex{T}}(undef,ws.N,ws.N)
    return solve(solver,basis,A,disc,ws,k;multithreaded=multithreaded,use_krylov=use_krylov,which=which)
end

function solve(solver::DLP_hyperbolic_log_product,basis::Ba,disc::DLPHypLogDiscretization{T},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:svd) where {T<:Real,Ba<:AbsBasis}
    ws=build_dlp_hyp_log_workspace(solver,disc,k)
    return solve(solver,basis,disc,ws,k;multithreaded=multithreaded,use_krylov=use_krylov,which=which)
end

"""
    solve_vect(solver,basis,A,disc,ws,k;
               multithreaded=true,
               tol=1e-12,
               maxiter=2000,
               krylovdim=40)

Compute the adjoint boundary vector for the hyperbolic DLP formulation.

This function assembles the weighted discrete adjoint Fredholm operator

    A† = I - D†,

where

    D†[i,j] = D[j,i] * ds[j] / ds[i],

corresponding to the adjoint with respect to the quadrature-weighted boundary
inner product.

It then computes the smallest right null vector using the iterative Krylov
solver

    smallest_nullvec_krylov!.

The returned vector is the natural adjoint boundary function for the
Dirichlet eigenproblem and is proportional to the physical normal derivative

    u ≈ ∂ₙψ.

This is the correct vector object for:
- Poincare-Husimi constructions,
- IPR statistics,
- wavefunction reconstruction.

Arguments:
- `solver`: hyperbolic DLP log-product solver.
- `basis`: interface compatibility argument.
- `A`: preallocated adjoint Fredholm matrix.
- `disc`: boundary discretization.
- `ws`: k-dependent workspace.
- `k`: spectral parameter.

Keyword arguments:
- `multithreaded=true`: threaded assembly.
- `tol=1e-12`: Krylov convergence tolerance.
- `maxiter=2000`: maximum Krylov iterations.
- `krylovdim=40`: Krylov subspace dimension.

Returns:
- `σ`: smallest singular / null residual diagnostic.
- `u`: adjoint boundary vector.
"""
function solve_vect(solver::DLP_hyperbolic_log_product,basis::Ba,A::AbstractMatrix{Complex{T}},disc::DLPHypLogDiscretization{T},ws::DLPHypLogWorkspace{T},k;multithreaded::Bool=true,tol=1e-12,maxiter::Int=2000,krylovdim::Int=40) where {T<:Real,Ba<:AbsBasis}
    D=similar(A)
    @blas_1 adjoint_fredholm_matrix!(A,D,solver,disc,ws;multithreaded=multithreaded)
    σ,u,_=smallest_nullvec_krylov!(A;nev=1,tol=tol,maxiter=maxiter,krylovdim=krylovdim)
    return σ,u
end

"""
    solve_vect(solver,basis,disc,ws,k; kwargs...)

Adjoint vector solve with internal matrix allocation.

Equivalent to the fully explicit `solve_vect` method, but allocates the dense
adjoint matrix internally.

Useful when repeated solves reuse the same workspace but explicit matrix
management is unnecessary.
"""
function solve_vect(solver::DLP_hyperbolic_log_product,basis::Ba,disc::DLPHypLogDiscretization{T},ws::DLPHypLogWorkspace{T},k;multithreaded::Bool=true,tol=1e-12,maxiter::Int=2000,krylovdim::Int=40) where {T<:Real,Ba<:AbsBasis}
    A=Matrix{Complex{T}}(undef,ws.N,ws.N)
    return solve_vect(solver,basis,A,disc,ws,k;multithreaded=multithreaded,tol=tol,maxiter=maxiter,krylovdim=krylovdim)
end

"""
    solve_vect(solver,basis,disc,k; kwargs...)

High-level adjoint vector solve.

This constructs the k-dependent hyperbolic assembly workspace internally,
assembles the weighted adjoint Fredholm matrix, and computes the smallest
null vector.

This is the simplest interface for obtaining boundary normal-derivative
vectors, but repeated calls should prefer the workspace-aware overloads to
avoid rebuilding the special-function Taylor tables.
"""
function solve_vect(solver::DLP_hyperbolic_log_product,basis::Ba,disc::DLPHypLogDiscretization{T},k;multithreaded::Bool=true,tol=1e-12,maxiter::Int=2000,krylovdim::Int=40) where {T<:Real,Ba<:AbsBasis}
    ws=build_dlp_hyp_log_workspace(solver,disc,k)
    return solve_vect(solver,basis,disc,ws,k;multithreaded=multithreaded,tol=tol,maxiter=maxiter,krylovdim=krylovdim)
end