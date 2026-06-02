# ==============================================================================
# Magnetic CFIE-Kress solver
# ==============================================================================
#
# This file implements Kress-corrected boundary-integral discretizations for
# two-dimensional magnetic billiards in a constant perpendicular magnetic field.
#
# The spectral parameter is ν and the magnetic scale used for resolution and
# CFIE coupling is
#
#     k(ν,B) = 2√ν/B.
#
# The magnetic Green function is written in the gauge-covariant form
#     G_B(x,y;ν) = exp[-i(x₁y₂-x₂y₁)/B²] Fν(|x-y|²/B²),
# where Fν has the logarithmic decomposition
#     Fν(z) = Aν(z) log(z) + Bν(z).
# and Fν(z) is the confluent hypergeometric U function with parameters depending on ν.
#
# Hence the boundary kernel has the same universal logarithmic singularity as
# the ordinary Helmholtz Green function. For a periodic boundary parameter σ,
# the singularity is split as
#     log(|x(σ)-x(σ')|²/B²)
#       = log(4sin²((σ-σ')/2)) + logratio(σ,σ'),
# where the first term is integrated by the Kress matrix and the second is a
# smooth remainder.
#
# The implemented operators are:
#   S_B(ν)      magnetic single-layer operator,
#   D_B(ν)      covariant source-normal magnetic double-layer operator,
#   C_B(ν)      combined-field operator D_B(ν)+i k(ν,B)S_B(ν).
#
# The Fredholm matrix used by default is the regularized magnetic CFIE
#     C_B(ν) + 0.5 cos(πν) I.
# In the optional unregularized convention the matrix is divided by cos(πν) and
# the jump term becomes 0.5 I. This is useful away from Landau levels only.
#
# Source-normal convention
# ------------------------
# The double-layer kernel differentiates with respect to the source normal.
# Therefore `solve` uses the assembled source-normal matrix for scalar spectral
# diagnostics, while `solve_vect` forms the weighted adjoint
#     A_adj[i,j] = A[j,i] w[j]/w[i],
# where w[j]=|γσ(σ_j)|dσ. This is the correct boundary-vector path for Husimi,
# boundary-current, and wavefunction reconstruction workflows.
#
# Corner handling
# ---------------
# `MagneticCFIE_kress` assumes one smooth periodic boundary. 
# `MagneticCFIE_kress_global_corners` supports one composite outer boundary. For
# corners, a global grading map t=t(σ) is used, with
#     γσ = γt tσ,     γσσ = γtt tσ² + γt tσσ.
# The Kress split remains in the computational σ-variable, so the singular
# quadrature uses `kress_R_corner!` whenever `dt/dσ` is nontrivial.
#
# Symmetry handling
# -----------------
# If no symmetry is supplied, the full periodic matrix is assembled. If a
# symmetry is supplied, the full Kress geometry is still built, but only the
# fundamental representative block is assembled. The physical singular copy is
# treated by Kress quadrature, while all other symmetry images are geometrically
# separated and are added as regular nonsingular image terms with their symmetry
# phases.
#
# Workspace structure
# -------------------
# Geometry data are ν-independent and cached once: pairwise r², magnetic phases,
# logarithmic split terms, source-normal derivatives, Kress weights, source
# weights, normals, curvature numerators, and zmax.
#
# Taylor data are ν-dependent: Fν, Fν′, Aν, Aν′ and Bν are rebuilt when ν changes.
# This separation makes sweeps and Beyn contour assembly much cheaper than
# rebuilding geometry for every spectral point.
#
# Main API
# --------
#   MagneticCFIE_kress(...)
#   MagneticCFIE_kress_global_corners(...)
#   evaluate_points(...)
#   build_magnetic_kress_workspace(...)
#   construct_matrices!(...)
#   construct_boundary_matrices!(...)
#   solve(...)
#   solve_vect(...)
# MO 1/6/2026
# ==============================================================================

const TWO_PI=2*π

"""
    k_from_ν_magnetic(ν,b)
Convert the magnetic spectral parameter `ν` to the associated scale `k=2√ν/b`.
"""
@inline k_from_ν_magnetic(ν,b)=2*sqrt(ν)/b

"""
    cisT(x)
Return `cos(x)+i sin(x)` with complex element type compatible with real type `T`.
"""
@inline cisT(x::T) where {T<:Real}=Complex{T}(cos(x),sin(x))

"""
    MagneticCFIE_kress(ppw,billiard; bmag, min_pts=80, eps=1e-15, symmetry=nothing)

Construct the smooth-boundary magnetic Kress solver.

The spectral parameter is ν and the magnetic scale is k(ν,B)=2√ν/B. The default
combined-field operator is C(ν)=D_B(ν)+i k(ν,B)S_B(ν)+0.5cos(πν)I, where S_B is
the magnetic single-layer operator and D_B is the covariant source-normal
double-layer operator. The boundary must be one smooth periodic curve.
"""
struct MagneticCFIE_kress{T<:Real,Bi<:AbsBilliard,Sym}<:SweepSolver
    sampler::Vector{LinearNodes}
    pts_scaling_factor::Vector{T}
    dim_scaling_factor::T
    eps::T
    min_dim::Int
    min_pts::Int
    billiard::Bi
    symmetry::Sym
    bmag::T
end

"""
    MagneticCFIE_kress_global_corners(ppw,billiard; bmag, min_pts=80, eps=1e-15, symmetry=nothing, kressq=4, min_t_spacing=1e-12)

Construct the corner-graded magnetic Kress solver.

For a composite boundary the computational parameter σ is mapped to the physical
global parameter t=t(σ). The stored data use pts.ts=σ, pts.original_ts=t(σ),
pts.ws=2π/N and pts.ws_der=dt/dσ. The singular split uses log(4sin²((σᵢ-σⱼ)/2))
and `kress_R_corner!` when grading is nontrivial.
"""
struct MagneticCFIE_kress_global_corners{T<:Real,Bi<:AbsBilliard,Sym}<:SweepSolver
    sampler::Vector{LinearNodes}
    pts_scaling_factor::Vector{T}
    dim_scaling_factor::T
    eps::T
    min_dim::Int
    min_pts::Int
    billiard::Bi
    symmetry::Sym
    bmag::T
    kressq::Int
    min_t_spacing::T
end

"""
    MagneticKressSolver
Union alias for the smooth and global-corner magnetic Kress solvers.
"""
const MagneticKressSolver=Union{MagneticCFIE_kress,MagneticCFIE_kress_global_corners}

function MagneticCFIE_kress(ppw::Union{T,Vector{T}},billiard::Bi;bmag::T,min_pts=80,eps=T(1e-15),symmetry::Union{Nothing,AbsSymmetry}=nothing) where {T<:Real,Bi<:AbsBilliard}
    bs=ppw isa T ? [ppw] : ppw
    return MagneticCFIE_kress{T,Bi,typeof(symmetry)}([LinearNodes()],bs,bs[1],eps,min_pts,min_pts,billiard,symmetry,bmag)
end

function MagneticCFIE_kress_global_corners(ppw::Union{T,Vector{T}},billiard::Bi;bmag::T,min_pts=80,eps=T(1e-15),symmetry::Union{Nothing,AbsSymmetry}=nothing,kressq::Int=4,min_t_spacing::T=T(1e-12)) where {T<:Real,Bi<:AbsBilliard}
    bs=ppw isa T ? [ppw] : ppw
    return MagneticCFIE_kress_global_corners{T,Bi,typeof(symmetry)}([LinearNodes()],bs,bs[1],eps,min_pts,min_pts,billiard,symmetry,bmag,kressq,min_t_spacing)
end

"""
    _magnetic_kress_use_reduced(solver)
Return `true` when a symmetry sector is requested and reduced matrix assembly
should be used.
"""
@inline _magnetic_kress_use_reduced(solver::MagneticKressSolver)=!isnothing(solver.symmetry)

"""
    _is_nontrivial_magnetic_grading(pts)
Detect whether the Kress parameterization has a nontrivial grading map by
checking whether `dt/dσ` differs from one.
"""
@inline function _is_nontrivial_magnetic_grading(pts::BoundaryPointsCFIE{T}) where {T<:Real}
    tol=sqrt(eps(T))
    @inbounds for x in pts.ws_der
        abs(x-one(T))>tol && return true
    end
    return false
end

"""
    _magnetic_needed_N(solver,N)

Adjust `N` so that the boundary discretization is compatible with the requested
symmetry. Reflections require multiples of 4; rotations require multiples of
the rotation order.
"""
function _magnetic_needed_N(solver,N)
    needed=2
    if !isnothing(solver.symmetry)
        sym=solver.symmetry
        sym isa Rotation && (needed=lcm(needed,sym.n))
        sym isa Reflection && (needed=lcm(needed,4))
    end
    r=mod(N,needed)
    return r==0 ? N : N+needed-r
end

"""
    MagneticKressGeomCache

Geometry-only cache for magnetic Kress assembly.
Stores pairwise distances, magnetic phases, logarithmic split data, source-normal
derivatives, Kress weights, Euclidean source weights, normals, curvature data,
boundary coordinates, and the maximum table argument `zmax`.
"""
struct MagneticKressGeomCache{T<:Real}
    r2::Matrix{T}
    phase::Matrix{Complex{T}}
    logratio::Matrix{T}
    Lper::Matrix{T}
    dzdn::Matrix{T}
    dchi::Matrix{T}
    Rsrc::Matrix{T}
    wsrc::Vector{T}
    speed::Vector{T}
    nx::Vector{T}
    ny::Vector{T}
    curvnum::Vector{T}
    X::Vector{T}
    Y::Vector{T}
    N::Int
    zmax::T
end

"""
    MagneticKressGeomWorkspace

Full-boundary magnetic geometry workspace.
Contains the geometry cache, boundary dimension, and magnetic length/field
parameter `bmag`.
"""
struct MagneticKressGeomWorkspace{T<:Real}
    G::MagneticKressGeomCache{T}
    N::Int
    bmag::T
end

"""
    MagneticKressReducedGeomWorkspace

Symmetry-reduced magnetic geometry workspace.

Keeps the full Kress geometry but stores fundamental representatives and
symmetry orbit data so that the singular physical copy is treated with Kress
quadrature and the remaining images are added as regular terms.
"""
struct MagneticKressReducedGeomWorkspace{T<:Real}
    full::MagneticKressGeomWorkspace{T}
    Ifund::Vector{Int}
    full_to_fund::Vector{Int}
    full_to_scale::Vector{Complex{T}}
    fund_to_full::Vector{Vector{Int}}
    fund_to_scale::Vector{Vector{Complex{T}}}
    m::Int
end

"""
    MagneticKressTaylorWorkspace

ν-dependent Taylor workspace for the magnetic Green function.
Stores the precomputation object, temporary workspace, Taylor table, and current
spectral parameter `ν`.
"""
mutable struct MagneticKressTaylorWorkspace
    pre
    tws
    tab
    ν::ComplexF64
end

"""
    _workspace_dim(ws)

Return the matrix dimension associated with a magnetic Kress workspace: the full
boundary size for `MagneticKressGeomWorkspace`, or the number of fundamental
representatives for `MagneticKressReducedGeomWorkspace`.
"""
@inline _workspace_dim(ws::MagneticKressGeomWorkspace)=ws.N
@inline _workspace_dim(ws::MagneticKressReducedGeomWorkspace)=ws.m

"""
    evaluate_points(solver::MagneticCFIE_kress,billiard,k)

Construct smooth periodic Kress nodes.

The node count is N≈kL·ppw/(2π), adjusted for symmetry. Derivatives are stored
with respect to σ∈[0,2π): γσ=γu/(2π), γσσ=γuu/(2π)², and dsⱼ=|γσ(σⱼ)|·2π/N.
"""
function evaluate_points(solver::MagneticCFIE_kress{T},billiard::Bi,k::T) where {T<:Real,Bi<:AbsBilliard}
    crv=billiard.full_boundary[1]
    L=crv.length
    N=max(solver.min_pts,round(Int,k*L*solver.pts_scaling_factor[1]/TWO_PI))
    N=_magnetic_needed_N(solver,N)
    ts=[s_mid(j,N) for j in 1:N]
    tsr=ts./TWO_PI
    xy=curve(crv,tsr)
    t1=tangent(crv,tsr)./TWO_PI
    t2=tangent_2(crv,tsr)./(TWO_PI^2)
    ss=arc_length(crv,tsr)
    ds=diff(ss)
    append!(ds,L+ss[1]-ss[end])
    ws=fill(T(TWO_PI/N),N)
    ws_der=ones(T,N)
    z=SVector(zero(T),zero(T))
    return BoundaryPointsCFIE(xy,t1,t2,ts,copy(ts),ws,ws_der,ds,1,true,z,z,z,z)
end

"""
    evaluate_points(solver::MagneticCFIE_kress_global_corners,billiard,k)

Construct globally graded Kress nodes for a composite cornered boundary.

If corners are present, t=t(σ) and the chain rule gives γσ=γt tσ and
γσσ=γtt tσ²+γt tσσ. The weight is dsⱼ=|γσ(σⱼ)|·2π/N. If no true corners are
detected, the method uses the ungraded parameterization t=σ.
"""
function evaluate_points(solver::MagneticCFIE_kress_global_corners{T},billiard::Bi,k::T) where {T<:Real,Bi<:AbsBilliard}
    boundary=billiard.full_boundary
    if length(boundary)==1 && !(boundary[1] isa AbstractVector)
        base=MagneticCFIE_kress(solver.pts_scaling_factor,solver.billiard;bmag=solver.bmag,min_pts=solver.min_pts,eps=solver.eps,symmetry=solver.symmetry)
        return evaluate_points(base,billiard,k)
    end
    _is_single_composite_boundary(boundary) || error("MagneticCFIE_kress_global_corners supports exactly one composite outer boundary.")
    comp=boundary
    corners=_component_corner_locations(T,comp)
    L=sum(crv.length for crv in comp)
    N=max(solver.min_pts,round(Int,k*L*solver.pts_scaling_factor[1]/TWO_PI))
    N=_magnetic_needed_N(solver,N)
    h=T(TWO_PI)/T(N)
    if isempty(corners)
        σ=[TWO_PI*(T(j)-T(0.5))/T(N) for j in 1:N]
        τ=copy(σ)
        jac=ones(T,N)
        jac2=zeros(T,N)
    else
        σ,τ,jac,jac2,_=multi_kress_graded_nodes_data(T,N,corners;q=solver.kressq,minsep_tol=solver.min_t_spacing)
    end
    xy=Vector{SVector{2,T}}(undef,N)
    t1=Vector{SVector{2,T}}(undef,N)
    t2=Vector{SVector{2,T}}(undef,N)
    ds=Vector{T}(undef,N)
    @inbounds for i in 1:N
        q,γt,γtt=_eval_composite_geom_global_t(T,comp,τ[i])
        γσ=γt*jac[i]
        γσσ=γtt*(jac[i]^2)+γt*jac2[i]
        xy[i]=q
        t1[i]=γσ
        t2[i]=γσσ
        ds[i]=hypot(γσ[1],γσ[2])*h
    end
    ws=fill(h,N)
    ws_der=jac
    z=SVector(zero(T),zero(T))
    return BoundaryPointsCFIE(xy,t1,t2,σ,τ,ws,ws_der,ds,1,true,z,z,z,z)
end

"""
    build_Rsrc_kress(solver,pts,speed)

Build the source-scaled Kress logarithmic matrix.

Returns Rsrc[i,j]=R[i,j]·|γσ(σⱼ)|. For corner-graded solvers, `kress_R_corner!`
is used when pts.ws_der is nontrivial; otherwise the standard periodic
`kress_R!` matrix is used.
"""
function build_Rsrc_kress(solver::MagneticCFIE_kress,pts::BoundaryPointsCFIE{T},speed::Vector{T}) where {T<:Real}
    N=length(pts.xy)
    R=zeros(T,N,N)
    kress_R!(R)
    Rsrc=Matrix{T}(undef,N,N)
    @inbounds for j in 1:N, i in 1:N
        Rsrc[i,j]=R[i,j]*speed[j]
    end
    return Rsrc
end
function build_Rsrc_kress(solver::MagneticCFIE_kress_global_corners,pts::BoundaryPointsCFIE{T},speed::Vector{T}) where {T<:Real}
    N=length(pts.xy)
    R=zeros(T,N,N)
    _is_nontrivial_magnetic_grading(pts) ? kress_R_corner!(R) : kress_R!(R)
    Rsrc=Matrix{T}(undef,N,N)
    @inbounds for j in 1:N, i in 1:N
        Rsrc[i,j]=R[i,j]*speed[j]
    end
    return Rsrc
end

"""
    build_magnetic_kress_geom_cache(solver,pts,bmag; safety=1.2)

Precompute geometry-only data for magnetic Kress assembly.

Stores r²=|xᵢ-xⱼ|², phase=exp[-i(xᵢyⱼ-yᵢxⱼ)/B²], Lper=log(4sin²((σᵢ-σⱼ)/2)),
logratio=log(r²/B²)-Lper, dzdn=∂ₙy(r²/B²), and dchi=∂ₙyχ. Also stores source
weights wsrc=|γσ|·dσ, source-scaled Kress weights Rsrc, normals, curvature data,
and zmax≈max(r²/B²).
"""
function build_magnetic_kress_geom_cache(solver::MagneticKressSolver,pts::BoundaryPointsCFIE{T},bmag::T;safety::T=T(1.2)) where {T<:Real}
    N=length(pts.xy)
    b2=bmag*bmag
    r2=Matrix{T}(undef,N,N)
    phase=Matrix{Complex{T}}(undef,N,N)
    logratio=Matrix{T}(undef,N,N)
    Lper=Matrix{T}(undef,N,N)
    dzdn=Matrix{T}(undef,N,N)
    dchi=Matrix{T}(undef,N,N)
    speed=Vector{T}(undef,N)
    nx=Vector{T}(undef,N)
    ny=Vector{T}(undef,N)
    curvnum=Vector{T}(undef,N)
    X=getindex.(pts.xy,1)
    Y=getindex.(pts.xy,2)
    @inbounds for j in 1:N
        tx=pts.tangent[j][1];ty=pts.tangent[j][2]
        txx=pts.tangent_2[j][1];tyy=pts.tangent_2[j][2]
        sp=hypot(tx,ty)
        speed[j]=sp
        nx[j]=ty/sp
        ny[j]=-tx/sp
        curvnum[j]=nx[j]*txx+ny[j]*tyy
    end
    @use_threads multithreading=true for j in 1:N
        xj=X[j];yj=Y[j];nxj=nx[j];nyj=ny[j]
        @inbounds for i in 1:N
            dx=X[i]-xj
            dy=Y[i]-yj
            rr=muladd(dx,dx,dy*dy)
            r2[i,j]=rr
            phase[i,j]=cisT(-(X[i]*yj-Y[i]*xj)/b2)
            if i==j
                Lper[i,j]=zero(T)
                logratio[i,j]=log(speed[j]^2/b2)
                dzdn[i,j]=zero(T)
                dchi[i,j]=zero(T)
            else
                dt=pts.ts[i]-pts.ts[j]
                lp=log(4*sin(0.5*dt)^2)
                drx=xj-X[i]
                dry=yj-Y[i]
                Lper[i,j]=lp
                logratio[i,j]=log(rr/b2)-lp
                dzdn[i,j]=2*(drx*nxj+dry*nyj)/b2
                dchi[i,j]=(drx*nyj-dry*nxj)/b2
            end
        end
    end
    Rsrc=build_Rsrc_kress(solver,pts,speed)
    wsrc=pts.ws.*speed
    zmax=safety*maximum(r2)/b2
    return MagneticKressGeomCache(r2,phase,logratio,Lper,dzdn,dchi,Rsrc,wsrc,speed,nx,ny,curvnum,X,Y,N,zmax)
end

function build_magnetic_kress_geom_workspace_full(solver::MagneticKressSolver,pts::BoundaryPointsCFIE{T}) where {T<:Real}
    G=build_magnetic_kress_geom_cache(solver,pts,solver.bmag)
    return MagneticKressGeomWorkspace(G,length(pts.xy),solver.bmag)
end

function build_magnetic_kress_geom_workspace_reduced(solver::MagneticKressSolver,pts::BoundaryPointsCFIE{T}) where {T<:Real}
    full=build_magnetic_kress_geom_workspace_full(solver,pts)
    Ifund,full_to_fund,full_to_scale,fund_to_full,fund_to_scale=symmetry_index_orbits(T,pts,solver.symmetry,solver.billiard)
    return MagneticKressReducedGeomWorkspace(full,Ifund,full_to_fund,full_to_scale,fund_to_full,fund_to_scale,length(Ifund))
end

"""
    build_magnetic_kress_geom_workspace(solver,pts)

Build the ν-independent geometry workspace.

If no symmetry is supplied, returns the full workspace. If symmetry is supplied,
returns a reduced workspace that keeps the full Kress geometry but assembles only
the fundamental-domain block plus regular symmetry-image contributions.
"""
function build_magnetic_kress_geom_workspace(solver::MagneticKressSolver,pts::BoundaryPointsCFIE{T}) where {T<:Real}
    _magnetic_kress_use_reduced(solver) ? build_magnetic_kress_geom_workspace_reduced(solver,pts) : build_magnetic_kress_geom_workspace_full(solver,pts)
end

"""
    build_magnetic_kress_taylor_workspace(ν,zmax; zmin=1e-3, zsmall=1e-3, h=1e-5, P=6, Msmall=30, mp_dps=30)

Build the ν-dependent Taylor workspace for the magnetic Green function.

The radial kernel is represented as F(z)=A(z)log(z)+B(z), z=r²/B². The table
stores F, Fz, A, Az and B on the required z-range and is reused for all
source-target pairs at fixed ν.
"""
function build_magnetic_kress_taylor_workspace(ν::ComplexF64,zmax;zmin=1e-3,zsmall=1e-3,h=1e-5,P=6,Msmall=30,mp_dps::Int=30)
    confluent_U_set_taylor_params!(h_patch=h,P_patch=P)
    pre=build_MagneticGreenSPrecomp(;zmin=zmin,zmax=zmax,zsmall=zsmall,Msmall=Msmall)
    tws=MagneticGreenSWorkspace(;threaded=false)
    tab=alloc_MagneticGreenSTaylorTable(pre;ν=ν)
    build_MagneticGreenSTaylorTable!(tab,pre,tws,ν;mp_dps=mp_dps)
    return MagneticKressTaylorWorkspace(pre,tws,tab,ν)
end

"""
    update_magnetic_kress_taylor_workspace!(kws,ν; mp_dps=30)

Update the magnetic Taylor tables in-place to a new spectral parameter ν.
Only the ν-dependent table is rebuilt; all geometry and Kress data remain
unchanged.
"""
function update_magnetic_kress_taylor_workspace!(kws::MagneticKressTaylorWorkspace,ν::ComplexF64;mp_dps::Int=30)
    build_MagneticGreenSTaylorTable!(kws.tab,kws.pre,kws.tws,ν;mp_dps=mp_dps)
    kws.ν=ν
    hasproperty(kws.tab,:ν) && setproperty!(kws.tab,:ν,ν)
    return kws
end

"""
    build_magnetic_kress_workspace(solver,pts,ν; h=1e-5, P=6, Msmall=30, mp_dps=30)

Build the complete magnetic Kress workspace `(gws,kws)`.
Here `gws` contains ν-independent geometry and symmetry data, while `kws`
contains ν-dependent Taylor tables for the magnetic Green function.
"""
function build_magnetic_kress_workspace(solver::MagneticKressSolver,pts::BoundaryPointsCFIE{T},ν;h=1e-5,P=6,Msmall=30,mp_dps::Int=30) where {T<:Real}
    gws=build_magnetic_kress_geom_workspace(solver,pts)
    zmax=gws isa MagneticKressGeomWorkspace ? gws.G.zmax : gws.full.G.zmax
    kws=build_magnetic_kress_taylor_workspace(ComplexF64(ν),zmax;h=h,P=P,Msmall=Msmall,mp_dps=mp_dps)
    return gws,kws
end

struct MagneticPoleSubtractionWorkspace
    ns::Vector{Int}
end

@inline _pole_sum(::Nothing,ν,z::Float64)=0.0+0.0im
@inline _pole_z_sum(::Nothing,ν,z::Float64)=0.0+0.0im

@inline function _pole_sum(pole_ws::MagneticPoleSubtractionWorkspace,ν,z::Float64)
    s=0.0+0.0im
    @inbounds for n in pole_ws.ns
        s+=landau_residue(n,z)/(ν-landau_pole(n))
    end
    return s
end

@inline function _pole_z_sum(pole_ws::MagneticPoleSubtractionWorkspace,ν,z::Float64)
    s=0.0+0.0im
    @inbounds for n in pole_ws.ns
        s+=landau_residue_z(n,z)/(ν-landau_pole(n))
    end
    return s
end

"""
    _mag_raw_regular_slp(tab,G,bmag,i,j,pole_ws)

Evaluate the regular off-diagonal magnetic single-layer entry.
This is the nonsingular image/source contribution Sᵢⱼ=phaseᵢⱼ F(zᵢⱼ) wⱼ with
zᵢⱼ=r²ᵢⱼ/B².
"""
@inline function _mag_raw_regular_slp(tab,ν,G::MagneticKressGeomCache{T},bmag::T,i::Int,j::Int,pole_ws::MagneticPoleSubtractionWorkspace) where {T<:Real}
    z=Float64(G.r2[i,j]/(bmag*bmag))
    F=_eval_F(tab,z)-_pole_sum(pole_ws,ν,z)
    return G.phase[i,j]*F*G.wsrc[j]
end

"""
    _mag_raw_regular_dlp_src(tab,G,bmag,i,j,pole_ws)

Evaluate the regular covariant source-normal magnetic double-layer entry.
Uses Dᵢⱼ=phaseᵢⱼ[Fz(zᵢⱼ)∂ₙz+i(∂ₙχ)F(zᵢⱼ)]wⱼ, with derivatives taken in the
source normal.
"""
@inline function _mag_raw_regular_dlp_src(tab,ν,G::MagneticKressGeomCache{T},bmag::T,i::Int,j::Int,pole_ws::MagneticPoleSubtractionWorkspace) where {T<:Real}
    z=Float64(G.r2[i,j]/(bmag*bmag))
    F=_eval_F(tab,z)-_pole_sum(pole_ws,ν,z)
    Fz=_eval_Fz(tab,z)-_pole_z_sum(pole_ws,ν,z)
    return G.phase[i,j]*(Fz*G.dzdn[i,j]+im*G.dchi[i,j]*F)*G.wsrc[j]
end

"""
    _mag_kress_slp(tab,G,bmag,i,j,pole_ws)

Evaluate the Kress-corrected magnetic single-layer matrix entry.
Uses F(z)=A(z)log(z)+B(z) and log(z)=Lper+logratio, so the singular part is
integrated by Rsrc·A and the smooth remainder by wsrc·(B+A·logratio).
"""
@inline function _mag_kress_slp(tab,ν,G::MagneticKressGeomCache{T},bmag::T,i::Int,j::Int,pole_ws::MagneticPoleSubtractionWorkspace) where {T<:Real}
    if i==j
        A=Complex{T}(tab.a_log)
        B=Complex{T}(tab.R0)-_pole_sum(pole_ws,ν,0.0)
        return G.Rsrc[i,j]*A+G.wsrc[j]*(B+A*G.logratio[i,i])
    else
        z=Float64(G.r2[i,j]/(bmag*bmag))
        ph=G.phase[i,j]
        A=_eval_Alog(tab,z)
        B=_eval_Blog(tab,z)-_pole_sum(pole_ws,ν,z)
        return G.Rsrc[i,j]*Complex{T}(ph*A)+G.wsrc[j]*Complex{T}(ph*(B+A*G.logratio[i,j]))
    end
end

"""
    _mag_kress_dlp_src(tab,G,bmag,i,j,pole_ws)

Evaluate the Kress-corrected covariant source-normal double-layer entry.
Differentiates F(z)=A(z)log(z)+B(z), splitting the logarithmic part against
Lper=log(4sin²((σᵢ-σⱼ)/2)). The diagonal uses the finite source-normal limit
-Alog·curvnum/|γσ|².
"""
@inline function _mag_kress_dlp_src(tab,ν,G::MagneticKressGeomCache{T},bmag::T,i::Int,j::Int,pole_ws::MagneticPoleSubtractionWorkspace) where {T<:Real}
    if i==j
        A=Complex{T}(tab.a_log)
        return G.wsrc[j]*(-A*G.curvnum[i]/G.speed[i]^2)
    else
        z=Float64(G.r2[i,j]/(bmag*bmag))
        ph=G.phase[i,j]
        A=_eval_Alog(tab,z)
        B=_eval_Blog(tab,z)-_pole_sum(pole_ws,ν,z)
        Az=_eval_Alog_z(tab,z)
        Fz=_eval_Fz(tab,z)-_pole_z_sum(pole_ws,ν,z)
        dz=G.dzdn[i,j]
        dc=G.dchi[i,j]
        lr=G.logratio[i,j]
        lp=G.Lper[i,j]
        K1=ph*(im*dc*A+dz*Az)
        K2=ph*(im*dc*(B+A*lr)+dz*(Fz-Az*lp))
        return G.Rsrc[i,j]*Complex{T}(K1)+G.wsrc[j]*Complex{T}(K2)
    end
end

"""
    _magnetic_entry(tab,G,bmag,i,j,matrix_kind,pole_ws)

Return one Kress-corrected entry of the requested operator.
Supported kinds are `:slp`, `:dlp_src`, and `:cfie_src`. The CFIE entry is
Dᵢⱼ+i k(ν,B)Sᵢⱼ with k(ν,B)=2√ν/B.
"""
@inline function _magnetic_entry(tab,ν,G::MagneticKressGeomCache{T},bmag::T,i::Int,j::Int,matrix_kind::Symbol,pole_ws::MagneticPoleSubtractionWorkspace) where {T<:Real}
    if matrix_kind===:slp
        return _mag_kress_slp(tab,ν,G,bmag,i,j,pole_ws)
    elseif matrix_kind===:dlp_src
        return _mag_kress_dlp_src(tab,ν,G,bmag,i,j,pole_ws)
    elseif matrix_kind===:cfie_src
        α=k_from_ν_magnetic(ν,bmag)
        return _mag_kress_dlp_src(tab,ν,G,bmag,i,j,pole_ws)+im*α*_mag_kress_slp(tab,ν,G,bmag,i,j,pole_ws)
    else
        error("Unknown matrix_kind=$matrix_kind.")
    end
end

"""
    _magnetic_regular_image_entry(tab,G,bmag,i,j,matrix_kind,pole_ws)

Return one nonsingular symmetry-image entry.
The same operator choices as `_magnetic_entry` are supported, but no Kress
logarithmic correction is applied because image sources are away from the
singular copy.
"""
@inline function _magnetic_regular_image_entry(tab,ν,G::MagneticKressGeomCache{T},bmag::T,i::Int,j::Int,matrix_kind::Symbol,pole_ws::MagneticPoleSubtractionWorkspace) where {T<:Real}
    if matrix_kind===:slp
        return _mag_raw_regular_slp(tab,ν,G,bmag,i,j,pole_ws)
    elseif matrix_kind===:dlp_src
        return _mag_raw_regular_dlp_src(tab,ν,G,bmag,i,j,pole_ws)
    elseif matrix_kind===:cfie_src
        α=k_from_ν_magnetic(ν,bmag)
        return _mag_raw_regular_dlp_src(tab,ν,G,bmag,i,j,pole_ws)+im*α*_mag_raw_regular_slp(tab,ν,G,bmag,i,j,pole_ws)
    else
        error("Unknown matrix_kind=$matrix_kind.")
    end
end

"""
    _magnetic_add_jump!(A,ν,matrix_kind,use_unregularized)

Add the magnetic double-layer jump term.
For `:slp` nothing is added. In the regularized convention the diagonal addition
is 0.5cos(πν)I. In the unregularized convention the whole matrix is divided by
cos(πν) and the diagonal addition is 0.5I.
"""
function _magnetic_add_jump!(A,ν,matrix_kind::Symbol,use_unregularized::Bool)
    matrix_kind===:slp && return A
    c=cospi(ν)
    if use_unregularized
        A ./= c
        @inbounds for i in axes(A,1)
            A[i,i]+=0.5
        end
    else
        @inbounds for i in axes(A,1)
            A[i,i]+=0.5*c
        end
    end
    return A
end

"""
    construct_magnetic_operator_matrix!(A,pts,gws,kws;matrix_kind=:cfie_src,use_unregularized=false,multithreaded=true,pole_ws=nothing)

Assemble the full magnetic boundary operator on the complete periodic boundary.

For `matrix_kind=:slp`, this fills A with the Kress-corrected magnetic
single-layer matrix S_B(ν). For `:dlp_src`, it fills A with the covariant
source-normal double-layer matrix D_B(ν). For `:cfie_src`, it fills A with
D_B(ν)+i k(ν,B)S_B(ν), where k(ν,B)=2√ν/B.

The singular same-boundary logarithmic part is handled entrywise by
`_magnetic_entry`, using the split F(z)=Alog(z)log(z)+Blog(z),
z=|x-y|²/B², and the periodic Kress logarithm log(4sin²((σᵢ-σⱼ)/2)).
After assembly, the magnetic double-layer jump is added: in the regularized
form this is 0.5cos(πν)I, while in the unregularized form the matrix is divided
by cos(πν) and 0.5I is added.
"""
function construct_magnetic_operator_matrix!(A::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},gws::MagneticKressGeomWorkspace{T},kws::MagneticKressTaylorWorkspace;matrix_kind::Symbol=:cfie_src,use_unregularized::Bool=false,multithreaded::Bool=true,pole_ws=nothing) where {T<:Real}
    G=gws.G;N=gws.N;tab=kws.tab;ν=kws.ν
    fill!(A,zero(Complex{T}))
    @use_threads multithreading=(multithreaded&&N>=32) for j in 1:N
        @inbounds for i in 1:N
            A[i,j]=_magnetic_entry(tab,ν,G,gws.bmag,i,j,matrix_kind,pole_ws)
        end
    end
    _magnetic_add_jump!(A,ν,matrix_kind,use_unregularized)
    return A
end

"""
    construct_magnetic_operator_matrix!(A,pts,rgws,kws;matrix_kind=:cfie_src,use_unregularized=false,multithreaded=true,pole_ws=nothing)

Assemble the symmetry-reduced magnetic boundary operator.

The reduced matrix is indexed by the fundamental representatives `Ifund`. The
same-copy interaction between representatives is assembled with the full
periodic Kress correction, preserving the logarithmic quadrature tied to the
complete boundary indexing. All additional symmetry images of each source are
then added as regular nonsingular image terms using their symmetry phases
`fund_to_scale`.

This mirrors the hyperbolic reduced Kress strategy: only the physical singular
copy uses Kress splitting, while all symmetry copies are geometrically separated
and are therefore evaluated by `_magnetic_regular_image_entry`. The final jump
term is added in the same convention as the full matrix.
"""
function construct_magnetic_operator_matrix!(A::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},rgws::MagneticKressReducedGeomWorkspace{T},kws::MagneticKressTaylorWorkspace;matrix_kind::Symbol=:cfie_src,use_unregularized::Bool=false,multithreaded::Bool=true,pole_ws=nothing) where {T<:Real}
    full=rgws.full
    G=full.G
    tab=kws.tab
    m=rgws.m
    Ifund=rgws.Ifund
    ν=kws.ν
    fill!(A,zero(Complex{T}))
    @use_threads multithreading=(multithreaded && m>=32) for b in 1:m
        j=Ifund[b]
        @inbounds for a in 1:m
            i=Ifund[a]
            A[a,b]=_magnetic_entry(tab,ν,G,full.bmag,i,j,matrix_kind,pole_ws)
        end
    end
    @inbounds for b in 1:m
        j=Ifund[b]
        imgs=rgws.fund_to_full[b]
        scales=rgws.fund_to_scale[b]
        for a in 1:m
            i=Ifund[a]
            s=zero(Complex{T})
            for l in eachindex(imgs)
                q=imgs[l]
                q==j && continue
                s+=scales[l]*_magnetic_regular_image_entry(tab,ν,G,full.bmag,i,q,matrix_kind,pole_ws)
            end
            A[a,b]+=s
        end
    end
    _magnetic_add_jump!(A,ν,matrix_kind,use_unregularized)
    return A
end

"""
    adjoint_magnetic_operator_matrix!(A,D,pts,gws,kws;matrix_kind=:cfie_src,use_unregularized=false,multithreaded=true,pole_ws=nothing)

Assemble the weighted adjoint of the full magnetic boundary operator.

The source source-normal operator is first assembled into `D`. The adjoint
matrix is then formed by the quadrature-weight similarity transpose
Aᵢⱼ = Dⱼᵢ wⱼ/wᵢ, where wⱼ=|γσ(σⱼ)|dσ is stored as `gws.G.wsrc[j]`.
"""
function adjoint_magnetic_operator_matrix!(A::AbstractMatrix{Complex{T}},D::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},gws::MagneticKressGeomWorkspace{T},kws::MagneticKressTaylorWorkspace;matrix_kind::Symbol=:cfie_src,use_unregularized::Bool=false,multithreaded::Bool=true,pole_ws=nothing) where {T<:Real}
    construct_magnetic_operator_matrix!(D,pts,gws,kws;matrix_kind=matrix_kind,use_unregularized=use_unregularized,multithreaded=multithreaded,pole_ws=pole_ws)
    w=gws.G.wsrc
    N=gws.N
    @inbounds for i in 1:N, j in 1:N
        A[i,j]=D[j,i]*w[j]/w[i]
    end
    return A
end

"""
    adjoint_magnetic_operator_matrix!(A,D,pts,rgws,kws; matrix_kind=:cfie_src,use_unregularized=false,multithreaded=true,pole_ws=nothing)

Assemble the weighted adjoint of the reduced magnetic boundary operator.

The reduced source-normal operator is first assembled into `D` on the fundamental
representatives. The adjoint is then formed as Aₐᵦ = Dᵦₐ wⱼ/wᵢ, where
i=Ifund[a], j=Ifund[b], and w is the full-boundary source quadrature weight.
Thus the reduced adjoint uses the same physical weights as the full operator,
but restricted to the representative indices.

Use this matrix when `solve_vect` is requested in a symmetry sector, so that the
returned null vector corresponds to the adjoint boundary data in the reduced
basis.
"""
function adjoint_magnetic_operator_matrix!(A::AbstractMatrix{Complex{T}},D::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},rgws::MagneticKressReducedGeomWorkspace{T},kws::MagneticKressTaylorWorkspace;matrix_kind::Symbol=:cfie_src,use_unregularized::Bool=false,multithreaded::Bool=true,pole_ws=nothing) where {T<:Real}
    construct_magnetic_operator_matrix!(D,pts,rgws,kws;matrix_kind=matrix_kind,use_unregularized=use_unregularized,multithreaded=multithreaded,pole_ws=pole_ws)
    w=rgws.full.G.wsrc
    Ifund=rgws.Ifund
    m=rgws.m
    @inbounds for a in 1:m, b in 1:m
        i=Ifund[a]
        j=Ifund[b]
        A[a,b]=D[b,a]*w[j]/w[i]
    end
    return A
end

"""
    construct_matrices!(solver,A,pts,gws,kws,ν; matrix_kind=:cfie_src,use_unregularized=false,mp_dps=30,multithreaded=true,pole_ws=nothing)

Update the ν-dependent magnetic Taylor table and assemble the requested matrix.

The geometry workspace `gws` is ν-independent and is reused. The Taylor workspace
`kws` is rebuilt at the new ν with precision `mp_dps`, after which
`construct_magnetic_operator_matrix!` fills `A`. This is the standard reusable
assembly path for sweeps and contour methods.
"""
function construct_matrices!(solver::MagneticKressSolver,A::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},gws::Union{MagneticKressGeomWorkspace{T},MagneticKressReducedGeomWorkspace{T}},kws::MagneticKressTaylorWorkspace,ν;matrix_kind::Symbol=:cfie_src,use_unregularized=false,mp_dps::Int=30,multithreaded::Bool=true,pole_ws=nothing) where {T<:Real}
    update_magnetic_kress_taylor_workspace!(kws,ComplexF64(ν);mp_dps=mp_dps)
    return construct_magnetic_operator_matrix!(A,pts,gws,kws;matrix_kind=matrix_kind,use_unregularized=use_unregularized,multithreaded=multithreaded,pole_ws=pole_ws)
end

"""
    solve(solver,basis,A,pts,gws,kws,ν; matrix_kind=:cfie_src,use_unregularized=false,mp_dps=30,multithreaded=true,use_krylov=true,which=:svd,pole_ws=nothing)

Assemble the magnetic operator at ν and return a scalar spectral diagnostic.

For `which=:svd`, the returned value is the smallest singular-value diagnostic
of the assembled matrix, whose minima indicate magnetic billiard eigenvalues.
For determinant-based diagnostics, `which=:det` follows the same dispatch as the
other boundary-integral solvers. The supplied matrix `A` and workspaces are
reused to avoid allocation in sweeps.
"""
function solve(solver::MagneticKressSolver,basis::Ba,A::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},gws::Union{MagneticKressGeomWorkspace{T},MagneticKressReducedGeomWorkspace{T}},kws::MagneticKressTaylorWorkspace,ν;matrix_kind::Symbol=:cfie_src,use_unregularized=false,mp_dps::Int=30,multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:svd,pole_ws=nothing) where {T<:Real,Ba<:AbsBasis}
    @blas_1 construct_matrices!(solver,A,pts,gws,kws,ν;matrix_kind=matrix_kind,use_unregularized=use_unregularized,mp_dps=mp_dps,multithreaded=multithreaded,pole_ws=pole_ws)
    @svd_or_det_solve A use_krylov which MAX_BLAS_THREADS
end

"""
    solve(solver,basis,pts,gws,kws,ν; kwargs...)
    solve(solver,basis,pts,ws,ν; kwargs...)
    solve(solver,basis,A,pts,ws,ν; kwargs...)
    solve(solver,basis,pts,ν; kwargs...)

Convenience wrappers for the scalar magnetic spectral diagnostic.

These methods progressively allocate missing storage or workspaces: if `A` is
not supplied, a matrix of size `_workspace_dim(gws)` is allocated; if the
workspace tuple is supplied, it is unpacked; if no workspace is supplied, both
the geometry and Taylor workspaces are built internally. All keyword arguments
are forwarded to the main `solve` method.
"""
function solve(solver::MagneticKressSolver,basis::Ba,pts::BoundaryPointsCFIE{T},gws::Union{MagneticKressGeomWorkspace{T},MagneticKressReducedGeomWorkspace{T}},kws::MagneticKressTaylorWorkspace,ν;kwargs...) where {T<:Real,Ba<:AbsBasis}
    A=Matrix{Complex{T}}(undef,_workspace_dim(gws),_workspace_dim(gws))
    return solve(solver,basis,A,pts,gws,kws,ν;kwargs...)
end

function solve(solver::MagneticKressSolver,basis::Ba,pts::BoundaryPointsCFIE{T},ws::Tuple{<:Union{MagneticKressGeomWorkspace,MagneticKressReducedGeomWorkspace},MagneticKressTaylorWorkspace},ν;kwargs...) where {T<:Real,Ba<:AbsBasis}
    return solve(solver,basis,pts,ws[1],ws[2],ν;kwargs...)
end

function solve(solver::MagneticKressSolver,basis::Ba,A::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},ws::Tuple{<:Union{MagneticKressGeomWorkspace,MagneticKressReducedGeomWorkspace},MagneticKressTaylorWorkspace},ν;kwargs...) where {T<:Real,Ba<:AbsBasis}
    return solve(solver,basis,A,pts,ws[1],ws[2],ν;kwargs...)
end

function solve(solver::MagneticKressSolver,basis::Ba,pts::BoundaryPointsCFIE{T},ν;h=1e-5,P=6,Msmall=30,mp_dps::Int=30,kwargs...) where {T<:Real,Ba<:AbsBasis}
    gws,kws=build_magnetic_kress_workspace(solver,pts,ν;h=h,P=P,Msmall=Msmall,mp_dps=mp_dps)
    return solve(solver,basis,pts,gws,kws,ν;mp_dps=mp_dps,kwargs...)
end

"""
    solve_vect(solver,basis,A,pts,gws,kws,ν; matrix_kind=:cfie_src,use_unregularized=false,mp_dps=30,multithreaded=true,tol=1e-12,maxiter=2000,krylovdim=40, pole_ws=nothing)

Assemble the weighted adjoint magnetic operator and compute its null vector.

The Taylor table is first updated to ν. Then the adjoint matrix is assembled via
Aᵢⱼ=Dⱼᵢwⱼ/wᵢ, or the reduced analogue in a symmetry sector. Finally
`smallest_nullvec_krylov!` computes `(σ,u)`, where σ is the smallest singular
value and `u` is the corresponding adjoint boundary vector.

This should be used for boundary-function postprocessing, Husimi transforms and
wavefunction reconstruction, whereas `solve` is sufficient for scalar
eigenvalue detection.
"""
function solve_vect(solver::MagneticKressSolver,basis::Ba,A::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},gws::Union{MagneticKressGeomWorkspace{T},MagneticKressReducedGeomWorkspace{T}},kws::MagneticKressTaylorWorkspace,ν;matrix_kind::Symbol=:cfie_src,use_unregularized=false,mp_dps::Int=30,multithreaded::Bool=true,tol=1e-12,maxiter::Int=2000,krylovdim::Int=40,pole_ws=nothing) where {T<:Real,Ba<:AbsBasis}
    update_magnetic_kress_taylor_workspace!(kws,ComplexF64(ν);mp_dps=mp_dps)
    D=similar(A)
    @blas_1 adjoint_magnetic_operator_matrix!(A,D,pts,gws,kws;matrix_kind=matrix_kind,use_unregularized=use_unregularized,multithreaded=multithreaded,pole_ws=pole_ws)
    σ,u,_=smallest_nullvec_krylov!(A;nev=1,tol=tol,maxiter=maxiter,krylovdim=krylovdim)
    return σ,u
end

"""
    solve_vect(solver,basis,pts,gws,kws,ν; kwargs...)
    solve_vect(solver,basis,pts,ws,ν; kwargs...)
    solve_vect(solver,basis,A,pts,ws,ν; kwargs...)
    solve_vect(solver,basis,pts,ν; kwargs...)

Convenience wrappers for magnetic null-vector extraction.

These variants allocate missing matrix storage or build missing workspaces, then
delegate to the main adjoint `solve_vect` method. All keyword arguments are
forwarded, including `matrix_kind`, `use_unregularized`, `mp_dps`, and Krylov
parameters.
"""
function solve_vect(solver::MagneticKressSolver,basis::Ba,pts::BoundaryPointsCFIE{T},gws::Union{MagneticKressGeomWorkspace{T},MagneticKressReducedGeomWorkspace{T}},kws::MagneticKressTaylorWorkspace,ν;kwargs...) where {T<:Real,Ba<:AbsBasis}
    A=Matrix{Complex{T}}(undef,_workspace_dim(gws),_workspace_dim(gws))
    return solve_vect(solver,basis,A,pts,gws,kws,ν;kwargs...)
end

function solve_vect(solver::MagneticKressSolver,basis::Ba,pts::BoundaryPointsCFIE{T},ws::Tuple{<:Union{MagneticKressGeomWorkspace,MagneticKressReducedGeomWorkspace},MagneticKressTaylorWorkspace},ν;kwargs...) where {T<:Real,Ba<:AbsBasis}
    return solve_vect(solver,basis,pts,ws[1],ws[2],ν;kwargs...)
end

function solve_vect(solver::MagneticKressSolver,basis::Ba,A::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},ws::Tuple{<:Union{MagneticKressGeomWorkspace,MagneticKressReducedGeomWorkspace},MagneticKressTaylorWorkspace},ν;kwargs...) where {T<:Real,Ba<:AbsBasis}
    return solve_vect(solver,basis,A,pts,ws[1],ws[2],ν;kwargs...)
end

function solve_vect(solver::MagneticKressSolver,basis::Ba,pts::BoundaryPointsCFIE{T},ν;h=1e-5,P=6,Msmall=30,mp_dps::Int=30,kwargs...) where {T<:Real,Ba<:AbsBasis}
    gws,kws=build_magnetic_kress_workspace(solver,pts,ν;h=h,P=P,Msmall=Msmall,mp_dps=mp_dps)
    return solve_vect(solver,basis,pts,gws,kws,ν;mp_dps=mp_dps,kwargs...)
end

"""
    construct_boundary_matrices!(Tbufs,solver,pts,zj; matrix_kind=:cfie_src,use_unregularized=false,h=1e-5,P=6,Msmall=30,mp_dps=30,multithreaded=true, timeit=false,pole_ws=nothing)

Assemble magnetic boundary matrices for several spectral parameters.

The ν-independent geometry workspace is built once from `pts`. A single Taylor
workspace is then updated for each `zj[q]`, and the corresponding matrix is
written into `Tbufs[q]`. This is the allocation-minimizing path used by Beyn
contour integration and multi-point spectral sweeps.

Each buffer must have size `_workspace_dim(gws) × _workspace_dim(gws)`, where
the dimension is either the full boundary size or the reduced symmetry-sector
size.
"""
function construct_boundary_matrices!(Tbufs::Vector{Matrix{ComplexF64}},solver::MagneticKressSolver,pts::BoundaryPointsCFIE{T},zj::AbstractVector{ComplexF64};matrix_kind::Symbol=:cfie_src,use_unregularized=false,h=1e-5,P=6,Msmall=30,mp_dps::Int=30,multithreaded::Bool=true,timeit::Bool=false,pole_ws=nothing) where {T<:Real}
    @blas_1 begin
        @benchit timeit=timeit "Magnetic Kress geometry workspace" gws=build_magnetic_kress_geom_workspace(solver,pts)
        n=_workspace_dim(gws)
        zmax=gws isa MagneticKressGeomWorkspace ? gws.G.zmax : gws.full.G.zmax
        kws=build_magnetic_kress_taylor_workspace(zj[1],zmax;h=h,P=P,Msmall=Msmall,mp_dps=mp_dps)
        @inbounds for q in eachindex(zj)
            @assert size(Tbufs[q])==(n,n) "Tbufs[$q] has size $(size(Tbufs[q])), but MagneticCFIE_kress requires ($n,$n)."
            @benchit timeit=timeit "Magnetic Kress Taylor update" update_magnetic_kress_taylor_workspace!(kws,zj[q];mp_dps=mp_dps)
            @benchit timeit=timeit "Magnetic Kress matrix assembly" construct_magnetic_operator_matrix!(Tbufs[q],pts,gws,kws;matrix_kind=matrix_kind,use_unregularized=use_unregularized,multithreaded=multithreaded,pole_ws=pole_ws)
        end
    end
    return nothing
end