#################################################################################
# Hyperbolic Helmholtz DLP kernel on the Poincaré disk
#
# Constructs double-layer potential (DLP) Fredholm matrices for the operator
#
#    (Δ_H + 1/4 + k^2) u = 0
#
# in the Poincaré disk model, using the Green's function
#
#    G_k(z,z0) = (1 / (2π)) * Q_ν(χ(z,z0)),   ν = -1/2 + i k
#
# where
#    χ(z,z0) = cosh d_H(z,z0) = 1 + 2|z - z0|^2 / ((1 - |z|^2) (1 - |z0|^2)).
#
#
# The DLP kernel we discretize is
#
#    K_k(z_i, z_j) = ∂/∂n_E(z_i) G_k(z_i, z_j)
#                  = (1/(2π)) * y(d) * ∂_{n_E} d(z_i, z_j)
#
#    y(d) = d/dd Q_ν(cosh d),   d = d_H(z_i, z_j)
#
# where n_E is the Euclidean unit normal at the *target* point z_i.
#
# IMPORTANT:
#   - We work entirely in Euclidean coordinates (x,y) in the disk due to the
#     conformal invariance of the DLP.
#   - For the double-layer, the hyperbolic metric factor cancels against ds_H,
#     so we use Euclidean normals and Euclidean quadrature weights ds (bp.ds).
#   - The boundary geometry (bp.xy, bp.normal, bp.curvature) is Euclidean.
#
# API
# -----------------------------------------------------------------
# _one_k_nosymm_DLP_hyperbolic!  - build one DLP matrix for a single complex k
# _all_k_nosymm_DLP_hyperbolic!  - build DLP matrices for a vector of ks
# compute_kernel_matrices_DLP_hyperbolic! - public entry points (single / multiple k)
#
# M0 / 22/12/2025
#################################################################################

################################################################################
# DEFAULT HYPERBOLIC KERNEL FUN FOR GENERIC BUILDERS
#
# Signature is identical to the Euclidean custom kernel_fun! used in
# _all_k_nosymm_DLP_chebyshev!(..., kernel_fun!) etc.
#
# IMPORTANT:
#   * This function DOES NOT know about curvature, so it cannot set the
#     diagonal κ/(2π) itself. For default hyperbolic DLP we use specialized
#     builders that see bp.curvature and handle the diagonal there.
#   * This function uses the *target* normal (nxi, nyi) (Bäcker convention).
################################################################################

################################################################################
# invlambda2_poincare
#
# PURPOSE
#   Compute the conformal prefactor 1/λ(x,y)^2 for the Poincaré disk model
#   (constant curvature -1).
#
#   Metric:    ds^2 = λ(x,y)^2 (dx^2 + dy^2),   λ(x,y) = 2 / (1 - r^2), r^2=x^2+y^2
#   Therefore: 1/λ^2 = (1 - r^2)^2 / 4
#
# INPUTS
#   x::T, y::T        (T<:Real) Euclidean coordinates in disk |x|^2+|y|^2 < 1
#
# OUTPUTS
#   invλ2::T          (T<:Real) 1/λ(x,y)^2
################################################################################
@inline function invlambda2_poincare(x::T,y::T) where {T<:Real}
    a=one(T)-muladd(x,x,y*y) # a = 1 - r^2
    return (a*a)/T(4)
end

################################################################################
# hyperbolic_cosh_d_poincare
#
# PURPOSE
#   Compute cosh(d_H(x_i,x_j)) in the Poincaré disk model using the closed form:
#
#     cosh d = 1 + 2 |x_i - x_j|^2 / ((1 - |x_i|^2)(1 - |x_j|^2))
#
# INPUTS
#   xi::T, yi::T      (T<:Real) target point in Euclidean coordinates
#   xj::T, yj::T      (T<:Real) source point in Euclidean coordinates
#
# OUTPUTS
#   χ::T              (T<:Real) χ = cosh(d_H)
################################################################################
@inline function hyperbolic_cosh_d_poincare(xi::T,yi::T,xj::T,yj::T) where {T<:Real}
    dx=xi-xj
    dy=yi-yj
    Δ2=muladd(dx,dx,dy*dy)
    r2=muladd(xi,xi,yi*yi)
    r02=muladd(xj,xj,yj*yj)
    denom=(one(T)-r2)*(one(T)-r02)
    χ=one(T)+2*Δ2/denom
    return max(χ,one(T))
end

################################################################################
# hyperbolic_distance_poincare
#
# PURPOSE
#   Compute hyperbolic distance d_H(x_i,x_j) in the Poincaré disk model:
#       d = acosh(cosh d)
#
# INPUTS
#   xi::T, yi::T      (T<:Real) target point
#   xj::T, yj::T      (T<:Real) source point
#
# OUTPUTS
#   d::T              (T<:Real) hyperbolic distance d >= 0
################################################################################

@inline function hyperbolic_distance_poincare(xi::T,yi::T,xj::T,yj::T) where {T<:Real}
    χ=hyperbolic_cosh_d_poincare(xi,yi,xj,yj)
    return acosh(χ)
end

################################################################################
# hyperbolic_dn_d_target
#
# PURPOSE
#   Compute ∂d/∂n_x at the TARGET point x=(xi,yi) for the hyperbolic distance
#   d(x,y) in the Poincaré disk model (curvature -1).
#
#   Here n_x = (nxi,nyi) is the outward Euclidean unit normal at the target.
#
# INPUTS
#   xi::T, yi::T      (T<:Real) target point x
#   xj::T, yj::T      (T<:Real) source point y
#   nxi::T, nyi::T    (T<:Real) Euclidean unit normal at target
#
# OUTPUTS
#   ddn::T            (T<:Real) ddn = ∂d(x,y)/∂n_x
#
# LOGIC
#   - Uses closed-form cosh(d) in terms of Euclidean coordinates.
#   - Differentiates cosh(d) w.r.t. target coordinates, then:
#       d = acosh(c), so ∂d = (∂c)/sinh(d)  with sinh(d)=sqrt(c^2-1).
#   - Contracts ∇_x d with the Euclidean normal n_x.
################################################################################
@inline function hyperbolic_dn_d_target(xi::T,yi::T,xj::T,yj::T,nxi::T,nyi::T) where{T<:Real}
    ax=one(T)-muladd(xi,xi,yi*yi)
    bx=one(T)-muladd(xj,xj,yj*yj)
    dx=xi-xj;dy=yi-yj
    r2=muladd(dx,dx,dy*dy)
    c=one(T)+2*r2/(ax*bx)
    sh=sqrt(max(c*c-one(T),zero(T)))
    dotdxn=muladd(dx,nxi,dy*nyi)                 # (x-x')·n
    dotxn=muladd(xi,nxi,yi*nyi)                  # x·n
    return (4/(ax*bx))*dotdxn/sh+(4*r2/(bx*ax*ax))*dotxn/sh
end

################################################################################
# hyperbolic_dn_d_source
#
# PURPOSE
#   Compute ∂d/∂n_y at the SOURCE point y=(xj,yj) for the hyperbolic distance
#   d(x,y) in the Poincaré disk model (curvature -1).
#
#   Here n_y = (nxj,nyj) is the outward Euclidean unit normal at the source.
#
# INPUTS
#   xi::T, yi::T      (T<:Real) target point x
#   xj::T, yj::T      (T<:Real) source point y
#   nxj::T, nyj::T    (T<:Real) Euclidean unit normal at source
#
# OUTPUTS
#   ddn::T            (T<:Real) ddn = ∂d(x,y)/∂n_y
################################################################################
@inline function hyperbolic_dn_d_source(xi::T,yi::T,xj::T,yj::T,nxj::T,nyj::T) where {T<:Real}
    ai=one(T)-muladd(xi,xi,yi*yi)
    aj=one(T)-muladd(xj,xj,yj*yj)
    dx=xj-xi
    dy=yj-yi
    r2=muladd(dx,dx,dy*dy)
    c=one(T)+2*r2/(ai*aj) # cosh(d)
    sh=sqrt(max(c*c-one(T),zero(T))) # sinh(d)
    dotdxn=muladd(dx,nxj,dy*nyj) # (xj-xi)·n_j
    dotxjn=muladd(xj,nxj,yj*nyj) # xj·n_j
    return (4/(ai*aj))*dotdxn/sh+(4*r2/(ai*aj*aj))*dotxjn/sh
end

################################################################################
# hyperbolic_dlp_kernel_scalar_target
#
# PURPOSE
#   Evaluate scalar Double-Layer Potential (DLP) kernel for hyperbolic Helmholtz
#   at target-normal derivative:
#
#     K(x_i,x_j) = (1/2π) * (d/dd Q_ν(cosh d)) * (∂d/∂n_x)
#
#   where d = d_H(x_i,x_j), ν = -1/2 + i k (encoded inside tab).
#
# INPUTS
#   tab::QTaylorTable   Precomputed table for Q_ν and its d-derivatives vs d
#   xi,yi::T            target point
#   xj,yj::T            source point
#   nxi,nyi::T          Euclidean unit normal at target
#
# OUTPUTS
#   Kij::T              Scalar kernel value 
################################################################################
@inline function hyperbolic_dlp_kernel_scalar_target(tab::QTaylorTable,xi::T,yi::T,xj::T,yj::T,nxi::T,nyi::T) where{T<:Real}
    d=Float64(hyperbolic_distance_poincare(xi,yi,xj,yj))
    y=_eval_dQdd(tab,d)
    dn=hyperbolic_dn_d_target(xi,yi,xj,yj,nxi,nyi)
    return (y*dn)*inv2π
end

################################################################################
# hyperbolic_dlp_kernel_scalar_source
#
# PURPOSE
#   Evaluate scalar Double-Layer Potential (DLP) kernel for hyperbolic Helmholtz
#   at source-normal derivative (the form used in your BIM assembly):
#
#     K(x_i,x_j) = (1/2π) * (d/dd Q_ν(cosh d)) * (∂d/∂n_y)
#
# INPUTS
#   tab::QTaylorTable   Table for Q_ν and derivatives
#   xi,yi::T            target point x_i
#   xj,yj::T            source point y_j
#   nxj,nyj::T          Euclidean unit normal at source point
#
# OUTPUTS
#   Kij::T              Scalar kernel value
@inline function hyperbolic_dlp_kernel_scalar_source(tab::QTaylorTable,xi::T,yi::T,xj::T,yj::T,nxj::T,nyj::T) where {T<:Real}
    d=Float64(hyperbolic_distance_poincare(xi,yi,xj,yj))
    y=_eval_dQdd(tab,d)                    # d/dd Qν(cosh d)
    dn=hyperbolic_dn_d_source(xi,yi,xj,yj,nxj,nyj)
    return (y*dn)*inv2π
end

################################################################################
# hyperbolic_slp_kernel_scalar_target
#
# PURPOSE
#   Evaluate scalar Single-Layer Potential (SLP) kernel in hyperbolic geometry:
#
#     S(x_i,x_j) = (1/2π) * Q_ν(cosh d_H(x_i,x_j))
#
# INPUTS
#   tab::QTaylorTable   Table for Q_ν(d)
#   xi,yi::T            target point
#   xj,yj::T            source point
#
# OUTPUTS
#   Sij::T              Scalar kernel value
################################################################################
@inline function hyperbolic_slp_kernel_scalar_target(tab::QTaylorTable,xi::T,yi::T,xj::T,yj::T) where {T<:Real}
    d=Float64(hyperbolic_distance_poincare(xi,yi,xj,yj))
    return _eval_Q(tab,d)*inv2π
end

################################################################################
# κ_geodesic_poincare
#
# PURPOSE
#   Convert Euclidean curvature κ_E of the embedded boundary curve in the
#   Poincaré disk into the geodesic curvature κ_g (hyperbolic metric).
#
#   Conformal relation for g = λ^2 δ:
#     κ_g = (1/λ)(κ_E + ∂_n log λ)
#
#   For λ = 2/(1-r^2) this simplifies to:
#     κ_g(x) = ((1-r^2)/2) * κ_E(x) + (x nx + y ny)
#
# INPUTS
#   x::T, y::T          boundary point in Euclidean coords
#   nx::T, ny::T        Euclidean unit normal at boundary point (outward)
#   κE::T               Euclidean curvature of boundary curve at that point
#
# OUTPUTS
#   κg::T               Geodesic curvature in hyperbolic metric (curvature -1)
################################################################################
@inline function κ_geodesic_poincare(x::T,y::T,nx::T,ny::T,κE::T) where {T<:Real}
    r2=muladd(x,x,y*y)
    return ((one(T)-r2)*κE)/2+muladd(x,nx,y*ny)
end

################################################################################
# dlp_diag_source_normal_poincare
#
# PURPOSE
#   Provide the principal value (PV) diagonal limit for the SOURCE-normal DLP
#   kernel in hyperbolic geometry. This is the geometric correction term that
#   replaces the singular evaluation.
#
#   Convention used here:
#     K_ii = - κ_g / (4π) = - κ_g / (2*TWO_PI)
#
# INPUTS
#   x::T, y::T          boundary point
#   nx::T, ny::T        Euclidean unit normal at boundary point
#   κE::T               Euclidean curvature at boundary point
#
# OUTPUTS
#   Kdiag::T            real diagonal PV limit contribution
################################################################################
@inline function dlp_diag_source_normal_poincare(x::T,y::T,nx::T,ny::T,κE::T) where {T<:Real}
    κg=κ_geodesic_poincare(x,y,nx,ny,κE)
    return -κg/(2*TWO_PI)
end

################################################################################
# _all_k_nosymm_DLP_hyperbolic!
#
# PURPOSE
#   Assemble DLP Fredholm matrices for multiple wavenumbers (tabs) for a single
#   boundary discretization (bp), WITHOUT symmetry images.
#
#   This fills Ks[m] with the discretized DLP operator K(k_m) using:
#     - off-diagonal: hyperbolic_dlp_kernel_scalar_source(...)
#     - diagonal PV : dlp_diag_source_normal_poincare(...)
#
# INPUTS
#   Ks::Vector{Matrix{Complex{T}}}
#       Preallocated matrices, length Mk, each N×N
#   bp::BoundaryPointsBIM{T}
#       Boundary discretization struct containing:
#         bp.xy         :: Vector{SVector{2,T}}  collocation points
#         bp.normal     :: Vector{SVector{2,T}}  outward normals
#         bp.curvature  :: Vector{T}             Euclidean curvature κ_E at points
#         bp.ds         :: Vector{T}             quadrature weights (used later)
#   tabs::Vector{QTaylorTable}
#       Precomputed Q_ν tables (one per k)
#
# KEYWORD INPUTS
#   multithreaded::Bool
#       Enable threading over m using @use_threads macro.
#
# OUTPUTS
#   nothing (fills Ks in-place)
################################################################################
function _all_k_nosymm_DLP_hyperbolic!(Ks::Vector{Matrix{Complex{T}}},bp::BoundaryPointsBIM{T},tabs::Vector{QTaylorTable};multithreaded::Bool=true) where {T<:Real}
    Mk=length(tabs)
    N=length(bp.xy)
    @use_threads multithreading=multithreaded for m in 1:Mk
        K=Ks[m]
        tab=tabs[m]
        fill!(K,zero(eltype(K)))
        @inbounds for i in 1:N
            xi,yi=bp.xy[i]
            κi=bp.curvature[i]
            for j in 1:N
                xj,yj=bp.xy[j]
                if i==j
                    nxj,nyj=bp.normal[j] 
                    K[i,i]=Complex{T}(dlp_diag_source_normal_poincare(xj,yj,nxj,nyj,κi),zero(T))
                else
                    nxj,nyj=bp.normal[j] 
                    K[i,j]=hyperbolic_dlp_kernel_scalar_source(tab,xi,yi,xj,yj,nxj,nyj)
                end
            end
        end
    end
    return nothing
end

################################################################################
# _one_k_nosymm_DLP_hyperbolic!
#
# PURPOSE
#   Assemble a single DLP Fredholm matrix K for one wavenumber table (tab),
#   WITHOUT symmetry images.
#
# INPUTS
#   K::Matrix{Complex{T}}   Preallocated N×N output
#   bp::BoundaryPointsBIM{T} boundary discretization
#   tab::QTaylorTable       Q_ν table for this k
#
# KEYWORD INPUTS
#   multithreaded::Bool     Thread over i using @use_threads
#
# OUTPUTS
#   nothing (fills K in-place)
################################################################################
function _one_k_nosymm_DLP_hyperbolic!(K::Matrix{Complex{T}},bp::BoundaryPointsBIM{T},tab::QTaylorTable;multithreaded::Bool=true) where {T<:Real}
    N=length(bp.xy)
    fill!(K,zero(eltype(K)))
    @use_threads multithreading=multithreaded for i in 1:N
        xi,yi=bp.xy[i]
        κi=bp.curvature[i]
        @inbounds for j in 1:N
            xj,yj=bp.xy[j]
            if i==j
                nxj,nyj=bp.normal[j]
                K[i,i]=Complex{T}(dlp_diag_source_normal_poincare(xj,yj,nxj,nyj,κi),zero(T))
            else
                nxj,nyj=bp.normal[j]
                K[i,j]=hyperbolic_dlp_kernel_scalar_source(tab,xi,yi,xj,yj,nxj,nyj)
            end
        end
    end
    return nothing
end

################################################################################
# _all_k_reflection_DLP_hyperbolic!
#
# PURPOSE
#   Assemble DLP Fredholm matrices for multiple wavenumbers INCLUDING reflection
#   symmetry images.
#
# INPUTS
#   Ks::Vector{Matrix{Complex{T}}}
#       Preallocated, length Mk, each N×N (filled in-place)
#   bp::BoundaryPointsBIM{T}
#       Boundary discretization:
#         bp.xy        :: Vector{SVector{2,T}}  collocation points (source/target)
#         bp.normal    :: Vector{SVector{2,T}}  Euclidean normals at source points
#         bp.curvature :: Vector{T}             Euclidean curvature at points
#         bp.shift_x   :: T                     reflection axis x-shift (if any)
#         bp.shift_y   :: T                     reflection axis y-shift (if any)
#   sym::Reflection
#       Reflection symmetry descriptor used by QuantumBilliards (encodes which
#       axes are reflected, and the parity sector via its internal sign)
#   tabs::Vector{QTaylorTable}
#       Q_ν tables, one per wavenumber
#
# KEYWORD INPUTS
#   multithreaded::Bool
#       Enable threading over i.
#
# OUTPUTS
#   nothing (fills Ks in-place)
################################################################################
function _all_k_reflection_DLP_hyperbolic!(Ks::Vector{Matrix{Complex{T}}},bp::BoundaryPointsBIM{T},sym::Reflection,tabs::Vector{QTaylorTable};multithreaded::Bool=true) where {T<:Real}
    _all_k_nosymm_DLP_hyperbolic!(Ks,bp,tabs;multithreaded)
    Mk=length(tabs)
    N=length(bp.xy)
    tol2=(eps(T))^2
    shift_x=bp.shift_x
    shift_y=bp.shift_y
    ops=_reflect_ops_and_scales(T,sym)
    nth=Threads.nthreads()
    pt_tls=[zeros(T,2) for _ in 1:nth]
    @use_threads multithreading=multithreaded for i in 1:N
        xi,yi=bp.xy[i]
        tid=Threads.threadid()
        pt=pt_tls[tid]
        @inbounds for j in 1:N
            xj0,yj0=bp.xy[j]
            nx0,ny0=bp.normal[j]
            @inbounds for (op,scale_r) in ops
                if op==1
                    x_reflect_point!(pt,xj0,yj0,shift_x)
                    nxr=-nx0;nyr=ny0
                elseif op==2
                    y_reflect_point!(pt,xj0,yj0,shift_y)
                    nxr=nx0;nyr=-ny0
                else
                    xy_reflect_point!(pt,xj0,yj0,shift_x,shift_y)
                    nxr=-nx0;nyr=-ny0
                end
                xjr,yjr=pt[1],pt[2]
                dx=xi-xjr
                dy=yi-yjr
                d2=muladd(dx,dx,dy*dy)
                if d2>tol2
                    sc=Complex{T}(scale_r,zero(T))
                    @inbounds for m in 1:Mk
                        tab=tabs[m]
                        val=hyperbolic_dlp_kernel_scalar_source(tab,xi,yi,xjr,yjr,nxr,nyr)*sc
                        Ks[m][i,j]+=val
                    end
                end
            end
        end
    end
    return nothing
end

################################################################################
# _one_k_reflection_DLP_hyperbolic!
#
# PURPOSE
#   Assemble a single DLP Fredholm matrix K for one wavenumber INCLUDING
#   reflection symmetry images.
#
# INPUTS
#   K::Matrix{Complex{T}}   Preallocated N×N output
#   bp::BoundaryPointsBIM{T}
#   sym::Reflection
#   tab::QTaylorTable
#
# KEYWORD INPUTS
#   multithreaded::Bool
#
# OUTPUTS
#   nothing (fills Ks in-place)
################################################################################
function _one_k_reflection_DLP_hyperbolic!(K::Matrix{Complex{T}},bp::BoundaryPointsBIM{T},sym::Reflection,tab::QTaylorTable;multithreaded::Bool=true) where {T<:Real}
    _one_k_nosymm_DLP_hyperbolic!(K,bp,tab;multithreaded)
    N=length(bp.xy)
    tol2=(eps(T))^2
    shift_x=bp.shift_x
    shift_y=bp.shift_y
    ops=_reflect_ops_and_scales(T,sym)
    pt=[zero(T),zero(T)]
    @use_threads multithreading=multithreaded for i in 1:N
        xi,yi=bp.xy[i]
        @inbounds for j in 1:N
            xj0,yj0=bp.xy[j]
            nx0,ny0=bp.normal[j] 
            @inbounds for (op,scale_r) in ops
                if op==1
                    x_reflect_point!(pt,xj0,yj0,shift_x)
                    nxr=-nx0;nyr=ny0
                elseif op==2
                    y_reflect_point!(pt,xj0,yj0,shift_y)
                    nxr=nx0;nyr=-ny0
                else
                    xy_reflect_point!(pt,xj0,yj0,shift_x,shift_y)
                    nxr=-nx0;nyr=-ny0
                end
                xjr,yjr=pt[1],pt[2]
                dx=xi-xjr
                dy=yi-yjr
                d2=muladd(dx,dx,dy*dy)
                if d2>tol2
                    sc=Complex{T}(scale_r,zero(T))
                    val=hyperbolic_dlp_kernel_scalar_source(tab,xi,yi,xjr,yjr,nxr,nyr)*sc
                    K[i,j]+=val
                end
            end
        end
    end
    return nothing
end

################################################################################
# _all_k_rotation_DLP_hyperbolic!
#
# PURPOSE
#   Assemble DLP Fredholm matrices for multiple wavenumbers INCLUDING rotation
#   symmetry images (Cn symmetry).
#
#   Strategy:
#     1) Fill no-symmetry contributions (physical boundary itself).
#     2) Add contributions from rotated copies of each source point:
#          y -> R^l(y),  l = 1,...,n-1
#        with phase/character factor χ[l] for the chosen symmetry sector.
#
# INPUTS
#   Ks::Vector{Matrix{Complex{T}}}   (Mk matrices, each N×N) filled in-place
#   bp::BoundaryPointsBIM{T}         boundary discretization
#   sym::Rotation                   rotation symmetry descriptor (n-fold, sector m)
#   tabs::Vector{QTaylorTable}       Q_ν tables for each wavenumber
#
# KEYWORD INPUTS
#   multithreaded::Bool
#
# OUTPUTS
#   nothing
################################################################################
function _all_k_rotation_DLP_hyperbolic!(Ks::Vector{Matrix{Complex{T}}},bp::BoundaryPointsBIM{T},sym::Rotation,tabs::Vector{QTaylorTable};multithreaded::Bool=true) where {T<:Real}
    _all_k_nosymm_DLP_hyperbolic!(Ks,bp,tabs;multithreaded)
    Mk=length(tabs)
    N=length(bp.xy)
    tol2=(eps(T))^2
    cx,cy=sym.center
    ctab,stab,χ=_rotation_tables(T,sym.n,mod(sym.m,sym.n))
    nth=Threads.nthreads()
    pt_tls=[zeros(T,2) for _ in 1:nth]
    @use_threads multithreading=multithreaded for i in 1:N
        xi,yi=bp.xy[i]
        tid=Threads.threadid()
        pt=pt_tls[tid]
        @inbounds for j in 1:N
            xj0,yj0=bp.xy[j]
            nx0,ny0=bp.normal[j] 
            @inbounds for l in 2:sym.n
                rot_point!(pt,xj0,yj0,cx,cy,ctab[l],stab[l])
                xjr,yjr=pt[1],pt[2]
                nxr=ctab[l]*nx0-stab[l]*ny0
                nyr=stab[l]*nx0+ctab[l]*ny0
                dx=xi-xjr
                dy=yi-yjr
                d2=muladd(dx,dx,dy*dy)
                if d2>tol2
                    phase=χ[l]
                    @inbounds for m in 1:Mk
                        tab=tabs[m]
                        val=hyperbolic_dlp_kernel_scalar_source(tab,xi,yi,xjr,yjr,nxr,nyr)*phase
                        Ks[m][i,j]+=val
                    end
                end
            end
        end
    end
    return nothing
end

################################################################################
# _one_k_rotation_DLP_hyperbolic!
#
# PURPOSE
#   Assemble a single DLP Fredholm matrix K for one wavenumber INCLUDING rotation
#   symmetry images.
#
# INPUTS
#   K::AbstractMatrix{Complex{T}}  Preallocated N×N output 
#   bp::BoundaryPointsBIM{T}
#   sym::Rotation
#   tab::QTaylorTable
#
# KEYWORD INPUTS
#   multithreaded::Bool
#
# OUTPUTS
#   nothing
################################################################################
function _one_k_rotation_DLP_hyperbolic!(K::AbstractMatrix{Complex{T}},bp::BoundaryPointsBIM{T},sym::Rotation,tab::QTaylorTable;multithreaded::Bool=true) where {T<:Real}
    _one_k_nosymm_DLP_hyperbolic!(K,bp,tab;multithreaded)
    N=length(bp.xy)
    tol2=(eps(T))^2
    cx,cy=sym.center
    ctab,stab,χ=_rotation_tables(T,sym.n,mod(sym.m,sym.n))
    pt=[zero(T),zero(T)]
    @use_threads multithreading=multithreaded for i in 1:N
        xi,yi=bp.xy[i]
        @inbounds for j in 1:N
            xj0,yj0=bp.xy[j]
            nx0,ny0=bp.normal[j] 
            @inbounds for l in 2:sym.n
                rot_point!(pt,xj0,yj0,cx,cy,ctab[l],stab[l])
                xjr,yjr=pt[1],pt[2]
                nxr=ctab[l]*nx0-stab[l]*ny0
                nyr=stab[l]*nx0+ctab[l]*ny0
                dx=xi-xjr
                dy=yi-yjr
                d2=muladd(dx,dx,dy*dy)
                if d2>tol2
                    phase=χ[l]
                    val=hyperbolic_dlp_kernel_scalar_source(tab,xi,yi,xjr,yjr,nxr,nyr)*phase
                    K[i,j]+=val
                end
            end
        end
    end
    return nothing
end

function compute_kernel_matrices_DLP_hyperbolic!(Ks::Vector{Matrix{Complex{T}}},bp::BoundaryPointsBIM{T},tabs::Vector{QTaylorTable};multithreaded::Bool=true) where {T<:Real}
    return _all_k_nosymm_DLP_hyperbolic!(Ks,bp,tabs;multithreaded)
end

function compute_kernel_matrices_DLP_hyperbolic!(K::Matrix{Complex{T}},bp::BoundaryPointsBIM{T},tab::QTaylorTable;multithreaded::Bool=true) where {T<:Real}
    return _one_k_nosymm_DLP_hyperbolic!(K,bp,tab;multithreaded)
end

function compute_kernel_matrices_DLP_hyperbolic!(Ks::Vector{Matrix{Complex{T}}},bp::BoundaryPointsBIM{T},sym::Reflection,ks::Vector{QTaylorTable};multithreaded::Bool=true) where {T<:Real}
    return _all_k_reflection_DLP_hyperbolic!(Ks,bp,sym,ks;multithreaded)
end

function compute_kernel_matrices_DLP_hyperbolic!(K::Matrix{Complex{T}},bp::BoundaryPointsBIM{T},sym::Reflection,tab::QTaylorTable;multithreaded::Bool=true) where {T<:Real}
    return _one_k_reflection_DLP_hyperbolic!(K,bp,sym,tab;multithreaded)
end

function compute_kernel_matrices_DLP_hyperbolic!(Ks::Vector{Matrix{Complex{T}}},bp::BoundaryPointsBIM{T},sym::Rotation,ks::Vector{QTaylorTable};multithreaded::Bool=true) where {T<:Real}
    return _all_k_rotation_DLP_hyperbolic!(Ks,bp,sym,ks;multithreaded)
end

function compute_kernel_matrices_DLP_hyperbolic!(K::Matrix{Complex{T}},bp::BoundaryPointsBIM{T},sym::Rotation,tab::QTaylorTable;multithreaded::Bool=true) where {T<:Real}
    return _one_k_rotation_DLP_hyperbolic!(K,bp,sym,tab;multithreaded)
end

################################################################################
# compute_kernel_matrices_DLP_hyperbolic!  (dispatch wrapper)
#
# PURPOSE
#   User-facing convenience wrapper that dispatches to the appropriate DLP
#   hyperbolic assembly routine depending on the symmetry specification.
#
# INPUTS (MULTI-k)
#   Ks::Vector{Matrix{Complex{T}}}
#   bp::BoundaryPointsBIM{T}
#   symmetry::Union{Vector{Any},Nothing}
#       - nothing or Vector with one symmetry element (Reflection or Rotation)
#   tabs::Vector{QTaylorTable}
#
# KEYWORD INPUTS
#   multithreaded::Bool
#   kernel_fun::Union{Symbol,Function} (currently unused; kept for API symmetry)
#
# OUTPUTS
#   nothing (fills Ks)
################################################################################
function compute_kernel_matrices_DLP_hyperbolic!(Ks::Vector{Matrix{Complex{T}}},bp::BoundaryPointsBIM{T},symmetry::Union{Vector{Any},Nothing},tabs::Vector{QTaylorTable};multithreaded::Bool=true) where {T<:Real}
    if symmetry===nothing
        return _all_k_nosymm_DLP_hyperbolic!(Ks,bp,tabs;multithreaded)
    else
        try
            s=symmetry[1]
            if s isa Reflection
                return _all_k_reflection_DLP_hyperbolic!(Ks,bp,s,tabs;multithreaded)
            elseif s isa Rotation
                return _all_k_rotation_DLP_hyperbolic!(Ks,bp,s,tabs;multithreaded)
            else
                error("Unsupported symmetry type: $(typeof(s))")
            end
        catch _
            error("Error computing hyperbolic kernel matrices with symmetry $(symmetry): ")
        end
    end
end

function compute_kernel_matrices_DLP_hyperbolic!(K::Matrix{Complex{T}},bp::BoundaryPointsBIM{T},symmetry::Union{Vector{Any},Nothing},tab::QTaylorTable;multithreaded::Bool=true) where {T<:Real}
    if symmetry===nothing
        return _one_k_nosymm_DLP_hyperbolic!(K,bp,tab;multithreaded)
    else
        try
            s=symmetry[1]
            if s isa Reflection
                return _one_k_reflection_DLP_hyperbolic!(K,bp,s,tab;multithreaded)
            elseif s isa Rotation
                return _one_k_rotation_DLP_hyperbolic!(K,bp,s,tab;multithreaded)
            else
                error("Unsupported symmetry type: $(typeof(s))")
            end
        catch _
            error("Error computing hyperbolic kernel matrices with symmetry $(symmetry): ")
        end
    end
end

################################################################################
# assemble_DLP_hyperbolic!
#
#   Discretizes the hyperbolic double–layer boundary integral operator
#
#       (K μ)(x) = ∫_Γ ∂_{n_y} G_k^ℍ(x,y) μ(y) ds_y
#
#   acting on the Dirichlet density μ represented in the Poincaré disk model.
#
# Inputs
#
#   K  : N×N matrix with entries K[i,j] = ∂_{n_y} G_k^ℍ(x_i,x_j)
#
#   bp : BoundaryPointsBIM containing quadrature weights ds_j
#        (Euclidean arclength elements along Γ)
#
# Discretization
#
#   1) Quadrature:
#
#        ∫_Γ f(y) ds_y  ≈  Σ_j f(y_j) ds_j
#
#      hence each column j is multiplied by ds_j.
#
#   2) Principal value limit (interior Dirichlet problem):
#
#        lim_{x→Γ^-} ∫_Γ ∂_{n_y} G_k^ℍ(x,y) μ(y) ds_y
#          = -½ μ(x) + K μ(x)
#
#      implemented by adding −1/2 to the diagonal.
#
#   3) filter_matrix! removes noise-induced numerical artifacts.
################################################################################
function assemble_DLP_hyperbolic!(K::Matrix{Complex{T}},bp::BoundaryPointsBIM{T}) where {T<:Real}
    @inbounds for j in axes(K,2)
        @views K[:,j].*=bp.ds[j]
    end
    @inbounds for i in axes(K,1)
        K[i,i]+= -0.5
    end
    filter_matrix!(K)
    return nothing
end

#------------------------------------------------------------------------------
# assemble_DLP_hyperbolic!(Ks,bp)::Nothing
#
# INPUTS:
#   Ks::Vector{Matrix{Complex{T}}}   Mk matrices, each N×N (filled in-place)
#   bp::BoundaryPointsBIM{T}         provides bp.ds (Euclidean quadrature weights)
#
# OUTPUTS:
#   nothing
#------------------------------------------------------------------------------
function assemble_DLP_hyperbolic!(Ks::Vector{Matrix{Complex{T}}},bp::BoundaryPointsBIM{T}) where {T<:Real}
    ds=bp.ds
    N=length(ds)
    Mk=length(Ks)
    @inbounds Threads.@threads for m in 1:Mk
        K=Ks[m]
        for j in 1:N
            sj=ds[j]
            @views K[:,j].*=sj
        end
        for i in 1:N
            K[i,i]+=Complex{T}(-T(1.0),zero(T))
        end
        filter_matrix!(K)
    end
    return nothing
end