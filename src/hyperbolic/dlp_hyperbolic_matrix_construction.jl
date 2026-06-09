#################################################################################
# Hyperbolic Helmholtz Boundary Integral Operators on the Poincaré Disk
#
# This file implements source-normal and adjoint double-layer boundary integral
# operators for the hyperbolic Helmholtz equation
#
#     (Δ_H + 1/4 + k²) u = 0,
#
# on bounded domains in the Poincaré disk model of the hyperbolic plane.
#
# ---------------------------------------------------------------------------
# Green's function
# ---------------------------------------------------------------------------
#
# Let
#
#     ν = -1/2 + i k.
#
# The outgoing Green's function is
#
#     G_k(z,z₀)
#       = (1/(2π)) Q_ν(χ(z,z₀)),
#
# where Q_ν is the Legendre function of the second kind and
#
#     χ(z,z₀)
#       = cosh d_H(z,z₀)
#       = 1 + 2|z-z₀|²
#           /((1-|z|²)(1-|z₀|²)).
#
# Here d_H denotes hyperbolic distance in the Poincaré disk.
#
# ---------------------------------------------------------------------------
# Double-layer operator
# ---------------------------------------------------------------------------
#
# The boundary integral operator used by BIM is the source-normal
# double-layer operator
#
#     (D μ)(x)
#       = ∫_Γ ∂_{n_y} G_k(x,y) μ(y) ds_y.
#
# Writing
#
#     y(d) = d/dd Q_ν(cosh d),
#
# the kernel becomes
#
#     ∂_{n_y} G_k(x,y)
#       = (1/(2π))
#         y(d_H(x,y))
#         ∂_{n_y} d_H(x,y).
#
# The corresponding Fredholm operator for the interior Dirichlet problem is
#
#     A = D - 1/2 I.
#
# ---------------------------------------------------------------------------
# Adjoint operator
# ---------------------------------------------------------------------------
#
# For boundary-function extraction and Husimi analysis we also assemble the
# weighted discrete adjoint
#
#     A† = D† - 1/2 I,
#
# where
#
#     D†[i,j]
#       = D[j,i] ds_j/ds_i.
#
# The null vectors of A† are proportional to the physical boundary normal
# derivative ∂ₙψ.
#
# ---------------------------------------------------------------------------
# Geometry conventions
# ---------------------------------------------------------------------------
#
# Everything is evaluated in Euclidean disk coordinates.
#
# The Poincaré metric is
#
#     ds_H² = λ²(dx²+dy²),
#
#     λ(x,y) = 2/(1-r²).
#
# Because the double-layer kernel is conformally invariant,
# Euclidean boundary coordinates, Euclidean unit normals, and Euclidean
# quadrature weights ds are used throughout.
#
# Hyperbolic geometric information enters only through:
#
#   • d_H(x,y)
#   • ∂ₙ d_H(x,y)
#   • the geodesic curvature κ_g
#
# appearing in the principal-value diagonal limit
#
#     K(x,x) = -κ_g(x)/(4π).
#
# M0 9/6/2026
#################################################################################

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
    dn=_∂n_d(xi,yi,xj,yj,nxj,nyj)
    return (y*dn)*inv2π
end

@inline function _hyp_bim_adjoint_entry(tab::QTaylorTable,xj::T,yj::T,xi::T,yi::T,nxi::T,nyi::T,dsj::T) where {T<:Real}
    return hyperbolic_dlp_kernel_scalar_source(tab,xj,yj,xi,yi,nxi,nyi)*dsj
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

"""
    _all_k_nosymm_DLP_hyperbolic!(Ks,bp,tabs;multithreaded=true)
    _one_k_nosymm_DLP_hyperbolic!(K,bp,tab;multithreaded=true)

Assemble the unweighted source-normal hyperbolic double-layer kernel matrices
without symmetry images.

For each spectral parameter `k`, encoded in the corresponding `QTaylorTable`,
the off-diagonal entries are

    K[i,j] = ∂_{n_y}G_k(x_i,x_j)
           = (1/2π) Q'_ν(cosh d_H(x_i,x_j)) ∂_{n_y}d_H(x_i,x_j),

where `ν = -1/2 + i k`. The normal is the Euclidean outward normal at the
source node `x_j`.

The diagonal is replaced by the principal-value source-normal limit

    K[i,i] = -κ_g(x_i)/(4π),

with the hyperbolic geodesic curvature computed from the Euclidean curvature by

    κ_g = (1/λ)(κ_E + ∂_n log λ).

These routines build only the raw DLP kernel. Quadrature weights and the
interior jump term `-1/2 I` are applied later by `assemble_DLP_hyperbolic!`.

`_all_k_...` fills one matrix per table. `_one_k_...` is the single-table
variant.
"""
function _all_k_nosymm_DLP_hyperbolic!(Ks::Vector{Matrix{Complex{T}}},bp::BoundaryPoints{T},tabs::Vector{QTaylorTable};multithreaded::Bool=true) where {T<:Real}
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

function _one_k_nosymm_DLP_hyperbolic!(K::Matrix{Complex{T}},bp::BoundaryPoints{T},tab::QTaylorTable;multithreaded::Bool=true) where {T<:Real}
    _all_k_nosymm_DLP_hyperbolic!([K],bp,[tab];multithreaded=multithreaded)
    return K
end

"""
    _all_k_nosymm_adjoint_DLP_hyperbolic!(As,bp,tabs;multithreaded=true)
    _one_k_nosymm_adjoint_DLP_hyperbolic!(A,bp,tab;multithreaded=true)

Assemble the weighted adjoint Fredholm matrices without symmetry images.

The source-normal DLP Nyström matrix has entries

    D[i,j] = ∂_{n_y}G_k(x_i,x_j) ds_j.

The weighted discrete adjoint is

    D†[i,j] = D[j,i] ds_j/ds_i,

which is equivalent here to evaluating the source-normal kernel with the roles
of target and source interchanged and multiplying by `ds_j`.

This routine directly fills the adjoint Fredholm matrix

    A = D† - 1/2 I,

using the same diagonal principal-value limit as the source matrix, weighted by
`ds_j`, and then applying the interior jump term. The resulting null vector is
the boundary normal derivative density used for Husimi plots, localization
diagnostics, and SLP reconstruction.

`_all_k_...` fills one matrix per table. `_one_k_...` calls the all-version on a
one-element vector.
"""
function _all_k_nosymm_adjoint_DLP_hyperbolic!(As::Vector{Matrix{Complex{T}}},bp::BoundaryPoints{T},tabs::Vector{QTaylorTable};multithreaded::Bool=true) where {T<:Real}
    Mk=length(tabs)
    N=length(bp.xy)
    @use_threads multithreading=multithreaded for m in 1:Mk
        A=As[m]
        tab=tabs[m]
        fill!(A,zero(Complex{T}))
        @inbounds for i in 1:N
            xi,yi=bp.xy[i]
            nxi,nyi=bp.normal[i]
            κi=bp.curvature[i]
            for j in 1:N
                dsj=bp.ds[j]
                if i==j
                    A[i,i]=Complex{T}(dlp_diag_source_normal_poincare(xi,yi,nxi,nyi,κi)*dsj-T(0.5),zero(T))
                else
                    xj,yj=bp.xy[j]
                    A[i,j]=_hyp_bim_adjoint_entry(tab,xj,yj,xi,yi,nxi,nyi,dsj)
                end
            end
        end
        filter_matrix!(A)
    end
    return nothing
end

function _one_k_nosymm_adjoint_DLP_hyperbolic!(A::Matrix{Complex{T}},bp::BoundaryPoints{T},tab::QTaylorTable;multithreaded::Bool=true) where {T<:Real}
    _all_k_nosymm_adjoint_DLP_hyperbolic!([A],bp,[tab];multithreaded=multithreaded)
    return A
end

"""
    _all_k_reflection_DLP_hyperbolic!(Ks,bp,sym,tabs;multithreaded=true)
    _one_k_reflection_DLP_hyperbolic!(K,bp,sym,tab;multithreaded=true)

Assemble source-normal hyperbolic DLP kernel matrices with reflection symmetry
images.

The physical boundary contribution is first assembled as in the no-symmetry
case. Reflected source copies are then added with the parity factors determined
by `sym`. For each reflected source image `R x_j`, the added off-diagonal term is

    χ_R ∂_{n_{Rj}}G_k(x_i,Rx_j),

where both the source point and its Euclidean normal are reflected. Image
self-coincidences are skipped using an `eps(T)^2` distance cutoff.

The matrices remain unweighted kernel matrices. Column quadrature weights and
the interior jump term are applied later by `assemble_DLP_hyperbolic!`.
"""
function _all_k_reflection_DLP_hyperbolic!(Ks::Vector{Matrix{Complex{T}}},bp::BoundaryPoints{T},sym::Reflection,tabs::Vector{QTaylorTable};multithreaded::Bool=true) where {T<:Real}
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

function _one_k_reflection_DLP_hyperbolic!(K::Matrix{Complex{T}},bp::BoundaryPoints{T},sym::Reflection,tab::QTaylorTable;multithreaded::Bool=true) where{T<:Real}
    _one_k_nosymm_DLP_hyperbolic!(K,bp,tab;multithreaded)
    N=length(bp.xy)
    tol2=(eps(T))^2
    shift_x=bp.shift_x;shift_y=bp.shift_y
    ops=_reflect_ops_and_scales(T,sym)
    pt_tls=[zeros(T,2) for _ in 1:Threads.nthreads()]
    @use_threads multithreading=multithreaded for i in 1:N
        xi,yi=bp.xy[i]
        pt=pt_tls[Threads.threadid()]
        @inbounds for j in 1:N
            xj0,yj0=bp.xy[j]
            nx0,ny0=bp.normal[j]
            @inbounds for (op,scale_r) in ops
                if op==1
                    x_reflect_point!(pt,xj0,yj0,shift_x);nxr=-nx0;nyr=ny0
                elseif op==2
                    y_reflect_point!(pt,xj0,yj0,shift_y);nxr=nx0;nyr=-ny0
                else
                    xy_reflect_point!(pt,xj0,yj0,shift_x,shift_y);nxr=-nx0;nyr=-ny0
                end
                xjr,yjr=pt[1],pt[2]
                dx=xi-xjr;dy=yi-yjr
                d2=muladd(dx,dx,dy*dy)
                if d2>tol2
                    sc=Complex{T}(scale_r,zero(T))
                    K[i,j]+=hyperbolic_dlp_kernel_scalar_source(tab,xi,yi,xjr,yjr,nxr,nyr)*sc
                end
            end
        end
    end
    return nothing
end

"""
    _all_k_reflection_adjoint_DLP_hyperbolic!(As,bp,sym,tabs;multithreaded=true)
    _one_k_reflection_adjoint_DLP_hyperbolic!(A,bp,sym,tab;multithreaded=true)

Assemble weighted adjoint Fredholm matrices with reflection symmetry images.

The no-symmetry adjoint matrix is assembled first. Reflection images are then
added by reflecting the adjoint-side source point and normal, with the parity
factor from `sym`. For each reflected target-side image in the adjoint
construction, the added term has the form

    χ_R ∂_{n_{Ri}}G_k(x_j,Rx_i) ds_j.

The matrices are already weighted Fredholm matrices, including the jump term.
No additional call to `assemble_DLP_hyperbolic!` should be made.
"""
function _all_k_reflection_adjoint_DLP_hyperbolic!(As::Vector{Matrix{Complex{T}}},bp::BoundaryPoints{T},sym::Reflection,tabs::Vector{QTaylorTable};multithreaded::Bool=true) where {T<:Real}
    _all_k_nosymm_adjoint_DLP_hyperbolic!(As,bp,tabs;multithreaded=multithreaded)
    Mk=length(tabs)
    N=length(bp.xy)
    tol2=eps(T)^2
    shift_x=bp.shift_x
    shift_y=bp.shift_y
    ops=_reflect_ops_and_scales(T,sym)
    pt_tls=[zeros(T,2) for _ in 1:Threads.nthreads()]
    @use_threads multithreading=multithreaded for i in 1:N
        xi,yi=bp.xy[i]
        nxi,nyi=bp.normal[i]
        pt=pt_tls[Threads.threadid()]
        @inbounds for (op,scale_r) in ops
            if op==1
                x_reflect_point!(pt,xi,yi,shift_x);nxr=-nxi;nyr=nyi
            elseif op==2
                y_reflect_point!(pt,xi,yi,shift_y);nxr=nxi;nyr=-nyi
            else
                xy_reflect_point!(pt,xi,yi,shift_x,shift_y);nxr=-nxi;nyr=-nyi
            end
            xir,yir=pt[1],pt[2]
            sc=Complex{T}(scale_r,zero(T))
            for j in 1:N
                xj,yj=bp.xy[j]
                dx=xj-xir
                dy=yj-yir
                muladd(dx,dx,dy*dy)<=tol2 && continue
                dsj=bp.ds[j]
                for m in 1:Mk
                    As[m][i,j]+=sc*_hyp_bim_adjoint_entry(tabs[m],xj,yj,xir,yir,nxr,nyr,dsj)
                end
            end
        end
    end
    @inbounds for A in As
        filter_matrix!(A)
    end
    return nothing
end

function _one_k_reflection_adjoint_DLP_hyperbolic!(A::Matrix{Complex{T}},bp::BoundaryPoints{T},sym::Reflection,tab::QTaylorTable;multithreaded::Bool=true) where {T<:Real}
    _all_k_reflection_adjoint_DLP_hyperbolic!([A],bp,sym,[tab];multithreaded=multithreaded)
    return A
end

"""
    _all_k_rotation_DLP_hyperbolic!(Ks,bp,sym,tabs;multithreaded=true)
    _one_k_rotation_DLP_hyperbolic!(K,bp,sym,tab;multithreaded=true)

Assemble source-normal hyperbolic DLP kernel matrices with rotation symmetry
images.

The physical boundary contribution is assembled first. For an `n`-fold rotation
sector `m`, the missing source copies

    x_j -> R^l x_j,    l = 1,...,n-1,

are added with character factors

    χ_l = exp(2π i m l/n).

Each image contribution uses the rotated Euclidean source normal, so the added
kernel is

    χ_l ∂_{n_{R^l j}}G_k(x_i,R^l x_j).

The result is an unweighted kernel matrix. The Nyström weights and interior
jump contribution are added by `assemble_DLP_hyperbolic!`.
"""
function _all_k_rotation_DLP_hyperbolic!(Ks::Vector{Matrix{Complex{T}}},bp::BoundaryPoints{T},sym::Rotation,tabs::Vector{QTaylorTable};multithreaded::Bool=true) where {T<:Real}
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

function _one_k_rotation_DLP_hyperbolic!(K::AbstractMatrix{Complex{T}},bp::BoundaryPoints{T},sym::Rotation,tab::QTaylorTable;multithreaded::Bool=true) where{T<:Real}
    _one_k_nosymm_DLP_hyperbolic!(K,bp,tab;multithreaded)
    N=length(bp.xy)
    tol2=(eps(T))^2
    cx,cy=sym.center
    ctab,stab,χ=_rotation_tables(T,sym.n,mod(sym.m,sym.n))
    pt_tls=[zeros(T,2) for _ in 1:Threads.nthreads()]
    @use_threads multithreading=multithreaded for i in 1:N
        xi,yi=bp.xy[i]
        pt=pt_tls[Threads.threadid()]
        @inbounds for j in 1:N
            xj0,yj0=bp.xy[j]
            nx0,ny0=bp.normal[j]
            @inbounds for l in 2:sym.n
                rot_point!(pt,xj0,yj0,cx,cy,ctab[l],stab[l])
                xjr,yjr=pt[1],pt[2]
                nxr=ctab[l]*nx0-stab[l]*ny0
                nyr=stab[l]*nx0+ctab[l]*ny0
                dx=xi-xjr;dy=yi-yjr
                d2=muladd(dx,dx,dy*dy)
                if d2>tol2
                    K[i,j]+=hyperbolic_dlp_kernel_scalar_source(tab,xi,yi,xjr,yjr,nxr,nyr)*χ[l]
                end
            end
        end
    end
    return nothing
end

"""
    _all_k_rotation_adjoint_DLP_hyperbolic!(As,bp,sym,tabs;multithreaded=true)
    _one_k_rotation_adjoint_DLP_hyperbolic!(A,bp,sym,tab;multithreaded=true)

Assemble weighted adjoint Fredholm matrices with rotation symmetry images.

The no-symmetry adjoint matrix is assembled first. For an `n`-fold rotation
sector `m`, rotated copies of the adjoint-side point are added with characters

    χ_l = exp(2π i m l/n),    l = 1,...,n-1.

Each image contribution uses the rotated Euclidean normal and has the form

    χ_l ∂_{n_{R^l i}}G_k(x_j,R^l x_i) ds_j.

The matrices are final adjoint Fredholm matrices, including quadrature weights
and the interior jump term.
"""
function _all_k_rotation_adjoint_DLP_hyperbolic!(As::Vector{Matrix{Complex{T}}},bp::BoundaryPoints{T},sym::Rotation,tabs::Vector{QTaylorTable};multithreaded::Bool=true) where {T<:Real}
    _all_k_nosymm_adjoint_DLP_hyperbolic!(As,bp,tabs;multithreaded=multithreaded)
    Mk=length(tabs)
    N=length(bp.xy)
    tol2=eps(T)^2
    cx,cy=sym.center
    ctab,stab,χ=_rotation_tables(T,sym.n,mod(sym.m,sym.n))
    pt_tls=[zeros(T,2) for _ in 1:Threads.nthreads()]
    @use_threads multithreading=multithreaded for i in 1:N
        xi,yi=bp.xy[i]
        nxi,nyi=bp.normal[i]
        pt=pt_tls[Threads.threadid()]
        @inbounds for l in 2:sym.n
            rot_point!(pt,xi,yi,cx,cy,ctab[l],stab[l])
            xir,yir=pt[1],pt[2]
            nxr=ctab[l]*nxi-stab[l]*nyi
            nyr=stab[l]*nxi+ctab[l]*nyi
            phase=Complex{T}(χ[l])
            for j in 1:N
                xj,yj=bp.xy[j]
                dx=xj-xir
                dy=yj-yir
                muladd(dx,dx,dy*dy)<=tol2 && continue
                dsj=bp.ds[j]
                for m in 1:Mk
                    As[m][i,j]+=phase*_hyp_bim_adjoint_entry(tabs[m],xj,yj,xir,yir,nxr,nyr,dsj)
                end
            end
        end
    end
    @inbounds for A in As
        filter_matrix!(A)
    end
    return nothing
end

function _one_k_rotation_adjoint_DLP_hyperbolic!(A::Matrix{Complex{T}},bp::BoundaryPoints{T},sym::Rotation,tab::QTaylorTable;multithreaded::Bool=true) where {T<:Real}
    _all_k_rotation_adjoint_DLP_hyperbolic!([A],bp,sym,[tab];multithreaded=multithreaded)
    return A
end

################################################################################

"""
    compute_kernel_matrices_DLP_hyperbolic!(solver,Ks,bp,tabs;multithreaded=true)
    compute_kernel_matrices_DLP_hyperbolic!(solver,K,bp,tab;multithreaded=true)

Public assembly wrapper for the source-normal hyperbolic DLP Fredholm matrix.

Depending on `solver.symmetry`, this dispatches to the no-symmetry, reflection,
or rotation source-kernel builder. It then applies Nyström column weights and
the interior Dirichlet jump term via

    A = D - 1/2 I,

where

    D[i,j] = ∂_{n_y}G_k(x_i,x_j) ds_j.

The one-matrix method is a compact wrapper around the vector method with
one-element vectors.
"""
function compute_kernel_matrices_DLP_hyperbolic!(Ks::Vector{Matrix{Complex{T}}},bp::BoundaryPoints{T},tabs::Vector{QTaylorTable};multithreaded::Bool=true) where {T<:Real}
    return _all_k_nosymm_DLP_hyperbolic!(Ks,bp,tabs;multithreaded)
end

function compute_kernel_matrices_DLP_hyperbolic!(K::Matrix{Complex{T}},bp::BoundaryPoints{T},tab::QTaylorTable;multithreaded::Bool=true) where {T<:Real}
    return _one_k_nosymm_DLP_hyperbolic!(K,bp,tab;multithreaded)
end

function compute_kernel_matrices_DLP_hyperbolic!(Ks::Vector{Matrix{Complex{T}}},bp::BoundaryPoints{T},sym::Reflection,ks::Vector{QTaylorTable};multithreaded::Bool=true) where {T<:Real}
    return _all_k_reflection_DLP_hyperbolic!(Ks,bp,sym,ks;multithreaded)
end

function compute_kernel_matrices_DLP_hyperbolic!(K::Matrix{Complex{T}},bp::BoundaryPoints{T},sym::Reflection,tab::QTaylorTable;multithreaded::Bool=true) where {T<:Real}
    return _one_k_reflection_DLP_hyperbolic!(K,bp,sym,tab;multithreaded)
end

function compute_kernel_matrices_DLP_hyperbolic!(Ks::Vector{Matrix{Complex{T}}},bp::BoundaryPoints{T},sym::Rotation,ks::Vector{QTaylorTable};multithreaded::Bool=true) where {T<:Real}
    return _all_k_rotation_DLP_hyperbolic!(Ks,bp,sym,ks;multithreaded)
end

function compute_kernel_matrices_DLP_hyperbolic!(K::Matrix{Complex{T}},bp::BoundaryPoints{T},sym::Rotation,tab::QTaylorTable;multithreaded::Bool=true) where {T<:Real}
    return _one_k_rotation_DLP_hyperbolic!(K,bp,sym,tab;multithreaded)
end

################################################################################

"""
    assemble_DLP_hyperbolic!(K,bp)
    assemble_DLP_hyperbolic!(Ks,bp)

Convert raw source-normal DLP kernel matrices into source Fredholm matrices.

The raw source builders fill

    K[i,j] = ∂_{n_y}G_k(x_i,x_j),

with the principal-value diagonal already inserted. This routine applies the
Nyström column weights and the interior Dirichlet jump term,

    A[i,j] = K[i,j] ds_j,
    A[i,i] = A[i,i] - 1/2.

Thus the resulting matrix is

    A = D - 1/2 I,

where

    D[i,j] = ∂_{n_y}G_k(x_i,x_j) ds_j.

The vector version applies this operation to every matrix in `Ks`.
"""
function assemble_DLP_hyperbolic!(K::Matrix{Complex{T}},bp::BoundaryPoints{T}) where {T<:Real}
    @inbounds for j in axes(K,2)
        @views K[:,j].*=bp.ds[j]
    end
    @inbounds for i in axes(K,1)
        K[i,i]+=Complex{T}(-T(0.5),zero(T))
    end
    filter_matrix!(K)
    return nothing
end

function assemble_DLP_hyperbolic!(Ks::Vector{Matrix{Complex{T}}},bp::BoundaryPoints{T}) where {T<:Real}
    @inbounds Threads.@threads for m in eachindex(Ks)
        assemble_DLP_hyperbolic!(Ks[m],bp)
    end
    return nothing
end

"""
    compute_kernel_matrices_DLP_hyperbolic!(solver,Ks,bp,tabs;multithreaded=true)
    compute_kernel_matrices_DLP_hyperbolic!(solver,K,bp,tab;multithreaded=true)

Assemble source-normal hyperbolic DLP Fredholm matrices.

Depending on `solver.symmetry`, this dispatches to the no-symmetry, reflection,
or rotation raw source-kernel builder. It then applies Euclidean quadrature
weights and the interior jump term, producing

    A = D - 1/2 I,

with

    D[i,j] = ∂_{n_y}G_k(x_i,x_j) ds_j.

This is the matrix used for eigenvalue searches and Beyn contour solves.
"""
function compute_kernel_matrices_DLP_hyperbolic!(solver::BIM_hyperbolic,Ks::Vector{Matrix{Complex{T}}},bp::BoundaryPoints{T},tabs::Vector{QTaylorTable};multithreaded::Bool=true) where {T<:Real}
    s=solver.symmetry
    if isnothing(s)
        _all_k_nosymm_DLP_hyperbolic!(Ks,bp,tabs;multithreaded=multithreaded)
    elseif s isa Reflection
        _all_k_reflection_DLP_hyperbolic!(Ks,bp,s,tabs;multithreaded=multithreaded)
    elseif s isa Rotation
        _all_k_rotation_DLP_hyperbolic!(Ks,bp,s,tabs;multithreaded=multithreaded)
    else
        error("Unsupported symmetry type: $(typeof(s))")
    end
    assemble_DLP_hyperbolic!(Ks,bp)
    return nothing
end

function compute_kernel_matrices_DLP_hyperbolic!(solver::BIM_hyperbolic,K::Matrix{Complex{T}},bp::BoundaryPoints{T},tab::QTaylorTable;multithreaded::Bool=true) where {T<:Real}
    compute_kernel_matrices_DLP_hyperbolic!(solver,[K],bp,[tab];multithreaded=multithreaded)
    return nothing
end

"""
    compute_adjoint_kernel_matrices_DLP_hyperbolic!(solver,As,bp,tabs;multithreaded=true)
    compute_adjoint_kernel_matrices_DLP_hyperbolic!(solver,A,bp,tab;multithreaded=true)

Assemble weighted adjoint hyperbolic DLP Fredholm matrices.

The source Nyström matrix is

    D[i,j] = ∂_{n_y}G_k(x_i,x_j) ds_j.

The weighted discrete adjoint is

    D†[i,j] = D[j,i] ds_j/ds_i.

The adjoint builders construct the Fredholm matrix directly as

    A = D† - 1/2 I.

These matrices are intended for boundary normal-derivative extraction; their
null vectors are proportional to `∂ₙψ` and are the correct objects for Husimi,
IPR/entropy, and SLP reconstruction.
"""
function compute_adjoint_kernel_matrices_DLP_hyperbolic!(solver::BIM_hyperbolic,As::Vector{Matrix{Complex{T}}},bp::BoundaryPoints{T},tabs::Vector{QTaylorTable};multithreaded::Bool=true) where {T<:Real}
    s=solver.symmetry
    if isnothing(s)
        _all_k_nosymm_adjoint_DLP_hyperbolic!(As,bp,tabs;multithreaded=multithreaded)
    elseif s isa Reflection
        _all_k_reflection_adjoint_DLP_hyperbolic!(As,bp,s,tabs;multithreaded=multithreaded)
    elseif s isa Rotation
        _all_k_rotation_adjoint_DLP_hyperbolic!(As,bp,s,tabs;multithreaded=multithreaded)
    else
        error("Unsupported symmetry type: $(typeof(s))")
    end
    return nothing
end

function compute_adjoint_kernel_matrices_DLP_hyperbolic!(solver::BIM_hyperbolic,A::Matrix{Complex{T}},bp::BoundaryPoints{T},tab::QTaylorTable;multithreaded::Bool=true) where {T<:Real}
    compute_adjoint_kernel_matrices_DLP_hyperbolic!(solver,[A],bp,[tab];multithreaded=multithreaded)
    return nothing
end

function construct_matrices!(solver::BIM_hyperbolic,A::Matrix{Complex{T}},pts::BoundaryPointsHyp{T},tab::QTaylorTable;multithreaded::Bool=true,adjoint_mode::Symbol=:source) where {T<:Real}
    bp=_BoundaryPointsHypBIM_to_BoundaryPoints(pts)
    if adjoint_mode===:source
        compute_kernel_matrices_DLP_hyperbolic!(solver,A,bp,tab;multithreaded=multithreaded)
    elseif adjoint_mode===:direct || adjoint_mode===:via_D
        compute_adjoint_kernel_matrices_DLP_hyperbolic!(solver,A,bp,tab;multithreaded=multithreaded)
    else
        error("Invalid adjoint_mode: $adjoint_mode. Expected :source, :direct, or :via_D.")
    end
    return nothing
end

"""
    construct_boundary_matrices!(Tbufs,solver::BIM_hyperbolic,pts,zj;multithreaded=true,timeit=false,adjoint_mode=:source)

Construct hyperbolic BIM boundary matrices for all complex contour nodes `zj`.

The hyperbolic Legendre-Q Taylor tables are built once per contour node over the
distance range of `pts`. The matrix type is selected by `adjoint_mode`:

- `:source` assembles the source-normal Fredholm matrix

      A(z) = D(z) - 1/2 I,

  used for eigenvalue searches and Beyn contour solves.

- `:direct` and `:via_D` assemble the weighted adjoint Fredholm matrix

      A†(z) = D†(z) - 1/2 I,

  used for boundary normal-derivative vectors.
"""
function construct_boundary_matrices!(Tbufs::Vector{Matrix{ComplexF64}},solver::BIM_hyperbolic,pts::BoundaryPointsHyp{T},zj::AbstractVector{ComplexF64};multithreaded::Bool=true,timeit::Bool=false,adjoint_mode::Symbol=:source) where {T<:Real}
    bp=_BoundaryPointsHypBIM_to_BoundaryPoints(pts)
    N=length(bp.xy)
    @inbounds for q in eachindex(Tbufs)
        @assert size(Tbufs[q])==(N,N) "Tbufs[$q] has size $(size(Tbufs[q])), but BIM_hyperbolic requires ($N,$N)."
        fill!(Tbufs[q],0.0+0.0im)
    end
    _,dmax=d_bounds_hyp(pts,solver.symmetry;dmin_floor=T(1e-15),pad_max=T(1.1))
    tabs=Vector{QTaylorTable}(undef,length(zj))
    @inbounds for q in eachindex(zj)
        tabs[q]=build_QTaylorTable(zj[q];dmin=legendre_d_threshold(),dmax=Float64(dmax)*1.05)
    end
    if adjoint_mode===:source
        @benchit timeit=timeit "BIM_hyperbolic SourceAssembly" compute_kernel_matrices_DLP_hyperbolic!(solver,Tbufs,bp,tabs;multithreaded=multithreaded)
    elseif adjoint_mode===:direct || adjoint_mode===:via_D
        @benchit timeit=timeit "BIM_hyperbolic AdjointAssembly" compute_adjoint_kernel_matrices_DLP_hyperbolic!(solver,Tbufs,bp,tabs;multithreaded=multithreaded)
    else
        error("Invalid adjoint_mode: $adjoint_mode. Expected :source, :direct, or :via_D.")
    end
    return nothing
end