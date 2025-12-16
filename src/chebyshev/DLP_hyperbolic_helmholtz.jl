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
#    χ(z,z0) = cosh d_H(z,z0)
#             = 1 + 2|z - z0|^2 / ((1 - |z|^2) (1 - |z0|^2)).
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
# ----
# hyperbolic_d_poincare(xi, yi, xj, yj) -> χ
# hyperbolic_dchi_dn_target(xi, yi, nxi, nyi, xj, yj) -> ∂χ/∂n_E(z_i)
#
# hyperbolic_Qprime(nu, χ) -> ∂Q_ν/∂χ (user must implement or overload)
#
# _one_k_nosymm_DLP_hyperbolic!  - build one DLP matrix for a single complex k
# _all_k_nosymm_DLP_hyperbolic!  - build DLP matrices for a vector of ks
#
# compute_kernel_matrices_DLP_hyperbolic! - public entry points (single / multiple k)
#
# assemble_fredholm_matrices_hyperbolic! - apply weights and add identity (as in Euclidean BIM)
#
# NOTE:
#   This file can build matrices either via the legacy χ/Q' hook (target normal), or via fast Taylor tables for y(d) (preferred).
#   Both χ-based helpers and new distance/source-normal helpers are present.
#
# M0/25/10/2025
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
@inline function hyperbolic_cosh_d_poincare(xi::T,yi::T,xj::T,yj::T) where {T<:Real}
    dx=xi-xj
    dy=yi-yj
    Δ2=muladd(dx,dx,dy*dy)
    r2=muladd(xi,xi,yi*yi)
    r02=muladd(xj,xj,yj*yj)
    denom=(one(T)-r2)*(one(T)-r02)
    return one(T)+2*Δ2/denom
end

@inline function hyperbolic_distance_poincare(xi::T,yi::T,xj::T,yj::T) where {T<:Real}
    χ=hyperbolic_cosh_d_poincare(xi,yi,xj,yj)
    return acosh(χ)
end

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

@inline function hyperbolic_dlp_kernel_scalar(tab::QTaylorTable,xi::T,yi::T,xj::T,yj::T,nxi::T,nyi::T) where{T<:Real}
    d=Float64(hyperbolic_distance_poincare(xi,yi,xj,yj))
    y=_eval_dQdd(tab,d)
    dn=hyperbolic_dn_d_target(xi,yi,xj,yj,nxi,nyi)
    return (y*dn)/TWO_PI
end

@inline function hyperbolic_slp_kernel_scalar(tab::QTaylorTable,xi::T,yi::T,xj::T,yj::T) where {T<:Real}
    d=Float64(hyperbolic_distance_poincare(xi,yi,xj,yj))
    return eval_Q(tab, d)/TWO_PI
end

function _all_k_nosymm_DLP_hyperbolic!(Ks::Vector{Matrix{Complex{T}}},bp::BoundaryPointsBIM{T},tabs::Vector{QTaylorTable};multithreaded::Bool=true) where {T<:Real}
    Mk=length(tabs)
    N=length(bp.xy)
    tol2=(eps(T))^2
    @use_threads multithreading=multithreaded for m in 1:Mk
        K=Ks[m]
        tab=tabs[m]
        @assert size(K,1)==N && size(K,2)==N
        fill!(K,zero(eltype(K)))
        @inbounds for i in 1:N
            xi,yi=bp.xy[i]
            nxi,nyi=bp.normal[i]
            κi=bp.curvature[i]
            for j in 1:N
                xj,yj=bp.xy[j]
                dx=xi-xj
                dy=yi-yj
                d2=muladd(dx,dx,dy*dy)
                if d2≤tol2 && i==j
                    K[i,i]=Complex{T}(κi/(TWO_PI),zero(T))
                else
                    K[i,j]=hyperbolic_dlp_kernel_scalar(tab,xi,yi,xj,yj,nxi,nyi)
                end
            end
        end
    end
    return nothing
end

function _one_k_nosymm_DLP_hyperbolic!(K::Matrix{Complex{T}},bp::BoundaryPointsBIM{T},tab::QTaylorTable;multithreaded::Bool=true) where {T<:Real}
    N=length(bp.xy)
    fill!(K,zero(eltype(K)))
    tol2=(eps(T))^2
    @use_threads multithreading=multithreaded for i in 1:N
        xi,yi=bp.xy[i]
        nxi,nyi=bp.normal[i]
        κi=bp.curvature[i]
        @inbounds for j in 1:N
            xj,yj=bp.xy[j]
            dx=xi-xj
            dy=yi-yj
            d2=muladd(dx,dx,dy*dy)
            if d2≤tol2 && i==j
                K[i,i]=Complex{T}(κi/(TWO_PI),zero(T))
            else
                K[i,j]=hyperbolic_dlp_kernel_scalar(tab,xi,yi,xj,yj,nxi,nyi)
            end
        end
    end
    return nothing
end

function _all_k_reflection_DLP_hyperbolic!(Ks::Vector{Matrix{Complex{T}}},bp::BoundaryPointsBIM{T},sym::Reflection,tabs::Vector{QTaylorTable};multithreaded::Bool=true) where {T<:Real}
    _all_k_nosymm_DLP_hyperbolic!(Ks,bp,tabs;multithreaded)   # direct only
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
        nxi,nyi=bp.normal[i]
        tid=Threads.threadid()
        pt=pt_tls[tid]
        @inbounds for j in 1:N
            xj0,yj0=bp.xy[j]
            @inbounds for (op,scale_r) in ops
                if op==1
                    x_reflect_point!(pt,xj0,yj0,shift_x)
                elseif op==2
                    y_reflect_point!(pt,xj0,yj0,shift_y)
                else
                    xy_reflect_point!(pt,xj0,yj0,shift_x,shift_y)
                end
                xjr,yjr=pt[1],pt[2]
                dx=xi-xjr
                dy=yi-yjr
                d2=muladd(dx,dx,dy*dy)
                if d2>tol2
                    sc=Complex{T}(scale_r,zero(T))
                    @inbounds for m in 1:Mk
                        tab=tabs[m]
                        val=hyperbolic_dlp_kernel_scalar(tab,xi,yi,pt[1],pt[2],nxi,nyi)*sc
                        Ks[m][i,j]+=val
                    end
                end
            end
        end
    end
    return nothing
end

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
        nxi,nyi=bp.normal[i]
        @inbounds for j in 1:N
            xj0,yj0=bp.xy[j]
            @inbounds for (op,scale_r) in ops
                if op==1
                    x_reflect_point!(pt,xj0,yj0,shift_x)
                elseif op==2
                    y_reflect_point!(pt,xj0,yj0,shift_y)
                else
                    xy_reflect_point!(pt,xj0,yj0,shift_x,shift_y)
                end
                xjr,yjr=pt[1],pt[2]
                dx=xi-xjr
                dy=yi-yjr
                d2=muladd(dx,dx,dy*dy)
                if d2>tol2
                    sc=Complex{T}(scale_r,zero(T))
                    val=hyperbolic_dlp_kernel_scalar(tab,xi,yi,pt[1],pt[2],nxi,nyi)*sc
                    K[i,j]+=val
                end
            end
        end
    end
    return K
end

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
        nxi,nyi=bp.normal[i]
        tid=Threads.threadid()
        pt=pt_tls[tid]
        @inbounds for j in 1:N
            xj0,yj0=bp.xy[j]
            @inbounds for l in 2:sym.n
                rot_point!(pt,xj0,yj0,cx,cy,ctab[l],stab[l])
                xjr,yjr=pt[1],pt[2]
                dx=xi-xjr
                dy=yi-yjr
                d2=muladd(dx,dx,dy*dy)
                if d2>tol2
                    phase=χ[l]
                    @inbounds for m in 1:Mk
                        tab=tabs[m]
                        val=hyperbolic_dlp_kernel_scalar(tab,xi,yi,pt[1],pt[2],nxi,nyi)*phase
                        Ks[m][i,j]+=val
                    end
                end
            end
        end
    end
    return nothing
end

function _one_k_rotation_DLP_hyperbolic!(K::AbstractMatrix{Complex{T}},bp::BoundaryPointsBIM{T},sym::Rotation,tab::QTaylorTable;multithreaded::Bool=true) where {T<:Real}
    _one_k_nosymm_DLP_hyperbolic!(K,bp,tab;multithreaded)
    N=length(bp.xy)
    tol2=(eps(T))^2
    cx,cy=sym.center
    ctab,stab,χ=_rotation_tables(T,sym.n,mod(sym.m,sym.n))
    pt=[zero(T),zero(T)]
    @use_threads multithreading=multithreaded for i in 1:N
        xi,yi=bp.xy[i]
        nxi,nyi=bp.normal[i]
        @inbounds for j in 1:N
            xj0,yj0=bp.xy[j]
            @inbounds for l in 2:sym.n
                rot_point!(pt,xj0,yj0,cx,cy,ctab[l],stab[l])
                xjr,yjr=pt[1],pt[2]
                dx=xi-xjr
                dy=yi-yjr
                d2=muladd(dx,dx,dy*dy)
                if d2>tol2
                    phase=χ[l]
                    val=hyperbolic_dlp_kernel_scalar(tab,xi,yi,pt[1],pt[2],nxi,nyi)*phase
                    K[i,j]+=val
                end
            end
        end
    end
    return K
end

function compute_kernel_matrices_DLP_hyperbolic!(Ks::Vector{Matrix{Complex{T}}},bp::BoundaryPointsBIM{T},symmetry::Union{Vector{Any},Nothing},tabs::Vector{QTaylorTable};multithreaded::Bool=true,kernel_fun::Union{Symbol,Function}=:default) where {T<:Real}
    if symmetry===nothing
        return compute_kernel_matrices_DLP_hyperbolic!(Ks,bp,tabs;multithreaded,kernel_fun)
    else
        try
            compute_kernel_matrices_DLP_hyperbolic!(Ks,bp,symmetry[1],tabs;multithreaded,kernel_fun)
        catch _
            error("Error computing hyperbolic kernel matrices with symmetry $(symmetry): ")
        end
    end
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

function compute_kernel_matrices_DLP_hyperbolic!(Ks::Vector{Matrix{Complex{T}}},bp::BoundaryPointsBIM{T},symmetry::Union{Vector{Any},Nothing},tabs::Vector{QTaylorTable};multithreaded::Bool=true) where {T<:Real}
    if symmetry===nothing
        return _all_k_nosymm_DLP_hyperbolic!(Ks,bp,tabs;multithreaded)
    else
        try
            s = symmetry[1]
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

function compute_kernel_matrices_DLP_hyperbolic!(K::Matrix{Complex{T}},bp::BoundaryPointsBIM{T},tab::QTaylorTable;multithreaded::Bool=true) where {T<:Real}
    return _one_k_nosymm_DLP_hyperbolic!(K,bp,tab;multithreaded)
end

function compute_kernel_matrices_DLP_hyperbolic!(K::Matrix{Complex{T}},bp::BoundaryPointsBIM{T},sym::Reflection,tab::QTaylorTable;multithreaded::Bool=true) where {T<:Real}
    return _one_k_reflection_DLP_hyperbolic!(K,bp,sym,tab;multithreaded)
end

function compute_kernel_matrices_DLP_hyperbolic!(K::Matrix{Complex{T}},bp::BoundaryPointsBIM{T},sym::Rotation,tab::QTaylorTable;multithreaded::Bool=true) where {T<:Real}
    return _one_k_rotation_DLP_hyperbolic!(K,bp,sym,tab;multithreaded)
end