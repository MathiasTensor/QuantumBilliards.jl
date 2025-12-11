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
# The DLP kernel we discretize is
#
#    K_k(z_i, z_j) = ∂/∂n_E(z_i) G_k(z_i, z_j)
#                  = (1 / (2π)) * Q_ν'(χ) * ∂χ/∂n_E(z_i),
#
# where n_E is the Euclidean unit normal at the target point z_i.
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
# hyperbolic_chi_poincare(xi, yi, xj, yj) -> χ
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
#   This file DOES NOT use Chebyshev–Hankel tables. The heavy part is
#   the evaluation of Q_ν'(χ), which is factored via hyperbolic_Qprime.
#
# M0/25/10/2025
#################################################################################

#################################################################################
# HYPERBOLIC GEOMETRY: χ(z,z0), ∂χ/∂(x_i,y_i), ∂χ/∂n_E
#################################################################################

@inline function hyperbolic_chi_poincare(xi::T,yi::T,xj::T,yj::T) where {T<:Real}
    dx=xi-xj
    dy=yi-yj
    Δ2=muladd(dx,dx,dy*dy)
    r2=muladd(xi,xi,yi*yi)
    r02=muladd(xj,xj,yj*yj)
    denom=(one(T)-r2)*(one(T)-r02)
    return one(T)+2*Δ2/denom
end

@inline function hyperbolic_grad_chi_target(xi::T,yi::T,xj::T,yj::T) where {T<:Real}
    dx=xi-xj
    dy=yi-yj
    Δ2=muladd(dx,dx,dy*dy)
    r2=muladd(xi,xi,yi*yi)
    r02=muladd(xj,xj,yj*yj)
    one_minus_r2=one(T)-r2
    one_minus_r02=one(T)-r02
    denom=(one_minus_r2*one_minus_r2)*one_minus_r02
    numx=dx*one_minus_r2+xi*Δ2
    numy=dy*one_minus_r2+yi*Δ2
    cx=4*numx/denom
    cy=4*numy/denom
    return cx,cy
end

@inline function hyperbolic_dchi_dn_target(xi::T,yi::T,nxi::T,nyi::T,xj::T,yj::T) where {T<:Real}
    cx,cy=hyperbolic_grad_chi_target(xi,yi,xj,yj)
    return muladd(cx,nxi,cy*nyi)
end

#################################################################################
# LEGENDRE Q DERIVATIVE (Q'_ν(χ)) – PLACEHOLDER HOOK
#################################################################################

function hyperbolic_Qprime(ν::Complex{T},χ::T) where {T<:Real}
    error("hyperbolic_Qprime(ν::Complex{$T}, χ::$T) is not implemented. " *
          "Provide an implementation using your stable Q'_ν(χ) code.")
end

#################################################################################
# SCALAR HYPERBOLIC DLP KERNEL EVALUATION
#################################################################################

@inline function hyperbolic_dlp_kernel_scalar(xi::T,yi::T,nxi::T,nyi::T,xj::T,yj::T,k::Complex{T}) where {T<:Real}
    ν=Complex{T}(-0.5,zero(T))+Complex{T}(zero(T),one(T))*k
    χ=hyperbolic_chi_poincare(xi,yi,xj,yj)
    dchi_dn=hyperbolic_dchi_dn_target(xi,yi,nxi,nyi,xj,yj)
    dQdχ=hyperbolic_Qprime(ν,χ)
    return (dQdχ*dchi_dn)/(2*T(pi))
end

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
#################################################################################

@inline function hyperbolic_default_kernel_fun!(K::AbstractMatrix{Complex{T}},i::Int,j::Int,xi::T,yi::T,nxi::T,nyi::T,xj::T,yj::T,nxj::T,nyj::T,k::Complex{T},scale::Complex{T}) where {T<:Real}
    dx=xi-xj
    dy=yi-yj
    d2=muladd(dx,dx,dy*dy)
    tol2=(eps(T))^2
    if d2≤tol2 && i==j
        return nothing
    end
    val=hyperbolic_dlp_kernel_scalar(xi,yi,nxi,nyi,xj,yj,k)*scale
    @inbounds K[i,j]+=val
    return nothing
end

################################################################################
# REFLECTION OPS AND SCALES (SAME INTERFACE AS EUCLIDEAN)
################################################################################

@inline function _reflect_ops_and_scales(::Type{T},sym::Reflection) where {T<:Real}
    if sym.axis===:y_axis
        return ((1,T(sym.parity<0 ? -1 : 1)),)
    elseif sym.axis===:x_axis
        return ((2,T(sym.parity<0 ? -1 : 1)),)
    elseif sym.axis===:origin
        px,py=sym.parity
        sx=T(px<0 ? -1 : 1)
        sy=T(py<0 ? -1 : 1)
        sxy=sx*sy
        return ((1,sx),(2,sy),(3,sxy))
    else
        error("Unknown reflection axis: $(sym.axis)")
    end
end

#################################################################################
# NO-SYMMETRY HYPERBOLIC DLP: MANY k AND SINGLE k (DEFAULT, WITH CURVATURE)
#################################################################################

function _all_k_nosymm_DLP_hyperbolic!(Ks::Vector{Matrix{Complex{T}}},bp::BoundaryPointsBIM{T},ks::Vector{Complex{T}};multithreaded::Bool=true) where {T<:Real}
    Mk=length(ks)
    N=length(bp.xy)
    @assert length(Ks)==Mk
    tol2=(eps(T))^2
    @use_threads multithreading=multithreaded for m in 1:Mk
        K=Ks[m]
        k=ks[m]
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
                    val=Complex{T}(κi/(2T(π)),zero(T))
                    K[i,i]=val
                else
                    val=hyperbolic_dlp_kernel_scalar(xi,yi,nxi,nyi,xj,yj,k)
                    K[i,j]=val
                end
            end
        end
    end
    return nothing
end

function _one_k_nosymm_DLP_hyperbolic!(K::Matrix{Complex{T}},bp::BoundaryPointsBIM{T},k::Complex{T};multithreaded::Bool=true) where {T<:Real}
    N=length(bp.xy)
    @assert size(K,1)==N && size(K,2)==N
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
                val=Complex{T}(κi/(2T(π)),zero(T))
                K[i,i]=val
            else
                val=hyperbolic_dlp_kernel_scalar(xi,yi,nxi,nyi,xj,yj,k)
                K[i,j]=val
            end
        end
    end
    return nothing
end

#################################################################################
# NO-SYMMETRY HYPERBOLIC DLP: GENERIC KERNEL_FUN! (LIKE EUCLIDEAN CUSTOM PATH)
#################################################################################

function _all_k_nosymm_DLP_hyperbolic!(Ks::Vector{Matrix{Complex{T}}},bp::BoundaryPointsBIM{T},ks::Vector{Complex{T}},kernel_fun!::Function;multithreaded::Bool=true) where {T<:Real}
    Mk=length(ks)
    N=length(bp.xy)
    @assert length(Ks)==Mk
    @use_threads multithreading=multithreaded for i in 1:N
        xi,yi=bp.xy[i]
        nxi,nyi=bp.normal[i]
        @inbounds for j in 1:N
            xj,yj=bp.xy[j]
            nxj,nyj=bp.normal[j]
            @inbounds for m in 1:Mk
                kernel_fun!(Ks[m],i,j,xi,yi,nxi,nyi,xj,yj,nxj,nyj,ks[m],one(Complex{T}))
            end
        end
    end
    return nothing
end

function _one_k_nosymm_DLP_hyperbolic!(K::Matrix{Complex{T}},bp::BoundaryPointsBIM{T},k::Complex{T},kernel_fun!::Function;multithreaded::Bool=true) where {T<:Real}
    N=length(bp.xy)
    fill!(K,zero(eltype(K)))
    @use_threads multithreading=multithreaded for i in 1:N
         xi,yi=bp.xy[i]
         nxi,nyi=bp.normal[i]
         @inbounds for j in 1:N
             xj,yj=bp.xy[j]
             nxj,nyj=bp.normal[j]
             kernel_fun!(K,i,j,xi,yi,nxi,nyi,xj,yj,nxj,nyj,k,one(Complex{T}))
         end
    end
    return nothing
end

#################################################################################
# REFLECTION SYMMETRY: MANY k AND SINGLE k (DEFAULT, WITH CURVATURE)
#################################################################################

function _all_k_reflection_DLP_hyperbolic!(Ks::Vector{Matrix{Complex{T}}},bp::BoundaryPointsBIM{T},sym::Reflection,ks::Vector{Complex{T}};multithreaded::Bool=true) where {T<:Real}
    _all_k_nosymm_DLP_hyperbolic!(Ks,bp,ks;multithreaded)   # direct only
    Mk=length(ks)
    N=length(bp.xy)
    tol2=(eps(T))^2
    shift_x=bp.shift_x
    shift_y=bp.shift_y
    ops=_reflect_ops_and_scales(T,sym)
    nth=Threads.nthreads()
    pt_tls=[zeros(T,2) for _ in 1:nth]
    nn_tls=[zeros(T,2) for _ in 1:nth]
    @use_threads multithreading=multithreaded for i in 1:N
        xi,yi=bp.xy[i]
        nxi,nyi=bp.normal[i]
        tid=Threads.threadid()
        pt=pt_tls[tid]
        nn=nn_tls[tid]
        @inbounds for j in 1:N
            xj0,yj0=bp.xy[j]
            nxj0,nyj0=bp.normal[j]
            @inbounds for (op,scale_r) in ops
                if op==1
                    x_reflect_point_normal!(pt,nn,xj0,yj0,nxj0,nyj0,shift_x)
                elseif op==2
                    y_reflect_point_normal!(pt,nn,xj0,yj0,nxj0,nyj0,shift_y)
                else
                    xy_reflect_point_normal!(pt,nn,xj0,yj0,nxj0,nyj0,shift_x,shift_y)
                end
                xjr,yjr=pt[1],pt[2]
                dx=xi-xjr
                dy=yi-yjr
                d2=muladd(dx,dx,dy*dy)
                if d2>tol2
                    sc=Complex{T}(scale_r,zero(T))
                    @inbounds for m in 1:Mk
                        k=ks[m]
                        val=hyperbolic_dlp_kernel_scalar(xi,yi,nxi,nyi,xjr,yjr,k)*sc
                        Ks[m][i,j]+=val
                    end
                end
            end
        end
    end
    return nothing
end

function _one_k_reflection_DLP_hyperbolic!(K::Matrix{Complex{T}},bp::BoundaryPointsBIM{T},sym::Reflection,k::Complex{T};multithreaded::Bool=true) where {T<:Real}
    _one_k_nosymm_DLP_hyperbolic!(K,bp,k;multithreaded)
    N=length(bp.xy)
    tol2=(eps(T))^2
    shift_x=bp.shift_x
    shift_y=bp.shift_y
    ops=_reflect_ops_and_scales(T,sym)
    pt=[zero(T),zero(T)]
    nn=[zero(T),zero(T)]
    @use_threads multithreading=multithreaded for i in 1:N
        xi,yi=bp.xy[i]
        nxi,nyi=bp.normal[i]
        @inbounds for j in 1:N
            xj0,yj0=bp.xy[j]
            nxj0,nyj0=bp.normal[j]
            @inbounds for (op,scale_r) in ops
                if op==1
                    x_reflect_point_normal!(pt,nn,xj0,yj0,nxj0,nyj0,shift_x)
                elseif op==2
                    y_reflect_point_normal!(pt,nn,xj0,yj0,nxj0,nyj0,shift_y)
                else
                    xy_reflect_point_normal!(pt,nn,xj0,yj0,nxj0,nyj0,shift_x,shift_y)
                end
                xjr,yjr=pt[1],pt[2]
                dx=xi-xjr
                dy=yi-yjr
                d2=muladd(dx,dx,dy*dy)
                if d2>tol2
                    sc=Complex{T}(scale_r,zero(T))
                    val=hyperbolic_dlp_kernel_scalar(xi,yi,nxi,nyi,xjr,yjr,k)*sc
                    K[i,j]+=val
                end
            end
        end
    end
    return K
end

#################################################################################
# REFLECTION SYMMETRY: GENERIC KERNEL_FUN! (LIKE EUCLIDEAN CUSTOM PATH)
#################################################################################

function _all_k_reflection_DLP_hyperbolic!(Ks::Vector{Matrix{Complex{T}}},bp::BoundaryPointsBIM{T},sym::Reflection,ks::Vector{Complex{T}},kernel_fun!::Function;multithreaded::Bool=true) where {T<:Real}
    Mk=length(ks)
    N=length(bp.xy)
    shift_x=bp.shift_x
    shift_y=bp.shift_y
    ops=_reflect_ops_and_scales(T,sym)
    nth=Threads.nthreads()
    pt_tls=[zeros(T,2) for _ in 1:nth]
    nn_tls=[zeros(T,2) for _ in 1:nth]
    @use_threads multithreading=multithreaded for i in 1:N
        xi,yi=bp.xy[i]
        nxi,nyi=bp.normal[i]
        tid=Threads.threadid()
        pt=pt_tls[tid]
        nn=nn_tls[tid]
        @inbounds for j in 1:N
            xj0,yj0=bp.xy[j]
            nxj0,nyj0=bp.normal[j]
            @inbounds for (op,scale_r) in ops
                if op==1
                    x_reflect_point_normal!(pt,nn,xj0,yj0,nxj0,nyj0,shift_x)
                elseif op==2
                    y_reflect_point_normal!(pt,nn,xj0,yj0,nxj0,nyj0,shift_y)
                else
                    xy_reflect_point_normal!(pt,nn,xj0,yj0,nxj0,nyj0,shift_x,shift_y)
                end
                scale=Complex{T}(scale_r,zero(T))
                @inbounds for m in 1:Mk
                    kernel_fun!(Ks[m],i,j,
                                xi,yi,nxi,nyi,
                                pt[1],pt[2],nn[1],nn[2],
                                ks[m],scale)
                end
            end
        end
    end
    return nothing
end

function _one_k_reflection_DLP_hyperbolic!(K::AbstractMatrix{Complex{T}},bp::BoundaryPointsBIM{T},sym::Reflection,k::Complex{T},kernel_fun!::Function;multithreaded::Bool=true) where {T<:Real}
    _one_k_nosymm_DLP_hyperbolic!(K,bp,k,kernel_fun!;multithreaded)
    N=length(bp.xy)
    shift_x=bp.shift_x
    shift_y=bp.shift_y
    ops=_reflect_ops_and_scales(T,sym)
    pt=[zero(T),zero(T)]
    nn=[zero(T),zero(T)]
    @use_threads multithreading=multithreaded for i in 1:N
        xi,yi=bp.xy[i]
        nxi,nyi=bp.normal[i]
        @inbounds for j in 1:N
            xj0,yj0=bp.xy[j]
            nxj0,nyj0=bp.normal[j]
            @inbounds for (op,scale_r) in ops
                if op==1
                    x_reflect_point_normal!(pt,nn,xj0,yj0,nxj0,nyj0,shift_x)
                elseif op==2
                    y_reflect_point_normal!(pt,nn,xj0,yj0,nxj0,nyj0,shift_y)
                else
                    xy_reflect_point_normal!(pt,nn,xj0,yj0,nxj0,nyj0,shift_x,shift_y)
                end
                kernel_fun!(K,i,j,xi,yi,nxi,nyi,pt[1],pt[2],nn[1],nn[2],k,Complex{T}(scale_r,zero(T)))
            end
        end
    end
    return K
end

#################################################################################
# ROTATION SYMMETRY: MANY k AND SINGLE k (DEFAULT, WITH CURVATURE)
#################################################################################

function _all_k_rotation_DLP_hyperbolic!(Ks::Vector{Matrix{Complex{T}}},bp::BoundaryPointsBIM{T},sym::Rotation,ks::Vector{Complex{T}};multithreaded::Bool=true) where {T<:Real}
    _all_k_nosymm_DLP_hyperbolic!(Ks,bp,ks;multithreaded)
    Mk=length(ks)
    N=length(bp.xy)
    tol2=(eps(T))^2
    cx,cy=sym.center
    ctab,stab,χ=_rotation_tables(T,sym.n,mod(sym.m,sym.n))
    nth=Threads.nthreads()
    pt_tls=[zeros(T,2) for _ in 1:nth]
    nn_tls=[zeros(T,2) for _ in 1:nth]
    @use_threads multithreading=multithreaded for i in 1:N
        xi,yi=bp.xy[i]
        nxi,nyi=bp.normal[i]
        tid=Threads.threadid()
        pt=pt_tls[tid]
        nn=nn_tls[tid]
        @inbounds for j in 1:N
            xj0,yj0=bp.xy[j]
            nxj0,nyj0=bp.normal[j]
            @inbounds for l in 2:sym.n
                rot_point_normal!(pt,nn,xj0,yj0,nxj0,nyj0,cx,cy,ctab[l],stab[l])
                xjr,yjr=pt[1],pt[2]
                dx=xi-xjr
                dy=yi-yjr
                d2=muladd(dx,dx,dy*dy)
                if d2>tol2
                    phase=χ[l]
                    @inbounds for m in 1:Mk
                        k=ks[m]
                        val=hyperbolic_dlp_kernel_scalar(xi,yi,nxi,nyi,xjr,yjr,k)*phase
                        Ks[m][i,j]+=val
                    end
                end
            end
        end
    end
    return nothing
end

function _one_k_rotation_DLP_hyperbolic!(K::AbstractMatrix{Complex{T}},bp::BoundaryPointsBIM{T},sym::Rotation,k::Complex{T};multithreaded::Bool=true) where {T<:Real}
    _one_k_nosymm_DLP_hyperbolic!(K,bp,k;multithreaded)
    N=length(bp.xy)
    tol2=(eps(T))^2
    cx,cy=sym.center
    ctab,stab,χ=_rotation_tables(T,sym.n,mod(sym.m,sym.n))
    pt=[zero(T),zero(T)]
    nn=[zero(T),zero(T)]
    @use_threads multithreading=multithreaded for i in 1:N
        xi,yi=bp.xy[i]
        nxi,nyi=bp.normal[i]
        @inbounds for j in 1:N
            xj0,yj0=bp.xy[j]
            nxj0,nyj0=bp.normal[j]
            @inbounds for l in 2:sym.n
                rot_point_normal!(pt,nn,xj0,yj0,nxj0,nyj0,cx,cy,ctab[l],stab[l])
                xjr,yjr=pt[1],pt[2]
                dx=xi-xjr
                dy=yi-yjr
                d2=muladd(dx,dx,dy*dy)
                if d2>tol2
                    phase=χ[l]
                    val=hyperbolic_dlp_kernel_scalar(xi,yi,nxi,nyi,xjr,yjr,k)*phase
                    K[i,j]+=val
                end
            end
        end
    end
    return K
end

#################################################################################
# ROTATION SYMMETRY: GENERIC KERNEL_FUN! (LIKE EUCLIDEAN CUSTOM PATH)
#################################################################################

function _all_k_rotation_DLP_hyperbolic!(Ks::Vector{Matrix{Complex{T}}},bp::BoundaryPointsBIM{T},sym::Rotation,ks::Vector{Complex{T}},kernel_fun!::Function;multithreaded::Bool=true) where {T<:Real}
    Mk=length(ks)
    N=length(bp.xy)
    cx,cy=sym.center
    ctab,stab,χ=_rotation_tables(T,sym.n,mod(sym.m,sym.n))
    nth=Threads.nthreads()
    pt_tls=[zeros(T,2) for _ in 1:nth]
    nn_tls=[zeros(T,2) for _ in 1:nth]
    @use_threads multithreading=multithreaded for i in 1:N
        xi,yi=bp.xy[i]
        nxi,nyi=bp.normal[i]
        tid=Threads.threadid()
        pt=pt_tls[tid]
        nn=nn_tls[tid]
        @inbounds for j in 1:N
            xj0,yj0=bp.xy[j]
            nxj0,nyj0=bp.normal[j]
            @inbounds for l in 2:sym.n
                rot_point_normal!(pt,nn,xj0,yj0,nxj0,nyj0,cx,cy,ctab[l],stab[l])
                phase=χ[l]
                @inbounds for m in 1:Mk
                    kernel_fun!(Ks[m],i,j,xi,yi,nxi,nyi,pt[1],pt[2],nn[1],nn[2],ks[m],phase)
                end
            end
        end
    end
    return nothing
end

function _one_k_rotation_DLP_hyperbolic!(K::AbstractMatrix{Complex{T}},bp::BoundaryPointsBIM{T},sym::Rotation,k::Complex{T},kernel_fun!::Function;multithreaded::Bool=true) where {T<:Real}
    _one_k_nosymm_DLP_hyperbolic!(K,bp,k,kernel_fun!;multithreaded)
    N=length(bp.xy)
    cx,cy=sym.center
    ctab,stab,χ=_rotation_tables(T,sym.n,mod(sym.m,sym.n))
    pt=[zero(T),zero(T)]
    nn=[zero(T),zero(T)]
    @use_threads multithreading=multithreaded for i in 1:N
        xi,yi=bp.xy[i]
        nxi,nyi=bp.normal[i]
        @inbounds for j in 1:N
            xj0,yj0=bp.xy[j]
            nxj0,nyj0=bp.normal[j]
            @inbounds for l in 2:sym.n
                rot_point_normal!(pt,nn,xj0,yj0,nxj0,nyj0,cx,cy,ctab[l],stab[l])
                kernel_fun!(K,i,j,xi,yi,nxi,nyi,pt[1],pt[2],nn[1],nn[2],k,χ[l])
            end
        end
    end
    return K
end

#################################################################################
# TOP-LEVEL DISPATCH: compute_kernel_matrices_DLP_hyperbolic!
#################################################################################

function compute_kernel_matrices_DLP_hyperbolic!(Ks::Vector{Matrix{Complex{T}}},bp::BoundaryPointsBIM{T},symmetry::Union{Vector{Any},Nothing},ks::Vector{Complex{T}};multithreaded::Bool=true,kernel_fun::Union{Symbol,Function}=:default) where {T<:Real}
    if symmetry===nothing
        return compute_kernel_matrices_DLP_hyperbolic!(Ks,bp,ks;multithreaded,kernel_fun)
    else
        try
            compute_kernel_matrices_DLP_hyperbolic!(Ks,bp,symmetry[1],ks;multithreaded,kernel_fun)
        catch _
            error("Error computing hyperbolic kernel matrices with symmetry $(symmetry): ")
        end
    end
end

##################################################################################
# Internal dispatchers for different symmetry cases (mirroring Euclidean)
##################################################################################

function compute_kernel_matrices_DLP_hyperbolic!(Ks::Vector{Matrix{Complex{T}}},bp::BoundaryPointsBIM{T},ks::Vector{Complex{T}};multithreaded::Bool=true,kernel_fun::Union{Symbol,Function}=:default) where {T<:Real}
    kernel_fun===:default && return _all_k_nosymm_DLP_hyperbolic!(Ks,bp,ks;multithreaded)
    return _all_k_nosymm_DLP_hyperbolic!(Ks,bp,ks,kernel_fun!;multithreaded)
end

function compute_kernel_matrices_DLP_hyperbolic!(K::Matrix{Complex{T}},bp::BoundaryPointsBIM{T},k::Complex{T};multithreaded::Bool=true,kernel_fun::Union{Symbol,Function}=:default) where {T<:Real}
    kernel_fun===:default && return _one_k_nosymm_DLP_hyperbolic!(K,bp,k;multithreaded)
    return _one_k_nosymm_DLP_hyperbolic!(K,bp,k,kernel_fun!;multithreaded)
end

function compute_kernel_matrices_DLP_hyperbolic!(Ks::Vector{Matrix{Complex{T}}},bp::BoundaryPointsBIM{T},sym::Reflection,ks::Vector{Complex{T}};multithreaded::Bool=true,kernel_fun::Union{Symbol,Function}=:default) where {T<:Real}
    kernel_fun===:default && return _all_k_reflection_DLP_hyperbolic!(Ks,bp,sym,ks;multithreaded)
    return _all_k_reflection_DLP_hyperbolic!(Ks,bp,sym,ks,kernel_fun!;multithreaded)
end

function compute_kernel_matrices_DLP_hyperbolic!(K::Matrix{Complex{T}},bp::BoundaryPointsBIM{T},sym::Reflection,k::Complex{T};multithreaded::Bool=true,kernel_fun::Union{Symbol,Function}=:default) where {T<:Real}
    kernel_fun===:default && return _one_k_reflection_DLP_hyperbolic!(K,bp,sym,k;multithreaded)
    return _one_k_reflection_DLP_hyperbolic!(K,bp,sym,k,kernel_fun!;multithreaded)
end

function compute_kernel_matrices_DLP_hyperbolic!(Ks::Vector{Matrix{Complex{T}}},bp::BoundaryPointsBIM{T},sym::Rotation,ks::Vector{Complex{T}};multithreaded::Bool=true,kernel_fun::Union{Symbol,Function}=:default) where {T<:Real}
    kernel_fun===:default && return _all_k_rotation_DLP_hyperbolic!(Ks,bp,sym,ks;multithreaded)
    return _all_k_rotation_DLP_hyperbolic!(Ks,bp,sym,ks,kernel_fun!;multithreaded)
end

function compute_kernel_matrices_DLP_hyperbolic!(K::Matrix{Complex{T}},bp::BoundaryPointsBIM{T},sym::Rotation,k::Complex{T};multithreaded::Bool=true,kernel_fun::Union{Symbol,Function}=:default) where {T<:Real}
    kernel_fun===:default && return _one_k_rotation_DLP_hyperbolic!(K,bp,sym,k;multithreaded)
    return _one_k_rotation_DLP_hyperbolic!(K,bp,sym,k,kernel_fun!;multithreaded)
end