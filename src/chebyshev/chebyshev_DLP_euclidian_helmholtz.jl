#################################################################################
# Contains efficient constructors and evaluators for piecewise-Chebyshev planed
# approximations of the scaled Hankel function H1x(z) = exp(-i z) * H1^(1)(z) 
# for multiple complex wavenumbers k simultaneously. It does this by storing
# multiple Chebyshev plans (one per k) and evaluating them in a single pass
# through the r values needed (not needing to reevaluate the geometries each new pass).
# This is used in the BEYN_CHEBYSHEV method implementation to speed up matrix constructions.
# The accuracy of the final matrices as compared to SpecialFunctions.jl is on the order of 1e-14
# in the relative Frobenius norm, even for large k values (k≈3000 and rmax≈4.0 giving the largest errors
# in the fourth quadrant of the complex plane where there is also the largest condition number of
# the matrix cond≈O(10^4) and absolute error can max out even at 1e-12 due to exponential growth
# there around the roots while relative error is ≈1e-14). There are otherwiese many sources of error
# (comprising of matrix constructions + lu! + ldiv! + axpy! + SVD + eigen)

# API
# compute_kernel_matrices_DLP_chebyshev! - main matrix construction function for DLP kernel with multiple k values using Chebyshev plans.
#   -> depending on the symmetry of the geometry, it calls different accumulation helpers for default or custom kernels.
#
# assemble_fredholm_matrices! - wrapper around compute_kernel_matrices_DLP_chebyshev! to assemble the full Fredholm matrices 
#      adding the identity and the weight corrections if present.
#
# estimate_rmin_rmax - helper function to estimate suitable rmin and rmax values for Chebyshev plans based on geometry size.
#      Critical to evaluate the chebyshev tables with enough panels and polynomial degree to reach high accuracy in the highly 
#      oscillatory regime of large k*r values.
# M0/30/10/25

#################################################################################
# Reflects a point (x,y) across the x-axis with a potentially shifted axis of reflection
# Inputs:  
#   xr_yr: preallocated vector of length 2 to store reflected point
#   x,y: coordinates of the original point
#   shift_x: location of the reflection axis
#################################################################################
@inline function x_reflect_point!(xr_yr::AbstractVector{T},x::T,y::T,shift_x::T) where {T<:Real}
    xr_yr[1]=2*shift_x-x;xr_yr[2]=y;return nothing
end

#################################################################################
# Reflects a point (x,y) across the y-axis with a potentially shifted axis of reflection
# Inputs:  
#   xr_yr: preallocated vector of length 2 to store reflected point
#   x,y: coordinates of the original point
#   shift_y: location of the reflection axis
#################################################################################
@inline function y_reflect_point!(xr_yr::AbstractVector{T},x::T,y::T,shift_y::T) where {T<:Real}
    xr_yr[1]=x;xr_yr[2]=2*shift_y-y;return nothing
end

#################################################################################
# Reflects a point (x,y) across the x & y-axis with potentially shifted axes of reflection x and y
# Inputs:  
#   xr_yr: preallocated vector of length 2 to store reflected point
#   x,y: coordinates of the original point
#   shift_x: location of the reflection axis
#   shift_y: location of the reflection axis
#################################################################################
@inline function xy_reflect_point!(xr_yr::AbstractVector{T},x::T,y::T,shift_x::T,shift_y::T) where {T<:Real}
    xr_yr[1]=2*shift_x-x;xr_yr[2]=2*shift_y-y;return nothing
end

#################################################################################
# Rotates a point (x,y) about center (cx,cy) by an angle θ given by (cl,sl)=cosθ,sinθ
# Inputs:
#   xr_yr: preallocated vector of length 2 to store rotated point
#   x,y: coordinates of the original point
#   cx,cy: center of rotation
#   cl,sl: cosθ,sinθ
##################################################################################
@inline function rot_point!(xr_yr::AbstractVector{T},x::T,y::T,cx::T,cy::T,cl::T,sl::T) where {T<:Real}
    dx=x-cx; dy=y-cy
    xr_yr[1]=cx+cl*dx-sl*dy
    xr_yr[2]=cy+sl*dx+cl*dy
    return nothing
end

# ==== Variants for custom kernels (need transformed source normals as well) ====

#################################################################################
# Reflects a point (x,y) and normal (nx,ny) across the x-axis with a potentially shifted axis of reflection
# Inputs:
#   xr_yr: preallocated vector of length 2 to store reflected point
#   nx_ny: preallocated vector of length 2 to store reflected normal
#   x,y: coordinates of the original point
#   nx,ny: components of the original normal
#   shift_x: location of the reflection axis
#################################################################################
@inline function x_reflect_point_normal!(xr_yr::AbstractVector{T},nx_ny::AbstractVector{T},x::T,y::T,nx::T,ny::T,shift_x::T) where {T<:Real}
    xr_yr[1]=2*shift_x-x;xr_yr[2]=y
    nx_ny[1]=-nx; nx_ny[2]=ny
    return nothing
end

#################################################################################
# Reflects a point (x,y) and normal (nx,ny) across the y-axis with a potentially shifted axis of reflection
# Inputs:
#   xr_yr: preallocated vector of length 2 to store reflected point
#   nx_ny: preallocated vector of length 2 to store reflected normal
#   x,y: coordinates of the original point
#   nx,ny: components of the original normal
#   shift_y: location of the reflection axis
#################################################################################
@inline function y_reflect_point_normal!(xr_yr::AbstractVector{T},nx_ny::AbstractVector{T},x::T,y::T,nx::T,ny::T,shift_y::T) where {T<:Real}
    xr_yr[1]=x;xr_yr[2]=2*shift_y-y
    nx_ny[1]=nx;nx_ny[2]=-ny
    return nothing
end

#################################################################################
# Reflects a point (x,y) and normal (nx,ny) across the x & y-axis with a potentially shifted axes of reflection
# Inputs:
#   xr_yr: preallocated vector of length 2 to store reflected point
#   nx_ny: preallocated vector of length 2 to store reflected normal
#   x,y: coordinates of the original point
#   nx,ny: components of the original normal
#   shift_x: location of the reflection axis
#   shift_y: location of the reflection axis
#################################################################################
@inline function xy_reflect_point_normal!(xr_yr::AbstractVector{T},nx_ny::AbstractVector{T},
x::T,y::T,nx::T,ny::T,shift_x::T,shift_y::T) where {T<:Real}
    xr_yr[1]=2*shift_x-x;xr_yr[2]=2*shift_y-y
    nx_ny[1]=-nx;nx_ny[2]=-ny
    return nothing
end

#################################################################################
# Rotates a point (x,y) and normal (nx,ny) about center (cx,cy) by an angle θ given by (cl,sl)=cosθ,sinθ
# Inputs:
#   xr_yr: preallocated vector of length 2 to store rotated point
#   nx_ny: preallocated vector of length 2 to store rotated normal
#   x,y: coordinates of the original point
#   cx,cy: center of rotation
#   cl,sl: cosθ,sinθ
#################################################################################
@inline function rot_point_normal!(xr_yr::AbstractVector{T},nx_ny::AbstractVector{T},x::T,y::T,nx::T,ny::T,cx::T,cy::T,cl::T,sl::T) where {T<:Real}
    dx=x-cx;dy=y-cy
    xr_yr[1]=cx+cl*dx-sl*dy
    xr_yr[2]=cy+sl*dx+cl*dy
    nx_ny[1]=cl*nx-sl*ny
    nx_ny[2]=sl*nx+cl*ny
    return nothing
end

#################################################################################
# Evaluate unscaled hankel functions hankelh1(...) for multiple complex k values hidden (used) in plans at a given r.
# invsqrt must be provided to prevent additional calculation in this loop. Internally it calculates the chebyshev interpolation of the scaled and sqrt(r) modified hankel
# function G(r) = sqrt(r) * hankelh1x(1,1,k*r), so that is why we need the complex phases exp(-i k r) and sqrt to get the hankelh1(1,k*r) back.
# Inputs:
#  hvals: preallocated vector of length equal to number of plans to store hankel function evaluations
#  phases: preallocated vector of length equal to number of plans to store complex phases exp(-i k r)
#  plans: vector of ChebHankelPlanH1x containing the k values and chebyshev tables for each k
#  pidx: panel index for the current r
#  t: chebyshev local coordinate for the current r
#  invsqrt: precomputed 1/sqrt(r) for the current r
#  r: r at which to evaluate hankel functions via argument k*r internally in _cheb_clenshaw
#  ab: vector of tuples containing (real(k),imag(k)) for each plan to compute complex phases
#################################################################################
@inline function h1_multi_ks_at_r!(hvals::AbstractVector{ComplexF64},phases::AbstractVector{ComplexF64},plans::AbstractVector{ChebHankelPlanH1x},pidx::Int32,t::Float64,invsqrt::Float64,r::Float64,ab::AbstractVector{NTuple{2,Float64}})
    @inbounds for m in eachindex(plans)
        a,b=ab[m]
        phases[m]=_exp_ikr(a,b,r)
        hvals[m]=phases[m]*_cheb_clenshaw(plans[m].panels[pidx].c1,t)*invsqrt
    end
    return nothing
end

####################### MODULAR ACCUMULATION HELPERS (DEFAULT) ###################

#################################################################################
# Accumulate one default-kernel contribution into K for a given preweighted Hankel from h1_multi_ks_at_r! for
# a geometry that does not containts symmetry. 
#   K: matrix to accumulate into
#   i,j: target and source indices
#   nxi,nyi: components of target normal at i
#   nxj,nyj: components of source normal at j
#   dx,dy: displacement components from source j to target i
#   invr: 1/r where r is the distance from source j to target i
#   h: preweighted Hankel function value at r (i.e., multiplied by -0.5i*k)
# Operation performed:
#   K[i,j] += ((nxi*dx+nyi*dy)/r) * h
#   and mirror: K[j,i] with target normal at j and displacement (x_j - x_i)=(-dx,-dy).
# Note: if i==j, only the first accumulation is done since diagonal is already handled separately. This part
# is missing in the sym! verison since the diagonal is not handled there because when adding all contribution the diagonal
# is well-behaved and does not need special treatment.
#################################################################################
@inline function _accum_dlp_default_nosym!(K::AbstractMatrix{Complex{T}},i::Int,j::Int,nxi::T,nyi::T,nxj::T,nyj::T,dx::T,dy::T,invr::T,h::Complex{T}) where {T<:Real}
    dot_i=(nxi*dx+nyi*dy)*invr
    @inbounds K[i,j]+=dot_i*h
    if i!=j
        dot_j=(nxj*(-dx)+nyj*(-dy))*invr
        @inbounds K[j,i]+=dot_j*h
    end
    return nothing
end

#################################################################################
# Accumulate one default-kernel contribution into K for a given preweighted Hankel from h1_multi_ks_at_r! for
# a geometry that containts symmetry.
# Inputs:
#   K: matrix to accumulate into
#   i,j: target and source indices
#   nxi,nyi: components of target normal at i
#   nxj,nyj: components of source normal at j
#   dx,dy: displacement components from source j to target i
#   invr: 1/r where r is the distance from source j to target i
#   h: preweighted Hankel function value at r (i.e., multiplied by -0.5i*k)
# Operation performed:
#   K[i,j] += ((nxi*dx+nyi*dy)/r) * h
#   and mirror: K[j,i] with target normal at j and displacement (x_j - x_i)=(-dx,-dy).
# Note: if i==j, only the first accumulation is done since diagonal is already handled separately. This part
# is missing in the sym! verison since the diagonal is not handled there because when adding all contribution the diagonal
# is well-behaved and does not need special treatment.
#################################################################################
@inline function _accum_dlp_default_sym!(K::AbstractMatrix{Complex{T}},i::Int,j::Int,nxi::T,nyi::T,nxj::T,nyj::T,dx::T,dy::T,invr::T,h::Complex{T}) where {T<:Real}
    dot_i=(nxi*dx+nyi*dy)*invr
    @inbounds K[i,j]+=dot_i*h
    return nothing
end

#################################################################################
# Accumulate one default-kernel contribution into K for a given preweighted Hankel from h1_multi_ks_at_r! for
# a geometry that does not containts symmetry. This is just a wrapper that accepts seperately the hankel value and the prefactor.
# Inputs:
#   K: matrix to accumulate into
#   i,j: target and source indices
#   nxi,nyi: components of target normal at i
#   nxj,nyj: components of source normal at j
#   dx,dy: displacement components from source j to target i
#   invr: 1/r where r is the distance from source j to target i
#   h: preweighted Hankel function value at r.
#   scale: the prefactor computed separately (-im*k/2)
#################################################################################
@inline function _accum_dlp_default_nosym!(K::AbstractMatrix{Complex{T}},i::Int,j::Int,
nxi::T,nyi::T,nxj::T,nyj::T,dx::T,dy::T,invr::T,h::Complex{T},scale)::Nothing where {T<:Real}
    _accum_dlp_default_nosym!(K,i,j,nxi,nyi,nxj,nyj,dx,dy,invr,scale*h); return nothing
end

#################################################################################
# Accumulate one default-kernel contribution into K for a given preweighted Hankel from h1_multi_ks_at_r! for
# a geometry that does contain symmetry. This is just a wrapper that accepts seperately the hankel value and the prefactor.
# Inputs:
#   K: matrix to accumulate into
#   i,j: target and source indices
#   nxi,nyi: components of target normal at i
#   nxj,nyj: components of source normal at j
#   dx,dy: displacement components from source j to target i
#   invr: 1/r where r is the distance from source j to target i
#   h: preweighted Hankel function value at r.
#   scale: the prefactor computed separately (-im*k/2)
#################################################################################
@inline function _accum_dlp_default_sym!(K::AbstractMatrix{Complex{T}},i::Int,j::Int,
nxi::T,nyi::T,nxj::T,nyj::T,dx::T,dy::T,invr::T,h::Complex{T},scale)::Nothing where {T<:Real}
    _accum_dlp_default_sym!(K,i,j,nxi,nyi,nxj,nyj,dx,dy,invr,scale*h); return nothing
end

#################################################################################
# Construct matrices along complex ks as defined in the plans for which chebyshev interpolations are already precomputed.
# this is optimized for hankel type complex k contour evaluations.
# Inputs:
#   Ks: vector of matrices to fill, one for each complex k in plans. This should be preallocated outside
#   bp: BoundaryPointsBIM containing the boundary points and normals
#   plans: vector of ChebHankelPlanH1x containing the k values and chebyshev tables for each k
#   multithreaded: whether to use multithreading or not
#################################################################################
function _all_k_nosymm_DLP_chebyshev!(Ks::Vector{Matrix{Complex{T}}},bp::BoundaryPointsBIM{T},plans::Vector{ChebHankelPlanH1x};multithreaded::Bool=true) where {T<:Real}
    Mk=length(plans);N=length(bp.xy);tol2=(eps(T))^2
    pref=Vector{Complex{T}}(undef,Mk);ab=Vector{NTuple{2,Float64}}(undef,Mk)
    @inbounds for m in 1:Mk
        km=plans[m].k
        pref[m]=Complex{T}(0,-0.5)*km
        ab[m]=(Float64(real(km)),Float64(imag(km)))
    end
    nth=Threads.nthreads()
    phases_tls=[Vector{ComplexF64}(undef,Mk) for _ in 1:nth]
    hvals_tls=[Vector{ComplexF64}(undef,Mk) for _ in 1:nth]
    pans_ref=plans[1].panels
    @use_threads multithreading=multithreaded for i in 1:N
        xi,yi=bp.xy[i];nxi,nyi=bp.normal[i]
        tid=Threads.threadid();phases=phases_tls[tid];hvals=hvals_tls[tid]
        @inbounds for j in 1:i
            xj,yj=bp.xy[j];nxj,nyj=bp.normal[j] # no temporaries
            dx=xi-xj;dy=yi-yj;d2=muladd(dx,dx,dy*dy) # this is the hypotenuse squared
            if d2≤tol2
                val=Complex{T}(bp.curvature[i]/(2π)) # diagonal limit for non-symmetric DLP
                @inbounds begin
                    @inbounds for m in 1:Mk
                        Ks[m][i,j]=val
                        if i!=j;Ks[m][j,i]=val;end
                    end
                end
            else
                r=sqrt(d2);invr=inv(r) # compute r and 1/r for that matrix entry
                p=_find_panel(pans_ref,r);P=pans_ref[p] # find which panel this r belongs to to do chebyshev interp
                t=(2*r-(P.b+P.a))/(P.b-P.a);invsqrt=inv(sqrt(r)) # compute local chebyshev coordinate and 1/sqrt(r) needed for hankel evaluation
                h1_multi_ks_at_r!(hvals,phases,plans,Int32(p),t,invsqrt,r,ab) # evaluate hankel functions for all ks at this r efficiently
                @inbounds for m in 1:Mk # accumulate into each matrix on the contour the respective contribution
                    h=pref[m]*hvals[m] # preweight hankel with -im*k/2
                    _accum_dlp_default_nosym!(Ks[m],i,j,nxi,nyi,nxj,nyj,dx,dy,invr,h)
                end
            end
        end
    end
    return nothing
end

#################################################################################
# Construct a single matrix at a complex k as defined in the plan for which chebyshev interpolations are already precomputed. This one only fills the upper triangle and mirrors it to the lower triangle and used for a single matrix.
# Inputs:
#   K: matrix to fill
#   bp: BoundaryPointsBIM containing the boundary points and normals
#   plan: ChebHankelPlanH1x containing the k value and chebyshev tables
#   multithreaded: whether to use multithreading or not
#################################################################################
function _one_k_nosymm_DLP_chebyshev!(K::AbstractMatrix{Complex{T}},bp::BoundaryPointsBIM{T},plan::ChebHankelPlanH1x;multithreaded::Bool=true) where {T<:Real}
    N=length(bp.xy);tol2=(eps(T))^2;k=plan.k;pref=Complex{T}(0,-0.5)*k;a,b=Float64(real(k)),Float64(imag(k));pans=plan.panels
    fill!(K,zero(eltype(K)))
    @use_threads multithreading=multithreaded for i in 1:N
        xi,yi=bp.xy[i];nxi,nyi=bp.normal[i]
        @inbounds for j in 1:i
            xj,yj=bp.xy[j];nxj,nyj=bp.normal[j];dx=xi-xj;dy=yi-yj;d2=muladd(dx,dx,dy*dy)
            if d2≤tol2
                val=Complex{T}(bp.curvature[i]/(2π));K[i,j]=val;if i!=j;K[j,i]=val;end
            else
                r=sqrt(d2);invr=inv(r)
                p=_find_panel(pans,r)
                P=pans[p]
                t=(2*r-(P.b+P.a))/(P.b-P.a)
                invsqrt=inv(sqrt(r))
                phase=_exp_ikr(a,b,r)
                h1x=_cheb_clenshaw(P.c1,t)
                h=pref*(phase*h1x*invsqrt)
                _accum_dlp_default_nosym!(K,i,j,nxi,nyi,nxj,nyj,dx,dy,invr,h)
            end
        end
    end
    return nothing
end

#################################################################################
# Construct matrices along complex ks as defined in the plans for which chebyshev interpolations are already precomputed.
# this is optimized for hankel type complex k contour evaluations.
# Inputs:
#   Ks: vector of matrices to fill, one for each complex k in plans. This should be preallocated outside
#   bp: BoundaryPointsBIM containing the boundary points and normals
#   plans: vector of ChebHankelPlanH1x containing the k values and chebyshev tables for each k
#   kernel_fun!: function to compute the kernel contribution directly. This should have the signature
#       : kernel_fun!(K::AbstractMatrix{Complex{T}},i::Int,j::Int,xi::T,yi::T,nxi::T,nyi::T,xj::T,yj::T,nxj::T,nyj::T,k::Complex{T},scale::Complex{T}) where {T<:Real}
#   multithreaded: whether to use multithreading or not
#################################################################################
function _all_k_nosymm_DLP_chebyshev!(Ks::Vector{Matrix{Complex{T}}},bp::BoundaryPointsBIM{T},plans::Vector{ChebHankelPlanH1x},kernel_fun!::Function;multithreaded::Bool=true) where {T<:Real}
    Mk=length(plans)
    N=length(bp.xy)
    nth=Threads.nthreads()
    @use_threads multithreading=multithreaded for i in 1:N
        xi,yi=bp.xy[i];nxi,nyi=bp.normal[i]
        @inbounds for j in 1:N
            xj,yj=bp.xy[j];nxj,nyj=bp.normal[j]
            @inbounds for m in 1:Mk
                kernel_fun!(Ks[m],i,j,xi,yi,nxi,nyi,xj,yj,nxj,nyj,plans[m].k,one(Complex{T}))
            end
        end
    end
    return nothing
end

#################################################################################
# Construct a single matrix at a complex k as defined in the plan for which chebyshev interpolations are already precomputed. This one only fills the whole matrix since custom kernels may not be symmetric.
# Inputs:
#   K: matrix to fill
#   bp: BoundaryPointsBIM containing the boundary points and normals
#   k: complex wavenumber
#   kernel_fun!: function to compute the kernel contribution directly. This should have the signature
#       : kernel_fun!(K::AbstractMatrix{Complex{T}},i::Int,j::Int,xi::T,yi::T,nxi::T,nyi::T,xj::T,yj::T,nxj::T,     #       nyj::T,k::Complex{T},scale::Complex{T}) where {T<:Real}
#   multithreaded: whether to use multithreading or not
#################################################################################
function _one_k_nosymm_DLP_chebyshev!(K::Matrix{Complex{T}},bp::BoundaryPointsBIM{T},plan::ChebHankelPlanH1x,kernel_fun!::Function;multithreaded::Bool=true) where {T<:Real}
    N=length(bp.xy);fill!(K,zero(eltype(K)))
    @use_threads multithreading=multithreaded for i in 1:N
        xi,yi=bp.xy[i];nxi,nyi=bp.normal[i]
        @inbounds for j in 1:N
            xj,yj=bp.xy[j];nxj,nyj=bp.normal[j];kernel_fun!(K,i,j,xi,yi,nxi,nyi,xj,yj,nxj,nyj,plan.k,one(Complex{T}))
        end
    end
    return nothing
end

#################################################################################
# Given a reflection symmetry, return the operations and scales needed to generate all image contributions
# Inputs:
#   T: real type
#   sym: Reflection symmetry object
# Outputs:
#   ops: tuple of tuples where each inner tuple is (operation_index,scale)
#       operation_index: 1 for x-axis reflection, 2 for y-axis reflection, 3 for origin reflection
#       scale: +1 or -1 depending on parity
#################################################################################
@inline function _reflect_ops_and_scales(::Type{T},sym::Reflection) where {T<:Real}
    if sym.axis===:y_axis
        return ((1,T(sym.parity<0 ? -1 : 1)),)
    elseif sym.axis===:x_axis
        return ((2,T(sym.parity<0 ? -1 : 1)),)
    elseif sym.axis===:origin
        px,py=sym.parity;sx=T(px<0 ? -1 : 1);sy=T(py<0 ? -1 : 1);sxy=sx*sy
        return ((1,sx),(2,sy),(3,sxy))
    else
        error("Unknown reflection axis: $(sym.axis)")
    end
end

#################################################################################
# Construct matrices along complex ks as defined in the plans for which chebyshev interpolations are already precomputed.
# this is optimized for hankel type complex k contour evaluations and reflection symmetry.
# Inputs:
#   Ks: vector of matrices to fill, one for each complex k in plans. This should be preallocated outside
#   bp: BoundaryPointsBIM containing the boundary points and normals  
#   sym: Reflection symmetry object
#   plans: vector of ChebHankelPlanH1x containing the k values and chebyshev tables for each k
#   multithreaded: whether to use multhreading or not
#################################################################################
function _all_k_reflection_DLP_chebyshev!(Ks::Vector{Matrix{Complex{T}}},bp::BoundaryPointsBIM{T},sym::Reflection,plans::Vector{ChebHankelPlanH1x};multithreaded::Bool=true) where {T<:Real}
    _all_k_nosymm_DLP_chebyshev!(Ks,bp,plans;multithreaded)  # direct only
    Mk=length(plans);N=length(bp.xy);tol2=(eps(T))^2;shift_x=bp.shift_x;shift_y=bp.shift_y;ops=_reflect_ops_and_scales(T,sym)
    pref=Vector{Complex{T}}(undef,Mk)
    ab=Vector{NTuple{2,Float64}}(undef,Mk)
    @inbounds for m in 1:Mk
        km=plans[m].k
        pref[m]=Complex{T}(0,-0.5)*km
        ab[m]=(Float64(real(km)),Float64(imag(km)))
    end
    nth=Threads.nthreads()
    phases_tls=[Vector{ComplexF64}(undef,Mk) for _ in 1:nth]
    hvals_tls=[Vector{ComplexF64}(undef,Mk) for _ in 1:nth]
    pt_tls=[zeros(T,2) for _ in 1:nth]
    pans_ref=plans[1].panels
    @use_threads multithreading=multithreaded for i in 1:N
        xi,yi=bp.xy[i];nxi,nyi=bp.normal[i]
        tid=Threads.threadid();phases=phases_tls[tid];hvals=hvals_tls[tid];pt=pt_tls[tid]
        @inbounds for j in 1:N
            @inbounds for (op,scale) in ops
                if op==1
                    x_reflect_point!(pt,bp.xy[j][1],bp.xy[j][2],shift_x)
                elseif op==2
                    y_reflect_point!(pt,bp.xy[j][1],bp.xy[j][2],shift_y)
                else
                    xy_reflect_point!(pt,bp.xy[j][1],bp.xy[j][2],shift_x,shift_y)
                end
                dx=xi-pt[1];dy=yi-pt[2];d2=muladd(dx,dx,dy*dy)
                if d2>tol2
                    r=sqrt(d2);invr=inv(r)
                    p=_find_panel(pans_ref,r)
                    P=pans_ref[p] # find which panel this r belongs to to do chebyshev interp
                    t=(2*r-(P.b+P.a))/(P.b-P.a) # compute local chebyshev coordinate and 1/sqrt(r) needed for hankel evaluation
                    invsqrt=inv(sqrt(r))
                    h1_multi_ks_at_r!(hvals,phases,plans,Int32(p),t,invsqrt,r,ab) # evaluate hankel functions for all ks at this r efficiently
                    @inbounds for m in 1:Mk # accumulate into each matrix on the contour the respective contribution
                        h=scale*pref[m]*hvals[m]
                        _accum_dlp_default_sym!(Ks[m],i,j,nxi,nyi,bp.normal[j][1],bp.normal[j][2],dx,dy,invr,h)
                    end
                end
            end
        end
    end
    return nothing
end

#################################################################################
# Construct a single matrix at a complex k as defined in the plan for which chebyshev interpolations are already precomputed. This one only fills the whole matrix since custom kernels may not be symmetric.
# Inputs:
#   K: matrix to fill
#   bp: BoundaryPointsBIM containing the boundary points and normals  
#   sym: Reflection symmetry object 
#   plan: ChebHankelPlanH1x containing the k value and chebyshev tables
#   multithreaded: whether to use multhreading or not
#################################################################################
function _one_k_reflection_DLP_chebyshev!(K::AbstractMatrix{Complex{T}},bp::BoundaryPointsBIM{T},sym::Reflection,plan::ChebHankelPlanH1x;multithreaded::Bool=true) where {T<:Real}
    _one_k_nosymm_DLP_chebyshev!(K,bp,plan;multithreaded)
    N=length(bp.xy);tol2=(eps(T))^2;k=plan.k;pref=Complex{T}(0,-0.5)*k;a,b=Float64(real(k)),Float64(imag(k));pans=plan.panels
    shift_x=bp.shift_x;shift_y=bp.shift_y;ops=_reflect_ops_and_scales(T,sym);pt=[zero(T),zero(T)]
    @use_threads multithreading=multithreaded for i in 1:N
        xi,yi=bp.xy[i];nxi,nyi=bp.normal[i];pt_local=pt
        @inbounds for j in 1:N
            xj,yj=bp.xy[j];nxj,nyj=bp.normal[j]
            @inbounds for (op,scale_r) in ops
                if op==1;x_reflect_point!(pt_local,xj,yj,shift_x)
                elseif op==2;y_reflect_point!(pt_local,xj,yj,shift_y)
                else;xy_reflect_point!(pt_local,xj,yj,shift_x,shift_y);end
                dx=xi-pt_local[1];dy=yi-pt_local[2];d2=muladd(dx,dx,dy*dy)
                if d2>tol2
                    r=sqrt(d2);invr=inv(r)
                    p=_find_panel(pans,r)
                    P=pans[p]
                    t=(2*r-(P.b+P.a))/(P.b-P.a)
                    invsqrt=inv(sqrt(r))
                    phase=_exp_ikr(a,b,r)
                    h1x=_cheb_clenshaw(P.c1,t)
                    h=Complex{T}(scale_r,zero(T))*pref*(phase*h1x*invsqrt)
                    _accum_dlp_default_sym!(K,i,j,nxi,nyi,nxj,nyj,dx,dy,invr,h)
                end
            end
        end
    end
    return K
end

#################################################################################
# Construct matrices along complex ks as defined in the plans for which chebyshev interpolations are already precomputed.
# this is optimized for hankel type complex k contour evaluations and reflection symmetry.
# Inputs:
#   Ks: vector of matrices to fill, one for each complex k in plans. This should be preallocated outside
#   bp: BoundaryPointsBIM containing the boundary points and normals  
#   sym: Reflection symmetry object
#   plans: vector of ChebHankelPlanH1x containing the k values and chebyshev tables for each k
#   kernel_fun!: function to compute the kernel contribution directly. This should have the signature
#       : kernel_fun!(K::AbstractMatrix{Complex{T}},i::Int,j::Int,xi::T,yi::T,nxi::T,nyi::T,xj::T,yj::T,nxj::T,nyj::T,k::Complex{T},scale::Complex{T}) where {T<:Real}
#   multithreaded: whether to use multhreading or not
#################################################################################
function _all_k_reflection_DLP_chebyshev!(Ks::Vector{Matrix{Complex{T}}},bp::BoundaryPointsBIM{T},sym::Reflection,plans::Vector{ChebHankelPlanH1x},kernel_fun!::Function;multithreaded::Bool=true) where {T<:Real}
    Mk=length(plans)
    N=length(bp.xy)
    shift_x=bp.shift_x;shift_y=bp.shift_y
    ops=_reflect_ops_and_scales(T,sym)
    nth=Threads.nthreads()
    pt_tls=[zeros(T,2) for _ in 1:nth];nn_tls=[zeros(T,2) for _ in 1:nth]
    @use_threads multithreading=multithreaded for i in 1:N
        xi,yi=bp.xy[i];nxi,nyi=bp.normal[i]
        tid=Threads.threadid();pt=pt_tls[tid];nn=nn_tls[tid]
        @inbounds for j in 1:N
            xj0,yj0=bp.xy[j];nxj0,nyj0=bp.normal[j]
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
                    kernel_fun!(Ks[m],i,j,xi,yi,nxi,nyi,pt[1],pt[2],nn[1],nn[2],plans[m].k,scale)
                end
            end
        end
    end
    return nothing
end

#################################################################################
# Construct a single matrix at a complex k as defined in the plan for which chebyshev interpolations are already precomputed. This one only fills the whole matrix since custom kernels may not be symmetric.
# Inputs:
#   K: matrix to fill
#   bp: BoundaryPointsBIM containing the boundary points and normals  
#   sym: Reflection symmetry object 
#   k: complex wavenumber
#   kernel_fun!: function to compute the kernel contribution directly. This should have the signature
#       : kernel_fun!(K::AbstractMatrix{Complex{T}},i::Int,j::Int,xi::T,yi::T,nxi::T,nyi::T,xj::T,yj::T,nxj::T,     #       nyj::T,k::Complex{T},scale::Complex{T}) where {T<:Real}
#   multithreaded: whether to use multhreading or not
#################################################################################
function _one_k_reflection_DLP_chebyshev!(K::AbstractMatrix{Complex{T}},bp::BoundaryPointsBIM{T},sym::Reflection,plan::ChebHankelPlanH1x,kernel_fun!::Function;multithreaded::Bool=true) where {T<:Real}
    _one_k_nosymm_DLP_chebyshev!(K,bp,plan,kernel_fun!;multithreaded)
    N=length(bp.xy);shift_x=bp.shift_x;shift_y=bp.shift_y;ops=_reflect_ops_and_scales(T,sym);pt=[zero(T),zero(T)];nn=[zero(T),zero(T)]
    @use_threads multithreading=multithreaded for i in 1:N
        xi,yi=bp.xy[i];nxi,nyi=bp.normal[i]
        @inbounds for j in 1:N
            xj0,yj0=bp.xy[j];nxj0,nyj0=bp.normal[j]
            @inbounds for (op,scale_r) in ops
                if op==1;x_reflect_point_normal!(pt,nn,xj0,yj0,nxj0,nyj0,shift_x)
                elseif op==2;y_reflect_point_normal!(pt,nn,xj0,yj0,nxj0,nyj0,shift_y)
                else;xy_reflect_point_normal!(pt,nn,xj0,yj0,nxj0,nyj0,shift_x,shift_y);end
                kernel_fun!(K,i,j,xi,yi,nxi,nyi,pt[1],pt[2],nn[1],nn[2],plan.k,Complex{T}(scale_r,zero(T)))
            end
        end
    end
    return K
end

#################################################################################
# Construct matrices along complex ks as defined in the plans for which chebyshev interpolations are already precomputed.
# this is optimized for hankel type complex k contour evaluations and rotational symmetry.
# Inputs:
#   Ks: vector of matrices to fill, one for each complex k in plans. This should be preallocated outside
#   bp: BoundaryPointsBIM containing the boundary points and normals  
#   sym: Rotation symmetry object
#   plans: vector of ChebHankelPlanH1x containing the k values and chebyshev tables for each k
#   multithreaded: whether to use multhreading or not
#################################################################################
function _all_k_rotation_DLP_chebyshev!(Ks::Vector{Matrix{Complex{T}}},bp::BoundaryPointsBIM{T},sym::Rotation,plans::Vector{ChebHankelPlanH1x};multithreaded::Bool=true) where {T<:Real}
    _all_k_nosymm_DLP_chebyshev!(Ks,bp,plans;multithreaded)
    Mk=length(plans)
    N=length(bp.xy)
    tol2=(eps(T))^2
    cx,cy=sym.center
    ctab,stab,χ=_rotation_tables(T,sym.n,mod(sym.m,sym.n))
    pref=Vector{Complex{T}}(undef,Mk)
    ab=Vector{NTuple{2,Float64}}(undef,Mk)
    @inbounds for m in 1:Mk
        km=plans[m].k
        pref[m]=Complex{T}(0,-0.5)*km
        ab[m]=(Float64(real(km)),Float64(imag(km)))
    end
    nth=Threads.nthreads()
    phases_tls=[Vector{ComplexF64}(undef,Mk) for _ in 1:nth]
    hvals_tls=[Vector{ComplexF64}(undef,Mk) for _ in 1:nth];pt_tls=[zeros(T,2) for _ in 1:nth]
    pans_ref=plans[1].panels
    @use_threads multithreading=multithreaded for i in 1:N
        xi,yi=bp.xy[i];nxi,nyi=bp.normal[i]
        tid=Threads.threadid();phases=phases_tls[tid];hvals=hvals_tls[tid];pt=pt_tls[tid]
        @inbounds for j in 1:N
            @inbounds for l in 2:sym.n
                rot_point!(pt,bp.xy[j][1],bp.xy[j][2],cx,cy,ctab[l],stab[l])
                dx=xi-pt[1];dy=yi-pt[2];d2=muladd(dx,dx,dy*dy)
                if d2>tol2
                    r=sqrt(d2);invr=inv(r)
                    p=_find_panel(pans_ref,r);P=pans_ref[p] # find which panel this r belongs to to do chebyshev interp
                    t=(2*r-(P.b+P.a))/(P.b-P.a) # compute local chebyshev coordinate and 1/sqrt(r) needed for hankel evaluation
                    invsqrt=inv(sqrt(r))
                    h1_multi_ks_at_r!(hvals,phases,plans,Int32(p),t,invsqrt,r,ab) # evaluate hankel functions for all ks at this r efficiently
                    χl=χ[l] # rotation phase for this image
                    @inbounds for m in 1:Mk # accumulate into each matrix on the contour the respective contribution
                        h=χl*pref[m]*hvals[m]
                        _accum_dlp_default_sym!(Ks[m],i,j,nxi,nyi,bp.normal[j][1],bp.normal[j][2],dx,dy,invr,h)
                    end
                end
            end
        end
    end
    return nothing
end

#################################################################################
# Construct a single matrix at a complex k as defined in the plan for which chebyshev interpolations are already precomputed. This one only fills the whole matrix since custom kernels may not be symmetric.
# Inputs:
#   K: matrix to fill
#   bp: BoundaryPointsBIM containing the boundary points and normals  
#   sym: Rotation symmetry object 
#   plan: ChebHankelPlanH1x containing the k value and chebyshev tables
#   multithreaded: whether to use multhreading or not
#################################################################################
function _one_k_rotation_DLP_chebyshev!(K::AbstractMatrix{Complex{T}},bp::BoundaryPointsBIM{T},sym::Rotation,plan::ChebHankelPlanH1x;multithreaded::Bool=true) where {T<:Real}
    _one_k_nosymm_DLP_chebyshev!(K,bp,plan;multithreaded)
    N=length(bp.xy);tol2=(eps(T))^2;k=plan.k;pref=Complex{T}(0,-0.5)*k;a,b=Float64(real(k)),Float64(imag(k));pans=plan.panels
    cx,cy=sym.center;ctab,stab,χ=_rotation_tables(T,sym.n,mod(sym.m,sym.n));pt=[zero(T),zero(T)]
    @use_threads multithreading=multithreaded for i in 1:N
        xi,yi=bp.xy[i];nxi,nyi=bp.normal[i];pt_local=pt
        @inbounds for j in 1:N
            nxj,nyj=bp.normal[j]
            @inbounds for l in 2:sym.n
                rot_point!(pt_local,bp.xy[j][1],bp.xy[j][2],cx,cy,ctab[l],stab[l])
                dx=xi-pt_local[1];dy=yi-pt_local[2];d2=muladd(dx,dx,dy*dy)
                if d2>tol2
                    r=sqrt(d2);invr=inv(r)
                    p=_find_panel(pans,r);P=pans[p]
                    t=(2*r-(P.b+P.a))/(P.b-P.a);invsqrt=inv(sqrt(r))
                    phase=_exp_ikr(a,b,r)
                    h1x=_cheb_clenshaw(P.c1,t);h=χ[l]*pref*(phase*h1x*invsqrt)
                    _accum_dlp_default_sym!(K,i,j,nxi,nyi,nxj,nyj,dx,dy,invr,h)
                end
            end
        end
    end
    return nothing
end

#################################################################################
# Construct matrices along complex ks as defined in the plans for which chebyshev interpolations are already precomputed.
# this is optimized for hankel type complex k contour evaluations and rotational symmetry.
# Inputs:
#   Ks: vector of matrices to fill, one for each complex k in plans. This  should be preallocated outside
#   bp: BoundaryPointsBIM containing the boundary points and normals  
#   sym: Rotation symmetry object
#   plans: vector of ChebHankelPlanH1x containing the k values  and chebyshev tables for each k
#   kernel_fun!: function to compute the kernel contribution directly. This should have the signature
#       : kernel_fun!(K::AbstractMatrix{Complex{T}},i::Int,j::Int,xi::T,yi::T,nxi::T,nyi::T,xj::T,yj::T,nxj::T,nyj::T,k::Complex{T},scale::Complex{T}) where {T<:Real}
#   multithreaded: whether to use multhreading or not   
#################################################################################
function _all_k_rotation_DLP_chebyshev!(Ks::Vector{Matrix{Complex{T}}},bp::BoundaryPointsBIM{T},sym::Rotation,plans::Vector{ChebHankelPlanH1x},kernel_fun!::Function;multithreaded::Bool=true) where {T<:Real}
    Mk=length(plans);N=length(bp.xy)
    cx,cy=sym.center;ctab,stab,χ=_rotation_tables(T,sym.n,mod(sym.m,sym.n))
    nth=Threads.nthreads()
    pt_tls=[zeros(T,2) for _ in 1:nth];nn_tls=[zeros(T,2) for _ in 1:nth]
    @use_threads multithreading=multithreaded for i in 1:N
        xi,yi=bp.xy[i];nxi,nyi=bp.normal[i]
        tid=Threads.threadid();pt=pt_tls[tid];nn=nn_tls[tid]
        @inbounds for j in 1:N
            xj0,yj0=bp.xy[j];nxj0,nyj0=bp.normal[j]
            @inbounds for l in 2:sym.n
                rot_point_normal!(pt,nn,xj0,yj0,nxj0,nyj0,cx,cy,ctab[l],stab[l]) # in the custom kernel we need both point and normal rotated perhaps
                phase=χ[l]
                @inbounds for m in 1:Mk
                    kernel_fun!(Ks[m],i,j,xi,yi,nxi,nyi,pt[1],pt[2],nn[1],nn[2],plans[m].k,phase)
                end
            end
        end
    end
    return nothing
end

#################################################################################
# Construct a single matrix at a complex k as defined in the plan for which chebyshev interpolations are already precomputed. This one only fills the whole matrix since custom kernels may not be symmetric.
# Inputs:
#   K: matrix to fill
#   bp: BoundaryPointsBIM containing the boundary points and normals  
#   sym: Rotation symmetry object 
#   k: complex wavenumber
#   kernel_fun!: function to compute the kernel contribution directly. This should have the signature
#       : kernel_fun!(K::AbstractMatrix{Complex{T}},i::Int,j::Int,xi::T,yi::T,nxi::T,nyi::T,xj::T,yj::T,nxj::T,     #       nyj::T,k::Complex{T},scale::Complex{T}) where {T<:Real}
#   multithreaded: whether to use multhreading or not
#################################################################################
function _one_k_rotation_DLP_chebyshev!(K::AbstractMatrix{Complex{T}},bp::BoundaryPointsBIM{T},sym::Rotation,plan::ChebHankelPlanH1x,kernel_fun!::Function;multithreaded::Bool=true) where {T<:Real}
    _one_k_nosymm_DLP_chebyshev!(K,bp,plan,kernel_fun!;multithreaded)
    N=length(bp.xy);cx,cy=sym.center;ctab,stab,χ=_rotation_tables(T,sym.n,mod(sym.m,sym.n));pt=[zero(T),zero(T)];nn=[zero(T),zero(T)]
    @use_threads multithreading=multithreaded for i in 1:N
        xi,yi=bp.xy[i];nxi,nyi=bp.normal[i]
        @inbounds for j in 1:N
            xj0,yj0=bp.xy[j];nxj0,nyj0=bp.normal[j]
            @inbounds for l in 2:sym.n
                rot_point_normal!(pt,nn,xj0,yj0,nxj0,nyj0,cx,cy,ctab[l],stab[l]);kernel_fun!(K,i,j,xi,yi,nxi,nyi,pt[1],pt[2],nn[1],nn[2],plan.k,χ[l])
            end
        end
    end
    return nothing
end

#################################################################################
# Main interface to compute double-layer potential kernel matrices using chebyshev-hankel plans.
# Inputs:
#   Ks: vector of matrices to fill, one for each complex k in plans. This should be preallocated outside
#   bp: BoundaryPointsBIM containing the boundary points and normals  
#   symmetry: either nothing, or a vector of symmetry objects (Reflection or Rotation)
#   plans: vector of ChebHankelPlanH1x containing the k values and chebyshev tables for each k
#   multithreaded: whether to use multhreading or not
#   kernel_fun: either :default to use the default DLP kernel, or a function with signature. OPTIONAL
#       : kernel_fun!(K::AbstractMatrix{Complex{T}},i::Int,j::Int,xi::T,yi::T,nxi::T,nyi::T,xj::T,yj::T,nxj::T,nyj::T,k::Complex{T},scale::Complex{T}) where {T<:Real}
#################################################################################
function compute_kernel_matrices_DLP_chebyshev!(Ks::Vector{Matrix{Complex{T}}},bp::BoundaryPointsBIM{T},symmetry::Union{Vector{Any},Nothing},plans::Vector{ChebHankelPlanH1x};multithreaded::Bool=true,kernel_fun::Union{Symbol,Function}=:default) where {T<:Real}
    if symmetry===nothing
        return compute_kernel_matrices_DLP_chebyshev!(Ks,bp,plans;multithreaded,kernel_fun)
    else
        try 
            compute_kernel_matrices_DLP_chebyshev!(Ks,bp,symmetry[1],plans;multithreaded,kernel_fun)
        catch _
            error("Error computing kernel matrices with symmetry $(symmetry): ")
            
        end
    end
end

##################################################################################
# Internal dispatchers for different symmetry cases
##################################################################################

function compute_kernel_matrices_DLP_chebyshev!(Ks::Vector{Matrix{Complex{T}}},bp::BoundaryPointsBIM{T},plans::Vector{ChebHankelPlanH1x};multithreaded::Bool=true,kernel_fun::Union{Symbol,Function}=:default) where {T<:Real}
    kernel_fun===:default && return _all_k_nosymm_DLP_chebyshev!(Ks,bp,plans;multithreaded)
    return _all_k_nosymm_DLP_chebyshev!(Ks,bp,plans;multithreaded,kernel_fun)
end

function compute_kernel_matrices_DLP_chebyshev!(K::Matrix{Complex{T}},bp::BoundaryPointsBIM{T},plan::ChebHankelPlanH1x;multithreaded::Bool=true,kernel_fun::Union{Symbol,Function}=:default) where {T<:Real}
    kernel_fun===:default && return _one_k_nosymm_DLP_chebyshev!(K,bp,plan;multithreaded)
    return _one_k_nosymm_DLP_chebyshev!(K,bp,plan;multithreaded,kernel_fun)
end

function compute_kernel_matrices_DLP_chebyshev!(Ks::Vector{Matrix{Complex{T}}},bp::BoundaryPointsBIM{T},sym::Reflection,plans::Vector{ChebHankelPlanH1x};multithreaded::Bool=true,kernel_fun::Union{Symbol,Function}=:default) where {T<:Real}
    kernel_fun===:default && return _all_k_reflection_DLP_chebyshev!(Ks,bp,sym,plans;multithreaded)
    return _all_k_reflection_DLP_chebyshev!(Ks,bp,sym,plans,kernel_fun;multithreaded)
end

function compute_kernel_matrices_DLP_chebyshev!(K::Matrix{Complex{T}},bp::BoundaryPointsBIM{T},sym::Reflection,plan::ChebHankelPlanH1x;multithreaded::Bool=true,kernel_fun::Union{Symbol,Function}=:default) where {T<:Real}
    kernel_fun===:default && return _one_k_reflection_DLP_chebyshev!(K,bp,sym,plan;multithreaded)
    return _one_k_reflection_DLP_chebyshev!(K,bp,sym,plan,kernel_fun;multithreaded)
end

function compute_kernel_matrices_DLP_chebyshev!(Ks::Vector{Matrix{Complex{T}}},bp::BoundaryPointsBIM{T},
sym::Rotation,plans::Vector{ChebHankelPlanH1x};multithreaded::Bool=true,kernel_fun::Union{Symbol,Function}=:default) where {T<:Real}
    kernel_fun===:default && return _all_k_rotation_DLP_chebyshev!(Ks,bp,sym,plans;multithreaded)
    return _all_k_rotation_DLP_chebyshev!(Ks,bp,sym,plans,kernel_fun;multithreaded)
end

function compute_kernel_matrices_DLP_chebyshev!(K::Matrix{Complex{T}},bp::BoundaryPointsBIM{T},
sym::Rotation,plan::ChebHankelPlanH1x;multithreaded::Bool=true,kernel_fun::Union{Symbol,Function}=:default) where {T<:Real}
    kernel_fun===:default && return _one_k_rotation_DLP_chebyshev!(K,bp,sym,plan;multithreaded)
    return _one_k_rotation_DLP_chebyshev!(K,bp,sym,plan,kernel_fun;multithreaded)
end

#################################################################################
# Assemble Fredholm matrices from kernel matrices by applying quadrature weights and adding identity.
# Inputs:
#   Ks: vector of matrices to modify in place
#   bp: BoundaryPointsBIM containing the boundary points and quadrature weights
#################################################################################
function assemble_fredholm_matrices!(Ks::Vector{Matrix{Complex{T}}},bp::BoundaryPointsBIM{T}) where {T<:Real}
    ds=bp.ds
    N=length(ds)
    Mk=length(Ks)
    @inbounds Threads.@threads for m in 1:Mk
        K=Ks[m]
        for j in 1:N
            s=-ds[j]
            @views K[:,j].*=s
        end
        for i in 1:N
            K[i,i]+=one(eltype(K))
        end
        filter_matrix!(K)
    end
    return nothing
end

#################################################################################
# Assemble Fredholm matrix from kernel matrix by applying quadrature weights and adding identity.
# Inputs:
#   Ks: single matrix to modify in place
#   bp: BoundaryPointsBIM containing the boundary points and quadrature weights
#################################################################################
function assemble_fredholm_matrices!(K::Matrix{Complex{T}},bp::BoundaryPointsBIM{T}) where {T<:Real}
    ds=bp.ds
    N=length(ds)
    @inbounds for j in 1:N
        @views K[:,j].*= -ds[j]
    end
    @inbounds for i in 1:N
        K[i,i]+=one(eltype(K))
    end
    filter_matrix!(K)
    return nothing
end

###############
#### UTILS ####
###############

##################################################################################
# Estimate suitable rmin and rmax for BIM based on boundary points and symmetry.
# Inputs:
#   bp: BoundaryPointsBIM containing the boundary points
#   sym: either nothing or a vector of symmetry objects (Reflection or Rotation) 
#   pad: tuple of (rmin_pad,rmax_pad) to pad the estimated rmin and rmax
#   rmax_factor: factor to multiply the estimated rmax by
# Outputs:
#   rmin,rmax: estimated minimum and maximum distances between boundary points considering symmetry
##################################################################################
function estimate_rmin_rmax(bp::BoundaryPointsBIM{T},sym::Union{Nothing,Vector{Any}}=nothing;pad=(T(0.9),T(1.1)),rmax_factor::Real=3.0) where {T<:Real}
    N=length(bp.xy);@assert N>1
    tol2=(eps(T))^2
    nth=Threads.nthreads()
    min2_tls=fill(T(Inf),nth)
    max2_tls=fill(zero(T),nth)
    pt_tls=[zeros(T,2) for _ in 1:nth]
    Threads.@threads for i in 1:N
        xi,yi=bp.xy[i]
        tid=Threads.threadid();pt=pt_tls[tid]
        lmin2=typemax(T);lmax2=zero(T)
        @inbounds for j in 1:N
            xj,yj=bp.xy[j]
            if i!=j
                dx=xi-xj;dy=yi-yj;d2=muladd(dx,dx,dy*dy)
                if d2>tol2
                    if d2<lmin2;lmin2=d2;end
                    if d2>lmax2;lmax2=d2;end
                end
            end
            if sym!==nothing
                @inbounds for s in sym
                    if s isa Reflection
                        if s.axis===:y_axis
                            x_reflect_point!(pt,xj,yj,bp.shift_x)
                        elseif s.axis===:x_axis
                            y_reflect_point!(pt,xj,yj,bp.shift_y)
                        else
                            xy_reflect_point!(pt,xj,yj,bp.shift_x,bp.shift_y)
                        end
                        dx=xi-pt[1];dy=yi-pt[2];d2=muladd(dx,dx,dy*dy)
                        if d2>tol2
                            if d2<lmin2;lmin2=d2;end
                            if d2>lmax2;lmax2=d2;end
                        end
                    elseif s isa Rotation
                        cx,cy=s.center
                        ctab,stab,_χ=_rotation_tables(T,s.n,mod(s.m,s.n))
                        @inbounds for l in 2:s.n
                            rot_point!(pt,xj,yj,cx,cy,ctab[l],stab[l])
                            dx=xi-pt[1];dy=yi-pt[2];d2=muladd(dx,dx,dy*dy)
                            if d2>tol2
                                if d2<lmin2;lmin2=d2;end
                                if d2>lmax2;lmax2=d2;end
                            end
                        end
                    end
                end
            end
        end
        if lmin2<min2_tls[tid];min2_tls[tid]=lmin2;end
        if lmax2>max2_tls[tid];max2_tls[tid]=lmax2;end
    end
    min2=minimum(min2_tls);max2=maximum(max2_tls)
    @assert isfinite(min2) && max2>zero(T) "estimate_rmin_rmax: degenerate geometry"
    rmin=pad[1]*sqrt(min2)
    rmax=pad[2]*rmax_factor*sqrt(max2)
    return rmin,rmax
end