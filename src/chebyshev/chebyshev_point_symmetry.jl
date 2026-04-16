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

##################################################################################
# Estimate suitable rmin and rmax for BIM based on boundary points and symmetry.
# Inputs:
#   bp: BoundaryPoints containing the boundary points
#   sym: either nothing or a symmetry object (Reflection or Rotation) 
#   pad: tuple of (rmin_pad,rmax_pad) to pad the estimated rmin and rmax
#   rmax_factor: factor to multiply the estimated rmax by
# Outputs:
#   rmin,rmax: estimated minimum and maximum distances between boundary points considering symmetry
##################################################################################
function estimate_rmin_rmax(bp::BoundaryPoints{T},sym=nothing;pad=(T(0.9),T(1.1)),rmax_factor::Real=3.0) where {T<:Real}
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
            if !isnothing(sym)
                if sym isa Reflection
                    if sym.axis===:y_axis
                        x_reflect_point!(pt,xj,yj,bp.shift_x)
                    elseif sym.axis===:x_axis
                        y_reflect_point!(pt,xj,yj,bp.shift_y)
                    else
                        xy_reflect_point!(pt,xj,yj,bp.shift_x,bp.shift_y)
                    end
                    dx=xi-pt[1];dy=yi-pt[2];d2=muladd(dx,dx,dy*dy)
                    if d2>tol2
                        if d2<lmin2;lmin2=d2;end
                        if d2>lmax2;lmax2=d2;end
                    end
                elseif sym isa Rotation
                    cx,cy=s.center
                    ctab,stab,_χ=_rotation_tables(T,sym.n,mod(sym.m,sym.n))
                    @inbounds for l in 2:sym.n
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
        if lmin2<min2_tls[tid];min2_tls[tid]=lmin2;end
        if lmax2>max2_tls[tid];max2_tls[tid]=lmax2;end
    end
    min2=minimum(min2_tls);max2=maximum(max2_tls)
    @assert isfinite(min2) && max2>zero(T) "estimate_rmin_rmax: degenerate geometry"
    rmin=pad[1]*sqrt(min2)
    rmax=pad[2]*rmax_factor*sqrt(max2)
    return rmin,rmax
end