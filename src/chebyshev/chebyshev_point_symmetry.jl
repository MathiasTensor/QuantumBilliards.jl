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