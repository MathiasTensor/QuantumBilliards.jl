#include("../abstracttypes.jl")
using CoordinateTransformations, Rotations, StaticArrays
#Polar coordinates
struct CartesianCS{T} <:CoordinateSystem where T<:Number
    origin::SVector{2,T}
    rot_angle::T
    affine_map::AffineMap{Angle2d{T}, SVector{2, T}}
    local_map::AffineMap{Angle2d{T}, SVector{2, T}}
end

#try moving into inner constructor
function CartesianCS(origin::SVector{2,T},rot_angle::T) where T<:Number
    Rot = LinearMap(Angle2d(rot_angle))
    Tran = Translation(origin[1],origin[2])
    Tran_inv = Translation(-origin[1],-origin[2])
    Rot_inv = LinearMap(Angle2d(-rot_angle))
    affine_map = compose(Tran, Rot)
    local_map = compose(Rot_inv, Tran_inv)
    return CartesianCS(origin,rot_angle,affine_map,local_map)
end

struct PolarCS{T} <:CoordinateSystem  where T<:Number
    origin::SVector{2,T}
    rot_angle::T
    affine_map::AffineMap{Angle2d{T}, SVector{2, T}}  #maps carthesian coordinates
    local_map::AffineMap{Angle2d{T}, SVector{2, T}} #transform carthesian into local polar coords
end

function PolarCS(origin::SVector{2,T},rot_angle::T) where T<:Number
    Rot = LinearMap(Angle2d(rot_angle))
    Tran = Translation(origin[1],origin[2])
    Tran_inv = Translation(-origin[1],-origin[2])
    Rot_inv = LinearMap(Angle2d(-rot_angle))
    affine_map = compose(Tran, Rot) #already in cartesian coordinates
    local_map = compose(Rot_inv, Tran_inv) # rotate in local polar coordinates
    return PolarCS(origin,rot_angle,affine_map,local_map)
end

function polar_to_cartesian(pt::SVector{2,T}) where T<:Number
    s,c = sincos(pt[2])
    return SVector(pt[1] * c, pt[1] * s)
end
    
function cartesian_to_polar(pt::SVector{2,T}; rotation_angle_discontinuity::T = zero(T)) where T<:Number
    if rotation_angle_discontinuity != zero(T)
        # Rotate the point (x, y) by the given rotation_angle_discontinuity
        x_rot = pt[1] * cos(rotation_angle_discontinuity) - pt[2] * sin(rotation_angle_discontinuity)
        y_rot = pt[1] * sin(rotation_angle_discontinuity) + pt[2] * cos(rotation_angle_discontinuity)

        # Convert the rotated point to polar coordinates
        r = hypot(x_rot, y_rot)
        θ = atan(y_rot, x_rot)

        # Subtract the rotation angle from θ to adjust the angle back
        return SVector(r, θ - rotation_angle_discontinuity)
    else
        # No rotation, directly convert to polar coordinates
        r = hypot(pt[1], pt[2])
        θ = atan(pt[2], pt[1])
        return SVector(r, θ)
    end
end

#=
#Complex coordinates
struct ComplexCS{T} <:CoordinateSystem where T<:Number
    origin::SVector{2,T}
    rot_angle::T
end

#Convex coordinates (for convex billiards only)
struct ConvexCS{T} <:CoordinateSystem where T<:Number
    origin::SVector{2,T}
    rot_angle::T
end

cs = CartesianCS(SVector(0.0,0.0),0.0)
=#