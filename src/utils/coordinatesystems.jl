#include("../abstracttypes.jl")
using CoordinateTransformations, Rotations, StaticArrays
#Polar coordinates
"""
This struct represents a Cartesian coordinate system with an origin and a rotation angle. It also includes mappings that handle affine transformations within the Cartesian system and between Cartesian and Polar coordinate systems.

# Fields
- `origin::SVector{2,T}`: The origin of the Cartesian coordinate system.
- `rot_angle::T`: The rotation angle of the coordinate system.
- `affine_map::AffineMap{Angle2d{T}, SVector{2, T}}`: An affine transformation map that includes rotation and translation within the Cartesian system.
- `local_map::AffineMap{Angle2d{T}, SVector{2, T}}`: A transformation map that handles the inverse operations for mapping between Cartesian and Polar coordinates.
"""
struct CartesianCS{T} <:CoordinateSystem where T<:Number
    origin::SVector{2,T}
    rot_angle::T
    affine_map::AffineMap{Angle2d{T}, SVector{2, T}}
    local_map::AffineMap{Angle2d{T}, SVector{2, T}}
end

#try moving into inner constructor
"""
This function creates a `CartesianCS` coordinate system with a specified origin and rotation angle. It initializes both the `affine_map` and `local_map` based on the origin and rotation.

# Logic
- The `affine_map` is constructed by first applying a rotation (`Rot`) and then a translation (`Tran`) based on the origin.
- The `local_map` is the inverse operation, first translating back to the origin (`Tran_inv`) and then applying the inverse rotation (`Rot_inv`).

# Arguments
- `origin::SVector{2,T}`: The origin of the Cartesian coordinate system.
- `rot_angle::T`: The rotation angle of the coordinate system.
"""
function CartesianCS(origin::SVector{2,T},rot_angle::T) where T<:Number
    Rot = LinearMap(Angle2d(rot_angle))
    Tran = Translation(origin[1],origin[2])
    Tran_inv = Translation(-origin[1],-origin[2])
    Rot_inv = LinearMap(Angle2d(-rot_angle))
    affine_map = compose(Tran, Rot)
    local_map = compose(Rot_inv, Tran_inv)
    return CartesianCS(origin,rot_angle,affine_map,local_map)
end

"""
This struct represents a Polar coordinate system with an origin and a rotation angle. It includes mappings that handle affine transformations within the Cartesian system and conversions between Cartesian and Polar coordinate systems.

# Fields
- `origin::SVector{2,T}`: The origin of the Polar coordinate system.
- `rot_angle::T`: The rotation angle of the coordinate system.
- `affine_map::AffineMap{Angle2d{T}, SVector{2, T}}`: An affine transformation map that includes rotation and translation within the Cartesian system.
- `local_map::AffineMap{Angle2d{T}, SVector{2, T}}`: A transformation map that handles the conversion from Cartesian to Polar coordinates.
"""
struct PolarCS{T} <:CoordinateSystem  where T<:Number
    origin::SVector{2,T}
    rot_angle::T
    affine_map::AffineMap{Angle2d{T}, SVector{2, T}}  #maps carthesian coordinates
    local_map::AffineMap{Angle2d{T}, SVector{2, T}} #transform carthesian into local polar coords
end

"""
This function creates a `PolarCS` coordinate system with a specified origin and rotation angle. It initializes both the `affine_map` and `local_map` to handle transformations between Cartesian and Polar coordinates.

# Logic
- The `affine_map` is constructed by first applying a rotation (`Rot`) and then a translation (`Tran`) based on the origin.
- The `local_map` handles the conversion from Cartesian to Polar coordinates by applying the inverse translation (`Tran_inv`) and inverse rotation (`Rot_inv`).

# Arguments
- `origin::SVector{2,T}`: The origin of the Polar coordinate system.
- `rot_angle::T`: The rotation angle of the coordinate system.
"""
function PolarCS(origin::SVector{2,T},rot_angle::T) where T<:Number
    Rot = LinearMap(Angle2d(rot_angle))
    Tran = Translation(origin[1],origin[2])
    Tran_inv = Translation(-origin[1],-origin[2])
    Rot_inv = LinearMap(Angle2d(-rot_angle))
    affine_map = compose(Tran, Rot) #already in cartesian coordinates
    local_map = compose(Rot_inv, Tran_inv) # rotate in local polar coordinates
    return PolarCS(origin,rot_angle,affine_map,local_map)
end

"""
- Given a point in Polar coordinates `pt = (r, θ)`, the Cartesian coordinates are computed as:
  - `x = r * cos(θ)`
  - `y = r * sin(θ)`

# Arguments
- `pt::SVector{2,T}`: A point in Polar coordinates.

# Returns
- An `SVector{2,T}` representing the corresponding point in Cartesian coordinates.
"""
function polar_to_cartesian(pt::SVector{2,T}) where T<:Number
    s,c = sincos(pt[2])
    return SVector(pt[1] * c, pt[1] * s)
end
    
"""
- Given a point in Cartesian coordinates `pt = (x, y)`, the Polar coordinates are computed as:
  - `r = hypot(x, y)`
  - `θ = atan(y, x)`

# Arguments
- `pt::SVector{2,T}`: A point in Cartesian coordinates.

# Returns
- An `SVector{2,T}` representing the corresponding point in Polar coordinates.
"""
function cartesian_to_polar(pt::SVector{2,T}) where T<:Number
    return SVector(hypot(pt[1], pt[2]), atan(pt[2], pt[1]))
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