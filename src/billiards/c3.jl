############################
# C3 BILLIARD (3-fold rot) #
############################

"""
    make_c3_fundamental(a::T; x0::T=zero(T), y0::T=zero(T), rot_angle::T=zero(T), scale::T=one(T)) where {T<:Real}

Constructs the **fundamental wedge** (one third) of a smooth C₃-symmetric billiard given in polar form
    r(φ) = 0.5 * (1 + a*(cos(3φ) - sin(6φ))) * scale.

This function returns one `PolarSegment` covering φ ∈ [0, 2π/3] and two `VirtualLineSegment`s closing the wedge to the center.

# Arguments
- `a::T`: Deformation amplitude in the radial law (small |a| recommended so r(φ)>0).
- `x0::T=zero(T)`: x-coordinate of the geometry center.
- `y0::T=zero(T)`: y-coordinate of the geometry center.
- `rot_angle::T=zero(T)`: Rotation of the entire geometry (radians).
- `scale::T=one(T)`: Global radial scale multiplier.

# Returns
- A tuple:
  - `boundary::Vector{Union{PolarSegment,VirtualLineSegment}}`: Fundamental boundary (one polar arc + two virtual radial edges).
  - `corners::Vector{SVector{2,T}}`: Corner points of the wedge (two arc endpoints and the origin).
"""
function make_c3_fundamental(a::T;x0::T=zero(T),y0::T=zero(T),rot_angle::T=zero(T),scale::T=one(T)) where {T<:Real}
    origin=SVector(x0,y0)
    φ_multiplier=T(2)/T(3)          # t ∈ [0,1] → φ ∈ [0, 2π/3]
    r_func=t->begin
        φ=φ_multiplier*π*t
        r=(one(T)/2)*(one(T)+a*(cos(T(3)*φ)-sin(T(6)*φ)))*scale
        SVector(r*cos(φ),r*sin(φ))
    end
    arc=PolarSegment(r_func;origin=origin,rot_angle=rot_angle)
    # arc endpoints at 0 and 2π/3
    pt0=curve(arc,zero(T))
    pt1=curve(arc,one(T))
    # close the wedge with two radial virtual segments to the center
    e1=VirtualLineSegment(pt1,SVector{2,T}(x0,y0);origin=origin,rot_angle=rot_angle)
    e0=VirtualLineSegment(SVector{2,T}(x0,y0),pt0;origin=origin,rot_angle=rot_angle)
    boundary=Union{PolarSegment,VirtualLineSegment}[arc,e1,e0] # the real segment should be first
    corners=[pt0,pt1,SVector(x0,y0)]
    return boundary,corners
end


"""
    make_c3_desymmetrized_full_boundary(a::T; x0::T=zero(T), y0::T=zero(T), rot_angle::T=zero(T), scale::T=one(T)) where {T<:Real}

Constructs the **desymmetrized one-third** of the C₃ boundary (no virtual segments). This is the piece you’ll
typically sample for the BIM when using rotation symmetry.

# Arguments
- `a::T`: Deformation amplitude in the radial law.
- `x0::T=zero(T)`: x-coordinate of the geometry center.
- `y0::T=zero(T)`: y-coordinate of the geometry center.
- `rot_angle::T=zero(T)`: Rotation of the entire geometry (radians).
- `scale::T=one(T)`: Global radial scale multiplier.

# Returns
- A tuple:
  - `boundary::Vector{PolarSegment{T}}`: Single polar arc over φ ∈ [0, 2π/3].
  - `nothing::Vector{SVector{2,T}}`: Empty corners vector for this desymmetrized arc.
"""
function make_c3_desymmetrized_full_boundary(a::T;x0::T=zero(T),y0::T=zero(T),rot_angle::T=zero(T),scale::T=one(T)) where {T<:Real}
    origin=SVector(x0,y0)
    φ_multiplier=T(2)/T(3)
    r_func=t->begin
        φ=φ_multiplier*π*t
        r=(one(T)/2)*(one(T)+a*(cos(T(3)*φ)-sin(T(6)*φ)))*scale
        SVector(r*cos(φ),r*sin(φ))
    end
    seg=PolarSegment(r_func;origin=origin,rot_angle=rot_angle)
    boundary=PolarSegment[seg]
    return boundary,Vector{SVector{2,T}}()
end


"""
    make_full_c3(a::T; x0::T=zero(T), y0::T=zero(T), rot_angle::T=zero(T), scale::T=one(T)) where {T<:Real}

Constructs the **full** C₃ boundary as a single smooth `PolarSegment` over φ ∈ [0, 2π].

# Arguments
- `a::T`: Deformation amplitude in the radial law.
- `x0::T=zero(T)`: x-coordinate of the geometry center.
- `y0::T=zero(T)`: y-coordinate of the geometry center.
- `rot_angle::T=zero(T)`: Rotation of the entire geometry (radians).
- `scale::T=one(T)`: Global radial scale multiplier.

# Returns
- A tuple:
  - `boundary::Vector{PolarSegment{T}}`: One full polar arc.
  - `corners::Vector{SVector{2,T}}`: Empty vector (smooth boundary).
  - `area_full::T`: Exact area enclosed by the full boundary.
"""
function make_full_c3(a::T;x0::T=zero(T),y0::T=zero(T),rot_angle::T=zero(T),scale::T=one(T)) where {T<:Real}
    origin=SVector(x0,y0)
    θ_multiplier=T(2)
    r_func=t->begin
        φ=θ_multiplier*π*t
        r=(one(T)/2)*(one(T)+a*(cos(T(3)*φ)-sin(T(6)*φ)))*scale
        SVector(r*cos(φ),r*sin(φ))
    end
    full_seg=PolarSegment(r_func;origin=origin,rot_angle=rot_angle)
    area_full=compute_area(full_seg)
    boundary=PolarSegment[full_seg]
    corners=SVector{2,T}[]
    return boundary,corners,area_full
end


"""
    struct C3Billiard{T} <: AbsBilliard where {T<:Real}

Defines a C₃-symmetric billiard with fundamental, desymmetrized one-third, and full boundaries.

# Fields
- `fundamental_boundary::Vector{Union{PolarSegment,VirtualLineSegment}}`: C₃ fundamental wedge.
- `full_boundary::Vector{PolarSegment}`: Full smooth boundary (φ ∈ [0,2π]).
- `desymmetrized_full_boundary::Vector{PolarSegment}`: One-third smooth arc (φ ∈ [0,2π/3]).
- `length::T`: Perimeter of the full boundary.
- `length_fundamental::T`: Length of the fundamental wedge (with symmetry accounting).
- `a::T`: Deformation parameter.
- `corners::Vector{SVector{2,T}}`: Wedge corners (fundamental).
- `area::T`: Area of the full billiard.
- `area_fundamental::T`: Area of the fundamental wedge.
- `angles::Vector`: Corner angles for the full boundary (empty here).
- `angles_fundamental::Vector`: Corner angles of the fundamental wedge (here `[2π/3]` at the origin).
- `s_shift::T`: Optional arc-length shift (usually `0`).
"""
struct C3Billiard{T} <: AbsBilliard where {T<:Real}
    fundamental_boundary::Vector{Union{PolarSegment,VirtualLineSegment}}
    full_boundary::Vector{PolarSegment}
    desymmetrized_full_boundary::Vector{PolarSegment}
    length::T
    length_fundamental::T
    a::T
    corners::Vector{SVector{2,T}}
    area::T
    area_fundamental::T
    angles::Vector
    angles_fundamental::Vector
    s_shift::T
end


"""
    C3Billiard(a::T; x0::T=zero(T), y0::T=zero(T), rot_angle::T=zero(T), scale::T=one(T)) :: C3Billiard where {T<:Real}

Constructs a C₃ billiard.

# Arguments
- `a::T`: Deformation amplitude.
- `x0::T=zero(T)`: Center x.
- `y0::T=zero(T)`: Center y.
- `rot_angle::T=zero(T)`: Global rotation (radians).
- `scale::T=one(T)`: Global radial scale.

# Returns
- `C3Billiard{T}`: An instance with fundamental, one-third desymmetrized, and full boundaries.
"""
function C3Billiard(a::T;x0::T=zero(T),y0::T=zero(T),rot_angle::T=zero(T),scale::T=one(T)) :: C3Billiard where {T<:Real}
    fundamental_boundary,corners=make_c3_fundamental(a;x0=x0,y0=y0,rot_angle=rot_angle,scale=scale)
    full_boundary,_,area_full=make_full_c3(a;x0=x0,y0=y0,rot_angle=rot_angle,scale=scale)
    desym_full_boundary,_=make_c3_desymmetrized_full_boundary(a;x0=x0,y0=y0,rot_angle=rot_angle,scale=scale)
    area_fundamental=area_full/T(3)
    length=sum(crv.length for crv in full_boundary)
    length_fundamental=symmetry_accounted_fundamental_boundary_length(fundamental_boundary)
    angles=Any[]
    angles_fundamental=[T(2)*π/T(3)]
    s_shift=zero(T)
    return C3Billiard(fundamental_boundary,full_boundary,desym_full_boundary,length,length_fundamental,a,corners,area_full,area_fundamental,angles,angles_fundamental,s_shift)
end


"""
    make_c3_and_basis(a::T; x0::T=zero(T), y0::T=zero(T), rot_angle::T=zero(T), scale::T=one(T), m::Int=0, nbasis::Int=10)
        -> Tuple{C3Billiard{T}, CornerAdaptedFourierBessel} where {T<:Real}

Convenience helper that builds a `C3Billiard` and a CAFB basis adapted to the C₃ wedge (angle `2π/3`),
with rotation symmetry `Rotation(3, m, (x0,y0))`. If `a` is too large then the basis fails and one should use the Fredholm matrix approach.

# Arguments
- `a::T`: Deformation amplitude.
- `x0::T=zero(T)`, `y0::T=zero(T)`: Center.
- `rot_angle::T=zero(T)`: Global rotation of the geometry (radians).
- `scale::T=one(T)`: Global radial scale.
- `m::Int=0`: Irrep index for C₃ (m = 0, 1, or 2).
- `nbasis::Int=10`: Basis size for CAFB.

# Returns
- A tuple:
  - `billiard::C3Billiard{T}`
  - `basis::CornerAdaptedFourierBessel`
"""
function make_c3_and_basis(a::T;x0::T=zero(T),y0::T=zero(T),rot_angle::T=zero(T),scale::T=one(T),m::Int=0,nbasis::Int=10) where {T<:Real}
    billiard=C3Billiard(a;x0=x0,y0=y0,rot_angle=rot_angle,scale=scale)
    wedge_angle=Float64(2π/3)
    symmetry=Vector{Any}([Rotation(3,m,(x0,y0))]) # Rotation symmetry centered at (x0,y0)
    basis=CornerAdaptedFourierBessel(nbasis,wedge_angle,SVector(x0,y0),rot_angle,symmetry)
    return billiard,basis
end