###############################################################################
################## Symmetry mapping and projection utilities ##################
###############################################################################

################
#### ALPERT ####
################

# Topology of joins between oriented boundary segments.
# With the current definition, angle = acos(dot(t̂_left, t̂_right)):
#   - angle ≈ 0   => smooth join
#   - angle > 0   => corner
# For now the angle is mainly used to classify joins; a later corner correction
# may need a more precise interior-angle convention.
struct AlpertCompositeTopology{T<:Real}
    prev::Vector{Int}
    next::Vector{Int}
    left_kind::Vector{Symbol}   # :smooth, :corner, :end
    right_kind::Vector{Symbol}
    left_angle::Vector{T}
    right_angle::Vector{T}
end

# needed normalized tangents for the dot product in _join_angle
@inline function _unit_tangent(v::SVector{2,T}) where {T<:Real}
    n=sqrt(v[1]^2+v[2]^2)
    return v/n
end

@inline function _endpoint_distance(a::SVector{2,T},b::SVector{2,T}) where {T<:Real}
    return sqrt((a[1]-b[1])^2+(a[2]-b[2])^2) # hypot(a-b)
end

# compute an approximate angle between segments using start and end tangents. It need not be exact since it is only used to decide whether to apply corner or smooth correction, and the corner correction is robust to angle misspecification.
@inline function _join_angle(t1::SVector{2,T},t2::SVector{2,T}) where {T<:Real}
    u1=_unit_tangent(t1)
    u2=_unit_tangent(t2)
    c=clamp(dot(u1,u2),-one(T),one(T))
    return acos(c)
end

# _is_closed_curve
# Checks if a curve is closed by comparing the start and end points.
# Inputs:
#   - crv::C : A curve object of type C <: AbsCurve.
#   - xtol::T : Tolerance for determining if the curve is closed.
# Outputs:
#   - Bool : True if the curve is closed, false otherwise.
@inline function _is_closed_curve(crv::C;xtol=1e-10) where {C<:AbsCurve}
    x0=curve(crv,[0.0])[1]
    x1=curve(crv,[1.0])[1]
    return _endpoint_distance(x0,x1)<=xtol
end

@inline function _all_closed_curves(boundary)
    return all(_is_closed_curve, boundary)
end

@inline function _is_single_composite_boundary(boundary)
    return !(boundary[1] isa AbstractVector) && ! _all_closed_curves(boundary)
end

# _is_component_closed
# Check if a a vector of curve segments forms a closed component by comparing the start point of the first segment with the end point of the last segment. This is used to determine if we should apply periodic corrections in the Alpert quadrature.
# Inputs:
#   - boundary::Vector{C} : Vector of curve segments that make up the boundary component, where C <: AbsCurve.
#   - xtol::T : Tolerance for checking if the start and end points are close enough to be considered a closed component.
# Outputs:
#   - Bool : True if the component is closed (start and end points are within xtol), false otherwise.
function _is_component_closed(boundary::Vector{C},xtol::T) where {T<:Real,C<:AbsCurve}
    xL=curve(boundary[1],[zero(T)])[1]
    xR=curve(boundary[end],[one(T)])[1]
    return _endpoint_distance(xR,xL)<=xtol
end

# build_join_topology
# Build the topology of joins between oriented boundary segments for Alpert quadrature. This function identifies the previous and next segments for each segment, classifies the type of join (smooth, corner, or end), and computes the angle between segments at the joins. The topology information is used to apply the correct quadrature corrections at joins, especially for corners.
# Inputs:
#   - pts::Vector{BoundaryPointsCFIE{T}} : Vector of boundary points for each segment, where T <: Real.
#   - xtol::T : Tolerance for checking if segments are joined (i.e., if the end of one segment is close enough to the start of the next segment).
#   - angtol::T : Tolerance for classifying joins as smooth or corner based on the angle between segments.
# Outputs:
#   - topos::Vector{AlpertCompositeTopology{T}} : Vector of composite topologies for each boundary component, containing information about joins and angles.
#   - gmaps::Vector{Vector{Int}} : Vector of index mappings for each component, indicating which segments belong to which component.
function build_join_topology(pts::Vector{BoundaryPointsCFIE{T}};xtol::T=T(1e-10),angtol::T=T(1e-8)) where {T<:Real}
    nc=maximum(p.compid for p in pts)
    topos=Vector{AlpertCompositeTopology{T}}(undef,nc)
    gmaps=Vector{Vector{Int}}(undef,nc)
    @inbounds for c in 1:nc
        gmap=findall(p->p.compid==c,pts)
        gmaps[c]=gmap
        n=length(gmap)
        prev=Vector{Int}(undef,n)
        next=Vector{Int}(undef,n)
        left_kind=Vector{Symbol}(undef,n)
        right_kind=Vector{Symbol}(undef,n)
        left_angle=Vector{T}(undef,n)
        right_angle=Vector{T}(undef,n)
        periodic=_endpoint_distance(pts[gmap[end]].xy[end],pts[gmap[1]].xy[1])<=xtol
        for l in 1:n
            prev[l]=(l==1) ? (periodic ? n : 0) : l-1
            next[l]=(l==n) ? (periodic ? 1 : 0) : l+1
        end
        for l in 1:n
            a=gmap[l]
            if prev[l]==0
                left_kind[l]=:end
                left_angle[l]=T(NaN)
            else
                ap=gmap[prev[l]]
                d=_endpoint_distance(pts[ap].xy[end],pts[a].xy[1])
                d>xtol && error("Boundary ordering mismatch at left join of segment $l in component $c: distance = $d")
                θ=_join_angle(pts[ap].tangent[end],pts[a].tangent[1])
                left_angle[l]=θ
                left_kind[l]=(θ<=angtol) ? :smooth : :corner
            end
            if next[l]==0
                right_kind[l]=:end
                right_angle[l]=T(NaN)
            else
                an=gmap[next[l]]
                d=_endpoint_distance(pts[a].xy[end],pts[an].xy[1])
                d>xtol && error("Boundary ordering mismatch at right join of segment $l in component $c: distance = $d")
                θ=_join_angle(pts[a].tangent[end],pts[an].tangent[1])
                right_angle[l]=θ
                right_kind[l]=(θ<=angtol) ? :smooth : :corner
            end
        end
        topos[c]=AlpertCompositeTopology(prev,next,left_kind,right_kind,left_angle,right_angle)
    end
    return topos,gmaps
end

####################
#### CFIE KRESS ####
####################

# mostly used for CFIE where we need to project onto an irrep since we cant construct with Kress's log split since it needs full domain
# this allows Beyn's method to handle symmetries even in the case of CFIE_kress.

# flatten_points
# Flatten a vector of CFIE_kress boundary components into one global point array for all components.
#
# Inputs:
#   - pts::Vector{<:BoundaryPointsCFIE{T}} :
#       Boundary components in CFIE_kress. Each component stores its own
#       `xy` points, and the components are assumed to be ordered exactly
#       as they appear in the global boundary assembly.
#
# Outputs:
#   - xy::Vector{SVector{2,T}} :
#       All boundary points concatenated into a single vector.
#   - offs::Vector{Int} :
#       Component offsets such that component `c` occupies the global range
#       `offs[c] : offs[c+1]-1`.
function flatten_points(pts::Vector{<:BoundaryPointsCFIE{T}}) where {T<:Real}
    offs::Vector{Int}=component_offsets(pts)
    N::Int=offs[end]-1
    xy::Vector{SVector{2,T}}=Vector{SVector{2,T}}(undef,N)
    for c in 1:length(pts)
        o=offs[c];pc=pts[c].xy
        @inbounds for i in 1:length(pc)
            xy[o+i-1]=pc[i]
        end
    end
    return xy,offs
end


# match_index
# Find the unique index in a flattened boundary point array that matches
# a target point `(xr,yr)` within a given tolerance.
#
# Inputs:
#   - xr::T, yr::T :
#       Coordinates of the target point.
#   - xy::Vector{SVector{2,T}} :
#       Flattened boundary point cloud to search in.
#   - tol::T=T(1e-10) :
#       Matching tolerance in Euclidean distance.
#
# Outputs:
#   - best::Int :
#       Unique matching index in `xy`.
#
# Errors:
#   - Throws if no point lies within tolerance.
#   - Throws if more than one point lies within tolerance.
#
# Notes:
#   - This is appropriate for reflection symmetries, where reflected nodes
#     are expected to coincide with sampled nodes up to roundoff.
#   - It is generally not robust enough for rotational symmetry on its own;
#     for rotations we instead use build_rotation_maps_components.
function match_index(xr::T,yr::T,xy::Vector{SVector{2,T}};tol::T=T(1e-10)) where {T<:Real}
    best::Int=0
    dmin::T=typemax(T)
    cnt::Int=0
    @inbounds for j in 1:length(xy)
        dx::T=xy[j][1]-xr
        dy::T=xy[j][2]-yr
        d::T=dx*dx+dy*dy
        if d<tol*tol
            best=j;cnt+=1
        elseif d<dmin
            dmin=d
        end
    end
    cnt==1 && return best
    cnt==0 && error("no match (dmin=$dmin)")
    error("ambiguous match ($cnt)")
end

# build_rotation_maps_components
# Construct the discrete action of a rotational symmetry on a multi-component
# CFIE_kress boundary by matching rotated components through cyclic index shifts.
#
# Inputs:
#   - pts::Vector{<:BoundaryPointsCFIE{T}} :
#       Boundary components of the full CFIE_kress geometry. Each component is
#       assumed to be sampled uniformly in its own periodic parameter.
#   - sym::Rotation :
#       Rotation symmetry object specifying:
#         * `n`      : order of the rotation group C_n
#         * `m`      : irrep index
#         * `center` : rotation center
#   - tol::T=T(1e-8) :
#       Tolerance used when verifying that one rotated component matches
#       another by a single cyclic shift.
#
# Outputs:
#   - Dict{Symbol,Any} with:
#       * `:rot` => rotmaps
#           A vector of global permutation maps. `rotmaps[l+1]` gives the
#           permutation corresponding to rotation by `2π*l/n`.
#       * `:χ` => χ
#           Character table values for the irrep, as returned by
#           `_rotation_tables(T,n,m)`.
#
# Idea:
#   1. For each nontrivial rotation power `l=1,...,n-1`, rotate each source
#      component.
#   2. For every candidate target component, attempt to match the rotated
#      source points by a single cyclic shift.
#   3. Once the target component and shift are found, assemble the global
#      permutation map using component offsets.
#
# Notes:
#   - This is component-aware and therefore works for outer boundaries plus
#     holes.
#   - It requires that a rotated component maps to another component with the
#     same number of sampled nodes -> always the case!.
function build_rotation_maps_components(pts::Vector{<:BoundaryPointsCFIE{T}},sym::Rotation;tol::T=T(1e-8)) where {T<:Real}
    ncomp=length(pts)
    offs=component_offsets(pts)
    lens=[length(p.xy) for p in pts]
    cx,cy=sym.center
    n=sym.n
    cos_tab,sin_tab,χ=_rotation_tables(T,n,sym.m)
    Ntot=offs[end]-1
    rotmaps=Vector{Vector{Int}}(undef,n)
    rotmaps[1]=collect(1:Ntot)
    @inline function find_shift_to_component(pa::Vector{SVector{2,T}},pb::Vector{SVector{2,T}},c::T,s::T)
        Na=length(pa)
        Nb=length(pb)
        Na==Nb || return nothing
        # rotate first point of source component
        p0=pa[1]
        x0r,y0r=_rot_point(p0[1],p0[2],cx,cy,c,s)
        bestj=0
        dmin=typemax(T)
        @inbounds for j in 1:Nb
            dx=pb[j][1]-x0r
            dy=pb[j][2]-y0r
            d=dx*dx+dy*dy
            if d<dmin
                dmin=d
                bestj=j
            end
        end
        dmin>tol^2 && return nothing
        sh=bestj-1
        # verify same shift works for all points
        @inbounds for j in 1:Na
            jj=mod1(j+sh,Nb)
            q=pa[j]
            xr,yr=_rot_point(q[1],q[2],cx,cy,c,s)
            dx=pb[jj][1]-xr
            dy=pb[jj][2]-yr
            d=dx*dx+dy*dy
            d>tol^2 && return nothing
        end
        return sh
    end
    for l in 1:n-1
        c=cos_tab[l+1]
        s=sin_tab[l+1]
        map=Vector{Int}(undef,Ntot)
        target_comp=Vector{Int}(undef,ncomp)
        shift_comp=Vector{Int}(undef,ncomp)
        for a in 1:ncomp
            pa=pts[a].xy
            found=false
            for b in 1:ncomp
                sh=find_shift_to_component(pa,pts[b].xy,c,s)
                if sh!==nothing
                    target_comp[a]=b
                    shift_comp[a]=sh
                    found=true
                    break
                end
            end
            found || error("Could not find rotated target component for component $a")
        end
        # build global permutation
        for a in 1:ncomp
            b=target_comp[a]
            Na=lens[a]
            Nb=lens[b]
            sh=shift_comp[a]
            oa=offs[a]
            ob=offs[b]
            @inbounds for j in 1:Na
                jj=mod1(j+sh,Nb)
                map[oa+j-1]=ob+jj-1
            end
        end
        rotmaps[l+1]=map
    end
    return Dict{Symbol,Any}(:rot=>rotmaps,:χ=>χ)
end

# build_symmetry_maps
# Construct discrete symmetry maps for a CFIE_kress boundary discretization.
#
# Inputs:
#   - pts::Vector{<:BoundaryPointsCFIE{T}} :
#       Full CFIE_kress boundary components.
#   - sym : Symmetry object, either a Reflection or a Rotation.
#   - tol::T=T(1e-10) :
#       Matching tolerance for reflection maps, and lower bound for the
#       rotation-component matcher.
#
# Outputs:
#   - maps::Dict{Symbol,Any} :
#       For reflections:
#         * `:x`, `:y`, `:xy` as needed, each a global permutation vector.
#       For rotations:
#         * `:rot` : vector of global permutation maps for each rotation power
#         * `:χ`   : character table for the chosen irrep
function build_symmetry_maps(pts::Vector{<:BoundaryPointsCFIE{T}},sym;tol::T=T(1e-10)) where {T<:Real}
    xy,_=flatten_points(pts)
    if sym isa Reflection
        maps=Dict{Symbol,Any}()
        if sym.axis==:y_axis
            maps[:x]=[match_index(-p[1],p[2],xy;tol=tol) for p in xy]
        elseif sym.axis==:x_axis
            maps[:y]=[match_index(p[1],-p[2],xy;tol=tol) for p in xy]
        elseif sym.axis==:origin
            maps[:x]=[match_index(-p[1],p[2],xy;tol=tol) for p in xy]
            maps[:y]=[match_index(p[1],-p[2],xy;tol=tol) for p in xy]
            maps[:xy]=[match_index(-p[1],-p[2],xy;tol=tol) for p in xy]
        else
            error("Unknown reflection axis $(sym.axis)")
        end
        return maps
    elseif sym isa Rotation
        return build_rotation_maps_components(pts,sym;tol=max(T(1e-8),tol))
    else
        error("Unknown symmetry type $(typeof(sym))")
    end
end

# apply_projection!
# Apply the symmetry projector to a block of vectors `V`, writing through a
# workspace `W` and copying the projected result back into `V`.
#
# Inputs:
#   - V::AbstractMatrix{Complex{T}} :
#       Block of vectors to project. Columns are independent probe vectors.
#   - W::AbstractMatrix{Complex{T}} :
#       Workspace of the same size as `V`. This is required so the projection
#       is applied from the original `V` rather than in-place during reading.
#   - maps::Dict{Symbol,Any} :
#       Symmetry maps produced by `build_symmetry_maps`.
#   - sym :
#       Symmetry object, either a Reflection or a Rotation.
#
# Outputs:
#   - Returns `V`, modified in-place to contain the projected vectors.
#
# Reflection projectors:
#   - y-axis reflection:
#       P = (I + σ R_x)/2
#   - x-axis reflection:
#       P = (I + σ R_y)/2
#   - origin / XY reflection:
#       P = (I + σ_x R_x + σ_y R_y + σ_xσ_y R_xy)/4
#
# Rotation projector:
#   - For a C_n irrep with characters χ_l,
#       P = (1/n) Σ_l conj(χ_l) R_l
function apply_projection!(V::AbstractMatrix{Complex{T}},W::AbstractMatrix{Complex{T}},maps::Dict{Symbol,Any},sym) where {T<:Real}
    isnothing(sym) && return V
    N,r=size(V)
    @assert size(W)==size(V)
    if sym isa Reflection
        if sym.axis==:y_axis
            σ=sym.parity
            map=maps[:x]
            @inbounds for j in 1:r, i in 1:N
                W[i,j]=(V[i,j]+σ*V[map[i],j])*0.5
            end
        elseif sym.axis==:x_axis
            σ=sym.parity
            map=maps[:y]
            @inbounds for j in 1:r, i in 1:N
                W[i,j]=(V[i,j]+σ*V[map[i],j])*0.5
            end
        elseif sym.axis==:origin
            σx,σy=sym.parity
            mx,my,mxy=maps[:x],maps[:y],maps[:xy]
            @inbounds for j in 1:r, i in 1:N
                W[i,j]=(V[i,j]+σx*V[mx[i],j]+σy*V[my[i],j]+σx*σy*V[mxy[i],j])*0.25
            end
        end
    elseif sym isa Rotation
        n=sym.n
        map_rot=maps[:rot]
        χ=maps[:χ]
        invn=inv(T(n))
        @inbounds for j in 1:r, i in 1:N
            s=zero(Complex{T})
            for l in 1:n
                s+=conj(χ[l])*V[map_rot[l][i],j]
            end
            W[i,j]=s*invn
        end
    end
    copyto!(V,W)
    return V
end

#####################
#### CFIE ALPERT ####
#####################

# _reflection_shifts
# Return the reflection-axis shifts carried by the billiard geometry.
#
# Inputs:
#   - ::Type{T} :
#       Real scalar type used in the current computation.
#   - billiard :
#       Geometry object which may carry `x_axis` and/or `y_axis`.
#
# Outputs:
#   - sx::T :
#       x-shift of the reflection axis for y-axis reflections.
#   - sy::T :
#       y-shift of the reflection axis for x-axis reflections.
#
# Notes:
#   - If the billiard does not define these properties, the shifts default to 0.
@inline function _reflection_shifts(::Type{T},billiard) where {T<:Real}
    sx=hasproperty(billiard,:x_axis) ? T(getproperty(billiard,:x_axis)) : zero(T)
    sy=hasproperty(billiard,:y_axis) ? T(getproperty(billiard,:y_axis)) : zero(T)
    return sx,sy
end

# image_point_x
# Reflect a point across the y-axis (possibly shifted to x = sx).
#
# Inputs:
#   - q::SVector{2,T} :
#       Source point.
#   - billiard :
#       Geometry object that may carry the reflection-axis shift `x_axis`.
#
# Outputs:
#   - qimg::SVector{2,T} :
#       Reflected point.
@inline function image_point_x(q::SVector{2,T},billiard) where {T<:Real}
    sx,_=_reflection_shifts(T,billiard)
    return SVector{2,T}(_x_reflect(q[1],sx),q[2])
end

# image_point_y
# Reflect a point across the x-axis (possibly shifted to y = sy).
#
# Inputs:
#   - q::SVector{2,T} :
#       Source point.
#   - billiard :
#       Geometry object that may carry the reflection-axis shift `y_axis`.
#
# Outputs:
#   - qimg::SVector{2,T} :
#       Reflected point.
@inline function image_point_y(q::SVector{2,T},billiard) where {T<:Real}
    _,sy=_reflection_shifts(T,billiard)
    return SVector{2,T}(q[1],_y_reflect(q[2],sy))
end

# image_point_xy
# Reflect a point across both coordinate axes (origin symmetry, possibly shifted axes).
#
# Inputs:
#   - q::SVector{2,T} :
#       Source point.
#   - billiard :
#       Geometry object that may carry `x_axis` and `y_axis`.
#
# Outputs:
#   - qimg::SVector{2,T} :
#       Doubly reflected point.
@inline function image_point_xy(q::SVector{2,T},billiard) where {T<:Real}
    sx,sy=_reflection_shifts(T,billiard)
    return SVector{2,T}(_x_reflect(q[1],sx),_y_reflect(q[2],sy))
end

@inline function image_tangent_x(t::SVector{2,T}) where {T<:Real}
    tx,ty=_x_reflect_normal(t[1],t[2])
    return SVector{2,T}(tx,ty)
end

@inline function image_tangent_y(t::SVector{2,T}) where {T<:Real}
    tx,ty=_y_reflect_normal(t[1],t[2])
    return SVector{2,T}(tx,ty)
end

@inline function image_tangent_xy(t::SVector{2,T}) where {T<:Real}
    tx,ty=_xy_reflect_normal(t[1],t[2])
    return SVector{2,T}(tx,ty)
end

# image_weight_x
# Return the scalar parity weight for the y-axis reflection image contribution.
#
# Inputs:
#   - sym::Reflection :
#       Reflection symmetry object with `axis == :origin`.
#
# Outputs:
#   - σx :
#       Weight of the x-image term.
@inline image_weight_x(sym::Reflection)=sym.parity[1]

# image_weight_y
# Return the scalar parity weight for the x-axis reflection image contribution.
#
# Inputs:
#   - sym::Reflection :
#       Reflection symmetry object with `axis == :origin`.
#
# Outputs:
#   - σy :
#       Weight of the y-image term.
@inline image_weight_y(sym::Reflection)=sym.parity[2]

# image_weight_xy
# Return the scalar parity weight for the double-reflection image contribution.
#
# Inputs:
#   - sym::Reflection :
#       Reflection symmetry object with `axis == :origin`.
#
# Outputs:
#   - σxy :
#       Weight of the xy-image term, equal to σx*σy.
@inline image_weight_xy(sym::Reflection)=sym.parity[1]*sym.parity[2]

# image_weight
# Return the parity weight for a single-axis reflection contribution.
#
# Inputs:
#   - sym::Reflection :
#       Reflection object with axis `:x_axis` or `:y_axis`.
#
# Outputs:
#   - σ :
#       Scalar parity/sign weight.
#
# Notes:
#   - Do not use this for `:origin`; that case must be split into x, y, and xy images.
@inline function image_weight(sym::Reflection)
    sym.axis==:origin && error("XY/origin reflection must be split into x, y, and xy image terms.")
    return sym.parity
end

# image_point
# Rotate a source point by the l-th nontrivial C_n image.
#
# Inputs:
#   - sym::Rotation :
#       Rotation symmetry descriptor.
#   - q::SVector{2,T} :
#       Source point.
#   - l::Int :
#       Rotation power.
#   - costab, sintab :
#       Precomputed cosine/sine tables from `_rotation_tables`.
#
# Outputs:
#   - qimg::SVector{2,T} :
#       Rotated point.
@inline function image_point(sym::Rotation,q::SVector{2,T},l::Int,costab,sintab) where {T<:Real}
    c=costab[l+1]
    s=sintab[l+1]
    cx,cy=sym.center
    xr,yr=_rot_point(q[1],q[2],T(cx),T(cy),c,s)
    return SVector{2,T}(xr,yr)
end

# image_tangent
# Rotate a source tangent by the l-th nontrivial C_n image.
#
# Inputs:
#   - sym::Rotation :
#       Rotation symmetry descriptor.
#   - t::SVector{2,T} :
#       Source tangent.
#   - l::Int :
#       Rotation power.
#   - costab, sintab :
#       Precomputed cosine/sine tables.
#
# Outputs:
#   - timg::SVector{2,T} :
#       Rotated tangent.
#
# Notes:
#   - Rotations preserve orientation, so no extra minus sign is needed.
@inline function image_tangent(sym::Rotation,t::SVector{2,T},l::Int,costab,sintab) where {T<:Real}
    c=costab[l+1]
    s=sintab[l+1]
    tx,ty=_rot_vec(t[1],t[2],c,s)
    return SVector{2,T}(tx,ty)
end

#
#    apply_symmetries_to_boundary_points(pts::Vector{BoundaryPointsCFIE{T}},symmetries::Union{Vector{Any},Nothing},billiard::Bi;same_direction::Bool=true) #  #where {Bi<:AbsBilliard,T<:Real}
#
#Extend CFIE boundary-point components from a desymmetrized boundary to the full
#boundary by applying reflections and/or rotations.
#
#Ordering is matched to `apply_symmetries_to_boundary_function`, so the expanded
#geometry and expanded boundary density stay aligned component-by-component.
#
#Reflection block order:
# - `YReflection`  -> `[orig..., x...]`
# - `XReflection`  -> `[orig..., y...]`
# - `XYReflection` -> `[orig..., x..., y..., xy...]`
#
# Rotation block order:
# - `[orig..., rot(l=1)..., rot(l=2)..., ...]`
#
#If `same_direction=true`, single reflections reverse point order so that the
#reflected copy has the same physical boundary orientation as the original.
#Double reflection and rotations preserve orientation and are not reversed.
function apply_symmetries_to_boundary_points(pts::Vector{BoundaryPointsCFIE{T}},symmetries::Union{AbsSymmetry,Nothing},billiard::Bi;same_direction::Bool=true) where {Bi<:AbsBilliard,T<:Real}
    isnothing(symmetries) && return pts
    full=copy(pts)
    sx=hasproperty(billiard,:x_axis) ? T(getproperty(billiard,:x_axis)) : zero(T)
    sy=hasproperty(billiard,:y_axis) ? T(getproperty(billiard,:y_axis)) : zero(T)
    sym=symmetries
    new_parts=BoundaryPointsCFIE{T}[]
    if sym isa Reflection
        if sym.axis===:y_axis || sym.axis===:origin
            for c in pts
                n=length(c.xy)
                rxy=Vector{SVector{2,T}}(undef,n)
                rt=Vector{SVector{2,T}}(undef,n)
                @inbounds for j in 1:n
                    p=c.xy[j]
                    t=c.tangent[j]
                    rxy[j]=SVector{2,T}(_x_reflect(p[1],sx),p[2])
                    tx,ty=_x_reflect_tangent(t[1],t[2])
                    rt[j]=same_direction ? SVector{2,T}(-tx,-ty) : SVector{2,T}(tx,ty)
                end
                xy2=same_direction ? reverse(rxy) : rxy
                t2=same_direction ? reverse(rt) : rt
                ds2=same_direction ? reverse(c.ds) : copy(c.ds)
                push!(new_parts,BoundaryPointsCFIE(xy2,t2,c.tangent_2,c.ts,c.ws,c.ws_der,ds2,c.compid,c.is_periodic))
            end
        end

        if sym.axis===:x_axis || sym.axis===:origin
            for c in pts
                n=length(c.xy)
                rxy=Vector{SVector{2,T}}(undef,n)
                rt=Vector{SVector{2,T}}(undef,n)
                @inbounds for j in 1:n
                    p=c.xy[j]
                    t=c.tangent[j]
                    rxy[j]=SVector{2,T}(p[1],_y_reflect(p[2],sy))
                    tx,ty=_y_reflect_tangent(t[1],t[2])
                    rt[j]=same_direction ? SVector{2,T}(-tx,-ty) : SVector{2,T}(tx,ty)
                end
                xy2=same_direction ? reverse(rxy) : rxy
                t2=same_direction ? reverse(rt) : rt
                ds2=same_direction ? reverse(c.ds) : copy(c.ds)
                push!(new_parts,BoundaryPointsCFIE(xy2,t2,c.tangent_2,c.ts,c.ws,c.ws_der,ds2,c.compid,c.is_periodic))
            end
        end
        if sym.axis===:origin
            for c in pts
                n=length(c.xy)
                rxy=Vector{SVector{2,T}}(undef,n)
                rt=Vector{SVector{2,T}}(undef,n)
                @inbounds for j in 1:n
                    p=c.xy[j]
                    t=c.tangent[j]
                    rxy[j]=SVector{2,T}(_x_reflect(p[1],sx),_y_reflect(p[2],sy))
                    tx,ty=_xy_reflect_tangent(t[1],t[2])
                    rt[j]=SVector{2,T}(tx,ty)
                end
                push!(new_parts,BoundaryPointsCFIE(rxy,rt,c.tangent_2,c.ts,c.ws,c.ws_der,copy(c.ds),c.compid,c.is_periodic))
            end
        end
    elseif sym isa Rotation
        cx=T(sym.center[1])
        cy=T(sym.center[2])
        for l in 1:sym.n-1
            θ=T(2π*l/sym.n)
            cθ=cos(θ)
            sθ=sin(θ)
            for c in pts
                n=length(c.xy)
                rxy=Vector{SVector{2,T}}(undef,n)
                rt=Vector{SVector{2,T}}(undef,n)
                @inbounds for j in 1:n
                    p=c.xy[j]
                    t=c.tangent[j]
                    x=cθ*(p[1]-cx)-sθ*(p[2]-cy)+cx
                    y=sθ*(p[1]-cx)+cθ*(p[2]-cy)+cy
                    tx=cθ*t[1]-sθ*t[2]
                    ty=sθ*t[1]+cθ*t[2]
                    rxy[j]=SVector{2,T}(x,y)
                    rt[j]=SVector{2,T}(tx,ty)
                end
                push!(new_parts,BoundaryPointsCFIE(rxy,rt,c.tangent_2,c.ts,c.ws,c.ws_der,copy(c.ds),c.compid,c.is_periodic))
            end
        end
    else
        error("Unknown symmetry type $(typeof(sym))")
    end
    append!(full,new_parts)
    return full
end

# apply_symmetries_to_boundary_function
# Expand boundary data (function values) from fundamental domain → full domain.
#
# Inputs:
#   u           : values on flattened fundamental boundary
#   pts         : corresponding vector of BoundaryPointsCFIE (fundamental domain)
#   symmetries  : AbsSymmetry or nothing
#
# Output:
#   u_full      : expanded vector consistent with geometry expansion
#
# Notes:
#   - Mirrors apply_symmetries_to_boundary_points.
#   - Reflection reverses ordering; rotation multiplies by character χ.
#   - Promotes to Complex if required by rotational irreps.
function apply_symmetries_to_boundary_function(u::AbstractVector{U},pts::Vector{BoundaryPointsCFIE{T}},symmetries::Union{AbsSymmetry,Nothing}) where {U<:Number,T<:Real}
    symmetries===nothing && return u
    lengths=[length(c.xy) for c in pts]
    offs=Vector{Int}(undef,length(lengths)+1)
    offs[1]=1
    @inbounds for i in 1:length(lengths)
        offs[i+1]=offs[i]+lengths[i]
    end
    comps=[u[offs[i]:(offs[i+1]-1)] for i in eachindex(lengths)]
    has_complex=!isnothing(symmetries) && symmetries isa Rotation && mod(symmetries.m,symmetries.n)!=0
    S=(U<:Real && has_complex) ? Complex{T} : U
    compsS=[S.(c) for c in comps]
    full=copy(compsS)
    sym=symmetries
    new_parts=Vector{Vector{S}}()
    if sym isa Reflection
        if sym.axis===:y_axis || sym.axis===:origin
            p=S(sym.axis===:origin ? sym.parity[1] : sym.parity)
            for c in compsS
                push!(new_parts,p.*reverse(c))
            end
        end
        if sym.axis===:x_axis || sym.axis===:origin
            p=S(sym.axis===:origin ? sym.parity[2] : sym.parity)
            for c in compsS
                push!(new_parts,p.*reverse(c))
            end
        end
        if sym.axis===:origin
            pxy=S(sym.parity[1]*sym.parity[2])
            for c in compsS
                push!(new_parts,pxy.*c)
            end
        end
    elseif sym isa Rotation
        n=sym.n
        m=mod(sym.m,n)
        for l in 1:n-1
            χ = m==0 ? one(Complex{T}) :
                Complex{T}(cos(T(2π)*T(m*l)/T(n)),sin(T(2π)*T(m*l)/T(n)))
            for c in compsS
                push!(new_parts,S.(χ.*c))
            end
        end
    else
        error("Unknown symmetry type $(typeof(sym))")
    end
    append!(full,new_parts)
    return vcat(full...)
end