
using CoordinateTransformations
using StaticArrays

reflect_x=CoordinateTransformations.LinearMap(SMatrix{2,2}([-1.0 0.0;0.0 1.0]))
reflect_y=CoordinateTransformations.LinearMap(SMatrix{2,2}([1.0 0.0;0.0 -1.0]))

"""
    Reflection

Represents a geometric reflection symmetry.

# Fields
- `sym_map::LinearMap{SMatrix{2,2,Float64,4}}`: The linear map defining the reflection.
- `parity::Union{Int64,Vector{Int64}}`: The parity of the wavefunction under reflection -> is vector for xy - reflection.
- `axis::Symbol`: The axis of reflection (`:x_axis`, `:y_axis`, or `:origin`).
"""
struct Reflection<:AbsSymmetry
    sym_map::CoordinateTransformations.LinearMap{SMatrix{2,2,Float64,4}}
    parity::Union{Int64,Vector{Int64}}
    axis::Symbol
end

# X,Y Reflections over potentially shifted axis (depeneding of the shift_x/shift_y of the billiard geometry), where only the coordiante reflection is dependant on the shifts of the reflection axes since the normals are just direction vectors
@inline _x_reflect(x::T,sx::T) where {T<:Real}=(2*sx-x)
@inline _y_reflect(y::T,sy::T) where {T<:Real}=(2*sy-y)
@inline _x_reflect_normal(nx::T,ny::T) where {T<:Real}=(-nx,ny)
@inline _y_reflect_normal(nx::T,ny::T) where {T<:Real}=(nx,-ny)
@inline _xy_reflect_normal(nx::T,ny::T) where {T<:Real}=(-nx,-ny)
@inline _x_reflect_tangent(tx::T,ty::T) where {T<:Real}=(-tx,ty)
@inline _y_reflect_tangent(tx::T,ty::T) where {T<:Real}=(tx,-ty)
@inline _xy_reflect_tangent(tx::T,ty::T) where {T<:Real}=(-tx,-ty)

"""
    Rotation

Cₙ rotation symmetry specification.

Fields
- n::Int                      # rotation order (e.g. 3,4,…)
- m::Int                      # irrep index in 0:(n-1)
- center::NTuple{2,Float64}   # rotation center (default (0,0))
"""
struct Rotation<:AbsSymmetry
    n::Int
    m::Int
    center::NTuple{2,Float64}
end

# use mod(m,n) so we can wrap m around in case needed
Rotation(n::Int,m::Int;center::Tuple{Real,Real}=(0.0,0.0))=Rotation(n,mod(m,n),(Float64(center[1]),Float64(center[2])))

# rotation of a point (x,y) by an angle α already wrapped in s,c=sincos(α) around a center of rotation (cx,cy). Added a separate K for type of center float.
@inline function _rot_point(x::T,y::T,cx::K,cy::K,c::T,s::T) where {T<:Real,K<:Real}
    Δx=x-cx;Δy=y-cy
    xr=cx+c*Δx-s*Δy
    yr=cy+s*Δx+c*Δy
    return xr,yr
end

# rotate (nx,ny) by angle with cos=c, sin=s
@inline _rot_vec(nx::T,ny::T,c::T,s::T) where {T<:Real}=(c*nx-s*ny,s*nx+c*ny)
@inline function _rotation_matrix(ang::T) where {T<:Real}
    s,c=sincos(ang)
    return SMatrix{2,2,T}((c,-s,s,c))
end

# tables for cos(lθ), sin(lθ), and characters χ_m(l)=e^{i2π ml/n}. Best place to do it here since it should not be called by user
@inline function _rotation_tables(::Type{T},n::Int,m::Int) where {T<:Real}
    θ=T(TWO_PI)/T(n)
    c1,s1=cos(θ),sin(θ)
    cos_tabulated=Vector{T}(undef,n);sin_tabulated=Vector{T}(undef,n)
    χ=Vector{Complex{T}}(undef,n)
    # for l=0 cos(0)=1 & sin(0)=0, therefore χ(l=0)=1.0
    cos_tabulated[1]=one(T)
    sin_tabulated[1]=zero(T)
    χ[1]=one(Complex{T})
    for l in 2:n
        cl,sl=cos_tabulated[l-1],sin_tabulated[l-1]
        cos_tabulated[l]=cl*c1-sl*s1
        sin_tabulated[l]=sl*c1+cl*s1
    end
    for l in 0:n-1
        χ[l+1]=cis(T(TWO_PI)*T(mod(m,n)*l)/T(n))
    end
    return cos_tabulated,sin_tabulated,χ
end

"""
    XReflection(parity::Int)

Creates a `Reflection` object that reflects over the `y`-axis.

# Arguments
- `parity::Int`: The wavefunction's parity under the reflection.

# Returns
- `Reflection`: A reflection object for the `y`-axis.

"""
function XReflection(parity)
    return Reflection(reflect_x,parity,:y_axis)
end

"""
    YReflection(parity::Int)

Creates a `Reflection` object that reflects over the `x`-axis.

# Arguments
- `parity::Int`: The wavefunction's parity under the reflection.

# Returns
- `Reflection`: A reflection object for the `x`-axis.
"""
function YReflection(parity)
    return Reflection(reflect_y,parity,:x_axis)
end

"""
    XYReflection(parity_x::Int, parity_y::Int)

Creates a `Reflection` object that reflects over both axes (origin symmetry).

# Arguments
- `parity_x::Int`: The wavefunction's parity under reflection over the `y`-axis.
- `parity_y::Int`: The wavefunction's parity under reflection over the `x`-axis.

# Returns
- `Reflection`: A reflection object for the origin symmetry.
"""
function XYReflection(parity_x,parity_y)
    return Reflection(reflect_x∘reflect_y,[parity_x,parity_y],:origin)
end

"""
    apply_symmetries_to_boundary_points(pts::BoundaryPoints{T},symmetries,billiard::Bi; same_direction::Bool=true) where {Bi<:AbsBilliard,T<:Real}

Extend a desymmetrized set of boundary points by applying reflections/rotations.

- Uses `_x_reflect/_y_reflect` and `_x_reflect_normal/_y_reflect_normal/_xy_reflect_normal`.
- Reflections are around `x=billiard.x_axis` / `y=billiard.y_axis` if present (0 otherwise).
- If `same_direction=true`, reflected copies are reversed (xy, normal, and ds) to maintain CCW orientation.
- Rotations are about `sym.center`; `ds` is copied; `s` is rebuilt via `cumsum(ds)` at the end.

Returns a new `BoundaryPoints{T}` on the full, symmetry-extended boundary.
"""
function apply_symmetries_to_boundary_points(pts::BoundaryPoints{T},symmetries,billiard::Bi;same_direction::Bool=true) where {Bi<:AbsBilliard,T<:Real}
    symmetries===nothing && return pts
    bxy=pts.xy
    bn=pts.normal
    bds=pts.ds
    full_xy=copy(bxy)
    full_normal=copy(bn)
    full_ds=copy(bds)
    copies=1+(
        isnothing(symmetries) ? 0 :
        symmetries isa Reflection ? (symmetries.axis===:origin ? 3 : 1) :
        symmetries isa Rotation   ? (symmetries.n-1) :
        0
    )
    sizehint!(full_xy,length(bxy)*copies)
    sizehint!(full_normal,length(bn)*copies)
    sizehint!(full_ds,length(bds)*copies)
    sx=hasproperty(billiard,:x_axis) ? T(billiard.x_axis) : zero(T)
    sy=hasproperty(billiard,:y_axis) ? T(billiard.y_axis) : zero(T)

    @inline function push_reflection!(which::Symbol)
        if which===:x
            rxy=[SVector(_x_reflect(p[1],sx),p[2]) for p in bxy]
            rn=[_x_reflect_normal(nv[1],nv[2]) for nv in bn]
        elseif which === :y
            rxy=[SVector(p[1], _y_reflect(p[2],sy)) for p in bxy]
            rn=[_y_reflect_normal(nv[1], nv[2]) for nv in bn]
        elseif which === :xy
            rxy=[SVector(_x_reflect(p[1],sx),_y_reflect(p[2], sy)) for p in bxy]
            rn=[_xy_reflect_normal(nv[1],nv[2]) for nv in bn]
        else
            error("unknown reflection kind $which")
        end
        do_reverse=same_direction && (which!=:xy)
        rds=do_reverse ? reverse(bds) : bds
        rxy=do_reverse ? reverse(rxy) : rxy
        rn=do_reverse ? reverse(rn)  : rn
        append!(full_xy,rxy)
        append!(full_normal,rn)
        append!(full_ds,rds)
        return nothing
    end
    s=symmetries
    if s isa Reflection
        if s.axis===:y_axis
            push_reflection!(:x)
        elseif s.axis===:x_axis
            push_reflection!(:y)
        elseif s.axis===:origin
            push_reflection!(:x)
            push_reflection!(:y)
            push_reflection!(:xy)
        else
            error("Unknown reflection axis $(s.axis)")
        end
    elseif s isa Rotation
        n=s.n;cx,cy=s.center
        Cx=T(cx);Cy=T(cy);θ=T(2π)/T(n)
        for l in 1:n-1
            cl=cos(T(l)*θ);sl=sin(T(l)*θ)
            rxy=[SVector(cl*(p[1]-Cx)-sl*(p[2]-Cy)+Cx,sl*(p[1]-Cx)+cl*(p[2]-Cy)+Cy) for p in bxy]
            rn =[SVector(cl*nv[1]-sl*nv[2],sl*nv[1]+cl*nv[2]) for nv in bn]
            append!(full_xy,rxy)
            append!(full_normal,rn)
            append!(full_ds,bds)
        end
    else
        error("Unknown symmetry type: $(typeof(s))")
    end
    full_s=cumsum(full_ds)
    empty_w=T[]
    empty_wn=T[]
    empty_curv=T[]
    empty_xyint=SVector{2,T}[]
    return BoundaryPoints{T}(full_xy,full_normal,full_s,full_ds,empty_w,empty_wn,empty_curv,empty_xyint,pts.shift_x,pts.shift_y)
end

"""
    apply_symmetries_to_boundary_function(u::AbstractVector{U},symmetries) where {U<:Number}

Symmetrize the desymmetrized boundary function `u(s)` for the full boundary.
Works with real or complex `u`. If any rotation has `m % n ≠ 0`, a complex
character phase χ_l is applied and the result is `Vector{Complex{T}}`
(with `T = real(eltype(u))`); otherwise the element type is preserved.

Reflection rules:
- `:y_axis` or `:x_axis`: append `parity * reverse(u)`.
- `:origin`: append `parity[1] * reverse(u)` (vertical), then
  `parity[2] * reverse([u; that_vertical_reflection])` (horizontal of the combined).

Rotation rules:
- For `l=1..n-1`, append `χ_l * u`, where `χ_l = exp(i 2π m l / n)`.
  If `m % n == 0`, χ_l=1 and no complex promotion occurs.

Returns the concatenated full-boundary function.
"""
function apply_symmetries_to_boundary_function(u::AbstractVector{U},symmetries) where {U<:Number}
    symmetries===nothing && return u
    T=U<:Real ? U : eltype(real(zero(U)))
    has_complex=!isnothing(symmetries) && symmetries isa Rotation && mod(symmetries.m,symmetries.n)!=0
    S=(U<:Real && has_complex) ? Complex{T} : U
    full_u=S.(u)
    base_u=copy(full_u) # not alias for rotations
    sym=symmetries
    if sym isa Reflection
        if sym.axis==:y_axis
        p=S(sym.parity)
        append!(full_u,p.*reverse(base_u))     # matches :x block in points
    elseif sym.axis==:x_axis
        p=S(sym.parity)
        append!(full_u,p.*reverse(base_u))     # matches :y block in points
    elseif sym.axis==:origin
        pY=S(sym.parity[1])  # parity for y-axis reflection (vertical)
        pX=S(sym.parity[2])  # parity for x-axis reflection (horizontal)
        uY=pY.*reverse(base_u)               #  :x block in points (y-axis reflection)
        uX=pX.*reverse(base_u)               #  :y block in points (x-axis reflection)
        uXY=(pX*pY).*base_u                   #  :xy block (no reverse!)
        append!(full_u,uY)
        append!(full_u,uX)
        append!(full_u,uXY)
    else
        error("Unknown reflection axis $(sym.axis)")
    end
    elseif sym isa Rotation
        n=sym.n; m=mod(sym.m,n)
        if m==0
            for l in 1:(n-1); append!(full_u,base_u); end # χ(m=0) = 1
        else
            for l in 1:(n-1)
                θ=T(2π)*T(m*l)/T(n)
                χ=Complex{T}(cos(θ),sin(θ))
                append!(full_u,χ.*base_u)
            end
        end
    else
        error("Unknown symmetry type: $(typeof(sym))")
    end
    return full_u
end

@inline function _endpoint_distance(a::SVector{2,T},b::SVector{2,T}) where {T<:Real}
    sqrt((a[1]-b[1])^2+(a[2]-b[2])^2)
end

@inline function _is_closed_curve(crv::C;xtol=1e-10) where {C<:AbsCurve}
    x0=curve(crv,[0.0])[1]
    x1=curve(crv,[1.0])[1]
    _endpoint_distance(x0,x1)<=xtol
end

@inline _all_closed_curves(boundary)=all(_is_closed_curve,boundary)

@inline function _is_single_composite_boundary(boundary)
    !(boundary[1] isa AbstractVector)&&!_all_closed_curves(boundary)
end

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

# used for DLP kress where outer boundary can be piecewise or periodic, but no holes -> no vector of BoundaryPointsCFIE, just one BoundaryPointsCFIE with all points in the xy field
function flatten_points(pts::BoundaryPointsCFIE{T}) where {T<:Real}
    N=length(pts.xy)
    xy=Vector{SVector{2,T}}(undef,N)
    @inbounds for i in 1:N
        xy[i]=pts.xy[i]
    end
    return xy,[1,N+1]
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

function build_symmetry_maps(pts::BoundaryPointsCFIE{T},sym;tol::T=T(1e-10)) where {T<:Real}
    build_symmetry_maps([pts],sym;tol=tol)
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

function apply_symmetries_to_boundary_points(pts::BoundaryPointsCFIE{T},symmetries::Union{AbsSymmetry,Nothing},billiard::Bi;same_direction::Bool=true) where {Bi<:AbsBilliard,T<:Real}
    apply_symmetries_to_boundary_points([pts],symmetries,billiard;same_direction=same_direction)
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

function apply_symmetries_to_boundary_function(u::AbstractVector{U},pts::BoundaryPointsCFIE{T},symmetries::Union{AbsSymmetry,Nothing}) where {U<:Number,T<:Real}
    apply_symmetries_to_boundary_function(u,[pts],symmetries)
end

"""
    reflect_wavefunction(Psi::Matrix, x_grid::Vector, y_grid::Vector, symmetries::Sy; x_axis=0.0, y_axis=0.0) -> (Psi::Matrix, x_grid::Vector, y_grid::Vector)

Applies reflection symmetries to a wavefunction and its grid.

# Arguments
- `Psi::Matrix`: The wavefunction defined on the `x_grid` and `y_grid`.
- `x_grid::Vector`: The x-coordinates of the grid points.
- `y_grid::Vector`: The y-coordinates of the grid points.
- `symmetries::Vector{Any}`: A vector of `Reflection` objects to apply. Should contain only 1 element.
- `x_axis=0.0`: The x-coordinate of the reflection axis. Default is 0.0.
- `y_axis=0.0`: The y-coordinate of the reflection axis. Default is 0.0.

# Returns
- `Psi::Matrix`: The modified wavefunction after applying the reflections.
- `x_grid::Vector`: The updated x-coordinates of the grid points.
- `y_grid::Vector`: The updated y-coordinates of the grid points.
"""
function reflect_wavefunction(Psi,x_grid,y_grid,symmetries;x_axis=0.0,y_axis=0.0)
    x_grid=x_grid.-x_axis  # Shift the grid to move the reflection axis to x=0
    y_grid=y_grid.-y_axis  # Shift the grid to move the reflection axis to y=0
    sym=symmetries
    if sym.axis==:y_axis
        x= -reverse(x_grid)
        Psi_ref=reverse(sym.parity.*Psi;dims=1)

        Psi=vcat(Psi,Psi_ref)
        x_grid=append!(x,x_grid)
        sorted_indices=sortperm(x_grid)
        x_grid=x_grid[sorted_indices]
    end
    if sym.axis==:x_axis
        y= -reverse(y_grid)
        Psi_ref=reverse(sym.parity.*Psi;dims=2)
        Psi=hcat(Psi_ref,Psi) 
        y_grid=append!(y,y_grid)
        sorted_indices=sortperm(y_grid)
        y_grid=y_grid[sorted_indices]
    end
    if sym.axis==:origin
        # Reflect over both axes (x -> -x, y -> -y)
        # First, reflect over y-axis
        x_reflected= -reverse(x_grid)
        Psi_y_reflected=reverse(sym.parity[1].*Psi;dims=2)
        Psi_y_combined=[Psi_y_reflected Psi]
        x_grid_combined=[x_reflected; x_grid]
        
        # Then, reflect over x-axis
        y_reflected= -reverse(y_grid)
        Psi_x_reflected=reverse(sym.parity[2].*Psi_y_combined;dims=1)
        Psi_x_combined=[Psi_x_reflected; Psi_y_combined]
        y_grid_combined=[y_reflected; y_grid]

        # Permute the indexes
        sorted_indices=sortperm(x_grid_combined)
        x_grid_combined=x_grid_combined[sorted_indices]
        sorted_indices=sortperm(y_grid_combined)
        y_grid_combined=y_grid_combined[sorted_indices]
        
        # Update Psi and grids
        Psi=Psi_x_combined
        x_grid=x_grid_combined
        y_grid=y_grid_combined
    end
    # Shift the grids back to their original positions before returning
    x_grid=x_grid.+x_axis
    y_grid=y_grid.+y_axis
    return Psi,x_grid,y_grid
end
