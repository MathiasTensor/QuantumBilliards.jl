###############################################################################
################## Symmetry mapping and projection utilities ##################
###############################################################################

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

##########################
#### CFIE_kress UTILS ####
##########################

function plot_boundary_with_weight_INFO(billiard::Bi,solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners};k=20.0,markersize=5) where {Bi<:AbsBilliard}
    pts_all=evaluate_points(solver,billiard,k)
    comps=_boundary_components(billiard.full_boundary)
    ncomp=length(pts_all)
    f=Figure(resolution=(1200,400+300*ncomp))
    ax=Axis(f[1,1],title="boundary + point-wise weights",aspect=DataAspect())
    sc=nothing
    for i in 1:ncomp
        pts=pts_all[i]
        xs=getindex.(pts.xy,1)
        ys=getindex.(pts.xy,2)
        ws_pts=pts.ws
        sc=scatter!(ax,xs,ys;markersize=markersize,color=ws_pts,colormap=:viridis,strokewidth=0)
        tx=getindex.(pts.tangent,1)
        ty=getindex.(pts.tangent,2)
        arrows!(ax,xs,ys,tx,ty;color=:black,lengthscale=0.08,linewidth=1)
    end
    hidespines!(ax,:t,:r)
    for j in 1:ncomp
        pts=pts_all[j]
        comp=comps[j]
        row=2+j
        a1=Axis(f[row,1],title="component $j: ws",xlabel=pts.is_periodic ? "parameter" : "u",ylabel="ws")
        a2=Axis(f[row+1,1],title="component $j: ws_der",xlabel=pts.is_periodic ? "parameter" : "u",ylabel="ws_der")
        ts=pts.ts
        ws=pts.ws
        ws_der=pts.ws_der
        scatter!(a1,ts,ws,markersize=7)
        scatter!(a2,ts,ws_der,markersize=7)
        if solver isa CFIE_kress
            h=length(ts)>1 ? ts[2]-ts[1] : 0.0
            lines!(a1,ts,fill(h,length(ts)),linewidth=2)
            lines!(a2,ts,fill(one(eltype(ts)),length(ts)),linewidth=2)
        elseif solver isa CFIE_kress_corners
            T=eltype(ts)
            qT=T(solver.kressq)
            tloc=collect(range(zero(T),T(2pi),length=800))
            h=T(pi/((length(ts)+1)÷2))
            wline=@. h*_kress_wprime(tloc,qT)
            wderline=@. _kress_wprime(tloc,qT)
            lines!(a1,tloc,wline,linewidth=2)
            lines!(a2,tloc,wderline,linewidth=2)
        elseif solver isa CFIE_kress_global_corners
            T=eltype(ts)
            tloc=collect(range(zero(T),T(2pi),length=800))
            h=T(pi/((length(ts)+1)÷2))
            corners=length(comp)==1 ? T[zero(T)] : _component_corner_locations(T,comp)
            _,_,wprime,wdoubleprime,_=multi_kress_graded_nodes_data(T,length(tloc)%2==1 ? length(tloc) : length(tloc)-1,corners;q=solver.kressq)
            lines!(a1,ts,ws,linewidth=2)
            lines!(a2,ts,ws_der,linewidth=2)
        end
        hidespines!(a1,:t,:r)
        hidespines!(a2,:t,:r)
    end
    return f
end