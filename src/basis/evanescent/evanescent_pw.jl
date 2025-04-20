using LinearAlgebra, CoordinateTransformations, Rotations, StaticArrays

#TODO Add the Evanescent PW basis. Make it so we can choose the pts we want to add them to (perhaps think of a way to automate this via the corners field in the billiard struct.)
#TODO This should primarily follow Barnett's definition of evanescent Plane Waves with evanescence parameters being an algebraic progression

"""
    max_i(k::Real) -> Int

Compute the maximum integer index `i` such that the evanescent decay parameter 
αᵢ = (3 + i) / (2k^(1/3)) does not exceed 3.
Further reading: https://users.flatironinstitute.org/~ahb/thesis_html/node157.html

This ensures that the evanescent plane wave parameter αᵢ remains within 
the recommended upper limit for numerical stability and efficiency.

# Arguments
- `k::Real`: The wavenumber used to define the decay rate of the evanescent plane wave.

# Returns
- `Int`: The maximum value of `i` such that αᵢ ≤ 3.
"""
max_i(k)=floor(Int,6*k^(1/3)-3)

function epw(pts::AbstractArray,idx::Int64,origin::SVector{2,T},normal::SVector{2,T},k::T) where {T<:Real}
    α=(3+idx)/(2*k^(1/3))  # Evanescence parameter
    θ=acos(-normal[1])          # Angle of propagation direction
    θ=normal[2]<0 ? -θ : θ    # Fix sign based on normal's y-component
    s,c=sincos(θ+im*α)     # Complex propagation direction
    x=getindex(pts,1).-origin[1]
    y=getindex(pts,2).-origin[2]
    return exp.(im.*(k*c.*x.+k*s.*y))
end

function epw(pts::AbstractArray,idx::Int64,origins::Vector{SVector{2,T}},normals::Vector{SVector{2,T}},k::T) where {T<:Real}
    α=(3+idx)/(2*k^(1/3))
    N=length(pts)
    M=length(origins)
    res=Matrix{Complex{T}}(undef,N,M) # pts x origins
    for i in eachindex(origins) 
        o,n=origins[i],normals[i]
        θ=acos(-n[1])  
        θ=n[2]<0 ? -θ : θ 
        s,c=sincos(θ+im*α) 
        x=getindex(pts,1).-o[1]
        y=getindex(pts,2).-o[2]
        res[:,i]=exp.(im.*(k*c.*x.+k*s.*y))
    end
    return sum(res,dims=2)[:] # for each row sum over all columns to get for each pt in pts all the different origin contributions
end

struct EvanescentPlaneWaves{T,Sy} <: AbsBasis where  {T<:Real,Sy<:Union{AbsSymmetry,Nothing}}
    dim::Int64 
    origins::Vector{SVector{2,T}}
    normals::Vector{SVector{2,T}}
    symmetries::Union{Vector{Any},Nothing}
end

function EvanescentPlaneWaves(billiard::Bi;fundamental=false) where {Bi<:AbsBilliard}
    origins,normals=get_origins_and_normals_(billiard;fundamental=fundamental)
    return EvanescentPlaneWaves(10,origins,normals,symmetries)
end

function EvanescentPlaneWaves(billiard::Bi,idxs::AbstractArray;fundamental=false) where {Bi<:AbsBilliard}
    origins,normals=get_origins_and_normals_(billiard,idxs;fundamental=fundamental)
    return EvanescentPlaneWaves(10,origins,normals,symmetries)
end

function EvanescentPlaneWaves(billiard::Bi,idx::Ti,which_pt::Union{Symbol,Tuple{Symbol,Symbol}}=(:start,:end);fundamental=false) where {Bi<:AbsBilliard,Ti<:Integer}
    origins,normals=get_origins_and_normals_(billiard,idx,which_pt;fundamental=fundamental)
    return EvanescentPlaneWaves(10,origins,normals,symmetries)
end

function get_origins_and_normals_(billiard::Bi,idx::Ti,which_pt::Union{Symbol,Tuple{Symbol,Symbol}}=(:start,:end);fundamental=false) where {Bi<:AbsBilliard,Ti<:Integer}
    boundary= fundamental ? billiard.fundamental_boundary : billiard.full_boundary
    elt=eltype(boundary[1].length)
    N=length(boundary)
    origins=Vector{SVector{2,elt}}()
    normals=Vector{SVector{2,elt}}()
    crv=boundary[idx]
    # Normalize to tuple form
    pts=isa(which_pt, Symbol) ? (which_pt,) : which_pt
    if :start in pts
        if crv isa AbsRealCurve && boundary[mod1(idx-1,N)] isa LineSegment
            push!(origins,curve(crv,zero(elt)))
            push!(normals,normal_vec(crv,zero(elt)))
        end
    end
    if :end in pts
        if crv isa AbsRealCurve && boundary[mod1(idx+1,N)] isa LineSegment
            push!(origins,curve(crv,one(elt)))
            push!(normals,normal_vec(crv,one(elt)))
        end
    end
    return origins,normals
end

function get_origins_and_normals_(billiard::Bi,idxs::AbstractArray;fundamental=false) where {Bi<:AbsBilliard}
    boundary= fundamental ? billiard.fundamental_boundary : billiard.full_boundary
    elt=eltype(boundary[1].length)
    N=length(boundary)
    @assert length(idxs)<=N "The number of idxs cannot be larger than the number of boundary segments. Check if fundamental kwarg is set correctly!"
    origins=Vector{SVector{2,elt}}()
    normals=Vector{SVector{2,elt}}()
    for idx in idxs 
        crv=boundary[idx]
        if typeof(crv) isa AbsRealCurve && typeof(boundary[mod1(idx-1,N)]) isa LineSegment # is the other curve is virtual then usually the BCs are already satisfied there so no need to add. Also only add the corners so must be a line segment adjacent to produce a corner.
            push!(origins,curve(crv,zero(elt))) # starting corner 
            push!(normals,normal_vec(crv,zero(elt))) # starting normal
        end
        if typeof(crv) isa AbsRealCurve && typeof(boundary[mod1(idx+1,N)]) isa LineSegment
            push!(origins,curve(crv,one(elt))) # ending corner
            push!(normals,normal_vec(crv,one(elt))) # ending normal
        end
    end
    return origins,normals
end

function get_origins_and_normals_(billiard::Bi;fundamental=false) where {Bi<:AbsBilliard}
    boundary=fundamental ? billiard.fundamental_boundary : billiard.full_boundary
    return get_origins_and_normals(billiard,eachindex(boundary);fundamental=fundamental)
end

toFloat32(basis::EvanescentPlaneWaves)=EvanescentPlaneWaves(basis.dim,Float32.(basis.origin),basis.symmetries)

function resize_basis(basis::EvanescentPlaneWaves,billiard::Bi,dim::Int,k) where {Bi<:AbsBilliard}
    new_dim=max_i(k)
    if new_dim==basis.dim
        return basis
    else
        return EvanescentPlaneWaves(new_dim,basis.origin,basis.symmetries)
    end
end

@inline function basis_fun(basis::EvanescentPlaneWaves{T},i::Int,k::T,pts::AbstractArray) where {T<:Real}
    return epw(pts,i,basis.origins,basis.normals,k)
end

@inline function basis_fun(basis::EvanescentPlaneWaves{T},indices::AbstractArray,k::T,pts::AbstractArray;multithreaded::Bool=true) where {T<:Real}
    #TODO
end