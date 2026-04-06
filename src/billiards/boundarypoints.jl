using StaticArrays


"""
    struct BoundaryPoints{T} <: AbsPoints

Stores discretized boundary information for a billiard curve:
- `xy::Vector{SVector{2,T}}`: Coordinates of boundary points in ℝ².
- `normal::Vector{SVector{2,T}}`: Outward unit normals at each boundary point.
- `s::Vector{T}`: Arc‐length parameters corresponding to each `xy` point.
- `ds::Vector{T}`: Integration weights (segment lengths) for each point.
- `w::Vector{T}`: Method‐specific weights (e.g. for SM: `ds ./ rn`, for DM: `ds`, for BIM: can be empty).
- `w_n::Vector{T}`: DM‐specific weights (e.g. `ds ./ rn`).
- `curvature::Vector{T}`: Curvature values at boundary points (used in BIM/EBIM/Beyn).
- `xy_int::Vector{SVector{2,T}}`: Interior points used in PSM.
- `shift_x::T`: Shift in x for symmetry operations in BIM/EBIM/Beyn.
- `shift_y::T`: Shift in y for symmetry operations in BIM/EBIM/Beyn.
"""
struct BoundaryPoints{T} <: AbsPoints where {T<:Real}
    xy::Vector{SVector{2,T}}
    normal::Vector{SVector{2,T}} #normal vectors in points
    s::Vector{T} # arc length coords
    ds::Vector{T} # integration weights
    w::Vector{T} # method-specific weights (SM: ds./rn, DM: ds, BIM: can leave empty)
    w_n::Vector{T} # DM specific weights
    curvature::Vector{T} # BIM/EBIM/Beyn specific curvature values at the boundary points
    xy_int::Vector{SVector{2,T}} # used in PSM for the interior points
    shift_x::T # used in BIM/EBIM/Beyn to get the correct reflections/rotations wrt new symmetry axes
    shift_y::T # used in BIM/EBIM/Beyn to get the correct reflections/rotations wrt new symmetry axes
end

"""
    boundary_coords(crv::C, sampler::S, N::Int) :: Tuple{Vector{SVector{2,T}}, Vector{SVector{2,T}}, Vector{T}, Vector{T}} 
    where {C<:AbsCurve, S<:AbsSampler, T<:Real}

# Arguments
- `crv::C<:AbsCurve`: Any curve segment (line, circle, polar, etc.).
- `sampler::S<:AbsSampler`: Sampler object that provides parameter samples `t` and weights `dt`.
- `N::Int`: Number of sample points.

# Returns
- `xy::Vector{SVector{2,T}}`: Coordinates on the curve.
- `normal::Vector{SVector{2,T}}`: Outward normals at those points.
- `s::Vector{T}`: Arc‐length values.
- `ds::Vector{T}`: Integration weights.

where `T` is the element type of `crv.length`.
"""
function boundary_coords(crv::C,sampler::S,N) where {C<:AbsCurve, S<:AbsSampler}
    L=crv.length
    t,dt=sample_points(sampler, N)
    xy=curve(crv,t)
    normal=normal_vec(crv,t)
    s=arc_length(crv,t)
    if crv isa PolarSegment
        ds=diff(s)
        append!(ds,L+s[1]-s[end]) # add the last difference as we have 1 less element. Add L to s[1] so we can logically subtract s[end]
    else
        ds=L.*dt
    end
    return xy,normal,s,ds
end

"""
    boundary_coords(crv::C, t::AbstractVector{T}, dt::AbstractVector{T}) :: Tuple{Vector{SVector{2,T}}, Vector{SVector{2,T}}, Vector{T}, Vector{T}}
    where {C<:AbsCurve, T<:Real}

# Arguments
- `crv::C<:AbsCurve`: A single curve segment.
- `t::AbstractVector{T}`: Parameter values in `[0,1]`.
- `dt::AbstractVector{T}`: Corresponding parameter increments.

# Returns
- `xy::Vector{SVector{2,T}}`: Coordinates.
- `normal::Vector{SVector{2,T}}`: Normals.
- `s::Vector{T}`: Arc‐length values.
- `ds::Vector{T}`: Integration weights.
"""
function boundary_coords(crv::C,t,dt) where {C<:AbsCurve}
    L=crv.length
    xy=curve(crv,t)
    normal=normal_vec(crv,t)
    s=arc_length(crv,t)
    if crv isa PolarSegment
        ds=diff(s)
        append!(ds,L+s[1]-s[end]) # add the last difference as we have 1 less element. Add L to s[1] so we can logically subtract s[end]
    else
        ds=L.*dt
    end
    return xy,normal,s,ds
end 

"""
    boundary_coords(billiard::Bi, sampler::FourierNodes, N::Int) :: BoundaryPoints{T} where {Bi<:AbsBilliard, T<:Real}

 # IMPORTANT: First curve must be real in the f`billiard.full_boundary`

Construct boundary points for each real segment of `billiard.full_boundary` by sampling with `FourierNodes`.
Only real (non‐virtual) segments are included. Returns a `BoundaryPoints` container combining:
- `xy_all`: concatenated coordinates,
- `normal_all`: concatenated normals,
- `s_all`: concatenated arc‐lengths (shifted by cumulative lengths),
- `ds_all`: concatenated weight segments.

# Arguments
- `billiard::Bi<:AbsBilliard`: A billiard object containing `full_boundary::Vector{AbsCurve}`.
- `sampler::FourierNodes`: Sampler for Fourier‐type nodes on each segment.
- `N::Int`: Total number of sample points distributed proportionally to each segment’s length.

# Returns
- `BoundaryPoints{T}`: T is the element type of the boundary curves’ lengths.
"""
function boundary_coords(billiard::Bi,sampler::FourierNodes,N) where {Bi<:AbsBilliard}
    let boundary=billiard.full_boundary
        ts,dts=sample_points(sampler,N)
        xy_all,normal_all,s_all,ds_all=boundary_coords(boundary[1],ts[1],dts[1])
        l=boundary[1].length
        for i in 2:length(ts)
            crv=boundary[i]
            if (typeof(crv) <: AbsRealCurve)
                Lc=crv.length
                xy,nxy,s,ds=boundary_coords(crv,ts[i],dts[i])
                append!(xy_all,xy)
                append!(normal_all,nxy)
                s=s.+l
                append!(s_all,s)
                append!(ds_all,ds)
                l+=Lc
            end    
        end
        return BoundaryPoints(xy_all,normal_all,s_all,ds_all,Vector{T}(),Vector{T}(),Vector{T}(),Vector{SVector{2,T}}(),zero(T),zero(T)) 
    end
end

"""
    boundary_coords(billiard::Bi, sampler::S, N::Int) :: BoundaryPoints{T} where {Bi<:AbsBilliard, S<:AbsSampler, T<:Real}

 # IMPORTANT: First curve must be real in the `billiard.full_boundary`

Sample the entire `billiard.full_boundary` using a generic `sampler`. Each real curve is sampled
with `Nc = round(Int, N * (crv.length / total_length))` points. Virtual curves are skipped.

# Arguments
- `billiard::Bi<:AbsBilliard`: Billiard containing `full_boundary`.
- `sampler::S<:AbsSampler`: Sampler object (e.g. uniform, Gauss‐Legendre).
- `N::Int`: Total number of points (distributed proportionally).

# Returns
- `BoundaryPoints{T}`: Combined discretization of all real boundary segments.
"""
function boundary_coords(billiard::Bi,sampler::S,N) where {Bi<:AbsBilliard,S<:AbsSampler}
    let boundary=billiard.full_boundary
            L=billiard.length
            Lc=boundary[1].length
            Nc=round(Int,N*Lc/L)
            xy_all,normal_all,s_all,ds_all=boundary_coords(boundary[1],sampler,Nc)
            l=boundary[1].length #cumulative length
            for crv in boundary[2:end]
                if (typeof(crv) <: AbsRealCurve)
                    Lc=crv.length
                    Nc=round(Int,N*Lc/L)
                    xy,nxy,s,ds=boundary_coords(crv,sampler,Nc)
                    append!(xy_all,xy)
                    append!(normal_all,nxy)
                    s=s.+l
                    append!(s_all,s)
                    append!(ds_all,ds)
                    l+=Lc
                end    
            end
        return BoundaryPoints(xy_all,normal_all,s_all,ds_all,Vector{T}(),Vector{T}(),Vector{T}(),Vector{SVector{2,T}}(),zero(T),zero(T)) 
    end
end

"""
    boundary_coords_desymmetrized_full_boundary(billiard::Bi, sampler::FourierNodes, N) where {Bi<:AbsBilliard}

#INTERNAL
Specialized boundary points construction function used exclusively for the boundary_function where we need to take into account the desymmetrized_full_boundary. Not used anywhere else.

# Arguments
- `billiard::Bi`: a billiard geometry
- `sampler::FourierNodes`: a sampler for Fourier nodes. Must be this one for consistency with boundary_function.
- `N`: the number of points to sample

# Returns
- `BoundaryPoints`: a struct containing the boundary points, normals, arc lengths, and integration weights for the boundary_function.
"""
function boundary_coords_desymmetrized_full_boundary(billiard::Bi,sampler::FourierNodes,N) where {Bi<:AbsBilliard}
    let boundary=billiard.desymmetrized_full_boundary
        ts,dts=sample_points(sampler,N)
        xy_all,normal_all,s_all,ds_all=boundary_coords(boundary[1],ts[1],dts[1])
        l=boundary[1].length
        for i in 2:length(ts)
            crv=boundary[i]
            if (typeof(crv) <: AbsRealCurve)
                Lc=crv.length
                xy,nxy,s,ds=boundary_coords(crv,ts[i],dts[i])
                append!(xy_all,xy)
                append!(normal_all,nxy)
                s=s.+l
                append!(s_all,s)
                append!(ds_all,ds)
                l+=Lc
            end    
        end
        return BoundaryPoints(xy_all,normal_all,s_all,ds_all,Vector{T}(),Vector{T}(),Vector{T}(),Vector{SVector{2,T}}(),zero(T),zero(T)) 
    end
end