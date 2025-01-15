
"""
    real_length(billiard::Bi) where Bi<:AbsBilliard

Calculates the total length of all `AbsRealCurve` objects in the fundamental boundary of a billiard.

# Arguments
- `billiard::Bi`: A billiard object of type `Bi <: AbsBilliard`.

# Returns
The total length of real curves (`AbsRealCurve`) in the fundamental boundary.
"""
function real_length(billiard::Bi) where Bi<:AbsBilliard
    L=0.0
    for curve in billiard.fundamental_boundary
        if typeof(curve) <: AbsRealCurve
            L+=curve.length
        end
    end
    return L 
end

"""
    virtual_length(billiard::Bi) where Bi<:AbsBilliard

Calculates the total length of all `AbsVirtualCurve` objects (virtual curves where the BC is due to basis automatically satisfied) in the fundamental boundary of a billiard.

# Arguments
- `billiard::Bi`: A billiard object of type `Bi <: AbsBilliard`.

# Returns
The total length of virtual curves (`AbsVirtualCurve`) in the fundamental boundary.
"""
function virtual_length(billiard::Bi) where Bi<:AbsBilliard
    L=0.0
    for curve in billiard.fundamental_boundary
        if typeof(curve) <: AbsVirtualCurve
            L+= curve.length
        end
    end
    return L 
end

"""
    curve_edge_lengths(billiard::Bi) where Bi<:AbsBilliard

Computes the cumulative lengths of all `AbsRealCurve` objects in the full boundary of a billiard.

# Arguments
- `billiard::Bi`: A billiard object of type `Bi <: AbsBilliard`.

# Returns
A vector of cumulative lengths of real curves (`AbsRealCurve`) in the full boundary.
"""
function curve_edge_lengths(billiard::Bi) where Bi<:AbsBilliard
    L=0.0
    res=[L]
    for crv in billiard.full_boundary
        if (typeof(crv) <: AbsRealCurve)
            L+=crv.length
            push!(res,L)
        end 
    end
    return res
end

"""
    is_inside(billiard::Bi, pt::SVector; fundamental_domain=true) where Bi<:AbsBilliard

Checks if a given point lies in the interior of a billiard.

# Arguments
- `billiard::Bi`: A billiard object of type `Bi <: AbsBilliard`.
- `pt::Vector{2,<:Real}`: A point to check.
- `fundamental_domain::Bool`: If `true`, checks the fundamental boundary; otherwise, checks the full boundary (default: `true`).

# Returns
`true` if the point lies inside the boundary; otherwise, `false`.
"""
function is_inside(billiard::Bi,pt;fundamental_domain=true) where Bi<:AbsBilliard
    if fundamental_domain 
        boundary=billiard.fundamental_boundary  
    else
        boundary=billiard.full_boundary
    end
    return all(is_inside(crv,pt) for crv in boundary) 
end

"""
    is_inside(billiard::Bi, pts::AbstractArray; fundamental_domain = true) where Bi<:AbsBilliard

Checks if an array of points lies inside the boundary of a billiard. It just iteratively call is_inside for pts.

# Arguments
- `billiard::Bi`: A billiard object of type `Bi <: AbsBilliard`.
- `pts::AbstractArray`: An array of points to check.
- `fundamental_domain::Bool`: If `true`, checks the fundamental boundary; otherwise, checks the full boundary (default: `true`).

# Returns
A boolean array indicating whether each point lies inside the boundary.
"""
function is_inside(billiard::Bi,pts::AbstractArray;fundamental_domain=true) where Bi<:AbsBilliard
    if fundamental_domain 
        curves=billiard.fundamental_boundary  
    else
        curves=billiard.full_boundary
    end
    inside=is_inside(curves[1],pts)
    for i in eachindex(curves)[2:end]
        inside=inside .& is_inside(curves[i],pts)
    end
    return inside
end

"""
    boundary_limits(curves::Vector{<:AbsCurve}; grd::Integer = 1000)

Calculates the x and y boundary limits (extrema) of a set of curves.

# Arguments
- `curvesVector{<:AbsCurve}`: A collection of curve objects.
- `grd::Integer`: Grid resolution used to sample points on the curves (default: `1000`).

# Returns
A tuple `(xlim, ylim)`:
- `xlim::Tuple{<:Real,<:Real}`: The x-coordinate extrema as a tuple `(x_min, x_max)`.
- `ylim::Tuple{<:Real,<:Real}`: The y-coordinate extrema as a tuple `(y_min, y_max)`.
"""
function boundary_limits(curves;grd=1000) 
    x_bnd=Vector{Any}()
    y_bnd=Vector{Any}()
    for crv in curves #names of variables not very nice
        L=crv.length
        N_bnd=max(512,round(Int,grd/L))
        t=range(0.0,1.0,N_bnd)[1:end-1]
        pts=curve(crv,t)
        append!(x_bnd,getindex.(pts,1))
        append!(y_bnd,getindex.(pts,2))
    end
    x_bnd[end]=x_bnd[1]
    y_bnd[end]=y_bnd[1]
    xlim=extrema(x_bnd)
    ylim=extrema(y_bnd)
    return xlim,ylim
end

