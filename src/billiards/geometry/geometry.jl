
#order of inclusions is important because linesegments need to be first
include("linesegment.jl")
include("circlesegment.jl")
include("dispersingcirclesegment.jl")
include("polarsegment.jl")
using StaticArrays,LinearAlgebra, ForwardDiff

"""
    tangent_vec(curve::L, ts::AbstractVector{T}) :: Vector{SVector{2,T}} where {T<:Real, L<:AbsCurve}

Compute the unit tangent vectors of an arbitrary curve at a collection of parameter values `ts`.

# Arguments
- `curve::L<:AbsCurve`: Any curve implementing `curve` and `tangent` methods.
- `ts::AbstractVector{T}`: Vector of parameter values (in `[0,1]`).

# Returns
- `Vector{SVector{2,T}}`: List of normalized tangent vectors `tangent(curve, t)/‖tangent(curve, t)‖` for each `t ∈ ts`.
"""
function tangent_vec(curve::L,ts::AbstractArray{T,1}) where {T<:Real,L<:AbsCurve}
    ta=tangent(curve,ts)
    return collect(ti/norm(ti) for ti in ta)
end

"""
    normal_vec(curve::L, ts::AbstractVector{T}) :: Vector{SVector{2,T}} where {T<:Real, L<:AbsCurve}

Compute the outward normal vectors of an arbitrary curve at parameter values `ts`, 
assuming a right‐handed orientation. Each normal is obtained by rotating the unit tangent 
vector by +90° (i.e. `(t_x, t_y) -> (t_y, -t_x)`).

# Arguments
- `curve::L<:AbsCurve`: Any curve type implementing `tangent_vec`.
- `ts::AbstractVector{T}`: Vector of parameter values.

# Returns
- `Vector{SVector{2,T}}`: List of normals at each `t ∈ ts`, given by `(tangent_vec(t)[2], -tangent_vec(t)[1])`.
"""
function normal_vec(curve::L,ts::AbstractArray{T,1}) where {T<:Real,L<:AbsCurve}
    ta=tangent_vec(curve,ts)
    return [SVector(ti[2],-ti[1]) for ti in ta]
end

"""
    curvature(curve::L, ts::AbstractVector{T}) :: Vector{T} where {T<:Real, L<:AbsCurve}

Compute the curvature κ(t) of a smooth curve at multiple parameter values `ts` using 
the formula:

    κ(t) = ( x′(t) y″(t) – y′(t) x″(t) ) / ( (x′(t)² + y′(t)²)^(3/2) )

# Arguments
- `curve::L<:AbsCurve`: Curve type implementing `curve()` and a `ForwardDiff`‐compatible derivative.
- `ts::AbstractVector{T}`: Parameter values at which to evaluate curvature.

# Returns
- `Vector{T}`: Curvature values κ(t) for each `t ∈ ts`.
"""
function curvature(crv::L,ts::AbstractArray{T,1}) where {T<:Real,L<:AbsCurve}
    let 
        r(t)=curve(crv,t)
        dr(t)=ForwardDiff.derivative(r,t)
        ddr(t)=ForwardDiff.derivative(dr,t)
        kappa=similar(ts)
        for i in eachindex(ts)
            der=dr(ts[i])
            der2=ddr(ts[i])
            norm=hypot(der[1],der[2])^3
            kap=der[1]*der2[2]-der[2]*der2[1]
            kappa[i]=kap/norm
        end
        return kappa
    end
end

"""
    curvature(curve::L, t::T) :: T where {T<:Real, L<:AbsCurve}

Compute the curvature κ(t) of a smooth curve at a single parameter value `t` using 
the same formula as the vector version.

# Arguments
- `curve::L<:AbsCurve`: Curve type implementing `curve()` and `ForwardDiff`‐compatible derivatives.
- `t::T`: Parameter value in `[0,1]`.

# Returns
- `T`: The curvature κ(t).
"""
function curvature(crv::L,t::T) where {T<:Real,L<:AbsCurve}
    let 
        r(t)=curve(crv,t)
        dr(t)=ForwardDiff.derivative(r,t)
        ddr(t)=ForwardDiff.derivative(dr,t)
        der=dr(t)
        der2=ddr(t)
        norm=hypot(der[1],der[2])^3
        kap=der[1]*der2[2]-der[2]*der2[1]
        return kap/norm
    end
end

"""
    make_polygon(corners::AbstractVector{SVector{2,T}}, curve_types::AbstractVector{Symbol}; origin::Tuple{T,T}=(0.0,0.0), rot_angle::T=0.0) :: Vector{C} where {T<:Real, C<:AbsCurve}

Construct a closed polygonal boundary by connecting consecutive vertices in `corners`. 
Each edge is either a real or virtual line segment, depending on `curve_types[i]` (`:Real` 
for `LineSegment`, otherwise `:VirtualLineSegment`). The last corner connects back to the first.

# Arguments
- `corners::Vector{SVector{2,T}}`: Sequence of 2D points (vertices) in local coordinates.
- `curve_types::Vector{Symbol}`: Vector of length `N` with entries `:Real` (real segment) or any other 
  symbol for a virtual segment.
- `origin::Tuple{T,T}`: Translation of the entire polygon (default `(0.0,0.0)`).
- `rot_angle::T`: Global rotation angle in radians (default `0.0`).

# Returns
- `Vector{C}`: List of `N` curve segments of type `LineSegments{T}` (either `LineSegment` or `VirtualLineSegment`).
"""
function make_polygon(corners,curve_types;origin=(zero(corners[1][1]),zero(corners[1][1])),rot_angle=zero(corners[1][1]))
    N=length(corners)
    boundary=[]
    circular_idx(i)=mod1(i,N)
    for i in 1:N
        idx0=circular_idx(i)
        c0=corners[idx0]
        c1=corners[circular_idx(i+1)]
        line= (curve_types[idx0]==:Real) ? LineSegment(c0,c1;origin=origin,rot_angle=rot_angle) : VirtualLineSegment(c0,c1;origin=origin,rot_angle=rot_angle)
        push!(boundary,line)
    end
    return boundary
end

"""
    symmetry_accounted_fundamental_boundary_length(fundamental_boundary::Vector{C}) :: T where {T<:Real, C<:AbsCurve}

Compute the net boundary length of a fundamental (desymmetrized) boundary, taking into account 
Dirichlet and Neumann segments. Dirichlet segments contribute +length, Neumann contribute −length, 
and real (unmarked) segments contribute +length.

# Arguments
- `fundamental_boundary::Vector{C<:AbsCurve}`: List of fundamental boundary curves, which may be 
  either real (`AbsRealCurve`) or virtual (`AbsVirtualCurve`). Virtual curves have a field 
  `symmetry_type::Symbol` which is `:Dirichlet` or `:Neumann`.

# Returns
- `T`: Sum of `length` fields, with sign depending on `symmetry_type`:
  - `AbsRealCurve` → `+crv.length`
  - `AbsVirtualCurve` with `symmetry_type == :Dirichlet` → `+crv.length`
  - `AbsVirtualCurve` with `symmetry_type == :Neumann` → `-crv.length`
"""
function symmetry_accounted_fundamental_boundary_length(fundamental_boundary::Vector{C}) where {C<:AbsCurve}
    L=0.0
    for crv in fundamental_boundary
        if crv isa AbsVirtualCurve
            if crv.symmetry_type==:Dirichlet
                L+=crv.length
            elseif crv.symmetry_type==:Neumann
                L-=crv.length
            end
        else
            L+=crv.length
        end
    end
    return L
end