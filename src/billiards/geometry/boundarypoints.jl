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
struct BoundaryPoints{T}<:AbsPoints where {T<:Real}
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

function boundary_matrix_size(pts::BoundaryPoints{T}) where {T<:Real}
    return length(pts.xy)
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
    boundary_coords_fourier(billiard::Bi,sampler::FourierNodes,N) where {Bi<:AbsBilliard}

Compute boundary coordinates for the billiard using Fourier sampling.
Used only for boundary_function when we comnstruct the gradient matrices via basis explicitely (VerginiSaraceno,ParticularSolutionsMethod,DecompositionMethod).

# Arguments
- `billiard::Bi<:AbsBilliard`: The billiard geometry.
- `sampler::FourierNodes`: Fourier sampling strategy.
- `N::Int`: Number of sample points.

# Returns
- `BoundaryPoints{T}`
"""
function boundary_coords_fourier(billiard::Bi,sampler::FourierNodes,N) where {Bi<:AbsBilliard}
    boundary=billiard.desymmetrized_full_boundary
    real_boundary=[crv for crv in boundary if crv isa AbsRealCurve]
    ts,dts=sample_points(sampler,N)
    xy_all,normal_all,s_all,ds_all=boundary_coords(real_boundary[1],ts[1],dts[1])
    l=real_boundary[1].length
    T=eltype(ds_all)
    for i in 2:length(real_boundary)
        crv=real_boundary[i]
        xy,nxy,s,ds=boundary_coords(crv,ts[i],dts[i])
        append!(xy_all,xy)
        append!(normal_all,nxy)
        append!(s_all,s .+ l)
        append!(ds_all,ds)
        l+=crv.length
    end
    return BoundaryPoints(xy_all,normal_all,s_all,ds_all,T[],T[],T[],SVector{2,T}[],zero(T),zero(T))
end

# helper to get he arclengths from the arclengths differences
@inline boundary_s(pts::BoundaryPoints{T}) where {T<:Real}=pts.s

#########################################################################
######################## BOUNDARY -> POLYGON HELPERS ####################
#########################################################################

"""
    _boundary_components_v2(boundary)

Normalize a boundary description into a vector of connected components,
where each component is a vector of curve pieces.

Accepted inputs:
- `[outer,hole1,hole2,...]` for multiple closed components
- `[seg1,seg2,...]` for one composite connected boundary
- `[[outer...],[hole1...],...]` for explicitly grouped components

Convention:
- component 1 = outer boundary
- components 2:end = holes
"""
function _boundary_components_v2(boundary)
    isempty(boundary) && return Vector{Vector{Any}}()
    if boundary[1] isa AbstractVector
        return [collect(comp) for comp in boundary]
    end
    all_closed=all(crv->begin
        p0=curve(crv,[0.0])[1]
        p1=curve(crv,[1.0])[1]
        hypot(p1[1]-p0[1],p1[2]-p0[2])<=1e-8
    end,boundary)

    if all_closed
        return [[crv] for crv in boundary]
    else
        return [collect(boundary)]
    end
end

"""
    billiard_boundary_components(billiard::Bi;fundamental_domain=true) where {Bi<:AbsBilliard}

Return the billiard boundary as `Vector{Vector}` of connected components.

If `fundamental_domain=true`, uses `billiard.fundamental_boundary` when present,
otherwise falls back to `billiard.full_boundary`.
"""
function billiard_boundary_components(billiard::Bi;fundamental_domain=true) where {Bi<:AbsBilliard}
    boundary=fundamental_domain && hasproperty(billiard,:fundamental_boundary) ? billiard.fundamental_boundary : billiard.full_boundary
    return _boundary_components_v2(boundary)
end

@inline function _component_length(comp)
    return sum(crv.length for crv in comp)
end

"""
    _sample_component_polygon(comp,N)

Sample one connected component into a polygon with about `N` points total,
distributed across its curve pieces proportionally to arclength.
"""
function _sample_component_polygon(comp,N::Int)
    npieces=length(comp)
    npieces==0 && return SVector{2,Float64}[]
    lengths=[crv.length for crv in comp]
    Ltot=sum(lengths)
    counts=Vector{Int}(undef,npieces)
    @inbounds for i in 1:npieces
        counts[i]=max(8,round(Int,N*lengths[i]/Ltot))
    end
    xy_parts=Vector{Vector}(undef,npieces)
    @inbounds for i in 1:npieces
        ts=sample_points(LinearNodes(),counts[i])[1]
        xy_parts[i]=curve(comp[i],ts)
    end
    return vcat(xy_parts...)
end

"""
    billiard_polygon_components(billiard::Bi,N_polygon_checks::Int;fundamental_domain=true) where {Bi<:AbsBilliard}

Sample each connected component of the billiard into one polygon.

Returns `Vector{Vector{SVector{2,T}}}` with convention:
- polygon_components[1] = outer boundary polygon
- polygon_components[2:end] = hole polygons
"""
function billiard_polygon_components(billiard::Bi,N_polygon_checks::Int;fundamental_domain=true) where {Bi<:AbsBilliard}
    comps=billiard_boundary_components(billiard;fundamental_domain=fundamental_domain)
    ncomp=length(comps)
    ncomp==0 && return Vector{Vector{SVector{2,Float64}}}()
    comp_lengths=[_component_length(comp) for comp in comps]
    Ltot=sum(comp_lengths)
    comp_points=Vector{Int}(undef,ncomp)
    @inbounds for i in 1:ncomp
        comp_points[i]=max(16,round(Int,N_polygon_checks*comp_lengths[i]/Ltot))
    end
    polys=Vector{Vector}(undef,ncomp)
    @inbounds for i in 1:ncomp
        polys[i]=_sample_component_polygon(comps[i],comp_points[i])
    end
    return polys
end

"""
    is_left(p1::SVector{2,T},p2::SVector{2,T},pt::SVector{2,T}) where {T<:Real}

Signed area test for whether `pt` lies to the left of the directed segment `p1 -> p2`.
"""
@inline function is_left(p1::SVector{2,T},p2::SVector{2,T},pt::SVector{2,T}) where {T<:Real}
    return (p2[1]-p1[1])*(pt[2]-p1[2])-(pt[1]-p1[1])*(p2[2]-p1[2])
end

"""
    is_point_in_polygon(polygon::Vector{SVector{2,T}},point::SVector{2,T})::Bool where {T<:Real}

Winding-number point-in-polygon test.
"""
function is_point_in_polygon(polygon::Vector{SVector{2,T}},point::SVector{2,T})::Bool where {T<:Real}
    winding_number=0
    n=length(polygon)
    @inbounds for i in 1:n
        p1=polygon[i]
        p2=polygon[(i%n)+1]
        if p1[2]<=point[2]
            if p2[2]>point[2] && is_left(p1,p2,point)>0
                winding_number+=1
            end
        else
            if p2[2]<=point[2] && is_left(p1,p2,point)<0
                winding_number-=1
            end
        end
    end
    return winding_number!=0
end

"""
    is_point_in_multiply_connected_polygon(components,point::SVector{2,T})::Bool where {T<:Real}

Test membership in a multiply connected domain.

Convention:
- `components[1]` is outer boundary
- `components[2:end]` are holes
"""
function is_point_in_multiply_connected_polygon(components,point::SVector{2,T})::Bool where {T<:Real}
    isempty(components) && return false
    is_point_in_polygon(components[1],point) || return false
    @inbounds for h in 2:length(components)
        if is_point_in_polygon(components[h],point)
            return false
        end
    end
    return true
end

"""
    points_in_billiard_polygon(pts::Vector{SVector{2,T}},billiard::Bi,N_polygon_checks::Int;fundamental_domain=true) where {T<:Real,Bi<:AbsBilliard}

Determine whether points lie inside the billiard domain by polygonizing each
connected boundary component.

Works for:
- single smooth closed boundaries
- composite outer boundaries
- outer boundary with holes
- composite holes
"""
function points_in_billiard_polygon(pts::Vector{SVector{2,T}},billiard::Bi,N_polygon_checks::Int;fundamental_domain=true) where {T<:Real,Bi<:AbsBilliard}
    polygon_components=billiard_polygon_components(billiard,N_polygon_checks;fundamental_domain=fundamental_domain)
    mask=fill(false,length(pts))
    Threads.@threads for i in eachindex(pts)
        mask[i]=is_point_in_multiply_connected_polygon(polygon_components,pts[i])
    end
    return mask
end

# Provides kress_R! to compute the circulant R matrix for the Kress method. kress_R! uses the FFT to compute the matrix efficiently, while kress_R! with ts computes it using a direct summation approach. Both functions modify the input matrix R0 in place.
# Ref: Kress, R., Boundary integral equations in time-harmonic acoustic scattering. Mathematics Comput. Modelling Vol 15, pp. 229-243). Pergamon Press, 1991, GB.
# Alex Barnett's code via ifft to get the circulant vector kernel and construct the circulant with circshift.
function kress_R_even!(R0::AbstractMatrix{T}) where {T<:Real}
    N=size(R0,1)
    n=N÷2 # integer division
    a=zeros(Complex{T},N) #  build the spectral vector a (first col)
    for m in 1:(n-1)
        a[m+1]=1/m     # positive freq
        a[N-m+1]=1/m     # negative freq
    end # leave a[n+1] == 0  (no 1/n term)
    rjn=real(FFTW.ifft(a)) # inverse FFT → rjn[j] = (2/N)*∑_{m=1..n-1} (1/m) cos(2π m (j-1)/N)
    ks=0:(N-1) # build the first column, adding the “alternating” correction
    alt=(-1).^ks # alt[j+1] = (-1)^j
    @. R0[:,1]=-two_pi*rjn-(2*two_pi/(N^2))*alt # R0[:,1] = -2π*rjn .- (4π/N^2)*alt, first col is ref
    for j in 2:N # fill out the rest circulantly:
        @views R0[:,j].=circshift(R0[:,j-1],1) # shift by +1 wrt previous column
    end
    return nothing
end

# This version of kress_R! computes the R matrix for the odd case (2n-1 points) where the Nyquist frequency is not included, so we only have m=1,...,n-1 positive and negative frequencies.
# The first column is built using the same FFT approach, but with the appropriate range of m. The rest of the matrix is filled circulantly as before.
function kress_R_odd!(R0::AbstractMatrix{T}) where {T<:Real}
    N=size(R0,1)
    n=(N-1)÷2
    a=zeros(Complex{T},N)   # spectral first column
    for m in 1:n
        a[m+1]=1/m   # positive freq
        a[N-m+1]=1/m   # negative freq
    end
    rjn=real(FFTW.ifft(a)) # gives (1/N) * sum_m a_m exp(2πimj/N)
    @. R0[:,1]= -two_pi*rjn
    for j in 2:N
        @views R0[:,j].=circshift(R0[:,j-1],1)
    end
    return nothing
end

"""
Provides kress_R! to compute the circulant R matrix for the Kress method.
Dispatches automatically to the even or odd periodic formula.

 Even N = 2n: includes the usual Nyquist correction term.

 Odd N = 2n+1: no Nyquist term appears, so we use the pure symmetric Fourier sum.

 Ref:
   Kress, R. (1991), Boundary integral equations in time-harmonic acoustic scattering.
   Barnett / Betcke MATLAB implementations of the periodic logarithmic quadrature idea.
"""
function kress_R!(R0::AbstractMatrix{T}) where {T<:Real}
    N=size(R0,1)
    if iseven(N)
        return kress_R_even!(R0)
    else
        return kress_R_odd!(R0)
    end
end

# Corner Kress, odd case:
function kress_R_corner_odd!(R0::AbstractMatrix{T}) where {T<:Real}
    N=size(R0,1)
    isodd(N) || error("kress_R_corner_odd! expects odd size.")
    kress_R_odd!(R0)
    return nothing
end

# Corner Kress, even case:
function kress_R_corner_even!(R0::AbstractMatrix{T}) where {T<:Real}
    N=size(R0,1)
    iseven(N) || error("kress_R_corner_even! expects even size 2n.")
    kress_R_even!(R0)
    return nothing
end

"""
    kress_R_corner!(R0::AbstractMatrix{T}) where {T<:Real}

Construct the Kress logarithmic correction matrix `R` for corner-graded
boundary discretizations, dispatching automatically based on the matrix size.

This function builds the circulant matrix associated with the periodic
logarithmic kernel used in Kress-type Nyström discretizations of boundary
integral operators with corner grading.

# Behavior
The construction depends on the parity of `N = size(R0,1)`:

- Odd size (`N = 2n-1`):
  Uses the classical restricted corner construction. The matrix is obtained
  by forming the full even periodic Kress matrix on `2n` nodes and restricting
  it to the interior nodes (removing one periodic endpoint).

- Even size (`N = 2n`):
  Uses a *full periodic graded construction*. The matrix is built directly on
  the full graded periodic node set without restriction. This is consistent
  with midpoint-shifted or endpoint-avoiding graded discretizations.

# Mathematical note
In both cases, `R` represents the discrete convolution operator corresponding
to the periodic logarithmic kernel

    log(4 sin^2((t - τ)/2)),

with the construction chosen to match the underlying node set:

- odd case: interior nodes of a periodic grid,
- even case: full periodic graded grid.

The two variants are consistent discretizations of the same continuous operator,
but differ at finite `N` due to the choice of node set.

# Arguments
- `R0::AbstractMatrix{T}`: Square matrix to be filled in-place with the Kress logarithmic correction.

# Returns
- `nothing` (matrix is modified in-place)
"""
function kress_R_corner!(R0::AbstractMatrix{T}) where {T<:Real}
    N=size(R0,1)
    if isodd(N)
        return kress_R_corner_odd!(R0)
    else
        return kress_R_corner_even!(R0)
    end
end

"""
    BoundaryPointsCFIE{T} <: AbsPoints

Boundary discretization container tailored for CFIE-type (Combined Field Integral Equation)
formulations on curves and panels.

# Fields
- `xy::Vector{SVector{2,T}}`:
  Cartesian coordinates of the discretization points on the boundary.

- `tangent::Vector{SVector{2,T}}`:
  First derivatives of the parametrization evaluated at the nodes.

- `tangent_2::Vector{SVector{2,T}}`:
  Second derivatives of the parametrization at the nodes.

- `ts::Vector{T}`:
  Parameter values associated with the discretization nodes. For periodic
  curves, these are typically distributed over `[0, 2π]`.

- `ws::Vector{T}`:
  Quadrature weights associated with the parameter nodes `ts`.

- `ws_der::Vector{T}`:
  Derivatives of the quadrature weights with respect to the parameter. These
  are required in formulations involving derivatives of the integral operator.

- `ds::Vector{T}`:
  Local arc-length increments corresponding to the discretization.

- `compid::Int`:
  Component index for multi-boundary geometries. The outer boundary should be
  indexed as `1`, and inner boundaries as `2, 3, ...`. This ordering is
  important for consistent orientation of tangents and derived normals.

- `is_periodic::Bool`:
  Indicates whether the boundary component is closed (`true`) or an open panel
  (`false`).

- `xL::SVector{2,T}`, `xR::SVector{2,T}`:
  Left and right endpoints of the boundary component. These are only relevant
  for non-periodic (panel) geometries.

- `tL::SVector{2,T}`, `tR::SVector{2,T}`:
  Tangent vectors at the left and right endpoints, respectively. Used for panel
  endpoint corrections and boundary treatments.
"""
struct BoundaryPointsCFIE{T}<:AbsPoints where {T<:Real}
    xy::Vector{SVector{2,T}} # the xy coords of the new mesh points
    tangent::Vector{SVector{2,T}} # tangents evaluated at the new mesh points
    tangent_2::Vector{SVector{2,T}} # derivatives of tangents evaluated at new mesh points
    ts::Vector{T} # parametrization that needs to go from [0,2π]
    ws::Vector{T} # the weights for the quadrature at ts
    ws_der::Vector{T} # the derivatives of the weights for the quadrature at ts
    ds::Vector{T} # diffs between crv lengths at ts
    compid::Int # index of the multi-domain, where the outer boundary is 1, the first inner boundary is 2,... It should be respected since otherwise the tangents/normals will be incorrectly oriented
    is_periodic::Bool # true = closed periodic curve, false = open panel
    xL::SVector{2,T} # left endpoint of the curve component, only used for panels
    xR::SVector{2,T} # right endpoint of the curve component, only used for panels
    tL::SVector{2,T} # tangent at left endpoint of the curve component, only used for panels
    tR::SVector{2,T} # tangent at right endpoint of the curve component, only used for panels
end

# helper to get the arclengths from the arclengths differences for multiple components
function boundary_s(pts::BoundaryPointsCFIE{T}) where {T<:Real}
    s=cumsum(pts.ds)
    s.-=s[1]
    return s
end

# helper to get the arclengths from the arclengths differences for multiple components
function boundary_s(pts::Vector{BoundaryPointsCFIE{T}}) where {T<:Real}
    isempty(pts) && return T[]
    s=T[]
    sizehint!(s,sum(length(p.ds) for p in pts))
    soff=zero(T)
    for p in pts
        local_s=cumsum(p.ds)
        local_s.-=local_s[1]
        local_s.+=soff
        append!(s,local_s)
        soff+=sum(p.ds)
    end
    return s
end

# For CFIE panel case this stores for eaach panel the relevant geometry (Alpert for now)
# so it does not need to be evaluated on the fly in a hoot loop 
struct CFIEPanelArrays{T<:Real}
    X::Vector{T}
    Y::Vector{T}
    dX::Vector{T}
    dY::Vector{T}
    s::Vector{T}
end

@inline function _panel_arrays_cache(pts::BoundaryPointsCFIE{T}) where {T<:Real}
    X=getindex.(pts.xy,1)
    Y=getindex.(pts.xy,2)
    dX=getindex.(pts.tangent,1)
    dY=getindex.(pts.tangent,2)
    s=@. sqrt(dX^2+dY^2)
    return CFIEPanelArrays(X,Y,dX,dY,s)
end

# For CFIE with holes, we compute this by looking at the component offsets
# which tell us where each component's points start and end in the concatenated array. 
# The last offset gives us the total count of points.
function boundary_matrix_size(pts::Vector{BoundaryPointsCFIE{T}}) where {T<:Real}
    offs=component_offsets(pts)
    return offs[end]-1
end

# For CFIE without holes, we just return the length of the points array since there is only one component.
function boundary_matrix_size(pts::BoundaryPointsCFIE{T}) where {T<:Real}
    return length(pts.xy)
end

# helper function to compute the offsets for each component of the boundary
# which are needed to correctly assemble the R matrix for the CFIE_kress method. 
# The offsets indicate the starting index of each component's points in the concatenated list of all boundary points. 
# For example, if we have 3 components with 10, 15, and 20 points respectively, the offsets would be [1, 11, 26, 46].
function component_offsets(comps::Vector{BoundaryPointsCFIE{T}}) where {T<:Real}
    nc=length(comps)
    offs=Vector{Int}(undef,nc+1)
    offs[1]=1
    for a in 1:nc
        offs[a+1]=offs[a]+length(comps[a].xy)
    end
    return offs
end

function component_offsets(pts::BoundaryPointsCFIE{T}) where {T<:Real}
    N=length(pts.xy)
    return [1,N+1]
end

# NECESSERY DUE TO LEGACY GEOMETRY CONVENTIONS
# Convert boundary into canonical form:
# comps = [comp1, comp2, ...]
# where each comp is a Vector of curves/segments forming one closed component.
function _boundary_components(boundary)
    if isempty(boundary)
        error("Boundary cannot be empty.")
    end
    if boundary[1] isa AbstractVector
        comps=boundary
    else
        comps=[[crv] for crv in boundary]
    end
    return comps
end

# Compute segment lengths, cumulative lengths, and total length
# comp = [seg₁, seg₂, ..., seg_m] forming one closed component
# global arc-length parametrization used to  map the global parameter t ∈ [0,2π) to segments.
function component_lengths(comp::Vector)
    lens=[crv.length for crv in comp]
    cum=Vector{eltype(lens)}(undef,length(lens)+1)
    cum[1]=zero(eltype(lens))
    for j in 1:length(lens)
        cum[j+1]=cum[j]+lens[j]
    end
    return lens,cum,cum[end]
end

# the logic for both start and end functions is to calculate the tangents to determine whether we have a true corner or a smooth join. 
# We use these to compute the angle between them at the junctions. If the angle is larger than a specified tolerance, we consider it a true corner.
@inline function _unit_tangent_at_start(crv,::Type{T}) where {T<:Real}
    v=SVector{2,T}(tangent(crv,zero(T)))
    return v/hypot(v[1],v[2])
end
@inline function _unit_tangent_at_end(crv,::Type{T}) where {T<:Real}
    v=SVector{2,T}(tangent(crv,one(T)))
    return v/hypot(v[1],v[2])
end

# compute the angle between the unit tangents at the junction of two segments. This is used to determine if we have a true corner or a smooth join.
# It can be a bit crude for angle > tol since it is not used afterward.
@inline function _junction_angle(cleft,cright,::Type{T}) where {T<:Real}
    tL=_unit_tangent_at_end(cleft,T)
    tR=_unit_tangent_at_start(cright,T)
    cr=tL[1]*tR[2]-tL[2]*tR[1]
    dt=clamp(tL[1]*tR[1]+tL[2]*tR[2],-one(T),one(T))
    return atan(abs(cr),dt)
end

@inline function _is_true_corner(cleft,cright,::Type{T};angle_tol=T(1e-8)) where {T<:Real}
    return _junction_angle(cleft,cright,T)>angle_tol
end

# A junction is treated as a true corner only when the unit tangent leaving the
# left segment and the unit tangent entering the right segment have a nonzero
# angle larger than `angle_tol`. Smooth joins, such as the line-arc joins in a
# properly oriented stadium, are therefore not graded.
#
# Locations are returned in the global periodic parameter σ ∈ [0,2π), with σ=0
# used for the periodic seam when the last-to-first join is a true corner.
function _component_corner_locations(::Type{T},comp::Vector;angle_tol=T(1e-8)) where {T<:Real}
    _,cum,Ltot=component_lengths(comp)
    corners=T[]
    m=length(comp)
    _is_true_corner(comp[end],comp[1],T;angle_tol=angle_tol)&&push!(corners,zero(T))
    @inbounds for j in 1:m-1
        if _is_true_corner(comp[j],comp[j+1],T;angle_tol=angle_tol)
            push!(corners,T(two_pi)*cum[j+1]/Ltot)
        end
    end
    return corners
end

# debugging tool to see if we picked up corners/joins correctly
function print_component_junctions(comp::Vector;T=Float64,angle_tol=1e-8)
    _,cum,Ltot=component_lengths(comp)
    m=length(comp)
    println("junction diagnostics:")
    @inbounds for j in 1:m
        jr=j==m ? 1 : j+1
        σ=j==m ? zero(T) : T(two_pi)*cum[j+1]/Ltot
        a=_junction_angle(comp[j],comp[jr],T)
        flag=a>T(angle_tol) ? "TRUE CORNER" : "smooth join"
        println("  $j -> $jr : σ = $σ, angle = $a, $flag")
    end
end

# Given a BoundaryPointsCFIE discretization, compute the outward normal vectors at each point.
# usually not needed for CFIE, but we compute it here for completeness and potential use in other formulations.
# the normals are weigthed by the local speed.
function component_normals(pts::BoundaryPointsCFIE{T}) where {T<:Real}
    tx=getindex.(pts.tangent,1)
    ty=getindex.(pts.tangent,2)
    sp=@. sqrt(tx^2+ty^2)
    nx=@.  ty/sp
    ny=@. -tx/sp
    return nx,ny,sp
end

function flatten_cfie_components(comps::Vector{BoundaryPointsCFIE{T}}) where {T<:Real}
    N=sum(length(c.xy) for c in comps)
    x=Vector{T}(undef,N)
    y=Vector{T}(undef,N)
    nx=Vector{T}(undef,N)
    ny=Vector{T}(undef,N)
    ds=Vector{T}(undef,N)
    offs=component_offsets(comps)
    p=1
    for c in comps
        cnx,cny,_=component_normals(c)
        for j in eachindex(c.xy)
            q=c.xy[j]
            x[p]=q[1]
            y[p]=q[2]
            nx[p]=cnx[j]
            ny[p]=cny[j]
            ds[p]=c.ds[j]
            p+=1
        end
    end
    return (;x,y,nx,ny,ds,offs)
end

function flatten_cfie_ds(comps::Vector{BoundaryPointsCFIE{T}}) where {T<:Real}
    Ntot=boundary_matrix_size(comps)
    ds=Vector{T}(undef,Ntot)
    p=1
    for c in comps
        n=length(c.ds)
        ds[p:p+n-1].=c.ds
        p+=n
    end
    return ds
end

# Map global periodic parameter t ∈ [0,2π) to:
# (j,u) where
# j = index of the segment in the composite component
# u ∈ [0,1] = local parameter on that segment
#
# The composite component comp=[seg₁,seg₂,...,seg_m] is viewed as one closed
# boundary with total length Ltot. The global parameter t is first converted to
# an arc-length coordinate s = (t / 2π) * Ltot, and then we determine which segment contains this arc-length position.
#
# Returns:
#   j = segment index
#   u = local parameter on comp[j]
# Special care is taken at the periodic seam so that t near 2π maps back to the first segment with u=0.
function _global_t_to_segment_u(::Type{T},comp::Vector,t::T) where {T<:Real}
    lens,cum,Ltot=component_lengths(comp)
    s=(t/T(two_pi))*Ltot
    s>=Ltot && return 1,zero(T)
    j=searchsortedlast(cum,s)
    j=clamp(j,1,length(comp))
    while j<length(comp) && s>=cum[j+1]
        j+=1
    end
    slocal=s-cum[j]
    u=lens[j]==zero(T) ? zero(T) : slocal/lens[j]
    return j,clamp(u,zero(T),one(T))
end

# Evaluate the composite boundary geometry at global parameter t ∈ [0,2π).
# The boundary component is given as an ordered list of smooth segments
# comp=[seg₁,seg₂,...,seg_m]. This helper:
#   1. maps t to the appropriate segment j and local parameter u,
#   2. evaluates the point γ(u) on that segment,
#   3. converts first and second derivatives from the local segment parameter u
#      to derivatives with respect to the global periodic parameter t.
# If γ_u and γ_uu are derivatives of the segment parameterization with respect to
# u ∈ [0,1], then since du/dt = Ltot / (2π * length(seg_j)), we obtain γ_t  = γ_u  * du/dt,
# γ_tt = γ_uu * (du/dt)^2, because the map from t to u is affine on each segment interior.
#
# Returns:
#   xy   = boundary point γ(t)
#   γt   = first derivative with respect to global parameter t
#   γtt  = second derivative with respect to global parameter t
function _eval_composite_geom_global_t(::Type{T},comp::Vector,t::T) where {T<:Real}
    lens,_,Ltot=component_lengths(comp)
    j,u=_global_t_to_segment_u(T,comp,t)
    crv=comp[j]
    xy=curve(crv,u)
    du_dt=lens[j]==zero(T) ? zero(T) : Ltot/(T(two_pi)*lens[j])
    γu=tangent(crv,u)
    γuu=tangent_2(crv,u)
    γt=γu*du_dt
    γtt=γuu*(du_dt^2)
    return xy,γt,γtt
end

#########################################
#### GEOMETRY CACHE FOR CFIE SOLVERS ####
#########################################

"""
    CFIEGeomCache{T} <: Any

Precomputed geometric cache for CFIE-type boundary integral formulations.

This structure stores pairwise geometric quantities and local differential
geometry derived from a `BoundaryPointsCFIE` discretization. It is designed to
avoid recomputation of distances, tangential interactions, logarithmic kernels,
and curvature-related terms during matrix assembly, especially for singular and
near-singular kernels.

# Fields
- `R::Matrix{T}`:
  Pairwise distances `R[i,j] = |x_i - x_j|` between boundary points. The diagonal
  is regularized to `1` to avoid division-by-zero in downstream computations (overwritten anyway)

- `invR::Matrix{T}`:
  Elementwise inverse distances `1 / R[i,j]`, with zeros on the diagonal (overwritten anyway).

- `inner::Matrix{T}`:
  Tangential interaction term

      inner[i,j] = t_j × (x_i - x_j)

  where `t_j` is the tangent at the source point. In 2D this corresponds to the
  scalar cross product `(dY_j * ΔX - dX_j * ΔY)` and appears in many CFIE kernels.

- `logterm::Matrix{T}`:
  Logarithmic kernel term

      log(4 sin²((t_i - t_j)/2))

  used in Kress-type logarithmic splitting of weakly singular kernels. The
  diagonal is set to zero (diagonal entries are computed separately).

- `speed::Vector{T}`:
  Parametrization speed `|γ'(t)|` at each node.

- `kappa::Vector{T}`:
  Curvature scaled by `1/(2π)`, computed as

      κ = (-(x' y'' - y' x'')) / (x'^2 + y'^2)

  and used in diagonal limits of some kernels.

- `original_ts::Vector{T}`:
  Copy of the original parameter grid when using Kress corner grading. This is
  required to correctly evaluate logarithmic terms in graded parametrizations.
  Empty if no corner treatment is used.
"""
struct CFIEGeomCache{T<:Real}
    R::Matrix{T}
    invR::Matrix{T}
    inner::Matrix{T}
    logterm::Matrix{T}
    speed::Vector{T}
    kappa::Vector{T}
    original_ts::Vector{T} # for kress with corners for keeping track of original trapzoidal discretization for log term.
end

"""
    cfie_geom_cache(pts::BoundaryPointsCFIE{T}; corner_kress::Bool=false) -> CFIEGeomCache{T}

Construct a geometric cache for CFIE boundary integral formulations from a
`BoundaryPointsCFIE` discretization.

This function precomputes pairwise distances, inverse distances, tangential
interaction terms, logarithmic kernel components, parametrization speeds, and
curvature values. These quantities are reused during matrix assembly to reduce
overhead cost.

# Arguments
- `pts::BoundaryPointsCFIE{T}`
- `corner_kress::Bool=false`:
  If `true`, store a copy of the parameter grid `ts` for use in Kress-type
  logarithmic splitting with graded meshes (e.g., near corners). If `false`,
  no parameter copy is stored.

# Returns
- `CFIEGeomCache{T}`
"""
function cfie_geom_cache(pts::BoundaryPointsCFIE{T},corner_kress::Bool=false) where {T<:Real}
    ts=pts.ts
    N=length(pts.xy)
    X=getindex.(pts.xy,1)
    Y=getindex.(pts.xy,2)
    dX=getindex.(pts.tangent,1)
    dY=getindex.(pts.tangent,2)
    ddX=getindex.(pts.tangent_2,1)
    ddY=getindex.(pts.tangent_2,2)
    ΔX=@. X-X'
    ΔY=@. Y-Y'
    R=hypot.(ΔX,ΔY)
    R[diagind(R)].=one(T)
    invR=inv.(R)
    invR[diagind(invR)].=zero(T)
    dX_row=reshape(dX,1,N)
    dY_row=reshape(dY,1,N)
    inner=@. (dY_row*ΔX-dX_row*ΔY)
    # In the graded Kress case, pts.ts already stores the computational
    # periodic grid σ on which the logarithmic split is defined. This works
    # for both odd and even graded variants, so we use it directly.
    original_ts=corner_kress ? copy(ts) : T[]
    ΔT=ts.-ts'
    logterm=log.(4 .*sin.(ΔT./2).^2)
    logterm[diagind(logterm)].=zero(T)
    speed=@. sqrt(dX^2+dY^2)
    κnum= -(dX.*ddY.-dY.*ddX)
    κden=dX.^2 .+dY.^2
    kappa=inv_two_pi.*(κnum./κden)
    return CFIEGeomCache(R,invR,inner,logterm,speed,kappa,original_ts)
end

