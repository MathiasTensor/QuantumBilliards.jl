# Useful reading:
#  - https://github.com/ahbarnett/mpspack - by Alex Barnett & Timo Betcke (MATLAB)
#  - Kress, R., Boundary integral equations in time-harmonic acoustic scattering. Mathematics Comput. Modelling Vol 15, pp. 229-243). Pergamon Press, 1991, GB.
#  - Barnett, A. H., & Betcke, T. (2007). Stability and convergence of the method of fundamental solutions for Helmholtz problems on analytic domains. Journal of Computational Physics, 227(14), 7003-7026.
#  - Zhao, L., & Barnett, A. (2015). Robust and efficient solution of the drum problem via Nyström approximation of the Fredholm determinant. SIAM Journal on Numerical Analysis, Stable URL: https://www.jstor.org/stable/24512689

##########################
#### BESSEL FUNCTIONS ####
##########################

H(n::Int,x::T) where {T<:Real}=Bessels.hankelh1(n,x)
H(n::Int,x::Complex{T}) where {T<:Real}=SpecialFunctions.besselh(n,1,x)
two_pi=2*pi
inv_two_pi=1/two_pi
euler_over_pi=MathConstants.eulergamma/pi

########################################
#### COMMON STRUCT FOR CFIE METHODS ####
########################################

abstract type CFIE<:SweepSolver end

################################
#### CONSTRUCTOR CFIE_kress ####
################################

struct CFIE_kress{T,Bi}<:CFIE where {T<:Real,Bi<:AbsBilliard} 
    sampler::Vector{LinearNodes} # placeholder since the trapezoidal rule will be rescaled
    pts_scaling_factor::Vector{T}
    dim_scaling_factor::T
    eps::T
    min_dim::Int64
    min_pts::Int64
    billiard::Bi
    symmetry::Union{Nothing,Vector{Any}}
end

function CFIE_kress(pts_scaling_factor::Union{T,Vector{T}},billiard::Bi;min_pts=20,eps=T(1e-15),symmetry::Union{Nothing,Vector{Any}}=nothing) where {T<:Real,Bi<:AbsBilliard}
    any([!((boundary isa PolarSegment) || (boundary isa CircleSegment)) for boundary in billiard.full_boundary]) && error("CFIE_kress only works with polar curves")
    bs=typeof(pts_scaling_factor)==T ? [pts_scaling_factor] : pts_scaling_factor
    sampler=[LinearNodes()]
    return CFIE_kress{T,Bi}(sampler,bs,bs[1],eps,min_pts,min_pts,billiard,symmetry)
end

#################################
#### CONSTRUCTOR CFIE_alpert ####
#################################

struct CFIE_alpert{T,Bi}<:CFIE where {T<:Real,Bi<:AbsBilliard}
    sampler::Vector{LinearNodes}
    pts_scaling_factor::Vector{T}
    dim_scaling_factor::T
    eps::T
    min_dim::Int64
    min_pts::Int64
    billiard::Bi
    symmetry::Union{Nothing,Vector{Any}}
    alpert_order::Int
end

function CFIE_alpert(pts_scaling_factor::Union{T,Vector{T}},billiard::Bi;min_pts::Int=20,eps::T=T(1e-15),symmetry::Union{Nothing,Vector{Any}}=nothing,alpert_order::Int=16) where {T<:Real,Bi<:AbsBilliard}
    !(alpert_order in (2,3,4,5,6,8,10,12,14,16)) && error("Alpert order not currently supported")
    _=alpert_log_rule(T,alpert_order)
    bs=pts_scaling_factor isa T ? [pts_scaling_factor] : pts_scaling_factor
    sampler=[LinearNodes()]
    return CFIE_alpert{T,Bi}(sampler,bs,bs[1],eps,min_pts,min_pts,billiard,symmetry,alpert_order)
end

#############################
#### BOUNDARY EVALUATION ####
#############################

# helper function to compute the offsets for each component of the boundary, which are needed to correctly assemble the R matrix for the CFIE_kress method. The offsets indicate the starting index of each component's points in the concatenated list of all boundary points. For example, if we have 3 components with 10, 15, and 20 points respectively, the offsets would be [1, 11, 26, 46].
function component_offsets(comps::Vector)
    nc=length(comps)
    offs=Vector{Int}(undef,nc+1)
    offs[1]=1
    for a in 1:nc
        offs[a+1]=offs[a]+length(comps[a].xy)
    end
    return offs
end

#### use N even for the algorithm - equidistant parameters ####
s(k::Int,N::Int)=two_pi*k/N

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
end

# reverse all BoundaryPointsCFIE except 1st as they correspond to holes in the outer domain.
function _reverse_component_orientation(solver::S,pts::BoundaryPointsCFIE{T}) where {T<:Real,S<:CFIE}
    N=length(pts.xy)
    xy=reverse(pts.xy)
    tangent=reverse(-pts.tangent)
    tangent_2=reverse(pts.tangent_2)
    if solver isa CFIE_kress # FIXME: Future: hacky stuff, kress can have reverse orientation since periodic but for alpert code structured differently
        ts=reverse(pts.ts) # dont touch kress here !!!
    else
        ts=[s(j,N) for j in 1:N]
    end
    ws=copy(pts.ws)
    ws_der=copy(pts.ws_der)
    ds=reverse(pts.ds)
    return BoundaryPointsCFIE(xy,tangent,tangent_2,ts,ws,ws_der,ds,pts.compid,pts.is_periodic)
end

###############
#### KRESS ####
###############

# single crv that builds either the outer or inner boundary (disambigued by idx). For example we can have for billiard.full_boundary = [outer, inner_1, inner_2, ...] where each is a separate crv <:AbsCurve
function _evaluate_points(solver::CFIE_kress{T},crv::C,k::T,idx::Int) where {T<:Real,C<:AbsCurve}
    L=crv.length
    bs=solver.pts_scaling_factor
    N=max(solver.min_pts,round(Int,k*L*bs[1]/(two_pi)))
    needed=2 # need it to. be even number of points for reflections and at same type divisible by rotation order for rotations. A bit hacky but valid for reflections/rotations
    if solver.symmetry!==nothing
        for sym in solver.symmetry
            if sym isa Rotation
                needed=lcm(needed,sym.n)
            end
        end
    end
    remN=mod(N,needed)
    remN!=0 && (N+=needed-remN)
    ts=[s(k,N) for k in 1:N]
    ts_rescaled=ts./two_pi # b/c our curves and tangents are defined on [0,1]
    xy=curve(crv,ts_rescaled) 
    tangent_1st=tangent(crv,ts_rescaled)./(two_pi) # ! Rescaled tangents due to chain rule ∂γ/∂θ = ∂γ/∂u * ∂u/∂θ = ∂γ/∂u * 1/(2π)
    tangent_2nd=tangent_2(crv,ts_rescaled)./(two_pi)^2 # ! Rescaled tangents due to chain rule ∂²γ/∂θ² = ∂²γ/∂u² * (∂u/∂θ)² + ∂γ/∂u * ∂²u/∂θ² = ∂²γ/∂u² * 1/(2π)^2 + ∂γ/∂u * 0 = ∂²γ/∂u² * 1/(2π)^2
    ss=arc_length(crv,ts_rescaled)
    ds=diff(ss)
    append!(ds,L+ss[1]-ss[end])
    ws=fill(T(two_pi/N),N)
    ws_der=ones(T,N) 
    return BoundaryPointsCFIE(xy,tangent_1st,tangent_2nd,ts,ws,ws_der,ds,idx,true)
end

# By default for kress fudamental=false since we need the full set of points for the log split -> for alpert we can use symmetries and can choose fundamental domain
function evaluate_points(solver::CFIE_kress{T},billiard::Bi,k::T) where {T<:Real,Bi<:AbsBilliard}
    boundary=billiard.full_boundary
    pts=Vector{BoundaryPointsCFIE{T}}(undef,length(boundary)) # the desymmetrized boudnary will contain the same number of pieces as the deymmetrized one, so we can use it for enumeration -> 1 for outer boundary, 2 for first hole, etc
    for (idx,crv) in enumerate(boundary)
        pts[idx]= idx==1 ? _evaluate_points(solver,crv,k,idx) : _reverse_component_orientation(solver,_evaluate_points(solver,crv,k,idx))
    end
    return pts
end

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
@inline function _is_closed_curve(crv::C;xtol::T=T(1e-10)) where {T<:Real,C<:AbsCurve}
    x0=curve(crv,[zero(T)])[1]
    x1=curve(crv,[one(T)])[1]
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

# _open_panel_weights
# Build simple open-panel geometric spacing weights from sampled arclength values.
#
# Inputs:
#   - ss::AbstractVector{T} :
#       Arclength values sampled on an open panel.
#
# Outputs:
#   - ds::Vector{T} :
#       Local geometric spacing weights for use in smooth quadrature parts.
function _open_panel_weights(ss::AbstractVector{T}) where {T<:Real}
    N=length(ss)
    ds=Vector{T}(undef,N)
    if N==1
        ds[1]=zero(T)
        return ds
    elseif N==2
        v=ss[2]-ss[1]
        ds[1]=v
        ds[2]=v
        return ds
    end
    ds[1]=ss[2]-ss[1]
    @inbounds for j in 2:N-1
        ds[j]=(ss[j+1]-ss[j-1])/2
    end
    ds[N]=ss[N]-ss[N-1]
    return ds
end

# _evaluate_points_periodic
# Build one closed boundary panel for CFIE_alpert. THis is used whenever the billiard boundary is not composite of many curves, but rather just just one curve
# E.g Ellipse,Circle 
#
# Inputs:
#   - solver::CFIE_alpert{T}
#   - crv::C
#   - k::T
#   - idx::Int
#
# Outputs:
#   - BoundaryPointsCFIE{T}
#
# Notes:
#   - No lcm / symmetry-dependent point adjustment is used here -> since we do it before Beyn.
function _evaluate_points_periodic(solver::CFIE_alpert{T},crv::C,k::T,idx::Int) where {T<:Real,C<:AbsCurve}
    L=crv.length
    bs=solver.pts_scaling_factor
    N=max(solver.min_pts,round(Int,k*L*bs[1]/two_pi))
    N=max(N,4)
    ts=[s(j,N) for j in 1:N]
    ts_rescaled=ts./two_pi
    xy=curve(crv,ts_rescaled)
    tangent_1st=tangent(crv,ts_rescaled)./(two_pi)
    tangent_2nd=tangent_2(crv,ts_rescaled)./(two_pi)^2
    ss=arc_length(crv,ts_rescaled)
    ds=diff(ss)
    append!(ds,L+ss[1]-ss[end])
    ws=fill(T(two_pi/N),N)
    ws_der=ones(T,N)
    return BoundaryPointsCFIE(xy,tangent_1st,tangent_2nd,ts,ws,ws_der,ds,idx,true)
end

# _evaluate_points_panel
# Build one open boundary panel for CFIE_alpert. This is used whenever the billiard boundary is composite of many curves, and we want to treat each curve as a separate panel. E.g for the stadium, we can treat the straight segments as one panel and the circular segments as another.
#
# Inputs:
#   - solver::CFIE_alpert{T}
#   - crv::C
#   - k::T
#   - idx::Int
#
# Outputs:
#   - BoundaryPointsCFIE{T}
# Notes:
#   - No lcm / symmetry-dependent point adjustment is used here -> since we do it before Beyn.
function _evaluate_points_panel(solver::CFIE_alpert{T},crv::C,k::T,idx::Int) where {T<:Real,C<:AbsCurve}
    L=crv.length
    bs=solver.pts_scaling_factor
    N=max(solver.min_pts,round(Int,k*L*bs[1]/two_pi))
    N<2 && (N=2)
    ts=collect(midpoints(range(zero(T),one(T),length=(N+1))))
    xy=curve(crv,ts)
    tangent_1st=tangent(crv,ts)
    tangent_2nd=tangent_2(crv,ts)
    ss=arc_length(crv,ts)
    ds=_open_panel_weights(ss)
    h=inv(T(N))   # midpoint spacing
    ws=fill(h,N)
    ws_der=ones(T,N)
    return BoundaryPointsCFIE(xy,tangent_1st,tangent_2nd,ts,ws,ws_der,ds,idx,false)
end

#=
# For alpert quadrature we can have desymmetrized domains already in the kernel construction since we dont need global periodic parametrization
# Structure: [[outer boundary pieces], [inner boundary 1 pieces], [inner boundary 2 pieces], ...] where each piece is a separate curve segment. 
function evaluate_points(solver::CFIE_alpert{T},billiard::Bi,k::T) where {T<:Real,Bi<:AbsBilliard}
    boundary=isnothing(solver.symmetry) ? billiard.full_boundary : billiard.desymmetrized_full_boundary
    # case 1: simple billiard with outer boundary [outer, hole1, hole2, ...], each is a single closed curve
    if !(boundary[1] isa AbstractVector)
        pts=Vector{BoundaryPointsCFIE{T}}(undef,length(boundary))
        for (idx,crv) in enumerate(boundary)
            p=_is_closed_curve(crv) ? _evaluate_points_periodic(solver,crv,k,idx) : _evaluate_points_panel(solver,crv,k,idx)
            pts[idx]=(idx==1) ? p : _reverse_component_orientation(solver,p)
        end
        return pts
    end
    # case 2: panelized style, 1st vector of boudnary is the outer boundary with potentially many components, second vector is the first hole etc. [[outer panels], [hole1 panels], ...]
    outer_boundary=boundary[1]
    inner_boundaries=boundary[2:end]
    n_outer=length(outer_boundary)
    n_inner=sum(length(comp) for comp in inner_boundaries)
    pts=Vector{BoundaryPointsCFIE{T}}(undef,n_outer+n_inner)
    pos=1 # global position in the concatenated pts array
    for crv in outer_boundary
        pts[pos]=_evaluate_points_panel(solver,crv,k,pos)
        pos+=1
    end
    for comp in inner_boundaries
        for crv in comp
            pts[pos]=_reverse_component_orientation(solver,_evaluate_points_panel(solver,crv,k,pos))
            pos+=1
        end
    end
    return pts
end
=#

function evaluate_points(solver::CFIE_alpert{T},billiard::Bi,k::T) where {T<:Real,Bi<:AbsBilliard}
    boundary=isnothing(solver.symmetry) ? billiard.full_boundary : billiard.desymmetrized_full_boundary
    # case 1: simple closed components [outer,hole1,hole2,...]
    if !(boundary[1] isa AbstractVector) && _all_closed_curves(boundary)
        pts=Vector{BoundaryPointsCFIE{T}}(undef,length(boundary))
        for (idx,crv) in enumerate(boundary)
            p=_evaluate_points_periodic(solver,crv,k,idx)
            pts[idx]=(idx==1) ? p : _reverse_component_orientation(solver,p)
        end
        return pts
    end
    # case 2: single composite boundary [seg1,seg2,...] (eg stadium)
    if _is_single_composite_boundary(boundary)
        pts=Vector{BoundaryPointsCFIE{T}}(undef,length(boundary))
        for (idx,crv) in enumerate(boundary)
            pts[idx]=_evaluate_points_panel(solver,crv,k,idx)
        end
        return pts
    end
    # case 3: nested composite components [[outer...],[hole1...],...]
    outer_boundary=boundary[1]
    inner_boundaries=boundary[2:end]
    n_outer=length(outer_boundary)
    n_inner=sum(length(comp) for comp in inner_boundaries)
    pts=Vector{BoundaryPointsCFIE{T}}(undef,n_outer+n_inner)
    pos=1
    for crv in outer_boundary
        pts[pos]=_evaluate_points_panel(solver,crv,k,pos)
        pos+=1
    end
    for comp in inner_boundaries
        for crv in comp
            pts[pos]=_reverse_component_orientation(solver,_evaluate_points_panel(solver,crv,k,pos))
            pos+=1
        end
    end
    return pts
end

# For CFIE with holes, we compute this by looking at the component offsets, which tell us where each component's points start and end in the concatenated array. The last offset gives us the total count of points.
function boundary_matrix_size(pts::Vector{BoundaryPointsCFIE{T}}) where {T<:Real}
    offs=component_offsets(pts)
    return offs[end]-1
end

#########################################
#### GEOMETRY CACHE FOR CFIE SOLVERS ####
#########################################

struct CFIEGeomCache{T<:Real}
    R::Matrix{T}
    invR::Matrix{T}
    inner::Matrix{T}
    logterm::Matrix{T}
    speed::Vector{T}
    kappa::Vector{T}
end

function cfie_geom_cache(pts::BoundaryPointsCFIE{T}) where {T<:Real}
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
    inner=@. dY_row*ΔX-dX_row*ΔY
    ΔT=ts.-ts'
    logterm=log.(4 .*sin.(ΔT./2).^2)
    logterm[diagind(logterm)].=zero(T)
    speed=@. sqrt(dX^2+dY^2)
    κnum=dX.*ddY.-dY.*ddX
    κden=dX.^2 .+ dY.^2
    kappa=inv_two_pi.*(κnum./κden)
    return CFIEGeomCache(R,invR,inner,logterm,speed,kappa)
end

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
#   - sym :
#       Currently expected to be a vector containing exactly one symmetry
#       object (`Reflection` or `Rotation`).
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
    sym=sym[1] #FIXME Stupid hack, get rid of this and keep only the fundamental domain's symmetry
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
#       Currently expected to be a vector containing exactly one symmetry object.
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
    sym=sym[1] #FIXME Stupid hack, get rid of this and keep only the fundamental domain's symmetry
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

# image_tangent_x
# Tangent map for a y-axis reflection image used in the desymmetrized CFIE assembly.
#
# Inputs:
#   - t::SVector{2,T} :
#       Source tangent on the fundamental boundary.
#
# Outputs:
#   - timg::SVector{2,T} :
#       Tangent of the reflected image curve with the correct full-boundary orientation.
#
# Notes:
#   - A single reflection reverses orientation.
#   - Therefore we must negate the reflected tangent.
#   - Reflection across the y-axis sends (tx,ty) -> (-tx,ty), and restoring the
#     physical CCW orientation gives -( -tx,ty ) = (tx,-ty).
@inline function image_tangent_x(t::SVector{2,T}) where {T<:Real}
    tx,ty=_x_reflect_normal(t[1],t[2])
    return SVector{2,T}(-tx,-ty)
end

# image_tangent_y
# Tangent map for an x-axis reflection image used in the desymmetrized CFIE assembly.
#
# Inputs:
#   - t::SVector{2,T} :
#       Source tangent on the fundamental boundary.
#
# Outputs:
#   - timg::SVector{2,T} :
#       Tangent of the reflected image curve with the correct full-boundary orientation.
#
# Notes:
#   - A single reflection reverses orientation.
#   - Therefore we must negate the reflected tangent.
#   - Reflection across the x-axis sends (tx,ty) -> (tx,-ty), and restoring the
#     physical CCW orientation gives -( tx,-ty ) = (-tx,ty).
@inline function image_tangent_y(t::SVector{2,T}) where {T<:Real}
    tx,ty=_y_reflect_normal(t[1],t[2])
    return SVector{2,T}(-tx,-ty)
end

# image_tangent_xy
# Tangent map for the double reflection (origin / XY image).
#
# Inputs:
#   - t::SVector{2,T} :
#       Source tangent on the fundamental boundary.
#
# Outputs:
#   - timg::SVector{2,T} :
#       Tangent of the doubly reflected image curve.
#
# Notes:
#   - The double reflection has determinant +1, so orientation is preserved.
#   - Hence no additional minus sign is needed here.
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