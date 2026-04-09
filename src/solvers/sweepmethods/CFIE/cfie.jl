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

struct CFIE_kress{T<:Real,Bi<:AbsBilliard,Sym}<:CFIE 
    sampler::Vector{LinearNodes} # placeholder since the trapezoidal rule can be changed. Not used currently.
    pts_scaling_factor::Vector{T} # scaling factor for the number of points per wavelength, which is used to determine the number of discretization points on the boundary based on the wavenumber and the length of the boundary. It can be a single value or a vector of values for different components of the boundary.
    dim_scaling_factor::T # UNUSED since no basis. Only for compatibility
    eps::T # UNUSED, for compatibility
    min_dim::Int64 # UNUSED, for compatibility
    min_pts::Int64 # minimum number of discretization points on the boundary, which ensures that even for low wavenumbers or short boundaries, we have enough points to accurately represent the geometry and solve the integral equation.
    billiard::Bi # the billiard domain for which we are solving the CFIE. It contains the geometry of the problem, including the boundary curves and their properties, which are essential for constructing the system matrix and solving the eigenvalue problem.
    symmetry::Sym # symmetry information for the billiard, which can be used to reduce the computational cost by exploiting symmetries in the geometry.
end

# case where we have corners and need to use gradings
struct CFIE_kress_corners{T<:Real,Bi<:AbsBilliard,Sym} <: CFIE
    sampler::Vector{LinearNodes} # placeholder since the trapezoidal rule will be changed by v(s,q) in kress_graging.jl
    pts_scaling_factor::Vector{T} 
    dim_scaling_factor::T # UNUSED since no basis. Only for compatibility
    eps::T # UNUSED, for compatibility
    min_dim::Int64 # UNUSED, for compatibility
    min_pts::Int64
    billiard::Bi 
    symmetry::Sym
    kressq::Int # the grading parameter q in the Kress grading formula, which controls how strongly the nodes are clustered near the corners. A larger value of q results in stronger clustering, which can improve accuracy for problems with sharp corners. Typical values are in the range of 4 to 16, with 8 being a common choice for many problems.
end

"""
    CFIE_kress(pts_scaling_factor::Union{T,Vector{T}},billiard::Bi;min_pts=20,eps=T(1e-15),symmetry::Sym=nothing)

Constructor for CFIE_kress solver.

# Inputs:
- `pts_scaling_factor`: A scaling factor for the number of points per wavelength, which is used to determine the number of discretization points on the boundary based on the wavenumber and the length of the boundary. It can be a single value or a vector of values for different components of the boundary.
- `billiard`: The billiard domain for which we are solving the CFIE. It contains the geometry of the problem, including the boundary curves and their properties, which are essential for constructing the system matrix and solving the eigenvalue problem.
- `min_pts`: Minimum number of discretization points on the boundary, which ensures that even for low wavenumbers or short boundaries, we have enough points to accurately represent the geometry and solve the integral equation. Default is 20.
- `eps`: Unused internally.
- `symmetry`: Symmetry information for the billiard, which can be used to reduce the computational cost by exploiting symmetries in the geometry. It can be `nothing` if no symmetry is present or an instance of a type that implements `AbsSymmetry` if symmetries are present. Default is `nothing`.

# Output:
- An instance of the `CFIE_kress` solver initialized with the provided parameters.
"""
function CFIE_kress(pts_scaling_factor::Union{T,Vector{T}},billiard::Bi;min_pts=20,eps=T(1e-15),symmetry::Union{Nothing,AbsSymmetry}=nothing) where {T<:Real,Bi<:AbsBilliard}
    any([!((boundary isa PolarSegment) || (boundary isa CircleSegment) || (boundary isa LineSegment)) for boundary in billiard.full_boundary]) && error("CFIE_kress only works with polar curves and line segments")
    bs=typeof(pts_scaling_factor)==T ? [pts_scaling_factor] : pts_scaling_factor
    sampler=[LinearNodes()]
    Sym=typeof(symmetry)
    return CFIE_kress{T,Bi,Sym}(sampler,bs,bs[1],eps,min_pts,min_pts,billiard,symmetry)
end

"""
    CFIE_kress_corners(pts_scaling_factor::Union{T,Vector{T}},billiard::Bi;min_pts=20,eps=T(1e-15),symmetry::Union{Nothing,AbsSymmetry}=nothing,kressq=8)

Constructor for CFIE_kress_corners solver, which is designed to handle billiards with corners by using Kress's grading technique. This method clusters the discretization points near the corners to improve accuracy. 

# Inputs:
- `pts_scaling_factor`: A scaling factor for the number of points per wavelength, which is used to determine the number of discretization points on the boundary based on the wavenumber and the length of the boundary. It can be a single value or a vector of values for different components of the boundary.
- `billiard`: The billiard domain for which we are solving the CFIE. It contains the geometry of the problem, including the boundary curves and their properties, which are essential for constructing the system matrix and solving the eigenvalue problem.
- `min_pts`: Minimum number of discretization points on the boundary, which ensures that even for low wavenumbers or short boundaries, we have enough points to accurately represent the geometry and solve the integral equation. Practically irrelevant since we will always be above this value. Default is 20.
- `eps`: Unused internally.
- `symmetry`: Symmetry information for the billiard, which can be used to reduce the computational cost by exploiting symmetries in the geometry. It can be `nothing` if no symmetry is present or an instance of a type that implements `AbsSymmetry` if symmetries are present. Default is `nothing`.
- `kressq`: The grading parameter q in Kress's grading technique, which controls the clustering of discretization points near the corners. Default is 8.
"""
function CFIE_kress_corners(pts_scaling_factor::Union{T,Vector{T}},billiard::Bi;min_pts=20,eps=T(1e-15),symmetry::Union{Nothing,AbsSymmetry}=nothing,kressq=8) where {T<:Real,Bi<:AbsBilliard}
    any([!((boundary isa PolarSegment) || (boundary isa CircleSegment) || (boundary isa LineSegment)) for boundary in billiard.full_boundary]) && error("CFIE_kress_corners only works with polar curves and line segments")
    bs=typeof(pts_scaling_factor)==T ? [pts_scaling_factor] : pts_scaling_factor
    sampler=[LinearNodes()]
    Sym=typeof(symmetry)
    return CFIE_kress_corners{T,Bi,Sym}(sampler,bs,bs[1],eps,min_pts,min_pts,billiard,symmetry,kressq)
end

#################################
#### CONSTRUCTOR CFIE_alpert ####
#################################

struct CFIE_alpert{T<:Real,Bi<:AbsBilliard,Sym}<:CFIE
    sampler::Vector{LinearNodes}
    pts_scaling_factor::Vector{T}
    dim_scaling_factor::T
    eps::T
    min_dim::Int64
    min_pts::Int64
    billiard::Bi
    symmetry::Sym
    alpert_order::Int
end

"""
    CFIE_alpert(pts_scaling_factor::Union{T,Vector{T}},billiard::Bi;min_pts=20,eps=T(1e-15),symmetry::Union{Nothing,S}=nothing,alpert_order=16)

Constructor for CFIE_alpert solver.

# Inputs:
- `pts_scaling_factor`: A scaling factor for the number of points per wavelength, which is used to determine the number of discretization points on the boundary based on the wavenumber and the length of the boundary. It can be a single value or a vector of values for different components of the boundary.
- `billiard`: The billiard domain for which we are solving the CFIE. It contains the geometry of the problem, including the boundary curves and their properties, which are essential for constructing the system matrix and solving the eigenvalue problem.
- `min_pts`: Minimum number of discretization points on the boundary, which ensures that even for low wavenumbers or short boundaries, we have enough points to accurately represent the geometry and solve the integral equation. Default is 20.
- `eps`: Unused internally.
- `symmetry`: Symmetry information for the billiard, which can be used to reduce the computational cost by exploiting symmetries in the geometry. It can be `nothing` if no symmetry is present or an instance of a type that implements `AbsSymmetry` if symmetries are present. Default is `nothing`.
- `alpert_order`: The order of the Alpert quadrature correction to use for near interactions. Supported values are 2, 3, 4, 5, 6, 8, 10, 12, 14, and 16. Default is 16.

# Output:
- An instance of the `CFIE_alpert` solver initialized with the provided parameters.
"""
function CFIE_alpert(pts_scaling_factor::Union{T,Vector{T}},billiard::Bi;min_pts::Int=20,eps::T=T(1e-15),symmetry::Union{Nothing,AbsSymmetry}=nothing,alpert_order::Int=16) where {T<:Real,Bi<:AbsBilliard}
    !(alpert_order in (2,3,4,5,6,8,10,12,14,16)) && error("Alpert order not currently supported")
    _=alpert_log_rule(T,alpert_order)
    bs=pts_scaling_factor isa T ? [pts_scaling_factor] : pts_scaling_factor
    sampler=[LinearNodes()]
    Sym=typeof(symmetry)
    return CFIE_alpert{T,Bi,Sym}(sampler,bs,bs[1],eps,min_pts,min_pts,billiard,symmetry,alpert_order)
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
# this function is really tricky since we need to reverse the order of the points but also flip the tangents and ds to maintain the correct orientation for the holes. We also need to be careful with the periodicity and the weights. The compid should remain unchanged since we are just reversing the order of points within the same component. Closed periodic polar curves ts behave differently from open panels due to the definiiton of the log analytic split for Kress needing [s(j,N) for j in 1:N] while for Alpert we chose midpoints.
#=
function _reverse_component_orientation(solver::S,pts::BoundaryPointsCFIE{T}) where {T<:Real,S<:CFIE}
    N=length(pts.xy)
    xy=reverse(pts.xy)
    tangent=reverse(-pts.tangent)
    tangent_2=reverse(pts.tangent_2)
    ts=if pts.is_periodic
        [s(j,N) for j in 1:N]
    else
        collect(midpoints(range(zero(T),one(T),length=N+1)))
    end
    ws=copy(pts.ws)
    ws_der=copy(pts.ws_der)
    ds=reverse(pts.ds)
    return BoundaryPointsCFIE(xy,tangent,tangent_2,ts,ws,ws_der,ds,pts.compid,pts.is_periodic)
end
=#
function _reverse_component_orientation(solver::S,pts::BoundaryPointsCFIE{T}) where {T<:Real,S<:CFIE}
    xy=reverse(pts.xy)
    tangent=reverse(-pts.tangent)
    tangent_2=reverse(pts.tangent_2)
    ts=reverse(pts.ts)
    ws=reverse(pts.ws)
    ws_der=reverse(pts.ws_der)
    ds=reverse(pts.ds)
    return BoundaryPointsCFIE(xy,tangent,tangent_2,ts,ws,ws_der,ds,pts.compid,pts.is_periodic)
end

###############
#### KRESS ####
###############

"""
    _evaluate_points(solver::CFIE_kress{T},crv::C,k::T,idx::Int) where {T<:Real,C<:AbsCurve}

Helper function to evaluate the boundary points, tangents, and weights for a single curve component of the billiard. This function is called by `evaluate_points` for each component of the boundary and constructs the `BoundaryPointsCFIE` struct for that component.

billiard.full_boundary is expected to be a vector of AbsCurve objects, where each AbsCurve represents a separate boundary component (e.g., outer boundary, hole 1, hole 2, etc.). The `idx` parameter is used to identify which component we are evaluating and to set the `compid` field in the `BoundaryPointsCFIE` struct accordingly.

# Inputs
- `solver`: The CFIE_kress solver instance containing the boundary discretization and weights.
- `crv`: The curve component (of type AbsCurve) for which to evaluate the boundary points and tangents.
- `k`: The wavenumber for which to evaluate the points and tangents.
- `idx`: The index of the boundary component (1 for outer boundary, 2 for first hole, etc.) which is used to set the `compid` field in the `BoundaryPointsCFIE` struct and to determine the orientation of the tangents and weights.

# Output
- A `BoundaryPointsCFIE` struct containing the evaluated boundary points, tangents, weights, and other relevant information for the specified curve component.
"""
function _evaluate_points(solver::CFIE_kress{T},crv::C,k::T,idx::Int) where {T<:Real,C<:AbsCurve}
    L=crv.length
    bs=solver.pts_scaling_factor
    N=max(solver.min_pts,round(Int,k*L*bs[1]/(two_pi)))
    needed=2 # need it to. be even number of points for reflections and at same type divisible by rotation order for rotations. A bit hacky but valid for reflections/rotations. If we dont do this build_rotation_maps_components crashes
    if !isnothing(solver.symmetry)
        sym=solver.symmetry
        if sym isa Rotation
            needed=lcm(needed,sym.n)
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
    ws_der=ones(T,N) # unused, legacy
    return BoundaryPointsCFIE(xy,tangent_1st,tangent_2nd,ts,ws,ws_der,ds,idx,true)
end

#######################
#### KRESS CORNERS ####
#######################

function _evaluate_points(solver::CFIE_kress_corners{T},crv::C,k::T,idx::Int) where {T<:Real,C<:AbsCurve}
    L=crv.length
    bs=solver.pts_scaling_factor
    N=max(solver.min_pts,round(Int,k*L*bs[1]/two_pi))
    needed=1
    if !isnothing(solver.symmetry)
        sym=solver.symmetry
        if sym isa Rotation
            iseven(sym.n) && error("Incompatible. If sym.n is even, please use reflections.")
            needed=lcm(needed,sym.n)
        end
    end
    remN=mod(N,needed)
    remN!=0 && (N+=needed-remN)
    iseven(N) && (N+=needed)

    σ,tmap,jac,jac2,_=kress_graded_nodes_data(T,N;q=solver.kressq)

    u=tmap./two_pi
    xy=curve(crv,u)
    γu=tangent(crv,u)
    γuu=tangent_2(crv,u)

    tangent_1st=[γu[i]*(jac[i]/two_pi) for i in eachindex(u)]
    tangent_2nd=[γuu[i]*(jac[i]/two_pi)^2 + γu[i]*(jac2[i]/two_pi) for i in eachindex(u)]

    ss=arc_length(crv,u)
    ds=diff(ss)
    append!(ds,L+ss[1]-ss[end])

    h=pi/T((N+1)÷2)

    ts=σ
    ws=fill(h,N)
    ws_der=jac

    return BoundaryPointsCFIE(xy,tangent_1st,tangent_2nd,ts,ws,ws_der,ds,idx,true)
end

####################
#### HIGH LEVEL ####
####################

"""
    evaluate_points(solver::Union{CFIE_kress{T},CFIE_kress_corners{T}},billiard::Bi,k::T) where {T<:Real,Bi<:AbsBilliard}

Evaluate the boundary points, tangents, and weights for all components of the billiard's boundary using the CFIE_kress or CFIE_kress_corners method. This function iterates over each component of the boundary, calls the `_evaluate_points` helper function to compute the necessary information for each component, and then assembles the results into a vector of `BoundaryPointsCFIE` structs. The function also handles the orientation of the tangents and weights for holes in the billiard by reversing their order and flipping their signs as needed.

Accepts either:
- A vector of curve components where each component is a single smooth closed curve (e.g., `[outer, hole1, hole2, ...]`).
- A vector of vectors where each inner vector contains a single curve component (e.g., `[[outer], [hole1], [hole2], ...]`). This is how Alpet expects the input, but we allow both for flexibility. The function will check the structure of the input and process it accordingly.
Rejects composite multi-segment components like `[[seg1, seg2, ...], ...]` since CFIE_kress requires each component to be a single smooth closed curve.

# Inputs:
- `solver`: The CFIE_kress or CFIE_kress_corners solver instance containing the boundary discretization and weights.
- `billiard`: The billiard domain for which we are solving the CFIE. It contains the geometry of the problem, including the boundary curves and their properties, which are essential for constructing the system matrix and solving the eigenvalue problem.
- `k`: The wavenumber for which to evaluate the boundary points and tangents.

# Output:
- A vector of `BoundaryPointsCFIE` structs, where each struct contains the evaluated boundary points, tangents, weights, and other relevant information for each component of the billiard's boundary. The first component corresponds to the outer boundary, and subsequent components correspond to holes in the billiard, with their tangents and weights appropriately oriented.
"""
function evaluate_points(solver::Union{CFIE_kress{T},CFIE_kress_corners{T}},billiard::Bi,k::T) where {T<:Real,Bi<:AbsBilliard}
    boundary=billiard.full_boundary
    if !(boundary[1] isa AbstractVector)
        pts=Vector{BoundaryPointsCFIE{T}}(undef,length(boundary))
        for (idx,crv) in enumerate(boundary)
            p=_evaluate_points(solver,crv,k,idx)
            pts[idx]=idx==1 ? p : _reverse_component_orientation(solver,p)
        end
        return pts
    end
    ncomp=length(boundary)
    pts=Vector{BoundaryPointsCFIE{T}}(undef,ncomp)
    for (idx,comp) in enumerate(boundary)
        length(comp)==1 || error("Kress requires each boundary component to be a single smooth closed curve.")
        crv=comp[1]
        p=_evaluate_points(solver,crv,k,idx)
        pts[idx]=idx==1 ? p : _reverse_component_orientation(solver,p)
    end

    return pts
end

################
#### ALPERT ####
################

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
    ts=[T(two_pi)*(j-T(1)/2)/T(N) for j in 1:N]
    #ts_rescaled=ts./two_pi
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

"""
    evaluate_points(solver::CFIE_alpert{T},billiard::Bi,k::T) where {T<:Real,Bi<:AbsBilliard}

Evaluate the boundary points, tangents, and weights for all components of the billiard's boundary using the CFIE_alpert method. This function iterates over each component of the boundary, determines whether it is a closed curve or an open panel, and calls the appropriate helper function to compute the necessary information for each component. The results are assembled into a vector of `BoundaryPointsCFIE` structs, with correct orientation for holes in the billiard.

Accepts either:
- A vector of curve components where each component is a single smooth closed curve (e.g., `[outer, hole1, hole2, ...]`).
- A vector of vectors where each vector contains multiple curve segments representing a composite boundary (e.g., `[[seg1, seg2, ...], [hole1_seg1, hole1_seg2, ...], ...]`). The function will check the structure of the input and process it accordingly, treating each inner vector as a separate component. For composite boundaries, it will treat each inner vector as a single component and apply open panel discretization to each segment, while for single closed curves, it will apply periodic discretization.

# Inputs:
- `solver`: The CFIE_alpert solver instance containing the boundary discretization and weights.
- `billiard`: The billiard domain for which we are solving the CFIE. It contains the geometry of the problem, including the boundary curves and their properties, which are essential for constructing the system matrix and solving the eigenvalue problem.
- `k`: The wavenumber for which to evaluate the boundary points and tangents.   

# Output:
- A vector of `BoundaryPointsCFIE` structs, where each struct contains the evaluated boundary points, tangents, weights, and other relevant information for each component of the billiard's boundary. The first component corresponds to the outer boundary, and subsequent components correspond to holes in the billiard, with their tangents and weights appropriately oriented. For composite boundaries, each inner vector of curve segments is treated as a single component, and the points are evaluated accordingly.
"""
function evaluate_points(solver::CFIE_alpert{T},billiard::Bi,k::T) where {T<:Real,Bi<:AbsBilliard}
    boundary=isnothing(solver.symmetry) ? billiard.full_boundary : billiard.desymmetrized_full_boundary
    if !(boundary[1] isa AbstractVector) && _all_closed_curves(boundary)
        pts=Vector{BoundaryPointsCFIE{T}}(undef,length(boundary))
        for (idx,crv) in enumerate(boundary)
            p=_evaluate_points_periodic(solver,crv,k,idx)
            pts[idx]=(idx==1) ? p : _reverse_component_orientation(solver,p)
        end
        return pts
    end
    if _is_single_composite_boundary(boundary)
        pts=Vector{BoundaryPointsCFIE{T}}(undef,length(boundary))
        for (idx,crv) in enumerate(boundary)
            pts[idx]=_evaluate_points_panel(solver,crv,k,1)
        end
        return pts
    end
    ncomps=length(boundary)
    npanels=sum(length(comp) for comp in boundary)
    pts=Vector{BoundaryPointsCFIE{T}}(undef,npanels)
    pos=1
    for compid in 1:ncomps
        comp=boundary[compid]
        for crv in comp
            p=_evaluate_points_panel(solver,crv,k,compid)
            pts[pos]=(compid==1) ? p : _reverse_component_orientation(solver,p)
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
    original_ts::Vector{T} # for kress with corners for keeping track of original trapzoidal discretization for log term.
end

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
    original_ts=Vector{T}(undef,N) # we need original ts before the grading for the log correction term,so we need to reconstruct them
    if corner_kress
        n=(N+1)÷2 # N is odd for Kress grading, so this is ok
        original_ts=[T(k*pi/n) for k in 1:N]
        ΔT=original_ts.-original_ts'
    else
        ΔT=ts.-ts'
    end
    logterm=log.(4 .*sin.(ΔT./2).^2)
    logterm[diagind(logterm)].=zero(T)
    speed=@. sqrt(dX^2+dY^2)
    κnum= -(dX.*ddY.-dY.*ddX)
    κden=dX.^2 .+dY.^2
    kappa=inv_two_pi.*(κnum./κden)
    return CFIEGeomCache(R,invR,inner,logterm,speed,kappa,original_ts)
end

