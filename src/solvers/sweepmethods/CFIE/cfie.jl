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
@inline function hankel_pair01(x);h0=H(0,x);h1=H(1,x);return h0,h1;end

########################################
#### COMMON STRUCT FOR CFIE METHODS ####
########################################



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

# same as above comments
struct CFIE_kress_global_corners{T<:Real,Bi<:AbsBilliard,Sym} <: CFIE
    sampler::Vector{LinearNodes}
    pts_scaling_factor::Vector{T}
    dim_scaling_factor::T
    eps::T
    min_dim::Int64
    min_pts::Int64
    billiard::Bi
    symmetry::Sym
    kressq::Int
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
- `kressq`: The grading parameter q in Kress's grading technique, which controls the clustering of discretization points near the corners. Default is 4. !!! DO NOT PUSH THIS PAST 4 IN FLOAT64
"""
function CFIE_kress_corners(pts_scaling_factor::Union{T,Vector{T}},billiard::Bi;min_pts=20,eps=T(1e-15),symmetry::Union{Nothing,AbsSymmetry}=nothing,kressq=4) where {T<:Real,Bi<:AbsBilliard}
    bs=typeof(pts_scaling_factor)==T ? [pts_scaling_factor] : pts_scaling_factor
    sampler=[LinearNodes()]
    Sym=typeof(symmetry)
    return CFIE_kress_corners{T,Bi,Sym}(sampler,bs,bs[1],eps,min_pts,min_pts,billiard,symmetry,kressq)
end

"""
    CFIE_kress_global_corners(pts_scaling_factor::Union{T,Vector{T}},billiard::Bi;min_pts=20,eps=T(1e-15),symmetry::Union{Nothing,AbsSymmetry}=nothing,kressq=8)

Constructor for CFIE_kress_global_corners solver, which is designed to handle billiards with corners by using a global Kress grading technique. This method clusters the discretization points near the corners based on a global grading function that takes into account all corner locations simultaneously. Therefore one can reuse the circulant R matrix logic.

# Inputs:
- `pts_scaling_factor`: A scaling factor for the number of points per wavelength, which is used to determine the number of discretization points on the boundary based on the wavenumber and the length of the boundary. It can be a single value or a vector of values for different components of the boundary.
- `billiard`: The billiard domain for which we are solving the CFIE. It contains the geometry of the problem, including the boundary curves and their properties, which are essential for constructing the system matrix and solving the eigenvalue problem.
- `min_pts`: Minimum number of discretization points on the boundary, which ensures that even for low wavenumbers or short boundaries, we have enough points to accurately represent the geometry and solve the integral equation. Practically irrelevant since we will always be above this value. Default is 20.
- `eps`: Unused internally.
- `symmetry`: Symmetry information for the billiard, which can be used to reduce the computational cost by exploiting symmetries in the geometry. It can be `nothing` if no symmetry is present or an instance of a type that implements `AbsSymmetry` if symmetries are present. Default is `nothing`.
- `kressq`: The grading parameter q in the global Kress grading technique, which controls the clustering of discretization points near the corners. Default is 8.
"""
function CFIE_kress_global_corners(pts_scaling_factor::Union{T,Vector{T}},billiard::Bi;min_pts=20,eps=T(1e-15),symmetry::Union{Nothing,AbsSymmetry}=nothing,kressq=4) where {T<:Real,Bi<:AbsBilliard}
    bs=pts_scaling_factor isa T ? [pts_scaling_factor] : pts_scaling_factor
    sampler=[LinearNodes()]
    Sym=typeof(symmetry)
    return CFIE_kress_global_corners{T,Bi,Sym}(sampler,bs,bs[1],eps,min_pts,min_pts,billiard,symmetry,kressq)
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
    alpertq::Int
end

# _warn_aggressive_alpert
# This function checks if the combination of the Alpert quadrature parameters (order and grading strength) and the geometry of the billiard (specifically the lengths of the boundary segments) may lead to under-resolution of the near-correction on the shortest boundary segment. 
# It calculates a heuristic danger ratio based on these parameters and issues a warning if the ratio exceeds certain thresholds, suggesting adjustments to the Alpert parameters for better accuracy.
function _warn_aggressive_alpert(pts_scaling_factor,billiard,alpert_order::Int,alpertq::Int)
    bs=pts_scaling_factor isa AbstractVector ? pts_scaling_factor : [pts_scaling_factor]
    bmin=minimum(bs)
    boundary=billiard.full_boundary
    lens=Float64[]
    if boundary[1] isa AbstractVector
        for comp in boundary;append!(lens,[crv.length for crv in comp]);end
    else;append!(lens,[crv.length for crv in boundary]);end
    Lmin=minimum(lens) # minimum length is the most problematic since the near-correction will be strongest there and we need to make sure we have enough points to resolve it. 
    Lavg=sum(lens)/length(lens) # The average length is also relevant since it gives us a sense of the overall discretization density.
    # heuristic danger ratio:
    # bigger order, bigger q, smaller b, shorter smallest panel => more dangerous
    R=(alpert_order*alpertq)/(bmin*Lmin/Lavg)
    if R>6.0
        b_suggest=(alpert_order*alpertq)/(4.0*Lmin/Lavg)
        q_suggest=(4.0*bmin*Lmin/Lavg)/alpert_order
        @warn "CFIE_alpert: aggressive grading / near-correction may be under-resolved on the shortest boundary segment." b=bmin alpert_order=alpert_order alpertq=alpertq shortest_segment=Lmin average_segment=Lavg ratio=R suggested_min_b=b_suggest suggested_max_q=q_suggest
    elseif R>4.0
        b_suggest=(alpert_order*alpertq)/(4.0*Lmin/Lavg)
        q_suggest=(4.0*bmin*Lmin/Lavg)/alpert_order
        @info "CFIE_alpert: borderline grading / correction strength." b=bmin alpert_order=alpert_order alpertq=alpertq shortest_segment=Lmin average_segment=Lavg ratio=R suggested_min_b=b_suggest suggested_max_q=q_suggest
    end
    return nothing
end

"""
    CFIE_alpert(pts_scaling_factor::Union{T,Vector{T}},billiard::Bi;min_pts=20,eps=T(1e-15),symmetry::Union{Nothing,S}=nothing,alpert_order=16,alpertq=8) where {T<:Real,Bi<:AbsBilliard,S<:AbsSymmetry}

Constructor for CFIE_alpert solver.

# Inputs:
- `pts_scaling_factor`: A scaling factor for the number of points per wavelength, which is used to determine the number of discretization points on the boundary based on the wavenumber and the length of the boundary. It can be a single value or a vector of values for different components of the boundary.
- `billiard`: The billiard domain for which we are solving the CFIE. It contains the geometry of the problem, including the boundary curves and their properties, which are essential for constructing the system matrix and solving the eigenvalue problem.
- `min_pts`: Minimum number of discretization points on the boundary, which ensures that even for low wavenumbers or short boundaries, we have enough points to accurately represent the geometry and solve the integral equation. Default is 20.
- `eps`: Unused internally.
- `symmetry`: Symmetry information for the billiard, which can be used to reduce the computational cost by exploiting symmetries in the geometry. It can be `nothing` if no symmetry is present or an instance of a type that implements `AbsSymmetry` if symmetries are present. Default is `nothing`.
- `alpert_order`: The order of the Alpert quadrature correction to use for near interactions. Supported values are 2, 3, 4, 5, 6, 8, 10, 12, 14, and 16. Default is 12.
- `alpertq`: The grading strength parameter for the Alpert quadrature. Default is 4.

# Output:
- An instance of the `CFIE_alpert` solver initialized with the provided parameters.
"""
function CFIE_alpert(pts_scaling_factor::Union{T,Vector{T}},billiard::Bi;min_pts::Int=20,eps::T=T(1e-15),symmetry::Union{Nothing,AbsSymmetry}=nothing,alpert_order::Int=12,alpertq::Int=4) where {T<:Real,Bi<:AbsBilliard}
    !(alpert_order in (2,3,4,5,6,8,10,12,14,16)) && error("Alpert order not currently supported")
    _warn_aggressive_alpert(pts_scaling_factor,billiard,alpert_order,alpertq)
    _=alpert_log_rule(T,alpert_order)
    bs=pts_scaling_factor isa T ? [pts_scaling_factor] : pts_scaling_factor
    sampler=[LinearNodes()]
    Sym=typeof(symmetry)
    return CFIE_alpert{T,Bi,Sym}(sampler,bs,bs[1],eps,min_pts,min_pts,billiard,symmetry,alpert_order,alpertq)
end

#### use N even for the algorithm - equidistant parameters ####
s(k::Int,N::Int)=two_pi*k/N

# reverse all BoundaryPointsCFIE except 1st as they correspond to holes in the outer domain.
# this function is really tricky since we need to reverse the order of the points but also flip the tangents and ds to maintain the correct orientation for the holes. We also need to be careful with the periodicity and the weights. The compid should remain unchanged since we are just reversing the order of points within the same component. Closed periodic polar curves ts behave differently from open panels due to the definiiton of the log analytic split for Kress needing [s(j,N) for j in 1:N] while for Alpert we chose midpoints.
function _reverse_component_orientation(solver::S,pts::BoundaryPointsCFIE{T}) where {T<:Real,S<:CFIE}
    xy=reverse(pts.xy)
    tangent=reverse(-pts.tangent)
    tangent_2=reverse(pts.tangent_2)
    ts=reverse(pts.ts)
    ws=reverse(pts.ws)
    ws_der=reverse(pts.ws_der)
    ds=reverse(pts.ds)
    xL=pts.xR
    xR=pts.xL
    tL=-pts.tR
    tR=-pts.tL
    return BoundaryPointsCFIE(xy,tangent,tangent_2,ts,ws,ws_der,ds,pts.compid,pts.is_periodic,xL,xR,tL,tR)
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
    return BoundaryPointsCFIE(xy,tangent_1st,tangent_2nd,ts,ws,ws_der,ds,idx,true,SVector(zero(T),zero(T)),SVector(zero(T),zero(T)),SVector(zero(T),zero(T)),SVector(zero(T),zero(T)))
end

#############################
#### KRESS SINGLE CORNER ####
#############################

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
    tangent_2nd=[γuu[i]*(jac[i]/two_pi)^2+γu[i]*(jac2[i]/two_pi) for i in eachindex(u)]
    ss=arc_length(crv,u)
    ds=diff(ss)
    append!(ds,L+ss[1]-ss[end])
    h=pi/T((N+1)÷2)
    ts=σ
    ws=fill(h,N)
    ws_der=jac
    return BoundaryPointsCFIE(xy,tangent_1st,tangent_2nd,ts,ws,ws_der,ds,idx,true,SVector(zero(T),zero(T)),SVector(zero(T),zero(T)),SVector(zero(T),zero(T)),SVector(zero(T),zero(T)))
end

############################
#### KRESS MULTI CORNER ####
############################

# Evaluate one closed composite boundary component using
# global multi-corner Kress grading.
#
# Steps:
#   1. Compute total boundary length Ltot
#   2. Choose total number of nodes N ~ k * Ltot
#   3. Build corner locations from segment joins
#   4. Generate graded nodes σ_k and map s = w(σ)
#   5. Evaluate geometry at each s_k via segment mapping
#   6. Apply chain rule to combine geometry + grading
#   7. Compute arc-length increments ds
#   8. Return BoundaryPointsCFIE object
#
# Output:
#   One BoundaryPointsCFIE representing the entire component - flag is_periodic=true since it's a closed curve.
#
# This replaces per-segment discretization with a single
# global discretization, which is required for Kress splitting.
function _evaluate_points(solver::CFIE_kress_global_corners{T},comp::Vector{C},k::T,idx::Int) where {T<:Real,C<:AbsCurve}
    # total length
    _,_,Ltot=component_lengths(comp)
    # choose number of nodes
    bs=solver.pts_scaling_factor
    N=max(solver.min_pts,round(Int,k*Ltot*bs[1]/two_pi))
    # enforce symmetry compatibility - like above 
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
    iseven(N) && (N+=needed)  # enforce odd N
    # build corner locations
    corners=_component_corner_locations(T,comp)
    # global graded nodes
    σ,tmap,jac,jac2,_=multi_kress_graded_nodes_data(T,N,corners;q=solver.kressq)
    # allocate outputs
    xy=Vector{SVector{2,T}}(undef,N)
    tangent_1st=Vector{SVector{2,T}}(undef,N)
    tangent_2nd=Vector{SVector{2,T}}(undef,N)
    @inbounds for i in 1:N
        q,γt,γtt=_eval_composite_geom_global_t(T,comp,tmap[i])
        # combine geometry derivatives with grading derivatives
        xy[i]=q
        tangent_1st[i]=γt*jac[i]
        tangent_2nd[i]=γtt*(jac[i]^2)+γt*jac2[i]
    end
    # compute arc-length increments ds
    ss=zeros(T,N)
    @inbounds for i in 2:N
        dx=xy[i][1]-xy[i-1][1]
        dy=xy[i][2]-xy[i-1][2]
        ss[i]=ss[i-1]+hypot(dx,dy)
    end
    ds=diff(ss)
    append!(ds,Ltot+ss[1]-ss[end])  # periodic closure
    # Kress weights
    h=pi/T((N+1)÷2)
    ts=σ  # computational nodes
    ws=fill(h,N) # trapezoidal weights
    ws_der=jac # w'(σ)
    return BoundaryPointsCFIE(xy,tangent_1st,tangent_2nd,ts,ws,ws_der,ds,idx,true,SVector(zero(T),zero(T)),SVector(zero(T),zero(T)),SVector(zero(T),zero(T)),SVector(zero(T),zero(T)))
end

####################
#### HIGH LEVEL ####
####################

function evaluate_points(solver::Union{CFIE_kress{T},CFIE_kress_corners{T}},billiard::Bi,k::T) where {T<:Real,Bi<:AbsBilliard}
    comps=_boundary_components(billiard.full_boundary)
    pts=Vector{BoundaryPointsCFIE{T}}(undef,length(comps))
    for (idx,comp) in enumerate(comps)
        length(comp)==1 || error("Kress requires each boundary component to be a single smooth closed curve.")
        crv=comp[1]
        p=_evaluate_points(solver,crv,k,idx)
        pts[idx]=idx==1 ? p : _reverse_component_orientation(solver,p)
    end
    return pts
end

function evaluate_points(solver::CFIE_kress_global_corners{T},billiard::Bi,k::T) where {T<:Real,Bi<:AbsBilliard}
    comps=_boundary_components(billiard.full_boundary)
    pts=Vector{BoundaryPointsCFIE{T}}(undef,length(comps))
    for (idx,comp) in enumerate(comps)
        isempty(comp) && error("Boundary component cannot be empty.")
        if length(comp)==1
            p=_evaluate_points(CFIE_kress_corners(solver.pts_scaling_factor,solver.billiard;min_pts=solver.min_pts,eps=solver.eps,symmetry=solver.symmetry,kressq=solver.kressq),comp[1],k,idx)
        else
            p=_evaluate_points(solver,comp,k,idx)
        end
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
    #ts=[s(j,N) for j in 1:N]
    ts=[T(two_pi)*(j-T(1)/2)/T(N) for j in 1:N]
    ts_rescaled=ts./two_pi
    xy=curve(crv,ts_rescaled)
    tangent_1st=tangent(crv,ts_rescaled)./(two_pi)
    tangent_2nd=tangent_2(crv,ts_rescaled)./(two_pi)^2
    ss=arc_length(crv,ts_rescaled)
    ds=diff(ss)
    append!(ds,L+ss[1]-ss[end])
    ws=fill(T(two_pi/N),N)
    ws_der=ones(T,N)
    return BoundaryPointsCFIE(xy,tangent_1st,tangent_2nd,ts,ws,ws_der,ds,idx,true,SVector(zero(T),zero(T)),SVector(zero(T),zero(T)),SVector(zero(T),zero(T)),SVector(zero(T),zero(T)))
end

# σ → (u, du/dσ, d²u/dσ²); map computational nodes to physical nodes and return Jacobian info for transforming geometry derivatives
@inline function _panel_sigma_to_u_jac(solver::CFIE_alpert{T},σ::T) where {T<:Real}
    q=solver.alpertq   # acts as grading strength parameter
    u=_panel_grade_map(σ,q)
    jac=_panel_grade_map_prime(σ,q)
    jac2=_panel_grade_map_doubleprime(σ,q)
    return u,jac,jac2
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
    hσ=inv(T(N))
    sig=[T(j-0.5)/T(N) for j in 1:N]
    xy=Vector{SVector{2,T}}(undef,N)
    tangent_1st=Vector{SVector{2,T}}(undef,N)
    tangent_2nd=Vector{SVector{2,T}}(undef,N)
    ds=Vector{T}(undef,N)
    @inbounds for j in 1:N
        σ=sig[j]
        u,jac,jac2=_panel_sigma_to_u_jac(solver,σ)
        q=curve(crv,u)
        tu=tangent(crv,u)
        t2u=tangent_2(crv,u)
        xy[j]=q
        tangent_1st[j]=tu*jac
        tangent_2nd[j]=t2u*(jac^2)+tu*jac2
        ds[j]=sqrt((tu[1]*jac)^2+(tu[2]*jac)^2)*hσ
    end
    ws=fill(hσ,N)
    ws_der=ones(T,N)
    xL=curve(crv,zero(T))
    xR=curve(crv,one(T))
    tL=tangent(crv,zero(T))
    tR=tangent(crv,one(T))
    return BoundaryPointsCFIE(xy,tangent_1st,tangent_2nd,sig,ws,ws_der,ds,idx,false,xL,xR,tL,tR)
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