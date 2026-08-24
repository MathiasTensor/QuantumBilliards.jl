# Useful reading:
# - https://github.com/ahbarnett/mpspack - by Alex Barnett & Timo Betcke (MATLAB)
# - Kress, R., Boundary integral equations in time-harmonic acoustic scattering. Mathematics Comput. Modelling Vol 15, pp. 229-243). Pergamon Press, 1991, GB.
# - Barnett, A. H., & Betcke, T. (2007). Stability and convergence of the method of fundamental solutions for Helmholtz problems on analytic domains. Journal of Computational Physics, 227(14), 7003-7026.
# - Zhao, L., & Barnett, A. (2015). Robust and efficient solution of the drum problem via Nyström approximation of the Fredholm determinant. SIAM Journal on Numerical Analysis, Stable URL: https://www.jstor.org/stable/24512689

const euler_over_pi=MathConstants.eulergamma/pi

############################
#### CONSTRUCTOR KRESS ######
############################

"""
    CFIE_kress{T,Bi,Sym} <: CFIE

Combined-field integral equation solver using Kress periodic logarithmic
splitting on smooth closed boundary components.

## Description
`CFIE_kress` implements the smooth-boundary Kress version of the combined-field
Fredholm formulation. Each connected boundary component must be represented by
one smooth closed periodic curve.

The discretized operator has the form

    A(k)=I-(D(k)+ikS(k)),

where `D(k)` is the Helmholtz double-layer operator and `S(k)` is the
single-layer operator.

For same-component interactions, both kernels are decomposed into

    logarithmic coefficient × log(4sin²((t-s)/2))
    + smooth remainder.

The universal periodic logarithmic contribution is integrated using the Kress
correction matrix, while the smooth remainder is evaluated with periodic
trapezoidal quadrature.

The combined-field term removes the spurious interior-nullspace problem that can
occur for a pure double-layer formulation on multiply connected geometries.

## Attributes
* `sampler`: Placeholder periodic sampler retained for the common solver API.
* `pts_scaling_factor`: Boundary-resolution scaling factors.
* `dim_scaling_factor`: Compatibility field for the generic solver interface.
* `eps`: Numerical tolerance placeholder.
* `min_dim`: Compatibility field for the generic solver interface.
* `min_pts`: Minimum number of points on each boundary component.
* `billiard`: Underlying billiard geometry.
* `symmetry`: Optional symmetry descriptor.

## Notes
For a boundary component of length `L`, the nominal node count is

    N ≈ k*L*b/(2π),

where `b` is the corresponding boundary-resolution scaling factor.

Use this solver when every connected component is represented by one smooth
closed curve. Composite piecewise smooth components should instead use
[`CFIE_kress_global_corners`](@ref).
"""
struct CFIE_kress{T<:Real,Bi<:AbsBilliard,Sym}<:CFIE
    sampler::Vector{LinearNodes}
    pts_scaling_factor::Vector{T}
    dim_scaling_factor::T
    eps::T
    min_dim::Int64
    min_pts::Int64
    billiard::Bi
    symmetry::Sym
end

"""
    CFIE_kress_corners{T,Bi,Sym} <: CFIE

Combined-field integral equation solver using a Kress grading transformation on
a single closed curve with corner-type parameter singularities.

## Description
This is the graded counterpart of [`CFIE_kress`](@ref) for a boundary component
that remains represented by one closed parameterized curve but requires
endpoint/corner clustering.

A uniform computational variable `σ` is transformed through a nonlinear map

    t=w(σ),

and the geometric derivatives are transformed by the chain rule. The resulting
operator remains

    A(k)=I-(D(k)+ikS(k)).

The Kress logarithmic splitting is performed in the computational periodic
variable.

## Attributes
* `sampler`: Placeholder periodic sampler retained for the common solver API.
* `pts_scaling_factor`: Boundary-resolution scaling factors.
* `dim_scaling_factor`: Compatibility field for the generic solver interface.
* `eps`: Numerical tolerance placeholder.
* `min_dim`: Compatibility field for the generic solver interface.
* `min_pts`: Minimum number of boundary points.
* `billiard`: Underlying billiard geometry.
* `symmetry`: Optional symmetry descriptor.
* `kressq`: Order of the Kress grading transformation.
* `min_t_spacing`: Minimum permitted physical-parameter spacing after grading.

## Notes
For `Float64`, excessively large grading orders may force neighboring mapped
nodes below machine-resolvable spacing. A grading order around `4` is therefore
the practical default.
"""
struct CFIE_kress_corners{T<:Real,Bi<:AbsBilliard,Sym}<:CFIE
    sampler::Vector{LinearNodes}
    pts_scaling_factor::Vector{T}
    dim_scaling_factor::T
    eps::T
    min_dim::Int64
    min_pts::Int64
    billiard::Bi
    symmetry::Sym
    kressq::Int
    min_t_spacing::Real
end

"""
    CFIE_kress_global_corners{T,Bi,Sym} <: CFIE

Combined-field integral equation solver using global periodic Kress grading on
closed composite boundary components.

## Description
This is the general Kress solver for closed components represented by several
joined curve segments.

For each connected component, all constituent segments are treated as one
global periodic boundary. True corner locations are detected from tangent jumps.
If corners are present, a global Kress grading map is constructed relative to
all corner locations simultaneously.

If the joins are smooth, the component is instead discretized on an ungraded
uniform periodic mesh.

The resulting Fredholm operator is

    A(k)=I-(D(k)+ikS(k)).

Treating each closed component globally is essential because the periodic
logarithmic singularity belongs to the complete closed boundary rather than to
the individual constituent segments.

## Attributes
* `sampler`: Placeholder periodic sampler retained for the common solver API.
* `pts_scaling_factor`: Boundary-resolution scaling factors.
* `dim_scaling_factor`: Compatibility field for the generic solver interface.
* `eps`: Numerical tolerance placeholder.
* `min_dim`: Compatibility field for the generic solver interface.
* `min_pts`: Minimum number of boundary points.
* `billiard`: Underlying billiard geometry.
* `symmetry`: Optional symmetry descriptor.
* `kressq`: Global Kress grading order.
* `min_t_spacing`: Minimum permitted physical-parameter spacing after grading.

## Notes
This solver supports composite outer boundaries and composite hole boundaries.
Each connected component is discretized as one periodic object.
"""
struct CFIE_kress_global_corners{T<:Real,Bi<:AbsBilliard,Sym}<:CFIE
    sampler::Vector{LinearNodes}
    pts_scaling_factor::Vector{T}
    dim_scaling_factor::T
    eps::T
    min_dim::Int64
    min_pts::Int64
    billiard::Bi
    symmetry::Sym
    kressq::Int
    min_t_spacing::Real
end

"""
    CFIE_kress(pts_scaling_factor,billiard;min_pts=20,eps=1e-15,symmetry=nothing) → solver::CFIE_kress

Constructs a smooth periodic Kress combined-field solver.

## Arguments
* `pts_scaling_factor`: Boundary-resolution scaling factor or vector of factors.
* `billiard`: Billiard geometry.

## Keyword arguments
* `min_pts`: Minimum number of points on each boundary component.
* `eps`: Numerical tolerance placeholder.
* `symmetry`: Optional reflection or rotation symmetry descriptor.

## Returns
* `solver`: Configured [`CFIE_kress`](@ref) instance.
"""
function CFIE_kress(pts_scaling_factor::Union{T,Vector{T}},billiard::Bi;min_pts=20,eps=T(1e-15),symmetry::Union{Nothing,AbsSymmetry}=nothing) where {T<:Real,Bi<:AbsBilliard}
    bs=pts_scaling_factor isa T ? [pts_scaling_factor] : pts_scaling_factor
    sampler=[LinearNodes()]
    Sym=typeof(symmetry)
    return CFIE_kress{T,Bi,Sym}(sampler,bs,bs[1],eps,min_pts,min_pts,billiard,symmetry)
end

"""
    CFIE_kress_corners(pts_scaling_factor,billiard;min_pts=20,eps=1e-15,symmetry=nothing,kressq=4,min_t_spacing=1e-12) → solver::CFIE_kress_corners

Constructs a single-curve corner-graded Kress combined-field solver.

## Arguments
* `pts_scaling_factor`: Boundary-resolution scaling factor or vector of factors.
* `billiard`: Billiard geometry.

## Keyword arguments
* `min_pts`: Minimum number of boundary points.
* `eps`: Numerical tolerance placeholder.
* `symmetry`: Optional reflection or rotation symmetry descriptor.
* `kressq`: Kress grading order.
* `min_t_spacing`: Minimum permitted mapped-parameter spacing.

## Returns
* `solver`: Configured [`CFIE_kress_corners`](@ref) instance.

## Notes
For `Float64`, values of `kressq` substantially larger than `4` should be used
with care because the graded nodes may become indistinguishable at machine
precision.
"""
function CFIE_kress_corners(pts_scaling_factor::Union{T,Vector{T}},billiard::Bi;min_pts=20,eps=T(1e-15),symmetry::Union{Nothing,AbsSymmetry}=nothing,kressq=4,min_t_spacing=1e-12) where {T<:Real,Bi<:AbsBilliard}
    bs=pts_scaling_factor isa T ? [pts_scaling_factor] : pts_scaling_factor
    sampler=[LinearNodes()]
    Sym=typeof(symmetry)
    return CFIE_kress_corners{T,Bi,Sym}(sampler,bs,bs[1],eps,min_pts,min_pts,billiard,symmetry,kressq,min_t_spacing)
end

"""
    CFIE_kress_global_corners(pts_scaling_factor,billiard;min_pts=20,eps=1e-15,symmetry=nothing,kressq=4,min_t_spacing=1e-12) → solver::CFIE_kress_global_corners

Constructs a globally graded Kress combined-field solver for composite boundary
components.

## Arguments
* `pts_scaling_factor`: Boundary-resolution scaling factor or vector of factors.
* `billiard`: Billiard geometry.

## Keyword arguments
* `min_pts`: Minimum number of boundary points.
* `eps`: Numerical tolerance placeholder.
* `symmetry`: Optional reflection or rotation symmetry descriptor.
* `kressq`: Global Kress grading order.
* `min_t_spacing`: Minimum permitted mapped-parameter spacing.

## Returns
* `solver`: Configured [`CFIE_kress_global_corners`](@ref) instance.
"""
function CFIE_kress_global_corners(pts_scaling_factor::Union{T,Vector{T}},billiard::Bi;min_pts=20,eps=T(1e-15),symmetry::Union{Nothing,AbsSymmetry}=nothing,kressq=4,min_t_spacing=1e-12) where {T<:Real,Bi<:AbsBilliard}
    bs=pts_scaling_factor isa T ? [pts_scaling_factor] : pts_scaling_factor
    sampler=[LinearNodes()]
    Sym=typeof(symmetry)
    return CFIE_kress_global_corners{T,Bi,Sym}(sampler,bs,bs[1],eps,min_pts,min_pts,billiard,symmetry,kressq,min_t_spacing)
end

#############################
#### CONSTRUCTOR ALPERT ######
#############################

"""
    CFIE_alpert{T,Bi,Sym} <: CFIE

Combined-field integral equation solver using Alpert hybrid
Gauss-trapezoidal singular quadrature on panelized boundaries.

## Description
Unlike the global periodic Kress approach, this solver treats the boundary
panel-by-panel and applies local Alpert corrections to singular and
near-singular interactions.

The underlying Fredholm operator remains

    A(k)=I-(D(k)+ikS(k)).

The panel-based formulation is especially suitable for polygonal and other
piecewise smooth boundaries with true corners, where a global periodic
parameterization is less natural.

## Attributes
* `sampler`: Placeholder sampler retained for the common solver API.
* `pts_scaling_factor`: Boundary-resolution scaling factors.
* `dim_scaling_factor`: Compatibility field for the generic solver interface.
* `eps`: Numerical tolerance placeholder.
* `min_dim`: Compatibility field for the generic solver interface.
* `min_pts`: Minimum number of nodes per panel.
* `billiard`: Underlying billiard geometry.
* `symmetry`: Optional symmetry descriptor.
* `alpert_order`: Order of the Alpert correction rule.
* `alpertq`: Endpoint grading order used on individual panels.
* `min_t_spacing`: Minimum permitted mapped panel-parameter spacing.
"""
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
    min_t_spacing::Real
end

"""
    CFIE_alpert(pts_scaling_factor,billiard;min_pts=20,eps=1e-15,symmetry=nothing,alpert_order=12,alpertq=4,min_t_spacing=1e-12) → solver::CFIE_alpert

Constructs an Alpert-corrected panel-based combined-field solver.

## Arguments
* `pts_scaling_factor`: Boundary-resolution scaling factor or vector of factors.
* `billiard`: Billiard geometry.

## Keyword arguments
* `min_pts`: Minimum number of nodes per panel.
* `eps`: Numerical tolerance placeholder.
* `symmetry`: Optional symmetry descriptor.
* `alpert_order`: Order of the Alpert hybrid quadrature correction.
* `alpertq`: Endpoint grading order.
* `min_t_spacing`: Minimum permitted mapped panel-parameter spacing.

## Returns
* `solver`: Configured [`CFIE_alpert`](@ref) instance.

## Notes
Supported Alpert orders are `2`, `3`, `4`, `5`, `6`, `8`, `10`, `12`, `14`
and `16`.
"""
function CFIE_alpert(pts_scaling_factor::Union{T,Vector{T}},billiard::Bi;min_pts::Int=20,eps::T=T(1e-15),symmetry::Union{Nothing,AbsSymmetry}=nothing,alpert_order::Int=12,alpertq::Int=4,min_t_spacing=1e-12) where {T<:Real,Bi<:AbsBilliard}
    alpert_order in (2,3,4,5,6,8,10,12,14,16)||error("Alpert order not currently supported")
    bs=pts_scaling_factor isa T ? [pts_scaling_factor] : pts_scaling_factor
    sampler=[LinearNodes()]
    Sym=typeof(symmetry)
    return CFIE_alpert{T,Bi,Sym}(sampler,bs,bs[1],eps,min_pts,min_pts,billiard,symmetry,alpert_order,alpertq,min_t_spacing)
end

#### Equispaced periodic parameters ####
@inline s(k::Int,N::Int)=two_pi*k/N
@inline s_mid(k::Int,N::Int)=two_pi*(k-0.5)/N

"""
    _reverse_component_orientation(solver::CFIE,pts::BoundaryPoints{T}) where {T<:Real} → pts_reversed::BoundaryPoints{T}

Reverses the orientation of one boundary component.

## Description
For multiply connected geometries, the outer boundary and hole boundaries must
carry opposite orientations.

The point ordering is reversed together with the first and second derivative
data. First derivatives change sign under orientation reversal, while second
derivatives retain their sign.

The endpoint metadata of open panels is also exchanged consistently.

The returned [`BoundaryPoints`](@ref) object automatically reconstructs its
outward-normal data from the reversed tangent orientation.

## Arguments
* `solver`: Combined-field solver.
* `pts`: Boundary component to reverse.

## Returns
* `pts_reversed`: New [`BoundaryPoints`](@ref) instance with reversed orientation.
"""
function _reverse_component_orientation(solver::S,pts::BoundaryPoints{T}) where {T<:Real,S<:CFIE}
    xy=reverse(pts.xy)
    tangent=reverse(-pts.tangent)
    tangent_2=reverse(pts.tangent_2)
    ts=reverse(pts.ts)
    tphys=reverse(pts.tphys)
    ws=reverse(pts.ws)
    ws_der=reverse(pts.ws_der)
    ds=reverse(pts.ds)
    xL=pts.xR
    xR=pts.xL
    tL=-pts.tR
    tR=-pts.tL
    return BoundaryPoints(xy,tangent,tangent_2,ts,tphys,ws,ws_der,ds,pts.compid,pts.is_periodic,xL,xR,tL,tR)
end

###############
#### KRESS ####
###############

"""
    _evaluate_points(solver::CFIE_kress{T},crv::C,k::T,idx::Int) where {T<:Real,C<:AbsCurve} → pts::BoundaryPoints{T}

Constructs the smooth periodic Kress discretization of one closed boundary
component.

## Description
For a component of length `L`, the nominal number of nodes is

    N ≈ k*L*b/(2π),

subject to the minimum node count and symmetry compatibility.

The Kress computational variable lies in `[0,2π)`, whereas the geometry curves
are parameterized on `[0,1]`. Therefore

    u=t/(2π),

and the geometric derivatives transform as

    γ_t=γ_u/(2π),

and

    γ_tt=γ_uu/(2π)².

The periodic quadrature weights are

    ws_j=2π/N.

Since no grading is applied,

    ws_der_j=1.

## Arguments
* `solver`: Smooth periodic Kress solver.
* `crv`: Smooth closed boundary curve.
* `k`: Real wavenumber controlling the node density.
* `idx`: Connected-component identifier.

## Returns
* `pts`: Periodic [`BoundaryPoints`](@ref) discretization.
"""
function _evaluate_points(solver::CFIE_kress{T},crv::C,k::T,idx::Int) where {T<:Real,C<:AbsCurve}
    L=crv.length
    bs=solver.pts_scaling_factor
    N=max(solver.min_pts,round(Int,k*L*bs[1]/two_pi))
    needed=2
    if !isnothing(solver.symmetry)
        sym=solver.symmetry
        if sym isa Rotation
            needed=lcm(needed,sym.n)
        elseif hasproperty(sym,:axis)
            needed=lcm(needed,4)
        end
    end
    remN=mod(N,needed)
    remN!=0&&(N+=needed-remN)
    ts=[s_mid(j,N) for j in 1:N]
    ts_rescaled=ts./two_pi
    xy=curve(crv,ts_rescaled)
    tangent_1st=tangent(crv,ts_rescaled)./two_pi
    tangent_2nd=tangent_2(crv,ts_rescaled)./(two_pi^2)
    ss=arc_length(crv,ts_rescaled)
    ds=diff(ss)
    append!(ds,L+ss[1]-ss[end])
    ws=fill(T(two_pi/N),N)
    ws_der=ones(T,N)
    z=SVector(zero(T),zero(T))
    return BoundaryPoints(xy,tangent_1st,tangent_2nd,ts,copy(ts),ws,ws_der,ds,idx,true,z,z,z,z)
end

"""
    _evaluate_points(solver::CFIE_kress_corners{T},crv::C,k::T,idx::Int) where {T<:Real,C<:AbsCurve} → pts::BoundaryPoints{T}

Constructs a Kress-graded periodic discretization of one closed curve.

## Description
The uniform computational variable `σ` is mapped to a physical periodic
parameter

    t=t(σ),

with first and second derivatives `jac` and `jac2`.

Because the underlying geometry parameter is

    u=t/(2π),

the transformed derivatives are

    γ_σ=γ_u*jac/(2π),

and

    γ_σσ=γ_uu*(jac/(2π))²+γ_u*jac2/(2π).

The grading Jacobian is retained in `ws_der`.

## Arguments
* `solver`: Single-curve corner-graded Kress solver.
* `crv`: Closed boundary curve.
* `k`: Real wavenumber controlling the node density.
* `idx`: Connected-component identifier.

## Returns
* `pts`: Graded periodic [`BoundaryPoints`](@ref) discretization.
"""
function _evaluate_points(solver::CFIE_kress_corners{T},crv::C,k::T,idx::Int) where {T<:Real,C<:AbsCurve}
    L=crv.length
    bs=solver.pts_scaling_factor
    N=max(solver.min_pts,round(Int,k*L*bs[1]/two_pi))
    needed=1
    if !isnothing(solver.symmetry)
        sym=solver.symmetry
        if sym isa Rotation
            needed=lcm(needed,sym.n)
        elseif hasproperty(sym,:axis)
            needed=lcm(needed,4)
        end
    end
    remN=mod(N,needed)
    remN!=0&&(N+=needed-remN)
    σ,tmap,jac,jac2,_=kress_graded_nodes_data(T,N;q=solver.kressq,minsep_tol=solver.min_t_spacing)
    u=tmap./two_pi
    xy=curve(crv,u)
    γu=tangent(crv,u)
    γuu=tangent_2(crv,u)
    tangent_1st=[γu[i]*(jac[i]/two_pi) for i in eachindex(u)]
    tangent_2nd=[γuu[i]*(jac[i]/two_pi)^2+γu[i]*(jac2[i]/two_pi) for i in eachindex(u)]
    ss=arc_length(crv,u)
    ds=diff(ss)
    append!(ds,L+ss[1]-ss[end])
    h=T(two_pi/N)
    ws=fill(h,N)
    z=SVector(zero(T),zero(T))
    return BoundaryPoints(xy,tangent_1st,tangent_2nd,σ,tmap,ws,jac,ds,idx,true,z,z,z,z)
end

############################
#### KRESS MULTI CORNER ####
############################

"""
    _evaluate_points_smooth_composite(solver::CFIE_kress_global_corners{T},comp::Vector{C},k::T,idx::Int) where {T<:Real,C<:AbsCurve} → pts::BoundaryPoints{T}

Constructs an ungraded periodic discretization of one smooth composite boundary
component.

## Description
This is the fallback used by [`CFIE_kress_global_corners`](@ref) when several
joined curve pieces form one closed component but all junctions are smooth.

A global parameter

    t∈[0,2π)

is used for the complete component and evaluated through
[`_eval_composite_geom_global_t`](@ref).

Since no grading is active, the quadrature is the ordinary periodic
trapezoidal rule.

## Arguments
* `solver`: Global-composite Kress solver.
* `comp`: Curve pieces forming one smooth closed component.
* `k`: Real wavenumber controlling the node density.
* `idx`: Connected-component identifier.

## Returns
* `pts`: Ungraded periodic [`BoundaryPoints`](@ref) discretization.
"""
function _evaluate_points_smooth_composite(solver::CFIE_kress_global_corners{T},comp::Vector{C},k::T,idx::Int) where {T<:Real,C<:AbsCurve}
    _,_,Ltot=component_lengths(comp)
    bs=solver.pts_scaling_factor
    N=max(solver.min_pts,round(Int,k*Ltot*bs[1]/two_pi))
    needed=2
    if !isnothing(solver.symmetry)
        sym=solver.symmetry
        if sym isa Rotation
            needed=lcm(needed,sym.n)
        elseif hasproperty(sym,:axis)
            needed=lcm(needed,4)
        end
    end
    remN=mod(N,needed)
    remN!=0&&(N+=needed-remN)
    ts=[s_mid(j,N) for j in 1:N]
    h=T(two_pi)/T(N)
    xy=Vector{SVector{2,T}}(undef,N)
    tangent_1st=Vector{SVector{2,T}}(undef,N)
    tangent_2nd=Vector{SVector{2,T}}(undef,N)
    ds=Vector{T}(undef,N)
    @inbounds for i in 1:N
        q,γt,γtt=_eval_composite_geom_global_t(T,comp,ts[i])
        xy[i]=q
        tangent_1st[i]=γt
        tangent_2nd[i]=γtt
        ds[i]=hypot(γt[1],γt[2])*h
    end
    ws=fill(h,N)
    ws_der=ones(T,N)
    z=SVector(zero(T),zero(T))
    return BoundaryPoints(xy,tangent_1st,tangent_2nd,ts,copy(ts),ws,ws_der,ds,idx,true,z,z,z,z)
end

"""
    _evaluate_points(solver::CFIE_kress_global_corners{T},comp::Vector{C},k::T,idx::Int) where {T<:Real,C<:AbsCurve} → pts::BoundaryPoints{T}

Constructs the global periodic Kress discretization of a composite closed
boundary component.

## Description
True corner locations are detected by
[`_component_corner_locations`](@ref).

If no true corners are present, the function delegates to
[`_evaluate_points_smooth_composite`](@ref).

Otherwise the global computational coordinate `σ` is transformed through a
multi-corner grading map

    t=t(σ).

For global composite derivatives,

    γ_σ=γ_t t_σ,

and

    γ_σσ=γ_tt(t_σ)²+γ_t t_σσ.

The physical arc-length quadrature element is

    ds_j=|γ_σ|*2π/N.

## Arguments
* `solver`: Global-composite Kress solver.
* `comp`: Curve segments forming one closed component.
* `k`: Real wavenumber controlling the node density.
* `idx`: Connected-component identifier.

## Returns
* `pts`: Globally graded [`BoundaryPoints`](@ref) discretization.
"""
function _evaluate_points(solver::CFIE_kress_global_corners{T},comp::Vector{C},k::T,idx::Int) where {T<:Real,C<:AbsCurve}
    corners=_component_corner_locations(T,comp)
    isempty(corners)&&return _evaluate_points_smooth_composite(solver,comp,k,idx)
    _,_,Ltot=component_lengths(comp)
    bs=solver.pts_scaling_factor
    N=max(solver.min_pts,round(Int,k*Ltot*bs[1]/two_pi))
    needed=1
    if !isnothing(solver.symmetry)
        sym=solver.symmetry
        if sym isa Rotation
            needed=lcm(needed,sym.n)
        elseif hasproperty(sym,:axis)
            needed=lcm(needed,4)
        end
    end
    remN=mod(N,needed)
    remN!=0&&(N+=needed-remN)
    σ,tmap,jac,jac2,_=multi_kress_graded_nodes_data(T,N,corners;q=solver.kressq,minsep_tol=solver.min_t_spacing)
    xy=Vector{SVector{2,T}}(undef,N)
    tangent_1st=Vector{SVector{2,T}}(undef,N)
    tangent_2nd=Vector{SVector{2,T}}(undef,N)
    @inbounds for i in 1:N
        q,γt,γtt=_eval_composite_geom_global_t(T,comp,tmap[i])
        xy[i]=q
        tangent_1st[i]=γt*jac[i]
        tangent_2nd[i]=γtt*(jac[i]^2)+γt*jac2[i]
    end
    h=T(two_pi)/T(N)
    ds=Vector{T}(undef,N)
    @inbounds for i in 1:N
        ds[i]=hypot(tangent_1st[i][1],tangent_1st[i][2])*h
    end
    ws=fill(h,N)
    z=SVector(zero(T),zero(T))
    return BoundaryPoints(xy,tangent_1st,tangent_2nd,σ,tmap,ws,jac,ds,idx,true,z,z,z,z)
end

####################
#### HIGH LEVEL ####
####################

"""
    evaluate_points(solver::Union{CFIE_kress{T},CFIE_kress_corners{T}},billiard::Bi,k::T) where {T<:Real,Bi<:AbsBilliard} → pts::Vector{BoundaryPoints{T}}

Constructs the Kress boundary discretizations of all connected boundary
components.

## Description
Every connected component must consist of exactly one closed curve.

The first component is interpreted as the outer boundary. Every subsequent
component is interpreted as a hole and has its orientation reversed through
[`_reverse_component_orientation`](@ref).

## Arguments
* `solver`: Smooth or single-curve graded Kress solver.
* `billiard`: Billiard geometry.
* `k`: Real wavenumber controlling the boundary resolution.

## Returns
* `pts`: Vector of [`BoundaryPoints`](@ref), one per connected component.

## Notes
Composite multi-segment components require
[`CFIE_kress_global_corners`](@ref).
"""
function evaluate_points(solver::Union{CFIE_kress{T},CFIE_kress_corners{T}},billiard::Bi,k::T) where {T<:Real,Bi<:AbsBilliard}
    comps=_boundary_components(billiard.full_boundary)
    pts=Vector{BoundaryPoints{T}}(undef,length(comps))
    for (idx,comp) in enumerate(comps)
        length(comp)==1||error("Periodic Kress requires each boundary component to be represented by one closed curve. Use CFIE_kress_global_corners for composite components.")
        p=_evaluate_points(solver,comp[1],k,idx)
        pts[idx]=idx==1 ? p : _reverse_component_orientation(solver,p)
    end
    return pts
end

"""
    evaluate_points(solver::CFIE_kress_global_corners{T},billiard::Bi,k::T) where {T<:Real,Bi<:AbsBilliard} → pts::Vector{BoundaryPoints{T}}

Constructs globally periodic Kress discretizations for all connected boundary
components.

## Description
The function supports three geometry layouts:

1. one smooth closed curve,
2. one composite closed outer boundary,
3. multiple connected components, each smooth or composite.

For multiply connected geometries, all components after the first are
orientation-reversed to represent holes.

Smooth one-curve components are delegated to [`CFIE_kress`](@ref). Composite
components are treated globally and graded only when true corners are detected.

## Arguments
* `solver`: Global-composite Kress solver.
* `billiard`: Billiard geometry.
* `k`: Real wavenumber controlling the boundary resolution.

## Returns
* `pts`: Vector of [`BoundaryPoints`](@ref), one per connected component.
"""
function evaluate_points(solver::CFIE_kress_global_corners{T},billiard::Bi,k::T) where {T<:Real,Bi<:AbsBilliard}
    boundary=billiard.full_boundary
    isempty(boundary)&&error("Boundary cannot be empty.")
    if length(boundary)==1&&!(boundary[1] isa AbstractVector)
        base=CFIE_kress(solver.pts_scaling_factor,solver.billiard;min_pts=solver.min_pts,eps=solver.eps,symmetry=solver.symmetry)
        return [_evaluate_points(base,boundary[1],k,1)]
    end
    if _is_single_composite_boundary(boundary)
        return [_evaluate_points(solver,boundary,k,1)]
    end
    comps=_boundary_components(boundary)
    pts=Vector{BoundaryPoints{T}}(undef,length(comps))
    for (idx,comp) in enumerate(comps)
        isempty(comp)&&error("Boundary component cannot be empty.")
        p=if length(comp)==1
            base=CFIE_kress(solver.pts_scaling_factor,solver.billiard;min_pts=solver.min_pts,eps=solver.eps,symmetry=solver.symmetry)
            _evaluate_points(base,comp[1],k,idx)
        else
            _evaluate_points(solver,comp,k,idx)
        end
        pts[idx]=idx==1 ? p : _reverse_component_orientation(solver,p)
    end
    return pts
end

################
#### ALPERT ####
################

"""
    _evaluate_points_periodic(solver::CFIE_alpert{T},crv::C,k::T,idx::Int) where {T<:Real,C<:AbsCurve} → pts::BoundaryPoints{T}

Constructs a periodic Alpert-compatible discretization of one smooth closed
curve.

## Description
Although the Alpert formulation is primarily panel based, smooth closed
components may be represented directly on a periodic midpoint grid.

The geometry curves are parameterized on `[0,1]`, while the computational
periodic variable lies in `[0,2π)`. Consequently,

    γ_t=γ_u/(2π),

and

    γ_tt=γ_uu/(2π)².

## Arguments
* `solver`: Alpert combined-field solver.
* `crv`: Smooth closed curve.
* `k`: Real wavenumber controlling the node density.
* `idx`: Connected-component identifier.

## Returns
* `pts`: Periodic [`BoundaryPoints`](@ref) discretization.
"""
function _evaluate_points_periodic(solver::CFIE_alpert{T},crv::C,k::T,idx::Int) where {T<:Real,C<:AbsCurve}
    L=crv.length
    bs=solver.pts_scaling_factor
    N=max(solver.min_pts,round(Int,k*L*bs[1]/two_pi))
    needed=1
    if !isnothing(solver.symmetry)
        sym=solver.symmetry
        if sym isa Rotation
            needed=lcm(needed,sym.n)
        elseif hasproperty(sym,:axis)
            needed=lcm(needed,4)
        end
    end
    remN=mod(N,needed)
    remN!=0&&(N+=needed-remN)
    ts=[T(two_pi)*(j-T(0.5))/T(N) for j in 1:N]
    ts_rescaled=ts./two_pi
    xy=curve(crv,ts_rescaled)
    tangent_1st=tangent(crv,ts_rescaled)./two_pi
    tangent_2nd=tangent_2(crv,ts_rescaled)./(two_pi^2)
    ss=arc_length(crv,ts_rescaled)
    ds=diff(ss)
    append!(ds,L+ss[1]-ss[end])
    ws=fill(T(two_pi/N),N)
    ws_der=ones(T,N)
    z=SVector(zero(T),zero(T))
    return BoundaryPoints(xy,tangent_1st,tangent_2nd,ts,copy(ts),ws,ws_der,ds,idx,true,z,z,z,z)
end

"""
    _panel_sigma_to_u_jac(solver::CFIE_alpert{T},σ::T,q::T) where {T<:Real} → u,jac,jac2

Evaluates the graded panel map and its first two derivatives.

## Description
The computational panel coordinate `σ∈[0,1]` is mapped to the physical curve
parameter

    u=u(σ).

The returned derivatives are used to transform the geometry by the chain rule.

## Arguments
* `solver`: Alpert combined-field solver.
* `σ`: Computational panel coordinate.
* `q`: Panel grading order.

## Returns
* `u`: Physical panel parameter.
* `jac`: First derivative `du/dσ`.
* `jac2`: Second derivative `d²u/dσ²`.
"""
@inline function _panel_sigma_to_u_jac(solver::CFIE_alpert{T},σ::T,q::T) where {T<:Real}
    u=_panel_grade_map(σ,q)
    jac=_panel_grade_map_prime(σ,q)
    jac2=_panel_grade_map_doubleprime(σ,q)
    return u,jac,jac2
end

@inline _panel_sigma_to_u_jac(solver::CFIE_alpert{T},σ::T) where {T<:Real}=_panel_sigma_to_u_jac(solver,σ,T(solver.alpertq))

"""
    _evaluate_points_panel(solver::CFIE_alpert{T},crv::C,k::T,idx::Int;minsep_tol=1e-12) where {T<:Real,C<:AbsCurve} → pts::BoundaryPoints{T}

Constructs one open graded panel for the Alpert combined-field method.

## Description
Midpoint nodes are first generated in the computational panel coordinate

    σ∈(0,1).

They are transformed through the panel grading map to the physical curve
parameter `u`.

The geometric derivatives are transformed as

    γ_σ=γ_u u_σ,

and

    γ_σσ=γ_uu(u_σ)²+γ_u u_σσ.

If the mapped physical parameter nodes become too closely spaced, the grading
order is reduced adaptively until the requested minimum spacing is satisfied or
the ungraded limit `q=1` is reached.

Because the resulting object represents an open panel, endpoint positions and
endpoint tangents are stored explicitly.

## Arguments
* `solver`: Alpert combined-field solver.
* `crv`: Curve segment representing one open panel.
* `k`: Real wavenumber controlling the node density.
* `idx`: Connected-component identifier.

## Keyword arguments
* `minsep_tol`: Minimum permitted physical panel-parameter spacing.

## Returns
* `pts`: Open-panel [`BoundaryPoints`](@ref) discretization.
"""
function _evaluate_points_panel(solver::CFIE_alpert{T},crv::C,k::T,idx::Int;minsep_tol=T(1e-12)) where {T<:Real,C<:AbsCurve}
    L=crv.length
    bs=solver.pts_scaling_factor
    N=max(solver.min_pts,round(Int,k*L*bs[1]/two_pi))
    needed=1
    if !isnothing(solver.symmetry)
        sym=solver.symmetry
        if sym isa Rotation
            needed=lcm(needed,sym.n)
        elseif hasproperty(sym,:axis)
            needed=lcm(needed,4)
        end
    end
    remN=mod(N,needed)
    remN!=0&&(N+=needed-remN)
    hσ=inv(T(N))
    sig=[T(j-0.5)/T(N) for j in 1:N]
    qT=T(solver.alpertq)
    while qT>=one(T)
        xy=Vector{SVector{2,T}}(undef,N)
        tangent_1st=Vector{SVector{2,T}}(undef,N)
        tangent_2nd=Vector{SVector{2,T}}(undef,N)
        ds=Vector{T}(undef,N)
        tmap=Vector{T}(undef,N)
        @inbounds for j in 1:N
            σ=sig[j]
            u,jac,jac2=_panel_sigma_to_u_jac(solver,σ,qT)
            tmap[j]=u
            q=curve(crv,u)
            tu=tangent(crv,u)
            t2u=tangent_2(crv,u)
            xy[j]=q
            tangent_1st[j]=tu*jac
            tangent_2nd[j]=t2u*(jac^2)+tu*jac2
            ds[j]=hypot(tangent_1st[j][1],tangent_1st[j][2])*hσ
        end
        minsep=minimum(diff(tmap))
        if minsep>=minsep_tol
            ws=fill(hσ,N)
            ws_der=ones(T,N)
            xL=curve(crv,zero(T))
            xR=curve(crv,one(T))
            tL=tangent(crv,zero(T))
            tR=tangent(crv,one(T))
            return BoundaryPoints(xy,tangent_1st,tangent_2nd,sig,tmap,ws,ws_der,ds,idx,false,xL,xR,tL,tR)
        end
        qT==one(T)&&break
        qnew=max(one(T),T(0.9)*qT)
        @warn "Alpert panel grading nodes too close; reducing q." q_old=qT q_new=qnew minsep=minsep minsep_tol=minsep_tol N=N
        qT=qnew
    end
    error("Alpert grading impossible: q reached 1 while minimum panel parameter spacing remained below $minsep_tol.")
end

"""
    evaluate_points(solver::CFIE_alpert{T},billiard::Bi,k::T) where {T<:Real,Bi<:AbsBilliard} → pts::Vector{BoundaryPoints{T}}

Constructs the complete Alpert boundary discretization of a billiard.

## Description
Three boundary representations are supported.

If the boundary is a flat vector of smooth closed curves, every curve is treated
as one periodic component.

If the boundary is one flat composite component, every constituent segment is
treated as a separate open panel sharing the same component identifier.

If the boundary contains several connected composite components, each segment
is treated as an open panel and all panels belonging to holes have their
orientation reversed.

## Arguments
* `solver`: Alpert combined-field solver.
* `billiard`: Billiard geometry.
* `k`: Real wavenumber controlling the boundary resolution.

## Returns
* `pts`: Vector of [`BoundaryPoints`](@ref). For panelized geometries,
  `pts[i].compid` identifies the connected boundary component to which each
  panel belongs.
"""
function evaluate_points(solver::CFIE_alpert{T},billiard::Bi,k::T) where {T<:Real,Bi<:AbsBilliard}
    boundary=billiard.full_boundary
    isempty(boundary)&&error("Boundary cannot be empty.")
    if !(boundary[1] isa AbstractVector)&&_all_closed_curves(boundary)
        pts=Vector{BoundaryPoints{T}}(undef,length(boundary))
        for (idx,crv) in enumerate(boundary)
            p=_evaluate_points_periodic(solver,crv,k,idx)
            pts[idx]=idx==1 ? p : _reverse_component_orientation(solver,p)
        end
        return pts
    end
    if _is_single_composite_boundary(boundary)
        pts=Vector{BoundaryPoints{T}}(undef,length(boundary))
        for (idx,crv) in enumerate(boundary)
            pts[idx]=_evaluate_points_panel(solver,crv,k,1;minsep_tol=T(solver.min_t_spacing))
        end
        return pts
    end
    ncomps=length(boundary)
    npanels=sum(length(comp) for comp in boundary)
    pts=Vector{BoundaryPoints{T}}(undef,npanels)
    pos=1
    for compid in 1:ncomps
        comp=boundary[compid]
        for crv in comp
            p=_evaluate_points_panel(solver,crv,k,compid;minsep_tol=T(solver.min_t_spacing))
            pts[pos]=compid==1 ? p : _reverse_component_orientation(solver,p)
            pos+=1
        end
    end
    return pts
end