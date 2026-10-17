################################################################################
# WEIGHTED PHYSICAL-DENSITY CFIE: SOURCE QUADRATURE AND MATRIX ASSEMBLY
#
# A Dirichlet eigenfunction is a single-layer potential ψ=S q, where q=∂ₙψ.
# Consequently S q=0 and K' q=q/2 on the boundary. We use the analytic family
#
#                         B(k)=I-2K'(k)-2ikS(k).
#
# K' differentiates the TARGET normal. This differs from the original CFIE's
# source-normal double-layer density μ; their densities must not be confused.
# For interior corner angle α, q=r^(π/α-1)v. The geometry layer supplies the
# mapped coordinate r∝t^p and physical cardinals
#
#              q(t)=Σⱼ Lⱼ(t) [ω(t)/ω(tⱼ)] qⱼ,
#              ω(t)=t^[p(π/α-1)].
#
# Every source rule below integrates these cardinals, including near-target
# rules and wavefunction reconstruction. Diagonal scaling of an already-built
# Kress matrix would leave its discrete roots unchanged and would not resolve
# the corner singularity.
#
# With d=x-y, r=|d| and c=nₓ⋅d/r, the raw integral kernel is
#
#              2∂ₙₓG+2ikG = -(i/2)c[kH₁(kr)]-(1/2)[kH₀(kr)].
#
# On a self-panel half-interval, r is proportional to the distance u from the
# target. Its log(u) coefficient is
#
#                 (c/π)[kJ₁(kr)]-(i/π)[kJ₀(kr)].
#
# Gaussian logarithmic product weights correct only this local coefficient.
# Other interactions use the stable full Hankel kernel. Higher k coefficients
# reuse the existing analytic Bessel differential-equation recurrences, so the
# direct, Beyn, CORK and expanded-BIM paths share the identical quadrature.
# On one panel, coefficient assembly is the matrix product K*B+L*B_log,
# where columns of K,L are kernel Taylor series and rows of B,B_log are
# integrated physical cardinals. Packing real and imaginary rows separately
# evaluates this same complex-linear contraction with real BLAS products;
# it does not conjugate the density or change the quadrature nodes or order.
################################################################################

# A subinterval is far enough for its Gaussian source rule when its sampled
# physical length is at most twice its distance from the target. Values 1-3
# are practical; reducing this ratio adds source work, not density unknowns.
const WEIGHTED_SOURCE_DISTANCE_RATIO = 2.0
# Limit automatic geometric bisection to 28 levels. Typical separated panels
# need 0-8. Reaching this limit is a resolution failure, never a silent pass.
const WEIGHTED_SOURCE_MAX_BISECTIONS = 28
# Below this relative separation, subtracting absolute coordinates can destroy
# the O(r²) normal chord. General charts instead integrate their derivatives;
# lines and circles use exact chord identities at every separation. The fourth
# root of epsilon balances cancellation against derivative-integration work.
const WEIGHTED_SOURCE_CLOSE_CHORD_RATIO = eps(Float64)^(1/4)
# Derivative integrals contain no kernel singularity. A roundoff allowance of
# 32 eps avoids futile subdivision; practical allowances are 8-128 eps.
const WEIGHTED_SOURCE_DERIVATIVE_TOLERANCE = 32eps(Float64)

# A Gaussian family depends only on its order and endpoint exponent, not on
# the panel or target. Reuse its reference nodes, ordinary weights and (for
# Legendre) logarithmic weights. Cached arrays are read-only: each source rule
# still owns its mapped parameters and cardinal matrices. The lock is used
# only while preparing rules, never while evaluating a boundary kernel.
const WEIGHTED_SOURCE_RULE_FAMILIES = Dict{Tuple{Int,Float64},NTuple{3,Vector{Float64}}}()
const WEIGHTED_SOURCE_RULE_FAMILY_LOCK = ReentrantLock()

"""
    _weighted_corner_gauss_rule(order::Int, exponent::Float64=0.0)

Return nodes and effective ordinary integration weights on `[0,1]`. If
`exponent > -1`, shifted Gauss-Jacobi quadrature integrates the endpoint
factor `u^exponent`: returned weights divide that factor out, because it is
already present in the physical density cardinal and geometric Jacobian.
`exponent=0` gives Gauss-Legendre. Inputs are unchanged; both returned arrays
are owned by the caller.

## Arguments
- `order::Int`: Source count, at least two; production minima are 32 and 96.
- `exponent::Float64=0.0`: Known integrable endpoint power, strictly above -1.

## Returns
- `Tuple{Vector{Float64},Vector{Float64}}`: Nodes and positive effective weights.
"""
function _weighted_corner_gauss_rule(order::Int, exponent::Float64=0.0)
    order >= 2 || throw(ArgumentError("weighted source quadrature needs order >= 2"))
    exponent > -1 || throw(ArgumentError("nonintegrable weighted source exponent"))
    x,w = exponent == 0 ? FastGaussQuadrature.gausslegendre(order) : FastGaussQuadrature.gaussjacobi(order,0.0,exponent)
    x .= (x .+ 1)./2
    w ./= 2.0^(exponent+1)
    exponent == 0 || (w .*= x.^(-exponent))
    all(isfinite,w) && all(>(0),w) || throw(ArgumentError("unresolved positive weighted source rule"))
    return x,w
end

"""
    _weighted_corner_log_weights(nodes::Vector{Float64}, weights::Vector{Float64})

For Gauss-Legendre nodes on `[0,1]`, return weights integrating `log(u)`
times every polynomial of degree less than the node count. Shifted Legendre
moments are `-1` for degree zero and `(-1)^(m+1)/(m(m+1))` thereafter.
Discrete Legendre orthogonality converts these moments into nodal weights
without a monomial Vandermonde solve. Arguments are read-only.

## Arguments
- `nodes::Vector{Float64}`: Increasing shifted Gauss-Legendre nodes.
- `weights::Vector{Float64}`: Corresponding ordinary `[0,1]` weights.

## Returns
- `Vector{Float64}`: Logarithmic product weights in the same node order.
"""
function _weighted_corner_log_weights(nodes::Vector{Float64},weights::Vector{Float64})
    n=length(nodes); out=similar(weights)
    @inbounds for j in 1:n
        x=2nodes[j]-1; previous=1.0; current=x; value=-1.0
        n>1 && (value += 1.5current)
        for m in 2:n-1
            next=((2m-1)*x*current-(m-1)*previous)/m
            value += (2m+1)*(-1.0)^(m+1)/(m*(m+1))*next
            previous,current=current,next
        end
        out[j]=weights[j]*value
    end
    return out
end

"""
    _weighted_corner_panel_rule(space, panel, a::Float64, b::Float64,
        order::Int; logarithmic::Bool=false, target_at_right::Bool=false)

Build a source rule on the computational subinterval `[a,b]` of `panel`.
`basis[s,j]` contains the physical weighted density cardinal, source speed
and ordinary integration weight. For a logarithmic interval, `logbasis`
contains the additional `(w_log-w*log(u))` correction, with the target at
`a` or, when `target_at_right=true`, at `b`. Ordinary corner intervals use
Jacobi exponent `p*lambda-1` at the actual corner. A logarithmic interval
must not also include the physical corner; self-panel construction separates
these endpoints before calling this function.

Returns a `WeightedSourceRule{Float64}`; geometry and solved density values
are not modified.

## Arguments
- `space::WeightedDensitySpace{Float64}`: Original physical geometry and panels.
- `panel::WeightedDensityPanel{Float64}`: Weighted amplitude interpolation chart.
- `a::Float64`, `b::Float64`: Computational limits with `0<=a<b<=1`.
- `order::Int`: Number of independent source integration nodes.

## Keyword arguments
- `logarithmic::Bool=false`: Include the local logarithmic product correction.
- `target_at_right::Bool=false`: Place that logarithm at the right endpoint.

## Returns
- `WeightedSourceRule{Float64}`: Owned geometry and integrated physical cardinals.
"""
function _weighted_corner_panel_rule(space::WeightedDensitySpace{Float64},panel::WeightedDensityPanel{Float64},a::Float64,b::Float64,order::Int;logarithmic::Bool=false,target_at_right::Bool=false)
    0 <= a < b <= 1 || throw(ArgumentError("invalid weighted source interval"))
    leftcorner=panel.lambda!=1 && !panel.reverse && a==0
    rightcorner=panel.lambda!=1 && panel.reverse && b==1
    logarithmic && (leftcorner||rightcorner) && throw(ArgumentError("split the corner endpoint away from the logarithmic endpoint"))
    exponent=(leftcorner||rightcorner) ? panel.power*panel.lambda-1 : 0.0
    family=lock(WEIGHTED_SOURCE_RULE_FAMILY_LOCK) do
        get!(WEIGHTED_SOURCE_RULE_FAMILIES,(order,exponent)) do
            x,w=_weighted_corner_gauss_rule(order,exponent)
            (x,w,exponent==0 ? _weighted_corner_log_weights(x,w) : Float64[])
        end
    end
    x,w,wl=family
    # Reflect only owned arrays. In particular reversing a self-logarithm
    # must not change the family subsequently used by a left endpoint.
    rightcorner && ((x,w)=(1 .- reverse(x),reverse(w)))
    wl=logarithmic ? (target_at_right ? reverse(wl) : wl) : Float64[]
    parameters=a .+(b-a).*x
    return _weighted_corner_panel_rule(space.curves[panel.curve_index],panel,parameters,w,wl,x,b-a,logarithmic,target_at_right)
end

# Curve dispatch is resolved once per source rule; the assembly loop consumes
# only concrete numeric arrays and never dispatches on abstract curve types.
function _weighted_corner_panel_rule(curve,panel,parameters,w,wl,x,width,logarithmic,target_at_right)
    n=length(parameters); m=length(panel.indices)
    xy=Vector{SVector{2,Float64}}(undef,n)
    basis=Matrix{Float64}(undef,n,m)
    logbasis=logarithmic ? similar(basis) : Matrix{Float64}(undef,0,0)
    cardinal=Vector{Float64}(undef,m)
    @inbounds for s in 1:n
        position,tangent,_=_weighted_panel_geometry(curve,panel,parameters[s])
        speed=hypot(tangent[1],tangent[2]); xy[s]=position
        _weighted_cardinal!(cardinal,panel,parameters[s])
        ordinary=width*w[s]*speed
        correction=logarithmic ? width*(wl[s]-w[s]*log(target_at_right ? 1-x[s] : x[s]))*speed : 0.0
        for j in 1:m
            basis[s,j]=ordinary*cardinal[j]
            logarithmic && (logbasis[s,j]=correction*cardinal[j])
        end
    end
    return WeightedSourceRule{Float64}(parameters,xy,basis,logbasis)
end

# Geometric adaptivity changes source integration only, retaining the original
# weighted trial space. Corner Jacobi factors therefore survive every split.
function _weighted_corner_separated(space,panel,target,a,b)
    return _weighted_corner_separated(space.curves[panel.curve_index],panel,target,a,b)
end

# Resolve the heterogeneous original curve once, outside the three-point
# separation test and the source-sample loops below. Its geometry arithmetic
# then specializes on the actual line, circle or user-defined chart.
function _weighted_corner_separated(curve::BilliardGeometry.AbsCurve,panel,target,a,b)
    xa=first(_weighted_panel_geometry(curve,panel,a))
    xm=first(_weighted_panel_geometry(curve,panel,(a+b)/2))
    xb=first(_weighted_panel_geometry(curve,panel,b))
    diameter=norm(xm-xa)+norm(xb-xm)
    distance=min(norm(target-xa),norm(target-xm),norm(target-xb))
    return diameter <= WEIGHTED_SOURCE_DISTANCE_RATIO*distance
end

function _weighted_corner_near_rules!(rules,space,panel,target,a,b,order,depth=0)
    if _weighted_corner_separated(space,panel,target,a,b)
        push!(rules,_weighted_corner_panel_rule(space,panel,a,b,order))
    else
        depth < WEIGHTED_SOURCE_MAX_BISECTIONS || throw(ArgumentError("weighted source interval remains unresolved after automatic bisection"))
        mid=(a+b)/2
        _weighted_corner_near_rules!(rules,space,panel,target,a,mid,order,depth+1)
        _weighted_corner_near_rules!(rules,space,panel,target,mid,b,order,depth+1)
    end
    return rules
end

# Difference original native parameters without subtracting rounded physical
# positions. Integer power maps use a factored difference of powers; other
# powers use log1p/expm1 so close mapped coordinates retain relative accuracy.
function _weighted_corner_parameter_difference(panel,target_panel,target_t,source_t,target_u)
    if panel.left==target_panel.left && panel.right==target_panel.right && panel.power==target_panel.power && panel.reverse==target_panel.reverse
        x=panel.reverse ? 1-target_t : target_t
        y=panel.reverse ? 1-source_t : source_t
        δ=panel.reverse ? source_t-target_t : target_t-source_t
        p=panel.power
        difference=p==1 ? δ : p==2 ? δ*(x+y) : p==3 ? δ*(x*x+x*y+y*y) : -x^p*expm1(p*log1p(-δ/x))
        return (panel.reverse ? -1 : 1)*(panel.right-panel.left)*difference
    end
    ξ=panel.reverse ? 1-source_t : source_t
    source_u=panel.reverse ? panel.right-(panel.right-panel.left)*ξ^panel.power : panel.left+(panel.right-panel.left)*ξ^panel.power
    return target_u-source_u
end

# The source-normal and target-normal self limits are especially sensitive to
# subtraction: n⋅(x-y)=O(r²), although each absolute coordinate is O(1).
# General charts use exact derivative-integral identities at close separation.
# The adaptively integrated scalar/vector functions are smooth, without any
# endpoint kernel singularity; their error tolerances follow machine precision.
function _weighted_corner_pair_geometry(curve,target_u,δ,target,normal,source)
    d=target-source; r=norm(d)
    if r > WEIGHTED_SOURCE_CLOSE_CHORD_RATIO*max(1.0,curve.length)
        return r,dot(normal,d)/r
    end
    # A short physical chord can cross the periodic endpoint of one closed
    # chart. Integrating over the almost complete native period would cancel
    # O(1) velocities to an O(r) answer. Two SHORT endpoint offsets avoid this
    # without assuming the chart is defined outside its native [0,1] interval.
    if abs(δ)>0.5
        first_point=SVector{2,Float64}(BilliardGeometry.curve(curve,0.0))
        last_point=SVector{2,Float64}(BilliardGeometry.curve(curve,1.0))
        tolerance=WEIGHTED_CORNER_GEOMETRY_DEFAULTS.endpoint_roundoff_factor*eps(Float64)*max(1.0,norm(first_point),norm(last_point))
        if norm(first_point-last_point)<=tolerance
            target_end=target_u<0.5 ? 0.0 : 1.0
            source_end=1-target_end
            target_offset=_weighted_corner_endpoint_offset(curve,target_end,target_u-target_end)
            source_offset=_weighted_corner_endpoint_offset(curve,source_end,target_u-δ-source_end)
            d=target_offset-source_offset; r=norm(d)
            return r,dot(normal,d)/r
        end
    end
    tolerance=WEIGHTED_SOURCE_DERIVATIVE_TOLERANCE
    speed_scale=max(norm(tangent(curve,target_u)),norm(tangent(curve,target_u-δ)),floatmin(Float64))
    acceleration_scale=max(norm(tangent_2(curve,target_u)),norm(tangent_2(curve,target_u-δ/2)),norm(tangent_2(curve,target_u-δ)),eps(Float64)*speed_scale)
    velocity,_=QuadGK.quadgk(t->SVector{2,Float64}(tangent(curve,target_u-t*δ)),0.0,1.0;rtol=tolerance,atol=tolerance*speed_scale)
    acceleration,_=QuadGK.quadgk(t->(1-t)*dot(normal,SVector{2,Float64}(tangent_2(curve,target_u-t*δ))),0.0,1.0;rtol=tolerance,atol=tolerance*acceleration_scale)
    r=abs(δ)*norm(velocity)
    return r,-δ^2*acceleration/r
end

@inline function _weighted_corner_pair_geometry(curve::BilliardGeometry.AbsLine,target_u,δ,target,normal,source)
    return abs(δ)*curve.length,0.0
end

@inline function _weighted_corner_pair_geometry(curve::BilliardGeometry.CircleSegment,target_u,δ,target,normal,source)
    r=abs(2curve.radius*sin(curve.arc_angle*δ/2))
    return r,sign(curve.arc_angle)*r/(2curve.radius)
end

# A corner shared by two charts must also be a shared floating-point anchor.
# Evaluating γ(1-hξ^p) first would round away part of a small native offset.
# Instead obtain γ(u₀+δ)-γ(u₀) directly; primitive charts have exact identities.
function _weighted_corner_endpoint_offset(curve,u0,δ)
    scale=max(norm(tangent(curve,u0)),norm(tangent(curve,u0+δ)),floatmin(Float64))
    tolerance=WEIGHTED_SOURCE_DERIVATIVE_TOLERANCE
    velocity,_=QuadGK.quadgk(t->SVector{2,Float64}(tangent(curve,u0+t*δ)),0.0,1.0;rtol=tolerance,atol=tolerance*scale)
    return δ*velocity
end

@inline _weighted_corner_endpoint_offset(curve::BilliardGeometry.AbsLine,u0,δ) = δ*(curve.pt1-curve.pt0)

@inline function _weighted_corner_endpoint_offset(curve::BilliardGeometry.CircleSegment,u0,δ)
    angle=curve.arc_angle*u0+curve.shift_angle
    sine,cosine=sincos(angle); phase=curve.arc_angle*δ
    cosine_difference=-2sin(phase/2)^2; sine_difference=sin(phase)
    return curve.radius*SVector(cosine*cosine_difference-sine*sine_difference,sine*cosine_difference+cosine*sine_difference)
end

# Native offset from an ORIGINAL curve endpoint, retaining small powers even
# when adding them to u=1 would round them away. This also handles an ordinary
# first/last panel at the smooth periodic seam of one generic closed chart.
@inline function _weighted_corner_native_endpoint_offset(panel,t,endpoint)
    h=panel.right-panel.left; p=panel.power
    if endpoint==0
        return panel.reverse ? panel.left-h*expm1(p*log1p(-t)) : panel.left+h*t^p
    end
    return panel.reverse ? panel.right-1-h*(1-t)^p : panel.right-1+h*expm1(p*log(t))
end

function _weighted_corner_cache_pair_geometry!(rules,space,pts,target_index,panel)
    return _weighted_corner_cache_pair_geometry!(rules,space.curves[panel.curve_index],space,pts,target_index,panel)
end

function _weighted_corner_cache_pair_geometry!(rules,curve::BilliardGeometry.AbsCurve,space,pts,target_index,panel)
    target=pts.xy[target_index]; normal=pts.normal[target_index]
    target_panel=space.panels[space.node_panel[target_index]]
    same_curve=panel.curve_index==target_panel.curve_index
    common_endpoint=false; target_offset=zero(target)
    source_anchor=panel.left==0 ? 0.0 : panel.right==1 ? 1.0 : -1.0
    target_anchor=target_panel.left==0 ? 0.0 : target_panel.right==1 ? 1.0 : -1.0
    if source_anchor>=0 && target_anchor>=0 && (!same_curve||source_anchor!=target_anchor)
        target_curve=space.curves[target_panel.curve_index]
        xt=SVector{2,Float64}(BilliardGeometry.curve(target_curve,target_anchor))
        xs=SVector{2,Float64}(BilliardGeometry.curve(curve,source_anchor))
        tolerance=WEIGHTED_CORNER_GEOMETRY_DEFAULTS.endpoint_roundoff_factor*eps(Float64)*max(1.0,norm(xt),norm(xs))
        common_endpoint=norm(xt-xs)<=tolerance && norm(target-xt)<=WEIGHTED_SOURCE_CLOSE_CHORD_RATIO*max(1.0,target_curve.length)
        if common_endpoint
            δ=_weighted_corner_native_endpoint_offset(target_panel,space.node_parameter[target_index],target_anchor)
            target_offset=_weighted_corner_endpoint_offset(target_curve,target_anchor,δ)
        end
    end
    for rule in rules
        resize!(rule.distances,length(rule.xy)); resize!(rule.normal_projections,length(rule.xy))
        for s in eachindex(rule.xy)
            if common_endpoint
                δ=_weighted_corner_native_endpoint_offset(panel,rule.parameters[s],source_anchor)
                source_offset=_weighted_corner_endpoint_offset(curve,source_anchor,δ)
                d=target_offset-source_offset; r=norm(d); projection=dot(normal,d)/r
            elseif same_curve
                δ=_weighted_corner_parameter_difference(panel,target_panel,space.node_parameter[target_index],rule.parameters[s],pts.tphys[target_index])
                r,projection=_weighted_corner_pair_geometry(curve,pts.tphys[target_index],δ,target,normal,rule.xy[s])
            else
                d=target-rule.xy[s]; r=norm(d); projection=dot(normal,d)/r
            end
            r>0 && isfinite(r) && isfinite(projection) || throw(ArgumentError("unresolved weighted source-target geometry"))
            rule.distances[s]=r; rule.normal_projections[s]=projection
        end
    end
    return rules
end

function _weighted_corner_target_rules(space,pts,target_index,panel_index,order)
    panel=space.panels[panel_index]; a=panel.tleft; b=panel.tright
    rules=WeightedSourceRule{Float64}[]
    if space.node_panel[target_index]==panel_index
        t=space.node_parameter[target_index]
        # Keep the fractional endpoint outside the self-logarithm interval,
        # and do not let an exterior branch point approach a long interval.
        # Ordinary outer pieces are then bisected by physical separation.
        lo,hi = panel.lambda==1 ? (a,b) : panel.reverse ? (max(a,2t-1),min(b,(t+1)/2)) : (max(a,t/2),min(b,2t))
        lo>a && _weighted_corner_near_rules!(rules,space,panel,pts.xy[target_index],a,lo,order)
        push!(rules,_weighted_corner_panel_rule(space,panel,lo,t,order;logarithmic=true,target_at_right=true))
        push!(rules,_weighted_corner_panel_rule(space,panel,t,hi,order;logarithmic=true))
        hi<b && _weighted_corner_near_rules!(rules,space,panel,pts.xy[target_index],hi,b,order)
    else
        if _weighted_corner_separated(space,panel,pts.xy[target_index],a,b)
            return space.source_cache[(0,panel_index,order)]
        end
        _weighted_corner_near_rules!(rules,space,panel,pts.xy[target_index],a,b,order)
    end
    return _weighted_corner_cache_pair_geometry!(rules,space,pts,target_index,panel)
end

"""
    _weighted_corner_source_plan(pts::BoundaryPoints, target_indices;
        source_order::Int=0, multithreaded::Bool=true)

Prepare geometry-only source quadratures for selected boundary target nodes.
`source_order=0` retains each panel's ordinary/corner minimum; a positive
value only increases those minima. Rules are cached in the density space by
target, panel and order. Cache preparation is locked and finishes before any
matrix evaluation. Independent targets may be prepared in parallel; the cache
is read-only during that work and updated afterward. Returned nested vectors
refer to read-only rules.

## Arguments
- `pts::BoundaryPoints{Float64}`: Solved weighted physical-density mesh.
- `target_indices::AbstractVector{Int}`: Full-boundary target node indices.

## Keyword arguments
- `source_order::Int=0`: Optional stricter minimum; zero retains panel settings.
- `multithreaded::Bool=true`: Prepare independent target rules in parallel.

## Returns
- `Vector{Vector{Vector{WeightedSourceRule{Float64}}}}`: Rules by target and panel.
"""
function _weighted_corner_source_plan(pts::BoundaryPoints{Float64},target_indices;source_order::Int=0,multithreaded::Bool=true)
    space=pts.density_space::WeightedDensitySpace{Float64}
    plan=Vector{Vector{Vector{WeightedSourceRule{Float64}}}}(undef,length(target_indices))
    lock(space.source_lock) do
        for (p,panel) in enumerate(space.panels)
            order=max(panel.source_order,source_order)
            get!(space.source_cache,(0,p,order)) do
                [_weighted_corner_panel_rule(space,panel,panel.tleft,panel.tright,order)]
            end
        end
        @use_threads multithreading=(multithreaded && length(target_indices)>1) for row in eachindex(target_indices)
            i=target_indices[row]
            panels=Vector{Vector{WeightedSourceRule{Float64}}}(undef,length(space.panels))
            for (p,panel) in enumerate(space.panels)
                order=max(panel.source_order,source_order)
                existing=get(space.source_cache,(i,p,order),nothing)
                panels[p]=existing===nothing ? _weighted_corner_target_rules(space,pts,i,p,order) : existing
            end
            plan[row]=panels
        end
        for (row,i) in enumerate(target_indices), (p,panel) in enumerate(space.panels)
            order=max(panel.source_order,source_order)
            plan[row][p]=get!(space.source_cache,(i,p,order),plan[row][p])
        end
    end
    return plan
end

"""
    _weighted_corner_bessel_pair(kind::Union{Val{:hankel},Val{:besselj}},
        z::ComplexF64) -> Tuple{ComplexF64,ComplexF64}

Evaluate consecutive orders zero and one of `H⁽¹⁾` or `J` in one AMOS call.
Both orders share AMOS initialization and argument reduction. This is the
same unscaled algorithm used by SpecialFunctions, without a new asymptotic
approximation. Real arguments retain the existing faster Bessels path.

The private OpenSpecFun Fortran ABI follows SpecialFunctions 2.9, already
required by this package: `FNU=0`, `KODE=1`, `N=2`, and `M=1` for Hankel.
Each output buffer contains two contiguous doubles. Return codes zero and
three are accepted, and underflow counts are ignored, exactly as in the
single-order wrappers. Other codes fall back to those public wrappers so
the paired call never imposes a stricter error policy.

## Arguments
- `kind::Union{Val{:hankel},Val{:besselj}}`: Select outgoing Hankel or regular
  Bessel functions through compile-time dispatch.
- `z::ComplexF64`: Complex radial argument; inputs are unchanged.

## Returns
- `Tuple{ComplexF64,ComplexF64}`: Function values of orders zero and one.
"""
@inline function _weighted_corner_bessel_pair(kind::Union{Val{:hankel},Val{:besselj}},z::ComplexF64)
    re=Ref{NTuple{2,Float64}}(); ii=Ref{NTuple{2,Float64}}()
    underflows=Ref{Int32}(); status=Ref{Int32}()
    if kind isa Val{:hankel}
        ccall((:zbesh_,SpecialFunctions.libopenspecfun),Cvoid,
            (Ref{Float64},Ref{Float64},Ref{Float64},Ref{Int32},Ref{Int32},Ref{Int32},
             Ref{NTuple{2,Float64}},Ref{NTuple{2,Float64}},Ref{Int32},Ref{Int32}),
            real(z),imag(z),0.0,Int32(1),Int32(1),Int32(2),re,ii,underflows,status)
    else
        ccall((:zbesj_,SpecialFunctions.libopenspecfun),Cvoid,
            (Ref{Float64},Ref{Float64},Ref{Float64},Ref{Int32},Ref{Int32},
             Ref{NTuple{2,Float64}},Ref{NTuple{2,Float64}},Ref{Int32},Ref{Int32}),
            real(z),imag(z),0.0,Int32(1),Int32(2),re,ii,underflows,status)
    end
    if status[]!=0 && status[]!=3
        return kind isa Val{:hankel} ? (SpecialFunctions.besselh(0,1,z),SpecialFunctions.besselh(1,1,z)) :
            (SpecialFunctions.besselj(0,z),SpecialFunctions.besselj(1,z))
    end
    return ComplexF64(re[][1],ii[][1]),ComplexF64(re[][2],ii[][2])
end

@inline function _weighted_corner_radial!(w::TaylorWorkspace,k::Number,r::Float64,degree::Int,split::Bool,normal_derivative::Bool=true)
    z=k*r
    # A direct interaction with an exactly zero normal projection needs only
    # H₀. Taylor transport still needs H₁ to seed the radial derivative of H₀.
    h0,h1 = degree==0 && !normal_derivative ? (ComplexF64(_bim_hankelh1(0,z)),0.0im) :
        z isa ComplexF64 ? _weighted_corner_bessel_pair(Val(:hankel),z) :
        (ComplexF64(_bim_hankelh1(0,z)),ComplexF64(_bim_hankelh1(1,z)))
    if degree==0
        w.H0[1]=k*h0; w.H1[1]=k*h1
        if split
            j0,j1=z isa ComplexF64 && normal_derivative ? _weighted_corner_bessel_pair(Val(:besselj),z) :
                (_bim_besselj(0,z,h0),normal_derivative ? _bim_besselj(1,z,h1) : 0.0im)
            w.J0[1]=k*j0; w.J1[1]=k*j1
        end
    else
        kr=Float64(k)
        radial1_taylor!(w.H1,kr,r*r,degree,kr*h1,kr*r*h0)
        radial0_k_taylor!(w.H0,w.yH0,kr,r*r,degree,h0,-r*h1)
        if split
            j0=ComplexF64(_bim_besselj(0,z,h0)); j1=ComplexF64(_bim_besselj(1,z,h1))
            radial1_taylor!(w.J1,kr,r*r,degree,kr*j1,kr*r*j0)
            radial0_k_taylor!(w.J0,w.yJ0,kr,r*r,degree,j0,-r*j1)
        end
    end
    return w
end

"""
    _weighted_corner_matrices(solver, pts::BoundaryPoints{Float64}, k0::Number,
        degree::Int; multithreaded::Bool=true, source_order::Int=0,
        reduce_symmetry::Bool=true)

Construct normalized Taylor matrices `A[l+1]=B⁽ˡ⁾(k0)/l!` of the weighted
physical-normal-density CFIE. `degree=0` also accepts complex `k0` and is the
independent direct evaluation path. Positive degrees require real positive
`k0` and reuse the library's analytic radial recurrences. Source quadrature
does not depend on the degree or expansion point.

If symmetry is selected, every full-boundary source image contributes to its
fundamental unknown with the corresponding character; only target rows are
restricted. The resulting vector contains `degree+1` complex square matrices.

## Arguments
- `solver::CombinedFieldIntegralEquationSolver`: CFIE configuration and sector.
- `pts::BoundaryPoints{Float64}`: Weighted physical normal-density discretization.
- `k0::Number`: Positive expansion center; complex values require degree zero.
- `degree::Int`: Highest retained normalized Taylor coefficient, at least zero.

## Keyword arguments
- `multithreaded::Bool=true`: Parallelize independent target-row chunks.
- `source_order::Int=0`: Raise independent source integration minima if positive.
- `reduce_symmetry::Bool=true`: Fold every source image into the selected sector;
  false returns the complete physical-boundary matrices for an independent audit.

## Returns
- `Vector{Matrix{ComplexF64}}`: Coefficients in increasing powers of `k-k0`.
"""
function _weighted_corner_matrices(solver,pts::BoundaryPoints{Float64},k0::Number,degree::Int;multithreaded::Bool=true,source_order::Int=0,reduce_symmetry::Bool=true)
    degree>=0 || throw(ArgumentError("Taylor degree must be nonnegative"))
    real(k0)>0 && (degree==0 || isreal(k0)) || throw(ArgumentError("weighted CFIE Taylor center must be positive and real"))
    k=degree==0 ? k0 : Float64(real(k0)); space=pts.density_space::WeightedDensitySpace{Float64}
    N=length(pts)
    orbits=(!reduce_symmetry||solver.symmetry===nothing) ? nothing : _fold_boundary(Float64,solver.billiard,pts,solver.symmetry,solver.sym_characters)
    targets=orbits===nothing ? collect(1:N) : orbits.fundamental_indices
    m=length(targets); columns=orbits===nothing ? collect(1:N) : orbits.orbit_of
    phase=orbits===nothing ? ones(ComplexF64,N) : orbits.phase
    plan=_weighted_corner_source_plan(pts,targets;source_order,multithreaded)
    matrices=[zeros(ComplexF64,m,m) for _ in 0:degree]
    # Private workspaces belong to row chunks, rather than thread IDs, so a
    # yielded task cannot share scratch with another task on the same thread.
    chunks=multithreaded ? min(Threads.nthreads(),m) : 1
    largest_panel=maximum(p->length(p.indices),space.panels)
    largest_order=maximum(p->max(p.source_order,source_order),space.panels)
    @use_threads multithreading=(multithreaded && chunks>1) for chunk in 1:chunks
        w=TaylorWorkspace(degree)
        # Integrating each cardinal is one small real matrix product. Packing
        # real and imaginary parts in successive rows retains the complex
        # operator while allowing BLAS to contract the same real quadrature
        # matrix without copying it or storing a second complex source cache.
        panel_coefficients=zeros(Float64,2(degree+1),largest_panel)
        kernel_coefficients=Matrix{Float64}(undef,2(degree+1),largest_order)
        logarithmic_coefficients=similar(kernel_coefficients)
        @inbounds for row in chunk:chunks:m
            i=targets[row]; target=pts.xy[i]; normal=pts.normal[i]
            for (p,panel) in enumerate(space.panels)
                fill!(panel_coefficients,0)
                block=@view panel_coefficients[:,1:length(panel.indices)]
                for rule in plan[row][p]
                    split=!isempty(rule.logbasis)
                    for s in eachindex(rule.xy)
                        if isempty(rule.distances)
                            d=target-rule.xy[s]; r=hypot(d[1],d[2])
                            r>0 || throw(ArgumentError("coincident weighted source and target"))
                            projection=(normal[1]*d[1]+normal[2]*d[2])/r
                        else
                            r=rule.distances[s]; projection=rule.normal_projections[s]
                        end
                        _weighted_corner_radial!(w,k,r,degree,split,projection!=0)
                        @simd for l in 1:degree+1
                            value=-(im/2)*projection*w.H1[l]-w.H0[l]/2
                            kernel_coefficients[2l-1,s]=real(value)
                            kernel_coefficients[2l,s]=imag(value)
                            if split
                                correction=(projection*w.J1[l]-im*w.J0[l])/pi
                                logarithmic_coefficients[2l-1,s]=real(correction)
                                logarithmic_coefficients[2l,s]=imag(correction)
                            end
                        end
                    end
                    mul!(block,@view(kernel_coefficients[:,1:length(rule.xy)]),rule.basis,1.0,1.0)
                    split && mul!(block,@view(logarithmic_coefficients[:,1:length(rule.xy)]),rule.logbasis,1.0,1.0)
                end
                for (j,column) in enumerate(panel.indices), l in 1:degree+1
                    value=ComplexF64(panel_coefficients[2l-1,j],panel_coefficients[2l,j])
                    matrices[l][row,columns[column]]-=phase[column]*value
                end
            end
            matrices[1][row,row]+=1
        end
    end
    all(A->all(isfinite,A),matrices) || throw(ArgumentError("nonfinite weighted CFIE matrix"))
    return matrices
end

"""
    _weighted_corner_potential(pts::BoundaryPoints{Float64}, q::AbstractVector,
        k::Number, targets::AbstractVector; source_order::Int=0,
        refinement::Int=1, multithreaded::Bool=true)

Reconstruct `ψ=S q` at interior Cartesian targets from the physical nodal
normal density `q`. The same weighted interpolation used by the boundary
matrix enters every ordinary and adaptively subdivided source rule.
`source_order` raises the panel minimum and `refinement` multiplies it for
independent source-quadrature audits; neither changes solved density values.
Returns a `Vector{ComplexF64}` in target order. Boundary self-evaluation is
not supported by this interior-potential routine.

## Arguments
- `pts::BoundaryPoints{Float64}`: Complete physical weighted-density mesh.
- `q::AbstractVector{<:Number}`: Physical nodal normal derivative, one per node.
- `k::Number`: Real or complex wavenumber for the outgoing Green function.
- `targets::AbstractVector{<:StaticVector{2}}`: Interior Cartesian evaluation points.

## Keyword arguments
- `source_order::Int=0`: Optional raised minimum integration order.
- `refinement::Int=1`: Positive multiplier for every source order.
- `multithreaded::Bool=true`: Parallelize independent interior targets.

## Returns
- `Vector{ComplexF64}`: Single-layer wavefunction values in target order.
"""
function _weighted_corner_potential(pts::BoundaryPoints{Float64},q::AbstractVector,k::Number,targets::AbstractVector;source_order::Int=0,refinement::Int=1,multithreaded::Bool=true)
    length(q)==length(pts) || throw(DimensionMismatch("potential needs the complete physical boundary density"))
    refinement>=1 || throw(ArgumentError("source refinement must be positive"))
    space=pts.density_space::WeightedDensitySpace{Float64}; np=length(space.panels)
    ordinary=Vector{WeightedSourceRule{Float64}}(undef,np)
    lock(space.source_lock) do
        for (p,panel) in enumerate(space.panels)
            order=refinement*max(panel.source_order,source_order)
            ordinary[p]=only(get!(space.source_cache,(0,p,order)) do
                [_weighted_corner_panel_rule(space,panel,panel.tleft,panel.tright,order)]
            end)
        end
    end
    amplitudes=Vector{Vector{ComplexF64}}(undef,np)
    for (p,panel) in enumerate(space.panels)
        rule=ordinary[p]; values=zeros(ComplexF64,length(rule.xy))
        @inbounds for (j,column) in enumerate(panel.indices), s in eachindex(values)
            values[s]+=rule.basis[s,j]*q[column]
        end
        amplitudes[p]=values
    end
    out=zeros(ComplexF64,length(targets))
    @use_threads multithreading=(multithreaded && length(targets)>=32) for i in eachindex(targets)
        target=targets[i]; value=0.0im
        for (p,panel) in enumerate(space.panels)
            # Keep the common well-separated interior-field path allocation
            # free. Only near-boundary targets need temporary refined rules.
            rule=ordinary[p]; nearest=minimum(x->norm(target-x),rule.xy)
            diameter=norm(first(rule.xy)-last(rule.xy))
            rules=nearest*WEIGHTED_SOURCE_DISTANCE_RATIO>=diameter ? (rule,) : _weighted_corner_near_rules!(WeightedSourceRule{Float64}[],space,panel,target,panel.tleft,panel.tright,refinement*max(panel.source_order,source_order))
            for source in rules, s in eachindex(source.xy)
                amplitude=source===rule ? amplitudes[p][s] : 0.0im
                if source!==rule
                    for (j,column) in enumerate(panel.indices)
                        amplitude+=source.basis[s,j]*q[column]
                    end
                end
                d=target-source.xy[s]; r=hypot(d[1],d[2])
                r>0 || throw(ArgumentError("single-layer reconstruction target lies on a source node"))
                value+=(im/4)*_bim_hankelh1(0,k*r)*amplitude
            end
        end
        out[i]=value
    end
    return out
end
