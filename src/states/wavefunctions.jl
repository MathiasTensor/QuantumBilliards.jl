using Bessels, LinearAlgebra, ProgressMeter

#########################################################################
####################### HELPERS FOR INTERIOR MASK #######################
#########################################################################

"""
    billiard_polygon_single_component(billiard::Bi, N_polygon_checks::Int; fundamental_domain=true) :: Vector where {Bi<:AbsBilliard}

Given <:AbsBilliard object, computes the points on the boundary of the billiard that are equidistant from each other, with the total number of points being N_polygon_checks. This only works for domains wihtout holes, otherwise see `_billiard_polygon_multi_component`.
    
# Arguments
- `billiard`: A billiard object with a fundamental_boundary or full_boundary field.
- `N_polygon_checks`: The total number of points to be distributed along the boundary.
- `fundamental_domain::Bool=true`: A flag indicating whether to compute points on the fundamental or full boundary.

# Returns
- `Vector{Vector{SVector{2,<:Real}}}`: For each crv in the billiard boundary (chosen by the flag fundamental_domain) form a Vector{SVector{2,<:Real}} object containing the discretization points for that curve. 
"""
function _billiard_polygon_single_component(billiard::Bi,N_polygon_checks::Int;fundamental_domain=true) where {Bi<:AbsBilliard}
    if fundamental_domain
        boundary=billiard.fundamental_boundary
    else
        boundary=billiard.full_boundary
    end
    billiard_composite_lengths=[crv.length for crv in boundary] # Find the fraction of lengths wrt to the boundary
    typ=eltype(billiard_composite_lengths[1])
    total_billiard_length=sum(billiard_composite_lengths)
    billiard_length_fractions=[crv.length/total_billiard_length for crv in boundary]
    distributed_points=[round(Int, fract*N_polygon_checks) for fract in billiard_length_fractions] # Redistribute points based on the fractions
    ts_vectors=[sample_points(LinearNodes(),crv_pts)[1] for crv_pts in distributed_points] # vector of vectors for each crv a vector of ts for the outer boundary
    xy_vectors=Vector{Vector}(undef,length(boundary))
    for (i,crv) in enumerate(boundary) 
        xy_vectors[i]=curve(crv,ts_vectors[i])
    end
    return xy_vectors
end

"""
    billiard_polygon_multi_component(billiard::Bi, N_polygon_checks::Int) :: Vector where {Bi<:AbsBilliard}

Given an AbsBilliard object with multiple connected components (holes), computes the points on the boundary of the billiard that are equidistant from each other, with the total number of points being N_polygon_checks. This only works for domains with holes, otherwise see `_billiard_polygon_single_component`.

# Arguments
- `billiard`: A billiard object with a full_boundary field that contains multiple connected components.
- `N_polygon_checks`: The total number of points to be distributed along the boundary.

# Returns
- `Vector{Vector{SVector{2,<:Real}}}`: For each crv in the billiard full_boundary form a Vector{SVector{2,<:Real}} object containing the discretization points for that curve.
"""
function _billiard_polygon_multi_component(billiard::Bi,N_polygon_checks::Int) where {Bi<:AbsBilliard}
    boundary=billiard.full_boundary # fundamental == full for multi-component billiards (CFIE_kress), so we just take full boundary
    ncomp=length(boundary)
    xy_components=Vector{Vector}(undef,ncomp)
    comp_lengths=zeros(Float64,ncomp) # determine component lengths
    for i in 1:ncomp
        comp=boundary[i]
        comp_lengths[i]=comp.length # comp cannot be composite due to Kress analytic splitting in CFIE_kress, so we can just take length of the component
    end
    total_length=sum(comp_lengths)
    comp_points=[max(16,round(Int,N_polygon_checks*L/total_length)) for L in comp_lengths]  # distribute total polygon budget across connected components
    for i in 1:ncomp
        comp=boundary[i]
        ts=sample_points(LinearNodes(),comp_points[i])[1]
        xy_components[i]=curve(comp,ts)
    end
    return xy_components
end

"""
    billiard_polygon(billiard::Bi, N_polygon_checks::Int; fundamental_domain=true, solver::Symbol=:OUTER) :: Vector where {Bi<:AbsBilliard}

Given an AbsBilliard object, computes the points on the boundary of the billiard that are equidistant from each other, with the total number of points being N_polygon_checks. This function handles both single-component and multi-component billiards by dispatching to the appropriate helper function.

# Arguments
- `billiard`: A billiard object with either a fundamental_boundary (for single-component) or full_boundary (for multi-component) field.
- `N_polygon_checks`: The total number of points to be distributed along the boundary.

# Keyword arguments
- `fundamental_domain::Bool=true`: A flag indicating whether to compute points on the fundamental or full boundary for single-component billiards. This argument is ignored for multi-component billiards since their fundamental and full boundaries are the same.
- `boundary_type::Symbol`: A symbol indicating single-component billiards `:OUTER` or `:OUTER_INNER` for multi-component billiards.

# Returns
- `Vector{Vector{SVector{2,<:Real}}}`: For each crv in the billiard boundary (chosen by the flag fundamental_domain for single-component billiards, or full_boundary for multi-component billiards) form a Vector{SVector{2,<:Real}} object containing the discretization points for that curve.
"""
function billiard_polygon(billiard::Bi,N_polygon_checks::Int;fundamental_domain=true,boundary_type::Symbol=:OUTER) where {Bi<:AbsBilliard}
    if boundary_type==:OUTER
        return _billiard_polygon_single_component(billiard,N_polygon_checks;fundamental_domain=fundamental_domain)
    elseif boundary_type==:OUTER_INNER
        return _billiard_polygon_multi_component(billiard,N_polygon_checks)
    else
        error("Unknown boundary type: $boundary_type. Use :OUTER or :OUTER_INNER.")
    end
end

"""
    is_left(p1::SVector{2,T}, p2::SVector{2,T}, pt::SVector{2,T}) where {T<:Real}

Determines whether the point `pt` is to the left of the line segment defined by `p1` and `p2`.

# Arguments
- `p1::SVector{2,T}`: The first point defining the line segment.
- `p2::SVector{2,T}`: The second point defining the line segment.
- `pt::SVector{2,T}`: The point to check.

# Returns
- `T`: + or - value depending if the point is to the left or right of the line segment.
"""
function is_left(p1::SVector{2,T},p2::SVector{2,T},pt::SVector{2,T}) where {T<:Real}
    return (p2[1]-p1[1])*(pt[2]-p1[2])-(pt[1]-p1[1])*(p2[2]-p1[2])
end


# Winding number algorithm to check if a point is inside a polygon
"""
    is_point_in_polygon(polygon::Vector{SVector{2,T}}, point::SVector{2,T})::Bool where T

Determines whether a single `point` is inside a billiard `polygon` formed by it's boundary points. It implements a winding number algorithm for the checking.

# Arguments
- `polygon::Vector{SVector{2,T}}`: A vector of points representing the boundary of the polygon.
- `point::SVector{2,T}`: A point to check if it's inside the polygon.

# Returns
- `Bool`: `true` if the point is inside the polygon, `false` otherwise.
"""
function is_point_in_polygon(polygon::Vector{SVector{2,T}},point::SVector{2,T})::Bool where T
    winding_number=0
    num_points=length(polygon)
    for i in 1:num_points
        p1=polygon[i]
        p2=polygon[(i%num_points)+1]
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
    is_point_in_multiply_connected_polygon(components, point::SVector{2,T})::Bool where {T<:Real}

Check whether `point` lies inside a multiply connected polygonal domain.

Convention:
- `components[1]` is the outer boundary polygon
- `components[2:end]` are hole polygons

A point is inside the domain iff it is inside the outer polygon and not inside any hole polygon.
"""
function is_point_in_multiply_connected_polygon(components,point::SVector{2,T})::Bool where {T<:Real}
    isempty(components) && return false
    is_point_in_polygon(components[1],point) || return false # must be inside outer boundary
    @inbounds for h in 2:length(components) # must be outside every hole
        if is_point_in_polygon(components[h],point)
            return false
        end
    end
    return true
end

"""
    points_in_billiard_polygon(pts::Vector{SVector{2,T}},billiard::Bi,N_polygon_checks::Int;fundamental_domain=true,boundary_type::Symbol=:OUTER) where {T<:Real,Bi<:AbsBilliard}

Determine whether the points `pts` lie inside the billiard polygon sampled with
`N_polygon_checks` boundary points.

# Arguments
- `pts::Vector{SVector{2,T}}`: points to test
- `billiard::Bi`: billiard geometry
- `N_polygon_checks::Int`: total polygon sampling budget

# Keyword arguments
- `fundamental_domain::Bool=true`: use the fundamental boundary for BIM. Ignored for CFIE_kress.
- `boundary_type::Symbol=:OUTER`: `:OUTER` for single connected domain, `:OUTER_INNER` for multiply connected domain.
    For `boundary_type == :OUTER`, the billiard is treated as a single connected domain.
    For `boundary_type == :OUTER_INNER`, the billiard is treated as a multiply connected domain:
    the first component is the outer boundary and the remaining components are holes.

# Returns
- `Vector{Bool}`: mask of points lying inside the billiard domain
"""
function points_in_billiard_polygon(pts::Vector{SVector{2,T}},billiard::Bi,N_polygon_checks::Int;fundamental_domain=true,boundary_type::Symbol=:OUTER) where {T<:Real,Bi<:AbsBilliard}
    polygon_xy_vectors=billiard_polygon(billiard,N_polygon_checks;fundamental_domain=fundamental_domain,boundary_type=boundary_type)
    mask=fill(false,length(pts))
    if boundary_type==:OUTER
        # single connected domain: concatenate all curve pieces into one polygon
        polygon_points=vcat(polygon_xy_vectors...)
        Threads.@threads for i in eachindex(pts)
            mask[i]=is_point_in_polygon(polygon_points,pts[i])
        end
    elseif boundary_type==:OUTER_INNER
        # multiply connected domain:
        # polygon_xy_vectors[1] = outer boundary
        # polygon_xy_vectors[2:end] = holes
        Threads.@threads for i in eachindex(pts)
            mask[i]=is_point_in_multiply_connected_polygon(polygon_xy_vectors,pts[i])
        end
    else
        error("Unknown boundary type: $boundary_type. Use :OUTER or :OUTER_INNER.")
    end
    return mask
end

###############################################################################
################# BOUNDARY INTEGRAL WAVEFUNCTION CONSTRUCTION #################
###############################################################################

"""
    ϕ(x::T, y::T, k::T, bd::BoundaryPoints{T}, u::AbstractVector{T}) where {T<:Real} -> T

Wavefunction via the boundary integral:
    Ψ(x,y) = (1/4) ∮ Y₀(k|q-q_s|) u(s) ds

Specialized for real `u` to keep everything in real arithmetic.
"""
@inline function ϕ(x::T,y::T,k::T,bd::BoundaryPoints{T},u::AbstractVector{T}) where {T<:Real}
    xy=bd.xy; ds=bd.ds
    s=zero(T)
    @inbounds @simd for j in eachindex(u)
        p=xy[j]
        r=hypot(x-p[1],y-p[2]) 
        y0=r<10^2*eps(T) ? zero(T) : Bessels.bessely0(k*r)
        s=muladd(y0*u[j],ds[j],s)
    end
    return s*T(0.25)
end

"""
    ϕ(x::T, y::T, k::T, bd::BoundaryPoints{T}, u::AbstractVector{Complex{T}}) where {T<:Real} -> Complex{T}

Same integral, but with complex boundary data `u`. Uses real kernel and
accumulates real/imag parts separately to avoid unnecessary complex multiplies.
"""
@inline function ϕ(x::T,y::T,k::T,bd::BoundaryPoints{T},u::AbstractVector{Complex{T}}) where {T<:Real}
    xy=bd.xy; ds=bd.ds
    sr=zero(T); si=zero(T)
    @inbounds @simd for j in eachindex(u)
        p=xy[j]
        r=hypot(x-p[1],y-p[2])
        w=r<10^2*eps(T) ? zero(T) : Bessels.bessely0(k*r)*ds[j] # real weight
        uj=u[j]
        sr=muladd(w,real(uj),sr)
        si=muladd(w,imag(uj),si)
    end
    return Complex(sr,si)*T(0.25)
end

"""
    ϕ_float32_bessel(x::T, y::T, k::T, bd::BoundaryPoints{T}, u::AbstractVector{T}) where {T<:Real} -> T

As `ϕ`, but calls `bessely0` in Float32 for speed; returns in `T`.
"""
@inline function ϕ_float32_bessel(x::T,y::T,k::T,bd::BoundaryPoints{T},u::AbstractVector{T}) where {T<:Real}
    xy=bd.xy; ds=bd.ds
    s=zero(T)
    @inbounds @simd for j in eachindex(u)
        p=xy[j]
        r=hypot(x-p[1],y-p[2])
        y0=r<10^2*eps(Float32) ? zero(T) : T(Bessels.bessely0(Float32(k*r))) # compute in Float32, cast back
        s=muladd(y0*u[j],ds[j],s)
    end
    return s*T(0.25)
end

"""
    ϕ_float32_bessel(x::T, y::T, k::T, bd::BoundaryPoints{T}, u::AbstractVector{Complex{T}}) where {T<:Real} -> Complex{T}

Float32-Bessel variant for complex `u`. Accumulates real/imag parts separately.
"""
@inline function ϕ_float32_bessel(x::T,y::T,k::T,bd::BoundaryPoints{T},u::AbstractVector{Complex{T}}) where {T<:Real}
    xy=bd.xy; ds=bd.ds
    sr=zero(T); si=zero(T)
    @inbounds @simd for j in eachindex(u)
        p=xy[j]
        r=hypot(x-p[1],y-p[2])
        w=r<10^2*eps(Float32) ? zero(T) : T(Bessels.bessely0(Float32(k*r)))*ds[j]
        uj=u[j]
        sr=muladd(w,real(uj),sr)
        si=muladd(w,imag(uj),si)
    end
    return Complex(sr,si)*T(0.25)
end

"""
    wavefunction_multi(ks::Vector{T},vec_us::Vector{Vector{T}},vec_bdPoints::Vector{BoundaryPoints{T}},billiard::Bi;b::Float64=5.0,inside_only::Bool=true,fundamental=true,MIN_CHUNK=4_096) where {Bi<:AbsBilliard,T<:Real}

Constructs a sequence of 2D wavefunctions as matrices over the same sized grid for easier computation of matrix elements. The matrices are constructed via the boundary integral.

# Arguments
- `ks`: Vector of eigenvalues.
- `vec_bdPoints`: Vector of `BoundaryPoints` objects, one for each eigenvalues.
- `billiard`: The billiard geometry.
- `vec_us::Vector{Vector}`: Vector of the boundary functions. Can be either complex or real, this determines whether the wavefunction is real or not.
- `b::Float64=5.0`: (Optional), Point scaling factor. Default is 5.0.
- `inside_only::Bool=true`: (Optional), Whether to only compute wavefunctions inside the billiard. Default is true.
- `fundamental::Bool=true`: (Optional), Whether to use fundamental domain for boundary integral. Default is true.
- `MIN_CHUNK::Int=4096`: keep ≥ this many boundary points per thread

# Returns
- `Psi2ds::Vector{Matrix{T}}`: Vector of 2D wavefunction matrices constructed on the same grid.
- `x_grid::Vector{T}`: Vector of x-coordinates for the grid.
- `y_grid::Vector{T}`: Vector of y-coordinates for the grid.
"""
function wavefunction_multi(ks::Vector{T},vec_us::Vector{<:AbstractVector},vec_bdPoints::Vector{BoundaryPoints{T}},billiard::Bi;b::Float64=5.0,inside_only::Bool=true,fundamental=true,MIN_CHUNK=4_096) where {Bi<:AbsBilliard,T<:Real}
    k_max=maximum(ks)
    type=eltype(k_max)
    L=billiard.length
    if fundamental
        xlim,ylim=boundary_limits(billiard.fundamental_boundary;grd=max(1000,round(Int,k_max*L*b/(2*pi))))
    else
        xlim,ylim=boundary_limits(billiard.full_boundary;grd=max(1000,round(Int,k_max*L*b/(2*pi))))
    end
    dx,dy=xlim[2]-xlim[1],ylim[2]-ylim[1]
    nx,ny=max(round(Int,k_max*dx*b/(2*pi)),512),max(round(Int,k_max*dy*b/(2*pi)),512)
    x_grid,y_grid=collect(type,range(xlim..., nx)),collect(type,range(ylim..., ny))
    pts=collect(SVector(x,y) for y in y_grid for x in x_grid)
    sz=length(pts)
    # Determine points inside the billiard only once if inside_only is true
    pts_mask=inside_only ? points_in_billiard_polygon(pts,billiard,round(Int,sqrt(sz));fundamental_domain=fundamental) : fill(true,sz)
    pts_masked_indices=findall(pts_mask)
    NT=Threads.nthreads()
    nmask=length(pts_masked_indices)
    S=eltype(vec_us[1])<:Real ? type : Complex{type}
    Psi_flat=zeros(S,nx*ny) # overwritten each iteration since pts_masked_indices is the same for each k in ks
    NT_eff=max(1,min(NT,cld(nmask,MIN_CHUNK)))
    Psi2ds=Vector{Matrix{S}}(undef,length(ks))
    progress=Progress(length(ks),desc="Constructing wavefunction matrices...")
    q,r=divrem(nmask,NT_eff)
    @inbounds for i in eachindex(ks)
        k,bdPoints,us=ks[i],vec_bdPoints[i],vec_us[i]
        @fastmath begin
            Threads.@threads :static for t in 1:NT_eff
                # compute this thread's block [lo:hi]
                lo=(t-1)*q+min(t-1,r) + 1
                hi=lo+q-1+(t<=r ? 1 : 0)
                @inbounds for jj in lo:hi
                    idx=pts_masked_indices[jj] # each interior point [idx] -> (x,y)
                    x,y=pts[idx]
                    Psi_flat[idx]=ϕ_float32_bessel(x,y,k,bdPoints,us) # Do it with floating point bessel computation, no need for double_precision here, and only for interior points
                end
            end
        end
        Psi2ds[i]=copy(reshape(Psi_flat,nx,ny))
        next!(progress)
    end
    return Psi2ds,x_grid,y_grid
end

"""
    wavefunction_multi_with_husimi(ks::Vector{T},vec_us::Vector{Vector{T}},vec_bdPoints::Vector{BoundaryPoints{T}},billiard::Bi;b::Float64=5.0, inside_only::Bool=true,fundamental=true,use_fixed_grid=true,xgrid_size=2000,ygrid_size=1000,MIN_CHUNK=4_096) where {Bi<:AbsBilliard,T<:Real}

Constructs a sequence of 2D wavefunctions as matrices over the same sized grid for easier computation of matrix elements. The matrices are constructed via the boundary integral. Additionally also constructs the husimi functions.

# Arguments
- `ks`: Vector of eigenvalues.
- `vec_bdPoints`: Vector of `BoundaryPoints` objects, one for each eigenvalues.
- `billiard`: The billiard geometry.
- `vec_us::Vector{Vector}`: Vector of the boundary functions. Can be either complex or real, this determines whether the wavefunction is real or not.
- `b::Float64=5.0`: (Optional), Point scaling factor. Default is 5.0.
- `inside_only::Bool=true`: (Optional), Whether to only compute wavefunctions inside the billiard. Default is true.
- `fundamental::Bool=true`: (Optional), Whether to use fundamental domain for boundary integral. Default is true.
- `xgrid_size::Int=2000`: (Optional), Size of the x grid for the husimi functions. Default is 2000.
- `ygrid_size::Int=1000`: (Optional), Size of the y grid for the husimi functions. Default is 1000.
- `use_fixed_grid::Bool=true`: (Optional), Whether to use a fixed grid for the husimi functions. Default is true.
- `MIN_CHUNK::Int=4096`: keep ≥ this many boundary points per thread

# Returns
- `Psi2ds::Vector{Matrix{T}}`: Vector of 2D wavefunction matrices constructed on the same grid.
- `x_grid::Vector{T}`: Vector of x-coordinates for the grid.
- `y_grid::Vector{T}`: Vector of y-coordinates for the grid.
- `Hs_list::Vector{Matrix{T}}`: Vector of 2D husimi function matrices.
- `ps_list::Vector{Vector{T}}`: Vector of ps grids for the husimi matrices.
- `qs_list::Vector{Vector{T}}`: Vector of qs grids for the husimi matrices.
"""
function wavefunction_multi_with_husimi(ks::Vector{T},vec_us::Vector{<:AbstractVector},vec_bdPoints::Vector{BoundaryPoints{T}},billiard::Bi;b::Float64=5.0, inside_only::Bool=true,fundamental=true,use_fixed_grid=true,xgrid_size=2000,ygrid_size=1000,MIN_CHUNK=4_096) where {Bi<:AbsBilliard,T<:Real}
    k_max=maximum(ks)
    type=eltype(k_max)
    L=billiard.length
    if fundamental
        xlim,ylim=boundary_limits(billiard.fundamental_boundary;grd=max(1000,round(Int,k_max*L*b/(2*pi))))
    else
        xlim,ylim=boundary_limits(billiard.full_boundary;grd=max(1000,round(Int,k_max*L*b/(2*pi))))
    end
    dx,dy=xlim[2]-xlim[1],ylim[2]-ylim[1]
    nx,ny=max(round(Int,k_max*dx*b/(2*pi)),512),max(round(Int,k_max*dy*b/(2*pi)),512)
    x_grid,y_grid=collect(type,range(xlim..., nx)),collect(type,range(ylim..., ny))
    pts=collect(SVector(x,y) for y in y_grid for x in x_grid)
    sz=length(pts)
    # Determine points inside the billiard only once if inside_only is true
    pts_mask=inside_only ? points_in_billiard_polygon(pts,billiard,round(Int,sqrt(sz));fundamental_domain=fundamental) : fill(true,sz)
    pts_masked_indices=findall(pts_mask)
    NT=Threads.nthreads()
    nmask=length(pts_masked_indices)
    S=eltype(vec_us[1])<:Real ? type : Complex{type}
    Psi_flat=zeros(S,nx*ny) # overwritten each iteration since pts_masked_indices is the same for each k in ks
    NT_eff=max(1,min(NT,cld(nmask,MIN_CHUNK)))
    Psi2ds=Vector{Matrix{S}}(undef,length(ks))
    progress=Progress(length(ks),desc="Constructing wavefunction matrices...")
    q,r=divrem(nmask,NT_eff)
    for i in eachindex(ks)
        k,bdPoints,us=ks[i],vec_bdPoints[i],vec_us[i]
        @fastmath begin
            Threads.@threads :static for t in 1:NT_eff
                # compute this thread's block [lo:hi]
                lo=(t-1)*q+min(t-1,r) + 1
                hi=lo+q-1+(t<=r ? 1 : 0)
                @inbounds for jj in lo:hi
                    idx=pts_masked_indices[jj] # each interior point [idx] -> (x,y)
                    x,y=pts[idx]
                    Psi_flat[idx]=ϕ_float32_bessel(x,y,k,bdPoints,us) # Do it with floating point bessel computation, no need for double_precision here, and only for interior points
                end
            end
        end
        Psi2ds[i]=copy(reshape(Psi_flat,nx,ny))
        next!(progress)
    end
    vec_of_s_vals=[bdPoints.s for bdPoints in vec_bdPoints]
    if use_fixed_grid
        Hs_list,ps,qs=husimi_functions_from_us_and_boundary_points_FIXED_GRID(ks,vec_us,vec_bdPoints,billiard,xgrid_size,ygrid_size)
        ps_list=fill(ps,length(Hs_list))
        qs_list=fill(qs,length(Hs_list))
    else
        Hs_list,ps_list,qs_list=husimi_functions_from_boundary_functions(ks,vec_us,vec_of_s_vals,billiard)
    end
    return Psi2ds,x_grid,y_grid,Hs_list,ps_list,qs_list
end

###########################################################################
############################ CFIE_kress CONSTRUCTION ############################
###########################################################################

# Flattne the CFIE_kress boundary points into a single cache for faster wavefunction reconstruction, and then evaluate the CFIE_kress wavefunction at many points from the flattened cache and boundary density `u`.
struct CFIEWavefunctionCache{T<:Real}
    x::Vector{T} # boundary x_j
    y::Vector{T} # boundary y_j
    tx::Vector{T} # tangent x-component
    ty::Vector{T} # tangent y-component
    sj::Vector{T} # |tangent_j|
    w::Vector{T} # quadrature weight w_j
end

"""
    flatten_cfie_wavefunction_cache(comps::Vector{BoundaryPointsCFIE{T}}) where {T<:Real} -> CFIEWavefunctionCache{T}

Flattens the CFIE_kress boundary points from the fundamental domain into a single cache for faster wavefunction reconstruction.

# Inputs:
- `comps`: Vector of `BoundaryPointsCFIE` objects, one for each component of the boundary.

# Outputs:
- `CFIEWavefunctionCache{T}`: A struct containing flattened vectors of boundary coordinates, tangents, quadrature weights, etc., for efficient CFIE_kress wavefunction evaluation.
"""
function flatten_cfie_wavefunction_cache(comps::Vector{BoundaryPointsCFIE{T}}) where {T<:Real}
    N=sum(length(c.xy) for c in comps)
    x=Vector{T}(undef,N)
    y=Vector{T}(undef,N)
    tx=Vector{T}(undef,N)
    ty=Vector{T}(undef,N)
    sj=Vector{T}(undef,N)
    w=Vector{T}(undef,N)
    p=1
    @inbounds for c in comps
        for j in eachindex(c.xy)
            q=c.xy[j]
            t=c.tangent[j]
            txj=t[1]
            tyj=t[2]
            x[p]=q[1]
            y[p]=q[2]
            tx[p]=txj
            ty[p]=tyj
            sj[p]=sqrt(txj*txj+tyj*tyj)
            w[p]=c.ws[j]
            p+=1
        end
    end
    return CFIEWavefunctionCache(x,y,tx,ty,sj,w)
end

"""
    ϕ_cfie(xp::T, yp::T, k::T,cache::CFIEWavefunctionCache{T},u::AbstractVector{Complex{T}};float32_bessel::Bool=false) where {T<:Real} -> Complex{T}

Evaluate the CFIE_kress-reconstructed wavefunction at point `(xp, yp)` from a
flattened boundary cache and boundary density `u`.

This uses the same kernel as the CFIE_kress assembly:

    ψ(x) = -∑_j w_j u_j [ (i k / 2) * inn * H1(k r) / r + i k * (i/2) * H0(k r) * s_j ]

where
    inn = t_y (x-x_j) - t_x (y-y_j)

# Arguments
- `xp, yp` : evaluation point p = SVector(xp, yp)
- `k::T`      : real wavenumber
- `cache::CFIEWavefunctionCache{T}`  : flattened CFIE_kress geometry cache
- `u::AbstractVector{Complex{T}}`      : complex boundary density, same ordering as flattening
- `float32_bessel::Bool`     : evaluate Hankels in Float32 and cast back

# Returns
- `Complex{T}`: the reconstructed wavefunction value at (xp, yp)
"""
@inline function ϕ_cfie(xp::T,yp::T,k::T,cache::CFIEWavefunctionCache{T},u::AbstractVector{Complex{T}};float32_bessel::Bool=false) where {T<:Real}
    x=cache.x
    y=cache.y
    tx=cache.tx
    ty=cache.ty
    sj=cache.sj
    w=cache.w
    N=length(x)
    @assert length(u)==N
    ψr=zero(T)
    ψi=zero(T)
    # Constants:
    # dterm = (i k / 2) * inn * H1 / r
    # sterm = (i / 2) * H0 * sj
    # contribution = -(w*u) * (dterm + i k * sterm)
    #
    # Since i*k*sterm = i*k*(i/2) H0 sj = -(k/2) H0 sj,
    # the kernel is
    #   K = (i k / 2) * inn * H1 / r  -  (k / 2) * H0 * sj
    khalf=k*T(0.5)
    tol2=(T(100)*eps(T))^2 # for near boundary skipping, squared for distance comparison
    @inbounds @fastmath for j in 1:N
        dx=xp-x[j]
        dy=yp-y[j]
        r2=muladd(dx,dx,dy*dy)
        r2<=tol2 && continue # skip near-boundary points
        r=sqrt(r2)
        invr=inv(r)
        inn=muladd(ty[j],dx,-(tx[j]*dy)) # ty*dx - tx*dy
        if float32_bessel
            zf=Float32(k*r)
            h0=Complex{T}(Bessels.hankelh1(0,zf))
            h1=Complex{T}(Bessels.hankelh1(1,zf))
        else
            z=k*r
            h0=Bessels.hankelh1(0,z)
            h1=Bessels.hankelh1(1,z)
        end
        # Kernel:
        # K = (i k / 2) * inn * H1 / r - (k / 2) * H0 * sj
        #
        # Let A = (k/2) * inn/r, B = (k/2) * sj
        # Then
        #   K = i*A*h1 - B*h0
        A=khalf*inn*invr
        B=khalf*sj[j]
        # i*A*h1 = (-A*imag(h1)) + i*(A*real(h1))
        Kr=muladd(-A,imag(h1),-B*real(h0))
        Ki=muladd(A,real(h1),-B*imag(h0))
        # contribution = -(w*u)*K
        uj=u[j]
        wr=w[j]*real(uj)
        wi=w[j]*imag(uj)
        # -(wr+i wi)(Kr+i Ki)
        ψr-= wr*Kr-wi*Ki
        ψi-= wr*Ki+wi*Kr
    end
    return Complex{T}(ψr, ψi)
end

"""
    ϕ_cfie!(ψ::AbstractVector{Complex{T}},pts::AbstractVector{SVector{2,T}},k::T,cache::CFIEWavefunctionCache{T},u::AbstractVector{Complex{T}},float32_bessel::Bool=false) where {T<:Real}

Compute the CFIE_kress wavefunction on many points.
"""
function ϕ_cfie_flat!(ψ::AbstractVector{Complex{T}},pts::AbstractVector,k::T,cache::CFIEWavefunctionCache{T},u::AbstractVector{Complex{T}};float32_bessel::Bool=false) where {T<:Real}
    Threads.@threads for i in eachindex(pts)
        p=pts[i]
        ψ[i]=ϕ_cfie_flat(p[1],p[2],k,cache,u;float32_bessel=float32_bessel)
    end
    return ψ
end

"""
        phase_fix_real(psi::AbstractArray{Complex{T}}) where {T<:Real} -> (AbstractArray{Complex{T}}, T)

Given a complex wavefunction `psi` that is expected to be real up to a global phase, find the global phase that makes it as close to real as possible and return the phase-corrected wavefunction along with the phase angle. Plotting separately the real and imaginary parts of the original and phase-corrected wavefunctions can be useful for verifying that the phase correction worked as intended.

# Arguments
- `psi::AbstractArray{Complex{T}}`: The input complex wavefunction array.

# Returns
- `(AbstractArray{Complex{T}}, T)`: A tuple containing the phase-corrected wavefunction and the phase angle that was applied. The corrected wavefunction should have a zero reduced imaginary part if the original wavefunction is correct.
"""
function phase_fix_real(psi::AbstractArray{Complex{T}}) where {T<:Real}
    s=sum(psi.^2)
    θ=0.5*Base.angle(s)
    return psi.*exp(-im*θ),θ
end

"""
    wavefunction_multi_cfie(ks::Vector{T},vec_us::Vector{<:AbstractVector},vec_comps::Vector{Vector{BoundaryPointsCFIE{T}}},billiard::Bi;b::Float64=5.0,inside_only::Bool=true,fundamental::Bool=false,MIN_CHUNK::Int=4096,float32_bessel::Bool=false) where {Bi<:AbsBilliard,T<:Real}

Construct a sequence of 2D wavefunction matrices for CFIE_kress states on a common grid.

# Arguments
- `ks`         : eigenvalues
- `vec_us`     : boundary densities, one per eigenstate
- `vec_comps`  : CFIE_kress boundary discretizations, one per eigenstate
- `billiard`   : billiard geometry

# Keyword arguments
- `b`                  : grid density scaling
- `inside_only`        : compute only inside the billiard
- `fundamental`        : use fundamental domain limits if desired
- `MIN_CHUNK`          : minimum masked points per thread chunk
- `float32_bessel`     : use Float32 Hankel evaluations

# Returns
- `Psi2ds` : vector of wavefunction matrices
- `x_grid` : x-coordinates of the grid
- `y_grid` : y-coordinates of the grid
"""
function wavefunction_multi(ks::Vector{T},vec_us::Vector{<:AbstractVector},vec_comps::Vector{Vector{BoundaryPointsCFIE{T}}},billiard::Bi;b::Float64=5.0,inside_only::Bool=true,fundamental::Bool=false,MIN_CHUNK::Int=4096,float32_bessel::Bool=false) where {Bi<:AbsBilliard,T<:Real}
    kmax=maximum(ks)
    L=billiard.length
    xlim,ylim=boundary_limits(billiard.full_boundary;grd=max(1000,round(Int,kmax*L*b/(2*pi)))) # this accepts vector of curves, so it works for cfie as well
    dx=xlim[2]-xlim[1]
    dy=ylim[2]-ylim[1]
    nx=max(round(Int,kmax*dx*b/(2π)),512)
    ny=max(round(Int,kmax*dy*b/(2π)),512)
    x_grid=collect(T,range(xlim[1],xlim[2],length=nx))
    y_grid=collect(T,range(ylim[1],ylim[2],length=ny))
    pts=collect(SVector(x,y) for y in y_grid for x in x_grid)
    npts=length(pts)
    pts_mask=inside_only ? points_in_billiard_polygon(pts,billiard,round(Int,sqrt(npts));fundamental_domain=false,boundary_type=:OUTER_INNER) : fill(true,npts)
    pts_masked_indices=findall(pts_mask)
    nmask=length(pts_masked_indices)
    NT=Threads.nthreads()
    NT_eff=max(1,min(NT,cld(nmask,MIN_CHUNK)))
    S=eltype(vec_us[1])
    nstates=length(ks)
    Psi2ds=Vector{Matrix{S}}(undef,nstates)
    caches=Vector{CFIEWavefunctionCache{T}}(undef,nstates)
    @inbounds for i in 1:nstates # Flatten caches once per state
        caches[i]=flatten_cfie_wavefunction_cache(vec_comps[i])
    end
    Psi_flat=zeros(S,nx*ny) # flat workspace reused per state
    progress=Progress(nstates,desc="Constructing CFIE_kress wavefunction matrices...")
    q,r=divrem(nmask,NT_eff)
    @inbounds for i in eachindex(ks)
        k=ks[i]
        cache=caches[i]
        us=vec_us[i]
        fill!(Psi_flat,zero(S))
        @fastmath begin
            Threads.@threads :static for t in 1:NT_eff
                lo=(t-1)*q+min(t-1,r)+1
                hi=lo+q-1+(t<=r ? 1 : 0)
                for jj in lo:hi
                    idx=pts_masked_indices[jj]
                    p=pts[idx]
                    Psi_flat[idx]=ϕ_cfie(p[1],p[2],k,cache,us;float32_bessel=float32_bessel)
                end
            end
        end
        Psi2ds[i]=copy(reshape(Psi_flat,nx,ny))
        next!(progress)
    end
    for i in eachindex(Psi2ds)
        Psi2ds[i],_=phase_fix_real(Psi2ds[i])
        nrm=sqrt(sum(abs2,Psi2ds[i][pts_masked_indices]))
        Psi2ds[i]./=nrm
    end
    return Psi2ds,x_grid,y_grid
end

###########################################################################
###########################################################################
###########################################################################

# Helpers for plotting billiard boundaries and curves

function plot_curve!(ax,crv::AbsRealCurve;plot_normal=true,dens=20.0,color_crv=:grey,linewidth=0.75)
    L=crv.length
    grid=max(round(Int,L*dens),3)
    t=range(0.0,1.0,grid)
    pts=curve(crv,t)
    lines!(ax,pts,color=color_crv,linewidth=linewidth)
    if plot_normal
        ns=normal_vec(crv,t)
        arrows!(ax,getindex.(pts,1),getindex.(pts,2),getindex.(ns,1),getindex.(ns,2),color=:black,lengthscale=0.1)
    end
    ax.aspect=DataAspect()
end

function plot_curve!(ax,crv::AbsVirtualCurve;plot_normal=false,dens=10.0,color_crv=:grey,linewidth=0.75)
    L=crv.length
    grid=max(round(Int,L*dens),3)
    t=range(0.0,1.0,grid)
    pts=curve(crv,t)
    lines!(ax,pts,color=color_crv,linestyle=:dash,linewidth=linewidth)
    if plot_normal
        ns=normal_vec(crv,t)
        arrows!(ax,getindex.(pts,1),getindex.(pts,2),getindex.(ns,1),getindex.(ns,2),color=:black,lengthscale=0.1)
    end
    ax.aspect=DataAspect()
end

function plot_boundary!(ax,billiard::AbsBilliard;fundamental_domain=true,desymmetrized_full_domain=false,dens=100.0,plot_normal=true,color_crv=:grey,linewidth=0.75)
    if fundamental_domain
        boundary=billiard.fundamental_boundary
    elseif desymmetrized_full_domain
        boundary=billiard.desymmetrized_full_boundary
    else
        boundary=billiard.full_boundary
    end
    for curve in boundary 
        plot_curve!(ax,curve;dens=dens,plot_normal=plot_normal,color_crv=color_crv,linewidth=linewidth)
    end
end

"""
    batch_wrapper(plot_func::Function, args...; N::Integer=100, kwargs...)

Splits a large dataset into batches and calls the provided plotting function on each batch. 

This is useful when plotting a large number of wavefunctions or other data items at once would 
either be too large or time-consuming. By batching, you can generate multiple figures, each 
containing a subset of the data.

# Arguments
- `plot_func::Function`: The plotting function to be called for each batch.
- `args...`: The argument lists. The first argument should be a vector (e.g., `ks`) that 
   determines the number of data items. All other arguments must also be indexable and have 
   a compatible length.
- `N::Integer=100`: Number of items per batch.
- `kwargs...`: Additional keyword arguments passed on to `plot_func`.

# Returns
- `figures::Vector{Figure}`: A vector of `Figure` objects, each produced by `plot_func` 
   on a batch of data.
"""
function batch_wrapper(plot_func::Function, args...; N::Integer=100, kwargs...)
    # Extract the data vectors and the number of items
    ks=args[1] # ks is always the first argument
    @assert length(ks)>0 "ks cannot be empty."
    num_batches=ceil(Int,length(ks)/N)
    figures=Vector{Figure}(undef,num_batches)
    for i in 1:num_batches
        start_idx=(i-1)*N+1
        end_idx=min(i*N,length(ks))
        range=start_idx:end_idx
        batched_args=map(arg-> 
        if arg isa AbstractVector  # check if the argument is a vector
            if length(arg)==length(ks)  # slice if it matches `ks`
                arg[range]
            else
                arg  # x_grid, y_grid
            end
        else
            arg # legacy for billiard arg, remove later
        end,args)
        figures[i]=plot_func(batched_args...;kwargs...) # Call the original plotting function for the batch
    end
    return figures
end

"""
    partition_vector(ks::Vector{<:Real}, N::Integer)

Partitions the ks::Vector into N chunks. This is a helper function that helps us map the partitioned figures from the plotting functions (which give us Vector{Figure}) with the corresponding k values they contain. It partitians ks=[k1,k2,...,k_m] as vectors of length N [[k1...k_n],[k_n+1,...,k_2n],.... It is compatible with N=1 (just returns each k as a separate 1-element vector) and length(ks) < N, in which case it return the input vector unchanged.

# Arguments
- `ks::Vector{<:Real}`: The vector of eigenvalues to be partitioned.
- `N::Integer`: The number of chunks to partition the ks vector into.

# Returns
- `partitions::Vector{Vector{<:Real}}`: A vector of vectors, each containing N elements from the input ks vector.
"""
function partition_vector(ks::Vector, N::Integer)
    partitions=[ks[i:min(i+N-1,length(ks))] for i in 1:N:length(ks)]
    return partitions
end

"""
    plot_wavefunctions_BATCH(ks::Vector, Psi2ds::Vector, x_grid::Vector, y_grid::Vector, billiard::Bi; b::Float64=5.0, width_ax::Integer=500, height_ax::Integer=500, max_cols::Integer=6, fundamental=true) where {Bi<:AbsBilliard}

Plots the wavefunctions into a grid (only the fundamental boundary). The x_grid and y_grid is supplied from the wavefunction_multi or a similar function.

# Arguments
- `ks`: Vector of eigenvalues.
- `Psi2ds::Vector{Matrix}`: Vector of 2D wavefunction matrices.
- `x_grid::Vector{<:Real}`: Vector of x-coordinates for the grid.
- `y_grid::Vector{<:Real}`: Vector of y-coordinates for the grid.
- `billiard<:AbsBilliard`: The billiard geometry.
- `b::Float64=5.0`: The point scaling factor.
- `width_ax::Integer=500`: The size of each axis in the grid layout.
- `height_ax::Integer=500`: The size of each axis in the grid layout.
- `max_cols::Integer=6`: The maximum number of columns in the grid layout.
- `fundamental::Bool=true`: If plotting just the desymmetrized part.
- `custom_label::Vector{String}`: The labels to be plotted for each Axis in the Figure. ! Needs to be the same length as ks, as it should be unique to each k in ks !.

 # Returns
- `f::Figure`: A Figure object containing the grid of wavefunctions.
"""
function plot_wavefunctions_BATCH(ks::Vector, Psi2ds::Vector, x_grid::Vector, y_grid::Vector, billiard::Bi; b::Float64=5.0, width_ax::Integer=300, height_ax::Integer=300, max_cols::Integer=6, fundamental=true, custom_label::Vector{String}=String[]) where {Bi<:AbsBilliard}
    for i in eachindex(Psi2ds)
        ψ=Psi2ds[i]
        s=maximum(abs,ψ)
        s= s>0 ? s : 1
        @. ψ=sign(real(ψ)/s)*abs(real(ψ)/s)^0.5
    end
    L=billiard.length
    if fundamental
        xlim,ylim=boundary_limits(billiard.fundamental_boundary;grd=max(1000,round(Int,maximum(ks)*L*b/(2*pi))))
    else
        xlim,ylim=boundary_limits(billiard.full_boundary;grd=max(1000,round(Int,maximum(ks)*L*b/(2*pi))))
    end
    n_rows=ceil(Int,length(ks)/max_cols)
    f = Figure(resolution=(round(Int,1.5*width_ax*max_cols),round(Int,2*height_ax*n_rows)),size=(round(Int,1.5*width_ax*max_cols),round(Int,2*height_ax*n_rows)))
    row=1
    col=1
    @showprogress desc="Plotting wavefunctions..." for j in eachindex(ks)
        title= isempty(custom_label) ? "$(ks[j])" : custom_label[j]
        local ax=Axis(f[row,col],title=title,aspect=DataAspect(),width=width_ax,height=height_ax)
        hm=heatmap!(ax,x_grid,y_grid,Psi2ds[j],colormap=:balance,colorrange=(-1,1))
        plot_boundary!(ax,billiard,fundamental_domain=fundamental,plot_normal=false)
        xlims!(ax,xlim)
        ylims!(ax,ylim)
        col+=1
        if col>max_cols
            row+=1
            col=1
        end
    end
    return f
end

"""
    plot_wavefunctions_BATCH(ks::Vector, Psi2ds::Vector, x_grid::Vector{Vector}, y_grid::Vector{Vector}, billiard::Bi; b::Float64=5.0, width_ax::Integer=500, height_ax::Integer=500, max_cols::Integer=6, fundamental=true) where {Bi<:AbsBilliard}

Plots the wavefunctions into a grid (only the fundamental boundary). The x_grid and y_grid is supplied from the `wavefunctions` method since it expects for each wavefunctions it's separate x and y grid.

# Arguments
- `ks`: Vector of eigenvalues.
- `Psi2ds::Vector{Matrix}`: Vector of 2D wavefunction matrices.
- `x_grid::Vector{Vector}`: Vector of x-coordinates for the grid for each wavefunction.
- `y_grid::Vector{Vector}`: Vector of y-coordinates for the grid for each wavefunction.
- `billiard<:AbsBilliard`: The billiard geometry.
- `b::Float64=5.0`: The point scaling factor.
- `width_ax::Integer=500`: The size of each axis in the grid layout.
- `height_ax::Integer=500`: The size of each axis in the grid layout.
- `max_cols::Integer=6`: The maximum number of columns in the grid layout.
- `fundamental::Bool=true`: If plotting just the desymmetrized part.
- `custom_label::Vector{String}`: The labels to be plotted for each Axis in the Figure. ! Needs to be the same length as ks, as it should be unique to each k in ks !.

 # Returns
- `f::Figure`: A Figure object containing the grid of wavefunctions.
"""
function plot_wavefunctions_BATCH(ks::Vector, Psi2ds::Vector, x_grid::Vector{Vector}, y_grid::Vector{Vector}, billiard::Bi; b::Float64=5.0, width_ax::Integer=300, height_ax::Integer=300, max_cols::Integer=6, fundamental=true, custom_label::Vector{String}=String[]) where {Bi<:AbsBilliard}
    for i in eachindex(Psi2ds)
        ψ=Psi2ds[i]
        amax=maximum(abs,ψ)
        if amax>0
            @. ψ=real(ψ)/amax
        else
            @. ψ=real(ψ)
        end
    end
    L=billiard.length
    if fundamental
        xlim,ylim=boundary_limits(billiard.fundamental_boundary;grd=max(1000,round(Int,maximum(ks)*L*b/(2*pi))))
    else
        xlim,ylim=boundary_limits(billiard.full_boundary;grd=max(1000,round(Int,maximum(ks)*L*b/(2*pi))))
    end
    n_rows=ceil(Int,length(ks)/max_cols)
    f=Figure(resolution=(round(Int,1.5*width_ax*max_cols),round(Int,2*height_ax*n_rows)),size=(round(Int,1.5*width_ax*max_cols),round(Int,2*height_ax*n_rows)))
    row=1
    col=1
    @showprogress desc="Plotting wavefunctions..." for j in eachindex(ks)
        title= isempty(custom_label) ? "$(ks[j])" : custom_label[j]
        local ax=Axis(f[row,col],title=title,aspect=DataAspect(),width=width_ax,height=height_ax)
        hm=heatmap!(ax,x_grid[j],y_grid[j],Psi2ds[j],colormap=:balance,colorrange=(-1,1))
        plot_boundary!(ax,billiard,fundamental_domain=fundamental,plot_normal=false)
        xlims!(ax,xlim)
        ylims!(ax,ylim)
        col+=1
        if col>max_cols
            row+=1
            col=1
        end
    end
    return f
end

"""
    plot_wavefunctions(ks::Vector, Psi2ds::Vector, x_grid::Vector, y_grid::Vector, billiard::Bi; N::Integer=100, kwargs...) where {Bi<:AbsBilliard}

Plots the wavefunctions specified by `Psi2ds` on the domain defined by `x_grid` and `y_grid`, 
for the billiard geometry `billiard`. The eigenvalues are provided in `ks`. When the number of 
wavefunctions is large, this function automatically splits the data into batches of size `N` 
and generates multiple figures.

# Arguments
- `ks::Vector`: Vector of eigenvalues.
- `Psi2ds::Vector{Matrix}`: Vector of 2D wavefunction matrices corresponding to `ks`.
- `x_grid::Vector{<:Real}`: Vector of x-coordinates for the grid.
- `y_grid::Vector{<:Real}`: Vector of y-coordinates for the grid.
- `billiard::Bi<:AbsBilliard`: The billiard geometry.
- `N::Integer=100`: The number of items per batch. If `length(ks) > N`, multiple figures are produced.
- `kwargs...`: Additional keyword arguments passed to the underlying plotting function (N axes per Figure and custom label, check _BATCH function)

# Returns
- `figures::Vector{Figure}`: A vector of `Figure` objects, one per batch.
"""
function plot_wavefunctions(ks::Vector, Psi2ds::Vector, x_grid::Vector, y_grid::Vector, billiard::Bi; N::Integer=100, kwargs...) where {Bi<:AbsBilliard}
    batch_wrapper(plot_wavefunctions_BATCH,ks,Psi2ds,x_grid,y_grid,billiard;N=N,kwargs...)
end

"""
    plot_wavefunctions(ks::Vector, Psi2ds::Vector, x_grid::Vector{Vector}, y_grid::Vector{Vector}, billiard::Bi; N::Integer=100, kwargs...) where {Bi<:AbsBilliard}

Similar to `plot_wavefunctions` above, but this version allows for a distinct `(x_grid, y_grid)` 
for each wavefunction in `Psi2ds`. This is useful if each wavefunction was computed on a different 
grid. Automatically splits the data into batches of size `N` if `ks` is large.

# Arguments
- `ks::Vector`: Vector of eigenvalues.
- `Psi2ds::Vector{Matrix}`: Vector of 2D wavefunction matrices, one for each `(x_grid[j], y_grid[j])`.
- `x_grid::Vector{Vector{<:Real}}`: A vector of x-coordinate vectors, one per wavefunction.
- `y_grid::Vector{Vector{<:Real}}`: A vector of y-coordinate vectors, one per wavefunction.
- `billiard::Bi<:AbsBilliard`: The billiard geometry.
- `N::Integer=100`: The number of items per batch.
- `kwargs...`: Additional keyword arguments passed to the underlying plotting function (N axes per Figure and custom label, check _BATCH function)

# Returns
- `figures::Vector{Figure}`: A vector of `Figure` objects, one per batch of wavefunctions.
"""
function plot_wavefunctions(ks::Vector, Psi2ds::Vector, x_grid::Vector{Vector}, y_grid::Vector{Vector}, billiard::Bi; N::Integer=100, kwargs...) where {Bi<:AbsBilliard}
    batch_wrapper(plot_wavefunctions_BATCH,ks,Psi2ds,x_grid,y_grid,billiard;N=N,kwargs...)
end

"""
    plot_wavefunctions_BATCH(ks::Vector, Psi2ds::Vector, x_grid::Vector, y_grid::Vector, billiard::Bi; b::Float64=5.0, width_ax::Integer=500, height_ax::Integer=500, max_cols::Integer=6) where {Bi<:AbsBilliard}

Plots the wavefunctions into a grid (only the fundamental boundary) together with the respective husimi function matrices on the provided grids. The x_grid and y_grid is supplied from the wavefunction_multi or a similar function, and the ps and qs grids mudt also be supplied for plotting the Husimi functions.

# Arguments
- `ks`: Vector of eigenvalues.
- `Psi2ds::Vector{Matrix}`: Vector of 2D wavefunction matrices.
- `x_grid::Vector{<:Real}`: Vector of x-coordinates for the grid.
- `y_grid::Vector{<:Real}`: Vector of y-coordinates for the grid.
- `Hs_list::Vector{Matrix}`: Vector of 2D husimi function matrices.
- `ps_list::Vector{Vector}`: Vector of ps grids for the husimi matrices.
- `qs_list::Vector{Vector}`: Vector of qs grids for the husimi matrices.
- `billiard<:AbsBilliard`: The billiard geometry.
- `b::Float64=5.0`: The point scaling factor.
- `width_ax::Integer=500`: The size of each axis in the grid layout.
- `height_ax::Integer=500`: The size of each axis in the grid layout.
- `max_cols::Integer=6`: The maximum number of columns in the grid layout.
- `fundamental::Bool=true`: If plotting just the desymmetrized part.
- `custom_label::Vector{String}`: The labels to be plotted for each Axis in the Figure. ! Needs to be the same length as ks, as it should be unique to each k in ks !.
- `use_projection_grid::Tuple{Vector,Vector}=([],[])`: A tuple containing the classical s and p values of the chaotic trajectory (in that order). These are used to construct the chaotic mask overlay so we can better observe the overlaps.

 # Returns
- `f::Figure`: A Figure object containing the grid of wavefunctions.
"""
function plot_wavefunctions_with_husimi_BATCH(ks::Vector, Psi2ds::Vector, x_grid::Vector, y_grid::Vector, Hs_list::Vector, ps_list::Vector, qs_list::Vector, billiard::Bi; b::Float64=5.0, width_ax::Integer=300, height_ax::Integer=300, max_cols::Integer=6, fundamental=true, custom_label::Vector{String}=String[], use_projection_grid::Tuple{Vector,Vector}=([],[])) where {Bi<:AbsBilliard}
    for i in eachindex(Psi2ds)
        ψ=Psi2ds[i]
        amax=maximum(abs,ψ)
        if amax>0
            @. ψ=real(ψ)/amax
        else
            @. ψ=real(ψ)
        end
    end
    L=billiard.length
    if fundamental
        xlim,ylim=boundary_limits(billiard.fundamental_boundary;grd=max(1000,round(Int,maximum(ks)*L*b/(2*pi))))
    else
        xlim,ylim=boundary_limits(billiard.full_boundary;grd=max(1000,round(Int,maximum(ks)*L*b/(2*pi))))
    end
    n_rows=ceil(Int,length(ks)/max_cols)
    f = Figure(resolution=(3*width_ax*max_cols,1.5*height_ax*n_rows),size=(3*width_ax*max_cols,1.5*height_ax*n_rows))
    row=1
    col=1
    @showprogress desc="Plotting wavefunctions and husimi..." for j in eachindex(ks)
        title= isempty(custom_label) ? "$(ks[j])" : custom_label[j]
        local ax=Axis(f[row,col][1,1],title=title,aspect=DataAspect(),width=width_ax,height=height_ax)
        local ax_h=Axis(f[row,col][1,2],width=width_ax,height=height_ax)
        hm=heatmap!(ax,x_grid,y_grid,Psi2ds[j],colormap=:balance,colorrange=(-1,1))
        plot_boundary!(ax,billiard,fundamental_domain=fundamental,plot_normal=false)
        if !isempty(use_projection_grid[1]) && !isempty(use_projection_grid[2])
            projection_grid=classical_phase_space_matrix(use_projection_grid[1],use_projection_grid[2],qs_list[j],ps_list[j])
            H_bg,chaotic_mask=husimi_with_chaotic_background(Hs_list[j],projection_grid)
            heatmap!(ax_h,qs_list[j],ps_list[j],H_bg; colormap=Reverse(:gist_heat), colorrange=(0.0, maximum(H_bg)))
            heatmap!(ax_h,qs_list[j],ps_list[j],chaotic_mask;colormap=cgrad([:white, :black]),alpha=0.05,colorrange=(0,1))
        else
            heatmap!(ax_h,qs_list[j],ps_list[j],Hs_list[j];colormap=Reverse(:gist_heat))
        end
        xlims!(ax,xlim)
        ylims!(ax,ylim)
        col+=1
        if col>max_cols
            row+=1
            col=1
        end
    end
    return f
end

"""
    plot_wavefunctions_BATCH(ks::Vector, Psi2ds::Vector, x_grid::Vector, y_grid::Vector, billiard::Bi; b::Float64=5.0, width_ax::Integer=500, height_ax::Integer=500, max_cols::Integer=6) where {Bi<:AbsBilliard}

Plots the wavefunctions into a grid (only the fundamental boundary) together with the respective husimi function matrices on the provided grids. The x_grid and y_grid is supplied from the wavefunction_multi or a similar function, and the ps and qs grids mudt also be supplied for plotting the Husimi functions. This version also accepts the us boundary functions and the corresponding arclength evaluation point (us_all -> Vector{Vector{T}} and s_vals_all -> Vector{Vector{T}}) that this function was evaluated on.

# Arguments
- `ks`: Vector of eigenvalues.
- `Psi2ds::Vector{Matrix}`: Vector of 2D wavefunction matrices.
- `x_grid::Vector{<:Real}`: Vector of x-coordinates for the grid.
- `y_grid::Vector{<:Real}`: Vector of y-coordinates for the grid.
- `Hs_list::Vector{Matrix}`: Vector of 2D husimi function matrices.
- `ps_list::Vector{Vector}`: Vector of ps grids for the husimi matrices.
- `qs_list::Vector{Vector}`: Vector of qs grids for the husimi matrices.
- `billiard<:AbsBilliard`: The billiard geometry.
- `us_all::Vector{Vector{T}}`: Vector of us boundary functions.
- `s_vals_all::Vector{Vector{T}}`: Vector of arclength evaluation points.
- `b::Float64=5.0`: The point scaling factor.
- `width_ax::Integer=500`: The size of each axis in the grid layout.
- `height_ax::Integer=500`: The size of each axis in the grid layout.
- `max_cols::Integer=6`: The maximum number of columns in the grid layout.
- `fundamental::Bool=true`: If plotting just the desymmetrized part.
- `custom_label::Vector{String}`: The labels to be plotted for each Axis in the Figure. ! Needs to be the same length as ks, as it should be unique to each k in ks !.
- `use_projection_grid::Tuple{Vector,Vector}=([],[])`: A tuple containing the classical s and p values of the chaotic trajectory (in that order). These are used to construct the chaotic mask overlay so we can better observe the overlaps.

 # Returns
- `f::Figure`: A Figure object containing the grid of wavefunctions.
"""
function plot_wavefunctions_with_husimi_BATCH(ks::Vector, Psi2ds::Vector, x_grid::Vector, y_grid::Vector, Hs_list::Vector, ps_list::Vector, qs_list::Vector, billiard::Bi, us_all::Vector, s_vals_all::Vector; b::Float64=5.0, width_ax::Integer=300, height_ax::Integer=300, max_cols::Integer=6, fundamental=true, custom_label::Vector{String}=String[], use_projection_grid::Tuple{Vector,Vector}=([],[])) where {Bi<:AbsBilliard}
    for i in eachindex(Psi2ds)
        ψ=Psi2ds[i]
        amax=maximum(abs,ψ)
        if amax>0
            @. ψ=real(ψ)/amax
        else
            @. ψ=real(ψ)
        end
    end
    L=billiard.length
    if fundamental
        xlim,ylim=boundary_limits(billiard.fundamental_boundary;grd=max(1000,round(Int,maximum(ks)*L*b/(2*pi))))
    else
        xlim,ylim=boundary_limits(billiard.full_boundary;grd=max(1000,round(Int,maximum(ks)*L*b/(2*pi))))
    end
    L_corners=0.0
    res=Dict{Float64, Bool}()  # Dictionary to store length and type (true for real, false for virtual)
    res[L_corners]=true # we should start at the real curve anyway
    for crv in billiard.full_boundary
        if crv isa AbsRealCurve
            L_corners+=crv.length
            res[L_corners]=true  # Add length with true (real curve)
        elseif crv isa AbsVirtualCurve
            L_corners+=crv.length
            res[L_corners]=false  # Add length with false (virtual curve)
        end
    end
    n_rows=ceil(Int,length(ks)/max_cols)
    f=Figure(resolution=(3*width_ax*max_cols,2*height_ax*n_rows),size=(3*width_ax*max_cols,2*height_ax*n_rows))
    row=1
    col=1
    @showprogress desc="Plotting wavefunctions and husimi..." for j in eachindex(ks)
        title= isempty(custom_label) ? "$(ks[j])" : custom_label[j]
        local ax_wave=Axis(f[row, col][1, 1],title=title,aspect=DataAspect(),width=width_ax,height=height_ax)
        hm_wave=heatmap!(ax_wave,x_grid,y_grid,Psi2ds[j],colormap=:balance,colorrange=(-1,1))
        plot_boundary!(ax_wave,billiard,fundamental_domain=fundamental,plot_normal=false)
        xlims!(ax_wave,xlim)
        ylims!(ax_wave,ylim)
        local ax_h=Axis(f[row, col][1, 2],width=width_ax,height=height_ax)
        if !isempty(use_projection_grid[1]) && !isempty(use_projection_grid[2])
            projection_grid=classical_phase_space_matrix(use_projection_grid[1],use_projection_grid[2],qs_list[j],ps_list[j])
            H_bg,chaotic_mask=husimi_with_chaotic_background(Hs_list[j],projection_grid)
            heatmap!(ax_h,qs_list[j],ps_list[j],H_bg; colormap=Reverse(:gist_heat), colorrange=(0.0, maximum(H_bg)))
            heatmap!(ax_h,qs_list[j],ps_list[j],chaotic_mask;colormap=cgrad([:white, :black]),alpha=0.05,colorrange=(0,1))
        else
            heatmap!(ax_h,qs_list[j],ps_list[j],Hs_list[j];colormap=Reverse(:gist_heat))
        end
        local ax_boundary = Axis(f[row, col][2, 1:2],xlabel="s",ylabel="u(s)",width=2*width_ax,height=height_ax/2)
        lines!(ax_boundary,s_vals_all[j],us_all[j],label="u(s)",linewidth=2)
        for (length, is_real) in res
            vlines!(ax_boundary,[length],color=(is_real ? :blue : :red),linestyle=(is_real ? :solid : :dash))
        end
        # Move to the next column
        col+=1
        if col>max_cols
            row+=1 
            col=1
        end
    end
    return f
end

"""
    plot_wavefunctions_with_husimi(ks::Vector, Psi2ds::Vector, x_grid::Vector, y_grid::Vector, Hs_list::Vector, ps_list::Vector, qs_list::Vector, billiard::Bi; N=100, kwargs...) where {Bi<:AbsBilliard}

Plots the wavefunctions along with their corresponding Husimi distributions. Automatically 
splits large datasets into batches of size `N`.

# Arguments
- `ks::Vector`: Vector of eigenvalues.
- `Psi2ds::Vector{Matrix}`: Wavefunction matrices.
- `x_grid::Vector{<:Real}`: X-coordinates for the wavefunction grid.
- `y_grid::Vector{<:Real}`: Y-coordinates for the wavefunction grid.
- `Hs_list::Vector{Matrix}`: Husimi function matrices associated with each wavefunction.
- `ps_list::Vector{Vector{<:Real}}`: Momentum-like coordinate grids for the Husimi functions.
- `qs_list::Vector{Vector{<:Real}}`: Position-like coordinate grids for the Husimi functions.
- `billiard::Bi<:AbsBilliard`: The billiard geometry.
- `N::Integer=100`: Number of items per batch.
- `kwargs...`: Additional keyword arguments passed to the underlying plotting function (N axes per Figure and custom label, also chaotic PS overlays, check _BATCH function)

# Returns
- `figures::Vector{Figure}`: A vector of `Figure` objects with wavefunction and Husimi plots, one per batch.
"""
function plot_wavefunctions_with_husimi(ks::Vector, Psi2ds::Vector, x_grid::Vector, y_grid::Vector, Hs_list::Vector, ps_list::Vector, qs_list::Vector, billiard::Bi; N=100, kwargs...) where {Bi<:AbsBilliard}
    batch_wrapper(plot_wavefunctions_with_husimi_BATCH,ks,Psi2ds,x_grid,y_grid,Hs_list,ps_list,qs_list,billiard;N=N,kwargs...)
end

"""
    plot_wavefunctions_with_husimi(ks::Vector, Psi2ds::Vector, x_grid::Vector, y_grid::Vector, Hs_list::Vector, ps_list::Vector, qs_list::Vector, billiard::Bi, us_all::Vector, s_vals_all::Vector; N=100, kwargs...) where {Bi<:AbsBilliard}

Plots the wavefunctions along with their Husimi distributions and boundary functions `us_all` 
evaluated at `s_vals_all`. This function also handles a large number of wavefunctions by batching 
the data into sets of size `N`.

# Arguments
- `ks::Vector`: Vector of eigenvalues.
- `Psi2ds::Vector{Matrix}`: Wavefunction matrices.
- `x_grid::Vector{<:Real}`: X-coordinates for the wavefunction grid.
- `y_grid::Vector{<:Real}`: Y-coordinates for the wavefunction grid.
- `Hs_list::Vector{Matrix}`: Husimi function matrices.
- `ps_list::Vector{Vector{<:Real}}`: Momentum-like coordinates for Husimi functions.
- `qs_list::Vector{Vector{<:Real}}`: Position-like coordinates for Husimi functions.
- `billiard::Bi<:AbsBilliard`: The billiard geometry.
- `us_all::Vector{Vector{T}}`: Boundary functions.
- `s_vals_all::Vector{Vector{T}}`: Arclength evaluation points for the boundary functions.
- `N::Integer=100`: Number of items per batch.
- `kwargs...`: Additional keyword arguments passed to the underlying plotting function (N axes per Figure and custom label, also chaotic PS overlays, check _BATCH function)

# Returns
- `figures::Vector{Figure}`: A vector of `Figure` objects, each containing wavefunction, Husimi plots, and boundary functions, one per batch.
"""
function plot_wavefunctions_with_husimi(ks::Vector, Psi2ds::Vector, x_grid::Vector, y_grid::Vector, Hs_list::Vector, ps_list::Vector, qs_list::Vector, billiard::Bi, us_all::Vector, s_vals_all::Vector; N=100, kwargs...) where {Bi<:AbsBilliard}
    batch_wrapper(plot_wavefunctions_with_husimi_BATCH,ks,Psi2ds,x_grid,y_grid,Hs_list,ps_list,qs_list,billiard,us_all,s_vals_all;N=N,kwargs...)
end


##############################################################################
#### TOOLS FOR CHECKING THE POWER SPECTRUM FOR CircleSegment part of u(s) ####
##############################################################################

"""
    compute_cm_circular_segment(u, s_vals, m, billiard)

Compute the angular momentum coefficient `cₘ` of the boundary function `u(s)` restricted to the CircleSegment of the billiard using trapezoidal integration.

# Arguments
- `u::Vector{T}`: Normal derivative of the wavefunction.
- `s_vals::Vector{T}`: Arclength positions of boundary points.
- `m::Integer`: Angular momentum index.
- `billiard::AbsBilliard`: The billiard geometry.

# Returns
- `Complex{T}`: Complex coefficient `cₘ`.
"""
function compute_cm_circular_segment(u::Vector{T},s_vals::Vector{T},m::Ti,billiard::Bi)::Complex{T} where {T<:Real,Ti<:Integer,Bi<:AbsBilliard}
    s_start,s_end=0.0,0.0
    total_length=0.0
    R=0.0
    found=false
    for seg in billiard.full_boundary
        seg_length=seg.length
        if seg isa CircleSegment
            s_start=total_length
            s_end=total_length+seg_length
            R=seg.radius
            found=true
            break
        end
        total_length+=seg_length
    end
    if !found
        error("No CircleSegment found in billiard boundary.")
    end
    filtered_idx=findall(s->s>=s_start && s<=s_end,s_vals) # filter u and s_vals on the CircleSegment
    us=u[filtered_idx]
    ss=s_vals[filtered_idx]
    N=length(us) #
    if N<2 # sanity check
        @warn "Not enough points on the CircleSegment to compute integral."
        return 0.0+0.0im
    end
    weights=zeros(T,N) # trapezoidal weights
    weights[1]=(ss[2]-ss[1])/2
    for i in 2:N-1
        weights[i]=(ss[i+1]-ss[i-1])/2
    end
    weights[end]=(ss[end]-ss[end-1])/2
    return sum(us[i]*exp(im*m*π*ss[i]/R)*weights[i] for i in 1:N)
end

"""
    compute_cm_circular_segment(u, s_vals, ms, billiard)

Compute multiple angular momentum coefficients `cₘ` for each `m` in `ms` from the boundary function `u(s)` on the CircleSegment.

# Arguments
- `u::Vector{T}`: Normal derivative of the wavefunction.
- `s_vals::Vector{T}`: Arclength positions of boundary points.
- `ms::Vector{Integer}`: Angular momentum indexes.
- `billiard::AbsBilliard`: The billiard geometry.

# Returns
- `Vector{Complex{T}}`: Vector of angular momentum coefficients.
"""
function compute_cm_circular_segment(u::Vector{T},s_vals::Vector{T},ms::Vector{Ti},billiard::Bi)::Vector{Complex{T}} where {T<:Real,Ti<:Integer,Bi<:AbsBilliard}
    s_start,s_end=0.0,0.0
    total_length=0.0
    R=0.0
    found=false
    for seg in billiard.full_boundary
        seg_length=seg.length
        if seg isa CircleSegment
            s_start=total_length
            s_end=total_length+seg_length
            R=seg.radius
            found=true
            break
        end
        total_length+=seg_length
    end
    if !found
        error("No CircleSegment found in billiard boundary.")
    end
    filtered_idx=findall(s->s>=s_start && s<=s_end,s_vals) # filter u and s_vals on the CircleSegment
    us=u[filtered_idx]
    ss=s_vals[filtered_idx]
    N=length(us) #
    if N<2 # sanity check
        @warn "Not enough points on the CircleSegment to compute integral."
        return 0.0+0.0im
    end
    weights=zeros(T,N) # trapezoidal weights
    weights[1]=(ss[2]-ss[1])/2
    for i in 2:N-1
        weights[i]=(ss[i+1]-ss[i-1])/2
    end
    weights[end]=(ss[end]-ss[end-1])/2
    cms=Vector{Complex{T}}(undef,length(ms))
    Threads.@threads for k in eachindex(ms)
        cms[k]=sum(us[i]*exp(im*ms[k]*π*ss[i]/R)*weights[i] for i in 1:N)
    end
    return cms
end

"""
    fraction_on_segments(u, s_vals, billiard; which_segments = :all)

Compute the L² norm fraction of the boundary function `u(s)` over selected segments of the billiard.

This is useful for distinguishing whether the boundary function is concentrated on specific segments
(e.g., the CircleSegment) or spread across others.

# Arguments
- `u::Vector{T}`: The boundary function.
- `s_vals::Vector{T}`: Arclength coordinates along the boundary.
- `billiard::AbsBilliard`: The billiard geometry.
- `which_segments::Union{Symbol, Vector{Int}} = :all`: Which segments to include:
    - `:circle` → only the CircleSegment,
    - `:all` → the entire boundary,
    - `Vector{Int}` → specify by segment indices in `billiard.full_boundary`.

# Returns
- `T`: Fraction of the total L² norm on the selected segments.
"""
function fraction_on_segments(u::Vector{T},s_vals::Vector{T},billiard::AbsBilliard;which_segments::Union{Symbol,Vector{Ti}}=:all)::T where {T<:Real,Ti<:Integer}
    @assert length(u)==length(s_vals) "u and s_vals must be the same length"
    N=length(u)
    # Precompute segment arclength intervals
    segment_bounds=Vector{Tuple{T,T}}()
    circle_idx=nothing
    total_length=zero(T)
    for (i,seg) in enumerate(billiard.full_boundary)
        L=seg.length
        push!(segment_bounds,(total_length,total_length+L))
        if seg isa CircleSegment
            circle_idx=i
        end
        total_length+=L
    end
    selected_bounds= if which_segments==:all
        segment_bounds
    elseif which_segments==:circle
        @assert circle_idx!==nothing "No CircleSegment found."
        [segment_bounds[circle_idx]]
    elseif isa(which_segments,Vector{Int})
        [segment_bounds[i] for i in which_segments]
    else
        error("Invalid `which_segments` value. Use :all, :circle, or a Vector of segment indices.")
    end
    # Trapezoidal weights
    weights = similar(u)
    @inbounds begin
        weights[1]=(s_vals[2]-s_vals[1])/2
        for i in 2:N-1
            weights[i]=(s_vals[i+1]-s_vals[i-1])/2
        end
        weights[end]=(s_vals[end]-s_vals[end-1])/2
    end
    total_per_thread=zeros(T,Threads.nthreads())
    selected_per_thread=zeros(T,Threads.nthreads())
    Threads.@threads for i in 1:N
        tid=Threads.threadid()
        val=abs2(u[i])*weights[i]
        total_per_thread[tid]+=val
        for (s_start,s_end) in selected_bounds
            if s_vals[i]≥s_start && s_vals[i]≤s_end
                selected_per_thread[tid]+=val
                break
            end
        end
    end
    total_norm=sum(total_per_thread)
    selected_norm=sum(selected_per_thread)
    return selected_norm/total_norm
end

"""
    compute_cm_circular_segment_and_fraction(u::Vector{T},s_vals::Vector{T},ms::Vector{Ti},billiard::Bi)::Tuple{Vector{Complex{T}},T} where {T<:Real,Ti<:Integer,Bi<:AbsBilliard}

Computes the cm coefficients of the angular momentum basis expansion and also the fraction of the boundary function on the CircularSegment using the Trapezoidal rule on the L^2 norm.

# Arguments
- `u::Vector{T}`: The boundary function.
- `s_vals::Vector{T}`: The arclengths of the entire billiard.
- `ms::Vector{Integer}`: Angular momentum indexes.
- `billiard<:AbsBilliard`: The billiard geometry that contains information on all the curve segments.
- `which_segments::Union{Symbol, Vector{Int}} = :all`: Which segments to take in the the calculation of the fraction of the boundary norm. The default value is a placeholder and the Vector{Int} should be used for the other relevant sections where we want to check the boundary function L2 norm.

For example in the mushroom billiard we would choose `which_segments = [1, 2, 6]` since the other segments are either the `CircleSegment` or `LineSegment`s that have overlap with the circle eigenfunction (the connectors of the stem with the cap are such cases with `idxs = [3, 5]`)

# Returns
- `cms::Vector{Complex{T}}`: The cm coefficient for each m in ms.
- `frac::T`: The fraction of the boundary function as per function description.
"""
function compute_cm_circular_segment_and_fraction(u::Vector{T},s_vals::Vector{T},ms::Vector{Ti},billiard::Bi;which_segments::Union{Symbol,Vector{Ti}}=:all)::Tuple{Vector{Complex{T}},T} where {T<:Real,Ti<:Integer,Bi<:AbsBilliard}
    cms=compute_cm_circular_segment(u,s_vals,ms,billiard)
    frac=fraction_on_segments(u,s_vals,billiard;which_segments=which_segments)
    return cms,frac
end

"""
    compute_P_m(cm::Complex{T})::T where {T<:Real}

Returns the power `|cₘ|²` from a single angular momentum coefficient.

# Arguments
- `cm::Complex{T}`: The angular momentum coefficient.

# Returns
- `T`: The associated `|cₘ|²` value.
"""
function compute_P_m(cm::Complex{T})::T where {T<:Real}
    return abs2(cm)
end

"""
    compute_P_m(cms::Vector{Complex{T}})::Vector{T} where {T<:Real}

Returns the power spectrum `|cₘ|²` for a vector of coefficients.

# Arguments
- `cms::Vector{Complex{T}}`: The angular momentum coefficients.

# Returns
- `Vector{T}`: The associated vector of NORMALIZED `|cₘ|²` values.
"""
function compute_P_m(cms::Vector{Complex{T}}) where {T<:Real}
    S=sum(abs2.(cms))
    return [abs2(cm)/S for cm in cms]
end

"""
    Shannon_entropy_cms(Pms)

Computes the Shannon entropy from a normalized angular momentum power distribution.

# Arguements
- `Pms::Vector{T}`: Normalized Power spectrum for a boundary function. Sometimes it is useful to take the log of this value since delta-like functions have negative Shannon entropy value

# Returns
- `T`: Shannon entropy value `S = -∑ pᵢ log(pᵢ)`
"""
function Shannon_entropy_cms(Pms::Vector{T}) where {T<:Real}
    return -sum(p_i>0.0 ? p_i*log(p_i) : 0.0 for p_i in Pms)
end

"""
    is_regular(Pms::Vector{T},frac::T;threshold::Float64=1.0,frac_threshold=0.1) where {T<:Real}

Determine if a state is "regular-like" based on low Shannon entropy in angular momentum space.

Returns `true` if `S < threshold`, suggesting localization around a conserved quantity.

# Arguments
- `Pms::Vector{T}`: Normalized power distribution.
- `frac::T`: THe threshold for the L^2 norm of the boundary function on non-circular segments that were chosen with `which_segments` in the bottom level functions.
- `threshold=1.0`: Entropy cutoff (default = 1.0 by obseving the behaviour of the boundary function on the `CircleSegment`). This also is useful since when we take the log of ot it is negative if below 1.0 and separration is clear.
- `frac_threshold=0.1`: The default threshold for the L2 norm of the boudnary function on the boundary chosen with `which_segments` in the bottom level functions. Benchmarking shows that the frac value is usually well below 10^-2, in most cases below < 10^-7. This value is analogous addition measure of the overlap between the wavefunction with the circle eigefunction (it's angular momentum component exp(i*pi*ϕ)).

# Returns
- `Bool`: Whether the state is regular.
"""
function is_regular(Pms::Vector{T},frac::T;threshold=1.0,frac_threshold=0.1) where {T<:Real}
    Shannon_entropy_cms(Pms)<threshold && frac<frac_threshold ? true : false
end

# HELPER FUNCTION SINCE THERE ARE ISOLATED CASES WHERE A REGULAR FUNCTION GRAZING THE 
function is_mushroom_MUPO(frac::T;frac_threshold=0.1) where {T<:Real}
    frac<frac_threshold ? true : false
end

###################################################################################
################# WAVEFUCNTION CONSTRUTION FOR BASIS TYPE METHODS #################
###################################################################################

"""
    compute_psi(vec::Vector, k::T, billiard::Bi, basis::Ba, x_grid, y_grid; inside_only=true, memory_limit = 10.0e9) where {Bi<:AbsBilliard, Ba<:AbsBasis, T<:Real}

Computs the wavefunction as a `Matrix` on a grid formed by the vectors `x_grid` and `y_grid`. This is a lower level function for wrappers that require the construction of a wavefunction from a vector of linear expansion coefficients and being constructed on a common grid.

# Arguments
- `vec::Vector{T}`: A vector of coefficients representing the linear expansion coefficients of the wavefunction.
- `k::T`: The k-eigenvalue at which the wavefunction is evaluated.
- `billiard<:AbsBilliard`: An instance of the abstract billiard type representing the physical billiard.
- `basis<:AbsBasis`: An instance of the abstract basis type representing the linear expansion basis.
- `x_grid::Vector{T}`: A vector of x-coordinates at which the wavefunction should be evaluated.
- `y_grid::Vector{T}`: A vector of y-coordinates at which the wavefunction should be evaluated.
# Keyword arguments
- `inside_only::Bool=true`: If true, only points inside the billiard are considered for evaluation.
- `memory_limit=10.0e9`: A limit on the memory usage for the computation in bytes. If the memory usage exceeds this limit, multithreading is disabled for the matrix construction.

# Returns
- `Psi::Matrix{T}`: A matrix representing the wavefunction evaluated on the grid formed by the vectors x_grid and y_grid.
"""
function compute_psi(vec::Vector,k::T,billiard::Bi,basis::Ba,x_grid::Vector,y_grid::Vector;inside_only=true,memory_limit=10.0e9) where {Bi<:AbsBilliard, Ba<:AbsBasis,T<:Real}
    eps=set_precision(vec[1])
    sz=length(x_grid)*length(y_grid)
    pts=collect(SVector(x,y) for y in y_grid for x in x_grid)
    if inside_only
        pts_mask=points_in_billiard_polygon(pts,billiard,round(Int,sqrt(sz));fundamental_domain=true)
        pts=pts[pts_mask]
    end
    n_pts=length(pts)
    type=eltype(vec)
    memory=sizeof(type)*basis.dim*n_pts #estimate max memory needed for the matrices
    Psi=zeros(type,sz)
    if memory<memory_limit
        B=basis_matrix(basis,k,pts)
        Psi_pts=B*vec
        if inside_only
            Psi[pts_mask].=Psi_pts
        else
            Psi.=Psi_pts
        end
    else
        println("Warning: memory limit of $(Base.format_bytes(memory_limit)) exceded $(Base.format_bytes(memory)).")
        if inside_only
            for i in eachindex(vec)
                if abs(vec[i])>eps 
                    Psi[pts_mask].+=vec[i].*basis_fun(basis,i,k,pts)
                end
            end
        else
            for i in eachindex(vec)
                if abs(vec[i])>eps 
                    Psi.+=vec[i].*basis_fun(basis,i,k,pts)
                end
            end
        end
    end
    return Psi
end

"""
    compute_psi(state::S,x_grid::Vector{T},y_grid::Vector{T};inside_only=true,memory_limit=10.0e9) where {S<:AbsState,T<:Real}

Constructs the wavefunction as a `Matrix` from an `Eigenstate` struct on a grid of vectors `x_grid` and `y_grid`.

# Arguments
- `state::S`: An `Eigenstate` struct with a `vec` field representing the wavefunction, a `k_basis` field representing the wavefunction basis, a `basis` field representing the basis set, a `billiard` field representing the billiard.
- `x_grid::Vector{T}`: A vector of `x` coordinates on which to evaluate the wavefunction.
- `y_grid::Vector{T}`: A vector of `y` coordinates on which to evaluate the wavefunction.
- `inside_only::Bool` (optional, default `true`): If `true`, only evaluate the wavefunction inside the billiard.
- `memory_limit::Real` (optional, default `10.0e9`): The maximum memory limit in bytes to use for constructing the wavefunction. If the memory required exceeds this multithreading is disabled.

# Returns
- `Psi::Matrix`: A `Matrix` representing the wavefunction evaluated on the grid.
"""
function compute_psi(state::S,x_grid::Vector{T},y_grid::Vector{T};inside_only=true,memory_limit=10.0e9) where {S<:AbsState,T<:Real}
    compute_psi(state.vec,state.k,state.billiard,state.basis,x_grid,y_grid;inside_only,memory_limit)
end

"""
    wavefunction(vec::Vector, k::T, billiard::Bi, basis::Ba; b=5.0, inside_only=true, fundamental_domain = true, memory_limit = 10.0e9) where {Bi<:AbsBilliard, Ba<:AbsBasis, T<:Real}     

Computes the wavefunction matrix and the x and y grids for heatmap plotting. It is contructed from the vec=X[i] of `StateData` and not directly from `StateData`.

# Arguments
- `vec::Vector{<:Real}`: The vector of coefficients of the basis expansion of the wavefunction. It's length determines the resizeing of the `basis`.
- `k<:Real`: The wavenumber for that vec = X[i].
- `billiard<:AbsBilliard`: The billiard geometry.
- `basis<:AbsBasis`: The basis used for constructing the wavefunction from `vec`. Must be the same as the one used for constructing `vec`.
- `b`: The point scalling factor. Default is 5.0.
- `inside_only::Bool`: If true, only the points inside the billiard are considered. Default is true.
- `fundamental_domain::Bool`: If true, the wavefunction information is only constructed in the fundamental domain. Default is true.
- `memory_limit`: The maximum amount of memory (in bytes) for constructing the wavefunction with julia broadcasting operations and the use of the `basis_matrix`. Otherwise we use the `basis_fun` directly. Default is 10.0e9.

# Returns
- `Psi2ds::Vector{Matrix}`: A vector of `Matrix` containing the wavefunction for each k in ks.
- `x_grids::Vector{Vector}`: A vector of `Vector` containing the x grid for each k in ks.
- `y_grids::Vector{Vector}`: A vector of `Vector` containing the y grid for each k in ks.
"""
function wavefunction(vec::Vector,k::T,billiard::Bi,basis::Ba;b=5.0,inside_only=true,fundamental_domain=true,memory_limit=10.0e9) where {Bi<:AbsBilliard, Ba<:AbsBasis,T<:Real}     
    dim=length(vec)
    dim=rescale_rpw_dimension(basis,dim)
    basis=resize_basis(basis,billiard,dim,k)
    symmetries=basis.symmetries
    type=eltype(vec)
    L=billiard.length
    xlim,ylim=boundary_limits(billiard.fundamental_boundary;grd=max(1000,round(Int,k*L*b/(2*pi))))
    dx=xlim[2]-xlim[1]
    dy=ylim[2]-ylim[1]
    nx=max(round(Int,k*dx*b/(2*pi)),512)
    ny=max(round(Int,k*dy*b/(2*pi)),512)
    x_grid::Vector{type}=collect(type,range(xlim...,nx))
    y_grid::Vector{type}=collect(type,range(ylim...,ny))
    Psi::Vector{type}=compute_psi(vec,k,billiard,basis,x_grid,y_grid; inside_only=inside_only, memory_limit=memory_limit) 
    Psi2d::Array{type,2}=reshape(Psi,(nx,ny))
    return Psi2d,x_grid,y_grid
end

"""
    wavefunction(state::S; b=5.0, inside_only=true, fundamental_domain = true, memory_limit = 10.0e9) where {S<:AbsState}

Constructs the wavefunction from a given state object (like Eigenstate).

# Arguments
- `state::S`: An instance of the abstract state type representing the state from which the wavefunction should be constructed.
- `b::Float64=5.0`: A scaling factor for the billiard size.
- `inside_only::Bool=true`: If true, only points inside the billiard are considered for evaluation.
- `fundamental_domain::Bool=true`: If true, the wavefunction is computed on the fundamental domain of the billiard.
- `memory_limit=10.0e9`: A limit on the memory usage for the computation in bytes. If the memory usage exceeds this limit, multithreading is disabled for the construction.

# Returns
- `Psi2d::Array{T,2}`: A 2D array representing the wavefunction evaluated on the grid formed by the vectors x_grid and y_grid.
- `x_grid::Vector{T}`: A Vector of x values where the matrix was evaluated.
- `y_grid::Vector{T}`: A Vector of y values where the matrix was evaluated.
"""
function wavefunction(state::S;b=5.0,inside_only=true,fundamental_domain=true,memory_limit=10.0e9) where {S<:AbsState}    
    wavefunction(state.vec,state.k,state.billiard,state.basis;b=b,inside_only=inside_only,fundamental_domain=fundamental_domain,memory_limit=memory_limit)
end

"""
    wavefunctions(X::Vector, ks::Vector, billiard::Bi, basis::Ba; b=5.0, inside_only=true, fundamental_domain = true, memory_limit = 10.0e9) where {Bi<:AbsBilliard, Ba<:AbsBasis, T<:Real}

High level wrapper for moer efficiently computing wavefunction matrices and the grids for plotting.

# Arguments
- `X::Vector`: A vector of coefficients of the basis expansion of the wavefunction for each k in ks.
- `ks::Vector`: A vector of wavenumbers for which to compute the wavefunction.
- `billiard::Bi`: An object representing the billiard.
- `basis::Ba`: An object representing the basis (rpw, cafb...).
- `b`: The point scalling factor. Default is 5.0.
- `inside_only::Bool`: If true, only the points inside the billiard are considered. Default is true.
- `fundamental_domain::Bool`: If true, the wavefunction is only constructed in the fundamental domain. Default is true.
- `memory_limit`: The maximum amount of memory (in bytes) for constructing the wavefunction with julia broadcasting operations and the use of the `basis_matrix`. Otherwise we use the `basis_fun` directly. Default is 10.0e9.

# Returns
- `vec_Psi::Vector{Matrix}`: A vector of `Matrix` containing the wavefunction for each k in ks.
- `vec_xs::Vector{Vector}`: A vector of `Vector` containing the x grid for each k in ks.
- `vec_ys::Vector{Vector}`: A vector of `Vector` containing the y grid for each k in ks.
"""
function wavefunctions(X::Vector,ks::Vector,billiard::Bi,basis::Ba;b=5.0,inside_only=true,fundamental_domain=true,memory_limit=10.0e9) where {Bi<:AbsBilliard, Ba<:AbsBasis}
    vec_Psi=Vector{Matrix}(undef,length(ks))
    vec_xs=Vector{Vector}(undef,length(ks))
    vec_ys=Vector{Vector}(undef,length(ks))
    p = Progress(length(ks),1)
    Threads.@threads for i in eachindex(ks) 
        vec=X[i]
        k=ks[i]
        Psi2d,x_grid,y_grid=wavefunction(vec,k,billiard,basis;b=b,inside_only=inside_only,fundamental_domain=fundamental_domain,memory_limit=memory_limit)
        vec_Psi[i]=Psi2d
        vec_xs[i]=x_grid
        vec_ys[i]=y_grid
        next!(p)
    end
    return vec_Psi,vec_xs,vec_ys
end

"""
    wavefunctions(state_data::StateData, billiard::Bi, basis::Ba; b=5.0, inside_only=true, fundamental_domain = true, memory_limit = 10.0e9) :: Tuple{Vector, Vector{Matrix}, Vector{Vector}, Vector{Vector}} where {Bi<:AbsBilliard, Ba<:AbsBasis}

High level wrapper for constructing the wavefunctions as a a `Tuple` of `Vector`s : `Tuple (ks::Vector, Psi2ds::Vector{Matrix}, x_grid::Vector{Vector}, y_grid::Vector{Vector})`.

# Arguments
- `state_data::StateData`: Object containing the wavenumbers, tensions and the coefficients of the wavefunction expansion as a vector of vectors for each k in ks.
- `billiard::Bi`: An object representing the billiard.
- `basis::Ba`: An object representing the basis (rpw, cafb...).
- `b`: The point scalling factor. Default is 5.0.
- `inside_only::Bool`: If true, only the points inside the billiard are considered. Default is true.
- `fundamental_domain::Bool`: If true, the wavefunction information is only constructed in the fundamental domain. Default is true.
- `memory_limit`: The maximum amount of memory (in bytes) for constructing the wavefunction with julia broadcasting operations and the use of the `basis_matrix`. Otherwise we use the `basis_fun` directly. Default is 10.0e9.

# Returns
- `ks::Vector{Float64}`: A vector of wavenumbers.
- `Psi2ds::Vector{Matrix}`: A vector of `Matrix` containing the wavefunction for each k in ks.
- `x_grids::Vector{Vector}`: A vector of `Vector` containing the x grid for each k in ks.
- `y_grids::Vector{Vector}`: A vector of `Vector` containing the y grid for each k in ks.
"""
function wavefunctions(state_data::StateData,billiard::Bi,basis::Ba;b=5.0,inside_only=true,fundamental_domain=true,memory_limit=10.0e9) where {Bi<:AbsBilliard, Ba<:AbsBasis}
    ks=state_data.ks
    tens=state_data.tens
    X=state_data.X
    Psi2ds=Vector{Matrix{eltype(ks)}}(undef,length(ks))
    x_grids=Vector{Vector{eltype(ks)}}(undef,length(ks))
    y_grids=Vector{Vector{eltype(ks)}}(undef,length(ks))
    for i in eachindex(ks) 
        vec=X[i] # vector of vectors
        dim=length(vec)
        dim=rescale_rpw_dimension(basis, dim)
        new_basis=resize_basis(basis, billiard, dim, ks[i])
        state=Eigenstate(ks[i], vec, tens[i], new_basis, billiard)
        Psi2d,x_grid,y_grid=wavefunction(state;b=b,inside_only=inside_only,fundamental_domain=fundamental_domain,memory_limit=memory_limit)
        Psi2ds[i]=Psi2d
        x_grids[i]=x_grid
        y_grids[i]=y_grid
    end
    return ks,Psi2ds,x_grids,y_grids
end

"""
    wavefunction(state::BasisState; xlim =(-2.0,2.0), ylim=(-2.0,2.0), b=5.0) 

Construct the wavefunction for a given basis function defined from a `BasisState` object. It is useful for visualizing the varius basis functions in the chosen basis.

# Arguments
- `state::BasisState`: An object representing the basis function.
- `xlim::Tuple{Float64,Float64}`: The range of x values for the wavefunction. Default is `(-2.0, 2.0)`.
- `ylim::Tuple{Float64,Float64}`: The range of y values for the wavefunction. Default is `(-2.0, 2.0)`.
- `b::Float64`: The point scalling factor. Default is 5.0.

# Returns
- `Psi2d::Array{Float64,2}`: The 2D wavefunction matrix for the given basis matrix.
- `x_grid::Vector{<:Real}`: The x grid formed from the `xlim`.
- `y_grid::Vector{<:Real}`: The y grid formed from the `ylim`.
"""
function wavefunction(state::BasisState;xlim =(-2.0,2.0),ylim=(-2.0,2.0),b=5.0) 
    let k=state.k,basis=state.basis      
        type=eltype(state.vec)
        #TODO try to find a lazy way to do this
        dx=xlim[2]-xlim[1]
        dy=ylim[2]-ylim[1]
        nx=max(round(Int,k*dx*b/(2*pi)),512)
        ny=max(round(Int,k*dy*b/(2*pi)),512)
        x_grid::Vector{type}=collect(type,range(xlim...,nx))
        y_grid::Vector{type}=collect(type,range(ylim...,ny))
        pts_grid=[SVector(x,y) for y in y_grid for x in x_grid]
        Psi::Vector{type}=basis_fun(basis,state.idx,k,pts_grid) 
        Psi2d::Array{type,2}=reshape(Psi,(nx,ny))
        return Psi2d,x_grid,y_grid
    end
end

#TODO this can be optimized
"""
    compute_psi(state_bundle::S, x_grid::Vector{T}, y_grid::Vector{T}; inside_only=true, memory_limit = 10.0e9) where {S<:EigenstateBundle, T<:Real}

Computs the wavefunction Matrix on an x_grid and y_grid from an EigenstateBundle object. All the matrices in the state bundle are computed on the same grid.

# Arguments
- `state_bundle::S`: An object representing the bundle of eigenstate.
- `x_grid::Vector{<:Real}`: A vector representing the x grid.
- `y_grid::Vector{<:Real}`: A vector representing the y grid.
- `inside_only::Bool`: If true, only the points inside the billiard are considered. Default is true.
- `memory_limit`: The maximum amount of memory (in bytes) for constructing the wavefunction with julia broadcasting operations and the use of the `basis_matrix`. Otherwise we use the non-multithread implementation.

# Returns
- `Psi_bundle::Matrix{<:Real}`: A matrix containing the wavefunction for each state in the bundle on the given grid.
"""
function compute_psi(state_bundle::S,x_grid::Vector{T},y_grid::Vector{T};inside_only=true,memory_limit=10.0e9) where {S<:EigenstateBundle,T<:Real}
    let k=state_bundle.k_basis,basis=state_bundle.basis,billiard=state_bundle.billiard,X=state_bundle.X #basis is correct size
        sz=length(x_grid)*length(y_grid)
        pts=collect(SVector(x,y) for y in y_grid for x in x_grid)
        if inside_only
            pts_mask=is_inside(billiard,pts)
            pts=pts[pts_mask]
        end
        n_pts=length(pts)
        n_states=length(state_bundle.ks)
        type=eltype(state_bundle.X) #estimate max memory needed for the matrices
        memory=sizeof(type)*basis.dim*n_pts
        Psi_bundle=zeros(type,(sz,n_states))  #Vector of results
        if memory<memory_limit
            B=basis_matrix(basis,k,pts)
            Psi_pts=B*X
            Psi_bundle[pts_mask,:].=Psi_pts
        else
            println("Warning: memory limit of $(Base.format_bytes(memory_limit)) exceded $(Base.format_bytes(memory)).")
            Psi_pts=zeros(type,(n_pts,n_states))
            for i in 1:basis.dim
                bf=basis_fun(basis,i,k,pts) #vector of length n_pts
                for j in 1:n_states
                    Psi_pts[:,j].+=X[i,j].*bf
                end
            end
            if inside_only
                Psi_bundle[pts_mask,:]=Psi_pts
            else
                Psi_bundle=Psi_pts
            end
        end
        return Psi_bundle #this is a matrix 
    end
end

"""
    wavefunction(state_bundle::S; b=5.0, inside_only=true, fundamental_domain=true, memory_limit=10.0e9) where {S<:EigenstateBundle}

Construct the wavefunction matrices from an EigenstateBundle on a common grid. Useful for a smaller number of wavefunctions.

# Arguments
- `state_bundle::S`: An object representing the bundle of eigenstates.
- `b`: The point scalling factor. Default is 5.0.
- `inside_only::Bool`: If true, only the points inside the billiard are considered. Default is true.
- `fundamental_domain::Bool`: If true, the wavefunction is only constructed in the fundamental domain. Default is true.
- `memory_limit`: The maximum amount of memory (in bytes) for constructing the wavefunction with julia broadcasting operations and the use of the `basis_matrix`. Otherwise we use the non-multithreaded implementation.

# Returns
- `Psi2ds::Vector{Matrix{<:Real}}`: A vector of `Matrix` objects containing the wavefunction for each state in the bundle.
- `x_grid::Vector{<:Real}`: A vector of x grid points common for the entire bundle.
- `y_grid::Vector{<:Real}`: A vector of y grid points common for the entire bundle.
"""
function wavefunction(state_bundle::S;b=5.0,inside_only=true,fundamental_domain=true,memory_limit=10.0e9) where {S<:EigenstateBundle}
    k=state_bundle.k_basis
    billiard=state_bundle.billiard
    symmetries=state_bundle.basis.symmetries
    T=Float64
    L=billiard.length
    xlim,ylim=boundary_limits(billiard.fundamental_boundary;grd=max(1000,round(Int,k*L*b/(2*pi))))
    dx=xlim[2]-xlim[1]
    dy=ylim[2]-ylim[1]
    nx=max(round(Int,k*dx*b/(2*pi)),512)
    ny=max(round(Int,k*dy*b/(2*pi)),512)
    x_grid::Vector{T}=collect(range(xlim...,nx))
    y_grid::Vector{T}=collect(range(ylim...,ny))
    Psi_bundle::Matrix{Complex{T}}=compute_psi(state_bundle,x_grid,y_grid;inside_only=inside_only,memory_limit=memory_limit)
    # Reshape each column of Psi_bundle into a matrix
    Psi2d::Vector{Matrix{Complex{T}}}=[reshape(Psi,(nx,ny)) for Psi in eachcol(Psi_bundle)]
    return Psi2d,x_grid,y_grid
end