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
    rjn=real(ifft(a)) # inverse FFT → rjn[j] = (2/N)*∑_{m=1..n-1} (1/m) cos(2π m (j-1)/N)
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
    rjn=real(ifft(a)) # gives (1/N) * sum_m a_m exp(2πimj/N)
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

# For CFIE with holes, we compute this by looking at the component offsets, which tell us where each component's points start and end in the concatenated array. The last offset gives us the total count of points.
function boundary_matrix_size(pts::Vector{BoundaryPointsCFIE{T}}) where {T<:Real}
    offs=component_offsets(pts)
    return offs[end]-1
end

function boundary_matrix_size(pts::BoundaryPointsCFIE{T}) where {T<:Real}
    return length(pts.xy)
end

# helper function to compute the offsets for each component of the boundary, which are needed to correctly assemble the R matrix for the CFIE_kress method. The offsets indicate the starting index of each component's points in the concatenated list of all boundary points. For example, if we have 3 components with 10, 15, and 20 points respectively, the offsets would be [1, 11, 26, 46].
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

# Build global corner locations in σ ∈ [0,2π)
# Each segment junction is treated as a "corner" for grading.
# If segment j ends at arc-length s = cum[j+1], then its σ-location is:
# σ = 2π * (cum[j+1]/Ltot)
# We include σ=0 explicitly (periodic corner).
function _component_corner_locations(::Type{T},comp::Vector) where {T<:Real}
    lens,cum,Ltot=component_lengths(comp)
    corners=T[zero(T)]
    for j in 1:length(comp)-1
        push!(corners,T(two_pi)*(cum[j+1]/Ltot))
    end
    return corners
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

########################################
#### GEOMETRY CACHE FOR BIE SOLVERS ####
########################################

struct CFIEGeomCache{T<:Real}
    R::Matrix{T}
    invR::Matrix{T}
    inner::Matrix{T}
    logterm::Matrix{T}
    speed::Vector{T}
    kappa::Vector{T}
    original_ts::Vector{T} # for kress with corners for keeping track of original trapzoidal discretization for log term.
    logjac::Vector{T} # corrections to the log term due to non-uniform quadrature
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
    logjac=corner_kress ? T[2*log(abs(wd)) for wd in pts.ws_der] : zeros(T,N)
    return CFIEGeomCache(R,invR,inner,logterm,speed,kappa,original_ts,logjac)
end

