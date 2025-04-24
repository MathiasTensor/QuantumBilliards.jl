using LinearAlgebra, CoordinateTransformations, Rotations, StaticArrays

"""
MAIN REFERENCE: Jan Wiersig, Gabriel G. Carlo, Evanescent wave approach to diﬀractive phenomena in convex billiards with corners
https://arxiv.org/pdf/nlin/0212011
AUXILIARY REFERENCE: Alex Barnett, PhD Thesis
https://users.flatironinstitute.org/~ahb/thesis_html/node157.html
"""

"""
    α(i::Int,k::T)

Barnett's algebraic decay constant for epw as used in the stadium. Used in automatic generation of decay constants for an angle set.
"""
α(i::Int,k::T) where {T<:Real} = (3+i)/(2*k^(1/3))
dα_dk(i::Int,k::T) where {T<:Real} =-(3+i)/(6*k^(4/3))
    
"""
    max_i(k::Real) -> Int

Compute the maximum integer index `i` such that the evanescent decay parameter 
αᵢ = (3 + i) / (2k^(1/3)) does not exceed 3.
Further reading: https://users.flatironinstitute.org/~ahb/thesis_html/node157.html

This ensures that the evanescent plane wave parameter αᵢ remains within 
the recommended upper limit for numerical stability and efficiency.

# Arguments
- `k::Real`: The wavenumber used to define the decay rate of the evanescent plane wave.

# Returns
- `Int`: The maximum value of `i` such that αᵢ ≤ 3.
"""
max_i(k::T) where {T<:Real} = floor(Int,6*k^(1/3)-3)

"""
    max_billiard_impact(θ, pts, origin, k) → T

Compute the maximum “impact parameter” b for an evanescent plane wave
centered at `origin` with propagation direction `θ`, scaled by the wavenumber `k`.

b = k * maxₚ [ d(θ) ⋅ (p - origin) ]  

# Arguments
- `θ::T`: Propagation angle in radians (measured from +x axis).
- `pts::AbstractVector{SVector{2,T}}`: Sample points along ∂Ω (each SVector{2,T}).
- `origin::SVector{2,T}`: The corner or diffraction center in the same coordinate system.
- `k::T`: Wavenumber; used to scale the raw impact.

# Returns
- `b::T`: The scaled maximum impact parameter.

"""
function max_billiard_impact(θ::T,pts::AbstractArray{<:SVector{2,T}},origin::SVector{2,T},k::T) where {T<:Real}
    s,c=sincos(θ)
    d=SVector(s,c)
    return k*maximum(dot(d,p-origin) for p in pts)
end

"""
    sinhcosh(x::T) where {T<:Real}

Compute `sinh` and `cosh` with one `exp` calculation.

# Returns
- `(sinh(x),cosh(x))::Tuple{T,T}`

"""
function sinhcosh(x::T) where {T<:Real}
    ex=exp(x)
    ex_inv=1/ex 
    return (ex-ex_inv)/2,(ex+ex_inv)/2
end

"""
    epw(pts::AbstractArray{<:SVector{2,T}},i::Int,Ni::Int,origin::SVector{2,T},angle_range::Vector{T},k::T) where {T<:Real}

Compute the evanescent plane wave function at given points for a single origin. It uses the following literature:
https://users.flatironinstitute.org/~ahb/thesis_html/node157.html # decay factor and comments
https://arxiv.org/pdf/nlin/0212011 # basis form and comments

# Arguments
- `pts::AbstractArray`: Points in space where the function is evaluated.
- `i::Int64`: Index of the basis function.
- `origin::SVector{2,T}`: Origin of the evanescent wave.
- `angle_range::Vector{T}`: The direction angle range for the EPW. Choose such that we do not get Inf anywhere.
- `k::T`: Wavenumber.

# Returns
- `Vector{T}`: Evaluated function values.
"""
function epw(pts::AbstractArray{<:SVector{2,T}},i::Int,origin::SVector{2,T},angle_range::Vector{T},k::T) where {T<:Real}
    θ=angle_range[i] # get our direction θ
    s,c=sincos(θ)
    n=@SVector [c,s] # oscillation n = (cosθ,sinθ)
    d=@SVector [-s,c] # decay d = (−sinθ,cosθ)
    Rs=k*[p.-origin for p in pts] # Vector of SVector{2,T}
    xs=getindex.(Rs,1)
    ys=getindex.(Rs,2) 
    As=@. d[1]*xs+d[2]*ys # ỹ = d⋅(r-C) as per the paper
    Bs=@. n[1]*xs+n[2]*ys # x̃ = n⋅(r-C) as per the paper 
    α_=α(i,k) #  Barnett's algebraic decay
    b=max_billiard_impact(θ,pts,origin,k) # as per Barnett's construction
    sinhα,coshα=sinhcosh(α_)
    offA=@. As+b
    decay=@. exp(-sinhα*offA)
    phase=@. coshα*Bs
    osc=iseven(i) ? cos.(phase) : sin.(phase)
    return @. decay*osc
end


"""
    epw_dk(pts::AbstractArray{<:SVector{2,T}},i::Int,Ni::Int,origin::SVector{2,T},angle_range::Vector{T},k::T) where {T<:Real}

Compute the derivative of the evanescent plane wave function with respect to wavenumber `k`. It uses the following literature:
https://users.flatironinstitute.org/~ahb/thesis_html/node157.html # decay factor and comments
https://arxiv.org/pdf/nlin/0212011 # basis form and comments

# Arguments
- `pts::AbstractArray`: Spatial evaluation points.
- `i::Int64`: Function index.
- `origin::SVector{2,T}`: Origin of the evanescent wave.
- `angle_range::Vector{T}`: The direction angle range for the EPW. Choose such that we do not get Inf anywhere.
- `k::T`: Wavenumber.

# Returns
- `Vector{T}`: Derivative of the function with respect to `k`.
"""
function epw_dk(pts::AbstractArray{<:SVector{2,T}},i::Int,origin::SVector{2,T},angle_range::Vector{T},k::T) where {T<:Real}
    θ=angle_range[i]
    s,c=sincos(θ)
    d= SVector(-s,c)
    n= SVector(c,s)
    Rs=k*[p.-origin for p in pts]
    xs,ys=getindex.(Rs,1),getindex.(Rs,2)
    As=@. d[1]*xs+d[2]*ys
    Bs=@. n[1]*xs+n[2]*ys
    α_=α(i,k)
    dα=dα_dk(i,k)
    b= max_billiard_impact(θ,pts,origin,k)
    b0=b/k # = ∂b/∂k
    sinhα,coshα=sinhcosh(α_)
    offA=@. As+b
    decay=@. exp(-sinhα*offA)
    phase=@. coshα*Bs
    # ∂decay/∂k = decay * [ −(∂sinhα/∂k) * offA  − sinhα * (∂b/∂k) ]
    dsinhα=@. coshα*dα         # since ∂sinh/∂α = coshα
    ddecay_dk=@. decay*(-dsinhα*offA-sinhα*b0 )
    # ∂osc/∂k = {even: −sin(phase), odd: cos(phase)} * [ (∂coshα/∂k)*Bs + coshα*(∂Bs/∂k) ]
    dcoshα=@. sinhα*dα # since ∂cosh/∂α = sinhα
    dBs_dk=b0 # since Bs = k*(n⋅(p-origin))
    if iseven(i)
        dosc_dk=@. -sin(phase)*(dcoshα*Bs+coshα*dBs_dk)
    else
        dosc_dk=@. cos(phase)*(dcoshα*Bs+coshα*dBs_dk)
    end
    return @. ddecay_dk*(iseven(i) ? cos(phase) : sin(phase))+decay*dosc_dk
end

"""
    epw_gradient(pts::AbstractArray{<:SVector{2,T}},i::Int,Ni::Int,origin::SVector{2,T},angle_range::Vector{T},k::T) where {T<:Real}

Compute the gradient (∂/∂x, ∂/∂y) of an evanescent plane wave basis function. It uses the following literature:
https://users.flatironinstitute.org/~ahb/thesis_html/node157.html # decay factor and comments
https://arxiv.org/pdf/nlin/0212011 # basis form and comments

# Arguments
- `pts::AbstractArray`: Points where the gradient is evaluated.
- `i::Int64`: Basis function index.
- `origin::SVector{2,T}`: Origin of the evanescent wave.
- `angle_range::Vector{T}`: The direction angle range for the EPW. Choose such that we do not get Inf anywhere.
- `k::T`: Wavenumber.

# Returns
- `Tuple{Vector{T}, Vector{T}}`: Gradients with respect to x and y.
"""
function epw_gradient(pts::AbstractArray{<:SVector{2,T}},i::Int,origin::SVector{2,T},angle_range::Vector{T},k::T) where {T<:Real}
    θ=angle_range[i]
    s,c=sincos(θ)
    d=SVector(-s,c)
    n=SVector(c,s)
    Rs=k*[p.-origin for p in pts]
    xs,ys=getindex.(Rs,1),getindex.(Rs,2)
    As=@. d[1]*xs+d[2]*ys
    Bs=@. n[1]*xs+n[2]*ys
    α_=α(i,k)
    b=max_billiard_impact(θ, pts, origin, k)
    sinhα,coshα=sinhcosh(α_)
    offA=@. As+b
    decay=@. exp(-sinhα*offA)
    phase=@. coshα*Bs
    # ∂decay/∂A (spatial) = −sinhα * decay
    ddecay_dA=@. -sinhα*decay
    # ∂osc/∂B = even: −sin(phase)*coshα,  odd: cos(phase)*coshα
    if iseven(i)
        dosc_dB=@. -coshα*sin(phase)
        osc=cos.(phase)
    else
        dosc_dB=@. coshα*cos(phase)
        osc=sin.(phase)
    end
    # ∇ψ = (∂ψ/∂A)⋅d + (∂ψ/∂B)⋅n
    dx=@. ddecay_dA*d[1]*osc+decay*dosc_dB*n[1]
    dy=@. ddecay_dA*d[2]*osc+decay*dosc_dB*n[2]
    return dx,dy
end

######################
#### INITILAIZERS ####
######################

struct EvanescentParams{T<:Real}
    angles::Vector{Vector{T}} # a list of angle vectors, one per origin
    origins::Vector{SVector{2,T}} # one origin per angle set
end

struct EvanescentPlaneWaves{T,Sy} <: AbsBasis where  {T<:Real,Sy<:Union{AbsSymmetry,Nothing}}
    cs::PolarCS{T}
    dim::Int64 
    params::EvanescentParams{T}
    symmetries::Union{Vector{Any},Nothing}
    shift_x::T
    shift_y::T
end

function EvanescentPlaneWaves(cs::PolarCS{T},params::EvanescentParams{T},symmetries::Union{Nothing,Vector{Any}}) where {T<:Real}
    dim=length(vcat(params.angles...))
    return EvanescentPlaneWaves{T,typeof(symmetries)}(cs,dim,params,symmetries,zero(T),zero(T))
end

function EvanescentPlaneWaves(cs::PolarCS{T},params::EvanescentParams{T},symmetries::Union{Nothing,Vector{Any}},shift_x::T,shift_y::T) where {T<:Real}
    dim=length(vcat(params.angles...))
    return EvanescentPlaneWaves{T,typeof(symmetries)}(cs,dim,params,symmetries,shift_x,shift_y)
end

function EvanescentPlaneWaves(billiard::Bi,origin_cs::SVector{2,T},params::EvanescentParams{T},rot_angle::T;fundamental=false) where {Bi<:AbsBilliard,T<:Real}
    shift_x=hasproperty(billiard,:x_axis) ? billiard.x_axis : T(0.0)
    shift_y=hasproperty(billiard,:y_axis) ? billiard.y_axis : T(0.0)
    return EvanescentPlaneWaves(PolarCS(origin_cs,rot_angle),params,nothing,shift_x,shift_y)
end

function EvanescentPlaneWaves(billiard::Bi,symmetries::Vector{Any},origin_cs::SVector{2,T},params::EvanescentParams{T},rot_angle::T;fundamental=false) where {Bi<:AbsBilliard,T<:Real}
    shift_x=hasproperty(billiard,:x_axis) ? billiard.x_axis : T(0.0)
    shift_y=hasproperty(billiard,:y_axis) ? billiard.y_axis : T(0.0)
    return EvanescentPlaneWaves(PolarCS(origin_cs,rot_angle),params,symmetries,shift_x,shift_y)
end

###########################################
#### ABSTRACTION WITH EvanescentParams ####
###########################################

"""
    epw(
        pts::AbstractArray{<:SVector{2,T}},
        i::Int,
        params::EvanescentParams{T},
        k::T
    ) -> Vector{T}

Compute the real-valued evanescent plane‐wave basis function at a set of points,
summing contributions from all configured origins and angles.

# Arguments
- `pts::AbstractArray{<:SVector{2,T}}`: Coordinates of evaluation points.
- `i::Int`: Linear index of the basis function (1 ≤ i ≤ total dim).
- `params::EvanescentParams{T}`: Contains:
  - `params.origins`: Vector of 2D corner origins.
  - `params.angles`: Vector of angle‐vectors, one per origin.
  - `params.αs`: Vector of decay constants (α) matching each origin.
- `k::T`: Wavenumber.

# Returns
- `Vector{T}`: Values of the i-th EPW function at each point in `pts`.
"""
function epw(pts::AbstractArray,i::Int64,params::EvanescentParams{T},k::T) where {T<:Real}
    origins=params.origins
    angle_ranges=params.angles
    N=length(pts)
    M=length(origins)
    res=Matrix{Complex{T}}(undef,N,M) # pts x origins
    for j in eachindex(origins) 
        @inbounds res[:,j]=epw(pts,i,origins[j],angle_ranges[j],k)
    end
    return sum(res,dims=2)[:] # for each row sum over all columns to get for each pt in pts all the different origin contributions. Converts Matrix (N,1) to a flat vector.
end

"""
    epw_dk(
        pts::AbstractArray{<:SVector{2,T}},
        i::Int,
        params::EvanescentParams{T},
        k::T
    ) -> Vector{T}

Compute the derivative of the evanescent plane‐wave basis function with respect to `k`,
summing contributions across all origins.

# Arguments
- `pts::AbstractArray{<:SVector{2,T}}`: Coordinates of evaluation points.
- `i::Int`: Basis function index.
- `params::EvanescentParams{T}`: As in `epw`.
- `k::T`: Wavenumber.

# Returns
- `Vector{T}`: ∂/∂k of the i-th EPW at each point.
"""
function epw_dk(pts::AbstractArray,i::Int64,params::EvanescentParams{T},k::T) where {T<:Real,Ti<:Integer}
    origins=params.origins
    angle_ranges=params.angles
    N=length(pts)
    M=length(origins)
    res=Matrix{Complex{T}}(undef,N,M)
    for j in eachindex(origins)
        @inbounds res[:,j]=epw_dk(pts,i,origins[j],angle_ranges[j],k)
    end
    return sum(res,dims=2)[:]
end

"""
    epw_gradient(
        pts::AbstractArray{<:SVector{2,T}},
        i::Int,
        params::EvanescentParams{T},
        k::T
    ) -> Tuple{Vector{T},Vector{T}}

Compute the spatial gradient (∂/∂x, ∂/∂y) of the evanescent plane‐wave basis function,
summing contributions from all origins.

# Arguments
- `pts::AbstractArray{<:SVector{2,T}}`: Coordinates of evaluation points.
- `i::Int`: Basis function index.
- `params::EvanescentParams{T}`: As in `epw`.
- `k::T`: Wavenumber.

# Returns
- `(dx, dy)::Tuple{Vector{T},Vector{T}}`: Partial derivatives of the i-th EPW with respect to x and y.
"""
function epw_gradient(pts::AbstractArray,i::Int64,params::EvanescentParams{T},k::T) where {T<:Real,Ti<:Integer}
    origins=params.origins
    angle_ranges=params.angles
    N=length(pts)
    M=length(origins)
    dx_mat=Matrix{Complex{T}}(undef,N,M)
    dy_mat=Matrix{Complex{T}}(undef,N,M)
    for j in eachindex(origins)
        dx,dy=epw_gradient(pts,i,origins[j],angle_ranges[j],k)
        @inbounds dx_mat[:,j]=dx
        @inbounds dy_mat[:,j]=dy
    end
    dx=sum(dx_mat,dims=2)[:]
    dy=sum(dy_mat,dims=2)[:]
    return dx,dy
end

#########################################################################################################
#### HELPERS FOR GETTING ALL THE POTENTIAL CORNERS AND SERVES AS POTENTIAL INPUT TO EPW CONSTRUCTORS ####
#########################################################################################################

"""
    get_origins_(billiard::Bi, idx::Ti; fundamental=false) -> Vector{SVector{2, elt}}

Compute the corner origins for a single boundary segment `idx` of the billiard.

# Arguments
- `billiard::Bi<:AbsBilliard`: The billiard object containing boundary curves.
- `idx::Ti<:Integer`: Index of the boundary segment to examine.
- `fundamental::Bool=false`: If `true`, use `fundamental_boundary`; otherwise `full_boundary`.

# Returns
- `Vector{SVector{2, elt}}`: A vector of 2D `SVector` points representing corner origins. 
  Returns a vector containing the starting point of the curve at `idx` if it is an `AbsRealCurve` immediately following a `AbsRealCurve` (i.e., a corner); otherwise an empty vector.
"""
function get_origins_(billiard::Bi,idx::Ti;fundamental=false) where {Bi<:AbsBilliard,Ti<:Integer}
    boundary= fundamental ? billiard.fundamental_boundary : billiard.full_boundary
    elt=eltype(boundary[1].length)
    N=length(boundary)
    origins=Vector{SVector{2,elt}}()
    crv=boundary[idx]
    if crv isa AbsRealCurve && boundary[mod1(idx-1,N)] isa AbsRealCurve
        push!(origins,curve(crv,zero(elt)))
    end
    return origins
end

"""
    get_origins_(billiard::Bi, idxs::AbstractArray; fundamental=false) -> Vector{SVector{2, elt}}

Compute the corner origins for multiple boundary segments `idxs` of the billiard.

# Arguments
- `billiard::Bi<:AbsBilliard`: The billiard object containing boundary curves.
- `idxs::AbstractArray{<:Integer}`: A collection of indices of boundary segments to examine.
- `fundamental::Bool=false`: If `true`, use `fundamental_boundary`; otherwise `full_boundary`.

# Returns
- `Vector{SVector{2, elt}}`: A vector of 2D `SVector` points for each segment in `idxs` that is an `AbsRealCurve` immediately following a `AbsRealCurve`.

# Throws
- AssertionError if `length(idxs)` exceeds the total number of boundary segments.
"""
function get_origins_(billiard::Bi,idxs::AbstractArray;fundamental=false) where {Bi<:AbsBilliard}
    boundary= fundamental ? billiard.fundamental_boundary : billiard.full_boundary
    elt=eltype(boundary[1].length)
    N=length(boundary)
    @assert length(idxs)<=N "The number of idxs cannot be larger than the number of boundary segments. Check if fundamental kwarg is set correctly!"
    origins=Vector{SVector{2,elt}}()
    for idx in idxs 
        crv=boundary[idx]
        if crv isa AbsRealCurve && boundary[mod1(idx-1,N)] isa AbsRealCurve # is the other curve is virtual then usually the BCs are already satisfied there so no need to add.
            push!(origins,curve(crv,zero(elt))) # starting corner 
        end
    end
    return origins
end

"""
    get_origins_(billiard::Bi; fundamental=false) -> Vector{SVector{2, elt}}

Compute all corner origins for the entire boundary of the billiard.

# Arguments
- `billiard::Bi<:AbsBilliard`: The billiard object containing boundary curves.
- `fundamental::Bool=false`: If `true`, use `fundamental_boundary`; otherwise `full_boundary`.

# Returns
- `Vector{SVector{2, elt}}`: A vector of all corner origin points, equivalent to
  `get_origins_(billiard, eachindex(boundary); fundamental=fundamental)`.
"""
function get_origins_(billiard::Bi;fundamental=false) where {Bi<:AbsBilliard}
    boundary=fundamental ? billiard.fundamental_boundary : billiard.full_boundary
    return get_origins_(billiard,eachindex(boundary);fundamental=fundamental)
end

###################################################################################
#### SINGLE INDEX CONSTRUCTION - SEPARATE SINCE WE SYMMETRIZE EPW AT THIS STEP ####
###################################################################################

toFloat32(basis::EvanescentPlaneWaves)=EvanescentPlaneWaves(PolarCS(Float32.(basis.cs.origin),basis.cs.rot_angle),basis.dim,basis.params,basis.symmetries)
function resize_basis(basis::EvanescentPlaneWaves,billiard::Bi,dim::Int,k) where {Bi<:AbsBilliard} # compatibility function, does not do anything
    return EvanescentPlaneWaves(basis.cs,basis.params,basis.symmetries)
end

@inline reflect_x_epw(p::SVector{2,T},shift_x::T) where {T<:Real} = SVector(2*shift_x-p[1],p[2])
@inline reflect_y_epw(p::SVector{2,T},shift_y::T) where {T<:Real} = SVector(p[1],2*shift_y-p[2])
@inline reflect_xy_epw(p::SVector{2,T},shift_x::T,shift_y::T) where {T<:Real} = SVector(2*shift_x-p[1],2*shift_y-p[2])

@inline function symmetrize_epw(f::F,basis::EvanescentPlaneWaves{T},i::Int,k::T,pts::Vector{SVector{2,T}}) where {F<:Function,T<:Real}
    syms=basis.symmetries
    isnothing(syms) && return f(basis,i,k,pts)  # No symmetry applied
    sym=syms[1]
    origin=basis.cs.origin
    fval=f(pts,i,basis.params,k)
    if sym.axis==:y_axis # XReflection
        px=sym.parity
        reflected_pts_x=reflect_x_epw.(pts,Ref(basis.shift_x))
        return 0.5*(fval.+px.*f(reflected_pts_x,i,basis.params,k))
    elseif sym.axis==:x_axis # YReflection
        py=sym.parity
        reflected_pts_y=reflect_y_epw.(pts,Ref(basis.shift_y))
        return 0.5*(fval.+py.*f(reflected_pts_y,i,basis.params,k))
    elseif sym.axis==:origin # XYReflection
        px,py=sym.parity
        reflected_pts_x=reflect_x_epw.(pts,Ref(basis.shift_x))
        reflected_pts_y=reflect_y_epw.(pts,Ref(basis.shift_y))
        reflected_pts_xy=reflect_xy_epw.(pts,Ref(basis.shift_x),Ref(basis.shift_y))
        return 0.25*(fval.+px.*f(reflected_pts_x,i,basis.params,k).+py.*f(reflected_pts_y,i,basis.params,k).+(px*py).*f(reflected_pts_xy,i,basis.params,k))
    else
        @error "Unsupported symmetry type: $(typeof(sym)). Symmetrization skipped."
        return fval
    end
end

@inline function symmetrize_epw_grad(f::F,basis::EvanescentPlaneWaves{T},i::Int,k::T,pts::Vector{SVector{2,T}}) where {F<:Function,T<:Real}
    syms=basis.symmetries
    isnothing(syms) && return f(pts,i,basis.params,k)
    sym=syms[1]
    origin=basis.cs.origin
    fval=f(pts,i,basis.params,k)
    if sym.axis==:y_axis # XReflection
        px=sym.parity
        reflected_pts_x=reflect_x_epw.(pts,Ref(basis.shift_x))
        fx=f(reflected_pts_x,i,basis.params,k)
        return (
            0.5.*(fval[1].+px.*fx[1]),
            0.5.*(fval[2].+px.*fx[2]))
    elseif sym.axis==:x_axis # YReflection
        py=sym.parity
        reflected_pts_y=reflect_y_epw.(pts,Ref(basis.shift_y))
        fy=f(reflected_pts_y,i,basis.params,k)
        return (
            0.5.*(fval[1].+py.*fy[1]),
            0.5.*(fval[2].+py.*fy[2]))
    elseif sym.axis==:origin # XYReflection
        px,py=sym.parity
        reflected_pts_x=reflect_x_epw.(pts,Ref(basis.shift_x))
        reflected_pts_y=reflect_y_epw.(pts,Ref(basis.shift_y))
        reflected_pts_xy=reflect_xy_epw.(pts,Ref(basis.shift_x),Ref(basis.shift_y))
        fx=f(reflected_pts_x,i,basis.params,k)
        fy=f(reflected_pts_y,i,basis.params,k)
        fxy=f(reflected_pts_xy,i,basis.params,k)
        return (
            0.25.*(fval[1].+px.*fx[1].+py.*fy[1].+(px*py).*fxy[1]),
            0.25.*(fval[2].+px.*fx[2].+py.*fy[2].+(px*py).*fxy[2]))
    else
        @error "Unsupported symmetry type: $(typeof(sym)). Symmetrization skipped."
        return fval
    end
end

@inline function basis_fun(basis::EvanescentPlaneWaves{T},i::Int,k::T,pts::AbstractArray) where {T<:Real}
    return symmetrize_epw(epw,basis,i,k,pts)
end

@inline function dk_fun(basis::EvanescentPlaneWaves{T},i::Int,k::T,pts::AbstractArray) where {T<:Real}
    return symmetrize_epw(epw_dk,basis,i,k,pts)
end

function gradient(basis::EvanescentPlaneWaves{T},i::Int,k::T,pts::AbstractArray) where {T<:Real}
    return symmetrize_epw_grad(epw_gradient,basis,i,k,pts)
end

function basis_and_gradient(basis::EvanescentPlaneWaves{T},i::Int,k::T,pts::AbstractArray) where {T<:Real}
    basis_vec=basis_fun(basis,i,k,pts)
    vec_dX,vec_dY=gradient(basis,i,k,pts)
    return basis_vec,vec_dX,vec_dY
end

##################################
#### MULTI INDEX CONSTRUCTION ####
##################################

"""
    basis_fun(basis::EvanescentPlaneWaves{T}, indices::AbstractArray, k::T, pts::AbstractArray; multithreaded=true) 
    -> Matrix{T}
 
Computes values of the Evanescent Plane Wave (EPW) basis functions at given points for the provided indices.
 
# Arguments
- `basis::EvanescentPlaneWaves{T}`: The evanescent basis to evaluate.
- `indices::AbstractArray`: Indices specifying which EPW functions to compute.
- `k::T`: Wavenumber parameter.
- `pts::AbstractArray`: 2D points where functions are evaluated.
- `multithreaded::Bool=true`: Enables multithreading support.
 
# Returns
- `Matrix{T}`: Matrix of function values.
"""
function basis_fun(basis::EvanescentPlaneWaves{T},indices::AbstractArray,k::T,pts::AbstractArray;multithreaded::Bool=true) where {T<:Real}
    N=length(pts)
    M=length(indices)
    mat=zeros(T,N,M)
    @use_threads multithreading=multithreaded for i in eachindex(indices) 
        @inbounds mat[:,i] .= basis_fun(basis,i,k,pts)
    end
    return mat
end

"""
    dk_fun(basis::EvanescentPlaneWaves{T}, indices::AbstractArray, k::T, pts::AbstractArray; multithreaded=true) 
    -> Matrix{T}
 
Evaluates the derivative of the Evanescent Plane Wave (EPW) basis functions with respect to the wavenumber `k`.
 
# Arguments
- `basis::EvanescentPlaneWaves{T}`: The evanescent basis structure.
- `indices::AbstractArray`: Indices of the EPW functions.
- `k::T`: Wavenumber.
- `pts::AbstractArray`: Spatial points where functions are evaluated.
- `multithreaded::Bool=true`: Enables threaded evaluation.
 
# Returns
- `Matrix{T}`: Matrix of ∂f/∂k evaluations.
"""
function dk_fun(basis::EvanescentPlaneWaves{T},indices::AbstractArray,k::T,pts::AbstractArray;multithreaded::Bool=true) where {T<:Real}
    N=length(pts)
    M=length(indices)
    mat=zeros(T,N,M)
    @use_threads multithreading=multithreaded for i in eachindex(indices) 
        @inbounds mat[:,i] .= dk_fun(basis,i,k,pts)
    end
    return mat
end

"""
    gradient(basis::EvanescentPlaneWaves{T}, indices::AbstractArray, k::T, pts::AbstractArray; multithreaded=true) 
    -> Tuple{Matrix{T}, Matrix{T}}
 
Computes the gradient (in x and y directions) of Evanescent Plane Wave (EPW) basis functions for a set of indices.
 
# Arguments
- `basis::EvanescentPlaneWaves{T}`: The evanescent basis object.
- `indices::AbstractArray`: Array of indices specifying which EPWs to evaluate.
- `k::T`: Wavenumber parameter.
- `pts::AbstractArray`: Evaluation points in 2D.
- `multithreaded::Bool=true`: Whether to use multithreading.
 
# Returns
- `Tuple{Matrix{T}, Matrix{T}}`: Matrices of partial derivatives with respect to x and y.
"""
function gradient(basis::EvanescentPlaneWaves{T},indices::AbstractArray,k::T,pts::AbstractArray;multithreaded::Bool=true) where {T<:Real}
    N=length(pts)
    M=length(indices)
    mat_dX=zeros(T,N,M)
    mat_dY=zeros(T,N,M)
    @use_threads multithreading=multithreaded for i in eachindex(indices) 
        dx,dy=gradient(basis,i,k,pts)
        @inbounds mat_dX[:,i]=dx
        @inbounds mat_dY[:,i]=dy
    end
    return mat_dX,mat_dY
end

"""
    basis_and_gradient(basis::EvanescentPlaneWaves{T}, indices::AbstractArray, k::T, pts::AbstractArray; multithreaded=true) 
    -> Tuple{Matrix{T}, Matrix{T}, Matrix{T}}
 
Computes both values and gradients of the Evanescent Plane Wave (EPW) basis functions over multiple indices.
 
# Arguments
- `basis::EvanescentPlaneWaves{T}`: The evanescent basis structure.
- `indices::AbstractArray`: Indices specifying which EPW functions to evaluate.
- `k::T`: Wavenumber parameter.
- `pts::AbstractArray`: Array of points where the basis is evaluated.
- `multithreaded::Bool=true`: Enables multithreaded computation if `true`.
 
# Returns
- `Tuple{Matrix{T}, Matrix{T}, Matrix{T}}`: Tuple containing values, x-derivatives, and y-derivatives of basis functions.
"""
function basis_and_gradient(basis::EvanescentPlaneWaves{T},indices::AbstractArray,k::T,pts::AbstractArray;multithreaded::Bool=true) where {T<:Real}
    mat=basis_fun(basis,indices,k,pts;multithreaded=multithreaded)
    mat_dX,mat_dY=gradient(basis,indices,k,pts;multithreaded=multithreaded)
    return mat,mat_dX,mat_dY
end
