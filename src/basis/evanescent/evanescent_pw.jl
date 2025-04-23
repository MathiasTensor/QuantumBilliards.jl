using LinearAlgebra, CoordinateTransformations, Rotations, StaticArrays

"""
MAIN REFERENCE: Jan Wiersig, Gabriel G. Carlo, Evanescent wave approach to diﬀractive phenomena in convex billiards with corners
https://arxiv.org/pdf/nlin/0212011
AUXILIARY REFERENCE: Alex Barnett, PhD Thesis
https://users.flatironinstitute.org/~ahb/thesis_html/node157.html
"""

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

# linearly sample i=1:Ni in [φ₁,φ₂]
function _theta_i(i::Ti,Ni::Ti,φ₁::K,φ₂::K) where {T<:Real,Ti<:Integer,K<:Real}
    return φ₁+(i-1)/(Ni-1)*(φ₂-φ₁)
end

"""
    epw(pts::AbstractArray{<:SVector{2,T}},i::Int,Ni::Int,origin::SVector{2,T},angle_range::Tuple{K,K},k::T) where {T<:Real,K<:Real}

Compute the evanescent plane wave function at given points for a single origin. It uses the following literature:
https://users.flatironinstitute.org/~ahb/thesis_html/node157.html # decay factor and comments
https://arxiv.org/pdf/nlin/0212011 # basis form and comments

# Arguments
- `pts::AbstractArray`: Points in space where the function is evaluated.
- `i::Int64`: Index of the basis function.
- `Ni::Ti`: Total number of basis functions.
- `origin::SVector{2,T}`: Origin of the evanescent wave.
- `angle_range::Tuple{K,K}`: The direction angle range for the EPW. Choose such that we do not get Inf anywhere.
- `k::T`: Wavenumber.

# Returns
- `Vector{T}`: Evaluated function values.
"""
function epw(pts::AbstractArray{<:SVector{2,T}},i::Int,Ni::Int,origin::SVector{2,T},angle_range::Tuple{K,K},k::T) where {T<:Real,K<:Real}
    φ₁,φ₂=sort(Vector(angle_range)) # unpack & sort wedge
    θ=_theta_i(i,Ni,φ₁,φ₂) # get our direction θ
    s,c=sincos(θ)
    n=@SVector [c,s] # oscillation n = (cosθ,sinθ)
    d=@SVector [-s,c] # decay d = (−sinθ,cosθ)
    Rs=[p.-origin for p in pts]  # shift to corner‐origin
    As=[dot(d,r) for r in Rs]   # ỹ = d⋅(r-C) as per the paper
    Bs=[dot(n,r) for r in Rs]   # x̃ = n⋅(r-C) as per the paper
    α=(3+i)/(2*k^(1/3)) # here I borrow Barnett's decay parameter
    decay=exp.(-k*sinh(α).*As) # decay and phase
    phase=k*cosh(α).*Bs
    osc=iseven(i) ? cos.(phase) : sin.(phase) # real‐basis: even=i⇒cos, odd⇒sin
    return decay.*osc
end


"""
    epw_dk(pts::AbstractArray{<:SVector{2,T}},i::Int,Ni::Int,origin::SVector{2,T},angle_range::Tuple{K,K},k::T) where {T<:Real,K<:Real}

Compute the derivative of the evanescent plane wave function with respect to wavenumber `k`. It uses the following literature:
https://users.flatironinstitute.org/~ahb/thesis_html/node157.html # decay factor and comments
https://arxiv.org/pdf/nlin/0212011 # basis form and comments

# Arguments
- `pts::AbstractArray`: Spatial evaluation points.
- `i::Int64`: Function index.
- `Ni::Ti`: Total number of basis functions.
- `origin::SVector{2,T}`: Origin of the evanescent wave.
- `angle_range::Tuple{K,K}`: The direction angle range for the EPW. Choose such that we do not get Inf anywhere.
- `k::T`: Wavenumber.

# Returns
- `Vector{T}`: Derivative of the function with respect to `k`.
"""
function epw_dk(pts::AbstractArray{<:SVector{2,T}},i::Int,Ni::Int,origin::SVector{2,T},angle_range::Tuple{K,K},k::T) where {T<:Real,K<:Real}
    φ₁,φ₂=sort(Vector(angle_range))
    θ=_theta_i(i,Ni,φ₁,φ₂)
    s,c=sincos(θ)
    n=@SVector [c,s]
    d=@SVector [-s,c]
    Rs=[p.-origin for p in pts]
    As=[dot(d,r) for r in Rs]
    Bs=[dot(n,r) for r in Rs]
    α=(3+i)/(2*k^(1/3))
    dα=-(3+i)/(6*k^(4/3))
    # decay = exp(-k*sinh(α)*A)
    # ∂decay/∂k = decay * [ -sinh(α)*A - k*cosh(α)*A*dα ]
    sinhα,coshα=sinhcosh(α)
    decay=exp.(-k*sinhα.*As)
    ddecay=decay.*(.-sinhα.*As.-(k*coshα.*As.*dα))
    # phase = k*cosh(α)*B
    # ∂phase/∂k = cosh(α)*B + k*sinh(α)*B*dα
    phase=k*coshα.*Bs
    dphase_dk=coshα.*Bs.+k*sinhα.*Bs.*dα
    if iseven(i) # osc = cos(phase) or sin(phase)
        osc=cos.(phase)
        dosc_dk=-sin.(phase).*dphase_dk
    else
        osc= sin.(phase)
        dosc_dk=cos.(phase).*dphase_dk
    end
    return ddecay.*osc.+decay.*dosc_dk
end

"""
    epw_gradient(pts::AbstractArray{<:SVector{2,T}},i::Int,Ni::Int,origin::SVector{2,T},angle_range::Tuple{K,K},k::T) where {T<:Real,K<:Real}

Compute the gradient (∂/∂x, ∂/∂y) of an evanescent plane wave basis function. It uses the following literature:
https://users.flatironinstitute.org/~ahb/thesis_html/node157.html # decay factor and comments
https://arxiv.org/pdf/nlin/0212011 # basis form and comments

# Arguments
- `pts::AbstractArray`: Points where the gradient is evaluated.
- `i::Int64`: Basis function index.
- `Ni::Ti`: Number of basis functions.
- `origin::SVector{2,T}`: Origin of the evanescent wave.
- `angle_range::Tuple{K,K}`: The direction angle range for the EPW. Choose such that we do not get Inf anywhere.
- `k::T`: Wavenumber.

# Returns
- `Tuple{Vector{T}, Vector{T}}`: Gradients with respect to x and y.
"""
function epw_gradient(pts::AbstractArray{<:SVector{2,T}},i::Int,Ni::Int,origin::SVector{2,T},angle_range::Tuple{K,K},k::T) where {T<:Real,K<:Real}
    φ₁,φ₂=sort(Vector(angle_range))
    θ=_theta_i(i, Ni, φ₁, φ₂)
    s,c=sincos(θ)
    n=@SVector [c,s];d=@SVector [-s,c]
    Rs=[p.-origin for p in pts]
    As=[dot(d,r) for r in Rs]
    Bs=[dot(n,r) for r in Rs]
    α=(3+i)/(2*k^(1/3))
    sinhα,coshα=sinhcosh(α)
    # decay = exp(-k*sinhα*A)
    decay=exp.(-k*sinhα.*As)
    # ∂decay/∂A = -k*sinhα * decay
    ddecay_dA=-k*sinhα.*decay
    # phase = k*coshα*B
    phase=k*coshα.*Bs
    # ∂osc/∂B = { -k*coshα*sin(phase),  k*coshα*cos(phase) }
    if iseven(i)
        osc=cos.(phase)
        dosc_dB=-k*coshα.*sin.(phase)
    else
        osc=sin.(phase)
        dosc_dB=k*coshα.*cos.(phase)
    end
    # ∇A = d,  ∇B = n
    # so ∂ψ/∂x = ddecay_dA*d_x * osc + decay * dosc_dB*n_x, etc.
    dx=[ddecay_dA[j]*d[1]*osc[j]+decay[j]*dosc_dB[j]*n[1] for j in eachindex(Rs)]
    dy=[ddecay_dA[j]*d[2]*osc[j]+decay[j]*dosc_dB[j]*n[2] for j in eachindex(Rs)]
    return dx, dy
end

"""
    epw(pts::AbstractArray,i::Int64,Ni::Ti,origins::Vector{SVector{2,T}},angle_ranges::Vector{Tuple{K,K}},k::T) where {T<:Real,Ti<:Integer,K<:Real}

Evaluate the evanescent plane wave by summing contributions from multiple origins.

# Arguments
- `pts::AbstractArray`: Evaluation points.
- `i::Int64`: Basis function index.
- `Ni::Ti`: Total number of basis functions.
- `origins::Vector{SVector{2,T}}`: List of origins.
- `angle_ranges::Vector{Tuple{K,K}}`: The direction angle range for the EPW. Choose such that we do not get Inf anywhere. For all origins index wise.
- `k::T`: Wavenumber.

# Returns
- `Vector{T}`: Summed function values at each point.
"""
function epw(pts::AbstractArray,i::Int64,Ni::Ti,origins::Vector{SVector{2,T}},angle_ranges::Vector{Tuple{K,K}},k::T) where {T<:Real,Ti<:Integer,K<:Real}
    N=length(pts)
    M=length(origins)
    res=Matrix{Complex{T}}(undef,N,M) # pts x origins
    for j in eachindex(origins) 
        @inbounds res[:,j]=epw(pts,i,Ni,origins[j],angle_ranges[j],k)
    end
    return sum(res,dims=2)[:] # for each row sum over all columns to get for each pt in pts all the different origin contributions. Converts Matrix (N,1) to a flat vector.
end

"""
    epw_dk(pts::AbstractArray,i::Int64,Ni::Ti,origins::Vector{SVector{2,T}},angle_ranges::Vector{Tuple{K,K}},k::T) where {T<:Real,Ti<:Integer,K<:Real,K<:Real}

Compute the summed ∂/∂k of EPW from multiple origins.

# Arguments
- `pts::AbstractArray`: Evaluation points.
- `i::Int64`: Basis function index.
- `Ni::Ti`: Number of basis functions.
- `origins::Vector{SVector{2,T}}`: Origins for summation.
- `angle_ranges::Vector{Tuple{K,K}}`: The direction angle range for the EPW. Choose such that we do not get Inf anywhere. For all origins index wise.
- `k::T`: Wavenumber.

# Returns
- `Vector{T}`: Derivatives summed across all origins.
"""
function epw_dk(pts::AbstractArray,i::Int64,Ni::Ti,origins::Vector{SVector{2,T}},angle_ranges::Vector{Tuple{K,K}},k::T) where {T<:Real,Ti<:Integer,K<:Real}
    N=length(pts)
    M=length(origins)
    res=Matrix{Complex{T}}(undef,N,M)
    for j in eachindex(origins)
        @inbounds res[:,j]=epw_dk(pts,i,Ni,origins[j],angle_ranges[j],k)
    end
    return sum(res,dims=2)[:]
end

"""
    epw_gradient(pts::AbstractArray,i::Int64,Ni::Ti,origins::Vector{SVector{2,T}},angle_ranges::Vector{Tuple{K,K}},k::T) where {T<:Real,Ti<:Integer,K<:Real}

Compute the gradient (∂/∂x, ∂/∂y) of the evanescent plane wave summed across all origins.

# Arguments
- `pts::AbstractArray`: Points to evaluate.
- `i::Int64`: Index of basis function.
- `Ni::Ti`: Number of total functions.
- `origins::Vector{SVector{2,T}}`: EPW origins.
- `angle_ranges::Vector{Tuple{K,K}}`: The direction angle range for the EPW. Choose such that we do not get Inf anywhere. For all origins index wise.
- `k::T`: Wavenumber.

# Returns
- `Tuple{Vector{T}, Vector{T}}`: Gradient in x and y directions.
"""
function epw_gradient(pts::AbstractArray,i::Int64,Ni::Ti,origins::Vector{SVector{2,T}},angle_ranges::Vector{Tuple{K,K}},k::T) where {T<:Real,Ti<:Integer,K<:Real}
    N=length(pts)
    M=length(origins)
    dx_mat=Matrix{Complex{T}}(undef,N,M)
    dy_mat=Matrix{Complex{T}}(undef,N,M)
    for j in eachindex(origins)
        dx,dy=epw_gradient(pts,i,Ni,origins[j],angle_ranges[j],k)
        @inbounds dx_mat[:,j]=dx
        @inbounds dy_mat[:,j]=dy
    end
    dx=sum(dx_mat,dims=2)[:]
    dy=sum(dy_mat,dims=2)[:]
    return dx,dy
end

struct EvanescentPlaneWaves{T,Sy,K} <: AbsBasis where  {T<:Real,Sy<:Union{AbsSymmetry,Nothing},K<:Real}
    cs::PolarCS{T}
    dim::Int64 
    origins::Vector{SVector{2,T}}
    angle_ranges::Vector{Tuple{K,K}}
    symmetries::Union{Vector{Any},Nothing}
    shift_x::T
    shift_y::T
end

function EvanescentPlaneWaves(cs::PolarCS{T},dim::Int,origins::Vector{SVector{2,T}},angle_ranges::Vector{Tuple{K,K}},symmetries::Union{Nothing,Vector{Any}}) where {T<:Real,K<:Real}
    EvanescentPlaneWaves{T,typeof(symmetries),K}(cs,dim,origins,angle_ranges,symmetries,zero(T),zero(T))
end

function EvanescentPlaneWaves(cs::PolarCS{T},dim::Int,origins::Vector{SVector{2,T}},angle_ranges::Vector{Tuple{K,K}},symmetries::Union{Nothing,Vector{Any}},shift_x::T,shift_y::T) where {T<:Real,K<:Real}
    EvanescentPlaneWaves{T,typeof(symmetries),K}(cs,dim,origins,angle_ranges,symmetries,shift_x,shift_y)
end

function EvanescentPlaneWaves(billiard::Bi,origin_cs::SVector{2,T},origins::Vector{SVector{2,T}},angle_ranges::Vector{Tuple{K,K}},rot_angle::T;fundamental=false) where {Bi<:AbsBilliard,T<:Real,K<:Real}
    shift_x=hasproperty(billiard,:x_axis) ? billiard.x_axis : T(0.0)
    shift_y=hasproperty(billiard,:y_axis) ? billiard.y_axis : T(0.0)
    return EvanescentPlaneWaves(PolarCS(origin_cs,rot_angle),10,origins,angle_ranges,nothing,shift_x,shift_y)
end

function EvanescentPlaneWaves(billiard::Bi,symmetries::Vector{Any},origin_cs::SVector{2,T},origins::Vector{SVector{2,T}},angle_ranges::Vector{Tuple{K,K}},rot_angle::T;fundamental=false) where {Bi<:AbsBilliard,T<:Real,K<:Real}
    shift_x=hasproperty(billiard,:x_axis) ? billiard.x_axis : T(0.0)
    shift_y=hasproperty(billiard,:y_axis) ? billiard.y_axis : T(0.0)
    return EvanescentPlaneWaves(PolarCS(origin_cs,rot_angle),10,origins,angle_ranges,symmetries,shift_x,shift_y)
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

toFloat32(basis::EvanescentPlaneWaves)=EvanescentPlaneWaves(PolarCS(Float32.(basis.cs.origin),basis.cs.rot_angle),basis.dim,Float32.(basis.origins),basis.angle_ranges,basis.symmetries)
function resize_basis(basis::EvanescentPlaneWaves,billiard::Bi,dim::Int,k) where {Bi<:AbsBilliard}
    new_dim=max_i(k) # use Barnett's algebraic progression eq. for the basis size determination
    if new_dim==basis.dim
        return basis
    else
        return EvanescentPlaneWaves(basis.cs,new_dim,basis.origins,basis.angle_ranges,basis.symmetries)
    end
end

@inline reflect_x_epw(p::SVector{2,T},shift_x::T) where {T<:Real} = SVector(2*shift_x-p[1],p[2])
@inline reflect_y_epw(p::SVector{2,T},shift_y::T) where {T<:Real} = SVector(p[1],2*shift_y-p[2])
@inline reflect_xy_epw(p::SVector{2,T},shift_x::T,shift_y::T) where {T<:Real} = SVector(2*shift_x-p[1],2*shift_y-p[2])

@inline function symmetrize_epw(f::F,basis::EvanescentPlaneWaves{T},i::Int,k::T,pts::Vector{SVector{2,T}}) where {F<:Function,T<:Real}
    syms=basis.symmetries
    isnothing(syms) && return f(basis,i,k,pts)  # No symmetry applied
    sym=syms[1]
    origin=basis.cs.origin
    fval=f(pts,i,basis.dim,basis.origins,basis.angle_ranges,k)
    if sym.axis==:y_axis # XReflection
        px=sym.parity
        reflected_pts_x=reflect_x_epw.(pts,Ref(basis.shift_x))
        return 0.5*(fval.+px.*f(reflected_pts_x,i,basis.dim,basis.origins,basis.angle_ranges,k))
    elseif sym.axis==:x_axis # YReflection
        py=sym.parity
        reflected_pts_y=reflect_y_epw.(pts,Ref(basis.shift_y))
        return 0.5*(fval.+py.*f(reflected_pts_y,i,basis.dim,basis.origins,basis.angle_ranges,k))
    elseif sym.axis==:origin # XYReflection
        px,py=sym.parity
        reflected_pts_x=reflect_x_epw.(pts,Ref(basis.shift_x))
        reflected_pts_y=reflect_y_epw.(pts,Ref(basis.shift_y))
        reflected_pts_xy=reflect_xy_epw.(pts,Ref(basis.shift_x),Ref(basis.shift_y))
        return 0.25*(fval.+px.*f(reflected_pts_x,i,basis.dim,basis.origins,basis.angle_ranges,k).+py.*f(reflected_pts_y,i,basis.dim,basis.origins,basis.angle_ranges,k).+(px*py).*f(reflected_pts_xy,i,basis.dim,basis.origins,basis.angle_ranges,k))
    else
        @error "Unsupported symmetry type: $(typeof(sym)). Symmetrization skipped."
        return fval
    end
end

@inline function symmetrize_epw_grad(f::F,basis::EvanescentPlaneWaves{T},i::Int,k::T,pts::Vector{SVector{2,T}}) where {F<:Function,T<:Real}
    syms=basis.symmetries
    isnothing(syms) && return f(pts,i,basis.dim,basis.origins,basis.angle_ranges,k)
    sym=syms[1]
    origin=basis.cs.origin
    fval=f(pts,i,basis.dim,basis.origins,basis.angle_ranges,k)
    if sym.axis==:y_axis # XReflection
        px=sym.parity
        reflected_pts_x=reflect_x_epw.(pts,Ref(basis.shift_x))
        fx=f(reflected_pts_x,i,basis.dim,basis.origins,basis.angle_ranges,k)
        return (
            0.5.*(fval[1].+px.*fx[1]),
            0.5.*(fval[2].+px.*fx[2]))
    elseif sym.axis==:x_axis # YReflection
        py=sym.parity
        reflected_pts_y=reflect_y_epw.(pts,Ref(basis.shift_y))
        fy=f(reflected_pts_y,i,basis.dim,basis.origins,basis.angle_ranges,k)
        return (
            0.5.*(fval[1].+py.*fy[1]),
            0.5.*(fval[2].+py.*fy[2]))
    elseif sym.axis==:origin # XYReflection
        px,py=sym.parity
        reflected_pts_x=reflect_x_epw.(pts,Ref(basis.shift_x))
        reflected_pts_y=reflect_y_epw.(pts,Ref(basis.shift_y))
        reflected_pts_xy=reflect_xy_epw.(pts,Ref(basis.shift_x),Ref(basis.shift_y))
        fx=f(reflected_pts_x,i,basis.dim,basis.origins,basis.angle_ranges,k)
        fy=f(reflected_pts_y,i,basis.dim,basis.origins,basis.angle_ranges,k)
        fxy=f(reflected_pts_xy,i,basis.dim,basis.origins,basis.angle_ranges,k)
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
