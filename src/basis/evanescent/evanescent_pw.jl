using LinearAlgebra, CoordinateTransformations, Rotations, StaticArrays

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
max_i(k)=floor(Int,6*k^(1/3)-3)

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

function b(crv::Crv) where {Crv<:AbsRealCurve}
    #TODO
end

function epw(pts::AbstractArray,i::Int64,Ni::Ti,origin::SVector{2,T},k::T) where {T<:Real,Ti<:Integer}
    x=getindex.(pts,1).-origin[1]
    y=getindex.(pts,2).-origin[2]
    θi=2*pi*(i-0.5)/Ni
    si,ci=sincos(θi)
    ni=SVector(ci,si)
    αi=(3+i)/(2*k^(1/3))  # Evanescence parameter
    decay=exp.(-sinh.(αi.*(-ni[2].*x.+ni[1].*y)))
    phase=cosh.(αi.*(ni[1].*x.+ni[2].*y))
    osc=iseven(i) ? cos.(phase) : sin.(phase)
    return decay.*osc
end

function epw_dk(pts::AbstractArray,i::Int64,Ni::Ti,origin::SVector{2,T},k::T) where {T<:Real,Ti<:Integer}
    x=getindex.(pts,1).-origin[1]
    y=getindex.(pts,2).-origin[2]
    θi=2*pi*(i-0.5)/Ni
    si,ci=sincos(θi)
    ni=SVector(ci,si)
    αi=(3+i)/(2*k^(1/3))
    dαdk=-(3+i)/(6*k^(4/3))
    A=-ni[2].*x.+ni[1].*y # for sinh decay
    B=ni[1].*x.+ni[2].*y # for sin/cos oscillation
    αA=αi.*A
    αB=αi.*B
    sinh_αA,cosh_αA=sinhcosh.(αA)
    decay=@. exp(-sinh_αA)
    ddecay_dk=@. decay*(-cosh_αA*A*dαdk)
    cosh_αB=cosh.(αB)
    sinh_αB=sinh.(αB)
    if iseven(i)
        osc=cos.(cosh_αB)
        dosc_dk=@. -sin(cosh_αB)*sinh_αB*dαdk*B
    else
        osc=sin.(cosh_αB)
        dosc_dk=@. cos(cosh_αB)*sinh_αB*dαdk*B
    end
    return ddecay_dk.*osc.+decay.*dosc_dk
end

function epw_gradient(pts::AbstractArray,i::Int64,Ni::Ti,origin::SVector{2,T},k::T) where {T<:Real,Ti<:Integer}
    x=getindex.(pts,1).-origin[1]
    y=getindex.(pts,2).-origin[2]
    θi=2*pi*(i-0.5)/Ni
    si,ci=sincos(θi)
    ni=SVector(ci,si)
    αi=(3+i)/(2*k^(1/3))
    A=-ni[2].*x.+ni[1].*y  # decay argument
    B=ni[1].*x.+ni[2].*y  # oscillatory argument
    αA=αi.*A
    αB=αi.*B
    sinhcosh_vals=sinhcosh.(αA)
    sinh_αA=getindex.(sinhcosh_vals,1)
    cosh_αA=getindex.(sinhcosh_vals,2)
    decay=@. exp(-sinh_αA)
    ddecay_dA=@. -αi*cosh_αA*decay
    cosh_αB=cosh.(αB)
    sinh_αB=sinh.(αB)
    if iseven(i)
        osc=cos.(cosh_αB)
        dosc_dB=@. -αi*sin(cosh_αB)*sinh_αB
    else
        osc=sin.(cosh_αB)
        dosc_dB=@. αi*cos(cosh_αB)*sinh_αB
    end
    dA_dx,dA_dy=-ni[2],ni[1] # Compute spatial derivatives of A and B
    dB_dx,dB_dy=ni[1],ni[2]
    dx=ddecay_dA.*dA_dx.*osc.+decay.*dosc_dB.*dB_dx # Gradient components
    dy=ddecay_dA.*dA_dy.*osc.+decay.*dosc_dB.*dB_dy
    return dx,dy
end

function epw(pts::AbstractArray,i::Int64,Ni::Ti,origins::Vector{SVector{2,T}},k::T) where {T<:Real,Ti<:Integer}
    N=length(pts)
    M=length(origins)
    res=Matrix{Complex{T}}(undef,N,M) # pts x origins
    for j in eachindex(origins) 
        @inbounds res[:,j]=epw(pts,i,Ni,origins[j],k)
    end
    return sum(res,dims=2)[:] # for each row sum over all columns to get for each pt in pts all the different origin contributions. Converts Matrix (N,1) to a flat vector.
end

function epw_dk(pts::AbstractArray,i::Int64,Ni::Ti,origins::Vector{SVector{2,T}},k::T) where {T<:Real,Ti<:Integer}
    N=length(pts)
    M=length(origins)
    res=Matrix{Complex{T}}(undef,N,M)
    for j in eachindex(origins)
        @inbounds res[:,j]=epw_dk(pts,i,Ni,origins[j],k)
    end
    return sum(res,dims=2)[:]
end

function epw_gradient(pts::AbstractArray,i::Int64,Ni::Ti,origins::Vector{SVector{2,T}},k::T) where {T<:Real,Ti<:Integer}
    N=length(pts)
    M=length(origins)
    dx_mat=Matrix{Complex{T}}(undef,N,M)
    dy_mat=Matrix{Complex{T}}(undef,N,M)
    for j in eachindex(origins)
        dx,dy=epw_gradient(pts,i,Ni,origins[j],k)
        @inbounds dx_mat[:,j]=dx
        @inbounds dy_mat[:,j]=dy
    end
    dx=sum(dx_mat,dims=2)[:]
    dy=sum(dy_mat,dims=2)[:]
    return dx,dy
end

struct EvanescentPlaneWaves{T,Sy} <: AbsBasis where  {T<:Real,Sy<:Union{AbsSymmetry,Nothing}}
    cs::PolarCS{T}
    dim::Int64 
    origins::Vector{SVector{2,T}}
    symmetries::Union{Vector{Any},Nothing}
    shift_x::T
    shift_y::T
end

function EvanescentPlaneWaves(cs::PolarCS{T},dim::Int,origins::Vector{SVector{2,T}},symmetries::Union{Nothing,Vector{Any}}) where {T<:Real}
    EvanescentPlaneWaves{T,typeof(symmetries)}(cs,dim,origins,symmetries,zero(T),zero(T))
end

function EvanescentPlaneWaves(cs::PolarCS{T},dim::Int,origins::Vector{SVector{2,T}},symmetries::Union{Nothing,Vector{Any}},shift_x::T,shift_y::T) where {T<:Real}
    EvanescentPlaneWaves{T,typeof(symmetries)}(cs,dim,origins,symmetries,shift_x,shift_y)
end

function EvanescentPlaneWaves(billiard::Bi,origin::SVector{2,T},rot_angle::T;fundamental=false) where {Bi<:AbsBilliard,T<:Real}
    origins=get_origins_(billiard;fundamental=fundamental)
    shift_x=hasproperty(billiard,:x_axis) ? billiard.x_axis : T(0.0)
    shift_y=hasproperty(billiard,:y_axis) ? billiard.y_axis : T(0.0)
    return EvanescentPlaneWaves(PolarCS(origin,rot_angle),10,origins,nothing,shift_x,shift_y)
end

function EvanescentPlaneWaves(billiard::Bi,symmetries::Vector{Any},origin::SVector{2,T},rot_angle::T;fundamental=false) where {Bi<:AbsBilliard,T<:Real}
    origins=get_origins_(billiard;fundamental=fundamental)
    shift_x=hasproperty(billiard,:x_axis) ? billiard.x_axis : T(0.0)
    shift_y=hasproperty(billiard,:y_axis) ? billiard.y_axis : T(0.0)
    return EvanescentPlaneWaves(PolarCS(origin,rot_angle),10,origins,symmetries,shift_x,shift_y)
end

function EvanescentPlaneWaves(billiard::Bi,idxs::AbstractArray,origin::SVector{2,T},rot_angle::T;fundamental=false) where {Bi<:AbsBilliard,T<:Real}
    origins=get_origins_(billiard,idxs;fundamental=fundamental)
    shift_x=hasproperty(billiard,:x_axis) ? billiard.x_axis : T(0.0)
    shift_y=hasproperty(billiard,:y_axis) ? billiard.y_axis : T(0.0)
    return EvanescentPlaneWaves(PolarCS(origin,rot_angle),10,origins,nothing,shift_x,shift_y)
end

function EvanescentPlaneWaves(billiard::Bi,idxs::AbstractArray,symmetries::Vector{Any},origin::SVector{2,T},rot_angle::T;fundamental=false) where {Bi<:AbsBilliard,T<:Real}
    origins=get_origins_(billiard,idxs;fundamental=fundamental)
    shift_x=hasproperty(billiard,:x_axis) ? billiard.x_axis : T(0.0)
    shift_y=hasproperty(billiard,:y_axis) ? billiard.y_axis : T(0.0)
    return EvanescentPlaneWaves(PolarCS(origin,rot_angle),10,origins,symmetries,shift_x,shift_y)
end

function EvanescentPlaneWaves(billiard::Bi,idx::Ti,origin::SVector{2,T},rot_angle::T;fundamental=false) where {Bi<:AbsBilliard,Ti<:Integer,T<:Real}
    origins=get_origins_(billiard,idx;fundamental=fundamental)
    shift_x=hasproperty(billiard,:x_axis) ? billiard.x_axis : T(0.0)
    shift_y=hasproperty(billiard,:y_axis) ? billiard.y_axis : T(0.0)
    return EvanescentPlaneWaves(PolarCS(origin,rot_angle),10,origins,nothing,shift_x,shift_y)
end

function EvanescentPlaneWaves(billiard::Bi,idx::Ti,symmetries::Vector{Any},origin::SVector{2,T},rot_angle::T;fundamental=false) where {Bi<:AbsBilliard,Ti<:Integer,T<:Real}
    origins=get_origins_(billiard,idx;fundamental=fundamental)
    shift_x=hasproperty(billiard,:x_axis) ? billiard.x_axis : T(0.0)
    shift_y=hasproperty(billiard,:y_axis) ? billiard.y_axis : T(0.0)
    return EvanescentPlaneWaves(PolarCS(origin,rot_angle),10,origins,symmetries,shift_x,shift_y)
end

function get_origins_(billiard::Bi,idx::Ti;fundamental=false) where {Bi<:AbsBilliard,Ti<:Integer}
    boundary= fundamental ? billiard.fundamental_boundary : billiard.full_boundary
    elt=eltype(boundary[1].length)
    N=length(boundary)
    origins=Vector{SVector{2,elt}}()
    crv=boundary[idx]
    if crv isa LineSegment && boundary[mod1(idx-1,N)] isa LineSegment
        push!(origins,curve(crv,zero(elt)))
    end
    return origins
end

function get_origins_(billiard::Bi,idxs::AbstractArray;fundamental=false) where {Bi<:AbsBilliard}
    boundary= fundamental ? billiard.fundamental_boundary : billiard.full_boundary
    elt=eltype(boundary[1].length)
    N=length(boundary)
    @assert length(idxs)<=N "The number of idxs cannot be larger than the number of boundary segments. Check if fundamental kwarg is set correctly!"
    origins=Vector{SVector{2,elt}}()
    for idx in idxs 
        crv=boundary[idx]
        if crv isa LineSegment && boundary[mod1(idx-1,N)] isa LineSegment # is the other curve is virtual then usually the BCs are already satisfied there so no need to add. Also only add the corners so must be a line segment adjacent to produce a corner.
            push!(origins,curve(crv,zero(elt))) # starting corner 
        end
    end
    return origins
end

function get_origins_(billiard::Bi;fundamental=false) where {Bi<:AbsBilliard}
    boundary=fundamental ? billiard.fundamental_boundary : billiard.full_boundary
    return get_origins_(billiard,eachindex(boundary);fundamental=fundamental)
end

toFloat32(basis::EvanescentPlaneWaves)=EvanescentPlaneWaves(PolarCS(Float32.(basis.cs.origin),basis.cs.rot_angle),basis.dim,Float32.(basis.origins),basis.symmetries)

function resize_basis(basis::EvanescentPlaneWaves,billiard::Bi,dim::Int,k) where {Bi<:AbsBilliard}
    new_dim=max_i(k)
    if new_dim==basis.dim
        return basis
    else
        return EvanescentPlaneWaves(basis.cs,new_dim,basis.origins,basis.symmetries)
    end
end

###################################################################################
#### SINGLE INDEX CONSTRUCTION - SEPARATE SINCE WE SYMMETRIZE EPW AT THIS STEP ####
###################################################################################

@inline reflect_x_epw(p::SVector{2,T},shift_x::T) where {T<:Real} = SVector(2*shift_x-p[1],p[2])
@inline reflect_y_epw(p::SVector{2,T},shift_y::T) where {T<:Real} = SVector(p[1],2*shift_y-p[2])
@inline reflect_xy_epw(p::SVector{2,T},shift_x::T,shift_y::T) where {T<:Real} = SVector(2*shift_x-p[1],2*shift_y-p[2])

@inline function symmetrize_epw(f::F,basis::EvanescentPlaneWaves{T},i::Int,k::T,pts::Vector{SVector{2,T}}) where {F<:Function,T<:Real}
    syms=basis.symmetries
    isnothing(syms) && return f(basis,i,k,pts)  # No symmetry applied
    sym=syms[1]
    origin=basis.cs.origin
    fval=f(pts,i,basis.dim,basis.origins,k)
    if sym.axis==:y_axis # XReflection
        px=sym.parity
        reflected_pts_x=reflect_x_epw.(pts,Ref(basis.shift_x))
        return 0.5*(fval.+px.*f(reflected_pts_x,i,basis.dim,basis.origins,k))
    elseif sym.axis==:x_axis # YReflection
        py=sym.parity
        reflected_pts_y=reflect_y_epw.(pts,Ref(basis.shift_y))
        return 0.5*(fval.+py.*f(reflected_pts_y,i,basis.dim,basis.origins,k))
    elseif sym.axis==:origin # XYReflection
        px,py=sym.parity
        reflected_pts_x=reflect_x_epw.(pts,Ref(basis.shift_x))
        reflected_pts_y=reflect_y_epw.(pts,Ref(basis.shift_y))
        reflected_pts_xy=reflect_xy_epw.(pts,Ref(basis.shift_x),Ref(basis.shift_y))
        return 0.25*(fval.+px.*f(reflected_pts_x,i,basis.dim,basis.origins,k).+py.*f(reflected_pts_y,i,basis.dim,basis.origins,k).+(px*py).*f(reflected_pts_xy,i,basis.dim,basis.origins,k))
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
    return symmetrize_epw(epw_gradient,basis,i,k,pts)
end

function basis_and_gradient(basis::EvanescentPlaneWaves{T},i::Int,k::T,pts::AbstractArray) where {T<:Real}
    basis_vec=basis_fun(basis,i,k,pts)
    vec_dX,vec_dY=gradient(basis,i,k,pts)
    return basis_vec,vec_dX,vec_dY
end

##################################
#### MULTI INDEX CONSTRUCTION ####
##################################

function basis_fun(basis::EvanescentPlaneWaves{T},indices::AbstractArray,k::T,pts::AbstractArray;multithreaded::Bool=true) where {T<:Real}
    N=length(pts)
    M=length(indices)
    mat=zeros(T,N,M)
    @use_threads multithreading=multithreaded for i in eachindex(indices) 
        @inbounds mat[:,i] .= basis_fun(basis,i,k,pts)
    end
    return mat
end

function dk_fun(basis::EvanescentPlaneWaves{T},indices::AbstractArray,k::T,pts::AbstractArray;multithreaded::Bool=true) where {T<:Real}
    N=length(pts)
    M=length(indices)
    mat=zeros(T,N,M)
    @use_threads multithreading=multithreaded for i in eachindex(indices) 
        @inbounds mat[:,i] .= dk_fun(basis,i,k,pts)
    end
    return mat
end

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

function basis_and_gradient(basis::EvanescentPlaneWaves{T},indices::AbstractArray,k::T,pts::AbstractArray;multithreaded::Bool=true) where {T<:Real}
    mat=basis_fun(basis,indices,k,pts;multithreaded=multithreaded)
    mat_dX,mat_dY=gradient(basis,indices,k,pts;multithreaded=multithreaded)
    return mat,mat_dX,mat_dY
end
