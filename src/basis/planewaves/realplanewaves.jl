
struct RealPlaneWaves{T,Sa} <: AbsBasis where  {T<:Real, Sa<:AbsSampler}
    #cs::PolarCS{T} #not fully implemented
    dim::Int64 #using concrete type
    symmetries::Union{Vector{Any},Nothing}
    angle_arc::T
    angle_shift::T
    angles::Vector{T}
    parity_x::Vector{Int64}
    parity_y::Vector{Int64}
    sampler::Sa
end

"""
     parity_pattern(symmetries::Vector{Any})

Helper function to determine the parity in the x and y direction wrt symmetry. This is neccesery since it determines the sign of the wavefunction in each quadrant.

# Arguments
- `symmetries::Vector{Any}`: Contains symmetry information to be transformed into quadrant rules.

# Returns
- `(parity_x,parity_y)::Tuple{Vector{Int},Vector{Int}}`: Quadrant rules in the x and y direction.
"""
function parity_pattern(symmetries)
    # Default parity vectors assuming no symmetries
    parity_x=[1,1,-1,-1]
    parity_y=[1,-1,1,-1]
    # Flags to track whether the axes are reflected
    x_reflected=false
    y_reflected=false
    xy_reflected=false
    # Loop over the symmetries
    if !isnothing(symmetries)
        for sym in symmetries
            if sym.axis==:y_axis
                # X-axis reflection: constrain parity_x
                parity_x=[sym.parity,sym.parity]
                x_reflected=true
            elseif sym.axis==:x_axis
                # Y-axis reflection: constrain parity_y
                parity_y=[sym.parity,sym.parity]
                y_reflected=true
            elseif sym.axis==:origin
                # XYReflection: constrain both axes to the same parity
                parity_x=[sym.parity[1]]
                parity_y=[sym.parity[2]]
                xy_reflected=true
                break  # Once XYReflection is applied, we must exit as there is nothing more to be done
            end
        end
    end
    if xy_reflected
        # XYReflection overrides any previous reflections, so lengths are already correct and we can terminate early
        return parity_x,parity_y
    end
    if x_reflected && !y_reflected # Ensure both parity_x and parity_y have the same length to avoid problems of taking lenght of them for effective dimension
        # If only XReflection is applied, adjust parity_y to match parity_x
        parity_y=[1,-1]
    elseif y_reflected && !x_reflected
        # If only YReflection is applied, adjust parity_x to match parity_y
        parity_x=[1,-1]
    end
    return parity_x,parity_y
end

function RealPlaneWaves(dim,symmetries;angle_arc=pi,angle_shift=0.0,sampler=LinearNodes())
    par_x,par_y=parity_pattern(symmetries)
    pl=length(par_x)
    eff_dim=dim*pl
    t,dt=sample_points(sampler, dim)
    angles=@. t*angle_arc + angle_shift
    angles=repeat(angles,inner=pl)
    par_x=repeat(par_x,outer=dim)
    par_y=repeat(par_y,outer=dim)
    return RealPlaneWaves(eff_dim,symmetries,angle_arc,angle_shift,angles,par_x,par_y,sampler)
end

function RealPlaneWaves(dim;angle_arc=pi,angle_shift=0.0,sampler=LinearNodes())
    symmetries=nothing
    par_x,par_y=parity_pattern(symmetries)
    pl=length(par_x)
    eff_dim=dim*pl
    t,dt=sample_points(sampler, dim)
    angles=@. t*angle_arc + angle_shift
    angles=repeat(angles,inner=pl)
    par_x=repeat(par_x,outer=dim)
    par_y=repeat(par_y,outer=dim)
    return RealPlaneWaves{eltype(angles),nothing,typeof(sampler)}(eff_dim,symmetries,angle_arc,angle_shift,angles,par_x,par_y,sampler)
end

"""
    resize_basis(basis::Ba,billiard::Bi,dim::Int,k) where {Ba<:RealPlaneWaves,Bi<:AbsBilliard}

This function resizes the `RealPlaneWaves` basis to a new dimension, if necessary. It checks whether the current dimension matches the desired dimension and returns the resized basis if they differ.
- If the dimensions match, the original basis is returned.
- If the dimensions differ, a new `RealPlaneWaves` object is created with the new dimension and the existing corner angle and coordinate system.

# Returns
- A `RealPlaneWaves` object with the updated dimension, or the original basis if no resizing is needed.
"""
function resize_basis(basis::Ba,billiard::Bi,dim::Int,k) where {Ba<:RealPlaneWaves,Bi<:AbsBilliard}
    return RealPlaneWaves(dim,basis.symmetries;angle_arc=basis.angle_arc,angle_shift=basis.angle_shift,sampler=basis.sampler)
end

@inline _cos(arg)=cos(arg)
@inline _sin(arg)=sin(arg)
@inline _rpw_fun(par::Int)=par==1 ? _cos : _sin
@inline _drpw_fun(par::Int)=par==1 ? (x->-sin(x)) : _cos

"""
    basis_fun(basis::RealPlaneWaves,i::Int,k::T,pts::AbstractArray) where {T<:Real}

Constructs the basis function Vector of the Real plane wave basis for column i.

# Arguments
- `basis::RealPlaneWaves`: Struct containing all the info to compute the matrix.
- `i::Int`: The column index of the matrix.
- `k::T`: Wavenumber to construct matrix at.
- `pts::AbstractArray`: Vector of xy points on the boundary.

# Returns
- `Vetor{T}`: Column of the basis matrix for index i.
"""
@inline function basis_fun(basis::RealPlaneWaves,i::Int,k::T,pts::AbstractArray) where {T<:Real}
    parx=basis.parity_x[i]
    pary=basis.parity_y[i]
    vx=cos(basis.angles[i])
    vy=sin(basis.angles[i])
    fx=_rpw_fun(parx)
    fy=_rpw_fun(pary)
    M=length(pts)
    out=Vector{T}(undef,M)
    @inbounds @simd for j=1:M
        x=pts[j][1]
        y=pts[j][2]
        out[j]=fx(k*vx*x)*fy(k*vy*y)
    end
    return out
end

"""
    basis_fun(basis::RealPlaneWaves,i::Int,k::T,pts::AbstractArray) where {T<:Real}

Constructs the basis function Matrix of the Real plane wave basis for column i.

# Arguments
- `basis::RealPlaneWaves`: Struct containing all the info to compute the matrix.
- `indices::AbstractArray`: The column indexes of the matrix.
- `k::T`: Wavenumber to construct matrix at.
- `pts::AbstractArray`: Vector of xy points on the boundary.
- `multithreaded::Bool=true`: If the matrix construction per columns is multithreaded.

# Returns
- `Matrix{T}`: The full basis matrix.
"""
@inline function basis_fun(basis::RealPlaneWaves,indices::AbstractArray,k::T,pts::AbstractArray;multithreaded::Bool=true) where {T<:Real}
    M=length(pts)
    N=length(indices)
    B=Matrix{T}(undef,M,N)
    QuantumBilliards.@use_threads multithreading=multithreaded for c in 1:N
        idx=indices[c]
        parx=basis.parity_x[idx]
        pary=basis.parity_y[idx]
        vx=cos(basis.angles[idx])
        vy=sin(basis.angles[idx])
        fx=_rpw_fun(parx)
        fy=_rpw_fun(pary)
        col=@view B[:,c]
        @inbounds @simd for j=1:M
            x=pts[j][1]
            y=pts[j][2]
            col[j]=fx(k*vx*x)*fy(k*vy*y)
        end
    end
    return B
end

"""
    gradient(basis::RealPlaneWaves,indices::AbstractArray,k::T,pts::AbstractArray;multithreaded::Bool=true) where {T<:Real}

Constructs the gradient basis function vectors of the Real plane wave basis for column i.

# Arguments
- `basis::RealPlaneWaves`: Struct containing all the info to compute the matrix.
- `i::Int`: The column index of the matrix.
- `k::T`: Wavenumber to construct matrix at.
- `pts::AbstractArray`: Vector of xy points on the boundary.
- `multithreaded::Bool=true`: If the matrix construction per columns is multithreaded.

# Returns
- `(dx,dy)::Tuple{Vector{T},Vector{T}}`: The full gradient vectors for x and y directions for the index i.
"""
function gradient(basis::RealPlaneWaves,i::Int,k::T,pts::AbstractArray) where {T<:Real}
    parx=basis.parity_x[i]
    pary=basis.parity_y[i]
    vx=cos(basis.angles[i])
    vy=sin(basis.angles[i])
    fx=_rpw_fun(parx)
    fy=_rpw_fun(pary)
    dfx=_drpw_fun(parx)
    dfy=_drpw_fun(pary)
    M=length(pts)
    dx=Vector{T}(undef,M)
    dy=Vector{T}(undef,M)
    @inbounds @simd for j=1:M
        x=pts[j][1]
        y=pts[j][2]
        ax=k*vx*x
        ay=k*vy*y
        bx=fx(ax)
        by=fy(ay)
        dx[j]=k*vx*dfx(ax)*by
        dy[j]=bx*k*vy*dfy(ay)
    end
    return dx,dy
end

"""
    gradient(basis::RealPlaneWaves,indices::AbstractArray,k::T,pts::AbstractArray;multithreaded::Bool=true) where {T<:Real}

Constructs the gradient basis function matrices of the Real plane wave basis.

# Arguments
- `basis::RealPlaneWaves`: Struct containing all the info to compute the matrix.
- `indices::AbstractArray`: The column indexes of the matrix.
- `k::T`: Wavenumber to construct matrix at.
- `pts::AbstractArray`: Vector of xy points on the boundary.
- `multithreaded::Bool=true`: If the matrix construction per columns is multithreaded.

# Returns
- `(dB_dx,dB_dy)::Tuple{Matrix{T},Matrix{T}}`: The full gradient matrices for x and y directions.
"""
function gradient(basis::RealPlaneWaves,indices::AbstractArray,k::T,pts::AbstractArray;multithreaded::Bool=true) where {T<:Real}
    M=length(pts); N=length(indices)
    dBdx=Matrix{T}(undef,M,N)
    dBdy=Matrix{T}(undef,M,N)
    QuantumBilliards.@use_threads multithreading=multithreaded for c in 1:N
        idx=indices[c]
        parx=basis.parity_x[idx]
        pary=basis.parity_y[idx]
        vx=cos(basis.angles[idx])
        vy=sin(basis.angles[idx])
        fx=_rpw_fun(parx)
        fy=_rpw_fun(pary)
        dfx=_drpw_fun(parx)
        dfy=_drpw_fun(pary)
        cx=@view dBdx[:,c]
        cy=@view dBdy[:,c]
        @inbounds @simd for j=1:M
            x=pts[j][1]
            y=pts[j][2]
            ax=k*vx*x
            ay=k*vy*y
            bx=fx(ax)
            by=fy(ay)
            cx[j]=k*vx*dfx(ax)*by
            cy[j]=bx*k*vy*dfy(ay)
        end
    end
    return dBdx,dBdy
end

"""
    basis_and_gradient(basis::RealPlaneWaves,i::Int,k::T,pts::AbstractArray) where {T<:Real}

Constructs the gradient basis function vectors of the Real plane wave basis for column i along with the basis vector.

# Arguments
- `basis::RealPlaneWaves`: Struct containing all the info to compute the matrix.
- `i::Int`: The column index of the matrix.
- `k::T`: Wavenumber to construct matrix at.
- `pts::AbstractArray`: Vector of xy points on the boundary.
- `multithreaded::Bool=true`: If the matrix construction per columns is multithreaded.

# Returns
- `(bf,dx,dy)::Tuple{Vector{T},Vector{T},Vector{T}}`: The full gradient vectors for x and y directions and basis function vector for the index i.
"""
function basis_and_gradient(basis::RealPlaneWaves,i::Int,k::T,pts::AbstractArray) where {T<:Real}
    parx=basis.parity_x[i]
    pary=basis.parity_y[i]
    vx=cos(basis.angles[i])
    vy=sin(basis.angles[i])
    fx=_rpw_fun(parx)
    fy=_rpw_fun(pary)
    dfx=_drpw_fun(parx)
    dfy=_drpw_fun(pary)
    M=length(pts)
    bf=Vector{T}(undef,M)
    dx=Vector{T}(undef,M)
    dy=Vector{T}(undef,M)
    @inbounds @simd for j=1:M
        x=pts[j][1]
        y=pts[j][2]
        ax=k*vx*x
        ay=k*vy*y
        bx=fx(ax)
        by=fy(ay)
        bf[j]=bx*by
        dx[j]=k*vx*dfx(ax)*by
        dy[j]=bx*k*vy*dfy(ay)
    end
    return bf,dx,dy
end

"""
    basis_and_gradient(basis::RealPlaneWaves,indices::AbstractArray,k::T,pts::AbstractArray;multithreaded::Bool=true) where {T<:Real}

Constructs the gradient basis function matrices and the gradient matrices of the Real plane wave basis.

# Arguments
- `basis::RealPlaneWaves`: Struct containing all the info to compute the matrix.
- `indices::AbstractArray`: The column indexes of the matrix.
- `k::T`: Wavenumber to construct matrix at.
- `pts::AbstractArray`: Vector of xy points on the boundary.
- `multithreaded::Bool=true`: If the matrix construction per columns is multithreaded.

# Returns
- `(B,dB_dx,dB_dy)::Tuple{Matrix{T},Matrix{T}}`: The full gradient matrices for x and y directionsand the basis matrix.
"""
function basis_and_gradient(basis::RealPlaneWaves,indices::AbstractArray,k::T,pts::AbstractArray;multithreaded::Bool=true) where {T<:Real}
    M=length(pts); N=length(indices)
    B=Matrix{T}(undef,M,N)
    dBdx=Matrix{T}(undef,M,N)
    dBdy=Matrix{T}(undef,M,N)
    QuantumBilliards.@use_threads multithreading=multithreaded for c in 1:N
        idx=indices[c]
        parx=basis.parity_x[idx]
        pary=basis.parity_y[idx]
        vx=cos(basis.angles[idx])
        vy=sin(basis.angles[idx])
        fx=_rpw_fun(parx)
        fy=_rpw_fun(pary)
        dfx=_drpw_fun(parx)
        dfy=_drpw_fun(pary)
        col=@view B[:,c]
        cx=@view dBdx[:,c]
        cy=@view dBdy[:,c]
        @inbounds @simd for j=1:M
            x=pts[j][1]
            y=pts[j][2]
            ax=k*vx*x
            ay=k*vy*y
            bx=fx(ax)
            by=fy(ay)
            col[j]=bx*by
            cx[j]=k*vx*dfx(ax)*by
            cy[j]=bx*k*vy*dfy(ay)
        end
    end
    return B,dBdx,dBdy
end

"""
    dk_fun(basis::RealPlaneWaves,i::Int,k::T,pts::AbstractArray) where {T<:Real}

Constructs the k-gradient of the basis matrix wrt k for column i.

# Arguments
- `basis::RealPlaneWaves`: Struct containing all the info to compute the matrix.
- `i::Int`: The column index of the matrix.
- `k::T`: Wavenumber to construct matrix at.
- `pts::AbstractArray`: Vector of xy points on the boundary.
- `multithreaded::Bool=true`: If the matrix construction per columns is multithreaded.

# Returns
- `dk::Vector{T}`: Vector representing the column of dB/dk for the index i.
"""
@inline function dk_fun(basis::RealPlaneWaves,i::Int,k::T,pts::AbstractArray) where {T<:Real}
    parx=basis.parity_x[i]
    pary=basis.parity_y[i]
    vx=cos(basis.angles[i])
    vy=sin(basis.angles[i])
    fx=_rpw_fun(parx)
    fy=_rpw_fun(pary)
    dfx=_drpw_fun(parx)
    dfy=_drpw_fun(pary)
    M=length(pts)
    dk=Vector{T}(undef,M)
    @inbounds @simd for j=1:M
        x=pts[j][1]
        y=pts[j][2]
        ax=k*vx*x
        ay=k*vy*y
        bx=fx(ax)
        by=fy(ay)
        dk[j]=vx*x*dfx(ax)*by + bx*vy*y*dfy(ay)
    end
    return dk
end
    
"""
    dk_fun(basis::RealPlaneWaves,indices::AbstractArray,k::T,pts::AbstractArray;multithreaded::Bool=true) where {T<:Real}

Constructs the k-gradient of the basis matrix.

# Arguments
- `basis::RealPlaneWaves`: Struct containing all the info to compute the matrix.
- `indices::AbstractArray`: The column indexes of the matrix.
- `k::T`: Wavenumber to construct matrix at.
- `pts::AbstractArray`: Vector of xy points on the boundary.
- `multithreaded::Bool=true`: If the matrix construction per columns is multithreaded.

# Returns
- `dB_dk::Matrix{T}`: matrix representing dB/dk.
"""
@inline function dk_fun(basis::RealPlaneWaves,indices::AbstractArray,k::T,pts::AbstractArray;multithreaded::Bool=true) where {T<:Real}
    M=length(pts)
    N=length(indices)
    dBdk=Matrix{T}(undef,M,N)
    QuantumBilliards.@use_threads multithreading=multithreaded for c in 1:N
        idx=indices[c]
        parx=basis.parity_x[idx]
        pary=basis.parity_y[idx]
        vx=cos(basis.angles[idx])
        vy=sin(basis.angles[idx])
        fx=_rpw_fun(parx)
        fy=_rpw_fun(pary)
        dfx=_drpw_fun(parx)
        dfy=_drpw_fun(pary)
        col=@view dBdk[:,c]
        @inbounds @simd for j=1:M
            x=pts[j][1]
            y=pts[j][2]
            ax=k*vx*x
            ay=k*vy*y
            bx=fx(ax)
            by=fy(ay)
            col[j]=vx*x*dfx(ax)*by + bx*vy*y*dfy(ay)
        end
    end
    return dBdk
end
