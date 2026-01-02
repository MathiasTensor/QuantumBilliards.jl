"""
Real plane wave basis with symmetry support.

# Quadrant and Parity Pattern Logic

The basis uses real plane waves of the form: f(x,y) = F_x(k·x) * F_y(k·y)
where F_x and F_y are either cos or sin, determined by parity patterns.

Quadrant ordering: (+x,+y), (+x,-y), (-x,+y), (-x,-y)
Parity encoding: +1 = cos, -1 = sin

# Symmetry Structure
Each symmetry in the `symmetries` vector has a corresponding quantum number in `sym_qnumbers` at the same index.
The order for multiple symmetries is: [YAxisReflection, XYAxisReflection, XAxisReflection]
with corresponding quantum numbers: [sym_x, sym_x*sym_y, sym_y]

# Symmetry Cases
- No symmetries: All 4 quadrants → 4 patterns: (cos,cos), (cos,sin), (sin,cos), (sin,sin)
- X-axis reflection (sym_y): 2 quadrants with fixed y-parity
  - sym_y = +1: (cos,cos), (sin,cos) → upper half-plane (y>0)
  - sym_y = -1: (cos,sin), (sin,sin) → lower half-plane (y<0)
- Y-axis reflection (sym_x): 2 quadrants with fixed x-parity
  - sym_x = +1: (cos,cos), (cos,sin) → right half-plane (x>0)
  - sym_x = -1: (sin,cos), (sin,sin) → left half-plane (x<0)
- XY-axis reflection (both sym_x and sym_y): 1 quadrant
  - (sym_y, sym_x) = (+1, +1): (cos,cos) → quadrant I
  - (sym_y, sym_x) = (+1, -1): (sin,cos) → quadrant II
  - (sym_y, sym_x) = (-1, +1): (cos,sin) → quadrant IV
  - (sym_y, sym_x) = (-1, -1): (sin,sin) → quadrant III
"""
struct RealPlaneWaves{T,Sa} <: AbsBasis where {T<:Real, Sa<:AbsSampler}
    dim::Int64
    symmetries::Union{Vector{BilliardGeometry.AbsReflection}, Nothing}
    sym_qnumbers::Union{Vector{T}, Nothing}
    angle_arc::T
    angle_shift::T
    angles::Vector{T}
    parity_x::Vector{Int64}
    parity_y::Vector{Int64}
    sampler::Sa
end

"""
     parity_pattern(symmetries, sym_qnumbers)

Helper function to determine the parity pattern (cos/sin selection) from symmetries and quantum numbers.

Each symmetry has a corresponding quantum number. The function processes these to determine
which cos/sin combinations to use for plane waves.

# Arguments
- `symmetries::Union{Vector{BilliardGeometry.AbsReflection}, Nothing}`: Reflection symmetries
- `sym_qnumbers::Union{Vector{T}, Nothing}`: Quantum numbers for each symmetry (same length as symmetries)

# Returns
- `(parity_x, parity_y)::Tuple{Vector{Int},Vector{Int}}`: Parity patterns where
  1 means use cos, -1 means use sin for that direction
"""
@inline function parity_pattern(::Nothing, ::Nothing)
    # No symmetries: use all four quadrants in order (+x,+y), (+x,-y), (-x,+y), (-x,-y)
    return Int[1, 1, -1, -1], Int[1, -1, 1, -1]
end

@inline function parity_pattern(symmetries::Vector{BG}, 
                                sym_qnumbers::Vector{T}) where {T<:Real, BG<:BilliardGeometry.AbsReflection}
    # Check which symmetries are present
    has_x = any(s -> s isa BilliardGeometry.XAxisReflection, symmetries)
    has_y = any(s -> s isa BilliardGeometry.YAxisReflection, symmetries)
    has_xy = any(s -> s isa BilliardGeometry.XYAxisReflection, symmetries)
    
    if has_xy
        # XY-axis reflection: single quadrant
        # Find the XY quantum number (should be the product sym_x * sym_y)
        xy_idx = findfirst(s -> s isa BilliardGeometry.XYAxisReflection, symmetries)
        x_idx = findfirst(s -> s isa BilliardGeometry.YAxisReflection, symmetries)
        y_idx = findfirst(s -> s isa BilliardGeometry.XAxisReflection, symmetries)
        
        x_par = Int(sym_qnumbers[x_idx])
        y_par = Int(sym_qnumbers[y_idx])
        
        return Int[x_par], Int[y_par]
    elseif has_x && !has_y
        # Only X-axis reflection: 2 quadrants with fixed y-parity
        x_idx = findfirst(s -> s isa BilliardGeometry.XAxisReflection, symmetries)
        y_par = Int(sym_qnumbers[x_idx])
        return Int[1, -1], Int[y_par, y_par]
    elseif has_y && !has_x
        # Only Y-axis reflection: 2 quadrants with fixed x-parity
        y_idx = findfirst(s -> s isa BilliardGeometry.YAxisReflection, symmetries)
        x_par = Int(sym_qnumbers[y_idx])
        return Int[x_par, x_par], Int[1, -1]
    else
        # Fallback to no symmetries
        return parity_pattern(nothing, nothing)
    end
end

"""
    infer_quantum_numbers(symmetries)

Infer quantum numbers from symmetry list. This is a helper for backward compatibility.
Returns a vector with one quantum number per symmetry (all default to +1).

# Arguments
- `symmetries::Vector{BilliardGeometry.AbsReflection}`: List of reflection symmetries

# Returns
- Quantum numbers vector (same length as symmetries, all default to +1)
"""
function infer_quantum_numbers(symmetries::Vector{BG}) where {BG<:BilliardGeometry.AbsReflection}
    isempty(symmetries) && return nothing
    # Default all quantum numbers to +1 (even parity)
    return ones(Float64, length(symmetries))
end

@inline infer_quantum_numbers(::Nothing) = nothing

"""
    RealPlaneWaves(dim, symmetries, sym_qnumbers; angle_arc=π, angle_shift=0.0, sampler=LinearNodes())

Constructor for RealPlaneWaves with symmetries and quantum numbers.

# Arguments
- `dim::Int`: Number of distinct angles to sample
- `symmetries::Union{Vector{<:BilliardGeometry.AbsReflection}, Nothing}`: Reflection symmetries
- `sym_qnumbers::Union{Vector{T}, Nothing}`: Quantum numbers for each symmetry (must be same length)
- `angle_arc::Real`: Angular range to sample (default π)
- `angle_shift::Real`: Angular offset (default 0.0)
- `sampler::AbsSampler`: Sampling strategy for angles
"""
function RealPlaneWaves(dim::Int, 
                       symmetries::Union{Vector{BG}, Nothing}, 
                       sym_qnumbers::Union{Vector{T}, Nothing}; 
                       angle_arc=π, angle_shift=0.0, 
                       sampler=LinearNodes()) where {T<:Real, BG<:BilliardGeometry.AbsReflection}
    # Validate that symmetries and sym_qnumbers have matching lengths
    if !isnothing(symmetries) && !isnothing(sym_qnumbers)
        @assert length(symmetries) == length(sym_qnumbers) "symmetries and sym_qnumbers must have the same length"
    end
    
    # Get parity pattern from symmetries and quantum numbers
    par_x, par_y = parity_pattern(symmetries, sym_qnumbers)
    pl = length(par_x)
    eff_dim = dim * pl
    
    # Sample angles from the sampler
    t, dt = sample_points(sampler, dim)
    
    # Preallocate and fill arrays more efficiently
    angles = Vector{eltype(t)}(undef, eff_dim)
    parity_x = Vector{Int}(undef, eff_dim)
    parity_y = Vector{Int}(undef, eff_dim)
    
    # Fill arrays using vectorized operations
    @inbounds for i in 1:dim
        angle = t[i] * angle_arc + angle_shift
        base_idx = (i-1) * pl
        for j in 1:pl
            idx = base_idx + j
            angles[idx] = angle
            parity_x[idx] = par_x[j]
            parity_y[idx] = par_y[j]
        end
    end
    
    Sa = typeof(sampler)
    
    return RealPlaneWaves{eltype(angles), Sa}(eff_dim, symmetries, sym_qnumbers, 
                                              angle_arc, angle_shift, angles, 
                                              parity_x, parity_y, sampler)
end

"""
    RealPlaneWaves(dim; sym_x=nothing, sym_y=nothing, angle_arc=nothing, angle_shift=nothing, sampler=LinearNodes())

Main constructor for RealPlaneWaves using quantum number specification.

# Arguments
- `dim::Int`: Number of distinct angles to sample
- `sym_x::Union{Int, Nothing}`: Quantum number for y-axis reflection (±1 for even/odd, nothing for no symmetry)
- `sym_y::Union{Int, Nothing}`: Quantum number for x-axis reflection (±1 for even/odd, nothing for no symmetry)
- `angle_arc::Union{Real, Nothing}`: Angular range to sample (default: auto-adjusted based on symmetries)
- `angle_shift::Union{Real, Nothing}`: Angular offset (default: auto-adjusted based on symmetries)
- `sampler::AbsSampler`: Sampling strategy for angles

# Details
The quantum numbers determine which quadrants and cos/sin patterns to use:
- `sym_x = nothing, sym_y = nothing`: All 4 quadrants, 4 combinations (arc=π, shift=0)
- `sym_x = nothing, sym_y = ±1`: 2 quadrants (x-axis reflection) (arc=π, shift=0)
- `sym_x = ±1, sym_y = nothing`: 2 quadrants (y-axis reflection) (arc=π, shift=-π/2)
- `sym_x = ±1, sym_y = ±1`: 1 quadrant (xy-axis reflection) (arc=π/2, shift=0)

When both symmetries are present, the symmetry order is: [YAxisReflection, XYAxisReflection, XAxisReflection]
with quantum numbers: [sym_x, sym_x*sym_y, sym_y]
"""
function RealPlaneWaves(dim::Int; sym_x::Union{Int,Nothing}=nothing, sym_y::Union{Int,Nothing}=nothing,
                       angle_arc::Union{Real,Nothing}=nothing, angle_shift::Union{Real,Nothing}=nothing, 
                       sampler=LinearNodes())
    # Build symmetries vector and quantum numbers based on sym_x and sym_y
    # Automatically adjust angle_arc and angle_shift based on symmetries if not provided
    
    if isnothing(sym_x) && isnothing(sym_y)
        # No symmetries - fast path
        symmetries = nothing
        sym_qnumbers = nothing
        arc = isnothing(angle_arc) ? π : angle_arc
        shift = isnothing(angle_shift) ? 0.0 : angle_shift
        
    elseif !isnothing(sym_x) && !isnothing(sym_y)
        # Both symmetries: YAxisReflection, XYAxisReflection, XAxisReflection (in that order)
        # Single quadrant → arc = π/2
        symmetries = BilliardGeometry.AbsReflection[
            BilliardGeometry.YAxisReflection(),
            BilliardGeometry.XYAxisReflection(),
            BilliardGeometry.XAxisReflection()
        ]
        # Quantum numbers: [sym_x, sym_x*sym_y, sym_y]
        sym_qnumbers = Float64[Float64(sym_x), Float64(sym_x * sym_y), Float64(sym_y)]
        arc = isnothing(angle_arc) ? π/2 : angle_arc
        shift = isnothing(angle_shift) ? 0.0 : angle_shift
        
    elseif !isnothing(sym_y)
        # Only x-axis reflection (reflects about x-axis, constrains y-parity)
        # 2 quadrants (upper or lower half-plane) → arc = π, shift = 0
        symmetries = BilliardGeometry.AbsReflection[BilliardGeometry.XAxisReflection()]
        sym_qnumbers = Float64[Float64(sym_y)]
        arc = isnothing(angle_arc) ? π : angle_arc
        shift = isnothing(angle_shift) ? 0.0 : angle_shift
        
    else  # !isnothing(sym_x)
        # Only y-axis reflection (reflects about y-axis, constrains x-parity)
        # 2 quadrants (right or left half-plane) → arc = π, shift = -π/2
        symmetries = BilliardGeometry.AbsReflection[BilliardGeometry.YAxisReflection()]
        sym_qnumbers = Float64[Float64(sym_x)]
        arc = isnothing(angle_arc) ? π : angle_arc
        shift = isnothing(angle_shift) ? -π/2 : angle_shift
    end
    
    # Call the main constructor
    return RealPlaneWaves(dim, symmetries, sym_qnumbers; 
                         angle_arc=arc, angle_shift=shift, sampler=sampler)
end

"""
    RealPlaneWaves(dim, symmetries; angle_arc=π, angle_shift=0.0, sampler=LinearNodes())

Constructor for RealPlaneWaves with symmetries but no explicit quantum numbers.
Infers default quantum numbers from symmetry types (all +1).
"""
function RealPlaneWaves(dim::Int, 
                       symmetries::Union{Vector{BG}, Nothing}; 
                       angle_arc=π, angle_shift=0.0, 
                       sampler=LinearNodes()) where {BG<:BilliardGeometry.AbsReflection}
    sym_qnumbers = infer_quantum_numbers(symmetries)
    return RealPlaneWaves(dim, symmetries, sym_qnumbers; 
                         angle_arc=angle_arc, angle_shift=angle_shift, sampler=sampler)
end

"""
    resize_basis(basis::RealPlaneWaves, billiard::AbsBilliard, dim::Int, k)

Resize the basis to a new dimension while preserving symmetries, quantum numbers, and sampling parameters.
"""
@inline function resize_basis(basis::RealPlaneWaves, billiard::AbsBilliard, dim::Int, k)
    return RealPlaneWaves(dim, basis.symmetries, basis.sym_qnumbers; 
                         angle_arc=basis.angle_arc, 
                         angle_shift=basis.angle_shift, 
                         sampler=basis.sampler)
end

# Helper functions for cos/sin pattern
# parity = 1 → cos, parity = -1 → sin
@inline _cos(arg) = cos(arg)
@inline _sin(arg) = sin(arg)
@inline _rpw_fun(par::Int) = par == 1 ? _cos : _sin
@inline _drpw_fun(par::Int) = par == 1 ? (x -> -sin(x)) : _cos


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
    @use_threads multithreading=multithreaded for c in 1:N
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
    @use_threads multithreading=multithreaded for c in 1:N
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
    @use_threads multithreading=multithreaded for c in 1:N
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
    @use_threads multithreading=multithreaded for c in 1:N
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