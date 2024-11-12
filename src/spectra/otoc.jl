using Bessels, LinearAlgebra

include("../billiards/boundarypoints.jl")
include("../states/wavefunctions.jl")



"""
    wavefunction_multi(ks::Vector{T}, vec_bdPoints::Vector{BoundaryPoints{T}}, billiard::Bi; b::Float64=5.0, inside_only::Bool=true) where {Bi<:AbsBilliard,T<:Real}

Constructs a sequence of 2D wavefunctions as matrices over the same sized grid for easier computation of matrix elements. The matrices are constructed via the boundary integral.

# Arguments
- `ks`: Vector of eigenvalues.
- `vec_bdPoints`: Vector of `BoundaryPoints` objects, one for each eigenvalues.
- `billiard`: The billiard geometry.
- `vec_us::Vector{Vector}`: Vector of the boundary functions
- `b::Float64=5.0`: (Optional), Point scaling factor. Default is 5.0.
- `inside_only::Bool=true`: (Optional), Whether to only compute wavefunctions inside the billiard. Default is true.

# Returns
- `Psi2ds::Vector{Matrix{T}}`: Vector of 2D wavefunction matrices constructed on the same grid.
- `x_grid::Vector{T}`: Vector of x-coordinates for the grid.
- `y_grid::Vector{T}`: Vector of y-coordinates for the grid.
"""
function wavefunction_multi(ks::Vector{T}, vec_us::Vector{T}, vec_bdPoints::Vector{BoundaryPoints{T}}, billiard::Bi; b::Float64=5.0, inside_only::Bool=true) where {Bi<:AbsBilliard,T<:Real}
    k_max = maximum(ks) # for most resolution
    type = eltype(k_max)
    L = billiard.length
    xlim,ylim = boundary_limits(billiard.full_boundary; grd=max(1000,round(Int, k_max*L*b/(2*pi))))
    dx = xlim[2] - xlim[1]
    dy = ylim[2] - ylim[1]
    nx = max(round(Int, k*dx*b/(2*pi)), 512)
    ny = max(round(Int, k*dy*b/(2*pi)), 512)
    x_grid::Vector{type} = collect(type,range(xlim... , nx))
    y_grid::Vector{type} = collect(type,range(ylim... , ny))
    sz = length(x_grid)*length(y_grid)
    pts = collect(SVector(x,y) for y in y_grid for x in x_grid)
    if inside_only
        pts_mask = points_in_billiard_polygon(pts, billiard, round(Int, sqrt(sz)); fundamental_domain=inside_only)
        pts = pts[pts_mask]
    end
    # wavefunction via boundary integral Ψ = 1/4∮Yₒ(k|q-qₛ|)u(s)ds
    function ϕ(x,y,k,bdPoints::BoundaryPoints, us::Vector)
        target_point = SVector(x, y)
        distances = norm.(target_point .- bdPoints.xy)          
        weighted_bessel_values = Bessels.bessely0.(k * distances) .* us .* bdPoints.ds  
        return sum(weighted_bessel_values) / 4                  
    end
    Psi2ds = Vector{Matrix{type}}(undef, length(ks))
    Threads.@threads for i in eachindex(ks) 
        k = ks[i]
        bdPoints = vec_bdPoints[i]
        us = vec_us[i]
        Psi_flat = [ϕ(x,y,k,bdPoints,us) for y in y_grid for x in x_grid]  # Flattened vector
        Psi_flat .= ifelse.(pts_mask, Psi_flat, zero(type))  # Zero out points outside billiard
        Psi2ds[i] = reshape(Psi_flat, ny, nx)  # Reshape back to 2D matrix
    end
    return Psi2ds, x_grid, y_grid
end

"""
    construct_wavefunctions_high_level(filename::String, billiard::Bi; b::Float64=5.0, inside_only::Bool=true) where {Bi<:AbsBilliard}

High-level wrapper for the `wavefunction_multi` that constructs the wavefunctions from the saved vector of boundary points and params `Vector{BoundaryPoints}` and vector of boundary functions `Vector{Vector}`.
"""
function construct_wavefunctions_high_level(filename::String, billiard::Bi; b::Float64=5.0, inside_only::Bool=true) where {Bi<:AbsBilliard}
    ks, vec_bd_points, vec_us = read_BoundaryPoints(filename)
    return wavefunction_multi(ks, vec_bd_points, vec_us, billiard; b=b, inside_only=inside_only)
end

