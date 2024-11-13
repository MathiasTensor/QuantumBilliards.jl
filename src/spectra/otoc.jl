using Bessels, LinearAlgebra, ProgressMeter

include("../billiards/boundarypoints.jl")
include("../states/wavefunctions.jl")



"""
    wavefunction_multi(ks::Vector{T}, vec_us::Vector{Vector{T}}, vec_bdPoints::Vector{BoundaryPoints{T}}, billiard::Bi; b::Float64=5.0, inside_only::Bool=true) where {Bi<:AbsBilliard,T<:Real}

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
function wavefunction_multi(ks::Vector{T}, vec_us::Vector{Vector{T}}, vec_bdPoints::Vector{BoundaryPoints{T}}, billiard::Bi; b::Float64=5.0, inside_only::Bool=true) where {Bi<:AbsBilliard,T<:Real}
    k_max = maximum(ks) # for most resolution
    type = eltype(k_max)
    L = billiard.length
    xlim,ylim = boundary_limits(billiard.full_boundary; grd=max(1000,round(Int, k_max*L*b/(2*pi))))
    dx = xlim[2] - xlim[1]
    dy = ylim[2] - ylim[1]
    nx = max(round(Int, k_max*dx*b/(2*pi)), 512)
    ny = max(round(Int, k_max*dy*b/(2*pi)), 512)
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
        distances = norm.(Ref(target_point) .- bdPoints.xy)      
        weighted_bessel_values = Bessels.bessely0.(k * distances) .* us .* bdPoints.ds  
        return sum(weighted_bessel_values) / 4                  
    end
    Psi2ds = Vector{Matrix{type}}(undef, length(ks))
    progress = Progress(length(ks), desc = "Constructing wavefunction matrices...")
    Threads.@threads for i in eachindex(ks) 
        k = ks[i]
        bdPoints = vec_bdPoints[i]
        us = vec_us[i]
        Psi_flat = [ϕ(x,y,k,bdPoints,us) for y in y_grid for x in x_grid]  # Flattened vector
        Psi_flat .= ifelse.(pts_mask, Psi_flat, zero(type))  # Zero out points outside billiard
        Psi2ds[i] = reshape(Psi_flat, ny, nx)  # Reshape back to 2D matrix
        next!(progress)
    end
    return Psi2ds, x_grid, y_grid
end

"""
    wavefunction_multi_with_husimi(ks::Vector{T}, vec_us::Vector{Vector{T}}, vec_bdPoints::Vector{BoundaryPoints{T}}, billiard::Bi; b::Float64=5.0, inside_only::Bool=true) where {Bi<:AbsBilliard,T<:Real}

Constructs a sequence of 2D wavefunctions as matrices over the same sized grid for easier computation of matrix elements. The matrices are constructed via the boundary integral. Additionally also constructs the husimi functions.

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
- `Hs_list::Vector{Matrix{T}}`: Vector of 2D husimi function matrices.
- `ps_list::Vector{Vector{T}}`: Vector of ps grids for the husimi matrices.
- `qs_list::Vector{Vector{T}}`: Vector of qs grids for the husimi matrices.
"""
function wavefunction_multi_with_husimi(ks::Vector{T}, vec_us::Vector{Vector{T}}, vec_bdPoints::Vector{BoundaryPoints{T}}, billiard::Bi; b::Float64=5.0, inside_only::Bool=true) where {Bi<:AbsBilliard,T<:Real}
    k_max = maximum(ks) # for most resolution
    type = eltype(k_max)
    L = billiard.length
    xlim,ylim = boundary_limits(billiard.full_boundary; grd=max(1000,round(Int, k_max*L*b/(2*pi))))
    dx = xlim[2] - xlim[1]
    dy = ylim[2] - ylim[1]
    nx = max(round(Int, k_max*dx*b/(2*pi)), 512)
    ny = max(round(Int, k_max*dy*b/(2*pi)), 512)
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
        distances = norm.(Ref(target_point) .- bdPoints.xy)      
        weighted_bessel_values = Bessels.bessely0.(k * distances) .* us .* bdPoints.ds  
        return sum(weighted_bessel_values) / 4                  
    end
    Psi2ds = Vector{Matrix{type}}(undef, length(ks))
    progress = Progress(length(ks), desc = "Constructing wavefunction matrices...")
    Threads.@threads for i in eachindex(ks) 
        k = ks[i]
        bdPoints = vec_bdPoints[i]
        us = vec_us[i]
        Psi_flat = [ϕ(x,y,k,bdPoints,us) for y in y_grid for x in x_grid]  # Flattened vector
        Psi_flat .= ifelse.(pts_mask, Psi_flat, zero(type))  # Zero out points outside billiard
        Psi2ds[i] = reshape(Psi_flat, ny, nx)  # Reshape back to 2D matrix
        next!(progress)
    end
    # husimi
    vec_of_s_vals = [bdPoints.s for bdPoints in vec_bdPoints]
    Hs_list, ps_list, qs_list = husimi_functions_from_boundary_functions(ks, vec_us, vec_of_s_vals, billiard)
    return Psi2ds, x_grid, y_grid, Hs_list, ps_list, qs_list
end

"""
    plot_wavefunctions(ks::Vector, Psi2ds::Vector, x_grid::Vector, y_grid::Vector, billiard::Bi; b::Float64=5.0, width_ax::Integer=500, height_ax::Integer=500, max_cols::Integer=6) where {Bi<:AbsBilliard}

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

 # Returns
- `f::Figure`: A Figure object containing the grid of wavefunctions.
"""
function plot_wavefunctions(ks::Vector, Psi2ds::Vector, x_grid::Vector, y_grid::Vector, billiard::Bi; b::Float64=5.0, width_ax::Integer=500, height_ax::Integer=500, max_cols::Integer=6) where {Bi<:AbsBilliard}
    L = billiard.length
    xlim,ylim = boundary_limits(billiard.fundamental_boundary; grd=max(1000,round(Int, maximum(ks)*L*b/(2*pi))))
    f = Figure()
    row = 1
    col = 1
    for j in eachindex(ks)
        local ax = Axis(f[row,col], title="$(ks[j])", aspect=DataAspect(), width=width_ax, height=height_ax)
        hm = heatmap!(ax, x_grid, y_grid, Psi2ds[j], colormap=:balance, colorrange=(-maximum(Psi2ds[j]), maximum(Psi2ds[j])))
        plot_boundary!(ax, billiard, fundamental_domain=true, plot_normal=false)
        xlims!(ax, xlim)
        ylims!(ax, ylim)
        col += 1
        if col > max_cols
            row += 1
            col = 1
        end
    end
    resize_to_layout!(f)
    return f
end

"""
    plot_wavefunctions(ks::Vector, Psi2ds::Vector, x_grid::Vector, y_grid::Vector, billiard::Bi; b::Float64=5.0, width_ax::Integer=500, height_ax::Integer=500, max_cols::Integer=6) where {Bi<:AbsBilliard}

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

 # Returns
- `f::Figure`: A Figure object containing the grid of wavefunctions.
"""
function plot_wavefunctions_with_husimi(ks::Vector, Psi2ds::Vector, x_grid::Vector, y_grid::Vector, Hs_list::Vector, ps_list::Vector, qs_list::Vector, billiard::Bi; b::Float64=5.0, width_ax::Integer=500, height_ax::Integer=500, max_cols::Integer=6) where {Bi<:AbsBilliard}
    L = billiard.length
    xlim,ylim = boundary_limits(billiard.fundamental_boundary; grd=max(1000,round(Int, maximum(ks)*L*b/(2*pi))))
    f = Figure()
    row = 1
    col = 1
    for j in eachindex(ks)
        local ax = Axis(f[row,col][1,1], title="$(ks[j])", aspect=DataAspect(), width=width_ax, height=height_ax)
        local ax_h = Axis(f[row,col][1,2], width=width_ax, height=height_ax)
        hm = heatmap!(ax, x_grid, y_grid, Psi2ds[j], colormap=:balance, colorrange=(-maximum(Psi2ds[j]), maximum(Psi2ds[j])))
        plot_boundary!(ax, billiard, fundamental_domain=true, plot_normal=false)
        hm_h = heatmap!(ax_h, qs_list[i], ps_list[i], Hs_list[i]; colormap=Reverse(:gist_heat))
        xlims!(ax, xlim)
        ylims!(ax, ylim)
        col += 1
        if col > max_cols
            row += 1
            col = 1
        end
    end
    resize_to_layout!(f)
    return f
end




