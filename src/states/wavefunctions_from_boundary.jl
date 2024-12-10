using Bessels, LinearAlgebra, ProgressMeter

include("../billiards/boundarypoints.jl")
include("../states/wavefunctions.jl")

"""
    ϕ(x::T, y::T, k::T, bdPoints::BoundaryPoints, us::Vector)

Computes the wavefunction via the boundary integral Ψ = 1/4∮Yₒ(k|q-qₛ|)u(s)ds. For a specific `k` it needs the boundary discretization information encoded into the `BoundaryPoints` struct:

```julia
struct BoundaryPoints{T} <: (AbsPoints where T <: Real)
xy::Vector{SVector{2, T}} # boundary (x,y) pts
normal::Vector{SVector{2, T}} # Normals at for those (x,y) pts
s::Vector{T} # Arclengths for those (x,y) pts
ds::Vector{T} # Differences between the arclengths of pts.
end
```

# Arguments
- `x::T`: x-coordinate of the point to compute the wavefunction.
- `y::T`: y-coordinate of the point to compute the wavefunction.
- `k::T`: The eigenvalue for which the wavefunction is to be computed.
- `bdPoints::BoundaryPoints`: Boundary discretization information.
- `us::Vector`: Vector of boundary functions.

# Returns
- `ϕ::T`: The value of the wavefunction at the given point (x,y).
"""
function ϕ(x::T,y::T,k::T,bdPoints::BoundaryPoints,us::Vector) where {T<:Real}
    target_point=SVector(x,y)
    distances=norm.(Ref(target_point).-bdPoints.xy)
    weighted_bessel_values=Bessels.bessely0.(k*distances).*us.*bdPoints.ds
    return sum(weighted_bessel_values)/4
end

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
- `fundamental::Bool=true`: (Optional), Whether to use fundamental domain for boundary integral. Default is true.

# Returns
- `Psi2ds::Vector{Matrix{T}}`: Vector of 2D wavefunction matrices constructed on the same grid.
- `x_grid::Vector{T}`: Vector of x-coordinates for the grid.
- `y_grid::Vector{T}`: Vector of y-coordinates for the grid.
"""
function wavefunction_multi(ks::Vector{T}, vec_us::Vector{Vector{T}}, vec_bdPoints::Vector{BoundaryPoints{T}}, billiard::Bi; b::Float64=5.0, inside_only::Bool=true, fundamental=true) where {Bi<:AbsBilliard,T<:Real}
    k_max=maximum(ks)
    type=eltype(k_max)
    L=billiard.length
    xlim,ylim=boundary_limits(billiard.full_boundary; grd=max(1000,round(Int,k_max*L*b/(2*pi))))
    dx,dy=xlim[2]-xlim[1],ylim[2]-ylim[1]
    nx,ny=max(round(Int,k_max*dx*b/(2*pi)),512),max(round(Int,k_max*dy*b/(2*pi)),512)
    x_grid,y_grid=collect(type,range(xlim..., nx)),collect(type,range(ylim..., ny))
    pts=collect(SVector(x,y) for y in y_grid for x in x_grid)
    sz=length(pts)
    # Determine points inside the billiard only once if inside_only is true
    pts_mask=inside_only ? points_in_billiard_polygon(pts,billiard,round(Int,sqrt(sz));fundamental_domain=fundamental) : fill(true,sz)
    pts_masked_indices=findall(pts_mask)
    Psi2ds=Vector{Matrix{type}}(undef,length(ks))
    progress=Progress(length(ks),desc="Constructing wavefunction matrices...")
    Threads.@threads for i in eachindex(ks)
        k,bdPoints,us=ks[i],vec_bdPoints[i],vec_us[i]
        Psi_flat=zeros(type,sz)
        @inbounds for idx in pts_masked_indices # no bounds checking
            x,y=pts[idx]
            Psi_flat[idx]=ϕ(x,y,k,bdPoints,us)
        end
        Psi2ds[i]=reshape(Psi_flat,ny,nx)
        next!(progress)
    end
    return Psi2ds,x_grid,y_grid
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
- `fundamental::Bool=true`: (Optional), Whether to use fundamental domain for boundary integral. Default is true.

# Returns
- `Psi2ds::Vector{Matrix{T}}`: Vector of 2D wavefunction matrices constructed on the same grid.
- `x_grid::Vector{T}`: Vector of x-coordinates for the grid.
- `y_grid::Vector{T}`: Vector of y-coordinates for the grid.
- `Hs_list::Vector{Matrix{T}}`: Vector of 2D husimi function matrices.
- `ps_list::Vector{Vector{T}}`: Vector of ps grids for the husimi matrices.
- `qs_list::Vector{Vector{T}}`: Vector of qs grids for the husimi matrices.
"""
function wavefunction_multi_with_husimi(ks::Vector{T}, vec_us::Vector{Vector{T}}, vec_bdPoints::Vector{BoundaryPoints{T}}, billiard::Bi; b::Float64=5.0, inside_only::Bool=true, fundamental=true) where {Bi<:AbsBilliard,T<:Real}
    k_max=maximum(ks)
    type=eltype(k_max)
    L=billiard.length
    xlim,ylim=boundary_limits(billiard.full_boundary;grd=max(1000,round(Int,k_max*L*b/(2*pi))))
    dx,dy=xlim[2]-xlim[1],ylim[2]-ylim[1]
    nx,ny=max(round(Int,k_max*dx*b/(2*pi)),512),max(round(Int,k_max*dy*b/(2*pi)),512)
    x_grid,y_grid=collect(type,range(xlim...,nx)),collect(type,range(ylim...,ny))
    pts=collect(SVector(x,y) for y in y_grid for x in x_grid)
    sz=length(pts)
    # Determine points inside the billiard only once if inside_only is true
    pts_mask=inside_only ? points_in_billiard_polygon(pts,billiard,round(Int,sqrt(sz));fundamental_domain=fundamental) : fill(true,sz)
    pts_masked_indices=findall(pts_mask)
    Psi2ds=Vector{Matrix{type}}(undef,length(ks))
    progress=Progress(length(ks),desc="Constructing wavefunction matrices...")
    Threads.@threads for i in eachindex(ks)
        k,bdPoints,us=ks[i],vec_bdPoints[i],vec_us[i]
        Psi_flat=zeros(type,sz)
        @inbounds for idx in pts_masked_indices # no bounds checking
            x,y=pts[idx]
            Psi_flat[idx]=ϕ(x,y,k,bdPoints,us)
        end
        Psi2ds[i]=reshape(Psi_flat,ny,nx)
        next!(progress)
    end
    # husimi
    vec_of_s_vals=[bdPoints.s for bdPoints in vec_bdPoints]
    Hs_list,ps_list,qs_list=husimi_functions_from_boundary_functions(ks,vec_us,vec_of_s_vals,billiard)
    return Psi2ds,x_grid,y_grid,Hs_list,ps_list,qs_list
end

"""
    plot_wavefunctions(ks::Vector, Psi2ds::Vector, x_grid::Vector, y_grid::Vector, billiard::Bi; b::Float64=5.0, width_ax::Integer=500, height_ax::Integer=500, max_cols::Integer=6, fundamental=true) where {Bi<:AbsBilliard}

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

 # Returns
- `f::Figure`: A Figure object containing the grid of wavefunctions.
"""
function plot_wavefunctions(ks::Vector, Psi2ds::Vector, x_grid::Vector, y_grid::Vector, billiard::Bi; b::Float64=5.0, width_ax::Integer=300, height_ax::Integer=300, max_cols::Integer=6, fundamental=true) where {Bi<:AbsBilliard}
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
    for j in eachindex(ks)
        local ax=Axis(f[row,col],title="$(ks[j])",aspect=DataAspect(),width=width_ax,height=height_ax)
        hm=heatmap!(ax,x_grid,y_grid,Psi2ds[j],colormap=:balance,colorrange=(-maximum(Psi2ds[j]),maximum(Psi2ds[j])))
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
    plot_wavefunctions(ks::Vector, Psi2ds::Vector, x_grid::Vector{Vector}, y_grid::Vector{Vector}, billiard::Bi; b::Float64=5.0, width_ax::Integer=500, height_ax::Integer=500, max_cols::Integer=6, fundamental=true) where {Bi<:AbsBilliard}

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

 # Returns
- `f::Figure`: A Figure object containing the grid of wavefunctions.
"""
function plot_wavefunctions(ks::Vector, Psi2ds::Vector, x_grid::Vector{Vector}, y_grid::Vector{Vector}, billiard::Bi; b::Float64=5.0, width_ax::Integer=300, height_ax::Integer=300, max_cols::Integer=6, fundamental=true) where {Bi<:AbsBilliard}
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
    for j in eachindex(ks)
        local ax=Axis(f[row,col],title="$(ks[j])",aspect=DataAspect(),width=width_ax,height=height_ax)
        hm=heatmap!(ax,x_grid[j],y_grid[j],Psi2ds[j],colormap=:balance,colorrange=(-maximum(Psi2ds[j]),maximum(Psi2ds[j])))
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
- `fundamental::Bool=true`: If plotting just the desymmetrized part.

 # Returns
- `f::Figure`: A Figure object containing the grid of wavefunctions.
"""
function plot_wavefunctions_with_husimi(ks::Vector, Psi2ds::Vector, x_grid::Vector, y_grid::Vector, Hs_list::Vector, ps_list::Vector, qs_list::Vector, billiard::Bi; b::Float64=5.0, width_ax::Integer=300, height_ax::Integer=300, max_cols::Integer=6, fundamental=true) where {Bi<:AbsBilliard}
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
    for j in eachindex(ks)
        local ax=Axis(f[row,col][1,1],title="$(ks[j])",aspect=DataAspect(),width=width_ax,height=height_ax)
        local ax_h=Axis(f[row,col][1,2],width=width_ax,height=height_ax)
        hm=heatmap!(ax,x_grid,y_grid,Psi2ds[j],colormap=:balance,colorrange=(-maximum(Psi2ds[j]),maximum(Psi2ds[j])))
        plot_boundary!(ax,billiard,fundamental_domain=fundamental,plot_normal=false)
        hm_h=heatmap!(ax_h,qs_list[j],ps_list[j],Hs_list[j];colormap=Reverse(:gist_heat))
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
    plot_wavefunctions(ks::Vector, Psi2ds::Vector, x_grid::Vector, y_grid::Vector, billiard::Bi; b::Float64=5.0, width_ax::Integer=500, height_ax::Integer=500, max_cols::Integer=6) where {Bi<:AbsBilliard}

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

 # Returns
- `f::Figure`: A Figure object containing the grid of wavefunctions.
"""
function plot_wavefunctions_with_husimi(ks::Vector, Psi2ds::Vector, x_grid::Vector, y_grid::Vector, Hs_list::Vector, ps_list::Vector, qs_list::Vector, billiard::Bi, us_all::Vector, s_vals_all::Vector; b::Float64=5.0, width_ax::Integer=300, height_ax::Integer=300, max_cols::Integer=6, fundamental=true) where {Bi<:AbsBilliard}
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
    for j in eachindex(ks)
        local ax_wave=Axis(f[row, col][1, 1],title="$(ks[j])",aspect=DataAspect(),width=width_ax,height=height_ax)
        hm_wave=heatmap!(ax_wave,x_grid,y_grid,Psi2ds[j],colormap=:balance,colorrange=(-maximum(Psi2ds[j]), maximum(Psi2ds[j])))
        plot_boundary!(ax_wave,billiard,fundamental_domain=fundamental,plot_normal=false)
        xlims!(ax_wave,xlim)
        ylims!(ax_wave,ylim)
        local ax_husimi=Axis(f[row, col][1, 2],width=width_ax,height=height_ax)
        hm_husimi=heatmap!(ax_husimi,qs_list[j],ps_list[j],Hs_list[j];colormap=Reverse(:gist_heat))
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


