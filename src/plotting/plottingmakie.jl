#include("../abstracttypes.jl")
#include("../utils/gridutils.jl")
using Makie
using StaticArrays
using CairoMakie
#helper functions
function plot_heatmap!(f,x,y,Z ;vmax = 1.0,log=(false,-5.0), cmap=Reverse(:gist_heat),hmargs=Dict(),axargs=Dict(),cbar = true)
    if log[1]
        X = log10.(Z)
        ax = Axis(f[1,1],axargs...)        
        m = findmax(X)[1]
        range_val = (log[2],m*vmax)
        hmap = heatmap!(ax,x, y, X, colormap = cmap, colorrange=range_val, hmargs...)
        ax.aspect=DataAspect()
        if cbar
            Colorbar(f[1,2], colormap = cmap, limits = Float64.(range_val),tellheight=true)
            rowsize!(f.layout, 1, ax.scene.px_area[].widths[2])
        end
    else
        ax = Axis(f[1,1],axargs...)        
        m = findmax(Z)[1]
        range_val = (0,m*vmax)
        hmap = heatmap!(ax,x, y, Z, colormap = cmap, colorrange=range_val, hmargs...)
        ax.aspect=DataAspect()
        if cbar
            Colorbar(f[1,2], colormap = cmap, limits = Float64.(range_val),tellheight=true)
            rowsize!(f.layout, 1, ax.scene.px_area[].widths[2])
        end
    end
    return hmap, ax
end

function plot_heatmap_balaced!(f,x,y,Z ;vmax = 1.0, cmap=Reverse(:balance),hmargs=Dict(),axargs=Dict())
    ax = Axis(f[1,1],axargs...)        
    m = findmax(abs, Z)[1]
    range_val = (-m*vmax,m*vmax)
    hmap = heatmap!(ax,x, y, Z, colormap = cmap, colorrange=range_val, hmargs...)
    ax.aspect=DataAspect()
    Colorbar(f[1,2], colormap = cmap, limits = Float64.(range_val),tellheight=true)
    rowsize!(f.layout, 1, ax.scene.px_area[].widths[2])
    return hmap, ax
end

#curve and billiard ploting
function plot_curve!(ax, crv::AbsRealCurve; plot_normal=true, dens = 20.0, color_crv=:grey)
    L = crv.length
    grid = max(round(Int, L*dens),3)
    t = range(0.0,1.0, grid)
    pts = curve(crv,t)
    lines!(ax,pts, color = color_crv, linewidth = 0.75 )
    if plot_normal
        ns = normal_vec(crv,t)
        arrows!(ax,getindex.(pts,1),getindex.(pts,2), getindex.(ns,1),getindex.(ns,2), color = :black, lengthscale = 0.1)
    end
    ax.aspect=DataAspect()
end

function plot_curve!(ax, crv::AbsVirtualCurve; plot_normal=false, dens = 10.0, color_crv=:grey)
    L = crv.length
    grid = max(round(Int, L*dens),3)
    t = range(0.0,1.0, grid)
    pts = curve(crv,t)
    lines!(ax,pts, color = color_crv, linestyle = :dash, linewidth = 0.75)
    if plot_normal
        ns = normal_vec(crv,t)
        arrows!(ax,getindex.(pts,1),getindex.(pts,2), getindex.(ns,1),getindex.(ns,2), color = :black, lengthscale = 0.1)
    end
    ax.aspect=DataAspect()
end

function plot_boundary!(ax, billiard::AbsBilliard; fundamental_domain = true, desymmetrized_full_domain=false, dens = 100.0, plot_normal=true, color_crv=:grey)
    if fundamental_domain
        boundary = billiard.fundamental_boundary
    elseif desymmetrized_full_domain
        boundary = billiard.desymmetrized_full_boundary
    else
        boundary = billiard.full_boundary
    end
    for curve in boundary 
        plot_curve!(ax, curve; dens = dens, plot_normal = plot_normal, color_crv=color_crv)
    end
end

"""
    function plot_boundary_orientation!(ax::Axis, billiard::Bi; fundamental_domain::Bool=true, dens::Float64=5.0, plot_normal::Bool=true, desymmetrized_full_domain=false) where {Bi<:AbsBilliard}

Plots the boundary orientation as a sequence of arrows for the points that form the given curve segments in the billiard boundary (either fundamental or full)

# Arguments
- `ax::Axis`: The `Makie::Axis` to plot on.
- `billiard<:AbsBilliard`: The billiard to plot.
- `fundamental_domain::Bool=true`: Whether to plot the boundary in the fundamental domain (default: true)
- `dens::Float64=5.0`: The density of points to plot on the boundary (default: 5.0, should be small as to )
"""
function plot_boundary_orientation!(ax::Axis, billiard::Bi; fundamental_domain::Bool=true, dens::Float64=5.0, plot_normal::Bool=true, desymmetrized_full_domain=false) where {Bi<:AbsBilliard}
    if fundamental_domain
        boundary = billiard.fundamental_boundary
    elseif desymmetrized_full_domain
        boundary = billiard.desymmetrized_full_boundary
    else
        boundary = billiard.full_boundary
    end
    all_pts = []
    for crv in boundary 
        L = crv.length
        grid = max(round(Int, L*dens),3)
        t = range(0.0,1.0, grid)
        pts = curve(crv,t)
        sps = Point{2, Float32}[]
        dirs = Point{2, Float32}[]
        for i in eachindex(pts)[1:end-1] # Plot arrows between each consecutive pair of points
            sp = pts[i]
            ep = pts[i + 1]
            dir = ep-sp
            push!(sps, sp)
            push!(dirs, dir)
        end
        arrows!(ax, sps, dirs; color=:blue, linewidth=2, arrowsize=20.0, arrowcolor=:red)
        if plot_normal
            ns = normal_vec(crv,t)
            arrows!(ax,getindex.(pts,1),getindex.(pts,2), getindex.(ns,1),getindex.(ns,2), color = :black, lengthscale = 0.1)
        end
        push!(all_pts, pts)
    end
    all_pts = []
    for crv in boundary
        L = crv.length
        grid = max(round(Int, L*100.0),3)
        t = range(0.0,1.0, grid)
        pts = curve(crv,t)
        push!(all_pts, pts)
    end
    all_pts = vcat(all_pts...)
    n = length(all_pts)
    area = 0.0 # signed area
    for i in 1:n
        x1, y1 = all_pts[i]
        x2, y2 = all_pts[mod1(i + 1, n)]
        area += x1 * y2 - y1 * x2
    end
    if area > 0.0
        println("Correct counterclockwise orientation")
    else
        println("Signed area: $(area)!!")
    end
end

"""
    plot_symmetry_adapted_boundary(basis::Ba, billiard::Bi; fundamental_or_desymmetrized_full::Bool=false, N::Integer=100, print_symmetrized_sizes=true) where {T<:Real, Ba<:AbsBasis, Bi<:AbsBilliard}

Debugging plotting function that checks if the symmetry procedures applied correctly. With different colors shows the old and then new boundary parts with the normals also plotted for visual inspection. The kwarg `fundamental_or_desymmetrized_full` is responsible for whether we apply the symmetries contained in the `basis` to the `fundamental_boundary` or the `desymmetrized_full_boundary` which is needed for any u(s) boundary function construction. Also plots the arclengths (`s`) and `ds` for all the components.
 
# Arguments
- `basis::Ba`: The basis which holds the symmetry information.
- `billiard::Bi`: The billiard from which we gather the starting geometry.
- `fundamental_or_desymmetrized_full::Bool=false`: If `true`, applies symmetries to the fundamental_boundary. If `false`, applies symmetry to the desymmetrized full boundary.
- `N::Integer=100`: Number of points to sample along the boundary.
- `print_symmetrized_sizes::Bool=true`: If `true`, prints the symmetrized and original sizes of the boundary parts.

# Returns
- `Figure`: A figure with an axis for the boundary and two axes for the arclength and ds.
"""
function plot_symmetry_adapted_boundary(basis::Ba, billiard::Bi; fundamental_or_desymmetrized_full::Bool=false, N::Integer=100, print_symmetrized_sizes=true) where {T<:Real, Ba<:AbsBasis, Bi<:AbsBilliard}
    f = Figure(size=(1500,1000), resolution=(1500,1000))
    ax_main = Axis(f[1,1][1,1:2], title="Desymmetrized boundary w/ Symmetry", width=500, height=500)
    ax_s = Axis(f[1,2][1,1], title="Arclengths")
    ax_ds = Axis(f[1,2][1,2], title="ds")
    sampler = LinearNodes()
    if fundamental_or_desymmetrized_full
        boundary = billiard.fundamental_boundary
    else
        boundary = billiard.desymmetrized_full_boundary
    end
    # generate first points without symmetry
    L = symmetry_accounted_fundamental_boundary_length(boundary)
    println(boundary)
    Lc = boundary[1].length
    Nc = round(Int, N*Lc/L)
    xy_all, normal_all, s_all, ds_all = boundary_coords(boundary[1],sampler,Nc)
    l = boundary[1].length #cumulative length
    for crv in boundary[2:end]
        if (typeof(crv) <: AbsRealCurve) # in the case of desymmetrized full all will be real
            Lc = crv.length
            Nc = round(Int, N*Lc/L)
            xy,nxy,s,ds = boundary_coords(crv,sampler,Nc)
            append!(xy_all, xy)
            append!(normal_all, nxy)
            s = s .+ l
            append!(s_all, s)
            append!(ds_all, ds)
            l += Lc
        end    
    end
    # desymmetrized points, plot in red
    pts_desym = BoundaryPoints(xy_all,normal_all,s_all,ds_all) 
    lines!(ax_main,pts_desym.xy, color=:red, linewidth=0.75)
    arrows!(ax_main,getindex.(pts_desym.xy,1),getindex.(pts_desym.xy,2), getindex.(pts_desym.normal,1),getindex.(pts_desym.normal,2), color=:black, lengthscale=0.1)
    pts_new = apply_symmetries_to_boundary_points(pts_desym, basis.symmetries, billiard; same_direction=true)
    pts_diff = difference_boundary_points(pts_new, pts_desym)
    lines!(ax_main,pts_diff.xy, color=:green, linewidth=0.75)
    arrows!(ax_main,getindex.(pts_diff.xy,1),getindex.(pts_diff.xy,2), getindex.(pts_diff.normal,1),getindex.(pts_diff.normal,2), color=:black, lengthscale=0.1)
    
    # Plot direction vectors for pts_desym
    desym_start_x = getindex.(pts_desym.xy, 1)[1:end-1]
    desym_start_y = getindex.(pts_desym.xy, 2)[1:end-1]
    desym_dir_x = getindex.(pts_desym.xy, 1)[2:end] .- desym_start_x
    desym_dir_y = getindex.(pts_desym.xy, 2)[2:end] .- desym_start_y
    arrows!(ax_main, desym_start_x, desym_start_y, desym_dir_x, desym_dir_y; color=:red, lengthscale=0.2)

    # Plot direction vectors for pts_diff
    diff_start_x = getindex.(pts_diff.xy, 1)[1:end-1]
    diff_start_y = getindex.(pts_diff.xy, 2)[1:end-1]
    diff_dir_x = getindex.(pts_diff.xy, 1)[2:end] .- diff_start_x
    diff_dir_y = getindex.(pts_diff.xy, 2)[2:end] .- diff_start_y
    arrows!(ax_main, diff_start_x, diff_start_y, diff_dir_x, diff_dir_y; color=:green, lengthscale=0.2)

    # plot s and ds sequentially
    idx_s = collect(1:length(pts_desym.s))
    idx_diff_s = collect(length(idx_s) + 1:length(idx_s) + length(pts_diff.s))
    lines!(ax_s, idx_s, pts_desym.s, color=:red, linewidth=0.75)
    lines!(ax_s, idx_diff_s, pts_diff.s, color=:green, linewidth=0.75)
    idx_ds = collect(1:length(pts_desym.ds))
    idx_diff_ds = collect(length(idx_ds) + 1:length(idx_ds) + length(pts_diff.ds))
    lines!(ax_ds, idx_ds, pts_desym.ds, color=:red, linewidth=0.75)
    lines!(ax_ds, idx_diff_ds, pts_diff.ds, color=:green, linewidth=0.75)
    if print_symmetrized_sizes
        println("NEW ONES")
        println("length xy: ", length(pts_new.xy))
        println("length normal: ", length(pts_new.normal))
        println("length s: ", length(pts_new.s))
        println("length ds: ", length(pts_new.ds))
        println("OLD ONES")
        println("length xy: ", length(pts_desym.xy))
        println("length normal: ", length(pts_desym.normal))
        println("length s: ", length(pts_desym.s))
        println("length ds: ", length(pts_desym.ds))
    end
    return f
end

function plot_domain_fun!(f, curve::C; xlim=(-1.0,1.0),ylim=(-1.0,1.0), dens=100.0, hmargs=Dict(),cmap=:binary) where {C<:AbsCurve}
    d = one(dens)/dens
    x_grid = range(xlim... ; step=d)
    y_grid = range(ylim... ; step=d)
    pts = [SVector(x,y) for y in y_grid for x in x_grid]
    Z = reshape(domain(curve,pts),length(x_grid),length(y_grid))
    hmap, ax = plot_heatmap_balaced!(f,x_grid,y_grid,Z) 
    ax.aspect=DataAspect()
    return ax, hmap
end

function plot_domain!(ax, curve::AbsCurve;xlim=(-1.0,1.0),ylim=(-1.0,1.0), dens=100.0, hmargs=Dict(),cmap=Reverse(:binary))
    d = one(dens)/dens
    x_grid = range(xlim... ; step=d)
    y_grid = range(ylim... ; step=d)
    pts = [SVector(x,y) for y in y_grid for x in x_grid]
    Z = reshape(is_inside(curve,pts),length(x_grid),length(y_grid))
    hmap = heatmap!(ax, x_grid, y_grid, Z, colormap = cmap, colorrange=(-1,1) ,hmargs...)
    ax.aspect=DataAspect()
end
#modify for consistency
function plot_domain!(f, billiard::AbsBilliard; fundamental_domain = true, dens=100.0, hmargs=Dict(),cmap=Reverse(:binary))
    d = one(dens)/dens
    #sz = (d,d)
    if fundamental_domain 
        curves = billiard.fundamental_boundary  
    else
        curves = billiard.full_boundary
    end
    xlim, ylim = boundary_limits(curves; grd=1000) 
    x_grid = range(xlim... ; step=d)
    y_grid = range(ylim... ; step=d)
    pts = [SVector(x,y) for y in y_grid for x in x_grid]
    Z = reshape(is_inside(billiard,pts;fundamental_domain=fundamental_domain),length(x_grid),length(y_grid))
    ax = Axis(f[1,1])
    hmap = heatmap!(ax, x_grid, y_grid, Z, colormap = cmap, colorrange=(-1,1) ,hmargs...)
    ax.aspect=DataAspect()
    return ax, hmap
end

#=
function plot_lattice!(ax, billiard::AbsBilliard; dens=50.0, scargs=Dict())
    d = one(dens)/dens
    sz = (d,d)
    x_plot, y_plot, gen = interior_grid(billiard,sz)
    X = [pt.xy  for pt in gen if pt.inside]
    hmap = scatter!(ax,X)
    ax.aspect=DataAspect()
end
=#


#wavefunction plotting

function plot_wavefunction!(f,state::AbsState; b=5.0,dens = 10.0, fundamental_domain = true,
    inside_only=true, plot_normal=false, vmax = 1.0, cmap=Reverse(:balance),hmargs=Dict(),axargs=Dict())
    Psi, x, y = wavefunction(state;b=b, fundamental_domain=fundamental_domain, inside_only=inside_only)
    #Psi[Psi .== zero(eltype(Psi))] .= NaN
    billiard = state.billiard
    hmap, ax = plot_heatmap_balaced!(f,x,y,Psi ;vmax = vmax, cmap=cmap,hmargs=hmargs,axargs=axargs)
    plot_boundary!(ax, billiard; dens = dens, plot_normal=plot_normal, fundamental_domain=fundamental_domain, desymmetrized_full_domain=false)
    return ax, hmap
end

function plot_wavefunction!(f,state::BasisState, billiard::AbsBilliard; b=5.0,dens = 10.0, 
    plot_normal=false, vmax = 1.0, cmap=Reverse(:balance),hmargs=Dict(),axargs=Dict())
    Psi, x, y = wavefunction(state;b=b)
    #Psi[Psi .== zero(eltype(Psi))] .= NaN
    hmap, ax = plot_heatmap_balaced!(f,x,y,Psi ;vmax = vmax, cmap=cmap,hmargs=hmargs,axargs=axargs)
    plot_boundary!(ax, billiard; dens = dens, plot_normal=plot_normal, desymmetrized_full_domain=false)
    return ax, hmap
end

#not finished yet
function plot_wavefunction_gradient!(f,state::AbsState; b=5.0,dens = 10.0, inside_only=true, plot_normal=false, lengthscale = 0.001, cmap=Reverse(:balance),hmargs=Dict(),axargs=Dict())
    #Psi[Psi .== zero(eltype(Psi))] .= NaN
    ax = Axis(f[1,1])  
    dX, dY, x_grid, y_grid =  wavefunction_gradient(state;b=b, inside_only=inside_only)
    arrows!(ax,x_grid,y_grid, dX,dY, color = :black, lengthscale = lengthscale)
    billiard = state.billiard
    plot_boundary!(ax, billiard; dens = dens, plot_normal=plot_normal, desymmetrized_full_domain=false)
    ax.aspect=DataAspect()
end

function plot_probability!(f,state::AbsState; b=5.0,dens = 100.0,log=false, fundamental_domain = true, inside_only=true, 
    plot_normal=false, vmax = 1.0, cmap=Reverse(:gist_heat),hmargs=Dict(),axargs=Dict(), memory_limit = 10.0e9,cbar=true)
    Psi, x, y = wavefunction(state;b=b,fundamental_domain=fundamental_domain, inside_only=inside_only, memory_limit=memory_limit)
    Psi = abs2.(Psi)
    #println("Psi type $(eltype(Psi)), $(memory_size(Psi))")
    
    hmap, ax = plot_heatmap!(f,x,y,Psi ;vmax, cmap,hmargs,axargs,log,cbar)
    billiard = state.billiard
    plot_boundary!(ax, billiard; dens, plot_normal, fundamental_domain)
    return ax, hmap
end


function plot_probability!(f,state_bundle::EigenstateBundle; 
    b=5.0,dens = 100.0,log=false, inside_only=true, fundamental_domain = true, plot_normal=false, 
    vmax = 1.0, cmap=Reverse(:gist_heat),hmargs=Dict(),axargs=Dict(), 
    memory_limit = 10.0e9, cbar=true)
    Psi_bundle, x, y = wavefunction(state_bundle;b=b,fundamental_domain=fundamental_domain, inside_only=inside_only, memory_limit=memory_limit)
    billiard = state_bundle.billiard
    for i in eachindex(Psi_bundle)
        P = abs2.(Psi_bundle[i])   
        hmap, ax = plot_heatmap!(f[i,1],x,y,P ;vmax, cmap,hmargs,axargs,log,cbar)
        plot_boundary!(ax, billiard; dens, plot_normal, fundamental_domain)
    end
end



function plot_boundary_function!(f,state::AbsState; 
    b=5.0, log=false, linesargs=Dict(),axargs=Dict())
    ax = Axis(f[i,1]; axargs...)
    u, s, norm = boundary_function(state; b=b)
    billiard = state.billiard
    edges = curve_edge_lengths(billiard)
    if log
        lines!(ax, s, log10.(abs.(u)); linesargs...)
        vlines!(ax, edges; color=:black, linewidth=0.5)
    else
        lines!(ax, s, u; linesargs...)
        vlines!(ax, edges; color=:black, linewidth=0.5)
    end
end

function plot_boundary_function!(f,state_bundle::EigenstateBundle; 
    b=5.0, log=false, linesargs=Dict(),axargs=Dict())
    us, s, norms = boundary_function(state_bundle; b=b)
    billiard = state_bundle.billiard
    edges = curve_edge_lengths(billiard)
    for i in eachindex(us)
        ax = Axis(f[i,1]; axargs...)
        if log
            lines!(ax, s, log10.(abs.(us[i])); linesargs...)
            vlines!(ax, edges; color=:black, linewidth=0.5)
        else
            lines!(ax, s, us[i]; linesargs...)
            vlines!(ax, edges; color=:black, linewidth=0.5)
        end
    end
end

function plot_momentum_function!(f,state::AbsState; 
    b=5.0, log=false,  linesargs=Dict(),axargs=Dict())
    mf, k_range = momentum_function(state; b=b)
    ax = Axis(f[1,1]; axargs...)
    if log
        lines!(ax, k_range, log10.(abs.(mf)); linesargs...)
        vlines!(ax, [state.k]; color=:black, linewidth=0.5)
    else
        lines!(ax, k_range, mf; linesargs...)
        vlines!(ax, [state.k]; color=:black, linewidth=0.5)
        xlims!(ax, 0.0, 1.2*state.k)
    end
end

function plot_momentum_function!(f,state_bundle::EigenstateBundle; 
    b=5.0, log=false, linesargs=Dict(),axargs=Dict())
    mfs, k_range = momentum_function(state_bundle; b=b)
    ks = state_bundle.ks
    for i in eachindex(mfs)
        ax = Axis(f[i,1]; axargs...)
        if log
            lines!(ax, k_range, log10.(abs.(mfs[i])); linesargs...)
            vlines!(ax, [ks[i]]; color=:black, linewidth=0.5)
            #xlims!(ax, 0.0, 1.2*k)
        else
            lines!(ax, k_range, mfs[i]; linesargs...)
            vlines!(ax, [ks[i]]; color=:black, linewidth=0.5)
            xlims!(ax, 0.0, 1.2*ks[i])
        end
    end
end

function plot_husimi_function!(f,state::AbsState; 
    b=5.0,log=false, vmax = 1.0, cmap=Reverse(:gist_heat),hmargs=Dict(),axargs=Dict())
    billiard = state.billiard
    L = billiard.length
    #u, s, norm = boundary_function(state; b=b)
    H, qs, ps = husimi_function(state;b)
    edges = curve_edge_lengths(billiard) 
    hmap, ax = plot_heatmap!(f,qs,ps,H; vmax = vmax, cmap=cmap,hmargs=hmargs,axargs=axargs,log=log)
    vlines!(ax, edges; color=:black, linewidth=0.5)
end


function plot_husimi_function!(f,state_bundle::EigenstateBundle; 
    b=5.0,log=false, vmax = 1.0, cmap=Reverse(:gist_heat),hmargs=Dict(),axargs=Dict())
    billiard = state_bundle.billiard
    L = billiard.length
    us, s, norms = boundary_function(state_bundle; b=b)
    ks = state_bundle.ks
    edges = curve_edge_lengths(billiard)    
    for i in eachindex(us)
        H, qs, ps = husimi_function(ks[i],us[i],s,L; w = 7.0)    
        hmap, ax = plot_heatmap!(f[i,1],qs,ps,H; vmax = vmax, cmap=cmap,hmargs=hmargs,axargs=axargs,log=log)
        vlines!(ax, edges; color=:black, linewidth=0.5)
    end
end


#=
function plot_basis_function!(ax, basis::AbsBasis, i, k; xlim=(-1,1), ylim=(-1,1), grid::Tuple = (200,200))
    x_plot = LinRange(xlim... , grid[1])
    y_plot = LinRange(ylim... , grid[2])
    x = repeat(x_plot , outer = length(y_plot))
    y = repeat(y_plot , inner = length(x_plot))
    phi = basis_fun(basis, i, k, x, y) 
    Z = reshape(phi, grid)
    heatmap!(ax, x_plot,y_plot,Z, colormap = :balance)
    ax.aspect=DataAspect()
end

function plot_basis_function!(ax, basis::AbsBasis, i, k, curve::AbsCurve, sampler;  grid::Int = 200)
    t, dt = sample_points(sampler, grid)
    x, y = curve.r(t)
    phi = basis_fun(basis, i, k, x, y) 
    
    lines!(ax,t,phi, color = :black )
end
=#




#### new additions #####


"""
    plot_radially_integrated_density!(ax::Axis, state::S; b::Float64=5.0, num_points::Int=500) where {S<:AbsState}

Plots the radially integrated momentum density `I(φ)` as a function of angle `φ` into the provided axis `ax`.

# Arguments
- `f::Figure`: Makie.Figure to plot into 
- `state::S`: An instance of a subtype of `AbsState`, representing the quantum state.
- `b::Float64=5.0`: An optional parameter controlling the number of boundary points. Defaults to `5.0`.
- `num_points::Int=300`: The number of points to use in the plot. Defaults to `300`.

# Description
This function computes the radially integrated momentum density using `computeRadiallyIntegratedDensityFromState` and plots `I(φ)` over the interval `φ ∈ [0, 2π]` into the provided axis `ax`.

# Notes
- The plot will display the momentum density as a function of angle `φ` in radians.
- The axis `ax` is modified in place and returned.
"""
function plot_radially_integrated_density!(f::Figure, state::S; b::Float64=5.0, num_points::Int=300) where {S<:AbsState}
    I_phi_function = computeRadiallyIntegratedDensityFromState(state; b)
    φ_values = range(0, 2π, length=num_points)
    I_values = [I_phi_function(φ) for φ in φ_values]
    I_values = I_values ./ maximum(I_values)
    idx_to_plot = findall(abs.(I_values) .>= 1e-6) # for Makie
    I_values = I_values[idx_to_plot]
    φ_values = φ_values[idx_to_plot]
    ax = Axis(f[1,1])
    lines!(ax, φ_values, I_values, label="I(φ)")
    ax.xlabel = "φ (radians)"
    ax.ylabel = "I(φ)"
    ax.title = "Radially Integrated Momentum Density"
end

"""
    plot_angularly_integrated_density(ax::Axis, state::S; b::Float64=5.0, num_points::Int=500) where {S<:AbsState}

Plots the angularly integrated momentum density `R(r)` as a function of radius `r` into the provided axis `ax`.

# Arguments
- `f::Figure`: Makie.Figure to plot into 
- `state::S`: An instance of a subtype of `AbsState`, representing the quantum state.
- `b::Float64=5.0`: An optional parameter controlling the number of boundary points. Defaults to `5.0`.
- `num_points::Int=300`: The number of points to use in the plot. Defaults to `300`.

# Description
This function computes the angularly integrated momentum density using `computeAngularIntegratedMomentumDensityFromState` and plots `R(r)` over the interval `r ∈ [0, r_max]` into the provided axis `ax`.

# Notes
- The plot will display the momentum density as a function of radius `r`.
- The axis `ax` is modified in place and returned.
"""
function plot_angularly_integrated_density!(f::Figure, state::S; b::Float64=5.0, r_max::Float64=10.0, num_points::Int=300) where {S<:AbsState}
    k = state.k
    R_r_function = computeAngularIntegratedMomentumDensityFromState(state; b)
    r_values = range(0, 1.5*k, length=num_points)
    R_values = [R_r_function(r) for r in r_values]
    R_values = R_values./ maximum(R_values) 
    idx_to_plot = findall(abs.(R_values) .>= 1e-6) # for Makie
    R_values = R_values[idx_to_plot]
    r_values = r_values[idx_to_plot]
    ax = Axis(f[1,1])
    scatter!(ax, r_values, R_values, label="R(r)", markersize=2)
    vlines!(ax, k; color=:red)
    ax.xlabel = "r"
    ax.ylabel = "R(r)"
    ax.title = "Angularly Integrated Momentum Density"
end



#### DO NOT USE, DOES NOT WORK. POLAR PLOTS ARE PROBLEMATIC
"""
    plot_momentum_representation_polar!(state::S; b::Float64=5.0, num_r::Int=100, num_θ::Int=100, r_max::Float64=1.5*state.k) where {S<:AbsState}

Plots the momentum representation of a quantum state using `PolarAxis` in `CairoMakie`.

# Arguments
- `f::Figure`: Makie.Figure to plot into 
- `state::S`: An instance of a subtype of `AbsState`, representing the quantum state.
- `b::Float64=5.0`: Optional parameter controlling boundary sampling density.
- `num_r::Int=100`: Number of radial points in the plot.
- `num_θ::Int=100`: Number of angular points in the plot.
- `r_max::Float64=1.5*state.k`: Maximum radial value (momentum magnitude) to plot.
"""
function plot_momentum_representation_polar!(f::Figure, state::S; b::Float64=5.0, num_r::Int=100, num_θ::Int=100, r_max::Float64=1.5*state.k) where {S<:AbsState}
    mom_function = momentum_representation_of_state(state; b)
    r_values = collect(range(0, r_max, length=num_r))
    θ_values = collect(range(0, 2π, length=num_θ))
    # Create matrices for r and θ
    R = repeat(r_values, 1, num_θ)
    Θ = repeat(θ_values', num_r, 1) 
    # Convert to Cartesian coordinates
    KX = R .* cos.(Θ)
    KY = R .* sin.(Θ)
    # Flatten KX and KY
    KX_flat = reshape(KX, :)
    KY_flat = reshape(KY, :)
    # Initialize mom_values_flat
    mom_values_flat = zeros(Float64, length(KX_flat))
    # Compute mom_values_flat
    Threads.@threads for idx in eachindex(KX_flat)
        p = SVector{2, Float64}(KX_flat[idx], KY_flat[idx])
        mom_p = mom_function(p)
        mom_values_flat[idx] = abs2(mom_p)
    end
    # Reshape mom_values_flat to mom_values
    mom_values = reshape(mom_values_flat, size(R))
    ax = CairoMakie.PolarAxis(f[1, 1])
    CairoMakie.heatmap!(ax, mom_values, colormap=:viridis)
    ax.title = "Momentum density - polar"
end


"""
    plot_momentum_cartesian_representation!(f::Figure, state::S; b::Float64=5.0, grid_size::Int=512) where {S<:AbsState}

Plots the Cartesian momentum representation |Ψ(p)|² of a quantum state `state` on a 2D grid using a heatmap, and overlays a green circle with radius `k` corresponding to the wavenumber of the state.

# Arguments
- `f::Figure`: The figure object to which the momentum representation plot and heatmap are added.
- `state::S`: The quantum state of type `S<:AbsState` for which the momentum representation is computed.
- `b::Float64`: A parameter controlling the resolution of the momentum density function (default = 5.0).
- `grid_size::Int`: The size of the grid (number of points in each dimension) on which the momentum representation is computed (default = 512).

"""
function plot_momentum_cartesian_representation!(f::Figure, state::S; b::Float64=5.0, grid_size::Int=512) where {S<:AbsState}
    # Obtain the momentum representation function and wavenumber k
    mom = momentum_representation_of_state(state; b=b)
    u_values, pts, k = setup_momentum_density(state; b=b)
    k_max = 1.2 * k
    kx_values = Float32.(collect(range(-k_max, k_max, length=grid_size)))
    ky_values = Float32.(collect(range(-k_max, k_max, length=grid_size)))
    momentum_matrix = zeros(Float32, grid_size, grid_size)
    for i in 1:grid_size
        for j in 1:grid_size
            kx = kx_values[i]
            ky = ky_values[j]
            p = SVector{2, Float64}(kx, ky)
            mom_p = mom(p)
            momentum_matrix[i, j] = abs2(mom_p)
        end
    end
    largest_val = maximum(momentum_matrix)
    for i in 1:length(grid_size)
        for j in 1:length(grid_size)
            momentum_matrix[i, j] /= largest_val
        end 
    end
    ax = Axis(f[1,1], aspect=DataAspect())
    hmap = heatmap!(ax, kx_values, ky_values, momentum_matrix, colormap=Reverse(:gist_heat))
    Colorbar(f[1,2], hmap)

    # Add the green circle with radius k
    θ_vals = range(0, stop=2π, length=200)
    circle_x = k * cos.(θ_vals)
    circle_y = k * sin.(θ_vals)
    lines!(ax, circle_x, circle_y, color=:green, linewidth=0.5, linestyle=:dash)
end

"""
    plot_point_distribution!(f::Figure, billiard::Bi, solver::Sol; plot_idxs::Bool=true, plot_normal::Bool=false, grid::Int=512)

Plots the point distribution for a given billiard system and solver, displaying the index of each curve and optionally plotting the normal vectors and colorbars for non-linear samplers.
# Arguments
- `f::Figure`: The figure object from Makie.jl where the plot will be drawn.
- `billiard::Bi`: An instance of the billiard system (of type `AbsBilliard` or a subtype).
- `solver::Sol`: An instance of a solver (of type `AbsSolver` or a subtype).
- `plot_idxs::Bool`: (Optional) Whether to plot the index of each curve at its midpoint. Defaults to `true`.
- `plot_normal::Bool`: (Optional) Whether to plot normal vectors along the curves. Defaults to `false`.
- `grid::Int`: (Optional) The number of grid points to sample along each curve. Defaults to `512`.

"""
function plot_point_distribution!(f::Figure, billiard::Bi, solver::Sol; plot_idxs::Bool=true, plot_normal::Bool=false, grid::Int = 512) where {Sol<:AbsSolver, Bi<:AbsBilliard}
    samplers = solver.sampler # get the samplers, this is only for the fundamental boundary since adjust_scaling_and_samplers only works on it and not the full boundary
    _, samplers = adjust_scaling_and_samplers(solver, billiard)
    curves_fundamental = billiard.fundamental_boundary
    ax = Axis(f[1,1], aspect=DataAspect())
    num_nl_samplers = sum([1 for sam in samplers if !(sam isa LinearNodes)])
    num_divs, remain = divrem(num_nl_samplers, 3)  # Max 3 in a line
    row = 1
    col = 1
    for (i, crv) in enumerate(curves_fundamental)
        if (crv isa PolarSegments) && (samplers[i] isa PolarSampler)
            ts, dts = sample_points(samplers[i], crv, grid)
        else
            ts, dts = sample_points(samplers[i], grid)
        end
        # Normalize dts_normalized to the range [0, 1] for better plotting
        min_val = minimum(dts)
        max_val = maximum(dts)
        if max_val - min_val == 0
            normalized_colors = fill(0.5, length(dts))  # Avoid division by zero
        else
            normalized_colors = (dts .- min_val) / (max_val - min_val)
        end
        # plot the curve
        pts = curve(crv,ts)
        sc = scatter!(ax, pts, color=normalized_colors, markersize=5)
        if plot_idxs
            mid_point = round(Int, length(pts) / 2)
            text!(ax, pts[mid_point][1], pts[mid_point][2], text = string(i), color=:black) # plot the index of the curve at the mid point
        end
        if plot_normal
            ns = normal_vec(crv,ts)
            arrows!(ax,getindex.(pts,1),getindex.(pts,2), getindex.(ns,1),getindex.(ns,2), color = :black, lengthscale = 0.1)
        end
        if !(samplers[i] isa LinearNodes) # avoid 0 range for colorbar for linear samplers
            Colorbar(f[1, 2][row, col], sc, label="$(i) - $(nameof(typeof(samplers[i])))", labelrotation=pi/2, height=Relative(0.4))
            col += 1
            if col > 3  # Move to next row after 3 columns
                col = 1
                row += 1
            end
        end
    end
end

"""
    plot_mean_level_spacing!(ax::Axis, billiard::Bi; avg_smallest_tension=1e-5, fundamental::Bool=true, step_size=0.01) where {Bi<:AbsBilliard}

Plots the mean level spacings up to the given tension input. It serves as a visual guide to determine the maximal k0 we can take in the scaling method.

# Arguments
- `ax::Axis`: The axis object from Makie where the plot will be drawn.
- `billiard::Bi`: An instance of the billiard.
- `avg_smallest_tension::Float64`: The average smallest tension for which the mean level spacing should be calculated. Defaults to 1e-5.
- `fundamental::Bool`: (Optional) Whether to consider the fundamental boundary only. Defaults to `true`.

# Returns
- `Nothing`
"""
function plot_mean_level_spacing!(ax::Axis, billiard::Bi; avg_smallest_tension=1e-5, fundamental::Bool=true, step_size=0.01) where {Bi<:AbsBilliard}
    k = 1.0
    mls = dos_weyl(k, billiard, fundamental=fundamental)
    while mls < 0.0
        k += step_size
        mls = dos_weyl(k, billiard, fundamental=fundamental)
    end
    mls_inv = 1.0/mls
    x = Float64[]
    y = Float64[]
    push!(x,k)
    push!(y,mls_inv)
    while mls_inv > avg_smallest_tension
        k += step_size
        mls_inv = 1.0/dos_weyl(k, billiard, fundamental=fundamental)
        push!(x,k)
        push!(y,mls_inv)
    end
    logy = log10.(y)
    scatter!(ax, x, logy)
    # visual indication what is the max k
    final_k = x[end]
    vlines!(ax, [final_k], color=:red, linewidth=2)
    min_y, max_y = extrema(logy)
    text!(ax, final_k, (min_y+max_y)/2, text = "k = $(round(final_k; sigdigits=3))", color=:red, rotation=pi/2)
    hlines!(ax, log10.(avg_smallest_tension), color=:green, linewidth=2)
end