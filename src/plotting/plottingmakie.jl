#include("../abstracttypes.jl")
#include("../utils/gridutils.jl")
using Makie
using StaticArrays
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
function plot_curve!(ax, crv::AbsRealCurve; plot_normal=true, dens = 20.0)
    L = crv.length
    grid = max(round(Int, L*dens),3)
    t = range(0.0,1.0, grid)
    pts = curve(crv,t)
    lines!(ax,pts, color = :grey, linewidth = 0.75 )
    if plot_normal
        ns = normal_vec(crv,t)
        arrows!(ax,getindex.(pts,1),getindex.(pts,2), getindex.(ns,1),getindex.(ns,2), color = :black, lengthscale = 0.1)
    end
    ax.aspect=DataAspect()
end

function plot_curve!(ax, crv::AbsVirtualCurve; plot_normal=false, dens = 10.0)
    L = crv.length
    grid = max(round(Int, L*dens),3)
    t = range(0.0,1.0, grid)
    pts = curve(crv,t)
    lines!(ax,pts, color = :grey, linestyle = :dash, linewidth = 0.75)
    if plot_normal
        ns = normal_vec(crv,t)
        arrows!(ax,getindex.(pts,1),getindex.(pts,2), getindex.(ns,1),getindex.(ns,2), color = :black, lengthscale = 0.1)
    end
    ax.aspect=DataAspect()
end

function plot_boundary!(ax, billiard::AbsBilliard; fundamental_domain = true, dens = 100.0, plot_normal=true)
    if fundamental_domain 
        boundary = billiard.fundamental_boundary  
    else
        boundary = billiard.full_boundary
    end
    
    for curve in boundary 
        plot_curve!(ax, curve; dens = dens, plot_normal = plot_normal)
    end
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
    plot_boundary!(ax, billiard; dens = dens, plot_normal=plot_normal, fundamental_domain=fundamental_domain)
    return ax, hmap
end

function plot_wavefunction!(f,state::BasisState, billiard::AbsBilliard; b=5.0,dens = 10.0, 
    plot_normal=false, vmax = 1.0, cmap=Reverse(:balance),hmargs=Dict(),axargs=Dict())
    Psi, x, y = wavefunction(state;b=b)
    #Psi[Psi .== zero(eltype(Psi))] .= NaN
    hmap, ax = plot_heatmap_balaced!(f,x,y,Psi ;vmax = vmax, cmap=cmap,hmargs=hmargs,axargs=axargs)
    plot_boundary!(ax, billiard; dens = dens, plot_normal=plot_normal)
    return ax, hmap
end

#not finished yet
function plot_wavefunction_gradient!(f,state::AbsState; b=5.0,dens = 10.0, inside_only=true, plot_normal=false, lengthscale = 0.001, cmap=Reverse(:balance),hmargs=Dict(),axargs=Dict())
    #Psi[Psi .== zero(eltype(Psi))] .= NaN
    ax = Axis(f[1,1])  
    dX, dY, x_grid, y_grid =  wavefunction_gradient(state;b=b, inside_only=inside_only)
    arrows!(ax,x_grid,y_grid, dX,dY, color = :black, lengthscale = lengthscale)
    billiard = state.billiard
    plot_boundary!(ax, billiard; dens = dens, plot_normal=plot_normal)
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
    plot_radially_integrated_density(ax::Axis, state::S; b::Float64=5.0, num_points::Int=500) where {S<:AbsState}

Plots the radially integrated momentum density `I(φ)` as a function of angle `φ` into the provided axis `ax`.

# Arguments
- `ax::Axis`: The `Axis` object to plot into.
- `state::S`: An instance of a subtype of `AbsState`, representing the quantum state.
- `b::Float64=5.0`: An optional parameter controlling the number of boundary points. Defaults to `5.0`.
- `num_points::Int=500`: The number of points to use in the plot. Defaults to `500`.

# Returns
- The modified `Axis` object with the plot added.

# Description
This function computes the radially integrated momentum density using `computeRadiallyIntegratedDensityFromState` and plots `I(φ)` over the interval `φ ∈ [0, 2π]` into the provided axis `ax`.

# Notes
- The plot will display the momentum density as a function of angle `φ` in radians.
- The axis `ax` is modified in place and returned.
"""
function plot_radially_integrated_density(ax::Axis, state::S; b::Float64=5.0, num_points::Int=500) where {S<:AbsState}
    I_phi_function = computeRadiallyIntegratedDensityFromState(state; b)
    φ_values = range(0, 2π, length=num_points)
    I_values = [I_phi_function(φ) for φ in φ_values]
    lines!(ax, φ_values, I_values, label="I(φ)")
    ax.xlabel = "φ (radians)"
    ax.ylabel = "I(φ)"
    ax.title = "Radially Integrated Momentum Density"
    return ax
end

"""
    plot_angularly_integrated_density(ax::Axis, state::S; b::Float64=5.0, r_max::Float64=10.0, num_points::Int=500) where {S<:AbsState}

Plots the angularly integrated momentum density `R(r)` as a function of radius `r` into the provided axis `ax`.

# Arguments
- `ax::Axis`: The `Axis` object to plot into.
- `state::S`: An instance of a subtype of `AbsState`, representing the quantum state.
- `b::Float64=5.0`: An optional parameter controlling the number of boundary points. Defaults to `5.0`.
- `r_max::Float64=10.0`: The maximum radius to plot. Defaults to `10.0`.
- `num_points::Int=500`: The number of points to use in the plot. Defaults to `500`.

# Returns
- The modified `Axis` object with the plot added.

# Description
This function computes the angularly integrated momentum density using `computeAngularIntegratedMomentumDensityFromState` and plots `R(r)` over the interval `r ∈ [0, r_max]` into the provided axis `ax`.

# Notes
- The plot will display the momentum density as a function of radius `r`.
- The axis `ax` is modified in place and returned.
"""
function plot_angularly_integrated_density(ax::Axis, state::S; b::Float64=5.0, r_max::Float64=10.0, num_points::Int=500) where {S<:AbsState}
    R_r_function = computeAngularIntegratedMomentumDensityFromState(state; b)
    r_values = range(0, r_max, length=num_points)
    R_values = [R_r_function(r) for r in r_values]
    lines!(ax, r_values, R_values, label="R(r)")
    ax.xlabel = "r"
    ax.ylabel = "R(r)"
    ax.title = "Angularly Integrated Momentum Density"
    return ax
end