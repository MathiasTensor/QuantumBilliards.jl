
using Makie
function plot_matrix!(f, A; log=false)
    Z = deepcopy(A)
    ax = Axis(f[1,1])
    if log
        m1 = log10(findmax(Z)[1])
        m2 = log10(abs(findmin(Z)[1]))
        S = sign.(Z)
        Z = abs.(Z)
        Z[Z.<eps()] .= NaN
        Z = log10.(Z)
        N = deepcopy(Z)
        N[S .>= 0.0] .= NaN #mask positive amplitudes
        Z[S .< 0.0] .= NaN #mask negative amplitudes

        #println("$m1, $m2")
        range_val1 = (-15,m1)
        range_val2 = (-15,m2)

        hmap1 = heatmap!(ax, N, colormap = :algae, colorrange=range_val1)
        hmap2 = heatmap!(ax, Z, colormap = :amp, colorrange=range_val2)
        ax.yreversed=true
        ax.aspect=DataAspect()
        Colorbar(f[1,2], colormap = :algae, limits = Float64.(range_val1),tellheight=true)
        rowsize!(f.layout, 1, ax.scene.px_area[].widths[2])
        Colorbar(f[1,3], colormap = :amp, limits = Float64.(range_val2),tellheight=true)
        rowsize!(f.layout, 1, ax.scene.px_area[].widths[2])
        #Colorbar(f[1, 2], colormap = :balance, limits = Float64.(range_val))
    else
        m = findmax(abs.(Z))[1]
        Z[abs.(Z).<eps()] .= NaN
        #println("$m")
        range_val = (-m,m) 
        hmap = heatmap!(ax, Z, colormap = :balance, colorrange=range_val)
        ax.yreversed=true
        ax.aspect=DataAspect()
        Colorbar(f[1,2], colormap = :balance, limits = Float64.(range_val),tellheight=true)
        rowsize!(f.layout, 1, ax.scene.px_area[].widths[2])
    end
    return ax
end

function plot_geometry_test!(f,billiard)
    ax, hmap = plot_domain!(f,billiard)
    plot_boundary!(ax,billiard;dens=20.0)
end

# TODO make this work
function raycast_to_rectangle(x0, y0, θ, x_limits, y_limits)
    # Extract rectangle limits
    x_min, x_max = x_limits
    y_min, y_max = y_limits
    # Direction of the ray from the angle θ
    dx, dy = cos(θ), sin(θ)
    # Helper function to check intersection with a boundary
    function check_intersection(x_int, y_int, t)
        (t > 0 && x_min <= x_int <= x_max && y_min <= y_int <= y_max) ? (x_int, y_int) : nothing
    end
    # Calculate intersection points
    intersections = [
        dx != 0 ? check_intersection(x_min, y0 + (x_min - x0) / dx * dy, (x_min - x0) / dx) : nothing,
        dx != 0 ? check_intersection(x_max, y0 + (x_max - x0) / dx * dy, (x_max - x0) / dx) : nothing,
        dy != 0 ? check_intersection(x0 + (y_min - y0) / dy * dx, y_min, (y_min - y0) / dy) : nothing,
        dy != 0 ? check_intersection(x0 + (y_max - y0) / dy * dx, y_max, (y_max - y0) / dy) : nothing
    ]
    valid_intersections = filter(!isnothing, intersections)

    # Check if there are any valid intersections
    if isempty(valid_intersections)
        return nothing
    else
        # Use `map` to apply the function for distance calculation, then find the minimum
        return minimum(valid_intersections, by = p -> hypot(p[1] - x0, p[2] - y0))
    end
end

# TODO make this work for the case of emphasized_discontinuity
function plot_basis_test!(f,basis,billiard;i=1,k=10.0, emphasized_discontinuity=false)
    basisstate = BasisState(basis,k, i)
    ax, hmap = plot_wavefunction!(f, basisstate,billiard; b=10.0, plot_normal=false) 
    if emphasized_discontinuity && (nameof(typeof(basis)) == nameof(CornerAdaptedFourierBessel))
        # get type of elem from corner angle
        typ = eltype(basis.corner_angle)
        x_limits = [ax.scene.plots[1][1][][1], ax.scene.plots[1][1][][end]] # get the x limits on the heatmap
        y_limits = [ax.scene.plots[1][2][][1], ax.scene.plots[1][2][][end]] # get the y limits of the heatmap
        x0 = basis.cs.origin[1] # get the x of origin
        y0 = basis.cs.origin[2] # get the y of origin
        angle = basis.rotation_angle_discontinuity # get the angle of the discontinuity from modified atan2
        pt_on_heatmap_boundary = raycast_to_rectangle(x0, y0, angle, x_limits, y_limits) # find the intersection on the boundary via raycasting in the direction where the angle is pointing
        if !isnothing(pt_on_heatmap_boundary) # This should always be true
            lines!(ax, [x0, pt_on_heatmap_boundary[1]], [y0, pt_on_heatmap_boundary[2]], linestyle=:dash) # plot the line from the origin to the heatmap boundary intersect point
        end
    end
end

function plot_solver_test!(f,acc_solver::S,basis,billiard,k1,k2,dk;log=true, tol=1e-4) where {S<:AcceleratedSolver}
    alpha = 0.6
    k0 = k1
    k0s = [k0]
    #initial computation
    k_res, ten_res = solve_spectrum(acc_solver, basis, billiard, k0, dk+tol)
    #println("initial: $k_res")
    control = [false for i in 1:length(k_res)]
    cycle = Makie.wong_colors()[1:6]
    ax = Axis(f[1,1])
    scatter!(ax, k_res, log10.(ten_res),color=(cycle[1], alpha))
    scatter!(ax, k_res, zeros(length(k_res)),color=(cycle[1], alpha))
    vlines!(ax, [k0], color=(cycle[1], 0.8), linewidth= 0.75)

    #println("iteration 0")
    #println("merged: $k_res")
    
    #println("overlaping: $ks")
    i=1
    while k0 < k2
        #println("iteration $i")
        k0 += dk
        push!(k0s)
        k_new, ten_new = solve_spectrum(acc_solver, basis, billiard, k0, dk+tol)
        scatter!(ax, k_new, log10.(ten_new),color=(cycle[mod1(i+1,6)], alpha))
        scatter!(ax, k_new, zeros(length(k_new)),color=(cycle[mod1(i+1,6)], alpha))
        vlines!(ax, [k0], color=(cycle[mod1(i+1,6)], 0.8), linewidth= 0.75)
        i+=1
        #println("new: $k_new")
        #println("overlap: $(k0-dk), $k0")
        overlap_and_merge!(k_res, ten_res, k_new, ten_new, control, k0-dk, k0; tol=tol)
        #println("merged: $k_res")
        #println("control: $control")
    end
    scatter!(ax, k_res, log10.(ten_res), color=(:black, 1.0), marker=:x,  ms = 100)
    scatter!(ax, k_res, zeros(length(k_res)), color=(:black, 1.0), marker=:x, ms = 100)
    #display(f)
    #return k_res, ten_res, control
end



function plot_solver_test!(f,sw_solver::S,basis,billiard,k1,k2,dk;log=true) where {S<:SweepSolver}
    ks = collect(range(k1, k2, step=dk))
    ts = k_sweep(sw_solver,basis,billiard,ks)
    if log
        ax = Axis(f[1,1],xlabel=L"k", ylabel=L"\log(t)")
        lines!(ax, ks, log10.(ts))
    else
        ax = Axis(f[1,1],xlabel=L"k", ylabel=L"t")
        lines!(ax, ks, ts)
    end
    #return ax
end

function plot_state_test!(f,state; b_psi=10.0, b_u = 20.0, log_psi =(true,-5))
    plot_probability!(f[1:2,1:2], state; b=b_psi, log = log_psi)
    k = state.k
    billiard=state.billiard
    L = billiard.length
    ax_u = Axis(f[3,1], xlabel=L"q", ylabel=L"u")
    u, s, norm = boundary_function(state; b=b_u)
    edges = curve_edge_lengths(billiard)
    lines!(ax_u, s, u)
    vlines!(ax_u, edges; color=:black, linewidth=0.5)
    text!(ax_u,1.0,1.0, text = "norm=$norm", align = (:right, :top), color = :black, space=:relative)

    ax_k = Axis(f[3,2], xlabel=L"k", ylabel=L"u_k")
    mf, ks = momentum_function(u,s)
    lines!(ax_k, ks, mf)
    vlines!(ax_k, [k]; color=:black, linewidth=0.5)
    xlims!(ax_k, 0.0, 1.2*k)
    
    H, qs, ps = husimi_function(k,u,s,L; w = 7.0)    
    hmap, ax_H = plot_heatmap!(f[4,1:2],qs,ps,H)
    vlines!(ax_H, edges; color=:black, linewidth=0.5)
end