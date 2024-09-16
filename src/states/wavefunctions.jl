using StaticArrays










function billiard_polygon(billiard::Bi, N_polygon_checks::Int; fundamental_domain=true) :: Vector where {Bi<:AbsBilliard}
    if fundamental_domain
        boundary = billiard.fundamental_boundary
    else
        boundary = billiard.full_boundary
    end
    # Find the fraction of lengths wrt to the boundary
    billiard_composite_lengths = [crv.length for crv in boundary]
    #println("Curve lengths: ", billiard_composite_lengths)
    typ = eltype(billiard_composite_lengths[1])
    total_billiard_length = sum(billiard_composite_lengths)
    billiard_length_fractions = [crv.length/total_billiard_length for crv in boundary]
    #println("Curve fractions: ", billiard_length_fractions)
    # Redistribute points based on the fractions
    distributed_points = [round(Int, fract*N_polygon_checks) for fract in billiard_length_fractions]
    #println("Curve points: ", distributed_points)
    # Use linear sampling 
    ts_vectors = [sample_points(LinearNodes(), crv_pts)[1] for crv_pts in distributed_points] # vector of vectors for each crv a vector of ts
    xy_vectors = Vector{Vector}(undef, length(boundary))
    for (i, crv) in enumerate(boundary) 
        xy_vectors[i] = curve(crv, ts_vectors[i])
    end
    return xy_vectors
end




# Helper function to check if a point is left of an edge
function is_left(p1::SVector{2,T}, p2::SVector{2,T}, pt::SVector{2,T}) where {T<:Real}
    return (p2[1] - p1[1]) * (pt[2] - p1[2]) - (pt[1] - p1[1]) * (p2[2] - p1[2])
end





# Winding number algorithm to check if a point is inside a polygon
function is_point_in_polygon(polygon::Vector{SVector{2,T}}, point::SVector{2,T})::Bool where T
    winding_number = 0
    num_points = length(polygon)
    for i in 1:num_points
        p1 = polygon[i]
        p2 = polygon[(i % num_points) + 1]
        
        if p1[2] <= point[2]
            if p2[2] > point[2] && is_left(p1, p2, point) > 0
                winding_number += 1
            end
        else
            if p2[2] <= point[2] && is_left(p1, p2, point) < 0
                winding_number -= 1
            end
        end
    end
    return winding_number != 0
end





function points_in_billiard_polygon(pts::Vector{SVector{2,T}}, billiard::Bi, N_polygon_checks::Int; fundamental_domain=true) where {T<:Real,Bi<:AbsBilliard}
    # Get the polygon points from the billiard boundary
    polygon_xy_vectors = billiard_polygon(billiard, N_polygon_checks; fundamental_domain=fundamental_domain)
    polygon_points = vcat(polygon_xy_vectors...)
    mask = fill(false, length(pts))
    Threads.@threads for i in 1:length(pts)
        mask[i] = is_point_in_polygon(polygon_points, pts[i])
    end
    return mask
end





function compute_psi(state::S, x_grid, y_grid; inside_only=true, memory_limit = 10.0e9) where {S<:AbsState}
    let vec = state.vec, k = state.k_basis, basis=state.basis, billiard=state.billiard, eps=state.eps #basis is correct size
        sz = length(x_grid)*length(y_grid)
        pts = collect(SVector(x,y) for y in y_grid for x in x_grid)
        if inside_only
            #pts_mask = is_inside(billiard,pts)
            pts_mask = points_in_billiard_polygon(pts, billiard, round(Int, sqrt(sz)); fundamental_domain=inside_only)
            pts = pts[pts_mask]
        end
        n_pts = length(pts)
        #estimate max memory needed for the matrices
        type = eltype(vec)
        memory = sizeof(type)*basis.dim*n_pts
        Psi = zeros(type,sz)

        if memory < memory_limit
            B = basis_matrix(basis, k, pts)
            Psi_pts = B*vec
            if inside_only
                Psi[pts_mask] .= Psi_pts
            else
                Psi .= Psi_pts
            end
        else
            println("Warning: memory limit of $(Base.format_bytes(memory_limit)) exceded $(Base.format_bytes(memory)).")
            if inside_only
                for i in eachindex(vec)
                    if abs(vec[i]) > eps 
                        Psi[pts_mask] .+= vec[i].*basis_fun(basis,i,k,pts)
                    end
                end
            else
                for i in eachindex(vec)
                    if abs(vec[i]) > eps 
                        Psi .+= vec[i].*basis_fun(basis,i,k,pts)
                    end
                end
            end
            #println("Psi type $(eltype(Psi)), $(memory_size(Psi))")
        end
        return Psi
    end
end




#=

#try using strided to optimize this
function compute_psi(state::S, x_grid, y_grid; inside_only=true, memory_limit = 10.0e9) where {S<:AbsState}
    let vec = state.vec, k = state.k_basis, basis=state.basis, billiard=state.billiard, eps=state.eps #basis is correct size
        sz = length(x_grid)*length(y_grid)
        pts = collect(SVector(x,y) for y in y_grid for x in x_grid)
        if inside_only
            pts_mask = is_inside(billiard,pts)
            pts = pts[pts_mask]
        end
        n_pts = length(pts)
        #estimate max memory needed for the matrices
        type = eltype(vec)
        memory = sizeof(type)*basis.dim*n_pts
        Psi = zeros(type,sz)

        if memory < memory_limit
            B = basis_matrix(basis, k, pts)
            Psi_pts = B*vec
            if inside_only
                Psi[pts_mask] .= Psi_pts
            else
                Psi .= Psi_pts
            end
        else
            println("Warning: memory limit of $(Base.format_bytes(memory_limit)) exceded $(Base.format_bytes(memory)).")
            if inside_only
                for i in eachindex(vec)
                    if abs(vec[i]) > eps 
                        Psi[pts_mask] .+= vec[i].*basis_fun(basis,i,k,pts)
                    end
                end
            else
                for i in eachindex(vec)
                    if abs(vec[i]) > eps 
                        Psi .+= vec[i].*basis_fun(basis,i,k,pts)
                    end
                end
            end
            #println("Psi type $(eltype(Psi)), $(memory_size(Psi))")
        end
        return Psi
    end
end


=#

function wavefunction(state::S; b=5.0, inside_only=true, fundamental_domain = true, memory_limit = 10.0e9) where {S<:AbsState}
    let k = state.k, billiard=state.billiard, symmetries=state.basis.symmetries       
        #println(new_basis.dim)
        type = eltype(state.vec)
        #try to find a lazy way to do this
        L = billiard.length
        
        xlim,ylim = boundary_limits(billiard.fundamental_boundary; grd=max(1000,round(Int, k*L*b/(2*pi))))
        dx = xlim[2] - xlim[1]
        dy = ylim[2] - ylim[1]
        nx = max(round(Int, k*dx*b/(2*pi)), 512)
        ny = max(round(Int, k*dy*b/(2*pi)), 512)
        x_grid::Vector{type} = collect(type,range(xlim... , nx))
        y_grid::Vector{type} = collect(type,range(ylim... , ny))
        Psi::Vector{type} = compute_psi(state,x_grid,y_grid; inside_only, memory_limit) 
        #println("Psi type $(eltype(Psi)), $(memory_size(Psi))")
        Psi2d::Array{type,2} = reshape(Psi, (nx,ny))
        if ~fundamental_domain 
            if ~isnothing(symmetries)
                x_axis = 0.0
                y_axis = 0.0
                if hasproperty(billiard, :x_axis)
                    println(nameof(typeof(billiard)), "has the :x_axis problems")
                    x_axis = billiard.x_axis
                end
                if hasproperty(billiard, :y_axis)
                    println(nameof(typeof(billiard)), "has the :y_axis problems")
                    y_axis = billiard.y_axis
                end
                Psi2d, x_grid, y_grid = reflect_wavefunction(Psi2d,x_grid,y_grid,symmetries; x_axis=x_axis, y_axis=y_axis)
            end
        end
        return Psi2d, x_grid, y_grid
    end
end

function wavefunction(state::BasisState; xlim =(-2.0,2.0), ylim=(-2.0,2.0), b=5.0) 
    let k = state.k, basis=state.basis      
        #println(new_basis.dim)
        type = eltype(state.vec)
        #try to find a lazy way to do this
        dx = xlim[2] - xlim[1]
        dy = ylim[2] - ylim[1]
        nx = max(round(Int, k*dx*b/(2*pi)), 512)
        ny = max(round(Int, k*dy*b/(2*pi)), 512)
        x_grid::Vector{type} = collect(type,range(xlim... , nx))
        y_grid::Vector{type} = collect(type,range(ylim... , ny))
        pts_grid = [SVector(x,y) for y in y_grid for x in x_grid]
        Psi::Vector{type} = basis_fun(basis,state.idx,k,pts_grid) 
        #println("Psi type $(eltype(Psi)), $(memory_size(Psi))")
        Psi2d::Array{type,2} = reshape(Psi, (nx,ny))
        return Psi2d, x_grid, y_grid
    end
end

#this can be optimized
function compute_psi(state_bundle::S, x_grid, y_grid; inside_only=true, memory_limit = 10.0e9) where {S<:EigenstateBundle}
    let k = state_bundle.k_basis, basis=state_bundle.basis, billiard=state_bundle.billiard, X=state_bundle.X #basis is correct size
        sz = length(x_grid)*length(y_grid)
        pts = collect(SVector(x,y) for y in y_grid for x in x_grid)
        if inside_only
            pts_mask = is_inside(billiard,pts)
            pts = pts[pts_mask]
        end
        n_pts = length(pts)
        n_states = length(state_bundle.ks)
        #estimate max memory needed for the matrices
        type = eltype(state_bundle.X)
        memory = sizeof(type)*basis.dim*n_pts
        #Vector of results
        Psi_bundle = zeros(type,(sz,n_states))    
        if memory < memory_limit
            #Psi = zeros(type,sz)
            B = basis_matrix(basis, k, pts)
            Psi_pts = B*X
            Psi_bundle[pts_mask,:] .= Psi_pts
        else
            println("Warning: memory limit of $(Base.format_bytes(memory_limit)) exceded $(Base.format_bytes(memory)).")
            
            Psi_pts = zeros(type,(n_pts,n_states))
            for i in 1:basis.dim
                bf = basis_fun(basis,i,k,pts) #vector of length n_pts
                for j in 1:n_states
                    Psi_pts[:,j] .+= X[i,j].*bf
                end
            end
            if inside_only
                Psi_bundle[pts_mask,:] = Psi_pts
            else
                Psi_bundle = Psi_pts
            end
            #println("Psi type $(eltype(Psi)), $(memory_size(Psi))")
        end
        return Psi_bundle #this is a matrix 
    end
end

function wavefunction(state_bundle::S; b=5.0, inside_only=true, fundamental_domain = true, memory_limit = 10.0e9) where {S<:EigenstateBundle}
    let k = state_bundle.k_basis, billiard=state_bundle.billiard, symmetries=state_bundle.basis.symmetries          
        #println(new_basis.dim)
        type = eltype(state_bundle.X)
        #try to find a lazy way to do this
        L = billiard.length
        xlim,ylim = boundary_limits(billiard.fundamental_boundary; grd=max(1000,round(Int, k*L*b/(2*pi))))
        dx = xlim[2] - xlim[1]
        dy = ylim[2] - ylim[1]
        nx = max(round(Int, k*dx*b/(2*pi)), 512)
        ny = max(round(Int, k*dy*b/(2*pi)), 512)
        x_grid::Vector{type} = collect(type,range(xlim... , nx))
        y_grid::Vector{type} = collect(type,range(ylim... , ny))
        Psi_bundle::Matrix{type} = compute_psi(state_bundle,x_grid,y_grid;inside_only=inside_only, memory_limit = memory_limit) 
        #println("Psi type $(eltype(Psi)), $(memory_size(Psi))")
        Psi2d::Vector{Array{type,2}} = [reshape(Psi, (nx,ny)) for Psi in eachcol(Psi_bundle)]
        if ~fundamental_domain 
            if ~isnothing(symmetries)
                for i in eachindex(Psi2d)
                    Psi_new, x_grid, y_grid = reflect_wavefunction(Psi2d[i],x_grid,y_grid,symmetries)
                    Psi2d[i] = Psi_new
                end
            end
        end
        return Psi2d, x_grid, y_grid
    end
end


