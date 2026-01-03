function boundary_limits(curves; grd=1000) 
    x_bnd = Vector{Any}()
    y_bnd = Vector{Any}()
    for crv in curves #names of variables not very nice
        L = crv.length
        N_bnd = max(512,round(Int, grd/L))
        t = range(0.0,1.0, N_bnd)[1:end-1]
        pts = curve(crv,t)
        append!(x_bnd, getindex.(pts,1))
        append!(y_bnd, getindex.(pts,2))
    end
    x_bnd[end] = x_bnd[1]
    y_bnd[end] = y_bnd[1]
    xlim = extrema(x_bnd)
    #dx =  xlim[2] - xlim[1]
    ylim = extrema(y_bnd)
    #dy =  ylim[2] - ylim[1]
    return xlim, ylim #,dx,dy
end


#try using strided to optimize this
function compute_psi(state::S, x_grid, y_grid; inside_only=true, memory_limit = 10.0e9, multithreaded = true) where {S<:AbsState}
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
            B = basis_matrix(basis, k, pts; multithreaded)
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
                #Psi[.~pts_mask] .= NaN
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

function wavefunction(state::S; b=5.0, inside_only=true, fundamental_domain = true, memory_limit = 10.0e9, multithreaded = true) where {S<:AbsState}
    let k = state.k, billiard=state.billiard, symmetries=state.basis.symmetries, sym_qnumbers=state.basis.sym_qnumbers       
        #println(new_basis.dim)
        type = eltype(state.vec)
        #try to find a lazy way to do this
        L = CompositeCurve(get_boundary_curves(billiard)).length
        
        xlim,ylim = boundary_limits(get_boundary_curves(billiard); grd=max(1000,round(Int, k*L*b/(2*pi))))
        dx = xlim[2] - xlim[1]
        dy = ylim[2] - ylim[1]
        nx = max(round(Int, k*dx*b/(2*pi)), 512)
        ny = max(round(Int, k*dy*b/(2*pi)), 512)
        x_grid::Vector{type} = collect(type,range(xlim... , nx))
        y_grid::Vector{type} = collect(type,range(ylim... , ny))
        Psi::Vector{type} = compute_psi(state,x_grid,y_grid; inside_only, memory_limit, multithreaded) 
        #println("Psi type $(eltype(Psi)), $(memory_size(Psi))")
        Psi2d::Array{type,2} = reshape(Psi, (nx,ny))
        if ~fundamental_domain 
            if ~isnothing(symmetries)
                Psi2d, x_grid, y_grid = symmetrize_wavefunction(Psi2d,x_grid,y_grid,symmetries,sym_qnumbers)
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
function compute_psi(state_bundle::S, x_grid, y_grid; inside_only=true, memory_limit = 10.0e9, multithreaded = true) where {S<:EigenstateBundle}
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
            B = basis_matrix(basis, k, pts; multithreaded)
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

function wavefunction(state_bundle::S; b=5.0, inside_only=true, fundamental_domain = true, memory_limit = 10.0e9, multithreaded = true) where {S<:EigenstateBundle}
    let k = state_bundle.k_basis, billiard=state_bundle.billiard, symmetries=state_bundle.basis.symmetries, sym_qnumbers=state_bundle.basis.sym_qnumbers          
        #println(new_basis.dim)
        type = eltype(state_bundle.X)
        #try to find a lazy way to do this
        L = CompositeCurve(get_boundary_curves(billiard)).length
        xlim,ylim = boundary_limits(get_all_curves(billiard); grd=max(1000,round(Int, k*L*b/(2*pi))))
        dx = xlim[2] - xlim[1]
        dy = ylim[2] - ylim[1]
        nx = max(round(Int, k*dx*b/(2*pi)), 512)
        ny = max(round(Int, k*dy*b/(2*pi)), 512)
        x_grid::Vector{type} = collect(type,range(xlim... , nx))
        y_grid::Vector{type} = collect(type,range(ylim... , ny))
        Psi_bundle::Matrix{type} = compute_psi(state_bundle,x_grid,y_grid;inside_only=inside_only, memory_limit = memory_limit, multithreaded=multithreaded) 
        #println("Psi type $(eltype(Psi)), $(memory_size(Psi))")
        Psi2d::Vector{Array{type,2}} = [reshape(Psi, (nx,ny)) for Psi in eachcol(Psi_bundle)]
        if ~fundamental_domain 
            if ~isnothing(symmetries)
                for i in eachindex(Psi2d)
                    Psi_new, x_grid, y_grid = symmetrize_wavefunction(Psi2d[i],x_grid,y_grid,symmetries,sym_qnumbers)
                    Psi2d[i] = Psi_new
                end
            end
        end
        return Psi2d, x_grid, y_grid
    end
end


