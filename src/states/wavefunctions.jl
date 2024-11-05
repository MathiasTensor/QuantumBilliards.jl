using StaticArrays







## NEW ##


function billiard_polygon(billiard::Bi, N_polygon_checks::Int; fundamental_domain=true) :: Vector where {Bi<:AbsBilliard}
    if fundamental_domain
        boundary = billiard.fundamental_boundary
    else
        boundary = billiard.full_boundary
    end
    # Find the fraction of lengths wrt to the boundary
    billiard_composite_lengths = [crv.length for crv in boundary]
    typ = eltype(billiard_composite_lengths[1])
    total_billiard_length = sum(billiard_composite_lengths)
    billiard_length_fractions = [crv.length/total_billiard_length for crv in boundary]
    # Redistribute points based on the fractions
    distributed_points = [round(Int, fract*N_polygon_checks) for fract in billiard_length_fractions]
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


## NEW ##





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

# DO NOT USE
# INTERNAL FOR COMPUTING WAVEFUNCTIONS FROM STATE DATA WHERE WE TAKE THE ONLY THE VECTOR OF COEFFICIENTS OF THE LINEAR EXPANSIONS AND THE K EIGENVALUE
function compute_psi(vec::Vector, k::T, billiard::Bi, basis::Ba, x_grid, y_grid; inside_only=true, memory_limit = 10.0e9) where {Bi<:AbsBilliard, Ba<:AbsBasis, T<:Real}
    eps = set_precision(vec[1])
    dim = length(vec) # only way to get dim of basis implicitely
    basis = resize_basis(basis, billiard, dim, k) # since we dont have the solver for adjust basis and samplers
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
# MAIN ONE USED, JUST HAS ADDITIONAL LOGIC INTRODUCED FOR SYMMETRIES
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
                if all([sym isa Reflection for sym in symmetries])
                    x_axis = 0.0
                    y_axis = 0.0
                    if hasproperty(billiard, :x_axis)
                        #println(nameof(typeof(billiard)), " has the :x_axis reflection")
                        x_axis = billiard.x_axis
                    end
                    if hasproperty(billiard, :y_axis)
                        #println(nameof(typeof(billiard)), " has the :y_axis reflection")
                        y_axis = billiard.y_axis
                    end
                    Psi2d, x_grid, y_grid = reflect_wavefunction(Psi2d,x_grid,y_grid,symmetries; x_axis=x_axis, y_axis=y_axis)
                elseif all([sym isa Rotation for sym in symmetries])
                    if length(symmetries) > 1
                        @error "Only one Rotation symmetry allowed"
                    end
                    center = SVector{2, Float64}(0.0, 0.0)
                    if hasproperty(billiard, :center)
                        center = SVector{2, Float64}(billiard.center)
                    end
                    Psi2d, x_grid, y_grid = rotate_wavefunction(Psi2d,x_grid,y_grid,symmetries[1], billiard; center=center)
                else
                    @error "Do not mix Reflections with Rotations"
                end
            end
        end
        return Psi2d, x_grid, y_grid
    end
end

# IN DEVELOPEMNT
# USEFUL FOR CONSTRUCTING LARGE NUMBER OF WAVEFUNCTIONS FROM StateData (saved ks[i], X[i], basis, billiard)
function wavefunction(vec::Vector, k::T, billiard::Bi, basis::Ba; b=5.0, inside_only=true, fundamental_domain = true, memory_limit = 10.0e9) where {S<:AbsState}
    symmetries=state.basis.symmetries       
    basis = resize_basis(basis, billiard, length(vec), k)
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
    Psi::Vector{type} = compute_psi(vec,k,billiard,basis,x_grid,y_grid; inside_only=inside_only, memory_limit=memory_limit) 
    #println("Psi type $(eltype(Psi)), $(memory_size(Psi))")
    Psi2d::Array{type,2} = reshape(Psi, (nx,ny))
    if ~fundamental_domain 
        if ~isnothing(symmetries)
            if all([sym isa Reflection for sym in symmetries])
                x_axis = 0.0
                y_axis = 0.0
                if hasproperty(billiard, :x_axis)
                    #println(nameof(typeof(billiard)), " has the :x_axis reflection")
                    x_axis = billiard.x_axis
                end
                if hasproperty(billiard, :y_axis)
                    #println(nameof(typeof(billiard)), " has the :y_axis reflection")
                    y_axis = billiard.y_axis
                end
                Psi2d, x_grid, y_grid = reflect_wavefunction(Psi2d,x_grid,y_grid,symmetries; x_axis=x_axis, y_axis=y_axis)
            elseif all([sym isa Rotation for sym in symmetries])
                if length(symmetries) > 1
                    @error "Only one Rotation symmetry allowed"
                end
                center = SVector{2, Float64}(0.0, 0.0)
                if hasproperty(billiard, :center)
                    center = SVector{2, Float64}(billiard.center)
                end
                Psi2d, x_grid, y_grid = rotate_wavefunction(Psi2d,x_grid,y_grid,symmetries[1], billiard; center=center)
            else
                @error "Do not mix Reflections with Rotations"
            end
        end
    end
    return Psi2d, x_grid, y_grid
end

### NEW ONE THAT USES StateData to generate the wavefunctions and the X, Y grids
"""
    wavefunctions(state_data::StateData, billiard::Bi, basis::Ba; b=5.0, inside_only=true, fundamental_domain = true, memory_limit = 10.0e9) :: Tuple{Vector, Vector{Matrix}, Vector{Vector}, Vector{Vector}} where {Bi<:AbsBilliard, Ba<:AbsBasis}

High level wrapper for constructing the wavefunctions as a a `Tuple` of `Vector`s : `Tuple (ks::Vector, Psi2ds::Vector{Matrix}, x_grid::Vector{Vector}, y_grid::Vector{Vector})`.

# Arguments
- `state_data::StateData`: Object containing the wavenumbers, tensions and the coefficients of the wavefunction expansion as a vector of vectors for each k in ks.
- `billiard::Bi`: An object representing the billiard.
- `basis::Ba`: An object representing the basis (rpw, cafb...).
- `b`: The point scalling factor. Default is 5.0.
- `inside_only::Bool`: If true, only the points inside the billiard are considered. Default is true.
- `fundamental_domain::Bool`: If true, the wavefunction information is only constructed in the fundamental domain. Default is true.
- `memory_limit`: The maximum amount of memory (in bytes) for constructing the wavefunction with julia broadcasting operations and the use of the `basis_matrix`. Otherwise we use the `basis_fun` directly. Default is 10.0e9.

# Returns
- `ks::Vector{Float64}`: A vector of wavenumbers.
- `Psi2ds::Vector{Matrix}`: A vector of `Matrix` containing the wavefunction for each k in ks.
- `x_grids::Vector{Vector}`: A vector of `Vector` containing the x grid for each k in ks.
- `y_grids::Vector{Vector}`: A vector of `Vector` containing the y grid for each k in ks.
"""
function wavefunctions(state_data::StateData, billiard::Bi, basis::Ba; b=5.0, inside_only=true, fundamental_domain = true, memory_limit = 10.0e9) where {Bi<:AbsBilliard, Ba<:AbsBasis}
    ks = state_data.ks
    tens = state_data.tens
    X = state_data.X
    Psi2ds = Vector{Matrix{eltype(ks)}}(undef, length(ks))
    x_grids = Vector{Vector{eltype(ks)}}(undef, length(ks))
    y_grids = Vector{Vector{eltype(ks)}}(undef, length(ks))
    for i in eachindex(ks) 
        vec = X[i] # vector of vectors
        dim = length(vec)
        new_basis = resize_basis(basis, billiard, dim, ks[i])
        state = Eigenstate(ks[i], vec, tens[i], new_basis, billiard)
        Psi2d, x_grid, y_grid = wavefunction(state; b=b, inside_only=inside_only, fundamental_domain=fundamental_domain, memory_limit=memory_limit)
        Psi2ds[i] = Psi2d
        x_grids[i] = x_grid
        y_grids[i] = y_grid
    end
    return ks, Psi2ds, x_grids, y_grids
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

# NEW STATE BUNDLE ONE; NEEDS FURTHER TESTING
function wavefunction(state_bundle::S; b=5.0, inside_only=true, fundamental_domain=true, memory_limit=10.0e9) where {S<:EigenstateBundle}
    let
        k = state_bundle.k_basis
        billiard = state_bundle.billiard
        symmetries = state_bundle.basis.symmetries
        T = Float64  # Ensure consistent type

        L = billiard.length
        xlim, ylim = boundary_limits(
            billiard.fundamental_boundary;
            grd=max(1000, round(Int, k * L * b / (2 * pi)))
        )
        dx = xlim[2] - xlim[1]
        dy = ylim[2] - ylim[1]
        nx = max(round(Int, k * dx * b / (2 * pi)), 512)
        ny = max(round(Int, k * dy * b / (2 * pi)), 512)
        x_grid::Vector{T} = collect(range(xlim..., nx))
        y_grid::Vector{T} = collect(range(ylim..., ny))
        Psi_bundle::Matrix{Complex{T}} = compute_psi(
            state_bundle,
            x_grid,
            y_grid;
            inside_only=inside_only,
            memory_limit=memory_limit
        )
        # Reshape each column of Psi_bundle into a matrix
        Psi2d::Vector{Matrix{Complex{T}}} = [reshape(Psi, (nx, ny)) for Psi in eachcol(Psi_bundle)]
        if !fundamental_domain
            if !isnothing(symmetries)
                if all([sym isa Reflection for sym in symmetries])
                    # Handle reflections
                    x_axis = 0.0
                    y_axis = 0.0
                    if hasproperty(billiard, :x_axis)
                        x_axis = billiard.x_axis
                    end
                    if hasproperty(billiard, :y_axis)
                        y_axis = billiard.y_axis
                    end
                    for i in eachindex(Psi2d)
                        Psi_new, x_grid_new, y_grid_new = reflect_wavefunction(Psi2d[i], x_grid, y_grid, symmetries; x_axis=x_axis,y_axis=y_axis)
                        Psi2d[i] = Psi_new
                    end
                    x_grid = x_grid_new
                    y_grid = y_grid_new
                elseif all([sym isa Rotation for sym in symmetries])
                    println("We have a rotation")
                    if length(symmetries) > 1
                        @error "Only one Rotation symmetry allowed"
                    end
                    center = SVector{2, T}(0.0, 0.0)
                    if hasproperty(billiard, :center)
                        center = SVector{2, T}(billiard.center...)
                    end
                    # Apply rotation to each wavefunction
                    for i in eachindex(Psi2d)
                        Psi_new, x_grid_new, y_grid_new = rotate_wavefunction(Psi2d[i], x_grid, y_grid, symmetries[1], billiard; center=center)
                        Psi2d[i] = Psi_new
                    end
                    x_grid = x_grid_new
                    y_grid = y_grid_new
                else
                    @error "Do not mix Reflections with Rotations"
                end
            end
        end
        return Psi2d, x_grid, y_grid
    end
end



#= OLD ONE
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

=#


### INTERNAL - ONLY FOR OTOC WHERE WE NEED BOTH WAVEFUNCTION MATRICES OF SAME SIZE
function wavefunction_fixed_grid_(state::S, nx::T, ny::T; b=5.0, memory_limit = 10.0e9) where {S<:AbsState, T<:Real}
    let k = state.k, billiard=state.billiard, symmetries=state.basis.symmetries       
        #println(new_basis.dim)
        type = eltype(state.vec)
        #try to find a lazy way to do this
        L = billiard.length
        xlim,ylim = boundary_limits(billiard.fundamental_boundary; grd=max(1000,round(Int, k*L*b/(2*pi))))
        x_grid::Vector{type} = collect(type,range(xlim... , nx))
        y_grid::Vector{type} = collect(type,range(ylim... , ny))
        Psi::Vector{type} = compute_psi(state,x_grid,y_grid; inside_only=true, memory_limit=memory_limit) 
        #println("Psi type $(eltype(Psi)), $(memory_size(Psi))")
        Psi2d::Array{type,2} = reshape(Psi, (nx,ny))
        if ~false 
            if ~isnothing(symmetries)
                if all([sym isa Reflection for sym in symmetries])
                    x_axis = 0.0
                    y_axis = 0.0
                    if hasproperty(billiard, :x_axis)
                        #println(nameof(typeof(billiard)), " has the :x_axis reflection")
                        x_axis = billiard.x_axis
                    end
                    if hasproperty(billiard, :y_axis)
                        #println(nameof(typeof(billiard)), " has the :y_axis reflection")
                        y_axis = billiard.y_axis
                    end
                    Psi2d, x_grid, y_grid = reflect_wavefunction(Psi2d,x_grid,y_grid,symmetries; x_axis=x_axis, y_axis=y_axis)
                elseif all([sym isa Rotation for sym in symmetries])
                    if length(symmetries) > 1
                        @error "Only one Rotation symmetry allowed"
                    end
                    center = SVector{2, Float64}(0.0, 0.0)
                    if hasproperty(billiard, :center)
                        center = SVector{2, Float64}(billiard.center)
                    end
                    Psi2d, x_grid, y_grid = rotate_wavefunction(Psi2d,x_grid,y_grid,symmetries[1], billiard; center=center)
                else
                    @error "Do not mix Reflections with Rotations"
                end
            end
        end
        return Psi2d, x_grid, y_grid
    end
end


