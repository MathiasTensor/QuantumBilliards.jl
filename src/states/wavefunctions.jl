using StaticArrays, JLD2, ProgressMeter







## NEW ##

"""
    symmetrize_Psi(Psi2d::Matrix, x_grid::Vector, y_grid::Vector, billiard::Bi, symmetries::Union{Nothing,Vector{Any}}, fundamental_domain::Bool) where {Bi<:AbsBilliard}

Implements symmmetries of the system into the wavefunction Matrix and it's grids and forms new instances of them that are fully symmetrized.

# Arguments
- `Psi2d::Matrix`: 2D wavefunction matrix
- `x_grid::Vector`: x-axis grid
- `y_grid::Vector`: y-axis grid
- `billiard::Bi`: Billiard type
- `symmetries::Union{Nothing,Vector{Any}}`: Vector of symmetries to apply (either `Reflection` or `Rotation` or `nothing`)
- `fundamental_domain::Bool`: Whether to apply symmetries to the fundamental domain only

# Returns
- `Psi2d::Matrix`: Symmetrized 2D wavefunction matrix
- `x_grid::Vector`: Symmetrized x-axis grid
- `y_grid::Vector`: Symmetrized y-axis grid
"""
function symmetrize_Psi(Psi2d::Matrix, x_grid::Vector, y_grid::Vector, billiard::Bi, symmetries::Union{Nothing,Vector{Any}}, fundamental_domain::Bool) where {Bi<:AbsBilliard}
    if ~fundamental_domain 
        if ~isnothing(symmetries)
            if all([sym isa Reflection for sym in symmetries])
                x_axis = 0.0
                y_axis = 0.0
                if hasproperty(billiard, :x_axis)
                    x_axis = billiard.x_axis
                end
                if hasproperty(billiard, :y_axis)
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

"""
    symmetrize_Psi(Psi2d::Vector{Matrix}, x_grid::Vector, y_grid::Vector, billiard::Bi, symmetries::Union{Nothing,Vector{Any}}, fundamental_domain::Bool) where {Bi<:AbsBilliard}

Implements symmmetries of the system into the wavefunction matrices and it's grids and forms new instances of them that are fully symmetrized. This is compatible with the functions that have `EigenstateBundle` as input.

# Arguments
- `Psi2d::Vector{Matrix}`: 2D wavefunction matrices
- `x_grid::Vector`: x-axis grid
- `y_grid::Vector`: y-axis grid
- `billiard::Bi`: Billiard type
- `symmetries::Union{Nothing,Vector{Any}}`: Vector of symmetries to apply (either `Reflection` or `Rotation` or `nothing`)
- `fundamental_domain::Bool`: Whether to apply symmetries to the fundamental domain only

# Returns
- `Psi2d::Vector{Matrix}`: Symmetrized 2D wavefunction matrices in the bundle
- `x_grid::Vector`: Symmetrized x-axis grid for the whole bundle
- `y_grid::Vector`: Symmetrized y-axis grid for the whole bundle
"""
function symmetrize_Psi(Psi2d::Vector, x_grid::Vector, y_grid::Vector, billiard::Bi, symmetries::Union{Nothing,Vector{Any}}, fundamental_domain::Bool) where {Bi<:AbsBilliard}
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

"""
    billiard_polygon(billiard::Bi, N_polygon_checks::Int; fundamental_domain=true) :: Vector where {Bi<:AbsBilliard}

Given <:AbsBilliard object, computes the points on the boundary of the billiard that are equidistant from each other, with the total number of points being N_polygon_checks.
    
# Arguments
- `billiard`: A billiard object with a fundamental_boundary or full_boundary field.
- `N_polygon_checks`: The total number of points to be distributed along the boundary.
- `fundamental_domain::Bool=true`: A flag indicating whether to compute points on the fundamental or full boundary.

# Returns
- `Vector{Vector{SVector{2,<:Real}}}`: For each crv in the billiard boundary (chosen by the flag fundamental_domain) form a Vector{SVector{2,<:Real}} object containing the discretization points for that curve. 
"""
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
"""
    is_left(p1::SVector{2,T}, p2::SVector{2,T}, pt::SVector{2,T}) where {T<:Real}

Determines whether the point `pt` is to the left of the line segment defined by `p1` and `p2`.

# Arguments
- `p1::SVector{2,T}`: The first point defining the line segment.
- `p2::SVector{2,T}`: The second point defining the line segment.
- `pt::SVector{2,T}`: The point to check.

# Returns
- `T`: + or - value depending if the point is to the left or right of the line segment.
"""
function is_left(p1::SVector{2,T}, p2::SVector{2,T}, pt::SVector{2,T}) where {T<:Real}
    return (p2[1] - p1[1]) * (pt[2] - p1[2]) - (pt[1] - p1[1]) * (p2[2] - p1[2])
end


# Winding number algorithm to check if a point is inside a polygon
"""
    is_point_in_polygon(polygon::Vector{SVector{2,T}}, point::SVector{2,T})::Bool where T

Determines whether a single `point` is inside a billiard `polygon` formed by it's boundary points. It implements a winding number algorithm for the checking.

# Arguments
- `polygon::Vector{SVector{2,T}}`: A vector of points representing the boundary of the polygon.
- `point::SVector{2,T}`: A point to check if it's inside the polygon.

# Returns
- `Bool`: `true` if the point is inside the polygon, `false` otherwise.
"""
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

"""
    points_in_billiard_polygon(pts::Vector{SVector{2,T}}, billiard::Bi, N_polygon_checks::Int; fundamental_domain=true) where {T<:Real,Bi<:AbsBilliard}

Determines whether the `pts` are in the billiard polygon formed from `N_polygon_check` points.

# Arguments
- `pts::Vector{SVector{2,T}}`: A vector of points to check.
- `billiard::Bi`: An `AbsBilliard` struct representing the billiard.
- `N_polygon_checks::Int`: The number of points to sample for the entire billiard polygon.
- `fundamental_domain::Bool=true`: If `true`, use the fundamental domain for the billiard polygon.

# Returns
- `Vector{Bool}`: A vector of `true` if the corresponding point in `pts` is in the billiard polygon, `false` otherwise.
"""
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

"""
    compute_psi(state::S, x_grid, y_grid; inside_only=true, memory_limit = 10.0e9) where {S<:AbsState}

Constructs the wavefunction as a `Matrix` from an `Eigenstate` struct on a grid of vectors `x_grid` and `y_grid`.

# Arguments
- `state::S`: An `Eigenstate` struct with a `vec` field representing the wavefunction, a `k_basis` field representing the wavefunction basis, a `basis` field representing the basis set, a `billiard` field representing the billiard.
- `x_grid::Vector`: A vector of `x` coordinates on which to evaluate the wavefunction.
- `y_grid::Vector`: A vector of `y` coordinates on which to evaluate the wavefunction.
- `inside_only::Bool` (optional, default `true`): If `true`, only evaluate the wavefunction inside the billiard.
- `memory_limit::Real` (optional, default `10.0e9`): The maximum memory limit in bytes to use for constructing the wavefunction. If the memory required exceeds this multithreading is disabled.

# Returns
- `Psi::Matrix`: A `Matrix` representing the wavefunction evaluated on the grid.
"""
function compute_psi(state::S, x_grid::Vector, y_grid::Vector; inside_only=true, memory_limit = 10.0e9) where {S<:AbsState}
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


# INTERNAL FOR COMPUTING WAVEFUNCTIONS FROM STATE DATA WHERE WE TAKE THE ONLY THE VECTOR OF COEFFICIENTS OF THE LINEAR EXPANSIONS AND THE K EIGENVALUE
"""
    compute_psi(vec::Vector, k::T, billiard::Bi, basis::Ba, x_grid, y_grid; inside_only=true, memory_limit = 10.0e9) where {Bi<:AbsBilliard, Ba<:AbsBasis, T<:Real}

Computs the wavefunction as a `Matrix` on a grid formed by the vectors `x_grid` and `y_grid`. This is a lower level function for wrappers that require the construction of a wavefunction from a vector of linear expansion coefficients and being constructed on a commom grid.

# Arguments
- `vec::Vector{T}`: A vector of coefficients representing the linear expansion coefficients of the wavefunction.
- `k::T`: The k-eigenvalue at which the wavefunction is evaluated.
- `billiard<:AbsBilliard`: An instance of the abstract billiard type representing the physical billiard.
- `basis<:AbsBasis`: An instance of the abstract basis type representing the linear expansion basis.
- `x_grid::Vector{T}`: A vector of x-coordinates at which the wavefunction should be evaluated.
- `y_grid::Vector{T}`: A vector of y-coordinates at which the wavefunction should be evaluated.
# Keyword arguments
- `inside_only::Bool=true`: If true, only points inside the billiard are considered for evaluation.
- `memory_limit=10.0e9`: A limit on the memory usage for the computation in bytes. If the memory usage exceeds this limit, multithreading is disabled for the matrix construction.

# Returns
- `Psi::Matrix{T}`: A matrix representing the wavefunction evaluated on the grid formed by the vectors x_grid and y_grid.
"""
function compute_psi(vec::Vector, k::T, billiard::Bi, basis::Ba, x_grid::Vector, y_grid::Vector; inside_only=true, memory_limit = 10.0e9) where {Bi<:AbsBilliard, Ba<:AbsBasis, T<:Real}
    eps = set_precision(vec[1])
    dim = length(vec) # only way to get dim of basis implicitely
    dim = rescale_rpw_dimension(basis, dim) # hack for rpw
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

# MAIN ONE USED, JUST HAS ADDITIONAL LOGIC INTRODUCED FOR SYMMETRIES
"""
    wavefunction(state::S; b=5.0, inside_only=true, fundamental_domain = true, memory_limit = 10.0e9) where {S<:AbsState}

Constructs the wavefunction from a given state object (like Eigenstate).

# Arguments
- `state::S`: An instance of the abstract state type representing the state from which the wavefunction should be constructed.
- `b::Float64=5.0`: A scaling factor for the billiard size.
- `inside_only::Bool=true`: If true, only points inside the billiard are considered for evaluation.
- `fundamental_domain::Bool=true`: If true, the wavefunction is computed on the fundamental domain of the billiard.
- `memory_limit=10.0e9`: A limit on the memory usage for the computation in bytes. If the memory usage exceeds this limit, multithreading is disabled for the construction.

# Returns
- `Psi2d::Array{T,2}`: A 2D array representing the wavefunction evaluated on the grid formed by the vectors x_grid and y_grid.
- `x_grid::Vector{T}`: A Vector of x values where the matrix was evaluated.
- `y_grid::Vector{T}`: A Vector of y values where the matrix was evaluated.
"""
function wavefunction(state::S; b=5.0, inside_only=true, fundamental_domain = true, memory_limit = 10.0e9) where {S<:AbsState}
    let k = state.k, billiard=state.billiard, symmetries=state.basis.symmetries       
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
        Psi2d::Array{type,2} = reshape(Psi, (nx,ny))
        Psi2d, x_grid, y_grid = symmetrize_Psi(Psi2d, x_grid, y_grid, billiard, symmetries, fundamental_domain)
        return Psi2d, x_grid, y_grid
    end
end

# INTERNAL
file_extension(file::String) = file[findlast(==('.'), file)+1:end]
"""
    save_vec_from_StateData(state_data::StateData, file_path::String)

Saves the X matrix into a a .jld2 file with ks for convenience.

# Arguments
- `state_data::StateData`: The `StateData` containing the eigenvalues (ks) and the X matrix that contains the basis expansion coefficients.
- `file_path::String`: The file path where the data will be saved.

# Returns
- `Nothing`: If the file is successfully saved.
"""
function save_vec_from_StateData!(state_data::StateData, file_path::String)
    ks = state_data.ks
    X = state_data.X
    @assert file_extension(file_path) == "jld2" "Must be a .jld2 file format"
    @assert length(ks) == length(X) "The number of eigenvalues should be the same as the number of columns in X"
    @save file_path ks X
end

"""
    load_vec_from_file(file_path::String)

Loads the saved ks and X matrix from a string file.

# Arguments
- `file_path::String`: The file path from which the data will be loaded.

# Returns
- `ks::Vector{<:Real}`: The vector of eigenvalues (wavenumbers).
- `X::Matrix{<:Real}`: The matrix containing the basis expansion coefficients.
"""
function load_vec_from_file(file_path::String)
    ks = nothing
    X = nothing
    @load file_path ks X
    @assert ks !== nothing && X !== nothing "Failed to load ks or X from file"
    return ks, X
end

# IN DEVELOPEMNT
# USEFUL FOR CONSTRUCTING LARGE NUMBER OF WAVEFUNCTIONS FROM StateData (saved ks[i], X[i], basis, billiard)
"""
    wavefunction(vec::Vector, k::T, billiard::Bi, basis::Ba; b=5.0, inside_only=true, fundamental_domain = true, memory_limit = 10.0e9) where {Bi<:AbsBilliard, Ba<:AbsBasis, T<:Real}     

Computes the wavefunction matrix and the x and y grids for heatmap plotting. It is contructed from the vec=X[i] of `StateData` and not directly from `StateData`.

# Arguments
- `vec::Vector{<:Real}`: The vector of coefficients of the basis expansion of the wavefunction. It's length determines the resizeing of the `basis`.ž
- `k<:Real`: The wavenumber for that vec = X[i].
- `billiard<:AbsBilliard`: The billiard geometry.
- `basis<:AbsBasis`: The basis used for constructing the wavefunction from `vec`. Must be the same as the one used for constructing `vec`.
- `b`: The point scalling factor. Default is 5.0.
- `inside_only::Bool`: If true, only the points inside the billiard are considered. Default is true.
- `fundamental_domain::Bool`: If true, the wavefunction information is only constructed in the fundamental domain. Default is true.
- `memory_limit`: The maximum amount of memory (in bytes) for constructing the wavefunction with julia broadcasting operations and the use of the `basis_matrix`. Otherwise we use the `basis_fun` directly. Default is 10.0e9.

# Returns
- `Psi2ds::Vector{Matrix}`: A vector of `Matrix` containing the wavefunction for each k in ks.
- `x_grids::Vector{Vector}`: A vector of `Vector` containing the x grid for each k in ks.
- `y_grids::Vector{Vector}`: A vector of `Vector` containing the y grid for each k in ks.
"""
function wavefunction(vec::Vector, k::T, billiard::Bi, basis::Ba; b=5.0, inside_only=true, fundamental_domain = true, memory_limit = 10.0e9) where {Bi<:AbsBilliard, Ba<:AbsBasis, T<:Real}     
    dim = length(vec)
    println("Starting dim: ", dim)
    dim = rescale_rpw_dimension(basis, dim)
    println("Rescaled rpw dim: ", dim)
    basis = resize_basis(basis, billiard, dim, k)
    println("New basis dim: ", dim)
    symmetries = basis.symmetries
    type = eltype(vec)
    L = billiard.length
    xlim,ylim = boundary_limits(billiard.fundamental_boundary; grd=max(1000,round(Int, k*L*b/(2*pi))))
    dx = xlim[2] - xlim[1]
    dy = ylim[2] - ylim[1]
    nx = max(round(Int, k*dx*b/(2*pi)), 512)
    ny = max(round(Int, k*dy*b/(2*pi)), 512)
    x_grid::Vector{type} = collect(type,range(xlim... , nx))
    y_grid::Vector{type} = collect(type,range(ylim... , ny))
    Psi::Vector{type} = compute_psi(vec,k,billiard,basis,x_grid,y_grid; inside_only=inside_only, memory_limit=memory_limit) 
    Psi2d::Array{type,2} = reshape(Psi, (nx,ny))
    Psi2d, x_grid, y_grid = symmetrize_Psi(Psi2d, x_grid, y_grid, billiard, symmetries, fundamental_domain)
    return Psi2d, x_grid, y_grid
end

"""
    wavefunctions(X::Vector, ks::Vector, billiard::Bi, basis::Ba; b=5.0, inside_only=true, fundamental_domain = true, memory_limit = 10.0e9) where {Bi<:AbsBilliard, Ba<:AbsBasis, T<:Real}

High level wrapper for moer efficiently computing wavefunction matrices and the grids for plotting.

# Arguments
- `X::Vector`: A vector of coefficients of the basis expansion of the wavefunction for each k in ks.
- `ks::Vector`: A vector of wavenumbers for which to compute the wavefunction.
- `billiard::Bi`: An object representing the billiard.
- `basis::Ba`: An object representing the basis (rpw, cafb...).
- `b`: The point scalling factor. Default is 5.0.
- `inside_only::Bool`: If true, only the points inside the billiard are considered. Default is true.
- `fundamental_domain::Bool`: If true, the wavefunction is only constructed in the fundamental domain. Default is true.
- `memory_limit`: The maximum amount of memory (in bytes) for constructing the wavefunction with julia broadcasting operations and the use of the `basis_matrix`. Otherwise we use the `basis_fun` directly. Default is 10.0e9.

# Returns
- `vec_Psi::Vector{Matrix}`: A vector of `Matrix` containing the wavefunction for each k in ks.
- `vec_xs::Vector{Vector}`: A vector of `Vector` containing the x grid for each k in ks.
- `vec_ys::Vector{Vector}`: A vector of `Vector` containing the y grid for each k in ks.
"""
function wavefunctions(X::Vector, ks::Vector, billiard::Bi, basis::Ba; b=5.0, inside_only=true, fundamental_domain = true, memory_limit = 10.0e9) where {Bi<:AbsBilliard, Ba<:AbsBasis}
    vec_Psi = Vector{Matrix}(undef, length(ks))
    vec_xs = Vector{Vector}(undef, length(ks))
    vec_ys = Vector{Vector}(undef, length(ks))
    p = Progress(length(ks), 1)
    Threads.@threads for i in eachindex(ks) 
        vec = X[i]
        k = ks[i]
        Psi2d, x_grid, y_grid = wavefunction(vec, k, billiard, basis; b=b, inside_only=inside_only, fundamental_domain=fundamental_domain, memory_limit=memory_limit)
        vec_Psi[i] = Psi2d
        vec_xs[i] = x_grid
        vec_ys[i] = y_grid
        next!(p)
    end
    return vec_Psi, vec_xs, vec_ys
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
        dim = rescale_rpw_dimension(basis, dim)
        new_basis = resize_basis(basis, billiard, dim, ks[i])
        state = Eigenstate(ks[i], vec, tens[i], new_basis, billiard)
        Psi2d, x_grid, y_grid = wavefunction(state; b=b, inside_only=inside_only, fundamental_domain=fundamental_domain, memory_limit=memory_limit)
        Psi2ds[i] = Psi2d
        x_grids[i] = x_grid
        y_grids[i] = y_grid
    end
    return ks, Psi2ds, x_grids, y_grids
end

"""
    wavefunction(state::BasisState; xlim =(-2.0,2.0), ylim=(-2.0,2.0), b=5.0) 

Construct the wavefunction for a given basis function defined from a `BasisState` object. It is useful for visualizing the varius basis functions in the chosen basis.

# Arguments
- `state::BasisState`: An object representing the basis function.
- `xlim::Tuple{Float64,Float64}`: The range of x values for the wavefunction. Default is `(-2.0, 2.0)`.
- `ylim::Tuple{Float64,Float64}`: The range of y values for the wavefunction. Default is `(-2.0, 2.0)`.
- `b::Float64`: The point scalling factor. Default is 5.0.

# Returns
- `Psi2d::Array{Float64,2}`: The 2D wavefunction matrix for the given basis matrix.
- `x_grid::Vector{<:Real}`: The x grid formed from the `xlim`.
- `y_grid::Vector{<:Real}`: The y grid formed from the `ylim`.
"""
function wavefunction(state::BasisState; xlim =(-2.0,2.0), ylim=(-2.0,2.0), b=5.0) 
    let k = state.k, basis=state.basis      
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
        Psi2d::Array{type,2} = reshape(Psi, (nx,ny))
        return Psi2d, x_grid, y_grid
    end
end

#this can be optimized
"""
    compute_psi(state_bundle::S, x_grid, y_grid; inside_only=true, memory_limit = 10.0e9) where {S<:EigenstateBundle}

Computs the wavefunction Matrix on an x_grid and y_grid from an EigenstateBundle object. All the matrices in the state bundle are computed on the same grid.

# Arguments
- `state_bundle::S`: An object representing the bundle of eigenstate.
- `x_grid::Vector{<:Real}`: A vector representing the x grid.
- `y_grid::Vector{<:Real}`: A vector representing the y grid.
- `inside_only::Bool`: If true, only the points inside the billiard are considered. Default is true.
- `memory_limit`: The maximum amount of memory (in bytes) for constructing the wavefunction with julia broadcasting operations and the use of the `basis_matrix`. Otherwise we use the non-multithread implementation.

# Returns
- `Psi_bundle::Matrix{<:Real}`: A matrix containing the wavefunction for each state in the bundle on the given grid.
"""
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
"""
    wavefunction(state_bundle::S; b=5.0, inside_only=true, fundamental_domain=true, memory_limit=10.0e9) where {S<:EigenstateBundle}

Construct the wavefunction matrices from an EigenstateBundle on a common grid. Useful for a smaller number of wavefunctions.

# Arguments
- `state_bundle::S`: An object representing the bundle of eigenstates.
- `b`: The point scalling factor. Default is 5.0.
- `inside_only::Bool`: If true, only the points inside the billiard are considered. Default is true.
- `fundamental_domain::Bool`: If true, the wavefunction is only constructed in the fundamental domain. Default is true.
- `memory_limit`: The maximum amount of memory (in bytes) for constructing the wavefunction with julia broadcasting operations and the use of the `basis_matrix`. Otherwise we use the non-multithreaded implementation.

# Returns
- `Psi2ds::Vector{Matrix{<:Real}}`: A vector of `Matrix` objects containing the wavefunction for each state in the bundle.
- `x_grid::Vector{<:Real}`: A vector of x grid points common for the entire bundle.
- `y_grid::Vector{<:Real}`: A vector of y grid points common for the entire bundle.
"""
function wavefunction(state_bundle::S; b=5.0, inside_only=true, fundamental_domain=true, memory_limit=10.0e9) where {S<:EigenstateBundle}
    let
        k = state_bundle.k_basis
        billiard = state_bundle.billiard
        symmetries = state_bundle.basis.symmetries
        T = Float64  # Ensure consistent type
        L = billiard.length
        xlim, ylim = boundary_limits(billiard.fundamental_boundary;grd=max(1000, round(Int, k * L * b / (2 * pi))))
        dx = xlim[2] - xlim[1]
        dy = ylim[2] - ylim[1]
        nx = max(round(Int, k * dx * b / (2 * pi)), 512)
        ny = max(round(Int, k * dy * b / (2 * pi)), 512)
        x_grid::Vector{T} = collect(range(xlim..., nx))
        y_grid::Vector{T} = collect(range(ylim..., ny))
        Psi_bundle::Matrix{Complex{T}} = compute_psi(state_bundle,x_grid,y_grid;inside_only=inside_only,memory_limit=memory_limit)
        # Reshape each column of Psi_bundle into a matrix
        Psi2d::Vector{Matrix{Complex{T}}} = [reshape(Psi, (nx, ny)) for Psi in eachcol(Psi_bundle)]
        Psi2d, x_grid, y_grid = symmetrize_Psi(Psi2d, x_grid, y_grid, billiard, symmetries, fundamental_domain)
        return Psi2d, x_grid, y_grid
    end
end

### INTERNAL - ONLY FOR OTOC WHERE WE NEED BOTH WAVEFUNCTION MATRICES OF SAME SIZE
"""
    wavefunction_fixed_grid_(state::S, nx::T, ny::T; b=5.0, memory_limit = 10.0e9) where {S<:AbsState, T<:Real}

Constructs the wavefunction on fixed x and y grids of size `nx` and `ny` respectively. The wavefunction `Matrix` itself is constructed from an `Eigenstate` struct. It is always constructed on the `fundamental_boundary`.

# Arguments
- `state::S`: The `Eigenstate` struct containing the wavefunction.
- `nx::T`: The number of grid points in the x direction.
- `ny::T`: The number of grid points in the y direction.
- `b::Real=5.0`: The point scaling factor. Default is 5.0.
- `memory_limit::Real=10.0e9`: The memory limit for multithreaded wavefunction construction. Default is `10.0e9`. If surpassed the `Matrix` is conctructed row wise in a non-multithreaded for loop.

# Returns
- `Psi::Matrix{T}`: The 2D wavefunction matrix on the fixed grid.
"""
function wavefunction_fixed_grid_(state::S, nx::T, ny::T; b=5.0, memory_limit = 10.0e9) where {S<:AbsState, T<:Real}
    let k = state.k, billiard=state.billiard, symmetries=state.basis.symmetries       
        type = eltype(state.vec)
        #try to find a lazy way to do this
        L = billiard.length
        xlim,ylim = boundary_limits(billiard.fundamental_boundary; grd=max(1000,round(Int, k*L*b/(2*pi))))
        x_grid::Vector{type} = collect(type,range(xlim... , nx))
        y_grid::Vector{type} = collect(type,range(ylim... , ny))
        Psi::Vector{type} = compute_psi(state,x_grid,y_grid; inside_only=true, memory_limit=memory_limit) 
        Psi2d::Array{type,2} = reshape(Psi, (nx,ny))
        Psi2d, x_grid, y_grid = symmetrize_Psi(Psi2d, x_grid, y_grid, billiard, symmetries, false)
        return Psi2d, x_grid, y_grid
    end
end


