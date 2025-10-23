"""
    compute_grad(state::S,x_grid,y_grid ;inside_only=true) where {S<:AbsState}
``
Testing function for checking the gradient of a wavefunction on a grid `x_grid * y_grid`. Internally call `basis_and_gradient` function.

# Arguments
- `state<:AbsState`: An instance of `AbsState` containing the wavefunction data (the vector of expansion coefficients in the given basis, the basis chosen, the billiard geometry...)
- `x_grid<:Vector`: A 1D array representing the x-coordinates of the grid points.
- `y_grid<:Vector`: A 1D array representing the y-coordinates of the grid points.
- `inside_only=true`: Whether to calculate the gradient only in interior of the boundary.
- `fundamental=true`: If do just the fundamental domain. Useful if doing the Vergini-Saraceno method or a method where we get basis coefficients.

# Returns
- `dX<:Vector`: A 1D array representing the gradient of the wavefunction with respect to the x-coordinates.
- `dY<:Vector`: A 1D array representing the gradient of the wavefunction with respect to the y-coordinates.
"""
function compute_grad(state::S,x_grid,y_grid;inside_only::Bool=true,fundamental::Bool=true) where {S<:AbsState}
    state_vec=state.vec;k=state.k;basis=state.basis
    billiard=state.billiard;eps=state.eps
    nx=length(x_grid);ny=length(y_grid)
    pts_mat=[SVector(x,y) for y in y_grid, x in x_grid]
    pts_all=vec(pts_mat)
    mask_vec=inside_only ? points_in_billiard_polygon(pts_all,billiard,round(Int,sqrt(length(pts_all)));fundamental_domain=fundamental) : fill(true,nx*ny)
    pts=pts_all[mask_vec];npts=length(pts)
    idxs=findall(i->abs(state_vec[i])>eps,eachindex(state_vec))
    isempty(idxs) && throw(ArgumentError("All basis vector coefficients are below numerical threshold!"))
    @blas_1 dBdx,dBdy=gradient(basis,idxs,k,pts;multithreaded=threaded)
    w=state_vec[idxs] # coefficients could be complex, coupled with a real basis -> associated with >1d irreps
    RT=promote_type(eltype(dBdx),eltype(w)) # need to promote matrix type of real basis to complex
    dX_in=zeros(RT,npts);dY_in=zeros(RT,npts)
    @blas_multi MAX_BLAS_THREADS mul!(dX_in,dBdx,w)
    @blas_multi_then_1 MAX_BLAS_THREADS mul!(dY_in,dBdy,w)
    DX_vec=zeros(RT,ny*nx)
    DY_vec=zeros(RT,ny*nx)
    @views DX_vec[mask_vec].=dX_in
    @views DY_vec[mask_vec].=dY_in
    DX=reshape(DX_vec,ny,nx)
    DY=reshape(DY_vec,ny,nx)
    return DX,DY
end

"""
    wavefunction_gradient(state::S;b=20.0,inside_only=true) where {S<:AbsState}

High level function that calculates the wavefunction for a given state. It calls internally the compute_grad function and reshapes the x and y gradient vectors into a Matrix for heatmap plotting.

# Arguments
- `state<:AbsState`: An instance of `AbsState` containing the wavefunction data (the vector of expansion coefficients in the given basis, the basis chosen, the billiard geometry...)
- `inside_only`: If true, the wavefunction gradient will be computed only for the interior of the boundary.
- b=20.0: The points scaling factor that determines the grid density (the resolution of the grid, higher is more refined grid)
- `fundamental=true`: If do just the fundamental domain. Useful if doing the Vergini-Saraceno method or a method where we get basis coefficients.

# Returns
- `dX2d<:Array{<:Real,2}}`: A 2D array representing the gradient of the wavefunction with respect to the x-coordinate.
- `dY2d<:Array{<:Real,2}}`: A 2D array representing the gradient of the wavefunction with respect to the y-coordinate.
- `x_grid::Vector`: The grid the Matrix is constructed on, useful for x axis in heatmap plotting etc.
- `y_grid::Vector`: The grid the Matrix is constructed on, useful for y axis in heatmap plotting etc.
"""
function wavefunction_gradient(state::S;b=20.0,inside_only=true,fundamental::Bool=true) where {S<:AbsState}
    type=eltype(state.vec)
    billiard=state.billiard
    k=state.k_basis
    L=billiard.length
    xlim,ylim=boundary_limits(billiard.fundamental_boundary;grd=round(Int,k*L*b/(2*pi)))
    dx=xlim[2]-xlim[1]
    dy=ylim[2]-ylim[1]
    nx=max(round(Int,b),10)
    ny=max(round(Int,b*dy/dx),10)
    x_grid::Vector{type}=collect(type,range(xlim...,nx))
    y_grid::Vector{type}=collect(type,range(ylim...,ny))
    dX2d,dY2d=compute_grad(state,x_grid,y_grid;inside_only=inside_only,fundamental=fundamental) 
    return dX2d,dY2d,x_grid,y_grid
end
