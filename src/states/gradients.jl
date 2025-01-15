

"""
    compute_grad(state::S,x_grid,y_grid ;inside_only=true) where {S<:AbsState}
``
Testing function for checking the gradient of a wavefunction on a grid `x_grid * y_grid`. Internally call `basis_and_gradient` function.

# Arguments
- `state<:AbsState`: An instance of `AbsState` containing the wavefunction data (the vector of expansion coefficients in the given basis, the basis chosen, the billiard geometry...)
- `x_grid<:Vector`: A 1D array representing the x-coordinates of the grid points.
- `y_grid<:Vector`: A 1D array representing the y-coordinates of the grid points.
- `inside_only=true`: Whether to calculate the gradient only in interior of the boundary.

# Returns
- `dX<:Vector`: A 1D array representing the gradient of the wavefunction with respect to the x-coordinates.
- `dY<:Vector`: A 1D array representing the gradient of the wavefunction with respect to the y-coordinates.
"""
function compute_grad(state::S,x_grid,y_grid ;inside_only=true) where {S<:AbsState}
    let vec=state.vec,k=state.k,basis=basis,eps=state.eps,basis=state.basis,billiard=state.billiard
        pts=collect(SVector(x,y) for y in y_grid for x in x_grid)
        dX=zeros(eltype(vec),length(pts))
        dY=zeros(eltype(vec),length(pts))
        if inside_only
            pts_mask=is_inside(billiard,x_grid,y_grid)
            pts=pts[pts_mask]
            for i in eachindex(vec)
                if abs(vec[i])>eps 
                    _,dx,dy=basis_and_gradient(basis,i,k,pts)
                    dX[pts_mask].+=vec[i].*dx
                    dY[pts_mask].+=vec[i].*dy
                end
            end
        else
            for i in eachindex(vec)
                if abs(vec[i])>eps 
                    _,dx,dy=basis_and_gradient(basis,i,k,pts)
                    dX.+=vec[i].*dx
                    dY.+=vec[i].*dy
                end
            end
        end
        return dX,dY
    end
end

"""
    wavefunction_gradient(state::S;b=20.0,inside_only=true) where {S<:AbsState}

High level function that calculates the wavefunction for a given state. It calls internally the compute_grad function and reshapes the x and y gradient vectors into a Matrix for heatmap plotting.

# Arguments
- `state<:AbsState`: An instance of `AbsState` containing the wavefunction data (the vector of expansion coefficients in the given basis, the basis chosen, the billiard geometry...)
- `inside_only`: If true, the wavefunction gradient will be computed only for the interior of the boundary.
- b=20.0: The points scaling factor that determines the grid density (the resolution of the grid, higher is more refined grid)

# Returns
- `dX2d<:Array{<:Real,2}}`: A 2D array representing the gradient of the wavefunction with respect to the x-coordinate.
- `dY2d<:Array{<:Real,2}}`: A 2D array representing the gradient of the wavefunction with respect to the y-coordinate.
- `x_grid::Vector`: The grid the Matrix is constructed on, useful for x axis in heatmap plotting etc.
- `y_grid::Vector`: The grid the Matrix is constructed on, useful for y axis in heatmap plotting etc.
"""
function wavefunction_gradient(state::S;b=20.0,inside_only=true) where {S<:AbsState}
    let type=eltype(state.vec),billiard=state.billiard
        k=state.k_basis
        #try to find a lazy way to do this
        L=billiard.length
        xlim,ylim=boundary_limits(billiard.fundamental_boundary; grd=round(Int,k*L*b/(2*pi)),type=type)
        dx=xlim[2]-xlim[1]
        dy=ylim[2]-ylim[1]
        nx=max(round(Int,b),10)
        ny=max(round(Int,b*dy/dx),10)
        x_grid::Vector{type}=collect(type,range(xlim... ,nx))
        y_grid::Vector{type}=collect(type,range(ylim... ,ny))
        dX::Vector{type},dY::Vector{type}=compute_grad(state,x_grid,y_grid;inside_only=inside_only) 
        dX2d::Array{type,2}=reshape(dX,(nx,ny))
        dY2d::Array{type,2}=reshape(dY,(nx,ny))
        return dX2d,dY2d,x_grid,y_grid
    end
end
