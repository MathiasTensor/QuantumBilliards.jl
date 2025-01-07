# this will be an anti-scar criteria based on the PH functions as for the case of the mushroom billiard we have a set od distinct localizations in the PH plot that is associated with BB modes in the stem.

### CRITERIA CONSTRUCTION ###

"""
    calculate_bb_bbox_localization_mushroom(w::T, h::T, r::T) where {T<:Real}

Calculate the bounding box edges for PH localization regions in the mushroom billiard.
The p limits correspond to a 45° angle based on classical BB modes.

# Arguments
- `w`: Full width of the billiard.
- `h`: Height of the billiard.
- `r`: Radius of the semicircular top.

# Returns
Bounding box edges: `x0_1, x1_1, y0_1, y1_1, x0_2, x1_2, y0_2, y1_2`.
"""
function calculate_bb_bbox_localization_mushroom(w::T,h::T,r::T) where {T<:Real}
    # starting from the bottom left corner (the start of the foot anticlockwise)
    x0_1=w # right wall
    y0_1=-0.5 # the p-
    x1_1=w+h # end of right wall
    y1_1=0.5  # the p+
    x0_2=w+h+(r-w/2)+π*r+(r-w/2) # left wall
    y0_2=-0.5 # again the p-
    x1_2=w+h+(r-w/2)+π*r+(r-w/2)+h # end of left wall
    y1_2=0.5 # again the p+
    return x0_1,x1_1,y0_1,y1_1,x0_2,x1_2,y0_2,y1_2
end

"""
    create_husimi_localization_mat(x0_1::T, x1_1::T, y0_1::T, y1_1::T,
                                   x0_2::T, x1_2::T, y0_2::T, y1_2::T,
                                   x_grid::Vector{T}, y_grid::Vector{T}) where {T<:Real}

Create a Husimi localization matrix with `+1` in two rectangular regions and `-1` elsewhere.

# Arguments
- `x0_1, x1_1, y0_1, y1_1`: Edges of the first bounding box.
- `x0_2, x1_2, y0_2, y1_2`: Edges of the second bounding box.
- `x_grid, y_grid`: Grids for the Husimi matrix.

# Returns
A matrix of the same size as `x_grid` × `y_grid` with `+1` in the localization boxes.
"""
function create_husimi_localization_mat(x0_1::T,x1_1::T,y0_1::T,y1_1::T,x0_2::T,x1_2::T,y0_2::T,y1_2::T,x_grid::Vector{T},y_grid::Vector{T}) where {T<:Real}
    # create 2 boxes that have the value +1 and all the other ones have -1. Size of x_grid and y_grid should be the same as the H : size(H) = (length(x_grid),length(y_grid))
    mat=fill(-1.0,length(x_grid),length(y_grid))
    x_indices_box1=findall(x->x0_1<=x<=x1_1,x_grid) # Find indices for the first box
    y_indices_box1=findall(y->y0_1<=y<=y1_1,y_grid)
    x_indices_box2=findall(x->x0_2<=x<=x1_2,x_grid) # Find indices for the second box
    y_indices_box2=findall(y->y0_2<=y<=y1_2,y_grid)
    for i in x_indices_box1, j in y_indices_box1 # Set the values for box1 to +1
        mat[i,j]=1
    end
    for i in x_indices_box2, j in y_indices_box2 # Set the values for box2 to +1
        mat[i,j]=1
    end
    return mat
end

"""
    calculate_overlap(mat::Matrix, H::Matrix)

Calculate the overlap between the localization matrix and a Husimi matrix.

# Arguments
- `mat`: Localization matrix.
- `H`: Husimi matrix.

# Returns
Normalized overlap as a scalar.
"""
function calculate_overlap(mat::Matrix,H::Matrix)
    @assert size(mat)==size(H) "Size of mat and H must match"
    overlap=sum(mat.*H)*prod(size(H))
    return overlap
end

"""
    calculate_overlap(mat::Matrix, Hs::Vector{Matrix})

Compute the overlaps between the localization matrix and multiple Husimi matrices.

# Arguments
- `mat`: Localization matrix.
- `Hs`: Vector of Husimi matrices.

# Returns
Vector of normalized overlaps for each Husimi matrix.
"""
function calculate_overlap(mat::Matrix,Hs::Vector)
    overlaps=zeros(length(Hs))
    Threads.@threads for i in eachindex(Hs)
        overlaps[i]=calculate_overlap(mat,Hs[i])
    end
    return overlaps
end

"""
    get_bb_localization_indexes(Hs::Vector{Matrix}, x_grid::Vector{T}, y_grid::Vector{T},
                                w::T, h::T, r::T; threshold=0.8) where {T<:Real}

Find indices of Husimi matrices that exceed the overlap threshold with bounding box localization. This means that the wavefunction is one of the BB modes and therefore will con tribute to the scar density.

# Arguments
- `Hs`: Vector of Husimi matrices.
- `x_grid, y_grid`: Grids for the Husimi matrix.
- `w, h, r`: Dimensions of the mushroom billiard.
- `threshold`: Minimum overlap value to consider (default: `0.8`).
- `print_overlaps`: Print the overlaps for each Husimi matrix (default: `true`).

# Returns
Boolean mask where overlaps exceed the threshold.
"""
function get_bb_localization_indexes(Hs::Vector,x_grid::Vector{T},y_grid::Vector{T},w::T,h::T,r::T;threshold=0.8,print_overlaps=true) where {T<:Real}
    x0_1,x1_1,y0_1,y1_1,x0_2,x1_2,y0_2,y1_2=calculate_bb_bbox_localization_mushroom(w,h,r)
    mat=create_husimi_localization_mat(x0_1,x1_1,y0_1,y1_1,x0_2,x1_2,y0_2,y1_2,x_grid,y_grid)
    overlaps=calculate_overlap(mat,Hs)
    print_overlaps ? println(overlaps) : nothing
    idxs=findall(x->x>threshold,overlaps)
    tmp=falses(length(overlaps))
    tmp[idxs].=true
    return tmp
end

### CUMULATIVE DENSITIES ###

"""
    calculate_cumulative_density_scar_antiscar(Psi2ds::Vector{Matrix}, scar_idxs::Vector{Bool})

Compute cumulative densities for scarred and antiscarred wavefunctions.

# Arguments
- `Psi2ds`: Vector of squared wavefunction matrices.
- `scar_idxs`: Boolean vector indicating scarred wavefunctions. This is gotten from the `get_bb_localization_indexes` function.

# Returns
Two normalized matrices: `cumulative_density_scar` and `cumulative_density_antiscar`.
"""
function calculate_cumulative_density_scar_antiscar(Psi2ds::Vector, scar_idxs::Vector{Bool})
   Psi2ds_bbs=Psi2ds[scar_idxs] # Split Psi2ds into scar and antiscar categories
   Psi2ds_no_bbs=Psi2ds[.!scar_idxs] 
   cumulative_density_scar=zeros(size(Psi2ds_bbs[1]))
   cumulative_density_antiscar=zeros(size(Psi2ds_no_bbs[1]))
   function parallel_sum!(result::Matrix, matrices::Vector) # parallelization helper
       Threads.@threads for i in 1:length(matrices)
           @inbounds result.+=matrices[i]
       end
   end
   parallel_sum!(cumulative_density_scar,Psi2ds_bbs) # Compute cumulative densities in parallel
   parallel_sum!(cumulative_density_antiscar,Psi2ds_no_bbs)
   scar_sum=sum(cumulative_density_scar) # Normalize scars
   antiscar_sum=sum(cumulative_density_antiscar) # Normalize antiscars
   cumulative_density_scar./=scar_sum
   cumulative_density_antiscar./=antiscar_sum
   return cumulative_density_scar,cumulative_density_antiscar
end