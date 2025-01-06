# this will be an anti-scar criteria based on the PH functions as for the case of the mushroom billiard we have a set od distinct localizations in the PH plot that is associated with BB modes in the stem.

# w is the full width, returns the edges of the bounding boxes that define the rectangle
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

# box1 and box2 are the rectangels that define the PH localization regions, H is the husimi matrix and x_grid and y_grid are axes of the Husimi where we will calculate the overlap
function create_husimi_localization_mat(x0_1::T,x1_1::T,y0_1::T,y1_1::T,x0_2::T,x1_2::T,y0_2::T,y1_2::T,x_grid::Vector{T},y_grid::Vector{T}) where {T<:Real}
    # create 2 boxes that have the value +1 and all the other ones have -1. Size of x_grid and y_grid should be the same as the H : size(H) = (length(x_grid),length(y_grid))
    mat=fill(-1,length(x_grid),length(y_grid))
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

function calculate_overlap(mat::Matrix,H::Matrix)
    @assert size(mat)==size(H) "Size of mat and H must match"
    overlap=sum(mat.*H)
    return overlap
end

# Hs is Vector{Matrix}
function calculate_overlap(mat::Matrix,Hs::Vector)
    overlaps=zeros(length(Hs))
    Threads.@threads for i in eachindex(Hs)
        overlaps[i]=calculate_overlap(mat,Hs[i])
    end
    return overlaps
end

function get_bb_localization_indexes(Hs::Vector,x_grid::Vector{T},y_grid::Vector{T},w::T,h::T,r::T;threshold=0.8) where {T<:Real}
    x0_1,x1_1,y0_1,y1_1,x0_2,x1_2,y0_2,y1_2=calculate_bbox_localization_mushroom(w,h,r)
    mat=create_husimi_localization_mat(x0_1,x1_1,y0_1,y1_1,x0_2,x1_2,y0_2,y1_2,x_grid,y_grid)
    overlaps=calculate_overlap(mat,Hs)
    idxs=findall(x->x>threshold,overlaps)
    return idxs
end