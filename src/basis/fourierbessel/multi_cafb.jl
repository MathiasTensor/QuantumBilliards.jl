"""
    MultiCornerAdaptedFourierBessel{T,B,Sy} <: AbsBasis

Direct sum of several `CornerAdaptedFourierBessel` blocks centered at different
corners.

Global basis indices run consecutively through the blocks. `offsets` stores the
cumulative block offsets and `weights` stores the relative block dimensions used
for automatic resizing.
"""
struct MultiCornerAdaptedFourierBessel{T<:Real,B<:Tuple,Sy} <: AbsBasis
    blocks::B
    dim::Int
    offsets::Vector{Int}
    weights::Vector{T}
    symmetries::Sy
    scaling_origin::SVector{2,T}
end

"""
    MultiCornerAdaptedFourierBessel(blocks::Vararg{CornerAdaptedFourierBessel,N};symmetries=nothing) where {N}

Construct a multi-corner basis from existing corner-adapted Fourier-Bessel blocks.
"""
function MultiCornerAdaptedFourierBessel(blocks::Vararg{CornerAdaptedFourierBessel,N};symmetries=nothing,scaling_origin=nothing) where {N}
    N>0||throw(ArgumentError("at least one CAFB block is required"))
    T=typeof(blocks[1].corner_angle)
    all(b->typeof(b.corner_angle)===T,blocks)||throw(ArgumentError("all CAFB blocks must use the same real type"))
    dims=Int[b.dim for b in blocks]
    all(>(0),dims)||throw(ArgumentError("all CAFB blocks must have positive dimension"))
    offsets=Vector{Int}(undef,N+1)
    offsets[1]=0
    @inbounds for j=1:N
        offsets[j+1]=offsets[j]+dims[j]
    end
    dim=offsets[end]
    weights=T.(dims)./T(dim)
    origin=scaling_origin===nothing ? SVector{2,T}(zero(T),zero(T)) : SVector{2,T}(scaling_origin)
    return MultiCornerAdaptedFourierBessel{T,typeof(blocks),typeof(symmetries)}(blocks,dim,offsets,weights,symmetries,origin)
end

MultiCornerAdaptedFourierBessel(blocks::Tuple;symmetries=nothing,scaling_origin=nothing)=MultiCornerAdaptedFourierBessel(blocks...;symmetries=symmetries,scaling_origin=scaling_origin)

"""
    length(basis::MultiCornerAdaptedFourierBessel)

Return the total basis dimension.
"""
Base.length(basis::MultiCornerAdaptedFourierBessel)=basis.dim

"""
    _global_to_local_basis_index(basis::MultiCornerAdaptedFourierBessel,i::Int)

Map global basis index `i` to `(block,local_index)`.
"""
@inline function _global_to_local_basis_index(basis::MultiCornerAdaptedFourierBessel,i::Int)
    1<=i<=basis.dim||throw(BoundsError(basis,i))
    block=searchsortedlast(basis.offsets,i-1)
    local_index=i-basis.offsets[block]
    return block,local_index
end

"""
    _distribute_basis_dimensions(weights::AbstractVector{T},dim::Int) where {T<:Real}

Distribute total dimension `dim` among blocks according to `weights`, assigning
at least one function to every block.
"""
function _distribute_basis_dimensions(weights::AbstractVector{T},dim::Int) where {T<:Real}
    nblocks=length(weights)
    dim>=nblocks||throw(ArgumentError("total dimension $dim must be at least the number of blocks $nblocks"))
    remaining=dim-nblocks
    target=remaining.*weights
    extra=floor.(Int,target)
    dims=extra.+1
    leftover=dim-sum(dims)
    if leftover>0
        fractions=target.-extra
        order=sortperm(fractions;rev=true)
        @inbounds for j=1:leftover
            dims[order[j]]+=1
        end
    end
    return dims
end

"""
    resize_basis(basis::MultiCornerAdaptedFourierBessel,billiard::Bi,dim::Int,k) where {Bi<:AbsBilliard}
    resize_basis(basis::MultiCornerAdaptedFourierBessel,billiard::Bi,dims::AbstractVector{<:Integer},k) where {Bi<:AbsBilliard}

Resize the total basis dimension while preserving the stored relative block
weights.
"""
function resize_basis(basis::MultiCornerAdaptedFourierBessel,billiard::Bi,dim::Int,k) where {Bi<:AbsBilliard}
    basis.dim==dim&&return basis
    dims=_distribute_basis_dimensions(basis.weights,dim)
    blocks=ntuple(j->resize_basis(basis.blocks[j],billiard,dims[j],k),length(basis.blocks))
    newbasis=MultiCornerAdaptedFourierBessel(blocks;symmetries=basis.symmetries,scaling_origin=basis.scaling_origin)
    return MultiCornerAdaptedFourierBessel{typeof(newbasis.weights[1]),typeof(newbasis.blocks),typeof(newbasis.symmetries)}(newbasis.blocks,newbasis.dim,newbasis.offsets,copy(basis.weights),newbasis.symmetries,newbasis.scaling_origin)
end

function resize_basis(basis::MultiCornerAdaptedFourierBessel,billiard::Bi,dims::AbstractVector{<:Integer},k) where {Bi<:AbsBilliard}
    length(dims)==length(basis.blocks)||throw(DimensionMismatch("received $(length(dims)) dimensions for $(length(basis.blocks)) blocks"))
    all(>(0),dims)||throw(ArgumentError("all block dimensions must be positive"))
    blocks=ntuple(j->resize_basis(basis.blocks[j],billiard,Int(dims[j]),k),length(basis.blocks))
    return MultiCornerAdaptedFourierBessel(blocks;symmetries=basis.symmetries,scaling_origin=basis.scaling_origin)
end

"""
    basis_fun(basis::MultiCornerAdaptedFourierBessel,i::Int,k::T,pts::AbstractArray) where {T<:Real}

Evaluate global basis function `i` at `pts`.
"""
@inline function basis_fun(basis::MultiCornerAdaptedFourierBessel,i::Int,k::T,pts::AbstractArray) where {T<:Real}
    block,local_index=_global_to_local_basis_index(basis,i)
    return basis_fun(basis.blocks[block],local_index,k,pts)
end

"""
    dk_fun(basis::MultiCornerAdaptedFourierBessel,i::Int,k::T,pts::AbstractArray) where {T<:Real}

Evaluate the Vergini-Saraceno common-scaling derivative of global basis
function `i` about `basis.scaling_origin`:

    ∂ₖᴠˢ φ = ((x-scaling_origin)⋅∇φ)/k.
"""
@inline function dk_fun(basis::MultiCornerAdaptedFourierBessel,i::Int,k::T,pts::AbstractArray) where {T<:Real}
    block,local_index=_global_to_local_basis_index(basis,i)
    dB_dx,dB_dy=gradient(basis.blocks[block],local_index,k,pts)
    M=length(pts)
    dB_dk=Vector{T}(undef,M)
    x0=basis.scaling_origin[1]
    y0=basis.scaling_origin[2]
    @inbounds @simd for j=1:M
        rx=pts[j][1]-x0
        ry=pts[j][2]-y0
        dB_dk[j]=(rx*dB_dx[j]+ry*dB_dy[j])/k
    end
    return dB_dk
end

"""
    gradient(basis::MultiCornerAdaptedFourierBessel,i::Int,k::T,pts::AbstractArray) where {T<:Real}

Evaluate the Cartesian gradient of global basis function `i`.
"""
@inline function gradient(basis::MultiCornerAdaptedFourierBessel,i::Int,k::T,pts::AbstractArray) where {T<:Real}
    block,local_index=_global_to_local_basis_index(basis,i)
    return gradient(basis.blocks[block],local_index,k,pts)
end

"""
    basis_and_gradient(basis::MultiCornerAdaptedFourierBessel,i::Int,k::T,pts::AbstractArray) where {T<:Real}

Evaluate global basis function `i` and its Cartesian gradient.
"""
@inline function basis_and_gradient(basis::MultiCornerAdaptedFourierBessel,i::Int,k::T,pts::AbstractArray) where {T<:Real}
    block,local_index=_global_to_local_basis_index(basis,i)
    return basis_and_gradient(basis.blocks[block],local_index,k,pts)
end

"""
    _group_basis_indices_by_block(basis::MultiCornerAdaptedFourierBessel,indices)

Group requested global indices by CAFB block while preserving output-column
positions.
"""
function _group_basis_indices_by_block(basis::MultiCornerAdaptedFourierBessel,indices)
    nblocks=length(basis.blocks)
    columns=[Int[] for _=1:nblocks]
    local_indices=[Int[] for _=1:nblocks]
    @inbounds for (column,i) in enumerate(indices)
        block,local_index=_global_to_local_basis_index(basis,i)
        push!(columns[block],column)
        push!(local_indices[block],local_index)
    end
    return columns,local_indices
end

"""
    basis_fun(basis::MultiCornerAdaptedFourierBessel,indices::AbstractArray,k::T,pts::AbstractArray;multithreaded::Bool=true) where {T<:Real}

Evaluate the requested global basis functions as a matrix.
"""
@inline function basis_fun(basis::MultiCornerAdaptedFourierBessel,indices::AbstractArray,k::T,pts::AbstractArray;multithreaded::Bool=true) where {T<:Real}
    M=length(pts)
    N=length(indices)
    B=Matrix{T}(undef,M,N)
    columns,local_indices=_group_basis_indices_by_block(basis,indices)
    @inbounds for block in eachindex(basis.blocks)
        isempty(columns[block])&&continue
        block_values=basis_fun(basis.blocks[block],local_indices[block],k,pts;multithreaded=multithreaded)
        @views B[:,columns[block]].=block_values
    end
    return B
end

"""
    dk_fun(basis::MultiCornerAdaptedFourierBessel,indices::AbstractArray,k::T,pts::AbstractArray;multithreaded::Bool=true) where {T<:Real}

Evaluate the Vergini-Saraceno common-scaling derivatives of the requested
multi-corner basis functions about `basis.scaling_origin`:

    ∂ₖᴠˢ φ = ((x-scaling_origin)⋅∇φ)/k.
"""
function dk_fun(basis::MultiCornerAdaptedFourierBessel,indices::AbstractArray,k::T,pts::AbstractArray;multithreaded::Bool=true) where {T<:Real}
    M=length(pts)
    N=length(indices)
    dB_dk=Matrix{T}(undef,M,N)
    columns,local_indices=_group_basis_indices_by_block(basis,indices)
    x0=basis.scaling_origin[1]
    y0=basis.scaling_origin[2]
    @inbounds for block in eachindex(basis.blocks)
        isempty(columns[block])&&continue
        block_dx,block_dy=gradient(basis.blocks[block],local_indices[block],k,pts;multithreaded=multithreaded)
        cols=columns[block]
        @views block_dk=dB_dk[:,cols]
        @inbounds for c in eachindex(cols)
            @simd for j=1:M
                rx=pts[j][1]-x0
                ry=pts[j][2]-y0
                block_dk[j,c]=(rx*block_dx[j,c]+ry*block_dy[j,c])/k
            end
        end
    end
    return dB_dk
end

"""
    gradient(basis::MultiCornerAdaptedFourierBessel,indices::AbstractArray,k::T,pts::AbstractArray;multithreaded::Bool=true) where {T<:Real}

Evaluate the Cartesian gradients of the requested global basis functions.
"""
function gradient(basis::MultiCornerAdaptedFourierBessel,indices::AbstractArray,k::T,pts::AbstractArray;multithreaded::Bool=true) where {T<:Real}
    M=length(pts)
    N=length(indices)
    dB_dx=Matrix{T}(undef,M,N)
    dB_dy=Matrix{T}(undef,M,N)
    columns,local_indices=_group_basis_indices_by_block(basis,indices)
    @inbounds for block in eachindex(basis.blocks)
        isempty(columns[block])&&continue
        block_dx,block_dy=gradient(basis.blocks[block],local_indices[block],k,pts;multithreaded=multithreaded)
        @views dB_dx[:,columns[block]].=block_dx
        @views dB_dy[:,columns[block]].=block_dy
    end
    return dB_dx,dB_dy
end

"""
    basis_and_gradient(basis::MultiCornerAdaptedFourierBessel,indices::AbstractArray,k::T,pts::AbstractArray;multithreaded::Bool=true) where {T<:Real}

Evaluate the requested global basis functions and their Cartesian gradients.
"""
function basis_and_gradient(basis::MultiCornerAdaptedFourierBessel,indices::AbstractArray,k::T,pts::AbstractArray;multithreaded::Bool=true) where {T<:Real}
    M=length(pts)
    N=length(indices)
    B=Matrix{T}(undef,M,N)
    dB_dx=Matrix{T}(undef,M,N)
    dB_dy=Matrix{T}(undef,M,N)
    columns,local_indices=_group_basis_indices_by_block(basis,indices)
    @inbounds for block in eachindex(basis.blocks)
        isempty(columns[block])&&continue
        block_values,block_dx,block_dy=basis_and_gradient(basis.blocks[block],local_indices[block],k,pts;multithreaded=multithreaded)
        @views B[:,columns[block]].=block_values
        @views dB_dx[:,columns[block]].=block_dx
        @views dB_dy[:,columns[block]].=block_dy
    end
    return B,dB_dx,dB_dy
end