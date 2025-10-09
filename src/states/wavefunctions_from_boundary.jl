using Bessels, LinearAlgebra, ProgressMeter


"""
    ϕ(x::T, y::T, k::T, bd::BoundaryPoints{T}, u::AbstractVector{T}) where {T<:Real} -> T

Wavefunction via the boundary integral:
    Ψ(x,y) = (1/4) ∮ Y₀(k|q-q_s|) u(s) ds

Specialized for real `u` to keep everything in real arithmetic.
"""
@inline function ϕ(x::T,y::T,k::T,bd::BoundaryPoints{T},u::AbstractVector{T}) where {T<:Real}
    xy=bd.xy; ds=bd.ds
    s=zero(T)
    @inbounds @simd for j in eachindex(u)
        p=xy[j]
        r=hypot(x-p[1],y-p[2])        # stays in T
        y0=Bessels.bessely0(k*r)      # real kernel
        s=muladd(y0*u[j],ds[j],s)     # FMA: s += (y0*u_j)*ds_j
    end
    return s*T(0.25)
end

"""
    ϕ(x::T, y::T, k::T, bd::BoundaryPoints{T}, u::AbstractVector{Complex{T}}) where {T<:Real} -> Complex{T}

Same integral, but with complex boundary data `u`. Uses real kernel and
accumulates real/imag parts separately to avoid unnecessary complex multiplies.
"""
@inline function ϕ(x::T,y::T,k::T,bd::BoundaryPoints{T},u::AbstractVector{Complex{T}}) where {T<:Real}
    xy=bd.xy; ds=bd.ds
    sr=zero(T); si=zero(T)
    @inbounds @simd for j in eachindex(u)
        p=xy[j]
        r=hypot(x-p[1],y-p[2])
        w=Bessels.bessely0(k*r)*ds[j] # real weight
        uj=u[j]
        sr=muladd(w,real(uj),sr)
        si=muladd(w,imag(uj),si)
    end
    return Complex(sr,si)*T(0.25)
end

"""
    ϕ_float_bessel(x::T, y::T, k::T, bd::BoundaryPoints{T}, u::AbstractVector{T}) where {T<:Real} -> T

As `ϕ`, but calls `bessely0` in Float32 for speed; returns in `T`.
"""
@inline function ϕ_float_bessel(x::T,y::T,k::T,bd::BoundaryPoints{T},u::AbstractVector{T}) where {T<:Real}
    xy=bd.xy; ds=bd.ds
    s=zero(T)
    @inbounds @simd for j in eachindex(u)
        p=xy[j]
        r=hypot(x-p[1],y-p[2])
        y0=T(Bessels.bessely0(Float32(k*r))) # compute in Float32, cast back
        s=muladd(y0*u[j],ds[j],s)
    end
    return s*T(0.25)
end

"""
    ϕ_float_bessel(x::T, y::T, k::T, bd::BoundaryPoints{T}, u::AbstractVector{Complex{T}}) where {T<:Real} -> Complex{T}

Float32-Bessel variant for complex `u`. Accumulates real/imag parts separately.
"""
@inline function ϕ_float_bessel(x::T,y::T,k::T,bd::BoundaryPoints{T},u::AbstractVector{Complex{T}}) where {T<:Real}
    xy=bd.xy; ds=bd.ds
    sr=zero(T); si=zero(T)
    @inbounds @simd for j in eachindex(u)
        p=xy[j]
        r=hypot(x-p[1],y-p[2])
        w=T(Bessels.bessely0(Float32(k*r)))*ds[j]
        uj=u[j]
        sr=muladd(w,real(uj),sr)
        si=muladd(w,imag(uj),si)
    end
    return Complex(sr,si)*T(0.25)
end

"""
    wavefunction_multi(ks::Vector{T},vec_us::Vector{Vector{T}},vec_bdPoints::Vector{BoundaryPoints{T}},billiard::Bi;b::Float64=5.0,inside_only::Bool=true,fundamental=true,MIN_CHUNK=4_096) where {Bi<:AbsBilliard,T<:Real}

Constructs a sequence of 2D wavefunctions as matrices over the same sized grid for easier computation of matrix elements. The matrices are constructed via the boundary integral.

# Arguments
- `ks`: Vector of eigenvalues.
- `vec_bdPoints`: Vector of `BoundaryPoints` objects, one for each eigenvalues.
- `billiard`: The billiard geometry.
- `vec_us::Vector{Vector}`: Vector of the boundary functions. Can be either complex or real, this determines whether the wavefunction is real or not.
- `b::Float64=5.0`: (Optional), Point scaling factor. Default is 5.0.
- `inside_only::Bool=true`: (Optional), Whether to only compute wavefunctions inside the billiard. Default is true.
- `fundamental::Bool=true`: (Optional), Whether to use fundamental domain for boundary integral. Default is true.
- `MIN_CHUNK::Int=4096`: keep ≥ this many boundary points per thread

# Returns
- `Psi2ds::Vector{Matrix{T}}`: Vector of 2D wavefunction matrices constructed on the same grid.
- `x_grid::Vector{T}`: Vector of x-coordinates for the grid.
- `y_grid::Vector{T}`: Vector of y-coordinates for the grid.
"""
function wavefunction_multi(ks::Vector{T},vec_us::Vector{<:AbstractVector},vec_bdPoints::Vector{BoundaryPoints{T}},billiard::Bi;b::Float64=5.0,inside_only::Bool=true,fundamental=true,MIN_CHUNK=4_096) where {Bi<:AbsBilliard,T<:Real}
    k_max=maximum(ks)
    type=eltype(k_max)
    L=billiard.length
    if fundamental
        xlim,ylim=boundary_limits(billiard.fundamental_boundary;grd=max(1000,round(Int,k_max*L*b/(2*pi))))
    else
        xlim,ylim=boundary_limits(billiard.full_boundary;grd=max(1000,round(Int,k_max*L*b/(2*pi))))
    end
    dx,dy=xlim[2]-xlim[1],ylim[2]-ylim[1]
    nx,ny=max(round(Int,k_max*dx*b/(2*pi)),512),max(round(Int,k_max*dy*b/(2*pi)),512)
    x_grid,y_grid=collect(type,range(xlim..., nx)),collect(type,range(ylim..., ny))
    pts=collect(SVector(x,y) for y in y_grid for x in x_grid)
    sz=length(pts)
    # Determine points inside the billiard only once if inside_only is true
    pts_mask=inside_only ? points_in_billiard_polygon(pts,billiard,round(Int,sqrt(sz));fundamental_domain=fundamental) : fill(true,sz)
    pts_masked_indices=findall(pts_mask)
    NT=Threads.nthreads()
    nmask=length(pts_masked_indices)
    S=eltype(vec_us[1])<:Real ? type : Complex{type}
    Psi_flat=zeros(S,nx*ny) # overwritten each iteration since pts_masked_indices is the same for each k in ks
    NT_eff=max(1,min(NT,cld(nmask,MIN_CHUNK)))
    Psi2ds=Vector{Matrix{S}}(undef,length(ks))
    progress=Progress(length(ks),desc="Constructing wavefunction matrices...")
    q,r=divrem(nmask,NT_eff)
    @inbounds for i in eachindex(ks)
        k,bdPoints,us=ks[i],vec_bdPoints[i],vec_us[i]
        @fastmath begin
            Threads.@threads :static for t in 1:NT_eff
                # compute this thread's block [lo:hi]
                lo=(t-1)*q+min(t-1,r) + 1
                hi=lo+q-1+(t<=r ? 1 : 0)
                @inbounds for jj in lo:hi
                    idx=pts_masked_indices[jj] # each interior point [idx] -> (x,y)
                    x,y=pts[idx]
                    Psi_flat[idx]=ϕ_float_bessel(x,y,k,bdPoints,us) # Do it with floating point bessel computation, no need for double_precision here, and only for interior points
                end
            end
        end
        Psi2ds[i]=copy(reshape(Psi_flat,nx,ny))
        next!(progress)
    end
    return Psi2ds,x_grid,y_grid
end

"""
    wavefunction_multi_with_husimi(ks::Vector{T},vec_us::Vector{Vector{T}},vec_bdPoints::Vector{BoundaryPoints{T}},billiard::Bi;b::Float64=5.0, inside_only::Bool=true,fundamental=true,use_fixed_grid=true,xgrid_size=2000,ygrid_size=1000,MIN_CHUNK=4_096) where {Bi<:AbsBilliard,T<:Real}

Constructs a sequence of 2D wavefunctions as matrices over the same sized grid for easier computation of matrix elements. The matrices are constructed via the boundary integral. Additionally also constructs the husimi functions.

# Arguments
- `ks`: Vector of eigenvalues.
- `vec_bdPoints`: Vector of `BoundaryPoints` objects, one for each eigenvalues.
- `billiard`: The billiard geometry.
- `vec_us::Vector{Vector}`: Vector of the boundary functions. Can be either complex or real, this determines whether the wavefunction is real or not.
- `b::Float64=5.0`: (Optional), Point scaling factor. Default is 5.0.
- `inside_only::Bool=true`: (Optional), Whether to only compute wavefunctions inside the billiard. Default is true.
- `fundamental::Bool=true`: (Optional), Whether to use fundamental domain for boundary integral. Default is true.
- `xgrid_size::Int=2000`: (Optional), Size of the x grid for the husimi functions. Default is 2000.
- `ygrid_size::Int=1000`: (Optional), Size of the y grid for the husimi functions. Default is 1000.
- `use_fixed_grid::Bool=true`: (Optional), Whether to use a fixed grid for the husimi functions. Default is true.
- `MIN_CHUNK::Int=4096`: keep ≥ this many boundary points per thread

# Returns
- `Psi2ds::Vector{Matrix{T}}`: Vector of 2D wavefunction matrices constructed on the same grid.
- `x_grid::Vector{T}`: Vector of x-coordinates for the grid.
- `y_grid::Vector{T}`: Vector of y-coordinates for the grid.
- `Hs_list::Vector{Matrix{T}}`: Vector of 2D husimi function matrices.
- `ps_list::Vector{Vector{T}}`: Vector of ps grids for the husimi matrices.
- `qs_list::Vector{Vector{T}}`: Vector of qs grids for the husimi matrices.
"""
function wavefunction_multi_with_husimi(ks::Vector{T},vec_us::Vector{<:AbstractVector},vec_bdPoints::Vector{BoundaryPoints{T}},billiard::Bi;b::Float64=5.0, inside_only::Bool=true,fundamental=true,use_fixed_grid=true,xgrid_size=2000,ygrid_size=1000,MIN_CHUNK=4_096) where {Bi<:AbsBilliard,T<:Real}
    k_max=maximum(ks)
    type=eltype(k_max)
    L=billiard.length
    if fundamental
        xlim,ylim=boundary_limits(billiard.fundamental_boundary;grd=max(1000,round(Int,k_max*L*b/(2*pi))))
    else
        xlim,ylim=boundary_limits(billiard.full_boundary;grd=max(1000,round(Int,k_max*L*b/(2*pi))))
    end
    dx,dy=xlim[2]-xlim[1],ylim[2]-ylim[1]
    nx,ny=max(round(Int,k_max*dx*b/(2*pi)),512),max(round(Int,k_max*dy*b/(2*pi)),512)
    x_grid,y_grid=collect(type,range(xlim..., nx)),collect(type,range(ylim..., ny))
    pts=collect(SVector(x,y) for y in y_grid for x in x_grid)
    sz=length(pts)
    # Determine points inside the billiard only once if inside_only is true
    pts_mask=inside_only ? points_in_billiard_polygon(pts,billiard,round(Int,sqrt(sz));fundamental_domain=fundamental) : fill(true,sz)
    pts_masked_indices=findall(pts_mask)
    NT=Threads.nthreads()
    nmask=length(pts_masked_indices)
    S=eltype(vec_us[1])<:Real ? type : Complex{type}
    Psi_flat=zeros(S,nx*ny) # overwritten each iteration since pts_masked_indices is the same for each k in ks
    NT_eff=max(1,min(NT,cld(nmask,MIN_CHUNK)))
    Psi2ds=Vector{Matrix{S}}(undef,length(ks))
    progress=Progress(length(ks),desc="Constructing wavefunction matrices...")
    q,r=divrem(nmask,NT_eff)
    for i in eachindex(ks)
        k,bdPoints,us=ks[i],vec_bdPoints[i],vec_us[i]
        @fastmath begin
            Threads.@threads :static for t in 1:NT_eff
                # compute this thread's block [lo:hi]
                lo=(t-1)*q+min(t-1,r) + 1
                hi=lo+q-1+(t<=r ? 1 : 0)
                @inbounds for jj in lo:hi
                    idx=pts_masked_indices[jj] # each interior point [idx] -> (x,y)
                    x,y=pts[idx]
                    Psi_flat[idx]=ϕ_float_bessel(x,y,k,bdPoints,us) # Do it with floating point bessel computation, no need for double_precision here, and only for interior points
                end
            end
        end
        Psi2ds[i]=copy(reshape(Psi_flat,nx,ny))
        next!(progress)
    end
    vec_of_s_vals=[bdPoints.s for bdPoints in vec_bdPoints]
    if use_fixed_grid
        Hs_list,ps,qs=husimi_functions_from_us_and_boundary_points_FIXED_GRID(ks,vec_us,vec_bdPoints,billiard,xgrid_size,ygrid_size)
        ps_list=fill(ps,length(Hs_list))
        qs_list=fill(qs,length(Hs_list))
    else
        Hs_list,ps_list,qs_list=husimi_functions_from_boundary_functions(ks,vec_us,vec_of_s_vals,billiard)
    end
    return Psi2ds,x_grid,y_grid,Hs_list,ps_list,qs_list
end

"""
    batch_wrapper(plot_func::Function, args...; N::Integer=100, kwargs...)

Splits a large dataset into batches and calls the provided plotting function on each batch. 

This is useful when plotting a large number of wavefunctions or other data items at once would 
either be too large or time-consuming. By batching, you can generate multiple figures, each 
containing a subset of the data.

# Arguments
- `plot_func::Function`: The plotting function to be called for each batch.
- `args...`: The argument lists. The first argument should be a vector (e.g., `ks`) that 
   determines the number of data items. All other arguments must also be indexable and have 
   a compatible length.
- `N::Integer=100`: Number of items per batch.
- `kwargs...`: Additional keyword arguments passed on to `plot_func`.

# Returns
- `figures::Vector{Figure}`: A vector of `Figure` objects, each produced by `plot_func` 
   on a batch of data.
"""
function batch_wrapper(plot_func::Function, args...; N::Integer=100, kwargs...)
    # Extract the data vectors and the number of items
    ks=args[1] # ks is always the first argument
    @assert length(ks)>0 "ks cannot be empty."
    num_batches=ceil(Int,length(ks)/N)
    figures=Vector{Figure}(undef,num_batches)
    for i in 1:num_batches
        start_idx=(i-1)*N+1
        end_idx=min(i*N,length(ks))
        range=start_idx:end_idx
        batched_args=map(arg-> 
        if arg isa AbstractVector  # check if the argument is a vector
            if length(arg)==length(ks)  # slice if it matches `ks`
                arg[range]
            else
                arg  # x_grid, y_grid
            end
        else
            arg # legacy for billiard arg, remove later
        end,args)
        figures[i]=plot_func(batched_args...;kwargs...) # Call the original plotting function for the batch
    end
    return figures
end

"""
    partition_vector(ks::Vector{<:Real}, N::Integer)

Partitions the ks::Vector into N chunks. This is a helper function that helps us map the partitioned figures from the plotting functions (which give us Vector{Figure}) with the corresponding k values they contain. It partitians ks=[k1,k2,...,k_m] as vectors of length N [[k1...k_n],[k_n+1,...,k_2n],.... It is compatible with N=1 (just returns each k as a separate 1-element vector) and length(ks) < N, in which case it return the input vector unchanged.

# Arguments
- `ks::Vector{<:Real}`: The vector of eigenvalues to be partitioned.
- `N::Integer`: The number of chunks to partition the ks vector into.

# Returns
- `partitions::Vector{Vector{<:Real}}`: A vector of vectors, each containing N elements from the input ks vector.
"""
function partition_vector(ks::Vector, N::Integer)
    partitions=[ks[i:min(i+N-1,length(ks))] for i in 1:N:length(ks)]
    return partitions
end

"""
    plot_wavefunctions_BATCH(ks::Vector, Psi2ds::Vector, x_grid::Vector, y_grid::Vector, billiard::Bi; b::Float64=5.0, width_ax::Integer=500, height_ax::Integer=500, max_cols::Integer=6, fundamental=true) where {Bi<:AbsBilliard}

Plots the wavefunctions into a grid (only the fundamental boundary). The x_grid and y_grid is supplied from the wavefunction_multi or a similar function.

# Arguments
- `ks`: Vector of eigenvalues.
- `Psi2ds::Vector{Matrix}`: Vector of 2D wavefunction matrices.
- `x_grid::Vector{<:Real}`: Vector of x-coordinates for the grid.
- `y_grid::Vector{<:Real}`: Vector of y-coordinates for the grid.
- `billiard<:AbsBilliard`: The billiard geometry.
- `b::Float64=5.0`: The point scaling factor.
- `width_ax::Integer=500`: The size of each axis in the grid layout.
- `height_ax::Integer=500`: The size of each axis in the grid layout.
- `max_cols::Integer=6`: The maximum number of columns in the grid layout.
- `fundamental::Bool=true`: If plotting just the desymmetrized part.
- `custom_label::Vector{String}`: The labels to be plotted for each Axis in the Figure. ! Needs to be the same length as ks, as it should be unique to each k in ks !.

 # Returns
- `f::Figure`: A Figure object containing the grid of wavefunctions.
"""
function plot_wavefunctions_BATCH(ks::Vector, Psi2ds::Vector, x_grid::Vector, y_grid::Vector, billiard::Bi; b::Float64=5.0, width_ax::Integer=300, height_ax::Integer=300, max_cols::Integer=6, fundamental=true, custom_label::Vector{String}=String[]) where {Bi<:AbsBilliard}
    L=billiard.length
    if fundamental
        xlim,ylim=boundary_limits(billiard.fundamental_boundary;grd=max(1000,round(Int,maximum(ks)*L*b/(2*pi))))
    else
        xlim,ylim=boundary_limits(billiard.full_boundary;grd=max(1000,round(Int,maximum(ks)*L*b/(2*pi))))
    end
    n_rows=ceil(Int,length(ks)/max_cols)
    f = Figure(resolution=(round(Int,1.5*width_ax*max_cols),round(Int,2*height_ax*n_rows)),size=(round(Int,1.5*width_ax*max_cols),round(Int,2*height_ax*n_rows)))
    row=1
    col=1
    @showprogress desc="Plotting wavefunctions..." for j in eachindex(ks)
        title= isempty(custom_label) ? "$(ks[j])" : custom_label[j]
        local ax=Axis(f[row,col],title=title,aspect=DataAspect(),width=width_ax,height=height_ax)
        hm=heatmap!(ax,x_grid,y_grid,Psi2ds[j],colormap=:balance,colorrange=(-maximum(Psi2ds[j]),maximum(Psi2ds[j])))
        plot_boundary!(ax,billiard,fundamental_domain=fundamental,plot_normal=false)
        xlims!(ax,xlim)
        ylims!(ax,ylim)
        col+=1
        if col>max_cols
            row+=1
            col=1
        end
    end
    return f
end

"""
    plot_wavefunctions_BATCH(ks::Vector, Psi2ds::Vector, x_grid::Vector{Vector}, y_grid::Vector{Vector}, billiard::Bi; b::Float64=5.0, width_ax::Integer=500, height_ax::Integer=500, max_cols::Integer=6, fundamental=true) where {Bi<:AbsBilliard}

Plots the wavefunctions into a grid (only the fundamental boundary). The x_grid and y_grid is supplied from the `wavefunctions` method since it expects for each wavefunctions it's separate x and y grid.

# Arguments
- `ks`: Vector of eigenvalues.
- `Psi2ds::Vector{Matrix}`: Vector of 2D wavefunction matrices.
- `x_grid::Vector{Vector}`: Vector of x-coordinates for the grid for each wavefunction.
- `y_grid::Vector{Vector}`: Vector of y-coordinates for the grid for each wavefunction.
- `billiard<:AbsBilliard`: The billiard geometry.
- `b::Float64=5.0`: The point scaling factor.
- `width_ax::Integer=500`: The size of each axis in the grid layout.
- `height_ax::Integer=500`: The size of each axis in the grid layout.
- `max_cols::Integer=6`: The maximum number of columns in the grid layout.
- `fundamental::Bool=true`: If plotting just the desymmetrized part.
- `custom_label::Vector{String}`: The labels to be plotted for each Axis in the Figure. ! Needs to be the same length as ks, as it should be unique to each k in ks !.

 # Returns
- `f::Figure`: A Figure object containing the grid of wavefunctions.
"""
function plot_wavefunctions_BATCH(ks::Vector, Psi2ds::Vector, x_grid::Vector{Vector}, y_grid::Vector{Vector}, billiard::Bi; b::Float64=5.0, width_ax::Integer=300, height_ax::Integer=300, max_cols::Integer=6, fundamental=true, custom_label::Vector{String}=String[]) where {Bi<:AbsBilliard}
    L=billiard.length
    if fundamental
        xlim,ylim=boundary_limits(billiard.fundamental_boundary;grd=max(1000,round(Int,maximum(ks)*L*b/(2*pi))))
    else
        xlim,ylim=boundary_limits(billiard.full_boundary;grd=max(1000,round(Int,maximum(ks)*L*b/(2*pi))))
    end
    n_rows=ceil(Int,length(ks)/max_cols)
    f=Figure(resolution=(round(Int,1.5*width_ax*max_cols),round(Int,2*height_ax*n_rows)),size=(round(Int,1.5*width_ax*max_cols),round(Int,2*height_ax*n_rows)))
    row=1
    col=1
    @showprogress desc="Plotting wavefunctions..." for j in eachindex(ks)
        title= isempty(custom_label) ? "$(ks[j])" : custom_label[j]
        local ax=Axis(f[row,col],title=title,aspect=DataAspect(),width=width_ax,height=height_ax)
        hm=heatmap!(ax,x_grid[j],y_grid[j],Psi2ds[j],colormap=:balance,colorrange=(-maximum(Psi2ds[j]),maximum(Psi2ds[j])))
        plot_boundary!(ax,billiard,fundamental_domain=fundamental,plot_normal=false)
        xlims!(ax,xlim)
        ylims!(ax,ylim)
        col+=1
        if col>max_cols
            row+=1
            col=1
        end
    end
    return f
end

"""
    plot_wavefunctions(ks::Vector, Psi2ds::Vector, x_grid::Vector, y_grid::Vector, billiard::Bi; N::Integer=100, kwargs...) where {Bi<:AbsBilliard}

Plots the wavefunctions specified by `Psi2ds` on the domain defined by `x_grid` and `y_grid`, 
for the billiard geometry `billiard`. The eigenvalues are provided in `ks`. When the number of 
wavefunctions is large, this function automatically splits the data into batches of size `N` 
and generates multiple figures.

# Arguments
- `ks::Vector`: Vector of eigenvalues.
- `Psi2ds::Vector{Matrix}`: Vector of 2D wavefunction matrices corresponding to `ks`.
- `x_grid::Vector{<:Real}`: Vector of x-coordinates for the grid.
- `y_grid::Vector{<:Real}`: Vector of y-coordinates for the grid.
- `billiard::Bi<:AbsBilliard`: The billiard geometry.
- `N::Integer=100`: The number of items per batch. If `length(ks) > N`, multiple figures are produced.
- `kwargs...`: Additional keyword arguments passed to the underlying plotting function (N axes per Figure and custom label, check _BATCH function)

# Returns
- `figures::Vector{Figure}`: A vector of `Figure` objects, one per batch.
"""
function plot_wavefunctions(ks::Vector, Psi2ds::Vector, x_grid::Vector, y_grid::Vector, billiard::Bi; N::Integer=100, kwargs...) where {Bi<:AbsBilliard}
    batch_wrapper(plot_wavefunctions_BATCH,ks,Psi2ds,x_grid,y_grid,billiard;N=N,kwargs...)
end

"""
    plot_wavefunctions(ks::Vector, Psi2ds::Vector, x_grid::Vector{Vector}, y_grid::Vector{Vector}, billiard::Bi; N::Integer=100, kwargs...) where {Bi<:AbsBilliard}

Similar to `plot_wavefunctions` above, but this version allows for a distinct `(x_grid, y_grid)` 
for each wavefunction in `Psi2ds`. This is useful if each wavefunction was computed on a different 
grid. Automatically splits the data into batches of size `N` if `ks` is large.

# Arguments
- `ks::Vector`: Vector of eigenvalues.
- `Psi2ds::Vector{Matrix}`: Vector of 2D wavefunction matrices, one for each `(x_grid[j], y_grid[j])`.
- `x_grid::Vector{Vector{<:Real}}`: A vector of x-coordinate vectors, one per wavefunction.
- `y_grid::Vector{Vector{<:Real}}`: A vector of y-coordinate vectors, one per wavefunction.
- `billiard::Bi<:AbsBilliard`: The billiard geometry.
- `N::Integer=100`: The number of items per batch.
- `kwargs...`: Additional keyword arguments passed to the underlying plotting function (N axes per Figure and custom label, check _BATCH function)

# Returns
- `figures::Vector{Figure}`: A vector of `Figure` objects, one per batch of wavefunctions.
"""
function plot_wavefunctions(ks::Vector, Psi2ds::Vector, x_grid::Vector{Vector}, y_grid::Vector{Vector}, billiard::Bi; N::Integer=100, kwargs...) where {Bi<:AbsBilliard}
    batch_wrapper(plot_wavefunctions_BATCH,ks,Psi2ds,x_grid,y_grid,billiard;N=N,kwargs...)
end

"""
    plot_wavefunctions_BATCH(ks::Vector, Psi2ds::Vector, x_grid::Vector, y_grid::Vector, billiard::Bi; b::Float64=5.0, width_ax::Integer=500, height_ax::Integer=500, max_cols::Integer=6) where {Bi<:AbsBilliard}

Plots the wavefunctions into a grid (only the fundamental boundary) together with the respective husimi function matrices on the provided grids. The x_grid and y_grid is supplied from the wavefunction_multi or a similar function, and the ps and qs grids mudt also be supplied for plotting the Husimi functions.

# Arguments
- `ks`: Vector of eigenvalues.
- `Psi2ds::Vector{Matrix}`: Vector of 2D wavefunction matrices.
- `x_grid::Vector{<:Real}`: Vector of x-coordinates for the grid.
- `y_grid::Vector{<:Real}`: Vector of y-coordinates for the grid.
- `Hs_list::Vector{Matrix}`: Vector of 2D husimi function matrices.
- `ps_list::Vector{Vector}`: Vector of ps grids for the husimi matrices.
- `qs_list::Vector{Vector}`: Vector of qs grids for the husimi matrices.
- `billiard<:AbsBilliard`: The billiard geometry.
- `b::Float64=5.0`: The point scaling factor.
- `width_ax::Integer=500`: The size of each axis in the grid layout.
- `height_ax::Integer=500`: The size of each axis in the grid layout.
- `max_cols::Integer=6`: The maximum number of columns in the grid layout.
- `fundamental::Bool=true`: If plotting just the desymmetrized part.
- `custom_label::Vector{String}`: The labels to be plotted for each Axis in the Figure. ! Needs to be the same length as ks, as it should be unique to each k in ks !.
- `use_projection_grid::Tuple{Vector,Vector}=([],[])`: A tuple containing the classical s and p values of the chaotic trajectory (in that order). These are used to construct the chaotic mask overlay so we can better observe the overlaps.

 # Returns
- `f::Figure`: A Figure object containing the grid of wavefunctions.
"""
function plot_wavefunctions_with_husimi_BATCH(ks::Vector, Psi2ds::Vector, x_grid::Vector, y_grid::Vector, Hs_list::Vector, ps_list::Vector, qs_list::Vector, billiard::Bi; b::Float64=5.0, width_ax::Integer=300, height_ax::Integer=300, max_cols::Integer=6, fundamental=true, custom_label::Vector{String}=String[], use_projection_grid::Tuple{Vector,Vector}=([],[])) where {Bi<:AbsBilliard}
    L=billiard.length
    if fundamental
        xlim,ylim=boundary_limits(billiard.fundamental_boundary;grd=max(1000,round(Int,maximum(ks)*L*b/(2*pi))))
    else
        xlim,ylim=boundary_limits(billiard.full_boundary;grd=max(1000,round(Int,maximum(ks)*L*b/(2*pi))))
    end
    n_rows=ceil(Int,length(ks)/max_cols)
    f = Figure(resolution=(3*width_ax*max_cols,1.5*height_ax*n_rows),size=(3*width_ax*max_cols,1.5*height_ax*n_rows))
    row=1
    col=1
    @showprogress desc="Plotting wavefunctions and husimi..." for j in eachindex(ks)
        title= isempty(custom_label) ? "$(ks[j])" : custom_label[j]
        local ax=Axis(f[row,col][1,1],title=title,aspect=DataAspect(),width=width_ax,height=height_ax)
        local ax_h=Axis(f[row,col][1,2],width=width_ax,height=height_ax)
        hm=heatmap!(ax,x_grid,y_grid,Psi2ds[j],colormap=:balance,colorrange=(-maximum(Psi2ds[j]),maximum(Psi2ds[j])))
        plot_boundary!(ax,billiard,fundamental_domain=fundamental,plot_normal=false)
        if !isempty(use_projection_grid[1]) && !isempty(use_projection_grid[2])
            projection_grid=classical_phase_space_matrix(use_projection_grid[1],use_projection_grid[2],qs_list[j],ps_list[j])
            H_bg,chaotic_mask=husimi_with_chaotic_background(Hs_list[j],projection_grid)
            heatmap!(ax_h,qs_list[j],ps_list[j],H_bg; colormap=Reverse(:gist_heat), colorrange=(0.0, maximum(H_bg)))
            heatmap!(ax_h,qs_list[j],ps_list[j],chaotic_mask;colormap=cgrad([:white, :black]),alpha=0.05,colorrange=(0,1))
        else
            heatmap!(ax_h,qs_list[j],ps_list[j],Hs_list[j];colormap=Reverse(:gist_heat))
        end
        xlims!(ax,xlim)
        ylims!(ax,ylim)
        col+=1
        if col>max_cols
            row+=1
            col=1
        end
    end
    return f
end

"""
    plot_wavefunctions_BATCH(ks::Vector, Psi2ds::Vector, x_grid::Vector, y_grid::Vector, billiard::Bi; b::Float64=5.0, width_ax::Integer=500, height_ax::Integer=500, max_cols::Integer=6) where {Bi<:AbsBilliard}

Plots the wavefunctions into a grid (only the fundamental boundary) together with the respective husimi function matrices on the provided grids. The x_grid and y_grid is supplied from the wavefunction_multi or a similar function, and the ps and qs grids mudt also be supplied for plotting the Husimi functions. This version also accepts the us boundary functions and the corresponding arclength evaluation point (us_all -> Vector{Vector{T}} and s_vals_all -> Vector{Vector{T}}) that this function was evaluated on.

# Arguments
- `ks`: Vector of eigenvalues.
- `Psi2ds::Vector{Matrix}`: Vector of 2D wavefunction matrices.
- `x_grid::Vector{<:Real}`: Vector of x-coordinates for the grid.
- `y_grid::Vector{<:Real}`: Vector of y-coordinates for the grid.
- `Hs_list::Vector{Matrix}`: Vector of 2D husimi function matrices.
- `ps_list::Vector{Vector}`: Vector of ps grids for the husimi matrices.
- `qs_list::Vector{Vector}`: Vector of qs grids for the husimi matrices.
- `billiard<:AbsBilliard`: The billiard geometry.
- `us_all::Vector{Vector{T}}`: Vector of us boundary functions.
- `s_vals_all::Vector{Vector{T}}`: Vector of arclength evaluation points.
- `b::Float64=5.0`: The point scaling factor.
- `width_ax::Integer=500`: The size of each axis in the grid layout.
- `height_ax::Integer=500`: The size of each axis in the grid layout.
- `max_cols::Integer=6`: The maximum number of columns in the grid layout.
- `fundamental::Bool=true`: If plotting just the desymmetrized part.
- `custom_label::Vector{String}`: The labels to be plotted for each Axis in the Figure. ! Needs to be the same length as ks, as it should be unique to each k in ks !.
- `use_projection_grid::Tuple{Vector,Vector}=([],[])`: A tuple containing the classical s and p values of the chaotic trajectory (in that order). These are used to construct the chaotic mask overlay so we can better observe the overlaps.

 # Returns
- `f::Figure`: A Figure object containing the grid of wavefunctions.
"""
function plot_wavefunctions_with_husimi_BATCH(ks::Vector, Psi2ds::Vector, x_grid::Vector, y_grid::Vector, Hs_list::Vector, ps_list::Vector, qs_list::Vector, billiard::Bi, us_all::Vector, s_vals_all::Vector; b::Float64=5.0, width_ax::Integer=300, height_ax::Integer=300, max_cols::Integer=6, fundamental=true, custom_label::Vector{String}=String[], use_projection_grid::Tuple{Vector,Vector}=([],[])) where {Bi<:AbsBilliard}
    L=billiard.length
    if fundamental
        xlim,ylim=boundary_limits(billiard.fundamental_boundary;grd=max(1000,round(Int,maximum(ks)*L*b/(2*pi))))
    else
        xlim,ylim=boundary_limits(billiard.full_boundary;grd=max(1000,round(Int,maximum(ks)*L*b/(2*pi))))
    end
    L_corners=0.0
    res=Dict{Float64, Bool}()  # Dictionary to store length and type (true for real, false for virtual)
    res[L_corners]=true # we should start at the real curve anyway
    for crv in billiard.full_boundary
        if crv isa AbsRealCurve
            L_corners+=crv.length
            res[L_corners]=true  # Add length with true (real curve)
        elseif crv isa AbsVirtualCurve
            L_corners+=crv.length
            res[L_corners]=false  # Add length with false (virtual curve)
        end
    end
    n_rows=ceil(Int,length(ks)/max_cols)
    f=Figure(resolution=(3*width_ax*max_cols,2*height_ax*n_rows),size=(3*width_ax*max_cols,2*height_ax*n_rows))
    row=1
    col=1
    @showprogress desc="Plotting wavefunctions and husimi..." for j in eachindex(ks)
        title= isempty(custom_label) ? "$(ks[j])" : custom_label[j]
        local ax_wave=Axis(f[row, col][1, 1],title=title,aspect=DataAspect(),width=width_ax,height=height_ax)
        hm_wave=heatmap!(ax_wave,x_grid,y_grid,Psi2ds[j],colormap=:balance,colorrange=(-maximum(Psi2ds[j]), maximum(Psi2ds[j])))
        plot_boundary!(ax_wave,billiard,fundamental_domain=fundamental,plot_normal=false)
        xlims!(ax_wave,xlim)
        ylims!(ax_wave,ylim)
        local ax_h=Axis(f[row, col][1, 2],width=width_ax,height=height_ax)
        if !isempty(use_projection_grid[1]) && !isempty(use_projection_grid[2])
            projection_grid=classical_phase_space_matrix(use_projection_grid[1],use_projection_grid[2],qs_list[j],ps_list[j])
            H_bg,chaotic_mask=husimi_with_chaotic_background(Hs_list[j],projection_grid)
            heatmap!(ax_h,qs_list[j],ps_list[j],H_bg; colormap=Reverse(:gist_heat), colorrange=(0.0, maximum(H_bg)))
            heatmap!(ax_h,qs_list[j],ps_list[j],chaotic_mask;colormap=cgrad([:white, :black]),alpha=0.05,colorrange=(0,1))
        else
            heatmap!(ax_h,qs_list[j],ps_list[j],Hs_list[j];colormap=Reverse(:gist_heat))
        end
        local ax_boundary = Axis(f[row, col][2, 1:2],xlabel="s",ylabel="u(s)",width=2*width_ax,height=height_ax/2)
        lines!(ax_boundary,s_vals_all[j],us_all[j],label="u(s)",linewidth=2)
        for (length, is_real) in res
            vlines!(ax_boundary,[length],color=(is_real ? :blue : :red),linestyle=(is_real ? :solid : :dash))
        end
        # Move to the next column
        col+=1
        if col>max_cols
            row+=1 
            col=1
        end
    end
    return f
end

"""
    plot_wavefunctions_with_husimi(ks::Vector, Psi2ds::Vector, x_grid::Vector, y_grid::Vector, Hs_list::Vector, ps_list::Vector, qs_list::Vector, billiard::Bi; N=100, kwargs...) where {Bi<:AbsBilliard}

Plots the wavefunctions along with their corresponding Husimi distributions. Automatically 
splits large datasets into batches of size `N`.

# Arguments
- `ks::Vector`: Vector of eigenvalues.
- `Psi2ds::Vector{Matrix}`: Wavefunction matrices.
- `x_grid::Vector{<:Real}`: X-coordinates for the wavefunction grid.
- `y_grid::Vector{<:Real}`: Y-coordinates for the wavefunction grid.
- `Hs_list::Vector{Matrix}`: Husimi function matrices associated with each wavefunction.
- `ps_list::Vector{Vector{<:Real}}`: Momentum-like coordinate grids for the Husimi functions.
- `qs_list::Vector{Vector{<:Real}}`: Position-like coordinate grids for the Husimi functions.
- `billiard::Bi<:AbsBilliard`: The billiard geometry.
- `N::Integer=100`: Number of items per batch.
- `kwargs...`: Additional keyword arguments passed to the underlying plotting function (N axes per Figure and custom label, also chaotic PS overlays, check _BATCH function)

# Returns
- `figures::Vector{Figure}`: A vector of `Figure` objects with wavefunction and Husimi plots, one per batch.
"""
function plot_wavefunctions_with_husimi(ks::Vector, Psi2ds::Vector, x_grid::Vector, y_grid::Vector, Hs_list::Vector, ps_list::Vector, qs_list::Vector, billiard::Bi; N=100, kwargs...) where {Bi<:AbsBilliard}
    batch_wrapper(plot_wavefunctions_with_husimi_BATCH,ks,Psi2ds,x_grid,y_grid,Hs_list,ps_list,qs_list,billiard;N=N,kwargs...)
end

"""
    plot_wavefunctions_with_husimi(ks::Vector, Psi2ds::Vector, x_grid::Vector, y_grid::Vector, Hs_list::Vector, ps_list::Vector, qs_list::Vector, billiard::Bi, us_all::Vector, s_vals_all::Vector; N=100, kwargs...) where {Bi<:AbsBilliard}

Plots the wavefunctions along with their Husimi distributions and boundary functions `us_all` 
evaluated at `s_vals_all`. This function also handles a large number of wavefunctions by batching 
the data into sets of size `N`.

# Arguments
- `ks::Vector`: Vector of eigenvalues.
- `Psi2ds::Vector{Matrix}`: Wavefunction matrices.
- `x_grid::Vector{<:Real}`: X-coordinates for the wavefunction grid.
- `y_grid::Vector{<:Real}`: Y-coordinates for the wavefunction grid.
- `Hs_list::Vector{Matrix}`: Husimi function matrices.
- `ps_list::Vector{Vector{<:Real}}`: Momentum-like coordinates for Husimi functions.
- `qs_list::Vector{Vector{<:Real}}`: Position-like coordinates for Husimi functions.
- `billiard::Bi<:AbsBilliard`: The billiard geometry.
- `us_all::Vector{Vector{T}}`: Boundary functions.
- `s_vals_all::Vector{Vector{T}}`: Arclength evaluation points for the boundary functions.
- `N::Integer=100`: Number of items per batch.
- `kwargs...`: Additional keyword arguments passed to the underlying plotting function (N axes per Figure and custom label, also chaotic PS overlays, check _BATCH function)

# Returns
- `figures::Vector{Figure}`: A vector of `Figure` objects, each containing wavefunction, Husimi plots, and boundary functions, one per batch.
"""
function plot_wavefunctions_with_husimi(ks::Vector, Psi2ds::Vector, x_grid::Vector, y_grid::Vector, Hs_list::Vector, ps_list::Vector, qs_list::Vector, billiard::Bi, us_all::Vector, s_vals_all::Vector; N=100, kwargs...) where {Bi<:AbsBilliard}
    batch_wrapper(plot_wavefunctions_with_husimi_BATCH,ks,Psi2ds,x_grid,y_grid,Hs_list,ps_list,qs_list,billiard,us_all,s_vals_all;N=N,kwargs...)
end


##############################################################################
#### TOOLS FOR CHECKING THE POWER SPECTRUM FOR CircleSegment part of u(s) ####
##############################################################################

"""
    compute_cm_circular_segment(u, s_vals, m, billiard)

Compute the angular momentum coefficient `cₘ` of the boundary function `u(s)` restricted to the CircleSegment of the billiard using trapezoidal integration.

# Arguments
- `u::Vector{T}`: Normal derivative of the wavefunction.
- `s_vals::Vector{T}`: Arclength positions of boundary points.
- `m::Integer`: Angular momentum index.
- `billiard::AbsBilliard`: The billiard geometry.

# Returns
- `Complex{T}`: Complex coefficient `cₘ`.
"""
function compute_cm_circular_segment(u::Vector{T},s_vals::Vector{T},m::Ti,billiard::Bi)::Complex{T} where {T<:Real,Ti<:Integer,Bi<:AbsBilliard}
    s_start,s_end=0.0,0.0
    total_length=0.0
    R=0.0
    found=false
    for seg in billiard.full_boundary
        seg_length=seg.length
        if seg isa CircleSegment
            s_start=total_length
            s_end=total_length+seg_length
            R=seg.radius
            found=true
            break
        end
        total_length+=seg_length
    end
    if !found
        error("No CircleSegment found in billiard boundary.")
    end
    filtered_idx=findall(s->s>=s_start && s<=s_end,s_vals) # filter u and s_vals on the CircleSegment
    us=u[filtered_idx]
    ss=s_vals[filtered_idx]
    N=length(us) #
    if N<2 # sanity check
        @warn "Not enough points on the CircleSegment to compute integral."
        return 0.0+0.0im
    end
    weights=zeros(T,N) # trapezoidal weights
    weights[1]=(ss[2]-ss[1])/2
    for i in 2:N-1
        weights[i]=(ss[i+1]-ss[i-1])/2
    end
    weights[end]=(ss[end]-ss[end-1])/2
    return sum(us[i]*exp(im*m*π*ss[i]/R)*weights[i] for i in 1:N)
end

"""
    compute_cm_circular_segment(u, s_vals, ms, billiard)

Compute multiple angular momentum coefficients `cₘ` for each `m` in `ms` from the boundary function `u(s)` on the CircleSegment.

# Arguments
- `u::Vector{T}`: Normal derivative of the wavefunction.
- `s_vals::Vector{T}`: Arclength positions of boundary points.
- `ms::Vector{Integer}`: Angular momentum indexes.
- `billiard::AbsBilliard`: The billiard geometry.

# Returns
- `Vector{Complex{T}}`: Vector of angular momentum coefficients.
"""
function compute_cm_circular_segment(u::Vector{T},s_vals::Vector{T},ms::Vector{Ti},billiard::Bi)::Vector{Complex{T}} where {T<:Real,Ti<:Integer,Bi<:AbsBilliard}
    s_start,s_end=0.0,0.0
    total_length=0.0
    R=0.0
    found=false
    for seg in billiard.full_boundary
        seg_length=seg.length
        if seg isa CircleSegment
            s_start=total_length
            s_end=total_length+seg_length
            R=seg.radius
            found=true
            break
        end
        total_length+=seg_length
    end
    if !found
        error("No CircleSegment found in billiard boundary.")
    end
    filtered_idx=findall(s->s>=s_start && s<=s_end,s_vals) # filter u and s_vals on the CircleSegment
    us=u[filtered_idx]
    ss=s_vals[filtered_idx]
    N=length(us) #
    if N<2 # sanity check
        @warn "Not enough points on the CircleSegment to compute integral."
        return 0.0+0.0im
    end
    weights=zeros(T,N) # trapezoidal weights
    weights[1]=(ss[2]-ss[1])/2
    for i in 2:N-1
        weights[i]=(ss[i+1]-ss[i-1])/2
    end
    weights[end]=(ss[end]-ss[end-1])/2
    cms=Vector{Complex{T}}(undef,length(ms))
    Threads.@threads for k in eachindex(ms)
        cms[k]=sum(us[i]*exp(im*ms[k]*π*ss[i]/R)*weights[i] for i in 1:N)
    end
    return cms
end

"""
    fraction_on_segments(u, s_vals, billiard; which_segments = :all)

Compute the L² norm fraction of the boundary function `u(s)` over selected segments of the billiard.

This is useful for distinguishing whether the boundary function is concentrated on specific segments
(e.g., the CircleSegment) or spread across others.

# Arguments
- `u::Vector{T}`: The boundary function.
- `s_vals::Vector{T}`: Arclength coordinates along the boundary.
- `billiard::AbsBilliard`: The billiard geometry.
- `which_segments::Union{Symbol, Vector{Int}} = :all`: Which segments to include:
    - `:circle` → only the CircleSegment,
    - `:all` → the entire boundary,
    - `Vector{Int}` → specify by segment indices in `billiard.full_boundary`.

# Returns
- `T`: Fraction of the total L² norm on the selected segments.
"""
function fraction_on_segments(u::Vector{T},s_vals::Vector{T},billiard::AbsBilliard;which_segments::Union{Symbol,Vector{Ti}}=:all)::T where {T<:Real,Ti<:Integer}
    @assert length(u)==length(s_vals) "u and s_vals must be the same length"
    N=length(u)
    # Precompute segment arclength intervals
    segment_bounds=Vector{Tuple{T,T}}()
    circle_idx=nothing
    total_length=zero(T)
    for (i,seg) in enumerate(billiard.full_boundary)
        L=seg.length
        push!(segment_bounds,(total_length,total_length+L))
        if seg isa CircleSegment
            circle_idx=i
        end
        total_length+=L
    end
    selected_bounds= if which_segments==:all
        segment_bounds
    elseif which_segments==:circle
        @assert circle_idx!==nothing "No CircleSegment found."
        [segment_bounds[circle_idx]]
    elseif isa(which_segments,Vector{Int})
        [segment_bounds[i] for i in which_segments]
    else
        error("Invalid `which_segments` value. Use :all, :circle, or a Vector of segment indices.")
    end
    # Trapezoidal weights
    weights = similar(u)
    @inbounds begin
        weights[1]=(s_vals[2]-s_vals[1])/2
        for i in 2:N-1
            weights[i]=(s_vals[i+1]-s_vals[i-1])/2
        end
        weights[end]=(s_vals[end]-s_vals[end-1])/2
    end
    total_per_thread=zeros(T,Threads.nthreads())
    selected_per_thread=zeros(T,Threads.nthreads())
    Threads.@threads for i in 1:N
        tid=Threads.threadid()
        val=abs2(u[i])*weights[i]
        total_per_thread[tid]+=val
        for (s_start,s_end) in selected_bounds
            if s_vals[i]≥s_start && s_vals[i]≤s_end
                selected_per_thread[tid]+=val
                break
            end
        end
    end
    total_norm=sum(total_per_thread)
    selected_norm=sum(selected_per_thread)
    return selected_norm/total_norm
end

"""
    compute_cm_circular_segment_and_fraction(u::Vector{T},s_vals::Vector{T},ms::Vector{Ti},billiard::Bi)::Tuple{Vector{Complex{T}},T} where {T<:Real,Ti<:Integer,Bi<:AbsBilliard}

Computes the cm coefficients of the angular momentum basis expansion and also the fraction of the boundary function on the CircularSegment using the Trapezoidal rule on the L^2 norm.

# Arguments
- `u::Vector{T}`: The boundary function.
- `s_vals::Vector{T}`: The arclengths of the entire billiard.
- `ms::Vector{Integer}`: Angular momentum indexes.
- `billiard<:AbsBilliard`: The billiard geometry that contains information on all the curve segments.
- `which_segments::Union{Symbol, Vector{Int}} = :all`: Which segments to take in the the calculation of the fraction of the boundary norm. The default value is a placeholder and the Vector{Int} should be used for the other relevant sections where we want to check the boundary function L2 norm.

For example in the mushroom billiard we would choose `which_segments = [1, 2, 6]` since the other segments are either the `CircleSegment` or `LineSegment`s that have overlap with the circle eigenfunction (the connectors of the stem with the cap are such cases with `idxs = [3, 5]`)

# Returns
- `cms::Vector{Complex{T}}`: The cm coefficient for each m in ms.
- `frac::T`: The fraction of the boundary function as per function description.
"""
function compute_cm_circular_segment_and_fraction(u::Vector{T},s_vals::Vector{T},ms::Vector{Ti},billiard::Bi;which_segments::Union{Symbol,Vector{Ti}}=:all)::Tuple{Vector{Complex{T}},T} where {T<:Real,Ti<:Integer,Bi<:AbsBilliard}
    cms=compute_cm_circular_segment(u,s_vals,ms,billiard)
    frac=fraction_on_segments(u,s_vals,billiard;which_segments=which_segments)
    return cms,frac
end

"""
    compute_P_m(cm::Complex{T})::T where {T<:Real}

Returns the power `|cₘ|²` from a single angular momentum coefficient.

# Arguments
- `cm::Complex{T}`: The angular momentum coefficient.

# Returns
- `T`: The associated `|cₘ|²` value.
"""
function compute_P_m(cm::Complex{T})::T where {T<:Real}
    return abs2(cm)
end

"""
    compute_P_m(cms::Vector{Complex{T}})::Vector{T} where {T<:Real}

Returns the power spectrum `|cₘ|²` for a vector of coefficients.

# Arguments
- `cms::Vector{Complex{T}}`: The angular momentum coefficients.

# Returns
- `Vector{T}`: The associated vector of NORMALIZED `|cₘ|²` values.
"""
function compute_P_m(cms::Vector{Complex{T}}) where {T<:Real}
    S=sum(abs2.(cms))
    return [abs2(cm)/S for cm in cms]
end

"""
    Shannon_entropy_cms(Pms)

Computes the Shannon entropy from a normalized angular momentum power distribution.

# Arguements
- `Pms::Vector{T}`: Normalized Power spectrum for a boundary function. Sometimes it is useful to take the log of this value since delta-like functions have negative Shannon entropy value

# Returns
- `T`: Shannon entropy value `S = -∑ pᵢ log(pᵢ)`
"""
function Shannon_entropy_cms(Pms::Vector{T}) where {T<:Real}
    return -sum(p_i>0.0 ? p_i*log(p_i) : 0.0 for p_i in Pms)
end

"""
    is_regular(Pms::Vector{T},frac::T;threshold::Float64=1.0,frac_threshold=0.1) where {T<:Real}

Determine if a state is "regular-like" based on low Shannon entropy in angular momentum space.

Returns `true` if `S < threshold`, suggesting localization around a conserved quantity.

# Arguments
- `Pms::Vector{T}`: Normalized power distribution.
- `frac::T`: THe threshold for the L^2 norm of the boundary function on non-circular segments that were chosen with `which_segments` in the bottom level functions.
- `threshold=1.0`: Entropy cutoff (default = 1.0 by obseving the behaviour of the boundary function on the `CircleSegment`). This also is useful since when we take the log of ot it is negative if below 1.0 and separration is clear.
- `frac_threshold=0.1`: The default threshold for the L2 norm of the boudnary function on the boundary chosen with `which_segments` in the bottom level functions. Benchmarking shows that the frac value is usually well below 10^-2, in most cases below < 10^-7. This value is analogous addition measure of the overlap between the wavefunction with the circle eigefunction (it's angular momentum component exp(i*pi*ϕ)).

# Returns
- `Bool`: Whether the state is regular.
"""
function is_regular(Pms::Vector{T},frac::T;threshold=1.0,frac_threshold=0.1) where {T<:Real}
    Shannon_entropy_cms(Pms)<threshold && frac<frac_threshold ? true : false
end

# HELPER FUNCTION SINCE THERE ARE ISOLATED CASES WHERE A REGULAR FUNCTION GRAZING THE 
function is_mushroom_MUPO(frac::T;frac_threshold=0.1) where {T<:Real}
    frac<frac_threshold ? true : false
end
