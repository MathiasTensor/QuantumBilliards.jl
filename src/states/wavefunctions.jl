"""
    wavefunction_multi(ks::Vector{T},vec_us::Vector{Vector{T}},vec_bdPoints::Vector{BoundaryPoints{T}},billiard::Bi;b::Float64=5.0,inside_only::Bool=true,fundamental=true,MIN_CHUNK=4_096) where {Bi<:AbsBilliard,T<:Real}

Constructs a sequence of 2D wavefunctions as matrices over the same sized grid for easier computation of matrix elements. The matrices are constructed via the boundary integral.

# Arguments
- `ks`: Vector of eigenvalues.
- `vec_bdPoints`: Vector of `BoundaryPoints` objects, one for each eigenvalue.
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
                    Psi_flat[idx]=ϕ_float32_bessel(x,y,k,bdPoints,us) # Do it with floating point bessel computation, no need for double_precision here, and only for interior points
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
                    Psi_flat[idx]=ϕ_float32_bessel(x,y,k,bdPoints,us) # Do it with floating point bessel computation, no need for double_precision here, and only for interior points
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
        phase_fix_real(psi::AbstractArray{Complex{T}}) where {T<:Real} -> (AbstractArray{Complex{T}}, T)

Given a complex wavefunction `psi` that is expected to be real up to a global phase, find the global phase that makes it as close to real as possible and return the phase-corrected wavefunction along with the phase angle. Plotting separately the real and imaginary parts of the original and phase-corrected wavefunctions can be useful for verifying that the phase correction worked as intended.

# Arguments
- `psi::AbstractArray{Complex{T}}`: The input complex wavefunction array.

# Returns
- `(AbstractArray{Complex{T}}, T)`: A tuple containing the phase-corrected wavefunction and the phase angle that was applied. The corrected wavefunction should have a zero reduced imaginary part if the original wavefunction is correct.
"""
function phase_fix_real(psi::AbstractArray{Complex{T}}) where {T<:Real}
    s=sum(psi.^2)
    θ=0.5*Base.angle(s)
    return psi.*exp(-im*θ),θ
end

"""
    wavefunction_multi_cfie(ks::Vector{T},vec_us::Vector{<:AbstractVector},vec_comps::Vector{Vector{BoundaryPointsCFIE{T}}},billiard::Bi;b::Float64=5.0,inside_only::Bool=true,fundamental::Bool=false,MIN_CHUNK::Int=4096,float32_bessel::Bool=false) where {Bi<:AbsBilliard,T<:Real}

Construct a sequence of 2D wavefunction matrices for CFIE_kress / CFIE_alpert on a common grid.

# Arguments
- `ks`         : eigenvalues
- `vec_us`     : boundary densities, one per eigenstate
- `vec_comps`  : CFIE_kress / CFIE_alpert boundary discretizations, one per eigenstate
- `billiard`   : billiard geometry

# Keyword arguments
- `b`                  : grid density scaling
- `inside_only`        : compute only inside the billiard
- `fundamental`        : use fundamental domain limits if desired
- `MIN_CHUNK`          : minimum masked points per thread chunk
- `float32_bessel`     : use Float32 Hankel evaluations

# Returns
- `Psi2ds` : vector of wavefunction matrices
- `x_grid` : x-coordinates of the grid
- `y_grid` : y-coordinates of the grid
"""
function wavefunction_multi(ks::Vector{T},vec_us::Vector{<:AbstractVector},vec_comps::AbstractVector,billiard::Bi;b::Float64=5.0,inside_only::Bool=true,fundamental::Bool=false,MIN_CHUNK::Int=4096,float32_bessel::Bool=false) where {Bi<:AbsBilliard,T<:Real}
    kmax=maximum(ks)
    L=billiard.length
    xlim,ylim=boundary_limits(billiard.full_boundary;grd=max(1000,round(Int,kmax*L*b/(2*pi)))) # this accepts vector of curves, so it works for cfie as well
    dx=xlim[2]-xlim[1]
    dy=ylim[2]-ylim[1]
    nx=max(round(Int,kmax*dx*b/(2π)),512)
    ny=max(round(Int,kmax*dy*b/(2π)),512)
    x_grid=collect(T,range(xlim[1],xlim[2],length=nx))
    y_grid=collect(T,range(ylim[1],ylim[2],length=ny))
    pts=collect(SVector(x,y) for y in y_grid for x in x_grid)
    npts=length(pts)
    pts_mask=inside_only ? points_in_billiard_polygon(pts,billiard,round(Int,sqrt(npts));fundamental_domain=fundamental) : fill(true,npts)
    pts_masked_indices=findall(pts_mask)
    nmask=length(pts_masked_indices)
    NT=Threads.nthreads()
    NT_eff=max(1,min(NT,cld(nmask,MIN_CHUNK)))
    S=eltype(vec_us[1])
    nstates=length(ks)
    Psi2ds=Vector{Matrix{S}}(undef,nstates)
    function _ensure_cfie_vec(x) # annoying type casting due to being able to have holes with panels etc. Really hate Julia's type sometimes
        x isa Vector{BoundaryPointsCFIE} && return x
        x isa BoundaryPointsCFIE && return [x]
        error("Expected CFIE boundary data, got $(typeof(x))")
    end
    caches=Vector{CFIEWavefunctionCache{T}}(undef,nstates)
    @inbounds for i in 1:nstates
        comps=_ensure_cfie_vec(vec_comps[i])
        caches[i]=flatten_cfie_wavefunction_cache(comps)
    end
    Psi_flat=zeros(S,nx*ny) # flat workspace reused per state
    progress=Progress(nstates,desc="Constructing CFIE wavefunction matrices...")
    q,r=divrem(nmask,NT_eff)
    @inbounds for i in eachindex(ks)
        k=ks[i]
        cache=caches[i]
        us=vec_us[i]
        fill!(Psi_flat,zero(S))
        @fastmath begin
            Threads.@threads :static for t in 1:NT_eff
                lo=(t-1)*q+min(t-1,r)+1
                hi=lo+q-1+(t<=r ? 1 : 0)
                for jj in lo:hi
                    idx=pts_masked_indices[jj]
                    p=pts[idx]
                    Psi_flat[idx]=ϕ_cfie(p[1],p[2],k,cache,us;float32_bessel=float32_bessel)
                end
            end
        end
        Psi2ds[i]=copy(reshape(Psi_flat,nx,ny))
        next!(progress)
    end
    for i in eachindex(Psi2ds)
        Psi2ds[i],_=phase_fix_real(Psi2ds[i])
        nrm=sqrt(sum(abs2,Psi2ds[i][pts_masked_indices]))
        Psi2ds[i]./=nrm
    end
    return Psi2ds,x_grid,y_grid
end

###########################################################################
###########################################################################
###########################################################################

function plot_curve!(ax,crv::AbsRealCurve;plot_normal=true,dens=20.0,color_crv=:grey,linewidth=0.75)
    L=crv.length
    grid=max(round(Int,L*dens),3)
    t=range(0.0,1.0,grid)
    pts=curve(crv,t)
    lines!(ax,pts,color=color_crv,linewidth=linewidth)
    if plot_normal
        ns=normal_vec(crv,t)
        arrows!(ax,getindex.(pts,1),getindex.(pts,2),getindex.(ns,1),getindex.(ns,2),color=:black,lengthscale=0.1)
    end
    ax.aspect=DataAspect()
end

function plot_curve!(ax,crv::AbsVirtualCurve;plot_normal=false,dens=10.0,color_crv=:grey,linewidth=0.75)
    L=crv.length
    grid=max(round(Int,L*dens),3)
    t=range(0.0,1.0,grid)
    pts=curve(crv,t)
    lines!(ax,pts,color=color_crv,linestyle=:dash,linewidth=linewidth)
    if plot_normal
        ns=normal_vec(crv,t)
        arrows!(ax,getindex.(pts,1),getindex.(pts,2),getindex.(ns,1),getindex.(ns,2),color=:black,lengthscale=0.1)
    end
    ax.aspect=DataAspect()
end

function _plot_boundary_object!(ax,boundary;dens=100.0,plot_normal=true,color_crv=:grey,linewidth=0.75)
    if boundary isa AbstractVector
        for obj in boundary
            _plot_boundary_object!(ax,obj;dens=dens,plot_normal=plot_normal,color_crv=color_crv,linewidth=linewidth)
        end
    else
        plot_curve!(ax,boundary;dens=dens,plot_normal=plot_normal,color_crv=color_crv,linewidth=linewidth)
    end
    return ax
end

function plot_boundary!(ax,billiard::AbsBilliard;fundamental_domain=true,desymmetrized_full_domain=false,dens=100.0,plot_normal=true,color_crv=:grey,linewidth=0.75)
    if fundamental_domain
        boundary=billiard.fundamental_boundary
    elseif desymmetrized_full_domain
        boundary=billiard.desymmetrized_full_boundary
    else
        boundary=billiard.full_boundary
    end
    _plot_boundary_object!(ax,boundary;dens=dens,plot_normal=plot_normal,color_crv=color_crv,linewidth=linewidth)
    return ax
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
    for i in eachindex(Psi2ds)
        ψ=Psi2ds[i]
        amax=maximum(abs,ψ)
        if amax>0
            @. ψ=real(ψ)/amax
        else
            @. ψ=real(ψ)
        end
    end
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
        hm=heatmap!(ax,x_grid,y_grid,Psi2ds[j],colormap=:balance,colorrange=(-1,1))
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
    for i in eachindex(Psi2ds)
        ψ=Psi2ds[i]
        amax=maximum(abs,ψ)
        if amax>0
            @. ψ=real(ψ)/amax
        else
            @. ψ=real(ψ)
        end
    end
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
        hm=heatmap!(ax,x_grid[j],y_grid[j],Psi2ds[j],colormap=:balance,colorrange=(-1,1))
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

################################################
################ WITH HUSIMIS ##################
################################################

# helper to concatenate multiple Husimi components into one big Husimi matrix with the qs grids concatenated with offsets to avoid overlap, and also return the seam locations for plotting vertical lines to indicate the component boundaries. This is useful for plotting multiple Husimi components together in a single plot while visually separating them along the q-axis.
function _husimi_concat_with_separation(Hs::Vector{<:AbstractMatrix{T}},qs_list::Vector{<:AbstractVector{T}}) where {T<:Real}
    length(Hs)==length(qs_list) || error("Hs and qs_list must have same length")
    isempty(Hs) && return zeros(T,0,0),T[],T[]
    ncomp=length(Hs)
    nps=size(Hs[1],2)
    for a in 1:ncomp
        size(Hs[a],1)==length(qs_list[a]) || error("Component $a has inconsistent Husimi/q-grid sizes")
        size(Hs[a],2)==nps || error("All Husimi components must share the same p-grid")
    end
    qcat=T[]
    seams=T[]
    Hblocks=Vector{Matrix{T}}(undef,ncomp)
    qoff=zero(T)
    for a in 1:ncomp
        qa=collect(qs_list[a])
        Ha=Matrix{T}(Hs[a])
        if a>1
            push!(seams,qoff)
        end
        qa_shift=qa.+qoff
        append!(qcat,qa_shift)
        Hblocks[a]=Ha
        qoff=isempty(qa_shift) ? qoff : qa_shift[end]
    end
    Hcat=vcat(Hblocks...)
    return Hcat,qcat,seams
end

# helper to plot vertical lines on the Husimi plot to indicate the separation between different Husimi components when they are concatenated together with offsets. This function takes in the axis to plot on, the seam locations, and the p-grid values to determine the y-limits of the lines.
function _plot_husimi_separation_lines!(ax,seams::AbstractVector{T},ps::AbstractVector{T};color=:cyan,linewidth=3,linestyle=:dash) where {T<:Real}
    isempty(seams) && return ax
    ymin=minimum(ps)
    ymax=maximum(ps)
    for s0 in seams
        lines!(ax,[s0,s0],[ymin,ymax],color=color,linewidth=linewidth,linestyle=linestyle)
    end
    return ax
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
- `seam_color=:cyan`: The color of the seam lines separating the Husimi components.
- `seam_linewidth=2`: The linewidth of the seam lines separating the Husimi components.

 # Returns
- `f::Figure`: A Figure object containing the grid of wavefunctions.
"""
function plot_wavefunctions_with_husimi_BATCH(ks::Vector,Psi2ds::Vector,x_grid::Vector,y_grid::Vector,Hs_list::Vector{<:Vector{<:AbstractMatrix}},ps_list::Vector,qs_list::Vector{<:Vector{<:AbstractVector}},billiard::Bi; b::Float64=5.0,width_ax::Integer=300,height_ax::Integer=300,max_cols::Integer=6,fundamental=true,custom_label::Vector{String}=String[],seam_color=:cyan,seam_linewidth=4) where {Bi<:AbsBilliard}
    for i in eachindex(Psi2ds)
        ψ=Psi2ds[i]
        amax=maximum(abs,ψ)
        if amax>0
            @. ψ=real(ψ)/amax
        else
            @. ψ=real(ψ)
        end
    end
    L=billiard.length
    if fundamental
        xlim,ylim=boundary_limits(billiard.fundamental_boundary;grd=max(1000,round(Int,maximum(ks)*L*b/(2*pi))))
    else
        xlim,ylim=boundary_limits(billiard.full_boundary;grd=max(1000,round(Int,maximum(ks)*L*b/(2*pi))))
    end
    n_rows=ceil(Int,length(ks)/max_cols)
    f=Figure(resolution=(3*width_ax*max_cols,1.5*height_ax*n_rows),size=(3*width_ax*max_cols,1.5*height_ax*n_rows))
    row=1
    col=1
    @showprogress desc="Plotting wavefunctions and husimi..." for j in eachindex(ks)
        title=isempty(custom_label) ? "$(ks[j])" : custom_label[j]
        local ax=Axis(f[row,col][1,1],title=title,aspect=DataAspect(),width=width_ax,height=height_ax)
        local ax_h=Axis(f[row,col][1,2],width=width_ax,height=height_ax)
        heatmap!(ax,x_grid,y_grid,Psi2ds[j],colormap=:balance,colorrange=(-1,1))
        plot_boundary!(ax,billiard,fundamental_domain=fundamental,plot_normal=false)
        Hcat,qcat,seams=_husimi_concat_with_separation(Hs_list[j],qs_list[j])
        heatmap!(ax_h,qcat,ps_list[j],Hcat;colormap=Reverse(:gist_heat))
        _plot_husimi_separation_lines!(ax_h,seams,ps_list[j];color=seam_color,linewidth=seam_linewidth)
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
- `seam_color=:cyan`: The color of the seam lines separating the Husimi components.
- `seam_linewidth=2`: The linewidth of the seam lines separating the Husimi components.

 # Returns
- `f::Figure`: A Figure object containing the grid of wavefunctions.
"""
function plot_wavefunctions_with_husimi_BATCH(ks::Vector,Psi2ds::Vector,x_grid::Vector,y_grid::Vector,Hs_list::Vector,ps_list::Vector,qs_list::Vector,billiard::Bi,us_all::Vector,s_vals_all::Vector;b::Float64=5.0,width_ax::Integer=300,height_ax::Integer=300,max_cols::Integer=6,fundamental=true,custom_label::Vector{String}=String[],seam_color=:cyan,seam_linewidth=4) where {Bi<:AbsBilliard}
    for i in eachindex(Psi2ds)
        ψ=Psi2ds[i]
        amax=maximum(abs,ψ)
        if amax>0
            @. ψ=real(ψ)/amax
        else
            @. ψ=real(ψ)
        end
    end
    L=billiard.length
    if fundamental
        xlim,ylim=boundary_limits(billiard.fundamental_boundary;grd=max(1000,round(Int,maximum(ks)*L*b/(2*pi))))
    else
        xlim,ylim=boundary_limits(billiard.full_boundary;grd=max(1000,round(Int,maximum(ks)*L*b/(2*pi))))
    end
    n_rows=ceil(Int,length(ks)/max_cols)
    f=Figure(resolution=(3*width_ax*max_cols,2*height_ax*n_rows),size=(3*width_ax*max_cols,2*height_ax*n_rows))
    row=1
    col=1
    @showprogress desc="Plotting wavefunctions and husimi..." for j in eachindex(ks)
        title=isempty(custom_label) ? "$(ks[j])" : custom_label[j]
        local ax_wave=Axis(f[row,col][1,1],title=title,aspect=DataAspect(),width=width_ax,height=height_ax)
        heatmap!(ax_wave,x_grid,y_grid,Psi2ds[j],colormap=:balance,colorrange=(-1,1))
        plot_boundary!(ax_wave,billiard,fundamental_domain=fundamental,plot_normal=false)
        xlims!(ax_wave,xlim)
        ylims!(ax_wave,ylim)
        local ax_h=Axis(f[row,col][1,2],width=width_ax,height=height_ax)
        Hcat,qcat,seams=_husimi_concat_with_separation(Hs_list[j],qs_list[j])
        heatmap!(ax_h,qcat,ps_list[j],Hcat;colormap=Reverse(:gist_heat))
        _plot_husimi_separation_lines!(ax_h,seams,ps_list[j];color=seam_color,linewidth=seam_linewidth)
        local ax_boundary=Axis(f[row,col][2,1:2],xlabel="s",ylabel="u(s)",width=2*width_ax,height=height_ax/2)
        lines!(ax_boundary,s_vals_all[j],real.(us_all[j]),label="Re u(s)",linewidth=2)
        maximum(abs.(imag.(us_all[j])))>0 && lines!(ax_boundary,s_vals_all[j],imag.(us_all[j]),label="Im u(s)",linewidth=2,linestyle=:dash)
        !isempty(seams) && vlines!(ax_boundary,seams,color=seam_color,linewidth=seam_linewidth,linestyle=:dash)
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

###################################################################################
################# WAVEFUNCTION CONSTRUTION FOR BASIS TYPE METHODS #################
###################################################################################

"""
    compute_psi(vec::Vector, k::T, billiard::Bi, basis::Ba, x_grid, y_grid; inside_only=true, memory_limit = 10.0e9) where {Bi<:AbsBilliard, Ba<:AbsBasis, T<:Real}

Computs the wavefunction as a `Matrix` on a grid formed by the vectors `x_grid` and `y_grid`. This is a lower level function for wrappers that require the construction of a wavefunction from a vector of linear expansion coefficients and being constructed on a common grid.

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
function compute_psi(vec::Vector,k::T,billiard::Bi,basis::Ba,x_grid::Vector,y_grid::Vector;inside_only=true,memory_limit=10.0e9) where {Bi<:AbsBilliard, Ba<:AbsBasis,T<:Real}
    eps=set_precision(vec[1])
    sz=length(x_grid)*length(y_grid)
    pts=collect(SVector(x,y) for y in y_grid for x in x_grid)
    if inside_only
        pts_mask=points_in_billiard_polygon(pts,billiard,round(Int,sqrt(sz));fundamental_domain=true)
        pts=pts[pts_mask]
    end
    n_pts=length(pts)
    type=eltype(vec)
    memory=sizeof(type)*basis.dim*n_pts #estimate max memory needed for the matrices
    Psi=zeros(type,sz)
    if memory<memory_limit
        B=basis_matrix(basis,k,pts)
        Psi_pts=B*vec
        if inside_only
            Psi[pts_mask].=Psi_pts
        else
            Psi.=Psi_pts
        end
    else
        println("Warning: memory limit of $(Base.format_bytes(memory_limit)) exceded $(Base.format_bytes(memory)).")
        if inside_only
            for i in eachindex(vec)
                if abs(vec[i])>eps 
                    Psi[pts_mask].+=vec[i].*basis_fun(basis,i,k,pts)
                end
            end
        else
            for i in eachindex(vec)
                if abs(vec[i])>eps 
                    Psi.+=vec[i].*basis_fun(basis,i,k,pts)
                end
            end
        end
    end
    return Psi
end

"""
    compute_psi(state::S,x_grid::Vector{T},y_grid::Vector{T};inside_only=true,memory_limit=10.0e9) where {S<:AbsState,T<:Real}

Constructs the wavefunction as a `Matrix` from an `Eigenstate` struct on a grid of vectors `x_grid` and `y_grid`.

# Arguments
- `state::S`: An `Eigenstate` struct with a `vec` field representing the wavefunction, a `k_basis` field representing the wavefunction basis, a `basis` field representing the basis set, a `billiard` field representing the billiard.
- `x_grid::Vector{T}`: A vector of `x` coordinates on which to evaluate the wavefunction.
- `y_grid::Vector{T}`: A vector of `y` coordinates on which to evaluate the wavefunction.
- `inside_only::Bool` (optional, default `true`): If `true`, only evaluate the wavefunction inside the billiard.
- `memory_limit::Real` (optional, default `10.0e9`): The maximum memory limit in bytes to use for constructing the wavefunction. If the memory required exceeds this multithreading is disabled.

# Returns
- `Psi::Matrix`: A `Matrix` representing the wavefunction evaluated on the grid.
"""
function compute_psi(state::S,x_grid::Vector{T},y_grid::Vector{T};inside_only=true,memory_limit=10.0e9) where {S<:AbsState,T<:Real}
    compute_psi(state.vec,state.k,state.billiard,state.basis,x_grid,y_grid;inside_only,memory_limit)
end

"""
    wavefunction(vec::Vector, k::T, billiard::Bi, basis::Ba; b=5.0, inside_only=true, fundamental_domain = true, memory_limit = 10.0e9) where {Bi<:AbsBilliard, Ba<:AbsBasis, T<:Real}     

Computes the wavefunction matrix and the x and y grids for heatmap plotting. It is contructed from the vec=X[i] of `StateData` and not directly from `StateData`.

# Arguments
- `vec::Vector{<:Real}`: The vector of coefficients of the basis expansion of the wavefunction. It's length determines the resizeing of the `basis`.
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
function wavefunction(vec::Vector,k::T,billiard::Bi,basis::Ba;b=5.0,inside_only=true,fundamental_domain=true,memory_limit=10.0e9) where {Bi<:AbsBilliard, Ba<:AbsBasis,T<:Real}     
    dim=length(vec)
    dim=rescale_rpw_dimension(basis,dim)
    basis=resize_basis(basis,billiard,dim,k)
    symmetries=basis.symmetries
    type=eltype(vec)
    L=billiard.length
    xlim,ylim=boundary_limits(billiard.fundamental_boundary;grd=max(1000,round(Int,k*L*b/(2*pi))))
    dx=xlim[2]-xlim[1]
    dy=ylim[2]-ylim[1]
    nx=max(round(Int,k*dx*b/(2*pi)),512)
    ny=max(round(Int,k*dy*b/(2*pi)),512)
    x_grid::Vector{type}=collect(type,range(xlim...,nx))
    y_grid::Vector{type}=collect(type,range(ylim...,ny))
    Psi::Vector{type}=compute_psi(vec,k,billiard,basis,x_grid,y_grid; inside_only=inside_only, memory_limit=memory_limit) 
    Psi2d::Array{type,2}=reshape(Psi,(nx,ny))
    return Psi2d,x_grid,y_grid
end

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
function wavefunction(state::S;b=5.0,inside_only=true,fundamental_domain=true,memory_limit=10.0e9) where {S<:AbsState}    
    wavefunction(state.vec,state.k,state.billiard,state.basis;b=b,inside_only=inside_only,fundamental_domain=fundamental_domain,memory_limit=memory_limit)
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
function wavefunctions(X::Vector,ks::Vector,billiard::Bi,basis::Ba;b=5.0,inside_only=true,fundamental_domain=true,memory_limit=10.0e9) where {Bi<:AbsBilliard, Ba<:AbsBasis}
    vec_Psi=Vector{Matrix}(undef,length(ks))
    vec_xs=Vector{Vector}(undef,length(ks))
    vec_ys=Vector{Vector}(undef,length(ks))
    p = Progress(length(ks),1)
    Threads.@threads for i in eachindex(ks) 
        vec=X[i]
        k=ks[i]
        Psi2d,x_grid,y_grid=wavefunction(vec,k,billiard,basis;b=b,inside_only=inside_only,fundamental_domain=fundamental_domain,memory_limit=memory_limit)
        vec_Psi[i]=Psi2d
        vec_xs[i]=x_grid
        vec_ys[i]=y_grid
        next!(p)
    end
    return vec_Psi,vec_xs,vec_ys
end

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
function wavefunctions(state_data::StateData,billiard::Bi,basis::Ba;b=5.0,inside_only=true,fundamental_domain=true,memory_limit=10.0e9) where {Bi<:AbsBilliard, Ba<:AbsBasis}
    ks=state_data.ks
    tens=state_data.tens
    X=state_data.X
    Psi2ds=Vector{Matrix{eltype(ks)}}(undef,length(ks))
    x_grids=Vector{Vector{eltype(ks)}}(undef,length(ks))
    y_grids=Vector{Vector{eltype(ks)}}(undef,length(ks))
    for i in eachindex(ks) 
        vec=X[i] # vector of vectors
        dim=length(vec)
        dim=rescale_rpw_dimension(basis, dim)
        new_basis=resize_basis(basis, billiard, dim, ks[i])
        state=Eigenstate(ks[i], vec, tens[i], new_basis, billiard)
        Psi2d,x_grid,y_grid=wavefunction(state;b=b,inside_only=inside_only,fundamental_domain=fundamental_domain,memory_limit=memory_limit)
        Psi2ds[i]=Psi2d
        x_grids[i]=x_grid
        y_grids[i]=y_grid
    end
    return ks,Psi2ds,x_grids,y_grids
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
function wavefunction(state::BasisState;xlim =(-2.0,2.0),ylim=(-2.0,2.0),b=5.0) 
    let k=state.k,basis=state.basis      
        type=eltype(state.vec)
        #TODO try to find a lazy way to do this
        dx=xlim[2]-xlim[1]
        dy=ylim[2]-ylim[1]
        nx=max(round(Int,k*dx*b/(2*pi)),512)
        ny=max(round(Int,k*dy*b/(2*pi)),512)
        x_grid::Vector{type}=collect(type,range(xlim...,nx))
        y_grid::Vector{type}=collect(type,range(ylim...,ny))
        pts_grid=[SVector(x,y) for y in y_grid for x in x_grid]
        Psi::Vector{type}=basis_fun(basis,state.idx,k,pts_grid) 
        Psi2d::Array{type,2}=reshape(Psi,(nx,ny))
        return Psi2d,x_grid,y_grid
    end
end

#TODO this can be optimized
"""
    compute_psi(state_bundle::S, x_grid::Vector{T}, y_grid::Vector{T}; inside_only=true, memory_limit = 10.0e9) where {S<:EigenstateBundle, T<:Real}

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
function compute_psi(state_bundle::S,x_grid::Vector{T},y_grid::Vector{T};inside_only=true,memory_limit=10.0e9) where {S<:EigenstateBundle,T<:Real}
    let k=state_bundle.k_basis,basis=state_bundle.basis,billiard=state_bundle.billiard,X=state_bundle.X #basis is correct size
        sz=length(x_grid)*length(y_grid)
        pts=collect(SVector(x,y) for y in y_grid for x in x_grid)
        if inside_only
            pts_mask=is_inside(billiard,pts)
            pts=pts[pts_mask]
        end
        n_pts=length(pts)
        n_states=length(state_bundle.ks)
        type=eltype(state_bundle.X) #estimate max memory needed for the matrices
        memory=sizeof(type)*basis.dim*n_pts
        Psi_bundle=zeros(type,(sz,n_states))  #Vector of results
        if memory<memory_limit
            B=basis_matrix(basis,k,pts)
            Psi_pts=B*X
            Psi_bundle[pts_mask,:].=Psi_pts
        else
            println("Warning: memory limit of $(Base.format_bytes(memory_limit)) exceded $(Base.format_bytes(memory)).")
            Psi_pts=zeros(type,(n_pts,n_states))
            for i in 1:basis.dim
                bf=basis_fun(basis,i,k,pts) #vector of length n_pts
                for j in 1:n_states
                    Psi_pts[:,j].+=X[i,j].*bf
                end
            end
            if inside_only
                Psi_bundle[pts_mask,:]=Psi_pts
            else
                Psi_bundle=Psi_pts
            end
        end
        return Psi_bundle #this is a matrix 
    end
end

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
function wavefunction(state_bundle::S;b=5.0,inside_only=true,fundamental_domain=true,memory_limit=10.0e9) where {S<:EigenstateBundle}
    k=state_bundle.k_basis
    billiard=state_bundle.billiard
    symmetries=state_bundle.basis.symmetries
    T=Float64
    L=billiard.length
    xlim,ylim=boundary_limits(billiard.fundamental_boundary;grd=max(1000,round(Int,k*L*b/(2*pi))))
    dx=xlim[2]-xlim[1]
    dy=ylim[2]-ylim[1]
    nx=max(round(Int,k*dx*b/(2*pi)),512)
    ny=max(round(Int,k*dy*b/(2*pi)),512)
    x_grid::Vector{T}=collect(range(xlim...,nx))
    y_grid::Vector{T}=collect(range(ylim...,ny))
    Psi_bundle::Matrix{Complex{T}}=compute_psi(state_bundle,x_grid,y_grid;inside_only=inside_only,memory_limit=memory_limit)
    # Reshape each column of Psi_bundle into a matrix
    Psi2d::Vector{Matrix{Complex{T}}}=[reshape(Psi,(nx,ny)) for Psi in eachcol(Psi_bundle)]
    return Psi2d,x_grid,y_grid
end