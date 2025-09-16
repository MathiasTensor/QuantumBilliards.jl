using Bessels, LinearAlgebra, ProgressMeter, FFTW, SparseArrays

############### WAVEPACKET EVOLUTION VIA BASIS SET ###############

"""
    Wavepacket{T<:Real}

Contains the initialization parameters that construct the Gaussian wavepacket.

# Arguments
- `x0::T`: Initial x position of the Gaussian wavepacket.
- `y0::T`: Initial y position of the Gaussian wavepacket.
- `sigma_x::T`: Standard deviation of the Gaussian wavepacket in the x direction.
- `sigma_y::T`: Standard deviation of the Gaussian wavepacket in the y direction.
- `px0::T`: momentum in the x direction.
- `py0::T`: momentum in the y direction.
- `ħ::T`: Reduced Planck constant
"""
struct Wavepacket{T<:Real}
    x0::T
    y0::T
    sigma_x::T
    sigma_y::T
    px0::T
    py0::T
    ħ::T
end

"""
    gaussian_wavepacket_2d(x::Vector{T}, y::Vector{T}, packet::Wavepacket{T}) where {T<:Real}

Generates a 2D Gaussian wavepacket in coordinate space on a grid of `(x,y)` values and then returns in on a grid as a `Matrix`. It is not normalized!

```math
f(x,y) = exp(-(x-x0)^2/2σx-(y-y0)^2/2σy)*exp(-i*px*(x-x0)/ħ)*exp(-i*py(y-y0)ħ)
```

# Arguments
- `x::Vector{T}, y::Vector{T}`: The spatial coordinates x and y as vectors (separately) that form a grid.
- `packet::Wavepacket`: Gaussian wavepacket struct containing the params.

# Returns
- `Matrix{Complex{T}}`: The value at those params on a 2D grid.
"""
function gaussian_wavepacket_2d(x::Vector{T},y::Vector{T},packet::Wavepacket{T}) where {T<:Real}
    x0=packet.x0;y0=packet.y0;sigma_x=packet.sigma_x;sigma_y=packet.sigma_y;px0=packet.px0;py0=packet.py0;ħ=packet.ħ
    dx=x.-x0
    dy=y.-y0
    gx_amp=@. exp(-dx^2/(2*sigma_x^2))
    gy_amp=@. exp(-dy^2/(2*sigma_y^2))
    gx_phase=@. exp(im*px0*dx/ħ)
    gy_phase=@. exp(im*py0*dy/ħ)
    gx=gx_amp.*gx_phase
    gy=gy_amp.*gy_phase
    result=(gx*transpose(gy)) # Outer product: result[i,j] = gx[i] * gy[j]
    return result
end

"""
    gaussian_wavepacket_2d(pts_in_billiard::Vector{SVector{2,T}},packet::Wavepacket{T}) where {T<:Real}

Computes the Gaussian wavepacket for a set of points inside the billiard. It is not normalized!

# Arguments
- `pts_in_billiard::Vector{SVector{2,T}}`: Points inside the billiard.
- `packet::Wavepacket`: Gaussian wavepacket struct containing the params.

# Returns
- `Vector{Complex{T}}`: Gaussian wavepacket evaluated at the points.
"""
function gaussian_wavepacket_2d(pts_in_billiard::Vector{SVector{2,T}},packet::Wavepacket{T}) where {T<:Real}
    x0=packet.x0;y0=packet.y0;sigma_x=packet.sigma_x;sigma_y=packet.sigma_y;px0=packet.px0;py0=packet.py0;ħ=packet.ħ
    xs=getindex.(pts_in_billiard,1)  # x-coordinates
    ys=getindex.(pts_in_billiard,2)  # y-coordinates
    dx=xs.-x0
    dy=ys.-y0
    amp=@. exp(-dx^2/(2*sigma_x^2)-dy^2/(2*sigma_y^2))
    phase=@. exp(im*(px0*dx+py0*dy)/ħ)
    return (amp.*phase)
end

@inline function _blockrange(t::Int,n::Int,nt::Int)
    q,r=divrem(n,nt)
    lo=(t-1)*q+min(t-1,r)+1
    hi=lo+q-1+(t<=r ? 1 : 0)
    return lo:hi
end

"""
    gaussian_coefficients(ks::Vector{T}, vec_us::Vector{Vector{T}}, vec_bdPoints::Vector{BoundaryPoints{T}}, 
                          billiard::Bi, packet::Wavepacket{T}; b::Float64=5.0) 
                          -> (Vector{Matrix{T}}, Vector{T}, Vector{T}, Vector{T}, BitVector, T, T)

Computes the eigenfunction matrices and their overlaps with a Gaussian wavepacket in a given billiard system. This gives us the coefficients and the basis eigenfunction for evolution and visualization. For constructing the basis elements uses single precision for efficiency.

# Arguments
- `ks::Vector{T}`: Vector of wavenumbers associated with the eigenfunctions.
- `vec_us::Vector{Vector{T}}`: Vector containing the eigenfunction coefficients.
- `vec_bdPoints::Vector{BoundaryPoints{T}}`: Boundary data for the eigenfunctions.
- `billiard::Bi`: The billiard system, which defines the boundary conditions.
- `packet::Wavepacket{T}`: The initial Gaussian wavepacket to be projected onto the eigenfunctions.
- `b::Float64=5.0`: Scaling parameter for the spatial resolution grid.
- `fundamental_domain::Bool=true`: If we consider only the desymmetrized billiard domain.

# Returns
- `Psi2ds::Vector{Matrix{T}}`: List of wavefunction matrices corresponding to the given wavenumbers.
- `overlaps::Vector{Complex{T}}`: Overlap coefficients of each eigenfunction with the initial Gaussian wavepacket.
- `x_grid::Vector{T}`: X-coordinates of the spatial grid.
- `y_grid::Vector{T}`: Y-coordinates of the spatial grid.
- `pts_mask::BitVector`: Mask indicating which grid points are inside the billiard.
- `dx::T`: Grid spacing in the x-direction.
- `dy::T`: Grid spacing in the y-direction.
"""
function gaussian_coefficients(ks::Vector{T},vec_us::Vector{Vector{T}},vec_bdPoints::Vector{BoundaryPoints{T}},billiard::Bi,packet::Wavepacket{T};b::Float64=5.0,fundamental_domain=true) where {Bi<:AbsBilliard,T<:Real}
    k_max=maximum(ks) # the wavefunction size must be the same for all k, therefore largest k size for all since we must resolve many points per wavelength
    L=billiard.length
    if fundamental_domain
        xlim,ylim=boundary_limits(billiard.fundamental_boundary;grd=max(1000,round(Int,k_max*L*b/(2*pi))))
    else
        xlim,ylim=boundary_limits(billiard.full_boundary;grd=max(1000,round(Int,k_max*L*b/(2*pi))))
    end
    dx=xlim[2]-xlim[1] # these represent the x and y limits
    dy=ylim[2]-ylim[1]
    nx,ny=max(round(Int,k_max*dx*b/(2*pi)),512),max(round(Int,k_max*dy*b/(2*pi)),512)
    x_grid,y_grid=collect(T,range(xlim..., nx)),collect(T,range(ylim..., ny))
    dx=x_grid[2]-x_grid[1] # these are now rebranded to grid differences
    dy=y_grid[2]-y_grid[1]
    w=dx*dy # the area element for sums, since linear grid approximates the integration measure
    pts=collect(SVector(x,y) for y in y_grid for x in x_grid) # all possible points on the maximal rectangle grid domain of the billiard
    sz=length(pts)
    pts_mask=points_in_billiard_polygon(pts,billiard,round(Int,sqrt(sz));fundamental_domain=fundamental_domain)
    pts_masked_indices=findall(pts_mask) # all these (i,j) in pts_masked_indices are inside the billiard domain
    Psi2ds=Vector{Matrix{T}}(undef,length(ks)) # Preallocate vector of all wavefunction matrices
    overlaps=Vector{Complex{T}}(undef,length(ks)) # Preallocate vector of all overlap coefficients
    progress=Progress(length(ks),desc="Constructing wavefunction matrices and computing overlaps...")
    gaussian_mat=gaussian_wavepacket_2d(x_grid,y_grid,packet) # create a 2d mat, unnormalized
    G=vec(gaussian_mat) # single vector for fast indexing
    G_norm=w*sum(abs2,@view G[pts_masked_indices]) # this is sum( G[i,j]*dx*dy for (i,j) in pts_masked_indices), where we do sum only on the interior points
    G_norm2=G_norm>zero(T) ? sqrt(G_norm) : one(T)
    G./=G_norm2  # now sum( |G[i,j|^2*dx*dy for (i,j) in pts_masked_indices ) ≈ 1
    Psi_flat=zeros(T,sz) # overwritten each iteration since pts_masked_indices is the same for each k in ks
    NT=Threads.nthreads()
    nmask=length(pts_masked_indices)
    MIN_CHUNK=4_096 # keep ≥ this many points per thread
    NT_eff=max(1,min(NT,cld(nmask,MIN_CHUNK)))
    thread_overlaps=Vector{Complex{T}}(undef,NT_eff) # each thread will have it's own calculation of ϕ[idx] and G[idx] and then later sum all the threads. Each thread works independently and no race conditions.
    thread_norm2=Vector{T}(undef,NT_eff) # since this function normalizes both overlaps and wavefunctions we use the same thread safe accumulator logic
    for i in eachindex(ks) # unless thousands of cores never multithread this
        @inbounds begin
            k,bdPoints,us=ks[i],vec_bdPoints[i],vec_us[i]
            #=
            @fastmath begin
                Threads.@threads for j in eachindex(pts_masked_indices) # multithread this one since has most elements to thread over
                    idx=pts_masked_indices[j] # each interior point [idx] -> (x,y)
                    tid=Threads.threadid()  # thread ID for safe accumulation, one per calculation
                    x,y=pts[idx]
                    psi_val=ϕ_float_bessel(x,y,k,bdPoints,us) # Construct the wavefunction value only in the interior points, less expensive than construction wavefunction matrix and then broadcasting product with G. Do it with floating point bessel computation, no need for double_precision here
                    Psi_flat[idx]=psi_val
                    thread_overlaps[tid]+=psi_val*G[idx] # no need for conj since Ψ is real, this is Ψ[i,j]*G[i,j]
                    thread_norm2[tid]+=abs2(psi_val) # accumulate local Ψ value for later normalization
                end
            end
            =#

            @fastmath begin
                Threads.@threads :static for t in 1:NT_eff
                    # compute this thread's block [lo:hi]
                    q,r=divrem(nmask, NT_eff)
                    lo=(t-1)*q+min(t-1,r) + 1
                    hi=lo+q-1+(t <= r ? 1 : 0)
                    local_o=zero(Complex{T})
                    local_n=zero(T)
                    @inbounds for jj in lo:hi
                        idx=pts_masked_indices[jj]
                        x,y=pts[idx]
                        ψ=ϕ_float_bessel(x,y,k,bdPoints,us)
                        Psi_flat[idx]=ψ
                        local_o+=ψ*G[idx] 
                        local_n+=abs2(ψ)
                    end
                    thread_overlaps[t]=local_o
                    thread_norm2[t]=local_n
                end
            end
            
            sum_norm2=sum(thread_norm2) # norm accumulator for a given eigenstate
            norm_i=sqrt(w*sum_norm2) # 1/norm_i*dx*dy, this should give sum( 1/√Norm*dx*dy Ψ^2 ) ≈ 1
            @inbounds @simd for jj in eachindex(pts_masked_indices)
                Psi_flat[pts_masked_indices[jj]]/=norm_i # make the wavefunctions normalized inplace, Ψ -> 1/norm_i * Ψ
            end
            Psi2ds[i]=copy(reshape(Psi_flat,ny,nx)) # also return the normalized wavefunction matrices
            overlaps[i]=sum(thread_overlaps)*(w/norm_i) # from the thread safe local accumulation we then multiply with the dx*dy element due to linear grid. This is 1/norm_i * sum( conj(Ψ) * G ) * w 
            next!(progress)
        end
    end
    a=sum(abs2,overlaps)
    if ((a<0.95) || (a>1.05))
        @error "Σ |overlap|^2 = $(a): basis may be too small or not well resolved."
    end
    @info "numerical Σ |overlap|^2 = $(a), renormalized to 1"
    overlaps./=sqrt(a) # inplace normalize the overlaps to 1
    return Psi2ds,overlaps,x_grid,y_grid,pts_mask,dx,dy
end

"""
    evolution_gaussian_coefficients(coeffs_init::Vector{Complex{T}},ks::Vector{T},ts::Vector{K},ħ::T;m::T=one(T)) where {T<:Real,K<:Real}
    -> Matrix{Complex{T}}

Computes the time evolution of the expansion coefficients of a wavepacket in a quantum billiard. One needs to be careful with the energy scaling as the original `ks` are gotten with ΔΨ=k*Ψ and not ħ^2/(2*m)ΔΨ=k*Ψ = α*ΔΨ=k*Ψ.

# Description
Given an initial set of expansion coefficients `coeffs_init`, corresponding wavenumbers `ks`, 
and time values `ts`, this function computes the time-evolved coefficients based on the 
Schrödinger equation solution:

`c_n(t) = c_n(0) e^{-i*E_n*t/ħ}`

# Arguments
- `coeffs_init::Vector{Complex{T}}`: The initial expansion coefficients of the wavepacket in the eigenbasis.
- `ks::Vector{T}`: The wavenumbers associated with each eigenfunction.
- `ts::Vector{K}`: A vector of time points at which to evaluate the evolved coefficients.
- `ħ::T`: Reduced Planck constant for the correct energy scaling.
- `m::T=one(T)`: Mass in the Schrodinger equation.

# Returns
- `Matrix{Complex{T}}`: A matrix mat of shape (length(ts),length(ks)) where mat[i, j] represents the coefficient for the j-th eigenmode at time ts[i].
"""
function evolution_gaussian_coefficients(coeffs_init::Vector{Complex{T}},ks::Vector{T},ts::Vector{K},ħ::T;m::T=one(T)) where {T<:Real,K<:Real}
    α=ħ^2/(2*m)
    N=length(coeffs_init)
    @assert N==length(ks) "The number of coefficients must match the number of eigenvalues, and they need to be correctly ordered: ks[i] -> Psi2d[i] -> coefficient[i]"
    Es=[α*k^2 for k in ks]
    mat=Matrix{Complex{T}}(undef, length(ts), N)
    @showprogress desc="Calculating the evolution of coefficients..." for i in eachindex(ts)
        t=ts[i]
        mat[i,:].=coeffs_init.*exp.(-im.*Es.*t./ħ) 
    end
    return mat
end

"""
    plot_gaussian_from_eigenfunction_expansion(ax::Axis, coeffs::Vector{Complex{T}}, Psi2ds::Vector{Matrix{T}}, x_grid::Vector{T}, y_grid::Vector{T}) where {T<:Real}

Plots the reconstructed Gaussian wavepacket from its expansion coefficients in the eigenbasis.

This function uses the expansion coefficients to reconstruct the Gaussian wavepacket as a sum 
of the scaled eigenfunctions and plots it into an `Axis`.

# Arguments:
- `ax::Axis`: Axis object from `CairoMakie` where the heatmap will be plotted.
- `coeffs::Vector{Complex{T}}`: Vector of expansion coefficients for each eigenfunction.
- `Psi2ds::Vector{Matrix{T}}`: Vector of wavefunction matrices, one for each eigenfunction.
- `x_grid::Vector{T}`: x-coordinates of the grid on which the wavefunctions are defined.
- `y_grid::Vector{T}`: y-coordinates of the grid on which the wavefunctions are defined.

# Returns:
- `MakieCore.Heatmap`: The heatmap object.
"""
function plot_gaussian_from_eigenfunction_expansion(ax::Axis,coeffs::Vector{Complex{T}},Psi2ds::Vector,x_grid::Vector{T},y_grid::Vector{T}) where {T<:Real}
    reconstructed_gaussian=abs.(sum(coeffs[i]*Psi2ds[i] for i in eachindex(coeffs)))
    hmap=heatmap!(ax,x_grid,y_grid,reconstructed_gaussian,colormap=:balance,colorrange=(-maximum(reconstructed_gaussian),maximum(reconstructed_gaussian)))
    return hmap
end

"""
    save_evolution_basis_params(filename::String, Psi2ds::Vector{Matrix{T}}, overlaps::Vector{T},
                                x_grid::Vector{T}, y_grid::Vector{T}, pts_mask::BitVector,
                                dx::T, dy::T) where {T<:Real}

Saves the evolution basis parameters required for reconstructing a wavepacket evolution.

# Arguments
- `filename::String`: The file name (e.g., `"wavepacket_data.jld2"`) where the data will be stored.
- `Psi2ds::Vector{Matrix{T}}`: Vector of eigenfunction matrices for different `ks`.
- `overlaps::Vector{T}`: Overlap coefficients between the wavepacket and eigenfunctions.
- `x_grid::Vector{T}`: The x-coordinates of the spatial grid.
- `y_grid::Vector{T}`: The y-coordinates of the spatial grid.
- `pts_mask::Vector{Bool}`: Boolean mask indicating which grid points are inside the billiard.
- `dx::T`: The grid spacing in the x-direction.
- `dy::T`: The grid spacing in the y-direction.
- `ks::Vector{T}`: Vector of wavenumbers.

# Returns
- `Nothing`
"""
function save_evolution_basis_params!(filename::String,Psi2ds::Vector{Matrix{T}},overlaps::Vector{Complex{T}},x_grid::Vector{T},y_grid::Vector{T},pts_mask::Vector{Bool},dx::T,dy::T,ks::Vector{T}) where {T<:Real}
    @save filename Psi2ds overlaps x_grid y_grid pts_mask dx dy ks
end

"""
    read_evolution_basis_params(filename::String)

Reads and loads the evolution basis parameters from a saved JLD2 file.

# Arguments
- `filename::String`: The file name from which to load the data.

# Returns
A tuple containing:
- `Psi2ds::Vector{Matrix{T}}`: Eigenfunction matrices.
- `overlaps::Vector{Complex{T}}`: Overlap coefficients.
- `x_grid::Vector{T}`: The x-coordinates of the spatial grid.
- `y_grid::Vector{T}`: The y-coordinates of the spatial grid.
- `pts_mask::Vector{Bool}`: Boolean mask indicating grid points inside the billiard.
- `dx::T`: Grid spacing in the x-direction.
- `dy::T`: Grid spacing in the y-direction.
- `ks::Vector{T}`: Vector of wavenumbers.

"""
function read_evolution_basis_params(filename::String)
    @load filename Psi2ds overlaps x_grid y_grid pts_mask dx dy ks
    return Psi2ds,overlaps,x_grid,y_grid,pts_mask,dx,dy,ks
end

"""
    animate_wavepacket_evolution!(filename::String,coeffs_matrix::Matrix{Complex{T}},Psi2ds::Vector{Matrix{T}},x_grid::Vector{T},y_grid::Vector{T},ts::Vector{T};framerate::Int=30) where {T<:Real}

Generates and saves an animation of the wavepacket evolution using precomputed coefficients.

# Arguments
- `filename::String`: Output file name (`.mp4` or `.gif`).
- `coeffs_matrix::Matrix{Complex{T}}`: Time-evolved coefficient matrix, shape `(length(ts), length(Psi2ds))`.
- `Psi2ds::Vector{Matrix{T}}`: Eigenfunction wavefunction matrices.
- `x_grid::Vector{T},y_grid::Vector{T}`: Spatial grid coordinates.
- `ts::Vector{T}`: Time values.
- `framerate::Int=30`: Animation frame rate.

# Returns
- Saves the animation as `filename`.
"""
function animate_wavepacket_evolution!(filename::String,coeffs_matrix::Matrix{Complex{T}},Psi2ds::Vector{Matrix{T}},x_grid::Vector{T},y_grid::Vector{T},ts::Vector{T};framerate::Int=30,momentum_x_lims::Union{Symbol,Tuple{Float64,Float64}}=:default,momentum_y_lims::Union{Symbol,Tuple{Float64,Float64}}=:default) where {T<:Real}
    psi_idxs=eachindex(Psi2ds)
    fig=Figure(size=(3200,1200),resolution=(3200,1200))
    ax_real=Axis(fig[1,1],title="Wavepacket Evolution",xlabel="x",ylabel="y",width=1800,height=1000)
    Psi=sum(coeffs_matrix[1,j]*Psi2ds[j] for j in psi_idxs)
    hm_real=heatmap!(ax_real,x_grid,y_grid,abs.(Psi),colormap=:balance)
    ax_momentum=Axis(fig[1,2],title="Momentum Distribution",xlabel="kx",ylabel="ky")
    dx,dy=x_grid[2]-x_grid[1],y_grid[2]-y_grid[1]
    kx_grid=fftshift(fftfreq(length(x_grid),1/dx)).*(2*pi)
    ky_grid=fftshift(fftfreq(length(y_grid),1/dy)).*(2*pi)
    compute_momentum_distribution(Ψ)=abs2.(fftshift(fft(Ψ)).*(dx*dy/(4*pi)))
    frames_real=Vector{Matrix{T}}(undef,length(ts))
    frames_momentum=Vector{Matrix{T}}(undef,length(ts))
    frames_real[1]=abs.(Psi)
    frames_momentum[1]=compute_momentum_distribution(Psi)
    @showprogress desc="Precomputing frames..." Threads.@threads for i in 2:length(ts)
        Psi=sum(coeffs_matrix[i,j]*Psi2ds[j] for j in psi_idxs)
        frames_real[i]=abs.(Psi)
        frames_momentum[i]=compute_momentum_distribution(Psi)
    end
    if momentum_x_lims==:default && momentum_y_lims==:default
        momentum_x_lims=(-maximum(kx_grid),maximum(kx_grid))
        momentum_y_lims=(-maximum(ky_grid),maximum(ky_grid))
    elseif momentum_x_lims==:default
        momentum_x_lims=(-maximum(kx_grid),maximum(kx_grid))
    elseif momentum_y_lims==:default
        momentum_y_lims=(-maximum(ky_grid),maximum(ky_grid))
    end
    hm_momentum=heatmap!(ax_momentum,kx_grid,ky_grid,frames_momentum[1],colormap=:hot,width=1800,height=1000)
    xlims!(ax_momentum,momentum_x_lims)
    ylims!(ax_momentum,momentum_y_lims)
    function update_frame(i)
        hm_real[3]=frames_real[i]
        hm_momentum[3]=frames_momentum[i]
        ax_real.title="t=$(round(ts[i],digits=3))"
    end
    record(fig,filename,2:length(ts);framerate=framerate) do i
        update_frame(i)
    end
    println("Animation saved as $filename")
end

############### WAVEPACKET EVOLUTION VIA CRANK-NICHOLSON ###############

"""
### More details: https://web.physics.utah.edu/~detar/phycs6730/handouts/crank_nicholson/crank_nicholson/

We discretize the time evolution using the Crank-Nicholson method, which is an 
implicit midpoint method that is unconditionally stable and unitary.

Using a finite difference approach, we approximate the time derivative as:

    iℏ (Ψⁿ⁺¹ - Ψⁿ) / Δt ≈ (1/2) H (Ψⁿ⁺¹ + Ψⁿ)

Rearranging the equation gives:

    ( I + iΔt H / 2ℏ ) Ψⁿ⁺¹ = ( I - iΔt H / 2ℏ ) Ψⁿ

which can be rewritten as:

    A Ψⁿ⁺¹ = B Ψⁿ

where:
- A = I + i(Δt / 2ℏ) H  → Left-hand side matrix.
- B = I - i(Δt / 2ℏ) H  → Right-hand side matrix.
- Ψⁿ⁺¹ is the wavefunction at the next time step.
- Ψⁿ is the wavefunction at the current time step.

At each time step, we solve the linear system:

    Ψⁿ⁺¹ = A⁻¹ B Ψⁿ

Since A does not change over time (also B does not change with time), we can factorize it once using LU decomposition for efficient iterative solving. 
Using LU decomposition ensures better numerical stability and prevents inaccuracies due to floating-point precision errors.

------------------------------------------------------------------------------------
3. DISCRETIZATION OF THE LAPLACIAN OPERATOR (FINITE DIFFERENCE METHOD)
------------------------------------------------------------------------------------

The Laplacian operator ∇²Ψ in 2D is discretized using finite differences:

    ∇²Ψ(x, y) ≈ (Ψᵢ₊₁ⱼ + Ψᵢ₋₁ⱼ - 2Ψᵢⱼ) / Δx²  +  (Ψᵢⱼ₊₁ + Ψᵢⱼ₋₁ - 2Ψᵢⱼ) / Δy²

This is implemented as a sparse matrix using Kronecker products:

    ∇² ≈ kron(I, Dxx) + kron(Dyy, I)

where:
- Dxx is the finite difference matrix** for the second derivative in x.
- Dyy is the finite difference matrix** for the second derivative in y.
- I is the identity matrix.

The Hamiltonian matrix is then:

    H = - (ℏ² / 2m) * (∇²) 

where we remove the indexes that defined the outside boundary. This eliminates the necessity for the use of V0 in the calculation.

------------------------------------------------------------------------------------
4. IMPLEMENTATION DETAILS
------------------------------------------------------------------------------------

### 4.1 GRID SETUP
- The computational grid is defined on a uniform 2D mesh with Nx × Ny points.
- The spatial step sizes are:
  
      dx = Lx / (Nx - 1),  dy = Ly / (Ny - 1)

### 4.2 INITIAL CONDITION: GAUSSIAN WAVE PACKET
- The initial wavefunction is chosen as a Gaussian wave packet with momentum (kx₀, ky₀):

      Ψ₀(x, y) = exp( -((x - x₀)² + (y - y₀)²) / 2σ² ) * exp(i(kx₀ x + ky₀ y))

- The wavefunction is normalized at every time step (just for safety, norm should stay unchanged during evolution):

      Ψ /= sqrt(sum(abs2.(Ψ) * dx * dy))

### 4.3 TIME EVOLUTION
- The wavefunction **evolves in time** by solving:

      Ψⁿ⁺¹ = A⁻¹ B Ψⁿ

- `A` is factorized once using:

      A_factor = lu(A)

- At each timestep, we solve:

      Ψ_new = solve(A_factor div. (B * Ψ_old))

- The probability density |Ψ(x, y)|² is stored in snapshots for visualization.

------------------------------------------------------------------------------------
5. ADVANTAGES OF THE CRANK-NICHOLSON METHOD
------------------------------------------------------------------------------------

- Unconditionally stable → Does not impose a restriction on time step Δt.  
- Norm-preserving → Ensures that ∫|Ψ|²dxdy remains constant.  
- Time-reversible → Accurately models quantum wave packet evolution.  
- Implicit and second-order accurate → More accurate than explicit schemes.  
- Sparse → Solved using LU decomposition**.

------------------------------------------------------------------------------------
"""

struct Crank_Nicholson{T<:Real,Bi<:AbsBilliard} 
    billiard::Bi
    fem::FiniteElementMethod{T}
    pts_mask::Vector{Bool} # this is the matrix that the wavefunction that is being flattened to a vector for convenience
    ℏ::T # Reduced Planck's constant
    m::T # Particle mass
    xlim::Tuple{T,T}
    ylim::Tuple{T,T}
    Nx::Integer # Grid points in x and y
    Ny::Integer  
    Lx::T # Domain lengths in x and y
    Ly::T  
    dx::T # Grid difference in x and y
    dy::T
    dt::T # Time step
    x_grid::Vector{T}
    y_grid::Vector{T}
    Nt::Integer # Number of time steps
end

"""
    Crank_Nicholson(billiard::Bi,Nt::Integer;fundamental::Bool=true,k_max=100.0,ℏ=1.0,m=1.0,Nx::Integer=1000,Ny::Integer=1000,dt=0.005) where {Bi<:AbsBilliard}

Create a new `Crank_Nicholson` object with the given parameters. This is a constructor function that under the hood call `FiniteElementMethod` to help construct the `Hamiltonian` matrix and the interior mask for the billiard.

# Arguments
- `billiard`: The billiard to simulate.
- `Nt`: The number of time steps.
- `fundamental::Bool=true`: Whether to use the fundamental boundary conditions.
- `k_max`: The maximum wavenumber for the Gaussian wave packet -> used for precise determination of the x_lim and y_lim.
- `ℏ`: Reduced Planck's constant.
- `m`: Particle mass.
- `Nx::Integer=1000`: Grid points in x.
- `Ny::Integer=1000`: Grid points in y.
- `dt::T=0.005`: Time step.

# Returns
- A new `Crank_Nicholson` object.
"""
function Crank_Nicholson(billiard::Bi,Nt::Integer;fundamental::Bool=true,k_max=100.0,ℏ=1.0,m=1.0,Nx::Integer=1000,Ny::Integer=1000,dt=0.005) where {Bi<:AbsBilliard}
    if fundamental
        boundary=billiard.fundamental_boundary
        L=billiard.length
        typ=eltype(L)
        xlim::Tuple{typ,typ},ylim::Tuple{typ,typ}=boundary_limits(boundary;grd=max(1000,round(Int,k_max*L*5.0/(2*pi)))) # set b=5.0
        Lx,Ly=xlim[2]-xlim[1],ylim[2]-ylim[1] # domain lengths in x and y
        dx,dy=Lx/(Nx-1),Ly/(Ny-1)
        nx,ny=Nx,Ny
        x_grid,y_grid=collect(typ,range(xlim..., nx)),collect(typ,range(ylim..., ny))
        pts=collect(SVector(x,y) for y in y_grid for x in x_grid)
        sz=length(pts)
        pts_mask=points_in_billiard_polygon(pts,billiard,round(Int,sqrt(sz));fundamental_domain=true)
        fem=FiniteElementMethod(billiard,Nx,Ny,ℏ=ℏ,m=m,fundamental=fundamental,offset_x_symmetric=0.0,offset_y_symmetric=0.0)
        return Crank_Nicholson(billiard,fem,pts_mask,ℏ,m,xlim,ylim,Nx,Ny,Lx,Ly,dx,dy,dt,x_grid,y_grid,Nt)
    else
        boundary=billiard.full_boundary
        L=billiard.length
        typ=eltype(L)
        xlim,ylim=boundary_limits(boundary;grd=max(1000,round(Int,k_max*L*5.0/(2*pi))))  # set b=5.0
        Lx,Ly=xlim[2]-xlim[1],ylim[2]-ylim[1]
        dx,dy=Lx/(Nx-1),Ly/(Ny-1)
        nx,ny=Nx,Ny
        x_grid,y_grid=collect(typ,range(xlim..., nx)),collect(typ,range(ylim..., ny))
        pts=collect(SVector(x,y) for y in y_grid for x in x_grid)
        sz=length(pts)
        pts_mask=points_in_billiard_polygon(pts,billiard,round(Int,sqrt(sz));fundamental_domain=false)
        fem=FiniteElementMethod(billiard,Nx,Ny,ℏ=ℏ,m=m,fundamental=fundamental,offset_x_symmetric=0.0,offset_y_symmetric=0.0)
        return Crank_Nicholson(billiard,fem,pts_mask,ℏ,m,xlim,ylim,Nx,Ny,Lx,Ly,dx,dy,dt,x_grid,y_grid,Nt)
    end
end

"""
    Hamiltonian(cn::Crank_Nicholson{T};V0=1e12)::SparseMatrixCSC where {T<:Real}

Constructs the Sparse matrix Hamiltonian from the 2D Laplacian contructed from the sum of Kroenecker products of the 1D Laplacians and with removed indexes for the outside of the boundary. Basically calls FEM_Hamiltonian and returns it with 0 potential.

# Arguments
- `cn::Crank_Nicholson`: A `Crank_Nicholson` object.

# Returns
- `SparseMatrixCSC`: A sparse matrix representing the Hamiltonian only in the interior grid.
"""
function Hamiltonian(cn::Crank_Nicholson{T})::SparseMatrixCSC where {T<:Real}
    return FEM_Hamiltonian(cn.fem)
end

"""
    flatten_fem_wavepacket(fem::FiniteElementMethod, ψ_full::Matrix{Complex{T}}) -> Vector{Complex{T}} where {T<:Real}

Extracts and flattens the wavepacket from the full grid to only interior points for Crank-Nicholson only for the interior points. This reduces computation cost and removed the requirement for the V0 to be implemented in Crank-Nicholson that would case ill-conditioned H evolution. This also ensured that the flattened wavepacket will have the correct dimension to allow multiplication with the Hamitlonian matrix with only interior points.

# Arguments:
- `cn::Crank_Nicholson`: The `Crank_Nicholson` instance which contains the relevant `FiniteElementMethod` struct which holds `interior_idx`.
- `ψ0_full::Matrix{Complex{T}}`: The full Nx × Ny grid wavepacket to be reduced to only interior points.

# Returns:
- `ψ0::Vector{Complex{T}}`: Flattened wave packet for FEM (only on interior points).
"""
function flatten_fem_wavepacket(cn::Crank_Nicholson{T},ψ0_full::Matrix{Complex{T}}) where {T<:Real}
    interior_idx=cn.fem.interior_idx # Interior index mapping (Nx × Ny) → (Q)
    ψ0=ψ0_full[interior_idx.>0]  # Keeps only interior indices (inner ones have interior_idx[i]>0 for interior i)
    ψ0/=norm(ψ0)  # Normalize the wavepacket
    return ψ0
end

"""
    reconstruct_fem_wavepacket(ψ_interior::Vector{Complex{T}}, cn::Crank_Nicholson{T}) -> Matrix{Complex{T}} where {T<:Real}

Reconstructs the full Nx × Ny grid wavefunction from the interior-indexed wavefunction evolved by FEM Hamiltonian. This is useful if one wants to use the full domain matrices for Postprocessing.

# Arguments
- `ψ_interior::Vector{Complex{T}}`: The flattened wavefunction vector that was evolved in FEM.
- `cn::Crank_Nicholson{T}`: The `Crank_Nicholson` instance, which contains `FiniteElementMethod`.

# Returns
- `ψ_full::Matrix{Complex{T}}`: The reconstructed wavefunction on the full grid (Nx × Ny), with zeros at exterior points and normalized.
"""
function reconstruct_fem_wavepacket(ψ_interior::Vector{Complex{T}}, cn::Crank_Nicholson{T}) where {T<:Real}
    Nx,Ny=cn.Nx,cn.Ny
    interior_idx=cn.fem.interior_idx  # The interior index mapping
    ψ_full=zeros(Complex{T},Nx,Ny)  # Initialize full grid with zeros (better for numerical operations)
    # Populate interior points using `interior_idx`
    for i in 1:length(interior_idx)
        if interior_idx[i]>0  # Only interior points have indices > 0
            ψ_full[i]=ψ_interior[interior_idx[i]]  # Map back to full grid
        end
    end
    return ψ_full # ./nomr(ψ_full) is not neccesery since the ψ_interior is already normalized on the interior and we are already only recreating interior points so the sum is ≈1.0
end

"""
    compute_shannon_entropy(ψ::Vector{Complex{T}}) where {T<:Real}

Computes the Shannon entropy of the wavefunction ψ. The wavefunction is given as a flattened Sparse matrix (Vector{Complex}).

# Arguments
- `ψ::Vector{Complex{T}}`: The flattened wavefunction vector.

# Returns
- `Vector{T}`: The Shannon entropy of the wavefunction.
"""
function compute_shannon_entropy(ψ::Vector{Complex{T}}) where {T<:Real}
    P=abs2.(ψ)
    P=P[.!isnan.(P)] # to remove the NaN's from influencing
    P.=max.(P,1e-12) # to remove anyting close to 0
    entropy=-sum(P.*log.(P)) # (ignoring "zero" values to avoid log(0))
    return entropy/log(length(ψ)) # normalize wrt grid since log(entropy) / log(log(length(ψ))) should give grid independant result
end

"""
evolve_clark_nicholson(cn::Crank_Nicholson{T},H::SparseMatrixCSC,ψ0::Matrix{Complex{T}};save_after_iterations::Integer=5)::Tuple{Vector{Matrix},Vector{T}} where {T<:Real}

Evolves the wavefunction ψ0 under the time-dependent Schrodinger equation using the Crank-Nicholson method, with the provided Hamiltonian matrix H and initial state ψ0.

# Arguments
- `cn::Crank_Nicholson`: A `Crank_Nicholson` object.
- `H::SparseMatrixCSC`: The Hamiltonian matrix representing the Schrodinger equation constructed from either `Hamiltonian` or `FEM_Hamiltonian`. Both give the same object, a sparse Hamitlonian defined on only the interior indexes.
- `ψ0::Matrix{Complex{T}}`: The initial state vector for the wavefunction in the full Nx × Ny form. This will internally be reduced to an interior vector.
- `save_after_iterations::Integer=5`: The number of iterations after which to save the snapshot.
- `threshold=1e-6`: The threshold for checking the norms of the wavefunctions inside the billiard. The relative norms are computed wrt the starting norm of the packet and then in the end checked.

# Returns
- `Tuple{Vector{Matrix},Vector{Matrix{Complex{T}}},Vector{T}}`: A tuple containing the snapshots of the wavefunction at regular intervals, their raw non abs2 versions and the corresponding Shannon entropies at each snapshot.
"""
function evolve_clark_nicholson(cn::Crank_Nicholson{T},H::SparseMatrixCSC,ψ0::Matrix{Complex{T}};save_after_iterations::Integer=5,threshold=1e-6)::Tuple{Vector{Matrix},Vector{Matrix{Complex{T}}},Vector{T}} where {T<:Real}
    ψ=flatten_fem_wavepacket(cn,ψ0)
    dx,dy=cn.dx,cn.dy;mask=cn.pts_mask;Q=cn.fem.Q;
    I_mat=sparse(I,Q,Q)
    A=I_mat+im*cn.dt/(2*cn.ℏ)*H
    B=I_mat-im*cn.dt/(2*cn.ℏ)*H;
    Afactor=lu(A)
    b=similar(ψ)
    nsnap=floor(Int,cn.Nt/save_after_iterations);
    raw_snapshots=Vector{Vector{Complex{T}}}(undef,nsnap)
    snap_idx=1
    @showprogress desc="Evolving the wavepacket..." for t in 1:cn.Nt
        mul!(b,B,ψ)  # Compute b = B * ψ efficiently (from Ψ_{n+1} = A^-1 * B * Ψ_{n} == A * Ψ_{n+1} = B * Ψ_{n}). This step is O(nnz(B)) * O(nnz(H)).
        ψ=Afactor\b  # Solve linear system (main expensive step) (solves A * Ψ = b for Ψ). For large matrices the default solve is much faster than calling gmres.
        if t % save_after_iterations==0
            raw_snapshots[snap_idx]=copy(ψ)  # Store raw wavefunction vector
            snap_idx+=1
        end
    end
    # Postprocessing (Separate from expensive loop)
    snapshots=Vector{Matrix{T}}(undef,nsnap)
    matrices_raw=Vector{Matrix{Complex{T}}}(undef,nsnap)
    shannon_entropy_values=Vector{T}(undef,nsnap)
    inside_norms=Vector{T}(undef,nsnap)
    Threads.@threads for i in 1:nsnap
        ψ_full=reconstruct_fem_wavepacket(raw_snapshots[i],cn)  # Reconstruct full wavefunction
        snapshots[i]=abs2.(ψ_full)  # Compute probability density
        matrices_raw[i]=ψ_full  # Store complex wavefunction
        shannon_entropy_values[i]=compute_shannon_entropy(raw_snapshots[i]) # Compute Shannon on the vector raw version
        inside_norms[i]=sqrt(sum(snapshots[i][mask])*dx*dy)  # Compute norm inside billiard to check consistency
    end
    @showprogress desc="Removing exterior for plotting..." Threads.@threads for i in 1:nsnap # this is useful for plotting the animation since we can get rid of the exterior by setting it to white color.
        @inbounds snapshots[i][mask.==zero(T)].=NaN
    end
    base_norm=inside_norms[1];
    @showprogress desc="Checking norm conservation..." Threads.@threads for n in 1:length(inside_norms)
        if abs(inside_norms[n]-base_norm)/base_norm >threshold
            @warn "Norm conservation check failed at snapshot $n: relative difference = $(abs(inside_norms[n]-base_norm)/base_norm)"
        end
    end
    return snapshots,matrices_raw,shannon_entropy_values
end

"""
    animate_wavepacket_clark_nicholson!(cn::Crank_Nicholson{T},info::Vector{Matrix},filename::String;framerate::Integer=15) where {T<:Real}

Animates the wavefunction evolution using the provided snapshots and saves the animation as a .mp4 file (best to use .mp4).

# Arguments
- `cn::Crank_Nicholson`: A `Crank_Nicholson` object.
- `info::Vector{Matrix}`: A vector containing the snapshots of the wavefunction at regular intervals.
- `filename::String`: The name of the output .mp4 file.
- `framerate::Integer=15`: The frame rate for the .mp4 animation.

# Returns
- `nothing`: The animation is saved as a .mp4 file.
"""
function animate_wavepacket_clark_nicholson!(cn::Crank_Nicholson{T},info::Vector{Matrix},filename::String;framerate::Integer=15) where {T<:Real}
    snapshots=info
    frames=length(snapshots)
    fig=Figure(resolution=(2000,1000))
    ax=Axis(fig[1,1],title="|ψ(x,y)|²")
    x_grid,y_grid=cn.x_grid,cn.y_grid
    hmap=heatmap!(ax,x_grid,y_grid,snapshots[1],colormap=:balance,nan_color=:white)
    progress=Progress(frames;desc="Rendering animation...")
    record(fig,filename,1:frames,framerate=framerate) do i
        hmap[3]=snapshots[i]
        next!(progress)
    end
    println("Animation saved as $(filename)")
end

"""
    animate_wavepacket_clark_nicholson!(cn::Crank_Nicholson{T},info::Tuple{Vector{Matrix},Vector{T}},filename::String;framerate::Integer=15,save_after_iterations::Integer=5) where {T<:Real}

Animates the wavefunction evolution using the provided snapshots and Shannon entropy values and saves the animation as a .mp4 file (best to use .mp4).

# Arguments
- `cn::Crank_Nicholson`: A `Crank_Nicholson` object.
- `info::Tuple{Vector{Matrix},Vector{T}}`: A tuple containing the snapshots of the wavefunction at regular intervals and the corresponding Shannon entropies at each snapshot.
- `filename::String`: The name of the output .mp4 file.
- `framerate::Integer=15`: The frame rate for the .mp4 animation.
- `save_after_iterations::Integer=5`: The number of iterations after which to save the snapshot.

# Returns
- `nothing`: The animation is saved as a .mp4 file.
"""
function animate_wavepacket_clark_nicholson!(cn::Crank_Nicholson{T},info::Tuple{Vector{Matrix},Vector{T}},filename::String;framerate::Integer=15,save_after_iterations::Integer=5) where {T<:Real}
    snapshots,shannon_entropy_values=info
    frames=length(snapshots)
    x_grid,y_grid=cn.x_grid,cn.y_grid
    fig=Figure(resolution=(2000,1000))
    ax=Axis(fig[1,1],title="|ψ(x,y)|²",width=1000,height=900)
    hmap=heatmap!(ax,x_grid,y_grid,snapshots[1],colormap=:balance,nan_color=:white)
    ax2=Axis(fig[1,2],title="Shannon Entropy Evolution",xlabel="Time Step * $(save_after_iterations)",ylabel="Shannon Entropy",width=800,height=600)
    xlims!(ax2,0.0,ceil(Int,cn.Nt/save_after_iterations))  # Fixed x-axis range
    ylims!(ax2,minimum(shannon_entropy_values),maximum(shannon_entropy_values)*1.2)
    progress=Progress(frames;desc="Rendering animation...")
    record(fig,filename,1:frames,framerate=framerate) do i
        hmap[3]=snapshots[i]
        scatter!(ax2,collect(1:i),shannon_entropy_values[1:i],color=:blue,linewidth=3)
        next!(progress)
    end
    println("Animation saved as $(filename)")
end

##### ENERGY EXPECTATION VALUES AND THE AUTOCORRELATION FUNCTION #####

"""
    energy_levels(ks::AbstractVector{T};ħ::T=one(T),m::T=one(T)) where {T<:Real}

Calculates the energy levels with the correct ħ and m scaling.

### Arguments:
- `ks::AbstractVector{T}`: The wavevectors of the basis.
- `ħ::T=one(T)`: Reduced Planck constant.
- `m::T=one(T)`: Mass of the problem.
"""
function energy_levels(ks::AbstractVector{T};ħ::T=one(T),m::T=one(T)) where {T<:Real}
    return (ħ^2/(2*m)).*(ks.^2)
end



"""
    _expectation_E(coeffs::Vector{Complex{T}}, ks::AbstractVector{T}; ħ::T=one(T), m::T=one(T)) where {T<:Real}

Energy expectation value for a single coefficient vector:
⟨E⟩ = ∑ₙ |cₙ|² Eₙ with `Eₙ` from `energy_levels(ks; ħ, m)`.

### Arguments:
- `coeffs::Vector{Complex{T}}`: Modal coefficients (same ordering as `ks`).
- `ks::AbstractVector{T}`: Wavenumbers of the modes.
- `ħ::T=one(T)`: Reduced Planck constant.
- `m::T=one(T)`: Particle mass.

### Returns:
- `T`: The expectation value ⟨E⟩. If `coeffs` are normalized (∑|c|²=1) this is the physical mean energy.
"""
function _expectation_E(coeffs::Vector{Complex{T}},ks::Vector{T};ħ::T=one(T),m::T=one(T)) where {T<:Real}
    return sum(abs2.(coeffs).*energy_levels(ks,ħ=ħ,m=m))
end

"""
    expectation_E(C::Matrix{Complex{T}}, ks::AbstractVector{T}; ħ::T=one(T), m::T=one(T)) where {T<:Real}

Energy expectation values for a **time series** of coefficient rows.
Assumes `C` has shape (ntimes, nmodes), with `C[i,j]` the coefficient of mode `j` at time index `i`.

Computes for each time `i`:
⟨E⟩(tᵢ) = ∑ⱼ |C[i,j]|² Eⱼ.

### Arguments:
- `C::Matrix{Complex{T}}`: Coefficient matrix of size (ntimes, nmodes).
- `ks::AbstractVector{T}`: Wavenumbers of the modes (length `nmodes`).
- `ħ::T=one(T)`: Reduced Planck constant.
- `m::T=one(T)`: Particle mass.

### Returns:
- `Vector{T}`: Vector `E_of_t` of length `ntimes` with the energy expectation at each time.
"""
function expectation_E(C::Matrix{Complex{T}},ks::Vector{T};ħ::T=one(T),m::T=one(T)) where {T<:Real}
    nt,_=size(C)
    Es=energy_levels(ks; ħ=ħ, m=m)
    E=Vector{T}(undef, nt)
    @inbounds for i in 1:nt
        E[i]=sum(abs2.(view(C,i,:)).*Es)
    end
    return E
end

"""
    _autocorr_from_coeffs_with_Es(c0::Vector{Complex{T}}, Es::AbstractVector{T}, ts::AbstractVector{T}; ħ::T=one(T)) where {T<:Real}

Autocorrelation using only the **initial** coefficients and energies:
C(t) = ⟨ψ(0)|ψ(t)⟩ = ∑ₙ |cₙ(0)|² * exp(-i Eₙ t / ħ).

### Arguments:
- `c0::Vector{Complex{T}}`: Initial coefficients at t=0 (ordering must match `Es`).
- `Es::AbstractVector{T}`: Energy levels (e.g. from `energy_levels`).
- `ts::AbstractVector{T}`: Times at which to evaluate `C(t)`.
- `ħ::T=one(T)`: Reduced Planck constant.

### Returns:
- `Vector{Complex{T}}`: Values of the autocorrelation `C(t)` at each `t` in `ts`.
"""
function _autocorr_from_coeffs_with_Es(c0::Vector{Complex{T}},Es::Vector{T},ts::Vector{T};ħ=one(T)) where {T<:Real}
    w0=abs2.(c0)
    C=Vector{Complex{T}}(undef,length(ts))
    @inbounds for i in eachindex(ts)
        t=ts[i]
        C[i]=sum(@. w0*exp(-im*Es*(t/ħ)))
    end
    return C
end

"""
    autocorr_from_coeffs(c0::Vector{Complex{T}}, ks::AbstractVector{T}, ts::AbstractVector{T}; ħ::T=one(T), m::T=one(T)) where {T<:Real}

Convenience wrapper to compute the autocorrelation
C(t) = ∑ₙ |cₙ(0)|² * exp(-i Eₙ t / ħ)
from initial coefficients `c0` and wavenumbers `ks`.

### Arguments:
- `c0::Vector{Complex{T}}`: Initial coefficients at t=0 (ordering matches `ks`).
- `ks::AbstractVector{T}`: Wavenumbers of the modes.
- `ts::AbstractVector{T}`: Times at which to evaluate `C(t)`.
- `ħ::T=one(T)`: Reduced Planck constant.
- `m::T=one(T)`: Particle mass.

### Returns:
- `Vector{Complex{T}}`: Values of the autocorrelation `C(t)` at each `t` in `ts`.
"""
function autocorr_from_coeffs(c0::Vector{Complex{T}},ks::Vector{T},ts::Vector{T};ħ=one(T),m=one(T)) where {T<:Real}
    Es=energy_levels(ks;ħ=ħ,m=m)
    return _autocorr_from_coeffs_with_Es(c0,Es,ts;ħ=ħ)
end

##### UNCERTANTIES ######

"""
    uncertainty_x(cn::Crank_Nicholson{T}, ψ::Matrix{Complex{T}}) where {T<:Real}

Computes the uncertainty (standard deviation) in position along the x-direction
for a given 2D wavefunction `ψ`.

### Arguments:
- `cn::Crank_Nicholson{T}`: A structure containing `Nx`, `Ny`, `dx`, `dy`, and `ℏ`.
- `ψ::Matrix{Complex{T}}`: A 2D matrix representing the spatial wavefunction.

### Returns:
- `T`: The standard deviation (uncertainty) in position along the x-direction.

### Dimension Handling:
- The wavefunction matrix `ψ[i, j] → ψ(yᵢ, xⱼ)` follows standard matrix indexing:
  - **Rows correspond to y-coordinates**.
  - **Columns correspond to x-coordinates**.
- To compute `P(x)`, we must **integrate out y**.
  - This requires summing over `dims=2` (columns).
  - The result is a **1D probability distribution `P(x)`**.
"""
function uncertainty_x(cn::Crank_Nicholson{T},ψ::Matrix{Complex{T}}) where {T<:Real}
    x=cn.x_grid  # x values corresponding to matrix columns
    dx,dy=cn.dx,cn.dy
    P=abs2.(ψ)  # Probability density |ψ(x, y)|²
    # Compute ⟨x⟩ and ⟨x²⟩
    bra_x_ket=sum(x'.* sum(P,dims=1))*dx*dy 
    bra_x_sq_ket=sum((x'.^2).*sum(P,dims=1))*dx*dy 
    return sqrt(abs(bra_x_sq_ket-bra_x_ket^2))
end

"""
    uncertainty_y(cn::Crank_Nicholson{T}, ψ::Matrix{Complex{T}}) where {T<:Real}

Computes the uncertainty (standard deviation) in position along the y-direction
for a given 2D wavefunction `ψ`.

### Arguments:
- `cn::Crank_Nicholson{T}`: A structure containing `Nx`, `Ny`, `dx`, `dy`, and `ℏ`.
- `ψ::Matrix{Complex{T}}`: A 2D matrix representing the spatial wavefunction.

### Returns:
- `T`: The standard deviation (uncertainty) in position along the y-direction.

### Dimension Handling:
- The wavefunction matrix `ψ[i, j] → ψ(yᵢ, xⱼ)` follows standard matrix indexing:
  - **Rows correspond to y-coordinates**.
  - **Columns correspond to x-coordinates**.
- To compute `P(y)`, we must **integrate out x**.
  - This requires summing over `dims=1` (rows).
  - The result is a **1D probability distribution `P(y)`**.
"""
function uncertainty_y(cn::Crank_Nicholson{T},ψ::Matrix{Complex{T}}) where {T<:Real}
    y=cn.y_grid  # y values corresponding to matrix rows
    dx,dy=cn.dx,cn.dy
    P=abs2.(ψ)  # Probability density |ψ(x, y)|²
    # Compute ⟨y⟩ and ⟨y²⟩
    bra_y_ket=sum(y.*sum(P,dims=2))*dx*dy  # Sum over x to get P(y), then multiply by y. 
    bra_y_sq_ket=sum((y.^2).*sum(P,dims=2))*dx*dy  # Sum over x to get P(y), then multiply by y²
    return sqrt(abs(bra_y_sq_ket-bra_y_ket^2))
end

function uncertainty_x(cn::Crank_Nicholson{T},ψ_list::Vector{Matrix{Complex{T}}}) where {T<:Real}
    results=Vector{T}(undef,length(ψ_list))
    Threads.@threads for i in eachindex(ψ_list)
        results[i]=uncertainty_x(cn,ψ_list[i])
    end
    return results
end

function uncertainty_y(cn::Crank_Nicholson{T},ψ_list::Vector{Matrix{Complex{T}}}) where {T<:Real}
    results=Vector{T}(undef,length(ψ_list))
    Threads.@threads for i in eachindex(ψ_list)
        results[i]=uncertainty_y(cn,ψ_list[i])
    end
    return results
end

"""
    create_momentum_grid(N, d, ℏ)

Generates a momentum grid for a given number of points `N` and spatial step `d`.
The momentum values are calculated using FFT frequencies and converted to
momentum space via: p = 2πℏ * frequency.

Arguments:
- `N::Ti`: The number of points in the momentum grid.
- `d::T`: The spatial step size (i.e dx & dy). This is internally used as 1/d since this represents the sampling rate which is defined as the reciprocal of sample spacing.
- `ℏ::T`: The reduced Planck constant.

Returns:
- `Vector{T}`: A vector containing the momentum values in momentum space.
"""
function create_momentum_grid(N::Ti,d::T,ℏ::T) where {T<:Real,Ti<:Integer}
    freqs=fftshift(fftfreq(N,fs=1/d)) # fftfreq gives frequencies in cycles per unit length
    return freqs*2π*ℏ
end

"""
    uncertainty_p(cn::Crank_Nicholson{T},ψ::Matrix{Complex{T}}) where {T<:Real}

Calculates the uncertainty in momentum in the xy-direction for a given 2D wavefunction ψ.

Arguments:
- `cn::Crank_Nicholson{T}`: A structure with parameters `Nx`, `Ny`, `dx`, `dy`, and ℏ.
- `ψ::Matrix{Complex{T}}`: A 2D matrix representing the spatial wavefunction.

Returns:
- `Tuple{T,T}`: The standard deviation (uncertainty) in momentum in the x and y direction.
"""
function uncertainty_p(cn::Crank_Nicholson{T}, ψ::Matrix{Complex{T}}) where {T<:Real}
    Nx,Ny=cn.Nx,cn.Ny
    dx,dy=cn.dx,cn.dy
    ℏ=cn.ℏ
    # Compute FFT to get momentum-space wavefunction
    ψ_kx_ky=fftshift(fft(ψ))*(dx*dy)  # 2d DFT to get momentum-space wavefunction, but needs to be multiplied by integration steps since by def it does not take it into account
    # Compute probability distribution in momentum space
    P_k=abs2.(ψ_kx_ky) # element-wise square of abs value
    P_k_norm=P_k./sum(P_k)  # Normalize probability distribution
    # Define momentum grid using FFT frequencies
    kx_grid=create_momentum_grid(Nx,dx,ℏ) # the kx_grid analogous to x_grid
    ky_grid=create_momentum_grid(Ny,dy,ℏ) # the ky_grid analogous to y_grid
    # Marginal probability distributions for p_x and p_y
    P_px=sum(P_k_norm,dims=1)*(ky_grid[2]-ky_grid[1])  # Integrate over ky so we integrate out py (The original matrix is ψ[i, j] → ψ(yᵢ, xⱼ) since i indexes y (rows),j indexes x (columns)). This after fft becomes Φ[i, j] → Φ[pxᵢ, pyⱼ]
    P_py=sum(P_k_norm,dims=2)*(kx_grid[2]-kx_grid[1])  # Integrate over kx so we integrate out px
    # Expectation values
    bra_px_ket=sum(kx_grid.*P_px) # P_px is now a vector
    bra_py_ket=sum(ky_grid.*P_py)  # P_py is now a vector
    # Expectation values of p_x^2 and p_y^2
    bra_px_sq_ket=sum((kx_grid.^2).*P_px)
    bra_py_sq_ket=sum((ky_grid.^2).*P_py)
    # Compute standard deviations of momentum in x and y directions
    Δp_x=sqrt(abs(bra_px_sq_ket-bra_px_ket^2))
    Δp_y=sqrt(abs(bra_py_sq_ket-bra_py_ket^2))
    return Δp_x,Δp_y
end

"""
    uncertainty_p(cn::Crank_Nicholson{T},ψ_list::Vector{Matrix{Complex{T}}}) where {T<:Real}

Calculates the uncertainty in momentum in the xy-direction for a given vector of  2D wavefunction ψ.

Arguments:
- `cn::Crank_Nicholson{T}`: A structure with parameters `Nx`, `Ny`, `dx`, `dy`, and ℏ.
- `ψ_list::Vector{Matrix{Complex{T}}}`: A vector of 2D matrices representing the spatial wavefunctions.

Returns:
- `Vector{Tuple{T,T}}`: The standard deviations (uncertanties) in momentums in the x and y direction.
"""
function uncertainty_p(cn::Crank_Nicholson{T},ψ_list::Vector{Matrix{Complex{T}}}) where {T<:Real}
    results=Vector{Tuple{T,T}}(undef,length(ψ_list))
    Threads.@threads for i in eachindex(ψ_list)
        results[i]=uncertainty_p(cn,ψ_list[i])
    end
    return results
end
