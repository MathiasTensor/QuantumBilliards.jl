using Bessels, LinearAlgebra, ProgressMeter, FFTW, SparseArrays
include("../billiards/boundarypoints.jl")
include("../states/wavefunctions.jl")

############### WAVEPACKET EVOLUTION VIA BASIS SET ###############

"""
    Wavepacket{T<:Real}

Contains the initialization parameters that construct the Gaussian wavepacket.

# Arguments
- `x0::T`: Initial x position of the Gaussian wavepacket.
- `y0::T`: Initial y position of the Gaussian wavepacket.
- `sigma_x::T`: Standard deviation of the Gaussian wavepacket in the x direction.
- `sigma_y::T`: Standard deviation of the Gaussian wavepacket in the y direction.
- `kx0::T`: wavevector in the x direction.
- `ky0::T`: wavevector in the y direction.
"""
struct Wavepacket{T<:Real}
    x0::T
    y0::T
    sigma_x::T
    sigma_y::T
    kx0::T
    ky0::T
end

"""
    gaussian_wavepacket_2d(x::Vector{T}, y::Vector{T}, packet::Wavepacket{T}) where {T<:Real}

Generates a 2D Gaussian wavepacket in coordinate space on a grid of `(x,y)` values and then returns in on a grid as a `Matrix`

```math
f(x,y) = 1/(2*π*σx*σy)*exp(-(x-x0)^2/2σx-(y-y0)^2/2σy)*exp(-ikx*(x-x0))*exp(-iky(y-y0))
```

# Arguments
- `x::Vector{T}, y::Vector{T}`: The spatial coordinates x and y as vectors (separately) that form a grid.
- `packet::Wavepacket`: Gaussian wavepacket struct containing the params.

# Returns
- `Matrix{Complex{T}}`: The value at those params on a 2D grid.
"""
function gaussian_wavepacket_2d(x::Vector{T},y::Vector{T},packet::Wavepacket{T}) where {T<:Real}
    x0=packet.x0;y0=packet.y0;sigma_x=packet.sigma_x;sigma_y=packet.sigma_y;kx0=packet.kx0;ky0=packet.ky0
    dx=x.-x0
    dy=y.-y0
    gx_amp= @. exp(-dx^2/(2*sigma_x^2))
    gy_amp= @. exp(-dy^2/(2*sigma_y^2))
    gx_phase= @. exp(im*kx0*dx)
    gy_phase= @. exp(im*ky0*dy)
    gx=gx_amp.*gx_phase
    gy=gy_amp.*gy_phase
    norm_factor=1/sqrt(sqrt(2π*sigma_x*sigma_y))
    result=norm_factor.*(gx*transpose(gy)) # Outer product: result[i,j] = norm_factor * gx[i] * gy[j]
    return result
end

"""
    gaussian_wavepacket_2d(pts_in_billiard::Vector{SVector{2,T}},packet::Wavepacket{T}) where {T<:Real}

Computes the Gaussian wavepacket for a set of points inside the billiard.

# Arguments
- `pts_in_billiard::Vector{SVector{2,T}}`: Points inside the billiard.
- `packet::Wavepacket`: Gaussian wavepacket struct containing the params.

# Returns
- `Vector{Complex{T}}`: Gaussian wavepacket evaluated at the points.
"""
function gaussian_wavepacket_2d(pts_in_billiard::Vector{SVector{2,T}},packet::Wavepacket{T}) where {T<:Real}
    x0=packet.x0;y0=packet.y0;sigma_x=packet.sigma_x;sigma_y=packet.sigma_y;kx0=packet.kx0;ky0=packet.ky0
    xs=getindex.(pts_in_billiard,1)  # x-coordinates
    ys=getindex.(pts_in_billiard,2)  # y-coordinates
    dx=xs.-x0
    dy=ys.-y0
    amp= @. exp(-dx^2/(2*sigma_x^2)-dy^2/(2*sigma_y^2))
    phase= @. exp(im*(kx0*dx+ky0*dy))
    norm_factor = 1/sqrt(2π*sigma_x*sigma_y)
    return norm_factor.*(amp.*phase)
end

"""
    gaussian_coefficients(ks::Vector{T}, vec_us::Vector{Vector{T}}, vec_bdPoints::Vector{BoundaryPoints{T}}, 
                          billiard::Bi, packet::Wavepacket{T}; b::Float64=5.0) 
                          -> (Vector{Matrix{T}}, Vector{T}, Vector{T}, Vector{T}, BitVector, T, T)

Computes the eigenfunction matrices and their overlaps with a Gaussian wavepacket in a given billiard system. This gives us the coefficients and the basis eigenfunction for evolution and visualization.

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
    k_max=maximum(ks)
    type=eltype(k_max)
    L=billiard.length
    if fundamental_domain
        xlim,ylim=boundary_limits(billiard.fundamental_boundary;grd=max(1000,round(Int,k_max*L*b/(2*pi))))
    else
        xlim,ylim=boundary_limits(billiard.full_boundary;grd=max(1000,round(Int,k_max*L*b/(2*pi))))
    end
    dx,dy=xlim[2]-xlim[1],ylim[2]-ylim[1]
    nx,ny=max(round(Int,k_max*dx*b/(2*pi)),512),max(round(Int,k_max*dy*b/(2*pi)),512)
    x_grid,y_grid=collect(type,range(xlim..., nx)),collect(type,range(ylim..., ny))
    pts=collect(SVector(x,y) for y in y_grid for x in x_grid)
    sz=length(pts)
    pts_mask=points_in_billiard_polygon(pts,billiard,round(Int,sqrt(sz));fundamental_domain=fundamental_domain)
    pts_masked_indices=findall(pts_mask)
    Psi2ds=Vector{Matrix{type}}(undef,length(ks))
    overlaps=Vector{Complex{type}}(undef,length(ks))
    progress=Progress(length(ks),desc="Constructing wavefunction matrices and computing overlaps...")
    gaussian_mat::Matrix{Complex{type}}=gaussian_wavepacket_2d(x_grid,y_grid,packet) # create a 2d mat
    for i in eachindex(ks)
        @inbounds begin
            k,bdPoints,us=ks[i],vec_bdPoints[i],vec_us[i]
            Psi_flat=zeros(type,sz)
            thread_overlaps=Vector{Complex{type}}(undef,Threads.nthreads())
            for t in 1:Threads.nthreads() # thread safe local accumulators
                thread_overlaps[t]=zero(type)
            end
            Threads.@threads for j in eachindex(pts_masked_indices) # multithread this one since has most elements to thread over
                idx=pts_masked_indices[j]
                tid=Threads.threadid()  # thread ID for safe accumulation
                @inbounds begin
                    x,y=pts[idx]
                    psi_val=ϕ(x,y,k,bdPoints,us)
                    Psi_flat[idx]=psi_val
                    thread_overlaps[tid]+=psi_val*gaussian_mat[idx]
                end
            end
            overlap=sum(thread_overlaps) # from the thread safe local accumulation we 
            Psi2ds[i]=reshape(Psi_flat,ny,nx) # also return the normalized wavefunction matrices
            overlaps[i]=overlap
            next!(progress)
        end
    end
    return Psi2ds,overlaps,x_grid,y_grid,pts_mask,dx,dy
end

"""
    evolution_gaussian_coefficients(coeffs_init::Vector{Complex{T}}, ks::Vector{T}, ts::Vector{K}) where {T<:Real, K<:Real} 
    -> Matrix{Complex{T}}

Computes the time evolution of the expansion coefficients of a wavepacket in a quantum billiard.

# Description
Given an initial set of expansion coefficients `coeffs_init`, corresponding wavenumbers `ks`, 
and time values `ts`, this function computes the time-evolved coefficients based on the 
Schrödinger equation solution:

`c_n(t) = c_n(0) e^{-i E_n t}`

# Arguments
`coeffs_init::Vector{Complex{T}}`: The initial expansion coefficients of the wavepacket in the eigenbasis.
`ks::Vector{T}`: The wavenumbers associated with each eigenfunction.
`ts::Vector{K}`: A vector of time points at which to evaluate the evolved coefficients.

# Returns
`Matrix{Complex{T}}`: A matrix mat of shape (length(ts),length(ks)) where mat[i, j] represents the coefficient for the j-th eigenmode at time ts[i].
"""
function evolution_gaussian_coefficients(coeffs_init::Vector{Complex{T}},ks::Vector{T},ts::Vector{K}) where {T<:Real,K<:Real}
    N=length(coeffs_init)
    @assert N==length(ks) "The number of coefficients must match the number of eigenvalues, and they need to be correctly ordered: ks[i] -> Psi2d[i] -> coefficient[i]"
    Es=[k^2 for k in ks]
    mat=Matrix{Complex{T}}(undef, length(ts), N)
    @showprogress desc="Calculating the evolution of coefficients..." for i in eachindex(ts)
        t=ts[i]
        mat[i,:].=coeffs_init.*exp.(-im.*Es.*t) 
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

# Returns
- `Nothing`
"""
function save_evolution_basis_params!(filename::String,Psi2ds::Vector{Matrix{T}},overlaps::Vector{Complex{T}},x_grid::Vector{T},y_grid::Vector{T},pts_mask::Vector{Bool},dx::T,dy::T) where {T<:Real}
    @save filename Psi2ds overlaps x_grid y_grid pts_mask dx dy
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
"""
function read_evolution_basis_params(filename::String)
    @load filename Psi2ds overlaps x_grid y_grid pts_mask dx dy
    return Psi2ds,overlaps,x_grid,y_grid,pts_mask,dx,dy
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
#=
function animate_wavepacket_evolution!(filename::String,coeffs_matrix::Matrix{Complex{T}},Psi2ds::Vector{Matrix{T}},x_grid::Vector{T},y_grid::Vector{T},ts::Vector{T};framerate::Int=30) where {T<:Real}
    psi_idxs=eachindex(Psi2ds)
    fig=Figure(size=(1000,1000),resolution=(1000,1000))
    ax=Axis(fig[1,1],title="Wavepacket Evolution",xlabel="x",ylabel="y")
    Psi=real.(sum(coeffs_matrix[1,j]*Psi2ds[j] for j in psi_idxs))
    hm=heatmap!(ax,x_grid,y_grid,Psi,colormap=:balance)
    frames=Vector{Matrix{T}}(undef,length(ts)) # precompute fore all times matrices
    @showprogress desc="Precomputing matrices for animation..." Threads.@threads for i in eachindex(ts)[2:end]
        frames[i]=real.(sum(coeffs_matrix[i,j]*Psi2ds[j] for j in psi_idxs))
    end
    function update_frame(i)
        hm[3]=frames[i]
        ax.title="t=$(round(ts[i],digits=3))"
    end
    record(fig,filename,2:length(ts);framerate=framerate) do i
        update_frame(i)
    end
    println("Animation saved as $filename")
end
=#
function animate_wavepacket_evolution!(filename::String,coeffs_matrix::Matrix{Complex{T}},Psi2ds::Vector{Matrix{T}},x_grid::Vector{T},y_grid::Vector{T},ts::Vector{T};framerate::Int=30) where {T<:Real}
    psi_idxs=eachindex(Psi2ds)
    fig=Figure(size=(1600,800))
    ax_real=Axis(fig[1,1],title="Wavepacket Evolution",xlabel="x",ylabel="y")
    Psi=sum(coeffs_matrix[1,j]*Psi2ds[j] for j in psi_idxs)
    hm_real=heatmap!(ax_real,x_grid,y_grid,abs.(Psi),colormap=:balance)
    ax_momentum=Axis(fig[1,2],title="Momentum Distribution",xlabel="kx",ylabel="ky")
    kx_grid=fftshift(fftfreq(length(x_grid)))*(2π/(x_grid[end]-x_grid[1]))
    ky_grid=fftshift(fftfreq(length(y_grid)))*(2π/(y_grid[end]-y_grid[1]))
    compute_momentum_distribution(Ψ_x)=abs2.(fftshift(fft(Ψ_x)))
    frames_real=Vector{Matrix{T}}(undef,length(ts))
    frames_momentum=Vector{Matrix{T}}(undef,length(ts))
    frames_real[1]=abs.(Psi)
    frames_momentum[1]=compute_momentum_distribution(Psi)
    @showprogress desc="Precomputing frames..." Threads.@threads for i in 2:length(ts)
        Psi=sum(coeffs_matrix[i,j]*Psi2ds[j] for j in psi_idxs)
        frames_real[i]=abs.(Psi)
        frames_momentum[i]=compute_momentum_distribution(Psi)
    end
    hm_momentum=heatmap!(ax_momentum,kx_grid,ky_grid,frames_momentum[1],colormap=:hot)
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
# More details: https://web.physics.utah.edu/~detar/phycs6730/handouts/crank_nicholson/crank_nicholson/

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

Since A does not change over time, we can factorize it once using LU decomposition for efficient iterative solving. 
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

    H = - (ℏ² / 2m) * (∇²) + V_op

where V_op is the potential energy matrix.

------------------------------------------------------------------------------------
4. IMPLEMENTATION DETAILS
------------------------------------------------------------------------------------

### 4.1 GRID SETUP
- The computational grid is defined on a uniform 2D mesh** with Nx × Ny points.
- The spatial step sizes are:
  
      dx = Lx / (Nx - 1),  dy = Ly / (Ny - 1)

### 4.2 INITIAL CONDITION: GAUSSIAN WAVE PACKET
- The initial wavefunction is chosen as a Gaussian wave packet with momentum (kx₀, ky₀):

      Ψ₀(x, y) = exp( -((x - x₀)² + (y - y₀)²) / 2σ² ) * exp(i(kx₀ x + ky₀ y))

- The wavefunction is **normalized** at every time step:

      Ψ /= sqrt(sum(abs2.(Ψ) * dx * dy))

### 4.3 TIME EVOLUTION
- The wavefunction **evolves in time** by solving:

      Ψⁿ⁺¹ = A⁻¹ B Ψⁿ

- `A` is factorized once using:

      A_factor = lu(A)

- At each timestep, we solve:

      Ψ_new = A_factor "\" (B * Ψ_old)

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
    pts_mask::Vector{Bool}
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
    Nt::Integer # Number of time steps
end

function Crank_Nicholson(billiard::Bi,Nt::T;fundamental::Bool=true,k_max=100.0,ℏ=1.0,m=1.0,Nx::Integer=10000,Ny::Integer=10000,dt::T=0.005) where {T<:Real,Bi<:AbsBilliard}
    if fundamental
        boundary=billiard.fundamental_boundary
        L=billiard.lenght
        xlim::Tuple{T,T},ylim::Tuple{T,T}=boundary_limits(boundary;grd=max(1000,round(Int,k_max*L*5.0/(2*pi)))) # set b=5.0
        Lx,Ly=xlim[2]-xlim[1],ylim[2]-ylim[1] # domain lengths in x and y
        dx,dy=Lx/(Nx-1),Ly/(Ny-1)
        nx,ny=Nx,Ny
        x_grid,y_grid=collect(T,range(xlim..., nx)),collect(T,range(ylim..., ny))
        pts=collect(SVector(x,y) for y in y_grid for x in x_grid)
        sz=length(pts)
        pts_mask=points_in_billiard_polygon(pts,billiard,round(Int,sqrt(sz));fundamental_domain=true)
        return Crank_Nicholson(billiard,pts_mask,ℏ,m,xlim,ylim,Nx,Ny,Lx,Ly,dx,dy,dt,Nt)
    else
        boundary=billiard.full_boundary
        L=billiard.lenght
        xlim,ylim=boundary_limits(boundary;grd=max(1000,round(Int,k_max*L*5.0/(2*pi))))  # set b=5.0
        Lx,Ly=xlim[2]-xlim[1],ylim[2]-ylim[1]
        dx,dy=Lx/(Nx-1),Ly/(Ny-1)
        nx,ny=Nx,Ny
        x_grid,y_grid=collect(T,range(xlim..., nx)),collect(T,range(ylim..., ny))
        pts=collect(SVector(x,y) for y in y_grid for x in x_grid)
        sz=length(pts)
        pts_mask=points_in_billiard_polygon(pts,billiard,round(Int,sqrt(sz));fundamental_domain=false)
        return Crank_Nicholson(billiard,pts_mask,ℏ,m,xlim,ylim,Nx,Ny,Lx,Ly,dx,dy,dt,Nt)
    end
end

"""
The Laplacian operator ∇²Ψ in one dimension is given by:
    ∇²Ψ(x) = ∂²Ψ / ∂x²
Using **central finite differences**, this is approximated as:
    ∂²Ψ / ∂x² ≈ (Ψ_{i+1} - 2Ψ_{i} + Ψ_{i-1}) / Δx²
This can be written as a matrix equation:
    D_{xx} Ψ = (1/Δx²) * M Ψ
where `M` is the **finite difference matrix**, given by:

    ⎡ -2   1   0   0  ⋯   0 ⎤
    ⎢  1  -2   1   0  ⋯   0 ⎥
    ⎢  0   1  -2   1  ⋯   0 ⎥
M=  ⎢  0   0   1  -2  ⋯   0 ⎥   (Nx × Nx)
    ⎢  ⋮   ⋮   ⋮   ⋮   ⋱   1 ⎥
    ⎣  0   0   0   0   1  -2⎦
"""
function build_1D_Laplacian(N::Integer,d::T)::SparseMatrixCSC where {T<:Real}
    return spdiagm(0=>fill(-2.0,N), 1=>fill(1.0,N-1), -1 => fill(1.0,N-1))/d^2
end

function boundary_condition_infinite_potential(cn::Crank_Nicholson{T};V0=1e12)::SparseMatrixCSC where {T<:Real}
    V=V0.*(1 .-cn.pts_mask)
    V_default=spdiagm(0=>vec(V)) 
    return V_default
end

function Hamiltonian(cn::Crank_Nicholson{T};V0=1e12)::SparseMatrixCSC where {T<:Real}
    dx,dy=cn.dx,cn.dy
    Nx,Ny=cn.Nx,cn.Ny
    xlim,ylim=cn.xlim,cn.ylim
    Dxx::SparseMatrixCSC=build_1D_Laplacian(Nx,dx)
    Dyy::SparseMatrixCSC=build_1D_Laplacian(Ny,dy)
    I_x::SparseMatrixCSC=sparse(I,Nx,Nx)
    I_y::SparseMatrixCSC=sparse(I,Ny,Ny)
    Laplacian_2D::SparseMatrixCSC=kron(I_y,Dxx)+kron(Dyy,I_x)
    x=LinRange(xlim[1],xlim[2],Nx)
    y=LinRange(ylim[1],ylim[2],Ny)
    X=repeat(reshape(x,Nx,1),1,Ny)
    Y=repeat(reshape(y,1,Ny),Nx,1)
    V_op::SparseMatrixCSC=boundary_condition_infinite_potential(cn,V0=V0)
    return -ℏ^2/(2*m)*Laplacian_2D+V_op
end

function Hamiltonian(cn::Crank_Nicholson{T},V::SparseMatrixCSC;V0=1e12)::SparseMatrixCSC where {T<:Real}
    H=Hamiltonian(cn,V0=V0)
    @assert size(H)==size(V) "The unperturbed Hamiltonian matrix and the external potential must be of the same size."
    return H+V
end

function compute_shannon_entropy(ψ::Vector{Complex{T}},dx::T,dy::T)
    P=abs2.(ψ)
    P./=sum(P)*dx*dy
    entropy=-sum(P[P.>0].*log.(P[P.>0]))*dx*dy # (ignoring zero values to avoid log(0))
    return entropy
end

function evolve_clark_nicholson(cn::Crank_Nicholson{T},H::SparseMatrixCSC,ψ0::Matrix{Complex{T}};save_after_iterations::Integer=5)::Tuple{Vector{Matrix},Vector{T}} where {T<:Real}
    ψ0=ψ0/norm(ψ0)
    ψ=vec(ψ0)
    dx,dy=cn.dx,cn.dy
    dim=cn.Nx*cn.Ny
    I_mat=sparse(I,dim,dim)
    A=I_mat+im*cn.dt/(2*cn.ℏ)*H
    B=I_mat-im*cn.dt/(2*cn.ℏ)*H
    Afactor=lu(A) # only need this for A^-1 * B
    snapshots=Vector{Matrix{T}}()
    shannon_entropy_values=T[]
    @showprogress for t in 1:cn.Nt
        global b=B*ψ
        global ψ=Afactor\b
        ψ/=norm(ψ) # probably unnecessary
        if t % save_after_iterations==0 # save only some
            entropy_t=compute_shannon_entropy(ψ,dx,dy)
            push!(shannon_entropy_values,entropy_t)
            push!(snapshots,reshape(abs2.(ψ),Nx,Ny))
        end
    end
    Threads.@threads for i in 1:length(snapshots) # for later nicer heatmap plots
        snapshots[i][mask.==zero(T)].=NaN
    end
    return snapshots,shannon_entropy_values
end

function animate_wavepacket_clark_nicholson(info::Vector{Matrix},filename::String;framerate::Integer=15) where {T<:Real}
    snapshots=info
    frames=length(snapshots)
    fig=Figure(resolution=(2000,1000))
    ax=Axis(fig[1,1],title="|ψ(x,y)|²")
    heatmap!(ax,x,y,snapshots[1],colormap=:balance,nan_color=:white)
    record(fig,filename,1:frames,framerate=framerate) do i
        heatmap!(ax,snapshots[i],colormap=:balance,nan_color=:white)
    end
    println("Animation saved as $(filename)")
end

function animate_wavepacket_clark_nicholson(info::Tuple{Vector{Matrix},Vector{T}},filename::String;framerate::Integer=15) where {T<:Real}
    snapshots,shannon_entropy_values=info
    frames=length(snapshots)
    fig=Figure(resolution=(2000,1000))
    ax=Axis(fig[1,1],title="|ψ(x,y)|²",width=1000,height=900)
    heatmap!(ax,x,y,snapshots[1],colormap=:balance,nan_color=:white)
    ax2=Axis(fig[1,2],title="Shannon Entropy Evolution",xlabel="Time Step * $(N_snapshots_between)",ylabel="Shannon Entropy",width=800,height=600)
    xlims!(ax2,0.0,ceil(Int,Nt/N_snapshots_between))  # Fixed x-axis range
    record(fig,filename,1:frames,framerate=framerate) do i
        heatmap!(ax,snapshots[i],colormap=:balance,nan_color=:white)
        scatter!(ax2,collect(1:i),shannon_entropy_values[1:i],color=:blue,linewidth=3)
    end
    println("Animation saved as $(filename)")
end