using Bessels, LinearAlgebra, ProgressMeter

include("../billiards/boundarypoints.jl")
include("../states/wavefunctions.jl")

# EFFICIENT CONSTRUCTION AND PLOTTING
# TODO Fix sampling for small u(s)

### MATRIX ELEMENTS - PROJECTIONS OF 2D GAUSSIAN TO BASIS SET
"""
    gaussian_wavepacket_2d(x::T, y::T, x0::T, y0::T, sigma_x::T, sigma_y::T, kx0::T, ky0::T) :: Complex{T} where {T<:Real}

Generates a 2D Gaussian wavepacket in coordinate space.

```math
f(x,y) = 1/(2*π*σx*σy)*exp(-(x-x0)^2/2σx-(y-y0)^2/2σy)*exp(-ikx*(x-x0))*exp(-iky(y-y0))
```

# Arguments
- `x::T, y::T`: The spatial coordinates (x,y).
- `x0::T, y0::T`: The center of the wavepacket in space.
- `sigma_x::T, sigma_y::T`: standard deviations of the wavepacket in the x and y directions.
- `kx0::T, ky0::T`: The central wavevectors in the x and y directions.

# Returns
- `Complex{T}`: The value at those params.
"""
function gaussian_wavepacket_2d(x::T, y::T, x0::T, y0::T, sigma_x::T, sigma_y::T, kx0::T, ky0::T) where {T<:Real}
    norm_factor = 1/sqrt(sqrt(2π*sigma_x*sigma_y)) # Normalization factor for 2D Gaussian
    exp_factor = exp(-((x-x0)^2 / (2*sigma_x^2)+(y-y0)^2/(2*sigma_y^2)))
    phase_factor = exp(im*(kx0*(x-x0)+ky0*(y-y0)))
    gaussian = norm_factor * exp_factor * phase_factor
    return gaussian
end

"""
    gaussian_wavepacket_2d(x::Vector{T}, y::Vector{T}, x0::T, y0::T, sigma_x::T, sigma_y::T, kx0::T, ky0::T) where {T<:Real}

Generates a 2D Gaussian wavepacket in coordinate space on a grid of `(x,y)` values and then returns in on a grid as a `Matrix`

```math
f(x,y) = 1/(2*π*σx*σy)*exp(-(x-x0)^2/2σx-(y-y0)^2/2σy)*exp(-ikx*(x-x0))*exp(-iky(y-y0))
```

# Arguments
- `x::Vector{T}, y::Vector{T}`: The spatial coordinates x and y as vectors (separately) that form a grid.
- `x0::T, y0::T`: The center of the wavepacket in space.
- `sigma_x::T, sigma_y::T`: standard deviations of the wavepacket in the x and y directions.
- `kx0::T, ky0::T`: The central wavevectors in the x and y directions.

# Returns
- `Matrix{Complex{T}}`: The value at those params on a 2D grid.
"""
function gaussian_wavepacket_2d(x::Vector{T}, y::Vector{T}, x0::T, y0::T, sigma_x::T, sigma_y::T, kx0::T, ky0::T) where {T<:Real}
    norm_factor = 1/sqrt(sqrt(2π*sigma_x*sigma_y))
    inv_sigma_x2 = 1/(2*sigma_x^2)
    inv_sigma_y2 = 1/(2*sigma_y^2)
    result = Matrix{Complex{T}}(undef, length(x), length(y))
    Threads.@threads for i in eachindex(x)
        for j in eachindex(y)
            dx = x[i]-x0
            dy = y[j]-y0
            exp_factor = exp(-(dx^2 * inv_sigma_x2 + dy^2 * inv_sigma_y2))
            phase_factor = exp(im * (kx0 * dx + ky0 * dy))
            result[i, j] = norm_factor * exp_factor * phase_factor
        end
    end
    return result
end

"""
    gaussian_wavepacket_eigenbasis_expansion_coefficient(k::T, us::Vector{T}, bdPoints::BoundaryPoints{T}, x_grid::Vector{T}, y_grid::Vector{T}, x0::T, y0::T, sigma_x::T, sigma_y::T, kx0::T, ky0::T, pts_mask::Vector) where {T<:Real}

For a single state |φₘ⟩ constructs the projection coefficient for the gaussian 2d wavepacket:
αₘ = ⟨Φ|φₘ⟩ = 1/4*∮dsuₘ(s)*[∫∫Yₒ(kₘ|(x,y)-(x(s),y(s)|))*Φ(x,y)dxdy]

# Arguments
- `k::T`: Eigenvalue k.
- `us::Vector{T}`: Vector of boundary function values for the n-th state.
- `bdPoints::BoundaryPoints{T}`: The BoundaryPoints struct that contains the (x,y) points on the boundary and the ds arclength differences.
- `x_grid::Vector{T}`: Vector of x grid points.
- `y_grid::Vector{T}`: Vector of y grid points.
- `x0::T, y0::T`: The center of the wavepacket in space.
- `sigma_x::T, sigma_y::T`: standard deviations of the wavepacket in the x and y directions.
- `kx0::T, ky0::T`: The wavevectors in the x and y directions for the wavepacket.
- `pts_mask::Vector{Bool}`: Vector indicating which points in the grid are inside the billiard. This is supplied from the top calling function. It uses the `inside_only=true` case always. In the top function this is the result of `points_in_billiard_polygon` function.

# Returns
- `T`: The projection coefficient αₘ for the given state.
"""
function gaussian_wavepacket_eigenbasis_expansion_coefficient(k::T, us::Vector{T}, bdPoints::BoundaryPoints{T}, x_grid::Vector{T}, y_grid::Vector{T}, x0::T, y0::T, sigma_x::T, sigma_y::T, kx0::T, ky0::T, pts_mask::Vector) where {T<:Real}
    # Compute the Gaussian wavepacket
    packet_full = gaussian_wavepacket_2d(x_grid, y_grid, x0, y0, sigma_x, sigma_y, kx0, ky0)
    dx = x_grid[2] - x_grid[1] # Precompute x grid spacing
    dy = y_grid[2] - y_grid[1] # Precompute y grid spacing
    dxdy = dx * dy # integration volume
    # Mask the packet values
    pts = collect(SVector(x, y) for y in y_grid for x in x_grid)
    pts_masked_indices = findall(pts_mask)
    packet_masked = packet_full[pts_masked_indices] # The 2d gaussian wavepacket on the non-trivial billiard grid (inside the billiard)

    # Make a flattened vector w/ same length as the non-trivial no-masked indices (true for inside the billiard)
    distances_masked = Vector{T}(undef, length(pts_masked_indices))

    function proj(xy_s::SVector{2,T}) # this is the s term in the coefficient of expansion of the gaussian wave packet in the eigenbasis (given as a matrix) ∫∫dxdyY0(k|(x,y) - (x_s, y_s)|)*Φ(x,y) where Φ(x,y) is the Gaussian wave packet
        x_s, y_s = xy_s
        # Compute distances for masked points
        Threads.@threads for idx in eachindex(pts_masked_indices)
            global_idx = pts_masked_indices[idx]
            distances_masked[idx] = norm(pts[global_idx] - SVector(x_s, y_s))
        end
        # Compute Bessel function values for masked points
        Y0_values = Bessels.bessely0.(k .* distances_masked)

        return sum(Y0_values .* packet_masked) * dxdy # element wise sum ∑_ij A[i,j]*B[i,j]dx*dy for the double integral approximation. The consturction of the grid is linear which implies that the dx and dy are constant
    end
     # Compute the sum of u(s) * F(x(s), y(s)) over the boundary
     total_sum = Threads.Atomic{T}(0.0)
     Threads.@threads for i in eachindex(us)
         xy_s = bdPoints.xy[i]  # Boundary point (x(s), y(s))
         contribution = us[i] * proj(xy_s) * bdPoints.ds[i]  # Include boundary element size ds
         Threads.atomic_add!(total_sum, contribution)
     end
     return total_sum[] * 0.25 # 1/4 * ∮ u_n(s) * ∫∫dxdyY0(k|(x,y) - (x_s, y_s)|)*Φ(x,y), used Atomic for threading since mutating same total_sum
end

"""
    compute_all_projection_coefficients(ks::Vector{T}, vec_us::Vector{Vector{T}}, vec_bdPoints::Vector{BoundaryPoints{T}}, x_grid::Vector{T}, y_grid::Vector{T}, x0::T, y0::T, sigma_x::T, sigma_y::T, kx0::T, ky0::T, billiard::Bi) where {Bi<:AbsBilliard, T<:Real}

High-level function that computes all the projection coefficients αₘ for a given Gaussian wavepacket
and a set of eigenstates.

# Arguments
- `ks::Vector{T}`: Vector of eigenvalues for the eigenstates.
- `vec_us::Vector{Vector{T}}`: Vector of boundary function values for each eigenstate.
- `vec_bdPoints::Vector{BoundaryPoints{T}}`: Vector of `BoundaryPoints` structs for each eigenstate.
- `x_grid::Vector{T}, y_grid::Vector{T}`: Grids defining the spatial domain of the billiard.
- `x0::T, y0::T`: The center of the Gaussian wavepacket in space.
- `sigma_x::T, sigma_y::T`: Standard deviations of the Gaussian wavepacket in x and y directions.
- `kx0::T, ky0::T`: Wavevectors of the Gaussian wavepacket in x and y directions.
- `billiard::Bi`: The billiard geometry.

# Returns
- `Vector{T}`: A vector of projection coefficients αₘ for each eigenstate.
"""
function compute_all_projection_coefficients(ks::Vector{T},vec_us::Vector{Vector{T}},vec_bdPoints::Vector{BoundaryPoints{T}},x_grid::Vector{T},y_grid::Vector{T},x0::T,y0::T,sigma_x::T,sigma_y::T,kx0::T,ky0::T,billiard::Bi) where {Bi<:AbsBilliard, T<:Real}
    # Generate the point mask for points inside the billiard
    pts = collect(SVector(x, y) for y in y_grid for x in x_grid)
    sz = length(pts)
    pts_mask = points_in_billiard_polygon(pts, billiard, round(Int, sqrt(sz)); fundamental_domain=true)
    projection_coefficients = Vector{T}(undef, length(ks))
    # Compute each projection coefficient in parallel
    progress = Progress(length(ks), desc = "Computing α_m for each k...")
    Threads.@threads for i in eachindex(ks)
        k = ks[i]
        us = vec_us[i]
        bdPoints = vec_bdPoints[i]
        projection_coefficients[i] = gaussian_wavepacket_eigenbasis_expansion_coefficient(k, us, bdPoints, x_grid, y_grid, x0, y0, sigma_x, sigma_y, kx0, ky0, pts_mask)
        next!(progress)
    end
    return projection_coefficients
end







### GENERAL OTOC CONSTRUCTION - NOT USED FREQUENTLY, STILL BEING TESTED

#=

"""
    b_nk(u_n::Vector, u_m::Vector, F_nk::Function)

This is a general term that computes the matrix element of 2 and 4 point OTOC (but just a single element in the matrix). As inputs we need the integrals of the operator Ô with the Neumann function ∫∫dxdy[Y0(k|(x,y) - (x_s, y_s)|)]Ô[Y0(k|(x,y) - (x_s, y_s)|)]. This determines the effect of the operator on the resulting matrix element in the micro and macrocanonical OTOC.

# Arguments
- `u_n::Vector{<:Real}`: Vector of boundary function values for the n-th state.
- `u_m::Vector{<:Real}`: Vector of boundary function values for the m-th state.
- `F_nk::Function`: `(s::T, s'::T) -> Complex{T}` Function that computes the integral opearator, should be in general Complex. The arguments are the arclength s values for each state along the boundary, used for boundary integrals.

# Returns
- `Complex{T}`: The computed matrix element.
"""
function b_nk(u_n::Vector, u_m::Vector, bdPoints_n::BoundaryPoints, bdPoints_m::BoundaryPoints, F_nk::Function)
    ds_n = bdPoints_n.ds 
    ds_m = bdPoints_m.ds
    s_n = bdPoints_n.s
    s_m = bdPoints_m.s
    total_sum = zero(Complex{T})
    Threads.@threads for i in eachindex(u_n)  # Iterate over boundary points of u_n and u_m
        local_sum = zero(Complex{T})
        for j in eachindex(u_m)
            local_sum += u_n[i] * u_m[j] * F_nk(s_n[i], s_m[j]) * ds_n[i] * ds_m[j]
        end
        total_sum += local_sum
    end
    return total_sum
end

"""
    b(vec_us::Vector{Vector{T}}, vec_bdPoints::Vector{BoundaryPoints{T}}, F_nk::Function)

Construct the matrix of the b_nk function. This one is finally used as the main varying part in the 2 and 4 point OTOC.

# Arguments
- `vec_us::Vector{Vector{T}}`: Vector of vectors of boundary function values for each state.
- `vec_bdPoints::Vector{BoundaryPoints{T}}`: Vector of BoundaryPoints structs for each state.
- `F_nk::Function`: `(s::T, s'::T) -> Complex{T}` Function that computes the integral opearator, should be in general Complex. The arguments are the arclength s values for each state along the boundary, used for boundary integrals.

# Returns
- `Matrix{T}`: The computed matrix of b_nk function values.
"""
function b(vec_us::Vector{Vector{T}}, vec_bdPoints::Vector{BoundaryPoints{T}}, F_nk::Function) where {T<:Real}
    @assert length(vec_us) == length(vec_bdPoints) "Length of boundary functions should be the same as the number of BoundaryPoints structs"
    N_states = length(vec_us)
    full_matrix = fill(zero(Complex{T}), N_states, N_states)
    for i in eachindex(vec_us) # lost of threading inside the inner loops
        for j in eachindex(vec_us) 
            full_matrix[i,j] = b_nk(vec_us[i], vec_us[j], vec_bdPoints[i], vec_bdPoints[j], F_nk)
        end
    end
    return full_matrix
end

"""
    B(ks::Vector{T}, A::Matrix, B::Matrix, t::T) where {T<:Real}

Constructs the bₙₘ matrix from which we can construct the OTOC expression.

# Arguments
- `ks::Vector{T}`: Vector of eigenvalues.
- `A::Matrix`: Matrix representing the operator A.
- `B::Matrix`: Matrix representing the operator B.
- `t::T`: Time value for the OTOC computation.

# Returns
- `Matrix{Complex{T}}`: The computed bₙₘ matrix at a time t.
"""
function B(ks::Vector{T}, A::Matrix, B::Matrix, t::T) where {T<:Real}
    function matrix_term(i::Int,j::Int,k::Int,t)
        return -im*(exp(im*(ks[i]-ks[j])*t)*A[i,k]*B[k,j] - exp(im*(ks[k]-ks[j])*t)*B[i,k]*A[k,m])
    end
    full_matrix = fill(zero(Complex{T}, size(A)))
    for i in eachindex(size(A)[1]) 
        for j in eachindex(size(A)[2]) # same as 1 since square matrix
            full_matrix[i,j] = sum([matrix_term(j,j,k,t) for k in ks])
        end
    end
    return full_matrix
end

"""
    B(ks::Vector{T}, A::Matrix, B::Matrix, t::T) where {T<:Real}

Constructs the bₙₘ matrix from which we can construct the OTOC expression. Just a wrapper for iterated single t function.

# Arguments
- `ks::Vector{T}`: Vector of eigenvalues.
- `A::Matrix`: Matrix representing the operator A.
- `B::Matrix`: Matrix representing the operator B.
- `t::T`: Time value for the OTOC computation.

# Returns
- `Vector{Matrix{Complex{T}}}`: The computed bₙₘ matrix for times ts.
"""
function B(ks::Vector{T}, A::Matrix, B::Matrix, ts::Vector) where {T<:Real}
    return [B(ks,A,B,t) for t in ts]
end

"""
    OTOC_2_Point_Precursor(ks::Vector{T}, vec_us::Vector{Vector{T}}, vec_bdPoints::Vector{BoundaryPoints{T}}, A_nk::Function, B_nk::Function, ts::Vector{T}, x_grid::Vector{T}, y_grid::Vector{T}, x0::T, y0::T, sigma_x::T, sigma_y::T, kx0::T, ky0::T) where {T<:Real}

High-level wrapper for the construction of 2 point OTOC using general operators, represented through the double integral / sum as A ↔ A_nk, B ↔ B_nk. These functions should be supplied by the user. For the most specific case where A = x̂ and B = p̂ we there is another simpler high level wrapper.

# Arguments
- `ks::Vector{T}`: Vector of eigenvalues.
- `vec_us::Vector{Vector{T}}`: Vector of vectors of boundary function values for each state.
- `bdPoints::BoundaryPoints{T}`: The BoundaryPoints struct that contains the (x,y) points on the boundary and the ds arclength differences.
- `A_nk::Function`: `(s::T, s'::T) -> Complex{T}` Function that computes the integral opearator, should be in general Complex.
- `B_nk::Function`: `(s::T, s'::T) -> Complex{T}` Function that computes the integral opearator, should be in general Complex.
- `ts::Vector{T}`: Vector of time values for the OTOC computation.
- `x_grid::Vector{T}`: Vector of x grid points.
- `y_grid::Vector{T}`: Vector of y grid points.
- `x0::T`: Initial x-position of the Gaussian wave packet.
- `y0::T`: Initial y-position of the Gaussian wave packet.
- `sigma_x::T`: Standard deviation of the Gaussian wave packet in the x-direction.
- `sigma_y::T`: Standard deviation of the Gaussian wave packet in the y-direction.
- `kx0::T`: Initial kx of the Gaussian wave packet.
- `ky0::T`: Initial ky of the Gaussian wave packet.

# Returns
- `Vector{Complex{T}}`: The computed 2-point OTOC at given times ts.
"""
function OTOC_2_Point_Precursor(ks::Vector{T}, vec_us::Vector{Vector{T}}, vec_bdPoints::Vector{BoundaryPoints{T}}, A_nk::Function, B_nk::Function, ts::Vector{T}, x_grid::Vector{T}, y_grid::Vector{T}, x0::T, y0::T, sigma_x::T, sigma_y::T, kx0::T, ky0::T) where {T<:Real}
    O1 = b(vec_us, vec_bdPoints, A_nk)
    O2 = b(vec_us, vec_bdPoints, B_nk)
    αs = Vector{Complex{T}}(undef, length(ks))
    println("Constructing wavepacket eigenbasis expansion coefficients...")
    for i in eachindex(ks)
        αs[i] = gaussian_wavepacket_eigenbasis_expansion_coefficient(ks[i], vec_us[i], vec_bdPoints[i], x_grid, y_grid, x0, y0, sigma_x, sigma_y, kx0, ky0)
    end
    println("Constructing the B matrix...")
    B_matrix = B(ks,O1, O2,ts) # this is the the time dependance
    result = Complex{0.0}
    result = zero(Complex{T})
    progress = Progress(length(ks), desc = "Computing OTOC precursor...")
    for i in eachindex(ks)
        for j in eachindex(ks)
            result += αs[i]*αs[j]*B_matrix[i, j]
        end
        next!(progress)
    end
    return result
end

"""
    OTOC_4_Point_Precursor(ks::Vector{T}, vec_us::Vector{Vector{T}}, vec_bdPoints::Vector{BoundaryPoints{T}}, A_nk::Function, B_nk::Function, ts::Vector{T}, x_grid::Vector{T}, y_grid::Vector{T}, x0::T, y0::T, sigma_x::T, sigma_y::T, kx0::T, ky0::T) where {T<:Real}

High-level wrapper for the construction of 4 point OTOC using general operators, represented through the double integral / sum as A ↔ A_nk, B ↔ B_nk. These functions should be supplied by the user. For the most specific case where A = x̂ and B = p̂ we there is another simpler high level wrapper.
This is one is different from the 2 Point in the fact that it has another B summation in the inside of the double sum.

# Arguments
- `ks::Vector{T}`: Vector of eigenvalues.
- `vec_us::Vector{Vector{T}}`: Vector of vectors of boundary function values for each state.
- `bdPoints::Vector{BoundaryPoints{T}}`: The BoundaryPoints struct that contains the (x,y) points on the boundary and the ds arclength differences.
- `A_nk::Function`: `(s::T, s'::T) -> Complex{T}` Function that computes the integral opearator, should be in general Complex.
- `B_nk::Function`: `(s::T, s'::T) -> Complex{T}` Function that computes the integral opearator, should be in general Complex.
- `ts::Vector{T}`: Vector of time values for the OTOC computation.
- `x_grid::Vector{T}`: Vector of x grid points.
- `y_grid::Vector{T}`: Vector of y grid points.
- `x0::T`: Initial x-position of the Gaussian wave packet.
- `y0::T`: Initial y-position of the Gaussian wave packet.
- `sigma_x::T`: Standard deviation of the Gaussian wave packet in the x-direction.
- `sigma_y::T`: Standard deviation of the Gaussian wave packet in the y-direction.
- `kx0::T`: Initial kx of the Gaussian wave packet.
- `ky0::T`: Initial ky of the Gaussian wave packet.

# Returns
- `Vector{Complex{T}}`: The computed 4-point OTOC at given times ts.
"""
function OTOC_4_Point_Precursor(ks::Vector{T}, vec_us::Vector{Vector{T}}, vec_bdPoints::Vector{BoundaryPoints{T}}, A_nk::Function, B_nk::Function, ts::Vector{T}, x_grid::Vector{T}, y_grid::Vector{T}, x0::T, y0::T, sigma_x::T, sigma_y::T, kx0::T, ky0::T) where {T<:Real}
    O1 = b(vec_us, vec_bdPoints, A_nk)
    O2 = b(vec_us, vec_bdPoints, B_nk)
    αs = Vector{Complex{T}}(undef, length(ks))
    println("Constructing wavepacket eigenbasis expansion coefficients...")
    for i in eachindex(ks)
        αs[i] = gaussian_wavepacket_eigenbasis_expansion_coefficient(ks[i], vec_us[i], vec_bdPoints[i], x_grid, y_grid, x0, y0, sigma_x, sigma_y, kx0, ky0)
    end
    println("Constructing the B matrix...")
    B_matrix = B(ks,O1,O2,ts)
    result = Complex{0.0}
    result = zero(Complex{T})
    progress = Progress(length(ks), desc = "Computing OTOC precursor...")
    for i in eachindex(ks)
        for j in eachindex(ks)
            result += αs[i]*αs[j]*[B_matrix[i, k]*B_matrix[j, k] for k in eachindex(ks)]
        end
        next!(progress)
    end
    return result
end

=#






### SPECIFIC AND MOST USEFUL CASE W/ TESTING : A = X, B = P -> [X,P] OTOC
# Taken from : Out-of-time-order correlators in quantum mechanics by Koji Hashimoto, Keiju Murata and Ryosuke Yoshii. For the most typical [x(t),p(0)] expectation value with the gaussian wavepacket the calculations are greatly simplified. We just need to calculate the xₘₙ terms and have the eigenvalues.

"""
    X_mn_standard(k_m::T, k_n::T, us_m::Vector{T}, us_n::Vector{T}, bdPoints_m::BoundaryPoints{T}, bdPoints_n::BoundaryPoints{T}, x_grid::Vector{T}, y_grid::Vector{T}, pts_mask::Vector) where {T<:Real}

Constructs the (m,n)-th component of the xₘₙ term for the coordinate matrix element xₘₙ=⟨m|x|n⟩. This is calculated as:
```math
1/4*∮uₘ(s')uₙ(s)*[∫∫Y0(kₘ|(x,y)-(xₛ, yₛ)|)*x*Y0(kₙ|(x,y)-(x'ₛ, y'ₛ)|)dxdy]*ds*ds'
```

# Arguments
- `k_m::T`: Eigenvalue for the m-th state.
- `k_n::T`: Eigenvalue for the m-th state.
- `us_m::Vector{T}`: Vector of boundary function values for the m-th state.
- `us_n::Vector{T}`: Vector of boundary function values for the n-th state.
- `bdPoints_m::BoundaryPoints{T}`: The BoundaryPoints struct for the m-th state.
- `bdPoints_n::BoundaryPoints{T}`: The BoundaryPoints struct for the n-th state.
- `x_grid::Vector{T}`: Vector of x grid points.
- `y_grid::Vector{T}`: Vector of y grid points.
- `pts_mask::Vector{Bool}`: Vector indicating which points in the grid are inside the billiard. This is supplied from the top calling function. It uses the `inside_only=true` case always. In the top function this is the result of `points_in_billiard_polygon` function.

# Returns
- `T`: The (m,n)-th component of the xₘₙ term.
"""
function X_mn_standard(k_m::T, k_n::T, us_m::Vector{T}, us_n::Vector{T}, bdPoints_m::BoundaryPoints{T}, bdPoints_n::BoundaryPoints{T}, x_grid::Vector{T}, y_grid::Vector{T}, pts_mask::Vector) where {T<:Real}
    # Grid spacings
    dx = x_grid[2] - x_grid[1]
    dy = y_grid[2] - y_grid[1]
    dxdy = dx * dy
    
    # Create grid points and apply mask to determine points inside the billiard
    pts = collect(SVector(x, y) for y in y_grid for x in x_grid)
    pts_masked = pts[pts_mask]
    
    # Preallocate arrays for distances
    distances_m = Vector{T}(undef, length(pts_masked))
    distances_n = Vector{T}(undef, length(pts_masked))
    
    # Function to compute the double integral for a given boundary point pair (s, s')
    function double_integral(xy_s_m::SVector{2, T}, xy_s_n::SVector{2, T})
        x_s_m, y_s_m = xy_s_m
        x_s_n, y_s_n = xy_s_n

        # Compute distances for masked points
        @inbounds Threads.@threads for idx in eachindex(pts_masked)
            distances_m[idx] = norm(pts_masked[idx] - SVector(x_s_m, y_s_m))
            distances_n[idx] = norm(pts_masked[idx] - SVector(x_s_n, y_s_n))
        end

        # Compute Bessel functions for masked points
        Y0_m = Bessels.bessely0.(k_m .* distances_m)
        Y0_n = Bessels.bessely0.(k_n .* distances_n)

        # Apply x-operator and perform element-wise multiplication
        x_operator_values = getindex.(pts_masked, 1)  # Extract x-coordinates of masked points
        integrand = Y0_m .* x_operator_values .* Y0_n

        # Compute the sum over the masked grid points
        return sum(integrand) * dxdy
    end
    # Compute the full double boundary integral
    total_result = Threads.Atomic{T}(0.0)
    progress = Progress(length(us_m)*length(us_n), desc="Computing X_mn for k_m=$(round(k_m; sigdigits=5)), k_n=$(round(k_n; sigdigits=5))...")
    #println("Computing X_mn for k_m=$(round(k_m; sigdigits=5)), k_n=$(round(k_n; sigdigits=5))...")
    Threads.@threads for i in eachindex(us_m)
        local_result = zero(T)  # Thread-local accumulator
        println("Thread $(Threads.threadid()): Starting i=", i)  # Thread-safe thread ID
        @inbounds for j in eachindex(us_n)
            xy_s_m = bdPoints_m.xy[i]
            xy_s_n = bdPoints_n.xy[j]
            local_result += us_m[i] * us_n[j] * double_integral(xy_s_m, xy_s_n) * bdPoints_m.ds[i] * bdPoints_n.ds[j]
        end
        Threads.atomic_add!(total_result, local_result)
        #println("Thread $(Threads.threadid()): Done i=", i)  # Thread-safe completion log
        next!(progress)
    end
    return total_result[] / 4.0
end

"""
    X_standard(ks::Vector{T}, vec_us::Vector{Vector{T}}, vec_bdPoints::Vector{BoundaryPoints{T}}, x_grid::Vector{T}, y_grid::Vector{T}, billiard::Bi) where {T<:Real, Bi<:AbsBilliard}

Computes the full X matrix of xₘₙ=⟨m|x|n⟩. Just a multithreaded wrapper for the `X_mn` function.

# Arguments
- `ks::Vector{T}`: Vector of eigenvalues.
- `vec_us::Vector{Vector{T}}`: Vector of vectors of boundary function values for each state.
- `vec_bdPoints::Vector{BoundaryPoints{T}}`: Vector of BoundaryPoints structs for each state.
- `x_grid::Vector{T}`: Vector of x grid points.
- `y_grid::Vector{T}`: Vector of y grid points.
- `billiard::Bi`: The billiard geometry.

# Returns
- `Matrix{T}`: The full X matrix of xₘₙ=⟨m|x|n⟩. It's real!
"""
function X_standard(ks::Vector{T}, vec_us::Vector{Vector{T}}, vec_bdPoints::Vector{BoundaryPoints{T}}, x_grid::Vector{T}, y_grid::Vector{T}, billiard::Bi) where {T<:Real, Bi<:AbsBilliard}
    full_matrix = fill(0.0, length(ks), length(ks))
    pts = collect(SVector(x, y) for y in y_grid for x in x_grid)
    sz = length(pts)
    pts_mask = points_in_billiard_polygon(pts, billiard, round(Int, sqrt(sz)); fundamental_domain=true) # only compute once
    progress = Progress(length(ks), desc="Computing X_mn matrix elements...")
    for i in eachindex(ks)
        for j in eachindex(ks)
            full_matrix[i,j] = X_mn_standard(ks[i], ks[j], vec_us[i], vec_us[j], vec_bdPoints[j], vec_bdPoints[j], x_grid, y_grid, pts_mask)
        end
        next!(progress)
    end
    return full_matrix
end

"""
    B_standard(ks::Vector{T}, vec_us::Vector{Vector{T}}, vec_bdPoints::Vector{BoundaryPoints{T}}, x_grid::Vector{T}, y_grid::Vector{T}, t::T, billiard::Bi) where {T<:Real, Bi<:AbsBilliard}

Standard `[x(t),p(0)]` OTOC B matrix construction, where we do not need to explicitely construct the Pₘₙ matrix since we only need Xₘₙ. This is based on the paper: Out-of-time-order correlators in quantum mechanics; Koji Hashimoto, Keiju Murata and Ryosuke Yoshii, specifically chapters 2 and 5.

# Arguements
- `ks::Vector{T}`: Vector of eigenvalues.
- `vec_us::Vector{Vector{T}}`: Vector of vectors of boundary function values for each state.
- `vec_bdPoints::Vector{BoundaryPoints{T}}`: Vector of BoundaryPoints structs for each state.
- `x_grid::Vector{T}`: Vector of x grid points.
- `y_grid::Vector{T}`: Vector of y grid points.
- `t::T`: Time parameter.
- `billiard::Bi`: The billiard geometry.

# Returns
- `Matrix{Complex{T}}`: The full B matrix of Bₘₙ=⟨m|B|n⟩. It's complex!
"""
function B_standard(ks::Vector{T}, vec_us::Vector{Vector{T}}, vec_bdPoints::Vector{BoundaryPoints{T}}, x_grid::Vector{T}, y_grid::Vector{T}, t::T, billiard::Bi) where {T<:Real, Bi<:AbsBilliard}
    Es = ks .^2 # get energies
    X_matrix = X_standard(ks, vec_us, vec_bdPoints, x_grid, y_grid, billiard)
    B_matrix = fill(Complex(0.0), length(ks), length(ks))
    Threads.@threads for i in eachindex(ks)
        local_result = Vector{Complex{T}}(undef, length(ks))  # Thread-local storage for row `i`
        for j in eachindex(ks)
            # Inner summation over k
            sum_k = zero(Complex{T})
            for k in eachindex(ks)
                E_ik = Es[i] - Es[k]
                E_kj = Es[k] - Es[j]
                # Compute the term for the summation
                term = X_matrix[i, k] * X_matrix[k, j] * (E_kj * exp(im * E_ik * t) - E_ik * exp(im * E_kj * t))
                sum_k += term  # Accumulate the term
            end
            local_result[j] = sum_k  # Store in thread-local result
        end
        B_matrix[i,:] = local_result # Doing it row by row
    end
    return 0.5*B_matrix
end

"""
    B_standard(ks::Vector{T}, vec_us::Vector{Vector{T}}, vec_bdPoints::Vector{BoundaryPoints{T}}, x_grid::Vector{T}, y_grid::Vector{T}, ts::Vector{T}, billiard::Bi) where {T<:Real, Bi<:AbsBilliard}

High level wrapper for standard `[x(t),p(0)]` OTOC B matrix construction. It just iterates the base `B_standard` function over a time interval.

# Arguments
- `ks::Vector{T}`: Vector of eigenvalues.
- `vec_us::Vector{Vector{T}}`: Vector of vectors of boundary function values for each state.
- `vec_bdPoints::Vector{BoundaryPoints{T}}`: Vector of BoundaryPoints structs for each state.
- `x_grid::Vector{T}`: Vector of x grid points.
- `y_grid::Vector{T}`: Vector of y grid points.
- `ts::Vector{T}`: Vector of time parameters.
- `billiard::Bi`: The billiard geometry.

# Returns
- `Vector{Matrix{Complex{T}}}`: Vector of B matrices for each time parameter in `ts`. The matrix elements are Complex.
"""
function B_standard(ks::Vector{T}, vec_us::Vector{Vector{T}}, vec_bdPoints::Vector{BoundaryPoints{T}}, x_grid::Vector{T}, y_grid::Vector{T}, ts::Vector{T}, billiard::Bi) where {T<:Real, Bi<:AbsBilliard}
    progress = Progress(length(ts), desc="Computing B matrices for all times...")
    B_matrices = Vector{Matrix{Complex{T}}}(undef, length(ts))
    for t_idx in eachindex(ts)
        println("t idx: ", t_idx)
        t = ts[t_idx]
        B_matrices[t_idx] = B_standard(ks, vec_us, vec_bdPoints, x_grid, y_grid, t, billiard)
        next!(progress)
    end
    return B_matrices
end

"""
    microcanocinal_Cn_standard(ks::Vector{T}, vec_us::Vector{Vector{T}}, vec_bdPoints::Vector{BoundaryPoints{T}}, alphas::Vector{Complex{T}},t::T, billiard::Bi) where {T<:Real, Bi<:AbsBilliard}

Computes the microcanonical cₙ(t) for all n for a time t.

# Arguments
- `ks::Vector{T}`: Vector of eigenvalues.
- `vec_us::Vector{Vector{T}}`: Vector of vectors of boundary function values for each state.
- `vec_bdPoints::Vector{BoundaryPoints{T}}`: Vector of BoundaryPoints structs for each state.
- `alphas::Vector{Complex{T}}`: Expansion coefficients for the Gaussian wavepacket in the eigenbasis.
- `t::T`: Time parameter.
- `billiard::Bi`: The billiard geometry.

# Returns
- `T`: The microcanonical cₙ(t) for all n.
"""
function microcanocinal_Cn_standard(ks::Vector{T}, vec_us::Vector{Vector{T}}, vec_bdPoints::Vector{BoundaryPoints{T}}, alphas::Vector{Complex{T}},t::T, billiard::Bi) where {T<:Real, Bi<:AbsBilliard}
    k_max = maximum(ks)
    type = eltype(k_max)
    L = billiard.length
    xlim, ylim = boundary_limits(billiard.full_boundary; grd=max(1000, round(Int, k_max * L * b / (2 * pi))))
    dx, dy = xlim[2] - xlim[1], ylim[2] - ylim[1]
    nx, ny = max(round(Int, k_max * dx * b / (2 * pi)), 512), max(round(Int, k_max * dy * b / (2 * pi)), 512)
    x_grid, y_grid = collect(type, range(xlim..., nx)), collect(type, range(ylim..., ny))
    B_standard_matrix = B_standard(ks, vec_us, vec_bdPoints, x_grid, y_grid, t, billiard)
    total_iterations = length(ks) * length(ks)
    progress = Progress(total_iterations, desc="Computing single-time c_φ(t)")
    c = Threads.Atomic{Complex{T}}(0.0)  # Thread-safe atomic variable

    Threads.@threads for n in eachindex(ks)
        local_sum = zero(Complex{T})  # Thread-local accumulator
        for m in eachindex(ks)
            alpha_factor = conj(alphas[n]) * alphas[m]  # Precompute αₙ* * αₘ since not depend on k
            for k in eachindex(ks)
                local_sum += alpha_factor * B_standard_matrix[n, k] * B_standard_matrix[k, m]
            end
            next!(progress)  # Update m+=1 otherwise bloat
        end
        Threads.atomic_add!(c, local_sum)  # Safely accumulate into the global result
    end
    return real(c[])  # Return real part / sanity check
end

"""
    microcanocinal_Cn_standard(ks::Vector{T}, vec_us::Vector{Vector{T}}, vec_bdPoints::Vector{BoundaryPoints{T}}, ts::Vector{T}) where {T<:Real}

Computes the microcanonical `cₙ(t)` for all `n` over a series of times `ts`.

# Arguments
- `ks::Vector{T}`: Vector of eigenvalues.
- `vec_us::Vector{Vector{T}}`: Vector of vectors of boundary function values for each state.
- `vec_bdPoints::Vector{BoundaryPoints{T}}`: Vector of BoundaryPoints structs for each state.
- `alphas::Vector{Complex{T}}`: Expansion coefficients for the Gaussian wavepacket in the eigenbasis.
- `ts::Vector{T}`: Vector of time points.
- `billiard::Bi`: The billiard geometry.

# Returns
- `Vector{T}`: Vector of `cₙ(t) values for each t in ts.
"""
function microcanocinal_Cn_standard(ks::Vector{T}, vec_us::Vector{Vector{T}}, vec_bdPoints::Vector{BoundaryPoints{T}}, alphas::Vector{Complex{T}}, ts::Vector{T}, billiard::Bi) where {T<:Real, Bi<:AbsBilliard}
    k_max = maximum(ks)
    type = eltype(k_max)
    L = billiard.length
    xlim, ylim = boundary_limits(billiard.full_boundary; grd=max(1000, round(Int, k_max * L * b / (2 * pi))))
    dx, dy = xlim[2] - xlim[1], ylim[2] - ylim[1]
    nx, ny = max(round(Int, k_max * dx * b / (2 * pi)), 512), max(round(Int, k_max * dy * b / (2 * pi)), 512)
    x_grid, y_grid = collect(type, range(xlim..., nx)), collect(type, range(ylim..., ny))
    total_iterations = length(ts) * length(ks) * length(ks)
    progress = Progress(total_iterations, desc="Computing multi-time c_φ(t)")
    B_matrices = B_standard(ks, vec_us, vec_bdPoints, x_grid, y_grid, ts, billiard)
    result = Vector{T}(undef, length(ts))  # Store c_φ(t) for each time step
    Threads.@threads for t_idx in eachindex(ts)
        B_standard_matrix = B_matrices[t_idx]
        c = Threads.Atomic{Complex{T}}(0.0)  # Thread-safe accumulator for the current t_idx
        Threads.@threads for n in eachindex(ks)
            local_sum = zero(Complex{T})  # Thread-local accumulator
            for m in eachindex(ks)
                alpha_factor = conj(alphas[n]) * alphas[m]  # Precompute αₙ* * αₘ
                for k in eachindex(ks)
                    local_sum += alpha_factor * B_standard_matrix[n, k] * B_standard_matrix[k, m]
                end
                next!(progress)  # Update progress for the `m` loop
            end
            Threads.atomic_add!(c, local_sum)  # Accumulate into thread-safe atomic
        end
        result[t_idx] = real(c[])  # Store the real part for this time step
    end
    return result  # Return a vector of c_φ(t) values for each t
end

"""
    plot_microcanonical_Cn!(ax::Axis, ts::Vector{T}, c_values::Matrix{T}, n::Int; log_scale::Bool=false) where {T<:Real}

Plots the microcanonical `cₙ(t)` for a given eigenstate `n` over times `ts`.

# Arguments
- `ax::Axis`: Axis object from CairoMakie for plotting.
- `ts::Vector{T}`: Vector of time points.
- `c_values::Matrix{T}`: Matrix of cₙ(t) values. Rows correspond to times, columns to eigenstates.
- `n::Int`: Index of the eigenstate to plot.
- `log_scale::Bool=false`: Whether to use a logarithmic scale for the y-axis.

# Returns
- `Nothing`
"""
function plot_microcanonical_Cn!(ax::Axis, ts::Vector{T}, c_values::Matrix{T}, n::Int; log_scale::Bool=false) where {T<:Real}
    @assert n > 0 && n <= size(c_values, 2) "Index `n` out of range for the c_values matrix." # Sanity check
    cn_t = c_values[:, n]
    if log_scale
        lines!(ax, ts, log10.(cn_t), linewidth=2, color=:blue, label="c_$(n)(t)")
        ax.ylabel = "log10(cₙ(t))"
    else
        lines!(ax, ts, cn_t, linewidth=2, color=:blue, label="c_$(n)(t)")
        ax.ylabel = "cₙ(t)"
    end
    ax.xlabel = "t"
    ax.title = "Microcanonical cₙ(t) for n = $n"
    axislegend(ax)
end

"""
    plot_microcanonical_Cn!(ax::Axis, ts::Vector{T}, c_values::Matrix{T}, indices::Vector{Int}; log_scale::Bool=false) where {T<:Real}

Plots the microcanonical `cₙ(t)` for a set of eigenstates `indices` over times `ts`.

# Arguments
- `ax::Axis`: Axis object from CairoMakie for plotting.
- `ts::Vector{T}`: Vector of time points.
- `c_values::Matrix{T}`: Matrix of `cₙ(t)` values. Rows correspond to times, columns to eigenstates.
- `indices::Vector{Int}`: Indices of the eigenstates to plot.
- `log_scale::Bool=false`: Whether to use a logarithmic scale for the y-axis.

# Returns
- `Nothing`
"""
function plot_microcanonical_Cn!(ax::Axis, ts::Vector{T}, c_values::Matrix{T}, indices::Vector{Int}; log_scale::Bool=false) where {T<:Real}
    for n in indices
        @assert n > 0 && n <= size(c_values, 2) "Index `n` out of range for the c_values matrix."
        cn_t = c_values[:, n]
        if log_scale
            lines!(ax, ts, log10.(cn_t), linewidth=2, label="c_$(n)(t)")
        else
            lines!(ax, ts, cn_t, linewidth=2, label="c_$(n)(t)")
        end
    end
    ax.xlabel = "t"
    ax.ylabel = log_scale ? "log10(cₙ(t))" : "cₙ(t)"
    ax.title = "Microcanonical cₙ(t) for n ∈ $(indices)"
    axislegend(ax)
end


### NO WAVEPACKET VERSIONS FOR Cₙ(t)

"""
    microcanonical_Cn_no_wavepacket(ks::Vector{T},vec_us::Vector{Vector{T}},vec_bdPoints::Vector{BoundaryPoints{T}},ts::Vector{T}, billiard::Bi) where {T<:Real, Bi<:AbsBilliard}

Computes the microcanonical `cₙ(t)` for all `n` over a series of times `ts` without using wavepacket coefficients.

# Arguments
- `ks::Vector{T}`: Vector of eigenvalues.
- `vec_us::Vector{Vector{T}}`: Vector of vectors of boundary function values for each state.
- `vec_bdPoints::Vector{BoundaryPoints{T}}`: Vector of BoundaryPoints structs for each state.
- `ts::Vector{T}`: Vector of time points.
- `billiard::Bi`: The billiard geometry.

# Returns
- `Matrix{T}`: A matrix where each row corresponds to a time `t` and each column to an eigenstate `n`.
"""
function microcanonical_Cn_no_wavepacket(ks::Vector{T},vec_us::Vector{Vector{T}},vec_bdPoints::Vector{BoundaryPoints{T}},ts::Vector{T}, billiard::Bi) where {T<:Real, Bi<:AbsBilliard}
    b = 5.0
    k_max = maximum(ks)
    type = eltype(k_max)
    L = billiard.length
    xlim, ylim = boundary_limits(billiard.full_boundary; grd=max(1000, round(Int, k_max * L * b / (2 * pi))))
    dx, dy = xlim[2] - xlim[1], ylim[2] - ylim[1]
    nx, ny = max(round(Int, k_max * dx * b / (2 * pi)), 512), max(round(Int, k_max * dy * b / (2 * pi)), 512)
    x_grid, y_grid = collect(type, range(xlim..., nx)), collect(type, range(ylim..., ny))
    # Precompute B matrices for all time values
    println("Constructing the B matrix...")
    @time B_matrices = B_standard(ks, vec_us, vec_bdPoints, x_grid, y_grid, ts, billiard)
    total_iterations = length(ts) * length(ks)
    progress = Progress(total_iterations, desc="Computing cₙ(t) without wavepacket coefficients")
    result = Matrix{T}(undef, length(ts), length(ks))  # Store cₙ(t) for each time step
    Threads.@threads for t_idx in eachindex(ts)
        B_standard_matrix = B_matrices[t_idx]
        local_result = Vector{T}(undef, length(ks))  # Store cₙ(t) for this `t_idx`
        Threads.@threads for n in eachindex(ks)  # Compute cₙ(t) = ∑ₘ |b_{nm}(t)|² for each row n
            local_result[n] = sum(abs2, B_standard_matrix[n, :])  # Sum over m
            next!(progress)  # Update the progress bar
        end
        result[t_idx, :] = local_result  # Store results for this time
    end
    return result  # Return a matrix where each row corresponds to cₙ(t) at different times
end

"""
    plot_microcanonical_Cn_no_wavepacket!(ax::Axis, c_values::Matrix{T}, ts::Vector{T}, indices::Vector{Int}) where {T<:Real}

Plots the microcanonical `cₙ(t)` values for specific `n` indices over time. NOT FOR THE WAVEPACKET VERSION

# Arguments
- `ax::Axis`: Axis object from CairoMakie for the plot.
- `c_values::Matrix{T}`: The matrix of `cₙ(t)` values where each row corresponds to a time `t` and each column to an eigenstate `n`.
- `ts::Vector{T}`: Vector of time points corresponding to the rows of `c_values`.
- `indices::Vector{Int}`: Vector of indices specifying which `n` values to plot.
"""
function plot_microcanonical_Cn_no_wavepacket!(ax::Axis, c_values::Matrix{T}, ts::Vector{T}, indices::Vector{Int}) where {T<:Real}
    for n in indices
        if n > size(c_values, 2)
            warn("Index $n is out of bounds for c_values with size $(size(c_values, 2))")
            continue
        end
        plot!(ax, ts, c_values[:, n], label="n = $n")
    end
    ax.xlabel = "Time (t)"
    ax.ylabel = "cₙ(t)"
    ax.title = "Microcanonical cₙ(t) for Selected n"
    ax.legend = true
end






### PLOTTING LOGIC AND EFFICIENT CONSTRUCTIONS - OK

"""
    ϕ(x::T, y::T, k::T, bdPoints::BoundaryPoints, us::Vector)

Computes the wavefunction via the boundary integral Ψ = 1/4∮Yₒ(k|q-qₛ|)u(s)ds. For a specific `k` it needs the boundary discretization information encoded into the `BoundaryPoints` struct:

```julia
struct BoundaryPoints{T} <: (AbsPoints where T <: Real)
xy::Vector{SVector{2, T}} # boundary (x,y) pts
normal::Vector{SVector{2, T}} # Normals at for those (x,y) pts
s::Vector{T} # Arclengths for those (x,y) pts
ds::Vector{T} # Differences between the arclengths of pts.
end
```

# Arguments
- `x::T`: x-coordinate of the point to compute the wavefunction.
- `y::T`: y-coordinate of the point to compute the wavefunction.
- `k::T`: The eigenvalue for which the wavefunction is to be computed.
- `bdPoints::BoundaryPoints`: Boundary discretization information.
- `us::Vector`: Vector of boundary functions.

# Returns
- `ϕ::T`: The value of the wavefunction at the given point (x,y).

"""
function ϕ(x::T, y::T, k::T, bdPoints::BoundaryPoints, us::Vector) where {T<:Real}
    target_point = SVector(x, y)
    distances = norm.(Ref(target_point) .- bdPoints.xy)
    weighted_bessel_values = Bessels.bessely0.(k * distances) .* us .* bdPoints.ds
    return sum(weighted_bessel_values) / 4
end

"""
    wavefunction_multi(ks::Vector{T}, vec_us::Vector{Vector{T}}, vec_bdPoints::Vector{BoundaryPoints{T}}, billiard::Bi; b::Float64=5.0, inside_only::Bool=true) where {Bi<:AbsBilliard,T<:Real}

Constructs a sequence of 2D wavefunctions as matrices over the same sized grid for easier computation of matrix elements. The matrices are constructed via the boundary integral.

# Arguments
- `ks`: Vector of eigenvalues.
- `vec_bdPoints`: Vector of `BoundaryPoints` objects, one for each eigenvalues.
- `billiard`: The billiard geometry.
- `vec_us::Vector{Vector}`: Vector of the boundary functions
- `b::Float64=5.0`: (Optional), Point scaling factor. Default is 5.0.
- `inside_only::Bool=true`: (Optional), Whether to only compute wavefunctions inside the billiard. Default is true.
- `fundamental::Bool=true`: (Optional), Whether to use fundamental domain for boundary integral. Default is true.

# Returns
- `Psi2ds::Vector{Matrix{T}}`: Vector of 2D wavefunction matrices constructed on the same grid.
- `x_grid::Vector{T}`: Vector of x-coordinates for the grid.
- `y_grid::Vector{T}`: Vector of y-coordinates for the grid.
"""
function wavefunction_multi(ks::Vector{T}, vec_us::Vector{Vector{T}}, vec_bdPoints::Vector{BoundaryPoints{T}}, billiard::Bi; b::Float64=5.0, inside_only::Bool=true, fundamental=true) where {Bi<:AbsBilliard,T<:Real}
    k_max = maximum(ks)
    type = eltype(k_max)
    L = billiard.length
    xlim, ylim = boundary_limits(billiard.full_boundary; grd=max(1000, round(Int, k_max * L * b / (2 * pi))))
    dx, dy = xlim[2] - xlim[1], ylim[2] - ylim[1]
    nx, ny = max(round(Int, k_max * dx * b / (2 * pi)), 512), max(round(Int, k_max * dy * b / (2 * pi)), 512)
    x_grid, y_grid = collect(type, range(xlim..., nx)), collect(type, range(ylim..., ny))
    pts = collect(SVector(x, y) for y in y_grid for x in x_grid)
    sz = length(pts)
    # Determine points inside the billiard only once if inside_only is true
    pts_mask = inside_only ? points_in_billiard_polygon(pts, billiard, round(Int, sqrt(sz)); fundamental_domain=fundamental) : fill(true, sz)
    pts_masked_indices = findall(pts_mask)
    # wavefunction via boundary integral Ψ = 1/4∮Yₒ(k|q-qₛ|)u(s)ds
    function ϕ(x, y, k, bdPoints::BoundaryPoints, us::Vector)
        target_point = SVector(x, y)
        distances = norm.(Ref(target_point) .- bdPoints.xy)
        weighted_bessel_values = Bessels.bessely0.(k * distances) .* us .* bdPoints.ds
        sum(weighted_bessel_values) / 4
    end
    Psi2ds = Vector{Matrix{type}}(undef, length(ks))
    progress = Progress(length(ks), desc="Constructing wavefunction matrices...")
    Threads.@threads for i in eachindex(ks)
        k, bdPoints, us = ks[i], vec_bdPoints[i], vec_us[i]
        Psi_flat = zeros(type, sz)
        @inbounds for idx in pts_masked_indices # no bounds checking
            x, y = pts[idx]
            Psi_flat[idx] = ϕ(x, y, k, bdPoints, us)
        end
        Psi2ds[i] = reshape(Psi_flat, ny, nx)
        next!(progress)
    end
    return Psi2ds, x_grid, y_grid
end

"""
    wavefunction_multi_with_husimi(ks::Vector{T}, vec_us::Vector{Vector{T}}, vec_bdPoints::Vector{BoundaryPoints{T}}, billiard::Bi; b::Float64=5.0, inside_only::Bool=true) where {Bi<:AbsBilliard,T<:Real}

Constructs a sequence of 2D wavefunctions as matrices over the same sized grid for easier computation of matrix elements. The matrices are constructed via the boundary integral. Additionally also constructs the husimi functions.

# Arguments
- `ks`: Vector of eigenvalues.
- `vec_bdPoints`: Vector of `BoundaryPoints` objects, one for each eigenvalues.
- `billiard`: The billiard geometry.
- `vec_us::Vector{Vector}`: Vector of the boundary functions
- `b::Float64=5.0`: (Optional), Point scaling factor. Default is 5.0.
- `inside_only::Bool=true`: (Optional), Whether to only compute wavefunctions inside the billiard. Default is true.
- `fundamental::Bool=true`: (Optional), Whether to use fundamental domain for boundary integral. Default is true.

# Returns
- `Psi2ds::Vector{Matrix{T}}`: Vector of 2D wavefunction matrices constructed on the same grid.
- `x_grid::Vector{T}`: Vector of x-coordinates for the grid.
- `y_grid::Vector{T}`: Vector of y-coordinates for the grid.
- `Hs_list::Vector{Matrix{T}}`: Vector of 2D husimi function matrices.
- `ps_list::Vector{Vector{T}}`: Vector of ps grids for the husimi matrices.
- `qs_list::Vector{Vector{T}}`: Vector of qs grids for the husimi matrices.
"""
function wavefunction_multi_with_husimi(ks::Vector{T}, vec_us::Vector{Vector{T}}, vec_bdPoints::Vector{BoundaryPoints{T}}, billiard::Bi; b::Float64=5.0, inside_only::Bool=true, fundamental=true) where {Bi<:AbsBilliard,T<:Real}
    k_max = maximum(ks)
    type = eltype(k_max)
    L = billiard.length
    xlim, ylim = boundary_limits(billiard.full_boundary; grd=max(1000, round(Int, k_max * L * b / (2 * pi))))
    dx, dy = xlim[2] - xlim[1], ylim[2] - ylim[1]
    nx, ny = max(round(Int, k_max * dx * b / (2 * pi)), 512), max(round(Int, k_max * dy * b / (2 * pi)), 512)
    println("nx: ", nx)
    println("ny: ", ny)
    println("Length of u(s) start, finish: ", length(vec_us[1]), ", ", length(vec_us[end]))
    x_grid, y_grid = collect(type, range(xlim..., nx)), collect(type, range(ylim..., ny))
    pts = collect(SVector(x, y) for y in y_grid for x in x_grid)
    sz = length(pts)
    # Determine points inside the billiard only once if inside_only is true
    pts_mask = inside_only ? points_in_billiard_polygon(pts, billiard, round(Int, sqrt(sz)); fundamental_domain=fundamental) : fill(true, sz)
    pts_masked_indices = findall(pts_mask)
    # wavefunction via boundary integral Ψ = 1/4∮Yₒ(k|q-qₛ|)u(s)ds
    function ϕ(x, y, k, bdPoints::BoundaryPoints, us::Vector)
        target_point = SVector(x, y)
        distances = norm.(Ref(target_point) .- bdPoints.xy)
        weighted_bessel_values = Bessels.bessely0.(k * distances) .* us .* bdPoints.ds
        sum(weighted_bessel_values) / 4
    end
    Psi2ds = Vector{Matrix{type}}(undef, length(ks))
    progress = Progress(length(ks), desc="Constructing wavefunction matrices...")
    Threads.@threads for i in eachindex(ks)
        k, bdPoints, us = ks[i], vec_bdPoints[i], vec_us[i]
        Psi_flat = zeros(type, sz)
        @inbounds for idx in pts_masked_indices # no bounds checking
            x, y = pts[idx]
            Psi_flat[idx] = ϕ(x, y, k, bdPoints, us)
        end
        Psi2ds[i] = reshape(Psi_flat, ny, nx)
        next!(progress)
    end
    # husimi
    vec_of_s_vals = [bdPoints.s for bdPoints in vec_bdPoints]
    Hs_list, ps_list, qs_list = husimi_functions_from_boundary_functions(ks, vec_us, vec_of_s_vals, billiard)
    return Psi2ds, x_grid, y_grid, Hs_list, ps_list, qs_list
end

"""
    plot_wavefunctions(ks::Vector, Psi2ds::Vector, x_grid::Vector, y_grid::Vector, billiard::Bi; b::Float64=5.0, width_ax::Integer=500, height_ax::Integer=500, max_cols::Integer=6, fundamental=true) where {Bi<:AbsBilliard}

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

 # Returns
- `f::Figure`: A Figure object containing the grid of wavefunctions.
"""
function plot_wavefunctions(ks::Vector, Psi2ds::Vector, x_grid::Vector, y_grid::Vector, billiard::Bi; b::Float64=5.0, width_ax::Integer=300, height_ax::Integer=300, max_cols::Integer=6, fundamental=true) where {Bi<:AbsBilliard}
    L = billiard.length
    if fundamental
        xlim,ylim = boundary_limits(billiard.fundamental_boundary; grd=max(1000,round(Int, maximum(ks)*L*b/(2*pi))))
    else
        xlim,ylim = boundary_limits(billiard.full_boundary; grd=max(1000,round(Int, maximum(ks)*L*b/(2*pi))))
    end
    n_rows = ceil(Int, length(ks) / max_cols)
    f = Figure(resolution=(1*width_ax * max_cols, 1*height_ax * n_rows), size=(1*width_ax * max_cols, 1*height_ax * n_rows))
    row = 1
    col = 1
    for j in eachindex(ks)
        local ax = Axis(f[row,col], title="$(ks[j])", aspect=DataAspect(), width=width_ax, height=height_ax)
        hm = heatmap!(ax, x_grid, y_grid, Psi2ds[j], colormap=:balance, colorrange=(-maximum(Psi2ds[j]), maximum(Psi2ds[j])))
        plot_boundary!(ax, billiard, fundamental_domain=fundamental, plot_normal=false)
        xlims!(ax, xlim)
        ylims!(ax, ylim)
        col += 1
        if col > max_cols
            row += 1
            col = 1
        end
    end
    return f
end

"""
    plot_wavefunctions(ks::Vector, Psi2ds::Vector, x_grid::Vector{Vector}, y_grid::Vector{Vector}, billiard::Bi; b::Float64=5.0, width_ax::Integer=500, height_ax::Integer=500, max_cols::Integer=6, fundamental=true) where {Bi<:AbsBilliard}

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

 # Returns
- `f::Figure`: A Figure object containing the grid of wavefunctions.
"""
function plot_wavefunctions(ks::Vector, Psi2ds::Vector, x_grid::Vector{Vector}, y_grid::Vector{Vector}, billiard::Bi; b::Float64=5.0, width_ax::Integer=300, height_ax::Integer=300, max_cols::Integer=6, fundamental=true) where {Bi<:AbsBilliard}
    L = billiard.length
    if fundamental
        xlim,ylim = boundary_limits(billiard.fundamental_boundary; grd=max(1000,round(Int, maximum(ks)*L*b/(2*pi))))
    else
        xlim,ylim = boundary_limits(billiard.full_boundary; grd=max(1000,round(Int, maximum(ks)*L*b/(2*pi))))
    end
    n_rows = ceil(Int, length(ks) / max_cols)
    f = Figure(resolution=(1*width_ax * max_cols, 1*height_ax * n_rows), size=(round(Int, 3.0*width_ax * max_cols), round(Int, 2.0*height_ax * n_rows)))
    row = 1
    col = 1
    for j in eachindex(ks)
        local ax = Axis(f[row,col], title="$(ks[j])", aspect=DataAspect(), width=width_ax, height=height_ax)
        hm = heatmap!(ax, x_grid[j], y_grid[j], Psi2ds[j], colormap=:balance, colorrange=(-maximum(Psi2ds[j]), maximum(Psi2ds[j])))
        plot_boundary!(ax, billiard, fundamental_domain=fundamental, plot_normal=false)
        xlims!(ax, xlim)
        ylims!(ax, ylim)
        col += 1
        if col > max_cols
            row += 1
            col = 1
        end
    end
    return f
end

"""
    plot_wavefunctions(ks::Vector, Psi2ds::Vector, x_grid::Vector, y_grid::Vector, billiard::Bi; b::Float64=5.0, width_ax::Integer=500, height_ax::Integer=500, max_cols::Integer=6) where {Bi<:AbsBilliard}

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

 # Returns
- `f::Figure`: A Figure object containing the grid of wavefunctions.
"""
function plot_wavefunctions_with_husimi(ks::Vector, Psi2ds::Vector, x_grid::Vector, y_grid::Vector, Hs_list::Vector, ps_list::Vector, qs_list::Vector, billiard::Bi; b::Float64=5.0, width_ax::Integer=300, height_ax::Integer=300, max_cols::Integer=6, fundamental=true) where {Bi<:AbsBilliard}
    L = billiard.length
    if fundamental
        xlim,ylim = boundary_limits(billiard.fundamental_boundary; grd=max(1000,round(Int, maximum(ks)*L*b/(2*pi))))
    else
        xlim,ylim = boundary_limits(billiard.full_boundary; grd=max(1000,round(Int, maximum(ks)*L*b/(2*pi))))
    end
    n_rows = ceil(Int, length(ks) / max_cols)
    f = Figure(resolution=(3*width_ax * max_cols, 1*height_ax * n_rows), size=(3*width_ax * max_cols, 1*height_ax * n_rows))
    row = 1
    col = 1
    for j in eachindex(ks)
        local ax = Axis(f[row,col][1,1], title="$(ks[j])", aspect=DataAspect(), width=width_ax, height=height_ax)
        local ax_h = Axis(f[row,col][1,2], width=width_ax, height=height_ax)
        hm = heatmap!(ax, x_grid, y_grid, Psi2ds[j], colormap=:balance, colorrange=(-maximum(Psi2ds[j]), maximum(Psi2ds[j])))
        plot_boundary!(ax, billiard, fundamental_domain=fundamental, plot_normal=false)
        hm_h = heatmap!(ax_h, qs_list[j], ps_list[j], Hs_list[j]; colormap=Reverse(:gist_heat))
        xlims!(ax, xlim)
        ylims!(ax, ylim)
        col += 1
        if col > max_cols
            row += 1
            col = 1
        end
    end
    return f
end

"""
    plot_wavefunctions(ks::Vector, Psi2ds::Vector, x_grid::Vector, y_grid::Vector, billiard::Bi; b::Float64=5.0, width_ax::Integer=500, height_ax::Integer=500, max_cols::Integer=6) where {Bi<:AbsBilliard}

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

 # Returns
- `f::Figure`: A Figure object containing the grid of wavefunctions.
"""
function plot_wavefunctions_with_husimi(ks::Vector, Psi2ds::Vector, x_grid::Vector, y_grid::Vector, Hs_list::Vector, ps_list::Vector, qs_list::Vector, billiard::Bi, us_all::Vector, s_vals_all::Vector; b::Float64=5.0, width_ax::Integer=300, height_ax::Integer=300, max_cols::Integer=6, fundamental=true) where {Bi<:AbsBilliard}
    L = billiard.length
    if fundamental
        xlim,ylim = boundary_limits(billiard.fundamental_boundary; grd=max(1000,round(Int, maximum(ks)*L*b/(2*pi))))
    else
        xlim,ylim = boundary_limits(billiard.full_boundary; grd=max(1000,round(Int, maximum(ks)*L*b/(2*pi))))
    end

    L_corners = 0.0
    res = Dict{Float64, Bool}()  # Dictionary to store length and type (true for real, false for virtual)
    res[L_corners] = true # we should start at the real curve anyway
    for crv in billiard.full_boundary
        if crv isa AbsRealCurve
            L_corners += crv.length
            res[L_corners] = true  # Add length with true (real curve)
        elseif crv isa AbsVirtualCurve
            L_corners += crv.length
            res[L_corners] = false  # Add length with false (virtual curve)
        end
    end

    n_rows = ceil(Int, length(ks) / max_cols)
    f = Figure(resolution=(3*width_ax * max_cols, 2*height_ax * n_rows), size=(3*width_ax * max_cols, 2*height_ax * n_rows))
    row = 1
    col = 1
    for j in eachindex(ks)
        local ax_wave = Axis(f[row, col][1, 1], title="$(ks[j])", aspect=DataAspect(), width=width_ax, height=height_ax)
        hm_wave = heatmap!(ax_wave, x_grid, y_grid, Psi2ds[j], colormap=:balance, colorrange=(-maximum(Psi2ds[j]), maximum(Psi2ds[j])))
        plot_boundary!(ax_wave, billiard, fundamental_domain=fundamental, plot_normal=false)
        xlims!(ax_wave, xlim)
        ylims!(ax_wave, ylim)
        local ax_husimi = Axis(f[row, col][1, 2], width=width_ax, height=height_ax)
        hm_husimi = heatmap!(ax_husimi, qs_list[j], ps_list[j], Hs_list[j]; colormap=Reverse(:gist_heat))
        local ax_boundary = Axis(f[row, col][2, 1:2], xlabel="s", ylabel="u(s)", width=2*width_ax, height=height_ax/2)
        lines!(ax_boundary, s_vals_all[j], us_all[j], label="u(s)", linewidth=2)

        for (length, is_real) in res
            vlines!(ax_boundary, [length], color=(is_real ? :blue : :red), linestyle=(is_real ? :solid : :dash))
        end
        # Move to the next column
        col += 1
        if col > max_cols
            row += 1 
            col = 1
        end
    end
    return f
end

# OTOC constructions






