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
    gaussian_wavepacket_eigenbasis_expansion_coefficient(k::T, us::Vector{T}, bdPoints::BoundaryPoints{T}, x_grid::Vector{T}, y_grid::Vector{T}, x0::T, y0::T, sigma_x::T, sigma_y::T, kx0::T, ky0::T) where {T<:Real}

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

# Returns
- `T`: The projection coefficient αₘ for the given state.
"""
function gaussian_wavepacket_eigenbasis_expansion_coefficient(k::T, us::Vector{T}, bdPoints::BoundaryPoints{T}, x_grid::Vector{T}, y_grid::Vector{T}, x0::T, y0::T, sigma_x::T, sigma_y::T, kx0::T, ky0::T) where {T<:Real}
    # Compute the Gaussian wavepacket
    packet = gaussian_wavepacket_2d(x_grid, y_grid, x0, y0, sigma_x, sigma_y, kx0, ky0)
    distances = Matrix{T}(undef, length(y_grid), length(x_grid))
    dx = x_grid[2] - x_grid[1] # Precompute x grid spacing
    dy = y_grid[2] - y_grid[1] # Precompute y grid spacing
    dxdy = dx * dy # integration volume
    function proj(xy_s::SVector{2,T}) # this is the s term in the coefficient of expansion of the gaussian wave packet in the eigenbasis (given as a matrix) ∫∫dxdyY0(k|(x,y) - (x_s, y_s)|)*Φ(x,y) where Φ(x,y) is the Gaussian wave packet
        x_s, y_s = xy_s
        for j in eachindex(y_grid) # significant overhead if threading inside loop
            for i in eachindex(x_grid)
                distances[j, i] = norm(SVector(x_grid[i] - x_s, y_grid[j] - y_s))
            end
        end
        Y0_values = Bessels.bessely0.(k .* distances)
        return sum(Y0_values .* packet) * dxdy # element wise sum ∑_ij A[i,j]*B[i,j]dx*dy for the double integral approximation. The consturction of the grid is linear which implies that the dx and dy are constant
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
- `pts_mask::Vector`: Vector indicating which points in the grid are inside the billiard. This is supplied from the top calling function. It uses the `inside_only=true` case always.

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
        Threads.@threads for idx in eachindex(pts_masked)
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
    result = Threads.Atomic{T}(0.0)
    Threads.@threads for i in eachindex(us_m)
        for j in eachindex(us_n)
            xy_s_m = bdPoints_m.xy[i]
            xy_s_n = bdPoints_n.xy[j]
            contribution = us_m[i] * us_n[j] * double_integral(xy_s_m, xy_s_n) * bdPoints_m.ds[i] * bdPoints_n.ds[j]
            Threads.atomic_add!(result, contribution)
        end
    end
    return result[] / 4.0  # Multiply by 1/4 as per the formula
end

"""
    X_standard(ks::Vector{T}, vec_us::Vector{Vector{T}}, vec_bdPoints::Vector{BoundaryPoints{T}}, x_grid::Vector{T}, y_grid::Vector{T}) where {T<:Real}

Computes the full X matrix of xₘₙ=⟨m|x|n⟩. Just a multithreaded wrapper for the `X_mn` function.

# Arguments
- `ks::Vector{T}`: Vector of eigenvalues.
- `vec_us::Vector{Vector{T}}`: Vector of vectors of boundary function values for each state.
- `vec_bdPoints::Vector{BoundaryPoints{T}}`: Vector of BoundaryPoints structs for each state.
- `x_grid::Vector{T}`: Vector of x grid points.
- `y_grid::Vector{T}`: Vector of y grid points.

# Returns
- `Matrix{T}`: The full X matrix of xₘₙ=⟨m|x|n⟩. It's real!
"""
function X_standard(ks::Vector{T}, vec_us::Vector{Vector{T}}, vec_bdPoints::Vector{BoundaryPoints{T}}, x_grid::Vector{T}, y_grid::Vector{T}) where {T<:Real}
    full_matrix = fill(0.0, length(ks), length(ks))
    pts = collect(SVector(x, y) for y in y_grid for x in x_grid)
    sz = length(pts)
    pts_mask = points_in_billiard_polygon(pts, billiard, round(Int, sqrt(sz)); fundamental_domain=true)
    Threads.@threads for i in eachindex(ks)
        for j in eachindex(ks)
            full_matrix[i,j] = X_mn_standard(ks[i], ks[j], vec_us[i], vec_us[j], vec_bdPoints[j], vec_bdPoints[j], x_grid, y_grid, pts_mask)
        end
    end
    return full_matrix
end

function B_standard(ks::Vector{T}, vec_us::Vector{Vector{T}}, vec_bdPoints::Vector{BoundaryPoints{T}}, x_grid::Vector{T}, y_grid::Vector{T}, t::T) where {T<:Real}
    Es = ks .^2 # get energies
    X_matrix = X_standard(ks, vec_us, vec_bdPoints, x_grid, y_grid)
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

function B_standard(ks::Vector{T}, vec_us::Vector{Vector{T}}, vec_bdPoints::Vector{BoundaryPoints{T}}, x_grid::Vector{T}, y_grid::Vector{T}, ts::Vector{T}) where {T<:Real}
    return [B_standard(ks, vec_us, vec_bdPoints,x_grid, y_grid,t) for t in ts]
end

#TODO finish the standard OTOC for p,x
function microcanocinal_Cn_standard(ks::Vector{T}, vec_us::Vector{Vector{T}}, vec_bdPoints::Vector{BoundaryPoints{T}}, x_grid::Vector{T}, y_grid::Vector{T}, t::T) where {T<:Real}
    
end


### PLOTTING LOGIC AND EFFICIENT CONSTRUCTIONS

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

# Returns
- `Psi2ds::Vector{Matrix{T}}`: Vector of 2D wavefunction matrices constructed on the same grid.
- `x_grid::Vector{T}`: Vector of x-coordinates for the grid.
- `y_grid::Vector{T}`: Vector of y-coordinates for the grid.
"""
function wavefunction_multi(ks::Vector{T}, vec_us::Vector{Vector{T}}, vec_bdPoints::Vector{BoundaryPoints{T}}, billiard::Bi; b::Float64=5.0, inside_only::Bool=true) where {Bi<:AbsBilliard,T<:Real}
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
    pts_mask = inside_only ? points_in_billiard_polygon(pts, billiard, round(Int, sqrt(sz)); fundamental_domain=true) : fill(true, sz)
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

# Returns
- `Psi2ds::Vector{Matrix{T}}`: Vector of 2D wavefunction matrices constructed on the same grid.
- `x_grid::Vector{T}`: Vector of x-coordinates for the grid.
- `y_grid::Vector{T}`: Vector of y-coordinates for the grid.
- `Hs_list::Vector{Matrix{T}}`: Vector of 2D husimi function matrices.
- `ps_list::Vector{Vector{T}}`: Vector of ps grids for the husimi matrices.
- `qs_list::Vector{Vector{T}}`: Vector of qs grids for the husimi matrices.
"""
function wavefunction_multi_with_husimi(ks::Vector{T}, vec_us::Vector{Vector{T}}, vec_bdPoints::Vector{BoundaryPoints{T}}, billiard::Bi; b::Float64=5.0, inside_only::Bool=true) where {Bi<:AbsBilliard,T<:Real}
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
    pts_mask = inside_only ? points_in_billiard_polygon(pts, billiard, round(Int, sqrt(sz)); fundamental_domain=true) : fill(true, sz)
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
    plot_wavefunctions(ks::Vector, Psi2ds::Vector, x_grid::Vector, y_grid::Vector, billiard::Bi; b::Float64=5.0, width_ax::Integer=500, height_ax::Integer=500, max_cols::Integer=6) where {Bi<:AbsBilliard}

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

 # Returns
- `f::Figure`: A Figure object containing the grid of wavefunctions.
"""
function plot_wavefunctions(ks::Vector, Psi2ds::Vector, x_grid::Vector, y_grid::Vector, billiard::Bi; b::Float64=5.0, width_ax::Integer=500, height_ax::Integer=500, max_cols::Integer=6) where {Bi<:AbsBilliard}
    L = billiard.length
    xlim,ylim = boundary_limits(billiard.fundamental_boundary; grd=max(1000,round(Int, maximum(ks)*L*b/(2*pi))))
    f = Figure()
    row = 1
    col = 1
    for j in eachindex(ks)
        local ax = Axis(f[row,col], title="$(ks[j])", aspect=DataAspect(), width=width_ax, height=height_ax)
        hm = heatmap!(ax, x_grid, y_grid, Psi2ds[j], colormap=:balance, colorrange=(-maximum(Psi2ds[j]), maximum(Psi2ds[j])))
        plot_boundary!(ax, billiard, fundamental_domain=true, plot_normal=false)
        xlims!(ax, xlim)
        ylims!(ax, ylim)
        col += 1
        if col > max_cols
            row += 1
            col = 1
        end
    end
    resize_to_layout!(f)
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

 # Returns
- `f::Figure`: A Figure object containing the grid of wavefunctions.
"""
function plot_wavefunctions_with_husimi(ks::Vector, Psi2ds::Vector, x_grid::Vector, y_grid::Vector, Hs_list::Vector, ps_list::Vector, qs_list::Vector, billiard::Bi; b::Float64=5.0, width_ax::Integer=500, height_ax::Integer=500, max_cols::Integer=6) where {Bi<:AbsBilliard}
    L = billiard.length
    xlim,ylim = boundary_limits(billiard.fundamental_boundary; grd=max(1000,round(Int, maximum(ks)*L*b/(2*pi))))
    f = Figure()
    row = 1
    col = 1
    for j in eachindex(ks)
        local ax = Axis(f[row,col][1,1], title="$(ks[j])", aspect=DataAspect(), width=width_ax, height=height_ax)
        local ax_h = Axis(f[row,col][1,2], width=width_ax, height=height_ax)
        hm = heatmap!(ax, x_grid, y_grid, Psi2ds[j], colormap=:balance, colorrange=(-maximum(Psi2ds[j]), maximum(Psi2ds[j])))
        plot_boundary!(ax, billiard, fundamental_domain=true, plot_normal=false)
        hm_h = heatmap!(ax_h, qs_list[j], ps_list[j], Hs_list[j]; colormap=Reverse(:gist_heat))
        xlims!(ax, xlim)
        ylims!(ax, ylim)
        col += 1
        if col > max_cols
            row += 1
            col = 1
        end
    end
    resize_to_layout!(f)
    return f
end

# OTOC constructions






